# Reinforcement Learning — Deep Q-Learning and Lunar Lander (complete lab reference)

Source: user-supplied full Andrew Ng Machine Learning Specialization notebook, Deep Q-Learning - Lunar Lander. Lab uses gym 0.24.0 / LunarLander-v2; do not silently substitute current Gymnasium APIs in graded cells. The student implemented both graded exercises independently, with public unit tests passing. This is a detailed reading reference, not an independent execution of the notebook or a claim about hidden grading.

## 1. Big picture: the missing source of X and y

In supervised ML, the dataset usually supplies a feature matrix X and corresponding correct target labels y. In this RL lab, a **physics simulator generates the observations and rewards**. No human needs to manually demonstrate three successful landings before learning begins.

The simulator (environment) creates a start state using env.reset(). The agent (software policy) observes its state, chooses an action, and env.step(action) advances the physics and returns next_state, reward, done, info. One interaction becomes an experience: (state, action, reward, next_state, done). By appending transitions to a replay buffer, the agent creates its own growing dataset. However, the true Q-value is not given to it as a label: it **constructs a Bellman training target** from observed rewards plus a learned estimate of future value.

The original observation has eight features: horizontal/vertical positions, horizontal/vertical velocities, angle, angular velocity, left-leg contact, right-leg contact. Output is four estimated Q-values, one for each possible action. Lab actions are: 0 do nothing; 1 right engine; 2 main engine; 3 left engine.

## 2. RL vocabulary with simple meanings

- State S: what the agent can observe now (eight numbers, including two contact flags).
- Action A: what it chooses to do.
- Policy pi: rule for choosing actions, often probabilities.
- Reward R: immediate feedback from the simulator after acting.
- Return G: total discounted rewards along a trajectory; not the immediate reward alone.
- Value V(s): expected discounted return from a state under some policy.
- Action value Q(s,a): expected discounted return if action a is chosen first in state s, then the relevant policy is followed.
- Discount gamma: factor for future rewards; rewards further away are weighted less if gamma < 1.
- Episode: one complete landing attempt until terminal condition or stop limit.
- Terminal/done: no more future rewards in this episode.
- Experience replay: keep transition records, randomly sample mini-batches to train.
- Bellman target: immediate reward plus discounted best *estimated* future Q, unless terminal.

Course quiz: gamma=0.25, state 3 goes left via state 2 to terminal reward 100. Return at state 3 under pictured convention = 0+0.25*0+0.25^2*100=6.25. If forced to go left once from state 5 then continue optimally and V(4)=2.5, Q(5,left)=0+0.25*2.5=0.625. Thus optimal state value and the value of a specified, possibly suboptimal, action differ.

## 3. Real simulator: rewards and termination

The notebook defines reward shaping: favor closeness to pad and lower velocity, penalize tilt; +10 for each leg contacting ground; -0.03 per frame of side-engine use and -0.3 per frame of main-engine use; an additional -100 on crash and +100 on safe landing. These are **programmed environmental reward rules**, not per-state correct-action labels.

The lander starts near the upper center with random force; pad center (0,0); infinite fuel in notebook simulator. Termination described by lab includes crash and leaving x bounds. Training also has a maximum of 1000 steps per episode as a loop bound. A time-limit exit is not necessarily represented as done=True in the saved experience.

An observed one-step table in supplied notebook: current [0.007961,1.391239,0.402603,-0.450192,-0.009118,-0.090260,False,False]; action 0; next [0.011941,1.380509,0.402617,-0.476892,-0.013628,-0.090203,False,False]; reward -1.397433; done False. These are recorded output for one run, not universal values.

## 4. Why two neural networks?

When the state contains continuous numbers, storing every possible state/action in a Q-table is infeasible. DQN uses a neural network that maps an 8-feature state to four action-value predictions. It learns general patterns across nearby states.

Q-network = current learned estimates Q(s,a;w); target Q-network = slower-moving estimates Qhat(s,a;w_minus). If one model determines both prediction and target and its weights move continually, the target becomes unstable. A second, gradually updated network makes the target change more slowly.

The target initially copies the main network weights through set_weights(get_weights()). The lab describes a **soft target update**: w_minus <- tau*w + (1-tau)*w_minus, for small tau. utils.update_target_network implements this; the helper's exact tau is not included in the supplied pasted notebook.

## 5. Exercise 1 architecture and Python tuples

Notebook prints state_size = (8,) and num_actions = 4. Both independent Sequential models use:
- Input(shape=state_size)
- Dense(64, activation='relu')
- Dense(64, activation='relu')
- Dense(num_actions, activation='linear')

They have same architecture, but separate weight arrays. Output Q-values are unrestricted real numbers, not action probabilities. The optimizer is Adam(learning_rate=ALPHA), with ALPHA=1e-3.

Student's early shape bug: Input(shape=(state_size,)) wraps an existing tuple (8,) into ((8,),), producing a Keras dimension TypeError. If input is integer 8, shape=(8,) is right; if input is already tuple (8,), shape=state_size is right. The comma creates the one-element Python tuple. Student also initially wrote 'ALPHA' in quotes: that is text, not the numeric ALPHA variable. Source reports all three public tests passed.

## 6. Experience replay: the agent's dataset

The lab defines:
~~~python
from collections import deque, namedtuple
experience = namedtuple('Experience', field_names=['state','action','reward','next_state','done'])
memory_buffer = deque(maxlen=MEMORY_SIZE)
~~~
MEMORY_SIZE=100_000. deque is bounded and discards oldest records once full; namedtuple permits field-based access and groups the five transition values. Consecutive states are strongly correlated, so randomly sampling mini-batches from replay reduces temporal correlation (not perfect independence) and allows multiple uses of past experiences.

## 7. Exercise 2: calculate a Bellman target

For each mini-batch transition:
if terminal, y = reward;
otherwise, y = reward + gamma * max over actions of Q_target(next_state,action).

Combining both via a numeric 0/1 done indicator gives:
~~~python
y_targets = rewards + (1 - done_vals) * (gamma * max_qsa)
~~~
The **reward is always retained**. The (1-done) mask applies only to future value. The student initially masked the entire expression, making terminal targets zero incorrectly, then fixed this. The screenshot's lab version passed tests. If done_vals is a Boolean tensor in a different TensorFlow setup, conversion to an arithmetic-compatible dtype may be required; do not change the passed graded implementation speculatively.

The target network receives next_states batch shape (B,8) and produces (B,4). tf.reduce_max(target_q_network(next_states),axis=-1) gives maximum next-action Q per sample, shape (B,). Axis -1 means the last dimension (four actions), NOT the largest value in the entire batch.

The main network receives states (B,8) and returns four estimates for each state. But the experience is about ONE action actually selected per state. The provided code builds [batch_row, action_index] pairs via tf.range, tf.cast, tf.stack and uses tf.gather_nd to extract the chosen Q for each row, shape (B,). E.g. [[2,7,1,4],[3,0,5,9]] with actions [1,2] selects [7,5].

Student's loss:
~~~python
loss = MSE(y_targets, q_values)
~~~
MSE averages squared differences between the Bellman target and selected Q predictions. Public test_compute_loss reported All tests passed!.

## 8. Where gradients enter: agent_learn

The notebook's supplied training function:
~~~python
@tf.function
def agent_learn(experiences, gamma):
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)
    gradients = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
    utils.update_target_network(q_network, target_q_network)
~~~
GradientTape records differentiable calculations; gradient gives slopes of loss w.r.t. main Q-network trainable weights; Adam apply_gradients updates main weights; zip pairs each gradient with its variable; target updated separately by soft copy. This is the same optimization foundation as supervised neural nets, but its target y is bootstrapped. @tf.function can reduce Python overhead; the notebook reports roughly twofold speedup in its environment, not a universal guarantee.

## 9. Entire training loop in natural language

Hyperparameters from lab: GAMMA=0.995, ALPHA=0.001, MEMORY_SIZE=100000, NUM_STEPS_FOR_UPDATE=4, max episodes=2000, max steps per episode=1000, moving-average window=100, initial epsilon=1.0 (equiprobable exploration), minimum epsilon described as 0.01.

- start=time.time() saves a runtime timestamp.
- Initialize replay deque and copy target weights from main network.
- Outer loop: begin a new landing attempt by env.reset(); set total_points to zero.
- Inner loop: np.expand_dims(state,axis=0) changes (8,) into (1,8), the batch dimension Keras expects.
- Forward pass q_network(state_qn) returns (1,4).
- utils.get_action(q_values,epsilon) selects action, initially random due to epsilon=1; later gradually more greedy.
- env.step(action) advances simulator, returns next_state/reward/done/ignored info.
- Save the experience in replay memory.
- utils.check_update_conditions tests periodicity (four steps) and sufficient replay samples. When True, sample experiences, compute loss, apply gradient updates, soft-update target.
- state=next_state.copy() replaces current observation (copy avoids sharing a mutable array reference).
- total_points += reward accumulates **undiscounted episode reward**, not return or Q.
- if done, break inner loop; otherwise proceed until max timesteps.
- append total_points to history, compute np.mean(total_point_history[-100:]), update epsilon each episode.
- If last-100 average >=200, mark solved, save q_network as lunar_lander_model.h5, exit. Else continue up to 2000 episodes.
- Plot reward history, and use provided video helper to demonstrate trained policy.

## 10. Final source-observed result

Last-100 moving average: episode 100 -154.14; 200 -88.374; 300 -39.86; 400 41.742; 500 103.20; 600 132.49; 650 201.03. The notebook prints **Environment solved in 650 episodes!** and runtime **593.10 seconds (9.89 minutes)**. Save is invoked on success. These are the uploaded run's results, not new independent evaluation, and video generation isn't proof of always succeeding across random seeds.

## 11. Modern engineering and LLM connection

This notebook's gym 0.24.0 is outdated; modern Gymnasium reset and step signatures differ, so avoid copying the old unpacking into new code unmodified. Preserve original notebook cells for grading, use a separate experiment when modernizing.

RL differs from supervised learning: it creates interactions, environmental rewards and bootstrapped labels rather than learning only from supplied (X,y) pairs. DQN is not the standard modern language-model training method; RL concepts transfer to model post-training and reward-driven optimization, but algorithms, data and objectives differ.

## 12. Essential code reference

| Syntax | Meaning |
|---|---|
| state_size=(8,) | One-element tuple holding number of features |
| Input(shape=state_size) | Pass existing tuple without nesting |
| Dense(64,activation='relu') | Hidden trainable layer |
| Dense(num_actions,activation='linear') | One unbounded Q per action |
| deque(maxlen=N) | Fixed-capacity replay |
| namedtuple(...) | Field-named experience |
| env.reset() / env.step(action) | Simulator initialization and transition |
| np.expand_dims(state,axis=0) | Add batch dimension |
| tf.reduce_max(...,axis=-1) | Max action Q for each sample |
| tf.stack([rowids,actions],axis=1) | Coordinates of chosen actions |
| tf.gather_nd(q,index_pairs) | Chosen-action Q values |
| MSE(target,prediction) | Scalar mean squared loss |
| tf.GradientTape / tape.gradient | Automatic differentiation |
| optimizer.apply_gradients(zip(...)) | Update main model parameters |
| target.set_weights(main.get_weights()) | Initial parameter snapshot |
| history[-100:] / np.mean(...) | Recent-episode moving average |

For continuity see the GOLDEN_RULE note, Python ML cheat sheet, and Complete Course 3 Booklet.
