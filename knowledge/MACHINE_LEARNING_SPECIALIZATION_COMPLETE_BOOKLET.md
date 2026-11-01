# Complete Machine Learning Specialization — Courses 1, 2 and 3
## A personal study booklet: theory, mathematical intuition, Python, NumPy, TensorFlow, lab evidence and real-world engineering

**November 2026 — Master reference, version 1.0**

**Purpose.** This is the one continuous, human-language account of the knowledge studied across Andrew Ng / DeepLearning.AI's Machine Learning Specialization. It connects the subjects in the order they build on one another and uses the actual exercises, errors, and source-based findings recorded in the AI-LLM-Training-Roadmap repository. It is meant to be read at any pace. The canonical syntax reference remains PYTHON_ML_CODING_CHEAT_SHEET.md, while individual detailed notes preserve the original lab evidence.

**Evidence boundary.** The repository includes detailed Course 1 regression, logistic gradient and numerical coding notes; Course 2 neural-network, computation-graph, model-selection, tree and ensemble notes; and Course 3 complete lab evidence. It does not include every graded cell or lecture from each course. This booklet explains the full covered curriculum and clearly separates recorded lab results from additional explanatory examples. The text is not a claim that every hidden examination or course certificate was independently verified.

# 0 — First understand what every machine-learning algorithm does

Imagine three jobs. First, predict a house price from its size and location. You have example houses and correct prices. This is **supervised learning**. Second, group unlabeled telecom traffic intervals by similarity or detect unusual behavior. You have features but no known answers for every row. This is **unsupervised learning**. Third, pilot a simulated lunar spacecraft. You do not know the correct action for every possible situation; instead you take actions and receive rewards. This is **reinforcement learning**.

Before studying any algorithm, answer six questions:

1. Where did its input values originate? A database, simulator, engineered measurement or human rating?
2. How are the numbers organized: how many examples, how many features, what shape and unit?
3. Does the dataset contain a correct target label?
4. What internal parameters does the model learn?
5. Which numerical objective determines what is good?
6. What test would demonstrate genuinely useful performance on unseen examples?

A **feature** is an input. A **label** is a known answer. A **prediction** is the model's output. A **loss** measures one or more errors. A **parameter** is a numerical quantity trained from data; a **hyperparameter** is configured by the engineer, such as learning rate, number of clusters or network width. Training changes fitted parameters. Inference uses fitted parameters. An application's database and its model weights are separate things.

### Arrays are the language of all three courses

Usually X has shape (m,n): m examples in rows and n features in columns. An individual row X[i] has shape (n,), and X[:,j] gets the j-th feature across all examples. An element X[i,j] is one number. These conventions are not universal: collaborative filtering uses movie rows and user columns in its ratings matrix.

Python's shape and indexing:
~~~python
import numpy as np
X = np.array([[2., 3., 4.], [5., 6., 7.]])
print(X.shape)         # (2, 3)
print(X[0])            # first row
print(X[:,1])          # [3., 6.]
print(np.sum(X,axis=0)) # sum of each feature across rows
print(np.sum(X,axis=1)) # sum of each row's features
~~~

The most valuable debugging habit across this entire specialization is to inspect **shape, dtype, input source, mathematical meaning** before changing the algorithm. A scalar, a vector (8,), and a batch (1,8) are not the same object.

---

# COURSE 1 — Supervised Machine Learning: Regression and Classification

## 1.1 Prediction from known examples

Suppose historical house data gives floor area x and sale price y. We want a mathematical function that converts x into an estimate of y. We learn the function's parameters from old houses and then predict on a future house. This is not memorization: a useful model must generalize to unseen cases.

### Linear regression

For one feature:

\[
f_{w,b}(x)=wx+b
\]

x = input, w = learned slope, b = learned baseline/intercept, and f = predicted result.

For multiple features:

\[
f_{\mathbf w,b}(\mathbf x)=\mathbf w\cdot\mathbf x+b
\]

Each input feature has a corresponding weight. Dot product means multiply matching entries and add. Example: features [2,4,1], weights [5,3,-2], bias 7. Prediction = 2*5+4*3+1*(-2)+7 = 27.

~~~python
import numpy as np
x = np.array([2.,4.,1.])
w = np.array([5.,3.,-2.])
b = 7.
prediction = np.dot(x,w) + b
~~~

The code is not magical; it implements exactly the mathematical weighted sum.

## 1.2 Error and cost are not the same

One prediction error is prediction minus truth. If the model predicts 315000 and actual price is 310000, error=+5000. We can square the error to prevent positive/negative mistakes cancelling and penalize large misses.

Andrew Ng's familiar regression cost:

\[
J(w,b)=\frac{1}{2m}\sum_{i=0}^{m-1}(f_{w,b}(x^{(i)})-y^{(i)})^2
\]

The sum runs across all training examples. The division by 2m happens once **after** accumulating the errors. The 1/2 makes the derivative cleaner; many libraries use unhalved MSE. Do not mix one-example loss and entire training-set cost.

### Independent coding example

~~~python
truth = np.array([3., 5., 7.])
guess = np.array([2., 6., 8.])
errors = guess - truth
squared = errors**2
print(np.sum(squared)/(2*len(truth))) # 0.5
~~~

## 1.3 How learning changes parameters: gradient descent

The cost tells us how wrong we currently are. The derivative tells us how this wrongness changes if we nudge one parameter.

A derivative is a **slope**, not an altitude or a parameter itself. Negative derivative means a small increase in the parameter locally decreases the cost. Gradient descent walks opposite the derivative.

\[
w_j\leftarrow w_j-\alpha\frac{\partial J}{\partial w_j},\quad
b\leftarrow b-\alpha\frac{\partial J}{\partial b}
\]

Alpha (learning rate) controls how large a step to take. Too small -> very slow. Too large -> overshooting/divergence. Proper learning rate -> cost decreases until sufficiently converged.

In a loop:
1. Compute predictions with the current parameter values.
2. Compute errors, accumulate gradient contributions.
3. Average gradients over examples if required by the equation.
4. Update weights and bias together.
5. Repeat.

It is important that gradients for the same iteration derive from the same old parameter state. Updating one parameter halfway through computing the others may change the intended algorithm.

The dedicated GRADIENT_DESCENT.md preserves the student's learning struggle around positive/negative slopes, partial derivatives, iterations, convergence and scientific notation.

## 1.4 Vectorization is the same mathematics faster

Instead of looping over examples manually, NumPy evaluates many predictions at once. For X shape (m,n), w shape (n,), and scalar b:

~~~python
predictions = X @ w + b
~~~

The @ operator performs the correct matrix/vector multiplication. Elementwise X*w is not the same as X@w, because it yields a value for each matrix element rather than a single weighted sum per example. Pay attention to dimensions and broadcasting.

## 1.5 Feature scaling: a fair numerical ruler

If an input feature is 50000–200000 and another is 0–5, optimization can become inefficient or sensitive to scale. Z-score standardization:

\[
x_{\rm scaled}=(x-\mu)/\sigma
\]

The mean mu and standard deviation sigma must be learned on the **training set**, then reused on validation/test/new inputs. Fit a scaler on test data and you overwrite the ruler and leak information.

~~~python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
~~~

Different transformations are NOT interchangeable:
- mean centering: x - mean;
- standardization: (x-mean)/std;
- mean normalization using a range: (x-mean)/(max-min);
- min-max scaling to a chosen interval;
- unit-vector L2 normalization: vector divided by Euclidean length.

A model deployed later must keep the original fitted scaler and exact feature ordering.

## 1.6 Polynomial features and nonlinear patterns

A linear model in trainable parameters can represent curves if we supply polynomial features: x, x², x³, or interactions such as x1*x2. More expressive features may reduce underfitting, but excessively high polynomial degree risks overfitting.

~~~python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2,include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_new_poly = poly.transform(X_new)
~~~

Again: the fitted preparation rules must stay consistent between training and prediction.

## 1.7 Logistic regression: classification instead of predicting price

For a yes/no problem, compute linear score z=w·x+b and convert it into a value from 0 to 1 using sigmoid:

\[
g(z)=1/(1+e^{-z})
\]

With a model calibrated and interpreted in binary classification, this represents the predicted positive-class probability. A threshold such as 0.5 converts it into a predicted class. The threshold is a decision rule and may be adjusted for false-positive/false-negative consequences.

Logistic cross-entropy per example:

\[
L=-y\log(\hat y)-(1-y)\log(1-\hat y)
\]

This is different from linear regression's squared loss. Stable TensorFlow APIs can compute binary cross-entropy directly from logits; when configured from_logits=True, avoid applying sigmoid a second time before the loss.

### The logistic gradient, as learned in the repository

For each training row:
1. Compute z from all features.
2. Apply sigmoid to obtain one predicted probability.
3. Error = predicted probability - actual label.
4. Add error*x_j to every corresponding weight gradient.
5. Add error to the bias gradient.
6. Average after all rows if the exercise requires.
7. Update parameters by learning rate.

The student's key difficulty was separating **one error per example**, **one gradient contribution per feature**, and the **whole-dataset accumulation**. The loop variable meanings are determined by array shapes, not memorized naming conventions.

## 1.8 Regularization, bias and variance preview

Overfitting occurs when a model learns noisy training detail and performs much worse on unseen observations. L2 regularization discourages overly large weights:

\[
J_{\rm total}=J_{\rm data}+\frac{\lambda}{2m}\sum_jw_j^2
\]

Lambda controls penalty strength. The exact normalization factor may differ across labs; use the original exercise, not a formula remembered from another model. Too strong a penalty can cause underfitting. Many introductory formulations exclude the bias term b from regularization.

~~~python
penalty = lambda_ * np.sum(w**2)/(2*m)
~~~

**Course 1 takeaway:** X describes the input; y describes the known answer; parameters control predictions; loss measures wrongness; gradients describe how to change parameters; regularization and validation help guard against memorization.

---

# COURSE 2 — Advanced Learning Algorithms

## 2.1 From a weighted sum to a neural network

A neuron still computes:

\[
z=w\cdot x+b,\qquad a=g(z)
\]

where g is an activation. One Dense layer contains multiple neurons, each with its own weights/bias. A neural network chains layers so the output of one becomes the input of the next. This adds expressiveness beyond one linear formula.

Activations:
- Linear: no nonlinear restriction; useful for regression and latent vector coordinates.
- Sigmoid: maps output into (0,1), often binary probability.
- ReLU: max(0,z), common hidden-layer activation.
- Softmax: maps K output scores to K positive probabilities summing to one, useful for multiclass classification.

There is no universal best output activation: it depends on the task, target and loss.

## 2.2 The 400 → 25 → 15 → 1 digit network

The Course 2 Week 1 source begins with a 20×20 pixel image. Flattened, each image has **400 input features** and one label identifying the image, not 400 different labels.

Network:
- Input width 400.
- Dense hidden layer of 25.
- Dense hidden layer of 15.
- Final one-neuron output for the binary case.

If m images are processed:
- X (m,400)
- first hidden activations (m,25)
- second hidden activations (m,15)
- output (m,1)

The output width is how many neurons are in that layer; the batch dimension stays separate.

### Read the Keras syntax

~~~python
import tensorflow as tf
example = tf.keras.Sequential([
    tf.keras.Input(shape=(4,)),
    tf.keras.layers.Dense(6,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid'),
])
print(example(tf.ones((3,4))).shape)  # (3,1)
~~~

This is a tiny independent example, not the original graded digit network. Dense(6) tells Keras how many neurons to create, not how many inputs exist.

## 2.3 Forward propagation versus backpropagation

Forward propagation: take current weights and input, calculate activations layer by layer until prediction, then compute loss. This is the same arithmetic taught in Course 1, repeated many times.

Backpropagation: use the chain rule to calculate how sensitive the final loss is to each parameter upstream. It does **not itself update weights**; it calculates derivatives. The optimizer uses those derivatives to update.

A computation graph breaks a formula into boxes. Example J=(w+1)²:
- a=w+1
- J=a²
- dJ/da=2a
- da/dw=1
- dJ/dw=(dJ/da)*(da/dw)=2(w+1)
At w=2, gradient=6, while the cost itself is 9. Gradient and cost are different kinds of numbers.

The dedicated computation-graph guide traces forward/backward passes and the difference between sensitivity and actual parameter updates.

## 2.4 TensorFlow/Keras: packaging the same steps

Core learning APIs:
- Sequential: stack one-input layers.
- Functional Model: connect multiple inputs, branches and outputs.
- compile: configure optimizer and loss.
- fit: repeatedly predict, calculate loss, compute gradients and update parameters.
- evaluate: compute held-out loss/metrics without weight updates.
- predict: forward predictions without training.

Custom training:
~~~python
with tf.GradientTape() as tape:
    loss = some_loss
grads = tape.gradient(loss,model.trainable_variables)
optimizer.apply_gradients(zip(grads,model.trainable_variables))
~~~

The recorded DQN and collaborative filtering labs return to this exact pattern. Adam updates parameters using gradients, with extra internal optimizer state. You do not need to memorize Adam's every internal equation to understand what it is doing at an engineering level.

## 2.5 Outputs and losses must match

For binary classification: sigmoid output plus binary cross-entropy, or a raw logit with loss configured from_logits=True.

For multiclass classification: K output scores, softmax or logits depending on loss, with correct integer-label versus one-hot-label interpretation. SparseCategoricalCrossentropy is for integer class labels; CategoricalCrossentropy usually assumes suitable one-hot targets.

For numerical regression: linear output and MSE are common. For DQN, linear outputs are Q-values (not probabilities), even though there are four action outputs.

A wrong loss/output pairing can produce misleading learning behavior without obvious syntax errors.

## 2.6 Train, validation, test: three different jobs

- Training: learn weights and preprocessing parameters.
- Cross-validation/validation: choose model and hyperparameters.
- Final test: estimate performance once after selection.

If you repeatedly choose designs by watching the test score, the test data has become selection feedback. That invalidates the claim that it was independent evidence.

The Week 3 optional lab explores polynomial model degree, preprocessing, bias and variance. A good workflow fits polynomial feature mapping and scaler on training data, then evaluates alternatives on validation. Only the selected configuration is finally evaluated on test.

### What does fit learn?

It depends on which object calls fit:
- scaler.fit learns means/scales;
- PolynomialFeatures.fit learns input schema/mapping;
- neural model.fit updates network weights;
- PCA.fit learns principal axes;
- KMeans.fit learns centroids.
The same method name can stand for different kinds of learned state.

## 2.7 Diagnosing high bias and high variance

High bias (underfitting): the model already struggles with training examples. It may be too simple, have insufficient features, inappropriate training or excess regularization.

High variance (overfitting): training error is small while validation/test error is significantly larger. Complexity, noise, too little representative data or insufficient regularization may contribute.

Potential fixes should follow diagnosis rather than a reflexive rule. More training examples might reduce variance; a richer model can reduce bias; regularization and early stopping may reduce overfitting. Bad labels or dataset shift can defeat architecture changes.

A learning curve shows how performance changes with dataset size. Classification error rate is not the same numerical quantity as cross-entropy loss.

## 2.8 Decision trees and entropy

Decision trees learn sequences of feature questions rather than neuron weights. Each internal node asks a question; each branch selects a subset; leaves provide predictions.

For binary labels with positive fraction p:

\[
H(p)=-p\log_2p-(1-p)\log_2(1-p)
\]

Entropy 0 means pure labels; entropy 1 at p=0.5 means maximally mixed binary labels.

Information gain measures how much uncertainty the split removes:

\[
IG=H(parent)-\frac{n_L}{n}H(left)-\frac{n_R}{n}H(right)
\]

The weights n_L/n and n_R/n represent **fractions of examples in each branch**, not the positive-class probability used inside H.

### The key graded coding confusion

The source decision-tree lab keeps row IDs referring to the original X and y arrays even after visiting a child node. To split that child, work with its selected original row IDs rather than accidentally indexing an independently filtered subarray using the wrong coordinate system.

Another important pattern: to find the best candidate split, compare each information gain with the **best gain found so far**, not merely the preceding candidate. Do not mix the split operation (which rows go where) with the information-gain calculation (was the split useful?).

Trees can overfit with excessive depth. Limits include max_depth, min_samples_split and min_samples_leaf.

## 2.9 Random forest and boosting

Random forests combine predictions from many decision trees trained with sources of randomness, often reducing instability and variance.

Gradient-boosted trees train sequentially, so each new tree addresses weaknesses of the current ensemble according to a chosen loss. XGBoost is one widely used implementation.

These differ from neural-network backpropagation. Tree depth, number of estimators, feature sampling and learning rate (in boosting) have different roles from Dense layers and Adam.

The repository preserves the second trees/ensemble lab with comparisons among decision tree, random forest and XGBoost and the caution that scores depend on specific split/seed/test procedure. For many tabular problems, ensembles are practical baselines before deep learning.

**Course 2 takeaway:** learn richer models, inspect their shapes and training mechanism, and evaluate generalization rather than only training error.

---

# COURSE 3 — Unsupervised Learning, Recommenders and Reinforcement Learning

## 3.1 K-means: group without known labels

Imagine K movable markers among a cloud of dots:
1. Initialize K centroids.
2. Assign each point to nearest centroid.
3. Move each centroid to the mean of assigned points.
4. Repeat.

Squared Euclidean distance:

\[
d^2(x,c)=\sum_j(x_j-c_j)^2
\]

You need the **index** of the closest centroid to know the assignment: np.argmin(distances) rather than np.min(distances). Updating centroid k requires selecting the examples assigned to k and averaging their feature vectors.

Your recorded graded K-means lab includes find_closest_centroids, compute_centroids and RGB image compression. In the image example, each RGB pixel is a 3-number feature vector, and K-means learns a small representative color palette.

K-means can struggle with poor initialization, non-round clusters, outliers and unscaled features. Cluster number K is configured; cluster IDs do not imply human-readable categories.

## 3.2 Anomaly detection: model normal, flag unusual

Instead of dividing every point into groups, anomaly detection estimates a distribution of typical examples and flags sufficiently unlikely observations.

The Gaussian model for each feature estimates training mean mu and variance sigma². The notebook combines feature densities under simplifying assumptions and compares p(x) against threshold epsilon.

The selection of epsilon is a trade-off:
- too strict: few alerts, misses possible;
- too permissive: many false positives.

The course uses labeled validation data to select epsilon via F1 even though the normal-density estimation itself is unsupervised.

Precision=TP/(TP+FP); recall=TP/(TP+FN); F1=2PR/(P+R), when defined. For rare incidents, high accuracy can coexist with almost zero anomaly detection usefulness.

Python techniques include np.mean/np.var(axis=0), Boolean masks for TP/FP/FN, threshold sweeps and zero-denominator guards. For production alerting, threshold should also reflect the consequences of false positives and missed severe cases.

## 3.3 Collaborative filtering: learn from user/movie ratings

Here Y[i,j] stores movie i and user j rating; R[i,j] is 1 only when that rating actually exists.

Learn a feature vector X[i] for each movie, a parameter vector W[j] for each user and a user bias b[0,j].

\[
\hat y(i,j)=W[j]\cdot X[i]+b[0,j]
\]

The cost accumulates squared errors only if R[i,j]==1. The 1/2 factor is applied once after summing; L2 regularization in this lab adds lambda/2 times sums of squares of X and W (no b term).

The student independently wrote the loops, R guard, dot product, prediction error, accumulation and regularization, and public tests passed. Notebook reference outputs: unregularized cost 13.67 and regularized example 28.09.

A new unseen user doesn't automatically have a learned W row. It requires a new latent-factor strategy, ratings, model extension or content-based alternative. A database record is not a learned embedding.

Mean normalization for this lab means subtract average movie rating calculated over actually observed ratings, and add it back to reconstructed predictions. It is NOT the different formula dividing by (max-min).

## 3.4 Content-based filtering: user and movie feature vectors versus learned embeddings

Raw user input = 14 per-genre average ratings. Raw movie input = 16 numbers: release year, average rating, and 14 genre indicators. Some stored metadata columns like user/movie IDs are excluded.

Two independent towers each produce 32 learned numbers:
- 256 Dense ReLU -> 128 Dense ReLU -> 32 Dense linear.

The output vectors are **embeddings**. They are not the same as original known features and are not 32 named genres. Their coordinates are learned numerical combinations useful for predicting ratings.

The Functional API L2-normalizes user/movie output vectors and computes their dot product. For unit vectors, dot product is cosine similarity. The lab has three different transformations: input StandardScaler, target MinMaxScaler((-1,1)), and output L2 normalization.

The network learns using labeled user/movie ratings, MSE, Adam and model.fit. The student correctly wrote both towers; both public tests passed.

### New users and new movies after training

The trained content-based network can process a new user with the same 14-feature preference schema and a new movie with the same 16-feature schema, using the original fitted scalers, **without immediately retraining**. If a user has no preference data or a new movie has no historical average rating, the feature schema has missing information: this is the cold-start problem.

The model does not automatically absorb a newly registered database record or change its weights when a user joins. Inference and training are separate.

## 3.5 Similar movie vectors: the second graded exercise

After training, movie embeddings have 32 dimensions. Compare two with squared Euclidean distance:

~~~python
d = np.sum((a-b)**2)
~~~

Small distance means closer in the learned representation. The student wrote this and public tests passed; the supplied examples returned 0.000, 0.030 and 2.000.

The lab calculates pairwise movie distances, excludes each movie's trivial self-match and selects the closest neighbor. On L2-normalized vectors, squared distance is 2-2*(their dot product). This is **movie-to-movie** similarity, not the **user-to-movie** rating score.

Recorded data: 397 users, 847 movies, 25,521 original ratings, expanded to 50,884 training rows. Training loss reported 0.0730; test 0.0812. The original content-based notebook fits scalers before splitting and repeats some examples; interpret held-out metrics carefully. The existing-user demo errored for uid=2, and similar-item table copied the first movie's genre metadata into the second-movie column. The detailed note preserves these issues.

## 3.6 PCA: find new directions that capture variation

PCA rotates the coordinate system to find orthogonal directions along which centered observations vary the most. It does not simply select original feature columns. PC1 explains maximum variance; PC2 maximum remaining orthogonal variance; more components follow.

For X (m,n), centered by mean mu, with k principal directions in V_k (n,k):

\[
Z=(X-\mu)V_k,\qquad \hat X=ZV_k^T+\mu
\]

Z (m,k) is the lower-dimensional representation. Inverse reconstruction loses discarded component information.

~~~python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
Z = pca.fit_transform(X_train)
new_Z = pca.transform(X_new)
X_approx = pca.inverse_transform(Z)
~~~

Fit learns mean and axes; transform applies learned axes; inverse_transform approximately reconstructs. PCA centers features but sklearn PCA **does not automatically standardize features to unit variance**. For features with heterogeneous scales, consider a train-only StandardScaler first.

Lab six-point dataset: first PC 99.244289% explained variance and second 0.755711%. Larger 500-row/1000-feature dataset: 2 PCs explain 14.57%, 3 PCs explain **20.81%**. The notebook prose incorrectly called the three-component result 19%, but the numeric output is 0.208062578.

PCA can help inspect high-dimensional embeddings, but explained variance is not guaranteed semantic accuracy or retained nearest-neighbor quality. A 2D plot does not prove a particular cluster count. PCA is not required for every LLM or RAG workflow.

## 3.7 Reinforcement learning: where does the training data originate?

This was the critical missing connection in Lunar Lander: nobody must manually perform two or three example landings to give the model correct labels.

The **Gym physics simulator** initializes a spacecraft and returns eight state values through env.reset(): horizontal and vertical position, x/y velocity, angle, angular velocity, left and right landing-leg contact. The agent chooses an action, and env.step(action) advances physics and returns next_state, reward and done.

Four actions in this lab: 0 do nothing; 1 fire right engine; 2 main engine; 3 left engine. The environment provides reward rules: approach the pad, reduce speed and tilt, gain leg-contact rewards, pay engine-use costs, receive ±100 at landing/crash. Rewards are part of the *environment's programming*, not hand-labeled optimal actions.

A transition becomes one experience:

~~~text
(state, action, reward, next_state, done)
~~~

The agent saves many experiences to replay memory, building its own learning dataset through interaction.

## 3.8 Reward versus return versus Q

- Reward: immediate feedback after an action.
- Return: discounted sum of later rewards along the remaining path.
- V(s): expected return from a state under a policy.
- Q(s,a): expected return if action a is taken first, then policy continues.
- Gamma: discount factor for later rewards.

Earlier quiz: state 3 moves left through state 2 and reaches terminal reward 100 after two transitions; gamma=0.25. Return is 0+0.25*0+0.25²*100=6.25. For Q(5,left), moving left once to state 4 with optimal value 2.5 gives Q=0+0.25*2.5=0.625. A state's optimal return is not necessarily the return for a deliberately suboptimal action.

Continuous-state MDP means state quantities like position and velocity vary continuously, not that it has many fields or that rewards are decimals. Actions in this lab are still four discrete choices.

## 3.9 DQN: why two networks?

Continuous state makes a table of every possible state/action Q infeasible, so a neural network approximates action values. Both Q models use:
- Input(state_size), where state_size is tuple (8,).
- Dense(64,relu).
- Dense(64,relu).
- Dense(num_actions,linear): four Q values, not probabilities.

The **main network** is optimized via gradients. The **target network** supplies relatively stable next-state value estimates for Bellman targets. It begins with a weight copy of the main network and later receives slow/soft updates. This prevents excessively rapidly moving target values from destabilizing learning.

### Original Python errors

state_size=(8,) is already a tuple. Input(shape=(state_size,)) creates a nested tuple ((8,),) and raised the lab's dimension error. Input(shape=state_size) passes the existing valid tuple.

Adam(learning_rate='ALPHA') passes a string; Adam(learning_rate=ALPHA) reads the numerical variable. The student corrected both errors; Exercise 1's three public tests passed.

## 3.10 Bellman targets and the student's graded loss

For a terminal experience, target y=reward. Otherwise:

\[
y=R+\gamma\max_{a'}\hat Q(S',a')
\]

Batch expression using numeric done values:

~~~python
y_targets = rewards + (1-done_vals)*(gamma*max_qsa)
loss = MSE(y_targets,q_values)
~~~

Both lines were student-written and public tests passed.

A major correction: mask ONLY the future value. If you multiply (1-done) by the entire expression, the immediate terminal reward is erroneously set to zero.

TensorFlow batch shapes:
- next_states (B,8);
- target Q output (B,4);
- tf.reduce_max(...,axis=-1) -> (B,), max next action per row;
- main Q output (B,4);
- tf.range/tf.stack/tf.gather_nd select the value for the action actually taken in each record -> (B,);
- MSE compares length-B targets and predictions.

These TensorFlow operations are not mysterious when you know the desired indexing: one row per experience, four choices per row, one chosen action per experience.

## 3.11 The real training loop

MEMORY_SIZE=100000, GAMMA=0.995, ALPHA=0.001 and NUM_STEPS_FOR_UPDATE=4. The replay buffer is deque(maxlen=MEMORY_SIZE), filled with namedtuple experience records. Sampling prior experiences reduces correlations between adjacent simulated states and reuses collected data.

For each of up to 2000 episodes and up to 1000 timesteps:
1. Reset environment and receive one 8-value state.
2. np.expand_dims(state,axis=0) reshapes (8,) to (1,8) for a Keras batch of one.
3. Q-network predicts four action values.
4. Epsilon-greedy chooses action; initially epsilon=1 means random exploration, later reduced toward 0.01.
5. env.step(action) simulates the action, returning new state, reward and done.
6. Save experience in replay buffer.
7. On the configured update interval and when enough experiences exist, sample mini-batch and run agent_learn.
8. agent_learn uses tf.GradientTape to calculate loss gradients, Adam apply_gradients to update main Q-network, and soft-update target weights.
9. Continue until done or the maximum timestep count.
10. Record **undiscounted** total episode reward. Take mean of most recent 100 episode scores. Stop and save model if average >=200.

The program is computationally costly because it performs many physics steps, forward passes and periodic gradient updates. Notebook's actual source run stopped after 650 episodes with average score **201.03**, runtime **593.10 seconds (9.89 minutes)**, and output "Environment solved in 650 episodes!". Public tests also passed for both network definitions, optimizer and compute_loss.

The notebook's training result meets its stated success criterion. It is not proof that the lander always succeeds across random seeds or a claim that hidden exams were verified.

## 3.12 RL and later LLM engineering

Policies, rewards, returns, optimization and evaluation are transferable ideas to some LLM post-training work. But a four-action Q-network trained with simulator replay is **not the default algorithm** for training modern language models. The strongest transferable engineering skills are array semantics, gradient loops, target construction, reproducibility, honest validation and understanding how feedback is created.

**Course 3 takeaway:** some algorithms discover structure from unlabeled data, recommenders learn from preferences or content, and RL learns actions from simulated interaction and rewards. Always identify where the signal comes from.

---

# 4 — Cross-course connections and engineering discipline

## 4.1 One model comparison table

| Model | Data/input | Learning signal | Learned object | Output |
|---|---|---|---|---|
| Linear regression | X/y numeric | Squared error | w,b | Predicted number |
| Logistic regression | X/y binary | Cross-entropy | w,b | Probability/class |
| Neural network | Features/labels | Loss gradients | Layer weights/biases | Task dependent |
| Decision tree | Labeled features | Information gain/impurity | Splits/leaves | Class/number |
| Random forest | Labeled features | Aggregated tree fitting | Many trees | Averaged/voted prediction |
| Boosted trees | Labeled features | Sequential loss improvement | Ensemble trees | Prediction |
| K-means | Unlabeled features | Distances/centroids | K centroids | Cluster index |
| Anomaly detection | Typical features + CV | Density, epsilon | Means/variances/threshold | Anomaly flag |
| Collaborative filtering | Ratings Y, mask R | Rating prediction errors | User/movie latent factors | Predicted ratings |
| Content-based recommender | User/movie features, ratings | MSE | Two neural towers | Embeddings/score |
| PCA | Feature matrix | Explained variance | Principal axes | Compressed coordinates |
| DQN | Environment transitions | Rewards + bootstrapped Q | Q-network weights | Action values |

## 4.2 Skills that connect them all

**Array shapes.** Every algorithm has input and output dimensions. Incorrect shapes can yield crashes or silently wrong broadcasting.

**Separation of roles.** Inputs, true labels, predictions, loss, gradients and parameters are distinct quantities. The algorithm fails if we swap them in our mental model.

**Fitting versus using.** A scaler, PCA, K-means model, regression model or neural network learns reusable state. A new observation should reuse the fitting state, not replace it.

**Validation.** A passing public unit test establishes particular code behavior. Low train loss is not proof of generalization. Visual clusters are not ground-truth labels. A successful simulated episode is not universal robustness.

**Debugging.** Before making architectural changes, check variable names, dtypes, units, indices, masks, axes, feature schema and whether a formula's coefficient is inside or outside the intended sum.

**Not every ML model is optimized using neural-network gradients.** K-means alternates assignment/centroid means, PCA uses matrix decomposition, and trees choose splits. Regression, neural nets, two-tower recommenders and DQN use gradient-based optimization in the versions studied.

## 4.3 Where a trained model fits in a real application

An application keeps a database, input feature definitions and model/scaler artifacts. A prediction endpoint gathers current feature values, applies the **saved** feature preparation, runs the model and returns an output.

Adding a user to a database does not automatically train the model. A new user may be represented immediately if a content-based input vector is available; a pure per-user latent model needs a way to create factors. New data may eventually justify retraining or fine-tuning.

Production engineering requires model/data versioning, train/serving consistency, validation splits, monitoring of errors and drift, cost/latency observation, and safe fallbacks. For telecom operations, incorrect source completeness or ambiguous call-leg definitions can invalidate even mathematically correct model calculations.

## 4.4 What a future ML/LLM engineer should memorize

Memorize the **meaning** of features, shapes, target, loss, gradient, training, inference, validation and the trade-offs between algorithms. Be able to implement NumPy loops, dot products, masks, basic sklearn pipelines and neural model training with ordinary documentation assistance.

It is reasonable to look up exact API arguments for tf.gather_nd, reshaping, advanced optimizers, PCA components, Gymnasium migrations and plotting.

The student's golden rule applies: write graded code independently, get targeted feedback, pass tests, then perform a deep explanation and retain the reusable concept in the canonical cheat sheet.

## 4.5 ML/LLM engineering connection without overstating it

These courses build the foundations for embeddings, evaluation, training loops, ranking, retrieval, anomaly monitoring and some RL concepts. Transformers, attention, tokenization, RAG, LLM fine-tuning, tool use, model serving and LLMOps remain additional subjects; this booklet does not claim they were fully taught by the specialization.

---

# 5 — Course-by-course evidence and deeper study references

### Course 1 references
- [Gradient Descent — Detailed Human Explanation](GRADIENT_DESCENT.md)
- [Python for ML — Quick Reference](PYTHON_FOR_ML_QUICK_REFERENCE.md)
- [NumPy for ML](NUMPY_FOR_ML.md)

### Course 2 references
- [Course 2 Week 1 Neural Networks](COURSE_2_WEEK_1_NEURAL_NETWORKS.md)
- [Course 2 Backpropagation/Computation Graph](COURSE_2_BACKPROPAGATION_COMPUTATION_GRAPH.md)
- [Fit, Transform and Predict](FIT_TRANSFORM_PREDICT_DEEP_DIVE.md)
- [Sklearn Preprocessing Reference](SKLEARN_PREPROCESSING_CHEAT_SHEET.md)
- [Decision Trees and Ensemble Learning](DECISION_TREES.md)
- [TensorFlow/Keras Detailed Reference](TENSORFLOW_KERAS_FOR_ML.md)

### Course 3 references
- [Unsupervised Learning / K-means](UNSUPERVISED_LEARNING_AND_KMEANS.md)
- [Anomaly Detection](ANOMALY_DETECTION.md)
- [Collaborative Filtering](COLLABORATIVE_FILTERING.md)
- [Content-Based Filtering](CONTENT_BASED_FILTERING.md)
- [PCA](PCA_PRINCIPAL_COMPONENT_ANALYSIS.md)
- [Reinforcement Learning and Lunar Lander DQN](REINFORCEMENT_LEARNING_DQN_LUNAR_LANDER.md)
- [Course 3 Complete Booklet](COURSE_3_COMPLETE_BOOKLET.md)

### Canonical project guidance
- [Python ML Coding Mega Cheat Sheet](PYTHON_ML_CODING_CHEAT_SHEET.md)
- [Guided Coding Golden Rule](GUIDED_CODING_GOLDEN_RULE.md)

### Recorded, not independently rerun, milestones
- Linear/logistic regression and gradient-descent foundations recorded in Course 1 notes, but not all original graded cells preserved in this repo.
- Course 2 neural network and backprop optional guides; Week 3 model-selection reference; decision-tree and ensemble lab records.
- K-means and anomaly-detection course lab results preserved in focused notes.
- Collaborative filtering cost exercise user-authored, public tests passed; notebook example costs 13.67 unregularized and 28.09 regularized.
- Content-based two towers and squared distance user-authored, public tests passed.
- PCA numeric observations include 99.24% first PC in first example and 20.81% three-component variance in second example.
- Lunar Lander DQN networks and loss public tests passed; training report 650 episodes and 201.03 moving average, about 9.89 minutes.

**Final mental map:** The source of the numbers determines the learning problem. The model represents a rule. Training adjusts or estimates parameters. Evaluation tells us whether the rule is useful beyond its training examples. The programmer's responsibility is to keep those facts distinct and verify their consequences.
