# Machine Learning Specialization — Course 3 Complete Learning Booklet
## Unsupervised Learning, Recommenders, and Reinforcement Learning
### Theory, Python, NumPy, TensorFlow, lab walkthroughs, engineering intuition

**Personal course booklet. Updated November 2026.**

**Audience:** A learner who finished a Python bootcamp and has implemented Andrew Ng labs; understands the broad concepts but wants an understandable, reusable mental model connecting math, code, input data, shapes, optimization and real applications.

**Scope note:** This booklet consolidates the Course 3 subjects supported by the existing repository, supplied labs and the present study history: K-means, anomaly detection, collaborative filtering, content-based filtering, PCA exploratory analysis, and Lunar Lander Deep Q-Learning. It adds links to relevant foundations taught earlier (gradient descent, neural networks, NumPy, feature preprocessing). It does not claim that every possible Course 3 video, optional exercise or hidden assessment was provided or independently checked. Existing detailed notes remain the source of deep code and recorded-lab evidence; this document is the **one coherent reading path**, not a replacement of all original evidence.

**Navigation:**
1. The universal ML mental model
2. Python and NumPy: seeing the code rather than memorizing it
3. Unsupervised learning — what does learning without labels mean?
4. K-means clustering — groups and centroids
5. Anomaly detection — unusual versus normal behavior
6. Collaborative filtering — learn from people's ratings
7. Content-based recommendation — convert features to embeddings
8. PCA — new axes and dimensionality reduction
9. Reinforcement learning — actions, rewards, returns and value
10. Deep Q-Learning — Lunar Lander complete data-to-training story
11. Comparing all the models
12. Engineering reality, failure modes, job-readiness and reusable code
13. Course completion evidence and study roadmap

---

# 1. The universal ML mental model

All the topics in this course become easier when you distinguish **data, features, labels or feedback, a model, a prediction, a cost/score, and a training procedure**.

- **Data:** raw observations, such as a movie rating, server measurements or spacecraft positions.
- **Features:** numbers given to a model, often organized as rows and columns. In code, usually X.
- **Labels:** known answers when they exist, often y. In supervised learning, labels guide training.
- **Model:** a mathematical function with parameters (weights, biases, centroids, statistical means) that map inputs to useful outputs.
- **Prediction/output:** what the model produces. Could be rating, cluster index, probability density, embedding, or Q-values.
- **Objective:** definition of what counts as good output. Could be low squared error, compact clusters, high likelihood, or high cumulative reward.
- **Training:** the process that creates or adjusts model parameters from data or interaction.
- **Inference:** using the already trained model on new inputs. Inference normally does not change its weights.

The universal question for every notebook line is: **Who produced this value? What is its shape? Is it an observed input, a known label, a model output, or a parameter being learned?**

## A tiny mathematical model

Linear regression is a familiar starting point:

~~~text
prediction = w*x + b
error = prediction - actual_y
squared_error = error**2
~~~

- x is data supplied by the world.
- y is a known training label.
- w and b are trainable parameters.
- prediction is the model's current guess.
- an optimizer changes w and b to reduce errors.

That is the foundation of neural networks: instead of one weighted sum, they combine many weighted sums with nonlinear activations. When you see TensorFlow, it is automating these calculations and their gradients, not introducing unrelated mathematics.

**Important distinction:** not all course algorithms have gradient descent. K-means updates centroids by taking group means, PCA decomposes variation, and anomaly detection can estimate Gaussian statistics directly. Do not force neural-network intuition onto every algorithm.

---

# 2. Python and NumPy: getting comfortable with shapes

The programming skill that connects almost every lab is understanding **array shapes and the meaning of an axis**.

A sample matrix X of shape (m,n) usually means **m examples, n features per example**.

~~~python
import numpy as np
X = np.array([[1, 2, 3],
              [4, 5, 6]])
print(X.shape)  # (2, 3)
print(X[0])     # first example, shape (3,)
print(X[:, 1])  # second feature across examples: [2,5]
print(X[1, 2])  # one scalar: 6
~~~

Indexing always follows the particular notebook's data layout. For example, the collaborative filtering Y matrix is **movies × users**, not the usual training matrix notation. Never assume i is the user without reading the diagram and array shapes.

## Axis intuition

- axis=0 in many NumPy reductions collapses rows, producing a result per column.
- axis=1 collapses columns, producing a result per row.
- axis=-1 is the last axis, often the action/features axis of a batch.

For matrix [[1,2,3],[4,5,6]]:
- sum(axis=0) = [5,7,9]
- sum(axis=1) = [6,15]

This is critical in DQN: Q-network output shape (batch, actions). Taking max along axis=-1 gets best action **for each sample**, not one biggest value over the whole batch.

## Dot products versus elementwise operations

Vector dot product: multiply matching entries and sum. [1,2] dot [3,4] = 1*3+2*4=11. NumPy np.dot(v1,v2). Matrix multiplication uses @ or np.matmul, and in TensorFlow tf.linalg.matmul.

Elementwise multiplication: [1,2] * [3,4] = [3,8], not 11. Elementwise squaring X**2 squares each entry; np.sum(X**2) adds them all. This appears in regularization and squared vector distance.

## Broadcasting and dimensionality

The content-based lab uses a 32-element embedding for each batch member: shape (batch,32). Collaborative filtering b has shape (1,num_users), so adding b to a (movies,users) prediction matrix broadcasts the user bias row across movie rows.

Lunar Lander state is one vector (8,), but Keras expects a batch (batch_size,8): np.expand_dims(state,axis=0) gives (1,8), adding a batch axis without adding a new feature.

Do not confuse a tuple (8,) with integer 8. If state_size already equals (8,), Input(shape=state_size) is correct. Input(shape=(state_size,)) creates nested ((8,),), which produced the student's actual Keras TypeError.

## Python patterns to master

~~~python
# Unpacking
height, width = X.shape

# Nested loops: first i across rows, then j across columns
for i in range(height):
    for j in range(width):
        value = X[i, j]

# Accumulator
running_total = 0
for n in [1, 2, 3]:
    running_total += n

# Conditional mask, useful in arrays
mask = np.array([True, False, True])
selected = np.array([10, 20, 30])[mask]  # [10,30]

# Numeric versus string variables
ALPHA = 0.001
print(ALPHA, 'ALPHA')  # 0.001 ALPHA
~~~

**Rule:** inspect shape and dtype first, then the mathematical formula. The course's recurring bugs were not exotic ML failures; they were tuple shapes, indexing, indentation, a mistaken variable name, and masking the wrong term.

---

# 3. Unsupervised learning: no answers column

In supervised learning we know the target y for training rows. In unsupervised learning we typically only have X; the algorithm explores the structure of those inputs.

Example: you collect time windows of server traffic: setup delay, failure rate, retransmission count, throughput. You don't have operator-confirmed labels saying which window belongs to which condition. An unsupervised algorithm can group similar observations or identify unusual ones, but it does **not** know what the groups mean unless you investigate afterward.

K-means: discover groups. Anomaly detection: model normal and flag low-likelihood observations. PCA: find directions of variance, potentially visualize or compress. These are different jobs.

A group in a graph is not automatically a root cause, label or diagnosis. ML makes structure visible; engineering judgment gives it meaning.

---

# 4. K-means clustering — build groups from data

## The intuitive story

Imagine 300 dots on a page. You place k markers (centroids), assign each dot to the nearest marker, then move each marker to the average location of its assigned dots. Repeat.

This is the entire K-means loop:
1. Choose number of clusters K and initialize K centroids.
2. **Assignment step:** for each observation, compute distance to each centroid and choose the closest index.
3. **Update step:** for each centroid, average all observations currently assigned to it.
4. Repeat until assignments/centroids stabilize or iteration limit is reached.

The input features must represent comparable quantities. The distance metric can be dominated by a feature whose numeric scale is much larger; scale features appropriately.

## The distance formula

Squared Euclidean distance between vectors x and c:

~~~text
distance_sq(x,c) = sum_over_features((x_feature - c_feature)**2)
~~~

A closer centroid has smaller squared distance. Square root is not needed for comparing distances because sqrt is monotonically increasing for nonnegative values.

## What you coded in the actual lab

The existing K-means study note records completion of find_closest_centroids and compute_centroids. The former finds the **index** of the smallest distance; the latter calculates a new centroid as the average of assigned points. Key distinction:

- min(distances) gives the **distance value**.
- argmin(distances) gives the **index** of the closest centroid.

For updating centroids, the logical operation is select all points with assignment==k and average by feature. Indexing/group masks must be correct for all clusters, including edge cases such as empty clusters.

## K-means and image compression

The completed notebook also used clusters as an image color palette. Treat each RGB pixel as a 3-number vector, assign pixels to K representative colors, and reconstruct using their centroid colors. The algorithm is still assignment/update; only the meaning of features changes.

## Where K-means can fail

- Requires a choice of K.
- Finds roughly compact clusters under Euclidean distance, not arbitrary nonconvex shapes.
- Sensitive to feature scale, outliers and initialization.
- Labels 0/1/2 are arbitrary identifiers, not ranked quality grades.
- A cluster is not proof of a real-world category or incident cause.

Use it for exploratory grouping, segmentation, compression, or initial candidate discovery. Validate whether the clusters are stable and operationally meaningful.

See UNSUPERVISED_LEARNING_AND_KMEANS.md for the student's exact completed lab, observed centroid values and image example.

---

# 5. Anomaly detection — what looks unusual?

## A different question from K-means

Instead of dividing every observation into K groups, anomaly detection asks whether an observation is sufficiently unlikely under a model of normal behavior.

The course lab estimates means and variances for server features, models their probability densities, and compares the joint density p(x) with a threshold epsilon.

Imagine server CPU temperature usually around 60 with modest variation. A value of 62 is common; a value of 110 is unusual. In real multidimensional data, anomalies may involve unusual **combinations** rather than only large single measurements.

## The Gaussian intuition

For one feature with mean mu and variance sigma_squared, a Gaussian curve is highest near mu and smaller far away. Model learns mu and sigma_squared from training examples. For independent-per-feature Gaussian modeling, combine per-feature densities by multiplying.

A Gaussian density is not exactly the probability of seeing one specific continuous numeric point; it is a **probability density**. For many features, products of densities can be extremely small; production systems often use logarithms for numerical stability.

The lab uses:

~~~text
if p(x) < epsilon: classify as anomaly
else: classify as normal
~~~

The threshold epsilon is important. Too strict: many false alarms. Too loose: miss real anomalies.

## Why precision, recall and F1?

In a rare-anomaly dataset, accuracy can mislead. A detector that always predicts "normal" might have high accuracy and miss every anomaly.

- True positive TP: flagged anomaly that really is anomalous.
- False positive FP: normal example flagged.
- False negative FN: missed anomaly.
- Precision = TP/(TP+FP): when alerting, how often correct?
- Recall = TP/(TP+FN): of true anomalies, how many caught?
- F1 = 2*precision*recall/(precision+recall), when denominator is meaningful.

The course uses labeled cross-validation examples to choose epsilon maximizing F1, even though training the normal density model is unsupervised. This is a **hybrid workflow:** fit density from normal/mostly normal examples, validate threshold using known anomaly labels.

For telecom operations, false positives cause alert fatigue and false negatives can hide outages. An F1-optimized threshold is not automatically the same as an operationally optimal threshold: severe failures may make recall more important.

## What the student learned in code

- Compute feature means and variances across examples (axis=0).
- Apply Gaussian model to every observation.
- Generate threshold candidates.
- Compare predicted flags with validation labels.
- Count TP, FP, FN correctly with Boolean expressions.
- Guard division by zero when a category has no predicted or actual positives.

Use ANOMALY_DETECTION.md for the exact uploaded lab outputs and selected threshold.

---

# 6. Collaborative filtering — learn from movie ratings

## What the data looks like

Movies form rows, users form columns. Y[i,j] is a rating; R[i,j] is 1 if the rating exists and 0 otherwise. A missing rating is not an actual zero-star opinion.

Each movie i gets a learned feature vector X[i] and each user j gets a learned preference/parameter vector W[j], plus bias b[0,j]. They have the same length (number of latent factors).

The prediction formula:

~~~text
prediction(i,j) = dot(W[j], X[i]) + b[0,j]
~~~

The vectors are **latent**: the algorithm is not explicitly given that coordinate 1 means "action" and coordinate 2 means "comedy." Their meanings arise through fitting rating patterns.

## Student-written cost function: every line has a reason

~~~python
# EXPLANATORY REPRODUCTION of code the student already completed and tested
for i in range(nm):           # each movie
    for j in range(nu):       # each user
        if R[i,j] == 1:       # only actual observed ratings
            prediction = np.dot(W[j], X[i]) + b[0,j]
            err = prediction - Y[i,j]
            J += err**2
J = 0.5 * J                    # AFTER both loops
reg_w = (lambda_/2) * np.sum(W**2)
reg_x = (lambda_/2) * np.sum(X**2)
J = J + reg_w + reg_x
~~~

- J starts at zero and accumulates squared errors for observed ratings.
- Multiply by 1/2 **once after both loops**, not after each sample.
- Regularize all values in X and W; the course formula does not regularize b.
- lambda_ determines regularization strength.
- The resulting J is a **total cost**, not average over number of ratings.

Student's unregularized test returned 13.67 with lambda_=0; regularized case expected 28.09; public unit tests passed. Those are source-observed, not independently rerun here.

## Vectorization and TensorFlow

Instead of looping movie/user pairs in Python, matrix multiplication X @ W.T produces all movie/user dot products. Add user bias row by broadcasting; subtract Y; multiply elementwise by R to ignore unrated pairs; square, reduce-sum, add regularization.

The supplied notebook trains raw variables X,W,b with tf.GradientTape and Adam. GradientTape computes derivatives of scalar cost with respect to these trainable variables; Adam changes them. Your cost function alone measures error and does **not** train anything.

## Mean normalization

For each movie, compute its average among actual observed ratings, subtract it from observed ratings before training, and add the mean back after prediction. Do not divide by the rating range or standard deviation when reproducing the specific mean-centering formula from this lab.

## What about a new user or new movie?

This flavor of collaborative filtering learns per-user/per-item parameter rows. A never-seen user/movie lacks a trained parameter row. The system must estimate one, retrain/extend parameters, or use a separate feature-based strategy. Merely inserting a database record does not magically create a meaningful learned vector.

See COLLABORATIVE_FILTERING.md for all shapes, the vectorized implementation, learning loop, tests and predictions.

---

# 7. Content-based filtering — two neural networks and embeddings

## Why two different vectors exist

The real content-based lab gives the user network **14 original input features**, representing average user ratings across 14 genres. The movie network gets **16 original input features**: release year, average movie rating and 14 genre flags. The original stored rows also include metadata ID/count columns, which the network excludes.

These feature vectors are **provided/engineered data**, NOT learned embeddings. The two neural networks transform them into new **32-dimensional learned vectors** that share a comparable output space.

This is a two-tower recommender:
- user_NN: 14 -> Dense(256,relu) -> Dense(128,relu) -> Dense(32,linear)
- item_NN: 16 -> Dense(256,relu) -> Dense(128,relu) -> Dense(32,linear)

Same hidden architecture, independently learned weights. There is no literal "genre 17" or human-readable interpretation for every output coordinate. The 32 numbers are learned representations useful for rating prediction.

## Why the final dot product works

The Keras Functional API connects the two towers. It applies L2 normalization to each output embedding (length approximately one), then computes their dot product. For nonzero unit vectors the dot product equals cosine similarity and lies between -1 and 1. The course also scales rating targets into that interval using MinMaxScaler((-1,1)).

Do not confuse the three preprocessing operations:
1. StandardScaler: standardize original numerical input columns.
2. MinMaxScaler: transform actual rating targets to [-1,1] for training, then inverse-transform predictions.
3. L2 normalization: make model-produced 32-coordinate vectors unit length before the dot product.

The notebook uses MeanSquaredError and Adam(learning_rate=0.01), trains with model.compile / model.fit(...,epochs=30). Keras automatically handles gradients, unlike the explicit custom GradientTape loop from collaborative filtering.

## How a new user works after training

The trained model is a **reusable function**, not a database. Build a new user's 14 genre-average feature vector, apply the original fitted user preprocessing, feed user_NN, and obtain a new 32-D embedding **without retraining the network**. Feed a movie's 16 correctly prepared features into item_NN to get its embedding, then predict compatibility.

If a new user has no preferences, or a new movie lacks the average-rating feature, you have a **cold-start / missing-feature challenge**. You must define a reasonable input strategy; the model cannot infer unknown preferences from nothing. The app database stores the new record, while the model uses features to produce predictions. New labeled ratings may later justify retraining.

## Finding similar movies

The student completed a separate squared Euclidean distance function:

~~~python
d = np.sum((a - b)**2)
~~~

Once every movie has a normalized 32-number embedding, compare movie A with movie B by squared distance. Smaller means closer in the learned space. For unit vectors, squared distance = 2 - 2*(dot product). The lab uses pairwise distances, masks each movie's self-match, and finds nearest other movies.

This is **movie-to-movie similarity**, different from a user-to-movie predicted rating. The code can serve both purposes because the item tower produces learned item representations.

## Critical lab evidence and caveats

Notebook data includes 397 users, 847 movies, 25,521 observed ratings and 50,884 training-example rows after repetition. The latter is **not** 50,884 independent original ratings. User and item tower public tests passed; squared-distance test passed. Reported final training MSE 0.0730 and test MSE 0.0812, but preprocessing was fitted before train/test split, and repeated examples can undermine naive held-out evaluation. Supplied existing-user example failed at uid=2; similar-item genre display copied the first movie's genre into the second movie column. These bugs are documented rather than silently ignored.

See CONTENT_BASED_FILTERING.md for shapes, train/test setup, and code details.

---

# 8. PCA — re-express the data on new axes

## Imagine a long thin cloud of points

If dots on a page lie mainly along a diagonal, you can rotate the axes so one axis follows the long spread. That's the first principal component PC1. The second component captures remaining perpendicular spread. Retaining only PC1 replaces two original coordinates with one new coordinate per point.

PCA chooses new axes as linear combinations of **all** original features. It does not simply choose the highest-variance original column. Principal directions are orthogonal and ordered by decreasing explained variance.

## Exact code and math

Given X shape (m,n), center each feature by its mean vector mu. Let V_k shape (n,k) contain chosen principal directions. Project Z=(X-mu) @ V_k, shape (m,k). Reconstruct approximately Xhat=Z @ V_k.T+mu. If all meaningful components retained, reconstruction is nearly exact; if components discarded, information along discarded axes is lost.

~~~python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
Z = pca.fit_transform(X)       # Learn PC axes and project input
ratio = pca.explained_variance_ratio_
X_approx = pca.inverse_transform(Z)
~~~

fit learns axes and mean. transform projects using **existing** fitted axes; inverse_transform reconstructs; fit_transform combines first two operations. For new data use transform, not fit again.

**Important correction to the lab:** sklearn PCA automatically **centers** columns but does **not** automatically standardize them to unit variance. Consider feature scales/units and fit a StandardScaler on training data only when needed.

## Actual lab checkpoints

The six-point lecture example has PC1 explained variance 0.99244289 (99.24%) and PC2 0.00755711 (0.76%), so 2D→1D keeps almost all variance. Using one PC reconstructs points along a line.

The second example has 500 observations and 1000 features. Scatter plots of original pairs and pairwise correlations show little obvious structure. PCA to 2 dimensions makes visual groups apparent while retaining only ~14.57% total variance; 3D PCA retains 0.208062578, or **20.81%**. The notebook prose says "19%" for three components, but its numeric output is 20.81%. Visual groups are **not automatically validated clusters**.

## Why PCA matters for LLM engineering

LLM-related applications often work with hundreds- or thousands-dimensional embeddings. PCA can show their broad geometry in two or three dimensions, inspect redundancies/drift or sometimes compress storage/search vectors. But high explained variance does not guarantee semantic similarity or nearest-neighbor rankings survive. 2D embedding plots are exploratory. PCA is not a required step for every transformer or RAG system.

Compare:
- PCA: dimensionality reduction/projection.
- K-means: cluster assignment.
- Neural embedding: learned representation optimized for a task.
- t-SNE/UMAP: nonlinear visualization alternatives with different distortions.

See PCA_PRINCIPAL_COMPONENT_ANALYSIS.md for full course notebook and correction notes.

---

# 9. Reinforcement Learning — an agent, not just a static dataset

## What changes from supervised ML?

In movie recommendation, a known rating acts as a label. In Lunar Lander, nobody provides a correct action for every conceivable combination of position, velocity and orientation. The **simulator** creates states and computes rewards based on how an action changes the physical environment. By acting repeatedly, the agent creates its own data.

This is the question the learner asked: "Did someone manually fly two or three landings so the model had an X input?" In this DQN lab: **No**. The simulator starts a lander; env.reset() returns eight state values. Initially epsilon=1.0 makes the agent explore randomly. env.step(action) generates next state, numerical reward and done flag using built-in physics and reward logic.

## What is in the state?

The eight values are x position, y position, x velocity, y velocity, angle, angular velocity, and Boolean left/right leg contacts. Continuous-state means coordinates such as velocity can take continuously varying values; it does NOT mean the state has many fields or that rewards have decimals. The agent still has four discrete engine actions.

The gym version supplied by lab is 0.24.0, LunarLander-v2. Action indexes in THIS source: 0 do nothing, 1 right engine, 2 main engine, 3 left engine. Standard reward shaping favors pad proximity and low velocity, penalizes tilt and engine firing, rewards leg contact, and assigns ±100 for safe landing/crash. Those rules are part of the environment, not labels manually assigned to examples.

## Reward versus return

A reward is immediate feedback from one transition. A return is the discounted sum of rewards from a point onward. Gamma weights future rewards. Q(s,a) is estimated return from a specific initial action; V(s) is estimated return from a state under a policy. If you force a different action, Q can differ from the state's best possible V.

The course quiz with gamma 0.25 and terminal 100 two steps away gives 0 + 0.25*0 + 0.25^2*100 = 6.25. The course quiz Q(5,left) uses next-state V(4)=2.5, giving 0 + 0.25*2.5 = 0.625. These illustrate that reward-at-the-start is not the whole return.

## Where is the learning target?

A transition record contains (S,A,R,S',done), but **no true Q-value**. The agent creates a provisional target:

~~~text
if done:
    y = R
else:
    y = R + gamma * max_a Q_target(S', a)
~~~

That is Bellman bootstrapping. The target reflects one actual observed reward plus the target network's estimate of the best future action.

In batch form the student wrote: rewards + (1-done_vals)*(gamma*max_qsa). The mask removes future contribution on a terminal step **without removing the immediate reward**.

---

# 10. Deep Q-Learning — the whole Lunar Lander notebook

## DQN architecture

A Q-table cannot realistically enumerate every continuous state. The network generalizes: eight input numbers -> Dense(64,relu) -> Dense(64,relu) -> Dense(4,linear). The four outputs estimate returns for four actions. These are **not probabilities** and can be negative.

There are two separate networks with identical architecture:
- main Q-network: predictions and gradient updates.
- target Q-network: slower-moving estimate of future Q for stable Bellman targets.

This is NOT the same as content-based filtering's user/movie two towers. In DQN the second model is a more stable copy of the first role, not a different input modality. The target model is initialized with a snapshot of main weights and subsequently soft-updated.

## Experience replay

Adjacent simulated states are correlated. The buffer retains previous (state,action,reward,next_state,done) records, and the algorithm samples mini-batches randomly rather than always learning only from the latest few consecutive steps. This improves data reuse and reduces temporal correlations.

The course uses deque(maxlen=100000) and a namedtuple Experience. A deque with maxlen automatically forgets oldest records at capacity. A namedtuple carries named transition fields. This is the agent's learning dataset.

## Bellman target and selected-action prediction

For batch size B:
- states shape (B,8).
- target model outputs (B,4) next-action Q estimates.
- tf.reduce_max(...,axis=-1) selects best future Q for **each experience**, producing (B,).
- main model outputs (B,4) current-state estimates.
- tf.stack of row indexes and action indexes + tf.gather_nd selects the chosen action's predicted Q value, shape (B,).
- MSE(y_targets,q_values) measures mean squared difference between those two length-B arrays.

You don't compare the target against all four action values for each experience: the transition recorded only one action actually taken.

## Training the weights

The notebook supplies a custom train function with tf.GradientTape, tape.gradient and optimizer.apply_gradients(zip(gradients, q_network.trainable_variables)). This is automatic differentiation of the scalar loss with respect to the main model's trainable parameters. Adam with ALPHA=1e-3 updates main weights. The target network gets a gradual soft update, not a second ordinary Adam learning pass.

The custom function is decorated with @tf.function to help TensorFlow reduce Python execution overhead.

## The full execution loop

1. Create simulator; initialize replay; synchronize target weights.
2. Start a new episode (env.reset()).
3. Read eight numbers and reshape (8,) to (1,8) with np.expand_dims so Keras sees a batch.
4. Main Q-network calculates four action values.
5. epsilon-greedy policy chooses a random action or highest-valued action. With epsilon=1 initially, it chooses random actions; epsilon later decays toward minimum 0.01.
6. env.step(action) simulates the action and returns next_state, reward, done and info.
7. Save this transition in replay.
8. Every four steps, when buffer has sufficient experiences, sample a mini-batch and call agent_learn.
9. Advance to next_state, accumulate the undiscounted reward, and end episode on done or maximum 1000 timesteps.
10. Add episode score to history and take mean of last 100 scores. If average >=200, save model and stop; otherwise begin another episode (up to 2000).

The 100-episode mean is an **undiscounted performance score**. The Bellman gamma-discounted return is a different concept. Keeping those distinct prevents confusion.

## The student's real errors and why they mattered

- Input(shape=(state_size,)) when state_size was already tuple (8,) created nested tuple ((8,),), causing dimension TypeError. Correct Input(shape=state_size).
- Adam(learning_rate='ALPHA') passes literal string, not numeric ALPHA variable.
- Multiplying (1-done) by the entire target expression erased terminal rewards; the mask must only apply to future Q.
- Typo gama versus gamma caused a variable mismatch.
- MSE(y_targets,q_values) was implemented correctly.
- Both network tests and optimizer test passed; compute_loss public test passed.

These are not incidental anecdotes: they teach the core engineering technique of inspecting **types, shapes, the intended mathematical scope of each operation, and source variable names**.

## The observed training result

Notebook's moving-average score:
- episode 100: -154.14
- episode 200: -88.374
- episode 300: -39.86
- episode 400: 41.742
- episode 500: 103.20
- episode 600: 132.49
- episode 650: 201.03

The notebook prints "Environment solved in 650 episodes!" and total runtime 593.10 seconds (9.89 min). It saves lunar_lander_model.h5 on reaching its criterion. The video code demonstrates the model in the simulator. This is source-observed evidence of meeting the notebook's training threshold, not an independently repeated generalization study across random seeds.

## How RL transfers to LLMs

Reinforcement Learning concepts matter for later model post-training: rewards, policies, value estimates, sampling, optimization and evaluation. But DQN with a four-action Lunar Lander and a Q-value network is **not** a direct template for training current language models. Don't assume LLM RLHF must implement this particular network and replay algorithm. The transferable foundation is the difference between choosing actions, observing feedback, optimizing expected outcomes, and evaluating whether a trained policy actually behaves as desired.

The dedicated REINFORCEMENT_LEARNING_DQN_LUNAR_LANDER.md contains the specific source details, tensor shapes, and full code-reference table.

---

# 11. Cross-topic comparison — all topics in one view

| Algorithm | Main input | Learning signal | What it produces | Typical fit process |
|---|---|---|---|---|
| K-means | X feature rows | Distances, centroid means | Cluster labels and centroids | Alternating assignment/update |
| Anomaly detection | Mostly normal X | Density/statistics, threshold validation | Density and anomaly flag | Estimate mean/variance, pick epsilon |
| Collaborative filtering | Observed ratings Y,R | Rating prediction errors | User/movie latent factors, ratings | Minimize cost (with regularization) |
| Content-based recommender | User/movie content + ratings | Rating MSE | Embeddings and preference scores | Backprop through two neural towers |
| PCA | X feature rows | Unsupervised variance | New coordinates and PCs | SVD/eigen-decomposition |
| DQN RL | States and transitions | Rewards + bootstrapped Q targets | Action values and learned policy | Replay + Bellman loss + gradients |

The most important question for recognizing an algorithm is **what feedback exists**:
- Known y label? Usually supervised.
- Only features, discover structure? Unsupervised.
- Agent controls actions and receives later rewards? RL.

The most important question for debugging an algorithm is **what object is being learned**:
- Centroid coordinates.
- Gaussian means/variances and threshold.
- User/movie latent factors.
- Neural-network weights that generate embeddings.
- PCA principal axes.
- Q-network weights estimating action values.

---

# 12. Engineering use: from notebook to real systems

## Fitting versus predicting versus updating

A common confusion across labs: a model is not the application's database.

Training/fitting creates parameters: centroid locations, PCA axes, recommender weights or network weights. Predicting uses already fitted parameters on new features. Inserting a new user/movie record into a database changes neither the model structure nor its learned weights automatically.

For content-based recommendation, a new user with sufficient genre-average features can be converted into a new embedding **without retraining** by using the same saved input schema, scaler and network. A new movie can likewise produce an embedding from its features; a missing average movie rating creates a feature-quality problem. For pure collaborative filtering with per-user/item learned parameters, a brand-new user or movie requires a strategy for a new latent vector or model update.

PCA can transform new observations using saved axes; it must not refit on the new observation. Anomaly detectors can evaluate new measurements using saved normal-distribution statistics. K-means can assign a new data point to the nearest fitted centroid. A trained Q-network can choose actions in future Lunar Lander episodes using learned values, without updating weights in inference mode.

## Evaluate beyond a passing notebook

A public unit test verifies particular implemented behavior, not full downstream model quality. Every algorithm needs task-specific evaluation:
- K-means: stability, distance structure, useful business interpretation.
- Anomaly detection: alert precision/recall, false alarms, missed incidents, data drift.
- Recommender: held-out rating error and actual top-K ranking/coverage, cold-start and leakage.
- PCA: explained variance **and** downstream predictive/retrieval quality when used for compression.
- RL: success distribution across different environment seeds, collisions, generalization and performance consistency, not one favorable episode.

Training/test leakage is a major operational concern. Fit scalers or feature extraction steps only on training data, then reuse them on validation/test/new inputs. The uploaded content-based lab pre-fits scalers and repeats examples before a row split; interpret its test loss with these caveats. A simulator reward function may also be poorly aligned with true operational safety (reward hacking).

## Data lineage and reproducibility

For a professional implementation retain:
- Input schema (feature names, order, units, handling of missing values).
- Saved scaler/statistics, model architecture, weights, and their versions.
- Training/validation split rules and random seeds.
- Exact evaluation metrics and datasets.
- Baseline models and tests for shape/typing.
- Evidence of runtime/accuracy/latency tradeoffs.
- Monitoring for model drift and production failures.

Those requirements are especially important in telecom/voice applications, where a statistically interesting model output can still be operationally misleading. For instance, a Q-value estimate or anomaly score is not itself a verified subscriber incident.

---

# 13. Your next learning method: what to remember, what to implement, what to look up

## Must understand without notes

1. Which data exists and who produces it.
2. What model parameters mean.
3. Distinction between observed label, model prediction, and loss.
4. Array shapes and axes.
5. The actual mathematical objective and what constitutes success.
6. Training versus inference; a model is not a database.
7. Missing data, cold start, overfitting, leakage, stability and evaluation limits.
8. Why each algorithm is appropriate for a specific type of question.

## Should be able to code with ordinary reference help

- NumPy loops, slicing, boolean masks, squared distance, mean/variance, broadcasting, dot products.
- sklearn fit/transform/predict patterns, proper preprocessing split, clustering/PCA workflow.
- Keras Sequential/Functional architecture setup and input/output shapes.
- Loss/optimizer wiring, GradientTape and one weight-update step.
- Construct an RL experience record, calculate a Bellman target correctly, and interpret chosen-action Q selection.
- Write tests and inspect intermediate values to diagnose why output is wrong.

## Reasonable to look up

- Exact tf.gather_nd signature, axis and shape intricacies.
- Dense/optimizer argument names and library-version API differences.
- Gym vs Gymnasium step/reset contracts.
- Plotting helper parameters.
- Advanced SVD math, optimizer derivations, more complex RL methods.

Learning is not memorizing every function call. Real engineering means you can **explain what must happen, locate the appropriate API, test it and recognize when it is wrong**.

## The golden guided-coding method

This training process is the recommended standard for all future exercises:
1. Use the exact lab instructions and starter notation; no invented tasks.
2. Explain *what* the next operation does, with a separate general Python/NumPy syntax mini-example.
3. Student writes graded implementation independently, one coherent piece at a time.
4. Instructor checks precise mistakes and the student fixes them.
5. Run the official public tests.
6. **After passing, on request**, explore the lower-level shapes, math and data flow in depth.
7. Update one canonical cheat sheet and focused topic notes; preserve previous established facts.

## Evidence index and further reading

- GRADIENT_DESCENT.md — cost and derivative foundations.
- NUMPY_FOR_ML.md — array dimensions, indexing, vectorization.
- TENSORFLOW_KERAS_FOR_ML.md — Sequential, Dense, Functional API, GradientTape, optimizers.
- UNSUPERVISED_LEARNING_AND_KMEANS.md — completed clustering exercises.
- ANOMALY_DETECTION.md — Gaussian thresholding, precision/recall/F1, original lab details.
- COLLABORATIVE_FILTERING.md — student-written cost and TensorFlow training.
- CONTENT_BASED_FILTERING.md — two-tower embeddings, preprocessing, model fitting and similar items.
- PCA_PRINCIPAL_COMPONENT_ANALYSIS.md — projection, variance and source corrections.
- REINFORCEMENT_LEARNING_DQN_LUNAR_LANDER.md — full Lunar Lander DQN study note, exact tensor and source detail.
- PYTHON_ML_CODING_CHEAT_SHEET.md — **the single canonical reusable syntax reference**.
- GUIDED_CODING_GOLDEN_RULE.md — teaching contract.

**Final mental map:** Data and observations -> features and proper shapes -> mathematical objective -> model fit/learning -> predictions/decisions -> evaluation -> safe reuse with new inputs. This map will carry forward to PyTorch, Transformers, embeddings, RAG systems, LLM evaluation and production AI services.
