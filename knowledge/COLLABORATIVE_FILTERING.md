# Collaborative Filtering — Andrew Ng lab, from loops to TensorFlow training

Source: supplied **Practice lab: Collaborative Filtering Recommender Systems** (Andrew Ng Machine Learning Specialization). This note explains the entire lab and its Python/TensorFlow operations. Graded Exercise 1 was independently completed by the student; public tests passed. No claim about hidden grading.

## Big picture — explain it simply

Imagine a room with movies and people. We know a few ratings, but most people have not rated most movies. The algorithm learns a small numerical profile for each movie (`X`) and each user (`W`), plus one bias per user (`b`). Comparing a movie profile to a user profile gives a predicted rating. It learns these profiles by repeatedly comparing predictions against ratings that actually exist, then changing the profiles to reduce errors.

There are **two different jobs**:
1. **Cost function**: measure how wrong the current predictions are (plus regularization).
2. **Training loop**: adjust the learned values until the cost becomes smaller.

## Dataset and shapes

The lab uses a reduced MovieLens-derived dataset: **4,778 movies, 443 users**. For the first cost exercise it uses **10 features**. Later training reinitializes with **100 features**. These are two different stages.

| Array | Lab shape before adding new user | Meaning |
|---|---|---|
| `Y` | (4778, 443) | Rating per movie-row, user-column; zero denotes unrated in this lab |
| `R` | (4778, 443) | 1 if observed rating, else 0 |
| `X` | (4778, 10) | A feature vector per movie |
| `W` | (443, 10) | A parameter vector per user |
| `b` | (1, 443) | One bias per user |

Mathematical movie index `i` selects row `X[i]`, and user index `j` selects row `W[j]`; `Y[i,j]` and `R[i,j]` refer to the same pair. `b[0,j]` uses row zero because `b` is shaped as a one-row two-dimensional array.

## Section 3 — load and inspect

`load_precalc_params_small()` supplies precomputed `X,W,b` for checking the cost before training. `load_ratings_small()` supplies `Y,R`. `Y.shape` returns (movies, users). `np.mean(Y[0, R[0,:].astype(bool)])` averages only observed ratings for the first movie: `R[0,:]` takes its rating-indicator row, `.astype(bool)` turns 0/1 into False/True, boolean indexing filters `Y[0,...]`.

## Section 4 — student-written for-loop cost

For every movie and every user, only if `R[i,j] == 1`:
- predict: `np.dot(W[j], X[i]) + b[0,j]`;
- error: prediction minus `Y[i,j]`;
- add the error squared to a running total.

After **both loops**, multiply accumulated squared error by `1/2` **once**. Add `lambda_/2 * np.sum(W**2)` and `lambda_/2 * np.sum(X**2)`; the lab does **not** regularize bias `b`. `lambda_` changes penalty strength; `lambda_=0` removes it. The result is **total**, not average, cost.

Tests in the supplied lab: unregularized `13.67`, regularized at `lambda_=1.5` `28.09`, public tests `All tests passed!`.

## Vectorization — same job, fewer explicit Python loops

The lab provides (not graded for student implementation) a TensorFlow cost:
```python
j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y) * R
J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
```
Shape story: `X` = (movies, features); `tf.transpose(W)` = (features, users); matrix multiplication gives (movies, users). `+ b` broadcasts one user-bias row across every movie. `- Y` yields errors. `* R` zeroes errors at unrated entries. Squaring and reducing gives the same data-fit cost. This is elementwise `*`, **not** matrix multiplication. Compare with the student loop: one pair at a time versus one full matrix at a time. TensorFlow operations remain differentiable for training.

## Section 5 — adding a new user and normalizing

`my_ratings = np.zeros(num_movies)` starts an unrated vector; setting selected indices records user-provided ratings. `np.c_[my_ratings, Y]` adds the new user as column zero of `Y`. `np.c_[(my_ratings != 0).astype(int), R]` adds a corresponding 0/1 column to `R`. After this, the user count increases by one. `normalizeRatings(Y, R)` returns normalized ratings `Ynorm` and movie means `Ymean`, so training works on mean-centered observed ratings; later the means are added back. This normalization helper comes from `recsys_utils`, not Python or TensorFlow.

## Initializing trainable TensorFlow variables

```python
num_movies, num_users = Y.shape
num_features = 100
tf.random.set_seed(1234)
W = tf.Variable(tf.random.normal((num_users, num_features), dtype=tf.float64), name='W')
X = tf.Variable(tf.random.normal((num_movies, num_features), dtype=tf.float64), name='X')
b = tf.Variable(tf.random.normal((1, num_users), dtype=tf.float64), name='b')
optimizer = keras.optimizers.Adam(learning_rate=1e-1)
```

`tf.random.normal` generates random starting parameters; `tf.Variable` makes them mutable and trainable; `dtype=tf.float64` specifies 64-bit floating-point values; `tf.random.set_seed` helps reproducibility (not a promise of identical results on every system). `Adam` is an optimizer that uses gradients to adapt parameter updates. `1e-1` means 0.1.

## The custom training loop — the central TensorFlow lesson

Unlike a typical Keras `model.compile(...)` / `model.fit(...)` setup, this lab optimizes raw matrix variables directly. The loop:
```python
iterations = 200
lambda_ = 1
for iter in range(iterations):
    with tf.GradientTape() as tape:
        cost_value = cofi_cost_func_v(X, W, b, Ynorm, R, lambda_)
    grads = tape.gradient(cost_value, [X, W, b])
    optimizer.apply_gradients(zip(grads, [X, W, b]))
```

- `with ... as tape`: record TensorFlow operations to support automatic differentiation.
- Calling the cost function: **forward computation**, computes one scalar cost.
- `tape.gradient(cost_value, [X,W,b])`: compute derivatives of cost with respect to each variable; result aligns with parameter list.
- `zip(grads, [X,W,b])`: pair each gradient with its variable (general Python utility).
- `apply_gradients(...)`: Adam changes variables to lower cost; the cost function alone never updates them.
- Repeat; record the loss periodically. The lab shows cost falling from about 2,321,191 at iteration 0 to 2,902 at iteration 180. This is a **training-loss trace**, not evidence of unseen-user generalization.

**Important:** automatic differentiation is not the same as constructing a neural-network layer. TensorFlow can optimize any differentiable computation built from supported operations, including collaborative filtering.

## Section 6 — predictions and recommendation ranking

`p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()` computes every predicted normalized rating. `.numpy()` converts eager TensorFlow tensors/variables to NumPy arrays (when supported). `pm = p + Ymean` restores per-movie means. `my_predictions = pm[:, 0]` extracts predictions for the newly inserted user (column zero). `tf.argsort(..., direction='DESCENDING')` yields indices ranked by predicted rating. Loop through top indices, exclude items already rated (`my_rated`), and print suggestions.

The notebook then uses a Pandas DataFrame to consider rating counts, average ratings and predicted ratings; sorting by another metric changes the final ranking. Predicted ratings for a new user are estimates, not certainty.

## Python / NumPy / TensorFlow translation table

| Goal | Syntax | Why |
|---|---|---|
| dimensions | `nm, nu = Y.shape` | tuple unpacking |
| check existing rating | `if R[i,j] == 1:` | guard |
| single-vector dot product | `np.dot(u, v)` | multiply matching coordinates, sum |
| sum squared elements | `np.sum(A**2)` | elementwise square and reduction |
| matrix prediction | `tf.linalg.matmul(X, tf.transpose(W))` | all movie-user combinations |
| mask missing ratings | `errors * R` | set missing entries to zero |
| reduce full tensor | `tf.reduce_sum(t)` | scalar aggregation |
| add new column | `np.c_[new_column, matrix]` | horizontally combine |
| normalize ratings | `normalizeRatings(Y,R)` | helper routine supplied by lab |
| trainable parameter | `tf.Variable(...)` | mutable differentiable value |
| random initial values | `tf.random.normal(shape, dtype=...)` | initialize optimization |
| record autodiff | `with tf.GradientTape() as tape:` | track differentiable operations |
| calculate gradients | `tape.gradient(cost, variables)` | derivatives |
| update parameters | `optimizer.apply_gradients(zip(grads, variables))` | optimizer step |
| tensor to NumPy | `variable.numpy()` | interoperability |
| rank indices | `tf.argsort(scores, direction='DESCENDING')` | descending ranking |

## Common pitfalls

- `i`=movie, `j`=user; swapping them breaks indexing.
- `b[0,j]` is a scalar; `b[j]` does not refer to user j in this lab's shape.
- Don't halve `J` inside the loops.
- Don't include unrated pairs in the observed-rating cost.
- Regularization includes **all** `X,W` entries, even for unobserved pairs; not `b`.
- `tf.linalg.matmul` and elementwise `*` mean different things.
- GradientTape needs differentiable TensorFlow operations and the variables used within recorded computation.
- User column zero is newly added **only after** the `np.c_` insertion.
- Don't confuse initial 10 feature vectors with training's later 100 feature vectors.
- Notebook instruction: do not change non-graded cells or add cells before grading.

## Verification and next learning

Public lab output: 13.67, 28.09 and `All tests passed!` for the student's loop implementation. Vectorized TensorFlow example in supplied notebook returns the same 13.67 and 28.09. These are notebook-observed outputs, not independently rerun results.

Next learning: alter a small independent TensorFlow variable using one `GradientTape` step, inspect gradient shape, then trace what `Adam` updates. Follow [guided coding golden rule](GUIDED_CODING_GOLDEN_RULE.md).
