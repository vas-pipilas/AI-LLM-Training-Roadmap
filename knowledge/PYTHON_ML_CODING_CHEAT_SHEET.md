# Python ML Coding Cheat Sheet — the canonical living reference

**Start here when I need Python syntax for an ML lab.** Read a small section, run it, change a value, and explain what changed. I do not need a second generic Python course or to memorize every API.

**Status: 18 October 2026.** Grounded in my completed Python foundation and my Machine Learning Specialization notes: NumPy forward passes, the `400 → 25 → 15 → 1` digit network, gradients, Week 3 preprocessing/model evaluation, and the λ bias/variance discussion. This is a study reference, not a claim that I have mastered every section. Intermediate additions are marked explicitly.

**Maintenance rule:** extend this same canonical file as I learn new coding topics. Add a small runnable example, expected shapes/output, the mistake that prompted it, and a navigation entry. Preserve the focused notes linked below. GitHub is the durable source of truth; chat memory is not a store for the full sheet verbatim. When I ask for “my Python cheat sheet,” use this file and its latest repository version.

## Navigation

| Level | Jump to | Use it for |
| --- | --- | --- |
| Start | [0. Run the examples](#0-run-the-examples) | Environment, notation, dependencies |
| Beginner | [1. Python essentials](#1-python-essentials) | Lists/dicts, loops, functions, unpacking, f-strings |
| Beginner | [2. NumPy creation and shapes](#2-numpy-creation-and-shapes) | Arrays, dtypes, rows, columns, reshape |
| Beginner | [3. NumPy selection and axes](#3-numpy-selection-and-axes) | Indexing, masks, manual mean, population variance; [quick lookup](#numpy-for-the-specialization-quick-lookup) |
| Beginner → intermediate | [4. NumPy ML calculations](#4-numpy-ml-calculations) | Broadcasting, vectorization, matmul, stability |
| Beginner | [5. Matplotlib pyplot](#5-matplotlib-pyplot) | Scatter, lines, histograms, subplots, export |
| Beginner | [6. TensorFlow and Keras foundations](#6-tensorflow-and-keras-foundations) | Tensors, Dense, Sequential, shapes |
| Beginner → intermediate | [7. Outputs, losses, and training](#7-outputs-losses-and-training) | Logits, sigmoid, softmax, compile/fit/predict |
| Intermediate practice | [8. TensorFlow task recipes](#8-tensorflow-task-recipes) | Complete binary, multiclass, regression examples |
| Intermediate practice | [9. GradientTape](#9-gradienttape) | See a gradient and one optimizer update |
| Beginner → intermediate | [10. sklearn preprocessing and evaluation](#10-sklearn-preprocessing-and-evaluation) | Fit/transform, Pipeline, train/CV/test |
| Intermediate practice | [11. Bias, variance, and regularization](#11-bias-variance-and-regularization) | Diagnose errors, tune λ, learning curves |
| Quick lookup | [12. Debugging and copy recipes](#12-debugging-and-copy-recipes) | Common mistakes and tiny reusable checks |
| Beginner → intermediate | [12a. Decision trees](#12a-decision-trees--concise-syntax) | Current-node IDs, entropy/splitting pitfalls; fit/predict and growth controls |
| Beginner → intermediate | [12b. Trees Ensemble lab](#12b-trees-ensemble-lab--pandas-forests-and-xgboost) | pandas encoding, tuning loops, random forest, XGBoost early stopping |
| Beginner → intermediate | [12c. Unsupervised learning and K-means](#12c-unsupervised-learning-and-k-means) | Coordinates, min/argmin, clustering syntax and LLM distinction |
| Beginner | [12d. Anomaly detection](#12d-anomaly-detection) | Why counts grade epsilon; Boolean counting, guards, threshold sweep and masks |
| Reference | [13. Related notes and API sources](#13-related-notes-and-api-sources) | Deeper explanations and official docs |
| Maintenance | [14. How this sheet grows](#14-how-this-sheet-grows) | Update policy and verification record |

## 0. Run the examples

Each Python block is runnable **independently**, except blocks explicitly labeled “Continue.” Install only what the section needs, in the environment/kernel actually running the notebook. TensorFlow examples use TensorFlow 2's `tf.keras` interface; hardware and releases can affect exact learned values.

```bash
python -m pip install numpy matplotlib scikit-learn
# Only for the TensorFlow sections, in a compatible environment:
python -m pip install tensorflow
```

```python
import sys
import numpy as np             # numerical arrays; np is an alias
import matplotlib.pyplot as plt  # plotting; plt is an alias
from sklearn.preprocessing import StandardScaler  # one named class

print(sys.executable)  # verify the interpreter behind this notebook/script
print(np.__version__)
```

An import makes names available; it does not train a model. `object.method(...)` calls behavior belonging to an object. `name=value` in a call supplies a keyword argument. A trailing underscore, as in sklearn's `mean_`, usually means an attribute learned during fitting.

**Shape vocabulary used throughout:**

| Symbol/shape | Meaning |
| --- | --- |
| `m`, `n`, `k` | examples, input features, output neurons/classes |
| `X: (m, n)` | feature matrix: examples are rows |
| `y: (m,)` | one target per example, flat |
| `y: (m, 1)` | one target per example, column |
| `W: (n, k)`, `b: (k,)` | Dense weights and biases |
| `X @ W + b: (m, k)` | one output row for each example |
| `Input(shape=(n,))` | Keras shape of **one** example; batch dimension omitted |

## 1. Python essentials

### Values, containers, and selection

```python
learning_rate = 1e-3  # float: 0.001; ** means power, not ^
epochs = 10          # int
training = True      # bool: True / False
result = None        # no result yet
features = [2.0, 3.0, 4.0]       # list: ordered, mutable
shape = (3, 1)                   # tuple: ordered, immutable
config = {"epochs": epochs, "learning_rate": learning_rate}  # dict

features.append(5.0)             # changes the list; returns None
print(features[0], features[-1]) # 2.0 5.0; indices start at zero
print(features[1:3])             # [3.0, 4.0]; stop index excluded
print(features[::2])             # [2.0, 4.0]; every second item
print(features[::-1])            # [5.0, 4.0, 3.0, 2.0]
print(config["epochs"])         # 10; missing key would raise KeyError
print(config.get("batch_size", 32))  # default 32 if key missing
print(2 ** 3, 7 // 2, 7 % 2)    # 8 3 1: power, floor division, remainder
```

`features = features.append(5)` is a mistake: it replaces the list with `None`. `=` assigns a value; `==` compares values. `is None` checks for the special missing-value object. Use `and`, `or`, `not` for ordinary Python booleans; array masks need different operators, shown later.

### Loops, conditions, and comprehensions

```python
losses = [0.9, 0.5, 0.2]
total = 0.0
for epoch, loss in enumerate(losses, start=1):
    total += loss  # total = total + loss
    print(f"epoch={epoch}, loss={loss:.2f}")  # epoch=1, loss=0.90 ...

for i in range(3):  # 0, 1, 2; range stop excluded
    if i == 1:
        continue   # skip the rest of this iteration
    print(i)       # 0, then 2

step = 0
while step < 2:
    step += 1      # update the condition, or risk an infinite loop

squares = [x ** 2 for x in range(4)]     # [0, 1, 4, 9]
large = [x for x in losses if x > 0.4]   # [0.9, 0.5]
by_epoch = {i: loss for i, loss in enumerate(losses)}
for name, value in {"train": 0.1, "cv": 0.3}.items():
    print(f"{name}: {value:.1%}")        # train: 10.0%; cv: 30.0%

for truth, guess in zip([0, 1, 1], [0, 0, 1]):
    print(truth == guess)                # True, False, True
assert squares == [0, 1, 4, 9]
```

Indentation defines the body. `break` exits a loop entirely. `zip` pairs corresponding items and normally stops at the shorter input; use `zip(a, b, strict=True)` when equal lengths are required (Python 3.10+). Comprehensions create containers; a NumPy vectorized calculation operates on arrays directly.

### Functions, return values, and unpacking

```python
def squared_error(y_true, y_pred, *, half=False):
    """Return MSE for equally long, non-empty Python sequences."""
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("Need equally long, non-empty sequences")
    errors = [(p - y) ** 2 for y, p in zip(y_true, y_pred)]
    mse = sum(errors) / len(errors)
    return mse / 2 if half else mse

print(squared_error([1, 3], [2, 1]))             # 2.5
print(squared_error([1, 3], [2, 1], half=True))  # 1.25

def dimensions():
    return 60, 4   # returns a tuple

m, n = dimensions()                    # unpack two returned values
first, *middle, last = [1, 2, 3, 4]     # 1, [2, 3], 4
options = {"half": True}
print(squared_error([1], [3], **options))  # ** unpacks keyword arguments: 2.0
print(f"X has {m} rows and {n} columns")

rows = [(1, 0.4), (2, 0.2)]
best = min(rows, key=lambda row: row[1])  # small unnamed function
assert best == (2, 0.2)
```

`def` defines a function; it runs when called. `return` sends a value to the caller and ends that call; `print` only displays something. `*` in the function signature makes `half` keyword-only. `lambda` is a Python keyword for an anonymous function; write `lambda_` for a numeric regularization variable. Avoid mutable defaults such as `history=[]`; use `None` and create a new list inside.

List copying also matters: `b = a` shares the same list. `b = a.copy()` copies its outer container; nested lists still share their inner objects unless copied more deeply.

## 2. NumPy creation and shapes

### Build arrays and inspect them

```python
import numpy as np

v = np.array([1, 2, 3], dtype=np.float64)
X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
print(v.shape, X.shape, X.ndim, X.size)  # (3,) (2, 3) 2 6
print(X.dtype)                          # float64
print(np.zeros((2, 3)))                 # two rows of three zeros
print(np.ones((1, 3)))                  # [[1. 1. 1.]]
print(np.full((2, 1), 7))               # [[7], [7]]
print(np.eye(2))                        # [[1., 0.], [0., 1.]]
print(np.arange(0, 6, 2))               # [0 2 4]; stop excluded
print(np.linspace(0, 1, 5))             # [0. 0.25 0.5 0.75 1.]; endpoint included

rng = np.random.default_rng(42)         # dedicated reproducible generator
noise = rng.normal(loc=0.0, scale=1.0, size=(4, 2))
indices = rng.integers(0, 10, size=3)    # integers 0..9
assert noise.shape == (4, 2)
assert np.asarray([1, 2]).shape == (2,)
print([1, 2] * 2, np.array([1, 2]) * 2)  # [1,2,1,2] versus [2 4]
```

`np.asarray` accepts an existing array or sequence and may reuse array storage. `astype(np.float32)` converts a dtype and usually allocates a new array. Integer arrays cannot safely hold arbitrary fractions; specify float when computing gradients or normalized data. Avoid `np.empty` until needed: it contains uninitialized values, not zeros.

### Count rows versus select a row

`np` is the NumPy module alias; `X` is an array variable I created. Call `np.array(...)` or `np.mean(X, axis=0)`; inspect `X.shape` or select `X[0]`. Writing `np.X` asks for a module attribute named X and does not access my variable.

```python
import numpy as np

X = np.array([[1., 2.], [3., 4.], [5., 6.]])
m, n = X.shape                 # unpack (3, 2): m rows, n features
assert X.shape == (3, 2)
assert X.shape[0] == m == 3     # integer COUNT
assert X.shape[1] == n == 2
np.testing.assert_array_equal(X[0], [1., 2.])  # first row's VALUES
assert X[0].shape == (2,)
assert list(range(m)) == [0, 1, 2]
```

`X[0]` is the first sample, not the row count. Even `len(X[0])` is the feature count for this 2-D matrix. `for i in range X[0]:` has invalid syntax. Use `for i in range(X.shape[0]):`: parentheses supply the integer count, the colon begins the body, and indentation puts work inside it. `m, n = X.shape` requires a 2-D array; a flat `(3,)` vector has no `shape[1]`.

### Reshape, transpose, and combine

```python
import numpy as np

v = np.arange(6)
column = v.reshape(-1, 1)  # (6, 1): six examples, one feature
row = v.reshape(1, -1)    # (1, 6): one example, six features
X = v.reshape(2, 3)       # [[0,1,2], [3,4,5]]
print(X.T)               # [[0,3], [1,4], [2,5]]; shape (3, 2)
print(v.T.shape)          # (6,); transposing a 1-D array adds no dimension!
assert column.shape == (6, 1) and row.shape == (1, 6)
assert X.ravel().shape == (6,)      # flatten; may share storage
assert X.flatten().shape == (6,)   # flatten; always copies
assert np.squeeze(column, axis=1).shape == (6,)
assert v[:, np.newaxis].shape == (6, 1)

print(np.column_stack([v, v ** 2]).shape)     # (6, 2): feature columns
print(np.concatenate([X, X], axis=0).shape)  # (4, 3): more rows
print(np.concatenate([X, X], axis=1).shape)  # (2, 6): more columns
print(np.stack([X, X], axis=0).shape)        # (2, 2, 3): NEW axis
```

Only one reshape dimension can be `-1`; the total element count must stay the same. Reshape changes grouping; transpose swaps axes. They are not interchangeable. `squeeze()` without an axis can accidentally remove the batch dimension of one example; specify which axis to remove.

## 3. NumPy selection and axes

### Indexing, slicing, views, and masks

```python
import numpy as np

X = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
print(X[1, 2])         # 60: row 1, column 2
print(X[0])            # [10 20 30]; shape (3,)
print(X[0:1, :])       # [[10 20 30]]; shape (1, 3), keep batch axis
print(X[:, 1])         # [20 50 80]; shape (3,)
print(X[:, 1:2])       # [[20], [50], [80]]; shape (3, 1)
print(X[:2, ::2])      # [[10 30], [40 60]]
print(X[[0, 2]])       # select rows 0 and 2; shape (2, 3)

view = X[:1]           # basic slice usually shares original storage
view[0, 0] = -10       # changes X too!
independent = X[:1].copy()
independent[0, 0] = 999
assert X[0, 0] == -10

mask = (X[:, 0] >= 40) & (X[:, 1] < 80)  # per-row conditions
print(X[mask])                              # [[40 50 60]]
print(X[X > 50])                            # [60 70 80 90]; flattened selection
print(np.where(X > 50, 1, 0))               # same (3, 3) shape
assert np.any(X < 0) and not np.all(X > 0)
```

Use parenthesized conditions with `&` (and), `|` (or), `~` (not) for arrays. `if X > 0:` is ambiguous: do I mean any value or all values? Use `.any()` or `.all()`. Boolean/advanced indexing returns a copy when reading; `X[mask] = value` can assign into the original array.

### Axis: which dimension disappears?

```python
import numpy as np

X = np.array([[1., 2., 3.], [4., 5., 6.]])  # (2 rows, 3 columns)
print(X.sum())                  # 21.0: all entries
print(X.sum(axis=0))            # [5. 7. 9.]: combine rows, one result per column
print(X.sum(axis=1))            # [6. 15.]: combine columns, one result per row
print(X.mean(axis=0))           # [2.5 3.5 4.5]; shape (3,)
print(X.mean(axis=1, keepdims=True))  # [[2.], [5.]]; shape (2, 1)
print(X.std(axis=0))            # [1.5 1.5 1.5]; population std, ddof=0
print(X.var(axis=0))            # [2.25 2.25 2.25]
print(X.min(axis=0), X.max(axis=0))  # [1. 2. 3.] [4. 5. 6.]
print(np.ptp(X, axis=0))        # [3. 3. 3.]: max minus min
print(np.median(X), np.percentile(X, 50))  # 3.5 3.5
print(X.argmax(axis=1))         # [2 2]: index of largest entry in each row
assert X.mean(axis=0, keepdims=True).shape == (1, 3)
```

For a 2-D ML matrix, `axis=0` gives a statistic **per feature**. `axis=1` gives one **per example**. `axis=-1` means the last axis. `keepdims=True` keeps the reduced axis at length 1, useful for broadcasting. `ddof=1` computes sample standard deviation; it differs from the default `ddof=0` used by StandardScaler.

### From my averaging loop to feature statistics

For three samples with two features, calculate two means. Start the accumulator before `+=`, add every complete row, then divide once **after** the loop.

```python
import numpy as np

X = np.array([[1., 2.], [3., 4.], [5., 6.]])
m, n = X.shape
mu_manual = np.zeros(n)         # [0., 0.]: one running sum per feature
for i in range(m):              # i = 0, 1, 2
    mu_manual += X[i]
mu_manual = mu_manual / m       # [9., 12.] / 3 -> [3., 4.]

mu = np.mean(X, axis=0)
var = np.var(X, axis=0, ddof=0)  # population variance; divide by m
np.testing.assert_allclose(mu_manual, [3., 4.])
np.testing.assert_allclose(mu, np.sum(X, axis=0) / m)
np.testing.assert_allclose(var, [8 / 3, 8 / 3])
np.testing.assert_allclose(var, np.mean((X - mu) ** 2, axis=0))
assert mu.shape == var.shape == (n,)
assert np.sum(X) == 21.0        # no axis: sum every entry
assert np.mean(X) == 3.5        # no axis: ONE mean of all six entries
np.testing.assert_allclose(np.mean(X, axis=1), [1.5, 3.5, 5.5])
```

Trace the running sums: `[0, 0] → [1, 2] → [4, 6] → [9, 12]`. The first mean is `(1+3+5)/3 = 3`; the second is `(2+4+6)/3 = 4`. For feature 0, squared deviations are `[4, 0, 4]`, so variance is `8/3`. Feature 1 has the same spread.

**axis=0 combines rows and keeps columns:** `(3, 2) → (2,)`, one result per feature. **axis=1 combines columns:** `(3, 2) → (3,)`, one result per sample. No axis combines all entries. Dividing the sum by `X[0]` would divide by data values instead of the number of samples. Starting with `mu += X[i]` before defining mu fails; rerunning a notebook cell with an old mu can also accidentally reuse a previous total.

In Andrew Ng's anomaly-detection Gaussian parameter estimate, use feature-wise `mu` and variance `var` (often named `sigma2`), both shape `(n,)`. The lab's divisor is m: NumPy's default **ddof=0**, made explicit above. `ddof=1` divides by m−1 and gives a different sample-variance estimate. Variance is not standard deviation: `np.std(X, axis=0)` is the square root of variance. These estimates alone are not a complete anomaly detector. See the official [mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html) and [variance](https://numpy.org/doc/stable/reference/generated/numpy.var.html) references.

### NumPy for the specialization: quick lookup

Use this table for recall; the examples above and [K-means walkthrough](UNSUPERVISED_LEARNING_AND_KMEANS.md) explain the steps. X is a 2-D numeric array with shape (m, n); idx is an integer NumPy array of shape (m,) holding cluster IDs.

| Need | Expression | Meaning / result |
| --- | --- | --- |
| Create data | `np.array([[1., 2.], [3., 4.], [5., 6.]])` | Three samples, two features |
| Count versus values | `X.shape[0]` / `X[0]` | Row count m / first complete row |
| Feature count | `X.shape[1]`; `m, n = X.shape` | n columns; unpack both counts |
| Select row / column | `X[i]` / `X[:, j]` | Shape (n,) / (m,); zero-based |
| Select a cluster | `X[idx == i]` | All rows whose cluster ID equals i |
| Initialize storage | `np.zeros(n)`; `np.zeros((K, n))` | Float vector / matrix; integer IDs need `dtype=int` |
| Add / average | `np.sum(X, axis=0)`; `np.mean(X, axis=0)` | n feature sums / means |
| Population variance | `np.var(X, axis=0, ddof=0)` | n mean squared deviations, divisor m |
| Point–center distance | `np.linalg.norm(X[i] - centroids[j])` | One scalar; subtract vectors first |
| Value versus index | `np.min(d)` / `np.argmin(d)` | Smallest distance / its zero-based position |
| Rearrange values | `np.reshape(X, (2, 3))` | Same six values in two rows; also `X.reshape(2, 3)` |
| Shuffle row indices | `randidx = np.random.permutation(m)` | Random ordering of 0 through m−1 |
| Choose K starting rows | `X[randidx[:K]]` | Shape (K, n); require 1 ≤ K ≤ m |
| Replace IDs with centers | `centroids[idx, :]` | Shape (m, n); repeated IDs repeat rows |
| Vector arithmetic | `X * 2`; `(X - mu) ** 2` | Elementwise; (n,) mu broadcasts across rows |
| Grow a Python list | `distances.append(distance)` | Add one item; modifies list and returns None |

## 4. NumPy ML calculations

### Broadcasting: line shapes up from the right

Two dimensions are compatible if equal or one is 1; missing leading dimensions behave like 1. NumPy conceptually stretches the size-1 dimension. It does not mean every pair of shapes is compatible.

```python
import numpy as np

X = np.array([[1., 2., 3.], [4., 5., 6.]])  # (2, 3)
b = np.array([10., 20., 30.])               #    (3,)
print(X + b)  # [[11,22,33], [14,25,36]]; b added to every row
row_means = X.mean(axis=1, keepdims=True)   # (2, 1)
print(X - row_means)  # [[-1,0,1], [-1,0,1]]; center each row

y = np.array([1., 2., 3.])
yhat = np.array([[1.], [2.], [3.]])
assert (yhat - y).shape == (3, 3)  # BUG for per-example errors!
assert (yhat.ravel() - y).shape == (3,)
```

An incompatible `(2, 3) + (2,)` raises an error; a compatible but unintended `(m, 1) - (m,)` silently creates `(m, m)`. Print and assert shapes before computing losses.

### Elementwise multiplication, dot product, and matrix multiplication

```python
import numpy as np

x = np.array([2., 3., 4.])
w = np.array([10., 20., 30.])
print(x * w)            # [20. 60. 120.]: three separate products
print(np.dot(x, w))     # 200.0: sum of those products
print(x @ w)            # 200.0: same for two 1-D vectors

X = np.array([[2., 3., 4.], [1., 1., 1.]])  # (2, 3)
print(X @ w + 1)         # [201. 61.]; shape (2,)
W = np.ones((3, 4))      # three inputs, four neurons
b = np.zeros(4)
Z = X @ W + b            # (2,3) @ (3,4) + (4,) -> (2,4)
assert Z.shape == (2, 4)
```

`*` is elementwise; `@`/`np.matmul` multiplies matrices. Inner dimensions must match. `np.dot` matches `@` for the vector/matrix cases above, but their higher-dimensional rules differ; use `@` for batched matrix multiplication. A dot of two vectors is scalar; NumPy does not automatically preserve a row/column axis.

### Vectorized prediction, MSE, gradients, and train-only scaling

```python
import numpy as np

X = np.array([[1., 10.], [2., 20.], [3., 30.]])
y = np.array([2., 4., 6.])
w = np.array([0.5, 0.1])
b = 0.0
m, n = X.shape

loop_predictions = np.array([np.dot(X[i], w) + b for i in range(m)])
predictions = X @ w + b              # whole dataset, same operation
assert np.allclose(predictions, loop_predictions)
assert predictions.shape == y.shape
error = predictions - y
mse = np.mean(error ** 2)            # sklearn convention
course_cost = mse / 2                # course's 1/(2m) convention
dj_dw = X.T @ error / m              # gradient of course_cost; shape (n,)
dj_db = np.mean(error)               # scalar gradient for bias
w = w - 0.001 * dj_dw                # one update, not a full training loop
b = b - 0.001 * dj_db
assert dj_dw.shape == (n,)
assert np.isclose(mse, 3.5 / 3)
assert np.allclose(dj_dw, [-7 / 3, -70 / 3])
assert np.isclose(dj_db, -1.0)

mu = X.mean(axis=0)                  # TRAINING statistics; shape (2,)
sigma = X.std(axis=0)
safe_sigma = np.where(sigma == 0, 1.0, sigma)  # constant column -> scale 1
X_scaled = (X - mu) / safe_sigma
X_new = np.array([[4., 40.]])
X_new_scaled = (X_new - mu) / safe_sigma  # reuse training ruler
assert np.allclose(X_scaled.mean(axis=0), 0.0)
assert X_new_scaled.shape == (1, 2)
```

Vectorization expresses work as array operations and avoids Python loops over every row. It often runs faster; it still allocates memory. `np.vectorize` is a convenience wrapper, generally not the same performance improvement. The learning rate `0.001` above controls the update size; it is not regularization strength.

### Small index and distance example

```python
import numpy as np

X = np.array([[1., 2.], [3., 4.], [5., 6.]])
centroids = np.array([[1., 2.], [4., 5.]])
idx = np.array([0, 1, 1], dtype=int)
i = 1                           # here i is a cluster ID
points = X[idx == i]             # mask [False, True, True]
np.testing.assert_array_equal(points, [[3., 4.], [5., 6.]])
np.testing.assert_allclose(np.mean(points, axis=0), [4., 5.])

distances = []                  # Python list: reset for each sample
for j in range(centroids.shape[0]):
    distance = np.linalg.norm(X[2] - centroids[j])
    distances.append(distance)
np.testing.assert_allclose(distances, [np.sqrt(32), np.sqrt(2)])
assert np.isclose(np.min(distances), np.sqrt(2))  # distance VALUE
assert np.argmin(distances) == 1                  # center INDEX
np.testing.assert_array_equal(centroids[idx, :],
                              [[1., 2.], [4., 5.], [4., 5.]])

K = 2
randidx = np.random.permutation(X.shape[0])
starting = X[randidx[:K]]
assert sorted(randidx.tolist()) == [0, 1, 2]
assert starting.shape == (K, X.shape[1])
np.testing.assert_array_equal(np.reshape(X, (2, 3)),
                              [[1., 2., 3.], [4., 5., 6.]])
```

`idx == i` compares every assignment; `idx[i]` selects one assignment. `X[idx == i]` filters samples; `centroids[idx, :]` retrieves the center named by each integer ID. In the distance calculation, `X[2]` selects row 2, and argmin returns center 1. These indices refer to different arrays. `np.min` returns a value that cannot serve as a center ID. `append` stores each distance without adding them together; do not write `distances = distances.append(distance)`.

The random order varies between runs; it avoids repeating row indices, though duplicate row values may exist. For new code, section 2's dedicated `rng.permutation(m)` is also available. Full assignment/update loops, empty-group handling, initialization and RGB reconstruction remain in the [existing K-means note](UNSUPERVISED_LEARNING_AND_KMEANS.md#5a-random-starting-rows).

### Numerical stability — intermediate practice

```python
import numpy as np

def sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    # logaddexp(0, -z) evaluates log(1 + exp(-z)) stably.
    return np.exp(-np.logaddexp(0.0, -z))

def softmax_rows(z):
    z = np.asarray(z, dtype=np.float64)
    shifted = z - z.max(axis=-1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)

print(sigmoid([-1000., 0., 1000.]))  # [0. 0.5 1.]; finite, no overflow
probabilities = softmax_rows([[1000., 1001., 1002.]])
assert probabilities.shape == (1, 3)
assert np.allclose(probabilities.sum(axis=1), 1.0)

y = np.array([0., 1., 1.])
z = np.array([-1000., 0., 1000.])
bce = np.mean(np.logaddexp(0.0, z) - y * z)  # binary loss from logits
assert np.isfinite(bce)
assert np.all(np.isfinite(probabilities))
assert np.allclose([0.1 + 0.2], [0.3])  # floats: approximate equality
```

Avoid direct `exp(1000)` and `log(0)`. Subtracting the row maximum preserves softmax while preventing exponential overflow for finite logits. If manually taking logs of probabilities, clipping to `[eps, 1-eps]` avoids `log(0)` but changes extreme values; use framework cross-entropy from logits for training. Do not fix every NaN with clipping: inspect missing inputs, division by zero, overflow, and learning rate first. `np.nanmean` ignores NaNs intentionally; ordinary `mean` propagates them.

## 5. Matplotlib pyplot

`figure` is the whole canvas; an `Axes` is one plotting panel. `plt` functions use the current panel; `ax` methods select a panel explicitly.

### Quick scatter and line overlay

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1., 2., 3., 4.])
y = np.array([2.2, 3.8, 6.1, 7.9])
order = np.argsort(x)
plt.figure(figsize=(6, 4))                   # width, height in inches
plt.scatter(x, y, c="tab:blue", marker="x", label="observations")
plt.plot(x[order], 2 * x[order], "r--", label="prediction")
plt.title("A tiny regression example")
plt.xlabel("Input feature")
plt.ylabel("Target")
plt.xlim(0, 5)
plt.ylim(0, 10)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("regression.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
```

`scatter` draws independent points; `plot` connects points in the given order. Sort x before connecting a fitted curve or it may zigzag. `label=` supplies legend text; call `legend()` to display it. Save before `show()` because some backends close/reset figures on display. The file is saved relative to the current working directory. See the [savefig API](https://matplotlib.org/3.11.0/api/_as_gen/matplotlib.pyplot.savefig.html) for format and resolution options.

### Histograms, subplots, and a training-history recipe

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
values = rng.normal(size=200)
fig, axes = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
axes[0].hist(values, bins=15, edgecolor="white")
axes[0].set(title="Feature distribution", xlabel="Value", ylabel="Count")

train_loss = [0.9, 0.5, 0.3, 0.2]
cv_loss = [1.0, 0.6, 0.5, 0.55]
epochs = np.arange(1, len(train_loss) + 1)
axes[1].plot(epochs, train_loss, label="train")
axes[1].plot(epochs, cv_loss, label="CV")
axes[1].set(title="Learning progress", xlabel="Epoch", ylabel="Loss")
axes[1].legend()
fig.savefig("diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)
```

`hist` groups values into bins; default heights are counts, while `density=True` makes total bar **area** one. `plt.subplots()` returns `(fig, ax)`; for one panel `ax` is a single Axes, for multiple panels it is usually an array. `plt.subplots(2, 2)` needs `axes[row, column]`. Use `squeeze=False` for a consistently 2-D axes array. For Keras history, substitute `history.history["loss"]` and `["val_loss"]`; the latter exists when validation data was supplied.

## 6. TensorFlow and Keras foundations

### Tensors, variables, and shapes

```python
import numpy as np
import tensorflow as tf

X = tf.constant([[1., 2.], [3., 4.]], dtype=tf.float32)
W = tf.constant([[1.], [2.]], dtype=tf.float32)
b = tf.Variable([0.], dtype=tf.float32)  # mutable, tracked for gradients
Z = tf.matmul(X, W) + b
print(Z.numpy())              # [[5.], [11.]] in eager execution
print(X.shape, tf.shape(X))   # static shape (2, 2), runtime shape tensor [2, 2]
print(tf.reshape(X, (-1, 1)).shape)  # (4, 1)
print(tf.reduce_mean(X, axis=0).numpy())  # [2. 3.]
assert Z.shape == (2, 1)
assert np.allclose(Z.numpy(), [[5.], [11.]])
```

A tensor is numerical data; a Variable supports updates such as `.assign(...)`. A plain constant is not automatically watched by GradientTape. `.numpy()` is for eager-mode inspection, not for calculations inside a differentiable TensorFlow training path. Converting to NumPy there breaks gradient tracking. Keep dtype consistent, commonly float32 for neural-network inputs.

### Dense and Sequential: connect to the digit lab

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.Input(shape=(400,)),       # one flattened 20x20 image
    tf.keras.layers.Dense(25, activation="sigmoid"),
    tf.keras.layers.Dense(15, activation="sigmoid"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.summary()
output = model(tf.zeros((2, 400)), training=False)
W1, b1 = model.layers[0].get_weights()
assert W1.shape == (400, 25) and b1.shape == (25,)
assert output.shape == (2, 1)
assert model.count_params() == 10421  # 10025 + 390 + 16
```

`Dense(25)` means **25 outputneurons**, each receiving all 400 input features. It computes `activation(X @ W + b)`. `Sequential` passes one layer's output into the next. The same weights serve every image. This untrained model performs forward propagation; it has not learned digit recognition. `get_weights()` inspects parameters, not training progress by itself.

| Layer | Input | Weights | Bias | Output |
| --- | --- | --- | --- | --- |
| Dense 25 | `(m, 400)` | `(400, 25)` | `(25,)` | `(m, 25)` |
| Dense 15 | `(m, 25)` | `(25, 15)` | `(15,)` | `(m, 15)` |
| Dense 1 | `(m, 15)` | `(15, 1)` | `(1,)` | `(m, 1)` |

One image has one target label; the 400 pixels are features, not 400 labels. The hidden sigmoid activations above match the early lab; ReLU is another common hidden activation, introduced in the next section.

## 7. Outputs, losses, and training

### Match the final layer, target encoding, and loss

| Task | Final Dense | Target shape/type | Loss | Decision |
| --- | --- | --- | --- | --- |
| Binary probability | `Dense(1, activation="sigmoid")` | `(m, 1)`, 0/1 | `BinaryCrossentropy(from_logits=False)` | probability ≥ 0.5 |
| Binary logits | `Dense(1)` | `(m, 1)`, 0/1 | `BinaryCrossentropy(from_logits=True)` | sigmoid, then ≥ 0.5; equivalently logit ≥ 0 |
| Multiclass logits, one class per row | `Dense(k)` | `(m,)`, integer 0..k-1 | `SparseCategoricalCrossentropy(from_logits=True)` | `argmax(axis=1)` |
| Multiclass probabilities | `Dense(k, activation="softmax")` | `(m,)`, integer 0..k-1 | `SparseCategoricalCrossentropy(from_logits=False)` | `argmax(axis=1)` |
| Multiclass one-hot labels | `Dense(k)` | `(m, k)`, one-hot | `CategoricalCrossentropy(from_logits=True)` | `argmax(axis=1)` |
| Regression, one continuous target | `Dense(1)` | `(m, 1)`, float | `MeanSquaredError()` or `"mse"` | raw numeric value |

**Logit** = unconstrained score before sigmoid/softmax. **Sigmoid** maps each score independently to 0–1. **Softmax** converts a row of competing class scores into probabilities summing to 1. **ReLU** (`activation="relu"`) computes `max(0, z)` and is common in hidden layers. **Linear** leaves z unchanged; it is the default Dense activation.

Do not set `from_logits=True` on an already-sigmoid/softmax output. Do not sigmoid a probability a second time. One-output softmax is always 1 and is unsuitable for binary classification. Multiple independent labels are a different task: typically k sigmoid outputs with binary cross-entropy, not a one-of-k softmax.

Cross-entropy from logits lets the framework combine operations stably; `from_logits` describes what the **model outputs**, not whether labels are integers. See [BinaryCrossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy).

### What the lifecycle methods do

| Call | Meaning | Return value |
| --- | --- | --- |
| `model.compile(...)` | Configure loss, optimizer, metrics | `None`; no learning yet |
| `model.fit(X_train, y_train, ...)` | Forward → loss → gradients → parameter updates | `History`, including `.history` dict |
| `model.evaluate(X_test, y_test, return_dict=True)` | Compute configured loss/metrics without training | Dict of numbers |
| `model.predict(X_new, verbose=0)` | Inference on batches, no parameter updates | Usually a NumPy array for these models |
| `model(X_batch, training=False)` | Direct inference call, useful for small batches | Tensor for these models |

An epoch is one pass over training examples. `batch_size=32` usually means an update per group of up to 32 rows. `epochs=10` is not ten individual weight updates. Adam uses gradients plus moving estimates of their moments to choose updates; `learning_rate` controls step scale. Backpropagation calculates gradients; the optimizer applies updates. Accuracy is a metric, not the cross-entropy objective.

For logit-output binary models, use `BinaryAccuracy(threshold=0.0)` if computing accuracy directly on logits; its default 0.5 threshold assumes probabilities. For sparse multiclass logits, `SparseCategoricalAccuracy()` uses the highest-scoring class.

## 8. TensorFlow task recipes

These are **intermediate practice extensions** using generated data. They demonstrate executable workflows, not measured course performance. Seeded training may still vary by platform. No accuracy claim relies on a particular run. Use a fresh model and optimizer for each candidate configuration.

### Binary classifier: split, scale, train logits, evaluate, threshold

```python
import numpy as np
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.keras.utils.set_random_seed(42)
X, y = make_classification(n_samples=300, n_features=4, n_informative=3,
                           n_redundant=0, random_state=42)
X_train, X_hold, y_train, y_hold = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y)
X_cv, X_test, y_cv, y_test = train_test_split(
    X_hold, y_hold, test_size=0.5, random_state=42, stratify=y_hold)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype("float32")
X_cv_s = scaler.transform(X_cv).astype("float32")
X_test_s = scaler.transform(X_test).astype("float32")
y_train = y_train.astype("float32").reshape(-1, 1)
y_cv = y_cv.astype("float32").reshape(-1, 1)
y_test = y_test.astype("float32").reshape(-1, 1)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(4,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1),  # raw logits
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.0)],
)
history = model.fit(X_train_s, y_train, validation_data=(X_cv_s, y_cv),
                    epochs=10, batch_size=32, verbose=0)
# Freeze configuration and the 0.5 probability threshold before test evaluation.
print(model.evaluate(X_test_s, y_test, return_dict=True, verbose=0))
logits = model.predict(X_test_s, verbose=0)  # (60, 1)
p = tf.math.sigmoid(logits).numpy()        # (60, 1)
labels = (p >= 0.5).astype(int)            # (60, 1)
assert labels.shape == y_test.shape == (60, 1)
classification_error = np.mean(labels != y_test)
print(f"Test classification error: {classification_error:.1%}")
assert len(history.history["loss"]) == 10
```

The first split holds out 40%; half of that is CV and half test, giving **60/20/20**. Scaling learns only from train. `.reshape(-1, 1)` makes labels match the model's `(m, 1)` output. A probability threshold 0.5 corresponds to logit 0. Tune thresholds on CV when needed, never on the final test set. An ordinary classifier score is not automatically a calibrated probability.

### Multiclass: integer labels, softmax, argmax

```python
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.keras.utils.set_random_seed(42)
X, y = load_iris(return_X_y=True)  # (150, 4); y is (150,), values 0,1,2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype("float32")
X_test_s = scaler.transform(X_test).astype("float32")
model = tf.keras.Sequential([
    tf.keras.Input(shape=(4,)),
    tf.keras.layers.Dense(12, activation="relu"),
    tf.keras.layers.Dense(3),  # one logit per class
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
model.fit(X_train_s, y_train, epochs=10, batch_size=16, verbose=0)
print(model.evaluate(X_test_s, y_test, return_dict=True, verbose=0))
logits = model.predict(X_test_s, verbose=0)
probabilities = tf.nn.softmax(logits, axis=-1).numpy()
labels = np.argmax(probabilities, axis=1)
assert probabilities.shape == (30, 3) and labels.shape == (30,)
assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
assert np.array_equal(labels, logits.argmax(axis=1))
```

This fixed-configuration demo has no hyperparameter search. If choosing architectures/epochs, first create a separate CV set or use cross-validation. Sparse targets are class IDs, not one-hot rows. For one-hot targets use `tf.one_hot(y, depth=3)` and `CategoricalCrossentropy`; do not mix the encodings.

### Regression: predict a continuous value

```python
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.keras.utils.set_random_seed(42)
rng = np.random.default_rng(42)
X = rng.normal(size=(200, 2)).astype("float32")
y = (2 * X[:, 0] - X[:, 1] + 0.5).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype("float32")
X_test_s = scaler.transform(X_test).astype("float32")
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(1),  # no sigmoid: targets may be negative or >1
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
              loss="mse", metrics=[tf.keras.metrics.MeanAbsoluteError()])
model.fit(X_train_s, y_train, epochs=20, batch_size=32, verbose=0)
print(model.evaluate(X_test_s, y_test, return_dict=True, verbose=0))
predictions = model.predict(X_test_s, verbose=0)
assert predictions.shape == y_test.shape == (40, 1)
print(predictions[:3])  # learned continuous values; exact values vary
```

Again this is a fixed configuration; add CV before tuning. Regression uses MSE/MAE, not thresholding or classification accuracy.

## 9. GradientTape

**Intermediate bridge to the computation-graph lab.** Tape records differentiable TensorFlow operations, computes gradients, then the optimizer changes variables. See the [GradientTape API](https://www.tensorflow.org/api_docs/python/tf/GradientTape).

```python
import numpy as np
import tensorflow as tf

w = tf.Variable(3.0)
with tf.GradientTape() as tape:
    J = (2.0 + 3.0 * w) ** 2
gradient = tape.gradient(J, w)
assert np.isclose(J.numpy(), 121.0)
assert np.isclose(gradient.numpy(), 66.0)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
optimizer.apply_gradients([(gradient, w)])
assert np.isclose(w.numpy(), 2.34)  # 3 - 0.01 * 66
```

`with ... as tape:` records the forward computation. `tape.gradient(J, w)` calculates dJ/dw; that line alone does not update w. A normal tape can be consumed once. Use `tape.watch(x)` for a floating constant you want to differentiate with respect to. A `None` gradient often means a disconnected computation, an unwatched input, or a NumPy operation that left the TensorFlow graph.

For a model, the equivalent pattern is `tape.gradient(loss, model.trainable_variables)`, followed by `optimizer.apply_gradients(zip(grads, model.trainable_variables))`. Compute the forward pass and loss **inside** the tape. When manually training with regularized layers, include `sum(model.losses)` in the objective if nonempty; `model.fit` handles those regularization losses automatically. Adam can replace SGD in the update, but one Adam step will not equal the simple numeric SGD update above.

## 10. sklearn preprocessing and evaluation

### Fit learns; transform reuses; predict guesses

| Object | `fit` learns | `transform` produces |
| --- | --- | --- |
| `StandardScaler` | training column mean/scale | standardized features |
| `PolynomialFeatures` | feature mapping structure | polynomial/interaction columns |
| `SimpleImputer` | e.g. training medians | features with missing entries filled |
| Prediction estimator | weights or other predictive state, using y | usually has `predict`, not `transform` |

sklearn `fit` usually returns the estimator itself; Keras `fit` returns a History. `fit_transform` fits **and** transforms. Do not mistake either `fit` for “give me transformed data.”

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

X_train = np.array([[10.], [20.], [30.]])
X_cv = np.array([[40.], [50.]])
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)  # learns train mean 20
X_cv_s = scaler.transform(X_cv)           # reuses train mean/scale
assert np.allclose(scaler.mean_, [20.])
assert np.allclose(scaler.scale_, [np.sqrt(200 / 3)])
assert np.allclose(X_cv_s.ravel(), [2.449489743, 3.674234614])

poly = PolynomialFeatures(degree=2, include_bias=False)
mapped = poly.fit_transform([[2.], [3.]])
assert np.array_equal(mapped, [[2., 4.], [3., 9.]])
```

Calling `scaler.fit(X_cv)` again **overwrites** the saved training ruler (mean becomes 45, scale 5); it does not maintain two rulers. `transform` leaves it intact. Default StandardScaler uses population std and a scale of 1 for constant columns. Scaling is not restricted to 0–1. CV columns need not have mean zero. See the existing [fit/transform deep dive](FIT_TRANSFORM_PREDICT_DEEP_DIVE.md).

### Complete train/CV/test recipe: choose polynomial degree

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(42)
X = np.linspace(-3, 3, 150).reshape(-1, 1)
y = 1 + 2 * X[:, 0] + X[:, 0] ** 2 + rng.normal(0, 0.5, 150)
X_train, X_hold, y_train, y_hold = train_test_split(
    X, y, test_size=0.4, random_state=42)
X_cv, X_test, y_cv, y_test = train_test_split(
    X_hold, y_hold, test_size=0.5, random_state=42)
assert (len(X_train), len(X_cv), len(X_test)) == (90, 30, 30)

candidates = []
for degree in [1, 2, 3, 6]:
    candidate = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=0.1)),
    ])
    candidate.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, candidate.predict(X_train))
    cv_mse = mean_squared_error(y_cv, candidate.predict(X_cv))
    candidates.append((degree, train_mse, cv_mse, candidate))

degree, train_mse, cv_mse, best = min(candidates, key=lambda row: row[2])
print(f"degree={degree}, train MSE={train_mse:.3f}, CV MSE={cv_mse:.3f}")
test_mse = mean_squared_error(y_test, best.predict(X_test))
print(f"Final test MSE={test_mse:.3f}")
new_predictions = best.predict([[1.5]])  # raw input; pipeline handles preparation
assert new_predictions.shape == (1,)
```

`Pipeline` keeps the mapping, fitted scaler, and fitted model together. `fit` fits preprocessing and then the predictor. `predict` transforms using saved state before prediction. Never pre-scale input to a pipeline that already scales. `include_bias=False` omits the constant polynomial column; Ridge still fits its intercept by default. `make_pipeline(...)` is a shorter variant with automatically generated step names.

Train fits parameters. CV/validation chooses degree, λ, architecture, epochs, and threshold. Test estimates performance of the frozen choice. “CV set” in the course often means one held-out development set; k-fold cross-validation instead rotates several training/validation folds.

For a final refit on train+CV after choosing settings, fit a **new complete pipeline** on that combined data, then evaluate once on untouched test data. This is a deliberate new training stage; it differs from accidentally refitting only the old scaler on CV. Do not keep changing settings after seeing test results.

### K-fold selection with preprocessing inside every fold — intermediate

```python
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(42)
X = np.linspace(-2, 2, 120).reshape(-1, 1)
y = X[:, 0] ** 2 + rng.normal(0, 0.2, len(X))
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scale", StandardScaler()),
    ("model", Ridge()),
])
search = GridSearchCV(
    pipeline,
    param_grid={"poly__degree": [1, 2, 3], "model__alpha": [0.01, 1.0, 100.0]},
    scoring="neg_mean_squared_error",
    cv=KFold(n_splits=3, shuffle=True, random_state=42),
    refit=True,
)
search.fit(X_dev, y_dev)
print(search.best_params_, -search.best_score_)  # positive mean validation MSE
print(mean_squared_error(y_test, search.predict(X_test)))
assert search.best_estimator_.predict([[1.0]]).shape == (1,)
```

`step__parameter` addresses a parameter inside a pipeline. sklearn scores conventionally reward **larger** numbers; negate negative MSE to report ordinary MSE. `refit=True` retrains the chosen pipeline on all `X_dev` after selection. Each fold gets its own fitted preprocessing. Fitting a scaler on all development data before k-fold CV leaks validation information. See sklearn's [leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

Random row splits assume appropriate independence. Use time-based splits for forecasting and group splits when rows share people/devices/sessions. For classification, stratification helps preserve class proportions but does not solve group or temporal leakage.

### Classification metrics: labels versus probabilities

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score, roc_auc_score, log_loss)

X, y = make_classification(n_samples=200, n_features=4, n_informative=3,
                           n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
labels = model.predict(X_test)         # sklearn returns class labels here
probabilities = model.predict_proba(X_test)
positive_column = list(model.classes_).index(1)
p_positive = probabilities[:, positive_column]
print("accuracy", accuracy_score(y_test, labels))
print("error", np.mean(labels != y_test))  # 1 - accuracy
print(confusion_matrix(y_test, labels, labels=[0, 1]))  # rows true, columns predicted
print("precision", precision_score(y_test, labels, zero_division=0))
print("recall", recall_score(y_test, labels, zero_division=0))
print("F1", f1_score(y_test, labels, zero_division=0))
print("ROC AUC", roc_auc_score(y_test, p_positive))
print("log loss", log_loss(y_test, probabilities, labels=model.classes_))
assert labels.shape == (40,) and probabilities.shape == (40, 2)
```

For binary labels `[0,1]`, confusion matrix is `[[TN, FP], [FN, TP]]`. Precision asks “of positive predictions, how many were right?” Recall asks “of actual positives, how many did we find?” F1 balances them. ROC AUC takes scores/probabilities, not thresholded labels, and requires both classes in the evaluated data. Accuracy alone can hide failures on imbalanced classes. Log loss evaluates probability predictions; it is not misclassification error.

## 11. Bias, variance, and regularization

### Diagnose before changing syntax

Think of train as practice questions and CV as fresh questions. Compare the **same metric** on train, CV, and a credible achievable baseline.

| Observation | Possible diagnosis | Controlled experiment |
| --- | --- | --- |
| Train error much worse than baseline | High bias / underfitting | Better features, more capacity, weaker regularization, check optimization |
| Train error low, CV much worse | High variance / overfitting | More representative data, stronger regularization, less capacity |
| Both problems appear | Bias and variance can coexist | Address one factor, measure both errors again |
| Both errors acceptable | Promising generalization | Confirm using untouched test data |

These are clues, not universal numerical cutoffs. Distribution shift, bugs, and bad labels can also cause gaps. Do not compare regularized training loss with unregularized CV MSE and call the difference variance; calculate comparable prediction metrics on both sets.

### λ is the regularization brake

**λ up → stronger penalty → often less variance, more bias. λ down → weaker penalty → often less bias, more variance. CV decides the useful strength.**

Course-style linear regression objective:

```text
J = sum((prediction - y)**2)/(2*m) + lambda_ * sum(w**2)/(2*m)
```

The intercept b is usually not penalized. λ changes what the optimizer minimizes; it is not a hard cap or a command to multiply weights by λ.

```python
import numpy as np

w = np.array([10., 2.])
m = 10
for lambda_ in [0.0, 1.0, 10.0]:
    penalty = lambda_ * np.sum(w ** 2) / (2 * m)
    print(f"lambda={lambda_:4.1f}, penalty={penalty:.1f}")
# penalties: 0.0, 5.2, 52.0, for these FIXED weights
```

In actual training the learned weights change too. More data can help variance; it usually does not rescue a model that cannot fit the training relationship.

### Tune Ridge strength on CV — quick-copy continuation

Continue the **complete train/CV/test polynomial recipe** from section 10. This is an alternative selection exercise: use fresh splits for a real new experiment if test results have already influenced decisions. No test score is used inside this sweep.

```python
results = []
for strength in [0.01, 0.1, 1.0, 10.0, 100.0]:
    candidate = Pipeline([
        ("poly", PolynomialFeatures(degree=3, include_bias=False)),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=strength)),
    ])
    candidate.fit(X_train, y_train)
    train_error = mean_squared_error(y_train, candidate.predict(X_train))
    cv_error = mean_squared_error(y_cv, candidate.predict(X_cv))
    results.append((strength, train_error, cv_error, candidate))
best_strength, train_error, cv_error, selected = min(results, key=lambda row: row[2])
print(f"Ridge alpha={best_strength}, train={train_error:.3f}, CV={cv_error:.3f}")
```

**Naming traps:** course gradient-descent α = learning rate; `Ridge(alpha=...)` = regularization strength. LogisticRegression uses `C`, whose direction is reversed: smaller C means stronger regularization. Python `lambda` = function syntax, unrelated to the numeric hyperparameter `lambda_`.

Ridge minimizes squared-error **sum** plus `alpha * sum(w²)`, so its alpha is not interchangeable with an arbitrary averaged-loss λ convention. See the [Ridge objective](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

### Keras regularization and normalization — intermediate

```python
import numpy as np
import tensorflow as tf

X_train = np.array([[1., 10.], [2., 20.], [3., 30.]], dtype="float32")
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(X_train)  # TRAIN only: learns statistics; does not train predictor
strength = 0.01
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    normalizer,
    tf.keras.layers.Dense(8, activation="relu",
                          kernel_regularizer=tf.keras.regularizers.L2(strength)),
    tf.keras.layers.Dense(1),
])
output = model(tf.constant([[4., 40.]]), training=False)
assert output.shape == (1, 1)
assert len(model.losses) == 1  # the hidden layer's L2 penalty
```

The Normalization layer is now part of the model, so pass raw features at prediction time. `adapt(X_cv)` would replace its ruler. **Normalization scales inputs; regularization penalizes weights.** This example regularizes the hidden kernel only; bias is not penalized unless `bias_regularizer` is supplied. Keras L2 adds `strength * sum(kernel²)`; it does not automatically insert the course's `1/(2m)` factor. See [Keras regularizers](https://keras.io/api/layers/regularizers/). AdamW's decoupled weight decay is related but is not generally the same update as adding L2 loss to Adam.

### Learning curves — intermediate quick-copy recipe

```python
import numpy as np
from sklearn.model_selection import learning_curve, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge

rng = np.random.default_rng(42)
X_train = rng.uniform(-2, 2, size=(100, 1))
y_train = X_train[:, 0] ** 2 + rng.normal(0, 0.2, 100)
sizes, train_scores, cv_scores = learning_curve(
    make_pipeline(PolynomialFeatures(degree=2, include_bias=False),
                  StandardScaler(), Ridge(alpha=0.1)),
    X_train, y_train,
    train_sizes=[0.3, 0.6, 1.0],
    cv=KFold(n_splits=3, shuffle=True, random_state=42),
    scoring="neg_mean_squared_error", shuffle=True, random_state=42,
)
train_mse = -train_scores.mean(axis=1)
cv_mse = -cv_scores.mean(axis=1)
print(list(zip(sizes, train_mse, cv_mse)))
assert train_scores.shape == cv_scores.shape == (3, 3)
```

This uses only supplied training data, with internal folds. Inspect whether the train–CV gap shrinks with more training rows. Plot sizes against both MSE arrays using `plt.plot`. More data, λ, and architecture are experiments, not guaranteed cures. Bias/variance matters in fine-tuning too, but LLM failures can also come from retrieval, context, data, serving, and evaluation design.

## 12. Debugging and copy recipes

### Tiny shape and prediction checks

```python
import numpy as np

X = np.array([[1., 2.], [3., 4.]])
y = np.array([0, 1])
probabilities = np.array([[0.2], [0.8]])
assert X.ndim == 2
assert len(X) == len(y)
assert np.isfinite(X).all()
labels = (probabilities.ravel() >= 0.5).astype(int)
assert labels.shape == y.shape
print(f"error={np.mean(labels != y):.1%}")  # error=0.0%
one_new_example = np.array([[5., 6.]])      # (1, 2)
many_one_feature_examples = np.array([5., 6.]).reshape(-1, 1)  # (2, 1)
assert one_new_example.shape == (1, 2)
assert many_one_feature_examples.shape == (2, 1)
```

Ravel is appropriate here because each example has exactly one label/output; flattening a multiclass probability matrix would destroy its class structure.

| Symptom | Likely issue / next check |
| --- | --- |
| `NameError` after copying | Missing import or ignored “Continue” dependency; restart and run in order |
| `ModuleNotFoundError` despite installation | Wrong notebook kernel/interpreter; inspect `sys.executable` |
| One example rejected by sklearn/Keras | Use `(1, n)`, not `(n,)` |
| Loss/error array becomes `(m, m)` | Target `(m,)` versus prediction `(m, 1)` broadcast |
| `matmul` dimension mismatch | Print X/W shapes; `(m,n) @ (n,k)` |
| Unexpected source array change | A slice shared storage; use `.copy()` |
| “Truth value of an array is ambiguous” | Use masks with `&`/`|`, or explicit `.any()`/`.all()` |
| CV looks suspiciously perfect | Leakage, duplicates, wrong splits, fitting before splitting |
| New predictions look wrong | Wrong feature order, stale/mismatched scaler, double scaling |
| Classification accuracy wrong for logits | Threshold raw binary logits at 0, not 0.5 |
| Binary model always outputs 1 | Check for a one-unit softmax |
| Loss NaN / infinite | Inspect finite inputs, dtypes, divisions, learning rate, stable loss |
| Good training fit, poor CV | Compare same metrics; inspect variance, shift, and data quality |
| Plot line crosses back and forth | Sort x and apply same index order to y |
| Empty saved figure | Save before display/close; save the intended figure |

**Code-reading order:** inputs → shapes → operation/loop → prediction → comparison with y → accumulated quantity → return value. Translate the important line into plain language before changing it.

## 12a. Decision trees — concise syntax

For entropy calculations, lab-style helper functions, split selection, overfitting and a full CV example, see [Decision Trees — detailed study note](DECISION_TREES.md). It reproduces the supplied original ten-row Andrew Ng ungraded lab with verified calculations, separately from the illustrative example and safer reusable helpers.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2,
    stratify=data.target, random_state=42,
)
tree = DecisionTreeClassifier(
    criterion="entropy", max_depth=3,
    min_samples_leaf=2, random_state=42,
)
tree.fit(X_train, y_train)
labels = tree.predict(X_test)            # (30,)
probabilities = tree.predict_proba(X_test)  # (30, 3)
print(tree.classes_)                    # [0 1 2]; probability-column order
print(tree.score(X_test, y_test))        # accuracy; no fixed score promised
print(export_text(tree, feature_names=list(data.feature_names)))
assert labels.shape == (30,)
assert probabilities.shape == (30, 3)
one_label = tree.predict(X_test[:1])     # (1, 4) input -> (1,) output
```

fit learns rules; predict follows them; predict_proba estimates class proportions in the reached leaf. Keep X two-dimensional and feature order unchanged. Standalone trees generally need no scaling. Smaller max_depth or larger min_samples_leaf restrict growth; ccp_alpha controls pruning. Select these with training/CV data before final test evaluation. sklearn's default criterion is Gini; entropy is specified here to connect to the study calculation. Original teaching example, not an exact course lab reproduction.

### Graded exercise reminder: row IDs survive every split

The [beginner walkthrough in Decision Trees, section 4a](DECISION_TREES.md#4a-graded-exercise-follow-the-original-row-ids) has the schoolchildren/mushroom analogy, executable entropy and split_dataset functions, nested splits, expected outputs and information-gain calculations. Keep using this canonical sheet for quick lookup.

- `X` stays the full feature matrix; `y` holds aligned labels (in this mushroom exercise: 1 edible, 0 poisonous). A feature's 1 means yes to that feature, not edible.
- `node_indices` contains original row IDs at the **current node**: all at the root, a subset in a child. `feature` selects one column; `X[i, feature]` is a scalar; `X[node_indices]` selects whole rows.
- `split_dataset` loops `for i in node_indices`, sends feature value 1 left and 0 right, and appends **i**. It returns IDs without changing X. Information gain separately uses y to evaluate that question; best-split selection compares candidate gains.
- `p_1` is the fraction of labels equal to 1 in `y[node_indices]`. Pure endpoints 0 and 1 have entropy 0: guard them before log2. Guard an empty node before dividing. Use `p_1` consistently; `p1` is a different variable.

**[What confused me in the graded lab / brain-burner troubleshooting](DECISION_TREES.md#what-confused-me-in-the-graded-lab--brain-burner-troubleshooting):** entropy → split → gain → best feature. `y=1` names the positive class, not a correct prediction. `p_1=sum(y_node)/len(y_node)` counts positives; `w_left/right=len(child)/len(node_indices)` count routing proportions. Parent entropy uses `y[node_indices]`, children use their own labels, and gain subtracts the **weighted** child entropies.

For `get_best_split`, loop over `range(X.shape[1])`, save `ig_i = compute_information_gain(...)`, and compare `ig_i > ig_max`. On improvement update **both** `ig_max = ig_i` and `best_feature = i`. Keep a running maximum, not merely the previous gain; never compare gain with a feature ID. Start with gain 0 and feature -1; return -1 if none has positive gain and check it before indexing. `ig_1` and `ig_i` are different names, just like `p1` and `p_1`.

Tree branch weights are sample fractions; NN connection weights are learned parameters. This lab chooses discrete questions by gain; a standard NN learns weighted activations through loss gradients/backpropagation. The detailed troubleshooting section has the worked weight example, corrected selection loop and self-checks.

Independent NumPy indexing reminder:

```python
import numpy as np

X = np.array([[1, 0], [1, 1], [0, 1], [0, 0], [1, 1], [0, 1]])
node_indices, feature = [2, 3, 5], 1
left_indices, right_indices = [], []
for i in node_indices:
    if X[i, feature] == 1:
        left_indices.append(i)
    elif X[i, feature] == 0:
        right_indices.append(i)
print(left_indices, right_indices)  # [2, 5] [3]
print(X[node_indices].shape)       # (3, 2)
assert left_indices == [2, 5] and right_indices == [3]
```

Avoid `X[i, node_indices]` (row IDs put in the column slot) and `append(X[i, node_indices])` (wrong lookup and wrong thing stored). `range(len(X))` creates root IDs but revisits every row if used inside every child. `range(len(node_indices))` produces positions `[0, 1, 2]`, not current IDs `[2, 3, 5]`; loop over the IDs themselves. Keep original IDs paired with full X/y, rather than applying them to a renumbered subset.

## 12b. Trees Ensemble lab — pandas, forests, and XGBoost

For the beginner walkthrough, full tuning loops, reported results and source corrections, see [the second lab in Decision Trees](DECISION_TREES.md#8a-second-lab-ungraded-lab---trees-ensemble). The earlier ten-animal lab and section 12a remain useful.

**Setup:** install `pandas scikit-learn xgboost` and put the supplied lab's `heart.csv` in the working directory. This block is independent. Its expected shapes require that 918-row file, not an arbitrary heart dataset.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

categorical = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
df = pd.read_csv("heart.csv")
df = pd.get_dummies(df, columns=categorical, prefix=categorical)
features = [name for name in df.columns if name != "HeartDisease"]
X, y = df[features], df["HeartDisease"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, train_size=0.8, random_state=55,
)
assert X.shape == (918, 20) and y.shape == (918,)
assert X_train.shape == (734, 20) and X_val.shape == (184, 20)

tree = DecisionTreeClassifier(max_depth=4, min_samples_split=50, random_state=55)
# A seed is deliberately added here for repeatable practice.
# The source lab's final forest omits it, so exact source scores may vary.
forest = RandomForestClassifier(
    n_estimators=100, max_depth=16, min_samples_split=10,
    max_features="sqrt", n_jobs=1, random_state=55,
)
for name, model in [("Tree", tree), ("Forest", forest)]:
    model.fit(X_train, y_train)
    print(
        f"{name}: train={accuracy_score(y_train, model.predict(X_train)):.4f}, "
        f"validation={accuracy_score(y_val, model.predict(X_val)):.4f}"
    )
```

There are 11 original features: six numeric plus five categorical. Encoding all five (including binary Sex and ExerciseAngina) gives `6 + 2 + 4 + 3 + 2 + 3 = 20` inputs; the target is a separate 21st column. Use exact `!= "HeartDisease"` or `.drop(columns="HeartDisease")`; the lab's `name not in "HeartDisease"` tests substrings rather than exact equality.

**Continue the setup block** to sweep one hyperparameter at a time:

```python
split_candidates = [2, 10, 30, 50, 100, 200, 300, 700]
depth_candidates = [1, 2, 3, 4, 8, 16, 32, 64, None]
for parameter, candidates in [
    ("min_samples_split", split_candidates), ("max_depth", depth_candidates),
]:
    results = []
    for value in candidates:
        candidate = DecisionTreeClassifier(random_state=55, **{parameter: value})
        candidate.fit(X_train, y_train)
        results.append((
            value, candidate.score(X_train, y_train), candidate.score(X_val, y_val),
        ))
    print(parameter, results)  # (candidate value, train accuracy, validation accuracy)
```

Each candidate is a new fit. Other settings stay at defaults; depth and split threshold are not jointly tuned. The detailed note runs the same sweeps for forests and tests `n_estimators=[10, 50, 100, 500]`. High train/low validation accuracy suggests overfitting; both low suggests underfitting (also inspect data/features). Reducing depth or increasing minimum split size restricts growth, but does not guarantee better validation accuracy. Similar scores are not proof of no overfitting.

**Continue the setup block** for modern XGBoost early stopping:

```python
from xgboost import XGBClassifier

n = int(len(X_train) * 0.8)  # 587 fit rows, 147 early-stopping rows
xgb = XGBClassifier(
    n_estimators=500, learning_rate=0.1, early_stopping_rounds=10,
    eval_metric="logloss", random_state=55,
)
xgb.fit(
    X_train.iloc[:n], y_train.iloc[:n],
    eval_set=[(X_train.iloc[n:], y_train.iloc[n:])], verbose=False,
)
print(xgb.best_iteration, xgb.best_score)
print(accuracy_score(y_train, xgb.predict(X_train)))
print(accuracy_score(y_val, xgb.predict(X_val)))
# sklearn wrapper defaults to the best iteration after early stopping.
assert (
    xgb.predict(X_val)
    == xgb.predict(X_val, iteration_range=(0, xgb.best_iteration + 1))
).all()
```

| Syntax / setting | Read it as |
| --- | --- |
| `pd.get_dummies(data, columns=..., prefix=...)` | Replace category columns with boolean/0–1 flags |
| `X = df[features]; y = df["HeartDisease"]` | Inputs versus answer; exclude the target from inputs |
| `train_size=.8, random_state=55` | Repeatable shuffled 734/184 lab split; no stratify in the source |
| `model.fit(X_train, y_train)` | Learn from training rows; returns the fitted estimator |
| `model.predict(X_val)` | Follow learned rules; returns class labels |
| `accuracy_score(y_true, y_pred)` / `model.score(X, y)` | Fraction of correct class labels |
| `max_depth` | Cap tree depth; None removes the explicit cap |
| `min_samples_split` | Minimum samples to attempt a node split; not a minimum per child |
| `RandomForestClassifier(n_estimators=100)` | 100 trees; default bootstrap samples rows with replacement |
| `max_features="sqrt"` | Random candidate features at each split, about sqrt(feature count) |
| `n_jobs=1` / `n_jobs=-1` | One worker / all processors for the forest; affects resources |
| `XGBClassifier(n_estimators=500, learning_rate=.1)` | Up to 500 sequential gradient-boosting rounds, each contribution shrunk |
| `early_stopping_rounds=10` | Constructor setting: stop after ten rounds without monitored improvement |
| `eval_set=[(X_eval, y_eval)]` | Evaluation pairs passed to fit for monitoring |
| `best_iteration` | Zero-indexed best round; use +1 for an exclusive iteration-range endpoint |
| `GridSearchCV(..., param_grid=..., cv=...)` | Evaluate combinations over folds, unlike separate one-setting sweeps |

**Source results (train / outer validation), not guarantees:** decision tree **.8665 / .8696**; unseeded final forest **.9346 / .8913**; XGBoost **.9251 / .8641**. The XGBoost log's best index **16** means 17 rounds, with best logloss **.43948**, correcting the source typo “4.3948.” Stopping after index **26** means 27 rounds were built. The sklearn wrapper's `predict`/`predict_proba` use the best round automatically, although the booster normally retains later rounds. The native booster needs an explicit iteration range to select the best prefix. The reported XGBoost train score includes the 147 monitoring rows, not just its 587 fit rows. See [XGBoost prediction behavior](https://xgboost.readthedocs.io/en/stable/prediction.html#early-stopping).

Training learns rules; validation chooses settings; an untouched test set evaluates the frozen procedure. The lab's 184 rows are **validation**, even where its print label says “test.” Repeated model/hyperparameter selection can overfit them. GridSearchCV also needs a separate final test set. The detailed note provides a runnable `Pipeline + ColumnTransformer + OneHotEncoder(handle_unknown="ignore")` search: encoding is fitted only on each training fold, and unseen categories are handled consistently. This is stronger real-world practice than `get_dummies` before splitting; fixed-category encoding alone is **not proof of target leakage**. See [sklearn's leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

## 12c. Unsupervised learning and K-means

**Concept:** clustering fits groups without supplied class labels. K counts groups; n counts features. Each row is a point, each feature is a coordinate. A centroid is a mean position, not necessarily a data row. Cluster IDs are arbitrary, not probabilities or known topic names.

For the slow walkthrough of coordinates, append/argmin, nested loops, updates, convergence, scaling and embedding examples, read [Unsupervised ML and K-means](UNSUPERVISED_LEARNING_AND_KMEANS.md). Keep detailed explanations there.

**Lab progress, 25 October 2026:** full Andrew Ng K-means clustering lab completed as reported by the user; high-level understanding achieved, with deeper fluency requiring hands-on reuse. Detailed mastery is not claimed. Exercise 1 `find_closest_centroids` completed; Exercise 2 `compute_centroids` passed all unit tests as confirmed by the user. See the [recorded centroid output](UNSUPERVISED_LEARNING_AND_KMEANS.md#completed-k-means-exercises), [Exercise 1 mental gymnastics](UNSUPERVISED_LEARNING_AND_KMEANS.md#exercise-1-mental-gymnastics-counts-vectors-and-identities) and [Exercise 2 mask/mean trace](UNSUPERVISED_LEARNING_AND_KMEANS.md#exercise-2-mental-gymnastics-one-group-all-its-rows).

| Operation | Meaning / shape |
| --- | --- |
| X; centers | (m, n); (K, n) |
| X.shape[0]; X[:, 1] | Integer sample count; second feature column `(m,)` |
| X[i] - centers[j] | Coordinate differences for one point/center pair: (n,) |
| np.linalg.norm(X[i] - centers[j]) | One Euclidean distance |
| distances.append(distance) | Save one scalar; reset list for each sample |
| np.min([5., 1.5, 4.]) | 1.5: smallest value |
| np.argmin([5., 1.5, 4.]) | 1: zero-based position of that value |
| idx[i] = np.argmin(distances) | Store nearest-center ID after all K comparisons |
| for i in range(K) | Update K centroid rows; here i is a cluster ID |
| idx[i] == i; idx == i | One Boolean; full `(m,)` Boolean assignment mask |
| X[idx == j].mean(axis=0) | New center: (n,); handle empty groups first |
| np.argmin(D2, axis=1) | For (m, K) squared distances, return (m,) IDs |

```python
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1., 1.], [2., 1.], [8., 8.], [9., 8.]])
model = KMeans(n_clusters=2, init="k-means++",
               n_init=10, random_state=42)
idx = model.fit_predict(X)                 # no y required
assert idx.shape == (4,)
assert idx[0] == idx[1] and idx[2] == idx[3]
assert idx[0] != idx[2]                     # numeric IDs can swap
assert model.cluster_centers_.shape == (2, 2)
assert model.predict([[1.5, 1.]]).shape == (1,)
assert model.transform(X).shape == (4, 2)   # distances, not IDs
```

Initialize → assign → average → repeat. K-means minimizes within-group **squared** Euclidean error; inertia_ is its sum. Stopping at a tolerance or iteration limit does not guarantee a global optimum. Choose K using geometry, stability and task inspection; inertia alone favors more groups.

**Exercise reminders:** `X` and `centroids` are local arrays, not `np.X` or `np.centroids`. Each indexed row is a complete feature vector; features need not be physical coordinates. Subtract before calling norm: its second argument is `ord`, not another point. Reset distances per sample, append inside the center loop, and save argmin after that loop. In Exercise 2, `points = X[idx == i]` is the whole group; average `points`, not `points[i]`. For `(r, n)` points, axis=0 returns n column means. Indexing one row first discards the other samples and its mean is a scalar that can silently broadcast into a centroid. Detailed examples and the [syntax lookup](UNSUPERVISED_LEARNING_AND_KMEANS.md#k-means-syntax-lookup) live in the topic note.

**Pitfalls:** scale numeric measurements when units would dominate distance; save and reuse preparation for new data. Initialization, outliers and noncompact/unequal groups matter. Empty-group means need a policy. For (m, K) distances, omitting argmin's axis chooses one flattened index.

**Image compression lookup:** pixels are samples, RGB are three features: `(128, 128, 3) → reshape(-1, 3) → (16384, 3)`. K=16 gives a `(16, 3)` palette; integer `idx` has shape `(16384,)` and holds color IDs, not distances. `centroids[idx, :]` retrieves one full RGB row per pixel, then `.reshape(original_img.shape)` restores the grid. This is lossy replacement with representative colors. Raw RGB uses 393,216 bits; an ideal 8-bit RGB palette plus packed 4-bit IDs uses 65,920 bits, about 6× smaller. The reconstructed NumPy RGB array is not an actually packed compressed file. See the [low-level palette walkthrough](UNSUPERVISED_LEARNING_AND_KMEANS.md#5b-image-compression-pixels-to-palette-and-back).

**Random initialization:** the lab uses `randidx = np.random.permutation(X.shape[0])`, then `centroids = X[randidx[:K]]`: K complete sample rows, without repeating row indices (duplicate colors can still occur). Run the existing assign/mean cycle; recompute idx against the final centers before lookup. See [random starting rows](UNSUPERVISED_LEARNING_AND_KMEANS.md#5a-random-starting-rows).

**LLM connection:** embed texts → cluster fixed vectors → inspect topics. This does not train the encoder. Self-supervised LLM pretraining derives token targets from text and updates neural weights with a prediction loss. Row L2 normalization and column standardization differ; ordinary K-means on unit embeddings is not automatically spherical/cosine K-means.

## 12d. Anomaly detection

**Start with WHY:** epsilon is an alarm setting. We need to choose a useful setting. A trial setting turns density scores into decisions; known validation answers reveal correct alarms, false alarms and missed anomalies (TP/FP/FN). Those counts give precision and recall; their F1 score grades the setting. Try settings and keep the highest validation F1. Learning the count expressions before that purpose chain made this lab hard to reconstruct from memory.

```text
X_train → feature means/variances → validation densities
densities + candidate epsilon → Boolean predictions
predictions + y_val → TP/FP/FN → precision/recall → F1
highest candidate F1 → chosen epsilon → alarms on future rows
```

The complete [Anomaly Detection lab guide](ANOMALY_DETECTION.md) owns the section-by-section lab explanation, Gaussian mathematics, full exercise functions, plots, glossary and observed results. This section owns the compact reusable syntax. Reuse [feature statistics](#from-my-averaging-loop-to-feature-statistics) for mean/variance rather than adding a competing explanation here.

### Boolean counting and guarded division

Runnable independently:

```python
import numpy as np

# Invented teaching data: density scores, not anomaly probabilities.
p_val = np.array([0.01, 0.04, 0.08, 0.20, 0.70])
y_val = np.array([1,    0,    1,    1,    0])
epsilon = 0.10
assert p_val.shape == y_val.shape == (5,)

prediction = p_val < epsilon
assert prediction.tolist() == [True, True, True, False, False]
tp = np.sum((prediction == 1) & (y_val == 1))
fp = np.sum((prediction == 1) & (y_val == 0))
fn = np.sum((prediction == 0) & (y_val == 1))
assert (tp, fp, fn) == (2, 1, 1)

if tp + fp == 0:
    prec = 0
else:
    prec = tp / (tp + fp)
if tp + fn == 0:
    rec = 0
else:
    rec = tp / (tp + fn)
if prec + rec == 0:
    F1 = 0
else:
    F1 = 2 * ((prec * rec) / (prec + rec))

np.testing.assert_allclose([prec, rec, F1], [2/3, 2/3, 2/3])
```

`prediction` is `(5,)`, not one Boolean. `(prediction == 1)` identifies alarms; `(y_val == 1)` identifies real anomalies; `&` keeps positions where both are True. `np.sum` counts True entries as 1. FP changes the truth condition to normal; FN changes the prediction condition to no alarm. No integer conversion is needed.

Precision = `TP/(TP+FP)` = correct alarms/all alarms. Recall = `TP/(TP+FN)` = caught anomalies/all actual anomalies. F1 is their harmonic mean; it is not accuracy. Each guard avoids its own zero denominator. The lab chooses 0 when a ratio is undefined; that convention does not make absent anomalies a useful evaluation set. TP=0 yields guarded F1=0 even when many normals are correctly ignored.

### Threshold sweep, shapes and syntax

`step_size = (max(p_val)-min(p_val))/1000` divides the score range into about 1000 equal steps. `np.arange(start, stop, step)` returns candidate values with the stop normally excluded; floating rounding can affect endpoints/count. For example, `np.arange(0,1,0.25)` gives `[0.,0.25,0.5,0.75]`. It is not `range(1000)`: those would be integer counters, not scaled density cutoffs. Equal scores make the lab's step zero; see the [external edge-case guidance](ANOMALY_DETECTION.md#8-mistakes-edge-cases-and-practical-limits). Keep the provided notebook scaffold intact.

Inside each trial, recalculate predictions, counts and metrics. `if F1 > best_F1:` remembers an improvement; equal scores keep the earlier choice. Return the best pair after the loop. See the [complete uploaded function and execution trace](ANOMALY_DETECTION.md#7-exercise-2--select_threshold). The finite grid chooses the best among tested candidates, not a guaranteed unique/global real-valued optimum.

| Syntax | Meaning / expected result |
| --- | --- |
| `X[:5]` | First five rows, stop excluded |
| `X[:, 0]`, `X[:, 1]` | All rows of first/second feature, each `(m,)` |
| `m, n = X.shape` | Unpack row/column counts; `X[0]` instead contains values |
| `np.mean(X, axis=0)` | One mean per feature, shape `(n,)` |
| `np.var(X, axis=0)` | Population variance with divisor m (`ddof=0`), shape `(n,)` |
| `X - mu` | `(m,n) - (n,) → (m,n)`; broadcast matching feature means across rows |
| `(X - mu) ** 2` | Square each centered entry; `^` is not exponentiation |
| `np.prod(feature_density, axis=1)` | Multiply features within each row → `(m,)` densities |
| `p_val < epsilon` | Broadcast scalar cutoff over `(m_val,)` scores → Boolean array |
| `=` / `==` | Store a value / compare values |
| `(a == 1) & (b == 1)` | Elementwise AND, with comparison parentheses |
| `~mask` | Elementwise NOT for a Boolean mask; not numeric subtraction |
| `np.sum(mask)` | Count True entries → scalar |
| `X[mask]` / `X[mask, 0]` | Selected rows `(k,n)` / their first feature `(k,)` |
| `if tp + fp == 0:` | One scalar condition; do not use `if prediction:` on a multi-element array |
| `return best_epsilon, best_F1` | Return two objects; caller unpacks them into two names |
| `%e`, `%f`, `%d` | Scientific, fixed decimal and integer display in the supplied print strings |

**Common mistakes:** `prediction` and `predictions` are different names; a stale notebook variable can hide the error. Use `&`, not Python `and`, for array masks. Require both scores and labels to have matching 1-D shapes and row order: `(m,)` with `(m,1)` can silently broadcast to `(m,m)`. Count against labels, not against scores. Guard precision, recall and F1. For anomaly density, **low** score triggers the alarm (`<`), unlike a high positive sigmoid probability. Density is not event probability or probability of normality. This model multiplies feature-wise independent Gaussian densities; it does not learn full covariance.

**Plot lookup:** `outliers = p < epsilon` uses training densities; `plt.plot(X_train[outliers,0], X_train[outliers,1], 'ro', markerfacecolor='none')` rings those rows. Match mask and data source even when training/validation happen to have equal lengths. Full styling and contours are explained in the [2D plotting walkthrough](ANOMALY_DETECTION.md#10-finish-the-2d-lab-score-contours-and-outlier-circles).

**Recorded lab results:** 2D shapes `(307,2)`, `(307,2)`, `(307,)`; mu `[14.11222578,14.99771051]`, var `[1.83263141,1.70974533]`; epsilon `8.990853e-05`, F1 `0.875000`, both exercise tests passed in the supplied output. High-dimensional shapes `(1000,11)`, `(100,11)`, `(100,)`; epsilon `1.377229e-18`, F1 `0.615385`, **117 training outliers**. These are uploaded observations, not results rerun here; the [11-feature walkthrough](ANOMALY_DETECTION.md#11-the-actual-11-feature-dataset) explains their interpretation. No non-graded notebook cells are changed.

## 13. Related notes and API sources

The focused notes remain useful; this page adds one canonical syntax entry point rather than replacing them:

- [Anomaly Detection](ANOMALY_DETECTION.md): complete uploaded server lab, ELI5 theory, formulas, both exercises, execution flow, contours and high-dimensional results.
- [Decision Trees](DECISION_TREES.md): supplied original lab reproduction and verified entropy/gain, a separate illustrative example, safer helpers, sklearn CV, overfitting and telecom applications.
- [Python for ML — Quick Reference](PYTHON_FOR_ML_QUICK_REFERENCE.md): original Python patterns, code-reading checklist, and detailed bias/variance and λ reminders (sections 20–21).
- [NumPy for ML](NUMPY_FOR_ML.md): focused arrays, vectorization, shapes, weights, thresholding.
- [TensorFlow/Keras for ML](TENSORFLOW_KERAS_FOR_ML.md): focused layer, training, and forward-pass explanations.
- [Course 2 Week 1 — Neural Networks](COURSE_2_WEEK_1_NEURAL_NETWORKS.md): the 400-feature digit example and worked shapes.
- [Backpropagation using a Computation Graph](COURSE_2_BACKPROPAGATION_COMPUTATION_GRAPH.md): where the GradientTape example's derivative 66 comes from.
- [Gradient Descent](GRADIENT_DESCENT.md): gradients, learning rate, updates, and convergence.
- [Fit, Transform, Predict Deep Dive](FIT_TRANSFORM_PREDICT_DEEP_DIVE.md): ruler analogy, scaler overwrite behavior, full lab walkthrough.
- [sklearn Preprocessing Cheat Sheet](SKLEARN_PREPROCESSING_CHEAT_SHEET.md): more manual/Pipeline examples and degree selection.

Official API references for checking details and extending examples:

- [Python tutorial](https://docs.python.org/3/tutorial/), [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html), [NumPy indexing](https://numpy.org/doc/stable/user/basics.indexing.html).
- [Matplotlib pyplot tutorial](https://matplotlib.org/stable/tutorials/pyplot.html), [savefig](https://matplotlib.org/3.11.0/api/_as_gen/matplotlib.pyplot.savefig.html).
- [TensorFlow Sequential guide](https://www.tensorflow.org/guide/keras/sequential_model), [BinaryCrossentropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy), [GradientTape](https://www.tensorflow.org/api_docs/python/tf/GradientTape), [Keras regularizers](https://keras.io/api/layers/regularizers/).
- [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html), [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html), [leakage pitfalls](https://scikit-learn.org/stable/common_pitfalls.html), [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

## 14. How this sheet grows

**Complete anomaly lab addition, 25 October 2026:** section 12d adds the purpose chain, one independent Boolean-counting/guard example, syntax lookup and links to the complete uploaded [lab guide](ANOMALY_DETECTION.md). Seven teaching fixtures ran; all 78 Python blocks across the five edited documents passed syntax checks; both documented exercise bodies match the uploaded source structurally; 74 links into edited documents passed. Python 3.12.14, NumPy 2.3.5. Hand-calculated metrics, guard cases, grid bounds, statistics and broadcasting passed. Recorded course outputs were transcribed, not rerun. Existing unrelated examples were not runtime-tested again. See the [full verification scope](ANOMALY_DETECTION.md#sources-and-verification).

**NumPy recall follow-up, 25 October 2026:** expanded existing sections 2–4 with counts versus row values, the corrected averaging loop, feature-wise mean/population variance, a quick lookup and a small index/distance example. All 38 sheet Python blocks passed syntax checks; all 11 NumPy blocks in sections 2–4 ran with their numerical/shape assertions. Original examples were preserved. New canonical anchors, code fences and whitespace checks passed. Environment: Python 3.12.14, NumPy 2.3.5. Other sections and the course grader were not runtime-tested again.

**K-means exercises follow-up, 25 October 2026:** section 12c now records both completed exercises and links to the specific mistake walkthroughs. All 11 companion-note blocks and this section's example ran; all 35 sheet blocks passed syntax checks. Masks, feature means, scalar broadcasting, singleton groups, ties, empty-group policy, links and preservation checks passed. See the [verification record](UNSUPERVISED_LEARNING_AND_KMEANS.md#sources-and-verification). Exercise 2's course unit-test success is user-confirmed; the course grader was not rerun here.

**Unsupervised ML addition, 25 October 2026:** section 12c adds a compact K-means syntax/concept reference linked to the dedicated study note. All eight note blocks and the new sheet example ran offline; distance/assignment/update, convergence, empty-group/tie and vectorization checks passed. All 35 sheet blocks passed syntax checks; new links/anchors and preservation checks passed. See the [detailed scope and environment](UNSUPERVISED_LEARNING_AND_KMEANS.md#sources-and-verification). Real embeddings, LLM training and course graders were not run.

**Brain-burner follow-up, 21 October 2026:** added a compact gain-selection reminder and a direct troubleshooting link in section 12a. The companion's new selection/weight examples and existing indexing snippet passed runtime checks; all 34 sheet blocks and 14 note blocks passed syntax checks. Relative links/anchors passed, and earlier content was preserved. See the [detailed verification record](DECISION_TREES.md#sources-and-verification) for checks and limits.

**Graded decision-tree addition, 21 October 2026:** added the current-node ID reminder and direct link to section 4a of the detailed note. Ran the new indexing snippet and both companion snippets with exact output/assertion checks, re-ran the existing section 4 helpers, syntax-checked all 34 sheet blocks and 13 note blocks, and verified new links/fences. Python 3.12.14, NumPy 2.3.5. The course grader was not run; see the [detailed verification record](DECISION_TREES.md#sources-and-verification) for scope. Earlier records are preserved.

**Trees Ensemble verification, 19 October 2026:** syntax-checked all 33 Python blocks in this sheet and checked the new links and code fences. Runtime-tested section 12a and section 12b's setup and tuning loops; the new examples used a synthetic 918-row schema fixture, not the original heart.csv. The companion note's original five blocks, 36 sweep fits, selected tree/forest models, plot and preprocessing grid also passed. **XGBoost was not installed**, so its two new examples were syntax-checked and API-reviewed only. The reported lab scores were not reproduced. See the [detailed verification record](DECISION_TREES.md#sources-and-verification) for exact checks, versions and limits. Earlier verification records remain unchanged.

1. Add newly learned coding topics here when they become useful in actual labs/projects. Keep beginner refreshers separate from intermediate extensions.
2. For each addition: show imports, data/setup or an explicit continuation dependency, expected shapes/output, and one common mistake. Prefer tiny generated data to unavailable course files.
3. Add the navigation link. Link to deeper notes instead of overwriting them or creating another competing canonical sheet.
4. Run examples where dependencies are available. Record what was executed versus only reviewed; do not invent model scores or claim course mastery.
5. Update the learning log when a meaningful topic is added. Commit changes to GitHub so this stays retrievable across chats.

**Verification, 18 October 2026:** all 29 Python blocks passed syntax checks. Executed 22 Python/NumPy/Matplotlib/sklearn blocks, including the explicit Ridge continuation, embedded numerical/shape assertions, model-selection and metric examples, and two successful PNG exports. Local navigation anchors and relative repository links passed checks. Environment: NumPy 2.5.2, Matplotlib 3.11.2, scikit-learn 1.9.1. The remaining 7 TensorFlow blocks were syntax-checked and API-reviewed; TensorFlow is not installed in the verification environment, so those blocks were **not runtime-tested**. Learned model scores are deliberately not promised.


**Addition, 19 October 2026:** added section 12a and the dedicated Decision Trees link. Reviewed against official sklearn APIs; the new Python example was not runtime-tested in this task environment. The 18 October verification record above remains unchanged.

**Lab verification update, 19 October 2026:** the linked Decision Trees note now reproduces the original ungraded lab excerpt supplied in the handoff. Section 12a's sklearn block was runtime-tested offline with Python 3.12.14 and scikit-learn 1.9.1; its shape assertions passed. The earlier verification records above describe the earlier runs and are preserved.







## 12e. Collaborative filtering + TensorFlow custom training (Andrew Ng lab)

**Canonical deep dive:** [Collaborative Filtering — from loops to training](COLLABORATIVE_FILTERING.md). **TensorFlow/Keras API detail:** [TensorFlow/Keras reference](TENSORFLOW_KERAS_FOR_ML.md).

Mental model: `Y[i,j]` is movie `i`, user `j`; `R[i,j]` says if rating exists. `X[i]` is a movie feature vector, `W[j]` a user preference vector, and `b[0,j]` the user's bias. The scored pair is a dot product plus bias. The scalar cost aggregates **observed** squared errors plus the squared lengths of **all** movie and user parameter vectors (not bias). The student independently built the for-loop version; notebook public tests passed, unregularized cost 13.67 and regularized cost 28.09.

General syntax worth keeping:
```python
import numpy as np
matrix = np.array([[2., 1.], [3., 4.]])
print(matrix.shape)           # (2, 2)
print(matrix[0, 1])          # 1.0: row 0, column 1
print(np.sum(matrix ** 2))   # 30.0: elementwise squares then sum
print(np.dot(matrix[0], matrix[1]))  # 10.0: one vector dot product
```

TensorFlow training building blocks (lab syntax): `tf.Variable` trainable parameters, `tf.random.normal` random starts, `tf.linalg.matmul` matrix product, `tf.transpose` swap axes, `tf.reduce_sum` scalar reduction, `tf.GradientTape` automatic differentiation, `tape.gradient` derivatives, `keras.optimizers.Adam` optimization, `optimizer.apply_gradients(zip(grads, variables))` parameter updates. `np.c_` inserts a user column; `tf.argsort` ranks prediction indices. **Cost computation and parameter updates are separate steps.** See focused note for shapes, normalization and exact lab sequence.

 
## 12f. Content-based filtering (two-tower recommender lab)

**Detailed theory, actual source and caveats:** [Content-based filtering](CONTENT_BASED_FILTERING.md). Keep this file as the canonical mega cheat sheet.

**Input is NOT embedding:** stored user row has id/count/mean + 14 genre averages, but training uses only the last 14 (u_s=3). Stored movie row has id + year/mean + 14 **multi-hot** genre flags, and uses only last 16 (i_s=1). A genre average of zero can mean no ratings, not dislike. Two independent towers map 14 and 16 original input features to two **learned 32-dimensional output embeddings**. Learned embeddings are not hardcoded genre labels or the model's own Dense weights.

Architecture of both towers: Dense(256,ReLU) -> Dense(128,ReLU) -> Dense(num_outputs=32,linear). Keras Functional API connects with L2-normalized output vectors and Dot(axes=1). Trainable params: 40,864 user + 41,376 item = 82,240. Normalized dot product = cosine similarity, [-1,1]. Target ratings are scaled to (-1,1) using MinMaxScaler; StandardScaler separately transforms input columns. Distinguish **three** scalers: input standardization, target min-max, output unit-vector L2 normalization.

Keras workflow: MeanSquaredError, Adam(learning_rate=0.01), model.compile, model.fit(...,epochs=30), model.evaluate, model.predict, inverse_transform and np.argsort(-scores). Unlike the earlier collaborative-filtering custom GradientTape loop, compile/fit handles backprop and optimizer updates.

General syntax (not graded):
~~~python
import numpy as np
import tensorflow as tf
a = np.array([1., 2., 3.])
b = np.array([2., 2., 1.])
print(np.sum((a-b)**2))   # 5.0
example = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(2)
])
print(example(tf.ones((4,3))).shape)  # (4, 2)
~~~

Similar movie search compares **movie embeddings to movie embeddings**, uses squared distance np.sum((a-b)**2), excludes self-match by diagonal masking, and chooses minimum with np.argmin. For L2-unit vectors, squared distance = 2-2*dot(a,b). Lab reported 397 users, 847 movies, 25,521 ratings, 50,884 example rows, training loss 0.0730, test loss 0.0812; both tower public tests and sq_dist public test passed. Source caveats: scaling fitted before split, repeated rows, existing-user helper error for uid 2, inverse-scaled new-user scores around 0.3 and a displayed movie2-genre metadata copy/paste bug. See dedicated note.
