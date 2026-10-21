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
| Beginner | [3. NumPy selection and axes](#3-numpy-selection-and-axes) | Indexing, slicing, masks, statistics |
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

`Dense(25)` means **25 output neurons**, each receiving all 400 input features. It computes `activation(X @ W + b)`. `Sequential` passes one layer's output into the next. The same weights serve every image. This untrained model performs forward propagation; it has not learned digit recognition. `get_weights()` inspects parameters, not training progress by itself.

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

## 13. Related notes and API sources

The focused notes remain useful; this page adds one canonical syntax entry point rather than replacing them:

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


