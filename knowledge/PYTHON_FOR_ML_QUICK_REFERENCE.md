# Python for ML — Quick Reference

This is not a second Python course and it is not a list of things to memorize.

It is the small set of Python and NumPy patterns I keep seeing in the Machine Learning Specialization. The goal is to have them readily available while I work through labs, until reading them becomes automatic.

---

## 1. Imports

```python
import numpy as np
import matplotlib.pyplot as plt
```

- `numpy` gives me numerical arrays and vector/matrix operations.
- `matplotlib.pyplot` is commonly used for plots.
- `np` and `plt` are conventional short names.

---

## 2. Python loop patterns

Loop over a fixed range:

```python
for i in range(5):
    print(i)
```

This produces `0, 1, 2, 3, 4`.

Loop over all training examples:

```python
m = X.shape[0]

for i in range(m):
    print(X[i])
```

Mental translation:

> `i` is the current training-example number.

Nested loops often mean:

```python
for i in range(m):      # each training example
    for j in range(n):  # each feature
        ...
```

For our ML work, `i` usually means **which example**, while `j` usually means **which feature/weight**.

---

## 3. Functions and returning values

```python
def predict(x, w, b):
    prediction = np.dot(x, w) + b
    return prediction
```

Calling it:

```python
result = predict(x, w, b)
```

Multiple return values are common:

```python
def example():
    return 10, 20

a, b = example()
```

This is why ML code can do:

```python
dj_dw, dj_db = compute_gradient(X, y, w, b)
```

---

## 4. NumPy arrays

```python
x = np.array([1, 2, 3])
```

Unlike a normal Python list, NumPy arrays are designed for numerical operations.

```python
x * 2
# array([2, 4, 6])
```

A weight vector may look like:

```python
w = np.array([1.5, 2.0, -0.5])
```

One feature generally means one weight. Three features generally mean three weights.

---

## 5. Shape — what does the data container look like?

```python
X.shape
```

For ML data:

```text
X.shape = (m, n)

m = number of training examples / rows
n = number of features / columns
```

Example:

```python
X = np.array([
    [1200, 3, 1, 40],
    [1800, 4, 2, 20],
    [900,  2, 1, 70]
])

print(X.shape)
# (3, 4)
```

That means:

> 3 examples, each with 4 features.

Useful shortcuts:

```python
m = X.shape[0]   # number of examples
n = X.shape[1]   # number of features
```

---

## 6. Indexing rows and features

One value:

```python
X[0, 1]
```

means row 0, column 1.

One complete training example:

```python
X[0]
```

or explicitly:

```python
X[0, :]
```

One complete feature/column:

```python
X[:, 0]
```

Mental model:

```text
X[i, j] → example i, feature j
X[i, :] → all features for example i
X[:, j] → feature j for all examples
```

---

## 7. Creating arrays of zeros

```python
np.zeros(3)
```

gives roughly:

```text
[0. 0. 0.]
```

In gradient descent:

```python
dj_dw = np.zeros(n)
```

means:

> Create one gradient accumulator for each of the `n` weights/features.

---

## 8. Dot product

```python
np.dot(x, w)
```

For:

```python
x = np.array([2, 3, 4])
w = np.array([10, 20, 30])
```

this means:

```text
2×10 + 3×20 + 4×30
```

So multiple linear regression can be written compactly as:

```python
prediction = np.dot(x, w) + b
```

instead of manually writing every `x[j] * w[j]`.

Mental translation:

> Multiply matching features and weights, then add the results.

---

## 9. Accumulating values with `+=`

```python
total = 0

for i in range(5):
    total += i
```

`+=` means:

```python
total = total + i
```

So:

```python
dj_dw[j] += something
```

means:

> Add this training example's contribution to the gradient already accumulated for weight `j`.

This pattern appears constantly in manual cost and gradient calculations.

---

## 10. Prediction → error

A central ML pattern:

```python
prediction = np.dot(X[i], w) + b
error = prediction - y[i]
```

Human translation:

```text
model's guess - reality = error
```

This is one of the most important pieces to recognize quickly.

---

## 11. Cost pattern

A simplified squared-error pattern looks like:

```python
cost = 0

for i in range(m):
    prediction = np.dot(X[i], w) + b
    error = prediction - y[i]
    cost += error ** 2
```

The exact cost formula may then divide/average the accumulated value.

Do not memorize the loop. Read it as:

> For every example, predict → compare with reality → square the error → add it to the total.

---

## 12. Gradient-descent update

The lines I should recognize immediately:

```python
dj_dw, dj_db = compute_gradient(X, y, w, b)

w = w - alpha * dj_dw
b = b - alpha * dj_db
```

Human translation:

> Calculate the slopes, then move the parameters a small step in the direction that reduces cost.

I do not need to reproduce the entire gradient algorithm from memory. I do need to understand what these updates mean.

---

## 13. Scientific notation

Common examples:

```text
1e-1  = 0.1
1e-2  = 0.01
1e-5  = 0.00001
1e-10 = 0.0000000001
```

Remember what happened experimentally:

> A ridiculously tiny learning rate did not corrupt gradient descent — it made it move ridiculously slowly.

---

## 14. Useful NumPy statistics

Mean:

```python
mu = np.mean(X, axis=0)
```

Standard deviation:

```python
sigma = np.std(X, axis=0)
```

Peak-to-peak range:

```python
np.ptp(X, axis=0)
```

Peak-to-peak simply means:

```text
maximum - minimum
```

With `axis=0`, calculate the statistic separately down every feature/column.

---

## 15. Z-score normalization

```python
X_norm = (X - mu) / sigma
```

Mental translation:

> For every feature value, subtract that feature's training mean and divide by that feature's training standard deviation.

Golden rule:

> **New prediction data must use the same mean and standard deviation calculated from the training data.**

Example:

```python
x_new_norm = (x_new - training_mu) / training_sigma
prediction = np.dot(x_new_norm, w) + b
```

---

## 16. scikit-learn StandardScaler

Instead of manually calculating Z-score normalization:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
```

Important distinction:

```text
fit       → learn the training-data statistics
transform → apply the learned transformation
```

For future/new data:

```python
X_new_norm = scaler.transform(X_new)
```

Do not `fit_transform()` the new example again. The existing fitted scaler preserves the training reference system.

---

## 17. Feature engineering

Create a squared feature:

```python
x_squared = x ** 2
```

Create polynomial features:

```python
X = np.c_[x, x**2, x**3]
```

This creates columns for:

```text
x    x²    x³
```

The important concept is not the syntax. Feature engineering means creating useful inputs that expose relationships the model can learn.

A real-world example:

```python
revenue_feature = items_sold * price_per_item
```

Ask:

> What does my engineered feature mean in the real world, and why might it help predict the target?

---

## 18. Reshaping

Sometimes a single feature starts as:

```python
x.shape
# (20,)
```

but a function expects a 2-D feature matrix:

```python
X = x.reshape(-1, 1)
```

Now:

```text
X.shape = (20, 1)
```

Meaning:

> 20 training examples, 1 feature each.

The `-1` tells NumPy to work out that dimension automatically.

---

## 19. The code-reading checklist

When a lab function looks intimidating, do not try to understand the entire block at once.

Ask, in this order:

1. What are the inputs?
2. What shape are they?
3. What is the loop iterating over?
4. Where is the prediction calculated?
5. Where is prediction compared with reality?
6. What value is being accumulated?
7. What does the function return?

Then translate each important line into English.

For example:

```python
f_wb = np.dot(X[i], w) + b
```

becomes:

> Make a prediction for training example `i`.

And:

```python
err = f_wb - y[i]
```

becomes:

> Compare the prediction with the real target for that example.

This translation skill matters more right now than memorizing syntax.

---

# 60-second cheat sheet

```python
# examples and features
m, n = X.shape

# one example
X[i]

# one value
X[i, j]

# prediction
prediction = np.dot(X[i], w) + b

# error
error = prediction - y[i]

# one zero per feature
dj_dw = np.zeros(n)

# accumulate
total += value

# gradient-descent update
w = w - alpha * dj_dw
b = b - alpha * dj_db

# feature statistics
mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)

# normalize
X_norm = (X - mu) / sigma

# polynomial feature
x_squared = x ** 2

# convert one feature to a 2-D feature matrix
X = x.reshape(-1, 1)
```

The goal is not to memorize this page. Keep using these patterns until they become boring.
