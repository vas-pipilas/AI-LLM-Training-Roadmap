# NumPy for ML — Detailed Cheat Sheet

This is my practical NumPy reference for machine-learning labs. The goal is not to memorize the NumPy API. It is to recognize the small set of array, shape, indexing, and linear-algebra operations that keep appearing in ML.

The most important rule:

> **Before debugging the mathematics, inspect the shapes.**

---

## 1. The ML data model

For a normal tabular training matrix:

```text
X.shape = (m, n)

m = number of training examples / rows
n = number of features / columns
```

Example:

```python
X.shape
# (1000, 400)
```

means 1000 examples and 400 features per example.

For the handwritten-digit lab, each original image is 20 × 20 pixels:

```text
20 × 20 = 400 pixel features
```

The image is flattened into one row:

```text
X[i] = [pixel_1, pixel_2, ... pixel_400]
```

while `y[i]` is the label for the **whole image**, not for an individual pixel.

---

## 2. Arrays and dimensions

```python
import numpy as np

x = np.array([1, 2, 3])
```

Common shapes:

```text
scalar       → ()
vector       → (n,)
matrix       → (m, n)
```

Examples:

```python
x = np.array([1, 2, 3])
x.shape
# (3,)

X = np.array([[1, 2, 3],
              [4, 5, 6]])
X.shape
# (2, 3)
```

Do not confuse `(3,)` with `(3,1)`. The first is a 1-D vector; the second is a 2-D column-shaped matrix.

---

## 3. Indexing

```python
X[i]       # complete example i
X[i, :]    # same idea explicitly
X[i, j]    # feature j of example i
X[:, j]    # feature j for every example
```

Mental rule:

> **i = which example; j = which feature/unit, depending on the code.**

For neural-network weights in Andrew Ng's NumPy convention:

```python
W[:, j]
```

means all incoming weights belonging to neuron/unit `j`.

---

## 4. Creating arrays

```python
np.zeros(3)
np.ones(3)
np.zeros((4, 2))
```

A common ML pattern:

```python
a_out = np.zeros(units)
```

means create a place to store one output activation per neuron.

```python
p = np.zeros((m, 1))
```

means create space for one prediction per training example.

---

## 5. Dot product: the core operation

For one neuron or linear model:

```python
z = np.dot(w, x) + b
```

means:

```text
w1*x1 + w2*x2 + ... + wn*xn + b
```

This produces one scalar `z`.

If a neuron receives `n` inputs, it needs `n` weights.

---

## 6. Matrix multiplication and vectorization

Manual:

```python
for i in range(m):
    z[i] = np.dot(X[i], w) + b
```

Vectorized:

```python
z = X @ w + b
```

If:

```text
X.shape = (m, n)
w.shape = (n,)
```

then:

```text
X @ w → shape (m,)
```

One result per example.

`@` is matrix multiplication. It is not element-by-element multiplication.

---

## 7. Element-wise operations

```python
x * 2
x ** 2
x + 5
```

operate element by element.

Example:

```python
x = np.array([1, 2, 3])
x ** 2
# [1, 4, 9]
```

Compare:

```text
*      → element-wise multiplication
@      → matrix multiplication
np.dot → dot product / linear algebra operation
```

---

## 8. Broadcasting

NumPy can automatically apply a compatible smaller value across a larger array.

```python
z = X @ w + b
```

Here `b` can be one scalar and NumPy adds it to every example's score.

Broadcasting is convenient, but when a result looks bizarre, inspect the shapes first.

---

## 9. Boolean comparisons and thresholding

```python
probabilities >= 0.5
```

might produce:

```text
[ True
  False
  True ]
```

Convert booleans to integer class decisions:

```python
yhat = (probabilities >= 0.5).astype(int)
```

So:

```text
True  → 1
False → 0
```

This is the compact NumPy equivalent of an `if/else` loop over every prediction.

---

## 10. Boolean indexing

```python
X[y == 1]
```

means:

> Give me the rows of X whose corresponding label is 1.

Likewise:

```python
X[y == 0]
```

selects examples in class 0.

---

## 11. Reshape and flattening

Convert 20 values into a 2-D one-feature dataset:

```python
X = x.reshape(-1, 1)
```

Flatten an image:

```python
flat = image.reshape(-1)
```

A 20 × 20 image becomes a 400-element vector.

Always ask:

> Am I changing the data, or only changing how the same values are arranged?

Reshape changes the arrangement/shape, not the underlying values.

---

## 12. Useful statistics

```python
mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)
minimum = np.min(X, axis=0)
maximum = np.max(X, axis=0)
```

With `axis=0`, calculate one statistic for each feature/column.

Z-score normalization:

```python
X_norm = (X - mu) / sigma
```

Normalization changes the scale of input features. It is **not regularization** and does not use lambda.

---

## 13. Accumulation

```python
total = 0

for i in range(m):
    contribution = ...
    total += contribution
```

`+=` means add the current contribution to what has already been collected.

A common mistake is accidentally accumulating an accumulator again. Keep the per-example value and total accumulator conceptually separate.

---

## 14. Neural-network weight matrices

Suppose a layer receives 5 inputs and has 100 neurons.

In the convention used in the current course labs:

```text
W.shape = (5, 100)
           ↑    ↑
        inputs neurons
```

Each column is one neuron's weight vector:

```python
w = W[:, j]
```

Each neuron receives all 5 inputs, so each has 5 weights.

The number of input features does **not** dictate the number of neurons.

---

## 15. Dense layer in plain NumPy

Conceptually:

```python
def dense(a_in, W, b):
    units = W.shape[1]
    a_out = np.zeros(units)

    for j in range(units):
        w = W[:, j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = activation(z)

    return a_out
```

Read it as:

```text
for every neuron:
    get that neuron's weights
    calculate weighted sum + bias
    apply activation
    save its output
```

---

## 16. Shapes through a network

Example architecture:

```text
400 inputs → 25 neurons → 15 neurons → 1 output
```

Using the course convention:

```text
W1.shape = (400, 25)
b1.shape = (25,)

W2.shape = (25, 15)
b2.shape = (15,)

W3.shape = (15, 1)
b3.shape = (1,)
```

Golden rule:

> **Every neuron has one weight for every input coming into that neuron, plus one bias.**

---

## 17. Probability versus class decision

A final sigmoid neuron may output:

```text
0.96
```

That is the model output/probability-like score for the positive class.

The class decision can then be:

```python
yhat = (probability >= 0.5).astype(int)
```

Mental pipeline:

```text
network → sigmoid output → threshold → class
              0.96          0.5       1
```

Do not confuse the continuous sigmoid output with the final 0/1 decision.

---

# NumPy 60-second reference

```python
m, n = X.shape

X[i]          # example i
X[i, j]       # one feature
X[:, j]       # one complete feature

w = W[:, j]   # all weights for neuron j

z = np.dot(w, x) + b
z_all = X @ w + b

a = np.zeros(units)

mask = y == 1
positive_examples = X[mask]

yhat = (probabilities >= 0.5).astype(int)

mu = np.mean(X, axis=0)
sigma = np.std(X, axis=0)
X_norm = (X - mu) / sigma

flat = image.reshape(-1)
column = x.reshape(-1, 1)
```

If confused: **print the shape first.**
