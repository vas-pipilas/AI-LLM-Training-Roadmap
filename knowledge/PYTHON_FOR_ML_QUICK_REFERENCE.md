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

---

## 20. Diagnosing high bias and high variance (Course 2, Week 3)

**ELI5:** high bias = model cannot learn even the practice questions; high variance = model memorizes practice questions but fails new questions. Compare training error to a realistic baseline, then CV error to training error. Both can be high at once.

| Signal | Interpretation | What to try |
|---|---|---|
| Train error far above baseline | High bias / underfitting | More useful features, higher polynomial degree/model capacity, weaker regularization |
| Train error near baseline but CV much worse | High variance / overfitting | More representative training examples, fewer irrelevant features, stronger regularization |
| Train and CV close to baseline | Promising generalization | Confirm once on untouched test data |

**Error comparison:** use the *same metric* on train, CV, and baseline. A high train–CV gap suggests variance; a high baseline–train gap suggests bias. These are diagnostic clues, not rigid numerical thresholds.

```python
baseline_error = 0.02  # e.g. human or credible reference error, 2%
train_error = 0.03
cv_error = 0.19

bias_gap = train_error - baseline_error
variance_gap = cv_error - train_error

print(f"Bias gap: {bias_gap:.1%}")
print(f"Generalization gap: {variance_gap:.1%}")
# 1% bias gap; 16% generalization gap -> high-variance warning
```

### 20.1 Fit preprocessing only on training examples

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # learns train mean/std
X_cv_scaled = scaler.transform(X_cv)             # reuses train mean/std
X_test_scaled = scaler.transform(X_test)         # reuses train mean/std

# scaler.fit(X_cv) would OVERWRITE mean_ and scale_ -> don't do it!
print(scaler.mean_, scaler.scale_)
```

### 20.2 Compare polynomial complexity on CV (avoid test leakage)

```python
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

candidates = []
for degree in range(1, 11):
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        LinearRegression(),
    )
    model.fit(X_train, y_train)
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    cv_mse = mean_squared_error(y_cv, model.predict(X_cv))
    candidates.append((degree, model, train_mse, cv_mse))

best_degree, best_model, best_train_mse, best_cv_mse = min(
    candidates, key=lambda row: row[3]
)
print(best_degree, best_train_mse, best_cv_mse)

# Final evaluation only AFTER model selection:
test_mse = mean_squared_error(y_test, best_model.predict(X_test))
print("Final test MSE:", test_mse)
```

**Why Pipeline?** It remembers the polynomial expansion and fitted training scaler for each candidate, so `predict(X_cv)` and `predict(X_test)` apply the same preprocessing automatically. `mean_squared_error` divides by m; Andrew Ng's lab divides this result by 2 to match its 1/(2m) convention. Be consistent when comparing models.

### 20.3 Regularization sweep: Ridge alpha (lambda-like strength)

```python
from sklearn.linear_model import Ridge

results = []
for alpha in [0.01, 0.1, 1, 10, 100]:
    model = make_pipeline(
        PolynomialFeatures(degree=4, include_bias=False),
        StandardScaler(),
        Ridge(alpha=alpha),
    )
    model.fit(X_train, y_train)
    cv_mse = mean_squared_error(y_cv, model.predict(X_cv))
    results.append((alpha, cv_mse, model))

best_alpha, best_cv_mse, best_model = min(results, key=lambda row: row[1])
print("Selected alpha:", best_alpha)
```

**Direction:** lower alpha = weaker penalty, often helps underfitting; higher alpha = stronger penalty, may help overfitting. Too much alpha causes underfitting. Ridge's `alpha` is not numerically identical to every textbook's lambda convention.

### 20.4 Learning curves: does more training data help?

```python
from sklearn.model_selection import learning_curve

sizes, train_scores, cv_scores = learning_curve(
    make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    X_train, y_train,
    train_sizes=np.linspace(0.2, 1.0, 5),
    cv=5,
    scoring="neg_mean_squared_error",
)
train_mses = -train_scores.mean(axis=1)
cv_mses = -cv_scores.mean(axis=1)
print(list(zip(sizes, train_mses, cv_mses)))
```

**Note:** `learning_curve` performs its own cross-validation *inside the supplied training data*. Keep the final test set separate. More data can reduce a variance gap, but often won't fix a model that is fundamentally too simple.

### 20.5 Classification error (not the same as training loss)

```python
import tensorflow as tf

# Model's final layer is linear; BinaryCrossentropy(from_logits=True)
logits = model.predict(X_cv_scaled, verbose=0)
probabilities = tf.math.sigmoid(logits).numpy()
predicted_labels = (probabilities >= 0.5).astype(int)
classification_error = np.mean(predicted_labels.reshape(-1) != np.asarray(y_cv).reshape(-1))
print(f"CV classification error: {classification_error:.1%}")
```

Use this classification example with a **trained binary neural-network model that outputs logits**. For a sigmoid-output model, do not apply sigmoid a second time. For regression, use MSE (or another appropriate regression metric), not a 0.5 classification threshold.

### 20.6 Real AI/LLM engineering connection

- Fine-tuning: memorizing training prompts but failing unseen prompts resembles high variance.
- RAG: poor answers can come from retrieval, context, prompting, or model behavior; do not automatically label every failure bias/variance.
- Evaluate on representative, held-out examples. Prevent train/dev/test leakage and monitor real-world drift.
- Diagnose first; change one controlled factor; measure again.

**One-minute memory rule:** baseline vs train = bias clue; train vs CV = variance clue; CV chooses configuration; test measures the final choice. More complexity can fix bias and worsen variance. Stronger regularization can reduce variance and worsen bias.

---

## 21. Lambda (λ): the regularization brake — ELI5

**Imagine a brake on the model's weights.** L2 regularization discourages very large weights. The model must balance fitting the training data against keeping weights reasonably small.

- **Small λ → weak brake → more flexibility → can overfit (high variance).**
- **Large λ → strong brake → less flexibility → can underfit (high bias).**
- **Goal: choose λ by cross-validation, not by maximizing or minimizing it.**

| Diagnostic signal | Possible action | Why |
|---|---|---|
| Train error far above achievable baseline (high bias) | **Decrease λ** if regularization is too strong | Give the model more freedom to fit training data |
| Train error low, CV error much higher (high variance) | **Increase λ** if overfitting is the issue | Penalize unnecessarily large weights |
| Both train and CV errors acceptable | Keep/tune λ by CV | Avoid fixing something that works |

**Course-style L2 objective:**

```text
J(w,b) = prediction_error + (lambda_ / (2*m)) * sum(w[j]**2)
```

Typically **bias/intercept b is not penalized**. λ changes the optimization objective; it does *not* directly multiply all weights or impose a hard weight limit.

### Tiny numeric penalty example

```python
import numpy as np

w = np.array([10.0, 2.0])
m = 10

for lambda_ in [0.0, 1.0, 10.0]:
    penalty = (lambda_ / (2 * m)) * np.sum(w ** 2)
    print(f"lambda={lambda_:>4}: L2 penalty={penalty:.1f}")

# lambda= 0.0: L2 penalty=0.0
# lambda= 1.0: L2 penalty=5.2
# lambda=10.0: L2 penalty=52.0
```

Here the weights are held fixed to **illustrate the penalty**. During real training, changing λ generally changes the learned weights too.

### Compare regularization strengths properly

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

results = []
for strength in [0.01, 0.1, 1.0, 10.0, 100.0]:
    model = make_pipeline(StandardScaler(), Ridge(alpha=strength))
    model.fit(X_train, y_train)   # scaler fits TRAIN only
    train_mse = mean_squared_error(y_train, model.predict(X_train))
    cv_mse = mean_squared_error(y_cv, model.predict(X_cv))
    results.append((strength, train_mse, cv_mse, model))

best_alpha, train_mse, cv_mse, best_model = min(
    results, key=lambda item: item[2]  # choose lowest CV error
)
print(f"Chosen Ridge alpha={best_alpha}, train MSE={train_mse:.3f}, CV MSE={cv_mse:.3f}")

# Evaluate on untouched X_test/y_test only AFTER selecting the configuration:
test_mse = mean_squared_error(y_test, best_model.predict(X_test))
```

**Naming trap:**
- Andrew Ng gradient descent **α (alpha)** = *learning rate*, the size of an update step.
- Andrew Ng regularization **λ (lambda)** = *penalty strength*.
- scikit-learn **Ridge(alpha=...)** = *regularization strength*, **not** learning rate.
- Ridge's numeric `alpha` convention is not identical to every textbook's λ scaling.

**Important nuance:** λ adjustments are possible remedies, not guarantees. Feature quality, model capacity, data quantity, and distribution shift also affect bias/variance. Use comparable train/CV metrics and a credible baseline.

**AI/LLM connection:** Weight decay in neural-network/LLM training is a related regularization idea; its implementation (e.g. AdamW decoupled weight decay) is not mathematically identical to every L2-penalty setup. Regularization is important for fine-tuning and generalization, but LLM engineering also requires data curation, retrieval evaluation, deployment, monitoring, and safety.

**Memorize:** high bias → consider **less** regularization (λ ↓); high variance → consider **more** regularization (λ ↑); **CV error decides**.
