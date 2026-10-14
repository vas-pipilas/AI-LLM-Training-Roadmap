# sklearn preprocessing — copy-friendly cheat sheet

Companion to the [fit / transform / predict deep dive](FIT_TRANSFORM_PREDICT_DEEP_DIVE.md), based on Course 2 Week 3's optional lab “Model Evaluation and Selection.”

## Remember the jobs

| Operation | Remember |
| --- | --- |
| `scaler.fit(X_train)` | Learn training-column mean/scale; return the scaler |
| `scaler.transform(X)` | Return scaled X using saved statistics |
| `scaler.fit_transform(X_train)` | Fit, then return transformed training X |
| `poly.fit_transform(X_train)` | Establish feature mapping, then create polynomial columns |
| `model.fit(X_prepared, y_train)` | Learn prediction weights/biases |
| `model.predict(X_prepared)` | Use those parameters to predict |

**Fit training only. Transform CV, test, and new inputs.** A fresh scaler fit replaces its state; it does not preserve the old ruler. Transform is not limited to scaling: imputers, encoders, PCA, and polynomial features also transform data.

## Paste and run: safest default with Pipeline

This complete example uses toy data, a 60/20/20 split, CV degree selection, and the lab's half-MSE convention. Needs NumPy and scikit-learn; no lab files required.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(7)
X = np.linspace(-3, 3, 50).reshape(-1, 1)
y = 2 + 0.5 * X - 0.8 * X**2 + rng.normal(0, 0.3, X.shape)
X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, test_size=0.40, random_state=1
)
X_cv, X_test, y_cv, y_test = train_test_split(
    X_rest, y_rest, test_size=0.50, random_state=1
)

degrees = list(range(1, 11))
pipes, cv_errors = [], []
for degree in degrees:
    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])
    pipe.fit(X_train, y_train)
    cv_errors.append(mean_squared_error(y_cv, pipe.predict(X_cv)) / 2)
    pipes.append(pipe)

best_index = int(np.argmin(cv_errors))
best = pipes[best_index]
print("Selected degree:", degrees[best_index])
print("Test half-MSE:", mean_squared_error(y_test, best.predict(X_test)) / 2)
print("New predictions:", best.predict(np.array([[0.0], [1.5]])).ravel())
```

Give this pipeline **raw X**. It already performs `poly → scaler → model`. During k-fold tuning, pass the entire unfitted pipeline to the CV tool so each fold fits its own preprocessing. Split strategy should match the real task; the random split above follows the lab.

## Manual equivalent for one candidate

Continue the block above; use these steps when reading the lab's code:

```python
degree = degrees[best_index]
poly = PolynomialFeatures(degree=degree, include_bias=False)
scaler = StandardScaler()
model = LinearRegression()

X_train_mapped = poly.fit_transform(X_train)
X_train_scaled = scaler.fit_transform(X_train_mapped)
model.fit(X_train_scaled, y_train)

cv_predictions = model.predict(scaler.transform(poly.transform(X_cv)))
test_predictions = model.predict(scaler.transform(poly.transform(X_test)))
new_predictions = model.predict(scaler.transform(poly.transform([[1.5]])))
assert np.allclose(cv_predictions, best.predict(X_cv))
```

In the lab's consecutive degree-1-through-10 lists, use **the same index** for `polys[degree - 1]`, `scalers[degree - 1]`, and `models[degree - 1]`. Instantiate fresh objects for each candidate. Select with CV error; use test error after the choice is frozen.

## Tiny scaler and polynomial reminders

- Default scaling: `Z = (X - scaler.mean_) / scaler.scale_`, per column. Scale uses population standard deviation (`ddof=0`); constant columns use scale 1. [StandardScaler documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
- Training `[10, 20, 30]`: mean 20, scale about 8.165. New 40 becomes about 2.449. Fitting again on `[40, 50]` changes mean to 45 and scale to 5; 40 then becomes -1. **Do not refit for evaluation.**
- One feature, degree 2: `[[2], [3]] → [[2, 4], [3, 9]]`. Two features, degree 2: `[a, b] → [a, b, a², a*b, b²]`.
- `include_bias=False` omits the ones column; `LinearRegression` still learns an intercept by default. Degree is chosen by CV, not learned by `poly.fit`. [PolynomialFeatures documentation](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html).
- With one feature and degree d: `(m, 1) → (m, d) → (m, d)`. Scaler statistics have shape `(d,)`.
- One new one-feature row: `np.array([[2500.0]])`. Many one-feature rows: `reshape(-1, 1)`. One multi-feature row: `reshape(1, -1)`.
- Lab targets `(m, 1)` produce regression predictions `(m, 1)`; targets `(m,)` produce `(m,)`. Match shapes before arithmetic or comparison.
- sklearn MSE is `sum(error²)/m`; the lab uses `mean_squared_error(y, yhat)/2`. Ranking is unchanged if every candidate uses the same convention.

## Binary lab evaluation: logits → sigmoid → threshold → error

Independent NumPy demo; for the actual Keras lab use `tf.math.sigmoid` on `model.predict(X_scaled)` logits.

```python
import numpy as np

logits = np.array([[-2.0], [0.0], [2.0], [-1.0], [1.0]])
probabilities = 1 / (1 + np.exp(-logits))  # Safe for these small demo logits.
labels = (probabilities >= 0.5).astype(int)
y_true = np.array([[1], [1], [1], [1], [1]])
assert labels.shape == y_true.shape
assert np.isclose(np.mean(labels != y_true), 0.4)
print(np.mean(labels != y_true))  # 40% wrong
```

The lab trains a linear output with `BinaryCrossentropy(from_logits=True)`. Do not sigmoid outputs that are already probabilities. sklearn classifier `predict` usually returns labels; `predict_proba` returns probabilities. Choose architecture/threshold on CV, then report test classification error.

## Before predicting

- Use the saved preparation/model together, in the same feature order.
- Never `fit_transform` CV/test/new inputs; never scale the whole dataset before splitting.
- Never pass raw X to a manually trained scaled model, or pre-scale X for a pipeline that scales it again.
- CV scaled columns need not have mean 0 or standard deviation 1; they use the training ruler.
- Do not compare `(m,)` with `(m, 1)` and accidentally create `(m, m)` through broadcasting.
- High degrees can overfit and create huge values. Low training error is not proof of generalization.
- LLM tokenizers map text to token IDs using model-specific vocabulary/rules. Tokenizing is not model training, and token IDs are not numeric measurements to StandardScale.

Return to the [deep dive](FIT_TRANSFORM_PREDICT_DEEP_DIVE.md) for worked arithmetic, overwrite behavior, the matching-degree loop, pitfalls, and self-checks. See also sklearn's [leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html) and [Pipeline API](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).
