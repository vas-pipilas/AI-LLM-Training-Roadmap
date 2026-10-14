# Fit, transform, and predict — explained like I am 5

*A study companion for Andrew Ng / DeepLearning.AI's Machine Learning Specialization, Course 2: Advanced Learning Algorithms, Week 3 optional lab “Model Evaluation and Selection,” supplied in the learning conversation.*

The [copy-friendly cheat sheet](SKLEARN_PREPROCESSING_CHEAT_SHEET.md) is the short version. This guide explains what happens inside the objects and why the order matters.

## 1. The whole story: a ruler and a guessing machine

Imagine measuring toy cars before teaching a machine to guess their prices.

First, choose a ruler. Then measure every car using that same ruler. Finally, teach the guessing machine what those measurements mean.

```text
Training: raw X → learn preparation rules → prepared X → learn model parameters using y
Later:    raw X → reuse preparation rules → prepared X → predict using saved parameters
```

The scaler is the ruler. The regression model or neural network is the guessing machine. Both can have a method called `fit`, but they learn different things.

> **Fit learns a rule. Transform uses a rule. Predict uses the trained model. Ask “which object?” before deciding what fit learned.**

## 2. What each method actually does

| Call | What changes inside the object? | What comes back? |
| --- | --- | --- |
| `scaler.fit(X_train)` | Saves training-column statistics | The fitted scaler itself |
| `scaler.transform(X)` | Reuses the saved statistics; does not refit them | An array of scaled features |
| `scaler.fit_transform(X_train)` | Fits, then transforms the training features | An array of scaled features |
| `poly.fit(X_train)` | Records input feature count and output mapping structure | The fitted polynomial transformer |
| `poly.transform(X)` | Applies that feature mapping | An array with polynomial columns |
| `model.fit(X_train_prepared, y_train)` | Learns prediction parameters from inputs and targets | The fitted sklearn model itself |
| `model.predict(X_prepared)` | Uses fitted prediction parameters | Predictions |

For `StandardScaler` and `PolynomialFeatures`, `fit_transform(X)` has the effect of `fit(X)` followed by `transform(X)`. Do not assume every specialized transformer works identically: some supervised encoders use special training-time behavior.

`fit` usually returns the object, not transformed data. So `X_scaled = scaler.fit(X)` would give you a scaler, not a feature matrix. Use `fit_transform`, or call `fit` and then `transform` separately.

The name `fit` is not exclusive to scaling or gradient descent. It means “prepare this estimator from data.” Imputers learn replacement values, PCA learns projection directions, and encoders learn categories. `transform` can fill missing values, rotate features, or create columns. A predictor does not necessarily implement `transform`, and a scaler does not predict prices.

## 3. StandardScaler: where do mean_ and scale_ come from?

For each feature column, the default scaler computes:

```text
mean μ = sum of the training values / number of training rows
variance = average of (value - μ)²
scale σ = square root of variance
transformed value z = (value - μ) / σ
```

Think “how far from the training average, measured in training standard deviations?” A result of 2 means two training standard deviations above the training mean. A negative result is below it. This does not force values into 0–1 or make the data normally distributed.

The learned attributes are `mean_`, `var_`, and `scale_`. The trailing underscore is sklearn's convention for fitted attributes. With defaults, `scale_` is the population standard deviation (`ddof=0`); a constant column uses scale 1 to avoid dividing by zero. Statistics are per column, not one number shared across the entire matrix. These details are documented in [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

### A tiny example that can be calculated by hand

Run this complete block. It needs NumPy and scikit-learn.

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

X_train = np.array([[10.0], [20.0], [30.0]])
X_cv = np.array([[40.0], [50.0]])
scaler = StandardScaler()

returned = scaler.fit(X_train)
assert returned is scaler
assert np.allclose(scaler.mean_, [20.0])
assert np.allclose(scaler.scale_, [np.sqrt(200.0 / 3.0)])

X_train_scaled = scaler.transform(X_train)
saved_mean = scaler.mean_.copy()
saved_scale = scaler.scale_.copy()
X_cv_scaled = scaler.transform(X_cv)

assert np.allclose(X_train_scaled.ravel(), [-1.224744871, 0.0, 1.224744871])
assert np.allclose(X_cv_scaled.ravel(), [2.449489743, 3.674234614])
assert np.array_equal(scaler.mean_, saved_mean)
assert np.array_equal(scaler.scale_, saved_scale)
assert np.allclose(X_cv_scaled, (X_cv - saved_mean) / saved_scale)
print(np.round(X_cv_scaled, 3))
# [[2.449]
#  [3.674]]
```

The mean is 20. The squared distances are 100, 0, and 100; their average is `200/3`. The scale is about 8.165. For CV value 40, we calculate `(40 - 20)/8.165 ≈ 2.449`. We do not ask CV data to invent a new ruler.

### Yes: fitting again on CV overwrites the ruler

Continue the previous block. This deliberately demonstrates the mistake; keep it away from the real model's scaler.

```python
wrong_scaler = StandardScaler().fit(X_train)
wrong_cv_scaled = wrong_scaler.fit_transform(X_cv)  # DELIBERATELY WRONG

assert np.allclose(wrong_scaler.mean_, [45.0])
assert np.allclose(wrong_scaler.scale_, [5.0])
assert np.allclose(wrong_cv_scaled.ravel(), [-1.0, 1.0])
assert np.allclose(wrong_scaler.transform([[40.0]]), [[-1.0]])
```

That answers the original question: **yes, another `fit` on the same scaler replaces its μ and σ with statistics from the new data.** `fit_transform` includes that new fit. It does not remember two rulers and automatically select the correct one. Ordinary `fit` resets the fitted state; incremental `partial_fit` is a separate API and is not the CV-evaluation operation.

The raw value 40 used to mean 2.449 to the model. Now it means -1. That changes the meaning of the model's input. Arrays already returned by the old transformation are not retroactively recalculated. The model's weights are not automatically updated either.

A single new example is especially revealing: refitting a default scaler on `[[40.0]]` makes its mean 40 and its scale 1, so that example becomes zero. Doing this separately for every new example destroys the information the model expected.

## 4. Is scaler.fit training? How is it different from model.fit?

In a broad sense, fitting a scaler learns preprocessing state from data. But it does not learn how to predict a target.

```text
scaler.fit(X_train):                  learn μ and σ from X
linear_model.fit(X_scaled, y_train):  learn w and b from X and y
neural_network.fit(X_scaled, y_train): learn W and b for the layers
```

For a linear model, `prediction = X_scaled @ w + b`. sklearn exposes these as `coef_` and `intercept_`. A neural network has weight matrices and bias vectors in its layers. The scaler's mean is not the model's bias; the scaler's scale is not a model weight.

The lab's `LinearRegression.fit` solves a least-squares problem; it does not normally run the gradient-descent training loop taught earlier. Keras `model.fit` typically performs batches, forward passes, loss calculation, backpropagation, and optimizer updates over epochs. Same method name, different implementation.

The distinction is easiest to see with numbers. This independent example learns `y = 2*x + 1`:

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

X = np.array([[10.0], [20.0], [30.0]])
y = np.array([21.0, 41.0, 61.0])
scaler = StandardScaler()
Z = scaler.fit_transform(X)
model = LinearRegression().fit(Z, y)

assert np.allclose(model.coef_, 2.0 * scaler.scale_)
assert np.isclose(model.intercept_, 41.0)
new_raw = np.array([[40.0]])
good_prediction = model.predict(scaler.transform(new_raw))
assert np.allclose(good_prediction, [81.0])
print(good_prediction)  # [81.]
```

The model's slope is about 16.33 in scaled units and its intercept is 41. The scaler still holds mean 20 and scale 8.165. They have separate jobs. `predict` calculates an answer using the learned parameters; it does not train again.

## 5. Train-only fitting: protect the exam

The lab divides 50 regression examples into 30 training, 10 CV, and 10 test rows: **60/20/20**. It first holds out 40%, then splits that remainder in half. The classification data has 200 rows, yielding 120/40/40.

| Set | Job | Allowed operations in this lab |
| --- | --- | --- |
| Training | Teach each candidate | Fit preprocessing and fit the model |
| CV / validation / dev | Choose degree or architecture | Transform, predict, compare errors |
| Test | Estimate the chosen model's generalization | Transform and evaluate after selection |
| New inputs | Use the chosen model | Transform and predict |

The lab's “cross validation set” is one held-out validation set, not k-fold cross-validation. With k-fold CV, each fold needs preprocessing fitted only on that fold's training portion.

**Data leakage** means information from held-out data influenced learning or selection in a way that compromises its evaluation role. Scaling the entire dataset before splitting lets held-out rows influence the training ruler, even without using their labels. Independently fitting a scaler on CV also uses held-out statistics and, worse, gives the trained model inputs in a different coordinate system. sklearn's [common pitfalls guide](https://scikit-learn.org/stable/common_pitfalls.html) explains train-only preprocessing and pipelines.

The exam analogy: training is practice, CV is choosing a study method, and test is the final exam. Choose methods using CV, then leave the final exam alone. Repeatedly changing the model after looking at test error turns the test into another development set.

For real data, choose the split to match deployment: time order for forecasting, group separation for repeated people/devices, and often stratification for classification. Random splitting is the lab's teaching setup, not a universal rule.

## 6. PolynomialFeatures: create columns before measuring them

A straight line may be too simple. We can give a linear regression model extra ingredients:

```text
degree 1: [x]           → prediction = w1*x + b
degree 2: [x, x²]       → prediction = w1*x + w2*x² + b
degree 3: [x, x², x³]   → prediction = w1*x + w2*x² + w3*x³ + b
```

The model is still linear in its learned coefficients, even though its prediction can curve as x changes. The transformer creates powers; it does not learn their usefulness from y. **Degree is a hyperparameter we choose**, not a parameter learned by `poly.fit`.

For this transformer, fitting establishes the input feature count and mapping metadata. The selected degree determines the expansion; no mean, scale, or predictive weights are learned. Details and multi-feature combinations are in [PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html).

This complete tiny block shows both one-feature and two-feature inputs:

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
mapped = poly.fit_transform(np.array([[2.0], [3.0]]))
assert np.array_equal(mapped, [[2.0, 4.0], [3.0, 9.0]])
assert np.array_equal(poly.transform([[4.0]]), [[4.0, 16.0]])

poly_two = PolynomialFeatures(degree=2, include_bias=False)
mapped_two = poly_two.fit_transform(np.array([[2.0, 3.0]]))
# [x1, x2, x1², x1*x2, x2²]
assert np.array_equal(mapped_two, [[2.0, 3.0, 4.0, 6.0, 9.0]])
print(poly_two.get_feature_names_out(["x1", "x2"]))
# ['x1' 'x2' 'x1^2' 'x1 x2' 'x2^2']
```

`include_bias=True` would also add a column of ones, the degree-zero feature. The lab uses `include_bias=False` because `LinearRegression` already learns an intercept by default. It does **not** remove the model's bias b. A constant ones column becomes zeros under centering anyway; a separate intercept is clearer here.

### Why the lab uses poly → scaler → model

In the lab, x is around 1600–3600, but x² is around 2.56–12.96 million. Expanding first lets the scaler learn a separate mean and scale for each generated column. This can help conditioning and gradient-based optimization. Scaling is especially useful for neural networks, distance-based methods, and regularized models; tree models generally do not need this kind of scaling. Plain least-squares regression is not an iterative gradient-descent method, but numerical conditioning can still matter.

```text
Training: poly.fit_transform → scaler.fit_transform → model.fit
CV/test/new: poly.transform  → scaler.transform     → model.predict
```

Squaring scaled x is a different feature representation from scaling x². Other consistent orders can define valid models, but they are not interchangeable preprocessing recipes. Follow this lab's order throughout training and prediction. Very high powers can overflow or be numerically unstable; scaling afterward cannot undo overflow that already happened.

## 7. Walk through the shapes

Rows are examples. Columns are features. Shape is `(number_of_examples, number_of_features)`.

For the lab's single input feature and degree 4:

| Step | Training | CV | Test | One new example |
| --- | --- | --- | --- | --- |
| Raw X | `(30, 1)` | `(10, 1)` | `(10, 1)` | `(1, 1)` |
| After poly | `(30, 4)` | `(10, 4)` | `(10, 4)` | `(1, 4)` |
| After scaler | `(30, 4)` | `(10, 4)` | `(10, 4)` | `(1, 4)` |
| Prediction, lab y shaped `(m, 1)` | `(30, 1)` | `(10, 1)` | `(10, 1)` | `(1, 1)` |

Each fitted scaler then has `mean_.shape == (4,)` and `scale_.shape == (4,)`. NumPy broadcasts these four column rules across all rows. Scaling changes values, not the number of rows or columns.

For one input feature, degree d gives d columns with `include_bias=False`. With several input features, interaction terms add more columns: two inputs at degree 2 give five, not two.

Use `np.array([[2500.0]])` for one new one-feature example. `np.array([2500.0])` is only 1-D. To turn a list of one-feature examples into rows, use `reshape(-1, 1)`. For one example with several features, use `reshape(1, -1)` instead. Preserve training feature order.

sklearn also accepts a 1-D target `(m,)`; its regression predictions then have shape `(m,)`. The lab uses `(m, 1)`, so predictions have that shape. Do not silently subtract `(m,)` from `(m, 1)`: NumPy may broadcast into an `(m, m)` matrix, producing a meaningless error score. See the [NumPy reference](NUMPY_FOR_ML.md).

## 8. Complete lab-style degree selection and prediction

Copy this whole block into a fresh file or notebook cell. It uses generated toy data so it runs without the lab's CSV files or `utils.py`. It follows the supplied lab's structure, but its selected degree and scores are not the lab's reported results.

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(7)
x = np.linspace(-3.0, 3.0, 50).reshape(-1, 1)
y = 2.0 + 0.5 * x - 0.8 * x**2 + rng.normal(0, 0.3, x.shape)

x_train, x_rest, y_train, y_rest = train_test_split(
    x, y, test_size=0.40, random_state=1
)
x_cv, x_test, y_cv, y_test = train_test_split(
    x_rest, y_rest, test_size=0.50, random_state=1
)
assert (x_train.shape, x_cv.shape, x_test.shape) == ((30, 1), (10, 1), (10, 1))

polys, scalers, models = [], [], []
train_mses, cv_mses = [], []
for degree in range(1, 11):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_mapped = poly.fit_transform(x_train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_mapped)
    model = LinearRegression().fit(X_train_scaled, y_train)

    X_cv_scaled = scaler.transform(poly.transform(x_cv))
    train_mses.append(mean_squared_error(y_train, model.predict(X_train_scaled)) / 2)
    cv_mses.append(mean_squared_error(y_cv, model.predict(X_cv_scaled)) / 2)
    polys.append(poly)
    scalers.append(scaler)
    models.append(model)

best_index = int(np.argmin(cv_mses))
degree = best_index + 1
assert best_index == degree - 1

# All three objects must come from the SAME candidate.
best_poly = polys[degree - 1]
best_scaler = scalers[degree - 1]
best_model = models[degree - 1]

def predict_raw(X_raw):
    X_mapped = best_poly.transform(X_raw)
    X_scaled = best_scaler.transform(X_mapped)
    return best_model.predict(X_scaled)

yhat_test = predict_raw(x_test)
test_mse = mean_squared_error(y_test, yhat_test) / 2
new_predictions = predict_raw(np.array([[0.0], [1.5]]))
assert yhat_test.shape == (10, 1)
assert new_predictions.shape == (2, 1)
print(f"Selected degree: {degree}")
print(f"Train: {train_mses[best_index]:.4f}, CV: {cv_mses[best_index]:.4f}")
print(f"Test: {test_mse:.4f}")
print("New predictions:", new_predictions.ravel())
```

`range(1, 11)` tries degrees 1 through 10. Python list position 0 holds degree 1, so degree d lives at index `d - 1`. This correspondence holds only because degrees are consecutive and appended in that order. For degrees `[1, 2, 4, 8]`, keep `best_index` and select `degrees[best_index]` instead.

Each candidate needs its **own** fitted polynomial transformer, scaler, and model. Reusing a single scaler in the loop and appending it repeatedly would store multiple references to the same object; later fits would overwrite the earlier candidates' rules. Even if two candidates have the same number of columns, they must use their own fitted statistics.

The lab chooses the smallest recorded CV error with `np.argmin(cv_mses)`. Training error helps diagnose underfitting or overfitting; it is not the degree-selection score. The supplied lab reports degree 4 and test half-MSE about 104.63. Those are source-lab results, not results reproduced by the toy data above.

The lab calls `mean_squared_error(...) / 2` MSE to match its `1/(2*m)` cost convention. sklearn's ordinary MSE uses `1/m`. Dividing all scores by two does not change which degree wins. Do not divide only some candidates' scores.

After freezing the choice, evaluate test data with the chosen triple. New inputs follow exactly the same route, but have no y available for computing an error. If deployment later calls for refitting on training plus CV data, refit the entire chosen preparation/model together; that is a deliberate new training stage, not part of the lab's evaluation path. Keep test data held out.

## 9. Pipeline: put the ruler and machine in one box

A [sklearn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) keeps the steps together. `fit` fits/transforms intermediate steps and fits the final model. `predict` transforms through the saved steps and calls the final model's prediction method.

Continue the complete degree-selection block above:

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
    ("scaler", StandardScaler()),
    ("model", LinearRegression()),
])
pipe.fit(x_train, y_train)  # Give the pipeline RAW inputs.
pipeline_cv = pipe.predict(x_cv)
assert np.allclose(pipeline_cv, predict_raw(x_cv))
assert np.allclose(pipe.named_steps["scaler"].mean_, best_scaler.mean_)
print(pipe.predict(np.array([[1.5]])))
```

Do not scale inputs before giving them to this pipeline: it already expands and scales them. A pipeline prevents many preparation mistakes, but it cannot protect a test set that you passed into `fit` yourself.

For k-fold tuning, pass the whole unfitted pipeline to a tool such as `GridSearchCV`, tuning `poly__degree`. It fits preprocessing separately inside each training fold. Scaling all rows first and then running CV would leak fold-validation information.

Keep the fitted preprocessing with the trained model when saving a deployment artifact. Recreating an unfitted scaler from the class name is not enough: the learned training statistics are part of the prediction recipe.

## 10. Binary classification: logits → sigmoid → threshold → error

The lab's binary neural networks have a linear final output and use `BinaryCrossentropy(from_logits=True)` during training. Their raw predictions are **logits**, unrestricted real scores, not probabilities. The loss handles logits internally. At evaluation time the lab applies sigmoid, then thresholding.

```text
raw X → training-fitted scaler.transform → network.predict → logits
logits → sigmoid → probabilities → threshold at 0.5 → labels 0/1
labels versus true y → fraction misclassified
```

For a logit z, sigmoid is `1/(1 + exp(-z))`. Negative logits give probabilities below 0.5; positive logits give probabilities above it. With the lab's `>= 0.5` rule, logit 0 becomes class 1.

This independent NumPy example isolates the evaluation arithmetic without installing TensorFlow:

```python
import numpy as np

logits = np.array([[-2.0], [0.0], [2.0], [-1.0], [1.0]])
probabilities = 1.0 / (1.0 + np.exp(-logits))
predictions = (probabilities >= 0.5).astype(int)
y_true = np.array([[1], [1], [1], [1], [1]])
assert predictions.shape == y_true.shape
assert np.array_equal(predictions.ravel(), [0, 1, 1, 0, 1])
fraction_error = np.mean(predictions != y_true)
assert np.isclose(fraction_error, 0.4)
print(fraction_error)  # 0.4 = 40% wrong; accuracy is 60%.
```

For extreme logits, use a numerically stable implementation such as the lab's `tf.math.sigmoid`; the small values above keep the NumPy formula safe.

For the actual Keras lab model, use `logits = model.predict(X_scaled)`, then `probabilities = tf.math.sigmoid(logits).numpy()`. Compare labels and predictions with matching shapes. Choose the architecture using CV classification error; report the chosen model's test classification error afterward. The supplied lab reports a CV tie between models 2 and 3 and chooses model 3; preferring a smaller tied model is also reasonable. Do not break the tie using test error.

Do not apply sigmoid twice if the model already outputs sigmoid probabilities. sklearn `LogisticRegression.predict` returns class labels, while `predict_proba` returns class probabilities; neither should be treated as the Keras lab's raw logits. Threshold selection, if tuned, belongs on validation data, not the final test set.

## 11. How this connects to AI and LLM work

The reusable principle is: **learn preparation rules at the proper stage and apply the same representation at inference.** This matters in tabular ML, neural networks, text classification, and deployed AI systems.

An LLM tokenizer, however, is not a StandardScaler. It maps text into tokens and IDs using a vocabulary and tokenization rules. Training a tokenizer can learn that vocabulary; using a pretrained tokenizer normally applies an existing one. Its API need not use sklearn's method names. Token IDs are discrete identifiers, not measurements to z-score before feeding them into an LLM.

Use the tokenizer associated with the model, including its special-token and formatting conventions. A mismatch changes the meaning of the model's inputs, much as using the wrong scaler changes a numeric model's inputs. This is an analogy about consistency, not identical mathematics. Tokenizing text does not update the LLM's weights. LLM training/fine-tuning updates model parameters in a separate process. Internal LayerNorm or RMSNorm is also a different operation from this training-dataset StandardScaler.

## 12. Pitfalls I want to recognize quickly

| Mistake | Why it hurts | Better habit |
| --- | --- | --- |
| `scaler.fit_transform(X_cv)` | Replaces the training ruler | `scaler.transform(X_cv)` |
| Scale all data, then split | Held-out rows influence preprocessing | Split first, fit on training |
| `model.predict(raw_X)` after scaled training | Wrong input representation | Use the fitted preparation steps |
| Scale an already scaled array | Applies preparation twice | Start prediction from raw inputs |
| Mix `polys[3]`, `scalers[4]`, `models[3]` | Candidate mismatch or shape failure | Select the same index for all three |
| Refit poly on new inputs | Can reset the feature schema | Reuse its fitted mapping |
| Assume scaled CV has mean 0 | Only training statistics define centering | Expect CV mean/variance to differ |
| Forget 2-D X | sklearn expects rows and columns | Check `X.shape` before fitting/predicting |
| Compare `(m,)` and `(m, 1)` | Broadcasting can make `(m, m)` | Align target/prediction shapes |
| Pick smallest training error | High degree can memorize training data | Choose using CV error |
| Keep tuning after test evaluation | Test becomes a development set | Freeze decisions before test |
| Treat every `predict` output as probability | Regressors, labels, and logits differ | Inspect the model's output contract |

Scaling X does not automatically scale y. In this lab, y stays in its original units and predictions remain in those units. If a different workflow scales targets too, it must separately reverse that target transformation before interpreting predictions.

## 13. Self-checks

1. If I call `scaler.fit(X_cv)`, does it keep the training mean?
2. Does `poly.fit` choose the best degree by looking at y?
3. Why do we use `degree - 1` in the lab's lists?
4. For `(30, 1)` input and degree 3, what shape reaches the model?
5. Does `include_bias=False` prevent LinearRegression from learning b?
6. Why can a CV scaled value be 4 even though training was standardized?
7. Where does sigmoid go for the lab's binary models?

<details>
<summary>Answers — open after trying</summary>

1. No. A fresh ordinary fit replaces the fitted statistics with CV statistics.
2. No. Degree is selected by comparing candidate CV errors; poly creates the requested mapping.
3. Degrees start at 1; list indices start at 0, with consecutive append order.
4. `(30, 3)` after polynomial expansion and scaling.
5. No. It omits a constant input column; the model still learns its own intercept by default.
6. CV is measured using the training mean/scale, not normalized to its own distribution. Values are not clipped.
7. After prediction logits, before thresholding and label-error calculation. The training loss is configured with `from_logits=True`.

</details>

## 60-second refresher

```text
FIT:           learn this object's state from training data.
TRANSFORM:     apply fitted preparation rules to any compatible X.
FIT_TRANSFORM: learn preparation rules, then apply them to training X.
MODEL.FIT:     learn prediction parameters from prepared X and y.
PREDICT:       use the trained model; preserve its input recipe.

Train:       poly.fit_transform → scaler.fit_transform → model.fit
CV/test/new: poly.transform     → scaler.transform     → model.predict
Selection:   lowest CV error → matching objects → final test error
```

**Source note:** The primary teaching source is the user-supplied Week 3 optional lab “Model Evaluation and Selection.” Its split, polynomial loop, degree-4 result, half-MSE convention, neural-network comparison, and binary evaluation path ground this guide. Tiny numerical examples, the synthetic runnable dataset, pipeline comparison, and self-checks are added study aids. The linked official sklearn documentation supports the library behavior. This is a study resource, not a claim of course completion or mastery.
