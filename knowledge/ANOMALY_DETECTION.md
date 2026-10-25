# Anomaly Detection — why the lab works, from ELI5 to code

**Study note, 25 October 2026.** This is the dedicated theory and lab explanation for Andrew Ng's Machine Learning Specialization anomaly-detection lab. For reusable syntax, start with the [canonical coding cheat sheet](PYTHON_ML_CODING_CHEAT_SHEET.md#12d-anomaly-detection) and its [Boolean counting reference](PYTHON_ML_CODING_CHEAT_SHEET.md#boolean-counting-and-guarded-division). [NumPy for ML](NUMPY_FOR_ML.md) provides the focused array reference; [Unsupervised ML and K-means](UNSUPERVISED_LEARNING_AND_KMEANS.md) explains the neighboring clustering task.

**Completed-lab record:** the user's supplied handoff reports best epsilon **8.990853e-05**, best F1 **0.875**, and **all unit tests passed**, per the user's screenshot. This update records that evidence; the screenshot is not available here for independent inspection, and the original course data, notebook and grader were not rerun. The tiny arrays below are teaching examples, not reproductions of the course result. Passing the lab establishes the recorded implementation success, not yet effortless conceptual recall.

## Navigation

- [1. The whole story before the formulas](#1-the-whole-story-before-the-formulas)
- [2. Mean, spread and squared deviations](#2-mean-spread-and-squared-deviations)
- [3. Gaussian density and multiple features](#3-gaussian-density-and-multiple-features)
- [4. Training, validation and epsilon](#4-training-validation-and-epsilon)
- [5. Why TP, FP and FN are necessary](#5-why-tp-fp-and-fn-are-necessary)
- [6. Exercise 1 — estimate_gaussian](#6-exercise-1--estimate_gaussian)
- [7. Exercise 2 — select_threshold](#7-exercise-2--select_threshold)
- [8. Mistakes, edge cases and practical limits](#8-mistakes-edge-cases-and-practical-limits)
- [9. Explain it without the notebook](#9-explain-it-without-the-notebook)
- [Sources and verification](#sources-and-verification)

## 1. The whole story before the formulas

Imagine a box of mostly normal toy engines. Measure two things about each: how fast it spins and how hot it gets. Learn the usual center and spread of each measurement. A new engine with very unusual measurements gets a low "how typical is this?" score.

Now choose when to raise an alarm. Too sensitive, and normal engines trigger alarms. Too relaxed, and broken engines slip through. A small labeled practice set tells us which engines really are abnormal. Try different alarm settings, count successes and mistakes, and keep the setting with the best chosen score.

| Story | Lab object | Job |
| --- | --- | --- |
| Measurements of mostly normal engines | `X_train`, shape `(m, n)` | Learn what normal looks like |
| Typical center and spread | `mu`, `var`, each `(n,)` | Parameters learned from training |
| Typicality score for each practice engine | `p_val`, shape `(m_val,)` | Continuous density, not a class |
| Alarm setting | `epsilon`, one number | Turn scores into decisions |
| Alarm decisions | `prediction = p_val < epsilon` | True means predicted anomaly |
| Known practice answers | `y_val`, shape `(m_val,)` | 1 = actual anomaly, 0 = normal |
| Counts and report card | TP/FP/FN → precision/recall → F1 | Compare alarm settings |

The dependency chain is:

```text
training X → mu and var → density scores p_val
candidate epsilon + p_val → prediction
prediction + y_val → TP, FP, FN (and TN)
TP, FP, FN → precision and recall → F1
compare F1 across candidates → best_epsilon, best_F1
new X + saved mu/var + chosen epsilon → future alarms
```

Only epsilon and the resulting predictions/counts/metrics change inside the threshold loop. The training statistics and validation densities stay fixed.

## 2. Mean, spread and squared deviations

The mean is the balance point. For measurements `[1, 3, 5]`, add them and divide by three: `mu = 9/3 = 3`.

Subtract the mean to see each measurement's displacement: `[-2, 0, 2]`. Adding these signed deviations gives zero, even though the measurements differ. Negative and positive offsets cancel. Squaring turns both directions into positive contributions and gives larger offsets more weight: `[4, 0, 4]`.

Variance is their average: `(4+0+4)/3 = 8/3 ≈ 2.667`. Standard deviation is `sqrt(8/3) ≈ 1.633`. Variance has squared measurement units; standard deviation has the original units.

For feature j and m training rows:

$$
\mu_j = \frac{1}{m}\sum_{i=1}^{m}x_j^{(i)},\qquad
\sigma_j^2 = \frac{1}{m}\sum_{i=1}^{m}(x_j^{(i)}-\mu_j)^2.
$$

The superscript (i) identifies the example; the subscript j identifies its feature. They are mathematical labels, not Python indexing syntax. Python indices begin at zero.

`var` and `sigma2` are common names for the same variance quantity, sigma squared. Do not square `var` again in the Gaussian formula. This lab uses divisor m, the Gaussian maximum-likelihood estimate: `np.var(X, axis=0)` defaults to `ddof=0`. A sample-variance estimator with `ddof=1` divides by m−1 and gives a different answer.

For two features:

```python
import numpy as np

X = np.array([[1., 2.], [3., 4.], [5., 6.]])
mu = np.mean(X, axis=0)                 # [3., 4.]
var = np.var(X, axis=0)                 # [8/3, 8/3]
centered = X - mu                       # [[-2,-2], [0,0], [2,2]]
np.testing.assert_allclose(centered, [[-2., -2.], [0., 0.], [2., 2.]])
np.testing.assert_allclose(var, np.mean(centered ** 2, axis=0))
assert mu.shape == var.shape == (2,)
```

`axis=0` reduces the row dimension, leaving one result per column. `X.shape[0]` is the count 3; `X[0]` is the row `[1., 2.]`. One describes the container; the other selects its contents.

**Why does subtracting the mean not change variance?** Moving every toy two meters left moves their center too; it does not change their distances from that center. For any fixed feature offset c:

$$
\operatorname{Var}(X-c)=\operatorname{Var}(X),\quad
(X-c)-(\mu-c)=X-\mu.
$$

Thus `np.var(X - mu, axis=0)` equals `np.var(X, axis=0)`, apart from floating-point effects. But `np.mean(X ** 2, axis=0)` is not generally variance: it measures squared distance from zero, not from the mean. For `[1,3,5]`, it is `35/3`, not `8/3`. Centering gives mean zero, not variance zero. Dividing by standard deviation afterward changes variance to one for nonconstant features.

## 3. Gaussian density and multiple features

A Gaussian is a bell-shaped model: values near the mean have high density; distant values have low density. Variance controls its width. Small variance makes a narrow, tall bell, so a modest displacement can be unusual. Large variance makes a wider bell, so the same displacement can be ordinary.

For one feature, with positive variance:

$$
p(x_j;\mu_j,\sigma_j^2)=
\frac{1}{\sqrt{2\pi\sigma_j^2}}
\exp\left(-\frac{(x_j-\mu_j)^2}{2\sigma_j^2}\right).
$$

Read the pieces: subtract the center → square the displacement → compare it with the spread → negate and exponentiate → apply the bell's normalization factor. At the mean the exponent is zero; farther away the exponential decreases.

**Density is not probability at an exact point.** For a continuous variable, probability over an interval is the area under the curve on that interval. The probability of one exact point is zero. The curve's height can exceed one when the bell is narrow; its total area is one. Consequently `p_val[i] = 0.00008` does not mean "0.008% chance of being normal" or an anomaly probability. Density depends on measurement units and feature dimension.

The lab models each feature separately and multiplies the feature densities:

$$
p(\mathbf{x})=\prod_{j=1}^{n}p(x_j;\mu_j,\sigma_j^2).
$$

This is a factorized Gaussian model: it assumes feature independence in the modeled normal distribution. It needs n means and n variances rather than a full covariance matrix. A point with an unlikely measurement tends to get a low product score.

For two independent standard-normal features, density at `[0,0]` is about `0.39894² = 0.15915`; at `[3,0]` it is about `0.004432 × 0.39894 = 0.001768`. With epsilon 0.01, the second point is flagged.

The assumption can miss unusual **combinations**: if two sensors normally rise together, "sensor A high, sensor B low" may be strange even when each value alone is ordinary. A full multivariate Gaussian models covariance; it is distinct from the independent product here. Feature engineering, transformations or other models may be needed when marginals are strongly non-Gaussian or relationships matter.

Teaching implementation of the density stage (the course supplies a helper for this stage; this is not a third graded exercise):

```python
import numpy as np

mu = np.array([0., 0.])
var = np.array([1., 1.])
X_val = np.array([[0., 0.], [3., 0.]])  # two examples, two features
feature_density = np.exp(-((X_val - mu) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)
p_val = np.prod(feature_density, axis=1)  # multiply features within each row
np.testing.assert_allclose(p_val, [1 / (2*np.pi), np.exp(-4.5) / (2*np.pi)])
assert p_val.shape == (2,)
assert (p_val < 0.01).tolist() == [False, True]
```

`X_val - mu` broadcasts `(n,)` over `(m_val,n)`; each column uses its own saved mean. `np.prod(..., axis=1)` combines feature densities into one score per example. Summing or multiplying over axis=0 here would mix different examples.

## 4. Training, validation and epsilon

**Training learns normality.** Fit mu and var on the mostly normal training examples, without anomaly labels in this fitting step. Heavy anomaly contamination can distort those statistics.

**Cross-validation chooses the alarm setting.** Compute validation densities using the saved training parameters. Validation contains normal and anomalous examples with known labels. Do not refit mu/var on it. Labels help choose epsilon even though fitting the density model is unsupervised; the complete selection procedure uses labeled validation information.

**A final untouched test set estimates performance** after model/features/epsilon are frozen. Repeatedly choosing settings on validation can overfit validation. Future inference uses measurements, saved parameters and epsilon; it does not need the future true labels to raise an alarm.

The decision is `p(x) < epsilon`. Low typicality means anomaly. This reverses the direction of a common sigmoid rule where a high positive-class probability predicts class 1.

- Raise epsilon: more scores fall below it, so more alarms. TP and FP cannot decrease on a fixed dataset; precision need not improve.
- Lower epsilon: fewer alarms; some real anomalies may be missed.
- Equal to epsilon: the strict `<` rule says normal.
- Epsilon is a density cutoff, not learning rate alpha, regularization lambda, or the Gaussian variance.

Small values are expected for some units/dimensions; there is no universal epsilon. `8.990853e-05 = 0.00008990853`. Its scale alone says nothing about detection quality.

## 5. Why TP, FP and FN are necessary

**The stumbling block:** "I followed the instructions and wrote the formulas, but why do I need these counts at all?"

A density score tells us whether measurements look unusual under the model. It does not tell us whether an alarm is correct. A prediction says "alarm"; a label says "actually abnormal." We must compare both to know whether an alarm setting is useful.

Think of a smoke alarm. Counting only the number of beeps rewards an alarm that never stops. Counting only correct decisions can reward an alarm that never beeps when fires are rare. We need to separate **real fires found**, **false alarms**, and **fires missed**. Those are TP, FP and FN.

Here "positive" means anomaly/alarm, not "good":

| | Actual anomaly: y=1 | Actual normal: y=0 |
| --- | --- | --- |
| Predict anomaly: True / 1 | TP: true positive, real anomaly caught | FP: false positive, false alarm |
| Predict normal: False / 0 | FN: false negative, missed anomaly | TN: true negative, normal correctly left alone |

The word true/false says whether the decision is correct; positive/negative says which class was predicted.

```text
Precision = TP / (TP + FP)
            correct alarms / all alarms

Recall    = TP / (TP + FN)
            anomalies caught / all actual anomalies

F1        = 2 * precision * recall / (precision + recall)
          = 2*TP / (2*TP + FP + FN), when the denominator is nonzero
```

Precision asks "Can I trust an alarm?" Recall asks "How many real problems did I catch?" F1 is their harmonic mean: it is high only when both are reasonably high. Precision 1 and recall 0.1 gives F1 about 0.182, not their arithmetic mean 0.55. In this lab F1 supplies **one comparison number for each candidate epsilon**.

TP/FP/FN are sufficient bookkeeping for this metric, not magic ingredients needed by every possible detector. We could calculate equivalent ratios directly from masks or choose another objective. The point is to measure both false alarms and missed anomalies before selecting a cutoff. TN is part of the full confusion matrix but does not appear in positive-class precision, recall or F1.

For 990 normal examples and 10 anomalies, predicting everything normal gets 99% accuracy, but catches zero anomalies. Recall and the lab-convention F1 are zero. This is why accuracy can mislead here.

### One tiny dataset, three alarm settings

These are invented density scores, sorted just to make the trace easy:

```python
import numpy as np

p_val = np.array([0.01, 0.04, 0.08, 0.20, 0.70])
y_val = np.array([1,    0,    1,    1,    0])
prediction = p_val < 0.10
assert prediction.tolist() == [True, True, True, False, False]
tp = np.sum((prediction == 1) & (y_val == 1))  # indices 0 and 2: 2
fp = np.sum((prediction == 1) & (y_val == 0))  # index 1: 1
fn = np.sum((prediction == 0) & (y_val == 1))  # index 3: 1
tn = np.sum((prediction == 0) & (y_val == 0))  # index 4: 1
assert (tp, fp, fn, tn) == (2, 1, 1, 1)
```

| Epsilon | Prediction | TP / FP / FN / TN | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| 0.03 | T F F F F | 1 / 0 / 2 / 2 | 1 | 1/3 | 0.5 |
| 0.10 | T T T F F | 2 / 1 / 1 / 1 | 2/3 | 2/3 | 2/3 |
| 0.30 | T T T T F | 3 / 1 / 0 / 1 | 3/4 | 1 | 6/7 ≈ 0.857 |

Among these three choices, 0.30 wins on F1. It catches one more anomaly without adding a false alarm compared with 0.10. This table is illustrative; the lab scans roughly 1000 thresholds, not just three.

**Understanding versus following instructions:** remembering `tp = ...` is syntax recall. Understanding means being able to predict which row changes when epsilon changes, why that changes precision/recall, and why the resulting F1 makes one epsilon preferable. The loop does not modify Gaussian parameters or use gradient descent. It compares candidate decision rules using known answers.

## 6. Exercise 1 — estimate_gaussian

The two key graded lines estimate parameters, not anomaly labels:

```python
import numpy as np

def estimate_gaussian(X):
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    return mu, var

X_train = np.array([[1., 2.], [3., 4.], [5., 6.]])
mu, var = estimate_gaussian(X_train)
np.testing.assert_allclose(mu, [3., 4.])
np.testing.assert_allclose(var, [8/3, 8/3])
assert mu.shape == var.shape == (2,)
```

| Line | Mathematical / practical meaning |
| --- | --- |
| `def estimate_gaussian(X):` | Receive m rows with n features; no labels needed |
| `mu = np.mean(X, axis=0)` | Apply the feature mean formula to each column |
| `var = np.var(X, axis=0)` | Apply mean squared deviations with divisor m |
| `return mu, var` | Return both vectors; caller unpacks them |

The lab scaffold may also unpack `m, n = X.shape` and initialize output arrays. Those lines establish dimensions/storage; vectorized assignments replace the placeholders with computed vectors. Between the graded markers, the two statistic lines are sufficient.

**What happens next outside this exercise:** the notebook's density helper uses these learned parameters to compute training/validation densities. Exercise 2 receives already computed `p_val`, not the raw feature matrix.

## 7. Exercise 2 — select_threshold

Below is a full teaching version of the lab-style search. The standard varying-score path uses the lab's arange sweep. A small explicit branch handles equal scores so step size cannot be zero; that branch is an extension for this reference, not a claim about the user's submitted cell. Inputs here must be finite, nonempty, aligned 1-D arrays with binary labels.

```python
import numpy as np

def select_threshold(y_val, p_val):
    best_epsilon = 0
    best_F1 = 0
    step_size = (max(p_val) - min(p_val)) / 1000

    if step_size == 0:  # extension: identical scores need explicit candidates
        score = float(p_val[0])
        candidates = [score, np.nextafter(score, np.inf)]
    else:
        candidates = np.arange(min(p_val), max(p_val), step_size)

    for epsilon in candidates:
        prediction = p_val < epsilon
        tp = np.sum((prediction == 1) & (y_val == 1))
        fp = np.sum((prediction == 1) & (y_val == 0))
        fn = np.sum((prediction == 0) & (y_val == 1))

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

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon

    return best_epsilon, best_F1

p_val = np.array([0.01, 0.04, 0.08, 0.20, 0.70])
y_val = np.array([1, 0, 1, 1, 0])
epsilon, F1 = select_threshold(y_val, p_val)
assert 0.20 < epsilon <= 0.70
np.testing.assert_allclose(F1, 6/7)
```

### Read every stage in plain language

| Code | Why it exists |
| --- | --- |
| `best_epsilon = 0; best_F1 = 0` | Remember the best result seen so far |
| `(max(p_val)-min(p_val))/1000` | Divide the observed density range into about 1000 steps |
| `np.arange(start, stop, step)` | Generate candidate cutoff values; stop is normally excluded |
| `for epsilon in candidates:` | Try each alarm setting |
| `prediction = p_val < epsilon` | Compare every score with this one cutoff, yielding a Boolean vector |
| `prediction == 1` | Identify predicted alarms; True compares equal to 1 |
| `y_val == 1` | Identify actual anomalies |
| `(prediction == 1) & (y_val == 1)` | Identify rows satisfying both conditions |
| `np.sum(...)` | Count those rows: booleans contribute 1 or 0 |
| FP mask: prediction=1, y=0 | Count alarms on normal examples |
| FN mask: prediction=0, y=1 | Count anomalies the detector missed |
| First if/else | No alarms means precision's denominator is zero; use 0 by convention |
| Second if/else | No actual anomalies means recall is undefined; use 0 by convention |
| Third if/else | Both metrics zero would give 0/0 in F1; use 0 |
| `if F1 > best_F1:` | Keep this threshold only if its validation F1 improves |
| `return best_epsilon, best_F1` | Report the chosen cutoff and its validation score |

A numeric example of mask construction at epsilon 0.10:

```text
prediction == 1: [T, T, T, F, F]
y_val == 1:      [T, F, T, T, F]
elementwise &:  [T, F, T, F, F]
sum:             1 + 0 + 1 + 0 + 0 = 2 true positives
```

The user's final submitted block used these three guards and consistent `prediction` naming. It computes one candidate's F1 **inside** the supplied loop. The surrounding scaffold generates candidates, updates the best result and returns it. Those outer lines explain why the inner formulas matter.

`np.arange(0, 1, 0.25)` gives `[0, .25, .5, .75]`. Unlike `range`, it supports floating steps and returns an array. Floating rounding can affect length/endpoints, so "about 1000" is intentional. The lab sweep normally omits the maximum score and does not explore every possible prediction partition. A dense grid is an approximation; the optimum among its candidates is not guaranteed to be a unique or globally best real-valued cutoff.

Only crossing an observed score changes predictions. Many neighboring epsilon values give identical counts/F1. With strict `>`, equal F1 does not replace the stored result, so the first improving candidate wins ties. If every F1 is zero, the lab-style initialization leaves best_epsilon at 0. A general utility might initialize best_F1 below zero to retain an evaluated candidate; that would change this tie/initialization behavior.

## 8. Mistakes, edge cases and practical limits

| Mistake or situation | What to understand / do |
| --- | --- |
| `prediction = ...` then using `predictions` | Different variable names. It may raise NameError or silently use a stale notebook value. Use one spelling everywhere; restart/run in order when debugging |
| Treating `p_val < epsilon` as one Boolean | It compares elementwise. For an array it produces an array; a Python list needs conversion first |
| `if prediction:` or using `and` on masks | An array contains many truth values. Use `&` for elementwise AND, with each comparison parenthesized |
| Missing parentheses around comparisons | Operator precedence can change the expression. Write `(a == 1) & (b == 1)` |
| Confusing equality and assignment | `=` stores; `==` compares |
| Converting to int before counting | Unnecessary: True==1 and False==0; np.sum on a Boolean mask already counts True entries |
| Comparing prediction with p_val | Scores are not answers. Compare decisions with `y_val` to evaluate correctness |
| `prediction.shape == (m,)`, `y_val.shape == (m,1)` | Broadcasting can create an incorrect `(m,m)` mask. Require matching 1-D shapes and matching row order |
| Guarding only precision | Recall or F1 can still divide by zero. Guard each relevant denominator |
| No predicted anomalies | TP=FP=0; precision is undefined mathematically, set to 0 by this lab convention |
| No actual anomalies | TP=FN=0; recall cannot measure anomaly detection; use the guard and obtain labeled anomalies for meaningful tuning |
| No correct positive predictions | TP=0; guarded positive-class F1 is 0, even if TN is large |
| Empty arrays or NaN/Inf densities | Validate inputs before min/max/search. Guards for ratios do not repair missing or nonfinite data |
| Identical scores | Lab arange step becomes zero. Explicitly evaluate "none flagged" and "all flagged" with cutoffs at and just above the common score, as in the teaching extension |
| Zero feature variance | The continuous Gaussian formula divides by zero. Investigate constant features; drop them or use a documented positive variance floor outside the graded exercise |
| Many features, tiny density products | Underflow may turn products into zero. Use sums of log densities and compare with a consistent log threshold in a practical implementation |
| Mostly normal data, few positives | F1 can change sharply when one anomaly is found or missed. Report counts and check representativeness |
| Claiming a low-density point is certainly faulty | The model detects unusual measurements, not their cause. Data errors, distribution shift and normal rare events can also trigger alarms |

For high-dimensional computation, log density is:

$$
\log p(\mathbf{x}) = -\frac12\sum_j\left[
\log(2\pi\sigma_j^2) + \frac{(x_j-\mu_j)^2}{\sigma_j^2}
\right].
$$

For positive epsilon, `p < epsilon` is equivalent to `log_p < log(epsilon)`. This is a practical numerical extension, not a replacement to slip into the graded lab without checking its expected interface.

F1 weights precision and recall symmetrically; it does not encode every operational cost. In a real alarm system, false alarms consume analyst time and missed faults may be costly. Choose metrics/constraints for that use case. This lab uses F1 to learn a clear threshold-selection procedure.

## 9. Explain it without the notebook

Try these before copying code:

1. **Why not average the signed deviations?** They cancel. Average squared deviations to measure spread.
2. **What does axis=0 preserve?** Features; it combines training rows into one statistic per column.
3. **If epsilon increases, does mu change?** No. More validation scores may trigger alarms; training parameters stay fixed.
4. **Why do I need y_val?** Unusual is not automatically incorrect or faulty. Labels tell us whether each alarm was correct during selection.
5. **Why TP/FP/FN?** They separate caught anomalies, false alarms and missed anomalies, which determine precision/recall/F1.
6. **Why not tune on test?** That uses the final exam to choose the answer; it no longer measures an untouched choice.
7. **What did Exercise 1 learn?** Normal feature means and variances.
8. **What did Exercise 2 choose?** A density cutoff by comparing candidate validation F1 scores.
9. **Does 0.875 mean 87.5% accuracy?** No. It is the recorded F1, not accuracy, precision alone or probability.
10. **Can I trace one row moving from FN to TP?** Raise epsilon above that anomaly's score. Recall rises; whether precision/F1 improve depends on other rows also crossing the cutoff.

The goal is to reconstruct the purpose of each stage, not memorize a particular epsilon or every mask character.

## Sources and verification

The two graded exercise names, final guard-based prediction block and recorded result come from the user's lab context/handoff. The course grader and source data were not run for this update. The full functions here make the dependency chain explicit; the equal-score fallback is identified as a teaching extension.

Official syntax references: [NumPy mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html), [variance](https://numpy.org/doc/stable/reference/generated/numpy.var.html), [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html), [arange and floating steps](https://numpy.org/doc/stable/reference/generated/numpy.arange.html), [sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html). The example calculations are original teaching fixtures.

**Documentation checks, 25 October 2026:** all 45 Python blocks across this note and the updated canonical sheet passed syntax checks. Executed all five note blocks and both new sheet blocks with their numerical/shape assertions. Recalculated all three threshold-table rows and exercised four threshold-search fixtures (equal scores, no positives, all positives, and separable scores). Checked 18 new navigation/cross-links, balanced fences and trailing whitespace across the six changed Markdown files. Python 3.12.14, NumPy 2.5.2. Existing unrelated sheet examples were not runtime-tested again. These checks validate teaching examples, not the reported course score.

