# Anomaly Detection — why the lab works, from ELI5 to code

**Study note, 25 October 2026.** This is the dedicated theory and lab explanation for Andrew Ng's Machine Learning Specialization anomaly-detection lab. For reusable syntax, start with the [canonical coding cheat sheet](PYTHON_ML_CODING_CHEAT_SHEET.md#12d-anomaly-detection) and its [Boolean counting reference](PYTHON_ML_CODING_CHEAT_SHEET.md#boolean-counting-and-guarded-division). [NumPy for ML](NUMPY_FOR_ML.md) provides the focused array reference; [Unsupervised ML and K-means](UNSUPERVISED_LEARNING_AND_KMEANS.md) explains the neighboring clustering task.

**Source and scope:** this explanation follows the user's complete uploaded `Pasted text(8).txt`, including the printed code and outputs, retrieved from the referenced “Python LLM Training” conversation. It covers every section of that supplied text. The underlying data arrays, `utils.py`, notebook images and course grader are not included in the text, so recorded lab outputs below are observed in the upload, not rerun here. Tiny teaching arrays are explicitly separate. Passing tests records implementation success; understanding grows by tracing why each stage exists.

**Notebook rule:** the lab prohibits editing/deleting non-graded cells and adding new cells. Keep this explanation and experiments external. In the submitted notebook, work only inside the permitted graded regions. The lab mentions post-completion experimentation instructions at the bottom; those instructions are not present in the uploaded excerpt, so do not invent them.

## Navigation

- [0. Follow the supplied lab in order](#0-follow-the-supplied-lab-in-order)
- [1. The whole story before the formulas](#1-the-whole-story-before-the-formulas)
- [2. Mean, spread and squared deviations](#2-mean-spread-and-squared-deviations)
- [3. Gaussian density and multiple features](#3-gaussian-density-and-multiple-features)
- [4. Training, validation and epsilon](#4-training-validation-and-epsilon)
- [5. Why TP, FP and FN are necessary](#5-why-tp-fp-and-fn-are-necessary)
- [6. Exercise 1 — estimate_gaussian](#6-exercise-1--estimate_gaussian)
- [7. Exercise 2 — select_threshold](#7-exercise-2--select_threshold)
- [8. Mistakes, edge cases and practical limits](#8-mistakes-edge-cases-and-practical-limits)
- [9. Explain it without the notebook](#9-explain-it-without-the-notebook)
- [10. Finish the 2D lab: score, contours and outlier circles](#10-finish-the-2d-lab-score-contours-and-outlier-circles)
- [11. The actual 11-feature dataset](#11-the-actual-11-feature-dataset)
- [12. Glossary and syntax lookup](#12-glossary-and-syntax-lookup)
- [Sources and verification](#sources-and-verification)

## 0. Follow the supplied lab in order

Think of a server as a restaurant taking orders. **Latency** is how long a response takes; **throughput** is how much work/data moves per unit time. They are different measurements. A server behaving unlike the usual crowd deserves investigation, though unusual does not prove broken.

| Uploaded lab section | What happens | Read here |
| --- | --- | --- |
| 1 — Packages | Import calculation, plotting and course helpers | Below |
| 2 / 2.1 — Problem | Detect unusual server measurements from mostly normal data | Below and section 1 |
| 2.2 — Dataset | Load, inspect rows/shapes, draw scatter plot | Below |
| 2.3 — Gaussian distribution | Describe usual measurements with bell curves | Sections 2–3 |
| 2.3.1 / Exercise 1 | Learn one mean and variance per feature | Section 6 |
| Fitted distribution visualization | Score training rows and draw contours | Section 10 |
| 2.3.2 / Exercise 2 | Grade trial cutoffs using labeled validation data | Sections 4–7 |
| Outlier visualization | Circle low-density training rows | Section 10 |
| 2.4 — High dimensional dataset | Reuse the same functions with 11 features | Section 11 |

### 1 — Packages: the tools in the toolbox

This is the supplied notebook setup (not ordinary standalone Python because of the `%` line):

```ipython
import numpy as np
import matplotlib.pyplot as plt
from utils import *
%matplotlib inline
```

`import` makes tools available. `as np` and `as plt` give short nicknames: `np.mean(...)` comes from NumPy, while `plt.scatter(...)` comes from Matplotlib's plotting module. NumPy stores rectangular numeric arrays and operates on them together. Matplotlib draws the pictures.

`utils.py` is the course helper file. The lab uses `load_data`, `load_data_multi`, `multivariate_gaussian` and `visualize_fit`. `from utils import *` imports the helper names made available by that module into the notebook's name space. Their source is not included in the upload; leave it unchanged. `%matplotlib inline` is an IPython/Jupyter magic asking for plots in notebook output. It is not a NumPy command or valid syntax in a plain `.py` script.

Later, `from public_tests import *` imports the course tests. Calling `estimate_gaussian_test(estimate_gaussian)` passes the function object to a tester; it does not call the estimator with no data.

### 2 / 2.1 — Problem: learn usual behavior, then look for surprises

There are **307 unlabeled training examples**, each containing two measurements. Most are believed normal; a few may be unusual. We fit the overall normal pattern, then give low-density rows an alarm. The label-free fitting part makes this an anomaly-detection/unsupervised learning exercise. Labeled validation later helps choose the alarm cutoff.

### 2.2 — Load and inspect the data

Continue in the supplied notebook environment:

```python
X_train, X_val, y_val = load_data()
print(X_train[:5])
print(X_val[:5])
print(y_val[:5])
print(X_train.shape, X_val.shape, y_val.shape)
```

The right-hand function returns three objects. Tuple unpacking gives each its own name. `X` means measurements; `y` means known answers; `train` and `val` identify their roles. The actual shapes printed in the upload are `(307, 2)`, `(307, 2)`, `(307,)`.

```text
X_train (307, 2)                 X_val (307, 2)        y_val (307,)
          feature 0 feature 1      two measurements      one answer per row
row 0     13.04681517 14.74115241   15.79025979 ...       0
row 1     13.40852019 13.76326960   13.63961877 ...       0
...       ...         ...         ...                  ...
row 306   ...         ...         ...                  ...

training rows → fit parameters
validation row i → score i → prediction i ↔ y_val[i]
```

The same row count does not mean the training and validation arrays contain the same examples. There is no supplied `y_train`. `y_val` contains one answer for the entire validation server example, not one answer per feature. Its first five printed values are `[0 0 0 0 0]`; that does not mean all validation examples are normal. Here 0 means normal and 1 means anomalous.

`X_train[:5]` selects rows 0 through 4; the stop 5 is excluded. `X_train[0]` is the first row, shape `(2,)`; `X_train[0, 1]` is its second feature, a scalar. `X_train.shape[0]` is the number 307, not a row. `(307,)` is one-dimensional; the comma denotes a one-element shape tuple. It differs from a two-dimensional `(307, 1)` column.

### Scatter plot: one cross per server example

Continue after loading the lab data and importing `plt`:

```python
plt.scatter(X_train[:, 0], X_train[:, 1], marker='x', c='b')
plt.title('The first dataset')
plt.ylabel('Throughput (mb/s)')
plt.xlabel('Latency (ms)')
plt.axis([0, 30, 0, 30])
plt.show()
```

`[:, 0]` means all rows, first feature; `[:, 1]` means all rows, second feature. Each is shape `(307,)`. The first array supplies horizontal coordinates and the second vertical coordinates; values in the same row form one blue cross. `marker='x'` chooses the symbol and `c='b'` makes it blue. The title and labels name the picture and axes. `axis([0,30,0,30])` sets horizontal minimum/maximum followed by vertical minimum/maximum; it does not rescale or modify the data. `show()` displays the plot.

**Feature-order detail from the actual code:** the introductory prose lists throughput and latency, but the supplied plot labels column 0 as **latency** and column 1 as **throughput**. This guide follows that plotted convention. The unit string `mb/s` is preserved as written; the excerpt does not clarify bit/byte capitalization. Always verify column meaning rather than infer it from prose order. Leave the notebook's non-graded plot cell unchanged.

Look for a crowded region and isolated crosses. A scatter plot builds intuition; it does not by itself supply a mathematically chosen cutoff. Section 3 explains how the Gaussian turns that crowd into density scores.

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

| Symbol | Say it aloud | Job in these formulas |
| --- | --- | --- |
| `m`, `n` | number of examples, number of features | Row count and column count |
| `i`, `j` | row counter, feature counter here | The lab also uses i for features and j for rows; meaning comes from the formula, not the letter |
| `x_j^(i)` | feature j in example i | One cell: `X[i-1, j-1]` if math indices start at 1 |
| `Σ` | add them all | Walk through rows, adding the indicated value |
| `1/m` | divide by m | Turn the sum into an average |
| `mu_j` / `μ_j` | mean for feature j | One learned center |
| `sigma_j²` / `σ_j²` | variance for feature j | Average squared displacement from that center |

Mean formula: choose one column → add all entries → divide by row count. Variance formula: choose the same column → subtract its own mean from every entry → square → add → divide by row count. Repeat for each column.

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

| Gaussian symbol/piece | Beginner reading |
| --- | --- |
| `p(x_j; mu_j, sigma_j²)` | Bell height at this value using this feature's center/spread; the semicolon separates input from parameters |
| `π` | Pi, approximately 3.14159 |
| `sqrt(2π sigma_j²)` | Square root used in the normalization factor |
| `(x_j - mu_j)²` | Squared distance from the usual center |
| `2 sigma_j²` | Spread scale used to judge that distance |
| `exp(t)` | e raised to t, with e approximately 2.71828; in code `np.exp(t)` |
| Leading negative sign | Makes farther points produce smaller exponential values |
| `1 / sqrt(...)` | Sets bell height so total area is one |
| `Π` in the joint formula | Multiply feature densities; different from summation `Σ` |

For mean 0 and variance 1: at x=0, squared distance is 0, exponent 0, `exp(0)=1`, and density is about 0.39894. At x=1 the exponent is −0.5, so density is about `0.39894 × 0.60653 = 0.24197`. These are heights, not percentages. The bell is symmetric: x=−1 has the same height as x=1.

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
- Recall cannot decrease as epsilon rises on the same labeled set with actual anomalies, because TP cannot fall and TP+FN is fixed. F1 can still rise or fall as false alarms change.
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
    m, n = X.shape
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

The uploaded scaffold unpacks `m, n = X.shape`: here m=307 and n=2. These names are not needed in the vectorized calls. `axis=0` collapses the row dimension; it does not select row 0. `axis=1` would average the two features within each server, producing `(307,)`. Omitting axis combines all cells into one scalar. Between the graded markers, the two statistic lines are sufficient.

Observed in the uploaded lab after `mu, var = estimate_gaussian(X_train)` and `estimate_gaussian_test(estimate_gaussian)`:

```text
Mean of each feature: [14.11222578 14.99771051]
Variance of each feature: [1.83263141 1.70974533]
All tests passed!
```

Both vectors have shape `(2,)`; element 0 belongs to column 0 and element 1 to column 1. They describe measurements, not counts of anomalous servers.

**What happens next outside this exercise:** the notebook's density helper uses these learned parameters to compute training/validation densities. Exercise 2 receives already computed `p_val`, not the raw feature matrix.

## 7. Exercise 2 — select_threshold

Below is the complete executable function from the uploaded lab, with spacing normalized and notebook markers omitted. The surrounding initialization, search and best-result update matter as much as the graded block. The return values are the best values stored during the search. Inputs must be finite, nonempty, aligned 1-D arrays with binary labels and varying scores for this exact scaffold.

```python
import numpy as np

def select_threshold(y_val, p_val):
    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    step_size = (max(p_val) - min(p_val)) / 1000

    for epsilon in np.arange(min(p_val), max(p_val), step_size):
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
| `F1 = 0` | Initialize current score; each iteration assigns it again |
| `(max(p_val)-min(p_val))/1000` | Divide the observed density range into about 1000 steps |
| `np.arange(start, stop, step)` | Generate candidate cutoff values; stop is normally excluded |
| `for epsilon in np.arange(...)` | Try each alarm setting, one scalar epsilon at a time |
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

Execution order: initialize best values once → calculate step once → take one epsilon → recompute full prediction vector → count TP/FP/FN → calculate guarded precision, recall and F1 → compare with remembered best → next epsilon. Only after the loop ends does `return` send two values back. `epsilon, F1 = select_threshold(...)` unpacks those returned best values into the caller's names. The last candidate need not be the best.

Indentation defines flow. The metric `if/else` pairs and best-result update belong inside `for`; `return` belongs outside it. Returning inside stops at the first cutoff. An `else` runs only when its matching `if` condition is false. `prec` and `rec` are scalar ratios; `prediction` is a whole Boolean array.

At miniature epsilon=0.10: TP=2, FP=1, FN=1 → precision=2/3 → recall=2/3 → F1=2×(2/3)×(2/3)/(4/3)=2/3. At epsilon=0.30: TP=3, FP=1, FN=0 → precision=3/4 → recall=1 → F1=1.5/1.75=6/7. That is why the latter cutoff wins among these two.

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
| Identical scores | Lab arange step becomes zero. An external robust utility could evaluate cutoffs at the common score and just above it (`np.nextafter(score, np.inf)`) to compare none versus all flagged; this is not part of the uploaded function |
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

## 10. Finish the 2D lab: score, contours and outlier circles

These snippets continue the lab session with its imports, loaded arrays and the two completed functions. They need course helpers; they are not independently runnable without those files.

### Score the training rows and visualize the fitted model

```python
p = multivariate_gaussian(X_train, mu, var)
visualize_fit(X_train, mu, var)
```

`p` has shape `(307,)`: one density per training row. Despite the helper's name, the lab supplies a variance vector, describing the feature-wise independent Gaussian model in section 3. Its helper implementation is not visible in the text; the formula and vector interface are the basis for this explanation.

Imagine a hill with the normal crowd near the top. Contours are like lines on a hiking map: each connects positions with the same height, here the same density. In two dimensions the independent Gaussian's constant-density lines are axis-aligned ellipses (circles if variances match). Near the fitted center, density is higher; far away it is lower. The numerical means put the center near `(14.11, 15.00)` in plotted column order. `visualize_fit` draws the fitted model alongside data; it does not choose epsilon. A contour is not automatically an anomaly boundary. The chosen threshold later identifies which side raises alarms.

The upload contains plotting calls but not their rendered images. This explains what those calls represent without claiming to inspect the original figures or a specific plotted point.

### Use the same training parameters to score validation

```python
p_val = multivariate_gaussian(X_val, mu, var)
epsilon, F1 = select_threshold(y_val, p_val)
print('Best epsilon found using cross-validation: %e' % epsilon)
print('Best F1 on Cross Validation Set: %f' % F1)
select_threshold_test(select_threshold)
```

`p_val` has shape `(307,)`; it belongs to validation rows, while `p` belongs to training rows. The shared `mu`/`var` come from training. Only the validation labels grade candidate thresholds. Observed upload output:

```text
Best epsilon found using cross-validation: 8.990853e-05
Best F1 on Cross Validation Set: 0.875000
All tests passed!
```

`%e` formats scientific notation; `%f` shows a fixed-point decimal (six places by default). They change display, not the fitted value. `8.990853e-05` means `8.990853 × 10^-5 = 0.00008990853`. F1 0.875 is a validation metric, not 87.5% accuracy or a probability that any individual alarm is correct. The printed rounding does not expose the full stored precision.

### Circle the training outliers

```python
outliers = p < epsilon
visualize_fit(X_train, mu, var)
plt.plot(X_train[outliers, 0], X_train[outliers, 1], 'ro',
         markersize=10, markerfacecolor='none', markeredgewidth=2)
```

The comparison produces a Boolean mask of shape `(307,)`; True marks a training row below the chosen cutoff. `X_train[outliers, 0]` selects column 0 only for True rows; `X_train[outliers, 1]` selects the matching column 1 values. If k rows are flagged, each selected coordinate array is `(k,)`; `X_train[outliers]` would be `(k,2)`.

Miniature indexing example, separate from course data:

```python
import numpy as np
X = np.array([[10., 20.], [30., 40.], [50., 60.]])
p = np.array([0.20, 0.001, 0.30])
outliers = p < 0.01
assert outliers.tolist() == [False, True, False]
assert X[outliers].tolist() == [[30., 40.]]
assert X[outliers, 0].tolist() == [30.]
assert X[outliers, 1].tolist() == [40.]
```

`'ro'` requests red (`r`) circle (`o`) markers. Size 10 controls the marker size; `markerfacecolor='none'` leaves the middle empty so the original cross remains visible; edge width 2 thickens the ring. These settings mark existing examples; they do not change predictions or retrain the Gaussian. The printed `[<matplotlib.lines.Line2D at ...>]` is a returned plotting-object representation, not an anomaly count or error. Its memory address varies.

Use training scores with training rows and validation scores with validation rows. Their equal lengths in this dataset can hide a mistaken pairing: the wrong mask could run successfully while selecting unrelated examples. The 2D upload does not print a training-outlier count; do not invent one.

## 11. The actual 11-feature dataset

The next part uses the same algorithm with more measurements per server. Eleven features means eleven columns, not eleven servers or classes. The supplied text does not name all eleven measurements, so there is no basis to label them CPU, memory, etc.

Continue with course helpers and the completed exercise functions:

```python
X_train_high, X_val_high, y_val_high = load_data_multi()
mu_high, var_high = estimate_gaussian(X_train_high)
p_high = multivariate_gaussian(X_train_high, mu_high, var_high)
p_val_high = multivariate_gaussian(X_val_high, mu_high, var_high)
epsilon_high, F1_high = select_threshold(y_val_high, p_val_high)
print('Best epsilon found using cross-validation: %e' % epsilon_high)
print('Best F1 on Cross Validation Set:  %f' % F1_high)
print('# Anomalies found: %d' % sum(p_high < epsilon_high))
```

| Object | Actual shape / role |
| --- | --- |
| `X_train_high` | `(1000,11)`: 1000 training rows, 11 measurements each |
| `X_val_high` | `(100,11)`: 100 validation rows, same feature schema |
| `y_val_high` | `(100,)`: one known answer per validation row |
| `mu_high`, `var_high` | `(11,)`: one learned center/spread per feature |
| `p_high` | `(1000,)`: one training density per row |
| `p_val_high` | `(100,)`: one validation density per row |
| `epsilon_high`, `F1_high` | Scalars: selected cutoff and validation F1 |
| `p_high < epsilon_high` | `(1000,)`: training alarm mask |

Read the execution as: load separate data → estimate eleven means/variances from training → score training → score validation with the same parameters → choose epsilon from validation labels → count flagged training rows. The `_high` suffix is a naming convention that keeps this model's values separate from the 2D model. It has no special Python behavior.

Observed output in the complete uploaded text:

```text
The shape of X_train_high is: (1000, 11)
The shape of X_val_high is: (100, 11)
The shape of y_val_high is:  (100,)
Best epsilon found using cross-validation: 1.377229e-18
Best F1 on Cross Validation Set:  0.615385
# Anomalies found: 117
```

`sum(p_high < epsilon_high)` counts True values; `%d` displays the integer count. The **117 are flagged training examples out of 1000**, not 117 known faulty servers, validation anomalies or test errors. F1 0.615385 is from the 100 labeled validation examples. The excerpt does not print their TP/FP/FN, precision, recall or accuracy, so these cannot be reconstructed uniquely from the rounded F1 alone.

The threshold `1.377229e-18 = 1.377229 × 10^-18` is much smaller than the 2D cutoff. Multiplying eleven densities often gives much smaller values than multiplying two, and scales/units also affect densities. This is why each dataset needs its own validation-selected cutoff. A smaller epsilon does not establish a better model. Its recorded F1 is lower than the 2D result, but different datasets make that an observation rather than a controlled feature-count experiment.

Eleven-dimensional points cannot be fully shown in one ordinary 2D scatter plot. Picking two features is a projection: it can hide unusual values in the other nine. The algorithm still computes one score using all eleven. A full covariance model would learn relationships between features; this exercise instead keeps the independent-feature assumption. See section 8 for log-density underflow and zero-variance cautions for external experiments.

### Shape diagram: vectorization and broadcasting end to end

```text
X_train_high (1000,11) -- mean/var(axis=0) --> mu/var (11,)

X_val_high (100,11)
    - mu_high  (11,)   broadcast the same feature means across all 100 rows
    ↓
centered   (100,11) -- square/divide/exp --> feature densities (100,11)
    -- product(axis=1) --> p_val_high (100,)
    < epsilon_high () --> prediction (100,) Boolean
    & matching label conditions (100,) --> masks (100,)
    -- sum --> scalar counts --> scalar precision/recall/F1
```

Broadcasting matches dimensions from the right: 11 matches 11, and the mean vector is reused over rows. A scalar epsilon can compare with every score. Vectorization describes doing array operations instead of writing a Python loop over rows. It does not remove all loops: the provided threshold function still loops over epsilon candidates, while calculating every example's decision together inside each iteration. `np.mean(..., axis=0)` combines examples per feature; `np.prod(..., axis=1)` combines features per example. They collapse different dimensions for different purposes.

## 12. Glossary and syntax lookup

| Word | Plain meaning |
| --- | --- |
| Example / sample | One row: measurements from one server observation |
| Feature / dimension | One measurement type; one column |
| Label / ground truth | Known normal/anomaly answer used for grading predictions |
| Parameter | Learned model value, here means and variances |
| Mean / variance / standard deviation | Center / average squared spread / square root of variance |
| Gaussian | Bell-shaped continuous density model |
| Density | Curve height or multidimensional density value; not event probability |
| Factorized / independent Gaussian | Multiply separately fitted feature densities, assuming independence |
| Threshold / epsilon | Density cutoff for raising an anomaly alarm |
| Prediction | Model decision; True/1 means predicted anomaly here |
| Validation / cross-validation set here | Labeled holdout used to choose epsilon; not a k-fold loop in this code |
| Outlier / anomaly | Unusual observation under the fitted rule; not automatically proven failure |
| TP / FP / FN / TN | Correct alarm / false alarm / missed anomaly / correctly unflagged normal |
| Precision / recall | Correct fraction of alarms / caught fraction of actual anomalies |
| F1 | Harmonic mean of precision and recall; grades each candidate epsilon |
| Boolean mask | True/False array selecting or identifying matching rows |
| Vectorization / broadcasting | Operate on arrays together / reuse compatible smaller arrays in operations |
| Population variance | Mean squared deviation with divisor m in this lab |

For reusable syntax, see [canonical anomaly syntax](PYTHON_ML_CODING_CHEAT_SHEET.md#12d-anomaly-detection) and [Boolean counting and guarded division](PYTHON_ML_CODING_CHEAT_SHEET.md#boolean-counting-and-guarded-division). They cover `=`, `==`, `<`, `&`, `~`, `**`, axes, slicing, `np.arange`, scalar versus array conditions, and guarded ratios. Section 8 above collects common mistakes; section 9 provides recall questions. The full function lives here once; those references link back instead of reproducing this entire lab.

## Sources and verification

Primary lab source: the complete user-uploaded `Pasted text(8).txt` attached to the referenced “Python LLM Training” conversation. All dataset shapes, printed parameter values, epsilon/F1 values, the two “All tests passed!” messages and 117 count above are transcribed from that text. This is stronger grounding than the earlier partial handoff. The file does not contain the underlying datasets/helper sources or rendered plots. The course notebook and grader were not run for this update. The exercise functions preserve the uploaded executable logic; robustness ideas in section 8 are external extensions, not changes to the lab.

**Verification, 25 October 2026:** Python 3.12.14 / NumPy 2.3.5. All 78 Python blocks across the five edited documents passed syntax checks. Seven independent teaching fixtures ran with assertions: six in this guide and the new canonical counting example. Additional checks covered hand-calculated TP/FP/FN/F1, no actual anomalies, all-positive labels, the grid's maximum-score exclusion, 11-feature statistics and broadcasting. Both exercise function bodies were compared structurally with the actual uploaded source and match it. All 74 links into the edited local documents passed target/anchor checks. Course-dependent snippets were syntax-checked and reviewed, not run against missing course helpers/data. Existing unrelated course examples were not runtime-tested again.

Official syntax references: [NumPy mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html), [variance](https://numpy.org/doc/stable/reference/generated/numpy.var.html), [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html), [arange and floating steps](https://numpy.org/doc/stable/reference/generated/numpy.arange.html), [sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html). The example calculations are original teaching fixtures.

**Documentation checks, 25 October 2026:** all 45 Python blocks across this note and the updated canonical sheet passed syntax checks. Executed all five note blocks and both new sheet blocks with their numerical/shape assertions. Recalculated all three threshold-table rows and exercised four threshold-search fixtures (equal scores, no positives, all positives, and separable scores). Checked 17 new navigation/cross-links, balanced fences and trailing whitespace across the six changed Markdown files. Python 3.12.14, NumPy 2.5.2. Existing unrelated sheet examples were not runtime-tested again. These checks validate teaching examples, not the reported course score.

