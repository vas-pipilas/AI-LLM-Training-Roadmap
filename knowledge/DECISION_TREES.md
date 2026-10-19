# Decision Trees — from a hand calculation to sklearn

**Added: 19 October 2026.** A deeper companion to the [canonical Python ML coding cheat sheet](PYTHON_ML_CODING_CHEAT_SHEET.md). This is a study resource, not course completion.

**Source boundary:** section 3a faithfully reproduces the original Andrew Ng ungraded Decision Trees lab excerpt supplied in the user's handoff, including its ten-row data and calculations. The four-row example is illustrative; section 4 provides separately labeled safer reusable helpers. The sklearn examples and practical explanations are independent teaching material.

## 1. What a tree learns

A decision tree learns a sequence of questions from labeled examples. Each internal node asks about an input feature; a branch chooses a route; a leaf supplies a prediction. Training chooses the questions and thresholds. Prediction follows the already learned questions.

For a telecom example, features might be setup delay, retransmission count and response-code proportions; the target might be an operator-confirmed incident category. The target must be separate from the features. A tree trained to predict an incident before it happens cannot use information recorded after the incident.

A classification tree predicts a category. A regression tree predicts a number, typically a leaf's mean target with squared-error training. Neither requires the neural-network training loop of epochs, backpropagation and a learning rate.

## 2. Entropy: how mixed are the labels?

For binary labels, let p be the fraction of examples labeled 1:

```text
H(p) = -p * log2(p) - (1-p) * log2(1-p)
```

| Labels in a node | p | Entropy in bits | Meaning |
| --- | --- | --- | --- |
| All 0 or all 1 | 0 or 1 | 0 | Pure |
| Half 0, half 1 | 0.5 | 1 | Most mixed binary node |
| Three 1, one 0 | 0.75 | 0.811278 | Mixed, but one class dominates |

The logarithm uses base 2. Treat the limiting term 0 * log2(0) as zero; do not actually evaluate log2(0). An empty node has no useful split: section 4's reusable helper returns zero entropy so its zero-weight contribution is safe. Entropy measures label mixture, not classification accuracy.

## 3. Information gain: did the question help?

```text
gain = parent entropy
       - (left count / parent count) * left entropy
       - (right count / parent count) * right entropy
```

The weights matter: a large impure child must count more than a tiny pure one.

**Illustrative four-row calculation (not the Andrew Ng lab data):** parent labels are [1, 1, 1, 0]. A binary feature sends two positive examples left and [1, 0] right.

- Parent: H(0.75) = 0.811278.
- Left: H(1) = 0; right: H(0.5) = 1.
- Weighted child entropy: (2/4)*0 + (2/4)*1 = 0.5.
- Information gain: 0.811278 - 0.5 = **0.311278 bits**.

A feature putting every example in one branch gives zero reduction. At each node, compare candidate splits on the rows reaching that node. The greedy choice maximizes the immediate gain; it does not prove that the resulting whole tree is globally optimal.

## 3a. Andrew Ng ungraded Decision Trees lab — faithful reproduction

**Source:** the user's handoff supplies the original ten-row arrays, feature mapping, four function definitions, root results and depth-2 structure reproduced here. This is a reproduction of that supplied lab excerpt, not a claim to have retrieved or executed the complete course notebook. The four-row example in section 3 remains a separate illustrative example.

### Original data and functions

This block is self-contained. The function logic below preserves the supplied lab code, including its assumptions about nonempty children.

```python
import numpy as np

X_train = np.array([
    [1, 1, 1],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 1],
    [1, 1, 1],
    [1, 1, 0],
    [0, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
])
y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])
feature_names = ["Ear Shape", "Face Shape", "Whiskers"]

def entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def split_indices(X, index_feature):
    left_indices, right_indices = [], []
    for i, x in enumerate(X):
        if x[index_feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices

def weighted_entropy(X, y, left_indices, right_indices):
    w_left = len(left_indices) / len(X)
    w_right = len(right_indices) / len(X)
    p_left = sum(y[left_indices]) / len(left_indices)
    p_right = sum(y[right_indices]) / len(right_indices)
    return w_left * entropy(p_left) + w_right * entropy(p_right)

def information_gain(X, y, left_indices, right_indices):
    p_node = sum(y) / len(y)
    h_node = entropy(p_node)
    return h_node - weighted_entropy(X, y, left_indices, right_indices)

print("Root entropy:", entropy(sum(y_train) / len(y_train)))
for feature, name in enumerate(feature_names):
    left, right = split_indices(X_train, feature)
    gain = information_gain(X_train, y_train, left, right)
    print(f"{name}: {gain:.2f}")
```

Expected printed results:

```text
Root entropy: 1.0
Ear Shape: 0.28
Face Shape: 0.03
Whiskers: 0.12
```

| Column index | Feature | Value 1 means |
| --- | --- | --- |
| 0 | Ear Shape | Pointy |
| 1 | Face Shape | Round |
| 2 | Whiskers | Present |

X_train has shape (10, 3), y_train has shape (10,). Five labels are 1 and five are 0, so p_node = 5/10 = 0.5 and H(0.5) = 1 bit.

### Walk through the code

1. **entropy(p)** takes a scalar class-1 fraction, not a label array. The two endpoint checks avoid evaluating log2(0); otherwise both class contributions are added.
2. **enumerate(X)** returns a row number i and its feature row x. Testing x[index_feature] selects one column; append(i) stores the row number, not the feature or label. With the full X_train, these are the original row IDs 0 through 9.
3. **y[left_indices]** selects the aligned target labels. Since the labels are 0/1, sum counts positive examples; dividing by the child size gives its positive fraction.
4. **w_left and w_right** measure each child's share of the rows passed as X. Multiplying entropy by these shares prevents a small child from counting as much as a large child.
5. **information_gain** computes parent entropy from the supplied y and subtracts weighted child entropy. Each candidate at a given node uses the same parent rows; the largest gain is the greedy choice.

### Verified root calculations

For Ear Shape, the original split is:

- Left, feature 0 == 1: **[0, 3, 4, 5, 7]**, labels [1, 0, 1, 1, 1], p_left = 4/5.
- Right, feature 0 == 0: **[1, 2, 6, 8, 9]**, labels [1, 0, 0, 0, 0], p_right = 1/5.
- Both children have entropy 0.7219280948873623 bits; both weights are 5/10.
- Weighted entropy = (5/10)H(4/5) + (5/10)H(1/5) = **0.7219280948873623**.
- Information gain = 1 - 0.7219280948873623 = **0.2780719051126377 bits**.

| Root feature | Left row IDs (value 1) | Right row IDs (value 0) | p_left | p_right | Weighted entropy | Gain (bits) | Printed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ear Shape (0) | [0, 3, 4, 5, 7] | [1, 2, 6, 8, 9] | 4/5 | 1/5 | 0.7219280948873623 | 0.2780719051126377 | 0.28 |
| Face Shape (1) | [0, 2, 4, 5, 7, 8, 9] | [1, 3, 6] | 4/7 | 1/3 | 0.9651484454403230 | 0.0348515545596770 | 0.03 |
| Whiskers (2) | [0, 1, 3, 4] | [2, 5, 6, 7, 8, 9] | 3/4 | 1/3 | 0.8754887502163469 | 0.1245112497836531 | 0.12 |

Rounding is only for display; compare unrounded gains during training.

### The supplied tree at max_depth=2

The root is depth 0. Every list below contains **original training row IDs**; value 1 follows the lab's left branch.

```text
depth 0: Ear Shape (feature 0), rows [0,1,2,3,4,5,6,7,8,9]
  left  [0,3,4,5,7], depth 1: Face Shape (feature 1)
    left  [0,4,5,7], depth 2: all y=1
    right [3],       depth 2: all y=0
  right [1,2,6,8,9], depth 1: Whiskers (feature 2)
    left  [1],       depth 2: all y=1
    right [2,6,8,9], depth 2: all y=0
```

Each depth-1 parent has entropy 0.7219280948873623. Its selected split creates two pure leaves, so weighted child entropy is 0 and node-local gain is 0.7219280948873623. All ten training labels are separated correctly; this is a training-data result, not evidence of accuracy on unseen examples.

### Original assumptions versus safer reusable helpers

**Empty children:** the original weighted_entropy divides by both child sizes unconditionally. A constant feature produces an empty child, so the calculation can raise ZeroDivisionError or produce NaN/warnings depending on scalar types. The entropy endpoint check does not fix division by zero that happens before entropy is called. Preserve that behavior in this reproduction; use the separate helpers in section 4 when empty branches are possible. An empty parent is also unsupported by the original code.

**Node-specific indexing:** split_indices enumerates whichever X it receives, starting at zero. On X_train it returns original IDs. On X_train[node_ids] it returns positions within that subset. To reuse the original functions below the root, pass matching X_node and y_node, then map local positions back to original IDs. This continuation demonstrates the mapping; it is explanatory glue, not an additional original lab function.

```python
# Continue the original lab block above.
node_ids = np.array([1, 2, 6, 8, 9])  # right child of the root
X_node, y_node = X_train[node_ids], y_train[node_ids]
left_local, right_local = split_indices(X_node, 2)
assert left_local == [0] and right_local == [1, 2, 3, 4]
left_original = node_ids[left_local].tolist()
right_original = node_ids[right_local].tolist()
assert left_original == [1]
assert right_original == [2, 6, 8, 9]
assert np.isclose(
    information_gain(X_node, y_node, left_local, right_local),
    0.7219280948873623,
)
```

Do not apply these local positions to y_train, and do not pass full-dataset X/y while evaluating a child's split: that would use the wrong labels, denominator or parent entropy. Section 4 instead keeps full X/y together with explicit original node_indices, so its weights divide by len(node_indices) and its parent entropy uses y[node_indices].

## 4. Read the Python helpers line by line

This single block is self-contained and is a **safer reusable teaching implementation, not the original lab code**. It intentionally handles **binary 0/1 features and targets**. Unlike section 3a, compute_entropy accepts labels and guards empty arrays; split_dataset keeps original row IDs through explicit node_indices. A split with an empty child has zero gain. A tree builder should still reject empty children and stop when get_best_split returns -1.

```python
import numpy as np

def compute_entropy(y):
    y = np.asarray(y)
    if len(y) == 0:
        return 0.0
    p = np.mean(y)  # valid because targets are binary 0/1
    if p == 0 or p == 1:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))

def split_dataset(X, node_indices, feature):
    left_indices, right_indices = [], []
    for i in node_indices:
        if X[i, feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices

def compute_information_gain(X, y, node_indices, feature):
    if len(node_indices) == 0:
        return 0.0
    left, right = split_dataset(X, node_indices, feature)
    parent_entropy = compute_entropy(y[node_indices])
    child_entropy = (
        len(left) / len(node_indices) * compute_entropy(y[left])
        + len(right) / len(node_indices) * compute_entropy(y[right])
    )
    return parent_entropy - child_entropy

def get_best_split(X, y, node_indices):
    best_feature, best_gain = -1, 0.0
    for feature in range(X.shape[1]):
        gain = compute_information_gain(X, y, node_indices, feature)
        if gain > best_gain:
            best_feature, best_gain = feature, gain
    return best_feature  # -1 means no positive-gain split

X = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])  # (4, 2)
y = np.array([1, 1, 1, 0])                      # (4,)
root_indices = list(range(len(y)))              # [0, 1, 2, 3]

left, right = split_dataset(X, root_indices, feature=0)
gain = compute_information_gain(X, y, root_indices, feature=0)
print(left, right)                  # [0, 1] [2, 3]
print(f"{compute_entropy(y):.6f}")  # 0.811278
print(f"{gain:.6f}")                # 0.311278
print(get_best_split(X, y, root_indices))  # 0: equal-gain tie keeps first
assert np.isclose(gain, 0.31127812445913283)
assert compute_entropy(np.array([])) == 0.0
assert compute_entropy(np.array([1, 1])) == 0.0
```

| Python expression | Read it as |
| --- | --- |
| `X.shape[1]` | Number of feature columns |
| `range(X.shape[1])` | Visit feature indices, starting at zero |
| `for i in node_indices` | Visit only original rows belonging to this node |
| `X[i, feature]` | One cell: this row, this feature |
| `y[node_indices]` | Targets of the rows reaching this node |
| `append(i)` | Store a row index, not the row's label |
| `left, right = ...` | Unpack the two returned lists |
| `return` | Send the result back to the caller |
| `gain > best_gain` | Replace the current choice only when strictly better |

The helpers use their arguments, not global names such as X_train. Otherwise they may appear correct on one dataset and fail on another.

**Why indices?** X and y keep their original alignment. A child contains a subset of original row IDs. If a child has [2, 7, 9], read X[2], X[7], X[9]; do not silently turn those into original rows 0, 1, 2.

**Scope of the helper:** other binary class encodings need conversion; multiclass entropy needs a sum across all classes. These helpers do not search thresholds for continuous features. They assume valid shapes and binary inputs rather than providing a production validation layer.

## 5. Building a whole tree

A recursive builder repeats the same job on each child:

1. Check a stopping condition.
2. Find the best split using this node's rows.
3. Split those rows.
4. Call the builder on each child, with depth increased by one.

Stop for pure labels, no useful split, a configured maximum depth, or too few examples. A leaf can predict its majority class. Decide an explicit tie policy. Avoid recursing on an empty branch.

The root is normally depth 0. A maximum depth of 2 allows at most two questions along any root-to-leaf path. It does not mean exactly two nodes or two leaves.

This is separate from **prediction**, which only traverses an existing tree and never chooses new splits.

## 6. Continuous and categorical features

For a continuous feature, try questions such as delay_ms <= 350. Training searches useful candidate thresholds between observed values. A feature can be reused deeper in the tree.

For unordered categories, do not invent a numeric order such as Athens=0, Thessaloniki=1, Patras=2 and assume that threshold splits mean what I intend. sklearn's standard DecisionTreeClassifier needs numeric input; one-hot encoding is a common option. Fit learned encoding on training data, ideally inside a Pipeline. Binary indicators already encoded as 0/1 are ready.

Standalone trees generally do not need StandardScaler: threshold ordering already gives the split information. This does not remove the need for sound missing-value handling, category encoding or consistent feature order.

## 7. Practical sklearn: training, validation and inspection

A complete example using a built-in dataset, with internal CV for model selection and one held-out test set. Run independently from the NumPy example.

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, train_test_split,
)
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target,
    test_size=0.2, stratify=data.target, random_state=42,
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(
    DecisionTreeClassifier(criterion="entropy", random_state=42),
    param_grid={
        "max_depth": [2, 3, None],
        "min_samples_leaf": [1, 3, 5],
        "ccp_alpha": [0.0, 0.01],
    },
    scoring="accuracy", cv=cv, refit=True, n_jobs=1,
)
search.fit(X_train, y_train)  # CV sees training data only
tree = search.best_estimator_  # automatically refitted on all training rows
predictions = tree.predict(X_test)
probabilities = tree.predict_proba(X_test)

print(search.best_params_)
print("mean CV accuracy:", search.best_score_)
print("training accuracy:", tree.score(X_train, y_train))
print("test accuracy:", accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions, labels=tree.classes_))
print(export_text(tree, feature_names=list(data.feature_names)))
print("probability columns correspond to:", tree.classes_)
assert X_train.shape == (120, 4)
assert X_test.shape == (30, 4)
assert predictions.shape == (30,)
assert probabilities.shape == (30, 3)
assert np.allclose(probabilities.sum(axis=1), 1.0)

one_example = X_test[:1]  # shape (1, 4), not (4,)
print(tree.predict(one_example))
```

No exact learned accuracy is promised. The shape expectations are fixed by this dataset and split.

| Call / attribute | What it does |
| --- | --- |
| DecisionTreeClassifier(...) | Creates an unfitted estimator and sets hyperparameters |
| fit(X_train, y_train) | Learns feature questions, thresholds and leaf distributions |
| predict(X_new) | Returns one class label per row |
| predict_proba(X_new) | Returns a column per class, ordered by classes_ |
| score(X, y) | Classifier accuracy, not entropy or loss |
| get_depth(), get_n_leaves() | Reports the learned structure's size |
| export_text(...) | Prints readable learned rules |
| feature_importances_ | Impurity-based importance; not evidence of causation |

To draw the fitted tree, **continue the sklearn block**:

```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(14, 7))
plot_tree(
    tree, feature_names=list(data.feature_names),
    class_names=list(data.target_names), filled=True, rounded=True,
)
plt.tight_layout()
plt.show()
```

sklearn normally uses Gini impurity by default. criterion="entropy" connects the example to the binary entropy calculation. sklearn learns CART-style binary threshold splits; a hand-coded “feature==1 goes left” lab implementation need not print an identical tree or use the same left/right convention.

Leaf class proportions are estimates from the training examples that reached the leaf. A tiny pure leaf's 1.0 output is not a guarantee about a new case.

## 8. Overfitting and control knobs

A large tree can memorize peculiarities of the training rows. Near-perfect training accuracy alongside weaker validation performance suggests variance; compare matching metrics and inspect leakage and distribution shift too.

| Parameter | Direction toward a simpler tree | Tradeoff |
| --- | --- | --- |
| max_depth | Decrease | Fewer successive questions |
| min_samples_leaf | Increase | Require more support per leaf |
| min_samples_split | Increase | Prevent splitting very small nodes |
| max_leaf_nodes | Decrease | Limit terminal regions |
| min_impurity_decrease | Increase | Require a larger weighted impurity reduction |
| ccp_alpha | Increase | Stronger cost-complexity pruning |

Too much restriction can underfit. Choose settings with CV, then evaluate the chosen procedure on the held-out test set. Do not repeatedly choose parameters based on test accuracy. ccp_alpha is a pruning parameter, not a gradient-descent learning rate or a direct synonym for neural-network lambda.

For imbalanced incident labels, accuracy alone may reward always predicting the common class. Inspect confusion matrices, per-class precision/recall and the cost of missed incidents. Tune class weights and scoring on training/CV data rather than assuming they always help.

Small changes to training rows can change a tree considerably. Random forests average diverse trees to reduce variance; boosting builds trees sequentially to improve the ensemble. Neither fixes leakage or bad labels.

## 9. Real applications and practical limits

These are possible projects, not claims about deployed models:

| Task | Input available at prediction time | Possible target / evaluation concern |
| --- | --- | --- |
| SIP incident triage | Windowed response classes, retransmissions, latency | Confirmed incident category; minority recall |
| Network service degradation | Prior counters and traffic history | Degraded/not degraded in a later window; time leakage |
| Call-setup delay regression | Route, load and prior-window measurements | Future setup delay; MAE and drift |
| Support-ticket routing | Encoded structured ticket fields | Owning team; rare-team performance |

Separate training and evaluation by time when predicting future operation. Keep related calls, subscribers or incident windows in the same group where needed to prevent near-duplicates appearing on both sides. Do not include raw identifiers merely because they improve training accuracy.

A small tree is useful when I need to inspect explicit rules. Deep trees become difficult to read, impurity importance can favor high-cardinality features, and fitted splits do not establish causes. Regression trees make piecewise constant predictions and do not reliably extrapolate beyond observed target behavior.

## 10. Self-check

1. Why is entropy zero for a pure node? There is no mixture of classes.
2. Why weight child entropy? Branches contain different numbers of examples.
3. Is the best split feature necessarily the target? No; the target supplies labels used to evaluate input questions.
4. Does predict fit new questions? No; it reuses learned rules.
5. Why use X_new=[[...]]? sklearn expects a batch dimension even for one row.
6. Is perfect training accuracy enough? No; assess unseen data with a suitable split.
7. Where are the original lab numbers? Section 3a reproduces the supplied ten-row lab, its root gains and depth-2 leaves; section 3's four-row example remains illustrative.

## Sources and verification

- Original ungraded Decision Trees lab excerpt: arrays, feature definitions, four functions and expected results supplied directly in the user's handoff; no complete notebook was retrieved.
- [Course 2 — Advanced Learning Algorithms](https://www.coursera.org/learn/advanced-learning-algorithms): decision-tree topic and lab context.
- [sklearn tree guide](https://scikit-learn.org/stable/modules/tree.html).
- [DecisionTreeClassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).
- [GridSearchCV API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
- [Cost-complexity pruning example](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html).

**Verification, 19 October 2026:** ran all five Python blocks in this note offline, with the stated continuation dependencies, using Python 3.12.14, NumPy 2.5.2, scikit-learn 1.9.1 and Matplotlib 3.11.2. Assertions checked all three root splits and gains against an independent class-count entropy calculation, the depth-2 split choices and four pure leaves, node-local to original index mapping, the four-row illustrative result, and safer-helper empty-parent, empty-child, constant-feature and pure-node cases. The original empty-child call raised ZeroDivisionError as expected. The Iris GridSearchCV example and prediction/probability assertions passed; the plot continuation exported a PNG using the noninteractive Agg backend (GUI display was not tested). The canonical sheet's section 12a sklearn block also ran successfully. The complete course notebook and any course plotting/recursive-builder utilities were not supplied or executed; this verifies the handoff reproduction and examples here. Earlier verification records for other cheat-sheet sections remain unchanged.
