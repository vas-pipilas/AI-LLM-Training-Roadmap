# Decision Trees — from a hand calculation to sklearn

**Added: 19 October 2026.** A deeper companion to the [canonical Python ML coding cheat sheet](PYTHON_ML_CODING_CHEAT_SHEET.md). This is a study resource, not course completion.

**Source boundary:** section 3a faithfully reproduces the original Andrew Ng ungraded Decision Trees lab excerpt supplied in the user's handoff, including its ten-row data and calculations. The four-row example is illustrative; section 4 provides separately labeled safer reusable helpers. Section 8a explains the second, uploaded Trees Ensemble lab; its reported results are transcribed and distinguished from modern runnable adaptations. Other sklearn examples and practical explanations are independent teaching material.

**Graded-exercise companion, 21 October 2026:** [section 4a](#4a-graded-exercise-follow-the-original-row-ids) explains current-node row IDs, mushroom labels, entropy, nested splits and the first-attempt indexing mistakes. Its six-row dataset is an invented executable example, separate from the preserved ungraded lab reproduction.

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

## 4a. Graded exercise: follow the original row IDs

This section explains the graded Andrew Ng decision-tree exercises discussed in chat, including `split_dataset` (UNQ_C2). The tiny mushroom data below are an **invented teaching example**, not a reproduction of the course dataset or a claim that the full graded notebook was executed. The earlier ungraded animal lab in section 3a stays separate.

### What confused me in the graded lab / brain-burner troubleshooting

Read this as a debugging route: **labels → entropy → split row IDs → weighted information gain → best feature → repeat on a child**. The indexing walkthrough and first-attempt table below cover the earlier mistakes; the later gain-selection walkthrough covers the follow-up attempts. The current request supplies the broader list of misunderstandings; the retrieved chat directly shows `ig_1 > best_feature`, the question about keeping the previous gain, and the attempt that selected a feature without updating `ig_max`.

### Picture a teacher sorting schoolchildren

Imagine a teacher with one big class register. Each child has a permanent register number. The teacher asks, “Are you wearing a hat?” Children answering yes go left; children answering no go right. Then the teacher asks only the children on the right another question. Their register numbers stay the same: children 2 and 5 do not become children 0 and 1.

For the mushroom exercise, replace children with mushrooms and the register with `X`. Each feature column answers one yes/no question about a mushroom. A separate answer sheet, `y`, says **1 = edible, 0 = poisonous**. Those label meanings belong to this exercise; they are not a universal encoding. A feature value of 1 means “yes to this feature,” not “edible.” **`y = 1` is the positive-class label, not a statement that a prediction was correct.** For example, `y_true = 1` and `y_pred = 0` is an incorrect prediction of a positive example. Correctness requires comparing a prediction with its label; entropy needs no predictions.

| Name or expression | What it means |
| --- | --- |
| `X` | Full feature matrix, shape `(n_samples, n_features)`; one mushroom per row |
| `y` | Aligned labels, shape `(n_samples,)`; `y[i]` is the answer for row `X[i]` |
| `node_indices` | Original row IDs of samples at the **current node** |
| Root's `node_indices` | All original row IDs, e.g. `[0, 1, 2, 3, 4, 5]` |
| Child's `node_indices` | Only IDs routed to that child, e.g. `[2, 3, 5]` |
| `feature` | One selected column index, e.g. `1`; not a label or row ID |
| `X[i, feature]` | One scalar cell: mushroom `i`'s value in that column |
| `X[node_indices]` | A matrix containing the selected rows and **all** feature columns |
| `y[node_indices]` | Labels of just the mushrooms reaching this node |
| `p_1` | Fraction of those labels equal to 1; a single number, not the list of labels |

Read `X[i, feature]` as “row first, column second.” For a two-dimensional NumPy array, `X[i]` is a whole feature row, while `X[i, feature]` is one value from that row. With a list of integer row IDs, `X[node_indices]` uses NumPy advanced indexing and produces a separate array. Merely reading that subset does not modify `X`.

### Entropy counts labels, not feature values

At the current node, first select its labels: `y_node = y[node_indices]`. Then:

```text
p_1 = number of labels equal to 1 / number of labels at this node
H = -p_1 * log2(p_1) - (1 - p_1) * log2(1 - p_1)
```

For binary labels, `p_1 = sum(y_node) / len(y_node)` (or NumPy's `np.sum`) is the class-1 fraction. Inside `compute_entropy(y)`, the argument is already the selected labels, so this becomes `p_1 = sum(y) / len(y)`.

With binary 0/1 labels, `np.sum(y_node)` counts the edible mushrooms. `p_1 = 0` means all poisonous; `p_1 = 1` means all edible. Both nodes are **pure**, so entropy is 0. Half edible and half poisonous gives maximum binary entropy, 1 bit. Entropy describes how mixed the answers are; it does not tell us which feature to choose by itself.

Handle the pure endpoints **before** calling the logarithm: NumPy cannot evaluate `log2(0)` as a finite number, and multiplying it by zero afterward can produce `nan`. For an empty label array, the fraction is undefined; the teaching helper below returns 0.0 by convention before dividing. That makes an empty child's zero-weight entropy contribution safe, but an empty child is not a useful leaf to recurse into.

In the graded entropy exercise, `compute_entropy(y)` receives the labels to summarize. At a child node, call it with `y[node_indices]`, rather than using all labels again. This differs from section 3a's `entropy(p)`, which receives an already computed scalar fraction.

### Runnable example: split once, then split a child

Run this whole block independently with NumPy installed. The feature meanings are invented yes/no flags for learning the indexing.

```python
import numpy as np

def compute_entropy(y):
    if len(y) == 0:
        return 0.0
    p_1 = np.sum(y) / len(y)  # y contains only binary 0/1 labels
    if p_1 == 0 or p_1 == 1:
        return 0.0
    return float(-p_1 * np.log2(p_1)
                 - (1 - p_1) * np.log2(1 - p_1))

def split_dataset(X, node_indices, feature):
    left_indices = []
    right_indices = []
    for i in node_indices:
        if X[i, feature] == 1:
            left_indices.append(i)
        elif X[i, feature] == 0:
            right_indices.append(i)
    return left_indices, right_indices

# Columns: 0 = has spots?, 1 = has a ring?
X = np.array([
    [1, 0],  # original row 0
    [1, 1],  # original row 1
    [0, 1],  # original row 2
    [0, 0],  # original row 3
    [1, 1],  # original row 4
    [0, 1],  # original row 5
])
y = np.array([1, 1, 0, 1, 0, 0])  # 1 edible, 0 poisonous
X_before = X.copy()
root_indices = list(range(len(X)))

left, right = split_dataset(X, root_indices, feature=0)
print("Root:", left, right)
# Now split ONLY the root's right child, still using the full X.
node_indices = right
child_left, child_right = split_dataset(X, node_indices, feature=1)
print("Right child:", child_left, child_right)
print("One cell:", X[5, 1])
print("Current rows:", X[node_indices].tolist())
print("Current labels:", y[node_indices].tolist())
p_1 = np.sum(y[node_indices]) / len(node_indices)
print(f"Current p_1: {p_1:.6f}")
print(f"Current entropy: {compute_entropy(y[node_indices]):.6f}")
print("X unchanged:", np.array_equal(X, X_before))

assert left == [0, 1, 4] and right == [2, 3, 5]
assert child_left == [2, 5] and child_right == [3]
assert X[node_indices].shape == (3, 2)
assert compute_entropy(np.array([0, 0])) == 0.0
assert compute_entropy(np.array([1, 1])) == 0.0
assert compute_entropy(np.array([])) == 0.0
assert compute_entropy(np.array([0, 1])) == 1.0
assert np.isclose(compute_entropy(y[node_indices]), 0.9182958340544896)
assert np.array_equal(X, X_before)
```

Expected output:

```text
Root: [0, 1, 4] [2, 3, 5]
Right child: [2, 5] [3]
One cell: 1
Current rows: [[0, 1], [0, 0], [0, 1]]
Current labels: [0, 1, 0]
Current p_1: 0.333333
Current entropy: 0.918296
X unchanged: True
```

The nested routes are:

```text
root rows [0, 1, 2, 3, 4, 5], ask column 0
  value 1 -> left  [0, 1, 4]
  value 0 -> right [2, 3, 5], ask column 1
    value 1 -> left  [2, 5], labels [0, 0]
    value 0 -> right [3],    labels [1]
```

At the right child, `for i in node_indices` visits **2, then 3, then 5**. It reads `X[2, 1] = 1`, `X[3, 1] = 0`, and `X[5, 1] = 1`. Appending `i` saves those original row IDs. The resulting lists let us look up either features or aligned labels later.

`split_dataset` creates new lists; it never assigns to `X`, removes rows, or renumbers them. The full register stays on the desk; only the list of children being considered changes. Keep passing the full `X` with original IDs. If you deliberately create `X_node = X[node_indices]`, its row positions start at 0; do not then index `X_node` using those original IDs. Section 3a demonstrates explicit local-to-original mapping for that different approach.

The shown `if/elif` follows the graded exercise's binary feature contract: 1 goes left and 0 goes right. An unexpected value such as 2 or NaN matches neither branch and would be omitted, so validate/encode data before reusing this outside the exercise. Section 4's `else` version assumes the same valid binary inputs; for those inputs the results are identical. Left/right is this exercise's convention, not a universal rule for every tree implementation.

### Splitting answers “where?”; information gain answers “was it helpful?”

`split_dataset` needs only `X`, the current IDs and a feature. It asks one already selected feature question and returns two ID lists. It does not inspect `y`, compute entropy, decide whether the groups are pure, or choose the best question.

`compute_information_gain` uses those lists to read the labels and score that candidate question. `get_best_split` compares scores for candidate columns at the same node. In the example, the root's column 0 was chosen just to show routing; it is **not** a claim that column 0 is best. Both root children have mixed labels, and that root question has gain only about 0.081704 bits.

Continue the runnable block above:

```python
def compute_information_gain(X, y, node_indices, feature):
    if len(node_indices) == 0:
        return 0.0
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    n = len(node_indices)  # CURRENT node size, not the full len(X)
    weighted_children = (
        len(left_indices) / n * compute_entropy(y[left_indices])
        + len(right_indices) / n * compute_entropy(y[right_indices])
    )
    return compute_entropy(y[node_indices]) - weighted_children

root_gain = compute_information_gain(X, y, root_indices, feature=0)
child_gain = compute_information_gain(X, y, right, feature=1)
print(f"Root gain, column 0: {root_gain:.6f}")
print(f"Right-child gain, column 1: {child_gain:.6f}")
assert np.isclose(root_gain, 0.08170416594551044)
assert np.isclose(child_gain, 0.9182958340544896)
assert compute_information_gain(X, y, [], feature=0) == 0.0
assert compute_information_gain(X, y, left, feature=0) == 0.0
```

Expected output:

```text
Root gain, column 0: 0.081704
Right-child gain, column 1: 0.918296
```

For the right child, the parent labels are `[0, 1, 0]`, so `p_1 = 1/3` and parent entropy is 0.918296. Its two children are pure: `[0, 0]` and `[1]`. Their weights are **2/3 and 1/3**, because three mushrooms reached this node, even though `X` still has six rows. Weighted child entropy is `(2/3)*0 + (1/3)*0 = 0`; gain equals the parent's entropy. For an unchanged/constant feature, one child is empty and the other contains the whole current node, so gain is zero.

### Mistakes from the first attempts

| Tempting code or idea | Why it goes wrong | Use instead |
| --- | --- | --- |
| `X[i, node_indices]` | The second slot selects **columns**; node_indices contains row IDs. It may return several cells or raise an out-of-bounds error. | `X[i, feature]` for one cell; `X[node_indices]` for whole selected rows |
| `left_indices.append(X[i, node_indices])` | Uses row IDs as columns and stores cell values/an array, losing sample identity. | `left_indices.append(i)` |
| Append `X[i, feature]` | Stores only 0/1 answers; many mushrooms share them. | Append the original row ID `i` |
| Append `X[i]` | Stores a whole feature row instead of an index usable with both X and y. | Append `i` |
| `for i in range(len(X))` at every node | Revisits all original rows, including mushrooms that went to another branch. | `for i in node_indices`; `range(len(X))` is useful to create the **root** ID list |
| `for i in range(len(node_indices))` then `X[i, feature]` | On `[2, 3, 5]`, visits original rows `[0, 1, 2]` instead of `[2, 3, 5]`. | Loop over IDs directly, or map a position k to `i = node_indices[k]` |
| Compute the child's `p_1` from all `y` or divide by `len(X)` | Measures the root's mixture instead of this node's mixture. | Use `y[node_indices]` and the current node size |
| Define `p_1` then use `p1` | These are two different Python names; an undefined name raises NameError, or a stale notebook variable can silently give a wrong value. | Spell `p_1` consistently; restart/run cells in order |
| Compute `log2(p_1)` before checking endpoints | Pure nodes can produce warnings/infinite or NaN terms. | Return 0 for `p_1 == 0` or `p_1 == 1` first |

A quick check before coding: **Which rows are here? Which column am I asking about? Am I returning the mushrooms' IDs or their answers?** Then evaluate the split separately using the labels.

### Why the child weights are not p_1

The parent is the **current node**, even when it is a child of an earlier node. Its entropy is `compute_entropy(y[node_indices])`; child entropies use `y[left_indices]` and `y[right_indices]`. Only at the root do all labels belong to the current parent.

| Quantity | Numerator / denominator | Question it answers |
| --- | --- | --- |
| Parent `p_1` | Positive labels at this node / samples at this node | How mixed are this node's classes? |
| Left child's `p_1` | Positive labels in left child / samples in left child | How mixed are the left labels? |
| `w_left` | `len(left_indices) / len(node_indices)` | What fraction of the current samples went left? |
| `w_right` | `len(right_indices) / len(node_indices)` | What fraction went right? |

For a nonempty node and valid binary routing, `w_left + w_right = 1`. A child weight can be zero; that child's `p_1` is undefined when empty, which is why the entropy helper guards it.

```text
IG = H(current parent) - [w_left * H(left labels) + w_right * H(right labels)]
```

The earlier three-row node has `p_1 = 1/3` and `w_left = 2/3`: they measure different things. Both child entropies happen to be zero there, so weights do not change that particular result.

To see why weighting matters, imagine a separate ten-row current node with five positives and five negatives. A candidate feature routes nine rows left (five positives, four negatives) and one negative right:

```text
parent: p_1 = 5/10, H = 1
left:   p_1 = 5/9,  H ≈ 0.991076, w_left = 9/10
right:  p_1 = 0/1,  H = 0,        w_right = 1/10
weighted child entropy ≈ 0.9 * 0.991076 + 0.1 * 0 = 0.891968
information gain ≈ 1 - 0.891968 = 0.108032 bits
```

An unweighted average would report `(0.991076 + 0) / 2 ≈ 0.495538` and gain `0.504462`. That gives the single pure row as much influence as the nine-row mixed group and exaggerates the improvement. We want average remaining uncertainty **per sample**, so branch sizes matter. Using the full dataset size in a deeper node's denominator would also give the wrong weights.

### Best feature: keep the best score so far, not just the previous score

`split_dataset` routes rows for **one specified feature**. `compute_information_gain` evaluates that candidate using labels. `get_best_split` tries the feature columns and remembers which candidate achieved the largest positive gain. It does not build the complete recursive tree.

In the six-row example, `X.shape == (6, 2)`: try features `0, 1` with `range(X.shape[1])`, not six sample positions with `range(len(y))`. Rows are the examples used to evaluate a question; columns are the candidate questions.

Continue the two section 4a runnable blocks above. This uses the same helper behavior as section 4, with the variable names from the student's attempt:

```python
def get_best_split(X, y, node_indices):
    num_features = X.shape[1]
    best_feature = -1
    ig_max = 0.0
    for i in range(num_features):  # i is a FEATURE index in this loop
        ig_i = compute_information_gain(X, y, node_indices, i)
        if ig_i > ig_max:
            best_feature = i
            ig_max = ig_i
    return best_feature

gains = [
    compute_information_gain(X, y, root_indices, feature)
    for feature in range(X.shape[1])
]
print("Root gains:", [round(gain, 6) for gain in gains])
print("Best root feature:", get_best_split(X, y, root_indices))
print("Best right-child feature:", get_best_split(X, y, right))
assert np.allclose(gains, [0.08170416594551044, 0.4591479170272448])
assert get_best_split(X, y, root_indices) == 1
assert get_best_split(X, y, right) == 1
assert get_best_split(X, y, [2, 5]) == -1  # already pure
assert get_best_split(X, y, []) == -1
assert get_best_split(np.zeros_like(X), y, root_indices) == -1

# Separate weight example above; no change to the six-row X/y.
left_labels = np.array([1] * 5 + [0] * 4)
right_labels = np.array([0])
weighted = 0.9 * compute_entropy(left_labels) + 0.1 * compute_entropy(right_labels)
assert np.isclose(weighted, 0.8919684538544)
assert np.isclose(1 - weighted, 0.1080315461456)
```

Expected output:

```text
Root gains: [0.081704, 0.459148]
Best root feature: 1
Best right-child feature: 1
```

`ig_i` stores the returned score for **this candidate**. `ig_max` stores the **largest score seen so far**. `best_feature` stores the **column ID that earned that score**. A score and a column ID have different meanings, even though both are Python numbers. The `i` in this feature loop has a different role from the row ID `i` inside `split_dataset`; variable names alone do not determine their meaning.

For candidate gains `[0.4, 0.1, 0.2]`, the running maximum stays `[0.4, 0.4, 0.4]` and feature 0 wins. Comparing only with the previous gain would let 0.2 beat 0.1 and wrongly replace the earlier 0.4 winner.

| Error from the attempts | Correct pattern / reason |
| --- | --- |
| Call `compute_information_gain(...)` without saving its return | `ig_i = compute_information_gain(...)` makes the score available for comparison |
| `if ig_1 > best_feature:` | Use `if ig_i > ig_max:`; compare gain with gain, and use letter `i`, not digit `1` |
| Update only `best_feature = i` | Also set `ig_max = ig_i`; otherwise every later positive gain beats the unchanged zero |
| Keep only the previous `ig_i` | Keep the maximum of **all** candidates seen so far |
| Loop over `len(y)` | Loop over `X.shape[1]`; labels count samples, not feature columns |
| Use `p1` after defining `p_1`, or `ig_1` after defining `ig_i` | Python treats each spelling as a separate name; stale notebook variables can hide the error |

These are wrong/correct patterns, not extra cells to run. The student's final addition of `ig_max = ig_i` fixes the missing running-maximum update.

Starting with `best_feature = -1` and `ig_max = 0.0` means **no strictly positive gain → return -1**. Check that sentinel before using it as an array index: Python otherwise treats `-1` as the last column. With strict `>`, equal positive gains keep the first feature. A zero-gain result means this greedy rule stops here; it does not prove that the data have no useful feature interactions (XOR is a classic counterexample).

### Keep trees separate from the neural-network mental model

| Idea | This binary decision-tree lab | Standard neural-network training |
| --- | --- | --- |
| How an input is processed | Ask a discrete feature question and follow one branch | Compute weighted sums plus biases, then activations across layers |
| What training chooses | Feature questions and recursive partitions, greedily using gain here | Numerical weights/biases, usually using gradients from backpropagation |
| What a “weight” means here | `w_left/right` are sample proportions calculated from branch sizes | Learned connection parameters in expressions such as `W @ x + b` |
| Role of labels | Count classes to evaluate each split; leaves predict classes | Compare outputs with targets through a loss |
| Prediction after training | Follow stored questions to a leaf | Run a forward pass with learned parameters |

Entropy here measures the mixture of labels in a node; it is not the NN's per-example prediction loss or a backpropagation step. Binary cross-entropy loss and binary entropy have related logarithms, but their inputs and jobs differ. Both models use labeled examples, and both need evaluation on unseen data. This comparison concerns the ordinary single-tree lab; boosted-tree methods elsewhere in this note do use loss gradients.

### Brain-burner self-check and quick reference

Try answering before reading the answer after the dash:

1. With `node_indices = [2, 3, 5]`, what does `range(len(node_indices))` give? — Positions `[0, 1, 2]`; actual original IDs are `2, 3, 5`.
2. Why store `i` when `X[i, feature] == 1`? — The value answers the question; the ID lets us retrieve that sample's features and label later.
3. If `y[5] == 1`, was row 5 predicted correctly? — Unknown until we compare a prediction with that label.
4. At the three-row node, which labels supply parent entropy? — `y[[2, 3, 5]]`, not all six labels.
5. Nine of ten samples go left and five of those nine are positive: what are `w_left` and left `p_1`? — `9/10` and `5/9`.
6. Gains arrive as `0.4, 0.1, 0.2`: what must remain stored? — Gain `0.4` and feature ID `0`.
7. Why update two variables in the `if`? — One remembers the best score; the other remembers where it came from.
8. All candidates have zero gain: what next? — Return `-1`; the builder stops instead of indexing column `-1`.

| Job | Quick pattern |
| --- | --- |
| Select current labels | `y_node = y[node_indices]` |
| Measure class mixture | Empty guard → `p_1 = sum(y_node)/len(y_node)` → endpoint guard → entropy |
| Route a sample | `for i in node_indices` → inspect `X[i, feature]` → append `i` |
| Score one question | Current-parent entropy minus child entropies weighted by `len(child)/len(node_indices)` |
| Choose a question | Loop over columns → save returned gain → compare with maximum → update score and feature together |
| Grow the tree | Check stopping rules → choose feature → route current IDs → repeat on children |
| Predict | Follow the already stored questions; do not recompute the best feature |

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

## 8a. Second lab: Ungraded Lab - Trees Ensemble

**Added 19 October 2026.** This walkthrough uses the uploaded `Pasted text(6).txt`. Section 3a's ten-animal Andrew Ng lab stays separate: there I build split calculations by hand; here libraries build and compare whole models.

### The story before the code

Imagine a question game. One decision tree asks a chain of questions and guesses a label. A random forest asks many different trees and combines their answers. Gradient boosting builds a team in order: each new tree tries to improve the team's current predictions using the loss gradient. For classification, that is more precise than saying it simply fits raw label residuals.

The worksheet is `heart.csv`, the **Heart Failure Prediction** dataset: **918 examples**, originally **11 input features** plus the binary target **HeartDisease** (`1` heart disease, `0` normal). The dataset's title is not the target's name; this is a heart-disease classification exercise.

| Original feature | What it records |
| --- | --- |
| Age, Sex | Age in years and recorded sex category |
| ChestPainType | Chest-pain category: TA, ATA, NAP, ASY |
| RestingBP, Cholesterol | Resting blood pressure and serum cholesterol |
| FastingBS | Already numeric binary indicator for fasting blood sugar above 120 mg/dl |
| RestingECG | Normal, ST, or LVH category |
| MaxHR, ExerciseAngina | Maximum heart rate and exercise-induced angina category |
| Oldpeak, ST_Slope | ST depression value and slope category (Up, Flat, Down) |

### Encoding and choosing X and y

One-hot encoding makes a yes/no column for each category: `Sex_M=True` means “this row has category M.” These columns do not imply a numeric ordering between categories. The lab encodes all five named categorical columns, **including the two-category Sex and ExerciseAngina**. Its prose about removing binary variables does not match the actual code.

Run this setup once; all code blocks in this section marked **Continue** depend on it. Install `numpy pandas matplotlib scikit-learn xgboost` in your active Python environment and place the lab's `heart.csv` in the working directory. The CSV is an external input, not bundled with these notes.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

RANDOM_STATE = 55
cat_variables = [
    "Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope",
]
raw = pd.read_csv("heart.csv")
df = pd.get_dummies(
    data=raw, prefix=cat_variables, columns=cat_variables,
)
features = [column for column in df.columns if column != "HeartDisease"]
X = df[features]
y = df["HeartDisease"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, train_size=0.8, random_state=RANDOM_STATE,
)
assert raw.shape == (918, 12)
assert X.shape == (918, 20) and y.shape == (918,)
assert "HeartDisease" not in X.columns
assert X_train.shape == (734, 20) and X_val.shape == (184, 20)
print(X_train.shape, X_val.shape)  # (734, 20) (184, 20)
```

`pd.get_dummies` replaces the five columns with 14 indicators: `2 + 4 + 3 + 2 + 3`. Six existing numeric features remain, giving **20 features** and **21 total columns including the target**. Depending on pandas/version/options, indicators may display as booleans or 0/1; both represent the same flags. `prefix` names each indicator's origin; `columns` chooses which inputs to encode. No `drop_first` is used.

The list comprehension reads “keep each column except HeartDisease.” The source uses `x not in 'HeartDisease'`, a **substring membership** test on a string. It happens to work for these names, but `column != "HeartDisease"` expresses exact exclusion. `raw.drop(columns="HeartDisease")` is another clear way to separate inputs. This is **target exclusion**, not a search for the most predictive subset of features. Including the answer as an input would leak the target.

### Training, validation, and test are different jobs

- **Training:** examples used by `fit` to learn split rules.
- **Validation:** examples used to choose hyperparameters or stopping time. Calling `predict` alone does not fit them, but choosing settings from their scores uses their information.
- **Test:** an untouched final exam used only after the entire selection procedure is fixed.

The lab uses `train_test_split(train_size=.8, random_state=55)`: 734 training rows and 184 validation rows, shuffled by default, **without stratification**. The function's name does not decide the role of its outputs. The lab sometimes labels validation scores “test”; because these rows help choose settings and compare models, they are validation, with **no separate untouched test set**. Adding `stratify=y` is often useful for classification but changes the lab split.

### Read the comparison loops without getting lost

A hyperparameter is a setting I choose before fitting; the tree's learned questions are model parameters. Each iteration below creates a **fresh estimator**, fits training rows, predicts both sets, computes accuracy (fraction correct), and stores the pair. `fit` returns the estimator, so `Model(...).fit(...)` can be chained. `.score(X, y)` for these sklearn classifiers also gives accuracy; explicit `accuracy_score(y_true, y_pred)` makes the calculation visible.

**Continue setup.** This reproduces the lab's separate sweeps, including the forest's estimator-count experiment. Printing a table avoids needing the course's plotting style file.

```python
def compare_setting(estimator_class, parameter, values):
    rows = []
    for value in values:
        model = estimator_class(
            random_state=RANDOM_STATE, **{parameter: value},
        ).fit(X_train, y_train)
        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        val_accuracy = accuracy_score(y_val, model.predict(X_val))
        rows.append((value, train_accuracy, val_accuracy))
    return pd.DataFrame(rows, columns=[parameter, "train_accuracy", "val_accuracy"])

split_values = [2, 10, 30, 50, 100, 200, 300, 700]
depth_values = [1, 2, 3, 4, 8, 16, 32, 64, None]
tree_split_scores = compare_setting(
    DecisionTreeClassifier, "min_samples_split", split_values,
)
tree_depth_scores = compare_setting(
    DecisionTreeClassifier, "max_depth", depth_values,
)
forest_split_scores = compare_setting(
    RandomForestClassifier, "min_samples_split", split_values,
)
forest_depth_scores = compare_setting(
    RandomForestClassifier, "max_depth", [2, 4, 8, 16, 32, 64, None],
)
forest_size_scores = compare_setting(
    RandomForestClassifier, "n_estimators", [10, 50, 100, 500],
)
print(tree_split_scores.to_string(index=False))
print(tree_depth_scores.to_string(index=False))
print(forest_split_scores.to_string(index=False))
print(forest_depth_scores.to_string(index=False))
print(forest_size_scores.to_string(index=False))
```

`**{parameter: value}` turns a dictionary into a named argument, for example `min_samples_split=50`. `.append` adds one result; it does not train the model. Each sweep changes **one setting while others stay at defaults**: the depth sweep does not carry forward the split threshold selected earlier. Combining selected values afterward is not guaranteed to be the best pair.

**Continue setup and the sweep block** for a curve like the lab's:

```python
import matplotlib.pyplot as plt

positions = range(len(depth_values))
plt.plot(positions, tree_depth_scores["train_accuracy"], label="Train")
plt.plot(positions, tree_depth_scores["val_accuracy"], label="Validation")
plt.xticks(list(positions), [str(value) for value in depth_values])
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()
```

The x positions are evenly spaced candidate indices, not literal numeric depths; `None` means no explicit depth cap. Read each train/validation pair together:

| Pattern | Interpretation | Possible adjustment |
| --- | --- | --- |
| High train score, weaker validation score | Possible overfitting: memorizing training quirks | Reduce depth or increase minimum split size |
| Both scores weak | Possible underfitting: too few useful questions | Allow more complexity; also inspect features/data |
| Similar, useful scores | Smaller observed generalization gap | Check stability on more splits; not proof of no overfitting |

An integer `min_samples_split=50` permits a split only when a node has at least 50 samples; it **does not require 50 samples in each child** (`min_samples_leaf` controls child support). A float such as `0.1` is a fraction of training samples, not the number 10 percent written as `10`. Shallower `max_depth` limits successive questions. Neither tuning direction guarantees better validation accuracy. The lab finds depth 4 useful; extremely shallow trees underfit. Its sweep alone does not establish a universal threshold such as “depth >= 5 always overfits.”

### Selected decision tree and random forest

**Continue setup.**

```python
decision_tree_model = DecisionTreeClassifier(
    max_depth=4, min_samples_split=50, random_state=RANDOM_STATE,
).fit(X_train, y_train)

# Faithful final lab configuration: random_state was omitted here.
random_forest_model = RandomForestClassifier(
    n_estimators=100, max_depth=16, min_samples_split=10,
).fit(X_train, y_train)

for name, model in [
    ("Decision tree", decision_tree_model),
    ("Random forest", random_forest_model),
]:
    print(
        f"{name}: train={accuracy_score(y_train, model.predict(X_train)):.4f}, "
        f"validation={accuracy_score(y_val, model.predict(X_val)):.4f}"
    )
# For a repeatable forest experiment, construct a new forest with
# random_state=RANDOM_STATE as well. Its score may differ from the source.
```

In a standard random forest (`bootstrap=True`), each tree gets a **bootstrap sample of rows, drawn with replacement**: some rows repeat and some are absent. At **each split**, a random subset of candidate features is considered; it is not one fixed feature subset assigned to the entire tree. sklearn classification combines the trees' class probabilities and chooses the class with the highest average probability (a useful refinement of the “trees vote” analogy). See the [official forest API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

| Forest setting | Meaning |
| --- | --- |
| n_estimators | Number of trees; more trees cost time/memory and often stabilize the average |
| max_features | Candidate features considered per split; default "sqrt" gives about sqrt(20), normally 4 here; search may inspect more to find a valid split |
| n_jobs | Parallel workers; None normally means 1, -1 uses all processors; a resource setting, not a regularizer |
| max_depth | Depth limit for each tree |
| min_samples_split | Minimum node sample count/fraction needed to attempt a split |
| random_state | Seed for bootstrap/feature randomness; the lab seeds sweeps but omits it in its final forest |

The individual forest trees can be deeper than the selected single tree because averaging diverse trees reduces variance; it does not make overfitting impossible. The source's estimator candidates are `[10, 50, 100, 500]`, even though its nearby prose mentions only three values.

### XGBoost: improve a team one round at a time

Gradient boosting adds trees **sequentially**, each guided by the current loss's gradients (XGBoost also uses second-order information). `learning_rate=.1` shrinks each new tree's contribution; `n_estimators=500` is the maximum number of boosting rounds here, not a promise to use all 500. The lab's binary, one-tree-per-round setup makes rounds and tree counts align.

The lab further divides the 734 training rows by position: `n = int(734 * .8) = 587` fit rows and **147 early-stopping evaluation rows**. It uses the first 587 and remaining 147 of an already shuffled training split; `.iloc` below makes positional slicing explicit. The outer 184 validation rows remain separate for model comparison.

**Continue setup.** In modern XGBoost, put `early_stopping_rounds` in the **constructor**, rather than `fit`. Use an explicit loss metric:

```python
from xgboost import XGBClassifier

n = int(len(X_train) * 0.8)
X_train_fit, X_train_eval = X_train.iloc[:n], X_train.iloc[n:]
y_train_fit, y_train_eval = y_train.iloc[:n], y_train.iloc[n:]
assert len(X_train_fit) == 587 and len(X_train_eval) == 147

xgb_model = XGBClassifier(
    n_estimators=500, learning_rate=0.1,
    early_stopping_rounds=10, eval_metric="logloss",
    random_state=RANDOM_STATE,
)
xgb_model.fit(
    X_train_fit, y_train_fit,
    eval_set=[(X_train_eval, y_train_eval)], verbose=False,
)
print("Best round index:", xgb_model.best_iteration)
print("Best evaluation logloss:", xgb_model.best_score)
print("Rounds stored:", xgb_model.get_booster().num_boosted_rounds())
print("Train:", accuracy_score(y_train, xgb_model.predict(X_train)))
print("Validation:", accuracy_score(y_val, xgb_model.predict(X_val)))

# Default sklearn-wrapper prediction uses the best round automatically.
best_predictions = xgb_model.predict(
    X_val, iteration_range=(0, xgb_model.best_iteration + 1),
)
assert (best_predictions == xgb_model.predict(X_val)).all()
```

`eval_set` is a list of `(X, y)` pairs monitored after each round. These rows do not supply the training gradients, but their loss chooses the stopping point. With multiple evaluation sets/metrics, the last set/metric controls early stopping by default. Log loss measures probability quality and penalizes confident wrong predictions; **lower is better**, unlike accuracy.

In the **supplied output**, `best_iteration=16` is zero-indexed: it means the **17th round**, with log loss **0.43948**. The source sentence “4.3948” is a typo. Rounds 17 through 26 give ten successive rounds without an improvement, so training stops after index **26**: **27 rounds were built**, not 26. Changing patience can change the run; its stopping index is not universally fixed.

The fitted booster normally retains trees through the stopping round. Nevertheless, `XGBClassifier.predict` and `predict_proba` automatically use rounds through `best_iteration` after early stopping. For this output, that means rounds 0–16 (`iteration_range=(0, 17)`, end excluded), not all 27 stored rounds. Native `Booster.predict` has different defaults; specify its iteration range explicitly when using the booster directly. See [XGBoost's prediction guide](https://xgboost.readthedocs.io/en/stable/prediction.html#early-stopping) and [sklearn API reference](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier). Constructor placement and this default prediction behavior follow the documented API; library versions/defaults can change exact scores.

### What the reported numbers say

These are **transcribed lab results, not newly reproduced benchmarks**:

| Selected model | Settings | Train accuracy | Outer validation accuracy |
| --- | --- | --- | --- |
| Decision tree | max_depth=4, min_samples_split=50, seed=55 | .8665 | .8696 |
| Random forest | n_estimators=100, max_depth=16, min_samples_split=10; final model unseeded | .9346 | .8913 |
| XGBoost | up to 500 rounds, learning_rate=.1, patience=10; best index 16 | .9251 | .8641 |

For XGBoost, the displayed “train” score covers **all 734 outer training rows**, including the 147 rows used for early stopping; it is not a pure fit-only score. A validation score slightly above training accuracy can happen with sampling variation; it is not a bug or proof of perfect generalization. The forest leads on this particular validation split, which is too little evidence for a universal winner. Accuracy alone also hides which class is missed.

### From lab sweeps to GridSearchCV and stronger preprocessing

One-parameter sweeps teach how a setting behaves. **GridSearchCV** tries combinations (for example, each candidate depth with each split threshold), fits each combination across training folds, and ranks mean validation scores. It can reveal interactions missed by separate default-based sweeps, at a higher compute cost. `best_score_` is a model-selection score, not final test performance. See the existing section 7 Iris example and the [sklearn tuning guide](https://scikit-learn.org/stable/modules/grid_search.html).

Repeatedly selecting hyperparameters, algorithms or stopping times on validation results can overfit those results. Reserve an untouched test set **before** this work, or use an appropriate nested evaluation procedure. The following independent real-world example changes the lab protocol: it reserves 20% for final testing and uses three-fold CV within the remaining data. It does not reproduce the lab's 734/184 split.

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

raw = pd.read_csv("heart.csv")
categorical = [
    "Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope",
]
X_raw = raw.drop(columns="HeartDisease")
y_raw = raw["HeartDisease"]
X_dev, X_test, y_dev, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, stratify=y_raw, random_state=55,
)
preprocess = ColumnTransformer(
    [("categories", OneHotEncoder(handle_unknown="ignore"), categorical)],
    remainder="passthrough",
)
pipeline = Pipeline([
    ("preprocess", preprocess),
    ("tree", DecisionTreeClassifier(random_state=55)),
])
search = GridSearchCV(
    pipeline,
    {"tree__max_depth": [3, 4, 8],
     "tree__min_samples_split": [10, 50]},
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=55),
    scoring="accuracy", refit=True,
)
search.fit(X_dev, y_dev)
print("Selected settings:", search.best_params_)
print("Mean CV accuracy:", search.best_score_)
# Freeze the procedure before this final evaluation; do not retune from it.
print("Final test accuracy:", search.score(X_test, y_test))
```

Inside each CV fold, the pipeline fits the encoder and tree on that fold's training rows only. Final refitting uses all development rows. `handle_unknown="ignore"` maps an unseen category to all zeros for that input's indicator block, avoiding an error. The `tree__` prefix routes each search setting to the pipeline step named `tree`.

This is stronger real-world practice than applying `pd.get_dummies` to the full table before splitting: it keeps the learned category vocabulary, column alignment and any future imputation/scaling inside training boundaries. **Fixed-category one-hot encoding alone does not prove target leakage**: it does not learn label relationships or numeric distribution statistics. The concern here is using held-out data to discover a vocabulary and maintaining preprocessing discipline, not claiming the lab's score is demonstrably inflated. Never fit an imputer/scaler or target-based encoder on held-out rows. See [pandas get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html), [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html), and [sklearn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

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

**Brain-burner follow-up verification, 21 October 2026:** ran all three section 4a Python blocks in sequence with exact expected-output comparisons and embedded assertions, including the new best-feature loop and uneven-child weight calculation. Checked running-maximum selection on controlled gain sequences, zero gain, equal-gain ties, iteration over feature count, pure/empty nodes, constant features and the XOR stopping limitation. Re-ran section 4's helper example and the sheet's indexing reminder with NumPy floating-point errors set to raise. Syntax-checked all 14 Python blocks in this note and 34 in the canonical sheet. All relative links/anchors in the note, sheet and updated learning log passed; prior text was preserved. Python 3.12.14, NumPy 2.3.5. Other existing examples received syntax checks only in this follow-up; the complete course notebook and grader were not run. Earlier records below retain their original scope.

- Graded-exercise context: the supplied chat excerpt includes UNQ_C2's split_dataset and the current request supplies the label meanings and first-attempt mistakes. Section 4a's six-row mushroom data are invented to demonstrate those concepts; the complete graded notebook and course grader were not retrieved or executed.

**Graded-exercise verification, 21 October 2026:** executed both section 4a Python blocks in order and the canonical sheet's new indexing reminder; exact printed outputs and all embedded assertions passed. Checked nested original-ID routing, current-node p_1/entropy/gain, unchanged X, empty-label and pure-label entropy, and zero gain for a constant feature/empty parent. Re-ran section 4's existing helper block with NumPy floating-point errors set to raise, including endpoint checks. Syntax-checked all 13 Python blocks in this note and all 34 in the sheet; checked balanced fences and all four new relative links/anchors. Compared against the main-branch originals: existing material is preserved, with only the sheet's navigation description replaced. Environment: Python 3.12.14, NumPy 2.3.5. Other existing examples were syntax-checked only in this update; their earlier runtime verification records below remain unchanged.

- Uploaded **Ungraded Lab - Trees Ensemble**, `Pasted text(6).txt`: second-lab code, printed scores and early-stopping log, explained in section 8a. No dataset CSV accompanied the text.
- [RandomForestClassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [pandas get_dummies](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html), [XGBoost sklearn API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBClassifier), and [prediction behavior](https://xgboost.readthedocs.io/en/stable/prediction.html#early-stopping): current-library corrections and adaptations.

**Trees Ensemble verification, 19 October 2026:** syntax-checked all 11 Python blocks in this note and all 33 in the canonical sheet; checked balanced code fences and the new navigation/detail anchors. Re-ran the original five note blocks and section 12a with their embedded assertions. Using a synthetic 918-row table with the lab's schema (not the original CSV), executed the new pandas encoding and 734/184 split assertions, all 36 decision-tree/forest sweep fits, selected tree/forest fits, the curve with the Agg backend, and the six-combination, three-fold preprocessing pipeline grid. Verified an unseen category maps to an all-zero indicator block and can be predicted. Executed the sheet's new setup and both tuning loops on that fixture; checked the 587/147 inner split sizes. Parsed the uploaded log to confirm 27 rounds, best index 16 with logloss .43948, and final index 26. Environment: Python 3.12.14, NumPy 2.5.2, pandas 3.0.5, scikit-learn 1.9.1, Matplotlib 3.11.2. **The two new XGBoost blocks were syntax-checked and reviewed against official documentation, not executed:** XGBoost was unavailable and could not be installed from the available package index. The original heart.csv was not supplied, so none of the quoted lab accuracies were reproduced; they remain attributed source results. Interactive plot display was not tested. Earlier verification records describe their original runs.

- Original ungraded Decision Trees lab excerpt: arrays, feature definitions, four functions and expected results supplied directly in the user's handoff; no complete notebook was retrieved.
- [Course 2 — Advanced Learning Algorithms](https://www.coursera.org/learn/advanced-learning-algorithms): decision-tree topic and lab context.
- [sklearn tree guide](https://scikit-learn.org/stable/modules/tree.html).
- [DecisionTreeClassifier API](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).
- [GridSearchCV API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).
- [Cost-complexity pruning example](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html).

**Verification, 19 October 2026:** ran all five Python blocks in this note offline, with the stated continuation dependencies, using Python 3.12.14, NumPy 2.5.2, scikit-learn 1.9.1 and Matplotlib 3.11.2. Assertions checked all three root splits and gains against an independent class-count entropy calculation, the depth-2 split choices and four pure leaves, node-local to original index mapping, the four-row illustrative result, and safer-helper empty-parent, empty-child, constant-feature and pure-node cases. The original empty-child call raised ZeroDivisionError as expected. The Iris GridSearchCV example and prediction/probability assertions passed; the plot continuation exported a PNG using the noninteractive Agg backend (GUI display was not tested). The canonical sheet's section 12a sklearn block also ran successfully. The complete course notebook and any course plotting/recursive-builder utilities were not supplied or executed; this verifies the handoff reproduction and examples here. Earlier verification records for other cheat-sheet sections remain unchanged.


