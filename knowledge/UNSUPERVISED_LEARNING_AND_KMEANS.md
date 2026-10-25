# Unsupervised ML and K-means — from coordinates to LLM embeddings

**Study note, 25 October 2026.** Read the tiny examples, run them, and explain each step aloud. This is a study resource, not course completion or a claim of mastery. All data below are invented teaching examples, not a reproduced course lab.

For syntax lookup, use the [canonical Python ML coding cheat sheet](PYTHON_ML_CODING_CHEAT_SHEET.md#12c-unsupervised-learning-and-k-means). For array basics, see [NumPy for ML](NUMPY_FOR_ML.md).

## Navigation

- [1. Learning without supplied labels](#1-learning-without-supplied-labels)
- [2. Features are coordinates](#2-features-are-coordinates)
- [3. Distances, min and argmin](#3-distances-min-and-argmin)
- [4. Read the nested loops](#4-read-the-nested-loops)
- [5. Update the centers and repeat](#5-update-the-centers-and-repeat)
- [6. Scaling and practical sklearn syntax](#6-scaling-and-practical-sklearn-syntax)
- [7. Choosing K and limitations](#7-choosing-k-and-limitations)
- [8. The connection to LLM engineering](#8-the-connection-to-llm-engineering)
- [9. Brain-burners and self-checks](#9-brain-burners-and-self-checks)
- [Sources and verification](#sources-and-verification)

## 1. Learning without supplied labels

Imagine putting similar toys into baskets without anyone telling you the correct basket for each toy. You still decide what “similar” means and how many baskets to use. K-means does a numerical version: put nearby points into K groups and represent each group by its average position.

**Unsupervised learning** looks for structure without supplied task labels such as spam/not-spam. Clustering is one task in this family; dimensionality reduction and density estimation are others. Unsupervised does not mean no objective, no human choices, or guaranteed meaningful discoveries.

| Setting | Training signal | Example |
| --- | --- | --- |
| Supervised classification | Supplied target classes | Ticket text → billing/network |
| Unsupervised clustering | Inputs without correct group labels | Group similar ticket embeddings |
| Self-supervised language pretraining | Targets constructed from text itself | Context tokens → next token |

For K-means, supply X and choose K; no y of correct classes is required. It learns center positions and assigns each sample an integer group ID. ID 0 does not inherently mean billing, good, or correct. Inspect the members before naming a group.

## 2. Features are coordinates

A feature is one measured property. In a two-feature dataset, one property can be the horizontal coordinate and the other the vertical coordinate. **One row is one point with both coordinates.**

```python
import numpy as np

X = np.array([[1., 1.], [2., 1.], [8., 8.], [9., 8.]])
centroids = np.array([[1., 1.], [9., 8.]])
m, n = X.shape
K = centroids.shape[0]

assert (m, n, K) == (4, 2, 2)
assert X[1].shape == (2,)
assert X[:, 0].shape == (4,)
assert X[1, 0] == 2.
```

These are abstract coordinates. In real data they could be duration and error rate, but then their units matter.

| Expression | Shape | Meaning |
| --- | --- | --- |
| X | (m, n) = (4, 2) | Four samples, two features each |
| X[i] | (n,) = (2,) | All coordinates of sample i |
| X[i, f] | scalar | Feature f of sample i |
| X[:, f] | (m,) | One feature across all samples |
| centroids | (K, n) = (2, 2) | Two centers, two coordinates each |
| centroids[j] | (n,) | Complete position of center j |
| idx | (m,) | One integer cluster ID per sample |
| Distance matrix | (m, K) | One distance per sample–center pair |

**A sample is an input row. A centroid is an average location.** A center need not equal any input row. Both have n coordinates because they occupy the same feature space.

Two features do not require two clusters: K=3 with n=2 gives centers of shape (3, 2). A flattened 20 × 20 image is one point with 400 coordinates. We cannot draw all those axes, but the arithmetic still works.

For one feature, np.array([[1.], [2.], [8.]]) has shape (3, 1): three rows, one column. The flat array np.array([1., 2., 8.]) has shape (3,), so it is not that two-dimensional matrix.

## 3. Distances, min and argmin

For two points [x1, x2] and [c1, c2], Euclidean distance is:

`sqrt((x1 - c1)**2 + (x2 - c2)**2)`

Subtract corresponding coordinates, square each difference, sum them, and take the square root. With n features, sum over all n coordinates.

```python
import numpy as np

point = np.array([1., 1.])
center = np.array([4., 5.])
delta = point - center
squared_distance = np.sum(delta ** 2)
distance = np.sqrt(squared_distance)

np.testing.assert_array_equal(delta, [-3., -4.])
assert squared_distance == 25.
assert distance == np.linalg.norm(point - center) == 5.

distances = [5., 1.5, 4.]
assert min(distances) == np.min(distances) == 1.5
assert np.argmin(distances) == 1
assert np.argmin([2., 2.]) == 0  # first tied minimum
```

### The 1.5 confusion: value versus position

For distances = [5.0, 1.5, 4.0]:

- **min returns 1.5:** how far away is the closest center?
- **argmin returns 1:** which list position contains that smallest distance?
- Positions start at zero, so 1 identifies the second centroid.

Therefore **idx[i] = np.argmin(distances)** stores 1, not 1.5. idx stores identities, not measurements. See [NumPy argmin](https://numpy.org/doc/stable/reference/generated/numpy.argmin.html) for axis and tie behavior.

Square roots preserve the order of nonnegative values, so assignment can compare squared distances instead. K-means optimizes squared Euclidean error; ordinary distance is useful for the geometric explanation.

## 4. Read the nested loops

**Continue section 2.** First assign points to fixed centers; moving the centers is a separate step.

```python
# Continue section 2.
def find_closest_centroids(X, centroids):
    m, n = X.shape
    K = centroids.shape[0]
    assert centroids.shape[1] == n
    idx = np.zeros(m, dtype=int)

    for i in range(m):                     # select ONE sample
        distances = []                     # fresh list for THIS sample
        for j in range(K):                 # visit EVERY center
            distance = np.linalg.norm(X[i] - centroids[j])
            distances.append(distance)     # save one scalar
        idx[i] = np.argmin(distances)        # choose after all K comparisons

    return idx

idx = find_closest_centroids(X, centroids)
np.testing.assert_array_equal(idx, [0, 0, 1, 1])
assert idx.shape == (4,)
```

Read the indentation as a schedule:

1. i selects a row.
2. Start a new distance list for that row.
3. j visits all centers while i stays fixed.
4. Save each scalar distance in center order.
5. After all comparisons, save the winning center ID.
6. Move to the next row.

| Row | Distances to centers 0 and 1 | Winner |
| --- | --- | --- |
| [1, 1] | [0, sqrt(113)] | 0 |
| [2, 1] | [1, sqrt(98)] | 0 |
| [8, 8] | [sqrt(98), 1] | 1 |
| [9, 8] | [sqrt(113), 0] | 1 |

**distances.append(distance)** adds the number just calculated to a Python list. It does not add the numbers together, choose a winner or move a center. After K appends the list contains K numbers. Do not assign the return value of append back to distances: append changes the list and returns None.

**Where is the feature loop?** X[i] - centroids[j] subtracts whole vectors; np.linalg.norm combines the feature differences. NumPy handles that work. Fully manual code would have three levels: samples i → centers j → features f. j means centroid in this code, not feature; names depend on context.

This vectorized equivalent continues the same setup:

```python
# Continue sections 2 and 4.
delta = X[:, None, :] - centroids[None, :, :]  # (m, K, n)
D2 = np.sum(delta ** 2, axis=2)               # (m, K)
idx_vectorized = np.argmin(D2, axis=1)        # (m,)
np.testing.assert_array_equal(idx_vectorized, idx)
```

The inserted size-one axes let every row meet every center. Summing axis 2 removes the feature dimension. Argmin along axis 1 chooses a center per row. Without an axis, argmin returns one position in the flattened distance matrix.

This version allocates an (m, K, n) intermediate. It teaches broadcasting but can be too large for an embedding collection; use library implementations for practical scale.

## 5. Update the centers and repeat

Average each group's assigned rows **column by column**:

- Group 0 contains [1, 1] and [2, 1] → new center [1.5, 1].
- Group 1 contains [8, 8] and [9, 8] → new center [8.5, 8].

The 1.5 here is a **coordinate**, calculated as (1 + 2) / 2. It is neither an ID nor necessarily a distance. Ask what role each number plays.

```python
# Continue sections 2 and 4.
def update_centroids(X, idx, old_centroids):
    new_centroids = old_centroids.astype(float, copy=True)
    for j in range(len(old_centroids)):
        members = X[idx == j]
        if len(members) > 0:
            new_centroids[j] = members.mean(axis=0)
        # Teaching policy: keep the old center for an empty group.
    return new_centroids

updated = update_centroids(X, idx, centroids)
np.testing.assert_allclose(updated, [[1.5, 1.], [8.5, 8.]])
assert updated.shape == (2, 2)
```

idx == j creates one Boolean per sample. X[idx == j] selects the matching rows. mean(axis=0) averages across those rows and returns n feature means. Calling mean() without an axis collapses everything into one scalar.

An empty group's mean is invalid, so handle it explicitly. Keeping the old center is a simple teaching policy that may leave a group empty; it does not reproduce sklearn's internal empty-center handling. Production implementations can relocate empty centers. Floating-point centers also prevent integer arrays from truncating values such as 1.5.

### The complete cycle

Initialize K centers → assign points → update centers → repeat. This is Lloyd's K-means algorithm. The objective is:

**SSE = sum over i of ||X[i] - centroids[idx[i]]||²**

Some course formulas divide by m; that rescales the objective without changing its minimizer. sklearn calls the unnormalized sum inertia_. Assignment chooses the best center while centers stay fixed. Updating to each mean minimizes that group's squared error while assignments stay fixed. No learning rate is needed for the mean update.

```python
# Continue sections 2, 4 and 5.
centers = centroids.copy()
history = []
max_iter = 100
tol = 1e-8

for iteration in range(max_iter):
    labels = find_closest_centroids(X, centers)
    new_centers = update_centroids(X, labels, centers)
    sse = np.sum((X - new_centers[labels]) ** 2)
    history.append(float(sse))
    shift = np.max(np.linalg.norm(new_centers - centers, axis=1))
    centers = new_centers
    if shift <= tol:
        break

labels = find_closest_centroids(X, centers)  # match final centers
np.testing.assert_allclose(centers, [[1.5, 1.], [8.5, 8.]])
np.testing.assert_array_equal(labels, [0, 0, 1, 1])
np.testing.assert_allclose(history, [1., 1.])
assert np.all(np.diff(history) <= 1e-10)
assert iteration + 1 == 2
```

Initial SSE was 2. The first update reduces it to 1. The next cycle changes nothing, so this loop stops after two cycles. These well-separated points keep their assignments; other data can change membership over several cycles.

**Convergence** means a chosen stopping rule is satisfied. Rules can check center movement, unchanged assignments or objective change; always cap iterations too. This loop uses maximum center movement, whereas sklearn has its own tolerance criterion. Exhausting max_iter does not demonstrate convergence.

Exact assignment and mean steps cannot increase SSE, apart from numerical effects and implementation policies. They can settle at a local solution. Initialization matters; multiple restarts and K-means++ help find better solutions. A fixed seed supports repeatability, not correctness. Consult the [KMeans API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) for these options and stopping details.

## 6. Scaling and practical sklearn syntax

Distances depend on coordinate units. Differences of 100 milliseconds and 0.1 in error-rate units contribute 10,000 and 0.01 to squared distance. Converting milliseconds to seconds changes their relative influence.

Standardization often helps numeric measurements: subtract each training-column mean and divide by its scale. It changes geometry, so equalizing scales must match the task. It is not regularization and does not remove outliers. See [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

This independent example keeps preparation and clustering together:

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Synthetic duration/error-rate measurements.
X_train = np.array([[100., .01], [110., .02],
                    [900., .40], [910., .42]])
pipeline = make_pipeline(
    StandardScaler(),
    KMeans(n_clusters=2, init="k-means++", n_init=10,
           random_state=42, max_iter=300),
)
labels = pipeline.fit_predict(X_train)  # no y required
new_labels = pipeline.predict([[105., .015]])
km = pipeline.named_steps["kmeans"]

assert labels.shape == (4,)
assert labels[0] == labels[1]
assert labels[2] == labels[3]
assert labels[0] != labels[2]            # numeric IDs are arbitrary
assert new_labels[0] == labels[0]
assert km.cluster_centers_.shape == (2, 2)
assert pipeline.transform(X_train).shape == (4, 2)
```

fit_predict learns from these rows and returns IDs. predict assigns new rows using saved preparation and centers; it does not move centers. transform returns distances to every center, not IDs. Here centers and distances use standardized coordinates. The fitted scaler's inverse_transform(km.cluster_centers_) expresses the centers in original units.

For future-data evaluation, fit preparation and clustering on the training/reference portion, then reuse them on held-out/new rows. Do not refit a scaler on every incoming batch. For exploratory clustering of a whole collection, state which collection was fitted. See the [fit/transform deep dive](FIT_TRANSFORM_PREDICT_DEEP_DIVE.md).

## 7. Choosing K and limitations

K requests a number of groups; it is not a feature count or a discovered truth.

- An **elbow plot** compares K with inertia. The optimal inertia falls or stays equal as K increases, so simply choosing the smallest value favors more groups. Actual runs also depend on local solutions.
- A **silhouette score** compares cohesion and separation. It requires at least two clusters and fewer clusters than samples. It measures geometry, not topic usefulness.
- Inspect representative items, sizes, stability across seeds and the actual task. If known labels are available for evaluation, use measures that allow arbitrary cluster-ID permutations rather than comparing IDs directly with class numbers.

K-means favors compact groups under Euclidean geometry. Curved groups, unequal densities/spreads, outliers and irrelevant features can produce poor partitions. Every sample receives a group even if it fits badly. High-dimensional distances can be hard to interpret; representation and metric choices matter.

Density-based methods such as DBSCAN/HDBSCAN can address different shapes and noise; hierarchical methods offer another grouping view. Each has assumptions, so evaluate alternatives. MiniBatchKMeans uses approximate batch updates for scale. The [sklearn clustering guide](https://scikit-learn.org/stable/modules/clustering.html#k-means) explains methods and evaluation.

## 8. The connection to LLM engineering

**Yes, K-means can be useful in ML/LLM engineering.** One application is organizing text represented as vectors. K-means itself does not read sentences or learn to generate language.

**Tickets/documents → embedding model → one vector per item → clustering → inspect groups**

Possible uses include discovering recurring support issues, exploring document topics, selecting diverse examples for manual review and organizing retrieval candidates. For telecom tickets, inspect whether groups capture SIP registration problems, call drops or billing issues. These are proposed applications, not measured repository results. Restricting retrieval to one cluster can miss relevant documents across boundaries; compare recall against an appropriate search baseline.

### Embeddings are learned coordinates

For m chunks with d embedding dimensions, E.shape is (m, d). Each row is one chunk. Coordinates are learned representation components, usually not individually named properties. Keep the embedding model/version, pooling and preparation consistent. Arbitrary hidden states from a generative LLM are not automatically useful sentence embeddings. [Sentence Transformers' similarity guide](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html) illustrates encoding and comparison.

This independent offline example uses **invented 3-D vectors**, not real model output or evidence of semantic quality:

```python
import numpy as np
from sklearn.cluster import KMeans

texts = [
    "SIP registration timeout", "Cannot register the phone",
    "Invoice amount is wrong", "Unexpected billing charge",
]
E = np.array([[1., .10, 0.], [.95, .15, 0.],
              [0., .10, 1.], [0., .15, .95]])
norms = np.linalg.norm(E, axis=1, keepdims=True)
assert np.all(norms > 0)  # zero vectors require an explicit policy
E_unit = E / norms

model = KMeans(n_clusters=2, n_init=10, random_state=42)
groups = model.fit_predict(E_unit)
assert groups[0] == groups[1]
assert groups[2] == groups[3]
assert groups[0] != groups[2]
for j in range(2):
    members = np.flatnonzero(groups == j)
    print([texts[i] for i in members])
```

Expected groups, in arbitrary ID order: the two registration texts together and the two billing texts together. This happens because we designed the vectors that way. A real experiment first generates embeddings from actual text using a chosen model, then examines mixed groups and errors. Topic names come from later interpretation.

### Cosine similarity and normalization

Cosine similarity compares directions. For **unit-length** vectors u and v:

**||u - v||² = 2 - 2 × dot(u, v) = 2 × (1 - cosine_similarity(u, v))**

Nearest-neighbor rankings by Euclidean distance and cosine similarity therefore agree for unit vectors. Row L2 normalization changes each vector's length; column standardization changes each feature's scale. They are different.

**Ordinary K-means on normalized inputs is not identical to spherical K-means.** Arithmetic-mean centers generally are not unit vectors. Assignment to centers with different lengths need not match cosine assignment. Spherical K-means normalizes centers and uses an angular objective. Follow the embedding model's metric guidance and evaluate; do not blindly standardize embedding columns as unrelated physical measurements.

### Clustering versus self-supervised LLM pretraining

Next-token pretraining constructs targets from text. For “the phone is ringing,” a context corresponding to “the phone is” supplies the next-token target; real tokenizers may split words into multiple tokens. A causal language model predicts token distributions, computes a loss such as cross-entropy, and updates neural weights with backpropagation and an optimizer. Masked-language modeling instead reconstructs masked tokens.

Nobody needs to annotate every next token, but there **is a prediction target**. **Self-supervised** is the precise term. It is sometimes placed under a broad unsupervised umbrella because the original text is unannotated; this does not make it clustering. See the [Hugging Face LLM course](https://huggingface.co/learn/llm-course/en/chapter1/4).

| Question | K-means on fixed embeddings | Self-supervised language pretraining |
| --- | --- | --- |
| What changes? | K center vectors | Neural-network parameters |
| Objective | Within-group squared error | Token prediction/reconstruction loss |
| Targets | No supplied correct topic labels | Automatically constructed from text |
| Result | Group IDs and centers | A trained language model |
| Does this update the text encoder? | No | Representation parameters can change |

A pretrained model may create embeddings before K-means runs. K-means then learns centers in that fixed space. It does not improve the language model's weights or replace pretraining.

## 9. Brain-burners and self-checks

| Confusion to revisit | Correction |
| --- | --- |
| Are two features two samples? | A row is a sample; its entries are coordinates. |
| Must K equal n? | K counts groups; n counts coordinates. |
| What does append calculate? | Nothing new; it saves the scalar just calculated. |
| Can distances live outside both loops? | Reset it for each sample to avoid mixing rows. |
| Where does argmin belong? | After all K center comparisons for this row. |
| Why is the answer 1 if the minimum is 1.5? | 1 is a position; 1.5 is the value there. |
| Why not average every number together? | A center needs one mean per feature: axis=0. |
| Is cluster 1 the positive class? | No; IDs are arbitrary and can swap across runs. |
| Does convergence prove the topics are correct? | No; numerical stability does not validate meaning. |
| Is this how LLMs learn language? | Clustering groups vectors; pretraining learns predictions. |

The retrieved chat preserves the append/argmin discussion and the request to explain it “like I am 5.” The other questions are teaching prompts, not claims about unseen conversation turns.

Try before reading the answers:

1. X.shape is (100, 5), K=3. What are center and assignment shapes?
2. For distances [4., .5, 2.], what do min and argmin return?
3. What is the center of [[2, 4], [4, 8]]?
4. Which loop repeats K times for each of m samples?
5. Is a cluster ID a probability or known topic name?
6. Does clustering fixed embeddings update the encoder?

**Answers:** (1) (3, 5) and (100,); (2) .5 and 1; (3) [3, 6]; (4) the inner center loop; (5) neither; (6) no.

## Sources and verification

The linked official NumPy, sklearn, Sentence Transformers and Hugging Face references support API and terminology checks. Coordinates, loops and telecom applications here are teaching examples, not reported course or production results.

**Verified 25 October 2026:** all eight Python blocks ran offline in their stated order, including the three explicitly marked continuation sections. Assertions checked coordinates, shapes, distances, min/argmin, exact assignments, center updates, the two-cycle SSE history, sklearn fit/predict/transform and the two expected synthetic text groups (allowing arbitrary ID order). Additional checks covered first-index ties, an empty group, floating means from integer centers, preservation of input centers, K different from n, loop/vectorized parity and nonincreasing SSE over 30 cycles on generated 3-D data.

The canonical sheet's new section also ran; all 35 sheet Python blocks passed syntax checks, with existing content preserved. New relative links/anchors and fences passed. Environment: Python 3.12.14, NumPy 2.5.2, scikit-learn 1.9.1. Windows core-count/MKL warnings appeared on the first run; the final check used OMP_NUM_THREADS=1 and LOKY_MAX_CPU_COUNT=1. Existing sheet sections were syntax-checked, not all runtime-tested again. Real embedding generation, language-model training and any course grader were not run.

