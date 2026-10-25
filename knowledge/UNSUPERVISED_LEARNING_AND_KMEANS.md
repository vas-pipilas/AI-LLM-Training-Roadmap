# Unsupervised ML and K-means — from coordinates to LLM embeddings

**Study note, updated 25 October 2026.** Read the tiny examples, run them, and explain each step aloud. This is a study resource, not a claim of detailed mastery. Runnable arrays below are invented teaching examples. Completed-lab progress and recorded output are attributed to the user's handoff and uploaded lab.

### Completed K-means exercises

| Andrew Ng ML Specialization exercise | Recorded status |
| --- | --- |
| Exercise 1: `find_closest_centroids` | Completed; assignment implementation and walkthrough preserved in section 4. |
| Exercise 2: `compute_centroids` | Completed; user confirms all unit tests passed. The accepted loop and a full teaching function appear in section 5. |

**Full Andrew Ng K-means clustering lab completed, as reported by the user.** The supplied full lab includes passing outputs for both graded functions, the sample-data iteration, random initialization and RGB image reconstruction. This does not mark the specialization complete.

**Self-assessment:** high-level understanding achieved; deeper fluency requires hands-on reuse. Reading the code and passing the lab do not yet establish detailed mastery. Revisit the reshape and palette lookup below by running them on another image and explaining each intermediate shape.

The uploaded full lab records these Exercise 2 centroids (correcting the earlier screenshot transcription):

| Centroid index | Feature 0 mean | Feature 1 mean |
| --- | --- | --- |
| 0 | 2.42830111 | 3.15792418 |
| 1 | 5.81350331 | 2.63365645 |
| 2 | 7.11938687 | 3.6166844 |

This is a `(3, 2)` result: three groups, two feature means per group. It records one centroid-update result; completion of the full lab is recorded separately above. The course data and grader were not rerun for this documentation update.

Start with [Exercise 1 mistakes](#exercise-1-mental-gymnastics-counts-vectors-and-identities), [Exercise 2 masks and means](#exercise-2-mental-gymnastics-one-group-all-its-rows), or the [syntax lookup](#k-means-syntax-lookup).

For syntax lookup, use the [canonical Python ML coding cheat sheet](PYTHON_ML_CODING_CHEAT_SHEET.md#12c-unsupervised-learning-and-k-means). For array basics, see [NumPy for ML](NUMPY_FOR_ML.md).

## Navigation

- [1. Learning without supplied labels](#1-learning-without-supplied-labels)
- [2. Features are coordinates](#2-features-are-coordinates)
- [3. Distances, min and argmin](#3-distances-min-and-argmin)
- [4. Read the nested loops](#4-read-the-nested-loops)
- [5. Update the centers and repeat](#5-update-the-centers-and-repeat)
- [5a. Random starting rows](#5a-random-starting-rows)
- [5b. Image compression: pixels to palette and back](#5b-image-compression-pixels-to-palette-and-back)
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

### Exercise 1 mental gymnastics: counts, vectors and identities

The mistakes in the continuation handoff mix three different questions: **how many rows**, **which row**, and **which center won**. Write the role next to each variable before writing more code.

| Attempt / confusion | What it actually means | Correct pattern |
| --- | --- | --- |
| Use `X[:, 1]` as the sample count | An array containing feature 1 for every row | `m = X.shape[0]`, an integer count |
| Read `X[i]` as only an x coordinate, `centroids[j]` as a y coordinate | Each expression is a whole feature vector | Subtract matching features: `X[i] - centroids[j]` |
| Write `np.X` or `np.centroids` | Look for attributes named X/centroids in the NumPy module | Use local variables `X`, `centroids`; use `np` for functions such as `np.mean` |
| `np.linalg.norm(X[i], centroids[j])` | The second argument is the norm-order option `ord`, not a second point | `np.linalg.norm(X[i] - centroids[j])` |
| Save the smallest distance into `idx[i]` | Save a measurement where a group identity is required | `idx[i] = np.argmin(distances)` |
| Use `np.min` to choose the center ID | Return the smallest value | `np.argmin` returns its zero-based position |

For `X = [[2, 4], [4, 8], [10, 20]]`, `X.shape` is `(3, 2)`, `X.shape[0]` is `3`, and `X[:, 1]` is `[4, 8, 20]`. `range` needs the count 3; that column is data, not a count. Meanwhile `X[0]` is `[2, 4]`: both features of sample 0.

Coordinates here mean positions in **feature space**, not necessarily physical map coordinates. A row could contain call duration and packet-loss rate, or hundreds of learned embedding components. `i` selects the sample; `j` selects the centroid. Neither variable singles out a feature in `X[i] - centroids[j]`.

For point `[1, 1]` and center `[4, 5]`, subtract first: `[-3, -4]`. Its norm is 5. Passing the two vectors as two arguments does not ask NumPy for that distance. See the [norm signature](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html).

#### Follow one outer-loop iteration

Suppose the three center distances for sample i are `[5, 1.5, 4]`:

| Moment | `j` | `distances` after append | Next action |
| --- | --- | --- | --- |
| Start sample i | — | `[]` | Compare center 0 |
| First comparison | 0 | `[5]` | Compare center 1 |
| Second comparison | 1 | `[5, 1.5]` | Compare center 2 |
| Third comparison | 2 | `[5, 1.5, 4]` | Exit inner loop |
| Choose once for sample i | — | `[5, 1.5, 4]` | Save `idx[i] = 1` |

The center ID is 1 because the winning value sits at position 1. A distance of 1.5 is not a valid center index. Even if the distance happened to be an integer, it would still have the wrong meaning.

Indentation controls when the actions happen. Resetting `distances` inside the inner loop erases earlier comparisons; resetting it outside the outer loop mixes samples. Appending after the inner loop saves only the last distance. Choosing with argmin inside the inner loop repeatedly chooses from an unfinished list. Use the section 4 function: reset once per sample, append once per center, choose once after all centers. `return idx` belongs after both loops so every sample is processed.

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

### Exercise 2 mental gymnastics: one group, all its rows

The accepted exercise body was:

```python
# Body inside compute_centroids, after allocating centroids with shape (K, n).
# Continue section 2's X, idx and K for this runnable illustration.
centroids_for_exercise = np.zeros((K, X.shape[1]))
for i in range(K):
    points = X[idx == i]
    centroids_for_exercise[i] = np.mean(points, axis=0)
np.testing.assert_allclose(centroids_for_exercise, [[1.5, 1.], [8.5, 8.]])
```

In the submitted loop the destination was named `centroids[i]`. The example uses `centroids_for_exercise[i]` to preserve the earlier initial centers for the lifecycle demonstration below. The three operations are exactly the same: **visit a group → collect every member → save its feature means**.

Exercise 1's outer loop visits **m samples**. Exercise 2's loop visits **K centroid slots**. Here `i` means a cluster ID; in Exercise 1 it meant a sample row ID. Python does not give the letter i a permanent meaning. Renaming it `cluster_id` can make the second job easier to read. The Boolean mask already checks all m assignments, so Exercise 2 needs no explicit inner sample loop.

#### Tiny data with interleaved groups

Use five rows and three groups so m, K and n are visibly different:

| Original row ID | Feature 0 | Feature 1 | `idx[row_id]` |
| --- | --- | --- | --- |
| 0 | 2 | 4 | 0 |
| 1 | 10 | 20 | 1 |
| 2 | 4 | 8 | 0 |
| 3 | 14 | 28 | 1 |
| 4 | 30 | 60 | 2 |

Think of idx as a group sticker attached to each row. To update basket 0, inspect **every sticker** and keep all rows with sticker 0. Do not inspect only sticker position 0.

For `i = 0`, `idx[i] == i` asks only “is the sticker on row 0 equal to 0?” It returns one Boolean, True. `idx == i` asks that question for every sticker: `[True, False, True, False, False]`. For `i = 1`, the single check is also True but still says nothing about row 3. For `i = 2`, the single check is False even though row 4 belongs to cluster 2. Row positions and cluster IDs serve different jobs.

`X[idx == i]` applies that full mask to X's rows. True means keep the whole row, including all n features; False means skip it. It does not change X. See [NumPy Boolean indexing](https://numpy.org/doc/stable/user/basics.indexing.html#boolean-array-indexing).

```python
import numpy as np

X_demo = np.array([[2., 4.], [10., 20.], [4., 8.],
                   [14., 28.], [30., 60.]])
idx_demo = np.array([0, 1, 0, 1, 2])
i = 0
mask_demo = idx_demo == i
points_demo = X_demo[mask_demo]
assert X_demo.shape == (5, 2)
assert bool(idx_demo[i] == i) is True       # one entry checked
np.testing.assert_array_equal(mask_demo, [True, False, True, False, False])
np.testing.assert_array_equal(points_demo, [[2., 4.], [4., 8.]])
assert points_demo.shape == (2, 2)
np.testing.assert_allclose(np.mean(points_demo, axis=0), [3., 6.])
```

Do not replace the array mask with `X[idx[i] == i]`. A scalar Boolean is not an m-entry row selector; NumPy's scalar-Boolean indexing has different shape behavior. Build the full mask `idx == i`.

#### The extra `[i]` throws away the group

After selection, `points` already contains the rows needed for **this** centroid. `points[i]` selects just one row within that smaller array. Its indices are local positions in the selection; the cluster ID i is not a row position inside it.

For cluster 0, `points = [[2, 4], [4, 8]]`. The correct calculation is two separate column means: `(2 + 4) / 2 = 3` and `(4 + 8) / 2 = 6`. The new center is `[3, 6]`.

| Expression on those points | Result | What survived? |
| --- | --- | --- |
| `points` | `[[2, 4], [4, 8]]`, shape `(2, 2)` | Both samples and both features |
| `points[0]` | `[2, 4]`, shape `(2,)` | Only the first sample |
| `np.mean(points, axis=0)` | `[3, 6]`, shape `(2,)` | One mean for each feature |
| `np.mean(points[0], axis=0)` | `3.0`, scalar | An average of two features from one sample |
| `np.mean(points, axis=1)` | `[3, 6]`, shape `(2,)` | One mean per sample; accidental numeric match here |
| `np.mean(points)` | `4.5`, scalar | All samples and features collapsed together |

The accidental match in the axis=1 row is why checking one result is not enough. For cluster 1, `points = [[10, 20], [14, 28]]`: axis=0 returns `[12, 24]`, while axis=1 returns `[15, 21]`.

**Read `axis=0` as “combine the rows, keep the feature columns.”** For shape `(r, n)`, it removes the r dimension and returns n values. Read `axis=1` as “combine the feature columns inside each row,” leaving r values. With `points[i]` the array is already 1-D; its axis 0 now spans feature entries, so averaging it produces a scalar. Adding axis=0 cannot restore the samples discarded by `[i]`. The [mean documentation](https://numpy.org/doc/stable/reference/generated/numpy.mean.html) defines this axis reduction.

Assigning that incorrect scalar to `centroids[i]` can silently broadcast it across all n feature slots. For cluster 0, the wrong calculation would fill `[3, 3]`, not `[3, 6]`. For cluster 2, `points` has one row, so `points[2]` would raise an index error. A cluster ID can be larger than its number of members.

#### Trace the whole update

| Loop i | `idx == i` | Selected original rows | `points.shape` | Mean saved into `centroids[i]` |
| --- | --- | --- | --- | --- |
| 0 | `[T, F, T, F, F]` | 0, 2 | `(2, 2)` | `[3, 6]` |
| 1 | `[F, T, F, T, F]` | 1, 3 | `(2, 2)` | `[12, 24]` |
| 2 | `[F, F, F, F, T]` | 4 | `(1, 2)` | `[30, 60]` |

idx stays fixed during this update. It came from the assignment step. The next assignment step may change those stickers after the centers move.

Here is the full exercise-style function, under the **nonempty-cluster assumption**:

```python
import numpy as np

def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))  # floating output preserves fractional means
    for i in range(K):          # i is a centroid index in THIS function
        points = X[idx == i]    # all assigned rows, shape (r, n)
        centroids[i] = np.mean(points, axis=0)  # shape (n,)
    return centroids

# Continue the independent five-row demo above.
result_demo = compute_centroids(X_demo, idx_demo, 3)
np.testing.assert_allclose(result_demo, [[3., 6.], [12., 24.], [30., 60.]])
assert result_demo.shape == (3, 2)
```

**Empty cluster caveat:** if no assignments equal i, points has shape `(0, n)`. Its mean produces invalid values (NaNs with warnings). The exercise-style function above does not handle that case. For reusable code, use the earlier `update_centroids` helper and its explicit keep-old-center policy, or choose a documented reinitialization policy. The three-argument exercise function has no old centers to keep. A valid single-member group has shape `(1, n)` and its mean is that one complete row.

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

## 5a. Random starting rows

The lab initializes centers by shuffling the sample **indices**, then taking the first K complete rows:

```python
import numpy as np

def kMeans_init_centroids(X, K):
    randidx = np.random.permutation(X.shape[0])
    return X[randidx[:K]]

# Independent teaching data: each row is distinct.
init_demo = np.arange(30.).reshape(10, 3)
starting_demo = kMeans_init_centroids(init_demo, 4)
assert starting_demo.shape == (4, 3)
assert np.unique(starting_demo, axis=0).shape[0] == 4
assert all(np.any(np.all(init_demo == row, axis=1)) for row in starting_demo)
```

For m=10, a shuffled order might begin `[7, 2, 9, 0, ...]`; K=3 selects rows 7, 2 and 9. The shuffle prevents selecting the same **row index** twice. Different rows can still have identical RGB values, so distinct indices do not guarantee distinct colors. Require 1 <= K <= m.

These are initial guesses, not final means. Feed them into the [existing assignment/update cycle](#the-complete-cycle). The lab uses ten cycles: assign all samples to the current centers, then average their assigned rows. Changing max_iters changes the number of cycles, not K. After the last mean update, recompute assignments against the final centers before reconstruction; the lab's run_kMeans returns assignments made just before that update.

The earlier lifecycle section already covers restarts and the distortion objective: compare separate runs at the same K and keep the lowest squared-error solution. Random row selection here differs from the K-means++ initialization used in the sklearn example.

## 5b. Image compression: pixels to palette and back

Think of a paint box: the original picture uses thousands of shades. K-means learns a box of **16 representative colors**, then each pixel gets the number of its closest paint. These colors are averages learned in RGB space, not simply the 16 most frequent exact colors.

### One pixel is one sample; RGB are its features

The lab's bird image has shape `(128, 128, 3)`: 128 rows, 128 columns, and three channels in red/green/blue order. `original_img[50, 33]` is a complete RGB row; `original_img[50, 33, 2]` is its blue channel.

Reshape to `(16384, 3)` because 128 × 128 = 16,384 pixels. Each row is one pixel with three color features. This task clusters **pixels by color**, rather than whole images. Pixel row/column positions are not features here. With the default reshape order, flat row p corresponds to image row `p // 128`, column `p % 128`; reshaping back preserves that order.

| Stage | Shape | Meaning |
| --- | --- | --- |
| original_img | (128, 128, 3) | Image grid of RGB triplets |
| X_img | (16384, 3) | One pixel per row |
| centroids, K=16 | (16, 3) | Palette: one complete RGB color per center |
| idx | (16384,) | One integer color ID, 0 through 15, per pixel |
| centroids[idx, :] | (16384, 3) | Replace each ID with its complete palette color |
| X_recovered | (128, 128, 3) | Those colors restored to the image grid |

### Read centroids[idx, :] one pixel at a time

`idx` holds **centroid/color indices, not distances**. If `idx[0] == 9`, pixel 0 uses palette row 9: `centroids[9, :]`. That row contains all three channel values. The colon means “all feature columns.”

NumPy's [integer advanced indexing](https://numpy.org/doc/stable/user/basics.indexing.html#integer-array-indexing) retrieves rows in the order requested, including repeated IDs:

```python
import numpy as np

palette_demo = np.array([[1., 0., 0.],     # row 0: red
                         [0., 1., 0.],     # row 1: green
                         [0., 0., 1.]])    # row 2: blue
ids_demo = np.array([2, 0, 2, 1])
rgb_demo = palette_demo[ids_demo, :]
np.testing.assert_array_equal(rgb_demo, [[0., 0., 1.],
                                         [1., 0., 0.],
                                         [0., 0., 1.],
                                         [0., 1., 0.]])
assert rgb_demo.shape == (4, 3)
tiny_recovered = rgb_demo.reshape(2, 2, 3)
np.testing.assert_array_equal(tiny_recovered[1, 0], palette_demo[2])
```

The lookup is equivalent to collecting `centroids[idx[p], :]` for every pixel p. It does not compute distances or averages. That work happened earlier. Repeated index 2 retrieves the same blue triplet twice. Advanced indexing creates a new array of RGB values; it does not pack the indices into a compressed file.

### Run the path on invented image data

This continuation uses the earlier assignment and update helpers. It follows the lab's K=16 and ten-cycle structure, with the reusable helper's explicit empty-group policy. The synthetic image is not the course bird image.

```python
# Continue sections 4, 5 and 5a.
# A synthetic RGB gradient; values are already in [0, 1].
level = np.linspace(0., 1., 128)
red, green = np.meshgrid(level, level)
original_img = np.stack([red, green, (red + green) / 2], axis=2)
X_img = original_img.reshape(-1, 3)
assert original_img.shape == (128, 128, 3)
assert X_img.shape == (16384, 3)
np.testing.assert_array_equal(X_img[50 * 128 + 33], original_img[50, 33])

K = 16
np.random.seed(7)  # repeatable teaching run; not evidence of a best solution
image_centroids = kMeans_init_centroids(X_img, K)
for iteration in range(10):
    idx_img = find_closest_centroids(X_img, image_centroids)
    image_centroids = update_centroids(X_img, idx_img, image_centroids)
idx_img = find_closest_centroids(X_img, image_centroids)  # final centers

X_recovered = image_centroids[idx_img, :]
assert image_centroids.shape == (16, 3)
assert idx_img.shape == (16384,)
assert np.issubdtype(idx_img.dtype, np.integer)
assert idx_img.min() >= 0 and idx_img.max() < K
assert X_recovered.shape == (16384, 3)
X_recovered = X_recovered.reshape(original_img.shape)
assert X_recovered.shape == (128, 128, 3)
assert np.unique(X_recovered.reshape(-1, 3), axis=0).shape[0] <= 16
assert np.mean((original_img - X_recovered) ** 2) > 0  # lossy approximation
```

For the lab's variable names, the final lookup is `X_recovered = centroids[idx, :]`, followed by `X_recovered = np.reshape(X_recovered, original_img.shape)`. It puts each replacement color back at the original pixel position.

Inspect the loaded image's shape, dtype and range before adapting this to another file. The lab PNG loads with values in [0, 1]; 8-bit integer RGB values in [0, 255] need floating conversion and division by 255 for that convention. Do not divide values that are already normalized, and handle an alpha channel explicitly rather than treating it as RGB.

### Where the theoretical storage saving comes from

Assume 8 bits per RGB channel, a palette stored at that precision, and exactly 4 packed bits per pixel ID:

| Representation | Calculation | Bits |
| --- | --- | --- |
| Original raw RGB | 128 × 128 × 3 × 8 | 393,216 |
| Palette | 16 × 3 × 8 | 384 |
| Pixel indices | 128 × 128 × 4 | 65,536 |
| Palette + indices | 384 + 65,536 | 65,920 |

Four bits encode 16 possibilities (0–15). The theoretical ratio is 393216 / 65920 ≈ **5.965**, about **6×**. This compares raw RGB with an ideal palette/index representation, excluding headers and other file-format overhead; it does not measure the size reduction from the original PNG.

The NumPy reconstruction is a full RGB array again, with three values per pixel. NumPy integer IDs also normally occupy more than four bits each, and learned centroids are floating-point values. The lab demonstrates the representation and reconstruction; it does not actually quantize a palette to 8-bit channels, pack two 4-bit indices per byte, or write a compressed image file. An actual encoder must do those storage steps (or use a suitable indexed image format); the measured file size depends on that format.

This is **lossy color approximation**: the original shade is replaced by a nearby palette color, so subtle gradients or details can change. Palette lookup reconstructs the approximated picture exactly from that palette and index map, but cannot recover discarded original shades. Quantizing floating-point centroids to an 8-bit stored palette can add rounding error.

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

The original retrieved chat preserved the append/argmin discussion and the request to explain it “like I am 5.” The continuation handoff supplies the specific Exercise 1 and Exercise 2 mistakes documented above. The cached conversation also preserves the accepted three-line update loop and the user's report that it passed. Other self-check questions are teaching prompts.

### K-means syntax lookup

| Need | Syntax | Result / reminder |
| --- | --- | --- |
| Count samples and features | `m, n = X.shape` | Two integers |
| Read second feature for all rows | `X[:, 1]` | m values; not a count |
| Read one whole sample / center | `X[row_id]`; `centroids[cluster_id]` | n values each |
| One distance | `np.linalg.norm(X[row_id] - centroids[cluster_id])` | Scalar |
| Save a candidate distance | `distances.append(distance)` | Inside center loop |
| Choose the closest identity | `idx[row_id] = np.argmin(distances)` | After center loop |
| Visit centroid slots | `for i in range(K):` | K updates, not m |
| Compare every assignment | `idx == i` | m Booleans |
| Collect a whole group | `points = X[idx == i]` | `(r, n)`, r selected rows |
| Average each feature | `np.mean(points, axis=0)` | n values; check r > 0 |
| Save one complete center | `centroids[i] = np.mean(points, axis=0)` | Updates one `(n,)` row |

Read the Exercise 2 sentence aloud: **For each cluster ID, take every row with that ID, average each feature column, and store the resulting vector in that cluster's centroid row.**

Try before reading the answers:

1. X.shape is (100, 5), K=3. What are center and assignment shapes?
2. For distances [4., .5, 2.], what do min and argmin return?
3. What is the center of [[2, 4], [4, 8]]?
4. Which loop repeats K times for each of m samples?
5. Is a cluster ID a probability or known topic name?
6. Does clustering fixed embeddings update the encoder?

**Answers:** (1) (3, 5) and (100,); (2) .5 and 1; (3) [3, 6]; (4) the inner center loop; (5) neither; (6) no.

## Sources and verification

The linked official NumPy, sklearn, Sentence Transformers and Hugging Face references support API and terminology checks. Runnable arrays and telecom applications are teaching examples. The completed-lab status and self-assessment come from the user's handoff and cached chat. The uploaded `Pasted text(7).txt` supplies the full lab, passing outputs, exact centroid values, image shapes and theoretical storage calculation. It was read as reference material; the bird image and course helper files were not supplied for rerunning the course notebook.

**Full-lab follow-up verification, 25 October 2026:** all 14 lesson Python blocks ran in order, including the synthetic 128 × 128 RGB reconstruction with K=16 and ten iterations. Assertions checked pixel order through reshaping, integer palette IDs, full RGB lookup, at most 16 output colors, nonzero lossy reconstruction error, random initialization from existing rows, duplicate-color behavior and the 393,216/65,920 bit calculation. The canonical K-means example ran, all 35 canonical Python blocks passed syntax checks, and new links/anchors, fences and preservation of earlier examples passed. Environment: Python 3.12.14, NumPy 2.5.2, scikit-learn 1.9.1; OMP_NUM_THREADS=1 and LOKY_MAX_CPU_COUNT=1. The full course notebook/grader and an actual packed image encoder were not run. Historical verification records below retain their original scope.

**Exercise follow-up verification, 25 October 2026:** all 11 note Python blocks ran in order with their assertions, including the preserved assignment, update, lifecycle and sklearn examples. Additional checks confirmed the interleaved masks, distinct axis=0/axis=1 outputs, wrong scalar broadcasting, singleton-group index error, first-center ties, the reusable helper's empty-group policy, unchanged old centers, fractional means from integer inputs, and K different from n. The canonical K-means example ran; all 35 canonical blocks passed syntax checks. New relative links/anchors and fences passed, and every earlier Python example was preserved exactly. Environment: Python 3.12.14, NumPy 2.5.2, scikit-learn 1.9.1; OMP_NUM_THREADS=1 and LOKY_MAX_CPU_COUNT=1. Other canonical sections were not runtime-tested again. Course unit-test success is user-confirmed; the course notebook/grader, real embeddings and LLM training were not run here. The earlier verification record below describes the original eight-block note.

**Verified 25 October 2026:** all eight Python blocks ran offline in their stated order, including the three explicitly marked continuation sections. Assertions checked coordinates, shapes, distances, min/argmin, exact assignments, center updates, the two-cycle SSE history, sklearn fit/predict/transform and the two expected synthetic text groups (allowing arbitrary ID order). Additional checks covered first-index ties, an empty group, floating means from integer centers, preservation of input centers, K different from n, loop/vectorized parity and nonincreasing SSE over 30 cycles on generated 3-D data.

The canonical sheet's new section also ran; all 35 sheet Python blocks passed syntax checks, with existing content preserved. New relative links/anchors and fences passed. Environment: Python 3.12.14, NumPy 2.5.2, scikit-learn 1.9.1. Windows core-count/MKL warnings appeared on the first run; the final check used OMP_NUM_THREADS=1 and LOKY_MAX_CPU_COUNT=1. Existing sheet sections were syntax-checked, not all runtime-tested again. Real embedding generation, language-model training and any course grader were not run.


