# Learning Log

This is the place for things worth remembering.

It is intentionally informal. If I repeatedly forget a concept, find an explanation especially useful, make an interesting mistake, or finally understand something that had been confusing, it belongs here.

## Milestones

### September 2026 — Python foundation completed

Finished Jose Portilla's *Complete Python Bootcamp: From Zero to Hero in Python*.

Decision after completion: **no second generic Python course**. Continue developing Python by using it in ML and AI work.

Next major learning step: **Andrew Ng / DeepLearning.AI Machine Learning Specialization**.

### September 2026 — Local ML / Jupyter environment ready

Set up the local workstation for the Machine Learning phase instead of using the global Python installation.

Current setup:

- Windows + Visual Studio Code
- Miniconda / Conda environment: `ml-foundations`
- Python 3.12
- Jupyter + ipykernel
- NumPy, pandas, Matplotlib and scikit-learn
- VS Code workspace bound to the `ml-foundations` interpreter
- Jupyter kernel registered as **Python (ML Foundations)**

The sanity-check notebook confirmed that Jupyter is executing the Python interpreter inside `miniconda3/envs/ml-foundations`, not the global Python installation.

The repository also now contains a conservative ML/AI `.gitignore` so local environments, secrets, large datasets, model weights, checkpoints and caches do not accidentally end up in GitHub.

Useful lesson from the setup: the Python interpreter selected by VS Code, the Conda environment active in a terminal, and the Jupyter kernel are related but separate things. The reliable check is always the interpreter that actually executes the code.

### October 2026 — Course 2 Week 1 concepts documented

I put the neural-network ideas from Andrew Ng's *Advanced Learning Algorithms* Week 1 into a [worked ELI5 study guide](knowledge/COURSE_2_WEEK_1_NEURAL_NETWORKS.md). This is a **study-note milestone**, not a claim that I have completed the course or mastered neural-network training.

What this pass helped clarify:

- One 20 × 20 digit image becomes **one** row with 400 pixel features and **one** label for the whole image. The 400 pixels do not imply 400 neurons or 400 labels.
- The architecture `400 → 25 → 15 → 1` describes layer widths. In the NumPy/Keras convention, `W.shape = (inputs, neurons)`; `W[:, j]` selects the incoming weights for neuron `j`. The same parameters serve every training example.
- A forward pass uses existing `W` and `b`. `compile` chooses loss and optimizer, `fit` changes parameters, and `get_weights` only lets me inspect them. A model with initial weights is built, but has not learned digit patterns yet.
- A final sigmoid output is a continuous probability-like score; thresholding creates a 0/1 decision. Normalization changes the scale of `X`, while lambda regularization changes the training objective.

Tiny reminder: for `A_in.shape = (m, 25)`, `W.shape = (25, 15)`, and `b.shape = (15,)`, `g(A_in @ W + b)` has shape `(m, 15)`. If I cannot explain that shape, return to the guide before adding more framework syntax.

Still to learn properly: how backpropagation calculates the gradients for all layers, how Adam uses them internally, and how later course material guides activation and architecture choices. The guide names these pieces without treating them as already understood.

### October 2026 — Backpropagation optional lab documented

Added a [beginner-friendly computation-graph guide](knowledge/COURSE_2_BACKPROPAGATION_COMPUTATION_GRAPH.md) based on the optional lab from Andrew Ng's *Advanced Learning Algorithms*. This records a **study resource**, not course completion or a claim of mastery.

The guide connects the earlier gradient-descent notes to the backward calculation: save values during the forward pass, then multiply local derivatives by the sensitivity arriving from the loss. Its two worked reminders are `J = (2 + 3*w)**2` at `w = 3` (gradient 66), and the single linear neuron with `x = 2, w = -2, b = 8, y = 1` (weight gradient 6, bias gradient 3).

Keep the jobs separate: backpropagation calculates gradients; gradient descent or Adam uses them to update parameters. The guide includes a separate update example and short exercises with answers. Full multilayer backpropagation and Adam's internal mechanics remain topics to study further.

### October 2026 — Week 3 model evaluation and preprocessing documented

Added a [fit / transform / predict deep dive](knowledge/FIT_TRANSFORM_PREDICT_DEEP_DIVE.md) and a [copy-friendly sklearn cheat sheet](knowledge/SKLEARN_PREPROCESSING_CHEAT_SHEET.md), grounded in Course 2 Week 3's optional lab “Model Evaluation and Selection.” This is a **study-resource milestone**, not course completion or a claim of mastery.

The question to keep returning to is: **which object is being fitted, and what does it save?** A scaler saves training-column mean and scale; a polynomial transformer records the feature mapping; a prediction model learns weights and biases. Fitting the same scaler again on CV data overwrites its training statistics. `transform` reuses them.

Tiny reminder: training values `[10, 20, 30]` give mean 20 and scale about 8.165, so a new value 40 becomes about 2.449. Refit on `[40, 50]` and the ruler becomes mean 45, scale 5; the same 40 now becomes -1. The trained model does not automatically adapt to that changed meaning.

The lab pattern to practice is **split 60/20/20 → fit preparation and model on training → select degree or architecture using CV error → evaluate the chosen model on test**. Polynomial predictions must keep the same `poly → scaler → model` recipe and the same `degree - 1` index across all three saved lists. A Pipeline keeps those steps together. For binary models that output logits, evaluation adds sigmoid, a threshold, and the fraction misclassified.

The runnable NumPy/sklearn examples were checked in the local `ml-foundations` environment. They use tiny or generated data rather than reproducing the course's CSV results. Still worth revisiting: why each k-fold training split needs its own fitted preprocessing, how high polynomial degrees affect conditioning and overfitting, and when a production split should follow time or groups instead of random rows.

### 18 October 2026 — Canonical Python ML coding cheat sheet

Added the [canonical Python ML Coding Cheat Sheet](knowledge/PYTHON_ML_CODING_CHEAT_SHEET.md) as the easy-to-find syntax entry point, linked from the README. This is a study-resource milestone, not a claim of mastery or course completion.

It connects Python essentials, NumPy shapes/axes/broadcasting and stable calculations, Matplotlib plots, TensorFlow/Keras outputs and losses, GradientTape, sklearn preparation/Pipeline, train/CV/test evaluation, and bias/variance/regularization. Beginner refreshers are separated from intermediate practice extensions. The existing detailed notes remain in place and are linked.

The reminders I most want to keep: fit learns the ruler, transform reuses it; compare prediction/target shapes before computing errors; logits need the matching loss and threshold; learning-rate alpha is different from Ridge alpha; lambda up strengthens regularization and can increase bias, while lambda down can increase variance. CV decides the useful setting.

**Ongoing rule:** extend this same canonical cheat sheet as I learn new coding topics. Add runnable syntax, expected shapes/output, pitfalls, and navigation links; record meaningful additions here. GitHub is the durable source of truth, not a promise that chat memory holds the sheet verbatim.

Verification details are recorded at the end of the sheet. NumPy/Matplotlib/sklearn examples are checked locally where available; TensorFlow is unavailable here, so its examples receive syntax checks and API review without a runtime-test claim.

### 19 October 2026 — Decision Trees reference added

Added one [dedicated Decision Trees study note](knowledge/DECISION_TREES.md) and a concise syntax section in the existing [canonical Python ML coding cheat sheet](knowledge/PYTHON_ML_CODING_CHEAT_SHEET.md#12a-decision-trees--concise-syntax). No second mega cheat sheet was created. This records a study resource, not course completion.

The note explains entropy, weighted information gain, row-index splitting, the lab-style helper functions, recursion/stopping, sklearn fit/predict/probabilities, CV selection, overfitting and possible telecom applications.

**Updated from the supplied lab handoff:** the same note now reproduces the original ten-row Andrew Ng ungraded lab data, four functions, root gains and depth-2 leaves, with a code walkthrough. The four-row example remains explicitly illustrative. Original lab assumptions about empty children and subset-local indices are distinguished from the safer reusable helpers.

Verification update: all five Decision Trees Python blocks and the canonical sheet's section 12a sklearn example ran offline with Python 3.12.14, NumPy 2.5.2, scikit-learn 1.9.1 and Matplotlib 3.11.2. Root calculations, split indices, depth-2 leaves, empty-branch behavior and prediction/probability assertions passed; the plot exported with Agg. The full course notebook and GUI display were not tested. Existing verification records for other topics are preserved.

### 21 October 2026 — Graded decision-tree indexing clarified

Extended the existing [Decision Trees note](knowledge/DECISION_TREES.md#4a-graded-exercise-follow-the-original-row-ids) and linked it from the [canonical coding cheat sheet](knowledge/PYTHON_ML_CODING_CHEAT_SHEET.md#graded-exercise-reminder-row-ids-survive-every-split). The schoolchildren/register analogy and a six-row mushroom example explain why X stays full while each node carries a subset of original row IDs. The example is invented teaching data, separate from the preserved ungraded animal lab.

The key reminder: loop over the IDs in node_indices, inspect one selected column with X[i, feature], and append i. Splitting routes examples; information gain uses their labels to evaluate the question. p_1 counts positive labels at the current node (1 edible, 0 poisonous here); pure endpoints need checks before logarithms. Recorded the first-attempt mistakes with X[i, node_indices], appending values/rows, looping over counts instead of original IDs, and p1 versus p_1. This documents exercise concepts, not course completion or a claim of running the course grader.

Verification: all three new snippets ran with exact expected outputs and embedded assertions; the existing section 4 helper example passed again. Syntax checks covered all 13 note blocks and 34 cheat-sheet blocks. New relative links/anchors and preservation checks passed. Python 3.12.14, NumPy 2.3.5; the full course notebook and grader were not run.

### 21 October 2026 — Decision-tree brain-burners and best-gain tracking

Extended the existing [graded-lab troubleshooting reference](knowledge/DECISION_TREES.md#what-confused-me-in-the-graded-lab--brain-burner-troubleshooting), with a brief reminder in the [same canonical cheat sheet](knowledge/PYTHON_ML_CODING_CHEAT_SHEET.md#graded-exercise-reminder-row-ids-survive-every-split). Earlier lab reproductions and accepted explanations remain in place.

The reminders to revisit: y=1 is a class label, not a correct prediction; p_1 measures class mixture while child weights measure routing proportions. Parent entropy belongs to the current node. The uneven-child example shows why weighting matters. For best-split selection, compare the returned candidate gain with the largest gain seen so far and update both the score and its feature ID; ig_1 and ig_i are different names. The student's final ig_max update fixed that missing step. Trees choose discrete questions recursively; NN connection weights and backpropagation have different jobs.

Added a corrected selection loop, worked gains, common error/correct patterns, self-checks and a concise lookup table. Verification covered exact walkthrough outputs, gain tracking, ties, zero gain, boundary cases, syntax and relative links/anchors; see the [scope and results](knowledge/DECISION_TREES.md#sources-and-verification). This is a study-note update, not course completion or a course-grader result.


### 25 October 2026 — Unsupervised ML and K-means reference

Added a [detailed study note](knowledge/UNSUPERVISED_LEARNING_AND_KMEANS.md) and a concise section in the [same canonical cheat sheet](knowledge/PYTHON_ML_CODING_CHEAT_SHEET.md#12c-unsupervised-learning-and-k-means). This records a study resource, not course completion or mastery.

The reminders to revisit: a sample is one row, features are its coordinates, and a centroid is an average location with the same feature count. The outer loop chooses a sample; the inner loop compares every center. Reset distances for each sample; append saves one scalar. With distances [5, 1.5, 4], min is 1.5 but argmin is 1, the zero-based center ID. A center coordinate of 1.5 has a different role. Assignment and column-wise mean updates repeat until a stopping rule or iteration cap; convergence does not guarantee the best or most meaningful groups.

The LLM connection is text → embeddings → clustering → inspect groups. Clustering fixed vectors does not train the encoder. Self-supervised pretraining constructs token targets from text and updates neural weights. The guide distinguishes feature scaling from row normalization and ordinary K-means from spherical K-means.

Verification: eight note blocks and the new canonical example ran offline; numerical, shape, loop/vectorized parity, empty-group, tie and objective checks passed. All 35 canonical blocks passed syntax checks; new links/anchors and preservation checks passed. See the note's [verification record](knowledge/UNSUPERVISED_LEARNING_AND_KMEANS.md#sources-and-verification) for versions and limits. Real embedding generation, LLM training and the course grader were not run.

### 25 October 2026 — K-means Exercises 1 and 2 completed

Recorded completion of Exercise 1 `find_closest_centroids` and the user's confirmation that Exercise 2 `compute_centroids` passed all unit tests. The handoff's approximate three-center output is saved in the [existing K-means note](knowledge/UNSUPERVISED_LEARNING_AND_KMEANS.md#completed-k-means-exercises). This is exercise progress, not specialization completion; this documentation task did not rerun the course grader.

The mental gymnastics worth remembering: `X.shape[0]` counts rows while `X[:, 1]` reads a column. `X[i]` and `centroids[j]` are whole feature vectors. NumPy supplies functions, so use `X` rather than `np.X`, and pass their difference to `np.linalg.norm`. Save argmin's centroid ID, not min's distance. Indentation schedules the work: reset once per sample, append once per center, choose after all comparisons.

For Exercise 2, i now selects a centroid slot, so loop over K. `idx[i] == i` checks only one assignment; `idx == i` checks all m assignments. `X[idx == i]` keeps every assigned row. The extra `[i]` in `np.mean(points[i], axis=0)` throws away the rest of the group; average `points` directly, column by column, to keep n feature means. The note uses interleaved groups, full masks, a step-by-step trace, and scalar-broadcast and index-error examples to make this visible. Empty clusters still require an explicit policy.

The [same canonical cheat sheet](knowledge/PYTHON_ML_CODING_CHEAT_SHEET.md#12c-unsupervised-learning-and-k-means) contains concise reminders and links to the detailed walkthrough. The existing assign/update lifecycle, ML engineering uses, embedding-clustering examples and distinction from self-supervised LLM pretraining remain in place.

Verification for this follow-up: all 11 note blocks and the canonical K-means example ran offline with numerical/shape assertions; all 35 canonical blocks passed syntax checks. Specific mask, axis, scalar-broadcast, singleton, tie, empty-group and K-versus-n checks passed, as did new links/anchors and preservation of all earlier code examples. See the note's verification record for environment and scope.

## Concepts to revisit

Add entries here as they appear during training. A useful entry answers three questions:

1. What was confusing?
2. What finally made it click?
3. What tiny example can remind me six months later?

Do not turn this into copied course notes. Keep the things that were personally useful.

## Useful patterns and lessons

This section can collect small Python/ML/LLM ideas that are worth coming back to: debugging patterns, Python idioms, model-evaluation lessons, RAG mistakes, useful commands, diagrams, or short explanations.

## Questions I still have

Keep unresolved questions here instead of pretending they are understood. Once a question is resolved, move the useful conclusion into the appropriate section above.

## Interesting side trails

AI changes quickly and interesting topics will appear before their scheduled phase. Record them here rather than derailing the main roadmap. We can decide later whether they deserve a small experiment or should wait.



