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
