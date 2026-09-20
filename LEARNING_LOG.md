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
