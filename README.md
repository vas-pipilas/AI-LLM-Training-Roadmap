# AI / LLM Engineering Training Roadmap

This repository is my long-term learning record for moving from Voice / Telecom engineering into AI and LLM engineering.

The goal is not to collect courses or badges. I want to understand the technology well enough to design, build, debug, deploy, and operate real AI systems. My existing telecom background is part of the plan rather than something to leave behind: over time I want to combine AI engineering with real Voice, IMS, SIP, SS7, Diameter, observability, and troubleshooting problems.

## Where I am now

**Current phase: Machine Learning foundations**

### Completed

- ✅ **Python foundation — Complete**
  - Jose Portilla — *Complete Python Bootcamp: From Zero to Hero in Python* (Udemy)
  - Completed September 2026.
  - This is the end of the generic Python-course phase. From here Python should improve mainly by writing code and using it in ML/AI projects.

### Next

- 🔄 **Andrew Ng / DeepLearning.AI — Machine Learning Specialization**
- Keep writing Python alongside the course instead of only watching lectures.
- Build small exercises/projects that force me to start from an empty file, reason about the problem, implement it, test it, and explain what I built.

See [ROADMAP.md](ROADMAP.md) for the full path and [LEARNING_LOG.md](LEARNING_LOG.md) for progress and things worth revisiting.

## Working principle

The loop for this project is:

**Learn → implement → get stuck → understand why → document the useful part → build → certify when the knowledge is real.**

Course completion is useful, but it is not the finish line. Projects and the ability to explain/debug my own work matter more.

## Python coding cheat sheet — start here

[**Canonical Python ML Coding Cheat Sheet**](knowledge/PYTHON_ML_CODING_CHEAT_SHEET.md) — an indexed, beginner-to-intermediate syntax reference for Python essentials, NumPy, Matplotlib, TensorFlow/Keras, sklearn preprocessing, evaluation, and regularization. Includes runnable examples, expected shapes, common mistakes, and copy-friendly recipes.

Extend this same canonical file as I learn new coding topics. Keep the focused notes below as deeper references. GitHub is the durable source of truth; chat memory cannot store the full sheet verbatim.

## Core knowledge notes

When a subject becomes foundational enough that I will need it again later, it gets its own detailed reference under `knowledge/`. These are not copied course notes. They are written around the explanations, mistakes, questions, and mental models that actually helped the subject click for me.

They should work both as a deeper refresher when I cannot watch a lecture and as a quick reference months later.

- [Gradient Descent](knowledge/GRADIENT_DESCENT.md) — cost, derivatives/gradients, learning rate, parameter updates, convergence, scientific notation, and the connection to later deep learning.
- [Python for ML — Quick Reference](knowledge/PYTHON_FOR_ML_QUICK_REFERENCE.md) — the Python patterns I keep seeing in ML labs: loops, functions, shapes, accumulation, normalization, feature engineering, and a code-reading checklist.
- [NumPy for ML — Detailed Cheat Sheet](knowledge/NUMPY_FOR_ML.md) — arrays, shapes, indexing, dot products, vectorization, broadcasting, boolean masks, reshaping, neural-network weight matrices, and prediction thresholding.
- [TensorFlow / Keras for ML — Detailed Cheat Sheet](knowledge/TENSORFLOW_KERAS_FOR_ML.md) — Dense layers, Sequential models, input/output shapes, activations, parameters, normalization, training, prediction, and the bridge from NumPy to neural networks.
- [Course 2 Week 1 — Neural Networks, ELI5 Study Guide](knowledge/COURSE_2_WEEK_1_NEURAL_NETWORKS.md) — a worked path from 20 × 20 digit pixels through the 400 → 25 → 15 → 1 network, with shapes, Keras and NumPy forward passes, common mistakes, and self-checks.

- [Course 2 — Backpropagation using a Computation Graph, ELI5 Guide](knowledge/COURSE_2_BACKPROPAGATION_COMPUTATION_GRAPH.md) — tiny-nudge derivatives, forward and backward passes, the chain rule, two worked lab examples, and how gradients feed parameter updates.
- [Fit, Transform, and Predict — ELI5 Deep Dive](knowledge/FIT_TRANSFORM_PREDICT_DEEP_DIVE.md) — what each fit learns, scaler overwrite behavior, train-only preparation, polynomial features, shapes, model selection, and the prediction path from raw inputs.
- [sklearn Preprocessing — Copy-Friendly Cheat Sheet](knowledge/SKLEARN_PREPROCESSING_CHEAT_SHEET.md) — runnable manual and Pipeline examples, CV degree selection, matching preprocessing/model objects, and binary classification evaluation.

- [Decision Trees — Detailed Study Note](knowledge/DECISION_TREES.md) — illustrative entropy/information-gain calculations, lab-style Python helpers, sklearn usage, overfitting controls and telecom applications. Original notebook calculations await the source excerpt.

This section should grow selectively. Not every lesson deserves a document; core concepts that later knowledge depends on do.

## Repository map

- [ROADMAP.md](ROADMAP.md) — the evolving long-term plan and current phase.
- [LEARNING_LOG.md](LEARNING_LOG.md) — milestones, concepts I struggled with, useful explanations, and subjects worth revisiting.
- [knowledge/](knowledge/) — detailed human-readable reference notes for core concepts I will repeatedly need.
- [PROJECTS.md](PROJECTS.md) — training projects and portfolio ideas.
- [CERTIFICATIONS.md](CERTIFICATIONS.md) — certification candidates and, importantly, when I should actually attempt them.

This repository is deliberately written as a human learning journal, not as a corporate project plan. It should change when my knowledge, interests, or the AI ecosystem changes.
