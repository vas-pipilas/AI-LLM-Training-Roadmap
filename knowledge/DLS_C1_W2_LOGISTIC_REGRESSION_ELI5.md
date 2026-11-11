# Deep Learning Specialization C1 W2 — Logistic Regression with a Neural Network Mindset (ELI5)

**Source:** Andrew Ng / Deep Learning Specialization, Course 1 Week 2 cat-classifier lab. Based on the completed notebook supplied 2026-11-11. This is an explanatory study note, not a replacement for the graded notebook.

## Big picture — teach a robot to recognize cats

A robot receives labeled photos: cat = 1, non-cat = 0. It does not initially understand whiskers; it receives RGB pixel numbers. It guesses, compares the guesses with labels, measures how wrong it was, and repeatedly adjusts its parameters. Finally we test it on photos it did not train on.

**Reusable model lifecycle:** data preparation → model structure → parameter initialization → forward pass → loss → gradients → parameter updates (repeat) → predictions → train/test evaluation.

This is a **single logistic-regression unit** presented with a neural-network mindset, **not a multilayer deep neural network**.

## Dataset and preparation

- Training: 209 images; test: 50 images; each image: 64 × 64 × 3 RGB.
- Raw training array: `(209, 64, 64, 3)`.
- Each image contains `64*64*3 = 12288` features.
- Flatten with `train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T`: result `(12288, 209)`, **each column is one image**. Test: `(12288, 50)`.
- Divide pixel channel values by `255.` to map the usual 0–255 scale to 0–1. Do the same preprocessing at prediction time.
- Labels: `Y_train.shape == (1,209)`; `Y_test.shape == (1,50)`.

## The six functions, ELI5

1. **`sigmoid(z)`**: convert a score to a number between 0 and 1, `1/(1+np.exp(-z))`.
2. **`initialize_with_zeros(dim)`**: create `w` of shape `(dim,1)` and scalar `b=0.0`. The weights are adjustable knobs.
3. **`propagate(w,b,X,Y)`**: forward pass `A = sigmoid(w.T @ X + b)`, calculate average binary cross-entropy cost, then gradients `dw` and `db`. Cost answers *how wrong?*; gradients answer *how to change each knob?*
4. **`optimize(...)`**: repeat `propagate`, then `w -= learning_rate*dw` and `b -= learning_rate*db`. Return learned parameters, last gradients, and sampled costs.
5. **`predict(w,b,X)`**: calculate probabilities and convert to numeric labels: `1` if `A[0,i] > 0.5`, else `0`.
6. **`model(...)`**: orchestrate initialization → optimization on **training** data → retrieve learned parameters → predict both training and test sets → return results.

The model does **not** learn from the test labels. They are used for evaluation.

## Matrix magic — write shapes beside every dot product!

Let `n_x=12288`, `m=209` for training.

| Name | Shape | Meaning |
|---|---|---|
| `X` | `(n_x,m)` | features × images |
| `w` | `(n_x,1)` | one weight per feature |
| `w.T` | `(1,n_x)` | weights oriented for forward dot |
| `w.T @ X` | `(1,m)` | one score per image |
| `A`, `Y`, `A-Y` | `(1,m)` | probabilities, labels, errors |
| `(A-Y).T` | `(m,1)` | errors oriented for gradient dot |
| `dw = X @ (A-Y).T / m` | `(n_x,1)` | gradient matches `w` |
| `db = np.sum(A-Y)/m` | scalar | gradient matches `b` |

**Rule:** `(a,b) @ (b,c) -> (a,c)`. Inner dimensions must agree; outer dimensions are the output. This is how to *see* the transpose while coding, rather than guessing it.

For the cost, `Y` and `A` both have shape `(1,m)`. `Y @ np.log(A).T` and `(1-Y) @ np.log(1-A).T` are each `(1,1)`; add and divide by `-m`. Elementwise multiplication and summation is another formulation, but the lab specifically practices `np.dot`.

## One training cycle in plain language

- Guess using current `w,b`.
- Compare each guess to its true label.
- Compute the mean cost.
- Compute `dw,db`, which indicate how cost changes as weights and bias change.
- Move parameters a small step opposite those gradients.
- Repeat. The learning rate controls step size; iteration count controls how many steps.

## What the uploaded completed notebook showed

The notebook's full training run used **2000 iterations** and **learning rate 0.005**. Recorded training cost fell from **0.693147** at iteration 0 to **0.140872** at iteration 1900. Training accuracy: **99.043%**; test accuracy: **70%**. These are results for that particular notebook run, not guaranteed results for all data.

**Important:** 99% training versus 70% test is a sign of **overfitting**. Learning the training pictures is not the same as generalizing to unseen pictures. More iterations may lower training cost while worsening test accuracy. The test set is not a training set.

## Practical coding checklist

- Before each `np.dot`, write the input and expected output shapes in comments.
- `shape[0]` is not universally “number of examples”: raw image batches use `(m,h,w,3)`, while this lab's flattened `X` uses `(features,m)`.
- `reshape(...,-1).T` preserves image data but changes its layout; `-1` lets NumPy infer the second dimension.
- `np.zeros((dim,1))` uses a **tuple** for the 2D shape.
- `np.dot` is matrix multiplication; `*` is elementwise multiplication for NumPy arrays.
- A probability is not a label: threshold it at `0.5` to obtain 0/1.
- `np.squeeze` can turn a one-element array into a scalar for reporting.
- Train on `X_train,Y_train`; evaluate on `X_test,Y_test`.
- A passing unit test verifies the required behavior on its test cases, not the model's real-world generalization.

## The transferable lesson

The same overall skeleton—preprocess → initialize → forward → loss → gradients → optimize → evaluate—reappears in multilayer neural networks. The difference is that more sophisticated models have many layers and parameters, and automatic differentiation often computes the gradients. Understanding this tiny logistic-regression pipeline makes the larger systems easier to reason about.
