# Course 2, Week 1 — Neural networks, explained simply

*A study companion for Andrew Ng / DeepLearning.AI's Machine Learning Specialization, Course 2: Advanced Learning Algorithms.* This is an original explanation and a small, separate example. It is **not** a solution to a graded lab.

## The whole story in one minute

Imagine one handwritten digit as a small 20 × 20 picture. Flatten its 400 pixel values into a list. A neural network passes that list through groups of tiny calculators called **neurons**. Each neuron asks, “How strongly do my inputs suggest something useful?” The first group makes 25 new numbers, the next makes 15, and the last makes one number between 0 and 1. For a binary 0-versus-1 task, that last number can be read as the model's estimated probability of the positive class.

```text
one 20 × 20 image → 400 pixel values → 25 neurons → 15 neurons → 1 sigmoid output
                                      hidden layer 1   hidden layer 2   probability
```

Two separate jobs happen:

- **Forward propagation (prediction):** use today's weights and biases to turn input into output.
- **Training:** compare output with the correct label, calculate a loss, find how the parameters should change, update them, and repeat.

Week 1's main goal is to understand the forward journey and how to express it in NumPy and Keras. The details of **backpropagation** and **Adam's internal update rules** are upcoming material. You can use `fit()` without pretending you already know those internals.

## 1. One neuron is a tiny adjustable calculator

Suppose a neuron receives `x = [2, 1]`. It has one **weight** for each input, perhaps `w = [1, -1]`, and one **bias**, perhaps `b = 0`.

```text
input x1 = 2 -- weight w1 =  1 -- contributes  2
input x2 = 1 -- weight w2 = -1 -- contributes -1
                                      sum = 1
                                      + bias 0
                                      z = 1
                                      sigmoid(z) ≈ 0.731
```

The formula is `z = w·x + b = w1*x1 + w2*x2 + b`. The dot means multiply matching entries and add them. **Weights** control the strength and direction of each input's effect. A **bias** shifts the total so the neuron is not forced to output the same value whenever the weighted inputs sum to zero. The activation `a = g(z)` is the neuron's output. With **sigmoid**, `g(z) = 1/(1 + exp(-z))`, so `g(0) = 0.5`, large positive `z` approaches 1, and large negative `z` approaches 0.

The numbers above are invented to make arithmetic visible. Trained weights have to be learned from examples; a weight is not automatically a named “stroke detector.” A hidden neuron's output is another input for the next layer. **Every neuron in a Dense layer receives all outputs from the previous layer**, but it owns its *own* weight vector and bias.

## 2. From pixels to labels

In the course's digit example, one 20 × 20 image becomes 400 features by flattening its rows. The values describe pixels; the label describes the **whole picture**.

```text
image 0: 20 × 20 pixels → X[0] has 400 values → y[0] is 0 or 1
image 1: 20 × 20 pixels → X[1] has 400 values → y[1] is 0 or 1
...
```

For `m` images, `X.shape == (m, 400)`. Labels might have shape `(m,)` or `(m, 1)`, depending on the notebook; there is one binary target per image. In a 0-versus-1 setup, `y = 1` means the image belongs to the positive class (digit 1), and `y = 0` means digit 0. There are **not** 400 labels per picture. Flattening changes the arrangement of the same pixel values; it does not create 400 new examples.

## 3. Follow the shapes through 400 → 25 → 15 → 1

Let `a0 = x` for one image. In the course/NumPy convention used here, a Dense layer with `n_in` incoming values and `n_out` neurons has:

```text
W.shape = (n_in, n_out)     b.shape = (n_out,)
z = a_in @ W + b            a_out = g(z)
parameters = n_in*n_out + n_out
```

Each **column** of `W` belongs to one neuron. Each **row** corresponds to one incoming feature. Keras Dense kernels use the same `(input_count, units)` orientation.

| Stage | One example in → out | Weights `W` | Bias `b` | Trainable parameters |
| --- | --- | --- | --- | ---: |
| Input | `(400,)` | — | — | 0 |
| Dense 1, 25 units | `(400,)` → `(25,)` | `(400, 25)` | `(25,)` | `400×25+25 = 10,025` |
| Dense 2, 15 units | `(25,)` → `(15,)` | `(25, 15)` | `(15,)` | `25×15+15 = 390` |
| Dense 3, 1 unit | `(15,)` → `(1,)` | `(15, 1)` | `(1,)` | `15×1+1 = 16` |
| **Total** | `(400,)` → `(1,)` | | | **10,431** |

For a batch of `m` images, put `m` in front of each activation shape: `(m,400) → (m,25) → (m,15) → (m,1)`. The **parameter count does not grow with `m`**. The same weights are reused for every image. The choices 25 and 15 describe model width; they are design choices, not numbers forced by the 400 pixels.

For one image, the arrows expand to:

```text
a0 (400,) @ W1 (400,25) + b1 (25,) → z1 (25,) → g → a1 (25,)
a1 (25,) @ W2 (25,15)  + b2 (15,) → z2 (15,) → g → a2 (15,)
a2 (15,) @ W3 (15,1)   + b3 (1,)  → z3 (1,)  → sigmoid → a3 (1,)
```

In this Week 1 example, sigmoid can be used in all three Dense layers. Later material introduces other activation choices and reasons to choose them.

## 4. Keras says the same thing with fewer lines

This independent sketch defines the shape of the model; it does not contain any graded exercise answer or trained digit weights.

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.Input(shape=(400,)),  # features per image; batch dimension is separate
    tf.keras.layers.Dense(25, activation="sigmoid"),
    tf.keras.layers.Dense(15, activation="sigmoid"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.summary()
```

`Dense(25)` means **25 neurons**, not 25 input pixels. `Sequential` passes each layer's output to the next in order. `Input(shape=(400,))` says what **one** example looks like; Keras adds the batch dimension when you pass many examples. `model.summary()` is a useful check: compare its output shapes and parameter counts with the table above.

### Where did the first weights come from?

A new Dense layer needs starting values before it can predict or train. Once it knows its input size and has been built, its kernel and bias exist. With the usual `tf.keras.layers.Dense` defaults, the kernel starts from **Glorot uniform** random initialization and the bias starts at **zero**. Initial values are only a starting point; they do not encode knowledge of digits. Distinct initial weights help hidden neurons learn different things instead of remaining identical.

```python
first_layer = model.layers[0]
W1, b1 = first_layer.get_weights()
print(W1.shape, b1.shape)  # (400, 25) (25,)
```

`get_weights()` gives the layer's **current** NumPy arrays. Before a layer is built, there may be no weights to get; explicitly declaring the input shape as above builds this Sequential model. Calling `get_weights()` immediately after building shows initialized values. Calling it after `fit()` shows the updated values. It does **not** itself train anything. If you train the model later, call it again to inspect the new values.

## 5. Compile, fit, and predict are three different verbs

For a final sigmoid and binary labels, a typical setup is:

```python
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)
model.fit(X_train, y_train, epochs=5)
probabilities = model.predict(X_new)
decisions = (probabilities >= 0.5).astype(int)
```

- `compile(...)` **configures** training: binary cross-entropy measures how poorly predicted probabilities match 0/1 labels; Adam is the optimizer that will use gradients to update parameters. Compiling alone does not learn.
- `fit(...)` **trains**. An **epoch** is one pass through the training data; each epoch can include multiple batches and parameter updates.
- `predict(...)` runs a **forward pass** with the current parameters. It does not learn from `X_new`.

A useful picture of what `fit` coordinates:

```text
current W,b → forward pass → probability → compare with y → loss
    ↑                                               ↓
    └──────── optimizer updates ← gradients ← backpropagation
```

At this stage, understand *what each box does*. The chain rule calculations in backpropagation and Adam's moving-average mechanics are **upcoming material**, not prerequisites for reading Week 1's forward pass.

### Probability versus a class decision

If the final sigmoid returns `0.82`, interpret it as an estimated probability-like score for class 1 under this binary model. A 0.5 threshold converts it to class `1`; `0.18` becomes class `0`. The threshold is a **decision rule after prediction**. It does not change the neural network's weights, and it is not automatically the best threshold for every real application. Compare probabilities to labels through the loss during training; use a threshold when you actually need a 0/1 decision.

## 6. Manual NumPy: one Dense layer, then a whole network

Before vectorizing, it helps to spell out one layer. This generic function takes already chosen weights; it is **forward propagation only**.

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dense_one_example(a_in, W, b):
    units = W.shape[1]
    a_out = np.zeros(units)
    for j in range(units):
        w_j = W[:, j]              # all incoming weights for neuron j
        z_j = np.dot(a_in, w_j) + b[j]
        a_out[j] = sigmoid(z_j)
    return a_out
```

With `W.shape == (3, 2)`, `W[:, 0]` has **three** numbers feeding neuron 0; `W[:, 1]` has three numbers feeding neuron 1. Here `j` means a **neuron index**, while in `X[i, j]` it usually means a **feature index**. Read the array's shape and context rather than memorizing one meaning for `j`.

A tiny worked check, separate from the digit task:

```python
a_in = np.array([2., 1.])
W = np.array([[1., 0.],
              [-1., 2.]])
b = np.array([0., -1.])
# neuron 0: z = 2*1 + 1*(-1) + 0 = 1   → sigmoid ≈ 0.731
# neuron 1: z = 2*0 + 1*2 + (-1) = 1    → sigmoid ≈ 0.731
a_out = dense_one_example(a_in, W, b)  # shape (2,)
```

For an entire network, feed the output of one call into the next:

```python
a1 = dense_one_example(x, W1, b1)
a2 = dense_one_example(a1, W2, b2)
a3 = dense_one_example(a2, W3, b3)
```

The function does not compare against `y`, calculate a loss, or change `W` and `b`. A notebook that supplies pretrained `W1`, `W2`, `W3` can demonstrate inference without training a model in that notebook.

## 7. Vectorization: same arithmetic, many values at once

The loop above computes each neuron separately. Matrix multiplication computes all neurons of a layer together:

```python
def dense_vectorized(A_in, W, b):
    Z = A_in @ W + b
    return sigmoid(Z)
```

For **one** example, `A_in.shape == (n_in,)`, `W.shape == (n_in, n_out)`, and the result has shape `(n_out,)`. For a **batch**, `A_in.shape == (m, n_in)`; then `Z` and `g(Z)` have shape `(m, n_out)`. NumPy broadcasts `b.shape == (n_out,)` across the `m` rows. The conceptual recipe is `Z = A_in @ W + b; A_out = g(Z)`, where `g` acts element by element.

A small shape sketch:

```text
A_in (2 examples, 3 features) @ W (3 inputs, 2 neurons)
                          + b (2 biases, reused for both examples)
                        → Z (2 examples, 2 neuron scores)
                        → g(Z) (2 examples, 2 activations)
```

Vectorization is a faster expression of the same forward calculation, not a different model. Compare one loop output to one vectorized output when debugging. If the result shape is surprising, inspect `A_in.shape`, `W.shape`, and `b.shape` first.

## 8. Two concepts with similar names but different jobs

| Concept | Changes | Why use it? | Tiny example |
| --- | --- | --- | --- |
| **Normalization** | The numerical scale of input features `X` | Help features live on comparable scales and often make training easier | A 0–255 pixel could be scaled to 0–1; apply the same training-time rule to new data. |
| **Lambda regularization** | The training objective by penalizing large weights | Discourage an overly complex fit and help with overfitting | `loss + λ × weight_penalty`; larger `λ` means stronger penalty. |

Normalization does **not** mean “set lambda,” and regularization does **not** rescale the pixels. If computing means/standard deviations for normalization, estimate them from the **training set**, then reuse them for validation, test, and new images. See the existing [NumPy note](NUMPY_FOR_ML.md) and [gradient descent note](GRADIENT_DESCENT.md) for the mechanics. Regularization is introduced here only to prevent a common mix-up; its deeper treatment comes later.

## 9. Common mistakes to catch early

| Mistake | Better check |
| --- | --- |
| “400 pixels means 400 labels or 400 neurons.” | 400 features describe **one** image; one label belongs to that image. Layer widths are design choices. |
| “`W[:, j]` is feature `j`.” | In `W.shape=(inputs, units)`, column `j` is all weights for **neuron** `j`. |
| “`*` and `@` are interchangeable.” | `*` multiplies element by element; `@` performs the layer's matrix multiplication. |
| “A bias is one extra weight per image.” | Each neuron has one bias; the same bias is reused across examples. |
| “`compile` or `get_weights` trained the model.” | `compile` configures; `get_weights` inspects; `fit` trains. |
| “The initialized model recognizes digits.” | Initial weights are a starting guess; useful behavior comes from training. |
| “0.82 is already the integer label 1.” | It is a continuous output. Apply a decision threshold if a class is needed. |
| “Normalization is regularization.” | One scales inputs; the other penalizes the training objective. |
| “I must know every Adam derivative now.” | For Week 1, track the forward shapes and the purpose of loss/optimizer. |

## 10. Five-minute self-check

Try answering without opening the cheat sheets; then use them to verify your reasoning.

1. A 20 × 20 digit image is flattened. How many features does **one** example have, and how many labels?
2. With 400 inputs and 25 Dense units, what are the shapes of `W`, `b`, and the output for one example? How many parameters?
3. Why does layer 2 have `W.shape == (25, 15)` even though the original image had 400 pixels?
4. In `W[:, 7]`, what does the `7` select?
5. If `A_in.shape == (32, 25)`, `W.shape == (25, 15)`, and `b.shape == (15,)`, what shape is `g(A_in @ W + b)`?
6. Which calls configure training, change parameters, inspect parameters, and only make predictions: `compile`, `fit`, `get_weights`, `predict`?
7. If a sigmoid output is `0.49`, what class would a 0.5 threshold choose? Has the network's probability changed?
8. What is the difference between scaling pixel inputs and adding a `λ` penalty?
9. Tell the journey from pixels to probability using the words **weights**, **bias**, **activation**, and **forward propagation**.
10. Which two training internals are intentionally left for later?

## Where to go next

Revisit the [NumPy cheat sheet](NUMPY_FOR_ML.md) for shapes, `@`, indexing, and broadcasting, or the [Keras cheat sheet](TENSORFLOW_KERAS_FOR_ML.md) for API syntax. The [gradient descent note](GRADIENT_DESCENT.md) explains the earlier single-model training idea that will later connect to backpropagation. Then try writing a **new tiny ungraded** two-input/two-neuron example from scratch and predict its output shape before running it.

Course outline: [DeepLearning.AI Machine Learning Specialization](https://www.deeplearning.ai/specializations/machine-learning/). Framework reference: [TensorFlow Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) and [TensorFlow Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model).
