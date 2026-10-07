# TensorFlow / Keras for ML — Detailed Cheat Sheet

This is my practical TensorFlow/Keras reference while learning neural networks. It connects the framework syntax to the NumPy and gradient-descent ideas I already understand.

The key mental model:

> **TensorFlow/Keras is packaging and automating the same weighted sums, activations, loss calculations, gradients, and parameter updates that I learned manually.**

---

## 1. TensorFlow versus Keras

For my current mental model:

- **TensorFlow** is the larger numerical/deep-learning framework.
- **Keras** is the convenient high-level interface used to define and train neural networks.
- `tf.keras` is the Keras interface available through TensorFlow.

Typical import:

```python
import tensorflow as tf
```

---

## 2. One neuron is still familiar mathematics

A neuron receives inputs and calculates:

```text
z = w·x + b
a = g(z)
```

where:

- `x` / `a_in` = inputs to the neuron
- `w` = one weight for every incoming input
- `b` = one bias for the neuron
- `z` = linear combination / pre-activation / raw score
- `g` = activation function
- `a` = activation/output

Nothing magical happened when moving from logistic regression to a neuron.

---

## 3. Dense layer

A dense layer means every neuron in the layer receives every output from the previous layer.

```python
tf.keras.layers.Dense(
    units=3,
    activation='sigmoid'
)
```

means:

```text
Create a layer with 3 neurons.
Each neuron receives all inputs.
Each neuron uses sigmoid after z = w·x + b.
```

`units` means number of neurons in the layer.

It does **not** mean number of input features.

---

## 4. Input shape

```python
tf.keras.Input(shape=(400,))
```

means:

> Each individual training example contains 400 input features.

For the handwritten-digit lab:

```text
20 × 20 pixel image
       ↓ flatten
400 pixel features
```

So the model receives one 400-value vector per image.

The batch/example dimension is not written inside `shape=(400,)`.

---

## 5. Sequential model

```python
model = tf.keras.Sequential([
    tf.keras.Input(shape=(400,)),
    tf.keras.layers.Dense(...),
    tf.keras.layers.Dense(...),
])
```

`Sequential` means:

> Send the output of one layer directly into the next layer in order.

Mental picture:

```text
X
↓
Layer 1
↓
a1
↓
Layer 2
↓
a2
↓
...
```

The activations from one layer become the inputs/features for the next layer.

---

## 6. Layer shapes

Suppose:

```text
400 inputs → 25 neurons → 15 neurons → 1 neuron
```

Conceptually:

```text
Layer 1:
400 inputs per neuron
25 neurons
→ 400×25 weights + 25 biases

Layer 2:
25 inputs per neuron
15 neurons
→ 25×15 weights + 15 biases

Output:
15 inputs
1 neuron
→ 15 weights + 1 bias
```

The number of neurons is a model-design choice. It does not have to equal the number of input features.

---

## 7. Activation functions

Sigmoid:

```python
activation='sigmoid'
```

maps a neuron's `z` to a value between 0 and 1.

Mental pipeline:

```text
inputs → weighted sum + bias → z → sigmoid → activation
```

Linear activation:

```python
activation='linear'
```

essentially leaves the linear output unchanged.

Later I will meet other activations; do not try to memorize all of them now.

---

## 8. model.summary()

```python
model.summary()
```

Use this to inspect the architecture.

Important things to read:

```text
Layer
Output Shape
Param #
```

For a Dense layer:

```text
parameters = input_count × neuron_count + neuron_count
             └──── weights ────┘   └─ biases ─┘
```

Example: 400 inputs and 25 neurons:

```text
400×25 + 25 = 10,025 parameters
```

---

## 9. Compile: decide how training should work

A binary-classification example:

```python
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
)
```

Human translation:

```text
loss      → how wrong are the predictions?
optimizer → how should trainable parameters be updated?
```

Binary cross-entropy is the same family of loss used with binary logistic classification.

Adam is an optimizer. It performs parameter updates using gradients with a more sophisticated strategy than the simple gradient-descent update I coded manually.

---

## 10. fit: train the model

```python
model.fit(X, y, epochs=10)
```

Conceptually TensorFlow repeatedly does:

```text
forward propagation
        ↓
predictions
        ↓
loss
        ↓
backpropagation / gradients
        ↓
optimizer updates W and b
        ↓
repeat
```

The neurons themselves do not independently run gradient descent.

The network contains the trainable parameters. Backpropagation computes gradients; the optimizer uses those gradients to update the parameters.

---

## 11. Epoch and batch

**Epoch** = one full pass through the training dataset.

**Batch** = a smaller group of examples processed before an update.

If there are 1000 examples and batch size 32, training processes groups of roughly 32 examples rather than treating all 1000 as one giant operation.

Do not confuse:

```text
epoch → full dataset pass
batch → subset within that pass
```

---

## 12. predict: forward propagation on new data

```python
probabilities = model.predict(X_new)
```

For a binary classifier whose final activation is sigmoid, the final outputs are continuous values between 0 and 1.

Example:

```text
0.96
0.03
0.51
```

These are not yet necessarily the final 0/1 decisions.

Thresholding:

```python
yhat = (probabilities >= 0.5).astype(int)
```

gives:

```text
0.96 → 1
0.03 → 0
0.51 → 1
```

Keep separate:

> **model output/probability → threshold → class decision**

---

## 13. y versus y-hat

```text
y     = reality / true label
y_hat = model's predicted class/value
```

For the current digit problem:

```text
y = 0 → entire image is digit 0
y = 1 → entire image is digit 1
```

The individual pixel intensities belong in `X`; they are not separate `y` labels.

---

## 14. Normalization layer

```python
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)
Xn = norm_l(X)
```

Mental translation:

```text
adapt(X)
→ learn normalization statistics from training X

norm_l(X)
→ apply that learned normalization
```

Normalization changes input-feature scale.

It is **not regularization**.

```text
normalization → changes/scales X
regularization → penalizes model complexity/large weights; lambda controls strength
```

---

## 15. Inspecting learned weights

For a layer:

```python
W, b = layer.get_weights()
```

This exposes the parameters TensorFlow learned.

A layer may need to be built / know its input shape before weights exist.

For educational experiments:

```python
layer.set_weights([W, b])
```

can manually replace the parameters. That is useful for demonstrations, but normal training learns them through `fit()`.

---

## 16. TensorFlow tensors

For now, the useful mental model is:

> **A TensorFlow tensor is array-like numerical data used by TensorFlow.**

NumPy:

```python
np.array(...)
```

TensorFlow:

```python
tf.constant(...)
```

There are deeper differences involving automatic differentiation, devices, graphs, and GPUs, but they are not required to understand the current neural-network labs.

---

## 17. NumPy implementation versus Keras

Manual NumPy neuron:

```python
z = np.dot(w, a_in) + b
a = sigmoid(z)
```

Keras:

```python
Dense(units=..., activation='sigmoid')
```

The Keras layer is packaging the same core computation for many neurons.

Manual NumPy network:

```text
my_dense(...)
↓
my_dense(...)
↓
prediction
```

Keras network:

```text
Sequential([
    Dense(...),
    Dense(...)
])
```

The abstraction changes. The underlying idea does not.

---

## 18. Forward propagation versus training

Forward propagation:

```text
X → layers → final activation → output
```

It uses the current W and b values.

Training:

```text
forward propagation
→ calculate loss
→ backpropagation
→ gradients
→ optimizer updates W,b
→ repeat
```

A NumPy lab that gives `W1_tmp`, `b1_tmp`, etc. may only be demonstrating forward propagation with already-trained parameters.

---

## 19. Reading a Keras assignment without panicking

When shown a blank model definition, do not start by guessing syntax. Read the architecture first.

Ask:

1. How many input features does one example have?
2. How many layers are requested?
3. How many neurons/units are in each layer?
4. Which activation belongs to each layer?
5. What should the final layer output?
6. What should the shapes be?
7. Does `model.summary()` agree with my reasoning?

Then translate architecture → Keras.

This keeps the exercise an ML reasoning problem rather than a syntax-memory test.

---

# TensorFlow/Keras 60-second reference

```python
import tensorflow as tf

# input: n features per example
tf.keras.Input(shape=(n,))

# dense layer
tf.keras.layers.Dense(
    units=number_of_neurons,
    activation='sigmoid'
)

# sequential network
model = tf.keras.Sequential([
    tf.keras.Input(shape=(n,)),
    # Dense(...),
    # Dense(...)
])

# inspect architecture
model.summary()

# configure training
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01)
)

# train
model.fit(X, y, epochs=10)

# forward prediction
probabilities = model.predict(X_new)

# binary decision
yhat = (probabilities >= 0.5).astype(int)

# inspect a layer's parameters
W, b = layer.get_weights()
```

Golden mental pipeline:

```text
features
   ↓
Dense neuron(s): W·input + b
   ↓
activation
   ↓
next layer
   ↓
final output/probability
   ↓
threshold if a class decision is needed
```

And during training:

```text
forward → loss → backprop → gradients → optimizer → new W,b
```
