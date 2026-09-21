# Gradient Descent — My Reference Guide

This is not meant to be a calculus textbook. It is the explanation I want to come back to when I need to remember what gradient descent is actually doing.

The goal is to connect the notation, the Python code, and the intuition.

---

## 1. Start with the model

For simple linear regression:

```text
f(x) = wx + b
```

The pieces have different jobs:

- `x` — the input / feature. Example: house size.
- `y` — what actually happened. Example: real selling price.
- `w` and `b` — parameters the model can learn.
- `f(x)` — the model's prediction.
- `m` — number of training examples.

The mental shortcut:

> **x = what I know**  
> **y = reality**  
> **w, b = knobs the model can adjust**  
> **f(x) = what the model thinks**

Example:

```text
x = 120
w = 2500
b = 15000

f(x) = 2500 × 120 + 15000
     = 315000
```

If the real selling price is `y = 310000`, the prediction is not perfect. Training is about finding better values for `w` and `b`.

---

## 2. Cost: how wrong is the model?

The cost function is written as:

```text
J(w,b)
```

For now, the most useful interpretation is:

> **Cost tells me how badly the current model parameters fit the training data.**

Large cost = bad fit.

Smaller cost = better fit.

Gradient descent's job is to find values of `w` and `b` that make `J(w,b)` as small as possible.

Picture a valley:

```text
Cost J
  ↑
  |  \                 /
  |   \               /
  |    \             /
  |     \___________/
  |          ^
  |       minimum
  +----------------------→ parameters
```

Training means trying to reach the bottom.

---

## 3. Derivative: slope, not position

This was an important distinction for me.

A derivative tells me the **slope / rate of change at the point where I currently am**.

For example:

```text
dJ/dw
```

means roughly:

> **If I change w slightly, in which direction and how strongly will J change?**

A point can be above zero while having a negative slope.

Think of standing 500 metres above sea level on a downhill road:

- altitude is positive;
- slope can still be negative.

So when a diagram shows a derivative of `-2`, it does **not** mean `J = -2`. It means the curve has a negative slope there.

For two parameters we need two partial derivatives:

```text
∂J/∂w   → slope of cost with respect to w
∂J/∂b   → slope of cost with respect to b
```

In code these often appear as:

```python
dj_dw
dj_db
```

---

## 4. Gradient descent: walk downhill

The update rules are:

```text
w = w - α × ∂J/∂w
b = b - α × ∂J/∂b
```

where `α` (alpha) is the **learning rate**.

The simplest mental model:

> **Derivative/gradient tells me which way is uphill. Gradient descent deliberately takes a step the other way.**

Or:

```text
feel the slope
     ↓
take a small downhill step
     ↓
feel the new slope
     ↓
take another downhill step
     ↓
repeat
```

That repetition is training.

---

## 5. The sign that originally confused me

Suppose:

```text
w = 10
alpha = 0.1
dj_dw = -2
```

Then:

```text
w_new = 10 - (0.1 × -2)
      = 10 - (-0.2)
      = 10.2
```

So a **negative gradient makes w increase**.

Why? Because subtracting a negative number means adding.

Useful shortcut:

```text
dj_dw < 0  → w increases
dj_dw > 0  → w decreases
```

The same sign logic applies to `b`.

Do not confuse:

> **negative slope**

with:

> **the parameter must become negative or decrease**

They are different statements.

---

## 6. What `compute_gradient()` is doing

A simple implementation may look conceptually like this:

```python
for each training example:
    prediction = w * x[i] + b
    error = prediction - y[i]

    calculate that example's contribution to dj_dw
    calculate that example's contribution to dj_db

average the gradients over all examples
return dj_dw, dj_db
```

The useful connection is:

```text
prediction - reality
     f(x)  - y
```

The calculus determines the exact gradient formulas, but I do not need to memorize a giant derivative table to understand the algorithm.

I need to understand what the gradients **mean** and how they affect the parameter updates.

---

## 7. What the gradient-descent loop is doing

The heart of the training code is surprisingly small:

```python
for i in range(num_iters):
    dj_dw, dj_db = compute_gradient(x, y, w, b)

    w = w - alpha * dj_dw
    b = b - alpha * dj_db
```

Human translation:

```text
Look at current w,b
       ↓
Make predictions
       ↓
Measure the error/cost
       ↓
Calculate the slopes for w,b
       ↓
Move w,b a small step downhill
       ↓
Repeat
```

The model is not being handed the correct `w` and `b`.

It **learns them from the training data by repeatedly reducing the cost**.

That is the important meaning of "learning" in this simple model.

---

## 8. Reading the training output

One experiment started at:

```text
w = 0
b = 0
```

and eventually converged very close to:

```text
w = 200
b = 100
```

The output showed roughly:

```text
Iteration 0       Cost ≈ 79300
Iteration 1000    Cost ≈ 3.41
Iteration 5000    Cost ≈ 0.00995
Iteration 9000    Cost ≈ 0.000029
```

At the same time, the gradients became tiny.

Early:

```text
dj_dw ≈ -650
dj_db ≈ -400
```

Later:

```text
dj_dw ≈ -0.001082
dj_db ≈  0.001751
```

Interpretation:

> At the beginning the hill is steep, so parameter corrections are large. Near the bottom the surface is almost flat, so corrections become tiny.

---

## 9. Scientific notation in ML output

Python frequently displays very large or very small numbers using scientific notation.

```text
6.500e+02 = 6.500 × 10² = 650
3.712e-01 = 3.712 × 10⁻¹ = 0.3712
2.004e-02 = 0.02004
1.082e-03 = 0.001082
2.90e-05  = 0.000029
```

Quick mental rule:

```text
e+02 → move decimal 2 places right
e+03 → move decimal 3 places right

e-02 → move decimal 2 places left
e-03 → move decimal 3 places left
```

A learning rate such as:

```python
alpha = 1e-2
```

simply means:

```text
alpha = 0.01
```

Scientific notation is only a compact representation of the same number.

---

## 10. Convergence: when more iterations stop helping

As gradient descent approaches the minimum:

```text
cost → very small
gradients → approximately zero
w,b → stop changing meaningfully
```

This is **convergence**.

In the experiment, increasing the run from 10,000 to 30,000 iterations produced values essentially equal to:

```text
w = 200
b = 100
```

with gradients down around billionths.

At that point, additional computation has no meaningful benefit for the problem.

The precise engineering idea is not "run until mathematically perfect." It is:

> **Run until the solution is sufficiently converged for the precision/performance we care about.**

Real datasets contain noise, and real training uses practical stopping criteria.

---

## 11. Learning rate: step size matters

The learning rate `alpha` controls how large each downhill step is.

### Too small

```text
tiny step → tiny step → tiny step → tiny step → ...
```

It may eventually converge, but training can be unnecessarily slow.

### Sensible

```text
step → step → smaller step → smaller step → minimum
```

Good.

### Too large

The algorithm can jump over the minimum:

```text
        minimum
           ↓
left  ←----+----→ right
       huge jumps
```

It can repeatedly overshoot and even move farther and farther away.

That is **divergence**.

So:

> **More aggressive learning is not automatically faster learning.**

---

## 12. Why not just run millions of iterations?

If the gradients are already approximately zero and the cost is no longer meaningfully improving, millions of additional iterations mostly waste computation.

More iterations cannot compensate for every problem either. For example, a learning rate that causes divergence will not become good just because we let it run longer.

The useful questions are:

- Is cost decreasing?
- Are parameter updates becoming smaller?
- Are the gradients approaching zero?
- Has the solution converged sufficiently?
- Is the learning rate appropriate?

---

## 13. What I need to remember about the maths

I do **not** need to become a human symbolic-calculus engine.

I do need to recognize and understand:

```text
J(w,b)      → cost / how wrong the model is

∂J/∂w       → slope of cost with respect to w
∂J/∂b       → slope of cost with respect to b

alpha       → learning rate / step size

w,b         → learnable parameters
```

During training it is useful to implement simple gradient descent manually because it exposes the machinery.

Later, libraries and frameworks handle much of this. Scikit-learn provides higher-level model training APIs, and deep-learning frameworks such as PyTorch can perform automatic differentiation.

The goal is not to manually differentiate giant neural networks. The goal is to understand what the framework is doing well enough to reason about training and debug it.

---

## 14. Connection to later Deep Learning / LLM work

Linear regression has only two learnable parameters in this example:

```text
w
b
```

A neural network may have millions or billions of parameters.

The scale changes enormously, but an important underlying pattern survives:

```text
input
  ↓
model
  ↓
prediction
  ↓
loss
  ↓
gradients
  ↓
update parameters
  ↓
repeat
```

Later terms such as **backpropagation**, **autograd**, **optimizer**, **learning rate**, **vanishing gradients**, and **exploding gradients** build on this foundation.

That is why gradient descent is core knowledge rather than something to memorize for one quiz.


---

## 15. Feature scaling / Z-score normalization

When different features live on wildly different numerical scales, gradient descent can become awkward and slow.

For example, one house may look like:

```text
size       = 2000
bedrooms   = 3
floors     = 1
age        = 40
```

Z-score normalization transforms each feature separately:

```text
x_norm = (x - mu) / sigma
```

where:

- `mu` is the mean of that feature in the training set.
- `sigma` is the standard deviation of that feature in the training set.

The useful intuition is that the normalized number tells me roughly **how far above or below the training-set average this value is, measured in standard deviations**.

Example:

```text
mean house size = 1500 sqft
standard deviation = 500 sqft

2000 sqft → (2000 - 1500) / 500 = +1
1500 sqft → (1500 - 1500) / 500 =  0
1000 sqft → (1000 - 1500) / 500 = -1
```

So `-1` does not mean a negative house size. It means one standard deviation below the training-set mean.

After normalization, features that originally looked like:

```text
[2000, 3, 1, 40]
```

may instead look roughly like:

```text
[1.2, 0.4, -0.7, 0.1]
```

Their scales are now much more comparable, which generally lets gradient descent make more balanced progress across the parameters.

### The golden rule for prediction

> **A model trained on normalized features must receive new inputs normalized with the SAME mean and standard deviation that were learned from the TRAINING data.**

This is not optional preprocessing decoration. Those saved values are part of the transformation the trained model expects.

Training:

```text
X_train
   ↓
calculate training mu and sigma
   ↓
X_train_norm = (X_train - mu) / sigma
   ↓
train model
   ↓
learn w,b
```

Later, for a real new example:

```text
real new X
   ↓
use the SAME saved training mu and sigma
   ↓
X_new_norm = (X_new - training_mu) / training_sigma
   ↓
trained model
   ↓
prediction
```

Do **not** calculate a new mean and standard deviation from the new example. That would put the new input on a different numerical reference system from the one the model learned.

A useful way to think about it:

> **The saved training mu and sigma let me speak the same numerical language to the model at prediction time that I used when training it.**

If only the input features `X` were normalized and the target `y` was left in its original units, the model's prediction is already in the target's original units. There is no need to "denormalize the prediction."

If I ever need to convert a normalized feature back to its original value, the inverse is:

```text
x = x_norm * sigma + mu
```

So the quick pipeline is:

```text
real X → normalize with TRAINING mu/sigma → model → real y prediction
```

---

# Quick refresh — 60 seconds

If I have almost no time, remember this:

> **Model:** `f(x) = wx + b`  
> **x:** input  
> **y:** reality  
> **w,b:** knobs the model learns  
> **J(w,b):** how wrong the current model is  
> **gradient:** slope — how cost changes when a parameter changes  
> **alpha:** how large a learning step to take  
> **gradient descent:** repeatedly move parameters opposite the uphill gradient to reduce cost  
> **convergence:** gradients become tiny, cost stops meaningfully improving, parameters stabilize  
> **bad alpha:** too small can be slow; too large can overshoot/diverge

And the sign check:

```text
negative gradient → subtract negative → parameter increases
positive gradient → subtract positive → parameter decreases
```

The one-line mental model:

> **Cost says how bad I am. Gradient says which way is uphill. Gradient descent says: go the other way.**
