# Backpropagation using a computation graph — explained like I am 5

*A study companion for Andrew Ng / DeepLearning.AI's Machine Learning Specialization, Course 2: Advanced Learning Algorithms. Based on the optional lab “Back propagation using a computation graph” supplied in the learning conversation.*

## The whole story in one minute

Imagine a toy machine with little calculators inside and two adjustable knobs.

You put a number in. The calculators pass their answers along. At the end, you compare the machine's guess with the answer you wanted.

Now ask: **“If I turn each knob just a tiny bit, how will the mistake score change?”**

That is what backpropagation finds.

```text
FORWARD:   inputs + current knobs → little calculations → prediction → loss
BACKWARD:  parameter gradients   ← pass sensitivities backward       ← loss
UPDATE:    optimizer uses those gradients to turn the knobs
```

- **Forward pass:** calculate what happens with the current parameters.
- **Backward / reverse pass (backpropagation):** calculate how sensitive the loss is to each parameter.
- **Optimizer update:** actually change the parameters.

Backpropagation supplies the slopes. [Gradient descent](GRADIENT_DESCENT.md) or [Adam](TENSORFLOW_KERAS_FOR_ML.md#9-compile-decide-how-training-should-work) uses them to choose updates.

## 1. A derivative is a tiny-nudge question

A **derivative** answers:

> If I change this input a tiny amount, approximately how much does the output change, per unit of input change?

For `J = w**2` at `w = 3`:

```text
Before: w = 3       → J = 9
After:  w = 3.001   → J = 9.006001

Input change  = 0.001
Output change = 0.006001
Change ratio  = 0.006001 / 0.001 = 6.001
```

The exact derivative here is **6**. The tiny-nudge calculation gets close.

“Derivative 6” means a sufficiently small positive nudge of `0.001` produces an output increase of **about** `0.006`. It does not mean the output itself is 6, or that every large change gets multiplied by 6. A curve's slope can change as we move.

Read the lab's names as little questions:

| Name | Human translation |
| --- | --- |
| `dJ_da` or `∂J/∂a` | How sensitive is J to a small change in a? |
| `da_dw` or `∂a/∂w` | How sensitive is a to a small change in w? |
| `dJ_dw` or `∂J/∂w` | How sensitive is the final J to a small change in w? |

“With respect to w” just means **“when w changes.”** With several independent parameters, a partial derivative changes one while holding the others fixed. Intermediate values downstream are recalculated.

A **gradient** collects the partial derivatives for all the parameters. For our two-knob machine, it contains a slope for `w` and a slope for `b`.

## 2. A computation graph is a recipe broken into steps

The lab starts with:

```text
J = (2 + 3*w)**2
```

Instead of treating this as one complicated calculation, give the middle answer a name:

```text
w → [multiply by 3, then add 2] → a → [square] → J
                 a = 2 + 3*w              J = a**2
```

Each box is a **node**: a small operation whose output we can calculate and whose derivative we can understand.

Think of a chain of toy gears. A tiny movement at the first gear affects the next gear, which affects the last. Each connection has its own multiplier. To understand the whole chain, combine those multipliers.

For curved operations such as squaring, the multiplier depends on the current value. It is a description of **tiny changes near that value**.

## 3. Forward pass: fill in the ordinary numbers

At `w = 3`:

```text
w = 3 → a = 2 + 3*3 = 11 → J = 11**2 = 121
```

That is all a forward pass does here. Keep `a = 11`: the backward pass needs it to know the slope of the square operation.

In the lab's interactive diagram, these are the **blue boxes**, filled from left to right.

## 4. Reverse pass: send sensitivity back through the recipe

Start at the right-hand end.

### The square box

For `J = a**2`, the local derivative is `2*a`.

At `a = 11`:

```text
dJ_da = 2*11 = 22
```

So a tiny increase in `a` makes `J` increase by about 22 times that amount.

### The first box

For `a = 2 + 3*w`:

```text
da_dw = 3
```

The added 2 stays fixed. An increase of `0.001` in `w` increases `a` by `0.003`.

### Put the two effects together: the chain rule

The nudge travels through both boxes:

```text
tiny change in w → ×3 → change in a → ×22 → change in J

dJ_dw = da_dw * dJ_da
       = 3 * 22
       = 66
```

**The chain rule means multiplying the local sensitivities along this path.**

Backpropagation works backward because the final sensitivity arrives from the right. Each earlier box passes it on after multiplying by its own local derivative.

```text
Forward values:       w = 3  →  a = 11  →  J = 121
Backward slopes:   dJ_dw = 66 ← dJ_da = 22
                       ×3        square's slope at 11
```

In the lab, the **green boxes** hold derivative values. Fill them from right to left.

### Beginner-friendly Python

Run this entire block from top to bottom. It needs only ordinary Python.

```python
# FORWARD: calculate values using today's w.
w = 3
a = 2 + 3 * w
J = a ** 2                  # ** means "raised to a power"

# BACKWARD: calculate slopes; w itself is still 3.
dJ_da = 2 * a               # local slope of J = a squared
da_dw = 3                  # local slope of a = 2 + 3*w
dJ_dw = dJ_da * da_dw       # chain rule: combine both effects

print(f"a={a}, J={J}")
print(f"dJ/da={dJ_da}, da/dw={da_dw}, dJ/dw={dJ_dw}")

# Check with a tiny nudge of w, recalculating the whole forward path.
epsilon = 0.001             # small input change, NOT a learning rate
a_nudged = 2 + 3 * (w + epsilon)
J_nudged = a_nudged ** 2
approx_slope = (J_nudged - J) / epsilon
print(f"Tiny-nudge check: {approx_slope:.3f}")
```

Expected output:

```text
a=11, J=121
dJ/da=22, da/dw=3, dJ/dw=66
Tiny-nudge check: 66.009
```

The exact gradient is **66**; **66.009** is an approximation using a finite nudge. Making epsilon smaller usually improves the approximation initially, but an extremely tiny epsilon can introduce floating-point rounding problems.

## 5. The lab's simple neural network

Now the machine has a weight `w` and a bias `b`.

| Symbol | Meaning | Starting value |
| --- | --- | ---: |
| `x` | Input for one example | 2 |
| `w` | Learnable weight: scales x | -2 |
| `b` | Learnable bias: shifts the prediction | 8 |
| `y` | Correct answer / target | 1 |
| `c` | Weighted input, w*x | Calculated |
| `a` | Prediction, c+b | Calculated |
| `d` | Prediction minus target, a-y | Calculated |
| `J` | Loss / mistake score, d squared divided by 2 | Calculated |

This is a single linear neuron with no sigmoid or ReLU. Here `a` is the prediction itself. It can be 4; it is **not a probability**.

The lab calls the score `J`. In this example it is the loss for **one training example**, rather than an average over a dataset.

### Forward pass

```text
w=-2, x=2 → [multiply] → c=-4
                           ↓
                     [add b=8] → a=4
                                    ↓
                              [subtract y=1] → d=3
                                                  ↓
                                            [square / 2] → J=4.5
```

The machine guessed 4 when the target was 1. The error is 3. Squaring makes either sign of error contribute a nonnegative score. The factor `1/2` makes the square's derivative especially simple.

### Backward pass

Walk back one box at a time:

| Box | Local derivative | Combine with arriving sensitivity |
| --- | --- | --- |
| `J = d**2 / 2` | `dJ_dd = d = 3` | Start with sensitivity 3 |
| `d = a - y` | `dd_da = 1` | `dJ_da = 3 * 1 = 3` |
| `a = c + b`, toward c | `da_dc = 1` | `dJ_dc = 3 * 1 = 3` |
| `a = c + b`, toward b | `da_db = 1` | `dJ_db = 3 * 1 = 3` |
| `c = w * x`, toward w | `dc_dw = x = 2` | `dJ_dw = 3 * 2 = 6` |

Why does adding have a derivative of 1? If you increase one input by a tiny amount while holding the other fixed, the sum increases by exactly that amount.

Why is the derivative toward `w` equal to `x`, rather than `w`? With `x = 2` held fixed, changing `w` by `0.001` changes `w*x` by `0.002`.

The sensitivity at `a` goes down **both** incoming branches: toward `c` and toward `b`. It is not divided between them. Each partial derivative asks a separate question about changing one input.

The answers we wanted are:

```text
weight gradient: dJ_dw = 6
bias gradient:   dJ_db = 3
```

We hold `x` and `y` fixed because this training task adjusts the model's parameters. Derivatives with respect to inputs can exist; this lab simply does not need them.

### Python that mirrors every box

```python
# One example and today's parameters.
x = 2
w = -2
b = 8
y = 1

# FORWARD: store each intermediate answer.
c = w * x
a = c + b
d = a - y
J = d ** 2 / 2

# BACKWARD: arriving sensitivity times local derivative.
dJ_dd = d                  # slope of half the squared error
dd_da = 1                  # y stays fixed
dJ_da = dJ_dd * dd_da

da_dc = 1                  # addition passes sensitivity to c
da_db = 1                  # addition also passes sensitivity to b
dJ_dc = dJ_da * da_dc
dJ_db = dJ_da * da_db

dc_dw = x                  # x stays fixed in w*x
dJ_dw = dJ_dc * dc_dw

print(f"c={c}, a={a}, d={d}, J={J}")
print(f"dJ/dw={dJ_dw}, dJ/db={dJ_db}")
print(f"Parameters before any update: w={w}, b={b}")
```

Expected output:

```text
c=-4, a=4, d=3, J=4.5
dJ/dw=6, dJ/db=3
Parameters before any update: w=-2, b=8
```

## 6. Gradients are instructions for an update, not new parameters

A gradient of 6 does **not** mean “set w to 6.”

Here both gradients are positive. Increasing either parameter slightly would increase the loss, holding the other fixed. Gradient descent subtracts a small multiple of each gradient.

Continue the previous Python block:

```python
# UPDATE: use both gradients from the SAME forward/backward pass.
alpha = 0.1                # learning rate: controls update size
w_new = w - alpha * dJ_dw   # -2 - 0.1*6 = -2.6
b_new = b - alpha * dJ_db   #  8 - 0.1*3 =  7.7

# A fresh forward pass checks the updated machine.
a_new = w_new * x + b_new
J_new = (a_new - y) ** 2 / 2
print(f"w_new={w_new:.1f}, b_new={b_new:.1f}")
print(f"prediction={a_new:.1f}, loss={J_new:.3f}")
```

Expected output:

```text
w_new=-2.6, b_new=7.7
prediction=2.5, loss=1.125
```

The loss fell from **4.5 to 1.125** in this example. A large learning rate can overshoot, so one successful step does not make every step size safe.

Notice that the negative weight became **more negative**. The gradient's sign describes the loss's slope, not the sign of the weight.

[The earlier gradient descent guide](GRADIENT_DESCENT.md#4-gradient-descent-walk-downhill) explains these updates. [The Keras guide's Adam introduction](TENSORFLOW_KERAS_FOR_ML.md#9-compile-decide-how-training-should-work) connects them to an optimizer in a framework.

Adam also consumes gradients, but uses moving averages of gradients and their squares to adapt updates for each parameter. Its updates are not generally the plain `parameter - alpha * current_gradient` calculation above. Backpropagation still supplies the gradients in either case.

## 7. Where SymPy and frameworks fit

The lab demonstrates two ways to understand a local derivative:

- **Arithmetic:** nudge an input and measure the output-change ratio.
- **Symbolic:** let SymPy derive an expression, then substitute current values.

Optional SymPy check, if SymPy is installed:

```python
from sympy import symbols, diff

sw = symbols("w")           # a symbolic placeholder, not today's numeric w
sJ = (2 + 3 * sw) ** 2      # a mathematical expression
slope = diff(sJ, sw)        # derive the expression with respect to w

print(slope.subs(sw, 3))    # substitute w=3 into the derivative: 66
```

Using SymPy to differentiate a whole formula is a useful check. The point of the computation graph is to see how the same answer emerges from **small local rules combined backward**.

Deep-learning frameworks automate that backward calculation using automatic differentiation. They do not normally estimate every parameter's gradient by nudging it separately, or ask SymPy to expand a giant network formula.

For larger graphs, keep one extra rule in mind: **if a value influences the loss through several paths, add the gradient contributions from those paths.** Multiply along a path; add contributions when paths meet.

For many training examples, an average loss also requires combining and averaging their contributions. Our lab example has only one example and one path from each parameter to the loss.

## 8. Common mix-ups

| Mix-up | Useful correction |
| --- | --- |
| “Backward means undo the prediction.” | It passes sensitivities backward using saved forward values. |
| “The error d is every gradient.” | Here d is 3, but the weight gradient is d*x = 6. |
| “A gradient of 66 means J is 66.” | J is 121; 66 describes its sensitivity to w. |
| “A local derivative is already the final parameter gradient.” | Combine it with the sensitivity arriving from the loss. |
| “Backpropagation changed the weights.” | Parameter changes happen in the optimizer update. |
| “Epsilon and alpha are the same.” | Epsilon probes a slope; alpha controls a learning step. |
| “All neuron outputs are probabilities.” | This linear neuron's output is 4. |
| “I can run notebook cells in any order.” | Reused names can hold stale values. Restart and run top to bottom. |

The lab's widgets and plotting helpers draw interactive boxes; they are not the backpropagation algorithm. If the diagram becomes unresponsive, rerun its plotting cell. The plain Python examples here do not need those helpers.

## 9. Tiny exercises

Try these before opening the answers.

1. In the first graph, what are `a` and `J` when `w = 3`?
2. Why is `dJ_dw` equal to 66 rather than 22?
3. At that point, a positive nudge of `0.001` in `w` changes `J` by approximately how much?
4. In the network, why is the weight gradient 6 while the bias gradient is 3?
5. After backpropagation alone, what are `w` and `b`?
6. With learning rate 0.1, what are their updated values?
7. If the same network input were `x = 0`, what would `dJ_dw` be, and why?

<details>
<summary>Answers — open after trying</summary>

1. `a = 11`, `J = 121`.
2. A nudge in w first gets multiplied by 3 on its way to a, then by the square's current sensitivity of 22: `3*22 = 66`.
3. About `66*0.001 = 0.066`. The actual increase is `0.066009`; the derivative describes the limiting tiny-change ratio.
4. The weight acts through multiplication by x=2; the bias acts through addition with local derivative 1. So `3*2 = 6` and `3*1 = 3`.
5. Still `w = -2`, `b = 8`. Gradients have been calculated, but no update has happened.
6. `w_new = -2.6`, `b_new = 7.7`, using the same pair of calculated gradients.
7. Zero. For this example, changing w cannot affect `w*x` when x is zero, even though the prediction and loss may still be wrong.

</details>

## 60-second refresher

```text
Forward:  calculate values and remember intermediate answers.
Backward: start at the loss and multiply by each local derivative.
Result:   a gradient for each learnable parameter.
Update:   the optimizer uses those gradients to change parameters.
Repeat:   run a fresh forward pass with the new parameters.
```

> **Forward tells me what happened. Backprop tells me how the loss responds to each knob. The optimizer turns the knobs.**

Continue with the [Week 1 neural-network guide](COURSE_2_WEEK_1_NEURAL_NETWORKS.md) to connect this tiny graph to layers, or the [Python quick reference](PYTHON_FOR_ML_QUICK_REFERENCE.md) to revisit the code patterns.

**Source note:** The primary source is the user-supplied optional lab “Back propagation using a computation graph,” including its two worked examples. These explanations and annotated code reorganize that material for a beginner. The gradient-descent update and exercises are added study aids, not claims of course completion or mastery.
