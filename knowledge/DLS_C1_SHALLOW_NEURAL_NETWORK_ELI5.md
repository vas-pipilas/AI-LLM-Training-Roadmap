# DLS Course 1 — Shallow Neural Network Lab, Explained Like I'm Five (Without Losing the Maths)

**Recorded: 15 November 2026.** Based on screenshots and user-written code for the one-hidden-layer neural network lab, especially Exercises 2 (layer_sizes), 3 (forward_propagation) and 5 (compute_cost). This is a study companion; no complete notebook for other graded exercises was supplied. Independent examples below are illustrative, not copied lab test inputs.

## 1. The entire lab is one story

A neural network receives example inputs X, computes a hidden representation through weights, biases and tanh, then produces probabilities through another set of weights, biases and sigmoid. The correct answers Y are known in supervised learning. A loss/cost function measures how badly predictions differ from answers. **Forward propagation predicts; calculating loss measures error; derivatives/backpropagation later tell us how changing weights changes that error; an optimizer makes the update.** Do not confuse these stages.

Imagine a learner: X is what they can see, weights encode how to combine measurements, the hidden layer creates intermediate signals, A2 is the current guess, Y is the true answer, and J is the average penalty for wrong/confident guesses. The analogy stops at the computational distinction: network weights are numerical parameters, not literal opinions.

## 2. The first trap: shape of an entire array versus size along an axis

Andrew Ng's DLS convention: columns represent examples. X has shape (n_x,m); Y has shape (n_y,m). For X.shape=(5,3), five features are each observed over three examples. The input layer size is 5, not (5,3).

Three Python expressions are not interchangeable:

~~~python
import numpy as np
X = np.arange(15).reshape(5,3)
print(np.shape(X))       # (5,3): shape of original matrix
print(np.shape(X[0]))    # (3,): shape of FIRST DATA ROW
print(np.shape(X)[0])    # 5: first number in the SHAPE tuple
~~~

The student first returned whole tuples for n_x and n_y, then accidentally indexed the data before finding its shape; finally corrected to retrieving the first dimension of the shape. The reported public test printed n_x=5, n_h=4, n_y=2 and **All tests passed!** The hidden width 4 was explicitly prescribed by the exercise.

**Remember:** index the thing whose contents you need. X[0] is a row of data; X.shape[0] is the number of rows. The first row is not the first dimension.

## 3. One neuron: two operations, in this order

First z = w·x + b (an affine weighted combination). Then a = g(z) (activation). With x=3, w=2 and b=-8, z=-2; ReLU(z)=0. Activation before the weighted sum is a different function.

In this lab the hidden layer uses tanh, which maps numbers into (-1,1), while the final layer uses sigmoid, which maps numbers into (0,1). The four hidden activations are intermediate learned signals, not four class labels. A2 is an estimated positive-class probability per example.

## 4. The full forward pass and shapes

The notebook gives four equations: Z1=W1 X+b1; A1=tanh(Z1); Z2=W2 A1+b2; A2=sigmoid(Z2). The student independently wrote the correct four NumPy operations using dot products, broadcasting, tanh and sigmoid. **The forward-propagation public test result has not been shown**, so the code is verified by inspection against the supplied formula, not recorded as a public-test PASS.

With four hidden neurons, one output neuron and m column-wise examples:

| Array | Shape | Meaning |
| --- | --- | --- |
| X | (n_x,m) | features by examples |
| W1 | (4,n_x) | four neurons each receiving n_x inputs |
| b1 | (4,1) | one bias per hidden neuron |
| Z1, A1 | (4,m) | four hidden values for every example |
| W2 | (1,4) | output neuron receives four hidden values |
| b2 | (1,1) | output bias |
| Z2, A2 | (1,m) | one output probability per example |

The matrix product (4,n_x) @ (n_x,m) creates (4,m); the product (1,4) @ (4,m) creates (1,m). Adding a (4,1) bias broadcasts it across m columns; tanh and sigmoid change element values, not shapes. Andrew's W orientation differs from the internal kernel orientation typically used in Keras Dense layers. Always follow the formula and course convention.

The parameters are retrieved from an existing dictionary, not regenerated. The student first tried a keyword-style dictionary index, then corrected it. General Python reminder: dictionary['name'] looks up a key; function(name=value) passes an argument.

## 5. Cross-entropy: the formula is a recipe with two cases

The notebook's binary classification cost is

J = -(1/m) * sum_i [ y_i*log(p_i) + (1-y_i)*log(1-p_i) ], with p_i=A2 for each example.

If the true label is 1: y=1 and 1-y=0, so the loss for that example is -log(p). If true label is 0: y=0, so the loss is -log(1-p). The y and (1-y) terms are selectors of which probability to penalize.

Correct and confident -> small cost; confidently wrong -> large cost. For example:

| Actual y | Predicted positive p | Loss |
| --- | --- | --- |
| 1 | 0.9 | -log(0.9) ≈ 0.10536 |
| 0 | 0.2 | -log(0.8) ≈ 0.22314 |
| 1 | 0.8 | -log(0.8) ≈ 0.22314 |

Sum ≈ 0.55164; **average over 3 examples** ≈ 0.18388. The crucial missing piece in the student's first attempt was division by m, the example count. All other essential pieces of their cost expression—including complementary log term, negative sign, transposes and aggregated dot products—were already correct.

### The NumPy mechanics without mysticism

Y and A2 have shape (1,m). A dot product (1,m) @ (m,1) gives one (1,1) total because the product sums over all m examples. A second dot product aggregates the complementary term. Divide the total by m to get an average. Using np.sum on an already single-element dot result extracts its sole aggregated value; it does not add an additional set of examples. np.squeeze and float in the given starter cell ensure the expected Python scalar type.

The student independently incorporated the missing /m after a hint. The screenshot then displayed **All tests passed!**, with recorded cost **0.6930587610394646**. This is public-test evidence for compute_cost; it is not a claim of independent model generalization.

**Numerical caution:** log(0) is undefined. Practical implementations use stable logits-based cross-entropy calculations; the notebook's direct logarithms are useful for understanding the mathematics.

## 6. What was the calculus actually doing?

Calculus is about how much a quantity changes when you change an input parameter a tiny amount. Loss J is the **height** of the mistake; derivative dJ/dw is the **slope** of that mistake with respect to weight w. These are not the same numerical object.

Independent easy example: prediction z=w*x+b; loss L=0.5*(z-y)^2. Let x=2, y=5, w=1, b=0. Then z=2, error=-3 and loss=4.5. By the chain rule, dL/dw=(dL/dz)*(dz/dw)=(-3)*2=-6. At learning rate 0.1, updating w by subtracting 0.1*(-6) gives 1.6. This illustration deliberately uses **simple squared loss**, not the exact derivative of the lab's tanh/sigmoid cross-entropy network.

For the actual shallow neural network, backpropagation will use chain rule through both weighted sums and both activations. The user has not supplied their later backprop exercise; therefore this document does not invent their submitted gradient code or claim derivative mastery.

## 7. The student's actual friction, and why it is not the whole story

**Observed strengths:** Independently produced the complete forward-pass formula-to-NumPy implementation; retrieved parameters correctly after a syntax fix; included the two difficult cross-entropy terms and transpose operations; passed layer_sizes and compute_cost public tests.

**Observed friction:** Confusing data indexing with shape indexing, interpreting layer dimensions, switching between examples-as-columns and examples-as-rows, occasional dictionary syntax confusion, and overlooking one coefficient (/m) while translating a complex equation. User reported that the calculus itself felt difficult, even though guided coding worked acceptably.

**Precise status:** Fixed these specific exercises with feedback; independent mastery of new examples and the full shallow-network backpropagation have **not yet been established**. No need to restart beginner Python or change the Deep Learning -> PyTorch -> LLM career destination.

## 8. The five-step formula-to-code audit

1. Who produced X and Y, and which axis represents examples?
2. Write the shapes of operands and outputs next to each matrix multiplication.
3. Identify every operation in the mathematical expression: nonlinear function, sum, negative sign, scalar factors, division.
4. Compare the entire equation against the completed code, paying particular attention to terms outside parentheses or summations (like 1/m).
5. Test a tiny hand-calculated case before relying on the official public test.

**Next targeted bridge, only when useful:** take one example and calculate (weighted sum -> tanh -> weighted sum -> sigmoid -> binary cross-entropy) with concrete numbers; then practice a one-parameter numerical derivative and connect it to the chain rule. Prefer a diagnostic hint before being told exactly which term is missing, because the student's stated career goal is increasingly independent debugging.

## References

- [DLS C1 earlier logistic-regression lab ELI5](DLS_C1_W2_LOGISTIC_REGRESSION_ELI5.md) (different lab).
- [Deep Learning Specialization learning tracker](DEEP_LEARNING_SPECIALIZATION_PROGRESS.md).
- [Canonical Python ML coding cheat sheet](PYTHON_ML_CODING_CHEAT_SHEET.md).
- [Backpropagation/computation graphs foundation](COURSE_2_BACKPROPAGATION_COMPUTATION_GRAPH.md).
- [Adaptive training contract](ADAPTIVE_ML_LLM_TRAINING_CONTRACT.md).

**Bottom line:** A formula is a detailed recipe. Getting the main operations right while missing the division by m is not proof you cannot do mathematics. The engineering skill to develop is comparing every ingredient of the recipe against the corresponding line of code.