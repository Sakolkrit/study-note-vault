---
tags: [machine-learning, neural-network, deep-learning, inference]
status: growing
level: 4
tier: 2
---

# Forward Propagation

## 🟢 1. Simple (age 12)
Forward propagation is **how a neural network turns an input into an answer by passing numbers forward, layer by layer**, like a row of people each doing one small calculation and handing the result to the next person. The last person says the answer.

**Example:** A factory line. Raw wood goes in, the first station cuts it, the second sands it, the third paints it, and a chair comes out at the end. Nobody goes backwards; the stuff only moves forward.

## 🔵 2. How it works
1. The input is a list of numbers, such as a coffee roast's temperature and duration.
2. Each **unit** (neuron) in the first layer takes all the inputs, multiplies each by its own weight, adds them up, adds its own bias, and passes the sum through an [[Activation Function]] (for example the sigmoid curve, which squeezes any number into 0 to 1).
3. The numbers that come out of the whole layer are the input of the next layer. Repeat for every layer.
4. The last layer gives the prediction, such as "chance this roast is good".
5. Nothing is learned in this step. It only **uses** the current weights. Learning (changing the weights) is [[Backpropagation]], which first needs this forward pass to know what the network answered.
6. One layer with one unit is exactly [[Logistic Regression]]; stacking layers is what makes a [[Neural Network]].

## 🟠 3. Technical detail
- **A layer in one line:** write the weights of all its units as a matrix $W$ (one column or row per unit) and the biases as a vector $b$. The whole layer is then one matrix product, which is why the loop over units disappears ([[Vectorization]]).
- **Shapes:** a layer with 2 inputs and 3 units has a weight matrix with 2×3 numbers and 3 biases, so 9 parameters. The next layer with 1 unit adds 3 weights and 1 bias, 4 more. (Those are the counts the course's coffee-roasting lab reports, 9 and 4; I reran the lab and got the same.)
- **Why the activation matters:** if every layer were just $Wx+b$ with no activation, stacking layers would still be one linear function. The curve between layers is what lets the network bend its answer ([[Activation Function]]).
- **Code:** in [[TensorFlow]] a `Dense` layer does this; in [[NumPy]] it is `sigmoid(W.T @ a + b)`. The repo's saved lab also builds the same network with plain NumPy for the forward pass only.
- **Cost:** one forward pass costs about one multiply-add per weight, so a network with a million weights does about a million per example.

```python
import numpy as np
sig = lambda z: 1 / (1 + np.exp(-z))
unit = lambda w, x: sig(w @ np.r_[1.0, x])        # w = [bias, w1, w2]
AND, NOR, OR = np.array([-30, 20, 20.]), np.array([10, -20, -20.]), np.array([-10, 20, 20.])
def xnor(x):                                       # two layers: (AND, NOR) then OR
    return unit(OR, np.array([unit(AND, x), unit(NOR, x)]))
for x in ([0, 0], [0, 1], [1, 0], [1, 1]):
    print(x, "AND ~", round(float(unit(AND, x)), 3), "| XNOR", int(round(float(xnor(x)))))
# [0, 0] AND ~ 0.0 | XNOR 1
# [0, 1] AND ~ 0.0 | XNOR 0
# [1, 0] AND ~ 0.0 | XNOR 0
# [1, 1] AND ~ 1.0 | XNOR 1
```

## 🔴 4. Deep / formal
**One layer of a network.** With $a^{[0]}=x$ as the input, layer $l$ computes:

$$
a^{[l]}=g\!\left(W^{[l]}\,a^{[l-1]}+b^{[l]}\right)
$$

**Where:**
- $a^{[l-1]}$ is the vector of numbers coming into layer $l$ (the outputs of the layer before). $a^{[l]}$ is the vector this layer produces.
- $W^{[l]}$ is the layer's weight matrix: how strongly each incoming number feeds each unit. $b^{[l]}$ is the vector of biases, one per unit.
- $g$ is the activation function, applied to each entry of its input separately.
- **Matters:** this one formula is applied again and again, so the whole network is a chain of simple steps. Without $g$ the chain collapses into a single $Wx+b$ and the network is no more powerful than [[Linear Regression]].

**Example (tiny numbers):** $W=\begin{pmatrix}1&2\\-1&0.5\end{pmatrix}$, $b=(0,\,1)$, input $a^{[0]}=(1,\,2)$, $g$ = sigmoid.
- Unit 1: $1\cdot1+2\cdot2+0=5$. Unit 2: $-1\cdot1+0.5\cdot2+1=1$.
- Squeeze: $\sigma(5)=0.9933$ and $\sigma(1)=0.7311$.
- Meaning: the layer turned the input $(1,2)$ into $(0.993,\,0.731)$, which is what the next layer receives.

**Why a hidden layer is needed (the XNOR example).** XNOR is 1 when both inputs are equal. No single unit can compute it, because one unit cuts the plane with one straight line ([[Decision Boundary]]) and XNOR's 1s sit at opposite corners. Two layers can: the first layer computes AND and NOR, and the second layer ORs them. The code above does exactly that with weights from the repo's saved notes.

**Limits:**
- Forward propagation only evaluates. Choosing the weights needs a cost, its gradient, and an optimiser ([[Backpropagation]], [[Gradient Descent]]).
- Deep chains can make signals shrink or blow up from layer to layer ([[Vanishing and Exploding Gradients]]).
- Numerically, the sigmoid saturates for large $|z|$, which is why frameworks often work with the raw score and a fused loss.

**What came before, and how this improved on it:**
- **Before:** one unit: a weighted sum and a threshold or sigmoid ([[Perceptron]], [[Logistic Regression]]).
- **Its limit:** one unit draws one straight boundary, so it cannot learn patterns such as XNOR.
- **What changed:** feed the outputs of some units into more units (hidden layers), computed forward layer by layer.
- **How much better:** in the code below, a grid search over 61³ weight settings of a single unit could get at most 3 of the 4 XNOR cases right (75%). The two-layer network above gets all 4 (100%).

**Check it yourself:**

```python
import numpy as np, itertools
pts = [(0, 0), (0, 1), (1, 0), (1, 1)]; ys = [1, 0, 0, 1]                  # XNOR
g = np.linspace(-3, 3, 61); best = 0
for w0, w1, w2 in itertools.product(g, g, g):
    acc = sum(int(w0 + w1 * a + w2 * c >= 0) == t for (a, c), t in zip(pts, ys)) / 4
    best = max(best, acc)
print("best single unit on XNOR:", best)                                    # 0.75
sig = lambda z: 1 / (1 + np.exp(-z))
W = np.array([[1.0, 2.0], [-1.0, 0.5]]); b = np.array([0.0, 1.0]); x = np.array([1.0, 2.0])
print(np.round(sig(W @ x + b), 4))                                           # [0.9933 0.7311]
```

## Related
- [[Neural Network]] · [[Activation Function]] · [[Backpropagation]] · [[Logistic Regression]] · [[Perceptron]] · [[Decision Boundary]] · [[Vectorization]] · [[TensorFlow]] · [[NumPy]] · [[Matrix]] · [[Vanishing and Exploding Gradients]] · [[Machine Learning Specialization Course 2 Week 1]]

## Sources
- Repo notes `2. Advanced Algorithms/Week 1/notes.pdf` (an older course's notes; AND/NOR/OR weights and the XNOR construction are from there) and the repo's lab notebooks (parameter counts 9 and 4, confirmed by rerun); see [[Machine Learning Specialization]]
- The 75% figure is measured in this note's code

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/neural-nets Forward propagation:::How a network turns an input into an answer by passing numbers forward layer by layer, each layer doing one small calculation and handing the result on.

#flashcards/ml/formula #flashcards/topic/neural-nets Write the forward step of one layer and explain each symbol.
?
$$a^{[l]}=g\!\left(W^{[l]}\,a^{[l-1]}+b^{[l]}\right)$$
- $a^{[l-1]}$: the vector of numbers coming into layer $l$ (the outputs of the layer before). $a^{[l]}$: the vector this layer produces.
- $W^{[l]}$: the weight matrix, how strongly each incoming number feeds each unit. $b^{[l]}$: the biases, one per unit.
- $g$: the activation function, applied to each entry separately.
- This one formula is applied again and again, so the network is a chain of simple steps. Without $g$ the chain collapses into a single $Wx+b$ and is no more powerful than one layer.

#flashcards/ml/example #flashcards/topic/neural-nets $W=\begin{pmatrix}1&2\\-1&0.5\end{pmatrix}$, $b=(0,1)$, input $(1,2)$, $g$ = sigmoid. What does the layer output?::Unit 1: $1\cdot1+2\cdot2+0=5$. Unit 2: $-1\cdot1+0.5\cdot2+1=1$. Sigmoid: $\sigma(5)=0.9933$, $\sigma(1)=0.7311$. The layer turns $(1,2)$ into $(0.993,0.731)$, which the next layer receives.

#flashcards/ml/concept #flashcards/topic/neural-nets Why does a network need an activation function between layers?::Without it the layers collapse into one $Wx+b$ and the network is no more powerful than a single layer.
