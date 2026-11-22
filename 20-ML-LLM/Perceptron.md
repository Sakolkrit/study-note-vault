---
tags: [machine-learning, neural-network, deep-learning, classification, history]
status: growing
level: 4
tier: 2
---

# Perceptron

The **perceptron** is the oldest learning neuron: a tiny program that adds up its inputs with weights, says "yes" or "no", and fixes its weights whenever it is wrong. Every unit inside a modern [[Neural Network]] is a descendant of it.

## 🟢 1. Simple (age 12)
A perceptron is a little judge. It looks at a few facts, gives each fact a "how much do I care" score, adds them up, and says **yes** if the total is big enough. When it says the wrong thing, it changes its scores a little so it does better next time.

**Example:** a judge decides "is it a good day for the beach?" from sun, wind and homework. Sun gets a big plus, wind a small minus, homework a big minus. If the judge says "go!" on a day you had to stay home, it makes the homework minus even bigger.

## 🔵 2. How it works
1. The input is a list of numbers, one per fact ([[Vector]]). For a picture it could be the brightness of each pixel.
2. Each input has a **weight**: how much it counts, and whether it counts for or against.
3. The perceptron multiplies each input by its weight and adds everything up. It also adds a **bias**, a fixed nudge that makes "yes" easier or harder.
4. If the total is above zero, it says 1 (yes). Otherwise it says 0 (no). This hard switch is a **step** [[Activation Function]].
5. **Learning:** show it an example with the right answer. If it is right, change nothing. If it is wrong, move the weights a little toward the right answer: add the input to the weights when it should have said yes, and subtract when it should have said no.
6. Repeat over all the examples until it makes no more mistakes.
7. **The catch:** one perceptron can only split things with a **straight line** (or a flat plane in more dimensions). Some questions, such as XOR ("one or the other, but not both"), cannot be cut with a single line. Stacking perceptrons in layers fixes that, and that is what a [[Neural Network]] is.

![[perceptron-unit.svg]]
*A perceptron as a picture: the inputs $x_0,\dots,x_n$ on the left (with $x_0$ fixed at 1, which carries the bias), each multiplied by its weight $w$ along the arrows, then added up ($\Sigma$) and passed through the switch ($\varphi$) in the green circle to give the output $o$. Image: MartinThoma, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Perceptron-unit.svg), CC0. I read the roles from the picture and its labels, not from the file text.*

## 🟠 3. Technical detail
- **Model:** a linear score $w\cdot x+b$ followed by a step. Geometrically, the weights define a line (or plane) and the perceptron says which side of it a point is on.
- **Learning rule:** for each example, compute the error $y-\hat y$, which is 0, +1 or -1, and move $w$ by (learning rate) × (error) × (input). Nothing happens when the answer is right.
- **Convergence:** if a line that separates the two groups exists, the rule is guaranteed to find one after a finite number of corrections (the perceptron convergence theorem).
- **When no line exists:** the rule never settles. It keeps correcting, and the mistakes per pass bounce around.
- **XOR and the fix:** XOR has no separating line, but two layers of perceptrons solve it: one unit says "at least one is on" (OR), another says "not both are on" (NAND), and a third says "both of those are true" (AND). Training such stacks needs [[Backpropagation]], and a step function has no useful slope, so later networks use smooth activations.
- **Relatives:** [[Logistic Regression]] is the same linear score with a smooth probability instead of a step. The [[Support Vector Machine]] also draws a straight line, but chooses the one with the widest gap. The learning rule is a simple cousin of [[Gradient Descent]].

```python
import numpy as np
step = lambda z: (z > 0).astype(int)
def train(X, y, epochs=100, eta=1.0):
    w, b, hist = np.zeros(X.shape[1]), 0.0, []
    for _ in range(epochs):
        mistakes = 0
        for xi, yi in zip(X, y):
            err = yi - int(xi @ w + b > 0)                # 0 if right, +1 or -1 if wrong
            if err: w = w + eta * err * xi; b += eta * err; mistakes += 1
        hist.append(mistakes)
        if mistakes == 0: break
    return w, b, hist
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for name, y in (("AND", [0, 0, 0, 1]), ("OR", [0, 1, 1, 1]), ("XOR", [0, 1, 1, 0])):
    w, b, hist = train(X, np.array(y))
    acc = (step(X @ w + b) == y).mean()
    print(f"{name:3s}: mistakes per pass {hist[:6]}{'...' if len(hist) > 6 else ''}  passes used {len(hist)}  final accuracy {acc:.2f}")
# AND: mistakes per pass [1, 3, 3, 2, 1, 0]  passes used 6  final accuracy 1.00
# OR : mistakes per pass [1, 2, 1, 0]  passes used 4  final accuracy 1.00
# XOR: mistakes per pass [2, 3, 4, 4, 4, 4]...  passes used 100  final accuracy 0.50
```

## 🔴 4. Deep / formal
**The decision.** With inputs $x$, weights $w$ and a bias $b$:

$$
\hat y=\mathbf 1\!\left[\,w\cdot x+b>0\,\right]
$$

**Where:**
- $x$ is the list of input numbers and $w$ the list of weights, one per input. $w\cdot x$ is the **dot product**: multiply each pair and add ([[Vector]]).
- $b$ is the bias. **Matters:** without it the dividing line must pass through the origin, so many easy problems become impossible.
- $\mathbf 1[\cdot]$ is 1 if the statement inside is true and 0 otherwise (the step). $\hat y$ is the prediction. **Matters:** the step makes the output a yes/no, but it has no slope, so gradient-based training does not work on it directly.

**Example (tiny numbers):** $w=(1,-1)$, $b=0$, input $x=(2,1)$.
- $w\cdot x+b=1\cdot2+(-1)\cdot1+0=\mathbf{1}$.
- $1>0$, so $\hat y=\mathbf 1$.
- Meaning: the point $(2,1)$ is on the "yes" side of the line $x_1=x_2$.

**The learning rule.** After seeing an example with true answer $y$ (0 or 1):

$$
w\leftarrow w+\eta\,(y-\hat y)\,x,\qquad b\leftarrow b+\eta\,(y-\hat y)
$$

**Where:**
- $\eta$ is the learning rate, the size of the correction. **Matters:** for a perceptron started at zero it only scales all the weights together, so it does not change which mistakes occur (it matters when you start from non-zero weights).
- $y-\hat y$ is the error: 0 when right, +1 when it should have said yes, -1 when it should have said no. **Matters:** when it is 0 nothing changes, so the rule only reacts to mistakes.
- $x$ is the input that was misjudged. **Matters:** inputs that were large get larger corrections, so the change points the line toward fixing this very example.

**Example (tiny numbers):** $w=(0,0)$, $b=0$, $\eta=1$, input $x=(1,2)$, true answer $y=1$.
- Score $=0$, which is not above 0, so $\hat y=0$ and the error is $1-0=\mathbf1$.
- New weights: $w=(0,0)+1\cdot(1,2)=\mathbf{(1,2)}$, new bias $b=0+1=\mathbf1$.
- Check: the score is now $1\cdot1+2\cdot2+1=\mathbf6>0$, so it now says yes.
- Meaning: one correction was enough to fix this example.

**How many corrections at most (Novikoff, 1962; from memory).** Suppose some line separates the data with a gap. With labels written as $\pm1$, and the bias counted as an extra input fixed at 1:

$$
\text{number of corrections}\;\le\;\left(\frac{R}{\gamma}\right)^{2}
$$

**Where:**
- $R$ is the length of the longest input vector (counting the extra bias input). **Matters:** bigger inputs allow more corrections.
- $\gamma$ is the **margin**: how far the closest point is from a separating line (for a line with weights of length 1). **Matters:** the narrower the gap, the more corrections may be needed, and with no gap ($\gamma=0$) the bound is useless.
- The bound does not depend on the number of examples or on the number of inputs.

**Example (tiny numbers):** $R=2$ and $\gamma=1$ give $(2/1)^2=\mathbf4$ corrections at most. In the code below (164 points), $R=1.61$ and $\gamma=0.145$ give $\mathbf{122.4}$, and the perceptron really needed $\mathbf6$. So the bound holds, but it is loose.

**Further notes:**
- **Limits:** the perceptron cannot separate classes that need a curved boundary, and it gives no probability, only a yes or a no.
- **Voted and averaged perceptrons** (from memory) average the weights seen during training, which makes the result steadier on noisy data.
- **Mark I Perceptron:** Rosenblatt's group built a machine for image recognition around 1957 to 1958 (from memory, not checked).
- **Why the 1969 book mattered:** Minsky and Papert showed that a single layer cannot do XOR, and some say that fed the later "AI winter" (see [[History of AI]]).

**What came before, and how this improved on it:**
- **Before:** the McCulloch and Pitts neuron (1943, checked on the Wikipedia timeline page): a switch that adds inputs and fires, but with weights that people set by hand.
- **Its limit:** nothing in it could learn. Every new problem meant working out the weights yourself.
- **What changed:** Rosenblatt's perceptron (1958; from memory) adds the learning rule above, so the weights come from examples. Stacking layers later removed the straight-line limit (see [[Neural Network]] and [[Backpropagation]]).
- **How much better:** in the code below, on 164 points with a clear gap, random weights score an average accuracy of 0.506 and the learned perceptron scores 1.000 after 6 corrections in total. **Honest catch:** one perceptron on XOR, even with the best of 68,921 weight settings tried, gets 0.75 at most, while two layers of perceptrons (weights set by hand) get 1.0.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
step = lambda z: (z > 0).astype(int)
X = rng.uniform(-1, 1, (200, 2)); X = X[np.abs(X @ [1, -1]) > 0.2]                # two groups with a clear gap between them
y = (X @ [1, -1] > 0).astype(int)
acc_random = np.mean([(step(X @ rng.standard_normal(2) + rng.standard_normal()) == y).mean() for _ in range(1000)])
w, b, mistakes = np.zeros(2), 0.0, 0
for _ in range(100):
    m = 0
    for xi, yi in zip(X, y):
        err = yi - int(xi @ w + b > 0)
        if err: w = w + err * xi; b += err; m += 1
    mistakes += m
    if m == 0: break
print(f"{len(X)} points with a gap: random weights average accuracy {acc_random:.3f}; after learning {(step(X @ w + b) == y).mean():.3f} with {mistakes} corrections in total")
Y = np.array([0, 1, 1, 0]); P = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
best = max((step(P @ np.array([a, c]) + d) == Y).mean() for a in np.linspace(-2, 2, 41) for c in np.linspace(-2, 2, 41) for d in np.linspace(-2, 2, 41))
print("XOR with ONE perceptron, best of 68,921 weight settings tried: accuracy", best)
or_ = lambda p: step(p @ [1, 1] - 0.5); nand = lambda p: step(-(p @ [1, 1]) + 1.5); and_ = lambda h: step(h @ [1, 1] - 1.5)
H = np.stack([or_(P), nand(P)], 1)
print("XOR with TWO layers of perceptrons (weights set by hand: OR, NAND, then AND):", and_(H), "accuracy", (and_(H) == Y).mean())
ws = np.array([1, -1]) / np.sqrt(2); gamma = np.abs(X @ ws).min(); R = np.sqrt((X ** 2).sum(1).max() + 1)       # +1: the bias counts as an extra input fixed at 1
print(f"Novikoff bound (R / margin)^2 = ({R:.2f} / {gamma:.3f})^2 = {(R / gamma) ** 2:.1f} corrections at most; it needed {mistakes}")
```

## Related
- [[Neural Network]] · [[Backpropagation]] · [[Activation Function]] · [[Logistic Regression]] · [[Support Vector Machine]] · [[Gradient Descent]] · [[Classification]] · [[Supervised Learning]] · [[History of AI]] · [[From FNN to RNN]] · [[Decision Boundary]]

## Sources
- McCulloch and Pitts (1943), Minsky and Papert, *Perceptrons* (1969): dates checked on the Wikipedia *Timeline of artificial intelligence* page.
- Rosenblatt (1958), the perceptron and its learning rule, and Novikoff (1962), the convergence bound: from memory, not opened.
- Image: [Perceptron-unit.svg](https://commons.wikimedia.org/wiki/File:Perceptron-unit.svg) by MartinThoma, CC0.
- All code results were run for this note (NumPy only, toy data).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/ml-foundations Perceptron:::A little judge. It multiplies each input by a weight, adds them up, and says yes if the total is big enough. When wrong, it changes its weights a little.

#flashcards/ml/formula #flashcards/topic/ml-foundations Write the perceptron prediction and explain each symbol.
?
$$\hat y=\mathbf 1\!\left[\,w\cdot x+b>0\,\right]$$
- $x$: the list of input numbers. $w$: the weights, one per input. $w\cdot x$: the dot product (multiply each pair and add).
- $b$: the bias. Without it the dividing line must pass through the origin, so many easy problems become impossible.
- $\mathbf 1[\cdot]$: 1 if the statement inside is true, else 0 (the step). $\hat y$: the prediction.
- The step gives a yes/no, but it has no slope, so gradient-based training does not work on it directly.

#flashcards/ml/example #flashcards/topic/ml-foundations $w=(1,-1)$, $b=0$, input $x=(2,1)$. What does the perceptron say?::$1\cdot2+(-1)\cdot1+0=1>0$, so $\hat y=1$. The point is on the "yes" side of the line $x_1=x_2$.

#flashcards/ml/formula #flashcards/topic/ml-foundations Write the perceptron learning rule and explain each symbol.
?
$$w\leftarrow w+\eta\,(y-\hat y)\,x,\qquad b\leftarrow b+\eta\,(y-\hat y)$$
- $\eta$: the learning rate, the size of the correction. For a perceptron started at zero it only scales the weights together.
- $y-\hat y$: the error. 0 when right, +1 when it should have said yes, −1 when it should have said no. At 0 nothing changes, so the rule reacts only to mistakes.
- $x$: the misjudged input. Large inputs get larger corrections, so the change points the line toward fixing this example.

#flashcards/ml/example #flashcards/topic/ml-foundations $w=(0,0)$, $b=0$, $\eta=1$, input $x=(1,2)$, true answer $y=1$. What is the update?::Score 0 is not above 0, so $\hat y=0$ and the error is 1. New $w=(1,2)$, $b=1$. The score is now $1+4+1=6>0$, so it says yes. One correction fixed it.

#flashcards/ml/formula #flashcards/topic/ml-foundations Write the perceptron mistake bound and explain each symbol.
?
$$\text{number of corrections}\;\le\;\left(\frac{R}{\gamma}\right)^{2}$$
- $R$: the length of the longest input vector (counting the extra bias input). Bigger inputs allow more corrections.
- $\gamma$: the margin, how far the closest point is from a separating line (weights of length 1). The narrower the gap, the more corrections may be needed. With $\gamma=0$ the bound is useless.
- The bound does not depend on the number of examples or inputs.

#flashcards/ml/example #flashcards/topic/ml-foundations $R=2$, $\gamma=1$. What is the bound? And in the note's code?::$(2/1)^2=4$ corrections at most. In the note's code (164 points) $R=1.61$, $\gamma=0.145$ give 122.4, and the perceptron really needed 6. The bound holds but is loose.

#flashcards/ml/image #flashcards/topic/ml-foundations ![[perceptron-unit.svg]]
What does this picture show?
?
A perceptron as a picture: inputs $x_0,\dots,x_n$ on the left (with $x_0$ fixed at 1, which carries the bias), each multiplied by its weight $w$ along the arrows, then added up ($\Sigma$) and passed through the switch ($\varphi$) in the green circle to give the output $o$. (From the note's caption: MartinThoma, Wikimedia Commons.)

#flashcards/ml/cloze #flashcards/topic/ml-foundations The perceptron rule changes the weights only on a ==mistake==.
