---
tags: [neural-network, activation-function, ReLU, deep-learning, training, gradient]
status: growing
level: 4
tier: 1
---

# ReLU

The simplest "bend" in a neural network: **if a number is negative, make it 0; otherwise leave it alone.** ReLU stands for *rectified linear unit*. It is the activation function ([[Activation Function]]) behind most deep networks, including the first [[Transformer]].

## 🟢 1. Simple (age 12)
Imagine a door that only opens one way. Anything coming from the positive side walks straight through unchanged. Anything coming from the negative side is stopped and becomes zero. That is all ReLU does to a number.

**Example:** A bank account shows only your *savings*. If your balance is +5 it shows 5. If it is -3 it shows 0 (you have no savings). The display never shows negative numbers.

## 🔵 2. How it works
1. Each little worker in a [[Neural Network]] (a "unit") adds up its inputs into one number, $z$. ReLU then turns $z$ into $\max(0,z)$: negatives become 0, positives stay.
2. Why not skip this step? Without a bend, many layers of maths collapse into one layer ([[Feed-Forward Network]] shows this with numbers). The bend is what lets a network draw curved shapes. ReLU's bend is a sharp corner at 0, and that is enough.
3. **Why it trains well.** Learning sends a correction signal backwards through the layers ([[Backpropagation]]). Each layer multiplies the signal by the activation's *slope*. A sigmoid's slope is never more than 0.25, so after 10 layers the signal is cut to at most $0.25^{10}$ of its size, about one in a million. ReLU's slope is exactly 1 for positive inputs, so the signal passes through undamaged ([[Vanishing and Exploding Gradients]]).
4. **It is cheap.** It is one comparison, with no exponentials.
5. **It switches units off.** About half the units output exactly 0 for a typical input, so each input uses only part of the network. This is called sparsity.
6. **The danger: a "dead" unit.** If a unit's input is negative for *every* example, its output is 0 and its slope is 0, so it gets no correction and can never recover. Small fixes: a leaky version that lets a little through for negatives, and careful starting values.
7. The code in level 3 measures steps 3, 5 and 6.

## 🟠 3. Technical detail
- **Definition.** $\mathrm{ReLU}(z)=\max(0,z)$. Slope 1 for $z>0$, 0 for $z<0$. At exactly 0 the slope is not defined, and software just picks 0 or 1.
- **Shape of the whole network.** A network of ReLUs is made of straight pieces: between bends it is just a linear function. With $n$ ReLU units on one input there are at most $n+1$ pieces. Deeper networks can make many more pieces than shallow ones of the same size (a standard result I did not check here).
- **Starting values matter.** ReLU zeroes about half its inputs, so the usual weight scale is larger by a factor of $\sqrt2$ than for sigmoid, to keep signal sizes stable ("He initialisation", from memory). The code uses this pairing: a scale of $\sqrt{2/\text{width}}$ for ReLU and $\sqrt{1/\text{width}}$ for sigmoid.
- **Variants.** Leaky ReLU keeps a small slope for negatives. Smoother relatives such as GELU and SiLU are common in modern [[LLM]]s ([[Activation Function]]). Which is best differs by model and is not tested here.
- **The code below is a toy.** Random networks of width 64, never trained. It measures how big the learning signal is at the input after 2 to 20 layers, how many units are exactly zero, what a dead unit looks like, and how many straight pieces a small ReLU network has.

```python
import numpy as np

rng = np.random.RandomState(0)
width, L = 64, 10


def input_gradient(act, dact, scale, depth):
    """Random MLP of `depth` layers. Return the size of d(output)/d(input), averaged over 200 inputs."""
    Ws = [rng.randn(width, width) * scale for _ in range(depth)]
    sizes = []
    for _ in range(200):
        h, zs = rng.randn(width), []
        for W in Ws:
            z = W @ h
            zs.append(z)
            h = act(z)
        g = np.ones(width)                      # pretend the loss is the sum of the outputs
        for W, z in zip(reversed(Ws), reversed(zs)):
            g = W.T @ (g * dact(z))
        sizes.append(np.linalg.norm(g))
    return float(np.mean(sizes))


sig = lambda z: 1 / (1 + np.exp(-z))
dsig = lambda z: sig(z) * (1 - sig(z))
relu = lambda z: np.maximum(0, z)
drelu = lambda z: (z > 0).astype(float)

print("largest slope of sigmoid:", dsig(np.array([0.0]))[0], "| of ReLU:", 1.0)
print(f"0.25 ** {L} = {0.25 ** L:.2e}")
for depth in (2, 5, 10, 20):
    gs = input_gradient(sig, dsig, np.sqrt(1 / width), depth)          # Xavier-style scale for sigmoid
    gr = input_gradient(relu, drelu, np.sqrt(2 / width), depth)        # He-style scale for ReLU
    print(f"depth {depth:2d}: gradient size at the input  sigmoid {gs:.2e}   ReLU {gr:.2e}")

h = relu(rng.randn(100000))
print("share of ReLU outputs that are exactly 0 for random inputs:", round(float((h == 0).mean()), 3))

# a dead unit: its input is negative for every example, so its output and its gradient are always 0
X = rng.randn(1000, 2)
z = X @ np.array([1.0, 1.0]) - 20.0
print("dead unit: share of examples with z > 0:", float((z > 0).mean()), "| mean gradient:", float(drelu(z).mean()),
      "| leaky ReLU gradient (alpha 0.1):", float(np.where(z > 0, 1.0, 0.1).mean()))

# pieces: a 1-D net with n ReLU units has at most n + 1 straight pieces
xs = np.linspace(-6, 6, 20001)
for n in (1, 3, 10):
    a, b, c = rng.randn(n), rng.randn(n), rng.randn(n)
    y = sum(c[k] * relu(a[k] * xs + b[k]) for k in range(n))
    bends = np.abs(np.diff(y, 2)) > 1e-9                       # second difference is non-zero only at a bend
    clusters = int(bends[0]) + int((bends[1:] & ~bends[:-1]).sum())   # a bend can touch 2 grid points: count groups
    print(f"{n:2d} ReLU units -> {1 + clusters} straight pieces on -6..6 (at most {n + 1})")
```

What it printed: the signal at the input after 10 layers was 3.57e-06 for sigmoid and 14.5 for ReLU (about 4 million times bigger); after 20 layers it was 3.09e-12 against 33.3. For random inputs, 0.497 of ReLU outputs were exactly 0. A unit with input always below 0 had 0 examples above zero and a mean gradient of 0.0, while the leaky version (slope 0.1) still had gradient 0.1. A 1-input network with 1, 3 and 10 ReLU units showed 2, 3 and 9 straight pieces on the range -6 to 6, never more than $n+1$ (some bends fall outside the range).

## 🔴 4. Deep / formal
The function and its slope:

$$
\mathrm{ReLU}(z)=\max(0,z),\qquad \mathrm{ReLU}'(z)=\begin{cases}1 & z>0\\ 0 & z<0\end{cases}
$$

**Where:**
- $z$ is the unit's input: the weighted sum of what came from the layer before.
- $\mathrm{ReLU}(z)$ is the unit's output. It equals $z$ for positive input and 0 otherwise.
- $\mathrm{ReLU}'(z)$ is the slope: how much the output changes if $z$ changes a little. At $z=0$ it is not defined.
- **Matters:** a slope of exactly 1 passes a learning signal through unchanged. A slope of exactly 0 passes nothing, so a unit with $z<0$ for all examples learns nothing (dead unit).

**Example (tiny numbers):** $z=3$ and $z=-2$.
- $z=3$: output $\mathbf{3}$, slope $\mathbf{1}$.
- $z=-2$: output $\mathbf{0}$, slope $\mathbf{0}$.
- Meaning: the first number passes through and lets learning happen. The second is silenced, and nothing in its path gets a correction from this example.

Through $L$ layers, the learning signal is multiplied by one slope per layer. In the simplest case of one unit per layer with weight $w$ in each layer:

$$
\frac{\partial\,\mathrm{loss}}{\partial\,\mathrm{input}}=\frac{\partial\,\mathrm{loss}}{\partial\,\mathrm{output}}\prod_{l=1}^{L}w\,f'(z_l)
$$

**Where:**
- $L$ is the number of layers and $f'(z_l)$ the activation's slope in layer $l$.
- $w$ is the weight between layers (taken equal to 1 below, to isolate the effect of the slope).
- **Matters:** the product has one factor per layer. If each factor is below 1 (sigmoid: at most 0.25) the signal shrinks exponentially with depth. If each is 1 (ReLU, for an active unit) it does not shrink. With factors above 1 it would blow up instead.

**Example (tiny numbers):** $L=10$ layers, $w=1$.
- Sigmoid, best case, slope 0.25 everywhere: $0.25^{10}=\mathbf{9.5\times10^{-7}}$.
- ReLU, every unit active, slope 1: $1^{10}=\mathbf{1}$.
- Meaning: after 10 layers, the sigmoid network gives the first layer at most about one-millionth of the correction, so it learns extremely slowly. The ReLU network gives it all of it, as long as the units on the path are active.

**Further notes (from the code):**
- **The real networks matched the arithmetic in order of size.** At depth 10 the measured signal was 3.57e-06 for sigmoid, against the $9.5\times10^{-7}$ best case of the one-unit example (the real networks have 64 units per layer and random weights, so the match is loose). For ReLU it was 14.5. At depth 20 the sigmoid signal was 3.09e-12.
- **Not a pure comparison.** The two networks used different usual starting scales (one for each activation), so the 4-million-fold gap mixes the slope effect with that choice. I did not run a test that changes only the activation.
- **Sparsity and dying:** 0.497 of outputs exactly 0 for random inputs; a dead unit stayed at gradient 0.0 on all 1000 examples, the leaky version at 0.1. How many units die in a real trained network was not measured.
- **What the toy cannot show:** accuracy, training speed on real data, or whether any of this makes real networks better. No network here is trained.

**What came before, and how this improved on it:**
- **Before:** the logistic sigmoid and tanh as the standard activations. Glorot, Bordes and Bengio (AISTATS 2011; abstract read) note that tanh trains multi-layer networks better than the sigmoid.
- **Its limit:** the slope of a sigmoid is at most 0.25 (code: 0.25), so the learning signal shrinks with depth: 0.25 to the power 10 is $9.5\times10^{-7}$.
- **What changed:** use the rectifier. Glorot et al. (2011) report that rectifier networks match or beat tanh networks, give sparse outputs with true zeros, and that deep rectifier networks reach their best performance on purely supervised tasks with large labelled sets without unsupervised pre-training (abstract read). The rectifier's use in earlier work on restricted Boltzmann machines (Nair and Hinton, 2010) and in [[AlexNet 2012]] is **from memory** apart from what that note records.
- **How much better:** in this note's random-network code, the learning signal at the input after 10 layers was 14.5 for ReLU against 3.57e-06 for sigmoid, about 4 million times larger (with the usual scale for each, and nothing trained). I did not find a published figure with ReLU as the only change, so I give none.

## Related
- [[Activation Function]] · [[Neural Network]] · [[Feed-Forward Network]] · [[Backpropagation]] · [[Vanishing and Exploding Gradients]] · [[Gradient Descent]] · [[Transformer]] · [[AlexNet 2012]] · [[Perceptron]] · [[LLM]] · [[Matrix]]

## Sources
- Glorot, Bordes, Bengio, *Deep Sparse Rectifier Neural Networks*, AISTATS 2011 (PMLR 15:315-323). **Abstract only.** Its page does not mention vanishing gradients, so that link comes from the code and [[Vanishing and Exploding Gradients]], not from this source.
- Nair and Hinton (2010), He initialisation and the standard results about the number of pieces are **from memory**, not checked here.
- Image: none. [[Activation Function]] already holds a plot of ReLU and its slope.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/neural-nets ReLU:::A one-way door. A positive number passes through unchanged, and a negative number is stopped and becomes zero.

#flashcards/ml/formula #flashcards/topic/neural-nets Write ReLU and its slope and explain each symbol.
?
$$\mathrm{ReLU}(z)=\max(0,z),\qquad \mathrm{ReLU}'(z)=\begin{cases}1 & z>0\\ 0 & z<0\end{cases}$$
- $z$: the unit's input, the weighted sum of what came from the layer before.
- $\mathrm{ReLU}(z)$: the unit's output. It equals $z$ for positive input and 0 otherwise.
- $\mathrm{ReLU}'(z)$: the slope, how much the output changes if $z$ changes a little. At $z=0$ it is not defined.
- A slope of exactly 1 passes a learning signal through unchanged. A slope of exactly 0 passes nothing, so a unit with $z<0$ for all examples learns nothing (a dead unit).

#flashcards/ml/example #flashcards/topic/neural-nets What are the output and slope for $z=3$ and $z=-2$?::$z=3$: output 3, slope 1. $z=-2$: output 0, slope 0. The first passes through and lets learning happen. The second is silenced and nothing in its path gets a correction from this example.

#flashcards/ml/formula #flashcards/topic/neural-nets Write how the learning signal reaches the first layer through $L$ layers and explain each symbol.
?
$$\frac{\partial\,\mathrm{loss}}{\partial\,\mathrm{input}}=\frac{\partial\,\mathrm{loss}}{\partial\,\mathrm{output}}\prod_{l=1}^{L}w\,f'(z_l)$$
- $L$: the number of layers. $f'(z_l)$: the activation's slope in layer $l$.
- $w$: the weight between layers (taken as 1 in the example to isolate the effect of the slope).
- The product has one factor per layer. If each is below 1 (sigmoid: at most 0.25) the signal shrinks exponentially with depth. If each is 1 (ReLU, active unit) it does not shrink. Factors above 1 would blow it up instead.

#flashcards/ml/example #flashcards/topic/neural-nets $L=10$, $w=1$. Compare the signal through sigmoid (best case slope 0.25) and ReLU (all active).::Sigmoid: $0.25^{10}=9.5\times10^{-7}$. ReLU: $1^{10}=1$. After 10 layers the sigmoid network gives the first layer at most about one-millionth of the correction, while ReLU gives it all, as long as the units on the path are active.

#flashcards/ml/cloze #flashcards/topic/neural-nets A ReLU unit with $z<0$ for all examples is a ==dead unit==: it learns nothing.
