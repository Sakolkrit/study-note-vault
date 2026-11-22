---
tags: [machine-learning, deep-learning, optimization, backpropagation, initialization]
status: growing
level: 4
tier: 2
---

# Vanishing and Exploding Gradients

## 🟢 1. Simple (age 12)
When a deep network learns, a message travels **backwards** through all its layers saying "here is how to fix each part". At every layer that message gets **multiplied** by a number. If those numbers are mostly **smaller than 1**, the message gets weaker and weaker until the early layers hear almost nothing: it **vanishes**. If they are mostly **bigger than 1**, the message gets louder and louder until it is a deafening roar: it **explodes**. Both make learning fail.

**Example:** A game of "telephone" in a line of 50 kids. If each kid whispers a bit quieter, the last kid hears nothing. If each kid shouts a bit louder, the message turns into screaming noise. You want every kid to pass it along at about the **same volume**.

## 🔵 2. How it works
1. Learning needs a **gradient** for every weight: how much the loss ([[Loss Function]]) changes if that weight changes. [[Backpropagation]] computes them from the last layer back to the first.
2. At each layer, backpropagation **multiplies** the incoming gradient by that layer's slope and weight (the chain rule, see [[Calculus]]).
3. In a deep network there are many such multiplications in a row. Many numbers below 1 multiplied together shrink towards 0. Many numbers above 1 multiplied together grow without limit.
4. **Vanishing:** the early layers get a gradient near zero, so their weights barely change. They learn very slowly, or not at all.
5. **Exploding:** the gradients become enormous, a step jumps far away, and the loss shoots up or turns into `nan` (not a number).
6. The usual cures keep the multiplications close to 1:
   - **Better starting weights** (Xavier or He initialisation) so signals stay a sensible size from the start.
   - **Better activation functions** such as ReLU, whose slope is 1 instead of at most 0.25 (see the picture).
   - **[[Normalization]]** inside the network to keep the numbers in range.
   - **Gradient clipping:** if the gradient is too big, scale it down before the step.
   - **Skip connections** (used in a [[Transformer]]) that give the gradient a short route around layers, and a sensible [[Learning Rate]].

![[activation-derivatives.svg]]
*Four activation functions (red, left scale) and their slopes, called derivatives (orange dashed, right scale). Top, sigmoid: the slope peaks at 0.25 in the middle and falls towards 0 on both sides, so every sigmoid layer multiplies the backward signal by at most 0.25. Second, ReLU: the slope is exactly 0 or 1. The other two are variants (leaky ReLU, SiLU). The captions (French, "Sigmoide") name the curves, and I read the numbers off the axes. Image: CapeItalist, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Fonctions_d%27activation_et_leur_d%C3%A9riv%C3%A9e.svg), CC0.*

## 🟠 3. Technical detail
- **Where it bites:** very deep networks, and recurrent networks that repeat the same step many times (a long sequence is like a very deep network).
- **Sigmoid and tanh:** their slope is small whenever the input is large in either direction ("saturation"), so these are prone to vanishing. Sigmoid's slope is at most $0.25$.
- **ReLU:** slope 1 for positive inputs, so it passes the gradient through unchanged. Its problem is a different one: a neuron stuck at negative inputs has slope 0 forever ("dying ReLU"). Leaky ReLU keeps a small slope there.
- **Weight initialisation:** pick the random starting weights with the right spread so that, layer after layer, the signal going forward and the gradient going backward keep about the same size.
  - **Xavier (Glorot)** is designed for sigmoid and tanh.
  - **He (Kaiming)** is designed for ReLU, which zeroes about half the signal, so it uses a larger spread.
- **Detecting it:** log the size (norm) of the gradient for each layer. A norm that falls by a factor of 10 every few layers going backwards means vanishing. A sudden spike, or a loss that turns to `nan`, means exploding.
- **Gradient clipping:** a safety net against rare huge gradients. It limits the size of a step without changing its direction. Common when training recurrent networks and [[LLM]]s.
- **Residual connections and [[Normalization]]** are a large part of why very deep [[Transformer]]s can be trained at all.
- **Interaction with the optimiser:** [[Adam Optimizer]] and [[RMSProp]] rescale each weight's step by its typical gradient size, which softens (but does not remove) vanishing and exploding.
- **Mixed precision:** very small gradients can also underflow to 0 in low-precision number formats, which looks just like vanishing (see [[GPU]]; from memory).

```python
import numpy as np
print([0.25 ** k for k in (1, 5, 10)])      # [0.25, 0.0009765625, 9.5367431640625e-07]  sigmoid layers: shrinking
print(round(1.5 ** 10, 1), round(0.9 ** 10, 3))   # 57.7 0.349     factor 1.5 explodes, 0.9 fades

# how the starting spread of the weights changes the signal after 20 ReLU layers of 100 neurons
rng = np.random.default_rng(0)
n = 100; x0 = rng.normal(size=n)                       # an input signal of typical size about 1
for name, std in (("small", 0.01), ("He", np.sqrt(2 / n)), ("big", 0.2)):
    h = x0.copy()
    for _ in range(20):
        h = np.maximum(0, rng.normal(0, std, (n, n)) @ h)
    print(name, round(float(std), 4), float(np.sqrt((h ** 2).mean())))
# small 0.01 1.2091571158618413e-23   vanished
# He 0.1414 0.6929855334715669        still about the size of the input
# big 0.2 766.8794472662901           exploded

g = np.array([3., 4.]); c = 1.0                        # clip the gradient to a maximum length c
print(np.linalg.norm(g), g * min(1, c / np.linalg.norm(g)))   # 5.0 [0.6 0.8]
```

## 🔴 4. Deep / formal
**The product that causes it.** Take the simplest deep network: one neuron per layer, $L$ layers, where layer $l$ has weight $w_l$ and activation slope $\sigma'(z_l)$. By the chain rule, the gradient reaching the first layer is the gradient at the end times one factor per layer:

$$
\frac{\partial \mathcal L}{\partial a_1}=\frac{\partial \mathcal L}{\partial a_L}\ \prod_{l=2}^{L}\sigma'(z_l)\,w_l
$$

**Where:**
- $\mathcal L$ is the loss and $a_l$ is the output of layer $l$. So $\partial\mathcal L/\partial a_1$ is the gradient reaching layer 1 and $\partial\mathcal L/\partial a_L$ is the gradient at the output end.
- $z_l$ is the input to layer $l$'s activation function, $\sigma'(z_l)$ is that function's slope at $z_l$, and $w_l$ is the layer's weight.
- $\prod_{l=2}^{L}$ means "multiply these factors together, one for each layer from 2 to $L$".
- **Matters:** each layer contributes the factor $f_l=\sigma'(z_l)\,w_l$. If the factors are typically below 1 in size, the product shrinks **exponentially** with depth (vanishing). If they are typically above 1, it grows exponentially (exploding). Only factors close to 1 keep the gradient usable.
- In real networks the layers are wide, so the factors are matrices, and what matters is the typical size by which each matrix stretches or shrinks the gradient. The one-neuron picture is a simplification that keeps the main idea.

**Example (tiny numbers):** 10 layers, each factor the same.
- Sigmoid with $w=1$: the slope is at most $0.25$, so $f=0.25$ and $0.25^{10}=\mathbf{9.5\times10^{-7}}$. The first layer gets about a millionth of the gradient.
- A factor of $f=1.5$ (for example big weights): $1.5^{10}=\mathbf{57.7}$. The gradient grows nearly sixty times.
- A factor of $f=1$: $1^{10}=\mathbf{1}$. The gradient arrives unchanged, which is what good initialisation and ReLU aim for.
- Meaning: because the layers multiply, a small difference per layer (0.9 against 1.1) becomes a large difference after 10 layers ($0.35$ against $2.6$).

**Choosing the starting weights.** To keep the signal at about the same size layer after layer, the starting weights are drawn at random with a variance set by the layer sizes:

$$
\operatorname{Var}(w)=\frac{2}{n_{\text{in}}+n_{\text{out}}}\ \ (\text{Xavier}),\qquad \operatorname{Var}(w)=\frac{2}{n_{\text{in}}}\ \ (\text{He})
$$

**Where:**
- $w$ is one starting weight, drawn at random with average 0 (for example from a [[Normal Distribution]]). $\operatorname{Var}(w)$ is its variance (spread squared).
- $n_{\text{in}}$ is how many inputs the layer has, and $n_{\text{out}}$ how many outputs.
- **Matters:** each neuron adds up $n_{\text{in}}$ terms, so the sum's spread grows with $n_{\text{in}}$. Making each weight's variance proportional to $1/n_{\text{in}}$ cancels that growth and keeps the layer's output about as spread out as its input. Xavier balances forward and backward signals (the average of $n_{\text{in}}$ and $n_{\text{out}}$ shows up through the sum). He uses the larger $2/n_{\text{in}}$ because ReLU throws away about half of the signal, so the weights must compensate.
- **Matters:** without this scaling, the demo in the code above shows the signal vanishing (spread $0.01$) or exploding (spread $0.2$) over 20 layers.
- These formulas come from a short derivation that assumes simple, independent signals. Treat them as sensible defaults, not guarantees (from memory; the papers give the derivation).

**Example (tiny numbers):** a layer with $n_{\text{in}}=n_{\text{out}}=100$.
- Xavier: $\operatorname{Var}=2/200=\mathbf{0.01}$, so the standard deviation is $\sqrt{0.01}=\mathbf{0.1}$.
- He: $\operatorname{Var}=2/100=\mathbf{0.02}$, standard deviation $\sqrt{0.02}=\mathbf{0.141}$.
- Meaning: in the code, the He value ($0.1414$) kept the signal around $0.69$ after 20 layers, while $0.01$ shrank it to $10^{-23}$ and $0.2$ blew it up to about $767$.

**Gradient clipping.** If the gradient $g$ (all weights together, as a [[Vector]]) is longer than a limit $c$, scale it down to length $c$:

$$
g\leftarrow g\cdot\min\!\Big(1,\ \frac{c}{\lVert g\rVert}\Big)
$$

**Where:**
- $g$ is the gradient vector, and $\lVert g\rVert$ is its length (norm): the square root of the sum of the squares of its entries.
- $c$ is the **clipping threshold** you choose, the longest gradient you will allow.
- $\min(1,\,c/\lVert g\rVert)$ is a scale factor: it is $1$ when the gradient is already short enough (nothing changes), and below $1$ when it is too long. **Matters:** it shrinks the step **without changing its direction**, so one freak batch cannot throw the weights far away.

**Example (tiny numbers):** $g=(3,4)$ and $c=1$.
- Length: $\lVert g\rVert=\sqrt{9+16}=\mathbf{5}$. Scale factor: $\min(1,1/5)=\mathbf{0.2}$.
- Clipped: $(3,4)\cdot0.2=\mathbf{(0.6,\,0.8)}$, whose length is exactly $1$.
- A small gradient $g=(0.3,0.4)$ has length $0.5<1$, so the factor is $1$ and it passes through unchanged.
- Meaning: big gradients are tamed to the limit, small ones are left alone.

**Further notes:**
- **Why very deep networks became trainable:** better initialisation, ReLU-style activations, [[Normalization]] and residual connections together removed most of the old obstacle. This is the usual account in the literature (from memory; the history is more tangled than one sentence).
- **Recurrent networks and long sequences** suffer worst, because the same factor is multiplied once per time step. This is one motivation behind [[Attention]], which connects distant positions directly.
- **The signal can also vanish going forward** (activations collapsing to 0), not only the gradient going backward. The initialisation formulas above address both.
- **Saddle points and flat regions** also slow training, but for a different reason: the gradient is small because the loss surface really is flat there, not because it was multiplied down.
- **Not covered yet:** a close look at residual connections and at recurrent networks.

**What came before, and how this improved on it:**
- **Before:** plain deep networks trained with ordinary initial weights.
- **Its limit:** the learning signal is multiplied by a factor at every layer, so it shrinks or grows exponentially. The RNN paper's abstract calls these two problems "widely known" and cites Bengio et al. (1994).
- **What changed:** several fixes: clipping the gradient's size and a soft constraint for RNNs (Pascanu et al., arXiv 1211.5063, 21 November 2012, abstract read), an initial scale that suits rectifiers (He et al., arXiv 1502.01852), and shortcut connections ([[Residual Connection]], arXiv 1512.03385).
- **How much better:** in the code below, 30 layers of tanh give a first-layer gradient of $8.1\times10^{-19}$ with small weights (vanished), and $390$ against $0.048$ at the last layer with large weights (grown 8,060 times towards the front). With shortcuts the sizes are 5.60 at the front and 4.37 at the back. A gradient of size $3.2\times10^{7}$ is clipped to 5.0.

**Check it yourself:**

```python
import torch
torch.manual_seed(0)
def grad_ratio(depth, residual, init):
    w = 32; L = [torch.nn.Linear(w, w) for _ in range(depth)]
    for l in L: torch.nn.init.normal_(l.weight, 0, init / w ** 0.5); torch.nn.init.zeros_(l.bias)
    x = torch.randn(64, w); h = x
    for l in L: h = h + torch.tanh(l(h)) if residual else torch.tanh(l(h))
    h.pow(2).mean().backward()
    return L[0].weight.grad.norm().item(), L[-1].weight.grad.norm().item()
print("30 layers, tanh. gradient size at the first layer versus the last layer:")
for name, res, init in (("plain, small weights", False, 0.5), ("plain, large weights", False, 3.0), ("with shortcuts", True, 0.5)):
    a, b = grad_ratio(30, res, init); print(f"  {name:21s} first {a:.2e}  last {b:.2e}  ratio {a/b:.2e}")
g = torch.randn(1000) * 1e6; clipped = g * min(1.0, 5.0 / g.norm().item())
print("clipping: a gradient of size", f"{g.norm().item():.2e}", "becomes", round(clipped.norm().item(), 3))
```

## Related
- [[Backpropagation]] · [[Neural Network]] · [[Calculus]] · [[Normalization]] · [[Learning Rate]] · [[Adam Optimizer]] · [[RMSProp]] · [[Stochastic Gradient Descent]] · [[Gradient Descent]] · [[Softmax]] · [[Transformer]] · [[Attention]] · [[Loss Function]] · [[Normal Distribution]] · [[Vector]] · [[Activation Function]] · [[Residual Connection]] · [[Recurrent Neural Network]] · [[Eigenvalue]] · [[ResNet]] · [[LSTM]] · [[ReLU]]

## Sources
- Goodfellow, Bengio & Courville, *Deep Learning*, 2016, chapters 6 and 8
- Glorot & Bengio, *Understanding the difficulty of training deep feedforward neural networks*, 2010 (Xavier; from memory)
- He et al., *Delving Deep into Rectifiers*, 2015 (He initialisation; from memory)
- Pascanu, Mikolov & Bengio, *On the difficulty of training recurrent neural networks*, 2013 (gradient clipping; from memory)
- Hochreiter, *Untersuchungen zu dynamischen neuronalen Netzen*, 1991 (early analysis of vanishing gradients; from memory)
- Pascanu et al., arXiv 1211.5063 and He et al., arXiv 1502.01852 (abstracts read for the history block)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/neural-nets Vanishing and exploding gradients:::The backward message in a deep network gets multiplied at every layer. If the factors are mostly below 1 it fades until the early layers hear nothing (vanishes). If they are above 1 it grows out of control (explodes).

#flashcards/ml/formula #flashcards/topic/neural-nets Write the gradient reaching layer 1 and explain each symbol.
?
$$\frac{\partial \mathcal L}{\partial a_1}=\frac{\partial \mathcal L}{\partial a_L}\ \prod_{l=2}^{L}\sigma'(z_l)\,w_l$$
- $\mathcal L$: the loss. $a_l$: the output of layer $l$. So $\partial\mathcal L/\partial a_1$ is the gradient reaching layer 1 and $\partial\mathcal L/\partial a_L$ is the gradient at the output end.
- $z_l$: the input to layer $l$'s activation function. $\sigma'(z_l)$: that function's slope at $z_l$. $w_l$: the layer's weight.
- $\prod$: multiply one factor per layer. With many layers, factors below 1 shrink the product exponentially and factors above 1 grow it exponentially.
- A sigmoid's slope is at most 0.25, so sigmoid layers shrink the signal fast.

#flashcards/ml/formula #flashcards/topic/neural-nets Write the Xavier and He initial variances and explain each symbol.
?
$$\operatorname{Var}(w)=\frac{2}{n_{\text{in}}+n_{\text{out}}}\ \ (\text{Xavier}),\qquad \operatorname{Var}(w)=\frac{2}{n_{\text{in}}}\ \ (\text{He})$$
- $w$: one starting weight, drawn at random with average 0. $\operatorname{Var}(w)$: its variance (spread squared).
- $n_{\text{in}}$: how many inputs the layer has. $n_{\text{out}}$: how many outputs.
- Each neuron adds up $n_{\text{in}}$ terms, so the sum's spread grows with $n_{\text{in}}$. Scaling the weights down by that count keeps the signal steady from layer to layer.

#flashcards/ml/formula #flashcards/topic/neural-nets Write gradient clipping and explain each symbol.
?
$$g\leftarrow g\cdot\min\!\Big(1,\ \frac{c}{\lVert g\rVert}\Big)$$
- $g$: the gradient vector. $\lVert g\rVert$: its length (norm), the square root of the sum of the squares of its entries.
- $c$: the clipping threshold you choose, the longest gradient you will allow.
- $\min(1,c/\lVert g\rVert)$: a scale factor. It is 1 when the gradient is already short enough, and smaller when it is too long, so the direction is kept but the length is capped at $c$.

#flashcards/ml/cloze #flashcards/topic/neural-nets Clipping caps the ==length== of the gradient but keeps its ==direction==.

#flashcards/ml/example #flashcards/topic/neural-nets Over 10 layers with the same factor each, what is the total gradient factor for sigmoid layers ($f=0.25$) and for $f=1.5$?::0.25¹⁰ = 9.5 × 10⁻⁷ (the first layer gets about a millionth of the gradient). 1.5¹⁰ = 57.7 (the gradient grows nearly sixty times).

#flashcards/ml/example #flashcards/topic/neural-nets For $n_{\text{in}}=n_{\text{out}}=100$, what are the Xavier and He variances and standard deviations?::Xavier: 2/200 = 0.01, standard deviation 0.1. He: 2/100 = 0.02, standard deviation 0.141.

#flashcards/ml/example #flashcards/topic/neural-nets $g=(3,4)$ and $c=1$. What is the clipped gradient, and what if $g=(0.3,0.4)$?::‖g‖ = 5, factor min(1, 1/5) = 0.2, clipped (0.6, 0.8) with length exactly 1. For (0.3, 0.4) the length is 0.5 < 1, so the factor is 1 and it passes unchanged.

#flashcards/ml/image #flashcards/topic/neural-nets ![[activation-derivatives.svg]]
What does this picture show?
?
Four activation functions (red) and their slopes, the derivatives (orange dashed). Top, sigmoid: the slope peaks at 0.25 in the middle and falls towards 0 on both sides, so every sigmoid layer multiplies the backward signal by at most 0.25. Second, ReLU: the slope is exactly 0 or 1. The other two are variants (leaky ReLU, SiLU).
