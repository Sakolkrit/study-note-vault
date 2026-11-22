---
tags: [machine-learning, deep-learning, neural-network, residual-connection, skip-connection, ResNet, transformer]
status: growing
level: 4
tier: 2
---

# Residual Connection

Also called a **skip connection** or **shortcut**. A network built from blocks that use it is a **residual network**, or **ResNet**.

## 🟢 1. Simple (age 12)
A residual connection is **a shortcut that lets a message skip past a few layers**. Normally each layer completely rewrites what it receives. With a shortcut, a layer is told: "keep what you got, and just **add your small improvement** on top." So information can travel through a very deep network easily, like a motorway running alongside a winding country road.

**Example:** You are editing a friend's essay. The **bad way** is to rewrite the whole thing from scratch every round, so after 50 rounds nothing of the original survives. The **good way** is to keep the essay and only add small corrections each round. The original is never lost, and each round only needs to find a small fix. A residual connection is the "keep the essay, add corrections" rule.

## 🔵 2. How it works
1. A normal layer (or a small group of layers) takes an input $x$ and produces a new output $F(x)$, which **replaces** $x$.
2. A **residual block** instead produces **$x + F(x)$**: the original input, plus what the layers computed. The input is carried over on the shortcut (an "identity" path that does nothing to it) and added back in at the end.
3. So the layers only have to learn the **change** needed ("the residual"), not the whole answer. If the best thing to do is nothing, they can just learn $F(x)\approx0$.
4. **Going backwards** ([[Backpropagation]]), the gradient also gets a shortcut: it can flow straight along the identity path, untouched by any layer. This stops it from shrinking away ([[Vanishing and Exploding Gradients]]).
5. Because of that, networks with **dozens, hundreds or more layers** become trainable. Before this idea, simply stacking more layers often made training results *worse*.
6. The same trick is inside every block of a [[Transformer]] (and so an [[LLM]]): the output of the attention part and of the feed-forward part is added back to its own input.

![[residual-block.png]]
*A residual block. The input $x$ goes down through two layers, which together compute $F(x)$. At the same time, $x$ is copied along the curved "identity" path on the right, which does nothing to it, and the two are added (the circled plus). The output is $x+F(x)$. Image: LunarLullaby, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:ResBlock.png), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **The shape:** output $=$ input $+$ (a small stack of layers applied to the input). The layers $F$ are usually a couple of linear layers with an [[Activation Function]] in between, or an attention block, or a feed-forward block.
- **Sizes must match:** the two things being added need the same shape. When a block changes the size, the shortcut gets a small linear layer (a "projection") to match.
- **Why it helps training:** it keeps a clean path for gradients (see level 4) and a good starting point: a block that is initialised to output about 0 starts as "do nothing", which is a safe start for a very deep stack.
- **A sign it works:** people found that deeper plain networks sometimes had *higher training error* than shallower ones, which is not overfitting but a training difficulty. Residual connections largely removed that effect (reported in the original ResNet paper; from memory).
- **In Transformers:**
  - Each block adds the attention output and the feed-forward output back to its input.
  - [[Normalization]] is placed either after the addition ("post-norm") or before the layers inside the branch ("pre-norm"). Many modern large models use pre-norm because it trains more stably (a common observation; check the model you care about).
  - Together, the shortcut and the normalisation are a big part of why very deep Transformers can be trained at all.
- **The "residual stream" view:** in a Transformer, all the shortcuts together form one running vector per token that each block reads from and adds to. This view is common in interpretability research on [[LLM]]s (from memory).
- **Related shortcuts:** dense connections (each layer gets all earlier outputs), highway networks with a learned gate, and U-shaped networks that link early and late layers. Not covered here.
- **Cost:** almost free. One addition per block, and no extra weights unless a projection is needed.
- **It does not fix everything:** you still need good initialisation, a sensible [[Learning Rate]], and [[Normalization]].

```python
import numpy as np
x = 3.0
F = lambda x: 0.1 * x                    # the "layers": a small correction
print(x + F(x))                          # 3.3    residual block output: input plus correction

# slope of one block, and of a stack of 10 blocks (every layer has slope f)
for f in (-0.1, 0.0, 0.1):
    print(f, round(f ** 10, 12), round((1 + f) ** 10, 3))
# -0.1 1e-10 0.349       plain stack: gradient factor f**10 vanishes ; residual: (1+f)**10 stays near 1
# 0.0 0.0 1.0
# 0.1 1e-10 2.594

# 30 layers of 100 units with small random weights, signal size at the end
n = 100
rng = np.random.default_rng(0); x0 = rng.normal(size=n)
def run(res, L=30, std=0.01):
    h = x0.copy()
    for _ in range(L):
        d = np.maximum(0, rng.normal(0, std, (n, n)) @ h)       # a layer with ReLU
        h = h + d if res else d                                  # residual: add the input back
    return float(np.sqrt((h ** 2).mean()))
print(round(float(np.sqrt((x0 ** 2).mean())), 3), run(False), run(True))
# 0.966 4.2778731230980735e-35 1.896610742420262    input size, plain stack (gone), residual stack (survives)
```

## 🔴 4. Deep / formal
**The block.** A residual block adds its input to what its layers compute:

$$
y=x+F(x)
$$

**Where:**
- $x$ is the block's input and $y$ is its output. Both are vectors of the same length (see [[Vector]]).
- $F(x)$ is whatever the block's layers compute from $x$: the **residual function**, the correction to be added.
- The "$x+$" is the **identity shortcut**: it passes $x$ through unchanged.
- **Matters:** the block only needs to learn the *difference* between the input and the desired output. If the best answer is "leave it alone", the layers can set $F(x)\approx0$, which is easy (small weights) and gives $y\approx x$. A plain block would have to learn to reproduce $x$ exactly, which is a much harder thing for a stack of curved layers.

**Example (tiny numbers):** $x=3$ and layers that compute $F(x)=0.1x$.
- $F(3)=\mathbf{0.3}$.
- $y=3+0.3=\mathbf{3.3}$.
- Meaning: the output is the input with a 10% correction added. If the layers learned $F\approx0$, $y$ would stay at about $3$.

**Why the gradient survives.** Differentiate the block with respect to its input (see [[Calculus]]). For a stack of $L$ blocks, the gradient from the end to the start is a product over the blocks:

$$
\frac{\partial y}{\partial x}=1+F'(x),\qquad \frac{\partial y_L}{\partial x_0}=\prod_{l=1}^{L}\big(1+F'_l\big)
$$

**Where:**
- $\partial y/\partial x$ is how much the output changes if the input changes slightly (the slope, or in several dimensions a matrix of slopes; the one-number version is used for clarity).
- $F'(x)$ is the slope of the layers' function. The extra **$1$** comes from the identity shortcut: changing $x$ by a small amount changes the shortcut's output by exactly that amount.
- $F'_l$ is the slope of block $l$'s layers, and $\prod_{l=1}^L$ multiplies the factors of all $L$ blocks (compare the product in [[Vanishing and Exploding Gradients]]).
- **Matters:** each block's factor is $1+F'_l$, not $F'_l$. With small slopes ($F'_l\approx0$) the factor is about $1$, so the product stays near $1$ however deep the stack is. For plain layers the factor would be $F'_l$ itself, and a product of small numbers vanishes.
- **Caution:** with *large* slopes, the product can still explode. Residual connections remove the easy route to vanishing but not the need for care with initialisation and the [[Learning Rate]]. With many dimensions the factors are matrices, and the real behaviour is more complicated than this one-number picture.

**Example (tiny numbers):** 10 blocks, each with the same slope $F'=f$.
- $f=0.1$: a plain stack gives $0.1^{10}=\mathbf{10^{-10}}$ (gone). The residual stack gives $(1.1)^{10}=\mathbf{2.594}$.
- $f=-0.1$: plain $=10^{-10}$ again. Residual: $(0.9)^{10}=\mathbf{0.349}$.
- $f=0$: plain $=0$. Residual: $1^{10}=\mathbf{1}$, the gradient arrives unchanged.
- Meaning: a plain stack of weak layers sends the gradient to essentially zero. A residual stack keeps it within a modest range around 1, which is why the early layers still learn.
- The code's 30-layer test shows the same in the forward direction: the signal drops to about $10^{-35}$ in the plain stack and survives (size $1.9$, from an input of $0.97$) in the residual stack.

**Further notes:**
- **Unrolled view:** a stack of residual blocks computes $x_L=x_0+F_1(x_0)+F_2(x_1)+\cdots+F_L(x_{L-1})$: the input plus a sum of corrections. So the network behaves a bit like an ensemble of paths of different lengths, most of them short. This view comes from the research literature (from memory; check the papers).
- **Link to the Euler method:** $x_{l+1}=x_l+F(x_l)$ looks like a step of a numerical solver for a differential equation, which led to "neural ODE" models. A research direction, not covered here.
- **Initialisation tricks:** scaling down or zeroing the last layer in each residual branch makes every block start as the identity. Used in some very deep training recipes (from memory).
- **Variance growth:** adding $L$ branch outputs to the stream makes the signal grow with depth unless the branches are scaled or normalised. In my tuning runs for the code above, raising the weight spread from $0.01$ to $0.03$ made the residual stack's final signal grow from about $1.9$ to about $22$ (those runs are not in the snippet).
- **Not the same as** [[Dropout]] or [[Regularization]]: those fight overfitting, while residual connections make deep networks *trainable*.

**What came before, and how this improved on it:**
- **Before:** plain stacks of layers where each layer replaces what it receives (AlexNet had 8 layers, and networks of about 16 to 19 layers followed, from memory; I did not check the counts).
- **Its limit:** with many layers the error signal fades or the network becomes hard to train. In the toy of [[ResNet]], 20 plain hidden layers did not train at all (loss stuck at 2.3024 = $\ln10$, accuracy 0.104).
- **What changed:** add the layer's input to its output, so the layers only have to learn a correction and the gradient has a direct path back.
- **How much better:** in the same toy, 20 hidden layers with shortcuts reached loss 0.0019 and accuracy 1.0, where the plain stack stayed at 2.3024 (the first layer's starting gradient was 2.012e+01 against 3.337e-09). At 40 layers, shortcuts alone were not enough (loss 0.7045, accuracy 0.756), so scale and [[Normalization]] matter too. In the published result, networks of up to 152 layers were trained and an ensemble reached 3.57% error on the ImageNet test set (arXiv abstract of the ResNet paper, which I read). See [[ResNet]].

## Related
- [[Neural Network]] · [[Vanishing and Exploding Gradients]] · [[Backpropagation]] · [[Activation Function]] · [[Normalization]] · [[Transformer]] · [[Attention]] · [[LLM]] · [[Calculus]] · [[Vector]] · [[Learning Rate]] · [[Gradient Descent]] · [[Convolutional Neural Network]] · [[Self-Attention]] · [[ResNet]] · [[Layer Normalization]] · [[LSTM]] · [[Feed-Forward Network]]

## Sources
- He, Zhang, Ren & Sun, *Deep Residual Learning for Image Recognition*, 2015 (ResNet; from memory)
- Vaswani et al., *Attention Is All You Need*, 2017 (residual connections in the Transformer)
- Veit, Wilber & Belongie, *Residual Networks Behave Like Ensembles of Relatively Shallow Networks*, 2016 (from memory)
- Elhage et al., *A Mathematical Framework for Transformer Circuits*, 2021 (the residual stream; from memory)
- Goodfellow, Bengio & Courville, *Deep Learning*, 2016, chapter 8

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/neural-nets Residual connection:::A shortcut that lets a message skip past a few layers. Each layer keeps what it received and only adds its small improvement on top, so information travels easily through a very deep network, like a motorway.

#flashcards/ml/formula #flashcards/topic/neural-nets Write a residual block and explain each symbol.
?
$$y=x+F(x)$$
- $x$: the block's input. $y$: its output. Both are vectors of the same length.
- $F(x)$: whatever the block's layers compute from $x$, the residual function, the correction to be added.
- The "$x+$" is the identity shortcut, which passes $x$ through unchanged.
- The block only needs to learn the change, not the whole mapping. If the best change is nothing, it can learn $F\approx0$.

#flashcards/ml/formula #flashcards/topic/neural-nets Write the gradient through residual blocks and explain each symbol.
?
$$\frac{\partial y}{\partial x}=1+F'(x),\qquad \frac{\partial y_L}{\partial x_0}=\prod_{l=1}^{L}\big(1+F'_l\big)$$
- $\partial y/\partial x$: how much the output changes if the input changes slightly (one-number version for clarity).
- $F'(x)$: the slope of the layers' function. The extra 1 comes from the identity shortcut: changing $x$ by a small amount changes $y$ by at least that amount.
- The product over $L$ blocks has factors near 1 instead of near 0, so the gradient does not vanish.

#flashcards/ml/cloze #flashcards/topic/neural-nets A residual block outputs ==x + F(x)==, and the extra ==1== in the gradient stops it from vanishing.

#flashcards/ml/example #flashcards/topic/neural-nets $x=3$ and $F(x)=0.1x$. What is $y$?::F(3) = 0.3, so y = 3 + 0.3 = 3.3. The output is the input with a 10% correction. If F ≈ 0, y stays at about 3.

#flashcards/ml/example #flashcards/topic/neural-nets 10 blocks, each with slope $F'=f$. What are the plain and residual gradients for $f=0.1$, $-0.1$ and $0$?::f = 0.1: plain 0.1¹⁰ = 10⁻¹⁰ (gone), residual 1.1¹⁰ = 2.594. f = −0.1: plain 10⁻¹⁰, residual 0.9¹⁰ = 0.349. f = 0: plain 0, residual 1¹⁰ = 1, the gradient arrives intact.

#flashcards/ml/image #flashcards/topic/neural-nets ![[residual-block.png]]
What does this picture show?
?
A residual block. The input x goes down through two layers, which together compute F(x). At the same time x is copied along the curved "identity" path on the right, which does nothing to it, and the two are added (the circled plus). The output is x + F(x).
