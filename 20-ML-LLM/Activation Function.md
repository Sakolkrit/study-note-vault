---
tags: [machine-learning, deep-learning, neural-network, activation-function, ReLU, sigmoid]
status: growing
level: 4
tier: 1
---

# Activation Function

## 🟢 1. Simple (age 12)
An activation function is a **small rule that decides how strongly a brain cell in a network "fires"** after it has added up its inputs. Without it, a network would only be able to draw straight lines. With it, the network can learn **curves and twists**, which is what makes recognising a cat or understanding a sentence possible.

**Example:** A light switch is a rule: "if the push is big enough, turn on, otherwise stay off". A dimmer is a gentler rule: "the harder you turn, the brighter, up to a limit". Different activation functions are different kinds of switches and dimmers placed after each cell.

## 🔵 2. How it works
1. A single unit (neuron) in a [[Neural Network]] adds up its inputs, each multiplied by a weight, and adds a constant. That gives one number, $z$.
2. The unit does **not** pass $z$ straight on. It first sends it through the **activation function**, which reshapes it, and passes on the result.
3. The reshaping must be **bent** (not a straight line). If it were a straight line, stacking many layers would be no better than one layer, because a straight line of a straight line is still a straight line.
4. Some popular shapes:
   - **Sigmoid:** squashes any number into a value between 0 and 1. Good for "how likely?" at the end of a model (see [[Classification]]).
   - **ReLU:** "if negative, output 0; otherwise pass it on unchanged". Very simple and very common inside networks. [[ReLU]]
   - **Leaky ReLU:** like ReLU, but negatives are not killed, only made small.
   - **SiLU** (also called swish): a smooth cousin of ReLU used in many modern networks.
   - **Softmax:** used at the end to turn several scores into chances that add up to 100% ([[Softmax]]).
5. The choice affects how well learning flows backwards ([[Backpropagation]]): the slope of the activation function gets multiplied in at every layer (see [[Vanishing and Exploding Gradients]]).

![[activation-derivatives.svg]]
*Four activation functions (red, left scale) and their slopes (orange dashed, right scale). Top, sigmoid: an S-shaped curve between 0 and 1, with a slope that peaks at 0.25. Second, ReLU: flat at 0 for negatives, then a straight rise; the slope is 0 or 1. Third, leaky ReLU (with $\alpha=0.2$ in this picture): a gentle slope for negatives. Bottom, SiLU: a smooth curve that dips slightly below 0 before rising. The captions on the file are in French ("Sigmoide"); I read the numbers off the axes. Image: CapeItalist, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Fonctions_d%27activation_et_leur_d%C3%A9riv%C3%A9e.svg), CC0.*

## 🟠 3. Technical detail
- **Where they sit:** after the weighted sum in each hidden layer. The output layer often uses a special one: sigmoid for a yes/no chance, [[Softmax]] for many classes, or none at all for a plain number ([[Regression]]).
- **Common choices:**
  - **ReLU:** $\max(0,z)$. Cheap, and its slope of 1 for positive inputs helps gradients flow. Downside: a unit whose input is always negative outputs 0 forever and stops learning ("dying ReLU").
  - **Leaky ReLU:** keeps a small slope for negatives, so units can recover.
  - **Sigmoid:** between 0 and 1. Its slope is at most $0.25$ and nearly 0 for large $|z|$, so deep stacks of it train badly.
  - **Tanh:** like sigmoid but ranges from $-1$ to $1$ and is centred on 0. It has the same saturation problem.
  - **SiLU / swish:** $z\cdot\sigma(z)$, smooth, and used in many recent networks.
  - **GELU:** another smooth ReLU relative, widely used in [[Transformer]] feed-forward layers. (It is defined with the normal-distribution curve, see [[Normal Distribution]]; I give no formula here, and the exact choice per model should be checked.)
- **Choosing:** a good default for hidden layers is ReLU or one of its smooth variants, matched with a good initialisation (see [[Vanishing and Exploding Gradients]]). Try alternatives only if training misbehaves.
- **Output layer and loss go together:** a sigmoid output pairs with a yes/no loss, [[Softmax]] with the multi-class [[Loss Function]], and no activation with squared error.
- **Saturation:** where a curve goes flat (the ends of sigmoid and tanh), the slope is near 0, so almost no gradient flows back. This is the main cause of vanishing gradients with these functions.
- **Computing cost:** ReLU is just a comparison. Sigmoid, tanh and SiLU need exponentials, which cost more but are small next to the matrix products ([[GPU]]).
- **Not to be confused with:** [[Normalization]] (rescaling the numbers in a layer) and [[Positional Encoding]]. They are separate steps.

```python
import numpy as np
sig = lambda z: 1 / (1 + np.exp(-z))

w1, w2 = 2, 3                                       # two layers with one weight each
for x in (-1, 1):
    print(x, w2 * (w1 * x), w2 * max(0, w1 * x))    # -1 -6 0    then    1 6 6
# without ReLU the two layers are just "times 6"; with ReLU in between the output bends at 0

for z in (-2, 0, 2, 5):
    s = sig(z); print(z, round(float(s), 4), round(float(s * (1 - s)), 4))   # sigmoid and its slope s*(1-s)
# -2 0.1192 0.105
# 0 0.5 0.25
# 2 0.8808 0.105
# 5 0.9933 0.0066     already flat: the slope is tiny

a = 0.1
for z in (3, -2):
    print(z, max(0, z), z if z > 0 else a * z)      # ReLU and leaky ReLU: 3 3 3 ; -2 0 -0.2
for z in (2, 0, -2): print(z, round(float(z * sig(z)), 3))   # SiLU: 1.762, 0.0, -0.238
```

## 🔴 4. Deep / formal
**Why a bend is needed: stacked straight lines are one straight line.** Two layers without an activation function multiply together:

$$
y=W_2\,(W_1\,x)=(W_2W_1)\,x
$$

**Where:**
- $x$ is the input, $W_1$ and $W_2$ are the weights of the first and second layers (matrices, see [[Matrix]]; here just single numbers for the example), and $y$ is the output.
- $W_2W_1$ is a single combined weight. **Matters:** the two layers collapse into **one** layer, so a hundred layers without activation functions would be no more powerful than one. Depth only helps when something non-linear is placed between the layers.
- Putting an activation $f$ between the layers, $y=W_2\,f(W_1x)$, breaks this collapse. **Matters:** with a bent $f$ such as ReLU, the network can represent curved shapes, and with enough units it can approximate a very wide range of functions (the "universal approximation" results; stated from memory, check the exact conditions).

**Example (tiny numbers):** $W_1=2$, $W_2=3$.
- Without an activation: $y=3\cdot(2x)=6x$. For $x=-1$: $\mathbf{-6}$. For $x=1$: $\mathbf{6}$. A straight line.
- With ReLU between: $y=3\cdot\max(0,2x)$. For $x=-1$: $\max(0,-2)=0$, so $y=\mathbf{0}$. For $x=1$: $y=3\cdot2=\mathbf{6}$.
- Meaning: with ReLU the output is flat at 0 for negative inputs and rises for positive ones. That is a bend, which a pure "times 6" could never make.

**Sigmoid and its slope.** The sigmoid squashes any number into a chance between 0 and 1, and its slope has a neat form:

$$
\sigma(z)=\frac{1}{1+e^{-z}},\qquad \sigma'(z)=\sigma(z)\big(1-\sigma(z)\big)
$$

**Where:**
- $z$ is the input (the weighted sum). $e$ is Euler's number ($\approx2.718$), and $e^{-z}$ is large when $z$ is very negative and tiny when $z$ is very positive.
- $\sigma(z)$ is the output, between 0 and 1: near 0 for very negative $z$, exactly $0.5$ at $z=0$, near 1 for very positive $z$.
- $\sigma'(z)$ is the slope (the derivative, see [[Calculus]]): how much the output changes per unit change of $z$. **Matters:** it is largest at $z=0$, where it equals $0.5\cdot0.5=0.25$, and it **falls towards 0 on both sides**. A saturated sigmoid passes almost no gradient back. This, with the slope never exceeding $0.25$, is why chains of sigmoids make gradients vanish ([[Vanishing and Exploding Gradients]]).
- The slope can be computed from the output alone, which made it convenient for early software.

**Example (tiny numbers):**
- $z=0$: $\sigma=\mathbf{0.5}$, slope $0.5\cdot0.5=\mathbf{0.25}$ (the maximum).
- $z=2$: $\sigma=\mathbf{0.881}$, slope $0.881\cdot0.119=\mathbf{0.105}$.
- $z=-2$: $\sigma=\mathbf{0.119}$, slope $\mathbf{0.105}$ (the same, by symmetry).
- $z=5$: $\sigma=\mathbf{0.9933}$, slope $\mathbf{0.0066}$.
- Meaning: moving from $z=0$ to $z=5$ cuts the slope about 38 times. A unit deep in the flat part barely learns.

**ReLU and leaky ReLU.** The most popular bend, and a version that never goes completely flat:

$$
\text{ReLU}(z)=\max(0,z),\qquad \text{LeakyReLU}(z)=\begin{cases}z & z>0\\ \alpha z & z\le0\end{cases}
$$

**Where:**
- $z$ is the input, and $\max(0,z)$ means "the larger of 0 and $z$": it returns $z$ if positive and $0$ if not.
- $\alpha$ ("alpha") is a small positive number, such as $0.01$ or $0.1$ (the picture above uses $0.2$). **Matters:** it is the slope for negative inputs.
- The slope of ReLU is $1$ for $z>0$ and $0$ for $z<0$ (at exactly $z=0$ it is not defined, and software just picks one). The slope of leaky ReLU is $1$ for $z>0$ and $\alpha$ for $z<0$.
- **Matters:** a slope of exactly 1 passes gradients through unchanged, which helps deep networks learn. A slope of 0 for negatives can leave a unit stuck forever, and $\alpha$ keeps a small gradient flowing so it can recover.

**Example (tiny numbers):** $\alpha=0.1$.
- $z=3$: ReLU gives $\mathbf{3}$, leaky gives $\mathbf{3}$, slope $\mathbf{1}$ for both.
- $z=-2$: ReLU gives $\mathbf{0}$ with slope $\mathbf{0}$. Leaky gives $0.1\cdot(-2)=\mathbf{-0.2}$ with slope $\mathbf{0.1}$.
- Meaning: for a negative input, ReLU has stopped listening completely, while leaky ReLU still lets a little learning signal through.

**SiLU: a smooth bend.** SiLU (sigmoid-weighted linear unit, also called swish) multiplies the input by its own sigmoid:

$$
\text{SiLU}(z)=z\cdot\sigma(z)
$$

**Where:**
- $z$ is the input and $\sigma(z)$ is the sigmoid from above, a number between 0 and 1 that acts like a smooth "how much of $z$ to let through" gate.
- **Matters:** for large positive $z$ the gate is almost 1, so SiLU is almost $z$ (like ReLU). For large negative $z$ the gate is almost 0, so SiLU is almost 0. In between it changes smoothly with no sharp corner, and it dips slightly below zero for small negative $z$.
- The smoothness gives a slope that varies gently, which some networks train better with (a common claim from the papers; the benefit depends on the model).

**Example (tiny numbers):**
- $z=2$: $2\cdot\sigma(2)=2\cdot0.881=\mathbf{1.762}$ (close to $z$).
- $z=0$: $0\cdot0.5=\mathbf{0}$.
- $z=-2$: $-2\cdot0.119=\mathbf{-0.238}$ (slightly negative, not cut to 0 as in ReLU).
- Meaning: SiLU behaves like a softened ReLU: nearly the input when it is big and positive, nearly zero when very negative, with a smooth transition.

**Further notes:**
- **Tanh** is a rescaled sigmoid: $\tanh(z)=2\sigma(2z)-1$ (a standard identity), with outputs between $-1$ and $1$, for example $\tanh(2)\approx0.964$ and $\tanh(-2)\approx-0.964$. Its slope at 0 is 1, larger than the sigmoid's 0.25, but it also saturates.
- **Gated units:** many [[Transformer]] and [[LLM]] designs use gated feed-forward layers, where one branch passes through an activation and multiplies another. The details vary by model; check the model you care about.
- **Smoothness and the loss surface:** the activation shapes the loss landscape that [[Gradient Descent]] walks on. Sharp corners (ReLU) are fine in practice even though the slope jumps there.
- **Which is "best":** mostly empirical. Results come from experiments and change with architecture, so treat any ranking as provisional.

**What came before, and how this improved on it:**
- **Before:** smooth squashing functions such as the sigmoid (slope at most 0.25) and tanh.
- **Its limit:** through many layers the learning signal is multiplied by those small slopes again and again.
- **What changed:** the rectifier (ReLU) has slope 1 for positive inputs, so the signal passes. The abstract of the "Delving Deep into Rectifiers" paper calls rectifiers "essential for state-of-the-art neural networks", and its PReLU network reached 4.94% top-5 error on ImageNet, a 26% relative improvement over GoogLeNet's 6.66% and the first result it says passed the human level of 5.1% (He et al., arXiv 1502.01852, 6 February 2015, abstract read). The ReLU's first use (Nair and Hinton, 2010) is from memory.
- **How much better:** in the code below, a 20-layer network with sigmoids gives a gradient of $1.67\times10^{-14}$ at the first layer against $1.16\times10^{-1}$ at the last, a ratio of $1.4\times10^{-13}$. With ReLU the same two sizes are 0.300 and 0.435, a ratio of 0.69. The weights were initialised in the usual way for each.

**Check it yourself:**

```python
import torch
torch.manual_seed(0)
def first_layer_grad(act, depth=20, width=32):
    layers = []
    for _ in range(depth): layers += [torch.nn.Linear(width, width), act()]
    net = torch.nn.Sequential(*layers)
    for m in net:
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight) if act is torch.nn.Sigmoid else torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
    x = torch.randn(64, width); net(x).pow(2).mean().backward()
    return net[0].weight.grad.norm().item(), net[-2].weight.grad.norm().item()
for name, a in (("sigmoid", torch.nn.Sigmoid), ("ReLU", torch.nn.ReLU)):
    first, last = first_layer_grad(a)
    print(f"{name:8s} 20 layers: gradient size at the last layer {last:.2e}, at the first layer {first:.2e}, ratio {first/last:.2e}")
print("largest slope of the sigmoid is 0.25, so 20 layers shrink a signal by at most 0.25**20 =", f"{0.25**20:.1e}")
```

## Related
- [[Neural Network]] · [[Softmax]] · [[Backpropagation]] · [[Vanishing and Exploding Gradients]] · [[Classification]] · [[Regression]] · [[Matrix]] · [[Calculus]] · [[Normalization]] · [[Transformer]] · [[LLM]] · [[Loss Function]] · [[Gradient Descent]] · [[GPU]] · [[Normal Distribution]] · [[Positional Encoding]] · [[Convolutional Neural Network]] · [[Logistic Regression]] · [[Convolution]] · [[Forward Propagation]] · [[ReLU]]

## Sources
- Goodfellow, Bengio & Courville, *Deep Learning*, 2016, chapter 6
- Glorot, Bordes & Bengio, *Deep Sparse Rectifier Neural Networks*, 2011 (ReLU; from memory)
- Ramachandran, Zoph & Le, *Searching for Activation Functions*, 2017 (swish / SiLU; from memory)
- Hendrycks & Gimpel, *Gaussian Error Linear Units (GELUs)*, 2016 (from memory)
- Cybenko, *Approximation by superpositions of a sigmoidal function*, 1989 (universal approximation; from memory)
- He et al., arXiv 1502.01852 (abstract read for the history block)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/neural-nets Activation function:::A small rule that decides how strongly a neuron fires after adding up its inputs. Without it a network could only draw straight lines.

#flashcards/ml/formula #flashcards/topic/neural-nets Show why a network without activation functions is no more powerful than one layer.
?
$$y=W_2\,(W_1\,x)=(W_2W_1)\,x$$
- $x$: the input. $W_1$, $W_2$: the weights of the first and second layers.
- $y$: the output.
- $W_2W_1$ is a single combined weight. The two layers collapse into one, so a hundred layers without activation functions are no stronger than one. Depth only helps when something non-linear sits between the layers.

#flashcards/ml/formula #flashcards/topic/neural-nets Write the sigmoid and its slope, and explain each symbol.
?
$$\sigma(z)=\frac{1}{1+e^{-z}},\qquad \sigma'(z)=\sigma(z)\big(1-\sigma(z)\big)$$
- $z$: the input (the weighted sum). $e$: Euler's number, about 2.718.
- $\sigma(z)$: the output, between 0 and 1. Near 0 for very negative $z$, exactly 0.5 at $z=0$, near 1 for very positive $z$.
- $\sigma'(z)$: the slope. It peaks at 0.25 at $z=0$ and gets tiny at the ends, which causes vanishing gradients.

#flashcards/ml/formula #flashcards/topic/neural-nets Write ReLU and Leaky ReLU and explain each symbol.
?
$$\text{ReLU}(z)=\max(0,z),\qquad \text{LeakyReLU}(z)=\begin{cases}z & z>0\\ \alpha z & z\le0\end{cases}$$
- $z$: the input. $\max(0,z)$: the larger of 0 and $z$, so $z$ if positive and 0 if not.
- $\alpha$: a small positive number such as 0.01 or 0.1. It is the slope for negative inputs.
- ReLU has slope 1 for $z>0$ and 0 for $z<0$. With a slope of 0, a neuron can get stuck "dead". Leaky ReLU keeps a small slope there.

#flashcards/ml/formula #flashcards/topic/neural-nets Write SiLU and explain each symbol.
?
$$\text{SiLU}(z)=z\cdot\sigma(z)$$
- $z$: the input. $\sigma(z)$: the sigmoid, a number between 0 and 1 that acts like a smooth gate for how much of $z$ to let through.
- For large positive $z$ the gate is near 1, so SiLU is near $z$ (like ReLU). For large negative $z$ it is near 0.
- In between it changes smoothly with no sharp corner.

#flashcards/ml/cloze #flashcards/topic/neural-nets The sigmoid's slope peaks at ==0.25== when $z=0$. ReLU has slope ==1== for positive inputs and ==0== for negative inputs.

#flashcards/ml/example #flashcards/topic/neural-nets $W_1=2$, $W_2=3$. What is the output for $x=-1$ and $x=1$ without an activation, and with ReLU in between?::Without: y = 6x, so −6 and 6, a straight line. With ReLU: x = −1 gives max(0, −2) = 0, so y = 0. x = 1 gives 3·2 = 6. The bend lets the network make a non-straight shape.

#flashcards/ml/example #flashcards/topic/neural-nets What are $\sigma(z)$ and its slope at $z=0$, $2$, $-2$ and $5$?::z = 0: 0.5, slope 0.25 (the maximum). z = 2: 0.881, slope 0.105. z = −2: 0.119, slope 0.105 (symmetric). z = 5: 0.9933, slope 0.0066.

#flashcards/ml/example #flashcards/topic/neural-nets With $\alpha=0.1$, what do ReLU and Leaky ReLU give for $z=-2$?::ReLU: 0 with slope 0. Leaky: 0.1 × (−2) = −0.2 with slope 0.1. ReLU stops listening, leaky ReLU still lets a little through.

#flashcards/ml/example #flashcards/topic/neural-nets What does SiLU give at $z=2$, $0$ and $-2$?::2 × 0.881 = 1.762. 0 × 0.5 = 0. −2 × 0.119 = −0.238 (slightly negative, not cut to 0).

#flashcards/ml/image #flashcards/topic/neural-nets ![[activation-derivatives.svg]]
What does this picture show?
?
Four activation functions (red, left scale) and their slopes (orange dashed, right scale). Top, sigmoid: an S-shaped curve between 0 and 1 whose slope peaks at 0.25. Second, ReLU: flat at 0 for negatives, then a straight rise.
