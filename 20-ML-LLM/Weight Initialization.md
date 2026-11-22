---
tags: [initialization, neural-network, deep-learning, training, He, Xavier, Glorot]
status: growing
level: 4
tier: 2
---

# Weight Initialization

How to pick the **random starting numbers** (the weights) of a [[Neural Network]] before training begins. The right size of those numbers decides whether training gets going at all.

## 🟢 1. Simple (age 12)
A neural network learns by changing its numbers a little at a time. But it has to start somewhere. If every number starts at the same value, all the little workers inside do the same job and never become different. If the numbers start too big, the signal gets louder and louder as it passes through the layers until it is a roar. If they start too small, it fades to a whisper. **Weight initialization is choosing starting numbers that are random, but of just the right size.**

**Example:** Pass a message down a line of 50 kids. If each kid repeats it a bit louder, the last kid is shouting. If each repeats it a bit quieter, the last kid says nothing. You want each kid to repeat it at about the same loudness.

## 🔵 2. How it works
1. A layer takes many inputs, multiplies each by a weight and **adds them up**. The sum is the layer's output.
2. Adding many numbers makes the total more spread out. A layer with 100 inputs adds up 100 terms, so its output is about 100 times more spread out (in variance) than a single term. See [[Variance]].
3. So the weights must be **small enough to cancel that growth**. The rule of thumb: make each weight's variance about **1 divided by the number of inputs**. That keeps each layer's output about as spread out as its input.
4. The activation changes the rule. [[ReLU]] throws away about half the signal (every negative number becomes 0), so its weights need to be a bit bigger, **2 divided by the number of inputs** (called **He** or **Kaiming** initialization). For tanh the 1-over-inputs rule (called **LeCun**, or close to **Xavier / Glorot**) is the usual one.
5. **Never start all weights at the same value** (zero or any other). Every hidden unit then sees the same thing and gets the same update, so they stay copies of each other forever. Random starts break that tie. Biases are usually started at 0.
6. **The error compounds with depth.** Being 10% too big per layer sounds harmless, but over 50 layers it is 117 times too big. Being 10% too small gives 0.005 of the signal. That is why depth makes this important (see [[Vanishing and Exploding Gradients]]).
7. **It still matters with modern tools,** but less than it did: layers such as [[Batch Normalization]] and [[Layer Normalization]] and skip paths ([[Residual Connection]]) rescale things during training and forgive a rough start.

No image: my Wikimedia Commons search for "weight initialization neural network" returned only scanned PDFs of unrelated reports, so I left it out.

## 🟠 3. Technical detail
- **Zero for biases, random for weights.** Zero weights leave all hidden units identical (the code shows no difference at all after 300 steps). Constant non-zero weights do the same thing.
- **The main recipes** (Wikipedia page, with the variance checked in the code):
  - **LeCun:** weights with mean 0 and variance $1/n_{\text{in}}$. Keeps the forward signal's size.
  - **Glorot / Xavier:** a compromise between keeping the forward signal and the backward gradient steady. The uniform version samples from $\pm\sqrt{6/(n_{\text{in}}+n_{\text{out}})}$.
  - **He / Kaiming:** normal with variance $2/n_{\text{in}}$, for ReLU networks.
  - **Orthogonal:** start each weight matrix as a random orthogonal one (page statement: this makes training time in a deep *linear* network independent of depth). Not run here.
- **Normal or uniform?** Either works if the variance is right. The code shows the Glorot uniform limit gives exactly the intended variance.
- **Other methods (page, not run):** Fixup (for residual networks without BatchNorm), LSUV (data-driven, rescales each layer from a test batch), sparse and "looks linear" initializations, and the SELU activation, which pulls layer statistics toward mean 0 and variance 1.
- **Small positive biases (page):** a bias like 0.1 for ReLU units, or 1 for an LSTM forget gate, are special cases. Not run.
- **Related fixes:** normalisation layers fix the scale *during* training, while initialization fixes it only at the *start*. The page describes a trade-off: BatchNorm makes the examples in a batch depend on each other, while a good initialization depends on the architecture.

```python
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
torch.set_num_threads(1)

# 1) one layer: output variance = n_in * Var(w) * E[x^2]  (100 inputs, each mean 0 and variance 1)
rng = np.random.RandomState(0); n = 100
x = rng.randn(20000, n)
for name, var_w in (("Var(w)=1/n (LeCun)", 1 / n), ("Var(w)=2/n (He)", 2 / n), ("Var(w)=1 (naive)", 1.0)):
    z = x @ (rng.randn(n, 50) * np.sqrt(var_w))
    print(f"{name:20s}: output variance {z.var():.2f} | after ReLU the mean square is {(np.maximum(z, 0) ** 2).mean():.2f}")
a = np.sqrt(6 / (200 + 100)); print("Glorot uniform limit for fan-in 200, fan-out 100:", round(a, 4), "| variance", round(a * a / 3, 4), "| 2/(200+100) =", round(2 / 300, 4))
print("compounding over 50 layers: 0.9^50 =", round(0.9 ** 50, 4), "| 1.1^50 =", round(1.1 ** 50, 1))

# 2) 50 layers, width 256: spread of the numbers at the last layer, for each starting scale
def last_std(act, std_of_w, depth=50, width=256):
    r = np.random.RandomState(1); h = r.randn(512, width)
    for _ in range(depth): h = act(h @ (r.randn(width, width) * std_of_w))
    return h.std()
relu = lambda v: np.maximum(v, 0)
schemes = (("fixed 0.01", 0.01), ("1/sqrt(n)", 256 ** -0.5), ("sqrt(2/n)", (2 / 256) ** 0.5), ("fixed 0.2", 0.2))
for aname, act in (("tanh", np.tanh), ("ReLU", relu)):
    print(f"{aname:5s}", " | ".join(f"{s}: {last_std(act, v):.3g}" for s, v in schemes))

# 3) all-equal or all-zero starts: the units never become different
def train(init, steps=300, seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 1))
    for p in net.parameters(): init(p)
    X = torch.randn(64, 3); y = (X[:, :1] * X[:, 1:2]).detach()
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    l0 = ((net(X) - y) ** 2).mean().item()
    for _ in range(steps):
        opt.zero_grad(); l = ((net(X) - y) ** 2).mean(); l.backward(); opt.step()
    w = net[0].weight.detach()
    return l0, l.item(), (w - w[0]).abs().max().item()
l0, l1, d = train(lambda p: nn.init.constant_(p, 0.5)); print(f"all weights 0.5 : loss {l0:.3f} -> {l1:.3f} | largest difference between hidden units {d:.1e}")
l0, l1, d = train(lambda p: nn.init.zeros_(p));        print(f"all weights 0   : loss {l0:.3f} -> {l1:.3f} | largest difference between hidden units {d:.1e}")
l0, l1, d = train(lambda p: nn.init.normal_(p, 0, 0.5) if p.dim() > 1 else nn.init.zeros_(p)); print(f"random weights  : loss {l0:.3f} -> {l1:.3f} | largest difference between hidden units {d:.1e}")

# 4) 10 hidden ReLU layers on 1,000 handwritten digits: training loss after 100 SGD steps
D = load_digits(); X = torch.tensor(D.data[:1000] / 16.0, dtype=torch.float32); y = torch.tensor(D.target[:1000]).long()
def run(std, seed, steps=100):
    torch.manual_seed(seed); layers = []
    for i in range(10): layers += [nn.Linear(64, 64), nn.ReLU()]
    net = nn.Sequential(*layers, nn.Linear(64, 10))
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias); nn.init.normal_(m.weight, 0, (2 / 64) ** 0.5 if std == "he" else std)
    opt = torch.optim.SGD(net.parameters(), lr=0.05)
    for _ in range(steps):
        opt.zero_grad(); l = nn.functional.cross_entropy(net(X), y); l.backward(); opt.step()
    return l.item()
for std in (0.01, "he", 0.5):
    L = [run(std, s) for s in range(3)]
    print(f"start std {std if std != 'he' else 'sqrt(2/64)=0.177'}: loss after 100 steps, 3 seeds -> {np.round(L, 3).tolist()} (2.303 = ln 10 is the loss of an even guess)")
```

What it printed: with 100 inputs, a weight variance of 1/n gave an output variance of 1.01, 2/n gave 1.99 (and a mean square of 0.99 after ReLU), and a naive variance of 1 gave 99.62. The Glorot uniform limit for 200 inputs and 100 outputs was 0.1414, with variance 0.0067, equal to 2/300. Over 50 layers of width 256, the last layer's spread was: for tanh, 1.48e-40 with std 0.01, 0.0948 with 1/√n, 0.56 with √(2/n) and 0.849 with 0.2. For ReLU it was 1.75e-48 with 0.01, 1.09e-08 with 1/√n, 0.365 with √(2/n) and 1.97e+17 with 0.2. A tiny network started with all weights 0.5 fell in loss from 3.788 to 0.738 with a largest difference between hidden units of exactly 0. Started at all zeros it went from 0.976 to 0.974 (also no difference). With random weights it went from 1.454 to 0.067, with units differing by up to 1.3. On handwritten digits, a 10-layer ReLU network after 100 steps had loss 2.302 for all three seeds with start std 0.01, 0.552, 0.36 and 0.242 with $\sqrt{2/64}=0.177$, and NaN for all three with std 0.5.

## 🔴 4. Deep / formal
**One layer's output spread.** For a neuron with $n_{\text{in}}$ inputs $x_j$ and weights $w_j$, where the weights are independent of the inputs and have mean 0:

$$
\operatorname{Var}(z)=n_{\text{in}}\,\operatorname{Var}(w)\,\mathbb{E}[x^{2}],\qquad z=\sum_{j=1}^{n_{\text{in}}}w_j x_j
$$

**Where:**
- $z$ is the neuron's sum before the activation, and $n_{\text{in}}$ is how many inputs it adds up (the "fan-in").
- $\operatorname{Var}(w)$ is the variance of one weight, and $\mathbb{E}[x^2]$ is the average squared size of an input (see [[Expected Value]]).
- **Matters:** the sum adds $n_{\text{in}}$ separate terms, so the spread grows with the width of the layer. To keep $\operatorname{Var}(z)$ equal to $\mathbb{E}[x^2]$, set $\operatorname{Var}(w)=1/n_{\text{in}}$. This relies on mean-zero weights and the independence assumed above, which are true at the start and drift later.

**Example (tiny numbers):** $n_{\text{in}}=100$ inputs with $\mathbb{E}[x^2]=1$.
- With $\operatorname{Var}(w)=0.01$ (that is $1/100$): $\operatorname{Var}(z)=100\times0.01\times1=\mathbf{1}$.
- With $\operatorname{Var}(w)=1$: $\operatorname{Var}(z)=100\times1\times1=\mathbf{100}$.
- Meaning: a weight variance of 1 makes the layer 100 times louder in variance. The code measured 1.01 and 99.62.

**The standard rules.** Pick the weight variance from the layer's size and activation:

$$
\operatorname{Var}(w)=\frac{1}{n_{\text{in}}}\ \text{(LeCun)},\qquad \frac{2}{n_{\text{in}}}\ \text{(He, ReLU)},\qquad \frac{2}{n_{\text{in}}+n_{\text{out}}}\ \text{(Glorot)}
$$

**Where:**
- $n_{\text{in}}$ is the number of inputs and $n_{\text{out}}$ is the number of outputs of the layer ("fan-out").
- LeCun: keeps $\operatorname{Var}(z)=\mathbb{E}[x^2]$ in the equation above.
- He: ReLU sets negative values to 0, which halves the mean square of the output. The factor 2 puts it back. **Matters:** with LeCun's rule on a ReLU network the signal shrinks by about half in mean square each layer, which the code shows (1.09e-08 after 50 layers).
- Glorot: the average of the forward rule ($1/n_{\text{in}}$) and the backward rule ($1/n_{\text{out}}$), so the gradient going back through [[Backpropagation]] is also kept steady. Its uniform form is $\pm\sqrt{6/(n_{\text{in}}+n_{\text{out}})}$, because a uniform spread over $\pm a$ has variance $a^2/3$.
- **Matters:** when $n_{\text{in}}=n_{\text{out}}$, Glorot equals LeCun.

**Example (tiny numbers):** $n_{\text{in}}=200$, $n_{\text{out}}=100$.
- Glorot: $2/(200+100)=\mathbf{0.00667}$. The uniform limit is $\sqrt{6/300}=\sqrt{0.02}=\mathbf{0.1414}$, and $0.1414^2/3=0.00667$, which matches.
- He: $2/200=\mathbf{0.01}$, so the standard deviation is $\mathbf{0.1}$.
- Meaning: the wider the layer's input, the smaller its starting weights. The code printed 0.1414, 0.0067 and 0.0067.

**Compounding through depth.** Each layer multiplies the signal's variance by about a fixed factor $g=n_{\text{in}}\operatorname{Var}(w)\,c$, where $c=1$ for a linear or small-input tanh layer and $c=\tfrac12$ for ReLU (the page and the code agree on the half). After $L$ layers:

$$
\operatorname{Var}\big(h_L\big)\approx g^{L}\,\operatorname{Var}\big(h_0\big)
$$

**Where:**
- $h_0$ is the input, $h_L$ is the output of layer $L$, and $g$ is the per-layer growth factor.
- **Matters:** only $g=1$ is safe. Any other value is raised to the power $L$, so a small mistake becomes huge or tiny at depth. This is also why the same mistake hurts a 50-layer network far more than a 3-layer one.
- The formula is a rough guide: tanh saturates, so very large $g$ stops growing (the code's tanh with std 0.2 gave 0.849, not an explosion).

**Example (tiny numbers):** $L=50$ layers.
- $g=0.9$: $0.9^{50}=\mathbf{0.0052}$. So the variance fades to about half a percent.
- $g=1.1$: $1.1^{50}=\mathbf{117.4}$.
- $g=1$: $1^{50}=\mathbf{1}$.
- Meaning: 10% off per layer is a 117-times blow-up or a fade to about 1/190 over 50 layers. The code printed 0.0052 and 117.4.

**Further notes:**
- **Why training broke, in the code.** With a start std of 0.5 the 10-layer network's loss became NaN (numbers overflowed). With 0.01 the signal faded and the loss stayed at 2.302, which equals $\ln 10$, the loss of guessing evenly among 10 classes. With $\sqrt{2/64}$ it learned (0.24 to 0.55). That is 3 seeds on one small task.
- **Why 0.365, not 1, for He.** The He rule keeps the *mean square* steady (the code's single-layer check gave 0.99). The table reports the standard deviation of the last layer, measured around its own mean, so it need not be 1. I did not check this further.
- **Not covered:** the formal derivations (the He and Glorot papers do these), how the rule changes for convolution layers, residual networks and Transformers, and what happens to the gradients. The code is a forward-pass test plus one small training run.

**What came before, and how this improved on it:**
- **Before:** the Wikipedia page says random initialization goes back to Rosenblatt's perceptrons, that LeCun et al. (1998) was an early work on it, and that before the 2010s deep networks were often started by unsupervised pre-training layer by layer (for example deep belief networks), because training them directly with backpropagation was hard.
- **Its limit:** a fixed random range (or a rule not matched to the activation) either fades or blows up with depth. The Glorot and Bengio abstract says the logistic sigmoid is unsuited to deep networks with random initialization, and that they propose a scheme with substantially faster convergence.
- **What changed:** Glorot and Bengio (AISTATS 2010, PMLR v9, abstract read) proposed the variance-balancing scheme now called Glorot or Xavier. He, Zhang, Ren and Sun (arXiv 1502.01852, 6 February 2015, abstract read) derived a rule for rectifier networks, which the abstract says "enables us to train extremely deep rectified models directly from scratch", and report 4.94% top-5 error on ImageNet, reported as the first to pass the 5.1% human-level figure they cite.
- **How much better:** in the code's 10-layer ReLU digits network, a too-small start (0.01) left the loss at 2.302 after 100 steps in all three seeds (no learning), a too-big start (0.5) gave NaN in all three, and the He start reached 0.552, 0.36 and 0.242. The page notes that normalization layers and adaptive optimizers have since made initialization matter less, though not unimportant.

## Related
- [[Neural Network]] · [[Vanishing and Exploding Gradients]] · [[Activation Function]] · [[ReLU]] · [[Batch Normalization]] · [[Layer Normalization]] · [[Normalization]] · [[Residual Connection]] · [[ResNet]] · [[Backpropagation]] · [[Gradient Descent]] · [[Learning Rate]] · [[Variance]] · [[Expected Value]] · [[Normal Distribution]]

## Sources
- Wikipedia, *Weight initialization* (read in full: the methods, the zero-initialization symmetry statement and the history section)
- Glorot and Bengio, *Understanding the difficulty of training deep feedforward neural networks*, AISTATS 2010 (PMLR v9; abstract read, the paper body was not)
- He, Zhang, Ren, Sun, *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*, arXiv 1502.01852 (abstract read, the body was not; the title is from memory)
- The code and its numbers are my own, run in this session.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/neural-nets Weight initialization:::Choosing the starting numbers of a network. If all start the same, every worker does the same job. If they start too big the signal explodes through the layers, and if too small it fades away.

#flashcards/ml/formula #flashcards/topic/neural-nets Write the variance of a neuron's sum and explain each symbol.
?
$$\operatorname{Var}(z)=n_{\text{in}}\,\operatorname{Var}(w)\,\mathbb{E}[x^{2}],\qquad z=\sum_{j=1}^{n_{\text{in}}}w_j x_j$$
- $z$: the neuron's sum before the activation. $n_{\text{in}}$: how many inputs it adds up (the fan-in).
- $\operatorname{Var}(w)$: the variance of one weight. $\mathbb{E}[x^2]$: the average squared size of an input.
- The sum adds $n_{\text{in}}$ separate terms, so the variance grows with $n_{\text{in}}$. To keep $\operatorname{Var}(z)$ steady, $\operatorname{Var}(w)$ must shrink as $n_{\text{in}}$ grows.

#flashcards/ml/formula #flashcards/topic/neural-nets Write the standard initialization variances and explain each symbol.
?
$$\operatorname{Var}(w)=\frac{1}{n_{\text{in}}}\ \text{(LeCun)},\qquad \frac{2}{n_{\text{in}}}\ \text{(He, ReLU)},\qquad \frac{2}{n_{\text{in}}+n_{\text{out}}}\ \text{(Glorot)}$$
- $n_{\text{in}}$: the number of inputs. $n_{\text{out}}$: the number of outputs of the layer (fan-out).
- LeCun: keeps $\operatorname{Var}(z)=\mathbb{E}[x^2]$ in the equation above.
- He: ReLU sets negative values to 0, which halves the mean square of the output. The factor 2 puts it back.
- Glorot: a compromise that also keeps the backward signal steady, using both fan-in and fan-out.

#flashcards/ml/formula #flashcards/topic/neural-nets Write how a wrong start grows with depth and explain each symbol.
?
$$\operatorname{Var}\big(h_L\big)\approx g^{L}\,\operatorname{Var}\big(h_0\big)$$
- $h_0$: the input. $h_L$: the output of layer $L$. $g$: the per-layer growth factor.
- Only $g=1$ is safe. Any other value is raised to the power $L$, so a small mistake becomes huge or tiny at depth.
- That is why the same mistake hurts a 50-layer network far more than a 3-layer one.

#flashcards/ml/cloze #flashcards/topic/neural-nets He initialization uses variance ==2/n_in== because ReLU throws away about ==half== of the signal.

#flashcards/ml/example #flashcards/topic/neural-nets $n_{\text{in}}=100$ inputs, $\mathbb{E}[x^2]=1$. What is $\operatorname{Var}(z)$ for $\operatorname{Var}(w)=0.01$ and $\operatorname{Var}(w)=1$?::0.01: 100 × 0.01 × 1 = 1. Weight variance 1: 100 × 1 × 1 = 100. A weight variance of 1 makes the layer 100 times too loud.

#flashcards/ml/example #flashcards/topic/neural-nets For $n_{\text{in}}=200$, $n_{\text{out}}=100$, what are the Glorot and He variances?::Glorot: 2/300 = 0.00667 (uniform limit √(6/300) = 0.1414, and 0.1414²/3 = 0.00667). He: 2/200 = 0.01, standard deviation 0.1. The wider the input, the smaller the weights.

#flashcards/ml/example #flashcards/topic/neural-nets Over 50 layers, what are $g^{50}$ for $g=0.9$, $1.1$ and $1$?::0.9⁵⁰ = 0.0052, 1.1⁵⁰ = 117.4, 1⁵⁰ = 1. Being 10% off per layer is a 117-times blow-up or a fade to about 1/190.
