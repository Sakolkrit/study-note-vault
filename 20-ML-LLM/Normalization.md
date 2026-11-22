---
tags: [normalization, layernorm, batchnorm, rmsnorm, neural-network, transformer, LLM, deep-learning]
status: growing
level: 4
tier: 2
---

# Normalization

## 🟢 1. Simple (age 12)
Normalization means **rescaling numbers so they are all in a similar, comfortable range**. It is like converting every runner's time to "how many steps faster or slower than average", so numbers stay tidy and nothing gets wildly big or tiny.

**Example:** Three friends compare their skills. One scores in thousands (video-game points), one in tens (spelling marks), one in decimals (a ratio). You cannot easily compare or add them. If you turn each score into "how far above or below that friend's usual score", they all become small numbers around 0, and they are easy to compare. A neural network does the same inside itself so its numbers do not get out of hand.

## 🔵 2. How it works
1. Inside a deep [[Neural Network]], numbers are multiplied and added layer after layer. Without care, they can drift to huge values or shrink to almost 0.
2. Huge or tiny numbers make learning unstable: the slopes that [[Backpropagation]] computes explode or vanish, and the learning step in [[Gradient Descent]] becomes a bad one.
3. A **normalization layer** is added between steps. It takes a group of numbers, finds their **average** and their **spread**, subtracts the average, and divides by the spread. Now the group has average about 0 and spread about 1.
4. Then it gives the model two **learnable dials** (a scale and a shift) so the network can undo the normalisation where that helps. The model gets stability without losing freedom.
5. The kinds differ in **which group of numbers** is normalised together:
   - **Batch normalization (BatchNorm):** one feature, across many examples in a batch. Common in image models. Full note: [[Batch Normalization]].
   - **Layer normalization (LayerNorm):** all the features of one example (for one word in a [[Transformer]]). The standard in text models.
   - **RMSNorm:** a lighter version of LayerNorm that skips subtracting the average. Common in modern [[LLM]]s.

No image: the only Wikimedia Commons file I found for the term "batch normalization" is a screenshot of brain-imaging software, unrelated to this topic, so I left it out. The [[Transformer]] note shows where the "Norm" boxes sit.

## 🟠 3. Technical detail
- **Where it goes in a Transformer:** a normalisation layer sits around each [[Attention]] block and each feed-forward block. Modern models usually normalise *before* the block ("pre-norm"), which makes very deep stacks train more stably than normalising after.
- **BatchNorm vs LayerNorm:**
  - BatchNorm uses statistics from the whole batch, so it behaves differently in training and when running the model, and works poorly with tiny batches. It must keep running averages for use at run time.
  - LayerNorm uses only the numbers of a single example, so it behaves identically in training and inference and does not care about batch size. That is why text models use it.
- **The learnable dials:** $\gamma$ (scale) and $\beta$ (shift), one pair per feature. They are trained with the other weights (for example with the [[Adam Optimizer]]).
- **Benefits:** faster and steadier training, tolerance for bigger learning rates, and less sensitivity to the starting weights.
- **Cost:** it adds a small amount of work and memory traffic per layer (see [[GPU]]); fused kernels reduce that.
- **Other kinds:** GroupNorm (groups of features, used in some image models), and weight normalization (rescales the weights instead of the activations).

```python
import numpy as np
def layernorm(x, g=1.0, b=0.0, eps=1e-5):
    return g * (x - x.mean()) / np.sqrt(x.var() + eps) + b

x = np.array([2., 4., 6., 8.])
print(layernorm(x).round(3))            # [-1.342 -0.447  0.447  1.342]
print(layernorm(x, 2, 1).round(3))      # [-1.683  0.106  1.894  3.683]

rms = lambda x, g=1.0, eps=1e-5: g * x / np.sqrt((x ** 2).mean() + eps)
print(rms(np.array([3., 4.])).round(3)) # [0.849 1.131]

X = np.array([[1., 2.], [3., 6.]])      # 2 examples (rows) x 2 features (columns)
print(X.mean(axis=1), X.mean(axis=0))   # [1.5 4.5] [2. 4.]  layer-style vs batch-style groups
```

## 🔴 4. Deep / formal
**Layer normalization.** For one example with $d$ features $x_1,\dots,x_d$ (for example the numbers describing one token):

$$
\mu=\frac1d\sum_{j=1}^{d}x_j,\qquad \sigma^2=\frac1d\sum_{j=1}^{d}(x_j-\mu)^2,\qquad
y_j=\gamma_j\,\frac{x_j-\mu}{\sqrt{\sigma^{2}+\epsilon}}+\beta_j
$$

**Where:**
- $x_j$ is the $j$-th input number and $d$ is how many numbers are in the group (the width of the layer). $y_j$ is the output.
- $\mu$ is the **mean** of the group: add them up and divide by $d$.
- $\sigma^2$ is the **variance**: the average squared distance from the mean, a measure of spread. $\sigma=\sqrt{\sigma^2}$ is the standard deviation (see [[Normal Distribution]]).
- $x_j-\mu$ is "how far above or below average". **Matters:** subtracting the mean re-centres the group at 0, so a constant offset in the numbers has no effect.
- Dividing by $\sqrt{\sigma^2+\epsilon}$ rescales the spread to about 1. **Matters:** it stops the numbers from growing or shrinking from layer to layer.
- $\epsilon$ is a tiny constant (such as $10^{-5}$). **Matters:** without it, a group whose numbers are all equal would give $\sigma=0$ and a division by zero.
- $\gamma_j$ (scale) and $\beta_j$ (shift) are **learned** numbers. **Matters:** they let the network re-stretch or move the normalised values if that works better. Without them, every layer would be forced to output zero-mean, unit-spread numbers.

**Example (tiny numbers):** $x=(2,4,6,8)$, so $d=4$. Ignore $\epsilon$ (it is tiny), and use $\gamma=1$, $\beta=0$ first.
- Mean: $\mu=(2+4+6+8)/4=\mathbf{5}$.
- Variance: $\big((-3)^2+(-1)^2+1^2+3^2\big)/4=20/4=\mathbf{5}$, so $\sigma=\sqrt5=2.236$.
- Normalised: $(2-5)/2.236=-1.342$, $(4-5)/2.236=-0.447$, $0.447$, $1.342$. So $y=(\mathbf{-1.342},\,\mathbf{-0.447},\,\mathbf{0.447},\,\mathbf{1.342})$. Their mean is 0 and their variance is 1.
- Now let the model use $\gamma=2$, $\beta=1$: $y=2\cdot(\text{those})+1=(\mathbf{-1.683},\,\mathbf{0.106},\,\mathbf{1.894},\,\mathbf{3.683})$.
- Meaning: whatever the scale of the input (try $(20,40,60,80)$: you get the same normalised values), the layer hands the next step numbers in a steady range. The dials $\gamma$ and $\beta$ then set the range the network actually wants.

**RMSNorm.** A cheaper cousin that skips the mean and rescales by the root-mean-square size only:

$$
y_j=\gamma_j\,\frac{x_j}{\sqrt{\frac1d\sum_{k=1}^{d}x_k^{2}+\epsilon}}
$$

**Where:**
- $x_j$, $d$, $\gamma_j$ and $\epsilon$ are as above ($\beta$ is dropped).
- $\frac1d\sum_k x_k^2$ is the average of the squares: the typical squared size of the numbers. Its square root is the **RMS** (root mean square), a typical size.
- Dividing each number by the RMS scales the group so its typical size is about 1. **Matters:** this keeps the stabilising effect but saves the work of computing and subtracting the mean, and it is a little faster. Many recent [[LLM]]s use it (stated from memory; check the specific model).

**Example (tiny numbers):** $x=(3,4)$, $\gamma=1$.
- Average of squares: $(9+16)/2=\mathbf{12.5}$. RMS $=\sqrt{12.5}=\mathbf{3.536}$.
- Outputs: $3/3.536=\mathbf{0.849}$ and $4/3.536=\mathbf{1.131}$.
- Meaning: the pattern (the 3-to-4 ratio) is kept, but the overall size is now about 1.

**Batch normalization** uses the same formula but computes $\mu$ and $\sigma^2$ for **one feature across the batch** (down a column of the table in the code) instead of across the features of one example (along a row). In the small table above, a layer-style group is a row, with means $1.5$ and $4.5$, and a batch-style group is a column, with means $2$ and $4$.

**Further notes:**
- **Why it works is debated.** The original BatchNorm paper argued it reduces "internal covariate shift" (the inputs of each layer changing as earlier layers learn). Later work argued the main effect is a smoother loss surface that makes [[Gradient Descent]] easier. (From memory of the literature; the debate is real.)
- **Scale invariance:** because the output ignores the input's overall scale, the layer's weights can grow without changing the output, which interacts with weight decay and learning rates (see [[Overfitting]]).
- **Residual connections:** pre-norm stacks keep a clean "skip path" from input to output, which helps gradients flow through hundreds of layers.
- **Cost of a mistake:** forgetting to switch BatchNorm to evaluation mode when running a model is a classic bug (it keeps using batch statistics).

**What came before, and how this improved on it:**
- **Before:** careful choice of starting numbers, small learning rates, and scaling the inputs once at the start. Each layer's inputs kept changing as the earlier layers learned, which slowed training.
- **Its limit:** deep networks trained slowly and could not take large learning rates safely.
- **What changed:** Batch Normalization (Ioffe and Szegedy, arXiv 1502.03167, 11 February 2015, abstract page read) normalises each layer's inputs over the batch during training. Its abstract reports matching the accuracy of a leading image classifier with 14 times fewer training steps, and an ensemble reaching 4.9% top-5 validation error on ImageNet.
- **How much better:** a toy check of my own (below): a plain network of 10 hidden layers on 1,000 handwritten 8×8 digits needed 170 to 316 Adam steps (mean 218.2) for the training loss to fall under 0.5, and the same network with BatchNorm needed 29 to 33 (mean 30.0), about 7 times fewer steps. This is one small case, so the size of the gain will differ elsewhere.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
torch.set_num_threads(1)
D = load_digits(); X = torch.tensor(D.data / 16.0, dtype=torch.float32); y = torch.tensor(D.target.astype(np.int64))
idx = np.random.RandomState(0).permutation(len(y)); small, te = idx[:100], idx[1000:]
acc = lambda m, ids: float((m(X[ids]).argmax(1) == y[ids]).float().mean())
# BatchNorm: 10 hidden layers of width 32; how many Adam steps until the training loss first falls below 0.5?
def bn_run(bn, seed, cap=400):
    torch.manual_seed(seed); layers = [nn.Linear(64, 32)]
    for _ in range(9): layers += [nn.BatchNorm1d(32)] * bn + [nn.ReLU(), nn.Linear(32, 32)]
    m = nn.Sequential(*layers, nn.ReLU(), nn.Linear(32, 10)); opt = torch.optim.Adam(m.parameters(), 0.003); xs, ys = X[idx[:1000]], y[idx[:1000]]
    for step in range(1, cap + 1):
        opt.zero_grad(); l = F.cross_entropy(m(xs), ys); l.backward(); opt.step()
        if l.item() < 0.5: return step
    return None
for bn in (0, 1):
    print('BatchNorm' if bn else 'plain    ', '| steps until loss < 0.5 for seeds 0 to 4 (None = not within 400):', [bn_run(bn, s) for s in range(5)])
```
Output:
- `plain     | steps until loss < 0.5 for seeds 0 to 4 (None = not within 400): [316, 183, 199, 223, 170]`
- `BatchNorm | steps until loss < 0.5 for seeds 0 to 4 (None = not within 400): [29, 29, 29, 30, 33]`

## Related
- [[Neural Network]] · [[Transformer]] · [[Attention]] · [[Backpropagation]] · [[Gradient Descent]] · [[Adam Optimizer]] · [[Normal Distribution]] · [[Overfitting]] · [[LLM]] · [[GPU]] · [[Positional Encoding]] · [[Vanishing and Exploding Gradients]] · [[Residual Connection]] · [[Convolutional Neural Network]] · [[k-NN]] · [[Support Vector Machine]] · [[ResNet]] · [[Layer Normalization]] · [[Feature Scaling]]

## Sources
- Ioffe & Szegedy, *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*, 2015
- Ba, Kiros & Hinton, *Layer Normalization*, 2016
- Zhang & Sennrich, *Root Mean Square Layer Normalization*, 2019 (from memory)
- Santurkar et al., *How Does Batch Normalization Help Optimization?*, 2018 (from memory)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/training Normalization:::Rescaling numbers so they are all in a similar, comfortable range, like converting every runner's time to "how many steps faster or slower than average".

#flashcards/ml/formula #flashcards/topic/training Write the general normalization and explain each symbol.
?
$$\mu=\frac1d\sum_{j=1}^{d}x_j,\qquad \sigma^2=\frac1d\sum_{j=1}^{d}(x_j-\mu)^2,\qquad y_j=\gamma_j\,\frac{x_j-\mu}{\sqrt{\sigma^{2}+\epsilon}}+\beta_j$$
- $x_j$: the $j$-th input number. $d$: how many numbers are in the group (the width of the layer). $y_j$: the output.
- $\mu$: the mean of the group. $\sigma^2$: the variance, the average squared distance from the mean. $\sigma=\sqrt{\sigma^2}$ is the spread.
- $\epsilon$: a tiny number that avoids dividing by zero.
- $\gamma_j$, $\beta_j$: learned scale and shift, so the network can undo the normalising if that helps.
- Which numbers form "the group" is what separates batch norm, layer norm and others.

#flashcards/ml/formula #flashcards/topic/training Write RMS normalization and explain each symbol.
?
$$y_j=\gamma_j\,\frac{x_j}{\sqrt{\frac1d\sum_{k=1}^{d}x_k^{2}+\epsilon}}$$
- $x_j$, $d$, $\gamma_j$, $\epsilon$: as above ($\beta$ is dropped).
- $\frac1d\sum_k x_k^2$: the average of the squares. Its square root is the RMS, a typical size.
- Dividing each number by the RMS scales the group so its typical size is about 1, and keeps the pattern (the ratios) intact.

#flashcards/ml/cloze #flashcards/topic/training Normalization subtracts the ==mean== and divides by the ==spread==, then applies a learned scale and shift.

#flashcards/ml/example #flashcards/topic/training For $x=(2,4,6,8)$, what is the normalised output (with $\gamma=1$, $\beta=0$)?::Mean 5. Variance (9 + 1 + 1 + 9)/4 = 5, σ = 2.236. Outputs −1.342, −0.447, 0.447, 1.342.

#flashcards/ml/example #flashcards/topic/training For $x=(3,4)$, what does RMS normalization give?::Average of squares (9 + 16)/2 = 12.5, RMS = 3.536. Outputs 0.849 and 1.131. The 3-to-4 ratio is kept and the overall size is about 1.
