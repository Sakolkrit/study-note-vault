---
tags: [normalization, transformer, LLM, layer-norm, RMSNorm, training]
status: growing
level: 4
tier: 2
---

# Layer Normalization

Rescales the numbers inside **one token's vector** so they have a steady size, which keeps deep [[Transformer]]s trainable. **RMSNorm** is a lighter version used by many modern models.

## 🟢 1. Simple (age 12)
When many calculations are chained, numbers can drift to be huge or tiny, and the next step struggles. Layer normalization is a **volume knob that resets each message to a comfortable level** before passing it on: it shifts and stretches the list of numbers so they are centred and about the same size every time.

**Example:** Five friends pass a message along a line, each one speaking louder or quieter by accident. Layer norm is a helper who, before each pass, turns the volume to "medium" so the last friend can still hear.

## 🔵 2. How it works
1. Each token in a [[Transformer]] is a list of numbers (a vector, see [[Vector]]), for example 768 numbers long.
2. For **one token at a time**, work out the **mean** of its numbers and their **spread** (standard deviation).
3. Subtract the mean and divide by the spread. Now this token's numbers have mean 0 and spread 1.
4. Multiply by a learned **scale** and add a learned **shift** (one pair per position in the vector), so the model can undo the normalisation if it helps.
5. Because each token is handled alone, the result does **not depend on the other tokens or the batch size**. That is the difference from batch normalization ([[Batch Normalization]], overview in [[Normalization]]), which uses statistics across a batch.
6. **RMSNorm** skips the mean: it only divides by the root-mean-square size of the numbers. It is cheaper and works about as well, according to its paper's abstract.
7. **Where it goes:** in the original Transformer the normalisation comes *after* the residual addition (post-norm). Today most models put it *before* the attention or feed-forward part (pre-norm), and the lecture says the default choice nowadays is **pre-norm with RMSNorm**.

## 🟠 3. Technical detail
- **Normalised over:** the feature dimension of a single token. The code checks it matches PyTorch's `LayerNorm` and that a token gives the same result alone as inside a batch.
- **Learned numbers:** LayerNorm has a scale and a shift for each feature ($2d$ numbers); RMSNorm has only a scale ($d$ numbers). At width 4096 that is 8,192 against 4,096.
- **Invariances (code):** LayerNorm ignores adding a constant to all the inputs and multiplying them by a constant. RMSNorm ignores only the multiplication, since it never subtracts the mean.
- **Why it helps (lecture):** it helps training stability and convergence. The Layer Normalization abstract says it can substantially reduce training time, does the same computation at training and test time, and works easily with recurrent networks (arXiv 1607.06450, read).
- **Post-norm and pre-norm (lecture):** the lecture cites a study of the Transformer's normalisation placement (Xiong et al., 2020), which I did not read, and says pre-norm is what is used nowadays.
- **RMSNorm paper claim:** comparable performance with LayerNorm while reducing running time by 7% to 64% on different models (abstract, read; arXiv 1910.07467).
- **Small constant:** a tiny $\epsilon$ is added under the square root so that an all-zero vector does not cause a division by zero.
- **Not the same as** the batch-level normalisation in [[Normalization]], which is better suited to image networks like [[ResNet]].

```python
import torch
torch.manual_seed(0)
x = torch.randn(3, 8, dtype=torch.float64) * 5 + 2             # 3 tokens, 8 numbers each
eps = 1e-5
def layer_norm(x, g=1.0, b=0.0): return (x - x.mean(-1, keepdim=True)) / torch.sqrt(x.var(-1, unbiased=False, keepdim=True) + eps) * g + b
def rms_norm(x, g=1.0): return x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps) * g
ref = torch.nn.LayerNorm(8, eps=eps).double()
print('matches torch.nn.LayerNorm:', torch.allclose(layer_norm(x), ref(x)))
y = layer_norm(x)
print('after LayerNorm: mean per token', y.mean(-1).abs().max().item() < 1e-12, '| variance per token', y.var(-1, unbiased=False).round(decimals=4).tolist())
# each token is handled alone: a token's result does not depend on the other tokens or on the batch size
print('same result for one token alone:', torch.allclose(layer_norm(x[:1]), y[:1]))
# LayerNorm ignores a shift and a rescale of its input; RMSNorm ignores only a rescale
print('LayerNorm of 3x + 7 equals LayerNorm of x:', torch.allclose(layer_norm(3 * x + 7), y, atol=1e-4))
print('RMSNorm of 3x equals RMSNorm of x:', torch.allclose(rms_norm(3 * x), rms_norm(x), atol=1e-4), '| RMSNorm of x + 7 equals RMSNorm of x:', torch.allclose(rms_norm(x + 7), rms_norm(x), atol=1e-4))
# learned scale and shift: 2 * 8 numbers for LayerNorm, 8 for RMSNorm
print('learned numbers for width 4096: LayerNorm', 2 * 4096, '| RMSNorm', 4096)
# worked example for one token of 3 numbers [1, 2, 6]
t = torch.tensor([1., 2., 6.], dtype=torch.float64)
print('token [1,2,6]: mean', t.mean().item(), '| variance', t.var(unbiased=False).item(), '| LayerNorm', layer_norm(t).round(decimals=3).tolist(), '| RMS', round(((t ** 2).mean()).sqrt().item(), 3), '| RMSNorm', rms_norm(t).round(decimals=3).tolist())
```

## 🔴 4. Deep / formal
**Layer normalization.** For one token's vector $x=(x_1,\dots,x_d)$:

$$
\text{LN}(x)_i=\gamma_i\,\frac{x_i-\mu}{\sqrt{\sigma^{2}+\epsilon}}+\beta_i,\qquad \mu=\frac1d\sum_{j}x_j,\quad \sigma^{2}=\frac1d\sum_{j}(x_j-\mu)^{2}
$$

**Where:**
- $x_i$ is the $i$-th number of the token's vector and $d$ is its length.
- $\mu$ is the mean of the numbers and $\sigma^2$ their variance (the average squared distance from the mean).
- $\epsilon$ is a tiny constant (such as $10^{-5}$) that keeps the division safe.
- $\gamma_i$ and $\beta_i$ are the learned scale and shift for position $i$.
- **Matters:** subtracting $\mu$ and dividing by $\sqrt{\sigma^2}$ gives every token the same average and spread before the learned part, so the next layer always sees inputs on a similar scale. Without $\gamma$ and $\beta$ the layer could not choose a different scale if that helped.

**Example (tiny numbers):** one token $x=(1,2,6)$, with $\gamma=1$, $\beta=0$.
- Mean: $\mu=(1+2+6)/3=\mathbf{3}$. Variance: $((-2)^2+(-1)^2+3^2)/3=14/3=\mathbf{4.667}$, so the spread is $2.160$.
- Results: $(1-3)/2.160=\mathbf{-0.926}$, $(2-3)/2.160=\mathbf{-0.463}$, $(6-3)/2.160=\mathbf{1.389}$.
- Meaning: the new numbers have mean 0 and variance 1, whatever the size of the originals.

**RMSNorm.** Drop the mean, and divide by the root-mean-square:

$$
\text{RMSNorm}(x)_i=\gamma_i\,\frac{x_i}{\sqrt{\tfrac1d\sum_{j}x_j^{2}+\epsilon}}
$$

**Where:**
- $\sqrt{\tfrac1d\sum_jx_j^2}$ is the root mean square (RMS): the typical size of the numbers, ignoring signs.
- $\gamma_i$ is the learned scale, and there is no shift.
- $\epsilon$ is the same tiny safety constant.
- **Matters:** RMSNorm only controls the *size* of the vector, not its average. It is slightly cheaper (no mean to compute) and the paper hypothesises that re-centring is not needed. It no longer ignores a constant added to all numbers (code: false for $x+7$).

**Example (tiny numbers):** the same token $(1,2,6)$.
- $\text{RMS}=\sqrt{(1+4+36)/3}=\sqrt{13.667}=\mathbf{3.697}$.
- Results: $1/3.697=\mathbf{0.271}$, $2/3.697=\mathbf{0.541}$, $6/3.697=\mathbf{1.623}$.
- Meaning: the shape of the vector is kept (the ratios $1:2:6$) and only its size is reset, so the mean is not zero.

**Further notes:**
- **Placement:** with pre-norm, the residual stream itself is never normalised inside the block; only the input to each sub-layer is. Why that trains more stably is studied in the cited paper; I did not run a comparison.
- **Not tested:** speed (the 7%–64% figure is from the abstract), effects on training, and the post-norm against pre-norm comparison.
- **Not covered:** other norm variants (group norm, instance norm), and partial RMSNorm.

**What came before, and how this improved on it:**
- **Before:** batch normalization (Ioffe and Szegedy, 2015, see [[Normalization]]), which uses the mean and variance across a batch of examples.
- **Its limit:** the Layer Normalization abstract says batch normalization depends on the mini-batch size and it is not obvious how to apply it to recurrent networks.
- **What changed:** compute the statistics over the features of a single example, so training and test do the same computation (Ba et al., arXiv 1607.06450, 21 July 2016, abstract read). RMSNorm then dropped the mean (Zhang and Sennrich, arXiv 1910.07467, 16 October 2019, abstract read).
- **How much better:** the RMSNorm abstract reports 7% to 64% less running time at comparable performance, which I did not reproduce. In this note's code: a token's result is the same alone or in a batch (batch norm would not), and RMSNorm holds 4,096 learned numbers at width 4096 against 8,192 for LayerNorm.

## Related
- [[Normalization]] · [[Transformer]] · [[Residual Connection]] · [[Self-Attention]] · [[LLM]] · [[Recurrent Neural Network]] · [[ResNet]] · [[BERT]] · [[Vector]] · [[Feed-Forward Network]]

## Sources
- Stanford CME 295 (Fall 2025), Lecture 2 slides, Afshine and Shervine Amidi
- Ba, Kiros and Hinton, *Layer Normalization*, 2016 (arXiv 1607.06450, abstract read)
- Zhang and Sennrich, *Root Mean Square Layer Normalization*, 2019 (arXiv 1910.07467, abstract read)
- Xiong et al., *On Layer Normalization in the Transformer Architecture*, 2020 (named in the lecture; not read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/training Layer normalization:::A volume knob that resets each token's list of numbers to a comfortable level before passing it on. It shifts and stretches the numbers so they are centred and about the same size every time.

#flashcards/ml/formula #flashcards/topic/training Write layer normalization and explain each symbol.
?
$$\text{LN}(x)_i=\gamma_i\,\frac{x_i-\mu}{\sqrt{\sigma^{2}+\epsilon}}+\beta_i,\qquad \mu=\frac1d\sum_{j}x_j,\quad \sigma^{2}=\frac1d\sum_{j}(x_j-\mu)^{2}$$
- $x_i$: the $i$-th number of the token's vector. $d$: its length.
- $\mu$: the mean of the numbers. $\sigma^2$: their variance (average squared distance from the mean).
- $\epsilon$: a tiny constant (such as $10^{-5}$) that keeps the division safe.
- $\gamma_i$, $\beta_i$: the learned scale and shift for each position, so the network can undo the normalising if that helps.
- The statistics come from the one token only, never from other examples.

#flashcards/ml/formula #flashcards/topic/training Write RMSNorm and explain each symbol.
?
$$\text{RMSNorm}(x)_i=\gamma_i\,\frac{x_i}{\sqrt{\tfrac1d\sum_{j}x_j^{2}+\epsilon}}$$
- $\sqrt{\tfrac1d\sum_jx_j^2}$: the root mean square (RMS), the typical size of the numbers ignoring signs.
- $\gamma_i$: the learned scale. There is no shift.
- $\epsilon$: the same tiny safety constant.
- RMSNorm only controls the size of the vector, not its average. It is slightly cheaper (no mean to subtract).

#flashcards/ml/cloze #flashcards/topic/training Layer normalization uses statistics from ==one token== only, unlike batch normalization, which uses the ==whole batch==.

#flashcards/ml/example #flashcards/topic/training For $x=(1,2,6)$, $\gamma=1$, $\beta=0$, what does layer normalization give?::Mean 3, variance 14/3 = 4.667, spread 2.160. Results (−2, −1, 3)/2.160 = −0.926, −0.463, 1.389. Mean 0 and variance about 1.

#flashcards/ml/example #flashcards/topic/training For the same $x=(1,2,6)$, what does RMSNorm give?::RMS = √((1 + 4 + 36)/3) = 3.697. Results 0.271, 0.541, 1.623. The ratios 1:2:6 are kept and only the size is reset, so the mean is not zero.
