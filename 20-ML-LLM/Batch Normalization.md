---
tags: [normalization, batchnorm, neural-network, deep-learning, training, CNN]
status: growing
level: 4
tier: 2
---

# Batch Normalization

Often shortened to **BatchNorm** or **batch norm**. A layer that rescales each feature using the **average and spread of the whole batch** of examples, so deep networks train faster and more steadily. This note is the close-up on BatchNorm. The overview of all the kinds (BatchNorm, LayerNorm, RMSNorm) is in [[Normalization]], and the version used inside text models is [[Layer Normalization]].

## 🟢 1. Simple (age 12)
Imagine a class where every student takes a test, and the teacher marks each score as "how far above or below the **class average** you were". A score of 70 means nothing alone, but "10 points above the class average" is easy to compare from test to test. BatchNorm does this inside a network: for each number the network computes, it looks at the **whole group of examples being processed together** and rescales the number relative to that group's average.

**Example:** Four kids run a race and take 1, 2, 6 and 3 minutes. The average is 3 minutes. Now the times become "how far from average": two minutes faster, one faster, three slower, and exactly average. Those are small, tidy numbers, even if the race had been measured in seconds instead.

## 🔵 2. How it works
1. A [[Neural Network]] is trained on a **batch** of examples at a time (for example 32 or 256 images), not one by one.
2. At a BatchNorm layer, take **one feature** (one number per example) and look at its values across the whole batch.
3. Work out the batch's **average** and **spread** (see [[Variance]]). Subtract the average, divide by the spread. Now that feature has average 0 and spread 1 across the batch.
4. Then multiply by a learned **scale** and add a learned **shift** for each feature, so the network can undo the normalising if that helps. Each feature has its own pair.
5. **Every example is affected by the others.** Because the average comes from the whole batch, the output for one example changes if its batch mates change. That is a big difference from [[Layer Normalization]], which looks only at one example.
6. **Running averages for later.** When you use the finished network on a single new example there is no batch. So during training the layer keeps a **running average** of the batch averages and spreads, and at test time it uses those fixed numbers instead. Training mode and test mode therefore give different answers on purpose.
7. **Small batches are a problem.** A batch of 2 gives a poor guess of the average, and a batch of 1 has no spread at all (see the code).
8. **After training it can disappear.** At test time the layer is just a fixed multiply-and-add, so it can be merged into the layer before it, making the finished network a little cheaper.
9. **Why it helps** is partly still debated (see the deep section). What is clear: it keeps the numbers from drifting huge or tiny from layer to layer, and it lets networks use bigger learning steps ([[Learning Rate]]).

No image: the only Wikimedia Commons file I found for "batch normalization" is a screenshot of brain-imaging software (the SPM batch editor), which has nothing to do with this topic, so I left it out. The same dead end is noted in [[Normalization]].

## 🟠 3. Technical detail
- **Where it goes:** usually between a linear or convolution layer and the activation ([[Activation Function]], [[ReLU]]), mostly in image networks such as [[ResNet]] and other [[Convolutional Neural Network]]s. Text models use [[Layer Normalization]] instead.
- **Statistics per feature:** one mean and one variance for each feature, taken down the batch. In a [[Convolutional Neural Network]] the usual choice is one per channel, pooled over the batch and all positions (stated from memory, not run here).
- **Learned numbers:** a scale $\gamma$ and a shift $\beta$ per feature, trained with the other weights (for example with the [[Adam Optimizer]]). The running mean and variance are **not** learned by [[Backpropagation]]; they are updated by a moving average on every training batch.
- **Training mode vs eval mode:** in training the layer uses the current batch. In evaluation it uses the stored running numbers. Forgetting to switch (for example calling the model in training mode on a tiny test batch) is a classic bug, and the code shows why.
- **Batch of 1:** my formula gives a constant output (just the shift). PyTorch refuses and raises an error.
- **Other side effects (Wikipedia page):** it tends to allow a higher learning rate, makes the network less sensitive to the starting numbers ([[Weight Initialization]]), and acts a bit like a regulariser, in some cases reducing the need for [[Dropout]] (see [[Overfitting]]). The page also notes that with BatchNorm the items in a batch are no longer independent, which can make the gradient estimate worse.
- **Cost:** a little extra work per layer, and extra memory for the stored running numbers. It can be merged away at inference (see the code).

```python
import torch
import torch.nn as nn
torch.manual_seed(0)
eps = 1e-5

def bn_train(x, g=1.0, b=0.0):               # x: (batch, features); statistics down each column
    mu, var = x.mean(0), x.var(0, unbiased=False)
    return g * (x - mu) / torch.sqrt(var + eps) + b

# 1) a hand-sized case: one feature, a batch of 4 examples
x = torch.tensor([[1.], [2.], [6.], [3.]], dtype=torch.float64)
print("batch [1,2,6,3]: mean", x.mean().item(), "| variance", x.var(unbiased=False).item(), "| output", bn_train(x).flatten().round(decimals=3).tolist())
bn = nn.BatchNorm1d(1, eps=eps).double(); bn(x)
print("running mean/var after one batch (momentum 0.1):", round(bn.running_mean.item(), 3), round(bn.running_var.item(), 3))
bn.eval(); print("eval-mode output for the value 6 using only those running numbers:", round(bn(torch.tensor([[6.]], dtype=torch.float64)).item(), 3))

# 2) same as PyTorch, then train mode vs eval mode
X = torch.randn(32, 5, dtype=torch.float64) * 3 + 2
bn = nn.BatchNorm1d(5, eps=eps).double()
print("matches torch.nn.BatchNorm1d:", torch.allclose(bn(X), bn_train(X)))
for _ in range(200): bn(torch.randn(64, 5, dtype=torch.float64) * 3 + 2)      # the true mean is 2 and variance 9
print("running mean (true 2):", bn.running_mean.mean().round(decimals=2).item(), "| running var (true 9):", bn.running_var.mean().round(decimals=2).item())
a = X[:1]; b1 = torch.cat([a, X[1:8]]); b2 = torch.cat([a, X[8:15]])           # the same example, two different batches
bn.train(); print("train mode, same example, two batches:", bn(b1)[0, 0].item() == bn(b2)[0, 0].item(), "| outputs", round(bn(b1)[0, 0].item(), 3), round(bn(b2)[0, 0].item(), 3))
bn.eval();  print("eval mode,  same example, two batches:", bn(b1)[0, 0].item() == bn(b2)[0, 0].item(), "| outputs", round(bn(b1)[0, 0].item(), 3), round(bn(b2)[0, 0].item(), 3))

# 3) tiny batches: the batch statistics get noisy; a batch of one has no spread at all
print("batch of 1, my formula:", bn_train(torch.tensor([[5.0]], dtype=torch.float64)).item(), "(always the shift, here 0)")
try: nn.BatchNorm1d(1)(torch.tensor([[5.0]]))
except ValueError as e: print("torch, batch of 1 in train mode:", str(e)[:50])
for m in (2, 8, 64, 512):
    out = []
    for _ in range(2000):
        v = torch.randn(m, 1, dtype=torch.float64); v[0] = 1.0                   # example 0 always has the value 1.0
        out.append(bn_train(v)[0].item())
    print(f"batch size {m:4d}: output for the same input 1.0 has std {torch.tensor(out).std().item():.3f} (mean {torch.tensor(out).mean().item():.3f})")

# 4) at inference BN is a fixed straight-line step, so it folds into the Linear layer before it
net = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3)).double()
for _ in range(100): net(torch.randn(64, 4, dtype=torch.float64) * 2 + 1)
net.eval(); lin, b = net[0], net[1]
s = b.weight / torch.sqrt(b.running_var + b.eps)
W, c = lin.weight * s[:, None], (lin.bias - b.running_mean) * s + b.bias
t = torch.randn(10, 4, dtype=torch.float64)
print("folded layer vs Linear+BatchNorm in eval mode: max difference", f"{(t @ W.T + c - net(t)).abs().max().item():.1e}", "| layers 2 -> 1")
print("by hand: y=2x+1, then BN with mean 5, var 4, scale 3, shift 1 -> 3x-5; at x=4:", 3 * (2 * 4 + 1 - 5) / 2 + 1, "=", 3 * 4 - 5)

# 5) 20 layers with a too-big or too-small start: how big are the numbers at the end?
def last_std(scale, use_bn):
    torch.manual_seed(1); h = torch.randn(256, 64)
    for _ in range(20):
        w = torch.randn(64, 64) * scale * (2 / 64) ** 0.5
        h = h @ w
        if use_bn: h = (h - h.mean(0)) / torch.sqrt(h.var(0, unbiased=False) + eps)
        h = torch.relu(h)
    return h.std().item()
for sc in (1.0, 1.5, 0.5):
    print(f"weight scale x{sc}: last-layer std without BN {last_std(sc, False):.3g} | with BN {last_std(sc, True):.3g}")
```

What it printed: for the batch 1, 2, 6, 3 the mean was 3.0, the variance 3.5 and the outputs −1.069, −0.535, 1.604 and 0.0. After that single batch the running mean was 0.3 and the running variance 1.367 (momentum 0.1), and using only those numbers the value 6 came out as 4.876. My formula matched `torch.nn.BatchNorm1d`. After 200 batches from data with true mean 2 and variance 9, the running numbers were 2.04 and 9.11. In training mode the same example gave −1.970 and −1.745 in two different batches, while in eval mode it gave −2.043 both times. A batch of 1 gave 0.0 with my formula and an error in PyTorch. For the same input 1.0, the output spread across random batches was 0.739 (batch size 2), 0.407 (8), 0.153 (64) and 0.054 (512). The merged layer matched Linear plus BatchNorm in eval mode to 2.2e-16. In the 20-layer test, with weights 1.5 times too large the last layer's numbers had spread 1.63e+03 without BatchNorm and 0.618 with it, and with weights 0.5 times too small it was 4.68e-07 without and 0.618 with.

## 🔴 4. Deep / formal
**The training-time transform.** For one feature, over a batch of $m$ examples $x_1,\dots,x_m$:

$$
\mu_B=\frac1m\sum_{i=1}^{m}x_i,\quad \sigma_B^2=\frac1m\sum_{i=1}^{m}(x_i-\mu_B)^2,\quad \hat x_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\quad y_i=\gamma\,\hat x_i+\beta
$$

**Where:**
- $x_i$ is the feature's value for example $i$, and $m$ is the batch size. $y_i$ is the output.
- $\mu_B$ is the batch mean and $\sigma_B^2$ the batch variance (average squared distance from the mean, see [[Variance]]).
- $\epsilon$ is a tiny constant such as $10^{-5}$. **Matters:** without it, a batch where every example has the same value gives $\sigma_B=0$ and a division by zero.
- $\hat x_i$ is the normalised value, with mean 0 and variance 1 over the batch (ignoring $\epsilon$). **Matters:** this is what stops the numbers drifting from layer to layer.
- $\gamma$ and $\beta$ are the learned scale and shift. **Matters:** without them every layer would be forced to output mean 0 and spread 1. With them the network can pick any scale and centre, including undoing the normalisation.

**Example (tiny numbers):** one feature, a batch of four values $1,2,6,3$, with $\gamma=1$, $\beta=0$ first.
- Mean: $\mu_B=(1+2+6+3)/4=\mathbf{3}$. Variance: $\big((-2)^2+(-1)^2+3^2+0^2\big)/4=14/4=\mathbf{3.5}$, so the spread is $\sqrt{3.5}=1.871$.
- Normalised: $-2/1.871=\mathbf{-1.069}$, $-1/1.871=\mathbf{-0.535}$, $3/1.871=\mathbf{1.604}$, $0/1.871=\mathbf{0}$.
- With $\gamma=2$, $\beta=1$ the outputs become $2\hat x+1$: $\mathbf{-1.138}$, $\mathbf{-0.069}$, $\mathbf{4.207}$, $\mathbf{1}$.
- Meaning: whatever the original units, the batch now has a standard shape, and $\gamma,\beta$ then set the shape the network actually wants. The code printed the first three outputs and 0.0.

**The test-time version.** At test time there is no batch, so use running averages kept during training. With momentum $\alpha$ (0.1 in PyTorch's default), after each training batch:

$$
r_\mu\leftarrow(1-\alpha)\,r_\mu+\alpha\,\mu_B,\qquad r_{\sigma^2}\leftarrow(1-\alpha)\,r_{\sigma^2}+\alpha\,\sigma_B^2,\qquad y=\gamma\,\frac{x-r_\mu}{\sqrt{r_{\sigma^2}+\epsilon}}+\beta
$$

**Where:**
- $r_\mu$ and $r_{\sigma^2}$ are the stored running mean and variance. They start at 0 and 1 in PyTorch (stated from memory, but the code's numbers agree).
- $\alpha$ is the momentum: how much weight the newest batch gets. **Matters:** too small and the stored numbers lag behind the weights, too large and they are noisy.
- $\mu_B$ and $\sigma_B^2$ are the batch numbers from the equation above. In PyTorch the running variance uses the unbiased version (divide by $m-1$), which the code's numbers confirm. The Wikipedia page gives the same $m/(m-1)$ factor for the population variance.
- $y$ is the test-time output. **Matters:** it depends only on $x$ and fixed numbers, so the answer for one example no longer depends on its neighbours.

**Example (tiny numbers):** the batch $1,2,6,3$ from before, with $r_\mu=0$, $r_{\sigma^2}=1$ at the start and $\alpha=0.1$. The unbiased variance is $14/3=4.667$.
- Running mean: $0.9\times0+0.1\times3=\mathbf{0.3}$.
- Running variance: $0.9\times1+0.1\times4.667=\mathbf{1.367}$.
- Test-time output for $x=6$ ($\gamma=1,\beta=0$): $(6-0.3)/\sqrt{1.367}=5.7/1.169=\mathbf{4.876}$.
- Meaning: after one batch the stored numbers are still far from the true 3 and 3.5, so the output is off (the batch version gave 1.604). The code shows that after 200 batches they land near the truth (2.04 against 2, 9.11 against 9).

**Merging into the layer before it.** If the layer before is $z=Wx+b$, then at test time BatchNorm is a fixed straight-line map, and the two can be merged into one layer:

$$
s=\frac{\gamma}{\sqrt{r_{\sigma^2}+\epsilon}},\qquad W'=\operatorname{diag}(s)\,W,\qquad b'=s\,(b-r_\mu)+\beta
$$

**Where:**
- $W$ and $b$ are the weights and bias of the layer before. $W'$ and $b'$ are the merged ones.
- $s$ is a per-feature multiplier, built from the stored numbers and $\gamma$.
- $\operatorname{diag}(s)$ means "multiply row $k$ of $W$ by $s_k$".
- **Matters:** this only works in eval mode, where $r_\mu$ and $r_{\sigma^2}$ are fixed. In training mode the batch numbers change every step, so there is nothing fixed to merge.

**Example (tiny numbers):** one input. The layer is $z=2x+1$. BatchNorm has $r_\mu=5$, $r_{\sigma^2}=4$ (ignore $\epsilon$), $\gamma=3$, $\beta=1$.
- $s=3/\sqrt4=\mathbf{1.5}$. So $W'=1.5\times2=\mathbf{3}$ and $b'=1.5\times(1-5)+1=\mathbf{-5}$.
- Merged layer: $3x-5$. Check $x=4$: the long way is $z=9$, then $3\cdot(9-5)/2+1=7$. The short way is $3\cdot4-5=\mathbf{7}$.
- Meaning: two layers become one with the same answer. The code checked this on a real layer to a difference of 2.2e-16, so the cost of BatchNorm at test time can be zero extra layers.

**Further notes:**
- **Why it works is debated (Wikipedia page).** The paper's explanation was *internal covariate shift*, the inputs of each layer changing as earlier layers learn. The page says newer work challenges this: an experiment that added noise on purpose to create shift still did better than no BatchNorm, and other work argues the real effect is a smoother loss surface. The page also says other explanations exist, such as separating a weight's length from its direction. Which one is right is not settled. I did not read these papers.
- **Deep stacks can blow up at the start (page statement).** The page says a deep network with BatchNorm can suffer exploding gradients at initialisation, and that this is relieved by skip connections, as in [[ResNet]] and [[Residual Connection]]. I did not run this. My 20-layer code is a forward pass only, with no gradients.
- **Batch size matters.** The code shows the output for a fixed input has a spread of 0.739 at batch size 2 but 0.054 at 512. This follows the [[Law of Large Numbers]] idea: more examples, a steadier average. That noise is also one possible reason BatchNorm acts like a regulariser, which is from memory, not tested here.
- **Not covered:** the formulas for the backward pass (the Wikipedia page lists them), GroupNorm and other batch-free variants, and what happens with very different test data.

**What came before, and how this improved on it:**
- **Before:** careful choice of starting numbers, small learning rates, and scaling the inputs once at the start (see [[Normalization]]).
- **Its limit:** the paper's abstract says changing inputs to each layer "slows down the training by requiring lower learning rates and careful parameter initialization", and makes it hard to train with saturating activations.
- **What changed:** Sergey Ioffe and Christian Szegedy introduced Batch Normalization in the paper "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" (arXiv 1502.03167, submitted 11 February 2015; the abstract was read and the Wikipedia page was read to the end of its learning-halfspace section, not the paper body). They normalise each layer's inputs for every mini-batch, and the abstract reports the same accuracy with 14 times fewer training steps on a state-of-the-art image classifier, and 4.9% top-5 validation error on ImageNet with an ensemble.
- **How much better:** in the code, a 20-layer network whose weights start 1.5 times too large ends with numbers of spread 1,630 without BatchNorm and 0.618 with it. At 0.5 times too small it is 4.68e-07 against 0.618. The [[Normalization]] note has a separate training test, where BatchNorm cut the Adam steps to reach a loss of 0.5 by about 7 times in one small case. This note's own test is a forward pass only, so it shows steady numbers, not faster training.

## Related
- [[Normalization]] · [[Layer Normalization]] · [[Neural Network]] · [[Convolutional Neural Network]] · [[ResNet]] · [[Residual Connection]] · [[Backpropagation]] · [[Gradient Descent]] · [[Learning Rate]] · [[Weight Initialization]] · [[Variance]] · [[Expected Value]] · [[Law of Large Numbers]] · [[Dropout]] · [[Overfitting]] · [[Activation Function]] · [[ReLU]] · [[Adam Optimizer]]

## Sources
- Ioffe and Szegedy, *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*, arXiv 1502.03167 (abstract read; the body was not read)
- Wikipedia, *Batch normalization* (read through its learning-halfspace section, which is most of the page; the transform, inference, theory and gradient-explosion statements come from it)
- The code and its numbers are my own, run in this session.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/training BatchNorm:::A layer that rescales each feature using the average and spread of the whole batch, so deep networks train faster and more steadily.
<!--SR:!2000-01-01,1,250!2026-11-16,0,210-->

#flashcards/ml/concept #flashcards/topic/training Why does BatchNorm use stored running averages at test time?::A single new example has no batch. During training the layer keeps a running average of the batch means and variances, and uses those fixed numbers later.

#flashcards/ml/concept #flashcards/topic/training Does BatchNorm's output for one example depend on the other examples in its batch?::In training mode, yes. The mean and variance come from the whole batch. In eval mode, no, because fixed running numbers are used.
<!--SR:!2026-11-16,0,170-->

#flashcards/ml/example #flashcards/topic/training What happens with a batch of 1 in training mode?::There is no spread. The note's own formula gives a constant output (just the shift), and PyTorch raises an error.

#flashcards/ml/cloze #flashcards/topic/training BatchNorm looks at one feature across the ==whole batch==, while Layer Normalization looks at ==one example== only.

#flashcards/ml/cloze #flashcards/topic/training BatchNorm first computes $\hat x_i=(x_i-\mu_B)/\sqrt{\sigma_B^2+\epsilon}$, then $y_i=\gamma\hat x_i+\beta$. The learned ==scale== is $\gamma$ and the learned ==shift== is $\beta$.
<!--SR:!2026-11-16,0,190!2000-01-01,1,250-->

#flashcards/ml/formula #flashcards/topic/training Write the full BatchNorm formula chain (batch mean, variance, normalised value, output).
?
$$
\mu_B=\frac1m\sum_{i=1}^{m}x_i,\quad \sigma_B^2=\frac1m\sum_{i=1}^{m}(x_i-\mu_B)^2,\quad \hat x_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\quad y_i=\gamma\,\hat x_i+\beta
$$
- $x_i$: the value of one feature for example $i$. $m$: the batch size.
- $\mu_B$: the batch average. $\sigma_B^2$: the batch variance (spread squared).
- $\hat x_i$: the value after subtracting the average and dividing by the spread. It has mean 0 and variance 1 over the batch.
- $\epsilon$: a tiny number that stops division by zero when the spread is 0.
- $\gamma$ (scale) and $\beta$ (shift): learned numbers, one pair per feature, so the network can undo the normalising if that helps.
- $y_i$: the final output of the layer.
<!--SR:!2026-11-16,0,230-->

#flashcards/ml/example #flashcards/topic/training A batch has the values 1, 2, 6, 3. What are its mean and variance?::Mean 3 and variance 3.5 (deviations −2, −1, 3, 0, squared 4, 1, 9, 0, sum 14, divided by 4).

#flashcards/ml/concept #flashcards/topic/training After training, BatchNorm can be merged into which layer?::The Linear (or convolution) layer before it, because at test time it is just a fixed multiply-and-add. The note's code folded 2 layers into 1.

#flashcards/ml/formula #flashcards/topic/training Write the BatchNorm running statistics and the test-time output, and explain each symbol.
?
$$r_\mu\leftarrow(1-\alpha)\,r_\mu+\alpha\,\mu_B,\qquad r_{\sigma^2}\leftarrow(1-\alpha)\,r_{\sigma^2}+\alpha\,\sigma_B^2,\qquad y=\gamma\,\frac{x-r_\mu}{\sqrt{r_{\sigma^2}+\epsilon}}+\beta$$
- $r_\mu$, $r_{\sigma^2}$: the stored running mean and variance. They start at 0 and 1 in PyTorch (from memory in the note, but the code's numbers agree).
- $\alpha$: the momentum, how much weight the newest batch gets. Too small and the stored numbers lag behind the weights, too large and they are noisy.
- $\mu_B$, $\sigma_B^2$: the batch mean and variance. In PyTorch the running variance uses the unbiased version (divide by $m-1$).
- $y$: the test-time output. It depends only on $x$ and fixed numbers, so one example gives the same answer whatever else is in the batch.

#flashcards/ml/example #flashcards/topic/training Running statistics start at $r_\mu=0$, $r_{\sigma^2}=1$ with $\alpha=0.1$. A batch 1, 2, 6, 3 arrives. What are the new running values and the test output for $x=6$ ($\gamma=1$, $\beta=0$)?::The unbiased variance is $14/3=4.667$. Running mean $0.9\times0+0.1\times3=0.3$. Running variance $0.9\times1+0.1\times4.667=1.367$. Output $(6-0.3)/\sqrt{1.367}=5.7/1.169=4.876$. After one batch the stored numbers are still far from the true 3 and 3.5, so the output is off (the batch version gave 1.604).

#flashcards/ml/formula #flashcards/topic/training Write how a BatchNorm layer is merged into the layer before it, and explain each symbol.
?
$$s=\frac{\gamma}{\sqrt{r_{\sigma^2}+\epsilon}},\qquad W'=\operatorname{diag}(s)\,W,\qquad b'=s\,(b-r_\mu)+\beta$$
- $W$, $b$: the weights and bias of the layer before. $W'$, $b'$: the merged ones.
- $s$: a per-feature multiplier built from the stored numbers and $\gamma$.
- $\operatorname{diag}(s)$ means multiply row $k$ of $W$ by $s_k$.
- This only works in eval mode, where $r_\mu$ and $r_{\sigma^2}$ are fixed. In training mode the batch numbers change every step, so there is nothing fixed to merge.

#flashcards/ml/example #flashcards/topic/training The layer is $z=2x+1$ and BatchNorm has $r_\mu=5$, $r_{\sigma^2}=4$ (ignore $\epsilon$), $\gamma=3$, $\beta=1$. What is the merged layer, and does it agree at $x=4$?::$s=3/\sqrt4=1.5$, $W'=1.5\times2=3$, $b'=1.5\times(1-5)+1=-5$. Merged layer $3x-5$. At $x=4$ the long way is $z=9$, then $3\cdot(9-5)/2+1=7$, and the short way is $3\cdot4-5=7$. Two layers become one with the same answer. The note's code checked this on a real layer to a difference of 2.2e-16.
