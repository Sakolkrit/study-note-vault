---
tags: [machine-learning, optimization, SGD, training, neural-network]
status: growing
level: 4
tier: 2
---

# Stochastic Gradient Descent

Often shortened to **SGD**. It is the workhorse version of [[Gradient Descent]] that almost all deep learning uses.

## 🟢 1. Simple (age 12)
Imagine you are on a foggy hill and want to walk down to the valley. Plain gradient descent says: "look at the **whole** hill around you, then take one careful step downhill." That is slow if the hill is huge. Stochastic gradient descent says: "glance at **a small handful** of the ground around you, take a quick step, and repeat." Each glance is a bit wrong, so your path wobbles, but you take **many more steps**, and you still end up near the bottom.

**Example:** A teacher wants to know how well the class understands a topic. She could mark all 1000 tests before changing her lesson. Or she could mark just 10 random tests, tweak the lesson, then mark 10 more, and tweak again. The small samples are noisy, but she improves the lesson hundreds of times faster.

## 🔵 2. How it works
1. You have a big pile of training examples and a model with adjustable numbers (its weights). A [[Loss Function]] says how wrong the model is.
2. **Shuffle** the examples and cut them into small groups called **mini-batches** (for example 32 or 256 examples).
3. Take one mini-batch. Run the model on it (the forward pass) and measure the loss.
4. Use [[Backpropagation]] to find the **gradient**: which way to nudge each weight to lower the loss on this mini-batch.
5. Move every weight a small step in that direction. The step size is the [[Learning Rate]].
6. Take the next mini-batch and repeat. Going once through every mini-batch is one **epoch**. Training usually runs for many epochs.

There are three ways to choose how many examples to use for each step:

| Name | Examples per step | Behaviour |
|---|---|---|
| Batch gradient descent | All of them | Smooth and exact, but each step is slow and costly |
| Stochastic (pure SGD) | One | Very noisy, but each step is cheap |
| Mini-batch | A small group | The usual choice: reasonably smooth, fast, and friendly to a [[GPU]] |

In practice people say "SGD" even when they mean mini-batch SGD.

No image: the only Commons hit for this topic was unsuitable (a photo with no teaching value), so I left it out. For a picture of the descent idea itself, see [[Gradient Descent]].

## 🟠 3. Technical detail
- **Why it works:** the gradient on a random mini-batch is, on average, equal to the gradient on the full dataset. It is a noisy but **unbiased** estimate, so on average the steps point the right way.
- **Why it is used:** with a million examples, computing one exact gradient costs a million forward and backward passes. With mini-batches, you can make a million small updates for the same cost, and the model improves long before one exact step would finish.
- **Noise is a feature and a problem:**
  - It lets training jiggle out of poor spots such as saddle points and shallow dips.
  - It stops the weights from settling exactly at the bottom, so people **shrink the learning rate** over time (see [[Learning Rate]]).
  - Some evidence suggests the noise biases training towards solutions that generalise better. This is an active research topic (from memory; check).
- **Choosing the batch size:**
  - Small batches: noisier, more updates per epoch, less memory.
  - Large batches: smoother, use a [[GPU]] more efficiently, but often need a larger learning rate and can generalise a little worse unless tuned. There is no single best number.
- **SGD with momentum:** keep a running "velocity" of recent gradients and move along it. This smooths the noise and speeds travel along long valleys. The idea is described in [[Adam Optimizer]], which adds a per-weight adaptive step on top.
- **Shuffling matters:** if the data is in a fixed order (all cats, then all dogs), the steps pull one way and then the other. Shuffle every epoch.
- **Practical defaults:** mini-batch SGD with momentum for many vision models, and [[Adam Optimizer]]-style methods for many language models. (A common practice, not a rule.)
- **Related fixes and checks:** [[Normalization]] helps training stay stable, and held-out data tells you when to stop (see [[Overfitting]] and [[Regularization]]).

```python
import numpy as np
x = np.array([0., 1., 2.]); y = np.array([1., 3., 2.]); w = 1.0      # model: y_hat = w * x, loss (w*x - y)^2
g = 2 * (w * x - y) * x                                              # gradient of the loss for each example
print(g, g.mean())                                                   # [-0. -4.  0.] -1.333...  the full-batch gradient (-0 is just 0)
print(w - 0.1 * g.mean())                                            # 1.1333...    one full-batch step, learning rate 0.1
print(g[1:].mean(), w - 0.1 * g[1:].mean())                          # -2.0 1.2     a mini-batch of examples 2 and 3: noisier
print(g[:1].mean(), w - 0.1 * g[:1].mean())                          # 0.0 1.0      a batch of one example: it says "do not move"

# training on 200 noisy points from y = 3x: full batch vs mini-batches of 10
rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, 200); Y = 3 * X + rng.normal(0, 0.3, 200)
for B in (200, 10):
    w = 0.0
    for epoch in range(20):
        idx = rng.permutation(200)                                   # shuffle each epoch
        for s in range(0, 200, B):
            b = idx[s:s + B]; w -= 0.1 * (2 * (w * X[b] - Y[b]) * X[b]).mean()
    print(B, round(w, 3))
# 200 2.32     full batch: only 20 updates in 20 epochs, still far from the true slope 3
# 10 2.968     mini-batches: 400 updates in the same 20 epochs, very close to 3
```

## 🔴 4. Deep / formal
**The mini-batch gradient and the update.** Let the model have weights $\theta$ and let $\ell_i(\theta)$ be the loss on example $i$. Pick a random mini-batch $\mathcal B$ of $B$ examples. SGD estimates the gradient from just those, then steps:

$$
g_t=\frac{1}{B}\sum_{i\in\mathcal B}\nabla_\theta\,\ell_i(\theta_t),\qquad \theta_{t+1}=\theta_t-\eta\,g_t
$$

**Where:**
- $\theta_t$ is the vector of all weights at step $t$. $\ell_i$ is the loss on example $i$ alone. $\nabla_\theta\ell_i$ (nabla, "gradient") is the list of slopes saying how that loss changes as each weight changes (see [[Calculus]]).
- $\mathcal B$ is the random mini-batch and $B$ is how many examples it holds. $\sum_{i\in\mathcal B}$ adds over just those examples, and $\tfrac1B$ turns the sum into an average.
- $g_t$ is the **stochastic gradient**: the mini-batch estimate of the true gradient. **Matters:** it needs only $B$ examples, not all of them, so one step is cheap.
- $\eta$ ("eta") is the [[Learning Rate]]. **Matters:** it scales the step. Too big and the weights overshoot, too small and training crawls.
- The minus sign moves **against** the gradient, downhill. **Matters:** with a plus sign the model would climb to a higher loss.
- **Matters:** because $\mathcal B$ is random, $g_t$ equals the full gradient only *on average*. Any single step is noisy.

**Example (tiny numbers):** model $\hat y=wx$ with loss $(wx-y)^2$ per example, so $\nabla_w\ell_i=2(wx_i-y_i)x_i$. Data $(0,1),(1,3),(2,2)$, current weight $w=1$, $\eta=0.1$.
- Per-example gradients: $x=0$: $2(0-1)\cdot0=0$. $x=1$: $2(1-3)\cdot1=-4$. $x=2$: $2(2-2)\cdot2=0$.
- Full batch ($B=3$): $g=(0-4+0)/3=\mathbf{-1.333}$, so $w\leftarrow1-0.1\cdot(-1.333)=\mathbf{1.133}$.
- Mini-batch of examples $\{x=1,\,x=2\}$ ($B=2$): $g=(-4+0)/2=\mathbf{-2}$, so $w\leftarrow\mathbf{1.2}$.
- Mini-batch of only $\{x=0\}$ ($B=1$): $g=\mathbf{0}$, so $w$ stays at $\mathbf{1}$.
- Meaning: all three steps are honest guesses at the same direction, but they differ a lot. The more examples per batch, the closer the guess is to the full-batch one.

**How noisy is it? The size of the noise.** If each example's gradient has variance $\sigma^2$ around the true gradient, then for a batch of $B$ examples drawn independently:

$$
\operatorname{Var}(g)=\frac{\sigma^{2}}{B}
$$

**Where:**
- $g$ is the mini-batch gradient from above, and $B$ is the batch size.
- $\sigma^2$ is the **variance of a single example's gradient**: how much one example's gradient typically differs from the average gradient. It depends on the data and the model.
- $\operatorname{Var}(g)$ is the variance of the batch estimate (see [[Probability]]). **Matters:** it falls as $1/B$, so doubling the batch halves the variance. The typical size of the noise (the standard deviation) falls as $1/\sqrt B$, so to **halve the noise** you must use **four times** as many examples.
- **Matters:** this is why very large batches give diminishing returns: you pay in compute for a smaller and smaller drop in noise.
- The formula assumes examples are drawn independently (with replacement). Drawing without replacement gives a slightly smaller variance.

**Example (tiny numbers):** the three per-example gradients above are $0,\,-4,\,0$, with average $-1.333$.
- Squared distances from the average: $1.778,\ 7.111,\ 1.778$. Their average is $\sigma^2=\mathbf{3.556}$.
- Batch of one ($B=1$): variance $3.556$ (standard deviation $\approx1.89$).
- Batch of two ($B=2$): variance $3.556/2=\mathbf{1.778}$. Checking by listing all 9 equally likely ordered pairs of examples and averaging each pair's gradient gives the same variance, $1.778$, and the same average, $-1.333$ (the average is the true gradient, so the estimate is unbiased).
- Meaning: using two examples instead of one halves the variance, but the estimate is still far from exact, which is why the full-batch, one-example and two-example steps above all differ.

**Further notes:**
- **Convergence:** for a smooth bowl-shaped loss, SGD with a shrinking learning rate converges to the minimum. With a fixed learning rate it hovers in a small region around it. For deep networks the loss is not a bowl, so guarantees are much weaker. (Standard results, stated from memory.)
- **Learning-rate schedules and warm-up** are usually part of how SGD is run in practice, see [[Learning Rate]].
- **The "S" has a history:** the idea goes back to stochastic approximation work in the 1950s (Robbins and Monro; from memory).
- **Link to bias-variance:** the noise here is a separate idea from the model's bias and variance in [[Bias-Variance]]. One is noise in the optimisation steps, the other is how the learned model changes with the training data.

**What came before, and how this improved on it:**
- **Before:** full-batch gradient descent, which reads every training example for each step.
- **Its limit:** with $N$ examples, each step costs $N$ example-gradients, so progress per unit of work is slow when $N$ is large.
- **What changed:** estimate the gradient from a small random batch and step more often. The 2016 review by Bottou, Curtis and Nocedal says the stochastic gradient method has traditionally played a central role in large-scale machine learning (arXiv 1606.04838, abstract read). The original Robbins and Monro paper (1951) is from memory.
- **How much better:** in the code below, both methods get the same work (500,000 example-gradients on 100,000 noisy points). Full batch makes 5 steps and ends at loss 0.9622. Mini-batches of 32 make 15,625 steps and end at 0.2505, where about 0.25 is the best possible (the noise level). The two step sizes were picked by hand, so the gap is for these settings.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
N, P = 100_000, 10
X = rng.standard_normal((N, P)); w_true = rng.standard_normal(P); y = X @ w_true + 0.5 * rng.standard_normal(N)
loss = lambda w: ((X @ w - y) ** 2).mean()
budget = 500_000                                      # same number of example-gradients for both
w = np.zeros(P)
for _ in range(budget // N):                          # full batch: 5 steps
    w -= 0.1 * 2 * X.T @ (X @ w - y) / N
full = loss(w)
w = np.zeros(P)
for _ in range(budget // 32):                         # mini-batch of 32: 15,625 steps
    i = rng.integers(0, N, 32); w -= 0.01 * 2 * X[i].T @ (X[i] @ w - y[i]) / 32
print("same work (500,000 example-gradients): full batch loss", round(full, 4), " mini-batch loss", round(loss(w), 4), " best possible about 0.25")
```

## Related
- [[Gradient Descent]] · [[Learning Rate]] · [[Adam Optimizer]] · [[Backpropagation]] · [[Loss Function]] · [[Regularization]] · [[Neural Network]] · [[Normalization]] · [[GPU]] · [[Overfitting]] · [[Calculus]] · [[Probability]] · [[Bias-Variance]] · [[Momentum]] · [[RMSProp]] · [[PyTorch]] · [[Gradient Accumulation]]
- [[Matrix Factorization]]
## Sources
- Goodfellow, Bengio & Courville, *Deep Learning*, 2016, chapter 8
- Bottou, Curtis & Nocedal, *Optimization Methods for Large-Scale Machine Learning*, 2018 (from memory)
- Robbins & Monro, *A Stochastic Approximation Method*, 1951 (from memory)
- Ruder, *An overview of gradient descent optimization algorithms*, 2016 (free online; from memory)
- Bottou, Curtis and Nocedal, arXiv 1606.04838 (abstract read for the history block)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/training Stochastic gradient descent (SGD):::Instead of looking at the whole hill before each step, glance at a small random handful of examples (a mini-batch), take a quick step, and repeat. Each step is noisier but much cheaper.

#flashcards/ml/formula #flashcards/topic/training Write the SGD step and explain each symbol.
?
$$g_t=\frac{1}{B}\sum_{i\in\mathcal B}\nabla_\theta\,\ell_i(\theta_t),\qquad \theta_{t+1}=\theta_t-\eta\,g_t$$
- $\theta_t$: the vector of all weights at step $t$. $\ell_i$: the loss on example $i$ alone. $\nabla_\theta\ell_i$: the slopes of that loss for each weight.
- $\mathcal B$: the random mini-batch. $B$: how many examples it holds. Averaging over it gives the estimate $g_t$ of the full gradient.
- $\eta$: the learning rate, the step size.
- The step goes against the gradient, so the loss goes down on average.

#flashcards/ml/formula #flashcards/topic/training Write the variance of a mini-batch gradient and explain each symbol.
?
$$\operatorname{Var}(g)=\frac{\sigma^{2}}{B}$$
- $g$: the mini-batch gradient from above. $B$: the batch size.
- $\sigma^2$: the variance of a single example's gradient, how much one example's gradient typically differs from the average. It depends on the data and the model.
- $\operatorname{Var}(g)$: how noisy the batch estimate is. Bigger batches are less noisy, but the noise only falls as $1/B$, so returns diminish.

#flashcards/ml/cloze #flashcards/topic/training SGD's noise falls as ==1/B==, so a batch ==4 times== bigger halves the standard deviation.

#flashcards/ml/example #flashcards/topic/training Model $\hat y=wx$, loss $(wx-y)^2$, data $(0,1),(1,3),(2,2)$, $w=1$, $\eta=0.1$. What are the per-example gradients and the full-batch step?::2(wx − y)x gives 0, −4, 0. Full batch g = (0 − 4 + 0)/3 = −1.333, so w ← 1 − 0.1 × (−1.333) = 1.133.

#flashcards/ml/example #flashcards/topic/training The three per-example gradients are 0, −4, 0 (average −1.333). What is $\sigma^2$, and the batch variance for $B=2$?::Squared distances 1.778, 7.111, 1.778, average σ² = 3.556 (standard deviation about 1.89 for B = 1). For B = 2: 3.556/2 = 1.778.
