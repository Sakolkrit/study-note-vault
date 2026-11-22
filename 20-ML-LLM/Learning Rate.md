---
tags: [machine-learning, optimization, learning-rate, training, hyperparameter]
status: growing
level: 4
tier: 1
---

# Learning Rate

## 🟢 1. Simple (age 12)
The learning rate is **how big a step the computer takes each time it tries to improve**. Imagine walking down a hill in the fog. If your steps are **tiny**, you will get down eventually, but it takes forever. If your steps are **giant leaps**, you may jump right over the bottom and land on the other side of the valley, then leap back, and never settle. A **good size** gets you down quickly and lets you stop at the bottom.

**Example:** Tuning a radio. If you turn the dial a huge amount each time, you keep flying past the station. If you turn it a tiny bit each time, it takes minutes. The best way is big turns at first, then smaller and smaller turns as you get close to the clear sound. Training a model often does exactly that.

## 🔵 2. How it works
1. Training changes the model's weights in small steps. Each step moves the weights **against the gradient** (see [[Gradient Descent]] and [[Stochastic Gradient Descent]]).
2. The gradient says *which direction* to move, but not *how far*. The **learning rate** is a number, usually written $\eta$ (the Greek letter "eta"), that sets how far.
3. **Too high:** the weights overshoot the bottom, bounce from side to side, and the loss zigzags or even grows.
4. **Too low:** each step is tiny, the loss falls slowly, and training may stall in a flat area long before it is done.
5. **Well tuned:** the loss drops quickly at first and then flattens as the model settles.
6. Because the best step is large at the start and small near the end, people use a **schedule** that lowers the learning rate over time. Some also start very small and ramp up for the first few steps (a **warm-up**).
7. The learning rate is not learned by the model. You choose it before training, so it is called a **hyperparameter**. You pick it by trying values and watching the loss on [[Supervised Learning]] validation data.

![[learning-rate-too-large.jpg]]
*Two walks down the same bumpy loss landscape (the coloured rings are contour lines of the loss). Red: a fixed step size of 0.15. It travels well at first but near the bottom it zigzags back and forth in a dense band instead of settling. Blue: steps of a carefully chosen size at each stage (a "line search"), which settle smoothly. Image: Tomasz59, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Gradient_descent_method_%E2%80%93_too_large_step_causes_oscillations_near_the_maximum.jpg), CC BY-SA 4.0. The file name says "maximum", but the plot and its description show a minimum.*

## 🟠 3. Technical detail
- **What it multiplies:** in the update $\theta\leftarrow\theta-\eta g$, the learning rate scales the gradient $g$. Doubling it doubles every step.
- **A rough guide:** typical values are between about $10^{-5}$ and $10^{-1}$, depending on the model and the optimiser. A common first try for [[Adam Optimizer]] is around $10^{-3}$, and smaller for fine-tuning a big model (see [[Fine-tuning]]). These are habits, not rules, so check on your own problem.
- **Finding a good value:**
  - Try values spaced by factors of 10 (for example $10^{-4},10^{-3},10^{-2}$) and compare the loss curves.
  - A **learning-rate range test**: start tiny, increase it a little every step, and watch where the loss starts to rise. Pick a value somewhat below that point.
- **Common schedules:**
  - **Step decay:** cut the rate by a fixed factor every so many epochs.
  - **Exponential decay:** multiply by a number slightly below 1 each step.
  - **Cosine decay:** follow the shape of a cosine curve from the starting rate down towards 0.
  - **Warm-up:** raise the rate gradually over the first steps, then decay. Very common when training a [[Transformer]]. (Common practice; the reasons are partly empirical.)
- **Signs to look for:**
  - Loss goes to `nan` or shoots up: rate too high (or the gradients are exploding).
  - Loss barely moves: rate too low, or something else is wrong.
  - Loss falls then bounces near the bottom: lower the rate (decay it).
- **It interacts with other settings:** a bigger batch usually allows a bigger rate ([[Stochastic Gradient Descent]]); adaptive optimisers like [[Adam Optimizer]] scale the step per weight, but still have a global learning rate; and [[Normalization]] makes training tolerate larger rates.
- **Per-weight rates:** different layers sometimes use different rates, for example a small one for old pretrained layers and a larger one for a new layer in [[Fine-tuning]].

```python
for eta in (0.1, 0.5, 1.0, 1.1):                 # minimise f(w) = w^2, whose gradient is 2w
    w = 1.0; path = []
    for _ in range(3):
        w = w - eta * 2 * w                      # one gradient descent step
        path.append(round(w, 3))
    print(eta, path)
# 0.1 [0.8, 0.64, 0.512]        slow, steady
# 0.5 [0.0, 0.0, 0.0]           perfect in one step (only on this exact bowl)
# 1.0 [-1.0, 1.0, -1.0]         bounces forever
# 1.1 [-1.2, 1.44, -1.728]      blows up

print([0.1 * 0.5 ** (t // 10) for t in (0, 9, 10, 20)])    # step decay: [0.1, 0.1, 0.05, 0.025]
```

## 🔴 4. Deep / formal
**Why there is a speed limit: a bowl-shaped loss.** The simplest loss is a one-dimensional bowl, $f(w)=\tfrac{h}{2}w^{2}$, whose lowest point is at $w=0$. Here $h>0$ is how sharply the bowl curves. The gradient is $hw$, so one step of gradient descent is:

$$
w_{t+1}=w_t-\eta\,h\,w_t=(1-\eta h)\,w_t
$$

**Where:**
- $w_t$ is the weight at step $t$ (we want it to reach 0). $\eta$ is the learning rate.
- $h$ is the **curvature** of the loss: how quickly the slope changes. A narrow, steep bowl has a large $h$, a wide, gentle one has a small $h$. In many dimensions it is the second derivative in each direction (see [[Calculus]]).
- $1-\eta h$ is the factor the weight is multiplied by at **every** step. **Matters:** after $t$ steps $w_t=(1-\eta h)^t w_0$, so what happens is decided entirely by this one factor.
- **Matters:** if $|1-\eta h|<1$ the weight shrinks towards 0 (it converges). That holds exactly when $0<\eta<2/h$. If $\eta h=1$, it lands on 0 in a single step. If $1<\eta h<2$ it still converges but flips sign each step (zigzag). If $\eta h=2$ it bounces forever. If $\eta h>2$ it grows (diverges).
- **Matters:** in a real model the loss has many directions with different curvatures. The **steepest** direction (largest $h$) sets the largest safe learning rate, while the **flattest** direction then moves slowly. This is why long narrow valleys are hard, and why [[Adam Optimizer]] tries to rescale each direction.

**Example (tiny numbers):** the loss $f(w)=w^2$, so $h=2$ and the safe range is $0<\eta<2/2=1$. Start at $w_0=1$.
- $\eta=0.1$: factor $1-0.2=0.8$. $w=0.8,\ 0.64,\ 0.512$. Converging, but slowly.
- $\eta=0.5$: factor $1-1=0$. $w=\mathbf{0}$ after one step. (Exactly right here only because the bowl is perfect.)
- $\eta=1.0$: factor $-1$. $w=-1,\ 1,\ -1$: it bounces between two points forever.
- $\eta=1.1$: factor $-1.2$. $w=-1.2,\ 1.44,\ -1.728$: each jump is bigger. Diverging.
- Meaning: right at $\eta=1$ (that is $2/h$) there is a cliff edge. The same code gives smooth convergence just below it and disaster just above it.

**Decaying the rate: a schedule.** To take big steps early and fine steps late, the rate is made to depend on the step number. A common simple choice is **step decay**:

$$
\eta_t=\eta_0\,\gamma^{\lfloor t/s\rfloor}
$$

**Where:**
- $\eta_t$ is the learning rate to use at step $t$. $\eta_0$ is the starting rate.
- $\gamma$ ("gamma") is the **decay factor**, a number between 0 and 1 (for example 0.5 halves the rate each time).
- $s$ is the **step length**: how many steps pass before the next cut. $\lfloor t/s\rfloor$ ("floor") means "$t/s$ rounded down", which counts how many cuts have happened so far.
- **Matters:** the rate stays flat for $s$ steps, then drops by the factor $\gamma$, and so on. Without the decay, a noisy method like [[Stochastic Gradient Descent]] keeps bouncing around the minimum at a size set by $\eta$. With it, the bounce gets smaller and the weights can settle.
- Other schedules (exponential, cosine, warm-up) are just other shapes for $\eta_t$.

**Example (tiny numbers):** $\eta_0=0.1$, $\gamma=0.5$, $s=10$.
- Steps 0 to 9: $\lfloor t/10\rfloor=0$, so $\eta=0.1\cdot0.5^0=\mathbf{0.1}$.
- Step 10: $\lfloor10/10\rfloor=1$, so $\eta=0.1\cdot0.5=\mathbf{0.05}$.
- Step 20: $\eta=0.1\cdot0.5^2=\mathbf{0.025}$.
- Meaning: the step size halves every 10 steps, so late in training the model only makes small corrections.

**Further notes:**
- **For non-bowl losses** the safe rate depends on the local curvature, which changes as training moves. This is one reason a rate that works early can become too big later, and why decay and warm-up help. (Stated informally; there is no single clean theorem for deep networks.)
- **Large rates can help generalisation** in some settings, by making training noisy enough to avoid sharp dips. This is reported in the research but not settled (from memory; check).
- **Learning rate vs weight decay:** with some optimisers the two interact, which is why variants such as decoupled weight decay exist (see [[Regularization]] and [[Adam Optimizer]]; from memory).
- **Gradient clipping** (cap the size of the gradient) is a separate safety net for rare huge gradients. It is not covered in a note of its own yet.

**What came before, and how this improved on it:**
- **Before:** one fixed learning rate for the whole run.
- **Its limit:** a large rate is fast at first but bounces around the minimum because the gradients are noisy; a small rate is calm but slow.
- **What changed:** change the rate during training, such as a cosine decay or warm restarts. The SGDR abstract reports results of 3.14% and 16.21% error on CIFAR-10 and CIFAR-100 with warm restarts (Loshchilov and Hutter, arXiv 1608.03983, 13 August 2016, abstract read).
- **How much better:** in the code below (noisy gradients of $w^2$, 2000 steps, 200 runs), a constant rate of 0.1 ends at an average loss of 0.10049, a constant 0.001 at 0.00893, and a cosine decay from 0.1 at 0.00304, which is 33 times lower than the first and 2.9 times lower than the second.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
def run(sched, steps=2000):
    w = 5.0
    for t in range(steps):
        g = 2 * w + 2 * rng.standard_normal()         # noisy gradient of w^2
        w -= sched(t, steps) * g
    return w ** 2
const = lambda t, T: 0.1
decay = lambda t, T: 0.1 * 0.5 * (1 + np.cos(np.pi * t / T))      # cosine decay to 0
small = lambda t, T: 0.001
res = {n: np.mean([run(s) for _ in range(200)]) for n, s in (("constant 0.1", const), ("cosine decay from 0.1", decay), ("constant 0.001", small))}
for n, v in res.items(): print(f"{n:22s} average final loss {v:.5f}")
```

## Related
- [[Gradient Descent]] · [[Stochastic Gradient Descent]] · [[Adam Optimizer]] · [[Regularization]] · [[Normalization]] · [[Loss Function]] · [[Backpropagation]] · [[Fine-tuning]] · [[Transformer]] · [[Calculus]] · [[Neural Network]] · [[Supervised Learning]] · [[Convex Optimization]] · [[Eigenvalue]] · [[Q-Learning]] · [[Binary Search]] · [[Feature Scaling]] · [[Hessian]]

## Sources
- Goodfellow, Bengio & Courville, *Deep Learning*, 2016, chapter 8
- Smith, *Cyclical Learning Rates for Training Neural Networks*, 2017 (learning-rate range test; from memory)
- Goyal et al., *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*, 2017 (warm-up; from memory)
- Loshchilov & Hutter, *SGDR: Stochastic Gradient Descent with Warm Restarts*, 2016 (cosine schedule; from memory)
- Loshchilov and Hutter, SGDR, arXiv 1608.03983 (abstract read for the history block)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/training Learning rate:::How big a step the computer takes each time it tries to improve. Too small is slow, too big overshoots the bottom and never settles.

#flashcards/ml/formula #flashcards/topic/training Write gradient descent on a bowl and explain each symbol.
?
$$w_{t+1}=w_t-\eta\,h\,w_t=(1-\eta h)\,w_t$$
- $w_t$: the weight at step $t$ (we want it to reach 0). $\eta$: the learning rate.
- $h$: the curvature of the loss, how quickly the slope changes. A narrow, steep bowl has large $h$.
- $1-\eta h$: the factor the weight is multiplied by each step. It must be between −1 and 1 to converge, which means $0<\eta<2/h$.
- Beyond $2/h$ the weight overshoots and grows each step.

#flashcards/ml/formula #flashcards/topic/training Write a step-decay learning rate schedule and explain each symbol.
?
$$\eta_t=\eta_0\,\gamma^{\lfloor t/s\rfloor}$$
- $\eta_t$: the learning rate at step $t$. $\eta_0$: the starting rate.
- $\gamma$: the decay factor between 0 and 1 (0.5 halves the rate each time).
- $s$: the step length, how many steps pass before the next cut. $\lfloor t/s\rfloor$ counts how many cuts have happened.
- Big steps early make fast progress, small steps late let the model settle.

#flashcards/ml/cloze #flashcards/topic/training On a bowl with curvature $h$, gradient descent is stable only when $\eta<$ ==2/h==.

#flashcards/ml/example #flashcards/topic/training For $f(w)=w^2$ ($h=2$) starting at $w_0=1$, what happens with $\eta=0.1$, $0.5$ and $1.0$?::η = 0.1: factor 0.8, w = 0.8, 0.64, 0.512 (converges slowly). η = 0.5: factor 0, w = 0 after one step. η = 1.0: factor −1, w = −1, 1, −1 (bounces forever). The safe range is 0 < η < 1.

#flashcards/ml/example #flashcards/topic/training With $\eta_0=0.1$, $\gamma=0.5$ and $s=10$, what is the rate at steps 0–9, 10 and 20?::0.1 for steps 0 to 9. Step 10: 0.05. Step 20: 0.025. It halves every 10 steps.

#flashcards/ml/image #flashcards/topic/training ![[learning-rate-too-large.jpg]]
What does this picture show?
?
Two walks down the same bumpy loss landscape. Red: a fixed step size of 0.15. It travels well at first but near the bottom it zigzags back and forth instead of settling.
