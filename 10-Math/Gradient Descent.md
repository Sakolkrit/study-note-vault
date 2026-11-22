---
tags: [gradient-descent, optimization, calculus, math, deep-learning, LLM]
status: growing
level: 4
tier: 2
---

# Gradient Descent

## 🟢 1. Simple (age 12)
Gradient descent is how a computer **finds the bottom of a valley while blindfolded**: feel which way the ground slopes down under your feet, take a small step that way, and repeat.

**Example:** You are on a hill in thick fog and want to reach the lowest spot. You cannot see it, but you can feel the slope. So you step downhill, feel again, step again. Eventually the ground is flat: you have reached a low spot.

In AI, "height" means **how wrong the model is** (the *loss*). "Walking downhill" means nudging the model's numbers so it gets less wrong.

## 🔵 2. How it works
1. Start with random numbers in the model. It is very wrong.
2. Show the model some examples and measure **how wrong** it is: one number called the **loss**.
3. For every model number, work out: "if I raise this a tiny bit, does the loss go up or down, and by how much?" That list of answers is the **gradient**.
4. Nudge every number a **small step in the opposite direction** of the gradient (downhill).
5. Repeat millions of times. The loss goes down and the model gets better.

The **step size** (learning rate) matters:
- too small: slow, like walking in baby steps
- too big: you jump over the valley and never settle
- just right: steady progress

![[gradient-descent.svg]]
*Contour lines of a landscape: each red arrow is one downhill step from x₀ toward the bottom. Image: Zerodamage, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Gradient_descent.svg), public domain.*

## 🟠 3. Technical detail
Update rule, with parameters $\theta$, learning rate $\eta$ and loss $L$:

$$
\theta_{t+1}=\theta_t-\eta\,\nabla_\theta L(\theta_t)
$$

**Where:**
- $\theta$ (theta) is the list of all the model's adjustable numbers (its parameters). $t$ is the step count, so $\theta_t$ is "the numbers now" and $\theta_{t+1}$ is "the numbers after one more step".
- $L(\theta)$ is the **loss**: how wrong the model is with those numbers. It is the height of the landscape we are walking on.
- $\nabla_\theta L$ (nabla, or "grad") is the **gradient**: for each parameter, how fast the loss rises if you increase it. **Matters:** it points straight uphill, so it tells us which way *not* to go.
- $\eta$ (eta) is the **learning rate**, the step size. **Matters:** too small is slow, too large overshoots and may diverge.
- The minus sign flips uphill into downhill. **Matters:** without it, the loss would climb instead of fall.

**Example (tiny numbers):** a one-number model with loss $L(\theta)=\theta^2$, so the gradient is $2\theta$. Start at $\theta_0=4$, learning rate $\eta=0.1$.
- Step 1: gradient $=8$. $\theta_1=4-0.1\cdot8=\mathbf{3.2}$.
- Step 2: gradient $=6.4$. $\theta_2=3.2-0.64=\mathbf{2.56}$.
- Step 3: $\theta_3=\mathbf{2.05}$. Each step is smaller because the slope flattens near the bottom at $\theta=0$.
- Too big a step: with $\eta=1.1$, $\theta_1=4-1.1\cdot8=-4.8$. It jumped over the bottom and is now *farther* away, and the next step is bigger still. That is divergence.

The gradient $\nabla L$ is a [[Vector]] of partial derivatives; it points uphill, so we subtract it.

How the gradient is computed: **backpropagation**, the chain rule applied layer by layer from the output back to the input, reusing results. It costs about as much as one forward pass ([[Matrix]] calculus gives each layer's rule).

Variants used in practice:
- **Batch GD:** gradient over the whole dataset. Accurate but slow.
- **Stochastic GD (SGD):** gradient from one example. Noisy but fast, and the noise can help escape bad spots.
- **Mini-batch SGD:** gradient from e.g. 32 to millions of tokens. The standard.
- **Momentum:** keep part of the previous step, like a rolling ball, to smooth the path and go faster through flat parts.
- **Adam / AdamW:** keeps a running average of gradients and of squared gradients, giving every parameter its own step size. The default optimizer for [[Transformer]]s and [[LLM]]s.
- **Learning-rate schedule:** warm up from small, then decay (e.g. cosine). Almost always used for LLM training.
- **Gradient clipping:** cap the gradient size so one bad batch cannot wreck the model.

```python
import numpy as np
# minimise f(x, y) = x^2 + 10 y^2 with plain gradient descent
grad = lambda p: np.array([2 * p[0], 20 * p[1]])
p, lr = np.array([4.0, 1.0]), 0.05
for step in range(100):
    p = p - lr * grad(p)
print(p)    # close to [0, 0]
```

Usage in AI: training and [[Fine-tuning]] a model is gradient descent on its [[Matrix]] weights, with the loss from the task (next-token prediction for an [[LLM]]).

## 🔴 4. Deep / formal
**Why it works locally:** by Taylor expansion, $L(\theta-\eta g)\approx L(\theta)-\eta\lVert g\rVert^2+O(\eta^2)$ with $g=\nabla L(\theta)$, so a small enough step always lowers the loss.

**Convergence (convex, $\beta$-smooth):** with $\eta\le1/\beta$, gradient descent satisfies $L(\theta_T)-L^*\le\frac{\lVert\theta_0-\theta^*\rVert^2}{2\eta T}$, an $O(1/T)$ rate. For $\mu$-strongly convex $L$ the rate is linear:

$$
\lVert\theta_T-\theta^*\rVert^2\le\left(1-\eta\mu\right)^T\lVert\theta_0-\theta^*\rVert^2
$$

**Where:**
- $\theta^*$ is the true best answer (the bottom of the valley). $\theta_0$ is the starting guess and $\theta_T$ is the guess after $T$ steps.
- $\lVert\theta_T-\theta^*\rVert^2$ is the squared distance from the best answer. **Matters:** it is the "how far am I still from done?" measure.
- $\mu$ (mu) is how strongly curved the bowl is at the least (strong convexity). **Matters:** a steeper, rounder bowl means every step makes more progress.
- $\eta$ is the learning rate. $(1-\eta\mu)$ is a number between 0 and 1: the fraction of the distance *left* after one step.
- The exponent $T$ means that fraction is applied again at every step. **Matters:** the error shrinks by a constant factor each step, so it falls exponentially fast (called *linear convergence*).
- The condition number $\kappa=\beta/\mu$ below compares the steepest curvature $\beta$ with the flattest $\mu$; a large gap means a long narrow valley and slow zig-zagging progress.

**Example (tiny numbers):** same bowl $L=\theta^2$, which has $\mu=2$. Use $\eta=0.1$, so the factor is $1-\eta\mu=0.8$.
- Guarantee: after $T=10$ steps the squared distance is at most $0.8^{10}\approx\mathbf{0.107}$ of the start. After $T=20$: at most $\mathbf{0.0115}$.
- Reality for this bowl is even better ($0.64^{T}$: $0.0115$ after 10 steps), so the formula is a safe promise, not an exact prediction.
- Takeaway: every extra step multiplies the remaining error by the same fraction, so the error vanishes quickly.

The **condition number** $\kappa=\beta/\mu$ controls speed; long narrow valleys (large $\kappa$, like $x^2+10y^2$ above) make plain GD zig-zag. Momentum improves the dependence from $\kappa$ to $\sqrt\kappa$.

**Adam:**

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\quad
\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

**Where:**
- $g_t$ is the gradient at step $t$ (the plain downhill direction from above).
- $m_t$ is a running average of recent gradients (the "momentum"). $\beta_1$ (about 0.9) says how much of the past to keep. **Matters:** averaging smooths out noisy gradients, so the path stops jittering.
- $v_t$ is a running average of the *squared* gradients. $\beta_2$ (about 0.999) is its memory. **Matters:** it measures how big each parameter's gradients usually are.
- $\hat m_t,\hat v_t$ are bias-corrected versions. **Matters:** both averages start at 0, which would make early steps too small; the correction fixes that.
- $\sqrt{\hat v_t}$ is the typical gradient size for that parameter. Dividing by it gives every parameter its **own step size**: parameters with huge gradients get smaller steps, those with tiny gradients get bigger ones.
- $\epsilon$ is a tiny number (like $10^{-8}$) added to avoid dividing by zero. $\eta$ is the learning rate as before.

**Example (tiny numbers):** very first step ($t=1$), $\beta_1=0.9$, $\beta_2=0.999$. Compare a parameter with a small gradient and one with a huge gradient.
- Small: $g=0.5$. $m_1=0.1\cdot0.5=0.05$, $\hat m_1=0.05/0.1=0.5$. $v_1=0.001\cdot0.25=0.00025$, $\hat v_1=0.25$, $\sqrt{\hat v_1}=0.5$. Step $=\eta\cdot0.5/0.5=\mathbf{1\cdot\eta}$.
- Huge: $g=50$. $\hat m_1=50$, $\sqrt{\hat v_1}=50$. Step $=\eta\cdot50/50=\mathbf{1\cdot\eta}$.
- Both parameters move by about the same amount $\eta$, even though one gradient is 100 times bigger. That per-parameter evening-out is why Adam is easy to use.

AdamW adds weight decay separately from the gradient step.

**Non-convex reality:** neural-network losses are non-convex with saddle points and many minima. In high dimension most critical points are saddles rather than bad local minima, and SGD noise helps leave them. Large networks are also heavily over-parameterised, so many different minima fit the data equally well; flatter minima tend to generalise better, an active research topic. Practical issues: vanishing/exploding gradients (fixed by residual connections, normalisation, careful initialisation), and sensitivity to learning rate, which is why schedules and warm-up matter.

**What came before, and how this improved on it:**
- **Before:** try many candidate values and keep the best (grid search), or solve the equations exactly when that is possible. Gradient descent is credited to Cauchy in 1847 (from memory, not checked).
- **Its limit:** a grid with only 10 values per weight needs $10^P$ evaluations for $P$ weights.
- **What changed:** look at the slope where you stand and step downhill, so the cost depends on the number of steps, not on the number of combinations.
- **How much better:** in the code below (a smooth bowl, where descent is easy), 10 weights need $10{,}000{,}000{,}000$ grid evaluations against 26 gradient steps. Real losses are not bowls: descent can stop in a local dip or a flat patch, and this toy does not show that.

**Check it yourself:**

```python
import numpy as np
# minimise f(w) = sum((w - 1)^2) over P weights, to within 0.01 of the best value
for P in (2, 5, 10):
    grid = 10 ** P                                   # try 10 values per weight, every combination
    w, steps = np.zeros(P), 0
    while ((w - 1) ** 2).sum() > 1e-2 ** 2:
        w -= 0.1 * 2 * (w - 1); steps += 1           # gradient descent
    print(f"P={P:2d} weights: grid search {grid:,} evaluations, gradient descent {steps} steps")
```

## Related
- [[Vector]] · [[Matrix]] · [[Fine-tuning]] · [[LLM]] · [[Transformer]] · [[RLHF]] · [[GPU]] · [[Stochastic Gradient Descent]] · [[Learning Rate]] · [[Regularization]] · [[Linear Regression]] · [[Convex Optimization]] · [[Eigenvalue]] · [[Q-Learning]] · [[Feature Scaling]] · [[Machine Learning Specialization]] · [[Hessian]]

## Sources
- Goodfellow, Bengio & Courville, *Deep Learning*, ch. 4 and 8
- Kingma & Ba, *Adam: A Method for Stochastic Optimization*, 2014
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW), 2017
- Boyd & Vandenberghe, *Convex Optimization*

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/calculus-optimization Gradient descent:::Finding the bottom of a valley while blindfolded: feel which way the ground slopes down, take a small step that way, and repeat.

#flashcards/math/formula #flashcards/topic/calculus-optimization Write the gradient descent update and explain each symbol.
?
$$\theta_{t+1}=\theta_t-\eta\,\nabla_\theta L(\theta_t)$$
- $\theta$: the list of all the model's adjustable numbers. $t$: the step count, so $\theta_t$ is "now" and $\theta_{t+1}$ is "after one more step".
- $L(\theta)$: the loss, how wrong the model is. It is the height of the landscape.
- $\nabla_\theta L$: the gradient, for each parameter how fast the loss rises if you increase it. It points uphill, so we subtract it.
- $\eta$: the learning rate, the step size. Too small is slow, too big overshoots.

#flashcards/math/formula #flashcards/topic/calculus-optimization Write the convergence bound for gradient descent on a strongly convex bowl and explain each symbol.
?
$$\lVert\theta_T-\theta^*\rVert^2\le\left(1-\eta\mu\right)^T\lVert\theta_0-\theta^*\rVert^2$$
- $\theta^*$: the true best answer. $\theta_0$: the start. $\theta_T$: the guess after $T$ steps.
- $\lVert\theta_T-\theta^*\rVert^2$: the squared distance still left to go.
- $\mu$: how strongly curved the bowl is at the least (strong convexity). A rounder bowl means more progress per step.
- $(1-\eta\mu)$: a number between 0 and 1, the fraction of the distance left after one step, applied $T$ times.

#flashcards/math/formula #flashcards/topic/calculus-optimization Write the Adam update and explain each symbol.
?
$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,\quad \theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$
- $g_t$: the gradient at step $t$.
- $m_t$: a running average of recent gradients (momentum). $\beta_1$ (about 0.9) is how much past to keep. It smooths noisy gradients.
- $v_t$: a running average of squared gradients. $\beta_2$ (about 0.999) is its memory. It measures how big each parameter's gradients usually are.
- $\hat m_t$, $\hat v_t$: bias-corrected versions, since both averages start at 0 and early steps would be too small.
- $\sqrt{\hat v_t}$: the typical gradient size, so each parameter gets its own step size. $\epsilon$: a tiny number (like $10^{-8}$) that avoids dividing by zero.

#flashcards/math/cloze #flashcards/topic/calculus-optimization The gradient points ==uphill==, so gradient descent ==subtracts== it. A learning rate that is too ==big== overshoots.

#flashcards/math/example #flashcards/topic/calculus-optimization For $L(\theta)=\theta^2$ with $\theta_0=4$ and $\eta=0.1$, what are the first three steps?::Gradient 8, so θ₁ = 4 − 0.8 = 3.2. Gradient 6.4, so θ₂ = 3.2 − 0.64 = 2.56. θ₃ = 2.05. Each step is smaller because the slope shrinks.

#flashcards/math/example #flashcards/topic/calculus-optimization For $L=\theta^2$ ($\mu=2$, $\eta=0.1$), what does the bound guarantee after 10 and 20 steps?::The factor is 1 − 0.2 = 0.8. After 10 steps at most 0.8¹⁰ ≈ 0.107 of the starting squared distance, after 20 at most 0.0115. The real rate here is even better (0.64ᵗ).

#flashcards/math/image #flashcards/topic/calculus-optimization ![[gradient-descent.svg]]
What does this picture show?
?
Contour lines of a landscape. Each red arrow is one downhill step from the start x₀ towards the bottom.
