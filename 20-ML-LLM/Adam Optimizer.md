---
tags: [adam, optimizer, optimization, gradient-descent, neural-network, LLM, deep-learning]
status: growing
level: 4
tier: 2
---

# Adam Optimizer

## 🟢 1. Simple (age 12)
Adam is a **smarter way for an AI to take its learning steps**. Plain [[Gradient Descent]] always steps in proportion to how steep the hill feels. Adam remembers where it has been going and checks how bumpy the ground is, so it takes steady, sensible steps and does not zig-zag or lurch.

**Example:** You are walking down a foggy hill to find the bottom. A careless walker takes a giant step when the slope is steep and a tiny step when it is flat, and keeps veering left and right. A wise walker remembers "I have been heading mostly downhill and to the left", keeps moving that way, and takes careful steps where the ground changes wildly. That is Adam.

## 🔵 2. How it works
1. Training needs a **step** for each number (weight) in the model. First, [[Backpropagation]] tells you the slope (the **gradient**) for every weight.
2. Plain gradient descent moves each weight by (learning rate) times (its slope).
3. Adam keeps **two memories per weight**:
   - **Direction memory:** a running average of recent slopes. Slopes that keep pointing the same way add up, and slopes that flip left and right cancel out. This calms the zig-zag. (This idea is called momentum.)
   - **Size memory:** a running average of how big the recent slopes were (squared).
4. It then steps in the remembered direction, but **divides by the typical size**. Weights with huge slopes get a smaller push, and weights with tiny slopes get a bigger one.
5. Result: every weight takes a step of about the same sensible size, no matter how steep or flat its own ground is.
6. It is the standard choice for training [[Neural Network]]s and [[LLM]]s, because it works well with little tuning.

![[gradient-descent-momentum.svg]]
*Why remembering helps: the contour lines are a bowl with its lowest point in the middle. Without moving averages (right, $t_1\to t_7$) the steps bounce back and forth across the bowl and make slow progress. With moving averages (left, $t_1\to t_6$) the bounces are smoothed out and the path heads toward the minimum. Image: Babayaga94, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Gradient_descent_with_momentum.svg), CC0.*

## 🟠 3. Technical detail
- **State per weight:** two numbers, $m$ (average slope) and $v$ (average squared slope). So the optimiser needs about **twice the memory of the weights**, which is why training uses so much [[GPU]] memory (the "12 of 16 bytes per parameter" in the GPU note is mostly Adam).
- **Hyperparameters:** learning rate $\alpha$ (the main one to tune), $\beta_1\approx0.9$, $\beta_2\approx0.999$, and a tiny $\epsilon\approx10^{-8}$ that stops division by zero. These defaults are from the original paper and from the common libraries (from memory; check your library).
- **Bias correction:** $m$ and $v$ start at 0, so early on they are too small. Adam divides by a correction factor that fades away as training goes on.
- **AdamW:** a popular variant that applies weight decay (the penalty from [[Overfitting]]) directly to the weights instead of mixing it into the gradient. Common for [[Transformer]] training.
- **Learning-rate schedule:** real runs usually warm up the learning rate from a small value, then slowly lower it.
- **Alternatives:** plain SGD with momentum (less memory, can generalise well, needs more tuning), Adafactor and 8-bit Adam (less memory), newer methods such as Lion.

```python
import math
def adam(grads, theta=1.0, a=0.1, b1=0.9, b2=0.999, eps=1e-8):
    m = v = 0.0
    for t, g in enumerate(grads, 1):
        m = b1 * m + (1 - b1) * g            # direction memory
        v = b2 * v + (1 - b2) * g * g        # size memory
        mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)   # bias correction
        theta -= a * mh / (math.sqrt(vh) + eps)
        print(t, round(theta, 4))

adam([2, 2])      # 1 0.9   then   2 0.8
adam([200])       # 1 0.9   same step even though the gradient is 100x bigger
print(0.1 * 2, 0.1 * 200)   # plain gradient descent steps: 0.2 vs 20.0
```

## 🔴 4. Deep / formal
**Step 1: the two running averages.** At step $t$, with gradient $g_t$ for one weight:

$$
m_t=\beta_1\,m_{t-1}+(1-\beta_1)\,g_t,\qquad v_t=\beta_2\,v_{t-1}+(1-\beta_2)\,g_t^{2}
$$

**Where:**
- $g_t$ is the gradient (slope of the [[Loss Function]]) for this weight at step $t$, from [[Backpropagation]].
- $m_t$ is the **first moment**: a running average of the gradients. **Matters:** it is the smoothed direction. Gradients that keep the same sign build up, and ones that flip sign cancel.
- $v_t$ is the **second moment**: a running average of the *squared* gradients. **Matters:** it measures the typical size of the gradient and ignores its sign (squares are positive).
- $m_{t-1}$ and $v_{t-1}$ are the previous values, so each is a blend of "what I knew" and "what I just saw". Both start at 0.
- $\beta_1$ and $\beta_2$ (between 0 and 1) set the **memory length**. **Matters:** a value near 1 (such as $0.999$) remembers a long history (about the last $1/(1-\beta)$ steps) and reacts slowly. A smaller one forgets faster.
- $(1-\beta)$ is the weight given to the newest gradient.

**Example (tiny numbers):** $\beta_1=0.9$, $\beta_2=0.999$. The gradient is $g=2$ at step 1 and again at step 2.
- Step 1: $m_1=0.9\cdot0+0.1\cdot2=\mathbf{0.2}$. $v_1=0.999\cdot0+0.001\cdot4=\mathbf{0.004}$.
- Step 2: $m_2=0.9\cdot0.2+0.1\cdot2=\mathbf{0.38}$. $v_2=0.999\cdot0.004+0.001\cdot4=\mathbf{0.007996}$.
- Notice $m_1=0.2$ is far below the true gradient $2$: the averages start too small because they start at 0. That is what the next step fixes.

**Step 2: correct and move.** Fix the early shrinkage, then take the step with learning rate $\alpha$:

$$
\hat m_t=\frac{m_t}{1-\beta_1^{\,t}},\quad \hat v_t=\frac{v_t}{1-\beta_2^{\,t}},\qquad \theta_t=\theta_{t-1}-\alpha\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

**Where:**
- $\theta_t$ is the weight after step $t$, and $\theta_{t-1}$ the weight before it. $\alpha$ is the **learning rate**: the maximum rough size of a step. **Matters:** it is the main knob. Too big and training becomes unstable, too small and it crawls.
- $\hat m_t,\hat v_t$ are the **bias-corrected** averages. $\beta^{\,t}$ is $\beta$ multiplied by itself $t$ times, which shrinks toward 0, so the correction factor $1-\beta^{\,t}$ rises toward 1. **Matters:** early on it scales the averages back up to their true size, and later it does almost nothing.
- $\sqrt{\hat v_t}$ is the typical gradient size (the square root undoes the squaring). **Matters:** dividing by it makes every weight take a step of similar size whatever its gradient scale. Weights with big, noisy gradients are slowed, and weights with small gradients are sped up.
- $\epsilon$ is a tiny constant such as $10^{-8}$. **Matters:** without it, a weight whose gradients are all exactly 0 would divide by 0.
- $\hat m_t/\sqrt{\hat v_t}$ is a ratio of "direction" to "size", roughly between $-1$ and $1$, so a step is about $\alpha$ at most.

**Example (tiny numbers):** continue with $\alpha=0.1$, starting weight $\theta_0=1$, $g=2$ each step.
- Step 1: $\hat m_1=0.2/(1-0.9)=\mathbf{2}$ and $\hat v_1=0.004/(1-0.999)=\mathbf{4}$. Step $=0.1\cdot2/(\sqrt4+10^{-8})=0.1\cdot2/2=\mathbf{0.1}$. So $\theta_1=1-0.1=\mathbf{0.9}$.
- Step 2: $\hat m_2=0.38/(1-0.81)=2.0$ and $\hat v_2=0.007996/(1-0.998001)=4.0$. Step is $0.1$ again, so $\theta_2=\mathbf{0.8}$.
- A hundred times bigger gradient, $g=200$: $\hat m_1=200$ and $\hat v_1=40000$, so the step is $0.1\cdot200/200=\mathbf{0.1}$. The same step.
- Plain gradient descent with the same $0.1$ rate would move $0.1\cdot2=0.2$ for $g=2$ but $0.1\cdot200=\mathbf{20}$ for $g=200$, a wild jump.
- Meaning: Adam's step size is set by $\alpha$, not by how loud the gradient is. That is why it is forgiving to use.

**Further notes:**
- **Origin:** Kingma and Ba, *Adam: A Method for Stochastic Optimization* (2014 preprint, ICLR 2015). The name comes from "adaptive moment estimation".
- **Convergence caveat:** the original convergence proof was later shown to have a flaw, and variants such as AMSGrad were proposed. In practice Adam works very well anyway. (From memory of the literature; check before relying on it.)
- **Memory cost:** the two extra numbers per weight are why a model needs several times its own size in memory to train (see [[GPU]]), and why people use sharding or 8-bit optimisers for very large models.
- **Generalisation:** some studies found plain SGD with momentum can generalise slightly better on some image tasks, while Adam-type methods dominate for [[Transformer]]s. This depends on the task and is debated.
- **Per-weight rates:** because of the division by $\sqrt{\hat v}$, every weight effectively gets its own learning rate. That is the "adaptive" part of the name.

**What came before, and how this improved on it:**
- **Before:** momentum (a velocity) and RMSProp (a per-weight scale), used separately.
- **Its limit:** you had to combine and tune them yourself.
- **What changed:** Adam keeps both running averages, with a correction for their start-up bias. The abstract says it is invariant to diagonal rescaling of the gradients, needs little tuning, and suits noisy or sparse gradients (Kingma and Ba, arXiv 1412.6980, 22 December 2014, abstract read).
- **How much better:** honestly, not on this toy. After 300 steps on a valley with curvatures 1 and 100, the losses are gradient descent $5.2\times10^{-6}$, momentum $1.9\times10^{-13}$, RMSProp $1.7\times10^{-18}$ and Adam $5.9\times10^{-13}$. All learning rates were hand-picked, and Adam's advertised strengths (noise, sparsity, little tuning) are not tested here.

**Check it yourself:**

```python
import torch
def run(make, steps=300):
    w = torch.tensor([1.0, 1.0], requires_grad=True); H = torch.tensor([1.0, 100.0]); opt = make([w])
    for _ in range(steps):
        opt.zero_grad(); loss = 0.5 * (H * w ** 2).sum(); loss.backward(); opt.step()
    return loss.item()
print("loss after 300 steps, curvatures 1 and 100:")
for name, mk in (("gradient descent", lambda p: torch.optim.SGD(p, 0.019)),
                 ("momentum 0.9", lambda p: torch.optim.SGD(p, 0.01, momentum=0.9)),
                 ("RMSProp", lambda p: torch.optim.RMSprop(p, 0.01)),
                 ("Adam", lambda p: torch.optim.Adam(p, 0.1))):
    print(f"  {name:17s} {run(mk):.2e}")
```

## Related
- [[Gradient Descent]] · [[Backpropagation]] · [[Loss Function]] · [[Neural Network]] · [[Overfitting]] · [[GPU]] · [[LLM]] · [[Transformer]] · [[Calculus]] · [[Stochastic Gradient Descent]] · [[Learning Rate]] · [[Regularization]] · [[Momentum]] · [[RMSProp]] · [[Vanishing and Exploding Gradients]] · [[GAN]] · [[Mixed Precision Training]] · [[Data Parallelism]]
- [[Deep Q-Network]]
## Sources
- Kingma & Ba, *Adam: A Method for Stochastic Optimization*, 2014/2015
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW), 2017/2019 (from memory)
- Ruder, *An overview of gradient descent optimization algorithms*, 2016 (free online)
- Kingma and Ba, arXiv 1412.6980 (abstract read for the history block)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/training Adam optimizer:::A smarter way to take learning steps. It remembers which way it has been going (momentum) and how big each weight's gradients usually are, so steps are steady and neither zig-zag nor lurch.

#flashcards/ml/formula #flashcards/topic/training Write Adam's two moving averages and explain each symbol.
?
$$m_t=\beta_1\,m_{t-1}+(1-\beta_1)\,g_t,\qquad v_t=\beta_2\,v_{t-1}+(1-\beta_2)\,g_t^{2}$$
- $g_t$: the gradient for this weight at step $t$, from backpropagation.
- $m_t$: the first moment, a running average of the gradients. It is the smoothed direction. Gradients that keep the same sign build up, ones that flip sign cancel.
- $v_t$: the second moment, a running average of the squared gradients. It tracks how big this weight's gradients typically are.
- $\beta_1$ (about 0.9) and $\beta_2$ (about 0.999): how much past to keep for each.

#flashcards/ml/formula #flashcards/topic/training Write Adam's bias correction and update, and explain each symbol.
?
$$\hat m_t=\frac{m_t}{1-\beta_1^{\,t}},\quad \hat v_t=\frac{v_t}{1-\beta_2^{\,t}},\qquad \theta_t=\theta_{t-1}-\alpha\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$
- $\theta_t$: the weight after step $t$. $\alpha$: the learning rate, the rough maximum size of a step.
- $\hat m_t$, $\hat v_t$: the bias-corrected averages. Both averages start at 0, so early on they are too small. Dividing by $1-\beta^t$ fixes that.
- $\sqrt{\hat v_t}$: the typical gradient size. Dividing by it gives each weight its own step size.
- $\epsilon$: a tiny number that avoids dividing by zero.

#flashcards/ml/cloze #flashcards/topic/training Adam divides the smoothed direction by the typical ==gradient size==, so each weight's step is about ==α== regardless of how large its gradient is.

#flashcards/ml/example #flashcards/topic/training $\beta_1=0.9$, $\beta_2=0.999$, $g=2$ at steps 1 and 2. What are $m$ and $v$?::Step 1: m₁ = 0.1 × 2 = 0.2, v₁ = 0.001 × 4 = 0.004. Step 2: m₂ = 0.9 × 0.2 + 0.2 = 0.38, v₂ = 0.999 × 0.004 + 0.004 = 0.007996. They start far below the true 2 and 4.

#flashcards/ml/example #flashcards/topic/training Continue with $\alpha=0.1$, $\theta_0=1$, $g=2$ each step. What are $\hat m$, $\hat v$ and $\theta_1$?::m̂₁ = 0.2/0.1 = 2, v̂₁ = 0.004/0.001 = 4. Step = 0.1 × 2/√4 = 0.1, so θ₁ = 0.9. At step 2: m̂₂ = 0.38/0.19 = 2.0, v̂₂ = 4.0, so another step of 0.1 and θ₂ = 0.8.

#flashcards/ml/image #flashcards/topic/training ![[gradient-descent-momentum.svg]]
What does this picture show?
?
Why remembering helps. The contour lines are a bowl with its lowest point in the middle. Without moving averages (right) the steps bounce back and forth across the bowl and make slow progress. With moving averages (left) the bounces are smoothed out and the path heads toward the minimum.
