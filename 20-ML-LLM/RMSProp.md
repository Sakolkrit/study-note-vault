---
tags: [machine-learning, optimization, RMSProp, adaptive-learning-rate, training]
status: growing
level: 4
tier: 2
---

# RMSProp

## 🟢 1. Simple (age 12)
RMSProp is a way of making **each number in the model take a sensible-sized step**. Some numbers sit on a steep slope and get huge gradients. Others sit on a flat slope and get tiny ones. If they all use the same step rule, the steep ones lurch around and the flat ones barely move. RMSProp **watches how big each number's recent gradients have been** and **divides by that**, so steep ones are calmed down and flat ones are speeded up.

**Example:** Two kids are pushing on a swing set. One swing is very light and flies high with the tiniest push. The other is very heavy and hardly moves. If both kids push with equal force, one swing goes wild and the other stays put. A good coach says: "push a little less on the light swing, a little more on the heavy one." RMSProp is that coach, working out the right push for every number.

## 🔵 2. How it works
1. Backpropagation ([[Backpropagation]]) gives a gradient (slope) for every weight. Plain [[Gradient Descent]] steps each weight by (learning rate) times (its gradient).
2. RMSProp keeps **one extra memory per weight**: a running average of the **squared** gradients. Squaring makes every gradient positive, so the memory records how **big** the recent gradients were, not which way they pointed.
3. Each step, update the memory: **new memory = (most of the old memory) + (a little of the new squared gradient)**.
4. Take the **square root** of the memory. That is the typical gradient size for this weight (the "root mean square", which is where the name comes from).
5. Step the weight by (learning rate) times (gradient) **divided by that typical size**.
6. Result: a weight with big gradients gets a smaller push. A weight with tiny gradients gets a bigger one. All weights move at about the **same speed**, whatever the steepness of their own ground.

No image: I did not find a suitable free picture of RMSProp itself on Wikimedia Commons (the search hits were unrelated), so I left it out. The pictures in [[Momentum]] and [[Adam Optimizer]] show related ideas.

## 🟠 3. Technical detail
- **State:** one number per weight (the average squared gradient). So it needs about as much extra memory as the weights, the same as [[Momentum]].
- **Settings:**
  - The learning rate $\eta$, often small (around $10^{-3}$ or $10^{-4}$ is a common habit; check on your problem).
  - The decay $\beta$, often $0.9$ (some write $0.99$). It sets how long the memory is: about $1/(1-\beta)$ steps.
  - A tiny $\epsilon$ (around $10^{-8}$) added so we never divide by zero.
- **Why it helps:**
  - **Different scales:** a weight whose gradients are always large and one whose gradients are always small both take comparable steps, so you do not need one perfect learning rate for every weight.
  - **Noisy or sparse gradients:** weights that rarely get a gradient get big steps when they do.
- **Why a running average and not a total:** an earlier method (Adagrad) adds up *all* squared gradients, so its steps shrink forever and eventually stall. RMSProp's average **forgets** old gradients, so it can keep learning. (From memory.)
- **Origin:** RMSProp was proposed by Geoffrey Hinton in a lecture of an online course (about 2012), not in a paper, so it is usually cited as that lecture (from memory; check).
- **Relation to Adam:** [[Adam Optimizer]] is, roughly, RMSProp plus [[Momentum]] on the gradient, plus a bias correction for the early steps. Many libraries also offer RMSProp with an optional momentum term.
- **Limits:** the step size now depends on a noisy estimate, so it can still be jumpy. It does not fix [[Vanishing and Exploding Gradients]] by itself, and a bad learning rate still hurts ([[Learning Rate]]).
- **Used in:** some reinforcement-learning systems and recurrent networks historically (from memory), and today most often replaced by [[Adam Optimizer]].

```python
import numpy as np
g = np.array([10., 0.1])                       # two weights: one with big gradients, one with tiny
v = (1 - 0.9) * g ** 2                         # first update of the memory (decay 0.9), starting from 0
print(v, np.sqrt(v))                           # [1.e+01 1.e-03] [3.16227766 0.03162278]
print(g / np.sqrt(v))                          # [3.16227766 3.16227766]   after dividing: both the same size
print(0.01 * g / np.sqrt(v))                   # [0.03162278 0.03162278]   so both weights move the same amount

v = 0
for t in range(1, 51):                         # a steady gradient of 5 over many steps
    v = 0.9 * v + 0.1 * 25
    if t in (1, 10, 50): print(t, round(v, 3), round(5 / np.sqrt(v), 3))
# 1 2.5 3.162
# 10 16.283 1.239
# 50 24.871 1.003       the scaled gradient settles to 1, so every step is about eta in size
```

## 🔴 4. Deep / formal
**The RMSProp update.** For one weight $\theta$ with gradient $g_t$ at step $t$ (and memory $v_0=0$):

$$
v_t=\beta\,v_{t-1}+(1-\beta)\,g_t^{2},\qquad \theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{v_t}+\epsilon}\,g_t
$$

**Where:**
- $g_t$ is this weight's gradient at step $t$, and $g_t^2$ is its square.
- $v_t$ is the memory: an **exponential moving average** of the squared gradients. $\beta$ ("beta") is the decay, between 0 and 1. **Matters:** a larger $\beta$ means a longer memory, a smoother estimate, but slower reaction when the gradient size changes. With a small $\beta$ the estimate is jumpy.
- $\sqrt{v_t}$ is the typical gradient size (root mean square). $\epsilon$ ("epsilon") is a tiny constant. **Matters:** without $\epsilon$, a weight whose gradients have all been zero would give a division by zero.
- $\eta$ is the [[Learning Rate]]. **Matters:** the step is $\eta\cdot g_t/\sqrt{v_t}$, so it is the *gradient divided by its own typical size*. Every weight takes a step of order $\eta$, no matter whether its gradients are huge or tiny.
- The first update starts from $v_0=0$, so the early $v_t$ values are too small, which makes the early steps larger than intended (Adam's bias correction fixes this; here it is left as is).
- Everything is done separately for each weight (element-wise), including the square and the division.

**Example (tiny numbers):** two weights with gradients $g=(10,\ 0.1)$, $\beta=0.9$, $\eta=0.01$, and $v_0=0$ (ignoring $\epsilon$, which is negligible here).
- Memory: $v=(0.1\cdot100,\ 0.1\cdot0.01)=\mathbf{(10,\ 0.001)}$. Typical sizes: $\sqrt v=\mathbf{(3.162,\ 0.0316)}$.
- Scaled gradients: $g/\sqrt v=(10/3.162,\ 0.1/0.0316)=\mathbf{(3.162,\ 3.162)}$. They are now equal, though the raw gradients differed by 100 times.
- Steps: $0.01\cdot3.162=\mathbf{0.0316}$ for both weights.
- Plain gradient descent with the same $\eta$ would move them by $0.1$ and $0.001$, a 100 times difference.
- Meaning: RMSProp has equalised the two weights' speeds.

**Where the steps settle: why it behaves like a sign-step.** If a weight's gradient stays at a constant value $g$, the memory climbs towards $g^2$:

$$
v_t=(1-\beta^{t})\,g^{2}\ \longrightarrow\ g^{2},\qquad \text{so the step}\ \to\ \eta\,\frac{g}{\lvert g\rvert}=\pm\eta
$$

**Where:**
- $\beta^t$ is $\beta$ multiplied by itself $t$ times. It shrinks towards 0 as $t$ grows, so $1-\beta^t$ rises from $1-\beta$ towards $1$.
- $v_t\to g^2$ means the long-run typical size equals the actual size of the constant gradient. The step $\eta g/\sqrt{v_t}$ then becomes $\eta g/|g|$, which is $+\eta$ or $-\eta$ (only the sign of $g$ matters).
- **Matters:** while the gradient is steady, RMSProp moves each weight about a fixed distance $\eta$ per step, in the direction of the gradient's sign, however small or large $|g|$ is. So the step does not shrink just because the gradient got small. To settle near a minimum you still lower $\eta$ over time ([[Learning Rate]]).
- This is a long-run description for constant $g$. With real, changing gradients it is only a rough guide.

**Example (tiny numbers):** a constant gradient $g=5$ (so $g^2=25$) and $\beta=0.9$.
- $t=1$: $v=(1-0.9)\cdot25=\mathbf{2.5}$, scaled gradient $5/\sqrt{2.5}=\mathbf{3.162}$ (an over-large early step).
- $t=10$: $v=(1-0.9^{10})\cdot25=0.6513\cdot25=\mathbf{16.28}$, scaled $\mathbf{1.239}$.
- $t=50$: $v\approx\mathbf{24.87}$, scaled $\mathbf{1.003}$.
- Meaning: after a few dozen steps the scaled gradient is 1 and each step has size $\eta$. The early over-large steps are the start-up effect that Adam's bias correction was added to remove.

**Further notes:**
- **A diagonal preconditioner:** dividing by $\sqrt{v_t}$ rescales each coordinate separately, like a cheap, diagonal approximation of using curvature information. It cannot undo a valley that is tilted across several weights. (A common explanation, stated informally.)
- **Adagrad, RMSProp, Adam:** Adagrad sums all squared gradients, RMSProp averages them with forgetting, and Adam adds momentum and bias correction to the RMSProp idea (from memory; the papers give the exact definitions).
- **Non-convergence worries:** adaptive methods can fail to converge on some constructed problems, which led to variants (research, from memory; no details given here).
- **Centered RMSProp:** a variant that also subtracts the squared running average of the gradient (so $v$ estimates a variance instead of a mean square). Offered in some libraries (from memory).

**What came before, and how this improved on it:**
- **Before:** one learning rate for every weight.
- **Its limit:** when weights have very different gradient sizes (here curvatures 1 and 1000), the rate must be small enough for the steepest one, which starves the rest.
- **What changed:** divide each weight's step by a running average of its recent gradient size. It came from a lecture by Hinton (2012; not a paper, from memory).
- **How much better:** in the code below, after 300 steps gradient descent (at its largest safe rate) is at loss 0.1597 and RMSProp with rate 0.01 at 0.0123, about 13 times lower. But RMSProp with rate 0.001 is still at 241.04, so the rate still matters a lot.

**Check it yourself:**

```python
import numpy as np
H = np.array([1.0, 1000.0])
def run(rule, lr, steps=300):
    w, s = np.array([1.0, 1.0]), np.zeros(2)
    for _ in range(steps):
        g = H * w
        if rule == "gd": w = w - lr * g
        else: s = 0.9 * s + 0.1 * g ** 2; w = w - lr * g / (np.sqrt(s) + 1e-8)
    return 0.5 * (H * w ** 2).sum()
print("loss after 300 steps, curvatures 1 and 1000:")
print("  gradient descent (largest safe lr 0.0019):", f"{run('gd', 0.0019):.4f}")
print("  RMSProp lr 0.01:", f"{run('rms', 0.01):.4f}", "  RMSProp lr 0.001:", f"{run('rms', 0.001):.4f}")
```

## Related
- [[Gradient Descent]] · [[Stochastic Gradient Descent]] · [[Momentum]] · [[Adam Optimizer]] · [[Learning Rate]] · [[Backpropagation]] · [[Loss Function]] · [[Neural Network]] · [[Vanishing and Exploding Gradients]] · [[Calculus]]

## Sources
- Tieleman & Hinton, *Lecture 6.5 - RMSProp*, COURSERA: Neural Networks for Machine Learning, 2012 (from memory)
- Duchi, Hazan & Singer, *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization* (Adagrad), 2011 (from memory)
- Kingma & Ba, *Adam: A Method for Stochastic Optimization*, 2015 (from memory)
- Ruder, *An overview of gradient descent optimization algorithms*, 2016 (free online; from memory)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/training RMSProp:::Making each number in the model take a sensibly sized step. It watches how big each weight's gradients usually are and divides by that, so steep weights stop lurching and flat ones still move.

#flashcards/ml/formula #flashcards/topic/training Write the RMSProp update and explain each symbol.
?
$$v_t=\beta\,v_{t-1}+(1-\beta)\,g_t^{2},\qquad \theta_{t+1}=\theta_t-\frac{\eta}{\sqrt{v_t}+\epsilon}\,g_t$$
- $g_t$: this weight's gradient at step $t$. $g_t^2$: its square.
- $v_t$: the memory, an exponential moving average of the squared gradients. $\beta$: the decay between 0 and 1. A larger $\beta$ means a longer memory and a smoother estimate, but a slower reaction when gradient size changes.
- $\sqrt{v_t}$: the typical gradient size. Dividing by it scales each weight's step.
- $\eta$: the learning rate. $\epsilon$: a tiny safety constant.

#flashcards/ml/formula #flashcards/topic/training Write what RMSProp does for a constant gradient and explain each symbol.
?
$$v_t=(1-\beta^{t})\,g^{2}\ \longrightarrow\ g^{2},\qquad \text{so the step}\ \to\ \eta\,\frac{g}{\lvert g\rvert}=\pm\eta$$
- $\beta^t$: $\beta$ multiplied by itself $t$ times. It shrinks towards 0, so $1-\beta^t$ rises from $1-\beta$ towards 1.
- $v_t\to g^2$: the typical size equals the actual size of the constant gradient.
- The step $\eta g/\sqrt{v_t}$ becomes $\eta g/|g|=\pm\eta$. The size of the step no longer depends on the size of the gradient, only its sign.

#flashcards/ml/cloze #flashcards/topic/training RMSProp divides each gradient by its recent ==typical size==, so large and small gradients give steps of similar ==size==.

#flashcards/ml/example #flashcards/topic/training Gradients $g=(10,0.1)$, $\beta=0.9$, $\eta=0.01$, $v_0=0$. What are $v$ and the scaled gradients?::v = (0.1 × 100, 0.1 × 0.01) = (10, 0.001). √v = (3.162, 0.0316). Scaled g/√v = (3.162, 3.162). Both weights get the same step size, 0.0316, even though their gradients differ by 100.

#flashcards/ml/example #flashcards/topic/training A constant gradient $g=5$ and $\beta=0.9$. What is the scaled gradient at $t=1$ and $t=10$?::t = 1: v = 0.1 × 25 = 2.5, scaled 5/√2.5 = 3.162 (an over-large early step). t = 10: v = (1 − 0.9¹⁰) × 25 = 16.28, scaled 1.239. It settles towards 1.
