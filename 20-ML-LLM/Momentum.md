---
tags: [machine-learning, optimization, momentum, SGD, training]
status: growing
level: 4
tier: 2
---

# Momentum

## 🟢 1. Simple (age 12)
Momentum is **making the learning steps behave like a rolling ball**. A ball rolling downhill does not stop and restart at every bump. It **builds up speed** when the slope keeps pointing the same way, and its speed **smooths out** little wobbles. Adding this "memory of where I was going" lets a model move faster down long slopes and zigzag less.

**Example:** Pushing a heavy shopping trolley down a long corridor. The first push barely moves it, but if you keep pushing in the same direction it speeds up. If someone gives it small sideways nudges, it hardly changes course, because its momentum carries it straight on. A light cart with no momentum would swerve at every nudge.

## 🔵 2. How it works
1. Ordinary training ([[Gradient Descent]], [[Stochastic Gradient Descent]]) takes each step only from the **current** slope (the gradient). It has no memory.
2. Momentum adds a memory called the **velocity**: a running total of recent gradients.
3. Each step, **new velocity = (a fraction of the old velocity) + (the new gradient)**. The fraction is a number $\beta$ close to 1, such as $0.9$.
4. Then the weights move along the **velocity**, not along the raw gradient.
5. When the gradient keeps pointing the **same way** step after step, the velocity **builds up**, and steps get bigger. This speeds up travel along a long gentle slope.
6. When the gradient **flips back and forth** (as it does across a narrow valley), the pushes **cancel** in the velocity. This calms the zigzag.
7. The cost is a little overshoot: with speed built up, the weights can roll past the bottom and come back, like a ball in a bowl.

![[gradient-descent-momentum.svg]]
*Why remembering helps: the contour lines are a bowl with its lowest point in the middle. Without moving averages (right) the steps bounce back and forth across the bowl and make slow progress. With moving averages (left) the bounces are smoothed out and the path heads toward the minimum. Image: Babayaga94, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Gradient_descent_with_momentum.svg), CC0. (The same image is used in [[Adam Optimizer]].)*

## 🟠 3. Technical detail
- **State:** one extra number per weight (the velocity), so it costs as much extra memory as the weights themselves.
- **The setting $\beta$:** the "momentum coefficient", often $0.9$ (a common habit; check on your problem). Closer to 1 means a longer memory, more speed on straight slopes, and more overshoot.
- **A rule of thumb:** the velocity averages roughly the last $1/(1-\beta)$ gradients. With $\beta=0.9$ that is about 10 steps. (An informal reading of the formula.)
- **Effective step:** on a steady slope the step ends up about $1/(1-\beta)$ times larger than plain SGD's, so people often lower the [[Learning Rate]] when they turn momentum on or raise $\beta$.
- **Two common ways to write it:** one adds the raw gradient to the velocity (used below, as in the infographic that inspired this note), another uses $(1-\beta)\times$ the gradient. They are the same method with a rescaled learning rate, so check which one your library uses.
- **Nesterov momentum:** a variant that computes the gradient at the point you would reach if you just kept coasting, a "look ahead". It often reacts a little better at turns (from memory; see the papers).
- **Used in:** SGD with momentum is a standard choice for training many vision networks. Adaptive methods such as [[RMSProp]] and [[Adam Optimizer]] combine the speed-up idea with a per-weight step size ([[Adam Optimizer]] is essentially momentum plus [[RMSProp]]-style scaling, with bias correction).
- **Problems it helps with:** long narrow valleys, noisy gradients from small batches, and slow progress on flat stretches. It does not fix a bad learning rate, or vanishing gradients ([[Vanishing and Exploding Gradients]]).

```python
import numpy as np

def mom(gs, b=0.9, eta=0.1):                      # momentum: v = b*v + g ; theta -= eta*v
    v = 0; th = 0; out = []
    for g in gs:
        v = b * v + g; th -= eta * v; out.append((round(v, 3), round(th, 4)))
    return out
print(mom([1, 1, 1]))          # [(1.0, -0.1), (1.9, -0.29), (2.71, -0.561)]   speeds up on a steady slope

v = 0
for t in range(200): v = 0.9 * v + 1                                  # steady gradient +1, +1, +1, ...
w = 0
for t in range(200): w = 0.9 * w + (1 if t % 2 == 0 else -1)          # gradient flipping +1, -1, +1, ...
print(round(v, 3), round(w, 3))    # 10.0 -0.526   steady: grows to 10x. flipping: stays small

# a narrow valley f = 0.5*(w1^2 + 25*w2^2): steps until both weights are within 0.001 of 0
def run(beta, eta, steps=2000):
    w = np.array([1., 1.]); vel = np.zeros(2); h = np.array([1., 25.])
    for t in range(steps):
        vel = beta * vel + h * w; w = w - eta * vel
        if np.abs(w).max() < 1e-3: return t + 1
print(run(0.0, 0.07), run(0.7, 0.03))      # 96 39     plain gradient descent vs momentum (settings picked for this toy)
```

The last line is a toy. I picked the settings for the momentum run by trying a small grid, and a different valley or other settings would give a different comparison. With $\beta=0.9$ and the same learning rates the same loop is not faster (it overshoots), so momentum needs the learning rate re-tuned.

## 🔴 4. Deep / formal
**The momentum update.** Let $g_t$ be the gradient at step $t$, $v_t$ the velocity (starting at $v_0=0$), $\eta$ the [[Learning Rate]] and $\beta\in[0,1)$ the momentum coefficient:

$$
v_t=\beta\,v_{t-1}+g_t,\qquad \theta_{t+1}=\theta_t-\eta\,v_t
$$

**Where:**
- $\theta_t$ is the weight at step $t$ and $g_t=\nabla L(\theta_t)$ is the gradient of the [[Loss Function]] there (see [[Calculus]]).
- $v_t$ is the **velocity**: a running total of the gradients, where each older gradient counts $\beta$ times less per step that passes.
- $\beta$ ("beta") sets the memory. **Matters:** with $\beta=0$ the velocity is just $g_t$ and this is plain gradient descent. With $\beta$ near 1 old gradients linger for a long time.
- $\eta$ scales the move. The minus sign again points downhill.
- **Matters:** unrolling the formula, $v_t=g_t+\beta g_{t-1}+\beta^2g_{t-2}+\cdots$, so each step is a weighted sum of all past gradients, with recent ones counting most.

**Example (tiny numbers):** a gradient of $g=1$ at every step, $\beta=0.9$, $\eta=0.1$, starting at $\theta=0$ and $v=0$.
- Step 1: $v=0.9\cdot0+1=\mathbf{1}$, so $\theta=0-0.1\cdot1=\mathbf{-0.1}$.
- Step 2: $v=0.9\cdot1+1=\mathbf{1.9}$, so $\theta=-0.1-0.19=\mathbf{-0.29}$.
- Step 3: $v=0.9\cdot1.9+1=\mathbf{2.71}$, so $\theta=-0.29-0.271=\mathbf{-0.561}$.
- Meaning: the steps are $0.1,\ 0.19,\ 0.271$ and keep growing, while plain gradient descent would take a fixed $0.1$ each time.

**How fast does it get? The terminal speed and the cancelling effect.** If the gradient stays at a constant $g$, the velocity settles at a limit. If the gradient alternates in sign, it settles at a much smaller size:

$$
v_\infty=\frac{g}{1-\beta}\quad(\text{steady }g),\qquad |v_\infty|=\frac{|g|}{1+\beta}\quad(\text{alternating }\pm g)
$$

**Where:**
- $v_\infty$ is the velocity after a very long time ($t\to\infty$). $g$ is the size of the gradient, and $\beta$ is the momentum coefficient.
- **Steady case:** setting $v=\beta v+g$ and solving gives $v=g/(1-\beta)$. **Matters:** the speed-up factor is $1/(1-\beta)$: ten times for $\beta=0.9$, a hundred times for $\beta=0.99$.
- **Alternating case:** the pushes cancel, and the size ends up near $|g|/(1+\beta)$, a bit over half of $|g|$ for $\beta=0.9$ (the exact value depends on which sign came last). **Matters:** the zigzag across a valley is damped (and does not grow tenfold), while progress along the valley is amplified. This one pair of numbers is the whole reason momentum helps in narrow valleys.
- The alternating formula is the size of the long-run swing and is checked numerically in the code, not by a full derivation here.

**Example (tiny numbers):** $g=1$ and $\beta=0.9$.
- Steady: $v_\infty=1/(1-0.9)=\mathbf{10}$. The code (200 steps) prints $10.0$.
- Alternating $+1,-1,+1,\dots$: $|v_\infty|\approx1/(1+0.9)=\mathbf{0.526}$. The code prints $-0.526$.
- Meaning: the same $\beta=0.9$ amplifies a consistent push tenfold, and keeps a back-and-forth push to about half its size, which is why the ball keeps rolling down the valley and does not zigzag.

**Further notes:**
- **Heavy-ball picture:** momentum can be read as a ball with mass moving through a thick fluid, and $\beta$ as how little friction there is. The update is a discretisation of that physics. (A standard intuition, originally due to Polyak, 1964; from memory.)
- **Convergence on a bowl:** on a smooth bowl-shaped loss, with settings chosen from the bowl's steepest and flattest curvature, momentum can converge in far fewer steps than plain gradient descent. The exact gain and the best settings are in the optimisation literature (from memory; check).
- **Overshoot and oscillation:** because speed builds up, the weights can pass the bottom and swing back. A too-high $\beta$ with a too-high $\eta$ can even diverge.
- **Interaction with batch noise:** averaging gradients over steps also averages some of the random noise from mini-batches ([[Stochastic Gradient Descent]]).

**What came before, and how this improved on it:**
- **Before:** plain gradient descent, which zigzags across a long narrow valley and creeps along it.
- **Its limit:** the step size is capped by the steepest direction, so progress along the flat direction is slow.
- **What changed:** keep a running velocity of past gradients, so the zigzags cancel and steady directions build up speed (Polyak, 1964; from memory).
- **How much better:** in the code below, on a valley whose steepness differs 100 times between directions, plain descent needs 440 steps to reach loss $10^{-6}$ and momentum 0.9 needs 138, 3.2 times fewer. Plain descent used a step size near its largest stable value, so it was given a fair try.

**Check it yourself:**

```python
import numpy as np
H = np.array([1.0, 100.0])                            # steep one way, flat the other: f = 0.5 * sum(H * w^2)
def run(beta, lr):
    w, v, n = np.array([1.0, 1.0]), np.zeros(2), 0
    while 0.5 * (H * w ** 2).sum() > 1e-6 and n < 10000:
        v = beta * v + H * w; w = w - lr * v; n += 1
    return n
print("steps to reach loss 1e-6:  plain gradient descent (best lr 0.0198):", run(0.0, 0.0198), "  momentum 0.9 (lr 0.01):", run(0.9, 0.01))
```

## Related
- [[Gradient Descent]] · [[Stochastic Gradient Descent]] · [[Learning Rate]] · [[RMSProp]] · [[Adam Optimizer]] · [[Loss Function]] · [[Calculus]] · [[Neural Network]] · [[Vanishing and Exploding Gradients]] · [[Convex Optimization]]

## Sources
- Polyak, *Some methods of speeding up the convergence of iteration methods*, 1964 (the heavy-ball method; from memory)
- Sutskever, Martens, Dahl & Hinton, *On the importance of initialization and momentum in deep learning*, 2013 (from memory)
- Goodfellow, Bengio & Courville, *Deep Learning*, 2016, chapter 8
- Ruder, *An overview of gradient descent optimization algorithms*, 2016 (free online; from memory)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/training Momentum:::Making learning steps behave like a rolling ball. It builds up speed when the slope keeps pointing the same way, and its speed smooths out little wobbles.

#flashcards/ml/formula #flashcards/topic/training Write the momentum update and explain each symbol.
?
$$v_t=\beta\,v_{t-1}+g_t,\qquad \theta_{t+1}=\theta_t-\eta\,v_t$$
- $\theta_t$: the weight at step $t$. $g_t=\nabla L(\theta_t)$: the gradient of the loss there.
- $v_t$: the velocity, a running total of the gradients where each older gradient counts $\beta$ times less per step that passes.
- $\beta$: sets the memory. Near 0 means almost no memory, near 1 means a long memory.
- $\eta$: the learning rate. The step uses the velocity instead of the raw gradient.

#flashcards/ml/formula #flashcards/topic/training Write the long-run velocity of momentum and explain each symbol.
?
$$v_\infty=\frac{g}{1-\beta}\quad(\text{steady }g),\qquad |v_\infty|=\frac{|g|}{1+\beta}\quad(\text{alternating }\pm g)$$
- $v_\infty$: the velocity after a very long time. $g$: the size of the gradient. $\beta$: the momentum coefficient.
- Steady case: setting $v=\beta v+g$ gives $v=g/(1-\beta)$. The speed-up is $1/(1-\beta)$: ten times for $\beta=0.9$, a hundred times for $\beta=0.99$.
- Alternating case: the back-and-forth pushes mostly cancel, so the velocity stays small. This is how momentum damps zig-zags.

#flashcards/ml/cloze #flashcards/topic/training Momentum speeds up a ==consistent== push by $1/(1-\beta)$ and ==damps== a back-and-forth one.

#flashcards/ml/example #flashcards/topic/training Gradient $g=1$ every step, $\beta=0.9$, $\eta=0.1$, start at 0. What are $v$ and $\theta$ for the first three steps?::Step 1: v = 1, θ = −0.1. Step 2: v = 1.9, θ = −0.29. Step 3: v = 2.71, θ = −0.561. The velocity keeps growing.

#flashcards/ml/example #flashcards/topic/training With $g=1$ and $\beta=0.9$, what is $v_\infty$ for a steady push, and for alternating +1, −1?::Steady: 1/(1 − 0.9) = 10. Alternating: 1/(1 + 0.9) = 0.526. The same β amplifies a consistent push ten times and keeps a back-and-forth one to about half.

#flashcards/ml/image #flashcards/topic/training ![[gradient-descent-momentum.svg]]
What does this picture show?
?
Why remembering helps. The contour lines are a bowl with its lowest point in the middle. Without moving averages (right) the steps bounce back and forth across the bowl and make slow progress. With moving averages (left) the bounces are smoothed out and the path heads toward the minimum.
