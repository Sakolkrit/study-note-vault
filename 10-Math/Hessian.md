---
tags: [hessian, calculus, math, optimization, matrix, curvature, deep-learning]
status: growing
level: 4
tier: 3
---

# Hessian

The **Hessian** is a table of numbers that says **how curved a landscape is**. The slope (gradient) tells you which way is downhill. The Hessian tells you **how fast that slope is changing**, in every direction.

## 🟢 1. Simple (age 12)
Picture a hill. The **slope** tells you which way is downhill. But there is a second question: *is the slope about to get steeper, or is it flattening out?* That is the **curvature**. The Hessian is a score card of the curvature, one score for each direction you could walk.

**Example:** a skate-park bowl curves up on every side, so a ball dropped in it rolls to the bottom and stays. A horse saddle curves up from side to side but down from front to back. A ball on a saddle stays for a moment, then rolls off one way. The Hessian is what tells the bowl and the saddle apart, even when the ground is flat right at the middle.

## 🔵 2. How it works
1. A model has a **loss**: a number that says how wrong it is ([[Loss Function]]). The loss depends on the model's weights, so it forms a landscape. Training walks downhill on it ([[Gradient Descent]]).
2. The slope at your feet is the **gradient**: one number per weight ([[Vector]]). Details in [[Calculus]].
3. Now measure how each slope changes when you nudge each *other* weight. With $P$ weights that is $P\times P$ numbers, a square [[Matrix]]. That square is the Hessian.
4. The table is mirrored along its diagonal (the number for "weight 1 against weight 2" equals "weight 2 against weight 1"), so its [[Eigenvalue]]s are all ordinary real numbers. Each eigenvalue is the curvature along one special direction.
5. **Reading the eigenvalues at a flat spot** (where the slope is 0):
   - all positive: a **bowl**, so you are at the bottom of a valley (a minimum);
   - all negative: the top of a hill (a maximum);
   - some positive, some negative: a **saddle**: it curves up one way and down the other;
   - a zero: flat in that direction, so this check cannot tell you more.

![[saddle-point.svg]]
*A saddle surface: it curves up along one axis and down along the other. The red dot is the flat spot in the middle. Image: "Saddle point.svg" by Nicoguaro, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Saddle_point.svg), CC BY 3.0.*

6. **Why training cares.** A steep direction (big eigenvalue) forces a small [[Learning Rate]]: step too far and you overshoot and bounce out. A flat direction (small eigenvalue) then moves very slowly with that same small step. A long narrow valley is exactly this problem.
7. **Newton's method** uses the Hessian to fix it: it takes a big step along flat directions and a small step along steep ones. On a perfect bowl it lands at the bottom in **one step**.
8. **The catch.** For a network with 830,464 weights (the size of the model in [[Tiny GPT from Scratch]]) the table has about 690 billion numbers, far too many to store. So people use tricks that never build the table (level 3).

## 🟠 3. Technical detail
- **What it is.** The Hessian of a loss $f(\mathbf w)$ is the $P\times P$ matrix of second partial derivatives. It is the derivative of the gradient. Where the second derivatives are continuous it is symmetric, so its eigenvalues are real (Wikipedia, read).
- **What it gives you.** (1) The curvature in any direction: $\mathbf d^\top H\mathbf d$. (2) The **second-derivative test**: at a point where the gradient is 0, all-positive eigenvalues mean a local minimum, all-negative a maximum, mixed signs a saddle, and a zero eigenvalue means the test is inconclusive (Wikipedia, read). (3) The **safe step size**: gradient descent on a bowl converges only if $\eta<2/\lambda_{\max}$ ([[Learning Rate]] has the one-weight version).
- **Condition number.** $\kappa=\lambda_{\max}/\lambda_{\min}$ (the steepest curvature over the flattest) tells how long and narrow the valley is, and so how slowly gradient descent crawls ([[Gradient Descent]]).
- **Cost.** $P$ weights give $P^2$ numbers. The memory for the full matrix grows as the square of $P$ (Wikipedia, read), so for large networks it is out of the question. Tricks:
  - **Hessian-vector product.** Computing $H\mathbf v$ for one vector, with [[Backpropagation]] run twice, never builds $H$. In the code below it cost about 2 to 5 times one gradient (three noisy timings: 2.1, 2.6, 4.5). The "a small multiple of a gradient" claim itself is from memory (Pearlmutter, 1994).
  - **Power iteration** uses only $H\mathbf v$ to find the largest eigenvalue. The code checks it against the full matrix.
  - **Gauss-Newton / Fisher.** A cheaper matrix that is always $\ge 0$ and close to the Hessian near a good fit. See [[Fisher Information]]: for logistic regression the loss curvature and the Fisher number were identical in that note's code.
  - **Quasi-Newton and truncated Newton** methods approximate the Hessian instead of building it (Wikipedia names BFGS as an example; not run here).
- **What optimizers do instead.** [[Momentum]] and [[Adam Optimizer]] use only first derivatives. Adam rescales each weight by a running estimate of its gradient size, which is a cheap, rough stand-in for per-weight curvature (a common description, **from memory**; not tested here).
- **Two code blocks below.** The first is NumPy only: a Hessian by finite differences, the eigenvalue test, gradient descent against Newton. The second trains a small [[Backpropagation]] network on the 1,797 8×8 digit images of scikit-learn (all of them, full batch) and looks at its real Hessian (610 weights, so the whole matrix fits).

**Block 1: toy landscapes.**

```python
import numpy as np

def num_hessian(f, p, h=1e-4):
    # central finite differences for every pair of directions
    n = len(p); H = np.zeros((n, n)); I = np.eye(n) * h
    for i in range(n):
        for j in range(n):
            H[i, j] = (f(p + I[i] + I[j]) - f(p + I[i] - I[j]) - f(p - I[i] + I[j]) + f(p - I[i] - I[j])) / (4 * h * h)
    return H

# 1) Hessian of f(x, y) = x^3 + 2xy + y^2: by hand it is [[6x, 2], [2, 2]]
f = lambda p: p[0] ** 3 + 2 * p[0] * p[1] + p[1] ** 2
p = np.array([1.0, 1.0])
H = num_hessian(f, p)
print("numeric Hessian at (1,1):\n", H.round(4))
print("by hand:\n", np.array([[6 * p[0], 2], [2, 2]]))
print("symmetric gap:", abs(H - H.T).max().round(6))

# 2) the eigenvalues say what kind of flat spot it is
for name, g in {"x^2+y^2": lambda p: p[0]**2 + p[1]**2, "-x^2-y^2": lambda p: -p[0]**2 - p[1]**2,
                "x^2-y^2": lambda p: p[0]**2 - p[1]**2, "x^4+y^2": lambda p: p[0]**4 + p[1]**2}.items():
    print(f"{name:9s} eigenvalues at (0,0):", np.linalg.eigvalsh(num_hessian(g, np.zeros(2))).round(3))

# 3) a long narrow bowl f = 0.5*(x^2 + 100*y^2): H = diag(1, 100), so safe step is below 2/100 = 0.02
Hq = np.diag([1.0, 100.0])
def gd(eta, steps=100000, tol=1e-3):
    w = np.array([1.0, 1.0])
    for t in range(steps):
        if np.linalg.norm(w) < tol: return t, np.linalg.norm(w)
        w = w - eta * Hq @ w
        if np.linalg.norm(w) > 1e6: return None, np.linalg.norm(w)
    return "not yet", np.linalg.norm(w)
for eta in [0.019, 0.02, 0.021]:
    print(f"GD step {eta}: steps to reach |w|<1e-3 (or blow-up) ->", gd(eta))
w = np.array([1.0, 1.0]); w = w - np.linalg.solve(Hq, Hq @ w)
print("Newton, one step from (1,1):", w)

# 4) Rosenbrock (a curved valley), start (-1.2, 1), minimum at (1, 1)
ros = lambda p: (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2
grad = lambda p: np.array([-2 * (1 - p[0]) - 400 * p[0] * (p[1] - p[0] ** 2), 200 * (p[1] - p[0] ** 2)])
hess = lambda p: np.array([[2 - 400 * (p[1] - p[0] ** 2) + 800 * p[0] ** 2, -400 * p[0]], [-400 * p[0], 200.0]])
print("analytic vs numeric Rosenbrock Hessian gap:", abs(hess(np.array([-1.2, 1.0])) - num_hessian(ros, np.array([-1.2, 1.0]), 1e-3)).max().round(3))
w = np.array([-1.2, 1.0])
for t in range(1, 50):
    w = w - np.linalg.solve(hess(w), grad(w))
    if np.linalg.norm(w - 1) < 1e-8: break
print("Newton: steps to be within 1e-8 of (1,1):", t, w)
w = np.array([-1.2, 1.0]); eta = 0.001
for t in range(1, 20001):
    w = w - eta * grad(w)
    if np.linalg.norm(w - 1) < 1e-8: break
print("GD eta=0.001: steps used", t, "distance left", np.linalg.norm(w - 1).round(4), w.round(4))

# 5) Newton is happy to land ON a saddle; GD slides off it
Hs = np.diag([2.0, -2.0]); gs = lambda w: Hs @ w
w = np.array([1.0, 0.5]); w = w - np.linalg.solve(Hs, gs(w))
print("Newton on x^2-y^2 from (1,0.5):", w)
w = np.array([1.0, 0.001])
for _ in range(10): w = w - 0.1 * gs(w)
print("GD on x^2-y^2 from (1,0.001) after 10 steps:", w.round(5))
```

What it printed:

```text
numeric Hessian at (1,1):
 [[6. 2.]
 [2. 2.]]
by hand:
 [[6. 2.]
 [2. 2.]]
symmetric gap: 0.0
x^2+y^2   eigenvalues at (0,0): [2. 2.]
-x^2-y^2  eigenvalues at (0,0): [-2. -2.]
x^2-y^2   eigenvalues at (0,0): [-2.  2.]
x^4+y^2   eigenvalues at (0,0): [0. 2.]
GD step 0.019: steps to reach |w|<1e-3 (or blow-up) -> (361, 0.0009829052712442116)
GD step 0.02: steps to reach |w|<1e-3 (or blow-up) -> ('not yet', 1.0)
GD step 0.021: steps to reach |w|<1e-3 (or blow-up) -> (None, 1004475.4989265597)
Newton, one step from (1,1): [0. 0.]
analytic vs numeric Rosenbrock Hessian gap: 0.001
Newton: steps to be within 1e-8 of (1,1): 6 [1. 1.]
GD eta=0.001: steps used 20000 distance left 0.0003 [0.9999 0.9997]
Newton on x^2-y^2 from (1,0.5): [0. 0.]
GD on x^2-y^2 from (1,0.001) after 10 steps: [0.10737 0.00619]
```

How to read it:
- **Hessian by finite differences** matches the by-hand $\begin{pmatrix}6&2\\2&2\end{pmatrix}$ at $(1,1)$, and it is symmetric.
- **The eigenvalue test** gave $(2,2)$ for the bowl, $(-2,-2)$ for the hilltop, $(-2,2)$ for the saddle, and $(0,2)$ for $x^4+y^2$. That last one is really a minimum, but the test cannot see it, because the zero eigenvalue is "inconclusive".
- **A long narrow bowl**, $H=\mathrm{diag}(1,100)$, so the safe limit is $2/100=0.02$. Step 0.019: converged in 361 steps. Step 0.02: bounces forever (after 100,000 steps it was still at distance 1.0). Step 0.021: blew up. Newton: **one step** to $(0,0)$.
- **Rosenbrock** (a curved banana-shaped valley, minimum at $(1,1)$): Newton got within $10^{-8}$ in **6 steps**. Gradient descent with step 0.001 had used **20,000 steps** and was still 0.0003 away. The hand-written Hessian differs from the finite-difference one by 0.001, which is the size of the finite-difference error at step $10^{-3}$ for entries in the hundreds (not a bug in the formula, as the other checks show; not separately proved).
- **The saddle:** Newton from $(1,0.5)$ jumped **straight onto the saddle** $(0,0)$ and would stay there, because it only looks for a flat spot. Gradient descent started a hair off the ridge, at $(1,0.001)$, and by 10 steps had already begun to slide away along $y$ (the $y$ value grew from 0.001 to 0.006 while $x$ shrank).

**Block 2: a real small network.**

```python
import numpy as np, torch
from sklearn.datasets import load_digits
from torch.autograd.functional import hessian

torch.set_default_dtype(torch.float64); torch.set_num_threads(4)
d = load_digits()
X = torch.tensor(d.data / 16.0); y = torch.tensor(d.target, dtype=torch.long)   # 1,797 images of 8x8, whole set (full batch)
H1, P = 8, 64 * 8 + 8 + 8 * 10 + 10                                             # 64 -> 8 -> 10 tanh net, 610 numbers

def logits(w, X):
    W1, b1, W2, b2 = w[:512].view(64, 8), w[512:520], w[520:600].view(8, 10), w[600:]
    return torch.tanh(X @ W1 + b1) @ W2 + b2
def loss(w): return torch.nn.functional.cross_entropy(logits(w, X), y)
def init(seed):
    g = torch.Generator().manual_seed(seed); w = torch.zeros(P)
    w[:512] = torch.randn(512, generator=g) * (1 / 64) ** 0.5; w[520:600] = torch.randn(80, generator=g) * (1 / 8) ** 0.5
    return w
def grad(w):
    w = w.clone().requires_grad_(True); return torch.autograd.grad(loss(w), w)[0]
def hvp(w, v):                                       # Hessian times a vector, without ever building the Hessian
    w = w.clone().requires_grad_(True); g = torch.autograd.grad(loss(w), w, create_graph=True)[0]
    return torch.autograd.grad(g @ v, w)[0]
def top_eig(w, iters=100):                           # power iteration (largest |eigenvalue|)
    v = torch.randn(P, generator=torch.Generator().manual_seed(0)); v /= v.norm()
    for _ in range(iters): u = hvp(w, v); lam = float(v @ u); v = u / u.norm()
    return lam

w = init(0)
Hf = hessian(loss, w)
ev = torch.linalg.eigvalsh(Hf)
print("size of the full Hessian:", tuple(Hf.shape), "=", P * P, "numbers | symmetric gap", float((Hf - Hf.T).abs().max()))
def summary(tag, ev):
    print(f"{tag}: top {float(ev[-1]):.4f}, most negative {float(ev[0]):.4f}, #<-1e-3: {int((ev < -1e-3).sum())}, #|.|<1e-3: {int((ev.abs() < 1e-3).sum())}, #>1e-3: {int((ev > 1e-3).sum())}")
summary("at start", ev)
print("power iteration vs eigvalsh:", round(top_eig(w), 6), round(float(ev[-1]), 6) if abs(ev[-1]) >= abs(ev[0]) else round(float(ev[0]), 6))
v = torch.randn(P); print("HVP vs Hessian@v gap:", float((hvp(w, v) - Hf @ v).abs().max()))
import time
t0 = time.time(); [grad(w) for _ in range(50)]; tg = time.time() - t0
t0 = time.time(); [hvp(w, v) for _ in range(50)]; th = time.time() - t0
print(f"time of one Hessian-vector product / one gradient: {th / tg:.1f}x (noisy)")
g0 = grad(w)
print("plain Newton needs H^-1, but the start Hessian has", int((ev.abs() < 1e-9).sum()), "eigenvalues that are exactly 0 (smallest |eigenvalue|", f"{float(ev.abs().min()):.1e})")
for damp in [0.0, 1.1]:                              # damped Newton: (H + damp*I)^-1 g. 1.1 is just above the most negative eigenvalue
    try:
        step = -torch.linalg.solve(Hf + damp * torch.eye(P), g0)
        print(f"damp {damp}: slope along the step {float(g0 @ step):.3f} (positive = uphill), loss {float(loss(w)):.3f} -> {float(loss(w + step)):.3f}")
    except Exception as e: print(f"damp {damp}: solve failed ({type(e).__name__})")
print(f"one gradient step of size 1.0: loss -> {float(loss(w - g0)):.3f}")


def ggn(w):                                           # Gauss-Newton matrix: J^T (diag(p) - p p^T) J / n, always >= 0
    J = torch.autograd.functional.jacobian(lambda w: logits(w, X), w)      # (n, 10, P)
    p = torch.softmax(logits(w, X), 1)
    S = torch.diag_embed(p) - p[:, :, None] * p[:, None, :]
    return torch.einsum("nkp,nkl,nlq->pq", J, S, J) / len(X)

for eta in [1.0, 2.0, 4.0]:                           # full-batch gradient descent, 2000 steps, track the top eigenvalue
    w = init(0); lam = {}
    for t in range(2001):
        if t % 20 == 0: lam[t] = top_eig(w, 30)
        w = w - eta * grad(w)
    mid = [lam[t] for t in range(200, 1001, 20)]
    print(f"eta={eta}: 2/eta={2/eta:.2f} | top eigenvalue at steps 0,20,60,200,1000,2000:", [round(lam[t], 2) for t in (0, 20, 60, 200, 1000, 2000)],
          f"| steps 200-1000 range {min(mid):.2f}..{max(mid):.2f} | final loss {float(loss(w)):.4f}")
    if eta == 1.0:
        evT = torch.linalg.eigvalsh(hessian(loss, w)); summary("after training (eta=1.0)", evT)
        G = ggn(w); evG = torch.linalg.eigvalsh(G)
        print("Gauss-Newton: smallest eigenvalue", f"{float(evG[0]):.2e}", "| top", round(float(evG[-1]), 4),
              "| relative gap to Hessian", round(float((hessian(loss, w) - G).norm() / hessian(loss, w).norm()), 4))
        wi = init(0); Gi = ggn(wi)
        print("at start the relative gap Hessian vs Gauss-Newton:", round(float((Hf - Gi).norm() / Hf.norm()), 4), "| Gauss-Newton smallest eigenvalue", f"{float(torch.linalg.eigvalsh(Gi)[0]):.2e}")
```

What it printed:

```text
size of the full Hessian: (610, 610) = 372100 numbers | symmetric gap 4.163336342344337e-17
at start: top 3.0300, most negative -1.0353, #<-1e-3: 146, #|.|<1e-3: 260, #>1e-3: 204
power iteration vs eigvalsh: 3.030004 3.030004
HVP vs Hessian@v gap: 4.440892098500626e-16
time of one Hessian-vector product / one gradient: 4.5x (noisy)
plain Newton needs H^-1, but the start Hessian has 33 eigenvalues that are exactly 0 (smallest |eigenvalue| 0.0e+00)
damp 0.0: solve failed (_LinAlgError)
damp 1.1: slope along the step -0.613 (positive = uphill), loss 2.504 -> 2.327
one gradient step of size 1.0: loss -> 2.424
eta=1.0: 2/eta=2.00 | top eigenvalue at steps 0,20,60,200,1000,2000: [3.03, 2.45, 2.16, 1.93, 1.29, 0.91] | steps 200-1000 range 1.29..1.95 | final loss 0.0127
after training (eta=1.0): top 0.9126, most negative -0.0009, #<-1e-3: 0, #|.|<1e-3: 447, #>1e-3: 163
Gauss-Newton: smallest eigenvalue -8.58e-19 | top 0.863 | relative gap to Hessian 0.0781
at start the relative gap Hessian vs Gauss-Newton: 0.7606 | Gauss-Newton smallest eigenvalue -1.03e-17
eta=2.0: 2/eta=1.00 | top eigenvalue at steps 0,20,60,200,1000,2000: [3.03, 2.31, 1.21, 1.12, 1.01, 0.67] | steps 200-1000 range 0.74..2.43 | final loss 0.0080
eta=4.0: 2/eta=0.50 | top eigenvalue at steps 0,20,60,200,1000,2000: [3.03, 1.68, 3.14, 2.87, 2.13, 1.87] | steps 200-1000 range 1.46..2.87 | final loss 2.7652
```

How to read it:
- **Size.** 610 weights give a $610\times610$ Hessian: 372,100 numbers, symmetric to $4\times10^{-17}$. A Hessian-vector product matched the full matrix times the same vector to below $10^{-15}$, and power iteration (only vector products, 100 rounds) found the top eigenvalue 3.030004, the same as the full eigenvalue solver.
- **At the start** (random weights, loss 2.504): the top eigenvalue is 3.03 and the most negative is $-1.035$. Of the 610 eigenvalues, **146 are below $-0.001$** (the loss curves *down* in those directions), 260 are within $\pm0.001$ (flat) and 204 are above. So a freshly initialised net sits on a saddle-like slope, not in a bowl. Of the flat ones, 33 are exactly zero; 24 of those come from the 3 pixels that are 0 in every image (3 pixels × 8 hidden units), and the other 9 were not explained.
- **Newton on the net, at the start.** Plain Newton **fails**: the matrix is singular (33 exact zeros), so $H^{-1}$ does not exist. Adding $1.1\,I$ to the Hessian (just above the most negative eigenvalue magnitude, 1.035) makes it work. That "damped" step went downhill (slope $-0.613$) and cut the loss from 2.504 to 2.327, while one plain gradient step of size 1.0 reached 2.424. That is one step from one start, not a race.
- **After training** with step 1.0 for 2,000 steps (loss 0.0127): the top eigenvalue is 0.913 and the most negative is $-0.0009$, so **no** direction is below $-0.001$. 447 of 610 eigenvalues are within $\pm0.001$ (flat) and 163 are above. Most directions are flat, and a few matter.
- **Gauss-Newton matrix** (the cheap cousin; see [[Fisher Information]]): smallest eigenvalue $-9\times10^{-19}$ (zero within rounding, so it never goes negative), top 0.863. Its relative gap to the Hessian is **0.76 at the start but 0.078 after training**. Near a good fit the cheap matrix is a decent copy; far from one it is not.
- **Step size against the top eigenvalue**, with $2/\eta$ as the classical limit:

| $\eta$ | $2/\eta$ | top eigenvalue at steps 0 / 20 / 60 / 200 / 1000 / 2000 | final loss |
|---|---|---|---|
| 1.0 | 2.00 | 3.03 / 2.45 / 2.16 / 1.93 / 1.29 / 0.91 | 0.0127 |
| 2.0 | 1.00 | 3.03 / 2.31 / 1.21 / 1.12 / 1.01 / 0.67 | 0.0080 |
| 4.0 | 0.50 | 3.03 / 1.68 / 3.14 / 2.87 / 2.13 / 1.87 | 2.7652 (no training) |

  With $\eta=1.0$ and $2.0$ the top eigenvalue started above $2/\eta$ and was pushed **down to about $2/\eta$**, then drifted lower as the loss fell. With $\eta=1.0$ it stayed between 1.29 and 1.95 for steps 200 to 1000, never above 2. With $\eta=2.0$ it ranged from 0.74 to 2.43 over the same steps, so it also went above $2/\eta=1$ at some sampled points. With $\eta=4.0$ the top eigenvalue stayed far above $2/\eta=0.5$ and the net **never trained** (loss 2.504 to 2.765). This matches the rule for a bowl, but it is a single seed and a tiny net. The published "edge of stability" result (Cohen et al. 2021, abstract read) says the top eigenvalue climbs to just above $2/\eta$ and hovers there. Here it came *down* to that level from above and then fell below, so this run does not reproduce the climbing part. It was sampled every 20 steps with 30 power-iteration rounds, which can miss short spikes.

## 🔴 4. Deep / formal
The Hessian of a loss $f$ with $P$ weights has one entry for each pair of weights:

$$
H_{ij}=\frac{\partial^{2} f}{\partial w_i\,\partial w_j}
$$

**Where:**
- $f$ is the loss. $w_i$ and $w_j$ are two of the weights. $H_{ij}$ is the entry in row $i$, column $j$.
- $\frac{\partial^2 f}{\partial w_i\partial w_j}$ means: take the slope of $f$ along $w_j$, then ask how fast that slope changes when $w_i$ moves.
- **Matters:** the diagonal says how curved each weight is on its own. The off-diagonal entries say how weights interact. A method that keeps only the diagonal ignores interactions, and long tilted valleys are made of exactly those.

**Example (tiny numbers):** $f(x,y)=x^3+2xy+y^2$ at the point $(1,1)$.
- Slope along $x$: $3x^2+2y$. Slope along $y$: $2x+2y$.
- Second derivatives: $\partial_x(3x^2+2y)=6x=6$. $\partial_y(3x^2+2y)=2$. $\partial_x(2x+2y)=2$. $\partial_y(2x+2y)=2$.
- So $H=\begin{pmatrix}6&2\\2&2\end{pmatrix}$, and the two mirrored entries are both 2.
- Its eigenvalues: trace 8, determinant $12-4=8$, so $\lambda=4\pm\sqrt8=\mathbf{6.83}$ and $\mathbf{1.17}$. Both positive: the surface curves **up** in every direction at this point. (The gradient there is $(5,4)$, not 0, so it is not a minimum, only a locally bowl-shaped slope.)

The Hessian turns the slope into a curved local picture (second-order Taylor expansion):

$$
f(\mathbf w+\mathbf d)\approx f(\mathbf w)+\mathbf g^{\top}\mathbf d+\tfrac12\,\mathbf d^{\top}H\,\mathbf d
$$

**Where:**
- $\mathbf w$ is where you stand. $\mathbf d$ is a small step. $\mathbf g$ is the gradient (the slope) at $\mathbf w$. $H$ is the Hessian there.
- $\mathbf g^\top\mathbf d$ is the straight-line guess. $\tfrac12\mathbf d^\top H\mathbf d$ is the bend correction: how much the surface curves away from that line.
- **Matters:** without the bend term you only have the straight-line guess that plain gradient descent trusts. It cannot tell a bowl from a saddle or say how far a step may safely go.

**Example (tiny numbers):** the same $f$ at $(1,1)$ with the step $\mathbf d=(0.1,-0.1)$.
- $f(1,1)=1+2+1=4$ and $\mathbf g=(5,4)$, so $\mathbf g^\top\mathbf d=0.5-0.4=0.1$.
- $\mathbf d^\top H\mathbf d=6(0.01)+2\cdot2\cdot(0.1)(-0.1)+2(0.01)=0.06-0.04+0.02=0.04$, and half of it is $0.02$.
- Estimate with bend: $4+0.1+0.02=\mathbf{4.12}$. Straight-line guess only: $4.10$. Exact $f(1.1,0.9)=1.331+1.98+0.81=\mathbf{4.121}$.
- Meaning: the bend term cut the error from 0.021 to 0.001.

Gradient descent on a perfect bowl $f=\tfrac12\mathbf w^\top H\mathbf w$ becomes simple once you look along the eigenvectors of $H$:

$$
w_i\;\leftarrow\;(1-\eta\,\lambda_i)\,w_i
$$

**Where:**
- $w_i$ is the part of the weights that lies along eigenvector $i$ of $H$. $\lambda_i$ is the matching eigenvalue (the curvature in that direction). $\eta$ is the [[Learning Rate]].
- $(1-\eta\lambda_i)$ is the factor by which that part shrinks (or grows) each step.
- **Matters:** every part must have $|1-\eta\lambda_i|<1$, which holds for all of them exactly when $\eta<2/\lambda_{\max}$. Beyond that the steepest direction blows up. And the slowest part, with the smallest $\lambda_i$, shrinks by only $1-\eta\lambda_{\min}$ per step, so the gap $\kappa=\lambda_{\max}/\lambda_{\min}$ sets the speed.

**Example (tiny numbers):** eigenvalues $\lambda=(1,100)$ from block 1, so $2/\lambda_{\max}=0.02$.
- $\eta=0.019$: factors $1-0.019=0.981$ and $1-1.9=-0.9$. Both have size below 1, so it converges. The slow part must fall from 1 to 0.001: $\ln(0.001)/\ln(0.981)\approx360.1$, so **361 steps**. (The code printed 361.)
- $\eta=0.021$: factors $0.979$ and $1-2.1=-1.1$. Size 1.1 is above 1, so the steep part grows by 10% per step and flips sign. It **blew up** (code).
- Meaning: the steep direction forces the step small, and the flat direction then needs hundreds of steps. Here $\kappa=100$.

Newton's method divides each direction's slope by that direction's curvature:

$$
\mathbf d_{\text{Newton}}=-H^{-1}\mathbf g
$$

**Where:**
- $\mathbf d_{\text{Newton}}$ is the step to take. $H^{-1}$ is the inverse of the Hessian (undoing it). $\mathbf g$ is the gradient. The minus sign means go downhill.
- **Matters:** on a perfect bowl this lands on the minimum in one step. If $H$ has a **negative** eigenvalue the step in that direction points **uphill** (toward a saddle or a maximum). If it has a **zero** eigenvalue, $H^{-1}$ does not exist. Both happened to the real net above at the start, so people use damping ($H+\mu I$) or the always-$\ge0$ Gauss-Newton matrix.

**Example (tiny numbers):** bowl $H=\mathrm{diag}(1,100)$, standing at $\mathbf w=(1,1)$.
- Gradient $\mathbf g=H\mathbf w=(1,100)$.
- $H^{-1}\mathbf g=(1/1,\;100/100)=(1,1)$, so $\mathbf d=(-1,-1)$.
- New point $(1,1)+(-1,-1)=\mathbf{(0,0)}$, the minimum, in **one step**, against 361 steps for gradient descent above.
- Meaning: dividing by the curvature turns a narrow valley into a round bowl, where every direction is solved at once.

**Further notes (from the code above):**
- **Newton can land on a saddle.** On $x^2-y^2$ it jumped to $(0,0)$ from $(1,0.5)$. Gradient descent slides off a saddle, but slowly when the escape direction is nearly flat.
- **Real nets are not bowls.** At random start: 146 of 610 eigenvalues below $-0.001$, 260 near zero. After training: 0 below $-0.001$, 447 near zero. Mostly flat, with a few steep directions. This is one tiny net, one seed.
- **Memory.** 610 weights gave 372,100 numbers. The 830,464-weight model of [[Tiny GPT from Scratch]] would give $830{,}464^2=689{,}670{,}455{,}296$ numbers (about 2.8 terabytes at 4 bytes each; arithmetic, not run). A Hessian-vector product avoids that.
- **Not checked here:** large networks, minibatch noise ([[Stochastic Gradient Descent]]), Newton-type optimizers in training, the climbing "progressive sharpening", other seeds, and the idea that training happens in a few steep directions (only hinted by 163 positive of 610).

**What came before, and how this improved on it:**
- **Before:** gradient descent uses only the slope ([[Gradient Descent]]). The Hessian itself is a 19th-century idea, named after the German mathematician Ludwig Otto Hesse (Wikipedia, read; the exact year was not on the page). Newton's method for finding flat spots is much older (from memory: 17th century).
- **Its limit:** the slope alone gives no curvature, so a steep direction caps the step and a flat direction crawls (361 steps in the narrow-bowl example). The full Hessian is $P^2$ numbers, and for a real net it can be singular or have negative curvature (33 exact zeros and 146 negative directions in the small net).
- **What changed:** use curvature without building the matrix: Hessian-vector products and power iteration (top eigenvalue to 6 decimals), always-$\ge0$ Gauss-Newton or Fisher matrices ([[Fisher Information]]), damping, and quasi-Newton approximations such as BFGS (named in Wikipedia, read; not run). Cohen et al. (arXiv 2103.00065, 2021, abstract read) then used the top Hessian eigenvalue to study how full-batch training behaves against $2/\eta$.
- **How much better:** in this note's code, Newton needed 6 steps on Rosenbrock where gradient descent was still 0.0003 away after 20,000, and 1 step against 361 on the narrow bowl. On the small net, one damped Newton step cut the loss to 2.327 against 2.424 for one gradient step (one start, one step only). The cost: a Hessian-vector product ran at about 2 to 5 times a gradient (three noisy timings: 2.1, 2.6 and 4.5).

## Related
- [[Calculus]] · [[Matrix]] · [[Eigenvalue]] · [[Vector]] · [[Gradient Descent]] · [[Learning Rate]] · [[Convex Optimization]] · [[Fisher Information]] · [[Backpropagation]] · [[Loss Function]] · [[Momentum]] · [[Adam Optimizer]] · [[Stochastic Gradient Descent]] · [[Weight Initialization]] · [[Batch Normalization]] · [[Vanishing and Exploding Gradients]] · [[Tiny GPT from Scratch]] · [[MNIST from Scratch]] · [[Jacobian]] · [[Chain Rule]]

## Sources
- Wikipedia, "Hessian matrix" (read): definition, symmetry, second-derivative test, link to Newton's method, quadratic memory cost, quasi-Newton and truncated-Newton methods, naming after Hesse.
- Cohen, Kaur, Li, Kolter, Talwalkar, "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability", arXiv 2103.00065 (2021), abstract read only.
- This vault's [[Learning Rate]], [[Fisher Information]] and [[Convex Optimization]] notes for the one-weight rule, the curvature-equals-Fisher check and the Hessian test.
- Pearlmutter (1994) on fast Hessian-vector products, the date of Newton's method, and the description of Adam as a rough curvature stand-in: **from memory**, not checked.
- All numbers in levels 3 and 4 come from the two code blocks above, run on Python 3.9.6, NumPy 1.24.4, PyTorch 2.8.0 (CPU, 4 threads), scikit-learn 1.1.2. Image: "Saddle point.svg" by Nicoguaro, CC BY 3.0, via Wikimedia Commons.

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/linear-algebra Hessian:::The matrix of all second derivatives of the loss, with one entry for each pair of weights. It describes the curvature.

#flashcards/math/formula #flashcards/topic/linear-algebra Write one entry $H_{ij}$ of the Hessian.
?
$$
H_{ij}=\frac{\partial^{2} f}{\partial w_i\,\partial w_j}
$$
Slope of $f$ along $w_j$, then how fast that slope changes when $w_i$ moves.
- $f$: the loss. $w_i$, $w_j$: two of the weights.
- $H_{ij}$: the entry in row $i$, column $j$. The diagonal is the curvature of each weight alone. The off-diagonal entries are how two weights interact.
- $\partial$: a partial derivative, the slope when only one weight moves.

#flashcards/math/formula #flashcards/topic/linear-algebra Write the second-order Taylor expansion of $f(\mathbf w+\mathbf d)$.
?
$$
f(\mathbf w+\mathbf d)\approx f(\mathbf w)+\mathbf g^{\top}\mathbf d+\tfrac12\,\mathbf d^{\top}H\,\mathbf d
$$
- $\mathbf w$: where you stand. $\mathbf d$: a small step away from it.
- $\mathbf g$: the gradient (the slope) at $\mathbf w$. $\mathbf g^\top\mathbf d$ is the straight-line guess.
- $H$: the Hessian at $\mathbf w$. $\tfrac12\mathbf d^\top H\mathbf d$ is the bend correction, how much the surface curves away from the straight line.
- Without the bend term you cannot tell a bowl from a saddle.

#flashcards/math/formula #flashcards/topic/linear-algebra Write the Newton step.
?
$$
\mathbf d_{\text{Newton}}=-H^{-1}\mathbf g
$$
On a perfect bowl it lands on the minimum in one step.
- $\mathbf d_{\text{Newton}}$: the step to take.
- $H^{-1}$: the inverse of the Hessian. It divides each direction's slope by that direction's curvature.
- $\mathbf g$: the gradient. The minus sign means go downhill.
- A negative eigenvalue makes the step go uphill. A zero eigenvalue means $H^{-1}$ does not exist.

#flashcards/math/formula #flashcards/topic/linear-algebra Write how gradient descent changes the part of the weights along eigenvector $i$ on a perfect bowl.
?
$$
w_i\;\leftarrow\;(1-\eta\,\lambda_i)\,w_i
$$
It is stable for every direction exactly when $\eta<2/\lambda_{\max}$.
- $w_i$: the part of the weights along eigenvector $i$ of $H$.
- $\lambda_i$: the matching eigenvalue, the curvature in that direction.
- $\eta$: the learning rate (the step size).
- $(1-\eta\lambda_i)$: the factor each step multiplies that part by. If its size is above 1, that part blows up.

#flashcards/math/cloze #flashcards/topic/linear-algebra Gradient descent on a bowl is stable only when the learning rate is below ==2/λmax==, where λmax is the largest eigenvalue of the Hessian.

#flashcards/math/formula #flashcards/topic/linear-algebra What is the Hessian of $f=x^3+2xy+y^2$ at $(1,1)$?
?
$$
H=\begin{pmatrix}6&2\2&2\end{pmatrix}
$$
Eigenvalues 6.83 and 1.17, both positive.
- The diagonal 6 and 2 are the curvatures along $x$ and along $y$.
- The two mirrored entries are both 2: how $x$ and $y$ interact. The Hessian is symmetric.
- Positive eigenvalues mean the surface curves up in every direction there.

#flashcards/math/example #flashcards/topic/linear-algebra On the narrow bowl $H=\mathrm{diag}(1,100)$, how many steps do gradient descent (η = 0.019) and Newton need?::Gradient descent needs 361 steps. Newton needs 1 step.

#flashcards/math/concept #flashcards/topic/linear-algebra Why can plain Newton fail on a real neural net?::The Hessian can have zero eigenvalues (no inverse) and negative ones (the step goes uphill). In the note's 610-weight net, 33 were exactly zero and 146 were negative at the start.

#flashcards/math/cloze #flashcards/topic/linear-algebra In the small 610-weight net, the number of eigenvalues below −0.001 was ==146== at the start and ==0== after training (one tiny net, one seed).

#flashcards/math/concept #flashcards/topic/linear-algebra Why use a Hessian-vector product?::It gives $H\mathbf v$ without building the $P\times P$ matrix. It cost about 2 to 5 gradients in the note's noisy timings.

#flashcards/math/image #flashcards/topic/linear-algebra ![[saddle-point.svg]]
What kind of surface is this, and what do the Hessian's eigenvalues look like at the red dot?
?
A saddle: it curves up in one direction and down in another. The Hessian there has one positive and one negative eigenvalue.
