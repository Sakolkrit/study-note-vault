---
tags: [machine-learning, regularization, L1, L2, weight-decay, overfitting]
status: growing
level: 4
tier: 2
---

# Regularization

This note covers **L1 and L2 regularization** (and the closely related **weight decay**) and the other tricks that keep a model from memorising its training data.

## 🟢 1. Simple (age 12)
Regularization is **a rule that stops a model from getting too fancy**. Without it, a model can learn to copy every tiny detail of its practice examples, including the accidental ones, and then fail on new questions. Regularization adds a small **penalty for being complicated**, so the model prefers the **simple explanation** unless the fancy one is clearly much better.

**Example:** A student prepares for a test by memorising the exact answers to last year's paper, including the typos. A teacher says: "You only get full marks for answers you can explain with a short, simple rule." Now the student looks for the simple rule that covers most questions, which also works on this year's paper. The "short, simple rule" requirement is the regularization.

## 🔵 2. How it works
1. A model has many adjustable numbers (weights). Big, extreme weights usually mean it is bending itself into a wiggly shape to catch every training point. This is [[Overfitting]].
2. Regularization adds a **penalty** to the loss ([[Loss Function]]) that grows when the weights get big. The model must now trade off two things: fit the data, and keep the weights small.
3. How strong the penalty is is set by one number, written $\lambda$ ("lambda"), which you choose before training. A bigger $\lambda$ means a simpler model.
4. There are two classic penalties:
   - **L2** adds up the **squares** of the weights. It makes all weights **smaller**, but rarely exactly zero.
   - **L1** adds up the **absolute values** of the weights. It pushes unimportant weights **exactly to zero**, so the model ends up using only a few inputs (this is called **sparsity**).
5. During training ([[Gradient Descent]]), the penalty acts like a gentle pull of every weight towards zero.
6. Other forms of regularization do not touch the loss: **early stopping** (stop training when the error on held-out data starts rising), **dropout** (randomly switch off parts of the network during training, see [[Dropout]]), and simply **getting more data**.

![[regularization-curves.svg]]
*Seven data points (red dots). The wiggly blue curve and the smooth green curve both pass through every point, so both have zero training error. Regularization steers the model towards the smooth green one, which is much more likely to predict new points well. Image: Nicoguaro, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Regularization.svg), CC BY 4.0.*

![[l1-l2-geometry.png]]
*Why L1 gives zeros and L2 does not. In each panel the axes are two weights, $w^1$ and $w^2$. The shape is the set of weights allowed by the penalty: a **diamond** for L1 ($\ell_1$) and a **circle** for L2 ($\ell_2$). The red dashed line stands for the best-fitting weights of the data. As I read the picture, the model takes the allowed point that touches that line. With the diamond, that is a **corner on an axis**, where one weight is exactly zero. With the circle, it is an ordinary point where neither weight is zero. Image: Mxwsn, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Sparsityl1.png), CC BY-SA 4.0. The file's own description is just "Sparsity", so the explanation here is mine.*

## 🟠 3. Technical detail
- **The recipe:** new loss = data loss + $\lambda\times$ penalty on the weights.
  - **L2** (also called **ridge** in [[Regression]]): penalty is the sum of squared weights.
  - **L1** (also called **lasso**): penalty is the sum of absolute weights.
  - **Elastic net:** a mix of both.
- **Weight decay:** with plain [[Stochastic Gradient Descent]], adding an L2 penalty is the same as shrinking every weight by a small fraction at each step. That is why the two names are often used for the same thing. With adaptive optimisers such as [[Adam Optimizer]] the two are *not* identical, which led to variants with "decoupled" weight decay (from memory; check the details).
- **Bias weights are often left out** of the penalty. The penalty is usually applied to the weights, not to the constant offset.
- **Choosing $\lambda$:** try several values and keep the one with the best error on validation data (see [[Supervised Learning]]). Too small and it has no effect; too big and the model **underfits** (it becomes too simple, see [[Bias-Variance]]).
- **What each one is good for:**
  - **L2:** the default. Smooth, works well in neural networks, handles inputs that are correlated with each other.
  - **L1:** useful when you suspect only a few inputs matter and want the model to **select** them. The result is easy to read.
- **Effect on bias-variance:** regularization adds a little bias to cut a lot of variance, and often lowers the total error ([[Bias-Variance]]).
- **Early stopping:** watch the validation loss. When it stops improving for a while, stop and keep the best weights. It is cheap, and acts like a mild L2 penalty in simple cases (a known result for simple models; stated from memory).
- **Scale first:** the penalty treats all weights alike, so inputs measured on very different scales get penalised unfairly. Rescale them (see [[Normalization]]).
- **Other ways to reduce overfitting:** more data, data augmentation (for example flipping or cropping images), smaller models, [[Dropout]].

```python
import numpy as np
w = np.array([3., -1.]); lam = 0.1; L = 2.0           # weights, strength lambda, a made-up data loss
print(L + lam * (w ** 2).sum(), L + lam * abs(w).sum())     # 3.0 2.4     L2-penalised and L1-penalised loss

gL = np.array([0.5, -0.2]); eta = 0.1                  # gradient of the data loss, learning rate
print(w - eta * (gL + 2 * lam * w))                    # [ 2.89 -0.96]   one step with the L2 penalty (weight decay)

for name, step in (("L1", lambda v: np.sign(v) * max(abs(v) - 0.1, 0)),     # L1 pull: subtract eta*lambda = 0.1
                   ("L2", lambda v: 0.8 * v)):                              # L2 pull: multiply by 1 - 2*eta*lambda = 0.8
    v = 0.25; out = []
    for _ in range(4):
        v = step(v); out.append(round(float(v), 3))
    print(name, out)
# L1 [0.15, 0.05, 0.0, 0.0]       reaches exactly zero and stays
# L2 [0.2, 0.16, 0.128, 0.102]    shrinks but never quite reaches zero
```

The last loop starts a weight at 0.25 and applies only the penalty's pull (as if the data loss had no opinion), with $\lambda=1$ and learning rate $0.1$, so you can see the difference between the two penalties.

## 🔴 4. Deep / formal
**L2 regularization and weight decay.** Let $L(w)$ be the usual loss on the data and $w$ the weights. The L2-regularized loss, and one gradient-descent step on it, are:

$$
J(w)=L(w)+\lambda\sum_j w_j^{2},\qquad w\leftarrow w-\eta\big(\nabla L(w)+2\lambda w\big)
$$

**Where:**
- $L(w)$ is the data loss (how wrong the model is on the training data). $w_j$ is the $j$-th weight, and $\sum_jw_j^2$ is the sum of the squares of all the weights.
- $\lambda\ge0$ is the **regularization strength**. **Matters:** at $\lambda=0$ there is no regularization. As $\lambda$ grows, small weights matter more than good fit.
- $J(w)$ is the new total loss that training actually minimises.
- $\nabla L(w)$ is the gradient of the data loss, and $\eta$ is the [[Learning Rate]].
- The gradient of $\lambda\sum_jw_j^2$ is $2\lambda w$, a pull towards 0 that is **proportional to the weight**. **Matters:** big weights get pulled hard, small weights barely at all. Written out, the update shrinks the weight by the factor $(1-2\eta\lambda)$ before applying the data gradient, which is why this is called **weight decay**.
- **Matters:** a pull proportional to the size means the weight gets smaller and smaller but never reaches exactly zero by this force alone.

**Example (tiny numbers):** weights $w=(3,-1)$, data loss $L=2$, $\lambda=0.1$, $\eta=0.1$, and data gradient $\nabla L=(0.5,-0.2)$.
- Penalty: $0.1\cdot(3^2+(-1)^2)=0.1\cdot10=\mathbf{1.0}$. Total loss $J=2+1.0=\mathbf{3.0}$.
- Step for $w_1=3$: $3-0.1\,(0.5+2\cdot0.1\cdot3)=3-0.1\cdot1.1=\mathbf{2.89}$.
- Step for $w_2=-1$: $-1-0.1\,(-0.2+2\cdot0.1\cdot(-1))=-1-0.1\cdot(-0.4)=\mathbf{-0.96}$.
- Meaning: the penalty added to the loss, and pulled both weights towards 0, the big one more strongly (the data gradient alone would have moved $w_1$ only to $2.95$).

**L1 regularization and sparsity.** Replace the squares with absolute values:

$$
J(w)=L(w)+\lambda\sum_j |w_j|
$$

**Where:**
- $|w_j|$ is the absolute value of weight $j$ (its size, ignoring the sign). The other symbols are as in the L2 equation above.
- The pull of this penalty on a weight is $\lambda\cdot\operatorname{sign}(w_j)$, a **constant** push towards zero (direction towards zero, size $\lambda$), no matter how big or small the weight is. **Matters:** with a constant push, a small weight is pushed all the way to **exactly zero** and held there, unless the data loss pulls back harder than $\lambda$.
- **Matters:** that gives **sparse** solutions, where many weights are exactly zero, so the model **selects** a few useful inputs. L2's pull weakens as the weight shrinks, which is why it only gets near zero.
- At $w_j=0$ the absolute value has a sharp corner, so the gradient is not defined there. Practical methods use special rules (such as soft-thresholding) for that point.
- Geometrically, the diamond shape in the picture above has corners on the axes, and these are where sparse solutions sit.

**Example (tiny numbers):** first, the penalty value. Take the same weights $w=(3,-1)$, data loss $L=2$ and $\lambda=0.1$ as in the L2 example.
- L1 penalty: $0.1\cdot(|3|+|-1|)=0.1\cdot4=\mathbf{0.4}$, so $J=2+0.4=\mathbf{2.4}$ (compare $3.0$ for L2: here L1 charges less for large weights).

Now the pull on one weight. Take $w=0.25$ and, to isolate the penalty, assume the data loss pushes neither way. Use $\lambda=1$ and $\eta=0.1$.
- L1 steps subtract $\eta\lambda=0.1$ each time: $0.25\to\mathbf{0.15}\to\mathbf{0.05}\to\mathbf{0}$ and stays at 0.
- L2 steps multiply by $1-2\eta\lambda=0.8$: $0.25\to\mathbf{0.2}\to\mathbf{0.16}\to\mathbf{0.128}\to\mathbf{0.102}$ and keeps shrinking without ever reaching 0.
- Meaning: L1 deletes unimportant weights, while L2 only shrinks them.

**Further notes:**
- **Bayesian view:** L2 corresponds to assuming, before seeing the data, that the weights are bell-shaped around 0 ([[Normal Distribution]]); L1 corresponds to a sharper, pointier prior (the Laplace distribution). The penalty is then the log of that prior. (A standard connection, stated from memory; check a textbook.)
- **Closed form for ridge:** for a line through the origin, $\hat y=wx$ ([[Regression]]), the best weight is $w=\sum x_iy_i/\sum x_i^2$ without a penalty, and $w=\sum x_iy_i/\big(\sum x_i^2+\lambda\big)$ with the L2 penalty, so the fit is shrunk towards 0. For the points $(0,1),(1,3),(2,2)$ that is $7/5=1.4$ without and $7/6\approx1.167$ with $\lambda=1$ (my arithmetic, not run in the note's code). With many inputs the same $+\lambda$ idea also steadies the weights when inputs are nearly copies of each other.
- **The geometry** in the picture (diamond vs circle) comes from rewriting the penalty as a *limit on weight size* instead of a cost. The two views are equivalent for the right choice of limit and $\lambda$ (standard result; from memory).
- **Not a cure-all:** regularization cannot fix bad data, wrong labels or a model that is the wrong shape for the problem. If error is high on training data too, regularizing more makes it worse ([[Bias-Variance]]).

**What came before, and how this improved on it:**
- **Before:** fit the training data as closely as possible.
- **Its limit:** a flexible model also fits the noise, and does badly on new data ([[Overfitting]]).
- **What changed:** add a penalty on large weights, so the model prefers smooth fits (ridge regression, Hoerl and Kennard 1970, and Tikhonov before them; from memory).
- **How much better:** in the code below, a degree-9 curve through 10 noisy points of a sine wave has a typical (median) error on new points of 0.16 without the penalty and 0.025 with a tiny penalty of 0.001, 6.4 times lower. The penalty size was picked by hand and the median was used because it is steadier than the average.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
f = lambda x: np.sin(2 * np.pi * x)
def fit(x, y, deg, lam):
    A = np.vander(x, deg + 1)   # penalty added as extra rows, solved by least squares
    return np.linalg.lstsq(np.vstack([A, np.sqrt(lam) * np.eye(deg + 1)]), np.append(y, np.zeros(deg + 1)), rcond=None)[0]
errs = {0: [], 1e-3: []}
xt = np.linspace(0, 1, 200)
for _ in range(200):
    x = np.linspace(0, 1, 10); y = f(x) + 0.2 * rng.standard_normal(10)
    for lam in errs:
        w = fit(x, y, 9, lam); errs[lam].append(((np.vander(xt, 10) @ w - f(xt)) ** 2).mean())
print("degree-9 curve through 10 noisy points, error on new points (median of 200 noise draws):")
print("  no penalty:", round(np.median(errs[0]), 3), "  with penalty 0.001:", round(np.median(errs[1e-3]), 3))
```

## Related
- [[Overfitting]] · [[Bias-Variance]] · [[Loss Function]] · [[Gradient Descent]] · [[Stochastic Gradient Descent]] · [[Learning Rate]] · [[Adam Optimizer]] · [[Regression]] · [[Normalization]] · [[Dropout]] · [[Neural Network]] · [[Normal Distribution]] · [[Supervised Learning]] · [[Bayesian Methods]] · [[Support Vector Machine]] · [[Linear Regression]] · [[Maximum Likelihood]] · [[Convex Optimization]] · [[Data Augmentation]] · [[Label Smoothing]] · [[Catastrophic Forgetting]] · [[Maximum A Posteriori]]
- [[Matrix Factorization]]
## Sources
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (free online), chapter 3 (ridge and lasso)
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online), chapter 6
- Goodfellow, Bengio & Courville, *Deep Learning*, 2016, chapter 7
- Tibshirani, *Regression Shrinkage and Selection via the Lasso*, 1996 (from memory)
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization*, 2017 (from memory)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/training Regularization:::A rule that stops a model from getting too fancy. It adds a small penalty for being complicated, so the model prefers simpler answers that work on new data.

#flashcards/ml/formula #flashcards/topic/training Write L2 regularization and the weight update, and explain each symbol.
?
$$J(w)=L(w)+\lambda\sum_j w_j^{2},\qquad w\leftarrow w-\eta\big(\nabla L(w)+2\lambda w\big)$$
- $L(w)$: the data loss, how wrong the model is on the training data. $w_j$: the $j$-th weight. $\sum_jw_j^2$: the sum of the squares of all weights.
- $\lambda\ge0$: the regularization strength. At 0 there is none. As $\lambda$ grows, large weights are charged more.
- $\eta$: the learning rate. $\nabla L$: the data gradient.
- $2\lambda w$: the extra pull towards zero, proportional to the weight size, so big weights shrink more.

#flashcards/ml/formula #flashcards/topic/training Write L1 regularization and explain each symbol.
?
$$J(w)=L(w)+\lambda\sum_j |w_j|$$
- $|w_j|$: the absolute value of weight $j$ (size, ignoring sign). The other symbols are as in L2.
- The pull on a weight is $\lambda\cdot\operatorname{sign}(w_j)$, a constant push towards zero no matter how small the weight already is.
- That constant push can make weights exactly 0, so L1 gives sparse models. L2 shrinks weights but rarely makes them exactly 0.

#flashcards/ml/cloze #flashcards/topic/training L1 gives ==zeros== (the diamond has corners on the axes), while L2 shrinks all weights smoothly (the ==circle== has none).

#flashcards/ml/example #flashcards/topic/training Weights $w=(3,-1)$, data loss 2, $\lambda=0.1$. What are the L2 and L1 totals?::L2 penalty: 0.1 × (9 + 1) = 1.0, J = 3.0. L1 penalty: 0.1 × (3 + 1) = 0.4, J = 2.4. L1 charges less for large weights.

#flashcards/ml/example #flashcards/topic/training With $w=(3,-1)$, $\eta=0.1$, $\lambda=0.1$ and $\nabla L=(0.5,-0.2)$, what are the new $w_1$ and $w_2$ under L2?::w₁ = 3 − 0.1 × (0.5 + 2 × 0.1 × 3) = 2.89. w₂ = −1 − 0.1 × (−0.2 + 2 × 0.1 × (−1)) = −0.96. Both move towards zero more than the data gradient alone would.

#flashcards/ml/image #flashcards/topic/training ![[regularization-curves.svg]]
What does this picture show?
?
Seven data points. The wiggly blue curve and the smooth green curve both pass through every point, so both have zero training error. Regularization steers the model towards the smooth green one, which is much more likely to predict new points well.

#flashcards/ml/image #flashcards/topic/training ![[l1-l2-geometry.png]]
What does this picture show?
?
Why L1 gives zeros and L2 does not. The axes are two weights. The shape is the set of weights allowed by the penalty: a diamond for L1 and a circle for L2. The model takes the allowed point that touches the best-fit line. With the diamond that is a corner on an axis, where one weight is exactly zero. With the circle it is an ordinary point.
