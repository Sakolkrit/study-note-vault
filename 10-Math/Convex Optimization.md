---
tags: [math, optimization, convexity, machine-learning, calculus]
status: growing
level: 4
tier: 3
---

# Convex Optimization

Convex optimization is the study of finding the lowest point of a **bowl-shaped** function. Bowls are special because they have **only one valley**, so rolling downhill is guaranteed to reach the very bottom.

## 🟢 1. Simple (age 12)
Picture two landscapes. In the first there is one big smooth bowl, like the inside of a salad bowl. Drop a marble anywhere, and it rolls to the same lowest spot every time. In the second landscape there are many hills and little dips. Drop a marble and it may get stuck in a small dip that is **not** the lowest place. Convex optimization is about the first kind of landscape, where you cannot get stuck, so finding the best answer is easy and reliable.

**Example:** You are looking for the cheapest price for a toy by walking along a shopping street where prices only ever go down until the cheapest shop, then up. A bowl shape. You simply keep walking in the direction where prices fall, and you will find the cheapest shop. If prices zigzag, you might stop at a shop that is cheap only compared with its neighbours.

## 🔵 2. How it works
1. A learning problem has a **loss** that depends on settings (see [[Loss Function]]). Training means finding the settings with the lowest loss, which is optimisation. In a picture with two settings, the loss is a landscape.
2. A function is **convex** if its landscape is a bowl. Test: pick any two points on the curve and draw a straight line between them. In a bowl, the line never dips below the curve.
3. If the landscape is a bowl, **any place where the slope is zero is the global lowest point**. There are no traps.
4. So [[Gradient Descent]] (always step downhill) is guaranteed to get there, if the steps are small enough.
5. Many classic [[Machine Learning]] problems are bowls: [[Linear Regression]] with squared error, [[Logistic Regression]], and the [[Support Vector Machine]]. That is part of why they are reliable.
6. Deep [[Neural Network]]s are **not** bowls. Their landscape is bumpy, with many dips, and training them still usually works well, but without a guarantee.

![[convex-function-chord.svg]]
*A convex curve. Pick two points P and Q on it and join them with a straight line: the line lies above the curve in between. This is the test for being bowl-shaped. Image: NalesnikLD, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Convex_function.svg), public domain.*

![[nonconvex-local-minima.svg]]
*The opposite: a bumpy, non-convex landscape (the two arrows θᵢ and θⱼ are two settings, the height is the loss). It has many separate dips, so a marble can get stuck in the wrong one. The picture comes from a page on non-linear least squares fitting. Image: Daniele de Rigo, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Multiple_local_minima.svg), CC BY-SA 3.0.*

## 🟠 3. Technical detail
- **Convex function:** the line between any two points on its graph lies on or above the graph. **Convex set:** a region where the line between any two of its points stays inside it. A convex optimisation problem asks for the lowest point of a convex function over a convex set.
- **Key fact:** for a convex function, *every local minimum is a global minimum*. The set of best points is itself convex, and if the function is strictly convex (a round bowl) there is only one best point.
- **Easy to check:** for smooth functions, convex means the slope is always increasing, or in many dimensions that the **curvature** (the [[Matrix]] of second derivatives, called the Hessian) is never negative in any direction (all its eigenvalues, see [[Eigenvalue]], are at least 0).
- **Step size rule:** gradient descent on a bowl converges if the step ([[Learning Rate]]) is small enough relative to how sharply the bowl curves. Too large a step overshoots the bottom and bounces away, shown below.
- **Examples that are convex:** squared error, the log loss of [[Logistic Regression]] (as a function of the weights), the hinge loss of the [[Support Vector Machine]], and a sum of convex pieces. L1 and L2 penalties in [[Regularization]] are convex too, so adding them keeps the problem convex.
- **Convex but not smooth:** the absolute value $|x|$ has a sharp corner but is still convex. That is why methods such as the lasso (L1) can be solved reliably (see [[Regularization]]).
- **Not convex:** [[Neural Network]] training, [[Clustering]] by k-means, and fitting [[Gaussian Mixture Model]]s ([[Expectation-Maximization]] only finds a local peak). These are bumpy.
- **Why deep learning works anyway:** it is an open research topic. Common observations (from memory) are that in very large networks most stationary points are saddle points rather than bad dips, that noise in [[Stochastic Gradient Descent]] helps escape them, and that many different dips give similar quality.
- **Tools:** specialised solvers (linear and quadratic programming, interior-point methods) handle convex problems exactly for moderate sizes. For huge problems, gradient-based methods win.
- **Strong guarantees:** for a convex, smooth function, gradient descent with a good step size reaches loss within about $1/T$ of the best after $T$ steps, and much faster (a steady shrink factor) for a strictly curved bowl (from memory; see the details in [[Gradient Descent]]).

```python
import numpy as np
rng = np.random.default_rng(0)
# 1) test the chord rule  f(l*x+(1-l)*y) <= l*f(x)+(1-l)*f(y)  on random points
def violations(f, n=20000):
    x, y = rng.uniform(-3, 3, n), rng.uniform(-3, 3, n); l = rng.uniform(0, 1, n)
    return int((f(l * x + (1 - l) * y) > l * f(x) + (1 - l) * f(y) + 1e-12).sum())
for name, f in [("x^2", lambda x: x ** 2), ("|x|", np.abs), ("exp", np.exp), ("sin", np.sin), ("x^4-3x^2+x", lambda x: x ** 4 - 3 * x ** 2 + x)]:
    print(name, violations(f))                     # x^2 0 | |x| 0 | exp 0 | sin 10068 | x^4-3x^2+x 4084
# 2) gradient descent: convex bowl -> same answer from any start; bumpy -> depends on the start
gd = lambda grad, x, lr, n=2000: [x := x - lr * grad(x) for _ in range(n)][-1]
bowl = lambda x: 2 * (x - 1)                        # f = (x-1)^2
bumpy = lambda x: 4 * x ** 3 - 6 * x + 1            # f = x^4 - 3x^2 + x   (two valleys)
print([round(gd(bowl, s, 0.1), 3) for s in (-5, 0, 7)])      # [1.0, 1.0, 1.0]
print([round(gd(bumpy, s, 0.01), 3) for s in (-2, 2)])       # [-1.301, 1.131]
f = lambda x: x ** 4 - 3 * x ** 2 + x; print([round(f(v), 3) for v in (gd(bumpy, -2, .01), gd(bumpy, 2, .01))])   # [-3.514, -1.07]
# 3) step-size rule on f = 1/2 * a * x^2:  x_t = (1 - lr*a)^t * x_0
a = 2.0
for lr in (0.25, 1.2):
    x = 4.0; xs = [x]
    for _ in range(4): x = x - lr * a * x; xs.append(round(x, 3))
    print(lr, xs)    # 0.25 [4.0, 2.0, 1.0, 0.5, 0.25]  |  1.2 [4.0, -5.6, 7.84, -10.976, 15.366]
# 4) Hessian test: eigenvalues all >= 0 means convex
H1 = np.array([[2., 0], [0, 2.]]); H2 = np.array([[2., 0], [0, -2.]])          # x^2+y^2  and  x^2-y^2
print(np.linalg.eigvalsh(H1), np.linalg.eigvalsh(H2))     # [2. 2.] [-2.  2.]
X = rng.normal(0, 1, (50, 3)); print(np.linalg.eigvalsh(2 * X.T @ X / 50).round(2).min() >= 0)   # True  (least squares loss is convex)
```

Reading the output. Test 1 tries the chord rule on 20,000 random pairs. The convex functions $x^2$, $|x|$ and $e^x$ never break it (0 violations), while $\sin x$ and the two-valley $x^4-3x^2+x$ break it thousands of times. Test 2 shows what that means for descent: on the bowl $(x-1)^2$ every start lands at 1.0, but on the two-valley function the start decides: from $-2$ it ends at $-1.301$ (loss $-3.514$, the true lowest point) and from $+2$ it ends at $1.131$ (loss $-1.07$, a worse dip). Test 3 shows the step-size rule: with a small step the value halves every time, while with a step that is too big it flips sign and grows. The last test checks that the curvature of a least-squares loss is never negative (one random data set; this is a check on a sample, not a proof).

## 🔴 4. Deep / formal
**Convexity.** A function $f$ is convex if, for any two points $x,y$ and any mixing weight $\lambda$ between 0 and 1, the value at the mixed point is no more than the mix of the values:

$$
f\big(\lambda x+(1-\lambda)y\big)\;\le\;\lambda f(x)+(1-\lambda)f(y)
$$

**Where:**
- $x$ and $y$ are two inputs (numbers, or lists of numbers in many dimensions). $f$ is the function, such as a loss.
- $\lambda$ ("lambda") is a number from 0 to 1. $\lambda x+(1-\lambda)y$ is a point on the straight segment between $x$ and $y$ (halfway if $\lambda=\tfrac12$).
- The left side is the height of the curve at that point. The right side is the height of the straight chord joining the two curve points at the same place.
- **Matters:** the inequality says "the curve never rises above its chord", which is the bowl shape. If it fails somewhere, there is a bump, and descent can get trapped there.

**Example (tiny numbers):** $f(x)=x^2$, with $x=0$, $y=4$ and $\lambda=0.5$.
- Mixed point: $0.5\times0+0.5\times4=2$. Left side: $f(2)=4$.
- Right side: $0.5\times f(0)+0.5\times f(4)=0.5\times0+0.5\times16=8$.
- $4\le8$ holds, as it must for a bowl.
- Meaning: the curve at the midpoint (4) is below the chord (8). For $f=\sin$ with $x=0$, $y=\pi$ the test fails: left $=\sin(\pi/2)=1$ against right $=0.5\times0+0.5\times0=0$, since $1>0$, so $\sin$ is not convex.

**Zero slope means the best point.** For a smooth convex function the tangent line lies below the curve everywhere:

$$
f(y)\;\ge\;f(x)+\nabla f(x)^{\top}(y-x)
$$

**Where:**
- $\nabla f(x)$ is the gradient (the slope, a list in many dimensions) at the point $x$. $(y-x)$ is the step from $x$ to any other point $y$.
- The right side is the **tangent plane** at $x$, a straight-line guess of the function near $x$. The inequality says the true function never goes below it.
- **Matters:** if the slope at $x^*$ is zero, the right side becomes just $f(x^*)$, so $f(y)\ge f(x^*)$ for **every** $y$. That is the proof that a flat point of a bowl is the global bottom. For a non-convex function the tangent can cross the curve, and a flat point can be just a local dip.

**Example (tiny numbers):** $f(x)=x^2$, tangent at $x=1$ is $f(1)+f'(1)(y-1)=1+2(y-1)$.
- At $y=3$: tangent gives $1+2\times2=5$, and the true value is $f(3)=9$. $9\ge5$.
- At $y=-1$: tangent gives $1+2\times(-2)=-3$, true value $f(-1)=1$. $1\ge-3$.
- At $x=0$ the slope is $0$, so the tangent is the flat line at height $0$, and $f(y)=y^2\ge0$ for all $y$.
- Meaning: the curve always sits above its tangent lines, and the one flat tangent (at $x=0$) marks the lowest point.

**Descent on a bowl, with a step-size rule.** For the simplest bowl $f(x)=\tfrac12 a x^2$ with $a>0$, the slope is $a x$, so gradient descent with step size $\eta$ gives:

$$
x_{t+1}=x_t-\eta\,a\,x_t=(1-\eta a)\,x_t\quad\Longrightarrow\quad x_t=(1-\eta a)^t\,x_0
$$

**Where:**
- $x_t$ is the position after $t$ steps and $x_0$ is the start. $\eta$ ("eta") is the step size ([[Learning Rate]]). $a$ is how sharply the bowl curves.
- $(1-\eta a)$ is the shrink factor applied at every step.
- **Matters:** the walk converges to the bottom $x=0$ exactly when $|1-\eta a|<1$, that is, $0<\eta<2/a$. A steeper bowl (bigger $a$) needs a smaller step. With $\eta>2/a$ the factor is less than $-1$, so each step overshoots and the position grows and alternates in sign, shown by the code.

**Example (tiny numbers):** $a=2$ and start $x_0=4$.
- Step size $\eta=0.25$: factor $1-0.5=0.5$, so $x=4,\,2,\,1,\,0.5,\,0.25$. It halves each step towards 0.
- Step size $\eta=1.2$: factor $1-2.4=-1.4$, so $x=4,\,-5.6,\,7.84,\,-10.976,\,15.366$. It flips sign and grows.
- Meaning: the safe range here is $\eta<2/a=1$. The first step size is inside it and the second is outside, matching the code output.

**The curvature test.** For a function that can be differentiated twice, convexity means the **matrix of second derivatives** (the Hessian $\nabla^2f$) is positive semi-definite, that is, none of its eigenvalues is negative:

$$
\nabla^2 f(x)\succeq 0\quad\Longleftrightarrow\quad\text{every eigenvalue of }\nabla^2 f(x)\ \text{is}\ \ge 0
$$

**Where:**
- $\nabla^2 f(x)$ is the Hessian: a [[Matrix]] holding how fast the slope changes in every pair of directions, at the point $x$.
- An **eigenvalue** is the curvature along one special direction (see [[Eigenvalue]]). Positive means curving up like a bowl in that direction, negative means curving down like a hill.
- $\succeq0$ is the shorthand for "positive semi-definite": no direction curves downwards.
- **Matters:** this turns "is it a bowl?" into a computable test. One negative eigenvalue means there is a direction in which the surface curves down, so it is a saddle or a hill, not a bowl.

**Example (tiny numbers):** two functions of two settings.
- $f=x^2+y^2$: Hessian $\begin{pmatrix}2&0\\0&2\end{pmatrix}$, eigenvalues $2$ and $2$. Both $\ge0$, so it is a bowl (**convex**).
- $f=x^2-y^2$: Hessian $\begin{pmatrix}2&0\\0&-2\end{pmatrix}$, eigenvalues $2$ and $-2$. One is negative, so it curves up along $x$ but down along $y$: a **saddle**, not convex.
- For a least-squares loss $\tfrac1n\lVert X\mathbf w-\mathbf y\rVert^2$ the Hessian is $\tfrac2n X^{\top}X$, which has no negative eigenvalues. The check in the code agrees.
- Meaning: the first has a single bottom. The second has no bottom at all: it falls forever in the $y$ direction.

**Further notes:**
- **Strong convexity:** if the curvature is at least some positive number in every direction, the bowl is round, there is exactly one bottom, and descent shrinks the error by a fixed factor each step (see [[Gradient Descent]]).
- **Duality:** convex problems have a "mirror" problem (the dual) whose answer bounds the original. This is used by methods such as the support vector machine (from memory; not covered here).
- **Constraints:** minimising over a convex region (for example "weights must be positive") stays easy, and the best point can lie on the edge.
- **Limits:** most interesting modern models are not convex. The theory still guides step-size choices, regularisation and ideas such as momentum ([[Momentum]]).

**What came before, and how this improved on it:**
- **Before:** general optimisation, where descent may stop in any dip and the result depends on where you start.
- **Its limit:** you cannot tell whether the answer you found is the best one.
- **What changed:** restrict to convex problems, where every dip is the global minimum, so a local answer is the answer. Many classic methods ([[Linear Regression]], [[Logistic Regression]], [[Support Vector Machine]]) are convex.
- **How much better:** in the code below, gradient descent on a convex bowl from 20 random starts ended at one point, 1.0. On a function with two valleys, the same starts ended at two different points, -1.0 and 1.0. Neural networks are not convex, which is why starting points and tricks matter there.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
def descend(grad, x, lr=0.01, steps=3000):
    for _ in range(steps): x -= lr * grad(x)
    return round(float(x), 2)
convex = lambda x: 2 * (x - 1)                       # f(x) = (x-1)^2
double_well = lambda x: 4 * x ** 3 - 4 * x   # f(x) = x^4 - 2x^2, two valleys at -1 and +1
starts = rng.uniform(-2, 2, 20)
print("convex bowl, 20 random starts, end points:", sorted({descend(convex, float(s)) for s in starts}))
print("two-valley function, same starts:         ", sorted({descend(double_well, float(s)) for s in starts}))
```

## Related
- [[Gradient Descent]] · [[Calculus]] · [[Matrix]] · [[Eigenvalue]] · [[Loss Function]] · [[Learning Rate]] · [[Momentum]] · [[Stochastic Gradient Descent]] · [[Regularization]] · [[Linear Regression]] · [[Logistic Regression]] · [[Support Vector Machine]] · [[Neural Network]] · [[Clustering]] · [[Gaussian Mixture Model]] · [[Expectation-Maximization]] · [[Machine Learning]] · [[Binary Search]] · [[Hessian]] · [[Lagrange Multipliers]]

## Sources
- Boyd & Vandenberghe, *Convex Optimization* (free online), chapters 2 to 4
- Nocedal & Wright, *Numerical Optimization* (from memory)
- Goodfellow, Bengio & Courville, *Deep Learning*, chapter 8 (free online)
- Bubeck, *Convex Optimization: Algorithms and Complexity* (free online; from memory)

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/calculus-optimization Convex optimization:::Finding the lowest point of a bowl-shaped function. You cannot get stuck in a small dip, so any flat point is the true bottom.

#flashcards/math/formula #flashcards/topic/calculus-optimization Write the definition of a convex function and explain each symbol.
?
$$f\big(\lambda x+(1-\lambda)y\big)\;\le\;\lambda f(x)+(1-\lambda)f(y)$$
- $x$, $y$: two inputs (numbers or lists of numbers). $f$: the function, such as a loss.
- $\lambda$: a number from 0 to 1. $\lambda x+(1-\lambda)y$ is a point on the straight segment between $x$ and $y$ (halfway for $\lambda=\tfrac12$).
- Left side: the height of the curve at that point. Right side: the height of the straight chord at the same place.
- The curve never rises above its chord. If this fails somewhere there is a bump, and descent can get trapped.

#flashcards/math/formula #flashcards/topic/calculus-optimization Write the first-order condition for convexity and explain each symbol.
?
$$f(y)\;\ge\;f(x)+\nabla f(x)^{\top}(y-x)$$
- $\nabla f(x)$: the gradient (the slope) at $x$. $(y-x)$: the step from $x$ to any other point $y$.
- Right side: the tangent plane at $x$, a straight-line guess near $x$. The function never goes below it.
- If the slope at $x^*$ is zero, then $f(y)\ge f(x^*)$ for every $y$. That proves a flat point of a bowl is the global bottom.

#flashcards/math/formula #flashcards/topic/calculus-optimization Write gradient descent on the 1-D bowl $f=\tfrac12 a x^2$ and say when it converges.
?
$$x_{t+1}=x_t-\eta\,a\,x_t=(1-\eta a)\,x_t\quad\Longrightarrow\quad x_t=(1-\eta a)^t\,x_0$$
- $x_t$: position after $t$ steps. $x_0$: the start. $\eta$: the step size (learning rate). $a$: how sharply the bowl curves.
- $(1-\eta a)$: the shrink factor applied every step.
- It converges to 0 exactly when $|1-\eta a|<1$, that is $0<\eta<2/a$. A steeper bowl needs a smaller step. Beyond that each step overshoots and the position grows and flips sign.

#flashcards/math/formula #flashcards/topic/calculus-optimization Write the Hessian test for convexity and explain each symbol.
?
$$\nabla^2 f(x)\succeq 0\quad\Longleftrightarrow\quad\text{every eigenvalue of }\nabla^2 f(x)\ \text{is}\ \ge 0$$
- $\nabla^2 f(x)$: the Hessian, a matrix of how fast the slope changes in every pair of directions at $x$.
- Eigenvalue: the curvature along one special direction. Positive curves up like a bowl, negative curves down like a hill.
- $\succeq0$: positive semi-definite, no direction curves downwards.
- One negative eigenvalue means a saddle or hill, not a bowl.

#flashcards/math/cloze #flashcards/topic/calculus-optimization A function is convex when every eigenvalue of its Hessian is ==≥ 0==. One ==negative== eigenvalue means a saddle or hill.

#flashcards/math/example #flashcards/topic/calculus-optimization For $f(x)=x^2$ with $x=0$, $y=4$, $\lambda=0.5$, what do the two sides of the convexity inequality give?::Left: f(2) = 4. Right: 0.5 × 0 + 0.5 × 16 = 8. 4 ≤ 8 holds. For sin with x = 0, y = π it fails (1 against 0).

#flashcards/math/example #flashcards/topic/calculus-optimization Gradient descent on $f=x^2$ ($a=2$) from $x_0=4$: what happens with $\eta=0.25$ and $\eta=1.2$?::η = 0.25: factor 0.5, so 4, 2, 1, 0.5, 0.25 (converges). η = 1.2: factor −1.4, so 4, −5.6, 7.84, −10.98, 15.37 (diverges). The safe range is η < 2/a = 1.

#flashcards/math/image #flashcards/topic/calculus-optimization ![[convex-function-chord.svg]]
What does this picture show, and what is the test?
?
A convex curve. Join two points P and Q on it with a straight line: the line lies above the curve in between. That is the test for being bowl-shaped.

#flashcards/math/image #flashcards/topic/calculus-optimization ![[nonconvex-local-minima.svg]]
What does this picture show, and why is it a problem for descent?
?
A bumpy, non-convex landscape (two settings, height is the loss) with many separate dips. A marble can get stuck in the wrong dip, not the lowest one.
