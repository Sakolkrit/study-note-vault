---
tags: [math, calculus, linear-algebra, derivative, backpropagation, deep-learning]
status: growing
level: 4
tier: 3
---

# Jacobian

The Jacobian is a **table of slopes**. For a machine with several inputs and several outputs, it says how much each output moves when you nudge each input.

## 🟢 1. Simple (age 12)
Imagine a cookie machine with two knobs, sugar and flour. It makes cookies with two measurable qualities: sweetness and softness. If you turn the sugar knob a tiny bit, sweetness changes by some amount and softness changes by some other amount. The same goes for the flour knob. That gives four numbers in a little square table. The table tells you, near where the knobs are right now, exactly how the cookies respond to small knob turns. That table is the Jacobian.

**Example:** A game character has two inputs, "push forward" and "push sideways", and two outputs, its new left-right position and its new up-down position. Four slopes answer "if I push a bit more forward, how much further right and up do I end up?", and the same for sideways. Those four numbers make the table.

## 🔵 2. How it works
1. Take a **map**: a rule that turns a list of numbers (the inputs) into another list of numbers (the outputs). See [[Vector]].
2. Pick a point, and nudge **one input** a tiny bit while holding the others still. See how much **each output** moves. Divide by the size of the nudge. Each of these answers is a slope (a "partial derivative"; see [[Calculus]]).
3. Do this for every input. Line up the answers in a table: **one row per output, one column per input**. This is a [[Matrix]], and it is the Jacobian.
4. Zoom in very close on a curvy map and it looks flat, like the ground looks flat when you stand on the round Earth. The Jacobian is that **flat copy**: it predicts the result of small nudges.
5. Multiplying the Jacobian by a list of small nudges gives the list of output changes.
6. The Jacobian also tells you how much the map **stretches or squashes areas** near that point. A small patch of area 1 may become area 3 or area 0.2 (see the picture).
7. In [[Backpropagation]], each layer of a network has a Jacobian, and the [[Chain Rule]] multiplies them together.

![[jacobian-determinant-distortion.svg]]
*A curved map $f$ sends a small square (left, red) to a bent patch (right, red). The black parallelogram is the flat copy that the Jacobian gives: close to the real patch, with the same corner and edge directions. The area of the parallelogram is $\lvert\det J\rvert$ times the area of the square. Image: Blacklemon67, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Jacobian_determinant_and_distortion.svg), CC BY-SA 3.0.*

## 🟠 3. Technical detail
- **Shape:** a map from $n$ inputs to $m$ outputs has an $m\times n$ Jacobian. The gradient is the special case of one output: a single row.
- **Chain rule:** the Jacobian of "do $g$, then $f$" is the product of the two Jacobians, in order. This is the matrix form of the [[Chain Rule]].
- **Never built in full:** a layer with 512 inputs and 512 outputs has $512\times512=262{,}144$ Jacobian entries. A network has many such layers, and batches multiply the size again. Frameworks therefore compute only **products**: "Jacobian times a vector" (forward mode) or "vector times Jacobian" (reverse mode, the one [[Backpropagation]] uses). These need only about as many numbers as the input or output has.
- **The softmax Jacobian** has a neat form: each entry is $s_i(\delta_{ij}-s_j)$, which is the matrix "diagonal of $s$ minus $s$ times its own flip" (see [[Softmax]]). Each row sums to 0, because pushing one score up must push the others down by the same total.
- **Determinant:** for a square Jacobian, its determinant is the local area (or volume) scale factor. It is how the "change of variables" rule in integrals works, for example the factor $r$ when switching to polar coordinates.
- **Singular Jacobian:** if the determinant is 0, the map squashes some direction flat near that point: some nudges change nothing, and the map cannot be undone there. The ratio of the biggest to the smallest stretch (from the [[Singular Value Decomposition]]) tells you how badly the map distorts.
- **Relation to the Hessian:** the [[Hessian]] is the Jacobian of the gradient.

```python
import numpy as np, torch
from torch.autograd.functional import jacobian

f = lambda v: torch.stack([v[0] * v[1], v[0] + v[1] ** 2])      # f(x1, x2) = (x1*x2, x1 + x2^2)
v = torch.tensor([2.0, 3.0], dtype=torch.float64)
J = jacobian(f, v)
print(J.tolist(), round(torch.det(J).item(), 6))                  # [[3, 2], [1, 6]]  det 16

d = torch.tensor([0.01, 0.0], dtype=torch.float64)                # nudge x1 by 0.01
print((f(v + d) - f(v)).tolist(), (J @ d).tolist())               # real change vs J times the nudge

r, th = 2.0, 0.7                                                  # polar -> x,y : the area scale factor is r
polar = lambda p: torch.stack([p[0] * torch.cos(p[1]), p[0] * torch.sin(p[1])])
print(round(torch.det(jacobian(polar, torch.tensor([r, th], dtype=torch.float64))).item(), 6))

softmax = lambda z: torch.softmax(z, 0)                           # softmax Jacobian = diag(s) - s s^T
z = torch.tensor([0.0, 0.0], dtype=torch.float64)
print(jacobian(softmax, z).tolist())

lin = torch.nn.Linear(512, 512).double()                          # why frameworks never build the full J
x0 = torch.randn(512, dtype=torch.float64)
Jfull = jacobian(lin, x0)
u = torch.randn(512, dtype=torch.float64)
xx = x0.clone().requires_grad_(True)
(lin(xx) @ u).backward()                                          # one backward pass gives u^T J
print(Jfull.numel(), xx.grad.numel(), torch.allclose(xx.grad, u @ Jfull))
```

Output: `[[3.0, 2.0], [1.0, 6.0]] 16.0`, then a real change of about `[0.03, 0.01]` that matches $J$ times the nudge, then `2.0` (the polar area factor $r$ at $r=2$), then `[[0.25, -0.25], [-0.25, 0.25]]` for the softmax of two equal scores, and finally `262144 512 True`: the full Jacobian has 262,144 numbers, one backward pass gives the 512 numbers of "vector times Jacobian", and they match.

## 🔴 4. Deep / formal
**Definition.** For a map $f$ that takes $n$ numbers and returns $m$ numbers, the Jacobian at a point is:

$$
J_{ij}=\frac{\partial f_i}{\partial x_j},\qquad J=\begin{pmatrix}\dfrac{\partial f_1}{\partial x_1}&\cdots&\dfrac{\partial f_1}{\partial x_n}\\ \vdots&\ddots&\vdots\\ \dfrac{\partial f_m}{\partial x_1}&\cdots&\dfrac{\partial f_m}{\partial x_n}\end{pmatrix}
$$

**Where:**
- $x_1,\dots,x_n$ are the inputs and $f_1,\dots,f_m$ the outputs.
- $\dfrac{\partial f_i}{\partial x_j}$ ("partial derivative") is how much output $i$ moves per tiny nudge of input $j$, with all other inputs held still.
- $i$ is the row number (which output) and $j$ the column number (which input).
- **Matters:** the table is only valid near the point where it was computed. Move to another point and the slopes change. Mixing up rows and columns gives the transpose, and backpropagation would then pass gradients the wrong way.

**Example (tiny numbers):** $f(x_1,x_2)=(x_1x_2,\;x_1+x_2^2)$ at $(2,3)$.
- Row 1 (output $x_1x_2$): $\partial/\partial x_1=x_2=3$ and $\partial/\partial x_2=x_1=2$.
- Row 2 (output $x_1+x_2^2$): $\partial/\partial x_1=1$ and $\partial/\partial x_2=2x_2=6$.
- $J=\begin{pmatrix}3&2\\1&6\end{pmatrix}$.
- Meaning: near $(2,3)$, raising $x_2$ by a tiny amount raises the second output six times as fast as the first input does.

**The flat copy.** Near a point, the map is approximately its Jacobian applied to the nudge:

$$
f(x+\delta)\;\approx\;f(x)+J\,\delta
$$

**Where:**
- $x$ is the point, and $\delta$ is a small nudge (a list of $n$ small numbers).
- $J\,\delta$ is the Jacobian (an $m\times n$ table) times the nudge. It predicts how the outputs change.
- $\approx$ means "close for small $\delta$". The error shrinks much faster than $\delta$ itself (it is second order).
- **Matters:** this is what "differentiable" means in practice, and it is the reason a [[Gradient Descent]] step works: a small step in the right direction changes the loss as the slopes predict. For a big step the flat copy is wrong.

**Example (tiny numbers):** the same $f$ at $(2,3)$, nudged by $\delta=(0.1,\,0.1)$.
- Prediction: $J\delta=(3\times0.1+2\times0.1,\;1\times0.1+6\times0.1)=(\mathbf{0.5},\,\mathbf{0.7})$.
- Real change: $f(2.1,\,3.1)=(6.51,\;11.71)$ and $f(2,3)=(6,\,11)$, so the change is $(0.51,\,0.71)$.
- Meaning: the flat copy is off by only $0.01$ in each part for this nudge. It would be much worse for a nudge of 1.

**Area stretching.** For a map with the same number of inputs as outputs, the Jacobian's determinant gives the local area (or volume) scale:

$$
\text{area after}\;\approx\;\lvert\det J\rvert\times\text{area before}
$$

**Where:**
- $\det J$ is the determinant of the square Jacobian (for a $2\times2$ table with entries $a,b,c,d$, it is $ad-bc$).
- $\lvert\cdot\rvert$ takes the size and drops the sign. A negative determinant means the map also flips orientation (like a mirror).
- "area before" is the area of a tiny patch of inputs; "area after" is the area of the patch it is sent to.
- **Matters:** when you change variables inside an integral or a probability density, you must multiply by this factor, or the total will be wrong. If $\det J=0$ the patch is squashed flat and the map cannot be undone there.

**Example (tiny numbers):** the same $J=\begin{pmatrix}3&2\\1&6\end{pmatrix}$ and a tiny square of area $0.01$ near $(2,3)$.
- $\det J=3\times6-2\times1=\mathbf{16}$.
- Area after: $16\times0.01=\mathbf{0.16}$.
- Meaning: near $(2,3)$ this map blows tiny areas up by 16 times. For polar coordinates the factor is $r$: at $r=2$ the code printed $2.0$, so areas double.

**Further notes:**
- **Forward and reverse mode:** products $J\,v$ (forward) and $u^{\top}J$ (reverse) are what [[Chain Rule]]-based software computes. Reverse mode is cheap when there are many inputs and one output (a loss), which is why it trains networks.
- **Conditioning:** very uneven stretches (a big gap between the largest and smallest singular values of $J$) make optimisation slow and numerically fragile (see [[Singular Value Decomposition]] and [[Eigenvalue]]).

**What came before, and how this improved on it:**
- **Before:** writing each partial derivative as a separate formula and handling them one by one.
- **Its limit:** the chain rule for several variables needs sums over many paths, which is messy to write and easy to get wrong, and there is no clean way to talk about "the derivative of a whole layer".
- **What changed:** stack all the slopes into one matrix, so the chain rule becomes matrix multiplication and a layer's derivative is one object. The determinant of such a table of slopes is named after Carl Jacobi, who studied it in the 1840s (from memory; not checked).
- **How much better:** in the code above, the full Jacobian of one $512\to512$ layer has 262,144 numbers, while the reverse-mode product that training needs has 512, which is 512 times smaller, and the two agree.

## Related
- [[Chain Rule]] · [[Calculus]] · [[Backpropagation]] · [[Matrix]] · [[Vector]] · [[Hessian]] · [[Singular Value Decomposition]] · [[Eigenvalue]] · [[Gradient Descent]] · [[Softmax]] · [[Neural Network]]

## Sources
- Goodfellow, Bengio & Courville, *Deep Learning*, section 4.3 (free online; from memory)
- Image: Blacklemon67, Wikimedia Commons, CC BY-SA 3.0 (see the caption in section 2)
- Numbers in this note come from the code above (my arithmetic and my run on this machine).

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/calculus-optimization Jacobian:::A table of slopes: one row per output, one column per input, each entry is how much that output moves per tiny nudge of that input.

#flashcards/math/formula #flashcards/topic/calculus-optimization Write the definition of the Jacobian and explain each symbol.
?
$$J_{ij}=\frac{\partial f_i}{\partial x_j},\qquad J=\begin{pmatrix}\dfrac{\partial f_1}{\partial x_1}&\cdots&\dfrac{\partial f_1}{\partial x_n}\\ \vdots&\ddots&\vdots\\ \dfrac{\partial f_m}{\partial x_1}&\cdots&\dfrac{\partial f_m}{\partial x_n}\end{pmatrix}$$
- $x_1,\dots,x_n$: the inputs. $f_1,\dots,f_m$: the outputs.
- $\partial f_i/\partial x_j$: how much output $i$ moves per tiny nudge of input $j$, other inputs held still.
- $i$ is the row (which output). $j$ is the column (which input).
- It is only valid near the point where it was computed. Mixing up rows and columns gives the transpose, which would pass gradients the wrong way in backpropagation.

#flashcards/math/example #flashcards/topic/calculus-optimization What is the Jacobian of $f(x_1,x_2)=(x_1x_2,\;x_1+x_2^2)$ at $(2,3)$?::Row 1: (x2, x1) = (3, 2). Row 2: (1, 2·x2) = (1, 6). `J = [[3, 2], [1, 6]]`. Raising x2 slightly raises the second output six times as fast as raising x1 does.

#flashcards/math/formula #flashcards/topic/calculus-optimization Write the first-order approximation with the Jacobian and explain each symbol.
?
$$f(x+\delta)\;\approx\;f(x)+J\,\delta$$
- $x$: the point. $\delta$: a small nudge (a list of $n$ small numbers).
- $J\,\delta$: the Jacobian times the nudge. It predicts the change in the outputs.
- $\approx$: close for small $\delta$. The error is second order, so it shrinks much faster than the nudge.
- This is why a small gradient descent step works. For a big step the flat copy is wrong.

#flashcards/math/example #flashcards/topic/calculus-optimization With the same $f$ at $(2,3)$ and a nudge $(0.1, 0.1)$, how close is $J\delta$ to the real change?::J times the nudge = (0.5, 0.7). Real change: f(2.1, 3.1) = (6.51, 11.71) minus f(2, 3) = (6, 11), so (0.51, 0.71). Off by only 0.01 in each part. My arithmetic.

#flashcards/math/formula #flashcards/topic/calculus-optimization Write how a map changes small areas using the Jacobian and explain each symbol.
?
$$\text{area after}\;\approx\;\lvert\det J\rvert\times\text{area before}$$
- $\det J$: the determinant of the square Jacobian (for a 2 by 2 table with entries $a,b,c,d$ it is $ad-bc$).
- $\lvert\cdot\rvert$: the size, dropping the sign. A negative sign means a flip (like a mirror).
- Area before: a tiny patch of inputs. Area after: the patch it is sent to.
- Needed when changing variables in an integral or a probability density. If the determinant is 0 the patch is squashed flat and the map cannot be undone there.

#flashcards/math/example #flashcards/topic/calculus-optimization With $J=\begin{pmatrix}3&2\\1&6\end{pmatrix}$, what is $\det J$ and what happens to a tiny square of area 0.01?::det J = 3·6 − 2·1 = 16. Area after = 16 × 0.01 = 0.16. Areas near that point blow up 16 times. For polar coordinates the factor is r, so at r = 2 areas double.

#flashcards/math/concept #flashcards/topic/calculus-optimization Why do frameworks never build the full Jacobian of a layer?
?
A 512 to 512 layer has 262,144 Jacobian entries. Reverse mode computes only the vector-times-Jacobian product that backpropagation needs, which has 512 numbers (512 times fewer, and it matches). Forward mode computes Jacobian-times-vector.

#flashcards/math/concept #flashcards/topic/calculus-optimization What is the Hessian in terms of the Jacobian?
?
The Hessian is the Jacobian of the gradient (the table of second derivatives).

#flashcards/math/concept #flashcards/topic/calculus-optimization What is the softmax Jacobian for two equal scores?::`[[0.25, -0.25], [-0.25, 0.25]]` (my run). Each row sums to 0, because pushing one probability up must push the others down by the same total.

#flashcards/math/image #flashcards/topic/calculus-optimization ![[jacobian-determinant-distortion.svg]]
What does this picture show?
?
A curved map $f$ sends a small square to a bent patch. The black parallelogram is the Jacobian's flat copy of that patch: it is close to the real patch, and its area is $\lvert\det J\rvert$ times the area of the square.

#flashcards/math/cloze #flashcards/topic/calculus-optimization The gradient is the Jacobian of a map with ==one output==, a single ==row==.
