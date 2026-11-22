---
tags: [eigenvalue, eigenvector, linear-algebra, math, optimization, PCA]
status: growing
level: 4
tier: 2
---

# Eigenvalue

For a [[Matrix]] (a machine that moves arrows), an **eigenvector** is a special arrow that the machine only **stretches or flips, without turning it**. The **eigenvalue** is how much it stretches.

## 🟢 1. Simple (age 12)
Imagine stretching a rubber sheet that has a picture on it. Most lines on the picture get tilted and bent. But a few special lines just get longer or shorter and stay pointing the same way. Those lines are the eigenvectors. How much each one stretches is its eigenvalue.

**Example:** Pull a trampoline mat sideways. Lines along your pull get longer (a stretch of, say, 3 times). Lines at a right angle might stay the same (a stretch of 1 time). Every other line gets tilted a bit. The two special directions and their stretch numbers describe the whole pull.

## 🔵 2. How it works
1. A [[Matrix]] times a [[Vector]] gives a new vector, usually pointing somewhere else.
2. For some special vectors the answer points along the **same line** as before, just longer, shorter, or flipped. These are the eigenvectors.
3. The number it was multiplied by is the eigenvalue. A value of 3 means three times longer. 0.5 means half as long. A negative value means flipped. 1 means unchanged.
4. **Why this is useful: it breaks a complicated machine into simple pieces.** In its own special directions, the matrix acts like plain multiplication by one number.
5. **Many repeats.** If you apply the same matrix again and again, an arrow along an eigenvector is multiplied by its eigenvalue every time. If the biggest eigenvalue (ignoring sign) is above 1, the arrow blows up. Below 1, it shrinks to nothing. This explains why signals vanish or explode in a [[Recurrent Neural Network]] ([[Vanishing and Exploding Gradients]]).
6. **For data.** Take a cloud of points. The eigenvectors of its "how things vary together" table point along the directions of the most spread, and the eigenvalues say how big that spread is. Keeping only the biggest few is [[PCA]].
7. **For learning.** The curvature of a loss bowl in each special direction is an eigenvalue. A steep direction (big eigenvalue) limits how big a step ([[Learning Rate]]) [[Gradient Descent]] can take without bouncing out ([[Convex Optimization]]).
8. The code in level 3 shows each of these on small examples.

![[symmetric-matrix-eigen-diagonalization.png]]
*A 2-by-2 symmetric matrix A (top left of the picture's formula: 0.75, 0.15, 0.15, 1.1) written as three simple moves. Q^T turns the grid so the red eigenvector lines become the axes, D stretches along those axes by the eigenvalues (about 1.155 and 0.695, shown in the middle matrix), and Q turns it back. The bottom-left grid (the whole matrix A at once) is the same as the result of those three steps. Author: Eteri byazrova, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Orthogonal_diagonalization_of_a_real_symmetric_matrix.png), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Only square matrices** have eigenvalues. An $n\times n$ matrix has $n$ of them, counting repeats; some can be complex numbers (a matrix that turns things has no real special direction).
- **Symmetric matrices are the friendly case.** If the matrix equals its own transpose (as covariance matrices and curvature matrices do), the eigenvalues are real and the eigenvectors are at right angles to each other. The code uses `eigh`, NumPy's routine for this case.
- **Handy facts.** The sum of the eigenvalues equals the sum of the diagonal numbers (the trace). The product equals the determinant. The code checks both on one matrix.
- **Finding them.** For tiny matrices, solve the characteristic equation (level 4). For big ones this is hopeless, because the equation has degree $n$. The **power method** instead keeps multiplying a random vector by the matrix and rescaling; it lines up with the eigenvector of the biggest eigenvalue. The code shows it reaching 3.0 in 30 steps.
- **In machine learning:**
  - [[PCA]]: eigenvectors of the covariance matrix are the principal directions, and the eigenvalues are the variances along them (see also [[Singular Value Decomposition]] and [[Autoencoder]]).
  - Optimisation: for a bowl-shaped loss with curvature matrix $H$, gradient descent converges if the learning rate is below $2/\lambda_{max}$ (code below, on a quadratic bowl only). Negative eigenvalues mean curving downwards ([[Convex Optimization]]).
  - Sequence models: the same matrix applied $t$ times scales along each eigenvector by $\lambda^t$.
- **The code below** checks the defining equation, the trace and determinant facts, the power method, a rotated data cloud, a learning-rate limit, and repeated multiplication.

```python
import numpy as np

A = np.array([[2.0, 1.0],
              [1.0, 2.0]])
vals, vecs = np.linalg.eigh(A)                  # eigh: for symmetric matrices, values come out sorted small to large
print("eigenvalues of A:", vals)
for lam, v in zip(vals, vecs.T):
    print(f"  direction {v.round(3)}: A @ v = {(A @ v).round(3)}  and  lambda * v = {(lam * v).round(3)}")
print("trace", np.trace(A), "= sum of eigenvalues", vals.sum(), "| determinant", round(np.linalg.det(A), 3), "= product", vals.prod())

# power method: keep multiplying a random vector by A and rescale; it lines up with the biggest eigenvalue's direction
v = np.random.RandomState(0).randn(2)
for _ in range(30):
    v = A @ v
    v /= np.linalg.norm(v)
print("power method after 30 steps: direction", v.round(3), "| estimate of largest eigenvalue", round(float(v @ A @ v), 4), "| error", f"{abs(float(v @ A @ v) - 3):.1e}")

# data cloud: eigenvalues of the covariance matrix are the spreads (variances) along the principal directions
rng = np.random.RandomState(1)
Z = rng.randn(5000, 2) * np.array([3.0, 0.5])    # spread 3 along one axis and 0.5 along the other
th = np.pi / 6
Rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
X = Z @ Rot.T                                    # tilt the cloud by 30 degrees
cv, cd = np.linalg.eigh(np.cov(X.T))
print("covariance eigenvalues:", cv.round(2), "| expected 0.25 and 9 | biggest-spread direction", cd[:, 1].round(2), "| the tilt gives", np.round([np.cos(th), np.sin(th)], 2))

# gradient descent on a bowl f(w) = 0.5 * w^T H w: it converges only if learning rate < 2 / (largest eigenvalue of H)
H = np.diag([1.0, 10.0])
lam_max = np.linalg.eigvalsh(H).max()
for lr in (0.05, 0.19, 0.21):
    w = np.array([1.0, 1.0])
    for _ in range(100):
        w = w - lr * (H @ w)
    print(f"learning rate {lr}: |w| after 100 steps = {np.linalg.norm(w):.3g}   (limit 2/lambda_max = {2 / lam_max})")

# repeating a matrix: the signal grows or shrinks like (largest |eigenvalue|) ** steps
for s in (0.9, 1.1):
    W = s * np.array([[0.0, 1.0], [1.0, 0.0]])   # eigenvalues are +s and -s
    h = np.array([1.0, 0.0])
    for _ in range(50):
        h = W @ h
    print(f"largest |eigenvalue| {s}: size after 50 repeats {np.linalg.norm(h):.3g}  (= {s} ** 50 = {s ** 50:.3g})")
```

What it printed: the matrix has eigenvalues 1 and 3. Along the direction (0.707, 0.707) the matrix gives (2.121, 2.121), which is 3 times the vector, and along (−0.707, 0.707) it changes nothing (a multiplier of 1). Trace 4 equals 1 + 3, determinant 3 equals 1 × 3. The power method reached 3.0 with an error of 4.4e-16. For the tilted data cloud the covariance eigenvalues were 0.25 and 9.09 (expected 0.25 and 9; the extra 0.09 is sampling noise from 5000 points), and the biggest-spread direction (−0.87, −0.5) is the 30-degree tilt (0.87, 0.5) with its sign flipped, which is the same line. With the loss bowl whose largest eigenvalue is 10 (limit 0.2), learning rates 0.05 and 0.19 shrank the point (size 0.00592 and 2.66e-05) while 0.21 blew it up (1.38e+04). A matrix with largest |eigenvalue| 0.9 shrank a vector to 0.00515 after 50 repeats (0.9⁵⁰), and 1.1 grew it to 117 (1.1⁵⁰).

## 🔴 4. Deep / formal
An eigenvector $v\neq0$ and its eigenvalue $\lambda$ for a square matrix $A$ satisfy:

$$
A\,v=\lambda\,v
$$

**Where:**
- $A$ is a square $n\times n$ matrix. $v$ is a vector with $n$ numbers, not all zero.
- $\lambda$ is a single number (the eigenvalue). Multiplying $v$ by the whole machine $A$ gives the same result as multiplying every number in $v$ by $\lambda$.
- Any multiple of $v$ is an eigenvector too, so only the direction counts (the code's $-0.87$ and $0.87$ are the same line).
- **Matters:** this is what lets a hard matrix act like a simple number along special directions. Then $A^t v=\lambda^t v$, so repeated use scales by $\lambda^t$. If $|\lambda|>1$ it explodes, if $|\lambda|<1$ it vanishes, which the last lines of the code show.

**Example (tiny numbers):** $A=\begin{pmatrix}2&1\\1&2\end{pmatrix}$ and $v=(1,1)$.
- $Av=(2\cdot1+1\cdot1,\;1\cdot1+2\cdot1)=(3,3)=\mathbf{3}\cdot(1,1)$, so $\lambda=3$.
- Try $v=(1,-1)$: $Av=(2-1,\;1-2)=(1,-1)=\mathbf{1}\cdot(1,-1)$, so $\lambda=1$.
- Meaning: along $(1,1)$ the matrix triples lengths, along $(1,-1)$ it leaves them alone. These are the same two directions the code found (written there with length 1, so $0.707$ in place of $1$).

To find $\lambda$ in general, notice that $Av=\lambda v$ means $(A-\lambda I)v=0$, which has a nonzero answer only if the matrix $A-\lambda I$ squashes something to zero, that is, its determinant is 0:

$$
\det(A-\lambda I)=0
$$

**Where:**
- $I$ is the identity matrix (1 on the diagonal, 0 elsewhere), so $\lambda I$ subtracts $\lambda$ from each diagonal number.
- $\det$ is the determinant, a single number computed from a square matrix. It is 0 exactly when the matrix flattens some direction to nothing.
- **Matters:** this turns "find special directions" into "find the roots of a polynomial in $\lambda$". It has degree $n$, so for big matrices one uses methods such as the power method instead.

**Example (tiny numbers):** same $A$.
- $A-\lambda I=\begin{pmatrix}2-\lambda&1\\1&2-\lambda\end{pmatrix}$, and its determinant is $(2-\lambda)(2-\lambda)-1\cdot1=(2-\lambda)^2-1$.
- Set it to 0: $(2-\lambda)^2=1$, so $2-\lambda=\pm1$, giving $\lambda=\mathbf{1}$ or $\lambda=\mathbf{3}$.
- Meaning: the same two values that were found by trying, and the same ones the code printed.

**Further notes (from the code above):**
- **Check:** trace $=$ sum of eigenvalues ($4=1+3$) and determinant $=$ product ($3=1\times3$), on this one matrix.
- **Learning-rate limit:** on a bowl with largest eigenvalue 10, rates 0.05 and 0.19 converged and 0.21 diverged, right on either side of $2/10=0.2$. This is for a quadratic bowl only. For a real network the curvature changes as it learns, and I did not test that.
- **Repeated product:** $0.9^{50}=0.00515$ and $1.1^{50}=117$, as the code's two runs printed.
- **What the code cannot show:** complex eigenvalues, matrices that are not symmetric, or any real network. Everything is on 2-by-2 examples or random data.

**What came before, and how this improved on it:**
- **Before:** the idea began in mechanics. In the 18th century Euler studied the spinning of a solid body and found the importance of its principal axes, and Lagrange saw that those axes are the eigenvectors of the inertia matrix. Cauchy used them in the early 19th century to classify surfaces, and coined the name "characteristic root" (all from Wikipedia's history section, read, a secondary source).
- **Its limit:** the idea was known only by solving a polynomial equation of degree $n$, which is hopeless for large matrices (the characteristic equation above).
- **What changed:** Hilbert used the German word *eigen* ("own") for them in 1904. The first numerical method, the power method, was published by von Mises in 1929, and the QR algorithm was proposed independently by Francis and by Kublanovskaya in 1961 (same page; read).
- **How much better:** in this note's code the power method found the largest eigenvalue of a 2-by-2 matrix as 3.0 with an error of 4.4e-16 after 30 matrix-vector products, without writing any equation. For a 512-by-512 matrix the characteristic equation has degree 512, while each power-method step is one matrix-vector product (about 262,144 multiplications). I only ran the 2-by-2 case.

## Related
- [[Matrix]] · [[Vector]] · [[PCA]] · [[Singular Value Decomposition]] · [[Autoencoder]] · [[Convex Optimization]] · [[Gradient Descent]] · [[Learning Rate]] · [[Recurrent Neural Network]] · [[Vanishing and Exploding Gradients]] · [[Calculus]] · [[Hessian]]

## Sources
- Wikipedia, "Eigenvalues and eigenvectors", History section (read through the Wikipedia API): the names and dates for Euler, Lagrange, Cauchy, Hilbert (1904), von Mises (1929) and Francis and Kublanovskaya (1961) are as that page states them. A secondary source; the original papers were not read.
- The $2/\lambda_{max}$ limit and the power-method behaviour are shown by the code only for the small cases printed, and are standard results, **from memory**, otherwise.
- Image: [File:Orthogonal diagonalization of a real symmetric matrix.png](https://commons.wikimedia.org/wiki/File:Orthogonal_diagonalization_of_a_real_symmetric_matrix.png) on Wikimedia Commons, CC BY-SA 4.0. I viewed it before captioning.

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/linear-algebra Eigenvector and eigenvalue:::An eigenvector is a special direction that a matrix only stretches or shrinks, without turning. The eigenvalue is how much it stretches.

#flashcards/math/formula #flashcards/topic/linear-algebra Write the eigenvalue equation and explain each symbol.
?
$$A\,v=\lambda\,v$$
- $A$: a square $n\times n$ matrix. $v$: a vector with $n$ numbers, not all zero (the eigenvector).
- $\lambda$: one number (the eigenvalue). Multiplying $v$ by the whole matrix gives the same as multiplying every number in $v$ by $\lambda$.
- Any multiple of $v$ is an eigenvector too, so only the direction counts.
- Along such a direction the hard matrix acts like a simple number, so $A^t v=\lambda^t v$. If $|\lambda|>1$ it explodes, if $|\lambda|<1$ it vanishes.

#flashcards/math/formula #flashcards/topic/linear-algebra Write the equation used to find eigenvalues and explain each symbol.
?
$$\det(A-\lambda I)=0$$
- $I$: the identity matrix (1 on the diagonal, 0 elsewhere), so $\lambda I$ subtracts $\lambda$ from each diagonal number.
- $\det$: the determinant, one number from a square matrix. It is 0 exactly when the matrix flattens some direction to nothing.
- This turns "find special directions" into "find the roots of a polynomial in $\lambda$". It has degree $n$, so big matrices use methods like the power method instead.

#flashcards/math/cloze #flashcards/topic/linear-algebra If $|\lambda|>1$ then repeated use of the matrix ==explodes== along that direction, and if $|\lambda|<1$ it ==vanishes==.

#flashcards/math/example #flashcards/topic/linear-algebra $A=\begin{pmatrix}2&1\\1&2\end{pmatrix}$. What are $Av$ and $\lambda$ for $v=(1,1)$ and $v=(1,-1)$?::A(1,1) = (3,3) = 3·(1,1), so λ = 3. A(1,−1) = (1,−1) = 1·(1,−1), so λ = 1. Along (1,1) lengths triple, along (1,−1) they stay the same.

#flashcards/math/example #flashcards/topic/linear-algebra Solve $\det(A-\lambda I)=0$ for $A=\begin{pmatrix}2&1\\1&2\end{pmatrix}$.::(2−λ)² − 1 = 0, so 2−λ = ±1, giving λ = 1 or λ = 3. The same values as found by trying.

#flashcards/math/image #flashcards/topic/linear-algebra ![[symmetric-matrix-eigen-diagonalization.png]]
What does this picture show?
?
A symmetric 2×2 matrix A written as three simple moves: Qᵀ turns the grid so the red eigenvector lines become the axes, D stretches along those axes by the eigenvalues, and Q turns it back.
