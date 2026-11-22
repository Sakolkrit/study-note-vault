---
tags: [math, linear-algebra, matrix, SVD, low-rank, machine-learning]
status: growing
level: 4
tier: 3
---

# Singular Value Decomposition

The singular value decomposition (SVD) takes **any** matrix and breaks it into three simple steps: **turn, stretch along the axes, turn again**. The stretch amounts, called singular values, tell you which parts of the matrix matter most.

## 🟢 1. Simple (age 12)
Imagine a perfectly round pizza dough. Someone pushes and twists it into a long oval. You can describe that mess in three easy moves: first **spin** the dough, then **stretch** it more in one direction than the other, then **spin** it again. The SVD finds those three moves for any squishing machine. The stretch numbers tell you the story: a few big ones mean the machine really only does a couple of big things, and the tiny ones are details you can ignore.

**Example:** A photo is just a big grid of numbers. The SVD finds the few "big patterns" of the photo, like the broad bands of light and dark, and ranks them. If you keep only the strongest patterns, you get a blurry but recognisable picture that needs far fewer numbers.

## 🔵 2. How it works
1. A [[Matrix]] can turn, stretch and squash arrows ([[Vector]]s). The shapes it makes can look messy.
2. Put a perfect circle of arrows through the matrix. It comes out as an **oval** (an ellipse).
3. The oval has a **long axis** and a **short axis**. Their lengths are the **singular values**, the biggest first.
4. The SVD names the whole journey: **turn** the input so the right axes line up, **stretch** along the axes by the singular values, then **turn** again into the output direction.
5. The singular values are ranked. The big ones carry the main behaviour, and the small ones are fine detail or noise.
6. **Keep just the top few** singular values, and you get a simpler matrix that is the best possible copy of the original with that little information (a low-rank approximation).

![[svd-geometry.svg]]
*The SVD of a slanting (shearing) matrix M, step by step. Top left: a blue circle with two arrows. Top right: after M, the circle becomes a tilted oval, whose two axes have lengths σ₁ (long) and σ₂ (short). The long way round is M = U · Σ · V*: first V* turns the circle (bottom left), then Σ stretches it along the horizontal and vertical axes (bottom right), and finally U turns it into place. Image: Georg-Johann, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Singular-Value-Decomposition.svg), CC BY-SA 3.0.*

![[svd-compression.jpg]]
*A photo (the original) rebuilt from only its top 1, 10 and 100 singular values. With 1 you get only broad bands of colour. With 10 you can make out a blurry red car. With 100 it looks almost like the original. Image: Samir.beall, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Svd_compression.jpg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Works for every matrix:** unlike eigenvalues (see [[Eigenvalue]]), which need a square matrix and can be awkward, the SVD exists for any matrix of any shape, and its singular values are always zero or positive.
- **The three parts:** $U$ and $V$ are rotations (their columns are at right angles to each other, each of length 1). $\Sigma$ is a diagonal matrix of the singular values, sorted from biggest to smallest.
- **Link to eigenvalues:** the singular values squared are the eigenvalues of $M^{\top}M$, and also of $MM^{\top}$. For a symmetric positive matrix the singular values and the eigenvalues are the same numbers.
- **Best approximation:** cutting the SVD after the top $k$ singular values gives the best matrix of "rank $k$" (the fewest independent patterns) that is closest to the original. This is the Eckart–Young result (from memory), and the code below checks the size of the error.
- **Rank:** the number of non-zero singular values. A matrix with few big singular values and many tiny ones is **close to low-rank**, so it is highly compressible.
- **Uses in ML and data science:**
  - **[[PCA]]:** the SVD of the centred data matrix gives the principal directions. It is how libraries usually do PCA, because it is more stable than computing the covariance matrix.
  - **Compression and denoising:** keep the top singular values and drop the rest (as in the car photo).
  - **[[Fine-tuning]] with LoRA:** instead of changing a huge weight matrix, you add a product of two thin matrices of rank $r$. This works on the idea that useful changes live in a low-rank space (stated from memory as the motivation of LoRA).
  - **Embeddings:** a classic way to get word vectors is to take the SVD of a word-pair count table and keep the top directions (see [[Embedding]] and [[GloVe]]).
  - **Solving least squares and finding the "pseudo-inverse"** of a matrix that has no true inverse (see [[Linear Regression]]).
  - **Stability analysis:** the biggest and smallest singular values, and their ratio, show how a layer stretches signals (see [[Vanishing and Exploding Gradients]]).
- **Cost:** a full SVD of an $m\times n$ matrix is expensive for huge matrices. For big data, people compute only the top few singular values with faster methods (from memory).

```python
import numpy as np
M = np.array([[1., 1], [0, 1]])                          # the shear in the picture
U, s, Vt = np.linalg.svd(M); print(s.round(3), np.allclose(U @ np.diag(s) @ Vt, M))     # [1.618 0.618] True
print(np.sqrt(np.linalg.eigvalsh(M.T @ M)).round(3)[::-1], (s ** 2).round(3), round(float(np.linalg.det(M)), 3), round(float(s.prod()), 3))
# [1.618 0.618] [2.618 0.382] 1.0 1.0
# low-rank approximation: a 6x5 matrix that is almost rank 2
rng = np.random.default_rng(0)
B = rng.normal(0, 1, (6, 2)) @ rng.normal(0, 1, (2, 5)) + 0.05 * rng.normal(0, 1, (6, 5))
U, s, Vt = np.linalg.svd(B, full_matrices=False); print(s.round(3))      # [5.054 2.148 0.116 0.079 0.024]
for k in (1, 2, 3):
    Bk = U[:, :k] * s[:k] @ Vt[:k]
    print(k, round(float(np.linalg.norm(B - Bk)), 4), round(float(np.sqrt((s[k:] ** 2).sum())), 4))   # 1 2.1522 2.1522 | 2 0.1424 0.1424 | 3 0.0825 0.0825
# storage / LoRA count
m = n = 1000; k = 10; print(m * n, k * (m + n + 1), round(k * (m + n + 1) / (m * n), 4))   # 1000000 20010 0.02
d = 4096; r = 8; print(d * d, 2 * d * r, round(2 * d * r / (d * d), 4))                     # 16777216 65536 0.0039
```

The shear has singular values 1.618 and 0.618, and their product is 1.0, the size of its determinant. They are the square roots of the eigenvalues of $M^{\top}M$ (2.618 and 0.382). The $6\times5$ test matrix was built from two real patterns plus a little noise, so its singular values drop sharply after the second (5.054, 2.148, then 0.116 and smaller). Keeping 2 patterns leaves an error of 0.1424, which is exactly the square root of the sum of the squares of the dropped singular values. The last lines count storage: a rank-10 copy of a $1000\times1000$ matrix needs 20,010 numbers instead of a million (2%), and a rank-8 add-on for a $4096\times4096$ layer needs 65,536 numbers instead of about 16.8 million (0.39%). Needs numpy.

## 🔴 4. Deep / formal
**The decomposition.** Any $m\times n$ matrix $M$ can be written as:

$$
M=U\,\Sigma\,V^{\top}
$$

**Where:**
- $U$ is an $m\times m$ rotation (its columns, the *left singular vectors*, are at right angles, each of length 1). $V$ is an $n\times n$ rotation (the *right singular vectors*). $V^{\top}$ is its transpose.
- $\Sigma$ ("sigma") is an $m\times n$ matrix with the singular values $\sigma_1\ge\sigma_2\ge\dots\ge0$ down its diagonal and zeros elsewhere.
- Read right to left: $V^{\top}$ turns the input, $\Sigma$ stretches each axis by its singular value, $U$ turns the result into the output space.
- **Matters:** the singular values are the stretch factors along the oval's axes. The product of all of them equals the absolute value of the determinant of a square matrix, and the biggest one is the largest amount that $M$ can stretch any arrow. Without the sorting (largest first), "keep the top few" would have no meaning.

**Example (tiny numbers):** the shear $M=\begin{pmatrix}1&1\\0&1\end{pmatrix}$ from the picture.
- Singular values from the code: $\sigma_1=\mathbf{1.618}$ and $\sigma_2=\mathbf{0.618}$.
- Product: $1.618\times0.618=1.000$, which equals the determinant of $M$ ($1\times1-1\times0=1$).
- Meaning: a circle of arrows becomes an oval whose long axis is 1.618 times the circle's radius, and whose short axis is 0.618 times it. The area stays the same (1.618 × 0.618 = 1), which is what a shear does.

**Where the singular values come from.** They are connected to eigenvalues (see [[Eigenvalue]]):

$$
\sigma_i=\sqrt{\lambda_i\!\left(M^{\top}M\right)}
$$

**Where:**
- $M^{\top}M$ is the matrix $M$ times its own transpose-from-the-left. It is always symmetric, so its eigenvalues are real and not negative.
- $\lambda_i(M^{\top}M)$ is the $i$-th eigenvalue of that matrix, sorted largest first. $\sigma_i$ is its square root.
- The eigenvectors of $M^{\top}M$ are the columns of $V$.
- **Matters:** this is how you can find the SVD by hand and why it always exists: $M^{\top}M$ is symmetric, so it has a clean set of eigen-axes. It also explains the link to [[PCA]]: for centred data $X$, the matrix $X^{\top}X$ (divided by $n$) is the covariance matrix.

**Example (tiny numbers):** the same shear.
- $M^{\top}M=\begin{pmatrix}1&0\\1&1\end{pmatrix}\begin{pmatrix}1&1\\0&1\end{pmatrix}=\begin{pmatrix}1&1\\1&2\end{pmatrix}$.
- Its eigenvalues (trace $3$, determinant $1$): $\lambda=\dfrac{3\pm\sqrt5}{2}=\mathbf{2.618}$ and $\mathbf{0.382}$.
- Square roots: $\sqrt{2.618}=\mathbf{1.618}$ and $\sqrt{0.382}=\mathbf{0.618}$.
- Meaning: the same numbers as the code. The shear itself has eigenvalues $1$ and $1$ (see [[Eigenvalue]]) but singular values $1.618$ and $0.618$: eigenvalues miss how much the shear really stretches.

**The best low-rank copy.** Keep only the first $k$ terms of the SVD, each a column times a row (a simple "pattern"):

$$
M_k=\sum_{i=1}^{k}\sigma_i\,\mathbf u_i\,\mathbf v_i^{\top},\qquad
\lVert M-M_k\rVert_F=\sqrt{\sum_{i>k}\sigma_i^{2}}
$$

**Where:**
- $\mathbf u_i$ and $\mathbf v_i$ are the $i$-th columns of $U$ and $V$. $\mathbf u_i\mathbf v_i^{\top}$ is a "pattern" matrix made from one column and one row, and $\sigma_i$ says how strongly it is used.
- $M_k$ is the sum of the top $k$ patterns, a matrix of rank $k$.
- $\lVert\cdot\rVert_F$ is the Frobenius norm: square every entry, add them up, take the square root (the ordinary length of the matrix as one long list of numbers).
- **Matters:** no other rank-$k$ matrix is closer to $M$ than $M_k$ (the Eckart–Young result; from memory, not proved here). The error is exactly what is hidden in the dropped singular values. So you can read off, from the singular values alone, how much you lose by cutting at any $k$. If the dropped values are tiny, you lose almost nothing.

**Example (tiny numbers):** $M=\begin{pmatrix}2&0&0\\0&1&0\\0&0&0.1\end{pmatrix}$, whose singular values are $2,\,1,\,0.1$.
- Keep $k=2$: $M_2$ keeps the entries $2$ and $1$ and drops the $0.1$.
- Error: $\sqrt{0.1^2}=\mathbf{0.1}$.
- Check directly: $M-M_2$ has the single entry $0.1$, so its length is $0.1$.
- Meaning: dropping the weakest pattern costs only 0.1, against a total size of about $\sqrt{4+1+0.01}=2.24$. In the code, the $6\times5$ matrix dropped from rank 5 to rank 2 gives an error of 0.1424, which equals $\sqrt{0.116^2+0.079^2+0.024^2}$.

**Why it saves space.** A rank-$k$ matrix of size $m\times n$ can be stored as $k$ columns, $k$ rows and $k$ singular values:

$$
\text{numbers stored}=k\,(m+n+1)\quad\text{instead of}\quad m\,n
$$

**Where:**
- $m$ and $n$ are the rows and columns. $k$ is the number of patterns kept.
- Each of the $k$ patterns needs a column of $m$ numbers, a row of $n$ numbers and one singular value.
- **Matters:** when $k$ is much smaller than $m$ and $n$, the saving is huge. It is the arithmetic behind image compression and LoRA (see [[Fine-tuning]]): a weight change is stored as two thin matrices instead of one fat one. If $k$ is close to $m$ or $n$, there is no saving.

**Example (tiny numbers):** a $1000\times1000$ matrix kept at $k=10$.
- Full: $1000\times1000=1{,}000{,}000$ numbers.
- Low-rank: $10\times(1000+1000+1)=\mathbf{20{,}010}$ numbers.
- Ratio: $20{,}010/1{,}000{,}000=\mathbf{2\%}$.
- Meaning: 2% of the storage. For LoRA, the two thin matrices $B$ and $A$ (without the singular-value vector) of rank $r=8$ on a $4096\times4096$ layer hold $2\times4096\times8=65{,}536$ numbers, about 0.39% of the 16.8 million.

**Further notes:**
- **Real photos are not exactly low-rank:** the car example uses 100 of its many singular values and still looks good, because natural images have fast-decaying singular values. Other data may not (from memory, an observation, not a law).
- **Truncation vs training:** LoRA does not compute an SVD of the weights. It trains thin matrices directly and *assumes* the useful change is low-rank.
- **Sign ambiguity:** the vectors in $U$ and $V$ are determined only up to a sign flip (applied to both together), so the printed vectors may differ between computers while the singular values do not.
- **Numerical use:** the ratio of the biggest to the smallest singular value (the condition number) says how risky it is to invert or solve with the matrix (see [[Eigenvalue]]).

**What came before, and how this improved on it:**
- **Before:** eigen-decomposition, which only applies to square matrices (and nicely only to special ones).
- **Its limit:** most real data tables, such as users by films or pixels by pictures, are not square.
- **What changed:** the SVD exists for every matrix and ranks its parts by importance. Keeping the top $k$ gives the best possible rank-$k$ approximation (Eckart and Young, 1936; from memory, not checked).
- **How much better:** in the code below, a 100 by 80 table (8,000 numbers) that is really 5 patterns plus a little noise is stored with 905 numbers, 8.8 times fewer, keeping 99.79% of its energy. The error of the approximation is 8.479, exactly what the discarded singular values predict.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
M = rng.standard_normal((100, 5)) @ rng.standard_normal((5, 80)) + 0.1 * rng.standard_normal((100, 80))
U, s, Vt = np.linalg.svd(M, full_matrices=False)
k = 5; Mk = (U[:, :k] * s[:k]) @ Vt[:k]
print("numbers stored: full", M.size, " rank-5 pieces", k * (100 + 80 + 1), " (", round(M.size / (k * 181), 1), "x fewer )")
print("error of the rank-5 version:", round(np.linalg.norm(M - Mk), 3), " leftover singular values predict:", round(np.sqrt((s[k:] ** 2).sum()), 3))
print("share of the matrix's energy kept:", round((s[:k] ** 2).sum() / (s ** 2).sum(), 4))
```

## Related
- [[Eigenvalue]] · [[Matrix]] · [[Vector]] · [[PCA]] · [[Fine-tuning]] · [[Embedding]] · [[GloVe]] · [[Linear Regression]] · [[Vanishing and Exploding Gradients]] · [[Normalization]] · [[Convex Optimization]] · [[Quantization]] · [[Autoencoder]] · [[LoRA]]
- [[Matrix Factorization]] · [[Recommender Systems]]
## Sources
- Strang, *Introduction to Linear Algebra*, chapter 7 (from memory)
- Golub & Van Loan, *Matrix Computations* (from memory)
- Eckart & Young, *The approximation of one matrix by another of lower rank*, 1936 (from memory)
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021 (from memory)
- Trefethen & Bau, *Numerical Linear Algebra*, lectures 4 and 5 (from memory)

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/linear-algebra Singular value decomposition (SVD):::Any matrix can be written as three simple moves: spin, stretch along the axes, spin again. The stretch numbers show how many big things the matrix really does.

#flashcards/math/formula #flashcards/topic/linear-algebra Write the SVD and explain each symbol.
?
$$M=U\,\Sigma\,V^{\top}$$
- $U$: an $m\times m$ rotation. Its columns (the left singular vectors) are at right angles and have length 1.
- $V$: an $n\times n$ rotation (the right singular vectors). $V^{\top}$ is its transpose.
- $\Sigma$: an $m\times n$ matrix with the singular values $\sigma_1\ge\sigma_2\ge\dots\ge0$ on its diagonal and zeros elsewhere.
- Read right to left: $V^{\top}$ turns the input, $\Sigma$ stretches each axis by its singular value, $U$ turns the result.

#flashcards/math/formula #flashcards/topic/linear-algebra Write how singular values come from eigenvalues, and explain each symbol.
?
$$\sigma_i=\sqrt{\lambda_i\!\left(M^{\top}M\right)}$$
- $M^{\top}M$: always symmetric, so its eigenvalues are real and not negative.
- $\lambda_i(M^{\top}M)$: its $i$-th eigenvalue, sorted largest first. $\sigma_i$ is its square root.
- The eigenvectors of $M^{\top}M$ are the columns of $V$.
- This is how you find the SVD by hand, and why it always exists: a symmetric matrix has a clean set of eigen-axes.

#flashcards/math/formula #flashcards/topic/linear-algebra Write the best rank-$k$ approximation and its error, and explain each symbol.
?
$$M_k=\sum_{i=1}^{k}\sigma_i\,\mathbf u_i\,\mathbf v_i^{\top},\qquad \lVert M-M_k\rVert_F=\sqrt{\sum_{i>k}\sigma_i^{2}}$$
- $\mathbf u_i$, $\mathbf v_i$: the $i$-th columns of $U$ and $V$. $\mathbf u_i\mathbf v_i^{\top}$ is a "pattern" matrix from one column and one row. $\sigma_i$ says how strongly it is used.
- $M_k$: the sum of the top $k$ patterns, a matrix of rank $k$.
- $\lVert\cdot\rVert_F$: the Frobenius norm, square every entry, add, take the square root.
- No other rank-$k$ matrix is closer to $M$.

#flashcards/math/formula #flashcards/topic/linear-algebra Write how many numbers a rank-$k$ SVD stores, and explain each symbol.
?
$$\text{numbers stored}=k\,(m+n+1)\quad\text{instead of}\quad m\,n$$
- $m$, $n$: the rows and columns. $k$: the number of patterns kept.
- Each pattern needs a column of $m$ numbers, a row of $n$ numbers and one singular value.
- When $k$ is much smaller than $m$ and $n$ the saving is huge. This is the arithmetic behind image compression and LoRA. If $k$ is close to $m$ or $n$ there is no saving.

#flashcards/math/cloze #flashcards/topic/linear-algebra The singular values are the ==square roots== of the eigenvalues of ==MᵀM==.

#flashcards/math/example #flashcards/topic/linear-algebra The shear $M=\begin{pmatrix}1&1\\0&1\end{pmatrix}$ has singular values 1.618 and 0.618. What do they multiply to, and how are they found?::1.618 × 0.618 = 1.000, the determinant of M. $M^\top M=\begin{pmatrix}1&1\\1&2\end{pmatrix}$ has eigenvalues (3 ± √5)/2 = 2.618 and 0.382, whose square roots are 1.618 and 0.618.

#flashcards/math/example #flashcards/topic/linear-algebra $M=\mathrm{diag}(2,1,0.1)$. What is the error of the best rank-2 approximation?::Keep 2 and 1, drop 0.1. Error = √(0.1²) = 0.1.

#flashcards/math/example #flashcards/topic/linear-algebra A 1000×1000 matrix is kept at rank $k=10$. How many numbers are stored compared with the full matrix?::10 × (1000 + 1000 + 1) = 20,010 numbers against 1,000,000. That is 2%.

#flashcards/math/image #flashcards/topic/linear-algebra ![[svd-geometry.svg]]
What does this picture show?
?
The SVD of a slanting (shear) matrix M step by step. A blue circle becomes a tilted oval after M. The oval's two axes have lengths σ₁ (long) and σ₂ (short).

#flashcards/math/image #flashcards/topic/linear-algebra ![[svd-compression.jpg]]
What does this picture show?
?
A photo rebuilt from only its top 1, 10 and 100 singular values. With 1 you get broad bands of colour. With 10 you can make out a blurry red car. With 100 it looks almost like the original.
