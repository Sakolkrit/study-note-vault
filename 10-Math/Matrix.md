---
tags: [matrix, linear-algebra, math, neural-network, LLM]
status: growing
level: 4
tier: 1
---

# Matrix

A **grid of numbers** with rows and columns. It stores a table of numbers, and it also acts as a machine: feed it a [[Vector]] and it gives back another vector. Almost everything a neural network does is matrices times vectors.

## 🟢 1. Simple (age 12)
A matrix is a table, like a school timetable or a spreadsheet: rows go across, columns go down, and every box holds a number. The surprising part is that a table can also **do something**. A recipe table that says "for each cake, how much flour, sugar and eggs" can be combined with "how many of each cake I'm baking" to give the total shopping list in one go.

**Example:** A shop sells 2 kinds of snack bags. Bag 1 has 1 apple, 2 nuts and 3 raisins. Bag 2 has 4 apples, 5 nuts and 6 raisins. That is a table with 2 rows and 3 columns. If you want "1 pile of apples, 0 piles of nuts, 2 piles of raisins" in each bag, you multiply each row of the table by that list and add up: bag 1 gives 7 things, bag 2 gives 16.

## 🔵 2. How it works
1. **Shape.** A matrix with 2 rows and 3 columns has shape $2\times3$ (rows first, then columns).
2. **Matrix times vector.** Take each row, multiply it box by box with the vector, and add up. The result has one number per row. The vector needs as many numbers as the matrix has columns (this is the shape rule: $2\times3$ times a 3-number vector gives a 2-number vector).
3. **Matrix times matrix.** Do step 2 for every column of the second matrix. A $2\times3$ matrix times a $3\times2$ matrix gives a $2\times2$ matrix. The inner numbers (3 and 3) must match and disappear. The outer numbers (2 and 2) give the new shape.
4. **Order matters.** $P$ times $Q$ is usually **not** the same as $Q$ times $P$. Even the shape can change (code below).
5. **Transpose.** Flip a matrix over its diagonal: rows become columns. A $2\times3$ becomes $3\times2$.
6. **A matrix is also a move.** A small matrix can turn, stretch or flip arrows. Multiplying two such matrices means "do one move, then the other".
7. **In AI.** A layer of a [[Neural Network]] is "multiply the input vector by a matrix of learned numbers (the weights), then add a bias vector". Words become vectors ([[Embedding]]). [[Attention]] and the [[Feed-Forward Network]] in a [[Transformer]] are made of such multiplications. Training is mostly about finding good numbers for these matrices ([[Gradient Descent]]).
8. Computers do these multiplications very fast, because the work can be done in parallel ([[Vectorization]], [[NumPy]]).

## 🟠 3. Technical detail
- **Notation.** $A\in\mathbb{R}^{m\times n}$ has $m$ rows and $n$ columns. $A_{ij}$ is the number in row $i$, column $j$.
- **Product rule.** $(m\times n)\cdot(n\times p)=(m\times p)$. The middle $n$ must agree. Cost: $m\cdot n\cdot p$ multiplications.
- **Not commutative, but associative.** $AB\ne BA$ in general, while $(AB)C=A(BC)$. This is why a chain of layers can be thought of as one big matrix when nothing bends in between (see [[ReLU]] for why networks add a bend).
- **Transpose rule.** $(AB)^\top=B^\top A^\top$ (checked in the code on one example).
- **Size of a layer.** A layer from 512 numbers to 2048 numbers has $512\times2048=1{,}048{,}576$ weights, and one token costs the same number of multiplications.
- **Low-rank trick.** A big matrix can be written as the product of two thin ones, which has far fewer numbers. [[LoRA]] uses this idea for cheap fine-tuning.
- **Related ideas** not covered here: how a matrix stretches directions ([[Eigenvalue]]).
- **The code below** checks every rule above on small matrices and compares plain Python loops with NumPy.

```python
import time

import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]])                       # 2 rows, 3 columns: shape (2, 3)
x = np.array([1, 0, 2])                         # a vector with 3 numbers
print("shape of A:", A.shape, "| A @ x =", A @ x)
print("by hand: row 1 -> 1*1 + 2*0 + 3*2 =", 1 * 1 + 2 * 0 + 3 * 2, "| row 2 -> 4*1 + 5*0 + 6*2 =", 4 * 1 + 5 * 0 + 6 * 2)

B = np.array([[1, 0], [2, 1], [0, 3]])          # shape (3, 2)
print("(2,3) @ (3,2) gives shape", (A @ B).shape, "| (3,2) @ (2,3) gives shape", (B @ A).shape)
print("A @ B =", (A @ B).tolist())

P, Q = np.array([[1, 2], [3, 4]]), np.array([[0, 1], [1, 0]])
print("order matters: P @ Q =", (P @ Q).tolist(), "| Q @ P =", (Q @ P).tolist())
print("transpose of A has shape", A.T.shape, "| (A @ B).T == B.T @ A.T:", bool(((A @ B).T == B.T @ A.T).all()))

R = np.array([[0, -1], [1, 0]])                 # a matrix is also a move: this one turns arrows 90 degrees left
print("turn [1, 0] and [0, 2]:", (R @ np.array([1, 0])).tolist(), (R @ np.array([0, 2])).tolist())
print("turn twice = R @ R =", (R @ R).tolist(), "(that is 'flip every sign')")

d_in, d_out = 512, 2048                         # one layer of the original Transformer's feed-forward network
print("weights in a 512 -> 2048 layer:", d_in * d_out, "| multiplications for one token:", d_in * d_out)

n = 120
M, N = np.random.RandomState(0).rand(n, n), np.random.RandomState(1).rand(n, n)
t = time.perf_counter()
C = [[sum(M[i][k] * N[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
loop = time.perf_counter() - t
t = time.perf_counter()
C2 = M @ N
fast = time.perf_counter() - t
print(f"{n}x{n} times {n}x{n}: same result {np.allclose(C, C2)}; plain loops are {loop / fast:,.0f} times slower than numpy on this run")
```

What it printed: `A @ x` is `[7 16]`, matching the hand sums. $(2,3)\cdot(3,2)$ gives a $2\times2$ result `[[5, 11], [14, 23]]`, while the other order gives a $3\times3$. For $P=\begin{pmatrix}1&2\\3&4\end{pmatrix}$ and $Q=\begin{pmatrix}0&1\\1&0\end{pmatrix}$, $PQ=\begin{pmatrix}2&1\\4&3\end{pmatrix}$ and $QP=\begin{pmatrix}3&4\\1&2\end{pmatrix}$. The transpose rule was `True`. The turn matrix sent `[1, 0]` to `[0, 1]` and `[0, 2]` to `[-2, 0]`, and applying it twice gave minus the identity. The 512-to-2048 layer has 1,048,576 weights. The timing line differs from run to run: two runs said 305 and 191 times slower for plain loops (120 by 120 matrices; the timing depends on the machine).

## 🔴 4. Deep / formal
Multiplying a matrix $A$ ($m$ rows, $n$ columns) by a vector $x$ with $n$ numbers gives a vector $y$ with $m$ numbers:

$$
y_i=\sum_{j=1}^{n}A_{ij}\,x_j\qquad (i=1,\dots,m)
$$

**Where:**
- $A_{ij}$ is the number in row $i$ and column $j$ of $A$. $x_j$ is the $j$-th number of the vector $x$.
- $y_i$ is the $i$-th number of the answer: row $i$ of $A$ multiplied box by box with $x$ and added up (a [[Vector]] dot product).
- $m$ is the number of rows (the answer's length) and $n$ the number of columns (it must equal the length of $x$).
- **Matters:** the sum over $j$ is why the shape rule exists. If $x$ had a different length than $A$ has columns, some boxes would have no partner and the sum could not be written. Each answer number uses *all* of $x$, which is how one layer mixes all its inputs.

**Example (tiny numbers):** $A=\begin{pmatrix}1&2&3\\4&5&6\end{pmatrix}$ and $x=(1,0,2)$.
- $y_1=1\cdot1+2\cdot0+3\cdot2=1+0+6=\mathbf{7}$.
- $y_2=4\cdot1+5\cdot0+6\cdot2=4+0+12=\mathbf{16}$.
- Meaning: the three-number vector was turned into a two-number vector $(7,16)$. A matrix can change the length of a vector, not only its contents. The code printed the same `[7 16]`.

**Further notes (from the code above):**
- **Product of matrices.** The $(i,j)$ box of $AB$ is row $i$ of $A$ times column $j$ of $B$. Each of the $m\cdot p$ boxes costs $n$ multiplications, so $m\cdot n\cdot p$ in all.
- **Order.** The swapped product of $P$ and $Q$ above had the same numbers in a different arrangement (rows swapped, then columns swapped). This is one example, not a proof that they always differ.
- **What the code cannot show:** that the rules hold for all matrices (it checks single examples), anything about inverses, determinants or special matrices, which are not covered here.
- **A layer's size, again:** a 512-to-2048 layer holds 1,048,576 weights, and it costs the same number of multiplications per token. It is the first matrix of the feed-forward network in [[Feed-Forward Network]].

**What came before, and how this improved on it:**
- **Before:** solving puzzles with several unknowns by writing out each equation separately. Arrays of coefficients were used for this long before they had a name: the Chinese text *The Nine Chapters on the Mathematical Art* is given as the first example (10th to 2nd century BCE), Cardano brought the method to Europe in 1545 and Seki used it in 1683 (Wikipedia history section, read; a summary, not the original texts).
- **Its limit:** the array was only a bookkeeping device for one system of equations. There was no arithmetic of arrays on their own.
- **What changed:** Sylvester coined the word "matrix" in 1850 for the array, and Cayley then defined addition, multiplication and so on for matrices themselves and showed the order of multiplication matters, in work culminating in his 1858 memoir (same Wikipedia page; read). A whole system became one symbol, $Ax=b$.
- **How much better:** in this note's code, the 3-number-to-2-number machine is 6 numbers held in one symbol $A$. A 512-to-2048 layer is 1,048,576 numbers in one symbol $W$, with one product. As a way of computing, NumPy's matrix product was between 191 and 305 times faster than plain Python loops for 120 by 120 matrices on two runs on my machine (a machine-dependent timing, not a law).

## Related
- [[Vector]] · [[Neural Network]] · [[Embedding]] · [[Attention]] · [[Transformer]] · [[Feed-Forward Network]] · [[ReLU]] · [[Gradient Descent]] · [[LoRA]] · [[Eigenvalue]] · [[Vectorization]] · [[NumPy]]

## Sources
- Wikipedia, "Matrix (mathematics)", History section (read through the Wikipedia API): the dates for the Nine Chapters, Cardano 1545, Seki 1683, Sylvester 1850 and Cayley 1858 are as that page states them. A secondary source; the original works were not read.
- Everything else in this note comes from the code above, which was run.
- Image: none looked up for this note.

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/linear-algebra Matrix:::A table of numbers (rows go across, columns go down). It can also do something: it turns one vector into another.

#flashcards/math/formula #flashcards/topic/linear-algebra Write matrix times vector and explain each symbol.
?
$$y_i=\sum_{j=1}^{n}A_{ij}\,x_j\qquad (i=1,\dots,m)$$
- $A_{ij}$: the number in row $i$ and column $j$ of $A$. $x_j$: the $j$-th number of the vector $x$.
- $y_i$: the $i$-th number of the answer, row $i$ of $A$ multiplied box by box with $x$ and added up (a dot product).
- $m$: the number of rows (the answer's length). $n$: the number of columns, which must equal the length of $x$.
- The sum over $j$ is why the shape rule exists. If $x$ had a different length than $A$ has columns, some boxes would have no partner.

#flashcards/math/cloze #flashcards/topic/linear-algebra The length of $x$ must equal the number of ==columns== of $A$, and the answer has as many numbers as $A$ has ==rows==.

#flashcards/math/example #flashcards/topic/linear-algebra Compute $A x$ for $A=\begin{pmatrix}1&2&3\\4&5&6\end{pmatrix}$ and $x=(1,0,2)$.::y₁ = 1·1 + 2·0 + 3·2 = 7. y₂ = 4·1 + 5·0 + 6·2 = 16. A 3-number vector became the 2-number vector (7, 16): a matrix can change the length of a vector, not only its contents.
