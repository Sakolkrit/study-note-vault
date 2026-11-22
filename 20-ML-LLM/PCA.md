---
tags: [machine-learning, unsupervised-learning, dimensionality-reduction, PCA, linear-algebra]
status: growing
level: 4
tier: 2
---

# PCA

PCA stands for **Principal Component Analysis**.

## 🟢 1. Simple (age 12)
PCA is a way to **squash a lot of numbers about each thing into just a few, while keeping the most important differences**. Imagine each person in your class described by ten measurements. Most of those measurements move together (tall people have long arms and big feet). PCA finds the one or two "directions" that capture most of the differences, so you can describe everyone with two numbers instead of ten.

**Example:** You photograph a pencil lying on a table from above. The pencil is long in one direction and thin in the other. If you could only keep **one number** per dot of graphite, the best one is "how far along the pencil". That is the **main direction**, and PCA finds it automatically for any cloud of points.

## 🔵 2. How it works
1. Each item is a list of numbers, a [[Vector]]. Put all items in a table (a [[Matrix]]), one row per item.
2. **Centre the data:** subtract the average of each column, so the cloud is centred on zero.
3. Find the **direction in which the cloud is stretched the most**. That is the **first principal component**.
4. Find the next direction that is **at right angles** to the first and stretched the most. That is the second. Continue for as many directions as you have columns.
5. Each direction comes with a number saying **how much of the spread** it holds.
6. **Keep only the top few directions.** Describe each item by how far along each kept direction it sits. That is your short list of numbers.
7. You can map back to the original numbers, getting a close copy with small errors.

![[pca-gaussian-scatter.svg]]
*A cloud of points that is stretched along a diagonal. The two arrows start at the middle of the cloud and show the two principal directions: the long arrow points along the direction of most spread, the short one at right angles to it points along the direction of least spread. The arrow lengths reflect how much spread each direction has. Image: Nicoguaro, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:GaussianScatterPCA.svg), CC BY 4.0.*

## 🟠 3. Technical detail
- **Type:** [[Unsupervised Learning]]. It uses only the inputs, never any labels.
- **Common uses:**
  - **Compression:** keep the top few directions and throw the rest away.
  - **Visualisation:** squash data down to 2 numbers per item so you can draw it (for example, a map of [[Embedding Model]] vectors).
  - **Noise removal:** the small-spread directions are often mostly noise.
  - **Pre-processing:** fewer, uncorrelated inputs can help another model.
- **Steps with a computer:**
  1. Centre the columns (usually also scale them, see [[Normalization]]).
  2. Compute the **covariance matrix**, which records how each pair of features varies together.
  3. Find its **eigenvectors** (the directions) and **eigenvalues** (the spread along each). In practice libraries use the **SVD**, a closely related method that is more stable.
  4. Sort by eigenvalue, keep the top $k$, and project.
- **Choosing $k$:** keep enough directions to explain, say, 90 to 95% of the total spread (a rule of thumb, not a law), or look for an "elbow" in a plot of eigenvalues.
- **Limits:**
  - **Linear only:** it finds straight directions. A curved shape (a spiral, a ring) is not captured well. Nonlinear methods such as t-SNE and UMAP, or an autoencoder (a [[Neural Network]]), handle this.
  - **Scale matters:** a feature measured in thousands will dominate unless you rescale.
  - **High spread is not the same as importance:** the direction with most spread might be irrelevant to your goal.
  - **Hard to interpret:** each new axis is a mix of the old features.
- **Relation to other notes:** the "squeeze through a narrow middle and rebuild" idea is also how autoencoders work. PCA also appears when analysing the numbers inside LLMs, for example to draw how embeddings cluster (see [[Embedding]]).

```python
import numpy as np
X = np.array([[0., 0.], [2., 1.], [4., 3.], [6., 4.]])      # 4 points in 2-D, stretched along a diagonal
Xc = X - X.mean(0)                                          # centre: means are (3, 2)
C = Xc.T @ Xc / (len(X) - 1)                                # covariance matrix
print(C.round(3).tolist())                                  # `[[6.667, 4.667], [4.667, 3.333]]`
vals, vecs = np.linalg.eigh(C)                              # eigenvalues come out smallest first
vals, vecs = vals[::-1], vecs[:, ::-1]                      # biggest first
v = vecs[:, 0] * np.sign(vecs[0, 0])                        # fix the arbitrary sign of the direction
print(vals.round(3), v.round(3), (vals / vals.sum()).round(3))   # [9.955 0.045] [0.817 0.576] [0.996 0.004]
z = Xc @ v                                                  # ONE number per point instead of two
print(z.round(3))                                           # [-3.604 -1.393  1.393  3.604]
back = np.outer(z, v) + X.mean(0)                           # rebuild approximate 2-D points
print(back.round(2).tolist(), ((X - back) ** 2).sum().round(3))
# [[0.05, -0.08], [1.86, 1.2], [4.14, 2.8], [5.95, 4.08]] 0.134   close to the originals
```

## 🔴 4. Deep / formal
**Directions of spread: the covariance matrix and its eigenvectors.** Let $X_c$ be the centred data with $n$ rows (items) and $d$ columns (features). The covariance matrix and the equation that defines the principal directions are:

$$
C=\frac{1}{n-1}X_c^{\top}X_c,\qquad C\,\mathbf v_i=\lambda_i\,\mathbf v_i
$$

**Where:**
- $X_c$ is the data after subtracting each column's average. $X_c^{\top}$ is its transpose (rows and columns swapped, see [[Matrix]]). $n$ is the number of items.
- $C$ is the $d\times d$ **covariance matrix**. The entry in row $i$, column $j$ says how features $i$ and $j$ vary together: large positive if they rise together, near 0 if unrelated. The diagonal entries are the spread of each feature on its own. **Matters:** dividing by $n-1$ (not $n$) is the standard correction when the average itself was estimated from the same data.
- $\mathbf v_i$ is an **eigenvector** of $C$: a direction that $C$ only stretches, never turns. $\lambda_i$ ("lambda") is the matching **eigenvalue**, the stretch factor.
- **Matters:** the eigenvector with the biggest $\lambda$ is the direction of most spread (the first principal component). For a covariance matrix the eigenvectors are at right angles to each other, and each $\lambda_i$ is exactly the variance of the data measured along $\mathbf v_i$. Without this, there would be no principled way to rank directions.

**Example (tiny numbers):** four points $(0,0),(2,1),(4,3),(6,4)$.
- Averages: $(3,2)$. Centred points: $(-3,-2),(-1,-1),(1,1),(3,2)$.
- $C=\frac13\begin{pmatrix}20&14\\14&10\end{pmatrix}=\begin{pmatrix}6.667&4.667\\4.667&3.333\end{pmatrix}$.
- The eigenvalues of a 2×2 matrix come from its trace (sum of the diagonal) and determinant: trace $=10$, determinant $=6.667\cdot3.333-4.667^2=0.444$. So $\lambda=\tfrac{10\pm\sqrt{100-4\cdot0.444}}{2}=\mathbf{9.955}$ and $\mathbf{0.045}$.
- The first direction is $\mathbf v_1\approx(0.817,\,0.576)$: it runs along the diagonal the points are stretched along.
- Meaning: nearly all the spread (9.955 out of 10) lies along one direction, so one number per point is almost enough.

**How much did we keep? Explained variance.** The share of the total spread held by the top $k$ directions:

$$
\text{explained}=\frac{\sum_{i=1}^{k}\lambda_i}{\sum_{i=1}^{d}\lambda_i}
$$

**Where:**
- $\lambda_i$ are the eigenvalues sorted from largest to smallest, $k$ is how many directions you keep, and $d$ is the total number of directions (the number of features).
- The top sum is the spread you keep. The bottom sum is the total spread, which also equals the sum of the diagonal of $C$.
- **Matters:** it is a number from 0 to 1 telling you how much information (in the sense of spread) survives the squashing. It is also what you use to choose $k$. The squared error left over after rebuilding the data is, when summed over all items, $(n-1)$ times the sum of the discarded $\lambda_i$.

**Example (tiny numbers):** the eigenvalues above, $9.955$ and $0.045$, keep $k=1$.
- Total spread: $9.955+0.045=\mathbf{10}$.
- Explained: $9.955/10=\mathbf{0.9955}$, about 99.6%.
- Check against the code: the total squared error after rebuilding is $3\times0.045=\mathbf{0.134}$, matching the printed $0.134$.
- Meaning: going from two numbers per point to one keeps 99.6% of the spread. The rebuilt points in the code are close to the originals.

**Further notes:**
- **Another view:** PCA finds the straight line (or flat surface) that is closest to all the points. Choosing the direction of most spread and choosing the one with least rebuild error are the same problem. (A standard result, stated from memory.)
- **SVD:** if $X_c=U\Sigma V^{\top}$, the columns of $V$ are the principal directions and $\lambda_i=\sigma_i^2/(n-1)$. This avoids forming $C$, which matters when $d$ is large.
- **Whitening:** dividing each kept direction by $\sqrt{\lambda_i}$ gives uncorrelated features with equal spread.
- **The sign is arbitrary:** $\mathbf v$ and $-\mathbf v$ are equally valid. That is why the code fixes the sign.
- **Not the same as clustering:** PCA reduces the number of features. [[Unsupervised Learning]] clustering (such as k-means) groups the items. They are often used together.

**What came before, and how this improved on it:**
- **Before:** pick a few of the original features by hand (or by their variance) and drop the rest.
- **Its limit:** information is spread over many features, so dropping columns throws a lot away.
- **What changed:** rotate to new axes that line up with the directions of largest spread, then keep the first few (Pearson 1901 and Hotelling 1933 are usually credited; from memory).
- **How much better:** in the code below (handwritten digits, 64 pixels), the 2 pixels with the most variance hold 7.0% of the total variance. The top 2 principal components hold 28.5%, about 4 times more. It still takes 21 components to keep 90% of the variance, so 2 numbers per picture is a coarse summary.

**Check it yourself:**

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
X, y = load_digits(return_X_y=True)
v = X.var(0)
print("keep the 2 pixels with the most variance:", round(np.sort(v)[-2:].sum() / v.sum(), 3), "of the total variance")
p = PCA().fit(X); cum = np.cumsum(p.explained_variance_ratio_)
print("keep 2 principal components:", round(cum[1], 3), "of the total variance")
print("components needed for 90% of the variance:", int(np.searchsorted(cum, 0.9)) + 1, "of 64")
```

## Related
- [[Unsupervised Learning]] · [[Vector]] · [[Matrix]] · [[Embedding]] · [[Embedding Model]] · [[Normalization]] · [[Normal Distribution]] · [[Neural Network]] · [[NumPy]] · [[Cosine Similarity]] · [[Generative Models]] · [[Clustering]] · [[Hierarchical Clustering]] · [[DBSCAN]] · [[Gaussian Mixture Model]] · [[Eigenvalue]] · [[Singular Value Decomposition]] · [[Autoencoder]] · [[Variational Autoencoder]] · [[Machine Learning Specialization]]
- [[Local Outlier Factor]]
## Sources
- Jolliffe, *Principal Component Analysis*, 2nd edition, 2002 (from memory)
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online), chapter 12
- Strang, *Introduction to Linear Algebra* (eigenvalues, SVD)
- Shlens, *A Tutorial on Principal Component Analysis* (free online; from memory)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/classic-models PCA (principal component analysis):::Squashes many numbers per thing into a few while keeping the most important differences. It finds the one or two directions that capture most of the spread in the data.

#flashcards/ml/formula #flashcards/topic/classic-models Write the PCA covariance matrix and its eigen equation, and explain each symbol.
?
$$C=\frac{1}{n-1}X_c^{\top}X_c,\qquad C\,\mathbf v_i=\lambda_i\,\mathbf v_i$$
- $X_c$: the data after subtracting each column's average. $X_c^\top$: its transpose. $n$: the number of items.
- $C$: the $d\times d$ covariance matrix. Row $i$, column $j$ says how features $i$ and $j$ vary together.
- $\mathbf v_i$: an eigenvector, a principal direction. $\lambda_i$: its eigenvalue, how much spread lies along that direction.
- Sorting by $\lambda_i$ from large to small gives the directions in order of importance.

#flashcards/ml/formula #flashcards/topic/classic-models Write the explained variance and explain each symbol.
?
$$\text{explained}=\frac{\sum_{i=1}^{k}\lambda_i}{\sum_{i=1}^{d}\lambda_i}$$
- $\lambda_i$: the eigenvalues sorted largest first. $k$: how many directions you keep. $d$: the total number of directions (the number of features).
- Top sum: the spread you keep. Bottom sum: the total spread, which also equals the sum of the diagonal of $C$.
- It is a number from 0 to 1. Close to 1 means little information is lost by keeping only $k$ directions.

#flashcards/ml/cloze #flashcards/topic/classic-models PCA keeps the directions with the largest ==eigenvalues==, which are the directions of most ==spread==.

#flashcards/ml/example #flashcards/topic/classic-models Points $(0,0),(2,1),(4,3),(6,4)$. What are the centred points, $C$ and its eigenvalues?::Averages (3, 2). Centred: (−3,−2), (−1,−1), (1,1), (3,2). C = `[[6.667, 4.667], [4.667, 3.333]]`. Trace 10, so the eigenvalues are 9.955 and 0.045.

#flashcards/ml/example #flashcards/topic/classic-models For those eigenvalues (9.955 and 0.045), what fraction is explained with $k=1$, and what is the rebuild error?::Total 10. Explained 9.955/10 = 0.9955, about 99.6%. Total squared error after rebuilding: 3 × 0.045 = 0.134, matching the code.

#flashcards/ml/image #flashcards/topic/classic-models ![[pca-gaussian-scatter.svg]]
What does this picture show?
?
A cloud of points stretched along a diagonal. The two arrows start at the middle of the cloud and show the two principal directions: the long arrow points along the most spread, the short one at right angles along the least. The arrow lengths reflect how much spread each direction has.
