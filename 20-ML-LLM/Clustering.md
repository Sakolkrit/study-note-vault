---
tags: [machine-learning, unsupervised-learning, clustering, k-means, DBSCAN, evaluation]
status: growing
level: 4
tier: 2
---

# Clustering

Clustering means **sorting things into groups by how alike they are, with nobody telling the computer what the groups are**.

## 🟢 1. Simple (age 12)
Imagine a huge box of mixed buttons and no labels. You start making piles: big round ones here, tiny flat ones there, shiny gold ones over there. Nobody told you the piles. You made them just by looking at which buttons look alike. A computer doing the same thing is **clustering**.

**Example:** A shop has 10,000 customers. A clustering program notices that some customers buy a little, often, and cheaply. Others buy rarely but spend a lot. The shop did not give it any names. It just finds the groups, and then the shop decides what to call them.

## 🔵 2. How it works
Every item is first turned into a list of numbers, a [[Vector]]. Then a method decides which vectors belong together. Three popular ways:
1. **k-means (centres):** pick how many groups you want, called $k$. Drop $k$ centres at random. Give each item to its nearest centre. Move each centre to the middle of its items. Repeat until nothing moves. (The full walk-through with the loss is in [[Unsupervised Learning]].)
2. **Hierarchical (family tree):** start with every item alone. Keep joining the two closest groups into one, again and again, until there is one big group. You get a tree, and you can cut it at any height to get as many groups as you like.
3. **DBSCAN (crowds):** a point in a crowded spot starts a group, and the group spreads to every nearby crowded point. Points left alone in empty space are called **noise**. You do not choose the number of groups.

![[clustering-kmeans-vs-spectral.png]]
*The same two interleaved half-moons grouped two ways (red and blue are the two groups found). Left: spectral clustering follows the curved shapes. Right: k-means cuts the data with a straight divide and mixes the colours. Image: BogdanShevchenko, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Spectral_clustering_vs_k_means.png), CC BY-SA 4.0.*

So the **shape of the groups** decides which method works. k-means likes round blobs. DBSCAN and spectral clustering cope with curves.

## 🟠 3. Technical detail
- **Type:** [[Unsupervised Learning]]. There are no labels, so there is no single "correct" answer to check against.
- **Distance decides everything:** "alike" means "close" by some distance. Features must be on a similar scale first (see [[Normalization]]), or the biggest numbers decide. Text and embeddings often use [[Cosine Similarity]].
- **Choosing $k$ for k-means:**
  - **Elbow:** run for $k=2,3,4,\dots$ and plot the total squared distance to the centres. It always falls as $k$ grows. Pick the $k$ where the fall suddenly gets slow (the "elbow").
  - **Silhouette score:** a number per point from $-1$ to $1$ that says how well it fits its own group compared with the next nearest group. Pick the $k$ with the best average.
- **k-means weak spots:** it assumes round, similar-sized groups, it is pulled around by outliers, and a bad random start can give a poor result (so run it several times, and use the "k-means++" start, from memory, which spreads the starting centres out).
- **DBSCAN knobs:** a radius `eps` and a minimum crowd size `min_samples`. It finds odd shapes and flags noise, but it struggles when some groups are much denser than others, and in very many dimensions, where distances all look alike (the same curse as in [[k-NN]]).
- **Not the same as k-NN:** [[k-NN]] uses labelled items to classify. Clustering has no labels and invents the groups.
- **Where it is used:** customer groups, grouping documents or images by their [[Embedding]] vectors, compressing colours, and spotting odd items that belong to no group. Grouping embeddings is also a common first step before building a [[Vector Database]] index.
- **Reducing first:** with many features, projecting down with [[PCA]] before clustering often helps.

```python
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_moons, make_blobs
from sklearn.metrics import silhouette_score

def agree(true, pred):                       # share of points labelled "right", best of the two label namings
    a = (true == pred).mean(); return round(max(a, 1 - a), 3)

# 1) round blobs: k-means is happy; the elbow and silhouette both point to k = 3
X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=0)
for k in range(2, 6):
    km = KMeans(k, n_init=10, random_state=0).fit(X)
    print(k, round(km.inertia_, 1), round(silhouette_score(X, km.labels_), 3))
# 2 655.3 0.553 | 3 212.4 0.66 | 4 182.8 0.541 | 5 158.3 0.443

# 2) two interleaved half-moons: k-means cuts them wrongly, DBSCAN follows the shape
M, t = make_moons(300, noise=0.05, random_state=0)
print("kmeans", agree(t, KMeans(2, n_init=10, random_state=0).fit_predict(M)))   # kmeans 0.747
print("dbscan", agree(t, DBSCAN(eps=0.2, min_samples=5).fit_predict(M)))         # dbscan 1.0
```

Needs scikit-learn. On the blobs, the squared-distance total drops hard from 2 to 3 groups (655 to 212) and then only a little (to 183), so the elbow is at 3, and the silhouette is also best at 3 (0.66). On the half-moons, k-means gets about 75% of the points into the right group, while DBSCAN gets all of them. This is a checkable demo of the picture above. The "agree" score uses the known moon labels only to grade the result, since the clustering itself never saw them.

## 🔴 4. Deep / formal
**The centre update in k-means.** Once the points are assigned to groups, each group's centre moves to the average of its members:

$$
\boldsymbol\mu_j=\frac{1}{|C_j|}\sum_{\mathbf x_i\in C_j}\mathbf x_i
$$

**Where:**
- $C_j$ is the set of points currently in group $j$, and $|C_j|$ is how many there are.
- $\mathbf x_i$ is one point (a list of numbers). $\boldsymbol\mu_j$ is the new centre of group $j$.
- The sum adds the points as lists, number by number, and dividing by $|C_j|$ gives the average.
- **Matters:** the average is exactly the spot that makes the **sum of squared distances** to the group as small as possible. That is why each k-means round can only lower the total or keep it the same. With a different "middle" (say the median) you would be doing a different method (k-medians).

**Example (tiny numbers):** a group of three points $(1,2)$, $(3,4)$ and $(5,0)$.
- Add the first numbers: $1+3+5=9$. Add the second numbers: $2+4+0=6$.
- Divide by $3$: $\boldsymbol\mu=(3,\,2)$.
- Check: the squared distances to an old centre $(2,2)$ add up to $1+5+13=19$. To the new centre $(3,2)$ they add up to $4+4+8=16$. The total fell from $\mathbf{19}$ to $\mathbf{16}$.
- Meaning: moving the centre to the average always shrinks (or keeps) the total, so k-means keeps improving until it settles. It settles at a *local* best, not always the global best.

**The silhouette of one point.** To judge a result without labels, compare each point's own group with the next-closest group:

$$
s(i)=\frac{b(i)-a(i)}{\max\{a(i),\,b(i)\}}
$$

**Where:**
- $a(i)$ is the average distance from point $i$ to the other points in its **own** group (small is good).
- $b(i)$ is the average distance from point $i$ to the points of the **nearest other** group (big is good).
- $\max\{a,b\}$ in the bottom scales the answer to lie between $-1$ and $1$.
- **Matters:** it turns "is this point in the right group?" into one number. Near $1$ means well placed, near $0$ means on the border, and negative means probably in the wrong group. Without the division, the number would depend on the units of the data.

**Example (tiny numbers):** a point has $a=1$ and $b=3$.
- Top: $b-a=3-1=2$. Bottom: $\max\{1,3\}=3$.
- $s=2/3=\mathbf{0.667}$.
- Meaning: it sits much closer to its own group than to the next one, so it is well placed. With the numbers swapped ($a=3$, $b=1$) the answer is $-0.667$, which says the point would fit better in the other group. With $a=b=2$ it is $0$, exactly on the border.

**Further notes:**
- **No ground truth:** two good methods can give different, equally sensible groups. The "right" answer depends on what you will use the groups for.
- **Silhouette likes round groups:** it scores compact blobs well, so it can score a correct DBSCAN result on curved shapes badly. Use it as a hint, not a verdict.
- **Soft clustering:** a Gaussian mixture model gives each point a *probability* for each group instead of one hard label. This is related to the ideas in [[Bayesian Methods]].
- **Hierarchical cost:** the plain method compares all pairs of points, so it gets slow quickly for big data (stated from memory).
- **Stability:** a sensible check is to re-run on resampled data and see if the groups stay roughly the same.

**What came before, and how this improved on it:**
- **Before:** group items by hand, for example by sorting on one measurement and cutting into equal bins.
- **Its limit:** groups that are clear in two measurements together can overlap in each one alone, and a person must pick the cut points.
- **What changed:** let an algorithm group by closeness in all the features at once (k-means has roots in the 1950s and 1960s; from memory).
- **How much better:** in the code below, four groups sit at the corners of a square. Cutting into 4 bins on one feature scores an adjusted Rand index of 0.330 (1 is perfect, 0 is chance) and k-means on both features 0.980. The groups here are cleanly round and well apart, which suits k-means; see [[DBSCAN]] and [[Gaussian Mixture Model]] for shapes it cannot handle.

**Check it yourself:**

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
rng = np.random.default_rng(0)
c = np.array([[0, 0], [3, 3], [0, 3], [3, 0]]); lab = np.repeat(np.arange(4), 100)
X = c[lab] + 0.5 * rng.standard_normal((400, 2))
cut = lambda v: np.digitize(v, np.quantile(v, [.25, .5, .75]))                # sort by one feature, 4 equal bins
print("grouping by one feature only: adjusted Rand", round(adjusted_rand_score(lab, cut(X[:, 0])), 3))
print("k-means on both features:     adjusted Rand", round(adjusted_rand_score(lab, KMeans(4, n_init=10, random_state=0).fit_predict(X)), 3))
```

## Related
- [[Unsupervised Learning]] · [[k-NN]] · [[Vector]] · [[Cosine Similarity]] · [[Normalization]] · [[PCA]] · [[Embedding]] · [[Vector Database]] · [[Bayesian Methods]] · [[Evaluation Metrics]] · [[Hierarchical Clustering]] · [[DBSCAN]] · [[Gaussian Mixture Model]] · [[Expectation-Maximization]] · [[K-Means]] · [[Machine Learning Specialization]]
- [[Local Outlier Factor]]
## Sources
- Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 2011 (from memory); the user guide chapter "Clustering" (free online)
- Rousseeuw, *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis*, 1987 (from memory)
- Ester, Kriegel, Sander & Xu, *A density-based algorithm for discovering clusters*, KDD 1996 (DBSCAN; from memory)
- Arthur & Vassilvitskii, *k-means++: the advantages of careful seeding*, 2007 (from memory)
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online), chapter 12

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/clustering Clustering:::Making piles of similar things with no labels, just by looking at which items look alike. Nobody told the computer the piles.

#flashcards/ml/formula #flashcards/topic/clustering Write the k-means centre update and explain each symbol.
?
$$\boldsymbol\mu_j=\frac{1}{|C_j|}\sum_{\mathbf x_i\in C_j}\mathbf x_i$$
- $C_j$: the set of points currently in group $j$. $|C_j|$: how many there are.
- $\mathbf x_i$: one point (a list of numbers). $\boldsymbol\mu_j$: the new centre of group $j$.
- The sum adds the points number by number, and dividing by $|C_j|$ gives the average.
- The average is exactly the spot that makes the squared distances to the group's points as small as possible.

#flashcards/ml/formula #flashcards/topic/clustering Write the silhouette score and explain each symbol.
?
$$s(i)=\frac{b(i)-a(i)}{\max\{a(i),\,b(i)\}}$$
- $a(i)$: the average distance from point $i$ to the other points in its own group (small is good).
- $b(i)$: the average distance from point $i$ to the points of the nearest other group (big is good).
- $\max\{a,b\}$: scales the answer to lie between −1 and 1.
- Near 1 means well placed, near 0 means on the border, negative means it would fit better in the other group.

#flashcards/ml/cloze #flashcards/topic/clustering K-means moves each centre to the ==average== of its points, and the silhouette compares ==own-group== distance with the ==nearest other== group.

#flashcards/ml/example #flashcards/topic/clustering A group has points $(1,2)$, $(3,4)$, $(5,0)$. What is the new centre, and how does the squared distance change from the old centre $(2,2)$?::(1+3+5)/3 = 3 and (2+4+0)/3 = 2, so μ = (3, 2). Squared distances to (2,2): 1 + 5 + 13 = 19. To (3,2): 4 + 4 + 8 = 16. The total fell.

#flashcards/ml/example #flashcards/topic/clustering A point has $a=1$ and $b=3$. What is the silhouette, and with the numbers swapped?::(3 − 1)/3 = 0.667, well placed. Swapped (a = 3, b = 1): −0.667, so it would fit better in the other group.

#flashcards/ml/image #flashcards/topic/clustering ![[clustering-kmeans-vs-spectral.png]]
What does this picture show?
?
The same two interleaved half-moons grouped two ways (red and blue are the two groups found). Left: spectral clustering follows the curved shapes. Right: k-means cuts the data with a straight divide and mixes the colours.
