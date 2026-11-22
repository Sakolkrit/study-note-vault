---
tags: [machine-learning, unsupervised-learning, clustering, k-means, AI]
status: growing
level: 4
tier: 1
---

# Unsupervised Learning

## 🟢 1. Simple (age 12)
Unsupervised learning is **finding patterns with no answer key**. You give the computer a big pile of data and nobody tells it what anything is. It has to look for things that are **alike** and group them, or find the main **ways the data differs**, all by itself.

**Example:** You pour a jar of mixed buttons onto the table and ask a friend to sort them. You do not say how. Your friend notices some are big and some small, some shiny and some dull, and makes piles that look similar. Nobody told them the "right" piles. They found a sensible grouping from the buttons themselves. That is what an unsupervised model does.

## 🔵 2. How it works
1. Collect lots of data, but **without labels**: photos with no captions, customers with no categories.
2. Turn each item into a list of numbers (a [[Vector]]).
3. Choose a way to measure **how alike** two items are, usually how close their vectors are (see [[Cosine Similarity]]).
4. Let the algorithm look for structure. The most common jobs are:
   - **Clustering:** put similar items in the same group.
   - **Dimensionality reduction:** squeeze many numbers per item into just a few, keeping what matters (handy for drawing a 2D map of the data).
   - **Anomaly detection:** spot the odd item that does not fit anywhere.
   - **Learning representations:** learn good lists of numbers to describe things ([[Embedding Model]]).
5. A human then looks at the result and decides if it is useful. There is no answer key to check it against, so judging the result takes care.

**A famous method, k-means,** works like this: choose how many groups you want ($k$), drop $k$ "centres" at random, then repeat two steps. First, give every item to its nearest centre. Second, move each centre to the middle of the items it just got. Stop when nothing changes.

![[kmeans-step1.svg]]
*Start of k-means with $k=3$: the grey squares are the data points, and the three coloured circles (red, green, blue) are the starting centres, dropped at random. Image: Weston.pace, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:K_Means_Example_Step_1.svg), CC BY-SA 3.0.*

![[kmeans-step4.svg]]
*After a few rounds: each point has been coloured by its nearest centre, the shaded regions show which centre "owns" which part of the plane, and each centre has moved into the middle of its group. Image: Weston.pace, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:K_Means_Example_Step_4.svg), CC BY-SA 3.0.*

## 🟠 3. Technical detail
- **Data:** only inputs $x$, no labels $y$. Compare with [[Supervised Learning]], where every example comes with an answer.
- **Clustering methods:**
  - **k-means:** fast and simple, finds round, similar-sized groups, and you must choose $k$.
  - **Hierarchical clustering:** builds a tree of groups inside groups.
  - **DBSCAN** and density-based methods: find groups of any shape and mark isolated points as noise.
  - **Gaussian mixture models:** treat each group as a bell-shaped blob ([[Normal Distribution]]) and give each point a chance of belonging to each.
- **Dimensionality reduction:** PCA (finds the directions along which the data varies most, using [[Matrix]] algebra), t-SNE and UMAP (for drawing 2D maps of high-dimensional data).
- **Neural methods:** autoencoders (squeeze the input through a narrow middle, then rebuild it), and generative models that learn what the data looks like and can create new examples.
- **Choosing $k$:** try several values and watch the total within-group distance ("elbow" method), or use a score such as silhouette. There is rarely one "correct" $k$.
- **Practical cautions:**
  - **Scale the features:** a feature measured in thousands will swamp one measured in decimals, so normalise first (see [[Normalization]] for the idea).
  - **Starting points matter:** k-means can end in different answers from different random starts, so run it several times.
  - **Outliers** can drag a centre away.
  - **Clusters are not guaranteed to mean anything:** the algorithm will happily draw groups in pure noise.
- **Common uses:** grouping customers, organising documents or images by topic, compressing data, finding fraud or faults, and exploring a new dataset.
- **Relation to LLMs:** an [[LLM]] is first trained on raw text with no human labels, using **self-supervised** learning (see [[Self-Supervised Learning]]): the data supplies its own answers (the next word). That is why it is often loosely called "unsupervised".

```python
import numpy as np
pts = np.array([1., 2., 9., 10.])               # 1-D data: two obvious groups
c = np.array([1., 5.])                          # k = 2 starting centres
for it in range(3):
    a = np.argmin(abs(pts[:, None] - c[None, :]), axis=1)     # assign each point to nearest centre
    c = np.array([pts[a == k].mean() for k in range(2)])      # move centres to group means
    print(it, a, c, ((pts - c[a]) ** 2).sum())
# 0 [0 0 1 1] [1.5 9.5] 1.0
# 1 [0 0 1 1] [1.5 9.5] 1.0     nothing changed, so it has converged
# 2 [0 0 1 1] [1.5 9.5] 1.0
```

## 🔴 4. Deep / formal
**The k-means objective.** Suppose the data are split into $k$ groups $C_1,\dots,C_k$ with group centres $\mu_1,\dots,\mu_k$. K-means looks for the split and centres that make this total as small as possible:

$$
J=\sum_{j=1}^{k}\ \sum_{x\in C_j}\ \lVert x-\mu_j\rVert^{2}
$$

**Where:**
- $k$ is the number of groups you chose. $C_j$ is the set of data points assigned to group $j$. $\mu_j$ ("mu") is the centre of group $j$.
- $x\in C_j$ means "each point $x$ in group $j$". $\lVert x-\mu_j\rVert$ is the distance between the point and its group's centre, and squaring it makes far-away points count much more.
- The inner sum adds up the squared distances inside one group, and the outer sum adds over all groups. **Matters:** $J$ is small when every point sits close to its own centre, that is, when the groups are tight. It is the loss that k-means lowers, with no labels needed.
- **Matters:** because the distances are squared, one far-off outlier can pull a centre a long way, so k-means is sensitive to outliers.
- Each round of the algorithm cannot increase $J$, so it settles down. It may settle on a local minimum rather than the best possible split.

**Example (tiny numbers):** points $1,\,2,\,9,\,10$ with $k=2$.
- Good split: $C_1=\{1,2\}$ with $\mu_1=1.5$ and $C_2=\{9,10\}$ with $\mu_2=9.5$. Squared distances: $0.25+0.25+0.25+0.25$. $J=\mathbf{1.0}$.
- Bad split: $C_1=\{1\}$ with $\mu_1=1$ and $C_2=\{2,9,10\}$ with $\mu_2=7$. Squared distances: $0$, then $(2-7)^2+(9-7)^2+(10-7)^2=25+4+9=38$. $J=\mathbf{38}$.
- Meaning: the natural split has a far smaller total, so k-means prefers it. Nobody told it that "1 and 2 belong together". It found that from the numbers alone.

**The two repeating steps.** Starting from rough centres, repeat (1) assign each point to its nearest centre, then (2) move each centre to the average of the points it got:

$$
\mu_j\leftarrow\frac{1}{\lvert C_j\rvert}\sum_{x\in C_j}x
$$

**Where:**
- $\mu_j\leftarrow$ means "set the new centre to". $C_j$ is the set of points currently assigned to centre $j$, and $\lvert C_j\rvert$ is how many points that is.
- $\sum_{x\in C_j}x$ adds up all those points, and dividing by $\lvert C_j\rvert$ gives their **average**. **Matters:** the average is exactly the spot that minimises the total squared distance to the points in the group, so this step can only lower $J$ (or leave it unchanged).
- Step (1), reassigning points to the nearest centre, also cannot raise $J$. Alternating the two steps drives $J$ down until nothing changes.

**Example (tiny numbers):** same points $1,2,9,10$, but start with centres $\mu_1=1$ and $\mu_2=5$.
- Assign: point $1$ is nearer to $1$ (distance 0 against 4). Point $2$ is nearer to $1$ (distance 1 against 3). Points $9$ and $10$ are nearer to $5$ (distance 4 and 5 against 8 and 9). So $C_1=\{1,2\}$ and $C_2=\{9,10\}$.
- Update: $\mu_1=(1+2)/2=\mathbf{1.5}$ and $\mu_2=(9+10)/2=\mathbf{9.5}$. Now $J=1.0$.
- Assign again: nothing changes, so it has converged (the printed output above shows it).
- Meaning: two quick rounds recovered the obvious grouping without any labels.

**Further notes:**
- **No ground truth:** quality is judged by internal scores (such as $J$ or silhouette), by whether the groups are stable and useful, or by checking against labels if some exist. A lower $J$ with more groups is not automatically better, since $k=n$ gives $J=0$.
- **Why PCA works:** it finds the directions that capture the most variance, using the eigenvectors of the data's covariance [[Matrix]]. Keeping only the top few directions compresses the data with the least loss. (Standard result, stated from memory.)
- **Curse of dimensionality:** in very many dimensions, all points look about equally far apart, which weakens distance-based methods. Reducing dimensions first often helps.
- **Semi-supervised and self-supervised:** in practice the line between supervised and unsupervised is blurry. Many modern systems train on unlabelled data first and use a few labels later ([[Self-Supervised Learning]], [[Fine-tuning]]).
- **Generative models** (which learn the data distribution well enough to produce new samples) are also learned mostly without labels. This is an active area.

**What came before, and how this improved on it:**
- **Before:** every useful pattern had to be taught with labels, and labels cost human time.
- **Its limit:** most data in the world has no labels.
- **What changed:** find structure in the data itself, such as clusters (k-means goes back to the 1950s and 1960s; from memory).
- **How much better:** in the code below, k-means sees 1,797 digit pictures with no labels. The pictures in each cluster share one majority digit 79.4% of the time (random clusters: 13.5%), and the adjusted Rand index is 0.669 (random: 0.001). It found digit groups with 0 labels, but it did not name them, and about one picture in five is not the majority digit of its cluster.

**Check it yourself:**

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
X, y = load_digits(return_X_y=True)
lab = KMeans(10, n_init=10, random_state=0).fit_predict(X)             # no labels used
pur = sum(np.bincount(y[lab == k]).max() for k in range(10)) / len(y)
rnd = np.random.default_rng(0).integers(0, 10, len(y))
print("labels used: 0.  Share of pictures in the cluster's majority digit:", round(pur, 3), " (random clusters:", round(sum(np.bincount(y[rnd == k]).max() for k in range(10)) / len(y), 3), ")")
print("adjusted Rand index (1 perfect, 0 chance): k-means", round(adjusted_rand_score(y, lab), 3), " random", round(adjusted_rand_score(y, rnd), 3))
```

## Related
- [[Supervised Learning]] · [[Self-Supervised Learning]] · [[Classification]] · [[Regression]] · [[Embedding Model]] · [[Cosine Similarity]] · [[Vector]] · [[Matrix]] · [[Normal Distribution]] · [[Normalization]] · [[LLM]] · [[Reinforcement Learning]] · [[PCA]] · [[Generative Models]] · [[Clustering]] · [[Hierarchical Clustering]] · [[DBSCAN]] · [[Gaussian Mixture Model]] · [[Expectation-Maximization]] · [[Autoencoder]] · [[K-Means]]

## Sources
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online), chapter 12
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (free online), chapter 14
- Lloyd, *Least squares quantization in PCM* (the k-means algorithm, written 1957, published 1982; from memory)
- Goodfellow, Bengio & Courville, *Deep Learning*, 2016, chapters 13 to 15

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/ml-foundations Unsupervised learning:::Finding patterns with no answer key. The computer groups things that are alike, or finds the main ways the data differs, by itself.

#flashcards/ml/formula #flashcards/topic/ml-foundations Write the k-means objective and explain each symbol.
?
$$J=\sum_{j=1}^{k}\ \sum_{x\in C_j}\ \lVert x-\mu_j\rVert^{2}$$
- $k$: the number of groups you chose. $C_j$: the set of points assigned to group $j$. $\mu_j$: the centre of group $j$.
- $x\in C_j$: each point in group $j$. $\lVert x-\mu_j\rVert$: the distance from the point to its group's centre. Squaring makes far-away points count much more.
- The inner sum adds the squared distances inside one group, the outer sum adds over all groups.
- $J$ is small when every point sits close to its own centre, that is, when the groups are tight.

#flashcards/ml/example #flashcards/topic/ml-foundations Points 1, 2, 9, 10 with $k=2$. What is $J$ for the split $\{1,2\},\{9,10\}$ and for $\{1\},\{2,9,10\}$?::Good split: $\mu=1.5$ and $9.5$, four squared distances of 0.25, $J=1.0$. Bad split: $\mu=1$ and $7$, $0$ then $25+4+9=38$, $J=38$. K-means prefers the natural split, found from the numbers alone.

#flashcards/ml/formula #flashcards/topic/ml-foundations Write the k-means centre update and explain each symbol.
?
$$\mu_j\leftarrow\frac{1}{\lvert C_j\rvert}\sum_{x\in C_j}x$$
- $\mu_j\leftarrow$: set the new centre to. $C_j$: the points currently assigned to centre $j$. $\lvert C_j\rvert$: how many there are.
- $\sum_{x\in C_j}x$ adds the points, and dividing gives their average.
- The average is exactly the spot that minimises the total squared distance to the group, so this step can only lower $J$ or leave it unchanged. Reassigning points to the nearest centre also cannot raise $J$. Alternating the two drives $J$ down.

#flashcards/ml/example #flashcards/topic/ml-foundations Points 1, 2, 9, 10, start centres 1 and 5. What happens?::Assign: 1 and 2 go to centre 1, 9 and 10 to centre 5. Update: $\mu_1=1.5$, $\mu_2=9.5$, $J=1.0$. Assigning again changes nothing, so it has converged in two rounds without any labels.

#flashcards/ml/image #flashcards/topic/ml-foundations ![[kmeans-step1.svg]]
What does this picture show?
?
Start of k-means with $k=3$: the grey squares are the data points and the three coloured circles (red, green, blue) are the starting centres, dropped at random. (From the note's caption: Weston.pace, Wikimedia Commons.)

#flashcards/ml/image #flashcards/topic/ml-foundations ![[kmeans-step4.svg]]
What does this picture show?
?
After a few rounds: each point is coloured by its nearest centre, the shaded regions show which centre owns which part of the plane, and each centre has moved into the middle of its group. (From the note's caption: Weston.pace, Wikimedia Commons.)

#flashcards/ml/cloze #flashcards/topic/ml-foundations Unsupervised learning has ==no answer key==. The model finds groups or main directions of difference on its own.
