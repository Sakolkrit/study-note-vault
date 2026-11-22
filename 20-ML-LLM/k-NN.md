---
tags: [machine-learning, supervised-learning, classification, k-NN, distance, instance-based]
status: growing
level: 4
tier: 2
---

# k-NN

k-NN stands for **k-Nearest Neighbours**.

## 🟢 1. Simple (age 12)
k-NN guesses about a new thing by **looking at the few things most like it and copying what most of them are**. It never really "learns" a rule. It just remembers everything and asks the neighbours when a question comes.

**Example:** You move to a new street and wonder if people there like dogs. You knock on the 3 nearest doors. Two have dogs, one has a cat. You guess "dog street". If you had only asked the single nearest door (the cat house), you would have guessed wrong. Asking a few neighbours is safer than asking one.

## 🔵 2. How it works
1. Keep every example you have, each with its answer (its label). This is your **training data**. (See [[Supervised Learning]].)
2. A new item arrives. Describe it as a list of numbers, a [[Vector]].
3. **Measure how far** it is from every stored example.
4. Pick the **k closest** ones. You choose k yourself, for example 3 or 5.
5. **Let them vote.** The label that most neighbours have is your answer. (For predicting a number, as in [[Regression]], take the average of their numbers.)

![[knn-classification.svg]]
*The green circle is the new item. Inside the solid circle (k = 3) there are 2 red triangles and 1 blue square, so the vote says triangle. Inside the dashed circle (k = 5) there are 2 triangles and 3 squares, so the vote says square. The answer depends on k. Image: Natana Jadri (Antti Ajanki), [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:KnnClassification.svg), CC BY-SA 3.0.*

## 🟠 3. Technical detail
- **Type:** [[Supervised Learning]]. Used for [[Classification]] (vote) and [[Regression]] (average).
- **"Lazy" learner:** there is no training step. All the work happens when you ask a question. That is cheap to set up, but each question compares against every stored example, which gets slow with lots of data.
- **Distance:** usually straight-line (Euclidean) distance. Other choices exist (Manhattan, [[Cosine Similarity]] for text and embeddings).
- **Choosing k:**
  - **Small k** (like 1): follows the data closely, so one odd or mislabelled neighbour can mislead it. Low bias, high variance (see [[Bias-Variance]]), prone to [[Overfitting]].
  - **Large k:** smoother, but ignores local detail, and at the extreme it just predicts the most common label overall.
  - Pick k with a validation set or cross-validation (see [[Evaluation Metrics]]). An odd k avoids ties in a two-class vote.
- **Scale matters a lot:** a feature in the tens of thousands (income) will swamp one in the tens (age). Rescale first (see [[Normalization]]).
- **Curse of dimensionality:** with many features, all points become about equally far apart, so "nearest" stops meaning much. k-NN works best with few, meaningful features. Reducing dimensions with [[PCA]] first can help.
- **Where it is used today:** as a quick baseline, and the idea of "find the nearest stored vectors" is exactly what a [[Vector Database]] does for search in [[RAG]]. Real systems use approximate nearest-neighbour search to stay fast.
- **Compared with a [[Decision Tree]]:** a tree compresses the data into a few rules. k-NN keeps all of it.

```python
import numpy as np
X = np.array([[1,1],[2,1],[1,2],[2,2],[4,4],[5,4],[4,5],[5,5],[3.5,3.5]], float)
y = np.array([0,0,0,0,1,1,1,1,0])          # the last point (3.5, 3.5) is a mislabelled "odd one"
q = np.array([3.4, 3.2])                   # the new item
d = np.sqrt(((X - q) ** 2).sum(1))         # distance to every stored point
order = np.argsort(d)                      # nearest first
print(d[order].round(3))                   # [0.316 1.    1.789 1.844 1.897 2.408 2.608 2.683 3.256]
for k in (1, 3, 5):
    votes = np.bincount(y[order[:k]], minlength=2)
    print(k, votes, "->", votes.argmax())  # 1 [1 0] -> 0 | 3 [1 2] -> 1 | 5 [2 3] -> 1

# scale: columns are (age, income)
P = np.array([[30, 40000], [60, 40100], [31, 45000]], float)
print(np.linalg.norm(P[0] - P[1]).round(1), np.linalg.norm(P[0] - P[2]).round(1))   # 104.4 5000.0
Z = (P - P.mean(0)) / P.std(0)             # rescale each column
print(np.linalg.norm(Z[0] - Z[1]).round(3), np.linalg.norm(Z[0] - Z[2]).round(3))   # 2.157 2.144
```

The last lines show why scaling matters. Raw distances say the first person is far closer to the 60-year-old (104) than to the 31-year-old (5000), only because income has the bigger numbers and a 30-year age gap barely registers. After rescaling, the two are almost equally far, and the order flips slightly.

## 🔴 4. Deep / formal
**Distance and the vote.** For a query $\mathbf q$, measure the distance to each stored example $\mathbf x_i$:

$$
d(\mathbf q,\mathbf x_i)=\sqrt{\sum_{j=1}^{m}\big(q_j-x_{ij}\big)^2}
$$

**Where:**
- $\mathbf q$ is the new item and $\mathbf x_i$ is the $i$-th stored example. Both are lists of $m$ numbers (features).
- $q_j$ and $x_{ij}$ are their values for feature number $j$. $\sum_{j=1}^{m}$ means "add up over all $m$ features".
- The square root of the sum of squared gaps is the ordinary straight-line distance (Pythagoras, extended to $m$ features).
- **Matters:** every feature counts by its size. If one feature has much bigger numbers, it decides almost everything. Without rescaling, the distance mostly measures the biggest-valued feature.

**Example (tiny numbers):** $\mathbf q=(3.4,\,3.2)$ and a stored point $(4,4)$.
- Gaps: $3.4-4=-0.6$ and $3.2-4=-0.8$.
- Squares: $0.36$ and $0.64$. Sum: $1.00$.
- Distance: $\sqrt{1.00}=\mathbf{1.0}$.
- Meaning: this point is exactly 1.0 away, and in the code it ranks second nearest (the mislabelled point at $(3.5,3.5)$, only 0.316 away, ranks first).

**The k-NN rule.** Let $N_k(\mathbf q)$ be the set of the $k$ stored examples with the smallest distance to $\mathbf q$. The predicted label is the most common label among them:

$$
\hat y(\mathbf q)=\arg\max_{c}\;\sum_{i\in N_k(\mathbf q)}\mathbf 1\!\left[y_i=c\right]
$$

**Where:**
- $\hat y$ ("y-hat") is the prediction. $c$ stands for a possible class label.
- $N_k(\mathbf q)$ is the set of the $k$ nearest neighbours. $y_i$ is the true label of neighbour $i$.
- $\mathbf 1[y_i=c]$ is 1 if neighbour $i$ has label $c$ and 0 otherwise, so the sum counts the votes for class $c$.
- $\arg\max_c$ means "the class $c$ with the biggest count".
- **Matters:** $k$ is the only knob. It sets how local the answer is. With $k=1$ one neighbour decides alone; with $k$ equal to all examples, every query gets the same answer.

**Example (tiny numbers):** the same query with $k=3$. The three nearest labels are $0, 1, 1$.
- Votes for class 0: $1$. Votes for class 1: $2$.
- $\arg\max$: class $\mathbf{1}$.
- Meaning: with $k=1$ the query is labelled 0 because the odd point is closest. With $k=3$ two honest neighbours outvote it. This matches the code output.

**Further notes:**
- **Weighted voting:** closer neighbours can count more (for example by weight $1/d$), so a far neighbour matters less.
- **Theory:** a classic result says that, with enough data, the error of 1-NN is at most about twice the best possible error (Cover and Hart, 1967; stated from memory). As the data grows and $k$ also grows (but slowly), k-NN can approach the best possible error.
- **Cost:** a query takes time proportional to the number of stored examples times the number of features, unless a search structure (a tree, or an approximate index) is used. Those structures help less in high dimensions.
- **Memory:** it stores the whole dataset, so it is poor when data is huge or private.
- **Not the same as k-means:** [[Unsupervised Learning]] k-means groups unlabelled points into $k$ clusters. k-NN uses labelled points to classify. The "k" means something different in each.

**What came before, and how this improved on it:**
- **Before:** the nearest-neighbour rule: copy the label of the single closest training example (Fix and Hodges, 1951, and Cover and Hart, 1967; from memory).
- **Its limit:** one noisy neighbour decides everything, so it fits the noise.
- **What changed:** let the $k$ closest neighbours vote.
- **How much better:** in the code below (two interleaved half-moons with noise), $k=1$ has training accuracy 1.000 and test accuracy 0.860. $k=15$ has 0.877 on training and 0.897 on test. Bigger $k$ (51) gives 0.890, so there is a sweet spot. The price of k-NN is that every prediction scans the stored data ([[Vector Database]]).

**Check it yourself:**

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier
X, y = make_moons(600, noise=0.35, random_state=0)
Xtr, Xte, ytr, yte = X[:300], X[300:], y[:300], y[300:]
for k in (1, 5, 15, 51):
    kn = KNeighborsClassifier(k).fit(Xtr, ytr)
    print(f"k={k:2d}  training accuracy {kn.score(Xtr, ytr):.3f}  test accuracy {kn.score(Xte, yte):.3f}")
```

## Related
- [[Supervised Learning]] · [[Classification]] · [[Regression]] · [[Vector]] · [[Cosine Similarity]] · [[Normalization]] · [[Bias-Variance]] · [[Overfitting]] · [[Evaluation Metrics]] · [[Vector Database]] · [[RAG]] · [[Embedding]] · [[PCA]] · [[Decision Tree]] · [[Support Vector Machine]] · [[Unsupervised Learning]] · [[Clustering]] · [[Hierarchical Clustering]] · [[DBSCAN]] · [[Gaussian Mixture Model]] · [[Big-O]] · [[Image Classification]]
- [[Collaborative Filtering]] · [[Local Outlier Factor]] · [[Anomaly Detection]] · [[Recommender Systems]]
## Sources
- Cover & Hart, *Nearest neighbor pattern classification*, IEEE Transactions on Information Theory, 1967 (from memory)
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, chapter 13 (from memory)
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online), chapter 2 (k-NN)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/classic-models k-NN (k nearest neighbours):::Guesses about a new thing by looking at the few things most like it and copying what most of them are. It never learns a rule, it just remembers everything and asks the neighbours.

#flashcards/ml/formula #flashcards/topic/classic-models Write the k-NN distance and explain each symbol.
?
$$d(\mathbf q,\mathbf x_i)=\sqrt{\sum_{j=1}^{m}\big(q_j-x_{ij}\big)^2}$$
- $\mathbf q$: the new item. $\mathbf x_i$: the $i$-th stored example. Both are lists of $m$ numbers (features).
- $q_j$, $x_{ij}$: their values for feature $j$. $\sum_{j=1}^m$: add up over all features.
- The square root of the sum of squared gaps is the ordinary straight-line distance (Pythagoras).
- Features on very different scales should be rescaled first, or the big one dominates the distance.

#flashcards/ml/formula #flashcards/topic/classic-models Write the k-NN prediction and explain each symbol.
?
$$\hat y(\mathbf q)=\arg\max_{c}\;\sum_{i\in N_k(\mathbf q)}\mathbf 1\!\left[y_i=c\right]$$
- $\hat y$: the prediction. $c$: a possible class label.
- $N_k(\mathbf q)$: the set of the $k$ nearest neighbours. $y_i$: the true label of neighbour $i$.
- $\mathbf 1[y_i=c]$: 1 if neighbour $i$ has label $c$, else 0. The sum counts the votes for class $c$.
- $\arg\max_c$: the class with the most votes.

#flashcards/ml/cloze #flashcards/topic/classic-models k-NN predicts by a ==vote== of the $k$ ==nearest== stored examples.

#flashcards/ml/example #flashcards/topic/classic-models $\mathbf q=(3.4,3.2)$ and a stored point $(4,4)$. What is the distance?::Gaps −0.6 and −0.8. Squares 0.36 and 0.64, sum 1.00. Distance √1.00 = 1.0.

#flashcards/ml/example #flashcards/topic/classic-models With $k=3$ the three nearest labels are 0, 1, 1. What is the prediction, and what with $k=1$?::Votes: class 0 gets 1, class 1 gets 2, so class 1. With k = 1 the nearest point is an odd mislabelled one, so the answer is 0. Two honest neighbours outvote it.

#flashcards/ml/image #flashcards/topic/classic-models ![[knn-classification.svg]]
What does this picture show?
?
The green circle is the new item. Inside the solid circle (k = 3) there are 2 red triangles and 1 blue square, so the vote says triangle. Inside the dashed circle (k = 5) there are 2 triangles and 3 squares, so the vote says square. The answer depends on k.
