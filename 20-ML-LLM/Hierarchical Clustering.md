---
tags: [machine-learning, unsupervised-learning, clustering, hierarchical, dendrogram]
status: growing
level: 4
tier: 2
---

# Hierarchical Clustering

Hierarchical clustering builds a **family tree of groups**: it starts with every item alone and keeps joining the two closest groups until only one big group is left.

## 🟢 1. Simple (age 12)
Picture a school photo day. At first everyone stands alone. Then the two kids standing closest hold hands. Then the next closest pair does. Then a pair joins a pair, then two bigger groups merge, until the whole school is one big crowd. If you write down **who joined whom, and when**, you get a tree. You can then pick any moment in the story to say "these are the groups".

**Example:** Five animals: cat, lion, dog, wolf, goldfish. Cat and lion are most alike, so they join first. Dog and wolf join next. Those two pairs join into "land mammals". Finally the goldfish joins last, because it is least like everything else.

## 🔵 2. How it works
This is the common **bottom-up** version (called agglomerative), a form of [[Clustering]] in [[Unsupervised Learning]].
1. Turn each item into a [[Vector]] and measure the distance between every pair of items.
2. Start with every item as its own group.
3. Find the **two closest groups** and merge them into one. Note the distance at which they merged.
4. Work out the distance from the new group to every other group. (You must decide how, see "linkage" below.)
5. Repeat steps 3 and 4 until one group is left.
6. Draw the whole history as a tree called a **dendrogram**. To get $k$ groups, cut the tree at the height where $k$ branches hang down.

![[dendrogram-upgma.svg]]
*Hierarchical clustering of five items a to e. The pair a, b merges first (height 8.5), then e joins them (11.0), then c and d merge (14.0), and finally the two groups merge into one (16.5). Image: Emmanuel Douzery, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:UPGMA_Dendrogram_Hierarchical.svg), CC BY-SA 4.0.*

Unlike k-means (see [[Clustering]]), you do not choose the number of groups first. You choose it afterwards by where you cut.

## 🟠 3. Technical detail
- **Linkage = how to measure the distance between two groups.** This choice changes the result a lot:
  - **Single:** the distance between the two *closest* members. It can chain groups together into long strings.
  - **Complete:** the distance between the two *farthest* members. It makes compact groups.
  - **Average:** the average over all cross pairs (the example below). A middle road.
  - **Ward:** merge the pair that increases the total squared spread the least. It tends to give round, even groups, and is related to the k-means loss.
- **Cost:** it needs a distance between all pairs of items, so the memory grows with the square of the number of items, and the time at least as fast. This is fine for thousands of items but not for millions (stated from memory; exact costs depend on the method).
- **No undo:** once two groups merge, they never split again, so an early bad merge stays.
- **Scale and distance:** as in all clustering, features should be on a similar scale first (see [[Normalization]]), and the distance you pick (straight line, [[Cosine Similarity]]) matters.
- **Good for:** small data where you want to *see* the structure, such as gene data, documents, or how a set of [[Embedding]] vectors group at several levels of detail.
- **Compared with k-means:** it gives a whole nested set of answers instead of one, and it is deterministic (no random start), but it is slower and less flexible than [[Clustering]] by centres.

```python
import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
D = np.array([[0,17,21,31,23],      # distances between items a b c d e
              [17,0,30,34,21],
              [21,30,0,28,39],
              [31,34,28,0,43],
              [23,21,39,43,0]], float)
Z = linkage(squareform(D), "average")
print(Z[:, :3].round(1).tolist())   # [[0.0, 1.0, 17.0], [4.0, 5.0, 22.0], [2.0, 3.0, 28.0], [6.0, 7.0, 33.0]]
print((Z[:, 2] / 2).tolist())       # [8.5, 11.0, 14.0, 16.5]   the heights drawn in the picture above
print((21 + 23) / 2, (21 + 31 + 30 + 34 + 39 + 43) / 6)   # 22.0 33.0  same numbers by hand
```

Each row of `Z` says "merge item/group $i$ with item/group $j$ at distance $d$". Items 0 to 4 are a to e, and new groups are numbered 5, 6, ... So the first row merges a and b (17), the second joins e (4) to the group {a,b} (5) at 22, the third merges c and d at 28, and the last merges the two groups at 33. The picture draws the heights as half the distance (8.5, 11.0, 14.0, 16.5), which is the convention of the UPGMA method used there, and my output matches those four numbers. The distance table is my own reconstruction chosen so that it reproduces the picture's heights, so it is a consistent example, not necessarily the exact data of the original figure.

## 🔴 4. Deep / formal
**Average linkage.** The distance between two groups $A$ and $B$ is the average of the distances between every member of $A$ and every member of $B$:

$$
d(A,B)=\frac{1}{|A|\,|B|}\sum_{a\in A}\sum_{b\in B} d(a,b)
$$

**Where:**
- $A$ and $B$ are two groups of items. $|A|$ and $|B|$ are how many items each has.
- $a$ and $b$ are single items, one from each group, and $d(a,b)$ is the distance between that pair.
- The double sum adds up the distance of every cross pair, so there are $|A|\,|B|$ pairs. Dividing by $|A||B|$ makes it an average.
- **Matters:** it uses *every* pair, so one stray item cannot decide the answer alone, unlike single linkage (closest pair only) or complete linkage (farthest pair only). Without the division, big groups would look far away just because they have more members.

**Example (tiny numbers):** group $A=\{a,b\}$ and group $B=\{e\}$, with $d(a,e)=23$ and $d(b,e)=21$.
- Number of cross pairs: $2\times1=2$.
- Sum: $23+21=44$.
- $d(A,B)=44/2=\mathbf{22}$.
- Meaning: the group {a,b} is 22 away from e on average. In the figure that merge is drawn at half of 22, which is 11.0. A bigger example: {a,b,e} against {c,d} has $3\times2=6$ pairs, whose distances add up to $21+31+30+34+39+43=198$, so the average is $198/6=33$, and the figure draws it at 16.5.

**Further notes:**
- **Updating cheaply:** after a merge, the new distances can be computed from the old table alone, without going back to the raw data (the Lance–Williams idea; stated from memory).
- **Reading a dendrogram:** the height of a join tells you how different the two groups were. A very tall join means "these were far apart", so cutting just below it is often a natural choice.
- **No single right cut:** like all [[Clustering]], there is no ground truth. Use [[Evaluation Metrics]]-style checks such as the silhouette score to compare cuts.
- **Alternative direction:** top-down (divisive) methods start with one group and split. They are less common (from memory).

**What came before, and how this improved on it:**
- **Before:** k-means, which needs `k` first and depends on a random start.
- **Its limit:** a bad start can give a poor answer, and a new `k` means running it again.
- **What changed:** build the whole tree of merges once, then cut it at any height to get any number of clusters. The Ward method is from 1963 (Ward; from memory, not checked).
- **How much better:** in the code below (160 points in 4 groups), one tree gave 2, 3 and 4 clusters from a single run, and its 4-cluster cut scored 0.983 against the true groups. Among 20 single-start k-means runs, the best scored 0.983, the worst 0.614, and 2 were worse than the tree. **Cost:** the tree needs a table of all pair distances, 12,720 for 160 points but 4,999,950,000 for 100,000 points.

**Check it yourself:**

```python
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari
rng = np.random.default_rng(0)
X = np.vstack([rng.normal(c, 0.6, (40, 2)) for c in ((0, 0), (3, 0), (10, 0), (10, 4))]); y = np.repeat(range(4), 40)
Z = linkage(X, "ward")                                   # one run builds the whole tree, no k needed yet
print("same tree cut into 2, 3, 4 clusters:", [len(set(fcluster(Z, k, "maxclust"))) for k in (2, 3, 4)])
tr = ari(y, fcluster(Z, 4, "maxclust")); print("tree cut at 4 clusters, agreement with the true groups:", round(tr, 3))
sc = [ari(y, KMeans(4, n_init=1, random_state=s).fit_predict(X)) for s in range(20)]
print("k-means with one random start, 20 starts: best", round(max(sc), 3), " worst", round(min(sc), 3), " runs worse than the tree:", sum(s < tr - 1e-9 for s in sc))
for m in (160, 100000): print(f"distance table the tree needs for {m:,} points: {m * (m - 1) // 2:,} pairs")
```

## Related
- [[Clustering]] · [[Unsupervised Learning]] · [[Vector]] · [[Normalization]] · [[Cosine Similarity]] · [[Embedding]] · [[Evaluation Metrics]] · [[DBSCAN]] · [[k-NN]]

## Sources
- Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 2011 (from memory); the user guide chapter "Clustering", section on hierarchical clustering (free online)
- SciPy documentation, `scipy.cluster.hierarchy.linkage` (free online)
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, section 14.3.12 (from memory)
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online), chapter 12

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/clustering Hierarchical clustering:::Start with every item alone, repeatedly merge the two closest groups, and record who joined whom and when. The result is a tree (dendrogram) you can cut at any height to get a number of groups.

#flashcards/ml/formula #flashcards/topic/clustering Write the average-linkage distance between two groups and explain each symbol.
?
$$d(A,B)=\frac{1}{|A|\,|B|}\sum_{a\in A}\sum_{b\in B} d(a,b)$$
- $A$, $B$: two groups of items. $|A|$, $|B|$: how many items each has.
- $a$, $b$: single items, one from each group. $d(a,b)$: the distance between that pair.
- The double sum adds the distance of every cross pair ($|A||B|$ pairs). Dividing makes it an average.
- It uses every pair, so one stray item cannot decide alone, unlike single linkage (closest pair only) or complete linkage (farthest pair only). Without the division, big groups would look far away just because they have more members.

#flashcards/ml/example #flashcards/topic/clustering $A=\{a,b\}$, $B=\{e\}$, $d(a,e)=23$, $d(b,e)=21$. What is $d(A,B)$ and where is the merge drawn in the picture?::2 cross pairs, sum 44, $d(A,B)=22$. The figure draws the merge at half of 22, which is 11.0. A bigger case: $\{a,b,e\}$ against $\{c,d\}$ has 6 pairs adding to 198, average 33, drawn at 16.5.

#flashcards/ml/image #flashcards/topic/clustering ![[dendrogram-upgma.svg]]
What does this picture show?
?
Hierarchical clustering of five items a to e. The pair a, b merges first (height 8.5), then e joins them (11.0), then c and d merge (14.0), and finally the two groups merge into one (16.5). (From the note's caption: Emmanuel Douzery, Wikimedia Commons.)

#flashcards/ml/cloze #flashcards/topic/clustering Single linkage uses the ==closest== pair of items, complete linkage the ==farthest== pair, and average linkage ==every== pair.
