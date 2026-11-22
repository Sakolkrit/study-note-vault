---
tags: [local-outlier-factor, LOF, anomaly-detection, outlier-detection, density, k-NN, ML]
status: growing
level: 4
tier: 3
---

# Local Outlier Factor

Local Outlier Factor (LOF) flags a point that is **much lonelier than its own neighbours are**. It compares each point's local crowding with the crowding around the points near it, so "odd" always means "odd for this neighbourhood".

## 🟢 1. Simple (age 12)
A house that is 100 metres from the next house is normal out in the countryside and very strange in the middle of a city. **Odd depends on where you are.** LOF asks of each point: "Are you lonelier than the people around you are?" If everyone nearby is just as spread out as you, you are fine. If you are much more alone than your neighbours, you are odd.

**Example:** in a village the houses are 50 metres apart. A new house stands 70 metres from the nearest one. That is only a bit lonelier than usual, so it is fine. In a city block the houses are 5 metres apart and one house stands 20 metres from its neighbours. That house is four times lonelier than normal there, so it is flagged, even though 20 metres is *closer* than the village house.

## 🔵 2. How it works
1. Choose a number $k$ (for example 20). For each point find its **$k$ nearest neighbours** ([[k-NN]]).
2. For each point measure its **local density**: if its neighbours are very close, the density is high; if far, low. (The exact recipe uses "reachability distance", see level 4.)
3. For each point, compare **its density with the average density of its neighbours**. Divide the neighbours' density by its own.
4. **LOF is that ratio.** Near 1 means "as crowded as my neighbours". Clearly above 1 means "lonelier than my neighbours", which is the anomaly. Below 1 means "even more crowded than my neighbours" (the heart of a group).
5. Rank the points by LOF and look at the top.
6. Unlike a plain distance to the nearest neighbour, LOF does **not** punish a point just for living in a sparse area, as long as its neighbours are sparse too.

## 🟠 3. Technical detail
- **Cost:** it needs the neighbours of every point, roughly $O(n^2)$ without a fast index, so large data sets use a sample or a spatial index. In my shoot-out I fitted LOF on a 6,000-row sample.
- **Key setting:** $k$. Too small and the density is noisy. Too big and "local" is no longer local. It also decides how big a *group of anomalies* can be and still be seen (see the results below).
- **Two modes in scikit-learn:** the default scores the training points themselves; `novelty=True` fits on training data and scores new points. I used `novelty=True`.
- **Code:** LOF written from scratch on a tiny set (8 tightly packed points, 8 loosely packed points, one point near each group), checked against scikit-learn (the script stops with an error if they disagree). It also prints the plain distance to the $k$-th neighbour for comparison.

```python
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

# a tight group (spacing about 0.1), a loose group (spacing about 1.0), one point 0.5 beside the tight group, one point 2.5 beside the loose group
tight = np.array([[0.0, 0], [0.11, 0], [0.19, 0], [0.32, 0], [0.4, 0], [0.52, 0], [0.59, 0], [0.7, 0]])
loose = np.array([[10.0, 0], [11.1, 0], [11.9, 0], [13.2, 0], [14, 0], [15.2, 0], [15.9, 0], [17, 0]])
X = np.vstack([tight, loose, [[1.2, 0]], [[19.5, 0]]])       # 1.2 is 0.5 from the tight group; 19.5 is 2.5 from the loose group
k = 3
D = np.linalg.norm(X[:, None] - X[None], axis=2)
np.fill_diagonal(D, np.inf)
nn = np.argsort(D, axis=1)[:, :k]                            # k nearest neighbours of every point
kdist = np.take_along_axis(D, nn[:, [k - 1]], axis=1)[:, 0]  # distance to the k-th neighbour
reach = np.maximum(D[np.arange(len(X))[:, None], nn], kdist[nn])      # reach-dist(p, o) = max(k-dist(o), d(p, o))
lrd = 1 / reach.mean(axis=1)                                 # local reachability density
lof = lrd[nn].mean(axis=1) / lrd                             # neighbours' density divided by my density
sk = -LocalOutlierFactor(n_neighbors=k).fit(X).negative_outlier_factor_
assert np.allclose(lof, sk), "my LOF differs from sklearn"
for name, i in (("beside tight group", 16), ("beside loose group", 17), ("inside tight group", 3), ("inside loose group", 11)):
    print(f"{name:20s} k-distance {kdist[i]:5.2f}   LOF {lof[i]:5.2f}")
```

Output (run on my machine; the assertion passed, so my numbers equal scikit-learn's):

```text
beside tight group   k-distance  0.68   LOF  2.66
beside loose group   k-distance  4.30   LOF  1.54
inside tight group   k-distance  0.20   LOF  0.89
inside loose group   k-distance  2.00   LOF  0.91
```

- **Reading it:** the point beside the loose group has a **6 times larger** distance to its 3rd neighbour (4.30 against 0.68) than the point beside the tight group, so a plain distance would call it the odder one. LOF says the opposite: 1.54 for the loose-side point and 2.66 for the tight-side point. Relative to its own neighbourhood the tight-side point is the lonelier one. Points inside the groups score about 0.9, as expected.
- **On the Shuttle data** ([[Anomaly Detection Shoot-out]], 49,097 rows, 7.15% anomalies, 5 splits): LOF reached ROC AUC **0.998** when fitted on normal rows only, but **0.515** (chance) when the training rows still contained the odd rows. The reason I tested: 93% of the anomalies are one class forming a crowd of thousands, so each anomaly sits in a dense group of other anomalies and its neighbourhood looks normal. Raising $k$ past the crowd's size fixed it (the big class is 6.65% of the rows, so about 400 of the 6,000 sampled rows, my arithmetic; split 0, mixed training set):

| $k$ | 5 | 20 | 100 | 500 | 1500 |
|---|---|---|---|---|---|
| ROC AUC | 0.516 | 0.515 | 0.434 | 0.985 | 0.993 |

  At $k=100$ the score was **below** chance (0.434) and I did not look into why. I used one split only and did not test other crowd sizes, so treat the explanation as supported, not proved.

## 🔴 4. Deep / formal
Let $N_k(p)$ be the $k$ nearest neighbours of $p$, $d(p,o)$ the distance from $p$ to $o$, and $k\text{-dist}(o)$ the distance from $o$ to its $k$-th nearest neighbour.

**Reachability distance.**

$$
\text{reach-dist}_k(p,o)=\max\big\{k\text{-dist}(o),\;d(p,o)\big\}
$$

**Where:**
- $p$ is the point we are measuring, $o$ one of its neighbours.
- $d(p,o)$ is the real distance from $p$ to $o$.
- $k\text{-dist}(o)$ is how far $o$ itself has to look to find $k$ neighbours: a floor for the distance.
- **Matters:** the floor stops points that sit *very* close to a neighbour from getting a tiny, noisy distance. Everything inside $o$'s own neighbourhood counts as the same distance, which makes the density estimate stable. Without the $\max$ the density of a tight clump would jump around with tiny differences.

**Example (tiny numbers):** the five points $0,1,2,3,8$ on a line with $k=2$ (my arithmetic in Python; scikit-learn gives the same final numbers). The 2-distances are $0\!:\!2$, $1\!:\!1$, $2\!:\!1$, $3\!:\!2$, $8\!:\!6$.
- From point 0 to its neighbour 1: $\max\{k\text{-dist}(1)=1,\;d=1\}=\mathbf{1}$.
- From point 1 to its neighbour 0: $\max\{k\text{-dist}(0)=2,\;d=1\}=\mathbf{2}$. The floor wins: 1 is close to 0, but 0 is itself in a loose spot.
- From point 8 to its neighbour 3: $\max\{k\text{-dist}(3)=2,\;d=5\}=\mathbf{5}$. The real distance wins.

**Local reachability density.**

$$
\text{lrd}_k(p)=\left(\frac{1}{|N_k(p)|}\sum_{o\in N_k(p)}\text{reach-dist}_k(p,o)\right)^{-1}
$$

**Where:**
- $|N_k(p)|$ is the number of neighbours (normally $k$).
- The sum adds the reachability distances from $p$ to its neighbours; dividing by $|N_k(p)|$ makes it an average.
- The power $-1$ turns the average distance into a density: far away means low density.
- **Matters:** it is the "how crowded am I" number. Without the inverse, big would mean sparse, and the ratio below would be upside down.

**Example (tiny numbers):** same five points.
- Point 0 has neighbours 1 and 2 with reach distances 1 and 2. Average 1.5, so $\text{lrd}=1/1.5=\mathbf{0.667}$.
- Point 8 has neighbours 3 and 2 with reach distances 5 and 6. Average 5.5, so $\text{lrd}=1/5.5=\mathbf{0.182}$.
- Meaning: point 8's neighbourhood is about 3.7 times sparser than point 0's.

**The factor.**

$$
\text{LOF}_k(p)=\frac{1}{|N_k(p)|}\sum_{o\in N_k(p)}\frac{\text{lrd}_k(o)}{\text{lrd}_k(p)}
$$

**Where:**
- $\text{lrd}_k(o)/\text{lrd}_k(p)$ is how much denser neighbour $o$'s surroundings are than $p$'s.
- The sum with the $1/|N_k(p)|$ in front is the average of that ratio over the neighbours.
- **Matters:** it is the comparison with the neighbourhood. 1 means "like my neighbours". Above 1 means "lonelier than my neighbours". Because it is a *ratio*, a point in a naturally sparse region is not flagged when its neighbours are sparse too. Without the ratio you would be back at a plain distance, which flags everything in sparse areas.

**Example (tiny numbers):** same five points.
- Point 8: neighbours 3 and 2, both with $\text{lrd}=0.667$. $\text{LOF}=0.667/0.182=\mathbf{3.67}$.
- Point 0: neighbours 1 and 2, both with $\text{lrd}=0.667$ and its own $\text{lrd}=0.667$. $\text{LOF}=\mathbf{1.00}$.
- Meaning: point 8 is 3.7 times lonelier than its neighbours and is the clear anomaly; the four points of the cluster all score exactly 1.0.

**Things to know.**
- **LOF is a ratio of local densities, not a probability.** There is no natural "LOF above 1.5 is an anomaly" line. Choose the cutoff from how many alerts you can handle.
- **It fails when the anomalies come as a crowd bigger than $k$** (my result above), and it can fail the other way: a $k$ larger than a normal cluster blurs genuinely small normal groups.
- **Many columns:** distances lose meaning in high dimensions and all densities look alike. Reduce the dimensions first ([[PCA]]) or use a method that does not rely on distance ([[Isolation Forest]]).
- **Scale matters:** the columns must be on comparable scales ([[Feature Scaling]]); I standardized them.

**What came before, and how this improved on it:**
- **Before:** global distance rules: "flag a point whose distance to its $k$-th neighbour (or to the average) is large", and distribution-based tests.
- **Its limit:** one global threshold cannot handle data with groups of different density: the loose group's normal points look odd and the tight group's real anomalies look normal.
- **What changed:** Breunig, Kriegel, Ng and Sander (2000, SIGMOD, from memory, not checked) compared each point's density with its neighbours', giving every point a *local* factor.
- **How much better:** measured here in the toy: the loose-side point has the larger plain distance (4.30 against 0.68) but the smaller LOF (1.54 against 2.66), so LOF ranks by local oddness. On the Shuttle data LOF reached AUC 0.998 on clean training data, but it was the **worst** detector (0.515) on a training set with a crowd of anomalies, where a global score such as the max z-score reached 0.989. I have no published figure to quote.

## Related
- [[Anomaly Detection]] · [[Isolation Forest]] · [[k-NN]] · [[Feature Scaling]] · [[PCA]] · [[Clustering]] · [[DBSCAN]] · [[Anomaly Detection Shoot-out]]

## Sources
- My own toy code (checked against scikit-learn's `LocalOutlierFactor`) and the experiment [[Anomaly Detection Shoot-out]], including the neighbour-count test. The 2000 paper is named from memory; not read.
- No image: I found no free, on-topic image of LOF on Wikimedia Commons, and did not use an unrelated one.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/anomaly-detection Local Outlier Factor (LOF):::An anomaly score that compares a point's local density with the density around its k nearest neighbours. About 1 means as crowded as the neighbours; clearly above 1 means lonelier than the neighbours.

#flashcards/ml/reverse #flashcards/topic/anomaly-detection k-distance of a point:::The distance from the point to its k-th nearest neighbour.

#flashcards/ml/concept #flashcards/topic/anomaly-detection Why is LOF better than "distance to the k-th neighbour" when groups have different densities?
?
A plain distance flags every point in a naturally sparse group. LOF divides by the neighbours' own density, so a point is odd only relative to its neighbourhood. In my toy, the loose-side point had the larger plain distance (4.30 vs 0.68) but the smaller LOF (1.54 vs 2.66).

#flashcards/ml/formula #flashcards/topic/anomaly-detection What is the reachability distance in LOF?
?
$$
\text{reach-dist}_k(p,o)=\max\big\{k\text{-dist}(o),\;d(p,o)\big\}
$$
- $p$ is the point measured, $o$ one of its neighbours. $d(p,o)$ is their real distance.
- $k\text{-dist}(o)$ is how far $o$ must look to find $k$ neighbours: a floor for the distance.
- The floor smooths the density: everything inside $o$'s own neighbourhood counts as the same distance, so tiny differences do not make the density jump around.

#flashcards/ml/concept #flashcards/topic/anomaly-detection For the points 0, 1, 2, 3, 8 with k = 2, what are reach-dist(1,0) and reach-dist(8,3)?
?
k-dist(0) = 2 and d(1,0) = 1, so reach-dist(1,0) = max(2,1) = 2 (the floor wins). k-dist(3) = 2 and d(8,3) = 5, so reach-dist(8,3) = max(2,5) = 5 (the real distance wins). (My arithmetic.)

#flashcards/ml/formula #flashcards/topic/anomaly-detection What is the local reachability density?
?
$$
\text{lrd}_k(p)=\left(\frac{1}{|N_k(p)|}\sum_{o\in N_k(p)}\text{reach-dist}_k(p,o)\right)^{-1}
$$
- $N_k(p)$ is the set of $p$'s $k$ nearest neighbours; the sum over them divided by $|N_k(p)|$ is the average reachability distance.
- The power $-1$ turns a far average into a low density.
- Without the inverse, big would mean sparse and the LOF ratio would be upside down.

#flashcards/ml/concept #flashcards/topic/anomaly-detection For the points 0, 1, 2, 3, 8 with k = 2, what are lrd of point 0 and of point 8?
?
Point 0: neighbours 1 and 2, reach distances 1 and 2, average 1.5, lrd = 1/1.5 = 0.667. Point 8: neighbours 3 and 2, reach distances 5 and 6, average 5.5, lrd = 1/5.5 = 0.182. (My arithmetic.)

#flashcards/ml/formula #flashcards/topic/anomaly-detection What is the LOF formula?
?
$$
\text{LOF}_k(p)=\frac{1}{|N_k(p)|}\sum_{o\in N_k(p)}\frac{\text{lrd}_k(o)}{\text{lrd}_k(p)}
$$
- $\text{lrd}_k(o)/\text{lrd}_k(p)$ is how much denser neighbour $o$'s surroundings are than $p$'s; the sum divided by $|N_k(p)|$ averages it over the neighbours.
- 1 means as crowded as the neighbours; above 1 means lonelier than them.
- Being a ratio, it does not punish points for living in a naturally sparse area.

#flashcards/ml/concept #flashcards/topic/anomaly-detection For the points 0, 1, 2, 3, 8 with k = 2, what is LOF of point 8 and of point 0?
?
Point 8: neighbours' lrd 0.667 and 0.667, own lrd 0.182, so LOF = 0.667/0.182 = 3.67. Point 0: neighbours' lrd 0.667, own 0.667, so LOF = 1.00. (My arithmetic; scikit-learn gives the same.)

#flashcards/ml/concept #flashcards/topic/anomaly-detection What happened to LOF on the contaminated Shuttle training set, and what fixed it?
?
ROC AUC 0.515 (chance) with k = 20 because 93% of the anomalies form one big crowd, so each anomaly's neighbours are other anomalies. Raising k above the crowd's size fixed it: k = 500 gave 0.985, k = 1500 gave 0.993 (split 0, one run). At k = 100 it was below chance, 0.434.

#flashcards/ml/concept #flashcards/topic/anomaly-detection How well did LOF do on the Shuttle data when fitted on normal rows only?
?
ROC AUC 0.998 (mean of 5 splits), as good as the other detectors there (0.989 to 0.999). Fitted on rows that still contained the odd ones, it fell to 0.515.

#flashcards/ml/cloze #flashcards/topic/anomaly-detection LOF fails when the anomalies form a ==crowd== bigger than ==k==.
