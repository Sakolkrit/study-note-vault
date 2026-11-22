---
tags: [machine-learning, unsupervised-learning, clustering, density, outlier-detection]
status: growing
level: 4
tier: 2
---

# DBSCAN

DBSCAN finds groups as **crowded regions separated by empty space**, and it labels the loners as noise. The name stands for Density-Based Spatial Clustering of Applications with Noise.

## 🟢 1. Simple (age 12)
Look at a map of a country at night, covered in city lights. Big bright blobs are cities. The dark gaps between are countryside. A single lonely light far from anything is just a farm. DBSCAN does the same with data: **a crowded patch of points is one group, and a lonely point belongs to no group.** The shape of the patch does not matter, it can be long, bent or round.

**Example:** You mark where every kid in a park is standing. Kids chatting in a tight bunch form one group. Another bunch near the swings is a second group. A kid standing alone by the fence is "noise". Nobody had to tell you there would be two groups.

## 🔵 2. How it works
This is a method of [[Clustering]] in [[Unsupervised Learning]]. You choose two numbers:
- a **radius** $\varepsilon$ ("epsilon"): how close counts as "near";
- a **crowd size** `minPts`: how many neighbours make a spot "crowded".

Then:
1. For each point, count how many points (including itself) lie within the radius. Distances come from the points' [[Vector]] positions.
2. If the count reaches `minPts`, the point is a **core point**. It sits in a crowded spot.
3. Pick an unvisited core point and start a new group. Add every point within its radius.
4. For each *core* point you just added, add its neighbours too. The group keeps **spreading through the crowd**.
5. Points that are near a core point but not crowded themselves become **border points**. They join the group but do not spread it.
6. Points that no core point can reach are **noise**.

![[dbscan-illustration.svg]]
*DBSCAN with minPts = 3. The red points around A are core points: their circles (radius ε) each hold enough neighbours, and they pass the group along through each other. B and C are yellow border points: inside a core point's circle but not crowded themselves. N is blue noise: no core point reaches it. Image: Chire, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:DBSCAN-Illustration.svg), CC BY-SA 3.0.*

## 🟠 3. Technical detail
- **You do not choose the number of groups.** It falls out of the radius and crowd size. This also makes it good at **spotting outliers** (the noise label).
- **Any shape:** because groups spread through neighbours, DBSCAN follows curves such as the two half-moons that k-means cut wrongly (see [[Clustering]]).
- **The two knobs are touchy:**
  - **Radius too small:** nothing is crowded enough, so everything becomes noise.
  - **Radius too large:** separate groups merge, and noise gets swallowed (shown in the code).
  - A common trick to pick the radius: for each point, find the distance to its $k$-th nearest neighbour (see [[k-NN]]), sort these, and look for a sudden bend in the curve (from memory, a rule of thumb).
- **One radius for all:** if one group is dense and another is loose, no single radius fits both. Later methods such as OPTICS and HDBSCAN handle this (stated from memory; not covered here).
- **High dimensions:** distances start to look alike, so "near" stops being meaningful. Scale features first (see [[Normalization]]) and consider reducing them with [[PCA]].
- **Border points can be ambiguous:** a border point near two groups joins whichever reached it first, so the result can depend slightly on the order of the data.
- **Cost:** with a search structure it is fast for moderate data. Without one, it compares many pairs of points.
- **Uses:** finding anomalies (fraud, sensor faults), grouping places on a map, and clustering [[Embedding]] vectors when the number of topics is unknown.

```python
import numpy as np
from sklearn.cluster import DBSCAN
X = np.array([[1,1],[1.2,1],[1,1.3],[1.3,1.2],     # a tight bunch
              [5,5],[5.2,5],[5,5.3],[5.3,5.2],     # a second bunch
              [3,9]], float)                        # a loner
for eps in (0.5, 0.2, 5.0):
    print(eps, DBSCAN(eps=eps, min_samples=3).fit_predict(X).tolist())
# 0.5 [0, 0, 0, 0, 1, 1, 1, 1, -1]    two groups, the loner is noise (label -1)
# 0.2 [-1, -1, -1, -1, -1, -1, -1, -1, -1]    radius too small: everything is noise
# 5.0 [0, 0, 0, 0, 1, 1, 1, 1, 1]    radius too big: the loner is swallowed into group 1

p = np.array([1, 2, 3, 4, 10.])                     # the tiny 1-D example from the maths below
print([int((abs(p - a) <= 1.5).sum()) for a in p])  # [2, 3, 3, 2, 1]  neighbours within 1.5 (counting itself)
print(DBSCAN(eps=1.5, min_samples=3).fit_predict(p[:, None]).tolist())   # [0, 0, 0, 0, -1]
```

In the 1-D example only the points at 2 and 3 are core points (3 neighbours each). The points at 1 and 4 are border points, near a core point but with only 2 neighbours. The point at 10 is noise. All four of the first points end up in one group.

## 🔴 4. Deep / formal
**Core points.** For a radius $\varepsilon$, the neighbourhood of a point $p$ is every point within that distance. The point is a *core point* if the neighbourhood is big enough:

$$
N_\varepsilon(p)=\{\,q : d(p,q)\le\varepsilon\,\},\qquad p\ \text{is core}\iff |N_\varepsilon(p)|\ge \text{minPts}
$$

**Where:**
- $p$ is the point being tested and $q$ is any point in the data (including $p$ itself).
- $d(p,q)$ is the distance between them, usually the straight-line distance (see [[k-NN]] for the formula).
- $\varepsilon$ is the radius. $N_\varepsilon(p)$ is the set of points inside that circle (or ball).
- $|N_\varepsilon(p)|$ is how many points are in the set. minPts is the crowd size you chose.
- **Matters:** this one test is the whole idea of "crowded". Raise minPts or shrink $\varepsilon$ and fewer points qualify, so you get more noise and smaller groups. Lower them and groups grow and merge.

**Example (tiny numbers):** points at $1,\,2,\,3,\,4,\,10$ on a line, with $\varepsilon=1.5$ and minPts $=3$.
- Point 2: neighbours within 1.5 are 1, 2 and 3. That is $3\ge3$, so **core**.
- Point 3: neighbours are 2, 3 and 4. That is $3$, so **core**.
- Point 1: neighbours are 1 and 2. That is $2<3$, so not core, but it is near the core point 2, so it is a **border** point. Same for point 4.
- Point 10: only itself, $1<3$, and no core point is near, so it is **noise**.
- Meaning: the result is one group $\{1,2,3,4\}$ plus one noise point, matching the code output `[0, 0, 0, 0, -1]`.

**Further notes:**
- **Density-reachable:** a point belongs to a group if you can get to it by hopping from core point to core point, each hop within $\varepsilon$. This chain idea is what lets a group bend.
- **Deterministic core:** the core points and their groups are always the same. Only the border points can depend on the order of the data.
- **No centre, no model of the group:** DBSCAN labels the points you gave it. It does not give you a centre or a rule to label a new point (unlike k-means or [[Gaussian Mixture Model]]).
- **Evaluation:** the silhouette score (see [[Clustering]]) favours round groups, so it can unfairly mark down a correct DBSCAN result on curved shapes.

**What came before, and how this improved on it:**
- **Before:** [[Clustering]] with k-means, which needs the number of clusters in advance and finds round blobs.
- **Its limit:** it splits curved shapes wrongly, and it has to put every point, even a far-away outlier, in some cluster.
- **What changed:** grow clusters from dense regions and leave lonely points as noise. DBSCAN was published in 1996 by Ester, Kriegel, Sander and Xu (from memory, not checked).
- **How much better:** in the code below (two interleaved half-moons, 400 points), k-means scored 0.274 against the true groups and DBSCAN scored 1.0 with 2 clusters, with no `k` given. Two far-away outliers were labelled -1 (noise). **Catch:** DBSCAN has its own settings (`eps`, `min_samples`), and I chose `eps=0.2` for this data; a bad choice would hurt.

**Check it yourself:**

```python
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score as ari
X, y = make_moons(400, noise=0.06, random_state=0)
km = KMeans(2, n_init=10, random_state=0).fit_predict(X)
db = DBSCAN(eps=0.2, min_samples=5).fit_predict(X)
print("two interleaved half-moons, 400 points (1.0 = perfect, 0 = no better than chance)")
print("  k-means (needs k=2, thinks in round blobs):", round(ari(y, km), 3))
print("  DBSCAN (finds dense regions, no k):        ", round(ari(y, db), 3), " clusters found:", len(set(db) - {-1}))
Xn = np.vstack([X, [[3, 3], [-2, 2.5]]]); dn = DBSCAN(eps=0.2, min_samples=5).fit_predict(Xn)
print("two far-away outlier points added: DBSCAN labels them", dn[-2:], "(-1 means noise); k-means must put them in a cluster")
```

## Related
- [[Clustering]] · [[Unsupervised Learning]] · [[Hierarchical Clustering]] · [[Gaussian Mixture Model]] · [[k-NN]] · [[Vector]] · [[Normalization]] · [[PCA]] · [[Embedding]] · [[K-Means]]
- [[Local Outlier Factor]]
## Sources
- Ester, Kriegel, Sander & Xu, *A density-based algorithm for discovering clusters in large spatial databases with noise*, KDD 1996 (from memory)
- Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 2011 (from memory); user guide, "Clustering", DBSCAN section (free online)
- Schubert, Sander, Ester, Kriegel & Xu, *DBSCAN revisited, revisited*, ACM TODS 2017 (from memory)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/clustering DBSCAN:::A crowded patch of points is one group, and a lonely point belongs to no group. The shape of the patch does not matter.

#flashcards/ml/formula #flashcards/topic/clustering Write the neighbourhood and core-point test and explain each symbol.
?
$$N_\varepsilon(p)=\{\,q : d(p,q)\le\varepsilon\,\},\qquad p\ \text{is core}\iff |N_\varepsilon(p)|\ge \text{minPts}$$
- $p$: the point being tested. $q$: any point in the data (including $p$ itself).
- $d(p,q)$: the distance between them, usually straight-line.
- $\varepsilon$: the radius. $N_\varepsilon(p)$: the set of points inside that circle.
- $|N_\varepsilon(p)|$: how many points are in it. minPts: the crowd size you chose.
- This one test is the whole idea of "crowded". Raise minPts or shrink $\varepsilon$ and fewer points qualify, giving more noise and smaller groups. Lower them and groups grow and merge.

#flashcards/ml/example #flashcards/topic/clustering Points 1, 2, 3, 4, 10 on a line, $\varepsilon=1.5$, minPts $=3$. Label each point.::Points 2 and 3 have 3 neighbours, so they are core. Points 1 and 4 have 2 neighbours (not core) but are near a core point, so they are border. Point 10 has only itself and no core point near, so it is noise. Result: one group {1,2,3,4} plus one noise point.

#flashcards/ml/image #flashcards/topic/clustering ![[dbscan-illustration.svg]]
What does this picture show?
?
DBSCAN with minPts = 3. The red points around A are core points: their circles (radius $\varepsilon$) each hold enough neighbours, and they pass the group along through each other. B and C are yellow border points: inside a core point's circle but not crowded themselves. N is blue noise: no core point reaches it. (From the note's caption: Chire, Wikimedia Commons.)

#flashcards/ml/reverse #flashcards/topic/clustering DBSCAN point types:::Core (enough neighbours within $\varepsilon$), border (near a core point but not crowded itself), noise (no core point reaches it).
