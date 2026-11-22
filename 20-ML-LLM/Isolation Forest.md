---
tags: [isolation-forest, anomaly-detection, outlier-detection, tree-ensemble, unsupervised-learning, ML]
status: growing
level: 4
tier: 3
---

# Isolation Forest

Isolation Forest finds odd points by **trying to cut each point off from all the others with random cuts**. A point that is odd is cut off after only a few cuts. A normal point, buried in a crowd, needs many.

## 🟢 1. Simple (age 12)
Picture a field with a big crowd of people in the middle and one person standing far away by the fence. Throw darts that draw random lines across the field. After each line, keep only the part that holds your target person. The person by the fence is alone after one or two lines. A person in the middle of the crowd needs many lines before they are alone. **The faster you can cut someone off, the odder they are.**

**Example:** the numbers 1, 2, 3, ..., 10 and then 40. Cut at a random place between 1 and 40. Most cuts land between 10 and 40, and the side with 40 in it holds just that one number. So 40 is isolated at once, while the number 5 needs several cuts.

## 🔵 2. How it works
1. Take a random handful of the data (a **sample**; 256 points is the usual size).
2. **Build a random tree.** Pick a column at random. Pick a cut value at random between the smallest and the largest value of that column in the current group. Split the group in two. Do the same inside each half, again and again, until each point is alone (or the tree is deep enough).
3. For one point, count how many cuts it took to be alone. That count is its **path length** (how far down the tree it sits).
4. **Build many trees** (100 is the usual number), each from its own random sample and its own random cuts. This is a "forest" ([[Random Forest]] builds trees too, but for another job).
5. **Average the path length** of a point over all trees. Short average = odd. Long average = normal.
6. Turn it into a score between 0 and 1 (formula in level 4). Close to 1 means "very odd". About 0.5 or below means "normal".
7. It uses **no distances and no density**, so it is fast and it does not need the data to have a particular shape. It never uses labels.

![[anomaly-isolation-forest-scores.png]]
*An isolation-forest style anomaly score map. Black dots are the data (eruption time and waiting time). Pale yellow means looks normal (the two dense groups) and dark red means most anomalous (the empty corners). Notice the **straight bands and boxes**: every cut is parallel to an axis, so the map is built from rectangles. Author Mbt19, [Normalized Anomaly Scores of Isolation Forest.png on Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Normalized_Anomaly_Scores_of_Isolation_Forest.png), CC BY 3.0. I read the colour meaning from the picture, which has no legend.*

## 🟠 3. Technical detail
- **Why short paths mean odd.** Anomalies are *few* and *different*: few points share their region, so it takes few cuts to separate them. Normal points are *many* and *similar*, so each needs many cuts.
- **Cheap.** Each tree uses a small sample, so training and scoring are fast and use little memory; roughly, scoring a point takes about the tree depth, $\log_2$ of the sample size, per tree.
- **Settings:** number of trees (100 is the default in scikit-learn), sample size per tree (256 or all if the data is smaller), and `contamination` (only used to turn scores into yes/no labels, not to compute the scores).
- **Code:** a from-scratch isolation depth on 11 numbers (1 to 10 and 40), compared with scikit-learn's `IsolationForest`. The from-scratch version cuts until the point is alone and does not limit the tree depth, so it differs a little from the library (which stops at a depth limit and adds a correction for the group it stopped in).

```python
import numpy as np
from sklearn.ensemble import IsolationForest

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 40.])          # 11 numbers, one far from the rest
n = len(x)
rs = np.random.RandomState(0)


def depth_to_isolate(q):                                     # cut at random until q sits alone; count the cuts
    pts, depth = x.copy(), 0
    while len(pts) > 1:
        cut = rs.uniform(pts.min(), pts.max())
        pts = pts[pts <= cut] if q <= cut else pts[pts > cut]
        depth += 1
    return depth


def c(n):                                                    # average depth of a failed search in a binary search tree of n points
    return 0.0 if n <= 1 else 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n


avg = np.array([np.mean([depth_to_isolate(q) for _ in range(4000)]) for q in x])
score = 2 ** (-avg / c(n))
ref = -IsolationForest(n_estimators=2000, max_samples=n, random_state=0).fit(x.reshape(-1, 1)).score_samples(x.reshape(-1, 1))
print(f"c({n}) = {c(n):.3f}")
print(" x   avg cuts to isolate   my score   sklearn score")
for q, a, s, r in zip(x, avg, score, ref): print(f"{q:4.0f} {a:12.2f} {s:14.3f} {r:12.3f}")
```

Output (run on my machine):

```text
c(11) = 3.941
 x   avg cuts to isolate   my score   sklearn score
   1         3.56          0.534        0.534
   2         4.52          0.451        0.460
   3         4.91          0.422        0.434
   4         5.10          0.408        0.419
   5         5.21          0.400        0.413
   6         5.27          0.396        0.409
   7         5.20          0.401        0.412
   8         5.01          0.414        0.424
   9         4.66          0.441        0.445
  10         3.79          0.514        0.509
  40         1.26          0.801        0.802
```

- **Reading it:** the far point 40 needs only 1.26 cuts on average and scores 0.80 in both versions. The middle numbers (4 to 7) need 5 or more cuts and score about 0.40. The two end points of the crowd (1 and 10) are easier to isolate than the middle (3.6 and 3.8 cuts) and score a bit higher (about 0.51 to 0.53). The two versions agree to within 0.013 everywhere.
- **On the Shuttle data** ([[Anomaly Detection Shoot-out]], 49,097 rows, 7.15% anomalies, 5 random splits, test half scored): when the training half still contained its odd rows, ROC AUC was **0.998** (spread 0.000) and average precision 0.979, the best of 8 detectors; with normal-only training, AUC 0.997 (about the same as the other detectors, which all reached 0.989 to 0.999 there). Neighbour-based detectors fell to 0.515 ([[Local Outlier Factor]]) and 0.731 (k-NN distance) on the mixed training set.

## 🔴 4. Deep / formal
**Normalizing the path length.** A random tree on $n$ points behaves like a binary search tree, whose average depth of an unsuccessful search is

$$
c(n)=2H(n-1)-\frac{2(n-1)}{n},\qquad H(m)\approx\ln m+0.5772
$$

**Where:**
- $n$ is the number of points in the sample (or in the group where a branch stopped).
- $H(m)$ is the $m$-th harmonic number $1+\tfrac12+\dots+\tfrac1m$; $\ln m$ is the natural logarithm and 0.5772 is Euler's constant.
- $c(n)$ is the typical number of cuts needed to separate a point among $n$ points.
- **Matters:** it is the yardstick. Raw path lengths depend on the sample size, so a path of 5 cuts means different things for $n=11$ and $n=256$. Dividing by $c(n)$ makes the number comparable across sample sizes. It grows slowly (like $\ln n$), which is why a tiny sample is enough.

**Example (tiny numbers):** $n=11$ (my arithmetic in Python).
- $\ln 10=2.3026$, so $H(10)\approx2.3026+0.5772=2.8798$.
- $c(11)=2\cdot2.8798-\dfrac{2\cdot10}{11}=5.7596-1.8182=\mathbf{3.94}$.
- Meaning: among 11 points a typical point needs about 3.94 cuts.

**The anomaly score.** With $E[h(x)]$ the average path length of point $x$ over all trees:

$$
s(x,n)=2^{-E[h(x)]/c(n)}
$$

**Where:**
- $E[h(x)]$ is the number of cuts it took to isolate $x$, averaged over all trees.
- $c(n)$ is the yardstick from above.
- $s(x,n)$ is the score, between 0 and 1.
- **Matters:** the exponent turns "short path" into a score near 1 and "long path" into a score near 0. When $E[h]=c(n)$ the score is exactly $2^{-1}=0.5$: an average point. Without the $2^{-(\cdot)}$ shape you would have to guess a threshold for each sample size.

**Example (tiny numbers):** the 11 numbers with $c(11)=3.94$ (from the code above).
- The far point 40: $E[h]=1.26$, so $s=2^{-1.26/3.94}=2^{-0.32}=\mathbf{0.80}$.
- A middle point 6: $E[h]=5.27$, so $s=2^{-5.27/3.94}=2^{-1.34}=\mathbf{0.40}$.
- Meaning: 40 is flagged (0.80, well above 0.5) and 6 is clearly normal (0.40, well below 0.5).

**Things to know.**
- **Axis-parallel cuts leave a shadow.** All cuts are parallel to the axes, so scores have rectangular artifacts (visible in the picture). A point can look more normal because it lines up with a crowd along one axis. The *Extended Isolation Forest* (Hariri, Kind and Brunner, from memory) uses sloped cuts to reduce this; I did not test it.
- **Irrelevant columns dilute it.** Cuts on a useless column isolate nothing, so with many such columns the signal is blurred.
- **Local anomalies are hard.** Isolation measures how *globally* easy a point is to separate. A point that is odd only compared with its nearby crowd (the case [[Local Outlier Factor]] is built for) can be missed.
- **Contamination hurts it less than neighbour methods.** Because the trees use small random samples and random cuts, a crowd of anomalies is still cut apart quickly, if it is small or far from normal. In my run a 7% crowd did not hurt it at all. Do not generalize: that crowd was far from the normal points (the evidence is the very high score), and I did not vary it.
- **The score is a rank, not a probability.** 0.8 does not mean "80% likely anomalous".

**What came before, and how this improved on it:**
- **Before:** distance and density methods that first model what normal looks like (k-NN distance, [[Local Outlier Factor]], one-class SVM), costing roughly $O(n^2)$ time.
- **Its limit:** they need a distance measure, struggle in many dimensions and are slow on big data sets.
- **What changed:** Liu, Ting and Zhou (2008, from memory, not checked) proposed isolating points with random cuts, with small samples, so no distance or density is computed.
- **How much better:** measured here, ROC AUC 0.998 against 0.515 (LOF) and 0.731 (k-NN distance) when the training set still had its odd rows, on one data set. In the same test it did **not** beat the other detectors on clean training data (0.997 against 0.989 to 0.999). Timing: the isolation forest took 2.2 to 2.9 seconds per run against 3.2 to 4.1 for LOF, but the laptop was busy and I treat timing as unreliable. I have no published figure to quote.

## Related
- [[Anomaly Detection]] · [[Local Outlier Factor]] · [[Random Forest]] · [[Decision Tree]] · [[Ensemble Methods]] · [[Gaussian Mixture Model]] · [[Autoencoder]] · [[ROC Curve and AUC]] · [[Precision-Recall Curve]] · [[Big-O]] · [[Anomaly Detection Shoot-out]]

## Sources
- My own toy code and the experiment [[Anomaly Detection Shoot-out]]. The formulas for $c(n)$ and $s(x,n)$ I wrote from memory and checked by running the toy: my from-scratch scores agree with scikit-learn's within 0.013. The 2008 paper and the Extended Isolation Forest are named from memory; not read.
- Image: Mbt19, [Normalized Anomaly Scores of Isolation Forest.png](https://commons.wikimedia.org/wiki/File:Normalized_Anomaly_Scores_of_Isolation_Forest.png), CC BY 3.0.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/anomaly-detection Isolation Forest:::An anomaly detector that cuts the data with random axis-parallel splits in many random trees. Points that are cut off after few splits (short average path) are the anomalies.

#flashcards/ml/reverse #flashcards/topic/anomaly-detection Path length (in an isolation tree):::The number of cuts needed to separate a point from all the others in one tree. Short means odd, long means normal.

#flashcards/ml/concept #flashcards/topic/anomaly-detection Why does an anomaly get isolated in few cuts?
?
Anomalies are few and different: few points share their region, so a random cut soon leaves the anomaly alone. Normal points are many and similar, so each needs many cuts.

#flashcards/ml/concept #flashcards/topic/anomaly-detection Does an isolation forest use distances, density or labels?
?
None of them. It only uses random cuts and path lengths, so it is fast and does not need the data to have a particular shape. It is unsupervised.

#flashcards/ml/image #flashcards/topic/anomaly-detection ![[anomaly-isolation-forest-scores.png]]
What does the shape of this score map tell you about isolation forests?
?
The map is made of straight bands and boxes, because every cut is parallel to an axis. Pale yellow is normal (dense groups of dots), dark red is most anomalous (empty corners). By Mbt19, CC BY 3.0; colour meaning read from the picture.

#flashcards/ml/formula #flashcards/topic/anomaly-detection What is the normalizing constant c(n) in the isolation forest score?
?
$$
c(n)=2H(n-1)-\frac{2(n-1)}{n},\qquad H(m)\approx\ln m+0.5772
$$
- $n$ is the sample size; $H(m)$ is the harmonic number $1+\tfrac12+\dots+\tfrac1m$.
- $c(n)$ is the typical number of cuts needed to separate a point among $n$ points (the average depth of an unsuccessful search in a binary search tree).
- It makes path lengths comparable across sample sizes and grows only like $\ln n$.

#flashcards/ml/concept #flashcards/topic/anomaly-detection What is c(11)?
?
H(10) ≈ ln 10 + 0.5772 = 2.3026 + 0.5772 = 2.8798. c(11) = 2 × 2.8798 − 2 × 10 / 11 = 5.7596 − 1.8182 = 3.94. (My arithmetic.)

#flashcards/ml/formula #flashcards/topic/anomaly-detection What is the isolation forest anomaly score?
?
$$
s(x,n)=2^{-E[h(x)]/c(n)}
$$
- $E[h(x)]$ is the average path length (number of cuts) of point $x$ over all trees; $c(n)$ is the normalizing yardstick.
- $s$ lies between 0 and 1: near 1 is very odd, about 0.5 is an average point, small is clearly normal.
- It is a rank-like score, not a probability.

#flashcards/ml/concept #flashcards/topic/anomaly-detection In the 11-number toy (c = 3.94), what are the scores of 40 (average path 1.26) and 6 (average path 5.27)?
?
40: 2^(−1.26/3.94) = 2^(−0.32) = 0.80. 6: 2^(−5.27/3.94) = 2^(−1.34) = 0.40. So 40 is flagged and 6 is clearly normal. (My arithmetic, matches the program.)

#flashcards/ml/concept #flashcards/topic/anomaly-detection What does a score of exactly 0.5 mean?
?
The average path length equals c(n), so the point is an average point: 2^(−1) = 0.5. Nothing stands out.

#flashcards/ml/concept #flashcards/topic/anomaly-detection How did the isolation forest do on the Shuttle data, and how did that depend on the training set?
?
Trained on a mix that still had about 7% odd rows: ROC AUC 0.998 and average precision 0.979, best of 8 detectors. Trained on normal rows only: AUC 0.997, about the same as the others (0.989 to 0.999). One data set, 5 splits, no tuning.

#flashcards/ml/concept #flashcards/topic/anomaly-detection Name three weaknesses of the isolation forest.
?
Axis-parallel cuts leave rectangular artifacts; irrelevant columns dilute the signal; locally odd points (odd only next to their neighbours) can be missed because isolation is a global measure.

#flashcards/ml/cloze #flashcards/topic/anomaly-detection The isolation forest score is close to ==one== for anomalies and about ==one half== for an average point.

#flashcards/ml/concept #flashcards/topic/anomaly-detection Where do the Isolation Forest history facts come from?
?
Liu, Ting and Zhou (2008) and the Extended Isolation Forest (Hariri, Kind and Brunner) are named from memory and were not checked or read.
