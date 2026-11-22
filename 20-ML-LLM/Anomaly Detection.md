---
tags: [anomaly-detection, outlier-detection, novelty-detection, unsupervised-learning, fraud, monitoring, ML]
status: growing
level: 4
tier: 2
---

# Anomaly Detection

Anomaly detection means **finding the few things that do not look like the rest**: a strange bank payment, a machine that starts to vibrate oddly, a sensor reading that makes no sense. You usually have lots of examples of "normal" and almost none of "odd".

## 🟢 1. Simple (age 12)
In a class photo where everyone smiles, you can spot the one person who is pulling a face. You do not need a list of all the ways to pull a face. You only need to know what "normal" looks like, and then pick out whatever does not fit. **Anomaly detection teaches a computer what normal looks like, so it can point at the odd one out.**

**Example:** a bank sees that you buy lunch near school every day. One day a payment of 2,000 dollars comes from a shop in another country. Nobody told the bank that this exact payment is bad, but it looks nothing like your normal days, so the bank asks, "Was this you?"

## 🔵 2. How it works
1. **Describe each thing with numbers** (a payment: amount, hour, shop type; a machine: temperature, vibration). See [[Feature Engineering]].
2. **Learn what normal looks like** from data. Two settings:
   - **Unsupervised:** the training data is a mix. It is mostly normal, with some odd items hiding in it, and nobody labelled them ([[Unsupervised Learning]]).
   - **Semi-supervised (one-class):** the training data is known to be normal only.
3. **Give every new item an oddness score.** Big score means "does not fit".
4. **Pick a cutoff** (or take the top few items) and send those for a human or a rule to check.
5. **Different ideas for the score**, each with its own note or section:
   - *Far from the average:* a z-score or a Mahalanobis distance (both explained below).
   - *Few neighbours nearby:* distance to the $k$-th nearest neighbour ([[k-NN]]) or [[Local Outlier Factor]].
   - *Easy to cut off from the rest:* [[Isolation Forest]].
   - *Low probability under a model of normal:* [[Gaussian Mixture Model]], One-Class SVM ([[Support Vector Machine]]).
   - *The model cannot rebuild it:* an [[Autoencoder]] that learned to copy normal data gets a big copying error on odd data.
   - *Different from the forecast:* for a sequence over time, forecast what should come next and flag a big surprise ([[Time Series]], [[Forecast Metrics]]).
6. **Three kinds of odd:** a **point** anomaly (one value is strange), a **contextual** one (30 degrees is normal in July and odd in January) and a **collective** one (each reading is fine, the sequence is not).
7. **Scoring the detector is tricky:** anomalies are rare, so "accuracy" is useless (a detector that says "all normal" is 93% accurate when 7% are odd). Use [[ROC Curve and AUC]], [[Precision-Recall Curve]] and precision in the top $k$ instead.

![[anomaly-isolation-forest-scores.png]]
*An anomaly score map. The black dots are the data (axes: eruption time and waiting time). Pale yellow means "looks normal" and dark red means "most anomalous": the two dense groups are pale and the empty corners are dark. Author Mbt19, [Normalized Anomaly Scores of Isolation Forest.png on Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Normalized_Anomaly_Scores_of_Isolation_Forest.png), CC BY 3.0. The file name says the scores are from an isolation forest; I read the colour meaning from the picture, as it has no legend.*

## 🟠 3. Technical detail
- **Why simple per-column rules fail.** A rule such as "flag anything more than 3 standard deviations out in any column" misses points that break the **relationship** between columns. The toy below has two measurements that move together; five points break the pattern but are not extreme on either axis.

```python
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

rs = np.random.RandomState(0)
normal = rs.multivariate_normal([0, 0], [[1, .9], [.9, 1]], 500)                 # 500 normal points: two measurements that move together
odd = np.array([[1.5, -1.5], [-1.5, 1.5], [1.2, -1.0], [2.0, -2.0], [-2.0, 2.0]])   # 5 points that break the pattern, none of them extreme on either axis
X = np.vstack([normal, odd]); y = np.r_[np.zeros(500), np.ones(5)]

mu, Sinv = normal.mean(0), np.linalg.inv(np.cov(normal.T))
z = np.abs((X - normal.mean(0)) / normal.std(0)).max(1)                           # largest |z-score| over the two axes
m = np.einsum("ij,jk,ik->i", X - mu, Sinv, X - mu)                                # squared Mahalanobis distance
f = -IsolationForest(random_state=0).fit(normal).score_samples(X)                 # isolation forest, fitted on the normal points
for name, s in (("max |z-score|", z), ("Mahalanobis", m), ("isolation forest", f)):
    rank = (-s).argsort().argsort() + 1                                           # 1 = most anomalous
    print(f"{name:18s} ROC AUC {roc_auc_score(y, s):.3f} | ranks of the 5 odd points {sorted(rank[500:].tolist())}")
```

Output (run on my machine):

```text
max |z-score|      ROC AUC 0.864 | ranks of the 5 odd points [25, 27, 72, 76, 156]
Mahalanobis        ROC AUC 1.000 | ranks of the 5 odd points [1, 2, 3, 4, 5]
isolation forest   ROC AUC 0.973 | ranks of the 5 odd points [7, 9, 18, 23, 26]
```

- **Reading it:** the five odd points are ranked 25th to 156th by the per-column z-score (AUC 0.864), but 1st to 5th by the Mahalanobis distance, which knows about the correlation (AUC 1.000). The isolation forest, which knows nothing about the shape, puts them at 7th to 26th (AUC 0.973). Hand-made toy, 505 points, one run: it shows the idea, not a ranking of methods.
- **The shoot-out** ([[Anomaly Detection Shoot-out]]): 8 detectors on the Shuttle data (49,097 rows, 9 numbers each, 7.15% anomalies, taken from the OpenML copy, with the classes 2, 3, 5, 6, 7 as anomalies). Test half scored, 5 random splits, **ROC AUC** (mean over 5 splits; the spread over splits was 0.000 to 0.016, except 0.083 for the autoencoder in the first column):

| Detector | Trained on a mix (about 7% odd rows) | Trained on normal rows only |
|---|---|---|
| max z-score | 0.989 | 0.994 |
| Mahalanobis | 0.983 | 0.995 |
| Gaussian mixture (3) | 0.988 | 0.999 |
| k-NN distance (k=5) | 0.731 | 0.999 |
| [[Local Outlier Factor]] | 0.515 | 0.998 |
| One-Class SVM | 0.845 | 0.999 |
| [[Isolation Forest]] | **0.998** | 0.997 |
| [[Autoencoder]] (9-4-9) | 0.868 | 0.989 |

- **What this says:**
  - With clean "normal only" training data, **almost every detector scores 0.99 or more**. On this data set the choice hardly matters.
  - With a training set that still contains the odd rows (the unsupervised, realistic case), the neighbour-based detectors **collapse** (LOF 0.515 is chance level; k-NN 0.731) while the isolation forest and the simple statistical scores stay near the top.
  - A reason I tested: 93% of the anomalies belong to one class (class 5: 3,267 of 3,511 rows) that forms a big crowd. In a crowd, every anomaly has many anomalies as neighbours, so "few neighbours nearby" is false. With the neighbour count raised above the crowd's size LOF recovered (k=500: AUC 0.985; k=20: 0.515). See [[Local Outlier Factor]].
- **Limits of the shoot-out:** one data set, whose "anomalies" are other classes of a labelled problem, not natural fraud or faults; default or chosen settings with no tuning; the "clean" set-up uses the true labels to remove anomalies, which you cannot do in real life.

## 🔴 4. Deep / formal
**Z-score.** How many standard deviations a value is from the average of normal data:

$$
z=\frac{x-\mu}{\sigma}
$$

**Where:**
- $x$ is the value being checked. $\mu$ is the average of the normal data. $\sigma$ is its standard deviation, the usual size of the spread ([[Variance]], [[Normal Distribution]]).
- $z$ is the distance from the average in units of that spread; the sign says above or below.
- **Matters:** the division by $\sigma$ makes scores of different columns comparable (a 20 degree gap is large for a thermostat and small for a stock price). Without it the column with the biggest units decides everything. It also assumes one column at a time and a roughly bell-shaped spread; and the average and $\sigma$ are themselves distorted if odd points are in the data used to compute them.

**Example (tiny numbers):** normal readings average $\mu=50$ with $\sigma=5$. A new reading is $x=70$.
- $z=(70-50)/5=\mathbf{4}$.
- Meaning: the reading is 4 standard deviations above normal. For a bell-shaped spread a gap that large is very rare (by the usual rule of thumb, beyond 3 is already unusual), so it gets flagged.

**Mahalanobis distance.** The same idea for several columns at once, taking their correlation into account. With $\boldsymbol\mu$ the average vector and $\Sigma$ the covariance table of the normal data ([[Covariance]]):

$$
d^{2}(\mathbf x)=(\mathbf x-\boldsymbol\mu)^{\top}\Sigma^{-1}(\mathbf x-\boldsymbol\mu)
$$

**Where:**
- $\mathbf x$ is the point (a list of numbers), $\boldsymbol\mu$ the list of column averages.
- $\Sigma$ is the table of variances (diagonal) and covariances (off-diagonal) of the normal data; $\Sigma^{-1}$ is its matrix inverse ([[Matrix]]).
- $d^2$ is the squared distance. With $\Sigma=I$ it is the ordinary squared distance.
- **Matters:** $\Sigma^{-1}$ makes directions along which the data **varies a lot** cheap and directions along which it hardly varies expensive. Without it a point that breaks the pattern (up where it should go down) looks no farther away than one that follows it.

**Example (tiny numbers):** two measurements with average $(0,0)$, variance 1 each and correlation 0.8, so $\Sigma=\begin{pmatrix}1&0.8\\0.8&1\end{pmatrix}$ and $\Sigma^{-1}=\tfrac{1}{0.36}\begin{pmatrix}1&-0.8\\-0.8&1\end{pmatrix}$ (my arithmetic in Python).
- Point $(1,1)$ follows the pattern. $d^2=\tfrac{1}{0.36}(1-0.8-0.8+1)=0.4/0.36=\mathbf{1.11}$.
- Point $(1,-1)$ breaks it. $d^2=\tfrac{1}{0.36}(1+0.8+0.8+1)=3.6/0.36=\mathbf{10}$.
- Both are the same ordinary distance $\sqrt2=1.41$ from the average, but the pattern-breaking point is 9 times farther in Mahalanobis terms.

**Things that decide whether a detector works.**
- **Contamination and masking.** If odd rows are in the training data and form a crowd, density and neighbour methods treat the crowd as normal (my run: LOF 0.515 on a mix, 0.998 on clean data).
- **The cutoff is a business decision, not a maths one.** Raise it and you miss real problems. Lower it and people ignore alerts. Fix it by the number of checks you can afford (top $k$) or by a measured false-alarm rate.
- **Evaluation needs labels,** which you usually do not have. A small labelled set, even 20 known problems, is worth more than any amount of tuning on a score.
- **Drift.** Normal changes over time; a detector trained last year may flag everything this year ([[Data Drift]]).
- **Contextual anomalies** need the context as an input column (hour, season) or a forecast ([[Time Series]]).

**What came before, and how this improved on it:**
- **Before:** rules and statistical tests on one column at a time: control charts, the Grubbs outlier test (1969) and the Mahalanobis distance (1936); neighbour methods and Local Outlier Factor (2000); one-class SVM (about 1999 to 2001); Isolation Forest (2008). All years from memory, not checked. A well-known survey by Chandola, Banerjee and Kumar (2009) organises the field (from memory).
- **Its limit:** one-column rules miss broken relationships; distance methods cost $O(n^2)$ and break on dense crowds of anomalies; model-based ones need the right shape of normal.
- **What changed:** methods that need no labels and no model of the shape of normal, such as isolating points with random cuts.
- **How much better:** measured here, on a training set with the odd rows still in it, the best simple statistical score reached ROC AUC 0.989 and the isolation forest 0.998, while LOF fell to 0.515. Average precision (area under the precision-recall curve, a score that punishes false alarms) was 0.903 for the max z-score and 0.979 for the isolation forest. One data set, 5 splits, no tuning. I have no published figure to quote.

## Related
- [[Isolation Forest]] · [[Local Outlier Factor]] · [[Autoencoder]] · [[Gaussian Mixture Model]] · [[Support Vector Machine]] · [[k-NN]] · [[Unsupervised Learning]] · [[Time Series]] · [[Forecast Metrics]] · [[Data Drift]] · [[ROC Curve and AUC]] · [[Precision-Recall Curve]] · [[Covariance]] · [[Normal Distribution]] · [[Variance]] · [[Feature Engineering]] · [[Anomaly Detection Shoot-out]]

## Sources
- My own experiment, [[Anomaly Detection Shoot-out]] (code, tables and run details there), and the toy above. Data: OpenML "shuttle" version 1 (downloaded through scikit-learn); I used the ODDS convention (class 1 normal, classes 2, 3, 5, 6, 7 anomalies, class 4 dropped), which I know from memory and did not check against ODDS.
- History and survey names from memory, not checked. Not read: any anomaly detection paper.
- Image: Mbt19, [Normalized Anomaly Scores of Isolation Forest.png](https://commons.wikimedia.org/wiki/File:Normalized_Anomaly_Scores_of_Isolation_Forest.png), CC BY 3.0.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/anomaly-detection Anomaly detection:::Finding the few items that do not look like the rest, usually by learning what normal looks like and scoring how far each item is from it.

#flashcards/ml/reverse #flashcards/topic/anomaly-detection Point, contextual and collective anomaly:::Point: one value is strange. Contextual: normal in one context and odd in another (30 degrees in July or January). Collective: each reading is fine but the sequence is not.

#flashcards/ml/concept #flashcards/topic/anomaly-detection Why is accuracy a bad score for an anomaly detector?
?
Anomalies are rare, so a detector that always says "normal" is almost always right (93% when 7% are odd) and finds nothing. Use ROC AUC, average precision or precision in the top k.

#flashcards/ml/concept #flashcards/topic/anomaly-detection What is the difference between unsupervised and semi-supervised (one-class) anomaly detection?
?
Unsupervised: the training data is a mix of normal and a few hidden odd items. Semi-supervised: the training data is known to be normal only.

#flashcards/ml/image #flashcards/topic/anomaly-detection ![[anomaly-isolation-forest-scores.png]]
What do the colours show in this figure?
?
An anomaly score map over two measurements (eruption time and waiting time). Pale yellow means looks normal (the two dense groups of black dots), dark red means most anomalous (empty corners). Author Mbt19, CC BY 3.0. The colour meaning was read from the picture, which has no legend.

#flashcards/ml/formula #flashcards/topic/anomaly-detection What is the z-score formula?
?
$$
z=\frac{x-\mu}{\sigma}
$$
- $x$ is the value, $\mu$ the average of normal data, $\sigma$ its standard deviation.
- $z$ is the distance from the average in units of spread.
- Dividing by $\sigma$ makes columns comparable. It looks at one column at a time and is distorted if odd points are in the data used for $\mu$ and $\sigma$.

#flashcards/ml/concept #flashcards/topic/anomaly-detection Normal readings average 50 with standard deviation 5. What is the z-score of 70?
?
(70 − 50) / 5 = 4: four standard deviations above normal, which is rare for a bell-shaped spread. (My arithmetic.)

#flashcards/ml/formula #flashcards/topic/anomaly-detection What is the Mahalanobis distance?
?
$$
d^{2}(\mathbf x)=(\mathbf x-\boldsymbol\mu)^{\top}\Sigma^{-1}(\mathbf x-\boldsymbol\mu)
$$
- $\mathbf x$ is the point, $\boldsymbol\mu$ the average vector, $\Sigma$ the covariance table of the normal data.
- $\Sigma^{-1}$ makes directions where the data varies a lot cheap and directions where it hardly varies expensive.
- Without it a point that breaks the pattern looks no farther than one that follows it. With $\Sigma=I$ it is the ordinary squared distance.

#flashcards/ml/concept #flashcards/topic/anomaly-detection With average (0,0), variances 1 and correlation 0.8, what are the squared Mahalanobis distances of (1,1) and (1,−1)?
?
Σ⁻¹ is (1/0.36) times the table with 1 on the diagonal and −0.8 off it, so (1,1) gives 0.4/0.36 = 1.11 and (1,−1) gives 3.6/0.36 = 10. Both are 1.41 away in ordinary distance, but the pattern-breaking point is 9 times farther. (My arithmetic.)

#flashcards/ml/concept #flashcards/topic/anomaly-detection In the toy with 5 pattern-breaking points, how did max z-score, Mahalanobis and isolation forest rank them?
?
Max z-score: ranks 25 to 156 (AUC 0.864). Mahalanobis: ranks 1 to 5 (AUC 1.000). Isolation forest: ranks 7 to 26 (AUC 0.973). Hand-made toy of 505 points, one run.

#flashcards/ml/concept #flashcards/topic/anomaly-detection On the Shuttle data, what happened to the detectors when the training set still contained the odd rows?
?
Isolation forest stayed at ROC AUC 0.998 and the simple z-score at 0.989, but LOF fell to 0.515 (chance) and k-NN distance to 0.731. With clean normal-only training almost all detectors scored 0.989 to 0.999.

#flashcards/ml/concept #flashcards/topic/anomaly-detection Why did neighbour-based detectors fail on the contaminated Shuttle data?
?
93% of the anomalies (3,267 of 3,511) are one class that forms a big crowd, so each anomaly has many anomalies as neighbours. Raising LOF's neighbour count above the crowd's size recovered it (k=500: AUC 0.985; k=20: 0.515).

#flashcards/ml/concept #flashcards/topic/anomaly-detection Name two limits of the anomaly shoot-out.
?
One data set whose anomalies are classes of a labelled problem (not natural faults), and no tuning; the clean set-up uses true labels to remove anomalies, which is impossible in real life.

#flashcards/ml/cloze #flashcards/topic/anomaly-detection The cutoff of an anomaly detector is a ==business decision==: raise it and you miss real problems, lower it and people ignore ==alerts==.
