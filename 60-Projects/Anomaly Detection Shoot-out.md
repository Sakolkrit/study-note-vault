---
tags: [project, anomaly-detection, isolation-forest, local-outlier-factor, scikit-learn, shuttle]
status: done
stack: Python 3.9.6, scikit-learn 1.1.2, numpy 1.24.4, scipy 1.8.1, OpenML "shuttle" data
goal: compare 8 anomaly detectors on one data set, trained on a mix and trained on clean data, and find out why one of them failed
---

# Anomaly Detection Shoot-out

> **In one line:** score eight anomaly detectors on the Shuttle data twice (once with training data that still holds the odd rows, once with normal rows only), and find out **which ones break when the training data is contaminated**, which is the realistic case.

## Goal
Done means: one script, `shootout.py`, that prints ROC AUC, average precision and precision-at-k for each detector in both set-ups, plus a second script, `lof_k.py`, that tests why one detector failed, so the claims in [[Anomaly Detection]], [[Isolation Forest]] and [[Local Outlier Factor]] come from numbers I measured.

## Result in short
- **Data:** 49,097 rows, 9 numbers each, 3,511 anomalies (7.15%). Normal is class 1 (45,586 rows). The anomalies are classes 2, 3, 5, 6, 7 (50, 171, 3,267, 10 and 13 rows). Class 4 (8,903 rows) is dropped, as in the common ODDS copy of this data (I know that convention from memory and did not check it against ODDS).
- **Protocol:** 5 random 50/50 splits. A detector is fitted on the training half and scores the test half. *Contaminated:* training half as it is (about 7% odd rows, no labels used). *Clean:* the normal rows of the training half only. The numbers are means over the 5 splits, with the standard deviation in brackets.
- **Contaminated training data (ROC AUC / average precision / precision at k, k = the true number of anomalies in the test half):**

| Detector | ROC AUC | Avg. precision | Precision@k |
|---|---|---|---|
| max z-score | 0.989 (0.001) | 0.903 | 0.935 |
| Mahalanobis | 0.983 (0.001) | 0.873 | 0.863 |
| Gaussian mixture (3) | 0.988 (0.004) | 0.717 | 0.805 |
| k-NN distance (k=5) | 0.731 (0.009) | 0.196 | 0.204 |
| Local Outlier Factor | 0.515 (0.009) | 0.124 | 0.132 |
| One-Class SVM | 0.845 (0.016) | 0.336 | 0.451 |
| **Isolation Forest** | **0.998 (0.000)** | **0.979** | **0.950** |
| Autoencoder (9-4-9) | 0.868 (0.083) | 0.474 | 0.531 |

- **Clean training data:** every detector scored ROC AUC 0.989 to 0.999. Average precision: Gaussian mixture 0.979 and isolation forest 0.974 on top; the rest 0.913 to 0.952. So **the detector hardly matters when the training data is clean, and matters a lot when it is not.**
- **Why LOF failed on the mixed data (tested):** 93% of the anomalies (3,267 of 3,511) are one class, a big crowd. Each of them has many anomalies as neighbours. With the neighbour count $k$ raised above the crowd's size, LOF's AUC went from 0.515 ($k=20$) to 0.985 ($k=500$) and 0.993 ($k=1500$), on split 0.
- **Honest limits:** one data set, whose "anomalies" are other classes of a labelled problem and not natural faults; no tuning; the "clean" set-up uses the true labels to remove anomalies, which you cannot do in practice; one split for the $k$ test; the autoencoder's AUC varies a lot between splits (0.083) and printed convergence warnings.

## What you need
- Python 3.9.6 with `numpy` 1.24.4, `scipy` 1.8.1 and `scikit-learn` 1.1.2 (versions of my machine).
- Internet for the first run: scikit-learn downloads the data set "shuttle" (version 1) from OpenML (about 30 seconds on my connection) and caches it in a folder `openml` next to the script.

## Steps
1. **Make an empty folder** and save the script below as `shootout.py`. What it does, in order:
   - Loads the data and keeps the rows that are not class 4; the label is 1 for classes 2, 3, 5, 6, 7 and 0 for class 1.
   - Defines eight detectors, each returning a score where higher means more anomalous: the largest absolute z-score over the 9 columns; the squared Mahalanobis distance to the mean; minus the log-likelihood under a 3-component Gaussian mixture ([[Gaussian Mixture Model]]); the distance to the 5th nearest training row ([[k-NN]]); [[Local Outlier Factor]] with $k=20$ in novelty mode; a One-Class SVM (`nu=0.05`) ([[Support Vector Machine]]); an [[Isolation Forest]] with 100 trees; and a small [[Autoencoder]] (9-4-9, scikit-learn's `MLPRegressor` trained to copy its input; the score is the squared copying error).
   - The three methods with quadratic cost (k-NN, LOF, One-Class SVM) are fitted on a random sample of 6,000 training rows.
   - Standardizes the columns with the training half's mean and spread ([[Feature Scaling]]), runs 5 splits in each set-up, and prints the mean and the standard deviation of ROC AUC, average precision, precision at k, and the seconds per fit.

```python
# file: shootout.py
"""Anomaly detection shoot-out on the Shuttle data (OpenML 'shuttle', version 1).
Normal = class 1. Anomaly = classes 2, 3, 5, 6, 7 (class 4 dropped, the same rule the ODDS copy of Shuttle uses).
Two set-ups, 5 random splits each:
  contaminated : the detector is fitted on a training half that still contains its ~7% anomalies (no labels used)
  clean        : the detector is fitted on the normal rows of the same training half only
Scores are always computed on the held-out half. Higher score = more anomalous."""
import sys, time
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

d = fetch_openml("shuttle", version=1, as_frame=False, data_home=".")
X, cls = d.data.astype(float), d.target.astype(int)
keep = cls != 4
X, cls = X[keep], cls[keep]
y = (cls != 1).astype(int)                                   # 1 = anomaly
print(f"rows {len(y)} | anomalies {y.sum()} ({100*y.mean():.2f}%)")

BIG = 6000                                                   # cap on rows given to the slow, quadratic-cost detectors


def zmax(Xtr, Xte):                                          # largest |z-score| over the 9 columns
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-9
    return np.abs((Xte - m) / s).max(1)


def mahal(Xtr, Xte):                                         # squared Mahalanobis distance to the training mean
    m = Xtr.mean(0); C = np.cov(Xtr.T) + 1e-6 * np.eye(Xtr.shape[1]); P = np.linalg.inv(C)
    D = Xte - m
    return np.einsum("ij,jk,ik->i", D, P, D)


def iforest(Xtr, Xte, seed):
    return -IsolationForest(n_estimators=100, random_state=seed).fit(Xtr).score_samples(Xte)


def lof(Xtr, Xte, seed):
    rs = np.random.RandomState(seed); Xs = Xtr[rs.choice(len(Xtr), min(BIG, len(Xtr)), replace=False)]
    return -LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Xs).score_samples(Xte)


def knn(Xtr, Xte, seed):                                     # distance to the 5th nearest training row
    rs = np.random.RandomState(seed); Xs = Xtr[rs.choice(len(Xtr), min(BIG, len(Xtr)), replace=False)]
    return NearestNeighbors(n_neighbors=5).fit(Xs).kneighbors(Xte)[0][:, -1]


def ocsvm(Xtr, Xte, seed):
    rs = np.random.RandomState(seed); Xs = Xtr[rs.choice(len(Xtr), min(BIG, len(Xtr)), replace=False)]
    return -OneClassSVM(gamma="scale", nu=0.05).fit(Xs).score_samples(Xte)


def gmm(Xtr, Xte, seed):
    return -GaussianMixture(n_components=3, covariance_type="full", random_state=seed).fit(Xtr).score_samples(Xte)


def autoenc(Xtr, Xte, seed):                                 # small autoencoder 9-4-9; score = squared reconstruction error
    net = MLPRegressor(hidden_layer_sizes=(4,), activation="relu", max_iter=200, random_state=seed).fit(Xtr, Xtr)
    return ((net.predict(Xte) - Xte) ** 2).sum(1)


METHODS = {"max |z-score|": lambda a, b, s: zmax(a, b), "Mahalanobis": lambda a, b, s: mahal(a, b),
           "Gaussian mixture (3)": gmm, "k-NN distance (k=5)": knn, "Local Outlier Factor": lof,
           "One-Class SVM": ocsvm, "Isolation Forest": iforest, "Autoencoder (9-4-9)": autoenc}

if __name__ == "__main__":
    REPEATS = 5
    res = {}
    for setup in ("contaminated", "clean"):
        for r in range(REPEATS):
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=r)
            if setup == "clean": Xtr = Xtr[ytr == 0]
            sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
            k = int(yte.sum())                               # top-k with k = the true number of anomalies
            for name, fn in METHODS.items():
                t = time.time(); s = fn(Xtr_s, Xte_s, r); dt = time.time() - t
                top = np.argsort(-s)[:k]
                res.setdefault((setup, name), []).append((roc_auc_score(yte, s), average_precision_score(yte, s), yte[top].mean(), dt))
            print(f"split {r} {setup} done", file=sys.stderr, flush=True)
    for setup in ("contaminated", "clean"):
        print(f"\n{setup}: mean (std) over {REPEATS} splits")
        print(f"{'method':24s} {'ROC AUC':>14s} {'avg precision':>14s} {'precision@k':>14s} {'sec':>6s}")
        for name in METHODS:
            a = np.array(res[(setup, name)])
            print(f"{name:24s} {a[:,0].mean():7.3f} ({a[:,0].std():.3f}) {a[:,1].mean():7.3f} ({a[:,1].std():.3f}) {a[:,2].mean():7.3f} ({a[:,2].std():.3f}) {a[:,3].mean():6.1f}")
```

2. **Run it:** `python shootout.py`. It took about 4 minutes on my laptop (by the file times of the output). Output:

```text
rows 49097 | anomalies 3511 (7.15%)

contaminated: mean (std) over 5 splits
method                          ROC AUC  avg precision    precision@k    sec
max |z-score|              0.989 (0.001)   0.903 (0.009)   0.935 (0.006)    0.0
Mahalanobis                0.983 (0.001)   0.873 (0.010)   0.863 (0.003)    0.0
Gaussian mixture (3)       0.988 (0.004)   0.717 (0.034)   0.805 (0.077)    0.9
k-NN distance (k=5)        0.731 (0.009)   0.196 (0.006)   0.204 (0.013)    2.0
Local Outlier Factor       0.515 (0.009)   0.124 (0.007)   0.132 (0.019)    3.2
One-Class SVM              0.845 (0.016)   0.336 (0.026)   0.451 (0.026)    1.4
Isolation Forest           0.998 (0.000)   0.979 (0.004)   0.950 (0.020)    2.2
Autoencoder (9-4-9)        0.868 (0.083)   0.474 (0.216)   0.531 (0.252)    7.9

clean: mean (std) over 5 splits
method                          ROC AUC  avg precision    precision@k    sec
max |z-score|              0.994 (0.000)   0.937 (0.008)   0.956 (0.005)    0.0
Mahalanobis                0.995 (0.000)   0.945 (0.010)   0.960 (0.004)    0.0
Gaussian mixture (3)       0.999 (0.000)   0.979 (0.011)   0.974 (0.002)    3.8
k-NN distance (k=5)        0.999 (0.000)   0.943 (0.007)   0.966 (0.004)    2.5
Local Outlier Factor       0.998 (0.001)   0.913 (0.016)   0.960 (0.004)    4.1
One-Class SVM              0.999 (0.000)   0.952 (0.008)   0.967 (0.004)    1.8
Isolation Forest           0.997 (0.000)   0.974 (0.006)   0.938 (0.024)    2.9
Autoencoder (9-4-9)        0.989 (0.002)   0.949 (0.010)   0.968 (0.005)   14.3
```

3. **Read it.** In the clean set-up the table is almost flat. In the contaminated set-up the neighbour-based detectors (k-NN distance, LOF) and the One-Class SVM and autoencoder drop, and the isolation forest and the simple statistical scores stay on top. The seconds column is not reliable: the laptop was busy with other jobs.
4. **Test the explanation for LOF.** Save the next script as `lof_k.py` in the same folder (it imports `shootout.py`, which prints its first line again), and run `python lof_k.py`. It counts the rows per anomaly class, then fits LOF on the 6,000-row sample of split 0's mixed training half with $k$ = 5, 20, 100, 500 and 1500:

```python
# file: lof_k.py
"""Why did Local Outlier Factor fail on the contaminated training set? Test the 'anomalies form their own crowd' idea:
count the anomaly classes, then try larger neighbour counts k on split 0 (contaminated set-up)."""
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from shootout import X, y, cls

print("rows per anomaly class:", {int(c): int((cls == c).sum()) for c in (2, 3, 5, 6, 7)}, "| normal class 1:", int((cls == 1).sum()))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
rs = np.random.RandomState(0); Xs = Xtr[rs.choice(len(Xtr), 6000, replace=False)]   # the same 6,000-row sample size the shoot-out used
print(f"sample of {len(Xs)} training rows")
for k in (5, 20, 100, 500, 1500):
    s = -LocalOutlierFactor(n_neighbors=k, novelty=True).fit(Xs).score_samples(Xte)
    print(f"k = {k:5d}   ROC AUC on the test half = {roc_auc_score(yte, s):.3f}")
```

```text
rows 49097 | anomalies 3511 (7.15%)
rows per anomaly class: {2: 50, 3: 171, 5: 3267, 6: 10, 7: 13} | normal class 1: 45586
sample of 6000 training rows
k =     5   ROC AUC on the test half = 0.516
k =    20   ROC AUC on the test half = 0.515
k =   100   ROC AUC on the test half = 0.434
k =   500   ROC AUC on the test half = 0.985
k =  1500   ROC AUC on the test half = 0.993
```

5. **What the result supports:** LOF with a small $k$ treats a big crowd of anomalies as normal. With $k$ above the crowd's size (about 400 sampled rows, my arithmetic from the class counts) it works. At $k=100$ the AUC is below chance (0.434), which I did not investigate.

## What went wrong, and the fix
- **The first idea was only a hunch.** I expected the neighbour methods to suffer on a mixed training set, but I did not know why LOF was *exactly* at chance. The class counts and the $k$ sweep were added to test the "crowd" guess. They support it; they do not prove it, since I tried one split and one crowd.
- **Convergence warnings.** The autoencoder printed 3 "Maximum iterations (200) reached" warnings. I did not raise the limit, so its numbers (especially the large spread, 0.083) may come from under-trained networks. That would be a fault of my set-up, not proof that autoencoders are bad detectors.
- **The OpenML download** took about 30 seconds the first time and works offline afterwards because of the cache.
- **Nothing else broke.** The script ran the first time.

## What I did not run
- **Any other data set** (credit-card fraud, KDD, time series).
- **Tuning** of any detector's settings (number of trees, `nu`, $k$, hidden size, mixture size).
- **More than one split** for the LOF $k$ sweep, and no significance tests.
- **Reliable timing:** the "sec" column is from a busy laptop.
- **Thresholds and alerts:** I report ranking scores only; no cutoff was chosen.
- **Time-series anomalies, deep detectors** beyond the tiny autoencoder, and the *Extended* Isolation Forest.

## Re-run check
Both scripts were extracted from this note into a fresh folder and run again with a copy of the cached data. Every score matched the output above. Only the seconds column changed (for example the autoencoder took 12.0 and 11.5 seconds instead of 7.9 and 14.3), which confirms that timing is unreliable here.

## Concepts used
- [[Anomaly Detection]] · [[Isolation Forest]] · [[Local Outlier Factor]] · [[Gaussian Mixture Model]] · [[Autoencoder]] · [[Support Vector Machine]] · [[k-NN]] · [[Feature Scaling]] · [[ROC Curve and AUC]] · [[Precision-Recall Curve]] · [[Covariance]]

## Next ideas
- Repeat on a second data set with natural anomalies (for example a credit-card fraud set).
- Mix in anomalies at several rates (1%, 3%, 7%, 15%) and see where each detector breaks.
- Raise the autoencoder's iteration limit and tune $k$ for the neighbour methods on a small labelled set.
- Try the Extended Isolation Forest.

## Flashcards
#flashcards/projects

#flashcards/projects/concept #flashcards/topic/project-classic-ml What was the setup of the Anomaly Detection Shoot-out?
?
Shuttle data from OpenML: 49,097 rows, 7.15% anomalies (classes 2, 3, 5, 6, 7 against normal class 1). Eight detectors, 5 random 50/50 splits, scored on the test half. Two training set-ups: contaminated (the mix as it is) and clean (normal rows only).

#flashcards/projects/concept #flashcards/topic/project-classic-ml What was the main finding of the Anomaly Detection Shoot-out?
?
With clean training data all detectors scored ROC AUC 0.989 to 0.999. With contaminated training data the neighbour-based detectors collapsed (LOF 0.515, k-NN distance 0.731) while the isolation forest (0.998) and simple z-score (0.989) stayed on top.

#flashcards/projects/concept #flashcards/topic/project-classic-ml Why did LOF fail on contaminated Shuttle data, and what is the evidence?
?
93% of the anomalies (3,267 of 3,511) are one crowd, so each anomaly's neighbours are other anomalies. With k = 500 or 1500 (above the crowd's size) LOF's AUC went from 0.515 to 0.985 or 0.993 (split 0, one run). Supported, not proved.

#flashcards/projects/concept #flashcards/topic/project-classic-ml Why might the autoencoder numbers in the shoot-out be unreliable?
?
It printed 3 convergence warnings (maximum 200 iterations reached) and its AUC varied a lot across splits (spread 0.083). I did not raise the iteration limit, so it may be under-trained.

#flashcards/projects/concept #flashcards/topic/project-classic-ml What is the difference between the "contaminated" and "clean" set-ups, and why is "clean" unrealistic?
?
Contaminated trains on the training half as it is (about 7% odd rows, no labels). Clean trains only on normal rows, which needs the true labels to remove the anomalies, something you cannot do in real life.

#flashcards/projects/concept #flashcards/topic/project-classic-ml Name three things the shoot-out did not run.
?
Any other data set; tuning of any detector's settings; more than one split for the LOF k sweep or any significance test; reliable timing; thresholds and alerts.

#flashcards/projects/cloze #flashcards/topic/project-classic-ml In the shoot-out the detector choice hardly mattered with ==clean== training data and mattered a lot with ==contaminated== training data.
