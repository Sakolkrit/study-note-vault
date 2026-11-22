---
tags: [machine-learning, evaluation, metrics, precision, recall, classification]
status: growing
level: 4
tier: 2
---

# Precision-Recall Curve

The **precision-recall curve** shows how a scoring model's [[Precision]] and [[Recall]] trade against each other at every threshold. Its area is called **average precision**. Back to the overview: [[Evaluation Metrics]].

## 🟢 1. Simple (age 12)
A model that gives each example a score has to choose a cut-off. A strict cut-off gives few alarms, and most of them are right, but it misses many real cases. A loose cut-off finds nearly every real case, but most of its alarms are false. The precision-recall curve is a picture of this trade for **every** cut-off.

**Example:** you are searching a beach for rare coins with a metal detector. On the most sensitive setting you find almost all the coins, but you also dig up thousands of bottle caps. On the strictest setting, nearly everything you dig up is a coin, but you leave many coins behind. The curve records both numbers for every dial setting.

## 🔵 2. How it works
1. The model gives each test example a **score**. Higher means more likely yes.
2. Choose a threshold: say yes if the score is at least that value. Build the [[Confusion Matrix]].
3. Compute [[Precision]] (how many of the yes-answers were real) and [[Recall]] (how many of the real cases were found).
4. Plot one point: recall across, precision up.
5. Move the threshold from very strict to very loose. The points join into the curve.
6. A curve that stays high and far to the right is good. A curve that drops quickly means finding more real cases costs a flood of false alarms.
7. The **average precision** is one number for the whole curve: roughly, the average of the precision at each real case that is found.
8. **The catch:** the curve (and the average precision) depends on how rare the positives are. A random model's average precision equals the share of positives, so compare with that line, not with 0.5.

![[precision-recall-curve.png]]
*A precision-recall curve. Recall runs across and precision up, and the colour of each point shows the threshold that produced it (the colour bar on the right). The red dot marks the threshold with the best F-score ([[F1 Score]]). I read the axes and colours off the picture; I do not use the number in its legend, which the file text does not explain. Image: Biggerj1, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:PR_curve_with_optimal_fscore.png), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Compared with ROC:** the [[ROC Curve and AUC]] plots recall against the false positive rate. When the negatives are very many, a small false positive rate is still a lot of false alarms, and the ROC curve hides that. Precision counts the false alarms against the alarms, so the precision-recall curve shows it. This is the reason it is usually recommended for rare positives (a common recommendation; the references argue it).
- **Baseline:** a model with no skill has a flat curve at the height of the share of positives. The code below has 1% positives, so the baseline is about 0.01.
- **Average precision:** the area-like summary of the curve, a sum of precision values weighted by how much the recall grew at each step. It is not the same as the plain area under a straight-line join of the points.
- **Choosing a threshold:** pick the point that suits your costs, or the one with the best [[F1 Score]] (the red dot in the picture).
- **Not a calibration check:** it is about ranking and trade-offs. Whether the scores are good probabilities is a separate question ([[Calibration]]).
- **Many classes:** one curve per class against the rest, then average.

```python
import numpy as np
from sklearn.metrics import precision_recall_curve
y = np.array([1, 1, 0, 1, 0, 0, 1, 0]); score = np.array([.9, .8, .7, .6, .5, .3, .4, .2])
P, R, T = precision_recall_curve(y, score)
for p, r, t in zip(P, R, np.r_[T, np.inf]): print(f"threshold {t:>4}: precision {p:.3f}  recall {r:.2f}")
```

## 🔴 4. Deep / formal
**Average precision.** Sort the examples by score from highest to lowest. Every time the threshold drops past a new real positive, the recall grows a little. Average precision adds up the precision at those moments, weighted by how much the recall grew:

$$
\text{AP}=\sum_{n}\big(R_n-R_{n-1}\big)\,P_n
$$

**Where:**
- $n$ numbers the thresholds, from strict to loose. $P_n$ and $R_n$ are the precision and recall at threshold number $n$, and $R_0=0$.
- $R_n-R_{n-1}$ is how much the recall grew at this step. It is zero when the step added only false alarms, and $1/(\text{number of real positives})$ when it added one real positive.
- **Matters:** only steps that find a new real positive count. So the number is the average of the precision measured at each real positive found, which is why a model that ranks the real positives first scores near 1.

**Example (tiny numbers):** four real positives with scores $0.9,\,0.8,\,0.6,\,0.4$ and four real negatives with scores $0.7,\,0.5,\,0.3,\,0.2$ (the same scores as in [[ROC Curve and AUC]]). Sorted by score, the real positives are the 1st, 2nd, 4th and 6th of the 8.
- After the 1st example: precision $1/1=\mathbf{1}$. After the 2nd: $2/2=\mathbf{1}$. After the 4th: $3/4=\mathbf{0.75}$. After the 6th: $4/6=\mathbf{0.667}$.
- Each real positive adds $1/4$ to the recall. So $\text{AP}=\tfrac14(1+1+0.75+0.667)=\mathbf{0.854}$.
- Meaning: on average, at the moments the model found a real case, 85% of its alarms so far were real. The same scores gave an ROC AUC of $0.8125$.

**Further notes:**
- **Interpolation:** some books smooth the curve before taking the area, which gives slightly different numbers (from memory). State which one you use.
- **Why 0.5 is not the baseline:** the baseline is the share of positives, so an AP of 0.26 can be good when only 1% are positive (see the check).

**What came before, and how this improved on it:**
- **Before:** the [[ROC Curve and AUC]]: recall against the false positive rate, with AUC as the summary.
- **Its limit:** when negatives vastly outnumber positives, a model can have a high AUC while most of its alarms are false, because a tiny false positive rate still means many false alarms.
- **What changed:** plot [[Precision]] against [[Recall]] instead, which counts the false alarms against the alarms. The precision-recall curve comes from information retrieval (from memory, not checked), and Saito and Rehmsmeier argued in 2015 that it is more informative than ROC for imbalanced data (the paper title is from memory; I did not open it).
- **How much better:** in the code below (100,000 cases, 987 of them positive), the same scores have an ROC AUC of 0.921, which looks strong, and an average precision of only 0.264. The curve says why: to find 50% of the positives the precision is 0.156, for 80% it is 0.062, and for 95% it is 0.026. A tiny example gives an AP of 0.8542, equal to scikit-learn's value.

**Check it yourself:**

```python
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
rng = np.random.default_rng(0)
n = 100000; y = (rng.random(n) < 0.01).astype(int)                   # 1% positives
score = np.where(y == 1, rng.normal(2.0, 1.0, n), rng.normal(0.0, 1.0, n))
print(f"{n:,} cases, {y.sum():,} positives (1%). Same scores, two summaries:")
print(f"  ROC AUC: {roc_auc_score(y, score):.3f}   average precision (area under the precision-recall curve): {average_precision_score(y, score):.3f}")
P, R, T = precision_recall_curve(y, score)
for r in (0.5, 0.8, 0.95):
    i = np.where(R >= r)[0][-1]
    print(f"  to find {r:.0%} of the positives, precision is only {P[i]:.3f}")
pos = np.array([.9, .8, .6, .4]); neg = np.array([.7, .5, .3, .2]); s = np.r_[pos, neg]; yy = np.r_[np.ones(4), np.zeros(4)]
order = np.argsort(-s); hits = np.cumsum(yy[order]); prec = hits / np.arange(1, 9)
ap = sum(prec[k] for k in range(8) if yy[order][k]) / 4
print("tiny example: precision at each correct hit", np.round([prec[k] for k in range(8) if yy[order][k]], 3), "-> average precision", round(ap, 4), " scikit-learn:", round(average_precision_score(yy, s), 4))
```

## Related
- [[Evaluation Metrics]] · [[Precision]] · [[Recall]] · [[F1 Score]] · [[ROC Curve and AUC]] · [[Confusion Matrix]] · [[Calibration]] · [[Classification]] · [[Retrieval Metrics]] · [[Object Detection]]
- [[Anomaly Detection]]
- [[Mean Average Precision]]
## Sources
- Saito & Rehmsmeier, *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*, 2015 (title from memory; not opened).
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online).
- Image: [PR_curve_with_optimal_fscore.png](https://commons.wikimedia.org/wiki/File:PR_curve_with_optimal_fscore.png) by Biggerj1, CC BY-SA 4.0.
- Every number was computed by the code in this note (toy data, NumPy and scikit-learn).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation Precision-recall curve:::A picture of the trade between precision and recall for every cut-off. A strict cut-off gives few, mostly right alarms but misses many cases. A loose one finds nearly every case but most alarms are false.

#flashcards/ml/formula #flashcards/topic/evaluation Write the average precision (AP) of a precision-recall curve and explain each symbol.
?
$$\text{AP}=\sum_{n}\big(R_n-R_{n-1}\big)\,P_n$$
- $n$: numbers the thresholds, from strict to loose.
- $P_n$, $R_n$: the precision and recall at threshold number $n$, and $R_0=0$.
- $R_n-R_{n-1}$: how much the recall grew at this step. It is zero when the step added only false alarms, and $1/(\text{number of real positives})$ when it added one real positive.
- AP: the average precision, a single number for the whole curve.

#flashcards/ml/cloze #flashcards/topic/evaluation On a precision-recall curve a strict cut-off gives ==high== precision and ==low== recall.

#flashcards/ml/example #flashcards/topic/evaluation Positives score 0.9, 0.8, 0.6, 0.4 and negatives 0.7, 0.5, 0.3, 0.2. What is the precision after each real positive is found, and the AP?::Sorted by score the positives are the 1st, 2nd, 4th and 6th of 8. Precision at those points: 1/1 = 1, 2/2 = 1, 3/4 = 0.75, 4/6 = 0.667. Each adds a recall step of 0.25, so AP = 0.25 × (1 + 1 + 0.75 + 0.667) = 0.854 (my arithmetic).

#flashcards/ml/image #flashcards/topic/evaluation ![[precision-recall-curve.png]]
What does this picture show?
?
A precision-recall curve. Recall runs across and precision up, and the colour of each point shows the threshold that produced it. The red dot marks the threshold with the best F-score.
