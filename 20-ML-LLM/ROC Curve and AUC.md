---
tags: [machine-learning, evaluation, metrics, ROC, AUC, classification]
status: growing
level: 4
tier: 2
---

# ROC Curve and AUC

The **ROC curve** shows how a scoring model does at every threshold at once, and **AUC** is the area under it, as one number. Back to the overview: [[Evaluation Metrics]].

## 🟢 1. Simple (age 12)
Many models do not just say yes or no. They give each example a score. You then pick a cut-off: "say yes if the score is above 0.7". The ROC curve is a picture of what happens for **every** possible cut-off, so you do not have to pick one first. AUC squeezes the picture into one number: 1.0 is perfect and 0.5 is a coin flip.

**Example:** A smoke alarm with a sensitivity dial. Turn it up and it catches more real fires but also rings for toast. The ROC curve records, for every dial setting, how many real fires were caught and how many false alarms there were.

## 🔵 2. How it works
1. The model gives each test example a **score**: higher means more likely yes.
2. Choose a **threshold**: say yes if the score is at least that value. Build the [[Confusion Matrix]] for it.
3. Compute two numbers: the **true positive rate** (the [[Recall]]: the share of real yes-cases found) and the **false positive rate** (the share of real no-cases wrongly called yes, see [[Specificity]]).
4. Plot that point: false positive rate across, true positive rate up.
5. Move the threshold from very high (say yes to nothing) to very low (say yes to everything). The points join into the **ROC curve**, from the bottom-left corner to the top-right corner.
6. A curve that bends towards the top-left corner is good. The diagonal is random guessing.
7. **AUC** is the area under the curve. It also equals the chance that the model gives a random real positive a higher score than a random real negative.
8. **The catch:** AUC only measures how well the model **ranks**, not whether its scores are good probabilities, and with very rare positives a high AUC can still go with poor [[Precision]].

![[roc-curve.svg]]
*ROC curves. Horizontal: the false positive rate (the share of real "no" cases wrongly called "yes"). Vertical: the true positive rate, which is the recall (the share of real "yes" cases found). Each curve shows one model across all thresholds. The dashed red diagonal is a random guesser, curves closer to the top-left corner are better, and the blue dot at the corner is a perfect classifier. Image: cmglee and MartinThoma, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Roc_curve.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Origin:** the name stands for "receiver operating characteristic", from radar signal detection (from memory, not checked).
- **Use it for:** comparing models without choosing a threshold, and for choosing a threshold by looking at the trade-off.
- **Reading AUC:** 0.5 means no skill, 1.0 means every positive is ranked above every negative. Below 0.5 means the ranking is backwards.
- **With rare positives:** look also at the precision-recall curve, which is usually more informative then (a common recommendation; the reasoning is in the references).
- **Not calibration:** a model can have a perfect AUC and still give poor probabilities ([[Evaluation Metrics]]).
- **Multi-class:** compute one curve for each class against the rest and average.

## 🔴 4. Deep / formal
**ROC and AUC: judging every threshold at once.** For a threshold on the model's score, two rates describe the result. Plotting them against each other as the threshold moves gives the ROC curve, and AUC measures it:

$$
\text{TPR}=\frac{TP}{TP+FN},\quad \text{FPR}=\frac{FP}{FP+TN},\quad \text{AUC}=\Pr\big(s_{\text{pos}}>s_{\text{neg}}\big)
$$

**Where:**
- TPR (true positive rate) is the share of real positives found, which is the recall. FPR (false positive rate) is the share of real negatives wrongly called positive. $TP$, $FP$, $FN$, $TN$ are the confusion-matrix counts at one threshold.
- $s_{\text{pos}}$ is the model's score for a randomly chosen real positive, and $s_{\text{neg}}$ the score for a randomly chosen real negative. $\Pr(\cdot)$ is a probability (see [[Probability]]). Ties count as half.
- **Matters:** AUC equals the area under the ROC curve (a standard equivalence, from memory) **and** the chance that the model scores a random positive above a random negative. So $\text{AUC}=0.5$ means "no better than a coin flip at ranking", and $\text{AUC}=1$ means "every positive is ranked above every negative".
- **Matters:** AUC does not depend on any threshold, and it measures only how well the model **ranks**, not whether its scores are well-calibrated probabilities.
- **Caution:** with very rare positives, a high AUC can still go with poor precision, which is why the precision-recall curve is also checked.

**Example (tiny numbers):** four real positives with scores $0.9,\,0.8,\,0.6,\,0.4$ and four real negatives with scores $0.7,\,0.5,\,0.3,\,0.2$.
- Threshold $0.65$ (say "yes" if the score is at least $0.65$): positives called yes: $0.9,0.8$, so TPR $=2/4=\mathbf{0.5}$. Negatives called yes: just $0.7$, so FPR $=1/4=\mathbf{0.25}$. That is one point on the curve.
- Lower the threshold to $0.45$: TPR $=\mathbf{0.75}$, FPR $=\mathbf{0.5}$. More real cases found, more false alarms.
- AUC by counting pairs: there are $4\times4=16$ (positive, negative) pairs. The positive $0.9$ beats all 4 negatives, $0.8$ beats 4, $0.6$ beats 3 ($0.5,0.3,0.2$), and $0.4$ beats 2 ($0.3,0.2$). Total $13$ out of $16$: $\text{AUC}=\mathbf{0.8125}$.
- Meaning: if you pick one real positive and one real negative at random, this model scores the positive higher about 81% of the time.

**Further notes:**
- **Choosing a point on the curve:** pick the threshold that matches your costs of a miss against a false alarm.
- **Ties:** examples with equal scores count as half a win in the pair-counting view.

**What came before, and how this improved on it:**
- **Before:** judge a classifier by its accuracy at one fixed threshold.
- **Its limit:** the number depends on the threshold chosen and on how common the positives are.
- **What changed:** plot the true positive rate against the false positive rate at every threshold, and summarise it with the area (AUC).
- **How much better:** in the code below, AUC by counting the 16 (positive, negative) pairs gives 0.8125, exactly what scikit-learn's `roc_auc_score` gives. Among 10,000 cases with 1% positives, a model that ranks every positive first scores an AUC of 1.0 and a model with pure-noise scores gets 0.469, near the 0.5 of a coin flip (it is not exactly 0.5 because only 89 positives are in the data).

**Check it yourself:**

```python
import numpy as np
from sklearn.metrics import roc_auc_score
pos = np.array([.9, .8, .6, .4]); neg = np.array([.7, .5, .3, .2])
pairs = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / 16
s = np.r_[pos, neg]; y = np.r_[np.ones(4), np.zeros(4)]
print("AUC by counting pairs:", pairs, "  scikit-learn roc_auc_score:", roc_auc_score(y, s))
for t in (0.85, 0.65, 0.45, 0.25): print(f"  threshold {t}: TPR {(pos >= t).mean():.2f}  FPR {(neg >= t).mean():.2f}")
rng = np.random.default_rng(0); yy = (rng.random(10000) < 0.01).astype(int)
print("1% positives: a model whose scores are pure noise has AUC", round(roc_auc_score(yy, rng.random(10000)), 3), "; a model that ranks every positive first has AUC", roc_auc_score(yy, yy + 0.0))
```

## Related
[[Evaluation Metrics]] · [[Confusion Matrix]] · [[Recall]] · [[Specificity]] · [[Precision]] · [[F1 Score]] · [[Classification]] · [[Logistic Regression]] · [[Heart Disease Classifier]]
- [[Precision-Recall Curve]] · [[Calibration]]
- [[Anomaly Detection]]
## Sources
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online), chapters 4 and 5
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (free online)
- Image: see the caption. Every number in the history block was computed by the code in this note (toy data; dates and names marked "from memory" are not checked).
- Fawcett, *An introduction to ROC analysis*, 2006 (from memory)
- Saito & Rehmsmeier, *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*, 2015 (from memory)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation ROC curve and AUC:::A picture of what happens at every possible cut-off for a model that gives scores. AUC squeezes the picture into one number: 1.0 is perfect and 0.5 is random guessing.

#flashcards/ml/formula #flashcards/topic/evaluation Write TPR, FPR and AUC and explain each symbol.
?
$$\text{TPR}=\frac{TP}{TP+FN},\quad \text{FPR}=\frac{FP}{FP+TN},\quad \text{AUC}=\Pr\big(s_{\text{pos}}>s_{\text{neg}}\big)$$
- TPR (true positive rate): the share of real positives found, which is the recall.
- FPR (false positive rate): the share of real negatives wrongly called positive.
- $TP$, $FP$, $FN$, $TN$: the confusion-matrix counts at one threshold.
- $s_{\text{pos}}$: the model's score for a randomly chosen real positive. $s_{\text{neg}}$: its score for a randomly chosen real negative.
- AUC is the chance that a random positive is scored above a random negative.

#flashcards/ml/cloze #flashcards/topic/evaluation AUC is the chance that a random ==positive== gets a higher score than a random ==negative==.

#flashcards/ml/example #flashcards/topic/evaluation Positives score 0.9, 0.8, 0.6, 0.4 and negatives 0.7, 0.5, 0.3, 0.2. What are TPR and FPR at threshold 0.65, and the AUC?::At 0.65: positives called yes are 0.9 and 0.8, so TPR = 2/4 = 0.5. Negatives called yes: only 0.7, so FPR = 1/4 = 0.25. AUC: 13 of the 16 positive-negative pairs have the positive higher, so 13/16 = 0.8125 (my pair count).

#flashcards/ml/image #flashcards/topic/evaluation ![[roc-curve.svg]]
What does this picture show?
?
ROC curves. Horizontal: the false positive rate. Vertical: the true positive rate (the recall). Each curve shows one model across all thresholds. The dashed red diagonal is a random guesser, curves closer to the top-left corner are better, and the blue dot at the corner is a perfect classifier.
