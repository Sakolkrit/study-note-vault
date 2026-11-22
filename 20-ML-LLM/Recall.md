---
tags: [machine-learning, evaluation, metrics, classification, recall]
status: growing
level: 4
tier: 1
---

# Recall

**Recall** answers: of all the real "yes" cases, how many did the model find? Back to the overview: [[Evaluation Metrics]].

## 🟢 1. Simple (age 12)
Recall is about **not missing things**. If there are 12 real fires and the alarm rang for 8 of them, the recall is 8 out of 12, or about 67%.

**Example:** You are searching a beach for lost coins. There are 12 coins and you find 8, so your recall is 8 out of 12. You may also pick up a few bottle caps by mistake, but recall only counts how many of the real coins you found.

## 🔵 2. How it works
1. Find all the examples that are **really** yes (the real positives).
2. Count how many of them the model also called yes (**TP**) and how many it missed (**FN**, false negatives).
3. Divide the found ones by all the real ones.
4. The result runs from 0 to 1. A high number means few misses.
5. **The catch 1:** you can get perfect recall by saying yes to everything. So read it together with [[Precision]].
6. **The catch 2:** many models give a score. Lowering the threshold raises recall and lowers precision. The code below shows recall rising from 0.158 to 1.000 as the threshold drops, while the precision falls from 0.941 to 0.102.

![[confusion-matrix-metrics.png]]
*The confusion matrix: rows are what is really true, columns are what the model predicted. TP = true positive (said yes, was yes), TN = true negative, FP = false positive (said yes, was no), FN = false negative (said no, was yes). The bottom-left panel is recall: the oval covers only the bottom row, the cases that really were yes, with TP over FN plus TP. Image: Hssiqueira, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Confusion_Matrix_Metrics.png), CC0.*

## 🟠 3. Technical detail
- **Also called:** sensitivity, or the true positive rate (TPR), in medicine and on the [[ROC Curve and AUC]] (from memory for the medical term).
- **Looks at:** the "really yes" row of the [[Confusion Matrix]] only. False alarms (FP) do not appear.
- **Use it when:** a miss is costly (a disease, a fraud, a security threat).
- **Trade-off:** pushing recall up costs [[Precision]]. [[F1 Score]] combines the two.
- **In search and RAG:** "recall at k" asks whether the right document was among the first k results ([[Retrieval Metrics]], [[RAG]]).

## 🔴 4. Deep / formal
**Recall.** True positives over all the real positives:

$$
\text{Recall}=\frac{TP}{TP+FN}
$$

**Where:**
- $TP$ is the number of real yes-cases the model found. $FN$ is the number of real yes-cases it missed.
- $TP+FN$ is all the real yes-cases (the whole bottom row of the [[Confusion Matrix]]).
- **Matters:** false alarms ($FP$) are not in the formula. A model that says yes to everything has $FN=0$ and a perfect recall of 1.0, however useless it is.

**Example (tiny numbers):** $TP=8$, $FN=4$.
- Recall $=8/(8+4)=8/12=\mathbf{0.667}$.
- Meaning: the model finds two thirds of the real cases and misses one third.
- The lazy "always no" model has $TP=0$, $FN=12$, so recall $=0/12=\mathbf{0}$.

**Recall at a threshold.** A scoring model says yes when its score is at least $t$. Lowering $t$ can only turn "no" answers into "yes", so $TP$ can only go up, $FN$ can only go down, and recall can only rise (or stay). Precision usually falls because new false alarms come in with the new true positives.

**Further notes:**
- **Recall and the cost of misses:** choose the threshold by the cost of a miss against the cost of a false alarm.
- **Recall at k** for retrieval is the same formula, applied to the first k results ([[Retrieval Metrics]]).

**What came before, and how this improved on it:**
- **Before:** precision alone, or accuracy.
- **Its limit:** a model that says yes rarely and carefully has high precision and misses most real cases.
- **What changed:** also count how many real cases were found (information retrieval, 1950s and 1960s; from memory, not checked).
- **How much better:** in the code below, the same scoring model has a precision of 0.941 and a recall of only 0.158 at the strictest threshold (it misses 84% of the positives), and a recall of 1.000 with a precision of 0.102 when it says yes to everyone. Precision alone looks excellent at the first point, and recall exposes it.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
y = rng.random(20000) < 0.1
score = np.where(y, rng.normal(1.0, 1.0, y.size), rng.normal(-1.0, 1.0, y.size))
print("same model, lowering the threshold ('say yes if score >= t'):")
for t in (2.0, 1.0, 0.0, -1.0, -9.0):
    called = score >= t; tp = (called & y).sum(); fp = (called & ~y).sum()
    print(f"  t = {t:+5.1f}: recall {tp / y.sum():.3f}  precision {tp / max(tp + fp, 1):.3f}  false alarms {fp:6d}")
print("saying yes to everyone (t = -9) gives recall 1.000 and precision equal to the share of positives:", round(y.mean(), 3))
```

## Related
[[Evaluation Metrics]] · [[Confusion Matrix]] · [[Precision]] · [[Specificity]] · [[F1 Score]] · [[ROC Curve and AUC]] · [[Retrieval Metrics]] · [[RAG]] · [[Classification]]
- [[Precision-Recall Curve]]

## Sources
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online), chapters 4 and 5
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (free online)
- Image: see the caption. Every number in the history block was computed by the code in this note (toy data; dates and names marked "from memory" are not checked).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation Recall:::Not missing things. If there are 12 real fires and the alarm rang for 8 of them, the recall is 8 out of 12, about 67%.

#flashcards/ml/formula #flashcards/topic/evaluation Write recall and explain each symbol.
?
$$\text{Recall}=\frac{TP}{TP+FN}$$
- $TP$: the real yes-cases the model found. $FN$: the real yes-cases it missed.
- $TP+FN$: all the real yes-cases (the whole "really yes" row of the confusion matrix).
- False alarms ($FP$) are not in the formula. A model that says yes to everything has $FN=0$ and a perfect recall of 1.

#flashcards/ml/cloze #flashcards/topic/evaluation Recall counts how many real cases were ==found==, and ignores ==false alarms==.

#flashcards/ml/example #flashcards/topic/evaluation $TP=8$ and $FN=4$. What is the recall, and for the lazy "always no" model?::8/(8 + 4) = 0.667. The model finds two thirds of the real cases. The lazy model has TP = 0 and FN = 12, so recall 0/12 = 0.

#flashcards/ml/image #flashcards/topic/evaluation ![[confusion-matrix-metrics.png]]
What does this picture show?
?
The confusion matrix: rows are what is really true, columns are what the model predicted. TP = true positive (said yes, was yes), TN = true negative, FP = false positive (said yes, was no), FN = false negative (said no, was yes). The four panels show the formulas for accuracy, precision, recall and specificity.
