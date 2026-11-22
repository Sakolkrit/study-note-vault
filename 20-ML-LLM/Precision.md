---
tags: [machine-learning, evaluation, metrics, classification, precision]
status: growing
level: 4
tier: 1
---

# Precision

**Precision** answers: when the model says "yes", how often is it right? Back to the overview: [[Evaluation Metrics]].

## 🟢 1. Simple (age 12)
Precision is about **trust in the alarm**. Every time the model shouts "yes!", did it turn out to be true? If it shouted 10 times and 8 were real, the precision is 80%.

**Example:** A friend who tells you "I think that's a rare bird!" ten times. If eight of the ten really were rare birds, you can trust your friend's shouts (precision 0.8).

## 🔵 2. How it works
1. Run the model on test examples and find all the ones where it said **yes**.
2. Among those, count how many were really yes (the **TP**, true positives) and how many were not (the **FP**, false positives).
3. Divide the true positives by the number of yes-answers.
4. The result runs from 0 to 1. A high number means few false alarms.
5. **The catch 1:** you can get a perfect precision by saying yes only once, for the surest case. So precision must be read together with [[Recall]], which counts how many real cases were found.
6. **The catch 2:** precision depends on how rare the real positives are. The same model has a lower precision when positives are rare, because the many negatives produce many false alarms.

![[confusion-matrix-metrics.png]]
*The confusion matrix: rows are what is really true, columns are what the model predicted. TP = true positive (said yes, was yes), TN = true negative, FP = false positive (said yes, was no), FN = false negative (said no, was yes). The top-right panel is precision: the oval covers only the right-hand column, the cases the model called yes, with TP over TP plus FP. Image: Hssiqueira, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Confusion_Matrix_Metrics.png), CC0.*

## 🟠 3. Technical detail
- **Also called:** positive predictive value (from memory).
- **Looks at:** the "predicted yes" column of the [[Confusion Matrix]] only. True negatives and misses (FN) do not appear.
- **Use it when:** a false alarm is costly (a good email sent to spam, an innocent person flagged).
- **Trade-off:** raising the threshold of a scoring model usually raises precision and lowers [[Recall]]. [[F1 Score]] combines them, and the precision-recall curve shows all thresholds ([[Evaluation Metrics]]).
- **Depends on how common positives are:** see the second formula below, which is [[Bayes' Theorem]] again.

## 🔴 4. Deep / formal
**Precision.** True positives over everything the model called positive:

$$
\text{Precision}=\frac{TP}{TP+FP}
$$

**Where:**
- $TP$ is the number of cases called yes that really were yes. $FP$ is the number called yes that really were no (false alarms).
- $TP+FP$ is everything the model called yes.
- **Matters:** misses ($FN$) are not in the formula at all. A model that says yes only for its single surest case can score 1.0, which is why you also need [[Recall]].

**Example (tiny numbers):** $TP=8$, $FP=2$.
- Precision $=8/(8+2)=\mathbf{0.80}$.
- Meaning: when this model says yes, it is right 80% of the time and wrong 20% of the time.

**Why rarity matters.** If a model catches a fraction $\text{sens}$ of the real positives and wrongly flags a fraction $1-\text{spec}$ of the negatives, and a fraction $\text{prev}$ of the cases are positive, then

$$
\text{Precision}=\frac{\text{sens}\cdot\text{prev}}{\text{sens}\cdot\text{prev}+(1-\text{spec})(1-\text{prev})}
$$

**Where:**
- $\text{sens}$ is the share of real positives found (the recall). $\text{spec}$ is the share of real negatives correctly called no ([[Specificity]]). $\text{prev}$ is the share of cases that are really positive (the prevalence).
- The top is the share of all cases that are true alarms. The bottom adds the false alarms, which come from the negatives: a fraction $1-\text{spec}$ of a fraction $1-\text{prev}$ of the cases.
- **Matters:** when $\text{prev}$ is tiny, the many negatives swamp the few positives, so even a small false-alarm rate gives many false alarms. This is the same calculation as in [[Bayes' Theorem]].

**Example (tiny numbers):** $\text{sens}=0.9$, $\text{spec}=0.9$.
- $\text{prev}=0.5$: $0.9\cdot0.5/(0.45+0.1\cdot0.5)=0.45/0.50=\mathbf{0.900}$.
- $\text{prev}=0.1$: $0.09/(0.09+0.1\cdot0.9)=0.09/0.18=\mathbf{0.500}$.
- $\text{prev}=0.01$: $0.009/(0.009+0.1\cdot0.99)=0.009/0.108=\mathbf{0.083}$.
- Meaning: the very same test is right 90% of the time when positives are common, and only 8.3% of the time when only 1 case in 100 is positive.

**What came before, and how this improved on it:**
- **Before:** accuracy, which counts all answers together.
- **Its limit:** accuracy cannot tell you how much to trust a "yes". Precision and recall come from information retrieval, where the question was "of the documents the system returned, how many were relevant?" (1950s and 1960s; from memory, not checked).
- **What changed:** look only at the cases the model called yes.
- **How much better:** in the code below, the same model (finds 90% of the positives, flags 10% of the negatives) has a precision of 0.900 when half the cases are positive, 0.500 when a tenth are, and 0.083 when 1% are, which the formula confirms. Accuracy would stay near 0.9 in all three cases and hide this.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
sens, spec = 0.9, 0.9                                              # the same test every time: catches 90% of positives, wrongly flags 10% of negatives
print("same model (recall 0.9, 10% false alarms), precision as the positives get rarer:")
for prev in (0.5, 0.1, 0.01):
    y = rng.random(1_000_000) < prev
    called = np.where(y, rng.random(y.size) < sens, rng.random(y.size) < 1 - spec)
    exact = sens * prev / (sens * prev + (1 - spec) * (1 - prev))
    print(f"  {prev:5.0%} positive: precision {y[called].mean():.3f}   (formula {exact:.3f})")
TP, FP = 8, 2
print("worked example: 8 / (8 + 2) =", TP / (TP + FP))
```

## Related
[[Evaluation Metrics]] · [[Confusion Matrix]] · [[Recall]] · [[Specificity]] · [[F1 Score]] · [[Accuracy]] · [[Bayes' Theorem]] · [[Retrieval Metrics]] · [[Classification]]
- [[Precision-Recall Curve]]

## Sources
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online), chapters 4 and 5
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (free online)
- Image: see the caption. Every number in the history block was computed by the code in this note (toy data; dates and names marked "from memory" are not checked).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation Precision:::Trust in the alarm. Every time the model shouts "yes!", did it turn out to be true? If it shouted 10 times and 8 were real, the precision is 80%.

#flashcards/ml/formula #flashcards/topic/evaluation Write precision and explain each symbol.
?
$$\text{Precision}=\frac{TP}{TP+FP}$$
- $TP$: the cases called yes that really were yes. $FP$: the cases called yes that really were no (false alarms).
- $TP+FP$: everything the model called yes.
- Misses ($FN$) are not in the formula at all. A model that says yes only for its single surest case can score 1.0 while missing nearly everything.

#flashcards/ml/formula #flashcards/topic/evaluation Write precision in terms of sensitivity, specificity and prevalence, and explain each symbol.
?
$$\text{Precision}=\frac{\text{sens}\cdot\text{prev}}{\text{sens}\cdot\text{prev}+(1-\text{spec})(1-\text{prev})}$$
- $\text{sens}$: the share of real positives found (the recall). $\text{spec}$: the share of real negatives correctly called no (specificity).
- $\text{prev}$: the share of cases that are really positive (the prevalence).
- The top is the share of all cases that are true alarms. The bottom adds the share that are false alarms.
- So the same model has lower precision when positives are rarer.

#flashcards/ml/cloze #flashcards/topic/evaluation Precision is the share of ==alarms== that were real, and it falls when the positive class is ==rare==.

#flashcards/ml/example #flashcards/topic/evaluation $TP=8$ and $FP=2$. What is the precision?::8/(8 + 2) = 0.80. When the model says yes it is right 80% of the time and wrong 20% of the time.

#flashcards/ml/example #flashcards/topic/evaluation With sensitivity 0.9 and specificity 0.9, what is the precision at prevalence 0.5, 0.1 and 0.01?::0.5: 0.45/0.50 = 0.900. 0.1: 0.09/(0.09 + 0.09) = 0.500. 0.01: 0.009/(0.009 + 0.099) = 0.083. The very same test gives mostly false alarms when the condition is rare.

#flashcards/ml/image #flashcards/topic/evaluation ![[confusion-matrix-metrics.png]]
What does this picture show?
?
The confusion matrix: rows are what is really true, columns are what the model predicted. TP = true positive (said yes, was yes), TN = true negative, FP = false positive (said yes, was no), FN = false negative (said no, was yes). The four panels show the formulas for accuracy, precision, recall and specificity.
