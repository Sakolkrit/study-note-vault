---
tags: [machine-learning, evaluation, metrics, classification, accuracy]
status: growing
level: 4
tier: 1
---

# Accuracy

**Accuracy** is the share of all answers that were right. Back to the overview: [[Evaluation Metrics]].

## 🟢 1. Simple (age 12)
Accuracy answers one simple question: out of everything the model was asked, how many did it get right? If it got 94 out of 100 right, the accuracy is 94%.

**Example:** A teacher marks a 10-question quiz. You got 9 right, so your accuracy is 9 out of 10, or 90%.

## 🔵 2. How it works
1. Test the model on examples whose true answers you know ([[Evaluation Metrics]]).
2. Count how many of its answers were right. In a [[Confusion Matrix]] these are the two good boxes, TP and TN.
3. Divide by the total number of examples.
4. The result is a number from 0 to 1 (or 0% to 100%). Higher is better.
5. **The catch:** accuracy treats every mistake the same and counts the common class most. If 99 of 100 emails are not spam, a model that always says "not spam" has an accuracy of 99% and has learned nothing. For rare things, look at [[Precision]], [[Recall]] and [[F1 Score]] too.

![[confusion-matrix-metrics.png]]
*The confusion matrix: rows are what is really true, columns are what the model predicted. TP = true positive (said yes, was yes), TN = true negative, FP = false positive (said yes, was no), FN = false negative (said no, was yes). The top-left panel is accuracy: the big circle covers all four cells, with the two correct ones, TP and TN, on top. Image: Hssiqueira, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Confusion_Matrix_Metrics.png), CC0.*

## 🟠 3. Technical detail
- **Good for:** problems where the classes are about equally common and the two kinds of mistake cost about the same.
- **Misleading for:** rare events. The "always no" baseline gets accuracy equal to one minus the share of positives, so it climbs as positives get rarer.
- **Always compare with a baseline:** the accuracy of answering the most common class every time. Accuracy only means something above that line ([[Classification]] and [[Machine Learning]] use this baseline).
- **For many classes:** the same rule works, with the correct answers on the diagonal of the table. For numbers there is no accuracy, see [[Regression]] and its error measures.
- **Error rate:** one minus accuracy.

## 🔴 4. Deep / formal
**Accuracy.** Right answers over all answers:

$$
\text{Accuracy}=\frac{TP+TN}{TP+FP+FN+TN}
$$

**Where:**
- $TP$ (true positives) and $TN$ (true negatives) are the right answers: said yes and it was yes, said no and it was no.
- $FP$ (false positives) and $FN$ (false negatives) are the wrong answers: a false alarm and a miss. The bottom is the number of all cases.
- **Matters:** TN is in the top, and in a rare-event problem TN is huge, so accuracy is dominated by the easy "no" cases. Without TN in the formula, the number would depend only on the positives (that is what [[Precision]] and [[Recall]] do).

**Example (tiny numbers):** $TP=8$, $FP=2$, $FN=4$, $TN=86$.
- Right answers: $8+86=\mathbf{94}$. All cases: $8+2+4+86=\mathbf{100}$.
- Accuracy $=94/100=\mathbf{0.94}$.
- A lazy model that always says "no" has $TP=0$, $TN=88$ (all real negatives), so accuracy $=88/100=\mathbf{0.88}$.
- Meaning: 0.94 sounds great, but the lazy model already gets 0.88, so the real gain is small, and recall ([[Recall]]) shows the model still misses a third of the real cases.

**Further notes:**
- **Balanced accuracy** (from memory) averages the recall on each class, so each class counts equally.
- **Confidence intervals:** on a small test set, the accuracy is only known roughly. Report a range ([[Evaluation Metrics]]).

**What came before, and how this improved on it:**
- **Before:** judge a model by looking at some of its answers by hand.
- **Its limit:** it does not scale, and two people disagree on what "looks good".
- **What changed:** count right answers over all answers: one repeatable number. It works well when the classes are balanced.
- **How much better:** in the code below, for a model that always says "no" the accuracy is 0.4990 when half of the cases are positive, and 0.9990 when only 0.1% are. The number is not worse with rare events, it is worse as a *measure*: the same useless model looks better and better. So accuracy needs a baseline.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
print("accuracy of a model that always says 'no', as positives get rarer:")
for prev in (0.5, 0.1, 0.01, 0.001):
    y = (rng.random(100000) < prev).astype(int); print(f"  {prev:6.1%} positive: accuracy {np.mean(y == 0):.4f}   (exact: 1 - {prev} = {1 - prev:.4f})")
TP, FP, FN, TN = 8, 2, 4, 86
print("worked example:", (TP + TN) / (TP + FP + FN + TN), " lazy model on the same 100 cases:", (0 + 88) / 100)
```

## Related
[[Evaluation Metrics]] · [[Confusion Matrix]] · [[Precision]] · [[Recall]] · [[F1 Score]] · [[Classification]] · [[Machine Learning]]

## Sources
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online), chapters 4 and 5
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (free online)
- Image: see the caption. Every number in the history block was computed by the code in this note (toy data; dates and names marked "from memory" are not checked).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation Accuracy:::Out of everything the model was asked, how many did it get right? 94 right out of 100 is an accuracy of 94%.

#flashcards/ml/formula #flashcards/topic/evaluation Write accuracy and explain each symbol.
?
$$\text{Accuracy}=\frac{TP+TN}{TP+FP+FN+TN}$$
- $TP$ (true positives) and $TN$ (true negatives): the right answers. Said yes and it was yes, said no and it was no.
- $FP$ (false positives) and $FN$ (false negatives): the wrong answers. A false alarm and a miss.
- The bottom is the number of all cases.
- TN is in the top, so in a rare-positive problem a lazy model that always says "no" still scores high.

#flashcards/ml/cloze #flashcards/topic/evaluation Accuracy can look ==great== for a lazy model when one class is ==rare==.

#flashcards/ml/example #flashcards/topic/evaluation $TP=8$, $FP=2$, $FN=4$, $TN=86$. What is the accuracy, and what for a lazy model that always says "no"?::Right: 8 + 86 = 94. All: 100. Accuracy 0.94. The lazy model has TP = 0, TN = 88, so accuracy 88/100 = 0.88. Sounds great, but it finds no real case.

#flashcards/ml/image #flashcards/topic/evaluation ![[confusion-matrix-metrics.png]]
What does this picture show?
?
The confusion matrix: rows are what is really true, columns are what the model predicted. TP = true positive (said yes, was yes), TN = true negative, FP = false positive (said yes, was no), FN = false negative (said no, was yes). The four panels show the formulas for accuracy, precision, recall and specificity.
