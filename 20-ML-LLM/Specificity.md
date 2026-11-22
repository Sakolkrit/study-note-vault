---
tags: [machine-learning, evaluation, metrics, classification, specificity]
status: growing
level: 4
tier: 1
---

# Specificity

**Specificity** answers: of all the real "no" cases, how many did the model correctly call no? Back to the overview: [[Evaluation Metrics]].

## 🟢 1. Simple (age 12)
Specificity is the mirror of recall. Recall asks "did you catch the real yes-cases?". Specificity asks "did you leave the real no-cases alone?". A model with high specificity rarely raises a false alarm.

**Example:** A metal detector at an airport. High specificity means it stays quiet for people carrying nothing metal. Low specificity means it beeps at belt buckles and keys all day.

## 🔵 2. How it works
1. Find all the examples that are **really** no (the real negatives).
2. Count how many of them the model correctly called no (**TN**, true negatives) and how many it wrongly called yes (**FP**, false positives).
3. Divide the correct no-answers by all the real no-cases.
4. The result runs from 0 to 1. A high number means few false alarms.
5. The **false positive rate** is one minus specificity: the share of real no-cases that raised a false alarm.
6. **The catch:** a model that says no to everything has specificity 1.0. So read it together with [[Recall]].

![[confusion-matrix-metrics.png]]
*The confusion matrix: rows are what is really true, columns are what the model predicted. TP = true positive (said yes, was yes), TN = true negative, FP = false positive (said yes, was no), FN = false negative (said no, was yes). The bottom-right panel is specificity: the oval covers the top row, the cases that really were no, with TN over TN plus FP. Image: Hssiqueira, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Confusion_Matrix_Metrics.png), CC0.*

## 🟠 3. Technical detail
- **Looks at:** the "really no" row of the [[Confusion Matrix]] only.
- **Used with recall:** in medicine the pair is called sensitivity (recall) and specificity (from memory).
- **On the ROC curve:** the horizontal axis is the false positive rate, which is $1-$ specificity ([[ROC Curve and AUC]]).
- **Why it matters for rare diseases:** even a high specificity gives many false alarms when the healthy people vastly outnumber the sick, which the code below shows ([[Precision]], [[Bayes' Theorem]]).

## 🔴 4. Deep / formal
**Specificity.** True negatives over all the real negatives:

$$
\text{Specificity}=\frac{TN}{TN+FP},\qquad \text{FPR}=\frac{FP}{TN+FP}=1-\text{Specificity}
$$

**Where:**
- $TN$ is the number of real no-cases called no. $FP$ is the number of real no-cases called yes (false alarms).
- $TN+FP$ is all the real no-cases. FPR is the false positive rate.
- **Matters:** misses ($FN$) are not in the formula. A model that always says no has $FP=0$ and a perfect specificity of 1.0.

**Example (tiny numbers):** $TN=86$, $FP=2$.
- Specificity $=86/(86+2)=86/88=\mathbf{0.977}$.
- FPR $=2/88=\mathbf{0.023}$, which is $1-0.977$.
- Meaning: the model leaves 97.7% of the real no-cases alone and raises a false alarm for 2.3% of them.

**Further notes:**
- **Prevalence matters:** a specificity of 0.95 sounds good, but with 1% sick people it means 495 false alarms among 10,000 people, five times the number of real cases (see the code).

**What came before, and how this improved on it:**
- **Before:** report sensitivity (the share of sick people caught) alone, for a medical test.
- **Its limit:** a test that says "sick" for everyone has a sensitivity of 100% and is useless.
- **What changed:** report specificity as well, the share of healthy people correctly cleared. The pair was introduced for medical tests (Yerushalmy, 1947; from memory, not checked).
- **How much better:** in the code below, screening 10,000 people of whom 100 are sick with a test that catches 99% of them, a specificity of 0.90 gives 990 false alarms and an alarm is real only 9.1% of the time. Specificity 0.95 gives 495 false alarms (16.7%), and 0.99 gives 99 (50.0%). Raising specificity from 0.90 to 0.99 turns the alarm from a rarely true one into a coin flip.

**Check it yourself:**

```python
import numpy as np
people, prev, sens = 10000, 0.01, 0.99
sick = int(people * prev); healthy = people - sick
print(f"screening {people:,} people, {sick} are sick, test catches {sens:.0%} of them:")
for spec in (0.90, 0.95, 0.99):
    fp = healthy * (1 - spec); tp = sick * sens
    print(f"  specificity {spec:.2f}: false alarms {fp:6.0f}, true alarms {tp:5.1f}, so an alarm is real only {tp / (tp + fp):.1%} of the time")
TN, FP = 86, 2
print("worked example: 86 / (86 + 2) =", round(TN / (TN + FP), 3), " false positive rate =", round(FP / (TN + FP), 3), " (= 1 - specificity)")
```

## Related
[[Evaluation Metrics]] · [[Confusion Matrix]] · [[Recall]] · [[Precision]] · [[ROC Curve and AUC]] · [[Bayes' Theorem]] · [[Classification]]

## Sources
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online), chapters 4 and 5
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (free online)
- Image: see the caption. Every number in the history block was computed by the code in this note (toy data; dates and names marked "from memory" are not checked).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation Specificity:::The mirror of recall. Recall asks "did you catch the real yes-cases?", specificity asks "did you leave the real no-cases alone?". A model with high specificity rarely raises a false alarm.

#flashcards/ml/formula #flashcards/topic/evaluation Write specificity and the false positive rate and explain each symbol.
?
$$\text{Specificity}=\frac{TN}{TN+FP},\qquad \text{FPR}=\frac{FP}{TN+FP}=1-\text{Specificity}$$
- $TN$: the real no-cases called no. $FP$: the real no-cases called yes (false alarms).
- $TN+FP$: all the real no-cases. FPR: the false positive rate.
- Misses ($FN$) are not in the formula. A model that always says no has $FP=0$ and a perfect specificity of 1.0.

#flashcards/ml/cloze #flashcards/topic/evaluation The false positive rate equals ==1 − specificity==.

#flashcards/ml/example #flashcards/topic/evaluation $TN=86$ and $FP=2$. What are the specificity and FPR?::86/(86 + 2) = 0.977. FPR = 2/88 = 0.023 = 1 − 0.977. The model leaves 97.7% of the real no-cases alone and raises a false alarm for 2.3% of them.

#flashcards/ml/image #flashcards/topic/evaluation ![[confusion-matrix-metrics.png]]
What does this picture show?
?
The confusion matrix: rows are what is really true, columns are what the model predicted. TP = true positive (said yes, was yes), TN = true negative, FP = false positive (said yes, was no), FN = false negative (said no, was yes). The four panels show the formulas for accuracy, precision, recall and specificity.
