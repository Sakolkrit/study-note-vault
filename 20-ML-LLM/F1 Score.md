---
tags: [machine-learning, evaluation, metrics, classification, F1]
status: growing
level: 4
tier: 2
---

# F1 Score

The **F1 score** squeezes precision and recall into one number. Back to the overview: [[Evaluation Metrics]].

## 🟢 1. Simple (age 12)
Precision says "can I trust the alarm?" and recall says "did it find them all?". F1 is one number that is only high when **both** are high. If one is bad, F1 is bad too.

**Example:** A student must get both the spelling test and the maths test right to pass. A top mark in one and a zero in the other is still a fail. F1 works the same way for precision and recall.

## 🔵 2. How it works
1. Compute [[Precision]] (how many alarms were real) and [[Recall]] (how many real cases were found) from the [[Confusion Matrix]].
2. Combine them with the **harmonic mean**, a kind of average that leans towards the smaller number.
3. If both are high, F1 is high. If either is low, F1 is low.
4. A model that finds nothing has F1 equal to 0.
5. **The catch:** F1 ignores the true negatives completely, and it treats precision and recall as equally important. If one of them matters more, use a weighted version.


## 🟠 3. Technical detail
- **Harmonic, not ordinary:** with precision 1.0 and recall 0.1 the ordinary average is 0.55, but F1 is 0.182.
- **F-beta (from memory):** a family that weights recall more or less than precision. F1 is the case with equal weight.
- **For many classes:** compute F1 per class and average: macro (each class counts equally) or micro (each example counts equally) ([[Evaluation Metrics]]).
- **Use it for:** imbalanced problems where accuracy fails ([[Accuracy]]).
- **Does not tell:** how the model does on the negative class. For that, see [[Specificity]].

## 🔴 4. Deep / formal
**F1: one number for precision and recall.** F1 is the harmonic mean of the two:

$$
F_1=\frac{2PR}{P+R}=\frac{2\,TP}{2\,TP+FP+FN}
$$

**Where:**
- $P$ is the precision, $TP/(TP+FP)$, and $R$ is the recall, $TP/(TP+FN)$ (see [[Precision]] and [[Recall]]). $TP$, $FP$ and $FN$ are the counts of true positives, false positives and false negatives.
- The two forms are the same number written two ways: substitute $P$ and $R$ and simplify.
- **Matters:** it is a **harmonic** mean, not the ordinary average. The harmonic mean is dragged towards the **smaller** of the two numbers, so a model cannot hide a terrible recall behind a perfect precision (or the other way round). F1 is high only when **both** are reasonably high.
- It ignores the true negatives completely. **Matters:** that is useful when negatives are common and boring, but it also means F1 says nothing about how well the model handles the negative class.

**Example (tiny numbers):** the earlier example of $TP=8$, $FP=2$, $FN=4$, with $P=0.8$ and $R=0.667$.
- $F_1=2\cdot0.8\cdot0.667/(0.8+0.667)=\mathbf{0.727}$. The other form: $2\cdot8/(16+2+4)=16/22=\mathbf{0.727}$.
- The ordinary average of $P$ and $R$ would be $0.733$, a little higher.
- A lopsided model with $P=1$, $R=0.1$: the average is $\mathbf{0.55}$, but $F_1=\mathbf{0.182}$.
- The lazy "always no" model from the [[Confusion Matrix]] example has $TP=0$, so $F_1=\mathbf{0}$ even though its accuracy was $0.88$.
- Meaning: F1 punishes imbalance between precision and recall, and exposes the useless model that accuracy hid.

**Further notes:**
- **F1 depends on the threshold:** like precision and recall, it changes with the cut-off, and the best F1 threshold is often not 0.5.
- **Alternatives:** the precision-recall curve and [[ROC Curve and AUC]] look at all thresholds at once.

**What came before, and how this improved on it:**
- **Before:** the ordinary average of precision and recall.
- **Its limit:** the average hides a disaster: a model with perfect precision and almost no recall still averages near 0.5.
- **What changed:** the harmonic mean, which stays close to the smaller of the two. It is called the F-measure in information retrieval and comes from van Rijsbergen's work in the 1970s (from memory, not checked).
- **How much better:** in the code below, for precision 1.00 and recall 0.01 the ordinary average is 0.505 and F1 is 0.020. For precision 1.00 and recall 0.10 the average is 0.550 and F1 is 0.182. For the balanced pair 0.5 and 0.5 the two are equal (0.500). The "always no" model has F1 = 0, although its accuracy on the 100 cases is 0.88.

**Check it yourself:**

```python
import numpy as np
f1 = lambda P, R: 2 * P * R / (P + R) if P + R else 0.0
print("precision recall   average  F1")
for P, R in ((0.8, 0.667), (1.0, 0.1), (0.5, 0.5), (1.0, 0.01)):
    print(f"   {P:4.2f}   {R:5.3f}   {(P + R) / 2:6.3f}  {f1(P, R):.3f}")
TP, FP, FN = 8, 2, 4
print("two forms of F1:", round(2 * 0.8 * (8 / 12) / (0.8 + 8 / 12), 3), round(2 * TP / (2 * TP + FP + FN), 3))
print("always-no model: TP = 0, so F1 =", f1(0.0, 0.0), "although its accuracy on the 100 cases is 0.88")
```

## Related
[[Evaluation Metrics]] · [[Confusion Matrix]] · [[Precision]] · [[Recall]] · [[Accuracy]] · [[ROC Curve and AUC]] · [[Classification]]
- [[Precision-Recall Curve]] · [[Image Segmentation]]

## Sources
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online), chapters 4 and 5
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (free online)
- Image: see the caption. Every number in the history block was computed by the code in this note (toy data; dates and names marked "from memory" are not checked).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation F1 score:::One number that is only high when both precision and recall are high. If one is bad, F1 is bad too, like passing only if you get both the spelling and the maths test right.

#flashcards/ml/formula #flashcards/topic/evaluation Write the F1 score in both forms and explain each symbol.
?
$$F_1=\frac{2PR}{P+R}=\frac{2\,TP}{2\,TP+FP+FN}$$
- $P$: the precision, $TP/(TP+FP)$. $R$: the recall, $TP/(TP+FN)$.
- $TP$, $FP$, $FN$: the counts of true positives, false positives and false negatives.
- The two forms are the same number written two ways: substitute $P$ and $R$ and simplify.
- It is the harmonic mean of $P$ and $R$, so a low one drags the score down more than a plain average would.

#flashcards/ml/cloze #flashcards/topic/evaluation F1 is the ==harmonic== mean of precision and recall, so one low value pulls it ==down==.

#flashcards/ml/example #flashcards/topic/evaluation $TP=8$, $FP=2$, $FN=4$, so $P=0.8$ and $R=0.667$. What is F1, and how does it compare with the plain average?::F1 = 2 × 0.8 × 0.667/(0.8 + 0.667) = 0.727. Other form: 2 × 8/(16 + 2 + 4) = 16/22 = 0.727. The plain average would be 0.733, a little higher.
