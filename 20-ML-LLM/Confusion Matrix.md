---
tags: [machine-learning, evaluation, metrics, classification, confusion-matrix]
status: growing
level: 4
tier: 1
---

# Confusion Matrix

A **confusion matrix** is a small table that counts a model's right and wrong answers, split by the kind of mistake. Back to the overview: [[Evaluation Metrics]].

## 🟢 1. Simple (age 12)
A confusion matrix is a scoreboard with four boxes. It counts how many times the model said **yes** and was right, said **yes** and was wrong, said **no** and was right, and said **no** and was wrong. Because the wrong answers are split into two kinds, you can see *what sort* of mistakes the model makes.

**Example:** A smoke alarm. Box 1: it rang and there was a fire (good). Box 2: it rang but there was no fire (false alarm). Box 3: it stayed quiet but there was a fire (the dangerous miss). Box 4: it stayed quiet and there was no fire (good).

## 🔵 2. How it works
1. Choose which answer counts as "positive", usually the rare or important one (the fire, the disease, the spam).
2. Run the model on examples whose true answers you already know, which it has never trained on ([[Evaluation Metrics]]).
3. For each example, look at two things: what is **really** true and what the model **predicted**.
4. Add one tally mark to one of four boxes:
   - **TP** (true positive): really yes, predicted yes.
   - **FP** (false positive): really no, predicted yes. A false alarm.
   - **FN** (false negative): really yes, predicted no. A miss.
   - **TN** (true negative): really no, predicted no.
5. The four counts add up to the number of examples you tested.
6. Every other score is made from these four counts: [[Accuracy]], [[Precision]], [[Recall]], [[Specificity]] and [[F1 Score]].
7. **The catch:** the matrix depends on a threshold. A model that outputs a score has a different matrix for every cut-off, which is what the [[ROC Curve and AUC]] summarises.

![[confusion-matrix-metrics.png]]
*The confusion matrix: rows are what is really true, columns are what the model predicted. TP = true positive (said yes, was yes), TN = true negative, FP = false positive (said yes, was no), FN = false negative (said no, was yes). The four panels show the formulas for accuracy, precision, recall and specificity: each circle or oval marks the cells that the formula uses. Image: Hssiqueira, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Confusion_Matrix_Metrics.png), CC0.*

## 🟠 3. Technical detail
- **Layout:** rows are the true answers and columns are the predictions (the picture follows this layout; some books swap rows and columns, so always read the labels).
- **Counts, not percentages:** keep the raw counts. Percentages hide how many cases there were.
- **More than two classes:** the table gets one row and one column per class. The diagonal holds the correct answers, and every cell off the diagonal says "this class was mistaken for that one".
- **Why not accuracy alone:** a single score hides which cell the mistakes fall in. The code below shows two models where accuracy even ranks the useless one first.
- **Related ideas:** [[Classification]] produces the predictions, and the matrix is the first thing to look at when judging them.

## 🔴 4. Deep / formal
**Reading the two matrices below.** With 10,000 cases of which 89 are positive, the model that always says "no" has $TP=0$, $FP=0$, $FN=89$, $TN=9911$. Its accuracy is $0.991$ and it misses every positive. A noisy model that is wrong 20% of the time has $TP=67$, $FP=2000$, $FN=22$, $TN=7911$ and an accuracy of $0.798$.

**Worked example (tiny numbers):** out of 100 cases a model gives $TP=8$, $FP=2$, $FN=4$, $TN=86$ (the example used in the metric notes).
- Total: $8+2+4+86=\mathbf{100}$.
- Real positives: $TP+FN=8+4=\mathbf{12}$. Real negatives: $FP+TN=2+86=\mathbf{88}$.
- Predicted positives: $TP+FP=8+2=\mathbf{10}$.
- Meaning: from these three row and column totals, every metric in [[Evaluation Metrics]] follows.

**Further notes:**
- **Costs:** you can attach a cost to each kind of mistake and multiply it by the counts, to choose a threshold by total cost.
- **Imbalanced data:** the matrix is how you notice that a rare class is being ignored.
- **Not covered here:** the matrix of a multi-class model, the Matthews correlation coefficient, and the kappa score.

**What came before, and how this improved on it:**
- **Before:** report one number, the accuracy.
- **Its limit:** one number cannot show which kind of mistake is happening. Two very different models can have the same accuracy.
- **What changed:** count the four outcomes in a table. The term "confusion matrix" is widely used in machine learning (the name goes back to the 1990s or earlier; from memory, not checked).
- **How much better:** in the code below, the lazy model scores an accuracy of 0.991 and the noisy one 0.798, so accuracy prefers the lazy model. The matrices show why that is wrong: the lazy model has FN = 89, which means it misses every one of the 89 real positives, while the noisy one finds 67 of them (and raises 2,000 false alarms).

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
y = (rng.random(10000) < 0.01).astype(int)                       # 1% really positive
lazy = np.zeros_like(y); rng2 = np.random.default_rng(1)
noisy = np.where(rng2.random(10000) < 0.8, y, 1 - y)             # a model that is wrong 20% of the time
def matrix(pred):
    return int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
for name, pred in (("always says no", lazy), ("noisy model", noisy)):
    tp, fp, fn, tn = matrix(pred)
    print(f"{name:15s} TP {tp:5d}  FP {fp:5d}  FN {fn:5d}  TN {tn:5d}   (total {tp + fp + fn + tn:,}; accuracy {(tp + tn) / (tp + fp + fn + tn):.3f})")
print("real positives:", int(y.sum()), "- the lazy model misses every one of them (FN), which accuracy alone does not show")
```

## Related
[[Evaluation Metrics]] · [[Accuracy]] · [[Precision]] · [[Recall]] · [[Specificity]] · [[F1 Score]] · [[ROC Curve and AUC]] · [[Classification]] · [[Logistic Regression]] · [[Heart Disease Classifier]] · [[Tiny Object Detector]]
## Sources
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online), chapters 4 and 5
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (free online)
- Image: see the caption. Every number in the history block was computed by the code in this note (toy data; dates and names marked "from memory" are not checked).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation Confusion matrix:::A scoreboard with four boxes that counts a model's right and wrong answers, split by the kind of mistake.

#flashcards/ml/concept #flashcards/topic/evaluation What are the four boxes TP, FP, FN and TN?::TP (true positive): really yes, predicted yes. FP (false positive): really no, predicted yes, a false alarm. FN (false negative): really yes, predicted no, a miss. TN (true negative): really no, predicted no.

#flashcards/ml/concept #flashcards/topic/evaluation Why is the confusion matrix better than accuracy alone?::A single score hides which cell the mistakes fall in. Two very different models can have the same accuracy, and accuracy can even rank a useless model first.

#flashcards/ml/cloze #flashcards/topic/evaluation In the picture's layout, ==rows== are what is really true and ==columns== are what the model predicted (some books swap them, so read the labels).

#flashcards/ml/example #flashcards/topic/evaluation Out of 100 cases a model gives $TP=8$, $FP=2$, $FN=4$, $TN=86$. What are the totals?::Total 100. Real positives TP + FN = 12. Real negatives FP + TN = 88. Predicted positives TP + FP = 10. Every other metric follows from these.

#flashcards/ml/example #flashcards/topic/evaluation With 10,000 cases of which 89 are positive, compare "always no" ($TP=0$, $FP=0$, $FN=89$, $TN=9911$) with a noisy model ($TP=67$, $FP=2000$, $FN=22$, $TN=7911$).::Always no: accuracy 0.991 but it misses all 89 positives. Noisy model: accuracy 0.798 but it finds 67 of the 89. Accuracy prefers the lazy model, which is wrong.

#flashcards/ml/cloze #flashcards/topic/evaluation A model that outputs a score has a ==different== confusion matrix for every ==threshold==, which the ROC curve summarises.

#flashcards/ml/image #flashcards/topic/evaluation ![[confusion-matrix-metrics.png]]
What does this picture show?
?
The confusion matrix: rows are what is really true, columns are what the model predicted. TP = true positive (said yes, was yes), TN = true negative, FP = false positive (said yes, was no), FN = false negative (said no, was yes). The four panels show the formulas for accuracy, precision, recall and specificity.
