---
tags: [machine-learning, evaluation, metrics, ROC, AUC, F1, cross-validation]
status: growing
level: 4
tier: 2
---

# Evaluation Metrics

This note is the **overview**. It shows how the scorecards fit together and links to one note for each: [[Confusion Matrix]], [[Accuracy]], [[Precision]], [[Recall]], [[Specificity]], [[F1 Score]], [[ROC Curve and AUC]], [[Precision-Recall Curve]], [[Calibration]] and [[Cross-Validation]].

## 🟢 1. Simple (age 12)
Evaluation is **checking how good a model really is**, with fair tests. It is not enough to ask "how many did it get right?", because a lazy model can score well by always guessing the common answer. So we use **better scorecards** (different numbers that catch different kinds of mistakes), and we test the model on **questions it has never seen**, just as a teacher uses new questions on the exam instead of the practice sheet.

**Example:** A doctor's test for a rare illness. If only 1 in 100 people has it, a "test" that always says "healthy" is right 99% of the time, and completely useless. A good scorecard asks: "Of the people who were really ill, how many did you catch?" and "When you said ill, how often were you right?". Those questions expose the useless test straight away.

## 🔵 2. How it works
1. **Keep some data hidden.** Split the examples into a **training** pile (to learn from), a **validation** pile (to choose settings) and a **test** pile (used once, at the very end). See [[Supervised Learning]].
2. **Pick a scorecard that matches the job.**
   - For a **number** ([[Regression]]): the average size of the miss, such as mean squared error, or $R^2$.
   - For a **category** ([[Classification]]): [[Accuracy]], [[Precision]], [[Recall]], [[F1 Score]], and the [[ROC Curve and AUC]].
3. **Count the four kinds of answers** in a [[Confusion Matrix]]: true positives, false positives, false negatives, true negatives. Every category score below is made from these four counts (see the picture).
4. **[[Precision]] and [[Recall]] pull in opposite directions.** Making the model say "yes" more easily finds more real cases (higher recall) but raises false alarms (lower precision). [[F1 Score]] squeezes both into one number.
5. **Many classifiers give a score, not just yes or no.** You then choose a **threshold**: "say yes if the score is above 0.7". The [[ROC Curve and AUC]] shows how well the model does at *every* threshold at once, and AUC is one number summarising the whole curve.
6. [[Cross-Validation]] gives a steadier estimate when you have little data: split the data into $k$ pieces, hold out one piece at a time as the test, train on the rest, and average the $k$ results.
7. **Look at the mistakes**, not just the scores: which examples does it get wrong, and for which groups of people or situations?

![[confusion-matrix-metrics.png]]
*The confusion matrix: rows are what is really true, columns are what the model predicted. TP = true positive (said yes, was yes), TN = true negative, FP = false positive (said yes, was no), FN = false negative (said no, was yes). The four panels show the formulas for accuracy, precision, recall and specificity: each circle or oval marks the cells that the formula uses. Image: Hssiqueira, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Confusion_Matrix_Metrics.png), CC0.*

## 🟠 3. Technical detail
- **For categories, from the confusion matrix** (TP, FP, FN, TN):
  - **[[Accuracy]]** = (TP + TN) / everything. Misleading when one class is rare.
  - **[[Precision]]** = TP / (TP + FP). **[[Recall]]** (also called sensitivity or true positive rate) = TP / (TP + FN).
  - **[[Specificity]]** = TN / (TN + FP). **False positive rate** = 1 − specificity.
  - **[[F1 Score]]** combines precision and recall. **F-beta** weighs recall more or less than precision.
  - **Multi-class:** compute per class, then average (macro: each class counts equally; micro: each example counts equally).
- **For scores:**
  - **[[ROC Curve and AUC|ROC curve]]:** true positive rate against false positive rate as the threshold moves. **AUC** is the area under it: 1 is perfect, 0.5 is random guessing.
  - **Precision-recall curve:** precision against recall as the threshold moves. It is usually more informative than ROC when positives are rare (a common recommendation; the reasoning is in the references).
  - **Log loss** (cross-entropy) judges the probabilities themselves, and punishes confident wrong answers heavily (see [[Loss Function]]).
  - **Calibration:** do the model's "80%" predictions come true about 80% of the time?
- **For numbers:** mean squared error, root mean squared error, mean absolute error (less thrown by outliers), and $R^2$ ([[Regression]]).
- **For language models:** [[Perplexity]], plus task-specific scores and human judgement. Choosing good tests for [[LLM]]s is an active research area.
- **For retrieval systems** such as [[RAG]]: whether the right documents were found (recall at k) and whether the final answer was correct and supported.
- **[[Cross-Validation]] variants:**
  - **k-fold** (often $k=5$ or $10$; a common habit).
  - **Stratified:** each fold keeps the same share of each class.
  - **Leave-one-out:** $k=n$, one example held out each time. Very thorough, but expensive.
  - **Time-ordered splits** for data over time: always train on the past and test on the future.
- **Common traps:**
  - **Leakage:** information about the answer leaks into the training inputs, or the same item appears in both train and test. Scores look great and mean nothing.
  - **Tuning on the test set:** every time you look and adjust, you quietly overfit to it ([[Overfitting]]). Use validation data for choices.
  - **Wrong metric for the cost of mistakes:** a missed disease and a false alarm rarely cost the same.
  - **Not looking at uncertainty:** a difference of 0.5 points on a small test set may be pure luck.
  - **Averages hide groups:** check results for different subgroups ([[Bias-Variance]] explains part of why scores differ between training and test).

```python
import numpy as np

# F1 from the Classification note's confusion matrix: TP=8, FP=2, FN=4
P, R = 8 / 10, 8 / 12
print(round(2 * P * R / (P + R), 3), round(2 * 8 / (2 * 8 + 2 + 4), 3))     # 0.727 0.727   (two equal ways to write F1)
print(round(2 * 1 * .1 / (1 + .1), 3), (1 + .1) / 2)                        # 0.182 0.55    F1 vs plain average for P=1, R=0.1

# ROC and AUC from scores: 4 real positives, 4 real negatives
pos = np.array([.9, .8, .6, .4]); neg = np.array([.7, .5, .3, .2])
for t in (0.85, 0.65, 0.45, 0.25):                                           # a few thresholds
    print(t, (pos >= t).mean(), (neg >= t).mean())                          # threshold, TPR, FPR
# 0.85 0.25 0.0
# 0.65 0.5 0.25
# 0.45 0.75 0.5
# 0.25 1.0 0.75
print(sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / 16)         # 0.8125   AUC = share of (pos, neg) pairs ranked correctly

# leave-one-out cross-validation for the straight line from the Regression note
x = np.array([0., 1., 2.]); y = np.array([1., 3., 2.]); errs = []
for i in range(3):
    keep = np.arange(3) != i
    w, b = np.polyfit(x[keep], y[keep], 1)                                   # fit on two points, test on the third
    errs.append((w * x[i] + b - y[i]) ** 2)
print(np.round(errs, 3), round(float(np.mean(errs)), 3))                     # [9.   2.25 9.  ] 6.75
w, b = np.polyfit(x, y, 1); print(round(float(((w * x + b - y) ** 2).mean()), 3))   # 0.5   the training error, for comparison
```

**For forecasts (added with the [[Time Series]] notes).** Accuracy-style scores do not apply to numbers over time. Use the average miss (MAE), the root mean squared miss (RMSE), a percent error (MAPE) or the scaled error MASE, always on forward-moving folds, and always next to the naive forecast ("same as last time"). MAPE breaks when a true value is zero, and MASE does not. See [[Forecast Metrics]] and [[Time Series Cross-Validation]].

## 🔴 4. Deep / formal
**Which scorecard for which question.** The equations now live in the note of each metric, so this overview only helps you choose.

| The question | The scorecard | Its note |
|---|---|---|
| Which kind of mistakes does the model make? | the four counts | [[Confusion Matrix]] |
| How many answers are right overall? (balanced classes) | accuracy | [[Accuracy]] |
| When it says yes, can I trust it? | precision | [[Precision]] |
| Did it find the real cases? | recall (sensitivity) | [[Recall]] |
| Does it leave the real no-cases alone? | specificity | [[Specificity]] |
| One number for precision and recall together | F1 | [[F1 Score]] |
| How good is the ranking, at every threshold? | ROC curve and AUC | [[ROC Curve and AUC]] |
| Positives are rare: how good is the trade-off? | precision-recall curve | [[Precision-Recall Curve]] |
| Can I believe the chances it gives? | calibration, ECE | [[Calibration]] |
| Is my estimate steady, with little data? | k-fold cross-validation | [[Cross-Validation]] |
| How far off are my numbers? (regression) | mean squared error, $R^2$ | [[Regression]] |
| How surprised is the language model? | perplexity | [[Perplexity]] |

**One data set, the scorecards side by side.** With 10,000 cases of which 1% are positive, the lazy model that always says "no" has an accuracy of 0.991, a recall of 0 and an F1 of 0. The noisy model, wrong 20% of the time, has an accuracy of 0.798, a recall of 0.753 and a precision of only 0.032. The numbers are from the check at the end of this note.

**Further notes:**
- **Choosing a metric is a decision about costs:** if a missed case is ten times worse than a false alarm, choose the threshold (or a weighted score) to match, and say so.
- **Confidence intervals:** report a range, not a single number, for example by repeating the evaluation on bootstrapped test sets ([[Ensemble Methods]] describes the bootstrap idea). The range is wide when the test set is small.
- **Distribution shift:** a test set taken from the same source as the training data can look far better than real life, where the data changes over time.
- **Fairness checks:** compare error rates between groups. Equal overall accuracy can hide very different error rates.
- **Not covered yet in this vault:** significance testing, calibration curves in detail, and metrics for ranking and recommendation.

**What came before, and how this improved on it:**
- **Before:** report accuracy alone.
- **Its limit:** when one class is rare, doing nothing looks excellent.
- **What changed:** look at precision (how many alarms were real), recall (how many real cases were caught) and F1, which combines the two. Precision and recall come from information retrieval in the 1950s and 1960s (from memory, not checked).
- **How much better:** in the code below, with 10,000 cases of which 1% are positive, a model that always says "no" has an accuracy of 0.991 and a recall and F1 of 0. A noisy model that is wrong 20% of the time has a lower accuracy, 0.798, but catches 75% of the real cases (recall 0.753), although its precision is only 0.032. Accuracy alone ranks the useless model first.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
y = (rng.random(10000) < 0.01).astype(int)           # 1% positives
always_no = np.zeros_like(y)
rng2 = np.random.default_rng(1); model = np.where(rng2.random(10000) < 0.8, y, 1 - y)  # 20% wrong everywhere
def report(name, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn); f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print(f"{name:22s} accuracy {np.mean(pred == y):.3f}  precision {prec:.3f}  recall {rec:.3f}  F1 {f1:.3f}")
print("10,000 cases, 1% are positive:")
report("always says 'no'", always_no); report("noisy model", model)
```

## Related
- [[Confusion Matrix]] · [[Accuracy]] · [[Precision]] · [[Recall]] · [[Specificity]] · [[F1 Score]] · [[ROC Curve and AUC]] · [[Cross-Validation]] · [[Classification]] · [[Regression]] · [[Supervised Learning]] · [[Overfitting]] · [[Bias-Variance]] · [[Loss Function]] · [[Probability]] · [[Perplexity]] · [[LLM]] · [[RAG]] · [[Ensemble Methods]] · [[Machine Learning]] · [[Regularization]] · [[Logistic Regression]] · [[Naive Bayes]] · [[Random Forest]] · [[Pandas]] · [[Matplotlib]] · [[Sorting]] · [[Pass at k]] · [[LLM as a Judge]] · [[Retrieval Metrics]] · [[Benchmarks]] · [[Precision-Recall Curve]] · [[Calibration]] · [[Heart Disease Classifier]] · [[Forecast Metrics]] · [[Time Series]]
- [[Confidence Interval]] · [[Bootstrap]] · [[Hypothesis Testing]]
- [[Permutation Importance]] · [[Interpretability]]
- [[AB Testing|A/B Testing]]
- [[Mean Average Precision]]
## Sources
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online), chapters 4 and 5
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (free online), chapter 7
- Fawcett, *An introduction to ROC analysis*, 2006 (from memory)
- Saito & Rehmsmeier, *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*, 2015 (from memory)
- Bengio & Grandvalet, *No Unbiased Estimator of the Variance of K-Fold Cross-Validation*, 2004 (from memory)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation Evaluation:::Checking how good a model really is with fair tests. Asking only "how many did it get right?" is not enough, because a lazy model can score well by always guessing the common answer.

#flashcards/ml/concept #flashcards/topic/evaluation What are the training, validation and test piles for?::Training: to learn from. Validation: to choose settings. Test: used once at the very end for an honest score.

#flashcards/ml/concept #flashcards/topic/evaluation Which scorecard should you use for a number (regression) and for a category (classification)?::For a number: the average size of the miss, such as mean squared error or R². For a category: accuracy, precision, recall, F1 and the ROC curve.

#flashcards/ml/concept #flashcards/topic/evaluation Which metrics are made from the confusion matrix counts?::Accuracy = (TP + TN)/everything. Precision = TP/(TP + FP). Recall = TP/(TP + FN). Specificity = TN/(TN + FP). F1 combines precision and recall. False positive rate = 1 − specificity.

#flashcards/ml/cloze #flashcards/topic/evaluation Precision and recall pull in ==opposite== directions, and ==F1== squeezes both into one number.

#flashcards/ml/concept #flashcards/topic/evaluation What is the difference between macro and micro averaging for many classes?::Compute the score per class, then average. Macro: each class counts equally. Micro: each example counts equally.

#flashcards/ml/concept #flashcards/topic/evaluation What are the common traps in evaluation?::Leakage (information about the answer leaks into the inputs, or the same item is in both train and test). Tuning on the test set (you quietly overfit to it). The wrong metric for the cost of mistakes. Ignoring uncertainty on small test sets. Averages hiding differences between groups.

#flashcards/ml/example #flashcards/topic/evaluation What does a doctor's test that always says "healthy" show about accuracy for a disease 1 in 100 people have?::It is right 99% of the time and completely useless. Better scorecards ask "of the really ill, how many did you catch?" and "when you said ill, how often were you right?".

#flashcards/ml/image #flashcards/topic/evaluation ![[confusion-matrix-metrics.png]]
What does this picture show?
?
The confusion matrix: rows are what is really true, columns are what the model predicted. TP = true positive (said yes, was yes), TN = true negative, FP = false positive (said yes, was no), FN = false negative (said no, was yes). The four panels show the formulas for accuracy, precision, recall and specificity.

#flashcards/ml/concept #flashcards/topic/evaluation Which scores are used for forecasts, and what must they be compared with?
?
MAE, RMSE, MAPE and MASE, on forward-moving folds. They must be compared with the naive forecast ("same as last time") or the seasonal naive. MAPE breaks when a true value is zero, and MASE does not.
