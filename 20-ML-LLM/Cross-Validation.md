---
tags: [machine-learning, evaluation, metrics, cross-validation, k-fold]
status: growing
level: 4
tier: 2
---

# Cross-Validation

**Cross-validation** tests a model several times on different hidden pieces of the data and averages the results. Back to the overview: [[Evaluation Metrics]].

## 🟢 1. Simple (age 12)
Imagine you want to know how well you will do on an exam. One practice test might be unusually easy or hard. If you take five different practice tests and average the marks, you get a steadier idea. Cross-validation does that for a model, using the data it already has.

**Example:** A class of 100 students takes a quiz that you cut into 5 parts of 20 questions. Each time you hide one part, study the other four, and take the hidden part as the test. After five rounds, every question has been the test once, and you average the five marks.

## 🔵 2. How it works
1. Shuffle the examples and cut them into $k$ equal pieces, called **folds** (often $k=5$ or $10$).
2. Hold out fold 1 as the **test** piece. Train the model on the other $k-1$ folds.
3. Score the model on the held-out fold.
4. Repeat, holding out a different fold each time, until every fold has been the test piece once.
5. Average the $k$ scores. The spread of the $k$ scores also tells you how unsteady the result is.
6. Choose settings with the cross-validation scores, then keep a final test set that you use only once ([[Evaluation Metrics]], [[Overfitting]]).
7. **The catch:** it costs $k$ trainings instead of one, and the folds overlap in their training data, so the scores are not fully independent.

![[k-fold-cross-validation.svg]]
*K-fold cross-validation. The whole row of examples is cut into $k$ pieces. In iteration 1 the first piece (boxed) is the test data and the rest is training data. In iteration 2 a different piece is held out, and so on until every piece has been the test piece once. The $k$ test scores are averaged. Image: Gufosowa, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:K-fold_cross_validation_EN.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Variants:**
  - **k-fold:** the default, often $k=5$ or $10$ (a common habit).
  - **Stratified:** each fold keeps the same share of each class, which matters for rare classes ([[Classification]]).
  - **Leave-one-out:** $k=n$, one example held out each time. Very thorough but expensive.
  - **Time-ordered splits:** for data over time, always train on the past and test on the future.
- **Use it for:** estimating how well a model will do on new data, comparing models, and choosing settings such as the strength of [[Regularization]].
- **Leakage warning:** do any scaling or feature choosing inside each training fold, not before the split, or information from the test fold leaks in.
- **Not for everything:** for very large models (such as [[LLM]]s) one held-out set is used, since many trainings are too costly.

**For time series (added with the [[Time Series]] notes).** Ordinary $k$-fold cross-validation shuffles the rows, which is wrong when the rows are ordered in time: the model trains on the future and tests on the past, and it never faces the situation of forecasting beyond what it has seen. Use forward-moving folds instead ([[Time Series Cross-Validation]]). In a measured example (a random forest with calendar features on the airline series) random folds reported an RMSE of 26.7 and forward folds 65.9, so the honest error was 2.5 times larger. With only past values as features the two agreed, so the damage depends on the features.

## 🔴 4. Deep / formal
**Cross-validation: a fairer estimate from small data.** Split the data into $k$ folds. For each fold $j$, train on the other $k-1$ folds, measure the error $E_j$ on fold $j$, and average:

$$
\text{CV}=\frac1k\sum_{j=1}^{k}E_j
$$

**Where:**
- $k$ is the number of folds. $E_j$ is the model's error (or score) on fold $j$, when that fold was **not** used for training.
- $\frac1k\sum_j$ is the average of the $k$ results.
- **Matters:** every example gets used for testing exactly once and for training $k-1$ times, so little data is wasted and the estimate depends less on one lucky or unlucky split. The spread of the $E_j$ also shows how unstable the result is.
- **Caution:** the $k$ models are trained on overlapping data, so the folds are not independent, and the spread of the $E_j$ understates the true uncertainty a little (a known caveat; from memory). Any choices made from the results (such as picking the best model) need a final untouched test set.

**Example (tiny numbers):** leave-one-out cross-validation ($k=n=3$) for the straight-line fit to $(0,1),(1,3),(2,2)$ from the [[Regression]] note.
- Hold out $(0,1)$: the line through the other two points is $\hat y=-x+4$, which predicts $4$ at $x=0$. Squared error $(4-1)^2=\mathbf{9}$.
- Hold out $(1,3)$: the line through the other two is $\hat y=0.5x+1$, predicting $1.5$ at $x=1$. Squared error $(1.5-3)^2=\mathbf{2.25}$.
- Hold out $(2,2)$: the line through the other two is $\hat y=2x+1$, predicting $5$ at $x=2$. Squared error $(5-2)^2=\mathbf{9}$.
- Average: $(9+2.25+9)/3=\mathbf{6.75}$.
- Meaning: the line fitted to all three points had a **training** error of just $0.5$, but the cross-validated error is $6.75$. With only three points the model badly overstates how good it is on unseen data, which is exactly what the honest test is meant to reveal.

**Further notes:**
- **Bias and variance of the estimate:** a small $k$ trains on less data in each round, so it tends to be a bit pessimistic. A large $k$ is closer to the full-data model but costs more (from memory; see [[Bias-Variance]]).
- **Report the spread:** the scores of the folds show how unsure the estimate is.

**What came before, and how this improved on it:**
- **Before:** one split of the data into a training part and a test part.
- **Its limit:** with little data, one split depends on luck. A different shuffle gives a different score.
- **What changed:** use every example for testing exactly once. Cross-validation in its modern form is credited to Stone and to Geisser in the 1970s (from memory, not checked).
- **How much better:** in the code below (120 examples), one 75/25 split gave accuracies from 0.900 to 1.000 across 30 shuffles, a spread of 0.100. The average of 5-fold cross-validation gave 0.925 to 0.958 across 30 shuffles, a spread of 0.033, about three times narrower. **Honest limit:** it is steadier, not exact, and the leave-one-out example shows the opposite danger: the 3-point line has a training error of 0.5 and a cross-validated error of 6.75.

**Check it yourself:**

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X, y = load_breast_cancer(return_X_y=True); X, y = X[:120], y[:120]            # only 120 examples, so luck matters
model = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
single = [model().fit(*[a for a in train_test_split(X, y, test_size=0.25, random_state=s)][::2]).score(*[a for a in train_test_split(X, y, test_size=0.25, random_state=s)][1::2]) for s in range(30)]
print(f"one 75/25 split, repeated with 30 different shuffles: accuracy ranges from {min(single):.3f} to {max(single):.3f}")
means = []
for s in range(30):
    sc = [model().fit(X[a], y[a]).score(X[b], y[b]) for a, b in KFold(5, shuffle=True, random_state=s).split(X)]; means.append(np.mean(sc))
print(f"5-fold cross-validation (average of 5 scores), repeated with 30 shuffles: from {min(means):.3f} to {max(means):.3f}")
x = np.array([0., 1., 2.]); yy = np.array([1., 3., 2.]); errs = []
for i in range(3):
    keep = np.arange(3) != i; w, b = np.polyfit(x[keep], yy[keep], 1); errs.append((w * x[i] + b - yy[i]) ** 2)
w, b = np.polyfit(x, yy, 1)
print("leave-one-out on the 3-point line:", np.round(errs, 2), "average", round(float(np.mean(errs)), 2), "vs training error", round(float(((w * x + b - yy) ** 2).mean()), 2))
```

## Related
[[Evaluation Metrics]] · [[Overfitting]] · [[Bias-Variance]] · [[Regularization]] · [[Supervised Learning]] · [[Classification]] · [[Regression]] · [[Learning Curve]] · [[Heart Disease Classifier]] · [[Time Series Cross-Validation]] · [[Time Series]] · [[Experiment Tracking]] · [[Confidence Interval]]
## Sources
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online), chapters 4 and 5
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (free online)
- Image: see the caption. Every number in the history block was computed by the code in this note (toy data; dates and names marked "from memory" are not checked).
- Bengio & Grandvalet, *No Unbiased Estimator of the Variance of K-Fold Cross-Validation*, 2004 (from memory)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation Cross-validation:::Taking several practice tests instead of one and averaging the marks. It tests a model on data it did not train on, using all the data you have.

#flashcards/ml/formula #flashcards/topic/evaluation Write the cross-validation score and explain each symbol.
?
$$\text{CV}=\frac1k\sum_{j=1}^{k}E_j$$
- $k$: the number of folds. $E_j$: the model's error (or score) on fold $j$, when that fold was not used for training.
- $\frac1k\sum_j$: the average of the $k$ results.
- Every example is used for testing exactly once and for training $k-1$ times, so little data is wasted and the estimate is steadier than a single split.

#flashcards/ml/cloze #flashcards/topic/evaluation In $k$-fold cross-validation each example is tested ==once== and used for training ==k − 1== times.

#flashcards/ml/example #flashcards/topic/evaluation For the points $(0,1),(1,3),(2,2)$ and a straight-line fit, what is the leave-one-out error?::Hold out (0,1): the line through the others is ŷ = −x + 4, predicts 4, squared error 9. Hold out (1,3): the line is ŷ = 0.5x + 1, predicts 1.5, squared error 2.25. Hold out (2,2): the line is ŷ = 2x + 1, predicts 5, squared error 9. Average 20.25/3 = 6.75 (the last two are my arithmetic).

#flashcards/ml/image #flashcards/topic/evaluation ![[k-fold-cross-validation.svg]]
What does this picture show?
?
K-fold cross-validation. The row of examples is cut into k pieces. In iteration 1 the first piece is the test data and the rest is training data. In iteration 2 a different piece is held out, and so on until every piece has been the test piece once. The k test scores are averaged.

#flashcards/ml/concept #flashcards/topic/evaluation Why is ordinary shuffled cross-validation unsafe for time series?
?
It trains on the future and tests on the past, so the model never has to forecast beyond what it has seen. In a measured example (random forest with calendar features, airline series) random folds gave RMSE 26.7 and forward folds 65.9, 2.5 times larger. Use forward-moving folds.
