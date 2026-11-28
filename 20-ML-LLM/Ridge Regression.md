---
tags: [ridge-regression, regression, regularization, L2, linear-model, overfitting, ML]
status: growing
level: 4
tier: 2
---

# Ridge Regression

**Ridge regression** is linear regression with a penalty on how big the weights are. Plain linear regression picks the weights that fit the training numbers as closely as possible. Ridge regression picks weights that fit well **and** stay small. The penalty is the sum of the squared weights, multiplied by a number $\lambda$ that you choose. Small weights make a calmer model that reacts less to noise in the training data.

## 🟢 1. Simple (age 12)
Imagine drawing a line through dots on a graph. If you let the line wiggle as much as it wants, it will bend to touch every dot, including the dots that are just wrong by accident. Ridge regression puts a gentle rubber band on the line: it may still follow the dots, but wild bends cost points. The stronger the band, the straighter the line. A good band gives a line that works on new dots too.

**Example:** you have only 20 measurements and 15 things you could measure. A free line chases the noise and does badly on new data. With a rubber band of the right strength it does much better.

## 🔵 2. How it works
1. Start with a [[Linear Regression]] model: the prediction is the inputs times a list of weights.
2. **Score** a choice of weights with two parts: how far the predictions are from the true values (the squared errors, summed), plus $\lambda$ times the sum of the squared weights.
3. **Choose $\lambda$.** With $\lambda=0$ you get ordinary linear regression. A larger $\lambda$ pushes all weights toward zero (but never exactly to zero).
4. **Solve.** For this penalty the best weights have a direct formula, so no repeated training is needed (see level 4).
5. **Pick $\lambda$ on held-out data** ([[Cross-Validation]]), not on the training data, because the training error always prefers $\lambda=0$.
6. **Scale the inputs first** ([[Feature Scaling]]): the penalty treats all weights alike, so a feature measured in big units gets an unfair advantage if you do not.
7. **Where it appears:** it is the "L2" penalty of [[Regularization]] on a linear model; and [[R-CNN 2013]] used it, with $\lambda=1000$, to learn the box corrections of its bounding-box regression.

No image: I did not look for a free picture of ridge regression. The numbers below come from a real run.

## 🟠 3. Technical detail
Numbers come from `ridge_demo.py` (Python 3.9.6, NumPy, scikit-learn 1.1.2, CPU). It (1) repeats the one-feature toy of the R-CNN note, (2) builds a problem with 20 training points and 15 correlated features where only the first 3 features matter, and fits it with plain least squares and with ridge at several $\lambda$ values, (3) checks the closed form against scikit-learn's `Ridge`, (4) repeats over 200 random training sets, and (5) looks at the singular values of the data.

```python
"""Ridge regression: closed form against scikit-learn, shrinkage, and test error against lambda on a noisy, correlated toy problem."""
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
rng = np.random.default_rng(0)
# 1. the tiny example used in the R-CNN note: one feature, no bias, three points
phi = np.array([1., 2., 3.]); t = np.array([.1, .2, .3])
for lam in (0, 1, 1000): print("lambda", lam, "w = %.5f" % ((phi @ t) / (phi @ phi + lam)))
print("scikit-learn Ridge(alpha=1, fit_intercept=False):", Ridge(alpha=1, fit_intercept=False).fit(phi[:, None], t).coef_)
# 2. a small problem where plain least squares overfits: 20 training points, 15 correlated features, noisy target
def make(n, p, w, noise, shared):
    z = rng.standard_normal((n, 1)); X = np.sqrt(shared) * z + np.sqrt(1 - shared) * rng.standard_normal((n, p))
    return X, X @ w + noise * rng.standard_normal(n)
p = 15; w_true = np.zeros(p); w_true[:3] = (2., -1., 0.5)
Xtr, ytr = make(20, p, w_true, 1.0, 0.8); Xte, yte = make(2000, p, w_true, 1.0, 0.8)
def ridge(X, y, lam): return np.linalg.solve(X.T @ X + lam*np.eye(X.shape[1]), X.T @ y)
print("\ncondition number of X'X: %.1f ; of X'X + 10 I: %.1f" % (np.linalg.cond(Xtr.T @ Xtr), np.linalg.cond(Xtr.T @ Xtr + 10*np.eye(p))))
print("lambda | train MSE | test MSE | |w|")
for lam in (0, 0.1, 1, 10, 100, 1000, 10000):
    w = ridge(Xtr, ytr, lam) if lam > 0 else np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    print("%6g | %9.3f | %8.3f | %.3f" % (lam, np.mean((Xtr @ w - ytr)**2), np.mean((Xte @ w - yte)**2), np.linalg.norm(w)))
print("true |w| = %.3f ; noise variance 1.0 is the best possible test MSE" % np.linalg.norm(w_true))
w1 = ridge(Xtr, ytr, 10.0); sk = Ridge(alpha=10.0, fit_intercept=False).fit(Xtr, ytr)
print("closed form against scikit-learn at lambda 10: max |difference| %.2e" % np.abs(w1 - sk.coef_).max())
# 3. repeat over 200 random training sets: how often does the best lambda beat plain least squares?
wins = 0; res = {0: [], 10: [], 100: []}
for r in range(200):
    X, y = make(20, p, w_true, 1.0, 0.8)
    for lam in res:
        w = np.linalg.lstsq(X, y, rcond=None)[0] if lam == 0 else ridge(X, y, lam)
        res[lam].append(np.mean((Xte @ w - yte)**2))
print("mean test MSE over 200 training sets:", {k: round(float(np.mean(v)), 3) for k, v in res.items()}, "| lambda 10 better than least squares in %d of 200 sets" % sum(a < b for a, b in zip(res[10], res[0])))
# 4. shrinkage along the singular directions
U, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
print("\nsingular values of X:", np.round(S[:6], 2), "...", np.round(S[-3:], 2))
print("shrink factor s/(s^2+lambda) * s for lambda 10 on the 3 largest and 3 smallest:", np.round(S[:3]**2/(S[:3]**2 + 10), 3), np.round(S[-3:]**2/(S[-3:]**2 + 10), 3))
```

I saw:
```
lambda 0 w = 0.10000
lambda 1 w = 0.09333
lambda 1000 w = 0.00138
scikit-learn Ridge(alpha=1, fit_intercept=False): [0.09333333]

condition number of X'X: 1205.2 ; of X'X + 10 I: 20.6
lambda | train MSE | test MSE | |w|
     0 |     0.015 |    3.882 | 4.227
   0.1 |     0.030 |    2.863 | 3.239
     1 |     0.163 |    2.061 | 1.640
    10 |     0.489 |    2.026 | 0.531
   100 |     0.795 |    2.398 | 0.229
  1000 |     1.467 |    3.406 | 0.055
 10000 |     1.762 |    3.823 | 0.007
true |w| = 2.291 ; noise variance 1.0 is the best possible test MSE
closed form against scikit-learn at lambda 10: max |difference| 3.40e-16
mean test MSE over 200 training sets: {0: 4.852, 10: 1.793, 100: 2.252} | lambda 10 better than least squares in 196 of 200 sets

singular values of X: [14.12  3.95  3.27  2.96  2.24  1.99] ... [0.62 0.53 0.41]
shrink factor s/(s^2+lambda) * s for lambda 10 on the 3 largest and 3 smallest: [0.952 0.609 0.517] [0.037 0.027 0.016]
```

**One problem, several $\lambda$ (20 training points, 15 correlated features, 2,000 test points).**

| $\lambda$ | Training error | Test error | Size of the weights |
|---|---|---|---|
| 0 (plain least squares) | 0.015 | 3.882 | 4.227 |
| 0.1 | 0.030 | 2.863 | 3.239 |
| 1 | 0.163 | 2.061 | 1.640 |
| 10 | 0.489 | 2.026 | 0.531 |
| 100 | 0.795 | 2.398 | 0.229 |
| 1,000 | 1.467 | 3.406 | 0.055 |
| 10,000 | 1.762 | 3.823 | 0.007 |

- **Training error / test error:** the mean squared error of the predictions on the 20 training points and on 2,000 fresh points. **Size of the weights:** the length of the weight vector (the true one has length 2.291).
- The noise in the data has variance 1.0, so 1.0 is the best any model could do on the test points.

**Summary: what the run shows.** Plain least squares fits the training points almost perfectly (0.015) and is bad on new data (3.882). The training error rises steadily with $\lambda$ while the test error falls to a minimum at $\lambda=10$ (2.026) and then rises again as the model becomes too stiff (3.823 at $\lambda=10{,}000$, where the weights are almost zero). Over 200 random training sets of the same kind, the mean test error was 4.852 for least squares, 1.793 for $\lambda=10$ and 2.252 for $\lambda=100$, and $\lambda=10$ beat least squares in 196 of 200 sets. The closed form matched scikit-learn to 3.4e-16. The matrix $X^{\top}X$ had a condition number of 1,205, and $X^{\top}X+10I$ of 20.6. Caution: synthetic data made by me, one problem shape, no intercept; $\lambda$ was read off the test points here to show the curve, which a real project must not do (use [[Cross-Validation]]).

## 🔴 4. Deep / formal
**The objective and its closed form.**

$$
\hat{\mathbf w}=\arg\min_{\mathbf w}\ \lVert \mathbf y-X\mathbf w\rVert^{2}+\lambda\lVert\mathbf w\rVert^{2}\;=\;\bigl(X^{\top}X+\lambda I\bigr)^{-1}X^{\top}\mathbf y
$$

**Where:**
- $X$: the $n\times p$ matrix of inputs (one row per example, one column per feature); $\mathbf y$: the $n$ targets.
- $\mathbf w$: the $p$ weights; $\lVert\mathbf y-X\mathbf w\rVert^{2}$: the sum of squared errors.
- $\lambda\ge0$: the penalty strength; $\lVert\mathbf w\rVert^{2}$: the sum of squared weights; $I$: the identity matrix.

**Matters:**
- The $\lambda\lVert\mathbf w\rVert^{2}$ term makes big weights expensive, which shrinks them and reduces the effect of noise.
- The $+\lambda I$ in the formula makes the matrix always invertible (for $\lambda>0$) and better conditioned: in my run 1,205 against 20.6 for $\lambda=10$. Without it, correlated features make $X^{\top}X$ nearly singular, and tiny changes in the data swing the weights.
- The bias (intercept) is normally left out of the penalty; my demo has no intercept at all.

**Example (tiny numbers):** one feature, no intercept, three points $(\phi,t)=(1,0.1),(2,0.2),(3,0.3)$ (the same toy as in [[R-CNN 2013]]).
- Here $X^{\top}X=1+4+9=14$ and $X^{\top}\mathbf y=0.1+0.4+0.9=1.4$.
- The formula gives $w=1.4/(14+\lambda)$.
- $\lambda=0$: $0.100000$. $\lambda=1$: $0.093333$ (the same as scikit-learn's `Ridge(alpha=1)`). $\lambda=1000$: $0.001381$.
- Meaning: the data lie exactly on a line with slope 0.1; a large $\lambda$ shrinks that slope almost to zero. The paper's $\lambda=1000$ is large for this toy, but its features and examples number in the thousands, which is a different scale.

**The singular-value view (my run).** With the singular value decomposition $X=U\Sigma V^{\top}$, ridge multiplies the fit along each direction by $\sigma_i^{2}/(\sigma_i^{2}+\lambda)$: directions with a large singular value $\sigma_i$ are hardly touched and directions with a small one (the noisy ones) are shrunk the most. In my run with $\lambda=10$: the three largest singular values (14.12, 3.95, 3.27) kept 0.952, 0.609 and 0.517 of their size; the three smallest (0.62, 0.53, 0.41) kept 0.037, 0.027 and 0.016. This is why ridge helps with correlated features: the weak, noisy directions are the ones it removes ([[Singular Value Decomposition]], [[PCA]]).

**Edge cases and details:**
- **Scale matters:** $\lambda$ is not comparable across data sets or after rescaling the features, so standardise first ([[Feature Scaling]]).
- **Lasso (L1) is different:** an L1 penalty can set weights exactly to zero; ridge only shrinks them (general knowledge, not tested here; see [[Regularization]]).
- **Weight decay** in neural-network training is the same L2 idea applied to weights by gradient descent (the AlexNet paper uses 0.0005; see [[AlexNet 2012]]).

**Limits and trade-offs:**
- It adds a number to tune, and a wrong $\lambda$ is worse than none (3.823 at $\lambda=10{,}000$ against 3.882 for least squares in my run: about the same, and both far above the best).
- It keeps all features (no selection) and assumes a linear relation.
- It trades a little bias for a lot less variance; with many examples and few features the gain is small.

**What came before, and how this improved on it:**
- **Before:** ordinary least squares, the weights that minimise the squared error on the training data alone.
- **Its limit:** with few examples, many features or correlated features it fits the noise: in my run the training error was 0.015 but the test error 3.882.
- **What changed:** add $\lambda\lVert\mathbf w\rVert^{2}$ to the objective, which still has a closed form.
- **How much better:** in my run the test error fell from 3.882 to 2.026 at $\lambda=10$ (48% lower, my arithmetic) and the mean over 200 training sets from 4.852 to 1.793 (63% lower). This is on my synthetic data; I did not run it on a real data set.

## Related
[[Linear Regression]] · [[Regression]] · [[Regularization]] · [[Overfitting]] · [[Cross-Validation]] · [[Feature Scaling]] · [[Singular Value Decomposition]] · [[PCA]] · [[R-CNN 2013]] · [[AlexNet 2012]] · [[Matrix]]

## Sources
- My own run: `ridge_demo.py` (Python 3.9.6, NumPy, scikit-learn 1.1.2), including the comparison with `sklearn.linear_model.Ridge`.
- Use in practice: [[R-CNN 2013]], Appendix C (read there): ridge regression "solved efficiently in closed form" with $\lambda=1000$ chosen on a validation set.
- The formula, the singular-value view and the lasso comparison are general knowledge; I did not check them against a textbook. The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/ml-foundations Ridge regression:::Linear regression whose objective adds lambda times the sum of squared weights, so weights are shrunk toward zero; it has a closed-form solution.

#flashcards/ml/concept #flashcards/topic/ml-foundations Why does ridge regression help when there are few examples or correlated features?::Plain least squares chases noise and its matrix X'X is nearly singular; the penalty shrinks the weights (most along noisy directions) and +lambda I makes the matrix well conditioned, so predictions on new data are steadier.

#flashcards/ml/concept #flashcards/topic/ml-foundations How do you choose lambda in ridge regression?::On held-out data or by cross-validation, because the training error always prefers lambda = 0; and standardise the features first, since the penalty treats all weights alike.

#flashcards/ml/formula #flashcards/topic/ml-foundations Write the ridge regression objective and its solution, and explain each part.
?
$$\hat{\mathbf w}=\arg\min_{\mathbf w}\lVert\mathbf y-X\mathbf w\rVert^{2}+\lambda\lVert\mathbf w\rVert^{2}=(X^{\top}X+\lambda I)^{-1}X^{\top}\mathbf y$$
- $X$: n by p inputs; $\mathbf y$: n targets; $\mathbf w$: p weights.
- $\lambda\ge0$: penalty strength; $\lambda=0$ is plain least squares.
- The $+\lambda I$ keeps the matrix invertible and better conditioned (1,205 against 20.6 for lambda 10 in my run); the penalty shrinks weights and cuts the effect of noise. The intercept is normally left out of the penalty.

#flashcards/ml/example #flashcards/topic/ml-foundations One feature, no intercept, points (1, 0.1), (2, 0.2), (3, 0.3): what is the ridge weight for lambda 0, 1 and 1000?::w = 1.4 / (14 + lambda): 0.1000, 0.0933 and 0.00138. The same value as scikit-learn's Ridge at lambda 1 (my arithmetic and run).

#flashcards/ml/concept #flashcards/topic/ml-foundations What does ridge do along the singular directions of the data?::It multiplies the fit along direction i by s^2/(s^2 + lambda): large singular values are barely touched, small (noisy) ones shrink most. In my run with lambda 10 the largest three kept 0.95, 0.61, 0.52 and the smallest three 0.04, 0.03, 0.02.

#flashcards/ml/example #flashcards/topic/ml-foundations On my toy with 20 training points and 15 correlated features, what happened to the test error as lambda grew from 0 to 10,000?::3.882 at 0, down to a minimum of 2.026 at lambda 10, then back up to 3.823 at 10,000 as the weights vanish; the training error rose steadily from 0.015 to 1.762 (synthetic data, lambda read off the test set to show the curve).

#flashcards/ml/concept #flashcards/topic/ml-foundations How does ridge differ from lasso and from weight decay?::Lasso uses an L1 penalty and can set weights exactly to zero; ridge (L2) only shrinks them. Weight decay in neural-network training is the same L2 idea applied by gradient descent (general knowledge).

#flashcards/ml/concept #flashcards/topic/ml-foundations Where did R-CNN use ridge regression?::To learn its bounding-box regressors: a linear map from the proposal's pool5 features to four box corrections, with lambda = 1000 chosen on a validation set and solved in closed form (R-CNN paper, Appendix C).

#flashcards/ml/cloze #flashcards/topic/ml-foundations Ridge regression adds ==lambda times the sum of squared weights== to the squared error, which ==shrinks== the weights toward zero.
