---
tags: [machine-learning, features, regression, polynomial]
status: growing
level: 4
tier: 1
---

# Feature Engineering

## 🟢 1. Simple (age 12)
Feature engineering means **making better clues for the computer to learn from**. A model can only use the clues you hand it. If you give it a house's width and length, it may struggle; if you give it the **area** (width times length), the answer might become easy.

**Example:** To guess how long a trip takes, "start and end town" is a weak clue. "Distance in kilometres" is a much better one. Making the clue is the job.

## 🔵 2. How it works
1. Start with the features you have (say width and depth of a plot of land).
2. Ask: is there a combination that matters more than the pieces? Land area is width × depth, and price may follow area much more closely.
3. Add the new number as a column, or replace the old ones with it.
4. A special case lets a straight-line model draw **curves**: also give it $x^2$ (and maybe $x^3$) as extra columns. This is **polynomial regression** ([[Linear Regression]] on those columns).
5. The model is still a plain weighted sum, so the same [[Gradient Descent]] works. Because $x^2$ is much bigger than $x$, you now want [[Feature Scaling]].
6. Too many added columns lets the model memorise the training data ([[Overfitting]]). Adding them is a trade-off, and [[Regularization]] helps.

## 🟠 3. Technical detail
- **Common transforms:** products of features (area), ratios (price per square foot), powers ($x^2$, $\sqrt{x}$), logs for skewed numbers, and turning categories into 0/1 columns.
- **Why a curve is still "linear":** the model is linear in the **weights**. $w_1x+w_2x^2+b$ is a weighted sum of the columns $x$ and $x^2$, so the same fitting method applies.
- **Choosing the degree:** compare error on data the model did not train on ([[Cross-Validation]]). Higher degree always fits the training set better and eventually fits new data worse.
- **Before neural networks:** features were designed by hand for most tasks. A [[Neural Network]] learns many of its own features in its hidden layers, which is one reason it replaced much hand engineering for images and text (see [[History of Image Classification]]).
- **Tooling:** in [[NumPy]] you add a column with `np.c_`, in scikit-learn with `PolynomialFeatures`.

```python
import numpy as np
x = np.arange(0, 6.0); y = 2 + x ** 2                         # the truth is a curve
for name, A in (("x only", np.c_[x, np.ones_like(x)]), ("x and x^2", np.c_[x, x ** 2, np.ones_like(x)])):
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    print(name, np.round(c, 3), "MSE", round(float(np.mean((A @ c - y) ** 2)), 4))
# x only    [ 5.    -1.333] MSE 6.2222
# x and x^2 [-0.  1.  2.] MSE 0.0
```

**For time series (added with the [[Time Series]] notes).** To use an ordinary model on a series, build features from the past only: **lag features** (the value 1, 2, 12 steps ago), **rolling statistics** (the average of the last 7 values), and **calendar features** (month, weekday, holiday). Every feature must be computable at the moment of forecasting. A centred moving average, or a scaler fitted on the whole series, leaks the future ([[Time Series Cross-Validation]]). Which lags to use can be read from the autocorrelation plot ([[Autocorrelation]]). Models that cannot extrapolate a trend, such as trees, usually need the series to be turned into growth rates or differences first ([[Stationarity]]).

## 🔴 4. Deep / formal
**A curve from a straight-line method.** With the engineered feature $x^2$ the model becomes:

$$
f_{\vec w,b}(x)=w_1x+w_2x^2+b
$$

**Where:**
- $x$ is the original input (for example house size). $x^2$ is the new, engineered feature: the input times itself.
- $w_1$ and $w_2$ are the weights for the two columns, and $b$ is the constant added at the end.
- **Matters:** the curve bends because of $w_2$. If $w_2=0$ the model is a straight line again. The fit is still found by minimising squared error, since the model is linear in $w_1,w_2,b$.
- **Matters:** the columns have very different sizes ($x$ up to 5, $x^2$ up to 25 here, and much worse for $x^3$), so scale them ([[Feature Scaling]]) when using gradient descent.

**Example (tiny numbers):** the true rule is $y=x^2+2$. The fit finds $w_1=0$, $w_2=1$, $b=2$.
- At $x=1$: $0\cdot1+1\cdot1+2=\mathbf{3}$. At $x=2$: $0+4+2=\mathbf{6}$. At $x=3$: $0+9+2=\mathbf{11}$.
- A straight line through the same six points ($x=0,\dots,5$) is $5x-1.33$; at $x=3$ it says $13.67$, not $11$.
- Meaning: one extra column took the error from "visibly wrong" to exactly zero (on this noise-free toy).

**Limits:**
- Added features are only useful if they carry real information. Throw in 100 random products and you mostly add noise and risk [[Overfitting]].
- High-degree polynomials oscillate wildly between data points and far outside the data, so they extrapolate badly.
- Features must be computed the same way at prediction time as in training, or the model sees different inputs.

**What came before, and how this improved on it:**
- **Before:** fit a straight line to the raw measurements.
- **Its limit:** if the true relation is curved, a straight line stays wrong however much data you give it.
- **What changed:** add engineered columns ($x^2$, products, ratios) and fit the same linear method on them.
- **How much better:** in the code below, on six noise-free points from $y=x^2+2$ the straight line had mean squared error 6.2222 and the line with $x^2$ added had 0.0 (exact). On 50 noisy points from a quadratic, the degree-1 fit had error 18.297 and the degree-2 fit 0.605, about 30 times lower. (Training error only; on a toy.)

**Check it yourself:**

```python
import numpy as np
x = np.arange(0, 6.0); y = 2 + x ** 2
for name, A in (("x only", np.c_[x, np.ones_like(x)]), ("x and x^2", np.c_[x, x ** 2, np.ones_like(x)])):
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    print(name, np.round(c, 3), "MSE %.4f" % np.mean((A @ c - y) ** 2))
xs = np.linspace(0, 10, 50)
ys = 3 + 0.5 * xs ** 2 - 2 * xs + np.random.default_rng(0).normal(0, 1, 50)
for deg in (1, 2):
    c = np.polyfit(xs, ys, deg); print("noisy, degree", deg, "MSE %.3f" % np.mean((np.polyval(c, xs) - ys) ** 2))
print("area from width x depth:", np.array([20, 30, 40, 25.0, 50]) * np.array([60, 50, 40, 80, 30.0]))
```

## Related
- [[Linear Regression]] · [[Feature Scaling]] · [[Overfitting]] · [[Regularization]] · [[Cross-Validation]] · [[Neural Network]] · [[Gradient Descent]] · [[Decision Boundary]] · [[NumPy]] · [[History of Image Classification]] · [[Machine Learning Specialization Course 1 Week 2]] · [[One-Hot Encoding]] · [[Time Series]] · [[Autocorrelation]] · [[Time Series Cross-Validation]]
- [[Permutation Importance]]
## Sources
- Course videos "Machine Learning Specialization" (Andrew Ng), Course 1 Week 2 (transcripts read; see [[Machine Learning Specialization]]). The house-width times depth example follows the course.
- The 6.2222 and 18.297 vs 0.605 figures are measured in this note's code

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/data-features Feature engineering:::Making better clues for the computer to learn from. A model can only use the clues you hand it. Giving it the area (width times length) can make a hard problem easy.

#flashcards/ml/formula #flashcards/topic/data-features Write a polynomial-feature model and explain each symbol.
?
$$f_{\vec w,b}(x)=w_1x+w_2x^2+b$$
- $x$: the original input (for example house size). $x^2$: the new engineered feature, the input times itself.
- $w_1$, $w_2$: the weights for the two columns. $b$: the constant added at the end.
- The curve bends because of $w_2$. If $w_2=0$ it is a straight line again. The fit is still found by minimising squared error, since the model is linear in $w_1,w_2,b$.
- The columns have very different sizes ($x$ up to 5, $x^2$ up to 25, worse for $x^3$), so scale them when using gradient descent.

#flashcards/ml/example #flashcards/topic/data-features True rule $y=x^2+2$, fit $w_1=0$, $w_2=1$, $b=2$. What does the model give at $x=1,2,3$, and what does a straight line give at $x=3$?::$x=1$: 3. $x=2$: 6. $x=3$: 11. A straight line through the same six points ($x=0..5$) is $5x-1.33$, which says 13.67 at $x=3$ instead of 11. One extra column took the error to exactly zero (on this noise-free toy).

#flashcards/ml/concept #flashcards/topic/data-features Why should engineered columns be scaled?::They can have very different sizes, which makes gradient descent clumsy. See feature scaling.

#flashcards/ml/concept #flashcards/topic/data-features What features turn a time series into a table for an ordinary model, and what rule must they obey?
?
Lag features (the value 1, 2, 12 steps ago), rolling statistics (the average of the last 7 values) and calendar features (month, weekday, holiday). Every feature must be computable at forecast time, so no centred windows and no scaling fitted on the whole series.
