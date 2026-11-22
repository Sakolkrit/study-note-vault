---
tags: [machine-learning, supervised-learning, regression, least-squares, statistics]
status: growing
level: 4
tier: 2
---

# Linear Regression

Linear regression fits the **best straight line** through a cloud of points, so you can predict a number from other numbers.

## 🟢 1. Simple (age 12)
You notice that the more hours you study, the higher your test score. If you plot every friend as a dot (hours across, score up), the dots drift upwards. Linear regression draws **one straight line through the middle of the dots**, as close to all of them as it can. Then, to guess the score for someone who studied 4 hours, you just read the height of the line at 4.

**Example:** A lemonade stand sells more cups on hotter days. Over a week you write down the temperature and the cups sold. Draw the best line through those 7 dots. Tomorrow's forecast says 30 degrees, so you read the line at 30 and know roughly how many cups to prepare.

## 🔵 2. How it works
1. Collect examples where you know both the input (say, hours studied) and the answer (the score). This is [[Supervised Learning]], and since the answer is a number it is [[Regression]].
2. A line has two knobs: a **slope** (how steeply it rises) and an **intercept** (where it starts). Together these are the model's **weights**.
3. For any choice of knobs, each dot is a little above or below the line. That gap is the dot's **error** (or residual).
4. Square each error (so ups and downs do not cancel, and big misses count extra) and average them. This is the **mean squared error**, a [[Loss Function]].
5. Choose the slope and intercept that make that average as small as possible. This is called **least squares**. For a line there is a direct formula, or you can walk downhill step by step with [[Gradient Descent]].

![[linear-regression-residuals.svg]]
*Top: a straight line fitted to 36 points, with the small red vertical lines marking each point's error. Bottom: those errors (r) plotted again. They scatter evenly around zero, which is what a good straight-line fit should look like. Image: Justinkunimune, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Linear_regression_residuals.svg), CC0.*

## 🟠 3. Technical detail
- **With many inputs:** the line becomes a flat sheet (a plane, or a "hyperplane"): $\hat y=w_1x_1+w_2x_2+\dots+b$. In vector form it is $\hat y=\mathbf w\cdot\mathbf x+b$ (see [[Vector]]).
- **Three ways to solve it:**
  - the closed formula for one input (below);
  - the **normal equation** $\mathbf w=(X^{\top}X)^{-1}X^{\top}\mathbf y$ for many inputs (see [[Matrix]]), which solves the whole thing in one go but gets slow and unstable with many features or features that copy each other;
  - [[Gradient Descent]] or [[Stochastic Gradient Descent]], which scales to huge data.
- **R squared:** a score for the fit. It is 1 minus (the squared errors of your line) divided by (the squared errors of just guessing the average). It is 1 for a perfect fit and about 0 when your line is no better than the average.
- **Weak spots:**
  - **Outliers:** because errors are squared, one wild point drags the line a lot (shown in the code).
  - **Curves:** a straight line cannot follow a bend. You can add features such as $x^2$ to bend it, which is still "linear" in the weights.
  - **Too many features:** the fit starts to memorise the noise. See [[Overfitting]] and [[Regularization]] (ridge and lasso regression are linear regression plus an L2 or L1 penalty).
- **Link to the neighbours:** [[Logistic Regression]] is this same idea with a squash on the end to give a probability. A [[Neural Network]] is many of these stacked, with bends in between ([[Activation Function]]).
- **Assumptions behind the maths:** errors are independent, have equal spread, and are roughly bell-shaped (see [[Normal Distribution]]). The tidy residual picture above is the check for this.

```python
import numpy as np
x = np.array([1, 2, 3, 4, 5.]); y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
xm, ym = x.mean(), y.mean()
w = ((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum(); b = ym - w * xm
print(round(w, 3), round(b, 3))                                   # 1.99 0.05   closed formula
print(np.polyfit(x, y, 1).round(3))                                # [1.99 0.05] same, via numpy
X = np.c_[x, np.ones_like(x)]; print((np.linalg.inv(X.T @ X) @ X.T @ y).round(3))   # [1.99 0.05] normal equation
pred = w * x + b; mse = ((pred - y) ** 2).mean(); print(round(mse, 4))               # 0.0214
print(round(1 - ((y - pred) ** 2).sum() / ((y - ym) ** 2).sum(), 4))                 # 0.9973  (R squared)
ww, bb = 0., 0.                                                    # gradient descent from zero
for _ in range(2000):
    e = ww * x + bb - y; ww -= 0.02 * (2 * e * x).mean(); bb -= 0.02 * (2 * e).mean()
print(round(ww, 3), round(bb, 3))                                  # 1.99 0.05
y2 = y.copy(); y2[4] = 20; ym2 = y2.mean()                         # one outlier: last score 10.1 -> 20
w2 = ((x - xm) * (y2 - ym2)).sum() / ((x - xm) ** 2).sum(); print(round(w2, 3), round(ym2 - w2 * xm, 3))   # 3.97 -3.91
```

All three methods land on the same line, slope 1.99 and intercept 0.05, which is what we expect from a nearly perfect straight pattern. Changing just the last score from 10.1 to 20 raises the slope from 1.99 to 3.97, which shows how much one outlier can pull.

**For time series (added with the [[Time Series]] notes).** Regressing a series on the time index (a straight line through time) treats each point as independent and has no idea about the yearly cycle. On the airline series, with the first 120 months for fitting and the last 24 held out, the straight line had an RMSE of 74.8, against 32.5 for Holt-Winters ([[Exponential Smoothing]]). Linear regression does become a strong forecaster when it is given the right columns: lags of the series itself (an autoregressive model, see [[ARIMA]]) and month indicators.

## 🔴 4. Deep / formal
**The model.** For one input $x$, the prediction is a straight line:

$$
\hat y = w\,x + b
$$

**Where:**
- $x$ is the input (hours studied). $\hat y$ ("y-hat") is the predicted answer (the score).
- $w$ is the **slope**: how much $\hat y$ rises when $x$ goes up by 1.
- $b$ is the **intercept**: the prediction when $x=0$.
- **Matters:** these two numbers are everything the model learns. With $w$ alone the line would be forced through the origin. Without $b$, the line cannot shift up or down to meet the data.

**Example (tiny numbers):** $w=2$, $b=1$, and someone who studied $x=4$ hours.
- $\hat y = 2\times4 + 1$.
- $\hat y=\mathbf{9}$.
- Meaning: the line predicts a score of 9. Each extra hour adds 2 points, and a person who studied 0 hours starts at 1.

**The loss.** How wrong is a given line over $n$ examples? Average the squared errors:

$$
\mathrm{MSE}(w,b)=\frac1n\sum_{i=1}^{n}\big(w\,x_i+b-y_i\big)^2
$$

**Where:**
- $n$ is the number of examples, and $(x_i,y_i)$ is the $i$-th input and its true answer.
- $w\,x_i+b-y_i$ is the error of example $i$: the prediction minus the truth.
- Squaring turns every error positive. $\frac1n\sum$ takes the average.
- **Matters:** squaring makes one big miss count much more than several small ones, and it makes the curve smooth with a single lowest point, so there is one best line. Without squaring, ups and downs would cancel and a terrible line could score 0.

**Example (tiny numbers):** three points $(1,2)$, $(2,3)$, $(3,7)$ and a guess of $w=1$, $b=1$.
- Predictions: $2,\,3,\,4$. Errors: $0,\,0,\,-3$.
- Squares: $0,\,0,\,9$. Average: $9/3=\mathbf{3.0}$.
- Meaning: this line fits two points perfectly but misses the third by 3, so the loss is 3. The best line for these points (next formula) has a loss of only $0.5$.

**The least-squares answer.** Setting the slope of the loss to zero (calculus; see [[Calculus]]) gives the best $w$ and $b$ directly:

$$
w=\frac{\sum_i (x_i-\bar x)(y_i-\bar y)}{\sum_i (x_i-\bar x)^2},\qquad b=\bar y-w\,\bar x
$$

**Where:**
- $\bar x$ and $\bar y$ are the averages of the inputs and of the answers.
- The top sum multiplies "how far $x$ is above its average" by "how far $y$ is above its average", for each point. It is large and positive when $x$ and $y$ rise together.
- The bottom sum is how spread out the $x$ values are. Dividing by it makes the slope "rise per unit of $x$".
- $b=\bar y-w\bar x$ forces the line to pass through the point of averages $(\bar x,\bar y)$.
- **Matters:** if all $x$ values are equal, the bottom is $0$ and no slope can be found (there is nothing to measure a slope against). It also shows the line always passes through the middle of the data.

**Example (tiny numbers):** the same three points $(1,2)$, $(2,3)$, $(3,7)$.
- Averages: $\bar x=2$ and $\bar y=4$.
- Top: $(-1)(-2)+(0)(-1)+(1)(3)=2+0+3=5$. Bottom: $1+0+1=2$.
- $w=5/2=\mathbf{2.5}$ and $b=4-2.5\times2=\mathbf{-1.0}$.
- Check: predictions $1.5,\,4,\,6.5$, errors $-0.5,\,1,\,-0.5$, loss $(0.25+1+0.25)/3=0.5$. That beats the guess above (3.0).
- Meaning: the best line rises 2.5 per step, and no other line has a smaller average squared error on these points.

**Further notes:**
- **Why squares?** If the errors are assumed to be bell-shaped noise, minimising squared error is the same as maximum likelihood ([[Maximum Likelihood]]; see [[Normal Distribution]]). This is the standard textbook argument, from memory.
- **Many inputs:** the same idea in matrix form gives the normal equation, which needs $X^{\top}X$ to be invertible.
- **Ridge:** adding an L2 penalty on $\mathbf w$ gives ridge regression, which also keeps the matrix invertible. See [[Regularization]].
- **Not causation:** a fitted slope shows a pattern in the data, not that changing $x$ changes $y$.

**Connecting the dots: the same score, two uses.** Linear regression outputs the score $z=wx+b$ as it is. [[Logistic Regression]] takes the very same $z$ and squashes it. With one feature, $z=wx+b$, the same "$mx+b$" as a school straight line (here $m$ is called $w$). Using the pass-the-exam numbers from [[Logistic Regression]] with $w=1.5$ and $b=-4$: for 2, 3 and 4 hours of study, $z=-1,\ 0.5,\ 2$. Linear regression would report those three numbers as they are. Logistic regression squashes them into $0.269,\ 0.622,\ 0.881$, which can be read as chances.

| Model | What it outputs | Range of the output | Used for |
|---|---|---|---|
| [[Linear Regression]] | the score $z$ itself | any number, below 0 or above 1 | [[Regression]]: predicting a number |
| [[Logistic Regression]] | the squashed score $\sigma(z)$ | always between 0 and 1 | [[Classification]]: a chance of yes |

**Can linear regression do the job of logistic regression?** Fit it to labels that are 0 and 1, then say "yes" when the output is above 0.5. The code below compares this with logistic regression on two clumps of points. Without extreme points the two cut-offs are nearly the same. With 10 far-away points of class 1, the straight line is dragged to a worse cut-off, and many outputs fall outside 0 to 1 and cannot be chances. That is the reason for the squash. The reverse link, that a straight line fitted to the log-odds gives back logistic regression's weights, is shown in [[Logistic Regression]].

**Check the connection:**

```python
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
rng = np.random.default_rng(0)
def data(far):
    x = np.r_[rng.normal(-1, 1, 100), rng.normal(1, 1, 100), np.full(far, 15.0)]
    return x[:, None], np.r_[np.zeros(100), np.ones(100), np.ones(far)]
for far in (0, 10):
    X, y = data(far); lin = LinearRegression().fit(X, y); lg = LogisticRegression().fit(X, y)
    pl = lin.predict(X); cut_lin = (0.5 - lin.intercept_) / lin.coef_[0]; cut_lg = -lg.intercept_[0] / lg.coef_[0][0]
    print(f"{'no' if far == 0 else far} far-away class-1 points: linear regression cut at 0.5 -> x = {cut_lin:+.2f}; logistic cut at 0.5 -> x = {cut_lg:+.2f}; "
          f"linear outputs outside 0..1: {int(((pl < 0) | (pl > 1)).sum())}; agree on {np.mean((pl > .5) == lg.predict(X)):.3f} of points")
```

**What came before, and how this improved on it:**
- **Before:** draw a line by eye, or try many slopes and intercepts and keep the best.
- **Its limit:** trying lines costs one error calculation per candidate, and the answer is only as fine as the grid.
- **What changed:** least squares gives the best line by solving equations directly (Legendre 1805 and Gauss 1809 are usually credited; from memory, not checked).
- **How much better:** in the code below, trying 10,000 lines gets a best squared error of 49.927 (slope 2.576, intercept 0.657). One calculation gets 49.669 (slope 2.552, intercept 0.757), which is lower, with one solve instead of 10,000 error calculations.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 50); y = 2.5 * x + 1 + rng.standard_normal(50)
sse = lambda a, b: ((a * x + b - y) ** 2).sum()
best = min((sse(a, b), a, b) for a in np.linspace(0, 5, 100) for b in np.linspace(-5, 5, 100))   # try 10,000 lines
A = np.stack([x, np.ones(50)], 1); a, b = np.linalg.solve(A.T @ A, A.T @ y)                         # one solve
print("trying 10,000 lines: best error", round(best[0], 3), " slope", round(best[1], 3), " intercept", round(best[2], 3))
print("one calculation:     best error", round(sse(a, b), 3), " slope", round(a, 3), " intercept", round(b, 3))
```

## Related
- [[Regression]] · [[Supervised Learning]] · [[Loss Function]] · [[Gradient Descent]] · [[Stochastic Gradient Descent]] · [[Logistic Regression]] · [[Regularization]] · [[Overfitting]] · [[Neural Network]] · [[Normal Distribution]] · [[Matrix]] · [[Vector]] · [[Calculus]] · [[Bias-Variance]] · [[Maximum Likelihood]] · [[Scaling Laws]] · [[Convex Optimization]] · [[Eigenvalue]] · [[Singular Value Decomposition]] · [[Classification]] · [[Feature Scaling]] · [[Feature Engineering]] · [[Vectorization]] · [[Machine Learning Specialization]] · [[Time Series]] · [[ARIMA]]

## Sources
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online), chapter 3
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, chapter 3 (from memory)
- Bishop, *Pattern Recognition and Machine Learning*, chapter 3 (from memory)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/ml-foundations Linear regression:::Drawing one straight line through the middle of the dots, as close to all of them as it can, then reading predictions off the line.

#flashcards/ml/formula #flashcards/topic/ml-foundations Write the linear regression model and explain each symbol.
?
$$\hat y = w\,x + b$$
- $x$: the input (for example hours studied). $\hat y$: the predicted answer (the score).
- $w$: the slope, how much $\hat y$ rises when $x$ goes up by 1.
- $b$: the intercept, the prediction when $x=0$.
- These two numbers are everything the model learns. Without $b$ the line would be forced through the origin.

#flashcards/ml/formula #flashcards/topic/ml-foundations Write the mean squared error of a line and explain each symbol.
?
$$\mathrm{MSE}(w,b)=\frac1n\sum_{i=1}^{n}\big(w\,x_i+b-y_i\big)^2$$
- $n$: the number of examples. $(x_i,y_i)$: the $i$-th input and its true answer.
- $w\,x_i+b-y_i$: the error of example $i$, prediction minus truth.
- Squaring makes every error positive, and one big miss counts much more than several small ones. $\frac1n\sum$ takes the average.

#flashcards/ml/formula #flashcards/topic/ml-foundations Write the least-squares solution for the best line and explain each symbol.
?
$$w=\frac{\sum_i (x_i-\bar x)(y_i-\bar y)}{\sum_i (x_i-\bar x)^2},\qquad b=\bar y-w\,\bar x$$
- $\bar x$, $\bar y$: the averages of the inputs and of the answers.
- Top sum: for each point, "how far $x$ is above its average" times "how far $y$ is above its average". Large and positive when they rise together.
- Bottom sum: how spread out the $x$ values are. Dividing by it makes the slope per unit of $x$.
- $b$ then makes the line pass through the point $(\bar x,\bar y)$.

#flashcards/ml/cloze #flashcards/topic/ml-foundations The best line always passes through the point of ==averages== $(\bar x,\bar y)$.

#flashcards/ml/example #flashcards/topic/ml-foundations $w=2$, $b=1$ and $x=4$ hours. What is the prediction?::ŷ = 2 × 4 + 1 = 9. Each extra hour adds 2 points, and someone who studied 0 hours starts at 1.

#flashcards/ml/example #flashcards/topic/ml-foundations Points $(1,2),(2,3),(3,7)$ and the guess $w=1$, $b=1$. What is the MSE?::Predictions 2, 3, 4. Errors 0, 0, −3. Squares 0, 0, 9. MSE = 9/3 = 3.0.

#flashcards/ml/example #flashcards/topic/ml-foundations For the same three points, what are the best $w$, $b$ and the loss?::Averages x̄ = 2, ȳ = 4. Top: (−1)(−2) + 0 + (1)(3) = 5. Bottom: 1 + 0 + 1 = 2. w = 2.5, b = 4 − 2.5 × 2 = −1.0. Loss (0.25 + 1 + 0.25)/3 = 0.5, much better than 3.0.

#flashcards/ml/image #flashcards/topic/ml-foundations ![[linear-regression-residuals.svg]]
What does this picture show?
?
Top: a straight line fitted to 36 points, with small red vertical lines marking each point's error. Bottom: those errors plotted again. They scatter evenly around zero, which is what a good straight-line fit should look like.

#flashcards/ml/example #flashcards/topic/ml-foundations On the last 24 airline months, how did a straight line through time compare with Holt-Winters?::RMSE 74.8 for the straight line against 32.5 for Holt-Winters (about 2.3 times worse). The line ignores the yearly cycle and treats points as independent. One series and one split. My run on this machine.
