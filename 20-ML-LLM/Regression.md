---
tags: [machine-learning, supervised-learning, regression, linear-regression, statistics]
status: growing
level: 4
tier: 1
---

# Regression

## 🟢 1. Simple (age 12)
Regression is a way for a computer to **predict a number**. You show it lots of examples, such as house sizes and their prices. It finds the **pattern** (bigger houses cost more) and draws a straight line through the dots. Then you give it a new house size, and it reads the price off the line.

**Example:** You track how many hours you studied and your test score for 10 tests. The dots go up from left to right. Draw the best straight line through them. Now someone tells you "I will study for 3 hours", and you find 3 on the line to guess their score. That line is a regression model.

## 🔵 2. How it works
1. Collect **pairs**: an input number (hours studied) and the number you want to predict (score). This is a kind of [[Supervised Learning]]: every example comes with its answer.
2. Plot them as dots on a graph.
3. Choose a simple rule: **prediction = slope × input + starting value**. The slope says how much the score goes up for each extra hour. The starting value says where the line begins when the input is 0.
4. Try different slopes and starting values. For each, measure how far the line misses each dot (the vertical gaps, called **residuals**).
5. Square the misses (so up and down misses do not cancel and big misses count more), then average them. This is the [[Loss Function]] called mean squared error.
6. Pick the line with the **smallest** average. You can find it with a formula, or by letting [[Gradient Descent]] creep towards it.
7. To predict, plug a new input into the line.

![[linear-regression-residuals.svg]]
*Top: 36 dots and the straight line (red) fitted to them. The thin vertical bars from each dot to the line are the **residuals**: the misses. Bottom: those residuals plotted by themselves. They scatter evenly around zero with no pattern, which suggests a straight line is a fair choice here. Image: Justinkunimune, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Linear_regression_residuals.svg), CC0.*

## 🟠 3. Technical detail
- **Linear regression:** the prediction is a weighted sum of the inputs plus a constant. With one input it is a line. With many inputs ("multiple regression") it is a flat surface in many dimensions, written as a [[Vector]] of weights times a vector of features.
- **Fitting:** minimise the mean squared error. For linear models there is an exact formula (the "normal equation", solved with [[Matrix]] algebra), and the same answer comes from [[Gradient Descent]]. Libraries such as [[NumPy]] and scikit-learn do this for you.
- **Beyond straight lines:**
  - **Polynomial regression** adds squared and cubed inputs to fit curves. More flexibility also means more risk of [[Overfitting]].
  - **Regularised regression** (ridge, lasso) adds a penalty on large weights to stay simple.
  - **Neural networks** can be used for regression by ending in a plain number output ([[Neural Network]]).
- **Other losses:** mean absolute error is less thrown off by outliers than squared error (see the picture in [[Loss Function]]).
- **Reading a result:** the **slope** says how much the prediction changes per unit of input, and **$R^2$** says how much of the ups and downs of the answers the model explains (0 = nothing, 1 = everything).
- **Checks before trusting a line:**
  - Plot the residuals. A curve or a fan shape means the straight-line assumption is wrong.
  - Watch for outliers: a few far-off points can drag the line.
  - Do not predict far outside the range of the data you trained on.
- **Correlation is not causation:** a good fit says the two move together, not that one causes the other.

```python
import numpy as np
x = np.array([0., 1., 2.]); y = np.array([1., 3., 2.])
w = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()   # slope
b = y.mean() - w * x.mean()                                                 # intercept
print(w, b)                              # 0.5 1.5
pred = w * x + b
print(pred)                              # [1.5 2.  2.5]
ss_res = ((y - pred) ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
print(1 - ss_res / ss_tot)               # 0.25   R squared
print(np.polyfit(x, y, 1))               # [0.5 1.5]   same answer from a library
print(w * 3 + b)                         # 3.0    prediction for a new input x = 3
X = np.column_stack([x, np.ones(3)])     # least squares with a column of ones for the intercept
print(np.linalg.lstsq(X, y, rcond=None)[0].round(3))   # [0.5 1.5]
```

## 🔴 4. Deep / formal
**The line and the best-fit formulas.** For one input $x$, the model predicts $\hat y=wx+b$. The $w$ and $b$ that minimise the average squared miss have a closed form:

$$
\hat y=wx+b,\qquad w=\frac{\sum_{i=1}^{n}(x_i-\bar x)(y_i-\bar y)}{\sum_{i=1}^{n}(x_i-\bar x)^2},\qquad b=\bar y-w\,\bar x
$$

**Where:**
- $x$ is the input and $\hat y$ ("y-hat") is the predicted number. $n$ is how many training examples there are, and $x_i,y_i$ is the $i$-th pair.
- $w$ is the **slope**: how much $\hat y$ rises when $x$ rises by 1. $b$ is the **intercept**: the prediction when $x=0$.
- $\bar x$ and $\bar y$ ("x-bar", "y-bar") are the averages of all the inputs and all the answers.
- $(x_i-\bar x)(y_i-\bar y)$ multiplies "how far this input is from average" by "how far this answer is from average". **Matters:** it is positive when both are above (or both below) average, so it measures whether the two move together. The sum over all points is the numerator of the slope.
- $\sum(x_i-\bar x)^2$ is how spread out the inputs are. **Matters:** dividing by it converts "moving together" into "slope per unit of $x$". If all inputs were equal, this would be 0 and no slope could be found.
- $b=\bar y-w\bar x$ **Matters:** it forces the line to pass through the point of averages $(\bar x,\bar y)$, a known property of the best-fit line.

**Example (tiny numbers):** three points $(0,1)$, $(1,3)$, $(2,2)$.
- Averages: $\bar x=(0+1+2)/3=1$ and $\bar y=(1+3+2)/3=2$.
- Top: $(0-1)(1-2)+(1-1)(3-2)+(2-1)(2-2)=(-1)(-1)+0+0=\mathbf{1}$. Bottom: $(-1)^2+0^2+1^2=\mathbf{2}$.
- Slope $w=1/2=\mathbf{0.5}$. Intercept $b=2-0.5\cdot1=\mathbf{1.5}$.
- The line is $\hat y=0.5x+1.5$. Predictions: $1.5,\ 2,\ 2.5$. The new input $x=3$ gives $\hat y=\mathbf{3.0}$.
- Meaning: each extra unit of $x$ adds about half a unit to the prediction. Residuals are $-0.5,\ 1,\ -0.5$ (the misses).

**How good is the fit? $R^2$.** Compare the line's total miss with the miss of the laziest model, one that always predicts the average:

$$
R^{2}=1-\frac{\sum_i\big(y_i-\hat y_i\big)^2}{\sum_i\big(y_i-\bar y\big)^2}
$$

**Where:**
- $y_i$ is the true answer, $\hat y_i$ is the model's prediction for it, and $\bar y$ is the average answer.
- The top sum, $\sum(y_i-\hat y_i)^2$, is the **total squared miss of the model** (the residual sum of squares).
- The bottom sum, $\sum(y_i-\bar y)^2$, is the **total squared miss of always guessing the average** (the total variation in the answers).
- Their ratio is the fraction of the variation the model leaves **unexplained**. Subtracting from 1 gives the fraction it **explains**. **Matters:** $R^2=1$ means a perfect fit, $R^2=0$ means no better than guessing the average, and it can be negative on new data if the model is worse than that.
- **Caveat:** on training data, adding more inputs never lowers $R^2$, even useless ones, so judge on held-out data ([[Overfitting]]).

**Example (tiny numbers):** same points, line $\hat y=0.5x+1.5$.
- Model's misses: $-0.5,\,1,\,-0.5$. Squares: $0.25+1+0.25=\mathbf{1.5}$.
- Guessing the average ($\bar y=2$): misses $-1,\,1,\,0$. Squares: $1+1+0=\mathbf{2}$.
- $R^2=1-1.5/2=\mathbf{0.25}$.
- Meaning: the line explains only a quarter of the ups and downs in the answers. With only 3 noisy points that is not surprising, and a poor fit is a real result, not an error.

**Further notes:**
- **Why squares?** If the noise on the answers is bell-shaped ([[Normal Distribution]]), minimising squared error is the same as finding the most likely line (maximum likelihood). That is a standard result, stated from memory.
- **The least-squares line is unique** when the inputs are not all identical; for many inputs the problem is still a smooth bowl with one minimum, so [[Gradient Descent]] cannot get stuck in a wrong valley (unlike in a deep [[Neural Network]]).
- **Multicollinearity:** if two inputs carry almost the same information, individual weights become unstable even though predictions stay fine.
- **Regression to the mean:** extreme values tend to be followed by more average ones. The name "regression" comes from this observation in early studies of heredity (from memory).
- **Uncertainty:** a good regression also reports how unsure it is (confidence and prediction intervals). Treat a single predicted number with care.

**Connecting the dots: from predicting a number to choosing a class.** [[Regression]] predicts a free number. [[Classification]] predicts a group. The two are closer than they look, because both can start from the same score $z$ (each feature times a weight, added up, plus a starting value). With one feature, $z=wx+b$, the same "$mx+b$" as a school straight line (here $m$ is called $w$). Using the pass-the-exam numbers from [[Logistic Regression]] with $w=1.5$ and $b=-4$: for 2, 3 and 4 hours of study, $z=-1,\ 0.5,\ 2$. Linear regression would report those three numbers as they are. Logistic regression squashes them into $0.269,\ 0.622,\ 0.881$, which can be read as chances.

| Model | What it outputs | Range of the output | Used for |
|---|---|---|---|
| [[Linear Regression]] | the score $z$ itself | any number, below 0 or above 1 | [[Regression]]: predicting a number |
| [[Logistic Regression]] | the squashed score $\sigma(z)$ | always between 0 and 1 | [[Classification]]: a chance of yes |

So logistic regression is "linear regression plus a squash" (the equation, and a check that a straight line fitted to its log-odds gives back the same weights, are in [[Logistic Regression]]). Going the other way, [[Linear Regression]] fitted to 0/1 labels can classify roughly, but its outputs leave the range 0 to 1 and a few extreme points pull it off (the numbers are in [[Linear Regression]]).

**What came before, and how this improved on it:**
- **Before:** predict the average for everybody.
- **Its limit:** it ignores everything that makes one case differ from another.
- **What changed:** learn how the number changes with the features.
- **How much better:** in the code below, predicting a diabetes-progression score with the training average has a root-mean-square error of 71.4 on new patients. Linear regression has 55.7, 22% lower. The remaining error shows the features explain only part of the outcome.

**Check it yourself:**

```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X, y = load_diabetes(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
rmse = lambda p: float(np.sqrt(((p - yte) ** 2).mean()))
print("always predict the average:", round(rmse(np.full(len(yte), ytr.mean())), 1))
print("linear regression:", round(rmse(LinearRegression().fit(Xtr, ytr).predict(Xte)), 1))
```

## Related
- [[Supervised Learning]] · [[Classification]] · [[Loss Function]] · [[Gradient Descent]] · [[Overfitting]] · [[Normal Distribution]] · [[Matrix]] · [[Vector]] · [[NumPy]] · [[Neural Network]] · [[Probability]] · [[Evaluation Metrics]] · [[Logistic Regression]] · [[Linear Regression]] · [[Random Forest]]

## Sources
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (free online), chapter 3
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online)
- Strang, *Introduction to Linear Algebra* (least squares)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/ml-foundations Regression:::Predicting a number. Show lots of examples, let the computer find the pattern, draw the line, and read off the answer for a new input.

#flashcards/ml/formula #flashcards/topic/ml-foundations Write the least-squares line for one input and explain each symbol.
?
$$\hat y=wx+b,\qquad w=\frac{\sum_{i=1}^{n}(x_i-\bar x)(y_i-\bar y)}{\sum_{i=1}^{n}(x_i-\bar x)^2},\qquad b=\bar y-w\,\bar x$$
- $x$: the input. $\hat y$: the predicted number. $n$: the number of examples. $x_i,y_i$: the $i$-th pair.
- $w$: the slope, how much $\hat y$ rises when $x$ rises by 1. $b$: the intercept, the prediction at $x=0$.
- $\bar x$, $\bar y$: the averages of all inputs and all answers.
- $(x_i-\bar x)(y_i-\bar y)$: positive when both are above (or both below) average, so it measures whether the two move together.
- The bottom sum measures how spread out the inputs are.

#flashcards/ml/example #flashcards/topic/ml-foundations Points $(0,1)$, $(1,3)$, $(2,2)$. What is the line, and what does $x=3$ give?::$\bar x=1$, $\bar y=2$. Top $=1$, bottom $=2$, so $w=0.5$, $b=2-0.5=1.5$. Line $\hat y=0.5x+1.5$. Predictions 1.5, 2, 2.5. At $x=3$, $\hat y=3.0$. Residuals $-0.5$, 1, $-0.5$.

#flashcards/ml/formula #flashcards/topic/ml-foundations Write $R^2$ and explain each symbol.
?
$$R^{2}=1-\frac{\sum_i\big(y_i-\hat y_i\big)^2}{\sum_i\big(y_i-\bar y\big)^2}$$
- $y_i$: the true answer. $\hat y_i$: the model's prediction. $\bar y$: the average answer.
- Top sum: the total squared miss of the model (residual sum of squares).
- Bottom sum: the total squared miss of always guessing the average (total variation).
- The ratio is the fraction left unexplained, so one minus it is the fraction explained. $R^2=1$ is a perfect fit, 0 is no better than guessing the average, and it can be negative.

#flashcards/ml/example #flashcards/topic/ml-foundations Same points, line $\hat y=0.5x+1.5$. What is $R^2$?::Model misses $-0.5,1,-0.5$, squares sum 1.5. Guessing the average gives misses $-1,1,0$, squares sum 2. $R^2=1-1.5/2=0.25$. The line explains only a quarter of the ups and downs (with 3 noisy points).

#flashcards/ml/image #flashcards/topic/ml-foundations ![[linear-regression-residuals.svg]]
What does this picture show?
?
Top: 36 dots and the straight line (red) fitted to them. The thin vertical bars from each dot to the line are the residuals, the misses. Bottom: those residuals plotted by themselves. They scatter evenly around zero with no pattern, which suggests a straight line is a fair choice here. (From the note's caption: Justinkunimune, Wikimedia Commons.)

#flashcards/ml/cloze #flashcards/topic/ml-foundations Residuals that scatter evenly around zero with no pattern suggest a straight line is a ==fair choice==.
