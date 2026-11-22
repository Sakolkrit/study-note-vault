---
tags: [machine-learning, ensemble, boosting, decision-tree, xgboost, tabular]
status: growing
level: 4
tier: 3
---

# Gradient Boosting

Gradient boosting builds a strong model by adding many small, simple models **one at a time, each one fixing the mistakes the team has made so far**.

## 🟢 1. Simple (age 12)
Imagine you are studying for a spelling test. You take a practice test and get some words wrong. Instead of studying everything again, you study **only the words you got wrong**. Then you take another test, find the new mistakes, and study those. After many rounds you spell almost everything right. Gradient boosting works like that: each new helper focuses on what the earlier helpers still get wrong.

**Example:** You guess how tall your classmates are. First you guess "1.5 m" for everyone. Then you look at who you got wrong (a tall boy: too low; a small girl: too high) and add a small correction for each kind of person. Then you fix what is still wrong, and so on. After lots of small corrections your guesses are close.

## 🔵 2. How it works
1. Start with a very plain guess for everyone, such as the **average** answer.
2. Work out each example's **mistake**: the true answer minus the current guess. (These are called residuals.)
3. Train a small [[Decision Tree]] (just a few questions deep) whose job is to **predict those mistakes**.
4. Add the tree's answer to the guess, but only a **small fraction** of it (this fraction is the [[Learning Rate]]). Small steps are safer.
5. Work out the new mistakes. Train another small tree on them. Add a small fraction of it.
6. Repeat a hundred times, or a thousand. The final prediction is the starting guess plus all the small corrections.
7. Why is it called "gradient"? The mistake tells you which way to move the guess to make the error smaller. That is exactly the downhill direction used in [[Gradient Descent]], except that here the "step" is a whole tree and not a number. This lets the method work for many kinds of [[Loss Function]], not just squared error.
8. Compare with a [[Random Forest]]: a forest trains many trees **at the same time and independently**, then averages them. Boosting trains them **one after another**, each depending on the last. See [[Ensemble Methods]].

![[gradient-boosting-error-curve.png]]
*A gradient boosting regressor's error on held-out data over 500 boosting rounds. It falls steeply at first (from about 13,000, read off the plot) and then flattens near 1,800 as the corrections get smaller. Image: Chamaemelum, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:GradientBOOST.png), CC BY-SA 4.0 (this is the 960-pixel-wide version of the file).*

## 🟠 3. Technical detail
- **Weak learners:** the trees are kept shallow (often 3 to 6 levels). A single shallow tree is a poor model, but a long sum of them is strong. Each tree on its own "underfits" ([[Bias-Variance]]).
- **Two main knobs:** the number of trees and the learning rate. Smaller steps need more trees but usually generalise better. Add early stopping, where you stop when error on held-out data stops improving ([[Cross-Validation]]).
- **More controls:** use only a random part of the rows or columns for each tree; cap tree size; add a penalty on the leaf values ([[Regularization]]). Without them, many rounds will [[Overfitting|overfit]].
- **Other tasks:** for yes/no [[Classification]] the model's score goes through the logistic function, and the "mistake" each tree learns is the true label minus the predicted chance ([[Logistic Regression]] uses the same quantity).
- **XGBoost:** a fast, popular implementation. It adds second-derivative information, regularisation, handling of sparse data and many speed tricks (arXiv 1603.02754, March 2016; abstract read).
- **Strengths:** top performer on **tabular** data (spreadsheets); needs no [[Feature Scaling]]; handles different kinds of columns.
- **Weak spots:** sequential, so harder to parallelise than a forest; sensitive to settings; can overfit noisy data; and it is not the best tool for images or text.

```python
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error as mse

x = np.array([[1.], [2.], [3.]]); y = np.array([1., 2., 6.])      # tiny example, learning rate 0.5
F0 = np.full(3, y.mean()); r = y - F0
stump = DecisionTreeRegressor(max_depth=1).fit(x, r)
F1 = F0 + 0.5 * stump.predict(x)
print(F0, r, stump.predict(x), F1, mse(y, F0).round(4), mse(y, F1).round(4))

g, h, lam = np.array([2., 1.]), np.array([1., 1.]), 1.0           # XGBoost leaf: two points, squared error, lambda = 1
print(-g.sum() / (h.sum() + lam), -g.sum() / h.sum())

p = 1 / (1 + np.exp(-0.0)); print(1 - p)                          # log-loss: label 1, score 0 -> residual y - p

class Boost:                                                      # gradient boosting for squared error, from scratch
    def __init__(self, n=100, lr=0.1, depth=3): self.n, self.lr, self.depth = n, lr, depth
    def fit(self, X, y):
        self.f0 = y.mean(); F = np.full(len(y), self.f0); self.trees = []
        for _ in range(self.n):
            t = DecisionTreeRegressor(max_depth=self.depth, random_state=0).fit(X, y - F)      # fit the current mistakes
            F += self.lr * t.predict(X); self.trees.append(t)
        return self
    def predict(self, X): return self.f0 + self.lr * sum(t.predict(X) for t in self.trees)

X, Y = make_friedman1(n_samples=2000, noise=1.0, random_state=0)
Xtr, Ytr, Xte, Yte = X[:1000], Y[:1000], X[1000:], Y[1000:]
tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(Xtr, Ytr)
rf = RandomForestRegressor(100, random_state=0).fit(Xtr, Ytr)
mine = Boost().fit(Xtr, Ytr)
sk = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0).fit(Xtr, Ytr)
for name, m in [("one tree depth 3", tree), ("random forest", rf), ("my boosting", mine), ("sklearn boosting", sk)]:
    print(name, round(mse(Yte, m.predict(Xte)), 3))
```

Output: the first line is `[3. 3. 3.] [-2. -1.  3.] [-1.5 -1.5  3. ] [2.25 2.25 4.5 ] 4.6667 1.2917`: the start guess, the mistakes, the stump's answer, the new guess, and the error going from 4.6667 to 1.2917. Then `-1.0 -1.5` (the XGBoost leaf value with and without the penalty), `0.5` (the log-loss mistake), and the test errors on a standard synthetic dataset with 1,000 training rows: one depth-3 tree 10.596, random forest 3.852, my from-scratch boosting 2.472, scikit-learn's boosting 2.475. The from-scratch version matches the library to within 0.003.

**For time series (added with the [[Time Series]] notes).** Boosting forecasts a series the same way a forest does: turn the past into columns (lags, rolling averages, calendar columns) and fit a model. Two cautions come from the trees inside it. First, like a forest, boosted trees cannot extrapolate a trend ([[Random Forest]] has a test), so the target should be something that stays in range, such as the monthly **growth rate** and not the level. Second, to forecast several steps ahead, a simple way is **recursive**: predict one step, feed the prediction back in as if it were data, and repeat. The errors then accumulate with the horizon. Test it with forward-moving folds ([[Time Series Cross-Validation]]) and compare it with the naive and seasonal naive forecasts ([[Forecast Metrics]]). The code uses the airline series (see [[Time Series]]), predicts the next monthly growth rate from the previous 12, and forecasts the last 24 months.

```python
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

ap = pd.read_csv("AirPassengers.csv")["value"].to_numpy(float)           # monthly airline passengers (see the Time Series note)
d = np.diff(np.log(ap))                                                  # monthly growth rates: the model predicts these, not the levels
L = 12
train_d = d[:119]                                                        # growth rates up to month 120 of the series
X = np.array([train_d[i:i + L] for i in range(len(train_d) - L)]); y = train_d[L:]    # 12 past rates -> the next rate
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2, random_state=0).fit(X, y)

hist = list(train_d)
for _ in range(24):                                                      # forecast 24 months, feeding each forecast back in
    hist.append(gb.predict(np.array(hist[-L:]).reshape(1, -1))[0])
forecast = ap[119] * np.exp(np.cumsum(hist[-24:]))                       # turn growth rates back into passenger counts
test = ap[120:]
print("boosting on lag features, last 24 months: RMSE", round(float(np.sqrt(((test - forecast) ** 2).mean())), 1))
```

Output: `boosting on lag features, last 24 months: RMSE 35.8`. For comparison, on the same 24 months the seasonal naive forecast had an RMSE of 77.0 and Holt-Winters 32.5 ([[Exponential Smoothing]]). So boosting on growth-rate lags was more than twice as good as seasonal naive, and about 10% worse than Holt-Winters. One series and one split, and that split turned out to be kind: over seven forecast origins in [[Forecasting Shoot-out]], boosting scored a MASE of 0.92 against 0.56 for Holt-Winters on the same series.

## 🔴 4. Deep / formal
**The update.** After $m$ rounds, the model is the previous model plus a small, shrunk copy of a new tree:

$$
F_m(x)=F_{m-1}(x)+\eta\,h_m(x)
$$

**Where:**
- $F_{m-1}(x)$ is the model's prediction for input $x$ before round $m$. $F_0$ is the starting guess (for squared error, the average).
- $h_m$ is the new small tree, trained to predict the current mistakes.
- $\eta$ ("eta") is the learning rate, a number between 0 and 1.
- $F_m(x)$ is the prediction after the round.
- **Matters:** $\eta$ is the safety brake. With $\eta=1$ each tree is fully trusted, which fits fast but also fits noise. A small $\eta$ makes each step cautious, so you need more trees, but the final model usually generalises better.

**Example (tiny numbers):** inputs $x=1,2,3$ with targets $y=1,2,6$, and $\eta=0.5$.
- Round 0: $F_0$ is the average, $(1+2+6)/3=3$ for everyone. Error: $(4+1+9)/3=4.667$.
- Mistakes: $1-3,\ 2-3,\ 6-3=-2,\,-1,\,3$.
- A one-question tree (a "stump") splits between $x=2$ and $x=3$. It answers the average mistake on each side: $-1.5$ on the left and $3$ on the right.
- Round 1: $F_1=3+0.5\times(-1.5,\,-1.5,\,3)=(\mathbf{2.25},\,\mathbf{2.25},\,\mathbf{4.5})$.
- New error: $\big((1.25)^2+(0.25)^2+(1.5)^2\big)/3=\mathbf{1.2917}$.
- Meaning: one cautious round cut the error from 4.667 to 1.292, and the third point moved most of the way toward its target.

**Why "gradient".** The mistake each tree learns is the negative slope of the loss with respect to the current prediction:

$$
r_{im}=-\left.\frac{\partial L\big(y_i,F(x_i)\big)}{\partial F(x_i)}\right|_{F=F_{m-1}}
$$

**Where:**
- $r_{im}$ is the "pseudo-mistake" of example $i$ in round $m$, the target for the new tree.
- $L(y_i,F(x_i))$ is the loss for example $i$: true value $y_i$ against prediction $F(x_i)$.
- The vertical bar says "evaluated at the current model $F_{m-1}$".
- The minus sign turns the uphill slope into the downhill direction.
- **Matters:** for squared error with the factor one half, $L=\tfrac12(y-F)^2$, this gives exactly $y-F$, the ordinary mistake. For other losses (log-loss for classification, absolute error for outliers) it gives a different but equally valid target, so one algorithm handles all of them.

**Example (tiny numbers):** squared error, the third point above: $y=6$, current prediction $F=3$.
- Loss: $\tfrac12(6-3)^2=4.5$.
- Slope with respect to $F$: $-(6-3)=-3$.
- $r=-(-3)=\mathbf{3}$, the same as the mistake used above.
- For log-loss with label $y=1$ and score $F=0$: the chance is $p=1/(1+e^{0})=0.5$, and $r=y-p=\mathbf{0.5}$ (the code printed this).
- Meaning: whatever the loss, the tree is trained to move predictions in the direction that reduces it.

**XGBoost's leaf value.** XGBoost uses the first and second derivatives of the loss. For one leaf, the best leaf value is:

$$
w^{*}=-\frac{G}{H+\lambda}
$$

**Where:**
- $G$ is the sum of the first derivatives of the loss for the examples in the leaf, and $H$ is the sum of the second derivatives (the curvature; see [[Hessian]]).
- $\lambda$ is a penalty on the leaf values (an L2 penalty; see [[Regularization]]).
- $w^{*}$ is the value the leaf adds to the prediction.
- **Matters:** the $H$ term makes the step like a Newton step (it takes bigger steps where the loss is flat). The $\lambda$ term shrinks the value toward 0, which protects against leaves built from very few or very noisy examples. (This formula is from my memory of the XGBoost paper, whose abstract is the only part I read.)

**Example (tiny numbers):** squared error, two examples in the leaf with targets $1$ and $2$, and current prediction $3$ for both, with $\lambda=1$.
- First derivatives: $F-y=3-1=2$ and $3-2=1$, so $G=3$.
- Second derivatives: $1$ each, so $H=2$.
- With the penalty: $w^{*}=-3/(2+1)=\mathbf{-1}$.
- Without it ($\lambda=0$): $-3/2=\mathbf{-1.5}$, the plain average mistake.
- Meaning: the penalty pulls the leaf's correction from $-1.5$ toward 0.

**Further notes:**
- **Connection to AdaBoost:** AdaBoost, the earlier boosting method, can be seen as boosting with an exponential loss (a standard result; from memory).
- **Bias and variance:** a forest mostly lowers variance by averaging; boosting mostly lowers bias by adding corrections one at a time ([[Bias-Variance]]).
- **Time:** training is sequential across rounds, so the work inside each round (finding good splits) is what gets parallelised.

**What came before, and how this improved on it:**
- **Before:** a single [[Decision Tree]], then bagging and the [[Random Forest]] (many trees averaged), and AdaBoost (Freund and Schapire, 1997; from memory).
- **Its limit:** one shallow tree underfits and one deep tree overfits; a forest of independent trees averages out noise but does not systematically fix the errors that every tree shares; AdaBoost was tied to one specific loss.
- **What changed:** treat boosting as gradient descent on any differentiable loss, adding a small tree per step. Friedman published this view around 2001 (from memory, not checked). XGBoost (Chen and Guestrin, arXiv 1603.02754, March 2016, checked) made it fast and scalable.
- **How much better:** in the code above (a synthetic dataset, 1,000 training rows, untuned settings), the test error is 10.596 for one depth-3 tree, 3.852 for a random forest and 2.472 for gradient boosting: 4.3 times lower than the single tree and 1.6 times lower than the forest (my arithmetic). This is one dataset and one setting, so it shows the pattern and does not prove boosting always wins.

## Related
- [[Ensemble Methods]] · [[Decision Tree]] · [[Random Forest]] · [[Gradient Descent]] · [[Loss Function]] · [[Learning Rate]] · [[Overfitting]] · [[Regularization]] · [[Cross-Validation]] · [[Bias-Variance]] · [[Logistic Regression]] · [[Classification]] · [[Feature Scaling]] · [[Hessian]] · [[Calculus]] · [[Time Series]] · [[Time Series Cross-Validation]] · [[Forecast Metrics]] · [[Exponential Smoothing]]
- [[Experiment Tracking]] · [[Model Monitoring Lab]]
## Sources
- Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System", arXiv 1603.02754, 9 March 2016 (abstract read; the leaf-value formula is from memory)
- Friedman, "Greedy function approximation: a gradient boosting machine", about 2001 (from memory, not checked)
- Freund & Schapire on AdaBoost, 1997 (from memory, not checked)
- Image: Chamaemelum, Wikimedia Commons, CC BY-SA 4.0 (see the caption in section 2)
- Numbers in this note come from the code above (my arithmetic and my run on this machine).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/classic-models Gradient boosting:::Building a strong model by adding many small trees one after another, each trained to predict the mistakes of the model so far, and added in small steps.

#flashcards/ml/concept #flashcards/topic/classic-models What are the steps of gradient boosting for squared error?
?
Start with the average. Compute the mistakes (true minus current guess). Train a small tree to predict those mistakes. Add a small fraction (the learning rate) of the tree's answer. Repeat many times. The prediction is the start plus all the small corrections.

#flashcards/ml/formula #flashcards/topic/classic-models Write the gradient boosting update and explain each symbol.
?
$$F_m(x)=F_{m-1}(x)+\eta\,h_m(x)$$
- $F_{m-1}(x)$: the model's prediction for input $x$ before round $m$. $F_0$ is the start (for squared error, the average).
- $h_m$: the new small tree, trained to predict the current mistakes.
- $\eta$: the learning rate, between 0 and 1.
- $\eta$ is the safety brake. With $\eta=1$ each tree is fully trusted and fits noise too. A small $\eta$ needs more trees but usually generalises better.

#flashcards/ml/example #flashcards/topic/classic-models Targets 1, 2, 6 at $x=1,2,3$, start at the average, a stump, $\eta=0.5$: what is the error before and after one round?::F0 = 3 for all, error (4+1+9)/3 = 4.667. Mistakes -2, -1, 3. The stump answers -1.5, -1.5, 3. F1 = 3 + 0.5 × those = 2.25, 2.25, 4.5. New error (1.5625 + 0.0625 + 2.25)/3 = 1.2917. My arithmetic.

#flashcards/ml/formula #flashcards/topic/classic-models Write what each boosting tree learns as the negative gradient and explain each symbol.
?
$$r_{im}=-\left.\frac{\partial L\big(y_i,F(x_i)\big)}{\partial F(x_i)}\right|_{F=F_{m-1}}$$
- $r_{im}$: the pseudo-mistake of example $i$ in round $m$, the target for the new tree.
- $L(y_i,F(x_i))$: the loss for example $i$ (true value against prediction).
- The bar means "evaluated at the current model $F_{m-1}$". The minus sign turns uphill into downhill.
- For squared error with a factor one half it equals $y-F$, the ordinary mistake. Other losses give other targets, so one algorithm handles them all.

#flashcards/ml/example #flashcards/topic/classic-models What are the pseudo-mistakes for squared error ($y=6$, $F=3$) and for log-loss ($y=1$, score 0)?::Squared error: loss 0.5 × 9 = 4.5, slope -3, so r = 3 (the ordinary mistake). Log-loss: p = 1/(1+e^0) = 0.5, so r = y - p = 0.5 (the code printed this). My arithmetic.

#flashcards/ml/formula #flashcards/topic/classic-models Write the XGBoost best leaf value and explain each symbol.
?
$$w^{*}=-\frac{G}{H+\lambda}$$
- $G$: the sum of first derivatives of the loss over the examples in the leaf. $H$: the sum of second derivatives (the curvature).
- $\lambda$: an L2 penalty on leaf values.
- $w^{*}$: the value the leaf adds to the prediction.
- The $H$ term makes the step Newton-like. The $\lambda$ term shrinks the value toward 0, protecting against leaves built from few or noisy examples. (Formula from memory of the XGBoost paper; only its abstract was read.)

#flashcards/ml/example #flashcards/topic/classic-models Two examples with targets 1 and 2, current prediction 3, squared error, $\lambda=1$: what is the XGBoost leaf value, and without the penalty?::Derivatives 2 and 1 give G = 3. Second derivatives 1 and 1 give H = 2. With penalty: -3/(2+1) = -1. Without: -3/2 = -1.5, the plain average mistake. The penalty pulls the correction toward 0.

#flashcards/ml/image #flashcards/topic/classic-models ![[gradient-boosting-error-curve.png]]
What does this picture show?
?
A gradient boosting regressor's error on held-out data over 500 boosting rounds. It falls steeply at first (from about 13,000, read off the plot) and then flattens near 1,800 as the corrections get smaller.

#flashcards/ml/cloze #flashcards/topic/classic-models A random forest trains trees ==in parallel== and averages them, while boosting trains them ==one after another==.

#flashcards/ml/concept #flashcards/topic/classic-models On the note's synthetic dataset, what were the test errors of one tree, a random forest and gradient boosting?::One depth-3 tree 10.596, random forest 3.852, gradient boosting 2.472 (my from-scratch version; scikit-learn's gave 2.475). Boosting was 4.3 times lower than the single tree and 1.6 times lower than the forest. One dataset, untuned settings.

#flashcards/ml/concept #flashcards/topic/classic-models Name two weak spots of gradient boosting.
?
It is sequential, so harder to parallelise than a forest, and it is sensitive to settings (learning rate, number of trees, depth). It can overfit noisy data, and it is not the best tool for images or text.

#flashcards/ml/concept #flashcards/topic/classic-models What two cautions apply when forecasting a series with gradient boosting?
?
Trees cannot extrapolate a trend, so predict something that stays in range (such as the growth rate) and not the level. For several steps ahead, the recursive method feeds each forecast back in, so errors accumulate with the horizon. Test with forward-moving folds and compare with the naive forecasts.

#flashcards/ml/example #flashcards/topic/classic-models On the last 24 airline months, what was the RMSE of boosting on 12 growth-rate lags, against seasonal naive and Holt-Winters?::Boosting 35.8, seasonal naive 77.0 and Holt-Winters 32.5. Boosting was more than twice as good as seasonal naive and about 10% worse than Holt-Winters. One series and one split, which was kind: over seven origins boosting scored MASE 0.92 against 0.56 for Holt-Winters. My run on this machine.
