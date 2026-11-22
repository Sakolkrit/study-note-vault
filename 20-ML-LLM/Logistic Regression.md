---
tags: [machine-learning, supervised-learning, classification, logistic-regression, sigmoid, probability, loss-function]
status: growing
level: 4
tier: 2
---

# Logistic Regression

Despite the name, this is a method for [[Classification]] (choosing a category), not for predicting a free number. The name is historical.

## 🟢 1. Simple (age 12)
Logistic regression answers **yes/no questions with a chance**. It takes a few facts, mixes them into one score, and then squashes the score into a number between 0 and 1 that you can read as "how likely is yes?".

**Example:** You want to guess whether a student passes a test from the hours they studied. With 1 hour, the chance of passing is low. With 6 hours, it is high. In between, the chance climbs smoothly, like a gentle S-shaped slide. That S-shaped curve is what logistic regression draws. It never goes below 0 or above 1, so it always makes sense as a chance.

## 🔵 2. How it works
1. Each item is a list of numbers, a [[Vector]] (for example, hours studied).
2. **Mix the numbers into one score:** multiply each by a weight, add them up, and add a starting offset. A big positive score means "probably yes", a big negative score means "probably no", and zero means "50/50". (Compare with [[Regression]], where a similar mix is the answer itself.)
3. **Squash the score** with the **sigmoid** (S-shaped) function. It turns any score into a chance between 0 and 1.
4. To decide yes or no, pick a **cut-off**, usually 0.5.
5. **Learn the weights:** start with guesses, measure how badly the chances match the true answers with a [[Loss Function]] (the log loss), and nudge the weights downhill with [[Gradient Descent]]. Repeat.

![[logistic-exam-pass.svg]]
*Chance of passing an exam against hours of study. Each dot is a student: 1 means passed, 0 means failed. The blue S-curve is the fitted chance. At about 2.7 hours it crosses 50%. Image: Canley, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Exam_pass_logistic_curve.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Type:** [[Supervised Learning]]. Needs labelled examples. Outputs a **chance**, so you can pick your own cut-off to trade false alarms against misses (see [[Evaluation Metrics]]).
- **Straight-line boundary:** the 50% line is where the score is exactly 0, which is a straight line (or flat sheet) in feature space. So logistic regression can only separate groups with a straight cut, unless you add new features (for example, squares of the old ones).
- **The score is the "log-odds":** the odds are chance of yes divided by chance of no. The score equals the logarithm of the odds. Each weight says how much one unit of that feature multiplies the odds (see Further notes).
- **More than two classes:** use [[Softmax]] instead of the sigmoid (a two-class softmax is exactly a sigmoid). This is the same output layer used at the end of many [[Neural Network]]s and [[LLM]]s.
- **A neural network of one neuron:** logistic regression is a single layer with a sigmoid [[Activation Function]]. Stacking many such layers gives a [[Neural Network]].
- **Training:** the loss is smooth with a single bottom (it is convex), so gradient descent reliably finds the best weights, with no local traps. Large datasets use [[Stochastic Gradient Descent]].
- **Regularization:** if the groups can be perfectly separated, the best weights are infinite (see Further notes), so a penalty is almost always added (see [[Regularization]]). Libraries usually switch it on by default.
- **Scale matters for training and for reading weights:** rescale features (see [[Normalization]]) to train faster and to compare weights fairly.
- **Strengths:** fast, hard to overfit with few features ([[Overfitting]], [[Bias-Variance]]), and the weights are easy to explain.
- **Limits:** only a straight boundary, and probabilities are trustworthy only if the model is roughly right. For curved boundaries, see [[Support Vector Machine]] with a kernel, a [[Decision Tree]] or [[Ensemble Methods]], or a [[Neural Network]].
- **Compared with naive Bayes:** [[Bayesian Methods]] naive Bayes models how each label produces the features and then flips it with Bayes' rule. Logistic regression models the chance of the label directly from the features. Both give chances.

```python
import numpy as np
sig = lambda z: 1 / (1 + np.exp(-z))                 # the sigmoid

# --- 1. Hours studied (20 students) and pass (1) / fail (0) ---
h = np.array([.5, .75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1.])
w = b = 0.0
for _ in range(20000):                               # gradient descent on the log loss
    p = sig(w * h + b)
    w -= 0.1 * ((p - y) * h).mean()                  # gradient for w: average of (p - y) * x
    b -= 0.1 * (p - y).mean()                        # gradient for b: average of (p - y)
print(round(w, 3), round(b, 3), round(-b / w, 3))    # 1.505 -4.078 2.71   50% point at 2.71 hours
print(round(sig(w * 2 + b), 3), round(sig(w * 4 + b), 3), ((sig(w * h + b) > .5) == y).mean())
# 0.256 0.874 0.8   2 hours -> 26%, 4 hours -> 87%, 80% of students labelled correctly

# --- 2. Two-class softmax is the sigmoid ---
e = np.exp([1.3, 0]); print(round(e[0] / e.sum(), 4), round(sig(1.3), 4))   # 0.7858 0.7858

# --- 3. Perfectly separable data: the weights never stop growing ---
hs, ys = np.array([1, 2, 3, 4.]), np.array([0, 0, 1, 1.])
w = b = 0.0
for i in range(1, 100001):
    p = sig(w * hs + b)
    w -= 0.5 * ((p - ys) * hs).mean(); b -= 0.5 * (p - ys).mean()
    if i in (100, 1000, 100000): print(i, round(w, 2), round(b, 2))
# 100 1.72 -3.96 | 1000 4.58 -11.26 | 100000 13.54 -33.68   still growing, never converges
```

Run with the standard `scikit-learn` `LogisticRegression` using almost no regularization, the first part gives the same 1.505 and −4.078 (checked while writing).

## 🔴 4. Deep / formal
**The model: a score squashed into a chance.** For an item with features $\mathbf x$, weights $\mathbf w$ and offset $b$:

$$
z=\mathbf w\cdot\mathbf x+b,\qquad p=\sigma(z)=\frac{1}{1+e^{-z}}
$$

**Where:**
- $\mathbf w\cdot\mathbf x$ is the dot product (multiply each feature by its weight and add, see [[Vector]]). $b$ is the offset (bias). $z$ is the **score**.
- $\sigma$ ("sigma") is the **sigmoid**. $e$ is the number 2.718… ($e^{-z}$ is "e to the power minus $z$"). $p$ is the predicted chance that the answer is yes (label 1).
- **Matters:** the sigmoid turns any score, however large or small, into a number strictly between 0 and 1. Without it, the score itself could be $-7$ or $12$, which cannot be read as a chance. At $z=0$ the chance is exactly 0.5, and the curve is steepest there.

**Example (tiny numbers):** one feature (hours), $w=1.5$, $b=-4$.
- 2 hours: $z=1.5\cdot2-4=-1$, so $p=1/(1+e^{1})=\mathbf{0.269}$.
- 3 hours: $z=0.5$, so $p=\mathbf{0.622}$.
- 4 hours: $z=2$, so $p=\mathbf{0.881}$.
- Meaning: more study raises the chance of passing along the S-curve. The 50/50 point is where $z=0$, at $4/1.5=2.67$ hours.

**The loss: log loss (binary cross-entropy).** For $n$ items with true labels $y_i\in\{0,1\}$ and predicted chances $p_i$:

$$
\mathcal L=-\frac1n\sum_{i=1}^{n}\Big[\,y_i\log p_i+(1-y_i)\log(1-p_i)\Big]
$$

**Where:**
- $y_i$ is the true label of item $i$ (1 for yes, 0 for no). $p_i$ is the model's chance of yes for it. $n$ is the number of items. $\log$ is the natural logarithm.
- Only one of the two terms is active for each item. If $y_i=1$ the loss is $-\log p_i$. If $y_i=0$ it is $-\log(1-p_i)$. That is "minus the log of the chance given to the **true** answer".
- **Matters:** the log punishes confident wrong answers extremely hard. Giving chance 0.01 to something that happens costs $4.6$, whereas a mild miss costs little. Without the log (using plain squared error), the loss surface for this model is bumpy and the learning signal vanishes for confident mistakes. This is the same cross-entropy used for [[Self-Supervised Learning]] and in [[Loss Function]].

**Example (tiny numbers):** three single predictions (natural logs).
- True yes, predicted $0.8$: $-\ln0.8=\mathbf{0.223}$.
- True no, predicted $0.3$: $-\ln(1-0.3)=-\ln0.7=\mathbf{0.357}$.
- True yes, predicted $0.1$ (confidently wrong): $-\ln0.1=\mathbf{2.303}$.
- Meaning: a good guess costs little and a confident mistake costs about ten times more. The average of many such numbers is $\mathcal L$.

**The gradient: how to nudge the weights.** Differentiating the loss gives a surprisingly clean result:

$$
\frac{\partial\mathcal L}{\partial \mathbf w}=\frac1n\sum_{i=1}^{n}(p_i-y_i)\,\mathbf x_i,\qquad \frac{\partial\mathcal L}{\partial b}=\frac1n\sum_{i=1}^{n}(p_i-y_i)
$$

**Where:**
- $p_i-y_i$ is the **error** of item $i$: predicted chance minus the truth (so $+0.4$ if you said 0.4 for something that did not happen, or $-0.6$ if you said 0.4 for something that did).
- $\mathbf x_i$ is the item's features, so each weight's gradient is the average of the error times that feature. $\partial$ means "how the loss changes when this number changes a little" (see [[Calculus]]).
- **Matters:** the gradient has the same simple shape as in linear [[Regression]] (error times input). Gradient descent subtracts a small multiple of it (the [[Learning Rate]] times the gradient), pushing the chances towards the truth. This clean form comes from pairing the sigmoid with the log loss (a standard derivation, not shown here).

**Example (tiny numbers):** three items with $x=1,2,4$ and labels $0,1,1$. Start with $w=0$, $b=0$ (so every $p=0.5$). Learning rate $0.5$.
- Errors $p-y$: $0.5,\ -0.5,\ -0.5$.
- Gradient for $w$: $\frac13(0.5\cdot1-0.5\cdot2-0.5\cdot4)=\frac13(-2.5)=\mathbf{-0.833}$. Gradient for $b$: $\frac13(0.5-0.5-0.5)=\mathbf{-0.167}$.
- Update: $w=0-0.5\cdot(-0.833)=\mathbf{0.417}$, $b=0-0.5\cdot(-0.167)=\mathbf{0.083}$.
- Loss before: $\ln2=\mathbf{0.693}$. Loss after: $\mathbf{0.490}$.
- Meaning: the gradient says "raise the weight" because the high-$x$ items are the passes. One step already lowers the loss, from 0.693 to 0.490.

**Further notes:**
- **Odds and log-odds:** if $p=0.8$, the odds are $0.8/0.2=4$ (four to one) and the log-odds are $\ln4=1.386$. The score $z$ is the log-odds. A weight of $1.5$ per hour means each extra hour multiplies the odds by $e^{1.5}\approx4.5$ (here $w=1.505$).
- **Separable data:** if a straight line splits the two groups perfectly, the loss keeps falling as the weights grow, so the best weights are infinite. The code shows $w$ going 1.72, 4.58, 13.54 and still rising. A [[Regularization]] penalty stops this, which is why default libraries include one.
- **Maximum likelihood view:** minimising log loss is the same as maximising the chance the model gives to the observed labels (see [[Bayesian Methods]] for MLE and MAP). Adding an L2 penalty corresponds to a Gaussian prior on the weights (a standard result, stated from memory).
- **Calibration:** the output can be trusted as a chance only if, among items given 0.8, about 80% really are yes. This is checked, not guaranteed (see [[Evaluation Metrics]]).
- **No closed form:** unlike linear regression, there is no one-line formula for the best weights, so iterative methods (gradient descent, Newton-style methods) are used (from memory for the second).
- **Multi-class:** with $K$ classes, one score per class goes through [[Softmax]], and the same log-loss idea applies.

**Connecting the dots: the same score as linear regression.** [[Linear Regression]] stops at the score $z=\mathbf w\cdot\mathbf x+b$ and reports it as a number. Logistic regression squashes the same score into a chance. With one feature, $z=wx+b$, the same "$mx+b$" as a school straight line (here $m$ is called $w$). Using the pass-the-exam numbers from [[Logistic Regression]] with $w=1.5$ and $b=-4$: for 2, 3 and 4 hours of study, $z=-1,\ 0.5,\ 2$. Linear regression would report those three numbers as they are. Logistic regression squashes them into $0.269,\ 0.622,\ 0.881$, which can be read as chances.

| Model | What it outputs | Range of the output | Used for |
|---|---|---|---|
| [[Linear Regression]] | the score $z$ itself | any number, below 0 or above 1 | [[Regression]]: predicting a number |
| [[Logistic Regression]] | the squashed score $\sigma(z)$ | always between 0 and 1 | [[Classification]]: a chance of yes |

Undo the squash and the straight line is back. The logarithm of the odds is the score:

$$
\ln\frac{p}{1-p}=z=\mathbf w\cdot\mathbf x+b
$$

**Where:**
- $p$ is the predicted chance of yes. $p/(1-p)$ is the **odds**: chance of yes divided by chance of no. $\ln$ is the natural logarithm, so the left side is called the **log-odds** (or logit).
- $z$ is the same score as in linear regression. $\mathbf w$, $\mathbf x$ and $b$ are the weights, the features and the starting value.
- **Matters:** this says logistic regression is linear regression on the log-odds. A chance is stuck between 0 and 1 and cannot follow a straight line for ever, but the log-odds can be any number, so there it can.

**Example (tiny numbers):** take the chances from above.
- $p=0.881$: odds $=0.881/0.119\approx7.40$, and $\ln7.40\approx\mathbf{2.0}$, which is the score $z=1.5\cdot4-4=2$.
- $p=0.269$: odds $=0.269/0.731\approx0.368$, and $\ln0.368\approx\mathbf{-1.0}$, which is $z$ for 2 hours.
- Meaning: the squash and the log-odds are exact opposites, so a straight line fitted to the log-odds of the model's own chances recovers its weights.

**Check the connection:**

```python
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
sig = lambda z: 1 / (1 + np.exp(-z))
w, b = 1.5, -4.0                                           # one score z = w*x + b, shared by both models
h = np.array([2.0, 3.0, 4.0]); z = w * h + b
print("hours      :", h)
print("z = w*x + b:", z, "  <- what linear regression would output (any number, here below 0 and above 1)")
print("sigmoid(z) :", np.round(sig(z), 3), "  <- what logistic regression outputs (always a chance between 0 and 1)")
print("log-odds ln(p/(1-p)) of those chances:", np.round(np.log(sig(z) / (1 - sig(z))), 3), "  <- gives z back")
hours = np.array([.5, .75, 1, 1.25, 1.5, 1.75, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.5])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
lg = LogisticRegression(C=1e6, max_iter=10000).fit(hours[:, None], y)
p = lg.predict_proba(hours[:, None])[:, 1]; logit = np.log(p / (1 - p))
fit = LinearRegression().fit(hours[:, None], logit)
print(f"fitted logistic regression: w = {lg.coef_[0][0]:.3f}, b = {lg.intercept_[0]:.3f}")
print(f"a straight line fitted to the log-odds of its own chances: slope {fit.coef_[0]:.3f}, intercept {fit.intercept_:.3f}, R^2 = {fit.score(hours[:, None], logit):.6f}")
print("sigmoid of the linear part reproduces predict_proba exactly:", np.allclose(sig(hours * lg.coef_[0][0] + lg.intercept_[0]), p))
```

**What came before, and how this improved on it:**
- **Before:** fit a straight line to the 0/1 labels (linear regression) and cut at 0.5.
- **Its limit:** the line's outputs run below 0 and above 1, so they are not probabilities, and far-away points pull the line.
- **What changed:** pass the line through the sigmoid so the output is always between 0 and 1 (the logistic function is from Verhulst in the 1800s and its use for classification developed by about the 1950s; from memory).
- **How much better:** in the code below (two clumps plus 10 distant class-1 points), 10 of 210 straight-line outputs fall outside 0..1. Logistic regression never does. **On accuracy there is no gain here:** the cut-at-0.5 line gets 0.833 and logistic regression 0.824. The benefit in this toy is valid probabilities, not more correct answers.

**Check it yourself:**

```python
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
rng = np.random.default_rng(0)
x = np.r_[rng.normal(-1, 1, 100), rng.normal(1, 1, 100), np.full(10, 15.0)]     # 10 very confident class-1 points far to the right
y = np.r_[np.zeros(100), np.ones(100), np.ones(10)]
X = x[:, None]
lin = LinearRegression().fit(X, y); lg = LogisticRegression().fit(X, y)
pl = lin.predict(X)
print("predictions outside 0..1 (linear regression):", int(((pl < 0) | (pl > 1)).sum()), "of", len(y))
print("accuracy: linear regression cut at 0.5:", round(((pl > 0.5) == y).mean(), 3), "  logistic regression:", round(lg.score(X, y), 3))
```

## Related
- [[Classification]] · [[Regression]] · [[Supervised Learning]] · [[Softmax]] · [[Loss Function]] · [[Gradient Descent]] · [[Stochastic Gradient Descent]] · [[Learning Rate]] · [[Regularization]] · [[Activation Function]] · [[Neural Network]] · [[Evaluation Metrics]] · [[Bayesian Methods]] · [[Support Vector Machine]] · [[Normalization]] · [[Overfitting]] · [[Bias-Variance]] · [[Vector]] · [[Calculus]] · [[Probability]] · [[Decision Tree]] · [[Ensemble Methods]] · [[Self-Supervised Learning]] · [[LLM]] · [[Entropy]] · [[KL Divergence]] · [[Linear Regression]] · [[Maximum Likelihood]] · [[Convex Optimization]] · [[Bayes' Theorem]] · [[Naive Bayes]] · [[Bradley-Terry Model]] · [[Perceptron]] · [[Confusion Matrix]] · [[ROC Curve and AUC]] · [[Calibration]] · [[Iris Classifier Service]] · [[Decision Boundary]] · [[Machine Learning Specialization]] · [[Heart Disease Classifier]]

## Sources
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, chapter 4 (from memory)
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online), chapter 4
- Bishop, *Pattern Recognition and Machine Learning*, 2006, chapter 4 (from memory)
- Hosmer & Lemeshow, *Applied Logistic Regression* (from memory)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/ml-foundations Logistic regression:::Answers yes/no questions with a chance. It mixes a few facts into one score, then squashes the score into a number between 0 and 1 you can read as "how likely is yes?".

#flashcards/ml/formula #flashcards/topic/ml-foundations Write the logistic regression model and explain each symbol.
?
$$z=\mathbf w\cdot\mathbf x+b,\qquad p=\sigma(z)=\frac{1}{1+e^{-z}}$$
- $\mathbf w\cdot\mathbf x$: the dot product, each feature times its weight, added up. $b$: the offset (bias). $z$: the score.
- $\sigma$: the sigmoid, which squashes any score into 0 to 1. $e\approx2.718$.
- $p$: the predicted chance that the answer is yes (label 1).
- The 50/50 point is where $z=0$.

#flashcards/ml/formula #flashcards/topic/ml-foundations Write the logistic regression loss (log loss) and explain each symbol.
?
$$\mathcal L=-\frac1n\sum_{i=1}^{n}\Big[\,y_i\log p_i+(1-y_i)\log(1-p_i)\Big]$$
- $y_i$: the true label of item $i$ (1 yes, 0 no). $p_i$: the model's chance of yes. $n$: the number of items. $\log$ is natural.
- Only one term is active per item: $-\log p_i$ if $y_i=1$, $-\log(1-p_i)$ if $y_i=0$. It is minus the log of the chance given to the truth.
- Confident mistakes cost a lot.

#flashcards/ml/formula #flashcards/topic/ml-foundations Write the gradient of the logistic loss and explain each symbol.
?
$$\frac{\partial\mathcal L}{\partial \mathbf w}=\frac1n\sum_{i=1}^{n}(p_i-y_i)\,\mathbf x_i,\qquad \frac{\partial\mathcal L}{\partial b}=\frac1n\sum_{i=1}^{n}(p_i-y_i)$$
- $p_i-y_i$: the error of item $i$, predicted chance minus the truth.
- $\mathbf x_i$: the item's features. Each weight's gradient is the average of the error times that feature.
- $\partial$: how much the loss changes when only that number moves. The same simple form appears for linear regression.

#flashcards/ml/formula #flashcards/topic/ml-foundations Write the log-odds form of logistic regression and explain each symbol.
?
$$\ln\frac{p}{1-p}=z=\mathbf w\cdot\mathbf x+b$$
- $p$: the predicted chance of yes. $p/(1-p)$: the odds, chance of yes divided by chance of no.
- $\ln$ of the odds is the log-odds (logit). $z$ is the same score as in linear regression.
- So logistic regression is a straight line in log-odds, and the sigmoid is its exact opposite.

#flashcards/ml/cloze #flashcards/topic/ml-foundations Logistic regression is a ==straight line== in the ==log-odds==, and the sigmoid turns the score back into a chance.

#flashcards/ml/example #flashcards/topic/ml-foundations One feature, $w=1.5$, $b=-4$. What are $p$ for 2, 3 and 4 hours, and where is the 50/50 point?::2 h: z = −1, p = 0.269. 3 h: z = 0.5, p = 0.622. 4 h: z = 2, p = 0.881. 50/50 at z = 0, so 4/1.5 = 2.67 hours.

#flashcards/ml/example #flashcards/topic/ml-foundations What is the loss for: true yes predicted 0.8; true no predicted 0.3; true yes predicted 0.1?::−ln 0.8 = 0.223. −ln 0.7 = 0.357. −ln 0.1 = 2.303. A confident mistake costs about ten times more.

#flashcards/ml/example #flashcards/topic/ml-foundations Items $x=1,2,4$ with labels 0, 1, 1, starting $w=0$, $b=0$, learning rate 0.5. What are the gradients?::Errors p − y: 0.5, −0.5, −0.5. ∂w = (0.5 − 1 − 2)/3 = −0.833. ∂b = (0.5 − 0.5 − 0.5)/3 = −0.167. The new w ≈ 0.417 and b ≈ 0.083 (computed).

#flashcards/ml/image #flashcards/topic/ml-foundations ![[logistic-exam-pass.svg]]
What does this picture show?
?
The chance of passing an exam against hours of study. Each dot is a student (1 passed, 0 failed). The blue S-curve is the fitted chance. At about 2.7 hours it crosses 50%.
