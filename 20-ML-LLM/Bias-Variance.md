---
tags: [machine-learning, statistics, generalisation, overfitting, evaluation]
status: growing
level: 4
tier: 2
---

# Bias-Variance

## 🟢 1. Simple (age 12)
When a computer learns from examples, it can be wrong in **two different ways**. It can be **too stubborn and simple** (always making the same kind of mistake), or **too jumpy** (changing its answer a lot depending on which examples it happened to see). The trick of machine learning is finding the balance between the two.

**Example:** Three kids throw darts at a bullseye. Ana's darts all land close together, but far from the middle: she is **consistent but off target** (a bias). Ben's darts are scattered all over the board, but their average is the middle: he is **on target on average but unpredictable** (high variance). Cleo's darts are tightly grouped around the bullseye: low bias and low variance, and that is the goal. A simple model is like Ana, an over-flexible model is like Ben.

## 🔵 2. How it works
1. A model learns from **one particular set** of training examples. A different set of examples would give a slightly different model.
2. Imagine repeating the whole experiment many times, each time with fresh examples, and writing down each model's guess for the same question.
3. **Bias** is how far the *average* guess is from the truth. A model that is too simple (a straight line for a curvy pattern) is wrong the same way every time.
4. **Variance** is how much the guesses *spread out* between the repeats. A model that is too flexible (a wiggly curve that chases every dot) changes a lot from one example set to another.
5. There is also **noise**: randomness in the real answers that no model can predict.
6. **Total error = bias² + variance + noise.** Making a model more flexible usually lowers bias but raises variance. The best model sits at the bottom of the total.
7. This is the same story as [[Overfitting]] (too much variance) and **underfitting** (too much bias).

![[bias-variance-tradeoff.svg]]
*As model complexity grows (right), the squared bias (red) falls and the variance (teal) rises. Their sum, the total error (black), is lowest at an in-between "optimum" complexity. This is a schematic sketch, not measured data. Image: Bigbossfarin, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Bias_and_variance_contributing_to_total_error.svg), CC0.*

## 🟠 3. Technical detail
- **Where bias comes from:** wrong assumptions. A straight line cannot bend, so for a curved pattern it is wrong on average, however much data you give it. See [[Regression]].
- **Where variance comes from:** too much freedom compared with the amount of data. The model fits the noise in the particular sample. See [[Overfitting]].
- **How to spot which one you have** (compare training and test errors, see [[Supervised Learning]] and [[Machine Learning]]):
  - **High bias (underfitting):** error is high on training data **and** on test data.
  - **High variance (overfitting):** error is low on training data but much higher on test data.
- **Fixes for high bias:** a more flexible model (more features, a bigger [[Neural Network]], a deeper [[Decision Tree]]), or train longer.
- **Fixes for high variance:** more training data, a simpler model, **regularisation** (penalise large weights), early stopping, or **averaging many models** (a random forest averages trees, which lowers variance).
- **Choosing the setting:** pick the model size by the error on a **validation** set, not on the training data.
- **Caution:** the decomposition is exact for squared error. For other losses (such as the one used in [[Classification]]) there is no single agreed formula, so treat it as a useful idea there, not an equation.
- **Modern note:** very large networks do not always follow the simple U-shaped curve. See the end of level 4.

The code repeats a "fresh examples" experiment 500 times for lines and curves of different flexibility and measures bias² and variance at one point.

```python
import numpy as np
rng = np.random.default_rng(0)
f = lambda x: np.sin(2 * np.pi * x); x0 = 0.25; n = 15; sigma = 0.3    # truth, test point, sample size, noise
for deg in (1, 3, 5):                                  # flexibility of the fitted curve
    preds = []
    for _ in range(500):                               # 500 fresh training sets
        x = rng.uniform(0, 1, n); y = f(x) + rng.normal(0, sigma, n)
        preds.append(np.polyval(np.polyfit(x, y, deg), x0))      # this model's guess at x0
    preds = np.array(preds)
    b2 = (preds.mean() - f(x0)) ** 2; v = preds.var()
    print(deg, round(b2, 3), round(v, 3), round(b2 + v + sigma**2, 3))
# 1 0.241 0.039 0.37     straight line: high bias, low variance
# 3 0.0 0.028 0.118      cubic: both low, the best of the three
# 5 0.002 2.434 2.526    degree 5: bias gone, variance huge
```

The exact numbers depend on the random seed (and the degree-5 variance is inflated by a few wild fits on unlucky samples), but the pattern is robust: simple means bias, flexible means variance. Total error here is the sum of the three parts.

## 🔴 4. Deep / formal
**The decomposition.** Suppose the true relation is $y=f(x)+\varepsilon$, where $\varepsilon$ is random noise with average 0 and variance $\sigma^2$. Train a model many times on different random training sets, and call its prediction at a point $x$ by $\hat f(x)$. For squared error, the expected error at $x$ splits into three parts:

$$
\mathbb{E}\big[(y-\hat f(x))^2\big]=\underbrace{\big(\mathbb{E}[\hat f(x)]-f(x)\big)^2}_{\text{bias}^2}+\underbrace{\mathbb{E}\Big[\big(\hat f(x)-\mathbb{E}[\hat f(x)]\big)^2\Big]}_{\text{variance}}+\underbrace{\sigma^2}_{\text{noise}}
$$

**Where:**
- $y$ is the real answer, $f(x)$ is the true underlying value (the best possible prediction), and $\varepsilon$ is the noise. $\sigma^2$ is the noise's variance (see [[Probability]], [[Normal Distribution]]).
- $\hat f(x)$ is the model's prediction at $x$. It is a **random** number, because the model depends on which training set was drawn.
- $\mathbb{E}[\cdot]$ is the expected value, the average over many repeats of the whole experiment (training sets and noise). So $\mathbb{E}[\hat f(x)]$ is the average prediction.
- **Bias²** compares that average prediction with the truth. **Matters:** it is the part that more data cannot remove, because it is built into the model's assumptions.
- **Variance** is the average squared distance of the predictions from *their own* average. **Matters:** it is the part that grows when the model chases the noise in one sample, and it shrinks with more data or averaging.
- $\sigma^2$ is the **noise** floor. **Matters:** it is the part no model can remove, so the total error can never go below it.
- The decomposition assumes the noise is independent of the training set and has average 0, and it is exact only for squared error.

**Example (tiny numbers):** the true value at a point is $f(x)=10$ and the noise variance is $\sigma^2=1$. Four models, each trained on a different training set, predict $8,\ 9,\ 8,\ 7$.
- Average prediction: $(8+9+8+7)/4=\mathbf{8}$.
- Bias² $=(8-10)^2=\mathbf{4}$.
- Variance: deviations from 8 are $0,1,0,-1$, squares $0,1,0,1$, average $=\mathbf{0.5}$.
- Total expected error: $4+0.5+1=\mathbf{5.5}$.
- Meaning: most of the error here is bias (the models are all too low, by about 2). Gathering more data would barely help. A more flexible model would.

**Bias and variance as two measurements.** In an experiment like the code above, you can estimate the two parts from the list of predictions $\hat f_1,\dots,\hat f_m$ at the same point, one from each of $m$ training sets:

$$
\overline{f}=\frac1m\sum_{j=1}^{m}\hat f_j,\qquad \text{bias}=\overline{f}-f(x),\qquad \text{variance}=\frac1m\sum_{j=1}^{m}\big(\hat f_j-\overline{f}\big)^2
$$

**Where:**
- $m$ is how many times the experiment was repeated, and $\hat f_j$ is the prediction from the $j$-th training set.
- $\overline{f}$ ("f-bar") is the average of those predictions: the estimate of $\mathbb{E}[\hat f(x)]$.
- $\text{bias}=\overline{f}-f(x)$ is the signed gap between the average prediction and the truth. **Matters:** the sign says whether the model usually guesses too high or too low. Squaring it for the decomposition removes the sign.
- The variance formula averages the squared distance from $\overline{f}$. **Matters:** it measures spread only, so a model that is wrong by the same amount every time has bias but zero variance.
- **Caution:** this needs the true $f(x)$, which you only know in simulations. On real data you cannot measure bias this way. You can only see its effect on the test error.

**Example (tiny numbers):** the same four predictions $8,\,9,\,8,\,7$ with $f(x)=10$.
- $\overline{f}=\mathbf{8}$.
- Bias $=8-10=\mathbf{-2}$ (too low by 2).
- Variance $=(0^2+1^2+0^2+(-1)^2)/4=\mathbf{0.5}$.
- Meaning: $(-2)^2=4$ matches the bias² above, and $0.5$ is the same variance, so the pieces add up to the total of 5.5 once the noise of 1 is included.

**Further notes:**
- **Why it is not the whole story in deep learning:** very large [[Neural Network]]s, trained until they fit the training set perfectly, can still generalise well, and test error can fall again after rising ("double descent", reported in recent research; from memory, check the papers). The classic U-shape is a good first picture, not a law.
- **Regularisation** deliberately adds a little bias to cut a lot of variance, and often lowers total error. The ridge penalty in [[Regression]] is the standard example.
- **Ensembles:** averaging $M$ independent models with the same bias cuts their variance by roughly a factor of $M$. In practice, models trained on similar data are correlated, so the cut is smaller.
- **Bayes error:** for classification, the lowest possible error is set by overlapping classes, much like the noise term here.
- **Terminology warning:** "bias" here is a statistical idea about averages. It is not the same as the **bias** weight $b$ in $wx+b$ ([[Classification]]), and it is not the same as social bias in data.

**What came before, and how this improved on it:**
- **Before:** judge a model by how well it fits the training data.
- **Its limit:** training fit says nothing about new data. A model can be wrong in two different ways.
- **What changed:** split the expected error into bias squared (the model is too simple), variance (the model changes a lot with the sample) and noise (Geman, Bienenstock and Doursat, 1992; from memory).
- **How much better:** the code below fits 500 different samples of 15 noisy points at $x=0.3$. A straight line has bias squared 0.2958 and variance 0.0309 (total 0.3267). A cubic has 0.0012 and 0.0224 (total 0.0236). A degree-9 curve has 0.0008 and 4.8986 (total 4.8994). The best model is in the middle, 13.8 times better than the line and 208 times better than degree 9, for this one point.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
f = lambda x: np.sin(2 * np.pi * x); x0 = 0.3
preds = {d: [] for d in (1, 3, 9)}
for _ in range(500):
    x = rng.uniform(0, 1, 15); y = f(x) + 0.3 * rng.standard_normal(15)
    for d in preds: preds[d].append(np.polyval(np.polyfit(x, y, d), x0))
print("true value at x=0.3:", round(f(x0), 3))
for d, p in preds.items():
    p = np.array(p); b2 = (p.mean() - f(x0)) ** 2; v = p.var()
    print(f"degree {d}: bias^2 {b2:.4f}  variance {v:.4f}  sum {b2+v:.4f}")
```

## Related
- [[Machine Learning]] · [[Overfitting]] · [[Supervised Learning]] · [[Regression]] · [[Classification]] · [[Decision Tree]] · [[Neural Network]] · [[Probability]] · [[Normal Distribution]] · [[Loss Function]] · [[Regularization]] · [[Stochastic Gradient Descent]] · [[Ensemble Methods]] · [[Linear Regression]] · [[Policy Gradient]] · [[Random Forest]] · [[Cross-Validation]] · [[Learning Curve]]

## Sources
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online), chapter 2
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (free online), chapter 7
- Geman, Bienenstock & Doursat, *Neural Networks and the Bias/Variance Dilemma*, 1992 (from memory)
- Belkin et al., *Reconciling modern machine-learning practice and the classical bias-variance trade-off*, 2019 (double descent; from memory)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation Bias-variance trade-off:::A model can be wrong in two ways. Too stubborn and simple (always the same kind of mistake, high bias) or too jumpy (changes its answer a lot depending on which examples it saw, high variance).

#flashcards/ml/formula #flashcards/topic/evaluation Write the bias-variance decomposition and explain each symbol.
?
$$\mathbb{E}\big[(y-\hat f(x))^2\big]=\underbrace{\big(\mathbb{E}[\hat f(x)]-f(x)\big)^2}_{\text{bias}^2}+\underbrace{\mathbb{E}\Big[\big(\hat f(x)-\mathbb{E}[\hat f(x)]\big)^2\Big]}_{\text{variance}}+\underbrace{\sigma^2}_{\text{noise}}$$
- $y$: the real answer. $f(x)$: the true underlying value (the best possible prediction). $\sigma^2$: the variance of the noise.
- $\hat f(x)$: the model's prediction at $x$. It is a random number, because the model depends on the random training set.
- Bias$^2$: how far the average prediction is from the truth. Variance: how much predictions scatter around their own average.
- Noise: the part no model can remove.

#flashcards/ml/formula #flashcards/topic/evaluation Write how to estimate bias and variance from repeated experiments and explain each symbol.
?
$$\overline{f}=\frac1m\sum_{j=1}^{m}\hat f_j,\qquad \text{bias}=\overline{f}-f(x),\qquad \text{variance}=\frac1m\sum_{j=1}^{m}\big(\hat f_j-\overline{f}\big)^2$$
- $m$: how many times the experiment was repeated. $\hat f_j$: the prediction from the $j$-th training set.
- $\overline{f}$: the average of those predictions, the estimate of $\mathbb{E}[\hat f(x)]$.
- bias: the signed gap between the average prediction and the truth.
- variance: the average squared distance of the predictions from their own average.

#flashcards/ml/cloze #flashcards/topic/evaluation Total error is ==bias²== plus ==variance== plus ==noise==, and it is lowest at an in-between complexity.

#flashcards/ml/example #flashcards/topic/evaluation The true value is $f(x)=10$, the noise variance is 1, and four models predict 8, 9, 8, 7. What are the bias², variance and total?::Average 8. Bias² = (8 − 10)² = 4. Variance: deviations 0, 1, 0, −1, squares sum 2, average 0.5. Noise 1. Total expected error 4 + 0.5 + 1 = 5.5.

#flashcards/ml/example #flashcards/topic/evaluation For the same four predictions, what are $\overline f$, the bias and the variance?::f̄ = 8. Bias = 8 − 10 = −2 (too low by 2). Variance = (0 + 1 + 0 + 1)/4 = 0.5. (−2)² = 4 matches the bias² above.

#flashcards/ml/image #flashcards/topic/evaluation ![[bias-variance-tradeoff.svg]]
What does this picture show?
?
As model complexity grows (right), the squared bias (red) falls and the variance (teal) rises. Their sum, the total error (black), is lowest at an in-between "optimum" complexity. A schematic sketch, not measured data.
