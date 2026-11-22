---
tags: [machine-learning, preprocessing, optimization, gradient-descent, features]
status: growing
level: 4
tier: 1
---

# Feature Scaling

## 🟢 1. Simple (age 12)
Feature scaling means **squeezing all your numbers into similar sizes before the computer learns from them**. If one number is "house size: 2,000" and another is "bedrooms: 3", the big number shouts and the small number whispers, and the learning gets clumsy. Put them on the same scale and both can be heard.

**Example:** You compare students by height in centimetres (about 150) and shoe size (about 38). If you just add them, height drowns out shoe size. Turn both into "how far above or below the class average", and they count about equally.

## 🔵 2. How it works
1. A model like [[Linear Regression]] gives each feature a weight. A feature with big numbers (size, 300 to 2,000) needs a **tiny** weight, and one with small numbers (bedrooms, 0 to 5) needs a **big** weight.
2. When you train with [[Gradient Descent]], one [[Learning Rate]] is used for all weights. A rate small enough for the big feature makes the small feature crawl, and a rate big enough for the small feature makes the big one blow up.
3. So the path to the best answer becomes a long zigzag down a narrow valley.
4. Scaling gives every feature a similar range, such as about -1 to 1. The valley becomes round and gradient descent walks almost straight to the bottom.
5. Common recipes: divide by the maximum, subtract the average and divide by the spread (called **z-score** scaling), or shrink to 0..1. Compute the average and spread on the **training** data only, then reuse those same numbers on new data.

## 🟠 3. Technical detail
- **Three common recipes:**
  - **Max scaling:** $x/x_{max}$. Easy, keeps zeros.
  - **Min-max:** $(x-x_{min})/(x_{max}-x_{min})$, gives 0..1. One huge outlier squashes everyone else.
  - **Z-score (standardisation):** $(x-\mu)/\sigma$, gives mean 0 and spread 1. A good default.
- **Fit on train, apply to test.** Compute $\mu$ and $\sigma$ from the training set, store them, and apply the same values to validation, test and live data. Using the test set to compute them leaks information (see [[Cross-Validation]]).
- **Who needs it:** methods that use distances or gradients on raw features: [[Linear Regression]], [[Logistic Regression]], [[k-NN]], [[Support Vector Machine]], [[PCA]], [[Neural Network]]. Tree methods such as [[Decision Tree]] and [[Random Forest]] mostly do not.
- **Not the same as normalisation inside a network:** layers like [[Layer Normalization]] and [[Normalization]] rescale activations during training. This note is about the input features, once, before training.
- **Predict, then undo:** if you scale the target too, remember to convert predictions back.
- **Check that it worked:** plot the cost over iterations (see [[Gradient Descent]]); it should fall smoothly and settle faster.

```python
import numpy as np
size = np.array([300, 1000, 2000.])
mu, sigma = size.mean(), size.std()
print(mu, round(sigma, 1))                    # 1100.0 697.6
print(np.round((size - mu) / sigma, 2))       # [-1.15 -0.14  1.29]
print(size / size.max())                      # [0.15 0.5  1.  ]
```

## 🔴 4. Deep / formal
**Z-score scaling of one feature $j$:**

$$
x_j' = \frac{x_j-\mu_j}{\sigma_j}
$$

**Where:**
- $x_j$ is the original value of feature $j$, and $x_j'$ is the scaled value.
- $\mu_j$ is the average of feature $j$ over the training set. $\sigma_j$ is its standard deviation (a typical distance from the average).
- **Matters:** subtracting $\mu_j$ centres the feature at 0. Dividing by $\sigma_j$ makes its spread 1. Without this, the loss surface of a linear model is a stretched bowl; the ratio of its steepest to its flattest direction (the condition number) is huge, and gradient descent must use a tiny step to stay stable ([[Learning Rate]]).
- If $\sigma_j=0$ (a constant feature), do not divide: drop the feature.

**Example (tiny numbers):** house sizes $300,\ 1000,\ 2000$ square feet.
- Average $\mu=(300+1000+2000)/3=1100$. Spread $\sigma\approx697.6$.
- $300\to(300-1100)/697.6=-1.15$. $1000\to(1000-1100)/697.6=-0.14$. $2000\to(2000-1100)/697.6=+1.29$.
- Meaning: the three sizes now sit near 0 on a scale where a bedroom count scaled the same way would be comparable.

**Why it helps, in one line.** For a squared-error loss the curvature along weight $j$ is proportional to the average of $x_j^2$. A feature in the thousands has a curvature millions of times bigger than one in single digits. The safe learning rate is set by the biggest curvature ([[Learning Rate]], deep section), while the progress rate is set by the smallest. Scaling makes them similar.

**Limits:**
- Scaling does not change the best possible fit of a plain linear model, only how quickly you reach it. (Regularised models do change: see [[Regularization]], where all weights should be penalised on a comparable scale.)
- Outliers hurt min-max and also inflate $\sigma$; robust versions use the median and quartile range.

**What came before, and how this improved on it:**
- **Before:** feed the raw numbers to gradient descent and pick a tiny learning rate by trial.
- **Its limit:** features with very different ranges force a tiny safe step, and training takes huge numbers of steps.
- **What changed:** rescale every feature to a similar range before training, using the training set's own average and spread.
- **How much better:** in the code below (10 houses with size and bedrooms, plain gradient descent, stopping when the cost is within 1% of the best possible), the raw features had to use a learning rate below about 8.7e-07 to avoid blowing up and had still **not** reached the target after 1,000,000 steps. The scaled features, with a rate of about 0.985, reached it in 53 steps. That is at least 18,868 times fewer steps. (A toy data set; the exact numbers depend on the data.)

**Check it yourself:**

```python
import numpy as np
size = np.array([952, 1244, 1947, 2100, 852, 1100, 1534, 1800, 1430, 1250.0])
beds = np.array([2, 3, 4, 4, 1, 2, 3, 3, 3, 2.0])
price = 0.2 * size + 15 * beds + 30 + np.random.default_rng(1).normal(0, 5, 10)
X = np.c_[size, beds]; Z = (X - X.mean(0)) / X.std(0)
def J_of(X, y, w, b):
    e = X @ w + b - y; return (e @ e) / (2 * len(y))
A = np.c_[X, np.ones(len(X))]; c, *_ = np.linalg.lstsq(A, price, rcond=None)
target = J_of(X, price, c[:2], c[2]) * 1.01
def steps_to(X, y, alpha, maxit=1_000_000):
    m, n = X.shape; w = np.zeros(n); b = 0.0
    for it in range(1, maxit + 1):
        e = X @ w + b - y
        w -= alpha * (X.T @ e) / m; b -= alpha * e.mean()
        if not np.isfinite(e @ e) or e @ e > 1e12: return None      # blew up
        if J_of(X, y, w, b) <= target: return it
    return maxit
def best_alpha(X, grid):
    ok = [(a, steps_to(X, price, a)) for a in grid]
    return [(a, s) for a, s in ok if s is not None][-1]
a_raw, s_raw = best_alpha(X, [1e-8 * 1.5 ** k for k in range(30)])
a_sc, s_sc = best_alpha(Z, [1e-3 * 1.5 ** k for k in range(25)])
print("raw: alpha %.3g -> %d steps (1000000 = gave up)" % (a_raw, s_raw))
print("scaled: alpha %.3g -> %d steps" % (a_sc, s_sc))
```

## Related
- [[Gradient Descent]] · [[Learning Rate]] · [[Linear Regression]] · [[Logistic Regression]] · [[Normalization]] · [[Layer Normalization]] · [[k-NN]] · [[PCA]] · [[Regularization]] · [[Cross-Validation]] · [[Feature Engineering]] · [[Vectorization]] · [[Machine Learning Specialization Course 1 Week 2]]
- [[Local Outlier Factor]] · [[Anomaly Detection Shoot-out]]
## Sources
- Course videos "Machine Learning Specialization" (Andrew Ng), Course 1 Week 2 (transcripts read; see [[Machine Learning Specialization]]). The max-scaling example (divide by the range maximum) follows the course.
- The 18,868x figure is measured in this note's code, not taken from a source

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/data-features Feature scaling:::Squeeze all numbers into similar sizes before learning, so a big number like house size does not shout over a small one like bedrooms.

#flashcards/ml/formula #flashcards/topic/data-features Write standardisation and explain each symbol.
?
$$x_j' = \frac{x_j-\mu_j}{\sigma_j}$$
- $x_j$: the original value of feature $j$. $x_j'$: the scaled value.
- $\mu_j$: the average of feature $j$ over the training set. $\sigma_j$: its standard deviation (a typical distance from the average).
- Subtracting $\mu_j$ centres the feature at 0. Dividing by $\sigma_j$ makes its spread 1. Without it the loss surface of a linear model is a stretched bowl with a huge condition number, and gradient descent needs a tiny step to stay stable.
- If $\sigma_j=0$ (a constant feature), do not divide: drop the feature.

#flashcards/ml/example #flashcards/topic/data-features House sizes 300, 1000, 2000. What are the scaled values?::$\mu=1100$, $\sigma\approx697.6$. $300\to-1.15$, $1000\to-0.14$, $2000\to+1.29$. The sizes now sit near 0 on a scale comparable with other scaled features.

#flashcards/ml/concept #flashcards/topic/data-features What does an unscaled feature do to gradient descent?::It stretches the loss surface into a long bowl, so the learning rate must be tiny to stay stable and training is slow.
