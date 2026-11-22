---
tags: [math, statistics, probability, bayesian, estimation, regularization, machine-learning]
status: growing
level: 4
tier: 2
---

# Maximum A Posteriori

Maximum a posteriori (MAP) is [[Maximum Likelihood]] with one extra ingredient: **a belief you held before you saw the data**. You pick the settings that fit the data *and* agree with that belief.

## 🟢 1. Simple (age 12)
You find a coin and flip it three times. It lands heads three times. Do you now believe the coin *always* lands heads? Probably not. You know that almost all coins are close to fair, so three heads in a row is more likely a lucky streak than a magic coin. MAP is the rule that mixes **what you saw** with **what you already believed**.

**Example:** A new student joins your class and gets full marks on their first spelling quiz. Is the student a perfect speller? Your guess is "very good, but not perfect", because you know that nobody is perfect. You mixed one result with a belief about how people usually are.

## 🔵 2. How it works
1. You have a model with some **settings** (parameters), like the chance of heads for a coin. See [[Probability]].
2. [[Maximum Likelihood]] asks: which settings make the data I saw the most likely? With very little data this can give silly answers (3 heads in 3 flips gives "heads chance = 100%").
3. Before looking at the data, you write down a **prior**: how plausible each setting is by itself. "Coins are usually near fair" is a prior.
4. For each setting, you score two things: **how well it explains the data** and **how believable it was to begin with**. You multiply the two (or add their logarithms).
5. You pick the setting with the best total score. That setting is the **MAP estimate**.
6. With lots of data, the data score dominates and the prior fades away. With little data, the prior keeps the answer sensible.
7. The rule behind this is [[Bayes' Theorem]]: the "posterior" (belief after the data) is the likelihood times the prior, up to a constant.
8. A famous special case: if your prior says "the weights of a model are probably small", MAP becomes **ridge regression**, which is [[Regularization]] by another name.

## 🟠 3. Technical detail
- **Posterior:** by [[Bayes' Theorem]], $p(\theta\mid D)\propto p(D\mid\theta)\,p(\theta)$. MAP is the **peak** (the mode) of this posterior. The constant it ignores, $p(D)$, is the same for every $\theta$, so it never changes which setting wins.
- **Prior choices:** a flat prior gives back plain [[Maximum Likelihood]]. A bell-shaped prior ([[Normal Distribution]]) on the weights gives an L2 penalty (ridge). A "spiky at zero" prior (Laplace) gives an L1 penalty (lasso). So the penalty you add to a [[Loss Function]] is secretly a belief about the weights (a standard result; from memory).
- **Coins:** with a Beta prior, the answer has a neat closed form: it is as if you had seen some extra "imaginary" flips before the real ones. The pseudo-counts come from the prior.
- **MAP is not the whole posterior.** It reports one point. The full Bayesian answer keeps the whole curve and can say how unsure you are (see [[Bayesian Methods]]). For the coin, the mean of the posterior (0.714) differs from its peak (0.8).
- **Weak spots:**
  - The prior is a choice. A bad prior pulls the answer the wrong way.
  - The peak depends on how you write the setting (for example $\theta$ versus $\log\theta$), which is odd for something called "the most probable value" (a textbook caveat; from memory).
  - It still reports a single number, with no error bar.

```python
import numpy as np
from sklearn.linear_model import Ridge

k, n, a, b = 3, 3, 2, 2                            # 3 heads in 3 flips, prior Beta(2, 2) = "probably near fair"
th = np.linspace(0.001, 0.999, 999)
post = k * np.log(th) + (n - k) * np.log(1 - th) + (a - 1) * np.log(th) + (b - 1) * np.log(1 - th)
print(k / n, round(th[post.argmax()], 3), (k + a - 1) / (n + a + b - 2))   # MLE 1.0, MAP 0.8 (grid and formula agree)
print(round((k + a) / (n + a + b), 3))                                   # posterior mean 0.714

x = np.array([[1.], [2.]]); y = np.array([2., 3.5])    # MAP with a bell-shaped prior = ridge regression
print((x.T @ y / (x.T @ x))[0], Ridge(alpha=1.0, fit_intercept=False).fit(x, y).coef_[0], (x.T @ y / (x.T @ x + 1))[0])

rng = np.random.default_rng(0); N, P, runs = 12, 10, 300      # few data, many features, prior matches the truth
e_ols = e_map = 0.0
for _ in range(runs):
    w = rng.normal(0, 1, P); X = rng.normal(0, 1, (N, P)); Y = X @ w + rng.normal(0, 1, N)
    Xt = rng.normal(0, 1, (500, P)); Yt = Xt @ w
    w_ols = np.linalg.lstsq(X, Y, rcond=None)[0]
    w_map = np.linalg.solve(X.T @ X + 1.0 * np.eye(P), X.T @ Y)      # lambda = sigma^2 / tau^2 = 1
    e_ols += ((Xt @ w_ols - Yt) ** 2).mean() / runs; e_map += ((Xt @ w_map - Yt) ** 2).mean() / runs
print(round(e_ols, 2), round(e_map, 2), round(e_ols / e_map, 1))
```

Output: `1.0 0.8 0.8`, then `0.714`, then `1.8 1.5 1.5`, then `13.96 2.21 6.3`. The grid search and the closed form agree on 0.8, while maximum likelihood says 1.0. The three numbers on the third line show that plain least squares gives slope 1.8 while ridge (scikit-learn, and the formula) gives 1.5. The last line is a small experiment: with only 12 data points and 10 features, ordinary least squares has a test error of 13.96 and the MAP version has 2.21.

## 🔴 4. Deep / formal
**The recipe.** By [[Bayes' Theorem]], the chance of the settings after the data is proportional to likelihood times prior. Take logs and drop the constant:

$$
\hat\theta_{\text{MAP}}=\arg\max_{\theta}\Big[\log p(D\mid\theta)+\log p(\theta)\Big]
$$

**Where:**
- $\theta$ stands for the model's settings and $D$ for the data you saw.
- $\log p(D\mid\theta)$ is the log-likelihood: how well the settings explain the data (the same quantity [[Maximum Likelihood]] uses).
- $\log p(\theta)$ is the log-prior: how believable the settings are before any data.
- $\arg\max_\theta$ means "the $\theta$ that makes the whole bracket biggest".
- **Matters:** drop the second term and you are back to maximum likelihood. Keep it, and settings that the prior finds absurd (like a coin that can never land tails) get a huge penalty, even if they fit the data perfectly.

**Example (tiny numbers):** 3 heads in 3 flips, with the prior $p(\theta)\propto\theta(1-\theta)$ (the Beta(2, 2) bump, centred on a fair coin). The log-likelihood is $3\ln\theta$ and the log-prior is $\ln\big(\theta(1-\theta)\big)$.
- $\theta=0.7$: $3\ln0.7=-1.070$ and $\ln(0.21)=-1.561$, total $-2.631$.
- $\theta=0.8$: $3\ln0.8=-0.669$ and $\ln(0.16)=-1.833$, total $\mathbf{-2.502}$.
- $\theta=0.9$: $3\ln0.9=-0.316$ and $\ln(0.09)=-2.408$, total $-2.724$.
- $\theta=1$: the likelihood part is $0$, but $\ln(1\times0)$ is minus infinity.
- Meaning: 0.8 scores best. The data pushes toward 1, the prior pulls back toward 0.5, and they settle at 0.8. (My arithmetic.)

**The coin answer.** For $k$ heads in $n$ flips and a Beta($\alpha,\beta$) prior, setting the slope to zero gives:

$$
\hat\theta_{\text{MAP}}=\frac{k+\alpha-1}{n+\alpha+\beta-2}
$$

**Where:**
- $k$ is the number of heads and $n$ the number of flips.
- $\alpha$ and $\beta$ are the prior's two numbers. $\alpha-1$ acts like extra imaginary heads and $\beta-1$ like extra imaginary tails, seen before the real flips.
- **Matters:** this is "the fraction of heads", computed on the real flips plus the imaginary ones. With $\alpha=\beta=1$ (a flat prior) the imaginary flips vanish and you get $k/n$ again. A stronger prior (bigger $\alpha,\beta$) means more imaginary flips, so the data has to shout louder to move the answer.

**Example (tiny numbers):** $k=3$, $n=3$, $\alpha=\beta=2$.
- Top: $3+2-1=4$. Bottom: $3+2+2-2=5$.
- $\hat\theta=4/5=\mathbf{0.8}$.
- Meaning: it is the same as having seen one imaginary head and one imaginary tail first: 4 heads in 5 flips. This matches the grid search in the code.

**A bell-shaped prior gives ridge regression.** If the noise in $y$ is bell-shaped with spread $\sigma$, and each weight has a bell-shaped prior centred at 0 with spread $\tau$, then MAP is the same as minimising squared error plus a penalty:

$$
\hat w=\arg\min_{w}\ \sum_{i=1}^{n}\big(y_i-w^\top x_i\big)^2+\lambda\,\|w\|^2,\qquad \lambda=\frac{\sigma^2}{\tau^2}
$$

**Where:**
- $w$ is the vector of weights, $x_i$ the features of example $i$, and $y_i$ its target. $w^\top x_i$ is the prediction.
- $\|w\|^2$ is the sum of the squared weights (the penalty).
- $\sigma$ is how noisy the data is. $\tau$ is how large you believe the weights are.
- $\lambda$ ("lambda") is the penalty strength. It grows when the data is noisy ($\sigma$ big) or when you are sure the weights are small ($\tau$ small).
- **Matters:** this is why [[Regularization]] works. Without the penalty, with few examples and many features, the weights swing wildly to chase noise ([[Overfitting]]).

**Example (tiny numbers):** one feature, two examples $(x,y)=(1,2)$ and $(2,3.5)$, and $\sigma=\tau=1$ so $\lambda=1$. For one weight the answer is $\hat w=\sum xy\,/\,(\sum x^2+\lambda)$.
- $\sum xy=1\times2+2\times3.5=9$. $\sum x^2=1+4=5$.
- Without the penalty ($\lambda=0$): $9/5=\mathbf{1.8}$.
- With it: $9/(5+1)=\mathbf{1.5}$.
- Meaning: the prior shrinks the slope toward 0. scikit-learn's `Ridge` gives the same 1.5.

**Further notes:**
- **Link to the full Bayesian view.** MAP is the peak of the posterior. Using the whole posterior (for example by [[Monte Carlo]] sampling) also tells you how unsure the answer is. See [[Bayesian Methods]].
- **Many training tricks are MAP in disguise.** Weight decay is a Gaussian prior on the weights (from memory; standard in textbooks).

**What came before, and how this improved on it:**
- **Before:** [[Maximum Likelihood]] (Fisher, 1910s to 1920s; from memory, not checked).
- **Its limit:** with little data it trusts the data completely. 3 heads in 3 flips becomes "heads chance = 100%", and many features with few examples give wild weights.
- **What changed:** add a prior and maximise the posterior instead. The idea goes back to Bayes' rule (1763; from memory).
- **How much better:** in the code above (12 examples, 10 features, 300 repeats), the test error falls from 13.96 (plain least squares) to 2.21 (MAP), 6.3 times lower. This is the best case for MAP, because the prior matches how the true weights were drawn. A wrong prior would help less, or hurt.

## Related
- [[Maximum Likelihood]] · [[Bayes' Theorem]] · [[Bayesian Methods]] · [[Regularization]] · [[Overfitting]] · [[Probability]] · [[Normal Distribution]] · [[Linear Regression]] · [[Loss Function]] · [[Monte Carlo]] · [[Naive Bayes]]

## Sources
- Bishop, *Pattern Recognition and Machine Learning*, chapters 1 and 3 (from memory)
- Murphy, *Machine Learning: A Probabilistic Perspective*, sections 3.2 and 7.5 (from memory)
- Numbers in this note come from the code above (my arithmetic and my run on this machine).

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/probability Maximum a posteriori (MAP):::The setting that scores best on "how well it explains the data" plus "how believable it was before the data" (the peak of the posterior).

#flashcards/math/formula #flashcards/topic/probability Write the MAP estimate and explain each symbol.
?
$$\hat\theta_{\text{MAP}}=\arg\max_{\theta}\Big[\log p(D\mid\theta)+\log p(\theta)\Big]$$
- $\theta$: the model's settings. $D$: the data you saw.
- $\log p(D\mid\theta)$: the log-likelihood, how well the settings explain the data.
- $\log p(\theta)$: the log-prior, how believable the settings are before any data.
- $\arg\max_\theta$: the $\theta$ that makes the whole bracket biggest.
- Drop the second term and you get maximum likelihood. Keep it, and settings the prior finds absurd get a huge penalty.

#flashcards/math/example #flashcards/topic/probability With 3 heads in 3 flips and prior proportional to $\theta(1-\theta)$, what are the log-posterior scores at 0.7, 0.8 and 0.9?::0.7 gives -1.070 + -1.561 = -2.631. 0.8 gives -0.669 + -1.833 = -2.502 (best). 0.9 gives -0.316 + -2.408 = -2.724. At 1 the log-prior is minus infinity. My arithmetic.

#flashcards/math/formula #flashcards/topic/probability Write the MAP answer for a coin with a Beta prior and explain each symbol.
?
$$\hat\theta_{\text{MAP}}=\frac{k+\alpha-1}{n+\alpha+\beta-2}$$
- $k$: heads. $n$: flips.
- $\alpha-1$: imaginary extra heads from the prior. $\beta-1$: imaginary extra tails.
- It is the fraction of heads counted on the real flips plus the imaginary ones. A flat prior ($\alpha=\beta=1$) gives back $k/n$. A stronger prior means more imaginary flips.

#flashcards/math/example #flashcards/topic/probability 3 heads in 3 flips with a Beta(2, 2) prior: what is the MAP estimate and what does it mean?::(3 + 2 - 1) / (3 + 2 + 2 - 2) = 4/5 = 0.8. It is the same as one imaginary head and one imaginary tail first: 4 heads in 5 flips. Maximum likelihood would say 1.0.

#flashcards/math/formula #flashcards/topic/probability Write the ridge regression objective as a MAP estimate and explain each symbol.
?
$$\hat w=\arg\min_{w}\ \sum_{i=1}^{n}\big(y_i-w^\top x_i\big)^2+\lambda\,\|w\|^2,\qquad \lambda=\frac{\sigma^2}{\tau^2}$$
- $w$: the weights. $x_i$: features of example $i$. $y_i$: its target. $w^\top x_i$: the prediction.
- $\|w\|^2$: the sum of squared weights (the penalty).
- $\sigma$: noise spread in the data. $\tau$: spread of the bell-shaped prior on each weight.
- $\lambda$: penalty strength. It grows when the data is noisy or when you are sure the weights are small.
- Without the penalty, few examples and many features make the weights swing wildly (overfitting).

#flashcards/math/example #flashcards/topic/probability Examples $(1,2)$ and $(2,3.5)$, one weight, $\lambda=1$: what are the least squares and ridge slopes?::Sum of xy = 9, sum of x squared = 5. Least squares: 9/5 = 1.8. Ridge: 9/(5+1) = 1.5. The prior shrinks the slope toward 0. Same as scikit-learn's Ridge.

#flashcards/math/cloze #flashcards/topic/probability MAP is maximum likelihood plus a ==prior==, and a flat prior gives back ==maximum likelihood==.

#flashcards/math/cloze #flashcards/topic/probability A Gaussian prior on the weights gives an ==L2== penalty (ridge), and a Laplace prior gives an ==L1== penalty (lasso).

#flashcards/math/concept #flashcards/topic/probability Why does MAP help when there is little data?
?
The prior keeps the answer sensible. Maximum likelihood trusts the few data points completely (3 heads in 3 flips gives 100% heads). MAP mixes the data with a belief, so one lucky streak cannot dominate. With lots of data the prior fades away.

#flashcards/math/concept #flashcards/topic/probability Name two weaknesses of MAP.
?
The prior is a choice, and a bad prior pulls the answer the wrong way. It also reports only the peak of the posterior (one number, no error bar). For the coin, the posterior mean is 0.714 but the peak is 0.8.

#flashcards/math/example #flashcards/topic/probability In the note's experiment (12 examples, 10 features, 300 repeats), what was the test error of least squares versus MAP?::13.96 versus 2.21, about 6.3 times lower. This is a best case: the prior matches how the true weights were drawn. My run on this machine.
