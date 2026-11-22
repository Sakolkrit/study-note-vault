---
tags: [math, statistics, probability, machine-learning, estimation, loss]
status: growing
level: 4
tier: 2
---

# Maximum Likelihood

Maximum likelihood is a rule for picking a model's settings: **choose the settings that make the data you actually saw as likely as possible**.

## 🟢 1. Simple (age 12)
You find a coin and flip it 10 times: 7 heads, 3 tails. Is it a fair coin? You can ask, "if the coin gave heads 50% of the time, how surprising would 7 heads be? What if it gave heads 70% of the time?" The 70% story makes your result *less surprising*. So the best guess for the coin is **the setting that makes what happened the least surprising**. That is maximum likelihood.

**Example:** A bakery sells 7 croissants in 10 minutes. You guess how fast it sells. A guess of "one per minute" would make 7 in 10 minutes quite unlikely. A guess of "about 0.7 per minute" fits what you saw best. You pick the guess that fits best.

## 🔵 2. How it works
1. You have a model with some **settings** (parameters). For a coin it is the chance of heads. For a bell curve it is the centre and the spread.
2. The model can say how likely any data is, for any choice of the settings. See [[Probability]].
3. You have real data. Try a setting and ask: **how likely was my data under this setting?** That answer is the **likelihood**.
4. Try other settings. Some make the data more likely, some less.
5. **Pick the setting with the highest likelihood.** In practice you let the computer climb to the top (see [[Gradient Descent]]), or use a formula when one exists.
6. Because multiplying many small chances gives tiny numbers, people take the **logarithm** first. That turns the product into a sum and does not change which setting is best.

![[maximum-likelihood-gradient.svg]]
*A likelihood "landscape" for a model with two settings (the two directions of the picture). The green centre is where the data is most likely. The blue path shows an optimiser climbing uphill to that peak, which is the maximum-likelihood answer. Image: Justinkunimune, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Gradient_descent_maximum_likelihood.svg), CC0.*

## 🟠 3. Technical detail
- **Likelihood vs probability:** probability asks "given the settings, how likely is this data?". Likelihood flips the question: "given this data, how plausible are these settings?" It is the same formula, read in a different direction. It is **not** the probability that the settings are right (that needs a prior; see [[Bayesian Methods]]).
- **Log-likelihood:** products of many numbers below 1 underflow to 0 on a computer. The log turns the product into a sum, which is stable and easier to differentiate.
- **Training = maximum likelihood:** most of [[Machine Learning]] is secretly this. Minimising a [[Loss Function]] usually means maximising a log-likelihood:
  - Squared error in [[Linear Regression]] is the likelihood of data with bell-shaped noise ([[Normal Distribution]]).
  - Log loss in [[Logistic Regression]] and cross-entropy for [[Classification]] are the likelihood of yes/no (or many-class) labels (see [[Entropy]]).
  - An [[LLM]] is trained to maximise the likelihood of the next word. Its [[Perplexity]] is built from that.
- **Link to KL:** maximising likelihood is the same as minimising the [[KL Divergence]] from the data's distribution to the model's (up to a constant).
- **Weak spots:**
  - With little data it can **overfit**: 3 heads in 3 flips gives "heads chance = 100%". Adding a prior fixes this (the MAP estimate; see [[Bayesian Methods]] and [[Regularization]]).
  - The formula for the spread of a bell curve is slightly **biased** (it divides by $n$, so on average it comes out too small).
  - It needs a model that is roughly right. If the model's shape is wrong, you get "the best of a bad lot".
- **Hidden variables:** when part of the story is unseen (which group made each point), the likelihood is hard to climb directly. [[Expectation-Maximization]] handles that, for example in a [[Gaussian Mixture Model]].

```python
import numpy as np
from scipy.optimize import minimize
n, k = 10, 7                                             # 10 flips, 7 heads
ll = lambda t: k * np.log(t) + (n - k) * np.log(1 - t)   # log-likelihood of the coin chance t
grid = np.linspace(0.01, 0.99, 99); print(round(grid[ll(grid).argmax()], 2), k / n)   # 0.7 0.7
print(round(ll(.5), 3), round(ll(.7), 3))                # -6.931 -6.109   (0.7 explains the data better)
x = np.array([2, 4, 6.]); print(x.mean(), round(((x - x.mean()) ** 2).mean(), 3), x.var(ddof=1))   # 4.0 2.667 4.0

# regression by maximum likelihood (bell-shaped noise): same line as least squares
xs = np.array([1, 2, 3, 4, 5.]); ys = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
def nll(p):                                              # negative log-likelihood: slope w, intercept b, log-spread
    w, b, ls = p; s = np.exp(ls); r = ys - (w * xs + b)
    return (r ** 2 / (2 * s * s) + np.log(s * np.sqrt(2 * np.pi))).sum()
res = minimize(nll, [0, 0, 0], method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 5000})
print(res.x[:2].round(3), round(float(np.exp(res.x[2])), 4), round(float(np.sqrt(0.0214)), 4))   # [1.99 0.05] 0.1463 0.1463
```

The grid search and the formula agree on 0.7. The three numbers 2, 4, 6 show the bias: the maximum-likelihood spread (divide by $n$) is 2.667, while the usual corrected one (divide by $n-1$) is 4.0. The last two lines fit a line by maximising likelihood. They land on the same slope and intercept as least squares in [[Linear Regression]] (1.99 and 0.05), and the fitted noise spread 0.1463 equals the square root of that note's mean squared error (0.0214).

## 🔴 4. Deep / formal
**The recipe.** For data $x_1,\dots,x_n$ that are independent, and a model with settings $\theta$, pick the settings that maximise the (log) chance of the data:

$$
\hat\theta=\arg\max_{\theta}\;\ell(\theta),\qquad \ell(\theta)=\sum_{i=1}^{n}\log p(x_i\mid\theta)
$$

**Where:**
- $\theta$ ("theta") stands for the model's settings. $\hat\theta$ is the best choice.
- $p(x_i\mid\theta)$ is the chance (or density) the model gives to data point $x_i$ when the settings are $\theta$.
- $\ell(\theta)$ is the **log-likelihood**: the sum of the logs of those chances. It is the log of the product of the chances, which is the chance of all the data when points are independent.
- $\arg\max_\theta$ means "the $\theta$ that makes $\ell$ biggest".
- **Matters:** the log does not move the peak, but it makes a product of thousands of tiny numbers into a manageable sum. Without independence, you could not split the chance of the data into one factor per point.

**Example (tiny numbers):** a coin, with $k=7$ heads in $n=10$ flips. A head has chance $\theta$, so $\ell(\theta)=7\ln\theta+3\ln(1-\theta)$.
- At $\theta=0.5$: $7\ln0.5+3\ln0.5=10\times(-0.693)=-6.931$.
- At $\theta=0.7$: $7\ln0.7+3\ln0.3=7(-0.357)+3(-1.204)=-2.497-3.612=-6.109$.
- Higher (less negative) is better, so $0.7$ beats $0.5$.
- Meaning: the data is more plausible under a 70% coin than a 50% coin. The best of all settings turns out to be exactly 0.7 (next formula).

**The coin answer.** Setting the slope of $\ell$ to zero (see [[Calculus]]) gives a closed form:

$$
\frac{d\ell}{d\theta}=\frac{k}{\theta}-\frac{n-k}{1-\theta}=0\quad\Longrightarrow\quad\hat\theta=\frac{k}{n}
$$

**Where:**
- $k$ is the number of heads and $n$ the number of flips. $n-k$ is the number of tails.
- $\frac{k}{\theta}$ is the push upwards from the heads. $\frac{n-k}{1-\theta}$ is the push downwards from the tails.
- The best $\theta$ is where the two pushes balance.
- **Matters:** the maximum-likelihood chance is just "what fraction of flips were heads". That sounds obvious, and it is a good sign that the method is sensible. It also exposes the weakness: with $k=n=3$ it says $\hat\theta=1$.

**Example (tiny numbers):** $k=7$, $n=10$.
- At $\theta=0.7$: $k/\theta=7/0.7=10$ and $(n-k)/(1-\theta)=3/0.3=10$.
- The two pushes are equal, so the slope is $0$: this is the top of the curve.
- $\hat\theta=7/10=\mathbf{0.7}$.
- Meaning: the peak is at 0.7, matching the grid search in the code.

**Bell-shaped noise gives squared error.** If a model predicts $\hat y$ and the real value $y$ is the prediction plus bell-shaped noise of spread $\sigma$, the negative log-likelihood of one point is:

$$
-\log\mathcal N(y\mid\hat y,\sigma^2)=\frac{(y-\hat y)^2}{2\sigma^2}+\log\!\big(\sigma\sqrt{2\pi}\big)
$$

**Where:**
- $y$ is the true value and $\hat y$ the model's prediction. $\sigma$ is the spread of the noise (assumed known here).
- $(y-\hat y)^2$ is the squared error. Dividing by $2\sigma^2$ scales it by how noisy the data is.
- $\log(\sigma\sqrt{2\pi})$ does not depend on the prediction $\hat y$, so it does not affect which model is best (when $\sigma$ is fixed).
- **Matters:** this is *why* least squares is the right tool for bell-shaped noise: maximising likelihood and minimising squared error pick the same model. If the noise were different (say, spiky), squared error would stop being the natural choice.

**Example (tiny numbers):** $y=3$, prediction $\hat y=2.5$, $\sigma=1$.
- First part: $(3-2.5)^2/(2\times1)=0.25/2=0.125$.
- Second part: $\log(1\times\sqrt{2\pi})=\log2.507=0.919$.
- Total: $0.125+0.919=\mathbf{1.044}$.
- Meaning: this one point costs 1.044 in negative log-likelihood. Only the 0.125 depends on how good the prediction is. A better prediction (say $\hat y=3$) would cut that part to 0.

**The bell-curve answer.** Maximising the likelihood of points $x_1,\dots,x_n$ for a bell curve gives the average for its centre and the average squared gap for its spread:

$$
\hat\mu=\frac1n\sum_{i=1}^{n}x_i,\qquad \hat\sigma^2=\frac1n\sum_{i=1}^{n}(x_i-\hat\mu)^2
$$

**Where:**
- $x_i$ are the data points and $n$ is how many there are.
- $\hat\mu$ is the best centre: the plain average. $\hat\sigma^2$ is the best spread (squared): the average of the squared distances from that centre.
- **Matters:** this is the source of the standard "mean and variance" recipe. Note the division by $n$ and not $n-1$: this estimate runs slightly small on average, because the centre was chosen to fit these very points. Dividing by $n-1$ corrects that.

**Example (tiny numbers):** data $2,\,4,\,6$.
- Centre: $(2+4+6)/3=\mathbf{4}$.
- Squared gaps: $4,\,0,\,4$. Their average: $8/3=\mathbf{2.667}$.
- Meaning: the likelihood-best bell curve is centred at 4 with squared spread 2.667. The corrected version, dividing by $n-1=2$, gives $8/2=4.0$.

**Further notes:**
- **Good behaviour:** with enough data, and a correct model, maximum likelihood is consistent (it homes in on the truth) and about as precise as any method can be. These are textbook results (from memory; not proved here).
- **Link to MAP:** adding a prior and maximising gives the MAP estimate. A Gaussian prior on weights is equivalent to an L2 penalty (see [[Regularization]]; from memory).
- **Not always solvable by formula:** for most models, we climb the log-likelihood with [[Stochastic Gradient Descent]].
- **Hidden variables:** the log of a sum appears when some variable is unseen. That is the problem EM solves ([[Expectation-Maximization]]).

**What came before, and how this improved on it:**
- **Before:** the method of moments (match the average and spread of the data to the model's; Pearson, 1890s; from memory).
- **Its limit:** it can be wasteful. To estimate the top of a range from samples, "twice the average" ignores that the largest sample is a strong clue.
- **What changed:** choose the parameter that makes the observed data most likely (Fisher, 1910s to 1920s; from memory, not checked).
- **How much better:** in the code below (20 samples from 0 to 1, repeated 50,000 times), twice the average has a mean squared error of 0.01652 and the maximum-likelihood answer, the largest sample, has 0.00424, 3.9 times smaller. The maximum-likelihood estimate is slightly biased (it averages 0.9528, not 1.0), so lower error does not mean unbiased.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0); theta, n, runs = 1.0, 20, 50000
x = rng.uniform(0, theta, (runs, n))
mom = 2 * x.mean(1); mle = x.max(1)                  # estimate the top of a range: twice the mean, or the largest value seen
mse = lambda e: float(((e - theta) ** 2).mean())
print("true top of the range 1.0, 20 samples, 50,000 repeats")
print("method of moments (2 x mean): average", round(mom.mean(), 4), " mean squared error", round(mse(mom), 5))
print("maximum likelihood (the max): average", round(mle.mean(), 4), " mean squared error", round(mse(mle), 5))
```

## Related
- [[Probability]] · [[Bayesian Methods]] · [[Entropy]] · [[KL Divergence]] · [[Normal Distribution]] · [[Loss Function]] · [[Linear Regression]] · [[Logistic Regression]] · [[Regularization]] · [[Gradient Descent]] · [[Calculus]] · [[Expectation-Maximization]] · [[Gaussian Mixture Model]] · [[Perplexity]] · [[LLM]] · [[Pre-training]] · [[Diffusion Models]] · [[Variational Autoencoder]] · [[Bayes' Theorem]] · [[Bradley-Terry Model]] · [[Fisher Information]] · [[Maximum A Posteriori]]

## Sources
- Bishop, *Pattern Recognition and Machine Learning*, chapters 1 and 2 (from memory)
- Murphy, *Machine Learning: A Probabilistic Perspective*, section 3.2 (from memory)
- Goodfellow, Bengio & Courville, *Deep Learning*, section 5.5 (free online)
- Fisher's early papers on the method, c. 1912 to 1922 (from memory; dates approximate)

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/probability Maximum likelihood:::Pick the setting that makes the data you actually saw the least surprising.

#flashcards/math/formula #flashcards/topic/probability Write the maximum likelihood estimate and the log-likelihood, and explain each symbol.
?
$$\hat\theta=\arg\max_{\theta}\;\ell(\theta),\qquad \ell(\theta)=\sum_{i=1}^{n}\log p(x_i\mid\theta)$$
- $\theta$: the model's settings. $\hat\theta$: the best choice.
- $p(x_i\mid\theta)$: the chance (or density) the model gives to data point $x_i$ when the settings are $\theta$.
- $\ell(\theta)$: the log-likelihood, the sum of the logs of those chances (the log of the chance of all the data when points are independent).
- $\arg\max_\theta$: the $\theta$ that makes $\ell$ biggest. The log does not move the peak, but it turns a product into a sum.

#flashcards/math/formula #flashcards/topic/probability Write the maximum likelihood answer for a coin and explain each symbol.
?
$$\frac{d\ell}{d\theta}=\frac{k}{\theta}-\frac{n-k}{1-\theta}=0\quad\Longrightarrow\quad\hat\theta=\frac{k}{n}$$
- $k$: the number of heads. $n$: the number of flips. $n-k$: the number of tails.
- $k/\theta$: the push upwards from the heads. $(n-k)/(1-\theta)$: the push downwards from the tails. The best $\theta$ is where they balance.
- So the best chance is just the fraction of heads. Weakness: with $k=n=3$ it says $\hat\theta=1$.

#flashcards/math/formula #flashcards/topic/probability Write the negative log-likelihood for bell-shaped noise and explain each symbol.
?
$$-\log\mathcal N(y\mid\hat y,\sigma^2)=\frac{(y-\hat y)^2}{2\sigma^2}+\log\!\big(\sigma\sqrt{2\pi}\big)$$
- $y$: the true value. $\hat y$: the model's prediction. $\sigma$: the spread of the noise (assumed known).
- $(y-\hat y)^2$: the squared error. Dividing by $2\sigma^2$ scales it by how noisy the data is.
- $\log(\sigma\sqrt{2\pi})$ does not depend on the prediction, so it does not change which model is best.
- So maximum likelihood and least squares pick the same model when the noise is bell-shaped.

#flashcards/math/formula #flashcards/topic/probability Write the maximum likelihood mean and variance for bell-shaped data and explain each symbol.
?
$$\hat\mu=\frac1n\sum_{i=1}^{n}x_i,\qquad \hat\sigma^2=\frac1n\sum_{i=1}^{n}(x_i-\hat\mu)^2$$
- $x_i$: the data points. $n$: how many there are.
- $\hat\mu$: the best centre, the plain average. $\hat\sigma^2$: the best squared spread, the average squared distance from that centre.
- It divides by $n$, not $n-1$. This runs slightly small on average because the centre was fitted to these same points. Dividing by $n-1$ corrects it.

#flashcards/math/cloze #flashcards/topic/probability Taking the ==log== of the likelihood does not move the peak, but turns a product into a ==sum==.

#flashcards/math/example #flashcards/topic/probability A coin gives 7 heads in 10 flips. What is $\ell$ at $\theta=0.5$ and at $\theta=0.7$, and what is $\hat\theta$?::ℓ(0.5) = 10 × (−0.693) = −6.931. ℓ(0.7) = 7(−0.357) + 3(−1.204) = −6.109. Higher is better, so 0.7 beats 0.5. θ̂ = 7/10 = 0.7 (at 0.7 both pushes equal 10).

#flashcards/math/example #flashcards/topic/probability For data 2, 4, 6, what are the maximum likelihood mean and variance, and the corrected variance?::Mean 4. Squared gaps 4, 0, 4. Variance 8/3 = 2.667. Dividing by n−1 = 2 gives 4.0.

#flashcards/math/image #flashcards/topic/probability ![[maximum-likelihood-gradient.svg]]
What does this picture show?
?
A likelihood landscape for a model with two settings. The green centre is where the data is most likely. The blue path is an optimiser climbing uphill to that peak, which is the maximum-likelihood answer.
