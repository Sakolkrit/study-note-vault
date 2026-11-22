---
tags: [math, probability, statistics, variance, standard-deviation, foundations]
status: growing
level: 4
tier: 1
---

# Variance

A number that says **how spread out** values are around their average. Its square root is the **standard deviation**, written $\sigma$.

## 🟢 1. Simple (age 12)
Two classes can both have an average test score of 5 out of 10. In one class everybody got exactly 5. In the other, half got 0 and half got 10. The averages match, but the classes are very different. **Variance is the number that tells those two classes apart**: small when everyone is close to the average, big when scores are scattered far from it.

**Example:** Scores 5, 5, 5, 5 are all on the average, so the spread is 0. Scores 0, 10, 0, 10 are each 5 away from the average, so the spread is big. Variance squares those distances (5 × 5 = 25) and averages them, which gives 25.

## 🔵 2. How it works
1. Find the average of the values (the [[Expected Value]]).
2. For each value, find how far it is from the average.
3. **Square** each distance. This makes every distance positive, so that values below the average do not cancel values above it. It also makes far-away values count much more.
4. Average the squares. That is the **variance**.
5. Take the square root to get back to the original units. That is the **standard deviation**. If the values are in metres, the variance is in square metres and the standard deviation is in metres.
6. For a bell curve ([[Normal Distribution]]) the standard deviation has a nice meaning: about two thirds of values fall within one standard deviation of the average.
7. **Handy rules:** adding a fixed number to every value does not change the variance. Multiplying every value by 3 multiplies the variance by 9. The variances of **independent** things add up.
8. **Where it shows up in AI:** a batch average is steadier than a single example, and variance says by how much. Weight setups ([[Normalization]]) and the [[Bias-Variance]] trade-off are about spread too.

![[normal-distribution-sd.svg]]
*The bell curve split into bands of one standard deviation ($\sigma$) each. The percentages are the share of values in each band: 34.1% just above and below the mean, 13.6% in the next band out, then 2.1% and 0.1%. The labels add to 99.8% only because of rounding. Image: M. W. Toews, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Standard_deviation_diagram.svg), CC BY 2.5. Same file and caption as in [[Normal Distribution]], where it was looked at first.*

## 🟠 3. Technical detail
- **Two formulas, same answer:** $\mathrm{Var}(X)=\mathbb{E}[(X-\mu)^2]=\mathbb{E}[X^2]-\mu^2$. The second is handy on paper, but on a computer it can lose accuracy when the mean is big and the spread is small (stated from memory, I did not test it).
- **Standard deviation** $\sigma=\sqrt{\mathrm{Var}(X)}$, in the same units as $X$.
- **Population vs sample:** if you have *all* the values, divide the sum of squares by $n$. If you only have a small *sample* and want to guess the spread of the whole, divide by $n-1$. The code shows why.
- **Covariance:** how two random things move together. Variance is the covariance of a thing with itself. For many things at once, the covariances form a covariance [[Matrix]].
- **Variance of an average:** the average of $n$ independent values has variance $\sigma^2/n$, so its standard deviation falls as $1/\sqrt{n}$. That is why bigger mini-batches give steadier estimates ([[Gradient Descent]]).
- **Weak spot:** one huge outlier can dominate the variance, because distances are squared. Some data have no finite variance at all.

```python
import itertools
import numpy as np

faces = np.arange(1, 7)
mean = faces.mean()
# 1) variance = average squared distance from the mean; two ways to compute it
v1 = np.mean((faces - mean) ** 2)
v2 = np.mean(faces ** 2) - mean ** 2
print(f"die: mean {mean}, variance {v1:.4f} = {v2:.4f}, standard deviation {np.sqrt(v1):.4f}")

# 2) same mean, different spread
A, B = np.array([5, 5, 5, 5]), np.array([0, 10, 0, 10])
print("A mean", A.mean(), "variance", A.var(), "| B mean", B.mean(), "variance", B.var())

# 3) shifting changes nothing, scaling by a multiplies variance by a^2
print("var(X+100):", round(np.var(faces + 100), 4), "| var(3X):", round(np.var(3 * faces), 4), "= 9 x", round(9 * v1, 4))

# 4) variances of independent things add; the same die twice does not
pairs = np.array(list(itertools.product(faces, faces)))
print("two independent dice: var of sum", round(np.var(pairs.sum(1)), 4), "= 2 x", round(v1, 4))
print("same die twice (X+X=2X): var", round(np.var(2 * faces), 4), "= 4 x", round(v1, 4))

# 5) the variance of an average of n values is variance / n, so the spread falls as 1/sqrt(n)
rng = np.random.RandomState(0)
for n in (1, 16, 256):
    means = rng.randint(1, 7, size=(20000, n)).mean(1)
    print(f"mean of {n:3d} rolls: spread {means.std():.3f} | formula sd/sqrt(n) {np.sqrt(v1) / np.sqrt(n):.3f}")

# 6) estimating variance from a small sample: divide by n or by n-1?
n = 5; S = rng.randint(1, 7, size=(100000, n))
print(f"true {v1:.4f} | divide by n: {S.var(1, ddof=0).mean():.4f} | divide by n-1: {S.var(1, ddof=1).mean():.4f}")

# 7) why squares: spread measured as mean absolute distance does not add across independent things
mad = lambda x: np.mean(np.abs(x - x.mean()))
print(f"mean absolute distance: one die {mad(faces):.4f}, two dice sum {mad(pairs.sum(1)):.4f}, (one + one = {2 * mad(faces):.4f})")
```

What it printed: a die has mean 3.5, variance 2.9167 (both formulas agree) and standard deviation 1.7078. Scores A have variance 0 and scores B have variance 25. Adding 100 leaves the variance at 2.9167, and tripling gives 26.25, which is 9 times 2.9167. Two independent dice give 5.8333, twice one die. The same die counted twice gives 11.6667, four times one die. The spread of an average of 1, 16 and 256 rolls was 1.705, 0.428 and 0.106, close to the formula's 1.708, 0.427 and 0.107. With samples of 5, dividing by $n$ gave an average of 2.3317 and dividing by $n-1$ gave 2.9146, against the true 2.9167.

## 🔴 4. Deep / formal
**The definition.** With $\mu=\mathbb{E}[X]$ the mean of the random value $X$:

$$
\mathrm{Var}(X)=\mathbb{E}\big[(X-\mu)^2\big]=\mathbb{E}[X^2]-\mu^2
$$

**Where:**
- $X$ is the random value and $\mu$ is its expected value.
- $X-\mu$ is the distance of a result from the average. **Matters:** without the squaring, distances above and below would cancel and the average distance would always be 0.
- $(\cdot)^2$ squares the distance. **Matters:** it counts far-away results much more than near ones (twice as far counts four times as much).
- $\mathbb{E}[\cdot]$ averages over all the results, weighted by their chances.
- The second form says "average of the squares minus square of the average". **Matters:** it gives the same number and is often quicker by hand.

**Example (tiny numbers):** scores $0, 10, 0, 10$, all equally likely, so $\mu=5$.
- Distances: $-5,\,5,\,-5,\,5$. Squared: $25,\,25,\,25,\,25$.
- $\mathrm{Var}=\mathbf{25}$, and $\sigma=\mathbf{5}$.
- Check with the second form: $\mathbb{E}[X^2]=(0+100+0+100)/4=50$, and $50-5^2=\mathbf{25}$.
- Meaning: a typical score is about 5 away from the average, which fits scores that are all either 0 or 10.

**The variance of a sum.**

$$
\mathrm{Var}(X+Y)=\mathrm{Var}(X)+\mathrm{Var}(Y)+2\,\mathrm{Cov}(X,Y)
$$

**Where:**
- $X$ and $Y$ are two random values.
- $\mathrm{Cov}(X,Y)=\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]$ is the covariance: positive if they tend to be high together, negative if one is high when the other is low, and 0 if they do not move together.
- **Matters:** for independent values $\mathrm{Cov}=0$ and the variances simply add. If they are linked, ignoring the covariance term gives the wrong spread, even though the average of the sum is still right ([[Expected Value]] linearity needs no such term).

**Example (tiny numbers):** a die has $\mathrm{Var}=35/12\approx2.9167$.
- Two independent dice: $\mathrm{Cov}=0$, so $2.9167+2.9167=\mathbf{5.8333}$.
- The same die twice: $\mathrm{Cov}(X,X)=\mathrm{Var}(X)=2.9167$, so $2.9167+2.9167+2\cdot2.9167=\mathbf{11.6667}$.
- Meaning: the code's listings give both numbers. Linked dice swing together, so their sum is twice as spread out as independent ones, even though the average is 7 in both cases.

**The variance of an average.** For $n$ independent values, each with variance $\sigma^2$:

$$
\mathrm{Var}\!\left(\frac{1}{n}\sum_{i=1}^{n}X_i\right)=\frac{\sigma^2}{n}
$$

**Where:**
- $X_i$ are the $n$ independent values, all with the same spread $\sigma^2$.
- $\frac1n\sum X_i$ is their average (for example a mini-batch loss).
- **Matters:** the average is $n$ times steadier in variance, and $\sqrt n$ times steadier in standard deviation. Without independence (linked values) this fails, and averaging helps much less.

**Example (tiny numbers):** a die, $\sigma^2=2.9167$, an average of $n=16$ rolls.
- $\mathrm{Var}=2.9167/16=\mathbf{0.1823}$.
- Standard deviation: $\sqrt{0.1823}=\mathbf{0.427}$.
- Meaning: the code's simulation of the mean of 16 rolls got 0.428, so averaging 16 rolls shrinks the spread from 1.708 to about a quarter of that.

**Further notes:**
- **Why $n-1$:** the sample mean is itself fitted to the sample, so the sample sits a bit closer to its own mean than to the true one, and dividing by $n$ comes out too small. In the code, with $n=5$, it came out at 2.3317, which is $4/5$ of the true 2.9167. Dividing by $n-1$ gave 2.9146. The exact proof is not shown here.
- **Not covered:** conditional variance, the law of total variance, variance with no finite value, covariance matrices in detail.

**What came before, and how this improved on it:**
- **Before:** measuring spread by the average of the absolute distances from the mean (the Wikipedia page mentions the expected absolute deviation as an alternative, but does not say it historically came first, so I do not claim it did).
- **Its limit:** it does not combine neatly across independent causes. In the code, one die has a mean absolute distance of 1.5 and the sum of two dice has 1.9444, not 1.5 + 1.5 = 3.
- **What changed:** Fisher introduced the term "variance" in his 1918 paper on Mendelian inheritance, arguing that independent causes of variability combine in quadrature, so the squared standard deviation is the more useful quantity. The page also credits the Bienaymé formula for the variance of a sum to 1853 (Wikipedia page read, but only its first 100,000 of about 132,700 characters; I did not read the originals).
- **How much better:** in the code, variances add exactly: two independent dice 5.8333 = 2 × 2.9167. The mean absolute distance does not: 1.9444 against 3.0.

## Related
- [[Expected Value]] · [[Probability]] · [[Normal Distribution]] · [[Bias-Variance]] · [[Normalization]] · [[Gradient Descent]] · [[Matrix]] · [[Covariance]] · [[Law of Large Numbers]]
- [[Anomaly Detection]]
## Sources
- Wikipedia, *Variance* (first 100,000 of about 132,700 characters read, for the history block: Fisher 1918, Bienaymé 1853, the sum and covariance formula)
- Figure: [Standard_deviation_diagram.svg](https://commons.wikimedia.org/wiki/File:Standard_deviation_diagram.svg) (M. W. Toews), CC BY 2.5, Wikimedia Commons.
- The code and its numbers are my own, run in this session.

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/probability Variance:::A number for how spread out results are around their average. Small when everyone is close to the average, big when they are scattered far from it.

#flashcards/math/formula #flashcards/topic/probability Write the variance formula and explain each symbol.
?
$$\mathrm{Var}(X)=\mathbb{E}\big[(X-\mu)^2\big]=\mathbb{E}[X^2]-\mu^2$$
- $X$: the random value. $\mu$: its expected value.
- $X-\mu$: the distance of a result from the average. Without squaring, distances above and below would cancel and the average distance would always be 0.
- $(\cdot)^2$: squares the distance, so far-away results count much more (twice as far counts four times as much).
- $\mathbb{E}[\cdot]$: averages over all results, weighted by their chances. The second form says "average of the squares minus the square of the average".

#flashcards/math/formula #flashcards/topic/probability Write the variance of a sum and explain each symbol.
?
$$\mathrm{Var}(X+Y)=\mathrm{Var}(X)+\mathrm{Var}(Y)+2\,\mathrm{Cov}(X,Y)$$
- $X$, $Y$: two random values.
- $\mathrm{Cov}(X,Y)$: the covariance. Positive if they tend to be high together, negative if one is high when the other is low, 0 if they do not move together.
- For independent values Cov = 0 and the variances simply add. If they are linked, ignoring the covariance term gives the wrong spread, even though the average of the sum is still right.

#flashcards/math/formula #flashcards/topic/probability Write the variance of an average of $n$ independent values and explain each symbol.
?
$$\mathrm{Var}\!\left(\frac{1}{n}\sum_{i=1}^{n}X_i\right)=\frac{\sigma^2}{n}$$
- $X_i$: the $n$ independent values, all with the same spread $\sigma^2$.
- $\frac1n\sum X_i$: their average (for example a mini-batch loss).
- The average is $n$ times steadier in variance and $\sqrt n$ times steadier in standard deviation. Without independence this fails and averaging helps much less.

#flashcards/math/cloze #flashcards/topic/probability Variance is the average of the ==squared== distances from the average. Standard deviation is its ==square root==.

#flashcards/math/example #flashcards/topic/probability Scores 0, 10, 0, 10 (equally likely). What are the variance and standard deviation?::μ = 5. Squared distances 25, 25, 25, 25. Var = 25, σ = 5. Check: E[X²] = 50, and 50 − 5² = 25.

#flashcards/math/example #flashcards/topic/probability A die has variance 35/12 ≈ 2.9167. What is the variance of the sum of two independent dice, and of the same die counted twice?::Independent: 2.9167 + 2.9167 = 5.8333. Same die twice: Cov = Var, so 2.9167 + 2.9167 + 2 × 2.9167 = 11.6667. Linked dice swing together.

#flashcards/math/example #flashcards/topic/probability A die has variance 2.9167. What is the variance and standard deviation of the average of 16 rolls?::2.9167 / 16 = 0.1823, so σ = 0.427 (the note's simulation gave 0.428). One roll has σ = 1.708, so averaging 16 shrinks it to a quarter.

#flashcards/math/image #flashcards/topic/probability ![[normal-distribution-sd.svg]]
What does this picture show?
?
The bell curve split into bands of one standard deviation each. Shares of values: 34.1% just above and below the mean, 13.6% in the next band out, then 2.1% and 0.1%.
