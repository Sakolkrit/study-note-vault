---
tags: [normal-distribution, gaussian, probability, statistics, math, machine-learning]
status: growing
level: 4
tier: 1
---

# Normal Distribution

## 🟢 1. Simple (age 12)
The normal distribution is the famous **bell curve**. When you measure lots of people's heights, most are near the **average**, and fewer and fewer are very short or very tall. Drawn as a picture, the heights make a smooth bell shape: tall in the middle, thin at the edges.

**Example:** In a school of 1,000 kids the average height of age-12 kids might be 150 cm. Lots of kids are 148 to 152 cm. Only a few are 130 cm, and only a few are 170 cm. A bell curve describes exactly this pattern with just two numbers: the **average** and how **spread out** the kids are.

## 🔵 2. How it works
1. Collect many measurements of something (heights, test marks, measurement errors).
2. Find the **mean** (the average). It is the middle of the bell.
3. Find the **standard deviation**, written $\sigma$ (the Greek letter sigma): a typical distance from the mean. A small $\sigma$ gives a tall thin bell, a big $\sigma$ gives a wide flat one.
4. The bell is **symmetric**: the left side mirrors the right side.
5. A handy rule: about **68%** of values are within 1 $\sigma$ of the mean, about **95%** within 2 $\sigma$, and about **99.7%** within 3 $\sigma$.
6. Values far out (beyond 3 $\sigma$) are **rare**. That is how you spot something unusual.

Why it appears everywhere: whenever many small independent random things add up, the total tends to look like a bell (the central limit theorem, see [[Probability]]).

![[normal-distribution-sd.svg]]
*The bell curve split into bands of one standard deviation ($\sigma$) each. The percentages are the share of values in each band: 34.1% just above and below the mean, 13.6% in the next band out, then 2.1% and 0.1%. The labels add to 99.8% only because of rounding. Image: M. W. Toews, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Standard_deviation_diagram.svg), CC BY 2.5.*

## 🟠 3. Technical detail
- **Two parameters:** the mean $\mu$ (centre) and standard deviation $\sigma$ (spread). Variance is $\sigma^2$. A **standard normal** has $\mu=0$, $\sigma=1$.
- **Continuous:** a normal is described by a **density curve**. The chance of landing in a range is the **area** under the curve over that range (found with the integrals in [[Calculus]]). The chance of landing on one exact value is 0.
- **Z-score:** how many standard deviations a value is from the mean. It puts any bell curve on the same scale.
- **Sums stay normal:** adding independent normal values gives another normal. This is one reason it is so convenient.
- **In machine learning:**
  - Starting weights of a [[Neural Network]] are often drawn from a normal.
  - Noise on data is often assumed to be normal, which leads to the mean squared error in [[Loss Function]].
  - Sampling and noise in image generators use normals.
  - Normalising data (subtract the mean, divide by $\sigma$) makes training easier.
- **Not everything is normal:** incomes, city sizes and word frequencies have much heavier tails, where extreme values are far more common than a bell predicts.

```python
import math
def pdf(x, mu=0, sigma=1):
    return math.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))
print(round(pdf(70, 70, 5), 4))          # 0.0798  peak of the bell
print(round(pdf(80, 70, 5), 4))          # 0.0108  two sigmas out

cdf = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))   # chance below z
for k in (1, 2, 3):
    print(k, round(cdf(k) - cdf(-k), 4))  # 0.6827 0.9545 0.9973
print(round(1 - cdf(2), 4))               # 0.0228  chance of being above +2 sigma
```

## 🔴 4. Deep / formal
**The density.** The bell curve with mean $\mu$ and standard deviation $\sigma$ is:

$$
f(x)=\frac{1}{\sigma\sqrt{2\pi}}\;e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

**Where:**
- $x$ is the value you ask about, and $f(x)$ is the height of the curve there (the **density**, not a chance by itself).
- $\mu$ (mu) is the mean. **Matters:** it is where the curve peaks. Changing $\mu$ slides the whole bell left or right without changing its shape.
- $\sigma$ (sigma) is the standard deviation. **Matters:** it controls the width. Larger $\sigma$ makes a wider, flatter bell. The $\sigma$ in front also shrinks the height as it widens, so the total area stays exactly 1.
- $(x-\mu)^2$ is the squared distance from the centre. **Matters:** squaring makes the curve symmetric (left and right distances count the same) and makes it fall away faster the further out you go.
- $e^{-(\cdots)}$ turns that distance into a height between 0 and 1: no distance gives $e^0=1$ (the top), and larger distance gives smaller and smaller values.
- $\sqrt{2\pi}$ and $\sigma$ in the front are a scaling constant. **Matters:** they make the total area under the curve equal to 1, as every probability distribution requires.

**Example (tiny numbers):** adult test marks with mean $\mu=70$ and $\sigma=5$.
- At the peak, $x=70$: exponent is $0$, so $e^0=1$ and $f=\dfrac{1}{5\cdot2.507}=\mathbf{0.0798}$.
- At $x=80$ (that is $2\sigma$ above): exponent is $-\dfrac{10^2}{2\cdot25}=-2$. $e^{-2}=0.1353$. So $f=0.0798\cdot0.1353=\mathbf{0.0108}$.
- Meaning: the curve at 80 is only about $14\%$ as tall as at 70, so marks near 80 are far rarer than marks near 70. The numbers are heights of the curve, not chances: a chance needs an area (a range of marks).

**The z-score.** To compare values across different bell curves, measure how far a value is from the mean in units of standard deviation:

$$
z=\frac{x-\mu}{\sigma}
$$

**Where:**
- $x$ is the value, $\mu$ is the mean and $\sigma$ the standard deviation of its group.
- $x-\mu$ is how far the value is from the average, with a sign: positive above, negative below.
- Dividing by $\sigma$ counts that distance in standard deviations. **Matters:** it removes the units. A $z$ of $2$ means "two typical steps above average" whether the thing is marks, centimetres or dollars.

**Example (tiny numbers):** marks with $\mu=70$, $\sigma=5$. A student scores $80$.
- $z=\dfrac{80-70}{5}=\mathbf{2}$. A student scoring $65$ has $z=\dfrac{65-70}{5}=\mathbf{-1}$.
- From the 68-95-99.7 rule, about $95\%$ of students are within $z=\pm2$, so the $5\%$ outside is split into two tails of about $2.3\%$ each. The exact chance of scoring above $z=2$ is $\mathbf{0.0228}$.
- Meaning: this student is in roughly the top $2.3\%$.

**Further notes:**
- **Why the central limit theorem matters:** the average of many independent draws from almost any distribution (with finite variance) tends toward a bell curve as the number of draws grows. Stated from memory; the exact conditions are in a probability textbook. Full note: [[Central Limit Theorem]].
- **Maximum-likelihood link:** if you assume the errors in your data are normal with fixed $\sigma$, then finding the best fit means minimising the sum of squared errors, which is the MSE in [[Loss Function]].
- **Multivariate normal:** extends the bell to many dimensions ([[Vector]]s), described by a mean vector and a covariance [[Matrix]].
- **Heavy tails:** a model that assumes a normal will be badly surprised by extreme events, so check whether your data really looks like a bell before using it.

**What came before, and how this improved on it:**
- **Before:** add up exact binomial probabilities one by one, which is slow by hand when there are many trials. De Moivre found the bell-curve approximation in 1733 (from memory, not checked).
- **Its limit:** the exact sum needs one term for every possible count.
- **What changed:** replace the sum with one smooth formula, the bell curve, which also appears as the shape of many sums of small random effects.
- **How much better:** in the code below (100 fair coin flips, chance of 60 heads or fewer), the exact sum of 61 terms gives 0.9824 and the bell curve with the usual +0.5 correction gives 0.98214 in one formula. Without the correction it gives 0.97725. The largest gap between the exact probabilities and the curve over all 101 counts is 0.00020.

**Check it yourself:**

```python
import math
n, p = 100, 0.5
pm = lambda k: math.comb(n, k) * p ** k * (1 - p) ** (n - k)
exact = sum(pm(k) for k in range(61))
mu, sd = n * p, math.sqrt(n * p * (1 - p))
cdf = lambda x: 0.5 * (1 + math.erf((x - mu) / (sd * math.sqrt(2))))
print("heads in 100 fair flips, chance of 60 or fewer:")
print("  exact, adding 61 terms:", round(exact, 5))
print("  bell curve, one formula (with the usual +0.5 fix):", round(cdf(60.5), 5), " without the fix:", round(cdf(60), 5))
print("largest gap between the 101 exact probabilities and the bell curve:", f"{max(abs(pm(k) - math.exp(-0.5 * ((k - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))) for k in range(101)):.5f}")
```

## Related
- [[Probability]] · [[Calculus]] · [[Loss Function]] · [[Neural Network]] · [[Vector]] · [[Matrix]] · [[Overfitting]] · [[KL Divergence]] · [[Linear Regression]] · [[Gaussian Mixture Model]] · [[Maximum Likelihood]] · [[Diffusion Models]] · [[Variational Autoencoder]] · [[Matplotlib]]
- [[Anomaly Detection]] · [[Data Drift]]
- [[Confidence Interval]] · [[Hypothesis Testing]]
## Sources
- Blitzstein & Hwang, *Introduction to Probability* (free online)
- 3Blue1Brown, *But what is the Central Limit Theorem?* (video)
- Goodfellow, Bengio & Courville, *Deep Learning*, 2016, chapter 3

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/probability Normal distribution:::The bell curve. Most values are near the average, and fewer and fewer are very low or very high.

#flashcards/math/formula #flashcards/topic/probability Write the normal distribution density and explain each symbol.
?
$$f(x)=\frac{1}{\sigma\sqrt{2\pi}}\;e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
- $x$: the value you ask about. $f(x)$: the height of the curve there (the density, not a chance by itself).
- $\mu$: the mean, where the curve peaks. Changing it slides the bell left or right without changing its shape.
- $\sigma$: the standard deviation, which controls the width. Larger $\sigma$ gives a wider, flatter bell. The $\sigma$ in front keeps the total area exactly 1.

#flashcards/math/formula #flashcards/topic/probability Write the z-score and explain each symbol.
?
$$z=\frac{x-\mu}{\sigma}$$
- $x$: the value. $\mu$: the mean and $\sigma$ the standard deviation of its group.
- $x-\mu$: how far the value is from the average, with a sign (positive above, negative below).
- Dividing by $\sigma$ counts that distance in standard deviations and removes the units. $z=2$ means two typical steps above average, whether the thing is marks, centimetres or dollars.

#flashcards/math/cloze #flashcards/topic/probability About ==68%== of values fall within 1 standard deviation of the mean, ==95%== within 2 and ==99.7%== within 3.

#flashcards/math/example #flashcards/topic/probability Marks have $\mu=70$ and $\sigma=5$. What is the density at $x=70$ and at $x=80$?::At 70: exponent 0, f = 1 / (5 × 2.507) = 0.0798. At 80: exponent −2, e⁻² = 0.1353, so f = 0.0798 × 0.1353 = 0.0108.

#flashcards/math/example #flashcards/topic/probability Marks have $\mu=70$ and $\sigma=5$. What are the z-scores of 80 and 65?::80: z = (80 − 70)/5 = 2. 65: z = (65 − 70)/5 = −1. About 95% of students are within z = ±2.

#flashcards/math/image #flashcards/topic/probability ![[normal-distribution-sd.svg]]
What does this picture show?
?
The bell curve split into bands of one standard deviation each. Shares of values: 34.1% just above and below the mean, 13.6% in the next band out, then 2.1% and 0.1%.
