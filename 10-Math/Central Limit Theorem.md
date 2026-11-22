---
tags: [math, probability, statistics, central-limit-theorem, normal-distribution, foundations]
status: growing
level: 4
tier: 2
---

# Central Limit Theorem

Often shortened to **CLT**. It says: **add up (or average) many small independent random things, and the result looks like a bell curve, whatever shape the small things had.**

## 🟢 1. Simple (age 12)
Roll one die and every number from 1 to 6 is equally likely, a flat shape. Roll two dice and add them, and the middle totals (like 7) show up more often than the ends (2 or 12). Add ten dice and the totals pile up in the middle in a smooth hill. **Adding lots of random things always tends to make the same hill shape**, the bell curve, even if each thing on its own looks nothing like a bell.

**Example:** Ask 100 people to guess the number of jelly beans in a jar. Some guess way too high, some way too low, but if you average everyone's guesses, the averages from different crowds of 100 land in a tidy bell-shaped pile around the same number.

## 🔵 2. How it works
1. Take a random thing with an average ([[Expected Value]]) and a spread ([[Variance]]). It can have a lopsided or odd shape.
2. Draw $n$ independent results and **average** them. Do it again, and again. The averages will be a bit different each time.
3. Look at the pile of those averages. They are centred on the true average, and their spread is the original spread divided by the square root of $n$ (the [[Law of Large Numbers]] says they shrink toward the average, and the CLT adds the **shape** of what is left).
4. As $n$ grows, the pile looks more and more like the [[Normal Distribution]].
5. So even when you know nothing about the odd shape underneath, you can say how big the error of an average is likely to be. For example, about 95% of the time an average is within about two of its standard errors of the truth.
6. **Warnings:** it is about **averages or sums**, not about single raw values. How big $n$ must be depends on how lopsided the source is. And it needs a finite spread, so very wild sources like the Cauchy distribution do not follow it.
7. **Where it shows up in AI:** error bars on test scores, why [[Monte Carlo]] errors are bell-shaped, why a mini-batch average of loss values behaves like a noisy bell around the true loss ([[Gradient Descent]]), and why so many things look normal.

![[dice-sum-distribution.svg]]
*A probability distribution: the chance of each total $S$ when you roll two dice. There are 36 equally likely pairs, and 6 of them add up to 7, so $p(7)=6/36=1/6$, the tallest bar. Totals of 2 and 12 have only one pair each, $1/36$. Image: Tim Stellmach, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Dice_Distribution_(bar).svg), public domain. Same file and caption as in [[Probability]], where it was looked at first. It is the first step of the story: one die is flat, two dice already make a peak.*

## 🟠 3. Technical detail
- **Classical statement:** for independent, identically distributed values with mean $\mu$ and finite variance $\sigma^2$, the standardised average $\sqrt n\,(\bar X_n-\mu)/\sigma$ gets closer and closer to a standard normal (mean 0, spread 1) as $n$ grows. "Closer" means in distribution (per the Wikipedia page).
- **Standard error:** the typical size of the error of the average is $\sigma/\sqrt n$. The code checks it through standardising.
- **Lopsided sources need more:** the skewness of the average falls like the source's skewness over $\sqrt n$ (my memory of a standard result; the code's numbers match it).
- **How fast:** the page says the Berry–Esseen theorem bounds the speed when the third moment is finite. It states the rate as on the order of $1/n$, but the standard bound is of order $1/\sqrt n$, and I treat the page's figure as an error. I did not check the theorem itself.
- **Not a rule of thumb:** the page warns that "n = 30 is enough" has no valid justification. The code has a case where it is not.
- **Fails without finite variance:** Cauchy averages keep the same spread. The code shows it.
- **Related versions:** versions exist for values that are not identical, and for dependent values such as [[Markov Chain]] averages (Markov himself proved one, per the page's history). Not covered.

```python
import math
import numpy as np
from itertools import product
rng = np.random.RandomState(0)
Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))

# 1) start from a lopsided shape: exponential (mean 1, sd 1, skewness 2). Average n of them: the shape evens out
for n in (1, 2, 5, 30, 200):
    m = rng.exponential(1, size=(100_000, n)).mean(1)
    z = (m - 1) / (1 / math.sqrt(n))                       # standardise: (mean - mu) / (sigma / sqrt n)
    skew = np.mean(z ** 3)
    print(f"n={n:3d}: skewness {skew:5.2f} (2/sqrt n = {2 / math.sqrt(n):4.2f}) | within 1 sd {np.mean(np.abs(z) < 1):.3f} | within 1.96 sd {np.mean(np.abs(z) < 1.96):.3f}")
print("bell curve says: within 1 sd 0.683 | within 1.96 sd 0.950")

# 2) dice: exact chance that the sum of 10 dice is at most 30, against the bell-curve guess
pmf = np.zeros(61); pmf[0] = 1
for _ in range(10): pmf = np.convolve(pmf, [0] + [1 / 6] * 6)[:61]
mu, sd = 10 * 3.5, math.sqrt(10 * 35 / 12)
print(f"10 dice, P(sum <= 30): exact {pmf[:31].sum():.4f} | bell curve {Phi((30 + 0.5 - mu) / sd):.4f} (with the half-step fix) | {Phi((30 - mu) / sd):.4f} (without)")

# 3) 'n = 30 is enough' is not a law: coverage of the 95% claim for a very lopsided source (lognormal, sigma 2)
src = lambda size: rng.lognormal(0, 2, size)
mu_l, sd_l = math.exp(2), math.sqrt((math.exp(4) - 1) * math.exp(4))
for n in (30, 1000):
    m = src((20_000, n)).mean(1)
    print(f"lognormal source, n={n:4d}: share of means within 1.96 sd of the truth {np.mean(np.abs(m - mu_l) < 1.96 * sd_l / math.sqrt(n)):.3f} (should be 0.950)")

# 4) it needs a finite variance: Cauchy means keep the same spread (interquartile range)
for n in (10, 1000):
    m = rng.standard_cauchy((5000, n)).mean(1); q = np.percentile(m, [25, 75])
    print(f"Cauchy, n={n:4d}: middle half of the means spans {q[1] - q[0]:.2f}")

# 5) a coin, n=100, error of at least 0.1: Chebyshev bound, bell-curve estimate, exact
n = 100
exact = sum(math.comb(n, k) for k in range(n + 1) if abs(k / n - 0.5) >= 0.1 - 1e-9) / 2 ** n
z = 0.1 / (0.5 / math.sqrt(n))
print(f"coin n=100, P(error >= 0.1): Chebyshev bound 0.2500 | bell curve (z = {z:.0f}) {2 * (1 - Phi(z)):.4f} | exact {exact:.4f}")
```

What it printed: for an exponential source, the standardised skewness of the average was 2.04 for $n=1$, 0.91 for $n=5$, 0.37 for $n=30$ and 0.12 for $n=200$, close to $2/\sqrt n$. The share within one standard deviation moved from 0.865 toward the bell curve's 0.683 (0.681 at $n=200$), and within 1.96 it was 0.950 at $n=200$. For 10 dice, the exact chance that the sum is at most 30 is 0.2050, the bell-curve guess with the half-step fix is 0.2024, and without it 0.1773. For a very lopsided lognormal source, the share of averages within 1.96 standard errors of the truth was 0.978 at $n=30$ and 0.968 at $n=1000$, not the promised 0.950 (I did not investigate why it is too high). Cauchy averages had a middle-half span of 2.06 at $n=10$ and 1.91 at $n=1000$, no narrower. For a coin with $n=100$ and an error of at least 0.1, the Chebyshev bound is 0.2500, the bell curve says 0.0455 and the exact answer is 0.0569.

## 🔴 4. Deep / formal
**The theorem.** For independent values $X_1,\dots,X_n$, each with mean $\mu$ and finite variance $\sigma^2$, and average $\bar X_n$:

$$
Z_n=\frac{\sqrt n\,(\bar X_n-\mu)}{\sigma}\;\longrightarrow\;\mathcal{N}(0,1)\qquad(n\to\infty)
$$

**Where:**
- $\bar X_n$ is the average of the $n$ values, $\mu$ their common mean and $\sigma$ their common standard deviation.
- $\bar X_n-\mu$ is the error of the average.
- $\sigma/\sqrt n$ is the standard error. **Matters:** dividing the error by it puts every source on one scale (centre 0, spread 1). Without this step the averages just shrink to a point and you see no shape.
- $\mathcal{N}(0,1)$ is the standard bell curve (see [[Normal Distribution]]). **Matters:** this is the shape that the standardised averages approach, whatever the shape of the source (given finite variance).
- $\longrightarrow$ means convergence in distribution: the chances of landing in any range approach the bell curve's chances.

**Example (tiny numbers):** a fair coin (heads $=1$), $\mu=0.5$, $\sigma=0.5$, $n=100$ flips. What is the chance the share of heads is off by at least 0.1?
- Standard error: $0.5/\sqrt{100}=\mathbf{0.05}$.
- Error 0.1 is $0.1/0.05=\mathbf{2}$ standard errors, so $z=2$.
- Bell curve: both tails past $z=2$ hold about $2\,(1-\Phi(2))=\mathbf{0.0455}$.
- Meaning: about a 4.6% chance, against the Chebyshev bound of 25% from [[Law of Large Numbers]]. The exact answer, summed over all counts of heads, is 0.0569, so the bell curve is close and much tighter than the bound.

**A 95% range for the average.**

$$
\mu\;\pm\;1.96\,\frac{\sigma}{\sqrt n}
$$

**Where:**
- $\mu$ is the true mean and $\sigma/\sqrt n$ the standard error.
- $1.96$ is the number of standard errors that holds 95% of a bell curve (the code's 0.950 at $n=200$ matches).
- **Matters:** it turns the theorem into a usable statement: about 95% of averages fall in this range. It holds only to the extent that the bell-curve approximation is good for the $n$ you have. Using $\pm1$ instead of $\pm1.96$ gives about 68% instead.

**Example (tiny numbers):** a source with $\mu=1$ and $\sigma=1$ (the exponential in the code), and $n=100$.
- Standard error: $1/\sqrt{100}=\mathbf{0.1}$.
- Range: $1\pm1.96\times0.1=1\pm0.196$, which is $\mathbf{0.804}$ to $\mathbf{1.196}$.
- Meaning: about 95% of averages of 100 such values land between 0.804 and 1.196. The code, at $n=200$ and $n=30$, found 0.950 and 0.952 inside the corresponding ranges.

**Further notes:**
- **Where it breaks (checked in the code):** heavy tails slow it down. For the lognormal source, even $n=1000$ gave 0.968 rather than 0.950. Cauchy has no finite variance and the theorem does not apply.
- **Known vs estimated $\sigma$:** in practice $\sigma$ is usually estimated from the same data. The code uses the true $\sigma$. The effect of estimating it is not covered.
- **Not covered:** proof (via characteristic functions or the Lindeberg condition), the Berry–Esseen bound, multi-dimensional versions.

**What came before, and how this improved on it:**
- **Before:** the law of large numbers, which only says the average converges to the mean. The Wikipedia page says it "says nothing about the shape of the fluctuations around" the mean.
- **Its limit:** no way to say how likely a given error is. The Chebyshev bound gives a safe but loose number: 0.2500 for the coin case in the code.
- **What changed:** de Moivre used the bell curve to approximate the number of heads in many coin tosses in 1733 (the earliest version, per the page), Laplace extended it to the binomial in 1812, Lyapunov stated and proved it in general in 1901, and George Pólya gave it the name "central limit theorem" in 1920. Lévy published the first complete proof of the generalised theorem in 1937 (Wikipedia page read, first 100,000 of about 140,500 characters; I did not read the originals).
- **How much better:** for the coin with $n=100$ and error at least 0.1, the bound said at most 0.2500 and the bell curve says 0.0455, against the exact 0.0569, so the CLT estimate is about 5 times tighter than the bound and about 20% under the truth in this case. In the code's exponential case, the bell curve's 0.683 and 0.950 are matched to within 0.002 at $n=200$.

## Related
- [[Normal Distribution]] · [[Law of Large Numbers]] · [[Expected Value]] · [[Variance]] · [[Probability]] · [[Monte Carlo]] · [[Markov Chain]] · [[Gradient Descent]]
- [[Confidence Interval]] · [[Bootstrap]]
## Sources
- Wikipedia, *Central limit theorem* (first 100,000 of about 140,500 characters read, for the history block, the statement, the Berry–Esseen remark and the Cauchy and n = 30 warnings)
- Figure: [Dice_Distribution_(bar).svg](https://commons.wikimedia.org/wiki/File:Dice_Distribution_(bar).svg) (Tim Stellmach), public domain, Wikimedia Commons. Same file as in [[Probability]].
- The code and its numbers are my own, run in this session.

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/probability Central Limit Theorem:::Averages (or sums) of many random values pile up in a bell curve, even if each value alone looks nothing like a bell (given finite variance).

#flashcards/math/formula #flashcards/topic/probability Write the Central Limit Theorem and explain each symbol.
?
$$Z_n=\frac{\sqrt n\,(\bar X_n-\mu)}{\sigma}\;\longrightarrow\;\mathcal{N}(0,1)\qquad(n\to\infty)$$
- $\bar X_n$: the average of the $n$ values. $\mu$: their common mean. $\sigma$: their common standard deviation.
- $\bar X_n-\mu$: the error of the average.
- $\sigma/\sqrt n$: the standard error. Dividing the error by it puts every source on one scale (centre 0, spread 1). Without it the averages just shrink to a point.
- $\mathcal{N}(0,1)$: the standard bell curve. The standardised averages approach this shape whatever the source shape.
- $\longrightarrow$: convergence in distribution, so chances for any range approach the bell curve's chances.

#flashcards/math/formula #flashcards/topic/probability Write the 95% range for an average and explain each symbol.
?
$$\mu\;\pm\;1.96\,\frac{\sigma}{\sqrt n}$$
- $\mu$: the true mean. $\sigma/\sqrt n$: the standard error.
- $1.96$: the number of standard errors that holds 95% of a bell curve.
- It says about 95% of averages fall in this range, as far as the bell-curve approximation is good for your $n$. Using $\pm1$ gives about 68% instead.

#flashcards/math/cloze #flashcards/topic/probability The standard error is ==σ/√n==. A range of ±1.96 standard errors holds about ==95%== of averages, and ±1 holds about ==68%==.

#flashcards/math/example #flashcards/topic/probability A fair coin is flipped 100 times. About how likely is the share of heads to be off by at least 0.1?::Standard error 0.5/√100 = 0.05. 0.1 is 2 standard errors, so about 2 × (1 − Φ(2)) = 0.0455, roughly 4.6%.

#flashcards/math/example #flashcards/topic/probability A source has $\mu=1$, $\sigma=1$ and $n=100$. What range holds about 95% of averages?::Standard error 0.1, so 1 ± 1.96 × 0.1 = 0.804 to 1.196.

#flashcards/math/image #flashcards/topic/probability ![[dice-sum-distribution.svg]]
What does this picture show?
?
The chance of each total when you roll two dice. 36 equally likely pairs, 6 of them add up to 7, so p(7) = 6/36 = 1/6 is the tallest bar. Totals 2 and 12 have one pair each (1/36). One die is flat, two dice already make a peak.
