---
tags: [confidence-interval, standard-error, statistics, probability, uncertainty, evaluation, math]
status: growing
level: 4
tier: 2
---

# Confidence Interval

A confidence interval is **a range of values that goes with an estimate to show how much it could be off just because of luck in the sample**. Instead of saying "the hit rate is 0.087", you say "0.087, and with a sample this size it could easily be anywhere from 0.069 to 0.105".

## 🟢 1. Simple (age 12)
You taste one spoon of soup from a big pot and say "it needs salt". But one spoon might be unlucky. If you tasted a different spoon, you might say something a little different. A **confidence interval** is a way to say how much the answer could change if you had tasted other spoons: "the saltiness is about 5, give or take 2.5". A small spoon means a wide "give or take"; a big spoon means a narrow one.

**Example:** two football teams score 3 and 2 goals in their last match. One match is a tiny spoon, so "team 1 is better" could easily be luck. After 100 matches the "give or take" is much smaller, and the same difference starts to mean something.

## 🔵 2. How it works
1. **Measure** something on a sample: an average score, a share of hits, a difference between two methods.
2. **Ask how much it would wobble** if you collected a fresh sample of the same size. That wobble is called the **standard error**.
3. **Build a range** around the measured value: the value, minus and plus about two standard errors.
4. **Read it.** A narrow range means the sample told you a lot. A wide range means it told you little.
5. **Compare ranges.** If a range for a *difference* includes 0, the data cannot tell whether there is a difference at all ([[Hypothesis Testing]]).
6. **No formula for your number?** Use the [[Bootstrap]]: reshuffle your own data many times and look at how the answer wobbles.

![[confidence-interval-95-bar-chart.svg]]
*Bars with 95% intervals, drawn against one reference effect "A" (the black bar). The red bars have intervals that still reach the level of A, so the difference could be luck ("non-significant"); the blue bars have intervals that stay clear of A ("significant"). Author: Leonhard Bamberg, CC BY-SA 4.0, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:95%25_confidence_interval.svg). The picture treats A as one fixed line; when A has its own error bars the rule is less simple (my note, not the picture's).*

## 🟠 3. Technical detail
- **The standard error of an average.** Take $n$ numbers $x_1,\dots,x_n$. The wobble of their mean is

$$
\text{SE}=\frac{s}{\sqrt{n}},\qquad s=\sqrt{\frac{\sum_{i=1}^{n}(x_i-\bar x)^2}{n-1}}
$$

**Where:**
- $n$ is how many numbers you have. $x_i$ is one of them; $\bar x$ is their mean.
- $s$ is the sample standard deviation: how spread out single values are ([[Variance]], the square root of it). The $n-1$ (not $n$) corrects for having used the same data to find $\bar x$.
- $\text{SE}$ is the standard error: the typical size of the gap between the sample mean and the true mean.
- **Matters:** it shrinks only with the **square root** of $n$: to halve the wobble you need four times the data. Spread of single values ($s$) and wobble of the mean (SE) are different things; mixing them up is a common mistake.

**Example (tiny numbers):** four values $2,4,6,8$ (made up; my arithmetic in Python).
- Mean $\bar x=5$. Differences from the mean: $-3,-1,1,3$. Squares: $9,1,1,9$, sum $20$.
- $s^2=20/3=6.667$, so $s=2.582$.
- $\text{SE}=2.582/\sqrt4=\mathbf{1.291}$.
- Meaning: the mean of 4 such numbers is typically about 1.3 away from the true mean.

- **The interval.** For a 95% interval, go about two standard errors either side:

$$
\bar x\;\pm\;z\cdot\text{SE},\qquad z=1.96\ \text{for 95\%}
$$

**Where:**
- $\bar x$ is the measured value and $\text{SE}$ its standard error from the equation above.
- $z$ is a multiplier from the [[Normal Distribution]]: 1.96 leaves 2.5% in each tail, 1.64 would give a 90% interval, 2.58 a 99% interval (from memory of the usual table).
- **Matters:** a higher confidence level gives a wider range. The recipe relies on the sample mean being roughly bell-shaped, which the [[Central Limit Theorem]] promises for large $n$ but not for tiny or very skewed samples (see the code below).

**Example (tiny numbers):** the same four values, $\bar x=5$, $\text{SE}=1.291$.
- $1.96\times1.291=2.53$.
- Interval: $5\pm2.53$, that is $\mathbf{2.47\ to\ 7.53}$.
- Meaning: if the numbers were 4 draws from a bell-shaped source, an interval built this way would contain the true mean in about 95% of repeats. With 4 numbers that promise is weak (the code shows why).

- **A share, such as a hit rate.** For a 0/1 outcome with $\hat p$ hits per try:

$$
\text{SE}_{\hat p}=\sqrt{\frac{\hat p\,(1-\hat p)}{n}}
$$

**Where:**
- $\hat p$ is the measured share (hits divided by tries) and $n$ the number of tries.
- $\hat p(1-\hat p)$ is the variance of one 0/1 outcome; dividing by $n$ and taking the root is the same rule as above.
- **Matters:** the wobble is biggest when $\hat p$ is near $0.5$ and small near 0 or 1, but a small $\hat p$ with small $n$ is still very uncertain. The interval $\hat p\pm1.96\,\text{SE}_{\hat p}$ can reach below 0 for tiny $\hat p$; the *Wilson* interval (used in [[Error Bars Lab]]) avoids that.

**Example (tiny numbers):** 9 hits in 100 tries (made up).
- $\hat p=0.09$. $\text{SE}=\sqrt{0.09\times0.91/100}=\sqrt{0.000819}=\mathbf{0.0286}$.
- $1.96\times0.0286=0.056$. Interval: $0.09\pm0.056$, that is $0.034$ to $0.146$.
- Meaning: 9 hits in 100 is compatible with a true rate anywhere from about 3% to 15%.

- **Code: three checks by simulation.** Part 1 draws many samples and measures how much the sample mean really wobbles. Part 2 builds the 95% interval 10,000 times from skewed data (exponential numbers, true mean 1) and counts how often it contains the truth. Part 3 gives the interval for a hit rate:

```python
# file: toy_ci.py
import numpy as np

rng = np.random.default_rng(0)
print("1) the standard error shrinks like 1/sqrt(n) (data: exponential, true mean 1, true spread 1)")
for n in (25, 100, 400):
    means = rng.exponential(1.0, size=(10000, n)).mean(axis=1)           # 10,000 repeats of "draw n values, take the mean"
    print(f"   n = {n:3d}: spread of the sample mean {means.std():.4f}   1/sqrt(n) = {1 / n ** 0.5:.4f}")

print("2) how often does  mean +/- 1.96 * s/sqrt(n)  contain the true mean 1?  (10,000 repeats)")
for n in (5, 10, 30, 200):
    x = rng.exponential(1.0, size=(10000, n))
    m, se = x.mean(axis=1), x.std(axis=1, ddof=1) / n ** 0.5
    print(f"   n = {n:3d}: covers the truth in {np.mean(np.abs(m - 1) <= 1.96 * se):.1%} of repeats")

print("3) a proportion, such as a hit rate: 9 hits in 100 people, and the real case 82 hits in 943")
for hits, n in ((9, 100), (82, 943)):
    p = hits / n; se = (p * (1 - p) / n) ** 0.5
    print(f"   p = {p:.4f}, standard error {se:.4f}, 95% interval {p - 1.96 * se:.3f} to {p + 1.96 * se:.3f}")
```

```text
1) the standard error shrinks like 1/sqrt(n) (data: exponential, true mean 1, true spread 1)
   n =  25: spread of the sample mean 0.2047   1/sqrt(n) = 0.2000
   n = 100: spread of the sample mean 0.0994   1/sqrt(n) = 0.1000
   n = 400: spread of the sample mean 0.0503   1/sqrt(n) = 0.0500
2) how often does  mean +/- 1.96 * s/sqrt(n)  contain the true mean 1?  (10,000 repeats)
   n =   5: covers the truth in 81.1% of repeats
   n =  10: covers the truth in 86.5% of repeats
   n =  30: covers the truth in 91.7% of repeats
   n = 200: covers the truth in 94.9% of repeats
3) a proportion, such as a hit rate: 9 hits in 100 people, and the real case 82 hits in 943
   p = 0.0900, standard error 0.0286, 95% interval 0.034 to 0.146
   p = 0.0870, standard error 0.0092, 95% interval 0.069 to 0.105
```

- **Reading it:** the wobble of the mean matches $1/\sqrt{n}$ (0.2047, 0.0994, 0.0503 against 0.2, 0.1, 0.05). The "95%" interval is **not** 95% in small, skewed samples: it contained the truth in 81.1% of repeats at $n=5$, 86.5% at $n=10$, 91.7% at $n=30$ and 94.9% at $n=200$. So the label is only as true as the conditions: bell-shaped mean, enough data. Part 3 shows that 82 hits in 943 people (the best method of [[Recommender Shoot-out]]) gives 0.069 to 0.105.

## 🔴 4. Deep / formal
- **What "95%" means.** It is a statement about the *recipe*: if you repeated the whole experiment many times and built an interval each time, about 95% of the intervals would contain the true value. It is **not** "there is a 95% chance the true value is inside this particular interval". The true value is fixed; the interval is what moves. (A different school, Bayesian credible intervals, answers the second question; see [[Bayes' Theorem]].)
- **Wobble from which source?** The interval covers only the randomness it was built for. In [[Error Bars Lab]] the intervals for the recommender hit rates cover the luck of *which people* are in the test set. They do not cover luck in the training (a different random start of the model), the choice of the split, or the choice of the data set.
- **Independence.** The $\sqrt n$ rule assumes the $n$ values are independent. Games played by one trained network are independent given the network; the five networks in [[Snake DQN]] shared the same 200 games, so five times the networks did not give five times the information.
- **Coverage is something you can measure.** In [[Error Bars Lab]] I simulated a hit rate of 0.087 with 943 people and built three kinds of 95% interval 2,000 times: the normal one, the Wilson one and a bootstrap one. All three contained the truth in only 93.4% to 93.8% of repeats (Monte Carlo error about 0.5 points), so with a rate this small and a count this size the "95%" intervals were slightly too optimistic. I did not test why.
- **Width versus information.** To halve the width you need 4 times the data; 10 times narrower needs 100 times the data. That is why a gap of 0.013 in a hit rate near 0.08 needs far more than 943 people to be pinned down.
- **Other ways to get an interval.** The [[Bootstrap]] (no formula needed), the exact binomial (Clopper-Pearson) interval for a share, the $t$ interval with a table value instead of 1.96 for small $n$ (I used it for 5 training runs in [[Error Bars Lab]]), and for a difference of two paired values, the interval of the paired differences.

**What came before, and how this improved on it:**
- **Before:** report a single number (a mean or a score) with no sign of how much luck is in it, or report only whether a result was "significant".
- **Its limit:** a single number hides whether a gap between two methods is bigger than the noise; a yes/no test hides how big the effect might be.
- **What changed:** an interval shows the size of the effect and the uncertainty at once. The confidence interval was introduced by Jerzy Neyman in the 1930s (from memory, not checked).
- **How much better:** measured in [[Error Bars Lab]]: the single numbers 0.087 and 0.074 look like a win for the first; the interval for their difference is $-0.007$ to $+0.033$ and includes 0. In the toy, the textbook interval on 5 skewed values is a 95% interval in name only (81.1% coverage). No published figure is quoted.

## Related
- [[Bootstrap]] · [[Hypothesis Testing]] · [[Normal Distribution]] · [[Central Limit Theorem]] · [[Law of Large Numbers]] · [[Variance]] · [[Expected Value]] · [[Probability]] · [[Monte Carlo]] · [[Bayes' Theorem]] · [[Evaluation Metrics]] · [[Retrieval Metrics]] · [[Cross-Validation]] · [[Error Bars Lab]] · [[Recommender Shoot-out]] · [[Snake DQN]]
- [[AB Testing|A/B Testing]]
## Sources
- My own toy (Python 3.9.6, NumPy 1.24.4) and the project [[Error Bars Lab]]. The multipliers 1.64 and 2.58, the 1930s history (Neyman), the Clopper-Pearson interval and the Bayesian credible interval are from memory; I did not read a source for this note.
- Picture: Leonhard Bamberg, CC BY-SA 4.0, [File:95% confidence interval.svg](https://commons.wikimedia.org/wiki/File:95%25_confidence_interval.svg), Wikimedia Commons. I rejected a second picture of a bell curve whose description called the 95% a probability about the parameter, which is the common wrong reading.

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/probability Confidence interval:::A range around an estimate, built so that if the whole experiment were repeated many times, about 95% (for a 95% interval) of the ranges would contain the true value. It shows how much luck in the sample could move the estimate.

#flashcards/math/reverse #flashcards/topic/probability Standard error:::The typical size of the gap between a sample estimate (such as a mean) and the true value; for a mean it is the sample standard deviation divided by the square root of n.

#flashcards/math/concept #flashcards/topic/probability What are the steps of building a confidence interval for an average?
?
Measure the average on a sample; work out its standard error (spread of single values divided by the square root of n); go about two standard errors either side (1.96 for 95%); read a narrow range as informative and a wide one as weak.

#flashcards/math/formula #flashcards/topic/probability What is the standard error of a sample mean?
?
$$\text{SE}=\frac{s}{\sqrt{n}},\qquad s=\sqrt{\frac{\sum_{i=1}^{n}(x_i-\bar x)^2}{n-1}}$$
- $n$: number of values. $x_i$: one value. $\bar x$: their mean.
- $s$: sample standard deviation (spread of single values); $n-1$ corrects for using the same data to find the mean.
- SE: typical gap between the sample mean and the true mean.
- Matters: it shrinks with the square root of $n$, so halving it needs four times the data. Spread of values and wobble of the mean are different things.

#flashcards/math/example #flashcards/topic/probability Values 2, 4, 6, 8: what are the mean, $s$ and the standard error?::Mean 5; squared differences $9+1+1+9=20$; $s^2=20/3=6.667$, $s=2.582$; $\text{SE}=2.582/\sqrt4=1.291$.

#flashcards/math/formula #flashcards/topic/probability What is the 95% confidence interval for a mean?
?
$$\bar x\;\pm\;z\cdot\text{SE},\qquad z=1.96\ \text{for 95\%}$$
- $\bar x$: the measured mean. SE: its standard error.
- $z$: multiplier from the normal distribution (1.96 for 95%; about 1.64 for 90% and 2.58 for 99%, from memory).
- Matters: higher confidence gives a wider range. It relies on the sample mean being roughly bell-shaped, which fails for tiny or very skewed samples.

#flashcards/math/example #flashcards/topic/probability Mean 5 and SE 1.291: what is the 95% interval?::$1.96\times1.291=2.53$, so $5\pm2.53$, that is 2.47 to 7.53. With only 4 values the 95% label is weak.

#flashcards/math/formula #flashcards/topic/probability What is the standard error of a share (such as a hit rate)?
?
$$\text{SE}_{\hat p}=\sqrt{\frac{\hat p\,(1-\hat p)}{n}}$$
- $\hat p$: measured share (hits divided by tries). $n$: number of tries.
- $\hat p(1-\hat p)$: the variance of one 0/1 outcome.
- Matters: the wobble is largest near $\hat p=0.5$, but a small $\hat p$ with small $n$ is still very uncertain. The plain interval can go below 0; the Wilson interval does not.

#flashcards/math/example #flashcards/topic/probability 9 hits in 100 tries: what are $\hat p$, SE and the 95% interval?::$\hat p=0.09$; $\text{SE}=\sqrt{0.09\times0.91/100}=0.0286$; $0.09\pm1.96\times0.0286=0.09\pm0.056$, that is 0.034 to 0.146.

#flashcards/math/image #flashcards/topic/probability ![[confidence-interval-95-bar-chart.svg]]
What does this picture show?
?
Bars with 95% intervals drawn against one reference effect A. Red bars have intervals that still reach A's level (difference could be luck); blue bars stay clear of it. It treats A as a fixed line. (Leonhard Bamberg, CC BY-SA 4.0, Wikimedia Commons.)

#flashcards/math/concept #flashcards/topic/probability What does "95%" in a 95% confidence interval mean, and what does it not mean?
?
If the whole experiment were repeated many times, about 95% of the intervals built this way would contain the true value. It does not mean there is a 95% chance that the true value lies in this one interval.

#flashcards/math/concept #flashcards/topic/probability What did the coverage simulation in the confidence-interval toy show?
?
Exponential data with true mean 1, 10,000 repeats: the 95% interval contained the truth in 81.1% of repeats at n=5, 86.5% at n=10, 91.7% at n=30 and 94.9% at n=200. The label is only as true as the conditions (bell-shaped mean, enough data).

#flashcards/math/concept #flashcards/topic/probability What did Error Bars Lab find about the coverage of 95% intervals for a hit rate of 0.087 with 943 people?
?
In 2,000 simulated repeats, the normal, Wilson and bootstrap intervals contained the truth in 93.5%, 93.8% and 93.4% (Monte Carlo error about 0.5 points), slightly under 95%. I did not test why.

#flashcards/math/concept #flashcards/topic/probability What sources of luck does a hit-rate interval over test people not cover?
?
Luck in the training (a different random start), the choice of the split, and the choice of data set. It covers only which people happen to be in the test set.

#flashcards/math/concept #flashcards/topic/probability Which facts in the confidence-interval note are from memory?
?
The 1.64 and 2.58 multipliers, Neyman in the 1930s as the origin, the Clopper-Pearson and Bayesian credible intervals. Not checked against a source.

#flashcards/math/cloze #flashcards/topic/probability To halve the width of a confidence interval you need ==four times== as much data, because the standard error shrinks with the ==square root== of n.
