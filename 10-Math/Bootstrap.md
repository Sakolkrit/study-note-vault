---
tags: [bootstrap, resampling, statistics, probability, uncertainty, evaluation, math]
status: growing
level: 4
tier: 2
---

# Bootstrap

The bootstrap is **a way to find out how much a result could wobble by reshuffling the data you already have**. You draw new "pretend" data sets from your own data, with replacement, measure your result on each one, and look at how much those results differ. No formula for the wobble is needed.

## 🟢 1. Simple (age 12)
You have a bag with 10 marbles of different weights, and you want to know how far off your guess of the "average weight" could be. You cannot get more marbles. So you do a trick: **pick a marble, write down its weight, put it back, and pick again**, 10 times. That gives a new pretend bag (some marbles show up twice, some not at all). Work out its average. Do this a thousand times. If the thousand averages are all close together, your guess is steady; if they jump around, it is shaky.

**Example:** you timed a robot on 8 laps and want to say "about 31 seconds". Reshuffling the 8 lap times a thousand times shows the average could plausibly be 28 or 34. So you say "31, give or take 3".

## 🔵 2. How it works
1. **Start with your data**: $n$ items (people, games, laps). Treat them as the best picture you have of the world.
2. **Resample:** draw $n$ items from it **with replacement**. Some items appear twice or more, some not at all.
3. **Measure** your result on this pretend data set (the average, the hit rate, the median, a difference between two methods).
4. **Repeat** a few thousand times and keep all the results.
5. **Read the spread:** the middle 95% of the results is a 95% range for your number ([[Confidence Interval]]); their spread is a standard error.
6. **Resample the right unit.** If two methods were tried on the same people, resample the *people* and keep each person's two results together, so the pairing survives.

*I found no free, on-topic picture of the bootstrap on Wikimedia Commons (the search returned papers and unrelated figures), so this note has no picture.*

## 🟠 3. Technical detail
- **The percentile interval.** Let $\hat\theta^*_1,\dots,\hat\theta^*_B$ be the result computed on each of the $B$ resamples. The $(1-\alpha)$ interval is

$$
\Big[\;\hat\theta^*_{(\alpha/2)},\;\hat\theta^*_{(1-\alpha/2)}\;\Big]
$$

**Where:**
- $\hat\theta^*_b$ is the result (mean, hit rate, median...) on resample number $b$; $B$ is how many resamples (10,000 in my project).
- $\hat\theta^*_{(q)}$ is the value that a share $q$ of the resample results lie below: the $q$-quantile. For a 95% interval $\alpha=0.05$, so the 2.5% and 97.5% quantiles.
- **Matters:** it reads the interval straight off the reshuffled results, so it works for statistics that have no simple formula, such as a median or a difference of hit rates. It can be too narrow for small $n$ (see the limits below).

**Example (tiny numbers):** the sample $2,4,6,8$ has only $4^4=256$ possible resamples, so I list them all (my arithmetic in Python).
- The 256 resample means average to exactly $5.0$, the sample mean.
- Sorted, the 2.5% and 97.5% quantiles are $3.0$ and $7.0$.
- Interval: $\mathbf{[3.0,\,7.0]}$.
- Meaning: the resampled means rarely fall below 3 or above 7. (Compare the formula interval $2.47$ to $7.53$ for the same numbers in [[Confidence Interval]].)

- **The bootstrap standard error.**

$$
\text{SE}_{\text{boot}}=\sqrt{\frac{1}{B-1}\sum_{b=1}^{B}\big(\hat\theta^*_b-\bar{\theta}^*\big)^2},\qquad \bar\theta^*=\frac1B\sum_{b=1}^{B}\hat\theta^*_b
$$

**Where:**
- $\hat\theta^*_b$ and $B$ are as above; $\bar\theta^*$ is the average of the resample results.
- The formula is the ordinary standard deviation, applied to the resample results.
- **Matters:** it gives the wobble of *any* statistic, with no algebra. More resamples ($B$) make this number steadier; they do not make it more correct for a small, unlucky sample.

**Example (tiny numbers):** only $B=4$ resample results, $4,5,6,5$ (made up).
- $\bar\theta^*=(4+5+6+5)/4=5$. Differences: $-1,0,1,0$. Squares: $1,0,1,0$, sum $2$.
- $\text{SE}_{\text{boot}}=\sqrt{2/3}=\mathbf{0.816}$.
- Meaning: the result typically moves by about 0.8 between resamples. (Four resamples is far too few in practice.)

- **How much of the data is in a resample.** When you draw $n$ items with replacement, an item is never drawn with probability

$$
\Big(1-\frac1n\Big)^{n}\;\longrightarrow\;e^{-1}\approx0.368\quad(n\to\infty)
$$

**Where:**
- $n$ is the number of items. Each draw misses a given item with chance $1-1/n$, and there are $n$ independent draws.
- $e\approx2.718$ is Euler's number, so $e^{-1}=1/e$.
- **Matters:** a resample holds only about 63.2% of the different items (the rest are repeats). That is why resamples are "noisier copies" of the data, and why the items left out can serve as a free test set (the "out-of-bag" idea in a [[Random Forest]], from memory).

**Example (tiny numbers):** (my arithmetic).
- $n=4$: $0.75^4=\mathbf{0.3164}$. $n=10$: $0.9^{10}=0.3487$. $n=1000$: $0.999^{1000}=0.3677$.
- Meaning: already at $n=10$ more than a third of the items are missing from each resample.

- **Code: the three ideas above, and a coverage check.** The first part lists all 256 resamples of 2, 4, 6, 8. The second measures the share of rows a resample leaves out. The third bootstraps the **median** of 30 skewed numbers (true median $\ln2=0.6931$; the median has no easy standard-error formula) and counts how often the 95% interval contains the truth over 1,000 repeats:

```python
# file: toy_boot.py
import itertools
import numpy as np

x = np.array([2.0, 4.0, 6.0, 8.0])
means = np.array([np.mean(c) for c in itertools.product(x, repeat=4)])   # every one of the 4^4 = 256 equally likely resamples
print("tiny sample 2, 4, 6, 8: resamples", len(means), "| mean of the resample means", means.mean(),
      "| 95% percentile interval", np.percentile(means, [2.5, 97.5]))

print("share of the original rows that a resample leaves out:")
rng = np.random.default_rng(0)
for n in (4, 10, 1000):
    left = np.mean([1 - len(set(rng.integers(n, size=n))) / n for _ in range(2000)])
    print(f"   n = {n:4d}: measured {left:.4f}   (1 - 1/n)^n = {(1 - 1 / n) ** n:.4f}")
print(f"   1/e = {np.exp(-1):.4f}")

print("bootstrap of the MEDIAN of 30 exponential values (true median ln 2 = 0.6931); does the 95% interval cover it?")
hit, trials, B = 0, 1000, 1000
for _ in range(trials):
    s = rng.exponential(1.0, 30)
    med = np.median(s[rng.integers(30, size=(B, 30))], axis=1)        # B resamples, the median of each
    lo, hi = np.percentile(med, [2.5, 97.5]); hit += lo <= np.log(2) <= hi
print(f"   covered in {hit / trials:.1%} of {trials} repeats")
```

```text
tiny sample 2, 4, 6, 8: resamples 256 | mean of the resample means 5.0 | 95% percentile interval [3. 7.]
share of the original rows that a resample leaves out:
   n =    4: measured 0.3135   (1 - 1/n)^n = 0.3164
   n =   10: measured 0.3453   (1 - 1/n)^n = 0.3487
   n = 1000: measured 0.3678   (1 - 1/n)^n = 0.3677
   1/e = 0.3679
bootstrap of the MEDIAN of 30 exponential values (true median ln 2 = 0.6931); does the 95% interval cover it?
   covered in 94.4% of 1000 repeats
```

- **Reading it:** the 256 resample means average to 5.0 and the interval is $[3.0, 7.0]$. The measured left-out shares (0.3135, 0.3453, 0.3678) match $(1-1/n)^n$ (0.3164, 0.3487, 0.3677) and approach $1/e=0.3679$. The bootstrap interval for the median contained the truth in 94.4% of 1,000 repeats (Monte Carlo error about 0.7 points), so for this median, with $n=30$, the "95%" was about right. That is one distribution and one $n$; it does not show the bootstrap works everywhere.

## 🔴 4. Deep / formal
- **The idea.** The data's own distribution stands in for the unknown true one. Resampling from it imitates "drawing a new sample from the world", so the spread of results across resamples imitates the spread across new samples. It is an approximation that gets better as $n$ grows.
- **Where it fails.** (a) Very small $n$: the resamples can only reshuffle what was seen, so intervals tend to be too narrow (my 5-run comparisons in [[Error Bars Lab]] use a $t$ interval instead for this reason). (b) Extremes such as the maximum, which depend on values you did not see. (c) Heavy-tailed data, where one huge value drives the result. (d) **Dependent data:** shuffling a time series destroys its order, and resampling single rows treats related rows as independent (in a time series, resampling blocks of neighbouring points is the usual repair; from memory, not run).
- **Resample the independent unit.** In the recommender test the unit is the *person*, not the (person, method) row. Resampling people keeps both methods' results for the same person together, which is what makes the interval for a paired difference tight.
- **Percentile is the simplest version.** Refinements exist (bias-corrected and accelerated, "BCa", and the bootstrap-$t$) that fix some of its bias for small samples (from memory, not run).
- **Cost.** $B$ resamples cost $B$ times one evaluation. For the means in my project this was fast; for retraining a model on each resample it would be far too slow, and a cheaper approach (resample only the test results) is the usual choice. My project resamples test results, not training sets, so it ignores training luck.
- **Why 0/1 data is easy.** A bootstrap resample of $n$ hit-or-miss values with $h$ hits has a hit count that follows a binomial distribution (my arithmetic: each draw is a hit with chance $h/n$), so for a single hit rate no loop is needed. I used that shortcut in the coverage test.

**What came before, and how this improved on it:**
- **Before:** the wobble of a statistic was found with a formula worked out for that statistic (for a mean, $s/\sqrt n$), or approximated by calculus (the delta method, from memory); for many statistics no formula was known.
- **Its limit:** a median, a ratio, a difference in hit rates or a model's metric each needs its own derivation, and the formula often assumes a bell shape.
- **What changed:** one recipe for any statistic: resample, recompute, read the spread. Bradley Efron introduced it in 1979 (from memory, not checked).
- **How much better:** measured in my toy: the bootstrap interval for the median of 30 skewed values contained the truth in 94.4% of 1,000 repeats, for a statistic with no simple standard-error formula. In my project's hit-rate test it was no better than the simple formulas (93.4% against 93.5% and 93.8%), so it is not a cure for everything.

## Related
- [[Confidence Interval]] · [[Hypothesis Testing]] · [[Monte Carlo]] · [[Law of Large Numbers]] · [[Central Limit Theorem]] · [[Variance]] · [[Probability]] · [[Random Forest]] · [[Cross-Validation]] · [[Evaluation Metrics]] · [[Error Bars Lab]] · [[Recommender Shoot-out]] · [[Snake DQN]]

## Sources
- My own toy (Python 3.9.6, NumPy 1.24.4) and the project [[Error Bars Lab]]. The history (Efron 1979), the delta method, the refinements (BCa, bootstrap-$t$), block resampling for time series and the out-of-bag use in random forests are from memory; I did not read a source for this note.
- No image: none free and on-topic was found.

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/probability Bootstrap:::Estimating how much a result could wobble by drawing many pretend data sets from your own data, with replacement, recomputing the result on each and reading the spread.

#flashcards/math/reverse #flashcards/topic/probability Resampling with replacement:::Drawing n items from n items, putting each back after it is drawn, so some items appear several times and some not at all.

#flashcards/math/concept #flashcards/topic/probability What are the steps of the bootstrap?
?
Start with your n items; draw n of them with replacement; measure your result on that pretend data set; repeat a few thousand times; read the middle 95% of the results as a 95% range (or their spread as a standard error); resample the independent unit (such as people) so pairings survive.

#flashcards/math/formula #flashcards/topic/probability What is the bootstrap percentile interval?
?
$$\Big[\;\hat\theta^*_{(\alpha/2)},\;\hat\theta^*_{(1-\alpha/2)}\;\Big]$$
- $\hat\theta^*_b$: the result on resample $b$ (of $B$ resamples).
- $\hat\theta^*_{(q)}$: the $q$-quantile of the resample results; for 95%, $\alpha=0.05$, so the 2.5% and 97.5% quantiles.
- Matters: it reads the interval off the reshuffled results, so it works for statistics with no formula (median, a difference of hit rates). It can be too narrow for small samples.

#flashcards/math/example #flashcards/topic/probability Sample 2, 4, 6, 8: what is the bootstrap 95% interval for the mean when all 256 resamples are listed?::The 256 resample means average exactly 5.0; the 2.5% and 97.5% quantiles are 3.0 and 7.0, so the interval is [3.0, 7.0] (the formula interval is 2.47 to 7.53).

#flashcards/math/formula #flashcards/topic/probability What is the bootstrap standard error?
?
$$\text{SE}_{\text{boot}}=\sqrt{\frac{1}{B-1}\sum_{b=1}^{B}\big(\hat\theta^*_b-\bar{\theta}^*\big)^2},\qquad \bar\theta^*=\frac1B\sum_{b=1}^{B}\hat\theta^*_b$$
- $\hat\theta^*_b$: result on resample $b$. $B$: number of resamples. $\bar\theta^*$: their average.
- It is the ordinary standard deviation of the resample results.
- Matters: it gives the wobble of any statistic with no algebra; more resamples make the number steadier, not more correct for a small, unlucky sample.

#flashcards/math/example #flashcards/topic/probability Resample results 4, 5, 6, 5: what is the bootstrap standard error?::Mean 5; differences −1, 0, 1, 0; squares sum to 2; $\sqrt{2/3}=0.816$.

#flashcards/math/formula #flashcards/topic/probability What share of the items does a bootstrap resample leave out?
?
$$\Big(1-\frac1n\Big)^{n}\;\longrightarrow\;e^{-1}\approx0.368\quad(n\to\infty)$$
- $n$: number of items. Each draw misses a given item with chance $1-1/n$; there are $n$ independent draws.
- $e^{-1}=1/e$ with $e\approx2.718$.
- Matters: a resample holds only about 63.2% of the different items; the left-out items can serve as a free test set (out-of-bag, from memory).

#flashcards/math/example #flashcards/topic/probability What is the left-out share for n = 4, 10 and 1000?::$0.75^4=0.3164$; $0.9^{10}=0.3487$; $0.999^{1000}=0.3677$, approaching $1/e=0.3679$.

#flashcards/math/concept #flashcards/topic/probability What did the bootstrap median toy show, and what did it not show?
?
For 30 exponential values (true median 0.6931), the 95% bootstrap interval for the median contained the truth in 94.4% of 1,000 repeats (Monte Carlo error about 0.7 points), for a statistic with no simple formula. It was one distribution and one n; it does not show the bootstrap works everywhere.

#flashcards/math/concept #flashcards/topic/probability When does the bootstrap fail or mislead?
?
Very small samples (intervals too narrow), extremes such as the maximum, heavy-tailed data, and dependent data such as time series, where shuffling rows breaks the order (block resampling is the usual repair, from memory).

#flashcards/math/concept #flashcards/topic/probability Why resample people instead of (person, method) rows when comparing two recommenders?
?
The unit that is independent is the person. Resampling people keeps both methods' results for the same person together, so the paired difference keeps its tight interval.

#flashcards/math/concept #flashcards/topic/probability How did the bootstrap compare with the simple formulas for a hit rate in Error Bars Lab?
?
For a true hit rate of 0.087 with 943 people, 2,000 repeats: coverage of the 95% interval was 93.4% (bootstrap), 93.5% (normal) and 93.8% (Wilson); the bootstrap was not better. It is not a cure for everything.

#flashcards/math/concept #flashcards/topic/probability Which facts in the bootstrap note are from memory?
?
Efron 1979 as the origin, the delta method, BCa and bootstrap-t refinements, block resampling for time series and the out-of-bag use in random forests. Not checked.

#flashcards/math/cloze #flashcards/topic/probability In the bootstrap you draw with ==replacement== from your own data, and the middle ==95%== of the resample results gives the interval.
