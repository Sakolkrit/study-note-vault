---
tags: [math, probability, statistics, law-of-large-numbers, foundations]
status: growing
level: 4
tier: 1
---

# Law of Large Numbers

A rule that says: **the more times you repeat a random thing, the closer the average of your results gets to the expected value.** Often shortened to **LLN**.

## 🟢 1. Simple (age 12)
Flip a coin 10 times and you might get 7 heads. Flip it a million times and you will get very close to half heads. A few tries are messy, but lots of tries even out. That is the law of large numbers: **luck averages out when you do something many times.**

**Example:** A shop owner cannot tell if the next customer will buy a drink. But after thousands of customers, the share who buy a drink is a steady number she can plan around.

## 🔵 2. How it works
1. Pick a random thing you can repeat, such as a coin flip or a die roll.
2. Repeat it many times and keep a **running average** of the results.
3. At the start the running average jumps around a lot. Each new result moves it less, because it is one of more and more.
4. Slowly the average settles near the [[Expected Value]]. The picture shows this for two dice: the average of the sum starts low and climbs to 7.
5. **A big misunderstanding:** the law does **not** say the coin "owes you" tails after a run of heads. Each flip is still 50/50. Early surprises are not fixed, they are just **drowned** by the many later flips.
6. The settling is also not free: the typical gap in the *count* of heads gets bigger (about 40 off after 10,000 flips, about 400 after a million), while the gap in the *share* gets smaller.
7. **It can fail.** If the random thing has no average (very wild, heavy-tailed values), the running average never settles. See the Cauchy test in the code.
8. **Where it shows up in AI:** it is why a mini-batch average estimates the true loss ([[Gradient Descent]]), why [[Monte Carlo]] methods work, and why test scores from enough examples can be trusted. How steady the average is depends on the [[Variance]].

![[law-of-large-numbers-dice.jpg]]
*The running average of the sum of two dice, plotted against the number of rolls. It starts low and climbs towards the line at 7, the expected value of the sum, as the rolls pile up. Image: Hihihi12345678, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:The_law_of_large_numbers_example_graph.jpg), CC0. Same file and caption as in [[Expected Value]], where it was looked at first.*

## 🟠 3. Technical detail
- **Sample mean:** $\bar X_n=\frac1n\sum_{i=1}^n X_i$, the average of $n$ independent results of the same random thing, with true mean $\mu$.
- **Weak law:** for any small margin $\varepsilon$, the chance that $\bar X_n$ is further than $\varepsilon$ from $\mu$ goes to 0 as $n$ grows ("converges in probability").
- **Strong law:** with probability 1, the sequence $\bar X_n$ itself ends up at $\mu$ ("converges almost surely"). The strong law implies the weak one (per the Wikipedia page).
- **A quantitative version:** Chebyshev's inequality gives a number for the weak law, using the [[Variance]] (see the next section).
- **Needs a mean:** the standard laws need the expected value to exist. For the Cauchy distribution it does not, and the average of $n$ Cauchy values has the same spread as a single one (page statement, checked in the code only by simulation).
- **Speed:** the share error falls like $1/\sqrt n$ (from the code, see the variance of an average in [[Variance]]). Ten times more data gives only about three times less error.

```python
import numpy as np
rng = np.random.RandomState(0)

# 1) share of heads settles to 0.5, but the COUNT gap does not shrink (average over 1000 runs)
for n in (10, 100, 10_000, 1_000_000):
    h = rng.binomial(n, 0.5, 1000)
    print(f"{n:9,d} flips: typical share error {np.abs(h / n - 0.5).mean():.4f} | typical heads gap |heads - n/2| = {np.abs(h - n / 2).mean():.1f}")

# 2) no 'catching up': after 10 heads in a row the next flip is still 50/50
flips = rng.randint(0, 2, 2_000_000)
streak = np.convolve(flips, np.ones(10), 'valid') == 10     # windows of 10 heads
nxt = flips[10:][streak[:len(flips) - 10]]
print(f"windows of 10 heads: {len(nxt)} | share of heads on the next flip: {nxt.mean():.3f}")

# 3) the law fails with no mean: Cauchy average does not settle, normal average does
for n in (100, 10_000, 1_000_000):
    c = [rng.standard_cauchy(n).mean() for _ in range(5)]
    g = [rng.randn(n).mean() for _ in range(5)]
    print(f"n={n:9,d} | Cauchy means {np.round(c, 2)} | normal means {np.round(g, 3)}")

# 4) Chebyshev bound P(|mean - mu| >= eps) <= var / (n eps^2), die rolls
var, eps, n = 35 / 12, 0.5, 100
m = rng.randint(1, 7, size=(100_000, n)).mean(1)
print(f"die, n={n}, eps={eps}: measured {np.mean(np.abs(m - 3.5) >= eps - 1e-9):.4f} | bound {var / (n * eps**2):.4f}")

# 5) how many fair-coin flips until the share is within 0.01 of 0.5 in 95% of runs?
bound = 0.25 / (0.01**2 * 0.05)
need = None
for n in range(1000, 60001, 1000):
    h = rng.binomial(n, 0.5, 20000)
    if np.mean(np.abs(h / n - 0.5) <= 0.01) >= 0.95: need = n; break
print(f"guarantee from the bound: {bound:,.0f} flips | simulation: about {need:,} flips")

# 6) a hand-sized case (1e-9 guards against float rounding at exactly 0.1): fair coin, n=100 flips, eps=0.1 -> bound 0.25/(100*0.01)
h = rng.binomial(100, 0.5, 200_000)
print(f"coin n=100, eps=0.1: measured {np.mean(np.abs(h / 100 - 0.5) >= 0.1 - 1e-9):.4f} | bound {0.25 / (100 * 0.1**2):.4f}")
# 7) running mean of five flips 1,0,1,1,0
fl = [1, 0, 1, 1, 0]; print("running means:", [round(sum(fl[:k]) / k, 3) for k in range(1, 6)])
```

What it printed: for coin flips the typical share error was 0.1238 at 10 flips, 0.0397 at 100, 0.0039 at 10,000 and 0.0004 at a million, while the typical gap in heads count grew from 1.2 to 4.0, 39.1 and 399.0. In 1,946 windows of 10 heads in a row, the next flip was heads 51.4% of the time, which is consistent with 50% (standard error about 1.1 points). With Cauchy values the means for $n=1{,}000{,}000$ ranged from −1.77 to 1.45, no tighter than at $n=100$ (−2.44 to 1.48), while the normal means shrank to about 0.001. For die rolls with $n=100$ and $\varepsilon=0.5$, the measured chance of being that far off was 0.0035, against a bound of 0.1167. For a fair coin and a 0.01 margin, the bound guarantees 50,000 flips are enough, while the simulation first passed at 10,000 (and failed at 9,000). The running means of the flips 1, 0, 1, 1, 0 were 1.0, 0.5, 0.667, 0.75, 0.6.

## 🔴 4. Deep / formal
**The law, in symbols.** With $\mu=\mathbb{E}[X]$ and $\bar X_n$ the average of $n$ independent copies of $X$:

$$
\bar X_n=\frac{1}{n}\sum_{i=1}^{n}X_i\;\longrightarrow\;\mu\qquad(n\to\infty)
$$

**Where:**
- $X_i$ is the $i$-th result, each drawn independently from the same random thing.
- $\bar X_n$ is the running average after $n$ results.
- $\mu$ is the expected value. **Matters:** the average settles at $\mu$ only if $\mu$ exists. Without it (Cauchy) the average keeps wandering.
- $\longrightarrow$ means "gets and stays as close as you like", in the weak or strong sense described in the technical section. **Matters:** the law says nothing about *how fast*, and nothing about any single run being exact.

**Example (tiny numbers):** a fair coin, heads $=1$ and tails $=0$, so $\mu=0.5$. Five flips: $1,0,1,1,0$.
- After 1 flip: $\bar X_1=1/1=\mathbf{1}$. After 2: $1/2=\mathbf{0.5}$. After 3: $2/3\approx\mathbf{0.667}$.
- After 4: $3/4=\mathbf{0.75}$. After 5: $3/5=\mathbf{0.6}$.
- Meaning: the early averages swing between 0.5 and 1, then move less and less. With more flips they would hover near 0.5.

**How far off might you be? (Chebyshev).** For results with variance $\sigma^2$ and any margin $\varepsilon>0$:

$$
\Pr\big(|\bar X_n-\mu|\ge\varepsilon\big)\le\frac{\sigma^2}{n\,\varepsilon^2}
$$

**Where:**
- $\Pr(\cdot)$ is the chance that the event inside happens.
- $|\bar X_n-\mu|\ge\varepsilon$ says "the average is off by at least $\varepsilon$".
- $\sigma^2$ is the variance of a single result. **Matters:** more spread means a looser guarantee.
- $n$ is the number of results. **Matters:** the bound falls as $1/n$, so more data tightens it, and it goes to 0, which proves the weak law when the variance is finite.
- $\varepsilon$ is how much error you accept. **Matters:** the bound grows with $1/\varepsilon^2$, so asking for half the error costs four times the data.

**Example (tiny numbers):** a fair coin, $\sigma^2=0.25$, $n=100$ flips, margin $\varepsilon=0.1$.
- Bound: $0.25/(100\times0.1^2)=0.25/1=\mathbf{0.25}$.
- So the share of heads is within 0.1 of 0.5 with chance at least $\mathbf{0.75}$.
- Meaning: the code measured the real chance of being 0.1 or more off as 0.0571, so the true guarantee is far better. The bound is safe, not sharp.

**Further notes:**
- **Related but different:** the [[Central Limit Theorem]] says what the *shape* of the error looks like (a bell curve), not just that it shrinks. Not written up here.
- **Not covered:** proofs of the weak and strong laws, and the exact conditions under which they hold. The statements come from the Wikipedia page, the proofs I did not read.
- **Independence:** the code and statements use independent results. Linked results can settle much more slowly or not at all.

**What came before, and how this improved on it:**
- **Before:** Gerolamo Cardano stated, without proof, that statistics from more trials are more accurate (Wikipedia page read, which gives his life dates as 1501–1576).
- **Its limit:** a feeling without a proof, and no number saying how many trials are enough.
- **What changed:** Jacob Bernoulli proved the first special case, for yes/no results. The page says he worked on it for over 20 years and published it in *Ars Conjectandi* in 1713, calling it his "golden theorem". Poisson named it the law of large numbers in 1837. Chebyshev handled the case of bounded variance, Khinchin showed in 1929 that a finite expected value is enough for the weak law of i.i.d. variables, and Kolmogorov proved the strong law in 1930 (all from the page; I did not read the originals, and I read only its first 100,000 of 102,853 characters).
- **How much better:** in the code, the guarantee from Chebyshev says 50,000 fair-coin flips are enough to be within 0.01 of the true share in 95% of runs, and the simulation first reached that at 10,000 flips (checked in steps of 1,000). So the proved bound is safe by about 5 times, and it turns "more is better" into a number.

## Related
- [[Expected Value]] · [[Variance]] · [[Probability]] · [[Normal Distribution]] · [[Gradient Descent]] · [[Monte Carlo]] · [[Central Limit Theorem]] · [[Cross-Entropy Loss]]
- [[Confidence Interval]]
## Sources
- Wikipedia, *Law of large numbers* (first 100,000 of 102,853 characters read, for the history block and the weak and strong statements)
- Figure: [The_law_of_large_numbers_example_graph.jpg](https://commons.wikimedia.org/wiki/File:The_law_of_large_numbers_example_graph.jpg) (Hihihi12345678), CC0, Wikimedia Commons.
- The code and its numbers are my own, run in this session.

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/probability Law of large numbers:::Luck averages out when you do something many times. A few tries are messy, lots of tries even out.

#flashcards/math/formula #flashcards/topic/probability Write the law of large numbers and explain each symbol.
?
$$\bar X_n=\frac{1}{n}\sum_{i=1}^{n}X_i\;\longrightarrow\;\mu\qquad(n\to\infty)$$
- $X_i$: the $i$-th result, each drawn independently from the same random thing.
- $\bar X_n$: the running average after $n$ results.
- $\mu$: the expected value. The average settles at $\mu$ only if $\mu$ exists. Without it (Cauchy) the average keeps wandering.
- $\longrightarrow$: "gets and stays as close as you like". The law says nothing about how fast, and nothing about any single run being exact.

#flashcards/math/formula #flashcards/topic/probability Write the Chebyshev-style bound used for the weak law and explain each symbol.
?
$$\Pr\big(|\bar X_n-\mu|\ge\varepsilon\big)\le\frac{\sigma^2}{n\,\varepsilon^2}$$
- $\Pr(\cdot)$: the chance of the event inside. $|\bar X_n-\mu|\ge\varepsilon$: the average is off by at least $\varepsilon$.
- $\sigma^2$: the variance of a single result. More spread gives a looser guarantee.
- $n$: the number of results. The bound falls as $1/n$, so more data tightens it and it goes to 0 (this proves the weak law when the variance is finite).
- $\varepsilon$: how much error you accept. The bound grows with $1/\varepsilon^2$, so asking for half the error costs 4 times the data.

#flashcards/math/cloze #flashcards/topic/probability The law of large numbers needs the expected value $\mu$ to ==exist==. It says nothing about ==how fast== the average settles.

#flashcards/math/example #flashcards/topic/probability Five coin flips give 1, 0, 1, 1, 0. What are the running averages?::1, 0.5, 0.667, 0.75, 0.6. The early averages swing, then move less and less towards 0.5.

#flashcards/math/example #flashcards/topic/probability A fair coin, $\sigma^2=0.25$, $n=100$, $\varepsilon=0.1$. What does the bound say, and what was the real chance?::Bound: 0.25 / (100 × 0.01) = 0.25, so the share of heads is within 0.1 of 0.5 with chance at least 0.75. The code measured the real chance of being 0.1 or more off as 0.0571, so the real guarantee is far better.

#flashcards/math/image #flashcards/topic/probability ![[law-of-large-numbers-dice.jpg]]
What does this picture show?
?
The running average of the sum of two dice against the number of rolls. It starts off and climbs towards the line at 7, the expected value of the sum, as the rolls pile up.
