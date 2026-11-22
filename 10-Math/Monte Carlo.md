---
tags: [math, probability, simulation, sampling, monte-carlo, foundations]
status: growing
level: 4
tier: 2
---

# Monte Carlo

Also called the **Monte Carlo method**. Answer a hard question by **trying it with random numbers many times and averaging**, instead of working it out exactly.

## 🟢 1. Simple (age 12)
Suppose you want to know how much of a square floor is covered by a rug, but the rug has a curvy shape. You could do hard maths. Or you could **throw 1,000 pebbles at random onto the floor and count how many land on the rug**. If 300 land on it, the rug covers about 30% of the floor. Monte Carlo is that: *use lots of random tries instead of hard calculation*.

**Example:** You want to know the chance of winning a card game. Instead of working out every possible deal, you play it 1,000 times and count the wins. If you win 120 times, the chance is about 12%.

## 🔵 2. How it works
1. Turn your question into an **average**: "what is the average of this thing when luck decides the inputs?" (see [[Expected Value]]).
2. Make **random inputs**, many of them.
3. For each random input, work out the thing (this is usually easy).
4. Average the results. That average is your answer.
5. **Why it works:** the [[Law of Large Numbers]] says the average of many random tries settles at the true expected value.
6. **How good is it?** The error shrinks like one over the square root of the number of tries. Ten times less error needs a hundred times more tries. That is slow, but it works.
7. **Its special strength:** the error shrinking this way does not depend on how many dimensions the problem has, while exact methods that cover the space with a grid get hopeless quickly (see the code for a 10-dimensional example).
8. **Where it shows up in AI:** estimating averages in [[Reinforcement Learning]] by playing many games, estimating the loss from a random mini-batch ([[Gradient Descent]]), sampling in [[Diffusion Models]], and drawing from hard distributions with a [[Markov Chain]] (called MCMC).

![[monte-carlo-pi.svg]]
*Estimating pi with random points. Points are scattered over the unit square. The red ones fall inside the quarter circle (1,100) and the blue ones fall outside (300), out of 1,400, so pi is about 4 × 1100 / 1400 = 3.1429. Image: Kmhkmh, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Pi_monte_carlo_all.svg), CC BY 4.0.*

## 🟠 3. Technical detail
- **The basic estimator:** to estimate $\mathbb{E}[f(X)]$, draw $x_1,\dots,x_N$ from the distribution of $X$ and average $f(x_i)$.
- **An integral is an average:** $\int_0^1 f(x)\,dx$ equals $\mathbb{E}[f(U)]$ for a uniform random $U$ between 0 and 1. So random points can compute integrals.
- **Probabilities are averages too:** the chance of an event is the expected value of a 0-or-1 indicator, so "count the hits and divide" is the same method (the pi code, the birthday code).
- **Error:** the standard error is $\sigma/\sqrt N$, where $\sigma^2$ is the [[Variance]] of one sample. To halve the error, use four times as many samples. Quadrupling the number of samples to halve the error, whatever the number of dimensions, is stated on the Wikipedia page.
- **Variance reduction:** cleverer sampling can lower $\sigma$ without more samples. Not covered here.
- **When you cannot sample directly:** MCMC builds a [[Markov Chain]] whose long-run behaviour follows the target distribution. Not covered here.
- **Weak spots:** rare events need huge numbers of samples to see even once, and the random number generator must be good.

```python
import math
import numpy as np
rng = np.random.RandomState(0)

# 1) pi by random points in the unit square: share inside the quarter circle ~ pi/4
for n in (100, 10_000, 1_000_000):
    p = rng.rand(n, 2); inside = ((p ** 2).sum(1) <= 1).sum()
    print(f"{n:9,d} points: inside {inside:,} -> pi ~ {4 * inside / n:.4f} (error {abs(4 * inside / n - math.pi):.4f})")

# 2) the typical error falls as 1/sqrt(n): 100x more points -> 10x less error
for n in (100, 10_000):
    est = [4 * ((rng.rand(n, 2) ** 2).sum(1) <= 1).mean() for _ in range(300)]
    print(f"n={n:6,d}: typical error over 300 runs {np.mean(np.abs(np.array(est) - math.pi)):.4f}")

# 3) integration with a fixed grid vs random points: volume of the unit ball in 10 dimensions
d = 10; exact = math.pi ** (d / 2) / math.gamma(d / 2 + 1)
N = 200_000; x = rng.uniform(-1, 1, size=(N, d))
est = 2 ** d * np.mean((x ** 2).sum(1) <= 1)
print(f"10-D ball volume: exact {exact:.4f} | random points ({N:,}) {est:.4f} | grid with 10 points per side would need {10 ** d:,} points")

# 4) count instead of calculate: chance that 2 of 23 people share a birthday
exact_b = 1 - np.prod([(365 - k) / 365 for k in range(23)])
sims = rng.randint(0, 365, size=(100_000, 23))
hit = np.mean([len(set(r)) < 23 for r in sims])
print(f"shared birthday among 23: exact {exact_b:.4f} | simulated {hit:.4f}")

# 5) the estimator by hand: integral of x^2 on [0,1] from three points
xs = np.array([0.2, 0.5, 0.9]); print("3-point estimate of the integral of x^2:", round((xs ** 2).mean(), 4), "| exact 1/3 =", round(1 / 3, 4))
```

What it printed: for pi, 100 points gave 3.3200 (error 0.1784), 10,000 gave 3.1516 (error 0.0100) and 1,000,000 gave 3.1409 (error 0.0007). Over 300 runs the typical error was 0.1265 at 100 points and 0.0136 at 10,000, a drop of about 9.3 times for 100 times more points, close to the 10 times the square-root rule predicts. In 10 dimensions, the unit ball's volume is exactly 2.5502, and 200,000 random points gave 2.5856, while a grid with 10 points per side would need 10,000,000,000 points. In a group of 23 people, the chance of a shared birthday is exactly 0.5073 and the simulation gave 0.5077. The 3-point estimate of the integral of $x^2$ was 0.3667, against the exact 0.3333.

## 🔴 4. Deep / formal
**The estimator.** To estimate $\mu=\mathbb{E}[f(X)]$ from $N$ independent random draws $x_i$ of $X$:

$$
\hat\mu_N=\frac{1}{N}\sum_{i=1}^{N}f(x_i)\;\approx\;\mathbb{E}[f(X)]
$$

**Where:**
- $x_i$ is the $i$-th random draw and $N$ is how many draws.
- $f(x_i)$ is the quantity you want averaged, worked out for that draw.
- $\hat\mu_N$ is the estimate. The hat means "an estimate, not the exact value".
- **Matters:** the draws must come from the right distribution and be independent. If they are not, the average is pulled to the wrong number or settles slowly.

**Example (tiny numbers):** estimate $\int_0^1x^2\,dx$ using $f(x)=x^2$ and three random points $x=0.2,\,0.5,\,0.9$.
- $f$ values: $0.2^2=0.04$, $0.5^2=0.25$, $0.9^2=0.81$.
- Average: $(0.04+0.25+0.81)/3=1.10/3\approx\mathbf{0.3667}$.
- The exact answer is $1/3\approx0.3333$. Meaning: three points are already in the right area, and the error would shrink with more points. The code printed the same 0.3667.

**The error.** The typical size of the error is the standard error:

$$
\mathrm{SE}(\hat\mu_N)=\frac{\sigma}{\sqrt N}
$$

**Where:**
- $\sigma$ is the standard deviation of a single $f(X)$ (the square root of its [[Variance]]).
- $N$ is the number of draws.
- **Matters:** it falls as $1/\sqrt N$, so the first samples help a lot and later ones help less. It does not depend on how many dimensions $X$ has, only on $\sigma$ (which can itself be large in hard problems).

**Example (tiny numbers):** the pi estimate. Each point is inside the quarter circle with chance $p=\pi/4\approx0.7854$, so one 0-or-1 indicator has $\sigma=\sqrt{p(1-p)}\approx0.4105$. The estimate is $4\times$ the share inside, so one sample has $\sigma\approx4\times0.4105=1.642$.
- $N=100$: $1.642/\sqrt{100}=\mathbf{0.164}$.
- $N=10{,}000$: $1.642/\sqrt{10{,}000}=\mathbf{0.0164}$.
- Meaning: 100 times more points gives 10 times less error. The code's typical absolute errors were 0.1265 and 0.0136, a bit smaller than these numbers, as expected: the average absolute error of a bell-shaped error is about 0.8 of its standard deviation (a standard fact, from memory).

**Further notes:**
- **Why it beats a grid in many dimensions:** a grid with $k$ points per side needs $k^d$ points. The Wikipedia page gives the example of 10 points per dimension needing $10^{100}$ points for 100 dimensions. The code's 10-dimensional case needs $10^{10}$ grid points versus 200,000 random ones.
- **Bell-shaped errors:** the error of a Monte Carlo average is roughly bell-shaped, by the [[Central Limit Theorem]] (not written up yet).
- **Not covered:** variance reduction, importance sampling, quasi-random numbers, MCMC in detail.

**What came before, and how this improved on it:**
- **Before:** work out probabilities and integrals exactly, or on a grid. Random sampling existed as a one-off idea too: the Wikipedia page mentions the Buffon's needle problem (estimating pi by dropping needles) and says Fermi tried the approach in the 1930s without publishing.
- **Its limit:** exact methods could not handle problems such as neutron diffusion in a weapon core, which the page says deterministic methods could not solve. A grid also explodes with dimensions.
- **What changed:** Stanisław Ulam thought of it in 1946 while wondering about the odds of winning a solitaire game, and proposed to "simply observe and count the number of successful plays" instead of calculating. He told von Neumann the same year, Nicholas Metropolis suggested the name "Monte Carlo" after the Monaco casino, and in spring 1948 the first fully automated runs were programmed on the ENIAC computer (Wikipedia page read, first 100,000 of about 207,300 characters; I did not read the originals, and I did not confirm a 1949 paper).
- **How much better:** in the code's 10-dimensional ball, 200,000 random points gave 2.5856 against the exact 2.5502 (1.4% off), where a 10-per-side grid needs 10,000,000,000 points. The cost is the slow $1/\sqrt N$ error: pi needed a million points to get within 0.0007.

## Related
- [[Expected Value]] · [[Law of Large Numbers]] · [[Variance]] · [[Probability]] · [[Gradient Descent]] · [[Reinforcement Learning]] · [[Diffusion Models]] · [[Markov Chain]] · [[Central Limit Theorem]]
- [[Bootstrap]]
## Sources
- Wikipedia, *Monte Carlo method* (first 100,000 of about 207,300 characters read, for the history block, the error statement and the grid example)
- Figure: [Pi_monte_carlo_all.svg](https://commons.wikimedia.org/wiki/File:Pi_monte_carlo_all.svg) (Kmhkmh), CC BY 4.0, Wikimedia Commons. I viewed it as a PNG thumbnail before writing the caption.
- The code and its numbers are my own, run in this session.

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/probability Monte Carlo:::Use lots of random tries instead of hard calculation. Throw random pebbles at a floor and count how many land on the rug to find its area.

#flashcards/math/formula #flashcards/topic/probability Write the Monte Carlo estimate and explain each symbol.
?
$$\hat\mu_N=\frac{1}{N}\sum_{i=1}^{N}f(x_i)\;\approx\;\mathbb{E}[f(X)]$$
- $x_i$: the $i$-th random draw. $N$: how many draws.
- $f(x_i)$: the quantity you want averaged, worked out for that draw.
- $\hat\mu_N$: the estimate. The hat means "an estimate, not the exact value".
- The draws must come from the right distribution and be independent. If not, the average is pulled to the wrong number or settles slowly.

#flashcards/math/formula #flashcards/topic/probability Write the Monte Carlo standard error and explain each symbol.
?
$$\mathrm{SE}(\hat\mu_N)=\frac{\sigma}{\sqrt N}$$
- $\sigma$: the standard deviation of a single $f(X)$ (square root of its variance).
- $N$: the number of draws.
- It falls as $1/\sqrt N$, so early samples help a lot and later ones help less. It does not depend on the number of dimensions, only on $\sigma$.

#flashcards/math/cloze #flashcards/topic/probability The Monte Carlo error falls as ==1/√N==, so 100 times more draws gives only ==10 times== smaller error.

#flashcards/math/example #flashcards/topic/probability Estimate $\int_0^1x^2\,dx$ using the random points 0.2, 0.5 and 0.9. What do you get?::f values 0.04, 0.25, 0.81. Average 1.10 / 3 ≈ 0.3667. The exact answer is 1/3 ≈ 0.3333, so three points are already in the right area.

#flashcards/math/example #flashcards/topic/probability When estimating pi, one sample has $\sigma\approx1.642$. What is the standard error for $N=100$ and $N=10{,}000$?::N = 100: 1.642 / 10 = 0.164. N = 10,000: 1.642 / 100 = 0.0164. 100 times more points gives a 10 times smaller error.

#flashcards/math/image #flashcards/topic/probability ![[monte-carlo-pi.svg]]
What does this picture show?
?
Estimating pi with random points over the unit square. The red ones fall inside the quarter circle (1,100) and the blue ones outside (300), out of 1,400, so pi ≈ 4 × 1100 / 1400 = 3.1429.
