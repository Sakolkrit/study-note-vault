---
tags: [math, probability, statistics, expectation, foundations]
status: growing
level: 4
tier: 1
---

# Expected Value

Also called the **mean**, **average** or **expectation**, written $\mathbb{E}[X]$. The number a random thing settles around if you repeat it many times.

## 🟢 1. Simple (age 12)
The expected value is the **average result you would get if you could play the same game again and again**. It does not have to be a result you can actually get. A normal die never shows 3.5, but if you roll it many times, the average of your rolls gets close to 3.5.

**Example:** A school fair game costs nothing to play. You win 10 coins one time in five, and lose 1 coin the other four times. Over 5 plays you expect to win 10 once and lose 1 four times, so 10 − 4 = 6 coins, or **1.2 coins per play** on average. It is a good game to play, even though most single plays lose.

## 🔵 2. How it works
1. List every result the random thing can give, and how likely each one is (see [[Probability]]).
2. Multiply each result by its chance.
3. Add them all up. That sum is the expected value.
4. Think of it as a **weighted average**: results that are more likely count more.
5. **Why it matters:** if you repeat the thing many times, the plain average of what you see gets close to the expected value (the [[Law of Large Numbers]]). The code below shows it for dice.
6. **A very handy trick:** the expected value of a sum is the sum of the expected values. A pile of 10 dice has an expected total of 10 × 3.5, and you never have to list the millions of ways the dice can land. This still holds if the dice are linked to each other.
7. **Warning:** the average hides the spread. Two games can have the same expected value and feel very different, one steady and one wild. For that you also need the [[Variance]].
8. **Where it shows up in AI:** a [[Loss Function]] is an expected value (the average error over all the data), and training estimates it from a small batch. In [[Reinforcement Learning]] the "value" of a state is an expected total reward ([[Bellman Equation]]).

![[law-of-large-numbers-dice.jpg]]
*The running average of the sum of two dice, plotted against the number of rolls. It starts low and climbs towards the line at 7, the expected value of the sum, as the rolls pile up. Image: Hihihi12345678, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:The_law_of_large_numbers_example_graph.jpg), CC0.*

## 🟠 3. Technical detail
- **Discrete case:** $\mathbb{E}[X]=\sum_x x\,p(x)$. **Continuous case:** replace the sum by an integral of $x\,f(x)$ (for example for the [[Normal Distribution]], the expected value is its centre).
- **Function of a random value:** $\mathbb{E}[g(X)]=\sum_x g(x)\,p(x)$. You do not need the distribution of $g(X)$ first. Squared error and [[Cross-Entropy Loss]] are of this type.
- **Linearity:** $\mathbb{E}[aX+bY]=a\,\mathbb{E}[X]+b\,\mathbb{E}[Y]$ always, independent or not. This is the most used property.
- **Not always defined:** some heavy-tailed random things have no finite expected value. A mean you compute from data still exists, but it will not settle down.
- **Estimating it:** the sample mean is the usual estimate. Replacing an expectation by an average over random draws is the idea behind [[Monte Carlo]] methods and behind mini-batch [[Gradient Descent]].
- **Not the same as "most likely":** the expected value can be a result that never happens (3.5 on a die), and it can be pulled far away from the typical result by rare, large values.

```python
import itertools
import numpy as np

# 1) expected value = sum of value x chance
die = {k: 1 / 6 for k in range(1, 7)}
ev = sum(x * p for x, p in die.items())
print("one die:", round(ev, 4))
game = {10: 0.2, -1: 0.8}                                  # win 10 with chance 0.2, else lose 1
print("game (win 10 at 0.2, lose 1 at 0.8):", round(sum(x * p for x, p in game.items()), 4))

# 2) linearity: E[X+Y] = E[X] + E[Y], even if Y depends on X (Y = X)
two = sum((a + b) / 36 for a, b in itertools.product(range(1, 7), repeat=2))
same = sum((a + a) / 6 for a in range(1, 7))
print("two independent dice:", round(two, 4), "| the same die counted twice:", round(same, 4))

# 3) the average of many rolls drifts to the expected value
rng = np.random.RandomState(0)
for n in (10, 100, 10000):
    print(f"mean of {n:5d} rolls: {rng.randint(1, 7, n).mean():.3f}")

# 4) a mini-batch average is an estimate of an expectation; how noisy is it?
pop = rng.randn(100000) * 2 + 3                            # per-example 'loss', true mean 3, spread 2
for b in (1, 16, 256):
    means = [rng.choice(pop, b).mean() for _ in range(2000)]
    print(f"batch {b:3d}: spread of the estimate {np.std(means):.3f}")

# 5) linearity vs listing every outcome: sum of 6 dice
total = sum(sum(p) for p in itertools.product(range(1, 7), repeat=6)) / 6**6
print("sum of 6 dice by listing", 6**6, "outcomes:", round(total, 4), "| by linearity (6 x 3.5):", 6 * 3.5)
print("for 10 dice listing would need", f"{6**10:,}", "outcomes; linearity still needs 10 additions")
```

What it printed: one die 3.5, the game 1.2, and 7.0 for both the two-dice sum and the "same die twice" case. The mean of 10 rolls was 3.900, of 100 rolls 3.390 and of 10,000 rolls 3.493. For batches, the spread of the estimate was 1.954 for batch size 1, 0.500 for 16 and 0.125 for 256: each 16 times larger batch cut the noise by about 4 times (my observation here, a square-root pattern I did not derive in this note).

## 🔴 4. Deep / formal
**The definition (discrete case).** For a random value $X$ that takes value $x$ with chance $p(x)$:

$$
\mathbb{E}[X]=\sum_{x}x\,p(x)
$$

**Where:**
- $X$ is the random value (the die roll, the game payout).
- $x$ runs over every value $X$ can take. $p(x)$ is the chance of that value, and the chances add up to 1.
- $x\,p(x)$ is the value times its chance. **Matters:** this is what weights likely results more than rare ones. If all chances were treated the same, you would get the plain average of the list of values, which is only correct when every value is equally likely.
- $\mathbb{E}[X]$ is the result: the long-run average, not a promise about any single trial.

**Example (tiny numbers):** the school-fair game. $X=10$ with chance $0.2$ and $X=-1$ with chance $0.8$.
- $10\times0.2=2$.
- $(-1)\times0.8=-0.8$.
- $\mathbb{E}[X]=2-0.8=\mathbf{1.2}$.
- Meaning: on average you gain 1.2 coins per play, even though you lose on 4 plays out of 5.

**Linearity.** For random values $X$ and $Y$ and fixed numbers $a$ and $b$:

$$
\mathbb{E}[aX+bY]=a\,\mathbb{E}[X]+b\,\mathbb{E}[Y]
$$

**Where:**
- $X$ and $Y$ are any two random values, they may be linked.
- $a$ and $b$ are fixed numbers (not random).
- **Matters:** it holds with **no** need for independence, which is why it is so useful. The matching rule for spread (the [[Variance]]) does need independence. If you wrongly believed linearity also needed it, you would take the long route of listing every pair of results.

**Example (tiny numbers):** two dice, $a=b=1$, each with $\mathbb{E}=3.5$.
- Independent dice: $3.5+3.5=\mathbf{7}$. The code, listing all 36 pairs, gives $7.0$.
- Same die counted twice ($Y=X$, as linked as it gets): $\mathbb{E}[X+X]=2\times3.5=\mathbf{7}$. The code, listing 6 outcomes, gives $7.0$.
- Meaning: linking the dice does not change the expected total, only how spread out the total is.

**Further notes:**
- **Law of large numbers:** the sample mean settles at $\mathbb{E}[X]$ as the number of trials grows. The dice code shows it by simulation (3.900, 3.390, 3.493), not by proof. The statement is the standard result, flagged as from memory.
- **A mini-batch is a noisy estimate:** in the code, the estimate's spread dropped from 1.954 to 0.500 to 0.125 as batches grew 1 → 16 → 256. Bigger batches give a steadier estimate of the loss, at more cost per step.
- **Not covered:** expected values of continuous random values with integrals, conditional expectation, Jensen's inequality, and cases with no finite mean.

**What came before, and how this improved on it:**
- **Before:** the problem of points, how to fairly divide the stakes of a game stopped early, debated for centuries according to the Wikipedia page (read).
- **Its limit:** no rule that gave the same fair answer for every game. The page says Chevalier de Méré gave the problem to Pascal in 1654 and claimed it could not be solved.
- **What changed:** Pascal and Fermat, writing to each other in 1654, found solutions based on the principle that a future gain is worth in proportion to its chance. Huygens published the idea in 1657 in *De ratiociniis in ludo aleæ*, described by the page as the first successful attempt to lay the foundations of probability theory. The word "expectation" in the modern sense is later: the page credits Laplace (1814) with "mathematical hope" and Whitworth (1901) with the letter E. (Wikipedia page read; I did not read the originals.)
- **How much better:** in the code, the expected total of 6 dice came out 21.0 by listing all 46,656 outcomes and 21.0 by linearity, which needs 6 multiplications and one addition. For 10 dice, listing would need 60,466,176 outcomes while linearity still needs 10 additions.

## Related
- [[Probability]] · [[Normal Distribution]] · [[Loss Function]] · [[Cross-Entropy Loss]] · [[Gradient Descent]] · [[Fisher Information]] · [[Bellman Equation]] · [[Markov Decision Process]] · [[Reinforcement Learning]] · [[Variance]] · [[Law of Large Numbers]] · [[Monte Carlo]]

## Sources
- Wikipedia, *Expected value* (page read for the history block and for the linearity statement)
- Figure: [The_law_of_large_numbers_example_graph.jpg](https://commons.wikimedia.org/wiki/File:The_law_of_large_numbers_example_graph.jpg) (Hihihi12345678), CC0, Wikimedia Commons. I looked at the image before writing the caption.
- The code and its numbers are my own, run in this session.

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/probability Expected value:::The average result you would get if you could play the same game again and again. It does not have to be a result you can actually get (a die never shows 3.5).

#flashcards/math/formula #flashcards/topic/probability Write the expected value formula and explain each symbol.
?
$$\mathbb{E}[X]=\sum_{x}x\,p(x)$$
- $X$: the random value (the die roll, the game payout).
- $x$: runs over every value $X$ can take. $p(x)$: the chance of that value. The chances add up to 1.
- $x\,p(x)$: value times its chance. This weights likely results more than rare ones.
- $\mathbb{E}[X]$: the long-run average, not a promise about any single trial.

#flashcards/math/formula #flashcards/topic/probability Write the linearity of expectation and explain each symbol.
?
$$\mathbb{E}[aX+bY]=a\,\mathbb{E}[X]+b\,\mathbb{E}[Y]$$
- $X$, $Y$: any two random values. They may be linked.
- $a$, $b$: fixed numbers (not random).
- It holds with no need for independence, which is why it is so useful. The matching rule for variance does need independence.

#flashcards/math/cloze #flashcards/topic/probability Linearity of expectation needs ==no== independence, but the matching rule for ==variance== does.

#flashcards/math/example #flashcards/topic/probability A game: win 10 coins with chance 0.2, lose 1 coin with chance 0.8. What is the expected value?::10 × 0.2 = 2, and (−1) × 0.8 = −0.8. E[X] = 1.2 coins per play, even though you lose 4 plays out of 5.

#flashcards/math/concept #flashcards/topic/probability What is the expected value of the sum of two dice, and does it change if both dice are the same die?::3.5 + 3.5 = 7 for independent dice. Same die counted twice: E[X + X] = 2 × 3.5 = 7. Linking them does not change the expected value.

#flashcards/math/image #flashcards/topic/probability ![[law-of-large-numbers-dice.jpg]]
What does this picture show?
?
The running average of the sum of two dice against the number of rolls. It starts off and climbs towards the line at 7, the expected value of the sum, as the rolls pile up.
