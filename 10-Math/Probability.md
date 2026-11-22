---
tags: [probability, statistics, math, machine-learning, LLM]
status: growing
level: 4
tier: 1
---

# Probability

## 🟢 1. Simple (age 12)
Probability is a number from **0 to 1** that says **how likely something is**. 0 means it can never happen, 1 means it always happens, and 0.5 means it is a coin flip. It lets us talk about "probably" with numbers instead of guesses.

**Example:** A bag has 3 red sweets and 1 blue sweet. You pick one without looking. There are 4 sweets and 3 are red, so the chance of red is $3/4=0.75$ (75%), and the chance of blue is $1/4=0.25$. Those two always add up to 1, because something has to be picked.

## 🔵 2. How it works
1. List every **outcome** that could happen (a die can land on 1, 2, 3, 4, 5 or 6).
2. Give each outcome a chance between 0 and 1. All the chances must **add up to 1**.
3. For an **event** (a group of outcomes, like "an even number"), add up the chances of its outcomes. Even on a die: $3/6=0.5$.
4. **Independent events** do not affect each other (two coin flips). To get the chance that both happen, **multiply**: heads then heads is $0.5\times0.5=0.25$.
5. **Conditional chance** is the chance of something once you already know something else ("it is cloudy, so how likely is rain?"). Learning new facts changes the chances. This is the idea behind **Bayes' rule**, below.
6. The **expected value** is the average result you would get if you could repeat the thing many, many times.

Where it shows up in AI: an [[LLM]] never "knows" the next word. It gives every possible word a probability (using [[Softmax]]) and one is picked. Training uses [[Loss Function]]s that measure how much probability the model gave to the right answer.

![[dice-sum-distribution.svg]]
*A probability distribution: the chance of each total $S$ when you roll two dice. There are 36 equally likely pairs, and 6 of them add up to 7, so $p(7)=6/36=1/6$, the tallest bar. Totals of 2 and 12 have only one pair each, $1/36$. Image: Tim Stellmach, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Dice_Distribution_(bar).svg), public domain.*

## 🟠 3. Technical detail
- **Sample space and events:** the set of all outcomes, and subsets of it. $P(\text{event})$ is between 0 and 1, and $P(\text{whole space})=1$.
- **Random variable:** a number that depends on chance (the total of two dice, the next token's ID). **Distribution:** the list of how likely each value is.
- **Discrete vs continuous:** discrete values (dice, words) get a chance each. Continuous values (heights, temperatures) use a **density**, and chances come from the area under a curve (see [[Calculus]]).
- **Rules you will use constantly:**
  - Not A: $P(\text{not }A)=1-P(A)$.
  - A or B (cannot both happen): add. A and B (independent): multiply.
  - Conditional: $P(A\mid B)=P(A\cap B)/P(B)$, read "chance of A given B".
- **Common distributions:** uniform (all equal), Bernoulli (one yes/no), binomial (count of yeses), normal or Gaussian (the bell curve, which appears whenever many small random things add up).
- **Mean and variance:** the expected value $\mathbb{E}[X]$ is the centre, and variance measures how spread out the values are.
- **Law of large numbers:** repeat an experiment many times and the average settles close to the expected value.
- **In ML:** models output distributions; training makes the real data likely under the model (maximum likelihood), which gives the cross-entropy loss in [[Loss Function]].

```python
from itertools import product
rolls = [a + b for a, b in product(range(1, 7), repeat=2)]   # all 36 pairs
print(round(rolls.count(7) / 36, 4))     # 0.1667  chance of total 7
print(sum(rolls) / 36)                   # 7.0     expected total

pd, se, fp = 0.01, 0.9, 0.05             # Bayes example from below
print(round(se * pd / (se * pd + fp * (1 - pd)), 3))   # 0.154
```

## 🔴 4. Deep / formal
**Expected value.** For a random variable $X$ that takes value $x_i$ with probability $p_i$:

$$
\mathbb{E}[X]=\sum_{i} x_i\,p_i
$$

**Where:**
- $X$ is the random quantity (for example your winnings in a game). $\mathbb{E}[X]$ is its expected value, the long-run average.
- $x_i$ is the $i$-th possible value $X$ can take.
- $p_i$ is the probability of that value. **Matters:** the probabilities must add up to 1, otherwise the sum is not an average.
- $\sum_i$ adds over every possible value. **Matters:** each value is **weighted by how likely it is**, so rare outcomes count a little and common outcomes count a lot.

**Example (tiny numbers):** a game: win $10$ with chance $0.2$, lose $1$ with chance $0.8$.
- Values and chances: $(x_1,p_1)=(10,\,0.2)$ and $(x_2,p_2)=(-1,\,0.8)$. The chances add to $1$.
- $\mathbb{E}[X]=10\cdot0.2+(-1)\cdot0.8=2-0.8=\mathbf{1.2}$.
- Meaning: you lose most rounds, but on average you gain $1.2$ per round over many rounds, so the game is worth playing. (For one fair die: $\tfrac16(1+2+\dots+6)=3.5$, a value the die can never actually show. The expected value is an average, not a promise.)

**Bayes' rule.** How to update a belief when you see new evidence. It comes from the fact that "A and B" can be built in two ways: $P(A\mid B)P(B)=P(A\cap B)=P(B\mid A)P(A)$ (see the diagram below). Dividing by $P(B)$:

$$
P(A\mid B)=\frac{P(B\mid A)\,P(A)}{P(B)}
$$

![[bayes-theorem.svg]]
*How the two conditional chances relate: every path between $A$ and $B$ is a probability, and the formula at the bottom, $P(A\mid B)\cdot P(B)=P(A\cap B)=P(B\mid A)\cdot P(A)$, says the two routes to "A and B" agree. Bayes' rule is that equation rearranged. Image: Qniemiec, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Bayes%27_Theorem_2D.svg), CC BY-SA 3.0.*

**Where:**
- $A$ is the thing you want to know about (for example "has the disease"). $B$ is the evidence you saw (for example "the test came back positive").
- $P(A)$ is the **prior**: how likely $A$ was before seeing the evidence. **Matters:** if $A$ is very rare, a positive sign alone is weak evidence.
- $P(B\mid A)$ is the **likelihood**: how often you would see the evidence if $A$ were true.
- $P(B)$ is the overall chance of seeing the evidence at all, whether or not $A$ is true. **Matters:** it is the divider that makes the answer a proper probability. It is found by adding up all the ways the evidence can appear: $P(B)=P(B\mid A)P(A)+P(B\mid\text{not }A)P(\text{not }A)$.
- $P(A\mid B)$ is the **posterior**: your updated belief after the evidence.

**Example (tiny numbers):** a disease affects $1\%$ of people: $P(A)=0.01$. A test catches it $90\%$ of the time: $P(B\mid A)=0.9$. It wrongly says positive for $5\%$ of healthy people: $P(B\mid\text{not }A)=0.05$.
- Sick and positive: $0.9\cdot0.01=0.009$.
- Healthy but positive: $0.05\cdot0.99=0.0495$.
- Total chance of a positive: $P(B)=0.009+0.0495=0.0585$.
- $P(A\mid B)=0.009/0.0585=\mathbf{0.154}$, about $15\%$.
- Meaning: even after a positive result, you are probably healthy (85%), because healthy people are so much more common that their false alarms outnumber the real cases. People often guess $90\%$ here and are surprised. The prior matters.

**Further notes:**
- **Joint, marginal, conditional:** $P(A,B)$ is the chance of both, and the chain rule $P(A,B,C)=P(A)\,P(B\mid A)\,P(C\mid A,B)$ is exactly how an [[LLM]] scores a sentence: the chance of each token given the ones before it ([[Tokenization]]).
- **Independence is rare in practice:** words in a sentence are not independent, which is why models need [[Attention]].
- **Two meanings of probability:** *frequentist* (long-run frequency) and *Bayesian* (degree of belief, updated with Bayes' rule). Both are used in machine learning.
- **Calibration:** a model that says "80% sure" should be right about 80% of the time. Modern networks are often over-confident.
- **Central limit theorem:** averages of many independent random values look bell-shaped regardless of the original distribution (stated from memory, with conditions). Full note: [[Central Limit Theorem]].

**What came before, and how this improved on it:**
- **Before:** gut feeling about chance.
- **Its limit:** gut feeling is often badly wrong when events combine. The birthday problem is the classic case.
- **What changed:** rules for combining chances (multiply for "and", add for "or"), first worked out in the 1600s in letters between Pascal and Fermat, and put on firm axioms by Kolmogorov in 1933 (both from memory, not checked).
- **How much better:** in the code below, for 23 people the gut-feel rule (23 days out of 365) says 0.063 and the rules give 0.507, which a simulation of 100,000 groups confirms at 0.507. The gut feel is 8 times too low.

**Check it yourself:**

```python
import numpy as np
n = 23
exact = 1 - np.prod([(365 - i) / 365 for i in range(n)])
rng = np.random.default_rng(0)
sim = np.mean([len(set(rng.integers(0, 365, n))) < n for _ in range(100000)])
print(f"{n} people, chance that two share a birthday:")
print("  gut-feel rule (23 out of 365 days):", round(n / 365, 3))
print("  exact, multiplying the chances of no match:", round(exact, 3))
print("  simulation of 100,000 groups:", round(sim, 3))
```

## Related
- [[Softmax]] · [[Loss Function]] · [[LLM]] · [[Tokenization]] · [[Attention]] · [[Neural Network]] · [[Calculus]] · [[Vector]] · [[Bayesian Methods]] · [[Entropy]] · [[KL Divergence]] · [[Maximum Likelihood]] · [[Bayes' Theorem]] · [[Q-Learning]] · [[Naive Bayes]] · [[Matplotlib]] · [[Hash Tables]]

## Sources
- Blitzstein & Hwang, *Introduction to Probability* (free online book and Harvard Stat 110 lectures)
- 3Blue1Brown, *Bayes' theorem* (video)
- Goodfellow, Bengio & Courville, *Deep Learning*, 2016, chapter 3

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/probability Probability:::A number from 0 to 1 that says how likely something is. 0 means it never happens, 1 means it always happens, 0.5 is a coin flip.

#flashcards/math/formula #flashcards/topic/probability Write the expected value in terms of probabilities and explain each symbol.
?
$$\mathbb{E}[X]=\sum_{i} x_i\,p_i$$
- $X$: the random quantity (for example your winnings in a game). $\mathbb{E}[X]$: its expected value, the long-run average.
- $x_i$: the $i$-th possible value $X$ can take.
- $p_i$: the probability of that value. The probabilities must add up to 1, otherwise the sum is not an average.
- $\sum_i$: add over every possible value. Each value is weighted by how likely it is, so rare outcomes count a little and common ones count a lot.

#flashcards/math/example #flashcards/topic/probability A bag has 3 red sweets and 1 blue sweet. What are the chances of red and of blue?::Red 3/4 = 0.75. Blue 1/4 = 0.25. The two chances add up to 1.

#flashcards/math/example #flashcards/topic/probability A game: win 10 with chance 0.2, lose 1 with chance 0.8. What is the expected value per round?::10 × 0.2 + (−1) × 0.8 = 2 − 0.8 = 1.2. You lose most rounds, but gain 1.2 per round on average.

#flashcards/math/cloze #flashcards/topic/probability Probabilities of all possible outcomes must add up to ==1==.

#flashcards/math/image #flashcards/topic/probability ![[dice-sum-distribution.svg]]
What does this picture show?
?
The chance of each total when you roll two dice. 36 equally likely pairs, 6 of them add up to 7, so p(7) = 6/36 = 1/6, the tallest bar. Totals 2 and 12 have one pair each (1/36).

#flashcards/math/image #flashcards/topic/probability ![[bayes-theorem.svg]]
What does this picture show?
?
How the two conditional chances relate. Every path between A and B is a probability, and the formula P(A|B)·P(B) = P(A∩B) = P(B|A)·P(A) says the two routes to "A and B" agree. This is the root of Bayes' theorem.

#flashcards/math/formula #flashcards/topic/probability Write Bayes' rule and explain each symbol.
?
$$P(A\mid B)=\frac{P(B\mid A)\,P(A)}{P(B)}$$
- $A$: the thing you want to know about (for example "has the disease"). $B$: the evidence you saw (for example "the test came back positive").
- $P(A)$: the prior, how likely $A$ was before the evidence. If $A$ is very rare, a positive sign alone is weak evidence.
- $P(B\mid A)$: the likelihood, how often you would see the evidence if $A$ were true.
- $P(B)$: the overall chance of the evidence, whether or not $A$ is true. It is the divider that makes the answer a proper probability, found by adding all the ways the evidence can appear: $P(B)=P(B\mid A)P(A)+P(B\mid\text{not }A)P(\text{not }A)$.
- $P(A\mid B)$: the posterior, your updated belief after the evidence.

#flashcards/math/example #flashcards/topic/probability A disease affects 1% of people, the test catches it 90% of the time and wrongly says positive for 5% of healthy people. What is the chance of being sick after a positive test?::Sick and positive: $0.9\cdot0.01=0.009$. Healthy but positive: $0.05\cdot0.99=0.0495$. $P(B)=0.0585$. $P(A\mid B)=0.009/0.0585=0.154$, about 15%. You are probably healthy (85%), because healthy people are so much more common that their false alarms outnumber the real cases. The prior matters.
