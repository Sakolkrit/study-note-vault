---
tags: [statistics, ranking, preferences, Bradley-Terry, RLHF, probability]
status: growing
level: 4
tier: 2
---

# Bradley-Terry Model

A rule that turns "who beat whom" into a **strength score** for each player, and a **chance of winning** for any pair.

## 🟢 1. Simple (age 12)
Imagine a school chess club. You do not know who is best, but you have a list of games and who won. The Bradley-Terry model gives each player a **strength number** that explains the results: a stronger player is more likely to win, but a weaker player can still win sometimes. Once you have the numbers, you can guess the winner of a game nobody has played yet.

**Example:** Ana beats Ben in 7 of 10 games. Ben beats Cy in 6 of 10. The model finds three strength numbers that fit both results, and then says how likely Ana is to beat Cy, even if they never met.

## 🔵 2. How it works
1. Give every player (or answer) a hidden score $r$. A bigger $r$ means stronger.
2. For a match between $i$ and $j$, the chance that $i$ wins depends only on the **gap** $r_i-r_j$. A gap of 0 means 50/50, a big positive gap means $i$ almost always wins.
3. A squashing function ([[Logistic Regression]]'s sigmoid) turns the gap into a chance between 0 and 1.
4. Equivalent view: give each player a positive strength $s=e^{r}$. The chance that $i$ beats $j$ is $s_i/(s_i+s_j)$: your share of the combined strength.
5. To find the scores, choose the numbers that make the observed wins as likely as possible ([[Maximum Likelihood]]). A simple repeating update does this.
6. In language models, "players" are **answers**, and a human picking the better of two answers is a "match". This is how a [[Reward Model]] is trained, and the formula behind [[RLHF]] and [[Direct Preference Optimization]].

## 🟠 3. Technical detail
- **Origin:** Bradley and Terry, 1952, on paired comparisons (named in the lecture; I did not read it).
- **Two equal forms:** the sigmoid of the reward gap, and the strength ratio $s_i/(s_i+s_j)$ with $s=e^r$. The code checks they agree.
- **Only differences matter:** adding a constant to all scores changes nothing, so one score is fixed (the code fixes player 0 at 0).
- **Fitting:** maximise the chance of the observed results. The classical iteration updates each strength by (its wins) divided by the sum over opponents of (games played) divided by (own strength plus opponent's strength). The code runs it.
- **In RLHF:** the reward model is fit with the same loss, with scores from a neural network instead of one free number per player.
- **Assumptions:** one number per player, so no cycles like rock-paper-scissors; independent games; and a fixed gap always giving the same chance.
- **Relative:** Elo ratings in chess use a similar logistic idea (from memory; I did not check the details).

```python
import numpy as np
rng = np.random.RandomState(0)
sig = lambda z: 1 / (1 + np.exp(-z))
r_true = np.array([0.0, 1.0, 2.0, 3.0])                      # hidden rewards of 4 players
s_true = np.exp(r_true)                                      # the same as 'strengths'
# 1) the strength form and the sigmoid form of one match agree: s_i / (s_i + s_j) = sigmoid(r_i - r_j)
print('P(player 1 beats player 0): strengths', round(s_true[1] / (s_true[1] + s_true[0]), 4), '| sigmoid of reward gap', round(float(sig(1.0)), 4))
# 2) play 200 games for every pair
n = 4; wins = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        a = (rng.rand(200) < s_true[i] / (s_true[i] + s_true[j])).sum(); wins[i, j], wins[j, i] = a, 200 - a
print('wins matrix (row beat column):\n', wins.astype(int))
# 3) fit the strengths by the classic iteration s_i <- (wins of i) / sum_j games_ij / (s_i + s_j)
s = np.ones(n)
for _ in range(500):
    for i in range(n):
        s[i] = wins[i].sum() / sum(200 / (s[i] + s[j]) for j in range(n) if j != i)
    s /= s[0]
r = np.log(s)
print('fitted rewards (player 0 fixed at 0):', r.round(2).tolist(), '| true:', r_true.tolist())
print('predicted P(3 beats 2):', round(s[3] / (s[3] + s[2]), 3), '| observed:', wins[3, 2] / 200, '| true:', round(float(sig(1.0)), 3))
print('ranking recovered:', bool((np.argsort(-r) == np.argsort(-r_true)).all()))
```

## 🔴 4. Deep / formal
**The win chance.** For players $i$ and $j$ with scores $r_i$ and $r_j$:

$$
P(i\succ j)=\sigma(r_i-r_j)=\frac{e^{r_i}}{e^{r_i}+e^{r_j}}=\frac{s_i}{s_i+s_j}
$$

**Where:**
- $i\succ j$ means "$i$ beats $j$", or "answer $i$ is preferred to $j$".
- $r_i$ is the score of $i$ and $s_i=e^{r_i}$ is its strength.
- $\sigma(z)=1/(1+e^{-z})$ is the sigmoid.
- **Matters:** the chance depends only on the gap, so $r$ is only defined up to adding a constant. A gap of 0 gives 0.5, and each extra unit of gap multiplies the odds of winning by $e\approx2.718$.

**Example (tiny numbers):** $r_i=1$ and $r_j=0$.
- Strengths: $s_i=e^1=2.718$, $s_j=e^0=1$.
- $P(i\succ j)=2.718/(2.718+1)=\mathbf{0.731}$.
- Same answer from the sigmoid: $1/(1+e^{-1})=\mathbf{0.731}$.
- Meaning: a player one unit stronger wins about 73% of the games.

**Further notes:**
- **What the toy found:** four players with true rewards $0,1,2,3$ played 200 games against each other. The fitted rewards were $[0,0.99,1.96,2.95]$, the ranking was recovered, and the model's predicted chance that player 3 beats player 2 was $0.728$, against $0.71$ observed and $0.731$ true.
- **Why it works with few games:** the strengths use all the results at once, so even pairs that played few games get a prediction.
- **Not covered:** ties, home advantage, players that change over time, and the statistics of the fit's uncertainty.

**What came before, and how this improved on it:**
- **Before:** rank by raw win counts or win rates (from memory; not a cited method).
- **Its limit:** a win against a weak player counts the same as a win against a strong one, and players who met different opponents cannot be compared.
- **What changed:** model each match as a chance that depends on the strength gap, so every result is weighed by who the opponent was (Bradley and Terry, 1952; I did not read the paper).
- **How much better:** I did not compare against win counts in code. In this note's toy, the fit recovered the true rewards to within $0.05$ with 200 games per pair, and its predicted win chance for players 3 and 2 ($0.728$) differed from the true one ($0.731$) by $0.003$.

## Related
- [[Reward Model]] · [[RLHF]] · [[Direct Preference Optimization]] · [[Logistic Regression]] · [[Maximum Likelihood]] · [[Probability]] · [[Softmax]] · [[Evaluation Metrics]] · [[Chatbot Arena]]

## Sources
- Stanford CME 295 (Fall 2025), Lecture 5 slides, Afshine and Shervine Amidi
- Bradley and Terry, *The rank analysis of incomplete block designs: I. The method of paired comparisons*, Biometrika, 1952 (named in the lecture; not read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/alignment-rl Bradley-Terry model:::Give each player (or answer) a strength number that explains who beats whom. A stronger one is more likely to win, but a weaker one can still win sometimes.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the Bradley-Terry win chance and explain each symbol.
?
$$P(i\succ j)=\sigma(r_i-r_j)=\frac{e^{r_i}}{e^{r_i}+e^{r_j}}=\frac{s_i}{s_i+s_j}$$
- $i\succ j$: "$i$ beats $j$", or "answer $i$ is preferred to $j$".
- $r_i$: the score of $i$. $s_i=e^{r_i}$: its strength.
- $\sigma(z)=1/(1+e^{-z})$: the sigmoid.
- The chance depends only on the gap, so $r$ is defined only up to adding a constant. A gap of 0 gives 0.5, and each extra unit of gap multiplies the odds of winning by $e\approx2.718$.

#flashcards/ml/example #flashcards/topic/alignment-rl $r_i=1$, $r_j=0$. What is the chance $i$ beats $j$?::$s_i=e=2.718$, $s_j=1$. $2.718/(2.718+1)=0.731$. The sigmoid gives the same: $1/(1+e^{-1})=0.731$. One unit stronger wins about 73% of games.

#flashcards/ml/example #flashcards/topic/alignment-rl Why can a score $r$ only be known up to a constant?::The win chance depends only on the gap $r_i-r_j$, so adding the same number to every score changes nothing.
