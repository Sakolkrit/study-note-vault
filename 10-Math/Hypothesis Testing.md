---
tags: [hypothesis-testing, p-value, permutation-test, significance, statistics, probability, evaluation, math]
status: growing
level: 4
tier: 2
---

# Hypothesis Testing

Hypothesis testing is **a way to ask whether a difference you see in your data is bigger than what plain luck could produce**. You assume that nothing real is going on (the *null hypothesis*), work out how often luck alone would give a difference at least as big as yours, and if that is rare you say the difference is probably real.

## 🟢 1. Simple (age 12)
Your friend says their coin is magic and always lands on heads. You flip it 3 times: heads, heads, heads. Is it magic? A normal coin does that 1 time in 8, which happens often enough. You flip it 20 times and get 20 heads: a normal coin does that about 1 time in a million. Now "luck" is a poor explanation. A **hypothesis test** puts a number on "how surprising is this if the coin were normal?"

**Example:** two apps are tried on 12 people who liked one app and not the other: 9 liked app A only and 3 liked app B only. Is A really better? A test asks how often a split this lopsided would appear if both apps were equally good.

## 🔵 2. How it works
1. **Pick a measure of the difference**, such as "average of group A minus average of group B", or "how many people hit with method X only, against Y only".
2. **Assume nothing real is going on** (the null hypothesis, $H_0$): the labels "A" and "B" make no difference.
3. **Ask how often luck alone gives a difference at least as big.** You can count it exactly (list every way the labels could have fallen), or simulate it.
4. **That share is the p-value.** A small p-value means "this would be rare if nothing were going on".
5. **Compare it with a cut-off you chose in advance**, often 0.05. Below it you say the difference is "statistically significant".
6. **Be careful:** run many tests and some will pass by luck; a p-value is not the chance that you are right; and "not significant" does not mean "no difference".

![[p-value-significance-testing.svg]]
*A p-value is the shaded green area: the probability of seeing the observed result, or one more extreme, if the null hypothesis is true. The boxed warning at the top says that "probability of the observation given the hypothesis" is not "probability of the hypothesis given the observation". Author: Repapetilto, CC BY-SA 3.0, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:P-value_in_statistical_significance_testing.svg).*

## 🟠 3. Technical detail
- **The p-value.** Let $T$ be the measure of the difference (the *test statistic*), $t_{\text{obs}}$ the value you actually saw, and $H_0$ the null hypothesis. Then

$$
p=\Pr\big(\,|T|\ge|t_{\text{obs}}|\;\big|\;H_0\,\big)
$$

**Where:**
- $T$ is the difference measured on a data set; $t_{\text{obs}}$ is its value on your real data.
- $H_0$ is "nothing real is going on", for example that the group labels are interchangeable.
- The vertical bar reads "given that"; the absolute values make the test **two-sided** (a difference in either direction counts).
- **Matters:** a small $p$ says "data like mine would be rare if $H_0$ were true". It does **not** say that $H_0$ has probability $p$ of being true, and it says nothing about how big or important the difference is.

**Example (tiny numbers):** a permutation test with two groups of three: $A=5,7,8$ and $B=2,3,4$ (made up).
- Observed difference of the means: $6.667-3=3.667$.
- If the labels meant nothing, any 3 of the 6 values could have been "A". There are $\binom63=20$ ways to pick them.
- Only 2 of the 20 splits give a difference as large as $3.667$ in either direction (A takes the top three, or the bottom three).
- $p=2/20=\mathbf{0.10}$.
- Meaning: with 3 + 3 values even this perfectly separated result is not below 0.05. The smallest possible p-value here is 0.10, so the test cannot say "significant" however clean the data look.

- **A test for two methods tried on the same people.** Some people hit with only method X ($b$ of them), some with only method Y ($c$ of them); the people who hit with both or neither say nothing. If the methods were equally good, each of the $n=b+c$ "only one hit" people is equally likely to belong to X or Y, like a fair coin. The exact two-sided p-value is

$$
p=\min\!\Bigg(1,\;2\sum_{k=\max(b,c)}^{n}\binom{n}{k}\frac{1}{2^{n}}\Bigg),\qquad n=b+c
$$

**Where:**
- $b$ and $c$ are the counts of people who hit with only X, and with only Y. $n=b+c$.
- $\binom{n}{k}$ is the number of ways to choose $k$ of the $n$ people; $2^n$ is the number of equally likely outcomes of $n$ fair coin flips.
- The sum adds up the chance of a split at least as lopsided as the one seen, in one direction; the $2$ makes it two-sided; $\min(1,\cdot)$ keeps it a probability.
- **Matters:** it uses only the people where the methods differ, and it needs no assumption about bell shapes. It is called the exact sign test (or the exact McNemar test, from memory).

**Example (tiny numbers):** $b=9$, $c=3$, so $n=12$ (made up; my arithmetic).
- Splits at least as lopsided: $k=9,10,11,12$ gives $\binom{12}{9}+\binom{12}{10}+\binom{12}{11}+\binom{12}{12}=220+66+12+1=299$.
- One side: $299/4096=0.073$. Two sides: $2\times0.073=\mathbf{0.146}$.
- Meaning: 9 against 3 sounds lopsided, yet it would appear 14.6% of the time between two equal methods. Not significant.

- **Many tests.** If you run $m$ separate tests, each at cut-off $\alpha$, and nothing real is going on, the chance that at least one passes by luck is

$$
1-(1-\alpha)^{m}
$$

**Where:**
- $\alpha$ is the cut-off for one test (often 0.05); $m$ is the number of tests.
- $(1-\alpha)^m$ is the chance that all $m$ tests stay quiet (assuming they are independent).
- **Matters:** with many tests, a "significant" one is expected from luck alone. The simplest repair, the **Bonferroni correction**, tests each at $\alpha/m$ (the same correction is used in [[Data Drift]]).

**Example (tiny numbers):** $m=20$, $\alpha=0.05$ (my arithmetic).
- $1-0.95^{20}=1-0.358=\mathbf{0.642}$.
- Bonferroni: each test at $0.05/20=0.0025$.
- Meaning: out of 20 comparisons with nothing real going on, you would see at least one "significant" result in about 64% of experiments.

- **Code: the three tests, and what luck does.** Part 1 lists all 20 splits of the six numbers. Part 2 is the exact sign test of the example. Part 3 gives two groups of 30 skewed numbers from the *same* distribution (nothing real) 20,000 times and counts how often a z-test says "significant". Part 4 repeats this for 20 separate comparisons per experiment:

```python
# file: toy_test.py
import itertools
import numpy as np
from math import comb

a, b = np.array([5.0, 7.0, 8.0]), np.array([2.0, 3.0, 4.0])
allv = np.concatenate([a, b]); obs = a.mean() - b.mean()
diffs = [allv[list(i)].mean() - np.delete(allv, list(i)).mean() for i in itertools.combinations(range(6), 3)]
print(f"1) permutation test: observed difference {obs:.3f}; {len(diffs)} ways to split the 6 values into 3 + 3;",
      f"{sum(abs(d) >= abs(obs) - 1e-12 for d in diffs)} of them are at least this extreme -> p = {sum(abs(d) >= abs(obs) - 1e-12 for d in diffs) / len(diffs):.2f}")

hit_x_only, hit_y_only = 9, 3
n = hit_x_only + hit_y_only
p = min(1.0, 2 * sum(comb(n, k) for k in range(max(hit_x_only, hit_y_only), n + 1)) / 2 ** n)
print(f"2) exact sign test: {hit_x_only} people hit only with X, {hit_y_only} only with Y -> two-sided p = {p:.3f}")

rng = np.random.default_rng(0)
x, y = rng.exponential(1.0, (20000, 30)), rng.exponential(1.0, (20000, 30))      # the SAME distribution: no real difference
z = (x.mean(1) - y.mean(1)) / np.sqrt(x.var(1, ddof=1) / 30 + y.var(1, ddof=1) / 30)
print(f"3) no real difference, n = 30 per group: the test says 'significant' (|z| > 1.96) in {np.mean(np.abs(z) > 1.96):.1%} of 20,000 experiments")
g = rng.exponential(1.0, (2000, 20, 2, 30))                                       # 2000 experiments, each with 20 separate null comparisons
zz = (g[:, :, 0].mean(2) - g[:, :, 1].mean(2)) / np.sqrt(g[:, :, 0].var(2, ddof=1) / 30 + g[:, :, 1].var(2, ddof=1) / 30)
print(f"4) 20 null comparisons per experiment: at least one 'significant' in {np.mean((np.abs(zz) > 1.96).any(1)):.1%} of experiments; 1 - 0.95^20 = {1 - 0.95 ** 20:.1%}")
```

```text
1) permutation test: observed difference 3.667; 20 ways to split the 6 values into 3 + 3; 2 of them are at least this extreme -> p = 0.10
2) exact sign test: 9 people hit only with X, 3 only with Y -> two-sided p = 0.146
3) no real difference, n = 30 per group: the test says 'significant' (|z| > 1.96) in 5.3% of 20,000 experiments
4) 20 null comparisons per experiment: at least one 'significant' in 65.6% of experiments; 1 - 0.95^20 = 64.2%
```

- **Reading it:** part 1 gives $p=0.10$ and part 2 gives $p=0.146$, as computed above. With no real difference the test said "significant" in 5.3% of 20,000 experiments, close to the promised 5%. With 20 comparisons per experiment, at least one was "significant" in 65.6% of 2,000 experiments, against $1-0.95^{20}=64.2\%$ from the equation (the single-test rate of 5.3%, a little above 5%, explains the small excess; Monte Carlo error is about 1 point). These are simulations with one distribution, not a proof.

## 🔴 4. Deep / formal
- **What a p-value is not.** It is not the probability that the null hypothesis is true, not the probability that the result is due to chance, and not a measure of importance. The picture above states the core error: confusing "chance of the data given the hypothesis" with "chance of the hypothesis given the data" (the "transposed conditional fallacy"). To get the second you need prior beliefs: see [[Bayes' Theorem]].
- **Not significant is not "no difference".** A test with little data cannot reject anything. In [[Error Bars Lab]] the best recommender (hit rate 0.087) against the second (0.074) gave $p=0.25$ with 943 people; that does not show the methods are equal, only that 943 people cannot separate a gap of 0.013. The interval for the difference ($-0.007$ to $+0.033$, see [[Confidence Interval]]) says the same thing more usefully.
- **Power.** The chance that a test finds a real difference of a given size. It rises with the sample size and the size of the effect. I did not compute the power of any test in my projects.
- **Exchangeability.** A permutation test assumes that, if nothing were going on, the labels could be shuffled freely. That fails when the data have order or groups (time series, people playing several games).
- **Multiple comparisons.** The $m$-test equation above is for independent tests; Bonferroni is safe but cautious. In [[Error Bars Lab]] I ran 6 comparisons of recommenders and so used the cut-off $0.05/6=0.0083$.
- **The 0.05 line is a convention.** It goes back to Ronald Fisher's early writing (from memory). The American Statistical Association warned against using one cut-off as the only criterion in 2016 (from memory, not checked).
- **Estimation beats yes/no.** An interval for the difference answers "how big, and how sure?"; a p-value answers only "how surprising under $H_0$?". Report both when you can.
- **Tests I did here:** a sign-flip permutation test (paired differences of game scores in [[Snake DQN]]), an exact permutation test (5 + 5 training-run means), and the exact sign test above. I did not run a $t$-test.

**What came before, and how this improved on it:**
- **Before:** judging a difference by eye ("method A looks better") or by a rule of thumb.
- **Its limit:** the eye cannot tell a real gap from luck, and people see patterns in noise.
- **What changed:** a rule that compares the data with what luck alone would give, with a stated false-alarm rate. Ronald Fisher's significance tests (1920s) and the Neyman-Pearson framework (1930s) (from memory, not checked).
- **How much better:** in my toy, with no real difference, the z-test cried "significant" in 5.3% of 20,000 experiments (target 5%); but 20 separate tests gave at least one false alarm in 65.6% of experiments, so the rule is only as good as how it is used. In [[Error Bars Lab]], 3 of 6 recommender comparisons passed the corrected cut-off and the one that looked like a win (0.087 against 0.074) did not.

## Related
- [[Confidence Interval]] · [[Bootstrap]] · [[Bayes' Theorem]] · [[Probability]] · [[Normal Distribution]] · [[Central Limit Theorem]] · [[Data Drift]] · [[Evaluation Metrics]] · [[Retrieval Metrics]] · [[Error Bars Lab]] · [[Recommender Shoot-out]] · [[Snake DQN]]
- [[AB Testing|A/B Testing]] · [[AB Test Lab|A/B Test Lab]]
## Sources
- My own toy (Python 3.9.6, NumPy 1.24.4) and the project [[Error Bars Lab]]. The history (Fisher, Neyman-Pearson, the 2016 statement of the American Statistical Association), the name "exact McNemar test" and the 0.05 convention are from memory; I did not read a source for this note.
- Picture: Repapetilto, CC BY-SA 3.0, [File:P-value in statistical significance testing.svg](https://commons.wikimedia.org/wiki/File:P-value_in_statistical_significance_testing.svg), Wikimedia Commons.

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/probability Null hypothesis:::The assumption that nothing real is going on (for example that the group labels make no difference). A test asks how surprising the data would be if it were true.

#flashcards/math/reverse #flashcards/topic/probability p-value:::The probability of seeing a difference at least as big as the observed one if the null hypothesis were true. It is not the probability that the null hypothesis is true.

#flashcards/math/concept #flashcards/topic/probability What are the steps of a hypothesis test?
?
Pick a measure of the difference; assume nothing real is going on (null hypothesis); find how often luck alone gives a difference at least as big (count exactly or simulate); that share is the p-value; compare it with a cut-off chosen in advance; beware of many tests, of reading p as the chance you are right, and of reading "not significant" as "no difference".

#flashcards/math/formula #flashcards/topic/probability What is the p-value?
?
$$p=\Pr\big(\,|T|\ge|t_{\text{obs}}|\;\big|\;H_0\,\big)$$
- $T$: the measure of the difference on a data set. $t_{\text{obs}}$: its value on the real data.
- $H_0$: the null hypothesis ("nothing real"). The bar means "given that". The absolute values make it two-sided.
- Matters: a small $p$ means data like mine would be rare if $H_0$ were true. It is not the probability that $H_0$ is true and says nothing about the size or importance of the difference.

#flashcards/math/example #flashcards/topic/probability Groups A = 5, 7, 8 and B = 2, 3, 4: what is the permutation-test p-value?::Observed difference $6.667-3=3.667$. There are $\binom63=20$ splits; 2 are at least this extreme in either direction, so $p=2/20=0.10$. With 3 + 3 values no result can give $p$ below 0.10.

#flashcards/math/formula #flashcards/topic/probability What is the exact sign test p-value for two methods tried on the same people?
?
$$p=\min\!\Bigg(1,\;2\sum_{k=\max(b,c)}^{n}\binom{n}{k}\frac{1}{2^{n}}\Bigg),\qquad n=b+c$$
- $b$: people who hit with only X. $c$: people who hit with only Y. $n=b+c$.
- $\binom{n}{k}/2^n$: chance of exactly $k$ heads in $n$ fair flips. The sum adds up splits at least as lopsided; the 2 makes it two-sided.
- Matters: only the people where the methods differ count; no bell-shape assumption.

#flashcards/math/example #flashcards/topic/probability 9 people hit only with X and 3 only with Y: what is the exact sign-test p?::$n=12$; $\binom{12}{9}+\binom{12}{10}+\binom{12}{11}+\binom{12}{12}=220+66+12+1=299$; $299/4096=0.073$; two-sided $p=0.146$, not significant.

#flashcards/math/formula #flashcards/topic/probability What is the chance of at least one false alarm among m independent tests?
?
$$1-(1-\alpha)^{m}$$
- $\alpha$: the cut-off for one test (often 0.05). $m$: number of tests.
- $(1-\alpha)^m$: chance that all $m$ tests stay quiet.
- Matters: with many tests, a "significant" one is expected from luck. Bonferroni tests each at $\alpha/m$.

#flashcards/math/example #flashcards/topic/probability 20 tests at 0.05 with nothing real: what is the chance of at least one false alarm, and the Bonferroni cut-off?::$1-0.95^{20}=1-0.358=0.642$; Bonferroni tests each at $0.05/20=0.0025$.

#flashcards/math/image #flashcards/topic/probability ![[p-value-significance-testing.svg]]
What does this picture show?
?
A p-value as the shaded green area under the curve of results expected if the null hypothesis is true: the probability of the observed or a more extreme result. The box warns that the probability of the data given the hypothesis is not the probability of the hypothesis given the data. (Repapetilto, CC BY-SA 3.0, Wikimedia Commons.)

#flashcards/math/concept #flashcards/topic/probability What did the hypothesis-testing toy show about false alarms?
?
With no real difference (two groups of 30 skewed values from the same distribution), the z-test said "significant" in 5.3% of 20,000 experiments. With 20 separate comparisons per experiment, at least one was "significant" in 65.6% of 2,000 experiments, against 64.2% from the equation. One distribution, not a proof.

#flashcards/math/concept #flashcards/topic/probability Why does "not significant" not mean "no difference"?
?
A test with little data cannot reject anything. In Error Bars Lab, hit rate 0.087 against 0.074 over 943 people gave p = 0.25 (interval for the difference −0.007 to +0.033): 943 people cannot separate a gap of 0.013, which is not the same as the methods being equal.

#flashcards/math/concept #flashcards/topic/probability What did the six recommender comparisons in Error Bars Lab show?
?
With the Bonferroni cut-off 0.05/6 = 0.0083: ALS beat item-kNN (p = 0.0067), ALS beat popular (0.0016) and popular beat the rating-trained MF (0.0013). ALS against user-kNN (0.25), user-kNN against item-kNN (0.054) and item-kNN against popular (0.37) did not pass.

#flashcards/math/concept #flashcards/topic/probability Which facts in the hypothesis-testing note are from memory?
?
Fisher's tests (1920s), Neyman-Pearson (1930s), the 0.05 convention going back to Fisher, the American Statistical Association statement of 2016, and the name "exact McNemar test". Not checked.

#flashcards/math/cloze #flashcards/topic/probability A ==p-value== is the probability of a result at least this extreme if the ==null hypothesis== were true.
