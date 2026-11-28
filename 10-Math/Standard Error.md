---
tags: [standard-error, statistics, probability, uncertainty, evaluation, sampling, math]
status: growing
level: 4
tier: 2
---

# Standard Error

The **standard error** is how much a number measured on a sample (an average, a hit rate, an accuracy) would wobble if you repeated the measurement on a fresh sample of the same size. It is the "give or take" that should sit beside every score: "0.72 give or take 0.03". It shrinks as the sample grows, but only with the square root of the sample size.

## 🟢 1. Simple (age 12)
You want to know how tall the kids in your school are, but you can only measure a few. Ask 5 kids and you might get a taller group by luck. Ask 200 kids and the average barely changes if you ask 200 different ones. The **standard error** is a number that says "if I did this measuring again with other kids, my average would move about this much".

**Example:** a robot answers 10 quiz questions and gets 6 right. Does it know 60% of everything? Not really: with another 10 questions it could easily get 4 or 8. After 1,000 questions the score hardly moves between tries. The standard error is the size of that movement, big for 10 questions and small for 1,000.

## 🔵 2. How it works
1. **Measure** a score on a sample: for a quiz, the share of questions answered right.
2. **Imagine repeating the test** with new questions of the same kind and the same count. You would get a slightly different score each time. The typical size of that difference is the standard error.
3. **You cannot repeat it for real, so you estimate it from one run.** For a hit rate the estimate needs only the score itself and the number of questions: a score near 0.5 wobbles most, a score near 0 or 1 wobbles least ([[Variance]]).
4. **More questions help slowly.** To halve the wobble you need four times as many questions, because it shrinks with the square root of the count ([[Law of Large Numbers]]).
5. **Turn it into a range.** About two standard errors either way is a rough 95% range ([[Confidence Interval]]). For a sum of many small pieces, the scores follow a bell curve, which is why this works ([[Central Limit Theorem]], [[Normal Distribution]]).
6. **For two models on the same questions, pair them.** Do not combine the two separate wobbles. Look only at the questions where the models disagree. This gives a much smaller, fairer wobble for the *difference* ([[Hypothesis Testing]]).

![[sampling-distribution-sample-mean.png]]
*Top left: a population. The other three panels: histograms of sample means when each sample has 5, 15 and 30 values. The spread of the means narrows as the samples grow; that spread is the standard error. Author: Kindredgrey, CC BY-SA 4.0, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Sampling_Distributions_of_the_Sample_Mean_from_a_Normal_Population.png). I looked at the picture and checked it shows narrowing with sample size; I did not check the numbers on its axes, and the last panel uses a different horizontal scale.*

## 🟠 3. Technical detail
All numbers below come from the script `se_demo.py` (Python 3.9.6, standard library only) and from my saved runs of [[LLM Evaluation Lab]] (Qwen2.5-0.5B-Instruct and 1.5B-Instruct, HellaSwag, first 200 questions).

```python
"""Standard error demo: formula vs real wobble, how it shrinks with n, paired vs unpaired."""
import json, math, random
random.seed(0)

# 1. A model that is truly right 60% of the time takes tests of different sizes, 2000 times each.
p_true = 0.6
print("n, formula s.e., measured spread of the score over 2000 repeats")
for n in (10, 50, 200, 1000):
    scores = [sum(random.random() < p_true for _ in range(n)) / n for _ in range(2000)]
    mean = sum(scores) / len(scores)
    spread = math.sqrt(sum((s - mean) ** 2 for s in scores) / (len(scores) - 1))
    print(n, round(math.sqrt(p_true * (1 - p_true) / n), 4), round(spread, 4))

# 2. How many questions to get +-0.01 (one s.e.) at p = 0.5, and +-0.02
for target in (0.05, 0.02, 0.01):
    print("s.e. %.2f needs n about %d" % (target, math.ceil(0.25 / target ** 2)))

# 3. Paired against unpaired, my real runs (first 200 questions, HellaSwag acc, 0.5B against 1.5B)
S, B = "Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct"
load = lambda m: json.load(open("runs/mine_%s_hellaswag_200.json" % m))["rows"]
a, b = load(S), load(B)
n = 200
only_b = sum(x["pred"] != x["gold"] and y["pred"] == y["gold"] for x, y in zip(a, b))
only_s = sum(x["pred"] == x["gold"] and y["pred"] != y["gold"] for x, y in zip(a, b))
pa = sum(x["pred"] == x["gold"] for x in a) / n
pb = sum(y["pred"] == y["gold"] for y in b) / n
se = lambda p: math.sqrt(p * (1 - p) / (n - 1))
print("scores", pa, pb, "each s.e.", round(se(pa), 3), round(se(pb), 3))
print("unpaired s.e. of the difference", round(math.sqrt(se(pa) ** 2 + se(pb) ** 2), 3))
d = (only_b - only_s) / n
paired = math.sqrt(only_b + only_s - (only_b - only_s) ** 2 / n) / n
print("only 1.5B right", only_b, "only 0.5B right", only_s, "difference", d, "paired s.e.", round(paired, 3), "z", round(d / paired, 1))

# 4. n against n-1 at n = 200, p = 0.72
print("n vs n-1:", round(math.sqrt(.72 * .28 / 200), 5), round(math.sqrt(.72 * .28 / 199), 5))
```

I saw:
```
n, formula s.e., measured spread of the score over 2000 repeats
10 0.1549 0.1596
50 0.0693 0.0686
200 0.0346 0.0344
1000 0.0155 0.0153
s.e. 0.05 needs n about 100
s.e. 0.02 needs n about 625
s.e. 0.01 needs n about 2500
scores 0.425 0.495 each s.e. 0.035 0.035
unpaired s.e. of the difference 0.05
only 1.5B right 17 only 0.5B right 3 difference 0.07 paired s.e. 0.022 z 3.2
n vs n-1: 0.03175 0.03183
```

**Part 1: the formula against a real wobble.** I simulated a model that is truly right 60% of the time, took a test of $n$ questions 2000 times, and measured how much the score moved.

| Questions $n$ | Formula standard error | Measured spread over 2000 repeats |
|---|---|---|
| 10 | 0.1549 | 0.1596 |
| 50 | 0.0693 | 0.0686 |
| 200 | 0.0346 | 0.0344 |
| 1000 | 0.0155 | 0.0153 |

- **Questions $n$:** how many questions the test had. More questions, less luck.
- **Formula standard error:** the square root of (score × (1 − score) / $n$) with the true score 0.6 (general formula, my arithmetic).
- **Measured spread:** the standard deviation of the 2000 simulated scores, so it is the wobble itself, not an estimate of it.

**Summary: what the table shows.** The formula and the real wobble agree to within about 0.005, the best at 200 and 1000 questions. The worst gap is at 10 questions (0.1549 against 0.1596), where the formula is least exact and where the score itself is almost useless: 0.6 give or take 0.15. Caution: this is a simulation of coin flips with a known true rate; real questions differ in difficulty and a real test set is one fixed sample, not a fresh draw each time.

**Part 2: how many questions you need.** At a score of 0.5 (the worst case): a standard error of 0.05 needs about 100 questions, 0.02 needs 625, and 0.01 needs 2,500 (my arithmetic). This is why ±0.035 on my 200-question runs is a hard limit, not something a better prompt would remove.

**Part 3: paired against unpaired, on real data.** HellaSwag `acc`: the 0.5B model scored 0.425 and the 1.5B model 0.495 on the same 200 questions, each with a standard error of 0.035.

| Way to measure the difference | Standard error of the difference | Difference in standard errors |
|---|---|---|
| Unpaired (combine the two ± values) | 0.050 | 0.07 / 0.050 = 1.4 (my arithmetic) |
| Paired (use the 17 and 3 below) | 0.022 | 3.2 (from the script) |

- **Unpaired:** the square root of the two squared standard errors added, which assumes the two scores are independent.
- **Paired:** only 20 questions were answered differently: 17 right only for the 1.5B model and 3 right only for the 0.5B model. The other 180 questions give no information about which is better, so they add no wobble.
- **Difference in standard errors:** how far the gain is from zero measured in wobbles. About 2 or more is usually read as "not just luck" (rough rule, general knowledge).

**Summary: what the table shows.** The same 0.07 gain looks unclear unpaired (1.4) and clear paired (3.2). Caution: the paired formula I used, the square root of (disagreements minus the squared difference divided by $n$), divided by $n$, is an approximation from my own lab code; I did not check it against another source. It also says the 1.5B model is better on these 200 questions, not by how much on the whole test.

**Part 4: $n$ against $n-1$.** At 200 questions and a score of 0.72, dividing by $n$ gives 0.03175 and by $n-1$ gives 0.03183. The difference is in the fourth decimal, so the choice does not matter at this size. The lab's code uses $n-1$, which matched the errors the official harness printed (I did not read its source).

## 🔴 4. Deep / formal
**Standard error of a sample mean** (for $n$ independent measurements, each with spread $\sigma$):

$$
\mathrm{SE} \;=\; \frac{\sigma}{\sqrt{n}}
$$

**Where:**
- $\mathrm{SE}$ is the standard error, the typical distance between the sample mean and the true mean. **Matters:** it is the "give or take" that goes beside the score.
- $\sigma$ is the spread of a single measurement. For a right-or-wrong question with true rate $p$, $\sigma=\sqrt{p(1-p)}$. **Matters:** a rate near 0.5 spreads most, so the same test is least precise there.
- $n$ is the number of independent measurements. **Matters:** the root means four times the data for half the error; questions on the same topic or near-copies are not independent and make the real error larger than the formula says.

**Example (tiny numbers):** a model scores 0.6 on a 200-question test.
- $\sigma=\sqrt{0.6\times0.4}=\sqrt{0.24}=0.490$.
- $\sqrt{200}=14.14$.
- $\mathrm{SE}=0.490/14.14=0.0346$.
- The answer means: with 200 questions the score moves about 0.035 between repeats, so "0.60" really says "about 0.53 to 0.67" (two standard errors). Four times the questions (800) would halve it to 0.017.

**Why the square root.** Averaging $n$ independent numbers adds their spreads as squares ([[Variance]] of a sum adds, $n\sigma^2$), then dividing by $n$ divides the variance by $n^2$, which leaves variance $\sigma^2/n$ and spread $\sigma/\sqrt{n}$ (general knowledge).

**What the formula assumes, and when it breaks:**
- *Independent questions.* Questions copied from one source, several questions about the same passage, or duplicates break this. The real wobble is bigger.
- *A random sample of the question population.* The first 200 questions of a file are not a random draw (my lab takes the first N), so the standard error describes luck within a made-up population, not a guarantee about the whole benchmark.
- *Near 0 or 1 or tiny $n$.* The bell-curve shortcut for the range is poor; use a proper interval or the [[Bootstrap]] ([[Confidence Interval]]).
- *Many comparisons.* If you test many ideas on the same test set, some "clear" gains appear by luck; the standard error of a single comparison does not cover that.
- *Standard deviation against standard error.* The standard deviation describes how spread out the individual measurements are and stays about the same as $n$ grows; the standard error describes how well the average is pinned down and shrinks (general knowledge).

**What came before, and how this improved on it:**
- **Before:** report the bare score ("0.72") and compare two models by which number is bigger.
- **Its limit:** a score from 200 questions can move by several points from luck, so a bigger number may mean nothing. In my run the HellaSwag `acc` scores were 0.425 and 0.495 and I first doubted the gain.
- **What changed:** give every score a standard error, and for two models on the same questions use the paired standard error of the difference.
- **How much better:** my run: the same +0.070 gain was 1.4 unpaired standard errors from zero (unclear) and 3.2 paired (clear), a factor of 2.3 (0.050 against 0.022, my arithmetic). Who first used the term "standard error" and when is not checked here.

## Related
[[Confidence Interval]] · [[Central Limit Theorem]] · [[Law of Large Numbers]] · [[Variance]] · [[Normal Distribution]] · [[Hypothesis Testing]] · [[Bootstrap]] · [[Probability]] · [[Expected Value]] · [[Benchmarks]] · [[Evaluation Metrics]] · [[LLM Evaluation Lab]]

## Sources
- My own runs: `se_demo.py` (Python 3.9.6) and the saved runs of [[LLM Evaluation Lab]] (Qwen2.5 models, Apache-2.0; HellaSwag validation, first 200 questions).
- Image: Kindredgrey, CC BY-SA 4.0, Wikimedia Commons (link under the picture).
- The formula, the square-root argument and the rule of thumb of two standard errors are general knowledge, not checked against a textbook here. The note was not opened in Obsidian.

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/probability Standard error:::How much a score measured on a sample would wobble if you repeated the measurement on a fresh sample of the same size; the "give or take" that goes beside a score.

#flashcards/math/formula #flashcards/topic/probability Write the standard error of a sample mean and explain each symbol.
?
$$\mathrm{SE} = \frac{\sigma}{\sqrt{n}}$$
- $\mathrm{SE}$: the typical distance between the sample mean and the true mean.
- $\sigma$: the spread of one measurement; for a right-or-wrong question with true rate $p$ it is $\sqrt{p(1-p)}$, largest at 0.5.
- $n$: the number of independent measurements; four times the data halves the error. Dependent or duplicate questions make the real error larger.

#flashcards/math/example #flashcards/topic/probability A model scores 0.6 on 200 questions. What is the standard error?::sigma = sqrt(0.6 x 0.4) = 0.490, sqrt(200) = 14.14, so SE = 0.490 / 14.14 = 0.0346. Two standard errors gives about 0.53 to 0.67 (my arithmetic).

#flashcards/math/concept #flashcards/topic/probability Why does it take four times the questions to halve the standard error?::The error shrinks with the square root of the number of questions, sigma over root n (general knowledge). At 0.5, s.e. 0.05 needs about 100 questions, 0.02 needs 625, 0.01 needs 2,500 (my arithmetic).

#flashcards/math/concept #flashcards/topic/probability In my simulation, how well did the formula match the real wobble?::Within about 0.005. For n = 200 the formula gave 0.0346 and the measured spread of 2000 repeats was 0.0344; for n = 10 it was 0.1549 against 0.1596 (simulation of a model with a true rate of 0.6).

#flashcards/math/concept #flashcards/topic/probability Why is a rate near 0.5 the hardest to pin down?::The spread of a right-or-wrong question is the square root of p(1-p), which is largest at 0.5, so the same number of questions gives the biggest standard error there.

#flashcards/math/concept #flashcards/topic/probability Two models answer the same questions. Why compare them paired and not by their two ± values?::The questions where both agree carry no information about which is better; combining the two separate errors counts them as noise. In my run (HellaSwag acc, 200 questions) the 0.07 gain was 1.4 standard errors unpaired (s.e. 0.050) and 3.2 paired (s.e. 0.022; 17 questions only the bigger model got, 3 only the smaller).

#flashcards/math/concept #flashcards/topic/probability What does the standard error not cover?::Dependent or duplicated questions (real error is larger), a sample that is not random (the first 200 questions are not), and trying many ideas on one test set, where some gains appear by luck.

#flashcards/math/concept #flashcards/topic/probability What is the difference between standard deviation and standard error?::The standard deviation says how spread out individual measurements are and stays about the same as the sample grows. The standard error says how well the average is pinned down and shrinks as the sample grows (general knowledge).

#flashcards/math/image #flashcards/topic/probability ![[sampling-distribution-sample-mean.png]]
What does the picture show, and what is the standard error in it?
?
A population (top left) and the histograms of sample means for samples of 5, 15 and 30 values. The means bunch more tightly as the sample grows; the spread of those means is the standard error. Author Kindredgrey, CC BY-SA 4.0 (I checked the narrowing by eye, not the axis numbers).

#flashcards/math/cloze #flashcards/topic/probability The standard error shrinks with the ==square root== of the number of questions, so halving it takes ==four times== as many.
