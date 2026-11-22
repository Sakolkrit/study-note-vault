---
tags: [A/B-testing, experiment, randomization, hypothesis-testing, statistics, evaluation, recommender, ML]
status: growing
level: 4
tier: 2
---

# A/B Testing

An A/B test is **an experiment in which real users are split at random into two groups, one group gets version A and the other gets version B, and you compare what they do**. Because the split is random, a difference in results can be blamed on the version and not on who happened to be in which group.

## 🟢 1. Simple (age 12)
You have two recipes for cookies and want to know which one people like more. You do not ask your friends "which one do you think is better?"; you **flip a coin for each guest** and give them either recipe A or recipe B, without letting them choose. At the end you count how many guests asked for a second cookie in each group. The coin flip matters: if the hungry guests all happened to get recipe B, you would learn nothing about the recipe.

**Example:** a shop has a blue "Buy" button and a green one. The first 1,000 visitors each see one at random. 50 buy with blue and 70 with green. Is green really better, or was that luck? That is the question an A/B test answers.

## 🔵 2. How it works
1. **Pick one thing to change** (version A is today's version, B has the change) and **one main number to compare**, such as "share of visitors who buy". Decide both before you start.
2. **Decide how many users you need** before starting ([[Hypothesis Testing]] has the idea of power; the sample-size equation is below). Too few users and a real improvement stays hidden.
3. **Assign users at random**: each user gets A or B by chance and sees only that one.
4. **Run until the planned number of users has arrived.** Do not stop early just because the numbers look good.
5. **Compare the two groups** with a test and an interval for the difference ([[Confidence Interval]]).
6. **Check for trouble:** are the group sizes what you planned, did anything else change during the test, did you look at many numbers and pick the best?
7. **Decide.** A "significant" win that is tiny may not be worth shipping; a "not significant" result is not proof of no effect.

![[ab-testing-example.png]]
*Two groups of visitors are sent to two versions of a page that differ in their button, and the click rates are compared (52% against 72% in this illustration). Author: Maxime Lorant, CC BY-SA 4.0, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:A-B_testing_example.png). The percentages are an illustration, not a measurement.*

## 🟠 3. Technical detail
- **Comparing two shares.** Let $n_A$ users see A and $h_A$ of them convert (buy, click, hit), and the same for B. The pooled two-proportion z statistic is

$$
z=\frac{\hat p_B-\hat p_A}{\sqrt{\hat p\,(1-\hat p)\,\big(\tfrac{1}{n_A}+\tfrac{1}{n_B}\big)}},\qquad \hat p_A=\frac{h_A}{n_A},\ \ \hat p_B=\frac{h_B}{n_B},\ \ \hat p=\frac{h_A+h_B}{n_A+n_B}
$$

**Where:**
- $\hat p_A$ and $\hat p_B$ are the share of users who converted in each group; $\hat p$ is the share over both groups together.
- $n_A,n_B$ are the numbers of users in each group.
- The bottom is the standard error of the difference if A and B were really the same ([[Confidence Interval]]); $z$ counts how many of those standard errors the observed difference is.
- **Matters:** $z$ is compared with 1.96 for a 5% two-sided test ([[Normal Distribution]]). Bigger groups shrink the bottom, so the same gap gives a larger $z$. It relies on enough conversions in each group (a handful is not enough).

**Example (tiny numbers):** A: 50 of 1,000 buy; B: 70 of 1,000 buy (my arithmetic in Python).
- $\hat p_A=0.05$, $\hat p_B=0.07$, pooled $\hat p=120/2000=0.06$.
- Bottom: $\sqrt{0.06\times0.94\times(1/1000+1/1000)}=\sqrt{0.0001128}=0.01062$.
- $z=0.02/0.01062=\mathbf{1.883}$, which gives a two-sided $p=0.0597$.
- Meaning: green looks 40% better (0.07 against 0.05), yet this is not significant at 5%: a gap this size appears 6% of the time by luck between equal buttons. You would need more users.

- **How many users?** For a 5% two-sided test with power $1-\beta$ (the chance of catching a real gap of size $\delta$), the number needed in **each** group is about

$$
n\;\approx\;\frac{2\,\big(z_{1-\alpha/2}+z_{1-\beta}\big)^{2}\;\bar p\,(1-\bar p)}{\delta^{2}}
$$

**Where:**
- $z_{1-\alpha/2}=1.96$ for a 5% two-sided test; $z_{1-\beta}=0.8416$ for 80% power (both from the normal table, from memory).
- $\bar p$ is the average of the two true rates and $\delta$ the gap you want to be able to detect (the *minimum detectable effect*).
- **Matters:** $n$ grows with $1/\delta^2$: to detect a gap half as big you need four times the users. It also grows when $\bar p$ is near 0.5 (noisy outcomes) and shrinks for rare events only in absolute terms (small $\bar p$ with small $\delta$ still needs a lot).

**Example (tiny numbers):** A converts 5%, you want to catch B at 6% ($\delta=0.01$, $\bar p=0.055$).
- $(1.96+0.8416)^2=7.849$; $2\times7.849=15.698$.
- $\bar p(1-\bar p)=0.055\times0.945=0.051975$.
- $n=15.698\times0.051975/0.0001=\mathbf{8{,}159}$ per group, so about 16,300 in total.
- Meaning: with 1,000 per group you would almost always miss a one-point gain (see the code).

- **Code: false alarms, peeking and power.** Part 1 is the example above. Part 2 is an **A/A test**: both groups get the same version (true rate 0.05, 1,000 per group), and I count how often the test says "significant". Part 3 repeats it but looks after every 200 users per group (10 looks) and stops at the first "significant". Part 4 gives a real 5% against 6% gap and counts how often the test finds it:

```python
# file: toy_ab.py
import numpy as np
from math import erf, sqrt

def two_sided_p(z):                                   # normal tail, no library needed
    return 1 - erf(abs(z) / sqrt(2))

def z_stat(hA, nA, hB, nB):                           # pooled two-proportion z statistic
    p = (hA + hB) / (nA + nB)
    return (hB / nB - hA / nA) / np.sqrt(p * (1 - p) * (1 / nA + 1 / nB) + 1e-300)

rng = np.random.default_rng(0)
a, b = 50, 70
z = z_stat(a, 1000, b, 1000)
print(f"1) A: {a}/1000, B: {b}/1000  ->  z = {z:.3f}, two-sided p = {two_sided_p(z):.4f}")

sims = 2000
print("2) A/A test: both arms have the same true rate 0.05, 1,000 users per arm, one look at the end")
hA, hB = rng.binomial(1000, 0.05, sims), rng.binomial(1000, 0.05, sims)
print(f"   'significant' (p < 0.05) in {np.mean(np.abs(z_stat(hA, 1000, hB, 1000)) > 1.96):.1%} of {sims} experiments")

print("3) the same A/A test, but looking after every 200 users per arm (10 looks) and stopping at the first p < 0.05")
xa, xb = rng.random((sims, 2000)) < 0.05, rng.random((sims, 2000)) < 0.05
ca, cb = np.cumsum(xa, axis=1), np.cumsum(xb, axis=1)
looks = np.arange(200, 2001, 200)
zz = z_stat(ca[:, looks - 1], looks, cb[:, looks - 1], looks)
print(f"   'significant' at some look in {np.mean((np.abs(zz) > 1.96).any(axis=1)):.1%} of {sims} experiments (one look at the end: {np.mean(np.abs(zz[:, -1]) > 1.96):.1%})")

print("4) a real difference: A converts 5%, B converts 6%; how many users per arm?")
need = 2 * (1.96 + 0.8416) ** 2 * 0.055 * 0.945 / 0.01 ** 2
print(f"   formula for 80% power: {need:,.0f} users per arm")
for n in (1000, int(np.ceil(need)), 15000):
    hA, hB = rng.binomial(n, 0.05, sims), rng.binomial(n, 0.06, sims)
    print(f"   {n:>6,} per arm: the test finds the difference in {np.mean(np.abs(z_stat(hA, n, hB, n)) > 1.96):.1%} of {sims} experiments")
```

```text
1) A: 50/1000, B: 70/1000  ->  z = 1.883, two-sided p = 0.0597
2) A/A test: both arms have the same true rate 0.05, 1,000 users per arm, one look at the end
   'significant' (p < 0.05) in 5.1% of 2000 experiments
3) the same A/A test, but looking after every 200 users per arm (10 looks) and stopping at the first p < 0.05
   'significant' at some look in 18.4% of 2000 experiments (one look at the end: 4.4%)
4) a real difference: A converts 5%, B converts 6%; how many users per arm?
   formula for 80% power: 8,159 users per arm
    1,000 per arm: the test finds the difference in 17.3% of 2000 experiments
    8,159 per arm: the test finds the difference in 79.1% of 2000 experiments
   15,000 per arm: the test finds the difference in 96.0% of 2000 experiments
```

- **Reading it:** the example gives $z=1.883$ and $p=0.0597$, as computed. With no real difference the test fires in 5.1% of 2,000 experiments, as promised. **Peeking ten times raises that to 18.4%** (one look at the end: 4.4%), so stopping at the first good-looking result makes false wins about four times more likely. For the real 5% against 6% gap, 1,000 per group finds it in only 17.3% of experiments, the formula's 8,159 per group finds it in 79.1% (close to the 80% promised), and 15,000 per group in 95.9%. Each figure is 2,000 simulated experiments, so it carries a Monte Carlo error of about 1 point.
- **A real data check.** [[AB Test Lab|A/B Test Lab]] imitates experiments on the per-person recommender results of [[Recommender Shoot-out]]: swapping the most-popular list for the ALS recommender (a true gap of 0.037 in hit rate) needs about 1,450 people in total, but ALS against user-kNN (gap 0.013) needs about 14,400.

## 🔴 4. Deep / formal
- **Why randomise.** Random assignment makes the two groups alike in everything, measured or not (age, mood, device), apart from the version. Without it, any difference could come from who chose what. This is the same idea as a randomised controlled trial in medicine.
- **Interference.** The simple analysis assumes that one user's version does not affect another user's result. That fails in social networks, marketplaces and anything with shared supply or prices (from memory); the test can then be biased.
- **Sample ratio mismatch.** If you planned a 50/50 split and the groups come out 50.8/49.2 with thousands of users, something broke in the assignment or the logging, and the result should not be trusted (from memory; a chi-square check on the counts is the usual test).
- **Peeking.** The toy shows the inflation: 18.4% false wins with 10 looks. Remedies are to fix the sample size in advance, or to use methods built for looking early (sequential tests, "always valid" p-values, from memory, none run here).
- **Many metrics and many variants.** Each extra metric or variant is another test ([[Hypothesis Testing]]: $1-(1-\alpha)^m$). Name one main metric in advance, treat the others as guardrails or hints, and correct for the number of comparisons.
- **Power and the minimum detectable effect.** A test that is too small cannot see real gains. "No significant difference" from a small test is weak evidence; report the interval for the difference.
- **Statistical against practical significance.** With millions of users, a gain of 0.0001 can be "significant" and still not worth shipping.
- **Offline against online.** In [[Recommender Shoot-out]] the methods were compared offline on held-out movies. An online A/B test measures what users do when shown the lists, which can differ (users may click on what is easy to see, and a rating-trained recommender and a list ranker are judged by different things). My lab uses the offline hits as stand-in outcomes, so it says nothing about real clicks.
- **Same people versus separate people.** A live test shows each user one version, so the comparison is *unpaired*. Offline you can score both methods on the same person (paired), which is much sharper: in [[AB Test Lab|A/B Test Lab]], user-kNN against ALS needs about 4,700 people paired and about 14,400 in two separate groups (80% power, my arithmetic).
- **Variance reduction and interleaving.** Using each user's pre-test behaviour to reduce noise (often called CUPED), and mixing the two methods' items in one list so each user compares them (interleaving), are ways to need fewer users (from memory, not run here).
- **Novelty and long-term effects.** A change can look good for a week because it is new. A short test may miss what happens after users get used to it (from memory).

**What came before, and how this improved on it:**
- **Before:** compare "before the change" with "after the change", or ask a few people, or follow the opinion of the loudest person in the room.
- **Its limit:** between "before" and "after" the season, the traffic, the weather and the competitors all change, so you cannot tell what the change did.
- **What changed:** run both versions at the same time on randomly split users. Randomised experiments go back to Ronald Fisher's agricultural work in the 1920s, and online companies began running them on websites around the early 2000s (from memory, not checked).
- **How much better:** measured in my toy and lab: with random groups and a planned sample size, an A/A test gave false wins in 5.1% of experiments (the promised 5%), and the sample-size formula gave 79.1% power in simulation (80% promised). Done badly the same test fails: ten peeks gave 18.4% false wins, and 1,000 users per group caught a real 5%-to-6% gain only 17.3% of the time. I have no published figure to quote.

## Related
- [[Hypothesis Testing]] · [[Confidence Interval]] · [[Bootstrap]] · [[Normal Distribution]] · [[Probability]] · [[Evaluation Metrics]] · [[Recommender Systems]] · [[Recommender Shoot-out]] · [[AB Test Lab|A/B Test Lab]] · [[Error Bars Lab]] · [[Data Drift]] · [[MLOps]]

## Sources
- My own toy and lab (Python 3.9.6, NumPy 1.24.4). The z-table values 1.96 and 0.8416, the history (Fisher in the 1920s, websites in the early 2000s), sample ratio mismatch, interference, sequential tests, CUPED, interleaving and novelty effects are from memory; I did not read a source for this note.
- Picture: Maxime Lorant, CC BY-SA 4.0, [File:A-B testing example.png](https://commons.wikimedia.org/wiki/File:A-B_testing_example.png), Wikimedia Commons.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation A/B test:::An experiment where users are split at random into two groups, one sees version A and the other version B, and a chosen metric is compared between them.

#flashcards/ml/reverse #flashcards/topic/evaluation A/A test:::An A/B test in which both groups get the same version. It should say "significant" only about as often as the chosen error rate (5%), so it checks that the setup and the test are sound.

#flashcards/ml/concept #flashcards/topic/evaluation What are the steps of an A/B test?
?
Pick one change and one main metric in advance; decide the number of users before starting; assign users at random so each sees only one version; run until the planned number of users has arrived; compare the groups with a test and an interval; check group sizes and other changes; decide with the size of the effect in mind.

#flashcards/ml/formula #flashcards/topic/evaluation What is the two-proportion z statistic used to compare A and B conversion rates?
?
$$z=\frac{\hat p_B-\hat p_A}{\sqrt{\hat p\,(1-\hat p)\,\big(\tfrac{1}{n_A}+\tfrac{1}{n_B}\big)}},\qquad \hat p_A=\frac{h_A}{n_A},\ \ \hat p_B=\frac{h_B}{n_B},\ \ \hat p=\frac{h_A+h_B}{n_A+n_B}$$
- $h_A,h_B$: users who converted in each group; $n_A,n_B$: group sizes; $\hat p$: pooled share.
- The bottom is the standard error of the difference if A and B were the same; $z$ counts how many of those the observed gap is.
- Matters: compare $z$ with 1.96 for 5% two-sided. Bigger groups give larger $z$ for the same gap; it needs enough conversions in each group.

#flashcards/ml/example #flashcards/topic/evaluation A: 50 of 1,000 buy; B: 70 of 1,000. What are $z$ and $p$?::Pooled $\hat p=0.06$; bottom $\sqrt{0.06\times0.94\times0.002}=0.01062$; $z=0.02/0.01062=1.883$; two-sided $p=0.0597$, not significant at 5% although B looks 40% better.

#flashcards/ml/formula #flashcards/topic/evaluation How many users per group does an A/B test need?
?
$$n\;\approx\;\frac{2\,\big(z_{1-\alpha/2}+z_{1-\beta}\big)^{2}\;\bar p\,(1-\bar p)}{\delta^{2}}$$
- $z_{1-\alpha/2}=1.96$ (5% two-sided); $z_{1-\beta}=0.8416$ (80% power), from the normal table (memory).
- $\bar p$: average of the two true rates; $\delta$: the smallest gap you want to catch.
- Matters: $n$ grows with $1/\delta^2$, so half the gap needs four times the users.

#flashcards/ml/example #flashcards/topic/evaluation A converts 5% and you want to catch 6%: how many users per group?::$\delta=0.01$, $\bar p=0.055$; $(1.96+0.8416)^2=7.849$; $n=2\times7.849\times0.055\times0.945/0.0001=8{,}159$ per group, about 16,300 in total.

#flashcards/ml/image #flashcards/topic/evaluation ![[ab-testing-example.png]]
What does this picture show?
?
Two groups of visitors sent to two versions of a web page that differ in their button, with the click rates compared (52% against 72%, an illustration and not a measurement). (Maxime Lorant, CC BY-SA 4.0, Wikimedia Commons.)

#flashcards/ml/concept #flashcards/topic/evaluation What did the A/B toy show about peeking?
?
Two equal versions (true rate 0.05, 1,000 users per group), 2,000 experiments: one look at the end said "significant" in 5.1% (4.4% in the separate set of experiments used for the peeking run); looking 10 times and stopping at the first "significant" gave 18.4%. Stopping at the first good-looking result makes false wins about four times more likely.

#flashcards/ml/concept #flashcards/topic/evaluation What did the A/B toy show about power?
?
A real 5% against 6% gap: 1,000 users per group found it in 17.3% of 2,000 experiments, the formula's 8,159 per group in 79.1% (80% promised), and 15,000 per group in 95.9%. Monte Carlo error about 1 point.

#flashcards/ml/concept #flashcards/topic/evaluation Why do users have to be assigned at random in an A/B test?
?
Random assignment makes the groups alike in everything, measured or not, apart from the version. Without it, a difference could come from who chose or was given which version.

#flashcards/ml/concept #flashcards/topic/evaluation Why is a live A/B test less sharp than an offline comparison on the same people?
?
A live test shows each user only one version (unpaired); offline both methods can be scored on the same person (paired). In A/B Test Lab, user-kNN against ALS needed about 4,700 people paired and about 14,400 in two separate groups for 80% power (my arithmetic).

#flashcards/ml/concept #flashcards/topic/evaluation What are sample ratio mismatch and interference?
?
Sample ratio mismatch: group sizes differ from the planned split by more than chance allows, which points to a broken assignment or logging. Interference: one user's version affects another user's result (social networks, marketplaces), which biases the simple comparison. Both from memory.

#flashcards/ml/concept #flashcards/topic/evaluation Which facts in the A/B testing note are from memory?
?
The z values 1.96 and 0.8416, Fisher in the 1920s and websites in the early 2000s, sample ratio mismatch, interference, sequential tests, CUPED, interleaving and novelty effects. Not checked.

#flashcards/ml/cloze #flashcards/topic/evaluation In an A/B test users are assigned ==at random== to a version, and the ==sample size== should be decided before starting.
