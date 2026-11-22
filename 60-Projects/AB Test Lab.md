---
tags: [project, ab-testing, experiment, hypothesis-testing, statistics, recommender, simulation, numpy]
status: done
stack: Python 3.9.6, NumPy 1.24.4
goal: simulate A/B tests on the per-person recommender results of an earlier project, and find out how many people are needed to see a real gap, whether the sample-size formula is right, and what separate groups cost compared with the same people
---

# A/B Test Lab

> **In one line:** use the 943 people of [[Recommender Shoot-out]] as a pretend crowd, run thousands of fake [[AB Testing|A/B Testing]] experiments on it (one half sees recommender A, the other half recommender B), and measure how often the test finds the real gap, with the sample-size formula checked against the simulation.

## Goal
Done means: (1) the real hit rate of three recommenders in the crowd, (2) an A/A test showing the false-alarm rate, (3) for two real gaps (most-popular against ALS, and user-kNN against ALS) the share of fake experiments that find the gap at several sizes, including the size the formula recommends, and (4) a comparison with the paired test of [[Error Bars Lab]]. **This is a simulation of an experiment, not an experiment:** nobody was shown any list, and the "outcome" of a person is the offline hit, not a click.

## Result in short
- **The pretend crowd:** the 943 people of the shoot-out. For each person the outcome under a method is 0 or 1: is the person's hidden movie in that method's top 10? True rates in this crowd: most-popular 0.0498, user-kNN 0.0742, ALS 0.0870.
- **An A/A test works as promised:** ALS against ALS on 2,000 people said "significant" in 5.7% of 2,000 fake experiments (the promised 5%; Monte Carlo error about 0.5 points).
- **A big gap is easy.** Most-popular to ALS (true gap +0.0371): the formula says 726 people per group, 1,452 in total, for 80% power. In simulation, 1,452 people in total found the gap in **80.2%** of fake experiments. With 943 people (the size of the whole crowd) the test found it in 62.5%, with 2,000 in 91.2% and with 4,000 in 99.7%.
- **A small gap is expensive.** User-kNN to ALS (true gap +0.0127): the formula says 7,183 people per group, **14,366 in total**. In simulation, 14,366 people found the gap in **81.2%**; 943 found it in 12.5%, 4,000 in 31.1%, 8,000 in 56.8% and 16,000 in 84.7%.
- **The formula matched the simulation** at both sizes it asked for (80.2% and 81.2% against 80% promised).
- **Separate groups cost about three times the people.** For user-kNN to ALS, a paired test (the same people see both methods, as in [[Error Bars Lab]]) needs about **4,721** people for 80% power, against 14,366 in two separate groups. A live test cannot show one user both lists, so it pays this price (interleaving and variance-reduction tricks, from memory, are how real teams fight it; I did not run them).
- **What it says about the earlier result:** in [[Error Bars Lab]], 943 people could not separate ALS from user-kNN ($p=0.25$). The simulation says why: a live A/B test on 943 people would find that gap in only 12.5% of experiments.
- **Not done:** no peeking test on this data (the toy in [[AB Testing|A/B Testing]] has it), no online outcome, one data set, no multiple metrics.

## What you need
- Python 3.9.6, `numpy` 1.24.4 (my existing environment on Windows; no fresh virtual environment).
- The file `ranks.npz` written by `ranks.py` in [[Error Bars Lab]] (the rank of each person's hidden movie for six methods; this project reads the `popular`, `user_knn` and `als` entries). A few seconds on a laptop; no GPU, no download.

## Steps
1. **Put `ranks.npz` in an empty folder** and save the script below as `ab_lab.py`. What it does, in order:
   - Turns ranks into hits (1 if the rank is 10 or better) for each method; these 943 pairs are the pretend crowd.
   - `experiment` draws a chosen number of people **with replacement** from the crowd (so it can imitate a bigger crowd than 943 by repeating people), gives the first half method A and the second half method B, and counts hits in each half.
   - `z_stat` is the pooled two-proportion statistic of [[AB Testing|A/B Testing]]; "finds the gap" means $|z|>1.96$.
   - `need_per_arm` is the sample-size formula for 80% power and a 5% two-sided test.
   - It runs an A/A test, then 2,000 fake experiments at each listed size for the two gaps, and finally the paired sample size for user-kNN against ALS.

```python
# file: ab_lab.py
import numpy as np
from math import erf, sqrt

R = np.load("ranks.npz")
hit = {k: (R[k] <= 10).astype(float) for k in R.files}                       # 1 if the person's hidden movie is in the top 10
pop, als, usr = hit["popular"], hit["als"], hit["user_knn"]                  # the "population": 943 people and what each method would do for them
rng = np.random.default_rng(0)
SIMS = 2000


def z_stat(hA, nA, hB, nB):                                                  # pooled two-proportion z statistic
    p = (hA + hB) / (nA + nB)
    return (hB / nB - hA / nA) / np.sqrt(p * (1 - p) * (1 / nA + 1 / nB) + 1e-300)


def experiment(a, b, n_total, sims=SIMS):
    """sims fake A/B tests: draw n_total people (with replacement) from the 943, give the first half method a and the second half method b"""
    idx = rng.integers(len(a), size=(sims, n_total)); h = n_total // 2
    return a[idx[:, :h]].sum(1), h, b[idx[:, h:2 * h]].sum(1), h


def power(a, b, n_total):
    return np.mean(np.abs(z_stat(*experiment(a, b, n_total))) > 1.96)


def need_per_arm(pa, pb):                                                    # formula for 80% power at the 5% level
    return 2 * (1.96 + 0.8416) ** 2 * ((pa + pb) / 2) * (1 - (pa + pb) / 2) / (pb - pa) ** 2


print(f"true hit rates in the 943 people: popular {pop.mean():.4f}, user-kNN {usr.mean():.4f}, ALS {als.mean():.4f}")
print("A/A test (ALS against ALS, 2,000 people): 'significant' in", f"{np.mean(np.abs(z_stat(*experiment(als, als, 2000))) > 1.96):.1%}", f"of {SIMS} fake experiments")
for name, a, b, sizes in (("popular -> ALS", pop, als, (943, 1452, 2000, 4000)), ("user-kNN -> ALS", usr, als, (943, 4000, 8000, 14366, 16000))):
    d = b.mean() - a.mean()
    print(f"\n{name}: true gap {d:+.4f}; formula says {need_per_arm(a.mean(), b.mean()):,.0f} people per arm ({2 * need_per_arm(a.mean(), b.mean()):,.0f} in total) for 80% power")
    for n in sizes:                                                           # the list includes the size the formula asks for
        print(f"   {n:>6,} people in total: the test finds the gap in {power(a, b, n):.1%} of {SIMS} fake experiments")

d = als - usr                                                                # the same people tried with both methods (possible offline, not in a live A/B test)
print(f"\nuser-kNN -> ALS if the same people see both methods (a paired test): about {(1.96 + 0.8416) ** 2 * d.var() / d.mean() ** 2:,.0f} people for 80% power, against {2 * 7183:,} in two separate arms")
```

2. **Run it:** `python ab_lab.py` (a few seconds). Output:

```text
true hit rates in the 943 people: popular 0.0498, user-kNN 0.0742, ALS 0.0870
A/A test (ALS against ALS, 2,000 people): 'significant' in 5.7% of 2000 fake experiments

popular -> ALS: true gap +0.0371; formula says 726 people per arm (1,452 in total) for 80% power
      943 people in total: the test finds the gap in 62.5% of 2000 fake experiments
    1,452 people in total: the test finds the gap in 80.2% of 2000 fake experiments
    2,000 people in total: the test finds the gap in 91.2% of 2000 fake experiments
    4,000 people in total: the test finds the gap in 99.7% of 2000 fake experiments

user-kNN -> ALS: true gap +0.0127; formula says 7,183 people per arm (14,366 in total) for 80% power
      943 people in total: the test finds the gap in 12.5% of 2000 fake experiments
    4,000 people in total: the test finds the gap in 31.1% of 2000 fake experiments
    8,000 people in total: the test finds the gap in 56.8% of 2000 fake experiments
   14,366 people in total: the test finds the gap in 81.2% of 2000 fake experiments
   16,000 people in total: the test finds the gap in 84.7% of 2000 fake experiments

user-kNN -> ALS if the same people see both methods (a paired test): about 4,721 people for 80% power, against 14,366 in two separate arms
```

3. **Read it.** Each "finds the gap" percentage is **power**, the chance that the experiment catches a gap that is really there. It climbs with the number of people and falls with the size of the gap: the same method of testing needs 1,452 people for one gap and 14,366 for the other. The last line is the price of using separate groups.

## What went wrong, and the fix
- **The same experiment gave slightly different answers in two runs.** In my first run, 2,000 people for most-popular against ALS found the gap in 89.9%; after I changed which sizes were printed, the same size gave 91.2%. Nothing was wrong: the random draws changed, which shows the Monte Carlo noise (about 1 point at 2,000 fake experiments). All numbers above are from the final script, and the note's re-run matches them exactly because the random seed is fixed.
- **My first list of sizes was wasteful.** It went up to 32,000 people per experiment, which made a table of about half a gigabyte in memory and added nothing (the gap was found 99% of the time already). I cut the list to the sizes that matter.
- **A fair warning about the comparison with the paired test.** My first estimate of the paired size in [[Error Bars Lab]] (about 2,300 people) was for an interval that *just* excludes 0 (about 50% power), not for 80% power. The comparison here uses 80% power on both sides (4,721 against 14,366).
- **Nothing else broke.**

## What I did not run
- **A real A/B test.** No user saw any list, and a hit in held-out data is not a click.
- **Peeking, sequential tests or "always valid" methods** on this data (the toy in [[AB Testing|A/B Testing]] shows peeking on made-up rates).
- **Variance reduction** (using each person's earlier behaviour) and **interleaving**.
- **More than one metric,** or a guardrail metric, or a Bayesian analysis.
- **Interference or novelty effects.** Each fake person's outcome is fixed and independent of the others.
- **A second data set.** The crowd is one data set of 943 people, resampled with repeats, so a fake experiment of 16,000 people repeats each real person about 17 times.

## Re-run check
Both scripts were extracted from this note into fresh, empty folders (`ab_lab.py` with a copy of `ranks.npz` from [[Error Bars Lab]]) and run again. The output of `ab_lab.py` matched the block above number for number (the random seed is fixed), and the toy in [[AB Testing|A/B Testing]] matched too. Not tested: another machine or other library versions.

## Concepts used
- [[AB Testing|A/B Testing]] · [[Hypothesis Testing]] · [[Confidence Interval]] · [[Bootstrap]] · [[Normal Distribution]] · [[Recommender Systems]] · [[Retrieval Metrics]] · [[Evaluation Metrics]] · [[Monte Carlo]] · [[Recommender Shoot-out]] · [[Error Bars Lab]]

## Next ideas
- Add peeking to this data: look after every 500 people and see how many false wins an A/A test gives.
- Add a variance-reduction step and see how many people it saves.
- Simulate interleaving (each person sees items from both methods) and compare its power with separate groups.
- Replace the offline hit with a simulated click model.

## Sources
- My own code and runs (versions above). The sample-size formula and the z values 1.96 and 0.8416 are from memory; the simulation checked the formula (80.2% and 81.2% for the 80% promised). No other source was read.
- No images in this note.

## Flashcards
#flashcards/projects

#flashcards/projects/concept #flashcards/topic/project-classic-ml What was the setup of A/B Test Lab?
?
The 943 people of the recommender shoot-out were a pretend crowd (outcome 1 if the hidden movie is in the method's top 10). Thousands of fake A/B experiments drew people with replacement, gave half method A and half method B, and tested the gap with a two-proportion z-test; the share that found the gap is the power.

#flashcards/projects/concept #flashcards/topic/project-classic-ml How many people did a live-style test need to find most-popular to ALS, and did the formula match?
?
The true gap was +0.0371. The formula said 726 per group (1,452 in total) for 80% power; in simulation 1,452 people found the gap in 80.2% of 2,000 fake experiments (943 people: 62.5%; 2,000: 91.2%; 4,000: 99.7%).

#flashcards/projects/concept #flashcards/topic/project-classic-ml How many people did it need for user-kNN to ALS?
?
The true gap was +0.0127. The formula said 7,183 per group (14,366 in total); in simulation 14,366 people found the gap in 81.2%, while 943 people found it in 12.5%, 4,000 in 31.1%, 8,000 in 56.8% and 16,000 in 84.7%.

#flashcards/projects/concept #flashcards/topic/project-classic-ml What did the A/A test in A/B Test Lab show?
?
ALS against ALS on 2,000 people said "significant" in 5.7% of 2,000 fake experiments, close to the promised 5% (Monte Carlo error about 0.5 points). The test does not cry wolf when nothing real is going on.

#flashcards/projects/concept #flashcards/topic/project-classic-ml What does using separate groups cost compared with the same people, for user-kNN against ALS?
?
About 14,366 people in two separate groups against about 4,721 in a paired test, both for 80% power (my arithmetic). A live test cannot show one user both lists, so it pays about three times the people.

#flashcards/projects/concept #flashcards/topic/project-classic-ml Why could 943 people not separate ALS from user-kNN in Error Bars Lab?
?
The gap is 0.0127. A live A/B test of that size on 943 people would find it in only 12.5% of experiments; about 14,400 people are needed for 80% power. (The paired test needs about 4,700.)

#flashcards/projects/concept #flashcards/topic/project-classic-ml What did A/B Test Lab not run?
?
A real online test (no user saw a list; a hit is not a click); peeking or sequential tests on this data; variance reduction and interleaving; several metrics or a Bayesian analysis; interference or novelty effects; a second data set. People are resampled with repeats.

#flashcards/projects/cloze #flashcards/topic/project-classic-ml In A/B Test Lab the sample-size formula asked for ==14,366== people for the small gap and the simulation found the gap in about ==81%== of experiments at that size.
