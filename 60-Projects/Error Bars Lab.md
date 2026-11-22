---
tags: [project, statistics, bootstrap, confidence-interval, hypothesis-testing, recommender, reinforcement-learning, numpy]
status: done
stack: Python 3.9.6, NumPy 1.24.4, SciPy 1.8.1 (only for one table value), PyTorch 2.8.0 CPU (to load the saved Snake networks)
goal: put error bars and tests on two earlier projects (the recommender top-10 table and the Snake DQN scores) to find out which of their small gaps are bigger than luck
---

# Error Bars Lab

> **In one line:** go back to two earlier projects whose notes said "I did not test whether these gaps are real", and put a [[Confidence Interval]] (by formula and by the [[Bootstrap]]) and a test ([[Hypothesis Testing]]) on each gap, then check by simulation that the intervals can be trusted.

## Goal
Done means: (1) the per-person results of the six recommenders in [[Recommender Shoot-out]] are rebuilt and match its table, (2) every hit rate has a 95% interval and every gap between neighbouring methods has an interval and an exact test for the **same people**, (3) the Snake players of [[Snake DQN]] are compared with the same tools, (4) a simulation with a known true value says how often my intervals contain it, and (5) what the intervals do **not** cover is written down.

## Result in short
- **Recommenders, 943 people, hit rate @10 with a 95% Wilson interval:**

| Method | Hit rate | 95% interval |
|---|---|---|
| random | 0.005 | 0.002 to 0.012 |
| most popular | 0.050 | 0.038 to 0.066 |
| item-kNN | 0.059 | 0.046 to 0.076 |
| user-kNN | 0.074 | 0.059 to 0.093 |
| implicit MF (ALS) | 0.087 | 0.071 to 0.107 |
| rating-trained MF (SGD) | 0.022 | 0.015 to 0.034 |

- **The question the shoot-out left open:** is ALS (0.087) really better than user-kNN (0.074)? The paired difference is +0.013 with a 95% interval of −0.007 to +0.033 and an exact sign test of $p=0.25$. **The data cannot separate them.** By my arithmetic, about 2,300 people would be needed to pin down a gap of that size, if the gap and the share of people who differ stayed the same.
- **Six comparisons, Bonferroni cut-off 0.05/6 = 0.0083.** Three pass: ALS beats item-kNN (+0.028, $p=0.0067$), ALS beats most-popular (+0.037, $p=0.0016$), and **most-popular beats the rating-trained MF** (+0.028, $p=0.0013$), which confirms the shoot-out's surprising finding. Three do not: user-kNN against item-kNN (+0.015, $p=0.054$), item-kNN against most-popular (+0.010, $p=0.37$), and ALS against user-kNN.
- **Overlapping intervals can hide a real gap.** ALS (0.071 to 0.107) and item-kNN (0.046 to 0.076) overlap, yet the interval for their paired difference (+0.008 to +0.047) excludes 0. Comparing the same people method against method is sharper than comparing two separate intervals.
- **Snake, 200 test games:** the rule player scores 14.85 (95% interval 14.21 to 15.52). Against it, the five full DQNs gave +1.17 (seed 0, $p=0.013$), +0.59, +0.23, +0.53 and +0.52 (seeds 1 to 4, $p$ from 0.21 to 0.61). Only seed 0 is separate from the rule on its own. **Averaging the five networks game by game: +0.61, interval +0.02 to +1.18, $p=0.046$**: right on the border. I would not call "the DQN beats the rule" settled.
- **The four DQN versions** (with or without replay and target network) do not differ at the end: exact permutation tests on 5 + 5 run means gave $p=0.52$, 0.87 and 0.65. With only 5 runs per version the smallest possible $p$ is 2/252 = 0.008, so this only rules out large differences.
- **Do the intervals deserve their "95%"?** In a simulation with a true hit rate of 0.087 and 943 people (2,000 repeats), the normal, Wilson and bootstrap intervals contained the truth in 93.5%, 93.8% and 93.4% of repeats: slightly under 95% (Monte Carlo error about 0.5 points), and the bootstrap was not better than the formulas.
- **Not done:** no error bars on the anomaly, drift, RAG or forecasting results; no power calculation; the intervals cover only the luck of which people or games were tested (see "What the intervals do not cover").

## What you need
- Python 3.9.6, `numpy` 1.24.4, `scipy` 1.8.1 (used once, for the $t$ table value), `torch` 2.8.0 (CPU; only to load the saved Snake networks). My existing environment on Windows; no fresh virtual environment.
- **Files from the two earlier projects, in the same folder:**
  - from [[Recommender Shoot-out]]: `recsys.py` and the folder `ml-100k` (the MovieLens 100K data);
  - from [[Snake DQN]]: `snake.py`, `dqn.py`, the folder `nets` (the 20 saved networks, written by its `run_all.py`) and its `results.json`, which I copied here as `snake_results.json`.
- About 6 minutes for `ranks.py` (it retrains the two matrix-factorization models); everything else takes seconds.

## Steps
1. **Rebuild the per-person results of the recommender test.** `ranks.py` repeats the six methods of the shoot-out in the same order with the same seeds and saves, for every person, the rank of the one hidden movie. It prints the hit rate and median rank so they can be compared with the shoot-out's table. They match (0.005, 0.050, 0.059, 0.074, 0.087, 0.022 and the same median ranks).

```python
# file: ranks.py
import numpy as np
from recsys import raw, NI, leave_last_out, to_matrix, wrmf, mf_sgd

train, test = leave_last_out(raw); B = to_matrix(train); Bb = (B > 0).astype(float)
rs = np.random.RandomState(0); pop = Bb.sum(0)
Sim = (Bb.T @ Bb) / (np.sqrt(np.outer(pop, pop)) + 1e-9); np.fill_diagonal(Sim, 0)
Su = (Bb @ Bb.T) / (np.sqrt(np.outer(Bb.sum(1), Bb.sum(1))) + 1e-9); np.fill_diagonal(Su, 0)
U, V = wrmf(Bb); m_, bu2, bi2, P, Q = mf_sgd(train, seed=0)
methods = {                                        # the same six methods, in the same order, as in the shoot-out
    "random": lambda u: rs.rand(NI),
    "popular": lambda u: pop + 1e-6 * rs.rand(NI),
    "item_knn": lambda u: Bb[u] @ Sim,
    "user_knn": lambda u: (lambda w: w @ Bb)(np.where(Su[u] >= np.sort(Su[u])[-50], Su[u], 0)),
    "als": lambda u: V @ U[u],
    "mf_sgd": lambda u: m_ + bu2[u] + bi2 + Q @ P[u]}
ranks = {}
for name, fn in methods.items():                   # rank of each person's held-out movie (1 = best)
    r = []
    for u, i in zip(test[:, 0] - 1, test[:, 1] - 1):
        s = fn(u).copy(); s[B[u] > 0] = -np.inf
        r.append(1 + int((s > s[i]).sum()))
    ranks[name] = np.array(r)
    print(f"{name:10s} people {len(r)}  hit@10 {np.mean(ranks[name] <= 10):.3f}  median rank {np.median(ranks[name]):.0f}", flush=True)
np.savez("ranks.npz", **ranks)
```

```text
random     people 943  hit@10 0.005  median rank 782
popular    people 943  hit@10 0.050  median rank 297
item_knn   people 943  hit@10 0.059  median rank 206
user_knn   people 943  hit@10 0.074  median rank 134
als        people 943  hit@10 0.087  median rank 114
mf_sgd     people 943  hit@10 0.022  median rank 677
```

2. **Write the small statistics helpers** ([[Confidence Interval]], [[Bootstrap]], [[Hypothesis Testing]]): a percentile bootstrap interval; the Wilson interval for a share; the exact sign test; a paired sign-flip permutation test; and an exact permutation test for two groups of means.

```python
# file: stats_lib.py
import itertools
from math import comb
import numpy as np


def boot_ci(x, stat=np.mean, B=10000, level=0.95, seed=0):
    """percentile bootstrap interval for stat(x): resample the items of x with replacement B times"""
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    idx = rng.integers(len(x), size=(B, len(x)))
    reps = x[idx].mean(axis=1) if stat is np.mean else np.array([stat(x[i]) for i in idx])
    a = (1 - level) / 2
    return tuple(float(v) for v in np.percentile(reps, [100 * a, 100 * (1 - a)]))


def wilson(hits, n, z=1.96):
    """Wilson score interval for a proportion"""
    p = hits / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return c - h, c + h


def sign_test(b, c):
    """exact two-sided p for 'b people hit only with X, c only with Y': is the split 50/50?"""
    n = b + c
    return min(1.0, 2 * sum(comb(n, k) for k in range(max(b, c), n + 1)) / 2 ** n) if n else 1.0


def sign_flip_test(d, B=20000, seed=0):
    """paired permutation test: under 'no difference' each paired difference is as likely to be + as -"""
    d = np.asarray(d, float); rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(B, len(d))) * d
    return (np.sum(np.abs(flips.mean(axis=1)) >= abs(d.mean()) - 1e-12) + 1) / (B + 1)


def exact_perm_test(a, b):
    """two-sided permutation test of the difference of two means: try every way to split the pooled values"""
    allv = np.concatenate([a, b]); n = len(a); obs = abs(np.mean(a) - np.mean(b)); cnt = tot = 0
    for idx in itertools.combinations(range(len(allv)), n):
        m = np.zeros(len(allv), bool); m[list(idx)] = True
        cnt += abs(allv[m].mean() - allv[~m].mean()) >= obs - 1e-12; tot += 1
    return cnt / tot
```

3. **Intervals and tests for the recommenders.** For every method it prints the hit rate with a Wilson and a bootstrap interval, and NDCG with a bootstrap interval. Then, for six pairs of methods **chosen by me after seeing the table** (the neighbours, plus the two extremes), the paired difference in hit rate with its bootstrap interval, the numbers of people who hit with only one method and the exact sign test. The bootstrap resamples **people** (10,000 times), keeping both methods' results for a person together.

```python
# file: recsys_ci.py
import numpy as np
from stats_lib import boot_ci, wilson, sign_test

R = np.load("ranks.npz"); names = ["random", "popular", "item_knn", "user_knn", "als", "mf_sgd"]
hit = {k: (R[k] <= 10).astype(float) for k in names}                               # 1 if the held-out movie is in the top 10
ndcg = {k: np.where(R[k] <= 10, 1 / np.log2(R[k] + 1), 0.0) for k in names}
print("people:", len(hit["als"]), "\n")
print(f"{'method':9s} {'hit@10':>6s}  {'Wilson 95%':>15s}  {'bootstrap 95%':>15s}   {'NDCG@10':>7s}  {'bootstrap 95%':>15s}")
for k in names:
    h, w, b, nb = hit[k], wilson(int(hit[k].sum()), len(hit[k])), boot_ci(hit[k]), boot_ci(ndcg[k])
    print(f"{k:9s} {h.mean():6.3f}  [{w[0]:.3f}, {w[1]:.3f}]  [{b[0]:.3f}, {b[1]:.3f}]   {ndcg[k].mean():7.3f}  [{nb[0]:.3f}, {nb[1]:.3f}]")
print("\npaired differences in hit@10 (the same people for both methods; 6 comparisons, so the Bonferroni cut-off is 0.05/6 = 0.0083)")
for a, b in [("als", "user_knn"), ("user_knn", "item_knn"), ("item_knn", "popular"), ("als", "item_knn"), ("als", "popular"), ("popular", "mf_sgd")]:
    d = hit[a] - hit[b]; lo, hi = boot_ci(d)
    oa, ob = int(((hit[a] == 1) & (hit[b] == 0)).sum()), int(((hit[b] == 1) & (hit[a] == 0)).sum())
    print(f"{a:8s} - {b:8s} {d.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]   only {a}: {oa:3d}  only {b}: {ob:3d}   exact sign test p = {sign_test(oa, ob):.4f}")

b, c, N = int(((hit["als"] == 1) & (hit["user_knn"] == 0)).sum()), int(((hit["user_knn"] == 1) & (hit["als"] == 0)).sum()), len(hit["als"])
d = (b - c) / N; var = (b + c) / N - d * d              # mean and variance of one person's paired difference (+1, 0 or -1)
print(f"\nals vs user_knn: gap {d:.4f}; if the gap and the share of people who differ stayed the same, about {1.96 ** 2 * var / d ** 2:.0f} people would give a 95% interval that just excludes 0")
```

```text
people: 943 

method    hit@10       Wilson 95%    bootstrap 95%   NDCG@10    bootstrap 95%
random     0.005  [0.002, 0.012]  [0.001, 0.011]     0.003  [0.000, 0.006]
popular    0.050  [0.038, 0.066]  [0.036, 0.065]     0.025  [0.018, 0.034]
item_knn   0.059  [0.046, 0.076]  [0.045, 0.075]     0.028  [0.021, 0.036]
user_knn   0.074  [0.059, 0.093]  [0.058, 0.091]     0.033  [0.025, 0.041]
als        0.087  [0.071, 0.107]  [0.069, 0.105]     0.040  [0.031, 0.049]
mf_sgd     0.022  [0.015, 0.034]  [0.014, 0.032]     0.010  [0.006, 0.014]

paired differences in hit@10 (the same people for both methods; 6 comparisons, so the Bonferroni cut-off is 0.05/6 = 0.0083)
als      - user_knn +0.013  [-0.007, +0.033]   only als:  52  only user_knn:  40   exact sign test p = 0.2513
user_knn - item_knn +0.015  [+0.001, +0.029]   only user_knn:  30  only item_knn:  16   exact sign test p = 0.0541
item_knn - popular  +0.010  [-0.010, +0.029]   only item_knn:  45  only popular:  36   exact sign test p = 0.3742
als      - item_knn +0.028  [+0.008, +0.047]   only als:  56  only item_knn:  30   exact sign test p = 0.0067
als      - popular  +0.037  [+0.015, +0.059]   only als:  76  only popular:  41   exact sign test p = 0.0016
popular  - mf_sgd   +0.028  [+0.012, +0.045]   only popular:  44  only mf_sgd:  18   exact sign test p = 0.0013

als vs user_knn: gap 0.0127; if the gap and the share of people who differ stayed the same, about 2311 people would give a 95% interval that just excludes 0
```

4. **Read it.** The six intervals for the hit rate overlap in a chain: random is clearly lowest, rating-MF is clearly low, and the four in the middle are hard to order. The bootstrap and Wilson intervals are almost the same. In the paired table the "only X / only Y" counts show why: ALS against user-kNN is 52 people against 40, only 12 apart, among 92 people who differ at all.

5. **Intervals and tests for Snake.** It replays the 200 test games for the rule player and for each of the five saved full DQNs, compares each network with the rule game by game, then averages the five networks game by game. It also gives a $t$ interval for the five training runs of each version and exact permutation tests between versions (the five run means of one version against the five of another):

```python
# file: snake_ci.py
import json
import numpy as np
import torch
from scipy import stats
from snake import Snake, rule_greedy
from dqn import make_net, greedy
from stats_lib import boot_ci, sign_flip_test, exact_perm_test


def scores(policy):                                  # the 200 test games of the Snake DQN project
    env, out = Snake(), []
    for s in range(1000, 1200):
        o, done = env.reset(seed=s), False
        while not done:
            o, _, done = env.step(policy(o))
        out.append(env.score)
    return np.array(out, float)


rule = scores(rule_greedy)
print(f"rule player: mean {rule.mean():.2f}, bootstrap 95% interval [{boot_ci(rule)[0]:.2f}, {boot_ci(rule)[1]:.2f}] (resampling the 200 games)\n")
print("A. each full DQN against the rule on the same 200 games (paired differences)")
nets = []
for seed in range(5):
    net = make_net(); net.load_state_dict(torch.load(f"nets/full_DQN_{seed}.pt")); nets.append(scores(greedy(net)))
    d = nets[-1] - rule; lo, hi = boot_ci(d)
    print(f"   seed {seed}: mean {nets[-1].mean():.2f}, minus rule {d.mean():+.2f}  [{lo:+.2f}, {hi:+.2f}]  sign-flip p = {sign_flip_test(d):.3f}")
d = np.mean(nets, axis=0) - rule; lo, hi = boot_ci(d)
print(f"   average of the 5 networks, game by game: {d.mean():+.2f}  [{lo:+.2f}, {hi:+.2f}]  sign-flip p = {sign_flip_test(d):.3f}")

res = json.load(open("snake_results.json")); seeds = {}
for name, seed, mean, curve, secs in res:
    seeds.setdefault(name, []).append(mean)
t4 = stats.t.ppf(0.975, 4)
print(f"\nB. the 5 training runs of each version (final mean score of each run); t interval uses t(4 df, 97.5%) = {t4:.3f}")
for name, v in seeds.items():
    v = np.array(v); h = t4 * v.std(ddof=1) / 5 ** 0.5
    print(f"   {name:18s} mean {v.mean():.2f}  [{v.mean() - h:.2f}, {v.mean() + h:.2f}]")
print("   exact permutation test on the 5 + 5 run means (all 252 splits):")
for other in ("no target network", "no replay", "neither"):
    print(f"   full DQN vs {other:18s} difference {np.mean(seeds['full DQN']) - np.mean(seeds[other]):+.2f}  p = {exact_perm_test(seeds['full DQN'], seeds[other]):.3f}")
```

```text
rule player: mean 14.85, bootstrap 95% interval [14.21, 15.52] (resampling the 200 games)

A. each full DQN against the rule on the same 200 games (paired differences)
   seed 0: mean 16.02, minus rule +1.17  [+0.28, +2.07]  sign-flip p = 0.013
   seed 1: mean 15.45, minus rule +0.59  [-0.30, +1.50]  sign-flip p = 0.207
   seed 2: mean 15.07, minus rule +0.23  [-0.59, +1.04]  sign-flip p = 0.606
   seed 3: mean 15.38, minus rule +0.53  [-0.38, +1.41]  sign-flip p = 0.246
   seed 4: mean 15.37, minus rule +0.52  [-0.32, +1.33]  sign-flip p = 0.228
   average of the 5 networks, game by game: +0.61  [+0.02, +1.18]  sign-flip p = 0.046

B. the 5 training runs of each version (final mean score of each run); t interval uses t(4 df, 97.5%) = 2.776
   full DQN           mean 15.46  [15.03, 15.89]
   no target network  mean 15.69  [14.98, 16.39]
   no replay          mean 15.52  [14.59, 16.45]
   neither            mean 15.34  [14.83, 15.86]
   exact permutation test on the 5 + 5 run means (all 252 splits):
   full DQN vs no target network  difference -0.23  p = 0.516
   full DQN vs no replay          difference -0.07  p = 0.865
   full DQN vs neither            difference +0.12  p = 0.651
```

6. **Check the intervals against a known truth.** `coverage.py` simulates a hit rate of 0.087 over 943 people 2,000 times, builds three kinds of 95% interval each time and counts how often each contains 0.087. (For 0/1 data, a bootstrap resample's hit count is a binomial draw, so the code uses that shortcut.)

```python
# file: coverage.py
import numpy as np
from stats_lib import wilson

rng = np.random.default_rng(0)
p, n, trials, B = 0.087, 943, 2000, 2000                 # a "true" hit rate like the shoot-out's best method, and its number of people
cov = {"normal (Wald)": 0, "Wilson": 0, "bootstrap": 0}; width = {k: 0.0 for k in cov}
for _ in range(trials):
    h = int(rng.binomial(n, p)); ph = h / n; se = (ph * (1 - ph) / n) ** 0.5
    ci = {"normal (Wald)": (ph - 1.96 * se, ph + 1.96 * se), "Wilson": wilson(h, n),
          "bootstrap": tuple(np.percentile(rng.binomial(n, ph, size=B) / n, [2.5, 97.5]))}   # a bootstrap resample of 0/1 data is a binomial draw
    for k, (lo, hi) in ci.items():
        cov[k] += lo <= p <= hi; width[k] += hi - lo
print(f"true hit rate {p}, {n} people, {trials} repeats")
for k in cov:
    print(f"   {k:14s} covers the truth in {cov[k] / trials:.1%}, average width {width[k] / trials:.4f}")
```

```text
true hit rate 0.087, 943 people, 2000 repeats
   normal (Wald)  covers the truth in 93.5%, average width 0.0359
   Wilson         covers the truth in 93.8%, average width 0.0360
   bootstrap      covers the truth in 93.4%, average width 0.0358
```

## What the intervals do not cover
- **Training luck.** Each recommender was trained once (seed 0). A different random start of the ALS or SGD models would move the hit rates, and none of the intervals above include that. The Snake runs are different: five training runs per version exist, and the run-level intervals cover them, but the game-level intervals (steps 5, part A) do not.
- **Which test people.** The intervals assume the 943 people are independent draws from some larger crowd. They are really all the users of one data set, each represented by their one latest movie; the leave-last-out protocol is one choice of many.
- **Other data sets, other splits, other settings.** Nothing was tuned and nothing was repeated on a second data set.
- **The comparisons were picked after seeing the table.** I chose six pairs that looked interesting, and I corrected for six, but not for the choices I did not make. Treat the borderline results (user-kNN against item-kNN at $p=0.054$, the Snake average at $p=0.046$) as undecided either way.
- **Five networks, 200 games.** The five full DQNs shared the same 200 test games, so the "average of five" interval has the information of 200 games, not 1,000.

## What went wrong, and the fix
- **A trap the design avoids.** ALS and item-kNN have overlapping hit-rate intervals, so comparing two separate intervals would say "not separable". The paired test on the same people says otherwise (+0.028, $p=0.0067$). That is why the comparisons are paired from the start.
- **The coverage test came out under 95%.** I expected about 95% and got 93.4% to 93.8%. With 2,000 repeats the Monte Carlo error is about 0.5 points, so this is a small but real shortfall. I did not test why (my guess is that the hit count is a small whole number, so the true coverage of any interval for it jumps around, but I did not check).
- **The first Snake re-run used the saved networks, not new training.** Training the 20 networks again takes about 32 minutes; I reused the saved ones, so this project says nothing about luck in training beyond what the 5 runs per version already show.
- **Nothing else broke.** The per-person hit rates matched the shoot-out's table on the first try.

## What I did not run
- **Error bars for the other projects:** the anomaly shoot-out, the model-monitoring lab, the RAG tuning lab and the forecasting shoot-out.
- **Retraining many times** to bootstrap over training luck (too slow here).
- **A power calculation** before the tests, or any Bayesian alternative to the intervals.
- **Bias-corrected bootstrap intervals** (BCa), the bootstrap-$t$, or exact (Clopper-Pearson) intervals for shares.
- **The reason** the intervals are slightly too narrow in the coverage test.

## Re-run check
Every code block in this note was extracted into a fresh, empty folder (with copies of the input files from the two earlier projects) and run again. `ranks.py` (which retrains the two matrix-factorization models), `recsys_ci.py`, `snake_ci.py` and `coverage.py` all gave output identical to the blocks above, number for number, so the results are repeatable on this machine. The helper file `stats_lib.py` has no output of its own. The snippets in [[Confidence Interval]], [[Bootstrap]] and [[Hypothesis Testing]] were re-run the same way and matched. Not tested: another machine or other library versions. The saved Snake networks were reused, not retrained.

## Concepts used
- [[Confidence Interval]] · [[Bootstrap]] · [[Hypothesis Testing]] · [[Normal Distribution]] · [[Central Limit Theorem]] · [[Probability]] · [[Evaluation Metrics]] · [[Retrieval Metrics]] · [[Recommender Systems]] · [[Matrix Factorization]] · [[Collaborative Filtering]] · [[Deep Q-Network]] · [[Recommender Shoot-out]] · [[Snake DQN]]
- [[AB Test Lab|A/B Test Lab]] · [[AB Testing|A/B Testing]]
## Next ideas
- Retrain each recommender with 20 different seeds and see how large the training luck is compared with the test-people luck.
- Put the same paired tests on the [[RAG Tuning Lab]] (the 0.69 against 0.53 top-1 drop after reranking was measured on 49 questions).
- Compute the power of each test before running it.
- Try the Bayesian version of the comparisons ([[Bayes' Theorem]]).

## Sources
- My own code and runs (versions above). The recipes (percentile bootstrap, Wilson interval, exact sign test, permutation tests) are standard; I wrote them from memory and checked them by simulation and by hand-worked examples in the concept notes, not against a textbook.
- No images in this note.

## Flashcards
#flashcards/projects

#flashcards/projects/concept #flashcards/topic/project-classic-ml What was the setup of Error Bars Lab?
?
Per-person results of the six recommenders (943 people) and per-game results of the Snake players (200 games) were rebuilt. Hit rates got Wilson and bootstrap 95% intervals; gaps between methods got paired bootstrap intervals and exact sign tests on the same people; Snake got paired and permutation tests; a simulation with a known true rate checked the intervals.

#flashcards/projects/concept #flashcards/topic/project-classic-ml Is ALS really better than user-kNN in the recommender shoot-out?
?
Not provably. Hit rates 0.087 against 0.074: the paired difference is +0.013 with a 95% interval of −0.007 to +0.033 and an exact sign test of p = 0.25 (52 people hit only with ALS, 40 only with user-kNN). About 2,300 people would be needed for a gap that size (my arithmetic, same gap and same share of differing people).

#flashcards/projects/concept #flashcards/topic/project-classic-ml Which recommender comparisons passed the corrected cut-off of 0.0083?
?
ALS beats item-kNN (+0.028, p = 0.0067), ALS beats most-popular (+0.037, p = 0.0016) and most-popular beats the rating-trained MF (+0.028, p = 0.0013). User-kNN against item-kNN (p = 0.054), item-kNN against most-popular (p = 0.37) and ALS against user-kNN (p = 0.25) did not.

#flashcards/projects/concept #flashcards/topic/project-classic-ml Why can overlapping intervals hide a real gap between two methods?
?
Each interval includes the luck of which people were tested, but that luck is shared by both methods. The paired difference on the same people removes it. ALS (0.071 to 0.107) and item-kNN (0.046 to 0.076) overlap, yet the interval for their difference (+0.008 to +0.047) excludes 0.

#flashcards/projects/concept #flashcards/topic/project-classic-ml Does the Snake DQN beat the rule player, according to Error Bars Lab?
?
Not settled. Against the rule (14.85) one network was clearly higher (+1.17, p = 0.013) and four were not (p from 0.21 to 0.61). The five-network average gave +0.61 with an interval of +0.02 to +1.18 and p = 0.046, on the border. The networks share the same 200 games.

#flashcards/projects/concept #flashcards/topic/project-classic-ml What did the permutation tests between the four DQN versions show, and what are their limits?
?
Full DQN against no target network, no replay and neither: p = 0.52, 0.87 and 0.65, so no difference at 60,000 steps. With 5 runs per version the smallest possible p is 2/252 = 0.008, so only large differences could be detected.

#flashcards/projects/concept #flashcards/topic/project-classic-ml How often did the 95% intervals contain the truth in the coverage simulation?
?
True hit rate 0.087, 943 people, 2,000 repeats: normal 93.5%, Wilson 93.8%, bootstrap 93.4%. Slightly under 95% (Monte Carlo error about 0.5 points), and the bootstrap was no better than the formulas. Why was not tested.

#flashcards/projects/concept #flashcards/topic/project-classic-ml What do the intervals in Error Bars Lab not cover?
?
Training luck (each recommender was trained once), the choice of split and data set, the fact that the six comparisons were picked after seeing the table, and the fact that five networks shared the same 200 games. They cover only which test people or games were drawn.

#flashcards/projects/concept #flashcards/topic/project-classic-ml What did Error Bars Lab not run?
?
Error bars for the anomaly, monitoring, RAG and forecasting projects; retraining many times; power calculations or Bayesian versions; BCa, bootstrap-t or Clopper-Pearson intervals; and the reason for the slight under-coverage.

#flashcards/projects/cloze #flashcards/topic/project-classic-ml In Error Bars Lab the ==paired== difference on the same people was sharper than comparing two separate intervals, and ALS against user-kNN was ==not separable== from luck.
