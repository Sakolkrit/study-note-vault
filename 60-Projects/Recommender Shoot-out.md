---
tags: [project, recommender-systems, collaborative-filtering, matrix-factorization, numpy, movielens]
status: done
stack: Python 3.9.6, numpy 1.24.4, MovieLens 100K
goal: compare popularity, kNN and matrix factorization recommenders on two tasks, rating error and top-10 lists
---

# Recommender Shoot-out

> **In one line:** build six recommenders from scratch with numpy on the MovieLens 100K ratings, and test them on two different jobs (guess the rating, put the right movie in a top-10 list) to see that **the best method for one job is not the best for the other**.

## Goal
Done means: one script, `recsys.py`, that prints a rating-error table (RMSE) and a top-10 table (hit rate, NDCG) for the methods of [[Recommender Systems]], so the claims in [[Collaborative Filtering]] and [[Matrix Factorization]] come from numbers I measured.

## Result in short
- **Rating error (RMSE on the official `u1.test` split, lower is better):** global mean 1.1537; item mean 1.0334; user and item bias model 0.9614; **item-kNN on the bias residuals 0.9328**; biased matrix factorization by SGD 0.9393 (mean of 3 seeds: 0.9393, min 0.9385, max 0.9401).
- **Top-10 lists (each person's latest movie hidden, ranked against about 1,577 unrated movies, 943 people, one run):**

| Method | Hit rate @10 | NDCG @10 | Median rank |
|---|---|---|---|
| random | 0.005 | 0.003 | 782 |
| most popular | 0.050 | 0.025 | 297 |
| item-kNN (cosine, all items) | 0.059 | 0.028 | 206 |
| user-kNN (cosine, 50 nearest users) | 0.074 | 0.033 | 134 |
| implicit MF, ALS (k=32) | **0.087** | **0.040** | **114** |
| rating-trained MF (SGD), ranked | 0.022 | 0.010 | 677 |

- **The finding:** the MF trained to predict ratings was a good rating predictor (RMSE 0.939) and a **bad** ranker (hit rate 0.022, worse than popularity at 0.050). The same family trained on "who rated what" ranked best (0.087). I did not test why.
- **Honest limits:** one data set, one split, one run for the top-10 tests, no tuning, and no test of whether the differences between neighbouring methods (for example 0.074 against 0.087) are real. (Tested afterwards in [[Error Bars Lab]]: 0.074 against 0.087 is not separable, $p=0.25$.) With 943 people one hit is 0.001, and 0.013 is about 12 people.

## What you need
- Python 3.9.6, `numpy` 1.24.4 (nothing else; the code uses only `numpy` and the standard library).
- The MovieLens 100K data set (GroupLens, 943 users, 1,682 movies, 100,000 ratings from 1 to 5): `https://files.grouplens.org/datasets/movielens/ml-100k.zip`. I did not read its licence text in full; read the README inside the zip before reusing the data for anything beyond learning.

## Steps
1. **Get the data.** In an empty folder:

```bash
curl -L -o ml-100k.zip https://files.grouplens.org/datasets/movielens/ml-100k.zip
python -c "import zipfile; zipfile.ZipFile('ml-100k.zip').extractall('.')"
```

   You should see a folder `ml-100k` with `u.data`, `u1.base`, `u1.test` and others. `u.data` has four columns: user, movie, rating, time of rating.
2. **Save the script** as `recsys.py` next to the `ml-100k` folder. What it does, in the order it runs:
   - **Task A, rating prediction.** Train on `u1.base` (80,000 ratings), test on `u1.test` (20,000). Methods: the global mean; each movie's mean; a bias model (overall mean plus a shrunk person bias plus a shrunk movie bias); item-kNN on what is left after the biases (cosine between movies, 30 nearest); and biased matrix factorization with SGD (20 factors, 30 sweeps, 3 seeds). Predictions are clipped to 1 to 5.
   - **Task B, top-10.** For every person, hide their most recent rating (ties broken by movie id) and train on the rest (99,057 ratings). Rank the hidden movie among all movies the person has not rated. Methods: random, most popular (ties broken by tiny random noise), item-kNN and user-kNN on 0/1 data (cosine), implicit MF by ALS (32 factors, confidence boost 40, 10 sweeps), and the SGD rating MF from task A used as a ranker.

```python
"""Recommender shoot-out on MovieLens 100K (943 users, 1682 movies, 100,000 ratings of 1 to 5).
Task A, rating prediction: official split u1.base / u1.test, error = RMSE.
Task B, top-10 list: for every user the LATEST rated movie is held out; all of the user's other ratings are training data;
        the held-out movie is ranked against every movie the user has not rated (about 1,600 candidates); metrics HR@10, NDCG@10.
Pure numpy, no recommender library."""
import numpy as np, time

DIR = "ml-100k/"
raw = np.loadtxt(DIR + "u.data", dtype=int)                  # user, item, rating, timestamp (ids start at 1)
NU, NI = raw[:, 0].max(), raw[:, 1].max()


def to_matrix(rows):                                         # users x items, 0 = not rated
    R = np.zeros((NU, NI))
    R[rows[:, 0] - 1, rows[:, 1] - 1] = rows[:, 2]
    return R


# ---------------------------------------------------------------- Task A: rating prediction
def bias_model(R, lam=10):                                   # mu + b_item + b_user, shrunk toward 0 by lam
    M = R > 0; mu = R[M].mean()
    bi = (np.where(M, R - mu, 0).sum(0)) / (lam + M.sum(0))
    bu = (np.where(M, R - mu - bi, 0).sum(1)) / (lam + M.sum(1))
    return mu, bu, bi


def item_knn_rating(R, base, k=30):                          # predict = baseline + similarity-weighted residuals of the user's other movies
    mu, bu, bi = base; M = R > 0
    E = np.where(M, R - (mu + bu[:, None] + bi[None, :]), 0)  # residual after the bias model
    Nn = np.linalg.norm(E, axis=0) + 1e-9
    S = (E.T @ E) / np.outer(Nn, Nn); np.fill_diagonal(S, 0)  # cosine between item residual columns
    def predict(u, i):
        rated = np.where(M[u])[0]
        if len(rated) == 0: return mu + bu[u] + bi[i]
        s = S[i, rated]; top = np.argsort(-np.abs(s))[:k]; s, r = s[top], rated[top]
        return mu + bu[u] + bi[i] + (s * E[u, r]).sum() / (np.abs(s).sum() + 1e-9)
    return predict


def mf_sgd(train_rows, k=20, epochs=30, lr=0.01, reg=0.05, seed=0):  # biased matrix factorization trained by SGD
    rs = np.random.RandomState(seed)
    mu = train_rows[:, 2].mean(); P = 0.1 * rs.randn(NU, k); Q = 0.1 * rs.randn(NI, k); bu = np.zeros(NU); bi = np.zeros(NI)
    u_, i_, r_ = train_rows[:, 0] - 1, train_rows[:, 1] - 1, train_rows[:, 2].astype(float)
    for ep in range(epochs):
        for n in rs.permutation(len(r_)):
            u, i = u_[n], i_[n]
            e = r_[n] - (mu + bu[u] + bi[i] + P[u] @ Q[i])
            bu[u] += lr * (e - reg * bu[u]); bi[i] += lr * (e - reg * bi[i])
            pu = P[u].copy(); P[u] += lr * (e * Q[i] - reg * P[u]); Q[i] += lr * (e * pu - reg * Q[i])
    return mu, bu, bi, P, Q


def rmse(pred, rows): return float(np.sqrt(np.mean((pred - rows[:, 2]) ** 2)))


# ---------------------------------------------------------------- Task B: top-10 list
def leave_last_out(raw):
    test, train = [], []
    for u in range(1, NU + 1):
        rows = raw[raw[:, 0] == u]; rows = rows[np.lexsort((rows[:, 1], rows[:, 3]))]   # oldest first, ties by item id
        test.append(rows[-1]); train.extend(rows[:-1])
    return np.array(train), np.array(test)


def ranking_scores(score_fn, B, test):                       # B = binary train matrix; score_fn(u) -> scores for all items
    ranks = []
    for u, i in zip(test[:, 0] - 1, test[:, 1] - 1):
        s = score_fn(u).copy(); s[B[u] > 0] = -np.inf          # never recommend what the user already rated
        ranks.append(1 + int((s > s[i]).sum()))                # rank of the held-out movie (1 = best); ties count in the item's favour
    ranks = np.array(ranks)
    return (ranks <= 10).mean(), np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0).mean(), float(np.median(ranks))


def wrmf(B, k=32, alpha=40, reg=0.1, iters=10, seed=0):     # implicit-feedback matrix factorization by alternating least squares
    rs = np.random.RandomState(seed); U = 0.01 * rs.randn(NU, k); V = 0.01 * rs.randn(NI, k); I = np.eye(k)
    for _ in range(iters):
        for X, Y, M in ((U, V, B), (V, U, B.T)):             # solve each row of X with Y fixed
            YtY = Y.T @ Y
            for r in range(X.shape[0]):
                m = M[r] > 0; Ym = Y[m]
                A = YtY + alpha * Ym.T @ Ym + reg * I          # confidence is 1 for unrated pairs, 1+alpha for rated pairs
                X[r] = np.linalg.solve(A, (1 + alpha) * Ym.sum(0))
    return U, V


if __name__ == "__main__":
    # ---------- Task A
    base_rows, test_rows = np.loadtxt(DIR + "u1.base", dtype=int), np.loadtxt(DIR + "u1.test", dtype=int)
    R = to_matrix(base_rows); ut, it = test_rows[:, 0] - 1, test_rows[:, 1] - 1
    print(f"Task A: train {len(base_rows)} ratings, test {len(test_rows)} ratings")
    res = {}
    mu = base_rows[:, 2].mean(); res["global mean"] = rmse(np.full(len(test_rows), mu), test_rows)
    M = R > 0; im = np.where(M.sum(0) > 0, R.sum(0) / np.maximum(M.sum(0), 1), mu)
    res["item mean"] = rmse(im[it], test_rows)
    base = bias_model(R); mu_, bu, bi = base
    res["bias model (mu + user + item)"] = rmse(np.clip(mu_ + bu[ut] + bi[it], 1, 5), test_rows)
    pk = item_knn_rating(R, base)
    res["item-kNN on residuals (k=30)"] = rmse(np.clip([pk(u, i) for u, i in zip(ut, it)], 1, 5), test_rows)
    runs = []
    for s in range(3):
        t = time.time(); m_, bu2, bi2, P, Q = mf_sgd(base_rows, seed=s)
        runs.append(rmse(np.clip(m_ + bu2[ut] + bi2[it] + (P[ut] * Q[it]).sum(1), 1, 5), test_rows)); print(f"  mf_sgd seed {s}: {runs[-1]:.4f} ({time.time()-t:.0f}s)", flush=True)
    res["matrix factorization, SGD (k=20)"] = float(np.mean(runs))
    for k_, v in res.items(): print(f"  {k_:34s} RMSE {v:.4f}")
    print(f"  mf_sgd over 3 seeds: mean {np.mean(runs):.4f}, min {min(runs):.4f}, max {max(runs):.4f}")

    # ---------- Task B
    train, test = leave_last_out(raw); B = to_matrix(train); Bb = (B > 0).astype(float)
    print(f"\nTask B: train {len(train)} ratings, {len(test)} held-out (one latest movie per user); candidates per user about {NI - int(Bb.sum(1).mean())}")
    rs = np.random.RandomState(0)
    pop = Bb.sum(0)
    Sim = (Bb.T @ Bb) / (np.sqrt(np.outer(pop, pop)) + 1e-9); np.fill_diagonal(Sim, 0)     # item-item cosine on 0/1 data
    Su = (Bb @ Bb.T) / (np.sqrt(np.outer(Bb.sum(1), Bb.sum(1))) + 1e-9); np.fill_diagonal(Su, 0)
    methods = {"random": lambda u: rs.rand(NI), "most popular": lambda u: pop + 1e-6 * rs.rand(NI),
               "item-kNN (cosine, all items)": lambda u: Bb[u] @ Sim,
               "user-kNN (cosine, 50 nearest users)": lambda u: (lambda w: w @ Bb)(np.where(Su[u] >= np.sort(Su[u])[-50], Su[u], 0))}
    U, V = wrmf(Bb); methods["implicit MF, ALS (k=32)"] = lambda u: V @ U[u]
    m_, bu2, bi2, P, Q = mf_sgd(train, seed=0)                # the rating-trained MF from Task A, now asked to rank
    methods["rating-trained MF (SGD), ranked"] = lambda u: m_ + bu2[u] + bi2 + Q @ P[u]
    print(f"  {'method':38s} {'HR@10':>6s} {'NDCG@10':>8s} {'median rank':>12s}")
    for name, fn in methods.items():
        hr, nd, med = ranking_scores(fn, B, test); print(f"  {name:38s} {hr:6.3f} {nd:8.3f} {med:12.0f}", flush=True)
```

3. **Run it** (`python recsys.py`). On my laptop, which was busy with other jobs, it took several minutes; the three SGD runs took 94 to 106 seconds each. Output:

```text
Task A: train 80000 ratings, test 20000 ratings
  mf_sgd seed 0: 0.9393 (106s)
  mf_sgd seed 1: 0.9385 (94s)
  mf_sgd seed 2: 0.9401 (94s)
  global mean                        RMSE 1.1537
  item mean                          RMSE 1.0334
  bias model (mu + user + item)      RMSE 0.9614
  item-kNN on residuals (k=30)       RMSE 0.9328
  matrix factorization, SGD (k=20)   RMSE 0.9393
  mf_sgd over 3 seeds: mean 0.9393, min 0.9385, max 0.9401

Task B: train 99057 ratings, 943 held-out (one latest movie per user); candidates per user about 1577
  method                                  HR@10  NDCG@10  median rank
  random                                  0.005    0.003          782
  most popular                            0.050    0.025          297
  item-kNN (cosine, all items)            0.059    0.028          206
  user-kNN (cosine, 50 nearest users)     0.074    0.033          134
  implicit MF, ALS (k=32)                 0.087    0.040          114
  rating-trained MF (SGD), ranked         0.022    0.010          677
```

4. **Read the two tables separately.** In task A, the learned models beat the bias model, and item-kNN edged out MF without any tuning. In task B, the order changes completely: the rating-trained MF falls below popularity.
5. **The toy versions** used in the concept notes (5 people, 6 movies) are inside [[Collaborative Filtering]], [[Matrix Factorization]] and [[Recommender Systems]]. They were run separately, and their code is in those notes.

## What went wrong, and the fix
- **The download was slow** (several minutes for a 5 MB file); the first command was moved to the background and I waited for it. Nothing to fix in the code.
- **The code ran without errors the first time.** Nothing in this project needed debugging, which does not mean it is bug-free. What I checked: the shapes (943 by 1,682), that ranks and metrics land in the expected range, and that random gets about 10/1,577 = 0.006 hit rate (it got 0.005).
- **Ties in rank.** If a method gives many movies the same score, the position of the hidden movie is ambiguous. The code counts only movies that score strictly higher than it, which favours the hidden movie in a tie. For popularity I broke ties with tiny random noise so many tied movies do not all look equal. I did not measure how much this changes any number.
- **Not an error but a trap:** a per-person "latest rating" hides one movie, but the other people's later ratings stay in the training data, so the model sees some of the "future". A strict global time split would be cleaner and I did not do one.

## What I did not run
- **No tuning** of the number of factors, penalty, learning rate, neighbours, confidence boost or number of sweeps. All values were chosen once.
- **No second split, no repeated seeds** for the top-10 tests (the SGD seeds were repeated only for the rating test), **no confidence intervals**, no significance test.
- **No deep or neural recommender** (two-tower, neural collaborative filtering), no BPR, no content features (movie genres are in the data and were not used).
- **No cold-start test** (new people or new movies) and **no live test** (A/B test).
- **The reason** the rating MF ranks badly was not tested.
- **Re-run check:** the code of step 2 was extracted from this note, saved in a fresh folder and run again with a copy of the data. Its output matched the output printed in step 3 apart from the seconds.

## Concepts used
- [[Recommender Systems]] · [[Collaborative Filtering]] · [[Matrix Factorization]] · [[Cosine Similarity]] · [[k-NN]] · [[Embedding]] · [[Retrieval Metrics]] · [[Stochastic Gradient Descent]] · [[Regularization]] · [[Overfitting]] · [[Singular Value Decomposition]]
- [[Error Bars Lab]] · [[Confidence Interval]]
- [[AB Test Lab|A/B Test Lab]]
- [[Cold Start Problem]] · [[Two-Tower Model]]
## Next ideas
- Test whether the top-10 gaps are real: repeat with 5 random "hide one movie" draws and report means with spread.
- Use a global time split instead of leave-last-out, and measure the leak.
- Train the MF for ranking (a pairwise loss such as BPR) and see if the rating MF's weakness goes away.
- Tune $k$ and the penalty for every method with a separate validation split.
- Add movie genres and try a content-based or hybrid method for cold start.

## Flashcards
#flashcards/projects

#flashcards/projects/concept #flashcards/topic/project-classic-ml What was the main finding of the Recommender Shoot-out?
?
The best method for rating error is not the best for top-10 lists. Biased MF trained on ratings had RMSE 0.939 but a top-10 hit rate of 0.022, below most popular (0.050). The same idea trained on implicit "who rated what" data reached 0.087. The cause was not tested.

#flashcards/projects/concept #flashcards/topic/project-classic-ml How was the top-10 test built?
?
Each person's latest rated movie was hidden (943 people), the model trained on the other 99,057 ratings, and the hidden movie was ranked against all movies the person had not rated (about 1,577). Metrics: hit rate at 10 and NDCG at 10.

#flashcards/projects/concept #flashcards/topic/project-classic-ml What were the RMSEs of the four rating baselines and MF in the shoot-out?
?
Global mean 1.1537, item mean 1.0334, bias model 0.9614, item-kNN on residuals 0.9328, biased MF by SGD 0.9393 (3 seeds: 0.9385 to 0.9401). Official u1 split; no tuning.

#flashcards/projects/concept #flashcards/topic/project-classic-ml Why is the leave-last-out test slightly unfair, and what would be cleaner?
?
Other people's later ratings stay in the training data, so the model sees part of the future. A strict global time split would be cleaner. I did not do one.

#flashcards/projects/concept #flashcards/topic/project-classic-ml Name three things the shoot-out did not run.
?
No tuning of any setting; no repeats or significance test for the top-10 numbers; no neural or BPR model, no genre features, no cold-start test and no A/B test.

#flashcards/projects/concept #flashcards/topic/project-classic-ml What would a fair chance level be for the top-10 test, and what did random score?
?
About 10 hidden-movie places out of about 1,577 candidates, so 10/1,577 = 0.006. Random scored 0.005 (my arithmetic).

#flashcards/projects/cloze #flashcards/topic/project-classic-ml In the shoot-out the rating-trained MF was a good ==rating predictor== and a poor ==ranker==.
