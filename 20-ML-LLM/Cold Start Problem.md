---
tags: [recommender, cold-start, collaborative-filtering, machine-learning]
status: growing
level: 4
tier: 2
---

# Cold Start Problem

The **cold start problem** is what a [[Recommender Systems]] does badly when it has **no history to learn from**: a brand-new user nobody knows yet, or a brand-new item nobody has tried yet.

## 🟢 1. Simple (age 12)
A new kid joins your class. Nobody knows what snacks they like, so you cannot guess what to offer. Or the shop gets a new snack that nobody has tasted, so no friend can tell you whether it is any good. A recommender learns from **what people did before**; if there is nothing yet, it has nothing to go on.

**Example:** you open a music app for the first time. It has never seen you play a song, so all it can do is show you what is popular with everyone, and hope.

## 🔵 2. How it works
1. A [[Collaborative Filtering]] system learns from a big table: rows are people, columns are items, and a cell is filled when the person rated or used the item. Most of the table is empty.
2. A **new person** is an empty row, and a **new item** is an empty column. Methods that work from the table (neighbours, [[Matrix Factorization]]) have nothing to compare, so they cannot place the newcomer.
3. The usual ways out:
   - **Show what is popular.** Easy and a safe fallback, but the same for everyone.
   - **Ask.** Let a new person pick a few favourites when signing up.
   - **Use descriptions instead of history.** An item has features (genre, text, picture) and a person has some too (age, country, device). A model that reads features can place a newcomer. The [[Two-Tower Model]] is built this way.
   - **Use the first few actions fast.** After even a couple of ratings the system can start to adapt.
   - **Try new items on purpose.** Show a new item to a few people to gather feedback (the same idea as exploring in [[Epsilon-Greedy]]).
4. How to **see** cold start in your data: split the test results by how much history a person has, or by how popular an item is, and compare the scores (the code below does both).

*I found no free, on-topic picture of the cold start problem on Wikimedia Commons (a search returned only photos of an unrelated event), so this note has no picture.*

## 🟠 3. Technical detail
- **Two kinds, plus a third.** *User cold start* (new person), *item cold start* (new item), and *system cold start* (a new service with almost no data at all).
- **Why the table is hard.** In the data used here (MovieLens 100K, see [[Recommender Shoot-out]]) 93.7% of the person-movie table is empty. Learned methods lean on the few filled cells.
- **A popularity fallback needs no history,** so it gives the floor: the best a newcomer can get without any work.
- **Features break the dependence on history** (from general knowledge; I did not build one here, except in [[Two-Tower Model]], where the user tower reads a person's history).

The script ranks people and movies by how much history there is. It needs `recsys.py` and the folder `ml-100k/` of [[Recommender Shoot-out]] and the file `ranks.npz` of [[Error Bars Lab]] (the rank of each person's held-out movie for six methods) in the same folder:

```python
# file: cold_start.py
import numpy as np
from recsys import raw, NU, NI, leave_last_out, to_matrix

train, test = leave_last_out(raw); B = to_matrix(train); Bb = (B > 0).astype(float)
ranks = np.load("ranks.npz"); show = ["popular", "item_knn", "user_knn", "als"]
n_hist = Bb.sum(1)                                   # how many movies each person rated (training part)
pop = Bb.sum(0)                                      # how many people rated each movie (training part)
hidden_pop = pop[test[:, 1] - 1]                     # popularity of each person's held-out movie

print(f"ratings: {len(raw):,} of {NU * NI:,} possible pairs, so {100 * (1 - len(raw) / (NU * NI)):.1f}% of the person-movie table is empty")
print(f"movies with no training rating at all: {int((pop == 0).sum())}; held-out movies with no training rating: {int((hidden_pop == 0).sum())} of {len(test)}")

def table(title, groups, labels):
    print(title); print(f"  {'group':22s} {'people':>6s}  " + "  ".join(f"{m:>9s}" for m in show))
    for g, lab in zip(groups, labels):
        print(f"  {lab:22s} {int(g.sum()):6d}  " + "  ".join(f"{np.mean(ranks[m][g] <= 10):6.3f}({int((ranks[m][g] <= 10).sum()):2d})" for m in show))

q = np.quantile(n_hist, [1 / 3, 2 / 3]); print(f"history sizes: median {np.median(n_hist):.0f}, smallest {int(n_hist.min())}, largest {int(n_hist.max())}")
table("1. by how many movies the person had already rated", [n_hist <= q[0], (n_hist > q[0]) & (n_hist <= q[1]), n_hist > q[1]],
      [f"few (up to {q[0]:.0f})", f"middle ({q[0]:.0f} to {q[1]:.0f})", f"many (over {q[1]:.0f})"])
table("2. by how many people had rated the held-out movie", [hidden_pop < 20, (hidden_pop >= 20) & (hidden_pop < 100), hidden_pop >= 100],
      ["rare movie (under 20)", "middle (20 to 99)", "popular (100 or more)"])

# 3. a new person who has rated only k movies, item-kNN against popular; the k OLDEST ratings (their first ones) or the k MOST RECENT
Sim = (Bb.T @ Bb) / (np.sqrt(np.outer(pop, pop)) + 1e-9); np.fill_diagonal(Sim, 0)
order = np.lexsort((train[:, 1], train[:, 3])); tr = train[order]                       # oldest first
print("3. a new person who has rated only k movies, item-kNN hit@10 (candidates as before: every movie they have not rated)")
print(f"  popular (needs no history): {np.mean(ranks['popular'] <= 10):.3f}")
print(f"  {'k':>4s}  {'first k ratings':>16s}  {'latest k ratings':>16s}")
for k in (1, 3, 5, 10, 20, 50, 1000):
    res = []
    for pick in ("first", "latest"):
        hits = []
        for u, i in zip(test[:, 0] - 1, test[:, 1] - 1):
            items = tr[tr[:, 0] == u + 1][:, 1] - 1; keep = items[:k] if pick == "first" else items[-k:]
            s = Sim[keep].sum(0); s[items] = -np.inf                                      # all rated movies are excluded, as before
            hits.append(1 + int((s > s[i]).sum()) <= 10)
        res.append(np.mean(hits))
    print(f"  {k if k < 1000 else 'all':>4}  {res[0]:16.3f}  {res[1]:16.3f}")
```

```text
ratings: 100,000 of 1,586,126 possible pairs, so 93.7% of the person-movie table is empty
movies with no training rating at all: 3; held-out movies with no training rating: 3 of 943
history sizes: median 64, smallest 19, largest 736
1. by how many movies the person had already rated
  group                  people    popular   item_knn   user_knn        als
  few (up to 41)            321   0.047(15)   0.084(27)   0.090(29)   0.109(35)
  middle (41 to 113)        308   0.036(11)   0.081(25)   0.101(31)   0.107(33)
  many (over 113)           314   0.067(21)   0.013( 4)   0.032(10)   0.045(14)
2. by how many people had rated the held-out movie
  group                  people    popular   item_knn   user_knn        als
  rare movie (under 20)     128   0.000( 0)   0.000( 0)   0.000( 0)   0.016( 2)
  middle (20 to 99)         383   0.000( 0)   0.000( 0)   0.003( 1)   0.044(17)
  popular (100 or more)     432   0.109(47)   0.130(56)   0.160(69)   0.146(63)
3. a new person who has rated only k movies, item-kNN hit@10 (candidates as before: every movie they have not rated)
  popular (needs no history): 0.050
     k   first k ratings  latest k ratings
     1             0.045             0.146
     3             0.054             0.135
     5             0.055             0.119
    10             0.067             0.103
    20             0.062             0.080
    50             0.062             0.065
   all             0.059             0.059
```

Reading it (all from my own run on 943 people, one data set; the figures in brackets are the number of people whose held-out movie was in the top 10):
- **By history (table 1): no cold start here.** People with few ratings were not worse off than people with many (the learned methods score 0.084 to 0.109 for "few" and 0.013 to 0.045 for "many"). The reason is the data: MovieLens only keeps people with at least 20 ratings (from memory of the data set's description; the smallest training history here is 19, after one movie is hidden), so nobody is really cold. Why heavy users are harder I did not look into.
- **By item (table 2): cold start is clear.** When the held-out movie was rare (rated by under 20 people) the popularity list, item-kNN and user-kNN found **none of 128**, and ALS found 2. For movies rated by 20 to 99 people it was 0 and 0 and 1 of 383 for the first three, and 17 for ALS. For popular movies (100 or more raters) the methods found 47 to 69 of 432. The learned methods rarely recommend an item few people have used.
- **A newcomer who has only rated k movies (table 3).** With just the **first** ratings a person made, item-kNN gets 0.045 at k=1 (about the popularity score of 0.050), and 0.067 at k=10, then stays near 0.06. That is a small gain over popularity (differences of 10 to 20 people out of 943, so I do not call it firmly established). With the **latest** ratings it gets 0.146 at k=1 and falls to 0.059 as older ratings are added. In this test the movie hidden is the person's last one, so the most recent rating is a much better clue than the whole history. My guess is that people rate movies in runs of similar ones; I did not test that. It also means "keep the latest k ratings" is not a fair imitation of a new user.

## 🔴 4. Deep / formal
- **It is structural, not a bug.** A method that learns a vector for each person from their ratings has no data for an empty row. Without features there is nothing to learn from, so the fallback is a rule such as popularity.
- **A feedback loop (general knowledge, not measured here).** Items that few people use are shown less, so they collect fewer ratings, so they are shown less again. Table 2 shows the first half of that loop in this data: rare movies are almost never recommended.
- **Measuring it properly needs a test built for it.** The usual way is to hold out whole people (or whole items) so the model never sees any of their history, and score them separately from everyone else. My tables only **group** a normal test (the last movie of each person) by history or popularity, so they show the effect for items but not a true new-user test.
- **My numbers are limited:** one data set of 943 people, one held-out movie per person, some groups with only 128 people, and no error bars ([[Confidence Interval]] and [[Bootstrap]] would give them).
- **In practice** systems combine several ways out (popular fallback, onboarding questions, content features, a little exploration); which combination works is decided by testing ([[AB Testing|A/B Testing]]).

**What came before, and how this improved on it:**
- **Before:** the same simple rule for everyone, such as a "most popular" list, and then recommenders that learned only from the table of past actions.
- **Its limit:** nothing at all for a new person or a new item, and a bias against items with little history.
- **What changed:** hybrid systems that also use features of people and items, and models that read features, such as the [[Two-Tower Model]]. (The names and dates of the early systems are from memory; I do not give them.)
- **How much better:** in my own run, with no history the popularity list found the held-out movie in the top 10 for 5.0% of people. With only the single latest rating, item-kNN found 14.6%, almost three times as many, and with the first 10 ratings 6.7%. For rare movies no method except ALS found any (0 of 128, ALS 2 of 128).

## Related
- [[Recommender Systems]] · [[Collaborative Filtering]] · [[Matrix Factorization]] · [[Two-Tower Model]] · [[Embedding]] · [[Retrieval Metrics]] · [[Recommender Shoot-out]] · [[Error Bars Lab]] · [[Epsilon-Greedy]] · [[AB Testing|A/B Testing]]

## Sources
- My own code and runs (Python 3.9.6, NumPy 1.24.4, MovieLens 100K as in [[Recommender Shoot-out]]; `ranks.npz` from [[Error Bars Lab]]). The kinds of cold start, the fixes, the feedback loop, and the remark that the table 3 effect comes from runs of similar movies are general knowledge or my guess, not checked against a paper. Every code block in this note was extracted into a fresh folder (with copies of the files it needs from the earlier projects) and run again; each output matched the block above. Not tested: another machine or other library versions.
- No image (see level 2).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/recommenders Cold start problem:::A recommender's difficulty with a new user or a new item that has no history, so a method that learns from past actions has nothing to compare and falls back on rules such as "most popular".

#flashcards/ml/concept #flashcards/topic/recommenders What are the usual ways out of the cold start problem?
?
Show what is popular (a fallback); ask a new person for a few favourites; use features of people and items (genre, text, age) in a model that reads features, such as a two-tower model; use the first few actions fast; try new items on a few people to gather feedback (exploration).

#flashcards/ml/example #flashcards/topic/recommenders In MovieLens 100K, what share of the person-movie table is empty?::93.7%: there are 100,000 ratings out of 943 × 1,682 = 1,586,126 possible pairs (my arithmetic, from the script).

#flashcards/ml/concept #flashcards/topic/recommenders Did people with few ratings do worse than people with many in the shoot-out data?
?
No. The learned methods scored 0.084 to 0.109 for people with up to 41 ratings and 0.013 to 0.045 for people with over 113. MovieLens only keeps people with at least 20 ratings, so nobody is truly cold. One data set, 943 people; why heavy users are harder was not looked into.

#flashcards/ml/concept #flashcards/topic/recommenders How did the held-out movie's popularity change what the methods found?
?
Movies rated by under 20 people: popularity, item-kNN and user-kNN found 0 of 128 and ALS found 2. Movies rated by 100 or more people: the methods found 47 to 69 of 432. Learned methods rarely recommend an item few people have used.

#flashcards/ml/concept #flashcards/topic/recommenders What does a newcomer's number of ratings do to item-kNN's hit rate?
?
With only the first k ratings: 0.045 at k=1 (the popularity list scores 0.050), 0.067 at k=10, then about 0.06. A small gain over popularity, with differences of 10 to 20 people out of 943, so not firmly established.

#flashcards/ml/concept #flashcards/topic/recommenders Why is keeping the latest k ratings not a fair imitation of a new user?
?
The held-out movie is the person's last one, so the most recent rating is a much better clue: item-kNN scores 0.146 with only the latest rating and falls to 0.059 as older ratings are added. My guess is that people rate in runs of similar movies (not tested).

#flashcards/ml/concept #flashcards/topic/recommenders What does the Cold Start Problem note not show?
?
A true new-user test (whole people held out), error bars, or a model that uses features; it only groups a normal last-movie test by history size or item popularity, on one data set of 943 people.

#flashcards/ml/cloze #flashcards/topic/recommenders In the shoot-out data cold start showed up for ==items== that few people had rated, not for people with short ==histories==.
