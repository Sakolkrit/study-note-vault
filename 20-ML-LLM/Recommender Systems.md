---
tags: [recommender-systems, recommendation, collaborative-filtering, ranking, personalization, ML]
status: growing
level: 4
tier: 3
---

# Recommender Systems

A recommender system **guesses what a person will like** from what they and many other people did before, and shows a short list. "People who watched this also watched that" is the best-known example.

## 🟢 1. Simple (age 12)
Imagine a friend who has watched a thousand films and remembers what everyone in your class liked. You ask, "What should I watch tonight?" Your friend thinks, "You loved the space film and the robot film. Sam loved those too, and Sam also loved this one. Try it." **A recommender system is that friend, made by a computer.** It does not understand films. It only notices who likes what together.

**Example:** you rate three songs. The computer finds people who rated those three songs like you, looks at songs they liked that you have not heard, and puts the most popular of those at the top of your list.

## 🔵 2. How it works
1. **The data** is a big table: one row per person, one column per item (movie, song, product). Most cells are empty because nobody has tried everything. A cell holds a rating ("4 stars") or just a click or a purchase. A rating is **explicit** feedback. A click is **implicit** feedback: it says "they looked", not "they liked".
2. **The job:** fill in the empty cells (guess how much you would like each unseen item), then show the items with the highest guesses.
3. **Three families of methods.**
   - **Popularity:** show everyone what most people like. No personal touch.
   - **Content-based:** describe each item (genre, words, price) and recommend items similar to what you liked.
   - **[[Collaborative Filtering]]:** ignore descriptions. Use only the table: people with similar tastes are neighbours, so borrow their ratings.
4. A more modern version of collaborative filtering gives each person and each item a short list of numbers, an [[Embedding]], so that "how much you like it" is how well the two lists match. That is [[Matrix Factorization]].
5. **Measuring:** hide some of what a person did, recommend, and see whether the hidden item shows up in the top 10. The scores come from [[Retrieval Metrics]] (hit rate, NDCG). A rating-error score (RMSE) is a different test and, as the project shows, **it does not tell you which method makes the best top-10 list**.
6. **Problems to know:** a brand-new person or item has no history (the **cold start problem**, see [[Cold Start Problem]]). Popular items get shown more, so they get more clicks, so they get shown more (a feedback loop).

![[recsys-collaborative-filtering.gif]]
*The data of a recommender: rows are people, columns are items (a picture, a book, a video, a game), a thumb up or down is a rating, and an empty cell is unknown. The highlighted cell is the one to predict. Author Moshanin, [Collaborative filtering table.gif on Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Collaborative_filtering_table.gif), CC BY-SA 3.0.*

## 🟠 3. Technical detail
- **Two tasks that look alike and are not.**
  - **Rating prediction:** guess the number a person would give; judged by RMSE.
  - **Top-N recommendation:** put the right few items first; judged by hit rate and NDCG. What users see is a list, so this is usually the one that matters.
- **Candidate generation then ranking.** Real systems first pick a few hundred candidates cheaply (nearest neighbours in an embedding space, see [[Vector Database]] and [[Two-Tower Model]]), then rank them with a heavier model. It is the same two-step idea as [[Reranking]].
- **My test (project [[Recommender Shoot-out]]).** MovieLens 100K: 943 people, 1,682 movies, 100,000 ratings. For the top-10 test I hid each person's **latest** rated movie and ranked it against about 1,600 movies they had not rated. Results (hit rate at 10 = the share of people whose hidden movie was in their top 10, 943 people, one run each):

| Method | Hit rate @10 | NDCG @10 | Median rank of the hidden movie |
|---|---|---|---|
| random | 0.005 | 0.003 | 782 |
| most popular | 0.050 | 0.025 | 297 |
| item-kNN (cosine) | 0.059 | 0.028 | 206 |
| user-kNN (cosine, 50 neighbours) | 0.074 | 0.033 | 134 |
| implicit-feedback MF (ALS, k=32) | **0.087** | **0.040** | **114** |
| rating-trained MF (SGD), used for ranking | 0.022 | 0.010 | 677 |

- **The surprise:** the matrix factorization trained to predict **ratings** (it gave the best rating error among the learned models, RMSE 0.939 against 0.961 for a simple bias model) ranked **worse than just showing the most popular movies**. The same idea trained on "who rated what" (implicit) ranked best. I did not test why (see the project note for the guesses).
- **Smallest code:** the popularity baseline and a hit-rate function, which every other method has to beat.

```python
import numpy as np

# 5 people x 6 movies, ratings 1 to 5, 0 = not rated. Hold out each person's last rated movie, recommend from popularity.
R = np.array([[5, 4, 0, 1, 1, 0], [4, 5, 4, 1, 0, 1], [1, 0, 1, 5, 4, 5], [1, 1, 0, 4, 5, 4], [5, 0, 5, 0, 1, 1]], float)
held = [1, 5, 5, 5, 5]                                   # the movie (column) we pretend each person rated last (chosen by hand for this toy)
train = R.copy()
for u, i in enumerate(held): train[u, i] = 0
popularity = (train > 0).sum(0)                          # how many people rated each movie
hits = 0
for u, i in enumerate(held):
    score = popularity.astype(float)
    score[train[u] > 0] = -1                             # never recommend what the person already rated
    top2 = np.argsort(-score, kind="stable")[:2]
    hits += i in top2
    print(f"person {u}: top-2 {top2.tolist()} hidden movie {i} -> {'hit' if i in top2 else 'miss'}")
print("hit rate @2 =", hits / len(held))
```

Output (run on my machine):

```text
person 0: top-2 [2, 1] hidden movie 1 -> hit
person 1: top-2 [4, 5] hidden movie 5 -> hit
person 2: top-2 [1, 5] hidden movie 5 -> hit
person 3: top-2 [2, 5] hidden movie 5 -> hit
person 4: top-2 [3, 1] hidden movie 5 -> miss
hit rate @2 = 0.8
```

This toy is **my own, tiny and hand-made** (the hidden movies are chosen by hand), so the hit rate says nothing about real quality. It only shows the mechanics.

## 🔴 4. Deep / formal
**Hit rate at K.** With $U$ the set of test people and $\text{rank}_u$ the place of person $u$'s hidden item in the list (1 is the top):

$$
\text{HR@}K=\frac{1}{|U|}\sum_{u\in U}\mathbf{1}\big[\text{rank}_u\le K\big]
$$

**Where:**
- $|U|$ is the number of people tested. $\sum_{u\in U}$ adds over all of them.
- $\text{rank}_u$ is where the hidden item landed when all candidate items were sorted by score, best first.
- $\mathbf{1}[\cdot]$ is 1 when the statement inside is true and 0 when false.
- $K$ is how long the shown list is (10 here).
- **Matters:** it is the share of people for whom the answer was on screen. Without the cut at $K$ you would be measuring the average rank, which is dominated by a few terrible misses. It ignores order **inside** the top $K$ (use NDCG in [[Retrieval Metrics]] for that).

**Example (tiny numbers):** five people whose hidden movie landed at ranks 3, 12, 1, 40 and 8 (made up, my arithmetic in Python).
- With $K=10$: ranks 3, 1 and 8 are at most 10, so the indicators are 1, 0, 1, 0, 1. The sum is 3.
- $\text{HR@}10=3/5=\mathbf{0.6}$.
- Meaning: for 60% of the people the answer was on screen. With $K=1$ only the person with rank 1 counts, so $\text{HR@}1=1/5=0.2$.

**Things that make the number mean less than it looks.**
- **Which items count as candidates.** I ranked against all unrated movies (about 1,600). Many papers rank the hidden item against only 100 random items, which gives much higher numbers. Never compare a hit rate across papers without checking this.
- **Leave-last-out is not the future.** MovieLens timestamps are when the rating was entered, not when the film was watched, so "latest" is a rough stand-in for "next".
- **Popularity bias.** Logged data comes from what an earlier system showed, so the test rewards methods that copy that system.
- **Offline is not online.** A better offline score does not guarantee better real results; the real test is an [[AB Testing|A/B Testing]] on live people (not run here).

**What came before, and how this improved on it:**
- **Before:** editors and best-seller lists: one list for everyone. Collaborative filtering as an idea dates to the early 1990s (Tapestry, 1992; GroupLens, 1994; from memory, not checked here).
- **Its limit:** one list ignores taste; early neighbour methods were slow on millions of people and struggled with a table that is almost all empty. (My MovieLens table is 93.7% empty: 100,000 of $943\times1682$ cells filled, my arithmetic.)
- **What changed:** learning a short vector for each person and item (matrix factorization became famous through the Netflix Prize, 2006 to 2009, from memory), and treating clicks as implicit signals (about 2008, from memory).
- **How much better:** measured here, on top-10 lists, **implicit MF found the hidden movie in the top 10 for 8.7% of people against 5.0% for "most popular"** (about 1.7 times), and the rating-trained MF reached only 2.2%. One run, one data set, one split, so the gaps between the middle methods (0.059, 0.074, 0.087) are small and I did not test whether they are real. I have no published figure to quote.

## Related
- [[Collaborative Filtering]] · [[Matrix Factorization]] · [[Embedding]] · [[Cosine Similarity]] · [[k-NN]] · [[Singular Value Decomposition]] · [[Retrieval Metrics]] · [[Reranking]] · [[Vector Database]] · [[Two-Tower Model]] · [[Cold Start Problem]] · [[AB Testing|A/B Testing]] · [[Overfitting]] · [[Recommender Shoot-out]]
- [[AB Test Lab|A/B Test Lab]]
## Sources
- My own experiment, [[Recommender Shoot-out]] (code, tables, run details there). Data: MovieLens 100K from GroupLens (`ml-100k.zip`, downloaded from files.grouplens.org); its README was not read in full by me for the licence terms, so check them before reusing the data.
- History (Tapestry, GroupLens, Netflix Prize, implicit feedback) is **from memory, not checked**. Not read: any recommender paper or survey.
- Image: Moshanin, [Collaborative filtering table.gif](https://commons.wikimedia.org/wiki/File:Collaborative_filtering_table.gif), CC BY-SA 3.0.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/recommenders Recommender system:::A system that guesses what a person will like from what they and many other people did before, and shows a short ranked list.

#flashcards/ml/reverse #flashcards/topic/recommenders Explicit vs implicit feedback:::Explicit is a stated opinion (a star rating). Implicit is behaviour (a click, a purchase): it says the person looked or bought, not that they liked it.

#flashcards/ml/concept #flashcards/topic/recommenders What are the three families of recommender methods?
?
Popularity (the same list for everyone), content-based (recommend items whose description is like what you liked), and collaborative filtering (use only the table of who rated what and borrow from people with similar tastes).

#flashcards/ml/concept #flashcards/topic/recommenders Why is "rating prediction" a different task from "top-N recommendation"?
?
Rating prediction guesses the number a person would give and is judged by RMSE. Top-N puts the right few items first and is judged by hit rate or NDCG. In my test the MF that was best at the rating task ranked worse than plain popularity on the top-10 task.

#flashcards/ml/image #flashcards/topic/recommenders ![[recsys-collaborative-filtering.gif]]
What does this figure show, and what is the highlighted cell?
?
The data of a recommender: rows are people, columns are items (a picture, a book, a video, a game), thumbs up or down are ratings and empty cells are unknown. The highlighted cell is the one the system must predict. By Moshanin, CC BY-SA 3.0.

#flashcards/ml/formula #flashcards/topic/recommenders What is the formula for hit rate at K?
?
$$
\text{HR@}K=\frac{1}{|U|}\sum_{u\in U}\mathbf{1}\big[\text{rank}_u\le K\big]
$$
- $|U|$ is the number of test people. $\text{rank}_u$ is the place of person $u$'s hidden item in the sorted list (1 is the top).
- $\mathbf{1}[\cdot]$ is 1 if the statement is true, 0 if false. $K$ is the list length shown.
- It is the share of people whose answer was on screen. It ignores the order inside the top $K$.

#flashcards/ml/concept #flashcards/topic/recommenders Five people's hidden items land at ranks 3, 12, 1, 40 and 8. What are HR@10 and HR@1?
?
HR@10 counts ranks 3, 1 and 8, so 3/5 = 0.6. HR@1 counts only rank 1, so 1/5 = 0.2. (My arithmetic.)

#flashcards/ml/cloze #flashcards/topic/recommenders A new person or item with no history is the ==cold start== problem.

#flashcards/ml/concept #flashcards/topic/recommenders In my MovieLens top-10 test, what were the hit rates of most popular, user-kNN and implicit MF?
?
0.050 (most popular), 0.074 (user-kNN) and 0.087 (implicit MF), over 943 people, one run each. The rating-trained MF got 0.022 and random 0.005. I did not test whether the gaps between the middle methods are real.

#flashcards/ml/concept #flashcards/topic/recommenders Why can hit rates not be compared across papers without checking the candidate set?
?
Ranking the hidden item against all unrated items (about 1,600 here) is much harder than ranking it against 100 random items, which gives much higher numbers.

#flashcards/ml/concept #flashcards/topic/recommenders Name two reasons an offline top-N score may mislead.
?
Logged data reflects what an earlier system showed (popularity bias), and "latest rating" is only a rough stand-in for "next thing watched" (MovieLens timestamps are rating times). Only an online A/B test shows real effect (not run here).

#flashcards/ml/concept #flashcards/topic/recommenders How empty was my MovieLens 100K table?
?
100,000 ratings in 943 × 1,682 cells, so 93.7% of the cells were empty (my arithmetic).

#flashcards/ml/concept #flashcards/topic/recommenders What do real systems do in two steps, and which other note uses the same idea?
?
Candidate generation (cheaply pick a few hundred items, for example nearest neighbours in an embedding space) then ranking (a heavier model orders them). Reranking in RAG is the same two-stage idea.

#flashcards/ml/concept #flashcards/topic/recommenders Where do the history dates in this note come from?
?
From memory (Tapestry 1992, GroupLens 1994, Netflix Prize 2006 to 2009, implicit feedback about 2008). Not checked against a source.
