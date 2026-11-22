---
tags: [collaborative-filtering, recommender-systems, similarity, k-NN, recommendation, ML]
status: growing
level: 4
tier: 2
---

# Collaborative Filtering

Collaborative filtering recommends by **borrowing from people who behave like you**. It uses only the table of who rated or clicked what. It never looks at what an item is.

## 🟢 1. Simple (age 12)
You and your friend Mia have liked almost the same films so far. Mia just loved a film you have not seen. You will probably like it too. **Collaborative filtering is "find people like me, copy what they liked."** The word "collaborative" means many people's choices work together to help one person.

**Example:** you gave 5 stars to a space film and 1 star to a cooking show. Mia did the same. Mia gave 5 stars to a robot film you have not seen, so the computer guesses 5 stars for you and puts that film high on your list.

## 🔵 2. How it works
1. Start with the table of ratings ([[Recommender Systems]]): people in rows, items in columns, many empty cells.
2. **Fix each person's habit.** Some people rate everything 4 or 5 and others are harsh. For each person, subtract their own average, so a rating becomes "more or less than usual for this person".
3. **Measure who is similar to whom.** Compare two people's "more or less than usual" lists with [[Cosine Similarity]]. Near +1 means they agree, near 0 means nothing in common, and near -1 means opposite tastes.
4. **Predict an empty cell.** To guess how you would rate a movie, look at the people who did rate it. Take their "more or less than usual" for that movie, weight each by how similar they are to you, add the result to your own average.
5. This is **user-based** collaborative filtering (neighbours are people). **Item-based** flips the table: two movies are similar if the same people rated them alike, and your rating of a movie is guessed from your ratings of similar movies. Item-based is the version behind "customers who bought this also bought".
6. The more two people's rows overlap, the more you can trust their similarity. The method breaks down for a new person or item with no ratings.

![[recsys-collaborative-filtering.gif]]
*A ratings table: people in rows, items in columns, thumbs up or down, empty cells unknown. Collaborative filtering predicts the empty cells (the highlighted cell here) from the rows and columns that look alike. Author Moshanin, [Collaborative filtering table.gif on Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Collaborative_filtering_table.gif), CC BY-SA 3.0.*

## 🟠 3. Technical detail
- **Memory-based:** no training step. The similarity table is built from the data and used directly. Cost grows with the number of people squared (all pairs), so big systems use sparse maths, sampling or approximate nearest-neighbour search ([[Vector Database]]).
- **Choices:** user-based or item-based; how many neighbours ($k$); how to centre (by person, by item or both); how to shrink a similarity that rests on few shared items.
- **Cosine on 0/1 data.** For clicks and purchases there are no ratings to centre, so each person is a 0/1 row and similarity is the plain cosine of the rows. My top-10 tests used this.
- **Code:** the toy below has 5 people and 6 movies (made up, chosen so the groups are clear). Similarities and predictions are computed with the formulas of level 4.

```python
import numpy as np

# 5 people rated 6 movies from 1 to 5 (0 = not rated yet)
R = np.array([[5, 4, 0, 1, 1, 0],      # Ann
              [4, 5, 4, 1, 0, 1],      # Ben
              [1, 0, 1, 5, 4, 5],      # Cy
              [1, 1, 0, 4, 5, 4],      # Di
              [5, 0, 5, 0, 1, 1]], float)   # Eli
names = ["Ann", "Ben", "Cy", "Di", "Eli"]
M = R > 0
mean = (R.sum(1) / M.sum(1))[:, None]             # each person's average rating
C = np.where(M, R - mean, 0)                      # centred: positive = liked more than usual
sim = (C @ C.T) / (np.linalg.norm(C, axis=1)[:, None] * np.linalg.norm(C, axis=1)[None] + 1e-9)   # cosine of centred rows
np.fill_diagonal(sim, 0)
print("similarity of Eli to the others:", dict(zip(names, sim[4].round(2))))


def predict(u, i):                                # mean + similarity-weighted deviations of people who rated item i
    who = np.where(M[:, i] & (np.arange(len(R)) != u))[0]
    w = sim[u, who]
    return mean[u, 0] + (w * C[who, i]).sum() / (np.abs(w).sum() + 1e-9)


print("Eli has not rated movie 1 (index 1). Eli's predicted rating:", round(predict(4, 1), 2))
print("Ann has not rated movie 3 (index 2); predicted:", round(predict(0, 2), 2))
```

Output (run on my machine):

```text
similarity of Eli to the others: {'Ann': 0.56, 'Ben': 0.53, 'Cy': -0.85, 'Di': -0.67, 'Eli': 0.0}
Eli has not rated movie 1 (index 1). Eli's predicted rating: 4.76
Ann has not rated movie 3 (index 2); predicted: 4.48
```

Eli likes movies 0 and 2 and dislikes 4 and 5. Ann and Ben also like the early movies and dislike the late ones, so they are Eli's neighbours (similarity +0.56 and +0.53). Cy and Di have the opposite taste (similarity -0.85 and -0.67). So Ann's and Ben's high ratings of movie 1 pull Eli's guess up (4.76), and Di's low rating **also** pushes it up, because the similarity is negative.

## 🔴 4. Deep / formal
**Similarity of two people.** Let $\bar r_u$ be person $u$'s average rating and $\tilde r_{ui}=r_{ui}-\bar r_u$ the rating relative to it, taken as 0 where $u$ did not rate item $i$:

$$
\text{sim}(u,v)=\frac{\sum_{i}\tilde r_{ui}\,\tilde r_{vi}}{\sqrt{\sum_{i}\tilde r_{ui}^{2}}\;\sqrt{\sum_{i}\tilde r_{vi}^{2}}}
$$

**Where:**
- $r_{ui}$ is the rating person $u$ gave item $i$. $\bar r_u$ is that person's average over the items they rated.
- $\tilde r_{ui}$ is "more or less than usual". Where $u$ did not rate $i$ it is 0.
- The top sum is over items; only items **both** people rated contribute, because any other product has a 0.
- The bottom is each person's own length (over all the items they rated).
- **Matters:** subtracting the average stops a generous rater from looking like everyone else's twin, and dividing by lengths makes the result a cosine between -1 and 1 ([[Cosine Similarity]]) that does not grow with how much a person rated. Without centring, almost everyone looks similar because all ratings are positive.

**Example (tiny numbers):** Eli and Ann from the toy (my arithmetic in Python). Eli rated movies 0, 2, 4, 5 with 5, 5, 1, 1 (average 3.0), so $\tilde r$ is $(2,0,2,0,-2,-2)$ over the six movies. Ann rated 0, 1, 3, 4 with 5, 4, 1, 1 (average 2.75), so $\tilde r$ is $(2.25,\,1.25,\,0,\,-1.75,\,-1.75,\,0)$.
- Top sum: only movies 0 and 4 are rated by both: $2\cdot2.25+(-2)(-1.75)=4.5+3.5=8$.
- Lengths: Eli $\sqrt{4+4+4+4}=4$; Ann $\sqrt{12.75}=3.571$.
- $\text{sim}=8/(4\times3.571)=\mathbf{0.56}$.
- Meaning: moderately similar, since they agree on the two movies they share.

**Predicting one cell.** Let $N_i(u)$ be the people (the neighbours) who rated item $i$:

$$
\hat r_{ui}=\bar r_u+\frac{\sum_{v\in N_i(u)}\text{sim}(u,v)\,\big(r_{vi}-\bar r_v\big)}{\sum_{v\in N_i(u)}\big|\text{sim}(u,v)\big|}
$$

**Where:**
- $\hat r_{ui}$ is the predicted rating of person $u$ for item $i$.
- $\bar r_u$ is the person's own average, the starting point.
- $N_i(u)$ is the set of other people who rated item $i$ (in practice the $k$ most similar of them).
- $r_{vi}-\bar r_v$ is how far above or below **their** usual person $v$ rated the item.
- The fraction is a weighted average of those deviations: weights are the similarities, divided by the sum of absolute similarities so the result stays on the rating scale.
- **Matters:** the absolute value in the bottom lets a negatively similar person count: their deviation is flipped by the negative weight. Without the bottom, a person with many neighbours would get larger and larger shifts. With no neighbours who rated the item the formula has nothing to say and you fall back on $\bar r_u$.

**Example (tiny numbers):** predict Eli's rating of movie 1 ($\bar r_{\text{Eli}}=3.0$). Three people rated it.
- Ann: similarity 0.560, rated 4 against her average 2.75, deviation $+1.25$.
- Ben: similarity 0.535, rated 5 against average 3.0, deviation $+2.0$.
- Di: similarity $-0.668$, rated 1 against average 3.0, deviation $-2.0$.
- Top: $0.560\cdot1.25+0.535\cdot2.0+(-0.668)(-2.0)=0.70+1.07+1.34=3.105$. Bottom: $0.560+0.535+0.668=1.763$. Fraction $=1.76$.
- $\hat r=3.0+1.76=\mathbf{4.76}$ (the program prints 4.76).
- Meaning: all three neighbours, including the opposite-taste Di, point to a rating well above Eli's average.

**Limits and open problems.**
- **Sparsity:** two people often share very few items, so the similarity is noisy. Shrinking it toward 0 when few items overlap helps.
- **Cold start:** a new person or item has no row or column. See [[Cold Start Problem]].
- **Popularity and the long tail:** neighbours' ratings are dominated by popular items.
- **Scale:** all-pairs similarity is quadratic in the number of people.
- **Why factorization took over for ratings:** it compresses the table into short vectors and needs no pair table ([[Matrix Factorization]]).

**What came before, and how this improved on it:**
- **Before:** non-personal lists (best sellers) and hand-made rules. Collaborative filtering was named in the early 1990s (Tapestry, 1992) and automated with ratings in GroupLens (1994). Item-based versions came around 2001 to 2003 (Sarwar et al.; Amazon's item-to-item filtering). All from memory, not checked.
- **Its limit:** one list for everyone ignores taste; early user-based versions did not scale and were hurt by the empty table.
- **What changed:** use the table itself as the model. People or items with similar rows are neighbours.
- **How much better:** measured here on MovieLens 100K (one run, see [[Recommender Shoot-out]]). Rating error (RMSE): global mean 1.154, user and item bias only 0.961, item-based kNN on top of the biases **0.933**. Top-10 hit rate over 943 people: most popular 0.050, item-kNN 0.059, user-kNN **0.074**. Small gaps, not tested for significance. I have no published figure to quote.

## Related
- [[Recommender Systems]] · [[Matrix Factorization]] · [[Cosine Similarity]] · [[k-NN]] · [[Embedding]] · [[Vector]] · [[Retrieval Metrics]] · [[Cold Start Problem]] · [[Vector Database]] · [[Recommender Shoot-out]]

## Sources
- My own toy code (above) and experiment [[Recommender Shoot-out]]. Maths formulas checked by running them and comparing a hand calculation with the program output. Naming and dates (Tapestry, GroupLens, Sarwar et al., Amazon) from memory, not checked. Not read: any collaborative filtering paper.
- Image: Moshanin, [Collaborative filtering table.gif](https://commons.wikimedia.org/wiki/File:Collaborative_filtering_table.gif), CC BY-SA 3.0.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/recommenders Collaborative filtering:::Recommending by borrowing from people (or items) with similar behaviour, using only the table of who rated what and never the items' descriptions.

#flashcards/ml/reverse #flashcards/topic/recommenders User-based vs item-based collaborative filtering:::User-based: neighbours are similar people, and you borrow their ratings. Item-based: neighbours are similar items (rated alike by the same people), and your rating is guessed from your ratings of those.

#flashcards/ml/concept #flashcards/topic/recommenders Why subtract each person's average rating before comparing people?
?
Some people rate everything high and others low. After subtracting, a rating means "more or less than usual for this person". Without it nearly everyone looks similar because all raw ratings are positive.

#flashcards/ml/cloze #flashcards/topic/recommenders In collaborative filtering a ==negative similarity== can still help: that person's deviation is flipped, so their low rating pushes the prediction up.

#flashcards/ml/image #flashcards/topic/recommenders ![[recsys-collaborative-filtering.gif]]
What does the highlighted cell stand for?
?
The unknown rating the system must predict for that person and item, from the rows and columns that look alike. The table has people in rows, items in columns and thumbs up or down. By Moshanin, CC BY-SA 3.0.

#flashcards/ml/formula #flashcards/topic/recommenders What is the similarity of two people in collaborative filtering?
?
$$
\text{sim}(u,v)=\frac{\sum_{i}\tilde r_{ui}\,\tilde r_{vi}}{\sqrt{\sum_{i}\tilde r_{ui}^{2}}\;\sqrt{\sum_{i}\tilde r_{vi}^{2}}}
$$
- $\tilde r_{ui}=r_{ui}-\bar r_u$ is the rating relative to person $u$'s average, 0 if not rated.
- The top sum is over items; only items both rated contribute. The bottom is each person's own length.
- Centring removes generous-rater bias; dividing by lengths gives a cosine in $[-1,1]$.

#flashcards/ml/concept #flashcards/topic/recommenders In the toy, Eli's centred ratings are (2, 0, 2, 0, -2, -2) and Ann's are (2.25, 1.25, 0, -1.75, -1.75, 0). What is sim(Eli, Ann)?
?
Top: 2·2.25 + (−2)(−1.75) = 4.5 + 3.5 = 8. Lengths: 4 and √12.75 = 3.571. 8 / (4 × 3.571) = 0.56. (My arithmetic.)

#flashcards/ml/formula #flashcards/topic/recommenders What is the user-based prediction formula?
?
$$
\hat r_{ui}=\bar r_u+\frac{\sum_{v\in N_i(u)}\text{sim}(u,v)\,\big(r_{vi}-\bar r_v\big)}{\sum_{v\in N_i(u)}\big|\text{sim}(u,v)\big|}
$$
- $\bar r_u$ is the person's own average, the starting point. $N_i(u)$ is the neighbours who rated item $i$.
- $r_{vi}-\bar r_v$ is a neighbour's deviation from their usual.
- The fraction is the similarity-weighted average of deviations; absolute values in the bottom keep it on the rating scale and let negative similarities count.

#flashcards/ml/concept #flashcards/topic/recommenders Predict Eli's rating of movie 1: Eli's average is 3.0; Ann sim 0.560 deviation +1.25; Ben sim 0.535 deviation +2.0; Di sim −0.668 deviation −2.0.
?
Top 0.560·1.25 + 0.535·2.0 + (−0.668)(−2.0) = 0.70 + 1.07 + 1.34 = 3.105. Bottom 0.560 + 0.535 + 0.668 = 1.763. Fraction 1.76, so 3.0 + 1.76 = 4.76. (My arithmetic, same as the program output.)

#flashcards/ml/concept #flashcards/topic/recommenders Why does all-pairs user similarity not scale, and what are the fixes?
?
The number of pairs grows with the square of the number of people. Fixes: sparse maths, sampling, shrinking similarities, approximate nearest-neighbour search, or compressing the table into short vectors (matrix factorization).

#flashcards/ml/concept #flashcards/topic/recommenders What did the measured results say about user-kNN and item-kNN in my MovieLens test?
?
Rating error (RMSE): item-kNN on top of biases 0.933, bias model 0.961, global mean 1.154. Top-10 hit rate: user-kNN 0.074, item-kNN 0.059, most popular 0.050. One run, gaps not tested for significance.

#flashcards/ml/concept #flashcards/topic/recommenders Which facts in the history block are from memory?
?
All of them: Tapestry 1992, GroupLens 1994, the item-based work around 2001 to 2003 (Sarwar et al.; Amazon). Not checked against a source.
