---
tags: [matrix-factorization, recommender-systems, collaborative-filtering, embedding, ALS, SGD, ML]
status: growing
level: 4
tier: 3
---

# Matrix Factorization

Matrix factorization **squeezes a big table with holes into two thin tables of short number lists**. Multiply the two thin tables back together and the holes are filled with guesses. In a [[Recommender Systems|recommender]] one thin table describes people and the other describes items.

## 🟢 1. Simple (age 12)
Suppose each person has a few hidden "taste dials" (a love-of-action dial, a love-of-comedy dial) and each film has matching dials (how much action, how much comedy). **How much you like a film is how well your dials match the film's dials.** Nobody wrote the dials down. The computer finds dial settings that explain the ratings it has already seen, then uses them to guess the ratings it has not seen.

**Example:** your dials are "action 9, comedy 2". A film has "action 8, comedy 1". Both are high on action, so you will probably like it. A film with "action 1, comedy 9" does not match, so you probably will not.

## 🔵 2. How it works
1. Start with the ratings table: people in rows, items in columns, mostly empty ([[Collaborative Filtering]]).
2. Choose a small number $k$, such as 20: the number of hidden dials ("factors").
3. Give every person a list of $k$ numbers and every item a list of $k$ numbers. At first they are random.
4. **Predicted rating = match between a person's list and an item's list** (multiply the numbers in pairs and add them up, a dot product, see [[Vector]]). Often add a small "this person rates high" number and a "this item is rated high" number (biases).
5. **Learn:** compare the predictions with the real ratings in the cells that are filled. Nudge the lists so the predictions get closer. Repeat many times. Two common ways to nudge: **SGD** (one rating at a time, see [[Stochastic Gradient Descent]]) and **ALS** (alternating least squares: freeze the item lists, solve for the best person lists in one go, then swap).
6. To stop the lists from memorizing the few ratings, add a penalty for big numbers ([[Regularization]]).
7. After learning, the product of the two thin tables is a full table. The guesses in the empty cells are the recommendations. The lists are [[Embedding]]s: close lists mean similar people or similar items.

![[recsys-matrix-factorization.png]]
*The idea: a user-by-item table is approximated by a user-by-feature table times a feature-by-item table. Author Xianteng, [Matrix factorization.png on Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Matrix_factorization.png), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Two flavours, trained for different goals.**
  - **Explicit (ratings):** fit the ratings that exist, ignore the empty cells. Judged by RMSE.
  - **Implicit (clicks, "who rated what"):** every cell is data: a filled cell says "interested" (confidence high), an empty cell says "probably not interested" (confidence low, but not zero). Judged by ranking scores. The usual method is ALS with confidence weights.
- **Relation to [[Singular Value Decomposition]]:** plain SVD factorizes a **complete** table. A ratings table has holes, so recommenders fit only the known cells (explicit) or treat the holes as weak zeros (implicit).
- **Settings:** $k$ (dials), the penalty $\lambda$, the learning rate (SGD) or number of sweeps (ALS), and for implicit the confidence boost $\alpha$.
- **Code 1: a toy with ALS** on 5 people and 6 movies (made up, as in [[Collaborative Filtering]]).

```python
import numpy as np

R = np.array([[5, 4, 0, 1, 1, 0], [4, 5, 4, 1, 0, 1], [1, 0, 1, 5, 4, 5], [1, 1, 0, 4, 5, 4], [5, 0, 5, 0, 1, 1]], float)
M = R > 0
k, reg = 2, 0.1
rs = np.random.RandomState(0)
P, Q = 0.5 * rs.randn(5, k), 0.5 * rs.randn(6, k)         # one k-number profile per person, one per movie
for it in range(30):                                       # alternating least squares: fix Q, solve each P row; then fix P, solve each Q row
    for u in range(5):
        Qm = Q[M[u]]; P[u] = np.linalg.solve(Qm.T @ Qm + reg * np.eye(k), Qm.T @ R[u, M[u]])
    for i in range(6):
        Pm = P[M[:, i]]; Q[i] = np.linalg.solve(Pm.T @ Pm + reg * np.eye(k), Pm.T @ R[M[:, i], i])
    if it in (0, 4, 29): print(f"iteration {it + 1:2d}: error on the known ratings (RMSE) = {np.sqrt((((P @ Q.T) - R)[M] ** 2).mean()):.3f}")
print("rebuilt matrix P @ Q.T (the empty cells now have guesses):")
print(np.round(P @ Q.T, 1))
print(f"guesses for the {int((~M).sum())} empty cells:", {f"person {u} movie {i}": round(float(P[u] @ Q[i]), 2) for u, i in zip(*np.where(~M))})
```

Output (run on my machine):

```text
iteration  1: error on the known ratings (RMSE) = 2.051
iteration  5: error on the known ratings (RMSE) = 0.339
iteration 30: error on the known ratings (RMSE) = 0.326
rebuilt matrix P @ Q.T (the empty cells now have guesses):
[[4.4 4.5 4.2 1.  1.  1. ]
 [4.4 4.4 4.2 1.  1.  1. ]
 [1.  1.  1.  4.6 4.6 4.6]
 [1.  1.  1.  4.3 4.2 4.3]
 [5.1 5.1 4.8 1.  1.  1. ]]
guesses for the 7 empty cells: {'person 0 movie 2': 4.2, 'person 0 movie 5': 1.0, 'person 1 movie 4': 1.01, 'person 2 movie 1': 1.02, 'person 3 movie 2': 0.99, 'person 4 movie 1': 5.15, 'person 4 movie 3': 1.01}
```

The error on the known ratings falls from 2.05 to 0.33 in five sweeps and is flat after that. The filled-in guess for person 4, movie 1 is 5.15, above the top of the 1 to 5 scale; real systems clip predictions to 5. On a toy this small the factors can only memorize, so it shows the mechanics, not quality.

- **Code 2: the real experiment.** On MovieLens 100K I trained a biased MF with SGD (rating task) and an implicit ALS (top-10 task) in pure numpy. All code and outputs are in [[Recommender Shoot-out]]. Results (one run each unless noted):

| Task | Method | Result |
|---|---|---|
| Rating error (RMSE, lower is better) | global mean | 1.154 |
| | user and item bias model | 0.961 |
| | item-kNN on bias residuals | 0.933 |
| | biased MF, SGD, $k=20$ (mean of 3 seeds) | 0.939 |
| Top-10 hit rate (higher is better) | most popular | 0.050 |
| | implicit MF, ALS, $k=32$ | **0.087** |
| | the SGD rating MF, used for ranking | 0.022 |

- **Reading it:** the rating MF beat the bias model (0.939 against 0.961) but did **not** beat the simpler item-kNN (0.933), and I did not tune $k$, the penalty or the learning rate, so do not read this as "MF is worse than kNN". For ranking, a model trained on ratings did far worse than the same family trained on implicit data.

## 🔴 4. Deep / formal
**Prediction.** For person $u$ and item $i$, with $\mu$ the overall average rating, $b_u$ and $b_i$ the person and item biases, and $\mathbf p_u,\mathbf q_i\in\mathbb R^k$ their factor lists:

$$
\hat r_{ui}=\mu+b_u+b_i+\mathbf p_u\cdot\mathbf q_i
$$

**Where:**
- $\mu$ is the average of all known ratings: the starting guess for any cell.
- $b_u$ is how much higher (or lower) than average this person tends to rate. $b_i$ is the same for the item.
- $\mathbf p_u$ and $\mathbf q_i$ are the person's and the item's lists of $k$ numbers. $\mathbf p_u\cdot\mathbf q_i=\sum_{f=1}^{k}p_{uf}q_{if}$ is their match.
- **Matters:** the biases soak up "this is a great film" and "this person is generous", so the dials only have to explain the taste-specific part. Without the biases the factors must also carry these effects, which is harder to learn and typically worse. Without $\mathbf p_u\cdot\mathbf q_i$ the model gives everyone the same ranking of items.

**Example (tiny numbers):** $\mu=3.5$, $b_u=+0.5$, $b_i=-0.2$, $\mathbf p_u=(1.2,\,0.5)$, $\mathbf q_i=(0.8,\,-0.4)$ (made up, my arithmetic in Python).
- Match: $1.2\cdot0.8+0.5\cdot(-0.4)=0.96-0.20=0.76$.
- $\hat r=3.5+0.5-0.2+0.76=\mathbf{4.56}$.
- Meaning: a generous person ($+0.5$) and a slightly under-rated item ($-0.2$) with a good taste match ($+0.76$) give a predicted 4.56 stars.

**Learning: fit the known ratings, penalize big numbers.** With $K$ the set of known (person, item) pairs and $\lambda$ the penalty strength:

$$
L=\sum_{(u,i)\in K}\Big(r_{ui}-\mu-b_u-b_i-\mathbf p_u\cdot\mathbf q_i\Big)^{2}+\lambda\Big(\|\mathbf p_u\|^{2}+\|\mathbf q_i\|^{2}+b_u^{2}+b_i^{2}\Big)
$$

**Where:**
- $r_{ui}$ is the real rating in a known cell; the bracket is the prediction error for it.
- $K$ restricts the sum to cells that actually have a rating (the holes do not count).
- $\|\mathbf p_u\|^2$ is the sum of squares of the numbers in the list ($\|\cdot\|$ is its length).
- $\lambda$ is the penalty weight. (In the full loss the penalty term sits inside the sum, once per known rating.)
- **Matters:** without the penalty the model can drive the error on known ratings to nearly 0 by memorizing them and then guess badly on the holes ([[Overfitting]]). Too much penalty shrinks every list to 0 and every prediction to the average.

**Example (tiny numbers):** one known rating $r=4$ with the same $\mu,b_u,b_i,\mathbf p_u,\mathbf q_i$ as above, $\lambda=0.1$ (my arithmetic in Python).
- Error: $4-4.56=-0.56$, squared $0.3136$.
- Penalty: $\|\mathbf p_u\|^2=1.44+0.25=1.69$; $\|\mathbf q_i\|^2=0.64+0.16=0.80$; $b_u^2=0.25$; $b_i^2=0.04$. Sum $2.78$, times 0.1 gives $0.278$.
- $L=0.3136+0.278=\mathbf{0.5916}$.
- Meaning: more than a third of this term's loss is the penalty. Learning will shrink the lists a little rather than chase the exact 4.

**ALS: solve one person at a time.** With the item lists fixed, the best list for person $u$ is a ridge regression on the items $u$ rated. Let $Q_u$ be the matrix whose rows are $\mathbf q_i$ for the items $u$ rated and $\mathbf r_u$ the vector of those ratings (biases left out, for clarity):

$$
\mathbf p_u=\big(Q_u^{\top}Q_u+\lambda I\big)^{-1}Q_u^{\top}\mathbf r_u
$$

**Where:**
- $Q_u^\top Q_u$ is a $k\times k$ table of how the item lists overlap; $I$ is the identity table (1 on the diagonal), so $\lambda I$ adds $\lambda$ to the diagonal.
- $Q_u^\top\mathbf r_u$ is how each factor lines up with the ratings.
- $(\cdot)^{-1}$ is the matrix inverse (in code, a linear solve).
- **Matters:** it gives the exact best list for $u$ in one step, so ALS needs only a few sweeps and each person (and each item) can be solved in parallel. $\lambda I$ keeps the table invertible and shrinks the answer. Without it a person who rated fewer than $k$ items gives a table that cannot be inverted.

**Example (tiny numbers):** $k=1$ so everything is a number. A person rated two items whose factors are $q=(1,2)$ with ratings $r=(3,5)$, and $\lambda=1$ (my arithmetic).
- $\sum q^2=1+4=5$. $\sum qr=1\cdot3+2\cdot5=13$.
- Without the penalty: $p=13/5=2.6$. With it: $p=13/(5+1)=\mathbf{2.17}$.
- Meaning: the penalty pulls the person's number from 2.6 toward 0. It matters most when the person has rated very few items.

**Implicit feedback: confidence weights.** For "who rated what" data set the preference $z_{ui}=1$ if person $u$ interacted with item $i$ and 0 otherwise, with confidence $c_{ui}=1+\alpha\,z_{ui}$, and minimize

$$
\sum_{u,i}c_{ui}\big(z_{ui}-\mathbf x_u\cdot\mathbf y_i\big)^{2}+\lambda\Big(\sum_u\|\mathbf x_u\|^{2}+\sum_i\|\mathbf y_i\|^{2}\Big)
$$

**Where:**
- $z_{ui}$ is 1 for "interacted", 0 for "no record". $c_{ui}$ says how sure we are of that value.
- $\alpha$ is the boost for observed interactions (40 in my run, a common value I chose without tuning).
- $\mathbf x_u,\mathbf y_i$ are the person and item lists (named differently from $\mathbf p,\mathbf q$ because there are no biases here).
- The sum runs over **all** pairs, not just known ones.
- **Matters:** the empty cells now count, weakly. Without them the model would predict "interested" for everything. Without the boost $\alpha$ the few observed interactions would be drowned out by millions of empty cells.

**Example (tiny numbers):** $\alpha=40$ (my arithmetic).
- A pair with an interaction: $c=1+40\cdot1=41$, preference 1, prediction 0.8: loss $41\cdot(1-0.8)^2=\mathbf{1.64}$.
- A pair with no record: $c=1$, preference 0, prediction 0.3: loss $1\cdot(0-0.3)^2=\mathbf{0.09}$.
- Meaning: missing the observed pair by 0.2 costs 1.64, while missing an empty pair by 0.3 costs only 0.09. The model is pushed hard to get the observed pairs right.

**Things to remember.**
- **Cold start:** a new person or item has no factor list yet ([[Cold Start Problem]]).
- **Objective mismatch (seen in my run):** training on rating error does not train the model to rank. The rating-trained MF ranked worse than popularity. I did not test why. Untested guesses: the item biases may push rarely rated items with a few high ratings to the top; the rating objective ignores which cells are empty (and that carries the signal); I did not tune it for ranking.
- **Neural versions** replace the dot product with a small network, or learn person and item towers separately ([[Two-Tower Model]]); not run here.

**What came before, and how this improved on it:**
- **Before:** neighbour methods ([[Collaborative Filtering]]), which keep the whole table and compare pairs. Factorization of ratings became widely known through the Netflix Prize (2006 to 2009); the biased SGD form and the implicit ALS form are from papers around 2008 to 2009 (Koren, Bell and Volinsky; Hu, Koren and Volinsky). From memory, not checked.
- **Its limit:** neighbour methods cost a lot at scale and degrade on a mostly empty table.
- **What changed:** learn short dense lists. Prediction is a short dot product, and the lists are reusable embeddings.
- **How much better:** measured here, MF's error (0.939) beat the bias-only model (0.961), about 2.3% lower, but not item-kNN (0.933). On the top-10 task, the implicit version reached a hit rate of 0.087 against 0.050 for most popular, about 1.7 times. One run each, no tuning. Published Netflix Prize figures are from memory and not quoted.

## Related
- [[Recommender Systems]] · [[Collaborative Filtering]] · [[Singular Value Decomposition]] · [[Embedding]] · [[Vector]] · [[Matrix]] · [[Stochastic Gradient Descent]] · [[Regularization]] · [[Overfitting]] · [[Gradient Descent]] · [[Two-Tower Model]] · [[Cold Start Problem]] · [[Recommender Shoot-out]]

## Sources
- My own toy code and experiment [[Recommender Shoot-out]]. Equations checked by comparing a hand calculation to Python. History and the two paper names (Koren, Bell, Volinsky; Hu, Koren, Volinsky) from memory, not read.
- Image: Xianteng, [Matrix factorization.png](https://commons.wikimedia.org/wiki/File:Matrix_factorization.png), CC BY-SA 4.0.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/recommenders Matrix factorization:::Approximating a big ratings table (people by items, mostly empty) as the product of two thin tables of short number lists, one per person and one per item. The product fills the empty cells with guesses.

#flashcards/ml/reverse #flashcards/topic/recommenders Latent factor / "taste dial":::One hidden number in a person's or an item's list. Nobody names it; it is learned so the match between a person's list and an item's list explains the ratings.

#flashcards/ml/concept #flashcards/topic/recommenders What is the difference between SGD and ALS for matrix factorization?
?
SGD nudges the lists one rating at a time. ALS freezes the item lists and solves for the best person lists exactly in one step (a ridge regression), then swaps roles. ALS needs few sweeps and can be run in parallel.

#flashcards/ml/concept #flashcards/topic/recommenders Why does plain SVD not directly solve recommendation?
?
SVD factorizes a complete table. A ratings table has holes, so recommenders fit only the known cells (explicit) or treat the holes as weak zeros (implicit).

#flashcards/ml/image #flashcards/topic/recommenders ![[recsys-matrix-factorization.png]]
What does this figure show?
?
A user-by-item table approximated by a user-by-feature table multiplied by a feature-by-item table, with few features. By Xianteng, CC BY-SA 4.0.

#flashcards/ml/formula #flashcards/topic/recommenders What is the prediction formula of biased matrix factorization?
?
$$
\hat r_{ui}=\mu+b_u+b_i+\mathbf p_u\cdot\mathbf q_i
$$
- $\mu$ is the overall average rating. $b_u$ and $b_i$ are the person's and item's bias (generous rater, well-liked item).
- $\mathbf p_u$ and $\mathbf q_i$ are the $k$-number lists; their dot product is the taste match.
- The biases absorb "great film" and "generous person" so the dials only carry taste.

#flashcards/ml/concept #flashcards/topic/recommenders μ = 3.5, b_u = +0.5, b_i = −0.2, p = (1.2, 0.5), q = (0.8, −0.4). What is the predicted rating?
?
Dot product: 1.2·0.8 + 0.5·(−0.4) = 0.96 − 0.20 = 0.76. Rating: 3.5 + 0.5 − 0.2 + 0.76 = 4.56. (My arithmetic.)

#flashcards/ml/formula #flashcards/topic/recommenders What is the training loss of biased matrix factorization?
?
$$
L=\sum_{(u,i)\in K}\Big(r_{ui}-\mu-b_u-b_i-\mathbf p_u\cdot\mathbf q_i\Big)^{2}+\lambda\Big(\|\mathbf p_u\|^{2}+\|\mathbf q_i\|^{2}+b_u^{2}+b_i^{2}\Big)
$$
- $K$ is the set of known (person, item) ratings; empty cells do not count.
- The first part is the squared prediction error. The second part ($\lambda$ times the squared sizes of the lists and biases) is the penalty.
- Without the penalty the model memorizes known ratings and guesses badly on the holes. Too much shrinks everything to the average.

#flashcards/ml/concept #flashcards/topic/recommenders One rating r = 4 with the prediction 4.56, squared sizes |p|² = 1.69, |q|² = 0.80, b_u² = 0.25, b_i² = 0.04 and λ = 0.1. What is the loss?
?
Error² = (4 − 4.56)² = 0.3136. Penalty = 0.1 × (1.69 + 0.80 + 0.25 + 0.04) = 0.278. Loss = 0.5916. (My arithmetic.)

#flashcards/ml/formula #flashcards/topic/recommenders What is the ALS update for one person's factor list?
?
$$
\mathbf p_u=\big(Q_u^{\top}Q_u+\lambda I\big)^{-1}Q_u^{\top}\mathbf r_u
$$
- $Q_u$ has one row per item that person $u$ rated (that item's factor list). $\mathbf r_u$ is the vector of those ratings.
- $\lambda I$ adds $\lambda$ to the diagonal: it keeps the table invertible and shrinks the answer.
- It is the exact best list for $u$ with the item lists fixed, so each person can be solved in parallel.

#flashcards/ml/concept #flashcards/topic/recommenders k = 1, item factors q = (1, 2), ratings r = (3, 5), λ = 1. What is the ALS value for the person, with and without the penalty?
?
Σq² = 5 and Σqr = 13. Without the penalty p = 13/5 = 2.6. With it p = 13/(5+1) = 2.17. The penalty pulls the value toward 0. (My arithmetic.)

#flashcards/ml/formula #flashcards/topic/recommenders What is the implicit-feedback factorization objective?
?
$$
\sum_{u,i}c_{ui}\big(z_{ui}-\mathbf x_u\cdot\mathbf y_i\big)^{2}+\lambda\Big(\sum_u\|\mathbf x_u\|^{2}+\sum_i\|\mathbf y_i\|^{2}\Big)
$$
- $z_{ui}$ is 1 if the person interacted with the item and 0 if there is no record. $c_{ui}=1+\alpha z_{ui}$ is the confidence.
- $\mathbf x_u,\mathbf y_i$ are the person and item lists. The sum runs over all pairs, not only observed ones.
- Empty cells count weakly (so the model does not predict "interested" for everything); the boost $\alpha$ stops millions of empty cells from drowning the few observed ones.

#flashcards/ml/concept #flashcards/topic/recommenders With α = 40, what is the loss of an observed pair predicted at 0.8 and an unobserved pair predicted at 0.3?
?
Observed: c = 41, preference 1, 41 × (1 − 0.8)² = 1.64. Unobserved: c = 1, preference 0, 1 × (0 − 0.3)² = 0.09. Missing the observed pair costs far more. (My arithmetic.)

#flashcards/ml/concept #flashcards/topic/recommenders What was the surprise when I used the rating-trained MF to make top-10 lists?
?
On MovieLens 100K its top-10 hit rate was 0.022, worse than "most popular" (0.050), while the implicit ALS version reached 0.087. The cause was not tested. Untested guesses: item biases favouring rarely rated items, ignoring the empty cells, no tuning for ranking.

#flashcards/ml/concept #flashcards/topic/recommenders How did the rating MF compare with simpler methods on RMSE in my test?
?
Biased MF (SGD, k=20, mean of 3 seeds) 0.939, bias model 0.961, item-kNN on bias residuals 0.933, global mean 1.154. MF beat the bias model but not item-kNN, and none of it was tuned.

#flashcards/ml/cloze #flashcards/topic/recommenders In ALS the ==ridge penalty== keeps the table invertible and shrinks the answer.
