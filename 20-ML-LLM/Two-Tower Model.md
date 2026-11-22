---
tags: [recommender, embedding, retrieval, neural-network, two-tower, machine-learning]
status: growing
level: 4
tier: 3
---

# Two-Tower Model

A **two-tower model** is a recommender (or search) model made of **two separate networks**. One, the *user tower*, turns what is known about a person into a list of numbers (a vector). The other, the *item tower*, does the same for an item. The match score is how closely the two vectors point the same way.

## 🟢 1. Simple (age 12)
Two helpers each write a "taste card". One helper writes a card about **you** from what you have watched. The other writes a card about **each movie**. If your card and a movie's card look alike, the movie is a good suggestion for you.

**Example:** think of a school club fair. You fill in a short form about what you like, and every club has a card about what it does. The cards with the most in common are the clubs you are shown first.

## 🔵 2. How it works
1. The **user tower** takes what is known about a person (the movies they rated, or their age, country and device) and outputs a vector, a short list of numbers, called an [[Embedding]].
2. The **item tower** takes what is known about an item (its ID, genre, text, picture) and outputs a vector of the same length.
3. The **score** of a person and an item is how well the two vectors line up: the dot product, which for vectors of length 1 is the [[Cosine Similarity]].
4. **Training.** Take a batch of pairs "this person chose this item". For each person, their own item should score higher than the items of the other people in the batch (the other items serve as wrong answers, called *in-batch negatives*). A softmax loss rewards that.
5. **Serving.** The two towers never meet until the last step. So all item vectors can be computed in advance and stored in a nearest-neighbour index ([[Vector Database]]). For a visitor, run the user tower once and look up the closest item vectors. This stays fast even with millions of items.
6. **Why it matters for new users.** The user tower reads features or history, not a fixed row of a table, so it can make a vector for a person the system has not seen before ([[Cold Start Problem]]).

*I found no free, on-topic picture of a two-tower model on Wikimedia Commons (the searches returned scanned documents and unrelated papers), so this note has no picture.*

## 🟠 3. Technical detail
- **My small version** (MovieLens 100K, the same data and test as [[Recommender Shoot-out]]): the user tower averages the learned vectors of the movies a person rated, then passes them through two small layers to a vector of 32 numbers, scaled to length 1. The item tower is a learned vector per movie, passed through two small layers, also scaled to length 1. The score is the dot product divided by a temperature $\tau$.
- **Training details:** pairs are the 99,057 (person, movie) ratings of the training part. For each pair the target movie is hidden from the person's history. Other movies in the batch are the wrong answers, but the same movie appearing twice in a batch is not counted as wrong. Batch of 512, AdamW, learning rate 0.003, and a "history dropout" that forgets a share of the rest of the history (so the tower cannot lean on a few rated movies).
- **Choosing the settings without touching the test movies:** I kept each person's last *training* movie as a validation movie, tried 8 settings, and picked by validation score; the test movies were used once at the end.
- **Matching the test of the shoot-out:** the held-out movie must be in the top 10 of the movies the person has not rated.

First the loss on a tiny batch (NumPy only):

```python
# file: two_tower_toy.py
import numpy as np

# one person vector p and one movie vector q, both of length 1: the score is their dot product divided by the temperature tau
p = np.array([0.6, 0.8]); q = np.array([1.0, 0.0])
print("p . q =", round(float(p @ q), 3), "; divided by tau = 1:", round(float(p @ q) / 1, 3), "; by tau = 0.1:", round(float(p @ q) / 0.1, 3))

# a batch of 2 people (rows of P) and the 2 movies they actually watched (rows of Q); every other movie in the batch is a wrong answer
P = np.array([[1.0, 0.0], [0.0, 1.0]]); Q = np.array([[2.0, 0.0], [0.0, 1.0]])
def batch_loss(S):
    prob = np.exp(S) / np.exp(S).sum(1, keepdims=True)                 # softmax over the movies of the batch, row by row
    return prob, -np.log(np.diag(prob))
S = P @ Q.T; prob, rows = batch_loss(S)
print("scores (rows: people, columns: movies):", S.tolist())
print("chance given to the right movie:", np.round(np.diag(prob), 3).tolist(), " loss per person:", np.round(rows, 3).tolist(), " mean loss:", round(float(rows.mean()), 4))
print("if the two movies scored the same for everyone, the loss would be log 2 =", round(float(np.log(2)), 4))
for tau in (1.0, 0.5, 0.1): print(f"temperature {tau}: mean loss {batch_loss(S / tau)[1].mean():.4f}")
```

```text
p . q = 0.6 ; divided by tau = 1: 0.6 ; by tau = 0.1: 6.0
scores (rows: people, columns: movies): [[2.0, 0.0], [0.0, 1.0]]
chance given to the right movie: [0.881, 0.731]  loss per person: [0.127, 0.313]  mean loss: 0.2201
if the two movies scored the same for everyone, the loss would be log 2 = 0.6931
temperature 1.0: mean loss 0.2201
temperature 0.5: mean loss 0.0725
temperature 0.1: mean loss 0.0000
```

Then the real model. It needs `recsys.py` and the folder `ml-100k/` of [[Recommender Shoot-out]] in the same folder, and PyTorch (CPU is enough; it ran for roughly 10 to 20 minutes on my laptop, I did not time it exactly):

```python
# file: two_tower.py
import itertools, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from recsys import raw, NU, NI, leave_last_out, to_matrix

torch.set_num_threads(4)
train, test = leave_last_out(raw)                         # test: each person's latest movie (used once, at the end)
train2, val = leave_last_out(train)                       # for choosing settings: the latest movie of the training part
D, H, BATCH, MAXEP = 32, 64, 512, 20

class Data:                                               # people x movies matrix of known ratings, the (person, movie) pairs, the held-out movie
    def __init__(self, rows, held):
        self.B = torch.tensor((to_matrix(rows) > 0).astype("float32"))
        self.u, self.i = torch.tensor(rows[:, 0] - 1), torch.tensor(rows[:, 1] - 1); self.held = torch.tensor(held[:, 1] - 1)

class TwoTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.hist_emb = nn.Parameter(0.05 * torch.randn(NI, H))                 # user tower: a person is the average of the movies they rated
        self.user_out = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Linear(H, D))
        self.item_emb = nn.Embedding(NI, H)                                     # item tower: a movie is its own learned vector
        self.item_out = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Linear(H, D))
    def user(self, hist):                                                       # hist: people x movies, 1 = known rating
        return F.normalize(self.user_out((hist / hist.sum(1, keepdim=True).clamp(min=1)) @ self.hist_emb), dim=1)
    def item(self, idx=None):
        e = self.item_emb.weight if idx is None else self.item_emb(idx)
        return F.normalize(self.item_out(e), dim=1)

def hit10(m, d, hist=None):                                                      # held-out movie in the top 10 of the movies not yet rated
    with torch.no_grad():
        S = (m.user(d.B if hist is None else hist) @ m.item().T).masked_fill(d.B > 0, -np.inf)
        top = S[torch.arange(NU), d.held].unsqueeze(1)
        return float(((S > top).sum(1) + 1 <= 10).float().mean())

def fit(d, tau, drop, wd, epochs, seed, watch=None):
    torch.manual_seed(seed); m = TwoTower(); opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=wd); curve = []
    for ep in range(epochs):
        perm = torch.randperm(len(d.u))
        for k in range(0, len(perm), BATCH):
            b = perm[k:k + BATCH]; u, i = d.u[b], d.i[b]
            hist = d.B[u].clone(); hist[torch.arange(len(b)), i] = 0                 # hide the target movie from the person's history
            hist = hist * (torch.rand_like(hist) >= drop)                            # and forget a share of the rest (history dropout)
            logits = m.user(hist) @ m.item(i).T / tau                                # every other movie in the batch is a negative
            same = (i[:, None] == i[None, :]) & ~torch.eye(len(b), dtype=torch.bool)  # the same movie twice is not a negative
            loss = F.cross_entropy(logits.masked_fill(same, -np.inf), torch.arange(len(b)))
            opt.zero_grad(); loss.backward(); opt.step()
        if watch is not None: curve.append(hit10(m, watch))
    return m, curve

# 1. choose the settings on the validation movies (seed 0, best of 20 epochs)
dval = Data(train2, val); results = []
print("choosing settings on the validation movies (people: 943, hit@10 of their latest training movie)")
for tau, drop, wd in itertools.product((0.1, 0.3), (0.0, 0.5), (0.0, 0.1)):
    _, curve = fit(dval, tau, drop, wd, MAXEP, 0, watch=dval); best = int(np.argmax(curve))
    results.append((curve[best], best + 1, tau, drop, wd))
    print(f"  temperature {tau}, history dropout {drop}, weight decay {wd}: best validation hit@10 {curve[best]:.3f} at epoch {best + 1}", flush=True)
score, epochs, tau, drop, wd = max(results)
print(f"chosen: temperature {tau}, history dropout {drop}, weight decay {wd}, {epochs} epochs (validation {score:.3f})")

# 2. train on the whole training part and test once on the held-out movies
dtest = Data(train, test); res = []
for seed in (0, 1, 2):
    m, _ = fit(dtest, tau, drop, wd, epochs, seed); res.append(hit10(m, dtest)); print(f"test, seed {seed}: hit@10 {res[-1]:.3f}", flush=True)
    if seed == 0: model0 = m
print(f"mean of 3 seeds {np.mean(res):.3f}  (shoot-out: popular 0.050, item-kNN 0.059, user-kNN 0.074, ALS 0.087)")

# 3. a person known only by their first k ratings: the user tower still gives them a vector
order = np.lexsort((train[:, 1], train[:, 3])); tr = train[order]; print("seed 0, person known only by their first k ratings:")
for k in (1, 5, 10, 20, 1000):
    Hk = torch.zeros(NU, NI)
    for u in range(NU): Hk[u, tr[tr[:, 0] == u + 1][:k, 1] - 1] = 1
    print(f"  k = {k if k < 1000 else 'all':>4}: hit@10 {hit10(model0, dtest, Hk):.3f}")
```

```text
choosing settings on the validation movies (people: 943, hit@10 of their latest training movie)
  temperature 0.1, history dropout 0.0, weight decay 0.0: best validation hit@10 0.043 at epoch 4
  temperature 0.1, history dropout 0.0, weight decay 0.1: best validation hit@10 0.041 at epoch 6
  temperature 0.1, history dropout 0.5, weight decay 0.0: best validation hit@10 0.055 at epoch 14
  temperature 0.1, history dropout 0.5, weight decay 0.1: best validation hit@10 0.054 at epoch 12
  temperature 0.3, history dropout 0.0, weight decay 0.0: best validation hit@10 0.048 at epoch 5
  temperature 0.3, history dropout 0.0, weight decay 0.1: best validation hit@10 0.048 at epoch 14
  temperature 0.3, history dropout 0.5, weight decay 0.0: best validation hit@10 0.050 at epoch 18
  temperature 0.3, history dropout 0.5, weight decay 0.1: best validation hit@10 0.048 at epoch 11
chosen: temperature 0.1, history dropout 0.5, weight decay 0.0, 14 epochs (validation 0.055)
test, seed 0: hit@10 0.059
test, seed 1: hit@10 0.058
test, seed 2: hit@10 0.057
mean of 3 seeds 0.058  (shoot-out: popular 0.050, item-kNN 0.059, user-kNN 0.074, ALS 0.087)
seed 0, person known only by their first k ratings:
  k =    1: hit@10 0.029
  k =    5: hit@10 0.033
  k =   10: hit@10 0.038
  k =   20: hit@10 0.046
  k =  all: hit@10 0.059
```

Reading it:
- **The eight settings** scored 0.041 to 0.055 on the validation movies. With 943 people a score near 0.05 has a noise of about 0.007 (my arithmetic: $\sqrt{0.05\times0.95/943}$), so most of these differences are not real; the history dropout looked most useful (0.054 and 0.055 for the dropout rows at temperature 0.1, against 0.041 to 0.048 for all four zero-dropout rows). I took the highest.
- **The test result** is 0.059, 0.058 and 0.057 for three seeds, mean **0.058**. In the shoot-out the same test gave popular 0.050, item-kNN 0.059, user-kNN 0.074 and ALS 0.087. So this model is as good as item-kNN, a little above the popularity list, and clearly below user-kNN and ALS.
- **A person known only by their first k ratings** (seed 0): 0.029 for k=1, 0.033 (5), 0.038 (10), 0.046 (20) and 0.059 with the whole history. The tower does produce a vector from a short history, but with fewer than 20 ratings it scored under the popularity list (0.050). It did not solve cold start on this data.

## 🔴 4. Deep / formal
**The score** of person $u$ and item $i$, with the two tower outputs $\mathbf p_u$ and $\mathbf q_i$ scaled to length 1:

$$
s(u,i)=\frac{\mathbf p_u\cdot\mathbf q_i}{\tau}
$$

**Where:**
- $\mathbf p_u$ is the user tower's output for person $u$ and $\mathbf q_i$ the item tower's output for item $i$ (vectors of the same length, here 32).
- $\mathbf p_u\cdot\mathbf q_i$ is their dot product; for length-1 vectors it equals the cosine of the angle between them (from 1 for the same direction to −1 for opposite).
- $\tau$ is the temperature, a small positive number.
- **Matters:** dividing by a small $\tau$ stretches the scores apart, so the softmax below can become confident. Without it, the scores stay inside −1 to 1 and the loss can never get small, however good the towers are.

**Example (tiny numbers):** $\mathbf p=(0.6,\,0.8)$ and $\mathbf q=(1,\,0)$.
- Dot product: $0.6\times1+0.8\times0=\mathbf{0.6}$.
- With $\tau=1$ the score is $0.6$; with $\tau=0.1$ it is $\mathbf{6.0}$.
- Meaning: the same pair looks ten times stronger with the small temperature.

**The training loss** over a batch of $B$ pairs $(u_b,i_b)$, where item $i_b$ is the one person $u_b$ chose:

$$
\mathcal L=-\frac1B\sum_{b=1}^{B}\log\frac{\exp s(u_b,i_b)}{\sum_{j=1}^{B}\exp s(u_b,i_j)}
$$

**Where:**
- $B$ is the batch size (512 in my run) and $b$ numbers the pairs.
- The fraction is the chance the softmax gives the right item $i_b$ among the $B$ items of the batch; $\log$ turns it into a cost that is 0 when the chance is 1.
- $j$ runs over the items of the batch: the right one and the $B-1$ wrong ones.
- **Matters:** the other items of the batch serve as free wrong answers, with no extra sampling. But popular items appear in batches more often, so they are used as wrong answers more often, which pushes their scores down (from memory, a known effect with a correction; I did not test it). Without wrong answers the towers could give every pair the same vector and score perfectly.

**Example (tiny numbers):** two people with $\tau=1$, person vectors $(1,0)$ and $(0,1)$, and their movies $(2,0)$ and $(0,1)$.
- Scores: person 1 gives $2$ to their movie and $0$ to the other; person 2 gives $1$ to theirs and $0$ to the other.
- Chance of the right movie: $e^2/(e^2+1)=\mathbf{0.881}$ and $e^1/(e^1+1)=\mathbf{0.731}$.
- Costs: $-\ln0.881=0.127$ and $-\ln0.731=0.313$. Loss = $(0.127+0.313)/2=\mathbf{0.220}$.
- Meaning: if both movies scored the same for everyone the loss would be $\ln2=0.693$; lower means the towers separate the right movie better.

**Limits and open points:**
- **No gain on this small data.** My model (0.058) did not beat the matrix factorization of the shoot-out (ALS 0.087). That is one small data set, one design and 8 settings; the design's real strengths (features, serving speed, new people) are not shown by it.
- **The dot product is the only way the two sides meet.** A tower cannot look at the pair together, so it cannot use "this person plus this item" interactions. Large systems therefore often use the two-tower model to pick a few hundred candidates, then a heavier model that looks at pairs to rank them (from general knowledge, not run here).
- **The item tower here is an ID table, so a new movie still has no vector.** A real item tower reads features such as genre or text; I did not try that.
- **My first try failed and I fixed it.** An earlier version without history dropout, weight decay or a validation split reached a test score of 0.022, below the popularity list. Its training loss fell from 6.14 to 1.80 over 20 epochs while the test score peaked at 0.043 after epoch 5 and fell to 0.025: the towers memorised the training pairs. (That version of the script was not kept, so these numbers are copied from my terminal and are not reproduced by the code above; the first row of the table above, 0.043 at epoch 4 with no dropout, is the same setup on the validation movies.)
- **Matrix factorization is the special case.** A two-tower model whose towers are only ID tables, trained with this loss, is matrix factorization with a different loss ([[Matrix Factorization]]; my reasoning, not tested).

**What came before, and how this improved on it:**
- **Before:** [[Matrix Factorization]] and neighbour methods learn one fixed vector per person and per item from the table of past actions.
- **Its limit:** they cannot use features and cannot give a vector to a person or item that is not in the table.
- **What changed:** let a network turn features or history into the vector, one network per side, so item vectors can be pre-computed and searched quickly. The idea comes from "twin tower" models for web search and from YouTube's candidate-generation network (from memory; I did not check titles or dates).
- **How much better:** not better on this data. In my run the two-tower got 0.058 against 0.087 for ALS (hit@10, 943 people), and 0.059 for item-kNN. The design's advantages (features, serving by nearest-neighbour search) were not measured here.

## Related
- [[Recommender Systems]] · [[Matrix Factorization]] · [[Collaborative Filtering]] · [[Cold Start Problem]] · [[Embedding]] · [[Cosine Similarity]] · [[Vector Database]] · [[Contrastive Learning]] · [[Retrieval Metrics]] · [[Recommender Shoot-out]] · [[Embedding Model]]

## Sources
- My own code and runs (Python 3.9.6, PyTorch 2.8.0 CPU, NumPy 1.24.4; data and test from [[Recommender Shoot-out]]). The comparison scores (popular, item-kNN, user-kNN, ALS) are from that project's `ranks.npz` in [[Error Bars Lab]]. Every code block in this note was extracted into a fresh folder (with copies of the files it needs from the earlier projects) and run again; each output matched the block above. Not tested: another machine or other library versions.
- The origin of the idea, the sampling-bias remark, the two-stage use and the closeness to matrix factorization are from memory or my own reasoning; I read no paper for this note.
- No image (see level 2).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/recommenders Two-tower model:::A model with two separate networks, a user tower and an item tower, each turning what is known into a vector; the score is the dot product of the two vectors, so item vectors can be pre-computed and searched by nearest neighbour.

#flashcards/ml/formula #flashcards/topic/recommenders Write the score of a two-tower model and explain each symbol.
?
$$s(u,i)=\frac{\mathbf p_u\cdot\mathbf q_i}{\tau}$$
- $\mathbf p_u$, $\mathbf q_i$: the user tower's and item tower's output vectors (same length), scaled to length 1.
- $\mathbf p_u\cdot\mathbf q_i$: their dot product, the cosine of the angle between them.
- $\tau$: the temperature. A small value stretches the scores apart so the softmax can become confident; without it the loss can never get small.

#flashcards/ml/example #flashcards/topic/recommenders With p = (0.6, 0.8) and q = (1, 0), what is the score at temperature 1 and at 0.1?::The dot product is 0.6 × 1 + 0.8 × 0 = 0.6. Divided by 1 it is 0.6; divided by 0.1 it is 6.0 (my arithmetic, checked in the code).

#flashcards/ml/formula #flashcards/topic/recommenders Write the in-batch softmax loss of a two-tower model and explain each symbol.
?
$$\mathcal L=-\frac1B\sum_{b=1}^{B}\log\frac{\exp s(u_b,i_b)}{\sum_{j=1}^{B}\exp s(u_b,i_j)}$$
- $B$: the batch size; $b$ numbers the pairs $(u_b,i_b)$, person and the item they chose.
- The fraction: the chance the softmax gives the right item among the $B$ items of the batch; the log makes it a cost that is 0 when the chance is 1.
- $j$: runs over the items of the batch (the right one and $B-1$ wrong ones). Popular items are used as wrong answers more often, which pushes their scores down (from memory, not tested).

#flashcards/ml/example #flashcards/topic/recommenders Two people with person vectors (1,0) and (0,1) and movies (2,0) and (0,1), temperature 1: what is the loss?::The scores of the right movies are 2 and 1, the wrong ones 0. Chances: e^2/(e^2+1) = 0.881 and e/(e+1) = 0.731. Costs 0.127 and 0.313, mean 0.220. If the movies scored equally the loss would be ln 2 = 0.693 (my arithmetic, checked in the code).

#flashcards/ml/concept #flashcards/topic/recommenders What are the steps of a two-tower model?
?
The user tower makes a vector from the person's features or history; the item tower makes a vector from the item; the score is their dot product (cosine for length 1); training makes each person's own item beat the other items in the batch (in-batch negatives); at serving time the item vectors are pre-computed and searched by nearest neighbour.

#flashcards/ml/concept #flashcards/topic/recommenders How did my two-tower model do on the MovieLens shoot-out test?
?
Hit@10 of 0.059, 0.058, 0.057 for three seeds (mean 0.058), against popular 0.050, item-kNN 0.059, user-kNN 0.074 and ALS 0.087. As good as item-kNN but clearly below ALS. One small data set, 943 people, 8 settings tried on a validation split (noise about 0.007).

#flashcards/ml/concept #flashcards/topic/recommenders Did the two-tower user tower solve cold start on MovieLens?
?
No. With only the first k ratings it scored 0.029 (k=1), 0.033 (5), 0.038 (10), 0.046 (20) and 0.059 (all), under the popularity list (0.050) until the full history. The item tower was an ID table, so new movies were not handled either.

#flashcards/ml/concept #flashcards/topic/recommenders What went wrong in my first two-tower try?
?
No dropout, weight decay or validation split: the training loss fell from 6.14 to 1.80 over 20 epochs while the test score peaked at 0.043 after epoch 5 and fell to 0.025 (mean 0.022 over three seeds): it memorised the training pairs. These numbers were copied from my terminal and are not reproduced by the kept code.

#flashcards/ml/concept #flashcards/topic/recommenders What are the limits of a two-tower model?
?
The two sides meet only in a dot product, so it cannot use pair interactions (big systems rank candidates with a heavier model, from general knowledge); in-batch negatives favour popularity effects; an ID-table item tower cannot embed a new item; and on my small data it did not beat matrix factorization.

#flashcards/ml/cloze #flashcards/topic/recommenders In a two-tower model the ==item vectors== can be computed in advance and searched by ==nearest neighbour==, because the two towers only meet in a final dot product.
