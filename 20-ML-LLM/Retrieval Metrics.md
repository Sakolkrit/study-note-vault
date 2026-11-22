---
tags: [retrieval, evaluation, RAG, metrics, precision, recall, NDCG]
status: growing
level: 4
tier: 2
---

# Retrieval Metrics

Numbers that say how good a search system's ranked list is: **precision@k**, **recall@k**, **reciprocal rank** and **NDCG@k**.

## 🟢 1. Simple (age 12)
You ask a librarian for books about bears and they hand you a pile in order. Retrieval metrics are the **report card** for that pile: Did the good books show up? Did the best one come first? Did you get most of the good books that exist? Different questions get different scores.

**Example:** The librarian gives 5 books and 2 are good, with 3 good books existing in the whole library. Precision is 2 out of 5 (how clean the pile is). Recall is 2 out of 3 (how many of the good books you got).

## 🔵 2. How it works
1. A [[RAG]] system first **retrieves** chunks of text, then the [[LLM]] writes an answer from them. If the right chunk is not retrieved, the answer cannot be good, so retrieval is measured on its own.
2. For each question we know which chunks are relevant (labelled by people or a model).
3. The system returns a ranked list. We look only at the top $k$.
4. **Precision@k:** of the $k$ returned, what share is relevant? It measures *cleanliness*. The lecture pairs it with the second stage (ranking).
5. **Recall@k:** of all relevant chunks that exist, what share is in the top $k$? It measures *coverage*. It pairs with the first stage (candidate retrieval).
6. **Reciprocal rank:** one divided by the rank of the **first** relevant result. Rank 1 gives 1, rank 5 gives 0.2.
7. **NDCG@k:** also cares about **order** and about **how relevant** each item is (graded), giving more credit to good results near the top.

## 🟠 3. Technical detail
- **Stages:** step 1 (candidate retrieval, for example [[BM25]] or embeddings) wants high recall, since the shortlist must contain the answers. Step 2 (re-ranking) wants high precision, so the top of the list is clean.
- **Order matters for some metrics, not others:** precision and recall at $k$ ignore the order inside the top $k$. Reciprocal rank and NDCG reward putting relevant items high.
- **Graded relevance:** NDCG uses scores such as 3, 2, 1, 0 for "very relevant" down to "not relevant". Binary labels are a special case.
- **Averages:** each metric is computed for every question and then averaged over a test set (mean reciprocal rank is the average of the reciprocal ranks).
- **Check against a library:** the code's NDCG matches scikit-learn's `ndcg_score` for the full list and for $k=3$.
- **Choosing $k$:** depends on the use. A chat app that passes the top 5 chunks to the model cares about $k=5$.
- **Limit:** these measure retrieval only. A good retrieval score does not guarantee a good final answer, and labels for relevance are costly.

```python
import numpy as np
from sklearn.metrics import ndcg_score
rel = [1, 0, 1, 0, 0]                  # is each retrieved chunk relevant, in the order the system ranked them
total_relevant = 3                     # relevant chunks that exist in the whole knowledge base
for k in (1, 3, 5):
    hit = sum(rel[:k])
    print(f'k = {k} | precision@k {hit / k:.3f} | recall@k {hit / total_relevant:.3f}')
rr = next((1 / (i + 1) for i, r in enumerate(rel) if r), 0)
print('reciprocal rank (first relevant at rank', rel.index(1) + 1, '):', rr)
# NDCG: graded relevance (2 = very relevant, 1 = somewhat, 0 = not); gain / log2(rank + 1)
g = [3, 2, 3, 0, 1]
dcg = lambda g: sum(x / np.log2(i + 2) for i, x in enumerate(g))
ideal = sorted(g, reverse=True)
print('DCG', round(dcg(g), 4), '| ideal DCG', round(dcg(ideal), 4), '| NDCG', round(dcg(g) / dcg(ideal), 4))
print('scikit-learn ndcg_score:', round(ndcg_score([g], [[5, 4, 3, 2, 1]]), 4), '(scores 5..1 keep the same order)')
print('first 3 only:', round(dcg(g[:3]) / dcg(ideal[:3]), 4), '| scikit-learn:', round(ndcg_score([g], [[5, 4, 3, 2, 1]], k=3), 4))
# the same list in two orders
print('relevant item at rank 1 vs rank 5 -> reciprocal rank 1 vs', 1 / 5, '| NDCG of [0,0,0,0,3] against ideal [3,0,0,0,0]:', round(dcg([0, 0, 0, 0, 3]) / dcg([3, 0, 0, 0, 0]), 4))
```

## 🔴 4. Deep / formal
**Precision and recall at $k$.** Let $\text{hits}_k$ be how many of the top $k$ results are relevant and $R$ the total number of relevant chunks that exist:

$$
\text{precision@}k=\frac{\text{hits}_k}{k},\qquad \text{recall@}k=\frac{\text{hits}_k}{R}
$$

**Where:**
- $k$ is how many results from the top of the list we look at.
- $\text{hits}_k$ is the number of relevant ones among them.
- $R$ is the number of relevant chunks in the whole collection.
- **Matters:** precision punishes junk in the list and recall punishes missing items. Raising $k$ usually raises recall and lowers precision, so each alone can be gamed (return everything for perfect recall).

**Example (tiny numbers):** ranked list with relevance $[1,0,1,0,0]$ and $R=3$.
- $k=3$: $\text{hits}=2$, so precision $=2/3=\mathbf{0.667}$ and recall $=2/3=\mathbf{0.667}$.
- $k=5$: $\text{hits}=2$, so precision $=2/5=\mathbf{0.4}$ and recall $=\mathbf{0.667}$.
- Meaning: looking at two more results added no new hits, so precision fell and recall stayed the same.

**Reciprocal rank.** With $\text{rank}_1$ the position of the first relevant result:

$$
\text{RR}=\frac{1}{\text{rank}_1}
$$

**Where:**
- $\text{rank}_1$ is 1 if the first result is relevant, 2 if the second is the first relevant one, and so on.
- **Matters:** it only cares about the first good hit and falls off fast: rank 2 gives 0.5, rank 10 gives 0.1. It suits "find me one good answer" cases.

**Example (tiny numbers):** relevance $[0,0,1,0,0]$.
- First relevant at rank 3, so $\text{RR}=1/3=\mathbf{0.333}$.
- For the code's list $[1,0,1,0,0]$ the first is at rank 1, so $\text{RR}=\mathbf{1}$.
- Meaning: a relevant item at rank 1 is worth three times one at rank 3.

**NDCG.** With graded relevance $g_i$ at rank $i$:

$$
\text{DCG}@k=\sum_{i=1}^{k}\frac{g_i}{\log_2(i+1)},\qquad \text{NDCG}@k=\frac{\text{DCG}@k}{\text{IDCG}@k}
$$

**Where:**
- $g_i$ is the relevance grade of the item at rank $i$ (for example 3, 2, 1, 0).
- $\log_2(i+1)$ is a discount that gets bigger further down the list, so lower ranks count less.
- $\text{IDCG}@k$ is the DCG of the *perfect* ordering (the same grades sorted from best to worst).
- **Matters:** dividing by the ideal puts the score between 0 and 1 and makes lists with different numbers of good items comparable. Without the discount, order would not matter.

**Example (tiny numbers):** grades $[3,2,3,0,1]$.
- $\text{DCG}=3/1+2/1.585+3/2+0/2.322+1/2.585=3+1.262+1.5+0+0.387=\mathbf{6.149}$.
- Ideal order $[3,3,2,1,0]$: $3+3/1.585+2/2+1/2.322+0=3+1.893+1+0.431=\mathbf{6.324}$.
- $\text{NDCG}=6.149/6.324=\mathbf{0.972}$.
- Meaning: the list is 97% as good as the perfect ordering. A single grade-3 item pushed from rank 1 to rank 5 gets only $0.387$ (the code's last line).

**Further notes:**
- **Not covered:** mean average precision, how to build relevance labels, and significance tests.

**What came before, and how this improved on it:**
- **Before:** precision and recall, which count hits but ignore order inside the list (from the information-retrieval tradition; I did not check the history).
- **Its limit:** a relevant item at rank 1 and at rank 10 both count the same inside the top 10, and all relevant items are treated as equally useful.
- **What changed:** order-aware scores: reciprocal rank (first hit), and NDCG (graded relevance with a position discount).
- **How much better:** in the code, moving one grade-3 item from the top to the bottom of a five-item list leaves precision@5 and recall@5 unchanged but cuts NDCG to $0.387$ and the reciprocal rank from 1 to 0.2. That is an illustration, not an experiment on a real system.

## Related
- [[RAG]] · [[BM25]] · [[Embedding Model]] · [[Vector Database]] · [[Evaluation Metrics]] · [[Cosine Similarity]] · [[LLM as a Judge]] · [[LLM]] · [[Contextual Retrieval]] · [[Precision]] · [[Recall]] · [[Vault Search with RAG]] · [[Chunking Strategies]] · [[Reranking]] · [[RAG Tuning Lab]]
- [[Recommender Systems]] · [[Recommender Shoot-out]]
- [[Confidence Interval]] · [[Bootstrap]] · [[Hypothesis Testing]]
- [[AB Testing|A/B Testing]]
## Sources
- Stanford CME 295 (Fall 2025), Lecture 7 slides, Afshine and Shervine Amidi
- scikit-learn documentation, `ndcg_score` (used for the check in the code)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation Retrieval metrics:::The report card for a ranked pile of search results. Did the good items show up? Did the best come first? Did you get most of the good ones that exist?

#flashcards/ml/formula #flashcards/topic/evaluation Write precision@k and recall@k and explain each symbol.
?
$$\text{precision@}k=\frac{\text{hits}_k}{k},\qquad \text{recall@}k=\frac{\text{hits}_k}{R}$$
- $k$: how many results from the top of the list we look at. $\text{hits}_k$: the number of relevant ones among them.
- $R$: the number of relevant chunks in the whole collection.
- Precision punishes junk in the list and recall punishes missing items. Raising $k$ usually raises recall and lowers precision.

#flashcards/ml/formula #flashcards/topic/evaluation Write the reciprocal rank and explain each symbol.
?
$$\text{RR}=\frac{1}{\text{rank}_1}$$
- $\text{rank}_1$: 1 if the first result is relevant, 2 if the second is the first relevant one, and so on.
- It only cares about the first good hit and falls off fast: rank 2 gives 0.5, rank 10 gives 0.1.
- It suits "find me one good answer" cases.

#flashcards/ml/formula #flashcards/topic/evaluation Write DCG and NDCG and explain each symbol.
?
$$\text{DCG}@k=\sum_{i=1}^{k}\frac{g_i}{\log_2(i+1)},\qquad \text{NDCG}@k=\frac{\text{DCG}@k}{\text{IDCG}@k}$$
- $g_i$: the relevance grade of the item at rank $i$ (for example 3, 2, 1, 0).
- $\log_2(i+1)$: a discount that gets bigger further down the list, so lower ranks count less.
- $\text{IDCG}@k$: the DCG of the perfect ordering (the same grades sorted best to worst).
- Dividing by it gives a score between 0 and 1, where 1 is a perfect ordering.

#flashcards/ml/cloze #flashcards/topic/evaluation Precision@k and recall@k trade off, and ==NDCG== also rewards putting the best items ==first==.

#flashcards/ml/example #flashcards/topic/evaluation A ranked list has relevance [1, 0, 1, 0, 0] and $R=3$. What are precision and recall at $k=3$ and $k=5$?::k = 3: hits 2, precision 2/3 = 0.667, recall 2/3 = 0.667. k = 5: hits 2, precision 2/5 = 0.4, recall 0.667. Two more results added no new hits, so precision fell and recall stayed.

#flashcards/ml/example #flashcards/topic/evaluation What is the reciprocal rank of the relevance list [0, 0, 1, 0, 0] and of [1, 0, 1, 0, 0]?::First relevant at rank 3: 1/3 = 0.333. For [1, 0, 1, 0, 0] the first is at rank 1: RR = 1. A hit at rank 1 is worth three times one at rank 3.

#flashcards/ml/example #flashcards/topic/evaluation Grades are [3, 2, 3, 0, 1]. What are DCG, IDCG and NDCG?::DCG = 3/1 + 2/1.585 + 3/2 + 0/2.322 + 1/2.585 = 6.149. Ideal order [3, 3, 2, 1, 0]: 3 + 1.893 + 1 + 0.431 = 6.324. NDCG = 6.149/6.324 = 0.972, so the list is 97% as good as the perfect order.
