---
tags: [cosine-similarity, vector, linear-algebra, math, embedding, RAG]
status: growing
level: 4
tier: 1
---

# Cosine Similarity

## 🟢 1. Simple (age 12)
Cosine similarity tells you **how much two arrows point the same way**, ignoring how long they are.

**Example:** Two friends walk out of the same door. If both head north, they point the same way: score **1**. If one goes north and one goes east, they are at a right angle: score **0**. If one goes north and the other south, they point opposite ways: score **-1**.

In AI, "pointing the same way" means "meaning the same kind of thing". The arrows for "car" and "truck" point nearly the same way, so their score is high. "Car" and "banana" point far apart, so the score is low.

## 🔵 2. How it works
1. Each word, sentence or picture is a [[Vector]] (an arrow, written as a list of numbers).
2. Measure the **angle** between the two arrows. A small angle means similar.
3. Turn the angle into a score: small angle gives a score near 1, 90° gives 0, 180° gives -1.
4. To search: compare your question's arrow with every stored arrow, and return the ones with the highest score. This is how [[RAG]] finds text relevant to your question.

Why ignore length? A long document and a short note on the same topic have arrows of different lengths but nearly the same direction. Direction carries the *meaning*, so comparing direction is fairer.

![[cosine-similarity.png]]
*Words as arrows from the origin: "car" and "truck" are 80° apart. A smaller angle means more similar. Image: StandingFuture, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:CosineSimilarity.png), CC BY-SA 4.0.*

## 🟠 3. Technical detail
Formula (dot product divided by the product of lengths):

$$
\cos(u,v)=\frac{u\cdot v}{\lVert u\rVert\,\lVert v\rVert}
$$

**Where:**
- $u$ and $v$ are the two [[Vector]]s being compared (for example the embeddings of "car" and "truck"). They must have the same number of entries.
- $u\cdot v$ is the **dot product**: multiply the numbers in matching positions, then add them all. **Matters:** it is big and positive when the two vectors push the same way, near 0 when unrelated, negative when opposed. This is the "how aligned?" part.
- $\lVert u\rVert$ and $\lVert v\rVert$ are the **lengths** of the vectors ($\sqrt{u\cdot u}$). **Matters:** dividing by both removes length from the answer, so only the angle between them counts. Without this, a long vector would look "more similar" to everything.
- $\cos(u,v)$ is the result, the cosine of the angle between the arrows.

**Example (tiny numbers):** `car` $=[0.9,\,0.4]$, `truck` $=[0.8,\,0.6]$, `banana` $=[-0.4,\,0.9]$.
- `car` vs `truck`: dot $=0.72+0.24=0.96$. Lengths $0.985$ and $1.0$. Score $\approx\mathbf{0.97}$ (almost the same direction).
- `car` vs `banana`: dot $=-0.36+0.36=0$. Score $=\mathbf{0}$ (right angle: unrelated).
- A search for "car" would list `truck` first and skip `banana`.

Range is $[-1,1]$. In practice text embeddings usually give scores between about 0 and 1.

Related measures:
- **Cosine distance** $=1-\cos(u,v)$, so 0 means identical direction.
- **Dot product:** same as cosine if vectors are normalised to length 1. Many embedding models output unit vectors, so the fast dot product is used.
- **Euclidean distance:** compares position, so length matters. For unit vectors it is monotonic in cosine: $\lVert u-v\rVert^2=2-2\cos(u,v)$.

```python
import numpy as np
def cos(u, v): return u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
a, b = np.array([1., 2., 3.]), np.array([2., 4., 6.])   # same direction, b is 2x longer
c = np.array([3., 0., -1.])
print(cos(a, b), cos(a, c))     # 1.0  and  ~0.0  (a . c = 0)
```

Practical notes:
- Normalise once at index time (store unit vectors), then search with dot product. Much cheaper at scale in a [[Vector Database]].
- Scores are only comparable within one embedding model. "0.8" means different things in different models.
- A zero vector has no direction, so cosine is undefined; guard against it.

## 🔴 4. Deep / formal
From the dot product identity $u\cdot v=\lVert u\rVert\lVert v\rVert\cos\theta$, cosine similarity is $\cos\theta$ where $\theta$ is the angle between $u$ and $v$. Cauchy–Schwarz guarantees $|\cos\theta|\le1$.

**Is cosine distance a metric?** $1-\cos$ violates the triangle inequality in general. The **angular distance** $\theta=\arccos(\cos(u,v))$ does satisfy it (it is the geodesic distance on the unit sphere). So algorithms that need a true metric should use angular distance, or unit-normalise and use Euclidean distance. The figure below compares the two normalised distances: both go from 0 to 1, with $(1-\cos\theta)/2$ flatter near 0 and $\pi$ and steeper in the middle.

![[cosine-distance.png]]
*Angular distance $\theta/\pi$ (blue) vs $(1-\cos\theta)/2$ (orange). Image: Cburke91, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Cosine-distance.png), CC BY-SA 4.0.*

**Concentration in high dimension:** for independent random unit vectors in $\mathbb{R}^n$, $\cos\theta$ has mean 0 and variance $1/n$. So with $n=768$ unrelated texts score near $0\pm0.036$, and a score of 0.3 is already far in the tail. This is why embeddings work, and why a single global threshold is fragile.

**Contrastive training link:** modern embedding models are trained with a loss that maximises cosine similarity of matching pairs relative to others, with temperature $\tau$:

$$
\mathcal{L}=-\log\frac{\exp(\cos(q,d^+)/\tau)}{\sum_j\exp(\cos(q,d_j)/\tau)}
$$

**Where:**
- $\mathcal{L}$ is the loss: one number measuring how wrong the model is. Training lowers it ([[Gradient Descent]]).
- $q$ is a query; $d^+$ is its correct match; $d_j$ runs over all candidates in the batch (the correct one plus wrong ones).
- $\cos(q,d)$ is the similarity from this note. **Matters:** the loss rewards high cosine for the right pair and low cosine for the wrong ones.
- $\tau$ (tau) is the **temperature**, a small positive number. Dividing by it stretches the scores, so near-misses are punished harder. **Matters:** because cosine only spans -1 to 1, without $\tau$ the contest would be too flat for the model to learn from.
- $\exp(\cdot)$ makes every score positive; the fraction is then the chance of picking the right match out of the lineup. $-\log$ turns "chance of being right" into a loss: near 1 gives about 0 loss, a tiny chance gives a big loss.

**Example (tiny numbers):** one question, cosine scores: correct match $0.8$, wrong ones $0.3$ and $0.1$.
- $\tau=0.1$: scores $8,\ 3,\ 1$. $\exp$: $2981,\ 20.1,\ 2.7$. Chance of the right one $=0.992$. Loss $=\mathbf{0.008}$.
- $\tau=1$: scores $0.8,\ 0.3,\ 0.1$. $\exp$: $2.23,\ 1.35,\ 1.11$. Chance $=0.475$. Loss $=\mathbf{0.74}$.
- Cosine only moves between $-1$ and $1$, so without a small $\tau$ the gap between right and wrong looks tiny and learning is weak.

Search at scale uses approximate nearest-neighbour indexes (HNSW, IVF, product quantisation) rather than comparing against every vector.

**What came before, and how this improved on it:**
- **Before:** compare two vectors by the straight-line distance between their tips.
- **Its limit:** length matters as much as direction, so a text and the same text repeated ten times look far apart.
- **What changed:** compare only the angle, by dividing the dot product by the two lengths. The idea is old geometry; its use for comparing documents was made popular by the vector space model of Salton and colleagues in the 1960s and 1970s (from memory, not checked).
- **How much better:** in the code below, a short text and the same text repeated 10 times are 20.125 apart in a straight line but have a cosine similarity of 1.0. An unrelated text is closer in straight-line distance (5.196) yet has a cosine similarity of only 0.175.

**Check it yourself:**

```python
import numpy as np
short = np.array([1.0, 2.0, 0.0]); long = 10 * short; other = np.array([0.0, 1.0, 5.0])
cos = lambda a, b: a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
print("short text and the same text repeated 10 times (same direction):")
print("  straight-line distance:", round(np.linalg.norm(short - long), 3), "  cosine similarity:", round(cos(short, long), 3))
print("short text and an unrelated text:")
print("  straight-line distance:", round(np.linalg.norm(short - other), 3), "  cosine similarity:", round(cos(short, other), 3))
```

## Related
- [[Vector]] · [[Embedding]] · [[Vector Database]] · [[RAG]] · [[Attention]] · [[k-NN]]
- [[Collaborative Filtering]] · [[Local Outlier Factor]]
- [[Two-Tower Model]]
## Sources
- Mikolov et al., *Efficient Estimation of Word Representations in Vector Space*, 2013
- Manning, Raghavan & Schütze, *Introduction to Information Retrieval*, ch. 6 (vector space model)

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/linear-algebra Cosine similarity:::A score for how much two arrows point the same way, ignoring their length. 1 means same direction, 0 a right angle, -1 opposite.

#flashcards/math/formula #flashcards/topic/linear-algebra Write the cosine similarity formula and explain each symbol.
?
$$\cos(u,v)=\frac{u\cdot v}{\lVert u\rVert\,\lVert v\rVert}$$
- $u$, $v$: the two vectors being compared (for example the embeddings of "car" and "truck"). They need the same number of entries.
- $u\cdot v$: the dot product, multiply matching positions and add. Big and positive when they push the same way, near 0 when unrelated, negative when opposed.
- $\lVert u\rVert$, $\lVert v\rVert$: the lengths of the vectors. Dividing by both removes length from the answer, so only the angle counts.

#flashcards/math/formula #flashcards/topic/linear-algebra Write the contrastive (InfoNCE-style) loss that uses cosine similarity and explain each symbol.
?
$$\mathcal{L}=-\log\frac{\exp(\cos(q,d^+)/\tau)}{\sum_j\exp(\cos(q,d_j)/\tau)}$$
- $\mathcal{L}$: the loss, one number for how wrong the model is. Training lowers it.
- $q$: a query. $d^+$: its correct match. $d_j$: every candidate in the batch (the correct one plus wrong ones).
- $\cos(q,d)$: the similarity from this note. The loss rewards high cosine for the right pair and low for the wrong ones.
- $\tau$: the temperature, a small positive number. Dividing by it stretches the scores, so near-misses are punished harder. Without it the contest would be too flat, since cosine only spans -1 to 1.

#flashcards/math/cloze #flashcards/topic/linear-algebra Cosine similarity is ==1== for the same direction, ==0== at a right angle and ==-1== for opposite directions.

#flashcards/math/cloze #flashcards/topic/linear-algebra In the contrastive loss, a small temperature $\tau$ ==stretches== the scores, so near-misses are ==punished harder==.

#flashcards/math/example #flashcards/topic/linear-algebra `car` = [0.9, 0.4], `truck` = [0.8, 0.6], `banana` = [-0.4, 0.9]. What is the cosine of car vs truck, and car vs banana?::car vs truck: dot 0.96, lengths 0.985 and 1.0, score about 0.97 (almost the same direction). car vs banana: dot 0, score 0 (right angle, unrelated).

#flashcards/math/example #flashcards/topic/linear-algebra Correct match has cosine 0.8, wrong ones 0.3 and 0.1. What is the loss with $\tau=0.1$ and with $\tau=1$?::τ = 0.1: chance of the right one 0.992, loss 0.008. τ = 1: chance 0.475, loss 0.74. A small temperature makes the loss sharper.

#flashcards/math/image #flashcards/topic/linear-algebra ![[cosine-similarity.png]]
What does this picture show?
?
Words as arrows from the origin: "car" and "truck" are 80° apart. A smaller angle means more similar.

#flashcards/math/image #flashcards/topic/linear-algebra ![[cosine-distance.png]]
What two curves does this picture compare?
?
Angular distance θ/π (blue) against (1 − cos θ)/2 (orange), two ways to turn cosine similarity into a distance.
