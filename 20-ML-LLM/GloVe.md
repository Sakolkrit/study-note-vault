---
tags: [machine-learning, NLP, embedding, word-vectors, co-occurrence]
status: growing
level: 4
tier: 2
---

# GloVe

GloVe (short for **Global Vectors**) is a way to learn word vectors by **counting how often words appear near each other across the whole text, then fitting vectors to those counts**.

## 🟢 1. Simple (age 12)
Imagine a giant table. Down the side are all the words, and across the top are all the words too. In each box you write **how many times those two words showed up close together** in a huge pile of books. "ice" and "cream" get a big number. "ice" and "tractor" get a tiny one. GloVe then gives every word a short code of numbers, chosen so that **words whose codes line up have big numbers in the table**.

**Example:** In your table, "dog" and "cat" both have big numbers with "pet", "fed" and "walk", and small numbers with "volcano". So their rows look alike. GloVe gives "dog" and "cat" similar codes because their rows look alike.

## 🔵 2. How it works
1. Go through the whole text once and count, for every pair of words, how often they appear within a few words of each other. Closer neighbours can count more. This is the **co-occurrence table**.
2. Give each word a random list of numbers, a [[Vector]], as in [[Embedding]]. (Each word actually gets two lists, one for when it is the centre and one for when it is a neighbour, like in [[Word2Vec]].)
3. For a pair of words, multiply their two vectors together (a dot product). The goal: this number should match the **logarithm of the count** from the table.
4. Where the match is bad, nudge the numbers (see [[Stochastic Gradient Descent]]). Pairs with very small counts are trusted less, and the very biggest counts are not allowed to dominate.
5. After many nudges, keep the vectors. Words used alike now sit close together, and you can measure it with [[Cosine Similarity]].

No picture here: I did not find a freely licensed Wikimedia Commons image on GloVe.

The difference from [[Word2Vec]] is the way of learning. Word2Vec slides over the text and learns from one small window at a time. GloVe first **counts everything**, then learns from the table.

## 🟠 3. Technical detail
- **Where it comes from:** introduced by Pennington, Socher and Manning in 2014 (authors, venue and date checked on the ACL Anthology page; that they were at Stanford is from memory). Pre-trained GloVe vectors trained on large text collections are freely available and were widely used as a starting point for NLP models.
- **Idea in one line:** fit vectors so that the dot product of two word vectors (plus a bias for each word) equals the log of how often they co-occur. This is a weighted least-squares fit (compare [[Linear Regression]]).
- **Why log:** raw counts range from 1 to millions. The logarithm squeezes that range so that a few very common pairs do not drown out the rest.
- **The weighting:** small counts are noisy, so they are down-weighted. Very large counts are capped so they cannot dominate. The paper used a cap near 100 and a power of 0.75 (from memory; the toy below uses a cap of 10 because its counts are tiny).
- **Pros and cons against Word2Vec:** GloVe uses the whole count table, which can be reused to train several sizes of vector. It needs the table in memory, which grows with vocabulary size squared (only non-zero entries are stored). In practice, careful tuning makes the two about equally good (a commonly stated view; from memory).
- **Still static:** one vector per word, whatever the sentence. Modern [[Transformer]] models give each word a different vector in each sentence.
- **Related view:** both methods can be seen as fitting a matrix of word-pair statistics (see the note on factorisation in [[Word2Vec]]).

```python
import numpy as np
text = """the king rules the land . the queen rules the land .
the king loves the queen . the queen loves the king .
the cat eats fish . the dog eats meat . the cat chases the dog .
the dog chases the cat . the man walks the dog . the woman walks the cat .""".split()
vocab = sorted(set(text)); ix = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
X = np.zeros((V, V))
for i in range(len(text)):
    for j in range(max(0, i - 2), min(len(text), i + 3)):
        if j != i: X[ix[text[i]], ix[text[j]]] += 1 / abs(i - j)      # nearer words count more
I, J = np.nonzero(X)
XMAX = 10                                                            # the paper used about 100; counts here are tiny
f = np.minimum((X[I, J] / XMAX) ** 0.75, 1)                          # weight: small counts count less, capped at 1
rng = np.random.default_rng(0)
W, Wt = rng.normal(0, .1, (V, 8)), rng.normal(0, .1, (V, 8)); b, bt = np.zeros(V), np.zeros(V)
lr = 0.05
for epoch in range(500):
    for k in rng.permutation(len(I)):
        i, j = I[k], J[k]
        e = W[i] @ Wt[j] + b[i] + bt[j] - np.log(X[i, j])           # how far the fit is from log(count)
        g = f[k] * e
        W[i], Wt[j] = W[i] - lr * g * Wt[j], Wt[j] - lr * g * W[i]
        b[i] -= lr * g; bt[j] -= lr * g
E = W + Wt                                                           # keep the sum of the two vector sets
def cos(a, c):
    x, y = E[ix[a]], E[ix[c]]; return round(float(x @ y / np.linalg.norm(x) / np.linalg.norm(y)), 3)
print(V, len(I), round(float(X.max()), 2))                           # 16 84 13.0
print(cos("king", "queen"), cos("cat", "dog"), cos("king", "cat"), cos("man", "woman"))   # 0.821 0.963 0.526 0.992
print(round(float(np.mean([(W[i] @ Wt[j] + b[i] + bt[j] - np.log(X[i, j])) ** 2 for i, j in zip(I, J)])), 4))   # 0.0003
```

The table has 84 non-zero entries for 16 words, and the largest count is 13.0. After training, the fit matches the log counts almost perfectly (average squared gap 0.0003). The pairs that appear in the same spots score high: "man/woman" (0.992), "cat/dog" (0.963), "king/queen" (0.821). The unrelated pair "king/cat" scores lower (0.526). With such a tiny text and only 8 numbers per word, the vectors are crude. When I first tried the paper's cap of 100 on this toy, training was too slow to settle (king/queen came out at 0.569, below king/cat at 0.644), because all the weights were tiny. That is why the toy uses a cap of 10.

## 🔴 4. Deep / formal
**The GloVe loss.** Let $X_{ij}$ be how often word $j$ appears near word $i$. The model fits:

$$
J=\sum_{i,j} f(X_{ij})\,\Big(\mathbf w_i^{\top}\tilde{\mathbf w}_j+b_i+\tilde b_j-\log X_{ij}\Big)^2
$$

**Where:**
- $i,j$ run over pairs of words that occur together at least once ($X_{ij}>0$). $X_{ij}$ is the co-occurrence count for the pair.
- $\mathbf w_i$ is the vector of word $i$ as a centre. $\tilde{\mathbf w}_j$ is the vector of word $j$ as a neighbour. $b_i$ and $\tilde b_j$ are one extra number per word, a bias.
- $\mathbf w_i^{\top}\tilde{\mathbf w}_j+b_i+\tilde b_j$ is the model's prediction for the **log** of the count. The squared gap from $\log X_{ij}$ is the error.
- $f(X_{ij})$ is a weight (next equation) that makes each pair count more or less.
- **Matters:** this is just weighted least squares. The biases soak up the fact that some words are simply common everywhere, so the vectors can focus on meaning. Without the log, huge counts would dominate. Without the weight, noisy tiny counts and gigantic counts would both distort the fit.

**Example (tiny numbers):** a pair with count $X_{ij}=20$, and the model's current score (dot product plus biases) is $2.5$.
- Target: $\log 20=2.996$. Error: $2.5-2.996=-0.496$. Squared: $0.246$.
- Weight (see below): $f(20)=0.299$.
- This pair's contribution to $J$: $0.299\times0.246=\mathbf{0.0735}$.
- Meaning: the model's score is a little too low for a pair seen 20 times, and training will raise it. The weight shrinks the contribution because a count of 20 is still below the cap.

**The weight.** Counts below a cap $x_{\max}$ are trusted in proportion, and counts above are all trusted equally:

$$
f(x)=\begin{cases}(x/x_{\max})^{\alpha} & x<x_{\max}\\[2pt] 1 & \text{otherwise}\end{cases}
$$

**Where:**
- $x$ is a co-occurrence count. $x_{\max}$ is the cap (the paper used about 100; from memory).
- $\alpha$ is a power below 1 (the paper used $0.75$; from memory), which makes the weight rise quickly at first and then flatten.
- **Matters:** it does two jobs. Small counts (noisy: maybe by luck) get small weights. Counts above the cap all get weight 1, so extremely frequent pairs (such as "of the") cannot swamp everything else. If the weight were a flat 1, those frequent pairs would dominate.

**Example (tiny numbers):** with $x_{\max}=100$ and $\alpha=0.75$.
- Count 1: $(1/100)^{0.75}=0.032$. Count 20: $(0.2)^{0.75}=0.299$. Count 50: $(0.5)^{0.75}=0.595$.
- Count 300 is above the cap, so $f=\mathbf{1}$.
- Meaning: a pair seen once has about 3% of the weight of a well-established pair, and nothing is allowed to count more than 1.

**Further notes:**
- **Where the loss comes from:** the authors motivated it from ratios of co-occurrence probabilities, which separate meaning better than raw probabilities (stated from memory; the derivation is in the paper and not shown here).
- **Two vector sets:** the final vector is often the sum (as in the toy) or average of the two sets, which reduces noise (a common practice; from memory).
- **Bias:** like all word vectors trained on text, they pick up the biases in that text. See the note in [[Word2Vec]].
- **Training detail:** the original used an adaptive step size (AdaGrad). The toy uses plain steps for simplicity.

**What came before, and how this improved on it:**
- **Before:** two older routes to word vectors: shrink a table of word-pair counts with linear algebra (the idea behind latent semantic analysis, early 1990s; from memory, and see [[Singular Value Decomposition]]), or learn by predicting nearby words one window at a time ([[Word2Vec]], 2013).
- **Its limit:** raw counts span a huge range, so a few very common pairs dominate, and the prediction methods never use the whole count table at once (this is how the GloVe authors framed it; from memory, I did not read the paper's text).
- **What changed:** fit vectors to the **logarithm** of the global co-occurrence counts, with a weight that trusts small counts less and caps big ones (Pennington, Socher and Manning, EMNLP, October 2014, pages 1532 to 1543; title, authors, venue and date checked on the ACL Anthology page, the abstract itself not read).
- **How much better:** in the code below (a 16-word toy text), the counts run from 0.50 to 13.00, a factor of 26, and their logs run from -0.69 to 2.56. Measuring how well related pairs (king/queen, cat/dog, man/woman) stand apart from an unrelated one (king/cat), the gap is 0.130 for rows of raw counts, 0.255 for an 8-number SVD of log(1+count), and 0.399 for the GloVe fit. **Honest limits:** this is a 16-word toy, the SVD of log counts already gets much of the benefit, and GloVe's own king/queen similarity (0.821) is lower than the others', so I make no claim about real-text results; the published comparisons were not read.

**Check it yourself:**

```python
import numpy as np
text = """the king rules the land . the queen rules the land .
the king loves the queen . the queen loves the king .
the cat eats fish . the dog eats meat . the cat chases the dog .
the dog chases the cat . the man walks the dog . the woman walks the cat .""".split()
vocab = sorted(set(text)); ix = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
X = np.zeros((V, V))
for i in range(len(text)):
    for j in range(max(0, i - 2), min(len(text), i + 3)):
        if j != i: X[ix[text[i]], ix[text[j]]] += 1 / abs(i - j)
cos = lambda M, a, c: float(M[ix[a]] @ M[ix[c]] / np.linalg.norm(M[ix[a]]) / np.linalg.norm(M[ix[c]]))
pairs = [("king", "queen"), ("cat", "dog"), ("man", "woman"), ("king", "cat")]
nz = X[X > 0]
print(f"{V} words, {len(nz)} non-zero counts out of {V * V}; counts run from {nz.min():.2f} to {nz.max():.2f} (a factor of {nz.max() / nz.min():.0f}); their logs run from {np.log(nz.min()):.2f} to {np.log(nz.max()):.2f}")
U, s, Vt = np.linalg.svd(np.log1p(X)); S = U[:, :8] * s[:8]
def show(name, M):
    c = [cos(M, a, b) for a, b in pairs]
    print(f"{name:34s}", "  ".join(f"{a}/{b} {v:.3f}" for (a, b), v in zip(pairs, c)), f"  gap {np.mean(c[:3]) - c[3]:.3f}")
show("rows of raw counts", X); show("8 numbers by SVD of log(1+count)", S)
I, J = np.nonzero(X); f = np.minimum((X[I, J] / 10) ** 0.75, 1); rng = np.random.default_rng(0)
W, Wt = rng.normal(0, .1, (V, 8)), rng.normal(0, .1, (V, 8)); b, bt = np.zeros(V), np.zeros(V)
for _ in range(500):
    for k in rng.permutation(len(I)):
        i, j = I[k], J[k]; g = f[k] * (W[i] @ Wt[j] + b[i] + bt[j] - np.log(X[i, j]))
        W[i], Wt[j] = W[i] - 0.05 * g * Wt[j], Wt[j] - 0.05 * g * W[i]; b[i] -= 0.05 * g; bt[j] -= 0.05 * g
E = W + Wt
show("GloVe fit (8 numbers, 500 passes)", E)
print("gap = average similarity of the 3 related pairs minus the similarity of king/cat (bigger = better separation)")
print("average squared gap between the GloVe score and log(count):", round(float(np.mean([(W[i] @ Wt[j] + b[i] + bt[j] - np.log(X[i, j])) ** 2 for i, j in zip(I, J)])), 4))
```

## Related
- [[Word2Vec]] · [[Embedding]] · [[Embedding Model]] · [[Vector]] · [[Cosine Similarity]] · [[Linear Regression]] · [[Stochastic Gradient Descent]] · [[Negative Sampling]] · [[Self-Supervised Learning]] · [[Transformer]] · [[Singular Value Decomposition]]

## Sources
- Pennington, Socher & Manning, *GloVe: Global Vectors for Word Representation*, EMNLP 2014, pages 1532 to 1543, https://aclanthology.org/D14-1162/ (title, authors, venue and date checked; abstract and body not read)
- Levy & Goldberg, *Neural Word Embedding as Implicit Matrix Factorization*, 2014 (from memory)
- Zhang, Lipton, Li & Smola, *Dive into Deep Learning* (free online), the chapter on word embedding with global vectors (GloVe)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/transformers-nlp GloVe:::Count how often every pair of words shows up close together in a huge text, then give each word a vector whose dot product explains those counts.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write the GloVe loss and explain each symbol.
?
$$J=\sum_{i,j} f(X_{ij})\,\Big(\mathbf w_i^{\top}\tilde{\mathbf w}_j+b_i+\tilde b_j-\log X_{ij}\Big)^2$$
- $i,j$: run over word pairs that occur together at least once ($X_{ij}>0$). $X_{ij}$: the co-occurrence count.
- $\mathbf w_i$: the vector of word $i$ as a centre. $\tilde{\mathbf w}_j$: the vector of word $j$ as a neighbour. $b_i$, $\tilde b_j$: one bias per word.
- $\mathbf w_i^{\top}\tilde{\mathbf w}_j+b_i+\tilde b_j$: the model's prediction for the log of the count. The squared gap from $\log X_{ij}$ is the error.
- $f(X_{ij})$: a weight that makes each pair count more or less.

#flashcards/ml/example #flashcards/topic/transformers-nlp A pair has count 20, the model's score is 2.5, and $f(20)=0.299$. What does it add to $J$?::Target $\log20=2.996$. Error $2.5-2.996=-0.496$, squared $0.246$. Contribution $0.299\times0.246=0.0735$. The score is a little too low, so training raises it.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write the GloVe weighting function and explain each symbol.
?
$$f(x)=\begin{cases}(x/x_{\max})^{\alpha} & x<x_{\max}\\[2pt] 1 & \text{otherwise}\end{cases}$$
- $x$: a co-occurrence count. $x_{\max}$: the cap (the paper used about 100, from memory).
- $\alpha$: a power below 1 (the paper used 0.75, from memory), so the weight rises quickly at first and then flattens.
- Small counts (noisy, maybe luck) get small weights. Counts above the cap all get weight 1, so very frequent pairs such as "of the" cannot swamp everything else.

#flashcards/ml/example #flashcards/topic/transformers-nlp $x_{\max}=100$, $\alpha=0.75$. What is $f$ for counts 1, 20, 50 and 300?::$f(1)=0.032$, $f(20)=0.299$, $f(50)=0.595$, $f(300)=1$. A pair seen once has about 3% of the weight of a well-established pair.
