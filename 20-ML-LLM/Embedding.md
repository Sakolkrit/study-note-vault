---
tags: [embedding, LLM, transformer, NLP, vector, RAG]
status: growing
level: 4
tier: 1
---

# Embedding

## 🟢 1. Simple (age 12)
An embedding turns a word (or sentence, or picture) into a **list of numbers** that works like an **address on a map**. Things with similar meaning get addresses close together.

**Example:** On a map of meaning, "cat" and "kitten" live on the same street. "Car" lives across town. A computer can't understand the word "cat", but it can measure that two addresses are close.

## 🔵 2. How it works
- Computers only do math, so words must become numbers first.
- Each word gets a list of numbers, called a [[Vector]]. For example `cat = [0.8, 0.1, 0.6, ...]`.
- Similar words get similar lists. "Close" is measured with a score, usually [[Cosine Similarity]].
- The computer **learns** these numbers by reading lots of text. Words used in similar sentences end up with similar numbers ("The ___ drank milk" fits both cat and kitten).
- Fun result: the numbers capture relationships. `king − man + woman ≈ queen`.

![[embedding-illustration.svg]]
*Each word is a point; the same arrow direction links each country to its capital. Image: Fschwarzentruber, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Word_embedding_illustration.svg), CC BY-SA 4.0.*

Why it matters: search by *meaning* instead of exact words, and it is how [[RAG]] finds the right text.

## 🟠 3. Technical detail
An embedding maps a discrete token (or whole text) to a dense vector in $\mathbb{R}^d$, typically $d$ = 384 to 3072 for text models.

Kinds:
- **Static word embeddings** (word2vec, GloVe): one vector per word, same in every sentence. "bank" (river) = "bank" (money).
- **Contextual embeddings** ([[Transformer]] models like BERT): the vector depends on the sentence, so the two "bank"s differ.
- **Sentence/document embeddings**: one vector for a whole passage. Used for search and [[RAG]].

In an [[LLM]], the first layer is a lookup table (the embedding matrix) that turns token IDs into vectors; it is trained along with the rest of the model.

```python
import numpy as np
# toy 3-d vectors, real ones have hundreds of dimensions
v = {"cat": np.array([.9, .1, .0]), "kitten": np.array([.8, .2, .1]), "car": np.array([.0, .1, .9])}
cos = lambda a, b: a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
print(cos(v["cat"], v["kitten"]), cos(v["cat"], v["car"]))  # ~0.98 vs ~0.01
```

Store and search millions of vectors with a [[Vector Database]] (approximate nearest neighbour, e.g. HNSW).

## 🔴 4. Deep / formal
The embedding lookup:

$$
e_i = E^\top x_i,\qquad E\in\mathbb{R}^{|V|\times d}
$$

**Where:**
- $E$ is the **embedding matrix**, a big table of learned numbers. Row $i$ is the vector for token $i$. **Matters:** these numbers *are* the learned meaning; training adjusts them.
- $|V|$ is the vocabulary size (how many different tokens exist, see [[Tokenization]]). It is the number of rows.
- $d$ is the embedding dimension (how many numbers per token, e.g. 768). It is the number of columns. **Matters:** bigger $d$ can store more detail but costs more memory and compute.
- $\mathbb{R}$ means real numbers, so $\mathbb{R}^{|V|\times d}$ says "a grid of $|V|$ rows and $d$ columns of ordinary decimals".
- $x_i$ is the **one-hot vector** for token $i$: all zeros with a single 1 in position $i$. **Matters:** multiplying by it simply picks out one row, so the "matrix multiplication" is really just a table lookup.
- $e_i$ is the resulting embedding vector for token $i$, which flows into the rest of the model. $^\top$ is the transpose (swap rows and columns) so the shapes line up.

**Example (tiny numbers):** vocabulary of 3 tokens, $d=2$ numbers each. The table $E$ has rows: `cat` $=[0.9,\,0.1]$, `dog` $=[0.8,\,0.2]$, `car` $=[0.0,\,0.9]$.
- Token `dog` is number 2, so $x=[0,1,0]$.
- $E^\top x=0\cdot[0.9,0.1]+1\cdot[0.8,0.2]+0\cdot[0.0,0.9]=\mathbf{[0.8,\,0.2]}$.
- The zeros erase the other rows. The "multiplication" just picked out the row for `dog`.

Similarity between vectors $u,v$:

$$
\cos(u,v)=\frac{u\cdot v}{\lVert u\rVert\,\lVert v\rVert}
$$

**Where:**
- $u$ and $v$ are the two embeddings being compared (for example "cat" and "kitten").
- $u\cdot v$ is the dot product: multiply matching positions, then add. **Matters:** large and positive when both vectors push in the same directions.
- $\lVert u\rVert$ and $\lVert v\rVert$ are lengths. **Matters:** dividing by them throws away size, so only the *direction* (meaning) is compared.
- The result is between -1 and 1: near 1 means similar meaning, near 0 unrelated, near -1 opposite. Details in [[Cosine Similarity]].

**Example (tiny numbers):** use the rows above: `cat` $=[0.9,0.1]$, `dog` $=[0.8,0.2]$, `car` $=[0.0,0.9]$.
- `cat` vs `dog`: dot $=0.72+0.02=0.74$. Lengths $\approx0.906$ and $\approx0.825$. Score $\approx\mathbf{0.99}$ (very similar).
- `cat` vs `car`: dot $=0+0.09=0.09$. Lengths $\approx0.906$ and $0.9$. Score $\approx\mathbf{0.11}$ (not similar).
- The computer "knows" `cat` is closer to `dog` than to `car` using only numbers.

**word2vec skip-gram** maximizes the probability of context words $c$ given centre word $w$:

$$
p(c\mid w)=\frac{\exp(u_c^\top v_w)}{\sum_{c'\in V}\exp(u_{c'}^\top v_w)}
$$

**Where:**
- $w$ is the **centre word** and $c$ is a **context word** that appears near it. $p(c\mid w)$ is "how likely is $c$ to show up near $w$?". Training pushes this up for words that really do appear together.
- $v_w$ is the vector of the centre word and $u_c$ is the vector of the context word (each word has two vectors, one per role). **Matters:** these are the numbers being learned; the final embedding is usually $v_w$.
- $u_c^\top v_w$ is a dot product, a score for "how well do $c$ and $w$ fit together?". **Matters:** it is the only place the two vectors meet, so training makes vectors of co-occurring words align.
- $\exp(\cdot)$ is $e^{x}$. **Matters:** it makes every score positive and magnifies differences, so a higher score gets a much higher share.
- The denominator $\sum_{c'\in V}\exp(u_{c'}^\top v_w)$ adds the same quantity over every word $c'$ in the vocabulary $V$. **Matters:** dividing by it makes all probabilities add up to exactly 1 (this is the *softmax*). It is also the expensive part, which is why negative sampling exists.

**Example (tiny numbers):** centre word $w=$ "cat". Only 3 words in the vocabulary. The dot-product scores $u_c^\top v_w$ are: `milk` $=2.0$, `bone` $=0.5$, `car` $=-1.0$.
- $\exp$ of each: $7.39,\ 1.65,\ 0.37$. Their sum (the denominator) is $9.41$.
- Probabilities: `milk` $=7.39/9.41=\mathbf{0.79}$, `bone` $=\mathbf{0.18}$, `car` $=\mathbf{0.04}$. They add to 1.
- If the real text says "cat drinks milk", training nudges the vectors so the `milk` score rises and its probability goes toward 1.

The full softmax is expensive, so negative sampling replaces it with a binary objective. Skip-gram with negative sampling implicitly factorizes a shifted PMI matrix (Levy & Goldberg, 2014).

**Contrastive training** (modern sentence embeddings): pull matching pairs together, push others apart (InfoNCE, temperature $\tau$):

$$
\mathcal{L}=-\log\frac{\exp(\mathrm{sim}(q,d^+)/\tau)}{\sum_{j}\exp(\mathrm{sim}(q,d_j)/\tau)}
$$

**Where:**
- $\mathcal{L}$ is the **loss**: one number saying how wrong the model currently is. Training tries to make it small ([[Gradient Descent]]).
- $q$ is a query (for example a question) and $d^+$ is its true matching document (the *positive*). $d_j$ runs over all candidates in the batch: the positive plus many wrong ones (*negatives*).
- $\mathrm{sim}(\cdot,\cdot)$ is the similarity, usually [[Cosine Similarity]]. $\mathrm{sim}(q,d^+)$ should be high; $\mathrm{sim}(q,d_j)$ for wrong ones should be low.
- $\tau$ (tau) is the **temperature**, a small positive number. **Matters:** dividing by a small $\tau$ sharpens the contest, punishing near-misses harder. Too small makes training unstable; too large makes it lazy.
- The fraction is the chance the model picks the right document out of the lineup, i.e. a softmax over all candidates. The numerator is the right one; the denominator is everyone.
- $-\log$ turns "chance of being right" into a loss: when the chance is near 1, the loss is near 0; when the chance is tiny, the loss is huge. **Matters:** it pulls matching pairs together and pushes the others apart in the same step.

**Example (tiny numbers):** a question has 3 candidate documents with cosine scores: correct $d^+=0.9$, wrong ones $0.2$ and $0.1$.
- With $\tau=0.1$: scores become $9,\ 2,\ 1$. $\exp$: $8103,\ 7.4,\ 2.7$. Chance of picking the right one $=0.9988$. Loss $=-\log(0.9988)=\mathbf{0.001}$ (almost nothing to fix).
- With $\tau=1$: scores stay $0.9,\ 0.2,\ 0.1$. $\exp$: $2.46,\ 1.22,\ 1.11$. Chance $=0.514$. Loss $=\mathbf{0.67}$ (still a lot to learn).
- So a small $\tau$ makes the model sure of itself fast, and a big $\tau$ keeps the pressure on.

Caveats: the `king − man + woman` analogy is cherry-picked and partly an artefact of excluding the input words; embeddings absorb biases from training text; high-dimensional cosine similarities concentrate, so absolute scores are not comparable across models.

**What came before, and how this improved on it:**
- **Before:** a word is a **one-hot** vector, as long as the vocabulary, with a single 1.
- **Its limit:** every pair of different words is exactly as far apart as any other, so "cat" is no closer to "dog" than to "king", and the vectors are huge.
- **What changed:** learn short dense vectors where words used in similar places end up close ([[Word2Vec]], 2013, and earlier methods based on word-count tables and [[Singular Value Decomposition]]).
- **How much better:** in the code below (an 18-word toy corpus, vectors from an SVD of word-pair counts), one-hot cosines are 0.0 for both pairs. The 4-number vectors give cat-dog 0.999 and cat-king 0.792, and are 4 numbers long instead of 18. Honest note: with 3 numbers cat-king was 0.979, so I chose 4 after trying a few sizes, and a corpus this small proves little.

**Check it yourself:**

```python
import numpy as np
text = ("the cat drinks milk . the dog drinks water . the cat chases the mouse . "
        "the dog chases the cat . a cat sleeps on the sofa . a dog sleeps on the rug . "
        "the king rules the land . the queen rules the land .").split()
vocab = sorted(set(text)); ix = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
C = np.zeros((V, V))
for i, w in enumerate(text):
    for j in (i - 2, i - 1, i + 1, i + 2):
        if 0 <= j < len(text): C[ix[w], ix[text[j]]] += 1
U, S, _ = np.linalg.svd(np.log1p(C)); E = U[:, :4] * S[:4]
cos = lambda a, b: a @ b / np.linalg.norm(a) / np.linalg.norm(b)
one = np.eye(V)
for a, b in (("cat", "dog"), ("cat", "king")):
    print(a, b, "one-hot cosine", cos(one[ix[a]], one[ix[b]]), " 4-number vector cosine", round(cos(E[ix[a]], E[ix[b]]), 3))
print("vocabulary", V, "-> one-hot length", V, "vs dense length 4")
```

## Related
- [[Vector]] · [[Cosine Similarity]] · [[Vector Database]] · [[Transformer]] · [[LLM]] · [[RAG]] · [[Word2Vec]] · [[Clustering]] · [[GloVe]] · [[Negative Sampling]] · [[Singular Value Decomposition]] · [[Byte-Pair Encoding]] · [[Autoencoder]]
- [[Matrix Factorization]] · [[Recommender Systems]] · [[Collaborative Filtering]]
- [[Two-Tower Model]]
## Sources
- Mikolov et al., *Efficient Estimation of Word Representations in Vector Space*, 2013
- Pennington et al., *GloVe*, 2014
- Levy & Goldberg, *Neural Word Embedding as Implicit Matrix Factorization*, 2014

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/transformers-nlp Embedding:::A list of numbers that works like an address on a map of meaning. Things with similar meaning get addresses close together.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write the embedding lookup and explain each symbol.
?
$$e_i = E^\top x_i,\qquad E\in\mathbb{R}^{|V|\times d}$$
- $E$: the embedding matrix, a big table of learned numbers. Row $i$ is the vector for token $i$. Training adjusts these numbers, and they are the learned meaning.
- $|V|$: the vocabulary size (the number of rows). $d$: the embedding dimension (numbers per token, e.g. 768).
- $x_i$: a one-hot vector for token $i$ (a 1 at position $i$, 0 elsewhere). The zeros erase every other row, so the multiplication just picks out row $i$.
- $e_i$: the resulting embedding.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write cosine similarity and explain each symbol.
?
$$\cos(u,v)=\frac{u\cdot v}{\lVert u\rVert\,\lVert v\rVert}$$
- $u$, $v$: the two embeddings being compared (for example "cat" and "kitten").
- $u\cdot v$: the dot product, large and positive when both push in the same directions.
- $\lVert u\rVert$, $\lVert v\rVert$: the lengths. Dividing by them throws away size, so only the direction (meaning) is compared.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write the skip-gram word probability and explain each symbol.
?
$$p(c\mid w)=\frac{\exp(u_c^\top v_w)}{\sum_{c'\in V}\exp(u_{c'}^\top v_w)}$$
- $w$: the centre word. $c$: a context word that appears near it. $p(c\mid w)$: how likely $c$ is to show up near $w$. Training pushes this up for words that really appear together.
- $v_w$: the vector of the centre word. $u_c$: the vector of the context word (each word has two vectors, one per role).
- The dot product is the score. The sum over all words $c'$ in the vocabulary is the softmax divider.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write the contrastive loss for training an embedding model and explain each symbol.
?
$$\mathcal{L}=-\log\frac{\exp(\mathrm{sim}(q,d^+)/\tau)}{\sum_{j}\exp(\mathrm{sim}(q,d_j)/\tau)}$$
- $\mathcal{L}$: the loss, how wrong the model currently is.
- $q$: a query. $d^+$: its true matching document (the positive). $d_j$: all candidates in the batch, the positive plus many wrong ones (negatives).
- $\mathrm{sim}$: similarity, usually cosine. $\tau$: the temperature. Small $\tau$ stretches the scores and punishes near-misses harder.

#flashcards/ml/cloze #flashcards/topic/transformers-nlp An embedding turns a word into a list of ==numbers== placed so that similar meanings are ==close== together.

#flashcards/ml/example #flashcards/topic/transformers-nlp With rows `cat` = [0.9, 0.1], `dog` = [0.8, 0.2], `car` = [0.0, 0.9], what is the embedding of `dog` using $E^\top x$?::dog is token 2, so x = [0, 1, 0]. E⊤x = [0.8, 0.2]. The zeros erase the other rows, so it just picks out row 2.

#flashcards/ml/concept #flashcards/topic/transformers-nlp What is the cosine of `cat` vs `dog`, and `cat` vs `car`?::cat vs dog: dot 0.74, lengths 0.906 and 0.825, score ≈ 0.99 (very similar). cat vs car: dot 0.09, score ≈ 0.11 (not similar).

#flashcards/ml/example #flashcards/topic/transformers-nlp For centre word "cat", scores are milk 2.0, bone 0.5, car −1.0. What are the probabilities?::exp: 7.39, 1.65, 0.37, sum 9.41. milk 0.79, bone 0.18, car 0.04.

#flashcards/ml/image #flashcards/topic/transformers-nlp ![[embedding-illustration.svg]]
What does this picture show?
?
Each word is a point. The same arrow direction links each country to its capital.
