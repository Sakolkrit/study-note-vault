---
tags: [embedding, word2vec, contrastive-learning, LLM, NLP, vector, RAG]
status: growing
level: 4
tier: 3
---

# Embedding Model

## 🟢 1. Simple (age 12)
An embedding model is the **machine that makes the "address on the map of meaning"** for a word or sentence (see [[Embedding]]). It learns how by playing a guessing game over and over on a huge pile of text, and the numbers it ends up with are the addresses.

**Example:** Imagine learning what "loves" means without a dictionary, only by seeing which words hang around it: "the man **loves** his son", "the girl **loves** her dog". Words that appear in the same kinds of places get similar meanings. The model keeps playing "which words are neighbours of this one?" and adjusts its numbers until neighbours end up close together on the map.

## 🔵 2. How it works
1. Collect a lot of text.
2. Make a **guessing game** out of it. Two popular games:
   - **Neighbour guessing:** pick a word and guess the words around it ("loves" should predict "the", "man", "his", "son").
   - **Match guessing:** given a question, pick the matching answer out of a lineup of wrong ones.
3. Give every word (or sentence) a starting list of random numbers.
4. Play: the model makes a guess, a [[Loss Function]] says how wrong it was, and [[Gradient Descent]] nudges the numbers a little.
5. After millions of rounds, words that fit similar situations end up with similar number lists.
6. Throw away the game. **Keep the numbers.** Now any text can be turned into a [[Vector]] and compared with [[Cosine Similarity]]. This is the engine behind search by meaning, [[RAG]] and the [[Vector Database]].

![[skip-gram.svg]]
*The neighbour-guessing game (called skip-gram): the centre word "loves" is used to predict the words around it, "the", "man", "his" and "son". Training pushes "loves" to sit near words that appear in such contexts. Image: Zhang, Lipton, Li and Smola (from the book "Dive into Deep Learning"), [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Skip-gram.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Static word embeddings (word2vec, GloVe):** one vector per word, learned from neighbour-prediction or co-occurrence counts. Fast and small, but "bank" gets one vector for both river and money.
- **Contextual embeddings:** a [[Transformer]] reads the whole sentence, so the same word gets a different vector in different sentences.
- **Sentence and text embedding models:** a Transformer trained so that a whole sentence or passage becomes one vector. The usual recipe:
  1. Start from a pre-trained language model ([[LLM]] family, usually an encoder-style one).
  2. Pool the output vectors (average them, or take a special start token's vector).
  3. Fine-tune ([[Fine-tuning]]) with **contrastive learning**: pull matching pairs (a question and its answer, two paraphrases) together and push non-matching pairs apart.
- **Negatives:** the "wrong answers" are often just the other examples in the same batch (**in-batch negatives**), which is cheap. **Hard negatives** (wrong answers that look right) teach more.
- **Training data:** pairs from question-and-answer sites, titles and bodies of articles, translations, search logs, and so on.
- **Choosing a model:** check the vector size (smaller means cheaper storage), the maximum text length, the languages, and a benchmark on *your* kind of data. Always use the **same** model for stored text and for queries.
- **Normalising:** vectors are often scaled to length 1 so [[Cosine Similarity]] equals a plain dot product.
- **Chunking:** long documents are split into chunks before embedding, because one vector can only hold so much meaning.

```python
import numpy as np
def info_nce(sims, tau=0.5):                 # sims[0] is the true (positive) pair
    e = np.exp(np.array(sims) / tau)
    return round(float(-np.log(e[0] / e.sum())), 4)
print(info_nce([0.8, 0.2, 0.1]))             # 0.4368  positive is the closest: small loss
print(info_nce([0.1, 0.8, 0.2]))             # 1.8368  a wrong item is closest: big loss
print(info_nce([0.8, 0.2, 0.1], tau=0.1))    # 0.0034  sharper temperature, easier win

s = np.array([2., 1., 0.])                   # skip-gram scores for 3 candidate words
p = np.exp(s) / np.exp(s).sum()
print(p.round(3), round(float(-np.log(p[0])), 3))   # [0.665 0.245 0.09 ] 0.408
```

## 🔴 4. Deep / formal
**Skip-gram objective.** Each word has two vectors: $\mathbf{v}_c$ when it is the centre word and $\mathbf{u}_o$ when it is a neighbour. The chance of seeing neighbour $o$ around centre $c$ is a [[Softmax]] of dot products:

$$
P(o\mid c)=\frac{\exp\!\big(\mathbf{u}_o^{\top}\mathbf{v}_c\big)}{\sum_{w\in V}\exp\!\big(\mathbf{u}_w^{\top}\mathbf{v}_c\big)}
$$

**Where:**
- $c$ is the **centre** word (such as "loves") and $o$ is an **outside** word seen near it (such as "son"). $V$ is the whole vocabulary: every word the model knows.
- $\mathbf{v}_c$ is the centre word's vector, and $\mathbf{u}_o$ is the outside word's vector. $\mathbf{u}_o^{\top}\mathbf{v}_c$ is their **dot product**: one number, large when the two vectors point the same way (see [[Vector]]).
- $\exp(\cdot)$ turns each score into a positive number, and dividing by the sum $\sum_{w\in V}$ makes all chances add to 1. **Matters:** this is a softmax over every word in the vocabulary, so training pushes the true neighbour's score up *relative to all others*.
- **Matters (cost):** the sum runs over the entire vocabulary (tens of thousands of words) for every training example, which is slow. This is why practical systems use shortcuts (negative sampling) or the contrastive loss below.
- Training minimises $-\log P(o\mid c)$ for real neighbour pairs. After training the vectors $\mathbf{v}$ become the embeddings.

**Example (tiny numbers):** centre word "loves". Three candidate neighbours have dot-product scores $(2,\,1,\,0)$ with it, and the true neighbour is the first one.
- $\exp$ of the scores: $7.389,\ 2.718,\ 1$. Sum $=11.107$.
- $P(\text{true}\mid c)=7.389/11.107=\mathbf{0.665}$. Loss $=-\ln0.665=\mathbf{0.408}$.
- If the true neighbour had the lowest score (0) instead, its chance would be $1/11.107=0.090$ and the loss $\mathbf{2.408}$, so training would pull its vector toward "loves".

**Contrastive loss (InfoNCE).** Modern text embedding models are trained with a lineup of one right answer and several wrong ones:

$$
\mathcal{L}=-\log\frac{\exp\!\big(\operatorname{sim}(q,d^{+})/\tau\big)}{\sum_{j=0}^{K}\exp\!\big(\operatorname{sim}(q,d_j)/\tau\big)}
$$

**Where:**
- $q$ is the query (for example a question) and $d^{+}$ is its true match (the right answer passage). $d_0=d^{+}$ and $d_1,\dots,d_K$ are the $K$ wrong ones (negatives), so the sum covers all $K+1$ items in the lineup.
- $\operatorname{sim}(q,d)$ is the similarity of their embeddings, usually [[Cosine Similarity]], between $-1$ and $1$.
- $\tau$ (tau) is the **temperature**, a small positive number. **Matters:** a small $\tau$ stretches the gaps so near-misses are punished hard; a large $\tau$ flattens them (the same temperature idea as in [[Softmax]]).
- The fraction is the chance the model picks the right answer from the lineup, and $-\log$ turns it into a loss (see [[Loss Function]]). **Matters:** it is the same cross-entropy as ordinary classification, where the "classes" are the items in the lineup. Pulling $q$ toward $d^{+}$ and away from the negatives is exactly what shapes the map of meaning.

**Example (tiny numbers):** $\tau=0.5$. Cosine similarities of the query with three items: the true answer $0.8$, and two wrong ones $0.2$ and $0.1$.
- Divide by $\tau$: $1.6,\ 0.4,\ 0.2$. $\exp$: $4.953,\ 1.492,\ 1.221$. Sum $=7.666$.
- Chance of the right one: $4.953/7.666=\mathbf{0.646}$. Loss $=-\ln0.646=\mathbf{0.437}$.
- If a wrong item were the closest instead (similarities $0.1$, $0.8$, $0.2$): the right one gets $1.221/7.666=0.159$, and the loss is $\mathbf{1.837}$, four times bigger. Training pushes the right pair closer.
- With a sharper $\tau=0.1$ the same good lineup gives a loss of only $\mathbf{0.0034}$.

**Further notes:**
- **Dimension and cost:** each stored vector costs $d$ numbers. A 768-number vector in 4-byte floats is about 3 KB, so a million chunks is roughly 3 GB before any index ([[Vector Database]]). Smaller or compressed vectors trade some accuracy for space.
- **Matryoshka-style training** makes the first few numbers of a vector usable on their own, so you can cut the length later. (From memory; check specific models.)
- **Limits:** embeddings blur negation and fine detail ("the drug is safe" and "the drug is not safe" can be close), reflect biases in the training text, and degrade on topics far from their training data.
- **Evaluation:** test on your own questions, checking whether the right passage is in the top few results (recall at $k$). A public leaderboard score is only a rough guide.
- **Relation to LLMs:** a generative [[LLM]] and an embedding model often share the Transformer design but are trained for different goals: the next word versus a good vector for similarity.

**What came before, and how this improved on it:**
- **Before:** to compare two sentences with BERT, feed both into the network together (a cross-encoder).
- **Its limit:** with $n$ sentences you must read $n(n-1)/2$ pairs. The Sentence-BERT abstract says that finding the most similar pair among 10,000 sentences takes about 50 million inferences (about 65 hours) with BERT.
- **What changed:** encode each sentence once into a vector and compare vectors with [[Cosine Similarity]] (Reimers and Gurevych, arXiv 1908.10084, 27 August 2019, abstract read).
- **How much better:** the abstract reports about 5 seconds instead of 65 hours while keeping BERT's accuracy. My count in the code below: 49,995,000 pairs against 10,000 encodings, 5,000 times fewer network passes. I did not time anything.

**Check it yourself:**

```python
n = 10_000
pairs = n * (n - 1) // 2
print("pairs a cross-encoder must read:", f"{pairs:,}", " sentences an embedding model must encode:", f"{n:,}", " ratio", round(pairs / n))
```

## Related
- [[Embedding]] · [[Cosine Similarity]] · [[Vector]] · [[Vector Database]] · [[RAG]] · [[Transformer]] · [[LLM]] · [[Fine-tuning]] · [[Softmax]] · [[Loss Function]] · [[Gradient Descent]] · [[Tokenization]] · [[Word2Vec]] · [[GloVe]] · [[Byte-Pair Encoding]] · [[BERT]] · [[BM25]] · [[Vault Search with RAG]] · [[CLIP]] · [[Contrastive Learning]] · [[Chunking Strategies]] · [[Reranking]]

## Sources
- Mikolov et al., *Efficient Estimation of Word Representations in Vector Space* (word2vec), 2013
- Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality*, 2013 (negative sampling)
- van den Oord et al., *Representation Learning with Contrastive Predictive Coding* (InfoNCE), 2018 (from memory)
- Reimers & Gurevych, *Sentence-BERT*, 2019 (from memory)
- Zhang, Lipton, Li & Smola, *Dive into Deep Learning* (free online), chapter on word embedding
- Reimers and Gurevych, arXiv 1908.10084 (abstract read for the history block)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/agents-rag Embedding model:::The machine that makes the address on the map of meaning for a word or sentence. It learns by playing a guessing game over a huge pile of text, and the numbers it ends up with are the addresses.

#flashcards/ml/formula #flashcards/topic/agents-rag Write the skip-gram probability and explain each symbol.
?
$$P(o\mid c)=\frac{\exp\!\big(\mathbf{u}_o^{\top}\mathbf{v}_c\big)}{\sum_{w\in V}\exp\!\big(\mathbf{u}_w^{\top}\mathbf{v}_c\big)}$$
- $c$: the centre word (such as "loves"). $o$: an outside word seen near it (such as "son"). $V$: the whole vocabulary.
- $\mathbf{v}_c$: the centre word's vector. $\mathbf{u}_o$: the outside word's vector. $\mathbf{u}_o^\top\mathbf{v}_c$: their dot product, a score for how well they go together.
- The softmax (exp over the sum) turns the scores into chances that add up to 1. Training pushes this up for words that really appear together.

#flashcards/ml/formula #flashcards/topic/agents-rag Write the contrastive loss for training a retrieval embedding and explain each symbol.
?
$$\mathcal{L}=-\log\frac{\exp\!\big(\operatorname{sim}(q,d^{+})/\tau\big)}{\sum_{j=0}^{K}\exp\!\big(\operatorname{sim}(q,d_j)/\tau\big)}$$
- $q$: the query (for example a question). $d^{+}$: its true match (the right answer passage).
- $d_0=d^+$ and $d_1,\dots,d_K$: the $K$ wrong ones (negatives), so the sum covers all $K+1$ items in the lineup.
- $\operatorname{sim}(q,d)$: the similarity of their embeddings, usually cosine. $\tau$: the temperature, which stretches the scores so near-misses are punished harder.

#flashcards/ml/cloze #flashcards/topic/agents-rag The embedding model learns to place words that appear in similar ==contexts== close together on the ==map==.

#flashcards/ml/example #flashcards/topic/agents-rag Centre word "loves" with dot-product scores (2, 1, 0) and the true neighbour first. What are the chance and loss?::exp: 7.389, 2.718, 1. Sum 11.107. P = 7.389/11.107 = 0.665. Loss = −ln 0.665 = 0.408.

#flashcards/ml/example #flashcards/topic/agents-rag $\tau=0.5$, cosine similarities 0.8 (true), 0.2 and 0.1. What is the chance of the right one and the loss?::Divide by τ: 1.6, 0.4, 0.2. exp: 4.953, 1.492, 1.221. Sum 7.666. Chance 4.953/7.666 = 0.646. Loss −ln 0.646 = 0.437.

#flashcards/ml/image #flashcards/topic/agents-rag ![[skip-gram.svg]]
What does this picture show?
?
The neighbour-guessing game (skip-gram): the centre word "loves" is used to predict the words around it, "the", "man", "his" and "son". Training pushes "loves" to sit near words that appear in such contexts.
