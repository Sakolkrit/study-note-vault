---
tags: [RAG, LLM, transformer, retrieval]
status: growing
level: 4
tier: 3
---

# RAG

## 🟢 1. Simple (age 12)
RAG is like an **open-book test** for an AI. Before answering, the AI looks up the right pages in a book, then writes its answer using those pages.

**Example:** You ask a chatbot, "What is our school's lunch menu today?" It never learned that. So it first searches the school website, finds today's menu, and then answers you.

## 🔵 2. How it works
1. You ask a question.
2. The computer **searches** a pile of documents for the pieces most like your question (see [[Embedding]] and [[Vector Database]]).
3. It pastes those pieces next to your question.
4. The [[LLM]] reads both and writes the answer.

Why bother? An LLM only knows what it saw in training. RAG lets it use new or private facts without retraining it.

## 🟠 3. Technical detail
Pipeline:
- **Index time:** split documents into chunks → embed each chunk → store vectors in a vector DB.
- **Query time:** embed query → top-k nearest chunks (cosine similarity) → build prompt `[context] + [question]` → LLM generates.

```python
def rag(q, k=4):
    chunks = vector_db.search(embed(q), k)
    prompt = "Context:\n" + "\n".join(chunks) + f"\n\nQuestion: {q}"
    return llm(prompt)
```

Common upgrades: hybrid search (BM25 + vectors), a reranker ([[Reranking]]), better chunking ([[Chunking Strategies]]), query rewriting. In my own test the reranker helped weak keyword search but not the blended search ([[RAG Tuning Lab]]).

**When RAG is the wrong tool (limits compared with fine-tuning).** RAG gives the model facts to read. It does not change the model. So it has limits that [[Fine-tuning]] does not:
- **Extra delay.** Every question first waits for a search, and then the model reads a longer prompt. If you paste 4 chunks of 170 words, the model reads about 680 extra words on every query (4 × 170). A fine-tuned model reads none. For fast, high-volume answers this adds up ([[LLM Inference]]).
- **The answer is only as good as the search.** If the right chunk is not retrieved, the model answers without it, or from the wrong text. In this vault's own test ([[Interview Question Index]], 115 questions, one run), the right note was in the top 3 results for 105 questions and missing from it for 10. That is a rough, hand-labelled test, but it shows misses are normal, not rare.
- **It does not teach a skill, a style or a format.** Retrieved text can be copied or summarised, but a model that has never learned to write in your house style or reason in your field's way does not learn it from a pasted page. Changing behaviour is what fine-tuning is for.
- **Long, noisy prompts hurt.** Too many chunks can bury the useful one ([[Context Rot]]).
- **More parts to run.** A vector store, a chunking step and an index to keep fresh ([[Vector Database]]).

The trade is real in the other direction too: RAG lets you update facts by editing a document, with no retraining, and shows where an answer came from (the RAG paper's abstract names both provenance and updating knowledge as open problems it targets). So the common advice is: use RAG for facts that change, fine-tuning for behaviour, and both when you need both. That advice is the usual rule of thumb, not something this note measured.

## 🔴 4. Deep / formal
Retrieval score, with query $q$ and chunk $d$:

$$
\text{score}(q,d)=\frac{E(q)\cdot E(d)}{\lVert E(q)\rVert\,\lVert E(d)\rVert}
$$

**Where:**
- $q$ is the user's question (the *query*). $d$ is one chunk of a document being checked. Every stored chunk gets its own score against the same $q$.
- $E(\cdot)$ is the [[Embedding]] model: it turns text into a [[Vector]]. **Matters:** the same model must embed both $q$ and $d$, otherwise their numbers live in different "maps" and cannot be compared.
- $E(q)\cdot E(d)$ is the dot product: multiply matching numbers and add them up. **Matters:** it is large when the two vectors point the same way, so it is the "do these match?" part.
- $\lVert E(q)\rVert$ and $\lVert E(d)\rVert$ are the vector lengths. **Matters:** dividing by them cancels length, so a long chunk does not win just because its numbers are bigger. Only direction (meaning) counts.
- $\text{score}(q,d)$ is the result, from -1 to 1. Higher means closer in meaning. The top-$k$ highest-scoring chunks are the ones retrieved. See [[Cosine Similarity]].

**Example (tiny numbers):** the question vector is $E(q)=[3,4]$. Chunk A (same topic) is $[4,3]$. Chunk B (unrelated topic) is $[-4,3]$.
- Chunk A: dot product $=3\cdot4+4\cdot3=24$. Both lengths are $5$. Score $=24/(5\cdot5)=\mathbf{0.96}$.
- Chunk B: dot product $=3\cdot(-4)+4\cdot3=0$. Score $=\mathbf{0}$.
- Chunk A wins and is handed to the LLM. Chunk B is ignored.

Original formulation (Lewis et al., 2020) marginalizes over retrieved documents $z$:

$$
p(y\mid x)\approx\sum_{z\in \text{top-}k} p_\eta(z\mid x)\,p_\theta(y\mid x,z)
$$

**Where:**
- $x$ is the input question and $y$ is the generated answer. $p(y\mid x)$ reads "the probability of answer $y$ given question $x$". It is what we want the whole system to produce.
- $z$ is one retrieved document (or chunk). **Matters:** it is a hidden helper variable. The model never has to be told which document is "right"; it considers several.
- $\sum_{z\in\text{top-}k}$ adds up the contribution of each of the $k$ best documents. **Matters:** the answer is a blend over several sources, not a bet on a single one. The "$\approx$" is there because the true sum would run over *all* documents, which is too expensive, so only the top $k$ are used.
- $p_\eta(z\mid x)$ is the **retriever**: how likely document $z$ is relevant to question $x$. $\eta$ stands for the retriever's own learned numbers. **Matters:** it is the weight that decides how much each document is trusted.
- $p_\theta(y\mid x,z)$ is the **generator** (the [[LLM]]): how likely answer $y$ is, given the question *and* document $z$. $\theta$ stands for the generator's learned numbers. **Matters:** this is where the retrieved text actually helps the answer.
- Multiplying the two terms, then summing, means: "trust each document as much as the retriever says, and ask the generator what it would answer using that document."

**Example (tiny numbers):** question: "Is the shop open on Sunday?" Only $k=2$ documents are used.
- Doc 1 ("opening hours page"): retriever trust $p_\eta=0.7$. Chance the answer "yes" comes out given this doc: $p_\theta=0.8$.
- Doc 2 ("old news post"): retriever trust $p_\eta=0.3$. Chance of "yes" given this doc: $p_\theta=0.1$.
- Total: $0.7\times0.8+0.3\times0.1=0.56+0.03=\mathbf{0.59}$.
- Reading: the good document pulls the answer toward "yes", the weak document barely changes it.

Failure modes: bad chunking, retrieval misses, context stuffing → "lost in the middle".

**What came before, and how this improved on it:**
- **Before:** ask a language model to answer from what it remembers in its weights.
- **Its limit:** the RAG abstract says such models have limited ability "to access and precisely manipulate knowledge", and that giving provenance and updating world knowledge are open problems (Lewis et al., arXiv 2005.11401, 22 May 2020, abstract read; first author from memory).
- **What changed:** fetch relevant text from an outside store first and give it to the model. The paper's RAG models set the state of the art on three open-domain question answering tasks and generated more specific, diverse and factual language than a parametric-only baseline (abstract; I did not reproduce it).
- **How much better:** the published gain is above. The code below is only a toy of the mechanism: with 1,000 short documents, a word-overlap search found the right one 100 of 100 times, the model reads 18 words instead of 6,000 (333 times fewer), and editing one document changed the answer source without any retraining.

**Check it yourself:**

```python
docs = [f"the capital of country{i} is city{i}" for i in range(1000)]
tok = lambda s: set(s.split())
hit = 0
for i in range(0, 1000, 10):
    q = tok(f"what is the capital of country{i}")
    best = max(range(1000), key=lambda j: len(q & tok(docs[j]))); hit += best == i
print(f"retrieval by shared words: right document found {hit} of 100 times")
per = len(docs[0].split())
print(f"words the model must read: all documents {1000 * per:,}, only the 3 best {3 * per} ({1000 // 3}x fewer)")
docs[7] = "the capital of country7 is newcity"
q = tok("what is the capital of country7"); print("after editing one document, the answer source says:", docs[max(range(1000), key=lambda j: len(q & tok(docs[j])))])
```

## Related
- [[LLM]] · [[Embedding]] · [[Vector Database]] · [[Transformer]] · [[Tool Calling]] · [[BM25]] · [[Retrieval Metrics]] · [[Contextual Retrieval]] · [[Context Rot]] · [[Vault Search with RAG]] · [[Hallucination]] · [[Chunking Strategies]] · [[Reranking]] · [[RAG Tuning Lab]]

## Sources
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, 2020 (arXiv 2005.11401; abstract read for the history block)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/agents-rag RAG (retrieval-augmented generation):::An open-book test for an AI. Before answering, it looks up the right pages in a book, then writes its answer using those pages.

#flashcards/ml/formula #flashcards/topic/agents-rag Write the retrieval score and explain each symbol.
?
$$\text{score}(q,d)=\frac{E(q)\cdot E(d)}{\lVert E(q)\rVert\,\lVert E(d)\rVert}$$
- $q$: the user's question (the query). $d$: one chunk of a document being checked. Every stored chunk gets its own score against the same $q$.
- $E(\cdot)$: the embedding model that turns text into a vector. The same model must embed both $q$ and $d$, otherwise their numbers live on different maps.
- The fraction is cosine similarity: dot product divided by both lengths, so only direction (meaning) counts.
- The top-scoring chunks are handed to the model.

#flashcards/ml/formula #flashcards/topic/agents-rag Write the RAG answer probability and explain each symbol.
?
$$p(y\mid x)\approx\sum_{z\in \text{top-}k} p_\eta(z\mid x)\,p_\theta(y\mid x,z)$$
- $x$: the input question. $y$: the generated answer. $p(y\mid x)$: the probability of answer $y$ given question $x$.
- $z$: one retrieved document (or chunk), a hidden helper variable.
- $p_\eta(z\mid x)$: how much the retriever trusts document $z$ for this question.
- $p_\theta(y\mid x,z)$: how likely the generator is to write $y$ given the question and that document.
- The sum mixes the answers from the top $k$ documents, weighted by trust.

#flashcards/ml/cloze #flashcards/topic/agents-rag In RAG the ==retriever== finds the chunks and the ==generator== writes the answer from them.

#flashcards/ml/example #flashcards/topic/agents-rag Question vector $E(q)=[3,4]$, chunk A $=[4,3]$, chunk B $=[-4,3]$. What are the scores?::A: dot 24, lengths 5 and 5, score 24/25 = 0.96. B: dot 3·(−4) + 4·3 = 0, score 0. Chunk A wins.
