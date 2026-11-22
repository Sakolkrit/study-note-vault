---
tags: [RAG, retrieval, BM25, embeddings, chunking, contextual-retrieval]
status: growing
level: 4
tier: 3
---

# Contextual Retrieval

A [[RAG]] trick: before indexing, **prepend a short explanation to every chunk** saying where it came from, so a search can find it.

## 🟢 1. Simple (age 12)
Imagine tearing a company report into small scraps. One scrap says "revenue grew by 3% over the previous quarter". Whose revenue? Which year? On its own, nobody can tell, so a search for "ACME's revenue" would never pick it. Contextual retrieval **staples a little label onto each scrap** ("this is from ACME's Q2 2024 report") before filing it, so the search can find the right one.

**Example:** A torn page that only says "He spent the whole day reading". With a label added ("From the story of Teddy the bear, chapter 2"), a search for "what did Teddy do" finds it.

## 🔵 2. How it works
1. A [[RAG]] system cuts documents into **chunks**, so it can retrieve only the relevant pieces for the model.
2. A chunk can lose its context when it is cut out. "The company's revenue grew by 3%" no longer says which company or when.
3. So for each chunk, ask a language model to write a **short context** (one or two sentences) that places the chunk in the whole document. The model is given the whole document and the chunk. The lecture shows the prompt: it asks for a short succinct context that situates the chunk within the document "for the purposes of improving search retrieval".
4. **Prepend** that context to the chunk.
5. Then index the combined text in both ways: make an [[Embedding]] of it, and also build a [[BM25]] keyword index.
6. At question time retrieve as usual, optionally rerank, and pass the chunks to the model.
7. The price: one extra model call per chunk when building the index, which is done once. **Prompt caching** keeps the whole document loaded, so each chunk's call is cheap.

## 🟠 3. Technical detail
- **Source:** Anthropic's post *Introducing Contextual Retrieval* (2024), named in the lecture. I read the page. It describes two sub-techniques, **Contextual Embeddings** and **Contextual BM25**.
- **Measured results (Anthropic's post, read, their own tests):** with the top 20 chunks retrieved, the failure rate was $5.7\%$ baseline. Contextual Embeddings cut it to $3.7\%$ (35% fewer failures), combined with Contextual BM25 to $2.9\%$ (49% fewer), and adding a reranking step to $1.9\%$ (67% fewer). They tested across codebases, fiction, arXiv papers and science papers, with several embedding models; the post says contextualising helped in every combination it evaluated.
- **Cost:** the post estimates the one-time cost of generating the contexts at \$1.02 per million document tokens, assuming 800-token chunks, 8,000-token documents, 50 tokens of context instructions and 100 tokens of context per chunk, using prompt caching. That is Anthropic's price at the time; it will differ for you.
- **Another idea from the lecture:** a hypothetical-document trick (I believe it is called HyDE, from memory) writes a fake answer to the question, and searches with its embedding, to bridge the gap between questions and documents. I did not read the paper.
- **Chunk settings:** embedding size, chunk size and overlap are hyperparameters of the knowledge base (lecture).
- **When not needed:** the same post notes that if the whole knowledge base is under about 200,000 tokens, you can simply put it all in the prompt.
- **The code:** a toy with four chunks and keyword search, to show the mechanism. It is not a test of the 49% claim.

```python
import math
from collections import Counter
def rank_docs(query, docs, k1=1.5, b=0.75):
    toks = [d.lower().replace('.', '').replace(',', '').split() for d in docs]
    N = len(docs); avgdl = sum(map(len, toks)) / N
    df = Counter(w for t in toks for w in set(t))
    scores = []
    for t in toks:
        tf = Counter(t); s = 0.0
        for w in query.lower().split():
            if w in tf: s += math.log((N - df[w] + 0.5) / (df[w] + 0.5) + 1) * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(t) / avgdl))
        scores.append(s)
    return scores
# chunks cut from annual reports of different companies; the chunk that answers the question never says which company it is about
chunks = ['The company revenue grew by 3% over the previous quarter.',
          'Revenue grew by 8% over the previous quarter, driven by teddy bear sales.',
          'The quarter saw a revenue growth of 5% compared with last year.',
          'Net costs fell by 2% over the previous quarter.']
query = 'ACME Corp revenue growth second quarter 2024'
context = ['This chunk is from ACME Corp\'s Q2 2024 report on quarterly results.',
           'This chunk is from Bearly Inc\'s Q1 2023 report on toy sales.',
           'This chunk is from Cuddle Ltd\'s Q2 2022 report on annual results.',
           'This chunk is from Dozy Plc\'s Q2 2021 report on costs.']
before = rank_docs(query, chunks); after = rank_docs(query, [c + ' ' + x for c, x in zip(chunks, context)])
print('before: scores', [round(s, 3) for s in before], '-> best chunk', before.index(max(before)), '(the right one is chunk 0)')
print('after : scores', [round(s, 3) for s in after], '-> best chunk', after.index(max(after)))
print('rank of the right chunk: before', sorted(before, reverse=True).index(before[0]) + 1, '| after', sorted(after, reverse=True).index(after[0]) + 1)
# the price: a short context added to every chunk
chunk_tokens, context_tokens = 800, 100
print('extra text per chunk:', context_tokens, 'of', chunk_tokens, 'tokens =', f'{100 * context_tokens / chunk_tokens:.1f}%', '| chunks in a 8000-token document:', 8000 // chunk_tokens)
```

## 🔴 4. Deep / formal
**Relative reduction in failures.** If the retrieval failure rate falls from $f_0$ to $f_1$:

$$
\text{reduction}=\frac{f_0-f_1}{f_0}
$$

**Where:**
- $f_0$ is the failure rate before (here the share of relevant chunks that are not in the top 20, which is $1-\text{recall@}20$, see [[Retrieval Metrics]]).
- $f_1$ is the failure rate after the change.
- **Matters:** it is a *relative* change, so going from $5.7\%$ to $2.9\%$ is called "49% fewer failures" although the two rates differ by only 2.8 percentage points. Always check both the relative and the absolute figures.

**Example (tiny numbers):** Anthropic's reported rates.
- $5.7\%\to3.7\%$: $(5.7-3.7)/5.7=\mathbf{0.351}$, so 35%.
- $5.7\%\to2.9\%$: $(5.7-2.9)/5.7=\mathbf{0.491}$, so 49%.
- $5.7\%\to1.9\%$: $(5.7-1.9)/5.7=\mathbf{0.667}$, so 67%.
- Meaning: the arithmetic reproduces the percentages in the post. These are Anthropic's measurements, not mine.

**Further notes:**
- **What the toy shows:** four chunks that all talk about quarterly revenue, only chunk 0 being from ACME. With the query "ACME Corp revenue growth second quarter 2024", the plain chunks ranked chunk 2 first and the right chunk 0 second. After adding the context sentences, chunk 0 scored $2.913$ against $0.482$ before and ranked first. The effect is large because I wrote the toy so that only the context contains "ACME". Real corpora are less clear-cut.
- **Index cost:** 100 tokens of context per 800-token chunk is $12.5\%$ more text to index, and a 8,000-token document makes 10 such chunks.
- **Not covered:** how to evaluate which context prompt works best, reranking models, and updating the index when documents change.

**What came before, and how this improved on it:**
- **Before:** chunks embedded and indexed as they were cut out of the document, with embeddings and BM25 used (alone or in a hybrid, lecture).
- **Its limit:** a chunk can lack the context needed to match a question, as in "revenue grew by 3%".
- **What changed:** prepend a model-written, chunk-specific context before embedding and before BM25 indexing (Anthropic, 2024).
- **How much better:** Anthropic's post reports the top-20 failure rate falling from $5.7\%$ to $2.9\%$ (49%), and to $1.9\%$ (67%) with reranking, on their test sets. In this note's toy, the right chunk moved from rank 2 to rank 1. I did not reproduce the published numbers.

## Related
- [[RAG]] · [[BM25]] · [[Retrieval Metrics]] · [[Embedding Model]] · [[Embedding]] · [[Vector Database]] · [[LLM]] · [[Prompt Engineering]] · [[Tokenization]] · [[Prompt Caching]] · [[Vault Search with RAG]] · [[Chunking Strategies]] · [[Reranking]] · [[RAG Tuning Lab]]

## Sources
- Stanford CME 295 (Fall 2025), Lecture 7 slides, Afshine and Shervine Amidi
- Anthropic, *Introducing Contextual Retrieval*, 2024 (anthropic.com/news/contextual-retrieval, page read)
- Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels*, 2022 (named in the lecture; not read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/agents-rag Contextual retrieval:::Stapling a little label onto each torn-off scrap of a document ("this is from ACME's report for that quarter") before indexing it, so a search can find scraps that make no sense on their own.

#flashcards/ml/formula #flashcards/topic/agents-rag Write the relative reduction in failures and explain each symbol.
?
$$\text{reduction}=\frac{f_0-f_1}{f_0}$$
- $f_0$: the failure rate before. Here it is the share of relevant chunks that are not in the top 20, which is $1-\text{recall@}20$.
- $f_1$: the failure rate after the change.
- The result is a relative change, so going from 5.7% to 2.9% is called "49% fewer failures", not "2.8 points".

#flashcards/ml/cloze #flashcards/topic/agents-rag A reduction in failures is a ==relative== change, not a drop in ==percentage points==.

#flashcards/ml/example #flashcards/topic/agents-rag Anthropic reported failure rates going from 5.7% to 3.7%, 2.9% and 1.9%. What are the reductions?::(5.7 − 3.7)/5.7 = 0.351, 35%. (5.7 − 2.9)/5.7 = 0.491, 49%. (5.7 − 1.9)/5.7 = 0.667, 67%. The arithmetic reproduces the percentages in the post (these are Anthropic's own measurements, not rerun).
