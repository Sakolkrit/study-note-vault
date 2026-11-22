---
tags: [RAG, reranking, retrieval, cross-encoder, bi-encoder, two-stage-retrieval, LLM, ML]
status: growing
level: 4
tier: 3
---

# Reranking

Reranking is a **second look at a search result list**. A fast first search picks a few dozen candidates. A slower, more careful model then re-orders just those candidates, so the best few go to the language model in a [[RAG]] system.

## 🟢 1. Simple (age 12)
You ask a librarian for a book on "how long to boil an egg". The librarian runs along the shelves and grabs 20 books whose covers look right. That is quick but rough. Then a careful reader opens each of the 20 with your question in mind and puts them in order, best first. **The quick grab is the first search. The careful reading is reranking.** You cannot read the whole library carefully, so the careful reader only sees the 20.

**Example:** the quick grab brings "Eggs: a guide", "Chickens at dawn" and "Soft-boiled eggs in 6 minutes". The careful reader reads all three next to your question and puts the third one on top.

## 🔵 2. How it works
1. A [[RAG]] system first **retrieves** a shortlist: each chunk of text ([[Chunking Strategies]]) was turned into a vector ahead of time by an [[Embedding Model]], and the question's vector is compared with all of them ([[Cosine Similarity]], [[Vector Database]]). Keyword scoring ([[BM25]]) can make a shortlist too, and the two can be blended.
2. This first stage is fast because the question and each chunk are **never read together**. Each was squeezed into one vector on its own. That squeezing loses detail. This kind of model is called a **bi-encoder** (two separate encodings).
3. A **cross-encoder** reads the question and one chunk **at the same time** as one piece of text, and outputs one score for how well the chunk answers the question. It can notice small things, like a word's meaning in this very question, because it sees both.
4. The catch is cost: a cross-encoder has to run once for **every** (question, chunk) pair, from scratch, for every new question. Nothing can be prepared ahead. Running it over a whole library is far too slow.
5. So we do both: the cheap first stage keeps the top $N$ chunks (for example 20 or 50), the cross-encoder re-scores only those $N$, and the best few go into the prompt.
6. Reranking can only **re-order** the shortlist. If the right chunk did not make it into the $N$, the reranker never sees it.
7. Measure it: the same questions, with and without the reranker, scored with [[Retrieval Metrics]]. In my test it **did not help** the blended search at all (details below).

![[rag-schema.svg]]
*A basic RAG pipeline: the question and the documents are encoded separately, the vector database returns context, and the language model answers. This figure has no reranker. A reranker would sit between the vector database and the context box, re-ordering what the database returns. Figure by Gknor, [RAG schema.svg on Wikimedia Commons](https://commons.wikimedia.org/wiki/File:RAG_schema.svg), CC BY-SA 4.0 (my note about where a reranker goes is not part of the figure).*

## 🟠 3. Technical detail
- **The models.** First stage: `all-MiniLM-L6-v2` (a bi-encoder, 22.7 million parameters, counted from the loaded model) and [[BM25]]. Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`, which the library loads as `BertForSequenceClassification` with one output, 6 layers, 384-wide, **22.7 million parameters**, the same size as the bi-encoder (all counted from the loaded models). It was trained on a web-search question-and-passage data set (called MS MARCO; from memory, I did not read its model card), which matters below.
- **Same size, very different cost.** The two models are the same size. The cross-encoder is slow because it runs once per pair at question time, while the bi-encoder runs once per chunk at index time.
- **Typical settings.** First stage $N$ of 20 to 100, then keep the top 3 to 10. A bigger $N$ raises the chance that the right chunk is in the shortlist and costs proportionally more reranking.
- **Blending.** Instead of trusting the reranker alone, blend its rank with the first-stage rank (for example by reciprocal rank fusion).
- **Code:** a toy with five passages, a bi-encoder and a cross-encoder (this is a **chosen toy example**: both models get first place right, and it only shows how the two scores differ).

```python
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

bi = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")          # bi-encoder: question and passage are turned into vectors separately
ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")                   # cross-encoder: question and passage are read together, one score out

query = "How long should I boil an egg for a runny yolk?"
passages = [
    "Eggs are a good source of protein and are sold in boxes of six or twelve.",                 # same topic, no answer
    "Boil the water first, then lower the egg in. A soft-boiled egg with a runny yolk needs about 6 minutes.",   # the answer
    "A runny nose is often the first sign of a cold; it usually lasts a few days.",              # shares 'runny'
    "The boiling point of water falls as you go up a mountain, so cooking takes longer.",        # shares 'boil', 'long'
    "Chickens lay about one egg a day when they get enough light.",
]
q = bi.encode([query], normalize_embeddings=True)[0]
P = bi.encode(passages, normalize_embeddings=True)
cos = P @ q                                                                  # bi-encoder score: cosine of two separately made vectors
cross = ce.predict([(query, p) for p in passages])                           # cross-encoder score: one number per (question, passage) pair
for name, s in (("bi-encoder cosine", cos), ("cross-encoder score", cross)):
    print(name)
    for r, i in enumerate(np.argsort(-s)): print(f"  {r + 1}. {s[i]:7.3f}  {passages[i][:70]}")
```

Output:

```text
bi-encoder cosine
  1.   0.783  Boil the water first, then lower the egg in. A soft-boiled egg with a 
  2.   0.407  Chickens lay about one egg a day when they get enough light.
  3.   0.381  Eggs are a good source of protein and are sold in boxes of six or twel
  4.   0.283  The boiling point of water falls as you go up a mountain, so cooking t
  5.   0.241  A runny nose is often the first sign of a cold; it usually lasts a few
cross-encoder score
  1.   8.756  Boil the water first, then lower the egg in. A soft-boiled egg with a 
  2.  -8.155  A runny nose is often the first sign of a cold; it usually lasts a few
  3. -10.268  Chickens lay about one egg a day when they get enough light.
  4. -10.683  The boiling point of water falls as you go up a mountain, so cooking t
  5. -11.009  Eggs are a good source of protein and are sold in boxes of six or twel
```

The bi-encoder cosines run from 0.783 to 0.241, and the cross-encoder scores from 8.756 to -11.009 (a raw score, not a probability). The cross-encoder also put the "runny nose" passage second because of the shared word "runny", ahead of the egg passages that do not answer the question. A reranker is a model, so it makes mistakes of its own.

**What I measured** ([[RAG Tuning Lab]]): the vault's 318 notes cut by heading into chunks of at most 170 words (4,866 chunks), 49 questions I wrote and 99 from a public list. Numbers are the share of questions with a correct note at rank 1 and in the top 5 (note level), and the mean reciprocal rank (MRR).

| First stage, then the cross-encoder on its top $N$ | 49 own: top 1 | top 5 | MRR | 99 outside: top 1 | top 5 | MRR |
|---|---|---|---|---|---|---|
| keywords only (BM25), no reranker | 0.53 | 0.90 | 0.709 | 0.41 | 0.83 | 0.577 |
| keywords, $N=50$, reranked | 0.49 | 0.96 | 0.695 | 0.57 | 0.92 | 0.710 |
| meaning only, no reranker | 0.57 | 0.90 | 0.717 | 0.67 | 0.92 | 0.771 |
| meaning, $N=10$, reranked | 0.61 | 0.94 | 0.754 | 0.60 | 0.91 | 0.733 |
| meaning, $N=50$, reranked | 0.53 | 0.96 | 0.700 | 0.54 | 0.91 | 0.691 |
| blend of both, no reranker | **0.69** | **1.00** | **0.822** | **0.63** | **0.93** | **0.754** |
| blend, $N=20$, reranked | 0.53 | 0.96 | 0.723 | 0.57 | 0.93 | 0.721 |
| blend, $N=50$, reranked | 0.51 | 0.94 | 0.699 | 0.57 | 0.90 | 0.706 |

- **It helped only the weakest first stage.** After keyword search, the reranker raised top-5 on the 99 outside questions from 0.83 to 0.92 (nine more questions) and the MRR from 0.577 to 0.710. Keyword search has the most room to improve (it matches words, not meaning).
- **It did not help the blend; it hurt it at rank 1.** The blended search without a reranker was the best or tied for best in most columns. Adding the reranker took top-1 from 0.69 to 0.53 on my 49 questions (eight fewer questions right at rank 1) and from 0.63 to 0.57 on the 99 (six fewer). Top-5 stayed within 0.07 of the unreranked blend.
- **Meaning search was a mixed case.** The reranker helped at $N=10$ on my questions (MRR 0.717 to 0.754) and hurt on the outside ones (0.771 to 0.733). One question is 2 points on 49 and 1 point on 99, so those moves are a few questions.
- **A bigger shortlist did not help either.** For the blend, $N=20$ and $N=50$ gave about the same results. The shortlist already held a correct note in 0.99 and 1.00 of the outside questions at those sizes, so the problem is the reranker's order, not what was missing.
- **A blend of the two ranks recovers most of it, not all.** Reciprocal-rank-fusing the reranker with the first-stage rank (blend, $N=20$) gave top-1 0.67 and 0.60 and MRR 0.800 and 0.743, against 0.69, 0.63, 0.822 and 0.754 with no reranker. So blending reduces the harm but did not beat the first stage here.
- **Leaving the title off the reranker's input made it worse** on the outside questions (MRR 0.660 against 0.721, top-3 0.75 against 0.86), so the chunk's note name and heading matter to the cross-encoder too.
- **I do not know why it hurt.** Candidates, none tested: the model was trained on web search questions, not on short study-notes questions; my labels are at note level, so a chunk from a different note that really answers the question would count as wrong; the blend was already strong (top-5 of 1.00 and 0.93 leaves little room); and these notes mix prose, markdown and maths.
- **Time** (not reliable, see "What I did not run"): the cross-encoder on 20 chunks took 2,492 ms per question in one run and 10,528 ms in another, against 36 ms and 1,169 ms for the whole first stage. The laptop was busy during the runs, so read it as "seconds, not milliseconds" and nothing more precise.

## 🔴 4. Deep / formal
**Two ways to score a chunk.** With $q$ the question and $d$ a chunk, $E$ the embedding model and $f_\theta$ the cross-encoder:

$$
s_{\text{bi}}(q,d)=\cos\big(E(q),\,E(d)\big),\qquad s_{\text{cross}}(q,d)=f_\theta\big([q\,;\,d]\big)
$$

**Where:**
- $q$ is the question and $d$ one chunk of text.
- $E(\cdot)$ turns a text into a vector on its own. $\cos$ is the angle score of two vectors ([[Cosine Similarity]]), between -1 and 1.
- $[q\,;\,d]$ means the question and the chunk glued into one input, so the model reads them together.
- $f_\theta$ is the cross-encoder with learned weights $\theta$. It returns one raw number, higher meaning a better match, with no fixed range.
- **Matters:** in $s_{\text{bi}}$ the chunk vector $E(d)$ does not depend on the question, so it is computed **once**, ahead of time, for every chunk. In $s_{\text{cross}}$ nothing can be computed ahead, since the input contains the question. That is why it is more precise and why it only fits a short list.

**Example (tiny numbers):** the egg question and two of the five toy passages (from the code above).
- The answer passage ("Boil the water first... about 6 minutes"): $s_{\text{bi}}=0.783$, $s_{\text{cross}}=8.756$.
- A passage about nothing relevant ("Eggs are a good source of protein..."): $s_{\text{bi}}=0.381$, $s_{\text{cross}}=-11.009$.
- Both models rank the answer first. The bi-encoder gap is $0.783-0.381=0.402$ on a scale of -1 to 1. The cross-encoder gap is $8.756-(-11.009)=19.765$ raw points. Meaning: the cross-encoder separates them more sharply, but its scores are not on a scale you can compare with cosines.

**What a question costs.** With $T_1$ the time of the first stage, $N$ the number of chunks reranked and $t_{\text{ce}}$ the time of the cross-encoder on one pair:

$$
T = T_1 + N\,t_{\text{ce}}
$$

**Where:**
- $T$ is the delay before the language model gets its chunks (it adds to every answer).
- $T_1$ is the first stage (embedding the question and searching the vectors, plus any keyword scoring).
- $N$ is the shortlist length handed to the reranker. $t_{\text{ce}}$ is the time to score one (question, chunk) pair.
- **Matters:** $T$ grows in a straight line with $N$. Doubling the shortlist roughly doubles the reranking part, and at typical numbers that part dominates $T_1$.

**Example (tiny numbers, made up to be easy):** $T_1=40$ ms and $t_{\text{ce}}=15$ ms.
- $N=20$: $T=40+20\cdot15=\mathbf{340}$ ms.
- $N=50$: $T=40+50\cdot15=\mathbf{790}$ ms.
- Meaning: going from 20 to 50 candidates more than doubles the delay (790 against 340 ms). My own measured times on a busy laptop were seconds per question, so the real $t_{\text{ce}}$ here was far above 15 ms.

**What a reranker cannot do.** Let $\text{recall}@N$ be the share of questions whose correct item is somewhere in the first-stage top $N$. After any reranking of those $N$:

$$
\text{hit}@k\ (\text{after reranking})\;\le\;\text{recall}@N\ (\text{first stage})
$$

**Where:**
- $\text{hit}@k$ is the share of questions with a correct item in the top $k$ of the final list.
- $\text{recall}@N$ is the share with a correct item anywhere in the shortlist of $N$.
- **Matters:** reranking only re-orders the shortlist, so it cannot put an item into the top $k$ that was never in the shortlist. The first stage sets a ceiling, and a better reranker cannot lift it. Raise $N$ to raise the ceiling, at the cost in the previous equation.

**Example (tiny numbers):** 10 questions, and a correct chunk is in the shortlist for 9 of them.
- The ceiling is $9/10=\mathbf{0.9}$. Even a perfect reranker cannot score above 0.9 on this set.
- In my data (note level, 99 outside questions, blended first stage): shortlist of 10 held a correct note for 0.91 of the questions, and the reranked top 5 scored 0.88, under the ceiling. With 20 it was 0.99 against 0.93, also under it.
- Meaning: here the ceiling was nearly 1, so the shortfall came from the reranker's order, not from a missing candidate.

- **Edge cases and limits.**
  - A cross-encoder reads a limited length (a BERT model has a fixed maximum input; I did not check this model's exact limit), so very long chunks are cut for it as well ([[Chunking Strategies]]).
  - Scores are only comparable within one question, so they cannot be used for a "no good answer" threshold without calibration ([[Calibration]]).
  - Rerankers trained on web search can underperform on other kinds of text. My result is one example, not a rule: the same family of rerankers is widely reported to help in other settings, which I did not check.
  - **Not tested here:** other rerankers (larger cross-encoders, models that are prompted to rank, or late-interaction models such as ColBERT, which I know only from memory), fine-tuning the reranker on my own data, listwise reranking, and whether a better final answer follows from a better order.

**What came before, and how this improved on it:**
- **Before:** one-stage retrieval. Rank every chunk with one fast score, either keywords ([[BM25]]) or a bi-encoder, and send the top few to the model.
- **Its limit:** a bi-encoder squeezes each chunk into one vector without seeing the question, so fine differences are lost. Keyword search misses meaning. In my test keyword search alone put a correct note at rank 1 for only 0.41 of the 99 outside questions.
- **What changed:** a second stage where a cross-encoder reads the question and each shortlisted chunk together. Using BERT-style models this way for passage ranking was published around 2019 (from memory, not checked here).
- **How much better:** measured here, **it depends on the first stage.** On keyword search it raised top-1 on the 99 outside questions from 0.41 to 0.57 (+16 questions, $N=50$). On the blend of keywords and meaning, which was already stronger, it **lowered** top-1 from 0.63 to 0.57. I have no published figure for this model.

## Related
- [[RAG]] · [[Chunking Strategies]] · [[Contextual Retrieval]] · [[Embedding Model]] · [[Vector Database]] · [[BM25]] · [[Cosine Similarity]] · [[Retrieval Metrics]] · [[Context Window]] · [[Context Rot]] · [[BERT]] · [[Transformer]] · [[Attention]] · [[Calibration]] · [[RAG Tuning Lab]] · [[Vault Search with RAG]]
- [[Recommender Systems]]
## Sources
- My own experiment, [[RAG Tuning Lab]] (code, tables and run details there). Questions as in [[Chunking Strategies]]: 49 by hand and 99 outside question titles (public repo *LLM Interview Questions and Answers Hub* by Kalyan KS, Apache-2.0), labelled by me.
- Model facts (architecture class, 6 layers, 22.7 million parameters) were read from the loaded models. The training data of the cross-encoder, the 2019 BERT passage-ranking work and ColBERT are named from memory.
- Not read: any paper on reranking, and the cross-encoder's model card.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/agents-rag Reranking:::A second, slower and more careful model that re-orders the shortlist from the fast first search, so the best few chunks go to the language model.

#flashcards/ml/reverse #flashcards/topic/agents-rag Cross-encoder:::A model that reads the question and one chunk together as a single input and outputs one relevance score. More precise than a bi-encoder, but it must run once per pair at question time.

#flashcards/ml/reverse #flashcards/topic/agents-rag Bi-encoder:::A model that turns the question and each chunk into vectors separately. The chunk vectors are computed once ahead of time, so search is fast, but each vector is made without seeing the question.

#flashcards/ml/concept #flashcards/topic/agents-rag Why is reranking applied only to a shortlist?
?
A cross-encoder must run from scratch for every (question, chunk) pair at question time, so running it over the whole collection is far too slow. The cheap first stage keeps the top N and only those are reranked.

#flashcards/ml/cloze #flashcards/topic/agents-rag A reranker can only ==re-order== the shortlist, so the first stage sets a ==ceiling== on what it can reach.

#flashcards/ml/formula #flashcards/topic/agents-rag How do a bi-encoder score and a cross-encoder score differ in form?
?
$$
s_{\text{bi}}(q,d)=\cos\big(E(q),\,E(d)\big),\qquad s_{\text{cross}}(q,d)=f_\theta\big([q\,;\,d]\big)
$$
- $q$ is the question, $d$ a chunk. $E$ embeds a text alone, $\cos$ compares two vectors, between -1 and 1.
- $[q\,;\,d]$ is the question and chunk read together by the model $f_\theta$, which returns one raw score.
- The chunk vector $E(d)$ does not depend on the question, so it is precomputed. $f_\theta$ depends on both, so nothing can be precomputed: more precise, much slower.

#flashcards/ml/concept #flashcards/topic/agents-rag In the egg toy example, what were the bi-encoder and cross-encoder scores for the answer and for an irrelevant passage?
?
Answer passage: cosine 0.783, cross-encoder 8.756. Irrelevant "good source of protein" passage: cosine 0.381, cross-encoder -11.009. Both rank the answer first. Gaps: 0.402 against 19.765 raw points. Chosen toy example, my run.

#flashcards/ml/formula #flashcards/topic/agents-rag Time per question with a first stage and a reranker over $N$ chunks?
?
$$
T = T_1 + N\,t_{\text{ce}}
$$
- $T$ is the delay before the language model gets its chunks. $T_1$ is the first stage time.
- $N$ is the number of chunks reranked. $t_{\text{ce}}$ is the cross-encoder time for one pair.
- $T$ grows linearly with $N$, and the reranking part usually dominates.

#flashcards/ml/concept #flashcards/topic/agents-rag With $T_1=40$ ms and $t_{\text{ce}}=15$ ms, what is $T$ for N = 20 and N = 50?
?
$N=20$: $40+20\cdot15=340$ ms. $N=50$: $40+50\cdot15=790$ ms. Going from 20 to 50 candidates more than doubles the delay. Made-up easy numbers, not measured.

#flashcards/ml/formula #flashcards/topic/agents-rag What upper limit does the first stage put on reranking?
?
$$
\text{hit}@k\ (\text{after reranking})\;\le\;\text{recall}@N\ (\text{first stage})
$$
- $\text{hit}@k$ is the share of questions with a correct item in the final top $k$.
- $\text{recall}@N$ is the share with a correct item anywhere in the first-stage top $N$.
- Reranking only re-orders the shortlist, so it cannot lift a result that was never in it. Raise $N$ to raise the ceiling, at higher cost.

#flashcards/ml/concept #flashcards/topic/agents-rag A shortlist holds a correct chunk for 9 of 10 questions. What is the best score any reranker can reach?
?
$9/10=0.9$ for hit@k. A perfect reranker can re-order but cannot add a chunk that is not in the shortlist.

#flashcards/ml/image #flashcards/topic/agents-rag ![[rag-schema.svg]]
Where would a reranker go in this pipeline?
?
Between the vector database and the context box, re-ordering what the database returns before it goes to the language model. The figure itself has no reranker. Figure by Gknor, Wikimedia Commons, CC BY-SA 4.0.

#flashcards/ml/concept #flashcards/topic/agents-rag In my test, when did the cross-encoder reranker help and when did it hurt?
?
It helped the weakest first stage: after keyword search, top-1 on 99 outside questions rose from 0.41 to 0.57 (N = 50). On the blend of keywords and meaning it hurt: top-1 fell from 0.69 to 0.53 on my 49 questions and from 0.63 to 0.57 on the 99. One model, one corpus, note-level labels, and the cause is untested.

#flashcards/ml/concept #flashcards/topic/agents-rag Did a bigger shortlist or a blend of ranks rescue the reranker in my test?
?
No. N = 20 and N = 50 gave about the same results, with a correct note already in the shortlist for 0.99 and 1.00 of outside questions. Blending the reranker rank with the first-stage rank cut the harm (top-1 0.67 and 0.60) but stayed below no reranker (0.69 and 0.63).

#flashcards/ml/concept #flashcards/topic/agents-rag What was not tested in the reranking experiment?
?
Other rerankers (larger cross-encoders, prompted rankers, ColBERT-style models, named from memory), fine-tuning on my own data, listwise reranking, reliable timing, and whether a better order gives a better final answer.
