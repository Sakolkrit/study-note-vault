---
tags: [RAG, chunking, retrieval, embeddings, text-splitting, LLM, ML]
status: growing
level: 4
tier: 2
---

# Chunking Strategies

Chunking is how a [[RAG]] system **cuts documents into pieces** before it indexes them. The pieces (chunks) are what gets searched, and what gets pasted into the model's prompt. How you cut decides what the search can find.

## 🟢 1. Simple (age 12)
Imagine a huge cookbook and you want "how long do I boil an egg?". You do not carry the whole book to the kitchen. You tear it into small cards, one idea per card, and you pick the right card. **Chunking is deciding where to tear.** Tear too small and a card says "boil it for 9 minutes" without saying what. Tear too big and the card is a whole chapter, and you cannot find the egg inside it.

**Example:** a card that says "Cook it for 12 minutes" is useless alone. A card that says "Hard-boiled eggs: cook them for 12 minutes" works. A good chunk can be understood on its own.

## 🔵 2. How it works
1. A [[RAG]] system cannot paste the whole library into the model, because of the [[Context Window]] and because long prompts get noisy ([[Context Rot]]). So it keeps only the best few pieces.
2. Each piece is turned into a vector by an [[Embedding Model]] and stored in a [[Vector Database]]. A question is turned into a vector too, and the closest pieces win ([[Cosine Similarity]]). Keywords can score pieces as well ([[BM25]]).
3. So the **piece** is the unit of search. If the answer is cut in half by a boundary, neither half looks like the question.
4. The common ways to cut:
   - **Fixed size:** every N words (or tokens), ignoring the text's shape. Simple, but it cuts sentences in half and glues unrelated topics together.
   - **Fixed size with overlap:** each window starts a bit before the last one ended, so a sentence cut by one boundary is whole in the next window. It costs more chunks.
   - **By sentences or paragraphs:** glue whole sentences until a size limit. Nothing is cut mid-sentence.
   - **By structure:** follow the document's own headings, so one section is one chunk (and long sections are cut again).
   - **Add context to each chunk:** put the document title and heading in front of the chunk's text, so a chunk that says "it grew by 3%" says what "it" is. [[Contextual Retrieval]] goes further and has a model write that context.
5. Different documents need different cuts: a table, a lecture, a dialogue and a manual are not shaped alike.

![[rag-document-types.png]]
*Four kinds of documents a RAG system meets: a table, a lecture, a dialogue and documentation. Each has a different natural boundary, so one cutting rule does not fit all. Figure by Numiri, [Rag-doc-styles.png on Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Rag-doc-styles.png), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Chunk size is a trade-off.** Small chunks are precise (the match is the answer) but lose context and make many vectors to store. Big chunks keep context but blur several ideas into one vector and fill the prompt with extra words.
- **The embedding model has a limit on what it reads.** The model used in `RAG Tuning Lab`, all-MiniLM-L6-v2, reads at most **256 word pieces** (`max_seq_length` printed by the library). Anything after that is silently ignored by the vector, though keyword search still sees it. The words-to-pieces ratio is worse than you think for notes with maths and markdown: **78% of my 164-word fixed chunks were over the limit**, 13% of the 98-word ones, and none of the 50-word ones (measured, see below).
- **Overlap** repeats text. 40 words of overlap on 170-word windows made 4,644 chunks instead of 3,577 (30% more to embed and store).
- **Put the title and heading in front of the chunk.** In my test this was the clearest effect of all (below).
- **Match the unit you measure.** If you score "did the right note come back" you are measuring notes, not chunks, and big chunks look better than they are. See the warning below.
- **Code to try:** four ways to cut a 39-word text (the same functions the experiment uses, in a tiny version).

```python
import re

def windows(words, size, overlap=0):                     # fixed-size windows, each starting (size - overlap) words after the last
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step) if len(words[i:i + size]) >= 3]

def by_sentences(text, target):                          # whole sentences glued together until about `target` words
    out, cur = [], []
    for s in re.split(r"(?<=[.!?])\s+", text):
        if cur and len(" ".join(cur + [s]).split()) > target:
            out.append(" ".join(cur)); cur = []
        cur.append(s)
    return out + [" ".join(cur)]

def by_heading(text):                                    # one chunk per '## ' section, the heading goes in front
    return [p.strip().replace("\n", " ") for p in re.split(r"\n(?=## )", text) if p.strip()]

doc = """## Cats
Cats sleep a lot. A cat can sleep sixteen hours a day. They hunt at dawn and dusk.
## Dogs
Dogs need walks. A dog that is walked twice a day is calmer. Many dogs also like to swim."""

words = doc.replace("## ", "").split()
print("total words:", len(words))
for name, chunks in [("fixed 12 words, no overlap", windows(words, 12)),
                     ("fixed 12 words, overlap 4", windows(words, 12, 4)),
                     ("by sentences, about 14 words", by_sentences(re.sub(r"## \w+\n", "", doc), 14)),
                     ("by heading", by_heading(doc))]:
    print(f"\n{name}: {len(chunks)} chunks")
    for c in chunks: print(f"  [{len(c.split()):2d} words] {c}")
```

Output:

```text
total words: 39

fixed 12 words, no overlap: 4 chunks
  [12 words] Cats Cats sleep a lot. A cat can sleep sixteen hours a
  [12 words] day. They hunt at dawn and dusk. Dogs Dogs need walks. A
  [12 words] dog that is walked twice a day is calmer. Many dogs also
  [ 3 words] like to swim.

fixed 12 words, overlap 4: 5 chunks
  [12 words] Cats Cats sleep a lot. A cat can sleep sixteen hours a
  [12 words] sleep sixteen hours a day. They hunt at dawn and dusk. Dogs
  [12 words] dawn and dusk. Dogs Dogs need walks. A dog that is walked
  [12 words] dog that is walked twice a day is calmer. Many dogs also
  [ 7 words] calmer. Many dogs also like to swim.

by sentences, about 14 words: 4 chunks
  [12 words] Cats sleep a lot. A cat can sleep sixteen hours a day.
  [ 9 words] They hunt at dawn and dusk. Dogs need walks.
  [10 words] A dog that is walked twice a day is calmer.
  [ 6 words] Many dogs also like to swim.

by heading: 2 chunks
  [20 words] ## Cats Cats sleep a lot. A cat can sleep sixteen hours a day. They hunt at dawn and dusk.
  [21 words] ## Dogs Dogs need walks. A dog that is walked twice a day is calmer. Many dogs also like to swim.
```

**What the experiment found** (details, code and the full table are in [[RAG Tuning Lab]]): 318 notes of this vault cut 11 ways, searched by keywords, by meaning and by both, on 49 questions I wrote and 99 questions from a public list ([[Retrieval Metrics]]). Numbers are hit@5, the share of questions with a correct note in the top 5 notes, for the blended search, on the 49 and then the 99 questions.

| Chunking | Chunks | Mean words | Over the 256 limit | Blend, 49 questions | Blend, 99 questions |
|---|---|---|---|---|---|
| whole note, one chunk | 317 | 1,847 | 99% | 0.98 | 0.92 |
| fixed 300 words | 2,102 | 279 | 94% | 0.98 | 0.96 |
| fixed 170 words | 3,577 | 164 | 78% | 0.96 | 0.94 |
| fixed 100 words | 5,976 | 98 | 13% | 0.96 | 0.93 |
| fixed 50 words | 11,795 | 50 | 0% | 0.98 | 0.94 |
| fixed 170, overlap 40 | 4,644 | 163 | 78% | 0.98 | 0.95 |
| by heading, whole section | 2,548 | 230 | 45% | 0.98 | 0.93 |
| by heading, max 170 words | 4,866 | 120 | 49% | 1.00 | 0.93 |
| by heading, max 100 words | 7,021 | 83 | 12% | 0.98 | 0.95 |
| by sentences, about 100 words | 7,590 | 77 | 8% | 0.98 | 0.92 |
| by heading, max 170, **no title in front** | 4,866 | 120 | 46% | 1.00 | **0.86** |

- **No chunk size won.** The ten rows with the title in front are within 0.04 of each other in both columns, and on 99 questions one question is 0.01 (on 49 questions, 0.02). That is a handful of questions, so I cannot rank these cuts. Mean reciprocal rank moved a bit more but is just as noisy.
- **Leaving the title and heading off was the only clear loss:** the blend dropped from 0.93 to 0.86 on the 99 outside questions (seven questions), its mean reciprocal rank from 0.75 to 0.61, keyword search alone from 0.83 to 0.68, and meaning search alone from 0.92 to 0.83. On my own 49 questions the blend stayed at 1.00 (mean reciprocal rank 0.82 to 0.77), so the effect shows up where the wording is different from the notes.
- **The whole-note row is the warning sign.** Treating each note as one chunk scored 0.98 and 0.92, as good as the others, because the test only checks that the right *note* comes back. But each retrieved item is then 1,847 words on average against 120 for heading chunks of at most 170 words, which is **15 times more text** (my arithmetic: 1,847 over 120) for the model to read per item. This test cannot see that cost, and cannot see whether the retrieved piece contains the answer. A test with the answer's exact chunk labelled would.

## 🔴 4. Deep / formal
**How many chunks.** A document of $L$ words, cut into windows of $s$ words where each window overlaps the previous one by $o$ words, gives

$$
n=\left\lceil \frac{L-o}{s-o} \right\rceil
$$

**Where:**
- $L$ is the length of the document in words (or tokens).
- $s$ is the window (chunk) size.
- $o$ is the overlap, the number of words a window shares with the one before it. $s-o$ is the step between window starts.
- $n$ is the number of chunks. $\lceil\cdot\rceil$ rounds up, because a short last window still counts.
- **Matters:** the number of chunks is what you embed, store and search, so cost grows with it. Halving $s$ roughly doubles $n$, and overlap raises it, so $o$ has to be paid for. As $o$ approaches $s$, the step shrinks and $n$ explodes. The formula assumes the last short window is kept; my chunker drops a last window under 12 words, which only matters at the very end.

**Example (tiny numbers):** a note of $L=500$ words, windows of $s=170$.
- No overlap ($o=0$): $n=\lceil 500/170\rceil=\lceil 2.94\rceil=\mathbf{3}$.
- Overlap $o=40$: step $=170-40=130$, so $n=\lceil (500-40)/130\rceil=\lceil 3.54\rceil=\mathbf{4}$.
- Meaning: 40 words of overlap cost one extra chunk here (a third more). I checked the formula against the real chunker for four settings and it matched every time (output below).

**How likely a boundary cuts an idea.** Say the idea you need is a span of $w$ consecutive words, at a random place, and chunks do not follow sentence ends. It is cut (not whole in any chunk) with probability about

$$
p_{\text{cut}}\approx\frac{\max(0,\;w-1-o)}{s-o}
$$

**Where:**
- $w$ is the length of the span (a sentence or an answer), in words.
- $s$ and $o$ are the window size and overlap as above.
- $p_{\text{cut}}$ is the chance that no window contains the whole span.
- **Matters:** it falls as chunks get bigger and reaches **zero when the overlap is at least $w-1$**: a span that short always fits whole inside some window. Without overlap, a 40-word sentence in 100-word windows is cut about four times in ten.

**Example (tiny numbers):** a sentence of $w=40$ words, windows of $s=170$.
- No overlap: $p=\max(0,39)/170=39/170=\mathbf{0.229}$. About 1 sentence in 4 is cut.
- Overlap $o=40$: $\max(0,\,39-40)=0$, so $p=\mathbf{0}$. No such sentence is ever cut.
- Meaning: an overlap as long as the idea removes cutting, at the price of more chunks. A simulation with random sentence positions gave 0.225 and 0.000 (output below). Real text has sentence ends, so cutting at them avoids the problem more cheaply than overlap does.

```python
import math, random

def windows(words, size, overlap, keep=12):             # same cutting rule as the experiments: drop a last window under 12 words
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step) if len(words[i:i + size]) >= keep]

def n_chunks(L, s, o):                                  # windows of s words, each starting s-o words after the last
    return math.ceil((L - o) / (s - o))

# 1. count formula against the real chunker (text of exactly L words)
for L, s, o in [(500, 170, 0), (500, 170, 40), (300, 100, 20), (1000, 50, 0)]:
    real = len(windows(["w"] * L, s, o))
    print(f"L={L} s={s} o={o}: formula {n_chunks(L, s, o)} | windows() {real}")

# 2. chance that a sentence of w words is cut by a boundary (random position), simulated
def cut_rate(w, s, o, L=2000, trials=20000, seed=0):
    random.seed(seed); step = s - o; starts = list(range(0, L, step)); cut = 0
    for _ in range(trials):
        a = random.randrange(0, L - w); b = a + w                     # the sentence covers words a .. b-1
        cut += not any(st <= a and b <= st + s for st in starts)      # is it whole inside at least one window?
    return cut / trials
for w, s, o in [(40, 170, 0), (40, 100, 0), (40, 170, 40), (40, 170, 20), (100, 100, 0), (100, 170, 40)]:
    p = max(0, (w - 1 - o)) / (s - o)
    print(f"sentence {w} words, window {s}, overlap {o}: formula {p:.3f} | simulated {cut_rate(w, s, o):.3f}")
```

Output:

```text
L=500 s=170 o=0: formula 3 | windows() 3
L=500 s=170 o=40: formula 4 | windows() 4
L=300 s=100 o=20: formula 4 | windows() 4
L=1000 s=50 o=0: formula 20 | windows() 20
sentence 40 words, window 170, overlap 0: formula 0.229 | simulated 0.225
sentence 40 words, window 100, overlap 0: formula 0.390 | simulated 0.382
sentence 40 words, window 170, overlap 40: formula 0.000 | simulated 0.000
sentence 40 words, window 170, overlap 20: formula 0.127 | simulated 0.124
sentence 100 words, window 100, overlap 0: formula 0.990 | simulated 0.989
sentence 100 words, window 170, overlap 40: formula 0.454 | simulated 0.437
```

- **Edge cases and limits.**
  - Cutting by words is not cutting by tokens: the model's limit is in word pieces, and the ratio changed from note to note (78% of 164-word chunks were over a limit of 256 pieces).
  - A chunk needs to stand alone. Pronouns ("it", "this method") and tables lose their meaning when cut ([[Contextual Retrieval]]).
  - Cutting by structure assumes the structure exists. Plain text without headings falls back to fixed windows.
  - **Parent-child retrieval** (search small chunks, but give the model the bigger section they came from) and **late chunking** (embed the whole text first, then cut the vectors) are known ideas that I did **not** test, and I am describing them from memory.
  - **What I did not test:** semantic chunking (cutting where the meaning changes), chunk-level labels, a long-context model reading whole documents, or any language model reading the chunks.

**What came before, and how this improved on it:**
- **Before:** search systems indexed and returned whole documents (how classic keyword search worked; from memory, not checked).
- **Its limit:** for a [[RAG]] system, a whole document is too long to embed (the average note here is 1,847 words, and 99% of whole notes were longer than the 256-piece limit of the embedding model) and too long to paste in (five whole notes would be about 9,200 words).
- **What changed:** cut documents into chunks of roughly 50 to 300 words, preferably at headings or sentences, and put the document and section name in front of each.
- **How much better:** 15 times fewer words per retrieved item (1,847 against 120, measured), with the same hit@5 on a note-level test (0.98 against 1.00 on my 49 questions, 0.92 against 0.93 on the 99). So the gain was **cost and focus**, not a higher score on this test. The one large measured gain was from the title in front: 0.86 to 0.93 on the 99 questions.

## Related
- [[RAG]] · [[Reranking]] · [[Contextual Retrieval]] · [[Embedding Model]] · [[Vector Database]] · [[BM25]] · [[Cosine Similarity]] · [[Retrieval Metrics]] · [[Context Window]] · [[Context Rot]] · [[Tokenization]] · [[RAG Tuning Lab]] · [[Vault Search with RAG]]

## Sources
- My own experiment, [[RAG Tuning Lab]] (code and tables there). Corpus: 318 notes copied from this vault. Questions: 49 written by hand (from [[Vault Search with RAG]]) and 99 question titles from the public repo *LLM Interview Questions and Answers Hub* by Kalyan KS (Apache-2.0), labelled by me.
- The 256 word-piece limit was read from the loaded model (`max_seq_length`), not from the model card.
- The formulas are my own derivations, checked by the simulation shown above. Parent-child retrieval, late chunking and semantic chunking are named from memory.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/agents-rag Chunking:::Cutting documents into pieces before indexing them in a RAG system. The pieces are what gets searched and pasted into the prompt.

#flashcards/ml/concept #flashcards/topic/agents-rag Why does a RAG system cut documents into chunks instead of indexing whole documents?
?
A whole document is too long to embed well (the embedding model reads only a limited number of word pieces) and too long to paste into the prompt. Chunks make the search return only the relevant pieces.

#flashcards/ml/concept #flashcards/topic/agents-rag Name four ways to cut a document into chunks.
?
Fixed size, fixed size with overlap, by sentences or paragraphs, and by structure (headings). A fifth idea is to put the title and heading in front of each chunk.

#flashcards/ml/concept #flashcards/topic/agents-rag What is the trade-off in choosing a chunk size?
?
Small chunks are precise but lose context and make many vectors. Big chunks keep context but blur several ideas into one vector and fill the prompt with extra words.

#flashcards/ml/cloze #flashcards/topic/agents-rag Overlap makes sure a sentence cut by one ==boundary== is whole in the ==next window==, at the price of more chunks.

#flashcards/ml/concept #flashcards/topic/agents-rag all-MiniLM-L6-v2 reads at most how many word pieces, and what share of 164-word fixed chunks went over?
?
256 word pieces. 78% of the 164-word chunks were over the limit, 13% of the 98-word chunks and none of the 50-word chunks (measured in RAG Tuning Lab, on this vault's notes).

#flashcards/ml/formula #flashcards/topic/agents-rag Number of chunks for a document of $L$ words, window $s$, overlap $o$?
?
$$
n=\left\lceil \frac{L-o}{s-o} \right\rceil
$$
- $L$ is the document length in words. $s$ is the chunk size. $o$ is the overlap shared with the previous chunk, so $s-o$ is the step.
- $n$ is the number of chunks (rounded up). It sets how many vectors you embed and store. Halving $s$ roughly doubles $n$, and more overlap means more chunks.

#flashcards/ml/concept #flashcards/topic/agents-rag A note of 500 words, windows of 170 words: how many chunks with no overlap and with overlap 40?
?
No overlap: $\lceil 500/170\rceil=3$. Overlap 40: step 130, $\lceil 460/130\rceil=\lceil 3.54\rceil=4$. The overlap cost one extra chunk, a third more. Matched the real chunker.

#flashcards/ml/formula #flashcards/topic/agents-rag Chance that a span of $w$ words is cut by chunk boundaries?
?
$$
p_{\text{cut}}\approx\frac{\max(0,\;w-1-o)}{s-o}
$$
- $w$ is the span length in words. $s$ is the chunk size. $o$ is the overlap.
- $p_{\text{cut}}$ is the chance that no chunk holds the whole span. It reaches zero when the overlap is at least $w-1$, and rises as chunks get smaller.

#flashcards/ml/concept #flashcards/topic/agents-rag A 40-word sentence, 170-word windows: chance of being cut with overlap 0 and overlap 40?
?
Overlap 0: $39/170=0.229$ (simulation 0.225). Overlap 40: $\max(0,39-40)=0$, so never cut (simulation 0.000).

#flashcards/ml/image #flashcards/topic/agents-rag ![[rag-document-types.png]]
What does this picture say about chunking?
?
Documents differ in shape (a table, a lecture, a dialogue, documentation), so each needs a boundary that fits it. One cutting rule does not suit all. Figure by Numiri, Wikimedia Commons, CC BY-SA 4.0.

#flashcards/ml/concept #flashcards/topic/agents-rag What was the one clear effect in the chunking experiment, and how big was it?
?
Leaving the note title and heading off the front of each chunk. The blend's hit@5 on 99 outside questions fell from 0.93 to 0.86 (seven questions), mean reciprocal rank from 0.75 to 0.61, and keyword search alone from 0.83 to 0.68. My run, one corpus, small question sets.

#flashcards/ml/concept #flashcards/topic/agents-rag Why can a note-level test make whole-note chunks look as good as small chunks?
?
The test only asks whether the right note comes back, not whether the retrieved piece holds the answer. Whole notes won that test (0.98 and 0.92) yet are about 15 times longer per item (1,847 against 120 words, my arithmetic), a cost the test cannot see.

#flashcards/ml/concept #flashcards/topic/agents-rag What did the chunk-size comparison conclude?
?
No chunk size won. The ten settings with the title in front were within 0.04 of each other in hit@5, and one question is 0.01 to 0.02, so the differences are a handful of questions. The ranking is not trustworthy.

#flashcards/ml/concept #flashcards/topic/agents-rag Which chunking ideas did the note name but not test?
?
Parent-child retrieval, late chunking and semantic chunking (named from memory), and chunk-level labels, long-context models reading whole documents, and any language model reading the chunks.
