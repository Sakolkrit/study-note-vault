---
tags: [project, python, RAG, retrieval, chunking, reranking, embeddings, evaluation, sentence-transformers, cpu]
status: done
stack: Python 3.9, sentence-transformers, PyTorch (CPU), NumPy
goal: test, one change at a time, how chunking and reranking change retrieval quality on this vault's own notes, and report honestly where they did not help
---

# RAG Tuning Lab

> **In one line:** take the retrieval half of [[Vault Search with RAG]] and change one thing at a time, how the notes are cut into chunks ([[Chunking Strategies]]) and whether a cross-encoder re-orders the results ([[Reranking]]), then measure on 148 questions what each change really does.

## Goal
Done means five checks pass: (1) a frozen copy of the vault is the corpus, so the numbers can be reproduced, (2) eleven ways of cutting it are indexed and scored with keywords, meaning, and a blend of both, (3) a cross-encoder reranker is tested on top of each first stage at three shortlist sizes, (4) the two question sets (49 hand-written, 99 from a public list) are reported separately, and (5) the code in this note, extracted into a fresh folder, matches the code that produced the numbers.

## Result in short
- **Data:** 318 notes (a copy of this vault's concept, course and research notes, 19 November 2026, excluding `60-Projects`, templates and attachments), 49 questions I wrote and 99 question titles from the public repo *LLM Interview Questions and Answers Hub* (Kalyan KS, Apache-2.0), each with the notes that count as a correct answer. Scoring is at **note level**: take the best chunk of each note and check whether a correct note is in the top 5.
- **Chunking: no size or style clearly won.** Eleven settings gave hit@5 for the blended search between 0.96 and 1.00 on the 49 questions and 0.92 and 0.96 on the 99. The one clear effect was **leaving the note title and heading off the front of each chunk**: the blend fell from 0.93 to 0.86 on the 99 questions (seven questions). Full table in "Steps", and the lesson in [[Chunking Strategies]].
- **A scoring trap:** treating each whole note as one chunk scored as well as the others (0.98 and 0.92), because the test only asks whether the right *note* comes back. Those items are 15 times longer (1,847 words against 120). This test cannot see that cost.
- **Reranking: it helped the weak first stage and hurt the strong one.** Cross-encoder `ms-marco-MiniLM-L-6-v2` on keyword search: top-1 on the 99 outside questions went from 0.41 to 0.57 ($N=50$). On the keyword-and-meaning blend it **lowered** top-1 from 0.69 to 0.53 (49 questions) and from 0.63 to 0.57 (99). A bigger shortlist did not help, and blending the reranker's rank with the first stage's rank reduced the harm without beating the first stage. I do not know why. See [[Reranking]].
- **How much to trust it:** one corpus (the vault), one embedding model, one reranker, note-level labels written by me, and small question sets (one question is 2 points on 49 and 1 point on 99). Treat every gap of a few points as noise.
- **Not done:** no language model reads the chunks, no chunk-level labels, no tuning, no other rerankers, and reliable timing failed (the laptop was busy). See "What I did not run".

## What you need
Versions I used on Windows 11, in my normal Python (no new environment this time, because everything was already installed):

| Tool | Version |
|---|---|
| Python | 3.9.6 |
| sentence-transformers | 3.3.1 |
| torch | 2.8.0 (CPU build) |
| transformers | 4.46.3 |
| numpy | 1.24.4 |

Two models are downloaded from the Hugging Face hub on first use: the bi-encoder `sentence-transformers/all-MiniLM-L6-v2` (about 90 MB, already cached from [[Vault Search with RAG]]) and the cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` (a similar size; I did not check its licence). The work folder is `D:\python456\rag-tuning`, outside the vault. **Everything is slow on a CPU:** the eleven chunkings took about 50 minutes in total, and the reranking run about an hour and a half.

## Steps
1. **Freeze the corpus and fetch the questions.** Copy the vault's notes (skipping `.obsidian`, `90-Templates`, `99-Attachments`, `60-Projects` and the files in the vault's top folder) into a `corpus` folder, so later edits to the vault cannot change the numbers. Clone the public question list:
   ```
   git clone --depth 1 https://github.com/KalyanKS-NLP/LLM-Interview-Questions-and-Answers-Hub
   ```
   Then take three files from [[Vault Search with RAG]] unchanged: `search.py` (the keyword scorer, step 3 there), `questions.py` (the 49 questions, step 4) and `labels.py` (labels for the public questions, step 7). I extracted them with a short script from that note.

2. **The shared code.** Cleaning, the chunkers, an index that can rank by keywords, meaning or both, and the scoring of ranked notes.

   **`tune.py`**
   ```python
   """Shared pieces for the chunking and reranking experiments: chunkers, an index, and the scoring of ranked notes."""
   import glob, re, json
   from pathlib import Path
   import numpy as np
   from search import BM25                                  # keyword scorer from the Vault Search project
   from questions import QUESTIONS                          # 49 hand-written questions: (question, {correct notes})
   from labels import LABELS                                # labels for the 115 outside questions

   MODEL = "sentence-transformers/all-MiniLM-L6-v2"

   def clean(text):                                         # same cleaning as the Vault Search chunker
       text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S)
       text = re.sub(r"```.*?```", " ", text, flags=re.S)
       text = re.sub(r"!\[\[.*?\]\]", " ", text)
       return re.sub(r"\[\[([^\]|]+)(\|[^\]]+)?\]\]", r"\1", text)

   def sections(text):                                      # [(heading, body)] split at '##' headings
       out = []
       for part in re.split(r"\n(?=## )", clean(text)):
           head = part.splitlines()[0].lstrip("# ").strip() if part.startswith("#") else ""
           out.append((head, re.sub(r"\s+", " ", part).strip()))
       return out

   def windows(words, size, overlap):                       # cut a word list into windows; the last window is dropped if it has under 12 words
       step = max(1, size - overlap)
       return [" ".join(words[i:i + size]) for i in range(0, len(words), step) if len(words[i:i + size]) >= 12]

   def sentence_groups(body, target):                       # glue whole sentences together until about `target` words
       out, cur = [], []
       for s in re.split(r"(?<=[.!?])\s+", body):
           if cur and len(" ".join(cur + [s]).split()) > target:
               out.append(" ".join(cur)); cur = []
           cur.append(s)
       if cur: out.append(" ".join(cur))
       return [c for c in out if len(c.split()) >= 12]

   def make_chunker(kind, size=170, overlap=0):
       def chunker(text, title):
           out = []
           if kind == "fixed":                              # ignore headings: slide a window over the whole note
               body = " ".join(b for _, b in sections(text))
               out = [("", w) for w in windows(body.split(), size, overlap)]
           elif kind == "heading":                          # one chunk per '##' section, cut again if longer than `size` words
               for head, body in sections(text):
                   out += [(head, w) for w in windows(body.split(), size, overlap)]
           elif kind == "sentence":                         # sections cut at sentence ends
               for head, body in sections(text):
                   out += [(head, c) for c in sentence_groups(body, size)]
           return [{"note": title, "section": h, "text": t} for h, t in out]
       return chunker

   def build(chunker, root="corpus"):
       chunks = []
       for p in sorted(Path(root).rglob("*.md")):
           chunks += chunker(p.read_text(encoding="utf-8"), p.stem)
       return chunks

   class Index:
       def __init__(self, chunks, model, cache=None, prefix=True):
           self.chunks, self.model = chunks, model
           self.texts = [(f"{c['note']}. {c['section']}. " if prefix else "") + c["text"] for c in chunks]
           self.bm25 = BM25(self.texts)
           try: self.emb = np.load(cache)
           except (TypeError, FileNotFoundError):
               self.emb = model.encode(self.texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
               if cache: np.save(cache, self.emb)

       def candidates(self, query, method):                 # chunk indices, best first
           kw = self.bm25.scores(query) if method != "meaning" else None
           dn = self.emb @ self.model.encode([query], normalize_embeddings=True)[0] if method != "keyword" else None
           if method == "keyword": return list(np.argsort(-kw))
           if method == "meaning": return list(np.argsort(-dn))
           rk = {i: r for r, i in enumerate(np.argsort(-kw))}
           rd = {i: r for r, i in enumerate(np.argsort(-dn))}
           fused = {i: 1 / (60 + rk[i]) + 1 / (60 + rd[i]) for i in rk}             # reciprocal rank fusion
           return sorted(fused, key=fused.get, reverse=True)

       def note_order(self, chunk_order):                   # distinct notes, in the order their best chunk appears
           seen, out = set(), []
           for i in chunk_order:
               n = self.chunks[i]["note"]
               if n not in seen: seen.add(n); out.append(n)
           return out

   def outside_questions(repo="LLM-Interview-Questions-and-Answers-Hub"):
       q = {}
       for path in glob.glob(f"{repo}/Interview_QA/QA_*.md"):
           for m in re.finditer(r"^## 📌 Q(\d+):\s*(.+)$", open(path, encoding="utf-8").read(), re.M):
               q[int(m.group(1))] = m.group(2).strip()
       return [(q[i], LABELS[i]) for i in sorted(LABELS) if LABELS[i]]

   def score(ranked_notes_per_question, qset):             # hit@1, hit@3, hit@5, mean reciprocal rank (note level)
       h1 = h3 = h5 = mrr = 0.0
       for ranked, (_, good) in zip(ranked_notes_per_question, qset):
           pos = next((r for r, n in enumerate(ranked[:10]) if n in good), None)
           if pos is not None: h1 += pos < 1; h3 += pos < 3; h5 += pos < 5; mrr += 1 / (pos + 1)
       n = len(qset); return h1 / n, h3 / n, h5 / n, mrr / n
   ```

3. **A tiny demo and a check of the formulas** (used in [[Chunking Strategies]]).

   **`chunk_demo.py`**
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
   **`chunk_math.py`**
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
   Output (the chunk count formula matched the real chunker four times out of four, and the simulation agreed with the cutting-chance formula to within 0.02):
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

4. **Run the chunking experiment.** Eleven ways of cutting the same 318 notes, each embedded and searched three ways, on both question sets. A check at the top stops the run if an answer label is not a real note name. "cut%" is the share of chunks longer than the embedding model's limit of 256 word pieces.

   **`chunk_test.py`**
   ```python
   import time, numpy as np
   from sentence_transformers import SentenceTransformer
   from tune import make_chunker, build, Index, outside_questions, score, QUESTIONS, MODEL

   model = SentenceTransformer(MODEL)
   own, outside = QUESTIONS, outside_questions()
   names = {p.stem for p in __import__("pathlib").Path("corpus").rglob("*.md")}
   assert not {n for _, g in own + outside for n in g} - names, "an answer label is not a note name"
   print(f"{len(names)} notes | {len(own)} own questions | {len(outside)} outside questions | model max input: {model.max_seq_length} word pieces")

   CONFIGS = [                                              # (label, chunker, add the note title and heading in front of each chunk?)
       ("whole note, one chunk",        make_chunker("fixed", 10 ** 9),   True),
       ("fixed 300 words",              make_chunker("fixed", 300),       True),
       ("fixed 170 words",              make_chunker("fixed", 170),       True),
       ("fixed 100 words",              make_chunker("fixed", 100),       True),
       ("fixed 50 words",               make_chunker("fixed", 50),        True),
       ("fixed 170, overlap 40",        make_chunker("fixed", 170, 40),   True),
       ("by heading, whole section",    make_chunker("heading", 10 ** 9), True),
       ("by heading, max 170 (used before)", make_chunker("heading", 170), True),
       ("by heading, max 100",          make_chunker("heading", 100),     True),
       ("by sentences, about 100",      make_chunker("sentence", 100),    True),
       ("by heading, max 170, no title in front", make_chunker("heading", 170), False),
   ]
   rows = []
   for k, (label, chunker, prefix) in enumerate(CONFIGS):
       t0 = time.time(); chunks = build(chunker)
       idx = Index(chunks, model, cache=f"emb_{k}.npy", prefix=prefix)
       cut = np.mean([len(model.tokenizer(t)["input_ids"]) > model.max_seq_length for t in idx.texts])   # share of chunks the model cannot read to the end
       res = {}
       for method in ("keyword", "meaning", "hybrid"):
           for tag, qset in (("own", own), ("out", outside)):
               res[method, tag] = score([idx.note_order(idx.candidates(q, method)) for q, _ in qset], qset)
       rows.append((label, len(chunks), np.mean([len(c["text"].split()) for c in chunks]), cut, res))
       print(f"[{k}] {label}: {len(chunks)} chunks, {time.time() - t0:.0f}s", flush=True)

   print("\nhit@5 (share of questions with a correct note in the top 5 notes) | mean reciprocal rank in brackets")
   print(f"{'chunking':38s} {'chunks':>6s} {'words':>5s} {'cut%':>5s} | " + " | ".join(f"{m[:4]:>4s} {t}" + " " * 7 for m in ("keyword", "meaning", "hybrid") for t in ("own", "out")))
   for label, n, w, cut, res in rows:
       cells = " | ".join(f"{res[m, t][2]:.2f} ({res[m, t][3]:.2f})" for m in ("keyword", "meaning", "hybrid") for t in ("own", "out"))
       print(f"{label:38s} {n:6d} {w:5.0f} {cut:5.0%} | {cells}")
   ```
   Run `python chunk_test.py`. Output (the library's warnings are removed; hit@5 with the mean reciprocal rank in brackets; "own" is the 49 hand-written questions, "out" the 99 public ones):
   ```text
   hit@5 (share of questions with a correct note in the top 5 notes) | mean reciprocal rank in brackets
   chunking                               chunks words  cut% | keyw own        | keyw out        | mean own        | mean out        | hybr own        | hybr out       
   whole note, one chunk                     317  1847   99% | 0.96 (0.78) | 0.80 (0.58) | 0.88 (0.66) | 0.78 (0.55) | 0.98 (0.84) | 0.92 (0.65)
   fixed 300 words                          2102   279   94% | 0.90 (0.72) | 0.89 (0.64) | 0.90 (0.73) | 0.88 (0.70) | 0.98 (0.81) | 0.96 (0.66)
   fixed 170 words                          3577   164   78% | 0.94 (0.71) | 0.84 (0.63) | 0.84 (0.73) | 0.94 (0.69) | 0.96 (0.76) | 0.94 (0.72)
   fixed 100 words                          5976    98   13% | 0.88 (0.70) | 0.85 (0.63) | 0.96 (0.73) | 0.93 (0.74) | 0.96 (0.76) | 0.93 (0.73)
   fixed 50 words                          11795    50    0% | 0.88 (0.64) | 0.87 (0.64) | 0.92 (0.73) | 0.95 (0.79) | 0.98 (0.79) | 0.94 (0.74)
   fixed 170, overlap 40                    4644   163   78% | 0.86 (0.69) | 0.84 (0.60) | 0.92 (0.68) | 0.91 (0.71) | 0.98 (0.77) | 0.95 (0.75)
   by heading, whole section                2548   230   45% | 0.94 (0.73) | 0.79 (0.53) | 0.90 (0.71) | 0.91 (0.75) | 0.98 (0.81) | 0.93 (0.68)
   by heading, max 170 (used before)        4866   120   49% | 0.90 (0.71) | 0.83 (0.58) | 0.90 (0.72) | 0.92 (0.77) | 1.00 (0.82) | 0.93 (0.75)
   by heading, max 100                      7021    83   12% | 0.92 (0.71) | 0.81 (0.58) | 0.94 (0.66) | 0.93 (0.78) | 0.98 (0.75) | 0.95 (0.77)
   by sentences, about 100                  7590    77    8% | 0.92 (0.72) | 0.82 (0.55) | 0.94 (0.71) | 0.94 (0.78) | 0.98 (0.79) | 0.92 (0.74)
   by heading, max 170, no title in front   4866   120   46% | 0.88 (0.69) | 0.68 (0.46) | 0.94 (0.70) | 0.83 (0.65) | 1.00 (0.77) | 0.86 (0.61)
   ```
   **What to look for:** the "hybr" columns are the blended search. The ten rows with the title in front are all within 0.04 of each other. The last row, with no title in front, is the only one that falls (0.86 on "hybr out"). The "whole note" row looks fine because of the note-level test, not because it is a good way to cut.

5. **Run the reranking experiment.** The chunking from the earlier project (cut by heading, at most 170 words), three first stages, three shortlist sizes, the cross-encoder re-scoring the shortlist and the notes re-ordered by their best chunk. The column "candidates holding a correct note" is the ceiling from [[Reranking]]: the share of questions where a correct note is in the shortlist at all.

   **`rerank_test.py`**
   ```python
   import time, numpy as np
   from sentence_transformers import SentenceTransformer, CrossEncoder
   from tune import make_chunker, build, Index, outside_questions, score, QUESTIONS, MODEL

   bi = SentenceTransformer(MODEL)
   ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")           # reads question and chunk together, outputs one relevance score
   idx = Index(build(make_chunker("heading", 170)), bi, cache="emb_h170.npy")   # same chunking as the Vault Search project
   sets = {"own": QUESTIONS, "out": outside_questions()}
   NS = (10, 20, 50)
   print(len(idx.chunks), "chunks |", {k: len(v) for k, v in sets.items()}, "questions")

   def rerank(query, cand):                                 # cross-encoder score for each candidate chunk, best first
       s = ce.predict([(query, idx.texts[i]) for i in cand], batch_size=32, show_progress_bar=False)
       return [cand[j] for j in np.argsort(-s)]

   rows = []
   for stage in ("keyword", "meaning", "hybrid"):
       for tag, qset in sets.items():
           base = [idx.candidates(q, stage) for q, _ in qset]
           rows.append((f"{stage}, no rerank", tag, score([idx.note_order(c) for c in base], qset), None))
           for N in NS:
               ceiling = np.mean([any(n in good for n in idx.note_order(c[:N])) for c, (_, good) in zip(base, qset)])   # a correct note is somewhere in the N candidates
               ranked = [idx.note_order(rerank(q, c[:N])) for c, (q, _) in zip(base, qset)]
               rows.append((f"{stage} top {N} -> rerank", tag, score(ranked, qset), ceiling))
       print("done", stage, flush=True)

   print("\n" + f"{'pipeline':28s} {'set':4s} {'top1':>5s} {'top3':>5s} {'top5':>5s} {'MRR':>6s}  candidates holding a correct note")
   for label, tag, (h1, h3, h5, mrr), ceil in rows:
       print(f"{label:28s} {tag:4s} {h1:5.2f} {h3:5.2f} {h5:5.2f} {mrr:6.3f}  " + ("" if ceil is None else f"{ceil:.2f}"))

   q = [q for q, _ in QUESTIONS]                            # timing, one question at a time, after one warm-up call
   idx.candidates(q[0], "hybrid"); rerank(q[0], idx.candidates(q[0], "hybrid")[:20])
   t = time.time(); cands = [idx.candidates(x, "hybrid") for x in q]; t_first = (time.time() - t) / len(q)
   for N in (20, 50):
       t = time.time(); [rerank(x, c[:N]) for x, c in zip(q, cands)]; print(f"time per question: first stage (hybrid) {t_first * 1000:.0f} ms | cross-encoder on {N} chunks {(time.time() - t) / len(q) * 1000:.0f} ms")
   ```
   Run `python rerank_test.py`. Output:
   ```text
   pipeline                     set   top1  top3  top5    MRR  candidates holding a correct note
   keyword, no rerank           own   0.53  0.86  0.90  0.709  
   keyword top 10 -> rerank     own   0.49  0.92  0.98  0.694  0.98
   keyword top 20 -> rerank     own   0.51  0.88  0.96  0.699  0.98
   keyword top 50 -> rerank     own   0.49  0.88  0.96  0.695  1.00
   keyword, no rerank           out   0.41  0.71  0.83  0.577  
   keyword top 10 -> rerank     out   0.56  0.77  0.90  0.684  0.93
   keyword top 20 -> rerank     out   0.54  0.76  0.88  0.671  0.97
   keyword top 50 -> rerank     out   0.57  0.81  0.92  0.710  1.00
   meaning, no rerank           own   0.57  0.84  0.90  0.717  
   meaning top 10 -> rerank     own   0.61  0.90  0.94  0.754  0.94
   meaning top 20 -> rerank     own   0.57  0.88  0.96  0.732  0.98
   meaning top 50 -> rerank     own   0.53  0.88  0.96  0.700  0.98
   meaning, no rerank           out   0.67  0.85  0.92  0.771  
   meaning top 10 -> rerank     out   0.60  0.89  0.91  0.733  0.92
   meaning top 20 -> rerank     out   0.54  0.87  0.91  0.706  0.97
   meaning top 50 -> rerank     out   0.54  0.83  0.91  0.691  1.00
   hybrid, no rerank            own   0.69  0.96  1.00  0.822  
   hybrid top 10 -> rerank      own   0.51  0.90  0.94  0.707  0.96
   hybrid top 20 -> rerank      own   0.53  0.92  0.96  0.723  1.00
   hybrid top 50 -> rerank      own   0.51  0.90  0.94  0.699  1.00
   hybrid, no rerank            out   0.63  0.86  0.93  0.754  
   hybrid top 10 -> rerank      out   0.60  0.85  0.88  0.717  0.91
   hybrid top 20 -> rerank      out   0.57  0.86  0.93  0.721  0.99
   hybrid top 50 -> rerank      out   0.57  0.78  0.90  0.706  1.00
   time per question: first stage (hybrid) 36 ms | cross-encoder on 20 chunks 2492 ms
   time per question: first stage (hybrid) 36 ms | cross-encoder on 50 chunks 22028 ms
   ```
   **What to look for:** the "keyword" rows on "out" rise after reranking (top-1 0.41 to 0.57); the "hybrid" rows fall at top-1 (0.69 to 0.51 on "own"). The last column is nearly 1 for $N=20$ and $N=50$, so the shortlist was not the problem.

6. **Try the obvious fixes** for the blend: mix the reranker's rank with the first-stage rank, and drop the title from the reranker's input.

   **`fuse_test.py`**
   ```python
   """Does the cross-encoder help when its score is blended with the first-stage rank, or when it does not see the title? (hybrid, top 20)"""
   import numpy as np
   from sentence_transformers import SentenceTransformer, CrossEncoder
   from tune import make_chunker, build, Index, outside_questions, score, QUESTIONS, MODEL

   bi = SentenceTransformer(MODEL); ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
   idx = Index(build(make_chunker("heading", 170)), bi, cache="emb_h170.npy")
   sets = {"own": QUESTIONS, "out": outside_questions()}
   N = 20
   plain = [c["text"] for c in idx.chunks]                  # chunk text without the 'Note. Section.' prefix

   def ce_scores(q, cand, texts): return ce.predict([(q, texts[i]) for i in cand], batch_size=32, show_progress_bar=False)

   def order(cand, s): return [cand[j] for j in np.argsort(-s)]

   def fused(cand, s):                                      # reciprocal rank fusion of the first-stage rank and the cross-encoder rank
       ce_rank = {cand[j]: r for r, j in enumerate(np.argsort(-s))}
       f = {c: 1 / (60 + r) + 1 / (60 + ce_rank[c]) for r, c in enumerate(cand)}
       return sorted(cand, key=f.get, reverse=True)

   print(f"{'pipeline (hybrid top 20)':34s} {'set':4s} {'top1':>5s} {'top3':>5s} {'top5':>5s} {'MRR':>6s}")
   for tag, qset in sets.items():
       runs = {"first stage only": [], "cross-encoder only": [], "blend of both ranks": [], "cross-encoder, no title": []}
       for q, _ in qset:
           cand = idx.candidates(q, "hybrid")[:N]
           s = ce_scores(q, cand, idx.texts)
           runs["first stage only"].append(cand)
           runs["cross-encoder only"].append(order(cand, s))
           runs["blend of both ranks"].append(fused(cand, s))
           runs["cross-encoder, no title"].append(order(cand, ce_scores(q, cand, plain)))
       for name, lists in runs.items():
           h1, h3, h5, mrr = score([idx.note_order(c) for c in lists], qset)
           print(f"{name:34s} {tag:4s} {h1:5.2f} {h3:5.2f} {h5:5.2f} {mrr:6.3f}", flush=True)
   ```
   Output:
   ```text
   pipeline (hybrid top 20)           set   top1  top3  top5    MRR
   first stage only                   own   0.69  0.96  1.00  0.822
   cross-encoder only                 own   0.53  0.92  0.96  0.723
   blend of both ranks                own   0.67  0.92  0.98  0.800
   cross-encoder, no title            own   0.53  0.90  0.96  0.723
   first stage only                   out   0.63  0.86  0.93  0.754
   cross-encoder only                 out   0.57  0.86  0.93  0.721
   blend of both ranks                out   0.60  0.87  0.90  0.743
   cross-encoder, no title            out   0.51  0.75  0.91  0.660
   ```
   **What to look for:** "blend of both ranks" sits between the reranker alone and the first stage alone, never above the first stage. "cross-encoder, no title" is the worst on "out" (MRR 0.660).

7. **A toy to see the two scores** (used in [[Reranking]]). A chosen example of five passages: both models put the answer first.

   **`rerank_demo.py`**
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

8. **Try to time it.** One question at a time, after a warm-up call.

   **`timing.py`**
   ```python
   import time, numpy as np
   from sentence_transformers import SentenceTransformer, CrossEncoder
   from tune import make_chunker, build, Index, QUESTIONS, MODEL
   bi = SentenceTransformer(MODEL); ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
   idx = Index(build(make_chunker("heading", 170)), bi, cache="emb_h170.npy")
   qs = [q for q, _ in QUESTIONS[:15]]; cands = [idx.candidates(q, "hybrid") for q in qs]
   ce.predict([("warm up", idx.texts[0])])                                   # warm-up call
   for N in (10, 20, 50, 20):
       t = time.perf_counter(); [ce.predict([(q, idx.texts[i]) for i in c[:N]], batch_size=32, show_progress_bar=False) for q, c in zip(qs, cands)]
       print(f"cross-encoder, {N:2d} chunks per question: {(time.perf_counter() - t) / len(qs) * 1000:6.0f} ms per question")
   t = time.perf_counter(); [idx.candidates(q, "hybrid") for q in qs]; print(f"first stage (hybrid, 4,866 chunks): {(time.perf_counter() - t) / len(qs) * 1000:.0f} ms per question")
   t = time.perf_counter(); bi.encode(idx.texts[:500], batch_size=64, show_progress_bar=False); print(f"bi-encoder, one chunk: {(time.perf_counter() - t) / 500 * 1000:.1f} ms (done once per chunk, at index time)")
   ```
   Output (this run, while the laptop was busy):
   ```text
   cross-encoder, 10 chunks per question:   5091 ms per question
   cross-encoder, 20 chunks per question:  10528 ms per question
   cross-encoder, 50 chunks per question:  22584 ms per question
   cross-encoder, 20 chunks per question:   9832 ms per question
   first stage (hybrid, 4,866 chunks): 1169 ms per question
   bi-encoder, one chunk: 222.2 ms (done once per chunk, at index time)
   ```
   **Do not trust these.** An earlier run inside the reranking experiment measured the first stage at 36 ms per question and the cross-encoder on 20 chunks at 2,492 ms, and on 50 chunks at 22,028 ms. This run gave 1,169 ms for the first stage and 10,528 ms for 20 chunks. The numbers differ by up to 30 times between runs, and the bi-encoder took 222 ms per chunk, far slower than usual for this model, so something else was using the CPU. I can say only that the cross-encoder cost **seconds per question** here, much more than the first stage in the first run, and I cannot give a reliable ratio.

## What went wrong, and the fix
- **The corpus drifted.** [[Vault Search with RAG]] used 247 notes, and the vault now has 318 (the time-series, LoRA and paper notes were added). The numbers are not comparable with that project's, so I froze a copy this time.
- **`python -I` hid a sibling module** in another project, and here the scripts import each other (`tune`, `search`), so they are run without `-I`.
- **The first reranking run was very slow** (about 25 minutes per first stage), because it scored the shortlists for $N=10$, $20$ and $50$ separately. I did not restart it, since the results are deterministic, but a version that scores the top 50 once and slices it would cost about 40% less.
- **The timing run was unreliable** (see step 8) and I could not fix it without a quiet machine.
- **Reranking made the strong first stage worse.** I first suspected a bug (a scoring or ordering error), so I checked the model on a toy where it ranks the answer first by a wide margin (step 7), and the fusion run (step 6) reproduced the same numbers as the main run for the same settings (0.53, 0.92, 0.96, 0.723 on my questions), which makes a random error unlikely. I did not find the cause.

## What I did not run
- **No language model read the chunks.** Everything measures retrieval only, so I cannot say whether a better order gives a better answer.
- **Note-level labels only.** A correct chunk is not labelled, so the test cannot tell a chunk that holds the answer from another chunk of the same note, and it favours big chunks.
- **Small question sets, labels written by me.** The 99 public questions are labelled by me against this vault, and 16 of the repo's 115 questions have no note and were skipped.
- **One corpus and two models.** No other embedding model, no other reranker, no other language of text, no tuning of chunk sizes, no semantic or parent-child chunking.
- **Timing:** not measured reliably (step 8).
- **The long runs were not repeated end to end.** After writing, I extracted every code block from this note and checked that each file is byte-for-byte the file that produced the numbers, then re-ran only the quick ones (steps 3 and 7) from the extracted copies. The chunking and reranking runs themselves took about 50 minutes and an hour and a half, and I did not run them twice.
- **Obsidian:** the notes were not opened in Obsidian, and the flashcards were not reviewed in the plugin.

## Concepts used
- [[RAG]] · [[Chunking Strategies]] · [[Reranking]] · [[Contextual Retrieval]] · [[Embedding Model]] · [[Embedding]] · [[Vector Database]] · [[BM25]] · [[Cosine Similarity]] · [[Retrieval Metrics]] · [[BERT]] · [[Transformer]] · [[Context Window]] · [[Tokenization]]

## Next ideas
- Label the exact chunk that answers each question, then re-run the chunking comparison at chunk level.
- Try a different reranker (a larger cross-encoder) and a reranker trained on my own question-and-note pairs.
- Test parent-child retrieval (search small chunks, return the section) and contextual retrieval (a model writes the context for each chunk).
- Score the top-1 version of the cross-encoder only when the first stage is unsure.
- Measure timing again on an idle machine.
- Let a language model answer from the retrieved chunks and grade it ([[LLM as a Judge]]).

## Flashcards
#flashcards/projects

#flashcards/projects/reverse #flashcards/topic/project-rag RAG Tuning Lab (this project):::Eleven chunkings and a cross-encoder reranker tested one at a time on 318 of this vault's notes, with 49 hand-written and 99 public questions, scored at note level.

#flashcards/projects/concept #flashcards/topic/project-rag What did the chunking part of the lab find?
?
No chunk size or style clearly won: hit@5 for the blend stayed within 0.04 across ten settings. The one clear effect was removing the title and heading from the front of each chunk, which cut the blend from 0.93 to 0.86 on the 99 public questions.

#flashcards/projects/concept #flashcards/topic/project-rag What did the reranking part of the lab find?
?
The cross-encoder helped keyword search (top-1 on 99 questions 0.41 to 0.57) and hurt the blend of keywords and meaning (top-1 0.69 to 0.53 on 49 questions, 0.63 to 0.57 on 99). A bigger shortlist or rank blending did not rescue it. The cause was not found.

#flashcards/projects/concept #flashcards/topic/project-rag Why did the whole-note chunking look as good as the small chunks?
?
The test scores whether the right note comes back, not whether the retrieved piece holds the answer, so big chunks are not penalised. Each whole note is 15 times longer than a heading chunk (1,847 against 120 words, my arithmetic), which this test cannot see.

#flashcards/projects/concept #flashcards/topic/project-rag Why was the corpus copied before the experiments?
?
The vault keeps changing (247 notes in the earlier project, 318 now), so a frozen copy makes the numbers reproducible and keeps new notes about chunking from answering the questions.

#flashcards/projects/concept #flashcards/topic/project-rag How trustworthy was the timing in the lab?
?
Not at all. The first-stage and reranker times changed by up to 30 times between two runs (cross-encoder on 20 chunks 2,492 ms against 10,528 ms), because the laptop was busy. Only "seconds per question for the cross-encoder" can be said.

#flashcards/projects/concept #flashcards/topic/project-rag What did the lab not run?
?
A language model reading the chunks, chunk-level labels, other embedding models and rerankers, tuning, semantic or parent-child chunking, reliable timing, and a second end-to-end run of the long experiments. Not opened in Obsidian either.
