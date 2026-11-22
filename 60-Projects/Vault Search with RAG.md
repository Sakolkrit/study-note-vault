---
tags: [project, python, RAG, retrieval, embeddings, evaluation, LLM]
status: built-retrieval-only
stack: Python 3.9, sentence-transformers, NumPy, pytest
goal: search this vault's own notes by meaning and by keyword, and measure how often the right note comes back
---

# Vault Search with RAG

> **In one line:** build the retrieval half of a [[RAG]] system over the notes in this vault (cut notes into chunks, score them by keywords and by meaning, blend the two) and, instead of trusting it, **measure** how often it finds the right note for 49 questions. The "generate" half is only prepared, not run.

## Goal
Done means five checks pass: (1) every note is cut into small chunks, (2) a keyword scorer ([[BM25]]) and a meaning scorer ([[Embedding Model]]) both rank the chunks, (3) a blended ranking exists, (4) the three rankings are compared on a fixed list of questions with [[Retrieval Metrics]] and a random baseline, and (5) a script builds the prompt that a chat model would receive.

## Result in short
- **Index:** 247 notes became 3,009 chunks of at most 170 words (code blocks, images, frontmatter and the `60-Projects` logs removed). Cutting the notes into chunks took half a second. Running the evaluation, which includes turning all the chunks into vectors on a laptop CPU, took about 2.5 to 3 minutes in two timed runs (the first run also downloads the model once).
- **Test:** 49 questions, each with a short list of notes that count as a correct answer. Five random notes would contain a correct one only 3.0% of the time.

| Method | correct note in top 1 | top 3 | top 5 | mean reciprocal rank |
|---|---|---|---|---|
| keywords only (BM25) | 0.61 | 0.90 | 0.94 | 0.756 |
| meaning only (embeddings) | 0.65 | 0.88 | 0.94 | 0.772 |
| blended (reciprocal rank fusion) | 0.71 | 0.94 | 0.98 | 0.823 |

- **How much to trust that:** 49 questions is small. One question is 2 percentage points, so "0.98 against 0.94" at top 5 is a difference of **two questions**. The blend is best or tied in every column, while keywords and meaning trade places, which is some evidence that blending helps, but not proof. The questions were also written by me, the person who wrote the notes, so the scores are probably **optimistic** compared with questions you would type.
- **One real failure:** "How do language models avoid recomputing keys and values for earlier words?" never put [[KV Cache]] in the top 5 (the correct note ranked 18th in the blend). I did not look into why; the question's wording may simply be far from the words the note uses.
- **A second test with questions I did not write.** I also scored 99 questions from the public repo *LLM Interview Questions and Answers Hub* (by Kalyan KS, Apache-2.0 licence), using only its question texts. 16 of its 115 questions have no note in this vault (for example continuous batching, gradient accumulation, latency), so they are skipped and counted. The answer labels are still mine, written before I saw any result.

| Method (99 outside questions) | top 1 | top 3 | top 5 | mean reciprocal rank |
|---|---|---|---|---|
| keywords only | 0.41 | 0.73 | 0.81 | 0.586 |
| meaning only | 0.72 | 0.89 | 0.93 | 0.807 |
| blended | 0.69 | 0.91 | 0.95 | 0.800 |

  Different wording hurts keyword search a lot (0.94 down to 0.81 at top 5), while meaning search barely moves (0.94 to 0.93). The blend is still best at top 3 and top 5, but **meaning alone is as good or better at top 1 and in mean reciprocal rank**, so this set does not support "blending always wins". Five blended misses: the position-wise feed-forward layer, autoregressive generation (two questions), diffusion versus normal language models, and speeding up fine-tuning.
- **Not done:** no language model reads the retrieved chunks. `ask.py` builds the prompt and stops (see "What I did not run").

## What you need
Versions I used (a fresh virtual environment, Windows, Python 3.9.6):

| Tool | Version |
|---|---|
| sentence-transformers | 3.3.1 |
| torch | 2.8.0 |
| transformers | 4.57.6 |
| numpy | 1.26.4 |
| pytest | 8.4.2 |

The embedding model is `sentence-transformers/all-MiniLM-L6-v2` (384-dimension vectors), downloaded automatically the first time you run it (about 90 MB; internet needed once). It is a small general-purpose English model. I did not check its licence.

## Steps
1. **Make a folder and a clean environment** ([[Virtual Environments]], [[Command Line]]). Installing the libraries downloads PyTorch, which is large and took several minutes here.
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install sentence-transformers==3.3.1 "numpy<2" pytest
   ```

2. **Cut the notes into chunks**. Each note is split at its `##` headings, then into pieces of at most 170 words. A chunk remembers its note and section. Code blocks, images and the frontmatter are removed, and `[[links]]` lose their brackets so the words stay searchable. Do not name this file `chunk.py`: Python already has a module of that name.

   **`chunker.py`**
   ```python
   import json
   import re
   import sys
   from pathlib import Path

   SKIP = {".obsidian", "source-repo", "90-Templates", "99-Attachments", "60-Projects", ".git"}   # project logs quote the questions, so they would answer themselves


   def split_note(text, title, words=170):
       """Cut one note into chunks at its '##' headings, then into pieces of at most `words` words."""
       text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.S)          # drop the frontmatter
       text = re.sub(r"```.*?```", " ", text, flags=re.S)                 # drop code blocks
       text = re.sub(r"!\[\[.*?\]\]", " ", text)                          # drop image embeds
       text = re.sub(r"\[\[([^\]|]+)(\|[^\]]+)?\]\]", r"\1", text)        # [[Link]] -> Link
       chunks = []
       for part in re.split(r"\n(?=## )", text):
           head = part.splitlines()[0].lstrip("# ").strip() if part.startswith("#") else ""
           body = re.sub(r"\s+", " ", part).strip()
           w = body.split()
           for i in range(0, len(w), words):
               piece = " ".join(w[i:i + words])
               if len(piece.split()) >= 12:
                   chunks.append({"note": title, "section": head, "text": piece})
       return chunks


   def build(root):
       out = []
       for p in sorted(Path(root).rglob("*.md")):
           if SKIP & set(p.relative_to(root).parts):
               continue
           out += split_note(p.read_text(encoding="utf-8"), p.stem)
       return out


   if __name__ == "__main__":
       chunks = build(sys.argv[1])
       json.dump(chunks, open("chunks.json", "w", encoding="utf-8"))
       print(len({c["note"] for c in chunks}), "notes ->", len(chunks), "chunks")
   ```
   Run `python chunker.py "path\to\your\vault"`. I saw `247 notes -> 3009 chunks`. The counts move whenever the vault changes. The `60-Projects` folder is skipped on purpose (see "What went wrong").

3. **Score the chunks two ways and blend them** ([[BM25]], [[Embedding]], [[Cosine Similarity]], [[Vector Database]]). Keyword scoring (BM25) rewards rare words that appear in a chunk and does not favour long chunks. Meaning scoring turns every chunk and the question into a vector and compares them by angle (the vectors are length-1, so a dot product is the cosine). The blend uses **reciprocal rank fusion**: each chunk gets $1/(60+\text{rank})$ from each list, and the two are added. A vector database is not used: 3,034 vectors fit in memory and one matrix product searches them all.

   **`search.py`**
   ```python
   import json
   import math
   import re
   from collections import Counter

   import numpy as np

   STOP = set("the a an of to in is it and or for on with as at by be this that are was what how why does do can i my you your from when which".split())


   def tokens(s):
       return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP and len(t) > 1]


   class BM25:
       """Keyword scoring: rare words count more, repeats count less, long chunks are not favoured."""

       def __init__(self, docs, k1=1.5, b=0.75):
           self.k1, self.b = k1, b
           self.tf = [Counter(tokens(d)) for d in docs]
           self.len = np.array([sum(t.values()) for t in self.tf], dtype=float)
           df = Counter(w for t in self.tf for w in t)
           n = len(docs)
           self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
           self.avg = self.len.mean()

       def scores(self, query):
           s = np.zeros(len(self.tf))
           for w in set(tokens(query)):
               if w not in self.idf:
                   continue
               for i, t in enumerate(self.tf):
                   f = t.get(w, 0)
                   if f:
                       s[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
           return s


   class Index:
       def __init__(self, chunks_path="chunks.json", emb_path="emb.npy", model="sentence-transformers/all-MiniLM-L6-v2"):
           self.chunks = json.load(open(chunks_path, encoding="utf-8"))
           self.texts = [f"{c['note']}. {c['section']}. {c['text']}" for c in self.chunks]
           self.bm25 = BM25(self.texts)
           from sentence_transformers import SentenceTransformer
           self.model = SentenceTransformer(model)
           try:
               self.emb = np.load(emb_path)
           except FileNotFoundError:
               self.emb = self.model.encode(self.texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
               np.save(emb_path, self.emb)

       def rank(self, query, method="hybrid"):
           """Return chunk indices, best first."""
           kw = self.bm25.scores(query)
           dn = self.emb @ self.model.encode([query], normalize_embeddings=True)[0]
           if method == "keyword":
               return list(np.argsort(-kw))
           if method == "meaning":
               return list(np.argsort(-dn))
           rk = {i: r for r, i in enumerate(np.argsort(-kw))}
           rd = {i: r for r, i in enumerate(np.argsort(-dn))}
           fused = {i: 1 / (60 + rk[i]) + 1 / (60 + rd[i]) for i in rk}      # reciprocal rank fusion
           return sorted(fused, key=fused.get, reverse=True)

       def notes(self, query, method="hybrid", k=5):
           """Top-k distinct notes, each with its best-matching chunk."""
           seen, out = set(), []
           for i in self.rank(query, method):
               n = self.chunks[i]["note"]
               if n not in seen:
                   seen.add(n); out.append((n, i))
               if len(out) == k:
                   break
           return out
   ```

4. **Write the questions and measure** ([[Retrieval Metrics]]). Each question lists the notes that count as correct. The test is at the note level: take the best chunk of each note, keep the first five notes, and see whether a correct one is there. A check at the top stops the run if an answer label is not a real note name.

   **`questions.py`**
   ```python
   # (question, notes that count as a correct answer). Written by hand from what each note is about.
   QUESTIONS = [
       ("Why does my loss turn into nan when the scores are around 1000?", {"Softmax"}),
       ("What does it mean when the cost goes up and down between steps?", {"Learning Rate", "Gradient Descent"}),
       ("How can a straight-line model fit a curve?", {"Feature Engineering"}),
       ("Which method sorts with far fewer comparisons than bubble sort?", {"Sorting"}),
       ("How do I find a value in a sorted list by halving the range?", {"Binary Search"}),
       ("Why use a container instead of a virtual machine?", {"Docker"}),
       ("When does logistic regression say class 1?", {"Decision Boundary", "Logistic Regression"}),
       ("How do language models avoid recomputing keys and values for earlier words?", {"KV Cache"}),
       ("Why can random starting centres give a bad grouping?", {"K-Means"}),
       ("Which clustering method finds dense regions and labels the rest as noise?", {"DBSCAN"}),
       ("How do I shrink thousands of features while keeping most of the variance?", {"PCA"}),
       ("How is a model trained from people ranking two answers?", {"Bradley-Terry Model", "RLHF", "Reward Model", "Direct Preference Optimization"}),
       ("How can I fine-tune a huge model when the weights are stored in 4 bits?", {"QLoRA"}),
       ("How do I test an API without starting a server?", {"pytest", "FastAPI"}),
       ("What is the dot product divided by the lengths used for?", {"Cosine Similarity"}),
       ("How does a lookup table find items in nearly constant time?", {"Hash Tables"}),
       ("What goes wrong with naive recursion for Fibonacci numbers?", {"Dynamic Programming", "Recursion"}),
       ("How do I stop a model from memorising by punishing big weights?", {"Regularization"}),
       ("Which curve turns any score into a chance between 0 and 1?", {"Logistic Regression"}),
       ("Why does attention use many heads?", {"Attention", "Self-Attention", "Transformer"}),
       ("How are word vectors learned by predicting neighbouring words?", {"Word2Vec"}),
       ("How do I measure whether a search system ranks the right documents first?", {"Retrieval Metrics"}),
       ("How do I combine keyword search with embeddings for retrieval?", {"Contextual Retrieval", "BM25", "RAG", "Vector Database"}),
       ("What are the four counts in a table of right and wrong predictions?", {"Confusion Matrix"}),
       ("What share of the real positives does my classifier catch?", {"Recall"}),
       ("Why is accuracy misleading when only a few cases are positive?", {"Accuracy", "Evaluation Metrics", "F1 Score", "Precision", "Recall"}),
       ("How do I keep a decision tree from memorising the training data?", {"Decision Tree", "Overfitting"}),
       ("Why do many trees voting beat one tree?", {"Random Forest", "Ensemble Methods"}),
       ("How are gradients computed backwards through the layers?", {"Backpropagation"}),
       ("How do a generator and a discriminator compete?", {"GAN"}),
       ("Which layer keeps only the biggest number in each patch of an image?", {"Pooling", "Convolutional Neural Network"}),
       ("How does a sliding filter detect edges?", {"Convolution", "Convolutional Neural Network"}),
       ("Which algorithm finds the shortest path by always expanding the closest unvisited node?", {"Dijkstra's Algorithm"}),
       ("Why can picking the biggest coin each time fail?", {"Greedy Algorithms"}),
       ("How do I solve eight queens by placing pieces and undoing mistakes?", {"Backtracking"}),
       ("Which structure gives back the newest item first?", {"Stacks and Queues"}),
       ("Can another language model grade answers for me?", {"LLM as a Judge"}),
       ("How do I train faster using 16-bit numbers?", {"Mixed Precision Training"}),
       ("How can a small model draft tokens for a big one to check?", {"Speculative Decoding"}),
       ("Why does a model's loss fall smoothly as compute and data grow?", {"Scaling Laws"}),
       ("How should I split data into train, validation and test sets?", {"Cross-Validation"}),
       ("How does temperature change which word a model picks next?", {"Decoding Strategies", "Softmax"}),
       ("How do I turn a scikit-learn model into a web page with a prediction?", {"Streamlit", "FastAPI"}),
       ("What is a good score for how surprised a language model is by text?", {"Perplexity"}),
       ("How do I get a computer to learn by trial and reward?", {"Reinforcement Learning", "Q-Learning"}),
       ("How can I make a straight line classifier use curved boundaries?", {"Decision Boundary", "Feature Engineering"}),
       ("What makes training stable when features have very different ranges?", {"Feature Scaling", "Normalization"}),
       ("How do I turn category names into numbers for a model?", {"One-Hot Encoding"}),
       ("How do I find out if more data would help my model?", {"Learning Curve", "Bias-Variance"}),
   ]
   ```
   **`evaluate.py`**
   ```python
   from questions import QUESTIONS
   from search import Index

   idx = Index()
   names = {c["note"] for c in idx.chunks}
   unknown = {n for _, g in QUESTIONS for n in g} - names
   assert not unknown, f"answer labels that are not note names: {unknown}"      # catches typos in the answer key
   import random
   random.seed(0)
   rand = sum(len(set(random.sample(sorted(names), 5)) & random.choice(QUESTIONS)[1]) > 0 for _ in range(20000)) / 20000
   print(f"for scale: five random notes would contain a correct one {rand:.3f} of the time")
   print(len(QUESTIONS), "questions,", len({c["note"] for c in idx.chunks}), "notes,", len(idx.chunks), "chunks")
   for method in ("keyword", "meaning", "hybrid"):
       hit1 = hit3 = hit5 = 0
       mrr = 0.0
       misses = []
       for q, good in QUESTIONS:
           ranked = [n for n, _ in idx.notes(q, method, k=10)]
           pos = next((r for r, n in enumerate(ranked) if n in good), None)
           if pos is not None:
               mrr += 1 / (pos + 1)
               hit1 += pos < 1; hit3 += pos < 3; hit5 += pos < 5
           if pos is None or pos >= 5:
               misses.append(q)
       n = len(QUESTIONS)
       print(f"{method:8s} found in top 1: {hit1/n:.2f}  top 3: {hit3/n:.2f}  top 5: {hit5/n:.2f}  mean reciprocal rank {mrr/n:.3f}")
       if method == "hybrid":
           print("  hybrid questions with no correct note in the top 5:")
           for q in misses: print("   -", q)
   ```
   Run `python evaluate.py`. I saw:
   ```
   for scale: five random notes would contain a correct one 0.030 of the time
   49 questions, 247 notes, 3009 chunks
   keyword  found in top 1: 0.61  top 3: 0.90  top 5: 0.94  mean reciprocal rank 0.756
   meaning  found in top 1: 0.65  top 3: 0.88  top 5: 0.94  mean reciprocal rank 0.772
   hybrid   found in top 1: 0.71  top 3: 0.94  top 5: 0.98  mean reciprocal rank 0.823
     hybrid questions with no correct note in the top 5:
      - How do language models avoid recomputing keys and values for earlier words?
   ```

5. **Prepare the answer step** ([[RAG]], [[Prompt Engineering]]). The script retrieves the top three notes for a question and builds the prompt a chat model would get: the chunks, a rule to answer only from them and cite them, and the question.

   **`ask.py`**
   ```python
   import sys

   from search import Index

   question = " ".join(sys.argv[1:]) or "Why does my loss turn into nan when the scores are large?"
   idx = Index()
   hits = idx.notes(question, "hybrid", k=3)
   context = "\n\n".join(f"[{i + 1}] (note: {idx.chunks[c]['note']}, section: {idx.chunks[c]['section']})\n{idx.chunks[c]['text']}" for i, (_, c) in enumerate(hits))
   prompt = (
       "Answer the question using only the notes below. Cite the note numbers you used. "
       "If the notes do not contain the answer, say so.\n\n"
       f"NOTES:\n{context}\n\nQUESTION: {question}\nANSWER:"
   )
   print("retrieved notes:", [n for n, _ in hits])
   print("-" * 60)
   print(prompt[:1500] + (" ..." if len(prompt) > 1500 else ""))
   print("-" * 60)
   print(f"prompt length: {len(prompt.split())} words. Paste it into any chat model, or send it to one from code (not done here).")
   ```
   Run `python ask.py "Why does my loss turn into nan when the scores are large?"`. I saw `retrieved notes: ['Pandas', 'Softmax', 'Embedding']`: the literal word `nan` pulls in the [[Pandas]] note first, and the right note, [[Softmax]], is second. A model that reads all three could still get it right; a system that only used the first hit would not.

6. **Test it** ([[pytest]]).

   **`test_rag.py`**
   ```python
   from chunker import split_note

   from questions import QUESTIONS
   from search import BM25, Index


   def test_chunker_drops_frontmatter_code_and_link_brackets():
       text = "---\ntags: [x]\n---\n# T\n\n## One\nSee [[Softmax]] and ![[pic.png]] " + "word " * 20 + "\n```python\nprint(1)\n```\n"
       chunks = split_note(text, "T")
       joined = " ".join(c["text"] for c in chunks)
       assert "tags:" not in joined and "print(1)" not in joined and "[[" not in joined and "Softmax" in joined
       assert chunks[0]["note"] == "T" and chunks[0]["section"] == "One"


   def test_bm25_prefers_the_chunk_with_the_rare_word():
       docs = ["cats and dogs and birds", "a zebra crossing the road", "dogs and dogs and more dogs"]
       s = BM25(docs).scores("zebra")
       assert s.argmax() == 1 and s[0] == 0


   def test_answer_key_names_exist_and_hybrid_search_finds_most_answers():
       idx = Index()
       names = {c["note"] for c in idx.chunks}
       assert all(g <= names for _, g in QUESTIONS)
       hits = sum(any(n in good for n, _ in idx.notes(q, "hybrid", 5)) for q, good in QUESTIONS)
       assert hits / len(QUESTIONS) >= 0.9
   ```
   Run `pytest -q`. I saw `3 passed`. The last test fails if the answer key has a typo or if the blended search finds a correct note in the top five for fewer than 90% of the questions, so changing the chunk size later cannot make it worse without notice.

7. **Test again with someone else's questions** ([[Retrieval Metrics]], [[LLM]]). Clone the public question list next to your scripts: `git clone --depth 1 https://github.com/KalyanKS-NLP/LLM-Interview-Questions-and-Answers-Hub` (about 2 MB, Apache-2.0 licence, by Kalyan KS). The script reads only the question titles from its `Interview_QA/QA_*.md` files. `labels.py` maps each question number to the vault notes that would answer it; an empty set means the vault has no note for it, and that question is skipped. The labels are generous for broad questions (many name [[Transformer]] or [[Self-Attention]]), which makes those easier than a strict key would.

   **`labels.py`**
   ```python
   # question number in the repo -> notes that count as a correct answer (empty = vault has no note for it)
   T="Transformer"; SA="Self-Attention"; A="Attention"; DS="Decoding Strategies"; FT="Fine-tuning"; PE="Prompt Engineering"
   LABELS={1:{"Positional Encoding","RoPE",T},2:{T,"KV Cache","Tokenization","LLM"},3:{"KV Cache"},4:{"Quantization"},
   5:{"KV Cache","PagedAttention","Grouped-Query Attention","Multi-head Latent Attention","Quantization","Sliding Window Attention"},
   6:{"Embedding","Tokenization",T},7:{"Tokenization","Byte-Pair Encoding"},8:{"Tokenization","Byte-Pair Encoding"},
   9:{SA,A,T},10:{SA,A,T},11:{"Residual Connection","Vanishing and Exploding Gradients",T},12:{"Tokenization"},
   13:{"Embedding",T},14:{"Embedding",T},15:{SA,A},16:{T},17:{T},18:{T},19:{SA,A,"Softmax"},20:{SA,A,T},21:{SA,A,T},
   22:{SA,A,T},23:{SA,A,T},24:{SA,A,T},25:{SA,A,T},26:{SA,A,T},27:{"Softmax"},28:{"Residual Connection"},
   29:{"Layer Normalization"},30:{"Loss Function","Softmax"},31:{T,"Recurrent Neural Network","LSTM","From FNN to RNN",SA},
   32:{T,SA},33:{T,"From FNN to RNN","Recurrent Neural Network"},34:{"Residual Connection","Vanishing and Exploding Gradients",T},
   35:{T},36:set(),37:set(),38:set(),39:set(),40:set(),41:{"Mixture of Experts"},42:{DS},43:{DS,"Beam Search"},44:{DS},
   45:{DS,"Beam Search"},46:{"Beam Search"},47:{"Beam Search",DS},48:{"Beam Search",DS},49:{DS,"Softmax"},50:{"Beam Search"},
   51:{DS},52:{DS},53:{"Context Rot","Sliding Window Attention","LLM"},54:{"Context Rot","Sliding Window Attention"},
   55:{DS,"Softmax"},56:{"LLM",DS,T,"KV Cache"},57:{"LLM",DS,"Multi-Token Prediction","Masked Diffusion Language Model"},
   58:{"Masked Diffusion Language Model","Diffusion Models"},59:{"Masked Diffusion Language Model"},60:set(),61:{"Speculative Decoding"},
   62:set(),63:set(),64:{"FlashAttention"},65:set(),66:{"Mixed Precision Training","Quantization"},67:set(),68:set(),69:set(),70:set(),
   71:set(),72:set(),73:{"Quantization","Speculative Decoding","KV Cache","FlashAttention"},74:{"Chain of Thought"},75:{"Chain of Thought"},
   76:{"Chain of Thought"},77:{PE},78:{PE,"In-Context Learning"},79:{"In-Context Learning",PE},80:{PE,"Context Rot"},81:{PE},
   82:{"In-Context Learning"},83:{"Chain of Thought",PE,DS},84:{PE,"Context Rot"},85:{PE,"RAG"},86:{PE},87:{"ReAct","Agent"},
   88:{"Pre-training",FT,"Instruction Tuning","RLHF","LLM"},89:{FT,"Instruction Tuning","LoRA","RLHF"},90:{"Instruction Tuning"},
   91:{"RLHF","Direct Preference Optimization",FT,"Reward Model"},92:{"Overfitting","Regularization",FT,"Dropout"},93:{FT},94:{FT},
   95:{"LoRA","QLoRA",FT},96:{PE,FT},97:{"RAG",FT},98:{"RAG"},99:{FT,"RAG"},100:{FT,PE},101:{"LoRA"},102:{"LoRA"},
   103:{"QLoRA","LoRA"},104:{"QLoRA"},105:{"QLoRA","LoRA","Quantization","Mixed Precision Training"},
   106:{"RLHF","Direct Preference Optimization","GRPO","PPO","Reward Model"},107:set(),108:{"LoRA","Mixed Precision Training","FlashAttention"},
   109:{"Pre-training","Self-Supervised Learning","LLM"},110:{"Pre-training","BERT","Self-Supervised Learning"},
   111:{"Tokenization","Byte-Pair Encoding"},112:{"Scaling Laws"},113:{"Mixture of Experts"},114:{"Data Parallelism"},
   115:{"Self-Supervised Learning","Pre-training"}}
   ```
   **`qa_eval.py`**
   ```python
   import glob, re, sys

   from labels import LABELS
   from search import Index

   repo = sys.argv[1] if len(sys.argv) > 1 else "LLM-Interview-Questions-and-Answers-Hub"
   questions = {}
   for path in glob.glob(f"{repo}/Interview_QA/QA_*.md"):
       for m in re.finditer(r"^## 📌 Q(\d+):\s*(.+)$", open(path, encoding="utf-8").read(), re.M):
           questions[int(m.group(1))] = m.group(2).strip()
   assert set(questions) == set(LABELS), "label file and repo disagree"

   idx = Index()
   names = {c["note"] for c in idx.chunks}
   assert not {n for g in LABELS.values() for n in g} - names, "answer label that is not a note name"
   scored = [(questions[i], LABELS[i]) for i in sorted(LABELS) if LABELS[i]]
   print(f"{len(questions)} questions in the repo, {len(scored)} scored, {len(questions) - len(scored)} skipped (no note in the vault covers them)")
   for method in ("keyword", "meaning", "hybrid"):
       h1 = h3 = h5 = 0
       mrr = 0.0
       misses = []
       for q, good in scored:
           ranked = [n for n, _ in idx.notes(q, method, k=10)]
           pos = next((r for r, n in enumerate(ranked) if n in good), None)
           if pos is not None:
               mrr += 1 / (pos + 1)
               h1 += pos < 1; h3 += pos < 3; h5 += pos < 5
           if pos is None or pos >= 5:
               misses.append(q)
       n = len(scored)
       print(f"{method:8s} top 1: {h1/n:.2f}  top 3: {h3/n:.2f}  top 5: {h5/n:.2f}  mean reciprocal rank {mrr/n:.3f}")
       if method == "hybrid":
           print(f"  hybrid: {len(misses)} questions with no correct note in the top 5:")
           for q in misses: print("   -", q)
   ```
   Run `python qa_eval.py LLM-Interview-Questions-and-Answers-Hub`. I saw:
   ```
   115 questions in the repo, 99 scored, 16 skipped (no note in the vault covers them)
   keyword  top 1: 0.41  top 3: 0.73  top 5: 0.81  mean reciprocal rank 0.586
   meaning  top 1: 0.72  top 3: 0.89  top 5: 0.93  mean reciprocal rank 0.807
   hybrid   top 1: 0.69  top 3: 0.91  top 5: 0.95  mean reciprocal rank 0.800
   ```

8. **Ask it a question and get notes back** ([[RAG]]). `ask.py` builds a prompt; this smaller script just prints the five best notes, each with the section and the start of the passage that matched. It is the "search box" without any language model.

   **`find.py`**
   ```python
   import sys

   from search import Index

   question = " ".join(sys.argv[1:]) or "Why does my loss turn into nan when the scores are large?"
   idx = Index()
   for rank, (note, c) in enumerate(idx.notes(question, "hybrid", k=5), 1):
       ch = idx.chunks[c]
       print(f"{rank}. [[{note}]]  ({ch['section']})")
       print("   " + ch["text"][:160] + "...")
   ```
   Run `python find.py "how does the model avoid recomputing old keys and values"`. I saw [[Grouped-Query Attention]], [[Multi-head Latent Attention]], [[KV Cache]], [[Self-Attention]], [[Hash Tables]]. The right note is third and the last one is clearly wrong, so read the list, not just the first line.

9. **Run all 115 questions and write the results down.** The same `search.py` run over every question in the interview list gave the table in [[Interview Question Index]], which also lists the questions the vault answers badly. That table is the gap list for the next notes to write.

## What went wrong, and the fix
- **Two wrong labels in my own answer key.** I wrote `Dijkstra` and `Greedy`; the notes are called `Dijkstra's Algorithm` and `Greedy Algorithms`. The first run counted those two questions as misses and gave blended top-5 of 0.94 instead of 0.98. I only noticed because the failures were odd. The fix was the check at the top of `evaluate.py` that stops if a label is not a real note name. **Lesson: check the answer key before blaming the search.**
- **The index answered its own questions.** When this project note was first added to the index, the question about `nan` returned this very note, because the note quotes that question in its text. Project logs repeat the test questions, so they would inflate the scores. The fix is to skip the `60-Projects` folder when indexing, and I replaced the one question whose answer was a project note (the web-page one) with a question about [[Streamlit]] and [[FastAPI]].
- **A module name that collides with the standard library.** Naming the file `chunk.py` would shadow Python's own `chunk` module. It is `chunker.py` now.
- **The exact word beats the right idea.** For the question about `nan`, the Pandas note ranks high because it contains the literal word. This is the weakness of keyword search; meaning search is meant to cover it, and here the right note, Softmax, came second.
- **A real miss I did not fix.** The KV cache question. Possible fixes (add a "common questions" line to each note, write queries with the notes' own terms, a bigger embedding model) are in "Next ideas". I did not try them.

## What I did not run
- **Any language model.** `ask.py` stops at the prompt. A chat model could read it, but I did not send it to one, so I have no measurement of answer quality, wrong answers or made-up citations. A full RAG system needs that check too ([[Evaluation Metrics]]). While checking whether a local model server was available I accidentally started the Ollama app on this PC; no model was downloaded or used.
- **A vector database** such as Chroma, and any scale test. The ranking code scans every chunk, which is fine for 3,034 and would not be for millions.
- **Questions from real users of this vault.** The first 49 questions are mine. The 99 from the interview repo are written by someone else, but they are interview questions, not what you would type while studying, and I chose the answer labels. Neither set is your own questions.
- **Other chunk sizes, other embedding models, other blending weights.** One setting each, so I do not know how sensitive the scores are. (Chunk sizes and a reranker were tested later in [[RAG Tuning Lab]]; other embedding models and blending weights still were not.)
- **Statistical confidence.** With 49 questions, the differences between methods are within what luck could produce.

## Concepts used
- [[RAG]] · [[Embedding Model]] · [[Embedding]] · [[Cosine Similarity]] · [[Vector Database]] · [[BM25]] · [[Contextual Retrieval]] · [[Retrieval Metrics]]
- [[Python]] · [[NumPy]] · [[pytest]] · [[Virtual Environments]] · [[Prompt Engineering]] · [[LLM]]

## Next ideas
- Send the prompt from `ask.py` to a chat model (a local one through Ollama, or an API) and grade the answers for correctness and for whether the citations match ([[LLM as a Judge]]).
- **Done once, with a caveat:** I wrote four notes for the gaps the interview test found ([[LLM Inference]], [[Continuous Batching]], [[Gradient Accumulation]], [[Model Parallelism]]), rebuilt the index (251 notes, 3,052 chunks) and re-labelled 15 of the 16 skipped questions (the inference-engine question stayed skipped). The blended search then put a correct note first for all 15, and in the top 5 for 15 of 15. **That is not a fair test:** I wrote those notes after reading those exact questions. It only shows the notes are findable. Over all 114 scored questions the blend found a correct note in the top 5 for 109 (0.956); I did not keep the script in this note.
- Use the gap list in [[Interview Question Index]]: write the missing notes it names, then re-run `qa_eval.py`.
- Collect 50 real questions you actually ask about your notes and re-run the table: that is the number that matters.
- Try the fix [[Contextual Retrieval]] describes: add one sentence of context to each chunk before indexing, and see whether the KV cache question gets found.
- Put the search behind an API and a small page, as in [[Iris Classifier Service]].
- **Done later:** chunk size, chunk style and a cross-encoder reranker were compared in [[RAG Tuning Lab]] ([[Chunking Strategies]], [[Reranking]]).
- Re-run the table whenever the vault grows, since new notes can crowd out old answers.

## Flashcards
#flashcards/projects

#flashcards/projects/concept #flashcards/topic/project-rag What were the five "done" checks of Vault Search with RAG?::(1) every note is cut into small chunks, (2) a keyword scorer (BM25) and a meaning scorer (embedding model) both rank the chunks, (3) a blended ranking exists, (4) the three rankings are compared on a fixed list of questions with retrieval metrics and a random baseline, (5) a script builds the prompt a chat model would receive.

#flashcards/projects/concept #flashcards/topic/project-rag What was indexed and what was left out?::Notes cut into chunks of at most 170 words, with code blocks, images, frontmatter and the `60-Projects` logs removed. (The note's chunk and note counts changed between runs, so the cards do not repeat them.)

#flashcards/projects/concept #flashcards/topic/project-rag Which embedding model was used?::`sentence-transformers/all-MiniLM-L6-v2`, 384-dimension vectors, about 90 MB, downloaded automatically the first time (internet needed once). The note did not check its licence.

#flashcards/projects/example #flashcards/topic/project-rag What did the three methods score on the 49 questions the note's author wrote?::Top 1 / top 3 / top 5 / mean reciprocal rank: BM25 0.61 / 0.90 / 0.94 / 0.756. Embeddings 0.65 / 0.88 / 0.94 / 0.772. Blended (reciprocal rank fusion) 0.71 / 0.94 / 0.98 / 0.823. A random pick of five notes would contain a correct one only 3.0% of the time.

#flashcards/projects/example #flashcards/topic/project-rag How much should the 49-question result be trusted?::Little. One question is about 2 percentage points, so 0.98 against 0.94 at top 5 is two questions. The blend is best or tied in every column, which is some evidence but not proof. The questions were written by the person who wrote the notes, so scores are probably optimistic.

#flashcards/projects/example #flashcards/topic/project-rag What did the 99 outside questions (from the Kalyan KS repo, Apache-2.0) show?::Top 1 / top 3 / top 5 / MRR: keywords 0.41 / 0.73 / 0.81 / 0.586. Meaning 0.72 / 0.89 / 0.93 / 0.807. Blended 0.69 / 0.91 / 0.95 / 0.800. Different wording hurts keyword search a lot (0.94 to 0.81 at top 5) and meaning search barely (0.94 to 0.93). Meaning alone is as good or better at top 1 and MRR, so this set does not support "blending always wins". 16 of 115 questions had no note and were skipped, and the labels are the note author's.

#flashcards/projects/concept #flashcards/topic/project-rag Which question failed completely?::"How do language models avoid recomputing keys and values for earlier words?" never put the KV Cache note in the top 5 (it ranked 18th in the blend). The note did not look into why. The wording may be far from the note's words.

#flashcards/projects/concept #flashcards/topic/project-rag What mistake was in the answer key?::Two labels `Dijkstra` and `Greedy`, when the notes are called `Dijkstra's Algorithm` and `Greedy Algorithms`. The first run counted them as misses (blended top-5 0.94 instead of 0.98). The fix is a check at the top of `evaluate.py` that stops if a label is not a real note name. Lesson: check the answer key before blaming the search.

#flashcards/projects/example #flashcards/topic/project-rag Why were the `60-Projects` logs skipped when indexing?::The index answered its own questions. A project note quoted a test question, so it was returned for it, which would inflate scores. The fix: skip the `60-Projects` folder and replace the one question whose answer was a project note.

#flashcards/projects/concept #flashcards/topic/project-rag Why is the file called `chunker.py` and not `chunk.py`?::`chunk.py` would shadow Python's own `chunk` module.

#flashcards/projects/concept #flashcards/topic/project-rag What does the `nan` question show about keyword search?::The exact word beats the right idea: the Pandas note ranks high because it contains the literal word. Meaning search is meant to cover this, and the right note (Softmax) came second.

#flashcards/projects/concept #flashcards/topic/project-rag What did Vault Search with RAG not run?::Any language model (`ask.py` stops at the prompt, so no measure of answer quality, wrong answers or made-up citations), a vector database such as Chroma or any scale test (the ranking code scans every chunk), real users' questions, other chunk sizes, embedding models or blending weights, and statistical confidence. A local model server (Ollama) was started by accident, but no model was downloaded or used.

#flashcards/projects/reverse #flashcards/topic/project-rag Reciprocal rank fusion:::The blended ranking that combines the BM25 and embedding rankings by rank position.

#flashcards/projects/cloze #flashcards/topic/project-rag Check the ==answer key== before blaming the search.
