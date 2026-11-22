---
tags: [project, python, pytorch, transformer, gpt, language-model, from-scratch]
status: done
stack: Python 3.9, PyTorch (CPU)
goal: build a small GPT (a Transformer that predicts the next character) from scratch, prove its attention and its "no peeking at the future" rule right, train it on this vault's own notes on a CPU, and measure how much better it is than simple counting models
---

# Tiny GPT from Scratch

> **In one line:** write a small decoder-only [[Transformer]] by hand, train it for about 25 minutes on a laptop CPU to predict the next character of this vault's notes, and measure what it learned against simple baselines.

## Goal
Done means five checks pass: (1) the attention I wrote matches PyTorch's own, (2) the model cannot see the future (changing the last character changes no earlier output), (3) the parameter count matches a formula I worked out by hand, (4) the trained model beats a one-character-memory counting model by a wide margin on notes it never saw, and (5) one design choice (position information) is ablated, with what that does and does not show written down.

## Result in short
- **The text:** this vault's 286 notes (3,016,498 characters) at the time of the run, with 162 distinct characters including emoji and math symbols. A random 28 notes (10%) were held out for validation: 2,718,735 characters to train on, 298,047 to test on.
- **The model:** 4 layers, width 128, 4 attention heads, 128 characters of context, **830,464 parameters** (the count and my formula agree).
- **Checks:** the attention matches PyTorch's to 1.2e-07, changing the last character changed earlier outputs by exactly 0.0, and the untrained loss was 5.079 against ln(162) = 5.088, the loss of a uniform guess. The model can memorise one batch of 1,024 characters (loss 3.358, then 0.008, then 0.001 after 100, 300 and 600 steps).
- **How good, on notes it never saw** (average loss per character in nats; lower is better):
  - uniform guess 5.088, letter frequencies 3.481, previous character only (bigram) 2.758;
  - tiny GPT: **1.429** (seed 0) and **1.447** (seed 1) after 1,500 steps. That is 2.06 and 2.09 bits per character, and a perplexity of about 4.2: the model is about as unsure as picking among 4 equally likely characters, against about 16 for the bigram model.
- **Position information matters:** with the position embedding removed, the same model scored 1.700 (one seed), 0.26 nats worse than the average of the two normal runs (the two normal seeds differ by 0.018).
- **Samples:** they look like this vault's formatting (headings, bullets, `$...$` maths, code fences) and use real words, but are not sensible text. Almost none of their 30-character windows appear in the training text, apart from boilerplate.
- **Not done:** one small size, one training length, no tuning, no GPU, and the model was still improving when I stopped.

## What you need
Versions I used: my existing Python environment on Windows, CPU only, 8 cores. **Not a fresh virtual environment** (PyTorch is a large install that I did not set up again).

| Tool | Version |
|---|---|
| Python | 3.9.6 |
| torch | 2.8.0+cpu |
| numpy | 1.24.4 |

No downloads. The text is this vault. Everything below runs from one folder. `corpus.txt` is built from the vault in step 1. **The vault changes over time, so a rebuilt corpus will not be identical.** Mine had the hash `37217eabb54b01f5` (first 16 characters of the SHA-256), which `checks.py` prints, and the numbers in this note belong to that snapshot.

## Steps
1. **Build the text file from the vault.** Each note is one record, and a separator character (U+001E) marks where a note ends. That lets me hold out whole notes.

   **`data.py`**
   ```python
   import hashlib, os, random
   import torch

   def build_corpus(vault, out="corpus.txt"):
       """every .md note under the vault (not .obsidian), sorted by path, joined with a separator character (U+001E) that marks where one note ends"""
       paths = sorted(os.path.join(r, f) for r, _, fs in os.walk(vault) if ".obsidian" not in r for f in fs if f.endswith(".md"))
       notes = [open(p, encoding="utf-8", errors="replace", newline="").read().replace("\r\n", "\n") for p in paths]
       open(out, "w", encoding="utf-8", newline="\n").write(chr(30).join(notes))
       return len(notes)

   def load(path="corpus.txt"):
       """split by whole notes: a random 10% of the notes (seed 0) are held out for validation"""
       raw = open(path, encoding="utf-8", newline="").read()
       notes = raw.split(chr(30))
       ids = list(range(len(notes))); random.Random(0).shuffle(ids); val_ids = set(ids[: len(notes) // 10])
       tr = chr(30).join(n for i, n in enumerate(notes) if i not in val_ids); va = chr(30).join(n for i, n in enumerate(notes) if i in val_ids)
       chars = sorted(set(raw)); stoi = {c: i for i, c in enumerate(chars)}
       enc = lambda t: torch.tensor([stoi[c] for c in t], dtype=torch.long)
       return enc(tr), enc(va), chars, tr, va, hashlib.sha256(raw.encode()).hexdigest()[:16]

   def batch(data, B, T, g):
       ix = torch.randint(0, len(data) - T - 1, (B,), generator=g)
       return torch.stack([data[i:i + T] for i in ix]), torch.stack([data[i + 1:i + T + 1] for i in ix])
   ```
   Run `python -c "from data import build_corpus; print(build_corpus(r'PATH TO YOUR VAULT'))"`. I saw `286 notes`. The validation set is 28 whole notes (a tenth, chosen with a fixed seed), so the model is scored on notes it never saw, not on neighbouring paragraphs of notes it did.

2. **Write the model.** Each block is: normalise ([[Layer Normalization]]), attention, add back ([[Residual Connection]]), normalise, a small two-layer network with a GELU curve, add back. Attention is [[Self-Attention]] with a mask that blocks the future. The output layer shares its weights with the letter table. Letters become vectors with an [[Embedding]], and a second table adds the position ([[Positional Encoding]]). Weights start as normal numbers with a spread of 0.02 ([[Weight Initialization]]).

   **`model.py`**
   ```python
   import math
   import torch, torch.nn as nn, torch.nn.functional as F

   class Block(nn.Module):
       def __init__(s, d, h):
           super().__init__(); s.h = h
           s.ln1, s.qkv, s.proj = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
           s.ln2, s.fc, s.out = nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
       def attn(s, x):
           B, T, D = x.shape; q, k, v = s.qkv(x).split(D, 2)
           q, k, v = [t.view(B, T, s.h, D // s.h).transpose(1, 2) for t in (q, k, v)]       # (batch, head, position, size)
           w = (q @ k.transpose(-2, -1)) / math.sqrt(D // s.h)                                # how much each position likes each other one
           w = w.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf")).softmax(-1)   # no looking at the future
           return s.proj((w @ v).transpose(1, 2).reshape(B, T, D))
       def forward(s, x):
           x = x + s.attn(s.ln1(x))                       # attention, added back to the stream (pre-norm)
           return x + s.out(F.gelu(s.fc(s.ln2(x))))       # feed-forward, added back

   class GPT(nn.Module):
       def __init__(s, V, T=128, d=128, L=4, h=4, pos=True):
           super().__init__(); s.T, s.pos = T, pos
           s.tok, s.posemb = nn.Embedding(V, d), nn.Embedding(T, d)
           s.blocks, s.lnf = nn.ModuleList(Block(d, h) for _ in range(L)), nn.LayerNorm(d)
           s.head = nn.Linear(d, V, bias=False); s.head.weight = s.tok.weight               # the output layer shares the letter table
           for m in s.modules():
               if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, 0, 0.02)
               if isinstance(m, nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
       def forward(s, idx):
           x = s.tok(idx) + (s.posemb(torch.arange(idx.shape[1])) if s.pos else 0)
           for b in s.blocks: x = b(x)
           return s.head(s.lnf(x))
       @torch.no_grad()
       def generate(s, idx, n, temp, g):
           for _ in range(n):
               p = F.softmax(s(idx[:, -s.T:])[:, -1] / temp, -1)
               idx = torch.cat([idx, torch.multinomial(p, 1, generator=g)], 1)
           return idx
   ```

3. **Check it is right.**

   **`checks.py`**
   ```python
   import math, torch, torch.nn.functional as F
   from data import *; from model import GPT, Block
   torch.set_num_threads(8); torch.manual_seed(0)
   tr, va, chars, trs, vas, h = load(); V = len(chars); print(f"corpus {h}: train {len(tr):,} characters, validation {len(va):,}, {V} distinct characters")

   # 1) my attention against PyTorch's built-in causal attention (same projections)
   b = Block(32, 4).eval(); x = torch.randn(2, 10, 32); q, k, v = [t.view(2, 10, 4, 8).transpose(1, 2) for t in b.qkv(x).split(32, 2)]
   ref = b.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(2, 10, 32))
   print("1) attention vs F.scaled_dot_product_attention: largest gap", f"{(b.attn(x) - ref).abs().max().item():.1e}")

   # 2) causality: changing the LAST character must not change any earlier output
   m = GPT(V).eval(); idx = torch.randint(0, V, (3, 64)); idx2 = idx.clone(); idx2[:, -1] = (idx2[:, -1] + 1) % V
   with torch.no_grad(): a, c = m(idx), m(idx2)
   print("2) change the last character: earlier outputs change by", (a[:, :-1] - c[:, :-1]).abs().max().item(), "| the last output changes by", round((a[:, -1] - c[:, -1]).abs().max().item(), 4))

   # 3) parameter count against the formula  L(12d^2+13d) + V*d + T*d + 2d
   d, L, T = 128, 4, 128; formula = L * (12 * d * d + 13 * d) + V * d + T * d + 2 * d
   print(f"3) parameters: counted {sum(p.numel() for p in m.parameters()):,} | formula {formula:,} | output layer shares the letter table: {m.head.weight is m.tok.weight}")

   # 4) an untrained model should be as bad as a uniform guess: loss near ln(V)
   g = torch.Generator().manual_seed(1); x, y = batch(va, 64, 128, g)
   with torch.no_grad(): print(f"4) untrained loss {F.cross_entropy(m(x).view(-1, V), y.view(-1)).item():.3f} | ln({V}) = {math.log(V):.3f}")

   # 5) the model should be able to memorise one batch (shows the learning loop works); the loss sits on a plateau for a while, then drops
   torch.manual_seed(0); m = GPT(V); opt = torch.optim.AdamW(m.parameters(), 3e-3); x, y = batch(tr, 8, 128, g); seen = []
   for step in range(1, 601):
       loss = F.cross_entropy(m(x).view(-1, V), y.view(-1)); opt.zero_grad(); loss.backward(); opt.step()
       if step in (100, 300, 600): seen.append(f"{loss.item():.3f}")
   print("5) one batch of 1,024 characters, loss after 100 / 300 / 600 steps:", " / ".join(seen))
   ```
   Run `python checks.py`. I saw:
   ```
   corpus 37217eabb54b01f5: train 2,718,735 characters, validation 298,047, 162 distinct characters
   1) attention vs F.scaled_dot_product_attention: largest gap 1.2e-07
   2) change the last character: earlier outputs change by 0.0 | the last output changes by 0.9502
   3) parameters: counted 830,464 | formula 830,464 | output layer shares the letter table: True
   4) untrained loss 5.079 | ln(162) = 5.088
   5) one batch of 1,024 characters, loss after 100 / 300 / 600 steps: 3.358 / 0.008 / 0.001
   ```
   What each one proves: (1) my attention equals PyTorch's built-in causal attention, up to rounding. (2) The last character changed the last output but changed nothing earlier, so the mask works. (3) Per block: 12 × d × d + 13 × d numbers (with d = 128 that is 198,272), times 4 blocks, plus 162×128 for the letters, 128×128 for positions and 256 for the final normalisation: 830,464, and the output layer adds nothing because it shares the letter table. (4) An untrained model is as good as a uniform guess. (5) The learning loop can fit one batch, but note the plateau: the loss sat at 3.358 after 100 steps and only fell later (see "What went wrong").

4. **Make the baselines.** These are counting models that need no neural network, using the training text only, scored on the validation notes: a uniform guess, letter frequencies (add-1 smoothing), and a bigram model, where the next character depends only on the previous one (add-0.1 smoothing, not tuned).

   **`baseline.py`**
   ```python
   import math, torch
   from data import *
   tr, va, chars, trs, vas, h = load(); V = len(chars)
   u = torch.bincount(tr, minlength=V).float() + 1; u /= u.sum()                                   # unigram, add-1
   pairs = torch.zeros(V, V); pairs.index_put_((tr[:-1], tr[1:]), torch.ones(len(tr) - 1), accumulate=True)
   bi = (pairs + 0.1) / (pairs + 0.1).sum(1, keepdim=True)                                          # bigram, add-0.1 (not tuned)
   nll = lambda p: -p.log().mean().item()
   print(f"uniform guess      : {math.log(V):.3f} nats per character")
   print(f"unigram (letter frequencies): {nll(u[va[1:]]):.3f}")
   print(f"bigram (previous character) : {nll(bi[va[:-1], va[1:]]):.3f}")
   ```
   Run `python baseline.py`. I saw:
   ```
   uniform guess      : 5.088 nats per character
   unigram (letter frequencies): 3.481
   bigram (previous character) : 2.758
   ```

5. **Train.** [[Adam Optimizer]] with weight decay (AdamW), a [[Learning Rate]] that warms up for 100 steps to 0.003 and then falls along a cosine curve to 0.0003, gradient clipping at 1.0, batches of 32 windows of 128 characters. 1,500 steps see 6,144,000 characters, which is about 2.26 passes over the training text. Every 250 steps it scores 40 fixed batches of training text and of validation text (the loss is [[Cross-Entropy Loss]]). At the end it writes two samples, at temperature 1.0 and 0.7, and checks how many 30-character windows of each sample appear word for word in the training text.

   **`train.py`**
   ```python
   import argparse, json, math, time
   import torch, torch.nn.functional as F
   from data import *; from model import GPT
   ap = argparse.ArgumentParser(); ap.add_argument("--name"); ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--seed", type=int, default=0)
   ap.add_argument("--pos", type=int, default=1); ap.add_argument("--threads", type=int, default=2); a = ap.parse_args()
   torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
   tr, va, chars, trs, vas, h = load(); V = len(chars); B, T = 32, 128
   m = GPT(V, T=T, pos=bool(a.pos)); opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.01); g = torch.Generator().manual_seed(a.seed)
   lr = lambda s: 3e-3 * (s + 1) / 100 if s < 100 else 3e-4 + (3e-3 - 3e-4) * 0.5 * (1 + math.cos(math.pi * (s - 100) / (a.steps - 100)))   # warm up, then cosine down to 3e-4
   def evaluate(data, seed):
       gg = torch.Generator().manual_seed(seed); tot = 0
       with torch.no_grad():
           for _ in range(40): x, y = batch(data, 32, T, gg); tot += F.cross_entropy(m(x).view(-1, V), y.view(-1)).item()
       return tot / 40
   log = []; t0 = time.time()
   for s in range(a.steps + 1):
       if s % 250 == 0 or s == a.steps:
           log.append((s, evaluate(tr, 456), evaluate(va, 123)))
           print(f"{a.name} step {s:5d} | train {log[-1][1]:.3f} | validation {log[-1][2]:.3f} | {time.time() - t0:.0f}s", flush=True)
       if s == a.steps: break
       for p in opt.param_groups: p["lr"] = lr(s)
       x, y = batch(tr, B, T, g); loss = F.cross_entropy(m(x).view(-1, V), y.view(-1)); opt.zero_grad(); loss.backward()
       torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
   # samples, and how much of a sample is copied from the training text (30-character windows found verbatim)
   gs = torch.Generator().manual_seed(7); prompt = "## \U0001f7e2 1. Simple (age 12)\n"; stoi = {c: i for i, c in enumerate(chars)}
   res = dict(name=a.name, log=log, params=sum(p.numel() for p in m.parameters()), seconds=time.time() - t0, samples={})
   for temp in (1.0, 0.7):
       out = m.generate(torch.tensor([[stoi[c] for c in prompt]]), 600, temp, gs)[0].tolist(); txt = "".join(chars[i] for i in out)
       w = [txt[i:i + 30] for i in range(len(prompt), len(txt) - 30)]; res["samples"][temp] = (txt, sum(x in trs for x in w) / len(w))
       print(f"--- {a.name}, temperature {temp}: {res['samples'][temp][1] * 100:.1f}% of 30-character windows appear verbatim in the training text ---\n{txt}\n", flush=True)
   json.dump(res, open(f"result_{a.name}.json", "w"))
   ```
   I ran three runs **at the same time**, with 2 threads each, so their run times are not a benchmark:
   ```
   python -u train.py --name main0 --seed 0 --pos 1 --threads 2 --steps 1500
   python -u train.py --name main1 --seed 1 --pos 1 --threads 2 --steps 1500
   python -u train.py --name nopos0 --seed 0 --pos 0 --threads 2 --steps 1500
   ```
   Each took about 25 minutes (1,545, 1,536 and 1,563 seconds). The loss lines (averaged loss per character in nats):
   ```
   main0 step     0 | train 5.119 | validation 5.117 | 29s
   main0 step   250 | train 2.482 | validation 2.507 | 299s
   main0 step   500 | train 1.875 | validation 1.935 | 526s
   main0 step   750 | train 1.599 | validation 1.683 | 771s
   main0 step  1000 | train 1.459 | validation 1.543 | 1025s
   main0 step  1250 | train 1.371 | validation 1.466 | 1288s
   main0 step  1500 | train 1.328 | validation 1.429 | 1545s
   main1 step     0 | train 5.115 | validation 5.116 | 28s
   main1 step   250 | train 2.478 | validation 2.507 | 283s
   main1 step   500 | train 1.943 | validation 2.004 | 511s
   main1 step   750 | train 1.668 | validation 1.747 | 759s
   main1 step  1000 | train 1.505 | validation 1.595 | 1007s
   main1 step  1250 | train 1.396 | validation 1.489 | 1271s
   main1 step  1500 | train 1.349 | validation 1.447 | 1536s
   nopos0 step     0 | train 5.108 | validation 5.104 | 24s
   nopos0 step   250 | train 2.610 | validation 2.633 | 302s
   nopos0 step   500 | train 2.434 | validation 2.468 | 533s
   nopos0 step   750 | train 2.204 | validation 2.253 | 782s
   nopos0 step  1000 | train 1.908 | validation 1.975 | 1039s
   nopos0 step  1250 | train 1.699 | validation 1.784 | 1298s
   nopos0 step  1500 | train 1.608 | validation 1.700 | 1563s
   ```

## What the numbers say
- **The model learned a lot, fast, then slowly.** Loss on unseen notes went 5.12 (a guess) to 2.51 after 250 steps, 1.94 after 500 and 1.43 at the end. The last 250 steps still gained 0.04, so it was **not converged** (the falling learning rate also makes the end of the curve look flatter than it is).
- **It beats counting by a wide margin.** The average of the two normal seeds, 1.438, against 2.758 for the bigram model is 1.32 nats lower. In perplexity that is about 4.2 against about 15.8, so the model is roughly 3.7 times less "surprised" per character. A bigram model is a weak baseline, and I did not try longer n-grams.
- **It is a little overfit, as expected for 2.26 passes over 2.7 million characters:** training loss was 1.328 and 1.349, validation 1.429 and 1.447 (a gap of about 0.10 nats). The first version of my split leaked, which hid most of that gap (see "What went wrong").
- **Seeds:** 1.429 and 1.447 differ by 0.018. With two seeds I cannot give an error bar, but the gaps below are much larger than this.
- **Position embedding:** removing it cost 0.26 nats on average (1.700 against 1.438), about 14 times the seed difference. Still, the model without it learned a great deal, well past the bigram (2.758). That suggests the attention mask itself carries some order information, but I did not test how. It was also **still falling fast** at the end (1.784 to 1.700 in the last 250 steps), so a longer run might narrow the gap. One seed only.
- **Samples are not good text.** A line from the model at temperature 0.7: `The code worst you and from the tool feature the same before and write as a small of the average coome a weight an a random (from memory)`. Words are mostly real, and structure (a bullet, bold `**Matters:**`, `$...$`) is learned, but the sentences do not mean anything. I judged this by reading a few samples. I did not measure quality in any other way.
- **Copying from the training text** (windows of 30 characters appearing verbatim): for the six samples, 0.0% to 13.5% (main0: 0.0% at temperature 1.0 and 1.8% at 0.7, main1: 13.5% and 0.0%, the no-position run: 0.0% and 0.0%). I looked at what matched in the 13.5% case. It was a standard code header (`import numpy as np` then `rng = np.`, which occurs 47 times in the training text) and a long run of spaces, not a unique passage. So I see no memorisation of unique text, but the check is crude.

## What went wrong, and the fix
- **My first train/validation split leaked.** The first `load()` split the text on blank lines, but notes are full of blank lines, so my "10% held-out notes" were actually 10% of the paragraphs, scattered through notes whose other paragraphs were in training. I noticed before writing results, and fixed it by marking note ends with a separator character and splitting on that. I **reran everything**. For the record, the first runs gave validation 1.393 (seed 0), 1.405 (seed 1) and 1.651 (no position) against 1.429, 1.447 and 1.700 after the fix. So the leak flattered the loss by 0.036 to 0.049 nats and shrank the train/validation gap from about 0.10 to about 0.03. The numbers in this note are from the fixed version only.
- **The "memorise one batch" check looked broken at first.** In my first draft it ran 300 steps and printed a loss of 2.245. A longer test (1,200 steps) showed a long plateau and then a sudden fall to about 0.005, so the model was fine. The plateau length depends on the random start: with a different start, one run was still on the plateau at 300 steps, and in the final run the loss was 3.358, 0.008 and 0.001 after 100, 300 and 600 steps. I changed the check to print three points and run to 600 steps. **I chose 600 steps after seeing this,** so it is a sanity check, not a result.
- **Evaluation is slow on a CPU.** Each scoring pass (80 batches) took about 25 seconds, so the 7 passes used roughly 3 minutes of each run. I kept them.

## What I did not run
- **A fresh virtual environment.** I used my existing one (versions above). The three runs and the checks ran in a fresh folder that held only the files from this note and a copy of the corpus snapshot.
- **A rerun of the three training runs from the note.** They were run once each, from files I then compared with the code blocks in this note (see below). Results may differ slightly if you rerun them with a different number of threads, because that changes the order of the additions.
- **Any tuning.** The learning rate, batch size, context length, width, depth and number of steps were each chosen once without comparing alternatives.
- **Other model sizes or longer training** (so no scaling statement), a GPU, [[Dropout]], or more than 2 seeds, and the position ablation has 1 seed.
- **Near-duplicate text between the training and validation notes.** The notes share templates and snippets, and I did not measure how much, so the validation score is probably better than it would be on text from a different source.
- **A fair comparison with longer n-gram models,** a quality score for the samples beyond reading a few, and looking at what the attention heads learned.
- **The note in Obsidian.** I did not open it there.

## Concepts used
- [[Transformer]] · [[Self-Attention]] · [[Attention]] · [[Layer Normalization]] · [[Residual Connection]] · [[Embedding]] · [[Positional Encoding]] · [[Tokenization]] · [[Softmax]] · [[Cross-Entropy Loss]] · [[Perplexity]] · [[Adam Optimizer]] · [[Learning Rate]] · [[Weight Initialization]] · [[Overfitting]] · [[Dropout]] · [[LLM]]
- Papers: [[Attention Is All You Need 2017]] and [[GPT-3 2020]] for the design this copies at a tiny scale. I did not re-read them for this project.

## Next ideas
- Train longer and plot where the validation loss stops falling.
- Add dropout and a small grid over learning rate and width, with a proper validation split used for choosing.
- Try a word-piece [[Tokenization]] and compare the loss per character.
- Add a trigram or 4-gram baseline to see how much of the gain over bigram is just longer memory.
- Look at the attention weights of one head on one sentence.

## Flashcards
#flashcards/projects

#flashcards/projects/concept #flashcards/topic/project-deep-learning What were the five "done" checks of Tiny GPT from Scratch?::(1) the attention written by hand matches PyTorch's own, (2) the model cannot see the future (changing the last character changes no earlier output), (3) the parameter count matches a hand-worked formula, (4) the trained model beats a one-character-memory counting model by a wide margin on notes it never saw, (5) position information is ablated with what that does and does not show.

#flashcards/projects/concept #flashcards/topic/project-deep-learning What text was used and how was it split?::This vault's 286 notes at the time of the run (3,016,498 characters, 162 distinct characters including emoji and math symbols). A random 28 notes (10%) were held out: 2,718,735 characters to train on and 298,047 to test on.

#flashcards/projects/concept #flashcards/topic/project-deep-learning What was the model?::4 layers, width 128, 4 attention heads, 128 characters of context, 830,464 parameters (the count and the hand formula agree).

#flashcards/projects/concept #flashcards/topic/project-deep-learning What were the sanity checks?::Attention matches PyTorch's to 1.2e-07. Changing the last character changed earlier outputs by exactly 0.0. The untrained loss was 5.079 against $\ln 162=5.088$, the loss of a uniform guess. The model could memorise one batch of 1,024 characters (loss 3.358, then 0.008, then 0.001 after 100, 300 and 600 steps).

#flashcards/projects/concept #flashcards/topic/project-deep-learning What losses did the baselines and the model get on unseen notes?::Uniform guess 5.088, letter frequencies 3.481, previous character only (bigram) 2.758 nats per character. Tiny GPT: 1.429 (seed 0) and 1.447 (seed 1) after 1,500 steps, which is 2.06 and 2.09 bits per character.

#flashcards/projects/example #flashcards/topic/project-deep-learning What does the tiny GPT's perplexity of about 4.2 mean?::The model is about as unsure as picking among 4 equally likely characters, against about 16 for the bigram model. The two-seed average of 1.438 is 1.32 nats lower than the bigram's 2.758, roughly 3.7 times less surprised per character.

#flashcards/projects/concept #flashcards/topic/project-deep-learning What did removing the position embedding do?::The same model scored 1.700 (one seed), 0.26 nats worse than the average of the two normal runs (the two normal seeds differ by 0.018), about 14 times the seed difference. It still learned a lot, well past the bigram. It was still falling fast at the end, so a longer run might narrow the gap. One seed only.

#flashcards/projects/concept #flashcards/topic/project-deep-learning How good were the samples?::They look like the vault's formatting (headings, bullets, `$...$` maths, code fences) and use real words, but are not sensible text. Almost none of their 30-character windows appear in the training text, apart from boilerplate such as `import numpy as np`.

#flashcards/projects/concept #flashcards/topic/project-deep-learning How overfit was the model?::A little. Training loss 1.328 and 1.349, validation 1.429 and 1.447, a gap of about 0.10 nats. That is expected for about 2.26 passes over 2.7 million characters.

#flashcards/projects/concept #flashcards/topic/project-deep-learning What was wrong with the first train/validation split?::It split the text on blank lines, but notes are full of blank lines, so the "10% held-out notes" were 10% of the paragraphs, scattered through notes whose other paragraphs were in training. The fix: mark note ends with a separator character and split on that, then rerun everything. The leak had flattered the loss by 0.036 to 0.049 nats (first runs gave validation 1.393, 1.405 and 1.651).

#flashcards/projects/concept #flashcards/topic/project-deep-learning Why did the "memorise one batch" check look broken at first?::It ran 300 steps and printed 2.245. A longer test (1,200 steps) showed a long plateau and then a sudden fall to about 0.005. The plateau length depends on the random start. The check was changed to print three points over 600 steps. That choice was made after seeing the result, so it is a sanity check, not a result.

#flashcards/projects/concept #flashcards/topic/project-deep-learning Why is the validation score probably better than on text from a different source?::The notes share templates and snippets, and the note did not measure the near-duplicate text between training and validation notes.

#flashcards/projects/concept #flashcards/topic/project-deep-learning What did Tiny GPT not run?::A fresh virtual environment, any tuning, other model sizes or longer training, a GPU, Dropout, more than 2 seeds (the position ablation has 1), a fair comparison with longer n-gram models, a quality score for samples beyond reading a few, and a look at what the attention heads learned. The model was still improving when it stopped.

#flashcards/projects/cloze #flashcards/topic/project-deep-learning A bigram model is a ==weak baseline==: the note did not try longer n-grams.
