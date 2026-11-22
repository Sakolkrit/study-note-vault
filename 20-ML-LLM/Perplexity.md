---
tags: [perplexity, evaluation, language-model, LLM, loss-function, probability, math]
status: growing
level: 4
tier: 2
---

# Perplexity

## 🟢 1. Simple (age 12)
Perplexity is a **"how confused is the AI?" score** for a language model. Low means it was rarely surprised by the text. High means it was very confused. A perplexity of 4 means the AI was about as unsure as if it had to pick between **4 equally likely words** at every step.

**Example:** You read a story aloud and a friend guesses each next word. If the story is "once upon a ..." they get it right easily, so they are not confused (low score). If the story is full of unusual words and surprises, they are often confused (high score). Perplexity measures that for an AI over a whole text.

## 🔵 2. How it works
1. Give the [[LLM]] a piece of text it has not seen.
2. Go through it one word-piece (token, see [[Tokenization]]) at a time. At each step, the model says how likely it thought the **real** next token was.
3. If it gave the real token a high chance, it was not surprised. A low chance means it was surprised.
4. Combine all those chances into one average "surprise" number (this is the [[Loss Function]] used in training).
5. Turn that average into a friendlier number: **perplexity**, the "number of equally likely choices" it feels like.
6. **Lower is better:** the model predicted the text well.

There is no image for this note: I did not find a suitable free picture on Wikimedia Commons, so I left it out rather than use an unrelated one.

## 🟠 3. Technical detail
- **Per token:** for each position the model gives a [[Probability]] to every token in its vocabulary (via [[Softmax]]). Perplexity uses the probability given to the token that actually came next.
- **Same as training loss:** the average negative log-probability is the cross-entropy loss. Perplexity is just that loss passed through $e^{x}$, so a falling training loss means a falling perplexity.
- **Held-out text:** measure on text the model did not train on. Perplexity on the training text hides memorisation ([[Overfitting]]).
- **Sliding windows:** for texts longer than the model's context, a window moves over the text and each token is scored with as much context as fits. Cutting the text into separate chunks instead gives a worse (higher) number, because most tokens then see little context. The Hugging Face guide reports, for GPT-2 large on WikiText-2, 19.44 with no overlap against 16.44 with a stride of 512 (from a page summary; I did not rerun it).
- **Used for:** comparing language models, tracking training progress, checking whether [[Fine-tuning]] helped on a domain, and spotting unusual text (very high perplexity can flag gibberish).
- **Limits:**
  - Only compare perplexities made with the **same tokenizer and the same text**. A model with a different token vocabulary splits the text into different pieces, so the numbers are not comparable.
  - A low perplexity does not mean the model is helpful, truthful or safe. A chat assistant tuned with [[RLHF]] often has a *worse* perplexity than its base model but behaves far better.
  - If test text leaked into training data, perplexity will look unrealistically good.

```python
import math
def ppl(probs):                       # probs = chance the model gave to each REAL next token
    return math.exp(-sum(math.log(p) for p in probs) / len(probs))
print(round(ppl([0.5, 0.25, 0.125]), 3))   # 4.0    like choosing among 4 options
print(round(ppl([0.9, 0.8, 0.7]), 3))      # 1.257  confident and right: near 1 (the best possible)
V = 1000                                   # a model guessing uniformly over 1000 tokens
print(round(ppl([1 / V] * 5), 1))          # 1000.0 perplexity equals the vocabulary size
```

## 🔴 4. Deep / formal
**Definition.** For a text of $N$ tokens $x_1,\dots,x_N$, where the model gives the real $i$-th token the chance $p(x_i\mid x_{<i})$:

$$
\text{PPL}=\exp\!\Big(-\frac1N\sum_{i=1}^{N}\ln p\big(x_i\mid x_{<i}\big)\Big)
$$

**Where:**
- $N$ is the number of tokens scored, and $i$ counts through them.
- $x_i$ is the real $i$-th token. $x_{<i}$ is all the tokens before it (the context).
- $p(x_i\mid x_{<i})$ is the chance the model gave to the real token, knowing the context. **Matters:** it is the only thing that is measured. The model is scored only on how much probability it put on what actually happened.
- $\ln p$ is the natural log, always $\le0$ for a chance. It is $0$ for a certain prediction and very negative for a surprising one. The minus sign flips it into a positive "surprise".
- $\frac1N\sum$ averages the surprise over the text. **Matters:** the average makes the score independent of text length.
- $\exp(\cdot)$ turns the average surprise back into a number of "equally likely choices". **Matters:** this is why it reads as "as confused as picking among PPL options".

**Example (tiny numbers):** a 3-token text where the model gave the real tokens the chances $0.5,\ 0.25,\ 0.125$.
- $\ln0.5=-0.693$, $\ln0.25=-1.386$, $\ln0.125=-2.079$. Sum $=-4.159$.
- Average surprise: $4.159/3=\mathbf{1.386}$.
- $\text{PPL}=e^{1.386}=\mathbf{4.0}$.
- A model more confident and right, with chances $0.9,0.8,0.7$: average surprise $0.228$, $\text{PPL}=\mathbf{1.257}$, close to the best possible value of $1$.
- Meaning: the first model behaved as if it was choosing among 4 equally likely tokens at each step.

**Link to the training loss.** Perplexity is the exponential of the average cross-entropy $L$ (see [[Loss Function]]):

$$
\text{PPL}=e^{L},\qquad L=-\frac1N\sum_{i}\ln p(x_i\mid x_{<i})\quad\text{(in nats)}
$$

**Where:**
- $L$ is the average cross-entropy per token, measured in **nats** because it uses the natural log. If it is measured with $\log_2$ it is in **bits**, and then $\text{PPL}=2^{L_{\text{bits}}}$. Both give the same perplexity.
- $\text{PPL}$ is the perplexity, and $e\approx2.718$.
- **Matters:** $L$ is what [[Gradient Descent]] lowers during training, and perplexity is the same quantity in a more readable form. A drop of $L$ by $0.1$ nats multiplies perplexity by $e^{-0.1}\approx0.905$, a $9.5\%$ reduction.

**Example (tiny numbers):** a vocabulary of $1000$ tokens.
- A model with no knowledge guesses uniformly: every token gets $0.001$. $L=-\ln0.001=6.908$ and $\text{PPL}=e^{6.908}=\mathbf{1000}$: exactly the vocabulary size, the worst sensible score.
- A model with $L=\ln4=1.386$ has $\text{PPL}=\mathbf{4}$.
- A model with $L=0$ would always give the real token chance $1$, so $\text{PPL}=\mathbf{1}$, a perfect prediction (essentially only possible for text it has memorised).

**Further notes:**
- **Bits per character or per byte** are used to compare models with different tokenizers, because the unit (a character or a byte) is the same for all of them. (A common practice; stated from memory. Equation 3 below uses it on a real model.)
- **Why exp?** Information theory: the exponential of the entropy is the effective number of equally likely outcomes. This is why perplexity is called a "branching factor".
- **Context matters:** longer context usually lowers perplexity, since the model has more clues.
- **Beyond perplexity:** modern evaluation also uses task benchmarks, human comparisons and preference tests ([[RLHF]]), because next-token accuracy does not capture everything we want from an assistant.

**Measured on a real model: what quantization does to perplexity.** To compare models that cut text into different pieces, use bits per byte, which has the same unit for everyone (equation 3).

$$\text{bits per byte}=\frac{1}{B}\sum_{i=1}^{N}-\log_2 p\big(x_i\mid x_{<i}\big)=\frac{N\,\log_2\text{PPL}}{B}$$

**Where:**
- $N$ is the number of tokens scored and $B$ is the number of bytes of text those tokens cover.
- $-\log_2 p(x_i\mid x_{<i})$ is the surprise of one token in bits.
- $\log_2\text{PPL}$ is the average surprise per token, in bits (the same as $L$ in bits).

**Matters:** perplexity depends on how the text is cut into tokens, but a byte is a byte. Dividing the total surprise by the bytes gives a number you can compare across models with different tokenizers, as long as the text is the same.

**Example (tiny numbers):** a model has perplexity 8 on 100 tokens that cover 400 bytes. Per token that is $\log_2 8=3$ bits, so the text costs $100\times3=300$ bits. $300/400=0.75$ bits per byte. In my real run below: $2047\times3.753/7118=1.079$.

I measured Qwen2.5-0.5B-Instruct on 2,048 tokens of this vault's own prose (eight notes with code, equations and flashcards removed), once with its original weights and then with the weights replaced by quantized ones that were turned back into ordinary numbers ([[Quantization]], [[GGUF]]). As far as I know the text was never published, so the model has not seen it. The script uses a sliding window of 512 tokens with a stride of 256.

**`ppl_quant.py`**
```python
"""Perplexity of Qwen2.5-0.5B-Instruct on this vault's own prose, before and after quantizing its weights.
Run from the project folder:  python ppl_quant.py FILE.gguf   (FILE.gguf = the file Ollama stores for qwen2.5:0.5b)"""
import math, re, sys, time, torch
import numpy as np
import gguf
from gguf import quants, GGMLQuantizationType as T
from transformers import AutoTokenizer, AutoModelForCausalLM

NOTES = ["Cross-Entropy Loss", "Transformer", "KV Cache", "Softmax", "Overfitting", "Embedding", "Attention", "Backpropagation"]
VAULT = "D:/python456/study&note stuff/20-ML-LLM/"
MAX_LEN, STRIDE, N_TOKENS = 512, 256, 2048

tok = AutoTokenizer.from_pretrained("qwen2.5-0.5b-instruct")
model = AutoModelForCausalLM.from_pretrained("qwen2.5-0.5b-instruct", torch_dtype=torch.float32).eval()
orig = {k: v.clone() for k, v in model.state_dict().items()}


FENCE = chr(96) * 3                                 # three backticks, built here so this file can sit inside a note


def prose(name):                                    # a note's text without frontmatter, code, equations, flashcards
    s = open(VAULT + name + ".md", encoding="utf-8").read().split("\n## Fl" + "ashcards")[0]
    s = re.sub(r"^---.*?---", "", s, flags=re.S)
    s = re.sub(FENCE + r".*?" + FENCE + r"|\$\$.*?\$\$", "", s, flags=re.S)
    return re.sub(r"\[\[([^\]|]+)(\|[^\]]*)?\]\]", r"\1", s).strip()


text = "\n\n".join(prose(n) for n in NOTES)
ids = tok(text, return_tensors="pt").input_ids[:, :N_TOKENS]
scored_text = tok.decode(ids[0, 1:])               # the first token has no context, so it is not scored
n_bytes, n_chars = len(scored_text.encode("utf-8")), len(scored_text)


def total_nll(m):                                   # sliding window: every token is scored once, with up to MAX_LEN-1 tokens of context
    n, total, count, prev_end = ids.size(1), 0.0, 0, 0
    for begin in range(0, n, STRIDE):
        end = min(begin + MAX_LEN, n); trg = end - prev_end
        x = ids[:, begin:end]; y = x.clone(); y[:, :-trg] = -100
        with torch.no_grad():
            loss = m(x, labels=y).loss.item()       # mean over the scored labels in this window
        k = int((y[:, 1:] != -100).sum())
        total += loss * k; count += k; prev_end = end
        if end == n: break
    assert count == n - 1
    return total, count


def report(label, bits_per_weight):
    t0 = time.time(); total, count = total_nll(model)
    print(f"{label:28s} {bits_per_weight:5.2f} bits/weight | perplexity {math.exp(total / count):7.3f} | "
          f"bits/token {total / count / math.log(2):.3f} | bits/byte {total / math.log(2) / n_bytes:.3f} | {time.time() - t0:.0f}s")


def set_weights(new):
    sd = dict(orig); sd.update(new); model.load_state_dict(sd)


print(f"{ids.size(1)} tokens, {n_chars} characters, {n_bytes} bytes scored | uniform guess: perplexity {model.config.vocab_size}")
report("original (bfloat16 file)", 16.0)

# 1) the weights of the GGUF file Ollama stores, turned back into numbers
names = {"token_embd": "model.embed_tokens", "output_norm": "model.norm"}
blk = {"attn_norm": "input_layernorm", "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
       "attn_output": "self_attn.o_proj", "ffn_norm": "post_attention_layernorm", "ffn_gate": "mlp.gate_proj",
       "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
new = {}
for t in gguf.GGUFReader(sys.argv[1]).tensors:
    p = t.name.split(".")
    hf = names[p[0]] + "." + p[1] if p[0] in names else f"model.layers.{p[1]}.{blk[p[2]]}.{p[3]}"
    new[hf] = torch.from_numpy(np.array(quants.dequantize(t.data, t.tensor_type), dtype=np.float32))
assert len(new) == 290 and all(new[k].shape == orig[k].shape for k in new)
set_weights(new); report("Ollama file (mixed types)", 6.345)

# 2) the same rounding rule applied to every weight matrix, one type at a time
for qt, bpw in ((T.Q8_0, 8.5), (T.Q5_0, 5.5), (T.Q4_0, 4.5)):
    new = {k: torch.from_numpy(quants.dequantize(quants.quantize(v.numpy(), qt), qt)) for k, v in orig.items() if v.dim() == 2}
    set_weights(new); report(f"all matrices as {qt.name}", bpw)
```

Run it as `python ppl_quant.py FILE` with the GGUF file Ollama stores for `qwen2.5:0.5b`. It printed (library warnings filtered out and not read):
```
2048 tokens, 7106 characters, 7118 bytes scored | uniform guess: perplexity 151936
original (bfloat16 file)     16.00 bits/weight | perplexity  13.483 | bits/token 3.753 | bits/byte 1.079 | 50s
Ollama file (mixed types)     6.34 bits/weight | perplexity  13.969 | bits/token 3.804 | bits/byte 1.094 | 39s
all matrices as Q8_0          8.50 bits/weight | perplexity  13.495 | bits/token 3.754 | bits/byte 1.080 | 40s
all matrices as Q5_0          5.50 bits/weight | perplexity  13.889 | bits/token 3.796 | bits/byte 1.092 | 37s
all matrices as Q4_0          4.50 bits/weight | perplexity  15.935 | bits/token 3.994 | bits/byte 1.149 | 39s
```
What it says, against the original's 13.483 (my arithmetic):
- **Q8_0 (8.5 bits): 13.495, up 0.09%.** Almost no change.
- **Q5_0 (5.5 bits): 13.889, up 3.0%.**
- **The Ollama file (6.3 bits on average): 13.969, up 3.6%.**
- **Q4_0 (4.5 bits): 15.935, up 18.2%.** A clear jump: the average surprise rose by 0.167 nats per token.
- **How to read it:** a uniform guess over this model's 151,936 tokens would score 151,936, and the model scores about 13.5, so even the worst variant is far better than guessing. Going from 16 to 8.5 bits costs almost nothing, 5.5 bits costs about 3%, and 4.5 bits costs about 18% on this text.
- **A puzzle I did not solve:** the Ollama file has more bits per weight on average (6.3) than uniform Q5_0 (5.5) but a slightly worse perplexity (13.969 against 13.889). It stores some tensors as Q4_K (4.5 bits), and the uniform run also quantizes the embedding table, which the file keeps at Q8_0, so the two are not the same recipe. My guess is that the 12 Q4_K tensors are the cause, but I did not test it.
- **Caveats:** one text of 7,106 characters, one model, one window setting, and the "bits per weight" for the uniform rows counts only the quantized matrices. The text is my own notes, which were written with an AI's help, so they may be easier to predict than ordinary writing. Perplexity says how well the model predicts this text and nothing about whether its answers are good. I did not test answers.
- **A tokenizer-free comparison, with a big caveat.** The model's 1.079 bits per byte on this text sits next to the 2.06 bits per character that the small model in [[Tiny GPT from Scratch]] reached on its own held-out notes. That is roughly 1.9 times fewer bits for the larger model, but the two texts are different samples (mine has the code, equations and flashcards removed, so it should be easier), so read it as a rough sign only. The two perplexity numbers (13.5 per token against 4.2 per character) cannot be compared at all.

**What came before, and how this improved on it:**
- **Before:** judge a language model by whether its single best guess was right (accuracy), or by raw loss numbers that are hard to picture.
- **Its limit:** accuracy ignores how much probability went to the right answer, and a loss of 2.8 means little to a reader.
- **What changed:** report $e^{\text{loss}}$ as an "effective number of equally likely choices". The measure was introduced for speech recognition at IBM in the 1970s (Jelinek and others; from memory, not checked).
- **How much better:** in the code below, a tiny model of 32 different characters that guesses evenly has a perplexity of 32, one that uses letter frequencies has 22.24, and one that also knows the previous letter has 16.91. Each number reads as "as unsure as picking among this many". **Caveat:** the text is only 856 characters (the Zen of Python), so these are toy values.

**Check it yourself:**

```python
import io, contextlib, math, collections, codecs
with contextlib.redirect_stdout(io.StringIO()): import this
text = codecs.decode(this.s, "rot13").lower()
tr, te = text[:len(text) // 2], text[len(text) // 2:]
V = sorted(set(text)); uni = collections.Counter(tr); big = collections.defaultdict(collections.Counter)
for a, b in zip(tr, tr[1:]): big[a][b] += 1
def ppl(prob):
    pairs = list(zip(te, te[1:])); return math.exp(-sum(math.log(prob(a, b)) for a, b in pairs) / len(pairs))
k = 0.5
p_uni = lambda a, b: (uni[b] + k) / (len(tr) + k * len(V))
p_big = lambda a, b: (big[a][b] + k) / (sum(big[a].values()) + k * len(V))
print(f"{len(V)} different characters; model trained on {len(tr)} characters, tested on the other {len(te)}")
print("perplexity, guessing evenly:      ", len(V))
print("perplexity, letter frequencies:   ", round(ppl(p_uni), 2))
print("perplexity, knows the last letter:", round(ppl(p_big), 2))
```

## Related
- [[Loss Function]] · [[Probability]] · [[Softmax]] · [[LLM]] · [[Tokenization]] · [[Fine-tuning]] · [[RLHF]] · [[Overfitting]] · [[Gradient Descent]] · [[Transformer]] · [[Evaluation Metrics]] · [[Entropy]] · [[KL Divergence]] · [[Maximum Likelihood]] · [[Pre-training]] · [[Scaling Laws]] · [[Quantization]] · [[Byte-Pair Encoding]] · [[Decoding Strategies]] · [[Masked Diffusion Language Model]] · [[Cross-Entropy Loss]] · [[GGUF]] · [[llama.cpp]] · [[Ollama]] · [[Tiny GPT from Scratch]]

## Sources
- Jurafsky & Martin, *Speech and Language Processing* (free online), chapter on n-gram language models and perplexity
- Shannon, *A Mathematical Theory of Communication*, 1948 (entropy)
- Hugging Face documentation, *Perplexity of fixed-length models* (https://huggingface.co/docs/transformers/en/perplexity), read on 17 November 2026 through a page summary, not word for word: the definition, the sliding-window method and the GPT-2 numbers (19.44 and 16.44). I did not rerun them.
- Wikipedia, *Perplexity* (https://en.wikipedia.org/wiki/Perplexity), read through a page summary on 17 November 2026: perplexity as 2 to the entropy, the "effective number of choices" reading and its limits. That page does not give the IBM history, so the "1970s, Jelinek" line above is still from memory.
- My own run: Qwen2.5-0.5B-Instruct (Apache-2.0), the Ollama GGUF file and the official `gguf` Python package 0.18.0, on this PC (Python 3.9.6, torch 2.8.0+cpu). The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation Perplexity:::A "how confused is the model?" score. A perplexity of 4 means the model was about as unsure as if it had to pick among 4 equally likely tokens at every step. Lower is better.

#flashcards/ml/formula #flashcards/topic/evaluation Write perplexity and explain each symbol.
?
$$\text{PPL}=\exp\!\Big(-\frac1N\sum_{i=1}^{N}\ln p\big(x_i\mid x_{<i}\big)\Big)$$
- $N$: the number of tokens scored. $i$ counts through them.
- $x_i$: the real $i$-th token. $x_{<i}$: all tokens before it (the context).
- $p(x_i\mid x_{<i})$: the chance the model gave to the real token. This is the only thing measured.
- $\ln p$ is at most 0: 0 for a certain prediction, very negative for a surprising one. The minus sign flips it into a positive "surprise".
- $\frac1N\sum$ averages the surprise, and $\exp$ turns it back into "number of equally likely choices".

#flashcards/ml/example #flashcards/topic/evaluation The real tokens got chances 0.5, 0.25, 0.125. What is the perplexity? What if they got 0.9, 0.8, 0.7?::Logs sum to $-4.159$, average surprise $1.386$, $\text{PPL}=e^{1.386}=4.0$. With 0.9, 0.8, 0.7: average surprise $0.228$, $\text{PPL}=1.257$, close to the best possible value 1.

#flashcards/ml/formula #flashcards/topic/evaluation Write perplexity in terms of average cross-entropy and explain each symbol.
?
$$\text{PPL}=e^{L},\qquad L=-\frac1N\sum_{i}\ln p(x_i\mid x_{<i})\quad\text{(in nats)}$$
- $L$: the average cross-entropy per token, in nats because it uses the natural log. With $\log_2$ it is in bits and $\text{PPL}=2^{L_{\text{bits}}}$. Both give the same perplexity.
- $\text{PPL}$: the perplexity. $e\approx2.718$.
- $L$ is what gradient descent lowers during training, so perplexity is the same quantity in a readable form.
- A drop of $L$ by 0.1 nats multiplies perplexity by $e^{-0.1}\approx0.905$, a 9.5% reduction.

#flashcards/ml/example #flashcards/topic/evaluation A model guesses uniformly over a 1000-token vocabulary. What is its perplexity? What if $L=0$?::$L=-\ln0.001=6.908$ and $\text{PPL}=e^{6.908}=1000$, exactly the vocabulary size (the worst sensible score). With $L=0$ the model always gives the real token chance 1, so $\text{PPL}=1$ (perfect, essentially only for memorised text).

#flashcards/ml/cloze #flashcards/topic/evaluation Lower perplexity is ==better==, because the model was ==less surprised== by the real text.

#flashcards/ml/concept #flashcards/topic/evaluation What is the relation between perplexity and the training loss?::Perplexity is $e^{\text{loss}}$ (loss in nats), the same number in a friendlier form.

#flashcards/ml/formula #flashcards/topic/evaluation Write bits per byte in terms of perplexity, and explain each symbol.
?
$$\text{bits per byte}=\frac{1}{B}\sum_{i=1}^{N}-\log_2 p\big(x_i\mid x_{<i}\big)=\frac{N\,\log_2\text{PPL}}{B}$$
- $N$: the number of tokens scored. $B$: the number of bytes of text those tokens cover.
- $-\log_2 p(x_i\mid x_{<i})$: the surprise of one token, in bits. $\log_2\text{PPL}$ is the average surprise per token.
- Perplexity depends on how the text is split into tokens, but a byte is the same for every model, so bits per byte compares models with different tokenizers (on the same text).

#flashcards/ml/example #flashcards/topic/evaluation A model has perplexity 8 on 100 tokens that cover 400 bytes. What is the bits per byte?::log2 8 = 3 bits per token, so 100 × 3 = 300 bits, and 300 / 400 = 0.75 bits per byte. In my real run: 2047 × 3.753 / 7118 = 1.079.

#flashcards/ml/example #flashcards/topic/evaluation Qwen2.5-0.5B on 2,048 tokens of this vault's prose: what was the perplexity with the original weights, with Q8_0, Q5_0 and Q4_0?::Original 13.483, Q8_0 13.495 (up 0.09%), Q5_0 13.889 (up 3.0%), Q4_0 15.935 (up 18.2%) (my arithmetic). One text, one model, one window setting.

#flashcards/ml/example #flashcards/topic/evaluation The Ollama GGUF file averages 6.3 bits per weight. What perplexity did it get against the original's 13.483?::13.969, up 3.6%. That is slightly worse than uniform Q5_0 (13.889, 5.5 bits). I guessed the 12 Q4_K tensors were the cause, but did not test it.

#flashcards/ml/concept #flashcards/topic/evaluation Why can you not compare perplexity numbers from two models with different tokenizers?::Perplexity is per token, and different tokenizers cut the same text into different numbers of pieces. Use bits per byte (or per character) on the same text instead.

#flashcards/ml/concept #flashcards/topic/evaluation Why is cutting a long text into separate chunks a poor way to measure perplexity, and what does the Hugging Face guide report?::Most tokens then have little context, so the score is higher (worse). For GPT-2 large on WikiText-2 it reports 19.44 without overlap and 16.44 with a stride of 512 (from a page summary, not rerun by me).

#flashcards/ml/concept #flashcards/topic/evaluation What does a low perplexity on quantized weights not tell you?::Only how well the model predicts that text. It says nothing about whether the answers are good, and the test used one text of 7,106 characters written with an AI's help, which may be easier than ordinary writing.
