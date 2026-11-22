---
tags: [LLM, inference, C++, CPU, quantization, local, ggml]
status: growing
level: 4
tier: 3
---

# llama.cpp

**llama.cpp** is a program, written in C and C++, that runs [[LLM]]s on ordinary computers, with or without a graphics card. It reads models stored as [[GGUF]] files, shrinks their numbers with [[Quantization]] so they fit in normal memory, and is the engine behind many local chat tools.

## 🟢 1. Simple (age 12)
Big AI models were made for expensive computers. llama.cpp is a clever program that squeezes the model's numbers into less space and then runs it on the kind of computer you might already own. It is like packing a huge wardrobe into a small suitcase by rolling the clothes tightly: you lose a tiny bit of neatness but everything fits.

**Example:** A model that needs a 2 GB box when stored the plain way fits in a 0.4 GB box after the squeeze. A laptop with no special chip can then read it fast enough to chat.

## 🔵 2. How it works
1. **A person gets a model file in the GGUF format.** It holds the numbers, the settings and the word list in one file ([[GGUF]]).
2. **The numbers are squeezed in small groups.** For each group of 32 numbers, llama.cpp stores one scale number and then a small whole number for each weight ([[Quantization]]).
3. **When the model runs, it turns the small numbers back** into near-original ones as it goes, group by group.
4. **The program is written in plain C and C++**, with the maths code tuned for the computer's own chip. It builds on a small maths library called ggml and does not need Python or [[PyTorch]].
5. **Writing one word means reading the whole model from memory once** ([[LLM Inference]]). Since a squeezed model is a few times smaller, that reading is a few times faster, and the reading is what usually limits the speed on a laptop.
6. **You use it in a few ways:** a command-line chat, a small web chat page, or a server that other programs can talk to. [[Ollama]] sits on top of it and hides the details.
7. **The chat format still matters.** The program needs the model's [[Chat Template]] to build the text the model expects.

![[llama-cpp-web-interface.png]]
*llama.cpp's built-in web chat page, called "llama-ui". The user asks "What is the best wiki?", a collapsible "Reasoning" box shows the model's thinking, and the answer follows. The model shown at the bottom right is Qwen3.5 4B. Screenshot by The ggml authors, [Commons file page](https://commons.wikimedia.org/wiki/File:Llama.cpp_screenshot.png), MIT licence.*

## 🟠 3. Technical detail
- **What the project says it is** (its README, read through a summary): an inference project written in C and C++, built on the ggml library, whose goal is LLM inference with "minimal setup and state-of-the-art performance" on a wide range of hardware. It can be installed from a script, Docker, prebuilt files or source, and its example downloads a GGUF model directly from Hugging Face.
- **Block quantization.** The simplest type, Q8_0, cuts the weights into blocks of 32, finds the largest absolute value in each block, and sets one scale from it. Each weight is then stored as one signed byte (equation 1).

$$\hat x_i=d\,q_i,\qquad d=\frac{\max_j|x_j|}{127},\qquad q_i=\operatorname{round}\!\left(\frac{x_i}{d}\right)$$

**Where:**
- $x_i$: one original weight in a block of 32. $j$ runs over the same block.
- $d$: the block's scale, stored once per block as a 16-bit float (2 bytes). The 127 is the largest whole number a signed byte can hold on the positive side.
- $q_i$: the stored whole number, from -127 to 127, one byte per weight.
- $\hat x_i$: the weight as it is read back when the model runs.

**Matters:** one block costs 2 + 32 = 34 bytes for 32 weights, which is 8.5 bits per weight. The error on each weight is at most half a step, $d/2$, so a block with one big weight gets a large step and loses detail on its small weights. Without the shared scale, a byte could not cover both tiny and big weights.

**Example (tiny numbers):** a block holds 0.5, -1.0, 0.25 (and 29 zeros). The largest absolute value is 1.0, so $d=1/127=0.00787$. Then $q=\operatorname{round}(0.5/0.00787)=\operatorname{round}(63.5)=64$, $q=-127$ for -1.0, and $q=\operatorname{round}(31.75)=32$. Read back: $64\times0.00787=0.5039$, $-127\times0.00787=-0.9999$, $32\times0.00787=0.2520$. The worst error is 0.0039, at most $d/2$, about 0.4% of the biggest weight.

- **Other types are variations.** Q5_0 uses 5 bits per weight and 22 bytes per block of 32. The "K" types (Q4_K, Q6_K) use blocks of 256 with scales for smaller sub-blocks. I describe the K types only from memory and did not read their code.
- **How good is it on real weights?** I compared the weights in the Ollama GGUF file of Qwen2.5-0.5B-Instruct with the original `model.safetensors` (bfloat16), one example tensor per type. Relative error is $\lVert W-\hat W\rVert/\lVert W\rVert$.

**`quant_demo.py`**
```python
"""Block quantization by hand (Q8_0) against the official gguf code, then the error on real weights."""
import numpy as np, sys
import gguf
from gguf import quants, GGMLQuantizationType as T
from safetensors import safe_open

# 1) my own Q8_0: blocks of 32 numbers share one scale d = max|x| / 127; each number is stored as round(x / d) in one byte
def q8_0(x):
    b = x.reshape(-1, 32).astype(np.float32)
    d = np.abs(b).max(axis=1) / 127                       # the scale, kept in float32 while rounding
    q = np.rint(b * np.where(d == 0, 0, 1 / np.where(d == 0, 1, d))[:, None]).astype(np.int8)
    return d.astype(np.float16), q                        # ...and stored as float16

rng = np.random.default_rng(0)
x = rng.standard_normal(32 * 1000).astype(np.float32)
d, q = q8_0(x)
mine = (d.astype(np.float32)[:, None] * q).reshape(-1)
official = quants.dequantize(quants.quantize(x, T.Q8_0), T.Q8_0)
print("my Q8_0 equals the official Q8_0:", np.array_equal(mine, official.reshape(-1)),
      "| max error", float(np.abs(x - mine).max()), "| bytes per 32 numbers", 2 + 32)

# 2) a tiny block done by eye
b = np.array([0.5, -1.0, 0.25] + [0.0] * 29, dtype=np.float32)
d, q = q8_0(b)
print("tiny block: d =", float(d[0]), "| q =", q[0][:3], "| back:", (float(d[0]) * q[0][:3]).round(4))

# 3) real weights: GGUF (quantized) against the original safetensors file (bfloat16)
r = gguf.GGUFReader(sys.argv[1])
st = safe_open("qwen2.5-0.5b-instruct/model.safetensors", framework="pt")
tmap = {"attn_q": "self_attn.q_proj", "attn_v": "self_attn.v_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj"}
seen = set()
print(f"{'tensor':24s} {'type':5s} {'relative error':>14s} {'worst weight error':>18s}")
for t in r.tensors:
    parts = t.name.split(".")
    if parts[0] != "blk" or parts[2] not in tmap or parts[3] != "weight" or t.tensor_type.name in seen:
        continue
    seen.add(t.tensor_type.name)
    w = st.get_tensor(f"model.layers.{parts[1]}.{tmap[parts[2]]}.weight").float().numpy()
    wq = quants.dequantize(t.data, t.tensor_type)
    assert wq.shape == w.shape, (wq.shape, w.shape)
    print(f"{t.name:24s} {t.tensor_type.name:5s} {np.linalg.norm(w - wq) / np.linalg.norm(w):14.4f} {np.abs(w - wq).max():18.5f}")
```
Run it as `python quant_demo.py FILE` with the path of the GGUF file. It printed:
```
my Q8_0 equals the official Q8_0: True | max error 0.0167158842086792 | bytes per 32 numbers 34
tiny block: d = 0.00787353515625 | q = [  64 -127   32] | back: [ 0.5039 -0.9999  0.252 ]
tensor                   type  relative error worst weight error
blk.0.ffn_down.weight    Q6_K          0.0188            0.00458
blk.0.ffn_up.weight      Q5_0          0.0439            0.00867
blk.0.attn_v.weight      Q8_0          0.0058            0.00044
blk.11.ffn_down.weight   Q4_K          0.0757            0.02064
```
What to look for: my 12-line Q8_0 is identical, number for number, to the official code (my first draft was not, because it rounded with the 16-bit scale instead of the full-precision one, and I fixed that). The error grows as the bits fall: 0.6% for Q8_0, 1.9% for Q6_K, 4.4% for Q5_0 and 7.6% for Q4_K. This is one tensor per type, not an average over the model.

**What I did not run: llama.cpp itself.** I did not build it or download its program, so every number about speed in this note comes from [[Ollama]], whose README lists llama.cpp as a backend (from a page summary). I did not check which version of llama.cpp my Ollama contains.

## 🔴 4. Deep / formal
- **Why decoding is limited by memory.** To write one token, each weight is read about once and used in one multiply and add (a simplification, from memory). On a laptop, reading is the slow part, so fewer bytes per weight means more tokens a second. My measurement on this PC: float32 transformers was 6.0 to 6.3 tokens/s, and the quantized model through Ollama was 39 to 50 tokens/s ([[Local CPU Chatbot]]). I did not separate the effect of smaller weights from the effect of a tuned C++ engine.
- **Error against size.** Fewer bits means a larger step and more error. The relative errors above (0.6% to 7.6%) are for the weights only. What they do to the model's predictions I measured separately with [[Perplexity]] on one text: 13.483 for the original weights, 13.495 with Q8_0 (up 0.09%), 13.889 with Q5_0 (up 3.0%) and 15.935 with Q4_0 (up 18.2%). I did not test the model's answers, so I claim nothing about their quality.
- **Mixed types are a design choice.** The file I read gives the embedding table 8.5 bits per weight and most layers 5.5 bits ([[GGUF]]), so the "4-bit" name hides an average of 6.345 bits per weight.
- **Hardware.** The README says it targets a range of hardware. I did not read the list of supported chips, and I used only the CPU here.
- **Limits of this note:** I did not test the server mode, the grammar feature, the GPU back ends or building from source.

**What came before, and how this improved on it:**
- **Before:** running a model like LLaMA meant PyTorch, usually a big graphics card, and the weights in 16 or 32 bits.
- **Its limit:** a 7-billion-weight model in 16 bits is about 14 GB (my arithmetic: 7 billion × 2 bytes), which I believe was more than most home graphics cards held in 2023 (from memory). On a CPU, a Python stack was slow (also from memory).
- **What changed:** a from-scratch C and C++ implementation with no outside dependencies and with quantized weights. The project's first commit is dated 10 March 2023 and its author is Georgi Gerganov (from a search summary of Wikipedia and the commit log; I did not open either). The ggml library came from the same author earlier (September 2022, from a less reliable source).
- **How much better:** on this PC the same 0.5-billion-weight model went from 1.98 GB (float32) to a 0.40 GB GGUF file (5.04 times smaller, my arithmetic) and from 6.0 to about 39 to 50 tokens/s when run through Ollama. I did not run llama.cpp directly, so I cannot attribute the speed-up to it alone.

## Related
[[GGUF]] · [[Quantization]] · [[Ollama]] · [[LLM Inference]] · [[Chat Template]] · [[LLM]] · [[GPU]] · [[Local CPU Chatbot]] · [[PyTorch]] · [[KV Cache]]

## Sources
- The llama.cpp README (https://github.com/ggml-org/llama.cpp), read on 17 November 2026 through a page summary, not word for word. It says what the project is and how it installs. It does not say the origin dates, which come from a web search summary only (Wikipedia and the commit log, not opened).
- The official `gguf` Python package, version 0.18.0 (it is part of the ggml project), used for the Q8_0 check and for reading the weights.
- The Qwen2.5-0.5B-Instruct model (Apache-2.0) from Hugging Face and the Ollama copy, on this PC (Python 3.9.6, numpy 1.24.4).
- Image: "Llama.cpp screenshot.png" by The ggml authors, MIT, https://commons.wikimedia.org/wiki/File:Llama.cpp_screenshot.png. I viewed the image before writing its caption.
- From memory: the K types, the "weights read once per token" rule, what the earlier way of running LLaMA needed, and the claim about home graphics memory. The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving llama.cpp:::A program written in C and C++ that runs language models from GGUF files on ordinary hardware, with or without a GPU, using quantized weights. It builds on the ggml library.

#flashcards/ml/image #flashcards/topic/llm-serving ![[llama-cpp-web-interface.png]]
What does this picture show?
?
llama.cpp's built-in web chat page, "llama-ui". The user asks "What is the best wiki?", a collapsible Reasoning box shows the model's thinking, and the answer follows. The model shown is Qwen3.5 4B. Screenshot by The ggml authors, MIT licence (Wikimedia Commons).

#flashcards/ml/concept #flashcards/topic/llm-serving Why does a smaller quantized model run faster on a laptop?::Writing one token reads about every weight once (a simplification), and on a laptop reading memory is the slow part, so fewer bytes per weight means more tokens per second.

#flashcards/ml/formula #flashcards/topic/llm-serving Write Q8_0 block quantization and explain each symbol.
?
$$\hat x_i=d\,q_i,\qquad d=\frac{\max_j|x_j|}{127},\qquad q_i=\operatorname{round}\!\left(\frac{x_i}{d}\right)$$
- $x_i$: one original weight in a block of 32. $j$ runs over the same block.
- $d$: the block's scale, stored once per block as a 16-bit float. 127 is the largest positive value of a signed byte.
- $q_i$: the stored whole number from -127 to 127, one byte per weight.
- $\hat x_i$: the weight as read back at run time. A block costs 2 + 32 = 34 bytes (8.5 bits per weight), and each weight's error is at most $d/2$.

#flashcards/ml/example #flashcards/topic/llm-serving A Q8_0 block holds 0.5, -1.0 and 0.25 (plus zeros). What are d, the stored numbers and the values read back?::$d=1/127=0.00787$. The stored numbers are 64, -127 and 32. Read back: 0.5039, -0.9999 and 0.2520. The worst error is 0.0039, at most $d/2$.

#flashcards/ml/example #flashcards/topic/llm-serving What did the check on the real Qwen2.5-0.5B weights show about error against bits?::One example tensor per type, relative to the original: Q8_0 0.6%, Q6_K 1.9%, Q5_0 4.4%, Q4_K 7.6%. Error grows as bits fall. It is not an average over the model, and it says nothing about answer quality.

#flashcards/ml/example #flashcards/topic/llm-serving Why did my first hand-written Q8_0 differ from the official one?::It rounded with the 16-bit scale. The official code rounds with the full-precision scale and only then stores it as 16 bits. After that fix the two agreed number for number.

#flashcards/ml/example #flashcards/topic/llm-serving Qwen2.5-0.5B went from 1.98 GB as float32 to a GGUF file of 0.40 GB. How many times smaller, and what did the speed do?::About 5.04 times smaller (my arithmetic). Speed went from 6.0 to 6.3 tokens/s (transformers float32) to about 39 to 50 tokens/s (Ollama). I did not run llama.cpp directly or split the two causes.

#flashcards/ml/concept #flashcards/topic/llm-serving What did I not run for this note?::llama.cpp itself (I did not build it or download its program), its server mode, GPU back ends, and the effect of quantization on answer quality. Speeds come from Ollama.

#flashcards/ml/concept #flashcards/topic/llm-serving When did llama.cpp begin, and how sure is that?::Its first commit is dated 10 March 2023, by Georgi Gerganov (from a search summary of Wikipedia and the commit log, which I did not open).
