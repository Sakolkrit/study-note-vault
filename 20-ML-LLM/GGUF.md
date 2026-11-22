---
tags: [LLM, file-format, quantization, inference, llama.cpp, local]
status: growing
level: 4
tier: 3
---

# GGUF

**GGUF** is a file format that packs a whole language model into one file: its numbers (the weights, usually shrunk by [[Quantization]]), its settings, and even its word list. It is the format that [[llama.cpp]] and [[Ollama]] load.

## 🟢 1. Simple (age 12)
A model is a huge pile of numbers. To use it, a program also needs to know what the pile means: how many layers, what the words are, how the numbers were squeezed. GGUF is a box that holds the numbers and a neat label list for everything inside, so one file is enough to run the model.

**Example:** A board game in one box with the rules printed on the lid, instead of a loose pile of pieces and a separate rule book you might lose.

## 🔵 2. How it works
1. **The file starts with the four letters `GGUF`.** That is how a program knows what it has opened.
2. **Next come a version number and two counts:** how many tensors (blocks of numbers) there are, and how many settings.
3. **Then the settings, called metadata.** Each is a name and a value, such as `qwen2.block_count = 24` or the chat layout ([[Chat Template]]). The values have types, so a program knows if it is reading a number or a word.
4. **Then a list of the tensors:** each has a name, a shape, a number type, and where its numbers start in the file.
5. **Then the numbers themselves,** lined up at fixed steps so a program can map the file straight into memory without copying it.
6. **The numbers are usually quantized.** Instead of 16 or 32 bits for each weight, small groups of weights are stored with fewer bits and one shared scale ([[Quantization]]), so the file is much smaller.
7. **One file, nothing else needed.** The word list for the [[Tokenization]] step is inside too.

No picture: I searched Wikimedia Commons for GGUF and found no free, on-topic image, so I added none.

## 🟠 3. Technical detail
- **Header, in order** (from the format document): magic bytes `GGUF`, a 32-bit version (3 in the spec I read), a 64-bit tensor count, a 64-bit count of metadata pairs, the metadata pairs, the tensor infos, zero padding up to the alignment, then the tensor data. Numbers are little-endian by default, and the default alignment is 32 bytes.
- **Metadata value types** have ids 0 to 12: unsigned and signed integers of 8, 16, 32 and 64 bits, 32- and 64-bit floats, bool, string and array.
- **Tensor names follow a pattern:** `token_embd.weight`, then per layer `blk.N.attn_q.weight`, `blk.N.ffn_up.weight` and so on, then `output_norm.weight`.
- **Each tensor has its own type.** The spec says the type ids mirror a list in the ggml source and "gain new values over time". In the file I read, five types appear, which I listed below.
- **Size rule.** Quantized types work in blocks: every block of `B` weights is stored in a fixed number of bytes. So a tensor's size is easy to compute from its shape and type (equation 1).

$$\text{bytes}=\frac{n}{B}\times s$$

**Where:**
- $n$: the number of weights in the tensor (the product of its shape).
- $B$: how many weights share one block (32 for Q5_0 and Q8_0, 256 for Q4_K and Q6_K; 1 for plain float32).
- $s$: how many bytes one block takes in the file.
- $\text{bytes}$: the size of the tensor's data in the file.

**Matters:** it is why a program can find every tensor without reading them all, and why a model's file size is just a sum. If $n$ is not a whole number of blocks the format cannot store the tensor in that type.

**Example (tiny numbers):** 64 weights stored as Q5_0 ($B=32$, $s=22$ bytes). Blocks: $64/32=2$. Bytes: $2\times22=44$. The same 64 weights as float16 take $64\times2=128$ bytes, so Q5_0 is about 2.9 times smaller. On the real file, `token_embd.weight` has $896\times151{,}936=136{,}134{,}656$ weights in Q8_0 ($B=32$, $s=34$), so $4{,}254{,}208$ blocks and $144{,}643{,}072$ bytes.

- **Bits per weight** is the fairest way to compare formats (equation 2).

$$b=\frac{8\,s}{B}\qquad\text{and for a whole file}\qquad b=\frac{8\,S}{N}$$

**Where:**
- $b$: bits used per weight, on average.
- $s$ and $B$: bytes per block and weights per block, as above. The 8 turns bytes into bits.
- $S$: all the bytes of tensor data in the file. $N$: the number of weights in the whole model.

**Matters:** it says how much a format shrinks the model, whatever the model's size. The header and metadata are not counted in $S$.

**Example (tiny numbers):** Q5_0: $8\times22/32=5.5$ bits per weight. Q8_0: $8\times34/32=8.5$. Whole file: $S=391{,}859{,}712$ bytes and $N=494{,}032{,}768$ weights, so $b=8\times391{,}859{,}712/494{,}032{,}768=6.345$ bits per weight. Float16 uses 16, so this file is 2.52 times smaller than float16 and 5.04 times smaller than float32.

**`gguf_peek.py`** (my own reader, standard library only)
```python
"""Read the header of a GGUF file with only the standard library. Usage: python gguf_peek.py FILE"""
import struct, sys, collections

F = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class R:
    def __init__(self, f): self.f = f
    def get(self, fmt): return struct.unpack(fmt, self.f.read(struct.calcsize(fmt)))[0]
    def string(self): return self.f.read(self.get("<Q")).decode("utf-8")
    def value(self, t):
        if t == 8: return self.string()
        if t == 9:
            et, n = self.get("<I"), self.get("<Q")
            return [self.value(et) for _ in range(n)]
        return self.get(F[t])


def peek(path):
    with open(path, "rb") as f:
        r = R(f)
        assert f.read(4) == b"GGUF", "not a GGUF file"
        version, n_tensors, n_kv = r.get("<I"), r.get("<Q"), r.get("<Q")
        meta = {}
        for _ in range(n_kv):
            k = r.string(); meta[k] = r.value(r.get("<I"))
        tensors = []
        for _ in range(n_tensors):
            name = r.string(); dims = [r.get("<Q") for _ in range(r.get("<I"))]
            tensors.append((name, dims, r.get("<I"), r.get("<Q")))   # name, shape, ggml type id, offset
    return version, meta, tensors


if __name__ == "__main__":
    version, meta, tensors = peek(sys.argv[1])
    print("version", version, "| metadata keys", len(meta), "| tensors", len(tensors))
    for k, v in meta.items():
        print(f"  {k} = {str(v)[:60]!r}" if not isinstance(v, list) else f"  {k} = list of {len(v)}")
    by_type = collections.Counter(t[2] for t in tensors)
    print("tensor types (ggml type id: count):", dict(by_type))
```
I ran it on the file Ollama stores for `qwen2.5:0.5b` (the large "model" layer, 397,807,936 bytes, found through the manifest in the Ollama models folder). Most of the output (the lines of metadata):
```
version 3 | metadata keys 34 | tensors 290
  general.architecture = 'qwen2'
  general.name = 'Qwen2.5 0.5B Instruct'
  general.license = 'apache-2.0'
  qwen2.block_count = '24'
  qwen2.context_length = '32768'
  qwen2.embedding_length = '896'
  qwen2.feed_forward_length = '4864'
  qwen2.attention.head_count = '14'
  qwen2.attention.head_count_kv = '2'
  general.file_type = '15'
  tokenizer.ggml.model = 'gpt2'
  tokenizer.ggml.tokens = list of 151936
  tokenizer.ggml.merges = list of 151387
  tokenizer.ggml.eos_token_id = '151645'
  tokenizer.chat_template = "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if"
  general.quantization_version = '2'
tensor types (ggml type id: count): {8: 13, 0: 121, 14: 12, 6: 132, 12: 12}
```
(I removed some lines here; the full run printed all 34 keys.) What to look for: the word list and the merge rules of the tokenizer are inside the file, and so is the chat template.

**`gguf_sizes.py`** (checks equation 1 and 2 against the official `gguf` package, which I installed with `pip install gguf`, version 0.18.0)
```python
"""Check the GGUF size rules on a real file, with my own header reader and the official `gguf` package."""
import sys, collections
import gguf
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType as T
from gguf_peek import peek

path = sys.argv[1]
version, meta, tensors = peek(path)                      # my reader: (name, shape, type id, offset)
official = gguf.GGUFReader(path)
print("tensors: mine", len(tensors), "| official", len(official.tensors))

tot_bytes = tot_elems = 0
per_type = collections.OrderedDict()
for (name, dims, tid, off), o in zip(tensors, official.tensors):
    n = 1
    for d in dims: n *= d
    block, size = GGML_QUANT_SIZES[T(tid)]
    mine = n // block * size                             # bytes = elements / block size * bytes per block
    assert mine == int(o.n_bytes) and n == int(o.n_elements), name
    tot_bytes += mine; tot_elems += n
    e = per_type.setdefault(T(tid).name, [0, 0, 0, block, size]); e[0] += 1; e[1] += n; e[2] += mine
print("my size rule matches the official reader for all tensors")
print(f"{'type':6s} {'tensors':>7s} {'weights':>12s} {'bytes':>12s} {'block':>5s} {'bytes/block':>11s} {'bits/weight':>11s}")
for k, (c, n, b, block, size) in per_type.items():
    print(f"{k:6s} {c:7d} {n:12,d} {b:12,d} {block:5d} {size:11d} {8 * size / block:11.3f}")
print(f"total: {tot_elems:,} weights, {tot_bytes:,} bytes in tensors -> {8 * tot_bytes / tot_elems:.3f} bits per weight")
import os
print(f"file size {os.path.getsize(path):,} bytes; everything that is not tensor data (header, metadata, tokenizer, padding): {os.path.getsize(path) - tot_bytes:,}")
print("ratio to float32:", round(4 * tot_elems / tot_bytes, 2), "| to float16:", round(2 * tot_elems / tot_bytes, 2))
for name, dims, tid, off in tensors:
    if name in ("token_embd.weight", "blk.0.attn_q.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight", "blk.0.attn_v.weight", "output_norm.weight"):
        print(f"  {name:22s} shape {dims} type {T(tid).name}")
```
Output:
```
tensors: mine 290 | official 290
my size rule matches the official reader for all tensors
type   tensors      weights        bytes block bytes/block bits/weight
Q8_0        13  137,510,912  146,105,344    32          34       8.500
F32        121       71,552      286,208     1           4      32.000
Q6_K        12   52,297,728   42,900,480   256         210       6.562
Q5_0       132  251,854,848  173,150,208    32          22       5.500
Q4_K        12   52,297,728   29,417,472   256         144       4.500
total: 494,032,768 weights, 391,859,712 bytes in tensors -> 6.345 bits per weight
file size 397,807,936 bytes; everything that is not tensor data (header, metadata, tokenizer, padding): 5,948,224
ratio to float32: 5.04 | to float16: 2.52
  token_embd.weight      shape [896, 151936] type Q8_0
  blk.0.ffn_down.weight  shape [4864, 896] type Q6_K
  blk.0.ffn_up.weight    shape [896, 4864] type Q5_0
  blk.0.attn_q.weight    shape [896, 896] type Q5_0
  blk.0.attn_v.weight    shape [896, 128] type Q8_0
  output_norm.weight     shape [896] type F32
```
What to look for: my size rule gives exactly the official byte count for all 290 tensors, and the file mixes five types.

## 🔴 4. Deep / formal
- **"Q4_K_M" is a name for a mix, not a width.** Ollama labels this model Q4_K_M and the file's `general.file_type` is 15. But by count of weights, 51% are Q5_0, 28% Q8_0 (the word-embedding table is Q8_0), 11% Q6_K and only 11% Q4_K, and 0.01% are float32 (the small norm and bias tensors). The average is 6.345 bits per weight, not 4. Across all 169 two-dimensional tensors, the 145 whose first dimension is 896 are Q5_0 or Q8_0, and the 24 whose first dimension is 4864 are Q6_K or Q4_K (I counted them). I did not check why (a guess from memory: the K types need rows in whole blocks of 256, and 896 is not a multiple of 256, but I did not confirm this).
- **A small model quantizes worse than a large one in this respect** (from memory, not tested): the embedding table is a large share of a 0.5-billion-parameter model, and here it is stored at the highest quality (Q8_0).
- **What is not tensor data.** 5,948,224 bytes, 1.5% of the file, hold the header, the 34 metadata pairs, the tokenizer (151,936 words and 151,387 merge rules) and padding.
- **mmap.** The data is aligned so that a program can map the file into memory and use the numbers in place instead of copying them. The spec lists this as a design goal. I did not test loading speed.
- **Big-endian** files exist, and the spec says there is no way yet to tell from the file which one you have. All values in this file read correctly as little-endian.
- **What I did not check:** other quantization types (Q2, Q3, the "IQ" types), split files, and the GGUF writer.

**What came before, and how this improved on it:**
- **Before:** the spec says GGUF succeeds three older formats: GGML (no version number), GGMF (one version) and GGJT (tensor data aligned so it can be mapped into memory).
- **Its limit:** per the spec, the older formats did not say which model architecture the file held, adding or removing a setting was a breaking change, every architecture needed its own conversion script, and the quantization version was packed into one field in a fragile way.
- **What changed:** a typed key-value list for all settings, so new keys do not break old readers, plus the architecture name in the file and the tokenizer inside it. The file I read has the architecture (`qwen2`), 34 typed pairs, the tokenizer and the chat template. (From a search summary I did not check: GGUF appeared in August 2023.)
- **How much better:** my Hugging Face download for the same model was 8 files; the GGUF is 1 file with all of that inside. My own reader is 40 lines of standard-library Python and agreed with the official package on all 290 tensors. The size of the model, from 1.98 GB as float32 to 0.40 GB, is mostly quantization and not the file format itself. What the quantization costs in [[Perplexity]] on one text: 13.483 for the original weights against 13.969 for the weights of this file (up 3.6%).

## Related
[[Quantization]] · [[llama.cpp]] · [[Ollama]] · [[Chat Template]] · [[Tokenization]] · [[LLM Inference]] · [[Local CPU Chatbot]] · [[Mixed Precision Training]] · [[Safetensors]] · [[ONNX]] · [[MLOps]]
## Sources
- The GGUF format document in the ggml repository (`docs/gguf.md`, https://github.com/ggml-org/ggml/blob/master/docs/gguf.md), read on 17 November 2026 through a page summary, not word for word: the header order, the value type ids, alignment, endianness and the older formats.
- The official `gguf` Python package, version 0.18.0, used to check my reader and the block sizes (`GGML_QUANT_SIZES`).
- The Ollama `qwen2.5:0.5b` file, read on this PC (Python 3.9.6). The 34 keys, 290 tensors and all the numbers above are from that run.
- From a search summary only: the August 2023 date for GGUF. "K" quantization names and the guess about 256-wide rows are from memory.
- No image was found. The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving GGUF:::A single-file format for a language model: weights (often quantized), typed metadata such as the settings and the tokenizer, and a list of the tensors. It is loaded by llama.cpp and Ollama.

#flashcards/ml/concept #flashcards/topic/llm-serving What are the parts of a GGUF file, in order?::The magic bytes GGUF, a version, the tensor count, the metadata count, the metadata key-value pairs, the tensor infos (name, shape, type, offset), padding, then the tensor data.

#flashcards/ml/concept #flashcards/topic/llm-serving Which older formats did GGUF replace, and what was wrong with them?::GGML, GGMF and GGJT (per the spec). They did not name the model architecture, adding a setting was a breaking change, and every architecture needed its own conversion script.

#flashcards/ml/formula #flashcards/topic/llm-serving Write how many bytes a quantized GGUF tensor takes, and explain each symbol.
?
$$\text{bytes}=\frac{n}{B}\times s$$
- $n$: the number of weights in the tensor (the product of its shape).
- $B$: how many weights share one block (32 for Q5_0 and Q8_0, 256 for Q4_K and Q6_K, 1 for float32).
- $s$: how many bytes one block takes in the file.
- $\text{bytes}$: the size of the tensor's data. Because it is a simple sum, a program can find every tensor without reading them all. $n$ must be a whole number of blocks.

#flashcards/ml/example #flashcards/topic/llm-serving 64 weights are stored as Q5_0 (32 weights and 22 bytes per block). How many bytes, and how does it compare with float16?::64/32 = 2 blocks, 2 × 22 = 44 bytes. Float16 would be 64 × 2 = 128 bytes, so Q5_0 is about 2.9 times smaller.

#flashcards/ml/formula #flashcards/topic/llm-serving Write bits per weight for a block format and for a whole file, and explain each symbol.
?
$$b=\frac{8\,s}{B}\qquad\text{and for a whole file}\qquad b=\frac{8\,S}{N}$$
- $b$: the average bits used per weight.
- $s$ and $B$: bytes per block and weights per block. The 8 turns bytes into bits.
- $S$: all the bytes of tensor data in the file (not the header or metadata). $N$: the number of weights in the model.
- It lets you compare formats whatever the size of the model.

#flashcards/ml/example #flashcards/topic/llm-serving The Qwen2.5-0.5B file has 391,859,712 bytes of tensor data for 494,032,768 weights. What is the bits per weight?::8 × 391,859,712 / 494,032,768 = 6.345 bits per weight (my arithmetic, checked against the official gguf package). It is 2.52 times smaller than float16.

#flashcards/ml/example #flashcards/topic/llm-serving The file Ollama calls Q4_K_M has an average of 6.345 bits per weight. Why is it not 4?::The name stands for a mix of types. By weight count it was 51% Q5_0, 28% Q8_0, 11% Q6_K, 11% Q4_K and a tiny part float32. Why the mix looks like this I did not check.

#flashcards/ml/example #flashcards/topic/llm-serving In the Qwen2.5-0.5B GGUF, how many bytes were not tensor data, and what did they hold?::5,948,224 bytes, 1.5% of the 397,807,936-byte file: the header, 34 metadata pairs, the tokenizer (151,936 words and 151,387 merge rules) and padding.

#flashcards/ml/concept #flashcards/topic/llm-serving Why is a GGUF file aligned in fixed steps?::So a program can map the file into memory (mmap) and use the numbers in place without copying them. The spec lists it as a design goal, and I did not test the loading speed.
