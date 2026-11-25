---
tags: [LLM, file-format, weights, safety, PyTorch, Hugging Face, local]
status: growing
level: 4
tier: 2
---

# Safetensors

**Safetensors** is a file format for saving the numbers (weights) of an AI model. A safetensors file holds a short list of what is inside, followed by the raw numbers, and nothing that can run as a program. That makes it safe to open even when a stranger made the file.

## 🟢 1. Simple (age 12)
Imagine you get a box from a stranger. One kind of box is a "magic box" that may do things the moment you open it, even bad things. A safetensors box is a plain box with a label on the lid that lists what is inside, and then just the things themselves. Nothing in it can jump out and do something when you open it.

**Example:** you download a model from the internet. If it is saved as a safetensors file, opening it can only give you numbers. If it were saved the old way, in a "magic box" kind of file, opening it could run someone else's program on your computer.

## 🔵 2. How it works
1. **A model is a pile of number tables.** Each table has a name, such as `model.embed_tokens.weight`, a shape (rows and columns) and a type of number ([[Hugging Face Transformers]] calls these tensors).
2. **The file starts with a length.** The first 8 bytes say how long the label is.
3. **Then comes the label.** It is a short piece of text in a plain format called JSON. For each table it says the number type, the shape and where its bytes start and stop in the file.
4. **Then come the raw numbers**, one table after another, with nothing else between them.
5. **To read one table, you jump straight to its bytes.** You look up the start and stop in the label, go there and read. You do not need to read the whole file first.
6. **Nothing in the file can run.** The label is only data, and the numbers are only numbers. That is the safety.

I found no free picture of this on Wikimedia Commons (the search for "safetensors" returned nothing, and "pickle" only returned food), so this note has no image.

## 🟠 3. Technical detail
- **The layout** (README, through a page summary, and checked by me on a real file): 8 bytes with an unsigned little-endian 64-bit integer $n$, then $n$ bytes of JSON, then the byte buffer. The JSON maps each tensor name to `dtype`, `shape` and `data_offsets` (start and end, counted from the start of the buffer). A key `__metadata__` can hold a map of text, and my file has `{'format': 'pt'}` there.
- **Sizes add up exactly.** For one tensor the bytes are the number of entries times the bytes per entry (see the rule below).

A tensor's size in bytes (end offset minus start offset) = bytes per number (b) × the product of its dimension sizes (d_j).

**Terms:**
- start and end are the two numbers in `data_offsets`, so end-start is the tensor's size in bytes.
- b is the bytes per number: 4 for `F32`, 2 for `BF16` or `F16`, 1 for an 8-bit type.
- d_j are the sizes of each dimension in `shape`, and the product of the d_j is the number of entries in the tensor.

**Matters:** the reader trusts these offsets to jump around the file. If the sizes did not match the shape, a reader would read the wrong bytes, so a good reader checks them (the README says addresses cannot overlap, which I did not test).

**Example (tiny numbers):** a table with shape 2×3 in `F32`. It has 2×3=6 entries, and each takes 4 bytes, so end-start=6×4=24 bytes. The tiny file in the demo below says `data_offsets: [0, 24]`, the same.

**`safetensors_peek.py`** (needs `torch` and `safetensors`, and the model folder from the [[Local CPU Chatbot]] project)
```python
"""Read a .safetensors file by hand (8-byte length + JSON header + raw bytes), then compare with pickle. Needs: torch, safetensors."""
import json, math, os, pickle, struct, tempfile

import torch
from safetensors.torch import save_file, load_file

PATH = "qwen2.5-0.5b-instruct/model.safetensors" # from the Local CPU Chatbot project


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0] # first 8 bytes: header length, little-endian unsigned integer
        return n, json.loads(f.read(n)) # next n bytes: the JSON header


# 1. the real model file, read with no library at all
n, h = header(PATH)
meta = h.pop("__metadata__")
size = os.path.getsize(PATH)
count = sum(math.prod(t["shape"]) for t in h.values()) # numbers in each tensor = product of its shape
data_end = max(t["data_offsets"][1] for t in h.values())
name, t = next(iter(h.items()))
print("file bytes:", size, "| header length:", n, "| data starts at byte", 8 + n, "| data bytes:", data_end, "| 8 + n + data =", 8 + n + data_end)
print("tensors:", len(h), "| metadata:", meta, "| parameters from shapes:", f"{count:,}", "| dtypes:", sorted({t["dtype"] for t in h.values()}))
print("first tensor:", name, t)

# 2. a tiny file: write it with the library, then decode the bytes myself
tmp = os.path.join(tempfile.mkdtemp(), "tiny.safetensors")
save_file({"a": torch.arange(6, dtype=torch.float32).reshape(2, 3)}, tmp)
n2, h2 = header(tmp)
raw = open(tmp, "rb").read()[8 + n2:]
print("tiny header:", h2["a"], "| by hand:", struct.unpack("<6f", raw), "| library:", load_file(tmp)["a"].tolist())


# 3. a pickle file can run code when it is loaded; a safetensors file is only JSON plus numbers
class Sneaky:
    def __reduce__(self): # pickle calls this on load: "build me by calling print(...)"
        return (print, (" -> this line ran while the pickle was loading",))


print("pickle load:")
pickle.loads(pickle.dumps(Sneaky()))
try:
    json.loads(open(tmp, "rb").read()[8:8 + n2]) # the only 'code-like' part of the file is JSON, which is just data
    print("safetensors header parsed as plain JSON: ok")
except Exception as e:
    print("unexpected:", e)
```
What it printed (safetensors 0.7.0, torch 2.8.0+cpu, Python 3.9.6):
```
file bytes: 988097824 | header length: 32280 | data starts at byte 32288 | data bytes: 988065536 | 8 + n + data = 988097824
tensors: 290 | metadata: {'format': 'pt'} | parameters from shapes: 494,032,768 | dtypes: ['BF16']
first tensor: model.embed_tokens.weight {'dtype': 'BF16', 'shape': [151936, 896], 'data_offsets': [0, 272269312]}
tiny header: {'dtype': 'F32', 'shape': [2, 3], 'data_offsets': [0, 24]} | by hand: (0.0, 1.0, 2.0, 3.0, 4.0, 5.0) | library: [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
pickle load:
  -> this line ran while the pickle was loading
safetensors header parsed as plain JSON: ok
```
What to look for: the 32,288 bytes that I could not explain in [[Hugging Face Transformers]] are the 8-byte length plus the 32,280-byte header, and 8 + 32,280 + 988,065,536 is exactly the file size, 988,097,824. The 290 tensors hold 494,032,768 numbers (the same as the model's parameter count), all `BF16`. The first tensor has $151{,}936\times896\times2=272{,}269{,}312$ bytes, matching its offsets (my arithmetic). The tiny file decoded by hand gave the same six numbers as the library. The pickle part shows the danger on purpose: loading my `Sneaky` pickle ran a `print` that I had hidden in it, and it could have run any code. I did not load a safetensors file and a pickle file of the same model side by side, so I have no speed comparison.

## 🔴 4. Deep / formal
- **Why the header is capped.** The README says the header is limited to 100 MB, so a reader will not try to parse an enormous JSON text (through a page summary). My real header was 32,280 bytes, far below.
- **What "safe" does and does not mean.** The format cannot carry code, so loading it cannot run code. It does not say the numbers are good. A model from a stranger can still be a bad model, or give harmful answers, and the tokenizer and config files that come with it are separate files. I did not test any of that, and the README text I read does not mention a security audit.
- **Lazy loading.** Because each tensor has its own offsets, a reader can open a file and take one table without reading the rest. I tested `load_file` on the tiny file only, not partial loading of the big one.
- **Zero-copy is not magic.** The README itself says no format is truly zero-copy, since data still moves from disk to memory, and that the header is not zero-copy (page summary).
- **Speed claims are the README's, not mine.** It says CPU loading is "extremely fast" compared to pickle, and quotes a case where loading across 8 GPUs fell from about 10 minutes to 45 seconds. I did not time anything.
- **How it relates to [[GGUF]].** Both are "header, then raw tensor bytes" formats. GGUF adds many key-value settings in its header (34 keys in the Qwen file I read) and stores quantized types such as Q8_0 and Q4_K ([[Quantization]]) for [[llama.cpp]] and [[Ollama]]. The safetensors file here is `BF16` throughout.
- **Limits of this note:** I did not test the old PyTorch `.bin` format, `torch.load` options, the Rust or other-language readers, files made by other tools, or malformed files.

**What came before, and how this improved on it:**
- **Before:** PyTorch saved models with Python's `pickle` (files often ended in `.bin` or `.pt`; this is from my memory, I did not check).
- **Its limit:** loading a pickle can run any code stored in it. My `Sneaky` demo above ran a hidden `print` while loading, and the README's own table calls pickle "Unsafe, runs arbitrary code".
- **What changed:** a format that is only data: a length, a JSON list of tensors and their byte ranges, and the raw bytes. The README says its goal is to remove the need for PyTorch's default pickle-based serialization.
- **How much better:** safety, shown by my demo (a pickle ran code, a safetensors header was parsed as plain JSON and nothing else). For speed I only have the README's numbers (about 10 minutes to 45 seconds for one multi-GPU case), not my own. The release date and authors are not in this note, because I did not check them.

## Related
[[Hugging Face Transformers]] · [[Hugging Face Hub]] · [[GGUF]] · [[Quantization]] · [[llama.cpp]] · [[Ollama]] · [[PyTorch]] · [[Mixed Precision Training]] · [[Python]] · [[Local CPU Chatbot]] · [[Agent Safety]] · [[ONNX]] · [[MLOps]]
## Sources
- Run on this PC: safetensors 0.7.0, torch 2.8.0+cpu, Python 3.9.6, and the file `qwen2.5-0.5b-instruct/model.safetensors` from the [[Local CPU Chatbot]] project. Every number in the output above is from my own run.
- The safetensors README (https://github.com/huggingface/safetensors), read on 18 November 2026 through a page summary: the file layout, the Apache-2.0 licence, the 100 MB header limit, the safety table, the speed claims, and the point that it is not truly zero-copy. I did not test the claims about overlap, polyglot files or speed.
- From my memory, not checked: that PyTorch's old files were pickle-based `.bin` or `.pt` files.
- No image: Wikimedia Commons had no on-topic file.
- The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving Safetensors:::A file format for model weights: an 8-byte length, a JSON list of the tensors, then the raw numbers. It holds only data, so loading it cannot run code.

#flashcards/ml/concept #flashcards/topic/llm-serving What is the layout of a safetensors file?
?
- The first 8 bytes: the header length $n$ (unsigned, little-endian).
- Then $n$ bytes of JSON: for each tensor its type, shape and byte range (and an optional `__metadata__` map).
- Then the raw bytes of all tensors.

#flashcards/ml/concept #flashcards/topic/llm-serving Why is a safetensors file safer than a pickle file?::A pickle can store instructions that run when it is loaded. A safetensors file is only a JSON list plus numbers, so there is nothing to run. In my demo a pickle ran a hidden `print` on load.

#flashcards/ml/concept #flashcards/topic/llm-serving How many bytes does one tensor take in a safetensors file, and what does each symbol mean?
?
A tensor's size in bytes (end offset minus start offset) = bytes per number (b) × the product of its dimension sizes (d_j).
- start and end: the two numbers in `data_offsets`.
- b: bytes per number (4 for `F32`, 2 for `BF16` or `F16`, 1 for an 8-bit type).
- d_j: the sizes in `shape`, so the product of the d_j is the number of entries.
- A reader trusts these offsets to jump around the file, so mismatched sizes would mean reading the wrong bytes.

#flashcards/ml/example #flashcards/topic/llm-serving A tensor has shape 2 by 3 and type `F32`. What are its `data_offsets`, if it is the first tensor?::2 × 3 = 6 entries × 4 bytes = 24, so `[0, 24]` (my arithmetic, matching the tiny file in the demo).

#flashcards/ml/example #flashcards/topic/llm-serving What explained the extra 32,288 bytes in the Qwen2.5-0.5B file?::The 8-byte length plus the 32,280-byte JSON header. 8 + 32,280 + 988,065,536 (tensor bytes) = 988,097,824, the file size (my arithmetic and my run).

#flashcards/ml/example #flashcards/topic/llm-serving What did the Qwen2.5-0.5B safetensors file hold?::290 tensors, all `BF16`, with 494,032,768 numbers in total (summed from the shapes), and `__metadata__` of `{'format': 'pt'}`.

#flashcards/ml/cloze #flashcards/topic/llm-serving The safetensors header is written in ==JSON==, so it is data and not a program.

#flashcards/ml/concept #flashcards/topic/llm-serving Does "safe" in safetensors mean the model is good or harmless?::No. It only means loading the file cannot run code. The model can still be bad or give harmful answers, and the tokenizer and config files are separate. I did not test any of that.

#flashcards/ml/concept #flashcards/topic/llm-serving How does safetensors compare with GGUF?::Both are a header followed by raw tensor bytes. GGUF adds many key-value settings and quantized types for llama.cpp and Ollama. The safetensors file in this vault's project is `BF16` throughout.

#flashcards/ml/concept #flashcards/topic/llm-serving What does the README say about the header size and about "zero-copy"? (page summary)::The header is limited to 100 MB so a reader does not parse an enormous JSON text. It also says no format is truly zero-copy, since data still moves from disk to memory, and the header is not zero-copy.

#flashcards/ml/concept #flashcards/topic/llm-serving What did I not test about safetensors?::Speed against pickle, partial loading of a big file, the old PyTorch `.bin` format, malformed files, overlap checks, and other readers or writers. I only decoded one real file and one tiny file.
