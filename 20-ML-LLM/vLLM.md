---
tags: [LLM, inference, serving, vLLM, KV-cache, PagedAttention, throughput]
status: growing
level: 4
tier: 3
---

# vLLM

An open-source **serving system** for large language models ([[LLM]]) from the 2023 paper by Kwon et al. Its main idea is [[PagedAttention]]: store each request's [[KV Cache]] in small blocks, so the GPU's memory is not wasted and more requests run at once.

## 🟢 1. Simple (age 12)
Imagine a restaurant with 20 big tables. The rule is that every group gets a whole big table, in case they bring a lot of friends. Most groups are two people, so the restaurant looks full while the tables are almost empty, and people queue outside. vLLM is the manager who says: "Give each group only the seats they need right now, and add a seat when another friend arrives." Many more groups fit, so the whole restaurant serves more people per hour.

**Example:** Hundreds of people chat with one chatbot at the same time. Some ask short questions, some ask for long essays. vLLM keeps all of them fed on the same GPU by giving each chat a little memory at a time.

## 🔵 2. How it works
1. A chatbot writes one token at a time ([[Autoregressive Generation]]). For each chat it remembers notes about the earlier tokens, called the [[KV Cache]]. These notes use a lot of GPU memory, and they grow as the answer grows ([[LLM Inference]]).
2. Before vLLM, serving systems usually reserved a long, unbroken stretch of memory for each chat, as long as the longest answer allowed. Most of that space stayed empty. That left room for only a few chats at once.
3. vLLM cuts each chat's notes into small **blocks**. A table keeps track of which block holds which part of the chat. Blocks can live anywhere in memory and are handed out only when needed ([[PagedAttention]]).
4. Chats that start with the same text (a shared prompt, or several answers from one prompt) can **share** blocks instead of making copies.
5. A **scheduler** decides each step which chats run. New chats join as soon as there is space and finished chats leave ([[Continuous Batching]]). Requests are served first-come, first-served.
6. If the GPU runs out of blocks, vLLM can pause some chats. It either copies their blocks to the computer's ordinary memory (swapping) or throws them away and rebuilds them later (recomputation).
7. Big models can be split across several GPUs ([[Model Parallelism]]), and vLLM supports this.
8. Result: more chats fit in the same memory, so the system answers more requests per second at about the same waiting time.

## 🟠 3. Technical detail
- **What it is.** A "high-throughput distributed LLM serving engine", with the memory manager and the attention method (PagedAttention) designed together (paper, introduction). Source code is public (stated in the paper's abstract).
- **Parts (paper, section 4):** a central scheduler, a KV block manager with a block table for each request (logical block to physical block), and GPU workers. The block manager has a GPU allocator and a CPU allocator (for swapping).
- **The problem it targets (paper, section 3):** on an A100 GPU with 40 GB serving a 13-billion-parameter model, about 65% of memory holds the weights and close to 30% the KV cache. One token's KV cache is 800 KB, so a 2048-token request can need 1.6 GB (the arithmetic is in level 4). Previous systems stored each request's cache in one unbroken piece sized for the longest possible answer.
- **Three kinds of waste (paper):** reserved space not yet used, internal fragmentation (space reserved but never used), and external fragmentation (gaps between pieces). The paper's measurement says only **20.4% to 38.2%** of KV cache memory held real token states in the previous systems it tested.
- **Sharing.** Parallel sampling and beam search share the prompt's blocks. Blocks are copied only when a sequence writes into a shared one ("copy-on-write"). A shared prefix, such as one long instruction used by many requests, can be cached the same way.
- **Block size.** The paper tests several sizes. It settles on **16 tokens** as the default: big enough for the GPU to work well, small enough to waste little. Larger sizes hurt when sequences are short.
- **Preemption.** Swapping or recomputation. The paper says which is better depends on the link between CPU and GPU memory and the GPU's speed; it measures both.
- **Compared with what.** The paper's comparison systems are FasterTransformer and Orca (three versions of Orca that differ in how well they guess output length). These are the systems of 2023. I did **not** read about any later system, and I did not compare other engines such as TensorRT-LLM, TGI or SGLang. The vault has no source for how those compare, so this note does not say which engine is best.
- **The code below is a toy.** It simulates a stream of 5,000 requests and compares "reserve the maximum" with "hand out blocks of 16", checks a block table, and counts blocks saved by sharing a prompt. The request lengths are my own made-up distribution.

```python
import math

import numpy as np

rng = np.random.RandomState(0)
MAX_LEN, BLOCK, MEMORY = 2048, 16, 40_000 # longest allowed text, tokens per block, KV-cache slots on the GPU

# a stream of requests: final length (prompt + answer) of each, mostly short, a few long
lengths = np.clip(rng.lognormal(mean=5.3, sigma=0.8, size=5000), 20, MAX_LEN).astype(int)
print(f"request lengths: median {int(np.median(lengths))}, mean {lengths.mean():.0f}, longest {lengths.max()} (limit {MAX_LEN})")

# way 1: reserve MAX_LEN slots for every request up front
fit_static = MEMORY // MAX_LEN
used_static = lengths.mean() / MAX_LEN
print(f"reserve-the-max: {fit_static} requests fit; slots actually holding tokens {used_static:.1%} (the rest is empty or waiting)")

# way 2: hand out blocks of 16 slots as the text grows; only the last block is partly empty
blocks = np.ceil(lengths / BLOCK)
slots_paged = blocks * BLOCK
fit_paged = int(MEMORY // slots_paged.mean())
used_paged = lengths.sum() / slots_paged.sum()
print(f"paged blocks: about {fit_paged} requests fit; slots holding tokens {used_paged:.1%}; waste is only the last block: {1 - used_paged:.1%}")
print(f"-> {fit_paged / fit_static:.1f} times as many requests in the same memory")

# a block table: logical block number -> physical block number, handed out from a free list
free = list(range(100))
rng.shuffle(free)
table = [free.pop() for _ in range(math.ceil(37 / BLOCK))] # a request with 37 tokens needs 3 blocks
print("37 tokens -> blocks needed:", len(table), "| block table (logical 0,1,2 -> physical):", table)

# sharing: 4 answers sampled from one prompt of 100 tokens, each answer 200 tokens
prompt, answer, n = 100, 200, 4
copy_all = n * math.ceil((prompt + answer) / BLOCK)
shared = prompt // BLOCK + n * math.ceil((prompt % BLOCK + answer) / BLOCK) # full prompt blocks are shared; the half-full one is copied
print(f"4 samples of one prompt: {copy_all} blocks if each copies the prompt, {shared} if the full prompt blocks are shared")
```

What it printed: lengths have median 196 and mean 269 (limit 2048). Reserving the maximum fits 19 requests, and only 13.1% of the reserved slots hold tokens. Paged blocks fit about 144 requests, with 97.3% of slots holding tokens and 2.7% waste (the unfilled last block). That is 7.6 times as many requests in the same memory. A 37-token request needs 3 blocks (the table maps them to physical blocks 53, 84 and 64, drawn at random). Four samples of one 100-token prompt need 76 blocks with copies and 58 with sharing.

**What this toy leaves out:** it counts each request at its **final** length, as if all were at their end at the same moment. A real server has requests at all stages, so the true gap depends on the traffic. It does not model time, the cost of the block table, or the speed of the attention kernel. The 7.6 is a property of my invented lengths. The paper's own measurements are above and below.

## 🔴 4. Deep / formal
The memory the cache needs for one token follows from the model's shape:

Bytes of cache per token = 2 (key and value) × hidden size (H) × number of layers (L) × bytes per number (b).

**Terms:**
- 2 is for the key and the value, both stored.
- H is the hidden size, the number of values in one token's vector (5120 for the 13-billion-parameter OPT model in the paper).
- L is the number of layers (40 in the paper's example). Each layer keeps its own keys and values.
- b is bytes per number (2 for FP16, a number format with 16 bits).
- **Matters:** the bill grows with each of the four factors, and with the length of the text. So the cache of one long request is large, and memory, not arithmetic, limits how many requests fit ([[LLM Inference]], [[Grouped-Query Attention]] which shrinks H for keys and values).

**Example (tiny numbers):** the paper's OPT model.
- Per token: 2×5120×40×2=819,200 bytes, which the paper rounds to 800 KB.
- For a request of 2048 tokens: 819,200×2048≈1.68×10^9 bytes, about 1.6 GiB (the paper says "as much as 1.6 GB").
- Meaning: a GPU with a few tens of GB left over for the cache can hold only a few tens of such requests. This is why wasting 60 to 80% of that memory (the paper's measure) matters.

**Further notes (paper, section 6, read; numbers are the paper's, not mine):**
- **Throughput.** The abstract says vLLM improves the throughput of popular LLMs by **2 to 4 times** at the same latency compared with FasterTransformer and Orca, and more for longer sequences, bigger models and more complex decoding.
- **Detail (ShareGPT trace).** vLLM sustained 1.7 to 2.7 times higher request rates than Orca (Oracle), 2.7 to 8 times higher than Orca (Max), and up to 22 times higher than FasterTransformer. For OPT-13B it batched 2.2 times as many requests as Orca (Oracle) and 4.3 times as many as Orca (Max).
- **Where the gain is small.** On the Alpaca trace with OPT-175B the gain over Orca (Oracle) and Orca (Pow2) was less clear, because that setup has a lot of free memory and short sequences, so the older systems could batch many requests anyway.
- **Sharing.** Memory saved by sharing blocks was 6.1% to 9.8% for parallel sampling and 37.6% to 55.2% for beam search (Alpaca trace); 16.2% to 30.5% and 44.3% to 66.3% on ShareGPT.
- **Toy against paper.** The toy's 13.1% for "reserve the max" is below the paper's 20.4% to 38.2% because the paper's systems were smarter than "reserve the max for everything". The toy's sharing saved $18/76\approx24\%$, which sits between the paper's two ranges only by accident of my numbers.
- **What I did not read:** the GPU kernel details, the distributed section beyond its first lines, and the ablation on block size in full.

**What came before, and how this improved on it:**
- **Before:** serving systems such as FasterTransformer and Orca stored each request's KV cache as one unbroken block of memory, sized for the longest possible output (paper, section 3.1).
- **Its limit:** memory was lost to reserved-but-unused space and to fragmentation. The paper measured only 20.4% to 38.2% of KV cache memory holding real token states, which kept the batch small.
- **What changed:** manage the cache like an operating system manages virtual memory: small blocks, a block table, handing out blocks on demand, sharing and copy-on-write (Kwon et al., arXiv 2309.06180, 2023; the paper's text read).
- **How much better:** published: **2 to 4 times** the throughput of the systems it was tested against, at similar latency (abstract and section 6). In this note's toy: 19 requests against about 144 in the same memory (7.6 times), with waste down from 86.9% to 2.7%, but that depends on my invented request lengths.

## Related
- [[PagedAttention]] · [[KV Cache]] · [[LLM Inference]] · [[Continuous Batching]] · [[Model Parallelism]] · [[Grouped-Query Attention]] · [[FlashAttention]] · [[Speculative Decoding]] · [[Autoregressive Generation]] · [[GPU]] · [[LLM]] · [[Interview Question Index]]

## Sources
- Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica, *Efficient Memory Management for Large Language Model Serving with PagedAttention*, arXiv 2309.06180 (2023). The PDF text was read for the abstract, section 3 (memory), section 4 (design, sharing, scheduling, preemption), and section 6 (results). Not read in detail: the kernel and distributed sections and the ablation figures. The authors and the date (v1, 12 Sep 2023) are as printed on the PDF's first page.
- The 2023 comparison systems (FasterTransformer, Orca) are as the paper describes them. Nothing here was checked about later versions of vLLM or other engines.
- Image: none. Three Wikimedia Commons searches found only unrelated pages and general Transformer diagrams, so none is used.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving vLLM:::A serving system for LLMs that cuts each chat's KV cache into small blocks (PagedAttention), shares blocks between chats with the same text, and uses a scheduler that lets new chats join as soon as there is room.

#flashcards/ml/concept #flashcards/topic/llm-serving Write the KV-cache bytes per token and explain each symbol.
?
Bytes of cache per token = 2 (key and value) × hidden size (H) × number of layers (L) × bytes per number (b).
- 2: key and value are both stored.
- H: the hidden size, the number of values in one token's vector (5120 for the 13-billion-parameter OPT model in the vLLM paper).
- L: the number of layers (40 in the paper's example). Each layer keeps its own keys and values.
- b: bytes per number (2 for FP16).
- The bill grows with all four factors and with the text length, so memory, not arithmetic, limits how many requests fit.

#flashcards/ml/example #flashcards/topic/llm-serving For the paper's OPT model ($H=5120$, $L=40$, FP16), what are the bytes per token and for a 2048-token request?::Per token $2\times5120\times40\times2=819{,}200$ bytes (the paper rounds to 800 KB). For 2048 tokens about $1.68\times10^9$ bytes, around 1.6 GiB (the paper says as much as 1.6 GB).

#flashcards/ml/cloze #flashcards/topic/llm-serving Before vLLM, serving systems reserved a ==long unbroken stretch== of memory per chat, so most of it stayed empty and only a few chats ran at once.

#flashcards/ml/concept #flashcards/topic/llm-serving What problem does vLLM fix, and how?::It fixes the memory waste of KV caches with block-based paging, so many more chats fit on the same GPU.
