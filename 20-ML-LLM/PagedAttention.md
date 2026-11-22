---
tags: [LLM, inference, serving, KV-cache, memory, vLLM, PagedAttention]
status: growing
level: 4
tier: 3
---

# PagedAttention

A way to store the [[KV Cache]] in small **pages**, like an operating system stores memory, so almost no memory is wasted.

## 🟢 1. Simple (age 12)
Suppose a hotel reserves **a whole floor for every guest**, because some guests might bring lots of luggage. Most guests only use one room, so the hotel is mostly empty but "full". PagedAttention hands out **one small room at a time** as each guest needs it, so many more guests fit. For an AI, the "guests" are users' chats, and the "rooms" are chunks of memory that hold the notes about each chat.

**Example:** A chat starts and could grow to 2,048 words, but it is 100 words long so far. The old way reserves space for 2,048 anyway. The paged way gives it 7 small pages for the 100 words and adds more only when it grows.

## 🔵 2. How it works
1. When an [[LLM]] writes text, it keeps a [[KV Cache]] for each chat. The cache grows by one entry for every new token, so nobody knows in advance how big it will end up.
2. The simple approach is to **reserve one big contiguous block** per chat, big enough for the longest possible chat. Most of it sits empty. This wastes memory and limits how many chats can run at the same time.
3. PagedAttention splits each chat's cache into small fixed-size **blocks** (pages), for example 16 tokens each, which need not sit next to each other in memory.
4. A small **page table** records, for each chat, which blocks hold which tokens.
5. A new block is handed out only when the chat needs it. When the chat ends, its blocks are freed for others.
6. Because blocks are separate, chats can **share** blocks, such as a common prompt, instead of copying it.
7. The attention calculation is changed so that it can read the keys and values from scattered blocks. The answers stay the same.
8. Result: more chats fit on the same GPU, so the whole system serves more users per second.

## 🟠 3. Technical detail
- **Idea (paper abstract, read):** PagedAttention is "inspired by the classical virtual memory and paging techniques in operating systems". It was built into a serving system called vLLM, which aims for near-zero waste in KV cache memory and flexible sharing of the cache within and across requests (arXiv 2309.06180).
- **Claimed result:** vLLM improves the throughput of popular LLMs by 2 to 4 times at the same latency compared with the state-of-the-art systems the paper tested, with larger gains for longer sequences, bigger models and more complex decoding (abstract).
- **Why memory limits speed:** serving needs many requests batched together, and the abstract says the KV cache of each request is huge and grows and shrinks dynamically. If this memory is badly managed, fragmentation and duplication limit the batch size.
- **Page size:** a trade-off. Small pages waste less but need a bigger page table and more lookups. The code uses 16 tokens, the number I picked, not a recommendation.
- **Sharing:** with [[Decoding Strategies]] that produce several answers from one prompt (such as beam search or sampling many replies), the prompt's pages can be shared and only the new parts copied.
- **Not the same as other cache tricks:** [[Grouped-Query Attention]] makes each token's entry *smaller*, [[Quantization]] makes the numbers shorter, and paging makes *placement* efficient.
- **The simulation in the code:** my own random model of request lengths, not the paper's measurements.

```python
import numpy as np
rng = np.random.RandomState(0)
max_len, block = 2048, 16                                       # tokens a request may grow to, and tokens per page
lengths = rng.randint(20, max_len, 1000)                         # how long each of 1000 requests actually gets
contiguous = max_len * len(lengths)                              # old way: reserve the maximum for every request
paged = int(np.ceil(lengths / block).sum()) * block              # paged way: only whole pages that are actually needed
used = int(lengths.sum())
print('tokens really used:', used)
print('contiguous reservation:', contiguous, '| wasted', round(100 * (contiguous - used) / contiguous, 1), '%')
print('paged:', paged, '| wasted', round(100 * (paged - used) / paged, 2), '%')
# how many requests fit in a cache that holds 1,000,000 tokens
budget = 1_000_000
print('fit with contiguous reservation:', budget // max_len, 'requests | with pages (average request):', int(budget // (paged / len(lengths))))
# the waste of paging is at most block-1 tokens per request, on average about half a page
print('average waste per request with pages:', round((paged - used) / len(lengths), 2), 'tokens | at most', block - 1)
# pages need a lookup table from the request's token positions to pages
print('page table entries for one 100-token request:', int(np.ceil(100 / block)), '| token 37 sits in page', 37 // block, 'at offset', 37 % block)
# sharing: two answers to the same prompt of 100 tokens share the prompt's pages
shared = int(np.ceil(100 / block)); print('pages for 2 answers with a shared 100-token prompt (each adds 1 page): ', shared + 2, '| without sharing:', 2 * (shared + 1))
```

## 🔴 4. Deep / formal
**How much memory a request takes.** With pages of $B$ tokens and a request of $\ell$ tokens so far:

$$
M_{\text{paged}}(\ell)=B\left\lceil\frac{\ell}{B}\right\rceil,\qquad \text{waste}_{\text{paged}}=M_{\text{paged}}-\ell\le B-1
$$

**Where:**
- $\ell$ is the number of tokens the request has produced so far, and $B$ is the number of tokens per page.
- $\lceil\cdot\rceil$ rounds up, because a partly filled last page is still a whole page.
- $M_{\text{paged}}$ is the number of token slots held, and the waste is the empty slots in the last page.
- **Matters:** the waste is at most $B-1$ slots, however long the request is. A contiguous reservation of $L_{\max}$ slots wastes $L_{\max}-\ell$, which can be nearly all of it for short requests.

**Example (tiny numbers):** $\ell=100$, $B=16$, $L_{\max}=2048$.
- Pages needed: $\lceil100/16\rceil=\mathbf{7}$, holding $7\times16=\mathbf{112}$ slots. Waste $=\mathbf{12}$ slots.
- Contiguous reservation waste: $2048-100=\mathbf{1948}$ slots.
- Meaning: the same request wastes 12 slots with paging against 1,948 with a fixed reservation. The code also places token 37 on page 2 at offset 5 ($37=2\times16+5$), which is what the page table does.

**Further notes:**
- **Simulation (my own):** 1,000 requests with random final lengths from 20 to 2,047 tokens used $1{,}050{,}863$ token slots. A $2{,}048$-slot reservation for each took $2{,}048{,}000$ (48.7% wasted); pages took $1{,}058{,}320$ (0.7% wasted, average waste $7.46$ tokens per request). In a cache of one million slots, that is 488 requests with reservations against about 944 with pages.
- **Sharing:** two answers to one 100-token prompt need $7+2=9$ pages if the prompt's 7 pages are shared (each answer adding one page), against $2\times8=16$ if not. Real sharing needs copy-on-write care that I did not model.
- **What the toy leaves out:** the cost of the page table and of reading scattered memory, which the paper handles in its kernel, and requests that are paused or moved.
- **Not covered:** scheduling, preemption, and how this combines with FlashAttention-style kernels.

**What came before, and how this improved on it:**
- **Before:** each request's KV cache stored in one contiguous block sized for the worst case (the simple way that the abstract says wastes memory through fragmentation and duplication).
- **Its limit:** limited batch size, and so limited throughput.
- **What changed:** paged blocks with a page table, borrowed from operating systems (Kwon et al., arXiv 2309.06180, 12 September 2023, abstract read).
- **How much better:** the abstract reports 2 to 4 times higher throughput at the same latency. In this note's own simulation, memory waste fell from 48.7% to 0.7% and the number of requests that fit in the same cache roughly doubled (488 to about 944). That simulation is invented data, not a measurement of any system.

## Related
- [[KV Cache]] · [[Grouped-Query Attention]] · [[Quantization]] · [[FlashAttention]] · [[GPU]] · [[LLM]] · [[Decoding Strategies]] · [[Speculative Decoding]] · [[Transformer]] · [[LLM Inference]] · [[Continuous Batching]] · [[vLLM]]

## Sources
- Stanford CME 295 (Fall 2025), Lecture 3 slides, Afshine and Shervine Amidi
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, 2023 (arXiv 2309.06180, abstract read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving PagedAttention:::Split each chat's KV cache into small fixed-size blocks (pages) that need not sit together in memory, with a page table that records which block holds which tokens. Blocks are given out only when needed.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the memory held and the waste with paging and explain each symbol.
?
$$M_{\text{paged}}(\ell)=B\left\lceil\frac{\ell}{B}\right\rceil,\qquad \text{waste}_{\text{paged}}=M_{\text{paged}}-\ell\le B-1$$
- $\ell$: the number of tokens the request has produced so far. $B$: the number of tokens per page.
- $\lceil\cdot\rceil$: rounds up, because a partly filled last page is still a whole page.
- $M_{\text{paged}}$: the token slots held. The waste is the empty slots in the last page.
- The waste is at most $B-1$ slots however long the request is. A contiguous reservation of $L_{\max}$ slots wastes $L_{\max}-\ell$, nearly all of it for short requests.

#flashcards/ml/example #flashcards/topic/llm-serving $\ell=100$, $B=16$, $L_{\max}=2048$. Compare the waste.::Pages: $\lceil100/16\rceil=7$, holding 112 slots, so waste is 12. A contiguous reservation wastes $2048-100=1948$ slots.

#flashcards/ml/cloze #flashcards/topic/llm-serving PagedAttention lets chats ==share== blocks, such as a common prompt, instead of copying it.

#flashcards/ml/cloze #flashcards/topic/llm-serving The old way reserved one big ==contiguous== block per chat, big enough for the ==longest== possible chat.

#flashcards/ml/concept #flashcards/topic/llm-serving What does the page table do?::For each chat it records which blocks hold which tokens, so blocks can live anywhere in memory.
