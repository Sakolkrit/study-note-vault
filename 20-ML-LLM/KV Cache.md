---
tags: [LLM, transformer, attention, inference, KV-cache, optimization]
status: growing
level: 4
tier: 3
---

# KV Cache

Short for **key-value cache**. It is the memory an [[LLM]] keeps while it writes an answer, so it does not redo old work for every new word.

## 🟢 1. Simple (age 12)
When an AI writes a sentence, it writes **one word at a time**, and each new word has to "look back" at all the words before it. The KV cache is a **notebook where the AI keeps what it already worked out** about the old words. Then, for each new word, it only does the work for that one word and reads the rest from the notebook.

**Example:** You are adding up a long shopping receipt, and the shop keeps handing you one more item. The slow way is to add up the whole receipt again from the top every time. The fast way is to **write down the running total** and just add the new item to it. The KV cache is that running total.

## 🔵 2. How it works
1. An LLM is a [[Transformer]]. To write the next token it uses [[Self-Attention]]: the newest token makes a **query**, and compares it with a **key** from every earlier token. It then mixes the earlier tokens' **values** according to those scores.
2. The key and the value of an old token are made from that token and the fixed weights of the model. They do **not change** when a new token arrives. (This is true for a model that reads left to right, which is what an LLM does. See the mask in [[Self-Attention]].)
3. So after computing the key and value of a token once, we **store them** for every layer. That store is the **KV cache**.
4. At each new step the model makes the query, key and value **only for the newest token**, adds the new key and value to the cache, and compares the query with everything in the cache.
5. Result: the same answers as before, with far less repeated work. The code below checks this on a tiny model.
6. The price is **memory**. The cache grows by one entry per token per layer, and for a long chat it can be gigabytes. Much of the engineering around serving LLMs is about making this cache smaller: [[Grouped-Query Attention]], paging it like an operating system does, [[Quantization]] of the cache, and compressing the keys and values.

## 🟠 3. Technical detail
- **Two phases of a request.**
  - **Prefill:** the whole prompt goes through the model at once and fills the cache. This is a big parallel job.
  - **Decode:** one token per step, each step reads the entire cache. This part is slow because it is limited by how fast memory can be read, not by arithmetic (see [[GPU]]).
- **What is cached:** the keys and values of every layer, for every head. Queries are not cached because nobody asks for an old query again.
- **What is saved:** with a cache, the cost of making keys and values over $n$ steps is $n$ pairs. Without it, step $t$ redoes all $t$ tokens, so the total is $1+2+\cdots+n$. The attention comparison itself still reads all $t$ cached keys at step $t$, so the cache removes the repeated *projection* work, not the reading.
- **Memory:** each token adds $2\cdot L\cdot d_{\text{model}}\cdot b$ bytes (keys and values, over $L$ layers, with $b$ bytes per number). A model with 32 layers and width 4096 at 16-bit numbers holds about 2.15 GB for one 4,096-token chat (the code computes this; the shape is a 7B-class model, from memory).
- **Why it limits how many users one GPU can serve:** each user's chat needs its own cache. Memory left over after the weights decides how many chats fit at once.
- **Paging:** the lecture covers *PagedAttention*, which stores the cache in small non-contiguous blocks to waste less memory, like virtual memory in an operating system. Its paper reports 2 to 4 times more throughput at the same latency (arXiv abstract, which I read).
- **Shrinking it:** fewer key/value heads ([[Grouped-Query Attention]]), compressing keys and values into a smaller hidden form (the lecture's "latent attention", from DeepSeek-V2, which I did not read), and [[Quantization]].
- **Beam search and sampling several answers:** copies of a prefix can share one cache until they differ.

```python
import numpy as np
rng = np.random.RandomState(0)
d = 8                                                      # width of every vector
Wq, Wk, Wv = (rng.randn(d, d) / np.sqrt(d) for _ in range(3))
E = rng.randn(20, d)                                       # embedding of 20 possible tokens
softmax = lambda s: np.exp(s - s.max()) / np.exp(s - s.max()).sum()

def generate(steps, cache):
    toks, K, V, made = [3], [], [], 0                      # start token 3; 'made' counts K,V pairs computed
    for t in range(steps):
        if cache and K:                                    # only the newest token needs a new key and value
            K.append(E[toks[-1]] @ Wk); V.append(E[toks[-1]] @ Wv); made += 1
        else:                                              # no cache (or first step): recompute for every token so far
            K = [E[i] @ Wk for i in toks]; V = [E[i] @ Wv for i in toks]; made += len(toks)
        q = E[toks[-1]] @ Wq
        w = softmax(np.array(K) @ q / np.sqrt(d))          # attention weights over all tokens so far
        out = w @ np.array(V)
        toks.append(int(np.argmax(E @ out)))               # greedy choice of the next token
    return toks, made

a, made_a = generate(12, cache=False)
b, made_b = generate(12, cache=True)
print('same tokens with and without cache:', a == b, '| tokens:', a)
print('key/value pairs computed for 12 steps: no cache', made_a, '= 1+2+...+12 =', 12 * 13 // 2, '| cache', made_b)
for n in (100, 1000, 10000):
    print('n =', n, '| no cache', n * (n + 1) // 2, '| cache', n, '| ratio', round(n * (n + 1) // 2 / n, 1))
# memory: 2 (K and V) * layers * tokens * width * bytes per number
L, n, dm, B = 32, 4096, 4096, 2                            # a 7B-size shape, 16-bit numbers
print('KV cache bytes:', 2 * L * n * dm * B, '=', round(2 * L * n * dm * B / 1e9, 2), 'GB for one sequence of', n, 'tokens')
```

## 🔴 4. Deep / formal
**Work with and without the cache.** Count how many key-value pairs must be made to write $n$ tokens:

$$
\underbrace{\sum_{t=1}^{n}t=\frac{n(n+1)}{2}}_{\text{no cache}}\qquad\text{against}\qquad \underbrace{n}_{\text{cache}}
$$

**Where:**
- $n$ is the number of tokens written so far (prompt included).
- $t$ is the step number. At step $t$, a model with no cache redoes the keys and values of all $t$ tokens.
- The curly bracket labels say which count belongs to which method.
- **Matters:** without the cache the work grows with the **square** of the length, so doubling the text makes it four times as big. With the cache the work for keys and values grows in a straight line. The saving is the factor $(n+1)/2$.

**Example (tiny numbers):** write $n=4$ tokens.
- No cache: $1+2+3+4=\mathbf{10}$ pairs made.
- Cache: $\mathbf{4}$ pairs made.
- Meaning: the cache avoids $6$ of the $10$. At $n=1000$ it is $500{,}500$ against $1{,}000$, a factor of $500.5$, as the code printed.

**Memory of the cache.** The price of the saving:

$$
M_{\text{KV}}=2\,L\,n\,d_{\text{model}}\,b
$$

**Where:**
- $2$ is for keeping both a key and a value for each token.
- $L$ is the number of layers; each layer has its own keys and values.
- $n$ is the number of tokens in the cache; $d_{\text{model}}$ is the width of the model (all heads together).
- $b$ is the number of bytes per stored number (2 for 16-bit, 1 for 8-bit).
- **Matters:** memory grows in a straight line with the length of the chat and with the number of layers and heads. Halving $b$ (see [[Quantization]]) halves the cache. If $d_{\text{model}}$ is replaced by fewer key/value heads, the cache shrinks in proportion ([[Grouped-Query Attention]]).

**Example (tiny numbers):** a toy model with $L=2$ layers, $n=3$ tokens, $d_{\text{model}}=4$ and $b=2$ bytes.
- $2\times2\times3\times4\times2=\mathbf{96}$ bytes.
- Meaning: 96 bytes for the whole toy chat. Scale to $L=32$, $n=4096$, $d_{\text{model}}=4096$, $b=2$ and you get $2{,}147{,}483{,}648$ bytes, about 2.15 GB, for **one** user.

**Further notes:**
- **Speed is limited by memory reads.** At each decode step the model reads all its weights and the whole cache to produce one token, so the time per token is set mostly by bytes moved. This is why the lecture groups the cache, [[Grouped-Query Attention]], [[Speculative Decoding]] and multi-token prediction under "inference optimizations".
- **Exactness:** the cache is an *exact* trick. The tokens match with and without it (the code's first line). Tiny differences can appear in real systems because adding numbers in a different order changes the last digits (the lecture points to a paper on this, which I did not read).
- **Not covered:** cache eviction for very long inputs, prefix caching across requests, and compressing the cache.

**What came before, and how this improved on it:**
- **Before:** recompute the keys and values of the whole text at every step (the plain way to run the equations of a Transformer).
- **Its limit:** the repeated work grows with the square of the length.
- **What changed:** store each token's keys and values the first time and reuse them.
- **How much better:** in the code's toy, 12 steps needed 78 key-value computations without the cache and 12 with it, with the same tokens. At 10,000 tokens that is 50,005,000 against 10,000, a factor of 5,000.5. The price is memory: 2.15 GB per 4,096-token chat for a 7B-class shape. This counts key-value projections only, not wall-clock time, which I did not measure. Origin of the technique: I do not know who first described it, and I did not check.

## Related
- [[Self-Attention]] · [[Attention]] · [[Transformer]] · [[LLM]] · [[Grouped-Query Attention]] · [[Decoding Strategies]] · [[Speculative Decoding]] · [[FlashAttention]] · [[Quantization]] · [[GPU]] · [[Big-O]] · [[PagedAttention]] · [[Sliding Window Attention]] · [[Prompt Caching]] · [[Multi-head Latent Attention]] · [[LLM Inference]] · [[Continuous Batching]] · [[vLLM]] · [[Context Window]]

## Sources
- Stanford CME 295 (Fall 2025), Lecture 3 slides, Afshine and Shervine Amidi (the lecture on inference optimizations)
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, 2023 (arXiv 2309.06180, abstract page read)
- Vaswani et al., *Attention Is All You Need*, 2017

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving KV cache:::A notebook where the AI keeps the keys and values it already worked out for the old words. Each new word only does its own work and reads the rest from the notebook.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the work with and without a KV cache and explain each symbol.
?
$$\underbrace{\sum_{t=1}^{n}t=\frac{n(n+1)}{2}}_{\text{no cache}}\qquad\text{against}\qquad \underbrace{n}_{\text{cache}}$$
- $n$: the number of tokens written so far (prompt included).
- $t$: the step number. At step $t$ a model with no cache redoes the keys and values of all $t$ tokens.
- Without the cache the work grows with the square of the length, so doubling the text makes it about 4 times more. With the cache each token is done once.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the KV cache memory and explain each symbol.
?
$$M_{\text{KV}}=2\,L\,n\,d_{\text{model}}\,b$$
- $2$: one key and one value kept for each token.
- $L$: the number of layers, each with its own keys and values.
- $n$: the number of tokens in the cache. $d_{\text{model}}$: the width of the model (all heads together).
- $b$: bytes per stored number (2 for 16-bit, 1 for 8-bit).
- Memory grows in step with the text length and is paid for every user.

#flashcards/ml/cloze #flashcards/topic/llm-serving A KV cache trades extra ==memory== for less repeated ==computation==.

#flashcards/ml/example #flashcards/topic/llm-serving For $n=4$ tokens, how many key-value pairs are made with and without the cache, and for $n=1000$?::n = 4: no cache 1+2+3+4 = 10, cache 4. n = 1000: 500,500 against 1,000, a factor of 500.5.

#flashcards/ml/example #flashcards/topic/llm-serving A toy model has $L=2$, $n=3$, $d_{\text{model}}=4$, $b=2$. What is the cache size, and what for $L=32$, $n=4096$, $d_{\text{model}}=4096$?::2 × 2 × 3 × 4 × 2 = 96 bytes. The big model: 2,147,483,648 bytes, about 2.15 GB for one user.
