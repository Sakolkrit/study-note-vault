---
tags: [LLM, transformer, attention, inference, KV-cache, GQA, MQA, optimization]
status: growing
level: 4
tier: 3
---

# Grouped-Query Attention

Often written **GQA**. It sits between two older designs, **multi-head attention (MHA)** and **multi-query attention (MQA)**.

## 🟢 1. Simple (age 12)
In a [[Transformer]], attention has several "readers" (called heads), and each reader normally keeps its **own private notebook** of what it has seen. Grouped-query attention lets **small teams of readers share one notebook**. The readers still ask their own questions, but they look things up in a shared notebook, so the AI needs much less memory.

**Example:** A classroom has 32 students and each one keeps a separate copy of the class notes. That takes 32 notebooks. Now put them in 8 groups of 4, and each group shares one set of notes. The students still think for themselves, but you only need 8 notebooks.

## 🔵 2. How it works
1. In ordinary **multi-head attention** every head $i$ has its own query, its own key and its own value ([[Self-Attention]]). With $h$ heads there are $h$ sets of keys and values to store in the [[KV Cache]].
2. In **multi-query attention** all heads share **one** set of keys and values. The cache becomes $h$ times smaller, but quality can drop a little.
3. In **grouped-query attention** the heads are split into $G$ **groups**. Each group shares one set of keys and values. $G=h$ is multi-head attention, $G=1$ is multi-query attention, and anything in between is GQA.
4. Each head still has **its own query**, so heads can still look for different things.
5. Nothing else in the model changes. Only the key and value matrices get thinner (they make $G$ heads' worth of numbers, not $h$).
6. The saving shows up when the model writes text: the cache is smaller, so there is less to read at each step, and more chats fit on one GPU.

## 🟠 3. Technical detail
- **Layout:** $h$ query heads, $G$ key/value heads, $h$ divisible by $G$. Query head $i$ reads key/value group $\lfloor i/(h/G)\rfloor$.
- **What it shrinks:** the [[KV Cache]] and the number of weights in the key and value projections. It does not change how many query computations there are.
- **Typical settings:** many current open models use a small number of key/value heads such as 8 with 32 or more query heads (a common pattern; check the model you use).
- **Getting one from an old model:** the GQA paper describes *uptraining*: take a multi-head checkpoint, merge its key/value heads into groups, and train a little more. Its abstract says this takes about 5% of the original pre-training compute.
- **Quality:** the paper's abstract says uptrained GQA reaches quality close to multi-head attention, with speed comparable to multi-query attention.
- **Related ideas in the same lecture:** sliding-window attention (each token only looks at nearby tokens, like the [[Receptive Field]] of a convolution) and latent attention that stores a compressed form of keys and values.

```python
import numpy as np
rng = np.random.RandomState(1)
softmax = lambda s: np.exp(s - s.max(-1, keepdims=True)) / np.exp(s - s.max(-1, keepdims=True)).sum(-1, keepdims=True)

def attention(Q, K, V, G):
    """Q: (h, n, dh) one set per query head; K, V: (G, n, dh) one set per group. Query head i uses group i // (h // G)."""
    h, n, dh = Q.shape
    out = np.zeros_like(Q)
    for i in range(h):
        g = i // (h // G)
        out[i] = softmax(Q[i] @ K[g].T / np.sqrt(dh)) @ V[g]
    return out

h, n, dh = 8, 5, 4
Q = rng.randn(h, n, dh)
K8, V8 = rng.randn(h, n, dh), rng.randn(h, n, dh)
# 1) G = h is ordinary multi-head attention: compare with a separate, plain implementation
plain = np.stack([softmax(Q[i] @ K8[i].T / np.sqrt(dh)) @ V8[i] for i in range(h)])
print('G = h equals plain multi-head attention:', np.allclose(attention(Q, K8, V8, 8), plain))
# 2) G = 2: copying each group's K and V to its 4 query heads gives the same answer as the grouped code
K2, V2 = rng.randn(2, n, dh), rng.randn(2, n, dh)
rep = np.stack([softmax(Q[i] @ K2[i // 4].T / np.sqrt(dh)) @ V2[i // 4] for i in range(h)])
print('G = 2 equals copying K,V to each head:', np.allclose(attention(Q, K2, V2, 2), rep))
# 3) G = 1 is multi-query attention: all heads read one K and one V
K1, V1 = rng.randn(1, n, dh), rng.randn(1, n, dh)
print('G = 1 (multi-query) output shape:', attention(Q, K1, V1, 1).shape, '| K,V numbers stored per token per layer:', 2 * 1 * dh, 'against', 2 * h * dh)
# 4) cache size for a 32-head, 32-layer model, 128 numbers per head, 4096 tokens, 16-bit numbers
L, H, DH, N, B = 32, 32, 128, 4096, 2
for name, G in (('MHA', 32), ('GQA', 8), ('MQA', 1)):
    size = 2 * L * G * DH * N * B
    print(name, 'G =', G, '| KV cache', size, 'bytes =', round(size / 1e9, 3), 'GB | ratio to MHA', round(size / (2 * L * H * DH * N * B), 4))
# 5) the extra weights: K and V projection matrices (width 4096 -> G*128 numbers each)
for name, G in (('MHA', 32), ('GQA', 8), ('MQA', 1)):
    print(name, 'K and V projection weights per layer:', 2 * 4096 * G * DH)
```

## 🔴 4. Deep / formal
**Which group a head reads.** With $h$ query heads and $G$ groups:

$$
g(i)=\left\lfloor \frac{i}{h/G}\right\rfloor,\qquad i=0,1,\dots,h-1
$$

**Where:**
- $i$ is the number of the query head, counting from 0.
- $h$ is the number of query heads and $G$ is the number of key/value groups.
- $h/G$ is how many query heads share one group, and $\lfloor\cdot\rfloor$ rounds down to a whole number.
- $g(i)$ is the group whose keys and values head $i$ uses.
- **Matters:** this rule splits the heads into equal neighbours. Without $G$ dividing $h$, the groups would be uneven.

**Example (tiny numbers):** $h=8$ query heads and $G=2$ groups, so each group has $8/2=4$ heads.
- Heads $0,1,2,3$: $g=\lfloor i/4\rfloor=\mathbf{0}$.
- Heads $4,5,6,7$: $g=\lfloor i/4\rfloor=\mathbf{1}$.
- Meaning: two sets of keys and values serve eight query heads. The code checks that this gives the same result as copying each group's keys and values to its four heads.

**The size of the cache.** The cache for one chat depends on the number of key/value heads, not on the number of query heads:

$$
M_{\text{KV}}=2\,L\,G\,d_h\,n\,b
$$

**Where:**
- $2$ is for keys and values, $L$ is the number of layers, $G$ is the number of key/value heads (groups), $d_h$ is the number of numbers per head, $n$ is the number of tokens, and $b$ is the bytes per number (see [[KV Cache]]).
- **Matters:** the cache is proportional to $G$. With $G=h$ (multi-head) it is the largest, and with $G=1$ (multi-query) the smallest. Quality moves the other way, which is the trade-off GQA sits in the middle of.

**Example (tiny numbers):** $L=2$, $d_h=4$, $n=3$, $b=2$.
- Multi-head with $h=8$: $2\cdot2\cdot8\cdot4\cdot3\cdot2=\mathbf{768}$ bytes.
- Grouped with $G=2$: $2\cdot2\cdot2\cdot4\cdot3\cdot2=\mathbf{192}$ bytes.
- Meaning: $192/768=0.25=G/h$, a quarter of the memory. For the code's 32-layer, 32-head model, $G=8$ gives $0.537$ GB against $2.147$ GB.

**Further notes:**
- **Why shrinking K and V is the right place to cut:** during writing, the model is limited by the bytes it must read at each step. Keys and values are read at every step by every head, so sharing them cuts the biggest read.
- **What it does not do:** it does not cut the arithmetic of attention scores by much, and it does not make a model remember more.
- **Measured in the code:** the weights of the key and value projections shrink too: $33{,}554{,}432$ numbers per layer for $G=32$, $8{,}388{,}608$ for $G=8$ and $1{,}048{,}576$ for $G=1$ (width 4096, 128 numbers per head).
- **Not tested:** model quality. The code counts memory and checks the maths, but trains nothing, so the quality statements come from the paper's abstract.

**What came before, and how this improved on it:**
- **Before:** multi-head attention from the original Transformer (2017), with one key and value set per head, and multi-query attention (Shazeer, arXiv 1911.02150, 6 November 2019, abstract read), where all heads share one set.
- **Its limit:** multi-head attention needs a large cache and heavy memory reads when writing text. Multi-query attention is fast but, according to the GQA abstract, can lose quality and would need a separate model trained just for it.
- **What changed:** use an in-between number of key/value groups, and convert an existing multi-head model by uptraining (Ainslie et al., arXiv 2305.13245, 22 May 2023, abstract read).
- **How much better:** counted in this note's code for a 32-layer, 32-head model: the cache is 2.147 GB for multi-head, 0.537 GB for GQA with 8 groups (4 times smaller) and 0.067 GB for multi-query (32 times smaller). The paper's abstract claims quality close to multi-head and speed comparable to multi-query; I did not reproduce either.

## Related
- [[KV Cache]] · [[Self-Attention]] · [[Attention]] · [[Transformer]] · [[LLM]] · [[Receptive Field]] · [[Quantization]] · [[FlashAttention]] · [[GPU]] · [[Sliding Window Attention]] · [[PagedAttention]] · [[Multi-head Latent Attention]]

## Sources
- Stanford CME 295 (Fall 2025), Lectures 2 and 3 slides, Afshine and Shervine Amidi
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, 2023 (arXiv 2305.13245, abstract read)
- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*, 2019 (arXiv 1911.02150, abstract read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving Grouped-query attention (GQA):::Split the $h$ query heads into $G$ groups. Each group shares one set of keys and values. $G=h$ is multi-head attention and $G=1$ is multi-query attention.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the rule that assigns a query head to a group and explain each symbol.
?
$$g(i)=\left\lfloor \frac{i}{h/G}\right\rfloor,\qquad i=0,1,\dots,h-1$$
- $i$: the number of the query head, counting from 0.
- $h$: the number of query heads. $G$: the number of key/value groups.
- $h/G$: how many query heads share one group. $\lfloor\cdot\rfloor$ rounds down.
- $g(i)$: the group whose keys and values head $i$ uses.
- It splits the heads into equal neighbours. If $G$ does not divide $h$ the groups would be uneven.

#flashcards/ml/example #flashcards/topic/llm-serving $h=8$ query heads and $G=2$ groups. Which group does each head use?::Each group has $8/2=4$ heads. Heads 0 to 3 give $\lfloor i/4\rfloor=0$. Heads 4 to 7 give 1. Two sets of keys and values serve eight query heads.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the KV-cache size and explain each symbol.
?
$$M_{\text{KV}}=2\,L\,G\,d_h\,n\,b$$
- $2$: keys and values. $L$: number of layers. $G$: number of key/value heads (groups).
- $d_h$: numbers per head. $n$: number of tokens. $b$: bytes per number.
- The cache is proportional to $G$: largest for multi-head ($G=h$), smallest for multi-query ($G=1$). Quality moves the other way, which is the trade-off GQA sits in the middle of.

#flashcards/ml/example #flashcards/topic/llm-serving $L=2$, $d_h=4$, $n=3$, $b=2$. Cache size for multi-head with $h=8$ and for GQA with $G=2$?::Multi-head: $2\cdot2\cdot8\cdot4\cdot3\cdot2=768$ bytes. GQA: $2\cdot2\cdot2\cdot4\cdot3\cdot2=192$ bytes. $192/768=0.25=G/h$, a quarter of the memory.

#flashcards/ml/cloze #flashcards/topic/llm-serving In GQA each head still has its own ==query==, so heads can look for different things. Only keys and values are ==shared==.

#flashcards/ml/reverse #flashcards/topic/llm-serving Multi-query attention:::All heads share one set of keys and values. The cache is $h$ times smaller, but quality can drop a little.
