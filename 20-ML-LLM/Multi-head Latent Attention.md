---
tags: [LLM, attention, inference, KV-cache, DeepSeek]
status: growing
level: 4
tier: 3
---

# Multi-head Latent Attention

**Multi-head Latent Attention (MLA)** shrinks the memory a language model keeps for earlier tokens by storing one small summary list per token instead of full keys and values.

## 🟢 1. Simple (age 12)
When a model writes a long text, it has to remember something about every earlier word. Normally that memory is big. MLA stores a small compressed note for each word and rebuilds what it needs from the note when it is needed.

**Example:** Instead of keeping a full-size photo of every page you have read, you keep a short sketch of each page and redraw the detail from the sketch when you need it.

## 🔵 2. How it works
1. When a [[Transformer]] writes text, it keeps the keys and values of all earlier tokens so it does not have to recompute them (the [[KV Cache]]).
2. That cache grows with every token and with every layer and head, and it becomes the main memory cost for long texts.
3. Fewer sets of keys and values ([[Grouped-Query Attention]]) shrink the cache by sharing them between heads.
4. MLA goes further: for each token it stores only a small **latent** list of numbers, a compressed summary of the key and the value.
5. When attention needs the keys and values, they are rebuilt from the latent list by learned matrices.
6. A trick avoids even doing the rebuilding: the rebuilding matrix can be folded into the query, so attention works directly on the small latent lists (the code below checks this with numbers).
7. The DeepSeek-V2 abstract reports a KV-cache reduction of 93.3% and 5.76 times the maximum generation throughput compared with DeepSeek 67B.
8. **The catch:** the compression has to be learned, and the model must be built for it from the start (from memory; the abstract describes it as an architecture choice).

## 🟠 3. Technical detail
- **Where it comes from:** DeepSeek-V2, a 236B-parameter mixture-of-experts model with 21B active per token and a 128K context (abstract), which combines MLA with DeepSeekMoE ([[Mixture of Experts]]). MLA also features in the models after it, such as [[DeepSeek-R1 2025]] (from memory).
- **Compared with sharing:** [[Grouped-Query Attention]] shares keys and values across groups of heads, MLA compresses them into a latent. Both shrink the [[KV Cache]].
- **Position handling (paper, Section 2.1):** RoPE ([[RoPE]]) does not fit the compressed form. It is position-sensitive for both keys and queries, so applying it to the compressed keys would put a position-dependent matrix between the query matrix and the key up-projection, and matrices do not commute, so the up-projection could not be folded into the query. The paper's fix is "decoupled RoPE": extra queries and one shared key of size $d_h^R$ carry the position, and they are cached too. That is why the cache holds $d_c+d_h^R$ numbers per layer, not just $d_c$.
- **What the paper's table says (Table 1):** the cache per token is $2n_hd_hl$ for full attention, $2n_gd_hl$ for grouped-query, $2d_hl$ for multi-query and $(d_c+d_h^R)l$ for MLA. For DeepSeek-V2 the paper sets $d_c=4d_h$ and $d_h^R=d_h/2$, so the MLA cache is $\tfrac92 d_hl$, "equal to GQA with only 2.25 groups", yet the paper says its performance is stronger than full multi-head attention.
- **What the abstract says:** MLA "guarantees efficient inference through significantly compressing the Key-Value (KV) cache into a latent vector". In the body, the paper says that during inference the up-projection for keys can be absorbed into the query matrix and the one for values into the output matrix, so the keys and values never need to be built.
- **Cost:** extra matrices to turn the latent back into keys and values, and a more complex design.
- **Related:** [[Sliding Window Attention]] and [[PagedAttention]] attack the same memory problem in other ways.

```python
import numpy as np
nh, dh, L, g, dc, dr = 32, 128, 60, 8, 512, 64                          # assumed sizes, in the range of a large model
mha, gqa, mla = 2 * nh * dh * L, 2 * g * dh * L, (dc + dr) * L          # numbers stored per token
print(f"numbers cached per token: full attention {mha:,}; grouped-query ({g} groups) {gqa:,}; latent attention {mla:,}")
print(f"latent attention is {mha / mla:.1f}x smaller than full attention and {gqa / mla:.1f}x smaller than grouped-query")
print(f"its cache equals grouped-query attention with {mla / (2 * dh * L):.2f} groups (the paper's Table 1 says 2.25 when d_c = 4 d_h and d_r = d_h / 2)")
ctx = 128_000; print(f"at {ctx:,} tokens of context, 16-bit cache: full {mha * ctx * 2 / 1e9:.1f} GB, grouped-query {gqa * ctx * 2 / 1e9:.1f} GB, latent {mla * ctx * 2 / 1e9:.1f} GB")
rng = np.random.default_rng(0); n, d, r = 50, 64, 8                      # the trick: keys come from a small latent c, so cache c and fold W_uk into the query
c = rng.standard_normal((n, r)); Wuk = rng.standard_normal((r, d)); q = rng.standard_normal(d)
s_full = q @ (c @ Wuk).T; s_lat = (Wuk @ q) @ c.T
print(f"scores from the full keys vs from the {r}-number latent only, largest difference: {np.abs(s_full - s_lat).max():.1e}  (cache {n * d} numbers -> {n * r})")
```

## 🔴 4. Deep / formal
**Cache per token.** Count how many numbers each method stores per token, over all $L$ layers. With $n_h$ heads of size $d_h$, $g$ key-value groups, a latent of size $d_c$ and a small position part $d_r$:

$$
\text{full attention: } 2\,n_h d_h L\qquad \text{grouped-query: } 2\,g\,d_h L\qquad \text{latent: } (d_c+d_r)\,L
$$

**Where:**
- $n_h$ is the number of attention heads and $d_h$ the size of each head. The 2 counts a key and a value.
- $g$ is the number of key-value groups in grouped-query attention ($g<n_h$). $L$ is the number of layers.
- $d_c$ is the size of the latent vector and $d_r$ the small extra part for position (my assumption).
- **Matters:** the latent size $d_c$ does not depend on the number of heads, so the cache can be made much smaller than full attention while the model keeps many heads.

**Example (tiny numbers):** sizes $n_h=32$, $d_h=128$, $L=60$, $g=8$, $d_c=512$, $d_r=64$. The ratios $d_c=4d_h$ and $d_r=d_h/2$ are the paper's choice for DeepSeek-V2, but the head count and layer count are my own assumptions, not DeepSeek-V2's real settings.
- Full: $2\cdot32\cdot128\cdot60=\mathbf{491{,}520}$ numbers per token.
- Grouped-query: $2\cdot8\cdot128\cdot60=\mathbf{122{,}880}$.
- Latent: $(512+64)\cdot60=\mathbf{34{,}560}$.
- Ratios: $491{,}520/34{,}560=\mathbf{14.2}$ times smaller than full, and $122{,}880/34{,}560=\mathbf{3.6}$ times smaller than grouped-query.
- At 128,000 tokens in 16-bit numbers that is about 125.8 GB, 31.5 GB and 8.8 GB.

**Why storing only the latent works.** If the keys are built from the stored latent list $c$ by a matrix $W_{UK}$, then the attention score of a query $q$ is the same whether you rebuild the key or fold the matrix into the query:

$$
q\cdot(c\,W_{UK})^{\top}=\big(W_{UK}\,q\big)\cdot c^{\top}
$$

**Where:**
- $c$ is the small latent vector stored for a token. $W_{UK}$ is the learned matrix that turns it into a key (the "up-projection").
- $q$ is the query. Both sides give the same number: the left rebuilds the key then compares, the right changes the query once and compares with the latent directly.
- **Matters:** the right-hand side never builds the full key, so the cache only holds $c$ and the work per token stays small.

**Example (tiny numbers):** let the latent be one number $c=2$, the matrix $W_{UK}=(3,\,1)$ (so the key is $(6,\,2)$) and the query $q=(1,\,4)$.
- Rebuild the key: $q\cdot(6,2)=1\cdot6+4\cdot2=\mathbf{14}$.
- Fold into the query: $W_{UK}\,q=3\cdot1+1\cdot4=7$, then $7\cdot c=7\cdot2=\mathbf{14}$.
- Meaning: the same score, and only the number $c=2$ had to be stored. The code below does this with a 50-token, 64-dimension example and a latent of 8, and the largest difference is 1.4e-14.

**Further notes:**
- **Values** are handled the same way, with the values' up-projection absorbed into the output matrix (the paper, Section 2.1).
- **The saving is a memory saving:** it cuts the cache and so lets longer texts and larger batches fit, which is where the throughput gain in the abstract comes from. The paper's 93.3% and 5.76 times compare DeepSeek-V2 with the dense DeepSeek 67B, so they also include the effect of the mixture-of-experts part and other differences, not MLA alone.

**What came before, and how this improved on it:**
- **Before:** multi-head attention with a full key and value for every head and every earlier token, then [[Grouped-Query Attention]], which shares them between heads.
- **Its limit:** the cache still grows with the number of heads (or groups), and for long texts it dominates memory.
- **What changed:** store one compressed latent per token and rebuild keys and values from it, with decoupled RoPE for the position (DeepSeek-V2, arXiv 2405.04434, 7 May 2024; abstract and parts of the body read).
- **How much better:** from the paper (copied by hand): the cache per token is $(d_c+d_h^R)l\approx\tfrac92d_hl$, the size of grouped-query attention with only 2.25 groups; and compared with the dense DeepSeek 67B, DeepSeek-V2 saves 42.5% of training costs, reduces the KV cache by 93.3% and raises the maximum generation throughput to 5.76 times (these also reflect the mixture-of-experts part; not reproduced here). In the code below (my own arithmetic with the paper's ratios and my assumed head and layer counts), the latent cache is 14.2 times smaller than full attention and 3.6 times smaller than grouped-query with 8 groups, it equals grouped-query with 2.25 groups exactly as the paper states, and the scores from the latent alone match the full keys to 1.4e-14. **Honest limits:** the paper's quality claims (stronger than full attention) are its own; I did not test them.

**Check it yourself:** run the code in level 3 above. It prints every number quoted in this block.

## Related
[[KV Cache]] · [[Grouped-Query Attention]] · [[Self-Attention]] · [[Attention]] · [[Mixture of Experts]] · [[DeepSeek-R1 2025]] · [[RoPE]] · [[Sliding Window Attention]] · [[PagedAttention]] · [[Transformer]]

## Sources
- Stanford CME 295 (Fall 2025), lecture slides, Afshine and Shervine Amidi (lectures 3 and 4, where latent attention is named)
- DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, arXiv 2405.04434 (2024): abstract and parts of the body read (Section 2.1 on MLA, Table 1, the decoupled-RoPE argument and the headline numbers), taken from an HTML copy of the paper; the author list is not given here. The numbers above were copied by hand.
- The head count and layer count in the code are my assumptions; the ratios $d_c=4d_h$ and $d_r=d_h/2$ follow the paper.
- No image: I found no free, on-topic picture for this note on Wikimedia Commons, so I added none.
- Every number in the history block was computed by the code in this note (toy data; dates and names marked "from memory" are not checked).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving Multi-head Latent Attention (MLA):::For each token store only a small latent vector (a compressed summary of key and value) and rebuild keys and values from it with learned matrices.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the cache size per token for full attention, grouped-query and latent attention, and explain each symbol.
?
$$\text{full attention: } 2\,n_h d_h L\qquad \text{grouped-query: } 2\,g\,d_h L\qquad \text{latent: } (d_c+d_r)\,L$$
- $n_h$: number of attention heads. $d_h$: size of each head. The 2 counts a key and a value.
- $g$: number of key-value groups in grouped-query attention ($g<n_h$). $L$: number of layers.
- $d_c$: size of the latent vector. $d_r$: the small extra part for position (the note says this reading is its own assumption).
- $d_c$ does not depend on the number of heads, so the cache can be much smaller than full attention while the model keeps many heads.

#flashcards/ml/example #flashcards/topic/llm-serving With $n_h=32$, $d_h=128$, $L=60$, $g=8$, $d_c=512$, $d_r=64$, how many numbers per token does each scheme store? (These sizes are the note's own assumptions, not DeepSeek-V2's real settings.)::Full: $2\cdot32\cdot128\cdot60=491{,}520$. Grouped-query (my arithmetic): $2\cdot8\cdot128\cdot60=122{,}880$. Latent (my arithmetic): $(512+64)\cdot60=34{,}560$. That is about 14 times smaller than full and 3.6 times smaller than GQA.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the identity that lets MLA avoid rebuilding the key, and explain each symbol.
?
$$q\cdot(c\,W_{UK})^{\top}=\big(W_{UK}\,q\big)\cdot c^{\top}$$
- $c$: the small latent vector stored for a token.
- $W_{UK}$: the learned matrix that turns it into a key (the up-projection).
- $q$: the query.
- Both sides give the same number. The left rebuilds the key then compares. The right changes the query once and compares with the latent directly, so the cache holds only $c$ and the work per token stays small.

#flashcards/ml/example #flashcards/topic/llm-serving $c=2$, $W_{UK}=(3,1)$, $q=(1,4)$. Show both sides give the same score.::Key $=(6,2)$, so $q\cdot(6,2)=1\cdot6+4\cdot2=14$. Fold into the query: $W_{UK}q=3\cdot1+1\cdot4=7$, then $7\cdot c=14$. Same score, and only $c=2$ had to be stored.

#flashcards/ml/cloze #flashcards/topic/llm-serving MLA stores a small ==latent== per token and rebuilds keys and values with ==learned matrices==.

#flashcards/ml/example #flashcards/topic/llm-serving Which numbers does the DeepSeek-V2 abstract report for MLA?::A KV-cache reduction of 93.3%. The note also records the paper's per-token cache $(d_c+d_h^R)l\approx\tfrac92d_hl$, the size of grouped-query attention with 2.25 groups.
