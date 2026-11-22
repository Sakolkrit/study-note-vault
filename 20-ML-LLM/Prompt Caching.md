---
tags: [LLM, inference, cost, KV-cache, API]
status: growing
level: 4
tier: 2
---

# Prompt Caching

**Prompt caching** means a language-model service remembers the work it did on the start of a prompt, so the next prompt that starts the same way is cheaper and faster.

## 🟢 1. Simple (age 12)
If you ask a friend ten questions about the same long story, you do not want them to re-read the whole story each time. You want them to read it once, and then answer each question. Prompt caching does that for a language model.

**Example:** A teacher who gives the same 20-page reading to a class every week, plus a different short question each time. If every pupil keeps their own notes of the reading, only the new question needs new work.

## 🔵 2. How it works
1. A language model reads your prompt token by token and builds an internal memory of it, the [[KV Cache]].
2. Many prompts repeat the same long start: a system message, instructions, a document, a few examples ([[Prompt Engineering]], [[In-Context Learning]]).
3. Without caching, the service redoes all of the reading for every call.
4. With caching, the service keeps the memory for the repeated start (the **prefix**) and reuses it.
5. Only the new part at the end has to be read fresh.
6. The result is the same answer as before, but cheaper and faster. The lecture slide shows a provider's pricing page where cached input tokens cost less than normal input tokens (I do not repeat the prices; they change).
7. **The catch:** the saving needs the repeated text to be at the **start** of the prompt and exactly the same. Putting the changing part at the beginning breaks the match. Caches also expire after a while, and providers differ in the rules (from memory).

## 🟠 3. Technical detail
- **What it is built on:** the [[KV Cache]], the stored keys and values of earlier tokens. Prompt caching keeps that cache between separate calls instead of throwing it away after each answer.
- **Prefix only:** because each token's keys and values depend only on the tokens before it, a shared start can be reused, but a change early on invalidates everything after it.
- **Design rule:** put stable material (instructions, tools, documents, examples) first and the user's changing question last.
- **Where it shows up:** long system prompts, [[Agent]] loops that re-send the whole history every step, [[RAG]] with a fixed document, and [[Contextual Retrieval]], which the lecture slide links to prompt caching to make adding context to every chunk affordable.
- **Not the same as:** an answer cache (storing whole answers for repeated questions) or [[PagedAttention]] (how the cache is laid out in memory).
- **Pricing and limits** are provider details that change, so check the provider's page.

```python
import numpy as np
rng = np.random.default_rng(0); d = 16
P, U, calls = 10000, 200, 50
print(f"{calls} calls, each = the same {P:,}-token prefix + {U} new tokens")
print(f"  without a cache: {calls * (P + U):,} tokens processed; with the prefix cached once: {P + calls * U:,} ({calls * (P + U) / (P + calls * U):.1f}x fewer)")
Wq, Wk, Wv = (rng.standard_normal((d, d)) for _ in range(3))
pre, new = rng.standard_normal((40, d)), rng.standard_normal((6, d))     # tiny stand-ins: 40 prefix tokens, 6 new ones
def attend(x, K, V, start):                                            # causal attention for the rows x placed at positions start...
    q = x @ Wq; s = q @ K.T / np.sqrt(d)
    for i in range(len(x)): s[i, start + i + 1:] = -np.inf
    e = np.exp(s - s.max(1, keepdims=True)); return (e / e.sum(1, keepdims=True)) @ V
full_x = np.vstack([pre, new]); out_full = attend(full_x, full_x @ Wk, full_x @ Wv, 0)[-6:]
Kc, Vc = pre @ Wk, pre @ Wv                                            # the cache: keys and values of the prefix, computed once
out_cached = attend(new, np.vstack([Kc, new @ Wk]), np.vstack([Vc, new @ Wv]), 40)
print("outputs for the new tokens, full recompute vs cached prefix, largest difference:", f"{np.abs(out_full - out_cached).max():.1e}")
```

## 🔴 4. Deep / formal
**What caching saves.** Suppose a prompt is a shared prefix of $P$ tokens plus $u$ new tokens, and you make $m$ calls. The number of tokens the model has to process is

$$
\text{without caching: } m\,(P+u)\qquad\text{with caching: } P+m\,u
$$

**Where:**
- $P$ is the length of the shared prefix, in tokens. $u$ is the length of the part that changes in each call. $m$ is the number of calls.
- Without caching, every call processes the prefix again: $m$ copies of $P$.
- With caching, the prefix is processed once ($P$) and only the new part is processed in each call ($mu$).
- **Matters:** the saving grows with $P$ and with $m$. If the prefix is short, or you make only one call, there is almost nothing to save. This counts tokens processed; prices per cached token are lower still but are the provider's choice.

**Example (tiny numbers):** $P=10{,}000$, $u=200$, $m=50$.
- Without caching: $50\cdot(10{,}000+200)=\mathbf{510{,}000}$ tokens.
- With caching: $10{,}000+50\cdot200=\mathbf{20{,}000}$ tokens.
- Ratio: $510{,}000/20{,}000=\mathbf{25.5}$ times fewer.
- Meaning: for a long repeated prefix and many calls, nearly all of the work disappears.

**Why the answer does not change.** The keys and values of the prefix tokens do not depend on what comes after them (attention is causal; see [[Self-Attention]]). So reading the prefix again would produce exactly the numbers that were stored. The code below checks this on a tiny attention layer: the output for the new tokens with the cached prefix matches a full recompute to 0.0e+00.

**Further notes:**
- **Memory cost:** the cache has to be kept somewhere ([[KV Cache]]), which is why providers expire it after a time.
- **Exact match needed:** a single changed character in the prefix breaks the reuse from that point on.

**What came before, and how this improved on it:**
- **Before:** each API call processed the whole prompt from scratch.
- **Its limit:** long repeated prefixes (instructions, documents, agent histories) made calls slow and costly, and the repeated work was pure waste.
- **What changed:** keep the prefix's keys and values between calls and reuse them. The lecture slide shows a provider pricing page with a cheaper rate for cached input tokens (screenshot dated 10 November 2025, which I do not copy because prices change). I found no paper to cite; the details of each provider are from memory or their docs, which I did not open.
- **How much better:** in the code below, 50 calls with a 10,000-token shared prefix and 200 new tokens each process 510,000 tokens without a cache and 20,000 with one, 25.5 times fewer. The cached and recomputed outputs of a small attention layer agree to 0.0e+00, so the saving costs no change in the answer.

**Check it yourself:** run the code in level 3 above. It prints every number quoted in this block.

## Related
[[KV Cache]] · [[PagedAttention]] · [[Prompt Engineering]] · [[In-Context Learning]] · [[Agent]] · [[RAG]] · [[Contextual Retrieval]] · [[Self-Attention]] · [[LLM]] · [[Context Window]] · [[System Prompt]]

## Sources
- Stanford CME 295 (Fall 2025), lecture slides, Afshine and Shervine Amidi (lecture 7, the slides on contextual retrieval and prompt caching, with a pricing screenshot)
- No paper found for this one; provider rules are from memory.
- No image: I found no free, on-topic picture for this note on Wikimedia Commons, so I added none.
- Every number in the history block was computed by the code in this note (toy data; dates and names marked "from memory" are not checked).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving Prompt caching:::Keep the keys and values (KV cache) of a repeated prompt start (the prefix) and reuse them, so only the new part at the end has to be read fresh.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the tokens processed with and without caching and explain each symbol.
?
$$\text{without caching: } m\,(P+u)\qquad\text{with caching: } P+m\,u$$
- $P$: length of the shared prefix in tokens. $u$: length of the part that changes in each call. $m$: number of calls.
- Without caching every call processes the prefix again: $m$ copies of $P$.
- With caching the prefix is processed once ($P$) and only the new part each call ($mu$).
- The saving grows with $P$ and $m$. A short prefix or a single call saves almost nothing. This counts tokens processed. Prices per cached token are the provider's choice.

#flashcards/ml/example #flashcards/topic/llm-serving $P=10{,}000$, $u=200$, $m=50$. How many tokens are processed with and without caching?::Without: $50\cdot10{,}200=510{,}000$. With: $10{,}000+50\cdot200=20{,}000$. That is 25.5 times fewer.

#flashcards/ml/cloze #flashcards/topic/llm-serving Prompt caching only helps if the repeated text is at the ==start== of the prompt and ==exactly the same==.

#flashcards/ml/concept #flashcards/topic/llm-serving Why should the changing part of a prompt go at the end?::A cache matches from the start. Putting the changing part first breaks the match, so nothing is reused.
