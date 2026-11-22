---
tags: [LLM, transformer, attention, GPU, efficiency, memory]
status: growing
level: 4
tier: 4
---

# FlashAttention

FlashAttention is a way of computing **exactly the same attention answer as usual, but without ever writing the huge table of attention scores to the GPU's big, slow memory**. It works through the data in small tiles that fit in the GPU's small, fast memory, and it keeps a running tally so that the answer comes out the same. The result is much less data moved, which is usually what makes attention slow. The maths of attention is in [[Self-Attention]], and the hardware side is in [[GPU]].

## 🟢 1. Simple (age 12)
Imagine you must add up a very long list of marks for a class, but your desk is tiny and the filing cabinet is across the room. The slow way: carry every sheet to your desk, write a big table on a huge poster, walk it to the cabinet, walk back to fetch it again, and so on. The fast way: take a small pile of sheets, work out that pile, **keep only a short running total on the desk**, then take the next pile. You never need the huge poster, and you almost never walk to the cabinet. The answer is exactly the same, and you finish sooner because you walked less.

**Example:** To find the average of 1,000 numbers, you do not have to write all 1,000 on a board first. You can read ten at a time and keep "sum so far" and "how many so far". FlashAttention does something similar for the attention calculation, with a slightly cleverer running total.

## 🔵 2. How it works
1. **Remember the problem.** Normal [[Attention]] makes a big square table, one score for every pair of words ($n\times n$ scores). It writes that table into the GPU's large main memory, reads it back to turn the scores into percentages, writes the percentages out, and reads them back again to blend the values. Most of the time goes to carrying numbers back and forth, not to maths (see [[GPU]]).
2. **Cut the work into tiles.** Split the queries into blocks of rows and the keys and values into blocks of rows.
3. **Load one tile of queries and one tile of keys and values** into the small fast memory on the chip.
4. **Work out that small block of scores** right there. Do not write it out.
5. **Keep a running tally** for each query row: the biggest score seen so far, the running total of the exponentials, and the running blended output. When a new tile has a bigger score, rescale the old tally so that everything stays consistent.
6. **Move to the next tile of keys and values**, update the tally, and repeat until every key has been seen.
7. **Write only the final answer** for that block of queries. The big square table was never stored.
8. **Same result:** the final numbers equal what ordinary attention gives (up to tiny rounding differences). It is a different *order* of doing the same sums.

![[memory-hierarchy-pyramid.svg]]
*A general picture of why moving data is the problem: the memory hierarchy of a computer. Near the top, memory is tiny and very fast (processor registers and cache). Further down it gets larger, slower and cheaper. A GPU has the same shape: a small, fast on-chip memory (SRAM) at the top and a big, slower main memory (HBM) below it. FlashAttention tries to do its work in the top part and visit the lower part rarely. (This is a general computer diagram, not a GPU one; the GPU levels are described in [[GPU]].) Image: Dan Lash (Wikimedia Commons user Danlash), [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:ComputerMemoryHierarchy.svg), public domain.*

## 🟠 3. Technical detail
- **What it changes and what it does not:** the output is the same as normal attention (it is *exact*, not an approximation). It changes only the order of computing and where intermediate numbers live. The number of multiplications stays about the same.
- **Why it speeds things up:** attention with a long text is limited by memory traffic, not arithmetic (the roofline idea in [[GPU]]). Moving fewer numbers to and from main memory saves time even when the maths is unchanged.
- **Why it saves memory:** the full $n\times n$ score grid never exists in main memory. Only small tiles and a few running numbers per row are kept, so memory grows with $n$ instead of with $n^2$ (equation 2 below). This is a big part of how models handle long texts.
- **The key trick, online softmax:** a [[Softmax]] needs the biggest score in the whole row and the total of all the exponentials, and you have only seen part of the row. The running tally (biggest score and running total) can be corrected when a new tile arrives (equation 1 below). Without it the tiles could not be processed one at a time.
- **Fusing:** it also joins several steps (scores, softmax, blend) into one GPU program, so intermediate results stay on the chip.
- **The backward pass:** for training, the big table is needed again to compute gradients. FlashAttention *recomputes* it tile by tile instead of storing it. This costs a little extra maths and saves a lot of memory (from memory; I did not implement the backward pass).
- **Causal masking:** for text generators (see [[Self-Attention]]), tiles that lie entirely in the "future" can be skipped completely. The code does that.
- **Related tricks:** the KV cache and grouped-query attention reduce the memory of stored keys and values during generation (see [[Self-Attention]]). FlashAttention is a different thing: it removes the score table. They are used together.
- **Versions and hardware:** there are several newer versions tuned to specific GPUs (from memory; details change quickly, and I have not checked them here).
- **Limits:** the cost still grows with the *square* of the length in arithmetic (every pair of words is still compared). Only the memory and the traffic are reduced. Models with very long contexts need other ideas as well. Also, this note's code runs on a CPU in NumPy, so it shows the *method* is exact, but it does **not** measure any speed-up. Real speed depends on the GPU and on a carefully written kernel.

The code does four things: (1) writes standard attention and a tiled version with the running tally, and checks that they agree for several tile shapes and with a causal mask; (2) checks the case where scores are so large that the plain formula overflows; (3) counts how big the biggest block held at once is; (4) uses a simple counting model of how many numbers move to and from main memory.

```python
import numpy as np
def softmax(s): e = np.exp(s - s.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
def standard(Q, K, V, causal=False):
    S = Q @ K.T / np.sqrt(Q.shape[1])                             # full N x N score grid
    if causal: S = S + np.triu(np.full(S.shape, -np.inf), 1)
    return softmax(S) @ V
def tiled(Q, K, V, Br=32, Bc=32, causal=False):
    N, d = Q.shape; O = np.zeros_like(V); biggest = 0
    for i in range(0, N, Br):
        q = Q[i:i + Br]; m = np.full(len(q), -np.inf); l = np.zeros(len(q)); acc = np.zeros((len(q), V.shape[1]))
        for j in range(0, N, Bc):
            if causal and j > i + Br - 1: break                       # this block is entirely in the future
            s = q @ K[j:j + Bc].T / np.sqrt(d); biggest = max(biggest, s.size)
            if causal: s = s + np.where(np.arange(i, i + len(q))[:, None] < np.arange(j, j + s.shape[1])[None, :], -np.inf, 0)
            m_new = np.maximum(m, s.max(1)); scale = np.exp(m - m_new)            # rescale what we had when the max grows
            p = np.exp(s - m_new[:, None])
            l = l * scale + p.sum(1); acc = acc * scale[:, None] + p @ V[j:j + Bc]; m = m_new
        O[i:i + Br] = acc / l[:, None]
    return O, biggest
rng = np.random.default_rng(0); N, d = 200, 16
Q, K, V = rng.normal(size=(3, N, d))
ref = standard(Q, K, V); out, big = tiled(Q, K, V)
print('same result:', np.allclose(ref, out), '| biggest score block held at once:', big, 'numbers vs full grid:', N * N)
for Br, Bc in ((7, 13), (200, 200), (1, 1)): print('blocks', Br, Bc, 'same:', np.allclose(ref, tiled(Q, K, V, Br, Bc)[0]))
print('causal same:', np.allclose(standard(Q, K, V, True), tiled(Q, K, V, 32, 32, True)[0]))
# big scores: the plain softmax formula overflows, the running-maximum version does not
Qb, Kb = Q * 30, K * 30
with np.errstate(all='ignore'): naive = (np.exp(Qb @ Kb.T / 4) / np.exp(Qb @ Kb.T / 4).sum(1, keepdims=True)) @ V
print('no max subtraction -> NaN present:', bool(np.isnan(naive).any()), '| tiled version finite:', bool(np.isfinite(tiled(Qb, Kb, V)[0]).all()), '| matches stable standard:', np.allclose(standard(Qb, Kb, V), tiled(Qb, Kb, V)[0]))
# a simple count of numbers moved to and from main memory (my own counting model, not a measurement)
def traffic(N, d, Br=128):
    std = 4 * N * N + 4 * N * d                       # write+read S, write+read P, plus Q, K, V in and O out
    fl = 2 * N * N * d / Br + 2 * N * d               # K and V re-read once per block of Br query rows, Q in, O out
    return std, fl
for N in (1024, 4096, 16384): s, f = traffic(N, 64); print(N, 'tokens | standard', f'{s:,.0f}', '| tiled', f'{f:,.0f}', '| ratio', round(s / f, 1))
print('score grid memory, 32 heads, fp16, N=4096:', round(32 * 4096 ** 2 * 2 / 1e9, 2), 'GB per layer')
# same result: True | biggest score block held at once: 1024 numbers vs full grid: 40000
# blocks 7 13 same: True
# blocks 200 200 same: True
# blocks 1 1 same: True
# causal same: True
# no max subtraction -> NaN present: True | tiled version finite: True | matches stable standard: True
# 1024 tokens | standard 4,456,448 | tiled 1,179,648 | ratio 3.8
# 4096 tokens | standard 68,157,440 | tiled 17,301,504 | ratio 3.9
# 16384 tokens | standard 1,077,936,128 | tiled 270,532,608 | ratio 4.0
# score grid memory, 32 heads, fp16, N=4096: 1.07 GB per layer
```
Reading the output (NumPy on a CPU, fixed seed):
- **Exactness:** the tiled version matches the standard one for tile shapes 32×32, 7×13, one giant tile (200×200) and one-number tiles (1×1), and also with a causal mask. So the result does not depend on how the work is cut up.
- **Memory:** with 200 tokens, the standard way holds a grid of 40,000 scores, while the tiled version never holds more than 1,024 at once (a 32×32 block). The gap grows with the square of the length.
- **Overflow:** with large scores, the plain formula $e^{s}/\sum e^{s}$ overflows and gives NaN (not-a-number). The running-maximum version stays finite and matches the careful standard version.
- **Traffic count:** in my simple counting model, the tiled way moves about 3.8 to 4 times fewer numbers (e.g. 68 million against 17 million at 4,096 tokens). The ratio comes out near $2B_r/d$ (here $2\times128/64=4$). This is a **counting model of my own**, with simplifications (it ignores the cost of the tally and how different GPUs behave), so it shows the *idea*. It is not a measured speed-up. I did not time anything on a GPU.
- **Memory of the score grid:** with 4,096 tokens, 32 heads and 16-bit numbers, the full score grids would take about 1.07 GB per layer, if they were all stored.

## 🔴 4. Deep / formal

**The running tally (online softmax).** Take one query row. Its scores against all keys are $s_1,\dots,s_n$ with values $v_1,\dots,v_n$. The target is $o=\sum_j \frac{e^{s_j}}{\sum_k e^{s_k}}v_j$. Process the keys in blocks. After some blocks, keep three things: $m$ (the biggest score so far), $\ell$ (the running total of $e^{s-m}$) and $a$ (the running blended sum $\sum e^{s-m}v$). When a new block with scores $s$ and values $v$ arrives, update

$$m'=\max\!\big(m,\max_j s_j\big),\quad \ell'=e^{m-m'}\,\ell+\sum_j e^{s_j-m'},\quad a'=e^{m-m'}\,a+\sum_j e^{s_j-m'}v_j,\quad o=\frac{a}{\ell}$$

**Where:**
- $m$ is the biggest score seen so far in this row. **Matters:** subtracting it before taking the exponential keeps numbers from overflowing (the code shows a NaN without it).
- $\ell$ is the running total of $e^{s-m}$, the denominator of the softmax so far. **Matters:** the final answer divides by it, so it must be exactly right.
- $a$ is the running sum of $e^{s-m}$ times the value. **Matters:** it is the blended output before the final division.
- $e^{m-m'}$ is the correction factor. **Matters:** when a new block raises the maximum from $m$ to $m'$, everything accumulated under the old maximum must shrink by this factor so that old and new parts use the same reference. Without it, blocks processed at different times would not combine.
- $o=a/\ell$ at the end is the exact attention output for this row. **Matters:** nothing was approximated, only reordered.

**Example (tiny numbers):** Scores $(2,1,3)$ with values $(10,20,30)$. First the one-shot answer: $e^2,e^1,e^3=7.389,\,2.718,\,20.086$, total $30.193$. Weights $0.2447,\,0.0900,\,0.6652$. Output $=0.2447\cdot10+0.0900\cdot20+0.6652\cdot30=24.205$.
1. Block 1 has scores $(2,1)$. $m=2$. $\ell=e^{0}+e^{-1}=1.3679$. $a=1\cdot10+0.3679\cdot20=17.358$. (Output so far: $17.358/1.3679=12.689$.)
2. Block 2 has the score $(3)$ with value 30. The new maximum is $m'=3$. Correction factor $e^{2-3}=0.3679$.
3. $\ell'=1.3679\times0.3679+e^{0}=0.5032+1=1.5032$. $a'=17.358\times0.3679+1\cdot30=6.386+30=36.386$.
4. Output $=36.386/1.5032=24.205$.
5. Meaning: the two-block route gives the same 24.205 as the one-shot route, even though the first block did not know the biggest score was coming. The correction factor 0.3679 repaired the earlier work.

**Memory for the score grid.** If the full table of scores is stored for every head, one layer needs

$$\text{memory}=h\,n^{2}\,b\ \text{bytes (standard)},\qquad \text{memory}\approx h\,n\,(d+2)\,b\ \text{bytes (tiled, rough)}$$

**Where:**
- $h$ is the number of attention heads, $n$ the number of tokens, $b$ the bytes per number (2 for 16-bit). **Matters:** the standard memory grows with the square of $n$, so doubling the text quadruples it.
- $d$ is the size of each head. **Matters:** the tiled version keeps only the output of size $n\times d$ per head plus two running numbers per row ($m$ and $\ell$), so it grows only in step with $n$.
- The tile itself is small and sits on the chip. **Matters:** its size is set by the on-chip memory, not by $n$, so it does not appear in the formula.
- The tiled formula is my rough reading of the idea, not a figure from the paper. It leaves out the inputs $Q,K,V$ (which are stored in either case).

**Example (tiny numbers):** $h=32$ heads, 16-bit numbers ($b=2$), one layer.
1. $n=4096$: $32\times4096^2\times2=1{,}073{,}741{,}824$ bytes, about $1.07$ GB.
2. $n=8192$: $32\times8192^2\times2=4.29$ GB, four times more for double the text.
3. Tiled, $n=4096$, $d=64$: $32\times4096\times66\times2=17.3$ MB (the same digits as the traffic count below, by a coincidence of the numbers).
4. Meaning: the stored score grids alone would take over a gigabyte per layer at 4,096 tokens and four gigabytes at 8,192, on top of the model. The tiled way needs tens of megabytes. That is why long texts need this trick.

**Counting the traffic.** Count the numbers moved between main memory and the chip for one head, with $n$ tokens and head size $d$. The standard way writes and re-reads the scores $S$ and the percentages $P$. The tiled way keeps the score tiles on the chip but re-reads $K$ and $V$ once for every block of $B_r$ query rows:

$$T_{\text{std}}\approx 4n^{2}+4nd,\qquad T_{\text{tiled}}\approx \frac{2n^{2}d}{B_r}+2nd,\qquad \frac{T_{\text{std}}}{T_{\text{tiled}}}\xrightarrow{\,n\to\infty\,}\frac{2B_r}{d}$$

**Where:**
- $4n^2$ counts writing and reading the score grid $S$ and the percentage grid $P$ (each $n^2$ numbers, each moved twice). **Matters:** it is the cost that FlashAttention removes.
- $4nd$ counts reading $Q$, $K$, $V$ and writing the output $O$, which is needed either way.
- $B_r$ is the number of query rows in a tile. **Matters:** a bigger tile means $K$ and $V$ are re-read fewer times. It is limited by how much fits in the on-chip memory (the larger the head size $d$, the fewer rows fit).
- $\frac{2n^2d}{B_r}$ is the cost of re-reading $K$ and $V$ ($2nd$ numbers) once for each of $n/B_r$ query blocks. **Matters:** this is the price paid for never storing the grid.
- The ratio $2B_r/d$ is the saving in this model. **Matters:** it grows with a bigger on-chip memory (bigger $B_r$) and shrinks for larger heads (bigger $d$). The real paper's analysis is more careful (from memory, it relates the traffic to the size of the on-chip memory $M$); this is only my simple count.

**Example (tiny numbers):** $n=4096$, $d=64$, $B_r=128$.
1. Standard: $4\times4096^2+4\times4096\times64=67{,}108{,}864+1{,}048{,}576=68{,}157{,}440$ numbers.
2. Tiled: $2\times4096^2\times64/128+2\times4096\times64=16{,}777{,}216+524{,}288=17{,}301{,}504$ numbers.
3. Ratio: $68{,}157{,}440/17{,}301{,}504=3.94$, near the limit $2\times128/64=4$.
4. Meaning: in this simple model the tiled way moves about a quarter as many numbers. If memory traffic is what limits attention, that points to a time saving of the same order. Real GPUs differ, so the actual speed-up is something to measure on the hardware, which I did not do here.

**Other points (from memory, not checked here):**
- FlashAttention was introduced by Dao and colleagues (2022). I did not open the paper for this note, and the paper's analysis of the traffic is more detailed than my count.
- Reported speed-ups depend on the GPU, the text length and the version, and I have no measurements of my own.
- The same tiling idea has been applied to other operations, and it is now built into common deep-learning libraries (from memory; check your own setup).
- Approximate methods (sparse or linear attention) reduce the arithmetic as well, at the price of changing the answer. FlashAttention does not.

**What came before, and how this improved on it:**
- **Before:** standard attention writes the whole $n\times n$ table of scores to the GPU's main memory, then reads it back.
- **Its limit:** the abstract says transformers are "slow and memory-hungry on long sequences" because attention is quadratic, and that approximate methods often give no real speedup (Dao et al., arXiv 2205.14135, 27 May 2022, abstract read; first author from memory).
- **What changed:** the same exact result, computed in small tiles that stay in fast on-chip memory, with a running maximum and sum for the softmax.
- **How much better:** the paper reports 15% faster end to end on BERT-large (sequence length 512), 3 times on GPT-2 (length 1K) and 2.4 times on long-range arena (abstract; not reproduced, since my laptop has no GPU). In the code below, the tiled version matches the standard one to 5.0e-16, and for a sequence of 16,384 the full score table would be 512 MiB per head against 2.0 MiB for a 64-wide tile.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0); n, d, B = 512, 32, 64
Q, K, Vv = (rng.standard_normal((n, d)) for _ in range(3))
S = Q @ K.T / np.sqrt(d); P = np.exp(S - S.max(1, keepdims=True)); ref = (P / P.sum(1, keepdims=True)) @ Vv     # standard: builds the full n x n table
out = np.zeros((n, d)); m = np.full(n, -np.inf); l = np.zeros(n)
for j in range(0, n, B):                                                       # tiled: one block of keys at a time, running max and sum
    s = Q @ K[j:j + B].T / np.sqrt(d); mn = np.maximum(m, s.max(1)); p = np.exp(s - mn[:, None]); a = np.exp(m - mn)
    out = out * a[:, None] + p @ Vv[j:j + B]; l = l * a + p.sum(1); m = mn
out /= l[:, None]
print("largest difference between standard and tiled attention:", f"{np.abs(out - ref).max():.1e}")
for L in (1024, 4096, 16384): print(f"sequence length {L:6,}: full score table {L * L * 2 / 2**20:8,.0f} MiB per head (16-bit numbers); one {B}-wide tile {L * B * 2 / 2**20:6,.1f} MiB")
```

## Related
[[Self-Attention]] · [[Attention]] · [[GPU]] · [[Softmax]] · [[Transformer]] · [[Matrix]] · [[LLM]] · [[Quantization]] · [[Positional Encoding]] · [[RoPE]] · [[Big-O]] · [[KV Cache]] · [[Mixed Precision Training]] · [[Triton]] · [[Kernel Fusion]] · [[CUDA]]

## Sources
- Dao, Fu, Ermon, Rudra and Ré (2022), "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (cited from memory when the note was written; the abstract, arXiv 2205.14135, was read later for the history block, and the speedups there are quoted from it, not reproduced).
- Milakov and Gimelshein (2018), "Online normalizer calculation for softmax", the running-softmax idea (from memory).
- Figure: [Wikimedia Commons, ComputerMemoryHierarchy.svg](https://commons.wikimedia.org/wiki/File:ComputerMemoryHierarchy.svg), by Dan Lash, public domain.
- The code, the traffic count, the memory sizes and every Example number were run for this note (NumPy only, on a CPU). No GPU timing was done.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving FlashAttention:::Computing attention in small tiles that fit in fast on-chip memory, so the huge score grid is never written to slow memory. The result is the same as normal attention but with much less memory traffic.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the online softmax update used by FlashAttention and explain each symbol.
?
$$m'=\max\!\big(m,\max_j s_j\big),\quad \ell'=e^{m-m'}\,\ell+\sum_j e^{s_j-m'},\quad a'=e^{m-m'}\,a+\sum_j e^{s_j-m'}v_j,\quad o=\frac{a}{\ell}$$
- $s_j$: the attention score for key $j$ in the new block. $v_j$: its value vector.
- $m$: the biggest score seen so far in this row. Subtracting it before the exponential keeps numbers from overflowing.
- $\ell$: the running total of $e^{s-m}$, the softmax denominator so far. The final answer divides by it, so it must be exact.
- $a$: the running weighted sum of values. $e^{m-m'}$ rescales the old totals whenever a new maximum appears.
- $o=a/\ell$: the output once all blocks are done.

#flashcards/ml/formula #flashcards/topic/llm-serving Write standard and tiled attention memory and explain each symbol.
?
$$\text{memory}=h\,n^{2}\,b\ \text{bytes (standard)},\qquad \text{memory}\approx h\,n\,(d+2)\,b\ \text{bytes (tiled, rough)}$$
- $h$: the number of attention heads. $n$: the number of tokens. $b$: bytes per number (2 for 16-bit). Standard memory grows with $n^2$, so doubling the text quadruples it.
- $d$: the size of each head. The tiled version keeps only the output of size $n\times d$ per head plus two running numbers per row ($m$ and $\ell$).
- Tiled memory grows only in step with $n$.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the memory traffic of standard and tiled attention and explain each symbol.
?
$$T_{\text{std}}\approx 4n^{2}+4nd,\qquad T_{\text{tiled}}\approx \frac{2n^{2}d}{B_r}+2nd,\qquad \frac{T_{\text{std}}}{T_{\text{tiled}}}\xrightarrow{\,n\to\infty\,}\frac{2B_r}{d}$$
- $4n^2$: writing and reading the score grid $S$ and the percentage grid $P$ (each $n^2$ numbers, each moved twice). This is the cost FlashAttention removes.
- $4nd$: reading $Q$, $K$, $V$ and writing the output $O$, which is needed either way.
- $B_r$: the number of query rows in a tile. A bigger tile means $K$ and $V$ are re-read fewer times.
- The ratio approaches $2B_r/d$.

#flashcards/ml/cloze #flashcards/topic/llm-serving FlashAttention never writes the full ==score grid== to slow memory, and it uses a ==running== softmax to stay exact.

#flashcards/ml/example #flashcards/topic/llm-serving Scores (2, 1, 3) with values (10, 20, 30). What is the one-shot attention output?::Weights 0.2447, 0.0900, 0.6652. Output 0.2447 × 10 + 0.0900 × 20 + 0.6652 × 30 = 24.205. The block-by-block method gives the same number.

#flashcards/ml/example #flashcards/topic/llm-serving For $h=32$ heads and 16-bit numbers, how much memory does the score grid need at $n=4096$ and $n=8192$, and the tiled version at $n=4096$, $d=64$?::n = 4096: 1,073,741,824 bytes, about 1.07 GB. n = 8192: 4.29 GB, four times more. Tiled: 32 × 4096 × 66 × 2 = 17.3 MB.

#flashcards/ml/example #flashcards/topic/llm-serving For $n=4096$, $d=64$, $B_r=128$, what are the standard and tiled memory traffic and their ratio?::Standard 68,157,440 numbers. Tiled 17,301,504. Ratio 3.94, near the limit 2 × 128 / 64 = 4.

#flashcards/ml/image #flashcards/topic/llm-serving ![[memory-hierarchy-pyramid.svg]]
What does this picture show?
?
Why moving data is the problem: the memory hierarchy of a computer. Near the top, memory is tiny and very fast (registers and cache). Further down it gets larger, slower and cheaper. A GPU has the same shape: small fast on-chip memory (SRAM) at the top, big slower main memory (HBM) below. FlashAttention does its work in the top part and visits the lower part rarely.
