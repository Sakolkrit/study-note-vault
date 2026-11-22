---
tags: [LLM, transformer, attention, long-context, sliding-window, efficiency]
status: growing
level: 4
tier: 3
---

# Sliding Window Attention

Often shortened to **SWA**. Each token looks only at the **nearest few tokens** instead of the whole text, which makes long texts affordable.

## 🟢 1. Simple (age 12)
Reading a very long book, you do not keep every earlier page in your head at once. You mostly remember the **last few pages** and rely on your notes for the rest. Sliding window attention gives the AI the same rule: each word looks only at **the words just before it**, a window that slides along the text as it goes. It still learns about far-back things because information is passed along layer by layer.

**Example:** In a queue of children passing a message, each child only hears the child right behind them. But after many children, the message has travelled a long way down the line.

## 🔵 2. How it works
1. Normal [[Self-Attention]] lets every token look at every earlier token. For a text of $n$ tokens that is about $n^2/2$ comparisons, so doubling the text makes the work four times larger.
2. In sliding window attention each token may look only at itself and the $w-1$ tokens before it (a window of size $w$).
3. The number of comparisons becomes about $n\times w$, which grows in a straight line with the length.
4. Far-away words are not lost. A layer lets token $i$ read from tokens within its window, and those tokens already read from theirs in the layer below. So after $L$ layers, information can come from about $L\times(w-1)$ tokens back.
5. This is like the **receptive field** of a convolution network, which grows with each layer ([[Receptive Field]]), as the lecture notes.
6. **Local and global mixes:** another idea is to give a few special tokens full vision (global attention) and keep the rest local. Some models interleave local and global layers (lecture).
7. In generation, the [[KV Cache]] only needs to keep the last $w$ tokens of each layer, which caps its memory.

## 🟠 3. Technical detail
- **The mask:** a band: token $i$ may attend to positions $j$ with $i-w<j\le i$ (the code prints a $6\times6$ example with $w=3$).
- **Longformer:** the Longformer abstract says standard self-attention scales quadratically, and its attention scales linearly by combining local windowed attention with task-motivated global attention (arXiv 2004.05150, read). The lecture shows it with special [CLS] tokens attending globally.
- **Mistral 7B:** its abstract says it uses grouped-query attention for faster inference together with sliding window attention to handle long sequences at lower cost (arXiv 2310.06825, read; see [[Grouped-Query Attention]]). The lecture shows its window picture.
- **Reach through depth:** the code measures it by passing influence through stacked layers: for window 8, the number of tokens that can influence the last token is $8, 15, 29, 57$ after $1, 2, 4, 8$ layers, equal to $L(w-1)+1$.
- **What is lost:** direct access. A fact far outside the window can only reach the current token through intermediate positions, which may blur it. I did not test retrieval from far back.
- **Memory:** with a cache limited to the window, memory no longer grows with total length. The code compares a 16,000-token text with a 4,096-token window: 3.91 times less.
- **Related tricks:** [[FlashAttention]] makes full attention faster without approximating it, and sparse patterns approximate it. Sliding windows are one such pattern.

```python
import numpy as np
def window_mask(n, w):
    """token i may look at itself and the w-1 tokens before it"""
    i = np.arange(n)[:, None]; j = np.arange(n)[None, :]
    return (j <= i) & (j > i - w)
print(window_mask(6, 3).astype(int))
n, w = 4096, 512
full_cells = n * (n + 1) // 2; win_cells = int(window_mask(n, w).sum())
print(f'n = {n}, w = {w} | causal full attention {full_cells} cells | sliding window {win_cells} cells | ratio {full_cells / win_cells:.2f}')
# information still travels: stack L layers and see how far back the last token can be influenced
def reach(n, w, L):
    M = window_mask(n, w).astype(float); R = np.eye(n)
    for _ in range(L): R = (M @ R > 0).astype(float)
    return int(R[n - 1].sum())                                  # number of input tokens that can influence the last one
for L in (1, 2, 4, 8):
    print(f'{L} layers, window 8 -> last token is influenced by {reach(200, 8, L)} tokens | formula L*(w-1)+1 = {L * 7 + 1}')
# with a window at least as long as the text, it is ordinary causal attention
print('window >= n equals full causal mask:', np.array_equal(window_mask(10, 10), np.tril(np.ones((10, 10), bool))))
# memory: the cache only needs the last w tokens per layer
L_, d_, B_ = 32, 4096, 2
print('KV cache bytes, 16000 tokens, full:', 2 * L_ * 16000 * d_ * B_, '| window 4096:', 2 * L_ * 4096 * d_ * B_, '| ratio', round(16000 / 4096, 2))
```

## 🔴 4. Deep / formal
**How many comparisons.** For a text of $n$ tokens and a causal window of $w$ (each token sees itself and the $w-1$ before it, but never more than exist):

$$
\text{cells}=n\,w-\frac{w(w-1)}{2}\qquad\text{against full causal attention}\qquad\frac{n(n+1)}{2}
$$

**Where:**
- $n$ is the number of tokens and $w$ is the window size (with $w\le n$).
- Each token near the start sees fewer than $w$ tokens, and the $-\tfrac{w(w-1)}{2}$ term takes those missing cells off.
- $\tfrac{n(n+1)}{2}$ is the count when every token sees every earlier token.
- **Matters:** the window version grows like $n\,w$, the full version like $n^2$. When $w\ll n$ the saving is large, and the ratio is roughly $n/(2w)$ when $w\ll n$. Without the correction term the formula would over-count the start.

**Example (tiny numbers):** $n=6$, $w=3$.
- Window: $6\cdot3-3\cdot2/2=18-3=\mathbf{15}$ cells (the matrix in the code has rows of $1,2,3,3,3,3$ ones, which sum to 15).
- Full causal: $6\cdot7/2=\mathbf{21}$.
- Meaning: 15 comparisons instead of 21 here. At $n=4096$, $w=512$ the code counts $1{,}966{,}336$ against $8{,}390{,}656$ (4.27 times fewer).

**How far information can travel.** After $L$ stacked layers with window $w$:

$$
\text{reach}=L\,(w-1)+1
$$

**Where:**
- $L$ is the number of layers and $w$ is the window size.
- Each layer extends the reach by $w-1$ tokens, and the $+1$ counts the token itself.
- **Matters:** reach grows with depth even though each layer is local. It is only a *possible* path: how much information actually survives is a separate question.

**Example (tiny numbers):** $w=8$ and $L=4$.
- $4\times(8-1)+1=\mathbf{29}$ tokens, which the code also finds by passing influence through the masks.
- Meaning: a model with 32 layers and a window of 4,096 could, in this idealised count, be influenced by about $32\times4095+1=131{,}041$ tokens back (my arithmetic from the formula, not a claim that it works well that far).

**Further notes:**
- **Not covered:** global tokens, dilated windows, and how position encodings interact with windows ([[RoPE]], [[Positional Encoding]]).
- **Not tested:** model quality or speed. The code counts cells and checks the reach formula.

**What came before, and how this improved on it:**
- **Before:** full self-attention, where every token attends to all earlier tokens ([[Transformer]], 2017).
- **Its limit:** the cost is quadratic in the text length, which the Longformer abstract names as the reason Transformers cannot process long sequences.
- **What changed:** restrict each token to a local window (plus optional global tokens) so the cost is linear (Beltagy et al., Longformer, arXiv 2004.05150, 10 April 2020, abstract read); Mistral 7B uses it together with GQA (arXiv 2310.06825, 10 October 2023, abstract read).
- **How much better:** in this note's count, a 4,096-token text with a 512 window needs 1,966,336 cells instead of 8,390,656 (4.27 times fewer), and the cache for a 16,000-token text is 3.91 times smaller with a 4,096-token window. The Mistral abstract says its 7B model outperforms Llama 2 13B on all evaluated benchmarks, but credits the model as a whole, so that is not a measure of SWA alone.

## Related
- [[Self-Attention]] · [[Attention]] · [[Transformer]] · [[KV Cache]] · [[Grouped-Query Attention]] · [[Receptive Field]] · [[FlashAttention]] · [[RoPE]] · [[Positional Encoding]] · [[LLM]] · [[Convolution]] · [[Context Window]]

## Sources
- Stanford CME 295 (Fall 2025), Lecture 2 slides, Afshine and Shervine Amidi
- Beltagy et al., *Longformer: The Long-Document Transformer*, 2020 (arXiv 2004.05150, abstract read)
- Jiang et al., *Mistral 7B*, 2023 (arXiv 2310.06825, abstract read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving Sliding window attention:::Each token looks only at itself and the $w-1$ tokens before it (a window of size $w$) instead of every earlier token.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the number of attention cells with a window and with full causal attention, and explain each symbol.
?
$$\text{cells}=n\,w-\frac{w(w-1)}{2}\qquad\text{against full causal attention}\qquad\frac{n(n+1)}{2}$$
- $n$: the number of tokens. $w$: the window size ($w\le n$).
- Tokens near the start see fewer than $w$ tokens, and the $-\tfrac{w(w-1)}{2}$ term takes those missing cells off.
- $\tfrac{n(n+1)}{2}$: the count when every token sees every earlier token.
- The window grows like $n\,w$, the full version like $n^2$. When $w\ll n$ the saving is large, about $n/(2w)$ times. Without the correction term the start is over-counted.

#flashcards/ml/example #flashcards/topic/llm-serving $n=6$, $w=3$. How many cells for the window and for full causal attention? What about $n=4096$, $w=512$?::Window: $6\cdot3-3\cdot2/2=15$ (rows of 1, 2, 3, 3, 3, 3 ones). Full: $6\cdot7/2=21$. At $n=4096$, $w=512$: 1,966,336 against 8,390,656, which is 4.27 times fewer.

#flashcards/ml/formula #flashcards/topic/llm-serving Write how far back information can reach after $L$ layers, and explain each symbol.
?
$$\text{reach}=L\,(w-1)+1$$
- $L$: the number of layers. $w$: the window size.
- Each layer extends the reach by $w-1$ tokens. The $+1$ counts the token itself.
- Reach grows with depth even though each layer is local. It is only a possible path, and how much information actually survives is a separate question.

#flashcards/ml/example #flashcards/topic/llm-serving $w=8$, $L=4$. What is the reach?::$4\times(8-1)+1=29$ tokens. For 32 layers and $w=4096$ the idealised count is $32\times4095+1=131{,}041$ (my arithmetic, not a claim that it works).

#flashcards/ml/cloze #flashcards/topic/llm-serving Sliding window attention makes the cost grow ==in a straight line== with text length, instead of with its ==square==.

#flashcards/ml/concept #flashcards/topic/llm-serving Far-away words are not lost in sliding window attention. How does the information get there?::Token $i$ reads from tokens in its window, and those already read from theirs in the layer below. After $L$ layers information can come from about $L(w-1)$ tokens back, like the receptive field of a CNN.
