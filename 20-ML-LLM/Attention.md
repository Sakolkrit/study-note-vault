---
tags: [attention, transformer, LLM, deep-learning, NLP]
status: growing
level: 4
tier: 3
---

# Attention

## 🟢 1. Simple (age 12)
Attention is how a computer decides **which parts of the input to focus on** right now, like highlighting the important words with a marker.

**Example:** "The trophy didn't fit in the suitcase because **it** was too big." What is "it"? You focus on "trophy" (not "suitcase") because a trophy being too big explains why it didn't fit. Attention is the computer doing that focusing, using numbers as the highlighter: a bright highlight is a big number, a faint one is a small number.

## 🔵 2. How it works
For one word, say "it":
1. **Ask:** "it" makes a question: what am I looking for?
2. **Compare:** every other word holds a label saying what it is about. The computer checks how well each label matches the question and gives a score.
3. **Turn scores into shares:** scores become percentages that add up to 100% (for example trophy 70%, suitcase 20%, the rest 10%). These shares are the **attention weights**.
4. **Mix:** the new meaning of "it" is a blend of the other words' information, using those shares. So "it" now carries a lot of "trophy".

Do this for every word at the same time, and many times with different questions (**multi-head**: one head may track grammar, another who-did-what). Then stack layers.

![[attention-overview.svg]]
*Overview of the attention mechanism: a query is compared with keys, producing weights over the values. Image: Zhang, Lipton, Li and Smola, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Attention_mechanism_overview.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
Each token vector $x$ is projected into three vectors:
- **Query (Q):** what this token looks for.
- **Key (K):** what this token offers to be matched on.
- **Value (V):** the information it passes along if chosen.

Weights = softmax of (query · key) scores. Output = weighted sum of values. This is **self-attention** when Q, K, V all come from the same sequence, and **cross-attention** when Q comes from one sequence (decoder) and K, V from another (encoder, or an image).

**Causal mask:** in a text generator, a token may only attend to earlier tokens, otherwise it could cheat by reading the answer.

```python
import numpy as np
def softmax(s): e = np.exp(s - s.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

def self_attention(X, Wq, Wk, Wv):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    w = softmax(Q @ K.T / np.sqrt(K.shape[-1]))   # n x n weights, each row sums to 1
    return w @ V, w
```

Practical points:
- Cost grows with $n^2$ for $n$ tokens: the main limit on context length.
- **KV cache:** when generating, past K and V never change, so store them instead of recomputing.
- **FlashAttention** computes the exact same result with far less memory traffic.
- **Grouped-query attention** shares K and V across several query heads to shrink the KV cache.

History: attention was first added to RNN translation models (Bahdanau et al., 2014) to stop the whole sentence being squeezed into one vector; the [[Transformer]] (2017) then dropped the RNN entirely.

## 🔴 4. Deep / formal
Scaled dot-product attention, $Q,K\in\mathbb{R}^{n\times d_k}$, $V\in\mathbb{R}^{n\times d_v}$:

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V
$$

where $M_{ij}=-\infty$ for $j>i$ (causal mask) and $0$ otherwise.

**Where:**
- $Q$, $K$, $V$ are the query, key and value [[Matrix]] grids described in Level 3 (one row per token). $n$ is the number of tokens; $d_k$ and $d_v$ are the lengths of key/query and value vectors.
- $QK^\top$ is an $n\times n$ grid of match scores: cell $(i,j)$ says how well token $i$'s question matches token $j$'s label. **Matters:** it is the whole "who should I look at?" step.
- $\sqrt{d_k}$ is a scaling divisor. **Matters:** it keeps scores in a sane range (see "Why $\sqrt{d_k}$" just below), so softmax does not collapse to a single pick.
- $M$ is the **mask** added to the scores. $M_{ij}=-\infty$ for $j>i$ means "token $i$ may not look at a *later* token $j$". **Matters:** $e^{-\infty}=0$, so after softmax those positions get exactly 0% attention. This stops a text generator from cheating by reading the answer.
- $\mathrm{softmax}$ turns each row into percentages that sum to 1 (the attention weights).
- Multiplying by $V$ blends the values by those percentages, giving each token a new vector that carries information from the tokens it attended to.

**Example (tiny numbers):** sentence "the cat sat" (3 tokens). Look at token 2, "cat". Its raw match scores against [the, cat, sat] are $[2.0,\ 1.0,\ 3.0]$.
- Mask: "sat" comes later, so its score becomes $-\infty$: $[2.0,\ 1.0,\ -\infty]$.
- Softmax: $e^2=7.39$, $e^1=2.72$, $e^{-\infty}=0$. Sum $=10.11$. Weights $=[\mathbf{0.73},\ \mathbf{0.27},\ \mathbf{0}]$.
- Values: "the" $=[1,0]$, "cat" $=[0,1]$. Output $=0.73\cdot[1,0]+0.27\cdot[0,1]=\mathbf{[0.73,\ 0.27]}$.
- "cat" looked mostly at "the" and itself, and could not peek at "sat" even though "sat" had the highest raw score.

**Why $\sqrt{d_k}$:** for $q,k$ with independent zero-mean, unit-variance entries, $\mathrm{Var}(q\cdot k)=d_k$. Without scaling, large $d_k$ pushes the softmax toward one-hot, so gradients vanish.

**As a kernel smoother:** the output at position $i$ is $\sum_j \alpha_{ij}v_j$ with $\alpha_{ij}\propto\exp(q_i\cdot k_j/\sqrt{d_k})$, a Nadaraya–Watson estimator with an exponential kernel. Linear attention replaces $\exp(q\cdot k)$ with $\phi(q)\cdot\phi(k)$, giving $O(n)$ cost by reordering the product to $\phi(Q)\,(\phi(K)^\top V)$.

**Multi-head:**

$$
\mathrm{MHA}(X)=\mathrm{Concat}_{i=1}^{h}\big[\mathrm{Attention}(XW_i^Q,XW_i^K,XW_i^V)\big]W^O
$$

**Where:**
- $X$ is the grid of token vectors going into the layer.
- $h$ is the number of heads; $i$ counts them from 1 to $h$. **Matters:** each head can specialise in a different pattern (grammar, references, nearby words).
- $W_i^Q, W_i^K, W_i^V$ are learned matrices for head $i$ that turn $X$ into its own queries, keys and values. **Matters:** they are the only reason heads behave differently.
- $\mathrm{Attention}(\cdot)$ is the formula above, run once per head.
- $\mathrm{Concat}$ places the $h$ results side by side. $W^O$ is a final learned matrix that mixes them into one output of the usual width. **Matters:** it lets the heads' findings combine.

**Example (tiny numbers):** same idea as in [[Transformer]]. Head 1 outputs $[0.9,0.1]$ (focused on grammar) and head 2 outputs $[0.2,0.8]$ (focused on meaning). Concat gives $[0.9,0.1,0.2,0.8]$, and a $W^O$ that averages the heads gives $\mathbf{[0.55,\ 0.45]}$.

Position is not in attention itself, since attention is permutation-equivariant, so it must be added (sinusoidal, learned, or rotary RoPE).

Caveat: attention weights are not automatically an *explanation* of the model's decision (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019 argue the nuance).

**What came before, and how this improved on it:**
- **Before:** an encoder-decoder [[Recurrent Neural Network]] squeezed the whole source sentence into **one fixed-length vector** before the decoder started writing.
- **Its limit:** Bahdanau and colleagues conjectured that this fixed-length vector is a bottleneck (arXiv 1409.0473, 1 September 2014, abstract read).
- **What changed:** let the decoder look back at all the encoder's states and take a weighted mix for each output word (a soft search over the source), so nothing has to be crushed into one vector.
- **How much better:** the paper reports translation quality "comparable to the existing state-of-the-art phrase-based system" for English to French (abstract). In the code below (my own toy, not the paper's test), a 32-number summary picks the true word over a decoy 98% of the time with 4 words, 75% with 32 words and 58% with 128 words, while attention, which keeps every word's vector, can address any word exactly.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
d = 32
for n in (4, 8, 16, 32, 64, 128):
    ok = 0
    for _ in range(200):
        items = rng.standard_normal((n, d)) / np.sqrt(d)   # n source "words"
        fixed = items.sum(0)                                 # one fixed-length summary
        j = rng.integers(n)
        # can the summary tell item j from a random decoy that was NOT in the sentence?
        decoy = rng.standard_normal(d) / np.sqrt(d)
        ok += (fixed @ items[j]) > (fixed @ decoy)
    print(f"n={n:3d}  fixed vector picks the real word {ok/200:.2f}")
print("attention keeps all n vectors, so it can address word j exactly (1.00)")
```

## Related
- [[Transformer]] · [[Embedding]] · [[LLM]] · [[Tokenization]] · [[GPU]] · [[Recurrent Neural Network]] · [[Self-Attention]] · [[RoPE]] · [[FlashAttention]] · [[KV Cache]] · [[Grouped-Query Attention]] · [[Decoding Strategies]] · [[Speculative Decoding]] · [[Sliding Window Attention]] · [[Attention Is All You Need 2017]] · [[Seq2Seq 2014]] · [[Context Rot]]
- [[Interpretability]]
## Sources
- Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate*, 2014
- Vaswani et al., *Attention Is All You Need*, 2017
- Dao et al., *FlashAttention*, 2022
- Bahdanau et al., arXiv 1409.0473 (abstract read for the history block)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/transformers-nlp Attention:::How a computer decides which parts of the input to focus on right now, like highlighting the important words with a marker.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write attention (with a causal mask) and explain each symbol.
?
$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V$$
- $Q$ (queries), $K$ (keys), $V$ (values): matrices with one row per token. Q is what a token looks for, K is what each token offers, V is the information handed over if chosen.
- $QK^\top$: a score for every query-key pair. Dividing by $\sqrt{d_k}$ ($d_k$ is the query/key size) stops big scores from making the softmax too sharp.
- $M$: the causal mask, $-\infty$ for later tokens ($j>i$) and 0 otherwise, so a token cannot look ahead.
- softmax turns each row of scores into weights that add up to 1. Multiplying by $V$ blends the values by those weights.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write multi-head attention and explain each symbol.
?
$$\mathrm{MHA}(X)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W^O,\quad \mathrm{head}_i=\mathrm{Attention}(XW_i^Q,XW_i^K,XW_i^V)$$
- $X$: the grid of token vectors going into the layer.
- $h$: the number of heads, counted by $i$. Each head can specialise in a different pattern (grammar, references, nearby words).
- $W_i^Q, W_i^K, W_i^V$: learned matrices for head $i$ that turn $X$ into its own queries, keys and values.
- $W^O$: the matrix that mixes the glued-together heads back to the model width.

#flashcards/ml/cloze #flashcards/topic/transformers-nlp In attention, a ==query== is compared with ==keys==, producing weights over the ==values==.

#flashcards/ml/example #flashcards/topic/transformers-nlp Two heads give [0.9, 0.1] and [0.2, 0.8]. If $W^O$ averages them, what is the result?::Concat gives [0.9, 0.1, 0.2, 0.8]. Averaging the two heads gives [0.55, 0.45].

#flashcards/ml/image #flashcards/topic/transformers-nlp ![[attention-overview.svg]]
What does this picture show?
?
Overview of the attention mechanism: a query is compared with keys, producing weights over the values.
