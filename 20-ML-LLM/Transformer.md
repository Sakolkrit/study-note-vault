---
tags: [transformer, attention, LLM, deep-learning, NLP]
status: growing
level: 4
tier: 4
---

# Transformer

## 🟢 1. Simple (age 12)
A transformer is a way for a computer to read a sentence where **every word looks at every other word and decides which ones matter most**.

**Example:** "The dog chased the ball because **it** was fast." What is "it"? You look back at the sentence and pick the word that makes sense. A transformer does the same: the word "it" pays extra *attention* to the words that help explain it.

## 🔵 2. How it works
1. Cut the text into small pieces called **tokens** (often a word or part of a word).
2. Turn each token into a list of numbers: an [[Embedding]].
3. Add a **position** signal, so the computer knows word order ("dog bites man" is not "man bites dog").
4. **Attention:** each token asks "which other tokens help me?" and mixes in information from them.
5. Pass the result through a small neural network to think about it.
6. Repeat steps 4 and 5 many times (stacked layers). Each layer understands a bit more.
7. At the end, predict something: for a chatbot, the next token.

Why it won: older models read one word at a time, slowly. Transformers read all words **at once**, so they train fast on huge amounts of text on [[GPU]]s. That made [[LLM]]s possible.

![[transformer-architecture.png]]
*The original encoder–decoder transformer. Image: dvgodoy, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Transformer,_full_architecture.png), CC BY 4.0.*

## 🟠 3. Technical detail
Introduced in *Attention Is All You Need* (Vaswani et al., 2017).

One layer (block) has two parts, each wrapped in a residual connection and layer norm:
- **Multi-head self-attention:** several attention "heads" run in parallel, each learning a different relation (syntax, coreference, ...).
- **Feed-forward network (MLP):** applied to each token separately.

Self-attention uses three projections of every token: **Q**uery ("what am I looking for?"), **K**ey ("what do I contain?"), **V**alue ("what do I pass on?"). Attention weight = how well a query matches a key; output = weighted sum of values.

```python
import numpy as np
def attention(Q, K, V, causal=True):
    s = Q @ K.T / np.sqrt(K.shape[-1])            # similarity of every pair
    if causal:                                     # decoder: can't see the future
        s = np.where(np.tril(np.ones_like(s)) == 1, s, -1e9)
    w = np.exp(s - s.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)                  # softmax
    return w @ V
```

Variants:
- **Encoder-only** (BERT): sees whole sentence, good for understanding and embeddings.
- **Decoder-only** (GPT, Llama, Claude): predicts next token left to right; the base of most [[LLM]]s.
- **Encoder–decoder** (original, T5): translation, summarisation.

Cost: attention is $O(n^2)$ in sequence length $n$, the reason long context is expensive. Position is injected by sinusoidal or learned encodings, or rotary embeddings (RoPE) in modern models.

## 🔴 4. Deep / formal
Scaled dot-product attention, for $Q,K\in\mathbb{R}^{n\times d_k}$, $V\in\mathbb{R}^{n\times d_v}$:

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

**Where:**
- $Q$ (queries), $K$ (keys), $V$ (values) are three [[Matrix]] grids, one row per token. $Q$ = what each token is looking for, $K$ = what each token offers, $V$ = the information it will hand over if chosen. They come from multiplying the token vectors by three learned matrices.
- $n$ is the number of tokens and $d_k$ is the size of each query/key vector; $d_v$ is the size of each value vector.
- $K^\top$ is $K$ transposed (rows and columns swapped). **Matters:** it makes the shapes line up so $QK^\top$ is an $n\times n$ grid whose cell $(i,j)$ is "how well does token $i$'s question match token $j$'s label?".
- $\sqrt{d_k}$ is a scaling divisor. **Matters:** without it, big vectors produce huge scores, softmax collapses to a one-hot pick, and learning stalls (full reason in the paragraph below).
- $\mathrm{softmax}$ turns each row of scores into percentages that add to 1: the **attention weights**. **Matters:** it lets every token spread its focus over others (for example 70% / 20% / 10%).
- Multiplying by $V$ blends the values using those weights. **Matters:** the result for each token is a new vector enriched with information from the tokens it paid attention to. See [[Attention]].

**Example (tiny numbers):** 2 tokens, $d_k=2$. Let $Q=K=\begin{bmatrix}1&0\\0&1\end{bmatrix}$ and $V=\begin{bmatrix}10&0\\0&10\end{bmatrix}$ (token 1 carries "10 units of A", token 2 carries "10 units of B").
- $QK^\top$ = the identity grid $\begin{bmatrix}1&0\\0&1\end{bmatrix}$: each token matches itself best.
- Divide by $\sqrt2\approx1.41$: $\begin{bmatrix}0.71&0\\0&0.71\end{bmatrix}$.
- Softmax of row 1 $[0.71,\ 0]$: $e^{0.71}=2.03$ and $e^0=1$, so weights $[\mathbf{0.67},\ \mathbf{0.33}]$.
- Output for token 1 $=0.67\cdot[10,0]+0.33\cdot[0,10]=\mathbf{[6.7,\ 3.3]}$.
- Token 1 is now mostly "A" (its own info) with a bit of "B" mixed in from token 2.

The $\sqrt{d_k}$ keeps dot products from growing with dimension: if entries of $q,k$ have mean 0 and variance 1, then $q\cdot k$ has variance $d_k$, which would push softmax into saturated, near one-hot regions with tiny gradients.

Multi-head with $h$ heads:

$$
\mathrm{MHA}(X)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W^O,\quad
\mathrm{head}_i=\mathrm{Attention}(XW_i^Q,XW_i^K,XW_i^V)
$$

**Where:**
- $X$ is the grid of token vectors entering the layer (one row per token).
- $h$ is the number of **heads** and $i$ numbers them. **Matters:** each head can focus on a different kind of relation (grammar, who-did-what, nearby words), so the model looks at the sentence in several ways at once.
- $W_i^Q, W_i^K, W_i^V$ are learned matrices for head $i$ that turn $X$ into that head's own $Q$, $K$, $V$. **Matters:** they are what makes heads different from each other.
- $\mathrm{head}_i$ is the attention result of head $i$ (the formula above).
- $\mathrm{Concat}$ glues the $h$ head outputs side by side into one wide vector per token.
- $W^O$ is a final learned matrix that mixes the glued outputs back to the model's normal width. **Matters:** it lets the heads' findings combine rather than just sit next to each other.

**Example (tiny numbers):** $h=2$ heads, each gives 2 numbers for one token. Head 1 (found the subject): $[0.9,\ 0.1]$. Head 2 (found the object): $[0.2,\ 0.8]$.
- $\mathrm{Concat}$ glues them: $[0.9,\ 0.1,\ 0.2,\ 0.8]$ (4 numbers).
- Let $W^O$ average the two heads: $\begin{bmatrix}0.5&0\\0&0.5\\0.5&0\\0&0.5\end{bmatrix}$.
- Result: $[0.9,0.1,0.2,0.8]\,W^O=\mathbf{[0.55,\ 0.45]}$, back to 2 numbers.
- Each head looked at the sentence its own way, then $W^O$ blended both views into one.

Block (pre-norm form used by most modern models):

$$
X' = X+\mathrm{MHA}(\mathrm{LN}(X)),\qquad
X''=X'+\mathrm{FFN}(\mathrm{LN}(X'))
$$

**Where:**
- $X$ is the block's input, $X'$ is the state after attention, and $X''$ is the block's output, which becomes the next block's input.
- $\mathrm{LN}$ is **layer normalisation**: rescales each token's numbers to a steady range. **Matters:** keeps values from drifting huge or tiny through dozens of layers, which makes deep stacks trainable.
- $\mathrm{MHA}$ is multi-head attention (above): tokens exchange information.
- $\mathrm{FFN}$ is the feed-forward network (a small two-[[Matrix]] neural net applied to each token on its own): "think about what you just gathered".
- The leading $X+\dots$ and $X'+\dots$ are **residual connections**: the layer *adds* its result to what came in instead of replacing it. **Matters:** the original information always has a direct path through, so gradients flow back easily and deep models train.

**Example (tiny numbers):** one token enters as $X=[1,\ 2]$.
- $\mathrm{LN}(X)$: mean $=1.5$, spread $=0.5$, so $\mathrm{LN}(X)=[-1,\ 1]$ (same shape, steady scale).
- Suppose attention returns $[0.5,\ -0.5]$. Then $X'=[1,2]+[0.5,-0.5]=\mathbf{[1.5,\ 1.5]}$. The original $[1,2]$ is still inside.
- Suppose the FFN returns $[0.2,\ 0.1]$. Then $X''=[1.5,1.5]+[0.2,0.1]=\mathbf{[1.7,\ 1.6]}$.
- Even if a layer returned all zeros, $X$ would pass through unchanged. That safe path is what lets 100 layers stack without losing the signal.

Parameters per block are about $12\,d_{\text{model}}^2$ (4 for attention, 8 for the MLP with 4x hidden width).

Active topics: FlashAttention (IO-aware exact attention), KV cache for fast generation, grouped-query attention, sparse and linear attention, mixture-of-experts MLPs, state-space models as alternatives.

**What came before, and how this improved on it:**
- **Before:** the best translation models were recurrent or convolutional encoder-decoders, linked by attention (abstract, read).
- **Its limit:** recurrence runs one step at a time, so training cannot use the GPU's parallelism along the sentence ([[GPU]]).
- **What changed:** drop recurrence and convolution entirely and build the model only from attention plus feed-forward layers (Vaswani et al., arXiv 1706.03762, 12 June 2017).
- **How much better:** the abstract reports 28.4 BLEU on WMT 2014 English-German (over 2 BLEU above the best earlier results, including ensembles) and 41.8 BLEU English-French after 3.5 days on eight GPUs, "a small fraction" of earlier training costs. The code below only counts: a sentence of 100 words takes 100 sequential steps in an RNN against 1 per Transformer layer, and a block has about $12d^2$ parameters (3,145,728 at $d=512$). I did not train or time anything.

**Check it yourself:**

```python
for d in (512, 768):
    print("d_model", d, ": parameters per block about 12*d^2 =", 12 * d * d)
n = 100
print("sentence of", n, "words: RNN needs", n, "sequential steps, a Transformer layer needs 1")
print("paper: 28.4 BLEU EN-DE, 41.8 BLEU EN-FR after 3.5 days on 8 GPUs (abstract)")
```

## Related
- [[Embedding]] · [[LLM]] · [[Attention]] · [[Tokenization]] · [[GPU]] · [[RAG]] · [[Residual Connection]] · [[Recurrent Neural Network]] · [[Pre-training]] · [[Scaling Laws]] · [[Mixture of Experts]] · [[Byte-Pair Encoding]] · [[Self-Attention]] · [[RoPE]] · [[FlashAttention]] · [[ResNet]] · [[KV Cache]] · [[Grouped-Query Attention]] · [[Decoding Strategies]] · [[Speculative Decoding]] · [[Layer Normalization]] · [[Vision Transformer]] · [[Sliding Window Attention]] · [[Attention Is All You Need 2017]] · [[History of Text Analytics]] · [[LSTM]] · [[Feed-Forward Network]] · [[Hugging Face Transformers]]

## Sources
- Vaswani et al., *Attention Is All You Need*, 2017
- Devlin et al., *BERT*, 2018
- Radford et al., *Improving Language Understanding by Generative Pre-Training* (GPT), 2018
- Vaswani et al., arXiv 1706.03762 (abstract read for the history block)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/transformers-nlp Transformer:::A way for a computer to read a sentence where every word looks at every other word and decides which ones matter most.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write scaled dot-product attention and explain each symbol.
?
$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
- $Q$ (queries), $K$ (keys), $V$ (values): matrices with one row per token. Q is what each token looks for, K is what it offers, V is the information it hands over if chosen. They come from multiplying the token vectors by three learned matrices.
- $n$: the number of tokens. $d_k$: the size of each query/key vector.
- $QK^\top$: a score for every pair of tokens. Dividing by $\sqrt{d_k}$ keeps the scores from getting too big.
- softmax turns each row into weights that add up to 1, and $V$ is blended by them.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write multi-head attention and explain each symbol.
?
$$\mathrm{MHA}(X)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W^O,\quad \mathrm{head}_i=\mathrm{Attention}(XW_i^Q,XW_i^K,XW_i^V)$$
- $X$: the grid of token vectors entering the layer (one row per token).
- $h$: the number of heads. Each head can focus on a different relation, so the model looks at the sentence in several ways at once.
- $W_i^Q, W_i^K, W_i^V$: learned matrices for head $i$. $W^O$: mixes the glued heads back to the original width.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write one transformer block (pre-norm) and explain each symbol.
?
$$X' = X+\mathrm{MHA}(\mathrm{LN}(X)),\qquad X''=X'+\mathrm{FFN}(\mathrm{LN}(X'))$$
- $X$: the block's input. $X'$: the state after attention. $X''$: the block's output, the next block's input.
- $\mathrm{LN}$: layer normalisation, rescales each token's numbers to a steady range. It keeps values from drifting huge or tiny through dozens of layers.
- $\mathrm{MHA}$: multi-head attention (tokens talk to each other). $\mathrm{FFN}$: the feed-forward network applied to each token.
- The "$X+$" parts are residual connections: the original stays inside, so deep stacks train.

#flashcards/ml/cloze #flashcards/topic/transformers-nlp A transformer block adds each sub-layer's output to its input, which is called a ==residual connection==. ==Layer normalisation== keeps the values steady.

#flashcards/ml/example #flashcards/topic/transformers-nlp One token $X=[1,2]$ enters a block. LN gives $[-1,1]$, attention returns $[0.5,-0.5]$ and the FFN returns $[0.2,0.1]$. What are $X'$ and $X''$?::X′ = [1,2] + [0.5,−0.5] = [1.5, 1.5]. X″ = [1.5,1.5] + [0.2,0.1] = [1.7, 1.6]. The original is still inside.

#flashcards/ml/image #flashcards/topic/transformers-nlp ![[transformer-architecture.png]]
What does this picture show?
?
The original encoder-decoder transformer architecture.
