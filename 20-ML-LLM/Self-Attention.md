---
tags: [attention, self-attention, transformer, LLM, deep-learning, NLP]
status: growing
level: 4
tier: 3
---

# Self-Attention

Self-attention lets **every item in a sequence look at every other item in the same sequence** and pull in the information it needs, all at once. It is the main building block of the [[Transformer]]. This note goes through it slowly, step by step. The overview of the general idea is in [[Attention]].

## 🟢 1. Simple (age 12)
Imagine a classroom where each student must write one sentence about themselves, but first they are allowed to look around the room at everyone else's notes. Each student asks a question ("who here is talking about the same thing as me?"), peeks at the notes that fit best, and then writes a better sentence that mixes in what they learned. Everyone does this **at the same time**.

**Example:** Take the sentence "The dog chased the cat because it was fast." The word "it" has no meaning by itself. It looks around at "dog" and "cat", decides which one fits best, and copies some of that meaning into itself. After that, "it" is no longer an empty word; it carries "dog" or "cat" with it.

## 🔵 2. How it works
For a sentence of $n$ words, each already turned into a list of numbers (see [[Embedding]]):
1. **Make three versions of every word.** Each word's numbers are multiplied by three learned grids of numbers (called [[Matrix]] weights). That gives a **query** (what am I looking for?), a **key** (what do I contain, so others can match me?) and a **value** (what I hand over if you pick me).
2. **Score every pair.** For each word, compare its query with the key of every word, including itself. A big match number means "this word is useful to me".
3. **Turn the scores into shares.** For each word, a [[Softmax]] turns its row of scores into percentages that add up to 100%.
4. **Mix the values.** Each word's new numbers are the values of all words, blended using its percentages.
5. **Each word's result is new.** It now holds a mixture of information from the whole sentence. The sentence keeps the same length: $n$ words in, $n$ new lists of numbers out.
6. **Stack many layers.** The next layer does the same thing again on the new lists, and over many layers, richer meanings build up.

![[self-attention-vs-cnn-rnn.svg]]
*Three ways to let tokens affect each other, with five inputs x₁ to x₅. In a CNN (left) each output sees only a small neighbourhood, so distant tokens need several layers to meet. In an RNN (top right) information passes along the chain one step at a time. In self-attention (bottom right) every output connects to every input directly, including the far ends (the solid line from x₅ to the first output). Image: Zhang, Lipton, Li and Smola (*Dive into Deep Learning*), [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Self-attention_in_CNN,_RNN,_and_self-attention.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Same sequence for all three:** in *self*-attention, the queries, keys and values all come from the same input $X$. (In cross-attention, the queries come from one sequence and the keys and values from another.)
- **Three learned weight grids** $W_Q$, $W_K$, $W_V$. These are the only learned parts inside the attention step itself. Everything else is fixed arithmetic.
- **Direct paths:** any token can affect any other in a single layer. In an [[Recurrent Neural Network]] the signal from far back has to survive many steps, which is where [[Vanishing and Exploding Gradients]] cause trouble. In a [[Convolutional Neural Network]], reaching far needs many layers.
- **All at once:** the whole $n\times n$ grid of scores is one matrix multiplication, so it suits a [[GPU]] well. An RNN has to go through the sequence one step at a time, so it cannot do that.
- **Quadratic cost:** the score grid has $n^2$ cells, so doubling the text length makes the grid four times larger. This is the main limit on how long the text can be (equation 3 below puts numbers on it).
- **The causal mask:** for a model that writes text one token at a time, each token is only allowed to look at itself and earlier tokens. The scores for later tokens are set to minus infinity before the softmax, so they get 0% (shown in the code).
- **No built-in order:** self-attention treats its input as a bag of items. If you shuffle the input, the output is shuffled the same way (shown in the code). So word order has to be added separately (see [[Positional Encoding]]).
- **Several heads:** the layer runs a handful of these in parallel with different weight grids, then joins the results. That is multi-head attention, covered in [[Attention]] and [[Transformer]].
- **Around the layer:** a real Transformer block wraps self-attention with a [[Residual Connection]], [[Normalization]] and a small feed-forward network, none of which is shown here.
- **Remembering for speed:** when generating, the keys and values of earlier tokens never change, so they are stored (the KV cache) instead of recomputed. The cache grows with the text length (equation 3 below).

![[self-attention-encoder-detailed.png]]
*One self-attention layer for two tokens, drawn out in full. Each token x is turned into a query (Q), a key (K) and a value (V). Each query is compared with the keys to get alignments (a), the alignments weight the values, and the sum is the "context" (c) that goes on to the feed-forward layer (FF). The diagram uses its own labels (alignments, context), but the steps are the ones above. Image: dvgodoy, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Encoder_self-attention,_detailed_diagram.png), CC BY 4.0.*

```python
import numpy as np
def softmax(s): e = np.exp(s - s.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
def self_attention(X, Wq, Wk, Wv, causal=False):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    s = Q @ K.T / np.sqrt(K.shape[-1])
    if causal: s = s + np.triu(np.full(s.shape, -np.inf), 1)   # -inf above the diagonal: no peeking ahead
    w = softmax(s)
    return w @ V, w, (Q, K, V, s)
# 1) tiny worked case: 3 tokens, 2 numbers each
X  = np.array([[1., 0.], [0., 1.], [1., 1.]])
Wq = np.array([[1., 0.], [0., 2.]]); Wk = np.array([[2., 0.], [0., 1.]]); Wv = np.array([[1., 0.], [1., 1.]])
Y, w, (Q, K, V, s) = self_attention(X, Wq, Wk, Wv)
print('Q', Q.tolist(), 'K', K.tolist(), 'V', V.tolist())
print('scores', s.round(3).tolist()); print('weights', w.round(3).tolist()); print('out', Y.round(3).tolist())
Yc, wc, _ = self_attention(X, Wq, Wk, Wv, causal=True); print('causal weights', wc.round(3).tolist()); print('causal out', Yc.round(3).tolist())
# 2) shuffle the tokens: the outputs shuffle the same way (self-attention has no sense of order by itself)
rng = np.random.default_rng(0); n, d = 6, 4
X = rng.normal(size=(n, d)); Wq, Wk, Wv = (rng.normal(size=(d, d)) for _ in range(3))
p = rng.permutation(n)
Y1 = self_attention(X, Wq, Wk, Wv)[0]; Y2 = self_attention(X[p], Wq, Wk, Wv)[0]
print('permuted input -> permuted output:', np.allclose(Y1[p], Y2), '| and not the same order:', not np.allclose(Y1, Y2))
# 3) causal mask: changing a LATER token never changes an earlier output
X2 = X.copy(); X2[-1] += 5.0
a = self_attention(X, Wq, Wk, Wv, causal=True)[0]; b = self_attention(X2, Wq, Wk, Wv, causal=True)[0]
print('earlier outputs unchanged:', np.allclose(a[:-1], b[:-1]), '| last output changed:', not np.allclose(a[-1], b[-1]))
# 4) cost: the score grid is n x n; KV cache is 2 * layers * n * width * bytes
for n in (1000, 2000, 4000): print(n, 'tokens -> score cells', n * n)
L, dm, n, B = 32, 4096, 4096, 2
print('KV cache GB:', round(2 * L * n * dm * B / 1e9, 2))
# Q [[1.0, 0.0], [0.0, 2.0], [1.0, 2.0]] K [[2.0, 0.0], [0.0, 1.0], [2.0, 1.0]] V [[1.0, 0.0], [1.0, 1.0], [2.0, 1.0]]
# scores [[1.414, 0.0, 1.414], [0.0, 1.414, 1.414], [1.414, 1.414, 2.828]]
# weights [[0.446, 0.108, 0.446], [0.108, 0.446, 0.446], [0.164, 0.164, 0.673]]
# out [[1.446, 0.554], [1.446, 0.892], [1.673, 0.836]]
# causal weights [[1.0, 0.0, 0.0], [0.196, 0.804, 0.0], [0.164, 0.164, 0.673]]
# causal out [[1.0, 0.0], [1.0, 0.804], [1.673, 0.836]]
# permuted input -> permuted output: True | and not the same order: True
# earlier outputs unchanged: True | last output changed: True
# 1000 tokens -> score cells 1000000
# 2000 tokens -> score cells 4000000
# 4000 tokens -> score cells 16000000
# KV cache GB: 2.15
```
The code ran with NumPy only.

## 🔴 4. Deep / formal

**Self-attention with its learned weights.** For a grid $X$ of $n$ token vectors (one per row), the whole layer is

$$Y=\mathrm{softmax}\!\left(\frac{(XW_Q)(XW_K)^{\top}}{\sqrt{d_k}}\right)(XW_V)$$

**Where:**
- $X$ is the $n\times d$ grid of input vectors. **Matters:** it is the only input, so each output is built from the sentence itself.
- $W_Q,W_K,W_V$ are learned grids that turn $X$ into queries, keys and values. **Matters:** they decide what counts as "relevant". With $W_Q=W_K$ the match score would be symmetric, so token A would need to look at B exactly as much as B looks at A. Separate grids remove that limit.
- $d_k$ is the length of a query or key. **Matters:** dividing by $\sqrt{d_k}$ stops large vectors from making the softmax collapse onto one choice (the reason is given in [[Attention]]).
- $\mathrm{softmax}$ is applied to each row. **Matters:** each token's weights add up to 1, so its output is a true blend of values and cannot blow up in size.
- $Y$ is the $n\times d_v$ output, one new vector per token.

**Example (tiny numbers):** three tokens $x_1=(1,0)$, $x_2=(0,1)$, $x_3=(1,1)$, with $d_k=2$ and the weights in the code. Follow token 1.
1. Queries, keys and values: $q_1=(1,0)$. The keys are $k_1=(2,0)$, $k_2=(0,1)$, $k_3=(2,1)$. The values are $v_1=(1,0)$, $v_2=(1,1)$, $v_3=(2,1)$.
2. Scores for token 1: $q_1\cdot k_j=2,\,0,\,2$. Divide by $\sqrt2=1.414$: $1.414,\,0,\,1.414$.
3. Softmax: $e^{1.414}=4.112$, $e^0=1$, $4.112$. Sum $9.224$. Weights $0.446,\ 0.108,\ 0.446$.
4. Output: $0.446(1,0)+0.108(1,1)+0.446(2,1)=(0.446+0.108+0.892,\ 0.108+0.446)=(1.446,\ 0.554)$.
5. Now add the causal mask. Token 1 can see only itself, so its weights are $(1,0,0)$ and its output is just $v_1=(1,0)$. Token 2's weights become $(0.196,\,0.804,\,0)$, and its output $(1.0,\,0.804)$.
6. Meaning: without the mask, token 1 gives 44.6% each to tokens 1 and 3 and only 10.8% to token 2. With the mask, it can only use itself. The mask changes the answer, not just a number inside it.

**Order is not built in.** Let $P$ be a permutation matrix, a reshuffling of the rows. Then

$$\mathrm{SelfAttn}(PX)=P\,\mathrm{SelfAttn}(X)$$

**Where:**
- $P$ is a permutation matrix (a grid of 0s and 1s with exactly one 1 in each row and column). **Matters:** multiplying by it reorders the rows, so $PX$ is the same sentence with the words shuffled.
- $\mathrm{SelfAttn}(\cdot)$ is the layer above (with no mask). **Matters:** this equation is the reason the layer cannot tell word order by itself.
- Both sides give the same rows, in the same shuffled order. **Matters:** "dog bites man" and "man bites dog" would get the same set of output vectors (just listed in a different order) unless position is added separately, for example with [[Positional Encoding]]. This holds without the mask. A causal mask does depend on order.

**Example (tiny numbers):** Use the 3-token example above and swap tokens 1 and 2, so the input is $(x_2,x_1,x_3)$.
1. Original outputs: $(1.446,0.554)$, $(1.446,0.892)$, $(1.673,0.836)$.
2. Run the code on the swapped input: outputs $(1.446,0.892)$, $(1.446,0.554)$, $(1.673,0.836)$.
3. These are the original rows 1 and 2 swapped, and row 3 unchanged. A random 6-token test in the code also gives the same result.
4. Meaning: the layer sees the same pairings no matter where the words sit. All the information about order has to come in through the input numbers.

**What it costs.** For $n$ tokens, a model with $L$ layers and width $d_{\text{model}}$, storing numbers of $b$ bytes:

$$\text{score cells}=n^{2},\qquad \text{KV cache bytes}=2\,L\,n\,d_{\text{model}}\,b$$

**Where:**
- $n$ is the number of tokens. **Matters:** the score grid grows with its square, but the cache grows only in step with $n$.
- $n^2$ is the number of query-key pairs per head and layer. **Matters:** this is the quadratic cost. Doubling the text makes it 4 times larger.
- $L$ is the number of layers and $d_{\text{model}}$ the width of each token vector. **Matters:** both multiply the cache size.
- The factor 2 counts both keys and values. $b$ is bytes per number (2 for 16-bit). **Matters:** using fewer bits per number ([[Quantization]]) shrinks the cache.
- The formula assumes every head keeps its own keys and values. Methods that share them between heads (see [[Attention]]) make the cache smaller. I am assuming that a 7B-parameter-like model has $L=32$ and $d_{\text{model}}=4096$ (these are typical sizes, from memory, not the figures of a specific model).

**Example (tiny numbers):**
1. Score cells: $1000^2=1{,}000{,}000$; $2000^2=4{,}000{,}000$; $4000^2=16{,}000{,}000$. Each doubling multiplies the grid by 4.
2. KV cache with $L=32$, $d_{\text{model}}=4096$, $n=4096$, $b=2$: $2\times32\times4096\times4096\times2=2{,}147{,}483{,}648$ bytes, about $2.15$ GB.
3. Meaning: a single 4,096-token conversation needs about 2 GB of [[GPU]] memory just to remember its keys and values, on top of the model weights. That is why long contexts are expensive, and why many users at once multiply the cost.

**Other points (from memory, not checked here):**
- Tricks that make attention cheaper or faster include exact methods that move less data (FlashAttention), approximations that skip or compress pairs, and sharing keys and values between heads.
- Some research asks whether attention weights explain a model's choices. They are a useful clue but not a proof (see the caveat in [[Attention]]).
- Each attention layer is a weighted average, so by itself it is a smooth mixing step. The nonlinear work in a block is done mostly by the feed-forward network around it (the usual explanation; I have not tested it).

**What came before, and how this improved on it:**
- **Before:** in a [[Recurrent Neural Network]], the first word reaches the last word only through every step in between, one step at a time.
- **Its limit:** a path of $n-1$ steps (fading gradients, see [[Vanishing and Exploding Gradients]]), and the steps cannot run in parallel.
- **What changed:** every word looks directly at every other word in one layer (Vaswani et al., arXiv 1706.03762, 12 June 2017, abstract read).
- **How much better:** the code below counts it for a 512-word text: the path between the first and last word is 511 steps for an RNN against 1, and the sequential steps drop from 512 to 1. The price is more arithmetic: $n^2d=16{,}777{,}216$ against $nd^2=2{,}097{,}152$ operations (with $d=64$), 8 times more at $n=512$ and 128 times more at $n=8192$. That cost is why [[FlashAttention]] and [[Sliding Window Attention]] exist.

**Check it yourself:**

```python
n, d = 512, 64
print("steps before the last word can see the first word:")
print("  RNN:", n - 1, "  self-attention: 1")
print("sequential steps to read the sentence: RNN", n, " self-attention 1 (per layer)")
print("operations per layer: RNN n*d^2 =", n * d * d, " self-attention n^2*d =", n * n * d)
n2 = 8192
print("at n=8192: RNN", n2 * d * d, " self-attention", n2 * n2 * d)
```

## Related
[[Attention]] · [[Transformer]] · [[Positional Encoding]] · [[Softmax]] · [[Matrix]] · [[Embedding]] · [[Recurrent Neural Network]] · [[Convolutional Neural Network]] · [[Residual Connection]] · [[Normalization]] · [[GPU]] · [[Quantization]] · [[Mixture of Experts]] · [[Vanishing and Exploding Gradients]] · [[LLM]] · [[RoPE]] · [[FlashAttention]] · [[Big-O]] · [[Receptive Field]] · [[KV Cache]] · [[Grouped-Query Attention]] · [[Decoding Strategies]] · [[Speculative Decoding]] · [[Sliding Window Attention]]

## Sources
- Vaswani and colleagues (2017), "Attention Is All You Need", which introduced the Transformer and scaled dot-product self-attention (cited from memory; not opened for this note).
- Figures: [Self-attention_in_CNN,_RNN,_and_self-attention.svg](https://commons.wikimedia.org/wiki/File:Self-attention_in_CNN,_RNN,_and_self-attention.svg) (Zhang, Lipton, Li and Smola, CC BY-SA 4.0) and [Encoder_self-attention,_detailed_diagram.png](https://commons.wikimedia.org/wiki/File:Encoder_self-attention,_detailed_diagram.png) (dvgodoy, CC BY 4.0), both on Wikimedia Commons.
- The code and every Example number were run for this note (NumPy only).
- Vaswani et al., arXiv 1706.03762 (abstract read for the history block)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/transformers-nlp Self-attention:::Every token looks at all the other tokens in the same sentence, decides which matter, and mixes in what it learned. Each token asks a question, peeks at the best-matching notes, and writes a better version of itself.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write self-attention and explain each symbol.
?
$$Y=\mathrm{softmax}\!\left(\frac{(XW_Q)(XW_K)^{\top}}{\sqrt{d_k}}\right)(XW_V)$$
- $X$: the $n\times d$ grid of input vectors. It is the only input, so each output is built from the sentence itself.
- $W_Q,W_K,W_V$: learned grids that turn $X$ into queries, keys and values. They decide what counts as "relevant". With $W_Q=W_K$ the match score would be symmetric.
- $\sqrt{d_k}$: the scaling by the query/key size, which keeps the scores from growing too large.
- softmax turns each row of scores into weights. Multiplying by $XW_V$ gives the weighted mix of values.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write the permutation property of self-attention and explain each symbol.
?
$$\mathrm{SelfAttn}(PX)=P\,\mathrm{SelfAttn}(X)$$
- $P$: a permutation matrix (0s and 1s with exactly one 1 in each row and column). Multiplying by it reorders the rows, so $PX$ is the same sentence with the words shuffled.
- $\mathrm{SelfAttn}(\cdot)$: the layer above, with no mask.
- The output is just shuffled in the same way. So the layer cannot tell word order by itself, and positional information must be added.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write the cost of self-attention and explain each symbol.
?
$$\text{score cells}=n^{2},\qquad \text{KV cache bytes}=2\,L\,n\,d_{\text{model}}\,b$$
- $n$: the number of tokens. The score grid grows with its square (the quadratic cost), the cache only grows in step with $n$.
- $n^2$: the number of query-key pairs per head and layer. Doubling the text makes it 4 times larger.
- $L$: the number of layers. $d_{\text{model}}$: the width of each token vector. $b$: bytes per number.
- The 2 is for storing both keys and values.

#flashcards/ml/cloze #flashcards/topic/transformers-nlp Self-attention alone cannot tell word ==order==, so ==positional== information must be added. Its score grid grows with ==n²==.

#flashcards/ml/example #flashcards/topic/transformers-nlp How many score cells are there for 1,000, 2,000 and 4,000 tokens?::1,000² = 1,000,000. 2,000² = 4,000,000. 4,000² = 16,000,000. Each doubling multiplies the grid by 4.

#flashcards/ml/example #flashcards/topic/transformers-nlp What is the KV cache size for $L=32$, $d_{\text{model}}=4096$, $n=4096$ and $b=2$?::2 × 32 × 4096 × 4096 × 2 = 2,147,483,648 bytes, about 2.15 GB for a single conversation.

#flashcards/ml/image #flashcards/topic/transformers-nlp ![[self-attention-vs-cnn-rnn.svg]]
What does this picture compare?
?
Three ways to let tokens affect each other, with five inputs. In a CNN each output sees only a small neighbourhood, so distant tokens need several layers to meet. In an RNN information passes along the chain. In self-attention every token can look at every other directly.

#flashcards/ml/image #flashcards/topic/transformers-nlp ![[self-attention-encoder-detailed.png]]
What does this picture show?
?
One self-attention layer for two tokens drawn out in full. Each token x is turned into a query (Q), a key (K) and a value (V).
