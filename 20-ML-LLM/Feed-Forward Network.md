---
tags: [transformer, LLM, neural-network, feed-forward, MLP, architecture]
status: growing
level: 4
tier: 2
---

# Feed-Forward Network

A small [[Neural Network]] with two layers that a [[Transformer]] runs on **each token on its own**, right after [[Attention]]. Attention lets tokens look at each other. The feed-forward network lets each token "think" about what it just gathered.

## 🟢 1. Simple (age 12)
Picture a group project. First everyone talks to each other and shares notes (that is attention). Then everyone goes back to their own desk and works on their own notes alone (that is the feed-forward network). The desk work uses the same method for every student, but each student only uses their own notes, and nobody peeks at a neighbour.

**Example:** In the sentence "The bank by the river was steep", attention lets the word "bank" notice "river". The feed-forward step then turns "bank + river" into "the side of a river", and not "a place with money".

## 🔵 2. How it works
1. A token is a list of numbers (a vector, see [[Embedding]]). Say it has 512 numbers.
2. **Step one: widen.** A [[Matrix]] turns the 512 numbers into a longer list, often 4 times longer (2048 numbers). A longer list gives the model more room to spot patterns.
3. **Step two: switch things off.** Every number below zero is set to zero (an [[Activation Function]]; the simple one used in the first Transformer is called [[ReLU]]). Only some of the 2048 "hidden units" stay on, depending on the token.
4. **Step three: shrink back.** A second matrix mixes the numbers that stayed on and brings the list back to 512 numbers.
5. The result is added back to the token's original vector (a [[Residual Connection]]) and tidied up with [[Layer Normalization]].
6. **Same method everywhere, but no talking.** The same two matrices are used for every token in the text, and one token's result never depends on another token. That is called "position-wise". All tokens can be done at the same time.
7. Why the switch-off step matters: without it, the two matrices would collapse into one, and the layer would add almost nothing. The code in level 3 shows this.

## 🟠 3. Technical detail
- **Place in the Transformer.** Each layer is: attention, then feed-forward, each wrapped with a [[Residual Connection]] and [[Layer Normalization]]. Attention is the only place where tokens exchange information. The feed-forward network works per token.
- **Shape.** Input width $d_{model}$, hidden width $d_{ff}$, output width $d_{model}$. The original Transformer used 512 and 2048 (read in the paper, section 3.3 and Table 3); a ratio of 4 is the common rule of thumb after that (**from memory**, not checked in a source).
- **Size.** In one layer of the original base model, the feed-forward weights are twice the size of the attention weights (the code below counts 2,099,712 against 1,050,624). That is two-thirds of the weights in a layer, ignoring the word-embedding table. Geva et al. say the same thing for real models ("about two-thirds of the parameters", as I read in their abstract).
- **Key-value view.** Read the output as a sum: for each hidden unit $j$, take its on/off value and multiply it by row $j$ of the second matrix. The first matrix's columns act like "keys" (patterns to spot) and the second matrix's rows like "values" (what to add to the token when the pattern is spotted). Geva et al. (2020) argue that trained models behave like this. I only read their abstract, so the details are not checked.
- **Variants (from memory).** Newer models often swap ReLU for smoother functions such as GELU or gated forms such as SwiGLU. Some replace one big feed-forward layer by many smaller ones and send each token to only a few of them: see [[Mixture of Experts]].
- **Fine-tuning.** [[LoRA]] often adds its small update matrices to the attention matrices, but it can be placed on the feed-forward matrices too.
- **The code below is a toy.** Random weights, 4 numbers per token, hidden width 16. It checks the position-wise property, the collapse without the switch-off step, the key-value reading, and the weight count.

```python
import numpy as np

rng = np.random.RandomState(0)
d, d_ff, T = 4, 16, 5                           # width of a token vector, hidden width, tokens in the text
W1, b1 = rng.randn(d, d_ff), rng.randn(d_ff)
W2, b2 = rng.randn(d_ff, d), rng.randn(d)


def ffn(x):
    return np.maximum(0, x @ W1 + b1) @ W2 + b2   # FFN(x) = max(0, x W1 + b1) W2 + b2


X = rng.randn(T, d)                             # 5 token vectors, one per row
whole = ffn(X)
one_by_one = np.stack([ffn(x) for x in X])
print("whole text at once == token by token:", np.allclose(whole, one_by_one))
perm = rng.permutation(T)
print("shuffle tokens, outputs shuffle the same way (no mixing between tokens):", np.allclose(ffn(X[perm]), whole[perm]))

# without the ReLU, two layers are just one matrix
lin = (X @ W1 + b1) @ W2 + b2
one = X @ (W1 @ W2) + (b1 @ W2 + b2)
print("no ReLU: two layers == one layer:", np.allclose(lin, one), "| with ReLU:", np.allclose(whole, one))

# the same output, read as a sum of memories: ReLU(x W1 + b1)_j times row j of W2
x = X[0]
h = np.maximum(0, x @ W1 + b1)
print("hidden units switched on for token 0:", int((h > 0).sum()), "of", d_ff)
print("output == sum of (activation j) * (row j of W2) + b2:", np.allclose(sum(h[j] * W2[j] for j in range(d_ff)) + b2, ffn(x)))

# size in the base model of Vaswani et al.: d_model 512, d_ff 2048, 8 heads of 64
dm, df = 512, 2048
ffn_p = dm * df + df + df * dm + dm
att_p = 4 * dm * dm + 4 * dm                    # W_Q, W_K, W_V, W_O with biases
print("one layer: FFN", ffn_p, "attention", att_p, "FFN share of the two:", round(ffn_p / (ffn_p + att_p), 3), "ratio", round(ffn_p / att_p, 2))
print("with d_ff = 4 * d_model the FFN weights are", 2 * dm * 4 * dm, "= 8 * d_model^2")
```

What it printed: all three yes/no checks about position-wise behaviour and the sum-of-memories view were `True`. Without ReLU, two layers equal one (`True`); with ReLU they do not (`False`). For token 0, 2 of the 16 hidden units were on (random weights, so this number says nothing about real models). The weight count was 2,099,712 against 1,050,624, a share of 0.667.

## 🔴 4. Deep / formal
For one token vector $x$ (a row of numbers), the layer in the original Transformer is:

$$
\mathrm{FFN}(x)=\max(0,\;xW_1+b_1)\,W_2+b_2
$$

**Where:**
- $x$ is the token's vector, $d_{model}$ numbers long (512 in the paper).
- $W_1$ is a $d_{model}\times d_{ff}$ matrix and $b_1$ a bias vector of length $d_{ff}$ (2048 in the paper). They widen the vector.
- $\max(0,\cdot)$ sets every negative number to 0 (ReLU), one number at a time.
- $W_2$ is a $d_{ff}\times d_{model}$ matrix and $b_2$ a bias vector of length $d_{model}$. They shrink the vector back.
- The same $W_1,b_1,W_2,b_2$ are used for every position in a layer; each layer has its own set.
- **Matters:** the $\max(0,\cdot)$ is what makes the layer do something new. Without it, $(xW_1+b_1)W_2+b_2=x(W_1W_2)+(b_1W_2+b_2)$, a single matrix, so the width $d_{ff}$ would be wasted. The code showed this collapse (`True` without ReLU, `False` with it). Because $x$ appears alone (no other token), the layer cannot move information between positions; that job stays with attention.

**Example (tiny numbers):** $x=(1,\,2)$, with $W_1=\begin{pmatrix}1&-1&0\\0&1&1\end{pmatrix}$, $b_1=(0,-2,0)$, $W_2=\begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix}$, $b_2=(0,0)$.
- Widen: $xW_1=1\cdot(1,-1,0)+2\cdot(0,1,1)=(1,1,2)$. Add $b_1$: $(1,-1,2)$.
- Switch off negatives: $(1,0,2)$. The middle unit is off.
- Shrink: $1\cdot(1,0)+0\cdot(0,1)+2\cdot(1,1)=(3,2)$. Add $b_2$: $\mathbf{(3,2)}$.
- Meaning: the first and third hidden units fired, so their rows of $W_2$ were added (once and twice). The middle unit stayed silent and contributed nothing. This is the "sum of memories" view in level 3.

**Further notes:**
- **Size trade.** With $d_{ff}=4d_{model}$ the two matrices hold $8d_{model}^2$ weights (code: 2,097,152 for 512). Attention holds $4d_{model}^2$. So the feed-forward part is where most weights sit, and also where most of the arithmetic goes for short texts.
- **Published sweep of the width (Table 3 of the paper, English-to-German dev set, other settings as in the base model).** $d_{ff}=1024$: BLEU 25.4, 53 million parameters. $d_{ff}=2048$ (base): 25.8, 65 million. $d_{ff}=4096$: 26.2, 90 million. A wider layer scored higher on this one test, at the cost of more weights. The paper did not test removing the layer, so this table does not show what the layer is worth in total.
- **What the toy cannot show:** any real behaviour of a trained model, and whether the key-value story holds beyond the cases Geva et al. looked at.

**What came before, and how this improved on it:**
- **Before:** translation models built on recurrence ([[Recurrent Neural Network]]) or convolutions, and, in the Transformer's own design, attention that mixes the tokens' value vectors by weighted averaging.
- **Its limit:** averaging gives a result that is a mix of the inputs. A layer made only of averaging and straight matrices stays linear in the values, so it cannot form new, non-linear combinations of what was gathered. The code shows the same effect for two plain matrices.
- **What changed:** Vaswani et al. (arXiv 1706.03762, 2017; section 3.3 read) put a per-token two-layer network with a ReLU after every attention sub-layer, with the same weights across positions and different weights per layer.
- **How much better:** I did not find a published test that removes the feed-forward layer, so there is no "with versus without" figure here. What I have: in the code, a share of 0.667 of the layer's weights, and the Table 3 sweep above (BLEU 25.4, 25.8, 26.2 for widths 1024, 2048, 4096).

## Related
- [[Transformer]] · [[Attention]] · [[Self-Attention]] · [[Residual Connection]] · [[Layer Normalization]] · [[Activation Function]] · [[ReLU]] · [[Neural Network]] · [[Perceptron]] · [[Matrix]] · [[Embedding]] · [[Mixture of Experts]] · [[LoRA]] · [[BERT]] · [[LLM]] · [[Attention Is All You Need 2017]] · [[ReLU]]

## Sources
- Vaswani et al., *Attention Is All You Need*, arXiv 1706.03762 (2017): section 3.3 and Table 3 read in the PDF text. See also [[Attention Is All You Need 2017]].
- Geva, Schuster, Berant, Levy, *Transformer Feed-Forward Layers Are Key-Value Memories*, arXiv 2012.14913 (submitted 29 Dec 2020; EMNLP 2021). **Abstract only** was read.
- The ratio of 4, and the names GELU and SwiGLU, are **from memory**.
- Image: none. A Wikimedia Commons search found only unrelated PDFs, so none is used.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/neural-nets Feed-forward network (in a transformer):::The step after attention where each token goes back to its own desk and works on its own notes alone. The same method is used for every token, and tokens do not talk to each other here.

#flashcards/ml/formula #flashcards/topic/neural-nets Write the transformer feed-forward network and explain each symbol.
?
$$\mathrm{FFN}(x)=\max(0,\;xW_1+b_1)\,W_2+b_2$$
- $x$: the token's vector, $d_{model}$ numbers long (512 in the original paper).
- $W_1$: a $d_{model}\times d_{ff}$ matrix and $b_1$ a bias of length $d_{ff}$ (2048 in the paper). They widen the vector.
- $\max(0,\cdot)$: sets every negative number to 0 (ReLU), one number at a time.
- $W_2$: a $d_{ff}\times d_{model}$ matrix and $b_2$ its bias. They shrink the vector back to $d_{model}$.

#flashcards/ml/cloze #flashcards/topic/neural-nets The feed-forward network ==widens== each token's vector, applies a ==ReLU==, then shrinks it back.

#flashcards/ml/example #flashcards/topic/neural-nets $x=(1,2)$, $W_1=\begin{pmatrix}1&-1&0\\0&1&1\end{pmatrix}$, $b_1=(0,-2,0)$, $W_2=\begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix}$, $b_2=0$. What is $\mathrm{FFN}(x)$?::Widen: xW₁ = (1, 1, 2), add b₁: (1, −1, 2). ReLU: (1, 0, 2), the middle unit is off. Shrink: 1·(1,0) + 0·(0,1) + 2·(1,1) = (3, 2).
