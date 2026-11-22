---
tags: [positional-encoding, transformer, attention, LLM, deep-learning, embedding]
status: growing
level: 4
tier: 2
---

# Positional Encoding

## 🟢 1. Simple (age 12)
Positional encoding is a **way to tell the AI which word comes first, second, third**. The AI looks at all the words at once, like a pile of word-cards thrown on a table. Without extra help it cannot tell "dog bites man" from "man bites dog". So each card gets a small **position stamp**.

**Example:** You throw the cards "dog", "bites", "man" into a bag. Order is lost. Now you write a tiny number in the corner of each card: 1, 2, 3. Even if the cards get shuffled, you can put them back in order. A positional encoding is that number in the corner, written in a clever way that a computer can use easily.

## 🔵 2. How it works
1. A [[Transformer]] reads all the words of a sentence **at the same time**, not one after another. This is what makes it fast on a [[GPU]].
2. Its main step, [[Attention]], treats the words as an unordered set. Swap two words and the maths for each word comes out the same, just swapped. So by itself the model is **blind to word order**.
3. Fix: before the first layer, **add a position signal** to the number-list ([[Embedding]]) of every word. Word 1 gets signal 1, word 2 gets signal 2, and so on.
4. The signal is a list of numbers that is **different for every position**, so the model can tell positions apart.
5. After the signal is added, the same word at position 1 and at position 5 looks slightly different, and attention can use the difference ("the word right before me", "three words back").
6. There are several ways to make the signal: a fixed pattern of waves (sine and cosine, the original idea), numbers the model learns, or twisting the attention maths itself (rotary, below).

![[positional-encoding.png]]
*The wave pattern from the original Transformer paper, drawn as a colour map. Across the bottom is the word position (0 to 1000), and up the side are the 100 numbers of the signal. Red is $+1$, blue is $-1$. Low down, the numbers wiggle quickly with position. High up, they change very slowly. Every position (every vertical stripe) has its own unique pattern of colours, like a fingerprint. Image: Nils Blümer, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Absolute_positional_encoding.png), CC BY 4.0.*

## 🟠 3. Technical detail
- **Why order is lost:** attention compares every word with every other and takes weighted sums. Reordering the inputs just reorders the outputs ("permutation equivariance"), so it never sees positions.
- **Sinusoidal (fixed):** from *Attention Is All You Need*. Use sine and cosine waves of many different speeds, one pair per two numbers of the signal. Nothing to learn, and it can be computed for any position, even longer than seen in training (in theory, though models often do poorly far past their training length).
- **Learned absolute:** a table with one trainable list per position (used in GPT-2 and BERT, from memory). Simple, but limited to the maximum length it was trained for.
- **Relative and rotary (RoPE):** instead of adding a signal to the words, rotate the query and key vectors by an angle that depends on position, so attention scores depend on the **distance** between two words. Used in many recent [[LLM]]s such as the LLaMA family (from memory; check your model).
- **ALiBi:** subtracts a penalty from attention scores that grows with distance.
- **Context length:** how long a text the model can handle is tied to its position scheme. Extending it usually means rescaling the positions and some extra training.
- **Where it is added:** once at the input (absolute schemes), or inside every attention layer (RoPE, ALiBi).

```python
import numpy as np
def pe(pos, d=4):
    i = np.arange(d // 2)
    w = pos / 10000 ** (2 * i / d)
    out = np.empty(d); out[0::2] = np.sin(w); out[1::2] = np.cos(w)
    return out
print(pe(0).round(4))              # [0. 1. 0. 1.]
print(pe(1).round(4))              # [0.8415 0.5403 0.01   1.    ]
print(pe(2).round(4))              # [ 0.9093 -0.4161  0.02    0.9998]
e = np.array([.2, -.1, .4, .3])    # a made-up word vector
print((e + pe(1)).round(3))        # [1.041 0.44  0.41  1.3  ]

# attention alone cannot see order: shuffling inputs just shuffles outputs
softmax = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
att = lambda X: softmax(X @ X.T) @ X
X = np.array([[1., 0.], [0., 1.], [1., 1.]]); p = [2, 0, 1]
print(np.allclose(att(X)[p], att(X[p])))   # True
```

## 🔴 4. Deep / formal
**The sinusoidal signal.** For a word at position $\text{pos}$ ($0,1,2,\dots$) and a signal of length $d$, number $2i$ and number $2i+1$ are:

$$
PE_{(\text{pos},\,2i)}=\sin\!\Big(\frac{\text{pos}}{10000^{2i/d}}\Big),\qquad
PE_{(\text{pos},\,2i+1)}=\cos\!\Big(\frac{\text{pos}}{10000^{2i/d}}\Big)
$$

**Where:**
- $\text{pos}$ is the position of the word in the sentence: 0 for the first, 1 for the second, and so on.
- $d$ is the length of the signal, which equals the length of the word vectors (so the two can be added). $i$ numbers the pairs: $i=0,1,\dots,d/2-1$.
- $2i$ and $2i+1$ are the two slots of pair $i$: the even slot gets a sine and the odd slot gets a cosine of the same angle.
- $10000^{2i/d}$ sets the **wave speed** of pair $i$. For $i=0$ it is $1$ (fast waves). As $i$ grows it climbs towards $10000$ (very slow waves). **Matters:** many speeds together work like the digits of a number: the fast pairs tell neighbouring positions apart, and the slow pairs tell far-apart positions apart. This is the pattern in the picture.
- $\sin$ and $\cos$ give values between $-1$ and $1$. **Matters:** the values stay small, so the signal does not drown out the word's own numbers, and using both sine and cosine means a shift by a fixed number of positions acts like a simple rotation, which makes "$k$ words later" easy to learn (stated from the original paper's argument, from memory).
- The constant $10000$ is a design choice from the original paper, not a law of nature.

**Example (tiny numbers):** $d=4$, so there are two pairs ($i=0,1$).
- Pair $i=0$: speed $10000^{0}=1$. Pair $i=1$: speed $10000^{2/4}=100$.
- Position 1: $\sin(1)=\mathbf{0.8415}$, $\cos(1)=\mathbf{0.5403}$; then $\sin(1/100)=\mathbf{0.0100}$, $\cos(0.01)=\mathbf{1.0000}$ (really $0.99995$). So $PE_1=(0.8415,\,0.5403,\,0.0100,\,1.0000)$.
- Position 2: $\sin(2)=0.9093$, $\cos(2)=-0.4161$; $\sin(0.02)=0.0200$, $\cos(0.02)=0.9998$. So $PE_2=(0.9093,\,-0.4161,\,0.0200,\,0.9998)$.
- Position 0: $(\sin0,\cos0,\sin0,\cos0)=(0,1,0,1)$.
- Meaning: the first pair changes a lot between positions 1 and 2, and the second pair barely moves. Together the four numbers are a unique "address" for each position.

**Adding it to the word.** The model's first input for a word is the sum of the word's vector and its position signal:

$$
\mathbf{x}_{\text{pos}}=\mathbf{e}_{\text{token}}+PE_{\text{pos}}
$$

**Where:**
- $\mathbf{e}_{\text{token}}$ is the word's [[Embedding]]: the list of numbers describing its meaning.
- $PE_{\text{pos}}$ is the position signal above for the word's position.
- $\mathbf{x}_{\text{pos}}$ is the combined input that goes into the first layer. **Matters:** meaning and position are mixed into one list, so attention gets both. Adding (not joining them end to end) keeps the vector the same length, so nothing else in the model changes size.

**Example (tiny numbers):** the word "dog" has $\mathbf{e}=(0.2,\,-0.1,\,0.4,\,0.3)$ and sits at position 1.
- $\mathbf{x}=(0.2+0.8415,\ -0.1+0.5403,\ 0.4+0.01,\ 0.3+1.0)=(\mathbf{1.041},\,\mathbf{0.440},\,\mathbf{0.410},\,\mathbf{1.300})$.
- The same word at position 2 gives $(0.2+0.9093,\ -0.1-0.4161,\ 0.4+0.02,\ 0.3+0.9998)=(\mathbf{1.109},\,\mathbf{-0.516},\,\mathbf{0.420},\,\mathbf{1.300})$.
- Meaning: the same word now looks different depending on where it sits, so the model can tell "dog" at the start of a sentence from "dog" later on.

**Further notes:**
- **Why not just use the number 1, 2, 3?** Raw numbers grow without limit and have a single scale. Waves stay between $-1$ and $1$ and encode position at many scales at once.
- **Extrapolation is hard:** a model trained on positions up to 4096 usually cannot be trusted at 100,000 without changing the position scheme and training further. (Common knowledge in the field; exact limits depend on the model.)
- **Position and [[Normalization]]:** the position signal is added before the first normalisation layer, so its size relative to the word vector matters; word vectors are often scaled to match.
- **Open question:** which position scheme is best for very long contexts is an active research area.

**What came before, and how this improved on it:**
- **Before:** no position information at all, so attention treats the words as a set.
- **Its limit:** "the dog bit the man" and "the man bit the dog" give the same word-by-word results, only reordered.
- **What changed:** add a position-dependent vector to each word (sinusoids in Vaswani et al., arXiv 1706.03762, abstract read).
- **How much better:** in the code below, shuffling 5 words changes the output only by a reordering (maximum difference $4.4\times10^{-16}$, which is just rounding) when no positions are added, and by up to 10.171 when they are, so order now matters.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0); n, d = 5, 8
X0 = rng.standard_normal((n, d)); Wq, Wk, Wv = (rng.standard_normal((d, d)) for _ in range(3))
def attn(X):
    s = (X @ Wq) @ (X @ Wk).T / np.sqrt(d); a = np.exp(s); a /= a.sum(1, keepdims=True); return a @ (X @ Wv)
perm = rng.permutation(n)
print("shuffle the words, no position info: output rows are just shuffled too, max difference",
      float(np.abs(attn(X0[perm]) - attn(X0)[perm]).max()))
P = np.array([[np.sin(p / 10000 ** (i / d)) if i % 2 == 0 else np.cos(p / 10000 ** ((i - 1) / d)) for i in range(d)] for p in range(n)])
print("with positions added: shuffled input gives a different result, max difference",
      round(float(np.abs(attn(X0[perm] + P) - attn(X0 + P)[perm]).max()), 3))
```

## Related
- [[Transformer]] · [[Attention]] · [[Embedding]] · [[Tokenization]] · [[LLM]] · [[Normalization]] · [[Vector]] · [[GPU]] · [[Neural Network]] · [[Self-Attention]] · [[RoPE]] · [[BERT]] · [[Vision Transformer]]

## Sources
- Vaswani et al., *Attention Is All You Need*, 2017 (sinusoidal encoding)
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021 (from memory)
- Press et al., *Train Short, Test Long: Attention with Linear Biases* (ALiBi), 2021 (from memory)
- Vaswani et al., arXiv 1706.03762 (abstract read for the history block)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/transformers-nlp Positional encoding:::A way to tell the AI which word comes first, second, third. Without it, a transformer sees the words like cards thrown on a table and cannot tell "dog bites man" from "man bites dog".

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write the sinusoidal positional encoding and explain each symbol.
?
$$PE_{(\text{pos},\,2i)}=\sin\!\Big(\frac{\text{pos}}{10000^{2i/d}}\Big),\qquad PE_{(\text{pos},\,2i+1)}=\cos\!\Big(\frac{\text{pos}}{10000^{2i/d}}\Big)$$
- $\text{pos}$: the position of the word in the sentence (0 for the first, 1 for the second, ...).
- $d$: the length of the signal, equal to the length of the word vectors so the two can be added. $i$: numbers the pairs, $i=0,1,\dots,d/2-1$.
- $2i$ and $2i+1$: the two slots of pair $i$, one sine and one cosine.
- $10000^{2i/d}$: sets how slow the wave is. Low pairs wiggle fast, high pairs change slowly, so every position gets its own unique pattern.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write how the position signal is combined with the word, and explain each symbol.
?
$$\mathbf{x}_{\text{pos}}=\mathbf{e}_{\text{token}}+PE_{\text{pos}}$$
- $\mathbf{e}_{\text{token}}$: the word's embedding, the list of numbers describing its meaning.
- $PE_{\text{pos}}$: the position signal for the word's position.
- $\mathbf{x}_{\text{pos}}$: the combined input that goes into the first layer. Meaning and position are mixed into one vector.

#flashcards/ml/cloze #flashcards/topic/transformers-nlp The position signal is ==added== to the word's embedding, and every position has its own unique wave ==pattern==.

#flashcards/ml/example #flashcards/topic/transformers-nlp With $d=4$, what are the first two numbers of $PE_1$, and what is $\mathbf{x}$ for "dog" ($\mathbf{e}=(0.2,-0.1,0.4,0.3)$) at position 1?::PE₁ starts (sin 1, cos 1) = (0.8415, 0.5403), then (sin 0.01, cos 0.01) ≈ (0.0100, 1.0000). x = (1.041, 0.440, 0.410, 1.300).

#flashcards/ml/image #flashcards/topic/transformers-nlp ![[positional-encoding.png]]
What does this picture show?
?
The wave pattern from the original Transformer paper as a colour map. Across the bottom is the word position (0 to 1000), up the side are the 100 numbers of the signal. Red is +1, blue is −1. Low numbers wiggle quickly, high ones change slowly. Each position has its own pattern, like a fingerprint.
