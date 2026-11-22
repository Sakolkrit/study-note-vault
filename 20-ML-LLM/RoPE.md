---
tags: [LLM, transformer, attention, positional-encoding, RoPE, linear-algebra]
status: growing
level: 4
tier: 4
---

# RoPE

RoPE (rotary position embedding) tells an attention layer **where each word is by turning its query and key vectors by an angle that depends on the position**. When two turned vectors are compared, only the *distance* between the two words is left over. [[Positional Encoding]] gives the overview of the position problem and mentions RoPE in a line. This note is the full story on RoPE, with a numeric check of its key property.

## 🟢 1. Simple (age 12)
Imagine every word in a sentence is a clock hand. The first word's hand points straight up. The second word's hand is turned a little. The third is turned a little more, and so on, so each word's hand shows its place in line. Now to see how far apart two words are, you do not need to know the exact places. You only compare how far apart their two hands are turned. Words next to each other have hands that are close, and words far apart have hands that are far apart. RoPE does exactly this to the numbers inside a language model.

**Example:** In a line of kids with arrows, kid number 3 and kid number 5 are 2 places apart, and so are kid 10 and kid 12. If each kid's arrow is turned by a little more than the kid before, the "gap" between the arrows of 3 and 5 is the same as between 10 and 12. The model can learn "two words apart" and it works anywhere in the sentence.

## 🔵 2. How it works
1. **Each word becomes a list of numbers** (see [[Embedding]]). In attention ([[Self-Attention]]) it is turned into a *query* and a *key*, two more lists of numbers.
2. **Group the numbers in pairs.** A pair, like (0.3, 0.8), can be drawn as a point on a flat page, or as an arrow from the centre.
3. **Choose a turning speed for each pair.** The first pairs turn fast and the last pairs turn very slowly.
4. **Turn each pair by (position × its speed).** The word at position 5 is turned 5 steps of that pair's speed, the word at position 6 is turned 6 steps, and so on. Only the query and the key are turned, not the value.
5. **Compare as usual.** The attention score between a query and a key is still a dot product (how much they point the same way). But because both were turned, the score now depends only on **how far apart the two positions are**, not on where they are.
6. **Nothing is added to the word's numbers.** The meaning in the word stays the same length. Only the *angle* is changed, so the length of each arrow does not change.

![[rotation-2d-axes.svg]]
*A turn by an angle θ in the flat page. The blue axes are the original directions, the red axes are the same page turned by θ, and P is a point. Turning a vector by θ gives the same new coordinates as turning the axes the other way by θ. RoPE uses this kind of flat turn, many times over, once for each pair of numbers. Image: Guy Vandegrift, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Rotation_of_coordinate_unit_vectors.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **A rotation per pair:** the $d$ numbers of a query or key are split into $d/2$ pairs. Pair $i$ is turned by the angle $\text{position}\times\theta_i$, with a different speed $\theta_i$ for each pair (equation 3 below).
- **Fast and slow pairs:** the first pairs turn fast, so they tell neighbouring words apart. The last pairs turn so slowly that they barely change between nearby words, but separate words that are very far apart. It works like the hands of a clock: seconds, minutes and hours.
- **Relative by construction:** the dot product of a turned query and a turned key depends only on the *difference* of the two positions (equation 2 and the code). Nothing else has to learn that "3 apart" is "3 apart" in every part of the text.
- **No extra learned weights, no added signal:** unlike a learned table of positions ([[Positional Encoding]]), RoPE has nothing to train and does not change the word vector's size. It is applied inside every attention layer, to the queries and keys only (queries and keys: checked in the RoFormer paper; that LLaMA models use RoPE is confirmed by the Position Interpolation paper's abstract, which calls LLaMA a RoPE-based model; other recent [[LLM]]s I did not check, so check your model).
- **Lengths are kept:** a turn does not change the size of a vector, so scores do not blow up or shrink because of position (checked in the code).
- **Compared with the older additive way:** adding sine and cosine waves to the word (the original [[Transformer]] method) mixes position into the meaning before attention. The score then depends on *both* positions, not just the gap (the code shows that).
- **Longer texts than the model saw:** positions beyond the training length mean angles the model never saw. A common fix is to scale positions down (*position interpolation*), for example turning position 6,000 into 3,000 to double the length (equation 3 below), usually with a little extra training (checked: Chen et al., 2023, arXiv:2306.15595, abstract: scaling positions down extends LLaMA 7B to 65B to a 32,768 window with fine-tuning within 1,000 steps; other methods vary, and quality at long length is an active research area, from memory).
- **A fade with distance:** with the standard speeds, the scores between similar vectors tend to get smaller as the distance grows, which fits the idea that near words matter more (a claim of the original paper, checked: RoFormer's abstract says the dependency between tokens decays with distance, and its Section 3.4.3 shows this with these speeds). The code shows one case where it falls, not a general proof.
- **Cost:** a handful of sine and cosine values per position and a few multiplications per pair. It is cheap compared with the attention itself. See also [[GPU]].

The code turns vectors by RoPE and checks five things: (1) one turn by hand, (2) that scores depend only on distance, (3) that the older additive method does not, (4) that shifting every position leaves the whole attention pattern unchanged, and (5) the turning speeds and the fading of a score with distance.

```python
import numpy as np
def rope(x, pos, base=10000.0):
    d = len(x); th = base ** (-2 * np.arange(d // 2) / d)          # one turning speed per pair of numbers
    a = pos * th; c, s = np.cos(a), np.sin(a)
    y = np.empty_like(x); y[0::2] = x[0::2] * c - x[1::2] * s; y[1::2] = x[0::2] * s + x[1::2] * c
    return y
# 1) one pair, by hand: rotate (1, 0) by position 2 at speed 0.5 -> angle 1 radian
a = 2 * 0.5; print('rotated:', np.round([np.cos(a) * 1 - np.sin(a) * 0, np.sin(a) * 1 + np.cos(a) * 0], 4))
# 2) the score depends only on the distance n - m
rng = np.random.default_rng(0); q, k = rng.normal(size=8), rng.normal(size=8)
sc = lambda m, n: float(rope(q, m) @ rope(k, n))
print('same offset 2:', [round(sc(m, m + 2), 4) for m in (0, 5, 50, 500)])
print('offset 2 vs 3:', round(sc(7, 9), 4), round(sc(7, 10), 4))
print('length kept:', round(float(np.linalg.norm(q)), 4), round(float(np.linalg.norm(rope(q, 123))), 4))
# 3) adding sinusoidal signals instead (the older way) does NOT give this
def pe(pos, d=8):
    w = pos / 10000 ** (2 * np.arange(d // 2) / d); out = np.empty(d); out[0::2] = np.sin(w); out[1::2] = np.cos(w); return out
add = lambda m, n: float((q + pe(m)) @ (k + pe(n)))
print('additive, same offset 2:', [round(add(m, m + 2), 4) for m in (0, 5, 50, 500)])
# 4) shifting every position by 100 leaves the whole attention pattern unchanged
n, d = 6, 8; Q, K = rng.normal(size=(n, d)), rng.normal(size=(n, d))
def attn(shift):
    S = np.array([[rope(Q[i], i + shift) @ rope(K[j], j + shift) for j in range(n)] for i in range(n)]) / np.sqrt(d)
    e = np.exp(S - S.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
print('weights identical after shift by 100:', np.allclose(attn(0), attn(100)))
# 5) turning speeds for d = 8, and how a score between identical vectors fades with distance (d = 64)
d8 = 8; print('speeds:', (10000.0 ** (-2 * np.arange(d8 // 2) / d8)).round(4).tolist())
ones = np.ones(64); print('score of equal vectors by distance:', {D: round(float(rope(ones, 0) @ rope(ones, D)), 1) for D in (0, 1, 10, 100, 1000)})
# rotated: [0.5403 0.8415]
# same offset 2: [-1.3804, -1.3804, -1.3804, -1.3804]
# offset 2 vs 3: -1.3804 -1.6284
# length kept: 1.8627 1.8627
# additive, same offset 2: [0.6204, 1.8311, -0.7415, 5.6334]
# weights identical after shift by 100: True
# speeds: [1.0, 0.1, 0.01, 0.001]
# score of equal vectors by distance: {0: 64.0, 1: 61.8, 10: 42.1, 100: 35.7, 1000: 17.9}
```
Reading the output (NumPy only; fixed seeds):
- With RoPE, the score between a query at position $m$ and a key at position $m+2$ is $-1.3804$ whether $m$ is 0, 5, 50 or 500. A gap of 3 gives a different score ($-1.6284$ against $-1.3804$ for gap 2), so the model can still tell gaps apart.
- The length of a vector stays at 1.8627 after turning it to position 123.
- With the older additive method, the same four pairs (gap 2 everywhere) give four different scores: $0.62, 1.83, -0.74, 5.63$. The score there depends on the absolute positions.
- Shifting all positions by 100 gives identical attention weights, so the whole attention pattern depends only on gaps.
- The speeds for $d=8$ run from 1 down to 0.001. For two identical 64-number vectors, the score falls from 64 (same place) to 61.8, 42.1, 35.7 and 17.9 at distances 1, 10, 100 and 1,000. That is a fall, not a smooth law, and it is one made-up vector (all ones). Other vectors would give other curves, so do not read it as a rule.

## 🔴 4. Deep / formal

**One pair, turned.** Take a query or key, split into pairs. For a pair $(x_0,x_1)$ at position $m$, with speed $\theta$, RoPE gives

$$\begin{pmatrix}x_0'\\ x_1'\end{pmatrix}=\begin{pmatrix}\cos m\theta & -\sin m\theta\\ \sin m\theta & \cos m\theta\end{pmatrix}\begin{pmatrix}x_0\\ x_1\end{pmatrix}$$

**Where:**
- $(x_0,x_1)$ is one pair of numbers from the query or key, drawn as a point on a flat page. **Matters:** the whole vector is handled pair by pair, so this is repeated $d/2$ times with different speeds.
- $m$ is the position of the word (0, 1, 2, ...). **Matters:** it is the only way position enters, so two words at different positions are turned by different amounts.
- $\theta$ is the speed, how many radians of turn each step of position adds. **Matters:** if every pair had the same speed, the model could not tell near from far at many scales. Different speeds give the clock-hand effect.
- The 2×2 grid of cosines and sines is a *rotation [[Matrix]]*. **Matters:** it turns a vector without changing its length. That keeps attention scores in a sane range (checked in the code).
- $(x_0',x_1')$ is the turned pair, which replaces the original in the query or key.

**Example (tiny numbers):** $\theta=0.5$, position $m=2$, so the angle is $m\theta=1$ radian. $\cos1=0.5403$ and $\sin1=0.8415$.
1. The pair $(1,0)$ becomes $(0.5403\times1-0.8415\times0,\ 0.8415\times1+0.5403\times0)=(0.5403,\,0.8415)$.
2. The pair $(0,1)$ becomes $(-0.8415,\,0.5403)$.
3. Meaning: the point $(1,0)$ moved about 57° up the circle. The length is still 1, since $0.5403^2+0.8415^2=1$. Only the direction changed, and by an amount that is set by the position.

**Why only the gap is left.** Let $q$ and $k$ be a query and a key (as pairs), at positions $m$ and $n$. Write $R(\alpha)$ for the turn by angle $\alpha$. Then

$$\big(R(m\theta)\,q\big)\cdot\big(R(n\theta)\,k\big)=q^{\top}R\big((n-m)\theta\big)\,k$$

**Where:**
- $R(m\theta)q$ and $R(n\theta)k$ are the turned query and turned key. **Matters:** these are what attention really compares.
- $\cdot$ is the dot product, the usual attention score before the softmax (see [[Softmax]]). **Matters:** it is how much the two vectors point the same way.
- $q^{\top}R((n-m)\theta)k$ is the right-hand side: the original query against the key turned by the *gap* $n-m$. **Matters:** positions $m$ and $n$ have disappeared and only $n-m$ is left. This is the whole point: the score is relative.
- The reason is that turning both vectors by the same extra amount does not change the dot product, and turns by $m\theta$ and $n\theta$ combine to a net turn of $(n-m)\theta$ (the standard rotation facts; I checked the result numerically and did not write out the algebra).

**Example (tiny numbers):** $\theta=0.5$, $q=(1,0)$, $k=(1,0.5)$.
1. Positions $m=3$, $n=5$. Turned query: angle $1.5$ gives $(0.0707,\,0.9975)$. Turned key: angle $2.5$ gives $(-1.1004,\,0.1979)$.
2. Dot product: $0.0707\times(-1.1004)+0.9975\times0.1979=-0.0778+0.1974=0.1196$.
3. Positions $m=10$, $n=12$ (same gap of 2) also give $0.1196$, and so do $m=100$, $n=102$.
4. The shortcut: turn the key $k$ by the gap angle $2\times0.5=1$ rad and dot with $q$: $R(1)k=(0.1196,\,1.1117)$, and $q\cdot R(1)k=0.1196$. Same.
5. A gap of 3 gives $-0.4280$, so different gaps give different scores.
6. Meaning: the score is the same wherever the pair sits, and different for other gaps. This is what lets the model learn "the word two places back" once and use it anywhere in the text.

**The speeds.** For a head of size $d$, the pair $i$ ($i=0,\dots,d/2-1$) gets the speed

$$\theta_i=10000^{-2i/d},\qquad \text{wavelength}_i=\frac{2\pi}{\theta_i}\ \text{positions}$$

**Where:**
- $i$ numbers the pairs and $d$ is the head size. **Matters:** the speed falls smoothly from 1 at $i=0$ toward $1/10000$ at the last pair.
- $10000$ is the *base*, a design choice carried over from the sinusoidal scheme, not a law (checked: RoFormer sets $\theta_i=10000^{-2i/d}$ "following Vaswani et al. [2017]", whose sinusoids use the same 10000). **Matters:** a larger base makes the slow pairs slower, so they can separate positions over longer texts. Some long-context models use a larger base (from memory).
- $\theta_i$ is the angle added per position. **Matters:** it sets how fast pair $i$'s clock hand turns.
- The wavelength is how many positions it takes pair $i$ to make a full turn. **Matters:** a pair can only tell apart positions *within* one wavelength. After a full turn, it looks the same again. The slowest pairs have wavelengths far longer than typical texts, so they give a unique "address".

**Example (tiny numbers):** $d=8$, so $i=0,1,2,3$.
1. Speeds: $10000^{0}=1$, $10000^{-0.25}=0.1$, $10000^{-0.5}=0.01$, $10000^{-0.75}=0.001$.
2. Wavelengths: $2\pi/1=6.28$, $62.8$, $628$, $6283$ positions.
3. The first pair repeats about every 6 words, and the last only every 6,283 words.
4. Position interpolation: a model trained on 4,096 positions is asked to read 8,192. Scaling positions by $4096/8192=0.5$ turns position 6,000 into 3,000, which is a position it has seen turn angles for.
5. Meaning: fast pairs separate neighbours, slow pairs separate distant words, and stretching positions is a way to fit a longer text into the range of angles the model already knows. The price is that neighbouring words now differ by a smaller angle, so fine detail is blurred, which is why a little extra training is usually used (from memory).

**Other points (from memory, not checked here):**
- RoPE was introduced in the RoFormer paper (Su and colleagues, submitted April 2021; checked on arXiv).
- It is related to, but different from, ALiBi (Press et al., 2021, arXiv:2108.12409), which, per its abstract, biases the query-key scores with a penalty proportional to distance instead of turning vectors (checked on the abstract).
- Many ways of extending the context length of a RoPE model exist, and they differ in how they rescale the speeds. I have not compared them here.
- RoPE assumes pairs of numbers can be turned independently. Different ways of pairing the numbers (neighbouring or split in halves) are used in different libraries, and mixing them up gives wrong results.

**What came before, and how this improved on it:**
- **Before:** sinusoidal position vectors added to the word vectors (Vaswani et al., 2017).
- **Its limit:** the attention score then depends on both absolute positions, not only on how far apart the two words are.
- **What changed:** rotate the query and key by an angle that grows with position, so their dot product depends only on the offset (Su et al., arXiv 2104.09864, 20 April 2021, abstract read).
- **How much better:** the code below fixes the offset at 3 and moves the pair to 8 different absolute positions. With RoPE the score is the same every time ($-8.446152$). With added sinusoids and random projections it ranges from $-13.83$ to $20.719$. The toy does not test language quality. The abstract says RoFormer "consistently overcomes its alternatives" on long text classification benchmarks.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0); d = 64
q, k = rng.standard_normal(d), rng.standard_normal(d)
def rot(x, p):
    th = p * 10000.0 ** (-np.arange(0, d, 2) / d); c, s = np.cos(th), np.sin(th)
    y = x.copy(); y[0::2] = x[0::2] * c - x[1::2] * s; y[1::2] = x[0::2] * s + x[1::2] * c; return y
def pe(p):
    th = p * 10000.0 ** (-np.arange(0, d, 2) / d); e = np.zeros(d); e[0::2] = np.sin(th); e[1::2] = np.cos(th); return e
W = rng.standard_normal((d, d)) / np.sqrt(d); offset = 3
rs = [rot(q, m + offset) @ rot(k, m) for m in range(0, 400, 50)]
ab = [(W @ (q + pe(m + offset))) @ (W @ (k + pe(m))) for m in range(0, 400, 50)]
print("same offset 3, 8 different absolute positions:")
print("  RoPE score: min", round(min(rs), 6), "max", round(max(rs), 6))
print("  added sinusoids score: min", round(min(ab), 3), "max", round(max(ab), 3))
```

## Related
[[Positional Encoding]] · [[Self-Attention]] · [[Attention]] · [[Transformer]] · [[Matrix]] · [[Eigenvalue]] · [[Embedding]] · [[Softmax]] · [[LLM]] · [[GPU]] · [[Vector]] · [[FlashAttention]] · [[RoFormer 2021]]

## Sources
- Su, Lu, Pan, Murtadha, Wen and Liu (2021), "RoFormer: Enhanced Transformer with Rotary Position Embedding", arXiv:2104.09864 (checked: abstract, the base of 10000, the decay section; I did not read the proofs).
- Vaswani and colleagues (2017), "Attention Is All You Need", arXiv:1706.03762 (checked: the sinusoidal formula with 10000).
- Chen, Wong, Chen and Tian (2023), "Extending Context Window of Large Language Models via Positional Interpolation", arXiv:2306.15595 (abstract only). Other 2023 long-context methods were not checked.
- Press, Smith and Lewis (2021), ALiBi, arXiv:2108.12409 (abstract only).
- Figure: [Wikimedia Commons, Rotation_of_coordinate_unit_vectors.svg](https://commons.wikimedia.org/wiki/File:Rotation_of_coordinate_unit_vectors.svg), by Guy Vandegrift, CC BY-SA 4.0.
- The code, the relative-position check and every Example number were run for this note (NumPy only).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/transformers-nlp RoPE (rotary position embedding):::Each word's query and key is turned (rotated) by an angle that grows with its position, like a clock hand. Attention then only depends on how far apart two words are.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write the RoPE rotation of one pair of numbers and explain each symbol.
?
$$\begin{pmatrix}x_0'\\ x_1'\end{pmatrix}=\begin{pmatrix}\cos m\theta & -\sin m\theta\\ \sin m\theta & \cos m\theta\end{pmatrix}\begin{pmatrix}x_0\\ x_1\end{pmatrix}$$
- $(x_0,x_1)$: one pair of numbers from the query or key, drawn as a point on a flat page. The whole vector is handled pair by pair, $d/2$ times, with different speeds.
- $m$: the position of the word (0, 1, 2, ...). It is the only way position enters.
- $\theta$: the turning speed for this pair. $m\theta$ is the angle.
- The matrix is a rotation: it turns the point by $m\theta$ without changing its length.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write the key property of RoPE and explain each symbol.
?
$$\big(R(m\theta)\,q\big)\cdot\big(R(n\theta)\,k\big)=q^{\top}R\big((n-m)\theta\big)\,k$$
- $R(m\theta)q$, $R(n\theta)k$: the turned query (position $m$) and turned key (position $n$). These are what attention really compares.
- $\cdot$: the dot product, the attention score before the softmax.
- The result depends only on the gap $n-m$, not on the exact positions. So the model learns relative distance.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write how RoPE chooses its turning speeds and wavelengths, and explain each symbol.
?
$$\theta_i=10000^{-2i/d},\qquad \text{wavelength}_i=\frac{2\pi}{\theta_i}\ \text{positions}$$
- $i$: numbers the pairs. $d$: the head size. The speed falls smoothly from 1 at $i=0$ toward $1/10000$ at the last pair.
- $10000$: the base, a design choice carried over from the sinusoidal scheme, not a law.
- The wavelength is how many positions it takes for a pair to go once around.

#flashcards/ml/cloze #flashcards/topic/transformers-nlp RoPE's score depends only on the ==gap== between two positions, because turning both by their positions leaves a turn of ==(n − m)θ==.

#flashcards/ml/example #flashcards/topic/transformers-nlp $\theta=0.5$, position $m=2$. Where do the pairs (1, 0) and (0, 1) go?::Angle 1 rad: cos 1 = 0.5403, sin 1 = 0.8415. (1, 0) → (0.5403, 0.8415). (0, 1) → (−0.8415, 0.5403).

#flashcards/ml/example #flashcards/topic/transformers-nlp $\theta=0.5$, $q=(1,0)$, $k=(1,0.5)$. What is the score at positions (3, 5) and at (10, 12)?::Both give 0.1196, because the gap is 2 in both cases. Only the gap matters.

#flashcards/ml/example #flashcards/topic/transformers-nlp For $d=8$, what are the speeds and wavelengths?::Speeds 1, 0.1, 0.01, 0.001. Wavelengths 6.28, 62.8, 628, 6283 positions. The first pair repeats about every 6 words, the last only every 6,283.

#flashcards/ml/image #flashcards/topic/transformers-nlp ![[rotation-2d-axes.svg]]
What does this picture show?
?
A turn by an angle θ in the flat page. The blue axes are the original directions, the red axes are the same page turned by θ, and P is a point. Turning a vector by θ gives the same new coordinates as turning the axes the other way.
