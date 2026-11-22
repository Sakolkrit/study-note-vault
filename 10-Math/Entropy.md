---
tags: [math, information-theory, probability, machine-learning, loss]
status: growing
level: 4
tier: 2
---

# Entropy

Entropy is a number that measures **how surprised you are, on average, by an outcome**. It is high when the outcome is hard to guess and zero when it is certain.

## 🟢 1. Simple (age 12)
Think of guessing games. If I tell you "I flipped a coin", you have no idea what happened, so you are very unsure. If I tell you "I rolled a die that always lands on 6", you are never unsure, because you already know. **Entropy is the score for how unsure you are.** More unsure means a bigger score.

**Example:** Two weather towns. In Sunnyville it is sunny 99 days out of 100, so the weather report is boring and you can nearly always guess. In Mixville it is sunny or rainy half the time each, so you cannot guess. Mixville has the higher entropy.

## 🔵 2. How it works
1. Each possible outcome has a chance, its [[Probability]]. The chances add up to 1.
2. A rare outcome is a big surprise. A common one is a small surprise. The surprise of one outcome is $-\log(\text{chance})$, so a chance of 1 gives no surprise at all.
3. **Entropy is the average surprise**: add up each outcome's surprise, weighted by how often it happens.
4. Using $\log_2$ the answer is in **bits**: roughly "how many yes/no questions you need on average to find the answer". A fair coin needs 1 question, so it is 1 bit.
5. **Cross-entropy** is a close cousin. It asks: if the world works one way, but you *believe* another way, what is your average surprise? You are always at least as surprised as the true entropy, and more when your beliefs are wrong. That is why it is used as the [[Loss Function]] for classifiers.

![[entropy-binary-curve.png]]
*Entropy of a yes/no event with chance p of "yes" (labels are in Galician: "Probabilidade p" is the chance, "Entropía" is the entropy in bits). It is 0 when p is 0 or 1 (certain) and tops out at 1 bit at p = 1/2 (a fair coin). Image: Juanciño, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Entrop%C3%ADa_binaria_e_compresi%C3%B3n.png), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Origin:** introduced by Claude Shannon in 1948 for sending messages (the name and date are from memory). The ideas now sit under [[Machine Learning]], compression and [[Decision Tree]] splitting.
- **Bits and nats:** with $\log_2$ the unit is bits. With the natural log $\ln$ it is nats. ML code usually uses nats. Only the unit changes.
- **Maximum:** for $n$ outcomes, entropy is biggest when they are all equally likely, and then it equals $\log_2 n$ bits.
- **Decision trees:** a [[Decision Tree]] picks the split that lowers entropy the most (called information gain).
- **Cross-entropy loss:** in [[Classification]], the true answer is a one-hot list (all weight on one class) and the model gives a list of probabilities (see [[Softmax]]). Cross-entropy then becomes $-\log$ of the probability given to the right class. This is the same number as the log loss in [[Logistic Regression]]: 0.357 for a 70% guess on the right class, and 2.303 for a 10% guess.
- **Language models:** an [[LLM]] is trained to lower the cross-entropy of the next word. [[Perplexity]] is just this number put through an exponential.
- **Link to KL:** cross-entropy equals entropy plus the gap between the two beliefs. That gap is the [[KL Divergence]].

```python
import numpy as np
def H(p):                                    # entropy in bits
    p = np.array(p, float); p = p[p > 0]; return -(p * np.log2(p)).sum()
def CE(p, q):                                # cross-entropy in nats: true p, model q
    p, q = np.array(p, float), np.array(q, float); return -(p * np.log(q)).sum()

print(round(H([.5, .5]), 3), round(H([.9, .1]), 3), round(H([.25] * 4), 3))   # 1.0 0.469 2.0
print(round(H([1/3] * 3), 3), round(H([.36, .48, .16]), 3))                    # 1.585 1.462
print(round(CE([1, 0], [.7, .3]), 3), round(CE([1, 0], [.1, .9]), 3))          # 0.357 2.303
```

The first line shows a fair coin is 1 bit, a lopsided coin (90/10) is much less (0.469), and four equal outcomes need 2 bits. The second shows that uneven chances (0.36, 0.48, 0.16) have less entropy (1.462) than three equal ones (1.585). The third line matches the log-loss numbers in [[Logistic Regression]].

## 🔴 4. Deep / formal
**Entropy.** For outcomes with probabilities $p_1,\dots,p_n$:

$$
H(p)=-\sum_{i=1}^{n}p_i\log_2 p_i
$$

**Where:**
- $p_i$ is the chance of outcome $i$. All $p_i$ are between 0 and 1 and they add up to 1. ($0\log 0$ is taken as $0$.)
- $-\log_2 p_i$ is the surprise of outcome $i$, in bits. It is $0$ when $p_i=1$ and grows as $p_i$ shrinks.
- $\sum_i p_i(\cdot)$ is the average of that surprise, weighted by how often each outcome happens.
- **Matters:** the weighting by $p_i$ is what stops rare outcomes from dominating. Without the minus sign, the answer would be negative, since $\log$ of a chance is negative.

**Example (tiny numbers):** a biased coin with $p=(0.9,\,0.1)$.
- Surprise of heads: $-\log_2 0.9=0.152$ bits. Surprise of tails: $-\log_2 0.1=3.322$ bits.
- Weighted: $0.9\times0.152+0.1\times3.322=0.137+0.332$.
- $H=\mathbf{0.469}$ bits.
- Meaning: each flip carries less than half a bit of news, because heads is so expected. A fair coin carries a full 1 bit.

**Cross-entropy.** Let $p$ be the true chances and $q$ the chances your model believes:

$$
H(p,q)=-\sum_{i=1}^{n}p_i\log q_i
$$

**Where:**
- $p_i$ is the true chance of outcome $i$ (in classification, 1 for the right class and 0 for the others).
- $q_i$ is the chance your model gave to outcome $i$. $\log$ here is natural, so the unit is nats.
- The sum averages your *model's* surprise, $-\log q_i$, over what *really* happens (weights $p_i$).
- **Matters:** if $q_i$ is tiny for something that actually happens, $-\log q_i$ is huge, so confident wrong answers cost a lot. That is what pushes a model to put honest probability on the truth.

**Example (tiny numbers):** the true class is the first one, so $p=(1,0)$. The model says $q=(0.7,\,0.3)$.
- Only the first term counts: $-1\times\ln 0.7$, and the second term is $0\times\ln 0.3=0$.
- $H(p,q)=\mathbf{0.357}$ nats.
- Meaning: a 70% belief in the right class costs 0.357. If the model had said $q=(0.1,0.9)$, the cost would be $-\ln0.1=2.303$, about six times worse.

**Further notes:**
- **Never below entropy:** $H(p,q)\ge H(p)$, with equality only when $q=p$ (shown through [[KL Divergence]]).
- **Meaning of bits:** the entropy is, roughly, the best average message length per outcome when you encode with the best possible code (the source coding result; stated from memory). Cross-entropy is the average length if you use a code built for the wrong chances $q$.
- **Continuous values:** for numbers on a continuum, the sum becomes an integral (differential entropy), and it can be negative, so it behaves a little differently. Not covered here.
- **Maximum-likelihood link:** minimising cross-entropy on data is the same as maximising the likelihood of the data (see [[Bayesian Methods]]).

**What came before, and how this improved on it:**
- **Before:** fixed-length codes, such as one byte (8 bits) per character, whatever the character.
- **Its limit:** common characters and rare characters cost the same, so space is wasted.
- **What changed:** Shannon (1948; from memory) defined the average information in a message, the entropy, which is the least average length any code can reach. Huffman (1952; from memory) gave a simple way to build a code that gets very close.
- **How much better:** in the code below, about 480,000 characters of this vault's text use 85 different characters. A fixed code needs 7 bits at the very least (8 as a byte). The entropy is 4.875 bits per character, and a Huffman code averages 4.913, within 0.04 bits of the limit. That is about 1.6 times smaller than a byte per character. The numbers shift a little as the vault text changes.

**Check it yourself:**

```python
import glob, heapq, math, collections
fs = sorted(glob.glob(r"D:/python456/study&note stuff/20-ML-LLM/*.md"))
text = " ".join(open(f, encoding="utf-8").read() for f in fs[:40]).lower()
cnt = collections.Counter(text); n = len(text)
H = -sum(c / n * math.log2(c / n) for c in cnt.values())
h = [[c, i, [ch]] for i, (ch, c) in enumerate(cnt.items())]; heapq.heapify(h); depth = collections.Counter(); k = len(h)
while len(h) > 1:
    a, b = heapq.heappop(h), heapq.heappop(h)
    for ch in a[2] + b[2]: depth[ch] += 1
    heapq.heappush(h, [a[0] + b[0], k, a[2] + b[2]]); k += 1
huff = sum(cnt[c] * depth[c] for c in cnt) / n
print(len(cnt), "different characters,", n, "characters of text")
print("fixed-length code: 8 bits per character (one byte);  minimum possible length, ceil(log2(", len(cnt), ")) =", math.ceil(math.log2(len(cnt))))
print("Shannon entropy:", round(H, 3), "bits per character;  Huffman code average:", round(huff, 3))
```

## Related
- [[Probability]] · [[KL Divergence]] · [[Loss Function]] · [[Softmax]] · [[Logistic Regression]] · [[Classification]] · [[Decision Tree]] · [[Perplexity]] · [[LLM]] · [[Machine Learning]] · [[Bayesian Methods]] · [[Maximum Likelihood]] · [[Pre-training]] · [[GAN]] · [[Greedy Algorithms]] · [[Cross-Entropy Loss]] · [[Mutual Information]]

## Sources
- Shannon, *A Mathematical Theory of Communication*, 1948 (from memory)
- Cover & Thomas, *Elements of Information Theory* (from memory)
- MacKay, *Information Theory, Inference, and Learning Algorithms* (free online; from memory)
- Goodfellow, Bengio & Courville, *Deep Learning*, chapter 3 (free online)

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/information-theory Entropy:::A score for how unsure you are about an outcome. More unsure means a bigger score. It is 0 when the outcome is certain.

#flashcards/math/formula #flashcards/topic/information-theory Write the entropy formula and explain each symbol.
?
$$H(p)=-\sum_{i=1}^{n}p_i\log_2 p_i$$
- $p_i$: the chance of outcome $i$. All are between 0 and 1 and add up to 1 ($0\log 0$ counts as 0).
- $-\log_2 p_i$: the surprise of outcome $i$ in bits. 0 when $p_i=1$, and it grows as $p_i$ shrinks.
- $\sum_i p_i(\cdot)$: the average of that surprise, weighted by how often each outcome happens. The weighting stops rare outcomes from dominating.
- The minus sign keeps the answer positive, since $\log$ of a chance is negative.

#flashcards/math/formula #flashcards/topic/information-theory Write the cross-entropy formula and explain each symbol.
?
$$H(p,q)=-\sum_{i=1}^{n}p_i\log q_i$$
- $p_i$: the true chance of outcome $i$ (in classification, 1 for the right class and 0 for the others).
- $q_i$: the chance your model gave to outcome $i$. $\log$ is natural here, so the unit is nats.
- The sum averages your model's surprise $-\log q_i$ over what really happens.
- If $q_i$ is tiny for something that happens, $-\log q_i$ is huge, so confident wrong answers cost a lot.

#flashcards/math/cloze #flashcards/topic/information-theory A fair coin has ==1 bit== of entropy. Entropy is ==0== when the outcome is certain.

#flashcards/math/example #flashcards/topic/information-theory A biased coin has $p=(0.9,0.1)$. What is its entropy?::Surprise of heads 0.152 bits, of tails 3.322 bits. 0.9 × 0.152 + 0.1 × 3.322 = 0.469 bits. Less than half a bit of news per flip, because heads is so expected.

#flashcards/math/example #flashcards/topic/information-theory The true class is the first one ($p=(1,0)$). What is the cross-entropy if the model says $q=(0.7,0.3)$, and if it says $q=(0.1,0.9)$?::Only the first term counts: −ln 0.7 = 0.357 nats. For q = (0.1, 0.9): −ln 0.1 = 2.303 nats. A confident wrong answer costs far more.

#flashcards/math/image #flashcards/topic/information-theory ![[entropy-binary-curve.png]]
What does this curve show?
?
Entropy of a yes/no event with chance p of "yes". It is 0 when p is 0 or 1 (certain) and tops out at 1 bit at p = 1/2 (a fair coin). (Labels are in Galician.)
