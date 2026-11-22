---
tags: [machine-learning, NLP, embedding, word2vec, training-trick, sampling]
status: growing
level: 4
tier: 2
---

# Negative Sampling

Negative sampling is a training shortcut: instead of checking a guess against **every** wrong answer, you check it against **a few randomly picked wrong answers**.

## 🟢 1. Simple (age 12)
Suppose you are teaching a dog to fetch a ball. To be sure it knows the ball, you could show it every object in the house and say "not this, not this, not this...". That takes forever. A faster way: show the ball ("yes!"), then show just **a few random other things** like a sock or a spoon ("no, not these"). The dog still learns the ball very well, and you save hours.

**Example:** Learning that "ice" goes with "cream", you do not need to prove that "ice" does *not* go with each of 100,000 other words. Check "cream" (yes) and five random words like "tractor" and "Tuesday" (no). Good enough.

## 🔵 2. How it works
This was invented to speed up [[Word2Vec]], and the idea is used well beyond it.
1. In the full method, to learn from one real pair (centre word, neighbour word), the model must compare the neighbour against the *whole vocabulary* with a [[Softmax]]. That means scoring perhaps hundreds of thousands of words, for every single pair.
2. Negative sampling replaces that with a mini guessing game: *"is this pair real or fake?"* (a yes/no question, as in [[Logistic Regression]]).
3. For each **real** pair, make a few **fake** pairs. Keep the same centre word, but swap in $K$ random words as the "neighbour". These fakes are the **negative samples**.
4. Train so that the real pair scores high and the fake pairs score low (see the loss in [[Word2Vec]]).
5. Only the real word and the $K$ fakes are updated each step, instead of the whole vocabulary.

How are the fakes picked? **Not evenly, and not by raw frequency.** Common words are picked more often than rare ones, but the choice is softened so that rare words still get a fair chance.

No picture here: I did not find a freely licensed Wikimedia Commons image that shows negative sampling.

## 🟠 3. Technical detail
- **Why it is cheaper:** a step touches $K+1$ vectors instead of the whole vocabulary of size $V$. Typical $K$ is a handful (the original paper suggests about 5 to 20 for small data and 2 to 5 for large data; from memory).
- **Not the same maths as softmax:** negative sampling does not compute proper probabilities over all words. It is a different, cheaper training goal that still gives good vectors (from memory, the original authors describe it as a simplified form of noise contrastive estimation).
- **Sampling distribution:** words are drawn with probability proportional to their frequency raised to the power $3/4$. Raising to a power below 1 squeezes the gap between common and rare words. The $3/4$ value was found by trying (from memory, as reported in the Word2Vec paper).
- **Fakes can be accidentally real:** a randomly drawn word might in truth be a neighbour. This is rare in large vocabularies and is accepted as a little noise.
- **The same trick elsewhere:** the family of "contrast the real thing with random fakes" training shows up in [[Self-Supervised Learning]], in recommender systems, and in contrastive learning of [[Embedding]] models (stated from memory; details differ).
- **A related word-vector method** that skips sampling by working from a count table is [[GloVe]].

```python
import numpy as np
counts = np.array([100, 10, 1.])                       # a very common, a medium and a rare word
raw = counts / counts.sum(); q = counts ** 0.75; q /= q.sum()
print(raw.round(3), q.round(3))                        # [0.901 0.09  0.009] [0.827 0.147 0.026]
rng = np.random.default_rng(0)
draws = rng.choice(3, size=100000, p=q); print((np.bincount(draws) / 100000).round(2))   # [0.83 0.14 0.03]
sg = lambda x: 1 / (1 + np.exp(-x))
v = np.array([1., .5]); up = np.array([.8, .4]); un = np.array([-.2, .6])    # centre, real context, fake context
loss = lambda: -np.log(sg(up @ v)) - np.log(sg(-(un @ v)))
print(round(up @ v, 3), round(un @ v, 3), round(loss(), 3))   # 1.0 0.1 1.058   scores and loss before
gp, gn = sg(up @ v) - 1, sg(un @ v)                    # dLoss/dscore for the real and the fake pair
print(round(gp, 3), round(gn, 3))                      # -0.269 0.525
lr = 0.5
dv = gp * up + gn * un
up = up - lr * gp * v; un = un - lr * gn * v; v = v - lr * dv
print(round(up @ v, 3), round(un @ v, 3), round(loss(), 3))   # 1.269 -0.351 0.781   after one step
```

The first line shows the effect of the power $3/4$: the rare word is drawn 2.6% of the time instead of 0.9%, and the very common word 82.7% instead of 90.1%. The next lines do one training step by hand: the real pair's score goes up (1.0 to 1.269), the fake pair's score goes down (0.1 to -0.351), and the loss falls from 1.058 to 0.781.

## 🔴 4. Deep / formal
**Choosing the fakes.** If word $w$ appears with relative frequency $f(w)$ in the text, a fake is drawn with chance:

$$
P_n(w)=\frac{f(w)^{3/4}}{\sum_{u=1}^{V} f(u)^{3/4}}
$$

**Where:**
- $w$ is a word, and $V$ is the number of words in the vocabulary. $u$ just runs over all of them in the sum.
- $f(w)$ is how common word $w$ is in the text (its count divided by the total, or simply the count, since the division cancels).
- The power $3/4$ squeezes frequencies: common words are still picked more, but not as much more.
- The sum on the bottom makes all the chances add up to 1.
- **Matters:** with power 1, the fakes would be mostly "the", "of", "and", and the model would learn little. With power 0 (all words equally likely), rare junk words would be sampled as often as useful ones, wasting steps on words that hardly matter. The power in between is a compromise found by trying.

**Example (tiny numbers):** two words with relative frequencies $f=(0.9,\,0.1)$.
- Raise to $3/4$: $0.9^{0.75}=0.924$ and $0.1^{0.75}=0.178$. Sum: $1.102$.
- Chances: $0.924/1.102=\mathbf{0.839}$ and $0.178/1.102=\mathbf{0.161}$.
- Meaning: the common word is 9 times more frequent in the text, but only about 5 times more likely to be drawn as a fake ($0.839/0.161=5.2$). The rare word gets a bigger share of the attention.

**The learning signal.** Let $s$ be the score (dot product) of a pair, and $\sigma$ the sigmoid. The loss gradient with respect to the score is:

$$
\frac{\partial\mathcal L}{\partial s_{\text{real}}}=\sigma(s_{\text{real}})-1,\qquad
\frac{\partial\mathcal L}{\partial s_{\text{fake}}}=\sigma(s_{\text{fake}})
$$

**Where:**
- $\mathcal L$ is the loss for one real pair plus its fakes (defined in [[Word2Vec]]).
- $s_{\text{real}}$ and $s_{\text{fake}}$ are the dot-product scores of the real pair and of one fake pair.
- $\sigma(s)=1/(1+e^{-s})$ is the sigmoid, between 0 and 1. For the real pair the gradient is negative (so a downhill step *raises* the score), and for the fake pair it is positive (so a step *lowers* the score).
- **Matters:** the size shows how wrong the model is. A real pair already scored high ($\sigma\approx1$) gives a gradient near $0$, so there is little left to learn. A fake pair already scored very low gives $\sigma\approx0$, so the same. The updates go where the model is making mistakes. It is the same neat "$p-y$" form as in [[Logistic Regression]].

**Example (tiny numbers):** real pair score $s=1.0$, fake pair score $s=0.1$.
- Real: $\sigma(1.0)=0.731$, so the gradient is $0.731-1=\mathbf{-0.269}$.
- Fake: $\sigma(0.1)=\mathbf{0.525}$.
- Meaning: the real pair gets a push up of size $0.269$, and the fake pair gets a bigger push down (0.525) because it is scored too high for something fake. This matches the code, where one step lifted the real score to $1.269$ and dropped the fake to $-0.351$.

**Further notes:**
- **Relation to a binary classifier:** each real-or-fake decision is logistic regression on the score. The whole method is many tiny logistic regressions sharing word vectors.
- **Link to noise contrastive estimation:** from memory, negative sampling is a simplification of that older idea, and it drops the correction that would make the outputs true probabilities.
- **Hidden effect on the vectors:** a later analysis (Levy and Goldberg, 2014; from memory) showed that with negative sampling, the dot product of word and context vectors approaches a shifted measure of how often the words occur together (pointwise mutual information).

**What came before, and how this improved on it:**
- **Before:** a full softmax over the whole vocabulary for every training pair.
- **Its limit:** with 50,000 words, that is 50,000 scores per pair.
- **What changed:** score the true word and only a handful of random wrong words. Mikolov and colleagues used it in [[Word2Vec 2013]] (1310.4546; from memory, not read here). The wrong words are drawn from the word frequency raised to the power 0.75 (also from memory).
- **How much better:** in the code below, 5 negatives need 6 scores instead of 50,000, which is 8,333 times fewer. With made-up Zipf-like counts, the most common word is drawn with probability 0.1021 by raw frequency and 0.0273 with the 0.75 power, while the rarest 5,000 words together go from 0.0710 to 0.1744.

**Check it yourself:**

```python
import numpy as np
V = 50000
print(f"full softmax: {V:,} scores per training pair; negative sampling with 5 negatives: {1 + 5} scores ({V // 6:,}x fewer)")
rng = np.random.default_rng(0)
counts = (1e6 / np.arange(1, 10001)).astype(int) + 1                    # word counts that follow a Zipf-like curve
raw = counts / counts.sum(); soft = counts ** 0.75; soft = soft / soft.sum()
print("chance of drawing the most common word as a negative: raw frequency", round(raw[0], 4), " frequency^0.75", round(soft[0], 4))
print("chance of drawing a word from the rarest 5,000:        raw frequency", round(raw[5000:].sum(), 4), " frequency^0.75", round(soft[5000:].sum(), 4))
```

## Related
- [[Word2Vec]] · [[Embedding]] · [[Softmax]] · [[Logistic Regression]] · [[Self-Supervised Learning]] · [[Loss Function]] · [[GloVe]] · [[Entropy]] · [[Probability]]

## Sources
- Mikolov, Sutskever, Chen, Corrado & Dean, *Distributed Representations of Words and Phrases and their Compositionality*, 2013 (from memory)
- Gutmann & Hyvärinen, *Noise-contrastive estimation*, 2010 (from memory)
- Levy & Goldberg, *Neural Word Embedding as Implicit Matrix Factorization*, 2014 (from memory)
- Zhang, Lipton, Li & Smola, *Dive into Deep Learning* (free online), the chapter on approximate training for word2vec

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/transformers-nlp Negative sampling:::Instead of checking a real pair against every word in the vocabulary, show the real pair ("yes!") and a few random fake pairs ("no").

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write the noise distribution for choosing fake words and explain each symbol.
?
$$P_n(w)=\frac{f(w)^{3/4}}{\sum_{u=1}^{V} f(u)^{3/4}}$$
- $w$: a word. $V$: the number of words in the vocabulary. $u$ runs over all of them in the sum.
- $f(w)$: how common word $w$ is in the text (its count, or count divided by the total, since the division cancels).
- The power $3/4$ squeezes frequencies: common words are still picked more, but not as much more.
- The sum at the bottom makes the chances add to 1.
- With power 1 the fakes would be mostly "the", "of", "and" and the model would learn little. With power 0 rare junk words would be drawn too often.

#flashcards/ml/example #flashcards/topic/transformers-nlp Two words with frequencies 0.9 and 0.1. What are the draw chances?::$0.9^{0.75}=0.924$, $0.1^{0.75}=0.178$, sum 1.102. Chances 0.839 and 0.161. The common word is 9 times more frequent but only about 5.2 times more likely to be drawn.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write the gradients of the negative-sampling loss and explain each symbol.
?
$$\frac{\partial\mathcal L}{\partial s_{\text{real}}}=\sigma(s_{\text{real}})-1,\qquad \frac{\partial\mathcal L}{\partial s_{\text{fake}}}=\sigma(s_{\text{fake}})$$
- $\mathcal L$: the loss for one real pair plus its fakes (as in Word2Vec).
- $s_{\text{real}}$, $s_{\text{fake}}$: the dot-product scores of the real pair and of one fake pair.
- $\sigma(s)$: the sigmoid, between 0 and 1.
- For the real pair the gradient is negative, so a downhill step raises the score. For the fake pair it is positive, so a step lowers the score.
- The size shows how wrong the model is. A real pair already scored high ($\sigma\approx1$) gives a gradient near 0.

#flashcards/ml/example #flashcards/topic/transformers-nlp Real score 1.0, fake score 0.1. What are the gradients?::Real: $\sigma(1)-1=0.731-1=-0.269$. Fake: $\sigma(0.1)=0.525$. The real pair gets a push up of 0.269 and the fake pair a bigger push down of 0.525. In the note's code one step lifted the real score to 1.269 and dropped the fake to −0.351.
