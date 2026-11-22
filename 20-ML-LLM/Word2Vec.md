---
tags: [machine-learning, NLP, embedding, self-supervised, word-vectors, LLM]
status: growing
level: 4
tier: 2
---

# Word2Vec

Word2Vec is a method that turns each word into a short list of numbers (a [[Vector]]), so that words used in similar ways get similar numbers.

## 🟢 1. Simple (age 12)
Word2Vec learns what a word means by **looking at the words that hang around it**. If two words keep showing up next to the same neighbours, the computer decides they must be alike, and gives them similar number-codes.

**Example:** Imagine you have never seen a "wug". You read: "I fed the wug and walked the wug." Then you read: "I fed the dog and walked the dog." Wugs and dogs get fed and walked in the same way, so you guess a wug is something like a dog. Word2Vec does that guess with numbers, over millions of sentences.

## 🔵 2. How it works
1. Take a huge pile of text. Cut it into words (see [[Tokenization]]).
2. Give every word a random list of numbers to start with. This is its [[Embedding]].
3. Slide a small window along the text. In each window, one word is the **centre** and the words around it are its **context**.
4. Train a small [[Neural Network]] on a guessing game: *"given the centre word, which words are likely nearby?"* (This is the **skip-gram** version. The other version, **CBOW**, guesses the centre word from its neighbours.)
5. Each time the guess is wrong, nudge the numbers a little (see [[Stochastic Gradient Descent]]) so the right neighbours become more likely.
6. After millions of nudges, throw away the guessing game and keep the numbers. Words used alike now sit **close together**, and you can measure that with [[Cosine Similarity]].

![[skip-gram.svg]]
*Skip-gram: the centre word "loves" is used to predict the words around it ("the", "man", "his", "son"). Image: Zhang, Lipton, Li and Smola, from "Dive into Deep Learning", [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Skip-gram.svg), CC BY-SA 4.0.*

No one labelled the text. The neighbours are the answers, so this is [[Self-Supervised Learning]].

## 🟠 3. Technical detail
- **Two models:** skip-gram (centre → context words) and CBOW (context words → centre). Skip-gram tends to be better for rare words; CBOW is faster (this is the usual rule of thumb, stated from memory).
- **Two sets of vectors:** every word has a *centre* vector $\mathbf v$ and a *context* vector $\mathbf u$. People normally keep the centre vectors (some average the two).
- **The costly part:** the honest guess is a [[Softmax]] over the whole vocabulary, which can be hundreds of thousands of words, for every training pair. **Negative sampling** is the shortcut: instead of scoring every word, score the real neighbour against just a few random "fake" neighbours, and learn to tell them apart.
- **Famous trick:** the differences between vectors often carry meaning, for example *king − man + woman ≈ queen*. This works on large real corpora with some caveats. It does **not** appear in the tiny example below.
- **Limit:** one word gets one vector, whatever the sentence. "bank" (river) and "bank" (money) share a single vector. Later models ([[Transformer]], [[Attention]]) give each word a different vector in each sentence.
- **Why it matters today:** it showed that [[Embedding]] vectors can be learned from raw text. The input layer of an [[LLM]] is a learned embedding table, built on the same idea.

Below is a tiny skip-gram trained with full softmax on a toy 16-word text. It is meant to show the idea, not to be useful.

```python
import numpy as np
text = """the king rules the land . the queen rules the land .
the king loves the queen . the queen loves the king .
the cat eats fish . the dog eats meat . the cat chases the dog .
the dog chases the cat . the man walks the dog . the woman walks the cat .""".split()
vocab = sorted(set(text)); ix = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
pairs = [(ix[text[i]], ix[text[j]]) for i in range(len(text))
         for j in range(max(0, i - 2), min(len(text), i + 3)) if j != i]   # (centre, context), window 2
rng = np.random.default_rng(0)
Vc = rng.normal(0, 0.1, (V, 8))      # "centre" vectors  (the ones we keep)
Uo = rng.normal(0, 0.1, (V, 8))      # "context" vectors
lr = 0.1
for epoch in range(300):
    for c, o in pairs:
        s = Uo @ Vc[c]; p = np.exp(s - s.max()); p /= p.sum()   # softmax over the vocabulary
        g = p.copy(); g[o] -= 1                                  # d loss / d scores = p - onehot
        gv = Uo.T @ g; Uo -= lr * np.outer(g, Vc[c]); Vc[c] -= lr * gv
def cos(a, b):
    x, y = Vc[ix[a]], Vc[ix[b]]; return round(float(x @ y / np.linalg.norm(x) / np.linalg.norm(y)), 3)
print(V, len(pairs))   # 16 226
print(cos("king", "queen"), cos("cat", "dog"), cos("king", "cat"), cos("queen", "dog"), cos("man", "woman"))
# 0.998 0.801 0.59 0.562 0.996
```

Read the output like this. "king/queen" (0.998) and "man/woman" (0.996) appear in almost identical spots, so they end up nearly the same. "cat/dog" (0.801) share many contexts, so they are close. The unrelated pairs (0.59, 0.562) are lower, but not near zero. With only 226 training pairs and 8 numbers per word, the vectors are crude, and other random seeds shift the middle numbers (cat/dog ranged from about 0.80 to 0.85 in my runs, and the ordering held). Real Word2Vec uses billions of words.

## 🔴 4. Deep / formal
**The skip-gram guess.** For a centre word $c$ and a possible context word $o$, the model gives a probability using the dot product of their vectors:

$$
P(o\mid c)=\frac{\exp\!\big(\mathbf u_o^{\top}\mathbf v_c\big)}{\sum_{w=1}^{V}\exp\!\big(\mathbf u_w^{\top}\mathbf v_c\big)}
$$

**Where:**
- $c$ is the centre word and $o$ is a context word we want to predict. $V$ is the number of words in the vocabulary.
- $\mathbf v_c$ is the centre vector of word $c$. $\mathbf u_o$ is the context vector of word $o$. Both are lists of numbers the model learns.
- $\mathbf u_o^{\top}\mathbf v_c$ is the dot product: big when the two vectors point the same way. It is the "score" for the pair.
- $\exp$ makes every score positive. The sum over all $V$ words in the bottom makes the probabilities add up to 1. This is a [[Softmax]].
- **Matters:** the bottom sum is what makes the model fair, but it needs $V$ dot products for every single training pair. Without it, the model could cheat by giving every word a huge score.

**Example (tiny numbers):** a vocabulary of 3 words. The scores $\mathbf u_w^{\top}\mathbf v_c$ for the three words are $2,\,1,\,0$, and the real neighbour is the first word.
- $\exp$ of the scores: $7.389,\ 2.718,\ 1.000$. Their sum: $11.107$.
- Probabilities: $7.389/11.107=0.665$, $2.718/11.107=0.245$, $1.000/11.107=0.090$.
- Loss for the real neighbour: $-\log 0.665=\mathbf{0.408}$.
- Meaning: the model gave the true neighbour 66.5%, so the loss is smallish. Training pushes this probability towards 1 by raising the score of the true pair and lowering the others.

**Negative sampling.** To avoid the sum over the whole vocabulary, keep the real pair $(c,o)$ and draw $K$ random words $w_1,\dots,w_K$ as fake neighbours. The loss for this pair is:

$$
\mathcal L=-\log\sigma\!\big(\mathbf u_o^{\top}\mathbf v_c\big)-\sum_{k=1}^{K}\log\sigma\!\big(-\mathbf u_{w_k}^{\top}\mathbf v_c\big)
$$

**Where:**
- $\sigma(x)=1/(1+e^{-x})$ is the sigmoid, which squashes any score into a number between 0 and 1 (see [[Logistic Regression]]).
- $\mathbf u_o^{\top}\mathbf v_c$ is the score of the real pair. $\mathbf u_{w_k}^{\top}\mathbf v_c$ is the score of fake pair number $k$.
- $K$ is how many fake words we draw per real pair. Small values like 5 or so are typical for big corpora (from memory).
- The first term wants the real pair's score **high**. The minus sign inside the second term wants each fake pair's score **low**.
- **Matters:** each training step touches only $K+1$ context vectors instead of $V$, so it is far cheaper. Without the second term, the easiest way to win is to make every score huge, and all words collapse to the same vector.

**Example (tiny numbers):** the real pair has score $2$, and there is $K=1$ fake word with score $-1$.
- Real term: $-\log\sigma(2)=-\log 0.881=0.127$.
- Fake term: $-\log\sigma(-(-1))=-\log\sigma(1)=-\log 0.731=0.313$.
- Total: $0.127+0.313=\mathbf{0.440}$.
- Meaning: a good state, since real is scored high and fake low. For comparison, if the scores were swapped (real $-2$, fake $+1$) the loss would be $2.127+1.313=3.440$, about eight times bigger, so training would push hard to fix it.

**Further notes:**
- **Why vector arithmetic can work:** a related line of work (Levy and Goldberg, 2014; stated from memory) showed that skip-gram with negative sampling behaves like factorising a table of how often word pairs appear together (a shifted pointwise mutual information table). Words with similar rows get similar vectors.
- **Other word-vector methods:** GloVe fits the same kind of co-occurrence table directly. fastText adds pieces of words, so it can handle words it never saw.
- **Bias:** vectors learned from text copy the biases in that text (for example, occupation words leaning towards one gender), a well-documented problem.
- **From static to contextual:** a word gets one vector here. [[Transformer]] models compute a new vector for each word in each sentence.

**What came before, and how this improved on it:**
- **Before:** neural-network language models learned word vectors, but were slow on large data (Mikolov et al. compare against them; details from memory).
- **Its limit:** a full softmax over the vocabulary needs one multiplication with every output word for every training pair.
- **What changed:** two small architectures (CBOW and skip-gram) in 2013, then negative sampling and subsampling of frequent words (arXiv 1301.3781, 16 January 2013; arXiv 1310.4546, 16 October 2013; abstracts read).
- **How much better:** the first paper says it takes "less than a day" to learn high-quality vectors from 1.6 billion words, with large accuracy gains at much lower computational cost (abstract). The code below counts the output work for $V=50{,}000$ and $d=300$: $15{,}000{,}000$ multiplications per pair for a full softmax against $1{,}800$ with 5 negative samples, about 8,333 times fewer.

**Check it yourself:**

```python
V, d, k = 50_000, 300, 5
full = V * d          # multiply the vector by every output word
neg = (k + 1) * d     # 1 real word + k random ones
print("output work per training pair, full softmax:", full, " negative sampling (k=5):", neg, " ratio", round(full / neg, 1))
```

## Related
- [[Embedding]] · [[Embedding Model]] · [[Self-Supervised Learning]] · [[Vector]] · [[Cosine Similarity]] · [[Softmax]] · [[Logistic Regression]] · [[Neural Network]] · [[Stochastic Gradient Descent]] · [[Tokenization]] · [[Attention]] · [[Transformer]] · [[LLM]] · [[Negative Sampling]] · [[GloVe]] · [[Byte-Pair Encoding]] · [[BERT]] · [[Word2Vec 2013]]

## Sources
- Mikolov, Chen, Corrado & Dean, *Efficient Estimation of Word Representations in Vector Space*, 2013 (from memory)
- Mikolov, Sutskever, Chen, Corrado & Dean, *Distributed Representations of Words and Phrases and their Compositionality*, 2013 (from memory)
- Levy & Goldberg, *Neural Word Embedding as Implicit Matrix Factorization*, 2014 (from memory)
- Zhang, Lipton, Li & Smola, *Dive into Deep Learning* (free online), the chapter on word embedding (word2vec)
- Mikolov et al., arXiv 1301.3781 and arXiv 1310.4546 (abstracts read for the history block)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/transformers-nlp Word2Vec:::Learn what a word means from the words that hang around it. Words with the same neighbours get similar number-codes.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write the skip-gram softmax probability and explain each symbol.
?
$$P(o\mid c)=\frac{\exp\!\big(\mathbf u_o^{\top}\mathbf v_c\big)}{\sum_{w=1}^{V}\exp\!\big(\mathbf u_w^{\top}\mathbf v_c\big)}$$
- $c$: the centre word. $o$: a context word to predict. $V$: the number of words in the vocabulary.
- $\mathbf v_c$: the centre vector of word $c$. $\mathbf u_o$: the context vector of word $o$. Both are learned.
- $\mathbf u_o^{\top}\mathbf v_c$: the dot product, big when the vectors point the same way. It is the score for the pair.
- $\exp$ makes scores positive. The sum over all $V$ words makes the probabilities add to 1 (a softmax). That sum over the whole vocabulary is the expensive part.

#flashcards/ml/example #flashcards/topic/transformers-nlp Scores 2, 1, 0 for three words, the real neighbour is the first. What are the probabilities and the loss?::$\exp$: 7.389, 2.718, 1.000, sum 11.107. Probabilities 0.665, 0.245, 0.090. Loss $-\log0.665=0.408$. Training pushes the true pair's probability toward 1.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write the negative-sampling loss and explain each symbol.
?
$$\mathcal L=-\log\sigma\!\big(\mathbf u_o^{\top}\mathbf v_c\big)-\sum_{k=1}^{K}\log\sigma\!\big(-\mathbf u_{w_k}^{\top}\mathbf v_c\big)$$
- $\sigma(x)=1/(1+e^{-x})$: the sigmoid, squashing any score into 0 to 1.
- $\mathbf u_o^{\top}\mathbf v_c$: the score of the real pair. $\mathbf u_{w_k}^{\top}\mathbf v_c$: the score of fake pair $k$.
- $K$: how many fake words are drawn per real pair. Small values like 5 are typical for big corpora (from memory).
- The first term wants the real score high. The minus sign inside the second term wants each fake score low.

#flashcards/ml/example #flashcards/topic/transformers-nlp Real score 2, one fake with score $-1$. What is the loss, and what if the scores were swapped?::Real term $-\log\sigma(2)=0.127$. Fake term $-\log\sigma(1)=0.313$. Total 0.440. If real were $-2$ and fake $+1$ the loss would be $2.127+1.313=3.440$, about eight times bigger.

#flashcards/ml/image #flashcards/topic/transformers-nlp ![[skip-gram.svg]]
What does this picture show?
?
Skip-gram: the centre word "loves" is used to predict the words around it ("the", "man", "his", "son"). (From the note's caption: Zhang, Lipton, Li and Smola, Dive into Deep Learning, Wikimedia Commons.)

#flashcards/ml/cloze #flashcards/topic/transformers-nlp Skip-gram uses the ==centre word== to predict the ==context words== around it.
