---
tags: [LLM, generation, autoregressive, language-model, inference, probability]
status: growing
level: 4
tier: 3
---

# Autoregressive Generation

Writing text **one token at a time, left to right**, where each new token is chosen using everything written so far. This is how almost every chatbot ([[LLM]]) writes.

## 🟢 1. Simple (age 12)
Think of a story told around a campfire where each person adds one word and everyone must remember what was said before. You can never skip ahead, and you cannot take a word back once it is said. Each next word is a good guess given the words before it.

**Example:** Your phone's keyboard suggests the next word. You tap one, it suggests another, and a whole sentence appears. A chatbot does the same thing, but with a far better guesser and a much longer memory of what came before.

## 🔵 2. How it works
1. Text is cut into small pieces called tokens ([[Tokenization]]). A text is a list of tokens: $x_1, x_2, \dots, x_n$.
2. A trained model reads the tokens so far and gives a **score for every possible next token**. A [[Softmax]] turns the scores into probabilities that add up to 1.
3. One token is chosen. How it is chosen (always the top one, or a random pick weighted by the probabilities) is the job of [[Decoding Strategies]], for example [[Beam Search]].
4. The chosen token is added to the end of the text, and steps 2 and 3 repeat, until the model writes a special stop token.
5. So a reply of 500 tokens needs 500 model runs, one after another. Each run has to wait for the one before it, because it needs the token that run chose. That is why long answers appear word by word. ([[LLM Inference]] shows what each run costs.)
6. **Training is not like this.** During training the real text is already known, so the model can check its guess at *every* position at once: "given the first 3 tokens, how likely was the true 4th?", "given the first 4, how likely was the 5th?", and so on, all side by side. A mask stops each position from peeking at later tokens ([[Self-Attention]]). This is why training can use many chips at once, while writing cannot.
7. Why it is useful: one simple job (guess the next token) covers translating, answering, and coding, because all of them are text that continues from a prompt ([[Pre-training]]).

## 🟠 3. Technical detail
- **Chain rule.** The probability of a whole text is split into one next-token guess per position (the formula is in level 4). Nothing is approximated: any distribution over texts can be written this way. The only choice is how well the model learns each factor.
- **Training signal.** The loss is how surprised the model was by the true next token, averaged over positions ([[Cross-Entropy Loss]], [[Loss Function]]). Lower loss means a higher probability for real text, and the usual score for that is [[Perplexity]].
- **Teacher forcing.** In training the model is always fed the *true* earlier tokens, not its own guesses. This makes all positions independent of each other and therefore parallel.
- **Exposure gap.** When writing, the model is fed its *own* earlier tokens. A bad early token stays in the text and every later token must continue from it. This mismatch between training and writing is often called exposure bias (standard term, **from memory**, not sourced here).
- **Speed.** $n$ tokens need $n$ runs in a row. The [[KV Cache]] keeps the earlier work so each run does not start from scratch, [[Speculative Decoding]] checks several guessed tokens in one run, and [[Continuous Batching]] fills the machine with other users while one waits.
- **Order.** The order is fixed left to right. The model cannot first write the ending and then fill in the middle. [[Masked Diffusion Language Model]]s drop this order and reveal several tokens per run, as [[BERT]]-style models can, at the cost of other trade-offs.
- **The code below is a toy.** A rule turns the text so far into 3 probabilities (it stands in for a trained model, with random numbers instead of learning). It checks the chain rule, counts model runs, and shows that writing token by token gives texts at the frequencies the chain rule predicts.

```python
import itertools
from collections import Counter

import numpy as np

V, n = 3, 3                                    # a vocabulary of 3 tokens (a, b, c) and texts of 3 tokens


def next_probs(prefix):
    """Stand-in for a trained model: any rule that turns the text so far into 3 probabilities."""
    seed = 0
    for t in prefix:
        seed = seed * V + t + 1
    p = np.exp(np.random.RandomState(seed).randn(V))
    return p / p.sum()


def prob(text):
    """Chain rule: probability of the whole text = product of next-token probabilities."""
    p = 1.0
    for t, tok in enumerate(text):
        p *= next_probs(text[:t])[tok]
    return p


texts = list(itertools.product(range(V), repeat=n))
print(f"all {len(texts)} possible texts: probabilities add up to {sum(prob(x) for x in texts):.6f}")

x = (0, 2, 1)                                  # the text "a c b"
steps = [next_probs(x[:t])[x[t]] for t in range(n)]
print("steps for a c b:", [round(float(s), 3) for s in steps], "product", round(float(np.prod(steps)), 4), "= prob()", round(prob(x), 4))
print("sum of logs", round(float(np.sum(np.log(steps))), 4), "= log of product", round(float(np.log(prob(x))), 4))

calls = 0


def write(rng):
    """Writing: each token needs the one before it, so the n calls cannot run side by side."""
    global calls
    text = []
    for _ in range(n):
        calls += 1
        text.append(int(rng.choice(V, p=next_probs(text))))
    return tuple(text)


rng, N = np.random.RandomState(1), 20000
seen = Counter(write(rng) for _ in range(N))
print("model calls per text written:", calls / N)
print("largest gap between how often a text was written and its chain-rule probability:", round(max(abs(seen[t] / N - prob(t)) for t in texts), 4))

print(f"contexts a table must hold for 5 words and 50,000 word types: {50_000 ** 5:.2e}")
```

## 🔴 4. Deep / formal
A text $x_1,\dots,x_n$ gets its probability from the chain rule, and the training score is the log of it:

$$
\log p(x_1,\dots,x_n)=\sum_{t=1}^{n}\log p\left(x_t\mid x_1,\dots,x_{t-1}\right)
$$

**Where:**
- $x_t$ is the token at position $t$, and $n$ the length of the text.
- $p(x_t\mid x_1,\dots,x_{t-1})$ is the model's probability for the true token at position $t$, given only the tokens before it. This is one number from the model's softmax at position $t$.
- $\log$ turns the long product into a sum, so many small probabilities do not shrink to zero on a computer.
- **Matters:** the "given only the tokens before it" part is the rule that makes this a valid probability. If a factor could look at later tokens, the product would no longer add up to 1 over all texts and the model could not be used to write. The sum is also the reason training can be parallel: with the true text known, all $n$ terms can be computed at the same time.

**Example (tiny numbers):** the text "a b c" with $p(a)=0.5$, $p(b\mid a)=0.4$, $p(c\mid a\,b)=0.5$.
- Product: $0.5\times0.4\times0.5=\mathbf{0.1}$.
- Logs: $\ln0.5+\ln0.4+\ln0.5=-0.693-0.916-0.693=\mathbf{-2.303}$, which is $\ln0.1$.
- Meaning: the model thought this text had a 1-in-10 chance. Training pushes this number up for real text. Writing a different text means choosing different tokens at some step, and each choice changes all the later factors.

**Further notes (from the code: 3 tokens, texts of 3 tokens, 27 possible texts):**
- The probabilities of all 27 texts added up to 1.000000, so the chain rule gives a proper distribution for any next-token rule.
- For the text "a c b" the three steps were 0.584, 0.095 and 0.512; their product 0.0284 equalled the direct computation, and the sum of logs ($-3.5607$) equalled the log of the product.
- Writing 20,000 texts token by token took exactly 3.0 model calls per text. The most any text's written frequency differed from its chain-rule probability was 0.0025, which fits ordinary sampling noise for 20,000 samples.
- **What the toy cannot show:** real quality, how the model is learned, the cost of each call, or the training/writing mismatch. The "model" is a random rule.
- **Limits of the method itself:** the order is fixed; a wrong early token cannot be revised; $n$ tokens need $n$ runs in sequence; and errors can grow along the text. How big these effects are depends on the model and is not measured here. The claim that this makes models "hallucinate" ([[Hallucination]]) is common but I did not test it.

**What came before, and how this improved on it:**
- **Before:** *n-gram* models, which also use the chain rule but look only at the last few words and keep a table of counts. Neural models then replaced the counts: Bengio et al., *A Neural Probabilistic Language Model* (Journal of Machine Learning Research 3, 2003, pages 1137 to 1155; abstract read) learned word vectors and the probability function together. Later recurrent networks ([[Recurrent Neural Network]]) and then the [[Transformer]] ([[Attention Is All You Need 2017]]) were used as the next-token guesser.
- **Its limit:** a table over the last $k$ words needs one row for each possible context, and almost all contexts never appear in training text, so the table has to guess for contexts it has never seen. The Bengio abstract names this as the problem: test sequences often differ from every training sequence. Recurrent networks read tokens one after another even in training, which makes training slow.
- **What changed:** a neural network scores the next token from learned word vectors, so similar contexts share what was learned. With the Transformer, training uses all positions in parallel (the mask in step 6 of level 2). GPT-3 then scaled this recipe to 175 billion parameters ([[GPT-3 2020]]; abstract as recorded in that note).
- **How much better:** in this note's code, a count table for 5 words of context and 50,000 word types would need $3.13\times10^{23}$ rows (arithmetic, not a trained system), while a neural model has a fixed number of parameters. The Bengio abstract reports that on two corpora its model clearly beat the best n-gram models and could use longer context, with **no figure in the abstract**. The Transformer paper's figures are in [[Attention Is All You Need 2017]]. The cost that remains is in writing: $n$ tokens need $n$ runs in a row.

## Related
- [[LLM]] · [[LLM Inference]] · [[Decoding Strategies]] · [[Beam Search]] · [[KV Cache]] · [[Speculative Decoding]] · [[Masked Diffusion Language Model]] · [[Generative Models]] · [[Transformer]] · [[Self-Attention]] · [[Perplexity]] · [[Cross-Entropy Loss]] · [[Pre-training]] · [[Tokenization]] · [[Softmax]] · [[Hallucination]]

## Sources
- Bengio, Ducharme, Vincent, Jauvin, *A Neural Probabilistic Language Model*, Journal of Machine Learning Research 3 (2003), pages 1137 to 1155 (journal page; abstract read, paper body not read)
- Vaswani et al. 2017 and Brown et al. 2020, as recorded in [[Attention Is All You Need 2017]] and [[GPT-3 2020]] (abstracts read there)
- "Teacher forcing", "exposure bias", and the claim that a causal mask lets training run in parallel are standard descriptions, **from memory, not re-read in a source here**. The code checks only the chain rule and the call count.
- Image: none. A Wikimedia Commons search found no free image on this topic (the closest were about proteins, old time-series reports, and image generation), so none is used.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/transformers-nlp Autoregressive generation:::Writing text one token at a time, where each new token is chosen given all the tokens before it, then added to the text so the next step can use it.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write the chain rule for the probability of a whole text and explain each symbol.
?
$$\log p(x_1,\dots,x_n)=\sum_{t=1}^{n}\log p\left(x_t\mid x_1,\dots,x_{t-1}\right)$$
- $x_t$: the token at position $t$. $n$: the length of the text.
- $p(x_t\mid x_1,\dots,x_{t-1})$: the model's probability for the true token at position $t$, given only the tokens before it (one number from the softmax at that position).
- $\log$: turns the long product into a sum, so many small probabilities do not shrink to zero on a computer.
- "Given only the tokens before it" is what makes this a valid probability over all texts.

#flashcards/ml/example #flashcards/topic/transformers-nlp Text "a b c" with $p(a)=0.5$, $p(b\mid a)=0.4$, $p(c\mid a\,b)=0.5$. What is the probability and log-probability of the text?::Product $0.5\times0.4\times0.5=0.1$. Logs: $-0.693-0.916-0.693=-2.303=\ln0.1$. The model gave this text a 1-in-10 chance.

#flashcards/ml/cloze #flashcards/topic/transformers-nlp A reply of 500 tokens needs ==500== model runs, one ==after== another, because each run needs the token the previous run chose.

#flashcards/ml/concept #flashcards/topic/transformers-nlp Why do long chatbot answers appear word by word?::Generation is autoregressive. Each token needs the token before it, so the model runs once per token, in order.
