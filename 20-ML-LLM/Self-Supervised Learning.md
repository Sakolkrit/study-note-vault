---
tags: [machine-learning, self-supervised-learning, LLM, pretraining, AI]
status: growing
level: 4
tier: 2
---

# Self-Supervised Learning

## 🟢 1. Simple (age 12)
Self-supervised learning is **making your own homework questions from a pile of text or pictures, and learning by answering them**. Nobody has to write an answer key. The data hides a piece of itself, and the computer tries to guess the hidden piece. Then it peeks at the real piece to see if it was right.

**Example:** You have a book. You cover one word in a sentence with your thumb: "The cat sat on the ___." You guess "mat", lift your thumb, and check. You can do this a million times with any book, and the book itself is the answer key. After enough practice you get very good at language, even though no teacher ever labelled anything.

## 🔵 2. How it works
1. Collect a huge pile of **raw, unlabelled** data, such as text from books and web pages.
2. **Hide part of it.** Two popular ways: cover the **next word** (what an [[LLM]] does), or cover a **word in the middle** (what BERT does, in the picture below).
3. Feed the visible part into a [[Neural Network]] (usually a [[Transformer]]). The text is first cut into pieces by [[Tokenization]].
4. The model guesses the hidden piece. A [[Loss Function]] measures how surprised it was by the real answer.
5. Nudge the model's numbers to be less surprised next time ([[Gradient Descent]] with [[Backpropagation]]).
6. Repeat over billions of examples. Because the data supplies its own answers, there is no limit set by how many labels people could write.
7. Afterwards, the model can be **adapted** to a real task with a small labelled set ([[Fine-tuning]]) or steered with a prompt ([[Prompt Engineering]]).

![[bert-masked-lm.png]]
*Masked prediction. The bottom row is the real sentence cut into tokens. One token ("##ica") is replaced by [MASK], the sentence goes through a Transformer encoder (BERT), and the model must output scores (logits) for what the hidden token was. The label comes from the sentence itself. Image: Daniel Voigt Godoy, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:BERT_masked_language_modelling_task.png), CC BY 4.0.*

For the cow example of the three kinds of learning side by side, see the picture in [[Supervised Learning]]. Self-supervised is the right-hand panel there.

## 🟠 3. Technical detail
- **The trick:** turn an unlabelled dataset into a labelled one by **inventing the task**. These made-up tasks are called **pretext tasks**.
- **Common pretext tasks:**
  - **Next-token prediction** (causal language modelling): predict each token from the ones before it. This is how GPT-style [[LLM]]s are trained.
  - **Masked prediction:** hide some tokens and predict them from both sides (BERT-style). Also used for images (hide patches and rebuild them).
  - **Contrastive learning:** make two altered views of the same item land close together in space and different items far apart. This is one way to train an [[Embedding Model]] (compare with [[Cosine Similarity]]).
  - **Denoising and reconstruction:** corrupt the input and learn to repair it.
- **Why it matters for LLMs:** the first stage of training an LLM is self-supervised **pretraining** on a very large amount of text. Later stages use labelled examples ([[Fine-tuning]]) and human preference signals ([[RLHF]]).
- **Relation to the other kinds:** the training step looks exactly like [[Supervised Learning]] (input, label, loss, update). The only difference is **where the label comes from**: not from people but from the data. It is often loosely called "unsupervised" ([[Unsupervised Learning]]), but there is a real label.
- **Quality check:** [[Perplexity]] measures how surprised a language model is on text it has not seen.
- **Practical cautions:**
  - The model learns whatever patterns are in the data, including mistakes, bias and rude text.
  - Good scores on the pretext task do not guarantee good scores on the real job.
  - Facts in the training text can be memorised, and can be wrong.

```python
import math
from collections import Counter
text = "the cat sat on the mat the cat ate".split()
pairs = list(zip(text, text[1:]))                 # self-made (input, label) pairs: word -> next word
c = Counter(pairs); tot = Counter(w for w, _ in pairs)
p = lambda a, b: c[(a, b)] / tot[a]               # chance of the next word, counted from the text
print(len(pairs), p("the", "cat"), round(p("cat", "sat"), 2))   # 8 0.6666666666666666 0.5
loss = -sum(math.log(p(a, b)) for a, b in pairs) / len(pairs)
print(round(loss, 3), round(math.exp(loss), 3))                 # 0.412 1.51   average surprise, perplexity
```

No neural network here, just counting, but the idea is the same: the text produced its own question-and-answer pairs, and the score is how surprised we were.

## 🔴 4. Deep / formal
**The chain rule for language.** The chance of a whole sentence of $T$ tokens can be split into one prediction per position, each using the tokens before it. Next-token pretraining learns exactly these pieces:

$$
P(w_1,\dots,w_T)=\prod_{t=1}^{T}P\big(w_t\mid w_1,\dots,w_{t-1}\big)
$$

**Where:**
- $w_t$ is the token at position $t$, and $T$ is how many tokens there are. $w_1,\dots,w_{t-1}$ are all the tokens **before** position $t$.
- $P(w_t\mid w_1,\dots,w_{t-1})$ reads "the chance of token $w_t$ given everything before it" (see [[Probability]]).
- $\prod_{t=1}^{T}$ means "multiply all of these together, one for each position". **Matters:** the chance of a sentence is the product of the chances of each step, so a single very unlikely step makes the whole sentence unlikely.
- **Matters:** this is exact, not an approximation, so a model that gets every conditional chance right gets the whole distribution right. A model that predicts only the next token is therefore not a weak target.

**Example (tiny numbers):** the sentence "the cat sat" with made-up chances $P(\text{the})=0.2$, $P(\text{cat}\mid\text{the})=0.1$, $P(\text{sat}\mid\text{the cat})=0.5$.
- Multiply: $0.2\times0.1\times0.5=\mathbf{0.01}$.
- Meaning: the model gives this exact three-word sentence a 1% chance. Every extra word multiplies by another number below 1, so long sentences get tiny chances, which is why we work with logarithms (next).

**The training loss.** Multiplying many small numbers underflows, so we take logarithms, turning the product into a sum. The usual loss is the average negative log-chance of the true next token:

$$
\mathcal{L}=-\frac{1}{T}\sum_{t=1}^{T}\log P_\theta\big(w_t\mid w_{<t}\big)
$$

**Where:**
- $P_\theta$ is the model's predicted chance. $\theta$ (theta) stands for all the model's adjustable numbers. $w_{<t}$ is a short way to write "all tokens before $t$".
- $\log P_\theta(\cdot)$ is the logarithm of the chance the model gave to the **true** token. It is $0$ when the chance is 1 and a large negative number when the chance is near 0. The minus sign makes it positive.
- $\frac1T\sum_t$ averages over the positions. **Matters:** the average makes the loss independent of text length. This is the cross-entropy loss of [[Loss Function]], and the model's chances come from a [[Softmax]] over the vocabulary.
- **Matters:** minimising $\mathcal{L}$ pushes the model to put high chance on what really comes next. Without the log, confident wrong answers would not be punished hard enough.
- $e^{\mathcal L}$ is the [[Perplexity]]: roughly, how many equally likely choices the model is torn between at each step.

**Example (tiny numbers):** the same three chances $0.2,\ 0.1,\ 0.5$ for the true tokens (natural logarithms).
- Surprises: $-\ln0.2=\mathbf{1.609}$, $-\ln0.1=\mathbf{2.303}$, $-\ln0.5=\mathbf{0.693}$.
- Average: $(1.609+2.303+0.693)/3=\mathbf{1.535}$.
- Perplexity: $e^{1.535}=\mathbf{4.64}$, the same as $0.01^{-1/3}$.
- Meaning: on this sentence the model behaves as if it were choosing among about 4.6 equally likely words at each step. A perfect model would score 1.

**Further notes:**
- **Masked vs next-token:** masked models see both sides, which suits understanding tasks. Next-token models see only the left, which makes generating text natural. Both are self-supervised.
- **Scaling:** results on next-token prediction are reported to improve smoothly with more data, more parameters and more compute (the "scaling laws" papers; from memory, check the exact claims).
- **What is really learned?** To predict the next word well you need grammar, facts and some reasoning. Whether this amounts to understanding is debated; this note does not take a side.
- **Limits:** self-supervision supplies knowledge from the data, but not the *goal* of being helpful or honest. That is what [[Fine-tuning]] and [[RLHF]] add, and [[Reinforcement Learning]] is the idea behind the second.

**What came before, and how this improved on it:**
- **Before:** supervised learning, where people write a label for every example.
- **Its limit:** labels cost human time, and there is far more unlabelled data than labelled.
- **What changed:** make the label out of the data itself, such as the next word, or a hidden word ([[Word2Vec 2013]], [[BERT 2018]], [[GPT-3 2020]]).
- **How much better:** in the code below, a model that only counts which character follows which, learned from about 780,000 characters of this vault's own text with 0 human labels, needs 3.88 bits per character on unseen text. Guessing evenly among the 97 different characters would need 6.6. The numbers shift a little whenever the vault text changes. A character-pair counter is the simplest case; a real language model uses the same trick with far more context.

**Check it yourself:**

```python
import glob, collections, math
fs = sorted(glob.glob(r"D:/python456/study&note stuff/20-ML-LLM/*.md"))
rd = lambda L: " ".join(open(f, encoding="utf-8").read() for f in L).lower()
tr, te = rd(fs[::2]), rd(fs[1::2])
V = sorted(set(tr)); idx = {c: i for i, c in enumerate(V)}
cnt = collections.defaultdict(collections.Counter)
for a, b in zip(tr, tr[1:]): cnt[a][b] += 1                      # labels come from the text itself: next character
def bits(a, b):
    c = cnt[a]; return -math.log2((c[b] + 0.1) / (sum(c.values()) + 0.1 * len(V)))
pairs = [(a, b) for a, b in zip(te, te[1:]) if a in idx and b in idx]
print("characters in the training text:", len(tr), " human-written labels used: 0")
print("average bits per character: guessing evenly", round(math.log2(len(V)), 2), "  learned from the text itself", round(sum(bits(a, b) for a, b in pairs) / len(pairs), 2))
```

## Related
- [[Supervised Learning]] · [[Unsupervised Learning]] · [[Reinforcement Learning]] · [[LLM]] · [[Transformer]] · [[Tokenization]] · [[Perplexity]] · [[Loss Function]] · [[Softmax]] · [[Embedding Model]] · [[Cosine Similarity]] · [[Fine-tuning]] · [[RLHF]] · [[Probability]] · [[Neural Network]] · [[Prompt Engineering]] · [[Generative Models]] · [[Word2Vec]] · [[Negative Sampling]] · [[Pre-training]] · [[Autoencoder]] · [[Diffusion Models]] · [[Transfer Learning]] · [[BERT]] · [[Contrastive Learning]] · [[CLIP]]

## Sources
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, 2018 (the masked prediction task)
- Radford et al., *Improving Language Understanding by Generative Pre-Training* (GPT), 2018 (from memory)
- Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR), 2020 (from memory)
- Kaplan et al., *Scaling Laws for Neural Language Models*, 2020 (from memory)
- Goodfellow, Bengio & Courville, *Deep Learning*, 2016, chapter 10

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/ml-foundations Self-supervised learning:::Make your own homework questions from a pile of text or pictures and learn by answering them. The data hides a piece of itself, the model guesses it, then checks the real piece. No answer key is needed.

#flashcards/ml/formula #flashcards/topic/ml-foundations Write the chain rule for the probability of a sentence and explain each symbol.
?
$$P(w_1,\dots,w_T)=\prod_{t=1}^{T}P\big(w_t\mid w_1,\dots,w_{t-1}\big)$$
- $w_t$: the token at position $t$. $T$: how many tokens. $w_1,\dots,w_{t-1}$: all tokens before position $t$.
- $P(w_t\mid w_1,\dots,w_{t-1})$: the chance of token $w_t$ given everything before it.
- $\prod_{t=1}^{T}$: multiply all of these together, one per position. A single very unlikely step makes the whole sentence unlikely.
- This is exact, not an approximation.

#flashcards/ml/example #flashcards/topic/ml-foundations "the cat sat" with $P(\text{the})=0.2$, $P(\text{cat}\mid\text{the})=0.1$, $P(\text{sat}\mid\text{the cat})=0.5$. What is the sentence probability?::$0.2\times0.1\times0.5=0.01$. A 1% chance. Every extra word multiplies by another number below 1, so long sentences get tiny chances, which is why we use logarithms.

#flashcards/ml/formula #flashcards/topic/ml-foundations Write the next-token training loss and explain each symbol.
?
$$\mathcal{L}=-\frac{1}{T}\sum_{t=1}^{T}\log P_\theta\big(w_t\mid w_{<t}\big)$$
- $P_\theta$: the model's predicted chance. $\theta$: all the model's adjustable numbers. $w_{<t}$: all tokens before $t$.
- $\log P_\theta(\cdot)$: the log of the chance the model gave to the true token. It is 0 when the chance is 1 and very negative near 0. The minus sign makes it positive.
- $\frac1T\sum_t$: the average over positions, so the loss does not depend on text length. This is the cross-entropy loss.

#flashcards/ml/example #flashcards/topic/ml-foundations Same chances 0.2, 0.1, 0.5. What are the loss and the perplexity?::Surprises $1.609$, $2.303$, $0.693$. Average $1.535$. Perplexity $e^{1.535}=4.64$ (also $0.01^{-1/3}$). The model acts as if choosing among about 4.6 equally likely words at each step. A perfect model would score 1.

#flashcards/ml/image #flashcards/topic/ml-foundations ![[bert-masked-lm.png]]
What does this picture show?
?
Masked prediction. The bottom row is the real sentence cut into tokens. One token ("##ica") is replaced by [MASK], the sentence goes through a Transformer encoder (BERT), and the model must output scores (logits) for what the hidden token was. The label comes from the sentence itself. (From the note's caption: Daniel Voigt Godoy, Wikimedia Commons.)

#flashcards/ml/cloze #flashcards/topic/ml-foundations In self-supervised learning the ==label comes from the data itself==, so nobody writes an answer key.
