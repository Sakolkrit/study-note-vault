---
tags: [LLM, machine-learning, training, self-supervised, scaling, transformer]
status: growing
level: 4
tier: 3
---

# Pre-training

Pre-training is the first, biggest stage of building an [[LLM]]: the model reads a huge pile of text and **practises guessing the next word, over and over**, until it has soaked up language and a lot of general knowledge.

## 🟢 1. Simple (age 12)
Imagine learning to speak by reading a million books, but with a game: you read a sentence, cover the last word, and try to guess it. "The cat sat on the ___". After millions of guesses, you get very good. Along the way you also pick up grammar, facts, and how stories go, simply because they help you guess better. Nobody ever had to explain the rules to you.

**Example:** A child who has heard thousands of bedtime stories knows that after "once upon a" comes "time". No one taught them that as a rule. They just heard it so often that the guess became automatic. Pre-training is that, with trillions of words instead of thousands.

## 🔵 2. How it works
1. **Gather text:** web pages, books, code, articles. Billions or trillions of words.
2. **Cut it into tokens:** small pieces of words (see [[Tokenization]]).
3. **Make a guessing task:** show the model the first part of a passage and ask for the next token. The right answer is simply the next token in the text, so the data labels itself. That is [[Self-Supervised Learning]].
4. **Score each guess:** the model gives a chance to every possible next token (see [[Softmax]]). The score is how much chance it gave to the *right* one. A low chance gives a big penalty (a [[Loss Function]]).
5. **Nudge the model:** use [[Backpropagation]] to find which weights to change, and a method like [[Stochastic Gradient Descent]] (in practice [[Adam Optimizer]]) to change them a tiny bit.
6. **Repeat** for trillions of tokens, on thousands of [[GPU]]s, over weeks or months.
7. **Result:** a **base model**. It is very good at continuing text but not yet good at following instructions. That comes next, in [[Fine-tuning]] and [[RLHF]].

![[ai-training-flop-trends.svg]]
*Each dot is an AI model. Across is the year, up is the training compute in FLOP (the number of arithmetic operations), on a log scale, so each gridline is 100 times more. The pink lines show the trends: frontier language models (bottom left) have grown about 5 times per year. Image: Epoch AI, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Trends_in_AI_training_FLOP_over_time_(2010-2025).svg), CC BY 4.0.*

## 🟠 3. Technical detail
- **The task:** *next-token prediction* (also "causal language modelling"). For a text $x_1,\dots,x_T$, at each position the model sees only the earlier tokens ([[Attention]] is masked so it cannot peek ahead) and predicts the next one.
- **The architecture:** almost always a [[Transformer]]: a stack of [[Attention]] layers with [[Residual Connection]]s and [[Normalization]], plus [[Positional Encoding]] to know word order.
- **What it learns:** grammar, facts, some reasoning patterns, code, many languages. All of this is a side effect of lowering the next-token loss. It also learns the **mistakes and biases** present in the text.
- **The loss number:** training lowers the average cross-entropy of the next token (see [[Entropy]]). Its exponential is [[Perplexity]]: roughly "how many equally likely tokens the model is torn between".
- **Scale is the main lever:** more parameters, more data and more compute reliably lower the loss, following smooth curves called [[Scaling Laws]]. A well-known finding (the "Chinchilla" result, DeepMind, 2022; checked against the paper) is that for a fixed compute budget, parameters and tokens should grow equally. Chinchilla itself has 70B parameters and 1.4T tokens, which is **20 tokens per parameter**, much more data than earlier models used. The "20 tokens per parameter" rule is that ratio. I did not find it stated as a general rule in the paper (see [[Scaling Laws]]).
- **Training recipe in practice:** a [[Learning Rate]] schedule with a warm-up and a slow decay, large batches, [[Normalization]], and often lower-precision numbers (such as 16-bit) to save memory. The model is split across many GPUs. Details vary by lab and are partly secret.
- **Data matters as much as size:** filtering junk, removing duplicates, and mixing in code and books changes the result a lot. Duplicated text can be memorised (see [[Overfitting]]).
- **Cost:** real runs cost millions of dollars of compute (for GPT-3, secondary compilations put it around 4.6 million dollars, with estimates up to 12 to 15 million; I did not find a primary source, and these are estimates, not disclosed figures). Fine-tuning afterwards is far cheaper.
- **After pre-training:** the base model is tuned to follow instructions ([[Fine-tuning]]), aligned with human preferences ([[RLHF]]), and used with [[Prompt Engineering]] or [[RAG]].
- **Not only for text:** the same recipe (guess the hidden part) is used for images, audio and code. Related pre-training for encoders hides a word in the middle (masked prediction) instead of the last one (stated from memory).

```python
import numpy as np
text = ("the cat sat on the mat . the dog sat on the log . the cat saw the dog . "
        "the dog saw the cat . the cat ate the fish . the dog ate the bone .").split()
vocab = sorted(set(text)); ix = {w: i for i, w in enumerate(vocab)}; V = len(vocab)
ids = np.array([ix[w] for w in text]); inp, tgt = ids[:-1], ids[1:]       # predict each next word from the one before
rng = np.random.default_rng(0)
W = rng.normal(0, 0.01, (V, V))                                            # row = current word, columns = scores for next word
def loss_and_probs():
    s = W[inp]; p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    return -np.log(p[np.arange(len(tgt)), tgt]).mean(), p                  # average cross-entropy of the right next word
l0, _ = loss_and_probs(); print(V, len(tgt), round(l0, 3), round(float(np.log(V)), 3))   # 12 37 2.482 2.485
for step in range(500):
    l, p = loss_and_probs(); g = p.copy(); g[np.arange(len(tgt)), tgt] -= 1  # gradient of cross-entropy w.r.t. the scores
    np.add.at(W, inp, -0.5 * g / len(tgt))                                   # step downhill (learning rate 0.5)
    if step in (0, 99, 499): print(step, round(l, 3), round(float(np.exp(l)), 2))
    # 0 2.482 11.97 | 99 1.331 3.79 | 499 0.895 2.45
cur = ix["the"]; s = W[cur]; q = np.exp(s - s.max()); q /= q.sum()
print([(vocab[i], round(float(q[i]), 2)) for i in np.argsort(-q)[:3]])     # [('dog', 0.33), ('cat', 0.33), ('mat', 0.08)]
N, D = 1e9, 2e10; C = 6 * N * D; print(C, round(C / 1e14 / 86400, 1))      # 1.2e+20 13.9
```

This is a toy "language model" with a context of just one word, trained on a 37-pair story. Before training the loss is 2.482, which is almost exactly $\ln 12=2.485$: the model knows nothing and spreads its chance evenly over 12 words (perplexity 12). After 500 steps the loss is 0.895 (perplexity 2.45), and after "the" it says "dog" or "cat" with about a third of the chance each, which fits the story. The best a one-word-context model could ever do on this exact text is a loss of 0.806 (I counted it separately from the word-pair frequencies), so there is a little room left. No toy like this can learn facts or grammar beyond one word of context. A real LLM looks at thousands of earlier tokens. The last line is the compute arithmetic from the formulas below.

## 🔴 4. Deep / formal
**The objective.** For a training text of $T$ tokens, pre-training minimises the average negative log chance of each token given the ones before it:

$$
\mathcal L(\theta)=-\frac1T\sum_{t=1}^{T}\log p_\theta\!\big(x_t\mid x_1,\dots,x_{t-1}\big)
$$

**Where:**
- $\theta$ stands for all the model's weights (billions of numbers). $x_t$ is the token at position $t$, and $x_1,\dots,x_{t-1}$ are the tokens before it.
- $p_\theta(x_t\mid\cdot)$ is the chance the model gives to the **actual** next token. $-\log$ of it is the penalty: 0 if the chance was 1, and bigger the less chance the model gave.
- $\frac1T\sum_t$ averages the penalty over all positions.
- **Matters:** this one number is the whole training signal. A model that is confidently wrong is punished heavily, so it learns to spread its chance sensibly. It is the same cross-entropy as in [[Entropy]] and the same as maximum likelihood on the text ([[Maximum Likelihood]]). Without the averaging, the loss would grow with the text length and not be comparable between runs.

**Example (tiny numbers):** a three-token text where the model gave the right next token the chances $0.5,\;0.25,\;0.125$.
- Penalties: $-\ln0.5=0.693$, $-\ln0.25=1.386$, $-\ln0.125=2.079$.
- Average: $(0.693+1.386+2.079)/3=\mathbf{1.386}$.
- Perplexity: $e^{1.386}=\mathbf{4.0}$.
- Meaning: on average, the model acts as if it were choosing among about 4 equally likely tokens at each step. A perfect model would score 0, with perplexity 1.

**Scale and loss.** Empirically, the loss falls as a smooth **power law** as the model gets bigger. A simple form (the exact numbers differ by study) is:

$$
\mathcal L(N)\approx L_\infty+\Big(\frac{N_c}{N}\Big)^{\alpha}
$$

**Where:**
- $N$ is the number of parameters. $L_\infty$ is a floor the loss cannot go below (the unavoidable uncertainty in language).
- $N_c$ and $\alpha$ are constants found by fitting measurements of many trained models. $\alpha$ is a small positive number (checked: Kaplan et al. report about 0.076 for model size, 0.095 for data and 0.050 for compute; the Chinchilla fit has 0.34 for model size and 0.28 for data, so values from 0.05 to 0.35 have been reported).
- $(N_c/N)^{\alpha}$ is the part of the loss that more parameters can remove. It shrinks slowly as $N$ grows.
- **Matters:** because the exponent is small, each doubling of the model removes only a small slice of the remaining gap, so progress needs *multiplying* the scale, not adding to it. The floor also means scale alone cannot give a perfect model. Data and compute need to grow along with $N$, or the curve flattens.

**Example (tiny numbers, illustrative, not a real model):** suppose $\alpha=0.1$.
- Doubling $N$ multiplies the reducible part by $2^{-0.1}=\mathbf{0.933}$.
- So each doubling removes about $6.7\%$ of what was left to gain.
- Ten doublings (a 1,024 times bigger model) give $0.933^{10}=0.5$: only half of the reducible loss is gone.
- Meaning: gains are steady but slow. This is why labs chase huge scale, and why the compute chart above climbs so fast.

**Compute.** A common estimate for training cost (Kaplan et al. write it as $C=6NBS$, with $B\times S$ the tokens processed; checked in the paper), in floating-point operations (FLOP), with $N$ parameters and $D$ training tokens:

$$
C\approx 6\,N\,D
$$

**Where:**
- $C$ is the total compute in FLOP. $N$ is the number of parameters. $D$ is the number of tokens trained on.
- The $6$ is a rule-of-thumb count: roughly 2 operations per parameter per token to run the model forward, and about 4 more for the backward pass ([[Backpropagation]]). It ignores smaller costs.
- **Matters:** it shows that cost grows with the *product* of model size and data size. Doubling both quadruples the cost. It lets you budget a run before you start.

**Example (tiny numbers):** $N=10^9$ parameters and $D=2\times10^{10}$ tokens (20 tokens per parameter).
- $C=6\times10^{9}\times2\times10^{10}=\mathbf{1.2\times10^{20}}$ FLOP.
- If one GPU delivers an effective $10^{14}$ FLOP per second (an assumed figure, real values vary): $1.2\times10^{20}/10^{14}=1.2\times10^{6}$ seconds, which is **13.9 days** on one GPU.
- Meaning: a small 1-billion-parameter model is a two-week job for one GPU, or an hour or so for a few hundred. Models 100 times bigger on 100 times more data cost 10,000 times more.

**Further notes:**
- **Emergent abilities:** some skills seem to appear only past a certain size. Whether they truly jump or only look that way on certain tests is debated (Wei et al., 2022 introduced the idea; Schaeffer et al., 2023 argued the jumps depend on the metric; abstracts checked only).
- **Data limits:** the supply of good public text is finite, so researchers study repeating data, synthetic data, and quality filters (an open area).
- **Memorisation:** large models can reproduce rare training text word for word, which raises privacy and copyright questions.
- **Compute trend:** the picture above shows frontier training compute growing about 5 times a year for language models in the period it covers. That trend may not continue.

**What came before, and how this improved on it:**
- **Before:** train a model from scratch on each task's own labelled data.
- **Its limit:** labels are costly, so small tasks gave weak models.
- **What changed:** first learn from huge unlabelled text, then adapt to the task ([[Fine-tuning]]). The BERT abstract says the pre-trained model can be fine-tuned "with just one additional output layer" for many tasks (arXiv 1810.04805, 11 October 2018, abstract read).
- **How much better:** the same abstract reports GLUE 80.5% (a 7.7-point absolute gain) and new best results on eleven tasks. In the code below, a toy of my own with invented data: with 2 labelled examples a classifier from scratch gets 0.775 and one using vectors learned from unlabelled sentences gets 1.000. With 20 examples scratch reaches 0.990, so the benefit shrinks as labels grow. A toy, not a language-model experiment.

**Check it yourself:**

```python
import numpy as np
# 20 labelled examples, 2 classes. Features: "bag of words" over a vocabulary.
# Scratch: learn word vectors from the 20 examples only. Pre-trained: reuse co-occurrence vectors from a big unlabelled text.
rng = np.random.default_rng(1)
topics = {0: ["goal", "match", "team", "coach", "score", "league"], 1: ["vote", "party", "law", "minister", "poll", "senate"]}
syn = {"goal": "score", "match": "league", "team": "coach", "vote": "poll", "party": "senate", "law": "minister"}
def doc(c, pool): return [rng.choice(topics[c]) for _ in range(6)] if pool else None
words = sorted({w for t in topics.values() for w in t}); ix = {w: i for i, w in enumerate(words)}
# unlabelled "big text": sentences each from one topic -> co-occurrence reveals topic
C = np.zeros((12, 12))
for _ in range(2000):
    c = rng.integers(2); s = rng.choice(topics[c], 4)
    for u in s:
        for v in s: C[ix[u], ix[v]] += 1
U, S, _ = np.linalg.svd(np.log1p(C)); E = U[:, :2] * S[:2]
def feat_scratch(d): v = np.zeros(12); [v.__setitem__(ix[w], 1) for w in d]; return v
def feat_pre(d): return np.mean([E[ix[w]] for w in d], 0)
def run(feat, n_train):
    acc = []
    for _ in range(200):
        tr = [(rng.choice(topics[c], 2), c) for c in rng.integers(2, size=n_train)]
        te = [(rng.choice(topics[c], 2), c) for c in rng.integers(2, size=200)]
        if len({c for _, c in tr}) < 2: continue
        X = np.array([feat(d) for d, _ in tr]); y = np.array([c for _, c in tr])
        m0, m1 = X[y == 0].mean(0), X[y == 1].mean(0)
        pr = [int(np.linalg.norm(feat(d) - m1) < np.linalg.norm(feat(d) - m0)) for d, _ in te]
        acc.append(np.mean(np.array(pr) == [c for _, c in te]))
    return np.mean(acc)
for n in (2, 4, 8, 20):
    print(f"{n:2d} labelled examples: from scratch {run(feat_scratch, n):.3f}   with pre-trained vectors {run(feat_pre, n):.3f}")
```

## Related
- [[LLM]] · [[Transformer]] · [[Attention]] · [[Tokenization]] · [[Self-Supervised Learning]] · [[Fine-tuning]] · [[RLHF]] · [[Perplexity]] · [[Entropy]] · [[Maximum Likelihood]] · [[Loss Function]] · [[Backpropagation]] · [[Stochastic Gradient Descent]] · [[Adam Optimizer]] · [[Learning Rate]] · [[GPU]] · [[Overfitting]] · [[Scaling Laws]] · [[Mixture of Experts]] · [[Quantization]] · [[Byte-Pair Encoding]] · [[Knowledge Distillation]] · [[Transfer Learning]] · [[BERT]] · [[Data Parallelism]] · [[Instruction Tuning]] · [[GPT-3 2020]] · [[LLaMA 2023]] · [[ELMo 2018]]

## Sources
- Kaplan et al., *Scaling Laws for Neural Language Models*, OpenAI, 2020, arXiv:2001.08361 (checked)
- Hoffmann et al., *Training Compute-Optimal Large Language Models* ("Chinchilla"), DeepMind, 2022, arXiv:2203.15556 (checked)
- Brown et al., *Language Models are Few-Shot Learners* (GPT-3), 2020, arXiv:2005.14165 (checked: authors, date and the 175B size on the abstract page)
- Radford et al., *Improving Language Understanding by Generative Pre-Training*, OpenAI, 2018 (an OpenAI technical report with no arXiv number; title and authors confirmed by search results only, not the original PDF)
- Wei et al., arXiv:2206.07682, and Schaeffer et al., arXiv:2304.15004 (abstracts only)
- Epoch AI, *Training compute of frontier AI models grows by 4-5x per year* (the source of the picture)
- Devlin et al., arXiv 1810.04805 (abstract read for the history block)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm Pre-training:::Learning to speak by reading a huge pile of text with a game: read a sentence, cover the last word, and guess it. After millions of guesses the model picks up grammar, facts and how stories go.

#flashcards/ml/formula #flashcards/topic/llm Write the pre-training loss and explain each symbol.
?
$$\mathcal L(\theta)=-\frac1T\sum_{t=1}^{T}\log p_\theta\!\big(x_t\mid x_1,\dots,x_{t-1}\big)$$
- $\theta$: all the model's weights (billions of numbers). $T$: the number of tokens.
- $x_t$: the token at position $t$. $x_1,\dots,x_{t-1}$: the tokens before it.
- $p_\theta(x_t\mid\cdot)$: the chance the model gives to the actual next token. $-\log$ of it is the penalty: 0 if the chance was 1, bigger the less chance it gave.
- $\frac1T\sum$: the average penalty over the text.

#flashcards/ml/formula #flashcards/topic/llm Write the loss against model size and explain each symbol.
?
$$\mathcal L(N)\approx L_\infty+\Big(\frac{N_c}{N}\Big)^{\alpha}$$
- $N$: the number of parameters. $L_\infty$: a floor the loss cannot go below (the unavoidable uncertainty in language).
- $N_c$, $\alpha$: constants found by fitting measurements of many trained models. $\alpha$ is a small positive number.
- Loss falls smoothly as a power law, with ever-smaller gains.

#flashcards/ml/formula #flashcards/topic/llm Write the rough training compute and explain each symbol.
?
$$C\approx 6\,N\,D$$
- $C$: the training compute in FLOP (arithmetic operations). $N$: the number of parameters. $D$: the number of training tokens.
- The 6 is a rule of thumb (about 2 operations per weight per token for the forward pass and 4 for the backward pass; the usual rule, not derived here).
- Doubling either the model or the data doubles the compute.

#flashcards/ml/cloze #flashcards/topic/llm Pre-training teaches a model by predicting the ==next token==, using lots of ==unlabelled== text.

#flashcards/ml/example #flashcards/topic/llm A 3-token text where the model gave the right next token 0.5, 0.25 and 0.125. What are the loss and the perplexity?::Penalties 0.693, 1.386, 2.079. Average 1.386. Perplexity e^1.386 = 4.0. The model acts as if it were choosing among about 4 equally likely words.

#flashcards/ml/example #flashcards/topic/llm $N=10^9$ parameters and $D=2\times10^{10}$ tokens. What is $C$, and how long on one GPU at an assumed $10^{14}$ FLOP/s?::C = 6 × 10⁹ × 2 × 10¹⁰ = 1.2 × 10²⁰ FLOP. 1.2 × 10²⁰ / 10¹⁴ = 1.2 × 10⁶ seconds, 13.9 days (the GPU speed is an assumed figure).

#flashcards/ml/image #flashcards/topic/llm ![[ai-training-flop-trends.svg]]
What does this picture show?
?
Each dot is an AI model. Across is the year, up is the training compute in FLOP on a log scale (each gridline is 100 times more). The pink lines show the trends: frontier language models have grown about 5 times per year.
