---
tags: [LLM, decoding, sampling, beam-search, temperature, top-p, generation]
status: growing
level: 4
tier: 2
---

# Decoding Strategies

How an [[LLM]] turns a list of probabilities into the **next word**: always take the best, keep several candidates, or roll a weighted die.

## 🟢 1. Simple (age 12)
An AI writes one word at a time. At each step it has a **scoreboard** that says how likely every possible next word is. A *decoding strategy* is the rule for choosing from the scoreboard. You can **always pick the top word** (safe but dull), **try a few paths and keep the best** (careful), or **roll a weighted die** so likely words win most of the time but unusual ones can appear (creative).

**Example:** Finishing the sentence "The cat sat on the ___". Picking the top word always gives "mat". Rolling a weighted die sometimes gives "roof" or "sofa", so the story varies each time.

## 🔵 2. How it works
1. The model reads the text so far and gives a **score** (a logit) to every token in its vocabulary. [[Softmax]] turns the scores into probabilities that add up to 1.
2. **Greedy decoding:** pick the token with the highest probability, every time. It is fast, but the text can be repetitive and not the best overall.
3. **Beam search:** keep the $k$ most likely **partial sentences** at each step, extend each, and keep the best $k$ again. It looks further ahead than greedy, but costs $k$ times the work and still tends to be bland.
4. **Sampling:** draw the next token at random, using the probabilities as weights.
5. **Temperature** reshapes the probabilities before the draw. Low temperature makes the top token even more likely (safe). High temperature flattens them (more surprising).
6. **Top-k** keeps only the $k$ most likely tokens and renormalises. **Top-p** (also called nucleus sampling) keeps the smallest set of tokens whose probabilities add up to at least $p$, so the set grows when the model is unsure and shrinks when it is sure.
7. **Guided decoding:** forbid tokens that would break a required format (for example, only allow tokens that keep the output valid JSON).
8. The result then goes back into the model as input for the next step, and the loop repeats until a stop token. A [[KV Cache]] makes the loop fast.

![[beam-search-width-3.gif]]
*A search tree. At each level the search keeps only the 3 best branches (red) and drops the rest (grey), which is what beam search with width 3 does. The numbers are scores of nodes in a generic tree, not words. Image: BogdanShevchenko, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Beam_search.gif), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **What the model really outputs:** one number per vocabulary token, the **logit** $z_i$. The probability is the softmax of the logits (level 4).
- **Greedy:** $\arg\max$ of the probabilities. Deterministic, up to tiny numeric differences in the hardware.
- **Beam search:** it keeps $k$ hypotheses ranked by total probability (usually as a sum of logs, because multiplying many small numbers underflows). It stops a path at the end-of-sequence token. The lecture's limits: it needs extra computation and lacks diversity.
- **Sampling with temperature $T$:** divide the logits by $T$ before the softmax. $T\to0$ approaches greedy, $T=1$ is the model's own distribution, $T>1$ is flatter.
- **Top-$k$ and top-$p$:** cut off the unlikely tail. Top-$k$ uses a fixed count, top-$p$ a fixed amount of probability. They can be combined with temperature.
- **Why cut the tail:** the paper behind nucleus sampling says text produced by maximising likelihood tends to be bland and strangely repetitive, and that sampling from the "dynamic nucleus" of the distribution gives text closer to human text (arXiv 1904.09751 abstract, read).
- **Guided decoding:** at each step a checker marks which next tokens are allowed, and the others get probability 0 before the draw. The model's own preferences still choose among the allowed ones.
- **Repeatability:** greedy is the most repeatable, but even then exact repeatability needs care with batch size and hardware. The lecture points to a 2025 paper on it, which I did not read.
- **Practical habit:** low temperature for facts, code and judging; higher for brainstorming. The lecture's "LLM as a judge" practice uses a low temperature for reproducibility (see [[Prompt Engineering]]).

```python
import numpy as np
words = ['the', 'a', 'cat', 'dog', 'sat', 'ran']
logits = np.array([2.0, 1.5, 1.0, 0.5, 0.0, -1.0])
softmax = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
p = softmax(logits)
print('probabilities:', dict(zip(words, p.round(3))))
print('greedy picks:', words[int(np.argmax(p))])
# temperature: divide the logits by T before the softmax
for T in (0.5, 1.0, 2.0):
    print('T =', T, '->', softmax(logits / T).round(3).tolist())
# top-k: keep the k most probable tokens, renormalise
def top_k(p, k):
    q = np.zeros_like(p); idx = np.argsort(-p)[:k]; q[idx] = p[idx]; return q / q.sum()
print('top-3 :', top_k(p, 3).round(3).tolist())
# top-p: smallest set of tokens whose probabilities add up to at least p0
def top_p(p, p0):
    order = np.argsort(-p); c = np.cumsum(p[order]); m = int(np.searchsorted(c, p0) + 1)
    q = np.zeros_like(p); q[order[:m]] = p[order[:m]]; return q / q.sum(), m
q, m = top_p(p, 0.9)
print('top-p 0.9 keeps', m, 'tokens:', q.round(3).tolist())
# sampling: draw 100000 tokens and compare the frequencies with the probabilities
rng = np.random.RandomState(0)
draw = rng.choice(6, 100000, p=p)
print('sample frequencies:', (np.bincount(draw, minlength=6) / 100000).round(3).tolist())
# greedy against beam search on a two-step toy: first word 'x' (0.6) or 'y' (0.4); after x the next-word choices are flat, after y one word is almost certain
first = {'x': 0.6, 'y': 0.4}
second = {'x': {'p': 0.4, 'q': 0.3, 'r': 0.3}, 'y': {'p': 0.9, 'q': 0.05, 'r': 0.05}}
g1 = max(first, key=first.get); g2 = max(second[g1], key=second[g1].get)
print('greedy path:', g1 + g2, 'probability', round(first[g1] * second[g1][g2], 3))
paths = {a + b: first[a] * second[a][b] for a in first for b in second[a]}
best = max(paths, key=paths.get)
print('best path (beam search with 2 beams finds it):', best, 'probability', round(paths[best], 3))
print('beam with 2 beams keeps first words:', sorted(first, key=first.get, reverse=True)[:2])
```

## 🔴 4. Deep / formal
**Temperature.** Given logits $z_1,\dots,z_V$ for $V$ possible tokens:

$$
p_i(T)=\frac{e^{z_i/T}}{\sum_{j=1}^{V}e^{z_j/T}}
$$

**Where:**
- $z_i$ is the model's raw score (logit) for token $i$, and $V$ is the vocabulary size.
- $T>0$ is the temperature. It divides the scores before the softmax.
- $e^{(\cdot)}$ makes every score positive, and the sum in the bottom makes the probabilities add up to 1 (see [[Softmax]]).
- **Matters:** as $T$ gets small the gaps between scores are magnified and the top token takes almost all the probability. As $T$ gets large the gaps shrink and the probabilities become nearly equal. Without $T$ you only have the model's own $T=1$ distribution.

**Example (tiny numbers):** three tokens with logits $2.0,\ 1.5,\ 1.0$.
- $T=1$: $e^{2}=7.389$, $e^{1.5}=4.482$, $e^{1}=2.718$, sum $=14.589$. Probabilities $\mathbf{0.507,\ 0.307,\ 0.186}$.
- $T=0.5$: the scores become $4.0,3.0,2.0$, giving $e^4=54.60$, $e^3=20.09$, $e^2=7.389$, sum $82.08$. Probabilities $\mathbf{0.665,\ 0.245,\ 0.090}$.
- $T=2$: scores $1.0,0.75,0.5$ give $\mathbf{0.419,\ 0.327,\ 0.254}$.
- Meaning: the top token goes from $0.507$ to $0.665$ when the temperature is halved, and falls to $0.419$ when it is doubled.

**Probability of a whole sentence.** A model gives the probability of a sentence as a product (the chain rule, see [[Probability]]):

$$
P(y_1,\dots,y_n)=\prod_{t=1}^{n}P\!\left(y_t\mid y_1,\dots,y_{t-1}\right)
$$

**Where:**
- $y_t$ is the token at position $t$, and $n$ is the sentence length.
- $P(y_t\mid y_1,\dots,y_{t-1})$ is the probability of that token given the tokens before it (what the model outputs at step $t$).
- $\prod$ multiplies all of these together.
- **Matters:** the best next token at each step is not always on the best whole sentence, because an early choice changes all the later probabilities. This is why greedy can lose to beam search. It also means long sentences have tiny probabilities (so real code adds logs instead).

**Example (tiny numbers):** two first words, $x$ ($0.6$) and $y$ ($0.4$). After $x$ the best next word has probability $0.4$. After $y$ the best next word has probability $0.9$.
- Greedy takes $x$ (the larger first step) and then its best, so the path has $0.6\times0.4=\mathbf{0.24}$.
- The path through $y$ has $0.4\times0.9=\mathbf{0.36}$, which is better.
- Meaning: beam search with 2 beams keeps both first words and finds the $0.36$ path. Greedy missed it. The numbers in the code are a toy, not a real model.

**Further notes:**
- **Top-$p$ in the code:** with probabilities $0.42, 0.255, 0.154, 0.094, 0.057, 0.021$ and $p=0.9$, the running total is $0.42, 0.675, 0.829, 0.923$, so 4 tokens are kept and renormalised.
- **No free lunch:** more randomness gives variety and also more mistakes. Fewer paths gives safety and also repetition.
- **Reasoning models:** the lecture on reasoning gives the idea of sampling several answers and taking the majority (self-consistency) as a way to trade cost for accuracy (see [[Prompt Engineering]]).

**What came before, and how this improved on it:**
- **Before:** greedy choice and beam search, the standard decoding methods in older machine translation and sequence models (from memory; I did not check dates or authors).
- **Its limit:** for open-ended writing, maximising likelihood gives bland and repetitive text, according to the nucleus-sampling paper's abstract (arXiv 1904.09751, 22 April 2019, read).
- **What changed:** sample from the distribution, with temperature to reshape it and top-$k$ or top-$p$ to cut the unreliable tail.
- **How much better:** I did not measure text quality. What the code does show: temperature changes the top token's probability from $0.42$ to $0.635$ ($T=0.5$) or $0.29$ ($T=2$) for the same scores; top-$p$ at $0.9$ keeps 4 of 6 tokens in the toy; the sample frequencies match the probabilities to about three decimals; and on the two-step toy, beam search found a path of probability $0.36$ against greedy's $0.24$.

## Related
- [[LLM]] · [[Softmax]] · [[Probability]] · [[KV Cache]] · [[Speculative Decoding]] · [[Prompt Engineering]] · [[Perplexity]] · [[Transformer]] · [[Entropy]] · [[Beam Search]] · [[Chain of Thought]] · [[Multi-Token Prediction]] · [[Autoregressive Generation]] · [[Hallucination]] · [[Structured Output]]

## Sources
- Stanford CME 295 (Fall 2025), Lecture 3 slides, Afshine and Shervine Amidi
- Holtzman et al., *The Curious Case of Neural Text Degeneration*, 2019 (arXiv 1904.09751, abstract read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving Decoding strategy:::The rule for choosing the next token from the model's scoreboard of probabilities (always the top one, keep several paths, or sample).

#flashcards/ml/formula #flashcards/topic/llm-serving Write the softmax with temperature and explain each symbol.
?
$$p_i(T)=\frac{e^{z_i/T}}{\sum_{j=1}^{V}e^{z_j/T}}$$
- $z_i$: the model's raw score (logit) for token $i$. $V$: the vocabulary size.
- $T>0$: the temperature. It divides the scores before the softmax.
- $e^{(\cdot)}$ makes every score positive. The sum at the bottom makes the probabilities add to 1.
- Small $T$ magnifies gaps, so the top token takes almost all the probability. Large $T$ shrinks gaps, so probabilities become nearly equal. $T=1$ is the model's own distribution.

#flashcards/ml/example #flashcards/topic/llm-serving Logits 2.0, 1.5, 1.0. What are the probabilities at $T=1$ and at $T=0.5$?::$T=1$: about 0.507, 0.307, 0.186. $T=0.5$: 0.665, 0.245, 0.090. Lower temperature makes the top token more likely.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the probability of a whole sentence and explain each symbol.
?
$$P(y_1,\dots,y_n)=\prod_{t=1}^{n}P\!\left(y_t\mid y_1,\dots,y_{t-1}\right)$$
- $y_t$: the token at position $t$. $n$: the sentence length.
- $P(y_t\mid y_1,\dots,y_{t-1})$: the probability of that token given the tokens before it (what the model outputs at step $t$).
- $\prod$: multiplies all these together.
- The best next token at each step is not always on the best whole sentence, so greedy can lose to beam search. Long sentences get tiny probabilities, so real code adds logs instead.

#flashcards/ml/example #flashcards/topic/llm-serving First words $x$ (0.6) and $y$ (0.4). After $x$ the best next word has 0.4, after $y$ it has 0.9. What does greedy get, and what is the better path?::Greedy takes $x$ and gets $0.6\times0.4=0.24$. The path through $y$ gives $0.4\times0.9=0.36$, which is better. Greedy misses it.

#flashcards/ml/image #flashcards/topic/llm-serving ![[beam-search-width-3.gif]]
What does this picture show?
?
A search tree. At each level only the 3 best branches (red) are kept and the rest (grey) are dropped, which is beam search with width 3. The numbers are scores of nodes in a generic tree, not words. (From the note's caption: BogdanShevch, Wikimedia Commons.)

#flashcards/ml/reverse #flashcards/topic/llm-serving Greedy decoding:::Pick the highest-probability token every time. Fast, but can be repetitive and miss the best whole sentence.

#flashcards/ml/reverse #flashcards/topic/llm-serving Top-k:::Keep only the $k$ most likely tokens, renormalise, then sample.

#flashcards/ml/reverse #flashcards/topic/llm-serving Top-p (nucleus sampling):::Keep the smallest set of tokens whose probabilities add up to at least $p$, then sample. The set grows when the model is unsure and shrinks when it is sure.

#flashcards/ml/cloze #flashcards/topic/llm-serving Low temperature makes output ==safer==, and high temperature makes it ==more surprising==.

#flashcards/ml/cloze #flashcards/topic/llm-serving Beam search costs about ==k times== the work of greedy decoding.
