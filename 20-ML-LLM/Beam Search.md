---
tags: [decoding, search, beam-search, LLM, algorithms, generation]
status: growing
level: 4
tier: 2
---

# Beam Search

A way of writing text that keeps the **$k$ best partial sentences** at every step, instead of committing to just one.

## 🟢 1. Simple (age 12)
When you write a sentence one word at a time, the best-looking first word might lead to a dull ending. Beam search is like **walking down several paths at once**: at every step you keep the few most promising paths and drop the rest. At the end you pick the best path you found. It takes more effort than picking the single best word each time, but it often finds a better whole sentence.

**Example:** Choosing a route through a maze with a torch that lights only a few paths at a time. A walker who always takes the brightest next corridor can end up in a dead end. A team of three walkers who keep the three best routes is more likely to get through.

## 🔵 2. How it works
1. A language model gives, for each possible next word, a probability ([[Decoding Strategies]]).
2. **Greedy** decoding takes the top word each time. It never looks back, and can miss a better full sentence.
3. Beam search keeps $k$ candidates (the **beam width**). Start with the empty sentence.
4. At each step, extend every candidate by every possible next word. That gives many new candidates.
5. Score each by the **total probability** of the whole sentence so far (adding the logs of the probabilities).
6. Keep only the best $k$. Throw the rest away (the picture shows kept branches in red and dropped branches in grey).
7. A candidate that ends (reaches the end-of-text token) is set aside as a finished sentence.
8. At the end, return the best finished sentence.
9. With $k=1$ beam search is exactly greedy. With a huge $k$ it becomes a full search of every sentence. In between it trades work for quality.

![[beam-search-width-3.gif]]
*A search tree. At each level the search keeps only the 3 best branches (red) and drops the rest (grey), which is what beam search with width 3 does. The numbers are scores of nodes in a generic tree, not words. Image: BogdanShevchenko, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Beam_search.gif), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Cost:** about $k$ times the work of greedy decoding at each step, because $k$ candidates are extended. Each candidate needs its own [[KV Cache]], though shared starts can share it.
- **Not a guarantee:** beam search finds a high-probability sentence, not necessarily *the* best. The code compares with an exhaustive search over all $5^4=625$ sequences.
- **Lecture's limits:** it needs extra computation and lacks diversity or creativity. For open-ended chat, sampling is more common ([[Decoding Strategies]]).
- **Length bias:** because probabilities multiply, shorter sentences tend to score higher. Real systems often normalise by length (a common practice; not in the lecture, and not tested here).
- **Where it is used:** tasks with a fairly well-defined best output such as machine translation and speech recognition (from memory; I did not check).
- **Related searches:** the same "keep the best few" idea appears in other algorithms ([[Graphs]], [[Greedy Algorithms]]).
- **In the code:** a made-up model where the next-token probabilities depend on the position and the previous token. For one random seed greedy gets probability $0.03043$ and the best sequence has $0.09816$.

```python
import itertools
import numpy as np
rng = np.random.RandomState(149)
V, T = 5, 4                                                      # 5 possible tokens, sequences of length 4
# a made-up language model: the next-token probabilities depend on the previous token and the position
table = rng.dirichlet(np.ones(V) * 0.4, size=(T, V))             # table[t][prev] is a distribution over the next token
def logp(seq):
    prev, total = 0, 0.0
    for t, tok in enumerate(seq): total += np.log(table[t][prev][tok]); prev = tok
    return total
def beam(k):
    beams = [((), 0.0)]
    for t in range(T):
        cand = []
        for seq, lp in beams:
            prev = seq[-1] if seq else 0
            for tok in range(V): cand.append((seq + (tok,), lp + np.log(table[t][prev][tok])))
        beams = sorted(cand, key=lambda c: -c[1])[:k]
    return beams[0]
best = max(itertools.product(range(V), repeat=T), key=logp)      # try all 5^4 = 625 sequences
print('exhaustive best:', best, 'probability', round(float(np.exp(logp(best))), 5))
greedy = []; prev = 0
for t in range(T): tok = int(np.argmax(table[t][prev])); greedy.append(tok); prev = tok
print('greedy         :', tuple(greedy), 'probability', round(float(np.exp(logp(greedy))), 5))
print('beam k = 1 equals greedy:', beam(1)[0] == tuple(greedy))
for k in (1, 2, 3, 5, 25, 125):
    seq, lp = beam(k); print(f'beam k = {k:3d}: {seq} probability {np.exp(lp):.5f} | is the exhaustive best: {seq == best}')
# how often does each beam width find the best sequence, over 300 random models?
hits = {k: 0 for k in (1, 3, 10)}
for trial in range(300):
    table = rng.dirichlet(np.ones(V) * 0.4, size=(T, V))
    b = max(itertools.product(range(V), repeat=T), key=logp)
    for k in hits: hits[k] += beam(k)[0] == b
print('share of 300 random models where the best sequence is found:', {k: round(v / 300, 3) for k, v in hits.items()}, '| work per step grows with k')
```

## 🔴 4. Deep / formal
**The score beam search keeps.** For a candidate sentence $y_1,\dots,y_t$:

$$
\text{score}(y_{1:t})=\sum_{i=1}^{t}\log P\!\left(y_i\mid y_{1:i-1}\right)
$$

**Where:**
- $y_i$ is the word at position $i$ and $P(y_i\mid y_{1:i-1})$ is the model's probability for it given the words before.
- $\log$ turns the product of probabilities into a sum, so many small numbers do not underflow to 0.
- The beam keeps the $k$ candidates with the highest score.
- **Matters:** the sum of logs ranks candidates exactly like the product of probabilities does. A candidate with a so-so first word and strong later words can overtake a candidate with a top first word and weak later words, which greedy decoding can never do.

**Example (tiny numbers):** two first words $x$ ($0.6$) and $y$ ($0.4$). After $x$ the best next word has $0.4$, after $y$ it has $0.9$.
- Path $x$-then-best: $\ln0.6+\ln0.4=-0.511-0.916=\mathbf{-1.427}$, probability $0.24$.
- Path $y$-then-best: $\ln0.4+\ln0.9=-0.916-0.105=\mathbf{-1.022}$, probability $0.36$.
- Meaning: greedy takes $x$ and ends with $0.24$. Beam search with $k=2$ keeps both first words and finds the $0.36$ path.

**Further notes:**
- **What the code shows (one random model):** beam with $k=1$ equals greedy ($0.03043$). With $k=2$ it found $(0,0,3,3)$ at $0.06400$. With $k=3$ and above it found the exhaustive best $(1,3,3,3)$ at $0.09816$, about 3.2 times greedy's probability.
- **How often (300 random models):** the best sequence was found by $k=1$ in $68.7\%$ of models, by $k=3$ in $96.7\%$ and by $k=10$ in $100\%$. These are for tiny sequences of length 4 with 5 tokens; real texts are much longer and bigger, so I would not carry the percentages over.
- **Not covered:** length normalisation, diverse beam search, and constrained decoding.

**What came before, and how this improved on it:**
- **Before:** greedy decoding, taking the single most likely word at every step.
- **Its limit:** a locally best word can lead to a poor sentence. In the code's random model, greedy found $0.03043$ where the best was $0.09816$.
- **What changed:** keep several partial sentences alive at once and choose the best total score (a classic search idea; I did not trace who first applied it to language models).
- **How much better:** in this note's code, one model went from $0.03043$ (greedy) to $0.09816$ with a beam of 3, and over 300 random models the best sequence was found 68.7% of the time by greedy against 96.7% with a beam of 3. The price is roughly $k$ times the work per step, and the lecture notes it still lacks diversity.

## Related
- [[Decoding Strategies]] · [[LLM]] · [[KV Cache]] · [[Speculative Decoding]] · [[Probability]] · [[Greedy Algorithms]] · [[Graphs]] · [[Softmax]] · [[Perplexity]]

## Sources
- Stanford CME 295 (Fall 2025), Lecture 3 slides, Afshine and Shervine Amidi
- Holtzman et al., *The Curious Case of Neural Text Degeneration*, 2019 (arXiv 1904.09751, abstract read; discusses likelihood-maximising decoding)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving Beam search:::Keep the $k$ most promising partial sentences at every step (the beam width), extend each, and keep the best $k$ again. Return the best finished sentence at the end.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the beam score of a partial sentence and explain each symbol.
?
$$\text{score}(y_{1:t})=\sum_{i=1}^{t}\log P\!\left(y_i\mid y_{1:i-1}\right)$$
- $y_i$: the word at position $i$. $P(y_i\mid y_{1:i-1})$: the model's probability for it given the words before.
- $\log$: turns the product of probabilities into a sum, so small numbers do not underflow to 0.
- The beam keeps the $k$ candidates with the highest score.
- The sum of logs ranks candidates exactly as the product of probabilities does. A so-so first word with strong later words can overtake a top first word with weak later words, which greedy can never do.

#flashcards/ml/example #flashcards/topic/llm-serving First words $x$ (0.6), $y$ (0.4). After $x$ best next is 0.4, after $y$ it is 0.9. Give both log-scores.::Path $x$: $\ln0.6+\ln0.4=-1.427$ (probability 0.24). Path $y$: $\ln0.4+\ln0.9=-1.022$ (probability 0.36). Beam search with $k\ge2$ keeps $y$ and finds the better sentence. Greedy takes $x$ and misses it.

#flashcards/ml/image #flashcards/topic/llm-serving ![[beam-search-width-3.gif]]
What does this picture show?
?
A search tree. At each level only the 3 best branches (red) are kept and the rest (grey) are dropped, which is beam search with width 3. The numbers are scores of nodes in a generic tree, not words. (From the note's caption: BogdanShevch, Wikimedia Commons.)

#flashcards/ml/cloze #flashcards/topic/llm-serving With $k=1$, beam search is exactly ==greedy== decoding. With a huge $k$ it becomes a ==full search== of every sentence.

#flashcards/ml/concept #flashcards/topic/llm-serving Why is beam search compared with greedy decoding?::Greedy never looks back and can miss a better whole sentence. Beam search keeps several paths so a weak early choice can still win later.

#flashcards/ml/cloze #flashcards/topic/llm-serving Beam search costs ==k times== the work of greedy, and still tends to give bland text.
