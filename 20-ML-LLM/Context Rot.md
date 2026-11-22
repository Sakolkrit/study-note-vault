---
tags: [LLM, long-context, evaluation, attention, retrieval]
status: growing
level: 4
tier: 2
---

# Context Rot

**Context rot** is the drop in a language model's quality as its input gets longer, even when the task stays the same.

## 🟢 1. Simple (age 12)
A model can accept a very long text, but that does not mean it uses all of it well. The longer the text, the more likely the model is to miss the one sentence that matters, like a student who has to find one fact in a 500-page book.

**Example:** You ask a friend a question about a story. If you give them one page, they answer at once. If you hand them the whole library and say "the answer is somewhere in here", they are more likely to get lost, even though the answer is still there.

## 🔵 2. How it works
1. A language model reads its input as a list of [[Tokenization|tokens]] inside its **context window** (the most it can read at once).
2. Modern models accept windows of tens or hundreds of thousands of tokens.
3. But accepting the text is not the same as using it. When the input is longer, quality often drops, even on a task that does not get harder.
4. One reason is that [[Attention]] spreads a fixed amount of attention over more and more tokens, so each token gets a smaller share (the code below shows this with numbers).
5. Another observed effect: where the information sits matters. The "Lost in the Middle" study found performance is often highest when the relevant information is at the beginning or end of the input, and worst in the middle.
6. Ways to cope: put the important parts at the start or end, keep the context short with [[RAG]] or [[Contextual Retrieval]], and test the model at the lengths you will really use.
7. **The catch:** a long window is a maximum, not a promise. Always measure on your own task ([[Benchmarks]]).

## 🟠 3. Technical detail
- **Name and source:** the lecture slide warns "Beware of context rot" and cites "Context Rot: How Increasing Input Tokens Impacts LLM Performance", Hong et al., 2025. That is a technical report I could not read here, so I quote nothing from it.
- **What I did read:** the abstract of "Lost in the Middle" (Liu et al., arXiv 2307.03172, from memory for the first author): language models "do not robustly make use of information in long input contexts", and performance degrades significantly when the relevant information is in the middle of a long context, even for explicitly long-context models.
- **Related terms:** context length, context size and window size all mean how many tokens fit (lecture slide).
- **Why attention dilutes:** softmax attention gives out weights that add up to 1 ([[Softmax]], [[Self-Attention]]). Many near-equal distractors take most of the weight, unless the relevant token scores much higher.
- **Cost side:** longer contexts also mean more memory for the [[KV Cache]].
- **Mitigations:** [[RAG]] to fetch only the relevant passages, summarising, and putting key instructions at the start or end.

```python
import numpy as np
delta = 3.0                                                           # the one relevant token scores 3 points higher than each distractor
print("attention weight on the one relevant token, among n distractors with equal scores:")
for n in (10, 100, 1000, 100000):
    w = np.exp(delta) / (np.exp(delta) + n); print(f"  n = {n:>7,}: {w:.4f}   (the distractors share the other {1 - w:.4f})")
print("scores needed to keep 50% of the attention at n = 100,000:", round(np.log(100000), 2), "points above each distractor")
```

## 🔴 4. Deep / formal
**Attention among distractors.** Suppose one relevant token has an attention score that is $\Delta$ higher than each of $n$ equal distractors. Its attention weight, after the softmax, is

$$
w=\frac{e^{\Delta}}{e^{\Delta}+n}
$$

**Where:**
- $w$ is the share of attention that goes to the relevant token, a number between 0 and 1.
- $\Delta$ is how much higher its score is than each distractor's score. $e^{\Delta}$ is $e$ (about 2.718) to the power $\Delta$: the relevant token's weight before dividing.
- $n$ is the number of distractor tokens, each with weight $e^{0}=1$ before dividing.
- **Matters:** with a fixed $\Delta$, the share falls roughly as $1/n$ when $n$ is large. To keep the same share in a longer text the model has to make the gap $\Delta$ grow like $\ln n$.

**Example (tiny numbers):** $\Delta=3$, so $e^{3}\approx20.1$.
- $n=10$: $w=20.1/(20.1+10)=\mathbf{0.668}$.
- $n=100$: $w=20.1/120.1=\mathbf{0.167}$.
- $n=1000$: $w=20.1/1020.1=\mathbf{0.0197}$.
- Meaning: the same relevant token, with the same advantage, gets 67% of the attention among 10 distractors and about 2% among 1,000. To keep 50% of the attention among 100,000 distractors, the gap would need to be $\ln100{,}000\approx\mathbf{11.5}$ points.

**Further notes:**
- **A toy, not the cause:** real models are many layers and heads deep and can learn sharp attention, so this shows one pressure, not the whole story. Studies of real models are the "Lost in the Middle" and "Context Rot" work, which I read only as an abstract and a slide title.
- **Position effects** (start and end better than the middle) are not simulated here.
- **Open question:** how to train models that use long inputs evenly.

**What came before, and how this improved on it:**
- **Before:** short context windows (a few thousand tokens), where everything fit and long-range use was not tested.
- **Its limit:** when windows grew to long sizes, people assumed the model used all of it equally.
- **What changed:** tests that vary the length and the position of the relevant information. "Lost in the Middle" (arXiv 2307.03172, 6 July 2023, abstract read) showed performance often highest at the beginning or end and significantly worse in the middle. The 2025 "Context Rot" report (named on the lecture slide) extends the question to increasing input length.
- **How much better:** this is an evaluation insight, not a fix, so there is no improvement number. What the code below measures is the pressure: with a fixed advantage of 3 points, the attention on the relevant token falls from 0.668 among 10 distractors to 0.167 among 100 and 0.0002 among 100,000.

**Check it yourself:** run the code in level 3 above. It prints every number quoted in this block.

## Related
[[Attention]] · [[Self-Attention]] · [[Softmax]] · [[KV Cache]] · [[RAG]] · [[Contextual Retrieval]] · [[Benchmarks]] · [[In-Context Learning]] · [[Prompt Engineering]] · [[LLM]] · [[Context Window]]

## Sources
- Stanford CME 295 (Fall 2025), lecture slides, Afshine and Shervine Amidi (lecture 3, the slide "Beware of context rot!")
- Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, arXiv 2307.03172 (2023): abstract read; the first author is from memory.
- Hong et al., *Context Rot: How Increasing Input Tokens Impacts LLM Performance* (2025): named on the slide, not read.
- No image: I found no free, on-topic picture for this note on Wikimedia Commons, so I added none.
- Every number in the history block was computed by the code in this note (toy data; dates and names marked "from memory" are not checked).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/agents-rag Context rot:::A model can accept a very long text, but it does not use all of it equally well. The longer the text, the more likely it misses the one sentence that matters.

#flashcards/ml/formula #flashcards/topic/agents-rag Write the attention share of one relevant token among distractors and explain each symbol.
?
$$w=\frac{e^{\Delta}}{e^{\Delta}+n}$$
- $w$: the share of attention that goes to the relevant token, between 0 and 1.
- $\Delta$: how much higher its score is than each distractor's score. $e^{\Delta}$: $e$ (about 2.718) to the power $\Delta$, the relevant token's weight before dividing.
- $n$: the number of distractors, each with weight $e^0=1$.
- As $n$ grows, the same advantage gets a smaller and smaller share.

#flashcards/ml/cloze #flashcards/topic/agents-rag With more distractors, the relevant token's attention share ==shrinks==, even with the same ==advantage==.

#flashcards/ml/example #flashcards/topic/agents-rag With $\Delta=3$ ($e^3\approx20.1$), what is the share for $n=10$, 100 and 1000 distractors?::n = 10: 20.1/30.1 = 0.668. n = 100: 20.1/120.1 = 0.167. n = 1000: 20.1/1020.1 = 0.0197. The same relevant token gets 67% of the attention among 10 distractors and about 2% among 1000.
