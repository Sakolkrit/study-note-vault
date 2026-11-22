---
tags: [LLM, prompting, reasoning, chain-of-thought, self-consistency]
status: growing
level: 4
tier: 2
---

# Chain of Thought

Often shortened to **CoT**. Asking a model to write out its reasoning steps before giving the answer.

## 🟢 1. Simple (age 12)
If you ask a friend "what is 17 × 24?" and demand an instant answer, they may blurt out a wrong number. If you let them **show their working** ("17 × 20 is 340, 17 × 4 is 68, so 408"), they get it right far more often. Chain of thought is letting the AI show its working, and the working itself helps it reach the answer.

**Example:** "A bear was born in 2020. How old will it be next year?" Instead of just answering "5", the AI writes: "The bear is 4 now (2024 − 2020), so next year it will be 5." Each small step is easy, and the steps together are right.

## 🔵 2. How it works
1. An [[LLM]] writes one token at a time, and each token can only use a fixed amount of "thinking" ([[Transformer]]). A hard question may need more thinking than one token allows.
2. Written steps give it more room: every step it writes becomes part of the input for the next token, so the model can build on its own earlier work.
3. **Few-shot chain of thought:** the prompt includes a few examples that show question, reasoning and answer ([[Prompt Engineering]]). The model copies the style.
4. **Zero-shot style:** you only ask it to reason step by step (a common trick; not covered in the lecture).
5. **Self-consistency:** sample several different reasoning chains with some randomness ([[Decoding Strategies]]), then take the answer that most chains agree on (a majority vote).
6. The prices are more tokens (more cost and delay) and answers that may *look* convincing even when a step is wrong.
7. The newest [[Reasoning Model]]s are trained to do this at a much larger scale, by themselves.

## 🟠 3. Technical detail
- **What the lecture says:** explaining reasoning improves performance; it also gives interpretability and an explanation, but costs more tokens and so more latency. Self-consistency trades performance against added cost.
- **When it helps:** tasks with several steps (arithmetic word problems, logic, code). The CoT paper's abstract says these abilities emerge in sufficiently large models.
- **Example result:** prompting a 540-billion-parameter model with 8 chain-of-thought examples reached the best published accuracy on GSM8K, a set of maths word problems, surpassing a fine-tuned GPT-3 with a verifier (arXiv 2201.11903 abstract, read).
- **Self-consistency result:** the paper's abstract reports gains over plain chain of thought of +17.9% on GSM8K, +11.0% on SVAMP, +12.2% on AQuA, +6.4% on StrategyQA and +3.9% on ARC-challenge (arXiv 2203.11171 abstract, read).
- **How to vote:** only the *final answers* are compared, not the reasoning text, so different working can still agree.
- **Why voting can fail:** if the samples all make the same mistake, the vote repeats it. The code shows how much the gain shrinks when questions differ in difficulty.
- **Faithfulness:** the written reasoning is not guaranteed to be the real cause of the answer (a research topic; from memory, not covered in the lecture).

```python
import numpy as np
from math import comb
rng = np.random.RandomState(0)

def vote_acc(p, n):
    """chance that the majority of n independent samples is right, when each sample is right with chance p and the only other choice is one wrong answer"""
    return sum(comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n // 2 + 1, n + 1))

p = 0.6
for n in (1, 5, 11, 21, 51):
    sim = np.mean([(rng.rand(n) < p).sum() > n / 2 for _ in range(20000)])
    print(f'n = {n:2d} samples | majority vote right: formula {vote_acc(p, n):.4f} | simulated {sim:.4f} | cost: {n} times the tokens')
# the same average chance (0.6), but made of easy questions (0.9) and hard ones (0.3): the samples are no longer independent across questions
for n in (1, 5, 21, 51):
    mixed = 0.5 * vote_acc(0.9, n) + 0.5 * vote_acc(0.3, n)
    print(f'n = {n:2d} | easy half 0.9 and hard half 0.3 | majority vote right: {mixed:.4f}')
```

## 🔴 4. Deep / formal
**Majority vote.** Suppose each of $n$ sampled chains (with $n$ odd) reaches the right answer independently with chance $p$, and the only other outcome is one wrong answer:

$$
P(\text{vote right})=\sum_{k=\lceil n/2\rceil}^{n}\binom{n}{k}p^{k}(1-p)^{n-k}
$$

**Where:**
- $n$ is the number of sampled chains and $k$ is how many of them are right.
- $p$ is the chance a single chain is right.
- $\binom{n}{k}$ counts the ways to choose which $k$ chains are right, and $p^k(1-p)^{n-k}$ is the chance of one such pattern.
- The sum adds up every case where more than half are right.
- **Matters:** if $p>0.5$, more samples push the vote toward certainty. If $p<0.5$, more samples make the vote *worse*. The assumption of independence is strong: real samples from one model on one question are linked.

**Example (tiny numbers):** $n=3$, $p=0.6$.
- Exactly 2 right: $\binom32\,0.6^2\,0.4=3\times0.36\times0.4=\mathbf{0.432}$.
- All 3 right: $0.6^3=\mathbf{0.216}$.
- Sum: $\mathbf{0.648}$.
- Meaning: three votes lift a 60% single-chain accuracy to 64.8%. The code gives $0.6826$ for 5 chains, $0.8256$ for 21 and $0.9265$ for 51.

**Further notes:**
- **Easy and hard questions:** with the same overall chance of $0.6$ made of easy questions at $0.9$ and hard ones at $0.3$, the code's vote gives only $0.5773$ at 5 chains, $0.5132$ at 21 and $0.5007$ at 51. On the hard half the vote converges to *wrong*, on the easy half to right, so more votes do not rescue hard questions. This is a toy with made-up numbers, not a measurement of any model.
- **Cost:** $n$ chains cost about $n$ times the tokens.
- **Not covered:** tree-of-thought, process rewards, and what happens when wrong answers differ (then a plurality can win even if $p<0.5$).

**What came before, and how this improved on it:**
- **Before:** asking for the answer directly, or giving a few question and answer examples without reasoning (few-shot prompting, Brown et al. 2020, named in the lecture; not read here).
- **Its limit:** multi-step problems were weak points for large models (the CoT abstract frames it as a gap for arithmetic, commonsense and symbolic reasoning).
- **What changed:** show worked steps in the examples (Wei et al., arXiv 2201.11903, 28 January 2022, abstract read), then sample many chains and vote (Wang et al., arXiv 2203.11171, 21 March 2022, abstract read).
- **How much better:** published: state of the art on GSM8K with a 540B model, and self-consistency adds up to +17.9% on GSM8K. In this note's code, under the independence assumption a 60% chain becomes 82.6% with 21 votes, but with mixed easy and hard questions it only reaches 51.3%. I did not run a real model.

## Related
- [[Prompt Engineering]] · [[LLM]] · [[Reasoning Model]] · [[Decoding Strategies]] · [[Pass at k]] · [[Probability]] · [[GRPO]] · [[ReAct]] · [[Transformer]] · [[Evaluation Metrics]] · [[In-Context Learning]]

## Sources
- Stanford CME 295 (Fall 2025), Lectures 3 and 6 slides, Afshine and Shervine Amidi
- Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*, 2022 (arXiv 2201.11903, abstract read)
- Wang et al., *Self-Consistency Improves Chain of Thought Reasoning in Language Models*, 2022 (arXiv 2203.11171, abstract read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm Chain of thought:::Letting the AI show its working step by step, and the working itself helps it reach the right answer.

#flashcards/ml/formula #flashcards/topic/llm Write the chance that a majority vote over sampled chains is right, and explain each symbol.
?
$$P(\text{vote right})=\sum_{k=\lceil n/2\rceil}^{n}\binom{n}{k}p^{k}(1-p)^{n-k}$$
- $n$: the number of sampled chains. $k$: how many of them are right.
- $p$: the chance a single chain is right.
- $\binom{n}{k}$: the number of ways to choose which $k$ chains are right. $p^k(1-p)^{n-k}$: the chance of one such pattern.
- The sum adds up every case where at least half are right.
- If $p>0.5$, more samples push the vote's accuracy up. If $p<0.5$ they push it down.

#flashcards/ml/cloze #flashcards/topic/llm Voting over many chains raises accuracy only if one chain is right more than ==half== the time.

#flashcards/ml/example #flashcards/topic/llm $n=3$, $p=0.6$. What is the chance the majority vote is right?::Exactly 2 right: 3 × 0.36 × 0.4 = 0.432. All 3 right: 0.216. Sum 0.648. Three votes lift a 60% single-chain accuracy to 64.8%. The note's code: 5 chains 0.6826, 21 chains 0.8256, 51 chains 0.9265.
