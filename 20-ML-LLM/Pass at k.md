---
tags: [LLM, evaluation, metrics, code-generation, reasoning, pass-at-k]
status: growing
level: 4
tier: 2
---

# Pass at k

Written **pass@k**. The chance that at least one of $k$ tries at a problem works. Its stricter cousin **pass^k** asks that all $k$ tries work.

## 🟢 1. Simple (age 12)
An AI tries a puzzle and sometimes gets it right, sometimes wrong. **Pass@k** asks: "if we let it try $k$ times, how likely is it that **at least one** try is right?" It is like a basketball player taking 5 shots: even a player who only scores 30% of the time will usually score at least once in 5 shots. The stricter version, **pass^k**, asks for **all** shots to go in.

**Example:** A model solves a coding puzzle 3 times out of 10. With one try the chance is 30%. With 5 tries the chance that at least one works is about 92%.

## 🔵 2. How it works
1. For tasks like code and maths, the answer can be checked by a program: run the unit tests, or compare with the known answer ([[Reasoning Model]]).
2. Ask the model for several answers to the same problem and check each one.
3. **Pass@1:** one try. It is the right measure when a user only gets one answer.
4. **Pass@k:** is any of $k$ tries right? It suits cases where checking is easy and a little extra delay is fine (you run all $k$ and keep one that passes).
5. **Cons@k** ("consensus at $k$"): take the majority answer of $k$ tries and check just that one, as in self-consistency ([[Chain of Thought]]).
6. **Pass^k:** do **all** $k$ tries succeed? It measures *reliability*, for example for an agent that must work every time ([[Agent]]).
7. To avoid wasting effort, we do not need to try exactly $k$ times for every score. We take $n$ samples (more than $k$), count how many were right, and use a formula that estimates pass@k fairly.

## 🟠 3. Technical detail
- **Origin of pass@k:** the Codex paper (Chen et al., 2021) used it to measure whether generated code passes unit tests. Its abstract reports 28.8% solved with one sample and 70.2% with 100 samples per problem (arXiv 2107.03374, read).
- **Why a special formula:** the naive way (draw exactly $k$ samples per problem, once) is noisy. Drawing $n>k$ samples and computing the chance that a random group of $k$ of them contains a correct one uses all the samples and has less variance. The formula is in level 4; the paper's version is from memory, and my code checks it against simulation.
- **Pass@k rises with $k$** and approaches 1 for any problem the model can solve at all. So large $k$ measures "can it ever", while small $k$ measures "does it usually".
- **Pass^k falls with $k$** and measures "does it always". The lecture introduces pass^k with $\tau$-bench, an agent benchmark, for consistency and reliability.
- **Use:** pass@1 for single-shot products; pass@k when verification is cheap; cons@k when there is no verifier but answers can be compared; pass^k for agents.

```python
import numpy as np
from math import comb
rng = np.random.RandomState(0)
def pass_at_k(n, c, k):
    """unbiased estimate from n samples of which c were correct: 1 - C(n-c, k) / C(n, k)"""
    return 1.0 if n - c < k else 1 - comb(n - c, k) / comb(n, k)
n, c = 10, 3
print('10 samples, 3 correct | pass@1', round(pass_at_k(n, c, 1), 4), '| pass@5', round(pass_at_k(n, c, 5), 4), '| pass@10', pass_at_k(n, c, 10))
print('by hand for pass@5: 1 - C(7,5)/C(10,5) = 1 -', comb(7, 5), '/', comb(10, 5), '=', round(1 - comb(7, 5) / comb(10, 5), 4))
# a problem the model solves with chance p each time: truth is 1-(1-p)^k ; the estimator from only 10 samples should average to it
p = 0.3
for k in (1, 5):
    est = np.mean([pass_at_k(10, int((rng.rand(10) < p).sum()), k) for _ in range(20000)])
    print(f'p = {p}, k = {k} | true pass@k {1 - (1 - p) ** k:.4f} | mean of the estimator {est:.4f}')
# the stricter pass^k: all k attempts must succeed
for k in (1, 5, 10):
    print(f'p = {p}, k = {k} | pass@k {1 - (1 - p) ** k:.4f} | pass^k {p ** k:.6f}')
p = 0.9
print('p = 0.9, k = 8 | pass@k', round(1 - (1 - p) ** 8, 8), '| pass^k', round(p ** 8, 4))
```

## 🔴 4. Deep / formal
**Estimating pass@k from $n$ samples.** If $c$ of $n$ samples were correct:

$$
\widehat{\text{pass@}k}=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}
$$

**Where:**
- $n$ is how many samples were drawn for the problem and $c$ is how many of them passed.
- $k$ is the number of tries we are asking about, with $k\le n$.
- $\binom{n}{k}$ counts all the groups of $k$ samples, and $\binom{n-c}{k}$ counts the groups made only of wrong samples.
- The fraction is the chance that a random group of $k$ is all wrong, and one minus it is the chance of at least one right.
- **Matters:** it is an *unbiased* estimate: averaged over many repetitions it equals the true value (the code checks this). If $n-c<k$ there is no all-wrong group, so the estimate is 1.

**Example (tiny numbers):** $n=10$ samples, $c=3$ correct, $k=5$.
- Wrong samples: $n-c=7$. Groups of 5 from 7 wrong ones: $\binom75=\mathbf{21}$.
- Groups of 5 from all 10: $\binom{10}{5}=\mathbf{252}$.
- $1-21/252=1-0.0833=\mathbf{0.9167}$.
- Meaning: a random set of 5 of these samples contains a correct one about 92% of the time, and for $k=1$ the estimate is $c/n=0.3$.

**The true value for a fixed chance.** If every try is right with chance $p$, independently:

$$
\text{pass@}k=1-(1-p)^{k},\qquad \text{pass}^{k}=p^{k}
$$

**Where:**
- $p$ is the chance a single try succeeds, and $1-p$ the chance it fails.
- $(1-p)^k$ is the chance that all $k$ tries fail, so one minus it is the chance that at least one works.
- $p^k$ is the chance that all $k$ tries work.
- **Matters:** the two measures move in opposite directions with $k$. A model with $p=0.9$ has pass@8 of nearly 1 and pass^8 of only $0.43$. Independence is an assumption: real tries are linked (the same weakness repeats).

**Example (tiny numbers):** $p=0.3$, $k=5$, then $p=0.9$, $k=8$.
- pass@5 $=1-0.7^5=1-0.168=\mathbf{0.8319}$. The code's estimator, averaged over 20,000 repeats of 10 samples, gave $0.8364$.
- pass^5 $=0.3^5=\mathbf{0.00243}$.
- For $p=0.9$: pass^8 $=0.9^8=\mathbf{0.4305}$.
- Meaning: a model that is right 90% of the time on each try still fails to be right on all of 8 tries more than half the time.

**Further notes:**
- **Sampling noise:** with 10 samples, the pass@5 estimate for a problem with $p=0.3$ landed at $0.8364$ on average and not exactly $0.8319$, because 20,000 repeats still carry a little noise.
- **What it hides:** pass@k says nothing about how expensive finding the right answer among $k$ is, or whether a user can tell which one is right.

**What came before, and how this improved on it:**
- **Before:** judging generated code or text by how similar it is to a reference (word overlap such as BLEU, named in the lecture), which can mark a working but different program as wrong.
- **Its limit:** two programs can look different and both work, or look alike and one fail.
- **What changed:** run the code against tests and count a problem solved if a sampled program passes (Chen et al., arXiv 2107.03374, 7 July 2021, abstract read).
- **How much better:** the Codex abstract shows why $k$ matters: 28.8% with one sample against 70.2% with 100. In this note's code the same sample of 10 gives pass@1 $=0.3$ and pass@5 $=0.9167$, and the estimator matched the true value within $0.005$ over 20,000 repeats.

## Related
- [[Evaluation Metrics]] · [[Reasoning Model]] · [[Chain of Thought]] · [[GRPO]] · [[Agent]] · [[Probability]] · [[LLM]] · [[Tool Calling]] · [[Benchmarks]]

## Sources
- Stanford CME 295 (Fall 2025), Lectures 6 and 8 slides, Afshine and Shervine Amidi
- Chen et al., *Evaluating Large Language Models Trained on Code*, 2021 (arXiv 2107.03374, abstract read)
- Yao et al., *τ-bench*, 2024 (named in the lecture; not read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation Pass@k:::The chance that at least one of $k$ tries at a problem is right.

#flashcards/ml/reverse #flashcards/topic/evaluation Pass^k:::The chance that all $k$ tries are right. It measures reliability, for example for an agent that must work every time.

#flashcards/ml/formula #flashcards/topic/evaluation Write the unbiased estimate of pass@k from $n$ samples and explain each symbol.
?
$$\widehat{\text{pass@}k}=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}$$
- $n$: how many samples were drawn for the problem. $c$: how many of them passed.
- $k$: the number of tries asked about, with $k\le n$.
- $\binom{n}{k}$ counts all groups of $k$ samples. $\binom{n-c}{k}$ counts the groups made only of wrong samples.
- The fraction is the chance a random group of $k$ is all wrong, so one minus it is the chance of at least one right.
- It is unbiased (averaged over many repeats it equals the true value). If $n-c<k$ there is no all-wrong group, so the estimate is 1.

#flashcards/ml/example #flashcards/topic/evaluation $n=10$ samples, $c=3$ correct, $k=5$. What is pass@5?::$\binom75=21$ all-wrong groups out of $\binom{10}{5}=252$. $1-21/252=1-0.0833=0.9167$. A random set of 5 contains a correct one about 92% of the time. For $k=1$ the estimate is $c/n=0.3$.

#flashcards/ml/formula #flashcards/topic/evaluation Write pass@k and pass^k when each try succeeds independently with chance $p$, and explain each symbol.
?
$$\text{pass@}k=1-(1-p)^{k},\qquad \text{pass}^{k}=p^{k}$$
- $p$: the chance a single try succeeds. $1-p$: the chance it fails.
- $(1-p)^k$: all $k$ tries fail, so one minus it is at least one works.
- $p^k$: all $k$ tries work.
- The two move in opposite directions as $k$ grows. Independence is an assumption: real tries are linked because the same weakness repeats.

#flashcards/ml/example #flashcards/topic/evaluation $p=0.3$, $k=5$: pass@5 and pass^5? And $p=0.9$, $k=8$?::pass@5 $=1-0.7^5=0.8319$. pass^5 $=0.3^5=0.00243$. For $p=0.9$: pass^8 $=0.9^8=0.4305$. A model right 90% of the time per try still fails to be right on all 8 tries more than half the time.

#flashcards/ml/cloze #flashcards/topic/evaluation Pass@1 is the right measure when a user gets only ==one== answer. Pass^k asks that ==all== $k$ tries succeed.

#flashcards/ml/reverse #flashcards/topic/evaluation Cons@k:::Take the majority answer of $k$ tries and check just that one (as in self-consistency).
