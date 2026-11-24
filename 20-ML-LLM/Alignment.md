---
tags: [alignment, RLHF, safety, LLM, reward-model, instruction-tuning, ML]
status: growing
level: 4
tier: 4
---

# Alignment

**Alignment** means making an AI system do what its users actually *intend* and what is good for them, instead of only what it was literally trained to do. For a [[LLM|language model]] the gap is easy to see: it was trained to continue text, but people want it to follow instructions, tell the truth and avoid harm. This note explains the problem, the main fixes in this vault ([[Instruction Tuning]], [[RLHF]], [[Direct Preference Optimization]]) and one failure that every fix must avoid: optimising a flawed score.

## 🟢 1. Simple (age 12)
Imagine a genie that gives you exactly what you say, not what you mean. You ask for "a big pile of homework done fast" and get 500 pages of scribbles. Alignment is teaching the genie to understand what you *really* want.

**Example:** You ask a robot to "make the room tidy". A badly aligned robot throws everything out of the window, and the room is empty and "tidy". A well aligned robot puts things where they belong.

## 🔵 2. How it works
1. **The gap.** A language model is first trained ([[Pre-training]]) to guess the next word of web text. That goal is not "help this person". So the model may ramble, make things up ([[Hallucination]]) or say harmful things.
2. **What we want instead.** The InstructGPT paper asks for three things, following earlier work (paper, 1): **helpful** (solves the user's task), **honest** (does not make things up) and **harmless** (does not cause harm).
3. **Show it good answers.** People write ideal answers and the model is fine-tuned on them ([[Instruction Tuning]]).
4. **Let people rank.** People rank several answers to the same question. A [[Reward Model]] is trained to give higher scores to the answers people prefer ([[Bradley-Terry Model]]).
5. **Improve against the scorer.** The model is trained to get high scores ([[RLHF]], with [[PPO]]), or directly from the rankings without a separate scorer ([[Direct Preference Optimization]]).
6. **Keep a leash.** The model is punished for drifting too far from its starting version, because otherwise it learns to trick the scorer. That is the danger shown in the toy code below.

![[rlhf-diagram.svg]]
*The RLHF pipeline: a supervised model writes answers, human annotators rank them, the ranking trains a reward model, and the reward model guides training (with PPO) to produce the aligned model. Image: PopoDameron, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:RLHF_diagram.svg), CC BY-SA 4.0 (the same image as in [[RLHF]]).*

## 🟠 3. Technical detail
Numbers about InstructGPT come from my note [[InstructGPT 2022]], which was built from the paper's text. The table below comes from my own toy code, which is not a language model.

- **What the paper reports.** Labelers preferred 175B InstructGPT outputs to 175B GPT-3 outputs 85 ± 3% of the time. The abstract says outputs of the 1.3B InstructGPT model are preferred to those of 175B GPT-3, "despite having 100x fewer parameters" (my arithmetic: 175 divided by 1.3 is 134.6, so "100x" is conservative). The paper also reports that RLHF made some public benchmarks worse, "an example of an alignment tax", and that mixing in some pre-training data (PPO-ptx) greatly reduces it (all from the paper, via my note).
- **The toy.** The code makes a pool of $n$ candidate answers. Each answer has a real quality $q$ and a length $L$. The reward model is flawed: it scores $q+L$, so it loves long answers. What we truly want is $q-0.3L$ (quality, but waffle annoys). We keep the answer the flawed scorer likes best ("best of $n$"). This is *not* RLHF, just the simplest way to optimise against a score.

| Pool size $n$ | Score the scorer gives | True value |
|---|---|---|
| 1 | 1.39 | -0.41 |
| 2 | 2.24 | -0.14 |
| 4 | 3.08 | -0.06 |
| 16 | 4.93 | -0.45 |
| 64 | 7.31 | -1.35 |
| 256 | 10.38 | -2.47 |
| 1024 | 14.18 | -3.74 |

Table written by me from my own run (20,000 trials per row, fixed seed). *Pool size* is how many answers were generated before the best-scoring one was kept. *Score the scorer gives* is the average flawed score of the kept answer. *True value* is the average of $q-0.3L$ for the kept answer.

**Summary: what the table shows.** The scorer's number rises all the way, from 1.39 to 14.18, but the true value improves only up to a small pool (-0.41 to -0.06 at $n=4$) and then falls to -3.74, below where it started. More optimisation against a flawed score made the real result worse. **Caution:** this is a made-up toy with one chosen flaw. It shows the mechanism, not how large the effect is in real systems.
- **A good scorer does not do this.** With a scorer that ignores length, the true value kept rising: -0.41, 0.61, 1.35, 2.41 for $n=1, 4, 16, 256$. So the problem here comes from the flaw in the scorer, not from selecting the best.
- **Limits:** one seed (`default_rng(0)`), one flaw, no training. I did not test any real reward model, any real language model, or the strength of a KL leash.

```python
import numpy as np

rng = np.random.default_rng(0)

def best_of_n(n, proxy_likes_length, trials=20000):
    q = rng.normal(size=(trials, n))                  # real quality of each candidate answer
    L = rng.lognormal(0, 0.8, size=(trials, n))       # length of each candidate answer
    proxy = q + (1.0 if proxy_likes_length else 0.0) * L   # what the (flawed) reward model scores
    true = q - 0.3 * L                                # what we actually want: quality, but waffle annoys
    pick = proxy.argmax(1); r = np.arange(trials)     # keep the answer the reward model likes best
    return proxy[r, pick].mean(), true[r, pick].mean()

print('flawed scorer (likes length)')
print('   n   score the scorer gives   true value')
for n in (1, 2, 4, 16, 64, 256, 1024):
    p, t = best_of_n(n, True); print(f'{n:5d} {p:12.2f} {t:18.2f}')
print('honest scorer (ignores length): true value for n = 1, 4, 16, 256 ->', [round(best_of_n(n, False)[1], 2) for n in (1, 4, 16, 256)])
```

## 🔴 4. Deep / formal
**The leash on the reward.** In RLHF the model $\pi$ is trained to get high reward but is charged for moving away from the reference model $\pi_{\text{ref}}$ (the fine-tuned model it started from). A simplified form (the InstructGPT objective in my [[InstructGPT 2022]] note has one more term for the pre-training mix):

$$
\max_{\pi}\ \ \mathbb{E}_{x,\;y\sim\pi(\cdot\mid x)}\Big[\, r(x,y)\;-\;\beta\,\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} \Big]
$$

**Where:**
- $x$ is the prompt and $y$ the model's answer.
- $r(x,y)$ is the reward model's score. **Matters:** it is the only link to what people prefer; if it is flawed the model learns the flaw (the toy table above).
- $\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$ measures how much more likely the new model makes this answer than the reference did. Its average over answers is the [[KL Divergence]].
- $\beta$ is the strength of the leash (the InstructGPT paper uses 0.02 for its PPO runs, from my note). **Matters:** $\beta=0$ lets the model chase the score freely; a large $\beta$ keeps it close to the reference and so limits both gains and damage.

**Example (tiny numbers):** reward $r=3.0$, log ratio $=2.0$ (the model makes this answer $e^{2}\approx7.4$ times as likely as the reference did). With $\beta=0.5$: $3.0-0.5\times2.0=2.0$. With $\beta=0$ the score would be 3.0, and with $\beta=1$ it would be 1.0 (my arithmetic). The answer means: an answer that is very different from what the reference model would write must earn much more reward to be worth it.

**What is open (general knowledge, not from a source in this note):**
- *What do "people want"?* Labelers disagree, and the paper's labelers were a small group; a model aligned to one group is not aligned to everyone. The InstructGPT paper itself discusses whose preferences are used (I did not reread that section for this note).
- *Flawed scores.* My toy shows one way a scorer can mislead. Real reward models may have other flaws, and measuring them is hard.
- *Inspecting what the model learned.* See [[Interpretability]]; knowing a model gives good answers is not the same as knowing why.
- *Attacks and misuse.* See [[Prompt Injection]] and [[Agent Safety]]: an aligned model can still be tricked or misused.
- *Cost.* The paper's alignment tax shows that gains on the target can cost accuracy elsewhere unless mitigated.

**What came before, and how this improved on it:**
- **Before:** plain pre-trained language models such as GPT-3 (2020), which were prompted with clever examples (few-shot) to behave. (Dates from my notes [[GPT-3 2020]] and [[InstructGPT 2022]].)
- **Its limit:** the paper says the next-word objective is "misaligned" with following instructions helpfully and safely, so large models can be "untruthful, toxic, or simply not helpful" (paper, abstract and 1).
- **What changed:** fine-tune on human-written answers, learn a reward model from human rankings, and optimise the model against it with a leash ([[RLHF]]), or learn directly from the rankings ([[Direct Preference Optimization]]).
- **How much better:** published: labelers preferred 175B InstructGPT to 175B GPT-3 85 ± 3% of the time and a 1.3B InstructGPT to the 175B GPT-3 (paper). My toy shows the other side: the same idea applied to a flawed scorer got worse as optimisation grew (true value -0.06 at $n=4$ to -3.74 at $n=1024$).

## Related
[[RLHF]] · [[Reward Model]] · [[Instruction Tuning]] · [[Direct Preference Optimization]] · [[PPO]] · [[Bradley-Terry Model]] · [[KL Divergence]] · [[Pre-training]] · [[Hallucination]] · [[Interpretability]] · [[Prompt Injection]] · [[Agent Safety]] · [[InstructGPT 2022]] · [[GPT-3 2020]] · [[LLM]]

## Sources
- Ouyang et al., *Training language models to follow instructions with human feedback* (2022): the three goals (helpful, honest, harmless), the quotes, the 85 ± 3% preference and the alignment tax, all through my note [[InstructGPT 2022]], which was built from the paper's text. I did not reread the paper for this note.
- My own toy run (Python 3.9.6, NumPy 1.24.4, CPU).
- Image: [RLHF_diagram.svg](https://commons.wikimedia.org/wiki/File:RLHF_diagram.svg) (PopoDameron, CC BY-SA 4.0), already used in [[RLHF]].
- Not checked: the note was not opened in Obsidian. The "what is open" list is general knowledge, not sourced.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/alignment-rl Alignment:::Making an AI do what its users actually intend and what is good for them, not only what it was literally trained to do.

#flashcards/ml/concept #flashcards/topic/alignment-rl Why is a pre-trained language model not automatically aligned?::It is trained to predict the next word of web text, which "is different from" following the user's instructions helpfully and safely, so it can be untruthful, toxic or unhelpful (InstructGPT paper, 1).

#flashcards/ml/concept #flashcards/topic/alignment-rl What are the three goals the InstructGPT paper sets for an aligned model?::Helpful (solves the task), honest (does not fabricate) and harmless (does not cause harm), following earlier work (paper, 1).

#flashcards/ml/concept #flashcards/topic/alignment-rl In order, what are the steps of aligning a model with human feedback?::Fine-tune on human-written answers; have people rank answers and train a reward model; optimise the model against the reward model with a leash that keeps it near its starting version (RLHF), or learn directly from rankings (DPO).

#flashcards/ml/concept #flashcards/topic/alignment-rl What is the "alignment tax"?::The paper's name for performance regressions on public benchmarks (such as SQuAD and DROP) caused by RLHF; mixing in pre-training data (PPO-ptx) greatly reduces them (paper, 1 and 4.2).

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the simplified RLHF objective with the leash and explain each symbol.
?
$$\max_{\pi}\ \ \mathbb{E}_{x,\;y\sim\pi(\cdot\mid x)}\Big[\, r(x,y)\;-\;\beta\,\log\frac{\pi(y\mid x)}{\pi_{\text{ref}}(y\mid x)} \Big]$$
- $x$: the prompt; $y$: the model's answer.
- $r(x,y)$: the reward model's score; if it is flawed the model learns the flaw.
- $\log\frac{\pi}{\pi_{\text{ref}}}$: how much more likely the new model makes this answer than the reference did; its average is the KL divergence.
- $\beta$: the strength of the leash (0.02 in the InstructGPT PPO runs, from my note). $\beta=0$ lets the model chase the score freely.

#flashcards/ml/example #flashcards/topic/alignment-rl With $r=3.0$, log ratio $2.0$ and $\beta=0.5$, what is the leashed objective? What for $\beta=0$ and $\beta=1$?::$3.0-0.5\times2.0=2.0$; for $\beta=0$ it is 3.0; for $\beta=1$ it is 1.0 (my arithmetic).

#flashcards/ml/concept #flashcards/topic/alignment-rl Why does RLHF keep the model near its starting version?::Without the leash it can learn to trick the reward model instead of being truly helpful. In my toy the true value fell from -0.06 to -3.74 while the scorer's number rose, as optimisation grew.

#flashcards/ml/concept #flashcards/topic/alignment-rl What did my toy simulation show about optimising a flawed score?::The scorer's number rose from 1.39 to 14.18 as the pool grew from 1 to 1024, but the true value peaked near $n=4$ (-0.06) and fell to -3.74. With a scorer that ignored length, it kept rising (2.41 at $n=256$). The toy is made up, not a real model.

#flashcards/ml/concept #flashcards/topic/alignment-rl How big a result did the InstructGPT paper report for alignment by human feedback?::Labelers preferred 175B InstructGPT to 175B GPT-3 85 ± 3% of the time, and the abstract says a 1.3B InstructGPT is preferred to the 175B GPT-3 despite "100x fewer parameters" (my arithmetic: 134.6 times fewer).

#flashcards/ml/cloze #flashcards/topic/alignment-rl The InstructGPT paper aims for models that are ==helpful==, ==honest== and ==harmless==.

#flashcards/ml/image #flashcards/topic/alignment-rl ![[rlhf-diagram.svg]]
What does this diagram show?
?
The RLHF pipeline: a supervised model writes answers, human annotators rank them, the ranking trains a reward model, and the reward model guides training (with PPO) to produce the aligned model. (PopoDameron, CC BY-SA 4.0.)
