---
tags: [LLM, evaluation, benchmarks, MMLU, SWE-bench, leaderboard, contamination]
status: growing
level: 4
tier: 3
---

# Benchmarks

Standard tests for language models: question sets and tasks with a fixed way of scoring, so that different models can be compared.

## 🟢 1. Simple (age 12)
How do you compare two AIs? You give them **the same exam** and compare marks. A benchmark is such an exam: a set of questions with right answers, and a scoring rule. Different exams test different skills (facts, maths, coding, safety). But exams can mislead: a small exam gives a shaky mark, an AI that has **seen the answers before** gets an inflated mark, and a single mark never tells the full story, so you still have to try the AI yourself.

**Example:** Two students both score 70% on a 30-question quiz. With so few questions, luck could easily move either score up or down by 8 points, so you cannot be sure who is better.

## 🔵 2. How it works
1. Pick a **skill** to measure and build a test set: questions, tasks, or problems with a way to check the answer automatically.
2. Run each model on it with the same instructions. Compute a score, usually the share correct.
3. Publish the scores so models can be compared in one table or **leaderboard**.
4. The lecture groups common benchmarks by skill: **knowledge** (MMLU), **reasoning** (AIME for maths, PIQA for everyday physical common sense), **coding** (SWE-bench), **safety** (HarmBench) and **agents** ($\tau$-bench).
5. Many benchmarks have **multiple-choice** answers, matched by a hard rule (is it A, B, C or D?). Others check that code passes tests ([[Pass at k]]).
6. Each benchmark is a **projection** of the model onto one axis. Different models are good at different things, so scores form a profile.
7. Cost and speed matter too, so people look at a **Pareto frontier**: models that no other model beats on both quality and cost.
8. Things to watch: **contamination** (the test was in the training data), noise from small test sets, and **Goodhart's law**: when a measure becomes a target it stops being a good measure.
9. The lecture's advice: do not over-rely on benchmarks, use organic human-vote comparisons such as Chatbot Arena as well, and try a few models yourself.

## 🟠 3. Technical detail
- **Benchmarks the lecture covers (figures marked "read" are from the paper abstracts I read):**
  - **MMLU:** four choices per question, 57 tasks (such as elementary maths, US history, computer science, law) (abstract, read). Scored by exact match on A/B/C/D.
  - **AIME:** about 30 maths problems with a three-digit answer (lecture). Needs long reasoning ([[Reasoning Model]]).
  - **PIQA:** two choices per question about physical common sense. Its abstract says humans get 95% and large pretrained models at the time got 77% (read).
  - **SWE-bench:** 2,294 software problems from real GitHub issues in 12 Python repositories. The model must edit the code so that the tests pass. The abstract says the best model then, Claude 2, solved 1.96% (read).
  - **HarmBench:** a framework for automated red teaming and refusal. The lecture says it has 510 harmful behaviours and scores the attack success rate with a classifier (the count is from the slide).
  - **$\tau$-bench:** agents using tools with a simulated user, scored by comparing the final database state with the goal. Its abstract says even strong function-calling agents succeed on under 50% of tasks and are inconsistent (pass^8 under 25% in retail) (read; see [[Pass at k]]).
- **Human-vote leaderboards:** sites such as Chatbot Arena show two answers and let people vote ([[LLM as a Judge]] is the cheap automated cousin). The lecture lists problems: unequal exposure of models, easy to rig, people cannot judge factuality well, personal preference bias and safety penalisation.
- **Contamination defences (lecture):** unique identifiers (such as a hash or canary string) in benchmark files, block-lists for tools (so an agent cannot look up the answers), and newer versions of the tests.
- **Training on the test task:** the lecture cites a paper arguing that comparing models fairly may require fine-tuning each on the test task (I did not read it).
- **Noise:** the code shows how much the measured score moves with the number of questions.

```python
import numpy as np
rng = np.random.RandomState(0)
# 1) how noisy is a benchmark score? A model that truly answers 70% of questions correctly, tested on n questions
p = 0.7
for n in (30, 500, 10000):
    scores = rng.binomial(n, p, 20000) / n
    print(f'n = {n:5d} questions | standard error {np.sqrt(p * (1 - p) / n):.3f} | middle 95% of measured scores {np.percentile(scores, 2.5):.3f} to {np.percentile(scores, 97.5):.3f}')
# 2) what pure guessing scores
for name, k in (('4-choice (like MMLU)', 4), ('2-choice (like PIQA)', 2)):
    guess = rng.randint(0, k, 100000) == 0
    print(f'{name}: guessing gets {guess.mean():.3f} (exactly {1 / k})')
# 3) a Pareto frontier: models that no other model beats on BOTH quality and cost (made-up numbers)
models = {'A': (80, 10.0), 'B': (75, 2.0), 'C': (60, 0.5), 'D': (70, 5.0), 'E': (82, 30.0), 'F': (55, 1.0)}   # (quality, cost)
frontier = [m for m, (q, c) in models.items() if not any(q2 >= q and c2 <= c and (q2, c2) != (q, c) for q2, c2 in models.values())]
print('frontier (best quality for its cost):', frontier, '| dominated:', [m for m in models if m not in frontier])
# 4) a score can be inflated by contamination: if a share c of test questions were seen in training and are always answered right
for c in (0.0, 0.1, 0.3):
    print(f'contaminated share {c:.1f} | true skill 0.70 -> measured {c + (1 - c) * 0.70:.3f}')
```

## 🔴 4. Deep / formal
**How noisy a score is.** If a model truly gets a share $p$ of such questions right, and the test has $n$ independent questions, the measured score has a standard error of:

$$
\text{SE}=\sqrt{\frac{p\,(1-p)}{n}}
$$

**Where:**
- $p$ is the true share of questions the model would get right (its real skill on this kind of question).
- $n$ is the number of questions in the test.
- $\text{SE}$ is the typical size of the random error in the measured score.
- **Matters:** the error shrinks only with the **square root** of the test size: to halve it you need four times the questions. Differences smaller than a couple of SEs may just be luck. (Questions are not perfectly independent, and model answers vary, which adds noise this formula leaves out.)

**Example (tiny numbers):** $p=0.7$.
- $n=30$: $\sqrt{0.21/30}=\mathbf{0.084}$, about 8 points.
- $n=500$: $\sqrt{0.21/500}=\mathbf{0.020}$, about 2 points.
- $n=10{,}000$: $\mathbf{0.005}$.
- Meaning: the code's simulation showed measured scores from $0.533$ to $0.867$ (the middle 95%) at 30 questions, $0.658$ to $0.740$ at 500, and $0.691$ to $0.709$ at 10,000. A 30-question exam like AIME is very noisy for a single run.

**Contamination.** If a share $c$ of the test questions leaked into training and are always answered right, while the model's real skill on the unseen questions is $s$:

$$
\text{measured}=c+(1-c)\,s
$$

**Where:**
- $c$ is the share of contaminated questions (between 0 and 1), and $s$ is the true accuracy on clean questions.
- $c$ counts as fully right, and the remaining $1-c$ questions give accuracy $s$.
- **Matters:** the score is pushed up by an amount $c(1-s)$ with no real skill gain. A model that memorised the test looks better than one that did not, which makes comparisons unfair. The "always right" assumption is my simplification.

**Example (tiny numbers):** $s=0.7$.
- $c=0.1$: $0.1+0.9\times0.7=\mathbf{0.73}$.
- $c=0.3$: $0.3+0.7\times0.7=\mathbf{0.79}$.
- Meaning: with 30% leaked, a 70% model looks like a 79% model, a 9-point gain from nothing.

**Further notes:**
- **Guessing baselines (code):** random guessing gets $0.25$ on four choices and $0.5$ on two, which sets the floor. MMLU's abstract says most models then had near-random accuracy, and the largest GPT-3 improved by almost 20 points over chance (read).
- **Pareto frontier (code):** for six made-up models given as (quality, cost), the frontier was A, B, C, E, and D and F were dominated.
- **Goodhart's law (lecture):** quoted from a 1997 article on audit in British universities; I did not check it.
- **Not covered:** how each benchmark's scoring script works in detail, and significance testing between two models.

**What came before, and how this improved on it:**
- **Before:** task-specific test sets, one per task (for example one for translation, one for entailment), and hand-judged samples of conversations.
- **Its limit:** one test per task does not show breadth. The MMLU abstract notes that models need extensive world knowledge and problem solving across many subjects, which single tasks do not capture.
- **What changed:** broad suites (57 tasks in MMLU), realistic tests (real GitHub issues in SWE-bench), and human-vote arenas for the "feel" of a model.
- **How much better:** the benchmarks' own abstracts show the headroom they exposed: models near random on MMLU, 1.96% on SWE-bench for the best model at the time, and under 50% task success on $\tau$-bench for strong agents. In this note's code, a 30-question test has a standard error of 8.4 points against 0.5 at 10,000 questions, and 30% contamination lifts a 70% model to 79%.

## Related
- [[Evaluation Metrics]] · [[Pass at k]] · [[LLM as a Judge]] · [[Reasoning Model]] · [[LLM]] · [[Agent]] · [[Tool Calling]] · [[Instruction Tuning]] · [[Probability]] · [[RLHF]] · [[Context Rot]] · [[Chatbot Arena]]

## Sources
- Stanford CME 295 (Fall 2025), Lectures 4 and 8 slides, Afshine and Shervine Amidi
- Hendrycks et al., *Measuring Massive Multitask Language Understanding*, 2020 (arXiv 2009.03300, abstract read)
- Jimenez et al., *SWE-bench*, 2023 (arXiv 2310.06770, abstract read)
- Bisk et al., *PIQA*, 2019 (arXiv 1911.11641, abstract read)
- Mazeika et al., *HarmBench*, 2024 (arXiv 2402.04249, abstract read; the 510 figure is from the slide)
- Yao et al., *$\tau$-bench*, 2024 (arXiv 2406.12045, abstract read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation Benchmark:::A fixed exam for AI models: a set of questions with right answers and a scoring rule, so different models can be compared by their marks.

#flashcards/ml/formula #flashcards/topic/evaluation Write the standard error of a benchmark score and explain each symbol.
?
$$\text{SE}=\sqrt{\frac{p\,(1-p)}{n}}$$
- $p$: the true share of questions the model would get right.
- $n$: the number of questions in the test.
- $\text{SE}$: the typical size of the random error in the measured score.
- The error shrinks only with the square root of the test size: to halve it you need four times the questions. Differences under a couple of SEs may be luck. Questions are not perfectly independent and model answers vary, which adds noise this formula leaves out.

#flashcards/ml/example #flashcards/topic/evaluation $p=0.7$. What is the SE at $n=30$, $500$ and $10{,}000$?::$\sqrt{0.21/30}=0.084$ (about 8 points), $\sqrt{0.21/500}=0.020$ (about 2 points), $0.005$ at 10,000. In the note's simulation the middle 95% of scores ran from 0.533 to 0.867 at 30 questions, 0.658 to 0.740 at 500 and 0.691 to 0.709 at 10,000.

#flashcards/ml/formula #flashcards/topic/evaluation Write the score when some test questions leaked into training and explain each symbol.
?
$$\text{measured}=c+(1-c)\,s$$
- $c$: the share of contaminated questions (0 to 1).
- $s$: the true accuracy on clean questions.
- Contaminated questions count as fully right, and the other $1-c$ give accuracy $s$.
- The score is pushed up by $c(1-s)$ with no real skill gain, so a model that memorised the test looks better than one that did not. "Always right" on leaked questions is the note's simplification.

#flashcards/ml/example #flashcards/topic/evaluation True accuracy $s=0.7$. What is the measured score at $c=0.1$ and $c=0.3$?::$c=0.1$: $0.1+0.9\times0.7=0.73$. $c=0.3$: $0.3+0.7\times0.7=0.79$. With 30% leaked, a 70% model looks like a 79% model, a 9-point gain from nothing.

#flashcards/ml/cloze #flashcards/topic/evaluation A benchmark with 30 questions, like AIME, gives a very ==noisy== score from a single run.

#flashcards/ml/concept #flashcards/topic/evaluation Which benchmarks does the lecture group by skill?::Knowledge: MMLU. Reasoning: AIME (maths) and PIQA (everyday physical common sense). Coding: SWE-bench. Safety: HarmBench. Agents: tau-bench.

#flashcards/ml/concept #flashcards/topic/evaluation Name three ways a benchmark score can mislead.::A small test gives a shaky mark, a model that has seen the answers gets an inflated mark, and a single mark never tells the full story (each benchmark is one axis).
