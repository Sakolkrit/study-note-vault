---
tags: [lecture, course, CME295, LLM]
course: CME 295
lecture: 6
lecturer: Afshine Amidi and Shervine Amidi
term: Fall 2025
source: Stanford CME 295 lecture 6 slides
status: growing
---

# CME 295 Lecture 6: Reasoning models

> **In one line:** Explains how models are trained to think before answering: reinforcement learning on checkable rewards, the GRPO algorithm, and the DeepSeek-R1 recipe. (The lecture title is my own label, not the course's.)

## What the lecture covers
1. **Recap and the limits of plain LLMs.** Preference tuning and PPO, then a list of strengths and caveats of vanilla LLMs.
2. **What reasoning means.** The ability to solve a problem that needs several steps; the core idea is to teach the model to explain its steps before the answer. Reasoning-model releases, and how to spot one. See [[Reasoning Model]] and [[Chain of Thought]].
3. **Benchmarks and metrics for reasoning.** Coding and maths tasks, and the scores pass@k (the probability that at least one of k tries is right) and consensus at k. See [[Pass at k]] and [[Benchmarks]].
4. **Test-time scaling and RL.** Give the model room to reason before it answers, and train that with reinforcement learning. Two rewards: one checks that a reasoning chain is present (inside think tags), the other checks that the final solution is correct (a code check or a maths check), so the reward is "verifiable".
5. **Controlling the thinking at inference time.** Not all prompts need the same amount of thinking, so the length can be controlled.
6. **GRPO.** Group Relative Policy Optimization, compared with PPO. See [[GRPO]].
7. **The "increasing output length" phenomenon** and ways to mitigate it, and other adjustments such as the bias linked to question difficulty.
8. **The DeepSeek recipes.** R1-Zero (RL on a base model) and R1 (a longer recipe that adds fine-tuning stages and a reward for helpfulness and harmlessness), with results. See [[DeepSeek-R1 2025]].
9. **Distillation.** Passing reasoning to small models by distilling from the big one, linked back to lecture 2. See [[Knowledge Distillation]].

## Key ideas to remember
- A reasoning model writes out its steps before the answer, and RL on checkable rewards teaches it to do so ([[Reasoning Model]]).
- GRPO replaces PPO's learned value baseline with a comparison inside a group of answers ([[GRPO]]).
- Spending more tokens at test time can raise accuracy, which is test-time scaling.
- Small models can inherit the reasoning by distillation ([[Knowledge Distillation]]).

## Papers named in the lecture
- "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" ([[DeepSeek-R1 2025]])
- "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" (the GRPO paper, named on the slides)
- "Understanding R1-Zero-Like Training" (named on the slides)
- "DeepSeek-V3 Technical Report" (named on the slides)
- "Evaluating Large Language Models Trained on Code" (the pass@k estimator, named on the slides)

## Questions I still have
- Why does the output length grow during RL, and is a longer answer really a better one?

## What this note is built from
- The lecture slides (PDF), read as extracted text. I did not see the figures, any speaker notes or the recordings, and I did not open the course playlist.
- Topic order and wording of the sections follow the slides. The detail and the tested code live in the concept notes linked above, each of which says what it was checked against.
- Where a paper is only "named on the slides", I did not read it for this note. The concept and paper notes say which abstracts or bodies I did read.

## Course
- Part of [[CME 295]]. Lecture before: [[CME 295 Lecture 5]]. Lecture after: [[CME 295 Lecture 7]].
- Concept notes this lecture feeds: [[Reasoning Model]], [[GRPO]], [[Pass at k]], [[Chain of Thought]], [[Knowledge Distillation]]

## Flashcards
#flashcards/courses

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 6: what does it cover?::What reasoning means, benchmarks and metrics (pass@k), test-time scaling with RL, controlling the thinking, GRPO, the output-length phenomenon, the DeepSeek recipes (R1-Zero and R1), and distillation.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 6: what is the core idea of a reasoning model?::Teach the model to explain its steps before the answer. RL on checkable rewards teaches it to do so.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 6: what are the two rewards in reasoning RL?::One checks that a reasoning chain is present (inside think tags), the other checks that the final solution is correct (a code check or a maths check), so the reward is verifiable.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 6: what are pass@k and consensus at k?::Pass@k is the probability that at least one of k tries is right. Consensus at k takes the majority answer of k tries.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 6: how does GRPO differ from PPO?::GRPO replaces PPO's learned value baseline with a comparison inside a group of answers.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 6: what is test-time scaling?::Spending more tokens at test time (room to reason before answering) can raise accuracy.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 6: R1-Zero against R1?::R1-Zero is RL on a base model. R1 is a longer recipe that adds fine-tuning stages and a reward for helpfulness and harmlessness.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 6: which papers are named?::DeepSeek-R1 (has a paper note), and named on slides only: DeepSeekMath (the GRPO paper), "Understanding R1-Zero-Like Training", the DeepSeek-V3 Technical Report, and "Evaluating Large Language Models Trained on Code" (the pass@k estimator).

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 6: the open question the note raises?::Why does the output length grow during RL, and is a longer answer really a better one?

#flashcards/courses/concept #flashcards/topic/course-cme295 How was this CME 295 lecture note built, and what was not seen?::From the lecture slides (PDF) read as extracted text. The note did not see the figures, any speaker notes or the recordings, and did not open the course playlist. Papers only "named on the slides" were not read. The lecture title is the note's own label, not the instructors'.
