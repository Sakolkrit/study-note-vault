---
tags: [lecture, course, CME295, LLM]
course: CME 295
lecture: 5
lecturer: Afshine Amidi and Shervine Amidi
term: Fall 2025
source: Stanford CME 295 lecture 5 slides
status: growing
---

# CME 295 Lecture 5: Preference tuning

> **In one line:** Shows how a model is steered to what people prefer: collect comparisons, train a reward model, use reinforcement learning (PPO), or skip the reinforcement learning with DPO. (The lecture title is my own label, not the course's.)

## What the lecture covers
1. **Why preference tuning.** A pretrained or fine-tuned model can still misbehave, and it is easier to say "A is better than B" than to write the ideal answer. See [[Reinforcement Learning]].
2. **Preference data.** Pairs of (prompt, response); a recipe: generate two responses to a prompt, then have them labelled. Pointwise and pairwise forms.
3. **Reinforcement learning, in LLM terms.** The agent is the LLM and the environment is the conversation. See [[Policy Gradient]].
4. **RLHF.** Step 1, a reward model trained on pairwise preferences with a pairwise preference model (the slide cites a paper on incomplete block designs, the origin of the Bradley-Terry model). Step 2, reinforcement learning against that reward. See [[RLHF]], [[Reward Model]], [[Bradley-Terry Model]] and [[InstructGPT 2022]].
5. **PPO.** The common algorithm: it works with advantages, not only rewards, and has two variants, clipping the ratio between the new and old policy and a KL penalty. See [[PPO]].
6. **Challenges and a workaround.** RL needs a separate reward model and is hard to run. Best-of-N (sample N answers and keep the best) is a workaround without RL.
7. **DPO.** Direct Preference Optimization: start from the PPO objective and get a supervised loss on the preference pairs, with no reward model and no RL loop. The lecture asks whether PPO-based RLHF or DPO is better, citing a paper titled "Is DPO Superior to PPO for LLM Alignment?". See [[Direct Preference Optimization]] and [[DPO 2023]].

## Key ideas to remember
- People compare answers more easily than they write ideal ones, so preferences are the training signal.
- RLHF = train a reward model on comparisons, then optimise the LLM against it with PPO ([[RLHF]], [[PPO]]).
- DPO reaches a similar goal with a simple classification-style loss ([[Direct Preference Optimization]]).
- Best-of-N is a quick way to use a reward model without any RL ([[Reward Model]]).

## Papers named in the lecture
- "Training language models to follow instructions with human feedback", Ouyang et al., 2022 ([[InstructGPT 2022]])
- "Proximal Policy Optimization Algorithms", Schulman et al., 2017 ([[PPO]])
- "Direct Preference Optimization", Rafailov et al., 2023 ([[DPO 2023]])
- "The rank analysis of incomplete block designs" (Bradley and Terry; [[Bradley-Terry Model]])
- "Is DPO Superior to PPO for LLM Alignment?" (named on the slides)

## Questions I still have
- Which of PPO and DPO would I pick for a small project, and why?

## What this note is built from
- The lecture slides (PDF), read as extracted text. I did not see the figures, any speaker notes or the recordings, and I did not open the course playlist.
- Topic order and wording of the sections follow the slides. The detail and the tested code live in the concept notes linked above, each of which says what it was checked against.
- Where a paper is only "named on the slides", I did not read it for this note. The concept and paper notes say which abstracts or bodies I did read.

## Course
- Part of [[CME 295]]. Lecture before: [[CME 295 Lecture 4]]. Lecture after: [[CME 295 Lecture 6]].
- Concept notes this lecture feeds: [[RLHF]], [[Reward Model]], [[Bradley-Terry Model]], [[PPO]], [[Direct Preference Optimization]], [[Reinforcement Learning]], [[Policy Gradient]]

## Flashcards
#flashcards/courses

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 5: what does it cover?::Why preference tuning, preference data, reinforcement learning in LLM terms, RLHF, PPO, challenges (best-of-N), and DPO.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 5: why use preferences?::A model can still misbehave after pretraining and fine-tuning, and it is easier to say "A is better than B" than to write the ideal answer.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 5: how is reinforcement learning described in LLM terms?::The agent is the LLM and the environment is the conversation.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 5: what are the two steps of RLHF?::Step 1, train a reward model on pairwise preferences with a pairwise preference model (the slide cites the paper on incomplete block designs, the origin of the Bradley-Terry model). Step 2, reinforcement learning against that reward.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 5: what is PPO's design here?::It works with advantages, not only rewards, and has two variants: clipping the ratio between the new and old policy, and a KL penalty.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 5: what is best-of-N?::Sample N answers and keep the best by the reward model. It is a workaround without reinforcement learning, because RL needs a separate reward model and is hard to run.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 5: how does DPO work?::Start from the PPO objective and get a supervised loss on the preference pairs, with no reward model and no RL loop. The lecture asks whether PPO-based RLHF or DPO is better, citing "Is DPO Superior to PPO for LLM Alignment?".

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 5: which papers are named?::InstructGPT, PPO ("Proximal Policy Optimization Algorithms", Schulman et al., 2017), DPO, "The rank analysis of incomplete block designs" (Bradley and Terry) and "Is DPO Superior to PPO for LLM Alignment?". Only some have paper or concept notes. The rest were named on slides and not read.

#flashcards/courses/concept #flashcards/topic/course-cme295 How was this CME 295 lecture note built, and what was not seen?::From the lecture slides (PDF) read as extracted text. The note did not see the figures, any speaker notes or the recordings, and did not open the course playlist. Papers only "named on the slides" were not read. The lecture title is the note's own label, not the instructors'.
