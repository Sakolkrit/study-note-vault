---
tags: [paper, LLM, reasoning, reinforcement-learning]
link: https://arxiv.org/abs/2501.12948
authors: DeepSeek-AI (from memory)
year: 2025
status: abstract-read
field: text / reasoning
---

# DeepSeek-R1 2025

> **In one line for a 12-year-old:** A model that learns to think step by step because it is rewarded when its final answer is right, not because people showed it how.

## The problem
Get better reasoning without relying on human-written reasoning examples.

## What people did before
Models taught by human demonstrations plus [[Chain of Thought]] prompting, then tuned with [[RLHF]].

## The idea
Incentivise reasoning with pure [[Reinforcement Learning]] on tasks whose answers can be checked ([[Reasoning Model]], trained with [[GRPO]] per the CME 295 lecture).

## What they found
The abstract reports that self-reflection, verification and dynamic strategy adaptation emerged, with superior performance on maths, coding contests and STEM tasks compared with models trained on human demonstrations, and that these patterns can help smaller models. The abstract gives no figures, and I did not read the body.

## Why it mattered
Made RL with verifiable rewards a main recipe for reasoning models.

## Limits and open questions
Works on tasks with checkable answers. Details like the 671B/37B model size, data counts and the 'cold start' stage are from the lecture slides (see [[Reasoning Model]]), not verified in the paper by me.

## Concepts used
[[Reasoning Model]], [[GRPO]], [[Reinforcement Learning]], [[Chain of Thought]], [[Mixture of Experts]]

## Reading trail
- **Came from:** [[LLaMA 2023]]
- **Led to:** -

## How much I read
Abstract read. Authors written as "(from memory)" were not checked. Numbers marked "abstract" come from the arXiv abstract page. Not opened in Obsidian.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-llm DeepSeek-R1 (DeepSeek-AI, 2025, arXiv 2501.12948): what problem did it attack?::Getting better reasoning without relying on human-written reasoning examples.

#flashcards/research/concept #flashcards/topic/research-llm What did people do before DeepSeek-R1?::Models taught by human demonstrations plus chain-of-thought prompting, then tuned with RLHF.

#flashcards/research/concept #flashcards/topic/research-llm DeepSeek-R1: what was the idea?::Incentivise reasoning with pure reinforcement learning on tasks whose answers can be checked, trained with GRPO (per the CME 295 lecture).

#flashcards/research/concept #flashcards/topic/research-llm DeepSeek-R1: what did the abstract report?::Self-reflection, verification and dynamic strategy adaptation emerged, with superior performance on maths, coding contests and STEM tasks compared with models trained on human demonstrations, and these patterns can help smaller models. The abstract gives no figures.

#flashcards/research/concept #flashcards/topic/research-llm DeepSeek-R1: which details are not verified?::The 671B/37B model size, data counts and the "cold start" stage are from the lecture slides (see the Reasoning Model note), not verified in the paper.

#flashcards/research/concept #flashcards/topic/research-llm DeepSeek-R1: what is its limit?::It works on tasks with checkable answers.

#flashcards/research/concept #flashcards/topic/research-llm DeepSeek-R1: why did it matter?::It made reinforcement learning with verifiable rewards a main recipe for reasoning models.

#flashcards/research/concept #flashcards/topic/research-llm How much of this paper did the note read?::Abstract only; the paper body was not read. Authors marked "from memory" in the note were not checked.
