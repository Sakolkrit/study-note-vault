---
tags: [paper, NLP, alignment, RLHF]
link: https://arxiv.org/abs/2203.02155
authors: Ouyang et al. (first author from memory)
year: 2022
status: abstract-read
field: text analytics
---

# InstructGPT 2022

> **In one line for a 12-year-old:** Big language models were good at continuing text but not at doing what you asked. People ranked the answers, and the model was trained to give the kind of answers people liked.

## The problem
Making language models bigger does not inherently make them better at following a user's intent (first sentence of the abstract).

## What people did before
Plain pre-trained models such as [[GPT-3 2020]], which can be untruthful, toxic or unhelpful (abstract).

## The idea
Fine-tune GPT-3 on labeler-written demonstrations with supervised learning ([[Instruction Tuning]]), then collect rankings of model outputs and fine-tune further with reinforcement learning from human feedback ([[RLHF]], [[Reward Model]], [[PPO]]).

## What they found
Human raters preferred outputs of the 1.3B InstructGPT model to those of the 175B GPT-3, despite 100x fewer parameters. InstructGPT also improved truthfulness and reduced toxic output with minimal regressions on public NLP datasets (abstract).

## Why it mattered
It is the recipe behind many chat assistants (from memory), and it showed that alignment can matter more than size.

## Limits and open questions
InstructGPT still makes simple mistakes (abstract). The pipeline is complicated; [[Direct Preference Optimization]] later simplified it.

## Concepts used
[[RLHF]], [[Reward Model]], [[PPO]], [[Instruction Tuning]], [[Fine-tuning]], [[Bradley-Terry Model]]

## Reading trail
- **Came from:** [[GPT-3 2020]]
- **Led to:** [[DPO 2023]]

## How much I read
Abstract read. Authors written as "(from memory)" were not checked. Numbers marked "abstract" come from the arXiv abstract page. Not opened in Obsidian.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-llm InstructGPT (Ouyang et al., 2022, arXiv 2203.02155): what problem did it attack? (abstract)::Making language models bigger does not inherently make them better at following a user's intent.

#flashcards/research/concept #flashcards/topic/research-llm What did people do before InstructGPT?::Plain pre-trained models such as GPT-3, which can be untruthful, toxic or unhelpful (abstract).

#flashcards/research/concept #flashcards/topic/research-llm InstructGPT: what was the idea?::Fine-tune GPT-3 on labeler-written demonstrations with supervised learning, then collect rankings of model outputs and fine-tune further with reinforcement learning from human feedback (reward model plus PPO).

#flashcards/research/concept #flashcards/topic/research-llm InstructGPT: what did it find? (abstract)::Human raters preferred outputs of the 1.3B InstructGPT model to those of the 175B GPT-3, despite 100x fewer parameters. It also improved truthfulness and reduced toxic output, with minimal regressions on public NLP datasets.

#flashcards/research/concept #flashcards/topic/research-llm InstructGPT: what is its limit?::It still makes simple mistakes (abstract). The pipeline is complicated, and DPO later simplified it.

#flashcards/research/concept #flashcards/topic/research-llm InstructGPT: reading trail?::It came from GPT-3 and led to DPO. It is the recipe behind many chat assistants (from memory) and showed that alignment can matter more than size.

#flashcards/research/concept #flashcards/topic/research-llm How much of this paper did the note read?::Abstract only; the paper body was not read. Authors marked "from memory" in the note were not checked.
