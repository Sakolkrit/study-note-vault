---
tags: [paper, NLP, mixture-of-experts]
link: https://arxiv.org/abs/2401.04088
authors: Jiang et al. (first author from memory)
year: 2024
status: abstract-read
field: text analytics
---

# Mixtral 2024

> **In one line for a 12-year-old:** Instead of one big brain doing all the work, the model has eight smaller specialists in each layer, and for each word it asks only the two best ones.

## The problem
Get the quality of a big model while paying the running cost of a small one.

## What people did before
Dense models in which every parameter is used for every word, such as [[LLaMA 2023]]. The abstract compares with Llama 2 70B and GPT-3.5.

## The idea
A sparse [[Mixture of Experts]]: each layer has 8 feed-forward blocks (experts), and a router network picks two for every token at every layer and combines their outputs. The architecture is otherwise the same as Mistral 7B (abstract).

## What they found
Each token has access to 47B parameters but uses only 13B active parameters during inference. Trained with a 32k context, it outperforms or matches Llama 2 70B and GPT-3.5 across all evaluated benchmarks, and vastly outperforms Llama 2 70B on mathematics, code generation and multilingual benchmarks. The instruct version surpasses GPT-3.5 Turbo, Claude-2.1, Gemini Pro and Llama 2 70B chat on human benchmarks. Released under Apache 2.0 (abstract).

## Why it mattered
It showed that sparse experts work well in a released, open model.

## Limits and open questions
All 47B parameters must still be stored in memory even though only 13B are used per token (my arithmetic from the abstract's numbers; see also [[Mixture of Experts]]).

## Concepts used
[[Mixture of Experts]], [[Transformer]], [[LLM]], [[Benchmarks]]

## Reading trail
- **Came from:** [[LLaMA 2023]]
- **Led to:** [[DeepSeek-R1 2025]] (also a mixture of experts, from memory)

## How much I read
Abstract read. Authors written as "(from memory)" were not checked. Numbers marked "abstract" come from the arXiv abstract page. Not opened in Obsidian.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-llm Mixtral (Jiang et al., 2024, arXiv 2401.04088): what problem did it attack?::Getting the quality of a big model while paying the running cost of a small one.

#flashcards/research/concept #flashcards/topic/research-llm What did people do before Mixtral?::Dense models in which every parameter is used for every word, such as LLaMA. The abstract compares with Llama 2 70B and GPT-3.5.

#flashcards/research/concept #flashcards/topic/research-llm Mixtral: what was the idea? (abstract)::A sparse mixture of experts: each layer has 8 feed-forward blocks (experts), and a router network picks two for every token at every layer and combines their outputs. Otherwise the architecture is the same as Mistral 7B.

#flashcards/research/concept #flashcards/topic/research-llm Mixtral: how many parameters does a token use? (abstract)::Each token has access to 47B parameters but uses only 13B active parameters during inference.

#flashcards/research/concept #flashcards/topic/research-llm Mixtral: what did it find? (abstract)::With a 32k context it outperforms or matches Llama 2 70B and GPT-3.5 across all evaluated benchmarks, and vastly outperforms Llama 2 70B on mathematics, code generation and multilingual benchmarks. The instruct version surpasses GPT-3.5 Turbo, Claude-2.1, Gemini Pro and Llama 2 70B chat on human benchmarks. Released under Apache 2.0.

#flashcards/research/concept #flashcards/topic/research-llm Mixtral: what is its limit?::All 47B parameters must still be stored in memory even though only 13B are used per token (the note's arithmetic from the abstract's numbers).

#flashcards/research/concept #flashcards/topic/research-llm Mixtral: reading trail?::It came from LLaMA and led to DeepSeek-R1 (also a mixture of experts, from memory). It showed sparse experts work well in a released, open model.

#flashcards/research/concept #flashcards/topic/research-llm How much of this paper did the note read?::Abstract only; the paper body was not read. Authors marked "from memory" in the note were not checked.
