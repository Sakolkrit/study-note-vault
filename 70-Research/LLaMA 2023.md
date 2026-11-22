---
tags: [paper, NLP, LLM, open]
link: https://arxiv.org/abs/2302.13971
authors: Touvron et al. (first author from memory)
year: 2023
status: abstract-read
field: text / LLM
---

# LLaMA 2023

> **In one line for a 12-year-old:** A family of language models, trained for longer on public text, small enough that more people can run them.

## The problem
Get strong language models without proprietary data, and in sizes that research groups can use.

## What people did before
Very large closed models such as [[GPT-3 2020]], which the [[Scaling Laws]] work suggests were under-trained for their size (Chinchilla).

## The idea
Train models of 7B to 65B parameters on trillions of tokens from publicly available data only, and release them.

## What they found
LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, about 13.5 times smaller by my arithmetic. LLaMA-65B is competitive with Chinchilla-70B and PaLM-540B (abstract).

## Why it mattered
Opened the door to open-weight models and a fast wave of fine-tuned variants ([[LoRA]], [[Quantization]]).

## Limits and open questions
Its architecture details (pre-norm, RMS normalisation, SwiGLU, rotary positions) are from my memory of the paper, not from the abstract. I did not read the body.

## Concepts used
[[LLM]], [[Pre-training]], [[Scaling Laws]], [[RoPE]]

## Reading trail
- **Came from:** [[GPT-3 2020]]
- **Led to:** [[Mixtral 2024]], [[DeepSeek-R1 2025]], [[QLoRA 2023]] (fine-tunes LLaMA models in 4 bits)

## How much I read
Abstract read. Authors written as "(from memory)" were not checked. Numbers marked "abstract" come from the arXiv abstract page. Not opened in Obsidian.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-llm LLaMA (Touvron et al., 2023, arXiv 2302.13971): what problem did it attack?::Getting strong language models without proprietary data, in sizes research groups can use.

#flashcards/research/concept #flashcards/topic/research-llm What did people do before LLaMA?::Very large closed models such as GPT-3, which the Chinchilla scaling work suggests were under-trained for their size.

#flashcards/research/concept #flashcards/topic/research-llm LLaMA: what was the idea?::Train models of 7B to 65B parameters on trillions of tokens from publicly available data only, and release them.

#flashcards/research/concept #flashcards/topic/research-llm LLaMA: what did it find? (abstract)::LLaMA-13B outperforms GPT-3 (175B) on most benchmarks, about 13.5 times smaller by the note's arithmetic. LLaMA-65B is competitive with Chinchilla-70B and PaLM-540B.

#flashcards/research/concept #flashcards/topic/research-llm LLaMA: which details are from memory, not the abstract?::The architecture details: pre-norm, RMS normalisation, SwiGLU and rotary positions. The note did not read the body.

#flashcards/research/concept #flashcards/topic/research-llm LLaMA: reading trail?::It came from GPT-3 and led to Mixtral and DeepSeek-R1. It opened the door to open-weight models and fine-tuned variants (LoRA, quantization).

#flashcards/research/concept #flashcards/topic/research-llm How much of this paper did the note read?::Abstract only; the paper body was not read. Authors marked "from memory" in the note were not checked.
