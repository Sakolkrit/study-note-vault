---
tags: [paper, NLP, scaling]
link: https://arxiv.org/abs/2203.15556
authors: Hoffmann et al. (first author from memory)
year: 2022
status: abstract-read
field: text analytics
---

# Chinchilla 2022

> **In one line for a 12-year-old:** People were making models bigger and bigger but feeding them the same amount of text. These authors found it works better to grow both the model and the text together.

## The problem
For a fixed computing budget, how big should the model be and how much text should it read?

## What people did before
Scaling advice from earlier work ([[Scaling Laws]]), which favoured very large models trained on modest data.

## The idea
Train over 400 language models (70 million to over 16 billion parameters, on 5 to 500 billion tokens) and fit how the best size and data amount grow with compute. Then test the prediction by training Chinchilla.

## What they found
For compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the tokens should also be doubled. Chinchilla (70B parameters, 4x more data, same compute as Gopher) uniformly and significantly outperforms Gopher (280B), GPT-3 (175B), Jurassic-1 (178B) and Megatron-Turing NLG (530B), and reaches 67.5% on MMLU, more than 7% above Gopher (abstract).

## Why it mattered
It changed how models are planned: smaller, longer-trained models that are also cheaper to use.

## Limits and open questions
It assumes the same training recipe and a fixed budget; it says little about quality of data (from memory; the abstract does not discuss it).

## Concepts used
[[Scaling Laws]], [[Pre-training]], [[LLM]], [[Benchmarks]]

## Reading trail
- **Came from:** [[GPT-3 2020]], [[Scaling Laws 2020]] (the allocation it revised)
- **Led to:** [[LLaMA 2023]]

## How much I read
Abstract read. Authors written as "(from memory)" were not checked. Numbers marked "abstract" come from the arXiv abstract page. Not opened in Obsidian.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-llm Chinchilla (Hoffmann et al., 2022, arXiv 2203.15556): what question did it ask?::For a fixed computing budget, how big should the model be and how much text should it read?

#flashcards/research/concept #flashcards/topic/research-llm What did people do before Chinchilla?::Followed earlier scaling advice, which favoured very large models trained on modest data.

#flashcards/research/concept #flashcards/topic/research-llm Chinchilla: what did the authors train to find the answer? (abstract)::Over 400 language models (70 million to over 16 billion parameters, on 5 to 500 billion tokens), fitting how the best size and data amount grow with compute, then testing the prediction by training Chinchilla.

#flashcards/research/concept #flashcards/topic/research-llm Chinchilla: what is the headline rule? (abstract)::For compute-optimal training, model size and training tokens should be scaled equally: for every doubling of model size the tokens should also be doubled.

#flashcards/research/concept #flashcards/topic/research-llm Chinchilla: how did the 70B model compare? (abstract)::Chinchilla (70B parameters, 4x more data, same compute as Gopher) uniformly and significantly outperforms Gopher (280B), GPT-3 (175B), Jurassic-1 (178B) and Megatron-Turing NLG (530B), and reaches 67.5% on MMLU, more than 7% above Gopher.

#flashcards/research/concept #flashcards/topic/research-llm Chinchilla: what are its limits?::It assumes the same training recipe and a fixed budget. It says little about quality of data (from memory; the abstract does not discuss it).

#flashcards/research/concept #flashcards/topic/research-llm Chinchilla: why did it matter and what is its place in the reading trail?::It changed how models are planned: smaller, longer-trained models that are also cheaper to use. It came from GPT-3 and led to LLaMA.

#flashcards/research/concept #flashcards/topic/research-llm How much of this paper did the note read?::Abstract only; the paper body was not read. Authors marked "from memory" in the note were not checked.
