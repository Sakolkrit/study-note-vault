---
tags: [paper, NLP, position, RoPE]
link: https://arxiv.org/abs/2104.09864
authors: Su et al. (vault note [[RoPE]] checked the abstract)
year: 2021
status: abstract-read
field: text / positions
---

# RoFormer 2021

> **In one line for a 12-year-old:** Instead of adding a position label to each word, turn each word's numbers by an angle that depends on its place. Then two words only 'see' how far apart they are.

## The problem
Give attention a clean sense of word order and distance.

## What people did before
Added absolute sinusoidal or learned position vectors ([[Positional Encoding]]).

## The idea
Rotary Position Embedding: encode absolute position with a rotation matrix so that relative position enters the attention product explicitly.

## What they found
The abstract lists flexibility of sequence length, decaying dependency with distance, and use with linear attention. RoFormer consistently beat its alternatives on long text classification benchmarks (abstract). In this vault's toy: the same-offset score stayed constant under rotation but varied from -13.83 to 20.719 with added sinusoids ([[RoPE]]).

## Why it mattered
Used inside many modern LLMs (I believe LLaMA among them; from memory).

## Limits and open questions
Stretching beyond the trained length needs extra tricks (see the note [[RoPE]]).

## Concepts used
[[RoPE]], [[Positional Encoding]], [[Self-Attention]]

## Reading trail
- **Came from:** [[Attention Is All You Need 2017]]
- **Led to:** [[LLaMA 2023]]

## How much I read
Abstract read. Authors written as "(from memory)" were not checked. Numbers marked "abstract" come from the arXiv abstract page. Not opened in Obsidian.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-nlp RoFormer (Su et al., 2021, arXiv 2104.09864): what problem did it attack?::Giving attention a clean sense of word order and distance.

#flashcards/research/concept #flashcards/topic/research-nlp What did people do before RoFormer?::Added absolute sinusoidal or learned position vectors.

#flashcards/research/concept #flashcards/topic/research-nlp RoFormer: what was the idea?::Rotary Position Embedding: encode absolute position with a rotation matrix so that relative position enters the attention product explicitly. In plain words: turn each word's numbers by an angle that depends on its place.

#flashcards/research/concept #flashcards/topic/research-nlp RoFormer: what did the abstract list? (abstract)::Flexibility of sequence length, decaying dependency with distance, and use with linear attention. RoFormer consistently beat its alternatives on long text classification benchmarks.

#flashcards/research/concept #flashcards/topic/research-nlp RoFormer: what did the vault's own toy show?::The same-offset score stayed constant under rotation but varied from −13.83 to 20.719 with added sinusoids (see the RoPE note).

#flashcards/research/concept #flashcards/topic/research-nlp RoFormer: what is its limit?::Stretching beyond the trained length needs extra tricks (see the RoPE note).

#flashcards/research/concept #flashcards/topic/research-nlp RoFormer: reading trail?::It came from the Transformer paper and led to LLaMA. It is used inside many modern LLMs (the note believes LLaMA among them; from memory).

#flashcards/research/concept #flashcards/topic/research-nlp How much of RoFormer was read?::Abstract only; the paper body was not read. The abstract was checked in the RoPE note.
