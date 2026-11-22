---
tags: [paper, NLP, RNN, translation]
link: https://arxiv.org/abs/1409.3215
authors: Sutskever, Vinyals, Le (from memory)
year: 2014
status: abstract-read
field: text / translation
---

# Seq2Seq 2014

> **In one line for a 12-year-old:** One network reads a sentence and squeezes it into a summary, and another network writes the translation from that summary.

## The problem
Map one sequence to another sequence of a different length, such as French to English.

## What people did before
Deep networks needed fixed-size inputs and outputs (the abstract: they 'cannot be used to map sequences to sequences').

## The idea
Use one multilayer LSTM to read the input into a fixed-size vector and another to decode the output.

## What they found
On WMT'14 English to French the LSTM got 34.8 BLEU against 33.3 for a phrase-based system, and reranking that system's 1000 hypotheses reached 36.5. Reversing the source sentence markedly improved results (abstract).

## Why it mattered
Began the neural machine translation era.

## Limits and open questions
The fixed-size vector is a bottleneck (the conjecture of [[Attention]]'s first paper).

## Concepts used
[[Recurrent Neural Network]], [[Embedding]], [[Decoding Strategies]]

## Reading trail
- **Came from:** [[From FNN to RNN]]
- **Led to:** [[Attention Is All You Need 2017]]

## How much I read
Abstract read. Authors written as "(from memory)" were not checked. Numbers marked "abstract" come from the arXiv abstract page. Not opened in Obsidian.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-nlp Seq2Seq (Sutskever, Vinyals, Le, 2014, arXiv 1409.3215): what problem did it attack?::Mapping one sequence to another sequence of a different length, such as French to English.

#flashcards/research/concept #flashcards/topic/research-nlp What did people do before Seq2Seq? (abstract)::Deep networks needed fixed-size inputs and outputs and "cannot be used to map sequences to sequences".

#flashcards/research/concept #flashcards/topic/research-nlp Seq2Seq: what was the idea?::Use one multilayer LSTM to read the input into a fixed-size vector and another LSTM to decode the output. One network reads and squeezes the sentence, the other writes the translation.

#flashcards/research/concept #flashcards/topic/research-nlp Seq2Seq: what did it find? (abstract)::On WMT'14 English to French the LSTM got 34.8 BLEU against 33.3 for a phrase-based system, and reranking that system's 1000 hypotheses reached 36.5. Reversing the source sentence markedly improved results.

#flashcards/research/concept #flashcards/topic/research-nlp Seq2Seq: what is its limit?::The fixed-size vector is a bottleneck. Attention's first paper conjectured this.

#flashcards/research/concept #flashcards/topic/research-nlp Seq2Seq: reading trail?::It came from the FNN-to-RNN timeline and led to the Transformer paper. It began the neural machine translation era.

#flashcards/research/concept #flashcards/topic/research-nlp How much of this paper did the note read?::Abstract only; the paper body was not read. Authors marked "from memory" in the note were not checked.
