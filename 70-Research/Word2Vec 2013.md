---
tags: [paper, NLP, embedding]
link: https://arxiv.org/abs/1301.3781
authors: Mikolov et al. (first author from memory)
year: 2013
status: abstract-read
field: text / word vectors
---

# Word2Vec 2013

> **In one line for a 12-year-old:** Give every word a short list of numbers so that words used in similar places get similar lists.

## The problem
Learn good word vectors from huge text quickly.

## What people did before
Neural-network language models, which were slow, and counting tables ([[Embedding]]).

## The idea
Two simple model architectures (named CBOW and skip-gram in the paper, from memory) for continuous word vectors; extended with negative sampling and subsampling in a second paper (arXiv 1310.4546).

## What they found
High-quality vectors from a 1.6 billion word set in under a day, with large accuracy gains at lower cost, and state-of-the-art results on a word-relationship test set (abstract).

## Why it mattered
Made 'pre-train on plain text, reuse everywhere' normal for language.

## Limits and open questions
One vector per word, whatever the sentence (see [[BERT 2018]] and ELMo for the context-dependent fix).

## Concepts used
[[Word2Vec]], [[Negative Sampling]], [[Self-Supervised Learning]]

## Reading trail
- **Came from:** bag-of-words and count tables
- **Led to:** [[ELMo 2018]], [[Attention Is All You Need 2017]]

## How much I read
Abstract read. Authors written as "(from memory)" were not checked. Numbers marked "abstract" come from the arXiv abstract page. Not opened in Obsidian.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-nlp Word2Vec (Mikolov et al., 2013, arXiv 1301.3781): what problem did it attack?::Learning good word vectors from huge text quickly.

#flashcards/research/concept #flashcards/topic/research-nlp What did people do before Word2Vec?::Neural-network language models, which were slow, and counting tables.

#flashcards/research/concept #flashcards/topic/research-nlp Word2Vec: what was the idea?::Two simple model architectures (named CBOW and skip-gram in the paper, from memory) for continuous word vectors. A second paper (arXiv 1310.4546) extended them with negative sampling and subsampling.

#flashcards/research/concept #flashcards/topic/research-nlp Word2Vec: what did it find? (abstract)::High-quality vectors from a 1.6 billion word set in under a day, with large accuracy gains at lower cost, and state-of-the-art results on a word-relationship test set.

#flashcards/research/concept #flashcards/topic/research-nlp Word2Vec: what is its limit?::One vector per word, whatever the sentence. BERT and ELMo give the context-dependent fix.

#flashcards/research/concept #flashcards/topic/research-nlp Word2Vec: reading trail?::It came from bag-of-words and count tables and led to ELMo and the Transformer paper. It made "pre-train on plain text, reuse everywhere" normal for language.

#flashcards/research/concept #flashcards/topic/research-nlp How much of this paper did the note read?::Abstract only; the paper body was not read. Authors marked "from memory" in the note were not checked.
