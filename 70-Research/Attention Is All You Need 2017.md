---
tags: [paper, NLP, transformer, attention]
link: https://arxiv.org/abs/1706.03762
authors: Vaswani et al. (first author from memory)
year: 2017
status: abstract-read
field: text / architecture
---

# Attention Is All You Need 2017

> **In one line for a 12-year-old:** Let every word look at every other word directly and decide which matter. No more reading one word at a time.

## The problem
Make sequence models faster to train and better at long-range links.

## What people did before
Recurrent or convolutional encoder-decoders, linked by an attention mechanism ([[Seq2Seq 2014]]).

## The idea
The [[Transformer]]: only [[Attention]] and feed-forward layers, no recurrence or convolution.

## What they found
28.4 BLEU on WMT 2014 English-German (over 2 BLEU above the best earlier results including ensembles) and 41.8 BLEU English-French after 3.5 days on eight GPUs, a small fraction of earlier training cost. It also worked on constituency parsing (abstract).

## Why it mattered
Almost every large language model since is a Transformer.

## Limits and open questions
Cost grows with the square of the text length, which led to [[FlashAttention]], [[Sliding Window Attention]] and [[KV Cache]] tricks.

## Concepts used
[[Self-Attention]], [[Positional Encoding]], [[Residual Connection]], [[Layer Normalization]]

## Reading trail
- **Came from:** [[Seq2Seq 2014]]
- **Led to:** [[BERT 2018]], [[DLinear 2022]] (questions Transformers for long-horizon forecasting), [[Scaling Laws 2020]] (studies how Transformer language models scale)

## How much I read
Abstract read. Authors written as "(from memory)" were not checked. Numbers marked "abstract" come from the arXiv abstract page. Not opened in Obsidian.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-nlp Attention Is All You Need (Vaswani et al., 2017, arXiv 1706.03762): what problem did it attack?::Making sequence models faster to train and better at long-range links.

#flashcards/research/concept #flashcards/topic/research-nlp What did people do before the Transformer?::Recurrent or convolutional encoder-decoders linked by an attention mechanism (seq2seq).

#flashcards/research/concept #flashcards/topic/research-nlp Transformer: what was the idea?::Use only attention and feed-forward layers, with no recurrence or convolution.

#flashcards/research/concept #flashcards/topic/research-nlp Transformer: what did it find? (abstract)::28.4 BLEU on WMT 2014 English-German (over 2 BLEU above the best earlier results including ensembles) and 41.8 BLEU English-French after 3.5 days on eight GPUs, a small fraction of earlier training cost. It also worked on constituency parsing.

#flashcards/research/concept #flashcards/topic/research-nlp Transformer: what limit did it leave?::Cost grows with the square of the text length, which led to FlashAttention, sliding window attention and KV cache tricks.

#flashcards/research/concept #flashcards/topic/research-nlp Transformer: why did it matter and what is its place in the reading trail?::Almost every large language model since is a Transformer. It came from Seq2Seq and led to BERT.

#flashcards/research/concept #flashcards/topic/research-nlp How much of this paper did the note read?::Abstract only; the paper body was not read. Authors marked "from memory" in the note were not checked.
