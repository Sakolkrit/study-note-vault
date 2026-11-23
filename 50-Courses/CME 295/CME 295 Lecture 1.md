---
tags: [lecture, course, CME295, LLM]
course: CME 295
lecture: 1
lecturer: Afshine Amidi and Shervine Amidi
term: Fall 2025
source: Stanford CME 295 lecture 1 slides
status: growing
---

# CME 295 Lecture 1: NLP basics up to the Transformer

> **In one line:** Starts from what language tasks look like and how text becomes numbers, then walks from word vectors and recurrent networks to the Transformer. (The lecture title is my own label, not the course's.)

## What the lecture covers
1. **NLP tasks and how they are scored.** Classification (sentiment, intent, language, topic), tagging of every word (part of speech, named entities, parsing) and text generation (translation, question answering, summarising). The slides give example datasets (Amazon reviews, IMDB, annotated Reuters news, the WMT 2014 translation sets) and the scores: accuracy, precision, recall and F1 for labels; BLEU, ROUGE and perplexity for generated text. See [[Evaluation Metrics]], [[Precision]], [[Recall]], [[F1 Score]] and [[Perplexity]].
2. **Tokenization.** Cutting text into pieces a model can count: words, sub-words or characters. The slides summarise the methods and their trade-offs. See [[Tokenization]] and [[Byte-Pair Encoding]].
3. **Token representations.** Turning each token into a list of numbers. The lecture shows word2vec, a small network trained on a proxy task over billions of words. See [[Embedding]] and [[Word2Vec]] (paper: [[Word2Vec 2013]]).
4. **Recurrent networks.** Reading a sentence one token at a time with a memory, and the LSTM that keeps the memory longer. See [[Recurrent Neural Network]], [[LSTM]] and [[From FNN to RNN]]. A short history places attention as introduced in 2014 ([[Seq2Seq 2014]]).
5. **The Transformer.** The 2017 "Attention Is All You Need" design: queries, keys and values, computed efficiently with matrices; multi-head attention; positional encoding "with a trick"; the encoder and the decoder with its output shifted right. See [[Attention]], [[Self-Attention]], [[Positional Encoding]] and [[Transformer]] (paper: [[Attention Is All You Need 2017]]).
6. **An end-to-end example** using one sentence ("A cute teddy bear is reading.") through tokenization, embeddings, position information, attention and the output layer, with a note on label smoothing as a computational trick. See [[Label Smoothing]] and [[Softmax]].

## Key ideas to remember
- Text has to become numbers before a model can use it: tokens first, then vectors ([[Tokenization]], [[Embedding]]).
- Word vectors give each word one fixed vector; recurrent networks read in order but forget over long spans ([[Recurrent Neural Network]]).
- Attention lets every token look at every other token in one step, which is what the Transformer is built on ([[Attention]], [[Transformer]]).
- Position information has to be added by hand, because attention alone does not know the order ([[Positional Encoding]]).

## Papers named in the lecture
- "Attention Is All You Need", Vaswani et al., 2017 ([[Attention Is All You Need 2017]])
- word2vec ([[Word2Vec 2013]])
- The Seq2Seq line of work that introduced attention in 2014 ([[Seq2Seq 2014]])
- The textbook "Super Study Guide: Transformers and Large Language Models", Amidi et al., 2024 (named on the slides)

## Questions I still have
- Which tokenizer does each modern model use, and how much does the choice matter?
- Why does attention cost grow with the square of the length, and what are the cheaper variants (lecture 2 and 3)?

## What this note is built from
- The lecture slides (PDF), read as extracted text. I did not see the figures, any speaker notes or the recordings, and I did not open the course playlist.
- Topic order and wording of the sections follow the slides. The detail and the tested code live in the concept notes linked above, each of which says what it was checked against.
- Where a paper is only "named on the slides", I did not read it for this note. The concept and paper notes say which abstracts or bodies I did read.

## Course
- Part of [[CME 295]]. Lecture before: (none). Lecture after: [[CME 295 Lecture 2]].
- Concept notes this lecture feeds: [[Tokenization]], [[Byte-Pair Encoding]], [[Embedding]], [[Word2Vec]], [[Recurrent Neural Network]], [[LSTM]], [[Attention]], [[Self-Attention]], [[Positional Encoding]], [[Transformer]], [[Label Smoothing]]

## Flashcards
#flashcards/courses

#flashcards/courses/concept #flashcards/topic/course-cme295 Which scores are used for labels and which for generated text?::Accuracy, precision, recall and F1 for labels. BLEU, ROUGE and perplexity for generated text.

#flashcards/courses/concept #flashcards/topic/course-cme295 Which families of NLP tasks are there?::Classification (sentiment, intent, language, topic), tagging of every word (part of speech, named entities, parsing) and text generation (translation, question answering, summarising).

#flashcards/courses/concept #flashcards/topic/course-cme295 What are the main parts of the Transformer design?::Queries, keys and values computed efficiently with matrices, multi-head attention, positional encoding, and the encoder and the decoder with its output shifted right.

#flashcards/courses/concept #flashcards/topic/course-cme295 Why must position information be added to a Transformer?::Attention alone does not know the order of tokens.
