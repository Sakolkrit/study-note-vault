---
tags: [lecture, course, CME295, LLM]
course: CME 295
lecture: 2
lecturer: Afshine Amidi and Shervine Amidi
term: Fall 2025
source: Stanford CME 295 lecture 2 slides
status: growing
---

# CME 295 Lecture 2: Making the Transformer work, and BERT

> **In one line:** Improves the pieces of the Transformer (position information, normalisation, cheaper attention) and then introduces the encoder-only model BERT. (The lecture title is my own label, not the course's.)

## What the lecture covers
1. **Position information.** Why attention loses the order, then four answers: learned position embeddings, hard-coded ones, a linear bias added to the attention scores, and rotating the queries and keys, which the slides call the default choice nowadays. The rotation gives an attention weight with a long-term decay. See [[Positional Encoding]], [[RoPE]] and [[RoFormer 2021]].
2. **Layer normalization.** Post-norm and pre-norm placements and the kinds of normalisation. See [[Layer Normalization]] and [[Normalization]].
3. **Cheaper attention.** Sparse attention as in Longformer, mixing local and global attention with a sliding window, and sharing key and value heads between several query heads. See [[Sliding Window Attention]] and [[Grouped-Query Attention]].
4. **Three kinds of Transformer models.** Encoder-only, decoder-only and encoder-decoder, and which models belong to which.
5. **BERT.** Encoder-only; WordPiece input; input embeddings; two proxy tasks, masked language modelling (hide some input tokens and predict them) and next-sentence prediction; hyperparameters and sizes. See [[BERT]] and [[BERT 2018]].
6. **Fine-tuning BERT.** Using its embeddings for a task such as sentiment extraction, with the benefits and shortcomings.
7. **Distillation and variants.** Training a small model to copy a big one (the soft targets idea), DistilBERT for efficiency and RoBERTa for performance. See [[Knowledge Distillation]] and [[Fine-tuning]].

## Key ideas to remember
- Rotary position embeddings are the usual choice today ([[RoPE]]).
- Models split into encoder-only (understand text), decoder-only (write text) and encoder-decoder (text to text).
- BERT learns from a fill-in-the-blank task, then is fine-tuned for each job ([[BERT]], [[Pre-training]]).
- A small model can be trained to copy a big one ([[Knowledge Distillation]]).

## Papers named in the lecture
- "Attention Is All You Need", Vaswani et al., 2017 (recap) ([[Attention Is All You Need 2017]])
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding ([[BERT 2018]])
- RoFormer, the rotary position paper ([[RoFormer 2021]])
- RoBERTa: A Robustly Optimized BERT Pretraining Approach (named on the slides)
- DistilBERT and the idea of soft targets (named on the slides)

## Questions I still have
- Why did decoder-only models win for general use, when BERT-style models were so strong at understanding tasks?

## What this note is built from
- The lecture slides (PDF), read as extracted text. I did not see the figures, any speaker notes or the recordings, and I did not open the course playlist.
- Topic order and wording of the sections follow the slides. The detail and the tested code live in the concept notes linked above, each of which says what it was checked against.
- Where a paper is only "named on the slides", I did not read it for this note. The concept and paper notes say which abstracts or bodies I did read.

## Course
- Part of [[CME 295]]. Lecture before: [[CME 295 Lecture 1]]. Lecture after: [[CME 295 Lecture 3]].
- Concept notes this lecture feeds: [[Positional Encoding]], [[RoPE]], [[Layer Normalization]], [[Normalization]], [[Sliding Window Attention]], [[Grouped-Query Attention]], [[BERT]], [[Knowledge Distillation]]

## Flashcards
#flashcards/courses

#flashcards/courses/concept #flashcards/topic/course-cme295 What are four ways to give a Transformer position information?::Learned position embeddings, hard-coded ones, a linear bias added to the attention scores, and rotating the queries and keys. Rotation is now the common default, and it gives an attention weight with a long-term decay (from the course slides).

#flashcards/courses/concept #flashcards/topic/course-cme295 Which ideas make attention cheaper?::Sparse attention as in Longformer, mixing local and global attention with a sliding window, and sharing key and value heads between several query heads (grouped-query attention).

#flashcards/courses/concept #flashcards/topic/course-cme295 Name the three kinds of Transformer models.::Encoder-only (understand text), decoder-only (write text) and encoder-decoder (text to text).

#flashcards/courses/concept #flashcards/topic/course-cme295 What are BERT's two proxy tasks?::Masked language modelling (hide some input tokens and predict them) and next-sentence prediction. BERT is encoder-only with WordPiece input.

#flashcards/courses/concept #flashcards/topic/course-cme295 What does distillation mean for language models?::Training a small model to copy a big one (the soft-targets idea). DistilBERT aims at efficiency and RoBERTa, a related variant, at performance.
