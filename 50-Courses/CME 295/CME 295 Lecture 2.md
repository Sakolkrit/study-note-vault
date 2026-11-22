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

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 2: what does it cover?::Position information, layer normalization, cheaper attention, the three kinds of Transformer models, BERT, fine-tuning BERT, and distillation with variants (DistilBERT, RoBERTa).

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 2: what four answers to the position problem does it show?::Learned position embeddings, hard-coded ones, a linear bias added to the attention scores, and rotating the queries and keys. The slides call the rotation the default choice nowadays, and it gives an attention weight with a long-term decay.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 2: what cheaper attention ideas appear?::Sparse attention as in Longformer, mixing local and global attention with a sliding window, and sharing key and value heads between several query heads (grouped-query attention).

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 2: name the three kinds of Transformer models.::Encoder-only (understand text), decoder-only (write text) and encoder-decoder (text to text).

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 2: what are BERT's two proxy tasks?::Masked language modelling (hide some input tokens and predict them) and next-sentence prediction. BERT is encoder-only with WordPiece input.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 2: what does distillation mean here?::Training a small model to copy a big one (the soft-targets idea). DistilBERT is for efficiency and RoBERTa for performance.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 2: which papers are named?::"Attention Is All You Need" (recap), BERT and RoFormer (both have paper notes), plus RoBERTa and DistilBERT (named on the slides only).

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 2: the open question the note raises?::Why did decoder-only models win for general use when BERT-style models were so strong at understanding tasks?

#flashcards/courses/concept #flashcards/topic/course-cme295 How was this CME 295 lecture note built, and what was not seen?::From the lecture slides (PDF) read as extracted text. The note did not see the figures, any speaker notes or the recordings, and did not open the course playlist. Papers only "named on the slides" were not read. The lecture title is the note's own label, not the instructors'.
