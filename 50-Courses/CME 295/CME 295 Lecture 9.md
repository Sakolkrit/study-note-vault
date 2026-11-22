---
tags: [lecture, course, CME295, LLM]
course: CME 295
lecture: 9
lecturer: Afshine Amidi and Shervine Amidi
term: Fall 2025
source: Stanford CME 295 lecture 9 slides
status: growing
---

# CME 295 Lecture 9: Beyond text, diffusion LLMs and closing thoughts

> **In one line:** Looks at Transformers used for images and for images with text, a newer way of generating text with diffusion, and where the field is going. (The lecture title is my own label, not the course's.)

## What the lecture covers
1. **Rewinding the quarter.** A recap of lectures 1 to 8.
2. **Transformers for images.** Attention applied to image patches, projecting patches into vectors, the Vision Transformer and an end-to-end example with a classification token. See [[Vision Transformer]] and [[ViT 2020]].
3. **Vision language models.** Two methods: recycle a decoder-only LLM by feeding it image embeddings (Visual Instruction Tuning), or add cross-attention layers (as in the Llama 3 report).
4. **Beyond autoregressive text.** Today's LLMs write one token after another. Diffusion gives another way: for images, add noise and learn to remove it; for text, mask tokens and learn to unmask them (a masked diffusion model), which can decode in fewer passes. See [[Diffusion Models]] and [[Masked Diffusion Language Model]].
5. **Trends.** Ideas travel between modalities, foundational research is still very active, interest in new frontiers and in hardware (an analog in-memory attention paper is shown), and how LLMs have already changed daily life.
6. **Ongoing challenges.** Fixed weights (no continuous learning), hallucinations, personalisation, interpretability and safety, and how to keep up (arXiv, conferences, code and trending papers).

## Key ideas to remember
- The attention idea works on image patches too, with a very similar model ([[Vision Transformer]]).
- A text model can be built by unmasking instead of next-token prediction ([[Masked Diffusion Language Model]]).
- The open problems are fixed weights, hallucination, personalisation, interpretability and safety ([[Agent Safety]]).

## Papers named in the lecture
- "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" ([[ViT 2020]])
- "Visual Instruction Tuning" (named on the slides)
- "The Llama 3 Herd of Models" (named on the slides; [[LLaMA 2023]] is the first Llama)
- "Denoising Diffusion Probabilistic Models" (named on the slides; [[Diffusion Models]])
- "Analog in-memory computing attention mechanism..." (named on the slides)

## Questions I still have
- Will diffusion decoding replace next-token generation for some tasks, as the lecture hints?

## What this note is built from
- The lecture slides (PDF), read as extracted text. I did not see the figures, any speaker notes or the recordings, and I did not open the course playlist.
- Topic order and wording of the sections follow the slides. The detail and the tested code live in the concept notes linked above, each of which says what it was checked against.
- Where a paper is only "named on the slides", I did not read it for this note. The concept and paper notes say which abstracts or bodies I did read.

## Course
- Part of [[CME 295]]. Lecture before: [[CME 295 Lecture 8]]. Lecture after: (none).
- Concept notes this lecture feeds: [[Vision Transformer]], [[Diffusion Models]], [[Masked Diffusion Language Model]]

## Flashcards
#flashcards/courses

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 9: what does it cover?::A recap of lectures 1 to 8, Transformers for images, vision language models, diffusion beyond autoregressive text, trends, and ongoing challenges.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 9: how are Transformers used for images?::Attention applied to image patches, projecting patches into vectors, the Vision Transformer, with an end-to-end example using a classification token.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 9: what are the two vision language model methods?::Recycle a decoder-only LLM by feeding it image embeddings (Visual Instruction Tuning), or add cross-attention layers (as in the Llama 3 report).

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 9: how can diffusion generate text?::Today's LLMs write one token after another. For text, mask tokens and learn to unmask them (a masked diffusion model), which can decode in fewer passes. For images, add noise and learn to remove it.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 9: which ongoing challenges are listed?::Fixed weights (no continuous learning), hallucinations, personalisation, interpretability and safety, and how to keep up (arXiv, conferences, code and trending papers).

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 9: which papers are named?::"An Image is Worth 16x16 Words" (ViT, has a paper note), and named on slides only: "Visual Instruction Tuning", "The Llama 3 Herd of Models", "Denoising Diffusion Probabilistic Models" and an analog in-memory attention paper.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 9: the open question the note raises?::Will diffusion decoding replace next-token generation for some tasks, as the lecture hints?

#flashcards/courses/concept #flashcards/topic/course-cme295 How was this CME 295 lecture note built, and what was not seen?::From the lecture slides (PDF) read as extracted text. The note did not see the figures, any speaker notes or the recordings, and did not open the course playlist. Papers only "named on the slides" were not read. The lecture title is the note's own label, not the instructors'.
