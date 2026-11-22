---
tags: [lecture, course, CME295, LLM]
course: CME 295
lecture: 4
lecturer: Afshine Amidi and Shervine Amidi
term: Fall 2025
source: Stanford CME 295 lecture 4 slides
status: growing
---

# CME 295 Lecture 4: Training LLMs

> **In one line:** Covers how a model is trained from scratch, why memory is the bottleneck and how it is shared over many GPUs, then how it is tuned for tasks and fine-tuned cheaply. (The lecture title is my own label, not the course's.)

## What the lecture covers
1. **Pretraining.** The paradigm shift from training a model per task to training one model on language and code first. The slides cover scaling, compute counted in floating-point operations, and the cost and challenges. See [[Pre-training]] and [[Scaling Laws]].
2. **The memory bottleneck.** Why a single GPU cannot hold training, then data parallelism (with ZeRO, which removes duplicate copies) and model parallelism. See [[Data Parallelism]] and [[GPU]].
3. **FlashAttention.** Why standard attention writes a big table to slow memory and how a tiled version avoids it, in the forward and the backward pass. See [[FlashAttention]].
4. **Number precision.** How a floating-point number is stored, and mixed precision training to go faster and use less memory. See [[Mixed Precision Training]] and [[Quantization]].
5. **Supervised fine-tuning (instruction tuning).** Turning a model that only continues text into a helpful assistant, with the need for very high-quality data. See [[Instruction Tuning]] and [[Fine-tuning]].
6. **Benchmarks.** What dimensions to test and the "real-life feeling" of using the model. See [[Benchmarks]].
7. **Parameter-efficient fine-tuning.** LoRA (train two small matrices instead of the full one, and swap them to swap tasks), where to apply it, some training-dynamics facts, and QLoRA, which quantizes the frozen weights. See [[LoRA]] and [[QLoRA]].

## Key ideas to remember
- Pretraining teaches the model language and code; later steps change its behaviour ([[Pre-training]]).
- Memory, not arithmetic, is the main constraint, so training is spread across GPUs and made lighter ([[Data Parallelism]], [[FlashAttention]], [[Mixed Precision Training]]).
- Instruction tuning turns a text-continuer into an assistant ([[Instruction Tuning]]).
- LoRA changes a model by training a small add-on, and QLoRA does it on a squeezed model ([[LoRA]], [[QLoRA]]).

## Papers named in the lecture
- "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", Dao et al., 2022 ([[FlashAttention]])
- "LoRA: Low-Rank Adaptation of Large Language Models" (named on the slides)
- "QLoRA: Efficient Finetuning of Quantized LLMs" ([[QLoRA]])
- "NVIDIA H100 Tensor Core GPU", NVIDIA website (cited for the hardware figures)

## Questions I still have
- How do the scaling-law results of lecture 4 change what size model to train for a fixed budget ([[Chinchilla 2022]])?

## What this note is built from
- The lecture slides (PDF), read as extracted text. I did not see the figures, any speaker notes or the recordings, and I did not open the course playlist.
- Topic order and wording of the sections follow the slides. The detail and the tested code live in the concept notes linked above, each of which says what it was checked against.
- Where a paper is only "named on the slides", I did not read it for this note. The concept and paper notes say which abstracts or bodies I did read.
The slides of this lecture also carry the midterm logistics (see the course index).

## Course
- Part of [[CME 295]]. Lecture before: [[CME 295 Lecture 3]]. Lecture after: [[CME 295 Lecture 5]].
- Concept notes this lecture feeds: [[Pre-training]], [[Scaling Laws]], [[Data Parallelism]], [[FlashAttention]], [[Mixed Precision Training]], [[Instruction Tuning]], [[Benchmarks]], [[LoRA]], [[QLoRA]]

## Flashcards
#flashcards/courses

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 4: what does it cover?::Pretraining, the memory bottleneck and parallelism, FlashAttention, number precision, supervised fine-tuning (instruction tuning), benchmarks, and parameter-efficient fine-tuning (LoRA and QLoRA).

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 4: what is the paradigm shift of pretraining?::From training a model per task to training one model on language and code first. The slides cover scaling, compute counted in floating-point operations, and the cost and challenges.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 4: why is memory the bottleneck?::A single GPU cannot hold training. Training is spread with data parallelism (ZeRO removes duplicate copies) and model parallelism.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 4: what does FlashAttention fix?::Standard attention writes a big table to slow memory. A tiled version avoids it, in the forward and the backward pass.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 4: what does mixed precision training do?::It uses lower-precision floating-point numbers to go faster and use less memory. The lecture explains how a floating-point number is stored.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 4: what is the key idea of LoRA and QLoRA?::LoRA trains two small matrices instead of the full one, and they can be swapped to swap tasks. QLoRA quantizes the frozen weights.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 4: which papers are named?::FlashAttention (Dao et al., 2022) and QLoRA (both have concept notes), plus the LoRA paper and the NVIDIA H100 website (cited for hardware figures), named on slides only.

#flashcards/courses/concept #flashcards/topic/course-cme295 CME 295 Lecture 4: what else is on these slides?::The midterm logistics (see the course index).

#flashcards/courses/concept #flashcards/topic/course-cme295 How was this CME 295 lecture note built, and what was not seen?::From the lecture slides (PDF) read as extracted text. The note did not see the figures, any speaker notes or the recordings, and did not open the course playlist. Papers only "named on the slides" were not read. The lecture title is the note's own label, not the instructors'.
