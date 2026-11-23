---
tags: [lecture, course, CME295, LLM]
course: CME 295
lecture: 3
lecturer: Afshine Amidi and Shervine Amidi
term: Fall 2025
source: Stanford CME 295 lecture 3 slides
status: growing
---

# CME 295 Lecture 3: Large language models

> **In one line:** Defines an LLM, explains how a mixture of experts makes it huge but cheap per token, how it picks its next word, how to prompt it, and how to make generation faster. (The lecture title is my own label, not the course's.)

## What the lecture covers
1. **What an LLM is.** A decoder-only Transformer-based model, with the terminology of context length, context size and window size. The slides add a warning about "context rot". See [[LLM]] and [[Context Rot]].
2. **Mixture of experts.** Several expert networks and a router, training problems such as routing collapse, and what the experts learn (Mixtral). See [[Mixture of Experts]] and [[Mixtral 2024]].
3. **Choosing the next token.** Greedy decoding, beam search, sampling among the top k or top p tokens, how the probabilities come from a softmax, how the temperature changes them, and guided decoding that only allows valid outputs. See [[Decoding Strategies]], [[Beam Search]] and [[Softmax]].
4. **Prompting.** The main structure of a prompt, in-context learning (zero-shot and few-shot), chain of thought and self-consistency. See [[In-Context Learning]], [[Chain of Thought]], [[Prompt Engineering]] and [[GPT-3 2020]].
5. **Making generation efficient.** Reusing computation with a KV cache, sharing attention heads (grouped-query), managing memory with PagedAttention, shrinking the cache with latent attention, and speeding up decoding with speculative decoding and multi-token prediction. See [[KV Cache]], [[Grouped-Query Attention]], [[PagedAttention]], [[Multi-head Latent Attention]], [[Speculative Decoding]] and [[Multi-Token Prediction]].

## Key ideas to remember
- An LLM is a very large decoder-only model that writes one token at a time ([[LLM]]).
- A mixture of experts keeps the total size large but uses only a few experts per token ([[Mixture of Experts]]).
- How the next token is picked (greedy, beam, top-k, top-p, temperature) changes the output a lot ([[Decoding Strategies]]).
- Prompts can teach a model a task with examples and with reasoning steps ([[In-Context Learning]], [[Chain of Thought]]).
- Most speed-ups come from remembering and sharing work: the KV cache and its cheaper variants ([[KV Cache]]).

## Papers named in the lecture
- "Mixtral of Experts", Jiang et al., 2024 ([[Mixtral 2024]])
- "Language Models are Few-Shot Learners", Brown et al., 2020 ([[GPT-3 2020]])
- "Context Rot: How Increasing Input Tokens Impacts LLM Performance", Hong et al., 2025 (named on a slide, not read)
- "Efficient Memory Management for Large Language Model Serving with PagedAttention" (named on the slides)
- "DeepSeek-V2" for latent attention (named on the slides; [[Multi-head Latent Attention]])
- "Accelerating Large Language Model Decoding with Speculative Sampling" (named on the slides)

## Questions I still have
- How do the sampling settings interact (temperature together with top-p)?
- When is a mixture of experts the right choice over a dense model of the same active size?

## What this note is built from
- The lecture slides (PDF), read as extracted text. I did not see the figures, any speaker notes or the recordings, and I did not open the course playlist.
- Topic order and wording of the sections follow the slides. The detail and the tested code live in the concept notes linked above, each of which says what it was checked against.
- Where a paper is only "named on the slides", I did not read it for this note. The concept and paper notes say which abstracts or bodies I did read.

## Course
- Part of [[CME 295]]. Lecture before: [[CME 295 Lecture 2]]. Lecture after: [[CME 295 Lecture 4]].
- Concept notes this lecture feeds: [[Mixture of Experts]], [[Decoding Strategies]], [[Beam Search]], [[In-Context Learning]], [[Chain of Thought]], [[KV Cache]], [[Grouped-Query Attention]], [[PagedAttention]], [[Multi-head Latent Attention]], [[Speculative Decoding]], [[Multi-Token Prediction]], [[Context Rot]]

## Flashcards
#flashcards/courses

#flashcards/courses/concept #flashcards/topic/course-cme295 What is an LLM?::A decoder-only Transformer-based model; related terms are context length, context size and window size. Long inputs can suffer from "context rot" (named in the course slides, not read further).

#flashcards/courses/concept #flashcards/topic/course-cme295 How does a mixture of experts work?::Several expert networks and a router. It keeps the total size large but uses only a few experts per token. Training problems include routing collapse.

#flashcards/courses/concept #flashcards/topic/course-cme295 Which ways of choosing the next token exist?::Greedy decoding, beam search, sampling among the top k or top p tokens, probabilities from a softmax, temperature, and guided decoding that only allows valid outputs.

#flashcards/courses/concept #flashcards/topic/course-cme295 What are the main ideas of prompting?::The main structure of a prompt, in-context learning (zero-shot and few-shot), chain of thought and self-consistency.

#flashcards/courses/concept #flashcards/topic/course-cme295 Which tricks make generation efficient?::KV cache, grouped-query attention, PagedAttention, latent attention (DeepSeek-V2), speculative decoding and multi-token prediction. Most speed-ups come from remembering and sharing work.
