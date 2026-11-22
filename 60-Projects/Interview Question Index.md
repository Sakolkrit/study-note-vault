---
tags: [project, RAG, retrieval, evaluation, LLM, index]
status: built
stack: Python 3.9, sentence-transformers, NumPy
goal: for each of 115 public LLM interview questions, show which vault notes the search returns, and list where the vault falls short
---

# Interview Question Index

> **In one line:** a lookup table. For each of the 115 questions in a public LLM interview list, it shows the three notes that [[Vault Search with RAG]] returns, and then lists the questions the vault answers badly.

This note lives in `60-Projects` on purpose. The search skips that folder, because a note full of questions would answer its own questions (see "What went wrong" in [[Vault Search with RAG]]).

## How to read it
- **Source of the questions:** *LLM Interview Questions and Answers Hub* by Kalyan KS (GitHub, Apache-2.0). Only the question titles are used, not its answers.
- **Notes column:** the top three notes from the blended search ([[BM25]] keywords plus meaning with [[Embedding Model]], mixed by reciprocal rank). Order matters: first is the best guess.
- **Last column** compares those results with my own answer key (the notes I think should answer the question):
  - ✅ a key note is in the top three
  - 🟡 a key note is only 4th or 5th
  - ❌ no key note in the top five
  - ➖ the vault has no note for this topic, so there is no key
- **Warning:** the key is mine and generous for broad questions. It was changed twice *after* I wrote notes for the questions: 15 labels after writing [[LLM Inference]], [[Continuous Batching]], [[Gradient Accumulation]] and [[Model Parallelism]], and 7 more (Q30, 35, 56, 57, 71, 85, 93) after writing [[Cross-Entropy Loss]], [[Feed-Forward Network]], [[Autoregressive Generation]], [[Hallucination]], [[Catastrophic Forgetting]] and [[vLLM]]. For those 22 questions a ✅ proves only that the new note is findable.
- **This table is the re-score of 14 November 2026** (260 notes, 3,165 chunks). The first version (252 notes at the time, then 3,065 chunks) is described under "Re-score" below.

## Result in short
With the updated key: of 115 questions, 105 ✅, 6 🟡 (Q1, 14, 73, 91, 98, 105), 4 ❌ (Q2, 32, 59, 108) and 0 ➖. Leaving out the 22 questions whose key I changed after writing notes for them, 83 of the other 93 are ✅ (89%), against 86 of 93 (92%) in the earlier run. So the new notes fixed the questions they were written for, and cost a little elsewhere. See the next section.

## Re-score (14 November 2026)
**What was run:** the same 115 questions, the same search code (BM25 plus MiniLM, reciprocal rank fusion), re-indexed after 11 new notes ([[Autoregressive Generation]], [[Cross-Entropy Loss]], [[Feed-Forward Network]], [[Hallucination]], [[Catastrophic Forgetting]], [[Continual Learning]], [[vLLM]], [[ReLU]], [[Matrix]], [[Eigenvalue]], [[Fisher Information]]), plus backlinks added to about 40 older notes. Two keys: the **old key** (the one the first table used) and the **new key** (old key plus the new notes for the 7 questions they were written for).

| Search | Key | Top 1 | Top 3 | Top 5 |
|---|---|---|---|---|
| blend (hybrid) | old, 114 questions | 73 (0.64) | 100 (0.88) | 108 (0.95) |
| blend (hybrid) | new, 115 questions | 79 (0.69) | 105 (0.91) | 111 (0.97) |
| meaning only | old | 81 (0.71) | 101 (0.89) | 106 (0.93) |
| keyword only | old | 48 (0.42) | 77 (0.68) | 92 (0.81) |
| blend, 93 unbiased questions | old, earlier run | | 86 (0.92) | 91 (0.98) |
| blend, 93 unbiased questions | old, now | | 83 (0.89) | 89 (0.96) |

**What got better** (under the new key, where the new notes count): Q35 now has [[Feed-Forward Network]] first. Q56 and Q57 have [[Autoregressive Generation]] first. Q30 has [[Cross-Entropy Loss]] first. Q85 has [[Hallucination]] first, and its best-chunk similarity rose from 0.28 to 0.48. Q93 has [[Catastrophic Forgetting]] first. Q71 has [[vLLM]] third. All of these are biased tests, because I wrote each note after reading its question.

**What got worse** (old key, 5 questions, none of them a question the new notes were written for):
- Q2 (basic steps of an LLM inference query), 🟡 to ❌: [[vLLM]] took the third slot and pushed [[LLM]] out of the top five.
- Q32 (limits of the Transformer), 🟡 to ❌: [[Autoregressive Generation]] took third place.
- Q73 (ways to speed up inference), ✅ to 🟡: [[vLLM]] displaced [[Quantization]].
- Q14 (the embedding layer), ✅ to 🟡: [[Feed-Forward Network]] displaced [[Embedding]] from the top three.
- Q91 (alignment tuning), ✅ to 🟡: the key note [[Fine-tuning]] slipped to fourth. I did not trace why.

**Does a new note steal good slots?** Partly. [[vLLM]] is in the top three for 12 questions but is a key note for only one (Q71); most are serving questions where it is a fair answer, but it caused two of the five regressions. [[Continual Learning]] is in the top three of 3 questions (Q8, 93, 111) and is a key note for none; Q8 (large vocabularies) and Q111 are not about it. [[ReLU]], [[Matrix]], [[Eigenvalue]] and [[Fisher Information]] are in no top three. Net effect on the 93 unbiased questions: top-3 86 to 83, top-5 91 to 89, from one run.

**A guess I did not test:** the search indexes each note's "Related" line, and I added new names to about 40 of those lines. A line full of unrelated note names gives a chunk many words to match by accident, which may explain some of the odd matches (for example [[LoRA]] in the top three for Q34). Dropping the Related and Sources sections from the index is the cheap test.

**Caution:** this is one run on one embedding model. A change of 3 questions out of 93 is small, and I did not repeat it with a different model or key.

## The table

| #   | Question                                                                                                                            | Top 3 notes                                                                                 |     |
| --- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | --- |
| 1 | CNNs and RNNs don’t use positional embeddings. Why do transformers use positional embeddings? | [[CME 295 Lecture 2]] · [[Vision Transformer]] · [[Recurrent Neural Network]] | 🟡 |
| 2 | Tell me the basic steps involved in running an inference query on an LLM. | [[LLM Inference]] · [[Continuous Batching]] · [[vLLM]] | ❌ |
| 3 | Explain how KV Cache accelerates LLM inference. | [[LLM Inference]] · [[KV Cache]] · [[Continuous Batching]] | ✅ |
| 4 | How does quantization affect inference speed and memory requirements? | [[Quantization]] · [[Mixed Precision Training]] · [[LLM Inference]] | ✅ |
| 5 | How do you handle the large memory requirements of KV cache in LLM inference? | [[LLM Inference]] · [[KV Cache]] · [[CME 295 Lecture 3]] | ✅ |
| 6 | After tokenization, how are tokens converted into embeddings in the Transformer model? | [[BERT]] · [[Embedding]] · [[CME 295 Lecture 1]] | ✅ |
| 7 | Explain why subword tokenization is preferred over word-level tokenization in the Transformer model. | [[Tokenization]] · [[BERT]] · [[Transformer]] | ✅ |
| 8 | Explain the trade-offs in using a large vocabulary in LLMs. | [[CME 295 Lecture 6]] · [[Byte-Pair Encoding]] · [[Continual Learning]] | ✅ |
| 9 | Explain how self-attention is computed in the Transformer model step by step. | [[Self-Attention]] · [[CME 295 Lecture 1]] · [[Transformer]] | ✅ |
| 10 | What is the computational complexity of self-attention in the Transformer model? | [[Self-Attention]] · [[CME 295 Lecture 1]] · [[Transformer]] | ✅ |
| 11 | How do Transformer model address the vanishing gradient problem? | [[Vanishing and Exploding Gradients]] · [[Activation Function]] · [[Residual Connection]] | ✅ |
| 12 | What is tokenization, and why is it necessary in LLMs? | [[Tokenization]] · [[Byte-Pair Encoding]] · [[CME 295 Lecture 1]] | ✅ |
| 13 | Explain the role of token embeddings in the Transformer model. | [[BERT]] · [[Embedding]] · [[Multi-Token Prediction]] | ✅ |
| 14 | Explain the working of the embedding layer in the transformer model. | [[Vision Transformer]] · [[Feed-Forward Network]] · [[Positional Encoding]] | 🟡 |
| 15 | What is the role of self-attention in the Transformer model, and why is it called “self-attention”? | [[Self-Attention]] · [[Multi-head Latent Attention]] · [[Transformer]] | ✅ |
| 16 | What is the purpose of the encoder in a transformer model? | [[CME 295 Lecture 2]] · [[Vision Transformer]] · [[Transformer]] | ✅ |
| 17 | What is the purpose of the decoder in a transformer model? | [[LLM]] · [[Transformer]] · [[CME 295 Lecture 2]] | ✅ |
| 18 | How does the encoder-decoder structure work at a high level in the Transformer model? | [[CME 295 Lecture 2]] · [[Transformer]] · [[Variational Autoencoder]] | ✅ |
| 19 | What is the purpose of scaling in the self-attention mechanism in the Transformer model? | [[Self-Attention]] · [[Transformer]] · [[Multi-head Latent Attention]] | ✅ |
| 20 | Why does the Transformer model use multiple self-attention heads instead of a single self-attention head? | [[Self-Attention]] · [[Multi-head Latent Attention]] · [[Grouped-Query Attention]] | ✅ |
| 21 | How are the outputs of multiple heads combined and projected back in the multi-head attention in the Transformer model? | [[Transformer]] · [[Grouped-Query Attention]] · [[Multi-head Latent Attention]] | ✅ |
| 22 | How does masked self-attention differ from regular self-attention, and where is it used in a Transformer? | [[Self-Attention]] · [[Multi-head Latent Attention]] · [[Transformer]] | ✅ |
| 23 | Discuss the pros and cons of the self-attention mechanism in the Transformer model. | [[Self-Attention]] · [[Transformer]] · [[Multi-head Latent Attention]] | ✅ |
| 24 | What is the purpose of masked self-attention in the Transformer decoder? | [[Self-Attention]] · [[Transformer]] · [[CME 295 Lecture 1]] | ✅ |
| 25 | Explain how masking works in masked self-attention in Transformer. | [[Self-Attention]] · [[Masked Diffusion Language Model]] · [[CME 295 Lecture 9]] | ✅ |
| 26 | Explain why self-attention in the decoder is referred to as cross-attention. How does it differ from self-attention in the encoder? | [[Transformer]] · [[Attention]] · [[Self-Attention]] | ✅ |
| 27 | What is the softmax function, and where is it applied in the Transformer model? | [[Softmax]] · [[Machine Learning Specialization Course 2 Week 2]] · [[Transformer]] | ✅ |
| 28 | What is the purpose of residual (skip) connections in Transformer model layers? | [[Residual Connection]] · [[Backpropagation]] · [[Normalization]] | ✅ |
| 29 | Why is layer normalization used, and where is it applied in Transformer? | [[Layer Normalization]] · [[Normalization]] · [[Vision Transformer]] | ✅ |
| 30 | What is cross-entropy loss, and how is it applied during transformer training? | [[Cross-Entropy Loss]] · [[Entropy]] · [[Probability]] | ✅ |
| 31 | Compare transformers and RNNs in terms of handling long-range dependencies. | [[Recurrent Neural Network]] · [[From FNN to RNN]] · [[CME 295]] | ✅ |
| 32 | What are the fundamental limitations of the Transformer model? | [[CME 295 Lecture 2]] · [[Attention Is All You Need 2017]] · [[Autoregressive Generation]] | ❌ |
| 33 | How do transformers address the limitations of traditional deep learning models like CNNs and RNNs? | [[Convolutional Neural Network]] · [[Vision Transformer]] · [[Recurrent Neural Network]] | ✅ |
| 34 | How does Transformer model address the vanishing gradient problem? | [[Vanishing and Exploding Gradients]] · [[Residual Connection]] · [[LoRA]] | ✅ |
| 35 | What is the purpose of the position-wise feed-forward sublayer in the Transformer model? | [[Feed-Forward Network]] · [[Vision Transformer]] · [[Attention Is All You Need 2017]] | ✅ |
| 36 | Can you briefly explain the difference between LLM training and LLM inference? | [[LLM Inference]] · [[Continuous Batching]] · [[Mixture of Experts]] | ✅ |
| 37 | What is latency in LLM inference, and why is it important? | [[LLM Inference]] · [[Continuous Batching]] · [[LLM]] | ✅ |
| 38 | What is batch inference, and how does it differ from single-query inference? | [[Continuous Batching]] · [[LLM Inference]] · [[PagedAttention]] | ✅ |
| 39 | How does batching generally help with LLM inference efficiency? | [[Continuous Batching]] · [[LLM Inference]] · [[vLLM]] | ✅ |
| 40 | Explain the trade-offs between batching and latency in LLM serving. | [[LLM Inference]] · [[PagedAttention]] · [[Continuous Batching]] | ✅ |
| 41 | How can techniques like mixture-of-experts (MoE) optimize inference efficiency? | [[Mixture of Experts]] · [[Quantization]] · [[Model Parallelism]] | ✅ |
| 42 | Explain the role of decoding strategy in LLM text generation. | [[Decoding Strategies]] · [[Autoregressive Generation]] · [[LLM Inference]] | ✅ |
| 43 | What are the different decoding strategies in LLMs? | [[Decoding Strategies]] · [[Beam Search]] · [[LLM Inference]] | ✅ |
| 44 | Explain the impact of the decoding strategy on LLM-generated output quality and latency. | [[LLM Inference]] · [[Decoding Strategies]] · [[CME 295 Lecture 3]] | ✅ |
| 45 | Explain the greedy search decoding strategy and its main drawback. | [[Beam Search]] · [[Decoding Strategies]] · [[Speculative Decoding]] | ✅ |
| 46 | How does Beam Search improve upon Greedy Search, and what is the role of the beam width parameter? | [[Beam Search]] · [[Decoding Strategies]] · [[Greedy Algorithms]] | ✅ |
| 47 | When is a deterministic strategy (like Beam Search) preferable to a stochastic (sampling) strategy? Provide a specific use case. | [[Beam Search]] · [[Decoding Strategies]] · [[PagedAttention]] | ✅ |
| 48 | Discuss the primary trade-off between the computational cost and the output quality when comparing Greedy Search and Beam Search. | [[Beam Search]] · [[Decoding Strategies]] · [[Greedy Algorithms]] | ✅ |
| 49 | When you set the temperature to 0.0, which decoding strategy are you using? | [[Decoding Strategies]] · [[Speculative Decoding]] · [[Beam Search]] | ✅ |
| 50 | How is Beam Search fundamentally different from a Breadth-First Search (BFS) or Depth-First Search (DFS)? | [[Graphs]] · [[Beam Search]] · [[Decoding Strategies]] | ✅ |
| 51 | Explain the criteria for choosing different decoding strategies. | [[Decoding Strategies]] · [[Beam Search]] · [[Speculative Decoding]] | ✅ |
| 52 | Compare deterministic and stochastic decoding methods in LLMs. | [[Decoding Strategies]] · [[LLM Inference]] · [[Beam Search]] | ✅ |
| 53 | What is the role of the context window during LLM inference? | [[LLM Inference]] · [[CME 295 Lecture 3]] · [[Context Rot]] | ✅ |
| 54 | Explain the pros and cons of large and small context windows in LLM inference. | [[LLM Inference]] · [[In-Context Learning]] · [[Context Rot]] | ✅ |
| 55 | What is the purpose of temperature in LLM inference, and how does it affect the output? | [[LLM Inference]] · [[Softmax]] · [[Decoding Strategies]] | ✅ |
| 56 | What is autoregressive generation in the context of LLMs? | [[Autoregressive Generation]] · [[LLM Inference]] · [[vLLM]] | ✅ |
| 57 | Explain the strengths and limitations of autoregressive text generation in LLMs. | [[Autoregressive Generation]] · [[LLM Inference]] · [[vLLM]] | ✅ |
| 58 | Explain how diffusion language models (DLMs) differ from Large Language Models (LLMs). | [[Masked Diffusion Language Model]] · [[Diffusion Models]] · [[CME 295]] | ✅ |
| 59 | Do you prefer DLMs or LLMs for latency-sensitive applications? | [[vLLM]] · [[LLM Inference]] · [[PagedAttention]] | ❌ |
| 60 | Explain the concept of token streaming during inference. | [[LLM Inference]] · [[Autoregressive Generation]] · [[Multi-Token Prediction]] | ✅ |
| 61 | What is speculative decoding, and when would you use it? | [[Speculative Decoding]] · [[CME 295 Lecture 3]] · [[Beam Search]] | ✅ |
| 62 | What are the challenges in performing distributed inference across multiple GPUs? | [[Model Parallelism]] · [[GPU]] · [[LLM Inference]] | ✅ |
| 63 | How would you design a scalable LLM inference system for real-time applications? | [[LLM Inference]] · [[Continuous Batching]] · [[vLLM]] | ✅ |
| 64 | Explain the role of flash attention in reducing memory bottlenecks during inference. | [[LLM Inference]] · [[FlashAttention]] · [[Sliding Window Attention]] | ✅ |
| 65 | What is continuous batching, and how does it differ from static batching? | [[Continuous Batching]] · [[LLM Inference]] · [[PagedAttention]] | ✅ |
| 66 | What is mixed precision (e.g., FP16) and why is it used during inference? | [[Mixed Precision Training]] · [[Precision]] · [[Scaling Laws]] | ✅ |
| 67 | Differentiate between online and offline LLM inference deployment scenarios and discuss their respective requirements. | [[Continuous Batching]] · [[LLM Inference]] · [[vLLM]] | ✅ |
| 68 | Explain the throughput vs. latency tradeoff in LLM inference. | [[LLM Inference]] · [[vLLM]] · [[KV Cache]] | ✅ |
| 69 | What are the various bottlenecks in a typical LLM inference pipeline when running on a modern GPU? | [[Model Parallelism]] · [[LLM Inference]] · [[Continuous Batching]] | ✅ |
| 70 | How do you measure LLM inference performance? | [[LLM Inference]] · [[Continuous Batching]] · [[vLLM]] | ✅ |
| 71 | What are the different LLM inference engines available? Which one do you prefer? | [[LLM Inference]] · [[Continuous Batching]] · [[vLLM]] | ✅ |
| 72 | What are the challenges in LLM inference? | [[LLM Inference]] · [[Continuous Batching]] · [[vLLM]] | ✅ |
| 73 | What are the possible options for accelerating LLM inference? | [[LLM Inference]] · [[Continuous Batching]] · [[vLLM]] | 🟡 |
| 74 | What is Chain-of-Thought (CoT) prompting, and when is it most useful? | [[Chain of Thought]] · [[Prompt Engineering]] · [[ReAct]] | ✅ |
| 75 | Explain the reason behind the effectiveness of Chain-of-Thought (CoT) prompting. | [[Chain of Thought]] · [[Prompt Engineering]] · [[CME 295 Lecture 6]] | ✅ |
| 76 | Explain the trade-offs in using CoT prompting. | [[Chain of Thought]] · [[GPT-3 2020]] · [[Prompt Engineering]] | ✅ |
| 77 | What is prompt engineering, and why is it important for LLMs? | [[Prompt Engineering]] · [[Prompt Caching]] · [[Chain of Thought]] | ✅ |
| 78 | What is the difference between zero-shot and few-shot prompting? | [[In-Context Learning]] · [[GPT-3 2020]] · [[Prompt Engineering]] | ✅ |
| 79 | What are the different approaches for choosing examples for few-shot prompting? | [[In-Context Learning]] · [[Prompt Engineering]] · [[GPT-3 2020]] | ✅ |
| 80 | Why is context length important when designing prompts for LLMs? | [[In-Context Learning]] · [[Prompt Engineering]] · [[Context Rot]] | ✅ |
| 81 | What is a system prompt, and how does it differ from a user prompt? | [[Prompt Engineering]] · [[Prompt Caching]] · [[Agent Safety]] | ✅ |
| 82 | What is In-Context Learning (ICL), and how is few-shot prompting related? | [[In-Context Learning]] · [[CME 295 Lecture 3]] · [[GPT-3 2020]] | ✅ |
| 83 | What is self-consistency prompting, and how does it improve reasoning? | [[Chain of Thought]] · [[Prompt Engineering]] · [[Reasoning Model]] | ✅ |
| 84 | Why is context important in designing prompts? | [[In-Context Learning]] · [[Prompt Engineering]] · [[Context Rot]] | ✅ |
| 85 | Describe a strategy for reducing hallucinations via prompt design. | [[Hallucination]] · [[Prompt Engineering]] · [[Prompt Caching]] | ✅ |
| 86 | How would you structure a prompt to ensure the LLM output is in a specific format, like JSON? | [[Prompt Engineering]] · [[In-Context Learning]] · [[LLM Inference]] | ✅ |
| 87 | Explain the purpose of ReAct prompting in AI Agents. | [[ReAct]] · [[Agent]] · [[Prompt Engineering]] | ✅ |
| 88 | What are the different phases in LLM development? | [[LLM Inference]] · [[LLM as a Judge]] · [[LLM]] | ✅ |
| 89 | What are the different types of LLM fine-tuning? | [[Instruction Tuning]] · [[Fine-tuning]] · [[LLM]] | ✅ |
| 90 | What role does instruction tuning play in improving an LLM’s usability? | [[Instruction Tuning]] · [[LLM]] · [[Fine-tuning]] | ✅ |
| 91 | What role does alignment tuning play in improving an LLM's usability? | [[LLM]] · [[InstructGPT 2022]] · [[Instruction Tuning]] | 🟡 |
| 92 | How do you prevent overfitting during fine-tuning? | [[Fine-tuning]] · [[Overfitting]] · [[Catastrophic Forgetting]] | ✅ |
| 93 | What is catastrophic forgetting, and why is it a concern in fine-tuning? | [[Catastrophic Forgetting]] · [[Continual Learning]] · [[Fine-tuning]] | ✅ |
| 94 | What are the strengths and limitations of full fine-tuning? | [[Fine-tuning]] · [[CME 295 Lecture 4]] · [[Instruction Tuning]] | ✅ |
| 95 | Explain how parameter efficient fine-tuning addresses the limitations of full fine-tuning. | [[CME 295 Lecture 4]] · [[Instruction Tuning]] · [[Fine-tuning]] | ✅ |
| 96 | When might prompt engineering be preferred over task-specific fine-tuning? | [[Prompt Engineering]] · [[Instruction Tuning]] · [[In-Context Learning]] | ✅ |
| 97 | When should you use fine-tuning vs. RAG? | [[Fine-tuning]] · [[RAG]] · [[Instruction Tuning]] | ✅ |
| 98 | Explain the limitations of using RAG over LLM fine-tuning. | [[Fine-tuning]] · [[LLM]] · [[Instruction Tuning]] | 🟡 |
| 99 | Explain the limitations of using LLM fine-tuning over RAG. | [[Fine-tuning]] · [[LLM]] · [[RAG]] | ✅ |
| 100 | When should you prefer task-specific fine-tuning over prompt engineering? | [[Prompt Engineering]] · [[Instruction Tuning]] · [[In-Context Learning]] | ✅ |
| 101 | What is LoRA, and how does it work at a high level? | [[LoRA]] · [[CME 295 Lecture 4]] · [[QLoRA]] | ✅ |
| 102 | Explain the key ingredient behind the effectiveness of the LoRA technique. | [[LoRA]] · [[CME 295 Lecture 4]] · [[LLaMA 2023]] | ✅ |
| 103 | What is QLoRA, and how does it differ from LoRA? | [[QLoRA]] · [[CME 295 Lecture 4]] · [[LoRA]] | ✅ |
| 104 | When would you use QLoRA instead of standard LoRA? | [[QLoRA]] · [[CME 295 Lecture 4]] · [[LoRA]] | ✅ |
| 105 | How would you handle LLM fine-tuning on consumer hardware with limited GPU memory? | [[Fine-tuning]] · [[LLM Inference]] · [[GPU]] | 🟡 |
| 106 | Explain different preference alignment methods and their trade-offs. | [[CME 295 Lecture 5]] · [[Direct Preference Optimization]] · [[InstructGPT 2022]] | ✅ |
| 107 | What is gradient accumulation, and how does it help with fine-tuning large models? | [[Gradient Accumulation]] · [[Fine-tuning]] · [[Gradient Descent]] | ✅ |
| 108 | What are the possible options to speed up LLM fine-tuning? | [[Fine-tuning]] · [[Instruction Tuning]] · [[Gradient Descent]] | ❌ |
| 109 | Explain the pretraining objective used in LLM pretraining. | [[CME 295 Lecture 4]] · [[Instruction Tuning]] · [[Self-Supervised Learning]] | ✅ |
| 110 | What is the difference between casual language modeling and masked language modeling? | [[Masked Diffusion Language Model]] · [[BERT]] · [[CME 295 Lecture 9]] | ✅ |
| 111 | How do LLMs handle out-of-vocabulary (OOV) words? | [[Tokenization]] · [[Word2Vec]] · [[Continual Learning]] | ✅ |
| 112 | In the context of LLM pretraining, what is scaling law? | [[CME 295 Lecture 4]] · [[LLM]] · [[Scaling Laws]] | ✅ |
| 113 | Explain the concept of Mixture-of-Experts (MoE) architecture and its role in LLM pretraining. | [[Mixture of Experts]] · [[Mixtral 2024]] · [[CME 295 Lecture 3]] | ✅ |
| 114 | What is model parallelism, and how is it used in LLM pre-training? | [[Data Parallelism]] · [[Model Parallelism]] · [[Pre-training]] | ✅ |
| 115 | What is the significance of self-supervised learning in LLM pretraining? | [[Self-Supervised Learning]] · [[Supervised Learning]] · [[Transfer Learning]] | ✅ |

## Gaps: what the vault answers badly
Each item is a question group, why it fails, and the fix. "Missing note" means I searched the vault's file names and text for the topic.

| Questions | Problem | Kind | Fix |
|---|---|---|---|
| 35 (position-wise feed-forward layer) | ❌. Several notes mention feed-forward layers in passing ([[BERT]], [[Layer Normalization]], [[LoRA]]), but none explains this layer | missing note | **Done**: [[Feed-Forward Network]] written; re-score: it is first for Q35. Biased test, but it also took a top-3 slot from [[Embedding]] on Q14 |
| 56, 57 (autoregressive generation, strengths and limits) | ❌. Top results are diffusion and lecture notes. No note defines "one token at a time, left to right" and its limits | missing note | **Done:** [[Autoregressive Generation]] written; re-score: first for both questions. Not a fair test: I wrote it after reading them. It also took third place on Q32 (see below) |
| 30 (cross-entropy loss) | 🟡. Results are [[Entropy]], probability and KL divergence; no note is named for the loss itself (I checked file names only) | missing note | **Done**: [[Cross-Entropy Loss]] written; re-score: first for Q30 (biased test) |
| 85 (reduce hallucinations by prompt design) | weakest match of all 115 (best-chunk similarity 0.28 of 1). It is marked ✅ only because [[Prompt Engineering]] is a generous key; that note never mentions hallucination | missing topic | **Done**: [[Hallucination]] written; re-score: first for Q85, best-chunk similarity 0.28 to 0.48 (biased test) |
| 93 (catastrophic forgetting) | ✅ by the key, but the term lives only in [[Fine-tuning]], with no note of its own | thin coverage | **Done**: [[Catastrophic Forgetting]] written; re-score: first for Q93 (biased test) |
| 71 (inference engines such as vLLM) | ➖. A list of products; needs a source read for each | missing note, needs sources | **Done**: [[vLLM]] written from its paper; no "which engine is best" claim (other engines not read); re-score: third for Q71, and first for Q59 where it does not belong |
| 59 (diffusion vs normal LLMs for latency) | ❌, still. [[Masked Diffusion Language Model]] exists, but search returns [[vLLM]] and inference notes first | ranking or note wording | add one comparison paragraph to [[Masked Diffusion Language Model]] |
| 108, 105 (speed up fine-tuning; fine-tune on small GPU) | ❌ and 🟡, unchanged. The answer is spread over [[LoRA]], [[QLoRA]], [[Mixed Precision Training]] and [[Gradient Accumulation]]; no note gathers it | spread across notes | add a "making fine-tuning cheaper" section to [[Fine-tuning]] |
| 98 (limits of RAG vs fine-tuning) | 🟡, unchanged. [[Fine-tuning]] ranks first; [[RAG]] is 4th or 5th | ranking | add a "when RAG is the wrong tool" section to [[RAG]] |
| 1 | 🟡. [[Positional Encoding]] is fourth | ranking | none needed; the answer is findable |
| 2, 32 | ❌ now (were 🟡). Q2: [[vLLM]] pushed [[LLM]] out of the top five. Q32: [[Autoregressive Generation]] took third place, and the key notes ([[Transformer]], [[Self-Attention]]) are outside the top five | ranking, caused by new notes | check whether dropping the Related lines from the index helps; otherwise accept |
| 14, 73, 91 | 🟡 now (were ✅). New notes took a top-three slot on Q14 and Q73; Q91 slipped without a clear cause | ranking | same test |

How the gaps were found: from the table above. A 🟡/❌/➖ flag comes from my key, and the "missing" claims come from searching file names and text. I did **not** read the answers in the source repo, so a topic could be missing from the vault yet still be covered well enough by a note I did not think of.

## Weak spots that look fine in the table
- The 8 questions with the lowest best-chunk similarity now score between 0.42 and 0.52 (earlier run: the lowest 12 were 0.28 to 0.53). The median over all 115 is 0.64 (was 0.63). The lowest five are Q47, Q85, Q76, Q32 and Q79 (0.42 to 0.50). The similarity numbers are from the MiniLM model and are only a rough guide, not a calibrated score.
- Course lecture notes ([[CME 295 Lecture 2]] and others) show up often for broad questions, because each lecture summarises many topics. No question had only lecture notes in its top three (checked again in the re-score).

## What was not checked
- The table was not opened in Obsidian, so I have not seen how the wide table renders.
- The answer key is mine. A second person would label some rows differently.
- No answer text was generated. The index points at notes; it does not say the notes answer the question well.
- These are interview questions, not the questions you type while studying. Real questions from you would be the better test.
- The re-score is one run with one embedding model, and the effect of the new notes on the other questions is a few questions out of 93. The "Related line" explanation for odd matches was not tested.
- I did not re-run the answer-generation or `qa_eval.py` steps, and I did not update the numbers inside [[Vault Search with RAG]]; they describe the earlier 252-note index.

## Related
- [[Vault Search with RAG]] · [[Retrieval Metrics]] · [[RAG]] · [[LLM]] · [[BM25]] · [[Embedding Model]]
