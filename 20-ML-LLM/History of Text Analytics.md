---
tags: [history, NLP, LLM, text, transformer, timeline]
status: growing
level: 4
tier: 3
---

# History of Text Analytics

How computers moved from counting words to writing and reasoning in language, and what each step fixed.

## 🟢 1. Simple (age 12)
First, computers just **counted words**: a page with "goal" and "match" is about sport. Then they learned that words have **meanings** that can be written as lists of numbers. Then they learned to **read a sentence in order**, then to look at **all the words at once**, and finally to learn from the whole internet and answer questions like a helpful friend.

**Example:** The word "bank". Counting treats it the same everywhere. Number-lists (word2vec) give it one meaning. Modern models read "river bank" and "bank account" and give them different meanings.

## 🔵 2. How it works
1. **Counting words.** Bag of words, then TF-IDF and [[BM25]]. Fast, but order and meaning are lost.
2. **Word vectors** ([[Word2Vec 2013]]). Each word gets a short list of numbers ([[Embedding]]); similar words get similar lists. Limit: one vector per word, whatever the sentence.
3. **Reading in order** ([[From FNN to RNN]]). A [[Recurrent Neural Network]] and LSTM carry a memory along the sentence.
4. **Sequence to sequence** ([[Seq2Seq 2014]]). One network reads, another writes: translation. Limit: the whole sentence squeezed into one vector.
5. **Attention** ([[Attention]]). The writer looks back at every input word.
6. **Contextual vectors.** ELMo (arXiv 1802.05365) made each word's vector depend on its sentence, improving six NLP problems.
7. **The Transformer** ([[Attention Is All You Need 2017]]). Attention only, parallel to train.
8. **Pre-train, then fine-tune** ([[BERT 2018]], and GPT). Learn language from unlabelled text first.
9. **Scale and prompts** ([[GPT-3 2020]]). 175B parameters, tasks given as text ([[In-Context Learning]]).
10. **Make it helpful** ([[Instruction Tuning]], [[RLHF]], [[Direct Preference Optimization]]).
11. **Open, efficient models** ([[LLaMA 2023]]) and better positions ([[RoFormer 2021]]).
12. **Reasoning by reinforcement learning** ([[DeepSeek-R1 2025]], [[GRPO]]).

## 🟠 3. Technical detail
| Step | Year | New idea | Number from the source | Status |
|---|---|---|---|---|
| LSTM | 1997 | gated memory cell | - | date checked on the Wikipedia timeline |
| word2vec | 2013 | cheap word vectors | 1.6 billion words in under a day | abstract read |
| Seq2Seq | 2014 | LSTM encoder and decoder | 34.8 BLEU against 33.3 for phrase-based (WMT'14 En-Fr) | abstract read |
| Attention | 2014 | soft search over the source | comparable to the state-of-the-art phrase-based system | abstract read |
| ELMo | 2018 | context-dependent vectors | improves six NLP problems | abstract read |
| Transformer | 2017 | attention only | 28.4 BLEU En-De, 41.8 En-Fr in 3.5 days on 8 GPUs | abstract read |
| BERT | 2018 | bidirectional pre-training | GLUE 80.5% (+7.7 points) | abstract read |
| GPT-3 | 2020 | scale, few-shot | 175B parameters | abstract read |
| RoPE | 2021 | rotary positions | beat alternatives on long text classification | abstract read |
| LLaMA | 2023 | open models trained longer | 13B beats GPT-3 175B on most benchmarks | abstract read |
| DeepSeek-R1 | 2025 | reasoning from pure RL | emergent self-reflection, verification | abstract read, no figures |

**How a modern LLM works, step by step** (the pieces each have their own note):
1. [[Tokenization]] ([[Byte-Pair Encoding]]) cuts text into tokens.
2. Each token becomes an [[Embedding]].
3. A stack of [[Transformer]] blocks mixes them with [[Self-Attention]], with positions from [[RoPE]]. Faster versions use [[Grouped-Query Attention]], a [[KV Cache]] and [[FlashAttention]]. Some models use [[Mixture of Experts]].
4. [[Pre-training]] on huge text with next-word prediction ([[Scaling Laws]]).
5. [[Instruction Tuning]], then preference tuning ([[RLHF]], [[Direct Preference Optimization]]).
6. For reasoning models, reinforcement learning on checkable answers ([[GRPO]], [[Reasoning Model]]).
7. At use time: [[Decoding Strategies]], [[Speculative Decoding]], [[PagedAttention]], [[Quantization]], and tools through [[Tool Calling]] and [[Agent]]s.
- **What I know about Llama and DeepSeek-R1 beyond the abstracts:** LLaMA's design details (pre-norm, RMS normalisation, SwiGLU, rotary positions) and DeepSeek's sizes (671B total, 37B active) are from memory or the lecture slides. I did not verify them in the papers.

The code turns the published figures into ratios.

```python
# Published numbers (abstracts read) and what they mean as ratios.
print("Seq2Seq LSTM 34.8 BLEU vs phrase-based SMT 33.3 on WMT'14 En-Fr: +", round(34.8 - 33.3, 1), "BLEU")
print("Transformer 41.8 BLEU on En-Fr, trained 3.5 days on 8 GPUs = ", 3.5 * 8 * 24, "GPU-hours")
print("BERT GLUE 80.5%, a 7.7 point gain, so the old best was", round(80.5 - 7.7, 1), "%")
print("GPT-3: 175B parameters. LLaMA-13B beats it on most benchmarks (abstract): size ratio", round(13 / 175, 3), "=", round(175 / 13, 1), "x smaller")
print("word2vec: 1.6 billion words in less than a day (abstract) -> at least", round(1.6e9 / 86400), "words per second")
```

## 🔴 4. Deep / formal
This note has no equation. Reading the chain critically:
- **Two bottlenecks repeat:** squeezing a sentence into a fixed vector (fixed by attention) and doing work one word at a time (fixed by the Transformer). Then a third arrived: attention's cost grows with the square of the length.
- **Scale beat clever design for a while.** GPT-3's lesson was size; LLaMA's was that training longer matters more than size (13B against 175B). The [[Scaling Laws]] note holds the checked numbers.
- **Numbers are not comparable across rows.** BLEU, GLUE and benchmark scores measure different things, so the table is a history, not a leaderboard ([[Benchmarks]]).
- **Reasoning is the least verified link.** Any figures for DeepSeek-R1 are in the paper body, which I did not read. The abstract gives no numbers.
- **Not covered:** GPT-1 and GPT-2 in detail, T5, PaLM, Chinchilla in this note (see [[Pre-training]]), safety work, multimodal models, and speech.

**What came before, and how this improved on it:**
- **Before:** word counts and hand-made rules.
- **Its limit:** no meaning, no order, no context.
- **What changed:** vectors, then recurrence, then attention, then pre-training, then scale, then feedback and reinforcement.
- **How much better:** translation 34.8 BLEU against 33.3 for the phrase-based system (2014); GLUE 80.5%, up 7.7 points (2018); a 13B model beating a 175B model (2023). The ratios are in the code.

## Related
- [[History of AI]] · [[History of Image Classification]] · [[From FNN to RNN]] · [[LLM]] · [[Transformer]] · [[Attention]] · [[Word2Vec]] · [[BERT]] · [[RoPE]] · [[Reasoning Model]] · [[Pre-training]] · [[RLHF]]

## Sources
- arXiv abstracts read: 1301.3781, 1409.3215, 1409.0473, 1802.05365, 1706.03762, 1810.04805, 2005.14165, 2104.09864, 2302.13971, 2501.12948
- Wikipedia *Timeline of artificial intelligence* for the LSTM date
- Stanford CME 295 (Fall 2025) lecture slides for the LLM pipeline and the DeepSeek-R1 recipe

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/history Counting words:::Bag of words, then TF-IDF and BM25. Fast, but order and meaning are lost.

#flashcards/ml/reverse #flashcards/topic/history Word vectors (Word2Vec):::Each word gets a short list of numbers, and similar words get similar lists. Limit: one vector per word, whatever the sentence.

#flashcards/ml/reverse #flashcards/topic/history Reading in order:::An RNN and LSTM carry a memory along the sentence.

#flashcards/ml/reverse #flashcards/topic/history Sequence to sequence:::One network reads, another writes, as in translation. Limit: the whole sentence squeezed into one vector.

#flashcards/ml/reverse #flashcards/topic/history Contextual vectors:::ELMo (arXiv 1802.05365) made each word's vector depend on its sentence, improving six NLP problems.

#flashcards/ml/reverse #flashcards/topic/history The Transformer:::Attention only, parallel to train.

#flashcards/ml/reverse #flashcards/topic/history Pre-train, then fine-tune (BERT and GPT):::Learn language from unlabelled text first, then adapt.

#flashcards/ml/reverse #flashcards/topic/history Scale and prompts (GPT-3):::175B parameters, tasks given as text (in-context learning).

#flashcards/ml/reverse #flashcards/topic/history Make it helpful:::Instruction tuning, RLHF and DPO.

#flashcards/ml/reverse #flashcards/topic/history Open, efficient models and better positions:::LLaMA, and RoFormer for positions.

#flashcards/ml/reverse #flashcards/topic/history Reasoning by reinforcement learning:::DeepSeek-R1 and GRPO.

#flashcards/ml/concept #flashcards/topic/history What limit of word counting did word vectors fix?::Counting loses order and meaning. Word vectors give each word a list of numbers so similar words are close.

#flashcards/ml/concept #flashcards/topic/history What limit of one-vector-per-word did contextual vectors fix?::A word's meaning depends on its sentence. Contextual vectors (ELMo) change with the sentence.

#flashcards/ml/example #flashcards/topic/history What limit of seq2seq did attention fix?::The whole input was squeezed into one vector. With attention the writer looks back at every input word.
