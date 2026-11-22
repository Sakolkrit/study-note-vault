---
tags: [LLM, inference, serving, latency, throughput, GPU, memory]
status: growing
level: 4
tier: 3
---

# LLM Inference

**Inference** is what happens when a trained [[LLM]] answers you: no more learning, just reading the question and writing the reply one piece at a time.

## 🟢 1. Simple (age 12)
Training is like a student studying for years. Inference is the student sitting the test: the knowledge is fixed, and the only job is to write the answer as quickly as possible. The model writes its answer one small word-piece at a time, and for each piece it has to look through everything it already knows, which is a lot of reading.

**Example:** A librarian who must fetch the same huge book from a far shelf for every single word she writes. The book is heavy, so the trip takes longer than the writing. If she answers eight people's questions in the same trip, each trip is shared, and the library gets many more words written per hour. But each person may wait a little longer for each word.

## 🔵 2. How it works
1. **Training versus inference.** In training the model's numbers (its *weights*) are changed again and again using a [[Loss Function]] and [[Backpropagation]]. In inference the weights stay frozen and the model only runs forward. Inference is the part users pay for every day.
2. **Cut the text into pieces.** The question is split into tokens ([[Tokenization]]) and each token becomes a list of numbers.
3. **Prefill.** The model reads the whole question at once. All tokens can be processed side by side, so the [[GPU]] is kept busy with calculation. Along the way it fills the [[KV Cache]] with notes about each token.
4. **Decode.** Now it writes the answer one token at a time. Each new token needs the model to read all its weights and the whole cache again, but it only does a small amount of calculation. This step is limited by how fast the GPU can read its memory, not by how fast it can calculate.
5. **Pick the next token.** The scores become probabilities ([[Softmax]]) and a [[Decoding Strategies]] rule picks one. It is added to the text and the loop repeats until the model writes a stop token.
6. **Streaming.** Because words come out one by one, the server can send each word to your screen as soon as it exists, instead of waiting for the full answer. The total time is the same, but the first word shows up sooner.
7. **Two clocks.** *Latency* is how long one user waits: the time to the first token, then the time per token. *Throughput* is how many tokens the whole machine writes per second for everyone together.
8. **Batching.** Serving several requests in the same step shares the cost of reading the weights, which raises throughput. It also makes each step slower, which raises latency for each user. See [[Continuous Batching]].
9. **Online and offline.** *Online* serving (a chat window) cares about how soon each user sees words, so latency matters. *Offline* serving (summarising a million documents overnight) cares only about total work per hour, so throughput matters and large batches are fine.

## 🟠 3. Technical detail
- **Prefill is compute-bound, decode is memory-bound.** Prefill does many operations for each byte it reads. Decode at a small batch does about 2 operations per weight but must read every weight, so the memory speed sets the time. This is the reason batching helps so much.
- **The cache grows.** Every token in every active request adds keys and values to the [[KV Cache]], so memory, not arithmetic, limits how many requests fit. Fixes: [[PagedAttention]] (no wasted space), [[Grouped-Query Attention]] and [[Multi-head Latent Attention]] (smaller entries), [[Quantization]] (shorter numbers), [[Sliding Window Attention]] (forget old tokens).
- **Other speed-ups:** [[FlashAttention]] (fewer memory trips inside attention), [[Speculative Decoding]] (a small model drafts, the big one checks several tokens in one step), 16-bit numbers as in [[Mixed Precision Training]], and [[Mixture of Experts]] (only some weights are used per token, though all must still be stored).
- **Bottlenecks in practice:** reading weights, reading the cache, sending numbers between several GPUs (see [[Model Parallelism]]), and the waiting caused by uneven request lengths.
- **How to measure:** time to first token, time per output token, tokens per second for the whole server, and the memory used, all at a stated batch size and prompt length. A number without its batch size and prompt length cannot be compared.
- **The model below uses assumed round numbers, not a measurement of any real GPU or server.**

```python
# ASSUMED hardware and model, round numbers for a 7-billion-parameter model on one big GPU. Not measured.
P = 7e9                    # parameters
weights = P * 2            # bytes, 16-bit
bw = 2.0e12                # memory bandwidth, bytes per second
flops = 312e12             # 16-bit compute, operations per second
kv_per_token = 2 * 2 * 32 * 4096   # keys+values, 2 bytes, 32 layers, width 4096
ctx = 1000                 # tokens already in each request's cache

print("ridge point (operations per byte where compute = memory):", round(flops / bw))
for b in (1, 8, 32, 64, 120):
    mem_t = (weights + b * ctx * kv_per_token) / bw       # read weights once + every request's cache
    comp_t = 2 * P * b / flops                            # about 2 operations per parameter per token
    step = max(mem_t, comp_t)
    print(f"batch {b:>3}: step {step*1000:5.1f} ms ({'memory' if mem_t > comp_t else 'compute'}-bound) | per-user speed {1/step:5.1f} tok/s | whole GPU {b/step:7.0f} tok/s")
print("KV cache of one 1000-token request:", round(ctx * kv_per_token / 1e9, 2), "GB | 80 GB GPU minus weights holds", int((80e9 - weights) // (ctx * kv_per_token)), "such requests")
print("prefill of a 1000-token prompt (compute-bound):", round(2 * P * 1000 / flops * 1000, 1), "ms")
```

## 🔴 4. Deep / formal
The time of one decode step with $b$ requests in the batch is the larger of the memory time and the compute time:

$$
t(b)=\max\!\left(\frac{W+b\,\ell\,c}{\text{BW}},\ \frac{2\,P\,b}{F}\right)
$$

**Where:**
- $W$ is the number of bytes of weights, $P$ the number of parameters, $b$ the batch size (requests running together).
- $\ell$ is the tokens already in each request's cache, and $c$ the bytes of cache one token needs.
- $\text{BW}$ is the memory speed of the GPU in bytes per second, and $F$ its calculation speed in operations per second.
- **Matters:** the first term is "read the weights once, plus every request's cache". Without the $b\ell c$ part you would think batching is free; with it, big batches slow down. Without the $\max$ you would miss the point where the GPU stops waiting for memory and starts being busy calculating.

**Example (tiny numbers):** weights $W=14$ GB, cache per request $\ell c=0.5$ GB, $\text{BW}=2$ TB/s, and the compute time is well under 1 ms.
- One request: $(14+0.5)/2000=\mathbf{7.25}$ ms. Eight requests: $(14+8\times0.5)/2000=\mathbf{9}$ ms.
- Eight requests get 8 tokens in 9 ms, against 1 token in 7.25 ms.
- Meaning: the machine writes about $6.4\times$ more tokens per second, while each user's tokens come out 24% slower. That is the throughput versus latency trade.

**Further notes (from the code: an assumed 7-billion-parameter 16-bit model, 1,000 tokens already in each cache, 2 TB/s, 312 trillion operations per second):**
- The "break-even" point where calculation matches memory reading is $312/2=156$ operations per byte. In this table every step stays memory-bound, because the growing cache keeps the memory time larger.
- Batch 1 gives 138 tokens per second for the one user. Batch 32 gives 2,079 tokens per second for the machine (15 times more) while each user gets 65 per second (about half). Batch 120 gives 3,120 per second in total and 26 per user.
- One 1,000-token request needs 0.52 GB of cache, so an 80 GB GPU with 14 GB of weights holds about 125 of them, which caps the batch.
- Prefill of a 1,000-token prompt is about 45 ms in this model (compute-bound), which is a floor for the "time to first token".
- **Limits of this model:** it ignores the time to talk between GPUs, the cost of attention itself and scheduling delays, and real kernels never reach the peak numbers. Real servers must be measured.

**What came before, and how this improved on it:**
- **Before:** generating each new token by running the whole text again from the beginning, and serving one request at a time.
- **Its limit:** most of the work was repeated, and the expensive GPU sat mostly idle waiting for memory.
- **What changed:** keep a [[KV Cache]], batch requests together, and manage the cache memory well (for example [[PagedAttention]]).
- **How much better:** in this note's assumed model, going from batch 1 to batch 32 raises machine throughput from 138 to 2,079 tokens per second (15 times) at the price of per-user speed falling from 138 to 65 tokens per second. For a real system, the vLLM paper's abstract reports 2 to 4 times higher throughput at the same latency than the earlier systems it tested (arXiv 2309.06180, as recorded in [[PagedAttention]]).

## Related
- [[LLM]] · [[KV Cache]] · [[Continuous Batching]] · [[PagedAttention]] · [[FlashAttention]] · [[Speculative Decoding]] · [[Quantization]] · [[Model Parallelism]] · [[Decoding Strategies]] · [[GPU]] · [[Transformer]] · [[Autoregressive Generation]] · [[vLLM]] · [[Local CPU Chatbot]] · [[GGUF]] · [[llama.cpp]] · [[Ollama]]

## Sources
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, 2023 (arXiv 2309.06180, abstract read; see [[PagedAttention]])
- Hardware numbers (2 TB/s, 312 trillion operations per second) are round figures typical of a large data-centre GPU, **from memory, not checked against a data sheet**.
- "Prefill is compute-bound and decode is memory-bound" is a standard explanation in the serving literature; **from memory, not sourced here**. The code's table is the only evidence in this note.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving LLM inference:::Running a trained model to produce an answer. The weights are frozen and the model only runs forward.

#flashcards/ml/reverse #flashcards/topic/llm-serving Prefill:::The model reads the whole prompt at once. All tokens go side by side, so it is compute-bound. It fills the KV cache.

#flashcards/ml/reverse #flashcards/topic/llm-serving Decode:::The model writes the answer one token at a time. Each token needs all weights and the whole cache read again for little calculation, so it is memory-bound.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the time for one decode step and explain each symbol.
?
$$t(b)=\max\!\left(\frac{W+b\,\ell\,c}{\text{BW}},\ \frac{2\,P\,b}{F}\right)$$
- $W$: bytes of weights. $P$: number of parameters. $b$: batch size (requests run together).
- $\ell$: tokens already in each request's cache. $c$: bytes of cache one token needs.
- $\text{BW}$: GPU memory speed in bytes per second. $F$: GPU calculation speed in operations per second.
- First term: read the weights once plus every request's cache. Without $b\ell c$ batching looks free, with it big batches slow down. Second term: the calculation time. The $\max$ shows the point where the GPU stops waiting for memory and starts being limited by calculation.

#flashcards/ml/example #flashcards/topic/llm-serving $W=14$ GB, cache per request $\ell c=0.5$ GB, $\text{BW}=2$ TB/s, compute well under 1 ms. Time for 1 and for 8 requests?::One request: $(14+0.5)/2000=7.25$ ms. Eight: $(14+8\times0.5)/2000=9$ ms. Eight tokens in 9 ms against one in 7.25 ms, about 6.4 times more tokens per second.

#flashcards/ml/cloze #flashcards/topic/llm-serving Decode at a small batch is limited by ==memory speed==, not by ==calculation speed==.

#flashcards/ml/concept #flashcards/topic/llm-serving Why is batching worth it in decode?::The weights are read once for the whole batch, so extra requests add little time until the cache reads or the calculation start to dominate.

#flashcards/ml/cloze #flashcards/topic/llm-serving In training the weights ==change==. In inference they stay ==frozen==.
