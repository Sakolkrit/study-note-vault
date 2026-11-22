---
tags: [LLM, inference, serving, batching, scheduling, throughput]
status: growing
level: 4
tier: 3
---

# Continuous Batching

Serving many users' requests together, and letting a new request take the place of a finished one **at the very next step**, instead of waiting for the whole group to finish.

## 🟢 1. Simple (age 12)
Imagine a bus that must wait until **every** passenger has reached their stop before anyone new may board. Passengers with short trips make the bus sit half empty. Continuous batching is a bus where a new person hops on at every stop that someone leaves, so the seats are almost always full.

**Example:** Eight people each write a story. One needs 5 words and one needs 250. The "wait for everyone" way keeps seven pens idle for most of the time. The "swap in" way hands the idle pen to the next person at once.

## 🔵 2. How it works
1. An [[LLM]] writes one token per step for each request, so each request runs for as many steps as its answer has tokens ([[LLM Inference]]).
2. Answers have very different lengths, and nobody knows in advance how long.
3. **Static batching:** take a fixed group of requests and run them together until the **longest** is finished. Short requests are done early but their slots sit empty, and they are not returned until the group ends.
4. **Continuous batching:** after *every single step*, check which requests have finished. Return them at once and put waiting requests into the free slots.
5. Because new requests join in the middle, the batch changes at every step. The model code must handle requests at different positions, and the [[KV Cache]] of each request must be tracked separately ([[PagedAttention]] is one way to do that).
6. Result: the [[GPU]] does useful work in nearly every slot, short answers go back sooner, and new requests wait less.

## 🟠 3. Technical detail
- Also called *iteration-level scheduling*: the scheduler runs **one model iteration** at a time and decides again after each one.
- The Orca paper (OSDI 2022) calls the idea iteration-level scheduling and pairs it with *selective batching*, so that only some operations are batched across requests of different lengths. I read a summary of this, not the whole paper.
- It does not change the answers. It only changes when each request is run.
- It needs memory for more requests at once, so it works best with a memory manager like [[PagedAttention]] and with the shorter cache entries from [[Grouped-Query Attention]] or [[Quantization]].
- The new request still needs its prefill (reading the question), which is bigger than a decode step, so real servers decide how to mix prefill and decode steps. This is not modelled below.
- **The simulation is my own with invented request lengths (8 to 256 tokens), 8 slots, all 1,000 requests waiting at time zero, and every step costing the same.** Real steps cost different amounts.

```python
import numpy as np

rng = np.random.RandomState(0)
lengths = rng.randint(8, 257, 1000)          # tokens each request will generate (invented lengths)
B = 8                                         # requests the GPU can run together

# static batching: take 8 requests, run until the LONGEST finishes, then take the next 8

t, returned_static = 0, []
for i in range(0, len(lengths), B):
    group = lengths[i:i + B]
    t += group.max()
    returned_static += [t] * len(group)

# continuous batching: when a request finishes, the next waiting one takes its slot on the very next step
queue = list(lengths); slots = []; t = 0; returned_cont = []
while queue or slots:
    while queue and len(slots) < B:
        slots.append(queue.pop(0))
    t += 1
    slots = [s - 1 for s in slots]
    returned_cont += [t] * sum(s == 0 for s in slots)
    slots = [s for s in slots if s > 0]

useful = lengths.sum()
print("tokens to generate:", useful)
print("static:     steps", returned_static[-1], "| slot use", round(100 * useful / (returned_static[-1] * B), 1), "% | mean time until a request is returned", round(np.mean(returned_static), 1))
print("continuous: steps", t, "| slot use", round(100 * useful / (t * B), 1), "% | mean time until a request is returned", round(np.mean(returned_cont), 1))
print("speedup in steps:", round(returned_static[-1] / t, 2))
```

## 🔴 4. Deep / formal
The share of slot-steps doing useful work is

$$
U=\frac{\sum_i \ell_i}{B\cdot T}
$$

**Where:**
- $\ell_i$ is how many steps request $i$ needs (its answer length in tokens), so $\sum_i \ell_i$ is the real work.
- $B$ is the number of slots in the batch.
- $T$ is the total number of steps until everything is done.
- **Matters:** $B\cdot T$ is the work the GPU *could* have done. Static batching makes $T$ larger because each group lasts as long as its longest member, so $U$ drops. Continuous batching keeps $T$ close to $\sum_i\ell_i/B$, so $U$ is close to 1.

**Example (tiny numbers):** three requests of 2, 6 and 2 steps, $B=2$ slots.
- Static: group one is (2, 6), lasting 6 steps; group two is (2) alone, lasting 2 steps. $T=8$, so $U=10/(2\times8)=\mathbf{0.625}$.
- Continuous: both start; the 2-step request ends after step 2 and the third request takes its slot for steps 3 and 4; the 6-step request ends after step 6. $T=6$, so $U=10/(2\times6)=\mathbf{0.833}$.
- Meaning: the same three answers take 6 steps instead of 8, with no change to any answer.

**Further notes:**
- **Simulation result (invented lengths):** 130,965 tokens to write. Static batching needed 28,092 steps (slot use 58.3%); continuous batching needed 16,437 steps (slot use 99.6%), a 1.71 times saving. The mean time until a request is returned fell from 14,054 to 8,081 steps. That number includes time spent waiting in the queue, because everything arrived at time zero.
- The saving depends on how uneven the lengths are. With all requests the same length, the two methods are identical.
- **Order of waiting requests:** a real scheduler must decide who goes first; I used first come, first served.
- **Not modelled:** memory limits (a new request needs cache space), prefill cost, and requests that arrive over time.

**What came before, and how this improved on it:**
- **Before:** request-level (static) batching, where a batch is formed, run until all its requests finish, and then the next batch starts. The Orca paper compares against NVIDIA's FasterTransformer, which works this way.
- **Its limit:** early-finishing requests wait for the longest one, and new requests wait for the whole batch, so the GPU and the users both wait.
- **What changed:** schedule at the level of one iteration instead of one request (Yu et al., Orca, OSDI 2022, pages 521 to 538; the USENIX page and a search summary were read, the paper itself was not).
- **How much better:** the Orca paper's headline result, as reported in the search summary, is 36.9 times higher throughput at the same latency than FasterTransformer on a GPT-3 175B model. That is the authors' own test against one baseline. In this note's invented simulation the saving is 1.71 times fewer steps, and slot use rises from 58.3% to 99.6%.

## Related
- [[LLM Inference]] · [[KV Cache]] · [[PagedAttention]] · [[Grouped-Query Attention]] · [[Quantization]] · [[Speculative Decoding]] · [[GPU]] · [[LLM]] · [[vLLM]]

## Sources
- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models*, OSDI 2022, pages 521 to 538 (USENIX conference page; the 36.9 times figure comes from a web-search summary of the paper's introduction; **paper body not read**)
- The simulation and the tiny example are my own.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving Continuous batching:::After every single step, return finished requests at once and put waiting requests into the free slots, so the GPU slots are almost always full.

#flashcards/ml/reverse #flashcards/topic/llm-serving Static batching:::Run a fixed group of requests together until the longest one finishes. Short requests finish early, but their slots sit empty.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the GPU utilisation of a batch and explain each symbol.
?
$$U=\frac{\sum_i \ell_i}{B\cdot T}$$
- $\ell_i$: how many steps request $i$ needs (its answer length in tokens), so $\sum_i\ell_i$ is the real work.
- $B$: the number of slots in the batch.
- $T$: the total number of steps until everything is done.
- $B\cdot T$ is the work the GPU could have done. Static batching makes $T$ larger because each group lasts as long as its longest member, so $U$ drops. Continuous batching keeps $T$ near $\sum_i\ell_i/B$, so $U$ is near 1.

#flashcards/ml/example #flashcards/topic/llm-serving Requests of 2, 6 and 2 steps, $B=2$ slots. What is $U$ for static and for continuous batching?::Static: groups (2, 6) and (2) last 6 + 2 = 8 steps, so $U=10/(2\times8)=0.625$. Continuous: the third request takes the freed slot, so $T=6$ and $U=10/(2\times6)=0.833$.

#flashcards/ml/cloze #flashcards/topic/llm-serving In continuous batching the batch changes after ==every step==, so each request's ==KV cache== must be tracked separately.

#flashcards/ml/concept #flashcards/topic/llm-serving Why do answers of different lengths hurt static batching?::Nobody knows the length in advance, and the group waits for its longest member, so slots of finished short requests sit empty.
