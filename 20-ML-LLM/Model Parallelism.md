---
tags: [training, inference, parallelism, GPU, distributed, LLM]
status: growing
level: 4
tier: 3
---

# Model Parallelism

Splitting **one model** across several [[GPU]]s because it is too big for one. (Data parallelism does the opposite: every GPU holds a whole copy and the *data* is split. See [[Data Parallelism]].)

## 🟢 1. Simple (age 12)
A giant jigsaw puzzle does not fit on one table. You put half on one table and half on another, and people at each table work on their half, passing pieces across when needed.

**Example:** A car factory has a line: one worker fits the wheels, the next paints, the next checks. Every worker holds only their own tools (a piece of the model). While one car is being painted, the next can already be getting wheels.

## 🔵 2. How it works
1. A modern [[LLM]] has billions of numbers (weights). Just storing them takes more memory than one GPU has, and training needs several times more.
2. **Pipeline parallelism** cuts the model by *layers*. GPU 1 holds the first layers, GPU 2 the next, and so on. The data flows from GPU to GPU like a line in a factory.
3. If one batch goes through, only one GPU works at a time and the rest wait. To fix that, cut the batch into **micro-batches** and send them one after another, so that all GPUs are busy at once. The idle time left at the start and end is called the **bubble**.
4. **Tensor parallelism** cuts *inside* a layer. One big [[Matrix]] multiplication is split: each GPU holds a slice of the weight matrix and computes its slice of the answer, and then the slices are glued together or added up.
5. Both kinds need the GPUs to **send numbers to each other** often, so the speed of the links between GPUs matters a lot.
6. Real systems mix all of these: model parallelism inside a machine, [[Data Parallelism]] across machines. [[Mixture of Experts]] is a related idea in which different GPUs hold different experts.
7. The same ideas help [[LLM Inference]] when a model is too big for one GPU, at the cost of sending numbers between GPUs for every token.

## 🟠 3. Technical detail
- **Column split:** if the weight matrix is cut into column halves $W=[W_1\ W_2]$, each GPU computes $XW_i$ and the results are glued side by side. No adding is needed.
- **Row split:** if it is cut into row halves, each GPU takes the matching half of the inputs, produces a *partial sum*, and the partial sums are added together (an *all-reduce*, one message exchange among all GPUs).
- **Pairing them in a transformer:** the Megatron-LM paper splits the two big matrices of the feed-forward part and the attention heads this way, so that only a few all-reduces are needed per layer (see [[Transformer]]).
- **Pipeline bubble:** with $p$ stages and $m$ micro-batches, the share of time a stage is idle is $(p-1)/(m+p-1)$. More micro-batches shrink the bubble, but each one is smaller, which uses the GPU less well and needs more memory to store results for the backward pass.
- **Memory:** each GPU holds about $1/p$ of the weights in pipeline parallelism (or $1/t$ with $t$-way tensor parallelism). The activations and the optimizer state need their own care.
- **The code below** checks the matrix splits with [[NumPy]] and simulates the forward-only pipeline schedule. It does not use real GPUs, so it says nothing about speed or communication cost.

```python
import numpy as np

rng = np.random.RandomState(0)
X, W = rng.randn(4, 6), rng.randn(6, 8)
full = X @ W
# tensor parallel, split the weight COLUMNS over 2 devices: each makes half of the outputs, then glue them
col = np.concatenate([X @ W[:, :4], X @ W[:, 4:]], axis=1)
# split the weight ROWS: each device holds half of the inputs and makes a partial sum, then ADD them (an all-reduce)
row = X[:, :3] @ W[:3] + X[:, 3:] @ W[3:]
print("column split matches:", np.allclose(full, col), "| row split matches:", np.allclose(full, row))
print("weights per device:", W.size // 2, "of", W.size)

def pipeline_steps(stages, micro):
    """forward only: stage s can start micro-batch m at step s+m. Returns total steps."""
    return max(s + m for s in range(stages) for m in range(micro)) + 1

for p, m in [(4, 1), (4, 4), (4, 16), (4, 64)]:
    steps = pipeline_steps(p, m)
    busy = p * m                                  # stage-steps that do real work
    print(f"{p} stages, {m:>2} micro-batches: {steps:>2} steps, devices busy {100 * busy / (steps * p):.1f}% (formula {100 * m / (m + p - 1):.1f}%)")
```

## 🔴 4. Deep / formal
With $p$ pipeline stages and $m$ equal micro-batches (forward pass only, each stage taking one time unit per micro-batch), the share of time that stages do useful work is

$$
U=\frac{m}{m+p-1}
$$

**Where:**
- $p$ is the number of stages (GPUs in the pipeline) and $m$ the number of micro-batches.
- The pipeline needs $m+p-1$ time units in total: $m$ for the micro-batches to enter one by one, plus $p-1$ for the last one to reach the end.
- **Matters:** the $p-1$ is the *bubble*, the cost of filling and emptying the line. With $m=1$ the formula gives $1/p$: only one stage is ever working. If $m$ is much bigger than $p$, $U$ goes toward 1.

**Example (tiny numbers):** $p=4$ stages.
- One micro-batch ($m=1$): $U=1/4=\mathbf{0.25}$, so three GPUs of four idle at any time.
- Four micro-batches ($m=4$): $U=4/(4+3)=4/7=\mathbf{0.571}$.
- Sixteen ($m=16$): $U=16/19=\mathbf{0.842}$.
- Meaning: cutting the batch into more pieces fills the line, and the bubble matters less.

**Further notes:**
- **Simulation (forward only):** 4 stages gave 25.0%, 57.1%, 84.2% and 95.5% busy for 1, 4, 16 and 64 micro-batches, matching the formula in every row. The column and row splits both reproduced the full matrix product exactly, with half the weights on each device (24 of 48).
- **Training is more complicated:** the backward pass doubles the work and the schedule (which micro-batch goes forward or backward when) changes the bubble and the memory. Not simulated.
- **Not covered:** communication cost, which often decides the real speed; sequence and expert parallelism; and how to choose $p$ and $t$ for a real cluster.

**What came before, and how this improved on it:**
- **Before:** one model on one GPU, or [[Data Parallelism]] with a full copy of the model on every GPU.
- **Its limit:** a model whose weights do not fit on one GPU cannot be trained or served at all, however many copies you make.
- **What changed:** cut the model itself. GPipe (arXiv 1811.06965, submitted 16 November 2018) uses pipeline parallelism with micro-batches; Megatron-LM (arXiv 1909.08053, submitted 17 September 2019) splits the matrices inside each layer (abstracts read).
- **How much better:** the GPipe abstract reports a 6-billion-parameter, 128-layer Transformer and "almost linear speedup" with more accelerators, with no single figure. The Megatron-LM abstract reports models up to 8.3 billion parameters on 512 GPUs, sustaining 15.1 petaflops with 76% scaling efficiency against a 39-teraflops one-GPU baseline. In this note's simulation, the share of busy devices goes from 25.0% to 95.5% as micro-batches go from 1 to 64 on a 4-stage pipeline.

## Related
- [[Data Parallelism]] · [[GPU]] · [[Mixture of Experts]] · [[Transformer]] · [[Matrix]] · [[NumPy]] · [[LLM]] · [[LLM Inference]] · [[Gradient Accumulation]] · [[Mixed Precision Training]]

## Sources
- Huang et al., *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism*, arXiv 1811.06965 (submitted 16 Nov 2018; abstract read)
- Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*, arXiv 1909.08053 (submitted 17 Sep 2019; abstract read)
- The all-reduce description and the bubble formula's use in GPipe are **from memory**, not checked in the papers' bodies; the formula is checked by the simulation above.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving Model parallelism:::Split one model across several GPUs because it is too big for one.

#flashcards/ml/reverse #flashcards/topic/llm-serving Pipeline parallelism:::Cut the model by layers. GPU 1 holds the first layers, GPU 2 the next, and the data flows from GPU to GPU like a factory line. The batch is cut into micro-batches so all GPUs stay busy.

#flashcards/ml/reverse #flashcards/topic/llm-serving Tensor parallelism:::Cut inside a layer. One big matrix multiplication is split, and each GPU holds a slice of the weight matrix and computes its slice of the answer.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the pipeline utilisation and explain each symbol.
?
$$U=\frac{m}{m+p-1}$$
- $p$: the number of stages (GPUs in the pipeline). $m$: the number of micro-batches.
- The pipeline needs $m+p-1$ time units in total: $m$ for the micro-batches to enter one by one, plus $p-1$ for the last one to reach the end.
- The $p-1$ is the bubble, the cost of filling and emptying the line. With $m=1$ the formula gives $1/p$: only one stage works at a time. If $m$ is much bigger than $p$, $U$ goes toward 1.

#flashcards/ml/example #flashcards/topic/llm-serving $p=4$ stages. What is $U$ for $m=1$, $4$ and $16$?::$m=1$: $1/4=0.25$ (three of four GPUs idle). $m=4$: $4/7=0.571$. $m=16$: $16/19=0.842$. More micro-batches fill the line and shrink the bubble.

#flashcards/ml/cloze #flashcards/topic/llm-serving The idle time at the start and end of a pipeline is called the ==bubble==.
