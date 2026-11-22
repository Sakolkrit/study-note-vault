---
tags: [training, GPU, parallelism, ZeRO, distributed, LLM, memory]
status: growing
level: 4
tier: 3
---

# Data Parallelism

Training one model on **many GPUs at once** by giving each GPU a slice of the data. **ZeRO** is a way to stop each GPU from wasting memory on identical copies.

## 🟢 1. Simple (age 12)
A giant homework pile is too big for one student, so a team of students shares it: each takes a **slice of the pile**, does their part, and then they **compare answers** and agree on one result. That is data parallelism. The catch is that every student keeps a full copy of the textbook and all their notes, which wastes desk space. ZeRO says: **don't everyone keep the whole book, split the book up between you and borrow pages when needed**.

**Example:** Four students must mark 32 exam papers. Each marks 8, and they average their feelings about "how hard was the exam". The average is the same as if one student had marked all 32.

## 🔵 2. How it works
1. Training a big [[LLM]] needs more memory than one [[GPU]] has (tens of GB). For each parameter the GPU holds: the weight, its gradient, and the optimizer's own numbers ([[Adam Optimizer]]). On top of that come the activations.
2. **Data parallelism:** copy the whole model to each of $N$ devices. Split the batch into $N$ pieces. Each device computes the gradient on its own piece.
3. The devices **average their gradients**, so everyone applies the same update and the copies stay identical. This gives exactly the same gradient as one device processing the entire batch.
4. The waste: every device keeps its own copy of the weights, the gradients and the optimizer numbers, which are identical.
5. **ZeRO** (Zero Redundancy Optimizer) removes that redundancy in three stages: **stage 1** shares out the optimizer numbers; **stage 2** also shares the gradients; **stage 3** also shares the parameters themselves.
6. The price of sharing is that devices must fetch the pieces they need from each other, which costs communication.
7. Data parallelism is one of several ways to split a job. **Model parallelism** splits the model itself: tensor, pipeline, sequence, context and expert parallelism are the variants named in the lecture (not covered here).

## 🟠 3. Technical detail
- **What takes memory (lecture):** parameters, activations (depend on model size, batch size and context length), gradients, and optimizer state. A single H100-class GPU has tens of GB, a limit the lecture calls the bottleneck.
- **ZeRO stages:** ZeRO-1 shares the optimizer state; ZeRO-2 shares optimizer state and gradients; ZeRO-3 shares optimizer state, gradients and parameters (lecture; matches the paper's idea, which I read only the abstract of).
- **Paper claims (abstract, read):** ZeRO removes memory redundancy in data- and model-parallel training while keeping communication low, trained models of over 100 billion parameters with super-linear speed-up on 400 GPUs, and could in principle scale past a trillion parameters (arXiv 1910.02054).
- **Memory accounting used here:** 2 bytes for 16-bit weights, 2 for 16-bit gradients and 12 for the 32-bit copy, momentum and variance, so 16 bytes per parameter, the same as in [[Mixed Precision Training]]. I took this breakdown from my memory of the ZeRO paper and from the arithmetic of that note, and did not find it in the abstract.
- **Trade-off:** more sharing saves more memory but needs more communication between devices.
- **Combine with other tricks:** [[FlashAttention]] (less memory traffic inside attention), mixed precision, and model parallelism for the biggest models.
- **LoRA instead:** if you cannot afford the memory of full training, train a small add-on ([[LoRA]]).

```python
import torch
torch.manual_seed(0); torch.set_num_threads(1)
# 1) data parallelism: 4 devices each take a quarter of the batch; averaging their gradients equals the gradient of the whole batch
X = torch.randn(32, 5, dtype=torch.float64); y = torch.randn(32, dtype=torch.float64)
w = torch.randn(5, dtype=torch.float64, requires_grad=True)
def grad(xb, yb):
    w.grad = None; ((xb @ w - yb) ** 2).mean().backward(); return w.grad.clone()
full = grad(X, y)
shards = [grad(X[i * 8:(i + 1) * 8], y[i * 8:(i + 1) * 8]) for i in range(4)]
print('average of 4 shard gradients equals the full-batch gradient:', torch.allclose(sum(shards) / 4, full))
# 2) memory per parameter in mixed-precision Adam training: weights 2, gradients 2, optimizer state (32-bit copy, momentum, variance) 12 bytes
P, W_, G_, K = 7.5e9, 2, 2, 12
base = (W_ + G_ + K) * P
print('7.5 billion parameters, every device keeps everything:', round(base / 1e9), 'GB per device')
for N in (8, 64, 1024):
    z1 = (W_ + G_ + K / N) * P; z2 = (W_ + (G_ + K) / N) * P; z3 = (W_ + G_ + K) / N * P
    print(f'N = {N:4d} devices | stage 1 (share optimizer state): {z1 / 1e9:6.1f} GB | stage 2 (+gradients): {z2 / 1e9:6.1f} GB | stage 3 (+parameters): {z3 / 1e9:6.2f} GB')
print('stage 1 saving at N = 64:', round(base / ((W_ + G_ + K / 64) * P), 2), 'times | stage 2:', round(base / ((W_ + (G_ + K) / 64) * P), 2), 'times | stage 3:', round(base / ((W_ + G_ + K) / 64 * P), 1), 'times')
```

## 🔴 4. Deep / formal
**Averaging gradients across devices.** With $N$ devices, each with an equal share of the batch and local gradient $g_i$:

$$
g=\frac{1}{N}\sum_{i=1}^{N}g_i
$$

**Where:**
- $N$ is the number of devices and $g_i$ is the gradient computed by device $i$ on its slice of the batch.
- $g$ is the gradient every device then uses to update its copy of the model.
- **Matters:** for a loss that is an average over examples, the average of equal-sized slices' gradients equals the gradient of the whole batch (the code checks this to floating-point accuracy). Without averaging, the copies would drift apart. If the slices were unequal, a plain average would be slightly wrong and a weighted one is needed.

**Example (tiny numbers):** 4 devices with local gradients $1,2,3,6$ for one weight.
- Sum $=12$, divided by $4$: $g=\mathbf{3}$.
- Meaning: all four devices apply the same step of $3$ (times the learning rate), so their copies remain identical.

**Memory per device with ZeRO.** With $\Psi$ parameters, $N$ devices, and $K=12$ bytes of optimizer state per parameter, the bytes held by one device are:

$$
M_1=\left(2+2+\tfrac{K}{N}\right)\Psi,\qquad M_2=\left(2+\tfrac{2+K}{N}\right)\Psi,\qquad M_3=\frac{2+2+K}{N}\,\Psi
$$

**Where:**
- $M_1$, $M_2$ and $M_3$ are the memory per device under ZeRO stages 1, 2 and 3. Plain data parallelism keeps $(2+2+K)\Psi$.
- The first 2 is bytes for a weight and the second 2 is bytes for a gradient (16-bit). $K$ is bytes of optimizer state.
- A term divided by $N$ is the part that is shared out across devices.
- **Matters:** stage 1 can never get below $4\Psi$ bytes (the weights and gradients are still in every copy), and stage 2 can never get below $2\Psi$. Only stage 3 shrinks everything in proportion to $N$. Activations and communication buffers are ignored here, and they also take memory.

**Example (tiny numbers):** $\Psi=7.5\times10^9$ parameters, $N=64$ devices, $K=12$.
- Plain: $16\times7.5=\mathbf{120}$ GB per device.
- Stage 1: $(4+12/64)\times7.5=4.1875\times7.5=\mathbf{31.4}$ GB.
- Stage 2: $(2+14/64)\times7.5=2.21875\times7.5=\mathbf{16.6}$ GB.
- Stage 3: $16/64\times7.5=\mathbf{1.88}$ GB.
- Meaning: savings of $3.8$, $7.2$ and $64$ times (the code prints the same). The 120 GB would not even fit on one GPU; stage 3 at 1.88 GB would, before counting activations.

**Further notes:**
- **A floor:** at $N=1024$ the code gives $30.1$ GB for stage 1 and $15.1$ GB for stage 2, because the unshared 4 and 2 bytes per parameter remain.
- **Not tested here:** real multi-GPU speed or communication volume (no GPUs here), and accuracy. The "7.5 billion" size is an arbitrary round example.
- **Not covered:** the other parallelisms, overlap of communication and compute, and offloading to the CPU.

**What came before, and how this improved on it:**
- **Before:** plain data parallelism (every device holds a full copy of everything) and model parallelism (splitting the layers or matrices, harder to apply).
- **Its limit:** the ZeRO abstract says these hit fundamental limits in fitting models into limited device memory while keeping speed and ease of use.
- **What changed:** share out the optimizer state, then gradients, then parameters, so nothing is held redundantly (Rajbhandari et al., arXiv 1910.02054, 4 October 2019, abstract read).
- **How much better:** the abstract reports an 8 times larger trainable model and 10 times higher performance than the state of the art, and models of up to 13B parameters without model parallelism. In this note's arithmetic, a 7.5B-parameter model needs 120 GB per device with plain data parallelism and 31.4, 16.6 and 1.88 GB with stages 1, 2 and 3 on 64 devices. The note counts parameter state only.

## Related
- [[Mixed Precision Training]] · [[GPU]] · [[Adam Optimizer]] · [[LoRA]] · [[FlashAttention]] · [[LLM]] · [[Pre-training]] · [[Stochastic Gradient Descent]] · [[Gradient Descent]] · [[Model Parallelism]]

## Sources
- Stanford CME 295 (Fall 2025), Lecture 4 slides, Afshine and Shervine Amidi
- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*, 2019 (arXiv 1910.02054, abstract read)
- Hugging Face, *The Ultra-Scale Playbook*, 2025 (named in the lecture; not read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving Data parallelism:::Copy the whole model to each of $N$ devices, give each a slice of the batch, and average their gradients so every copy applies the same update.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the averaged gradient and explain each symbol.
?
$$g=\frac{1}{N}\sum_{i=1}^{N}g_i$$
- $N$: the number of devices. $g_i$: the gradient computed by device $i$ on its slice of the batch.
- $g$: the gradient every device then uses to update its copy.
- For a loss that is an average over examples, the average of equal-sized slices' gradients equals the gradient of the whole batch. Without averaging the copies would drift apart. Unequal slices would need a weighted average.

#flashcards/ml/example #flashcards/topic/llm-serving 4 devices with local gradients 1, 2, 3, 6. What is $g$?::Sum 12, divided by 4: $g=3$. All four devices apply the same step, so their copies stay identical.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the memory per device under ZeRO stages 1, 2 and 3 and explain each symbol.
?
$$M_1=\left(2+2+\tfrac{K}{N}\right)\Psi,\qquad M_2=\left(2+\tfrac{2+K}{N}\right)\Psi,\qquad M_3=\frac{2+2+K}{N}\,\Psi$$
- $M_1,M_2,M_3$: memory per device at ZeRO stages 1, 2, 3. Plain data parallelism keeps $(2+2+K)\Psi$.
- $\Psi$: the number of parameters. The first 2 is bytes per weight, the second 2 bytes per gradient (16-bit). $K$: bytes of optimizer state.
- A term divided by $N$ is the part shared out across devices.
- Stage 1 never gets below $4\Psi$ bytes, stage 2 never below $2\Psi$. Only stage 3 shrinks everything in proportion to $N$. Activations and communication buffers are ignored here.

#flashcards/ml/example #flashcards/topic/llm-serving $\Psi=7.5\times10^9$, $N=64$, $K=12$. What is the memory per device for plain, stage 1, 2 and 3?::Plain: $16\times7.5=120$ GB. Stage 1: $(4+12/64)\times7.5=31.4$ GB. Stage 2: $(2+14/64)\times7.5=16.6$ GB. Stage 3: $16/64\times7.5=1.88$ GB. Savings of 3.8, 7.2 and 64 times.

#flashcards/ml/reverse #flashcards/topic/llm-serving ZeRO:::Zero Redundancy Optimizer. Instead of every device keeping identical copies of weights, gradients and optimizer numbers, these are split between devices and borrowed when needed.

#flashcards/ml/concept #flashcards/topic/llm-serving What is wasteful in plain data parallelism?::Every device keeps its own identical copy of the weights, the gradients and the optimizer numbers.
