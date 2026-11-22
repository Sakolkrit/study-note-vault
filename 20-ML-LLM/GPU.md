---
tags: [GPU, hardware, parallel-computing, deep-learning, LLM, CUDA]
status: growing
level: 4
tier: 3
---

# GPU

## 🟢 1. Simple (age 12)
A GPU (**Graphics Processing Unit**) is a chip with **thousands of tiny workers** that all do simple maths at the same time. It was invented to draw video-game pictures, and it turns out AI needs the very same kind of maths.

**Example:** You have to colour 1,000,000 dots on a poster. One super-smart artist (a normal CPU) colours them one by one, very fast, but still one at a time. A GPU is like hiring 5,000 helpers who each colour a few dots at once. Each helper is not clever, but together they finish far sooner. AI training is mostly "do the same small multiplication on millions of numbers", so the helpers are perfect for it.

## 🔵 2. How it works
- **CPU:** a few powerful cores, great at many different, tricky tasks one after another.
- **GPU:** thousands of simple cores, great at **one task repeated on lots of data** (this idea is called SIMD: *Single Instruction, Multiple Data*).
- AI maths is mostly [[Matrix]] multiplication, which is exactly "multiply and add a huge pile of numbers", the same instruction again and again. So a GPU can do it far faster than a CPU.
- The GPU has its own fast memory (VRAM) next to it. The numbers of the model must fit there, or the work becomes slow or impossible. That is why people say AI is "limited by GPU memory".
- Several GPUs can be linked together so one big model can be shared across them.

![[gpu-simd.jpg]]
*A SIMD unit in a GPU: one instruction (blue arrow, "Inst/Op") is sent to many ALUs (small calculators), and each works on its own input at the same time. Image: Sing0512, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:SIMD_GPGPU.jpg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
Hardware ideas:
- **Streaming multiprocessors (SMs):** the GPU is built of many SMs. Each runs many **threads** in groups (NVIDIA calls a group of 32 a *warp*) that execute the same instruction together.
- **Tensor cores:** special units that multiply small matrix blocks in one step, the reason modern GPUs are so fast at deep learning.
- **Memory levels, fastest to slowest:** registers, on-chip shared memory (SRAM), then main GPU memory (HBM, "VRAM"). Moving data between them is often slower than the maths itself.
- **Number formats:** FP32 (accurate, big), FP16 / BF16 (half the size, much faster, standard for training), FP8 and INT8/INT4 (smaller still, mostly for running models).
- **Software:** CUDA (NVIDIA), ROCm (AMD). Libraries such as PyTorch call optimised kernels, so you rarely write GPU code yourself.

Why it matters for the topics in this vault:
- A [[Transformer]] is mostly big matrix multiplications plus [[Attention]], which fit GPUs very well.
- Running or training an [[LLM]] is limited by (a) maths speed and (b) memory size and memory speed.
- A [[Vector Database]] can use a GPU to compare millions of [[Vector]]s at once.
- Techniques like FlashAttention rearrange the work so less data is moved between slow and fast memory.
- Fine-tuning tricks like LoRA ([[Fine-tuning]]) exist mainly to cut the memory a GPU needs.

Splitting work over many GPUs:
- **Data parallelism:** each GPU holds a full model copy and processes different data, then they average their gradients.
- **Tensor parallelism:** one big matrix is split across GPUs.
- **Pipeline parallelism:** different layers live on different GPUs, like an assembly line.
- Fast links between GPUs (such as NVLink) matter because gradients and activations constantly travel between them.

![[gpu-die.jpg]]
*A real GPU chip: an early NVIDIA G80 engineering sample (the silver package labelled "G80-ENG-A1") on a graphics-card board. Image: Hyins, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:NVIDIA_G80_GPU_Core.jpg), public domain.*

```python
# rough GPU check with illustrative numbers (not a real benchmark)
peak_flops = 100e12        # 100 TFLOP/s of maths
bandwidth  = 2e12          # 2 TB/s memory speed
def speed(intensity):      # intensity = maths done per byte moved
    return min(peak_flops, bandwidth * intensity) / 1e12   # TFLOP/s reached
print(speed(1 / 6))        # 0.33  (adding two vectors: memory-bound)
print(speed(1365))         # 100.0 (4096x4096 matrix multiply: compute-bound)
```

## 🔴 4. Deep / formal
**The roofline model.** A GPU can be slow for two different reasons: it ran out of maths speed, or it is waiting for data. The roofline model says which one limits you:

$$
P_{\text{attainable}}=\min\big(P_{\text{peak}},\ B\cdot I\big),\qquad I=\frac{\text{FLOPs}}{\text{bytes moved}}
$$

**Where:**
- $P_{\text{attainable}}$ is the speed you actually get, in FLOP/s (floating-point operations per second, a measure of maths speed).
- $P_{\text{peak}}$ is the chip's top maths speed, a fixed ceiling from its spec sheet. **Matters:** no program can exceed it.
- $B$ is the memory bandwidth: how many bytes per second can move between memory and the chip.
- $I$ is the **arithmetic intensity**: how many maths operations you do for each byte you load or store. **Matters:** it is the key property of an algorithm. High $I$ means you reuse each loaded number many times.
- $B\cdot I$ is the speed memory can *feed* the maths. $\min(\cdot,\cdot)$ picks the smaller of the two limits. **Matters:** if $B\cdot I$ is smaller, the GPU spends its time waiting for data (**memory-bound**); if $P_{\text{peak}}$ is smaller, the maths units are the bottleneck (**compute-bound**).

**Example (illustrative numbers):** a GPU with $P_{\text{peak}}=100$ TFLOP/s and $B=2$ TB/s. Numbers stored in 16 bits (2 bytes).
- **Adding two vectors:** per element: 1 operation, and $2+2$ bytes read plus $2$ bytes written $=6$ bytes. $I=1/6=0.17$. $B\cdot I=2\cdot10^{12}\cdot0.17=0.33$ TFLOP/s. Speed reached $=\mathbf{0.33}$ TFLOP/s, only $0.3\%$ of the peak. **Memory-bound.**
- **Multiplying two $4096\times4096$ matrices:** maths $=2n^3=1.37\cdot10^{11}$ operations. Data $=3$ matrices $\cdot\,n^2\cdot2$ bytes $=1.0\cdot10^{8}$ bytes. $I=1365$. $B\cdot I=2730$ TFLOP/s, more than the peak, so the speed is $\min=\mathbf{100}$ TFLOP/s. **Compute-bound.** It takes about $1.37\cdot10^{11}/10^{14}=\mathbf{1.4}$ ms.
- **A small $64\times64$ matrix multiply:** $I=64/3=21$, $B\cdot I=43$ TFLOP/s, so only $\mathbf{43\%}$ of the peak. Small matrices underuse the GPU.
- That is why big matrix multiplications love GPUs, while element-by-element operations (and small batches) are limited by memory speed. It is also why [[Attention]] is rewritten to move less data.

**Memory to train a model.** Rule of thumb for mixed-precision training with the Adam optimiser ([[Gradient Descent]]), ignoring activations:

$$
\text{Memory}\approx 16\,N\ \text{bytes}
$$

**Where:**
- $N$ is the number of parameters in the model (for example $7\times10^{9}$ for a "7B" model).
- $16$ is the bytes needed *per parameter*, adding up (from memory of the ZeRO paper; treat as approximate): $2$ for the working weights in FP16, $2$ for their gradients, and $12$ for the optimiser (an FP32 master copy of the weight $=4$, plus Adam's two running averages $=4+4$).
- **Matters:** training needs about **8 times** the memory of the weights alone, because gradients and optimiser state are stored too. Activations (the intermediate results kept for the backward pass) come on top, so the real figure is higher.

**Example (tiny numbers):** a 7-billion-parameter model.
- Training: $16\times7\cdot10^{9}=1.12\cdot10^{11}$ bytes $=\mathbf{112}$ GB. That does not fit on one 80 GB GPU, so the state must be split across several GPUs.
- Just running it in FP16 (weights only): $2\times7\cdot10^{9}=\mathbf{14}$ GB, which fits on one GPU.
- LoRA fine-tuning ([[Fine-tuning]]): keep the 14 GB of frozen weights plus a tiny trainable add-on, so it can fit on a single GPU.

**Compute to train a model.** For a dense [[Transformer]] the total maths is approximately

$$
C\approx 6\,N\,D\ \text{FLOPs}
$$

**Where:**
- $C$ is the total number of floating-point operations for the whole training run.
- $N$ is the number of parameters and $D$ is the number of training tokens.
- $6$ is a rule of thumb (from the scaling-law literature, from memory): about $2$ operations per parameter per token for the forward pass, and about $4$ for the backward pass (computing gradients costs about twice the forward pass).
- **Matters:** it lets you estimate the cost of a run before you start, and it explains the scaling laws in [[LLM]]: more parameters *or* more data both multiply the cost.

**Example (tiny numbers):** $N=7\times10^{9}$, $D=10^{12}$ tokens.
- $C=6\cdot7\cdot10^{9}\cdot10^{12}=\mathbf{4.2\times10^{22}}$ FLOPs.
- One GPU sustaining $100$ TFLOP/s ($10^{14}$): $4.2\cdot10^{22}/10^{14}=4.2\cdot10^{8}$ s $\approx\mathbf{13}$ years.
- $1000$ such GPUs working together: about $\mathbf{4.9}$ days (in practice somewhat longer, since GPUs also spend time talking to each other).
- This is why LLMs are trained on large clusters, not single machines.

**Caveats:**
- **Model FLOPs utilisation (MFU):** real training reaches only a fraction (often 30 to 50%) of the spec-sheet peak because of memory limits, communication and waiting.
- The peak numbers in the examples are **illustrative**, not any specific product. Real chips differ by generation, number format and whether sparsity is used, so check the maker's datasheet.
- Cost drivers beyond the GPU: electricity, cooling, networking, and the time to move data.

**What came before, and how this improved on it:**
- **Before:** a CPU with a few powerful cores, built to run one hard job quickly.
- **Its limit:** neural networks do millions of small, independent multiplications, and a few cores must take them in turn.
- **What changed:** a graphics chip with thousands of simpler cores, first made for drawing pictures and used for general computing from the mid-2000s (NVIDIA CUDA, 2007; from memory, not checked).
- **How much better:** the code below counts steps and does not time any hardware. A 1024 by 1024 matrix product has 1,048,576 independent outputs. With an assumed 16 CPU cores the work takes 67,108,864 steps in turn, and with an assumed 10,000 GPU cores 107,520, which is about 10,000 times fewer than one worker. The core counts are round numbers I chose, not a real chip, and real speedups are smaller because of memory and core speed.

**Check it yourself:**

```python
import math
n = 1024
outputs = n * n; work = n                                # each output number = n multiply-adds, all independent of each other
print(f"{n}x{n} matrix product: {outputs:,} independent outputs, {work:,} multiply-adds each, {outputs * work:,} in total")
for name, workers in (("1 worker", 1), ("16 CPU cores (assumed)", 16), ("10,000 GPU cores (assumed)", 10000)):
    rounds = math.ceil(outputs / workers) * work
    print(f"  {name:28s} {rounds:>14,} steps one after another")
print("the assumed core counts are round numbers, not any particular chip")
```

## Related
- [[Matrix]] · [[Transformer]] · [[Attention]] · [[LLM]] · [[Fine-tuning]] · [[Vector Database]] · [[Gradient Descent]] · [[RLHF]] · [[Pre-training]] · [[Scaling Laws]] · [[Mixture of Experts]] · [[Quantization]] · [[FlashAttention]] · [[PyTorch]] · [[TensorFlow]] · [[LeNet and AlexNet]] · [[KV Cache]] · [[Mixed Precision Training]] · [[Data Parallelism]] · [[PagedAttention]] · [[Docker]] · [[Triton]] · [[CUDA]] · [[Kernel Fusion]]

## Sources
- Williams, Waterman & Patterson, *Roofline: An Insightful Visual Performance Model for Multicore Architectures*, 2009
- Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models*, 2019
- Kaplan et al., *Scaling Laws for Neural Language Models*, 2020 (the $6ND$ estimate)
- Dao et al., *FlashAttention*, 2022

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving GPU:::Graphics Processing Unit. A chip with thousands of simple cores that all do the same kind of maths at once. Built for drawing pictures, and AI needs the same maths.

#flashcards/ml/reverse #flashcards/topic/llm-serving CPU against GPU:::A CPU has a few powerful cores for many different tricky tasks one after another. A GPU has thousands of simple cores for one task repeated on lots of data (SIMD: Single Instruction, Multiple Data).

#flashcards/ml/formula #flashcards/topic/llm-serving Write the roofline speed limit and explain each symbol.
?
$$P_{\text{attainable}}=\min\big(P_{\text{peak}},\ B\cdot I\big),\qquad I=\frac{\text{FLOPs}}{\text{bytes moved}}$$
- $P_{\text{attainable}}$: the speed you actually get, in FLOP/s (floating-point operations per second).
- $P_{\text{peak}}$: the chip's top maths speed, a fixed ceiling from its spec sheet. No program can exceed it.
- $B$: the memory bandwidth, bytes per second moved between memory and the chip.
- $I$: the arithmetic intensity, maths operations per byte loaded or stored. High $I$ means each loaded number is reused many times.
- $B\cdot I$: the speed memory can feed the maths. You get the smaller of the two limits.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the rule of thumb for the cost of training and explain each symbol.
?
$$C\approx 6\,N\,D\ \text{FLOPs}$$
- $C$: the total floating-point operations for the whole training run.
- $N$: the number of parameters. $D$: the number of training tokens.
- $6$: a rule of thumb (from the scaling-law literature, from memory): about 2 operations per parameter per token forward, about 4 backward (gradients cost about twice the forward pass).
- It lets you estimate a run's cost before starting, and both more parameters or more data multiply the cost.

#flashcards/ml/example #flashcards/topic/llm-serving $N=7\times10^9$, $D=10^{12}$ tokens. What are $C$, the time on one 100 TFLOP/s GPU, and on 1000 such GPUs?::$C=6\cdot7\cdot10^9\cdot10^{12}=4.2\times10^{22}$ FLOPs. One GPU at $10^{14}$ FLOP/s: $4.2\cdot10^8$ s, about 13 years. 1000 GPUs: about 4.9 days (in practice longer, since GPUs also spend time talking to each other).

#flashcards/ml/formula #flashcards/topic/llm-serving Write the rule of thumb for the memory needed to train a model and explain each symbol.
?
$$\text{Memory}\approx 16\,N\ \text{bytes}$$
- $N$: the number of parameters (for example $7\times10^{9}$ for a "7B" model).
- $16$: bytes per parameter, from memory of the ZeRO paper (treat as approximate): 2 for working FP16 weights, 2 for their gradients, and 12 for the optimiser (an FP32 master copy 4, plus Adam's two running averages 4 + 4).
- Training needs about 8 times the memory of the weights alone, because gradients and optimiser state are stored too. Activations come on top, so the real figure is higher.

#flashcards/ml/example #flashcards/topic/llm-serving A 7-billion-parameter model: memory to train it against running it in FP16?::Training: $16\times7\cdot10^9=112$ GB, which does not fit on one 80 GB GPU, so the state must be split across GPUs. Running in FP16 (weights only): $2\times7\cdot10^9=14$ GB, which fits on one GPU.

#flashcards/ml/image #flashcards/topic/llm-serving ![[gpu-simd.jpg]]
What does this picture show?
?
A SIMD unit in a GPU: one instruction (blue arrow, "Inst/Op") is sent to many ALUs (small calculators), and each works on its own input at the same time. (From the note's caption: Sing0512, Wikimedia Commons.)

#flashcards/ml/image #flashcards/topic/llm-serving ![[gpu-die.jpg]]
What does this picture show?
?
A real GPU chip: an early NVIDIA G80 engineering sample (the silver package labelled "G80-ENG-A1") on a graphics-card board. (From the note's caption: Hyins, Wikimedia Commons.)

#flashcards/ml/cloze #flashcards/topic/llm-serving AI maths is mostly ==matrix multiplication==, which is the same instruction again and again, so a GPU suits it.

#flashcards/ml/concept #flashcards/topic/llm-serving Why is AI "limited by GPU memory"?::The model's numbers must fit in the GPU's own fast memory (VRAM), or the work becomes slow or impossible.
