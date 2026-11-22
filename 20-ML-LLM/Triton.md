---
tags: [GPU, Triton, kernels, programming, LLM, training, performance]
status: growing
level: 4
tier: 3
---

# Triton

**Triton** is a programming language and compiler, used from Python, for writing fast custom programs (called kernels) that run on a graphics card. Many AI tools, such as [[Unsloth]], use it to speed up the maths of training.

## 🟢 1. Simple (age 12)
A graphics card is a huge kitchen with thousands of tiny cooks. Writing instructions for all of them one by one is hard, and a small mistake makes the kitchen slow. Triton lets you write the recipe for one team of cooks in short, simple steps, and it works out the tricky details of how to run the whole kitchen well.

**Example:** You want to add two lists of a million numbers. You write "this team adds numbers 0 to 1023, the next team adds the next 1024", and Triton makes sure all the teams work at once.

## 🔵 2. How it works
1. **A GPU runs the same small program many times at once.** Each copy works on its own piece of the data ([[GPU]], [[Vectorization]]).
2. **You write the program for one piece.** In Triton that piece is a block of values, for example 1,024 numbers. Each copy is called a **program** and has an ID.
3. **Each program finds its own numbers from its ID.** The first handles positions 0 to 1023, the second 1024 to 2047, and so on.
4. **It loads the numbers, does the maths, and stores the result.** The last program may run past the end of the data, so you add a **mask** that says which positions are real.
5. **You launch many programs together.** The **grid** says how many: the data size divided by the block size, rounded up.
6. **The compiler handles the hard parts.** The authors say Triton automates GPU optimisations, so you can focus on the maths instead of managing individual threads ([[CUDA]]).
7. **Why people bother.** Doing several steps inside one kernel (called [[Kernel Fusion]]) means the numbers are read from slow memory once instead of many times, and for memory-bound work that is where the time goes ([[FlashAttention]] and [[Unsloth]] use this idea).
8. **The catch:** Triton needs a supported GPU and is hard to debug, and a fast kernel is still a lot of work. I could not run Triton here (no NVIDIA GPU and it is not installed).

![[cuda-thread-hierarchy.png]]
*A CUDA launch written `printHelloGpu<<<2, 5>>>()`: the green rectangle is the **grid**, it holds 2 grey **blocks**, and each block holds 5 blue **threads**. In CUDA you think in threads. In a Triton kernel you write the program for one block and the threads inside it are managed by the compiler (my reading of the Triton tutorial, which has a program ID but no thread index). The Commons description (in Catalan) says the numbers of threads and blocks are set in the call. Image: Ilze Lucero, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Jerarquia_de_fils_d%27execuci%C3%B3_CUDA.png), public domain. The picture is tiny (433 by 222 pixels).*

## 🟠 3. Technical detail
- **What it is.** The Triton README describes it as "a language and compiler for writing highly efficient custom Deep-Learning primitives". It lists Linux, NVIDIA GPUs (Compute Capability 8.0 and up) and AMD GPUs (ROCm 6.2 and up), with CPUs "under development" (README as read on 16 November 2026). Windows is not mentioned there, while other tools that use Triton say they run on Windows, so check the version you install.
- **Origin.** The paper is *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations* (MAPL 2019). OpenAI's announcement, dated 28 July 2021, is by Philippe Tillet, who started the project as a Harvard graduate student (from search results; I could not open the OpenAI page itself, which returned an error).
- **The authors' claim.** Tillet says Triton lets researchers with no CUDA experience write GPU code that is often close to what an expert produces, and that it can write FP16 matrix multiplication kernels matching the performance of cuBLAS (from search results quoting the post). A "fewer than 25 lines" detail appeared in only one secondary source, so I treat it as unverified.
- **The programming model.** You write a function marked `@triton.jit`. Each instance gets a `program_id`, builds a range of offsets with `tl.arange`, loads with `tl.load`, computes, and stores with `tl.store`. Sizes known at compile time are `tl.constexpr`. A `mask` guards the edge. The launch grid is `triton.cdiv(n, BLOCK_SIZE)` programs (Triton vector-add tutorial).
- **Who uses it.** The [[Unsloth]] blog says it rewrites PyTorch modules into Triton kernels. The Triton tutorials show a fused softmax and a matrix multiplication (I read only the vector-add page).

The Triton kernel below is rebuilt from the lines the tutorial's vector-add page describes. **I did not run it**: it needs a supported GPU and the `triton` package, and neither is here.

```python
# NOT RUN. Needs a supported GPU and `pip install triton`.
import torch, triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                      # which block am I?
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements                      # guard the end of the vector
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)

def add(x, y):
    output = torch.empty_like(x)
    n = output.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
    return output
```

The next snippet is my own and **was run**. It models the same launch in NumPy (one function call per program) and does the arithmetic of the equations below. The bandwidth (1000 GB/s) and peak speed (20 TFLOP/s) in it are assumed round numbers for illustration, not the spec of any real GPU.

```python
import math
import numpy as np

def add_program(pid, x, y, out, n, BLOCK):
    offsets = pid * BLOCK + np.arange(BLOCK)      # which elements this program owns
    mask = offsets < n                            # the last program may run past the end
    idx = offsets[mask]
    out[idx] = x[idx] + y[idx]                    # load, add, store (only the unmasked lanes)
    return int(mask.sum())

n, BLOCK = 98432, 1024                            # the tutorial's vector size and block size
x, y = np.random.RandomState(0).rand(n).astype("float32"), np.random.RandomState(1).rand(n).astype("float32")
out = np.empty(n, dtype="float32")
grid = math.ceil(n / BLOCK)                       # triton.cdiv(n, BLOCK)
lanes = [add_program(p, x, y, out, n, BLOCK) for p in range(grid)]
print("programs:", grid, " lanes launched:", grid * BLOCK, " wasted lanes:", grid * BLOCK - n)
print("last program active lanes:", lanes[-1], " total active:", sum(lanes), " correct:", np.array_equal(out, x + y))

N, b, k = 1_000_000, 4, 3                         # a chain of k elementwise ops on N float32 values
unfused, fused = k * 2 * N * b, 2 * N * b
print("unfused bytes:", unfused, " fused bytes:", fused, " ratio:", unfused // fused)
BW = 1000e9                                       # ASSUMED bandwidth in bytes per second
print("time at BW 1000 GB/s: unfused %.2f us, fused %.2f us" % (unfused / BW * 1e6, fused / BW * 1e6))
I = 1 / (3 * 4)                                   # vector add: 1 FLOP, 2 reads + 1 write of 4 bytes
print("intensity:", round(I, 4), "FLOP/byte; memory-bound speed:", round(BW * I / 1e9, 1), "GFLOP/s")
PEAK = 20e12                                      # ASSUMED peak in FLOP/s
print("attainable:", min(PEAK, BW * I) / 1e9, "GFLOP/s, share of peak:", round(100 * min(PEAK, BW * I) / PEAK, 2), "%")
```

It printed: `programs: 97`, `lanes launched: 99328`, `wasted lanes: 896`, `last program active lanes: 128`, `total active: 98432`, `correct: True`. Then `unfused bytes: 24000000`, `fused bytes: 8000000`, `ratio: 3`, times 24.00 and 8.00 microseconds, intensity 0.0833, memory-bound speed 83.3 GFLOP/s, and 0.42% of the assumed peak.

## 🔴 4. Deep / formal
**How many programs, and how many lanes are wasted.** For $N$ elements and block size $B$:

$$
G=\left\lceil\frac{N}{B}\right\rceil,\qquad \text{wasted lanes}=G\,B-N
$$

**Where:**
- $N$ is the number of elements. **Matters:** it fixes the amount of real work.
- $B$ is the block size (`BLOCK_SIZE`), the number of elements one program handles. **Matters:** a larger $B$ means fewer programs but more wasted lanes at the edge, and it must be known at compile time.
- $G$ is the grid size, the number of programs launched. **Matters:** it should be big enough to keep every part of the GPU busy.
- $GB-N$ is the lanes past the end of the data. **Matters:** the mask turns them off, so they cost almost nothing, but without the mask they would read and write memory outside the vector.

**Example (tiny numbers):** the tutorial's size, $N=98{,}432$ and $B=1024$.
- $98432/1024=96.125$, rounded up to $G=97$.
- $97\times1024=99{,}328$ lanes, so $99{,}328-98{,}432=896$ are wasted.
- The first 96 programs are full ($96\times1024=98{,}304$), so the last one has $98{,}432-98{,}304=128$ active lanes.
- Meaning: one extra, mostly empty program covers the tail. The NumPy model gave 97, 896 and 128 and its sum matched `x + y` exactly.

**Why fusing steps helps.** Take a chain of $k$ elementwise operations on $N$ values of $b$ bytes each. Run as separate kernels, each one reads $N$ values from memory and writes $N$ back. Fused into one kernel, the data is read once and written once:

$$
\text{bytes}_{\text{unfused}}=2kNb,\qquad \text{bytes}_{\text{fused}}=2Nb,\qquad \frac{\text{unfused}}{\text{fused}}=k
$$

**Where:**
- $k$ is the number of operations in the chain. **Matters:** the saving grows with $k$, so long chains of cheap operations gain most.
- $N$ and $b$ are the number of values and bytes per value. **Matters:** they set the size of each read and write, and both versions scale with them.
- The factor 2 is one read plus one write per pass over the data. **Matters:** the intermediate results never have to touch slow memory in the fused version, they stay in fast on-chip storage.
- When a kernel is limited by memory speed (see the roofline equation in [[GPU]]), time is about bytes divided by bandwidth. **Matters:** then fewer bytes means a proportionally shorter time. If the kernel were limited by arithmetic instead, fusion would help less.

**Example (tiny numbers):** $N=1{,}000{,}000$ float32 values ($b=4$) and $k=3$ operations.
- Unfused: $2\times3\times10^6\times4=24{,}000{,}000$ bytes. Fused: $2\times10^6\times4=8{,}000{,}000$ bytes.
- At an assumed 1000 GB/s: $24\ \mu\text{s}$ against $8\ \mu\text{s}$, a factor of 3.
- Meaning: three times less memory traffic and, if the kernel is memory-bound, about three times faster. The bandwidth is my assumption, not a measurement.

**Why simple kernels are memory-bound.** The roofline limit from [[GPU]] is $P=\min(P_{\text{peak}},B\cdot I)$, with arithmetic intensity $I$ = floating-point operations per byte moved. For vector add:

$$
I=\frac{1\ \text{FLOP}}{3\times4\ \text{bytes}}=\frac1{12}\ \text{FLOP/byte}
$$

**Where:**
- 1 FLOP is the one addition per element. **Matters:** it is almost no arithmetic.
- $3\times4$ bytes is two reads and one write of a 4-byte float. **Matters:** the data movement dominates.
- $I$ is the intensity. **Matters:** a small $I$ puts the kernel on the sloped, memory-limited part of the roofline, where only moving fewer bytes helps.

**Example (tiny numbers):** assume $B=1000$ GB/s and $P_{\text{peak}}=20$ TFLOP/s.
- $B\cdot I=1000\times10^9/12=83.3$ GFLOP/s.
- $\min(20{,}000,\,83.3)=83.3$ GFLOP/s, which is $0.42\%$ of the assumed peak.
- Meaning: the GPU spends almost all its time waiting for memory, so no clever arithmetic would speed this up. That is why the tutorial reports bandwidth (GB/s) instead of speed in FLOP/s for this kernel.

**Honest limits and open questions**
- **Not run.** Triton is not installed here and there is no supported GPU, so no Triton kernel ran. The tutorial kernel is rebuilt from a description of its lines, and the NumPy model only mimics the block logic.
- **The tutorial's result.** The page I read says Triton and Torch give identical bandwidth at every size (peaking at about 1684 GB/s for 134,217,728 elements) and a maximum difference of 0.0. It did not name the GPU in what I read, so I cannot say what that bandwidth is a fraction of. This is a simple kernel where PyTorch is already fast, so it shows correctness more than a speed-up.
- **The OpenAI page.** I could not open it. The performance claims about cuBLAS come from search results quoting it, and the "fewer than 25 lines" figure from a single secondary source.
- **Not covered:** Triton's internals (the tile-based intermediate language and compiler passes), how it picks memory layouts, auto-tuning block sizes, debugging, and how it compares with hand-written CUDA across workloads.
- **Fusion is not free.** A fused kernel uses more on-chip registers and memory, and the best fusion depends on the GPU and the sizes.

**What came before, and how this improved on it:**
- **Before:** writing GPU kernels by hand in CUDA, thinking in threads, memory access patterns and scheduling, or using fixed vendor libraries such as cuBLAS ([[CUDA]]).
- **Its limit:** hand-tuned CUDA needs deep expertise, and a library gives you only the operations it already has, so a custom fused operation was either slow or a lot of work.
- **What changed:** a Python-embedded language where you write the program for one block of data and a compiler handles the low-level optimisation (the MAPL 2019 paper, with Tillet as the OpenAI announcement's author, then announced by OpenAI on 28 July 2021; I did not read the paper, the other authors' names are unchecked, and the OpenAI page was not openable).
- **How much better:** published claims, not mine. The authors say Triton can write FP16 matrix multiplication kernels that match cuBLAS, and in the vector-add tutorial Triton and Torch show equal bandwidth. In my own run, only the arithmetic was checked: fusing 3 elementwise operations cuts memory traffic from 24 MB to 8 MB for a million floats, and a vector add reaches 0.42% of an assumed peak because it is memory-bound.

**Check it yourself:** run the second code block in level 3 (it needs only NumPy). The Triton kernel needs a supported GPU and the `triton` package, which I did not have.

## Related
[[GPU]] · [[CUDA]] · [[Kernel Fusion]] · [[FlashAttention]] · [[Unsloth]] · [[Vectorization]] · [[NumPy]] · [[PyTorch]] · [[Softmax]] · [[Matrix]] · [[Mixed Precision Training]] · [[Quantization]] · [[Gradient Checkpointing]] · [[Python]]

## Sources
- Triton GitHub README (https://github.com/triton-lang/triton), read on 16 November 2026: the description, the supported hardware and the paper citation (*Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*, MAPL 2019). The README does not name authors or a licence in the text I read. The authors Tillet, Kung and Cox come from the paper link's file name and are not checked.
- Triton tutorial, *Vector Addition* (https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html), read on 16 November 2026: the kernel's structure, the grid, the test size 98,432 and the bandwidth result. I read only this page.
- OpenAI, *Introducing Triton: Open-source GPU programming for neural networks*, 28 July 2021 (https://openai.com/index/triton/): **not opened** (HTTP 403). The date, the author and the claims about CUDA-free programming and cuBLAS-level FP16 matrix multiplication come from search results that quote it, and "fewer than 25 lines" from one secondary article.
- Daniel (Unsloth), Hugging Face blog, 10 January 2024, for the statement that Unsloth rewrites PyTorch modules as Triton kernels (see [[Unsloth]]).
- Image: Ilze Lucero, *Jerarquia de fils d'execució CUDA.png*, Wikimedia Commons, public domain (https://commons.wikimedia.org/wiki/File:Jerarquia_de_fils_d%27execuci%C3%B3_CUDA.png). I viewed it before captioning it. It is saved as `99-Attachments/cuda-thread-hierarchy.png` and is only 433 by 222 pixels.
- The assumed bandwidth and peak speed in the arithmetic are round numbers, not any GPU's spec. The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving Triton:::A programming language and compiler, used from Python, for writing fast custom GPU programs (kernels). You write the program for one block of data and the compiler handles the low-level GPU optimisation.

#flashcards/ml/example #flashcards/topic/llm-serving What does the Triton README (read 16 November 2026) say about itself and its hardware?::"a language and compiler for writing highly efficient custom Deep-Learning primitives". It lists Linux, NVIDIA GPUs (Compute Capability 8.0 and up) and AMD GPUs (ROCm 6.2 and up), with CPUs "under development". Windows is not mentioned there.

#flashcards/ml/concept #flashcards/topic/llm-serving Where does Triton come from?::The paper *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations* (MAPL 2019). OpenAI's announcement, dated 28 July 2021, is by Philippe Tillet, who began the project as a Harvard graduate student (from search results; the OpenAI page could not be opened).

#flashcards/ml/concept #flashcards/topic/llm-serving What does the author claim Triton can do? (search results quoting the OpenAI post)::Let researchers with no CUDA experience write GPU code often close to an expert's, and write FP16 matrix multiplication kernels matching cuBLAS. A "fewer than 25 lines" detail appeared in one secondary source only and is unverified.

#flashcards/ml/concept #flashcards/topic/llm-serving Which tool uses Triton kernels?::Unsloth, whose blog says it rewrites PyTorch modules into Triton kernels.

#flashcards/ml/reverse #flashcards/topic/llm-serving Program, block and grid in Triton:::A program is one instance of the kernel with its own `program_id`, handling one block of data (for example 1,024 values). The grid is the number of programs launched, `triton.cdiv(n, BLOCK_SIZE)`.

#flashcards/ml/concept #flashcards/topic/llm-serving In the tutorial kernel, what do `tl.program_id`, `tl.arange`, `mask`, `tl.load` and `tl.store` do?::`tl.program_id(axis=0)` gives this program's block number. `block_start + tl.arange(0, BLOCK_SIZE)` gives the element indices it owns. `mask = offsets < n_elements` guards the end of the vector. `tl.load` and `tl.store` read and write memory using that mask. `BLOCK_SIZE` is a `tl.constexpr`, known at compile time.

#flashcards/ml/concept #flashcards/topic/llm-serving Why is a mask needed?::The last program may run past the end of the data. The mask turns the extra lanes off, so they do not read or write memory outside the vector.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the grid size and wasted lanes of a Triton launch and explain each symbol.
?
$$G=\left\lceil\frac{N}{B}\right\rceil,\qquad \text{wasted lanes}=G\,B-N$$
- $N$: the number of elements, which fixes the real work.
- $B$: the block size (`BLOCK_SIZE`), elements per program. A larger $B$ means fewer programs but more wasted lanes at the edge, and it must be known at compile time.
- $G$: the grid size, the number of programs launched. It should be big enough to keep every part of the GPU busy.
- $GB-N$: the lanes past the end. The mask turns them off, so they cost almost nothing.

#flashcards/ml/example #flashcards/topic/llm-serving The tutorial's size is $N=98{,}432$ with $B=1024$. What are $G$, the wasted lanes and the last program's active lanes?::$98432/1024=96.125$, so $G=97$. $97\times1024=99{,}328$ lanes, so $896$ are wasted. The first 96 programs are full ($98{,}304$), so the last has $98{,}432-98{,}304=128$ active lanes. The NumPy model gave 97, 896 and 128, and its output matched `x + y` exactly.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the memory traffic of a chain of elementwise operations unfused and fused, and explain each symbol.
?
$$\text{bytes}_{\text{unfused}}=2kNb,\qquad \text{bytes}_{\text{fused}}=2Nb,\qquad \frac{\text{unfused}}{\text{fused}}=k$$
- $k$: the number of operations in the chain. The saving grows with $k$, so long chains of cheap operations gain most.
- $N$, $b$: the number of values and bytes per value. They set the size of each read and write.
- The factor 2: one read and one write per pass over the data. In the fused kernel intermediate results stay in fast on-chip storage.
- When a kernel is memory-bound, time is about bytes divided by bandwidth, so fewer bytes means a proportionally shorter time. If it were arithmetic-bound, fusion would help less.

#flashcards/ml/example #flashcards/topic/llm-serving $N=10^6$ float32 values and $k=3$ operations. What are the bytes and the time at an assumed 1000 GB/s?::Unfused $2\times3\times10^6\times4=24{,}000{,}000$ bytes, fused $8{,}000{,}000$ bytes. At the assumed 1000 GB/s: 24 microseconds against 8, a factor of 3. The bandwidth is the note's assumption, not a measurement.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the arithmetic intensity of vector add and explain each symbol.
?
$$I=\frac{1\ \text{FLOP}}{3\times4\ \text{bytes}}=\frac1{12}\ \text{FLOP/byte}$$
- 1 FLOP: the one addition per element, almost no arithmetic.
- $3\times4$ bytes: two reads and one write of a 4-byte float, so data movement dominates.
- $I$: the intensity. A small $I$ puts the kernel on the sloped, memory-limited part of the roofline $P=\min(P_{\text{peak}},B\cdot I)$, where only moving fewer bytes helps.

#flashcards/ml/example #flashcards/topic/llm-serving With an assumed $B=1000$ GB/s and $P_{\text{peak}}=20$ TFLOP/s, what speed does vector add reach?::$B\cdot I=10^{12}/12=83.3$ GFLOP/s, and $\min(20{,}000,83.3)=83.3$ GFLOP/s, which is 0.42% of the assumed peak. The GPU waits for memory almost all the time, which is why the tutorial reports bandwidth in GB/s instead of FLOP/s.

#flashcards/ml/image #flashcards/topic/llm-serving ![[cuda-thread-hierarchy.png]]
What does this picture show?
?
A CUDA launch written `printHelloGpu<<<2, 5>>>()`: the green rectangle is the grid, it holds 2 grey blocks, and each block holds 5 blue threads. In CUDA you think in threads. In a Triton kernel you write the program for one block and the threads inside it are handled by the compiler (the note's reading of the tutorial, which has a program ID but no thread index). The Commons description (in Catalan) says the numbers of threads and blocks are set in the call. (From the note's caption: Ilze Lucero, Wikimedia Commons, public domain. The picture is only 433 by 222 pixels.)

#flashcards/ml/concept #flashcards/topic/llm-serving What did the tutorial's benchmark show, and what is its limit?::Triton and Torch gave identical bandwidth at every size (peaking at about 1684 GB/s for 134,217,728 elements) and a maximum output difference of 0.0. The page did not name the GPU in what the note read, and vector add is a simple kernel where PyTorch is already fast, so it shows correctness more than a speed-up.

#flashcards/ml/concept #flashcards/topic/llm-serving What was not run for the Triton note?::No Triton kernel ran. Triton is not installed here and there is no supported GPU. The tutorial kernel is rebuilt from a description of its lines. The NumPy model only mimics the block logic, and the bandwidth and peak numbers are assumed round values.

#flashcards/ml/cloze #flashcards/topic/llm-serving A Triton kernel handles one ==block== of data per program, and the ==mask== guards the end of the vector.
