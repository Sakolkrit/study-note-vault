---
tags: [GPU, CUDA, NVIDIA, kernels, programming, parallel, performance, LLM]
status: growing
level: 4
tier: 2
---

# CUDA

**CUDA** is NVIDIA's platform and programming model for running general computing, not just graphics, on its graphics cards ([[GPU]]). You write a small function called a kernel and the GPU runs it on thousands of threads at once.

## 🟢 1. Simple (age 12)
A normal computer chip is like one very clever chef who cooks one dish at a time. A graphics card is like a kitchen with thousands of helpers who each only know a simple job. CUDA is the way you hand all of those helpers the same recipe at once, and tell each one which part of the pile of food is theirs.

**Example:** You want to add 1,000 pairs of numbers. You do not ask one chef to do 1,000 additions. You write one rule, "add the pair that matches your own number", and ask 1,024 helpers to follow it. The 24 helpers with no pair left just stand still.

## 🔵 2. How it works
1. **Two sides.** The CPU and its memory are the **host**. The graphics card and its memory are the **device**. A program always starts on the CPU ([[GPU]]).
2. **Copy the data over.** The numbers go from main memory to the GPU's memory (step 1 in the picture below).
3. **The CPU launches a kernel.** A kernel is a function that runs on the GPU. The launch says how many **blocks** of **threads** to start, written `kernel<<<blocks, threads>>>(...)`.
4. **Every thread runs the same code on its own piece.** Each thread works out its own number from its block number and its position in the block, then handles the data at that spot.
5. **Threads come in groups.** Threads are grouped into blocks, and blocks form a **grid**. Inside a block, threads run in groups of 32 called **warps**.
6. **Copy the result back** to main memory (step 4 in the picture).
7. **Why it suits AI.** [[Neural Network]] maths is mostly the same simple step on huge piles of numbers, such as [[Matrix]] multiplication, so a GPU does it much faster than a CPU ([[Vectorization]]).
8. **The catch:** CUDA only runs on NVIDIA GPUs, the memory copies between host and device cost time, and a kernel is fast only if its threads do similar work and use memory well. I could not run any CUDA here: this PC has no NVIDIA GPU (see the end of level 4).

![[cuda-processing-flow.png]]
*The processing flow on CUDA, as drawn for a GeForce 8800. (1) Copy the data from main memory to GPU memory. (2) The CPU instructs the processing. (3) The GPU executes in parallel in each core. (4) Copy the result back to main memory. Image: Tosaka, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:CUDA_processing_flow_(En).PNG), CC BY 3.0 (attribution required).*

![[cuda-thread-hierarchy.png]]
*A launch written `printHelloGpu<<<2, 5>>>()`: the green rectangle is the grid, it holds 2 grey blocks, and each block holds 5 blue threads. The Commons description (in Catalan) says the numbers of threads and blocks are set in the call. Image: Ilze Lucero, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Jerarquia_de_fils_d%27execuci%C3%B3_CUDA.png), public domain. It is only 433 by 222 pixels.*

## 🟠 3. Technical detail
- **What it is.** NVIDIA's guide defines CUDA as "a parallel computing platform and programming model developed by NVIDIA". Wikipedia calls it proprietary, says it was created in 2004 and officially released in 2007 (the first SDK on 15 February 2007), and credits Ian Buck and John Nickolls as the original authors. It supports CUDA C/C++ (compiled with `nvcc`), Fortran, and Python through wrappers such as CuPy and PyCUDA (Wikipedia).
- **Hierarchy.** Threads are grouped into thread blocks, which form a grid. Blocks and grids can be 1, 2 or 3 dimensional. All threads of a block run on a single streaming multiprocessor (SM). Different blocks "may be executed in any order", so a block cannot rely on another block's results (NVIDIA programming guide).
- **Built-in variables.** `threadIdx` is a thread's index in its block, `blockIdx` the block's index in the grid, `blockDim` and `gridDim` the sizes set at launch (guide).
- **A kernel.** Declared with `__global__`, returns `void`, launched as `name<<<grid, block, sharedMemoryBytes>>>(...)` where the third argument is optional (guide). The guide's vector add uses `int workIndex = threadIdx.x + blockIdx.x*blockDim.x;`, a bounds check `if(workIndex < vectorLength)` and `C[workIndex] = A[workIndex] + B[workIndex];`.
- **Warps.** "Within a thread block, threads are organized into groups of 32 threads called warps." Block sizes that are a multiple of 32 are recommended, because a partial last warp leaves lanes unused (guide). The threads follow SIMT: each can keep its own state and take its own path, but performance is better when threads in the same warp do not diverge (guide).
- **Memory.** Registers are per thread on the SM. Shared memory is on-chip, per SM, and shared by the threads of a block. Global memory is the GPU's DRAM, visible to all SMs. Each SM has an L1 cache and a constant cache, and one larger L2 cache is shared by all SMs (guide). The guide gives no sizes for these on the page I read.
- **Limits are per device.** The guide's example table for compute capability 10.0 lists `maxThreadsPerBlock` 1024, `maxThreadsPerMultiProcessor` 2048, `maxBlocksPerMultiProcessor` 32, `regsPerBlock` 65536 and `sharedMemPerBlock` 49152 bytes. The page says these are an example device, not general limits, so check your GPU's properties.
- **Related tools.** [[Triton]] lets you write a kernel one block at a time from Python and hides the threads, and [[Kernel Fusion]] is the main reason people write custom kernels. [[PyTorch]], [[Unsloth]] and [[FlashAttention]] all run on CUDA kernels.

The next snippet is **not CUDA**. It is my own Python model of a CUDA launch: it calls the kernel function once for every (block, thread) pair, with the guide's index formula. It needs only NumPy and I ran it.

```python
import math
import numpy as np

WARP = 32                                          # threads per warp (NVIDIA CUDA programming guide)

def launch(kernel, grid, block, *args):
    """Run `kernel` once for every (block, thread) pair, like kernel<<<grid, block>>>(...)."""
    for blockIdx in range(grid):
        for threadIdx in range(block):
            kernel(blockIdx, threadIdx, block, grid, *args)

def vec_add(blockIdx, threadIdx, blockDim, gridDim, A, B, C, n):
    i = threadIdx + blockIdx * blockDim            # the guide's workIndex
    if i < n:                                      # bounds check
        C[i] = A[i] + B[i]

n, block = 1000, 256
A = np.random.RandomState(0).rand(n).astype("float32")
B = np.random.RandomState(1).rand(n).astype("float32")
C = np.zeros(n, dtype="float32")
grid = math.ceil(n / block)                        # (n + block - 1) // block
launch(vec_add, grid, block, A, B, C, n)
print("grid:", grid, " threads launched:", grid * block, " idle threads:", grid * block - n, " correct:", np.array_equal(C, A + B))

for b in (256, 100, 96, 33):
    w = math.ceil(b / WARP)
    print("block", b, "-> warps", w, " lanes unused", w * WARP - b, " (%.1f%% of the warp slots idle)" % (100 * (w * WARP - b) / (w * WARP)))

def grid_stride(blockIdx, threadIdx, blockDim, gridDim, A, B, C, n):
    i = threadIdx + blockIdx * blockDim            # first element for this thread
    stride = blockDim * gridDim                    # total number of threads
    while i < n:
        C[i] = A[i] + B[i]
        i += stride

C2 = np.zeros(n, dtype="float32")
g2, b2 = 2, 128                                    # only 256 threads for 1000 elements
launch(grid_stride, g2, b2, A, B, C2, n)
print("grid-stride: threads", g2 * b2, " passes per thread (max):", math.ceil(n / (g2 * b2)), " correct:", np.array_equal(C2, A + B))
```

It printed: `grid: 4  threads launched: 1024  idle threads: 24  correct: True`. Then for block sizes 256, 100, 96 and 33: 8 warps with 0 lanes unused, 4 warps with 28 unused (21.9% idle), 3 warps with 0 unused, and 2 warps with 31 unused (48.4% idle). Last: `grid-stride: threads 256  passes per thread (max): 4  correct: True`.

The real CUDA kernel for the same job is the guide's vector add. The file below is the shape of it, **not compiled and not run** (see the end of level 4).

```cuda
// NOT COMPILED, NOT RUN. Shape of the guide's vector add.
__global__ void vecAdd(const float *A, const float *B, float *C, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;   // the guide's workIndex
    if (i < n) C[i] = A[i] + B[i];                   // bounds check
}
// host side, after copying A and B to the device:
//   int block = 256, grid = (n + block - 1) / block;
//   vecAdd<<<grid, block>>>(dA, dB, dC, n);
```

## 🔴 4. Deep / formal
**Which element a thread owns, and how many blocks to launch.** For $N$ elements and $B$ threads per block, thread $t$ of block $b$ takes the element

$$
i=t+b\,B,\qquad G=\left\lceil\frac{N}{B}\right\rceil,\qquad \text{idle threads}=G\,B-N
$$

**Where:**
- $t$ is `threadIdx.x`, the thread's index in its block. **Matters:** it separates the threads inside one block.
- $b$ is `blockIdx.x`, the block's index in the grid. **Matters:** it separates the blocks, so every thread gets a different $i$.
- $B$ is `blockDim.x`, the threads per block. **Matters:** it must be set at launch, it is limited by the device, and it also sets how many warps a block has.
- $G$ is `gridDim.x`, the number of blocks. **Matters:** it has to be big enough that $GB\ge N$, or some elements get no thread.
- $GB-N$ is the threads past the end. **Matters:** the `if (i < n)` check makes them do nothing, and without it they would read and write outside the arrays.

**Example (tiny numbers):** $N=1000$ and $B=256$.
- $G=\lceil1000/256\rceil=\lceil3.9\rceil=4$ blocks.
- $4\times256=1024$ threads, so $1024-1000=24$ are idle.
- Thread 3 of block 2 takes $i=3+2\times256=515$.
- Meaning: four blocks cover the vector and 24 threads stay idle. The Python model matched `A + B` exactly.

**How many warps a block really occupies.** The hardware runs threads in groups of 32, so a block of $B$ threads uses

$$
W=\left\lceil\frac{B}{32}\right\rceil\ \text{warps},\qquad \text{unused lanes}=32\,W-B
$$

**Where:**
- 32 is the warp size. **Matters:** it is a fixed property of NVIDIA's hardware as the guide describes it, and every launch is rounded up to whole warps.
- $B$ is the block size. **Matters:** a multiple of 32 wastes nothing, which is why the guide recommends it.
- $W$ is the number of warps. **Matters:** warps are what the SM actually schedules.
- $32W-B$ is the lanes of the last warp that hold no thread. **Matters:** those slots are reserved but do no useful work.

**Example (tiny numbers):**
- $B=256$: $W=8$, unused $0$.
- $B=100$: $W=\lceil3.125\rceil=4$, unused $128-100=28$, so $28/128=21.9\%$ of the warp slots idle.
- $B=33$: $W=2$, unused $31$, almost half idle ($48.4\%$).
- Meaning: a block of 33 threads pays for 64 lanes. Choose a multiple of 32.

**Fewer threads than elements: the grid-stride loop.** A thread can handle several elements, stepping by the total number of threads:

$$
i_0=t+b\,B,\qquad i_{m+1}=i_m+B\,G,\qquad \text{passes}=\left\lceil\frac{N}{B\,G}\right\rceil
$$

**Where:**
- $i_0$ is the thread's first element, as in the first equation. **Matters:** the threads still start on different elements.
- $B\,G$ is the total number of threads, the stride. **Matters:** adding it moves a thread to the same relative spot in the next "layer" of the data, so no two threads ever meet on the same element.
- passes is how many elements the busiest thread handles. **Matters:** it lets one launch size cover any $N$, so the grid does not have to grow with the data.

**Example (tiny numbers):** $N=1000$ elements with only $G=2$ blocks of $B=128$ threads, so 256 threads.
- The stride is $256$ and the passes are $\lceil1000/256\rceil=4$.
- Thread 0 of block 0 handles elements $0, 256, 512, 768$.
- Meaning: 256 threads cover 1,000 elements in 4 passes. The model's output matched `A + B` exactly.

**Honest limits and open questions**
- **Nothing here ran on a GPU.** This PC has no NVIDIA GPU (it has Intel Iris Xe graphics). `nvcc` 11.7 is installed, but compiling the vector add with `nvcc -ptx` failed with `Cannot find compiler 'cl.exe' in PATH`, because the Windows host compiler is missing. So no CUDA kernel was compiled or run, and every CUDA fact is from the documents named below.
- **The code above is a model.** It shows the index arithmetic, not the speed or the real scheduling. A real GPU runs the threads in parallel in warps, not in a loop.
- **The limits quoted are one example device.** 1024 threads per block and the others are from a compute-capability-10.0 example table. They differ between GPUs.
- **Wikipedia is a secondary source,** and I read only its first 100,000 of about 170,000 characters. The dates and authors are from there.
- **Not covered:** shared-memory programming and bank conflicts, memory coalescing, streams, the CUDA libraries (cuBLAS and cuDNN), clusters on newer GPUs, and the alternatives for other makers' GPUs.

**What came before, and how this improved on it:**
- **Before:** running general computations on a GPU meant disguising them as graphics work and using a graphics programming interface (from memory; I did not check this against a source).
- **Its limit:** that was awkward and needed graphics knowledge, so few people used GPUs for general maths.
- **What changed:** NVIDIA's CUDA, created in 2004 and released in 2007 (Wikipedia, read), let programmers write kernels in C and C++ with threads, blocks and grids, and copy data between host and device.
- **How much better:** I found no source with a speed comparison, and I measured none, so I give no figure. This note shows how a launch is organised, not how fast it is.

**Check it yourself:** run the Python model in level 3 (it needs only NumPy). The CUDA code needs an NVIDIA GPU and a host C++ compiler, which this PC lacks.

## Related
[[GPU]] · [[Triton]] · [[Kernel Fusion]] · [[FlashAttention]] · [[Unsloth]] · [[PyTorch]] · [[NumPy]] · [[Vectorization]] · [[Matrix]] · [[Neural Network]] · [[Mixed Precision Training]] · [[Gradient Checkpointing]] · [[Python]]

## Sources
- NVIDIA, *CUDA Programming Guide*, archived 13.2.1: the pages *Programming Model* and *2.2 Writing CUDA SIMT Kernels* (https://docs.nvidia.com/cuda/archive/13.2.1/cuda-programming-guide/), read on 16 November 2026: the thread hierarchy, the built-in variables, the kernel launch and vector add lines, the warp size of 32, SIMT, the memory spaces and the example device table. The older guide URLs returned HTTP 404.
- NVIDIA's CUDA guide landing page, for the one-sentence definition of CUDA.
- Wikipedia, *CUDA* (https://en.wikipedia.org/wiki/CUDA), read to about the first 100,000 characters on 16 November 2026: the history, authors, languages, hardware and limitations. A secondary source.
- Image 1: Tosaka, *CUDA processing flow (En).PNG*, Wikimedia Commons, CC BY 3.0, attribution required (https://commons.wikimedia.org/wiki/File:CUDA_processing_flow_(En).PNG), saved as `99-Attachments/cuda-processing-flow.png` (600 by 580). I viewed it before captioning it.
- Image 2: Ilze Lucero, *Jerarquia de fils d'execució CUDA.png*, Wikimedia Commons, public domain, saved as `99-Attachments/cuda-thread-hierarchy.png` (433 by 222). I viewed it before captioning it.
- Not checked: any CUDA compilation or run, the "graphics-API" history in the "before" bullet (from memory), and the note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving CUDA:::NVIDIA's parallel computing platform and programming model for running general computing on its GPUs. You write a kernel and the GPU runs it on thousands of threads at once.

#flashcards/ml/concept #flashcards/topic/llm-serving Who made CUDA and when? (Wikipedia, a secondary source)::NVIDIA, created in 2004 and officially released in 2007 (the first SDK on 15 February 2007). Original authors: Ian Buck and John Nickolls. It is proprietary and runs only on NVIDIA GPUs.

#flashcards/ml/concept #flashcards/topic/llm-serving What are host and device in CUDA?::The host is the CPU and its memory. The device is the GPU and its memory. A program always starts on the CPU, and a kernel is started by a launch.

#flashcards/ml/concept #flashcards/topic/llm-serving What are the four steps of the CUDA processing flow?::(1) Copy the data from main memory to GPU memory. (2) The CPU instructs the processing. (3) The GPU executes in parallel in each core. (4) Copy the result back to main memory.

#flashcards/ml/reverse #flashcards/topic/llm-serving Thread, block, grid, warp:::Threads are grouped into thread blocks, which form a grid (1, 2 or 3 dimensional). All threads of a block run on one streaming multiprocessor. Inside a block, threads run in groups of 32 called warps.

#flashcards/ml/concept #flashcards/topic/llm-serving Can thread blocks rely on each other's results?::No. Different blocks "may be executed in any order" (NVIDIA programming guide), so a block cannot rely on another block's results.

#flashcards/ml/concept #flashcards/topic/llm-serving How is a kernel declared and launched?::Declared with `__global__` and returning `void`. Launched as `name<<<grid, block, sharedMemoryBytes>>>(...)`, where the third argument is optional.

#flashcards/ml/concept #flashcards/topic/llm-serving What do `threadIdx`, `blockIdx`, `blockDim` and `gridDim` give?::`threadIdx` is the thread's index in its block, `blockIdx` the block's index in the grid, `blockDim` and `gridDim` the block and grid sizes set at launch.

#flashcards/ml/concept #flashcards/topic/llm-serving Describe CUDA's memory spaces.::Registers are per thread on the SM. Shared memory is on-chip, per SM, shared by the threads of a block. Global memory is the GPU's DRAM, visible to all SMs. Each SM has an L1 and a constant cache, and one larger L2 cache is shared by all SMs.

#flashcards/ml/concept #flashcards/topic/llm-serving What is SIMT and what is divergence?::Each thread can keep its own state and take its own code path, but performance is better when threads in the same warp do not take divergent paths.

#flashcards/ml/example #flashcards/topic/llm-serving Are the example limits 1024 threads per block and 2048 per SM general?::No. They come from the guide's example table for compute capability 10.0 (also 32 blocks per SM, 65536 registers per block, 49152 bytes of shared memory per block). The page says to check your own GPU's properties.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the global index of a thread, the number of blocks and the idle threads, and explain each symbol.
?
$$i=t+b\,B,\qquad G=\left\lceil\frac{N}{B}\right\rceil,\qquad \text{idle threads}=G\,B-N$$
- $t$: `threadIdx.x`, the thread's index in its block, which separates threads inside one block.
- $b$: `blockIdx.x`, the block's index in the grid, which separates the blocks so every thread gets a different $i$.
- $B$: `blockDim.x`, threads per block. It is set at launch, limited by the device, and sets how many warps a block has.
- $G$: `gridDim.x`, the number of blocks. It must be big enough that $GB\ge N$, or some elements get no thread.
- $GB-N$: the threads past the end. The `if (i < n)` check makes them do nothing. Without it they would read and write outside the arrays.

#flashcards/ml/example #flashcards/topic/llm-serving $N=1000$ and $B=256$. What are $G$, the idle threads, and the element of thread 3 in block 2?::$G=\lceil1000/256\rceil=4$ blocks, $4\times256=1024$ threads, so 24 idle. Thread 3 of block 2 takes $i=3+2\times256=515$. The Python model of the launch matched `A + B` exactly.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the warps a block occupies and the unused lanes, and explain each symbol.
?
$$W=\left\lceil\frac{B}{32}\right\rceil\ \text{warps},\qquad \text{unused lanes}=32\,W-B$$
- 32: the warp size, fixed by the hardware as the guide describes it. Every launch is rounded up to whole warps.
- $B$: the block size. A multiple of 32 wastes nothing, which is why the guide recommends it.
- $W$: the number of warps, which is what the SM actually schedules.
- $32W-B$: the lanes of the last warp that hold no thread. They are reserved but do no useful work.

#flashcards/ml/example #flashcards/topic/llm-serving What are the warps and idle share for $B=256$, $B=100$ and $B=33$?::$B=256$: 8 warps, 0 unused. $B=100$: 4 warps, $128-100=28$ unused, 21.9% of the warp slots idle. $B=33$: 2 warps, 31 unused, 48.4% idle. A block of 33 threads pays for 64 lanes, so choose a multiple of 32.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the grid-stride loop and its number of passes, and explain each symbol.
?
$$i_0=t+b\,B,\qquad i_{m+1}=i_m+B\,G,\qquad \text{passes}=\left\lceil\frac{N}{B\,G}\right\rceil$$
- $i_0$: the thread's first element, as in the first equation, so threads still start on different elements.
- $B\,G$: the total number of threads, the stride. Adding it moves a thread to the same relative spot in the next layer of the data, so no two threads meet on the same element.
- passes: how many elements the busiest thread handles. One launch size can cover any $N$, so the grid does not have to grow with the data.

#flashcards/ml/example #flashcards/topic/llm-serving $N=1000$ elements with $G=2$ blocks of $B=128$ threads. What are the stride, the passes and thread 0 of block 0's elements?::256 threads, stride 256, passes $\lceil1000/256\rceil=4$. Thread 0 of block 0 handles elements 0, 256, 512 and 768. The model's output matched `A + B`.

#flashcards/ml/image #flashcards/topic/llm-serving ![[cuda-processing-flow.png]]
What does this picture show?
?
The processing flow on CUDA, drawn for a GeForce 8800. (1) Copy the data from main memory to GPU memory. (2) The CPU instructs the processing. (3) The GPU executes in parallel in each core. (4) Copy the result back to main memory. (From the note's caption: Tosaka, Wikimedia Commons, CC BY 3.0, attribution required.)

#flashcards/ml/image #flashcards/topic/llm-serving ![[cuda-thread-hierarchy.png]]
What does this picture show?
?
A launch written `printHelloGpu<<<2, 5>>>()`: the green rectangle is the grid, it holds 2 grey blocks, and each block holds 5 blue threads. The Commons description (in Catalan) says the numbers of threads and blocks are set in the call. (From the note's caption: Ilze Lucero, Wikimedia Commons, public domain. The picture is only 433 by 222 pixels.)

#flashcards/ml/concept #flashcards/topic/llm-serving What was not done for the CUDA note?::Nothing ran on a GPU. This PC has no NVIDIA GPU (it has Intel Iris Xe). `nvcc` 11.7 is installed, but `nvcc -ptx` failed with `Cannot find compiler 'cl.exe' in PATH`, so no kernel was compiled. The Python code only models the index arithmetic. The "graphics-API" history is from memory, and the note reads only the first 100,000 characters of the Wikipedia page.

#flashcards/ml/cloze #flashcards/topic/llm-serving A block size that is a ==multiple of 32== wastes no warp lanes.
