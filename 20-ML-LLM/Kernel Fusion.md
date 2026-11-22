---
tags: [GPU, kernels, performance, memory, compilers, LLM, training]
status: growing
level: 4
tier: 3
---

# Kernel Fusion

**Kernel fusion** means doing several small steps inside one GPU program (one [[CUDA]] or [[Triton]] kernel) instead of running them as separate programs. The numbers between the steps stay in fast on-chip storage instead of being written out to slow memory and read back in.

## 🟢 1. Simple (age 12)
Imagine washing, drying and folding a pile of clothes. One way: wash everything and carry the whole pile to a store room, then fetch the whole pile to dry it, carry it to the store room again, then fetch it to fold. The other way: take each item through all three steps while you are holding it. The work is the same, but you stop carrying the pile back and forth.

**Example:** A chef must chop, season and fry. If every step ends with putting all the food in the fridge and taking it out again, most of the time goes on the fridge door. Doing all three steps at the stove, one portion at a time, saves most of that time.

## 🔵 2. How it works
1. **A GPU is often waiting for memory, not for arithmetic.** Reading and writing the big main memory of the [[GPU]] is slow compared with doing the maths ([[GPU]], roofline).
2. **A simple step, like "multiply by 2", does very little maths per number.** It reads each number, does one tiny calculation, and writes it back.
3. **A chain of simple steps run as separate kernels repeats that.** Each kernel reads the whole array from main memory and writes the whole result back, so the data crosses the slow memory again and again.
4. **A fused kernel does the whole chain on a number while it is held on the chip.** It reads the input once, runs every step, and writes only the final answer.
5. **Less traffic means less time** when the kernel is limited by memory. Each separate kernel also has a small start-up cost, and fusion removes some of those too.
6. **Tools do it for you or let you do it by hand.** A compiler can fuse steps automatically, as PyTorch's compiler does, and a language like [[Triton]] lets you write the fused kernel yourself ([[PyTorch]], [[Unsloth]]).
7. **The same idea speeds up attention.** [[FlashAttention]] avoids writing the big attention table to slow memory.
8. **The catch:** fusion helps only when memory is the bottleneck, a fused kernel is harder to write, and the best fusion depends on the GPU. I could not test any of it on a GPU here.

No picture: I searched Wikimedia Commons and found no free, on-topic image for kernel fusion, so I added none.

## 🟠 3. Technical detail
- **What gets fused.** The easiest case is a chain of elementwise steps, for example `relu(x * w + b)`: multiply, add and ReLU ([[Activation Function]]). Other common targets are the steps of [[Softmax]] and [[Layer Normalization]], and a sequence of optimizer updates ([[Adam Optimizer]]).
- **The reason, in one line.** A kernel's time is limited by the larger of its arithmetic time and its memory time (the roofline equation in [[GPU]]). A chain of simple steps has a low arithmetic intensity, so it is limited by memory, and fusion cuts the bytes moved.
- **In the PyTorch compiler.** The PyTorch 2 introduction says its low-level operations are "low-level enough that you need to fuse them back together to get good performance", that TorchInductor "creates fusion groups", and that for NVIDIA and AMD GPUs it "uses OpenAI Triton as a key building block". It reports "a speedup between 1.5x and 2.x in training Transformers models" with one added line, and 43% faster training on an A100 across 163 open-source models (21% at Float32 and 51% with AMP, measured separately). It says speedups are lower on a desktop GPU such as a 3090 (PyTorch 2 page, an older release; the supported hardware has probably grown since).
- **In attention.** The FlashAttention abstract says it uses tiling to reduce reads and writes between the GPU's high-bandwidth memory and on-chip SRAM, with about a 15% wall-clock speed-up on BERT-large, about 3× on GPT-2 and about 2.4× on Long-Range Arena. The abstract does not use the word "fusion", so I describe it only as a related way to cut memory traffic.
- **In fine-tuning tools.** [[Unsloth]]'s blog says it rewrites PyTorch modules into Triton kernels.
- **Cost of fusing.** A fused kernel keeps more values in registers and on-chip memory at once, which can limit how many threads run together ([[CUDA]]). It is also harder to write, to debug and to change.

The code below is **not a GPU kernel**. It is a CPU analogue in NumPy that I ran: four steps done as four full passes over an 80 MB array (like four kernels), against the same four steps done block by block so each block stays in the CPU cache.

```python
import time
import numpy as np

N = 20_000_000                                   # 80 MB of float32: far bigger than any CPU cache
rng = np.random.RandomState(0)
x = rng.rand(N).astype("float32")
y_un = np.empty_like(x)
y_fu = np.empty_like(x)
a, b, c, d = np.float32(1.5), np.float32(0.25), np.float32(0.5), np.float32(-0.1)

def unfused(x, out):
    """Four separate passes over the whole array, like four separate kernels."""
    np.multiply(x, a, out=out)
    np.add(out, b, out=out)
    np.multiply(out, c, out=out)
    np.add(out, d, out=out)

def fused(x, out, block=1 << 15):
    """The same four steps, done block by block so each block stays in cache."""
    for s in range(0, len(x), block):
        o = out[s:s + block]
        np.multiply(x[s:s + block], a, out=o)
        np.add(o, b, out=o)
        np.multiply(o, c, out=o)
        np.add(o, d, out=o)

def best(f, *args, reps=7):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        f(*args)
        ts.append(time.perf_counter() - t)
    return min(ts), sorted(ts)[len(ts) // 2]

unfused(x, y_un)
fused(x, y_fu)
print("same result:", np.array_equal(y_un, y_fu))
tu, mu = best(unfused, x, y_un)
tf, mf = best(fused, x, y_fu)
print("unfused: best %.1f ms, median %.1f ms" % (tu * 1e3, mu * 1e3))
print("fused  : best %.1f ms, median %.1f ms" % (tf * 1e3, mf * 1e3))
print("speed-up (best): %.2fx, (median): %.2fx" % (tu / tf, mu / mf))
print("model bytes unfused:", 4 * 2 * N * 4, " fused:", 2 * N * 4, " ratio:", 4)
```

Four runs of it on this PC printed `same result: True` and a speed-up (best of 7) of 2.37x, 2.06x, 1.80x and 2.25x, with medians of 2.27x, 2.25x, 1.97x and 2.95x (for example 47.9 ms against 20.2 ms in the first run). The 7 timings within a run also vary a lot (the unfused median was 56.7 ms against a best of 42.4 ms in the last run). The model's byte ratio is 4 ($640{,}000{,}000$ against $160{,}000{,}000$ bytes). So the speed-up was real and repeatable but about half the traffic ratio. The timings are one machine, four runs, and noisy.

## 🔴 4. Deep / formal
**Memory traffic of a chain.** Let a chain of $k$ elementwise steps work on arrays of $N$ values of $b$ bytes. Each separate kernel $j$ reads $r_j$ arrays and writes $w_j$. A fused kernel reads only the arrays that come from outside the chain, $r_{\text{ext}}$, and writes only the final arrays, $w_{\text{out}}$:

$$
\text{bytes}_{\text{unfused}}=\sum_{j=1}^{k}(r_j+w_j)\,N\,b,\qquad \text{bytes}_{\text{fused}}=(r_{\text{ext}}+w_{\text{out}})\,N\,b
$$

**Where:**
- $r_j, w_j$ are the arrays kernel $j$ reads and writes. **Matters:** every intermediate result is written by one kernel and read by the next, so it is counted twice.
- $r_{\text{ext}}$ is the inputs that come from outside the chain, and $w_{\text{out}}$ the outputs the chain must produce. **Matters:** these are the only transfers fusion cannot avoid.
- $N$ and $b$ are the number of values and the bytes per value. **Matters:** both versions grow with them, so the ratio does not depend on size.
- Intermediates stay in registers or on-chip memory in the fused kernel. **Matters:** that is the whole saving.

**Example (tiny numbers):** $y=\text{relu}(x\cdot w+\text{bias})$ with $x$, $w$ and $\text{bias}$ each $N=10^6$ float32 values ($b=4$).
- Three kernels: multiply reads 2 and writes 1, add reads 2 and writes 1, ReLU reads 1 and writes 1, so $(3+3+2)=8$ array transfers, $8\times4\,\text{MB}=32$ MB.
- Fused: reads $x$, $w$, bias (3) and writes $y$ (1), so $4$ transfers, 16 MB.
- Meaning: half the traffic, 2 times less. The ratio is $8/4=2$. (For a pure one-input chain of $k$ steps it would be $k$ times.)

**Time, with a cost per kernel.** Take the memory time as bytes divided by bandwidth and add a start-up cost $t_\ell$ for each kernel launched:

$$
t\approx\max\!\Big(t_{\text{compute}},\ \frac{\text{bytes}}{\text{BW}}\Big)+n_{\text{kernels}}\,t_{\ell}
$$

**Where:**
- $t_{\text{compute}}$ is the arithmetic time, FLOPs over the GPU's peak speed. **Matters:** if it is the larger term the kernel is compute-bound and fusion saves little.
- bytes/BW is the memory time. **Matters:** this is the term fusion shrinks.
- $n_{\text{kernels}}$ is the number of kernels, $k$ unfused and 1 fused. **Matters:** each extra launch adds a fixed cost, which matters most for small arrays.
- $t_\ell$ is the launch cost. **Matters:** it is a number I assume below, not a measured value.

**Example (tiny numbers):** use the 32 MB against 16 MB above, an **assumed** bandwidth of 1000 GB/s and an **assumed** launch cost of 5 microseconds, with the arithmetic time negligible.
- Unfused: $32\,\text{MB}/1000\,\text{GB/s}=32\ \mu\text{s}$, plus $3\times5=15\ \mu\text{s}$, total $47\ \mu\text{s}$.
- Fused: $16\ \mu\text{s}+1\times5=21\ \mu\text{s}$.
- Meaning: $47/21=2.2$ times faster, slightly more than the traffic ratio 2 because fusion also removes two launches. The bandwidth and launch cost are round numbers, not any GPU's spec.

**Fusion cannot beat the arithmetic.** The speed-up from fusing is limited by whichever bound the fused kernel hits:

$$
S=\frac{\max(T_c,\,T_{m,\text{unfused}})}{\max(T_c,\,T_{m,\text{fused}})}\ \longrightarrow\ 1\ \ \text{as}\ \ T_c\gg T_m
$$

**Where:**
- $T_c$ is the compute time, and $T_{m,\text{unfused}}$, $T_{m,\text{fused}}$ the memory times of the two versions. **Matters:** only the memory times change with fusion.
- $S$ is the speed-up (launch costs ignored here). **Matters:** it is largest when memory dominates and shrinks to 1 when arithmetic dominates, so a heavy kernel such as a big matrix multiplication gains little from fusing the small steps around it.

**Example (tiny numbers):** $T_{m,\text{unfused}}=32$ and $T_{m,\text{fused}}=16$ (microseconds, from above).
- If $T_c=10$: $S=\max(10,32)/\max(10,16)=32/16=2$.
- If $T_c=40$: $S=\max(40,32)/\max(40,16)=40/40=1$.
- Meaning: the same fusion gives 2× when memory-bound and nothing when compute-bound.

**Honest limits and open questions**
- **No GPU result.** I measured nothing on a GPU. The traffic and time arithmetic is a model with an assumed bandwidth and launch cost, and the only measurement is the CPU analogue.
- **The CPU analogue is not kernel fusion.** It is cache blocking in NumPy. It shows the same idea (do all the steps while the data is close), and it gave 1.8x to 2.4x (best of 7, over four runs) where the traffic model predicts 4. That gap probably comes from the arithmetic and the per-block call overhead, which I did not separate.
- **The PyTorch figures are the PyTorch team's, for an older release,** on A100 and 163 models, and the page is truncated in what I read. They are not my measurements, and the supported GPUs have likely changed since.
- **FlashAttention's abstract** does not mention kernel fusion by name. Its speed-ups come from tiling that cuts memory reads and writes. I call it a related idea, not an example I verified.
- **Not covered:** fusing matrix multiplications with their neighbours, fusing in the backward pass, the cost of recomputing instead of storing ([[Gradient Checkpointing]] is a related trade), fusion across devices, and automatic fusion rules in detail.
- **When fusion backfires:** a fused kernel that needs too many registers can lower the number of threads that run together, so it can be slower than the separate kernels.

**What came before, and how this improved on it:**
- **Before:** running each operation of a model as its own GPU kernel, as an eager framework does, with every intermediate result written to main memory ([[PyTorch]]).
- **Its limit:** for simple operations the GPU spends its time moving data to and from memory, and every kernel adds a launch cost.
- **What changed:** combining chains of operations into single kernels, by compilers such as PyTorch's TorchInductor or by hand-written kernels in [[Triton]] or [[CUDA]].
- **How much better:** published figures, not mine. The PyTorch 2 page reports 43% faster training on an A100 across 163 models (21% at Float32, 51% with AMP) and "between 1.5x and 2.x" for Transformers, using a compiler that fuses (not fusion alone). In my own CPU analogue, four steps on 20 million floats ran 1.8x to 2.4x faster (best of 7, four runs) when done block by block, against a traffic ratio of 4. In the arithmetic model, fusing 3 steps halves the traffic of the example (32 MB to 16 MB).

**Check it yourself:** run the code in level 3. It needs only NumPy. The GPU claims need a GPU, which this PC lacks.

## Related
[[GPU]] · [[CUDA]] · [[Triton]] · [[FlashAttention]] · [[Unsloth]] · [[PyTorch]] · [[Vectorization]] · [[Softmax]] · [[Layer Normalization]] · [[Activation Function]] · [[Adam Optimizer]] · [[Mixed Precision Training]] · [[Gradient Checkpointing]] · [[Quantization]] · [[ONNX]]
## Sources
- PyTorch, *Introduction to PyTorch 2* (https://pytorch.org/get-started/pytorch-2-x/), read on 16 November 2026, but only the first 100,000 of about 137,000 characters: the quotes on fusion, TorchInductor and Triton, and the speed-up numbers with their conditions. It is an older release page.
- Dao, Fu, Ermon, Rudra and Ré, *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, arXiv 2205.14135 (27 May 2022): abstract only, for the tiling and the speed-ups.
- The Unsloth Hugging Face blog (10 January 2024) and the Triton tutorial pages, as in the [[Unsloth]] and [[Triton]] notes.
- The NVIDIA CUDA programming guide for the memory spaces and warps, as in [[CUDA]].
- My own arithmetic: the bandwidth (1000 GB/s) and the launch cost (5 microseconds) are assumed round numbers, not specs. The CPU timings are four runs on this PC.
- No image: no free, on-topic picture for this note was found on Wikimedia Commons. The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving Kernel fusion:::Doing several small steps inside one GPU kernel instead of running them as separate kernels. The numbers between the steps stay in fast on-chip storage instead of being written to slow main memory and read back.

#flashcards/ml/concept #flashcards/topic/llm-serving Why does a chain of simple GPU steps waste time?::Each simple step does very little maths per number, so the GPU is limited by memory. Run as separate kernels, every step reads the whole array from main memory and writes the whole result back, and each kernel adds a start-up cost.

#flashcards/ml/concept #flashcards/topic/llm-serving What does a fused kernel do instead?::It reads the input once, runs every step on a number while it is held on the chip, and writes only the final answer. It saves memory traffic and some launch costs, but only when the kernel is memory-bound.

#flashcards/ml/concept #flashcards/topic/llm-serving What kinds of steps are typically fused?::Chains of elementwise steps (multiply, add, ReLU), the steps of softmax and layer normalization, and sequences of optimizer updates.

#flashcards/ml/example #flashcards/topic/llm-serving What do the PyTorch 2 page (an older release) and the PyTorch team say about fusion?::The low-level operations are "low-level enough that you need to fuse them back together to get good performance". TorchInductor "creates fusion groups" and for NVIDIA and AMD GPUs "uses OpenAI Triton as a key building block". They report "a speedup between 1.5x and 2.x" for training Transformers, and 43% faster training on an A100 across 163 models (21% at Float32 and 51% with AMP). Speed-ups are lower on a 3090. These are the PyTorch team's figures, not the note's.

#flashcards/ml/concept #flashcards/topic/llm-serving What does the FlashAttention abstract say, and does it say "fusion"?::It uses tiling to reduce reads and writes between GPU high-bandwidth memory and on-chip SRAM, with about a 15% wall-clock speed-up on BERT-large, about 3x on GPT-2 and about 2.4x on Long-Range Arena. The abstract does not use the word "fusion", so the note calls it only a related way to cut memory traffic.

#flashcards/ml/concept #flashcards/topic/llm-serving When can fusion backfire?::A fused kernel that needs too many registers or too much on-chip memory can lower the number of threads running together, so it can be slower than separate kernels. It is also harder to write, debug and change.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the memory traffic of a chain unfused and fused, and explain each symbol.
?
$$\text{bytes}_{\text{unfused}}=\sum_{j=1}^{k}(r_j+w_j)\,N\,b,\qquad \text{bytes}_{\text{fused}}=(r_{\text{ext}}+w_{\text{out}})\,N\,b$$
- $r_j$, $w_j$: the arrays kernel $j$ reads and writes. Every intermediate result is written by one kernel and read by the next, so it is counted twice.
- $r_{\text{ext}}$: the inputs from outside the chain. $w_{\text{out}}$: the outputs the chain must produce. These are the only transfers fusion cannot avoid.
- $N$, $b$: the number of values and bytes per value. Both versions grow with them, so the ratio does not depend on size.
- In the fused kernel the intermediates stay in registers or on-chip memory. That is the whole saving.

#flashcards/ml/example #flashcards/topic/llm-serving $y=\text{relu}(x\cdot w+\text{bias})$ with three arrays of $10^6$ float32 values. What are the bytes unfused and fused?::Three kernels: multiply reads 2 and writes 1, add reads 2 and writes 1, ReLU reads 1 and writes 1, so 8 array transfers of 4 MB each, 32 MB. Fused: reads $x$, $w$, bias and writes $y$, 4 transfers, 16 MB. Half the traffic. For a pure one-input chain of $k$ steps the ratio would be $k$.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the kernel time with a launch cost and explain each symbol.
?
$$t\approx\max\!\Big(t_{\text{compute}},\ \frac{\text{bytes}}{\text{BW}}\Big)+n_{\text{kernels}}\,t_{\ell}$$
- $t_{\text{compute}}$: the arithmetic time, FLOPs over peak speed. If it is the larger term the kernel is compute-bound and fusion saves little.
- bytes/BW: the memory time, the term fusion shrinks.
- $n_{\text{kernels}}$: the number of kernels, $k$ unfused and 1 fused. Each extra launch adds a fixed cost, which matters most for small arrays.
- $t_\ell$: the launch cost, an assumed number in the note, not a measured value.

#flashcards/ml/example #flashcards/topic/llm-serving With an assumed 1000 GB/s and an assumed 5 microsecond launch cost, and negligible arithmetic, what are the unfused and fused times for 32 MB against 16 MB?::Unfused: 32 microseconds plus $3\times5=15$, total 47. Fused: 16 plus 5, total 21. That is $47/21=2.2$ times faster, slightly more than the traffic ratio 2 because two launches are removed. The bandwidth and launch cost are round numbers, not any GPU's spec.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the speed-up limit of fusion and explain each symbol.
?
$$S=\frac{\max(T_c,\,T_{m,\text{unfused}})}{\max(T_c,\,T_{m,\text{fused}})}\ \longrightarrow\ 1\ \ \text{as}\ \ T_c\gg T_m$$
- $T_c$: the compute time. $T_{m,\text{unfused}}$, $T_{m,\text{fused}}$: the memory times of the two versions. Only the memory times change with fusion.
- $S$: the speed-up (launch costs ignored). It is largest when memory dominates and shrinks to 1 when arithmetic dominates, so a heavy kernel such as a big matrix multiplication gains little from fusing the small steps around it.

#flashcards/ml/example #flashcards/topic/llm-serving $T_{m,\text{unfused}}=32$ and $T_{m,\text{fused}}=16$ microseconds. What is $S$ for $T_c=10$ and $T_c=40$?::$T_c=10$: $\max(10,32)/\max(10,16)=32/16=2$. $T_c=40$: $\max(40,32)/\max(40,16)=40/40=1$. The same fusion gives 2x when memory-bound and nothing when compute-bound.

#flashcards/ml/concept #flashcards/topic/llm-serving What did the CPU analogue measure?::Four steps on 20 million float32 values (80 MB): four full passes against the same four steps done block by block so each block stays in cache. Best-of-7 speed-ups over four runs were 2.37x, 2.06x, 1.80x and 2.25x (medians 2.27x, 2.25x, 1.97x and 2.95x), against a traffic ratio of 4 in the model. The timings are one machine, four runs, and noisy.

#flashcards/ml/concept #flashcards/topic/llm-serving Why is the CPU analogue not real kernel fusion, and why is the speed-up below the traffic ratio?::It is cache blocking in NumPy, which shows the same idea (do all the steps while the data is close). The gap to the model's 4 probably comes from the arithmetic and the per-block call overhead, which the note did not separate.

#flashcards/ml/concept #flashcards/topic/llm-serving What did the note not measure?::Anything on a GPU. The traffic and time arithmetic uses an assumed bandwidth and launch cost, and the only measurement is the CPU analogue. Fusing matrix multiplications, the backward pass and automatic fusion rules are not covered.

#flashcards/ml/cloze #flashcards/topic/llm-serving Fusion helps only when a kernel is ==memory-bound==. When it is compute-bound the speed-up goes to 1.
