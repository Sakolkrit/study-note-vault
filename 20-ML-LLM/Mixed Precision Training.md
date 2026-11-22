---
tags: [training, GPU, mixed-precision, float16, bfloat16, efficiency, LLM]
status: growing
level: 4
tier: 3
---

# Mixed Precision Training

Training a model with a mix of **short (16-bit) and long (32-bit) numbers**, to go faster and save memory without losing accuracy.

## 🟢 1. Simple (age 12)
Computers store numbers with a fixed amount of space. Long numbers are very exact but take up more room and are slower to work with. Short numbers are quick and compact but a bit blurry. **Mixed precision** uses short numbers for most of the work and keeps **one careful, exact copy** of the important numbers so that tiny improvements are not lost.

**Example:** You are saving up pocket money, adding 1 cent a day. If your notebook only has room for whole dollars, every day's cent rounds away and your savings never grow. So you keep a **precise ledger** for the cents and use the whole-dollar notebook for quick checks.

## 🔵 2. How it works
1. A computer stores a decimal number as a **sign**, an **exponent** (how big) and a **mantissa** (the digits). Different formats give these different amounts of room: FP32 (32 bits), FP16 (16 bits) and BF16 (16 bits, a different split).
2. A GPU does arithmetic **faster on fewer bits** ([[GPU]]), and numbers take less memory.
3. In training, the **forward pass** (compute the answer and the loss) can use 16-bit activations.
4. The **backward pass** (compute gradients, see [[Backpropagation]]) can also use 16 bits.
5. But the **weight update** keeps a 32-bit copy of the weights (the "master copy"), because the updates are tiny and a 16-bit number would round them away.
6. Another trap: very small gradients become exactly 0 in FP16. The fix is **loss scaling**: multiply the loss by a big number before the backward pass so gradients stay representable, and divide it back before the update.
7. BF16 has the same range as FP32 but fewer digits, so it usually does not need loss scaling (a common practice; I checked the range in the code, not the practice).
8. The lecture's summary: the aim is to speed up training and decrease memory needs; activations and gradients in low precision, weights updated in high precision.

## 🟠 3. Technical detail
- **Formats (code output):**
  - FP32: largest about $3.4\times10^{38}$, smallest normal number about $1.2\times10^{-38}$, 23 mantissa bits.
  - FP16: largest only $65{,}504$, smallest normal about $6.1\times10^{-5}$, 10 mantissa bits.
  - BF16: largest about $3.4\times10^{38}$ (same range as FP32), 7 mantissa bits.
- **Why FP16 needs care:** the small range means very small gradients underflow to 0 and very large values overflow. Loss scaling handles the first.
- **The recipe of the paper:** keep a single-precision copy of the weights that accumulates the gradient after each step, round it to half precision for the pass, and scale the loss (arXiv 1710.03740 abstract, read).
- **What it saves, honestly:** the paper's abstract says memory falls by nearly 2 times. The code shows that the *parameter state* does **not** shrink with Adam: $2+2+4+4+4=16$ bytes per parameter against $4+4+4+4=16$ in full 32-bit. The saving therefore comes from other things, mainly the activations stored for the backward pass (my reading of why, which I did not measure) and from faster 16-bit arithmetic on hardware that supports it.
- **Related memory tricks:** sharing the optimizer state across devices ([[Data Parallelism]]), recomputing instead of storing activations, and [[FlashAttention]].
- **Inference is different:** [[Quantization]] goes much lower (8 or 4 bits) for running a trained model.

```python
import numpy as np
import torch
# 1) the range and precision of 16-bit numbers against 32-bit
f16, f32 = np.finfo(np.float16), np.finfo(np.float32)
print('float16: largest', float(f16.max), '| smallest normal', float(f16.tiny), '| bits for precision', f16.nmant, '| bytes', 2)
print('float32: largest', float(f32.max), '| smallest normal', float(f32.tiny), '| bits for precision', f32.nmant, '| bytes', 4)
bf = torch.finfo(torch.bfloat16)
print('bfloat16: largest', float(bf.max), '| smallest normal', float(bf.tiny), '| bits for precision', 7, '| bytes', 2)
# 2) a tiny gradient vanishes in 16 bits, and loss scaling rescues it
g = 1e-8
print('gradient 1e-8 stored as float16:', float(np.float16(g)), '| times 1024 first:', float(np.float16(g * 1024)), '-> divide back:', float(np.float16(g * 1024)) / 1024)
# 3) a small update is lost when added to a 16-bit weight, but kept by a 32-bit master copy
w16 = np.float16(1.0); w32 = np.float32(1.0)
for _ in range(1000): w16 = np.float16(w16 + np.float16(1e-4)); w32 = np.float32(w32 + np.float32(1e-4))
print('after 1000 updates of +0.0001: float16 weight', float(w16), '| float32 master copy', round(float(w32), 4), '| exact', 1.1)
# 4) numbers kept per parameter in mixed-precision training with Adam: 16-bit weights and gradients, 32-bit master weights, momentum, variance
per_param = 2 + 2 + 4 + 4 + 4
print('bytes per parameter:', per_param, '| for 1 billion parameters:', per_param, 'GB | all in float32 with Adam (weights, grads, m, v):', 4 + 4 + 4 + 4, 'GB')
```

## 🔴 4. Deep / formal
**Loss scaling.** Multiply the loss by a scale $s$ before the backward pass, then divide the gradient by $s$ before the update:

$$
g_{\text{stored}}=\text{fp16}(s\cdot g),\qquad g_{\text{used}}=\frac{g_{\text{stored}}}{s}
$$

**Where:**
- $g$ is the true gradient and $s>1$ is the scale (for example 1024).
- $\text{fp16}(\cdot)$ means "rounded to a 16-bit number", which turns anything below about $6\times10^{-8}$ into 0 (and loses precision below $6.1\times10^{-5}$).
- $g_{\text{stored}}$ is what the backward pass keeps in 16 bits and $g_{\text{used}}$ is the gradient actually used to update the 32-bit master weights.
- **Matters:** because the chain rule is linear, scaling the loss scales every gradient by $s$, which lifts tiny values into the range FP16 can hold. A scale too large overflows (above $65{,}504$), so real systems adjust $s$ as training goes (a common practice; not checked here).

**Example (tiny numbers):** a gradient $g=10^{-8}$ and $s=1024$.
- Stored directly in FP16: $\mathbf{0}$ (it underflows).
- Scaled first: $1024\times10^{-8}=1.02\times10^{-5}$, which FP16 can hold, stored as $1.0252\times10^{-5}$.
- Divide back: $1.0252\times10^{-5}/1024=\mathbf{1.0012\times10^{-8}}$.
- Meaning: the gradient survived, with a small rounding error of about 0.1%.

**Why the master copy.** Memory per parameter in mixed-precision training with Adam:

$$
\underbrace{2}_{\text{FP16 weights}}+\underbrace{2}_{\text{FP16 grads}}+\underbrace{4}_{\text{FP32 master}}+\underbrace{4}_{\text{momentum}}+\underbrace{4}_{\text{variance}}=16\text{ bytes}
$$

**Where:**
- The first two terms are the working copies used in the passes.
- The master weights, the momentum and the variance are the 32-bit state the optimizer keeps ([[Adam Optimizer]]).
- **Matters:** almost all of it is the optimizer's 12 bytes. Without the 32-bit master, tiny updates are lost, as the example shows. This 16-byte total is also the figure that [[Data Parallelism]]'s memory sharing starts from.

**Example (tiny numbers):** add $+0.0001$ to a weight of $1.0$, 1,000 times.
- FP16 weight: $1.0+0.0001$ rounds back to $1.0$ every time, so after 1,000 updates it is still $\mathbf{1.0}$.
- FP32 master copy: $\mathbf{1.1}$ (the exact answer).
- Meaning: the 16-bit weight never moved. A 1 billion-parameter model needs 16 GB of parameter state either way: mixed precision does not shrink that part, the code's last line shows both are 16.

**Further notes:**
- **Not tested:** speed (no GPU here), whole-training accuracy, and BF16 training.
- **Precision of 16-bit numbers:** FP16 has about 3 decimal digits, which is why a step of $10^{-4}$ on a weight near 1 vanishes (the spacing of FP16 numbers near 1 is about $10^{-3}$, from the 10 mantissa bits).

**What came before, and how this improved on it:**
- **Before:** training everything in 32-bit floating point.
- **Its limit:** 32-bit numbers use twice the memory and move twice the data of 16-bit ones, and GPUs run 16-bit math faster (the lecture's picture of lower precision giving faster processing; I did not measure it).
- **What changed:** 16-bit working copies, a 32-bit master copy and loss scaling (Micikevicius et al., arXiv 1710.03740, 10 October 2017, abstract read).
- **How much better:** the abstract reports nearly 2 times less memory and expects speed-ups on future half-precision hardware, with accuracy kept across convolutional, recurrent and generative models and models over 100 million parameters. In this note's code, the 16-bit-only weight stayed at $1.0$ while the master copy reached $1.1$, and the parameter state is 16 bytes per parameter in both setups, so the saving does not come from there.

## Related
- [[GPU]] · [[Quantization]] · [[Data Parallelism]] · [[Adam Optimizer]] · [[Backpropagation]] · [[FlashAttention]] · [[LLM]] · [[Pre-training]] · [[Gradient Descent]] · [[Gradient Accumulation]] · [[Gradient Checkpointing]]

## Sources
- Stanford CME 295 (Fall 2025), Lecture 4 slides, Afshine and Shervine Amidi
- Micikevicius et al., *Mixed Precision Training*, 2017 (arXiv 1710.03740, abstract read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/training Mixed precision training:::Use short 16-bit numbers for most of the work and keep one careful 32-bit master copy of the weights, so tiny updates are not lost.

#flashcards/ml/formula #flashcards/topic/training Write loss scaling and explain each symbol.
?
$$g_{\text{stored}}=\text{fp16}(s\cdot g),\qquad g_{\text{used}}=\frac{g_{\text{stored}}}{s}$$
- $g$: the true gradient. $s>1$: the scale (for example 1024).
- $\text{fp16}(\cdot)$: rounded to a 16-bit number, which turns anything below about $6\times10^{-8}$ into 0 and loses precision below $6.1\times10^{-5}$.
- $g_{\text{stored}}$: what the backward pass keeps in 16 bits. $g_{\text{used}}$: the gradient actually used to update the 32-bit master weights.
- Scaling the loss scales every gradient by $s$ (the chain rule is linear), lifting tiny values into FP16 range. Too large a scale overflows (above 65,504), so real systems adjust it.

#flashcards/ml/example #flashcards/topic/training $g=10^{-8}$, $s=1024$. What happens with and without scaling?::Stored directly in FP16 it is 0 (underflow). Scaled: $1024\times10^{-8}=1.02\times10^{-5}$, which FP16 holds as $1.0252\times10^{-5}$. Divide back: $1.0012\times10^{-8}$. The gradient survived with about 0.1% rounding error.

#flashcards/ml/formula #flashcards/topic/training Write the bytes of training state per parameter and explain each term.
?
$$\underbrace{2}_{\text{FP16 weights}}+\underbrace{2}_{\text{FP16 grads}}+\underbrace{4}_{\text{FP32 master}}+\underbrace{4}_{\text{momentum}}+\underbrace{4}_{\text{variance}}=16\text{ bytes}$$
- The first two terms are the working copies used in the passes.
- The master weights, the momentum and the variance are the 32-bit state the optimizer keeps (Adam).
- Almost all of it is the optimizer's 12 bytes. Without the 32-bit master, tiny updates are lost.
- This 16-byte total is also the figure the memory sharing in data parallelism starts from.

#flashcards/ml/example #flashcards/topic/training Add $+0.0001$ to a weight of $1.0$, 1,000 times. What do FP16 and FP32 end at?::FP16 weight: $1.0+0.0001$ rounds back to $1.0$ every time, so it stays 1.0. FP32 master copy: 1.1 (the exact answer). A 1-billion-parameter model needs 16 GB of parameter state either way: mixed precision does not shrink that part.

#flashcards/ml/cloze #flashcards/topic/training The weight update keeps a ==32-bit master copy==, because 16-bit numbers would round tiny updates away.

#flashcards/ml/example #flashcards/topic/training Why do very small gradients need loss scaling in FP16?::They become exactly 0 in FP16. Scaling the loss lifts them into range, and they are divided back before the update.
