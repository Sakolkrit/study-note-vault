---
tags: [LLM, machine-learning, inference, compression, efficiency, numerics]
status: growing
level: 4
tier: 3
---

# Quantization

Quantization means **storing a model's numbers with fewer bits**, rounding each one to the nearest of a small set of allowed values, so the model gets much smaller and faster, at the cost of a little accuracy.

## 🟢 1. Simple (age 12)
Imagine measuring your height with a ruler that has centimetre marks, instead of one with tiny millimetre marks. You will say "152 cm" instead of "151.7 cm". The number you write is a bit off, but it is shorter to write down and easier to handle. If you use a ruler with only a few big marks (every 10 cm), you save even more effort, but your answers get rougher. Quantization is choosing a **simpler ruler for the model's numbers**: a smaller set of allowed values, so each number takes less space.

**Example:** A thermometer that shows only whole degrees tells you "21" instead of "21.4". For deciding whether to wear a coat, nothing is lost. If you used a thermometer that only said "cold", "warm" or "hot", most choices would still work, but some would go wrong.

## 🔵 2. How it works
A trained [[LLM]] is mostly a giant pile of numbers (its weights, see [[Neural Network]]). Normally each one uses 16 bits. Quantization shrinks them.
1. **Look at the range:** find the biggest weight (in size) in a group of weights.
2. **Pick a scale:** decide how big one step of the new, coarser ruler is, so that the biggest weight just fits at the top.
3. **Round:** for each weight, divide by the scale and round to the nearest whole number. These small whole numbers (for example from -7 to 7 for 4 bits) are what you store.
4. **Store the scale** next to the little whole numbers.
5. **To use the weight,** multiply the whole number back by the scale. You get something close to the original, but not exactly it. That gap is the **quantization error**.
6. **Check the damage:** test the model. With 8 bits it is usually almost unchanged, and with 4 bits it is often still good, especially with clever methods. With 2 bits, quality usually falls apart.

![[quantization-staircase.svg]]
*Quantization of a smooth signal. The grey curve is the original. The red staircase is the same signal forced onto a few allowed levels (the dashed lines). The gap between grey and red is the quantization error. With more levels (more bits), the stairs get finer and the gap shrinks. The same idea applies to a model's weights. Image: Wikimedia Commons file [Quantized.signal.svg](https://commons.wikimedia.org/wiki/File:Quantized.signal.svg), public domain (author not given on the file page; the page notes it was assumed to be by Rbj).*

## 🟠 3. Technical detail
- **Why bother:** [[GPU]] memory and the speed of reading it are the main limits when running a big model. Fewer bits per weight means a smaller model that fits on cheaper hardware and loads faster. Running a model is often limited by memory speed, not arithmetic, so smaller weights can also mean faster answers (from memory; depends on hardware).
- **Common formats:** 16-bit floating point (the usual training format), 8-bit integers, and 4-bit integers or 4-bit special number formats. Some research goes to 3, 2 or even about 1.6 bits (from memory).
- **When it is applied:**
  - **After training** (post-training quantization): take a finished model and round it. Cheap and common.
  - **During training** (quantization-aware training): the model learns while seeing the rounding, so it adapts. More work, better quality at very low bit counts.
  - **For fine-tuning:** a quantized base model can be frozen while a small add-on is trained (the idea behind QLoRA, from memory; see [[Fine-tuning]]).
- **One scale vs many:** using one scale for a whole weight matrix is simple, but a single large weight (an *outlier*) forces a big scale, and then all the ordinary weights round to zero (shown in the code). The fix is **group-wise scales**: one scale for each small block, such as 64 or 128 weights. Some methods also protect or rescale the most important weights (names such as GPTQ and AWQ; from memory, not covered here).
- **Weights vs activations vs the cache:** you can quantize the stored weights, the temporary values flowing through the layers (activations), and the saved keys and values of [[Attention]] (the KV cache). Activations are harder because they have large outliers and change with every input.
- **Integer arithmetic:** if both inputs are small integers, the hardware can multiply them as integers (fast and cheap) and rescale once at the end. This is shown below.
- **Rounding is not free:** the error adds up through many layers, and some layers and tasks are more sensitive than others. Always measure quality afterwards, for example with [[Perplexity]] and real tasks (see [[Evaluation Metrics]]).
- **Not the same as pruning or distillation:** pruning deletes weights, and distillation trains a smaller model to copy a larger one. Quantization keeps the same structure and only coarsens the numbers. They can be combined.
- **A cousin of low-rank:** both quantization and low-rank tricks (see [[Singular Value Decomposition]], LoRA) shrink models, in different ways: fewer bits per number against fewer numbers.
- **Connection to Mixture of Experts:** an MoE model has a huge total size ([[Mixture of Experts]]), so quantizing it is a common way to make it fit in memory.

```python
import numpy as np
def quant(x, bits, scale=None):
    qmax = 2 ** (bits - 1) - 1                               # 127 for 8 bits, 7 for 4 bits
    s = np.abs(x).max() / qmax if scale is None else scale   # one scale: biggest weight maps to qmax
    q = np.clip(np.round(x / s), -qmax, qmax).astype(np.int32)
    return q, s
w = np.array([0.9, -0.31, 0.04, -1.2])
for b in (8, 4):
    q, s = quant(w, b); print(b, round(float(s), 5), q.tolist(), (q * s).round(3).tolist(), round(float(np.abs(w - q * s).max()), 4), round(float(s / 2), 4))
# 8 0.00945 [95, -33, 4, -127] [0.898, -0.312, 0.038, -1.2] 0.0024 0.0047
# 4 0.17143 [5, -2, 0, -7] [0.857, -0.343, 0.0, -1.2] 0.0429 0.0857
# a random layer at different bit widths: relative error of the layer's output
rng = np.random.default_rng(0)
W = rng.normal(0, 0.05, (256, 256)); x = rng.normal(0, 1, 256)
ref = W @ x
for b in (8, 4, 2):
    q, s = quant(W, b); y = (q * s) @ x
    print(b, round(float(np.linalg.norm(y - ref) / np.linalg.norm(ref)), 4))     # 8 0.0111 | 4 0.2159 | 2 0.9704
# integer matmul: quantise both, multiply integers, rescale once
qw, sw = quant(W, 8); qx, sx = quant(x, 8)
y_int = (qw @ qx) * (sw * sx); y_deq = (qw * sw) @ (qx * sx)
print(np.allclose(y_int, y_deq), round(float(np.linalg.norm(y_int - ref) / np.linalg.norm(ref)), 4))   # True 0.0138
# one outlier ruins a single scale; per-group scales limit the damage
v = rng.normal(0, 0.1, 256); v[0] = 10.0
q, s = quant(v, 4); rest = v[1:]
e1 = np.linalg.norm(rest - (q * s)[1:]) / np.linalg.norm(rest)
dq = np.concatenate([quant(r, 4)[0] * quant(r, 4)[1] for r in v.reshape(4, 64)])
e3 = np.linalg.norm(rest[63:] - dq[64:]) / np.linalg.norm(rest[63:])
print(round(float(e1), 3), round(float(e3), 3), int((q[1:] == 0).sum()), round(float(s), 3))   # 1.0 0.106 255 1.429
for bits in (16, 8, 4): print(bits, 7e9 * bits / 8 / 1e9)    # 16 14.0 | 8 7.0 | 4 3.5   (GB for a 7-billion-weight model)
```

Reading the output. The four weights at 8 bits come back within 0.0024 of the originals (the largest allowed error is half a step, 0.0047). At 4 bits the steps are coarser (0.171) and the worst error is 0.043 (allowed 0.086). The 0.04 weight became exactly 0, and the 0.9 weight came back as 0.857. On a random $256\times256$ layer, rounding the weights changes the layer's output by about 1.1% at 8 bits, 21.6% at 4 bits, and 97% at 2 bits, where only the values $-1$, $0$ and $1$ are left. The random weights here are the easy case for plain rounding. Real networks have outliers and need smarter methods to do well at 4 bits. The integer multiplication gives the same answer as multiplying the rounded values back in decimal, up to tiny floating-point differences (`np.allclose`), with 1.4% error against the original. The outlier test: with one weight of 10 among small ones (about 0.1), the single scale is 1.43, so all 255 ordinary weights round to 0 and the error on them is 100%. With separate scales for blocks of 64, the three blocks without the outlier come back with about 10.6% error, though the outlier's own block still loses its small weights. The last line is the memory: a 7-billion-weight model takes 14 GB at 16 bits, 7 GB at 8 bits and 3.5 GB at 4 bits, counting the weights only (not the scales or other overheads).

## 🔴 4. Deep / formal
**Round and scale.** For a weight $x$, a bit width $b$ and a scale $s$, quantizing gives a whole number $q$, and the stored value is read back as $\hat x$:

$$
q=\operatorname{clip}\!\Big(\operatorname{round}\!\big(\tfrac{x}{s}\big),\,-q_{\max},\,q_{\max}\Big),\qquad \hat x=s\,q,\qquad s=\frac{\max|x|}{q_{\max}},\quad q_{\max}=2^{\,b-1}-1
$$

**Where:**
- $x$ is the original weight. $\max|x|$ is the biggest weight in the group, ignoring sign.
- $b$ is the number of bits. $q_{\max}=2^{b-1}-1$ is the largest whole number that fits (127 for 8 bits, 7 for 4 bits). One bit goes to the sign.
- $s$ is the scale: the size of one step. Dividing by it turns $x$ into a count of steps.
- $\operatorname{round}$ picks the nearest whole number. $\operatorname{clip}$ forces the result to stay between $-q_{\max}$ and $q_{\max}$.
- $q$ is the whole number you store (small). $\hat x=s\,q$ is the rebuilt weight.
- **Matters:** the scale ties the coarse ruler to the data, so the biggest weight lands exactly on the top mark. Too small a scale clips big weights flat. Too big a scale throws away small weights (they round to 0). Without the scale, you could not map the floating weights onto few whole numbers at all.

**Example (tiny numbers):** four weights $0.9,\,-0.31,\,0.04,\,-1.2$ at $b=4$.
- $q_{\max}=7$ and $\max|x|=1.2$, so $s=1.2/7=\mathbf{0.1714}$.
- $0.9/0.1714=5.25\to q=\mathbf{5}$, and $\hat x=5\times0.1714=0.857$.
- $-0.31/0.1714=-1.81\to q=-2$, and $\hat x=-0.343$. $\;0.04/0.1714=0.23\to q=0$, and $\hat x=0$. $\;-1.2/0.1714=-7\to q=-7$, and $\hat x=-1.2$ exactly.
- Meaning: the largest weight is kept exactly. The others move by up to 0.043. The smallest weight (0.04) disappears to zero.

**Size of the error.** If no weight is clipped, rounding to the nearest step can never miss by more than half a step:

$$
\lvert x-\hat x\rvert\;\le\;\frac{s}{2}
$$

**Where:**
- $x$ and $\hat x$ are the original and the rebuilt weight. $s$ is the scale (the step size).
- **Matters:** the error is set by the scale, which is set by the biggest weight. A group with an outlier has a big $s$, so *every* weight in that group is allowed a big error. That is the reason for per-group scales and for protecting outliers. More bits make $s$ smaller (halving it for every extra bit, roughly), so the error falls fast with bit width.

**Example (tiny numbers):** the 4-bit case above, $s=0.1714$.
- Allowed error: $s/2=\mathbf{0.0857}$.
- Actual errors: $0.9\to0.857$ gives $0.043$. $-0.31\to-0.343$ gives $0.033$. $0.04\to0$ gives $0.04$. $-1.2\to-1.2$ gives $0$.
- Biggest actual error: $0.043\le0.0857$.
- Meaning: all errors are within half a step, and in this case even within a quarter step. At 8 bits the step is $0.00945$, so the allowed error is only $0.0047$ (the code shows $0.0024$).

**Multiplying quantized numbers.** In a layer, most work is a sum of weight times input. If both are quantized, the whole sum can be done on whole numbers, then rescaled once:

$$
\sum_i w_i\,x_i\;\approx\;s_w\,s_x\sum_i q^{w}_i\,q^{x}_i
$$

**Where:**
- $w_i$ and $x_i$ are the real weights and inputs, with scales $s_w$ and $s_x$. $q^{w}_i$ and $q^{x}_i$ are their stored whole numbers.
- The sum on the right uses **only integers**, which hardware can do fast and cheaply. The two scales are multiplied in once, at the end.
- **Matters:** this is why quantization can speed things up, not just save space. The rounding errors are still there, so the answer is only approximate, but the cost of the main sum is much smaller. Without the single rescale, you would have to convert every number back to decimals before multiplying and lose the saving.

**Example (tiny numbers):** weights $w=(0.3,\,0.7)$ and inputs $x=(1.4,\,0.6)$, with scales $s_w=0.1$ and $s_x=0.2$.
- Whole numbers: $q^{w}=(3,\,7)$ and $q^{x}=(7,\,3)$.
- Integer sum: $3\times7+7\times3=21+21=42$.
- Rescale: $0.1\times0.2\times42=\mathbf{0.84}$.
- Check with the real numbers: $0.3\times1.4+0.7\times0.6=0.42+0.42=0.84$. They match, because these numbers were picked to sit exactly on the steps. With real data there is a small error.

**Memory.** The weights take this much space:

$$
\text{bytes}=\text{(number of weights)}\times\frac{\text{bits}}{8}
$$

**Where:**
- The number of weights is the parameter count. Bits is the width per weight. Dividing by 8 turns bits into bytes.
- **Matters:** it is a plain proportion, so halving the bits halves the memory. It ignores the scales (a small extra), the KV cache, and other things that also need memory, so real use is somewhat higher.

**Example (tiny numbers):** a model with $7\times10^{9}$ weights.
- 16 bits: $7\times10^{9}\times2=14\times10^{9}$ bytes $=\mathbf{14}$ GB.
- 8 bits: $\mathbf{7}$ GB. 4 bits: $\mathbf{3.5}$ GB.
- Meaning: a model that does not fit on a 12 GB graphics card at 16 bits (14 GB) fits easily at 8 bits or 4 bits.

**Further notes:**
- **Symmetric vs shifted:** here zero maps to zero. Variants add an offset (a "zero point") so the range need not be centred on zero, which suits values that are always positive.
- **Non-uniform levels:** the allowed values need not be evenly spaced. Some 4-bit formats place more levels near zero, where weights are densest (from memory; for example the NF4 format).
- **Rounding up or down:** plain rounding is not always best. Some methods choose roundings that keep the layer's output close, not each weight (from memory).
- **Floating point is also a quantization:** 16-bit floats already round numbers; going to 8-bit floats or integers is the same idea with fewer bits.
- **Open questions:** how low the bit count can go, how it hurts reasoning or rare knowledge, and how best to handle outliers in activations are still being researched.

**What came before, and how this improved on it:**
- **Before:** store and compute with 16- or 32-bit floating-point numbers.
- **Its limit:** big models need huge memory. The LLM.int8() abstract says large language models "require significant GPU memory for inference".
- **What changed:** store the weights as 8-bit integers or fewer, with a scale. LLM.int8() halves inference memory with no degradation they could measure, by keeping the few outlier features (under 0.1% of values) in 16 bits (Dettmers et al., arXiv 2208.07339, 15 August 2022, abstract read; the author name is from memory). GPTQ compresses a 175B-parameter model to 3 or 4 bits per weight in about four GPU hours, with negligible loss, and reports inference speedups of about 3.25 times on A100 GPUs (arXiv 2210.17323, 31 October 2022, abstract read).
- **How much better:** in the code below, int8 makes a 512 by 512 matrix 4 times smaller (1,048,576 bytes to 262,144), with a relative error of 0.0109 in the product. After making one column 60 times larger, one scale for the whole matrix gives 0.2170 error, and one scale per row gives 0.0567. That is the same outlier problem LLM.int8() describes, in miniature.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
W = rng.standard_normal((512, 512)).astype(np.float32); x = rng.standard_normal(512).astype(np.float32)
def q8(M, axis=None):
    s = np.abs(M).max(axis=axis, keepdims=True) / 127; return np.round(M / s).astype(np.int8), s
def err(W, x):
    Wq, s = q8(W); y8 = (Wq.astype(np.float32) * s) @ x; y = W @ x; return np.linalg.norm(y8 - y) / np.linalg.norm(y)
print("memory: float32", W.nbytes, "bytes, int8", W.size, "bytes (", W.nbytes // W.size, "x smaller )")
print("relative error of W @ x with one 8-bit scale for the whole matrix:", f"{err(W, x):.4f}")
W2 = W.copy(); W2[:, 7] *= 60                                 # one outlier feature, like those described in the LLM.int8() paper
print("same, after making one column 60 times larger:", f"{err(W2, x):.4f}")
Wq, s = q8(W2, axis=1); print("same, with one scale per row instead:", f"{np.linalg.norm((Wq.astype(np.float32) * s) @ x - W2 @ x) / np.linalg.norm(W2 @ x):.4f}")
```

## Related
- [[LLM]] · [[Neural Network]] · [[GPU]] · [[Fine-tuning]] · [[Attention]] · [[Mixture of Experts]] · [[Singular Value Decomposition]] · [[Perplexity]] · [[Evaluation Metrics]] · [[Pre-training]] · [[Scaling Laws]] · [[Normalization]] · [[Vector Database]] · [[Autoencoder]] · [[Knowledge Distillation]] · [[Self-Attention]] · [[LoRA]] · [[KV Cache]] · [[Mixed Precision Training]] · [[PagedAttention]] · [[QLoRA]] · [[Local CPU Chatbot]] · [[GGUF]] · [[llama.cpp]] · [[Ollama]] · [[Hugging Face Transformers]]
- [[ONNX]]
- [[MobileNet]]
## Sources
- Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*, 2018 (from memory)
- Frantar et al., *GPTQ*, 2022 (from memory)
- Lin et al., *AWQ: Activation-aware Weight Quantization*, 2023 (from memory)
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, 2023 (from memory)
- Gholami et al., *A Survey of Quantization Methods for Efficient Neural Network Inference*, 2021 (from memory)
- Dettmers et al., LLM.int8(), arXiv 2208.07339 and Frantar et al., GPTQ, arXiv 2210.17323 (abstracts read for the history block)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving Quantization:::Storing a model's numbers with fewer bits, like measuring with a ruler that has fewer marks. The numbers are a bit off, but they take less memory and are faster to use.

#flashcards/ml/formula #flashcards/topic/llm-serving Write symmetric quantization and explain each symbol.
?
$$q=\operatorname{clip}\!\Big(\operatorname{round}\!\big(\tfrac{x}{s}\big),\,-q_{\max},\,q_{\max}\Big),\qquad \hat x=s\,q,\qquad s=\frac{\max|x|}{q_{\max}},\quad q_{\max}=2^{\,b-1}-1$$
- $x$: the original weight. $\max|x|$: the biggest weight in the group, ignoring sign.
- $b$: the number of bits. $q_{\max}=2^{b-1}-1$: the largest whole number that fits (127 for 8 bits, 7 for 4 bits). One bit goes to the sign.
- $s$: the scale, the size of one step. Dividing by it turns $x$ into a count of steps.
- $q$: the stored whole number after rounding and clipping. $\hat x=sq$: the rebuilt weight.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the quantization error bound and explain each symbol.
?
$$\lvert x-\hat x\rvert\;\le\;\frac{s}{2}$$
- $x$, $\hat x$: the original and the rebuilt weight. $s$: the scale (step size).
- The error is at most half a step. It is set by the scale, which is set by the biggest weight.
- A group with an outlier has a big $s$, so every weight in that group is allowed a big error. That is why per-group scales and protecting outliers matter. More bits make $s$ smaller.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the integer dot product trick and explain each symbol.
?
$$\sum_i w_i\,x_i\;\approx\;s_w\,s_x\sum_i q^{w}_i\,q^{x}_i$$
- $w_i$, $x_i$: the real weights and inputs, with scales $s_w$ and $s_x$. $q^w_i$, $q^x_i$: their stored whole numbers.
- The sum on the right uses only integers, which hardware does fast and cheaply. The two scales are multiplied in once, at the end.
- This is why quantization can speed things up, not just save memory.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the memory a model needs and explain each symbol.
?
$$\text{bytes}=\text{(number of weights)}\times\frac{\text{bits}}{8}$$
- The number of weights is the parameter count. Bits is the width per weight. Dividing by 8 turns bits into bytes.
- It is a plain proportion, so halving the bits halves the memory. It ignores the scales, the KV cache and other memory, so real use is somewhat higher.

#flashcards/ml/cloze #flashcards/topic/llm-serving Quantization error is at most ==half a step== (s/2), and the step is set by the ==biggest== weight in the group.

#flashcards/ml/example #flashcards/topic/llm-serving Weights 0.9, −0.31, 0.04, −1.2 at $b=4$. What are $s$, $q$ and $\hat x$?::q_max = 7, s = 1.2/7 = 0.1714. q = 5, −2, 0, −7. x̂ = 0.857, −0.343, 0, −1.2. Errors 0.043, 0.033, 0.04, 0, all within s/2 = 0.0857.

#flashcards/ml/example #flashcards/topic/llm-serving $w=(0.3,0.7)$, $x=(1.4,0.6)$, $s_w=0.1$, $s_x=0.2$. Do the integer trick.::q^w = (3, 7), q^x = (7, 3). Integer sum 21 + 21 = 42. 0.1 × 0.2 × 42 = 0.84. The real value 0.42 + 0.42 = 0.84 matches.

#flashcards/ml/example #flashcards/topic/llm-serving How much memory does a 7-billion-weight model need at 16, 8 and 4 bits?::16 bits: 14 GB. 8 bits: 7 GB. 4 bits: 3.5 GB. A model that does not fit a 12 GB card at 16 bits fits at 8 or 4.

#flashcards/ml/image #flashcards/topic/llm-serving ![[quantization-staircase.svg]]
What does this picture show?
?
Quantization of a smooth signal. The grey curve is the original. The red staircase is the same signal forced onto a few allowed levels (the dashed lines). The gap between grey and red is the quantization error. With more levels (more bits) the stairs get finer and the gap shrinks.
