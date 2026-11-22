---
tags: [LLM, fine-tuning, quantization, LoRA, memory]
status: growing
level: 4
tier: 3
---

# QLoRA

**QLoRA** fine-tunes a big language model whose weights are squeezed to 4 bits, training only a small add-on, so a huge model fits on one GPU.

## 🟢 1. Simple (age 12)
Teaching a very big model new tricks normally needs a huge computer. QLoRA keeps the big model in a tiny, squeezed form that is frozen, and teaches only a small add-on notebook on the side. That is enough to fit a giant on a single graphics card.

**Example:** Instead of rewriting a whole encyclopedia, you keep it on a shelf in a small, compressed edition and just write notes in the margin. The margin notes are the only thing you change.

## 🔵 2. How it works
1. Start with a large pre-trained model ([[Pre-training]]). Its weights are big numbers that take a lot of memory.
2. **Squeeze** the weights to 4 bits each ([[Quantization]]), using a special 4-bit scale designed for bell-shaped weights (called NF4, "NormalFloat").
3. **Freeze** these squeezed weights: they are never changed during fine-tuning.
4. Add small trainable add-ons, [[LoRA]] adapters, beside the frozen layers.
5. During training, the squeezed weights are turned back into normal numbers one piece at a time, only to do the math, then discarded.
6. Gradients flow through the frozen layers into the adapters, and only the adapters are updated ([[Backpropagation]]).
7. Two extra tricks save more memory: **double quantization** (squeezing the scaling numbers too) and **paged optimisers** (moving memory spikes to the computer's main memory).
8. **The catch:** it saves memory, not training time (the squeezing and unsqueezing cost some speed), and 4 bits is still a loss of detail compared with the original (the paper reports little loss; I only read its abstract).

## 🟠 3. Technical detail
- **Built from:** [[Quantization]] (the 4-bit frozen base model), [[LoRA]] (the small trainable adapters) and [[Fine-tuning]].
- **NF4:** a 4-bit data type whose 16 levels are placed to suit weights that follow a bell curve (a normal distribution), so more levels sit where more weights are. The abstract calls it "information theoretically optimal for normally distributed weights".
- **Block-wise scaling:** weights are cut into small blocks, and each block is scaled by its own largest value so that outliers in one block do not spoil the others (the same idea as in the outlier example of [[Quantization]]).
- **Memory arithmetic:** 4 bits is a quarter of 16 bits. For 65 billion parameters that is about 32.5 GB for the weights alone, against 130 GB in 16 bits (my arithmetic). The paper's introduction says regular 16-bit fine-tuning of a 65B LLaMA model needs more than 780 GB of GPU memory and that QLoRA brings the average below 48 GB. The gap between 130 GB and 780 GB is, I think, the gradients, optimiser states and activations that fine-tuning also needs; that explanation is mine, not quoted.
- **Double quantization, with the paper's numbers:** with 32-bit scale constants and a block size of 64, the constants add 32/64 = 0.5 bits per parameter. Quantizing the constants a second time (8-bit floats, block size 256) saves about 0.37 bits per parameter, roughly 3 GB for a 65B model (introduction and Section 3; the arithmetic is in the code below).
- **Results (paper body):** the Guanaco 65B model, 41 GB in size, scores an Elo of 1022 against 966 for ChatGPT and 1348 for GPT-4 in the paper's Table 1 tournament (GPT-4 as the judge, on the Vicuna benchmark), and the paper reports 99.3% of ChatGPT's performance on that benchmark. The smallest model, Guanaco 7B, needs 5 GB and beats a 26 GB Alpaca model by more than 20 percentage points on the Vicuna benchmark. Against GPT-4, Guanaco 65B and 33B have an expected win probability of 30% by human-annotator Elo.
- **Does 4-bit lose accuracy? (Table 3):** on Super-NaturalInstructions with a 3-billion-parameter T5, the score (RougeL) is 54.3 for 16-bit full fine-tuning, 55.4 for 16-bit LoRA and 55.3 for QLoRA with NF4 and double quantization, so QLoRA matches 16-bit LoRA there. The paper also says that, in a second test (Table 4: LLaMA 7B to 65B fine-tuned on Alpaca and FLAN v2, 5-shot MMLU), NF4 with double quantization fully recovers the 16-bit LoRA score while QLoRA with FP4 lags it by about 1 percentage point.
- **Data quality over quantity:** the paper finds that raising dataset size or epochs improves MMLU only marginally (0.0 to 0.5 points), while the choice of dataset changes it by up to 1.5 to 8.0 points. It also says current chatbot benchmarks are not trustworthy ([[Benchmarks]], [[LLM as a Judge]]).
- **Relatives:** [[Mixed Precision Training]] is about the numbers used while training, QLoRA is about storing the frozen weights in fewer bits.

```python
import numpy as np
from statistics import NormalDist
rng = np.random.default_rng(0); w = rng.standard_normal(64 * 4096)                    # weights roughly bell-shaped
blocks = w.reshape(-1, 64); am = np.abs(blocks).max(1, keepdims=True)
nd = NormalDist(); q = np.array([nd.inv_cdf((i + 0.5) / 16) for i in range(16)]); q = q / np.abs(q).max()     # 16 levels bunched where bell-shaped weights are
uni = np.linspace(-1, 1, 16)                                                          # 16 evenly spaced levels
def quant(levels):
    n = blocks / am; idx = np.abs(n[..., None] - levels).argmin(-1); return levels[idx] * am
for name, lv in (("evenly spaced 4-bit levels", uni), ("bell-curve-spaced 4-bit levels (NF4 idea)", q)):
    print(f"{name:44s} mean squared error {np.mean((quant(lv) - blocks) ** 2):.5f}")
for p in (7e9, 65e9): print(f"{p / 1e9:.0f}B parameters: 16-bit {p * 2 / 1e9:.1f} GB, 4-bit {p * 0.5 / 1e9:.1f} GB (weights only, before the small extra for scales and adapters)")
const = 32 / 64                                                                       # one 32-bit scale per block of 64 weights = 0.5 bits per weight
print(f"scale constants: {const} bits per parameter = {65e9 * const / 8 / 1e9:.1f} GB for 65B; double quantization saves about 0.37 of them = {65e9 * 0.37 / 8 / 1e9:.1f} GB (the paper's figure)")
```

## 🔴 4. Deep / formal
**Squeezing a block of weights.** Take a block of weights $w$ and divide by the block's largest size $a=\max|w|$. Then each normalised weight is replaced by the nearest of 16 fixed levels $\ell_i$ (the 4-bit codes):

$$
\hat w = a\cdot\ell_{\,j},\qquad j=\arg\min_i\left|\frac{w}{a}-\ell_i\right|
$$

**Where:**
- $w$ is a weight in the block and $\hat w$ is its squeezed version. $a=\max|w|$ is the block's largest size (the scale, stored once per block).
- $\ell_i$ are the 16 levels between $-1$ and $1$. In NF4 they are spaced to match a bell curve; evenly spaced levels are the simple alternative.
- $\arg\min_i$ means "the index $i$ that makes the gap smallest". $|\cdot|$ is the size of the gap.
- **Matters:** the scale $a$ lets one set of 16 levels fit every block. Placing the levels where the weights are thick (near zero for a bell curve) makes the typical error smaller than evenly spaced levels.

**Example (tiny numbers):** a block with largest size $a=0.6$, four levels $\{-1,-0.3,0.3,1\}$ for simplicity, and a weight $w=0.2$.
- Normalised: $0.2/0.6=\mathbf{0.333}$.
- Nearest level: $0.3$ (the gap is $0.033$; the next one, $-0.3$, is $0.633$ away).
- Squeezed value: $\hat w=0.6\cdot0.3=\mathbf{0.18}$.
- Meaning: the weight $0.20$ is stored as $0.18$, an error of $0.02$ that the code of 4 bits per weight (16 levels in the real thing) keeps small.

**Memory for the weights.** With $N$ parameters stored in $b$ bits each, the weights take

$$
\text{bytes}=N\cdot\frac{b}{8}
$$

**Where:**
- $N$ is the number of parameters. $b$ is the number of bits stored per parameter (16 normally, 4 for the frozen weights in QLoRA). Dividing by 8 turns bits into bytes.
- **Matters:** cutting $b$ from 16 to 4 cuts the weight memory by 4 times. Activations, the adapters, optimiser states and the per-block scales come on top, which is why the real total is larger.

**Example (tiny numbers):** $N=65\times10^{9}$.
- 16 bits: $65\cdot10^9\cdot16/8=\mathbf{130}$ GB.
- 4 bits: $65\cdot10^9\cdot4/8=\mathbf{32.5}$ GB.
- Meaning: 4-bit weights bring a 65B model within reach of a single 48 GB GPU, which is what the abstract reports.

**Further notes:**
- **My NF4 is simplified:** the code uses equal-probability slices of a normal curve, scaled to $\pm1$. The real NF4 has details I did not reproduce (for example an exact zero level; from memory).
- **Paged optimisers** (using NVIDIA unified memory to absorb memory spikes) are described in the paper but not simulated here. Double quantization is only counted, not simulated.

**What came before, and how this improved on it:**
- **Before:** full 16-bit fine-tuning of the whole model, or [[LoRA]] on top of a 16-bit base.
- **Its limit:** a 65B model needs far more memory than one GPU has, even with LoRA, because the frozen base itself is large.
- **What changed:** store the frozen base in 4 bits with a data type fitted to bell-shaped weights, and train LoRA adapters through it (Dettmers et al., arXiv 2305.14314, 23 May 2023; abstract and parts of the body read; first author from memory).
- **How much better:** from the paper (introduction and tables, copied by hand): 16-bit fine-tuning of a 65B model needs more than 780 GB of GPU memory, QLoRA needs under 48 GB, and on one 3B-parameter T5 test QLoRA with NF4 scores 55.3 against 55.4 for 16-bit LoRA. In the code below (my own toy), 4-bit storage cuts a 65B model's weights from 130 GB to 32.5 GB, and bell-curve-spaced levels give a mean squared error of 0.00840 against 0.01017 for evenly spaced ones on bell-shaped weights, 17% lower (a modest gain in the toy). The scale constants cost 0.5 bits per parameter (4.1 GB for 65B), and double quantization saves 0.37 of them (3.0 GB). **Honest limits:** the quality results are the paper's, not reproduced, and its Elo table is judged by GPT-4 on the small Vicuna benchmark.

**Check it yourself:** run the code in level 3 above. It prints every number quoted in this block.

## Related
[[Quantization]] · [[LoRA]] · [[Fine-tuning]] · [[Pre-training]] · [[Mixed Precision Training]] · [[Backpropagation]] · [[Benchmarks]] · [[LLM as a Judge]] · [[Knowledge Distillation]] · [[Gradient Accumulation]] · [[Catastrophic Forgetting]] · [[Unsloth]] · [[LoRA Fine-tuning on CPU]] · [[QLoRA 2023]]

## Sources
- Stanford CME 295 (Fall 2025), lecture slides, Afshine and Shervine Amidi (lecture 4, the slides on parameter-efficient fine-tuning)
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, arXiv 2305.14314 (2023): abstract and parts of the body read (introduction, Section 3 on double quantization, Tables 1 and 3, the Guanaco results and the dataset-size findings), taken from an HTML copy of the paper; the first author is from memory. The numbers above were copied by hand. (a fuller reading, with every result table, is in [[QLoRA 2023]])
- No image: I found no free, on-topic picture for this note on Wikimedia Commons, so I added none.
- Every number in the history block was computed by the code in this note (toy data; dates and names marked "from memory" are not checked).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm QLoRA:::Keep the big model frozen in a tiny squeezed 4-bit form and teach only a small add-on (LoRA adapters) on the side, so a giant model fits on a single graphics card.

#flashcards/ml/formula #flashcards/topic/llm Write block quantisation to 16 levels and explain each symbol.
?
$$\hat w = a\cdot\ell_{\,j},\qquad j=\arg\min_i\left|\frac{w}{a}-\ell_i\right|$$
- $w$: a weight in the block. $\hat w$: its squeezed version. $a=\max|w|$: the block's largest size (the scale, stored once per block).
- $\ell_i$: the 16 levels between $-1$ and $1$. In NF4 they are spaced to match a bell curve. Evenly spaced levels are the simple alternative.
- $\arg\min_i$: the index $i$ that makes the gap smallest.
- The scale $a$ lets one set of 16 levels fit every block. Placing levels where the weights are thick (near zero for a bell curve) makes the typical error smaller.

#flashcards/ml/example #flashcards/topic/llm Block largest size $a=0.6$, levels $\{-1,-0.3,0.3,1\}$, weight $w=0.2$. What is $\hat w$?::Normalised $0.2/0.6=0.333$. Nearest level 0.3 (gap 0.033, against 0.633 for $-0.3$). $\hat w=0.6\cdot0.3=0.18$. The weight 0.20 is stored as 0.18, an error of 0.02. The real thing has 16 levels, which keeps the error smaller.

#flashcards/ml/formula #flashcards/topic/llm Write the memory for weights stored at $b$ bits and explain each symbol.
?
$$\text{bytes}=N\cdot\frac{b}{8}$$
- $N$: the number of parameters. $b$: bits stored per parameter (16 normally, 4 for the frozen weights in QLoRA). Dividing by 8 turns bits into bytes.
- Cutting $b$ from 16 to 4 cuts the weight memory 4 times. Activations, adapters, optimiser states and per-block scales come on top, so the real total is larger.

#flashcards/ml/example #flashcards/topic/llm $N=65\times10^9$. What is the weight memory at 16 bits and at 4 bits?::16 bits: $65\cdot10^9\cdot16/8=130$ GB. 4 bits: $65\cdot10^9\cdot4/8=32.5$ GB. That brings a 65B model within reach of a single 48 GB GPU, which the abstract reports.

#flashcards/ml/cloze #flashcards/topic/llm In QLoRA the big model is ==frozen== in 4-bit form and only the ==adapters== are trained.
