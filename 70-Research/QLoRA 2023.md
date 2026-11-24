---
tags: [paper, llm, fine-tuning, quantization, lora, parameter-efficient]
link: https://arxiv.org/abs/2305.14314
authors: Dettmers, Pagnoni (equal first authors), Holtzman, Zettlemoyer (University of Washington; from the paper's first page)
year: 2023
status: body-read
field: large language models, fine-tuning
---

# QLoRA 2023

> **In one line for a 12-year-old:** Squeeze a huge language model down to 4-bit numbers so it fits on one computer, freeze it, and teach it new things by training only tiny add-on pieces. The squeezed model learns just as well, and this made it possible to customise giant models on a single graphics card.

Paper: "QLoRA: Efficient Finetuning of Quantized LLMs", Tim Dettmers, Artidoro Pagnoni (equal contribution), Ari Holtzman and Luke Zettlemoyer (University of Washington), arXiv 2305.14314, first posted 23 May 2023 (the first page says "Preprint. Under review"). The arXiv licence is **CC BY 4.0**, so three of its figures are cropped from the paper below. Concept note: [[QLoRA]]. It builds directly on [[LoRA 2021]]. Reference note for this layout: [[Attention Is All You Need 2017]].

## The problem
Fine-tuning a very large model is prohibitively expensive: regular 16-bit fine-tuning of a 65-billion-parameter LLaMA model needs more than 780 GB of GPU memory (paper, 1). My reconstruction of that number, which the paper does not explain: 65 billion parameters times 12 bytes each (2 for the weight, 2 for its gradient, 8 for the two Adam moments in 32 bits) is exactly 780 GB. Quantization (storing weights with fewer bits, see [[Quantization]]) shrinks models, but the paper says such techniques "only work for inference and break down during training" (paper, 1). Could one fine-tune a model stored in 4 bits without losing quality?

## What people did before
- **Full 16-bit fine-tuning:** the 780 GB above.
- **LoRA ([[LoRA 2021]]):** train small adapters on a frozen model. This removes most of the trainable-parameter and optimizer memory, but the frozen 16-bit model itself is still large (paper, 2).
- **Quantization for inference:** methods such as LLM.int8() and SmoothQuant manage outlier values to keep 16-bit quality; others study the effect of plain rounding or optimise the rounding. The paper says 4-bit inference quantization "leads to performance degradation relative to 16-bit" (paper, 4 and 7). The only earlier work on back-propagating through quantized weights beyond 1B parameters is SwitchBack layers (paper, 7).
- **Other parameter-efficient methods** (prompt tuning, IA3, bias tuning and others) are named in the related work; the paper tests only LoRA (paper, 7 and 8).

## The idea
1. **Store the frozen base model in 4 bits and compute in 16 bits.** Whenever a weight block is needed it is converted back ("dequantized") to BFloat16 and the matrix multiplication runs in 16 bits. Gradients flow through the frozen 4-bit model into the [[LoRA]] adapters, and only the adapters get weight updates (paper, 3).
2. **4-bit NormalFloat (NF4).** Pre-trained weights are roughly bell-shaped ([[Normal Distribution]]). NF4 places its 16 levels at quantiles of a normal curve, so each level is used by about the same number of weights, and it includes an exact zero (paper, 3).
3. **Double quantization.** Weights are quantized in blocks of 64, each with its own scaling constant. Storing those constants in 32 bits costs 32/64 = 0.5 bits per weight. Quantizing the constants too (8-bit floats, blocks of 256) brings that to 8/64 + 32/(64 x 256) = 0.127 bits per weight, a saving of 0.373 bits, about 3 GB for a 65B model (paper, 3).
4. **Paged optimizers.** Use NVIDIA unified memory to move optimizer state to CPU memory automatically when the GPU runs out of memory during a long-sequence memory spike (paper, 3).
5. **Adapters on every linear layer.** LoRA on only the query and value matrices (the usual choice) cannot match full fine-tuning for large models; adapters on **all** linear layers are needed. The adapter rank $r$ made no difference in their tests (paper, 4).
6. **Use it at scale.** They fine-tuned more than 1,000 models across eight instruction datasets, LLaMA and T5, 80 million to 65 billion parameters, and released the Guanaco chatbot family (abstract and paper, 1).

## The method, step by step
QLoRA is not a new network shape. It is a recipe with five parts around one linear layer. The first figure shows where the memory goes in three ways of fine-tuning.

![[qlora-fig1-memory-methods.png]]
*The paper's own Figure 1, cropped from the PDF: three ways to fine-tune. Full fine-tuning (left) keeps a 32-bit optimizer state for every weight of a 16-bit transformer; LoRA (middle) keeps a small optimizer state for small 16-bit adapters on a 16-bit transformer; QLoRA (right) does the same with a 4-bit transformer, and the optimizer state can be paged out to the CPU (the pink arrows). Blue arrows are parameter updates, green are gradient flow. Authors: Dettmers, Pagnoni, Holtzman and Zettlemoyer, Figure 1 of arXiv 2305.14314, CC BY 4.0, cropped from the paper and not redrawn.*

### Part 1: block-wise quantization (the basics, paper section 2)
- **What it does:** to store numbers in fewer bits, divide each number by the largest absolute value in its group, multiply by 127 and round to a whole number between -127 and 127 (the 8-bit case in the paper's equation 1). Dequantizing divides by the same scale.
- **Why it matters:** the scale ($c$) is stored once per group so the full range of the low-bit type is used. But one large outlier makes the scale poor for the rest of the group; so the tensor is cut into small **blocks**, each with its own constant (paper, 2).
- **Worked example below** (my arithmetic): with the numbers $0.5, -2.0, 1.0$ the 8-bit scale is 63.5 and the round trip is nearly exact (errors 0.004, 0, 0.008). Add one outlier of 50 to the same group and the scale drops to 2.54, so 0.5 comes back as 0.394 and 1.0 as 1.181 (errors 0.11 and 0.18).

### Part 2: the 4-bit NormalFloat data type (NF4)
- **What it does:** it replaces evenly spaced levels by 16 levels placed so each is equally likely for normally distributed weights. The weights of a block are first divided by their largest absolute value so they lie between -1 and 1, then each weight is replaced by the nearest of the 16 levels (paper, 3 and Appendix E).
- **Why it matters:** most weights are near zero, so dense levels near zero waste less than evenly spaced ones. The paper reports a better perplexity than Int4 and Float4 (its Table 2) and I reproduced the direction with my own test below.
- **The 16 levels (Appendix E)** are: -1, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0, 0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1: seven negative, one zero and eight positive. This matches the paper's recipe: take 8 quantiles for the negative side and 9 for the positive side, each including 0, and drop the duplicate zero (7 + 1 + 8 = 16).

![[qlora-fig3-4bit-datatypes.png]]
*The paper's own Figure 3, cropped from the PDF: mean zero-shot accuracy (five common-sense tasks) of 4-bit LLaMA models against total model bits, for three data types: plain 4-bit Float (blue), NormalFloat (orange) and NormalFloat with double quantization (green). Each line has four points, one per LLaMA size. The NormalFloat lines sit above the Float line at every size; the green line is slightly left of the orange one because double quantization uses fewer bits. Authors: Dettmers, Pagnoni, Holtzman and Zettlemoyer, Figure 3 of arXiv 2305.14314, CC BY 4.0, cropped from the paper.*

### Part 3: double quantization
- **What it does:** the per-block constants (one 32-bit number for every 64 weights) are themselves quantized to 8-bit floats in blocks of 256; the mean is subtracted first because the constants are all positive (paper, 3).
- **Why it matters:** the constants cost 0.5 bits per weight, which is significant next to 4 bits; after double quantization they cost 0.127. For a 65B model the saving is about 3 GB (my arithmetic: 0.373 bits times 65 billion, divided by 8, is 3.03 GB). The paper says it "only leads to minor gains" in accuracy, and its main use is a fine-grained memory control that fits 33B and 65B models into 24 GB and 48 GB GPUs (Figure 3 caption).
- **Worked example (my arithmetic, 8-bit integers in place of the paper's 8-bit floats):** four block constants $0.60, 0.55, 0.72, 0.64$ have mean 0.6275. After subtracting it and rescaling to $\pm127$ they are stored as the integers $-38, -106, 127, 17$ and come back as $0.5998, 0.5503, 0.7200, 0.6399$, errors under 0.0004. Storage: 4 times 32 = 128 bits becomes 4 times 8 + 32 = 64 bits.

### Part 4: LoRA adapters on every layer
- **What it does:** each frozen linear layer gets a small trainable add-on $s\,X L_1 L_2$ that is added to its output (paper, 2).
- **Why it matters:** the adapters are tiny, so their memory is small: for a 7B model with about 0.2% adapter weights they take 26 MB, against 5,048 MB for the 4-bit base model (paper, 2: my arithmetic, 0.5% of the base). The larger memory item is the gradient of the layers' inputs: 567 MB without gradient checkpointing and 18 MB per sequence with it (paper, 2). So the paper concludes that shrinking the adapters saves little, and that more adapters can be used (paper, 2).
- **Which layers?** Figure 2 shows that putting adapters on all linear layers matches the 16-bit result and putting them only on the attention layers does not.

![[qlora-fig2-rougel-adapters.png]]
*The paper's own Figure 2, cropped from the PDF: RougeL on the Alpaca data for LLaMA 7B models. Each dot is one run with a different random seed. Blue dots are 4-bit QLoRA with adapters on all layers (QLoRA-All), only the feed-forward layers (QLoRA-FFN) or only the attention layers (QLoRA-Attention); orange dots are 16-bit full fine-tuning ("Alpaca (ours)", the authors' improved baseline, and Stanford-Alpaca, the original). Authors: Dettmers, Pagnoni, Holtzman and Zettlemoyer, Figure 2 of arXiv 2305.14314, CC BY 4.0, cropped from the paper. I read the values by eye: QLoRA-All about 64.5, QLoRA-FFN about 63.9, QLoRA-Attention about 63.4, Alpaca (ours) about 64.3, Stanford-Alpaca about 59.7 and 61.2.*

### Part 5: paged optimizers
- **What it does:** the optimizer's state lives in memory that the system moves between GPU and CPU automatically, as in ordinary paging between RAM and disk (paper, 3).
- **Why it matters:** a rare long sequence makes gradient checkpointing spike the memory and crash a 33B or 65B run on one GPU. The paper gives no hard measurements for paging, "since the paging only occurs when processing mini-batches with long sequence lengths, which is rare"; it only reports that paged optimizers gave the same speed as normal ones for 65B at batch size 16 on a 48 GB GPU (paper, 4).

### Putting it together: one QLoRA layer
Store the weight in NF4 with 64-weight blocks and double-quantized constants. At each use, convert it back to BFloat16, multiply, and add the adapter's output. Backward: gradients are computed for the adapters only, but they must pass through the dequantized weights (paper, 3). All experiments use NF4 with double quantization and BFloat16 computation; LoRA $r=64$ and $\alpha=16$ on all linear layers; a constant learning rate; max gradient norm 0.3; LoRA dropout 0.1 (up to 13B) or 0.05 (33B and 65B); for 33B and 65B the learning rate is halved and the batch size doubled (paper, Appendix B.2 and 5.1).

**Sizes of what is stored (paper's numbers and my arithmetic):**

| Model | Weights alone at 4.127 bits (mine) | Paper's memory column (Table 6) |
|---|---|---|
| 7B (6.7B parameters) | 3.5 GB | 5 GB |
| 13B | 6.7 GB | 10 GB |
| 33B (32.5B parameters) | 16.8 GB | 21 GB |
| 65B (65.2B parameters) | 33.6 GB | 41 GB |

Written by me, an AI. The last column is copied from the paper's Table 6; the paper does not say what the memory column includes (adapters, activations, the quantization constants of other layers and the embedding layers are candidates, my guess), so the difference is unexplained. **Column meaning:** *weights alone* is parameters times (4 + 0.127) bits divided by 8; *paper's memory* is what the authors report for the stored model.

## The equations
The paper numbers six equations; I add the size saving, which the text gives without a number, and the Elo formula the evaluation relies on (not printed in the paper).

### Absmax quantization (paper's equation 1) and dequantization (equation 2)

$$
\mathbf{X}^{\text{Int8}}=\operatorname{round}\!\left(\frac{127}{\operatorname{absmax}(\mathbf{X}^{\text{FP32}})}\,\mathbf{X}^{\text{FP32}}\right)=\operatorname{round}\big(c^{\text{FP32}}\cdot\mathbf{X}^{\text{FP32}}\big),\qquad \operatorname{dequant}\big(c,\mathbf{X}^{\text{Int8}}\big)=\frac{\mathbf{X}^{\text{Int8}}}{c}
$$

**Where:**
- $\mathbf{X}^{\text{FP32}}$ is the block of original 32-bit numbers and $\mathbf{X}^{\text{Int8}}$ its 8-bit integers (-127 to 127).
- $\operatorname{absmax}$ is the largest absolute value in the block. **Matters:** dividing by it fits the whole block into the integer range; without it either the biggest values overflow or the range is wasted.
- $c=127/\operatorname{absmax}$ is the quantization constant (scale). **Matters:** it must be stored with every block, which is the cost double quantization reduces.
- $\operatorname{round}$ drops the fraction. **Matters:** this is where the information is lost.

**Example (tiny numbers):** block $(0.5,-2.0,1.0)$: absmax 2, $c=63.5$, integers $(32,-127,64)$, dequantized $(0.5039,-2.0,1.0079)$. With an outlier, block $(0.5,-2.0,1.0,50)$: $c=2.54$, integers $(1,-5,3,127)$, dequantized $(0.394,-1.969,1.181,50)$. The answer means: one outlier ruins the precision of its neighbours, so blocks must be small.

### LoRA (paper's equation 3)

$$
\mathbf{Y}=\mathbf{X}\mathbf{W}+s\,\mathbf{X}\mathbf{L}_1\mathbf{L}_2
$$

**Where:**
- $\mathbf{X}$ is the input ($b\times h$), $\mathbf{W}$ the frozen weight ($h\times o$) and $\mathbf{Y}$ the output.
- $\mathbf{L}_1$ ($h\times r$) and $\mathbf{L}_2$ ($r\times o$) are the two small trainable matrices. **Matters:** their product is as big as $\mathbf{W}$ but has only $r(h+o)$ numbers; without them nothing would be trained.
- $s$ is a scale. **Matters:** it controls how strongly the add-on changes the output.

**Example (tiny numbers):** $\mathbf{X}=(1,2)$, $\mathbf{W}=(0.3,-0.1)^T$, $\mathbf{L}_1=(0.1,0.2)^T$, $\mathbf{L}_2=0.3$, $s=1$. $\mathbf{XW}=0.3-0.2=0.1$. $\mathbf{XL}_1=0.1+0.4=0.5$, times $\mathbf{L}_2$ is $0.15$. $\mathbf{Y}=0.1+0.15=0.25$. The answer means: the frozen layer says 0.1 and the adapter shifts it by 0.15.

### The NormalFloat levels (paper's equation 4)

$$
q_i=\frac12\Big(Q_X\!\Big(\frac{i}{2^k+1}\Big)+Q_X\!\Big(\frac{i+1}{2^k+1}\Big)\Big)
$$

**Where:**
- $k$ is the number of bits and $2^k$ the number of levels.
- $Q_X(\cdot)$ is the quantile function of the standard normal distribution: $Q_X(p)$ is the value below which a share $p$ of a normal population lies. **Matters:** using quantiles puts the same share of weights between neighbouring levels.
- $q_i$ is the $i$-th level: the middle of two neighbouring quantiles. **Matters:** the levels are then normalised to the range -1 to 1 (paper, 3).

**Example (tiny numbers):** $k=2$, so $2^k+1=5$. $Q_X(0.2)=-0.8416$, $Q_X(0.4)=-0.2533$, $Q_X(0.6)=0.2533$, $Q_X(0.8)=0.8416$. Then $q_1=\tfrac12(-0.8416-0.2533)=-0.5475$, $q_2=\tfrac12(-0.2533+0.2533)=0$ and $q_3=+0.5475$. The answer means: the levels bunch around zero where the bell is tall. **A problem with the printed formula:** at the ends it needs $Q_X(0)=-\infty$ or $Q_X(1)=+\infty$, so it cannot produce the paper's -1 and +1; the real NF4 recipe is the asymmetric one in the paper's text. I checked an implementation of that recipe, written from my memory of the library code, against the paper's Appendix E list; it matched all 16 values to $10^{-7}$. The recipe itself is not in the paper.

### One QLoRA layer (paper's equations 5 and 6)

$$
\begin{aligned}
\mathbf{Y}^{\text{BF16}}&=\mathbf{X}^{\text{BF16}}\,\operatorname{doubleDequant}\big(c_1^{\text{FP32}},c_2^{k\text{-bit}},\mathbf{W}^{\text{NF4}}\big)+\mathbf{X}^{\text{BF16}}\mathbf{L}_1^{\text{BF16}}\mathbf{L}_2^{\text{BF16}}\
\operatorname{doubleDequant}\big(c_1,c_2,\mathbf{W}\big)&=\operatorname{dequant}\big(\operatorname{dequant}(c_1,c_2),\mathbf{W}^{\text{4bit}}\big)=\mathbf{W}^{\text{BF16}}
\end{aligned}
$$

**Where:**
- $\mathbf{W}^{\text{NF4}}$ is the stored 4-bit weight and $\mathbf{W}^{\text{BF16}}$ what it becomes when used.
- $c_2^{k\text{-bit}}$ is the stored (quantized) per-block constant, and $c_1^{\text{FP32}}$ the second-level constant used to unpack it. **Matters:** two steps of unpacking: first the constants, then the weights.
- $\mathbf{X}^{\text{BF16}}$, $\mathbf{L}_1^{\text{BF16}}$, $\mathbf{L}_2^{\text{BF16}}$: input and the 16-bit adapters. **Matters:** only the adapters are trained; the 4-bit weight never changes. (The paper uses NF4 for $\mathbf{W}$ and 8-bit float for $c_2$, blocks of 64 and 256.)
- Backward: only $\partial E/\partial\mathbf{L}_i$ is needed, but computing it needs the weight unpacked to 16 bits to pass the error through (paper, 3).

**Example (tiny numbers):** take the block $w=(0.30,-0.12,0.05,-0.60)$. Divide by its absmax 0.60: $(0.5,-0.2,0.0833,-1.0)$. Nearest NF4 levels: $0.4407,-0.1848,0.0796,-1.0$. Dequantized: $(0.2644,-0.1109,0.0477,-0.6)$, errors $0.036, 0.009, 0.002, 0$. For $\mathbf{X}=(1,2,0,1)$ the layer computes $\mathbf{XW}=0.2644-0.2218+0+(-0.6)=-0.5573$ (exact weights would give $-0.54$), and with an adapter $\mathbf{X}\mathbf{L}_1=0.6$, times $\mathbf{L}_2=0.3$ is $0.18$, so $\mathbf{Y}=-0.3773$. The answer means: storing the weights in 4 bits costs 0.017 of error in this output, and the adapter can later learn to move the output wherever training says.

### What double quantization costs per weight (paper, section 3)

$$
\underbrace{\tfrac{32}{64}}_{=0.5}\ \longrightarrow\ \tfrac{8}{64}+\tfrac{32}{64\cdot256}=0.127\quad\text{bits per parameter}
$$

**Where:**
- 32 is the size of one constant in bits, and 64 the block size of the weights. **Matters:** a constant for every 64 weights costs 32/64 bits per weight.
- 8 is the bit width after quantizing the constants; 256 the block size of the second quantization. **Matters:** the second-level constants (32 bits each, one per 256 constants) add only $32/(64\cdot256)$.

**Example (tiny numbers):** $8/64=0.125$ and $32/16384=0.00195$, together $0.12695$, so the saving is $0.5-0.12695=0.373$ bits per weight. For a 65-billion-parameter model that is $65\cdot10^9\cdot0.373/8=3.03\cdot10^9$ bytes, 3 GB. The answer means: 3 GB saved on a model that otherwise needs about 33 GB for its 4-bit weights.

### Elo expected score (general knowledge, not printed in the paper)

$$
E_A=\frac{1}{1+10^{(R_B-R_A)/400}}
$$

**Where:**
- $R_A$ and $R_B$ are the Elo ratings of players $A$ and $B$.
- $E_A$ is the expected share of matches $A$ wins. **Matters:** it turns rating gaps into win chances; the paper uses it to rank chatbots from pairwise judgments, starting at 1,000 with $K=32$ and averaging over 10,000 random match orders (paper, 5.2).
- 400 is the scale constant of the usual Elo definition. **Matters:** a 400-point gap means 10 to 1 odds.

**Example (tiny numbers):** the paper says an Elo of 1100 against 1000 gives about 65%; the formula gives $1/(1+10^{-0.25})=0.640$. A 10-point gap near an even match shifts the chance by 1.44 points, which is the paper's "approximately 1.5% win-rate per 10 points" (Table 7 caption; my arithmetic).

## What they found
The paper has three kinds of results: memory, whether 4-bit fine-tuning matches 16-bit, and the Guanaco chatbots. Numbers are marked "abstract" or "paper" (read in the extracted text; I also looked at the page picture of Table 6).

- **Memory (abstract and paper):** QLoRA reduces the average memory needed to fine-tune a 65B model from more than 780 GB to under 48 GB, "without degrading the runtime or predictive performance compared to a 16-bit fully finetuned baseline" (paper, 1). The paper then states in its limitations that it "did not establish that QLoRA can match full 16-bit finetuning performance at 33B and 65B scales" (paper, 8). The text says full fine-tuning was compared up to 3B, yet Table 3 has a BF16 value at 11B (62.0); the paper does not say where it comes from.
- **Guanaco (abstract):** reaches 99.3% of ChatGPT's score on the Vicuna benchmark, with 24 hours of fine-tuning on a single GPU for the 65B model; the second-best (33B) reaches 97.8% in under 12 hours on a consumer GPU (paper, 1).
- **Data matters more than size:** a 9,000-example dataset (OASST1) beat a 450,000-example one (FLAN v2 subsampled) on chatbot quality (paper, 1).

### 4-bit data types (paper's Table 2 and Figure 3)

| Data type | Mean perplexity (Pile Common Crawl) | Against NF4 plus DQ (mine) |
|---|---|---|
| Int4 | 34.34 | +6.93 (+25%) |
| Float4 (E2M1) | 31.07 | +3.66 |
| Float4 (E3M0) | 29.48 | +2.07 |
| **NormalFloat4 with double quantization** | **27.41** | 0 |

Written by me, an AI, from the paper's Table 2 (all four rows). The last column is mine.
- **Data type:** how the 16 levels are spaced. E2M1 and E3M0 are two ways of splitting a 4-bit float into exponent and mantissa bits (general knowledge).
- **Mean perplexity:** the average, over OPT, BLOOM, LLaMA and Pythia models from 125M to 13B parameters, of the perplexity on text from the Pile Common Crawl (paper, 4); lower is better; see [[Perplexity]].
- **Against NF4 plus DQ (mine):** the difference to the best row; 6.93 as a share of 27.41 is 25% (6.93 as a share of 34.34 is 20%).

**Summary: what the table shows.** NF4 with double quantization is best (27.41); Int4 is worst (34.34), which is 6.93 higher. Float4 is in between and depends on its bit split (31.07 or 29.48). **Caution:** the paper says double quantization does not lose accuracy but this table cannot separate NF4 from double quantization, because only the combined row is shown; Figure 3 (above) does separate them and shows only a small difference.

**My own check (not in the paper).** I quantized 20,000 blocks of 64 random normal numbers with 15-level Int4, 15-level E2M1 float and NF4 (levels from Appendix E) and measured the round-trip error. The root-mean-square errors were 0.1077 (Int4), 0.1058 (Float4) and 0.0920 (NF4), and the mean absolute errors 0.0914, 0.0803 and 0.0728. So NF4 has about 15% lower RMS error than Int4 on bell-shaped data. This is one seed and made-up data, not trained weights; it only shows the mechanism. I also put one outlier of 40 among 4,096 normal numbers: quantizing the whole tensor with one scale left an error of 0.87 on the other numbers, while blocks of 64 left 0.139.

### Does 4-bit match 16-bit? (paper's Table 3: GLUE and Super-NaturalInstructions)

| Model | BF16 | BF16 replication | LoRA BF16 | QLoRA Int8 | QLoRA FP4 | QLoRA NF4 + DQ |
|---|---|---|---|---|---|---|
| RoBERTa-large (GLUE accuracy) | 88.6 | 88.6 | 88.8 | 88.8 | 88.6 | - |
| T5-80M (RougeL) | 40.1 | 40.0 | 40.5 | 40.4 | 40.3 | 40.4 |
| T5-250M | 42.1 | 42.2 | 42.6 | 42.9 | 42.4 | 42.7 |
| T5-780M | 48.0 | 47.3 | 47.1 | 45.4 | 47.5 | 47.7 |
| T5-3B | 54.3 | 54.9 | 55.4 | 56.5 | 55.6 | 55.3 |
| T5-11B | 62.0 | - | 60.7 | 60.7 | 60.9 | 60.9 |

Written by me, an AI, from the paper's Table 3 (all rows and columns; "-" is a blank in the paper, not a zero). The first column ("BF16") is full 16-bit fine-tuning; "BF16 replication" is the authors' re-run of it.
- **Model:** RoBERTa-large (encoder) on GLUE, measured in accuracy; T5 sizes from 80 million to 11 billion parameters on Super-NaturalInstructions, measured in RougeL, a text-overlap score (higher is better; general knowledge).
- **Columns:** full 16-bit fine-tuning, LoRA adapters on a 16-bit model, and QLoRA with the base model stored as Int8, FP4 or NF4 with double quantization.
- **What matters:** the paper's claim is that the QLoRA columns equal the BF16 column.

**Summary: what the table shows.** For most rows all columns are within about 1 point of each other (NF4 plus DQ is 0.3 and 0.6 above full fine-tuning at 80M and 250M, 0.3 below at 780M, 1.0 above at 3B; mine). The largest gaps are T5-780M with QLoRA Int8 (45.4, 2.6 below the best of the row) and T5-11B, where every adapter column (60.7 to 60.9) is 1.1 to 1.3 below full fine-tuning (62.0). **Caution:** single runs with no error bars, and the paper does not comment on the 11B gap or on the 780M Int8 dip; so "matches full fine-tuning" is well supported up to 3B, and at 11B only against LoRA, not against full fine-tuning.

### Does 4-bit match 16-bit LoRA at 7B to 65B? (paper's Table 4: MMLU)

| LLaMA size, dataset | BFloat16 | Float4 | NFloat4 + DQ |
|---|---|---|---|
| 7B Alpaca | 38.4 | 37.2 | 39.0 |
| 7B FLAN v2 | 45.6 | 44.0 | 44.5 |
| 13B Alpaca | 47.2 | 47.3 | 47.5 |
| 13B FLAN v2 | 50.6 | 50.0 | 50.7 |
| 33B Alpaca | 57.7 | 55.9 | 57.3 |
| 33B FLAN v2 | 60.5 | 58.5 | 59.2 |
| 65B Alpaca | 61.8 | 61.3 | 61.8 |
| 65B FLAN v2 | 62.5 | 63.3 | 63.9 |
| Mean (paper) | 53.0 | 52.2 | 53.1 |

Written by me, an AI, from the paper's Table 4 (all 8 columns of the paper's table, laid out as rows; the Mean row is the paper's). All cells are mean 5-shot MMLU accuracy in percent (see [[Benchmarks]]); the three data columns are the adapters' base-model type: 16-bit, Float4 and NF4 with double quantization.
- **Size and dataset:** the LLaMA size and the instruction dataset used to fine-tune (Alpaca or FLAN v2).

**Summary: what the table shows.** NF4 plus DQ is within 1.4 points of BFloat16 in every row (differences +0.6, -1.1, +0.3, +0.1, -0.4, -1.3, 0.0, +1.4; mine) and has the same mean (my mean of the eight cells is 52.99; the paper prints 53.1). Float4 is lower in six of the eight rows, by 0.5 to 2.0 points, with a mean about 0.85 below (mine). **Caution:** the paper says FP4 is "consistently one percentage point behind", but it is ahead in two rows (13B Alpaca +0.1, 65B FLAN v2 +0.8), so "on average about 0.85 behind" is the honest wording; single runs, and my sum of the NF4 row gives 52.99, not the printed 53.1.

### What the fine-tuning data does (paper's Table 5: MMLU)

| Dataset | 7B | 13B | 33B | 65B |
|---|---|---|---|---|
| LLaMA no tuning | 35.1 | 46.9 | 57.8 | 63.4 |
| Self-Instruct | 36.4 | 33.3 | 53.0 | 56.7 |
| Longform | 32.1 | 43.2 | 56.6 | 59.7 |
| Chip2 | 34.5 | 41.6 | 53.6 | 59.8 |
| HH-RLHF | 34.9 | 44.6 | 55.8 | 60.1 |
| Unnatural Instruct | 41.9 | 48.1 | 57.3 | 61.3 |
| Guanaco (OASST1) | 36.6 | 46.4 | 57.0 | 62.2 |
| Alpaca | 38.8 | 47.8 | 57.3 | 62.5 |
| **FLAN v2** | **44.5** | **51.4** | **59.2** | **63.9** |

Written by me, an AI, from the paper's Table 5 (all rows). Cells are 5-shot MMLU accuracy in percent after QLoRA fine-tuning of LLaMA on that dataset; the first row is the untuned model.
- **Dataset:** the instruction data used; FLAN v2 is a collection of academic tasks close in style to MMLU (paper, 8).

**Summary: what the table shows.** FLAN v2 is best at every size (44.5, 51.4, 59.2, 63.9) and Guanaco's OASST1 data is mid-table for MMLU. Compared with the untuned model, at 13B only FLAN v2 (+4.5), Unnatural Instruct (+1.2) and Alpaca (+0.9) help, at 33B only FLAN v2 (+1.4), and at 65B only FLAN v2 (+0.5); the other seven datasets are 0.9 to 6.7 points lower at 65B (mine). The gap between best and worst dataset is 12.4, 18.1, 6.2 and 7.2 points at the four sizes (mine). The worst cell is Self-Instruct at 13B (33.3, 13.6 below untuned). **Caution:** a possible odd value; the paper does not comment on it. The paper's conclusion: data suitability matters more than data size, and MMLU strength does not imply chatbot strength.

### Chatbot quality against ChatGPT (paper's Table 6, Vicuna benchmark scored by GPT-4)

| Model / dataset | Params | Bits | Memory | ChatGPT vs Sys | Sys vs ChatGPT | Mean | 95% CI |
|---|---|---|---|---|---|---|---|
| GPT-4 | - | - | - | 119.4% | 110.1% | 114.5% | 2.6% |
| Bard | - | - | - | 93.2% | 96.4% | 94.8% | 4.1% |
| **Guanaco** | 65B | 4-bit | 41 GB | 96.7% | 101.9% | **99.3%** | 4.4% |
| Alpaca | 65B | 4-bit | 41 GB | 63.0% | 77.9% | 70.7% | 4.3% |
| FLAN v2 | 65B | 4-bit | 41 GB | 37.0% | 59.6% | 48.4% | 4.6% |
| **Guanaco** | 33B | 4-bit | 21 GB | 96.5% | 99.2% | **97.8%** | 4.4% |
| Open Assistant | 33B | 16-bit | 66 GB | 91.2% | 98.7% | 94.9% | 4.5% |
| Alpaca | 33B | 4-bit | 21 GB | 67.2% | 79.7% | 73.6% | 4.2% |
| FLAN v2 | 33B | 4-bit | 21 GB | 26.3% | 49.7% | 38.0% | 3.9% |
| Vicuna | 13B | 16-bit | 26 GB | 91.2% | 98.7% | 94.9% | 4.5% |
| **Guanaco** | 13B | 4-bit | 10 GB | 87.3% | 93.4% | **90.4%** | 5.2% |
| Alpaca | 13B | 4-bit | 10 GB | 63.8% | 76.7% | 69.4% | 4.2% |
| HH-RLHF | 13B | 4-bit | 10 GB | 55.5% | 69.1% | 62.5% | 4.7% |
| Unnatural Instr. | 13B | 4-bit | 10 GB | 50.6% | 69.8% | 60.5% | 4.2% |
| Chip2 | 13B | 4-bit | 10 GB | 49.2% | 69.3% | 59.5% | 4.7% |
| Longform | 13B | 4-bit | 10 GB | 44.9% | 62.0% | 53.6% | 5.2% |
| Self-Instruct | 13B | 4-bit | 10 GB | 38.0% | 60.5% | 49.1% | 4.6% |
| FLAN v2 | 13B | 4-bit | 10 GB | 32.4% | 61.2% | 47.0% | 3.6% |
| **Guanaco** | 7B | 4-bit | 5 GB | 84.1% | 89.8% | **87.0%** | 5.4% |
| Alpaca | 7B | 4-bit | 5 GB | 57.3% | 71.2% | 64.4% | 5.0% |
| FLAN v2 | 7B | 4-bit | 5 GB | 33.3% | 56.1% | 44.8% | 4.0% |

Written by me, an AI, from the paper's Table 6 (all 21 rows, checked against the page picture; I added the bold marks). 
- **Model / dataset:** a model fine-tuned by QLoRA on that dataset; GPT-4 and Bard are commercial chatbots; Vicuna and Open Assistant are other chatbots.
- **Params, Bits and Memory:** size, how many bits per weight, and the memory the paper lists (the extra over weights alone is unexplained, see above).
- **ChatGPT vs Sys and Sys vs ChatGPT:** GPT-4 is shown the answers of ChatGPT and of the system in the two possible orders and scores both out of 10; the number is the system's share of ChatGPT's score. The two columns are the two orders (the paper found "a significant ordering effect": GPT-4 raises the score of whichever answer comes first).
- **Mean and 95% CI:** the average of the two orders and its confidence interval. **Matters:** the paper recommends reporting the mean over both orders.

**Summary: what the table shows.** After GPT-4 (114.5%), Guanaco 65B is best at 99.3%, then Guanaco 33B at 97.8%. Guanaco 33B is 2.9 points above Vicuna 13B (94.9) with 21 GB against 26 GB. The worst rows are FLAN v2 (38.0% to 48.4%). The order effect is large: the second column is higher than the first by 5 to 13 points for most rows and by 28.8 for FLAN v2 13B (mine). **Caution:** the 95% intervals are about ±4 to ±5 points, so Guanaco 65B (94.9 to 103.7) overlaps Open Assistant, Vicuna and Bard; the paper says as much ("very wide confidence intervals"). **Problems in the table:** (1) the Vicuna 13B row and the Open Assistant 33B row are identical in every cell (91.2, 98.7, 94.9, 4.5) in the page picture; the paper does not explain, and I cannot tell which is wrong; (2) for most rows the Mean is the average of the two columns before it, but not for Alpaca 13B (the average is 70.25, the printed mean 69.4); (3) the text says Guanaco 7B beats "a 26 GB Alpaca model by more than 20 percentage points" (section 1) and later "nearly 20 percentage points" higher than Alpaca 13B, but the table's Alpaca 13B (10 GB) row is 17.6 below Guanaco 7B, and it has no 26 GB Alpaca row.

### Chatbot Elo (paper's Table 1 and Table 7)

| Model | Size | Elo, GPT-4 judge (Table 1) |
|---|---|---|
| GPT-4 | - | 1348 ± 1 |
| Guanaco 65B | 41 GB | 1022 ± 1 |
| Guanaco 33B | 21 GB | 992 ± 1 |
| Vicuna 13B | 26 GB | 974 ± 1 |
| ChatGPT | - | 966 ± 1 |
| Guanaco 13B | 10 GB | 916 ± 1 |
| Bard | - | 902 ± 1 |
| Guanaco 7B | 6 GB | 879 ± 1 |

| Model | Vicuna, human raters: Elo (rank) | Vicuna, GPT-4: Elo (rank) | Open Assistant, GPT-4: Elo (rank) |
|---|---|---|---|
| GPT-4 | 1176 (1) | 1348 (1) | 1294 (1) |
| Guanaco-65B | 1023 (2) | 1022 (2) | 1008 (3) |
| Guanaco-33B | 1009 (4) | 992 (3) | 1002 (4) |
| ChatGPT-3.5 Turbo | 916 (7) | 966 (5) | 1015 (2) |
| Vicuna-13B | 984 (5) | 974 (4) | 936 (5) |
| Guanaco-13B | 975 (6) | 913 (6) | 885 (6) |
| Guanaco-7B | 1010 (3) | 879 (8) | 860 (7) |
| Bard | 909 (8) | 902 (7) | - |

Written by me, an AI, from the paper's Tables 1 and 7 (all rows; the Median Rank column of Table 7 is left out; "-" is blank in the paper). Table 7 uses 80 Vicuna prompts for the first two columns and 953 prompts from the OASST1 validation set for the third.
- **Elo:** a rating from pairwise matches in which a judge picks the better answer; the paper starts at 1,000, uses $K=32$ and averages 10,000 random match orders. A 10-point gap is about a 1.5% win-rate difference (paper). **Table 1** also shows the memory size of each model.
- **Rank:** position in that column.

**Summary: what the table shows.** GPT-4 is first in every column. Guanaco 65B is second by humans and by GPT-4 on the Vicuna prompts, and third on the 953-prompt benchmark, where ChatGPT is second (1015 against 1008). The Elo gap between GPT-4 and Guanaco 65B by human raters (1176 against 1023, 153 points) gives Guanaco about a 29% win chance, and 33B 28% (mine), which the paper rounds to "30%". **Caution:** the Guanaco-7B rank is 3 for humans but 8 for GPT-4 (a 131-point gap), the paper's own example of disagreement; human raters agree only moderately with each other (Fleiss kappa 0.42), and GPT-4 and humans agree weakly on single examples (kappa 0.25) and moderately across systems (Kendall tau 0.43, Spearman 0.55). GPT-4 also rates itself higher (1348 against 1176 from humans); the paper says this is "an additional 20% probability of winning", but I get a 16-point gap against Guanaco 65B (86.7% against 70.7%) and an 8-point gap against ChatGPT (90.0% against 81.7%), so I could not reproduce "20%". **A mismatch:** Guanaco 13B is 916 in Table 1 and 913 in Table 7, and 916 is the human-raters value for ChatGPT in Table 7; these look like a mix-up between two numbers. Also Guanaco 7B is listed as 6 GB in Table 1 but 5 GB in Table 6 and in the text.

### Bias on CrowS-Pairs (paper's Table 8)

| Category | LLaMA-65B | GPT-3 | OPT-175B | Guanaco-65B |
|---|---|---|---|---|
| Average | 66.6 | 67.2 | 69.5 | 43.5 |

Only the average row of the paper's Table 8 is copied (the table has nine categories: gender, religion, race, sexual orientation, age, nationality, disability, physical appearance and socioeconomic status; I left them out). A lower score means a lower likelihood of generating biased sequences. **What it shows:** Guanaco-65B is 23.1 points below its base model; the paper says fine-tuning on OASST1 seems to reduce bias, but it is "unclear if Guanaco does also well when assessed on other types of biases". **Caution:** a score below 50 on a test whose unbiased value is 50 is not obviously "better" (my reading; the paper does not discuss it), and it is a single test.

## Questions I had, answered

**Why not just quantize the whole tensor with one scale?**
Because one large value ruins the scale for all the others. My test: one outlier of 40 in 4,096 normal numbers left an error of 0.87 on the others with a single scale, against 0.139 with blocks of 64. The price is one constant per block (0.5 bits per weight), which is why double quantization exists.

**Why are NF4's levels not evenly spaced?**
Because weights are bell-shaped (the paper checks this with a Shapiro-Wilk test in Appendix F; I did not read it). Evenly spaced levels waste many levels in the thin tails. In my test on made-up normal numbers NF4 had about 15% lower RMS error than Int4. The paper's own evidence is Table 2 and Figure 3.

**Is the 4-bit model changed by training?**
No. The 4-bit weights are frozen; only the 16-bit adapters get updates. The gradient still has to pass back through the dequantized weights to reach the adapters (paper, 3).

**Where does "more than 780 GB" come from?**
The paper does not say. My reconstruction: 65 billion parameters at 12 bytes each (16-bit weight and gradient, and two 32-bit Adam moments) is 780 GB. It matches exactly, which suggests this is the count, but I have not seen it confirmed in the text.

**Why does quantizing and then adding adapters recover the quality?**
The paper's reading is that "the performance lost due to the imprecise quantization can be fully recovered through adapter finetuning after quantization" (paper, 4). The evidence is Tables 3 and 4: within about 1 point of 16-bit for models up to 3B and, on MMLU, for LLaMA 7B to 65B. The limits: no full-fine-tuning comparison beyond 11B, and at 11B the adapters are 1.1 to 1.3 below full fine-tuning in Table 3. In my own tiny example above, the quantization error of 0.017 in the output is small compared with what an adapter can shift (0.18), which is the idea, not a proof.

**What do the "paged" optimizers help with?**
They only matter when memory spikes, for example a long sequence under gradient checkpointing. The paper reports no speed cost for 65B at batch size 16 on a 48 GB GPU, but no other measurements.

**Why does the paper say benchmarks "are not trustworthy"?**
Because MMLU and the chatbot benchmarks disagree (FLAN v2 is best on MMLU in Table 5 and worst on the Vicuna benchmark in Table 6), and because GPT-4 and human judgments agree only moderately (paper, 5.3 and 8).

## Why it mattered
It made fine-tuning of 33B to 65B models possible on a single GPU, which the authors call an equalizer for small teams, and showed that quality was not lost by storing the base model in 4 bits at the sizes tested (paper, broader impacts and 4). It also gave a recipe (NF4, double quantization, paged optimizers, adapters on all layers). From memory (not checked in this paper): it was integrated into the Hugging Face and bitsandbytes tooling and became a standard way to fine-tune open models on limited hardware; see [[Unsloth]].

## Limits and open questions
- **Not shown at the largest sizes:** the authors did not establish that QLoRA matches full 16-bit fine-tuning at 33B and 65B (paper, 8).
- **Evidence for matching full fine-tuning is mixed at 11B:** the adapter columns are 1.1 to 1.3 below the full fine-tuning column in Table 3.
- **Evaluation limits:** only MMLU, the Vicuna benchmark and the OA benchmark. BigBench, RAFT and HELM were not tried. The authors say performance likely depends on how similar the fine-tuning data is to the benchmark (paper, 8).
- **Chatbot evaluation is shaky:** wide confidence intervals, moderate agreement among raters, an order effect and self-preference in GPT-4; Table 6 has duplicated and mismatched cells (see above).
- **Qualitative failures (paper, 6.1):** confident wrong answers on obscure facts (a wrong singer and year), a secret word revealed after a trick prompt, a wrong prime factorisation (1833 is called prime, though it is 3 times 17 times 43), and unreliable theory-of-mind answers.
- **Limited safety evaluation:** one bias test; nothing wider.
- **Choices not explored:** bit widths below 4 and adapter methods other than LoRA (paper, 8).
- **Speed:** the paper claims lower memory, not faster training.
- **Dual-use:** the authors note that fine-tuning can be abused for harm (paper, 9).

## Concepts used
[[QLoRA]], [[LoRA]], [[Quantization]], [[Fine-tuning]], [[Normal Distribution]], [[Instruction Tuning]], [[Gradient Checkpointing]], [[Mixed Precision Training]], [[LLM as a Judge]], [[Benchmarks]], [[Chatbot Arena]], [[GPU]], [[Cross-Entropy Loss]], [[Perplexity]], [[Adam Optimizer]], [[RLHF]]

## Reading trail
- **Came from:** [[LoRA 2021]] (the adapters it trains), [[LLaMA 2023]] (the model family it fine-tunes) and [[GPT-3 2020]] (the scale of model that motivates it).
- **Led to:** no paper in this vault yet. Its Guanaco models and evaluation methods are used in later work (from memory, not checked).

## How much I read
I read the whole main text as extracted text (pages 1 to 16): abstract, introduction, background, method, experiments, chatbot evaluation, qualitative analysis, considerations, related work, limitations and broader impacts. Of the appendix I read the experimental setup, the hyperparameters and ablation tables (Appendix A and B), the exact NF4 values (Appendix E) and the memory-footprint caption (Appendix G, page 26). I did not read the full qualitative examples after the first few, the Shapiro-Wilk test in Appendix F, the human-evaluation form, the pairwise GPT-4 tables (Tables 12 and 13) or the rest of the extra results. I looked at the page pictures of Table 6 (page 10) and of pages 3 and 6, and at the cropped Figures 1, 2 and 3. Figures 5 and 6 (the annotation form and the memory breakdown) were not looked at as pictures. I did not run QLoRA; I ran only my own small checks (quantization examples, the data-type error test, Elo arithmetic). Numbers from the paper are marked "paper" or "abstract"; sums and tiny examples are mine. Not opened in Obsidian, so the flashcards are unreviewed.

## Sources
- Dettmers, Pagnoni, Holtzman & Zettlemoyer (2023), "QLoRA: Efficient Finetuning of Quantized LLMs", arXiv 2305.14314 (v1, 23 May 2023): <https://arxiv.org/abs/2305.14314>. Licence on the arXiv page: Creative Commons Attribution 4.0 (CC BY 4.0). Figures 1, 2 and 3 were cropped from the PDF with PyMuPDF at 220 dpi and are credited under each figure.
- The NF4 construction recipe I used for the check (not in the paper) is from my memory of the bitsandbytes code, verified only against the paper's Appendix E list.
- Elo formula: general knowledge.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-llm QLoRA (Dettmers, Pagnoni, Holtzman and Zettlemoyer, 2023, arXiv 2305.14314): what problem did it attack?::Fine-tuning a very large model is prohibitively expensive (regular 16-bit fine-tuning of LLaMA 65B needs more than 780 GB of GPU memory) and quantization had only worked for inference, breaking down during training.

#flashcards/research/concept #flashcards/topic/research-llm What did people do before QLoRA?::Full 16-bit fine-tuning (the 780 GB); LoRA adapters on a 16-bit frozen model (small trainable part but a large frozen model); and quantization for inference, which for 4 bits degraded quality relative to 16-bit. SwitchBack layers were the only earlier work back-propagating through quantized weights beyond 1B parameters.

#flashcards/research/concept #flashcards/topic/research-llm QLoRA: what is the idea?::Store the frozen base model in 4 bits (NF4), dequantize to BFloat16 when a layer is used, and back-propagate through it into 16-bit LoRA adapters on every linear layer; add double quantization and paged optimizers to save memory.

#flashcards/research/concept #flashcards/topic/research-llm Where does the 780 GB for full fine-tuning of a 65B model come from?::The paper does not say. My reconstruction: 65 billion parameters × 12 bytes (2 for the 16-bit weight, 2 for the gradient, 8 for two 32-bit Adam moments) = 780 GB.

#flashcards/research/concept #flashcards/topic/research-llm Why does block-wise quantization use blocks and not one scale for a whole tensor?::One outlier ruins the scale for all other numbers. In my test, one outlier of 40 among 4,096 normal numbers left an error of 0.87 on the others with a single scale and 0.139 with blocks of 64. The price is one constant per block.

#flashcards/research/concept #flashcards/topic/research-llm Write absmax quantization and dequantization (paper's equations 1 and 2) and explain each symbol.
?
$$\mathbf{X}^{\text{Int8}}=\operatorname{round}\!\left(\frac{127}{\operatorname{absmax}(\mathbf{X}^{\text{FP32}})}\,\mathbf{X}^{\text{FP32}}\right)=\operatorname{round}\big(c\cdot\mathbf{X}^{\text{FP32}}\big),\qquad \operatorname{dequant}\big(c,\mathbf{X}^{\text{Int8}}\big)=\frac{\mathbf{X}^{\text{Int8}}}{c}$$
- $\mathbf{X}^{\text{FP32}}$: the block of original numbers; $\mathbf{X}^{\text{Int8}}$: its integers from -127 to 127.
- $\operatorname{absmax}$: the largest absolute value in the block; it fits the block into the integer range.
- $c=127/\operatorname{absmax}$: the quantization constant, stored once per block (the cost double quantization reduces).
- $\operatorname{round}$: where information is lost.

#flashcards/research/concept #flashcards/topic/research-llm Absmax example: block $(0.5,-2.0,1.0)$ and the same block with an outlier 50 added. What are the dequantized values?::Without the outlier: $c=63.5$, integers $(32,-127,64)$, back $(0.5039,-2.0,1.0079)$. With it: $c=2.54$, integers $(1,-5,3,127)$, back $(0.394,-1.969,1.181,50)$. One outlier ruins its neighbours (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Why is there a "127" in absmax quantization?::It is the largest 8-bit integer magnitude in the paper's range of -127 to 127, so dividing by the absmax and multiplying by 127 maps the biggest number exactly to 127. With absmax 2 the scale is 127/2 = 63.5; without the 127 the numbers would round to 0 or ±1.

#flashcards/research/concept #flashcards/topic/research-llm Write LoRA's output (the paper's equation 3) and explain each symbol.
?
$$\mathbf{Y}=\mathbf{X}\mathbf{W}+s\,\mathbf{X}\mathbf{L}_1\mathbf{L}_2$$
- $\mathbf{X}$: the input; $\mathbf{W}$: the frozen weight; $\mathbf{Y}$: the output.
- $\mathbf{L}_1$ ($h\times r$) and $\mathbf{L}_2$ ($r\times o$): the two small trainable matrices; their product has the size of $\mathbf{W}$ with only $r(h+o)$ numbers.
- $s$: a scale that controls how strongly the add-on changes the output.

#flashcards/research/concept #flashcards/topic/research-llm LoRA example: $\mathbf{X}=(1,2)$, $\mathbf{W}=(0.3,-0.1)^T$, $\mathbf{L}_1=(0.1,0.2)^T$, $\mathbf{L}_2=0.3$, $s=1$. What is $\mathbf{Y}$?::$\mathbf{XW}=0.1$; $\mathbf{XL}_1=0.5$, times 0.3 is 0.15; $\mathbf{Y}=0.25$. The adapter shifts the frozen layer's output by 0.15 (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Why does the LoRA product $\mathbf{L}_1\mathbf{L}_2$ save parameters, and by how much in a tiny case?::It has $r(h+o)$ numbers instead of $h\times o$. For $h=o=4096$ and $r=64$ that is 524,288 against 16,777,216, a factor of 32 (my arithmetic). Without the factorisation each adapter would be as big as the layer.

#flashcards/research/concept #flashcards/topic/research-llm How much memory do the adapters need compared with the rest, for LLaMA 7B?::Adapters about 0.2% of the weights take 26 MB; the input gradients take 567 MB (18 MB per sequence with gradient checkpointing); the 4-bit base model takes 5,048 MB. So shrinking the adapters saves little and more adapters can be used.

#flashcards/research/concept #flashcards/topic/research-llm Which layers should QLoRA put adapters on?::All linear layers of the transformer blocks. Adapters on only the query and value matrices (the usual LoRA choice) did not match 16-bit fine-tuning for large models; the rank $r$ did not matter (paper's Figure 2).

#flashcards/research/concept #flashcards/topic/research-llm What is 4-bit NormalFloat (NF4) and why is it better than evenly spaced 4-bit levels?::16 levels at quantiles of a normal distribution, so each level holds about the same number of weights, with an exact zero. Weights are bell-shaped, so levels bunch near zero where most weights are. Paper: perplexity 27.41 against 34.34 for Int4. My test on made-up normal data: RMS error 0.0920 against 0.1077.

#flashcards/research/concept #flashcards/topic/research-llm How many negative, zero and positive levels does NF4 have, and why?::Seven negative, one zero and eight positive (16 in all, from the paper's Appendix E). The recipe takes 8 negative-side and 9 positive-side quantiles, each including 0, and drops the duplicate zero, so an exact zero exists and all 16 codes are used.

#flashcards/research/concept #flashcards/topic/research-llm Write the NormalFloat level formula (paper's equation 4) and explain each symbol.
?
$$q_i=\frac12\Big(Q_X\!\Big(\frac{i}{2^k+1}\Big)+Q_X\!\Big(\frac{i+1}{2^k+1}\Big)\Big)$$
- $k$: number of bits; $2^k$ levels.
- $Q_X$: the quantile function of the standard normal; $Q_X(p)$ is the value below which a share $p$ of a normal population lies, so equal shares sit between neighbouring levels.
- $q_i$: the middle of two neighbouring quantiles; the levels are then scaled to -1 to 1.

#flashcards/research/concept #flashcards/topic/research-llm NF level example for $k=2$: what are $q_1,q_2,q_3$?::With $2^k+1=5$: $q_1=\tfrac12(Q(0.2)+Q(0.4))=\tfrac12(-0.8416-0.2533)=-0.5475$; $q_2=0$; $q_3=+0.5475$ (my arithmetic). The end levels would need $Q(0)=-\infty$ or $Q(1)=\infty$, so the printed formula cannot give the paper's -1 and 1.

#flashcards/research/concept #flashcards/topic/research-llm Why does the printed NormalFloat formula fail at the ends, and what does the real recipe do?::At $i=0$ it needs $Q_X(0)=-\infty$ and at $i=2^k$ it needs $Q_X(1)=+\infty$. The real NF4 values (Appendix E) include exactly -1 and +1. My check: an asymmetric recipe from memory of the library code, with an offset near 0.9677, reproduces all 16 Appendix E values to $10^{-7}$; that recipe is not in the paper.

#flashcards/research/concept #flashcards/topic/research-llm What does double quantization do and how much does it save?::It quantizes the per-block constants (8-bit floats, blocks of 256). The constants' cost falls from 32/64 = 0.5 bits per weight to 8/64 + 32/(64·256) = 0.127 bits, saving 0.373 bits, about 3 GB for a 65B model.

#flashcards/research/concept #flashcards/topic/research-llm Write the per-weight cost of the quantization constants with and without double quantization and explain each number.
?
$$\tfrac{32}{64}=0.5\ \longrightarrow\ \tfrac{8}{64}+\tfrac{32}{64\cdot256}=0.127\quad\text{bits per parameter}$$
- 32: bits of one FP32 constant; 64: the weight block size, so one constant per 64 weights.
- 8: bits after quantizing the constants; 256: the block size of the second quantization.
- $32/(64\cdot256)$: the second-level constants, only 0.002 bits per weight.

#flashcards/research/concept #flashcards/topic/research-llm Double quantization example: what is the saving per weight and for 65 billion weights?::$0.5-(0.125+0.00195)=0.373$ bits per weight; $65\cdot10^9\times0.373/8\approx3.03\cdot10^9$ bytes, 3 GB (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Why is the mean subtracted before the second quantization?::The block constants are all positive, so subtracting their mean centres them around zero and allows symmetric quantization (paper, 3). In my tiny example, constants 0.60, 0.55, 0.72, 0.64 (mean 0.6275) are stored as the integers -38, -106, 127, 17 and recovered with errors under 0.0004 (8-bit integers in place of the paper's 8-bit floats).

#flashcards/research/concept #flashcards/topic/research-llm What are paged optimizers and how do we know they help?::They place optimizer state in NVIDIA unified memory that moves between GPU and CPU automatically when the GPU runs out of memory during a long-sequence spike. The paper gives no hard measurements ("paging only occurs ... which is rare"); it only reports the same speed for 65B at batch size 16 on a 48 GB GPU.

#flashcards/research/concept #flashcards/topic/research-llm Write one QLoRA layer (the paper's equations 5 and 6) and explain each symbol.
?
$$\mathbf{Y}^{\text{BF16}}=\mathbf{X}^{\text{BF16}}\,\operatorname{doubleDequant}\big(c_1,c_2,\mathbf{W}^{\text{NF4}}\big)+\mathbf{X}^{\text{BF16}}\mathbf{L}_1^{\text{BF16}}\mathbf{L}_2^{\text{BF16}},\qquad \operatorname{doubleDequant}=\operatorname{dequant}\big(\operatorname{dequant}(c_1,c_2),\mathbf{W}^{\text{4bit}}\big)$$
- $\mathbf{W}^{\text{NF4}}$: the stored 4-bit weight; dequantizing gives $\mathbf{W}^{\text{BF16}}$.
- $c_2$: the stored per-block constants (quantized); $c_1$: the second-level constant that unpacks them.
- $\mathbf{L}_1,\mathbf{L}_2$: the 16-bit adapters, the only trained part. The 4-bit weight never changes.

#flashcards/research/concept #flashcards/topic/research-llm QLoRA layer example: block $(0.30,-0.12,0.05,-0.60)$ in NF4 and input $(1,2,0,1)$. What is the output without and with an adapter that contributes 0.18?::NF4 levels $0.4407,-0.1848,0.0796,-1.0$ times 0.6 give $(0.2644,-0.1109,0.0477,-0.6)$ (errors 0.036, 0.009, 0.002, 0). $\mathbf{XW}=-0.5573$ (exact $-0.54$); with the adapter's 0.18, $\mathbf{Y}=-0.3773$ (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Why must QLoRA dequantize to BFloat16 and not compute in 4 bits?::There is one storage type (4-bit) and one computation type (BFloat16): the matrix multiplication and the gradient with respect to the layer's input run in 16 bits, since the adapters' gradients need the unpacked weights. Only adapter weights get updates (paper, 3).

#flashcards/research/concept #flashcards/topic/research-llm ![[qlora-fig1-memory-methods.png]]
What does the paper's Figure 1 show?
?
Three ways to fine-tune: full fine-tuning (32-bit optimizer state for a 16-bit transformer), LoRA (small optimizer state for small 16-bit adapters) and QLoRA (the same with a 4-bit transformer, with the optimizer state paged out to the CPU). Blue arrows are parameter updates, green gradient flow. (Cropped from the paper, CC BY 4.0.)

#flashcards/research/concept #flashcards/topic/research-llm ![[qlora-fig2-rougel-adapters.png]]
What does the paper's Figure 2 show about where to put LoRA adapters?
?
RougeL on Alpaca for LLaMA 7B, one dot per seed: QLoRA with adapters on all layers (about 64.5) matches 16-bit fine-tuning (about 64.3), while adapters only on the feed-forward layers (about 63.9) or only attention (about 63.4) fall short; Stanford-Alpaca's original baseline is far lower (about 59.7 and 61.2). Values read by eye. (Cropped from the paper, CC BY 4.0.)

#flashcards/research/concept #flashcards/topic/research-llm ![[qlora-fig3-4bit-datatypes.png]]
What does the paper's Figure 3 show about 4-bit data types?
?
Mean zero-shot accuracy of 4-bit LLaMA against total model bits for Float, NormalFloat and NormalFloat plus double quantization. The NormalFloat lines are above the Float line at every size; double quantization (green) gives only a small gain in accuracy but fewer bits. (Cropped from the paper, CC BY 4.0.)

#flashcards/research/concept #flashcards/topic/research-llm How did the 4-bit data types compare in Table 2, and what is the caution?::Mean perplexity: Int4 34.34, Float4 E2M1 31.07, Float4 E3M0 29.48, NormalFloat4 with double quantization 27.41 (best). Int4 is 6.93 above (25% of 27.41, mine). Caution: only the combined NF4 plus DQ row is shown, so the table cannot separate the two; Figure 3 does.

#flashcards/research/concept #flashcards/topic/research-llm Did QLoRA match 16-bit fine-tuning in Table 3?::For most models within about 1 point, up to T5-3B (NF4 plus DQ is 1.0 above full fine-tuning at 3B). At T5-11B all adapter columns (60.7 to 60.9) are 1.1 to 1.3 below full fine-tuning (62.0), and QLoRA Int8 dips to 45.4 at 780M. Single runs; the paper does not comment on those.

#flashcards/research/concept #flashcards/topic/research-llm How did NF4 plus DQ compare with BFloat16 and Float4 in Table 4 (MMLU)?::NF4 plus DQ is within 1.4 points of BFloat16 in every one of 8 settings, with a paper mean of 53.1 against 53.0 (my sum of the NF4 cells is 52.99). Float4 is lower in six of eight rows and about 0.85 lower on average; the paper's "consistently one point behind" is not exact (FP4 is ahead in two rows).

#flashcards/research/concept #flashcards/topic/research-llm What did Table 5 show about fine-tuning data?::FLAN v2 is best on MMLU at every size (44.5, 51.4, 59.2, 63.9). At 65B every dataset except FLAN v2 lowered MMLU below the untuned model (63.4). The best-worst gap is up to 18.1 points (13B). Data suitability matters more than size; strong MMLU does not mean a strong chatbot.

#flashcards/research/concept #flashcards/topic/research-llm How did Guanaco do in Table 6 and what is the caution?::After GPT-4 (114.5%), Guanaco 65B scored 99.3% of ChatGPT and 33B 97.8%, 2.9 above Vicuna 13B with less memory (21 against 26 GB). Caution: 95% intervals of ±4 to 5 points overlap Open Assistant, Vicuna and Bard, and GPT-4 favours whichever answer comes first (5 to 29 points).

#flashcards/research/concept #flashcards/topic/research-llm Where do the numbers in the QLoRA paper's Table 6 and text disagree?::The Vicuna 13B and Open Assistant 33B rows are identical in every cell (91.2, 98.7, 94.9, 4.5); the printed mean for Alpaca 13B (69.4) is not the average of its two columns (70.25); the text claims Guanaco 7B beats a 26 GB Alpaca by more than 20 points but the table's Alpaca 13B is 17.6 below it. Guanaco 13B is 916 in Table 1 and 913 in Table 7.

#flashcards/research/concept #flashcards/topic/research-llm What does the Elo comparison say about GPT-4 against Guanaco, and what can't I reproduce?::By human Elo (1176 against 1023) Guanaco 65B has about a 29% chance against GPT-4, 33B about 28%, which the paper rounds to 30%. The paper says GPT-4 judges itself an "additional 20%" more likely to win than humans do, but I get 16 points against Guanaco 65B and 8 against ChatGPT.

#flashcards/research/concept #flashcards/topic/research-llm Write the Elo expected-score formula and explain each symbol.
?
$$E_A=\frac{1}{1+10^{(R_B-R_A)/400}}$$
- $R_A$, $R_B$: the Elo ratings of players $A$ and $B$.
- $E_A$: the expected share of matches $A$ wins; it turns rating gaps into win chances.
- 400: the scale constant: a 400-point gap means 10 to 1 odds. (General knowledge; the paper prints no formula.)

#flashcards/research/concept #flashcards/topic/research-llm Elo example: what is the expected win rate at 1100 against 1000, and the shift for a 10-point gap near an even match?::$1/(1+10^{-0.25})=0.640$ (the paper says about 65%); a 10-point gap near 50% shifts the chance by 1.44 points, the paper's "about 1.5%" (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Why does the paper say chatbot benchmarks are not trustworthy?::MMLU and chatbot scores disagree (FLAN v2 is best on MMLU and worst on Vicuna), human raters agree only moderately (Fleiss kappa 0.42), GPT-4 and humans agree weakly per example (0.25) and moderately per system (Kendall tau 0.43), and GPT-4 favours the first answer and its own outputs.

#flashcards/research/concept #flashcards/topic/research-llm What did the CrowS-Pairs bias test show for Guanaco-65B?::An average of 43.5 against 66.6 for its base LLaMA-65B, GPT-3 67.2 and OPT-175B 69.5 (a 23.1-point drop); fine-tuning on OASST1 seems to reduce bias, but it is one test and the paper says it is unclear for other biases.

#flashcards/research/concept #flashcards/topic/research-llm What limits does the QLoRA paper state or show?::Not shown to match full fine-tuning at 33B and 65B; mixed at T5-11B in Table 3; only MMLU, Vicuna and OA benchmarks; shaky chatbot evaluation; confident wrong answers, a leaked secret word and a wrong factorisation; one bias test; no bit widths below 4 and no other adapter methods; it claims lower memory, not faster training.

#flashcards/research/concept #flashcards/topic/research-llm How much of the QLoRA paper did the note read?::The whole main text (pages 1 to 16) and, of the appendix, the setup, hyperparameters, ablation tables, NF4 values and the memory caption; not the extra qualitative examples, the normality test, the annotation form or Tables 12 and 13. I looked at page pictures of Table 6 and the three cropped figures; I ran only my own small checks.
