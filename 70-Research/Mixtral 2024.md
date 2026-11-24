---
tags: [paper, NLP, LLM, mixture-of-experts, sparse]
link: https://arxiv.org/abs/2401.04088
authors: Jiang et al. (Albert Q. Jiang first of 26 authors, Mistral AI; from the paper's first page)
year: 2024
status: body-read
field: large language models
---

# Mixtral 2024

> **In one line for a 12-year-old:** Instead of one big brain doing all the work, each layer of the model has eight smaller specialists, and for every word a little manager picks the best two. So the model knows as much as a very big one but only does the work of a small one.

Paper: "Mixtral of Experts", Albert Q. Jiang and 25 co-authors (Mistral AI), arXiv 2401.04088, submitted 8 January 2024 (arXiv page and the first page). The arXiv licence is **CC BY 4.0**, so the figures below are cropped from the paper itself. Concept note: [[Mixture of Experts]]. Reference note for this layout: [[Attention Is All You Need 2017]].

## The problem
Bigger models are better but cost more to run, since every parameter is used for every word. Can a model have many parameters (to hold knowledge) while using only a few of them per word (to stay cheap and fast)? The paper says the technique "increases the number of parameters of a model while controlling cost and latency, as the model only uses a fraction of the total set of parameters per token" (paper, 1).

## What people did before
- **Dense models,** such as [[LLaMA 2023]]-style Llama 2 70B and GPT-3.5, which the paper compares with. In a dense model every parameter works for every token.
- **Mixture-of-experts layers:** Shazeer and others' sparsely-gated layer (the paper's reference 28, which it uses for the gate), and GShard (reference 21). The paper says its formulation is similar to GShard, except that Mixtral replaces *all* feed-forward blocks by expert layers (GShard replaces every other block) and GShard uses "a more elaborate gating strategy for the second expert" (paper, 2.1).
- **Mistral 7B** (reference 18, not read): the dense model that Mixtral reuses almost entirely.

## The idea
Keep Mistral 7B's architecture, but replace each layer's single feed-forward block by **8 feed-forward blocks (experts)** and add a small **router**. For every token, at every layer, the router picks the 2 best experts, runs only those, and adds their outputs with weights from the router. As a result "each token has access to 47B parameters, but only uses 13B active parameters during inference" (abstract). It was trained with a 32,000-token context and released with open weights under the Apache 2.0 licence (abstract).

## The architecture, part by part
Mixtral is a decoder-only [[Transformer]] "based on a transformer architecture and uses the same modifications as described in [18]" (Mistral 7B), with two exceptions: it supports "a fully dense context length of 32k tokens", and its feed-forward blocks are replaced by mixture-of-expert layers (paper, 2). The sizes are in the paper's Table 1.

![[mixtral-fig1-moe-layer.png]]
*The paper's own Figure 1, cropped from the PDF: "Mixture of Experts Layer". An input goes to a router, which sends it to 2 of the 8 experts; the layer's output is the weighted sum of the two selected experts' outputs (the circled plus), with the weights called "gating weights" in the figure. Authors: Jiang and others (Mistral AI), Figure 1 of arXiv 2401.04088, licensed CC BY 4.0; cropped from the paper and not redrawn. The figure draws only 3 stacked expert boxes to suggest many; the paper has 8.*

### The model sizes (paper's Table 1)

| Parameter | Value | Meaning |
|---|---|---|
| `dim` | 4096 | width of the token vector |
| `n_layers` | 32 | number of transformer blocks |
| `head_dim` | 128 | size of one attention head |
| `hidden_dim` | 14336 | inside width of a feed-forward block |
| `n_heads` | 32 | attention heads for queries |
| `n_kv_heads` | 8 | attention heads for keys and values |
| `context_len` | 32768 | longest text it reads at once |
| `vocab_size` | 32000 | number of different tokens |
| `num_experts` | 8 | experts per layer |
| `top_k_experts` | 2 | experts used per token |

Written by me, an AI, from the paper's Table 1 (all ten rows); the Meaning column is mine, partly general knowledge about transformers.
- **Parameter and Value:** the paper's setting names and numbers. **Matters:** they let one recompute the size of the model (next sub-section).
- **`n_heads` 32 and `n_kv_heads` 8:** four query heads share each key and value head, which is [[Grouped-Query Attention]] (my reading; the paper only lists the two numbers). It shrinks the [[KV Cache]] to a quarter (general knowledge).
- **`num_experts` and `top_k_experts`:** 8 experts, 2 used. **Matters:** these two numbers set the gap between total and active parameters.

**Summary: what the table shows.** It is the same as the dense Mistral 7B except for the two bottom rows. **Caution:** the table gives no total parameter count; the paper states 47B and 13B in the text only.

### The parts of one layer
- **Attention sub-block (shared, dense).**
  - *What it does:* ordinary multi-head attention with 32 query heads and 8 key and value heads, over a context of 32,768 tokens (Table 1). Every token uses it fully.
  - *Why it matters:* only the feed-forward block is split into experts; the attention weights are not duplicated. That is why the whole model is 47B and not 8 times 7B = 56B (see "Questions").
- **Router (gate).**
  - *What it does:* a single linear layer, matrix $W_g$, that turns the token vector (4,096 numbers) into 8 scores, one per expert; the top 2 scores are kept (paper, 2.1).
  - *Why it matters:* it decides which experts work for this token. It is tiny: 4,096 times 8 = 32,768 numbers per layer (my arithmetic).
- **Experts (8 feed-forward blocks per layer).**
  - *What it does:* each expert is a SwiGLU feed-forward block with its own weights, hidden width 14,336 (paper, 2.1 and Table 1; the SwiGLU formula is in my [[LLaMA 2023]] note).
  - *Why it matters:* the experts hold most of the model: about 96.6% of all parameters (my arithmetic, below). Using only 2 of 8 per token is what saves compute.
- **Weighted sum.**
  - *What it does:* the two chosen experts' outputs are added, each multiplied by its router weight (Figure 1 and paper, 2.1).
  - *Why it matters:* the router weights add up to 1 for the two chosen experts, so the result stays on the scale of a single expert's output (mine; see the example below).
- **Why 8 experts and top 2, and not other numbers?** The paper does not say and shows no experiment on it. The only statement is that $K$ (experts per token) "modulates the amount of compute used to process each token", and that increasing $n$ at fixed $K$ increases the parameter count while keeping compute "effectively constant" (paper, 2.1). The routing analysis (Section 5) is its only look at what the experts do.

### Size check: where the 47B and 13B come from (paper's Table 1, my arithmetic)

| Part | Rule | Per layer (millions) |
|---|---|---|
| Attention | 4096 by 4096 (queries) + 4096 by 1024 (keys) + 4096 by 1024 (values) + 4096 by 4096 (output) | 41.94 |
| One expert | 3 matrices of 4096 by 14336 | 176.16 |
| 8 experts | 8 times one expert | 1409.29 |
| Router | 4096 by 8 | 0.03 |
| **Total per layer (all experts)** | | **1451.26** |
| **Active per layer (2 experts)** | 41.94 + 2 times 176.16 + 0.03 | **394.30** |

Over 32 layers plus 2 tables of 32,000 by 4,096 (input and output embeddings, 262.1 million): total **46.70 billion**, active **12.88 billion**. This matches the paper's "47B" and "13B" (abstract). Written by me from the paper's Table 1; I assume no bias terms, untied embeddings, and I ignore the small normalisation weights; the paper does not give this breakdown.
- **Rule:** how I counted. The three matrices of an expert are the SwiGLU's three (two to widen, one to narrow back; general knowledge, see [[LLaMA 2023]]); $q$, $k$, $v$ and output sizes come from 32 heads and 8 key and value heads of 128 numbers.
- **Per layer (millions):** learned numbers in that part of one layer. **Matters:** it shows that 8 experts hold 1409 of 1451 million per layer, and that an active layer needs only 394 million.
- A dense model with the same sizes and a single expert would have 7.24 billion parameters (my arithmetic), which is about the size of Mistral 7B.

**Summary: what the table shows.** The experts make up 96.6% of the model, so adding experts is what makes the model large. Per token only 12.88 of 46.70 billion are used (27.6%). **Caution:** the memory needed is proportional to 47B, not 13B (paper, 3, "Size and Efficiency").

## The core mechanism, step by step

### One token through a mixture-of-experts layer
The idea: a switchboard operator (the router) looks at each token and connects it to the 2 most suitable of 8 specialists.

1. **Attention first, as usual.** *What it does:* the layer's attention sub-block mixes information between tokens (not described in detail in the paper). *Why it matters:* every token still sees the whole context; only the feed-forward part is sparse.
2. **Score the experts.** *What it does:* multiply the token vector $x$ by $W_g$ to get 8 numbers (logits), one per expert. *Why it matters:* a higher score means "this expert suits this token".
3. **Keep the top 2.** *What it does:* set the other six scores to minus infinity (TopK). *Why it matters:* it makes the gate sparse, so the other experts need not be computed at all (paper, 2.1).
4. **Softmax over the two.** *What it does:* turn the two kept scores into weights that add up to 1; minus infinity becomes weight 0. *Why it matters:* a clean weighted average.
5. **Run only the chosen experts.** *What it does:* feed $x$ to expert $i$ for the two chosen $i$. *Why it matters:* this is the whole saving: 2 of 8 feed-forward blocks per token.
6. **Add up.** *What it does:* multiply each expert output by its weight and add. *Why it matters:* this is the layer's output for the token (then the usual transformer residual connection follows, general knowledge, not printed in the paper).

**Worked example (my own tiny numbers, 4 experts instead of 8, $K=2$).** Router scores (logits) $(2.0,\ 1.0,\ 0.5,\ -1.0)$. Top 2: experts 0 and 1, so the scores become $(2.0,\ 1.0,\ -\infty,\ -\infty)$. Softmax: $e^{2.0}=7.389$, $e^{1.0}=2.718$, sum 10.107, so the weights are $(0.7311,\ 0.2689,\ 0,\ 0)$. If the experts' outputs were the single numbers 10 and 20, the layer outputs $0.7311\times10+0.2689\times20=12.689$. For contrast, a softmax over all four scores *before* picking would give $(0.6095,\ 0.2242,\ 0.1360,\ 0.0303)$; keeping the top two would leave weights summing to only 0.8337.

**Shapes (paper's sizes):**

| Thing | Shape |
|---|---|
| Token vector $x$ | 4096 |
| Router matrix $W_g$ | 4096 by 8 |
| Router scores | 8 |
| Chosen experts and weights | 2 indices, 2 numbers |
| Inside one expert | 4096 to 14336 to 4096 |
| Layer output | 4096 |

### Running it on real hardware
The paper says an expert layer can be run quickly on one GPU with specialised kernels: Megablocks turns the feed-forward operation into large sparse matrix multiplications and "naturally" handles experts that get different numbers of tokens (paper, 2.1). Across several GPUs, the layer can use standard model parallelism or expert parallelism, where tokens go to the GPU holding their expert and the result comes back; the catch is load balancing, because an overloaded GPU becomes a bottleneck (paper, 2.1). The authors also contributed changes to the vLLM project (paper, 1), see [[vLLM]].

## The equations

### The mixture of experts layer (paper, section 2.1)

$$
y=\sum_{i=0}^{n-1} G(x)_i\cdot E_i(x)
$$

**Where:**
- $x$ is the input vector of one token and $y$ the layer's output.
- $n$ is the number of experts (8 in Mixtral).
- $E_i(x)$ is the output of expert $i$ (an ordinary feed-forward block). **Matters:** each expert is a separate set of weights, so more experts means more parameters.
- $G(x)_i$ is the gate's weight for expert $i$. **Matters:** if it is 0, expert $i$ need not be computed at all (this is the saving); if the gate were never zero, every expert would run and the layer would cost 8 times as much.

**Example (tiny numbers):** two experts get weights 0.7311 and 0.2689, the others 0; their outputs are 10 and 20. Then $y=0.7311\times10+0.2689\times20=12.689$. The answer means: the layer returns a blend, mostly the first expert, and the other experts cost nothing.

### The gate (paper, section 2.1)

$$
G(x):=\operatorname{Softmax}\big(\operatorname{TopK}(x\cdot W_g)\big),\qquad \big(\operatorname{TopK}(\ell)\big)_i:=\begin{cases}\ell_i & \text{if } \ell_i \text{ is among the top-}K\text{ of }\ell\\ -\infty & \text{otherwise}\end{cases}
$$

**Where:**
- $x\cdot W_g$ is the vector of $n$ scores ("logits"), $\ell$, from a linear layer with weights $W_g$. **Matters:** it is the only learned part of the choice.
- $\operatorname{TopK}$ keeps the $K$ largest scores and replaces the rest by minus infinity. **Matters:** it makes the gate sparse.
- $\operatorname{Softmax}$ turns scores into positive weights that add to 1; $e^{-\infty}=0$, so unchosen experts get weight exactly 0. **Matters:** the weights are a proper average of the chosen experts only.
- $K$ is the number of experts per token (2 here). **Matters:** it sets the compute per token ("a hyper-parameter that modulates the amount of compute", paper 2.1).

**Example (tiny numbers):** scores $(2.0,1.0,0.5,-1.0)$, $K=2$: after TopK, $(2.0,1.0,-\infty,-\infty)$; softmax gives $(0.7311,0.2689,0,0)$. The answer means: only experts 0 and 1 are used, with 73% and 27% of the say.

### Mixtral's layer (paper, section 2.1)

$$
y=\sum_{i=0}^{n-1}\operatorname{Softmax}\big(\operatorname{Top2}(x\cdot W_g)\big)_i\cdot\operatorname{SwiGLU}_i(x)
$$

**Where:**
- $\operatorname{Top2}$ is TopK with $K=2$ (Table 1: `top_k_experts` is 2).
- $\operatorname{SwiGLU}_i(x)$ is expert $i$: a SwiGLU feed-forward block with its own weights ([[LLaMA 2023]] has its formula). **Matters:** it is the same block as Mistral 7B's, only 8 copies with different weights.
- The rest is as above. **Matters:** this is the layer that replaces the feed-forward block in every one of the 32 blocks.

**Example (tiny numbers):** with $n=8$ and two experts chosen with weights 0.7 and 0.3, each token pays for 2 feed-forward blocks, not 8: $2/8=25\%$ of the feed-forward compute (my arithmetic). Including the shared attention and the embeddings, the model uses 12.88 of 46.70 billion parameters per token, 27.6%.

### Expected expert repeats (paper's Table 5 caption)

$$
P_{\text{repeat, 2 choices}}=1-\frac{6}{8}\cdot\frac{5}{7}\approx 46\%,\qquad P_{\text{repeat, first choice}}=\frac{1}{8}=12.5\%
$$

**Where:**
- The first formula is the chance that, among 2 random experts for token $i+1$, at least one is among the 2 experts of token $i$. $\tfrac68\cdot\tfrac57$ is the chance both picks avoid the 2 earlier experts (6 of 8 experts remain for the first pick, then 5 of 7). **Matters:** it is the baseline to tell "locality" from chance.
- The second is the chance that the first choice of token $i+1$ equals the first choice of token $i$ when experts are picked uniformly at random.

**Example (tiny numbers):** $1-\tfrac{6}{8}\cdot\tfrac57=1-\tfrac{30}{56}=1-0.5357=0.4643$, which is the same as $13/28$ (my arithmetic; I also checked it with $1-\binom62/\binom82=1-15/28$). The answer means: if the router chose at random, 46% of tokens would repeat one of the previous token's two experts. The real routers repeated 62% to 67% at layer 15 (Table 5), far above chance.

## What they found
- **Headline (abstract):** Mixtral "outperforms or matches Llama 2 70B and GPT-3.5 across all evaluated benchmarks", and "vastly outperforms Llama 2 70B on mathematics, code generation, and multilingual benchmarks". The tables below contradict "all": see the notice at the end of this section.
- **Size and cost (paper, 3):** 13B active parameters against 70B, "5x lower" (my arithmetic: 70 divided by 13 is 5.4). But memory is proportional to the 47B total, and sparse layers "introduce additional overhead" from routing; they suit batched workloads (paper, 3).
- **Open weights:** both base and instruct models released under Apache 2.0.

![[mixtral-fig3-size-vs-llama2.png]]
*The paper's Figure 3, cropped from the PDF: six panels (MMLU, knowledge, reasoning, comprehension, math, code) of score against active parameters for Mistral 7B, Mixtral 8x7B and Llama 2 (7B, 13B, 70B). At 13B active parameters Mixtral 8x7B (orange) is above Llama 2 13B (red) in every panel, and above the 70B in every panel except comprehension, where the 70B is higher (the paper's text says the same). Authors: Jiang and others, Figure 3 of arXiv 2401.04088, CC BY 4.0, cropped from the paper.*

### Comparison with the Llama models (paper's Table 2)

| Model | Active params | MMLU | HellaS | WinoG | PIQA | Arc-e | Arc-c | NQ | TriQA | HumanE | MBPP | Math | GSM8K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaMA 2 7B | 7B | 44.4 | 77.1 | 69.5 | 77.9 | 68.7 | 43.2 | 17.5 | 56.6 | 11.6 | 26.1 | 3.9 | 16.0 |
| LLaMA 2 13B | 13B | 55.6 | 80.7 | 72.9 | 80.8 | 75.2 | 48.8 | 16.7 | 64.0 | 18.9 | 35.4 | 6.0 | 34.3 |
| LLaMA 1 33B | 33B | 56.8 | 83.7 | 76.2 | 82.2 | 79.6 | 54.4 | 24.1 | 68.5 | 25.0 | 40.9 | 8.4 | 44.1 |
| LLaMA 2 70B | 70B | 69.9 | **85.4** | **80.4** | 82.6 | 79.9 | 56.5 | 25.4 | **73.0** | 29.3 | 49.8 | 13.8 | 69.6 |
| Mistral 7B | 7B | 62.5 | 81.0 | 74.2 | 82.2 | 80.5 | 54.9 | 23.2 | 62.5 | 26.2 | 50.2 | 12.7 | 50.0 |
| Mixtral 8x7B | 13B | **70.6** | 84.4 | 77.2 | **83.6** | **83.1** | **59.7** | **30.6** | 71.5 | **40.2** | **60.7** | **28.4** | **74.4** |
| *Mixtral minus LLaMA 2 70B (mine)* | | +0.7 | -1.0 | -3.2 | +1.0 | +3.2 | +3.2 | +5.2 | -1.5 | +10.9 | +10.9 | +14.6 | +4.8 |

Written by me, an AI, from the paper's Table 2 (all six rows and twelve columns; the values in the paper carry a % sign, which I dropped). The bold marks and the last row are mine. In the paper's text the benchmarks are footnoted as follows.
- **MMLU** (5-shot, [[Benchmarks]]): multiple-choice questions on 57 subjects. **HellaS** (HellaSwag), **WinoG** (WinoGrande), **PIQA**, **Arc-e / Arc-c** (ARC easy and challenge): common-sense reasoning, all 0-shot. **NQ** (NaturalQuestions) and **TriQA** (TriviaQA): world knowledge, 5-shot. **HumanE** (HumanEval, 0-shot) and **MBPP** (3-shot): writing code that passes tests. **Math** (MATH, 4-shot with majority vote over 4) and **GSM8K** (8-shot with majority vote over 8): maths problems (all from paper, 3).
- **Active params:** the parameters used per token. It is the paper's measure of inference cost. **Matters:** it is how the paper shows Mixtral is cheap.
- All values are percentages; higher is better. All models were re-evaluated by the authors' own pipeline (paper, 3).

**Summary: what the table shows.** Mixtral beats LLaMA 2 70B in 9 of the 12 columns, by the most on Math (+14.6, 2.06 times the score), HumanEval and MBPP (+10.9 each, my arithmetic). It is lower on HellaSwag (-1.0), WinoGrande (-3.2) and TriviaQA (-1.5). The mean of the 12 columns is 63.70 against 59.63, +4.07 (my arithmetic; a mean over mixed benchmarks is only a rough guide). **Caution:** the abstract says "across all evaluated benchmarks" but three columns here and two in the next table are lower; the paper's own caption says "almost all". The authors ran all benchmarks themselves, so the numbers may differ from published Llama 2 numbers (the paper notes differences on MBPP, where it uses a hand-verified subset, and TriviaQA, where it gives no Wikipedia context).

### Comparison with Llama 2 70B and GPT-3.5 (paper's Table 3)

| Benchmark | LLaMA 2 70B | GPT-3.5 | Mixtral 8x7B | Mixtral minus LLaMA (mine) | Mixtral minus GPT-3.5 (mine) |
|---|---|---|---|---|---|
| MMLU (5-shot, 57 subjects) | 69.9 | 70.0 | 70.6 | +0.7 | +0.6 |
| HellaSwag (10-shot) | 87.1 | 85.5 | 86.7 | -0.4 | +1.2 |
| ARC Challenge (25-shot) | 85.1 | 85.2 | 85.8 | +0.7 | +0.6 |
| WinoGrande (5-shot) | 83.2 | 81.6 | 81.2 | -2.0 | -0.4 |
| MBPP (pass@1) | 49.8 | 52.2 | 60.7 | +10.9 | +8.5 |
| GSM-8K (5-shot) | 53.6 | 57.1 | 58.4 | +4.8 | +1.3 |
| MT Bench (for Instruct models) | 6.86 | 8.32 | 8.30 | +1.44 | -0.02 |

Written by me, an AI, from the paper's Table 3 (all seven rows); the last two columns are mine. The MT Bench row compares the instruct versions (Llama 2 70B chat, GPT-3.5 Turbo 1106, Mixtral 8x7B Instruct).
- **Benchmark and shots:** the setting changes the number. **Matters:** Table 3 uses other shot counts than Table 2, so HellaSwag is 84.4 there and 86.7 here, and GSM8K is 74.4 there (8-shot, majority vote over 8) and 58.4 here (5-shot): these are not contradictions.
- **MT Bench:** a 10-point score from a judge model on multi-turn questions ([[LLM as a Judge]]), 1 to 10; higher is better.

**Summary: what the table shows.** Mixtral is best of the three on 4 of the 6 base benchmarks (MMLU, ARC-C, MBPP and GSM-8K), second on HellaSwag (0.4 below Llama 2 70B) and last on WinoGrande (0.4 below GPT-3.5 and 2.0 below Llama 2 70B). On MT Bench it ties GPT-3.5 (8.30 against 8.32) and is far above Llama 2 70B chat (6.86). **Caution:** the margins on MMLU, ARC-C and WinoGrande are under 3 points, with no error bars. The paper's own text calls MMLU "despite its significantly smaller capacity (47B tokens compared to 70B)", an obvious typo for parameters.

### Multilingual (paper's Table 4)

| Model (active params) | French Arc-c | French HellaS | French MMLU | German Arc-c | German HellaS | German MMLU | Spanish Arc-c | Spanish HellaS | Spanish MMLU | Italian Arc-c | Italian HellaS | Italian MMLU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLaMA 1 33B (33B) | 39.3 | 68.1 | 49.9 | 41.1 | 63.3 | 48.7 | 45.7 | 69.8 | 52.3 | 42.9 | 65.4 | 49.0 |
| LLaMA 2 70B (70B) | 49.9 | 72.5 | 64.3 | 47.3 | 68.7 | 64.2 | 50.5 | 74.5 | 66.0 | 49.4 | 70.9 | 65.1 |
| Mixtral 8x7B (13B) | 58.2 | 77.4 | 70.9 | 54.3 | 73.0 | 71.5 | 55.4 | 77.6 | 72.5 | 52.8 | 75.1 | 70.9 |

Written by me, an AI, from the paper's Table 4 (all three rows, all twelve columns).
- **Language and test:** the benchmark translated into French, German, Spanish and Italian; **Arc-c**, **HellaS** and **MMLU** as above. **Matters:** the paper says that, compared with Mistral 7B, it "significantly upsample[d]" the share of multilingual data in pre-training (paper, 3.1).

**Summary: what the table shows.** Mixtral beats LLaMA 2 70B in all 12 cells, by 3.1 to 8.3 points, an average of +5.5 (my arithmetic), the most on French Arc-c (+8.3) and German MMLU (+7.3). **Caution:** four European languages only; no error bars; the cause may be the extra multilingual data, not the experts (the paper does not separate the two).

### Long context (paper's Figure 4)

![[mixtral-fig4-long-range.png]]
*The paper's Figure 4, cropped from the PDF. Left: passkey retrieval, where the model must find a short code hidden at a random place in a long text; the whole shaded area is green (retrieval accuracy 1.0). Right: the perplexity on a subset of the proof-pile dataset falls as the context length grows. Authors: Jiang and others, Figure 4 of arXiv 2401.04088, CC BY 4.0, cropped from the paper.*

The paper says Mixtral gets 100% on passkey retrieval whatever the context length or the position, and that perplexity "decreases monotonically as the context length increases" (paper, 3.2). I looked at both plots. In the right plot the perplexity falls from about 3.8 at very short context to about 2.0 by 8,000 tokens and then flattens near 1.9 (read by eye). **Caution:** passkey retrieval is a synthetic and easy test; it does not show that the model can use long documents well.

### Bias (paper's Figure 5, a table)

| | LLaMA 2 70B | Mixtral 8x7B |
|---|---|---|
| BBQ accuracy | 51.5% | 56.0% |
| BOLD sentiment, gender | 0.293 ± 0.073 | 0.323 ± 0.045 |
| BOLD sentiment, profession | 0.218 ± 0.073 | 0.243 ± 0.087 |
| BOLD sentiment, religious ideology | 0.188 ± 0.133 | 0.144 ± 0.089 |
| BOLD sentiment, political ideology | 0.149 ± 0.140 | 0.186 ± 0.146 |
| BOLD sentiment, race | 0.232 ± 0.049 | 0.232 ± 0.052 |

Written by me, an AI, from the paper's Figure 5 (a small table printed as a figure; all rows).
- **BBQ accuracy:** share of correct answers on a set of hand-written questions about nine social categories (age, gender identity, race and so on); higher means less biased behaviour (paper, 3.3).
- **BOLD sentiment:** the average sentiment (positive or negative tone, "avg ± std") of text the model writes from 23,679 prompts in five domains. A higher average means a more positive tone; a lower standard deviation means less difference within the group (paper, 3.3).

**Summary: what the table shows.** On BBQ Mixtral is 4.5 points higher (my arithmetic). On BOLD it is more positive in gender, profession and political ideology, lower in religious ideology and equal in race; its spread is lower for gender and religion but higher for profession and political ideology. **Caution:** the paper's text says the results are "in Table 5" and that the variances are "similar", but its Table 5 is about expert repeats and its Figure 5 shows the bias numbers; the spread is not uniformly lower, so "less bias" holds clearly only for BBQ.

### Instruction-tuned version (paper, section 4)
Mixtral 8x7B Instruct was made with supervised fine-tuning on an instruction dataset followed by [[Direct Preference Optimization]] on paired feedback (paper, 4). It scores 8.30 on MT Bench ("the best open-weights model as of December 2023") and, in the independent human evaluation of the LMSYS [[Chatbot Arena]] on 22 December 2023, an Elo rating of 1121 against Claude-2.1 1117, GPT-3.5-Turbo 1117 (best version), Gemini Pro 1111 and Llama-2-70b-chat 1077 (paper, 4 and Figure 6; I read the figure's numbers from its caption text, not from the screenshot). My arithmetic: a 4-point Elo gap means about a 50.6% chance of being preferred in a one-to-one comparison (standard Elo formula, general knowledge), so 1121 against 1117 is almost a tie; the 44-point gap to Llama-2-70b-chat means about 56%.

### Routing analysis (paper's Section 5, Figures 7 and 8, Table 5)
The authors looked at which expert the router picks, on different kinds of text from The Pile, at layers 0, 15 and 31.

![[mixtral-fig7-expert-selection.png]]
*The paper's Figure 7, cropped from the PDF: the share of tokens sent to each of the 8 experts (x-axis) for layers 0, 15 and 31 (the three panels), for eight kinds of text (the colours: ArXiv, DM Mathematics, Github, Gutenberg, PhilPapers, PubMed Abstracts, StackExchange, Wikipedia). The dashed line is 1/8, what a uniform choice would give. Authors: Jiang and others, Figure 7 of arXiv 2401.04088, CC BY 4.0, cropped from the paper.*

"Surprisingly, we do not observe obvious patterns in the assignment of experts based on the topic": the distribution is very similar for ArXiv, biology (PubMed) and philosophy (PhilPapers) at all layers; only DM Mathematics differs a little, which the authors put down to its synthetic nature (paper, 5). I looked at the figure: the bars of the eight kinds of text sit within about the same range for each expert, with the DM Mathematics bar (yellow) often the odd one out, for example at layer 31 for expert 0 and expert 2.

![[mixtral-fig8-token-experts.png]]
*The paper's Figure 8, cropped from the PDF: text samples (Python code, maths questions, English) in which each token's background colour shows its first-choice expert, for layers 0, 15 and 31. Authors: Jiang and others, Figure 8 of arXiv 2401.04088, CC BY 4.0, cropped from the paper.*

The colours follow syntax more than topic: words such as `self` in Python and `Question` in English go to the same expert, and the indentation tokens in code always go to the same experts, mostly in the first and last layers (paper, 5). Consecutive tokens often get the same expert, which the paper measures in its Table 5.

| Text from The Pile | First choice, layer 0 | First choice, layer 15 | First choice, layer 31 | First or second, layer 0 | First or second, layer 15 | First or second, layer 31 |
|---|---|---|---|---|---|---|
| ArXiv | 14.0 | 27.9 | 22.7 | 46.5 | 62.3 | 52.9 |
| DM Mathematics | 14.1 | 28.4 | 19.7 | 44.9 | 67.0 | 44.5 |
| Github | 14.9 | 28.1 | 19.7 | 49.9 | 66.9 | 49.2 |
| Gutenberg | 13.9 | 26.1 | 26.3 | 49.5 | 63.1 | 52.2 |
| PhilPapers | 13.6 | 25.3 | 22.1 | 46.9 | 61.9 | 51.3 |
| PubMed Abstracts | 14.2 | 24.6 | 22.0 | 48.6 | 61.6 | 51.8 |
| StackExchange | 13.6 | 27.2 | 23.6 | 48.2 | 64.6 | 53.6 |
| Wikipedia (en) | 14.4 | 23.6 | 25.3 | 49.8 | 62.1 | 51.8 |
| *Mean of the eight (mine)* | 14.09 | 26.40 | 22.68 | 48.04 | 63.69 | 50.91 |
| *Chance (paper)* | 12.5 | 12.5 | 12.5 | about 46 | about 46 | about 46 |

Written by me, an AI, from the paper's Table 5 (all eight rows and six columns, values in percent); the mean row is mine and the chance row comes from the paper's caption.
- **First choice:** how often token $i+1$ gets the same expert as token $i$ as the router's *first* choice. **First or second:** the same expert appears among the two chosen by both tokens. **Matters:** high values mean the router keeps sending neighbouring tokens to the same experts, the "temporal locality".
- **Layer:** 0 is the first block, 31 the last.
- **Chance:** what random routing would give, 12.5% and about 46% (the equation above).

**Summary: what the table shows.** At layer 0 the repeats are close to chance (14.1% against 12.5%, 48.0% against 46%). At layers 15 and 31 they are well above it: on average 26.4% and 22.7% for the first choice (about 2.1 and 1.8 times chance, mine) and 63.7% and 50.9% for either choice. The highest value is DM Mathematics at layer 15 (67.0%). **Caution:** the paper itself points out that locality is good for caching experts, but it also risks overloading single experts under expert parallelism. All the data is English-dominated text from The Pile; the paper does not study other languages here.

**Notices where the paper is inconsistent:**
- The abstract and introduction say Mixtral "outperforms or matches Llama 2 70B ... across all evaluated benchmarks", but its Tables 2 and 3 show it lower on HellaSwag (-1.0 and -0.4), WinoGrande (-3.2 and -2.0) and TriviaQA (-1.5), and the Table 2 caption says "almost all".
- The text on page 4 says "47B tokens compared to 70B", which must mean parameters.
- The text says the bias results are in "Table 5" (they are in Figure 5; Table 5 is expert repeats), and "see Table 2" for MT Bench (it is in Table 3).
- The Llama row is "LLaMA 1 33B" in Table 2, but the text and footnote call it "Llama 1 34B" and "Llama 2 34B".

## Questions I had, answered

**Does "8x7B" mean 56 billion parameters?**
No, 46.70 billion by my arithmetic (the paper says 47B). Only the feed-forward blocks are eight-fold: 32 layers times 8 experts times 176.16 million is 45.10 billion; the attention (1.34 billion) and the embeddings (0.26 billion) are used once. Eight times 7 would be 56 billion. The name is a nickname (my reading; the paper does not discuss it).

**If only 13B parameters work per token, why does the model need memory for 47B?**
Because any token may need any expert, so all of them must be loaded; the paper says the memory cost is proportional to the sparse count, 47B, "which is still smaller than Llama 2 70B" (paper, 3). The saving is in compute, not memory. Example: 12.88 of 46.70 billion are used per token (27.6%), but 100% must sit in memory.

**Is each expert a specialist in one subject, like a maths expert?**
The paper looked for this and did not find it: expert use is nearly the same for ArXiv, biology and philosophy text; the pattern seems to follow syntax (words like `self`, indentation) more than topic (paper, 5). Only the synthetic DM Mathematics data looks a little different. So the name "expert" is misleading. The analysis is on one dataset and three layers; it is not proof that no specialisation exists.

**Why does the gate apply softmax after keeping the top two, not before?**
The paper does not say why. In my example it gives weights (0.7311, 0.2689) that add up to exactly 1, whereas softmax over all four scores first and then keeping two gives 0.6095 and 0.2242, which add up to only 0.8337. That is my observation, not the authors' reason.

**Why does the router pick different experts for the same word in different layers?**
Each layer has its own router weights $W_g$, and the token vector has changed after the earlier layers. The paper's Figure 7 shows different distributions for layers 0, 15 and 31, and says repeats at layer 0 are near chance (Table 5). (The first half is general knowledge.)

## Why it mattered
It was an open-weights mixture-of-experts model that matched or beat a dense 70B model on most of the paper's tests while using 13B parameters per token (paper). The authors call it "the first mixture-of-experts network to reach a state-of-the-art performance among open-source models" (paper, 6). It showed that sparse experts work in a released model; DeepSeek-R1 also uses a mixture of experts (my [[DeepSeek-R1 2025]] note).

## Limits and open questions
- **Memory:** all 47B parameters must be stored; sparsity saves compute only (paper, 3).
- **Hardware efficiency:** routing adds overhead, and the layer fits batched workloads better; with one expert per device and uneven loads, devices may sit idle (paper, 3 and 2.1).
- **Load balancing:** experts must get similar numbers of tokens or some GPUs become a bottleneck (paper, 2.1); the repeat locality seen at higher layers can make this worse (paper, 5).
- **What the experts learn:** no clear topic specialisation was found, and the paper offers no full explanation.
- **Missing details:** the paper does not describe the training data, the training cost or why 8 and 2 were chosen; it gives no ablation on the number of experts.
- **Self-run evaluation:** all benchmark numbers come from the authors' own pipeline, and the margins on some are small.

## Concepts used
[[Mixture of Experts]], [[Transformer]], [[Feed-Forward Network]], [[Softmax]], [[Attention]], [[Grouped-Query Attention]], [[KV Cache]], [[Context Window]], [[vLLM]], [[Benchmarks]], [[Perplexity]], [[Chatbot Arena]], [[LLM as a Judge]], [[Direct Preference Optimization]], [[Instruction Tuning]], [[LLaMA 2023]]

## Reading trail
- **Came from:** [[LLaMA 2023]] (the dense model it is compared with); Mistral 7B (reference 18, not read); Shazeer and others' sparsely-gated experts and GShard (named in the paper, not read).
- **Led to:** [[DeepSeek-R1 2025]] (also a mixture-of-experts model, my note).

## How much I read
I read the whole main text as extracted text (pages 1 to 8) including all tables; I did not read the reference list (pages 9 to 11) beyond names, nor the appendix figures 9 and 10 (pages 12 and 13) as anything but captions. I looked at the cropped Figures 1, 3, 4, 7 and 8 and at page 2 and 4 as pictures. I did not look at Figure 2 (a radar chart) or Figure 6 (a screenshot of the leaderboard); their numbers are from the text. I did not run Mixtral. Numbers from the paper are marked "paper" or "abstract"; the sums, differences and all tiny examples are my arithmetic. Not opened in Obsidian, so the flashcards are unreviewed.

## Sources
- Jiang, A. Q. and 25 others (2024). *Mixtral of Experts.* arXiv 2401.04088 (submitted 8 January 2024). <https://arxiv.org/abs/2401.04088>. Licence on the arXiv page: Creative Commons Attribution 4.0 (CC BY 4.0). Figures 1, 3, 4, 7 and 8 were cropped from the PDF with PyMuPDF at 220 dpi and are credited under each figure.
- Standard Elo formula for the win chance: general knowledge, not from the paper.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-llm Mixtral (Jiang et al., Mistral AI, 2024, arXiv 2401.04088): what problem did it attack?::A model with many parameters (knowledge) but a small cost per token: dense models use every parameter for every word. A sparse mixture of experts "increases the number of parameters of a model while controlling cost and latency".

#flashcards/research/concept #flashcards/topic/research-llm What did people do before Mixtral?::Dense models such as Llama 2 70B and GPT-3.5, and earlier mixture-of-experts layers (Shazeer and others; GShard, which replaces every other feed-forward block and uses a more elaborate gate for the second expert, unlike Mixtral).

#flashcards/research/concept #flashcards/topic/research-llm Mixtral: what is the idea?::Take Mistral 7B and replace each layer's feed-forward block by 8 experts plus a router. For every token at every layer the router picks 2 experts and adds their outputs with the router's weights, so each token reaches 47B parameters but uses only 13B.

#flashcards/research/concept #flashcards/topic/research-llm What are Mixtral's sizes (paper's Table 1)?::dim 4096, 32 layers, head dim 128, hidden dim 14336, 32 query heads and 8 key-value heads, context 32768, vocabulary 32000, 8 experts with top 2.

#flashcards/research/concept #flashcards/topic/research-llm How many of Mixtral's parameters are in the experts, and which are shared?::About 96.6%: 32 layers × 8 experts × 176.16M = 45.10B (my arithmetic). Shared (used by every token): attention (1.34B) and embeddings (0.26B).

#flashcards/research/concept #flashcards/topic/research-llm How do 47B total and 13B active parameters come out of Table 1?::My arithmetic with no biases and untied embeddings: per layer attention 41.94M, one expert 176.16M, router 0.03M. All 8 experts: 32 × 1451.26M + 262.1M = 46.70B. Two experts: 32 × 394.30M + 262.1M = 12.88B. The paper's 47B and 13B match.

#flashcards/research/concept #flashcards/topic/research-llm What does Mixtral's router do and how big is it?::One linear layer $W_g$ turns the 4,096-number token vector into 8 scores, one per expert, and the top 2 are kept. It has 4,096 × 8 = 32,768 numbers per layer (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm What are the steps of a Mixtral mixture-of-experts layer for one token?::Attention as usual; score 8 experts with the router; keep the top 2 (others to minus infinity); softmax over the 2; run only those 2 experts; add their outputs weighted; then the usual residual connection (general knowledge).

#flashcards/research/concept #flashcards/topic/research-llm Why does the memory for Mixtral follow 47B and not 13B?::Any token may need any expert, so all experts must be loaded; sparsity saves compute only. The paper says the memory cost is proportional to the sparse count (47B), "still smaller than Llama 2 70B".

#flashcards/research/concept #flashcards/topic/research-llm Why 8 experts and top 2 in Mixtral?::The paper does not say and shows no ablation. It only says that $K$ sets the compute per token and that raising $n$ with $K$ fixed raises the parameters while keeping compute "effectively constant".

#flashcards/research/concept #flashcards/topic/research-llm Write the mixture of experts layer output and explain each symbol.
?
$$y=\sum_{i=0}^{n-1} G(x)_i\cdot E_i(x)$$
- $x$: the token vector; $y$: the layer output.
- $n$: number of experts (8).
- $E_i(x)$: output of expert $i$, a feed-forward block with its own weights.
- $G(x)_i$: the gate weight of expert $i$. If it is 0 the expert need not be computed; that is the saving.

#flashcards/research/concept #flashcards/topic/research-llm Example: gate weights 0.7311 and 0.2689 (others 0) and expert outputs 10 and 20. What is the layer output?::$0.7311\times10+0.2689\times20=12.689$: a blend mostly of the first expert; the other experts cost nothing (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Why must some gate weights be exactly zero in a mixture of experts?::A zero weight means that expert's output need not be computed, which is where the compute is saved; if every weight were non-zero all 8 experts would run and the layer would cost about 4 times what Mixtral's does (8 against 2, my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Write Mixtral's gate and explain each symbol.
?
$$G(x):=\operatorname{Softmax}\big(\operatorname{TopK}(x\cdot W_g)\big),\qquad \big(\operatorname{TopK}(\ell)\big)_i:=\begin{cases}\ell_i & \text{if } \ell_i \text{ is among the top-}K\text{ of }\ell\\ -\infty & \text{otherwise}\end{cases}$$
- $x\cdot W_g$: the $n$ scores (logits) from a linear layer; the only learned part of the choice.
- $\operatorname{TopK}$: keeps the $K$ largest scores, others to $-\infty$; it makes the gate sparse.
- $\operatorname{Softmax}$: positive weights adding to 1; $e^{-\infty}=0$, so unchosen experts get exactly 0.
- $K$: experts per token (2); it sets the compute per token.

#flashcards/research/concept #flashcards/topic/research-llm Gate example: scores $(2.0,1.0,0.5,-1.0)$ and $K=2$. What are the gate weights?::After TopK $(2.0,1.0,-\infty,-\infty)$; softmax gives $(0.7311,0.2689,0,0)$ (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Why softmax after TopK and not before?::The paper does not say. In my example softmax after keeps weights 0.7311 and 0.2689, which add up to 1; softmax first then keeping two would give 0.6095 and 0.2242, adding to only 0.8337. (My observation, not the authors' reason.)

#flashcards/research/concept #flashcards/topic/research-llm Write Mixtral's full layer formula and explain each symbol.
?
$$y=\sum_{i=0}^{n-1}\operatorname{Softmax}\big(\operatorname{Top2}(x\cdot W_g)\big)_i\cdot\operatorname{SwiGLU}_i(x)$$
- $\operatorname{Top2}$: TopK with $K=2$ (`top_k_experts`).
- $\operatorname{SwiGLU}_i(x)$: expert $i$, a SwiGLU feed-forward block with its own weights; the same block as Mistral 7B's, in 8 copies.
- The rest is as in the gate formula; this replaces the feed-forward block in all 32 layers.

#flashcards/research/concept #flashcards/topic/research-llm Example: with 8 experts and 2 chosen, what share of the feed-forward compute does a Mixtral token use, and of all parameters?::2/8 = 25% of the feed-forward compute; and 12.88 of 46.70 billion parameters in all, 27.6% (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Write the paper's expected-repeat formulas (Table 5 caption) and explain each symbol.
?
$$P_{\text{repeat, 2 choices}}=1-\frac{6}{8}\cdot\frac{5}{7}\approx 46\%,\qquad P_{\text{repeat, first choice}}=\frac{1}{8}=12.5\%$$
- $\tfrac68\cdot\tfrac57$: the chance that two random picks both avoid the 2 experts of the previous token (6 of 8 remain, then 5 of 7).
- $1-$ that: the chance of at least one repeat.
- $\tfrac18$: the chance that the first choice equals the previous first choice by luck.
- They are the baselines for judging whether routing has locality.

#flashcards/research/concept #flashcards/topic/research-llm Compute the expected repeat chance for two random experts out of eight.::$1-\tfrac68\cdot\tfrac57=1-\tfrac{30}{56}=0.4643$, equal to $13/28$ (also $1-\binom62/\binom82=1-15/28$). My arithmetic. The real layers 15 and 31 got 62 to 67% and 44 to 54%.

#flashcards/research/concept #flashcards/topic/research-llm Why compare the repeat rates with 12.5% and about 46%?::Those are the rates if routing were random; layer 0 is near them (14.1% and 48.0% on average) while layer 15 is well above (26.4% first choice, about 2.1 times chance, and 63.7% either; my arithmetic). Only above-chance repeats show real locality.

#flashcards/research/concept #flashcards/topic/research-llm ![[mixtral-fig1-moe-layer.png]]
What does the paper's Figure 1 show?
?
A mixture-of-experts layer: an input goes to a router, which sends it to 2 of the 8 experts; the layer's output is the weighted sum (circled plus) of the two experts' outputs, using the gating weights from the router. (Cropped from the paper, CC BY 4.0.)

#flashcards/research/concept #flashcards/topic/research-llm How did Mixtral compare with the Llama models in Table 2?::It beat LLaMA 2 70B in 9 of 12 columns, most on Math (28.4 against 13.8, +14.6), HumanEval and MBPP (+10.9 each), but was lower on HellaSwag (-1.0), WinoGrande (-3.2) and TriviaQA (-1.5), with 13B active parameters against 70B. The mean was 63.70 against 59.63 (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Where is Mixtral's abstract contradicted by its own tables?::The abstract says it outperforms or matches Llama 2 70B "across all evaluated benchmarks", but Tables 2 and 3 show it lower on HellaSwag, WinoGrande and TriviaQA; the Table 2 caption says "almost all". The text also says "47B tokens" for parameters and points to the wrong table numbers for bias and MT Bench.

#flashcards/research/concept #flashcards/topic/research-llm ![[mixtral-fig3-size-vs-llama2.png]]
What do the six panels of this figure show?
?
Score against active parameters (7B, 13B, 34B, 70B) for MMLU, knowledge, reasoning, comprehension, math and code, comparing Mistral 7B and Mixtral 8x7B (orange) with Llama 2 7B, 13B and 70B (red). At 13B active parameters Mixtral 8x7B is above Llama 2 13B everywhere, and above the 70B everywhere except comprehension. (Cropped from the paper, CC BY 4.0.)

#flashcards/research/concept #flashcards/topic/research-llm How did Mixtral compare with Llama 2 70B and GPT-3.5 in Table 3?::Best or tied on MMLU (70.6), ARC-C (85.8), MBPP (60.7) and GSM-8K (58.4), below both on WinoGrande (81.2) and below Llama on HellaSwag by 0.4. MT Bench: 8.30 against GPT-3.5's 8.32 and Llama 2 70B chat's 6.86. Caution: several margins under 3 points with no error bars.

#flashcards/research/concept #flashcards/topic/research-llm Why do the same benchmarks have different numbers in Mixtral's Tables 2 and 3?::They use different shot counts and protocols: HellaSwag is 84.4 in Table 2 and 86.7 (10-shot) in Table 3; GSM8K is 74.4 (8-shot with majority vote over 8) and 58.4 (5-shot). They are not contradictions.

#flashcards/research/concept #flashcards/topic/research-llm How did Mixtral do on the multilingual benchmarks (Table 4)?::It beat LLaMA 2 70B in all 12 cells (French, German, Spanish and Italian on Arc-c, HellaSwag and MMLU) by 3.1 to 8.3 points, +5.5 on average (my arithmetic). Caution: the cause may be the extra multilingual data, not the experts; the paper does not separate them.

#flashcards/research/concept #flashcards/topic/research-llm ![[mixtral-fig4-long-range.png]]
What do the two plots of this figure show?
?
Left: passkey retrieval, where the model must find a hidden code in a long text; the whole area is green, so 100% retrieval whatever the context length and position. Right: perplexity on proof-pile falls as the context grows, from about 3.8 to about 2.0 by 8,000 tokens (read by eye). The test is synthetic and easy. (Cropped from the paper, CC BY 4.0.)

#flashcards/research/concept #flashcards/topic/research-llm What do Mixtral's bias results show and what is the caution?::BBQ accuracy 56.0% against Llama 2 70B's 51.5% (+4.5); BOLD sentiment is more positive in gender, profession and political ideology, lower in religious ideology, equal in race. Caution: the spreads are not uniformly lower, and the text points to Table 5 (expert repeats) instead of Figure 5.

#flashcards/research/concept #flashcards/topic/research-llm How was Mixtral 8x7B Instruct made and how did it rank?::Supervised fine-tuning followed by Direct Preference Optimization. It scored 8.30 on MT Bench and 1121 Elo on the LMSYS Arena (22 December 2023) against Claude-2.1 1117, GPT-3.5-Turbo 1117 and Llama-2-70b-chat 1077. A 4-point gap is only about 50.6% win chance (standard Elo formula).

#flashcards/research/concept #flashcards/topic/research-llm ![[mixtral-fig7-expert-selection.png]]
What does the routing analysis figure show?
?
The share of tokens sent to each of the 8 experts at layers 0, 15 and 31 for eight kinds of text (ArXiv, DM Mathematics, Github, Gutenberg, PhilPapers, PubMed, StackExchange, Wikipedia); the dashed line is 1/8. No obvious topic pattern; only DM Mathematics differs a little. (Cropped from the paper, CC BY 4.0.)

#flashcards/research/concept #flashcards/topic/research-llm ![[mixtral-fig8-token-experts.png]]
What do the coloured tokens in this figure show?
?
Python code, maths questions and English text with each token's background coloured by its first-choice expert, at layers 0, 15 and 31. The colours follow syntax more than topic: `self`, `Question` and the indentation tokens in code often go to the same expert. (Cropped from the paper, CC BY 4.0.)

#flashcards/research/concept #flashcards/topic/research-llm Do Mixtral's experts specialise in topics?::The paper looked and found no obvious topic pattern (ArXiv, biology and philosophy text use experts almost the same way); selection looks more syntactic. Only the synthetic DM Mathematics data differs a little. One dataset and three layers, so it is not proof of no specialisation.

#flashcards/research/concept #flashcards/topic/research-llm What does Mixtral's Table 5 show about routing locality?::Repeat rates for consecutive tokens are near chance at layer 0 (14.1% first choice and 48.0% either, on average) but well above at layer 15 (26.4% and 63.7%) and layer 31 (22.7% and 50.9%); highest is DM Mathematics at layer 15 (67.0%). Locality helps caching but can overload single experts.

#flashcards/research/concept #flashcards/topic/research-llm Does "8x7B" mean 56 billion parameters?::No: 46.70B by my arithmetic (the paper says 47B). Only the feed-forward blocks are eight-fold (45.10B); attention (1.34B) and embeddings (0.26B) are used once. It is a nickname.

#flashcards/research/concept #flashcards/topic/research-llm What limits does the Mixtral paper state or show?::All 47B parameters must be stored; routing adds overhead and suits batched workloads; load balancing across GPUs is hard (and locality can overload experts); the training data, cost and the choice of 8 and 2 are not described; all benchmarks are self-run.

#flashcards/research/concept #flashcards/topic/research-llm How much of the Mixtral paper did the note read?::The whole main text (pages 1 to 8) including all tables, but not the references or the appendix plots as anything but captions; Figures 2 and 6 were not looked at as pictures. Mixtral itself was not run. Sums and tiny examples are my arithmetic.
