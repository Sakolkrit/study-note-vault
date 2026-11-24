---
tags: [paper, llm, scaling-laws, transformer, training, power-law]
link: https://arxiv.org/abs/2001.08361
authors: Kaplan, McCandlish (equal first authors), Henighan, Brown, Chess, Child, Gray, Radford, Wu, Amodei (OpenAI; Kaplan also Johns Hopkins; from the paper's first page)
year: 2020
status: body-read
field: large language models, training
---

# Scaling Laws 2020

> **In one line for a 12-year-old:** The authors trained lots of language models of many different sizes and found that, as you make the model, the data and the computer time bigger, the model's mistakes shrink in a smooth, predictable way, like a recipe where you can tell in advance how much better a bigger batch will be.

Paper: "Scaling Laws for Neural Language Models", Jared Kaplan, Sam McCandlish and eight co-authors (OpenAI), arXiv 2001.08361, posted 23 January 2020 (the first page). Concept note: [[Scaling Laws]]. Its main follow-up in this vault is [[Chinchilla 2022]]. Reference note for this layout: [[Attention Is All You Need 2017]]. The arXiv licence is the standard non-exclusive one, so none of the paper's figures is copied here; I found no free redrawing of them either (a Commons search for scaling-law figures gave only unrelated images), so the one figure below is a generic power-law illustration, not the paper's.

## The problem
How good a language model gets depends on its architecture, its size, the computer time spent training it, and the amount of data. Before this paper there was no clear, quantitative picture of how these trade off for Transformers. If you have a fixed budget, should you build a bigger model or train a smaller one for longer? The authors measure the dependence of the loss on each factor, for the [[Transformer]] and the cross-entropy loss ([[Cross-Entropy Loss]], [[Perplexity]]) (paper, 1).

## What people did before
- **Earlier scaling work:** early work found power-law links between performance and dataset size, and more recent work looked at model size against data size. The paper calls Hestness et al. (2017) and a 2019 follow-up the closest. They found the data needed grows **faster** than linearly with model size, whereas this paper finds it grows **slower** than linearly (paper, 7).
- **Typical practice:** train smaller models until they appear to converge. The paper says researchers usually train smaller models for longer than would be compute-efficient, because of hardware limits (paper, 1.2).
- **Competing explanations:** EfficientNet advocates scaling depth and width exponentially for image models. This paper finds that for language models width over depth should stay roughly fixed and that the precise shape matters little (paper, 7).

## The idea
1. **Train a very wide range of models.** From 768 to 1.5 billion non-embedding parameters, on datasets from 22 million to 23 billion tokens, with different shapes (depth, width, attention heads, feed-forward size), context lengths and batch sizes (paper, 3).
2. **Define "size" as non-embedding parameters $N$.** Excluding the vocabulary and position tables gives much cleaner trends (paper, 2.1 and Figure 6).
3. **Estimate compute as $C\approx6NBS$.**
4. **Measure the test loss** as a function of $N$, the dataset size $D$ and the compute, each while the other two are not the bottleneck, and fit power laws.
5. **Combine** the fits into formulas for the loss in terms of $N$ and $D$ (which also gives overfitting), and of $N$ and training steps.
6. **Use the formulas** to say how a fixed compute budget should be split between model size, batch size and steps (paper, 6).

## The setup, part by part
This paper has no new network. Its "architecture" is the experimental set-up, which is a decoder-only [[Transformer]] of varying shape.

- **The models.**
  - *What they are:* decoder-only Transformers, parameterised by `n_layer` (layers), `d_model` (width of the residual stream), `d_ff` (feed-forward width), `d_attn` (attention output width) and `n_heads`, with the usual `d_attn = d_ff/4 = d_model`; context 1,024 tokens (paper, 2.1). They also train LSTMs and recurrent ("Universal") Transformers for comparison.
  - *Why it matters:* the shape is varied on purpose to see whether it matters. It barely does at fixed $N$ (paper, Figure 5).
- **The data.**
  - *What it is:* WebText2, an extended version of WebText: web pages linked from Reddit posts with at least 3 karma, now through October 2018. In total 20.3 million documents, 96 GB of text, $1.62\times10^{10}$ words and $2.29\times10^{10}$ tokens after byte-pair encoding ([[Byte-Pair Encoding]], vocabulary 50,257), of which $6.6\times10^{8}$ tokens are kept as a test set (paper, 2.3). Also tested: Books Corpus, Common Crawl, English Wikipedia and public Internet books.
  - *Why it matters:* the fitted constants belong to this data and this tokenisation. My check: 2.29e10 tokens divided by the paper's 1.4 tokens per word is 1.64e10 words, matching the 1.62e10 words.
- **The training recipe.**
  - *What it is:* Adam ([[Adam Optimizer]]) for a fixed $2.5\times10^{5}$ steps with a batch of 512 sequences of 1,024 tokens; the largest models (above 1B parameters) used Adafactor; a 3,000-step linear warmup then cosine decay to zero; dropout 10% in the data-limited runs (paper, 2.2 and 4.2).
  - *Why it matters:* the paper says results at convergence were "largely independent of learning rate schedule" (2.2). My arithmetic: 512 × 1,024 = 524,288 tokens a step times $2.5\times10^5$ steps is $1.31\times10^{11}$ tokens, about 5.7 passes over the 2.29e10 training tokens, so the default runs reuse data (the data-limited and compute-optimal runs are separate).
- **Why 6 layers up to 207 layers, and why these sizes?** The paper does not give a reason for the particular grid; it only states the ranges (shapes from (6, 4288) to (207, 768) for billion-parameter models, paper 3.2).

### How parameters and compute are counted (paper's Table 1)

| Operation | Parameters | FLOPs per token |
|---|---|---|
| Embed | $(n_{\text{vocab}}+n_{\text{ctx}})\,d_{\text{model}}$ | $4\,d_{\text{model}}$ |
| Attention: QKV | $n_{\text{layer}}\,d_{\text{model}}\,3d_{\text{attn}}$ | $2n_{\text{layer}}\,d_{\text{model}}\,3d_{\text{attn}}$ |
| Attention: Mask | none | $2n_{\text{layer}}\,n_{\text{ctx}}\,d_{\text{attn}}$ |
| Attention: Project | $n_{\text{layer}}\,d_{\text{attn}}\,d_{\text{model}}$ | $2n_{\text{layer}}\,d_{\text{attn}}\,d_{\text{embd}}$ |
| Feedforward | $n_{\text{layer}}\,2d_{\text{model}}\,d_{\text{ff}}$ | $2n_{\text{layer}}\,2d_{\text{model}}\,d_{\text{ff}}$ |
| De-embed | none | $2d_{\text{model}}\,n_{\text{vocab}}$ |
| **Total (non-embedding)** | $N=2d_{\text{model}}\,n_{\text{layer}}(2d_{\text{attn}}+d_{\text{ff}})$ | $C_{\text{forward}}=2N+2n_{\text{layer}}\,n_{\text{ctx}}\,d_{\text{attn}}$ |

Written by me, an AI, from the paper's Table 1 (all rows; the cells are formulas, copied from the extracted text and checked against the totals). Terms from nonlinearities, biases and layer normalisation are left out by the paper.
- **Operation:** a part of the Transformer; *QKV* are the query, key and value projections of [[Attention]]; *Mask* is the attention score step; *Project* the output projection.
- **Parameters:** how many learned numbers that part holds. **Matters:** adding the non-embedding rows gives $N$.
- **FLOPs per token:** floating-point operations to push one token through that part in the forward pass; the factor 2 is one multiply and one add. **Matters:** it is where $C\approx6N$ per token comes from (the factor 3 for forward plus backward is explained under the equations).

**Check (my arithmetic).** For $n_{\text{layer}}=12$, $d_{\text{model}}=768$, $d_{\text{ff}}=3072$: QKV 21.23 million, Project 7.08 million, Feedforward 56.62 million, sum 84.93 million, equal to $12\,n_{\text{layer}}d_{\text{model}}^2=84{,}934{,}656$. The embedding table adds $(50{,}257+1{,}024)\times768=39.4$ million, so the total is 124.3 million and 31.7% of it is embeddings. The forward cost per token is $2N=1.70\times10^{8}$ plus a context term $2n_{\text{layer}}n_{\text{ctx}}d_{\text{attn}}=1.89\times10^{7}$, which is 10% of the sum.

**Summary: what the table shows.** The feed-forward part holds two thirds of the parameters (56.6 of 84.9 million here) and attention one third. The context-dependent term is small unless the context is long compared with the width: the paper drops it when $d_{\text{model}}>n_{\text{ctx}}/12$ (768 > 85.3 in this case; mine). **Caution:** the formulas are the paper's estimates, not measured counts.

## The core mechanisms, step by step

### Reading a power law
A **power law** is a rule $y=(x_c/x)^{\alpha}$. On axes where both sides are drawn on a logarithmic scale it is a straight line, whose slope is $-\alpha$. That is why the paper's figures are straight lines on log-log axes, and why each doubling of $x$ always multiplies $y$ by the same number, $2^{-\alpha}$.

![[power-law-log-log.png]]
*A generic illustration of power laws on ordinary (left) and log-log (right) axes: $y=x$, $y=x^{1/2}$ and $y=x^2$ are curved on the left and straight lines on the right. Image: Talgalili, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Comparison_of_simple_power_law_curves_in_original_and_log-log_scale.png), CC BY-SA 4.0. It is not a figure of the paper: the paper's laws have negative exponents (loss falls as $N$ grows, so its lines slope downwards), and its axes are loss, parameters, data and compute rather than $x$ and $y$.*

### Measuring a scaling law
1. **Pick the factor to study.** *What it does:* for example $N$. *Why it matters:* every other factor must not be the bottleneck, or the curve flattens for the wrong reason.
2. **Train a family of models** across a wide range of that factor. *What it does:* for $N$, train to near convergence on the full 22 billion tokens, with models from (2, 128) to billion-parameter shapes (paper, 3.2).
3. **Record the test loss** at the best point of training (for data limits, with early stopping).
4. **Fit the power law** on log-log axes and read off the exponent. *Why it matters:* a straight line over many orders of magnitude is the evidence that the law holds there; the paper says it sees "no signs of deviation" at the top end (1.1).
5. **Combine factors** with one formula that reduces to each single-factor law when the other is unlimited ($L(N,D)$ and $L(N,S_{\min})$ below).

### Training at the critical batch size
Bigger batches use fewer steps but waste compute once the batch is too big. The **critical batch size** $B_{\text{crit}}$ is the best compromise, and the paper (building on McCandlish and others, named in the paper, not read) uses it to put all runs on the same footing:

1. **Measure** the steps $S$ and examples $E=BS$ needed to reach a loss at different batch sizes $B$.
2. **They satisfy** $(S/S_{\min}-1)(E/E_{\min}-1)=1$, where $S_{\min}$ is the fewest steps (huge batch) and $E_{\min}$ the fewest examples (tiny batch).
3. **Define** $B_{\text{crit}}=E_{\min}/S_{\min}$: training there needs $2S_{\min}$ steps and $2E_{\min}$ examples.
4. **Convert** any run with batch $B$ to the ideal ones: $S_{\min}=S/(1+B_{\text{crit}}/B)$ and $C_{\min}=C/(1+B/B_{\text{crit}})$.

**Worked example (my invented numbers):** $S_{\min}=1{,}000$ steps and $E_{\min}=10^{8}$ examples give $B_{\text{crit}}=10^{5}$. At $B=B_{\text{crit}}$ you need 2,000 steps and $2\times10^8$ examples ($(2-1)(2-1)=1$). At $B=10\,B_{\text{crit}}$ solving the relation gives $S=1.1\,S_{\min}$ and $E=11\,E_{\min}$: a 10 times larger batch saves almost no steps but processes 11 times the examples.

### The compute-efficient frontier
For each compute budget, find the model size that reaches the lowest loss; the paper does this empirically (Figures 13 and 14) and theoretically (Appendix B) and the two agree roughly (paper, 6). The result: most of any extra compute should go into a bigger model, a little into a bigger batch, almost none into more steps. For a billion-fold more compute my arithmetic from the paper's exponents gives a $3.7\times10^{6}$ times larger model, a 145 times larger batch, 1.9 times more steps and 270 times more data, matching the paper's Figure 3 labels ("100x batch size, <10x serial steps, >1,000,000x model size").

## The equations
Eleven equations from the paper, in the order the argument uses them. The constants have no fundamental meaning: they depend on the vocabulary and tokenisation (paper, 1.2).

### Non-embedding parameters (paper's equation 2.1)

$$
N\approx 2\,d_{\text{model}}\,n_{\text{layer}}\,(2d_{\text{attn}}+d_{\text{ff}})=12\,n_{\text{layer}}\,d_{\text{model}}^{2}\quad\text{with } d_{\text{attn}}=d_{\text{ff}}/4=d_{\text{model}}
$$

**Where:**
- $n_{\text{layer}}$ is the number of Transformer layers and $d_{\text{model}}$ the width of the residual stream.
- $d_{\text{attn}}$ and $d_{\text{ff}}$ are the attention and feed-forward widths (set to $d_{\text{model}}$ and $4d_{\text{model}}$).
- $N$ is the size used in every law. **Matters:** it excludes the embedding tables; including them hides the trend (paper, Figure 6).

**Example (tiny numbers):** $n_{\text{layer}}=12$, $d_{\text{model}}=768$: $12\times12\times768^2=84.9$ million. For the $(48,1600)$ shape: $12\times48\times1600^2=1.475\times10^{9}$, the paper's 1.5 billion; for $(6,4288)$ it is $1.32\times10^{9}$. The answer means: very different shapes give nearly the same $N$, which is how the paper compares shapes at fixed size.

### Training compute (paper, 1.3 and 2.1)

$$
C\approx 6\,N\,B\,S
$$

**Where:**
- $N$ is non-embedding parameters; $B$ the batch size in tokens; $S$ the number of parameter updates, so $BS$ is the number of tokens processed.
- 6 is about 2 operations per parameter per token in the forward pass plus about 4 in the backward pass, which costs roughly twice the forward pass (paper, 2.1). **Matters:** it makes cost proportional to both model size and tokens processed.
- $C$ is in floating-point operations, quoted as PF-days: $1\text{ PF-day}=10^{15}\times24\times3600=8.64\times10^{19}$ operations.

**Example (tiny numbers):** $N=10^9$, $B=524{,}288$ tokens, $S=2.5\times10^{5}$: $C=6\times10^{9}\times524{,}288\times2.5\times10^{5}=7.86\times10^{20}$ operations $=9.1$ PF-days. The answer means: the paper's default run of a billion-parameter model costs about 9 days of a computer that does $10^{15}$ operations per second. (For $N=10^8$ it is 0.91 PF-days.)

### Loss against model size (paper's equations 1.1 and 3.1)

$$
L(N)=\left(\frac{N_c}{N}\right)^{\alpha_N},\qquad \alpha_N\approx0.076,\quad N_c\approx8.8\times10^{13}
$$

**Where:**
- $L$ is the test loss in nats per token (lower is better).
- $N$ is non-embedding parameters; the law holds for models trained to convergence on enough data.
- $N_c$ is a tokenisation-dependent constant with no fundamental meaning. $\alpha_N$ is the exponent. **Matters:** the smaller it is, the slower the loss falls; doubling $N$ multiplies the loss by $2^{-\alpha_N}=0.949$.

**Example (tiny numbers):** $N=10^8$: $L=(8.8\times10^{5})^{0.076}=2.830$. $N=2\times10^8$: $2.685$, a ratio of $0.949$. $N=10^9$: $2.376$. The answer means: doubling the model removes about 5% of the loss, and a 10-fold increase takes it from 2.83 to 2.38.

### Loss against model and data together (paper's equation 1.5 and 4.1)

$$
L(N,D)=\left[\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D}+\frac{D_c}{D}\right]^{\alpha_D}
$$

**Where:**
- $D$ is the dataset size in tokens; $D_c$ and $\alpha_D$ are the data constant and exponent.
- The fit in the paper's Table 2: $\alpha_N=0.076$, $\alpha_D=0.103$, $N_c=6.4\times10^{13}$, $D_c=1.8\times10^{13}$ (a little different from the single-factor values because both are fitted together).
- **Matters:** with unlimited data the second term vanishes and you get $L(N)$; with unlimited $N$ the first vanishes and you get $L(D)$. In between each one bottlenecks the other. The paper chose this form on three principles: it must rescale with the vocabulary, it must reduce to $L(N)$ and $L(D)$, and it should be analytic in $1/D$ (the last "significantly weaker" in theory) (paper, 4.1).

**Example (tiny numbers):** $N=10^8$. Infinite data: $L=2.762$. $D=10^{10}$ tokens: the first term is $(6.4\times10^{5})^{0.738}=19{,}239$ and $D_c/D=1{,}800$, so $L=(19{,}239+1{,}800)^{0.103}=2.788$, a penalty of 0.93%. $D=10^{9}$: $D_c/D=18{,}000$, $L=2.957$, a penalty of 7.0%. The answer means: ten times less data makes the penalty about 7.6 times bigger, so a model this size is starved at $10^9$ tokens.

### Data needed to avoid overfitting (paper's equation 4.4)

$$
D\gtrsim\left(5\times10^{3}\right)N^{0.74}
$$

**Where:**
- $D$ is training tokens and $N$ non-embedding parameters.
- $0.74$ is $\alpha_N/\alpha_D$ from the fit (0.076/0.103 = 0.738). **Matters:** it is below 1, so data may grow more slowly than the model.
- The constant is set by requiring the loss to be within about 0.02 of converged, the typical variation between random seeds. **Matters:** a smaller penalty would be hidden in the noise.

**Example (tiny numbers):** $N=10^9$: $D\gtrsim5\times10^{3}\times(10^{9})^{0.74}=2.29\times10^{10}$ tokens, about the whole of WebText2. $N=10^8$: $4.2\times10^{9}$. A model 8 times larger needs only $8^{0.74}=4.7$ times more data. The answer means: this is the minimum to avoid overfitting, not the amount that makes training compute-efficient (paper, 4.2).

### The critical batch size relation (paper's equations 5.1 and 5.2)

$$
\left(\frac{S}{S_{\min}}-1\right)\left(\frac{E}{E_{\min}}-1\right)=1,\qquad B_{\text{crit}}\equiv\frac{E_{\min}}{S_{\min}}
$$

**Where:**
- $S$ is the number of steps and $E=BS$ the number of examples used to reach a fixed loss.
- $S_{\min}$ is the fewest steps possible (batch far above the critical size); $E_{\min}$ the fewest examples possible (batch far below it).
- $B_{\text{crit}}$ is their ratio. **Matters:** at $B=B_{\text{crit}}$ both costs are doubled ($S=2S_{\min}$, $E=2E_{\min}$), the best trade-off between time and compute (paper, 5.1).

**Example (tiny numbers):** $S_{\min}=1{,}000$, $E_{\min}=10^{8}$, so $B_{\text{crit}}=10^{5}$. At $B=B_{\text{crit}}$: $(2-1)(2-1)=1$ holds with $S=2{,}000$. At $B=10B_{\text{crit}}$: $S=1.1S_{\min}$ and $E=11E_{\min}$ (check: $0.1\times10=1$). The answer means: a batch ten times the critical one cuts the steps only from $2S_{\min}$ (at the critical batch) to $1.1S_{\min}$, but raises the data used from $2E_{\min}$ to $11E_{\min}$.

### The critical batch size against loss (paper's equations 1.4 and 5.3)

$$
B_{\text{crit}}(L)=\frac{B_{*}}{L^{1/\alpha_B}},\qquad B_{*}\approx2\times10^{8}\text{ tokens},\quad\alpha_B\approx0.21
$$

**Where:**
- $L$ is the target loss; $B_{*}$ and $\alpha_B$ are fitted constants (Table 5 gives $B_*=2.1\times10^{8}$).
- **Matters:** the critical batch size depends on the loss only, not on the model size (paper, Figure 10): the lower the loss you want, the bigger the batch you can use. The form makes $B_{\text{crit}}$ diverge as $L\to0$, because the gradient noise is expected to diverge there; the paper says the true minimum loss is above zero but unknown.

**Example (tiny numbers):** $L=3.0$: $B_{\text{crit}}=2.1\times10^{8}/3^{4.76}=1.12\times10^{6}$ tokens; $L=2.0$: $7.7\times10^{6}$ tokens. A 13% lower loss multiplies it by $0.87^{-4.76}=1.94$, the paper's "doubles for every 13% decrease in loss". The answer means: better models want much bigger batches.

### Adjusting steps and compute to the critical batch size (paper's equations 5.4 and 5.5)

$$
S_{\min}(S)=\frac{S}{1+B_{\text{crit}}(L)/B},\qquad C_{\min}(C)=\frac{C}{1+B/B_{\text{crit}}(L)}
$$

**Where:**
- $S$ and $C$ are the steps and compute actually used at batch size $B$; $S_{\min}$ and $C_{\min}$ are what they would be in the ideal limits (large and small batch respectively).
- $B_{\text{crit}}(L)$ is from the previous equation. **Matters:** most of the paper's runs used a fixed batch of $2^{19}$ tokens, which is not optimal; this converts them into clean trends.

**Example (tiny numbers):** $B=2^{19}=524{,}288$ and $L=3.0$ ($B_{\text{crit}}=1.12\times10^{6}$): $S_{\min}=S/(1+2.14)=0.318\,S$, so $10^{5}$ steps count as 31,836; $C_{\min}=C/(1+0.468)=0.682\,C$. The answer means: that run used a batch too small for its loss, so it spent about 32% more compute than the ideal one.

### Loss against model size and training steps (paper's equations 1.6 and 5.6)

$$
L(N,S_{\min})=\left(\frac{N_c}{N}\right)^{\alpha_N}+\left(\frac{S_c}{S_{\min}}\right)^{\alpha_S}
$$

**Where:**
- The first term is the loss floor for a model of size $N$ (the infinite-training limit); the second falls as training goes on.
- The paper's fit (Table 3): $\alpha_N=0.077$, $\alpha_S=0.76$, $N_c=6.5\times10^{13}$, $S_c=2.1\times10^{3}$ steps.
- **Matters:** it separates "the model is too small" from "it has not trained long enough". It holds after an initial transient and for models trained with Adam; it breaks down early in training (paper, 5.2 and Figure 11).

**Example (tiny numbers):** $N=10^8$: first term $2.803$. With $S_{\min}=10^{4}$: second term $(0.21)^{0.76}=0.305$, so $L=3.108$. With $S_{\min}=10^{5}$: second term $0.056$, $L=2.856$. With endless training: $2.803$. The answer means: at $10^{5}$ steps this model is within 2% of its limit; more training can never get below 2.803.

### How to split a growing budget (paper's equations 1.7 and 1.8)

$$
\alpha_C^{\min}=\frac{1}{1/\alpha_S+1/\alpha_B+1/\alpha_N},\qquad N\propto C_{\min}^{\alpha_C^{\min}/\alpha_N},\ \ B\propto C_{\min}^{\alpha_C^{\min}/\alpha_B},\ \ S\propto C_{\min}^{\alpha_C^{\min}/\alpha_S},\ \ D=B\cdot S
$$

**Where:**
- $\alpha_C^{\min}$ is the exponent of the loss against the best compute; the other three exponents split the budget between model size, batch size and steps.
- $\alpha_S,\alpha_B,\alpha_N$ are the three exponents above (0.76, 0.21 and about 0.076).
- The measured values are $N\propto C_{\min}^{0.73}$, $B\propto C_{\min}^{0.24}$, $S\propto C_{\min}^{0.03}$ and $D\propto C_{\min}^{0.27}$ (paper, 6.1 and Table 6). **Matters:** most extra compute should go into model size.

**Example (tiny numbers):** a 10-fold larger budget: model $10^{0.73}=5.4$ times larger, batch $10^{0.24}=1.7$ times, steps $10^{0.03}=1.07$ times, data $10^{0.27}=1.9$ times. The formula with the paper's own numbers gives $1/(1/0.76+1/0.21+1/0.076)=0.052$, with exponents $0.68$ for $N$, $0.25$ for $B$ and $0.07$ for $S$ (my arithmetic). The answer means: ten times the compute buys a model more than 5 times bigger but only about twice the data; this is the allocation [[Chinchilla 2022]] later disputed.

### Loss against optimal compute (paper's equations 1.3 and 6.3)

$$
L(C_{\min})=\left(\frac{C_c^{\min}}{C_{\min}}\right)^{\alpha_C^{\min}},\qquad \alpha_C^{\min}\approx0.050,\quad C_c^{\min}\approx3.1\times10^{8}\text{ PF-days}
$$

**Where:**
- $C_{\min}$ is the least compute needed to reach a loss (batch much below critical).
- $C_c^{\min}$ is a constant with no fundamental meaning; $\alpha_C^{\min}$ is the exponent. **Matters:** it is the law the authors expect to extrapolate best, since it corrects for the batch size (paper, Figure 13 caption).

**Example (tiny numbers):** $C_{\min}=1$ PF-day: $L=(3.1\times10^{8})^{0.050}=2.658$. $C_{\min}=10^{4}$ PF-days: $L=1.677$. The answer means: 10,000 times more compute lowers the loss from 2.66 to 1.68, about 37%. (Figure 13's own legend uses $2.3\times10^{8}$ instead of $3.1\times10^{8}$; see the notice below.)

**The breakdown the paper predicts (paper, 6.3, equations 6.6 to 6.8).** If you never reuse data, compute-efficient training uses $D(C_{\min})=2C_{\min}/(6N(C_{\min}))$ tokens, and the data law $L(D)\propto D^{-0.095}$ then gives a loss that falls only as $C_{\min}^{-0.03}$ (my arithmetic: $0.27\times0.095=0.026$), more slowly than $L(C_{\min})\propto C_{\min}^{-0.050}$. So the curves must meet, at roughly $C^{*}\sim10^{4}$ PF-days, $N^{*}\sim10^{12}$ parameters, $D^{*}\sim10^{12}$ tokens and $L^{*}\sim1.7$ nats per token, "highly uncertain" by an order of magnitude either way. My recomputation with $N=1.3\times10^{9}C^{0.73}$ and the paper's constants finds the crossing at $C^{*}=1.7\times10^{4}$ PF-days, $N^{*}=1.6\times10^{12}$, $D^{*}=3.1\times10^{11}$ and $L^{*}=1.63$: the same order of magnitude except for $D^{*}$ (3 times lower than the paper's $10^{12}$). The authors read the crossing as the point where the laws must break down, and conjecture it might estimate the entropy of text.

## What they found
Each number is marked "abstract" or "paper" (read in the paper's body text, as extracted from the PDF; I also looked at the page picture of Figures 12 and 13). The constants depend on the vocabulary and tokenisation, so they have no fundamental meaning (paper).
- **Smooth power laws (abstract):** the loss follows a power law in model size, dataset size and compute, with some trends spanning more than seven orders of magnitude.
- **Shape matters little (paper, 3.1 and Figure 5):** at fixed $N$, aspect ratio can vary by a factor of 40 while only slightly changing the loss; a $(6,\,4288)$ model reaches a loss within 3% of the $(48,\,1600)$ model used for GPT-2. Models with fewer than 2 layers or extreme shapes are exceptions (Figure 6). The paper's Figure 5 also reads "22% additional compute compensates for 1% loss increase".
- **Overfitting depends on $N^{0.74}/D$ (paper):** every time the model size grows 8 times, about 5 times more data avoids a penalty.
- **Sample efficiency:** larger models reach the same loss with fewer samples; compute-efficient training uses very large models on modest data and stops far short of convergence (abstract). My arithmetic from Appendix B.3: that stopping point is about 10% above the converged loss ($\alpha_N/\alpha_S=0.1$); compared with training to 2% above it, it uses 2.7 times more parameters, 7.7 times fewer steps and 65% less compute (the paper's numbers 2.7, 0.13 and 0.35; my recomputation 2.70, 0.133 and 0.359).
- **Transformers against LSTMs (Figure 7):** LSTMs match Transformers early in the context but plateau after about 100 tokens, while Transformers keep improving through the whole context.
- **Transfer (paper, 3.2.2 and Figure 8):** on other text distributions (Books, Common Crawl, Wikipedia, internet books) the loss improves in parallel with the training distribution, with a roughly constant offset, and depends almost only on the in-distribution loss, not on training duration or model depth.
- **Models off the optimal size (paper, Figure 12 and Appendix B.4):** models between 0.6 and 2.2 times the optimal size cost only 20% more compute; a 2.2 times larger model needs 45% fewer steps.
- **Their reading (discussion):** "Big models may be more important than big data."

### Fit of L(N, D) (paper's Table 2)

| Parameter | $\alpha_N$ | $\alpha_D$ | $N_c$ | $D_c$ |
|---|---|---|---|---|
| Value | 0.076 | 0.103 | $6.4\times10^{13}$ | $1.8\times10^{13}$ |

Written by me, an AI, from the paper's Table 2 (the whole table).
- **$\alpha_N$ and $\alpha_D$:** how fast the loss falls with model size and with data; **$N_c$ and $D_c$:** scale constants, in parameters and tokens (tokenisation-dependent).

**Summary: what the table shows.** The data exponent is bigger than the model exponent (0.103 against 0.076), so by this fit one extra doubling of data helps more than one of parameters, until the other becomes the bottleneck. **Caution:** the paper says the fit is "excellent" except where the dataset was cut to about $2\times10^{7}$ tokens (an epoch of 40 steps), and the values differ slightly from the single-factor ones ($\alpha_D=0.095$, $N_c=8.8\times10^{13}$) because they come from a joint fit.

### Fit of L(N, S) (paper's Table 3)

| Parameter | $\alpha_N$ | $\alpha_S$ | $N_c$ | $S_c$ |
|---|---|---|---|---|
| Value | 0.077 | 0.76 | $6.5\times10^{13}$ | $2.1\times10^{3}$ |

Written by me, an AI, from the paper's Table 3 (the whole table). $\alpha_S$ is the exponent for training steps (at the critical batch size), $S_c$ a scale in steps.

**Summary: what the table shows.** Training time has a far larger exponent (0.76) than model size (0.077), so extra steps pay off quickly at first and the loss floor is set by $N$. **Caution:** the paper calls the fits "imperfect" and "mediocre at small $S$"; the law breaks down early in training.

### The summary of fitted power laws (paper's Tables 5 and 6, Appendix A)

| Power law | Exponent | Scale |
|---|---|---|
| $L(N)$ | $\alpha_N=0.076$ | $N_c=8.8\times10^{13}$ parameters |
| $L(D)$ | $\alpha_D=0.095$ | $D_c=5.4\times10^{13}$ tokens |
| $L(C)$, naive (fixed batch) | $\alpha_C=0.057$ | $C_c=1.6\times10^{7}$ PF-days |
| $L(C_{\min})$ | $\alpha_C^{\min}=0.050$ | $C_c^{\min}=3.1\times10^{8}$ PF-days |
| Critical batch size | $\alpha_B=0.21$ | $B_*=2.1\times10^{8}$ tokens |
| Steps | $\alpha_S=0.76$ | $S_c=2.1\times10^{3}$ steps |
| Optimal model size $N_{\text{opt}}$ | $p_N=0.73$ | $N_e=1.3\times10^{9}$ parameters |
| Batch size ($B\ll B_{\text{crit}}$) | $p_B=0.24$ | $B_e=2.0\times10^{6}$ tokens |
| Steps $S_{\min}$ (lower bound) | $p_S=0.03$ | $S_e=5.4\times10^{3}$ steps |
| Data $D_{\text{opt}}$ (one epoch) | $p_D=0.27$ | $D_e=2\times10^{10}$ tokens |

Written by me, an AI, from the paper's Tables 5 and 6, combined into one table (all rows of both).
- **Exponent:** the slope on a log-log plot; for the last four rows the exponent of the optimal quantity against $C_{\min}$ in PF-days.
- **Scale:** the value at 1 unit (for the last four rows, at $C_{\min}=1$ PF-day).

**Summary: what the table shows.** Model size grows fastest with compute ($p_N=0.73$); steps hardly at all ($p_S=0.03$); the loss falls slowly in every law (exponents 0.05 to 0.1). **Check (mine):** at $C_{\min}=1$ PF-day the table gives $N=1.3\times10^{9}$ and $D=2\times10^{10}$ tokens, and $6ND=1.56\times10^{20}$ operations, or 1.8 PF-days, which is $2C_{\min}$ as the one-epoch rule says (the factor 2 from training at the critical batch size). **Caution:** these are fits over a limited range; the paper itself says they cannot hold at very large scale (see the breakdown above).

**Notices where the paper is inconsistent:**
- **Orders of magnitude:** the abstract says "more than seven orders of magnitude", section 1.1 says "more than six orders of magnitude" for $N$, $D$ and $C$, and section 1.2 says "eight orders of magnitude in $C_{\min}$, six orders of magnitude in $N$, and over two orders of magnitude in $D$".
- **The compute law's constants:** equation 1.3 and Table 5 give $C_c^{\min}=3.1\times10^{8}$ PF-days and $C_c=1.6\times10^{7}$; the legend of Figure 13 (which I looked at on the page picture) gives $2.3\times10^{8}$ and $2.0\times10^{7}$.
- **The compute exponent:** 0.050 (measured, 1.3 and Table 5), 0.054 (equation 6.4) and 0.052 (equation B.7). With the paper's own $\alpha_S$, $\alpha_B$ and $\alpha_N$ the formula gives 0.052, not 0.054. Because of this, equation 6.5 gives 0.71 for the model-size exponent, while 0.052 would give 0.68, against the measured 0.73.
- **The steps exponent:** the theory gives $0.052/0.76=0.07$ for $S$ while the measured value is 0.03, yet the paper says the theory "closely matches" the measured values. Both are small.
- **The data law:** equation 6.7 reads $D(C_{\min})\approx(4\times10^{10})\,C_{\min}^{0.26}$, but Table 6 says $2\times10^{10}$ and exponent 0.27, which is also what I get by computing $2C_{\min}/(6N)$ with $N=1.3\times10^{9}C^{0.73}$ ($2.2\times10^{10}\,C^{0.27}$).
- **Placeholders:** Appendix B.4 refers to "Figure X" and "Figure Y", which are not numbers of real figures.
- **Notation:** Table 1's attention-projection FLOPs use $d_{\text{embd}}$, which is not defined; it seems to mean $d_{\text{model}}$.
- **Minor:** $B_*$ is "about $2\times10^{8}$" in the text and $2.1\times10^{8}$ in the figure and tables.

## Questions I had, answered

**What does "non-embedding parameters" mean and why leave the embeddings out?**
The embedding tables turn each token into a vector; they are lookups, not the "thinking" layers. For a 12-layer, 768-wide model the thinking layers have 84.9 million parameters and the tables 39.4 million (31.7% of the total of 124.3 million; my arithmetic). The paper found that counting only the non-embedding ones gives one clean trend for all depths, while counting everything makes shallow models look different (Figure 6).

**Why is it called a "power law" and why are the figures straight lines?**
Because $L=(N_c/N)^{\alpha}$ becomes $\log L=\alpha\log N_c-\alpha\log N$, a straight line when both axes are logarithmic, with slope $-\alpha$. See the illustration above. In numbers: $2^{-0.076}=0.949$ means every doubling cuts the loss by the same 5.1%.

**What is the "critical batch size" in plain words?**
A batch is how many examples the model learns from at once. A very small batch wastes time (many steps); a very big one wastes data (little extra progress). The critical batch is the point in between where neither is badly wasted. In my example, $B_{\text{crit}}=10^5$: at 10 times that, you save almost nothing in steps and use 11 times the examples. It depends on the loss: for a loss of 3.0 it is 1.12 million tokens, for 2.0 it is 7.7 million.

**Why does the paper tell us to train very large models and stop early?**
For a fixed compute budget, a larger model gets a lower loss per unit of compute even before it converges. The paper's Appendix B.3 compares stopping at about 10% above the converged loss with the common habit of about 2%: 2.7 times more parameters, 7.7 times fewer steps and 65% less compute for the same loss. The caveat is that this is a fit over the range they tested.

**Does a lower loss mean a model is smarter?**
The paper does not claim so. It says it is open whether improving loss improves real tasks, and that smooth improvement can hide "qualitative improvements" ("more is different") (discussion).

## Why it mattered
It turned "bigger is better" into a quantitative, predictable rule for choosing model size, data and compute, and it argued for putting most extra compute into larger models. [[GPT-3 2020]] was trained by the same lab (OpenAI) and was a very large model; that the paper shaped it is from memory and not checked here. The paper's allocation advice was later revised: [[Chinchilla 2022]] (see that note) reported that model size and training tokens should be scaled equally, and that current large models were under-trained. Today scaling laws are also used to plan training runs and compare architectures (from memory).

## Limits and open questions
All from the paper's own caveats (Appendix C) and discussion unless marked.
- **No theory:** the authors have no solid theoretical understanding of the laws, "especially" those for model size and compute, so it is hard to know when they can be trusted.
- **Out-of-range predictions:** they are not confident about the critical batch size far outside the range they explored, and the laws must break down at large scale because language has non-zero entropy (the contradiction above).
- **Small-data regime:** fits were poor for the smallest datasets (an epoch of 40 steps), and they did not experiment with regularisation or data augmentation.
- **Compute estimate:** $C\approx6NBS$ leaves out context-length terms, so compute scalings may be confounded when the context is very long (about 12 times the model width or more).
- **Possibly untuned hyperparameters:** they tuned learning rates and schedules but may have missed one (for example the initial scale), and the best learning rate depends on the target loss.
- **Loss, not abilities:** whether lower loss means better results on real tasks is open.
- **One dataset and one architecture family:** the constants are for WebText2 and Transformers; the form is conjectured to carry to other generative models but is not tested here.
- **Internal inconsistencies:** several constants and exponents differ between sections (see the notice above).
- **Later correction (from memory and the Chinchilla note):** the compute-optimal split between parameters and tokens is not the one later found by [[Chinchilla 2022]].

## Concepts used
[[Scaling Laws]] · [[Transformer]] · [[Cross-Entropy Loss]] · [[Perplexity]] · [[Byte-Pair Encoding]] · [[Overfitting]] · [[Adam Optimizer]] · [[LSTM]] · [[Pre-training]] · [[Context Window]] · [[Learning Rate]] · [[Data Parallelism]] · [[Attention]] · [[Benchmarks]]

## Reading trail
- **Came from:** [[Attention Is All You Need 2017]] (the architecture it studies).
- **Led to:** [[Chinchilla 2022]] (revised the compute-optimal allocation) and, in the same lab a few months later, [[GPT-3 2020]] (that GPT-3 was shaped by it is from memory).

## How much I read
I read the whole main text as extracted text (pages 1 to 19) and Appendices A, B and C (pages 20 to 22): abstract, introduction and summary (1.1, 1.2), methods (2.1 to 2.3), the basic power laws (3), overfitting and the $L(N,D)$ fit (4, with Table 2), training time and the critical batch size (5, with Table 3), optimal allocation (6, including 6.3), related work, the discussion, the summary tables of power laws, the efficient-frontier derivation (B.1 to B.4) and the caveats. I did **not** read the supplemental figures and sections after page 22 (Appendix D) beyond what the text cites, and I did not look at most figures: I looked at page 15 (Figures 12 and 13) as a picture; the other figures I know only from their captions and the text. I recomputed every number marked "mine" with Python; I did not re-run any experiment. Not opened in Obsidian, so the flashcards are unreviewed.

## Sources
- Kaplan, McCandlish, Henighan, Brown, Chess, Child, Gray, Radford, Wu, Amodei, "Scaling Laws for Neural Language Models", arXiv 2001.08361 (23 January 2020): <https://arxiv.org/abs/2001.08361>. Licence on the arXiv page: standard non-exclusive distribution licence, so none of its figures is copied.
- Image: [Comparison of simple power law curves in original and log-log scale.png](https://commons.wikimedia.org/wiki/File:Comparison_of_simple_power_law_curves_in_original_and_log-log_scale.png) (Talgalili, CC BY-SA 4.0), Wikimedia Commons.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-llm Scaling Laws (Kaplan, McCandlish et al., OpenAI, 2020, arXiv 2001.08361): what did it study?::How the cross-entropy loss of Transformer language models depends on model size (non-embedding parameters $N$), dataset size $D$ and training compute $C$. It found power laws, with some trends spanning more than seven orders of magnitude (abstract).

#flashcards/research/concept #flashcards/topic/research-llm What did people do before the scaling-laws paper?::Some earlier work found power laws of performance against data size, and Hestness and others studied model against data size but found data needing to grow faster than linearly with model size. Practice was to train smaller models to near convergence. EfficientNet argued that depth and width scaling matters for images.

#flashcards/research/concept #flashcards/topic/research-llm Scaling Laws: what is the idea?::Train a wide range of Transformer language models, measure the test loss against $N$, $D$ and compute, fit power laws, combine them into formulas ($L(N,D)$ and $L(N,S_{\min})$), and use them to split a compute budget between model size, batch size and steps.

#flashcards/research/concept #flashcards/topic/research-llm What models and data did the scaling-laws paper train on?::Decoder-only Transformers from 768 to 1.5 billion non-embedding parameters with varied shapes, on WebText2 (20.3 million documents, 96 GB, 2.29×10^10 tokens, byte-pair encoding with a vocabulary of 50,257), context 1,024. Adam for $2.5\times10^5$ steps with batches of 512 sequences, warmup then cosine decay; Adafactor above 1B parameters.

#flashcards/research/concept #flashcards/topic/research-llm Why does the paper count only non-embedding parameters?::Counting only them makes all depths follow one clean trend; including the embedding tables hides it (Figure 6). For a 12-layer 768-wide model the embedding tables hold 39.4M of 124.3M parameters (31.7%, my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Write the paper's parameter count for a Transformer (equation 2.1) and explain each symbol.
?
$$N\approx 2\,d_{\text{model}}\,n_{\text{layer}}\,(2d_{\text{attn}}+d_{\text{ff}})=12\,n_{\text{layer}}\,d_{\text{model}}^{2}\quad\text{with } d_{\text{attn}}=d_{\text{ff}}/4=d_{\text{model}}$$
- $n_{\text{layer}}$: number of layers; $d_{\text{model}}$: width of the residual stream.
- $d_{\text{attn}}$, $d_{\text{ff}}$: attention and feed-forward widths (here $d_{\text{model}}$ and $4d_{\text{model}}$).
- $N$: the size used in every law; embedding tables are excluded.

#flashcards/research/concept #flashcards/topic/research-llm Parameter-count example: what is $N$ for $(n_{\text{layer}},d_{\text{model}})=(12,768)$, $(48,1600)$ and $(6,4288)$?::$12\times12\times768^2=84.9$ million; $12\times48\times1600^2=1.475\times10^9$ (the paper's 1.5B); $12\times6\times4288^2=1.32\times10^9$ (my arithmetic). Different shapes give nearly the same $N$.

#flashcards/research/concept #flashcards/topic/research-llm Why is the Transformer's non-embedding size $12\,n_{\text{layer}}d_{\text{model}}^2$?::Per layer, attention has $4d_{\text{model}}^2$ (query, key, value and output projections with $d_{\text{attn}}=d_{\text{model}}$) and the feed-forward block $2\times d_{\text{model}}\times4d_{\text{model}}=8d_{\text{model}}^2$; together $12d_{\text{model}}^2$. For $d_{\text{model}}=768$ that is 21.2M + 7.1M + 56.6M = 84.9M per 12 layers (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Write the paper's training-compute estimate and explain each symbol.
?
$$C\approx 6\,N\,B\,S$$
- $N$: non-embedding parameters; $B$: batch size in tokens; $S$: number of updates, so $BS$ is the tokens processed.
- 6: about 2 operations per parameter per token forward plus about 4 backward (the backward pass costs about twice the forward pass).
- $C$: floating-point operations; quoted in PF-days, 1 PF-day $=8.64\times10^{19}$ operations.

#flashcards/research/concept #flashcards/topic/research-llm Compute example: $N=10^9$, $B=524{,}288$ tokens and $S=2.5\times10^5$ steps. What is $C$?::$6\times10^9\times524{,}288\times2.5\times10^5=7.86\times10^{20}$ operations, 9.1 PF-days (my arithmetic). For $N=10^8$: 0.91 PF-days.

#flashcards/research/concept #flashcards/topic/research-llm Why is the compute factor 6 and not 2?::2 operations per parameter per token is the forward pass; the backward pass costs about twice as much (about 4), so 2 + 4 = 6. With only the forward pass, $N=10^9$ and $1.31\times10^{11}$ tokens would be 2.6×10^20 operations, a third of the 7.86×10^20 (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm What do the rows of the paper's Table 1 tell us about where Transformer parameters sit?::Feed-forward holds two thirds (56.6M of 84.9M for 12 layers at width 768), attention one third (QKV 21.2M and projection 7.1M); the embedding table adds 39.4M. The context-dependent cost per token is only 10% of the forward cost there, and the paper drops it when $d_{\text{model}}>n_{\text{ctx}}/12$.

#flashcards/research/concept #flashcards/topic/research-llm ![[power-law-log-log.png]]
What does this illustration show, and how does it differ from the paper's figures?
?
Power laws $y=x$, $y=x^{1/2}$ and $y=x^2$ drawn on ordinary axes (curved) and log-log axes (straight lines). It is a generic image (Talgalili, CC BY-SA 4.0), not the paper's figure: the paper's laws have negative exponents, so its lines slope downward on log-log axes.

#flashcards/research/concept #flashcards/topic/research-llm Why are the paper's loss curves straight lines on log-log axes, and what does the slope mean?::$L=(N_c/N)^{\alpha}$ gives $\log L=\alpha\log N_c-\alpha\log N$, a line with slope $-\alpha$. Every doubling multiplies the loss by the same factor $2^{-\alpha}$; for $\alpha_N=0.076$ that is 0.949, a 5.1% drop.

#flashcards/research/concept #flashcards/topic/research-llm Write the loss-against-model-size law (equation 1.1) and explain each symbol.
?
$$L(N)=\left(\frac{N_c}{N}\right)^{\alpha_N},\qquad \alpha_N\approx0.076,\quad N_c\approx8.8\times10^{13}$$
- $L$: test loss in nats per token (lower is better).
- $N$: non-embedding parameters; the law holds for models trained to convergence on enough data.
- $N_c$: a tokenisation-dependent constant with no fundamental meaning.
- $\alpha_N$: the exponent; doubling $N$ multiplies the loss by $2^{-\alpha_N}=0.949$.

#flashcards/research/concept #flashcards/topic/research-llm $L(N)$ example: what are $L$ at $N=10^8$, $2\times10^8$ and $10^9$?::2.830, 2.685 (ratio 0.949) and 2.376: doubling removes about 5% of the loss and a 10-fold increase takes it from 2.83 to 2.38 (my arithmetic from the paper's constants).

#flashcards/research/concept #flashcards/topic/research-llm Why is the exponent 0.076 so small, and what does it imply?::A small exponent means slow but steady progress: each doubling of $N$ removes only about 5% of the loss. Getting from 2.83 to 2.38 took a 10-fold increase from $10^8$ to $10^9$ parameters. Returns are steady, never sudden, on this metric.

#flashcards/research/concept #flashcards/topic/research-llm Write the joint loss law in $N$ and $D$ (equation 1.5) and explain each symbol.
?
$$L(N,D)=\left[\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D}+\frac{D_c}{D}\right]^{\alpha_D}$$
- $D$: dataset size in tokens; $D_c$, $\alpha_D$: data constant and exponent.
- Table 2 fit: $\alpha_N=0.076$, $\alpha_D=0.103$, $N_c=6.4\times10^{13}$, $D_c=1.8\times10^{13}$.
- With unlimited $D$ the second term vanishes and gives $L(N)$; with unlimited $N$ the first vanishes and gives $L(D)$; in between each bottlenecks the other.

#flashcards/research/concept #flashcards/topic/research-llm $L(N,D)$ example: for $N=10^8$, what are the loss and overfitting penalty at $D=10^{10}$ and $D=10^9$ tokens?::Infinite data: 2.762. $D=10^{10}$: 2.788, a 0.93% penalty. $D=10^9$: 2.957, a 7.0% penalty (my arithmetic with the Table 2 constants): ten times less data makes the penalty about 7.6 times bigger.

#flashcards/research/concept #flashcards/topic/research-llm Why does the $L(N,D)$ form have the exponent $\alpha_N/\alpha_D$ on the first term and the power $\alpha_D$ outside?::So that $D\to\infty$ gives exactly $L(N)=(N_c/N)^{\alpha_N}$ and $N\to\infty$ gives exactly $L(D)=(D_c/D)^{\alpha_D}$: $[(N_c/N)^{\alpha_N/\alpha_D}]^{\alpha_D}=(N_c/N)^{\alpha_N}$. The paper chose it on three principles (rescaling with vocabulary, the two limits, and analyticity in $1/D$, the weakest).

#flashcards/research/concept #flashcards/topic/research-llm Write the paper's rule for data needed to avoid overfitting (equation 4.4) and explain each symbol.
?
$$D\gtrsim\left(5\times10^{3}\right)N^{0.74}$$
- $D$: training tokens; $N$: non-embedding parameters.
- 0.74 is $\alpha_N/\alpha_D$ (0.076/0.103); it is below 1, so data may grow more slowly than the model.
- The constant is set by keeping the loss within about 0.02 of converged, the typical variation between random seeds.

#flashcards/research/concept #flashcards/topic/research-llm Data example: what data does a $10^8$-parameter and a $10^9$-parameter model need, and what if the model grows 8 times?::$10^8$: $4.2\times10^9$ tokens; $10^9$: $2.29\times10^{10}$ tokens, about all of WebText2. 8 times the model needs $8^{0.74}=4.7$ times the data (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Why is the data requirement $N^{0.74}$ and not $N^1$?::Because $\alpha_N/\alpha_D=0.076/0.103=0.74$: the loss falls a little faster with data than with parameters, so data can grow slower than model size, 4.7 times for an 8-fold model instead of 8 times. It is the minimum to avoid overfitting, not what makes training compute-efficient.

#flashcards/research/concept #flashcards/topic/research-llm What is the critical batch size, and what is the relation behind it (equations 5.1 and 5.2)?
?
$$\left(\frac{S}{S_{\min}}-1\right)\left(\frac{E}{E_{\min}}-1\right)=1,\qquad B_{\text{crit}}\equiv\frac{E_{\min}}{S_{\min}}$$
- $S$: steps and $E=BS$: examples to reach a fixed loss.
- $S_{\min}$: fewest steps (huge batch); $E_{\min}$: fewest examples (tiny batch).
- $B_{\text{crit}}$: their ratio; at that batch both costs double ($S=2S_{\min}$, $E=2E_{\min}$), the best time and compute trade-off.

#flashcards/research/concept #flashcards/topic/research-llm Critical-batch example: $S_{\min}=1{,}000$ and $E_{\min}=10^8$. What is $B_{\text{crit}}$ and what happens at $10B_{\text{crit}}$?::$B_{\text{crit}}=10^5$. At $B_{\text{crit}}$: 2,000 steps and $2\times10^8$ examples. At $10B_{\text{crit}}$: $S=1.1S_{\min}$ and $E=11E_{\min}$: almost no saving in steps but 11 times the examples (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Write how $B_{\text{crit}}$ depends on the loss (equations 1.4 and 5.3) and explain each symbol.
?
$$B_{\text{crit}}(L)=\frac{B_{*}}{L^{1/\alpha_B}},\qquad B_{*}\approx2\times10^{8}\text{ tokens},\quad\alpha_B\approx0.21$$
- $L$: the target loss; $B_*$, $\alpha_B$: fitted constants (Table 5: $B_*=2.1\times10^8$).
- It depends on the loss, not on model size.
- It diverges as $L\to0$, because the gradient noise is expected to diverge there.

#flashcards/research/concept #flashcards/topic/research-llm $B_{\text{crit}}$ example: what is it at $L=3.0$ and $L=2.0$, and what does a 13% lower loss do?::$L=3.0$: $1.12\times10^6$ tokens; $L=2.0$: $7.7\times10^6$. A 13% lower loss multiplies it by $0.87^{-4.76}=1.94$, the paper's "doubles for every 13% decrease in loss" (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Why is $B_{\text{crit}}$ written with the power $1/\alpha_B$ of the loss?::So that it grows without limit as the loss goes to zero, tracking the gradient noise scale, which is expected to diverge. $1/0.21=4.76$, so a 13% fall in loss gives a factor $0.87^{-4.76}\approx2$ in the batch size.

#flashcards/research/concept #flashcards/topic/research-llm Write the steps and compute adjustments to the critical batch size (equations 5.4 and 5.5) and explain each symbol.
?
$$S_{\min}(S)=\frac{S}{1+B_{\text{crit}}(L)/B},\qquad C_{\min}(C)=\frac{C}{1+B/B_{\text{crit}}(L)}$$
- $S$, $C$: steps and compute actually used at batch size $B$.
- $S_{\min}$, $C_{\min}$: what they would be at a batch far above or below the critical one.
- It turns runs at a non-optimal fixed batch into clean trends.

#flashcards/research/concept #flashcards/topic/research-llm Adjustment example: $B=2^{19}$, $L=3.0$. What are $S_{\min}$ for $10^5$ steps and $C_{\min}/C$?::$B_{\text{crit}}=1.12\times10^6$, so $S_{\min}=S/(1+2.14)=0.318S=31{,}836$ and $C_{\min}=C/(1+0.468)=0.682C$: that run used about 32% too much compute (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Write the loss law in model size and training steps (equation 1.6) and explain each symbol.
?
$$L(N,S_{\min})=\left(\frac{N_c}{N}\right)^{\alpha_N}+\left(\frac{S_c}{S_{\min}}\right)^{\alpha_S}$$
- First term: the loss floor for a model of size $N$; second: the part that falls with training.
- Table 3: $\alpha_N=0.077$, $\alpha_S=0.76$, $N_c=6.5\times10^{13}$, $S_c=2.1\times10^3$.
- It separates "too small a model" from "not trained long enough"; it fails early in training.

#flashcards/research/concept #flashcards/topic/research-llm $L(N,S_{\min})$ example: $N=10^8$ with $S_{\min}=10^4$, $10^5$ and unlimited steps.::First term 2.803. $S_{\min}=10^4$: second term 0.305, $L=3.108$. $S_{\min}=10^5$: 0.056, $L=2.856$. Unlimited: 2.803 (my arithmetic). At $10^5$ steps the model is within 2% of its limit.

#flashcards/research/concept #flashcards/topic/research-llm Why is $\alpha_S=0.76$ much bigger than $\alpha_N=0.077$ in the $L(N,S_{\min})$ law?::Training progress is fast at first: going from $10^4$ to $10^5$ steps cuts the second term from 0.305 to 0.056 (5.5 times), while a 10-fold bigger model only multiplies the first term by $10^{-0.077}=0.84$. The floor is set by the model size.

#flashcards/research/concept #flashcards/topic/research-llm Write how the paper splits a growing compute budget (equations 1.7 and 1.8) and explain each symbol.
?
$$\alpha_C^{\min}=\frac{1}{1/\alpha_S+1/\alpha_B+1/\alpha_N},\qquad N\propto C_{\min}^{\alpha_C^{\min}/\alpha_N},\ B\propto C_{\min}^{\alpha_C^{\min}/\alpha_B},\ S\propto C_{\min}^{\alpha_C^{\min}/\alpha_S},\ D=B\cdot S$$
- $\alpha_C^{\min}$: the exponent of the loss against the best compute.
- $\alpha_S,\alpha_B,\alpha_N$: 0.76, 0.21 and about 0.076.
- Measured: $N\propto C^{0.73}$, $B\propto C^{0.24}$, $S\propto C^{0.03}$, $D\propto C^{0.27}$; most extra compute should go into model size.

#flashcards/research/concept #flashcards/topic/research-llm Allocation example: compute grows 10 times. By how much do model size, batch, steps and data grow, and what does the formula give?::Model $10^{0.73}=5.4$ times, batch $1.7$ times, steps $1.07$ times, data $1.9$ times. The formula with the paper's numbers gives $\alpha_C^{\min}=0.052$ and exponents 0.68, 0.25, 0.07 for $N$, $B$, $S$ (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Why does the allocation formula put most extra compute into model size?::Because $\alpha_N=0.076$ is by far the smallest exponent, so $1/\alpha_N=13.2$ dominates the sum $1/0.76+1/0.21+1/0.076=19.2$ and the share $\alpha_C^{\min}/\alpha_N$ is the largest (0.68 against 0.25 and 0.07; my arithmetic). Model size is the slow-moving factor, so it needs the most growth.

#flashcards/research/concept #flashcards/topic/research-llm Write the loss against optimal compute (equation 1.3) and explain each symbol.
?
$$L(C_{\min})=\left(\frac{C_c^{\min}}{C_{\min}}\right)^{\alpha_C^{\min}},\qquad \alpha_C^{\min}\approx0.050,\quad C_c^{\min}\approx3.1\times10^{8}\text{ PF-days}$$
- $C_{\min}$: the least compute to reach a loss (batch much below critical).
- $C_c^{\min}$: a scale constant with no fundamental meaning; $\alpha_C^{\min}$: the exponent.
- The law the authors expect to extrapolate best (Figure 13 caption).

#flashcards/research/concept #flashcards/topic/research-llm $L(C_{\min})$ example: what are the losses at 1 PF-day and 10^4 PF-days?::$(3.1\times10^8)^{0.050}=2.658$ and $(3.1\times10^4)^{0.050}=1.677$ (my arithmetic): 10,000 times more compute lowers the loss by about 37%. With Figure 13's constant $2.3\times10^8$ they would be 2.62 and 1.65.

#flashcards/research/concept #flashcards/topic/research-llm Why do the compute law and the data law have to meet, and where?::Compute-efficient training uses data growing only as $C^{0.27}$, so once data-limited the loss falls as $C^{-0.03}$ (my arithmetic: $0.27\times0.095$), slower than $C^{-0.050}$. The curves cross at about $C^*\sim10^4$ PF-days, $N^*\sim10^{12}$, $D^*\sim10^{12}$, $L^*\sim1.7$ nats (paper, highly uncertain); my recomputation gives $1.7\times10^4$ PF-days, $1.6\times10^{12}$ parameters, $3.1\times10^{11}$ tokens and 1.63.

#flashcards/research/concept #flashcards/topic/research-llm What is the paper's conjecture about the crossing point?::That it may be where Transformer language models reach maximal performance, with $L^*\sim1.7$ nats per token as a rough estimate of the entropy of text. The authors say the laws must break down at or before it (paper, 6.3).

#flashcards/research/concept #flashcards/topic/research-llm How much does the shape of a Transformer matter at a fixed size?::Very little: aspect ratio can vary by a factor of 40 with only a slight change in loss, and a $(6,\,4288)$ model is within 3% of the $(48,\,1600)$ model. Models with fewer than 2 layers or extreme shapes are exceptions (paper, Figures 5 and 6).

#flashcards/research/concept #flashcards/topic/research-llm How did the paper compare Transformers with LSTMs, and what happened to transfer?::LSTMs match Transformers on early tokens but plateau after about 100 tokens, while Transformers keep improving through the context (Figure 7). On other text (Books, Common Crawl, Wikipedia) the loss improves in parallel with the training distribution with a roughly constant offset (3.2.2).

#flashcards/research/concept #flashcards/topic/research-llm What do the paper's Tables 2 and 3 say, and what is the caution?::Table 2 ($L(N,D)$): $\alpha_N=0.076$, $\alpha_D=0.103$, $N_c=6.4\times10^{13}$, $D_c=1.8\times10^{13}$. Table 3 ($L(N,S)$): $\alpha_N=0.077$, $\alpha_S=0.76$, $N_c=6.5\times10^{13}$, $S_c=2.1\times10^3$. The fits are excellent except for the tiniest datasets and "imperfect" early in training, and the constants differ slightly from the single-factor ones.

#flashcards/research/concept #flashcards/topic/research-llm What are the compute-efficient optimal values in the paper's Table 6, and what check do they pass?::$N_{\text{opt}}=1.3\times10^9\,C^{0.73}$, $B=2.0\times10^6\,C^{0.24}$ tokens, $S_{\min}=5.4\times10^3\,C^{0.03}$, $D_{\text{opt}}=2\times10^{10}\,C^{0.27}$ (C in PF-days). At 1 PF-day $6ND=1.56\times10^{20}$ operations, 1.8 PF-days, about $2C_{\min}$ as the critical-batch rule says (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Why does the paper say training should stop far short of convergence?::For a fixed budget a bigger model reaches a lower loss even before converging. Appendix B.3: stopping about 10% above the converged loss ($\alpha_N/\alpha_S$), instead of about 2%, uses 2.7 times more parameters, 7.7 times fewer steps and 65% less compute (the paper's numbers; my recomputation 2.70, 0.133 and 0.359).

#flashcards/research/concept #flashcards/topic/research-llm Where does the scaling-laws paper contradict itself?::The number of orders of magnitude (seven, six, or eight, six and two); the compute constants ($3.1\times10^8$ and $1.6\times10^7$ in equation 1.3 and Table 5, $2.3\times10^8$ and $2.0\times10^7$ in Figure 13's legend); the compute exponent (0.050, 0.054 and 0.052); and equation 6.7's $4\times10^{10}\,C^{0.26}$ against Table 6's $2\times10^{10}\,C^{0.27}$. Appendix B.4 also cites "Figure X" and "Figure Y".

#flashcards/research/concept #flashcards/topic/research-llm Does the paper's theory for the optimal number of steps match the measurement?::The paper says the formula "closely matches" the measured exponents, but the step exponent is 0.07 by the formula with the paper's own numbers ($0.052/0.76$) and 0.03 measured (possibly zero). Both are small; the model-size exponent is 0.68 by the formula (0.71 with the paper's 0.054) against 0.73 measured.

#flashcards/research/concept #flashcards/topic/research-llm What limits does the scaling-laws paper state?::No solid theory; unreliable far outside the tested range (the compute and data laws must break down); poor fits for tiny datasets; regularisation and data augmentation not explored; context terms left out of $C\approx6NBS$; hyperparameters may be untuned; loss may not mean ability ("more is different"); only one dataset and one architecture family.

#flashcards/research/concept #flashcards/topic/research-llm How did Scaling Laws relate to later work, and what is marked "memory"?::Chinchilla later reported that model size and training tokens should scale equally, revising this paper's advice to put most extra compute into model size. That GPT-3 was shaped by this paper, and today's use of scaling laws to plan runs, are from memory, not checked in this note.

#flashcards/research/concept #flashcards/topic/research-llm How much of the scaling-laws paper did the note read?::The whole main text (pages 1 to 19) and Appendices A, B and C; not Appendix D's supplemental figures. I looked at page 15 (Figures 12 and 13) as a picture and know the other figures from captions. Every "mine" number was recomputed in Python; no experiment was re-run.
