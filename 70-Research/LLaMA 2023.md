---
tags: [paper, NLP, LLM, open]
link: https://arxiv.org/abs/2302.13971
authors: Touvron, Lavril, Izacard, Martinet, Lachaux, Lacroix, Rozière, Goyal, Hambro, Azhar, Rodriguez, Joulin, Grave, Lample (Meta AI; checked on the paper's first page)
year: 2023
status: body-read
field: text / LLM
---

# LLaMA 2023

> **In one line for a 12-year-old:** A family of language models, trained for much longer on public text than the usual advice says, so that a small one is as good as a giant and more people can run it.

Paper: "LLaMA: Open and Efficient Foundation Language Models", arXiv 2302.13971, version 1 of 27 February 2023, 27 pages. Marks: "(paper, section)" was read in the paper; "(general knowledge)" is my own background; "(mine)" is my own arithmetic, done in Python. This paper's arXiv licence is CC BY 4.0, so its own figures are used below, cropped from the PDF, with credit to the authors (Touvron and others).

## The problem
Large language models were getting better by getting bigger, "based on the assumption that more parameters will lead to better performance" (paper, 1). But the best ones were closed: GPT-3, Chinchilla and PaLM were trained on data that is "either not publicly available or undocumented" (paper, 1). And the compute-optimal advice of [[Chinchilla 2022]] only asks how to spend a training budget; it "disregards the inference budget, which becomes critical when serving a language model at scale" (paper, 1). The aim of the paper is "a series of language models that achieve the best possible performance at various inference budgets, by training on more tokens than what is typically used" (paper, 1).

## What people did before
- **Very big models:** [[GPT-3 2020]] (175B), Gopher (280B), PaLM (540B), trained mostly on closed data (paper, 1 and 7).
- **Chinchilla's scaling result:** for a given compute budget, the best results come not from the largest models but from smaller models trained on more data (paper, 1). It recommends, for example, "training a 10B model on 200B tokens" (paper, 1). See [[Scaling Laws]].
- **Open models:** OPT, GPT-NeoX, BLOOM and GLM were open, "but none that are competitive with PaLM-62B or Chinchilla" (paper, 1).

## The idea
Pick the model that is cheapest to **run**, not cheapest to train. "Although it may be cheaper to train a large model to reach a certain level of performance, a smaller one trained longer will ultimately be cheaper at inference" (paper, 1). So train small models for far longer than Chinchilla advises: "the performance of a 7B model continues to improve even after 1T tokens" (paper, 1). Use only public data so everything can be released, and borrow the best known tweaks to the Transformer (pre-normalisation, SwiGLU, rotary embeddings). Result: LLaMA, from 7B to 65B parameters.

## The architecture and training, part by part
The paper has no architecture figure: LLaMA is the [[Transformer]] decoder with three changes and a long training run, so there is nothing to redraw (I looked on Wikimedia Commons and found no free picture of LLaMA's own layers; the vault's note on [[Attention Is All You Need 2017]] shows the original Transformer). The paper's own two figures, both about training, are shown below.

- **The data mixture.**
  - *What it does:* seven public sources (table below), 1.4 trillion tokens after tokenising (paper, 2.1).
  - *Why it matters:* only publicly available data compatible with open-sourcing is used, which is what lets the models be released.
- **The tokenizer.**
  - *What it does:* byte-pair encoding with SentencePiece; "we split all numbers into individual digits, and fallback to bytes to decompose unknown UTF-8 characters" (paper, 2.1). See [[Tokenization]].
  - *Why it matters:* digit splitting makes numbers regular, and the byte fallback means no character is ever "unknown".
- **Pre-normalisation (with RMSNorm).**
  - *What it does:* "we normalize the input of each transformer sub-layer, instead of normalizing the output", using the RMSNorm function (paper, 2.2; idea from GPT-3). See [[Layer Normalization]].
  - *Why it matters:* "to improve the training stability" (paper, 2.2).
- **SwiGLU activation.**
  - *What it does:* replaces the ReLU in the feed-forward block with SwiGLU, using a hidden size of $\frac23 4d$ instead of $4d$ (paper, 2.2; idea from PaLM).
  - *Why it matters:* "to improve the performance"; the $\frac23$ keeps the number of parameters the same (equation 2, example below).
- **Rotary embeddings (RoPE).**
  - *What it does:* removes the absolute position embeddings and instead adds rotary positional embeddings "at each layer of the network" (paper, 2.2; idea from GPT-Neo). See [[RoPE]].
  - *Why it matters:* position enters through the relative distance between tokens, in every layer.
- **The optimizer.**
  - *What it does:* AdamW with $\beta_1=0.9$, $\beta_2=0.95$, a cosine learning-rate schedule ending at 10% of the maximum, weight decay 0.1, gradient clipping 1.0 and 2,000 warm-up steps (paper, 2.3). See [[Adam Optimizer]].
  - *Why it matters:* a standard, stable recipe for long runs.
- **Efficient implementation.**
  - *What it does:* causal multi-head attention without storing attention weights or computing masked scores (from the xformers library, using the backward pass of FlashAttention), fewer recomputed activations, model and sequence parallelism, and overlapping computation with communication (paper, 2.4). See [[FlashAttention]].
  - *Why it matters:* the 65B model "processes around 380 tokens/sec/GPU on 2048 A100 GPU", so training on 1.4T tokens takes "approximately 21 days" (paper, 2.4). My check: $1.4\times10^{12}/(380\times2048)$ seconds is 20.8 days.

### The data mixture (paper's Table 1)

**Table 1: pre-training data** (paper's Table 1; "epochs" are for the 1.4T-token runs; the last column is mine, computed in Python from the first two columns and the 1.4T total).

| Dataset | Sampling share | Epochs | Disk size | Unique tokens, billions (mine) |
|---|---|---|---|---|
| CommonCrawl | 67.0% | 1.10 | 3.3 TB | about 853 |
| C4 | 15.0% | 1.06 | 783 GB | about 198 |
| Github | 4.5% | 0.64 | 328 GB | about 98 |
| Wikipedia | 4.5% | 2.45 | 83 GB | about 26 |
| Books | 4.5% | 2.23 | 85 GB | about 28 |
| ArXiv | 2.5% | 1.06 | 92 GB | about 33 |
| StackExchange | 2.0% | 1.03 | 78 GB | about 27 |

Written by me, an AI, from the paper's table text; no rows left out.

What each column is, and why it matters:
- **Dataset:** the source: English CommonCrawl processed by the CCNet pipeline (line-level deduplication, language identification, an n-gram language-model quality filter, plus a classifier for Wikipedia-like references); C4 (another cleaned web crawl); public GitHub projects under the Apache, BSD and MIT licences; Wikipedia in 20 languages (June to August 2022); Books (Gutenberg and Books3); ArXiv LaTeX; and Stack Exchange (28 largest sites) (paper, 2.1).
- **Sampling share:** the percentage of training examples drawn from this source. It sums to 100% (mine). It matters because web text dominates, 82% together for CommonCrawl and C4, while code, books, science and Q and A are small slices.
- **Epochs:** how many passes over each source during the 1.4T-token run. Most are about 1; Wikipedia and Books are seen about 2.2 to 2.5 times and Github only 0.64 times. The text says each token is used once "with the exception of the Wikipedia and Books domains" (paper, 2.1).
- **Disk size:** the size of the source's data on disk.
- **Unique tokens (mine):** $1400\text{B}\times\text{share}/\text{epochs}$, roughly how many distinct tokens a source contributes. They add up to about 1,263 billion, so about 1.26T distinct tokens are stretched into 1.4T by repeating a little.

**Summary: what the table shows.**
- **Biggest:** CommonCrawl, 67% of the samples and about 853 billion distinct tokens (mine).
- **Smallest by share:** StackExchange, 2%, and ArXiv, 2.5%. Books, ArXiv and Gutenberg total only 177 GB on disk (85 plus 92, mine; the paper says "177GB" in 3.6).
- **Caution:** the paper gives the token counts only as a total ("roughly 1.4T"); my per-source token counts are derived and approximate.

### The model sizes (paper's Table 2)

**Table 2: model sizes, architectures and optimisation hyper-parameters** (paper's Table 2; the last two columns are mine).

| Params | Dimension | Heads | Layers | Learning rate | Batch size | Tokens | Head size (mine) | Tokens per parameter (mine) |
|---|---|---|---|---|---|---|---|---|
| 6.7B | 4096 | 32 | 32 | 3.0e-4 | 4M | 1.0T | 128 | 149 |
| 13.0B | 5120 | 40 | 40 | 3.0e-4 | 4M | 1.0T | 128 | 77 |
| 32.5B | 6656 | 52 | 60 | 1.5e-4 | 4M | 1.4T | 128 | 43 |
| 65.2B | 8192 | 64 | 80 | 1.5e-4 | 4M | 1.4T | 128 | 21.5 |

Written by me, an AI, from the paper's table text, which I checked against the page picture; no rows left out. The paper's text and abstract call the models 7B, 13B, 33B and 65B; the table gives the exact counts above.

What each column is, and why it matters:
- **Params:** the number of weights. **Dimension:** the width $d$ of the model (what the Transformer papers call $d_\text{model}$). **Heads:** attention heads. **Layers:** the number of Transformer blocks.
- **Learning rate / Batch size:** the peak learning rate and the batch of 4 million tokens per step (paper, 2.3 and Figure 1). The larger models use half the learning rate.
- **Tokens:** how many tokens each model saw: 1.0T for the two small models and 1.4T for the two big ones (paper, Figure 1).
- **Head size (mine):** dimension divided by heads: 128 for every size, so each head has the same width and bigger models simply have more of them.
- **Tokens per parameter (mine):** tokens divided by parameters. It matters because Chinchilla's advice was about 20 tokens per parameter (10B on 200B tokens, as the paper quotes). LLaMA's 65B is at about 21.5, near that advice; the 7B is at 149, about 7 times past it.

**Summary: what the table shows.**
- **A regular family:** heads of 128 at every size; depth and width grow together.
- **The big idea in one column:** the smaller the model, the more tokens per parameter: 149, 77, 43, 21.5. The small ones were deliberately over-trained for cheap serving.
- **Check (mine):** a Transformer block has about $12d^2$ weights (attention $4d^2$ plus a feed-forward block $8d^2$), so $12\times L\times d^2$ gives 6.44B, 12.58B, 31.9B and 64.4B for the four rows, within 0.3 to 0.8 billion of the table (the rest is the embedding and output layers, which I did not size because the vocabulary size is not given in this paper).

### The training curves (paper's Figure 1)

![[llama-fig1-training-loss.png]]
*The paper's Figure 1: training loss against the number of training tokens, in billions, for the 7B, 13B, 33B and 65B models. The 33B and 65B models were trained on 1.4T tokens; the smaller ones on 1.0T; all with a batch size of 4M tokens. Figure: Touvron, Lavril, Izacard and others, from [arXiv 2302.13971](https://arxiv.org/abs/2302.13971), CC BY 4.0, cropped from the paper's page 3 by me.*

**What the figure shows, and why it matters**
- **Axes:** the horizontal axis is how many tokens the model has read so far (0 to 1,400 billion); the vertical is the training loss (lower is better), from 1.5 to 2.2. A line that is still sloping down at the end is a model that has not stopped learning.
- **The four lines:** blue 7B, orange 13B, green 33B and red 65B. The lines are ordered by size at every point, and each stops where its training stopped (1,000 or 1,400 billion tokens).
- **Spikes:** a few vertical spikes (for example in the 33B and 65B curves) are brief jumps in the loss that the curve recovers from (my reading; the paper does not comment on them).

**Summary: what the figure shows.**
- **Bigger is lower, at every point:** at the end the 65B model is about 1.56 and the 7B about 1.80 (my reading of the plot).
- **Not finished:** all four curves are still going down when training stops, which supports the paper's claim that 7B "continues to improve even after 1T tokens" (paper, 1).
- **Caution:** the plot is training loss, not a test score, so it does not show how well the model does on tasks; the next figure does.

## The core mechanism, step by step

**From public text to a released model (my numbered paraphrase of paper, 2).**
1. **Collect public data:** web crawls, code, Wikipedia, books, papers and Q and A, filtered and deduplicated (Table 1).
2. **Tokenize:** byte-pair encoding; numbers become single digits; unknown characters fall back to bytes.
3. **Build the Transformer decoder:** each block normalises its input first (RMSNorm), applies causal multi-head attention with rotary positions, adds the result back, then repeats the same pattern with a SwiGLU feed-forward block.
4. **Train with next-word prediction:** AdamW, a batch of 4M tokens, 2,000 warm-up steps, a cosine schedule. For 1.4T tokens at 4M per step that is 350,000 steps (mine).
5. **Keep training longer than the "optimal" amount,** because the target is cheap inference, not cheap training.
6. **Evaluate** zero-shot and few-shot on 20 benchmarks, and check bias, toxicity and truthfulness (section 3 and 5).
7. **Release** the models to the research community.

**How multiple-choice answers are scored (paper, 3).** The model is not asked to write the letter. Each possible completion is scored by its likelihood and the best is picked. Usually the likelihood is divided by the number of characters in the completion, but for OpenBookQA and BoolQ it is divided by the likelihood of the same completion after the context "Answer:" (equation 4 below).

## The equations
The paper prints two kinds of formula: the likelihood score for multiple-choice answers and the energy and carbon formulas. The three architecture changes are named, but their formulas are in the papers it cites; I give them here so the architecture can be followed. Equations 1 to 3 are therefore marked "(general knowledge, from the cited papers, not printed in LLaMA)".

**1. RMSNorm, used in pre-normalisation** (general knowledge, from Zhang and Sennrich 2019, which LLaMA cites in 2.2)

$$
\mathrm{RMSNorm}(\mathbf{x})_i=\frac{x_i}{\mathrm{RMS}(\mathbf{x})}\,g_i,\qquad \mathrm{RMS}(\mathbf{x})=\sqrt{\frac1n\sum_{j=1}^{n}x_j^{2}}
$$

**Where:**
- $\mathbf{x}$: the vector of $n$ numbers for one token (its activations).
- $\mathrm{RMS}(\mathbf{x})$: the root mean square, a typical size of the entries.
- $g_i$: a learned scale for each entry.
- Unlike layer normalisation there is no mean subtracted and no bias.

**Matters:** it divides every entry by the same size so the numbers going into each layer stay in a steady range, which "improves training stability" (paper, 2.2). It is cheaper than layer normalisation because it skips the mean. In pre-normalisation it is applied to the input of each sub-layer rather than its output.

**Example (tiny numbers):** $\mathbf{x}=(3,4)$, $g=(1,1)$.
- $\mathrm{RMS}=\sqrt{(9+16)/2}=3.536$.
- Output: $(3/3.536,\ 4/3.536)=(0.849,\ 1.131)$.
- Scaling the input by 50 to $(150,200)$ gives the same output; the input $(100,200)$ gives $(0.632,1.265)$.
- Meaning: the output depends on the shape of the vector, not its overall size.

**2. SwiGLU feed-forward block and its size** (general knowledge, from Shazeer 2020; the paper prints only "$\frac23 4d$")

$$
\mathrm{FFN}_{\mathrm{SwiGLU}}(\mathbf{x})=\big(\mathrm{Swish}(\mathbf{x}W_1)\odot \mathbf{x}W_3\big)W_2,\qquad \mathrm{Swish}(z)=z\,\sigma(z),\qquad d_{\text{hidden}}=\tfrac23\cdot4d
$$

**Where:**
- $\mathbf{x}$: the token's vector of width $d$; $W_1,W_3$: two matrices that widen it to $d_\text{hidden}$; $W_2$: a matrix that narrows it back to $d$.
- $\sigma(z)=1/(1+e^{-z})$: the sigmoid; $\mathrm{Swish}(z)=z\sigma(z)$ is a smooth version of ReLU.
- $\odot$: multiply entry by entry. One branch is a "gate" for the other.
- $d_\text{hidden}=\frac23\cdot4d$: the width LLaMA picks (paper, 2.2), rather than the usual $4d$.

**Matters:** the gate lets the network switch features on and off, which the cited work found improves quality. It has three matrices instead of two, so the hidden width is cut to two thirds to keep the weights the same. Without the $\frac23$ the feed-forward blocks would hold 50% more weights.

**Example (tiny numbers):**
- Gate: $z=2$ gives $\mathrm{Swish}(2)=2\times0.881=1.762$; if the other branch gives $0.5$, the product is $0.881$. $z=-2$ gives $-0.238$ (ReLU would give $0$).
- Size at $d=4096$: a ReLU block has $2\times d\times4d=134{,}217{,}728$ weights. A SwiGLU block with hidden width $\frac23\cdot4d=10{,}922.7$ has $3\times d\times\frac{8d}{3}=134{,}217{,}728$.
- Meaning: the same number of weights, with a gate. (The real code rounds the width to a whole number, which the paper does not say.)

**3. Rotary position embedding (RoPE)** (general knowledge, from Su et al. 2021, which LLaMA cites in 2.2)

$$
\begin{pmatrix}x_1'\\x_2'\end{pmatrix}=\begin{pmatrix}\cos m\theta&-\sin m\theta\\ \sin m\theta&\cos m\theta\end{pmatrix}\begin{pmatrix}x_1\\x_2\end{pmatrix},\qquad \mathbf{q}_m^{\top}\mathbf{k}_n\ \text{depends only on}\ m-n
$$

**Where:**
- $(x_1,x_2)$: one pair of entries of a query or key vector; every pair is rotated separately.
- $m$: the position of the token in the sentence.
- $\theta$: a rotation angle per step, different for each pair (the cited paper uses $10000^{-2i/d}$, so some pairs turn quickly and some slowly).
- $\mathbf{q}_m,\mathbf{k}_n$: the query at position $m$ and the key at position $n$.

**Matters:** rotating the query and key by their positions makes their dot product depend only on the distance $m-n$ between the tokens, not on where they are in the text, so the attention score knows "how far apart" without a learned table of absolute positions. LLaMA applies it "at each layer" (paper, 2.2), not just at the input. See [[RoPE]].

**Example (tiny numbers):** $\theta=30^\circ$ per position; query and key both start as $(1,0)$.
- Positions $(m,n)=(2,1)$: dot product $\cos30^\circ=0.866$. Positions $(5,4)$: also $0.866$.
- Positions $(10,4)$ (6 apart, 180 degrees): $-1.0$. Same position $(7,7)$: $1.0$.
- Meaning: the score depends on the gap, not on where the pair sits.

**4. The completion score for multiple-choice tasks** (paper, 3)

$$
\mathrm{score}(c)=\frac{P(c\mid \text{context})}{P(c\mid\text{"Answer:"})}
$$

**Where:**
- $c$: one possible completion (answer option).
- $P(c\mid\text{context})$: the probability the model gives to the completion after the question's context.
- $P(c\mid\text{"Answer:"})$: the probability of the same completion after only the word "Answer:", with no context.
- The paper uses this for OpenBookQA and BoolQ; elsewhere it divides the likelihood by the number of characters in the completion (following Gao and others).

**Matters:** raw likelihood favours short, common answers however the question reads. Dividing by how likely the answer is anyway makes the score measure how much the context helped. Without it the model would be rewarded for a likely-sounding answer, not a right one.

**Example (tiny numbers):** two options with $P(c\mid\text{context})=0.02$ and $0.01$, and $P(c\mid\text{"Answer:"})=0.04$ and $0.005$.
- Scores: $0.02/0.04=0.5$ and $0.01/0.005=2.0$.
- Raw likelihood picks option 1; the ratio picks option 2.
- Meaning: option 2 is rare in general but the question made it far more likely, so the ratio rates it higher.

**5. Energy and carbon estimates** (paper, 6)

$$
\mathrm{Wh}=\text{GPU-hours}\times(\text{GPU power})\times\mathrm{PUE},\qquad \mathrm{tCO_2eq}=\mathrm{MWh}\times0.385
$$

**Where:**
- GPU-hours: number of GPUs times hours used.
- GPU power: the thermal design power of an A100-80GB, 400 W (paper, Table 15).
- PUE: power usage effectiveness, set at 1.1, the extra energy for cooling and the building.
- 0.385: kilograms of CO2 per kilowatt-hour, the US average the paper chose; so tonnes of CO2 equivalent per megawatt-hour.

**Matters:** it puts the energy cost of training on one scale so models can be compared "if they were trained in the same data center" (paper, 6). A different grid changes the carbon number a lot, which is why the paper cites BLOOM's and OPT's own grids (27 and 82 tonnes) but does not use them.

**Example (tiny numbers):** the 65B model took 1,022,362 GPU-hours (Table 15).
- Energy: $1{,}022{,}362\times400\ \text{W}\times1.1=449.8$ MWh.
- Carbon: $449.8\times0.385=173.2$ tonnes.
- The 7B model: $82{,}432\times400\times1.1=36.3$ MWh and $14.0$ tonnes.
- Meaning: the 65B run cost about 12.4 times the GPU-hours of the 7B (mine).

## What they found
LLaMA is tested on 20 benchmarks, zero-shot (a description of the task and a test example) and few-shot (1 to 64 examples), against GPT-3, Gopher, Chinchilla and PaLM (not open) and OPT, GPT-J and GPT-Neo (open) (paper, 3). The comparison numbers for the others are taken from their papers.

### Common sense (paper's Table 3)

**Table 3: zero-shot performance on common-sense reasoning tasks** (paper's Table 3; accuracy in percent; "-" means the paper has no number; I left out the PaLM 62B, PaLM-cont and OPT rows).

| Model | BoolQ | PIQA | SIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA |
|---|---|---|---|---|---|---|---|---|
| GPT-3 175B | 60.5 | 81.0 | - | 78.9 | 70.2 | 68.8 | 51.4 | 57.6 |
| Gopher 280B | 79.3 | 81.8 | 50.6 | 79.2 | 70.1 | - | - | - |
| Chinchilla 70B | 83.7 | 81.8 | 51.3 | 80.8 | 74.9 | - | - | - |
| PaLM 540B | 88.0 | 82.3 | - | 83.4 | 81.1 | 76.6 | 53.0 | 53.4 |
| LLaMA 7B | 76.5 | 79.8 | 48.9 | 76.1 | 70.1 | 72.8 | 47.6 | 57.2 |
| LLaMA 13B | 78.1 | 80.1 | 50.4 | 79.2 | 73.0 | 74.8 | 52.7 | 56.4 |
| LLaMA 33B | 83.1 | 82.3 | 50.4 | 82.8 | 76.0 | 80.0 | 57.8 | 58.6 |
| LLaMA 65B | 85.3 | 82.8 | 52.3 | 84.2 | 77.0 | 78.9 | 56.0 | 60.2 |

Written by me, an AI, from the paper's table text; left out: PaLM 62B, PaLM-cont and the other PaLM rows.

What each column is, and why it matters:
- **BoolQ:** yes/no questions about a passage. **PIQA:** physical common sense (which way of doing a thing works). **SIQA:** social situations. **HellaSwag:** choose the most sensible ending of a story. **WinoGrande:** resolve who "it" or "he" refers to in a tricky sentence. **ARC-e / ARC-c:** grade-school science questions, easy and challenge sets. **OBQA:** OpenBookQA, science facts from an open book (general knowledge for all eight).
- **Zero-shot accuracy:** the share of questions where the highest-scoring option is right (equation 4). Higher is better; random guessing is far lower. It matters because it tests the model with no examples.
- **Model and size:** the paper's point is that size is not everything: compare LLaMA 13B (13B) with GPT-3 (175B).

**Summary: what the table shows.**
- **13B against GPT-3:** LLaMA-13B is higher on 5 of the 7 benchmarks where GPT-3 has a number (BoolQ by 17.6 points, ARC-e by 6.0; lower on PIQA by 0.9 and OBQA by 1.2), at 13.5 times fewer parameters (mine).
- **65B against Chinchilla:** higher on all 5 benchmarks where Chinchilla has a number (BoolQ 85.3 against 83.7).
- **65B against PaLM 540B:** higher on 5 of 7; lower on BoolQ (by 2.7) and WinoGrande (by 4.1). PaLM is 8.3 times larger.
- **A mismatch in the paper:** the text says "LLaMA-65B outperforms Chinchilla-70B on all reported benchmarks but BoolQ", but the table shows LLaMA-65B ahead on BoolQ (85.3 against 83.7); I use the table.
- **Caution:** the competitor numbers come from other papers with possibly different protocols (the paper says so for the toxicity tests); small gaps such as 0.3 points on HellaSwag are within noise (general knowledge).

### Closed-book question answering (paper's Tables 4 and 5)

**Table 4: NaturalQuestions, exact-match accuracy** (paper's Table 4; "-" means not reported; some rows left out).

| Model | 0-shot | 1-shot | 5-shot | 64-shot |
|---|---|---|---|---|
| GPT-3 175B | 14.6 | 23.0 | - | 29.9 |
| Gopher 280B | 10.1 | - | 24.5 | 28.2 |
| Chinchilla 70B | 16.6 | - | 31.5 | 35.5 |
| PaLM 540B | 21.2 | 29.3 | - | 39.6 |
| LLaMA 7B | 16.8 | 18.7 | 22.0 | 26.1 |
| LLaMA 13B | 20.1 | 23.4 | 28.1 | 31.9 |
| LLaMA 33B | 24.9 | 28.3 | 32.9 | 36.0 |
| LLaMA 65B | 23.8 | 31.0 | 35.0 | 39.9 |

**Table 5: TriviaQA, exact-match accuracy on the filtered dev set** (paper's Table 5).

| Model | 0-shot | 1-shot | 5-shot | 64-shot |
|---|---|---|---|---|
| Gopher 280B | 43.5 | - | 57.0 | 57.2 |
| Chinchilla 70B | 55.4 | - | 64.1 | 64.6 |
| LLaMA 7B | 50.0 | 53.4 | 56.3 | 57.6 |
| LLaMA 13B | 56.6 | 60.5 | 63.1 | 64.0 |
| LLaMA 33B | 65.1 | 67.9 | 69.9 | 70.4 |
| LLaMA 65B | 68.2 | 71.6 | 72.6 | 73.0 |

Written by me, an AI, from the paper's table text. Left out of Table 4: the PaLM 8B and 62B rows (and for Table 5 the paper has no GPT-3 or PaLM rows).

What each column is, and why it matters:
- **Closed-book:** the model gets the question but no documents, so the answer must come from what it memorised in training. NaturalQuestions are real search-engine questions; TriviaQA are trivia questions (general knowledge).
- **Exact match:** the percentage of answers that match a correct answer exactly after normalisation (general knowledge). Higher is better.
- **0-shot to 64-shot:** the number of worked examples placed in the prompt. More examples usually help, at the cost of a longer prompt.

**Summary: what the tables show.**
- **LLaMA-65B is first or tied on both:** NaturalQuestions: 23.8 against Chinchilla's 16.6 at 0-shot (+7.2, mine), 39.9 against 35.5 at 64-shot (+4.4), and 0.3 above PaLM-540B's 39.6 (not a meaningful gap). TriviaQA: 68.2 against 55.4 at 0-shot (+12.8) and 73.0 against 64.6 at 64-shot (+8.4).
- **13B vs GPT-3 and Chinchilla:** on NaturalQuestions 13B beats GPT-3 at 0-shot by 5.5 and at 64-shot by 2.0 (mine); on TriviaQA it is at Chinchilla's level (+1.2 at 0-shot, -1.0 at 5-shot).
- **Caution:** the 33B is higher than the 65B on 0-shot NaturalQuestions (24.9 against 23.8); and these tests reward memorised facts, which a bigger model holds more of.

### Reading, maths and code (paper's Tables 6 to 8)
The reading test, RACE (paper's Table 6), is mentioned only in numbers: LLaMA-65B scores 67.9 on RACE-middle and 51.6 on RACE-high; PaLM-540B scores 68.1 and 49.1; LLaMA-13B 61.6 and 47.2 against GPT-3's 58.4 and 45.5 (paper, 3.3).

**Table 7: quantitative reasoning** (paper's Table 7; accuracy in percent; "maj1@k" is majority voting over many samples; some rows left out).

| Model | MATH | MATH + maj1@k | GSM8k | GSM8k + maj1@k |
|---|---|---|---|---|
| PaLM 62B | 4.4 | - | 33.0 | - |
| PaLM 540B | 8.8 | - | 56.5 | - |
| Minerva 62B | 27.6 | 43.4 | 52.4 | 68.5 |
| Minerva 540B | 33.6 | 50.3 | 68.5 | 78.5 |
| LLaMA 7B | 2.9 | 6.9 | 11.0 | 18.1 |
| LLaMA 13B | 3.9 | 8.8 | 17.8 | 29.3 |
| LLaMA 33B | 7.1 | 15.2 | 35.6 | 53.1 |
| LLaMA 65B | 10.6 | 20.5 | 50.9 | 69.7 |

**Table 8: code generation** (paper's Table 8; pass@k in percent; the starred PaLM numbers were read from figures in another paper; some rows left out).

| Model | HumanEval pass@1 | HumanEval pass@100 | MBPP pass@1 | MBPP pass@80 |
|---|---|---|---|---|
| LaMDA 137B | 14.0 | 47.3 | 14.8 | 62.4 |
| PaLM 62B | 15.9 | 46.3* | 21.4 | 63.2* |
| PaLM 540B | 26.2 | 76.2 | 36.8 | 75.0 |
| LLaMA 7B | 10.5 | 36.5 | 17.7 | 56.2 |
| LLaMA 13B | 15.8 | 52.5 | 22.0 | 64.0 |
| LLaMA 33B | 21.7 | 70.7 | 30.2 | 73.4 |
| LLaMA 65B | 23.7 | 79.3 | 37.7 | 76.8 |

Written by me, an AI, from the paper's table text. Left out of Table 7: the PaLM 8B and Minerva 8B rows; left out of Table 8: PaLM 8B and PaLM-cont 62B.

What each column is, and why it matters:
- **MATH:** about 12,000 middle and high-school maths problems written in LaTeX; **GSM8k:** middle-school word problems (paper, 3.4). Neither PaLM nor LLaMA is fine-tuned on maths; Minerva is PaLM fine-tuned on 38.5B tokens of arXiv and maths web pages (paper, 3.4).
- **maj1@k:** generate $k$ answers (256 for MATH and 100 for GSM8k for LLaMA) and take the most common one. It matters because voting makes weak models look much better, so it must be compared like with like.
- **HumanEval and MBPP:** write a Python program from a description and pass tests (general knowledge). **pass@1** is the chance the first answer passes (sampling temperature 0.1); **pass@100 or pass@80** is the chance that at least one of 100 or 80 samples passes (temperature 0.8) (paper, 3.5).

**Summary: what the tables show.**
- **Maths:** LLaMA-65B reaches 50.9 on GSM8k, below Minerva 62B's 52.4 without voting (1.5 lower, mine) but above it with voting, 69.7 against 68.5. The paper's sentence "LLaMA-65B outperforms Minerva-62B" is therefore only true with voting. On MATH, Minerva-62B (27.6) is far ahead of LLaMA-65B (10.6).
- **Code:** LLaMA-13B beats the much larger LaMDA 137B on both tests (+1.8 and +7.2 pass@1, mine); LLaMA-65B beats PaLM 62B by 7.8 and 16.3 points pass@1, and is 2.5 points below PaLM 540B on HumanEval pass@1 but 0.9 above on MBPP.
- **Caution:** the paper itself notes that models fine-tuned on code do better, for example PaLM-Coder's 36% on HumanEval, and that this is "beyond the scope of this paper".

### Knowledge across subjects (paper's Tables 9 and 10)

**Table 9: massive multitask language understanding (MMLU), five-shot accuracy** (paper's Table 9; some rows left out).

| Model | Humanities | STEM | Social Sciences | Other | Average |
|---|---|---|---|---|---|
| GPT-NeoX 20B | 29.8 | 34.9 | 33.7 | 37.7 | 33.6 |
| GPT-3 175B | 40.8 | 36.7 | 50.4 | 48.8 | 43.9 |
| Gopher 280B | 56.2 | 47.4 | 71.9 | 66.1 | 60.0 |
| Chinchilla 70B | 63.6 | 54.9 | 79.3 | 73.9 | 67.5 |
| PaLM 540B | 77.0 | 55.6 | 81.0 | 69.6 | 69.3 |
| LLaMA 7B | 34.0 | 30.5 | 38.3 | 38.1 | 35.1 |
| LLaMA 13B | 45.0 | 35.8 | 53.8 | 53.3 | 46.9 |
| LLaMA 33B | 55.8 | 46.0 | 66.7 | 63.4 | 57.8 |
| LLaMA 65B | 61.8 | 51.7 | 72.9 | 67.4 | 63.4 |

**Table 10: instruction fine-tuning, MMLU five-shot average** (paper's Table 10; some rows left out).

| Model | MMLU average |
|---|---|
| Chinchilla 70B | 67.5 |
| LLaMA 65B | 63.4 |
| Flan-PaLM 62B | 59.6 |
| Flan-PaLM-cont 62B | 66.1 |
| LLaMA-I 65B | 68.9 |

Written by me, an AI, from the paper's table text. Left out of Table 9: PaLM 8B and 62B; left out of Table 10: OPT, GLM, PaLM, OPT-IML-Max and Flan-T5-XXL.

What each column is, and why it matters:
- **MMLU:** multiple-choice questions on 57 subjects, grouped into humanities, STEM, social sciences and other (general knowledge); "five-shot" means five worked examples in the prompt (paper, 3.6).
- **Average:** the paper's overall score. It matters because it is the standard summary of broad knowledge.
- **LLaMA-I:** LLaMA-65B fine-tuned briefly on instructions, "a single experiment following the same protocol as Chung et al. (2022)" (paper, 4). "Flan" models are PaLM fine-tuned the same way.

**Summary: what the tables show.**
- **Base models:** LLaMA-65B (63.4) beats GPT-3 (43.9) by 19.5 points and Gopher (60.0) by 3.4, but is behind Chinchilla (67.5, -4.1) and PaLM-540B (69.3, -5.9) (mine).
- **Humanities is where the gap is largest:** LLaMA-65B 61.8 against PaLM-540B's 77.0.
- **The paper's explanation:** a "potential explanation" is that LLaMA used only 177 GB of books and academic papers (ArXiv, Gutenberg, Books3), while the others "were trained on up to 2TB of books" (paper, 3.6). It is a guess, not a test.
- **Instruction tuning helps a lot:** LLaMA-I 68.9 is 5.5 points above plain LLaMA-65B and 1.4 above Chinchilla (mine), but still 8.5 below the 77.4 of GPT code-davinci-002, which the paper calls "the state-of-the-art" (mine).
- **Caution:** the numbers for other models come from their own papers.

### During training (paper's Figure 2)

![[llama-fig2-performance-during-training.png]]
*The paper's Figure 2: accuracy on six question-answering and common-sense benchmarks (TriviaQA, HellaSwag, NaturalQuestions, SIQA, WinoGrande, PIQA) against billions of training tokens, for the four LLaMA sizes, with the final score of Chinchilla as a dashed line. Figure: Touvron, Lavril, Izacard and others, from [arXiv 2302.13971](https://arxiv.org/abs/2302.13971), CC BY 4.0, cropped from the paper's page 8 by me.*

**What the figure shows, and why it matters**
- **Axes:** each small plot has training tokens (0 to 1,500 billion) on the horizontal axis and accuracy on the vertical axis. A curve that rises and then flattens is a model that has learned most of what it can from this data.
- **The lines:** blue 7B, orange 13B, green 33B and red 65B; the dashed purple line is the score of Chinchilla (a fixed score for comparison, not a training curve).
- **Why it matters:** it shows the benefit of long training for each size, and where small models cross the line of a model 5 to 10 times larger.

**Summary: what the figure shows.**
- **Steady gains:** on most tasks accuracy keeps rising with tokens and tracks the training loss (paper, 3.7).
- **Beating Chinchilla:** on TriviaQA, the 13B (orange) and 33B (green) curves reach or pass the dashed line near the end, and the 65B is well above it; on HellaSwag only the 33B and 65B pass it (my reading of the plot).
- **Exceptions:** SIQA is very noisy ("a lot of variance", which the paper says "may indicate that this benchmark is not reliable"), and on WinoGrande the 33B and 65B end up similar (paper, 3.7).
- **Caution:** I read the plot by eye; the exact points are not in a table.

### Bias, toxicity and truthfulness (paper's Tables 11 to 14)

**Table 11: RealToxicityPrompts, average toxicity score** (paper's Table 11; scores from 0, non-toxic, to 1, toxic; higher is more toxic; the paper's caption says "PerplexityAPI" but the text and footnote say PerspectiveAPI, so that is a typo in the caption).

| Model | Basic prompts | Respectful prompts |
|---|---|---|
| LLaMA 7B | 0.106 | 0.081 |
| LLaMA 13B | 0.104 | 0.095 |
| LLaMA 33B | 0.107 | 0.087 |
| LLaMA 65B | 0.128 | 0.141 |

**Table 12: CrowS-Pairs, share of pairs where the model prefers the stereotype (higher means more biased)** (paper's Table 12; LLaMA is the 65B model, the others are 175B).

| Category | LLaMA | GPT-3 | OPT |
|---|---|---|---|
| Gender | 70.6 | 62.6 | 65.7 |
| Religion | 79.0 | 73.3 | 68.6 |
| Race/Color | 57.0 | 64.7 | 68.6 |
| Sexual orientation | 81.0 | 76.2 | 78.6 |
| Age | 70.1 | 64.4 | 67.8 |
| Nationality | 64.2 | 61.6 | 62.9 |
| Disability | 66.7 | 76.7 | 76.7 |
| Physical appearance | 77.8 | 74.6 | 76.2 |
| Socioeconomic status | 71.5 | 73.8 | 76.2 |
| Average | 66.6 | 67.2 | 69.5 |

**Table 13: WinoGender, co-reference accuracy by pronoun** (paper's Table 13; "gotcha" cases are those where the pronoun does not match the occupation's majority gender and the occupation is the right answer).

| Case | 7B | 13B | 33B | 65B |
|---|---|---|---|---|
| All | 66.0 | 64.7 | 69.0 | 77.5 |
| her/her/she | 65.0 | 66.7 | 66.7 | 78.8 |
| his/him/he | 60.8 | 62.5 | 62.1 | 72.1 |
| their/them/someone | 72.1 | 65.0 | 78.3 | 81.7 |
| her/her/she (gotcha) | 64.2 | 65.8 | 61.7 | 75.0 |
| his/him/he (gotcha) | 55.0 | 55.8 | 55.8 | 63.3 |

**Table 14: TruthfulQA, fraction of answers that are truthful and truthful times informative** (paper's Table 14; scored by specially trained models via the OpenAI API; GPT-3 numbers are from the InstructGPT paper).

| Model | Truthful | Truthful*Inf |
|---|---|---|
| GPT-3 1.3B | 0.31 | 0.19 |
| GPT-3 6B | 0.22 | 0.19 |
| GPT-3 175B | 0.28 | 0.25 |
| LLaMA 7B | 0.33 | 0.29 |
| LLaMA 13B | 0.47 | 0.41 |
| LLaMA 33B | 0.52 | 0.48 |
| LLaMA 65B | 0.57 | 0.53 |

Written by me, an AI, from the paper's table text; no rows left out.

What each column is, and why it matters:
- **Toxicity:** RealToxicityPrompts has about 100k prompts the model must complete; the toxicity of the completion is scored by PerspectiveAPI, a third-party service the authors do not control, so comparison with other papers is hard (paper, 5.1). "Respectful" prompts start with "Complete the following sentence in a polite, respectful, and unbiased manner:".
- **CrowS-Pairs:** each example has a stereotyping sentence and an anti-stereotyping sentence; the score is how often the model gives the stereotyping one the better perplexity, so 50 would be no bias (paper, 5.2; the "no bias" value is general knowledge).
- **WinoGender:** pronoun resolution in sentences with an occupation and a participant, testing whether the answer depends on the pronoun's gender (paper, 5.3).
- **TruthfulQA:** adversarial questions in 38 categories about common false beliefs; "Truthful" is the fraction of true answers, "Truthful*Inf" the fraction that are both true and informative (paper, 5.4).

**Summary: what the tables show.**
- **Toxicity rises with size:** 65B scores 0.128 and 0.141 against 7B's 0.106 and 0.081 (1.2 and 1.7 times, mine), but the rise is not smooth: 13B is lower than 7B on basic prompts, and 33B is lower than 13B on respectful ones. The paper says the relation "may only apply within a model family".
- **Bias:** on average LLaMA is slightly better than GPT-3 and OPT on CrowS-Pairs (66.6 against 67.2 and 69.5) but worse in religion (79.0 against OPT's 68.6, +10.4), age and gender. The paper expects the bias "to come from CommonCrawl".
- **WinoGender:** accuracy is higher with "their/them/someone" than with gendered pronouns, and lower on "gotcha" cases (for 65B, his/him/he falls from 72.1 to 63.3, my arithmetic), "clearly showing that it captures societal biases related to gender and occupation".
- **Truthfulness:** LLaMA-65B 0.57 truthful and 0.53 truthful and informative against GPT-3 175B's 0.28 and 0.25, though "the rate of correct answers is still low".
- **Caution:** the paper says these benchmarks "are not sufficient to fully understand the risks".

### Carbon footprint (paper's Table 15)

**Table 15: carbon footprint of training in the same data centre** (paper's Table 15; all A100-80GB GPUs at 400 W; PUE 1.1; 0.385 kg CO2e per kWh).

| Model | GPU-hours | Energy | Carbon (tCO2eq) |
|---|---|---|---|
| OPT-175B | 809,472 | 356 MWh | 137 |
| BLOOM-175B | 1,082,880 | 475 MWh | 183 |
| LLaMA-7B | 82,432 | 36 MWh | 14 |
| LLaMA-13B | 135,168 | 59 MWh | 23 |
| LLaMA-33B | 530,432 | 233 MWh | 90 |
| LLaMA-65B | 1,022,362 | 449 MWh | 173 |

Written by me, an AI, from the paper's table text; I recomputed every row with equation 5 (all agree to rounding; BLOOM comes to 476 MWh).

What each column is, and why it matters:
- **GPU-hours:** the total GPU time used for training the final model. **Energy:** GPU-hours times 400 W times 1.1 (equation 5). **Carbon:** energy times 0.385. All are for the same assumed data centre, to compare models fairly.

**Summary: what the table shows.**
- **LLaMA-13B costs far less than OPT-175B:** 135,168 against 809,472 GPU-hours (6 times fewer, mine) for a model that is at least as good on most benchmarks.
- **LLaMA-65B is similar to the 175B models:** 449 MWh against OPT's 356 and BLOOM's 475.
- **The whole project:** the paper estimates 2,638 MWh and 1,015 tonnes for developing all the models, on 2048 A100s "for a period of approximately 5 months". My check: 2,638 MWh at 2048 GPUs of 440 W each (including the PUE) is about 122 days, about 4 months, so full-power use does not fit "5 months"; the authors may mean less than full use. The four final runs in the table add up to only 777 MWh (mine).
- **Caution:** the figures are the authors' estimates, using one assumed US-average carbon factor.

## Questions I had, answered

**Why train a small model for so long, if Chinchilla says the best use of compute is a bigger model?** Chinchilla answers a different question: which model is best for a given training budget. If the model will be used billions of times, the cost that matters is the cost per answer, which grows with the number of parameters. A smaller model trained longer costs more to train but less to run (paper, 1). My arithmetic: the 7B model sees 149 tokens per parameter, about 7 times past the 20 that Chinchilla's own example implies.

**Why does SwiGLU use a hidden width of 2/3 of 4d?** A SwiGLU block has three matrices where a ReLU block has two. Cutting the hidden width to $\frac23$ keeps the number of weights the same: at $d=4096$ both have 134,217,728 (my arithmetic, equation 2).

**What does pre-normalisation change?** The normalisation is applied to what goes into each attention or feed-forward block, instead of to what comes out, "to improve the training stability" (paper, 2.2). The reason, from general knowledge: the residual path then stays untouched.

**Why split numbers into digits?** Then a number like 1,247 is always the same four pieces, so the model can learn arithmetic-like patterns, and nothing depends on which long numbers happened to be common enough to get their own token (general knowledge; the paper only states the choice).

**Why is LLaMA worse than Chinchilla and PaLM on MMLU?** The paper offers one possible reason: it used about 177 GB of books and academic papers, while those models used up to 2 TB (paper, 3.6). That is a guess; the paper does not test it.

**Is it fair to say LLaMA-13B beats GPT-3?** On the tables I could check, yes for most benchmarks: it is higher on 5 of the 7 common-sense benchmarks and on NaturalQuestions at 0-shot and 64-shot, with 13.5 times fewer parameters (mine). It is not higher on every benchmark (it loses on PIQA and OBQA), and the abstract says "most".

## Why it mattered
It showed that models trained on public data only could match closed giants, and that a model that "can be run on a single GPU" (paper, 1; the 13B) was good enough to study. Its release started a wave of open models and fine-tuned variants, including [[LoRA 2021]] and [[QLoRA 2023]] style fine-tuning (from memory; not claimed in this paper).

## Limits and open questions
- **Models are not safe:** they reproduce biases (worst in religion, age and gender), make toxic content more often as they grow, and often hallucinate (TruthfulQA: 0.57 truthful at best) (paper, 5).
- **Weaker on knowledge-heavy subjects than Chinchilla and PaLM:** MMLU is 4 to 6 points lower, possibly because of fewer books (the paper's guess).
- **Little instruction tuning:** only one small experiment (LLaMA-I); the paper plans more work.
- **Comparisons depend on other papers' numbers:** competitor results are copied from their papers, with possibly different protocols.
- **Not tested:** the paper does not measure the cost of running the models, which was its stated reason for the design. It argues but does not show the inference saving, beyond the statement that the 13B runs on one GPU (my reading).
- **The authors' plans:** larger models on larger corpora, since "a constant improvement in performance as we were scaling" (paper, 8).

## Concepts used
[[LLM]], [[Pre-training]], [[Scaling Laws]], [[RoPE]], [[Transformer]], [[Layer Normalization]], [[Tokenization]], [[Adam Optimizer]], [[FlashAttention]]

## Reading trail
- **Came from:** [[GPT-3 2020]] and [[Chinchilla 2022]]
- **Led to:** [[Mixtral 2024]], [[DeepSeek-R1 2025]], [[QLoRA 2023]] (fine-tunes LLaMA models in 4 bits) (from memory)

## How much I read
The paper's body as text: pages 1 to 11 (introduction, approach, all results sections, bias and toxicity, carbon footprint, related work and conclusion, Tables 1 to 15 and Figures 1 and 2). I looked at the two figures and a picture of page 3 (Table 2) to check them. Not read: the reference list and the appendix (pages 12 to 27: the 57-task MMLU table, the generations and the other evaluation details). The note says nothing from memory about the paper's content, except where marked. The numbers marked "mine" are my own arithmetic in Python. The figures are cropped from the paper (licensed CC BY 4.0) and the tables are written by me. Not opened in Obsidian, so the flashcards are unreviewed.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-llm LLaMA (Touvron et al., 2023, arXiv 2302.13971): what problem did it attack?::Strong models were closed (GPT-3, Chinchilla, PaLM used non-public data) and the scaling advice ignored the cost of running a model; the aim was the best performance at various inference budgets using only public data.

#flashcards/research/concept #flashcards/topic/research-llm What did people do before LLaMA?::Very big models (GPT-3 175B, Gopher 280B, PaLM 540B) on closed data; Chinchilla showed that smaller models trained on more data are better for a fixed training budget; open models (OPT, GPT-NeoX, BLOOM, GLM) were not competitive with PaLM-62B or Chinchilla.

#flashcards/research/concept #flashcards/topic/research-llm LLaMA: what is the idea?::Pick the model cheapest to run, not to train: a smaller model trained much longer is cheaper at inference. Train 7B to 65B models on trillions of tokens of public data only, with pre-normalisation, SwiGLU and rotary embeddings, and release them.

#flashcards/research/concept #flashcards/topic/research-llm What is in LLaMA's training data (paper's Table 1)?::1.4 trillion tokens: CommonCrawl 67%, C4 15%, Github 4.5%, Wikipedia 4.5%, Books 4.5%, ArXiv 2.5% and StackExchange 2%. Wikipedia and Books are seen about 2.2 to 2.5 times, the rest about once. All data is public and compatible with open-sourcing.

#flashcards/research/concept #flashcards/topic/research-llm How does LLaMA's tokenizer work?::Byte-pair encoding with SentencePiece. Numbers are split into single digits, and unknown UTF-8 characters fall back to bytes, so no character is ever unknown.

#flashcards/research/concept #flashcards/topic/research-llm What three changes does LLaMA make to the Transformer?::Pre-normalisation with RMSNorm (normalise each sub-layer's input, for stability); SwiGLU instead of ReLU with hidden size $\frac23 4d$ (for performance); rotary position embeddings at every layer instead of absolute positions.

#flashcards/research/concept #flashcards/topic/research-llm How was LLaMA trained and made efficient?::AdamW ($\beta_1=0.9$, $\beta_2=0.95$), cosine schedule to 10%, weight decay 0.1, clipping 1.0, 2,000 warm-up steps, batch 4M tokens; memory-saving attention (xformers, FlashAttention backward), fewer recomputed activations, model and sequence parallelism. The 65B took about 21 days on 2048 A100s.

#flashcards/research/concept #flashcards/topic/research-llm What are the sizes of the LLaMA models (paper's Table 2)?::6.7B (width 4096, 32 heads, 32 layers), 13.0B (5120, 40, 40), 32.5B (6656, 52, 60) and 65.2B (8192, 64, 80); learning rate 3.0e-4 for the first two and 1.5e-4 for the others; 4M-token batches; 1.0T tokens for the first two and 1.4T for the others. Every head is 128 wide (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm How many tokens per parameter does each LLaMA see?::7B: about 149; 13B: 77; 33B: 43; 65B: 21.5 (my arithmetic). Chinchilla's example advice is about 20, so the 65B is near it and the small models are far past it on purpose.

#flashcards/research/concept #flashcards/topic/research-llm ![[llama-fig1-training-loss.png]]
What does this figure show?
?
The paper's Figure 1: training loss against billions of training tokens for LLaMA 7B, 13B, 33B and 65B (cropped from the paper, CC BY 4.0). The curves are ordered by size and all still falling when training stops (1.0T tokens for 7B and 13B, 1.4T for the others), which supports training small models for longer. It shows training loss, not task scores.

#flashcards/research/concept #flashcards/topic/research-llm What are the steps from public text to a LLaMA model?::Collect and filter public data; tokenize with digit splitting; build the decoder with RMSNorm pre-normalisation, rotary attention and SwiGLU; train on next-word prediction with AdamW for about 350,000 steps of 4M tokens; train longer than "optimal"; evaluate zero-shot and few-shot; release the models.

#flashcards/research/concept #flashcards/topic/research-llm Write RMSNorm and explain each symbol.
?
$$\mathrm{RMSNorm}(\mathbf{x})_i=\frac{x_i}{\mathrm{RMS}(\mathbf{x})}\,g_i,\qquad \mathrm{RMS}(\mathbf{x})=\sqrt{\frac1n\sum_{j=1}^{n}x_j^{2}}$$
- $\mathbf{x}$: the token's $n$ activations; $\mathrm{RMS}$: root mean square, a typical size of the entries; $g_i$: a learned scale per entry.
- It divides by the typical size so the layer inputs stay in a steady range; unlike layer normalisation it subtracts no mean, so it is cheaper. (General knowledge, from the cited paper, not printed in LLaMA.)

#flashcards/research/concept #flashcards/topic/research-llm RMSNorm example: $\mathbf{x}=(3,4)$, $g=(1,1)$. What is the output, and for $(100,200)$?
?
$\mathrm{RMS}=\sqrt{25/2}=3.536$, output $(0.849,1.131)$. For $(100,200)$: RMS $158.1$, output $(0.632,1.265)$. The output depends on the shape of the vector, not its overall size (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Why does RMSNorm divide by the root mean square and have no mean subtraction?
?
Dividing by the typical size keeps activations from growing or shrinking from layer to layer (for example $(150,200)$ and $(3,4)$ give exactly the same output, because one is 50 times the other), which is what stabilises training. Skipping the mean and bias makes it cheaper than layer normalisation; the cited work found it just as effective (general knowledge).

#flashcards/research/concept #flashcards/topic/research-llm Write the SwiGLU feed-forward block and explain each symbol.
?
$$\mathrm{FFN}_{\mathrm{SwiGLU}}(\mathbf{x})=\big(\mathrm{Swish}(\mathbf{x}W_1)\odot \mathbf{x}W_3\big)W_2,\qquad \mathrm{Swish}(z)=z\,\sigma(z),\qquad d_{\text{hidden}}=\tfrac23\cdot4d$$
- $W_1,W_3$: widen to $d_\text{hidden}$; $W_2$: narrows back to $d$; $\sigma$: sigmoid; $\odot$: entry-wise product, one branch gating the other; $\frac23\cdot4d$: LLaMA's hidden width.
- The gate switches features on and off; the $\frac23$ keeps the weights the same as a ReLU block. (General knowledge from Shazeer 2020; the paper prints only the $\frac23 4d$.)

#flashcards/research/concept #flashcards/topic/research-llm SwiGLU example: $\mathrm{Swish}(2)$ with the other branch at $0.5$, and the weights at $d=4096$.
?
$\mathrm{Swish}(2)=2\times0.881=1.762$; times $0.5$ gives $0.881$. $\mathrm{Swish}(-2)=-0.238$ (ReLU gives $0$). Weights: ReLU block $2\times d\times4d=134{,}217{,}728$; SwiGLU block with width $\frac{8d}{3}$: $3\times d\times\frac{8d}{3}=134{,}217{,}728$ (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Why is the SwiGLU hidden width $\frac23 4d$ and not $4d$?
?
A SwiGLU block has three weight matrices where a ReLU block has two. With width $4d$ it would hold $3\times d\times4d=12d^2$ weights, 50% more than the ReLU block's $8d^2$; with $\frac23 4d=\frac{8d}{3}$ it holds $3\times d\times\frac{8d}{3}=8d^2$. So the parameter count stays the same.

#flashcards/research/concept #flashcards/topic/research-llm Write the rotary position embedding and explain each symbol.
?
$$\begin{pmatrix}x_1'\\x_2'\end{pmatrix}=\begin{pmatrix}\cos m\theta&-\sin m\theta\\ \sin m\theta&\cos m\theta\end{pmatrix}\begin{pmatrix}x_1\\x_2\end{pmatrix},\qquad \mathbf{q}_m^{\top}\mathbf{k}_n\ \text{depends only on}\ m-n$$
- $(x_1,x_2)$: a pair of entries of a query or key; $m$: the token's position; $\theta$: rotation angle per step, different per pair; $\mathbf{q}_m,\mathbf{k}_n$: query and key at positions $m$ and $n$.
- Rotating by position makes the attention score depend only on the distance between tokens, with no learned table of absolute positions. (General knowledge, from the cited paper.)

#flashcards/research/concept #flashcards/topic/research-llm RoPE example: $\theta=30^\circ$, query and key both $(1,0)$. What are the dot products for positions $(2,1)$, $(5,4)$, $(10,4)$ and $(7,7)$?
?
$0.866$, $0.866$, $-1.0$ and $1.0$ (my arithmetic). Positions 2 and 1, and 5 and 4, are both one step apart, so they score the same; 10 and 4 are six steps (180 degrees) apart.

#flashcards/research/concept #flashcards/topic/research-llm Why does RoPE rotate pairs of entries, and why does that give relative positions?
?
Rotating the query by $m\theta$ and the key by $n\theta$ and taking their dot product leaves a rotation by $(m-n)\theta$, so the score is $\cos((m-n)\theta)$ times the lengths: $0.866$ for any gap of one step at $\theta=30^\circ$. Rotating pairs is the simplest way to turn position into an angle. Without it, position must be learned or added once at the input.

#flashcards/research/concept #flashcards/topic/research-llm Write LLaMA's multiple-choice completion score and explain each symbol.
?
$$\mathrm{score}(c)=\frac{P(c\mid \text{context})}{P(c\mid\text{"Answer:"})}$$
- $c$: an answer option; $P(c\mid\text{context})$: its probability after the question; $P(c\mid\text{"Answer:"})$: its probability with no context.
- Used for OpenBookQA and BoolQ; elsewhere the likelihood is divided by the number of characters. It measures how much the context raised the answer's probability.

#flashcards/research/concept #flashcards/topic/research-llm Completion-score example: $P(c\mid\text{context})=0.02, 0.01$ and $P(c\mid\text{"Answer:"})=0.04, 0.005$. Which option wins, and by raw likelihood?
?
Scores $0.5$ and $2.0$, so option 2 wins; raw likelihood would pick option 1. Option 2 is rare in general but the question made it far more likely (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Why divide by $P(c\mid\text{"Answer:"})$ instead of using the raw likelihood?
?
Raw likelihood favours short, common answers whatever the question: in the example it picks the option with $0.02$. Dividing by how likely the answer is anyway turns the score into how much the context helped ($0.5$ against $2.0$). The paper uses it for OpenBookQA and BoolQ, and divides by the number of characters elsewhere to stop longer answers being punished.

#flashcards/research/concept #flashcards/topic/research-llm Write the paper's energy and carbon formulas and explain each symbol.
?
$$\mathrm{Wh}=\text{GPU-hours}\times(\text{GPU power})\times\mathrm{PUE},\qquad \mathrm{tCO_2eq}=\mathrm{MWh}\times0.385$$
- GPU-hours: GPUs times hours; GPU power: 400 W per A100-80GB; PUE: 1.1, extra energy for cooling and buildings; 0.385: kg CO2 per kWh, the US average.
- It puts models on one scale as if trained in the same data centre; a different grid changes the carbon number a lot.

#flashcards/research/concept #flashcards/topic/research-llm Carbon example: the 65B model, 1,022,362 GPU-hours. What are the energy and carbon, and for the 7B?
?
$1{,}022{,}362\times400\times1.1=449.8$ MWh and $449.8\times0.385=173.2$ tonnes. The 7B: $82{,}432$ GPU-hours give $36.3$ MWh and $14.0$ tonnes. The 65B used 12.4 times the GPU-hours (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Why does the carbon formula use one fixed factor of 0.385 and a PUE of 1.1?
?
The authors want to compare models "as if trained in the same data center", so they use the US average (0.385 kg per kWh) for every model instead of each one's real grid, which would give very different numbers (the paper cites 0.057 for BLOOM's grid and 0.231 for OPT's). The 1.1 PUE adds 10% for cooling and facilities.

#flashcards/research/concept #flashcards/topic/research-llm How did LLaMA do on common-sense reasoning (paper's Table 3)?::LLaMA-13B is above GPT-3 175B on 5 of the 7 benchmarks where GPT-3 has a number. LLaMA-65B is above Chinchilla-70B on all 5 where it has a number and above PaLM-540B on 5 of 7 (lower on BoolQ by 2.7 and WinoGrande by 4.1). My arithmetic from the table.

#flashcards/research/concept #flashcards/topic/research-llm What does the LLaMA paper say about Chinchilla on BoolQ and what does the table show?::The text says LLaMA-65B beats Chinchilla-70B on all reported benchmarks except BoolQ, but Table 3 has LLaMA-65B at 85.3 and Chinchilla at 83.7 on BoolQ, so LLaMA is ahead there too. The paper contradicts itself; the table is used.

#flashcards/research/concept #flashcards/topic/research-llm How did LLaMA do on closed-book question answering (paper's Tables 4 and 5)?::LLaMA-65B is first or tied: NaturalQuestions 23.8 against Chinchilla's 16.6 at 0-shot and 39.9 against 35.5 at 64-shot; TriviaQA 68.2 against 55.4 at 0-shot. LLaMA-13B beats GPT-3 on NaturalQuestions and matches Chinchilla on TriviaQA. The 33B beats the 65B on 0-shot NaturalQuestions.

#flashcards/research/concept #flashcards/topic/research-llm How did LLaMA do on maths and code (paper's Tables 7 and 8)?::GSM8k: 65B scores 50.9, below Minerva 62B's 52.4 without voting but above it with voting (69.7 against 68.5); on MATH, Minerva is far ahead (27.6 against 10.6). Code: LLaMA-13B beats LaMDA 137B on HumanEval and MBPP; the 65B beats PaLM 62B and is within 2.5 points of PaLM 540B on HumanEval pass@1.

#flashcards/research/concept #flashcards/topic/research-llm Why is "LLaMA-65B outperforms Minerva-62B on GSM8k" only half true?::Without majority voting LLaMA-65B scores 50.9 and Minerva 62B scores 52.4 (LLaMA is 1.5 lower); with majority voting over many samples LLaMA scores 69.7 against 68.5. Minerva was also fine-tuned on maths, LLaMA was not.

#flashcards/research/concept #flashcards/topic/research-llm How did LLaMA do on MMLU, and what explanation does the paper give (paper's Tables 9 and 10)?::LLaMA-65B averages 63.4: 19.5 above GPT-3 but 4.1 below Chinchilla and 5.9 below PaLM-540B. A possible explanation is only 177 GB of books and papers against up to 2 TB for those models. A brief instruction fine-tune (LLaMA-I) reaches 68.9, 5.5 higher but below the 77.4 of GPT code-davinci-002.

#flashcards/research/concept #flashcards/topic/research-llm ![[llama-fig2-performance-during-training.png]]
What does this figure show?
?
The paper's Figure 2: accuracy on TriviaQA, HellaSwag, NaturalQuestions, SIQA, WinoGrande and PIQA against billions of training tokens for the four LLaMA sizes, with Chinchilla's final score as a dashed line (cropped from the paper, CC BY 4.0). Accuracy mostly rises steadily with tokens and tracks training loss; SIQA is very noisy and WinoGrande flattens for the 33B and 65B.

#flashcards/research/concept #flashcards/topic/research-llm What did LLaMA's bias, toxicity and truthfulness tests show (paper's Tables 11 to 14)?::Toxicity rises with size but not smoothly (65B 0.128 and 0.141 against 7B 0.106 and 0.081). CrowS-Pairs average 66.6 against GPT-3's 67.2 and OPT's 69.5, but religion is 10.4 worse than OPT. WinoGender is worse on gendered pronouns and on "gotcha" cases. TruthfulQA: 0.57 truthful and 0.53 truthful times informative for the 65B against GPT-3's 0.28 and 0.25.

#flashcards/research/concept #flashcards/topic/research-llm What did LLaMA's training cost (paper's Table 15)?::The 65B took 1,022,362 A100-GPU-hours, 449 MWh and about 173 tonnes CO2 (assumed US-average grid); the 7B took 82,432 GPU-hours and 14 tonnes. The 13B took 6 times fewer GPU-hours than OPT-175B (my arithmetic). The whole project is estimated at 2,638 MWh and 1,015 tonnes.

#flashcards/research/concept #flashcards/topic/research-llm Does the paper's "5 months" fit its 2,638 MWh?::Not at full power: 2,638 MWh at 2048 GPUs of 440 W each (400 W times the PUE of 1.1) is about 122 days, about 4 months (my arithmetic), so either use was below full power or the figure is rounded.

#flashcards/research/concept #flashcards/topic/research-llm Why train a small model for so long when Chinchilla says to train a bigger one?::Chinchilla optimises for a fixed training budget, but a model is used far more often than it is trained, so the cost per answer matters, and it grows with parameters. A smaller model trained longer costs more to train but less to run. LLaMA 7B sees 149 tokens per parameter against Chinchilla's roughly 20 (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm LLaMA: what limits does the paper name?::Models are biased (worst in religion, age and gender), toxicity rises with size, they often hallucinate, MMLU is 4 to 6 points below Chinchilla and PaLM (possibly fewer books), only one small instruction-tuning experiment, and rival numbers are copied from other papers. The paper argues the inference saving but does not measure it.

#flashcards/research/concept #flashcards/topic/research-llm LLaMA: why did it matter and what is its place in the reading trail?::It showed public-data-only models could match closed giants, and the 13B was good enough on one GPU to study; open models and fine-tuning methods such as LoRA and QLoRA followed (from memory). It came from GPT-3 and Chinchilla and led to Mixtral and DeepSeek-R1 (from memory).

#flashcards/research/concept #flashcards/topic/research-llm How much of the LLaMA paper did the note read?::The body as text (pages 1 to 11, all tables and both figures); not the references or the appendix (pages 12 to 27, including the 57-task MMLU table). Figures are cropped from the paper (CC BY 4.0), tables are written by me, and the RMSNorm, SwiGLU and RoPE formulas come from the cited papers, not from LLaMA.
