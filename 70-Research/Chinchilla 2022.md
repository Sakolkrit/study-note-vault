---
tags: [paper, NLP, scaling]
link: https://arxiv.org/abs/2203.15556
authors: Hoffmann, Borgeaud, Mensch and Sifre (equal contribution) and many others, including Rae and Vinyals (DeepMind; checked on the paper's first page)
year: 2022
status: body-read
field: text analytics
---

# Chinchilla 2022

> **In one line for a 12-year-old:** People were making models bigger and bigger but feeding them the same amount of text. These authors found it works better to grow both the model and the text together.

Paper: "Training Compute-Optimal Large Language Models", arXiv 2203.15556, read as version 1 of 29 March 2022. Marks: "(paper, section)" was read in the paper; "(general knowledge)" is my own background; "(mine)" is my arithmetic. The paper's arXiv licence is not open, so I do not copy its figures or tables; the tables below are written by me.

## The problem
For a fixed computing budget, how big should the model be and how much text should it read? Training a large model is so costly that "it is typically only feasible to train these large models once", so choosing the size and the amount of data well matters (paper, 1).

## What people did before
Earlier scaling advice from Kaplan et al. (2020) ([[Scaling Laws 2020]]) showed a power-law link between parameters and loss and concluded that model size should grow much faster than training data: for a 10 times bigger budget, a 5.5 times bigger model but only 1.8 times more tokens (paper, 1). Following that, the field trained bigger and bigger models, most on about 300 billion tokens (Table 1 below).

## The idea
Treat it as a measurement problem: train over 400 language models of many sizes for different amounts of data, find for each compute budget which size gave the lowest loss, and fit how the best size and the best amount of data grow with compute. Three different ways of doing this all say the same thing: parameters and tokens should grow in equal proportions. Then test the prediction by training a model, Chinchilla, with the same compute as Gopher but 4 times fewer parameters and 4 times more data (paper, abstract and 1).

## The method (what was done, step by step)
This paper has no new network; its "architecture" is an experimental design. Marks as above.

### Part by part
- **The compute budget $C$ and the training set-up.**
  - *What it does:* the budget is measured in floating-point operations (FLOPs), counted from the model's real operations, including embeddings (paper, Appendix F). It is closely approximated by $C\approx6ND$ for $N$ parameters and $D$ training tokens (paper, 3.3, from Kaplan et al.).
  - *Why it matters:* a budget fixes the trade-off: more parameters means fewer tokens, and the reverse. The question is where the best point on that line lies.
- **Over 400 models.**
  - *What it does:* models from 70 million to over 16 billion parameters, trained on 5 to over 400 billion tokens, each configuration for several different training lengths (paper, 1; the abstract says up to 500 billion tokens).
  - *Why it matters:* a wide range of sizes and data lets the authors fit curves instead of guessing; they say the majority of their models have over 500 million parameters, unlike earlier work whose majority were under 100 million (paper, 2).
- **Matching the learning-rate schedule to the training length.**
  - *What it does:* the learning rate decays by 10 times over about as many tokens as the model will see (paper, 2 and Appendix B).
  - *Why it matters:* the paper argues that Kaplan et al. used one fixed schedule for all runs, so their losses for shorter runs were overestimates; this "contributes to the conclusion that model size should increase faster than training data size" (paper, 2). It is a main reason the two papers disagree.
- **Approach 1: fix model sizes, vary the number of training tokens.**
  - *What it does:* for each size train 4 models whose learning-rate schedules differ in length by a factor of 16; smooth and interpolate each loss curve; at 1,500 log-spaced compute values pick which run has the lowest loss; fit power laws $N_{\text{opt}}\propto C^a$ and $D_{\text{opt}}\propto C^b$ (paper, 3.1).
  - *Why it matters:* it reads the best size straight off the training curves. Result: $a=0.50$, $b=0.50$.
- **Approach 2: IsoFLOP profiles.**
  - *What it does:* for 9 fixed compute amounts (from $6\times10^{18}$ to $3\times10^{21}$ FLOPs), train models of different sizes, each given exactly that many FLOPs; plot final loss against size, fit a parabola to each curve and take the bottom (paper, 3.2).
  - *Why it matters:* for each budget it directly shows a valley, so there is an optimal size for each budget. Result: $a=0.49$, $b=0.51$.
- **Approach 3: fit a formula for the loss.**
  - *What it does:* fit $\hat L(N,D)=E+A/N^{\alpha}+B/D^{\beta}$ to all the results, then minimise it for each budget (paper, 3.3; Eq. 2 to 4 below).
  - *Why it matters:* it gives a closed-form answer. Result: $a=0.46$, $b=0.54$, a slightly smaller optimal model at large budgets.
- **Chinchilla: the test of the prediction.**
  - *What it does:* a 70-billion-parameter model trained on 1.4 trillion tokens, with the same architecture and set-up as Gopher except the changes listed below (paper, 4).
  - *Why it matters:* it checks the prediction on a real downstream benchmark suite, and a 4 times smaller model is also cheaper to fine-tune and run.

### Chinchilla against Gopher (paper's Table 4)

**Table 4: architecture details** (paper's Table 4; the feed-forward size is always 4 times $d_{\text{model}}$; the batch size is in tokens and is doubled midway through training for both).

| Model | Layers | Heads | Key/value size | $d_{\text{model}}$ | Max learning rate | Batch size |
|---|---|---|---|---|---|---|
| Gopher 280B | 80 | 128 | 128 | 16,384 | $4\times10^{-5}$ | 3M to 6M |
| Chinchilla 70B | 80 | 64 | 128 | 8,192 | $1\times10^{-4}$ | 1.5M to 3M |

*Written by me, an AI, from the paper's table text and checked against the extracted text; it is not a copy of the paper's table.*

**What each column is, and why it matters**
- **Layers, Heads, Key/value size, $d_{\text{model}}$:** the Transformer's shape: how many layers, how many attention heads, the size of each head's key and value vectors, and the width of the vectors that flow through (all as in [[Attention Is All You Need 2017]]). They set the model's size.
- **Max learning rate:** the largest step size used during training; the smaller model uses a larger one, which is typical for smaller models (general knowledge).
- **Batch size:** how many tokens are processed per update; a smaller model can use a smaller batch.

**Summary: what the table shows**
- **The only changes in shape:** half the width ($8{,}192$ against $16{,}384$) and half the heads (64 against 128); the number of layers (80) and the key and value size (128) are the same. So the model is 4 times smaller in parameters (70B against 280B) with the same depth.
- **Training differs too:** Chinchilla has a 2.5 times larger maximum learning rate and a half-size batch (paper).
- **Check (mine):** counting only the main weights, $80\times12\times8192^2\approx64$ billion against 70 billion reported (and $258$ billion against 280 for Gopher); the same 8% gap in both suggests parts I left out, which I did not check.
- **Other differences from Gopher (paper, 4.1):** the AdamW optimizer instead of Adam (better loss and downstream results after fine-tuning); a slightly different tokenizer without NFKC normalisation (94.15% of tokens are the same), which helps maths and chemistry; a float32 copy of the weights kept in the optimizer; and a slightly different mixture of the same MassiveText data.

## The core mechanism, step by step

### Finding the compute-optimal size for a budget
1. **Fix a compute budget $C$.**
   - *What it does:* choose how many FLOPs you will spend, for example Gopher's $5.76\times10^{23}$.
   - *Why it matters:* the budget is the constraint; everything else is the trade-off inside it.
2. **List the (size, tokens) pairs that cost exactly $C$.**
   - *What it does:* with $C\approx6ND$, a bigger model $N$ means fewer tokens $D=C/6N$.
   - *Why it matters:* it turns one question (size and tokens) into one (size), since the tokens follow.
3. **Find the pair with the lowest loss.**
   - *What it does:* Approach 1 reads it off the training curves, Approach 2 from the bottom of a parabola fitted to each IsoFLOP curve, Approach 3 from the fitted formula.
   - *Why it matters:* this is the "compute-optimal" model for that budget.
4. **Repeat for many budgets and fit a power law.**
   - *What it does:* plot best size and best tokens against $C$ on a log-log scale and fit straight lines: $N_{\text{opt}}\propto C^{a}$, $D_{\text{opt}}\propto C^{b}$.
   - *Why it matters:* it lets you extrapolate to budgets you cannot afford to test. The paper warns that extrapolating "many orders of magnitude" carries "significant uncertainty" (paper, 3.4).
5. **Check by training the prediction.**
   - *What it does:* train Chinchilla at the predicted size for Gopher's budget.
   - *Why it matters:* the fit says that size should beat the larger model; the downstream results confirm it.

**Worked example (mine; the fitted formula's valley).** Using the paper's appendix constants $E=1.69$, $A=406.4$, $B=410.7$, $\alpha=0.34$, $\beta=0.28$ (D.2), take Gopher's budget $C=5.76\times10^{23}$ and set $D=C/(6N)$.

| Size $N$ | Tokens $D$ | Predicted loss $\hat L$ |
|---|---|---|
| 10B | 9,600B | 1.947 |
| 20B | 4,800B | 1.933 |
| 32B | 3,000B | 1.931 |
| 40B | 2,400B | 1.931 |
| 70B (Chinchilla) | 1,371B | 1.938 |
| 140B | 686B | 1.955 |
| 280B (Gopher's size) | 343B | 1.984 |

The curve has a flat-bottomed valley: between about 20B and 70B the predicted loss stays within 0.007 of its minimum (about 1.931 at 32B to 40B), while Gopher's 280B is 0.053 higher. That is the meaning of the paper's "optimal model is somewhere between 40 and 70 billion" for this budget.

## The equations
Equation numbers are the paper's; the compute approximation is stated in the text.

**1. The question** (Eq. 1)

$$
N_{\text{opt}}(C),\ D_{\text{opt}}(C)=\mathop{\mathrm{argmin}}_{N,\,D\ \text{s.t.}\ \mathrm{FLOPs}(N,D)=C}\ L(N,D)
$$

**Where:**
- $C$: the compute budget in FLOPs. $N$: the number of model parameters. $D$: the number of training tokens.
- $L(N,D)$: the final training loss of a model with $N$ parameters trained on $D$ tokens.
- $N_{\text{opt}}(C)$, $D_{\text{opt}}(C)$: the best size and the best number of tokens for budget $C$.

**Matters:** it states the problem exactly: among all ways to spend $C$, pick the lowest loss. Everything else in the paper is a way to estimate these two functions.

**Example (tiny numbers):** with $C=5.76\times10^{23}$ the table above is a slice of this problem: each row spends the same budget; the argmin is the row with the lowest loss (about 32B to 40B by the fitted formula).

**2. The compute approximation**

$$
C\approx6\,N\,D
$$

**Where:**
- $C$: training FLOPs; $N$: parameters; $D$: training tokens. The factor 6 is about 2 for the forward pass plus about 4 for the backward pass, per parameter and token (the paper assumes the backward pass costs twice the forward pass, appendix F).

**Matters:** it makes the budget a simple product of size and data, so doubling the model while keeping the budget halves the tokens. The paper uses it for Approach 3 (from Kaplan et al.) and checks it against its exact count (Table A4): "the differences in FLOP calculation are very small".

**Example (tiny numbers):** Chinchilla: $6\times70\times10^{9}\times1.4\times10^{12}=5.88\times10^{23}$. Gopher: $6\times280\times10^{9}\times300\times10^{9}=5.04\times10^{23}$.
- The paper's exact count puts both at about $5.76\times10^{23}$ ("the same number of FLOPs"); the approximation is off by 2% for Chinchilla and 13% for Gopher in my numbers.
- Meaning: the two models cost about the same, though one has 4 times the parameters and the other 4.7 times the tokens.

**3. A formula for the loss** (Eq. 2)

$$
\hat L(N,D)\triangleq E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}
$$

**Where:**
- $E$: the loss of an ideal model on this data, which the paper says should correspond to the entropy of natural text; the loss cannot go below it.
- $A/N^{\alpha}$: how much worse a perfectly trained model with $N$ parameters is than the ideal; it shrinks as $N$ grows.
- $B/D^{\beta}$: how much worse a model is because it was trained for finitely many steps on a finite sample; it shrinks as $D$ grows.
- $A$, $B$, $\alpha$, $\beta$: constants fitted to the data. Fitted values (appendix D.2): $E=1.69$, $A=406.4$, $B=410.7$, $\alpha=0.34$, $\beta=0.28$.

**Matters:** the loss has three parts, and the last two pull in opposite directions at a fixed budget: a bigger model leaves less for the data, and the reverse. The best split balances the two terms. The paper notes that both exponents are below $1/2$ and that this "is expected for the data-efficiency coefficient" (appendix D.2).

**Example (tiny numbers):** Chinchilla ($N=70\text{B}$, $D=1.4\text{T}$) against Gopher ($N=280\text{B}$, $D=300\text{B}$).
- Chinchilla: $1.69+406.4/70\text{B}^{0.34}+410.7/1.4\text{T}^{0.28}=1.69+0.084+0.163=1.937$.
- Gopher: $1.69+0.052+0.250=1.993$.
- Meaning: Gopher's bigger size shrinks the second term by 0.03 but its small data leaves the third term 0.09 higher, so Chinchilla's loss is lower by about 0.057 (my arithmetic with the rounded constants).

**4. How the fit is done** (Eq. 3)

$$
\min_{A,B,E,\alpha,\beta}\ \sum_{\text{runs }i}\ \mathrm{Huber}_{\delta}\Big(\log\hat L(N_i,D_i)-\log L_i\Big)
$$

**Where:**
- The sum is over all training runs $i$ with $N_i$ parameters, $D_i$ tokens and observed final loss $L_i$.
- $\log\hat L-\log L_i$: the error of the formula, in log terms. $\mathrm{Huber}_\delta$: a loss that is quadratic for small errors and only linear for large ones; the paper uses $\delta=10^{-3}$ and the L-BFGS algorithm, with a grid of starting points to avoid local minima.

**Matters:** the Huber loss makes the fit robust to outliers, which the paper says is important for predicting held-out points: the low-compute runs with larger errors are in effect treated as outliers, which also pulls Approach 3 towards a slightly smaller $N_{\text{opt}}$.

**Example (tiny numbers):** Huber with $\delta=0.001$: for $|r|\le\delta$ it is $r^2/2$, otherwise $\delta(|r|-\delta/2)$ (the usual definition, general knowledge).
- Error $r=0.0005$: Huber $=1.25\times10^{-7}$, the same as $r^2/2$.
- Error $r=0.05$: Huber $=4.95\times10^{-5}$ against $r^2/2=1.25\times10^{-3}$, 25 times smaller.
- Meaning: one large miss costs only 25 times less than a plain squared error would charge, so a few outliers cannot dominate the fit.

**5. The closed-form frontier** (Eq. 4)

$$
N_{\text{opt}}(C)=G\left(\frac{C}{6}\right)^{a},\qquad D_{\text{opt}}(C)=G^{-1}\left(\frac{C}{6}\right)^{b},\qquad G=\left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}},\quad a=\frac{\beta}{\alpha+\beta},\quad b=\frac{\alpha}{\alpha+\beta}
$$

**Where:**
- $C/6$: the budget divided by 6, from $C\approx6ND$, so $N\cdot D=C/6$. $G$: a constant built from the fitted $A$, $B$, $\alpha$, $\beta$.
- $a$ and $b$: the exponents: how fast the best size and the best tokens grow with compute. They add up to 1, since $N\cdot D=C/6$.

**Matters:** minimising $\hat L$ under $C\approx6ND$ balances the two data-dependent terms and gives power laws "by construction". When $\alpha$ and $\beta$ are close, $a$ and $b$ are close to $1/2$: equal growth of size and data. Kaplan et al.'s exponents are $0.73$ and $0.27$ by comparison (Table 2).

**Example (tiny numbers):** with the rounded constants, $\alpha=0.34$, $\beta=0.28$: $a=0.28/0.62=0.452$, $b=0.548$, $G=(0.34\cdot406.4/(0.28\cdot410.7))^{1/0.62}=1.345$.
- For Gopher's budget: $N_{\text{opt}}=1.345\times(5.76\times10^{23}/6)^{0.452}=32$ billion, $D_{\text{opt}}=3.0$ trillion (my arithmetic).
- The paper reports $a=0.46$ and an optimum of 40 billion for this budget (Figure 4). My numbers from the rounded appendix constants (0.45 and 32 billion) are close but do not reproduce them exactly, probably because of rounding of the constants; I did not check.

## What they found
For compute-optimal training, model size and training tokens should be scaled in equal proportions: for every doubling of model size, the tokens should also double (abstract). All three approaches agree (Table 2). Chinchilla (70B parameters, 1.4T tokens) "uniformly and significantly outperforms" Gopher (280B), GPT-3 (175B), Jurassic-1 (178B) and Megatron-Turing NLG (530B) on a large range of downstream tasks, and reaches a state-of-the-art average accuracy of 67.5% on MMLU, more than 7% above Gopher (abstract).

### The models of the day (paper's Table 1)

**Table 1: current large language models** (paper's Table 1; size in parameters, training tokens; I add a column of tokens per parameter).

| Model | Parameters | Training tokens | Tokens per parameter (mine) |
|---|---|---|---|
| LaMDA | 137 billion | 168 billion | 1.2 |
| GPT-3 | 175 billion | 300 billion | 1.7 |
| Jurassic | 178 billion | 300 billion | 1.7 |
| Gopher | 280 billion | 300 billion | 1.1 |
| MT-NLG 530B | 530 billion | 270 billion | 0.5 |
| **Chinchilla** | **70 billion** | **1.4 trillion** | **20.0** |

*Written by me, an AI, from the paper's table text and checked against the extracted text; it is not a copy of the paper's table.*

**What each column is, and why it matters**
- **Parameters:** the size of the model. **Training tokens:** how many tokens (word pieces) it saw during training. Tokens per parameter (mine) divides the second by the first. It matters because the paper's thesis is that this ratio was far too low.

**Summary: what the table shows**
- **Before Chinchilla the field read about 300 billion tokens whatever the size:** all but LaMDA were trained for about 300 billion tokens (and MT-NLG for 270 billion), so the biggest model had the fewest tokens per parameter (0.5 for MT-NLG).
- **Chinchilla is the outlier:** 20 tokens per parameter, about 12 to 40 times more than the others (my arithmetic), at the smallest size.

### How the best size grows with compute (paper's Table 2)

**Table 2: estimated scaling exponents** (paper's Table 2; $N_{\text{opt}}\propto C^{a}$ and $D_{\text{opt}}\propto C^{b}$; in brackets the 10th and 90th percentiles from bootstrapping, where 80% of the data is sampled 100 times).

| Approach | $a$ (for $N_{\text{opt}}$) | $b$ (for $D_{\text{opt}}$) |
|---|---|---|
| 1. Minimum over training curves | 0.50 (0.488, 0.502) | 0.50 (0.501, 0.512) |
| 2. IsoFLOP profiles | 0.49 (0.462, 0.534) | 0.51 (0.483, 0.529) |
| 3. Parametric modelling of the loss | 0.46 (0.454, 0.455) | 0.54 (0.542, 0.543) |
| Kaplan et al. (2020) | 0.73 | 0.27 |

*Written by me, an AI, from the paper's table text and checked against the extracted text; it is not a copy of the paper's table.*

**What each column is, and why it matters**
- **Approach:** the three ways of estimating the optimum (above); the last row is the earlier paper's estimate.
- **$a$ and $b$:** the exponents. If $a=b=0.5$, size and data grow equally; $a=0.73$ means the model should grow much faster than the data. They add up to 1 because $C\approx6ND$.
- **Brackets:** the spread of the estimate when 80% of the data is resampled: a measure of how stable the fit is.

**Summary: what the table shows**
- **The three approaches agree:** $a$ between 0.46 and 0.50 and $b$ between 0.50 and 0.54, so about equal scaling, with Approach 3 giving a slightly smaller optimal model.
- **They disagree sharply with Kaplan et al.:** $0.73$ and $0.27$. For a 10 times bigger budget this means a model about 5.4 times larger and 1.9 times more tokens (the paper's text says 5.5 and 1.8), against about 3.2 times for each with $0.5$ (my arithmetic).
- **Caution:** the Approach 3 interval is very narrow (0.454 to 0.455), which I read as one fit repeatedly resampled, not agreement of independent evidence (my reading); the paper itself notes that Approach 3 gives a smaller $N_{\text{opt}}$ at large budgets because of curvature (appendix E).

### How big and how many tokens for a given budget (paper's Table 3)

**Table 3: estimated optimal training FLOPs and tokens for various model sizes** (paper's Table 3; projections from Approach 1; "Gopher unit" is the FLOPs divided by Gopher's $5.76\times10^{23}$; I add the tokens per parameter).

| Parameters | FLOPs | FLOPs in Gopher units | Tokens | Tokens per parameter (mine) |
|---|---|---|---|---|
| 400 million | $1.92\times10^{19}$ | 1/29,968 | 8.0 billion | 20.0 |
| 1 billion | $1.21\times10^{20}$ | 1/4,761 | 20.2 billion | 20.2 |
| 10 billion | $1.23\times10^{22}$ | 1/46 | 205.1 billion | 20.5 |
| 67 billion | $5.76\times10^{23}$ | 1 | 1.5 trillion | 22.4 |
| 175 billion | $3.85\times10^{24}$ | 6.7 | 3.7 trillion | 21.1 |
| 280 billion | $9.90\times10^{24}$ | 17.2 | 5.9 trillion | 21.1 |
| 520 billion | $3.43\times10^{25}$ | 59.5 | 11.0 trillion | 21.2 |
| 1 trillion | $1.27\times10^{26}$ | 221.3 | 21.2 trillion | 21.2 |
| 10 trillion | $1.30\times10^{28}$ | 22,515.9 | 216.2 trillion | 21.6 |

*Written by me, an AI, from the paper's table text and checked against the extracted text; it is not a copy of the paper's table.*

**What each column is, and why it matters**
- **Parameters:** a model size. **FLOPs:** the training compute that makes that size optimal. **Gopher units:** the same, in multiples of Gopher's budget, so 1 is Gopher's budget. **Tokens:** the training data that size should see.
- **Tokens per parameter (mine):** the ratio of the last two columns. It matters because it shows the rule the table implies.

**Summary: what the table shows**
- **A near-constant ratio:** the optimal number of tokens is about 20 to 22 times the number of parameters at every size from 400 million to 10 trillion (my arithmetic). This is a pattern in the table, not a rule the paper states in these words.
- **Current models were far over-sized for their budgets:** a 175-billion model should use $3.85\times10^{24}$ FLOPs and 3.7 trillion tokens, 12 times the 300 billion GPT-3 actually saw (my arithmetic).
- **Data becomes the bottleneck:** the paper says the data needed "is far beyond what is currently used" and stresses dataset collection; a trillion-parameter model is "unlikely to be the optimal model to train" unless the budget reaches about $10^{26}$ FLOPs.
- **Caution:** these are extrapolations over many orders of magnitude, with, in the paper's words, "significant uncertainty". And a mismatch: the paper's text says a 175-billion model should be trained with $4.41\times10^{24}$ FLOPs and over 4.2 trillion tokens, while this table (Approach 1) says $3.85\times10^{24}$ and 3.7 trillion; I did not find which approach the text uses.

### Does it work? Language understanding (paper's Table 6)

**Table 6: MMLU average 5-shot accuracy over 57 tasks** (paper's Table 6; the rows other than Chinchilla and Gopher are comparisons taken from the MMLU paper, plus forecasts by 73 human forecasters).

| Model or reference | Accuracy |
|---|---|
| Random guessing | 25.0% |
| Average human rater | 34.5% |
| GPT-3, 5-shot | 43.9% |
| June 2022 forecast for the best model | 57.1% |
| Gopher, 5-shot | 60.0% |
| June 2023 forecast for the best model | 63.4% |
| **Chinchilla, 5-shot** | **67.6%** |
| Average human expert | 89.8% |

*Written by me, an AI, from the paper's table text and checked against the extracted text; it is not a copy of the paper's table.*

**What each column is, and why it matters**
- **MMLU:** a test of exam-like multiple-choice questions on 57 academic subjects (such as chemistry or law). **5-shot:** the model is shown 5 solved examples in the prompt before the question, with no training on the task. It matters because it is a broad measure of knowledge that is less affected by the training data containing the test text than the language-modelling scores (paper, 4.2).
- **Forecasts:** what 73 competitive forecasters predicted the best model would score in June 2022 and June 2023. It matters because it shows how surprising the result was.

**Summary: what the table shows**
- **Best:** Chinchilla, 67.6%, +7.6 over Gopher and +23.7 over GPT-3 (my arithmetic), above the forecast for June 2023 (63.4%) by 4.2 points.
- **Still far from experts:** average human experts get 89.8%, 22.2 points higher (my arithmetic).
- **Task by task:** Chinchilla is better on 51 of the 57 tasks, the same on 2 and worse on 4 (college mathematics, econometrics, moral scenarios and formal logic); it exceeds 90% on four tasks (paper, 4.2.2).
- **A small mismatch in the paper:** the abstract says 67.5% while Table 6 and the text say 67.6%.

### Other tests: reading, common sense, questions (paper's Tables 7 to 9)

**Table 7: reading comprehension** (paper's Table 7; accuracy %; the GPT-3 and MT-NLG results use a different prompt format on RACE, so they are not comparable to the other columns).

| Task | Chinchilla | Gopher | GPT-3 | MT-NLG 530B |
|---|---|---|---|---|
| LAMBADA, zero-shot | 77.4 | 74.5 | 76.2 | 76.6 |
| RACE-m, few-shot | 86.8 | 75.1 | 58.1 | - |
| RACE-h, few-shot | 82.3 | 71.6 | 46.8 | 47.9 |

**Table 8: zero-shot common-sense benchmarks** (paper's Table 8; accuracy %).

| Task | Chinchilla | Gopher | GPT-3 | MT-NLG 530B | Supervised SOTA |
|---|---|---|---|---|---|
| HellaSwag | 80.8 | 79.2 | 78.9 | 80.2 | 93.9 |
| PIQA | 81.8 | 81.8 | 81.0 | 82.0 | 90.1 |
| Winogrande | 74.9 | 70.1 | 70.2 | 73.0 | 91.3 |
| SIQA | 51.3 | 50.6 | - | - | 83.2 |
| BoolQ | 83.7 | 79.3 | 60.5 | 78.2 | 91.4 |

**Table 9 (excerpt): closed-book question answering** (paper's Table 9; accuracy %; a dash means not reported).

| Task and setting | Chinchilla | Gopher | GPT-3 | Open-book SOTA |
|---|---|---|---|---|
| Natural Questions, 0-shot | 16.6 | 10.1 | 14.6 | 54.4 |
| Natural Questions, 5-shot | 31.5 | 24.5 | - | |
| Natural Questions, 64-shot | 35.5 | 28.2 | 29.9 | |
| TriviaQA unfiltered, 0-shot | 67.0 | 52.8 | 64.3 | - |
| TriviaQA unfiltered, 5-shot | 73.2 | 63.6 | - | - |
| TriviaQA unfiltered, 64-shot | 72.3 | 61.3 | 71.2 | - |

*All three tables were written by me, an AI, from the paper's table text and checked against the extracted text; they are not copies. Table 9 also has a filtered TriviaQA section, which I left out.*

**What each column is, and why it matters**
- **The tasks:** LAMBADA (predict the last word of a passage), RACE-m and RACE-h (reading questions for middle and high school), HellaSwag, PIQA, Winogrande, SIQA and BoolQ (common-sense and yes-or-no reasoning), Natural Questions and TriviaQA (factual questions answered without looking at documents, "closed-book").
- **Zero-shot, few-shot, 5-shot, 64-shot:** how many solved examples are given in the prompt (none, a few, 5, 64). More shots usually help.
- **Supervised SOTA and open-book SOTA:** the best results from models trained on the task or allowed to read documents. It matters because they are the upper reference: the general model is far below them on common sense and Natural Questions.

**Summary: what the tables show**
- **Chinchilla beats Gopher on every row where both are reported**, except a tie on PIQA (81.8 each): gains such as +10.7 on RACE-h, +11.7 on RACE-m, +4.4 on BoolQ, +4.8 on Winogrande, +7.0 and +7.3 on Natural Questions (5-shot and 64-shot) and +14.2 on zero-shot TriviaQA (my arithmetic).
- **Against the 530-billion MT-NLG:** Chinchilla is better on all common-sense tasks but one (PIQA, where MT-NLG scores 82.0).
- **Far from the specialised systems:** for example 51.3 against 83.2 on SIQA and 35.5 against 54.4 on open-book Natural Questions.
- **A mismatch in the paper:** the text says Gopher's 5-shot Natural Questions score is 21%, while Table 9 says 24.5; the table values are used here.
- **Caution:** the paper says Chinchilla's language-modelling scores need care, since it saw 4 times more data and test text could have leaked into it (paper, 4.2.1), which is why it stresses MMLU and BIG-bench.

### Bias and toxicity (text only)
The paper expects Chinchilla to carry risks similar to Gopher's since it uses the same data. On Winogender (pronoun resolution) it is more accurate than Gopher in every group (78.3% against 71.4% overall), but the gain is smaller for male pronouns (+3.2) than for female (+8.3) or neutral (+9.2) pronouns, so the improvement "can be uneven". On toxicity, 25,000 unprompted samples scored by PerspectiveAPI showed negligible differences (mean 0.087 against Gopher's 0.081) (paper, 4.2.7).

## Questions I had, answered

**Why did Kaplan et al. and this paper disagree?** The paper points to two reasons. First, the earlier study used one fixed learning-rate schedule for all runs, so its losses for shorter runs were overestimates and it underestimated the benefit of training small models on less data. Second, it used many very small models and so missed slight curvature in the compute-loss frontier (paper, 2).

**Is the "20 tokens per parameter" rule in the paper?** Not as a stated rule. It is a pattern I computed from the paper's Table 3 (the ratio stays between 20 and 22). The popular rule of thumb is consistent with that table.

**Do later studies agree with the paper's fit?** This is outside the paper. My vault's [[Scaling Laws]] note includes a figure from a later re-fit by Epoch AI, which disagrees with the original fit's tokens-per-parameter trend: the original rises with budget while the re-fit stays near 15 to 25 (that note's caption; I did not check the original source here).

![[chinchilla-token-param-ratio.png]]
*Not from the paper: how many training tokens per parameter to use as the compute budget grows. The green line is the original Chinchilla fit; the teal line is a later re-fit by Epoch AI, which stays near 15 to 25 with a wide uncertainty band; the dashed line is the "20 tokens per parameter" rule of thumb. The two fits disagree, a reminder that scaling laws are measured, not exact. Image: Epoch AI, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Optimal_ratio_of_training_tokens_to_model_parameters_for_Chinchilla_scaling_law.png), CC BY 4.0 (caption as in my Scaling Laws note).*

**Does this mean bigger models are pointless?** No: for a bigger budget the optimal model is bigger too (Table 3: 175 billion at about 4 times Gopher's compute). The claim is that for the budgets of the time models were too big and under-trained. And the paper adds a second benefit: a smaller model costs less to fine-tune and to run, which matters since "downstream fine-tuning and inference also make up substantial compute usage" (paper, 4).

**Why a 70B model when the fits said 40B to 67B?** The paper trained 70B "due to both dataset and computational efficiency considerations", at the larger end of the 40 to 70 billion range (paper, 4).

**Is data a limit?** The paper says so: "larger, high quality datasets will play a key role in any further scaling" and the projected token counts are far beyond current training sets (paper, 2 and 3.4).

## Why it mattered
It changed how models are planned: smaller, longer-trained models that are also cheaper to use.

## Limits and open questions
- The three approaches assume a power-law relationship; the paper notes there is slight curvature at large sizes (appendix E) and that extrapolating over orders of magnitude is uncertain (paper, 3 and 3.4).
- The training runs see less than one pass over the data (the paper's footnote 2 in section 1: "the number of training tokens is less than the number of tokens in the entire corpus"), so what happens when data is repeated is not tested here (my reading).
- Only two other data sets (C4 and GitHub code) repeat the IsoFLOP analysis, with the same conclusion (paper, 3.4 and Appendix C).
- The training set-up (learning-rate schedule, optimizer) is fixed by rules of thumb; only model size and token count are optimised (paper, 2).
- Chinchilla has not been checked against much bigger budgets: the largest compute budget tested is Gopher's, about $5.76\times10^{23}$ FLOPs (paper, 3).
- It says little about the quality of the data (the authors call for larger, high-quality datasets); the loss it measures is on a fixed data mixture.
- Bias and toxicity remain, and the improvement was uneven across groups (paper, 4.2.7).

## Concepts used
[[Scaling Laws]], [[LLM]], [[Pre-training]], [[Benchmarks]]

## Reading trail
- **Came from:** [[Scaling Laws 2020]] and [[GPT-3 2020]]
- **Led to:** [[LLaMA 2023]] (from memory: models trained for far more tokens than the earlier scaling advice suggested; I did not check this in either paper here)

## How much I read
The paper's body as text: pages 1 to 15 (introduction, related work, the three approaches, Tables 1 to 10, the Chinchilla results, the discussion start) and from the appendix the parametric-fit details (D.2) and the FLOPs computation (F). The figures were read through their captions and surrounding text, not looked at; the plot data are not reproduced. Not read: the reference list, the rest of the appendix (the data mixture, the learning-rate study B, the extra data sets C, the extra details D.1, D.3 and D.4, the curvature analysis E, the optimiser comparison G and the extra results H). The tables were written by me from the extracted text. All the numbers marked "mine" (the compute checks, the loss comparisons, the closed-form numbers, the ratios) are my own arithmetic in Python. Not opened in Obsidian, so the flashcards are unreviewed.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-llm Chinchilla (Hoffmann et al., 2022, arXiv 2203.15556): what problem did it attack?::For a fixed computing budget, how big should a language model be and how many tokens should it train on? Training is so costly it is usually done once, so the choice matters.

#flashcards/research/concept #flashcards/topic/research-llm What did people do before Chinchilla?::Followed Kaplan et al. (2020), whose scaling advice favoured very large models on modest data (for 10 times more compute, 5.5 times the model but only 1.8 times the tokens). Most large models were trained on about 300 billion tokens.

#flashcards/research/concept #flashcards/topic/research-llm Chinchilla: what was the idea?::Train over 400 models of many sizes and data amounts, find the best size and best tokens for each compute budget in three different ways, and fit how they grow with compute. Then test the prediction by training Chinchilla, with Gopher's compute but 4 times fewer parameters and 4 times more data.

#flashcards/research/concept #flashcards/topic/research-llm Chinchilla: what did it find? (abstract and paper)::Model size and training tokens should be scaled equally (doubling the model should double the tokens). Chinchilla (70B parameters, 1.4T tokens) outperforms Gopher (280B), GPT-3, Jurassic-1 and MT-NLG on a large range of tasks, with 67.6% average accuracy on MMLU, 7.6 above Gopher.

#flashcards/research/concept #flashcards/topic/research-llm Chinchilla: what are its limits?::The fits assume power laws and extrapolate over many orders of magnitude with significant uncertainty; the biggest budget tested is Gopher's; only model size and tokens are optimised; data quality is not addressed; bias and toxicity remain; a later re-fit by others disagrees on details.

#flashcards/research/concept #flashcards/topic/research-llm Chinchilla: why did it matter and what is its place in the reading trail?::It changed how models are planned: smaller, longer-trained models that are also cheaper to use. It came from the scaling-laws paper and GPT-3 and led to models such as LLaMA.

#flashcards/research/concept #flashcards/topic/research-llm How much of the Chinchilla paper did the note read?::The body as text (pages 1 to 15) plus appendix D.2 and F; the figures only through captions and text; not the references or most of the appendix. The tables are written by me from the paper.

#flashcards/research/concept #flashcards/topic/research-llm What are the three approaches in the Chinchilla paper, and what does each give?::1. Fix model sizes, vary tokens, take the minimum over training curves: $a=0.50$, $b=0.50$. 2. IsoFLOP profiles: for fixed compute, vary size and fit a parabola: $a=0.49$, $b=0.51$. 3. Fit the loss formula $E+A/N^\alpha+B/D^\beta$: $a=0.46$, $b=0.54$. All say size and tokens should grow about equally with compute.

#flashcards/research/concept #flashcards/topic/research-llm Why does the Chinchilla paper disagree with Kaplan et al.?::Kaplan et al. used one fixed learning-rate schedule for all runs, so losses for shorter runs were overestimates and training on less data looked worse; and they used many very small models, missing slight curvature in the frontier. Matching the schedule to the training length gives equal scaling.

#flashcards/research/concept #flashcards/topic/research-llm What were the steps for finding the compute-optimal size?
?
1. Fix a compute budget $C$. 2. List the (size, tokens) pairs that cost exactly $C$ (with $C\approx6ND$, $D=C/6N$). 3. Find the pair with the lowest loss (from curves, parabolas, or the formula). 4. Repeat for many budgets and fit power laws $N_{\text{opt}}\propto C^a$, $D_{\text{opt}}\propto C^b$. 5. Test the prediction by training that model.

#flashcards/research/concept #flashcards/topic/research-llm At Gopher's budget, how does the fitted loss formula vary with model size?::With $D=C/6N$ and the appendix constants (my arithmetic): 10B 1.947, 20B 1.933, 32B 1.931, 40B 1.931, 70B 1.938, 140B 1.955, 280B 1.984. A flat-bottomed valley: between 20B and 70B the loss stays within 0.007 of its minimum, while Gopher's 280B is 0.053 higher.

#flashcards/research/concept #flashcards/topic/research-llm Write the question the Chinchilla paper asks (Eq. 1) and explain each symbol.
?
$$N_{\text{opt}}(C),\ D_{\text{opt}}(C)=\mathop{\mathrm{argmin}}_{N,D\ \text{s.t.}\ \mathrm{FLOPs}(N,D)=C}\ L(N,D)$$
- $C$: compute budget; $N$: parameters; $D$: training tokens; $L(N,D)$: final loss of a model with $N$ parameters trained on $D$ tokens.
- Among all ways to spend $C$, choose the lowest loss; everything else in the paper estimates these two functions.

#flashcards/research/concept #flashcards/topic/research-llm Write the compute approximation $C\approx6ND$ and say what the 6 is.
?
$$C\approx6\,N\,D$$
- $C$: training FLOPs; $N$: parameters; $D$: tokens. The 6 is about 2 for the forward pass plus about 4 for the backward pass per parameter and token (the backward pass is assumed to cost twice the forward).
- It makes the budget a product of size and data. Example: Chinchilla $6\times70\text{B}\times1.4\text{T}=5.88\times10^{23}$, Gopher $5.04\times10^{23}$ (the paper's exact count: about $5.76\times10^{23}$ for both).

#flashcards/research/concept #flashcards/topic/research-llm Write the Chinchilla loss formula (Eq. 2) and explain each term.
?
$$\hat L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}$$
- $E$: loss of an ideal model on this data (the entropy of text; a floor). $A/N^\alpha$: how much a perfectly trained model of $N$ parameters falls short; shrinks as $N$ grows. $B/D^\beta$: the shortfall from finite training on finite data; shrinks as $D$ grows.
- Fitted: $E=1.69$, $A=406.4$, $B=410.7$, $\alpha=0.34$, $\beta=0.28$. At a fixed budget the last two terms pull in opposite directions.

#flashcards/research/concept #flashcards/topic/research-llm Loss formula example: what are the predicted losses of Chinchilla and Gopher?
?
Chinchilla ($70\text{B}$, $1.4\text{T}$): $1.69+0.084+0.163=1.937$. Gopher ($280\text{B}$, $300\text{B}$): $1.69+0.052+0.250=1.993$. Chinchilla's loss is about 0.057 lower: Gopher's size shrinks the second term by 0.03 but its small data leaves the third term 0.09 higher (my arithmetic with the rounded constants).

#flashcards/research/concept #flashcards/topic/research-llm Write the Chinchilla fitting objective (Eq. 3) and say why the Huber loss is used.
?
$$\min_{A,B,E,\alpha,\beta}\sum_{\text{runs }i}\mathrm{Huber}_\delta\big(\log\hat L(N_i,D_i)-\log L_i\big)$$
- Sum over all runs $i$ of the error of the formula in log terms. Huber is quadratic for small errors and linear for large ones ($\delta=10^{-3}$); fitted with L-BFGS from a grid of starting points.
- It is robust to outliers: with $\delta=0.001$, an error of 0.05 is charged $4.95\times10^{-5}$ instead of $1.25\times10^{-3}$, 25 times less (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Write the closed-form compute-optimal frontier (Eq. 4) and explain each symbol.
?
$$N_{\text{opt}}=G\Big(\frac{C}{6}\Big)^{a},\quad D_{\text{opt}}=G^{-1}\Big(\frac{C}{6}\Big)^{b},\quad G=\Big(\frac{\alpha A}{\beta B}\Big)^{\frac{1}{\alpha+\beta}},\quad a=\frac{\beta}{\alpha+\beta},\quad b=\frac{\alpha}{\alpha+\beta}$$
- $C/6=N\cdot D$ (from $C\approx6ND$); $G$ is a constant from $A$, $B$, $\alpha$, $\beta$; $a+b=1$.
- Minimising $\hat L$ under the budget balances the two data-dependent terms and gives power laws. If $\alpha\approx\beta$ then $a\approx b\approx1/2$: equal growth.

#flashcards/research/concept #flashcards/topic/research-llm Closed-form example: with $\alpha=0.34$, $\beta=0.28$, $A=406.4$, $B=410.7$, what are $a$, $b$, $G$ and the optimum for Gopher's budget?
?
$a=0.28/0.62=0.452$, $b=0.548$, $G=1.345$. For $C=5.76\times10^{23}$: $N_{\text{opt}}=32$ billion, $D_{\text{opt}}=3.0$ trillion. The paper reports $a=0.46$ and 40 billion; my numbers from the rounded appendix constants are close but do not reproduce them exactly, probably from rounding (not checked).

#flashcards/research/concept #flashcards/topic/research-llm What did Chinchilla's Table 2 show about the exponents, and how do they compare with Kaplan et al.?::The three approaches give $a$ of 0.46 to 0.50 and $b$ of 0.50 to 0.54: size and tokens grow about equally. Kaplan et al.: $a=0.73$, $b=0.27$. For 10 times more compute: about 5.4 times the model and 1.9 times the tokens (Kaplan) against about 3.2 times each (equal scaling) (my arithmetic). Approach 3's narrow interval reflects one fit resampled, not independent agreement.

#flashcards/research/concept #flashcards/topic/research-llm What pattern does the paper's Table 3 show about tokens and parameters?::The optimal tokens are about 20 to 22 times the parameters at every size from 400 million (8.0 billion tokens) to 10 trillion (216.2 trillion): 1 billion parameters needs 20.2 billion tokens, 175 billion needs 3.7 trillion. The "20 tokens per parameter" is my reading of the table, not a stated rule. A 175-billion model should see 12 times the 300 billion tokens GPT-3 saw.

#flashcards/research/concept #flashcards/topic/research-llm How did Chinchilla differ from Gopher in design and training (Table 4 and 4.1)?::Same 80 layers and key/value size 128, but half the width (8,192 against 16,384) and half the heads (64 against 128), giving 70B against 280B parameters; max learning rate $10^{-4}$ against $4\times10^{-5}$; batch 1.5M to 3M against 3M to 6M tokens. Also AdamW instead of Adam, a tokenizer without NFKC normalisation, a float32 weight copy, and a slightly different data mixture.

#flashcards/research/concept #flashcards/topic/research-llm How did the large models of the day compare in tokens per parameter (paper's Table 1)?::LaMDA 1.2, GPT-3 1.7, Jurassic 1.7, Gopher 1.1, MT-NLG 0.5, against Chinchilla's 20.0 (my arithmetic). All but LaMDA saw about 300 billion tokens (MT-NLG 270 billion), whatever the size.

#flashcards/research/concept #flashcards/topic/research-llm How did Chinchilla do on MMLU (paper's Table 6)?::67.6% average 5-shot over 57 tasks: +7.6 over Gopher (60.0), +23.7 over GPT-3 (43.9), above the forecast for June 2023 (63.4), but far below human experts (89.8). Better on 51 of 57 tasks, same on 2, worse on 4. The abstract says 67.5%, a small mismatch with the table.

#flashcards/research/concept #flashcards/topic/research-llm How did Chinchilla do on reading, common sense and closed-book questions (Tables 7 to 9)?::It beat Gopher on every row where both are reported except a tie on PIQA (81.8): RACE-h +10.7, RACE-m +11.7, BoolQ +4.4, Winogrande +4.8, Natural Questions +7.0 (5-shot) and +7.3 (64-shot), zero-shot TriviaQA +14.2. It beat the 530B MT-NLG on all common-sense tasks but one (PIQA). It stays far below supervised or open-book systems (51.3 against 83.2 on SIQA).

#flashcards/research/concept #flashcards/topic/research-llm What did the Chinchilla paper find about bias and toxicity?::Winogender: more accurate than Gopher in every group (78.3% against 71.4%), but the gain is smaller for male pronouns (+3.2) than female (+8.3) or neutral (+9.2), so improvements are uneven. Toxicity: 25,000 unprompted samples showed negligible differences (mean 0.087 against Gopher's 0.081).

#flashcards/research/concept #flashcards/topic/research-llm ![[chinchilla-token-param-ratio.png]]
What does this picture show?
?
Not from the paper: how many training tokens per parameter to use as compute grows. The green line is the original Chinchilla fit; the teal line is a later re-fit by Epoch AI that stays near 15 to 25 with a wide uncertainty band; the dashed line is the "20 tokens per parameter" rule of thumb. The two fits disagree: scaling laws are measured, not exact. Image: Epoch AI (Wikimedia Commons, CC BY 4.0).

#flashcards/research/concept #flashcards/topic/research-llm Does the Chinchilla result mean bigger models are pointless?::No: for bigger budgets the optimal model is bigger too (175 billion at about 6.7 times Gopher's compute). The claim is that for the budgets of the time models were too big and under-trained. A smaller model is also cheaper to fine-tune and run, which matters since fine-tuning and inference are a substantial share of compute.
