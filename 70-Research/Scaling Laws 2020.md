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

Paper: "Scaling Laws for Neural Language Models", arXiv 2001.08361, posted 23 January 2020 (paper's first page). Concept note: [[Scaling Laws]]. Its main follow-up in this vault is [[Chinchilla 2022]].

## The problem
How good a language model gets depends on its architecture, its size, the computer time spent training it, and the amount of data. Before this paper there was no clear, quantitative picture of how these trade off for Transformers. If you have a fixed budget, should you build a bigger model or train a smaller one for longer? The authors measure the dependence of the loss on each factor, focusing on the [[Transformer]] and the cross-entropy loss ([[Cross-Entropy Loss]], [[Perplexity]]) (paper).

## What people did before
- **Earlier scaling work:** some early work found power-law links between performance and dataset size, and more recent work looked at model size against data size. The paper calls Hestness et al. (2017) and a 2019 follow-up the closest. They found the amount of data needed grows **faster** than linearly with model size, whereas this paper finds it grows **slower** than linearly (paper, related work).
- **Typical practice:** train smaller models until they converge. The paper says researchers usually train smaller models for longer than would be compute-efficient, because of hardware limits (paper).
- **Competing explanations:** other work argued that architectural details such as depth and width matter a lot (for example EfficientNet's recipe for images). This paper finds they matter little for language models (paper).

## The idea
1. **Train a very wide range of models.** From 768 to 1.5 billion non-embedding parameters, on datasets from 22 million to 23 billion tokens, with different shapes (depth, width, attention heads, feed-forward size), context lengths and batch sizes. The data is WebText2, an extended version of the WebText data used for GPT-2, with 22.9 billion tokens after tokenisation ([[Byte-Pair Encoding]], vocabulary 50,257) (paper).
2. **Define "size" as non-embedding parameters $N$.** The count excludes the vocabulary and position embeddings, which gives much cleaner trends than the total count (paper, section 2.1 and Figure 6).
3. **Estimate compute as $C\approx6NBS$:** about 6 floating-point operations per parameter per training token (2 for the forward pass and about 4 for the backward pass), with batch size $B$ and $S$ training steps (paper).
4. **Measure the test loss** as a function of $N$, the dataset size $D$ (in tokens) and the compute, each while the other two are not the bottleneck, and fit power laws.
5. **Combine** them into single formulas for the loss in terms of $N$ and $D$ (which also gives the amount of overfitting), and in terms of $N$ and training steps.
6. **Use the formulas** to say how a fixed compute budget should be split between model size, batch size and number of steps (paper).

## What they found
Each number is marked "abstract" or "paper" (read in the paper's body text, as extracted from the PDF). The numerical constants depend on the vocabulary and tokenisation, so they have no fundamental meaning (paper).
- **Smooth power laws (abstract and paper):** the loss follows a power law in model size, dataset size and compute, with some trends spanning more than seven orders of magnitude (abstract; the introduction says "more than six orders" for $N$, $D$ and $C$ in Figure 1).
- **Model size alone** (trained to convergence on enough data): $L(N)=(N_c/N)^{\alpha_N}$ with $\alpha_N\approx0.076$ and $N_c\approx8.8\times10^{13}$. Doubling the parameters multiplies the loss by $2^{-0.076}=0.95$ (paper).
- **Dataset size alone** (large model, early stopping): $\alpha_D\approx0.095$ and $D_c\approx5.4\times10^{13}$ tokens. **Compute alone** (optimal model size, small enough batch): $\alpha_C^{\min}\approx0.050$ (paper, equations 1.2 and 1.3).
- **Shape matters little (paper, Figure 5 and 6):** at a fixed $N$, a wide range of depth and width ratios reach similar loss. Aspect ratio can vary by a factor of 40 while only slightly changing the loss, and a 6-layer, 4288-wide model reaches a loss within 3% of a 48-layer, 1600-wide model. Models with fewer than 2 layers or with extreme shapes are the exceptions.
- **Overfitting depends on $N^{0.74}/D$ (paper):** the penalty depends predictably on that ratio, so every time you increase the model size 8 times you need only about 5 times more data to avoid a penalty. To keep the loss within about 0.02 of convergence, $D\gtrsim5\times10^3\,N^{0.74}$ tokens.
- **Optimal allocation of a fixed compute budget (paper, section 6):** the best model size grows as $C_{\min}^{0.73}$, the batch size as $C_{\min}^{0.24}$ and the number of serial steps as $C_{\min}^{0.03}$ (possibly zero), with data as $D\propto C^{0.27}$. Most of an increase in compute should go into a bigger model: a 10-fold increase in compute gives about a 5-fold bigger model and about a 2-fold increase in data processed.
- **Sample efficiency and early stopping (abstract and paper):** larger models reach the same loss with fewer samples, so optimally compute-efficient training uses very large models on a relatively modest amount of data and stops significantly before convergence (abstract).
- **Critical batch size (paper):** it depends on the loss only, not on the model size, and roughly doubles for every 13% decrease in the loss. It is 1 to 2 million tokens for the largest models at convergence.
- **Transformers against LSTMs (paper, Figure 7):** LSTMs match Transformers on tokens early in the context but plateau after about 100 tokens, while Transformers keep improving through the whole context.
- **Transfer (paper):** on other text distributions (Books, Common Crawl, Wikipedia) the loss improves in parallel with the training distribution, with a roughly constant offset and almost no dependence on how long the model was trained.
- **A predicted breakdown (paper, section 6.3):** extrapolated far enough, the compute-efficient law and the data law contradict each other. The two curves meet at roughly $C^*\sim10^4$ PF-days, $N^*\sim10^{12}$ parameters, $D^*\sim10^{12}$ tokens and $L^*\sim1.7$ nats per token, "highly uncertain" by an order of magnitude either way. The authors conjecture that this might even estimate the entropy of text.
- **Their reading (paper, discussion):** "Big models may be more important than big data."

## The main formulas, in detail
**Loss against model size.** For models trained to convergence on enough data:

$$
L(N)=\left(\frac{N_c}{N}\right)^{\alpha_N},\qquad \alpha_N\approx0.076,\quad N_c\approx8.8\times10^{13}
$$

**Where:**
- $L$ is the test loss in nats per token (the cross-entropy; lower is better).
- $N$ is the number of non-embedding parameters.
- $N_c$ is a constant that depends on the tokenisation and has no fundamental meaning. $\alpha_N$ is the exponent: the smaller it is, the slower the loss falls.
- **Matters:** because it is a power law, every doubling of $N$ multiplies the loss by the same factor, $2^{-\alpha_N}$, which is about 0.95. The returns are steady but small: you need very large changes in size to see large changes in loss. A straight line on a log-log plot is the signature.

**Example (tiny numbers):** using the paper's constants.
- $N=10^8$: $L=(8.8\times10^{13}/10^8)^{0.076}=(8.8\times10^{5})^{0.076}=\mathbf{2.83}$.
- $N=2\times10^8$: $L=\mathbf{2.685}$. The ratio is $2.685/2.83=0.949=2^{-0.076}$.
- $N=10^9$: $L=\mathbf{2.376}$.
- Meaning: doubling the model removes about 5% of the loss, and a 10-fold increase from $10^8$ to $10^9$ takes it from 2.83 to 2.38. (My arithmetic from the paper's constants.)

**Loss against model size and data together.** With early stopping, the joint fit is:

$$
L(N,D)=\left[\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D}+\frac{D_c}{D}\right]^{\alpha_D}
$$

**Where:**
- $D$ is the dataset size in tokens. $D_c$ is a constant (tokenisation-dependent).
- $\alpha_D$ is the data exponent. The paper's fit of this form (its Table 2) gives $\alpha_N=0.076$, $\alpha_D=0.103$, $N_c=6.4\times10^{13}$ and $D_c=1.8\times10^{13}$. These differ slightly from the single-factor fits because here both are fitted at once.
- **Matters:** with huge $D$ the second term in the bracket vanishes and you get $L(N)$. With a huge $N$ the first term vanishes and you get $L(D)$. In between, each one bottlenecks the other, and the second term measures the overfitting penalty. If either $N$ or $D$ is held fixed, more of the other brings diminishing returns.

**Example (tiny numbers):** $N=10^8$ and the Table 2 constants.
- $D=10^{10}$ tokens: the first term is $(6.4\times10^5)^{0.738}\approx19{,}239$ and $D_c/D=1{,}800$. $L=(19{,}239+1{,}800)^{0.103}=\mathbf{2.788}$, against $2.762$ with infinite data: an overfitting penalty of about $\mathbf{0.9\%}$.
- $D=10^9$ tokens: $D_c/D=18{,}000$, so $L=\mathbf{2.957}$, a penalty of about $\mathbf{7.0\%}$.
- Meaning: ten times less data makes the penalty seven to eight times bigger for this model, so a model of this size is clearly starved with $10^9$ tokens. (My arithmetic from the paper's fit.)

**How much compute training costs.** The paper estimates the non-embedding training compute as:

$$
C\approx6\,N\,B\,S
$$

**Where:**
- $N$ is the number of non-embedding parameters. $B$ is the batch size in tokens and $S$ is the number of training steps, so $B\,S$ is the number of tokens processed.
- The factor 6 is about 2 floating-point operations per parameter per token for the forward pass plus about 4 for the backward pass (the backward pass costs roughly twice the forward pass).
- $C$ is counted in floating-point operations. The paper reports it in PF-days: one PF-day is $10^{15}\times86{,}400=8.64\times10^{19}$ operations.
- **Matters:** it leaves out the cost that depends on the context length, which is fine unless the context is very long compared with the model width. It makes training cost proportional to both model size and data, which is why the choice of how to split a budget between them is the whole question.

**Example (tiny numbers):** $N=10^9$ parameters and $B\,S=2\times10^{10}$ tokens.
- $C=6\times10^9\times2\times10^{10}=\mathbf{1.2\times10^{20}}$ operations.
- In PF-days: $1.2\times10^{20}/8.64\times10^{19}=\mathbf{1.39}$ PF-days.
- Meaning: training a billion-parameter model on 20 billion tokens costs a bit more than one day of a computer that does $10^{15}$ operations a second. (My arithmetic.)

**How to split a growing budget.** With the compute spent as efficiently as possible ($C_{\min}$), the paper finds:

$$
N\propto C_{\min}^{\,0.73},\qquad B\propto C_{\min}^{\,0.24},\qquad S\propto C_{\min}^{\,0.03},\qquad D=B\,S\propto C_{\min}^{\,0.27}
$$

**Where:**
- $C_{\min}$ is the least compute needed to reach a given loss (the compute you would use at a small enough batch size).
- $N$, $B$, $S$ and $D$ are the best model size, batch size, number of steps and amount of data for that budget.
- The exponents add up: $0.24+0.03=0.27$ for data, and the paper's theory from its loss formula predicts $N\propto C^{0.71}$, close to the measured 0.73.
- **Matters:** nearly all of an increase in compute should go into the model size and the batch size. The number of serial steps hardly changes, so the best training run barely gets longer. This is why the paper says to train very large models and stop well short of convergence.

**Example (tiny numbers):** a 10-fold larger compute budget.
- Model size: $10^{0.73}=\mathbf{5.4}$ times larger.
- Batch size: $10^{0.24}=\mathbf{1.7}$ times larger. Steps: $10^{0.03}=\mathbf{1.07}$ times more.
- Data: $10^{0.27}=\mathbf{1.9}$ times more.
- Meaning: ten times the compute buys a model more than 5 times bigger but only about twice the data. This is the allocation that [[Chinchilla 2022]] later disputed. (My arithmetic from the paper's exponents.)

**Data needed to avoid overfitting.** From the $L(N,D)$ fit, to stay within about 0.02 of the converged loss:

$$
D\gtrsim5\times10^{3}\,N^{0.74}
$$

**Where:**
- $D$ is the number of training tokens and $N$ the number of non-embedding parameters.
- The $0.02$ is the typical variation of the loss between random seeds, so a smaller penalty would be hidden in the noise.
- **Matters:** data has to grow more slowly than the model, since $0.74<1$. A model 8 times larger needs only $8^{0.74}\approx4.7$ times more data. It is a minimum to avoid overfitting, not the amount that makes training most compute-efficient.

**Example (tiny numbers):**
- $N=10^9$: $D\gtrsim5\times10^3\times(10^9)^{0.74}=\mathbf{2.3\times10^{10}}$ tokens, about the 22.9 billion tokens of WebText2, which is why the paper's largest models start to overfit a little.
- $N=10^8$: $D\gtrsim\mathbf{4.2\times10^9}$ tokens.
- Meaning: a 100-million-parameter model needs at least about 4 billion tokens, and a billion-parameter model at least about 23 billion. (My arithmetic.)

## Why it mattered
It turned "bigger is better" into a quantitative, predictable rule for choosing model size, data and compute, and it argued for putting most extra compute into larger models. [[GPT-3 2020]] was trained by the same lab (OpenAI) and was a very large model; that the paper shaped it is from memory and not checked here. The paper's allocation advice was later revised: [[Chinchilla 2022]] (see that note) reported that model size and training tokens should be scaled equally, and that current large models were under-trained. Today scaling laws are also used to plan training runs and compare architectures (from memory).

## Limits and open questions
All from the paper's own caveats (appendix) and discussion unless marked.
- **No theory:** the authors have no solid theoretical understanding of the laws, "especially" the ones for model size and compute, so it is hard to know in what circumstances they can be trusted.
- **Out-of-range predictions:** they are not confident about the critical batch size far outside the range they explored, and the laws must break down at large scale because language has non-zero entropy (the contradiction above).
- **Small-data regime:** their fits were poor for the smallest datasets (an epoch of only 40 steps), and they did not experiment with regularisation or data augmentation, which could change the results.
- **Compute estimate:** $C\approx6NBS$ leaves out terms that depend on context length, so the compute scalings may be confounded when the context is very long (about 12 times the model width or more).
- **Possibly untuned hyperparameters:** they tuned learning rates and schedules, but may have missed one (for example the initial scale), and the best learning rate depends on the target loss.
- **Loss, not abilities:** whether lower loss translates into better results on real tasks is open. The authors warn that smooth improvement in the loss can hide qualitative changes ("more is different").
- **One data set and one architecture family:** the fitted constants are for WebText2 and Transformers. The authors conjecture that the form carries to other generative models but do not test it here.
- **Later correction (from memory and the Chinchilla note):** the compute-optimal split between parameters and tokens in this paper is not the one later found by [[Chinchilla 2022]].

## Concepts used
- [[Scaling Laws]] · [[Transformer]] · [[Cross-Entropy Loss]] · [[Perplexity]] · [[Byte-Pair Encoding]] · [[Overfitting]] · [[Adam Optimizer]] · [[LSTM]] · [[Pre-training]] · [[Context Window]] · [[Learning Rate]] · [[Data Parallelism]] · [[Model Parallelism]] · [[Benchmarks]]

## Reading trail
- **Came from:** [[Attention Is All You Need 2017]] (the architecture it studies).
- **Led to:** [[Chinchilla 2022]] (revised the compute-optimal allocation) and, in the same lab a few months later, [[GPT-3 2020]] (that GPT-3 was shaped by it is from memory).

## How much I read
I downloaded the full paper (30 pages) and read the extracted text of: the abstract, the introduction and summary of results (sections 1.1 and 1.2), the methods (sections 2.1 to 2.3), the basic power laws (section 3), overfitting and the $L(N,D)$ fit (section 4, including Table 2), the training-time law and critical batch size (section 5, including Table 3), the optimal allocation of compute (section 6, including the contradiction in section 6.3), related work, the discussion, the table of power laws in appendix A and the caveats in appendix C.

**Not read:** the remaining appendix material (the detailed derivation of the compute-efficient frontier in appendix B and the supplemental figures in appendix D, apart from their captions). The figures were read only through their captions and the surrounding text. The fitted constants come from text extracted from the PDF; I matched each to its equation. I did not re-run any experiment. Arithmetic in the worked examples is mine. The note was not opened in Obsidian.

## Sources
- Kaplan et al., "Scaling Laws for Neural Language Models", arXiv 2001.08361 (23 January 2020): https://arxiv.org/abs/2001.08361

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-llm Scaling Laws (Kaplan, McCandlish et al., OpenAI, 2020, arXiv 2001.08361): what did it study? (abstract, paper)::How the cross-entropy loss of Transformer language models depends on model size (non-embedding parameters N), dataset size D and training compute C. It found power laws, with some trends spanning more than seven orders of magnitude.

#flashcards/research/concept #flashcards/topic/research-llm Scaling Laws 2020: why are embedding parameters left out of N? (paper)::Using non-embedding parameters gives much cleaner scaling laws. With total parameters the trend depends on the number of layers, and without embeddings the models of different depths fall on one trend.

#flashcards/research/concept #flashcards/topic/research-llm Scaling Laws 2020: does model shape matter? (paper)::Very little at a fixed N. Aspect ratio can vary by a factor of 40 with only a slight effect, and a 6-layer 4288-wide model is within 3% of the loss of a 48-layer 1600-wide one. Scale matters far more than shape.

#flashcards/research/concept #flashcards/topic/research-llm Scaling Laws 2020: how fast does the data need to grow with model size? (paper)::To avoid overfitting, D grows as about N to the power 0.74: every 8 times larger model needs only about 5 times more data. A rule of thumb from the paper: D of at least 5 thousand times N^0.74 tokens.

#flashcards/research/concept #flashcards/topic/research-llm Scaling Laws 2020: how should extra compute be spent? (paper)::Mostly on a bigger model. Optimal N grows as C^0.73, batch size as C^0.24, serial steps as C^0.03, and data as about C^0.27. A 10-fold compute increase gives about a 5-fold larger model and about 2 times more data processed. Training should stop well before convergence.

#flashcards/research/concept #flashcards/topic/research-llm Write the loss against model size and explain each symbol. (paper)
?
$$L(N)=\left(\frac{N_c}{N}\right)^{\alpha_N},\qquad \alpha_N\approx0.076,\quad N_c\approx8.8\times10^{13}$$
- $L$: the test loss in nats per token. $N$: the number of non-embedding parameters.
- $N_c$: a tokenisation-dependent constant with no fundamental meaning. $\alpha_N$: the exponent, small so the loss falls slowly.
- Every doubling of $N$ multiplies the loss by $2^{-\alpha_N}\approx0.95$: steady but small returns. A straight line on a log-log plot.

#flashcards/research/concept #flashcards/topic/research-llm With $N_c=8.8\times10^{13}$ and $\alpha_N=0.076$, what are the losses for $N=10^8$, $2\times10^8$ and $10^9$?::2.83, 2.685 and 2.376. Doubling multiplies the loss by 2^-0.076 = 0.949. My arithmetic from the paper's constants.

#flashcards/research/concept #flashcards/topic/research-llm Write the joint loss for model size and data and explain each symbol. (paper)
?
$$L(N,D)=\left[\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D}+\frac{D_c}{D}\right]^{\alpha_D}$$
- $D$: the dataset size in tokens. $D_c$: a tokenisation-dependent constant. $\alpha_D$: the data exponent.
- The paper's fit (Table 2): $\alpha_N=0.076$, $\alpha_D=0.103$, $N_c=6.4\times10^{13}$, $D_c=1.8\times10^{13}$.
- Huge $D$ leaves $L(N)$, huge $N$ leaves $L(D)$. In between each bottlenecks the other, and the second term in the bracket measures the overfitting penalty.

#flashcards/research/concept #flashcards/topic/research-llm For $N=10^8$ with the Table 2 fit, what is the loss and the overfitting penalty at $D=10^{10}$ and $D=10^9$ tokens?::D = 10^10: L = 2.788 against 2.762 with infinite data, a penalty of about 0.9%. D = 10^9: L = 2.957, a penalty of about 7.0%. Ten times less data makes the penalty 7 to 8 times bigger. My arithmetic from the paper's fit.

#flashcards/research/concept #flashcards/topic/research-llm Write the paper's estimate of training compute and explain each symbol. (paper)
?
$$C\approx6\,N\,B\,S$$
- $N$: non-embedding parameters. $B$: batch size in tokens. $S$: training steps, so $BS$ is the tokens processed.
- The factor 6: about 2 operations per parameter per token for the forward pass plus about 4 for the backward pass.
- $C$ counts floating-point operations. One PF-day is $8.64\times10^{19}$ operations.
- It leaves out context-length-dependent costs, fine unless the context is very long compared with the model width.

#flashcards/research/concept #flashcards/topic/research-llm What is the training compute for $N=10^9$ parameters and $2\times10^{10}$ tokens?::C = 6 × 10^9 × 2 × 10^10 = 1.2 × 10^20 operations, which is 1.2e20 / 8.64e19 = 1.39 PF-days. My arithmetic.

#flashcards/research/concept #flashcards/topic/research-llm Write how the best model size, batch size, steps and data grow with compute, and explain each symbol. (paper)
?
$$N\propto C_{\min}^{\,0.73},\qquad B\propto C_{\min}^{\,0.24},\qquad S\propto C_{\min}^{\,0.03},\qquad D=B\,S\propto C_{\min}^{\,0.27}$$
- $C_{\min}$: the least compute needed to reach a given loss. $N$, $B$, $S$, $D$: the best model size, batch size, steps and data for that budget.
- The exponents add up: 0.24 + 0.03 = 0.27 for data. The paper's theory predicts $N\propto C^{0.71}$ against the measured 0.73.
- Nearly all extra compute goes to model size and batch size, while the number of serial steps barely changes. So train very large models and stop short of convergence.

#flashcards/research/concept #flashcards/topic/research-llm With 10 times more compute, how much larger should the model, batch, steps and data be?::Model: 10^0.73 = 5.4 times. Batch: 10^0.24 = 1.7 times. Steps: 10^0.03 = 1.07 times. Data: 10^0.27 = 1.9 times. Chinchilla 2022 later disputed this split. My arithmetic from the paper's exponents.

#flashcards/research/concept #flashcards/topic/research-llm Write the data needed to avoid overfitting and explain each symbol. (paper)
?
$$D\gtrsim5\times10^{3}\,N^{0.74}$$
- $D$: training tokens. $N$: non-embedding parameters.
- It keeps the loss within about 0.02 of the converged loss (0.02 is the seed-to-seed variation).
- Data grows more slowly than the model because 0.74 is below 1: a model 8 times larger needs about 4.7 times more data. It is a minimum to avoid overfitting, not the most compute-efficient amount.

#flashcards/research/concept #flashcards/topic/research-llm How many tokens does the rule need for $N=10^9$ and $N=10^8$?::10^9 parameters: at least 5e3 × (1e9)^0.74 = 2.3 × 10^10 tokens, close to the 22.9 billion tokens of WebText2. 10^8 parameters: at least 4.2 × 10^9 tokens. My arithmetic.

#flashcards/research/concept #flashcards/topic/research-llm Scaling Laws 2020: what did it say about Transformers and LSTMs? (paper)::LSTMs match Transformers on tokens early in the context but plateau after about 100 tokens. Transformers keep improving through the whole context, so they pull ahead at scale.

#flashcards/research/concept #flashcards/topic/research-llm Scaling Laws 2020: what is the critical batch size? (paper)::The batch size that makes a roughly optimal compromise between training time and compute. It depends on the loss only, not on model size, and roughly doubles for every 13% decrease in loss. It is 1 to 2 million tokens for the largest models at convergence.

#flashcards/research/concept #flashcards/topic/research-llm Scaling Laws 2020: what contradiction did the authors find at extreme scale? (paper)::Extrapolating far enough, the compute-efficient law and the data law cross, at about 10^4 PF-days, 10^12 parameters, 10^12 tokens and 1.7 nats per token, with an uncertainty of an order of magnitude. The laws must break down at or before that point.

#flashcards/research/concept #flashcards/topic/research-llm Scaling Laws 2020: what are its main limits? (paper)::No solid theory. Unknown reliability outside the explored range. Poor fits for small datasets, and no study of regularisation or augmentation. Compute estimate ignores context-length terms. Some hyperparameters may be untuned. It measures loss, not abilities. One data set and one architecture family. Its compute-optimal split was later revised by Chinchilla.

#flashcards/research/concept #flashcards/topic/research-llm Scaling Laws 2020: what is its place in the reading trail?::It came from the Transformer paper (the architecture it studies). It led to Chinchilla 2022, which revised the compute-optimal allocation, and to GPT-3 from the same lab a few months later (that link is from memory).

#flashcards/research/concept #flashcards/topic/research-llm How much of the Scaling Laws paper did the note read, and what was not read?::Abstract, introduction, methods, the basic power laws, overfitting, training-time law, compute allocation, related work, discussion and the caveats. Not read: appendix B (the derivation of the compute-efficient frontier) and the supplemental figures except their captions. Figures only through captions. No experiment re-run.
