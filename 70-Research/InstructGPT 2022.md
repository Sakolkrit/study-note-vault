---
tags: [paper, NLP, alignment, RLHF]
link: https://arxiv.org/abs/2203.02155
authors: Ouyang, Wu, Jiang, Almeida, Wainwright, Mishkin and 14 others including Schulman, Christiano, Leike and Lowe (OpenAI; checked on the paper's first page)
year: 2022
status: body-read
field: text analytics
---

# InstructGPT 2022

> **In one line for a 12-year-old:** Big language models were good at continuing text but not at doing what you asked. People wrote good answers and ranked answers, and the model was trained to give the kind of answers people liked. A small model trained this way was preferred over one 100 times bigger.

Paper: "Training language models to follow instructions with human feedback", arXiv 2203.02155, version 1 of 4 March 2022, 68 pages. Marks: "(paper, section)" was read in the paper; "(general knowledge)" is my own background; "(mine)" is my own arithmetic, done in Python. This paper's arXiv licence is not open (the usual arXiv licence), so its figures are not copied; a free redrawing of the general method is used instead.

## The problem
Making language models bigger "does not inherently make them better at following a user's intent": large models "can generate outputs that are untruthful, toxic, or simply not helpful to the user", that is, they are "not aligned with their users" (abstract). The reason, the paper says, is that the objective used to train them, predicting the next word on a web page, "is different from the objective 'follow the user's instructions helpfully and safely'", so "the language modeling objective is misaligned" (paper, 1). The authors want models that are **helpful** (solve the user's task), **honest** (do not fabricate) and **harmless** (do not cause harm), following Askell and others (paper, 1). See [[Alignment]].

## What people did before
- **Plain pre-trained models such as [[GPT-3 2020]],** which can be steered by a prompt with examples (few-shot), but "often express unintended behaviors" (paper, 1).
- **Instruction tuning on public NLP data** (FLAN, T0 and others): fine-tune a model on many NLP datasets, each turned into an instruction. This improves performance on held-out tasks (paper, 2). The paper compares against FLAN and T0 and finds they do not fit its prompts as well (below).
- **RLHF for simple robots and games, then for summarisation** (Christiano and others; Stiennon and others): learn a reward from human comparisons and train with reinforcement learning. This paper calls its work "a direct application of RLHF to aligning language models on a broad distribution of language tasks" (paper, 2). See [[RLHF]].

## The idea
Teach the model what people want in three steps (paper, 3.1 and Figure 2): (1) have people write the answers they want and fine-tune the model on them (supervised fine-tuning, SFT, see [[Instruction Tuning]]); (2) have people rank several answers the model gives to the same prompt, and train a second model, the **reward model**, to predict which answer people prefer ([[Reward Model]]); (3) use the reward model's score as the reward in reinforcement learning to improve the first model ([[PPO]]). Steps 2 and 3 can be repeated. All models use the GPT-3 architecture and come in three sizes, 1.3B, 6B and 175B parameters (paper, 1). The resulting models are called InstructGPT. See [[Fine-tuning]].

## The method, part by part

![[rlhf-diagram.svg]]
*A general RLHF pipeline: a supervised model is trained on prompt and response data; its responses to prompts are ranked by human annotators; the rankings train a reward model; and the reward model guides training of the aligned model through PPO. Figure: PopoDameron, from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:RLHF_diagram.svg), CC BY-SA 4.0. It is a redrawing of the general idea, not the paper's Figure 2: the paper's Figure 2 shows three numbered steps with boxes A to D ranked by a labeler; this drawing says the rankings are converted to scalar scores "using the Elo algorithm", which InstructGPT does not do (it trains the reward model directly on pairwise comparisons, equation 2 below); and it initialises both the reward model and the aligned model from the supervised model, which matches the paper only partly (the paper's reward model was a 6B model, see below).*

- **The prompts and the three datasets.**
  - *What it does:* prompts come mostly from "text prompts submitted to the OpenAI API" using an earlier InstructGPT version in the Playground, plus prompts written by labelers (paper, 3.2). From them come three datasets: the SFT dataset (demonstrations), the RM dataset (rankings) and the PPO dataset (prompts only, no labels) (paper, 3.2).
  - *Why it matters:* the prompts are real use, not an academic data set. Most are "generative, rather than classification or QA" (paper, 3.2). Prompts were deduplicated, limited to 200 per user, filtered for personal information, and train, validation and test were split by user so no user is in two splits.
- **The labelers.**
  - *What it does:* about 40 contractors hired through Upwork and ScaleAI, chosen by a screening test of how well they judged sensitive content and agreed with researchers (paper, 3.4).
  - *Why it matters:* the model learns their preferences. The paper says plainly it is "aligning to a set of labelers' preferences" influenced by the instructions they were given, and "to our preferences, as the researchers" (paper, 5.2). Their agreement with each other was 72.6 ± 1.5% (77.3 ± 1.3% for held-out labelers).
- **Step 1: supervised fine-tuning (SFT).**
  - *What it does:* fine-tune pre-trained GPT-3 on labeler demonstrations for 16 epochs, with a cosine learning-rate decay and residual dropout 0.2 (paper, 3.5).
  - *Why it matters:* it gives the model the basic shape of an answer to an instruction. The paper notes the SFT models overfit the validation loss after one epoch, yet "training for more epochs helps both the RM score and human preference ratings" (paper, 3.5), so the final model is picked by reward-model score.
- **Step 2: the reward model (RM).**
  - *What it does:* start from the SFT model with the last "unembedding" layer removed and train it to take a prompt and a response and output one number, the reward (paper, 3.5). Labelers rank between $K=4$ and $K=9$ responses per prompt, giving $\binom{K}{2}$ comparisons. Only a 6B reward model is used, even for the 175B policy.
  - *Why it matters:* it turns human judgment into a number a program can optimise. A 175B reward model "could be unstable" and is "less suitable" as the starting point of the RL value function (paper, 3.5; appendix C.2).
- **Step 3: reinforcement learning with PPO.**
  - *What it does:* the environment is a "bandit": a random prompt comes up, the model writes a response, the reward model scores it, and the episode ends. A per-token penalty on how far the model moves from the SFT model is added "to mitigate over-optimization of the reward model" (paper, 3.5). The value function starts from the reward model.
  - *Why it matters:* it is the step that pushes the model beyond imitating demonstrations toward whatever the reward model prefers.
- **The pretraining mix ("PPO-ptx").**
  - *What it does:* mixes gradients that raise the likelihood of the original pre-training data into the PPO updates (equation 4 below).
  - *Why it matters:* RLHF caused "performance regressions" on public data sets such as SQuAD, DROP, HellaSwag and French-to-English translation, "an example of an alignment tax"; the mix greatly reduces them (paper, 1 and 4.2). Unless stated, "InstructGPT" in the paper means the PPO-ptx model.

**Table: the sizes of the three datasets** (paper's Table 6; sizes are numbers of prompts; written by me, an AI, from the paper's table, checked against the page picture; the totals are mine).

| Dataset | Split | Labeler-written | Customer (API) | Total (mine) |
|---|---|---|---|---|
| SFT | train | 11,295 | 1,430 | 12,725 |
| SFT | valid | 1,550 | 103 | 1,653 |
| RM | train | 6,623 | 26,584 | 33,207 |
| RM | valid | 3,488 | 14,399 | 17,887 |
| PPO | train | not used | 31,144 | 31,144 |
| PPO | valid | not used | 16,185 | 16,185 |

What each column is, and why it matters:
- **Dataset and split:** which step the prompts train (SFT, RM, PPO) and whether they train or validate.
- **Labeler-written / Customer:** who wrote the prompt: contractors, or customers using the API Playground. It matters because the paper's SFT set is 89% labeler-written (11,295 of 12,725, mine), since at the start labelers wrote template instructions with few-shot examples, from which the paper "synthetically constructed multiple SFT datapoints", while the RM and PPO sets are mostly real customer prompts.
- **Total:** my sums; they agree with the text's "about 13k", "33k" and "31k" training prompts (paper, 3.2).
- **The PPO data has no human labels at all:** only prompts. The reward comes from the reward model.

**Summary: what the table shows.**
- **Small by modern standards:** the whole SFT training set is about 13 thousand prompts; the RM set is 33 thousand prompts, but each has 4 to 9 ranked answers, so "the number of ranked pairs … is an order of magnitude larger than the number of prompts" (paper, A.3).
- **Caution:** "customer" prompts are Playground users on an earlier InstructGPT model, not production traffic (paper, 3.2), and the dataset is over 96% English (paper, 3.3).

**Table: what the prompts are for** (paper's Table 1: share of use-case categories in the RM prompts, as labelled by the contractors; written by me, an AI).

| Use case | Share |
|---|---|
| Generation | 45.6% |
| Open QA | 12.4% |
| Brainstorming | 11.2% |
| Chat | 8.4% |
| Rewrite | 6.6% |
| Summarization | 4.2% |
| Classification | 3.5% |
| Other | 3.5% |
| Closed QA | 2.6% |
| Extract | 1.9% |

What each column is, and why it matters:
- **Use case:** the type of task the user's prompt asks for, assigned by contractors. Closed QA means the answer should come only from the text in the prompt.
- **Share:** the percentage of prompts. They sum to 99.9% (mine), so the table is rounded.

**Summary: what the table shows.**
- **Biggest:** plain generation, 45.6%, and generation plus brainstorming is 56.8% (the paper says "about 57%").
- **Smallest:** extraction, 1.9%, and closed QA, 2.6%. Classification and QA together are 18.5% (mine; the paper says "about 18%").
- **Why it matters for the paper:** public NLP data sets are mostly classification and QA, so they cover only a small part of what users do, one of the paper's two reasons that FLAN and T0 do worse here (paper, 4.1).

## The core mechanism, step by step

**From human rankings to a better model (my numbered paraphrase of paper, 3.1 and 3.5).**
1. **Collect demonstrations:** labelers write the desired answer to each prompt; fine-tune GPT-3 on them (SFT).
2. **Sample answers:** for each prompt in the RM set, the SFT model (and later the PPO models) produce $K$ answers, between 4 and 9.
3. **Rank:** a labeler ranks the $K$ answers from best to worst. A ranking of $K$ answers gives $\binom{K}{2}$ pairs (winner, loser). For $K=4$ and the ranking A, B, C, D the pairs are AB, AC, AD, BC, BD, CD.
4. **Train the reward model:** make it give the winner a higher score than the loser for every pair (equation 2). All $\binom{K}{2}$ pairs of one prompt go into one batch element, so only $K$ forward passes are needed rather than $\binom{K}{2}$, and the model no longer overfits.
5. **Normalise the reward:** the loss only looks at differences of rewards, so "the RM loss is invariant to shifts in reward"; a bias is added so labeler demonstrations average a reward of 0 before RL starts (paper, 3.5).
6. **Run PPO:** take a prompt, let the model write a response, get its reward from the reward model, subtract the KL penalty that keeps it near the SFT model, and update with PPO. In PPO-ptx also add the pre-training gradient (equation 4).
7. **Repeat if wanted:** steps 2 to 6 can be iterated, but "most of our comparison data comes from our supervised policies, with some coming from our PPO policies" (paper, 3.1).

**Training details (paper, appendix C).** SFT: batch size 32 and learning rate 9.65e-6 for 1.3B and 6B, batch size 8 and 5.03e-6 for 175B. Reward model: 6B, one epoch (more epochs "quickly overfit"), learning rate 9e-6, batch size 64 prompts, so up to $64\times36=2{,}304$ comparisons per batch (mine). PPO: 256k episodes over about 31k unique prompts (about 8 passes over each prompt, mine), batch 512 split into 8 minibatches of 64, KL coefficient $\beta=0.02$, pre-training coefficient $\gamma=27.8$ with 8 times as many pre-training examples as episodes, PPO clip ratio 0.2, sampling temperature 1.

## The equations

**1. The preference model: reward differences are log odds** (paper, 3.5, in the text)

$$
P(y_w\succ y_l\mid x)=\sigma\big(r_\theta(x,y_w)-r_\theta(x,y_l)\big)
$$

**Where:**
- $x$: the prompt; $y_w$, $y_l$: the preferred and the rejected response.
- $r_\theta(x,y)$: the reward model's score for response $y$ to prompt $x$.
- $\sigma$: the logistic (sigmoid) function, $\sigma(t)=1/(1+e^{-t})$, which turns any number into a probability between 0 and 1.
- $P(y_w\succ y_l\mid x)$: the model's chance that people prefer $y_w$.

**Matters:** the paper says "the difference in rewards represents the log odds that one response will be preferred to the other by a human labeler" (paper, 3.5). That is what lets simple numbers stand for human taste. Because only the difference counts, adding the same constant to every reward changes nothing, which is why the paper normalises the reward with a bias. See [[Bradley-Terry Model]].

**Example (tiny numbers):** reward 1.5 for $y_w$ and 0.5 for $y_l$.
- Difference $=1.0$, so $\sigma(1)=0.731$ and the log odds are $\ln(0.731/0.269)=1.0$.
- A difference of 0 gives 0.5 (a coin flip); a difference of 3 gives 0.953.
- Meaning: a reward gap of 1 means the preferred answer wins about 73% of the time.

**2. The reward-model loss** (paper, Eq. 1)

$$
\mathrm{loss}(\theta)=-\frac{1}{\binom{K}{2}}\,\mathbb{E}_{(x,y_w,y_l)\sim D}\Big[\log\sigma\big(r_\theta(x,y_w)-r_\theta(x,y_l)\big)\Big]
$$

**Where:**
- $\theta$: the weights of the reward model.
- $K$: the number of responses ranked per prompt (4 to 9); $\binom{K}{2}=K(K-1)/2$ is the number of pairs.
- $D$: the data set of human comparisons; the average is over all pairs.
- $\log\sigma(\cdot)$: the logarithm of the probability the model gives to the human's actual choice.
- The minus sign makes it a loss to minimise.

**Matters:** it is a cross-entropy loss with the human choice as the label. It is small when the model gives the winner a clearly higher reward and large when it ranks the pair the wrong way round. Dividing by $\binom{K}{2}$ makes prompts with more answers count the same as those with fewer. Without the sigmoid and logarithm, the model could just inflate all scores.

**Example (tiny numbers):** one pair with a reward gap of $+1$ for the winner.
- Loss $=-\ln\sigma(1)=-\ln0.731=0.313$.
- Ranked the wrong way round (gap $-1$): $-\ln\sigma(-1)=1.313$. A tie (gap 0) gives $0.693$. A gap of $+3$ gives $0.049$ and a gap of $-3$ gives $3.049$.
- For four answers with rewards $A=2$, $B=1$, $C=0.5$, $D=-1$ the six pair losses are 0.313, 0.201, 0.049, 0.474, 0.127 and 0.201; their mean is $0.228$.
- Meaning: the loss punishes a wrong ranking about four times as hard as it rewards a right one at the same gap.

**3. How many comparisons from one ranking** (paper, 3.5, in the text)

$$
\binom{K}{2}=\frac{K(K-1)}{2}
$$

**Where:**
- $K$: how many responses a labeler ranks for one prompt.
- $\binom{K}{2}$: the number of ordered pairs (winner, loser) implied by the ranking.

**Matters:** ranking gives many comparisons for one labeling effort: $K=4$ gives 6, $K=9$ gives 36. The paper trains on all pairs of a prompt in one batch element; if each pair were a separate data point, each answer would be used in $K-1$ gradient updates and "the model tends to overfit after a single epoch" (paper, 3.5, footnote 5). Without batching this way, it needs $\binom{K}{2}$ forward passes instead of $K$.

**Example (tiny numbers):** $K=4$ and $K=9$.
- $\binom{4}{2}=6$ pairs from 4 forward passes; $\binom{9}{2}=36$ pairs from 9 forward passes.
- A batch of 64 prompts with $K=9$ has $64\times36=2{,}304$ comparisons, the maximum the paper states.
- Meaning: the effort of ranking 9 answers gives 36 training comparisons.

**4. The reinforcement-learning objective with the pretraining mix** (paper, Eq. 2)

$$
\mathrm{objective}(\phi)=\mathbb{E}_{(x,y)\sim D_{\pi_\phi^{\mathrm{RL}}}}\Big[r_\theta(x,y)-\beta\log\frac{\pi_\phi^{\mathrm{RL}}(y\mid x)}{\pi^{\mathrm{SFT}}(y\mid x)}\Big]+\gamma\,\mathbb{E}_{x\sim D_{\mathrm{pretrain}}}\Big[\log\pi_\phi^{\mathrm{RL}}(x)\Big]
$$

**Where:**
- $\pi_\phi^{\mathrm{RL}}$: the policy being trained (the language model), with weights $\phi$.
- $\pi^{\mathrm{SFT}}$: the supervised model it started from, kept as a reference.
- $r_\theta(x,y)$: the reward model's score for the response $y$ to prompt $x$.
- $\beta$: the KL reward coefficient (0.02 in the paper's runs).
- $\log\frac{\pi^{\mathrm{RL}}}{\pi^{\mathrm{SFT}}}$: how much more likely the new model finds this response than the SFT model does; the paper applies it per token.
- $D_{\mathrm{pretrain}}$: the original pre-training text; $\gamma$: the pre-training coefficient (27.8). For plain "PPO" models $\gamma=0$.

**Matters:** the first part wants high reward; the $\beta$ term subtracts a penalty for drifting away from the SFT model, which the paper says helps "mitigate over-optimization of the reward model". The last part keeps the model good at ordinary text, which is what fixes the performance regressions. The paper also finds that the pretraining mix is a better fix than just raising $\beta$, which cut validation reward a lot and "never fully recovers on DROP and SQuAD" (paper, 4.2). See [[KL Divergence]].

**Example (tiny numbers):** $\beta=0.02$ and a response with reward 2.0 (my own numbers; the paper's real scale is not given).
- A response whose total log-ratio is 30: penalty $0.02\times30=0.6$, net $1.4$.
- Log-ratio 100: penalty $2.0$, net $0.0$. Log-ratio 150: penalty $3.0$, net $-1.0$.
- Pre-training term: if the log-likelihood of a pre-training sample improves from $-2.00$ to $-1.90$, that adds $27.8\times0.10=2.78$ to the objective.
- Meaning: the farther the model drifts, the more reward it must earn to be worth it; $\gamma$ is large because pre-training log-likelihoods change by small amounts.

## What they found
Labelers rate the three sizes of InstructGPT above the GPT-3 baselines, and the model also improves truthfulness and some safety measures, with a few honest failures (paper, 4). The paper's main figures show win rates and rates in plots with 95% confidence intervals, and the exact values are mostly not printed, so the numbers below are those in the text.

### Which answers do labelers prefer? (paper's Figures 1 and 3, and text)

**Table: head-to-head preference results from the text** (not a paper table; I collected the numbers the text gives; the model on the left is compared with the one on the right; the rate is how often the left one's output was preferred by labelers).

| Left model | Right model | Preferred | Section |
|---|---|---|---|
| InstructGPT 175B | GPT-3 175B | 85 ± 3% | paper, 1 and 4.1 |
| InstructGPT 175B | GPT-3 175B with a few-shot prefix | 71 ± 4% | paper, 1 and 4.1 |
| InstructGPT 175B | GPT-3 175B fine-tuned on FLAN | 78 ± 4% | paper, 4.1 |
| InstructGPT 175B | GPT-3 175B fine-tuned on T0 | 79 ± 4% | paper, 4.1 |
| InstructGPT (win rate against the SFT baseline) | the 175B SFT model | 73.4 ± 2% | paper, 1 |
| T0 (win rate against the SFT baseline) | the 175B SFT model | 26.8 ± 2% | paper, 1 |
| FLAN (win rate against the SFT baseline) | the 175B SFT model | 29.8 ± 2% | paper, 1 |

Written by me, an AI, from the paper's text. The paper's Figure 1 also shows the 1.3B and 6B sizes and the PPO, SFT and prompted GPT lines; I read these as a plot only, not as numbers.

What each column is, and why it matters:
- **Left model / Right model:** what is compared. "Few-shot prefix" means GPT-3 was given a hand-picked prefix to make it follow instructions (it was chosen by an hour-long competition between two authors, judged by reward-model score, paper, footnote 6). FLAN and T0 are public instruction data sets used to fine-tune the same 175B GPT-3.
- **Preferred:** the share of comparisons won by the left model, by the labelers; the number after "±" is a 95% confidence interval. 50% means no preference.
- **Section:** the place in the paper.

**Summary: what the table shows.**
- **Biggest win:** against plain GPT-3, 85% (a gap of 35 points from a tie); even against GPT-3 with a prefix made for the task, 71%.
- **FLAN and T0:** InstructGPT wins 78% and 79%; against the SFT baseline FLAN and T0 score only 29.8% and 26.8%, below 50%, so they are slightly worse than the SFT baseline (paper, 1).
- **Most famous result:** "outputs from the 1.3B parameter InstructGPT model are preferred to those from the 175B GPT-3, despite having 100x fewer parameters" (abstract); my arithmetic gives 134.6 times fewer (175 divided by 1.3).
- **Caution:** the prompts come from an earlier InstructGPT model's users, which "likely disadvantage the GPT-3 baselines" (paper, 3.6); the paper checks prompts sent to GPT-3 models and the result "do[es] not change significantly", though PPO-ptx does "slightly worse at larger model sizes" (paper, 4.1). The judges are the same small labeler group that trained the model.

### Other measures: truthfulness, toxicity, bias and regressions (paper, 4.1 and 4.2)
The numbers in the text:
- **Hallucination:** on "closed-domain" tasks (summarisation, closed QA) InstructGPT makes up information 21% of the time against GPT-3's 41% (paper, 1). That is a 20-point fall, about half (my arithmetic: 49% fewer).
- **Truthfulness:** on TruthfulQA, InstructGPT gives truthful and informative answers "about twice as often as GPT-3" (paper, 1). The exception: the 1.3B PPO-ptx model "performs slightly worse than a GPT-3 model of the same size" (paper, 4.2).
- **Toxicity:** about 25% fewer toxic outputs than GPT-3 when told to be respectful (paper, 1); but this advantage "disappears when the respectful prompt is removed", and when explicitly asked for toxic output InstructGPT is "much more toxic" (paper, 4.2).
- **Bias:** no significant improvement on Winogender and CrowS-Pairs (paper, 1 and 4.2).
- **Regressions and the fix:** PPO alone lowers scores on SQuAD, DROP, HellaSwag and translation; PPO-ptx mitigates this on all, "even surpasses GPT-3 on HellaSwag", but still lags on DROP, SQuADv2 and translation (paper, 4.2).
- **Reward model generalisation:** reward models predict the preferences of held-out labeler groups with 69.6 ± 0.9% accuracy against 72.4 ± 0.4% on their own training groups, a 2.8-point drop (mine) (paper, 4.1).

**Summary: what these results show.**
- **Best:** hallucination halved, truthfulness about doubled, preference won against every baseline.
- **Worst:** no bias improvement; more toxic when asked to be; some benchmarks still below GPT-3 (DROP, SQuADv2, translation).
- **Caution:** the toxicity and bias benchmarks are narrow proxies; the paper sampled RealToxicityPrompts prompts "uniformly according to prompt toxicity", which makes its absolute toxicity numbers "inflated" (paper, 4.2).

### The cost of alignment (paper, 5.1)
- **Table: compute used** (from the text; written by me, an AI).

| Model | Compute (petaflops/s-days) | As a share of GPT-3's (mine) |
|---|---|---|
| GPT-3 175B (pre-training) | 3,640 | 100% |
| InstructGPT 175B SFT | 4.9 | 0.13% |
| InstructGPT 175B PPO-ptx | 60 | 1.65% |

What each column is, and why it matters:
- **Compute:** training cost in petaflops per second for a day (one petaflop per second sustained for one day), a standard unit for large training runs (general knowledge).
- **Share (mine):** the cost divided by GPT-3's. It matters because it supports the paper's claim that alignment is cheap next to pre-training.

**Summary: what the table shows.** Fine-tuning with human feedback costs about 1.7% of GPT-3's pre-training compute (the SFT and PPO-ptx runs together are about 1.8%, mine), yet beats a 100 times larger model on labeler preference. **Caution:** the figure covers training compute only; the paper adds that "the cost of collecting our data" is "a fraction of what was spent to train GPT-3" without giving the number.

## Questions I had, answered

**Why not just train on good answers (SFT) and stop?** SFT only imitates the writers. The reward model can also say which of two imperfect answers is better, and PPO can then push the model to give answers that are preferred even beyond what the demonstrations covered. The paper's Figure 1 shows each step improving the win rate: prompted GPT-3, then SFT, then PPO (paper, 4.1).

**Why train the reward model on pairs and not on scores?** People find ranking easier and more consistent than giving absolute scores (general knowledge; the paper only says it uses rankings to speed up comparison collection, 3.5). Loss on pairs only needs the difference of rewards (equations 1 and 2).

**Why use a 6B reward model for a 175B policy?** A 175B reward model "could be unstable" and "greatly increase[s] the compute requirements of PPO"; 6B models were stable across a wide range of learning rates and "led to equally strong PPO models" (paper, appendix C.2).

**What does the KL penalty do, and why is it needed?** It subtracts from the reward in proportion to how far the new model's probabilities have moved from the SFT model's, "to mitigate over-optimization of the reward model" (paper, 3.5). The reward model is only trustworthy near what it was trained on, so a model that wanders far may find outputs that score high but are poor. In the tiny example, with $\beta=0.02$ a drift of log-ratio 100 cancels a reward of 2.

**What is the "alignment tax" and how is it removed?** RLHF alone made the model worse on some standard benchmarks; that cost of alignment is the tax. Mixing pre-training updates into PPO (PPO-ptx) reverses most of it, and does better than just raising the KL coefficient (paper, 4.2).

**Why does the model hedge so much?** The paper suspects that "we instruct labelers to reward epistemic humility", so labelers rewarded hedging and the reward model learned it (paper, 4.3). It also fails with false premises ("Why is it important to eat socks after meditating?"), which it suspects is because few training prompts had false premises (paper, 4.3).

**Whose preferences does the model follow?** Those of about 40 mostly English-speaking labelers, as guided by the researchers' instructions, and implicitly those of API customers; the paper says one cannot "train a system that is aligned to everyone's preferences at once" (paper, 5.2).

## Why it mattered
It showed that a small model trained on human feedback beats a model 100 times larger on what users want, at a small fraction of the pre-training cost, and made the recipe of supervised fine-tuning, a reward model and PPO the standard way to build assistants (the second part from memory; the paper itself says RLHF is "a promising direction").

## Limits and open questions
The authors list these (paper, 4.3, 5.2, 5.3 and 5.4):
- **Still wrong sometimes:** "neither fully aligned nor fully safe"; it still makes things up, can be toxic or biased, and follows harmful instructions ("in most cases, they follow the user's instruction, even if that could lead to harm").
- **Narrow labeler group:** about 40 contractors, mostly English speakers; most comparisons were labeled by only one person, so disagreement is hidden.
- **Hedging and false premises** (above) and a drop in quality when instructions have many explicit constraints.
- **Alignment tax only partly removed:** PPO-ptx "does not completely mitigate performance regressions".
- **Open questions:** better ways to get feedback than comparisons (edits or critiques), making models refuse harmful requests, adversarial data collection, filtering the pre-training mix, and a fair process for deciding whose preferences to follow.
- **Other algorithms:** the paper says RLHF is only one of many algorithms that could be tried on this data (expert iteration, behaviour cloning on a subset of comparisons and others, paper, 5.4). Later work, [[DPO 2023]], replaced the reward model and PPO with a simpler loss (from memory of that paper's own claim, which is in this vault).

## Concepts used
[[RLHF]], [[Reward Model]], [[PPO]], [[Instruction Tuning]], [[Fine-tuning]], [[Bradley-Terry Model]], [[KL Divergence]], [[Alignment]]

## Reading trail
- **Came from:** [[GPT-3 2020]]
- **Led to:** [[DPO 2023]]

## How much I read
The paper's body as text: pages 1 to 20 (introduction, related work, methods, results, discussion) and from the appendix the dataset sizes (A.3) and the training details of SFT, the reward model and RLHF (appendix C). I looked at the page with Table 6 as a picture to check its layout. Not read: the rest of the appendix (the labeler instructions, examples, the extra evaluation plots and tables, the model-card sections) and the references. The figures were read through their captions only, so numbers I could not find in the text are not given. The numbers marked "mine" are my own arithmetic in Python. The figure is a free redrawing and differs from the paper's, as its caption says. Not opened in Obsidian, so the flashcards are unreviewed.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-llm InstructGPT (Ouyang et al., 2022, arXiv 2203.02155): what problem did it attack?::Making language models bigger does not make them follow the user's intent: they can be untruthful, toxic or unhelpful, because next-word prediction is a different objective from following instructions helpfully and safely (the language modelling objective is misaligned).

#flashcards/research/concept #flashcards/topic/research-llm What did people do before InstructGPT?::Prompted plain models such as GPT-3; instruction tuning on public NLP datasets (FLAN, T0); and RLHF for robots, games and summarisation. InstructGPT applies RLHF to a broad range of language tasks.

#flashcards/research/concept #flashcards/topic/research-llm InstructGPT: what are helpful, honest and harmless?::The paper's definition of aligned: helpful (solves the user's task), honest (does not fabricate or mislead) and harmless (does not cause harm). Honesty is measured as truthfulness (hallucination rate and TruthfulQA).

#flashcards/research/concept #flashcards/topic/research-llm What are the three steps of InstructGPT's method?::1. Collect labeler demonstrations and fine-tune GPT-3 on them (SFT). 2. Collect labeler rankings of several model outputs and train a reward model to predict the preferred one. 3. Fine-tune the SFT model with PPO to maximise the reward model's score. Steps 2 and 3 can be repeated.

#flashcards/research/concept #flashcards/topic/research-llm ![[rlhf-diagram.svg]]
What does this picture show, and how does it differ from InstructGPT's Figure 2?
?
A general RLHF pipeline: a supervised model trained on prompt and response data; human annotators rank its responses; the rankings train a reward model; the reward model guides PPO training of the aligned model. It is a free redrawing by PopoDameron (CC BY-SA 4.0). InstructGPT's own Figure 2 has three numbered steps with boxes A to D ranked; this drawing says rankings become scores by the Elo algorithm, which InstructGPT does not do (it uses pairwise comparisons in the loss).

#flashcards/research/concept #flashcards/topic/research-llm Where did InstructGPT's prompts and labelers come from?::Mostly prompts customers sent to an earlier InstructGPT in the API Playground, plus labeler-written prompts; about 40 contractors from Upwork and ScaleAI chosen by a screening test. Training labelers agreed with each other 72.6% of the time.

#flashcards/research/concept #flashcards/topic/research-llm How big are InstructGPT's three datasets (paper's Table 6)?::SFT: about 12.7 thousand training prompts, 89% labeler-written. RM: about 33.2 thousand training prompts, 80% customer. PPO: about 31.1 thousand customer prompts with no labels. Totals are my arithmetic. The reward model sees many more comparisons than prompts.

#flashcards/research/concept #flashcards/topic/research-llm What are most of the prompts for (paper's Table 1)?::Generation 45.6%, open QA 12.4%, brainstorming 11.2%, chat 8.4%; classification and QA together only about 18.5% (my arithmetic). Public NLP sets cover mostly the small QA and classification share, a reason FLAN and T0 do worse.

#flashcards/research/concept #flashcards/topic/research-llm How is InstructGPT's SFT done, and why train 16 epochs?::GPT-3 is fine-tuned on labeler demonstrations for 16 epochs with cosine learning-rate decay and residual dropout 0.2. The validation loss overfits after 1 epoch, but more epochs help the reward-model score and human preference, which choose the final model.

#flashcards/research/concept #flashcards/topic/research-llm How is the reward model built and why only 6B?::Take the SFT model, remove the final unembedding layer and output one scalar for a prompt and response. A 6B model is used for all policies, since a 175B reward model could be unstable, is a poor starting point for the value function and costs far more compute; 6B gave equally strong PPO models.

#flashcards/research/concept #flashcards/topic/research-llm What is the environment in InstructGPT's RL step?::A bandit: a random prompt appears, the model writes a response, the reward model gives a reward and the episode ends. A per-token KL penalty from the SFT model is added, and the value function starts from the reward model.

#flashcards/research/concept #flashcards/topic/research-llm What is PPO-ptx and why was it made?::PPO plus gradients that increase the likelihood of the pre-training data. It fixes the performance regressions (alignment tax) that PPO alone causes on SQuAD, DROP, HellaSwag and translation, without lowering labeler preference.

#flashcards/research/concept #flashcards/topic/research-llm What are the steps from human ranking to a better model?::Sample K answers per prompt; a labeler ranks them; all pairs from a ranking go into one batch element; the reward model is trained on the pairs; the reward is shifted so demonstrations average 0; PPO optimises reward minus the KL penalty (plus the pre-training term); repeat if wanted.

#flashcards/research/concept #flashcards/topic/research-llm Write the preference model of the reward model and explain each symbol.
?
$$P(y_w\succ y_l\mid x)=\sigma\big(r_\theta(x,y_w)-r_\theta(x,y_l)\big)$$
- $x$: prompt; $y_w,y_l$: preferred and rejected responses; $r_\theta$: the reward model's score; $\sigma$: the sigmoid.
- The difference of rewards is the log odds that people prefer $y_w$; adding the same constant to all rewards changes nothing.

#flashcards/research/concept #flashcards/topic/research-llm Preference example: rewards 1.5 and 0.5. What is the chance the first is preferred, and what for differences 0 and 3?
?
Difference $1$: $\sigma(1)=0.731$ (log odds exactly 1). Difference $0$: $0.5$. Difference $3$: $0.953$ (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Why are reward differences, and not the rewards themselves, what the model learns?
?
The loss only uses $r(y_w)-r(y_l)$, so adding the same number to every reward changes nothing: the loss is "invariant to shifts in reward". That is why the paper adds a bias so labeler demonstrations average 0 before RL. The difference is a log odds, so a gap of 1 means about a 73% chance of preference.

#flashcards/research/concept #flashcards/topic/research-llm Write the reward-model loss (Eq. 1) and explain each symbol.
?
$$\mathrm{loss}(\theta)=-\frac{1}{\binom{K}{2}}\,\mathbb{E}_{(x,y_w,y_l)\sim D}\Big[\log\sigma\big(r_\theta(x,y_w)-r_\theta(x,y_l)\big)\Big]$$
- $K$: responses ranked per prompt (4 to 9); $\binom{K}{2}$: pairs; $D$: human comparisons; $\theta$: reward-model weights.
- A cross-entropy on the human choice: small when the winner gets the higher reward, large when ranked wrongly.

#flashcards/research/concept #flashcards/topic/research-llm Reward-loss example: reward gap $+1$, $-1$, $0$, $+3$, $-3$. What are the losses?
?
$0.313$, $1.313$, $0.693$, $0.049$ and $3.049$ (my arithmetic). Four answers with rewards 2, 1, 0.5 and $-1$ have six pair losses averaging $0.228$.

#flashcards/research/concept #flashcards/topic/research-llm Why does the reward-model loss use a sigmoid and a logarithm?
?
The sigmoid turns a reward gap into the probability the human choice is right; the logarithm turns it into the cross-entropy loss that punishes confident mistakes: gap $-3$ costs $3.049$ against $0.049$ for $+3$. Without them the model could minimise a plain gap by inflating all scores and it would not be a likelihood of human choices.

#flashcards/research/concept #flashcards/topic/research-llm Why is the loss divided by $\binom{K}{2}$?
?
A prompt ranked with $K=9$ gives 36 pairs but one with $K=4$ only 6; dividing by the number of pairs lets each prompt count the same whatever $K$ is. Otherwise prompts with more ranked answers would dominate the batch.

#flashcards/research/concept #flashcards/topic/research-llm Write the number of comparisons from ranking $K$ answers and explain each symbol.
?
$$inom{K}{2}=rac{K(K-1)}{2}$$
- $K$: how many responses a labeler ranks for one prompt (4 to 9); $inom{K}{2}$: the number of (winner, loser) pairs the ranking implies.
- Ranking gives many comparisons for one effort: 6 for $K=4$, 36 for $K=9$. All pairs of one prompt go in one batch element, so only $K$ forward passes are needed.

#flashcards/research/concept #flashcards/topic/research-llm Why train the reward model on all pairs of a prompt together?
?
$\binom{K}{2}=K(K-1)/2$: 6 for $K=4$ and 36 for $K=9$. All pairs from one prompt form a single batch element: only $K$ forward passes are needed (not $\binom{K}{2}$), and treating pairs as separate data points would reuse each answer in $K-1$ updates and overfit after one epoch.

#flashcards/research/concept #flashcards/topic/research-llm Comparison-count example: a batch of 64 prompts with $K=9$. How many comparisons?
?
$64\times36=2{,}304$, the maximum the paper states for a batch (appendix C.2, my arithmetic). For $K=4$ a prompt gives 6 pairs from 4 forward passes.

#flashcards/research/concept #flashcards/topic/research-llm Write InstructGPT's RL objective with the pretraining mix (Eq. 2) and explain each symbol.
?
$$\mathrm{objective}(\phi)=\mathbb{E}_{(x,y)\sim D_{\pi_\phi^{\mathrm{RL}}}}\Big[r_\theta(x,y)-\beta\log\frac{\pi_\phi^{\mathrm{RL}}(y\mid x)}{\pi^{\mathrm{SFT}}(y\mid x)}\Big]+\gamma\,\mathbb{E}_{x\sim D_{\mathrm{pretrain}}}\Big[\log\pi_\phi^{\mathrm{RL}}(x)\Big]$$
- $\pi^{\mathrm{RL}}_\phi$: the policy being trained; $\pi^{\mathrm{SFT}}$: the supervised reference; $r_\theta$: the reward; $\beta=0.02$: the KL coefficient; $\gamma=27.8$: the pre-training coefficient ($0$ for plain PPO); $D_{\mathrm{pretrain}}$: the pre-training text.
- It wants high reward, subtracts a penalty for drifting from the SFT model, and keeps the model good at ordinary text.

#flashcards/research/concept #flashcards/topic/research-llm RL-objective example: reward 2.0, $\beta=0.02$, log-ratios 30, 100 and 150. What are the net values, and the gain from a $+0.10$ pre-training likelihood with $\gamma=27.8$?
?
$2-0.6=1.4$, $2-2.0=0.0$ and $2-3.0=-1.0$; the pre-training gain is $27.8\times0.10=2.78$ (my arithmetic with my own numbers, not the paper's scale). The farther the model drifts, the more reward it must earn.

#flashcards/research/concept #flashcards/topic/research-llm Why is there a KL penalty $\beta\log(\pi^{\mathrm{RL}}/\pi^{\mathrm{SFT}})$ in the reward?
?
The reward model is only reliable near the data it saw; a policy that drifts far can find outputs that score high but are poor (over-optimisation). The penalty makes drift cost reward: with $\beta=0.02$ a log-ratio of 100 cancels a reward of 2. Without it the policy would chase the reward model wherever it leads.

#flashcards/research/concept #flashcards/topic/research-llm Why is $\gamma$ so large (27.8) and why is the pre-training term there?
?
The term keeps ordinary language skills: without it PPO caused regressions on SQuAD, DROP, HellaSwag and translation (the alignment tax). Pre-training log-likelihoods change in small steps, so a large $\gamma$ is needed for them to matter against the reward: an improvement of $0.10$ adds $2.78$. Raising $\beta$ instead cut reward a lot and never fixed DROP and SQuAD.

#flashcards/research/concept #flashcards/topic/research-llm How did labelers rate InstructGPT against baselines (paper's text and Figure 1)?::175B InstructGPT beat 175B GPT-3 in 85 ± 3% of comparisons and few-shot GPT-3 in 71 ± 4%; it beat FLAN-tuned GPT-3 78 ± 4% and T0-tuned 79 ± 4%. The 1.3B InstructGPT was preferred to the 175B GPT-3, though it has about 135 times fewer parameters (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Why is InstructGPT's win over GPT-3 not a perfectly fair test?::The prompts come from users of an earlier InstructGPT model, which likely disadvantages GPT-3; the paper checks prompts sent to GPT-3 models and finds similar results (PPO-ptx slightly worse at larger sizes). The judges are the same small, mostly English-speaking labeler group that trained the model.

#flashcards/research/concept #flashcards/topic/research-llm How did InstructGPT do on truthfulness, toxicity and bias?::Hallucination on closed-domain tasks 21% against GPT-3's 41%; truthful and informative answers on TruthfulQA about twice as often; about 25% fewer toxic outputs when told to be respectful, but not without that prompt and much more toxic when asked to be; no significant bias improvement.

#flashcards/research/concept #flashcards/topic/research-llm How much did alignment cost in compute, compared with GPT-3's pre-training?::GPT-3 175B took 3,640 petaflops/s-days; the 175B SFT model 4.9 and the PPO-ptx model 60, so about 1.65% for PPO-ptx and about 1.8% for both (my arithmetic). Alignment was far cheaper than pre-training, yet more helpful than a 100 times bigger model.

#flashcards/research/concept #flashcards/topic/research-llm What is the "alignment tax"?::The drop in performance on some standard benchmarks (SQuAD, DROP, HellaSwag, translation) that RLHF alone caused. PPO-ptx removes most of it, even beating GPT-3 on HellaSwag, but it still lags on DROP, SQuADv2 and translation.

#flashcards/research/concept #flashcards/topic/research-llm Why did InstructGPT beat FLAN and T0 fine-tuning on its prompts?::Public NLP data sets are mostly classification and QA, only about 18% of what customers do, while open-ended generation and brainstorming are about 57%; and such data sets lack the diversity of real user inputs. The paper says the best model would combine both kinds of data.

#flashcards/research/concept #flashcards/topic/research-llm Why does InstructGPT hedge and go along with false premises?::The paper suspects hedging comes from labelers being told to reward epistemic humility, which the reward model learned; and that false-premise failures come from few such prompts in the training data. Adversarial data collection could reduce both.

#flashcards/research/concept #flashcards/topic/research-llm Whose preferences does InstructGPT align to?::About 40 mostly English-speaking labelers as guided by the researchers' instructions, and implicitly API customers; the paper stresses that one cannot train a system aligned to everyone's preferences at once.

#flashcards/research/concept #flashcards/topic/research-llm InstructGPT: what limits does the paper name?::Not fully aligned or safe (makes things up, can be toxic or biased, follows harmful instructions); a small, unrepresentative labeler group with most comparisons labeled once; hedging and false premises; the alignment tax is only partly removed.

#flashcards/research/concept #flashcards/topic/research-llm InstructGPT: why did it matter and what is its place in the reading trail?::It showed human-feedback training beating a 100 times bigger model at about 2% of pre-training compute, and made supervised fine-tuning, a reward model and PPO the standard recipe for assistants (from memory). It came from GPT-3 and led to DPO, which replaced the reward model and PPO with one loss.

#flashcards/research/concept #flashcards/topic/research-llm How much of the InstructGPT paper did the note read?::The body (pages 1 to 20) and the appendix dataset sizes and training details (A.3, C); the figures only by captions; not the labeler instructions, the extra evaluation plots, the rest of the appendix or the references. Tables are written by me and the figure used is a free redrawing.
