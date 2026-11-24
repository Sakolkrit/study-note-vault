---
tags: [paper, LLM, reasoning, reinforcement-learning, GRPO]
link: https://arxiv.org/abs/2501.12948
authors: DeepSeek-AI (the author is given as "DeepSeek-AI" on the first page; a long contributor list is on pages 11 and 12)
year: 2025
status: body-read
field: text / reasoning
---

# DeepSeek-R1 2025

> **In one line for a 12-year-old:** A model that learns to think step by step because it is rewarded when its final answer is right, not because people showed it how. Nobody told it to double-check its work, yet it started to.

Paper: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", DeepSeek-AI, arXiv 2501.12948. **The PDF I read is version 2, stamped 4 January 2026 on its first page**; it is much longer (86 pages) than a first version would be, and I did not read version 1, so some details may differ from the January 2025 text. Concept notes: [[Reasoning Model]], [[GRPO]]. Reference note for this layout: [[Attention Is All You Need 2017]]. The arXiv licence is the standard non-exclusive one, so no figure is cropped from the PDF; the one figure below is the Nature-published copy of the paper's pipeline figure, which is CC BY 4.0 on Wikimedia Commons.

## The problem
Large models can reason with help from [[Chain of Thought]] prompts or from human-written reasoning examples, but that depends on "extensive human-annotated demonstrations", it does not scale, it introduces human biases, and it caps the model at human ways of thinking (abstract and paper, 1). Can reasoning be *incentivised* with reinforcement learning ([[Reinforcement Learning]]) alone, with no human-labelled reasoning steps?

## What people did before
- **Supervised fine-tuning on human reasoning traces** (the usual SFT then RL recipe, [[RLHF]] and [[InstructGPT 2022]]): the paper says human answers "often omit critical reasoning components such as explicit reflection and verification steps" (appendix A.2).
- **Chain-of-thought prompting:** few-shot examples or "let's think step by step" (paper, 1 and H.1).
- **Reward models for reasoning** (outcome or process based): the authors say neural reward models "are susceptible to reward hacking during large-scale reinforcement learning" (paper, 2.2). Their failed attempts with a process reward model and with Monte Carlo tree search are in appendix G.2.
- **[[PPO]] with a value model** as the RL algorithm (as in InstructGPT).
- **Scaling inference compute** by majority voting or tree search (appendix H.2).

## The idea
1. **Start from a strong base model, DeepSeek-V3-Base, and skip supervised fine-tuning.** Train it with GRPO ([[GRPO]]) and a reward that only checks whether the final answer is correct and the format is right. The result is **DeepSeek-R1-Zero** (paper, 2).
2. **R1-Zero's reasoning grows by itself:** longer thinking, self-checking, trying other approaches. But it is hard to read and mixes English and Chinese (paper, 1).
3. **DeepSeek-R1 repairs that with four stages:** a small cold-start SFT, a first RL stage with an added language-consistency reward, rejection sampling plus a larger SFT with reasoning and non-reasoning data, and a second RL stage for helpfulness and harmlessness (paper, 3).
4. **Distil** R1's reasoning into smaller models by fine-tuning them on 800,000 R1-written samples (paper, appendix B.3.3 and F).

## The architecture and training, part by part
R1 has no new network. The base model is **DeepSeek-V3-Base**, a mixture-of-experts model with 671 billion parameters of which 37 billion are active per token, pre-trained on 14.8 trillion tokens, with multi-head latent attention and multi-token prediction (appendix A.1). The new thing is the training pipeline.

![[deepseek-r1-pipeline.png]]
*The multi-stage pipeline of DeepSeek-R1: left, R1-Zero is trained by RL from DeepSeek-V3 Base with reasoning prompts and accuracy-and-format rewards, then sampled and filtered to make training data; middle, V3 Base is fine-tuned on cold-start long chain-of-thought data (Dev1), then RL with rule-based and language-consistency rewards (Dev2); right, V3 Base is fine-tuned on sampled reasoning data and non-reasoning data from V3 (Dev3), then RL with diverse prompts and rule-based plus preference rewards gives DeepSeek-R1. Authors: DeepSeek-AI (Daya Guo and others), Figure 2 of the Nature article (https://www.nature.com/articles/s41586-025-09422-z), [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:The_multistage_pipeline_of_DeepSeek-R1.png), CC BY 4.0; I used the Commons file as it is, not cropped from the arXiv PDF. I compared it with Figure 2 on page 6 of the arXiv PDF by eye: the same boxes and arrows; label wording differs slightly (for example "Accuracy and format" and "Dev1" against "Accuracy & Format" and "Dev-1").*

### R1-Zero (paper, section 2)
- **The base model and no SFT.**
  - *What it does:* reinforcement learning starts directly from DeepSeek-V3-Base.
  - *Why it matters:* the authors hypothesise that human-defined reasoning patterns "may limit model exploration". The paper offers no ablation of SFT then RL against RL only for the 671B model; its support is the result and appendix G.1.
- **The prompt template (Table 1).**
  - *What it does:* a conversation between User and Assistant; the assistant thinks first, inside `<think>...</think>`, then answers inside `<answer>...</answer>`.
  - *Why it matters:* it only constrains the structure, "avoiding any content-specific biases".
- **Rule-based rewards.**
  - *What they do:* an accuracy reward (the final answer matches the ground truth, or code passes tests) and a format reward (reasoning inside the think tags) are added with the same weight (section 2.2).
  - *Why they matter:* no neural reward model is used for reasoning because those can be hacked. A rule cannot be argued with.
- **GRPO.**
  - *What it does:* for each question, sample a group of 16 answers, score them, and push up the answers that scored above their group's average.
  - *Why it matters:* it needs no value model (see the equations and the questions).
- **Training settings (section 2.1).** Learning rate 3e-6, KL coefficient 0.001, sampling temperature 1; 16 outputs per question, maximum length 32,768 tokens before step 8,200 and 65,536 tokens afterwards; 32 questions per step (batch of 512 answers); 10,400 steps; the reference model is replaced by the latest policy every 400 steps; each rollout makes 8,192 outputs split into 16 mini-batches of one inner epoch. The paper reports a jump in both accuracy and length at step 8,200, when the limit was doubled.
- **Why 16 outputs, why these learning settings?** The paper gives no ablation for these numbers; they are stated settings.

### The four stages of R1 (paper, section 3 and appendix B)
- **Stage 1, cold start.**
  - *What it does:* thousands of long chain-of-thought examples, made by sampling R1-Zero at temperature 1.0, keeping correct and readable ones, and having DeepSeek-V3 plus human annotators rewrite them in a first-person conversational style, fine-tune DeepSeek-V3-Base (appendix B.3.2).
  - *Why it matters:* it fixes readability. The cost: the resulting Dev1 is worse than R1-Zero on AIME (59.0 against 77.9) because the cold-start set is small (paper, 4 and Table 3).
- **Stage 2, reasoning RL.**
  - *What it does:* the same RL as R1-Zero (clip ratio 10, KL 0.001, 16 samples up to 32,768 tokens) plus a language-consistency reward (section 3.2.1).
  - *Why it matters:* the paper says it slightly degrades performance but makes reasoning more readable (appendix B.6).
- **Stage 3, rejection sampling and SFT (800K samples).**
  - *What it does:* sample answers from the stage-2 checkpoint, keep the correct ones (judged by rules or by DeepSeek-V3), drop mixed languages and very long paragraphs: about 600,000 reasoning samples; plus about 200,000 non-reasoning samples (writing, factual questions, translation, programming repair); fine-tune DeepSeek-V3-Base again (appendix B.3.3 and B.4.2).
  - *Why it matters:* it adds writing and general abilities. Table 5 below gives the exact mix.
- **Stage 4, second RL.**
  - *What it does:* reasoning prompts with rule rewards plus general prompts judged by reward models for helpfulness and harmlessness; 1,700 steps with temperature 0.7, with the general preference data used only in the last 400 steps (section 3.2.2).
  - *Why it matters:* the paper found that more steps with the model-based preference reward "may lead to reward hacking" (appendix B.5, Figure 6: reward rises while Codeforces test performance falls).

### The reward models (paper, 3.1)
- **Helpful reward model:** trained on 66,000 preference pairs made with DeepSeek-V3 judgments (each pair judged four times with positions swapped, kept only if the score difference exceeded 1, with similar lengths for chosen and rejected answers); batch 256, learning rate 6e-6, one epoch. It scores only the final summary, not the reasoning.
- **Safety reward model:** trained point-wise on 106,000 responses labelled safe or unsafe; it scores the whole response, including the reasoning.
- **Why only the summary for helpfulness?** To limit interference with the reasoning (paper, 3.1). No ablation is shown.

### The RL and SFT data (paper's Table 4 and Table 5)

| Data type (RL, Table 4) | Prompts | Question type | Output type |
|---|---|---|---|
| Math | 26K | quantitative reasoning | number, expression or equation |
| Code | 17K | algorithm and bug fixing | code solution |
| STEM | 22K | multiple choice | option |
| Logic | 15K | choice or quantitative reasoning | option or number |
| General | 66K | helpfulness and harmlessness | ranked responses |

| Domain (SFT, Table 5) | Samples | Average rounds | Average tokens |
|---|---|---|---|
| Math | 395,285 | 1.0 | 6,094.2 |
| Code | 211,129 | 1.1 | 7,435.7 |
| STEM | 10,124 | 1.0 | 4,928.8 |
| Logic | 10,395 | 1.0 | 2,739.0 |
| General | 177,812 | 1.1 | 1,419.8 |
| **Total** | **804,745** | **1.0** | **5,355.3** |

Written by me, an AI, from the paper's Tables 4 and 5 (all rows). The Code row of Table 4 is "17K" in the table and the text says "17k algorithm competition questions, along with 8k bug fixing problems".
- **Prompts and Samples:** the number of questions used for RL, and the number of question-answer examples used for supervised fine-tuning and distillation.
- **Question and output type:** how correctness is checked (a match with the reference answer gives a reward of 1, otherwise 0; code runs against hidden tests).
- **Average rounds:** turns of conversation; almost all data is single-turn. **Average tokens:** length of a sample; the math and code samples are long because they include the reasoning.

**Summary: what the tables show.** The reasoning RL data is only 80K questions (my sum 26 + 17 + 22 + 15; 88K with the bug-fixing questions), mostly math and code, whose answers can be checked by a rule. The SFT data is 804,745 samples; my sums: 626,933 reasoning-related (matching "about 600k") and 177,812 general ("approximately 200k"); 75.4% is math or code; the weighted average length 5,355.3 matches the paper's total (my check). **Caution:** the data is almost all single-turn, which the paper says may limit multi-turn skill.

## The core mechanisms, step by step

### One GRPO update
1. **Sample a group.** *What it does:* for one question $q$, the current ("old") policy writes $G=16$ answers. *Why it matters:* the group is the comparison set; no value model is needed.
2. **Score each answer with the rule reward.** *What it does:* 1 for correct plus the format reward, 0 otherwise (see the equations). *Why it matters:* the signal cannot be gamed by a reward model's quirks.
3. **Compute each answer's advantage** by subtracting the group mean and dividing by the group standard deviation. *Why it matters:* an answer is rewarded for being better than its siblings, so easy and hard questions give similar-sized signals.
4. **Update the policy** to make above-average answers more likely, with the probability ratio clipped and a KL penalty towards a reference model. *Why it matters:* the clip and KL keep the model near the sampling policy and the reference.
5. **Every 400 steps, replace the reference model** by the latest policy (paper, 2.1). *Why it matters:* after thousands of steps the model drifts far from the starting point, and a fixed reference would hold it back.

**Worked example (my own tiny numbers):** four answers to one question, two correct (rewards 1, 0, 0, 1). Mean 0.5; standard deviation 0.5 if computed over the four numbers (the paper does not say which standard deviation), so the advantages are $+1, -1, -1, +1$. With one correct out of four: advantages $+1.73, -0.58, -0.58, -0.58$. With all four correct, the standard deviation is 0 and the advantage is $0/0$; the paper does not say how that case is handled.

### Why GRPO and not PPO (paper, appendix A.3)
PPO learns a value model of the same size as the policy to estimate how good a partial answer is; with only an outcome reward at the end of a very long chain of thought, which reflects and revises itself, that is "inherently difficult". PPO also adds its KL penalty as a per-token reward, which the authors say may implicitly penalise length. GRPO drops the value model and adds an unbiased KL estimate straight into the loss. On the MATH task (a 16B mixture-of-experts coder model, Figure 4) PPO with the common GAE setting $\lambda=0.95$ did considerably worse than GRPO; with $\lambda=1.0$ it came close. The paper concludes GRPO is more practical, not that PPO cannot match it.

### Distillation (paper, appendix B.4.3 and F)
Fine-tune a smaller base model for 2 to 3 epochs on the 800K samples with a cosine schedule to one tenth of the starting rate, batch 64, context 32,768. The base models and starting rates are in the paper's Table 6: Qwen2.5-Math-1.5B (1e-4), Qwen2.5-Math-7B (8e-5), Qwen2.5-14B (7e-5), Qwen2.5-32B (6e-5), Llama-3.1-8B (5e-5) and Llama-3.3-70B-Instruct (2e-5). No RL is run on these (paper, F).

## The equations

### The GRPO objective (paper's equation 1, repeated as 11)

$$
\mathcal{J}_{\text{GRPO}}(\theta)=\mathbb{E}\Big[\tfrac1G\sum_{i=1}^{G}\Big(\min\Big(\tfrac{\pi_\theta(o_i\mid q)}{\pi_{\theta_{\text{old}}}(o_i\mid q)}A_i,\ \operatorname{clip}\Big(\tfrac{\pi_\theta(o_i\mid q)}{\pi_{\theta_{\text{old}}}(o_i\mid q)},\,1-\varepsilon,\,1+\varepsilon\Big)A_i\Big)-\beta\,\mathbb{D}_{KL}\big(\pi_\theta\,\|\,\pi_{\text{ref}}\big)\Big)\Big]
$$

**Where:**
- $q$ is a question and $o_1,\dots,o_G$ are the $G$ answers sampled for it from the old policy $\pi_{\theta_{\text{old}}}$. **Matters:** the group is what the advantage is measured against.
- $\pi_\theta(o_i\mid q)$ is the probability the model being trained gives to answer $o_i$. The ratio to the old probability tells how much this update changes the answer's likelihood.
- $A_i$ is the advantage of answer $i$ (next equation). **Matters:** positive means "make this answer likelier", negative the opposite.
- $\operatorname{clip}(\cdot,1-\varepsilon,1+\varepsilon)$ limits the ratio and the $\min$ takes the more cautious term. **Matters:** it stops one update from changing the policy too much. The paper's first RL stage sets $\varepsilon=10$, which is an unusual value, so the allowed ratio range is $[-9,11]$ and in practice only the upper clip can act.
- $\beta\,\mathbb{D}_{KL}(\pi_\theta\,\|\,\pi_{\text{ref}})$ is the leash to the reference model. $\beta=0.001$ here. **Matters:** without it the model may drift or exploit the reward.

**Example (tiny numbers):** advantages $(+1,-1,-1,+1)$, ratios $(1.1,0.9,1.3,1.0)$, $\varepsilon=0.2$, a KL estimate of 0.02 and $\beta=0.001$. The four terms are $\min(1.1,1.1)=1.1$, $\min(-0.9,-0.9)=-0.9$, $\min(-1.3,-1.2)=-1.3$, and $1.0$; the mean is $-0.025$, and the KL term subtracts $0.001\times0.02=0.00002$: objective $-0.02502$. The answer means: the third answer, which was bad and became 30% likelier, pulls the objective down the most. With $\varepsilon=10$ instead, a ratio of 1.3 with a positive advantage is not clipped at all (1.3 rather than 1.2).

### The KL estimate (paper's equation 2)

$$
\mathbb{D}_{KL}\big(\pi_\theta\,\|\,\pi_{\text{ref}}\big)=\frac{\pi_{\text{ref}}(o_i\mid q)}{\pi_\theta(o_i\mid q)}-\log\frac{\pi_{\text{ref}}(o_i\mid q)}{\pi_\theta(o_i\mid q)}-1
$$

**Where:**
- $\pi_{\text{ref}}$ is the reference policy and $\pi_\theta$ the current one, both evaluated on the sampled answer $o_i$.
- Write $r=\pi_{\text{ref}}/\pi_\theta$; the estimate is $r-\log r-1$. **Matters:** it is never negative and is zero exactly when $r=1$, so it measures how far the two models have drifted apart using only sampled answers. The paper calls it "an unbiased estimator" and cites Schulman (2020).

**Example (tiny numbers):** $r=1$: $0$. $r=0.8$: $0.8-\ln0.8-1=0.0231$. $r=2$: $2-0.693-1=0.307$. $r=0.5$: $0.193$. The answer means: if the new model makes this answer twice as likely as the reference did ($r=0.5$), the leash charges $0.193$; a small move costs very little.

### The advantage (paper's equation 3)

$$
A_i=\frac{r_i-\operatorname{mean}(\{r_1,\dots,r_G\})}{\operatorname{std}(\{r_1,\dots,r_G\})}
$$

**Where:**
- $r_i$ is the reward of answer $i$ (not the ratio of the KL equation) and $G$ the group size (16).
- $\operatorname{mean}$ and $\operatorname{std}$ are over the group. **Matters:** subtracting the mean makes the advantage zero on average, so only the *relative* quality counts and no value model is needed; dividing by the standard deviation rescales hard and easy questions to similar sizes.

**Example (tiny numbers):** rewards $(1,0,0,1)$: mean 0.5, std 0.5, advantages $(+1,-1,-1,+1)$. Rewards $(1,0,0,0)$: mean 0.25, std 0.433, advantages $(+1.73,-0.58,-0.58,-0.58)$. The answer means: a lone success on a hard question gets a bigger push (1.73) than one of two successes (1.0).

### The reward for R1-Zero (paper's equation 4)

$$
\text{Reward}_{\text{rule}}=\text{Reward}_{\text{acc}}+\text{Reward}_{\text{format}}
$$

**Where:**
- $\text{Reward}_{\text{acc}}$ is 1 if the final answer matches the reference (or the code passes the tests), else 0 (appendix B.3.1).
- $\text{Reward}_{\text{format}}$ rewards putting the reasoning inside the `<think>` tags. **Matters:** it makes the thought process visible; the paper does not state its numeric value, so I take it as 1 for the example.
- They are combined "with the same weight" (section 2.2). **Matters:** without the accuracy part nothing teaches correctness; without the format part the reasoning is not delimited.

**Example (tiny numbers, with my assumption that the format reward is 1):** a correct, well-formatted answer scores $1+1=2$; correct but unformatted, 1; wrong but formatted, 1; wrong and unformatted, 0. The answer means: the model first earns the format points, then the correctness points that actually teach it.

### The language-consistency reward (paper's equation 7)

$$
\text{Reward}_{\text{language}}=\frac{\operatorname{Num}(\text{Words}_{\text{target}})}{\operatorname{Num}(\text{Words})}
$$

**Where:**
- $\operatorname{Num}(\text{Words}_{\text{target}})$ is the number of words in the chain of thought written in the language of the question; $\operatorname{Num}(\text{Words})$ is all words. **Matters:** it is 1 for a thought written fully in the target language and falls as English and Chinese mix.
- It is added directly to the final reward in both reasoning and non-reasoning data (section 3.2.1). **Matters:** the paper's ablation shows it keeps language consistent at a slight cost in benchmark score (appendix B.6).

**Example (tiny numbers):** a thought of 50 words of which 45 are in the question's language: $45/50=0.9$. The answer means: 10% of the thought is in the wrong language, and the reward is 0.9.

### The reward in the second RL stage (paper's equations 8 to 10)

$$
\text{Reward}=\text{Reward}_{\text{reasoning}}+\text{Reward}_{\text{general}}+\text{Reward}_{\text{language}},\quad \text{Reward}_{\text{reasoning}}=\text{Reward}_{\text{rule}},\quad \text{Reward}_{\text{general}}=\text{Reward}_{\text{reward\_model}}+\text{Reward}_{\text{format}}
$$

**Where:**
- $\text{Reward}_{\text{reasoning}}$ applies to math, code, STEM and logic prompts and equals the rule reward above.
- $\text{Reward}_{\text{general}}$ applies to general prompts: a score from the helpful or safety reward model (the paper's equations 5 and 6) plus the format reward.
- $\text{Reward}_{\text{language}}$ is the language-consistency term. **Matters:** each prompt gets the reward matching its type, so one RL run improves reasoning and general behaviour together.

**Example (tiny numbers, my assumptions):** a math prompt answered correctly and formatted: $2$ from the rule reward, plus a language score $0.9$: $2.9$. A general prompt with a reward-model score of $0.7$, format 1 and language $0.9$: $0.7+1+0.9=2.6$. The answer means: the two kinds of prompt are scored on one common sum, but the scale of the reward-model part is not given in the paper.

## What they found
Numbers are marked "paper" or "abstract". I read the main text, appendices A to C, D.2, F, G and the start of H; the safety report (D.3), the benchmark descriptions and the extra analyses (E) I did not read in detail.
- **R1-Zero learns to reason (paper, 2.3 and Figure 1):** the average pass@1 score on AIME 2024 rose from 15.6% to 77.9% during RL, and to 86.7% with self-consistency decoding over 16 samples (cons@16); the average response length grew steadily to many thousands of tokens.
- **Reflection grows (appendix C.2):** the count of reflective words ("wait", "verify", "check" and others, picked by 3 human experts) rose 5 to 7 times; the word "wait" was nearly absent early, occasional between steps 4,000 and 7,000, and spiked after step 8,000. Table 2 shows an "aha moment" in an intermediate checkpoint: it writes "Wait, wait. Wait. That's an aha moment I can flag here." and re-checks. This is one example, not a measurement.
- **By difficulty (appendix C.1):** on MATH, level-5 problems improved from about 0.55 to 0.90 accuracy and level 4 from about 0.78 to 0.95 (read from the paper's text).
- **Cost (appendix B.4.4 and Table 7):** about 147K H800 GPU hours in all, 101K for R1-Zero, 5K for creating SFT data and 41K for R1, which at $2 per GPU hour is $294K (the paper's rental assumption).

### Each stage of R1 (paper's Table 3)

| Benchmark (metric) | R1-Zero | R1-Dev1 | R1-Dev2 | R1-Dev3 | R1 | R1 minus R1-Zero (mine) |
|---|---|---|---|---|---|---|
| MMLU (EM) | 88.8 | 89.1 | 91.2 | 91.0 | 90.8 | +2.0 |
| MMLU-Redux (EM) | 85.6 | 90.0 | 93.0 | 93.1 | 92.9 | +7.3 |
| MMLU-Pro (EM) | 68.9 | 74.1 | 83.8 | 83.1 | 84.0 | +15.1 |
| DROP (3-shot F1) | 89.1 | 89.8 | 91.1 | 88.7 | 92.2 | +3.1 |
| IF-Eval (prompt strict) | 46.6 | 71.7 | 72.0 | 78.1 | 83.3 | +36.7 |
| GPQA Diamond (pass@1) | 75.8 | 66.1 | 70.7 | 71.2 | 71.5 | -4.3 |
| SimpleQA (correct) | 30.3 | 17.8 | 28.2 | 24.9 | 30.1 | -0.2 |
| FRAMES (acc.) | 82.3 | 78.5 | 81.8 | 81.9 | 82.5 | +0.2 |
| AlpacaEval 2.0 (LC win rate) | 24.7 | 50.1 | 55.8 | 62.1 | 87.6 | +62.9 |
| ArenaHard (GPT-4-1106) | 53.6 | 77.0 | 73.2 | 75.6 | 92.3 | +38.7 |
| LiveCodeBench (pass@1-CoT) | 50.0 | 57.5 | 63.5 | 64.6 | 65.9 | +15.9 |
| Codeforces (percentile) | 80.4 | 84.5 | 90.5 | 92.1 | 96.3 | +15.9 |
| Codeforces (rating) | 1444 | 1534 | 1687 | 1746 | 2029 | +585 |
| SWE Verified (resolved) | 43.2 | 39.6 | 44.6 | 45.6 | 49.2 | +6.0 |
| Aider-Polyglot (acc.) | 12.2 | 6.7 | 25.6 | 44.8 | 53.3 | +41.1 |
| AIME 2024 (pass@1) | 77.9 | 59.0 | 74.0 | 78.1 | 79.8 | +1.9 |
| MATH-500 (pass@1) | 95.9 | 94.2 | 95.9 | 95.4 | 97.3 | +1.4 |
| CNMO 2024 (pass@1) | 88.1 | 58.0 | 73.9 | 77.3 | 78.8 | -9.3 |
| CLUEWSC (EM) | 93.1 | 92.8 | 92.6 | 91.6 | 92.8 | -0.3 |
| C-Eval (EM) | 92.8 | 85.7 | 91.9 | 86.4 | 91.8 | -1.0 |
| C-SimpleQA (correct) | 66.4 | 58.8 | 64.2 | 66.9 | 63.7 | -2.7 |

Written by me, an AI, from the paper's Table 3 (all 21 rows and 5 columns; the paper bolds the cells that are statistically significant by a t-test with p below 0.01, but the bold is lost in the text extraction, so I cannot say which cells). The last column is mine.
- **Benchmark (metric):** EM is exact match; F1 is overlap of answer words; pass@1 is the share of problems solved on the first try (averaged over samples; see [[Pass at k]]); Codeforces percentile and rating compare with human contestants; "resolved" is the share of real software issues fixed. MMLU, MMLU-Redux and MMLU-Pro are multiple-choice knowledge tests; GPQA Diamond is graduate-level science questions; AIME 2024, MATH-500 and CNMO 2024 are maths contests and problem sets; AlpacaEval 2.0 and ArenaHard are chat-quality tests judged by a model ([[LLM as a Judge]]); IF-Eval measures following format instructions; SimpleQA and C-SimpleQA are short factual questions in English and Chinese (general knowledge for these descriptions).
- **R1-Zero to R1:** the five checkpoints of Figure 2, in order. Dev1 is after cold-start SFT, Dev2 after reasoning RL, Dev3 after the 800K-sample SFT, R1 after the last RL.

**Summary: what the table shows.** R1 is better than R1-Zero on 15 of 21 rows. The biggest gains are the chat and instruction rows (AlpacaEval +62.9, IF-Eval +36.7, ArenaHard +38.7), Aider-Polyglot (+41.1) and the Codeforces rating (+585). R1 is lower on six rows: CNMO (-9.3), GPQA Diamond (-4.3), C-SimpleQA (-2.7), C-Eval (-1.0), CLUEWSC (-0.3) and SimpleQA (-0.2). The cold-start step first hurt reasoning: Dev1 is 18.9 points below R1-Zero on AIME and 30.1 on CNMO. **Caution:** the text says AlpacaEval improved "by 25%" and ArenaHard "by 17%", but those are points from Dev3 to R1 (62.1 to 87.6 and 75.6 to 92.3); relative gains are 41% and 22% (my arithmetic). Single runs, no error bars in the table beyond the unreadable bold marks.

### Against other models (paper's Table 8)

| Benchmark (metric) | Claude-3.5-Sonnet-1022 | GPT-4o 0513 | DeepSeek-V3 | OpenAI o1-mini | OpenAI o1-1217 | DeepSeek-R1 | R1 minus V3 (mine) | R1 minus o1-1217 (mine) |
|---|---|---|---|---|---|---|---|---|
| MMLU (EM) | 88.3 | 87.2 | 88.5 | 85.2 | 91.8 | 90.8 | +2.3 | -1.0 |
| MMLU-Redux (EM) | 88.9 | 88.0 | 89.1 | 86.7 | - | 92.9 | +3.8 | - |
| MMLU-Pro (EM) | 78.0 | 72.6 | 75.9 | 80.3 | - | 84.0 | +8.1 | - |
| DROP (3-shot F1) | 88.3 | 83.7 | 91.6 | 83.9 | 90.2 | 92.2 | +0.6 | +2.0 |
| IF-Eval (prompt strict) | 86.5 | 84.3 | 86.1 | 84.8 | - | 83.3 | -2.8 | - |
| GPQA Diamond (pass@1) | 65.0 | 49.9 | 59.1 | 60.0 | 75.7 | 71.5 | +12.4 | -4.2 |
| SimpleQA (correct) | 28.4 | 38.2 | 24.9 | 7.0 | 47.0 | 30.1 | +5.2 | -16.9 |
| FRAMES (acc.) | 72.5 | 80.5 | 73.3 | 76.9 | - | 82.5 | +9.2 | - |
| AlpacaEval 2.0 (LC win rate) | 52.0 | 51.1 | 70.0 | 57.8 | - | 87.6 | +17.6 | - |
| ArenaHard (GPT-4-1106) | 85.2 | 80.4 | 85.5 | 92.0 | - | 92.3 | +6.8 | - |
| LiveCodeBench (pass@1-CoT) | 38.9 | 32.9 | 36.2 | 53.8 | 63.4 | 65.9 | +29.7 | +2.5 |
| Codeforces (percentile) | 20.3 | 23.6 | 58.7 | 93.4 | 96.6 | 96.3 | +37.6 | -0.3 |
| Codeforces (rating) | 717 | 759 | 1134 | 1820 | 2061 | 2029 | +895 | -32 |
| SWE Verified (resolved) | 50.8 | 38.8 | 42.0 | 41.6 | 48.9 | 49.2 | +7.2 | +0.3 |
| Aider-Polyglot (acc.) | 45.3 | 16.0 | 49.6 | 32.9 | 61.7 | 53.3 | +3.7 | -8.4 |
| AIME 2024 (pass@1) | 16.0 | 9.3 | 39.2 | 63.6 | 79.2 | 79.8 | +40.6 | +0.6 |
| MATH-500 (pass@1) | 78.3 | 74.6 | 90.2 | 90.0 | 96.4 | 97.3 | +7.1 | +0.9 |
| CNMO 2024 (pass@1) | 13.1 | 10.8 | 43.2 | 67.6 | - | 78.8 | +35.6 | - |
| CLUEWSC (EM) | 85.4 | 87.9 | 90.9 | 89.9 | - | 92.8 | +1.9 | - |
| C-Eval (EM) | 76.7 | 76.0 | 86.5 | 68.9 | - | 91.8 | +5.3 | - |
| C-SimpleQA (correct) | 55.4 | 58.7 | 68.0 | 40.3 | - | 63.7 | -4.3 | - |

Written by me, an AI, from the paper's Table 8 (all 21 rows and 6 model columns; "-" is a blank in the paper, not zero; the paper's bold marks are lost in extraction). The last two columns are mine. The table also says that DeepSeek-V3 and R1 are mixture-of-experts models with 671B total and 37B active parameters; the others are not described.
- **Columns:** the first five are the paper's comparison models; the benchmark names are explained under the previous table.
- **R1 minus V3 (mine):** the effect of the whole reasoning pipeline, since R1 starts from the same base as V3. **R1 minus o1-1217 (mine):** only the 11 rows where the paper gives an o1 value.

**Summary: what the table shows.** R1 beats DeepSeek-V3 on 19 of 21 rows, by the most on AIME (+40.6), Codeforces (+37.6 percentile) and CNMO (+35.6), and loses on IF-Eval (-2.8) and C-SimpleQA (-4.3). Against OpenAI o1-1217 it is higher on 5 of the 11 rows with a value (DROP, LiveCodeBench, SWE Verified, AIME, MATH-500, each by 2.5 or less) and lower on the rest, most on SimpleQA (-16.9), Aider-Polyglot (-8.4) and GPQA Diamond (-4.2). **Caution:** the "on par" claim for maths rests on gaps of 0.6 and 0.9 points from a single reported number each; the paper itself notes that o1 is better on Aider and that R1's engineering gain over V3 is small.

### Distilled models (paper's Table 15)

| Model | AIME 2024 pass@1 | AIME 2024 cons@64 | MATH-500 pass@1 | GPQA Diamond pass@1 | LiveCodeBench pass@1 | Codeforces rating |
|---|---|---|---|---|---|---|
| GPT-4o-0513 | 9.3 | 13.4 | 74.6 | 49.9 | 32.9 | 759 |
| Claude-3.5-Sonnet-1022 | 16.0 | 26.7 | 78.3 | 65.0 | 38.9 | 717 |
| R1-Distill-Qwen-1.5B | 28.9 | 52.7 | 83.9 | 33.8 | 16.9 | 954 |
| R1-Distill-Qwen-7B | 55.5 | 83.3 | 92.8 | 49.1 | 37.6 | 1189 |
| R1-Distill-Qwen-14B | 69.7 | 80.0 | 93.9 | 59.1 | 53.1 | 1481 |
| R1-Distill-Qwen-32B | 72.6 | 83.3 | 94.3 | 62.1 | 57.2 | 1691 |
| R1-Distill-Llama-8B | 50.4 | 80.0 | 89.1 | 49.0 | 39.6 | 1205 |
| R1-Distill-Llama-70B | 70.0 | 86.7 | 94.5 | 65.2 | 57.5 | 1633 |

Written by me, an AI, from the paper's Table 15 (all 8 rows, 6 columns). The cons@64 column is the score by majority vote over 64 samples.
- **Rows:** two closed non-reasoning models, then six models fine-tuned on R1's 800K samples.

**Summary: what the table shows.** Even the 1.5B distilled model beats GPT-4o and Claude-3.5-Sonnet on the maths columns (AIME +19.6 and +12.9, MATH-500 +9.3 and +5.6 against them; my arithmetic) but is far behind on GPQA Diamond (33.8 against 49.9 and 65.0) and LiveCodeBench (16.9 against 32.9 and 38.9). Scores rise with size, and the 32B and 70B distilled models beat both closed models on every column except GPQA Diamond for the 32B against Claude (62.1 against 65.0). **Caution:** the paper's phrase that a 1.5B model "achieves superior performance compared to the best closed-source models" holds for maths only.

### Distilling or doing RL on the small model? (paper's Table 16)

| Model | AIME 2024 pass@1 | AIME 2024 cons@64 | MATH-500 pass@1 | GPQA Diamond pass@1 | LiveCodeBench pass@1 |
|---|---|---|---|---|---|
| QwQ-32B-Preview | 50.0 | 60.0 | 90.6 | 54.5 | 41.9 |
| Qwen2.5-32B-Zero (RL only) | 47.0 | 60.0 | 91.6 | 55.0 | 40.2 |
| R1-Distill-Qwen-32B | 72.6 | 83.3 | 94.3 | 62.1 | 57.2 |

Written by me, an AI, from the paper's Table 16 (all rows). Qwen2.5-32B-Zero is the 32B base model trained with the same kind of large-scale RL for over 10,000 steps (appendix F.1).

**Summary: what the table shows.** The distilled 32B beats the RL-only 32B by 25.6 on AIME pass@1, 23.3 on AIME cons@64, 2.7 on MATH-500, 7.1 on GPQA Diamond and 17.0 on LiveCodeBench (my arithmetic), while RL-only is only on par with QwQ-32B-Preview. The paper's conclusion: distilling a stronger model is economical and effective; going beyond human-level reasoning may still need stronger base models and larger-scale RL. **Caution:** one base model family at 32B and a different amount of compute for each route; the paper gives no cost for the RL-only run beyond "enormous".

### The paper's own inconsistencies and oddities
- **Training time:** appendix B.4.4 says R1's training took "about 4 days, or roughly 80 hours"; 4 days is 96 hours. The GPU hours in Table 7 fit 80 hours: 512 GPUs times 80 hours is 40,960, which is the table's 41K (and 512 times 198 hours is 101,376, the table's 101K; my arithmetic).
- **Model size:** appendix B.4.4 says "660B R1-Zero and R1", while appendix A.1 and Table 8 say 671B.
- **Epochs:** 10,400 steps of 32 questions is 332,800 question visits (my arithmetic). The text says this is "1.6 training epochs", which would mean about 208,000 different prompts, but Table 4 lists 80K reasoning prompts (88K with the bug-fixing set), which would be about 4 epochs. The paper does not say what data R1-Zero used.
- **The clip ratio:** the first RL stage of R1 uses $\varepsilon=10$, far from the usual 0.2, although the text says the clip ratio "plays a crucial role"; I take the printed value at face value.
- **Percent against points:** the "25%" and "17%" gains are points (see Table 3 above).

## Questions I had, answered

**What is a "verifiable" reward and why does it matter so much?**
A reward you can compute by a rule: a math answer equals the reference, code passes hidden tests, a multiple-choice option matches. It cannot be fooled by style. The paper's conclusion is that what unlocks reasoning is "hard reasoning questions, a reliable verifier, and sufficient computational resources" rather than human annotation. The reverse also holds: with a neural reward model on general prompts, they saw reward hacking (Figure 6: the reward rose over 700 steps while Codeforces pass@1 fell from about 0.35 to about 0.28; read by eye from the axis ranges).

**Why divide by the standard deviation in the advantage?**
So that every question gives a signal of similar size. Example: with one correct answer out of four, the correct one gets +1.73 and the others -0.58; with two out of four, the right ones get +1 and the wrong ones -1. The paper does not discuss why it divides; this is my reading.

**What is the "aha moment", really?**
In the paper's Table 2, an intermediate R1-Zero checkpoint stops in the middle of solving an equation and writes "Wait, wait. Wait. That's an aha moment I can flag here", then restarts the check. The paper links it to a rise in the word "wait" after step 8,000. It is one example plus a word count, a hint of reflection, and the same appendix says the vivid human-like style in the final R1 is "DeepSeek-engineered heuristics", not a sign of human-like intelligence (appendix B.3.2).

**Is R1 a mixture-of-experts model?**
Yes, through its base: DeepSeek-V3-Base has 671B parameters and activates 37B per token, 5.5% of them (my arithmetic; appendix A.1). See [[Mixtral 2024]] and [[Mixture of Experts]] for the idea.

**Why did the paper keep SFT if RL alone worked?**
Because R1-Zero was hard to read, mixed languages and was weak at writing and open questions. The authors say both are indispensable: RL for exploring reasoning, SFT for tasks where no reliable reward exists (appendix G.1).

## Why it mattered
It showed that reasoning behaviour such as reflection and self-checking can be grown by RL with only a rule-based correctness reward, on a strong enough base model, and it released the models, including distilled ones (paper). Whether it triggered the later wave of open reasoning models and its place against closed models is general knowledge and not checked here. The authors themselves warn that they could not make pure RL work with 7B and 16B bases (appendix G.1).

## Limits and open questions
- **Needs a big base model:** with a 7B dense and a 16B mixture-of-experts base the RL "consistently failed to yield meaningful improvements" (appendix G.1).
- **Needs a reliable verifier:** for writing and open tasks, pure RL is an open challenge and reward hacking appears (paper, 6).
- **Capability gaps:** weak structured output and no tool use; token inefficiency and overthinking on simple questions; language mixing outside Chinese and English; sensitivity to prompts (few-shot prompting makes it worse); little gain over V3 on software engineering (paper, 6).
- **Safety:** the authors say jailbreaks can produce dangerous plans, that the public model can be fine-tuned to remove safeguards, and that its safety is "generally at a moderate level (comparable to GPT-4o)" (paper, 5). I did not read the safety appendix.
- **Single-turn data:** the SFT data is mostly single-turn.
- **Unsuccessful attempts (appendix G.2):** a process reward model (hard to define steps, to check them, and prone to hacking) and Monte Carlo tree search (a huge token search space and a hard-to-train value model).
- **Possible contamination:** the paper says web pages in the base model's data contain many answers generated by OpenAI models and that it did not add such data on purpose (appendix A.1); a decontamination procedure is in appendix D.1, which I did not read.
- **Internal inconsistencies:** see the notice above.

## Concepts used
[[Reasoning Model]], [[GRPO]], [[PPO]], [[Reinforcement Learning]], [[Chain of Thought]], [[Reward Model]], [[Instruction Tuning]], [[Knowledge Distillation]], [[Mixture of Experts]], [[Pass at k]], [[KL Divergence]], [[LLM as a Judge]], [[Benchmarks]], [[Chatbot Arena]], [[Fine-tuning]]

## Reading trail
- **Came from:** [[Mixtral 2024]] (also a mixture-of-experts model; R1's base is one), [[InstructGPT 2022]] (the SFT-then-RL recipe it departs from) and [[Scaling Laws 2020]] (cited in the introduction for reasoning emerging with scale).
- **Led to:** no paper in this vault yet; the later use of its recipe is from memory and not checked.

## How much I read
I read the PDF (arXiv version 2, 86 pages) as extracted text: the abstract and main text (pages 1 to 12), appendix A (background and GRPO against PPO, pages 13 to 16), the RL data and SFT data descriptions and statistics (B.3, pages 19 to 27), training details, cost, reward hacking and the language-consistency ablation (B.4 to B.6, pages 35 to 37), the self-evolution analysis (C), the main results and the Chatbot Arena paragraph (D.2, pages 41 to 43), the distillation sections (F, pages 60 to 62), the discussion and unsuccessful attempts (G, pages 62 to 64) and the start of the related work. I did **not** read: the example prompts and reasoning traces (pages 23 to 34), the safety report (D.3, pages 44 to 54), the comparison with DeepSeek-V3 and the fresh-test-set, test-time-scaling and maths-category analyses (E, pages 55 to 60), the benchmark descriptions (pages 66 to 78) or the references. I looked at the pipeline figure on page 6 as a picture and the Commons copy; the other figures I know from captions and text. Numbers marked "mine" are my arithmetic. Not opened in Obsidian, so the flashcards are unreviewed.

## Sources
- DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", arXiv 2501.12948 (version 2, 4 January 2026): <https://arxiv.org/abs/2501.12948>. Licence on the arXiv page: standard non-exclusive distribution licence, so nothing was copied from the PDF.
- Figure: [The multistage pipeline of DeepSeek-R1.png](https://commons.wikimedia.org/wiki/File:The_multistage_pipeline_of_DeepSeek-R1.png) (DeepSeek-AI authors, from the Nature article s41586-025-09422-z, CC BY 4.0), Wikimedia Commons.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-llm DeepSeek-R1 (DeepSeek-AI, 2025, arXiv 2501.12948): what problem did it attack?::Reasoning in LLMs depended on human-written reasoning examples, which do not scale, add human bias and cap the model at human ways of thinking. Can reasoning be incentivised with reinforcement learning alone, with no human-labelled reasoning steps?

#flashcards/research/concept #flashcards/topic/research-llm What did people do before DeepSeek-R1?::Supervised fine-tuning on human reasoning traces followed by RL with PPO and a learned reward model; chain-of-thought prompting; neural process or outcome reward models (which the authors found prone to reward hacking); majority voting and tree search at inference.

#flashcards/research/concept #flashcards/topic/research-llm DeepSeek-R1: what is the idea?::Skip SFT: train DeepSeek-V3-Base with GRPO and a rule-based reward that only checks the final answer's correctness and the format (R1-Zero). Then repair readability and language mixing with cold-start SFT, two RL stages and a rejection-sampling SFT, and distil into smaller models.

#flashcards/research/concept #flashcards/topic/research-llm What base model does R1 use and how big is it?::DeepSeek-V3-Base, a mixture-of-experts model with 671B parameters, 37B active per token (5.5%, my arithmetic), pre-trained on 14.8T tokens with multi-head latent attention and multi-token prediction (appendix A.1).

#flashcards/research/concept #flashcards/topic/research-llm How was DeepSeek-R1-Zero trained (settings)?::GRPO from DeepSeek-V3-Base with no SFT: learning rate 3e-6, KL coefficient 0.001, temperature 1, 16 outputs per question up to 32,768 tokens (65,536 after step 8,200), 32 questions per step (512 answers), 10,400 steps, reference model replaced every 400 steps.

#flashcards/research/concept #flashcards/topic/research-llm What was the prompt template of R1-Zero and why so simple?::A conversation where the assistant first thinks inside think tags then answers inside answer tags. It constrains only the structure, so the authors can observe the model's natural progress without content bias.

#flashcards/research/concept #flashcards/topic/research-llm Why does R1-Zero use rule-based rewards and no neural reward model for reasoning?::Neural reward models are susceptible to reward hacking in large-scale RL and need retraining. A rule (answer matches the reference, code passes tests) cannot be argued with. The rewards are accuracy plus format with equal weight.

#flashcards/research/concept #flashcards/topic/research-llm What are the four stages of the DeepSeek-R1 pipeline?::(1) Cold-start SFT on thousands of long chain-of-thought examples; (2) reasoning RL with a language-consistency reward; (3) rejection sampling and SFT on about 800K samples (about 600K reasoning, 200K non-reasoning) starting from V3-Base; (4) a second RL with rule rewards plus helpfulness and safety reward models (1,700 steps, preference data only in the last 400).

#flashcards/research/concept #flashcards/topic/research-llm ![[deepseek-r1-pipeline.png]]
What does this pipeline figure show?
?
Left: R1-Zero (RL from V3 Base with accuracy-and-format rewards), whose samples are filtered and refined into data. Middle: cold-start SFT (Dev1) then RL with rule-based and language-consistency rewards (Dev2). Right: SFT on sampled reasoning plus V3 non-reasoning data (Dev3) then RL with rule-based and preference rewards gives R1. (Nature copy of the paper's Figure 2, CC BY 4.0.)

#flashcards/research/concept #flashcards/topic/research-llm Why does R1 need a cold-start stage, and what did it cost?::R1-Zero is hard to read and mixes English and Chinese. Cold-start data in a first-person conversational style fixes that, but the small dataset made Dev1 worse on reasoning: AIME 59.0 against R1-Zero's 77.9 (-18.9, my arithmetic) and CNMO 58.0 against 88.1.

#flashcards/research/concept #flashcards/topic/research-llm What is in R1's 800K SFT data (Table 5)?::804,745 samples: Math 395,285, Code 211,129, STEM 10,124, Logic 10,395, General 177,812; average 5,355.3 tokens; almost all single-turn. 626,933 are reasoning-related (about 600K) and 177,812 general (about 200K).

#flashcards/research/concept #flashcards/topic/research-llm What RL prompts did R1 use (Table 4)?::Math 26K, Code 17K (plus 8K bug-fixing), STEM 22K (multiple choice), Logic 15K and General 66K (helpfulness and harmlessness). The reasoning prompts total 80K by my sum (88K with the bug-fixing set).

#flashcards/research/concept #flashcards/topic/research-llm How were R1's reward models for general prompts built?::Helpful: 66,000 preference pairs from DeepSeek-V3 judgments (judged four times with swapped positions, kept if the score gap exceeded 1, equal lengths), scoring only the final summary. Safety: 106,000 responses labelled safe or unsafe, point-wise, scoring the whole response.

#flashcards/research/concept #flashcards/topic/research-llm What are the steps of one GRPO update?::Sample a group of $G=16$ answers per question; score each with the rule reward; compute each advantage as reward minus group mean over group std; update the policy with a clipped probability ratio and a KL penalty to a reference model; replace the reference by the latest policy every 400 steps.

#flashcards/research/concept #flashcards/topic/research-llm Why GRPO and not PPO?::PPO's value model is as big as the policy and hard to train when only a final outcome reward exists and the chain of thought revises itself; PPO's per-token KL reward may also penalise length. GRPO needs no value model. In the paper's MATH test PPO with GAE lambda 0.95 was much worse; with lambda 1.0 it came close.

#flashcards/research/concept #flashcards/topic/research-llm Write the GRPO objective (equation 1) and explain each symbol.
?
$$\mathcal{J}_{\text{GRPO}}(\theta)=\mathbb{E}\Big[\tfrac1G\sum_{i=1}^{G}\Big(\min\Big(\tfrac{\pi_\theta(o_i\mid q)}{\pi_{\theta_{\text{old}}}(o_i\mid q)}A_i,\ \operatorname{clip}\Big(\tfrac{\pi_\theta(o_i\mid q)}{\pi_{\theta_{\text{old}}}(o_i\mid q)},\,1-\varepsilon,\,1+\varepsilon\Big)A_i\Big)-\beta\,\mathbb{D}_{KL}\big(\pi_\theta\,\|\,\pi_{\text{ref}}\big)\Big)\Big]$$
- $q$, $o_i$: the question and the $G$ answers sampled from the old policy.
- The probability ratio: how much this update changes the answer's likelihood.
- $A_i$: the advantage; positive means make this answer likelier.
- clip with $\varepsilon$, then min: stops one update from changing the policy too much ($\varepsilon=10$ in R1's first RL stage, so only the upper clip at 11 can act).
- $\beta\,\mathbb{D}_{KL}$: the leash to the reference model ($\beta=0.001$).

#flashcards/research/concept #flashcards/topic/research-llm GRPO example: advantages $(+1,-1,-1,+1)$, ratios $(1.1,0.9,1.3,1.0)$, $\varepsilon=0.2$, KL 0.02, $\beta=0.001$. What is the objective?::Terms 1.1, -0.9, -1.3 (min of -1.3 and -1.2) and 1.0; mean -0.025; minus $0.001\times0.02$ gives -0.02502 (my arithmetic). With $\varepsilon=10$ a ratio of 1.3 with a positive advantage is not clipped.

#flashcards/research/concept #flashcards/topic/research-llm Why the min of the unclipped and clipped terms in GRPO?::So the update takes the more cautious of the two. With advantage +1 and ratio 1.3 and $\varepsilon=0.2$ it uses 1.2, not 1.3; with advantage -1 it uses -1.3, the worse one, so bad answers are always fully penalised. With $\varepsilon=10$ the clip range is $[-9,11]$ and almost never acts.

#flashcards/research/concept #flashcards/topic/research-llm Write the GRPO KL estimate (equation 2) and explain each symbol.
?
$$\mathbb{D}_{KL}\big(\pi_\theta\,\|\,\pi_{\text{ref}}\big)=\frac{\pi_{\text{ref}}(o_i\mid q)}{\pi_\theta(o_i\mid q)}-\log\frac{\pi_{\text{ref}}(o_i\mid q)}{\pi_\theta(o_i\mid q)}-1$$
- $\pi_{\text{ref}}$, $\pi_\theta$: reference and current policy, evaluated on the sampled answer $o_i$.
- With $r=\pi_{\text{ref}}/\pi_\theta$ it is $r-\log r-1$: never negative and zero exactly at $r=1$; the paper calls it unbiased (Schulman, 2020).

#flashcards/research/concept #flashcards/topic/research-llm KL example: what are $r-\log r-1$ for $r=1$, $0.8$, $2$ and $0.5$?::0; 0.0231; 0.307; 0.193 (my arithmetic). A bigger drift costs more, and a tiny move costs almost nothing.

#flashcards/research/concept #flashcards/topic/research-llm Why is the KL estimate $r-\log r-1$ and not simply $\log r$?::$\log r$ can be negative for single samples, but $r-\log r-1\ge0$ with equality only at $r=1$ ($r=0.5$ gives 0.193, $r=2$ gives 0.307). It is an unbiased estimate of the KL divergence from samples and is added straight to the loss.

#flashcards/research/concept #flashcards/topic/research-llm Write the GRPO advantage (equation 3) and explain each symbol.
?
$$A_i=\frac{r_i-\operatorname{mean}(\{r_1,\dots,r_G\})}{\operatorname{std}(\{r_1,\dots,r_G\})}$$
- $r_i$: the reward of answer $i$; $G$: the group size (16).
- mean and std over the group: the advantage is relative, so no value model is needed, and hard and easy questions give similar-sized signals.

#flashcards/research/concept #flashcards/topic/research-llm Advantage example: rewards $(1,0,0,1)$ and $(1,0,0,0)$. What are the advantages?::$(+1,-1,-1,+1)$ with std 0.5, and $(+1.73,-0.58,-0.58,-0.58)$ with std 0.433 (my arithmetic, std over the four numbers; the paper does not say which std). With all four correct the std is 0 and the advantage is 0/0, which the paper does not discuss.

#flashcards/research/concept #flashcards/topic/research-llm Why does GRPO divide by the group standard deviation?::It rescales questions so each gives a signal of similar size: a lone success among four gets +1.73, one of two gets +1. The paper does not discuss this choice; this is my reading.

#flashcards/research/concept #flashcards/topic/research-llm Write R1-Zero's reward (equation 4) and explain each symbol.
?
$$\text{Reward}_{\text{rule}}=\text{Reward}_{\text{acc}}+\text{Reward}_{\text{format}}$$
- $\text{Reward}_{\text{acc}}$: 1 if the final answer matches the reference or the code passes the tests, else 0.
- $\text{Reward}_{\text{format}}$: rewards putting the reasoning inside the think tags (its numeric value is not stated).
- Equal weights; the neural reward models are deliberately not used.

#flashcards/research/concept #flashcards/topic/research-llm Rule-reward example (format reward assumed to be 1): what do the four combinations score?::Correct and formatted 2; correct but unformatted 1; wrong but formatted 1; wrong and unformatted 0 (my assumption for the format value).

#flashcards/research/concept #flashcards/topic/research-llm Write the language-consistency reward (equation 7) and explain each symbol.
?
$$\text{Reward}_{\text{language}}=\frac{\operatorname{Num}(\text{Words}_{\text{target}})}{\operatorname{Num}(\text{Words})}$$
- Numerator: words of the chain of thought in the language of the question; denominator: all words.
- It is 1 for a fully consistent thought and falls with mixing; it is added to the final reward and slightly lowers benchmark scores but makes reasoning readable.

#flashcards/research/concept #flashcards/topic/research-llm Language reward example: 45 of 50 words in the right language.::$45/50=0.9$ (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Write the second-stage RL reward (equations 8 to 10) and explain each symbol.
?
$$\text{Reward}=\text{Reward}_{\text{reasoning}}+\text{Reward}_{\text{general}}+\text{Reward}_{\text{language}},\quad \text{Reward}_{\text{reasoning}}=\text{Reward}_{\text{rule}},\quad \text{Reward}_{\text{general}}=\text{Reward}_{\text{reward\_model}}+\text{Reward}_{\text{format}}$$
- Reasoning prompts get the rule reward; general prompts get a reward-model score plus the format reward.
- The language term applies to both; one RL run thus improves reasoning and general behaviour.

#flashcards/research/concept #flashcards/topic/research-llm Second-stage reward example (my assumptions): a correct formatted math answer with language 0.9, and a general answer with RM score 0.7, format 1, language 0.9.::Math: $2+0.9=2.9$. General: $0.7+1+0.9=2.6$. The scale of the reward-model part is not given in the paper.

#flashcards/research/concept #flashcards/topic/research-llm How did R1-Zero's reasoning evolve during RL?::AIME 2024 pass@1 rose from 15.6% to 77.9% (86.7% with cons@16); response length grew steadily; reflective words (wait, verify, check and others) rose 5 to 7 times and "wait" spiked after step 8,000. Table 2 shows an "aha moment" example; it is one example, not a measurement.

#flashcards/research/concept #flashcards/topic/research-llm What does the "aha moment" show and what does it not?::An intermediate checkpoint writes "Wait, wait. Wait. That's an aha moment I can flag here" and re-checks its work; the word "wait" rises after step 8,000. The appendix says the human-like style in the final R1 is DeepSeek-engineered, not a sign of human-like intelligence.

#flashcards/research/concept #flashcards/topic/research-llm How did R1 do across its stages (Table 3)?::R1 beats R1-Zero on 15 of 21 rows, most on AlpacaEval 2.0 (+62.9), ArenaHard (+38.7), Aider-Polyglot (+41.1), IF-Eval (+36.7) and the Codeforces rating (+585); it is lower on CNMO (-9.3), GPQA Diamond (-4.3), C-SimpleQA, C-Eval, CLUEWSC and SimpleQA. Dev1 dropped on AIME to 59.0.

#flashcards/research/concept #flashcards/topic/research-llm The paper says AlpacaEval improved "by 25%" and ArenaHard "by 17%" from Dev3 to R1. What are they really?::Points: 62.1 to 87.6 (+25.5) and 75.6 to 92.3 (+16.7). Relative gains are 41% and 22% (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm How does R1 compare with DeepSeek-V3 and OpenAI o1-1217 (Table 8)?::R1 beats V3 on 19 of 21 rows (AIME +40.6, Codeforces percentile +37.6, CNMO +35.6) and loses on IF-Eval and C-SimpleQA. Against o1-1217 (11 rows with a value) it is higher on 5, each by 2.5 or less, and lower on the rest, most on SimpleQA (-16.9), Aider (-8.4) and GPQA Diamond (-4.2).

#flashcards/research/concept #flashcards/topic/research-llm How are the distilled models made and how well do they do (Table 15)?::Fine-tune smaller base models (Qwen 1.5B to 32B, Llama 8B and 70B) for 2 to 3 epochs on the 800K R1 samples, no RL. The 1.5B model beats GPT-4o and Claude-3.5-Sonnet on AIME and MATH-500 but not on GPQA Diamond or LiveCodeBench; the 32B scores 72.6 on AIME 2024 pass@1.

#flashcards/research/concept #flashcards/topic/research-llm Distillation or RL for a small model: what does Table 16 show?::The distilled Qwen-32B beats the RL-only Qwen2.5-32B-Zero by 25.6 on AIME pass@1, 7.1 on GPQA Diamond and 17.0 on LiveCodeBench (my arithmetic); RL-only matches QwQ-32B-Preview. Distilling a stronger model is economical; surpassing human reasoning may need stronger bases and more RL.

#flashcards/research/concept #flashcards/topic/research-llm What did R1 cost to train (Table 7)?::About 147K H800 GPU hours (R1-Zero 101K, SFT data creation 5K, R1 41K), $294K at $2 per GPU hour (the paper's rental assumption). 512 GPUs for 198 and 80 hours reproduce 101K and 41K (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-llm Where does the DeepSeek-R1 paper contradict itself?::R1 training "about 4 days, or roughly 80 hours" (4 days is 96 hours); "660B" against 671B; "1.6 epochs" for 10,400 steps of 32 questions (332,800 visits) against 80K reasoning prompts (about 4 epochs); the unusual clip ratio of 10; and "25%" and "17%" gains that are points.

#flashcards/research/concept #flashcards/topic/research-llm What limits does the R1 paper state?::Pure RL needed a large base (7B and 16B bases failed) and a reliable verifier; weak structured output and no tool use; overthinking; language mixing beyond Chinese and English; prompt sensitivity (few-shot hurts); little gain on software engineering; a moderate safety level; mostly single-turn data; failed attempts with a process reward model and MCTS.

#flashcards/research/concept #flashcards/topic/research-llm How much of the DeepSeek-R1 paper did the note read?::Version 2 of the arXiv PDF (86 pages): the main text, appendices A to C, D.2, F, G and the start of H. Not read: the example traces, the safety report, the extra analyses in E, the benchmark descriptions and the references. Version 1 was not read.
