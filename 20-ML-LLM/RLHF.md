---
tags: [RLHF, DPO, alignment, LLM, reinforcement-learning, fine-tuning]
status: growing
level: 4
tier: 4
---

# RLHF

## 🟢 1. Simple (age 12)
RLHF stands for **Reinforcement Learning from Human Feedback**. It is how an AI learns to give answers that people actually like, by having people tell it which of two answers is better.

**Example:** Imagine a kid learning to write stories. The kid writes two versions. A teacher says "I like version B more". After thousands of rounds of "A or B?", the kid learns what good stories feel like, without anyone writing the rules down.

## 🔵 2. How it works
1. **Start with a model that can follow basic instructions** (a [[Fine-tuning]] step called SFT).
2. **Collect opinions.** For many prompts, the model writes several answers. Human reviewers rank them: "this one is better than that one".
3. **Train a reward model.** It is a second AI that learns to give each answer a **score** that matches what the humans preferred.
4. **Train the main model to chase high scores.** It writes answers, the reward model scores them, and the main model is nudged toward answers that score higher.
5. **Put on a leash.** The main model is not allowed to drift too far from the original, otherwise it learns to trick the scorer with weird text instead of being truly helpful.

A newer shortcut called **DPO** skips the separate reward model and learns directly from the "A is better than B" pairs.

![[rlhf-diagram.svg]]
*The RLHF pipeline: a supervised model writes answers, human annotators rank them, the ranking trains a reward model, and the reward model guides training (with PPO) to produce the aligned model. Image: PopoDameron, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:RLHF_diagram.svg), CC BY-SA 4.0.*

Why it matters: a model trained only to predict text will happily continue anything. RLHF is a big part of what makes an [[LLM]] feel like a helpful assistant rather than an autocomplete.

## 🟠 3. Technical detail
Three-stage recipe (InstructGPT-style):
1. **SFT:** fine-tune the pre-trained model on good demonstration answers.
2. **Reward model (RM):** take the SFT model, replace the output with a single score. Train it on human **preference pairs** $(x,\,y_w,\,y_l)$: a prompt, the preferred answer $y_w$ (winner) and the rejected one $y_l$ (loser).
3. **RL:** optimise the model (the *policy*) to maximise the reward, with a KL penalty to stay near the SFT model (the objective is written out in [[LLM]]). The optimiser is usually **PPO**, a stable reinforcement-learning algorithm that limits how far each update can move the policy.

Alternatives and relatives:
- **DPO:** the same goal as RLHF but trained with a simple classification-style loss on preference pairs. No reward model, no RL loop, much simpler and cheaper.
- **RLAIF / Constitutional AI:** an AI (guided by written principles) provides the preferences instead of humans.
- **RL with verifiable rewards:** for maths and code, the reward is "did the answer check out?", no human needed.

Known problems:
- **Reward hacking:** the model finds quirks the reward model likes (long, flattering, confident-sounding answers) that humans do not really want.
- **Sycophancy:** if raters prefer agreement, the model learns to agree.
- **Annotator noise and bias:** the model learns the raters' tastes, including their mistakes.
- **Cost:** collecting good preference data is slow and expensive.

```python
import math
sigmoid = lambda z: 1 / (1 + math.exp(-z))

# reward model loss for one preference pair
r_win, r_lose = 2.0, 0.5
print(-math.log(sigmoid(r_win - r_lose)))           # 0.20  (ranks them correctly)
print(-math.log(sigmoid(r_lose - r_win)))           # 1.70  (ranks them wrongly)

# DPO loss, beta = 0.5; ratios are pi_theta / pi_ref for each answer
beta, ratio_win, ratio_lose = 0.5, 1.5, 0.8
z = beta * (math.log(ratio_win) - math.log(ratio_lose))
print(-math.log(sigmoid(z)))                        # 0.55
```

## 🔴 4. Deep / formal
**Reward model loss.** Human preferences are modelled with the Bradley–Terry model: the chance that $y_w$ beats $y_l$ depends on the gap between their reward scores. The reward model is trained to make that chance high on the labelled pairs:

$$
\mathcal{L}_{\text{RM}}(\phi)=-\,\mathbb{E}_{(x,y_w,y_l)}\Big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\Big]
$$

**Where:**
- $\phi$ (phi) is the reward model's learned numbers. $\mathcal{L}_{\text{RM}}$ is its loss, lowered by [[Gradient Descent]].
- $x$ is the prompt. $y_w$ is the answer humans **preferred** (winner) and $y_l$ the one they rejected (loser). $\mathbb{E}$ averages over many labelled pairs.
- $r_\phi(x,y)$ is the **reward score** the model gives to answer $y$ for prompt $x$: one number, higher means better.
- $r_\phi(x,y_w)-r_\phi(x,y_l)$ is the score gap. **Matters:** only the *difference* is used, so the absolute scale of the scores does not matter. The model just has to rank the winner above the loser.
- $\sigma(z)=\dfrac{1}{1+e^{-z}}$ is the **sigmoid**: it squashes any number into 0 to 1. **Matters:** $\sigma(\text{gap})$ reads as "the probability the model thinks the winner is better". A big positive gap gives near 1, a gap of 0 gives $0.5$ (a coin flip), a negative gap gives near 0.
- $-\log$ turns that probability into a loss: probability near 1 gives loss near 0, a small probability gives a big loss.

**Example (tiny numbers):** the humans preferred $y_w$. The reward model scores $r(y_w)=2.0$ and $r(y_l)=0.5$.
- Gap $=2.0-0.5=1.5$. $\sigma(1.5)=\mathbf{0.82}$: the model is 82% sure the winner is better (correct).
- Loss $=-\log(0.82)=\mathbf{0.20}$ (small).
- If instead it had scored them backwards, $r(y_w)=0.5$ and $r(y_l)=2.0$: gap $=-1.5$, $\sigma=0.18$, loss $=\mathbf{1.70}$ (large). Training pushes the scores to the correct order.

**DPO loss.** DPO (Rafailov et al., 2023) shows the RLHF objective (reward plus KL leash, see [[LLM]]) has a closed-form optimal policy, so the reward can be written in terms of the policy itself. Training then needs only preference pairs:

$$
\mathcal{L}_{\text{DPO}}(\theta)=-\,\mathbb{E}\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]
$$

**Where:**
- $\pi_\theta$ is the model being trained (its numbers are $\theta$). $\pi_{\text{ref}}$ is a **frozen copy** of the starting model (the reference).
- $\pi(y\mid x)$ is the probability that model gives to the full answer $y$ for prompt $x$. $y_w$ and $y_l$ are the preferred and rejected answers.
- $\dfrac{\pi_\theta(y)}{\pi_{\text{ref}}(y)}$ is a **ratio**: how much more (above 1) or less (below 1) likely the trained model makes this answer compared with the reference. $\log$ of the ratio is positive if the answer became more likely, negative if less. **Matters:** the reference is the anchor, so the model is judged on how it *changed*, which keeps it from drifting wildly.
- Subtracting the loser's log-ratio from the winner's log-ratio gives the **margin**: positive means "I raised the winner and/or lowered the loser", which is what we want.
- $\beta$ is a strength setting. **Matters:** a larger $\beta$ amplifies the margin before the sigmoid (and keeps the model tighter to the reference); the role is the same as the leash strength in the RLHF objective.
- $\sigma$ and $-\log$ work as in the reward model loss: a positive margin gives a small loss, a negative margin gives a large loss.

**Example (tiny numbers):** $\beta=0.5$. After some training the model makes the winner $1.5\times$ more likely than the reference did, and the loser $0.8\times$ as likely.
- $\log(1.5)=0.405$ and $\log(0.8)=-0.223$. Margin $=0.405-(-0.223)=0.628$.
- Multiply by $\beta$: $z=0.5\cdot0.628=\mathbf{0.314}$. $\sigma(0.314)=0.58$. Loss $=-\log(0.58)=\mathbf{0.55}$.
- If the model had gone the wrong way (winner $0.8\times$, loser $1.5\times$): $z=-0.314$, $\sigma=0.42$, loss $=\mathbf{0.86}$, higher, so training pushes it back.
- The loss keeps falling as the winner's ratio goes up and the loser's goes down, with no reward model needed.

**Further notes:**
- **PPO** adds a *clipped* update rule so that one batch of feedback cannot change the policy too much (its abstract, checked, says it allows several epochs of minibatch updates on the same data and is simpler than TRPO). That it needs a value network to reduce variance, and that this makes classic RLHF heavy to run (several large models in memory at once, see [[GPU]]), is standard textbook knowledge written from memory: I did not find it in the abstracts.
- **Reward over-optimisation:** as the policy chases the reward model, true quality can get worse. Gao et al. (2022, arXiv:2210.10760) say in their abstract that optimising a reward model too much can hinder ground-truth performance, and they study how this depends on the method and the KL penalty. The "first rises, then falls" shape and the claim that the KL leash and early stopping are the standard defences are from memory: the abstract does not describe the curve, and I did not open the full paper.
- Active areas: preference data from AI feedback, process rewards (grading each reasoning step), and online variants of DPO.

**What came before, and how this improved on it:**
- **Before:** supervised fine-tuning on human-written demonstrations.
- **Its limit:** "Making language models bigger does not inherently make them better at following a user's intent" (InstructGPT abstract). Models could be untruthful, toxic or unhelpful.
- **What changed:** collect human rankings of outputs, train a reward model, and optimise the language model with reinforcement learning (Ouyang et al., arXiv 2203.02155, 4 March 2022, abstract read; first author from memory).
- **How much better:** human raters preferred outputs of the 1.3B InstructGPT model to those of the 175B GPT-3, which has about 135 times more parameters (abstract). The code below is a different, smaller illustration of what a reward signal buys: if each answer's score is a standard bell curve, the best of 8 answers scores 1.43 on average against 0.00 for one answer. That is best-of-n choosing, not RLHF itself.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
print("expected reward of the best of n answers, when each answer's score is a standard bell curve:")
for n in (1, 2, 4, 8, 16, 64):
    print(f"  n={n:3d}: {rng.standard_normal((20000, n)).max(1).mean():.2f}")
print("InstructGPT: 1.3B model preferred to 175B GPT-3 by human raters (abstract); size ratio", round(175 / 1.3), "x")
```

## Related
- [[LLM]] · [[Fine-tuning]] · [[Gradient Descent]] · [[Transformer]] · [[GPU]] · [[Matrix]] · [[KL Divergence]] · [[Pre-training]] · [[Q-Learning]] · [[Policy Gradient]] · [[PPO]] · [[Direct Preference Optimization]] · [[GRPO]] · [[Reward Model]] · [[Bradley-Terry Model]] · [[Instruction Tuning]] · [[InstructGPT 2022]]

## Sources
- Christiano et al., *Deep Reinforcement Learning from Human Preferences*, 2017, arXiv:1706.03741 (checked: authors, date, abstract; human feedback on under 1% of interactions)
- Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT), 2022, arXiv:2203.02155 (checked: authors, date, abstract; the 1.3B model was preferred to the 175B GPT-3. The abstract names SFT then RLHF; the separate reward-model stage is from memory of the paper)
- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017, arXiv:1707.06347 (checked: abstract)
- Rafailov et al., *Direct Preference Optimization*, 2023, arXiv:2305.18290 (checked: abstract; closed-form reparameterisation, no sampling during fine-tuning)
- Bai et al., *Constitutional AI*, 2022, arXiv:2212.08073 (checked: abstract; AI judgments train the preference model)
- Gao et al., *Scaling Laws for Reward Model Overoptimization*, 2022, arXiv:2210.10760 (abstract only)
- The Bradley–Terry model, used for the reward-model loss, was not looked up here.
- Ouyang et al., InstructGPT, arXiv 2203.02155 (abstract read for the history block)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/alignment-rl RLHF (reinforcement learning from human feedback):::How an AI learns to give answers people like, by having people say which of two answers is better.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the reward model loss and explain each symbol.
?
$$\mathcal{L}_{\text{RM}}(\phi)=-\,\mathbb{E}_{(x,y_w,y_l)}\Big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\Big]$$
- $\phi$: the reward model's learned numbers. $\mathcal{L}_{\text{RM}}$: its loss.
- $x$: the prompt. $y_w$: the answer humans preferred (winner). $y_l$: the one they rejected (loser). $\mathbb{E}$ averages over many labelled pairs.
- $r_\phi(x,y)$: the score the reward model gives to answer $y$.
- $\sigma$: the sigmoid. The loss is small when the winner's score is above the loser's, and large when it is below.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the DPO loss and explain each symbol.
?
$$\mathcal{L}_{\text{DPO}}(\theta)=-\,\mathbb{E}\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]$$
- $\pi_\theta$: the model being trained. $\pi_{\text{ref}}$: a frozen copy of the starting model (the reference).
- $\pi(y\mid x)$: the probability that model gives to the full answer $y$ for prompt $x$. $y_w$, $y_l$: the preferred and rejected answers.
- Each log ratio says how much more likely the trained model makes an answer compared with the reference.
- $\beta$: strength of the tie to the reference. The loss rewards raising the winner and lowering the loser, with no separate reward model.

#flashcards/ml/cloze #flashcards/topic/alignment-rl In RLHF, a ==reward model== is trained from human ==preferences==, and then guides training of the language model.

#flashcards/ml/example #flashcards/topic/alignment-rl The reward model gives $r(y_w)=2.0$ and $r(y_l)=0.5$. What is the loss?::Gap 1.5. σ(1.5) = 0.82, so the model is 82% sure the winner is better (correct). Loss = −log 0.82 = 0.20 (small).

#flashcards/ml/example #flashcards/topic/alignment-rl DPO with $\beta=0.5$: the winner is 1.5× more likely than under the reference, the loser 0.8×. What is the sigmoid input?::log 1.5 = 0.405, log 0.8 = −0.223. Margin 0.628. Times β: 0.314. σ(0.314) = 0.58.

#flashcards/ml/image #flashcards/topic/alignment-rl ![[rlhf-diagram.svg]]
What does this picture show?
?
The RLHF pipeline: a supervised model writes answers, human annotators rank them, the ranking trains a reward model, and the reward model guides training (with PPO) to produce the aligned model.
