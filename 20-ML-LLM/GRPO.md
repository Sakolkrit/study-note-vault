---
tags: [LLM, reinforcement-learning, GRPO, reasoning, PPO, alignment, DeepSeek]
status: growing
level: 4
tier: 3
---

# GRPO

**Group Relative Policy Optimization.** A reinforcement-learning method that grades each answer by comparing it with the other answers to the same question.

## 🟢 1. Simple (age 12)
Imagine a teacher asks a student the same maths question 8 times, and the student gives 8 different answers. Instead of having a separate expert say how good each answer *should* score, the teacher just compares them to each other: "the answers that were better than the **class average** deserve a boost, and the ones that were worse should be discouraged". That is GRPO. The group of answers is its own yardstick.

**Example:** You take 8 shots at a basketball hoop and 3 go in. You do not need a coach to say what a "good" score is. The 3 that went in are better than your average, so you repeat what you did on those, and you avoid what you did on the 5 misses.

## 🔵 2. How it works
1. Start with a model that already writes answers (see [[Pre-training]] and [[Fine-tuning]]). Give it a question that has a **checkable** answer, such as a maths problem or a coding task (see the [[Reasoning Model]] note).
2. Ask for a **group** of answers to the same question, say 8.
3. Score each answer with a simple rule: 1 if it is right, 0 if not. (Some versions also add small points for correct formatting.)
4. Compute each answer's **advantage**: how much its score is above or below the **average of its group**, scaled by the group's spread.
5. Update the model ([[Policy Gradient]]) so that answers with a positive advantage become more likely and those with a negative advantage become less likely.
6. Keep the update small: the model is stopped from moving too far from where it was ("clipping", as in [[PPO]]) and, in the original version, from its reference model (a penalty on the difference, see [[KL Divergence]]).
7. If every answer in the group is right (or every one is wrong), all advantages are 0 and that question teaches nothing this round.
8. The key difference from [[PPO]]: **no separate value model** to estimate a baseline. The group average plays that role.

## 🟠 3. Technical detail
- **Where it comes from:** introduced in the DeepSeekMath paper (Shao et al., 2024) as a variant of PPO that also reduces memory use (arXiv 2402.03300 abstract, read). The lecture then shows it as the algorithm behind DeepSeek's R1 reasoning models.
- **What it removes:** PPO trains a value model that estimates, for each token, the reward to expect. GRPO replaces it with the mean reward of the group, so one fewer big model is trained and kept in memory.
- **What it keeps:** the probability ratio between new and old policy, the clipping of that ratio, and a penalty against a reference model. The lecture puts the differences as: the KL penalty is added to the loss (not folded into the reward) and the advantage comes from the group (not from a learned value function).
- **Rewards in reasoning training:** a check that the thinking is inside the expected tags, and a check that the final answer is correct, by running test cases for code or comparing with the known answer for maths. The lecture's example is the "R1-Zero" run, which used only these rules.
- **Known side effect (from the lecture):** with the original averaging, the response length tends to keep growing during training. Dividing by the answer's own length makes a long wrong answer's tokens each count for less, so long wrong answers are punished weakly per token. The lecture names two fixes: **DAPO** and **Dr. GRPO**, which equalise the contribution of every token. I did not read those papers, so the exact changes are not described here.
- **Other adjustments the lecture lists:** handling the bias linked to question difficulty, and encouraging diverse answers.
- **Do not confuse it with:** [[Direct Preference Optimization]], which learns from pairs with a supervised loss and needs no sampling. GRPO samples answers during training.

```python
import numpy as np
# 1) group-relative advantages: 8 answers to the same question, reward 1 if correct else 0
r = np.array([1, 0, 0, 1, 0, 0, 0, 1], float)
adv = (r - r.mean()) / r.std()
print('rewards   :', r.tolist(), '| mean', r.mean(), '| std', round(r.std(), 4))
print('advantages:', adv.round(3).tolist(), '| sum', round(float(adv.sum()), 12))
# 2) if every answer is right (or every one wrong) the group tells us nothing
for rr in ([1] * 8, [0] * 8):
    rr = np.array(rr, float); print('all', int(rr[0]), '-> advantages', ((rr - rr.mean()) / (rr.std() + 1e-8)).tolist())
# 3) clipped objective for one token: min(ratio*A, clip(ratio, 1-eps, 1+eps)*A)
eps = 0.2
def clipped(ratio, A): return min(ratio * A, np.clip(ratio, 1 - eps, 1 + eps) * A)
for ratio, A in ((1.0, 1.0), (1.5, 1.0), (0.5, 1.0), (1.5, -1.0), (0.5, -1.0)):
    print(f'ratio {ratio} advantage {A:+.0f} -> objective {clipped(ratio, A):+.2f}')
# 4) the length effect: average over each answer's own tokens, then over answers
A = -1.0                                                     # two wrong answers, same advantage
short, long_ = 10, 1000                                      # their lengths in tokens
print('per-token weight of a wrong answer, mean-per-answer: short', A / short, '| long', A / long_)
print('per-token weight with one shared length divisor     : short', A / 505, '| long', A / 505)
# 5) which models must sit in memory
print('PPO needs: policy, reference, reward model, value model =', 4, '| GRPO with a rule-based reward needs: policy, reference =', 2)
```

## 🔴 4. Deep / formal
**The group advantage.** For one question, sample $G$ answers and get rewards $r_1,\dots,r_G$. The advantage of answer $i$ is:

$$
\hat{A}_i=\frac{r_i-\operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)}
$$

**Where:**
- $r_i$ is the reward of answer $i$ (for example 1 if correct and 0 if wrong) and $G$ is the group size.
- $\operatorname{mean}$ is the average reward of the group, which plays the part of PPO's learned baseline.
- $\operatorname{std}$ is the standard deviation (spread) of the group's rewards. (The code uses the population version, dividing by $G$, which is a choice; implementations may differ, and I did not check which the papers use.)
- $\hat A_i$ is how many spreads answer $i$ is above ($+$) or below ($-$) its group.
- **Matters:** subtracting the mean means that only *relative* quality counts, and the advantages of a group add up to 0. Dividing by the spread makes the size of the push independent of how big the rewards are. If all rewards are equal, the spread is 0 and there is nothing to learn from. (Code adds a tiny number to the bottom to avoid dividing by zero.) Dr. GRPO, per the lecture, questions the spread division; I did not read why.

**Example (tiny numbers):** 8 answers, rewards $1,0,0,1,0,0,0,1$.
- Mean: $3/8=0.375$. Spread: $\sqrt{(3\cdot0.625^2+5\cdot0.375^2)/8}=0.4841$.
- Correct answers: $(1-0.375)/0.4841=\mathbf{+1.291}$ each.
- Wrong answers: $(0-0.375)/0.4841=\mathbf{-0.775}$ each.
- Check: $3(1.291)+5(-0.775)\approx0$.
- Meaning: the 3 correct answers get a boost and the 5 wrong ones get pushed down, with the rare successes boosted more than the common failures are punished.

**The clipped objective for one token.** With the probability ratio $\rho=\pi_\theta/\pi_{\text{old}}$ (new model over the model that produced the answer), the quantity to maximise is:

$$
\min\!\big(\rho\,\hat A,\ \operatorname{clip}(\rho,\,1-\varepsilon,\,1+\varepsilon)\,\hat A\big)
$$

**Where:**
- $\rho$ is how much more or less likely the new model finds the token than the old model did.
- $\hat A$ is the advantage from above.
- $\varepsilon$ is a small number such as $0.2$, and $\operatorname{clip}(\rho,1-\varepsilon,1+\varepsilon)$ forces $\rho$ into the range $[0.8,1.2]$.
- $\min$ takes the smaller of the two, which makes the objective pessimistic.
- **Matters:** when $\hat A>0$ the objective stops rewarding an increase of $\rho$ beyond $1+\varepsilon$, and when $\hat A<0$ it stops rewarding a decrease below $1-\varepsilon$, so one update cannot move the model far. (The full objective also has the reference-model penalty and the average over tokens and answers, which I left out.) This is the same clip as in [[PPO]].

**Example (tiny numbers):** $\varepsilon=0.2$.
- $\hat A=+1$, $\rho=1.5$: $\min(1.5,\ 1.2)=\mathbf{1.2}$. The extra increase is not rewarded.
- $\hat A=+1$, $\rho=0.5$: $\min(0.5,\ 0.8)=\mathbf{0.5}$. A drop is fully penalised.
- $\hat A=-1$, $\rho=0.5$: $\min(-0.5,\ -0.8)=\mathbf{-0.8}$.
- $\hat A=-1$, $\rho=1.5$: $\min(-1.5,\ -1.2)=\mathbf{-1.5}$.
- Meaning: the model gets no extra credit for pushing a good token past $1.2$ times its old probability, but a move in the wrong direction is not cushioned.

**Further notes:**
- **Memory:** PPO needs the policy, the reference, the reward model and the value model (4 models). GRPO with a rule-based reward needs only the policy and the reference (2). The code states this as a count only; I did not measure memory.
- **Length effect in numbers:** with an advantage of $-1$ and averaging over each answer's own tokens, a wrong 10-token answer has a weight of $-0.1$ per token, and a wrong 1,000-token answer has $-0.001$ per token. Sharing one length divisor (for example $505$) gives both $-0.00198$ per token. That is the problem that the lecture says DAPO and Dr. GRPO fix.
- **Limits of the example:** a toy of rewards only. No model is trained and no reasoning appears.
- **Not covered:** the full R1 recipe stages (the lecture describes R1-Zero with GRPO only, and R1 with extra supervised stages), and the math of why a group mean is a valid baseline.

**What came before, and how this improved on it:**
- **Before:** [[PPO]] for RLHF and reasoning, where a separate value model estimates a baseline for every token (the lecture, from slides).
- **Its limit:** the extra model costs memory and training effort, and the lecture counts four models in total for PPO-based RLHF.
- **What changed:** use the mean (and spread) of a group of sampled answers as the baseline, so there is no value model (Shao et al., arXiv 2402.03300, 5 February 2024, abstract read: "a variant of PPO that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO").
- **How much better:** in this note's code the number of models held in memory falls from 4 to 2 for a rule-based reward. The abstract reports DeepSeekMath 7B at 51.7% on the MATH benchmark without tools or voting, and 60.9% with self-consistency over 64 samples, but attributes this to the data and GRPO together, so it is not a measure of GRPO alone. I did not measure memory or accuracy.

## Related
- [[PPO]] · [[Policy Gradient]] · [[Reinforcement Learning]] · [[RLHF]] · [[Direct Preference Optimization]] · [[Reasoning Model]] · [[Chain of Thought]] · [[KL Divergence]] · [[LLM]] · [[Fine-tuning]] · [[Reward Model]] · [[Pass at k]] · [[DeepSeek-R1 2025]]

## Sources
- Stanford CME 295 (Fall 2025), Lecture 6 slides, Afshine and Shervine Amidi
- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, 2024 (arXiv 2402.03300, abstract read)
- DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, 2025 (arXiv 2501.12948, abstract read)
- Yu et al., *DAPO*, 2025, and Liu et al., *Understanding R1-Zero-Like Training*, 2025 (named in the lecture; not read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/alignment-rl GRPO (group relative policy optimization):::Ask the same question many times, then compare the answers to each other. Answers better than the group average get a boost, worse ones get pushed down. No separate learned judge is needed.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the GRPO advantage and explain each symbol.
?
$$\hat{A}_i=\frac{r_i-\operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)}$$
- $r_i$: the reward of answer $i$ (for example 1 if correct, 0 if wrong). $G$: the group size.
- $\operatorname{mean}$: the average reward of the group. It plays the part of PPO's learned baseline.
- $\operatorname{std}$: the standard deviation (spread) of the group's rewards. Dividing by it puts the scores on a common scale.
- $\hat A_i$: the advantage. Positive means better than the group, negative means worse.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the clipped objective and explain each symbol.
?
$$\min\!\big(\rho\,\hat A,\ \operatorname{clip}(\rho,\,1-\varepsilon,\,1+\varepsilon)\,\hat A\big)$$
- $\rho$: how much more or less likely the new model finds the token than the old model did.
- $\hat A$: the advantage from above.
- $\varepsilon$: a small number such as 0.2. $\operatorname{clip}(\rho,1-\varepsilon,1+\varepsilon)$ forces $\rho$ into $[0.8,1.2]$.
- $\min$: takes the smaller of the two, so a step cannot gain extra credit for moving the model too far in one update.

#flashcards/ml/cloze #flashcards/topic/alignment-rl GRPO's baseline is the ==group average== of rewards instead of a separately learned ==value model==.

#flashcards/ml/example #flashcards/topic/alignment-rl 8 answers get rewards 1, 0, 0, 1, 0, 0, 0, 1. What are the advantages?::Mean 3/8 = 0.375, spread 0.4841. Correct answers: (1 − 0.375)/0.4841 = +1.291. Wrong answers: −0.775. They add up to about 0.

#flashcards/ml/example #flashcards/topic/alignment-rl With $\varepsilon=0.2$, what does the clipped objective give for ($\hat A=+1$, $\rho=1.5$), ($+1$, $0.5$), ($-1$, $0.5$) and ($-1$, $1.5$)?::min(1.5, 1.2) = 1.2: no extra reward for going further. min(0.5, 0.8) = 0.5: a drop is fully penalised. min(−0.5, −0.8) = −0.8. min(−1.5, −1.2) = −1.5.
