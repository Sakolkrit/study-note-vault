---
tags: [LLM, alignment, DPO, preference-tuning, RLHF, fine-tuning]
status: growing
level: 4
tier: 3
---

# Direct Preference Optimization

Usually written **DPO**. It teaches a model which answers people prefer, using plain supervised training and no reinforcement-learning loop.

## 🟢 1. Simple (age 12)
Suppose you show a chatbot two answers to the same question and tell it "**this one is better than that one**". DPO nudges the chatbot to make the better answer **more likely** and the worse answer **less likely**, while making sure it does not wander too far from how it behaved before. It is like a teacher marking two essays and the student adjusting, without needing a separate judge to hand out scores.

**Example:** A teddy-bear shop assistant is asked for a new activity. Answer A says "watch a movie together" and answer B says "don't spend time with your bear". People pick A. After DPO training, the assistant is more likely to say things like A.

## 🔵 2. How it works
1. First the model is trained by [[Fine-tuning]] on good examples, and also by [[Pre-training]] before that. That version is the **reference model**. It is frozen and kept.
2. Collect **preference pairs**: for a prompt, a **better** answer and a **worse** answer, labelled by people (or by an AI judge, or a cheap proxy).
3. In [[RLHF]], the next steps would be (a) train a separate [[Reward Model]] on those pairs and (b) run a reinforcement-learning loop such as [[PPO]] against that reward model. That needs a reward model, a value model, a reference model and the policy: four models in memory, and a training loop that is touchy.
4. **DPO skips both steps.** It uses a clever rewriting of the RLHF goal so that the preference pairs can be used **directly** as a classification-style loss on the model being trained.
5. For each pair, DPO asks: "has the model become *relatively* more likely to say the good answer than the reference model was, compared with the bad answer?" If yes, the loss is small. If not, the loss is big and training pushes the model that way.
6. A setting $\beta$ controls how strongly the model is held back near the reference. It is the "don't wander" knob.
7. Only the model being trained and the frozen reference are needed. The lecture notes that people still debate which is better, PPO-based RLHF or DPO. It says there is no common agreement, and it depends on the task and on how well it is implemented.

## 🟠 3. Technical detail
- **Data:** pairs $(x,y_w,y_l)$: a prompt $x$, a preferred ("winning") answer $y_w$ and a rejected ("losing") answer $y_l$.
- **What you compute for each pair:** the log-probability that the current model and the reference model give to each of the two whole answers, so four log-probabilities. Then the margin (level 4) and the loss.
- **No sampling during training:** the answers are already in the data, so there is no need to generate text while training. That is what makes it cheap and stable compared with RL.
- **The role of $\beta$:** a small $\beta$ lets the model move far from the reference. A large $\beta$ keeps it close.
- **Hidden reward:** the model's own log-ratio to the reference acts as a reward, so the paper's title says the language model is "secretly a reward model". The code checks this on a toy.
- **Where it comes from, as the lecture explains it:** (1) start from the RLHF goal (maximise the reward minus a penalty for drifting from the reference); (2) solve for the best policy in closed form; (3) turn that around to write the reward in terms of the policy; (4) put that reward into the [[Bradley-Terry Model]] of preferences. The reward-model step disappears.
- **Known behaviour:** because the loss keeps rewarding a bigger margin, training on pairs that always agree pushes the probabilities of the two answers further apart without limit. The code shows this: the model ends up very close to certain at 4,000 steps whatever the $\beta$ in $\{0.1,1\}$.
- **Cost and ease:** one supervised-style stage; fewer models and fewer settings than [[PPO]].

```python
import numpy as np
sig = lambda z: 1 / (1 + np.exp(-z))
softmax = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()

# 1) the loss for one pair: -log sigmoid( beta*[log(pi/pi_ref)(good) - log(pi/pi_ref)(bad)] )
beta = 0.5
def dpo_loss(pi_w, pi_l, ref_w, ref_l):
    margin = beta * (np.log(pi_w / ref_w) - np.log(pi_l / ref_l))
    return -np.log(sig(margin)), margin
for name, args in (('policy same as reference', (0.2, 0.2, 0.2, 0.2)), ('good answer more likely', (0.4, 0.1, 0.2, 0.2)), ('bad answer more likely', (0.1, 0.4, 0.2, 0.2))):
    l, m = dpo_loss(*args); print(f'{name:26s} margin {m:+.3f}  loss {l:.3f}')

# 2) closed form of the best policy for 'maximise reward - beta*KL(policy || reference)':  pi*(y) = ref(y) * exp(r(y)/beta) / Z
ref = np.array([0.5, 0.3, 0.2]); r = np.array([0.0, 1.0, 2.0])
for beta in (0.1, 1.0, 10.0):
    star = ref * np.exp(r / beta); star /= star.sum()
    z = np.log(ref).copy()                                   # numerical check: gradient ascent on  E[r] - beta*KL  over softmax logits
    for _ in range(40000):
        p = softmax(z); obj_grad = p * (r - beta * (np.log(p / ref))) ; z += 0.05 * (obj_grad - p * obj_grad.sum())
    print('beta', beta, '| closed form', star.round(4).tolist(), '| numerical', softmax(z).round(4).tolist(), '| match', np.allclose(star, softmax(z), atol=1e-3))

# 3) the 'secret reward': beta*log(pi*/ref) = r - beta*log Z, so reward differences are recovered from the two policies
beta = 1.0; star = ref * np.exp(r / beta); star /= star.sum()
imp = beta * np.log(star / ref)
print('true reward differences  :', (r - r[0]).tolist(), '| implied by the policy:', (imp - imp[0]).round(6).tolist())

# 4) train a 3-answer policy with the DPO loss on two pairs (answer 2 beats 1, answer 1 beats 0) and watch it leave the reference
def train(beta, steps, lr=0.05):
    z = np.log(ref).copy()
    for _ in range(steps):
        p = softmax(z); g = np.zeros(3)
        for w, l in ((2, 1), (1, 0)):
            m = beta * (np.log(p[w] / ref[w]) - np.log(p[l] / ref[l]))
            g += -(1 - sig(m)) * beta * (np.eye(3)[w] - np.eye(3)[l])      # log p_w - log p_l = z_w - z_l, so the gradient is simple
        z -= lr * g / 2
    return softmax(z)
kl = lambda p: float((p[p > 0] * np.log(p[p > 0] / ref[p > 0])).sum())
for b in (0.1, 1.0, 5.0):
    for steps in (300, 4000):
        p = train(b, steps); print('beta', b, 'steps', steps, '->', p.round(3).tolist(), '| KL from reference', round(kl(p), 3))
```

## 🔴 4. Deep / formal
**The DPO loss.** For one pair $(x,y_w,y_l)$:

$$
\mathcal{L}_{\text{DPO}}=-\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)
$$

**Where:**
- $\pi_\theta(y\mid x)$ is the probability the model being trained gives to the whole answer $y$ for prompt $x$, and $\pi_{\text{ref}}$ is the same for the frozen reference model.
- $y_w$ is the preferred answer and $y_l$ the rejected one.
- $\log(\pi_\theta/\pi_{\text{ref}})$ is how much more (or less) likely the answer became compared with the reference.
- $\beta>0$ scales the margin. $\sigma(z)=1/(1+e^{-z})$ squashes a number into $(0,1)$ (the sigmoid).
- The minus log makes the loss large when $\sigma$ is near 0 and near 0 when $\sigma$ is near 1.
- **Matters:** the loss only cares about the *difference* between the two answers' changes. Raising both answers' probabilities equally does nothing. Without the reference, the model could win by simply making all answers extreme. $\beta$ sets how far a given change in the log-ratio counts.

**Example (tiny numbers):** $\beta=0.5$, reference probabilities $0.2$ for both answers.
- Policy unchanged ($0.2$ and $0.2$): the margin is $0.5(\ln1-\ln1)=0$, $\sigma(0)=0.5$, loss $=-\ln0.5=\mathbf{0.693}$.
- Good answer up to $0.4$, bad answer down to $0.1$: the margin is $0.5(\ln2-\ln0.5)=0.5\times1.386=0.693$, $\sigma(0.693)=0.667$, loss $=\mathbf{0.405}$.
- Good answer down to $0.1$, bad answer up to $0.4$: margin $-0.693$, loss $=\mathbf{1.099}$.
- Meaning: the loss falls when the model moves toward the preferred answer, and rises when it moves the wrong way.

**The best policy for the RLHF goal.** DPO starts from the goal "high reward without drifting from the reference". Its best solution has a closed form:

$$
\pi^{*}(y\mid x)=\frac{\pi_{\text{ref}}(y\mid x)\,e^{\,r(x,y)/\beta}}{Z(x)}
$$

**Where:**
- $r(x,y)$ is the reward of answer $y$ for prompt $x$, and $\beta$ is the same strength of the leash as before.
- $\pi_{\text{ref}}$ is the starting distribution, and $e^{r/\beta}$ is a boost that grows with the reward.
- $Z(x)=\sum_y\pi_{\text{ref}}(y\mid x)e^{r(x,y)/\beta}$ is the number that makes the probabilities add up to 1.
- $\pi^*$ is the best policy.
- **Matters:** a small $\beta$ makes the boost steep, so almost all the probability goes to the highest-reward answer. A large $\beta$ makes the boost flat, so $\pi^*$ stays near the reference. Reading the formula backwards gives $r=\beta\log(\pi^*/\pi_{\text{ref}})+\beta\log Z$: the reward can be written with the policy, which is the step DPO uses.

**Example (tiny numbers):** three answers with reference probabilities $0.5,0.3,0.2$, rewards $0,1,2$ and $\beta=1$.
- Boosted: $0.5\,e^{0}=0.5$, $0.3\,e^{1}=0.815$, $0.2\,e^{2}=1.478$. Their sum is $Z=\mathbf{2.793}$.
- Divide: $\pi^*=\mathbf{[0.179,\ 0.292,\ 0.529]}$.
- Meaning: the highest-reward answer went from $0.2$ to $0.529$. The code also reached the same numbers by gradient ascent on the original goal (reward minus $\beta\times$ KL), for $\beta=0.1$, $1$ and $10$.

**Preferences from rewards.** The [[Bradley-Terry Model]] says the chance that $y_w$ beats $y_l$ depends only on the reward gap:

$$
P(y_w\succ y_l)=\sigma\big(r(x,y_w)-r(x,y_l)\big)
$$

**Where:**
- $P(y_w\succ y_l)$ is the probability that a person prefers $y_w$ over $y_l$.
- $r(x,y_w)-r(x,y_l)$ is the reward gap. $\sigma$ is the sigmoid.
- **Matters:** a gap of 0 gives a 50% chance, and bigger gaps push the chance toward 100%. DPO puts $r=\beta\log(\pi_\theta/\pi_{\text{ref}})$ here, and the unknown $Z$ cancels in the difference, which is why no reward model is needed.

**Example (tiny numbers):** a reward gap of 1.
- $\sigma(1)=1/(1+e^{-1})=1/(1+0.368)=\mathbf{0.731}$.
- Meaning: an answer with one more unit of reward is preferred about 73% of the time. With a gap of 2 it is $0.881$.

**Further notes:**
- **Same implied reward, no model:** the code recovers the reward gaps $0,1,2$ exactly from the best policy and the reference ($\beta\log(\pi^*/\pi_{\text{ref}})$), as the maths says.
- **Training behaviour in the toy:** with two consistent pairs, after 4,000 steps the model was $0.982$ (for $\beta=0.1$) and $0.984$ (for $\beta=1$) certain of the top answer (KL from reference about $1.5$), and only $0.703$ for $\beta=5$ (KL $0.67$). After 300 steps the leash looked different (KL $0.046$, $0.859$, $0.349$ for $\beta=0.1,1,5$), so the effect of $\beta$ depends on how long you train. I did not tune or test this on real models.
- **Not covered:** variants of DPO, how the preference pairs are best made, and the comparison with PPO-based RLHF on real tasks, where the lecture says there is no consensus.

**What came before, and how this improved on it:**
- **Before:** [[RLHF]] with a separate [[Reward Model]] and a reinforcement-learning stage such as [[PPO]] (Ouyang et al. 2022, named in the lecture; I did not read it).
- **Its limit:** the DPO abstract calls RLHF "a complex and often unstable procedure": first fit a reward model, then fine-tune with RL while staying close to the original model. The lecture counts four models for PPO (policy, value, reward, reference) and many settings.
- **What changed:** a new parameterisation of the reward so that the best policy has a closed form, which turns the problem into a simple classification loss (Rafailov et al., arXiv 2305.18290, 29 May 2023, abstract read).
- **How much better:** the abstract says DPO is stable, performant and lightweight, needs no sampling during fine-tuning, and matches or beats PPO-based RLHF on the tasks tested (sentiment control, summarisation, single-turn dialogue). The lecture's own conclusion is that there is no common agreement across tasks. In this note's code, the needed models drop from 4 to 2 (the policy and its frozen reference), the closed form matched a numerical optimum for three values of $\beta$, and the toy shows the drift behaviour above. I did not measure quality or compute cost.

## Related
- [[RLHF]] · [[PPO]] · [[Reward Model]] · [[Bradley-Terry Model]] · [[Fine-tuning]] · [[Reinforcement Learning]] · [[Policy Gradient]] · [[KL Divergence]] · [[LLM]] · [[Softmax]] · [[GRPO]] · [[Pre-training]] · [[Probability]] · [[DPO 2023]]

## Sources
- Stanford CME 295 (Fall 2025), Lecture 5 slides, Afshine and Shervine Amidi
- Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, 2023 (arXiv 2305.18290, abstract read)
- Bradley and Terry, *The rank analysis of incomplete block designs: I. The method of paired comparisons*, 1952 (named in the lecture; not read)
- Xu et al., *Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study*, 2024 (named in the lecture; not read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/alignment-rl DPO (direct preference optimization):::Show a chatbot two answers and say which is better. DPO makes the better answer more likely and the worse one less likely, while keeping the model from wandering too far from how it behaved before. No separate reward model is needed.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the DPO loss and explain each symbol.
?
$$\mathcal{L}_{\text{DPO}}=-\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)$$
- $\pi_\theta(y\mid x)$: the probability the model being trained gives to the whole answer $y$ for prompt $x$. $\pi_{\text{ref}}$: the same for the frozen reference model.
- $y_w$: the preferred answer. $y_l$: the rejected one.
- $\log(\pi_\theta/\pi_{\text{ref}})$: how much more (or less) likely the answer became compared with the reference.
- $\beta$: the strength of the leash to the reference. The loss rewards raising the winner's ratio above the loser's.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the best policy for a reward with a leash, and explain each symbol.
?
$$\pi^{*}(y\mid x)=\frac{\pi_{\text{ref}}(y\mid x)\,e^{\,r(x,y)/\beta}}{Z(x)}$$
- $r(x,y)$: the reward of answer $y$ for prompt $x$. $\beta$: the strength of the leash.
- $\pi_{\text{ref}}$: the starting distribution. $e^{r/\beta}$: a boost that grows with the reward.
- $Z(x)=\sum_y\pi_{\text{ref}}(y\mid x)e^{r(x,y)/\beta}$: the number that makes the probabilities add up to 1.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the Bradley-Terry preference probability and explain each symbol.
?
$$P(y_w\succ y_l)=\sigma\big(r(x,y_w)-r(x,y_l)\big)$$
- $P(y_w\succ y_l)$: the probability that a person prefers $y_w$ over $y_l$.
- $r(x,y_w)-r(x,y_l)$: the reward gap. $\sigma$: the sigmoid.
- A gap of 0 gives 50%, and bigger gaps push the chance toward 100%.
- DPO puts $r=\beta\log(\pi_\theta/\pi_{\text{ref}})$ here, and the unknown $Z$ cancels in the difference, which is why no reward model is needed.

#flashcards/ml/cloze #flashcards/topic/alignment-rl DPO uses the policy-to-==reference== ratio, so the model cannot drift far from where it ==started==.

#flashcards/ml/example #flashcards/topic/alignment-rl $\beta=0.5$, reference probability 0.2 for both answers. What is the loss if the policy is unchanged, and if the good answer goes to 0.4 and the bad one to 0.1?::Unchanged: margin 0, σ(0) = 0.5, loss −ln 0.5 = 0.693. Good 0.4, bad 0.1: margin 0.5 × (ln 2 − ln 0.5) = 0.693, σ = 0.667, loss 0.405.

#flashcards/ml/example #flashcards/topic/alignment-rl Three answers with reference probabilities 0.5, 0.3, 0.2, rewards 0, 1, 2 and $\beta=1$. What is $\pi^*$?::Boosted: 0.5, 0.815, 1.478. Z = 2.793. π* = [0.179, 0.292, 0.529]. The highest-reward answer went from 0.2 to 0.529.

#flashcards/ml/example #flashcards/topic/alignment-rl A reward gap of 1 and of 2: what is the chance the better answer is preferred?::σ(1) = 0.731 (about 73%). σ(2) = 0.881.
