---
tags: [machine-learning, reinforcement-learning, policy-gradient, REINFORCE, agent, RLHF]
status: growing
level: 4
tier: 3
---

# Policy Gradient

A policy gradient method **teaches an agent by nudging its choice probabilities directly**: after each try, make the actions that led to a good score a bit more likely, and the ones that led to a bad score a bit less likely. [[Q-Learning]] learns how good each move is and then picks the best one. Policy gradient skips that and learns the *choosing rule itself*. The wider picture is in [[Reinforcement Learning]].

## 🟢 1. Simple (age 12)
Imagine you are learning to shoot basketball free throws, and you have a "habit" for how hard to throw. Every time you shoot and score, you tell yourself "do more of that". Every time you miss, you tell yourself "do less of that". You never write down a table of how good each throw is. You just keep nudging your habit toward what worked. After many shots, your habit is good.

**Example:** A robot has to pick between three doors. At first it picks each door equally often. Whenever it gets a treat behind a door, it becomes slightly more likely to pick that door next time. When it gets nothing, it becomes slightly less likely. After a few hundred tries, it mostly picks the door with the best treats.

## 🔵 2. How it works
1. **Start with a policy:** a rule that gives, for each situation (a **state**), a chance for each action. At first the chances are about equal, so the agent tries everything.
2. **Play a whole round:** let the agent act by drawing actions from those chances, and write down every state, action and reward.
3. **Score each step:** for each step, add up the rewards that came *after* it, counting later rewards a bit less (the **return**, see [[Reinforcement Learning]]).
4. **Nudge the chances:** for each step, make the action that was taken more likely if its return was high, and less likely if it was low. The bigger the return, the bigger the nudge. This nudge is a step of [[Gradient Descent]] (really "ascent", going uphill on the reward).
5. **Repeat** over many rounds.
6. **Use a baseline** (optional but very helpful): instead of the plain return, use *how much better than usual* the return was. That stops "everything scored high, so everything gets pushed up" from drowning out the real signal.

![[rl-diagram.svg]]
*The reinforcement learning loop that policy gradient works in. The agent (robot) takes an action in the environment (a maze). The environment returns a state and a reward, and the agent acts again. Policy gradient adjusts the agent's action chances from the rewards it collects. (The same figure is used in [[Reinforcement Learning]].) Image: Megajuice, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Reinforcement_learning_diagram.svg), CC0.*

## 🟠 3. Technical detail
- **A parameterised policy:** the chances come from a [[Neural Network]] (or a small table) with weights $\theta$. For a small set of actions, the last layer is a [[Softmax]] over scores called logits.
- **The core trick:** to change the average reward with respect to $\theta$, you do not need to differentiate through the world. You only need the gradient of the **log of the chance of the action you took**, which is easy to compute (equation 2 below). The world can be a black box.
- **REINFORCE:** the simplest policy gradient method: play, collect returns, nudge each action's log-chance in proportion to its return. It is simple and unbiased, but noisy.
- **Why noisy:** the returns of a whole round carry a lot of luck, so the nudges jump around. More rounds per update help. A **baseline** helps much more (equation 3 below and the code): it removes the part of the return that is shared by every action.
- **Actor-critic:** learn a second network (the *critic*) that estimates the value of a state, and use it as the baseline. This is where policy gradient meets value learning like [[Q-Learning]]. The policy is the *actor* (from memory, the usual name).
- **PPO and friends:** methods such as PPO keep each update small, by clipping how much the action chances can change in one step, so one unlucky batch does not wreck the policy (from memory; see [[RLHF]], where PPO is the common choice).
- **Where it is used:** continuous actions (robot joints), where taking a max over actions as Q-learning does is hard, and tuning [[LLM]]s with [[RLHF]]. There the "action" is the next word, the "policy" is the language model itself, and the reward comes from a learned judge (the details differ between systems; check the one you care about).
- **Costs and limits:** it needs many samples, it can get stuck in a mediocre habit (a local optimum), and it can be unstable. Because it follows the reward blindly, a badly designed reward gets exploited (see the caution in [[Reinforcement Learning]]).
- **On-policy:** the data must come from the *current* policy. After an update, old data no longer fits exactly, which is one reason it is data-hungry compared with off-policy methods like Q-learning.

The code does two things. First, a 3-armed bandit (a single choice with no sequence): it checks that the REINFORCE estimate matches the exact gradient on average, and how much a baseline reduces noise. Second, REINFORCE learning the 5-cell corridor from [[Reinforcement Learning]]: start at cell 0, reach cell 4 for a reward of 10, discount 0.9.

```python
import numpy as np
softmax = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
# 1) a 3-armed bandit: the REINFORCE estimate matches the exact gradient on average; a baseline cuts its noise
R = np.array([10.0, 11.0, 12.0]); theta = np.array([0.5, 0.0, -0.5]); p = softmax(theta)
exact = p * (R - p @ R)                                         # exact gradient of the expected reward w.r.t. the logits
rng = np.random.default_rng(0); a = rng.choice(3, size=200_000, p=p)
glog = np.eye(3)[a] - p                                         # gradient of log pi(a): one-hot minus probabilities
for name, b in (('no baseline', 0.0), ('baseline = mean reward', p @ R)):
    est = glog * (R[a] - b)[:, None]
    print(name, '| mean estimate', est.mean(0).round(3), '| std per draw', est.std(0).round(2))
print('exact gradient          ', exact.round(3))
# 2) learning with REINFORCE: the 5-cell corridor (go right to reach the goal, reward 10, discount 0.9)
def run(baseline, seed, iters=300, N=5, gamma=0.9, lr=0.2, batch=16, cap=40):
    rng = np.random.default_rng(seed); th = np.zeros((N, 2)); steps_hist = []
    for _ in range(iters):
        grad = np.zeros_like(th); G_all = []
        eps = []
        for _ in range(batch):
            s, traj = 0, []
            for _t in range(cap):
                pi = softmax(th[s]); act = rng.choice(2, p=pi); s2 = max(0, s - 1) if act == 0 else s + 1
                r = 10.0 if s2 == N - 1 else 0.0; traj.append((s, act, r)); s = s2
                if s == N - 1: break
            Gs, g = [], 0.0
            for (_, _, r) in reversed(traj): g = r + gamma * g; Gs.append(g)
            Gs = Gs[::-1]; eps.append((traj, Gs)); G_all += Gs; steps_hist.append(len(traj))
        b = np.mean(G_all) if baseline else 0.0
        for traj, Gs in eps:
            for (s, act, _), g in zip(traj, Gs):
                grad[s] += (np.eye(2)[act] - softmax(th[s])) * (g - b)
        th += lr * grad / batch
    return [round(float(softmax(th[s])[1]), 2) for s in range(N - 1)], float(np.mean(steps_hist[-100:]))
for base in (False, True):
    out = [run(base, sd) for sd in range(5)]
    print('baseline' if base else 'no baseline', '| P(right) per cell, seed 0:', out[0][0], '| mean steps to goal at the end (5 seeds):', round(float(np.mean([o[1] for o in out])), 1))
# no baseline | mean estimate [-0.325  0.103  0.222] | std per draw [5.35 5.01 4.54]
# baseline = mean reward | mean estimate [-0.344  0.1    0.244] | std per draw [0.17 0.24 0.4 ]
# exact gradient           [-0.344  0.098  0.246]
# no baseline | P(right) per cell, seed 0: [0.99, 1.0, 1.0, 1.0] | mean steps to goal at the end (5 seeds): 4.0
# baseline | P(right) per cell, seed 0: [0.99, 0.99, 1.0, 1.0] | mean steps to goal at the end (5 seeds): 4.1
```
Reading the output (NumPy only; fixed seeds, so it should repeat):
- **The estimate is right on average.** With 200,000 draws, the average of the REINFORCE nudges is close to the exact gradient. With the baseline, it is within 0.002 on each arm. Without it, it is off by about 0.02, 0.005 and 0.02, which is noise at this sample size (the spread of one draw is about 5, so the average of 200,000 draws still wobbles by about 0.01).
- **The baseline cuts the noise by about 10 to 30 times.** The spread of one nudge falls from about 5 to between 0.17 and 0.40. All three arms pay around 10 to 12, so without a baseline every arm gets a big push up and the real difference between them is a small part of the signal.
- **Learning the corridor:** the agent learns to go right in every cell (chance of "right" about 0.99 or 1.0), and reaches the goal in about 4 steps, the shortest possible, with and without a baseline.
- **Honest limit:** the corridor is so easy that the baseline made no visible difference to the final result (4.0 against 4.1 steps). The noise reduction is clear in the bandit. In this corridor I did not measure how fast each version learned, so I cannot say the baseline sped it up here.

## 🔴 4. Deep / formal

**The policy.** For a small set of actions, the policy uses a softmax over logits $z_a$ (the scores that $\theta$ produces):

$$\pi_\theta(a\mid s)=\frac{e^{z_a}}{\sum_{b}e^{z_b}},\qquad \frac{\partial\,\log\pi_\theta(a\mid s)}{\partial z_k}=\mathbf 1[k=a]-\pi_\theta(k\mid s)$$

**Where:**
- $z_a$ is the score the network gives to action $a$ in state $s$. **Matters:** only differences between scores matter, and bigger means likelier.
- $\pi_\theta(a\mid s)$ is the chance of choosing $a$. **Matters:** the chances add to 1, so raising one action's chance lowers the others.
- $\mathbf 1[k=a]$ is 1 for the action actually taken and 0 for the others. **Matters:** the taken action's score goes up and the rest go down, in proportion to their chances.
- $\mathbf 1[k=a]-\pi(k\mid s)$ is the gradient of the log-chance, with respect to each score. **Matters:** if the taken action was already very likely ($\pi\approx1$), this is near zero (there is little to learn). If it was a long shot, the nudge is big.
- The $\log$ is what makes the gradient simple. **Matters:** this exact form (one-hot minus probabilities) is the thing REINFORCE multiplies by the return.

**Example (tiny numbers):** Three actions with chances $\pi=(0.5,\,0.3,\,0.2)$, and the agent takes action 2 (the middle one, chance 0.3).
1. One-hot: $(0,1,0)$.
2. Gradient of the log-chance: $(0-0.5,\ 1-0.3,\ 0-0.2)=(-0.5,\ +0.7,\ -0.2)$.
3. Meaning: if this action earned a good return, the scores move by about $(-0.5,+0.7,-0.2)$ times the return size: the taken action's score goes up the most, and the others' scores go down. A rarely chosen action (chance 0.3) gets a stronger push than a common one would.

**The policy gradient (REINFORCE).** Let $J(\theta)$ be the average total reward under policy $\pi_\theta$. Its gradient is

$$\nabla_\theta J(\theta)=\mathbb{E}\Big[\sum_{t}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\;G_t\Big]$$

**Where:**
- $J(\theta)$ is the average reward the policy earns. **Matters:** it is the thing we want to push up.
- $\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ is the direction that makes the action taken at step $t$ more likely (as in the previous equation). **Matters:** it is the only place where the policy's weights appear, so the world does not need to be differentiable.
- $G_t$ is the return from step $t$ onward (rewards added up, later ones discounted by $\gamma$). **Matters:** it decides how hard to push: a high return pushes the action's chance up, and a low (or negative) return pushes it down.
- $\mathbb{E}$ is the average over many rounds of play. **Matters:** one round is a noisy sample. The estimate is unbiased (right on average), but you need many rounds, or a baseline, to make it steady.
- The identity behind this uses the "log-derivative trick" ($\nabla p=p\,\nabla\log p$) to turn a gradient of an average into an average of gradients. I checked the result numerically and did not write out the derivation here.

**Example (tiny numbers):** One choice (a bandit), two actions, chances $(0.5,0.5)$ (logits $(0,0)$). Action 0 pays reward 1, action 1 pays reward 0.
1. Exact gradient: $J=\pi_0\times1$, and $\partial J/\partial z_0=\pi_0(1-\pi_0)=0.25$, $\partial J/\partial z_1=-0.25$.
2. If action 0 is drawn (half of the time): gradient of the log-chance is $(1-0.5,\,0-0.5)=(0.5,-0.5)$, times return 1: $(0.5,-0.5)$.
3. If action 1 is drawn (the other half): return is 0, so the nudge is $(0,0)$.
4. Average over the two cases: $0.5\times(0.5,-0.5)+0.5\times(0,0)=(0.25,-0.25)$. This equals the exact gradient.
5. Meaning: each single draw is only a rough push (either $(0.5,-0.5)$ or nothing), but on average the pushes point exactly where the exact gradient points: up on action 0, down on action 1.

**The baseline.** You may subtract any number $b$ that does not depend on the action taken, and the average stays the same:

$$\nabla_\theta J(\theta)=\mathbb{E}\Big[\sum_{t}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\big(G_t-b(s_t)\big)\Big]$$

**Where:**
- $b(s_t)$ is the baseline, often an estimate of the average return from state $s_t$ (a critic does this job; see "actor-critic" above). **Matters:** it must not depend on the action chosen, or the estimate becomes biased. It *can* depend on the state.
- $G_t-b(s_t)$ is how much better (or worse) than usual this return was, sometimes called the *advantage*. **Matters:** now only the *difference* from normal pushes the policy, so a lucky or unlucky scale shared by all actions no longer adds noise.
- The rest is as before. **Matters:** the average is unchanged, because the average of $\nabla\log\pi$ over the policy's own draws is zero (checked numerically in the code, not derived here).
- A good baseline reduces the noise without changing what the method is aiming at, which is why it is nearly always used.

**Example (tiny numbers):** Two actions with chances $(0.5,0.5)$ pay rewards $10$ and $11$ (both large).
1. Exact gradient: $\partial J/\partial z_0=\pi_0(R_0-\bar R)=0.5\times(10-10.5)=-0.25$ (and $+0.25$ for action 1).
2. **No baseline.** Action 0 drawn: $(0.5,-0.5)\times10=(5,-5)$. Action 1 drawn: $(-0.5,0.5)\times11=(-5.5,5.5)$. Average: $(-0.25,\,0.25)$, which is right, but each single draw is a huge $\pm5$ push in a direction that flips with the draw.
3. **Baseline $b=10.5$.** Action 0: $(0.5,-0.5)\times(10-10.5)=(-0.25,0.25)$. Action 1: $(-0.5,0.5)\times(11-10.5)=(-0.25,0.25)$. Both draws give exactly $(-0.25,0.25)$.
4. Meaning: the average is the same $(-0.25,0.25)$, but the noise has dropped from $\pm5$ to zero. In this two-action case, the baseline removed all the noise. In real problems it removes part of it, as in the bandit above (about 5 down to 0.2 to 0.4).

**Other points (from memory, not checked here):**
- The first form of this idea is Williams' REINFORCE algorithm (1992). The general result for sequences of decisions is often called the policy gradient theorem (Sutton and colleagues, around 2000). I did not open either paper for this note.
- The variance of the estimate grows with the length of the round, which is why discounting, baselines and critics matter more for long tasks.
- In [[RLHF]], PPO adds a clipped update and a penalty that keeps the new policy near the old one (the KL leash, see [[KL Divergence]]). That machinery is not covered here.
- Methods such as DPO (see [[RLHF]]) avoid running this loop at all, by turning the preference data into a plain training loss.

**What came before, and how this improved on it:**
- **Before:** value-based methods such as [[Q-Learning]], which learn the value of actions and then pick the highest.
- **Its limit:** they work best with a small set of actions and a policy that is "pick the best"; learning the policy directly is more natural when actions are many or random choice is useful (this is my summary, not a quoted claim).
- **What changed:** nudge the policy's probabilities up for actions that earned more reward than expected (REINFORCE, Williams, 1992; from memory). Subtracting a baseline lowers the noise.
- **How much better:** in the code below (three slot machines, 300 steps, 100 runs), the policy puts a probability of 0.967 on the best machine without a baseline and 0.987 with one. The noise of the gradient estimate drops from 0.138 to 0.116, 16% lower. The gain is modest on this tiny problem.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
mean = np.array([0.2, 0.5, 0.8])
def run(baseline, steps=300, lr=0.5):
    th = np.zeros(3); gs = []
    for _ in range(steps):
        p = np.exp(th) / np.exp(th).sum(); a = rng.choice(3, p=p); r = float(rng.random() < mean[a])
        g = -p.copy(); g[a] += 1                                      # gradient of log p(a) with respect to th
        adv = r - (p @ mean if baseline else 0.0)                     # baseline: the reward the current policy expects
        th += lr * adv * g; gs.append(adv * g)
    p = np.exp(th) / np.exp(th).sum(); return p[2], np.std(gs)
for b in (False, True):
    res = [run(b) for _ in range(100)]
    print("with baseline   " if b else "without baseline", " probability of the best arm after 300 steps:", round(np.mean([r[0] for r in res]), 3), "  spread of the gradient estimate:", round(np.mean([r[1] for r in res]), 3))
```

## Related
[[Reinforcement Learning]] · [[Q-Learning]] · [[RLHF]] · [[Agent]] · [[LLM]] · [[Softmax]] · [[Gradient Descent]] · [[Neural Network]] · [[Probability]] · [[KL Divergence]] · [[Bias-Variance]] · [[Loss Function]] · [[PPO]] · [[GRPO]] · [[Deep Q-Network]]
## Sources
- Williams (1992), the REINFORCE algorithm, and Sutton, McAllester, Singh and Mansour (around 2000), the policy gradient theorem (both from memory; I did not open them for this note).
- Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition, the chapter on policy gradient methods (from memory).
- Schulman and colleagues (2017), "Proximal Policy Optimization Algorithms" (PPO; from memory).
- Figure: [Wikimedia Commons, Reinforcement_learning_diagram.svg](https://commons.wikimedia.org/wiki/File:Reinforcement_learning_diagram.svg), by Megajuice, CC0 (reused from [[Reinforcement Learning]]).
- The code and every Example number were run for this note (NumPy only).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/alignment-rl Policy gradient:::Learn the policy directly. After each try, nudge the action chances up when the return was good and down when it was bad. No table of action values is needed.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the softmax policy and the gradient of its log-chance and explain each symbol.
?
$$\pi_\theta(a\mid s)=\frac{e^{z_a}}{\sum_{b}e^{z_b}},\qquad \frac{\partial\,\log\pi_\theta(a\mid s)}{\partial z_k}=\mathbf 1[k=a]-\pi_\theta(k\mid s)$$
- $z_a$: the score the network gives to action $a$ in state $s$. Only differences matter.
- $\pi_\theta(a\mid s)$: the chance of choosing $a$. The chances add to 1, so raising one lowers the others.
- $\mathbf 1[k=a]$: 1 for the action taken, 0 for the others.
- $\mathbf 1[k=a]-\pi(k\mid s)$: the gradient of the log-chance for each score. The taken action's score goes up and the rest go down in proportion to their chances.

#flashcards/ml/example #flashcards/topic/alignment-rl Chances $\pi=(0.5,0.3,0.2)$, the agent takes action 2 (chance 0.3). What is the gradient of the log-chance?::One-hot $(0,1,0)$ minus $\pi$: $(-0.5,+0.7,-0.2)$. If this action earned a good return, the scores move by about this times the return size. A rarely chosen action gets a stronger push than a common one.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the policy gradient theorem and explain each symbol.
?
$$\nabla_\theta J(\theta)=\mathbb{E}\Big[\sum_{t}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\;G_t\Big]$$
- $J(\theta)$: the average reward the policy earns, the thing to push up.
- $\nabla_\theta\log\pi_\theta(a_t\mid s_t)$: the direction that makes the action at step $t$ more likely. It is the only place the policy's weights appear, so the world need not be differentiable.
- $G_t$: the return from step $t$ on (later rewards discounted by $\gamma$). A high return pushes the action's chance up, a low or negative one pushes it down.
- $\mathbb{E}$: the average over many tries.

#flashcards/ml/example #flashcards/topic/alignment-rl One choice, two actions, chances (0.5, 0.5). Action 0 pays 1, action 1 pays 0. What does a sampled push average to?::Exact gradient $\partial J/\partial z_0=\pi_0(1-\pi_0)=0.25$ and $-0.25$. Action 0 drawn (half the time): $(0.5,-0.5)\times1$. Action 1 drawn: $(0,0)$. Average $(0.25,-0.25)$, equal to the exact gradient. Each single draw is rough but on average points the right way.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the policy gradient with a baseline and explain each symbol.
?
$$\nabla_\theta J(\theta)=\mathbb{E}\Big[\sum_{t}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\big(G_t-b(s_t)\big)\Big]$$
- $b(s_t)$: the baseline, often an estimate of the average return from $s_t$ (a critic does this). It must not depend on the action chosen or the estimate becomes biased. It can depend on the state.
- $G_t-b(s_t)$: how much better or worse than usual this return was (the advantage). Only the difference from normal pushes the policy, so a shared lucky or unlucky scale no longer adds noise.
- The average is unchanged, but the noise is lower.

#flashcards/ml/example #flashcards/topic/alignment-rl Two actions with chances (0.5, 0.5) pay 10 and 11. Compare the pushes with no baseline and with $b=10.5$.::No baseline: action 0 gives $(5,-5)$, action 1 gives $(-5.5,5.5)$. Average $(-0.25,0.25)$ but each draw is a huge push of about ±5. With $b=10.5$ both draws give exactly $(-0.25,0.25)$. Same average, noise gone.

#flashcards/ml/image #flashcards/topic/alignment-rl ![[rl-diagram.svg]]
What does this picture show?
?
The reinforcement learning loop that policy gradient works in. The agent (robot) takes an action in the environment (a maze). The environment returns a state and a reward, and the agent acts again. Policy gradient adjusts the agent's action chances from the rewards it collects. (From the note's caption: Megajuice, Wikimedia Commons.)

#flashcards/ml/cloze #flashcards/topic/alignment-rl A baseline must not depend on the ==action==, but it can depend on the ==state==.
