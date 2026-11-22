---
tags: [machine-learning, reinforcement-learning, policy-gradient, PPO, RLHF, agent]
status: growing
level: 4
tier: 4
---

# PPO

PPO (proximal policy optimization) is a way to train an agent with [[Policy Gradient]] methods **while stopping each update from changing the agent's behaviour too much**. It does this by cutting off the reward for making any one action much more (or much less) likely than before. It is the usual optimiser in classic [[RLHF]]. The ideas of rewards and policies are in [[Reinforcement Learning]] and [[Policy Gradient]].

## 🟢 1. Simple (age 12)
Imagine you are learning to ride a bike, and a coach gives you tips after each practice. If you change your whole style after every tip, one bad tip can send you into a bush. A sensible coach says: "Change a little bit each time. If a tip tells you to change a lot, only change a little anyway." PPO is that coach for a learning machine: **make small, safe steps, and never trust one batch of practice so much that you jump far**.

**Example:** A robot learns which way to turn at a corner. After one lucky run through a left turn, it feels like "always turn left!". PPO says: "Fine, make left a bit more likely (say from 40% to 48%), but not from 40% to 99%. Try more runs before you go further."

## 🔵 2. How it works
1. **Play with the current policy** (the agent's choosing rule) and write down what happened: each state, the action taken, and the rewards that followed.
2. **Score each action:** for each step, work out whether it turned out *better or worse than expected*. That score is the **advantage** (see [[Policy Gradient]]). It uses a second small network, the **critic**, that guesses how good each state is.
3. **Freeze a copy** of the policy that made the choices (the "old" one).
4. **Compare new with old:** after each small update, ask for each action, "how many times more likely is it now than under the old policy?" That number is the **ratio**. 1.0 means unchanged, 1.25 means 25% more likely.
5. **Reward good changes, but only up to a limit.** If an action was good (positive advantage), raising its chance helps, but the credit stops once its ratio passes about 1.2. If an action was bad, lowering its chance helps, but the credit stops once the ratio falls below about 0.8.
6. **Reuse the same batch** for several small update rounds (this is what the limit makes safe), then throw it away and play again with the new policy.
7. **Repeat.** The agent improves in many small, careful steps.

![[rlhf-diagram.svg]]
*Where PPO is most often met: the RLHF pipeline. A supervised model writes answers, human annotators rank them, the ranking trains a reward model, and the reward model then guides training (with PPO) to produce the aligned model. (The same figure is used in [[RLHF]].) Image: PopoDameron, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:RLHF_diagram.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **The problem it solves:** a plain policy gradient uses each batch of experience once. That is wasteful, but reusing it many times is dangerous: after the policy has changed, the old batch no longer describes it, and a big step on a noisy batch can make the policy much worse (the code below shows this).
- **The ratio:** the new chance of an action divided by the old chance. It lets old data be reweighted to look like data from the new policy, for small changes.
- **The clip:** the objective takes the *smaller* of the plain reward and a version with the ratio clamped to a window around 1 (width set by $\varepsilon$, commonly 0.2, from memory). Beyond the window, the gradient is zero, so there is no push to go further.
- **Advantages with GAE:** instead of the raw return, PPO usually estimates the advantage by *generalised advantage estimation*, a blend of one-step guesses with the critic that trades noise against bias (equation 3 below).
- **Actor and critic:** the policy (actor) and the value guesser (critic) are often two heads of one [[Neural Network]], trained together. The critic's loss is a squared-error [[Loss Function]]. An entropy bonus is often added so the policy does not become sure of itself too early (from memory; not in the code).
- **A simpler cousin of trust-region methods:** older methods (TRPO, from memory) enforce a limit on how far the policy moves with a more complex constraint. PPO gets a similar effect with just a clipped formula, so it is easier to code and run.
- **In RLHF:** the "policy" is the language model ([[LLM]]), an "action" is the next token, and the reward comes from a reward model, usually with a [[KL Divergence]] penalty that keeps the tuned model near the starting one. Several large models sit in memory at once (policy, old policy, critic, reward model, reference model), which is why classic RLHF is heavy to run (see [[GPU]] and [[RLHF]]). Recipes differ between systems, so check the one you care about.
- **Limits:** results can depend on many small settings (batch size, learning rate, number of reuse rounds, clipping width). It is on-policy, so it needs fresh data after each round. The clip does not *guarantee* a safe step. It only removes the push to leave the window, and a policy can still drift beyond it.

The code does two things. First, a noisy 3-armed bandit: from one small noisy batch, take 200 gradient steps on the plain objective and on the clipped one, and compare how far the policy moves and how good it ends up. Second, PPO with GAE on the 5-cell corridor from [[Reinforcement Learning]] (reward 1 for reaching cell 4, discount 0.9).

```python
import numpy as np
softmax = lambda z: np.exp(z - z.max(-1, keepdims=True)) / np.exp(z - z.max(-1, keepdims=True)).sum(-1, keepdims=True)
# 1) one batch, many gradient steps: unclipped surrogate vs clipped (PPO) on a noisy 3-armed bandit
mu = np.array([1.0, 1.5, 2.0])                                   # true mean reward of each arm
def one_batch(clip, rng, n=8, steps=200, lr=0.5, eps=0.2):
    z0 = np.zeros(3); p_old = softmax(z0)                        # old policy: uniform
    a = rng.choice(3, size=n, p=p_old); R = mu[a] + rng.normal(0, 3.0, n); A = R - R.mean()      # noisy advantages
    z = z0.copy()
    for _ in range(steps):
        p = softmax(z); r = p[a] / p_old[a]
        live = np.ones(n, bool) if not clip else ~(((A > 0) & (r > 1 + eps)) | ((A < 0) & (r < 1 - eps)))   # clipped samples give no gradient
        g = np.zeros(3)
        for i in np.flatnonzero(live): g += r[i] * A[i] * (np.eye(3)[a[i]] - p)                     # gradient of r*A w.r.t. the logits
        z += lr * g / n
    p = softmax(z); r = p[a] / p_old[a]
    return float(np.abs(r - 1).max()), float(p @ mu)
rng = np.random.default_rng(0)
for clip in (False, True):
    out = np.array([one_batch(clip, rng) for _ in range(200)])
    print('clipped' if clip else 'unclipped', '| mean largest |ratio-1|:', out[:, 0].mean().round(2), '| true reward of new policy: mean', out[:, 1].mean().round(3), ', 10th percentile', np.percentile(out[:, 1], 10).round(3), '| worse than old:', (out[:, 1] < 1.5).mean().round(2))
print('(old policy: reward 1.5, best arm 2.0)')
# 2) PPO on the 5-cell corridor: learned policy table + learned value table + GAE
def ppo(seed, iters=40, N=5, gamma=0.9, lam=0.95, eps=0.2, epochs=8, lr=3.0, batch=16, cap=40):
    rng = np.random.default_rng(seed); z = np.zeros((N, 2)); V = np.zeros(N); steps_hist = []; clipped = []
    for _ in range(iters):
        S, Ac, ADV, RET = [], [], [], []
        for _e in range(batch):
            s, traj = 0, []
            for _t in range(cap):
                a = rng.choice(2, p=softmax(z[s])); s2 = max(0, s - 1) if a == 0 else s + 1
                traj.append((s, a, 1.0 if s2 == N - 1 else 0.0, s2)); s = s2
                if s == N - 1: break
            steps_hist.append(len(traj)); adv = 0.0; advs = []
            for (s_, a_, r_, s2_) in reversed(traj):                         # GAE: discounted sum of one-step errors
                delta = r_ + gamma * (0.0 if s2_ == N - 1 else V[s2_]) - V[s_]; adv = delta + gamma * lam * adv; advs.append(adv)
            advs = advs[::-1]
            for (s_, a_, _, _), ad in zip(traj, advs): S.append(s_); Ac.append(a_); ADV.append(ad); RET.append(ad + V[s_])
        S, Ac, ADV, RET = map(np.array, (S, Ac, ADV, RET)); p_old = softmax(z)[S, Ac]
        for _k in range(epochs):
            p = softmax(z); r = p[S, Ac] / p_old
            live = ~(((ADV > 0) & (r > 1 + eps)) | ((ADV < 0) & (r < 1 - eps))); clipped.append(1 - live.mean())
            g = np.zeros_like(z)
            for i in np.flatnonzero(live): g[S[i]] += r[i] * ADV[i] * (np.eye(2)[Ac[i]] - p[S[i]])
            z += lr * g / len(S)
        for i in range(len(S)): V[S[i]] += 0.05 * (RET[i] - V[S[i]])        # move the value table toward the observed return
    return [round(float(softmax(z[s])[1]), 2) for s in range(N - 1)], float(np.mean(steps_hist[-100:])), float(np.mean(clipped))
runs = [ppo(sd) for sd in range(5)]
print('PPO corridor | P(right) per cell, seed 0:', runs[0][0], '| mean steps to goal (5 seeds):', round(float(np.mean([r[1] for r in runs])), 1), '| share of samples clipped in an update:', round(float(np.mean([r[2] for r in runs])), 2))
# unclipped | mean largest |ratio-1|: 1.96 | true reward of new policy: mean 1.651 , 10th percentile 1.004 | worse than old: 0.28
# clipped | mean largest |ratio-1|: 0.36 | true reward of new policy: mean 1.533 , 10th percentile 1.41 | worse than old: 0.3
# (old policy: reward 1.5, best arm 2.0)
# PPO corridor | P(right) per cell, seed 0: [0.97, 0.99, 0.98, 0.99] | mean steps to goal (5 seeds): 4.1 | share of samples clipped in an update: 0.01
```
Reading the output (NumPy only; fixed seeds, so it should repeat):
- **How far it moves:** from one batch of only 8 noisy samples, 200 steps on the plain objective push some action's chance to almost 3 times its old value (the largest ratio is off from 1 by 1.96 on average). With the clip, the largest change is 0.36, near the 0.2 window. The clipped policy stays close to the old one, as intended. (It overshoots the 0.2 window a little, probably because the gradient only stops *after* a sample leaves the window and several samples push on the same action. I did not test that explanation.)
- **How good the result is:** the old policy earns 1.5. After the plain updates, the new policy earns 1.651 on average, but the **10th percentile is 1.004**: in the worst tenth of the batches, the policy ended up much worse than before. With the clip, the average is 1.533 and the 10th percentile is **1.41**. So the clip gave up a lot of the average gain, but it cut the worst case.
- **It does not remove bad batches:** a similar share of batches ended below the old policy's 1.5 (28% plain, 30% clipped). The clip limits *how much damage* a bad batch does, not *whether* it happens.
- **In an easier setting, there is no benefit to clipping:** with 64 samples per batch and less noise, I also tried the same test as a side check (not in the code). The plain version had a higher mean (1.84 against 1.56 for the clipped one) and about the same share of bad batches, so the clip only slowed the progress. So the clip is a safety device for noisy data, not a speed-up.
- **Corridor:** PPO learns to go right in each cell (chances 0.97 to 0.99) and reaches the goal in about 4.1 steps, close to the minimum of 4. The clipping was rarely triggered there (about 1% of samples), so in this tiny problem the clip did little. I picked the learning rate (3.0) and 8 reuse rounds by hand after an earlier setting (0.5 and 4 rounds) learned more slowly (5.2 steps after more iterations), so the corridor result does not show that PPO beats other methods.

## 🔴 4. Deep / formal

**The ratio and the plain objective.** Let $\pi_{\text{old}}$ be the policy that collected the data and $\pi_\theta$ the policy being updated. For an action $a_t$ in state $s_t$, the ratio is

$$r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)},\qquad L^{\text{plain}}(\theta)=\mathbb{E}\big[r_t(\theta)\,\hat A_t\big]$$

**Where:**
- $\pi_{\text{old}}(a_t\mid s_t)$ is the chance the old policy gave to the action taken. **Matters:** it is fixed during the update, since the data came from it.
- $\pi_\theta(a_t\mid s_t)$ is the chance the updated policy gives to the same action. **Matters:** it is what the gradient changes.
- $r_t(\theta)$ is their ratio. **Matters:** it is 1 at the start of every update. It reweights old data to look like data from the new policy (this is accurate only while the policies stay close).
- $\hat A_t$ is the estimated advantage of the action ("how much better than usual"). **Matters:** a positive value pushes the action's chance up, and a negative value pushes it down.
- $\mathbb{E}$ is the average over the batch. **Matters:** the gradient of $r_t$ at $r_t=1$ equals the policy gradient from [[Policy Gradient]], so this is that method with the ability to reuse a batch.
- Without any limit, repeated steps on this objective keep raising $r_t$ for good actions without end, which is the over-confident jump the code shows.

**Example (tiny numbers):** The old policy gave the action chance $0.40$. After some updates, the new policy gives it $0.50$. The advantage is $+2$.
1. Ratio: $r=0.50/0.40=1.25$.
2. Objective for this sample: $r\hat A=1.25\times2=2.5$.
3. At the start, $r=1$ gives $1\times2=2$, so the update already added $0.5$.
4. Meaning: a chance rising from 0.40 to 0.50 (25% more likely) raises the objective from 2 to 2.5, and there is nothing in the plain objective that says "stop". The next equation adds the stop.

**The clipped objective.** PPO replaces the plain objective by

$$L^{\text{clip}}(\theta)=\mathbb{E}\Big[\min\big(r_t\hat A_t,\ \operatorname{clip}(r_t,\,1-\varepsilon,\,1+\varepsilon)\,\hat A_t\big)\Big]$$

**Where:**
- $\varepsilon$ is the clipping width, commonly 0.2 (from memory). **Matters:** it sets the window $[0.8,\,1.2]$ within which a ratio earns full credit. A smaller $\varepsilon$ means more cautious steps, and a larger one means faster but riskier steps.
- $\operatorname{clip}(r,1-\varepsilon,1+\varepsilon)$ forces the ratio into the window (values below the window become $1-\varepsilon$, above it $1+\varepsilon$). **Matters:** inside the window it changes nothing. Outside, it makes the value constant, which has a zero gradient.
- $\min(\cdot,\cdot)$ takes the smaller (more pessimistic) of the two. **Matters:** the clip only cuts off *gains*: it stops credit for pushing a good action too high or a bad action too low. It never hides a *loss*, so if an update makes things worse, the full penalty remains and the gradient can fix it.
- $\hat A_t$ is the advantage as before. **Matters:** its sign decides which side of the window matters.

**Example (tiny numbers):** $\varepsilon=0.2$, so the window is $[0.8,1.2]$. Four cases:

| advantage $\hat A$ | ratio $r$ | plain $r\hat A$ | clipped version | $\min$ | gradient? |
|---|---|---|---|---|---|
| $+2$ | $1.25$ | $2.5$ | $1.2\times2=2.4$ | $2.4$ | none (clipped) |
| $+2$ | $0.70$ | $1.4$ | $0.8\times2=1.6$ | $1.4$ | yes |
| $-2$ | $0.70$ | $-1.4$ | $0.8\times(-2)=-1.6$ | $-1.6$ | none (clipped) |
| $-2$ | $1.25$ | $-2.5$ | $1.2\times(-2)=-2.4$ | $-2.5$ | yes |

1. Row 1: a good action already made 25% more likely. The credit stops at 2.4, so there is no more push.
2. Row 2: a good action that has become *less* likely (0.70). The plain value, 1.4, is the smaller one, so the gradient stays and pushes it back up.
3. Row 3: a bad action already made much less likely. Credit stops at $-1.6$, so there is no more push down.
4. Row 4: a bad action that became *more* likely. The penalty $-2.5$ stays in full, so the gradient pushes it back down.
5. Meaning: the clip switches off the push only in the two cases where the policy has already moved far in the *right* direction. In the two cases where it moved the wrong way, the gradient stays on.

**Estimating the advantage (GAE).** With a critic $V$, let the one-step error be $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$. The generalised advantage estimate is

$$\hat A_t=\sum_{l=0}^{\infty}(\gamma\lambda)^{l}\,\delta_{t+l}$$

**Where:**
- $r_t$ is the reward at step $t$ (not the ratio above; the letter is reused). $V(s)$ is the critic's guess of the future reward from state $s$. **Matters:** $\delta_t$ says whether this step turned out better ($>0$) or worse than the critic expected.
- $\gamma$ is the discount for later rewards, and $\lambda$ is a second number between 0 and 1. **Matters:** $\lambda=0$ uses only the one-step error $\delta_t$ (low noise, but it trusts the critic's guess). $\lambda=1$ gives the full return minus $V(s_t)$ (no critic bias, but more noise). Values like 0.95 sit in between (from memory, a common choice).
- $(\gamma\lambda)^l$ shrinks the errors of later steps. **Matters:** nearby steps count most, since they are the ones this action most plausibly caused.
- The sum ends at the end of the episode. **Matters:** after the last step there is no more reward, and $V$ of the end state is taken to be 0.

**Example (tiny numbers):** Three steps with rewards $(0,0,1)$ and the critic's guesses $V=(0.5,\,0.6,\,0.8)$. $\gamma=0.9$, $\lambda=0.95$, so $\gamma\lambda=0.855$.
1. One-step errors: $\delta_0=0+0.9\times0.6-0.5=0.04$. $\delta_1=0+0.9\times0.8-0.6=0.12$. $\delta_2=1+0-0.8=0.20$ (the episode ends).
2. Work backwards: $\hat A_2=0.20$. $\hat A_1=0.12+0.855\times0.20=0.291$. $\hat A_0=0.04+0.855\times0.291=0.289$.
3. Compare the two extremes for step 0. $\lambda=0$: $\hat A_0=0.04$. $\lambda=1$: the return $0.9^2\times1=0.81$ minus $V(s_0)=0.5$, giving $0.31$.
4. Meaning: all three steps were better than the critic expected (every advantage is positive), and later steps look more valuable since they are nearer the reward. $\lambda=0.95$ gives 0.289, between the cautious 0.04 and the full-return 0.31, and closer to the full return.

**Other points (from memory, not checked here):**
- PPO was introduced by Schulman and colleagues (2017) as a simpler alternative to TRPO. GAE comes from an earlier paper by the same group (2015). I did not open either paper for this note.
- In RLHF-style training, a per-token KL penalty to a reference model is often added to the reward. Some implementations apply the penalty inside the loss instead. The details differ between libraries.
- The clipped objective is only one of two versions in the paper. The other uses an adaptive KL penalty. I have not tested the second one.
- Methods such as DPO and others (see [[RLHF]]) avoid running a PPO loop at all, by turning preference data into a plain training loss.

**What came before, and how this improved on it:**
- **Before:** standard policy gradient, which makes one update per data sample. A step that is too big can wreck the policy (this is my summary of why trust regions were wanted).
- **Its limit:** one update per sample wastes data, and reusing data for more updates risks moving the policy too far.
- **What changed:** clip the objective so that updates stop pushing once the new policy differs from the old by more than a small amount, which allows several epochs on the same data (Schulman et al., arXiv 1707.06347, 20 July 2017, abstract read; first author from memory).
- **How much better:** the abstract says PPO has better sample complexity empirically than other online policy gradient methods, while being simpler than trust-region methods. In the code below, 40 repeated updates on one batch make the chosen action 2.74 times as likely as before without clipping, and 1.22 times as likely with clipping at 0.2 (the bound is 1.2, slightly passed by one step). The step size 0.1 was my first choice and the other sizes I tried gave a similar picture.

**Check it yourself:**

```python
import numpy as np
mean = np.array([0.2, 0.5, 0.8]); eps = 0.2
th0 = np.zeros(3); p0 = np.exp(th0) / np.exp(th0).sum()
a, adv = 2, 1.0                                    # a batch with one good action (arm 2) and a positive advantage
def run(clip, steps=40, lr=0.1):
    th = th0.copy()
    for _ in range(steps):
        p = np.exp(th) / np.exp(th).sum(); ratio = p[a] / p0[a]
        if clip and ratio > 1 + eps: break                            # clipped: no gradient once the ratio is past 1+eps
        g = -p.copy(); g[a] += 1; th += lr * adv * ratio * g          # gradient of ratio * advantage
    p = np.exp(th) / np.exp(th).sum(); return p[a] / p0[a], p[a]
for clip in (False, True):
    ratio, pa = run(clip)
    print("with clipping   " if clip else "without clipping", ": the chosen action became", round(ratio, 2), "x as likely as before (probability", round(pa, 3), "from 0.333)")
```

## Related
[[Policy Gradient]] · [[Reinforcement Learning]] · [[RLHF]] · [[Q-Learning]] · [[Agent]] · [[LLM]] · [[KL Divergence]] · [[Neural Network]] · [[Loss Function]] · [[Gradient Descent]] · [[GPU]] · [[Probability]] · [[GRPO]] · [[Direct Preference Optimization]] · [[Reward Model]] · [[Reasoning Model]] · [[Experience Replay]]
## Sources
- Schulman, Wolski, Dhariwal, Radford and Klimov (2017), "Proximal Policy Optimization Algorithms" (cited from memory; not opened for this note).
- Schulman and colleagues (2015), "High-Dimensional Continuous Control Using Generalized Advantage Estimation", and the TRPO paper (both from memory).
- Ouyang and colleagues (2022), the InstructGPT paper, for PPO in RLHF (from memory).
- Figure: [Wikimedia Commons, RLHF_diagram.svg](https://commons.wikimedia.org/wiki/File:RLHF_diagram.svg), by PopoDameron, CC BY-SA 4.0 (reused from [[RLHF]]).
- The code, the table of clipping cases, the GAE numbers and every Example number were run for this note (NumPy only). The bandit and corridor are small toys.
- Schulman et al., arXiv 1707.06347 (abstract read for the history block)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/alignment-rl PPO (Proximal Policy Optimization):::Make small, safe policy updates. Never trust one batch of practice so much that you change the policy a lot.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the probability ratio and the plain objective and explain each symbol.
?
$$r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)},\qquad L^{\text{plain}}(\theta)=\mathbb{E}\big[r_t(\theta)\,\hat A_t\big]$$
- $\pi_{\text{old}}(a_t\mid s_t)$: the chance the old policy gave to the action taken. Fixed during the update, since the data came from it.
- $\pi_\theta(a_t\mid s_t)$: the chance the updated policy gives to the same action. The gradient changes this.
- $r_t(\theta)$: their ratio. It is 1 at the start of each update and reweights old data to look like new-policy data (accurate only while the policies stay close).
- $\hat A_t$: the estimated advantage ("how much better than usual"). Positive pushes the action's chance up, negative pushes it down.

#flashcards/ml/example #flashcards/topic/alignment-rl Old policy gave the action 0.40, new gives 0.50, advantage +2. What is the ratio and the objective?::Ratio $0.50/0.40=1.25$. Objective $1.25\times2=2.5$. At the start $r=1$ gave 2, so the update has already added 0.5.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the PPO clipped objective and explain each symbol.
?
$$L^{\text{clip}}(\theta)=\mathbb{E}\Big[\min\big(r_t\hat A_t,\ \operatorname{clip}(r_t,\,1-\varepsilon,\,1+\varepsilon)\,\hat A_t\big)\Big]$$
- $\varepsilon$: the clipping width, commonly 0.2 (from memory). It sets the window $[0.8,1.2]$ within which a ratio earns full credit. Smaller means more cautious steps.
- $\operatorname{clip}(r,1-\varepsilon,1+\varepsilon)$: forces the ratio into the window. Inside it changes nothing, outside it is constant, so the gradient is zero.
- $\min(\cdot,\cdot)$: takes the more pessimistic of the two.

#flashcards/ml/example #flashcards/topic/alignment-rl With $\varepsilon=0.2$ and advantage +2, what happens for ratio 1.25? For advantage +2 at ratio 0.70? For advantage −2 at ratio 0.70, and at 1.25?::$A=+2$, $r=1.25$: plain 2.5, clipped $1.2\times2=2.4$, min 2.4, no more push. $A=+2$, $r=0.70$: plain 1.4, clipped 1.6, min 1.4, gradient stays and pushes it back up. $A=-2$, $r=0.70$: plain −1.4, clipped −1.6, min −1.6, no more push down. $A=-2$, $r=1.25$: plain −2.5, clipped −2.4, min −2.5, the penalty stays and pushes it back down. The clip switches off the push only when the policy has already moved far in the right direction.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write generalized advantage estimation (GAE) and explain each symbol.
?
$$\hat A_t=\sum_{l=0}^{\infty}(\gamma\lambda)^{l}\,\delta_{t+l}$$
- $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$: the one-step error. $r_t$ here is the reward at step $t$ (the letter is reused, not the ratio). $V(s)$ is the critic's guess of future reward. $\delta_t>0$ means better than the critic expected.
- $\gamma$: the discount. $\lambda\in[0,1]$: a second number. $\lambda=0$ uses only $\delta_t$ (low noise, trusts the critic). $\lambda=1$ gives the full return minus $V(s_t)$ (no critic bias, more noise). About 0.95 is common (from memory).
- $(\gamma\lambda)^l$ shrinks errors further ahead.

#flashcards/ml/example #flashcards/topic/alignment-rl Rewards (0,0,1), critic $V=(0.5,0.6,0.8)$, $\gamma=0.9$, $\lambda=0.95$. What are the advantages?::$\delta_0=0.04$, $\delta_1=0.12$, $\delta_2=0.20$. Backwards: $\hat A_2=0.20$, $\hat A_1=0.12+0.855\times0.20=0.291$, $\hat A_0=0.04+0.855\times0.291=0.289$. At $\lambda=0$ step 0 would be 0.04, at $\lambda=1$ it would be $0.81-0.5=0.31$.

#flashcards/ml/image #flashcards/topic/alignment-rl ![[rlhf-diagram.svg]]
What does this picture show?
?
Where PPO is most often met, the RLHF pipeline. A supervised model writes answers, human annotators rank them, the ranking trains a reward model, and the reward model then guides training (with PPO) to produce the aligned model. (From the note's caption: PopoDameron, Wikimedia Commons.)

#flashcards/ml/cloze #flashcards/topic/alignment-rl PPO's clip stops pushing when the policy has already moved ==far in the right direction==.
