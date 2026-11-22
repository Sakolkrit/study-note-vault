---
tags: [machine-learning, reinforcement-learning, Q-learning, agent, AI]
status: growing
level: 4
tier: 2
---

# Reinforcement Learning

## 🟢 1. Simple (age 12)
Reinforcement learning is **learning by trying things and getting rewards**. There is no answer key. The computer tries an action, sees what happens, and gets a score (a **reward**): good things give points, bad things lose points. Over many tries, it learns which actions lead to the most points.

**Example:** Training a puppy to sit. You never explain "sit" in words. The puppy tries things, and when it happens to sit, it gets a treat. Soon it sits whenever you ask. It also learns to look ahead: if turning a corner leads to the treat bowl later, turning the corner is a good move even though the treat is not there yet.

## 🔵 2. How it works
1. There is an **agent** (the learner) and an **environment** (the world it acts in, such as a game or a maze).
2. At each step the agent sees the **state** (where it is, what is on the screen).
3. It chooses an **action** (move left, press a button).
4. The environment changes, and the agent gets a **reward** (a number, often 0) and a new state.
5. The agent wants the **most total reward over time**, not just right now. A small reward now can lose to a big reward later.
6. Early on it must **explore** (try new things to find out what works). Later it should **exploit** (use what it already knows works). Balancing the two is a core problem.
7. Learning means changing its **policy** (its rule for picking actions) so the total reward grows. Many methods use [[Gradient Descent]] on a [[Neural Network]].

![[rl-diagram.svg]]
*The reinforcement learning loop. The agent (robot) takes an **action** in the **environment** (a maze). An **interpreter** turns what happened into a **state** and a **reward**, and both are fed back to the agent. Then it acts again. Image: Megajuice, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Reinforcement_learning_diagram.svg), CC0.*

## 🟠 3. Technical detail
- **The setting (a Markov decision process):** a set of states, a set of actions, rules for how the world changes, and a reward. "Markov" means the next state depends only on the current state and action, not on the longer past.
- **Key words:**
  - **Policy:** the agent's rule for choosing actions.
  - **Return:** the total (discounted) reward from now on.
  - **Value:** the average return you expect from a state, or from a state-and-action pair (the **Q-value**).
  - **Discount factor:** how much future rewards count compared with immediate ones.
- **Two big families:**
  - **Value-based** (such as Q-learning): learn how good each action is, then pick the best one.
  - **Policy-based** (policy gradient): change the policy directly to make rewarded actions more likely. Many modern methods mix both.
- **Deep RL:** use a [[Neural Network]] to estimate the values or the policy when there are too many states to list in a table (for example, game screens).
- **Why it is hard:**
  - Rewards can be **rare and delayed**, so it is unclear which earlier action deserves the credit (the credit assignment problem).
  - Learning is **unstable and data-hungry**, and results can depend heavily on random seeds.
  - **Reward hacking:** the agent finds a way to score points that you did not intend. The reward you write is the goal it chases, not the goal you meant.
- **Where it is used:** games and simulated robots, control problems, recommendation and ad systems (sometimes), and **tuning language models**: [[RLHF]] uses a reward learned from human preferences to adjust an [[LLM]]. An [[Agent]] that takes actions in the world (use tools, browse) can also be trained with rewards.
- **Compared with the others:** [[Supervised Learning]] gets the right answer for each example. [[Unsupervised Learning]] gets no answers at all. Reinforcement learning gets only a **score**, and only after the fact. See also [[Self-Supervised Learning]].

```python
import random
random.seed(0)
N = 5                                   # corridor of 5 cells; start at 0; reward 10 for reaching cell 4
Qt = [[0.0, 0.0] for _ in range(N)]     # Q-table: value of each action (0 = left, 1 = right) in each cell
for ep in range(200):
    s = 0
    while s != N - 1:
        act = random.randrange(2) if random.random() < 0.3 else max((0, 1), key=lambda k: Qt[s][k])   # explore 30% of the time
        s2 = max(0, s - 1) if act == 0 else s + 1
        r = 10 if s2 == N - 1 else 0
        Qt[s][act] += 0.5 * (r + 0.9 * max(Qt[s2]) - Qt[s][act])      # the Q-learning update
        s = s2
print([max((0, 1), key=lambda k: q[k]) for q in Qt[:-1]])   # [1, 1, 1, 1]   learned: always go right
print([round(max(q), 2) for q in Qt[:-1]])                  # [7.29, 8.1, 9.0, 10.0]
```

The agent was never told "go right". It found that out from the reward alone. The final values $7.29,\ 8.1,\ 9.0,\ 10$ are each 0.9 times the next one, because each step away from the goal discounts the reward once more.

## 🔴 4. Deep / formal
**The return: adding up the future.** The agent does not just want the next reward. It wants the total of all future rewards, with later ones counting a bit less:

$$
G_t=\sum_{k=0}^{\infty}\gamma^{k}\,r_{t+k+1}
$$

**Where:**
- $G_t$ is the **return** from time $t$: the total discounted reward the agent will collect from now on.
- $r_{t+k+1}$ is the reward received $k+1$ steps from now. The sum runs over all future steps, $k=0,1,2,\dots$
- $\gamma$ ("gamma") is the **discount factor**, a number between 0 and 1. $\gamma^{k}$ is gamma multiplied by itself $k$ times. **Matters:** with $\gamma$ near 0 the agent is short-sighted and cares only about the next reward. With $\gamma$ near 1 it is far-sighted. Also, for $\gamma<1$ the infinite sum stays finite whenever rewards are bounded, so the goal is well defined.
- **Matters:** this single number $G_t$ is what the agent tries to make big, so it is the formal version of "most points in the long run".

**Example (tiny numbers):** rewards $0,\ 0,\ 10$ over three steps and $\gamma=0.9$.
- Weights: $\gamma^0=1,\ \gamma^1=0.9,\ \gamma^2=0.81$.
- $G=1\cdot0+0.9\cdot0+0.81\cdot10=\mathbf{8.1}$.
- Meaning: a reward of 10 that arrives two steps late is worth only 8.1 today. That is why the corridor values in the code above shrink by a factor of 0.9 per step.

**Q-learning: learning values from experience.** The Q-value $Q(s,a)$ is the expected return from taking action $a$ in state $s$ and then acting well. After every step, the agent nudges its estimate towards what it just saw:

$$
Q(s,a)\leftarrow Q(s,a)+\alpha\Big[\,r+\gamma\max_{a'}Q(s',a')-Q(s,a)\Big]
$$

**Where:**
- $s$ is the state before the move, $a$ is the action taken, $r$ is the reward received, and $s'$ is the state after the move.
- $Q(s,a)$ is the current estimate of how good action $a$ is in state $s$. $\leftarrow$ means "replace with".
- $\max_{a'}Q(s',a')$ is the value of the **best** action available in the new state $s'$. $\gamma$ is the discount factor from above.
- $r+\gamma\max_{a'}Q(s',a')$ is the **target**: what we now think the move was worth (the reward we got, plus the discounted value of where we ended up).
- The bracket is the **error** (target minus current estimate). $\alpha$ ("alpha") is the **learning rate**, between 0 and 1. **Matters:** with $\alpha=0$ nothing is learned. With $\alpha=1$ the old value is thrown away on every step and noise takes over. In between, the estimate moves part of the way each time.
- **Matters:** the target uses the agent's *own* estimate of the next state, so values spread backwards from the reward one step at a time. This is how the goal's reward travels back along the corridor in the code above.

**Example (tiny numbers):** current $Q(s,a)=2$, $\alpha=0.5$, reward $r=1$, $\gamma=0.9$, and the best value in the next state is $\max Q(s',a')=4$.
- Target: $1+0.9\cdot4=\mathbf{4.6}$.
- Error: $4.6-2=\mathbf{2.6}$.
- New value: $2+0.5\cdot2.6=\mathbf{3.3}$.
- Meaning: this move turned out to be worth more than the agent believed, so its estimate rose from 2 to 3.3, half of the way towards 4.6. Repeat over many moves and the estimates settle.

**Further notes:**
- **Convergence:** for small tables, Q-learning is known to converge to the right values if every state-action pair is tried often enough and the learning rate shrinks appropriately (a standard result, stated from memory).
- **Exploration:** a simple choice, used in the code, is to act randomly some fraction of the time (epsilon-greedy). Smarter methods favour actions that are still uncertain.
- **Policy gradient:** instead of learning values, adjust the policy's numbers in the direction that makes high-return actions more likely. This idea sits behind much of the training used for language models (the details of any particular system are not covered here).
- **Reward design:** a badly specified reward gets exploited in surprising ways. Careful reward design and human oversight remain open problems.
- **Offline vs online:** learning only from recorded data (no new trial and error) is safer, but harder.

**What came before, and how this improved on it:**
- **Before:** supervised learning, which needs the right answer for every example.
- **Its limit:** in games, control and chat feedback, nobody can say what the right action was at each step. Only a result arrives, later.
- **What changed:** learn from reward alone, by trying actions and keeping what worked, while still exploring.
- **How much better:** in the code below (three slot machines that win 20%, 50% and 80% of the time, 1,000 pulls, 100 runs), choosing at random earns 0.505 per pull and learning from the rewards (exploring 10% of the time) earns 0.762, against a best possible 0.8. The learner never sees which machine is best; it finds out by playing, and pays for the exploring.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
mean = np.array([0.2, 0.5, 0.8])                                    # three slot machines, win chances unknown to the player
def play(choose, T=1000):
    est, cnt, tot = np.zeros(3), np.zeros(3), 0.0
    for t in range(T):
        a = choose(est, t); r = float(rng.random() < mean[a]); cnt[a] += 1; est[a] += (r - est[a]) / cnt[a]; tot += r
    return tot / T
rand = np.mean([play(lambda e, t: rng.integers(3)) for _ in range(100)])
greedy = np.mean([play(lambda e, t: rng.integers(3) if (rng.random() < 0.1 or t < 3) else int(e.argmax())) for _ in range(100)])
print("average reward per pull: choosing at random", round(rand, 3), " learning from rewards (10% exploring)", round(greedy, 3), " best possible 0.8")
```

## Related
- [[Supervised Learning]] · [[Unsupervised Learning]] · [[Self-Supervised Learning]] · [[Agent]] · [[RLHF]] · [[LLM]] · [[Neural Network]] · [[Gradient Descent]] · [[Probability]] · [[Fine-tuning]] · [[Q-Learning]] · [[Policy Gradient]] · [[PPO]] · [[Reasoning Model]] · [[Machine Learning Specialization]]
- [[Deep Q-Network]] · [[Epsilon-Greedy]] · [[Experience Replay]] · [[Snake DQN]]
## Sources
- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd edition (free online), chapters 1, 3, 6
- Watkins & Dayan, *Q-learning*, 1992 (from memory)
- Mnih et al., *Human-level control through deep reinforcement learning*, 2015 (from memory)
- Ouyang et al., *Training language models to follow instructions with human feedback*, 2022 (RLHF; from memory)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/alignment-rl Reinforcement learning:::Learning by trying things and getting rewards. There is no answer key. The agent tries actions, sees what happens, and learns which actions lead to the most points over many tries.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the discounted return and explain each symbol.
?
$$G_t=\sum_{k=0}^{\infty}\gamma^{k}\,r_{t+k+1}$$
- $G_t$: the return from time $t$, the total discounted reward the agent will collect from now on.
- $r_{t+k+1}$: the reward received $k+1$ steps from now. The sum runs over all future steps.
- $\gamma$: the discount factor, between 0 and 1. A small $\gamma$ cares about now, a $\gamma$ near 1 cares about the far future.
- $\gamma^k$: how much a reward $k$ steps away is shrunk.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the Q-learning update and explain each symbol.
?
$$Q(s,a)\leftarrow Q(s,a)+\alpha\Big[\,r+\gamma\max_{a'}Q(s',a')-Q(s,a)\Big]$$
- $s$: the state before the move. $a$: the action taken. $r$: the reward received. $s'$: the state after the move.
- $Q(s,a)$: the current estimate of how good action $a$ is in state $s$. $\leftarrow$ means "replace with".
- $\max_{a'}Q(s',a')$: the value of the best action available next.
- $r+\gamma\max Q(s',a')$: the target. The bracket is the error between target and current estimate.
- $\alpha$: the learning rate, how far to move towards the target.

#flashcards/ml/cloze #flashcards/topic/alignment-rl The discount factor $\gamma$ makes a reward that arrives later worth ==less== today.

#flashcards/ml/example #flashcards/topic/alignment-rl Rewards 0, 0, 10 over three steps with $\gamma=0.9$. What is the return?::0 + 0.9 × 0 + 0.81 × 10 = 8.1. A reward of 10 that arrives two steps late is worth only 8.1 today.

#flashcards/ml/example #flashcards/topic/alignment-rl $Q(s,a)=2$, $\alpha=0.5$, $r=1$, $\gamma=0.9$, best next value 4. What is the new $Q$?::Target 1 + 0.9 × 4 = 4.6. Error 4.6 − 2 = 2.6. New value 2 + 0.5 × 2.6 = 3.3.

#flashcards/ml/image #flashcards/topic/alignment-rl ![[rl-diagram.svg]]
What does this picture show?
?
The reinforcement learning loop. The agent (robot) takes an action in the environment (a maze). An interpreter turns what happened into a state and a reward, and both are fed back to the agent. Then it acts again.
