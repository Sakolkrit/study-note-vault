---
tags: [machine-learning, reinforcement-learning, Q-learning, agent, AI]
status: growing
level: 4
tier: 3
---

# Q-Learning

Q-learning is a way for an agent to **learn by trial and error how good each move is in each situation**, by keeping a table of scores and nudging each score toward what it just experienced. The big picture of learning from rewards is in [[Reinforcement Learning]], which also has a first small Q-learning example. This note goes further into why the update works, how exploring changes the answer, and how the table becomes a neural network.

## 🟢 1. Simple (age 12)
Imagine you are in a new school building and want to find the fastest way to the canteen. Every time you try a route, you write on a notepad how good that turn was: "left at the stairs: bad, ended up in the library", "right at the stairs: good, smelled lunch". After many days, your notepad tells you the best turn from every spot in the building. Q-learning is that notepad. Nobody tells you the route. You find it by trying turns and remembering how they turned out.

**Example:** A robot vacuum starts with a blank score for every direction in every part of a room. It bumps into a table (bad score) or finds the charging dock (good score). It quietly updates the scores in its notebook each time. After a few hundred tries, in each spot it just looks up the best direction and goes.

## 🔵 2. How it works
1. **Make the notepad.** List every situation (a **state**) and every move (an **action**). Give each pair a score, called its **Q-value**, starting at 0.
2. **Pick a move.** Usually choose the move with the best score. Now and then (say 1 time in 10) pick a random one instead, so you keep discovering things. This is called **epsilon-greedy** ([[Epsilon-Greedy]]).
3. **Make the move and look.** See the **reward** you get (for example −1 for each step taken, or +10 for reaching the goal) and the new situation you are in.
4. **Make a better guess for the old score.** The new guess is the reward you got, plus the best score you can now see from the new situation (shrunk a little, because future rewards count less).
5. **Move the old score part of the way toward the new guess.** Not all the way, since one try can be unlucky.
6. **Repeat** for thousands of moves. Good news spreads backward: a square next to the goal gets a good score first, then the square before it, and so on.
7. **Use it:** in each situation, pick the move with the highest score.

![[markov-decision-process.svg]]
*The kind of world Q-learning works in, called a Markov decision process. The green circles are states. The orange circles are actions you can take in a state. The numbers on the arrows are the chances of ending up in each next state. The wavy arrows show rewards (+5 and −1). Image: waldoalvarez, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Markov_Decision_Process.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **The Q-table:** a grid with one row per state and one column per action. The cell holds the learned value of taking that action in that state. Training fills it in from zeros.
- **Off-policy:** Q-learning learns the value of the *best* action in the next state, even if the agent's behaviour (with its random exploring) does not always take it. This lets it learn about the best way to behave while behaving a bit differently. The cousin method **SARSA** uses the action the agent *actually* takes next. That is called on-policy. The difference matters (shown in the code).
- **Exploration:** pure "always pick the best" can get stuck on a mediocre route. Random exploring (epsilon-greedy) fixes that. Often epsilon starts high and shrinks over time (not shown here).
- **The learning rate $\alpha$ and discount $\gamma$:** $\alpha$ is how far each update moves the score, and $\gamma$ is how much the future counts. See [[Learning Rate]].
- **It needs a table:** the table must have a cell for every state-action pair. For a game with images as states, there are far too many. **Deep Q-learning** ([[Deep Q-Network]]) replaces the table with a [[Neural Network]] that takes a state and outputs a Q-value for each action (equation 3 below). Two helpers make it more stable: a *replay buffer* ([[Experience Replay]]; train on old experiences picked at random, not only the latest ones) and a *target network* (a slowly updated copy used to make the targets) (from memory, not shown in code).
- **Overestimation:** because each update uses a *maximum* over estimates that contain noise, the values drift upward (the last line of the code shows the effect). A fix called *double Q-learning* keeps two estimates and uses one to pick the action and the other to score it (from memory).
- **What it cannot do well:** it needs small action sets (it must take the max over actions), and learning can be slow and unstable with a neural network. Methods that adjust the policy directly (see [[Reinforcement Learning]]) cover problems like continuous actions, and are what is behind much of [[RLHF]] (the systems differ; check each one).

![[q-learning-table-before-after.png]]
*A real Q-table before and after training. Each row is a state (here numbered 0 to 499) and each column is an action (South, North, East, West, Pickup, Dropoff). Top: every cell starts at 0. Bottom: after training, some rows have learned numbers, and the biggest number in a row marks the best action there (for state 499 it is West, 29). Rows never visited, like state 0, stay at 0. Image: LearnDataSci, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Q-Learning_Matrix_Initialized_and_After_Training.png), CC BY-SA 4.0.*

The code is the classic "cliff walking" world. The grid has 4 rows and 12 columns. You start at the bottom left and the goal is at the bottom right. Every step costs −1. The bottom row between them is a cliff. Stepping on it costs −100 and sends you back to the start. The agent moves randomly 10% of the time while learning, so walking right beside the cliff is risky for it. The code trains both Q-learning and SARSA and compares them.

```python
import numpy as np
H, W = 4, 12                                   # cliff walking: start bottom-left, goal bottom-right, cliff in between
START, GOAL = (3, 0), (3, 11)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]     # up, down, left, right
def step(s, a):
    r, c = s[0] + MOVES[a][0], s[1] + MOVES[a][1]
    r, c = min(max(r, 0), H - 1), min(max(c, 0), W - 1)
    if r == 3 and 0 < c < 11: return START, -100, False    # fell off the cliff: back to start
    return (r, c), -1, (r, c) == GOAL
def train(method, episodes=1000, alpha=0.5, gamma=1.0, eps=0.1, seed=0):
    rng = np.random.default_rng(seed); Q = np.zeros((H, W, 4)); returns = []
    pick = lambda s: int(rng.integers(4)) if rng.random() < eps else int(np.argmax(Q[s]))
    for _ in range(episodes):
        s, a, total, done = START, pick(START), 0, False
        while not done:
            s2, r, done = step(s, a); a2 = pick(s2); total += r
            nxt = Q[s2].max() if method == 'q' else Q[s2][a2]       # Q-learning: best next action. SARSA: the action it will really take.
            Q[s][a] += alpha * (r + gamma * (0 if done else nxt) - Q[s][a])
            s, a = s2, a2
        returns.append(total)
    return Q, float(np.mean(returns[-100:]))
def greedy_path(Q):
    s, path = START, [START]
    for _ in range(40):
        s, _, done = step(s, int(np.argmax(Q[s]))); path.append(s)
        if done: return path
    return path
for m in ('q', 'sarsa'):
    runs = [train(m, seed=sd) for sd in range(5)]
    p = greedy_path(runs[0][0]); rows = [s[0] for s in p]
    print(m, '| online return, last 100 episodes (mean of 5 seeds):', round(float(np.mean([r for _, r in runs])), 1), '| greedy path length (seed 0):', len(p) - 1)
    print('  path rows (2 = right beside the cliff, 0 = top row):', rows)
# the max over noisy estimates is biased upward: two actions that are both worth 0
rng = np.random.default_rng(0); noisy = rng.normal(0, 1, (1_000_000, 2))
print('mean of max over 2 noisy estimates of value 0:', round(float(noisy.max(1).mean()), 3), '(true value 0)')
# q | online return, last 100 episodes (mean of 5 seeds): -41.6 | greedy path length (seed 0): 13
#   path rows (2 = right beside the cliff, 0 = top row): [3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3]
# sarsa | online return, last 100 episodes (mean of 5 seeds): -24.0 | greedy path length (seed 0): 17
#   path rows (2 = right beside the cliff, 0 = top row): [3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3]
# mean of max over 2 noisy estimates of value 0: 0.565 (true value 0)
```
Reading the output (NumPy only; seeds fixed, so it should repeat exactly):
- **Q-learning** learns the shortest route, 13 steps, running along the row right next to the cliff (row 2). That is the best route if you never slip.
- But while it is *learning* (moving randomly 10% of the time), it sometimes falls off the cliff, so its average return over the last 100 training episodes is −41.6.
- **SARSA** learns a longer, safer route, 17 steps, going up to the top row (row 0) and coming back down. It learned to keep away from the cliff because it counts its own random slips in its estimates. Its average training return is −24.0, better than Q-learning's while exploring.
- If you turned the exploring off, Q-learning's route would be the better one. Which method is "better" depends on whether the agent's mistakes during learning matter.
- This is one small world and one setting, so treat the −41.6 and −24.0 as an illustration. With only 500 training episodes SARSA's greedy policy had not yet settled (in a trial run it got stuck and never reached the goal), so the code uses 1,000.
- The last line shows the overestimation effect: the average of the better of two noisy guesses is 0.565 when the true value is 0.

## 🔴 4. Deep / formal

**The Bellman optimality equation.** Write $Q^*(s,a)$ for the value of taking action $a$ in state $s$ and then acting as well as possible forever. It satisfies

$$Q^*(s,a)=\mathbb{E}\Big[\,r+\gamma\max_{a'}Q^*(s',a')\;\Big|\;s,a\Big]$$

**Where:**
- $Q^*(s,a)$ is the best possible value of the pair (state, action). **Matters:** it is the thing Q-learning is trying to find. If you know it, you just pick the best action in each state.
- $r$ is the reward for this step and $s'$ the state you land in. $\mathbb{E}$ is the average over where you might land (and the reward you might get), since the world can be random. **Matters:** the world's randomness is handled by averaging many tries, which is why Q-learning moves its score only part of the way each time.
- $\gamma$ is the discount between 0 and 1. **Matters:** it makes later rewards count less, and keeps the total finite.
- $\max_{a'}Q^*(s',a')$ is the value of the best next move. **Matters:** it assumes you will play well from the next step on. This "best next move" is what separates Q-learning from SARSA.
- The equation says the value now equals the immediate reward plus the discounted value of the best future. **Matters:** this is a *self-consistency* condition. It is what lets values spread backward from the goal one step at a time.

**Example (tiny numbers):** From state A, the action "go" gives reward 1 and lands in state B. In B, one move is worth 4 and another is worth 6. A different action at A, "safe", gives a plain reward of 2 and ends. Use $\gamma=0.9$.
1. Best move in B: $\max(4,6)=6$.
2. $Q^*(A,\text{go})=1+0.9\times6=6.4$.
3. $Q^*(A,\text{safe})=2$.
4. Meaning: "go" is worth 6.4 and "safe" only 2, so the best policy goes toward B even though the first step pays less. The score of a step includes what it leads to.

**Why the two methods differ.** The update is $Q(s,a)\leftarrow Q(s,a)+\alpha\,[\text{target}-Q(s,a)]$, and the only difference is the target:

$$\text{target}_{\text{Q-learning}}=r+\gamma\max_{a'}Q(s',a'),\qquad \text{target}_{\text{SARSA}}=r+\gamma\,Q(s',a')\ \ \text{with } a' \text{ the action actually taken}$$

**Where:**
- $r$, $s'$ and $\gamma$ are as above. The brackets $[\text{target}-Q(s,a)]$ is the error, and $\alpha$ is the learning rate (see [[Learning Rate]]). **Matters:** both methods nudge toward their target in the same way.
- $\max_{a'}$ takes the best next action in the table, whatever the agent will really do. **Matters:** Q-learning therefore learns the value of the *greedy* policy, while behaving otherwise (off-policy).
- $a'$ in SARSA is the next action actually chosen, including random ones. **Matters:** SARSA's values include the cost of its own random mistakes, which is why it learns to stay away from the cliff.

**Example (tiny numbers):** The agent stands one step from the cliff edge. From the next state the best action is worth $-3$. The one action that steps off the cliff is worth $-103$. The other two actions are worth $-3$ as well. The reward just received is $r=-1$, $\gamma=1$. The agent explores with $\varepsilon=0.1$ over 4 actions, so the chance of picking the cliff action is $0.1/4=0.025$.
1. Q-learning target: $-1+(-3)=-4$.
2. SARSA target, averaged over its choices: $-1+0.975\times(-3)+0.025\times(-103)=-1-2.925-2.575=-6.5$.
3. Meaning: SARSA's target is 2.5 lower because it admits that 1 time in 40 it will fall in. So the squares near the cliff look worse to SARSA and it picks the safe route. Q-learning ignores its own clumsiness and picks the shortest one. That is the pattern the code showed.

**Deep Q-learning loss.** When the table is replaced by a network $Q_\theta(s,a)$ with weights $\theta$, the update becomes one step of [[Gradient Descent]] on

$$\mathcal{L}(\theta)=\Big(r+\gamma\max_{a'}Q_{\bar\theta}(s',a')-Q_\theta(s,a)\Big)^2$$

**Where:**
- $Q_\theta(s,a)$ is the network's current estimate for the move that was made. **Matters:** this is what the gradient changes.
- $Q_{\bar\theta}$ is a *target network*: a copy of the same network with older weights $\bar\theta$, updated only now and then. **Matters:** if the target used the same weights that are being changed, every update would move its own goal and training can spiral (the reason is the usual explanation; from memory).
- $r+\gamma\max_{a'}Q_{\bar\theta}(s',a')$ is the target, treated as a fixed number (no gradient goes through it). **Matters:** this is what turns a reinforcement problem into a regression problem, with a squared-error [[Loss Function]].
- The square. **Matters:** a big error gets a big push and a small error a small one.

**Example (tiny numbers):** $r=1$, $\gamma=0.9$. The target network says the next state's actions are worth $(2,4)$, and the main network says $Q_\theta(s,a)=3$.
1. Target: $1+0.9\times\max(2,4)=1+3.6=4.6$.
2. Error: $4.6-3=1.6$.
3. Loss: $1.6^2=2.56$.
4. Meaning: the network underrates this move (3 against 4.6), so a gradient step raises $Q_\theta(s,a)$ toward 4.6. This is the tabular update from [[Reinforcement Learning]] (with the table replaced by weights).

**Convergence and caveats (from memory, not checked here):**
- For a table, Q-learning is known to converge to $Q^*$ if every state-action pair is visited often enough and the learning rate shrinks in a suitable way. This is a standard result I have not re-proved or tested beyond the small worlds here.
- With a neural network, that guarantee no longer holds. Learning can be unstable, which is why the replay buffer and target network exist.
- The upward drift from taking a max of noisy estimates is a known effect, and double Q-learning is the usual remedy. The code only shows the size of the effect for two actions.

**What came before, and how this improved on it:**
- **Before:** planning methods that need a full model of the world (how every action changes the state), or acting at random.
- **Its limit:** most real problems have no such model, and random acting rarely finds the goal.
- **What changed:** learn the value of each action in each state from experience, using the best next value as the target (Watkins, 1989; from memory, not checked).
- **How much better:** in the code below (a corridor of 6 cells with the goal at the end), a random walk takes 30.6 steps on average to reach the goal. After 300 practice runs, the greedy policy takes 5 steps, which is the shortest possible. Tiny table, no model of the corridor given to the learner.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
n = 6                                                                # states 0..5, goal at 5, start at 0, actions: left/right
def step(s, a): s2 = max(0, min(n - 1, s + (1 if a else -1))); return s2, (1.0 if s2 == n - 1 else 0.0)
def episode(policy):
    s, t = 0, 0
    while s != n - 1 and t < 1000: s, _ = step(s, policy(s)); t += 1
    return t
print("random walk: average steps to the goal", round(np.mean([episode(lambda s: rng.integers(2)) for _ in range(500)]), 1))
Q = np.zeros((n, 2))
for _ in range(300):
    s = 0
    while s != n - 1:
        a = rng.integers(2) if rng.random() < 0.3 else int(Q[s].argmax())
        s2, r = step(s, a); Q[s, a] += 0.5 * (r + 0.9 * Q[s2].max() - Q[s, a]); s = s2
print("after 300 practice runs, greedy steps to the goal:", episode(lambda s: int(Q[s].argmax())), "(shortest possible: 5)")
```

## Related
[[Reinforcement Learning]] · [[Agent]] · [[RLHF]] · [[Neural Network]] · [[Gradient Descent]] · [[Learning Rate]] · [[Loss Function]] · [[Probability]] · [[Supervised Learning]] · [[Backpropagation]] · [[Policy Gradient]] · [[PPO]] · [[Recursion]] · [[Dynamic Programming]] · [[Dijkstra's Algorithm]] · [[Machine Learning Specialization]] · [[Q-Learning Grid World]] · [[Deep Q-Network]] · [[Epsilon-Greedy]] · [[Experience Replay]] · [[Snake DQN]] · [[Looking Inside Snake DQN]]
## Sources
- Watkins and Dayan (1992), "Q-learning" (the algorithm and its convergence), and Mnih and colleagues (2015), the deep Q-network paper (both from memory; I did not open them for this note).
- Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition, which contains the cliff-walking comparison of Q-learning and SARSA (from memory; the numbers here are from my own run, not from the book).
- Figures: [Markov_Decision_Process.svg](https://commons.wikimedia.org/wiki/File:Markov_Decision_Process.svg) (waldoalvarez) and [Q-Learning_Matrix_Initialized_and_After_Training.png](https://commons.wikimedia.org/wiki/File:Q-Learning_Matrix_Initialized_and_After_Training.png) (LearnDataSci), both CC BY-SA 4.0, on Wikimedia Commons.
- The code and every Example number were run for this note (NumPy only).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/alignment-rl Q-learning:::A notepad (Q-table) of how good each action is in each state. The agent tries things, and nudges each entry toward reward plus the best value of where it landed. Nobody tells it the route.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the Q-learning optimality equation and explain each symbol.
?
$$Q^*(s,a)=\mathbb{E}\Big[\,r+\gamma\max_{a'}Q^*(s',a')\;\Big|\;s,a\Big]$$
- $Q^*(s,a)$: the best possible value of the pair (state, action). Knowing it, you pick the best action in every state.
- $r$: the reward for this step. $s'$: the state you land in. $\mathbb{E}$: the average over where you might land, since the world can be random. That is why Q-learning moves its score only part of the way each time.
- $\gamma\in(0,1)$: the discount, so later rewards count less.
- $\max_{a'}$: the best next action.

#flashcards/ml/example #flashcards/topic/alignment-rl From A, "go" gives reward 1 and lands in B, where moves are worth 4 and 6. "safe" gives 2 and ends. $\gamma=0.9$. What are $Q^*(A,\text{go})$ and $Q^*(A,\text{safe})$?::Best move in B: 6. $Q^*(A,\text{go})=1+0.9\times6=6.4$. $Q^*(A,\text{safe})=2$. "go" is worth more.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the Q-learning and SARSA targets and explain each symbol.
?
$$\text{target}_{\text{Q-learning}}=r+\gamma\max_{a'}Q(s',a'),\qquad \text{target}_{\text{SARSA}}=r+\gamma\,Q(s',a')\ \ \text{with } a' \text{ the action actually taken}$$
- $r$, $s'$, $\gamma$ as above. The error is $[\text{target}-Q(s,a)]$, scaled by the learning rate $\alpha$. Both methods nudge toward their target the same way.
- $\max_{a'}$: the best next action in the table, whatever the agent will really do. Q-learning therefore learns the value of the greedy policy while behaving otherwise (off-policy).
- $a'$ in SARSA: the next action actually chosen, including random ones. SARSA's values include the cost of its own random mistakes, so it learns to stay away from the cliff.

#flashcards/ml/example #flashcards/topic/alignment-rl One step from a cliff edge: best next action worth −3, the cliff action −103, $r=-1$, $\gamma=1$, $\varepsilon=0.1$ over 4 actions. What are the two targets?::Chance of the cliff action $0.1/4=0.025$. Q-learning: $-1+(-3)=-4$. SARSA averaged: $-1+0.975(-3)+0.025(-103)=-6.5$. SARSA's target is 2.5 lower because it admits it falls in 1 time in 40, so it picks the safe route. Q-learning ignores its own clumsiness and picks the shortest.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the deep Q-network (DQN) loss and explain each symbol.
?
$$\mathcal{L}(\theta)=\Big(r+\gamma\max_{a'}Q_{\bar\theta}(s',a')-Q_\theta(s,a)\Big)^2$$
- $Q_\theta(s,a)$: the network's current estimate for the move made. The gradient changes this.
- $Q_{\bar\theta}$: a target network, a copy with older weights $\bar\theta$ updated only now and then. If the target used the weights being changed, every update would move its own goal and training can spiral (the usual explanation, from memory).
- $r+\gamma\max_{a'}Q_{\bar\theta}(s',a')$: the target, treated as a fixed number (no gradient through it). This turns a reinforcement problem into a regression problem.

#flashcards/ml/example #flashcards/topic/alignment-rl $r=1$, $\gamma=0.9$, target network says next actions are worth (2, 4), main network says $Q_\theta(s,a)=3$. What is the loss?::Target $1+0.9\times4=4.6$. Error $4.6-3=1.6$. Loss $1.6^2=2.56$. The network underrates this move, so a gradient step raises $Q_\theta(s,a)$.

#flashcards/ml/image #flashcards/topic/alignment-rl ![[markov-decision-process.svg]]
What does this picture show?
?
The kind of world Q-learning works in, a Markov decision process. The green circles are states, the orange circles are actions, the numbers on the arrows are chances of ending up in each next state, and the wavy arrows show rewards (+5 and −1). (From the note's caption: waldoalvarez, Wikimedia Commons.)

#flashcards/ml/image #flashcards/topic/alignment-rl ![[q-learning-table-before-after.png]]
What does this picture show?
?
A real Q-table before and after training. Each row is a state (0 to 499), each column an action (South, North, East, West, Pickup, Dropoff). Top: every cell starts at 0. Bottom: after training some rows hold learned numbers, and the biggest in a row marks the best action (for state 499 it is West, 29). Unvisited rows like state 0 stay at 0. (From the note's caption: LearnDataSci, Wikimedia Commons.)

#flashcards/ml/cloze #flashcards/topic/alignment-rl Q-learning is ==off-policy==: it learns the value of the ==greedy== policy while behaving differently.
