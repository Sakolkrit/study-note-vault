---
tags: [DQN, deep-Q-learning, reinforcement-learning, neural-network, Q-learning, game, ML]
status: growing
level: 4
tier: 3
---

# Deep Q-Network

A deep Q-network (DQN) is **Q-learning where the table of scores is replaced by a neural network**. The network looks at the situation and guesses how good each move is. It learns by playing, and two helpers keep the learning steady: a memory of past moves ([[Experience Replay]]) and a slowly changing copy of itself (the target network). The table version is in [[Q-Learning]]; the big picture is in [[Reinforcement Learning]].

## 🟢 1. Simple (age 12)
Imagine you learn a video game by playing it. At first you press buttons at random. After a while you get a feeling: "when the food is to my right and nothing blocks me, going right is a good idea." A DQN has that feeling too, but it is a small **guessing machine** that looks at the screen and gives each button a score. After every game it checks which guesses were wrong and fixes them a little.

**Example:** in the game Snake the guessing machine sees "wall just ahead, food on the left". At the start it gives every move a similar score. After thousands of moves it learns that turning left scores high and going straight scores very low.

## 🔵 2. How it works
1. **Look.** The agent reads the state: the screen, or a list of numbers that describe it.
2. **Guess.** The network turns the state into one score per move. These scores are called Q-values ([[Q-Learning]]).
3. **Choose.** Mostly pick the move with the best score, but now and then pick a random one to discover new things ([[Epsilon-Greedy]]).
4. **Act.** Do the move and see the reward and the new state ([[Markov Decision Process]]).
5. **Remember.** Put the step (state, move, reward, new state, game over?) into a memory ([[Experience Replay]]).
6. **Pick a batch.** Take a random handful of old steps from the memory, not just the newest one.
7. **Make a goal for each step.** The goal is the reward plus a little of the best score the *next* state has, taken from a slowly updated copy of the network, the **target network**.
8. **Correct.** Nudge the network so its guess for that move moves closer to the goal ([[Gradient Descent]], [[Backpropagation]]).
9. **Refresh the copy.** Every few hundred steps, copy the network's weights into the target network. Go back to step 1.

![[deep-rl-challenges-and-tricks.jpg]]
*A chart of popular tricks in deep reinforcement learning and the problem each one answers. Two of them are in a DQN: experience replay (for data that arrives in a row, so each step looks like the one before) and the separated target network (for training that easily diverges). Author: Tsesea, CC BY-SA 4.0, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Challenges_and_Tricks_of_Deep_RL.jpg). I only used it for the two tricks named here; I did not check the other nine against a source.*

## 🟠 3. Technical detail
- **The network.** It takes the state and returns one number per move, so one pass gives the scores of all moves. In my [[Snake DQN]] project the state is 11 numbers, there are two hidden layers of 64 units ([[ReLU]]) and 3 outputs: $11\cdot64+64+64\cdot64+64+64\cdot3+3=5{,}123$ weights (my arithmetic).
- **The goal for one step.** For a step in which the agent was in state $s$, took move $a$, got reward $r$ and landed in $s'$:

$$
y=r+\gamma\,(1-d)\,\max_{a'}Q_{\bar\theta}(s',a')
$$

**Where:**
- $y$ is the goal (the "target"): a plain number that the network is asked to match. No gradient flows through it.
- $r$ is the reward just received. $\gamma$ is the discount between 0 and 1: how much the future counts (0.9 in my project).
- $d$ is 1 if the game ended on this step and 0 otherwise.
- $s'$ is the next state and $a'$ ranges over all moves possible in it.
- $Q_{\bar\theta}$ is the **target network**: a copy of the network whose weights $\bar\theta$ are old and only updated now and then.
- **Matters:** the $(1-d)$ factor stops the agent from counting a future that does not exist after the game is over. Taking the target from a copy keeps the goal still while the main network learns; if the goal came from the network being changed, every update would move its own goal. With $\gamma=0$ the agent only cares about the reward of this step.

**Example (tiny numbers):** $r=1$, $\gamma=0.9$. The target network says the two moves in $s'$ are worth $2$ and $1$ (made up; my arithmetic in Python).
- The best next move is worth $\max(2,1)=2$.
- Game still going ($d=0$): $y=1+0.9\times2=\mathbf{2.8}$.
- If instead the game had ended ($d=1$): $y=1$.
- Meaning: the move is worth its reward 1 plus 90% of the best that can follow.

- **The loss.** The network is trained on a batch of $B$ remembered steps:

$$
\mathcal{L}(\theta)=\frac{1}{B}\sum_{i=1}^{B}\Big(y_i-Q_\theta(s_i,a_i)\Big)^2
$$

**Where:**
- $B$ is the batch size (64 in my project); $i$ is one remembered step in it.
- $y_i$ is its goal from the equation above.
- $Q_\theta(s_i,a_i)$ is the main network's current guess for the move that was actually made in that step; $\theta$ are its weights.
- **Matters:** only the score of the move that was really made is pushed toward its goal; the other moves' scores are left alone for this step. The square makes a big miss count much more than a small one. Averaging over a random batch gives smoother steps than learning from one step.

**Example (tiny numbers):** a batch of two steps (made up): the first has goal $y=2.8$ and the network guesses $0.5$; the second has goal $1.0$ and a guess of $1.5$.
- Errors: $2.8-0.5=2.3$ and $1.0-1.5=-0.5$.
- Squares: $5.29$ and $0.25$.
- $\mathcal{L}=(5.29+0.25)/2=\mathbf{2.77}$.
- Meaning: the first step does almost all the work. The gradient step raises the first guess a lot and lowers the second a little.

- **Code: one step by hand.** The smallest possible "network" is a table stored as the weights of one linear layer (3 states, 2 moves). The goal is from the example above, and I repeat the correction three times with a learning rate of 0.1:

```python
# file: toy_dqn.py
import torch

# 3 states (one-hot) x 2 actions, Q is a table stored as the weights of one linear layer
online, target = torch.nn.Linear(3, 2, bias=False), torch.nn.Linear(3, 2, bias=False)
with torch.no_grad():
    online.weight.copy_(torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]))      # Q(s0,a0) = 0.5, everything else 0
    target.weight.copy_(torch.tensor([[0.0, 2.0, 0.0], [0.0, 1.0, 0.0]]))      # target net: Q(s1,a0) = 2, Q(s1,a1) = 1
s, a, r, s2, gamma = torch.eye(3)[0], 0, 1.0, torch.eye(3)[1], 0.9

with torch.no_grad():
    y = r + gamma * target(s2).max()                  # the target: reward + gamma * best next value (from the TARGET net)
print("target y =", round(float(y), 4))
for step in range(3):
    q = online(s)[a]                                  # the online net's current guess for Q(s0, a0)
    loss = (y - q) ** 2
    online.zero_grad(); loss.backward()
    with torch.no_grad():
        online.weight -= 0.1 * online.weight.grad     # one gradient-descent step, learning rate 0.1
    print(f"step {step}: Q(s0,a0) = {float(q.detach()):.4f}  loss = {float(loss.detach()):.4f}")
print("after 3 steps: Q(s0,a0) =", round(float(online(s)[a].detach()), 4), "(moving towards the target 2.8)")
```

```text
target y = 2.8
step 0: Q(s0,a0) = 0.5000  loss = 5.2900
step 1: Q(s0,a0) = 0.9600  loss = 3.3856
step 2: Q(s0,a0) = 1.3280  loss = 2.1668
after 3 steps: Q(s0,a0) = 1.6224 (moving towards the target 2.8)
```

- **Reading it:** the goal is 2.8, the guess starts at 0.5 and the loss is 5.29, the same as in the example. Each step moves the guess toward 2.8 (0.5, 0.96, 1.328, 1.6224) and the loss falls (5.29, 3.39, 2.17). The target network stays fixed during this, which is the point of having one.
- **What I used in [[Snake DQN]].** Learning rate $10^{-3}$ with [[Adam Optimizer]], $\gamma=0.9$, batch 64, a buffer of 20,000 steps, the target network refreshed every 500 steps, the squared error and gradient clipping at 10. Training started after 1,000 steps. These are my settings and I did not tune them.
- **What the project showed.** The DQN played Snake well (about 15.5 foods against 14.85 for a hand-written rule), the replay buffer made early learning much faster, and removing the target network changed nothing I could see at 60,000 steps.
- **Honest caution.** The 11 numbers in that project take at most 256 different values, so a table of at most $256\times3=768$ scores could have done the same job. The network has 5,123 weights, so it is *bigger* than that table, and I did not run the table. This project shows that a DQN works, not that it was needed.

## 🔴 4. Deep / formal
- **Why Q-learning plus a network is unstable.** Three things together are known to cause trouble (the "deadly triad", from memory): a function approximator, learning from your own guesses (bootstrapping: the goal $y$ contains the network's own estimate), and learning from old data. In a table each cell is updated alone. In a network, changing the weights for one state changes the scores of many other states, including the next state used in the goal, so the goal moves. Replay and the target network are the two practical patches.
- **The two patches in one line each.** Replay breaks the strong link between steps that follow each other and re-uses each step many times ([[Experience Replay]]). The target network freezes the goal for a while.
- **Overestimation.** The goal takes a $\max$ over noisy estimates, and the maximum of noisy numbers is on average too high. This bias builds up. *Double DQN* picks the best move with one network and scores it with the other (from memory, not run here).
- **Discrete moves only.** The $\max$ over moves is easy for 3 buttons and hard for a continuous force. For continuous actions, [[Policy Gradient]] methods or actor-critic methods are used instead (from memory).
- **Time limits.** I ended a game after 100 steps without food and treated that as "game over" with reward $-1$ ($d=1$). Strictly a time limit is not a real end of the game, and a careful implementation lets the goal still include the future in that case. I did not do this, and no test game ended that way.
- **Reward scale.** The squared error reacts strongly to large rewards. Published DQN work clips rewards and uses a smoother loss (Huber) for this reason (from memory); I used rewards of $-1$, $0$, $+1$ and the plain squared error.
- **Not on-policy.** Because the memory holds old steps made by older versions of the agent, DQN is an *off-policy* method, like Q-learning in [[Q-Learning]], and unlike [[PPO]].
- **Sample efficiency and cost.** A DQN for pixel games needs millions of steps (from memory, not run here). My small one needed tens of thousands and about 485 seconds per run on a laptop CPU with six runs at once.

**What came before, and how this improved on it:**
- **Before:** tabular Q-learning (introduced by Chris Watkins in 1989, from memory) kept one number per state and move. Networks had been tried with reinforcement learning before (for example TD-Gammon for backgammon in the early 1990s, from memory).
- **Its limit:** a table needs a cell for every state, and a game screen has far too many; plain Q-learning with a network was unstable.
- **What changed:** a network that generalises across similar states, trained with an experience-replay memory and a target network; DeepMind showed this on Atari games in 2013 and in *Nature* in 2015 (from memory, not checked, I have not read the papers).
- **How much better:** the published claim is human-level play on many of 49 Atari games (from memory, not checked). In my own code, the replay buffer helped early learning a lot: after 5,000 steps the full DQN averaged 15.1 foods and the version without replay 4.9 (30 test games, mean of 5 runs), while all versions ended at about 15.5. No number here compares a DQN with a table, because I did not run the table.

## Related
- [[Q-Learning]] · [[Reinforcement Learning]] · [[Markov Decision Process]] · [[Bellman Equation]] · [[Experience Replay]] · [[Epsilon-Greedy]] · [[Neural Network]] · [[Gradient Descent]] · [[Backpropagation]] · [[Adam Optimizer]] · [[Loss Function]] · [[Learning Rate]] · [[ReLU]] · [[PyTorch]] · [[Convolutional Neural Network]] · [[Policy Gradient]] · [[PPO]] · [[Snake DQN]] · [[Q-Learning Grid World]]
- [[Looking Inside Snake DQN]] · [[Interpretability]]
## Sources
- My own toy (PyTorch 2.8.0, NumPy 1.24.4, Python 3.9.6) and the [[Snake DQN]] project. The history (Watkins 1989, TD-Gammon, the Atari DQN papers of 2013 and 2015), the "deadly triad", double DQN, reward clipping and the Huber loss are from memory. Not read: the DQN papers.
- Picture: Tsesea, CC BY-SA 4.0, [File:Challenges_and_Tricks_of_Deep_RL.jpg](https://commons.wikimedia.org/wiki/File:Challenges_and_Tricks_of_Deep_RL.jpg), Wikimedia Commons.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/alignment-rl Deep Q-network (DQN):::Q-learning with a neural network instead of a table: the network takes a state and gives a score (Q-value) for each move; it learns with an experience-replay memory and a target network.

#flashcards/ml/reverse #flashcards/topic/alignment-rl Target network:::A copy of the Q-network with older weights, refreshed only every so often. It is used to make the goals, so the goal does not move with every update.

#flashcards/ml/concept #flashcards/topic/alignment-rl What are the steps of a DQN?
?
Look at the state; the network gives a score per move; choose a move (mostly the best, sometimes random); act and see the reward and next state; store the step in the memory; pick a random batch from the memory; make a goal from the reward plus the discounted best next score (from the target network); nudge the network toward the goals; now and then copy the network into the target network.

#flashcards/ml/formula #flashcards/topic/alignment-rl What is the goal (target) for one remembered step in a DQN?
?
$$y=r+\gamma\,(1-d)\,\max_{a'}Q_{\bar\theta}(s',a')$$
- $y$: the goal, a plain number the network must match (no gradient through it).
- $r$: reward just received. $\gamma$: discount between 0 and 1.
- $d$: 1 if the game ended on this step, else 0, so no future is counted after the end.
- $Q_{\bar\theta}$: the target network (older weights), $s'$ the next state, $a'$ any next move.
- Matters: the copy keeps the goal still while the main network learns; with $\gamma=0$ only this step's reward counts.

#flashcards/ml/example #flashcards/topic/alignment-rl $r=1$, $\gamma=0.9$, target network scores of the next state are 2 and 1. What is the goal $y$, and what if the game ended?::The best next score is 2, so $y=1+0.9\times2=2.8$. If the game had ended, $d=1$ and $y=1$.

#flashcards/ml/formula #flashcards/topic/alignment-rl What is the DQN loss on a batch?
?
$$\mathcal{L}(\theta)=\frac{1}{B}\sum_{i=1}^{B}\Big(y_i-Q_\theta(s_i,a_i)\Big)^2$$
- $B$: batch size (64 in my project); $i$: one remembered step.
- $y_i$: its goal. $Q_\theta(s_i,a_i)$: the network's guess for the move actually made. $\theta$: its weights.
- Matters: only the move really made is corrected. The square makes big misses count more; averaging a random batch gives smoother steps.

#flashcards/ml/example #flashcards/topic/alignment-rl Batch of two steps: goal 2.8 with guess 0.5, and goal 1.0 with guess 1.5. What is the loss?::Errors 2.3 and −0.5, squares 5.29 and 0.25, loss $(5.29+0.25)/2=2.77$. The first step does almost all the work.

#flashcards/ml/image #flashcards/topic/alignment-rl ![[deep-rl-challenges-and-tricks.jpg]]
What does this chart show, and which two tricks does a DQN use?
?
A chart of popular tricks in deep reinforcement learning and the problem each one answers. A DQN uses experience replay (for data that arrives in a row) and a separated target network (for easy divergence). I did not check the other tricks against a source. (Tsesea, CC BY-SA 4.0, Wikimedia Commons.)

#flashcards/ml/concept #flashcards/topic/alignment-rl Why does Q-learning with a neural network become unstable, and what are the two usual patches?
?
A network, bootstrapping (the goal contains the network's own guess) and old data together can make the goal move and training spiral (the "deadly triad", from memory). Patches: experience replay (random old steps, each used many times) and a target network (a frozen goal for a while).

#flashcards/ml/concept #flashcards/topic/alignment-rl Why does the max in the DQN goal cause overestimation, and what is the fix?
?
The maximum of several noisy estimates is on average too high, and the bias builds up. Double DQN chooses the best move with one network and scores it with the other (from memory, not run here).

#flashcards/ml/concept #flashcards/topic/alignment-rl What did the by-hand toy show about the DQN update?
?
With goal 2.8 and a start guess of 0.5, three steps at learning rate 0.1 moved the guess to 0.96, 1.328 and 1.6224, and the loss fell from 5.29 to 3.39 to 2.17. The goal stayed fixed because it came from the target network.

#flashcards/ml/concept #flashcards/topic/alignment-rl What network and settings did the Snake DQN project use?
?
11 inputs, two hidden layers of 64 ReLU units, 3 outputs (5,123 weights, my arithmetic). Adam with learning rate 0.001, discount 0.9, batch 64, buffer 20,000, target refreshed every 500 steps, squared error, gradient clipping at 10. Not tuned.

#flashcards/ml/concept #flashcards/topic/alignment-rl What did the Snake DQN project show, and what did it not show?
?
The DQN reached about 15.5 foods against 14.85 for a rule; replay made early learning faster (15.1 against 4.9 at 5,000 steps) and the target network showed no effect at 60,000 steps. It did not show that a network was needed: the 11 numbers take at most 256 values, so a table of 768 scores could work; I did not run it.

#flashcards/ml/concept #flashcards/topic/alignment-rl Why is a DQN usually for discrete moves only?
?
The goal needs a max over all moves, which is easy for a few buttons and hard for a continuous force. Continuous actions are usually handled with policy-gradient or actor-critic methods (from memory).

#flashcards/ml/concept #flashcards/topic/alignment-rl What came before DQN and which facts are from memory?
?
Tabular Q-learning (Watkins 1989) and early networks with reinforcement learning such as TD-Gammon; DeepMind's Atari DQN in 2013 and 2015 with the claim of human-level play on many of 49 games. All of this history is from memory; I have not read the papers.

#flashcards/ml/cloze #flashcards/topic/alignment-rl A DQN uses a ==target network== to keep the goal still and an ==experience replay== memory to learn from random old steps.
