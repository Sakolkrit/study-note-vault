---
tags: [reinforcement-learning, MDP, Markov, decision-making, AI]
status: growing
level: 4
tier: 2
---

# Markov Decision Process

Often written **MDP**. The standard way to describe a problem where an agent makes choices, the world reacts a bit randomly, and rewards come along the way.

## 🟢 1. Simple (age 12)
An MDP is a **board game written down as a list of rules**. It says where you can be, what moves you can make, where each move might take you (with some luck involved), and how many points you get. The one big rule: **only where you are now matters**, not how you got there.

**Example:** A board game piece sits on square 4. You may "walk" (usually move 1 square) or "run" (move 2 squares, but half the time you trip and go back to the start). Which move is best depends only on square 4, not on the squares you visited before.

## 🔵 2. How it works
An MDP has five parts:
1. **States** ($S$): the situations you can be in (a square on the board, a position in a maze).
2. **Actions** ($A$): the moves you can pick in a state.
3. **Transitions** ($P$): for each state and action, the chance of landing in each next state. This is where luck lives. The word "Markov" means these chances depend only on the **current state and action**, not on the longer past. (See [[Markov Chain]]: an MDP is a Markov chain where you get to steer.)
4. **Rewards** ($R$): the points you get for a move. They can be negative (a cost).
5. **Discount** ($\gamma$): a number between 0 and 1 that says later points count a bit less than points now.

The agent's job is to find a **policy**: a rule that says which action to pick in each state, so the total reward is as large as possible. This is the problem that [[Reinforcement Learning]] and [[Q-Learning]] try to solve, usually without being told the transition chances. The [[Agent]] only learns them by trying.

![[markov-decision-process.svg]]
*A small MDP. The green circles are states. The orange circles are actions you can take in a state. The numbers on the arrows are the chances of ending up in each next state. The wavy arrows show rewards (+5 and −1). Image: waldoalvarez, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Markov_Decision_Process.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **The tuple:** an MDP is written $(S, A, P, R, \gamma)$.
- **Policy** $\pi(a\mid s)$: the chance of choosing action $a$ in state $s$. A fixed single choice per state is a *deterministic* policy.
- **Return:** the discounted sum of rewards from now on. The agent wants a high *expected* return ([[Probability]] and averages, since the world is random).
- **Value of a state under a policy,** $V^\pi(s)$: the expected return if you start in $s$ and follow $\pi$. It obeys a self-consistency rule, the Bellman equation (see [[Bellman Equation]] and [[Q-Learning]], which uses the "best action" version).
- **Solving a known MDP** ([[Dynamic Programming]]): *value iteration* repeats "new value = best action's reward plus discounted value of where it leads" until the numbers stop changing. *Policy iteration* alternates between evaluating a policy and improving it.
- **Not known?** Then you cannot solve it directly. You learn from experience: [[Q-Learning]] (learn action values), [[Policy Gradient]] (learn the policy itself), and [[RLHF]] (the same idea used to train language models, with a [[Reward Model]] giving the reward).
- **Terminal states:** a goal where the game ends. In code it is a state that stays on itself with zero reward.

```python
import itertools
import numpy as np

# 1) the return: add up rewards, later ones shrunk by gamma
gamma = 0.9
rewards = [1, 1, 1, 1, 1]
G = sum(gamma**k * r for k, r in enumerate(rewards))
print("return of five 1-rewards, gamma 0.9:", round(G, 4))

# 2) one backup: action 'go' lands in X (value 10) with 0.8, in Y (value 0) with 0.2, reward 1
print("one-step value of 'go':", 1 + gamma * (0.8 * 10 + 0.2 * 0))

# 3) a small MDP: 6 states, 2 actions. P[a][s] = next-state probabilities, R[s][a] = reward. State 5 is the goal (stays there, no more reward).
n, nA = 6, 2
P = np.zeros((nA, n, n)); R = np.zeros((n, nA))
for s in range(n - 1):
    P[0, s, min(s + 1, n - 1)] = 0.9; P[0, s, s] = 0.1          # 'walk': slow, safe
    P[1, s, min(s + 2, n - 1)] = 0.5; P[1, s, 0] = 0.5          # 'run': fast, but may fall back to the start
    R[s, :] = -1                                                   # every step costs 1
P[:, n - 1, n - 1] = 1

def evaluate(pi):                                               # solve V = R_pi + gamma P_pi V exactly
    Ppi = np.array([P[pi[s], s] for s in range(n)]); Rpi = np.array([R[s, pi[s]] for s in range(n)])
    return np.linalg.solve(np.eye(n) - gamma * Ppi, Rpi)

# brute force: try every policy (2^6 = 64) and keep the best start value
best = max(itertools.product(range(nA), repeat=n), key=lambda pi: evaluate(pi)[0])
print("policies tried by brute force:", nA**n, "| best policy:", best, "| V(start) =", round(evaluate(best)[0], 3))

# value iteration: repeat V(s) <- max_a [ R + gamma * sum P V ]
V = np.zeros(n); sweeps = 0
while True:
    Vn = np.max(R + gamma * np.einsum('ast,t->sa', P, V), axis=1); sweeps += 1
    if np.abs(Vn - V).max() < 1e-9: break
    V = Vn
pi_vi = tuple(int(x) for x in np.argmax(R + gamma * np.einsum('ast,t->sa', P, V), axis=1))
print("value iteration sweeps:", sweeps, "| policy:", pi_vi, "| V(start) =", round(V[0], 3))
print("same value as brute force:", abs(V[0] - evaluate(best)[0]) < 1e-6)

# how brute force grows: S states, 2 actions -> 2^S policies, each needs a linear solve
for S in (6, 20, 50): print(f"{S} states, 2 actions: {2**S:,} policies")
```

What it printed: the five-reward return is 4.0951 and the one-step value of "go" is 8.2. Brute force tried all 64 policies and found the best one: **run** in the start state, **walk** everywhere else, with $V(\text{start})=-4.23$. Value iteration found the same policy and the same value after 31 sweeps. The goal state's action does not matter, because the goal never changes.

## 🔴 4. Deep / formal
**The return.** With rewards $r_0, r_1, r_2,\dots$ and discount $\gamma$:

$$
G=\sum_{k=0}^{\infty}\gamma^{k}\,r_k
$$

**Where:**
- $r_k$ is the reward received $k$ steps from now.
- $\gamma\in[0,1)$ is the discount. **Matters:** with $\gamma<1$ and bounded rewards the sum is finite. With $\gamma=1$ and a game that never ends, it can grow without limit. With $\gamma=0$ the agent only cares about the next reward.
- $\gamma^k$ shrinks a reward the further away it is. **Matters:** it expresses "sooner is better" and it makes the maths work.
- $G$ is the total, and the agent tries to make its *expected* value large.

**Example (tiny numbers):** rewards $2, 2, 4$ and $\gamma=0.5$.
- $k=0$: $0.5^0\cdot2=2$. $k=1$: $0.5^1\cdot2=1$. $k=2$: $0.5^2\cdot4=1$.
- $G=2+1+1=\mathbf{4}$.
- Meaning: the big reward of 4 at step 2 is worth only 1 today, because two steps of discounting cut it to a quarter.

**The Bellman equation for a policy.** The value of a state is what you get now plus the discounted value of where you land, averaged over your own choices and over luck:

$$
V^{\pi}(s)=\sum_{a}\pi(a\mid s)\Big[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,V^{\pi}(s')\Big]
$$

**Where:**
- $s$ is the current state and $s'$ is a possible next state. $a$ is an action.
- $\pi(a\mid s)$ is how likely the policy is to pick $a$ in $s$. **Matters:** it averages over the agent's own choices.
- $R(s,a)$ is the reward for taking $a$ in $s$.
- $P(s'\mid s,a)$ is the chance of landing in $s'$. **Matters:** it averages over luck. Without it the equation would only fit worlds with no randomness.
- $V^{\pi}(s')$ is the value of the next state. **Matters:** the equation refers to itself, so the values of all states must agree with each other. That is why you can solve for them (a linear system, as `evaluate` does in the code) or repeat the rule until they stop changing.
- $\gamma$ is the discount, as above.

**Example (tiny numbers):** a state with two actions, picked with chance 0.5 each, and $\gamma=0.9$.
- Action "go": reward 1. It lands in $X$ (value 10) with chance 0.8 and in $Y$ (value 0) with chance 0.2. Value: $1+0.9\,(0.8\cdot10+0.2\cdot0)=1+7.2=\mathbf{8.2}$.
- Action "stay": reward 0, it lands in a state of value 4 for sure. Value: $0+0.9\cdot4=\mathbf{3.6}$.
- $V^{\pi}(s)=0.5\cdot8.2+0.5\cdot3.6=4.1+1.8=\mathbf{5.9}$.
- Meaning: this state is worth 5.9 under that policy. If the policy always said "go" it would be worth 8.2, so the policy can be improved.

**Further notes:**
- **Partial view:** if the agent cannot see the whole state, the problem is a *partially observable* MDP (POMDP). Not covered here.
- **Optimal version:** replacing the average over actions by a maximum gives the Bellman *optimality* equation, written out in [[Q-Learning]].
- **The code's world** is made up (6 states, my own chances). It shows the method, not any real task.
- **Not covered:** convergence proofs, average-reward MDPs, continuous states.

**What came before, and how this improved on it:**
- **Before:** the Markov chain, a random process with no choices, named for Andrey Markov (the Wikipedia page on MDPs ties the name to him but gives no year). Choices under uncertainty were handled by hand, case by case.
- **Its limit:** a Markov chain cannot say which action is best, and trying every policy does not scale. In the code, 2 actions over $S$ states give $2^S$ policies: 64 for 6 states, 1,048,576 for 20 and 1,125,899,906,842,624 for 50.
- **What changed:** the MDP framework with a value that obeys a self-referring equation, solved by dynamic programming. Wikipedia says MDPs came out of operations research in the 1950s, with Bellman's 1957 book *Dynamic Programming* linked to value iteration, and Howard's 1960 book *Dynamic Programming and Markov Processes* credited with policy iteration (page read; I did not read the originals).
- **How much better:** in the code's 6-state world, value iteration reached the same best policy and value as brute force after 31 sweeps, instead of testing 64 policies. At 6 states that is no real saving (31 sweeps over 6 states and 2 actions is 372 one-state updates against 64 linear solves). The gain is in growth: brute force doubles with each added state, while I did not time value iteration at 20 or 50 states, so I cannot give a measured figure there.

## Related
- [[Reinforcement Learning]] · [[Q-Learning]] · [[Policy Gradient]] · [[RLHF]] · [[PPO]] · [[Reward Model]] · [[Agent]] · [[Dynamic Programming]] · [[Probability]] · [[Matrix]] · [[Markov Chain]] · [[Bellman Equation]]

## Sources
- Wikipedia, *Markov decision process* (page read for the history block; origins in 1950s operations research, Bellman 1957, Howard 1960, Shapley 1953)
- Figure: [Markov_Decision_Process.svg](https://commons.wikimedia.org/wiki/File:Markov_Decision_Process.svg) (waldoalvarez), CC BY-SA 4.0, Wikimedia Commons. Same file and caption as in [[Q-Learning]], where it was looked at first.
- The code and its numbers are my own, run in this session.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/alignment-rl Markov decision process (MDP):::A game written as rules: the states you can be in, the actions you can take, where each action may lead (with luck), and the reward you get. The key rule: only where you are now matters, not how you got there.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the discounted return and explain each symbol.
?
$$G=\sum_{k=0}^{\infty}\gamma^{k}\,r_k$$
- $r_k$: the reward received $k$ steps from now.
- $\gamma\in[0,1)$: the discount. With $\gamma<1$ and bounded rewards the sum is finite. With $\gamma=1$ in a game that never ends it can grow without limit. With $\gamma=0$ the agent only cares about the next reward.
- $\gamma^k$ shrinks a reward the further away it is ("sooner is better").
- $G$: the total. The agent tries to make its expected value large.

#flashcards/ml/example #flashcards/topic/alignment-rl Rewards 2, 2, 4 and $\gamma=0.5$. What is $G$?::$0.5^0\cdot2=2$, $0.5^1\cdot2=1$, $0.5^2\cdot4=1$. $G=4$. The reward of 4 at step 2 is worth only 1 today, because two steps of discounting cut it to a quarter.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the value of a state under a policy and explain each symbol.
?
$$V^{\pi}(s)=\sum_{a}\pi(a\mid s)\Big[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,V^{\pi}(s')\Big]$$
- $s$: the current state. $s'$: a possible next state. $a$: an action.
- $\pi(a\mid s)$: how likely the policy is to pick $a$ in $s$. It averages over the agent's own choices.
- $R(s,a)$: the reward for taking $a$ in $s$.
- $P(s'\mid s,a)$: the chance of landing in $s'$. It averages over luck.
- $V^\pi(s')$: the value of the next state. The equation refers to itself, so all state values must agree with each other, which is why they can be solved for (a linear system).

#flashcards/ml/example #flashcards/topic/alignment-rl A state has two actions picked with chance 0.5 each, $\gamma=0.9$. "go": reward 1, lands in $X$ (value 10) with chance 0.8 or $Y$ (value 0) with 0.2. "stay": reward 0, lands in a state of value 4. What is $V^\pi(s)$?::go: $1+0.9(0.8\cdot10+0.2\cdot0)=8.2$. stay: $0+0.9\cdot4=3.6$. $V^\pi(s)=0.5\cdot8.2+0.5\cdot3.6=5.9$. If the policy always said "go" it would be worth 8.2, so the policy can be improved.

#flashcards/ml/image #flashcards/topic/alignment-rl ![[markov-decision-process.svg]]
What does this picture show?
?
A small MDP. The green circles are states. The orange circles are actions you can take in a state. The numbers on the arrows are the chances of ending up in each next state. The wavy arrows show rewards (+5 and −1). (From the note's caption: waldoalvarez, Wikimedia Commons.)

#flashcards/ml/cloze #flashcards/topic/alignment-rl In an MDP, only the ==current state== matters, not how you got there.
