---
tags: [epsilon-greedy, exploration, exploitation, reinforcement-learning, bandit, Q-learning, ML]
status: growing
level: 4
tier: 1
---

# Epsilon-Greedy

Epsilon-greedy is the simplest way for a learning agent to **balance trying new things with using what it already knows**. Most of the time it picks the move that looks best; with a small chance $\varepsilon$ (the Greek letter epsilon) it picks a move completely at random instead.

## 🟢 1. Simple (age 12)
You have a favourite ice-cream flavour. If you always order it, you will never find out that the new flavour is even better. If you always pick something new, you never enjoy your favourite. So you decide: **most days, order the favourite; one day in ten, close your eyes and point at any flavour.** That small random habit is epsilon-greedy.

**Example:** a robot learning Snake mostly turns toward the food. One move in ten it turns in a random direction, and sometimes it learns something it would never have tried.

## 🔵 2. How it works
1. **Choose a small number** $\varepsilon$ between 0 and 1, such as 0.1 (10%).
2. **Roll a dice** with that chance. If it says "explore", pick any move at random, each move equally likely.
3. **Otherwise exploit:** pick the move with the best current score ([[Q-Learning]], [[Deep Q-Network]]).
4. **Learn** from what happened and update the scores.
5. **Often, make $\varepsilon$ fall over time.** At the start the agent knows nothing, so it explores a lot (near 1). Later it knows more, so it explores less (near 0.05).
6. **When testing**, set $\varepsilon$ to 0, so the agent plays its best.

## 🟠 3. Technical detail
- **The probabilities.** With $|A|$ possible moves, the policy is

$$
\pi(a\mid s)=\begin{cases}1-\varepsilon+\dfrac{\varepsilon}{|A|} & \text{if } a \text{ is the best-scoring move}\\[2mm] \dfrac{\varepsilon}{|A|} & \text{otherwise}\end{cases}
$$

**Where:**
- $\pi(a\mid s)$ is the chance of choosing move $a$ when in state $s$.
- $\varepsilon$ is the exploration rate. $|A|$ is the number of moves.
- The best-scoring move gets the "exploit" chance $1-\varepsilon$ **plus** its share of the random chance, because a random pick can land on it too.
- **Matters:** with $\varepsilon=0$ the agent is *greedy* and can lock onto the first move that looked fine. With $\varepsilon=1$ it is a random player and learns nothing about which moves are good. Without the extra $\varepsilon/|A|$ on the best move the probabilities would not add up to 1.

**Example (tiny numbers):** Snake has $|A|=3$ moves and $\varepsilon=0.1$ (my arithmetic in Python).
- Best move: $1-0.1+0.1/3=0.9333$.
- Each other move: $0.1/3=0.0333$.
- Check: $0.9333+0.0333+0.0333=1$.
- Meaning: the best move is chosen 93.3% of the time and each of the others 3.3%.

- **The schedule I used in [[Snake DQN]].** $\varepsilon$ falls in a straight line from 1 to a floor of 0.05:

$$
\varepsilon_t=\max\!\Big(\varepsilon_{\min},\;1-(1-\varepsilon_{\min})\,\frac{t}{T}\Big)
$$

**Where:**
- $t$ is the number of steps taken so far.
- $T$ is how many steps the fall takes (20,000 in my project).
- $\varepsilon_{\min}$ is the floor (0.05).
- $\max$ takes the larger of the two values, so $\varepsilon$ never goes below the floor.
- **Matters:** the start at 1 fills the memory ([[Experience Replay]]) with varied steps. Without the floor the agent would stop exploring completely and could not recover from a wrong belief.

**Example (tiny numbers):** $\varepsilon_{\min}=0.05$ and $T=20{,}000$ (my arithmetic).
- $t=5{,}000$: $1-0.95\times0.25=\mathbf{0.7625}$.
- $t=10{,}000$: $1-0.95\times0.5=\mathbf{0.525}$.
- $t=20{,}000$: $1-0.95\times1=0.05$, the floor. For any later $t$ it stays at $0.05$.
- Meaning: after 5,000 steps the agent still plays 76% random moves, and half-way through the fall it is at 52.5%.
- A detail of that project: the learning curves test the *greedy* player ($\varepsilon=0$), not the player that is still exploring, so the 5,000-step score (15.1 foods) is not of a snake that moves randomly 76% of the time.

- **Code: a 5-armed slot machine.** Five machines have true average rewards $0.1, 0.3, 0.5, 0.2, 0.9$ (the agent does not know them; the rewards are noisy, with standard deviation 1). The agent plays 1,000 pulls, keeps a running average per machine, and I average over 200 games for four values of $\varepsilon$:

```python
# file: toy_eps.py
import numpy as np

means = np.array([0.1, 0.3, 0.5, 0.2, 0.9])           # true average reward of 5 slot machines (arms); the agent does not know them
def run(eps, steps=1000, runs=200, seed=0):
    rng, total = np.random.default_rng(seed), 0.0
    for _ in range(runs):
        q, n = np.zeros(5), np.zeros(5)
        for _ in range(steps):
            a = int(rng.integers(5)) if rng.random() < eps else int(np.argmax(q))
            r = rng.normal(means[a], 1.0)
            n[a] += 1; q[a] += (r - q[a]) / n[a]       # running average of the rewards seen for this arm
            total += r
    return total / (steps * runs)
print("best possible average reward:", means.max())
for eps in (0.0, 0.01, 0.1, 0.5):
    print(f"epsilon {eps:<5} average reward per step: {run(eps):.3f}")
```

```text
best possible average reward: 0.9
epsilon 0.0   average reward per step: 0.435
epsilon 0.01  average reward per step: 0.599
epsilon 0.1   average reward per step: 0.777
epsilon 0.5   average reward per step: 0.640
```

- **Reading it:** the best possible is 0.9 per pull. Never exploring ($\varepsilon=0$) earned 0.435: it often locked onto a machine that looked good early. $\varepsilon=0.1$ earned 0.777, the best of the four. Exploring half the time ($\varepsilon=0.5$) earned 0.640: too much exploring wastes pulls. $\varepsilon=0.01$ earned 0.599. This is one run of 200 games with one seed, in a small made-up problem; it shows the *shape* (too little and too much both cost), not the best $\varepsilon$ for anything else.

## 🔴 4. Deep / formal
- **Undirected exploration.** A random move is as likely to be a bad idea as a good one, and the agent does not use what it knows about *how unsure* it is. In Snake a random move often runs into a wall: my random player scored 0.22 foods per game ([[Snake DQN]]). Smarter ways: *softmax* (Boltzmann) choice, which prefers moves with better scores; *UCB* (upper confidence bound), which adds a bonus for moves tried few times; *Thompson sampling*; *optimistic starting values*; and noise added to the network's weights (all from memory, none run here).
- **A fixed $\varepsilon$ never settles.** With $\varepsilon$ constant the agent keeps making random moves forever, so its average reward stays below the best possible by roughly $\varepsilon$ times the average loss of a random move. That is why $\varepsilon$ is usually lowered (a standard argument, from memory; the toy shows the cost of too much exploring, not a proof).
- **Greedy in the limit.** To be sure of finding the best moves in the long run, $\varepsilon$ must fall toward 0 slowly enough that every move is still tried infinitely often (a standard condition for tabular Q-learning to converge, from memory).
- **The tested policy.** The policy that is *tested* ($\varepsilon=0$) differs from the one that *collects the data* ($\varepsilon>0$). That is fine for off-policy methods such as [[Q-Learning]]. For on-policy SARSA the exploration changes what it learns (see [[Q-Learning]], where it chose a safer path along a cliff).
- **Settings I did not tune.** The start (1.0), the floor (0.05) and the fall time (20,000 steps) were fixed by me in advance. I did not test other values.

**What came before, and how this improved on it:**
- **Before:** always take the move that looks best (a purely greedy agent).
- **Its limit:** it can lock onto the first move that seemed fine and never find out about a better one.
- **What changed:** add a small, fixed chance of a random move, usually lowered over time. The idea is old and is used in Watkins' Q-learning work and in Sutton and Barto's textbook (from memory, not checked).
- **How much better:** measured in my slot-machine toy: 0.777 average reward with $\varepsilon=0.1$ against 0.435 with $\varepsilon=0$ (best possible 0.9; one seed, 200 games). Too much was worse again (0.640 at $\varepsilon=0.5$). No published figure is quoted.

## Related
- [[Q-Learning]] · [[Deep Q-Network]] · [[Experience Replay]] · [[Reinforcement Learning]] · [[Markov Decision Process]] · [[Policy Gradient]] · [[Probability]] · [[Snake DQN]] · [[Q-Learning Grid World]]
- [[Cold Start Problem]]
## Sources
- My own toy (Python 3.9.6, NumPy 1.24.4) and the [[Snake DQN]] project. The history, the alternatives (softmax, UCB, Thompson sampling, optimistic values, noisy weights), the convergence condition and the constant-$\varepsilon$ argument are from memory; I did not read a source for this note.
- No image: I searched Wikimedia Commons for "multi-armed bandit" and found only an unrelated photograph of a slot machine and a Thompson-sampling figure, neither of which shows epsilon-greedy, so I used none.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/alignment-rl Epsilon-greedy:::Pick the best-looking move most of the time and a completely random move with a small chance $\varepsilon$, so the agent keeps trying new things.

#flashcards/ml/reverse #flashcards/topic/alignment-rl Exploration versus exploitation:::The balance between trying new moves to learn more (explore) and using the moves that already look best (exploit).

#flashcards/ml/concept #flashcards/topic/alignment-rl What are the steps of epsilon-greedy?
?
Choose a small $\varepsilon$; roll a dice with that chance; if it says explore, pick a random move; otherwise pick the best-scoring move; learn from the result; usually lower $\varepsilon$ over time; set $\varepsilon$ to 0 when testing.

#flashcards/ml/formula #flashcards/topic/alignment-rl What are the move probabilities under epsilon-greedy?
?
$$\pi(a\mid s)=\begin{cases}1-\varepsilon+\dfrac{\varepsilon}{|A|} & \text{if } a \text{ is the best-scoring move}\\[2mm] \dfrac{\varepsilon}{|A|} & \text{otherwise}\end{cases}$$
- $\pi(a\mid s)$: chance of move $a$ in state $s$. $\varepsilon$: exploration rate. $|A|$: number of moves.
- The best move gets $1-\varepsilon$ plus its share of the random chance, because a random pick can land on it too.
- Matters: $\varepsilon=0$ is greedy and can lock onto a mediocre move; $\varepsilon=1$ is a random player.

#flashcards/ml/example #flashcards/topic/alignment-rl Snake has 3 moves and $\varepsilon=0.1$. What are the probabilities?::Best move $1-0.1+0.1/3=0.9333$; each other move $0.1/3=0.0333$; the three add up to 1.

#flashcards/ml/formula #flashcards/topic/alignment-rl What is the linear epsilon schedule used in the Snake DQN project?
?
$$\varepsilon_t=\max\!\Big(\varepsilon_{\min},\;1-(1-\varepsilon_{\min})\,\frac{t}{T}\Big)$$
- $t$: steps taken so far. $T$: steps the fall takes (20,000).
- $\varepsilon_{\min}$: the floor (0.05). $\max$ keeps $\varepsilon$ from going below it.
- Matters: the start at 1 fills the replay memory with varied steps; the floor keeps a little exploring so a wrong belief can still be fixed.

#flashcards/ml/example #flashcards/topic/alignment-rl $\varepsilon_{\min}=0.05$, $T=20000$. What is $\varepsilon$ at $t=5000$, $10000$ and $30000$?::$1-0.95\times0.25=0.7625$; $1-0.95\times0.5=0.525$; at $30000$ the line would be below the floor, so $\varepsilon=0.05$.

#flashcards/ml/concept #flashcards/topic/alignment-rl What did the slot-machine toy show about epsilon?
?
Five machines (true means 0.1, 0.3, 0.5, 0.2, 0.9; best possible 0.9), 1,000 pulls, 200 games, one seed: $\varepsilon=0$ earned 0.435, 0.01 earned 0.599, 0.1 earned 0.777, 0.5 earned 0.640. Too little and too much exploring both cost. It does not give the best $\varepsilon$ for other problems.

#flashcards/ml/concept #flashcards/topic/alignment-rl Why is epsilon-greedy called undirected exploration, and what are the alternatives?
?
A random move is as likely to be bad as good and ignores how unsure the agent is (in Snake random moves hit walls; random play scored 0.22 foods). Alternatives: softmax, UCB, Thompson sampling, optimistic starting values, noisy weights (all from memory, not run).

#flashcards/ml/concept #flashcards/topic/alignment-rl Why is the tested policy usually run with epsilon 0?
?
To see how well the agent plays its best. The policy that collects data (epsilon above 0) differs from the one that is tested; that is fine for off-policy methods like Q-learning. In the Snake project the 5,000-step score of 15.1 foods was for the greedy player, not the 76%-random one.

#flashcards/ml/concept #flashcards/topic/alignment-rl What did I not tune for epsilon in the Snake DQN project, and which facts are from memory?
?
The start (1.0), the floor (0.05) and the fall time (20,000 steps) were fixed in advance, with no other values tested. The history (Watkins, Sutton and Barto), the convergence condition and the alternatives are from memory.

#flashcards/ml/cloze #flashcards/topic/alignment-rl With epsilon-greedy the agent usually ==exploits== the best-looking move and with chance $\varepsilon$ it ==explores== with a random move.
