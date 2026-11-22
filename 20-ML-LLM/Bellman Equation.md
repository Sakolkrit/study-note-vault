---
tags: [reinforcement-learning, dynamic-programming, Bellman, value-function, AI]
status: growing
level: 4
tier: 2
---

# Bellman Equation

A rule that says: **the value of where you are = what you get now + the value of where you end up next.** It turns one huge planning problem into many small ones that lean on each other.

## 🟢 1. Simple (age 12)
Imagine you want the cheapest way home. Instead of checking every route, you ask at each corner one small question: "what does the next step cost, plus the cheapest trip from the place it takes me?" If you already know the cheapest trip from the next corner, then the answer for this corner is easy. The Bellman equation is that one small question, written as a rule.

**Example:** You are two stops from home. The next stop costs 3 coins and from there home costs 4 more. The other next stop costs 1 coin and from there home costs 8. The first is 3 + 4 = 7, the second is 1 + 8 = 9, so the best trip from here costs 7.

## 🔵 2. How it works
1. In a [[Markov Decision Process]] you pick actions, get rewards and land in new states, with some luck.
2. The **value** of a state is the total reward you can expect from there on, with later rewards counting a bit less (the discount).
3. Look one step ahead. The value of a state is: the reward now, plus the discounted value of the next state, averaged over the luck.
4. That sentence mentions the value of the next state, and the next state's value is given by the same sentence. So all the values must **fit together**. That is the equation.
5. Two flavours. If you follow a given plan (a *policy*), average over the plan's choices. If you want the best, take the **best** action instead of the average. The second is the *optimality* equation, the one [[Q-Learning]] aims at.
6. To use it, start with any guess for the values and apply the rule again and again. Each round pushes the numbers closer to the true ones. This is *value iteration* ([[Dynamic Programming]]).
7. Variation: instead of a value for each state, keep a value for each (state, action) pair, called a **Q-value**. That is the form Q-learning uses when the world's chances are not known.

## 🟠 3. Technical detail
- **Two things being defined:** $V(s)$, the value of a state, and $Q(s,a)$, the value of taking action $a$ in $s$ and then continuing well. They are linked: $V^*(s)=\max_a Q^*(s,a)$.
- **Backup:** one application of the rule to every state. It is called a "backup" because value information flows backward from the future to the present.
- **Why repeating works:** each backup shrinks the largest error by at least the discount $\gamma$ (a *contraction*), so the numbers settle on exactly one answer, whatever the start. The code checks this.
- **Known world vs unknown:** with known chances, apply the rule directly. Without them, [[Q-Learning]] replaces the average by one sampled step and nudges toward the result, and [[Policy Gradient]] skips values and changes the policy directly.
- **Limit:** you need a value for every state, so a world with many state variables gets too big (the curse of dimensionality). [[Neural Network]]s that guess the values, as in Deep Q-learning, are the usual answer.

```python
import itertools
import numpy as np

# 1) Q from V, V from Q (one state, two actions, gamma 0.9)
gamma = 0.9
Vnext = {"X": 10, "Y": 0, "Z": 4}
q_go = 1 + gamma * (0.8 * Vnext["X"] + 0.2 * Vnext["Y"])
q_stay = 0 + gamma * Vnext["Z"]
print("Q(go) =", round(q_go, 2), "| Q(stay) =", round(q_stay, 2), "| optimal V = max =", round(max(q_go, q_stay), 2))

# 2) the contraction: one sweep of the optimality equation shrinks the largest error by at least gamma
rng = np.random.RandomState(0)
n, nA = 8, 3
P = rng.rand(nA, n, n); P /= P.sum(2, keepdims=True); R = rng.rand(n, nA)
back = lambda V: np.max(R + gamma * np.einsum('ast,t->sa', P, V), axis=1)
Vs = np.zeros(n)
for _ in range(2000): Vs = back(Vs)                       # run to the fixed point
V = rng.randn(n) * 10; err = np.abs(V - Vs).max(); print("start error:", round(err, 3))
ok = True
for k in range(1, 6):
    V = back(V); e = np.abs(V - Vs).max(); ok &= e <= gamma * err + 1e-12
    print(f"sweep {k}: error {e:.4f} (bound {gamma * err:.4f})"); err = e
print("error never above gamma times the old error:", ok)

# 3) why bother: best path through L layers of 3 nodes, every node linked to all 3 in the next layer
L = 10; c = rng.randint(1, 10, size=(L, 3, 3))            # c[l][i][j] = cost of the step from node i to node j
paths = min(sum(c[l][p[l]][p[l + 1]] for l in range(L - 1)) for p in itertools.product(range(3), repeat=L))
J = np.zeros(3)                                            # cost-to-go from the last layer
for l in range(L - 2, -1, -1): J = (c[l] + J).min(1)       # Bellman: best step plus best rest
print("brute force paths checked:", 3**L, "| Bellman updates:", (L - 1) * 9)
print("same best cost:", paths, int(J.min()), paths == J.min())
```

What it printed: $Q(\text{go})=8.2$ and $Q(\text{stay})=3.6$, so the best value is 8.2. In the random 8-state world the start error was 22.484 and fell to 6.220, 4.776, 3.988, 3.559 and 3.193 over five sweeps, never above $\gamma$ times the old error (an upper bound, not the exact rate). In the layered path problem brute force checked 59,049 paths and the Bellman method did 81 updates, with the same best cost, 21.

## 🔴 4. Deep / formal
**The optimality equation for state values.** Written with $V^*$, the best value any policy can get:

$$
V^{*}(s)=\max_{a}\Big[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,V^{*}(s')\Big]
$$

**Where:**
- $V^*(s)$ is the best possible expected total reward starting from state $s$.
- $\max_a$ picks the best action. **Matters:** replace it with an average over a policy's choices and you get the equation for that policy (the one in [[Markov Decision Process]]). The max is what makes the answer "the best".
- $R(s,a)$ is the reward right now, $P(s'\mid s,a)$ the chance of landing in $s'$, and $\gamma$ the discount. **Matters:** without $P$ the equation would ignore luck. Without $\gamma$ the values can grow without limit in a game that never ends.
- $V^*(s')$ appears on both sides. **Matters:** it is a fixed-point equation. You cannot just read off the answer, but you can find it by repeating the rule.

**Example (tiny numbers):** $\gamma=0.9$. In state $s$, action "go" gives reward 1 and lands in $X$ (value 10) with chance 0.8 or $Y$ (value 0) with chance 0.2. Action "stay" gives reward 0 and lands in $Z$ (value 4) for sure.
- go: $1+0.9\,(0.8\cdot10+0.2\cdot0)=1+7.2=\mathbf{8.2}$.
- stay: $0+0.9\cdot4=\mathbf{3.6}$.
- $V^*(s)=\max(8.2,3.6)=\mathbf{8.2}$, and the best action is "go".
- Meaning: the value of $s$ is set by the better option, and the weaker option does not change it.

**The same idea for action values.**

$$
Q^{*}(s,a)=R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,\max_{a'}Q^{*}(s',a')
$$

**Where:**
- $Q^*(s,a)$ is the best total reward if you take $a$ now in $s$ and then act as well as possible.
- $\max_{a'}Q^*(s',a')$ is the value of the best action in the next state. **Matters:** this is what [[Q-Learning]] estimates from samples. It uses the best next action even if the agent explores differently.
- The rest is as above. **Matters:** the max sits **inside** the sum here, so a learner that only sees one sampled $s'$ can still use the equation without knowing $P$.

**Example (tiny numbers):** $\gamma=0.9$. Action $a$ in state $s$ gives reward 2. With chance 0.5 you land in $A$, where the actions have values $(3,5)$. With chance 0.5 you land in $B$, where they have values $(1,0)$.
- Best next value in $A$ is $5$, in $B$ is $1$.
- Average: $0.5\cdot5+0.5\cdot1=3$.
- $Q^*(s,a)=2+0.9\cdot3=\mathbf{4.7}$.
- Meaning: the action is worth 4.7 if you play well afterwards.

**Why repeating works (contraction).** Let $T$ be one backup (the right side of the first equation applied to every state). For any two value guesses $U$ and $W$:

$$
\lVert TU-TW\rVert_\infty\le\gamma\,\lVert U-W\rVert_\infty
$$

**Where:**
- $T$ is the backup operator, one sweep of the rule over all states.
- $U, W$ are two guesses for the values of all states.
- $\lVert\cdot\rVert_\infty$ is the largest difference over all states. **Matters:** it is the "worst state" error.
- $\gamma<1$ is the discount. **Matters:** it is the shrink factor. With $\gamma=1$ the guarantee disappears.

**Example (tiny numbers):** two guesses for two states are $U=(10,0)$ and $W=(6,3)$, and $\gamma=0.9$.
- Largest difference: $\max(|10-6|,|0-3|)=\mathbf{4}$.
- After one backup of each, the largest difference is at most $0.9\cdot4=\mathbf{3.6}$.
- Meaning: the two guesses get closer every sweep, so they end at the same place. That is why value iteration always lands on the one right answer (the code's error table is the same effect). The proof itself is not shown here; this is the standard result as I remember it.

**Further notes:**
- **Principle of optimality:** whatever the first decision is, the rest must be best for the state it leads to. The equation is that principle in symbols.
- **Finite plans:** with a fixed number of steps there is no discount needed, and you work backward from the last step (the layered-path code does this).
- **Not covered:** the proof of the contraction, policy iteration's convergence, continuous states.

**What came before, and how this improved on it:**
- **Before:** to find the best plan, list every plan and compare their totals.
- **Its limit:** the number of plans multiplies at every step. In the code, 3 choices per layer over 10 layers gives $3^{10}=59{,}049$ paths.
- **What changed:** Bellman's principle of optimality and the recursion built on it, in Bellman's work on dynamic programming. The Wikipedia page lists his 1952 paper *On the Theory of Dynamic Programming* and the 1957 book *Dynamic Programming* as the source of the principle (page read; I did not read the originals).
- **How much better:** in the code's layered problem, 81 updates instead of 59,049 paths, same best cost (21). That gap grows without limit as layers are added, because paths multiply while updates only add. Wikipedia also notes that adding more state variables can still make it too big (the curse of dimensionality), so it does not solve every problem.

## Related
- [[Markov Decision Process]] · [[Q-Learning]] · [[Reinforcement Learning]] · [[Policy Gradient]] · [[Dynamic Programming]] · [[Neural Network]] · [[RLHF]] · [[Expected Value]]

## Sources
- Wikipedia, *Bellman equation* (page read for the history block: Bellman's 1952 paper and 1957 book, principle of optimality, curse of dimensionality as a limit)
- Wikipedia, *Markov decision process* (read for the MDP note, see [[Markov Decision Process]])
- The code and its numbers are my own, run in this session. No image: I did not search Wikimedia Commons for one for this note.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/alignment-rl Bellman equation:::The value of a state is the best immediate reward plus the discounted value of where you land. If you know the value of the next corner, the value of this corner is easy.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the Bellman optimality equation for state values and explain each symbol.
?
$$V^{*}(s)=\max_{a}\Big[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,V^{*}(s')\Big]$$
- $V^*(s)$: the best possible expected total reward starting from state $s$.
- $\max_a$: picks the best action. Replace it with an average over a policy's choices and you get the equation for that policy. The max makes the answer "the best".
- $R(s,a)$: the reward right now. $P(s'\mid s,a)$: the chance of landing in $s'$. $\gamma$: the discount. Without $P$ luck is ignored. Without $\gamma$ values can grow without limit in a game that never ends.
- $V^*(s')$ appears on both sides, so it is a fixed-point equation.

#flashcards/ml/example #flashcards/topic/alignment-rl $\gamma=0.9$. "go": reward 1, lands in $X$ (10) w.p. 0.8 or $Y$ (0) w.p. 0.2. "stay": reward 0, lands in $Z$ (4). What is $V^*(s)$?::go: $1+0.9(0.8\cdot10)=8.2$. stay: $0.9\cdot4=3.6$. $V^*(s)=\max(8.2,3.6)=8.2$, best action "go". The weaker option does not change the value.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the Bellman equation for action values and explain each symbol.
?
$$Q^{*}(s,a)=R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,\max_{a'}Q^{*}(s',a')$$
- $Q^*(s,a)$: the best total reward if you take $a$ now in $s$ and then act as well as possible.
- $\max_{a'}Q^*(s',a')$: the value of the best action in the next state. This is what Q-learning estimates from samples.
- The max sits inside the sum, so a learner that sees only one sampled $s'$ can still use the equation without knowing $P$.

#flashcards/ml/example #flashcards/topic/alignment-rl $\gamma=0.9$. Action $a$ in $s$ gives reward 2. Chance 0.5 to land in $A$ (action values 3, 5) and 0.5 in $B$ (values 1, 0). What is $Q^*(s,a)$?::Best next values: 5 in $A$, 1 in $B$. Average $0.5\cdot5+0.5\cdot1=3$. $Q^*(s,a)=2+0.9\cdot3=4.7$.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the contraction property of the Bellman backup and explain each symbol.
?
$$\lVert TU-TW\rVert_\infty\le\gamma\,\lVert U-W\rVert_\infty$$
- $T$: the backup operator, one sweep of the rule over all states.
- $U,W$: two guesses for the values of all states.
- $\lVert\cdot\rVert_\infty$: the largest difference over all states (the "worst state" error).
- $\gamma<1$: the shrink factor. With $\gamma=1$ the guarantee disappears.

#flashcards/ml/example #flashcards/topic/alignment-rl Guesses $U=(10,0)$, $W=(6,3)$, $\gamma=0.9$. How far apart are they after one backup?::Largest difference now: $\max(4,3)=4$. After one backup at most $0.9\cdot4=3.6$. The guesses get closer every sweep, so value iteration lands on the one right answer. (The note says the proof is the standard result from memory.)

#flashcards/ml/cloze #flashcards/topic/alignment-rl The max over actions is what makes the Bellman equation say ==best== value instead of a given policy's value.
