---
tags: [math, probability, markov, stochastic-process, foundations]
status: growing
level: 4
tier: 2
---

# Markov Chain

A sequence of random steps where **the next step depends only on where you are now, not on how you got there.**

## 🟢 1. Simple (age 12)
Think of a board game where you roll dice to move. To know where you might go next, you only need to know **which square you are standing on now**. You do not need to remember the squares you visited before. A Markov chain is any process that works like that: a list of places, and for each place, the chances of moving to each other place.

**Example:** Weather in a simple town. If today is sunny, tomorrow is sunny with chance 0.9 and rainy with chance 0.1. If today is rainy, tomorrow is rainy with chance 0.5. To guess tomorrow you only look at today, not at last week.

## 🔵 2. How it works
1. List the **states**: the places the process can be (sunny and rainy, or squares on a board).
2. For each state, write the **chances of moving to each state next**. These chances add up to 1. Together they form a table called the **transition matrix**, one row for each "from" state (a [[Matrix]]).
3. Start somewhere (or with chances of starting somewhere), then take a random step using the row for the current state, again and again.
4. The key promise, the **Markov property**: the row you use depends only on the current state, not on the earlier path.
5. You can also ask: "if I start here, what are the chances of being in each state after 1 step, 2 steps, 5 steps?" Multiply the chances by the table each step (see the code).
6. **After many steps** most chains forget where they started and settle into a fixed mix: the **stationary distribution**. Run the chain for a long time and the share of time in each state matches it.
7. **Where it shows up:** an [[Markov Decision Process]] is a Markov chain where an agent also picks actions, used in [[Reinforcement Learning]]. [[Monte Carlo]] methods use chains to draw samples from hard distributions (MCMC). Web page ranking and simple text generators also use them. A language model that looked only at the last word would be a Markov chain, while real ones read a long context ([[Autoregressive Generation]]); that comparison is my own view, not from a source.

![[markov-chain-graph.svg]]
*A Markov chain with two states, E and A. Each arrow shows the chance of moving from one state to another: from E, stay with 0.3 or go to A with 0.7; from A, go to E with 0.4 or stay with 0.6. The arrows out of each state add up to 1. Image: Joxemai4, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Markovkate_01.svg), CC BY-SA 3.0. The chain in the code is this one.*

## 🟠 3. Technical detail
- **Transition matrix** $P$: entry $P_{ij}$ is the chance of going from state $i$ to state $j$. Each row sums to 1.
- **Distribution as a row of numbers:** $p_t$ is a row of chances over the states at time $t$. One step is the multiplication $p_{t+1}=p_tP$, so $t$ steps is $p_0P^t$.
- **Stationary distribution** $\pi$: a distribution that does not change, $\pi P=\pi$. It is a left [[Eigenvalue]] eigenvector of $P$ with eigenvalue 1 (page statement, and the code finds it that way).
- **When it settles:** for a chain that can get from every state to every other (irreducible) and does not cycle with a fixed period (aperiodic), the powers $P^t$ converge to a matrix whose rows are all $\pi$ (Perron–Frobenius, per the Wikipedia page).
- **How fast:** the speed is set by the ratio of the second-largest to the largest eigenvalue size. The smaller, the faster. In the code the eigenvalues are $1$ and $-0.1$, and the gap to $\pi$ shrinks 10 times per step.
- **Long-run averages without independence:** Markov's own result was an averages-converge result for dependent steps (see the history block). The code's long walk shows the time share matching $\pi$.
- **Memory, if you need it:** if the next step depends on the last two states, make the pair the state. That again fits the Markov rule, with more states.

```python
import numpy as np
rng = np.random.RandomState(0)

# the two-state chain in the picture: E -> E 0.3, E -> A 0.7, A -> E 0.4, A -> A 0.6
P = np.array([[0.3, 0.7],
              [0.4, 0.6]])                         # row = from, column = to; each row adds to 1

# 1) push a distribution through the chain: start surely in E
p = np.array([1.0, 0.0])
for t in range(1, 6):
    p = p @ P; print(f"step {t}: P(E), P(A) = {p.round(4)}")

# 2) the stationary distribution: pi P = pi (left eigenvector for eigenvalue 1)
w, v = np.linalg.eig(P.T)
pi = np.real(v[:, np.argmax(np.real(w))]); pi /= pi.sum()
print("stationary:", pi.round(4), "| pi P = pi:", np.allclose(pi @ P, pi), "| eigenvalues:", np.round(np.real(w), 3))

# 3) one long walk: share of time spent in each state (no independence between steps)
s, visits = 0, np.zeros(2)
for _ in range(200_000):
    s = rng.choice(2, p=P[s]); visits[s] += 1
print("share of time in E, A over 200,000 steps:", (visits / visits.sum()).round(4))

# 4) memoryless: the next state depends on the current one, not on the one before
seq = [0]
for _ in range(300_000): seq.append(rng.choice(2, p=P[seq[-1]]))
seq = np.array(seq)
for prev in (0, 1):
    m = (seq[1:-1] == 1) & (seq[:-2] == prev)      # current = A, previous = prev
    print(f"P(next = A | current = A, previous = {'EA'[prev]}) = {np.mean(seq[2:][m] == 1):.3f}")

# 5) how fast it forgets the start: the gap to the stationary distribution shrinks by |second eigenvalue| each step
q = np.array([1.0, 0.0]); gaps = []
for _ in range(5): q = q @ P; gaps.append(np.abs(q - pi).max())
print("gap per step:", np.round(gaps, 5), "| ratios:", np.round(np.array(gaps[1:]) / np.array(gaps[:-1]), 3))

# 6) does remembering the current state help predict the next one? average surprise (log loss, in nats)
def loss(P, n=200_000):
    w, v = np.linalg.eig(P.T); pi = np.real(v[:, np.argmax(np.real(w))]); pi /= pi.sum()
    H = lambda r: -(r * np.log(r)).sum()
    return H(pi), sum(pi[i] * H(P[i]) for i in range(2))
for name, Q in (("picture chain", P), ("sticky chain [[.95,.05],[.1,.9]]", np.array([[.95, .05], [.1, .9]]))):
    a, b = loss(Q); print(f"{name}: ignoring the current state {a:.4f} | using it {b:.4f}")
```

What it printed: starting surely in E, the chances of (E, A) were (0.3, 0.7) after 1 step, (0.37, 0.63) after 2, and (0.3636, 0.6364) after 5. The stationary distribution is (0.3636, 0.6364), and the check $\pi P=\pi$ passed. The eigenvalues are 1 and −0.1. A walk of 200,000 steps spent 0.3618 of its time in E and 0.6382 in A. The chance of A next, given the current state A, was 0.600 when the previous state was E and 0.599 when it was A, so the earlier state did not matter. The gap to $\pi$ shrank by exactly 0.1 each step. For prediction, using the current state lowered the average surprise (log loss) of the picture chain only from 0.6555 to 0.6504, while for a "sticky" chain it fell from 0.6365 to 0.2407.

## 🔴 4. Deep / formal
**The Markov property.** For a chain with transition matrix $P$:

$$
\Pr(X_{t+1}=j\mid X_t=i,\,X_{t-1},\dots,X_0)=\Pr(X_{t+1}=j\mid X_t=i)=P_{ij}
$$

**Where:**
- $X_t$ is the state at time $t$, $i$ and $j$ are states.
- $\Pr(\cdot\mid\cdot)$ reads "the chance of the left thing, given the right thing".
- $X_{t-1},\dots,X_0$ is the whole earlier path. **Matters:** it disappears from the right-hand side. That is the memoryless promise. If the process really did depend on the earlier path, this equation would be false and a plain transition table would give wrong predictions.
- $P_{ij}$ is the table entry. **Matters:** the same table is used at every time step (a "time-homogeneous" chain).

**Example (tiny numbers):** the picture chain. The state is E now. From E the chance of moving to A is $P_{EA}=\mathbf{0.7}$, whether yesterday was E or A. The code checked the A-to-A chance from both earlier states: 0.600 and 0.599, both about the table's 0.6.

**One step of the chain.** For a row $p_t$ of state chances at time $t$:

$$
p_{t+1}=p_t\,P
$$

**Where:**
- $p_t$ is the row of chances over states at time $t$ (its entries add up to 1).
- $P$ is the transition matrix.
- **Matters:** this is multiplication of a row by a [[Matrix]]. Each new chance adds up all the ways to arrive: "was in state $i$" times "moved from $i$ to $j$". Skipping any route would lose chance, and the total would no longer be 1.

**Example (tiny numbers):** start surely in E, so $p_0=(1,0)$, with the picture's $P$.
- $p_1=(1\cdot0.3+0\cdot0.4,\;1\cdot0.7+0\cdot0.6)=(\mathbf{0.3},\mathbf{0.7})$.
- $p_2=(0.3\cdot0.3+0.7\cdot0.4,\;0.3\cdot0.7+0.7\cdot0.6)=(0.09+0.28,\;0.21+0.42)=(\mathbf{0.37},\mathbf{0.63})$.
- Meaning: after two steps there is a 37% chance of being in E. The code printed the same numbers.

**The stationary distribution.**

$$
\pi P=\pi,\qquad \sum_i\pi_i=1,\qquad \pi_i\ge0
$$

**Where:**
- $\pi$ is a row of chances over states that the chain leaves unchanged.
- $\sum_i\pi_i=1$ and $\pi_i\ge0$ make it a proper set of chances. **Matters:** without them, $\pi=(0,0)$ would also satisfy $\pi P=\pi$ and tell you nothing.
- $\pi P=\pi$ says one step changes nothing. **Matters:** this is what the long-run time shares settle at, and the code's long walk agrees.

**Example (tiny numbers):** the picture chain, $\pi=(\pi_E,\pi_A)$.
- First column of $\pi P=\pi$: $0.3\pi_E+0.4\pi_A=\pi_E$, so $0.4\pi_A=0.7\pi_E$ and $\pi_E=\tfrac47\pi_A$.
- Adding up to 1: $\tfrac47\pi_A+\pi_A=1$, so $\pi_A=\tfrac{7}{11}\approx\mathbf{0.6364}$ and $\pi_E=\tfrac{4}{11}\approx\mathbf{0.3636}$.
- Meaning: in the long run the chain is in A about 64% of the time, which matches the code's 0.6382 from one finite walk and its 5-step distribution.

**Further notes:**
- **Does knowing the current state help?** It depends on the chain. For the picture chain, barely (0.6555 down to 0.6504). For a sticky chain (stay with 0.95 or 0.9) a lot (0.6365 down to 0.2407). Whether a model should keep memory is a question about data, not a rule.
- **Not covered:** periodic and reducible chains in detail, continuous-time chains, hitting times, and proofs of convergence.

**What came before, and how this improved on it:**
- **Before:** the weak law of large numbers was proved for independent steps, and the Wikipedia page says Pavel Nekrasov held that independence was necessary.
- **Its limit:** most real sequences (weather, letters in a text) depend on what came just before, so the independence assumption excluded them.
- **What changed:** Markov published his first paper on this in 1906 (page read). He showed that under certain conditions the averages of a chain converge to a fixed vector, which gives a weak law of large numbers without independence, and he set out to study "an extension of independent random sequences". He later used chains on the distribution of vowels in *Eugene Onegin* (the page gives no date for that work).
- **How much better:** in the code's sticky chain, the average surprise per step falls from 0.6365 (ignoring the current state) to 0.2407 (using it), about 2.6 times less. For the picture chain the gain is small (0.6555 to 0.6504), so the benefit of using the current state depends on the chain.

## Related
- [[Markov Decision Process]] · [[Probability]] · [[Matrix]] · [[Eigenvalue]] · [[Monte Carlo]] · [[Law of Large Numbers]] · [[Expected Value]] · [[Reinforcement Learning]] · [[Autoregressive Generation]] · [[Bellman Equation]]

## Sources
- Wikipedia, *Markov chain* (first 100,000 of about 183,500 characters read, for the history block and the stationary-distribution and convergence statements)
- Figure: [Markovkate_01.svg](https://commons.wikimedia.org/wiki/File:Markovkate_01.svg) (Joxemai4), CC BY-SA 3.0, Wikimedia Commons. I viewed it as a PNG thumbnail before writing the caption.
- The code and its numbers are my own, run in this session.

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/probability Markov chain:::A process with a list of places and, for each place, the chances of moving to each other place. To know what comes next you only need where you are now, not how you got there.

#flashcards/math/formula #flashcards/topic/probability Write the Markov (memoryless) property and explain each symbol.
?
$$\Pr(X_{t+1}=j\mid X_t=i,\,X_{t-1},\dots,X_0)=\Pr(X_{t+1}=j\mid X_t=i)=P_{ij}$$
- $X_t$: the state at time $t$. $i$, $j$: states.
- $\Pr(\cdot\mid\cdot)$: the chance of the left thing, given the right thing.
- $X_{t-1},\dots,X_0$: the whole earlier path. It disappears from the right side. That is the memoryless promise.
- $P_{ij}$: the table entry, the chance of moving from $i$ to $j$. The same table is used at every step (a time-homogeneous chain).

#flashcards/math/formula #flashcards/topic/probability Write how a Markov chain's chances move one step forward, and explain each symbol.
?
$$p_{t+1}=p_t\,P$$
- $p_t$: the row of chances over states at time $t$ (adds up to 1).
- $P$: the transition matrix.
- This is a row times a matrix. Each new chance adds up all the ways to arrive: "was in state $i$" times "moved from $i$ to $j$". Skipping a route would lose chance, and the total would no longer be 1.

#flashcards/math/formula #flashcards/topic/probability Write the stationary distribution conditions and explain each symbol.
?
$$\pi P=\pi,\qquad \sum_i\pi_i=1,\qquad \pi_i\ge0$$
- $\pi$: a row of chances over states that the chain leaves unchanged.
- $\sum_i\pi_i=1$ and $\pi_i\ge0$ make it proper chances. Without them $\pi=(0,0)$ would also satisfy $\pi P=\pi$ and tell you nothing.
- $\pi P=\pi$: one step changes nothing. This is where the long-run time shares settle.

#flashcards/math/cloze #flashcards/topic/probability The memoryless property means the next state depends only on the ==current== state, not on the ==earlier path==.

#flashcards/math/example #flashcards/topic/probability A chain: from E stay 0.3 or go to A 0.7; from A go to E 0.4 or stay 0.6. Starting surely in E, what are $p_1$ and $p_2$?::p₁ = (0.3, 0.7). p₂ = (0.3·0.3 + 0.7·0.4, 0.3·0.7 + 0.7·0.6) = (0.37, 0.63). After two steps there is a 37% chance of being in E.

#flashcards/math/example #flashcards/topic/probability For that chain, what is the stationary distribution $\pi=(\pi_E,\pi_A)$?::0.3π_E + 0.4π_A = π_E gives π_E = (4/7)π_A. With the sum equal to 1, π_A = 7/11 ≈ 0.6364 and π_E = 4/11 ≈ 0.3636.

#flashcards/math/image #flashcards/topic/probability ![[markov-chain-graph.svg]]
What does this picture show?
?
A Markov chain with two states, E and A. Each arrow is the chance of moving between states: from E, stay 0.3 or go to A 0.7; from A, go to E 0.4 or stay 0.6. The arrows out of each state add up to 1.
