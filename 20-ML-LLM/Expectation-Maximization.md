---
tags: [machine-learning, unsupervised-learning, statistics, EM, hidden-variables, optimisation]
status: growing
level: 4
tier: 3
---

# Expectation-Maximization

Expectation-Maximization (usually called **EM**) is a loop for fitting a model when **part of the story is hidden**: guess the hidden part, refit the model using that guess, and repeat.

## 🟢 1. Simple (age 12)
You have two jars of marbles, but the labels fell off, and someone has mixed up bags of marbles taken from them. You do not know which bag came from which jar, and you also do not know how many red marbles each jar holds. It is a chicken-and-egg puzzle: if you knew the jars' contents, you could guess where each bag came from. If you knew where each bag came from, you could work out the jars' contents.

EM breaks the deadlock with a **guess and fix loop**. Start with a rough guess about the jars. Use it to guess where each bag came from. Use those guesses to improve your idea of the jars. Do it again. Each round gets a bit better, until it stops changing.

**Example:** Two coins, A and B, are secretly used for sets of 10 flips. You only see the head counts of each set, not which coin was flipped. Guess "A lands heads 60%, B 50%". A set with 9 heads probably came from A. A set with 4 heads probably came from B. Re-estimate each coin from the sets it probably owns. Repeat.

## 🔵 2. How it works
EM is the standard way to fit models with a **hidden variable**, such as the group a point belongs to in a [[Gaussian Mixture Model]] (see also [[Clustering]]). It has two steps that you repeat.
1. **Start** with any guess for the model's settings.
2. **E-step (Expectation, "guess the hidden part"):** using the current settings, work out, for every data point, how likely each hidden option is. For a coin set: "70% likely coin A, 30% coin B". These shares are **soft guesses**, not hard labels.
3. **M-step (Maximization, "refit"):** pretend those soft guesses are the truth. Update the settings so the data is as likely as possible, counting each point partly towards each hidden option. This is [[Maximum Likelihood]] with weights.
4. Go back to step 2 with the new settings.
5. Stop when the settings (or the score) barely change.

![[em-old-faithful.gif]]
*EM fitting a two-bell Gaussian mixture to Old Faithful geyser data (each dot is one eruption: how long it lasted across, and the wait before it up). The red ovals start badly placed and, step by step, settle onto the two clumps. This is an animation, so you may need to view it in Obsidian to see it move. Image: 3mta3, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Em_old_faithful.gif), CC BY-SA 3.0.*

## 🟠 3. Technical detail
- **When to use it:** whenever the likelihood is easy if the hidden part is known, and hard if it is not. Typical cases are mixture models ([[Gaussian Mixture Model]]), models with missing data, and some probabilistic models of text and sequences.
- **It never makes things worse:** each round either raises the likelihood of the data or leaves it unchanged (a standard result; stated from memory, not proved here). The code below checks this on a small case.
- **But it can get stuck:** it climbs to a **local** peak, which depends on the starting guess. Run it from several starts and keep the best.
- **Soft vs hard:** if the E-step is replaced by "pick the single most likely option", you get a simpler, harder-edged version. k-means is that kind of method for round, equal bells (stated from memory; see [[Clustering]]).
- **Speed:** each round is cheap, but convergence can be slow, because it creeps up the hill in small steps.
- **Compared with gradient methods:** you could instead climb the likelihood directly with [[Gradient Descent]] (see [[Maximum Likelihood]]). EM needs no step size to tune and often has simple closed-form updates, but gradient methods fit more kinds of models. Deep networks use gradients, not EM.
- **Where it lives today:** classic statistics and [[Unsupervised Learning]], plus the idea behind variational methods used in [[Generative Models]] (from memory).
- **Link to Bayes:** the E-step is Bayes' rule applied to the hidden option (see [[Bayesian Methods]]).

```python
import numpy as np
heads = np.array([5, 9, 8, 4, 7]); n = 10                  # five sets of 10 flips; which coin was used is hidden
tA, tB = 0.6, 0.5                                          # starting guesses
prev = -1e9
for it in range(30):
    lA = tA ** heads * (1 - tA) ** (n - heads); lB = tB ** heads * (1 - tB) ** (n - heads)
    ll = np.log(0.5 * lA + 0.5 * lB).sum(); assert ll >= prev - 1e-12; prev = ll    # likelihood never drops
    r = lA / (lA + lB)                                     # E-step: chance each set came from coin A
    if it == 0: print(r.round(3).tolist())                 # [0.449, 0.805, 0.733, 0.352, 0.647]
    tA = (r * heads).sum() / (r * n).sum(); tB = ((1 - r) * heads).sum() / ((1 - r) * n).sum()   # M-step
    if it == 0: print(round(tA, 3), round(tB, 3), round(ll, 3))   # 0.713 0.581 -33.094
print(round(tA, 3), round(tB, 3), round(ll, 3))            # 0.797 0.52 -31.57
```

The first line shows the E-step: with the starting guess, the set with 9 heads is 80.5% likely to be coin A, and the set with 4 heads only 35.2%. The second line shows one M-step: the guesses move from (0.6, 0.5) to (0.713, 0.581). After 30 rounds, A settles near 0.80 and B near 0.52, and the score (log-likelihood) climbs from -33.09 to -31.57. The `assert` confirms that the score never went down. Coin sets are assumed equally likely to use A or B (0.5 each), which is fixed here for simplicity.

## 🔴 4. Deep / formal
**The hard part.** If each data point $x_i$ has a hidden label $z$, the log-likelihood of the data contains a **log of a sum** over the hidden options:

$$
\ell(\theta)=\sum_{i=1}^{n}\log\sum_{z}p(x_i,z\mid\theta)
$$

**Where:**
- $\theta$ stands for the model's settings. $x_i$ is the observed data point $i$.
- $z$ is the hidden variable (which coin, which bell). The inner sum runs over all its possible values.
- $p(x_i,z\mid\theta)$ is the chance of seeing both the data point and that hidden option.
- **Matters:** because the log sits outside a sum, the equation does not split into simple pieces, and there is usually no formula for the best $\theta$. If $z$ were seen, the inner sum would vanish and the problem would be easy. EM works around this by temporarily treating $z$ as known.

**Example (tiny numbers):** one coin set with 9 heads in 10 flips, with the two coins at $\theta_A=0.6$ and $\theta_B=0.5$, each used half the time.
- Coin A: $0.6^9\times0.4=0.004031$. Coin B: $0.5^{10}=0.000977$.
- Sum over the hidden coin: $0.5\times0.004031+0.5\times0.000977=0.002504$.
- Log: $\log0.002504=\mathbf{-5.990}$.
- Meaning: this is that one set's contribution to the log-likelihood. The log of a sum makes it hard to optimise directly.

**E-step.** Fix the current settings $\theta^{\text{old}}$. For each point, the chance of each hidden option is given by Bayes' rule:

$$
r_{iz}=p(z\mid x_i,\theta^{\text{old}})=\frac{p(x_i,z\mid\theta^{\text{old}})}{\sum_{z'}p(x_i,z'\mid\theta^{\text{old}})}
$$

**Where:**
- $r_{iz}$ is the **responsibility**: the chance, given the data point and the current settings, that its hidden label is $z$. For each $i$, the $r_{iz}$ over all $z$ add up to 1.
- The top is the chance of the point together with option $z$. The bottom adds that up over every option $z'$.
- **Matters:** this is just Bayes' rule (see [[Bayesian Methods]]). The division makes the shares add to 1. These shares are what "fill in" the hidden variable. If they were wrong, the M-step would fit the wrong thing.

**Example (tiny numbers):** the set with 9 heads again, equal chance of each coin.
- Coin A: $0.004031$. Coin B: $0.000977$. Total: $0.005008$.
- $r_A=0.004031/0.005008=\mathbf{0.805}$, and $r_B=0.195$.
- Meaning: given 9 heads, coin A is 80.5% likely under the current guess. A set with 5 heads would give coin A only $0.449$, since 5 heads is about what both coins give.

**M-step.** Choose the new settings that maximise the data's likelihood, with every point counted by its responsibilities:

$$
\theta^{\text{new}}=\arg\max_{\theta}\;\sum_{i=1}^{n}\sum_{z}r_{iz}\,\log p(x_i,z\mid\theta)
$$

**Where:**
- $r_{iz}$ are the weights from the E-step, held fixed here.
- $\log p(x_i,z\mid\theta)$ is the log chance of the point and option $z$ with the *new* settings $\theta$ to be found.
- The double sum counts each point once for every hidden option, weighted by how likely that option is.
- **Matters:** now the log is *inside* the sum, so the problem is easy again: it breaks into one simple piece per option. For the coins, the result is a weighted head fraction: for coin A, (total weighted heads) divided by (total weighted flips). Without the weights, you would be forced to pick one coin per set, throwing away the uncertainty.

**Example (tiny numbers):** the five coin sets (5, 9, 8, 4, 7 heads of 10) with the first-round responsibilities for coin A of $0.449,\,0.805,\,0.733,\,0.352,\,0.647$.
- Weighted heads for A: $0.449\times5+0.805\times9+0.733\times8+0.352\times4+0.647\times7=21.29$.
- Weighted flips for A: $(0.449+0.805+0.733+0.352+0.647)\times10=2.986\times10=29.86$.
- $\theta_A^{\text{new}}=21.29/29.86=\mathbf{0.713}$.
- Meaning: coin A moves from the guess 0.6 to 0.713, because the sets that likely came from A had many heads. Coin B moves the same way, to 0.581. This matches the code.

**Further notes:**
- **Why the score rises:** the M-step maximises a lower bound that touches the true log-likelihood at the current settings, so raising the bound cannot lower the likelihood. This argument is standard (from memory) and is not reproduced here.
- **Initialisation matters:** a poor start can end at a poor peak. Using k-means to place the starting centres is a common practice (from memory).
- **Stopping:** stop when the log-likelihood changes by less than a small tolerance, or after a fixed number of rounds.
- **Not a clustering method by itself:** it is a way of *fitting* a model with hidden variables. Clustering is one use.

**What came before, and how this improved on it:**
- **Before:** maximum likelihood on fully observed data, where the best parameters can be found directly.
- **Its limit:** when some variables are hidden (here, which group each point belongs to), the likelihood has no direct answer.
- **What changed:** alternate two steps: guess the hidden labels softly from the current parameters (E), then re-fit the parameters as if those guesses were right (M). Each round cannot lower the likelihood (Dempster, Laird and Rubin, 1977; from memory).
- **How much better:** in the code below, from a poor start (means 0 and 1, spread 2) the log-likelihood goes $-985.21$, $-955.65$ (iteration 1), $-954.83$ (5), $-945.86$ (10), $-830.18$ (15) and $-830.05$ (20). It never went down, and the fitted means 1.96 below and 2.96 above zero are close to the true $-2$ and $3$. The plateau around iterations 1 to 10 is real: EM can be slow, and it can stop at a local peak.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
x = np.r_[rng.normal(-2, 1, 200), rng.normal(3, 1, 200)]           # two hidden groups, labels not given
mu, sd, pi = np.array([0.0, 1.0]), np.array([2.0, 2.0]), np.array([0.5, 0.5])
N = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
prev, ll = -1e9, []
for it in range(31):
    p = np.stack([pi[k] * N(x, mu[k], sd[k]) for k in range(2)]); L = np.log(p.sum(0)).sum(); ll.append(L)
    r = p / p.sum(0)                                                # E step: how much each group owns each point
    nk = r.sum(1); mu = (r * x).sum(1) / nk; sd = np.sqrt((r * (x - mu[:, None]) ** 2).sum(1) / nk); pi = nk / len(x)   # M step
for i in (0, 1, 5, 10, 15, 20, 30): print(f"iteration {i:2d}: log-likelihood {ll[i]:.2f}")
print("never went down:", all(b >= a - 1e-9 for a, b in zip(ll, ll[1:])), " found means", np.round(np.sort(mu), 2), "(true -2 and 3)")
```

## Related
- [[Maximum Likelihood]] · [[Gaussian Mixture Model]] · [[Clustering]] · [[Bayesian Methods]] · [[Unsupervised Learning]] · [[Probability]] · [[Generative Models]] · [[Gradient Descent]] · [[Entropy]] · [[KL Divergence]] · [[Bayes' Theorem]]

## Sources
- Dempster, Laird & Rubin, *Maximum likelihood from incomplete data via the EM algorithm*, 1977 (from memory)
- Do & Batzoglou, *What is the expectation maximization algorithm?*, Nature Biotechnology, 2008 (from memory; a two-coin example of this kind appears there, and the numbers in this note were computed fresh)
- Bishop, *Pattern Recognition and Machine Learning*, chapter 9 (from memory)
- Murphy, *Machine Learning: A Probabilistic Perspective*, chapter 11 (from memory)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/clustering Expectation-Maximization (EM):::A chicken-and-egg fix. If you knew the hidden labels you could find the settings, and if you knew the settings you could guess the labels. EM guesses the labels softly (E-step), refits the settings (M-step), and repeats.

#flashcards/ml/formula #flashcards/topic/clustering Write the log-likelihood with a hidden variable and explain each symbol.
?
$$\ell(\theta)=\sum_{i=1}^{n}\log\sum_{z}p(x_i,z\mid\theta)$$
- $\theta$: the model's settings. $x_i$: the observed data point $i$.
- $z$: the hidden variable (which coin, which bell). The inner sum runs over all its values.
- $p(x_i,z\mid\theta)$: the chance of seeing both the data point and that hidden option.
- The log sits outside a sum, so the equation does not split into simple pieces and there is usually no formula for the best $\theta$. If $z$ were seen, the inner sum would vanish and the problem would be easy. EM temporarily treats $z$ as known.

#flashcards/ml/example #flashcards/topic/clustering One coin set with 9 heads in 10 flips, coins $\theta_A=0.6$, $\theta_B=0.5$, each used half the time. What is that set's log-likelihood?::Coin A: $0.6^9\times0.4=0.004031$. Coin B: $0.5^{10}=0.000977$. Mix $0.5\times0.004031+0.5\times0.000977=0.002504$. $\log=-5.990$. The log of a sum makes it hard to optimise directly.

#flashcards/ml/formula #flashcards/topic/clustering Write the E-step responsibility and explain each symbol.
?
$$r_{iz}=p(z\mid x_i,\theta^{\text{old}})=\frac{p(x_i,z\mid\theta^{\text{old}})}{\sum_{z'}p(x_i,z'\mid\theta^{\text{old}})}$$
- $r_{iz}$: the responsibility, the chance given the data point and current settings that its hidden label is $z$. For each $i$ the $r_{iz}$ over all $z$ add to 1.
- The top is the chance of the point together with option $z$. The bottom adds it over every option $z'$.
- This is Bayes' rule. The division makes the shares add to 1. These shares "fill in" the hidden variable. If they were wrong, the M-step would fit the wrong thing.

#flashcards/ml/example #flashcards/topic/clustering The 9-heads set with equal coin use. What are $r_A$ and $r_B$?::Coin A $0.004031$, coin B $0.000977$, total $0.005008$. $r_A=0.805$, $r_B=0.195$. A set with 5 heads would give coin A only 0.449, since 5 heads is about what both coins give.

#flashcards/ml/formula #flashcards/topic/clustering Write the M-step and explain each symbol.
?
$$\theta^{\text{new}}=\arg\max_{\theta}\;\sum_{i=1}^{n}\sum_{z}r_{iz}\,\log p(x_i,z\mid\theta)$$
- $r_{iz}$: the weights from the E-step, held fixed.
- $\log p(x_i,z\mid\theta)$: the log chance of the point and option $z$ with the new settings $\theta$ to be found.
- The double sum counts each point once per hidden option, weighted by how likely that option is.
- Now the log is inside the sum, so the problem is easy again and splits into one piece per option. For the coins the result is a weighted head fraction. Without the weights you would be forced to pick one coin per set.

#flashcards/ml/example #flashcards/topic/clustering Five coin sets (5, 9, 8, 4, 7 heads of 10) with coin-A responsibilities 0.449, 0.805, 0.733, 0.352, 0.647. What is the new $\theta_A$?::Weighted heads $0.449\times5+0.805\times9+0.733\times8+0.352\times4+0.647\times7=21.29$. Weighted flips $2.986\times10=29.86$. $\theta_A=21.29/29.86=0.713$. Coin A moves from 0.6 to 0.713. Coin B moves to 0.581.

#flashcards/ml/image #flashcards/topic/clustering ![[em-old-faithful.gif]]
What does this picture show?
?
EM fitting a two-bell Gaussian mixture to Old Faithful geyser data (each dot is one eruption: how long it lasted, and the wait before it). The red ovals start badly placed and, step by step, settle onto the two clumps. It is an animation. (From the note's caption: 3mta3, Wikimedia Commons.)

#flashcards/ml/cloze #flashcards/topic/clustering EM alternates the ==E-step== (guess hidden labels softly) and the ==M-step== (refit the settings with those weights).
