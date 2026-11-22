---
tags: [math, information-theory, probability, machine-learning, loss, distance]
status: growing
level: 4
tier: 2
---

# KL Divergence

KL divergence (full name **Kullback–Leibler divergence**) is a number that says **how different one set of chances is from another**.

## 🟢 1. Simple (age 12)
Say you have two spinners. Each one lands on red, green or blue, but they are painted differently. KL divergence answers: "If I *think* the game uses spinner B, but it really uses spinner A, how wrong will I be, on average?" If both spinners are painted the same, the answer is exactly 0. The more different they are, the bigger the number.

**Example:** You guess your friend's favourite ice cream. You assume all three flavours are equally liked. Your friend actually picks chocolate half the time. Your guesses are a bit off every time, and the KL number adds up how off you are across many guesses.

## 🔵 2. How it works
1. You have two sets of chances over the same outcomes: the **true** one $P$ and a **guess** $Q$. (See [[Probability]].)
2. For each outcome, compare how likely it really is with how likely you *thought* it was.
3. Outcomes that happen often but that you thought were rare cost you the most.
4. Add it up, weighting each outcome by how often it really happens.
5. The result is never negative. It is 0 only when $Q$ is exactly $P$.
6. It is **not** a true distance: swapping $P$ and $Q$ gives a different number.

![[kl-divergence-example.svg]]
*Two sets of chances over the outcomes 0, 1 and 2. Left, P: a binomial with chance 0.4 and 2 tries, giving about 0.36, 0.48 and 0.16. Right, Q: all three equal at 1/3. The KL divergence measures how far Q is from P. Image: K687 ypo, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Kullback%E2%80%93Leibler_distributions_example_1.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Also called relative entropy.** It is built from [[Entropy]]: it is the *extra* surprise you pay for using the wrong chances $Q$ instead of the true $P$.
- **Not symmetric:** $D(P\|Q)\ne D(Q\|P)$ in general. For the picture's two distributions the two directions give 0.123 and 0.141 bits. So the order is a real choice.
- **Where it shows up in ML:**
  - **Training a classifier:** minimising the cross-entropy loss is the same as minimising KL, because the entropy of the data does not depend on the model (see [[Loss Function]], [[Logistic Regression]]).
  - **Variational autoencoders:** a KL term keeps the learned codes near a simple distribution (see [[Generative Models]]).
  - **RLHF:** a KL penalty keeps the tuned model close to the original one, so it does not drift into odd text (see [[RLHF]]).
  - **Distillation:** a small model is trained to match a big model's output chances by lowering the KL (from memory, a common use).
- **Danger of zeros:** if $Q$ says an outcome is impossible ($q=0$) but it can really happen ($p>0$), the KL is infinite. Real code adds a tiny number to avoid this.
- **Direction in practice:** minimising $D(P\|Q)$ makes $Q$ spread out to cover everything $P$ does. Minimising $D(Q\|P)$ makes $Q$ lock onto one part of $P$ (a commonly stated rule of thumb; from memory).

```python
import numpy as np
def kl(p, q):                                # bits; p = true, q = guess
    p, q = np.array(p, float), np.array(q, float); m = p > 0
    return (p[m] * np.log2(p[m] / q[m])).sum()
def H(p): p = np.array(p, float); p = p[p > 0]; return -(p * np.log2(p)).sum()

P = [.36, .48, .16]; Q = [1/3, 1/3, 1/3]     # the two distributions in the picture
print(round(kl(P, Q), 4), round(kl(Q, P), 4))           # 0.1231 0.1406   (not symmetric)
print(round(kl(P, P), 4))                                # 0.0             (same -> zero)
ce = -(np.array(P) * np.log2(Q)).sum()                   # cross-entropy of P under Q
print(round(H(P), 4), round(ce, 4), round(H(P) + kl(P, Q), 4))   # 1.4619 1.585 1.585
```

The last line is the key identity: cross-entropy (1.585) equals entropy (1.4619) plus KL (0.1231). Since the entropy of the data is fixed, lowering the cross-entropy lowers the KL by the same amount.

## 🔴 4. Deep / formal
**Definition.** For true chances $P$ and guessed chances $Q$ over the same outcomes:

$$
D_{\mathrm{KL}}(P\,\|\,Q)=\sum_{i=1}^{n}p_i\log\frac{p_i}{q_i}
$$

**Where:**
- $p_i$ is the true chance of outcome $i$ and $q_i$ is the guessed chance. The $\log$ can be base 2 (bits) or natural (nats).
- $\log(p_i/q_i)$ says how much more (or less) likely the outcome really is compared with your guess. It is $0$ when they agree.
- The outer weight $p_i$ means the average is taken over what really happens, not over what you believed.
- **Matters:** this is why the order matters. Weighting by $p_i$ punishes you where reality lives. If $q_i=0$ while $p_i>0$, the term is infinite. Without the weights, the sum could cancel to zero for different distributions.

**Example (tiny numbers):** $P=(0.36,\,0.48,\,0.16)$ and $Q=(1/3,\,1/3,\,1/3)$, with $\log_2$.
- Outcome 0: $0.36\times\log_2(0.36/0.3333)=0.36\times0.1110=0.0400$.
- Outcome 1: $0.48\times\log_2(0.48/0.3333)=0.48\times0.5261=0.2525$.
- Outcome 2: $0.16\times\log_2(0.16/0.3333)=0.16\times(-1.0589)=-0.1694$.
- Sum: $0.0400+0.2525-0.1694=\mathbf{0.1231}$ bits.
- Meaning: using the flat guess instead of the true chances costs about 0.12 extra bits per outcome. One term is negative, but the total is never negative.

**The cross-entropy identity.** Cross-entropy splits into the entropy of the truth plus the KL:

$$
H(P,Q)=H(P)+D_{\mathrm{KL}}(P\,\|\,Q)
$$

**Where:**
- $H(P,Q)=-\sum_i p_i\log q_i$ is the cross-entropy: your average surprise using the guess $Q$ while reality is $P$.
- $H(P)=-\sum_i p_i\log p_i$ is the entropy: the unavoidable surprise even with perfect knowledge. It does not depend on your model.
- $D_{\mathrm{KL}}(P\|Q)$ is the extra surprise caused only by being wrong.
- **Matters:** because $H(P)$ is a fixed constant of the data, lowering the cross-entropy is exactly the same as lowering the KL. It also shows $H(P,Q)\ge H(P)$, since KL is never negative.

**Example (tiny numbers):** the same $P$ and $Q$.
- Entropy: $H(P)=1.4619$ bits.
- KL: $D=0.1231$ bits.
- Sum: $1.4619+0.1231=\mathbf{1.5850}$ bits.
- Check directly: $H(P,Q)=-\sum p_i\log_2(1/3)=\log_2 3=1.5850$. They match.
- Meaning: a model can never beat 1.4619 bits on this data. Everything above that is the cost of its wrong beliefs.

**Further notes:**
- **Why never negative:** this follows from a fact about the log curve (Jensen's inequality), sometimes called Gibbs' inequality. Stated from memory, the proof is not shown here.
- **Not a metric:** it breaks symmetry and the triangle rule. Symmetric cousins exist (for example Jensen–Shannon divergence, from memory).
- **Continuous version:** the sum becomes an integral, and for two normal distributions there is a closed form (see [[Normal Distribution]]; not derived here).

**What came before, and how this improved on it:**
- **Before:** compare two probability lists with the straight-line distance between them.
- **Its limit:** it treats a difference of 0.1 the same everywhere, but 1% against 11% is a very different surprise from 45% against 55%.
- **What changed:** measure the extra surprise of using $Q$ when the truth is $P$ (Kullback and Leibler, 1951; from memory, not checked). It weighs differences by how likely the true event is, and it is not symmetric.
- **How much better:** in the code below, both pairs of lists are 0.1414 apart in a straight line. The rare-event pair has a KL of 0.0814 nats and the coin pair 0.0201, four times smaller. And $\text{KL}(p\|q)=1.614$ while $\text{KL}(q\|p)=0.637$ for one pair, so order matters. That makes KL a loss function for probabilities, not a true distance.

**Check it yourself:**

```python
import numpy as np
kl = lambda p, q: float(np.sum(p * np.log(p / q)))
l2 = lambda p, q: float(np.linalg.norm(p - q))
A = (np.array([0.01, 0.99]), np.array([0.11, 0.89])); B = (np.array([0.45, 0.55]), np.array([0.55, 0.45]))
for name, (p, q) in (("rare event 1% vs 11%", A), ("coin 45/55 vs 55/45", B)):
    print(f"{name:22s} straight-line distance {l2(p, q):.4f}   KL {kl(p, q):.4f} nats")
p, q = np.array([0.5, 0.5]), np.array([0.99, 0.01])
print("not symmetric: KL(p||q) =", round(kl(p, q), 3), "  KL(q||p) =", round(kl(q, p), 3))
```

## Related
- [[Entropy]] · [[Probability]] · [[Loss Function]] · [[Logistic Regression]] · [[Softmax]] · [[Generative Models]] · [[RLHF]] · [[Normal Distribution]] · [[Bayesian Methods]] · [[Perplexity]] · [[Maximum Likelihood]] · [[Autoencoder]] · [[Knowledge Distillation]] · [[Diffusion Models]] · [[GAN]] · [[Variational Autoencoder]] · [[PPO]] · [[Fisher Information]] · [[Mutual Information]]

## Sources
- Kullback & Leibler, *On Information and Sufficiency*, 1951 (from memory)
- Cover & Thomas, *Elements of Information Theory*, chapter 2 (from memory)
- Goodfellow, Bengio & Courville, *Deep Learning*, section 3.13 (free online)
- Bishop, *Pattern Recognition and Machine Learning*, section 1.6 (from memory)

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/information-theory KL divergence:::How wrong you are, on average, if you believe distribution Q but reality is P. It is 0 when they match, and it grows as they differ.

#flashcards/math/formula #flashcards/topic/information-theory Write the KL divergence formula and explain each symbol.
?
$$D_{\mathrm{KL}}(P\,\|\,Q)=\sum_{i=1}^{n}p_i\log\frac{p_i}{q_i}$$
- $p_i$: the true chance of outcome $i$. $q_i$: the guessed chance. The log can be base 2 (bits) or natural (nats).
- $\log(p_i/q_i)$: how much more or less likely the outcome really is than your guess. 0 when they agree.
- The outer weight $p_i$ averages over what really happens, not what you believed. This is why the order matters.
- If $q_i=0$ while $p_i>0$, the term is infinite.

#flashcards/math/formula #flashcards/topic/information-theory Write how cross-entropy, entropy and KL divergence are related, and explain each symbol.
?
$$H(P,Q)=H(P)+D_{\mathrm{KL}}(P\,\|\,Q)$$
- $H(P,Q)=-\sum_i p_i\log q_i$: cross-entropy, your average surprise using the guess $Q$ while reality is $P$.
- $H(P)=-\sum_i p_i\log p_i$: entropy, the unavoidable surprise even with perfect knowledge. It does not depend on your model.
- $D_{\mathrm{KL}}(P\|Q)$: the extra surprise caused only by being wrong.
- $H(P)$ is a constant of the data, so lowering the cross-entropy is exactly lowering the KL. Since KL is never negative, $H(P,Q)\ge H(P)$.

#flashcards/math/cloze #flashcards/topic/information-theory Cross-entropy equals ==entropy== plus ==KL divergence==. KL divergence is never ==negative==.

#flashcards/math/example #flashcards/topic/information-theory $P=(0.36,0.48,0.16)$ and $Q=(1/3,1/3,1/3)$. What is $D_{\mathrm{KL}}(P\|Q)$ in bits?::0.36 × 0.1110 + 0.48 × 0.5261 + 0.16 × (−1.0589) = 0.0400 + 0.2525 − 0.1694 = 0.1231 bits.

#flashcards/math/example #flashcards/topic/information-theory For the same $P$ and $Q$, check $H(P,Q)=H(P)+D_{\mathrm{KL}}$.::H(P) = 1.4619 bits, KL = 0.1231 bits, sum = 1.5850 bits = log₂ 3 = H(P,Q). A model can never beat 1.4619 bits on this data. The rest is the cost of its wrong beliefs.

#flashcards/math/image #flashcards/topic/information-theory ![[kl-divergence-example.svg]]
What does this picture show?
?
Two sets of chances over the outcomes 0, 1, 2. Left, P: a binomial with chance 0.4 and 2 tries (about 0.36, 0.48, 0.16). Right, Q: all three equal at 1/3. KL divergence measures how far Q is from P.
