---
tags: [machine-learning, probability, bayesian, naive-bayes, classification, statistics]
status: growing
level: 4
tier: 3
---

# Bayesian Methods

## 🟢 1. Simple (age 12)
Bayesian methods are a way to **start with a guess, then change your guess a little every time you see new evidence**. You never throw the old guess away. You blend it with what you just learned. Strong evidence moves the guess a lot, weak evidence moves it a little.

**Example:** You think your friend is probably late today, because they are late about half the time. Then you see their bike is still in the shed. That is strong evidence, so you now believe "late" much more. If instead you only saw cloudy weather, you would shift your guess just a tiny bit.

## 🔵 2. How it works
1. Begin with a **prior**: how likely each possibility is *before* seeing any data. (See [[Probability]].)
2. Collect evidence (data).
3. For each possibility, ask: **"if this were true, how likely is the data I saw?"** That is the **likelihood**.
4. Multiply prior by likelihood. Possibilities that were already likely *and* explain the data well score high.
5. Divide by the total so the numbers add up to 1. The result is the **posterior**: your updated belief.
6. Today's posterior can be tomorrow's prior, so you can keep updating as more data arrives.

![[bayes-drugs-tree.svg]]
*A probability tree for a rare condition. The first split is the prior (0.5% of people are users, 99.5% are not). The second split is how a test behaves for each group (99% correct on users, but 1% false alarms on non-users). Multiplying along each branch gives the four outcomes on the right. Image: Gnathan87, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Bayes_theorem_drugs_example_tree.svg), CC0.*

For the basic rule and a different picture of it, see [[Probability]]. This note is about using it for learning from data.

## 🟠 3. Technical detail
- **Prior, likelihood, posterior:** posterior is proportional to likelihood times prior. The total (the **evidence**) just makes the answer add up to 1.
- **Natural frequencies:** it is often easiest to imagine 100,000 people and count. In the tree above, 495 people are users who test positive and 995 are non-users who test positive. So among positives, only 495 out of 1,490 are users.
- **Naive Bayes classifier:** a simple [[Classification]] method. To label a message (spam or not), it multiplies the prior by the chance of each word *given* each label, **pretending the words are independent**. That pretence is false, which is why it is called "naive", yet it often works well on text and is very fast.
- **Smoothing:** if a word never appeared with a label in training, its chance is 0 and wipes out the whole product. **Laplace smoothing** adds 1 to every count so nothing is exactly 0.
- **Beta-Bernoulli (coins):** to estimate a chance $p$ from yes/no outcomes, a Beta distribution as the prior stays a Beta distribution after seeing data. Updating is just adding counts.
- **Point estimates:** **MLE** (maximum likelihood) uses only the data. **MAP** (maximum a posteriori) picks the most probable value after including the prior. A full Bayesian keeps the whole posterior and so knows how unsure it is.
- **Links to the rest of the vault:**
  - [[Regularization]]: adding a penalty is the same as choosing a prior (see Further notes).
  - [[Overfitting]]: a prior is a gentle pull against fitting noise.
  - [[Generative Models]]: naive Bayes is a (very simple) generative classifier, because it models how the data arises from each label.
  - [[Evaluation Metrics]]: a classifier's posterior chances should be *calibrated*, which is something you can check.
- **Cautions:**
  - A different prior can give a different answer, especially with little data. Say which prior you used.
  - Exact Bayesian answers are often too costly to compute for big models, so approximations are used (sampling or variational methods; not covered here).
  - The naive independence assumption can make the printed probabilities too extreme, even when the ranking is right (commonly reported; from memory).

```python
import math
from collections import Counter

# --- 1. Bayes' rule on the tree numbers ---
pu, sens, fp = 0.005, 0.99, 0.01
pos_user, pos_non = pu * sens, (1 - pu) * fp
print(round(pos_user, 5), round(pos_non, 5), round(pos_user / (pos_user + pos_non), 4))
# 0.00495 0.00995 0.3322   a positive test means only a 33% chance of being a user

# --- 2. Naive Bayes with Laplace smoothing ---
spam = ["win money now", "win prize now", "win money"]
ham = ["meeting now", "lunch meeting today", "project meeting"]
V = len({w for m in spam + ham for w in m.split()})               # vocabulary size = 8
cs = Counter(w for m in spam for w in m.split()); ns = sum(cs.values())   # 8 spam words
ch = Counter(w for m in ham for w in m.split());  nh = sum(ch.values())   # 7 ham words
def score(msg, c, n, prior=0.5):
    return math.log(prior) + sum(math.log((c[w] + 1) / (n + V)) for w in msg.split())
ls, lh = score("win now", cs, ns), score("win now", ch, nh)
print(round(math.exp(ls), 5), round(math.exp(lh), 5), round(1 / (1 + math.exp(lh - ls)), 3))
# 0.02344 0.00444 0.841   "win now" is 84% likely to be spam

# --- 3. Beta-Bernoulli: prior Beta(2,2), then 7 heads and 3 tails ---
a, b, heads, tails = 2, 2, 7, 3
A, B = a + heads, b + tails
print(A, B, round(A / (A + B), 3), heads / (heads + tails), round((A - 1) / (A + B - 2), 3))
# 9 5 0.643 0.7 0.667   posterior Beta(9,5); mean 0.643; MLE 0.7; MAP 0.667
```

## 🔴 4. Deep / formal
**Bayes' rule for a model.** Let $\theta$ be whatever we do not know (a coin's chance, a model's weights, a class label) and $D$ the data:

$$
P(\theta\mid D)=\frac{P(D\mid\theta)\,P(\theta)}{P(D)},\qquad P(D)=\sum_{\theta}P(D\mid\theta)\,P(\theta)
$$

**Where:**
- $\theta$ ("theta") is the unknown thing. $D$ is the data you saw.
- $P(\theta)$ is the **prior**: belief before the data. $P(D\mid\theta)$ is the **likelihood**: how likely the data would be if $\theta$ were true. $P(\theta\mid D)$ is the **posterior**: belief after the data.
- $P(D)$ is the **evidence**, the sum over every possible $\theta$ of likelihood times prior. **Matters:** it is the same for every $\theta$, so it only rescales. Without dividing by it, the numbers would not add up to 1 and could not be read as chances.
- **Matters:** a rare cause (small prior) needs very strong evidence to become the likely one. Ignoring the prior (the "base rate") is a common mistake.

**Example (tiny numbers):** $\theta\in\{\text{user},\text{non-user}\}$ with prior $0.005$ and $0.995$. A positive test has likelihood $0.99$ for users and $0.01$ for non-users.
- Likelihood times prior: users $0.99\times0.005=\mathbf{0.00495}$, non-users $0.01\times0.995=\mathbf{0.00995}$.
- Evidence: $0.00495+0.00995=\mathbf{0.0149}$.
- Posterior for user: $0.00495/0.0149=\mathbf{0.332}$.
- Meaning: even with a 99%-accurate test, a positive result means only about a one in three chance, because real users are so rare that the false alarms from the huge non-user group outnumber them.

**Naive Bayes.** For a label $y$ and features $x_1,\dots,x_m$ (for example, which words appear), assume the features are independent *given the label*:

$$
P(y\mid x_1,\dots,x_m)\;\propto\;P(y)\prod_{j=1}^{m}P(x_j\mid y),\qquad P(x_j\mid y)=\frac{\text{count}(x_j,y)+1}{N_y+V}
$$

**Where:**
- $y$ is the label (spam or ham) and $x_j$ is the $j$-th word of the message. $m$ is how many words it has.
- $\propto$ means "proportional to": compute the right side for every label, then rescale so they add to 1.
- $P(y)$ is the prior share of each label. $\prod_j$ means "multiply over all the words".
- $\text{count}(x_j,y)$ is how many times word $x_j$ appeared in training messages with label $y$. $N_y$ is the total word count for label $y$. $V$ is the number of different words in the vocabulary.
- The $+1$ and $+V$ are **Laplace smoothing**. **Matters:** without them a word never seen with a label gives chance 0, and the whole product becomes 0 no matter what the other words say.
- **Matters:** the independence assumption turns a huge table of word combinations into a few simple counts. It is wrong in practice (words come in pairs like "free" and "prize"), so treat the output as a ranking more than a trustworthy probability.

**Example (tiny numbers):** message "win now", with $V=8$, spam words $N=8$ (spam counts: win 3, now 2), ham words $N=7$ (ham counts: win 0, now 1), equal priors $0.5$.
- Spam: $0.5\times\frac{3+1}{8+8}\times\frac{2+1}{8+8}=0.5\times0.25\times0.1875=\mathbf{0.02344}$.
- Ham: $0.5\times\frac{0+1}{7+8}\times\frac{1+1}{7+8}=0.5\times0.0667\times0.1333=\mathbf{0.00444}$.
- Rescale: $0.02344/(0.02344+0.00444)=\mathbf{0.841}$.
- Meaning: the message is about 84% likely to be spam. Note "win" never appeared in ham, so without smoothing the ham score would be exactly 0 and the answer would claim 100% certainty from a tiny sample.

**Updating a coin estimate: Beta prior.** Take the unknown chance of heads $p$, with prior $\text{Beta}(a,b)$. After seeing $h$ heads and $t$ tails, the posterior is also a Beta:

$$
p\mid D\;\sim\;\text{Beta}(a+h,\;b+t),\qquad \mathbb E[p\mid D]=\frac{a+h}{a+b+h+t}
$$

**Where:**
- $a$ and $b$ are the prior's two numbers. You can read them as "imaginary earlier heads and tails". $\text{Beta}(2,2)$ is a mild guess that the coin is near fair.
- $h$ and $t$ are the real heads and tails counted. $n=h+t$.
- $\mathbb E[p\mid D]$ is the **posterior mean**: the average value of $p$ under the updated belief.
- **Matters:** the posterior mean sits between the prior's mean $a/(a+b)$ and the data's rate $h/n$. With little data it stays near the prior. With lots of data the prior's pull fades. Without the prior counts, three heads in a row would force you to claim $p=1$.

**Example (tiny numbers):** prior $\text{Beta}(2,2)$, then 7 heads and 3 tails.
- New Beta: $(2+7,\ 2+3)=\mathbf{(9,5)}$.
- Posterior mean: $9/(9+5)=\mathbf{0.643}$.
- For comparison, the data alone says $7/10=0.7$, so the prior pulled the answer down toward 0.5.
- Meaning: the estimate is a compromise between a mild "probably fair" belief and the observed 70%. Feeding the 10 flips in one at a time gives the same $\text{Beta}(9,5)$, so order does not matter.

**Further notes:**
- **MAP and regularization:** maximising the posterior means maximising $\log P(D\mid\theta)+\log P(\theta)$. The second term acts like a penalty. A Gaussian prior on weights gives an L2 penalty ([[Regularization]]), and a Laplace prior gives L1. This is a standard result, stated from memory here, and the link between the strength of the prior and $\lambda$ depends on the noise level.
- **MAP for the coin:** the mode of $\text{Beta}(9,5)$ is $(9-1)/(9+5-2)=0.667$, between the MLE 0.7 and the mean 0.643. The three summaries differ when data is small.
- **Discriminative vs generative:** naive Bayes models $P(x\mid y)$ and flips it with Bayes' rule. Logistic regression models $P(y\mid x)$ directly. Discriminative models often win with lots of data, naive Bayes can be better with very little (a commonly cited result, from memory).
- **Bayesian neural networks** and **Gaussian processes** keep a distribution over weights or functions to express uncertainty. They are costly, and this note does not cover them.
- **Not the same as "Bayesian" vs "frequentist" being a fight:** both are used in machine learning. Bayesian methods are natural when you have real prior knowledge or need honest uncertainty.

**What came before, and how this improved on it:**
- **Before:** estimate a number from the data alone, as in [[Maximum Likelihood]].
- **Its limit:** with little data the estimate can be extreme. Three heads in three flips makes "tails is impossible".
- **What changed:** state what you believed before (the prior), then update it with the data (Bayes' theorem, published in 1763; from memory). The answer is a whole range of likely values, not one number.
- **How much better:** in the code below, after 3 flips and 3 heads, the data alone says the chance of heads is 1.000 and a flat prior plus the data says 0.800. After 100 flips with 60 heads, data alone says 0.600 and the Bayesian answer 0.598, so with plenty of data the prior fades. The price is that you must choose a prior.

**Check it yourself:**

```python
heads, flips = 3, 3
mle = heads / flips
a, b = 1, 1                                   # prior: before seeing data, every coin bias is equally likely
bayes = (a + heads) / (a + b + flips)
print(f"{flips} flips, {heads} heads: chance of heads next time")
print(f"  just the data (maximum likelihood): {mle:.3f}  -> says tails is impossible")
print(f"  prior + data (posterior mean):      {bayes:.3f}")
heads, flips = 60, 100
print(f"{flips} flips, {heads} heads: data alone {heads / flips:.3f}, prior + data {(a + heads) / (a + b + flips):.3f}  (more data, the prior fades)")
```

## Related
- [[Probability]] · [[Classification]] · [[Regularization]] · [[Overfitting]] · [[Generative Models]] · [[Evaluation Metrics]] · [[Machine Learning]] · [[Supervised Learning]] · [[Normal Distribution]] · [[Loss Function]] · [[Bias-Variance]] · [[Logistic Regression]] · [[Clustering]] · [[Entropy]] · [[KL Divergence]] · [[Gaussian Mixture Model]] · [[Maximum Likelihood]] · [[Variational Autoencoder]] · [[Bayes' Theorem]] · [[Naive Bayes]] · [[Maximum A Posteriori]]

## Sources
- Bishop, *Pattern Recognition and Machine Learning*, 2006, chapters 1–2 (from memory)
- Murphy, *Probabilistic Machine Learning: An Introduction*, 2022 (from memory)
- Gelman et al., *Bayesian Data Analysis*, 3rd edition (from memory)
- Manning, Raghavan & Schütze, *Introduction to Information Retrieval*, chapter 13 (naive Bayes text classification; from memory)
- 3Blue1Brown, *Bayes' theorem* (video)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/classic-models Bayesian methods:::Start with a guess (the prior), then update it each time you see new evidence. Strong evidence moves the guess a lot, weak evidence a little. The old guess is never thrown away, only blended.

#flashcards/ml/formula #flashcards/topic/classic-models Write Bayes' rule with its evidence term and explain each symbol.
?
$$P(\theta\mid D)=\frac{P(D\mid\theta)\,P(\theta)}{P(D)},\qquad P(D)=\sum_{\theta}P(D\mid\theta)\,P(\theta)$$
- $\theta$: the unknown thing. $D$: the data you saw.
- $P(\theta)$: the prior, belief before the data. $P(D\mid\theta)$: the likelihood, how likely the data would be if $\theta$ were true. $P(\theta\mid D)$: the posterior, belief after the data.
- $P(D)$: the evidence, the sum over every $\theta$ of likelihood times prior. It is the same for every $\theta$, so it only rescales. Without dividing by it the numbers would not add to 1.
- A rare cause (small prior) needs very strong evidence to become the likely one.

#flashcards/ml/example #flashcards/topic/classic-models Prior: user 0.005, non-user 0.995. A positive test has likelihood 0.99 for users and 0.01 for non-users. What is $P(\text{user}\mid\text{positive})$?::Users $0.99\times0.005=0.00495$. Non-users $0.01\times0.995=0.00995$. Evidence $0.0149$. Posterior $0.00495/0.0149=0.332$. Even a 99%-accurate test gives only about a one in three chance, because real users are so rare that false alarms outnumber them.

#flashcards/ml/formula #flashcards/topic/classic-models Write the naive Bayes score with Laplace smoothing and explain each symbol.
?
$$P(y\mid x_1,\dots,x_m)\;\propto\;P(y)\prod_{j=1}^{m}P(x_j\mid y),\qquad P(x_j\mid y)=\frac{\text{count}(x_j,y)+1}{N_y+V}$$
- $y$: the label (spam or ham). $x_j$: the $j$-th word. $m$: how many words.
- $\propto$: proportional to. Compute the right side for every label, then rescale to add to 1.
- $P(y)$: the prior share of each label. $\prod_j$: multiply over all words.
- $\text{count}(x_j,y)$: how often word $x_j$ appeared in training messages with label $y$. $N_y$: total word count for label $y$. $V$: the number of different words.
- The $+1$ and $+V$ are Laplace smoothing. Without them a word never seen with a label gives chance 0 and wipes out the whole product.

#flashcards/ml/example #flashcards/topic/classic-models Message "win now", $V=8$, spam $N=8$ (win 3, now 2), ham $N=7$ (win 0, now 1), equal priors. What is the spam chance?::Spam: $0.5\times\frac{4}{16}\times\frac{3}{16}=0.02344$. Ham: $0.5\times\frac{1}{15}\times\frac{2}{15}=0.00444$. Rescale $0.02344/(0.02344+0.00444)=0.841$. About 84% spam. "win" never appeared in ham, so without smoothing the ham score would be 0.

#flashcards/ml/formula #flashcards/topic/classic-models Write the Beta posterior for a coin and explain each symbol.
?
$$p\mid D\;\sim\;\text{Beta}(a+h,\;b+t),\qquad \mathbb E[p\mid D]=\frac{a+h}{a+b+h+t}$$
- $a$, $b$: the prior's two numbers, read as imaginary earlier heads and tails. Beta(2,2) is a mild guess that the coin is near fair.
- $h$, $t$: the real heads and tails counted. $n=h+t$.
- $\mathbb E[p\mid D]$: the posterior mean, the average value of $p$ under the updated belief.
- It sits between the prior mean $a/(a+b)$ and the data's rate $h/n$. With little data it stays near the prior, with lots of data the prior's pull fades. Without prior counts, three heads in a row would force $p=1$.

#flashcards/ml/example #flashcards/topic/classic-models Prior Beta(2,2), then 7 heads and 3 tails. What is the posterior mean?::New Beta $(2+7,2+3)=(9,5)$. Mean $9/14=0.643$. The data alone says 0.7, so the prior pulled it toward 0.5. Feeding the flips in one at a time gives the same Beta(9,5), so order does not matter.

#flashcards/ml/image #flashcards/topic/classic-models ![[bayes-drugs-tree.svg]]
What does this picture show?
?
A probability tree for a rare condition. The first split is the prior (0.5% users, 99.5% not). The second split is how the test behaves for each group (99% correct on users, 1% false alarms on non-users). Multiplying along each branch gives the four outcomes on the right. (From the note's caption: Gnathan87, Wikimedia Commons.)

#flashcards/ml/cloze #flashcards/topic/classic-models Bayesian methods blend a ==prior== with the ==likelihood== of new data to get the ==posterior==.
