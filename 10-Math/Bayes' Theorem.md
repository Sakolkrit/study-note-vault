---
tags: [math, probability, Bayes, statistics, machine-learning]
status: growing
level: 4
tier: 1
---

# Bayes' Theorem

Bayes' theorem is the rule for **changing your mind in the right amount when you see new evidence**: you start with how likely something was, you look at how well it explains what you saw, and you end up with an updated chance. The basic formula is already introduced in [[Probability]], and [[Bayesian Methods]] uses it for learning from data. This note slows down on the rule itself and shows three ways to look at it: counting people, odds, and updating again and again.

## 🟢 1. Simple (age 12)
Suppose a friend says "I hear a meow outside". You first think about how likely a cat is in your street at all. If your street has lots of cats, "cat" is a good guess. If your street has almost none, you might wonder if it was a squeaky door. **Your guess depends on two things: how likely each answer was before you heard anything, and how well each answer explains the sound.** Bayes' theorem is the exact recipe for mixing those two things.

**Example:** You get a text with the word "FREE!" in it. About 3 in 10 of your texts are junk. Junk texts often say "free", but a few normal texts do too. If you picture 1,000 texts, you can count how many have "free" and see what share of those are junk. That share is what Bayes' theorem gives you.

## 🔵 2. How it works
Here is the counting way, which needs no formulas. The numbers are for a spam filter.
1. **Imagine 1,000 emails.** About 30% are spam, so 300 are spam and 700 are normal (this is the **prior**: how likely spam was before looking).
2. **Say how often a word shows up in each group.** The word "free" appears in 60% of spam and in 5% of normal mail (these are the **likelihoods**).
3. **Count the emails that have the word.** Spam with "free": $300\times0.6=180$. Normal mail with "free": $700\times0.05=35$.
4. **Look only at emails with the word.** There are $180+35=215$ of them. The share that is spam is $180/215\approx0.84$.
5. **That share is the answer:** after seeing "free", the email is about 84% likely to be spam (this is the **posterior**: how likely spam is after looking).
6. **Notice what happened:** 30% became 84%. The evidence moved the belief a lot, but not to 100%, because some normal mail also says "free".

![[bayes-visual-proof.svg]]
*A picture proof of Bayes' theorem. The grey box is everything that can happen (U). The hexagon is event A and the circle is event B. Where they overlap is "A and B". The top rows write each chance as one area divided by another. Multiplying P(A) by P(B|A) and multiplying P(B) by P(A|B) both give the same overlap area divided by the whole box. Setting the two equal gives the theorem (bottom right). Image: Cmglee, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Bayes_theorem_visual_proof.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **The formula:** $P(A\mid B)=\dfrac{P(B\mid A)\,P(A)}{P(B)}$. Here $A$ is the thing you want to know (spam) and $B$ is what you saw (the word "free"). Section 4 writes it out with each symbol explained.
- **Base rates matter:** a rare event (small prior) stays fairly unlikely even after fairly strong evidence. In the code, with only 1 spam email in 1,000, the same word "free" lifts the chance of spam only to about 1.2%. Forgetting the prior is called the *base-rate fallacy*, and it is a very common mistake.
- **Likelihood is not the answer:** $P(B\mid A)$ ("how often spam says free") is not the same as $P(A\mid B)$ ("how often a mail saying free is spam"). Mixing them up is called the *prosecutor's fallacy*. In the example, 60% against 84% shows they differ.
- **Three views of the same rule:**
  - **Counts:** imagine many cases and count (section 2). Easiest for people to get right.
  - **Odds form:** multiply the prior odds by a *likelihood ratio* (equation 2 below). Each piece of evidence just multiplies.
  - **Probability form:** the usual formula (equation 1 below).
- **Updating again and again:** today's posterior is tomorrow's prior. If the pieces of evidence are independent *given the cause*, you multiply their likelihood ratios one after another (equation 3 below), and the order does not matter. That is exactly what [[Bayesian Methods]] calls Naive Bayes.
- **Where it shows up in machine learning:**
  - **Spam filters and other classifiers** ([[Classification]]).
  - **Estimating unknown numbers:** [[Maximum Likelihood]] picks the value that makes the data likeliest. Bayes' theorem adds a prior, giving MAP or a full posterior (see [[Bayesian Methods]]).
  - **[[Expectation-Maximization]]:** the "responsibility" step in a [[Gaussian Mixture Model]] is Bayes' theorem applied to each data point.
  - **[[Variational Autoencoder]]:** the encoder is an approximation of a posterior that Bayes' theorem would give but that is too hard to compute exactly.
  - **[[LLM]]s:** the chance of a whole sentence is built from the chain rule, a close relative (see [[Probability]]).
- **Limits:** the answer is only as good as the prior and the likelihoods. If you guess them badly, the result looks exact but is wrong. And conditional independence of evidence is often only roughly true.

```python
import numpy as np
def post(prior, likes):                       # likes: list of (P(word|spam), P(word|ham)), one per word seen
    odds = prior / (1 - prior)
    for ls, lh in likes: odds *= ls / lh      # each word multiplies the odds by its likelihood ratio
    return odds / (1 + odds)
free, winner = (0.60, 0.05), (0.40, 0.05)
print('prior 0.30: after "free"', round(post(0.30, [free]), 4), '| after "free"+"winner"', round(post(0.30, [free, winner]), 4))
print('prior 0.001 (very rare spam): after "free"', round(post(0.001, [free]), 4))
# check against a simulation of 2,000,000 emails (words independent given spam/ham)
rng = np.random.default_rng(0); n = 2_000_000
spam = rng.random(n) < 0.30
f = rng.random(n) < np.where(spam, 0.60, 0.05); w = rng.random(n) < np.where(spam, 0.40, 0.05)
print('simulated: P(spam | free)', round(spam[f].mean(), 4), '| P(spam | free, winner)', round(spam[f & w].mean(), 4))
# natural frequencies: 1000 emails
print('counts:', 300 * 0.6, 700 * 0.05, 180 / (180 + 35))
# log-odds add up
print('log-odds: prior', round(np.log(.3 / .7), 3), '+ free', round(np.log(12), 3), '+ winner', round(np.log(8), 3), '=', round(np.log(.3 / .7) + np.log(12) + np.log(8), 3), '-> p', round(1 / (1 + np.exp(-(np.log(.3 / .7) + np.log(12) + np.log(8)))), 4))
# order does not matter
print(round(post(0.30, [winner, free]), 4) == round(post(0.30, [free, winner]), 4))
# prior 0.30: after "free" 0.8372 | after "free"+"winner" 0.9763
# prior 0.001 (very rare spam): after "free" 0.0119
# simulated: P(spam | free) 0.838 | P(spam | free, winner) 0.9765
# counts: 180.0 35.0 0.8372093023255814
# log-odds: prior -0.847 + free 2.485 + winner 2.079 = 3.717 -> p 0.9763
# True
```
The simulated chances (0.838 and 0.9765) agree with the formula (0.8372 and 0.9763) to within the random noise of 2,000,000 emails. The numbers in the example (0.30, 0.60, 0.05, 0.40) are made up for teaching and are not measured from real mail.

## 🔴 4. Deep / formal

**The theorem.** For an unknown $A$ (spam or not) and evidence $B$ (the word):

$$P(A\mid B)=\frac{P(B\mid A)\,P(A)}{P(B)},\qquad P(B)=P(B\mid A)P(A)+P(B\mid\neg A)P(\neg A)$$

**Where:**
- $P(A)$ is the prior, how likely $A$ was before the evidence. **Matters:** a tiny prior keeps the answer small even for strong evidence.
- $P(B\mid A)$ is the likelihood, how often you would see the evidence if $A$ were true. **Matters:** it measures how well $A$ explains what you saw.
- $P(B\mid\neg A)$ is the same chance if $A$ were false ($\neg A$ means "not $A$"). **Matters:** without it you cannot tell how common the evidence is among the other cases. This is the part people forget.
- $P(B)$ is the total chance of the evidence, adding both cases. **Matters:** it is the divider that makes the answer a proper chance between 0 and 1.
- $P(A\mid B)$ is the posterior, the updated chance of $A$.

**Example (tiny numbers):** $P(\text{spam})=0.3$, $P(\text{free}\mid\text{spam})=0.6$, $P(\text{free}\mid\text{not spam})=0.05$.
1. Top: $0.6\times0.3=0.18$.
2. $P(\text{free})=0.18+0.05\times0.7=0.18+0.035=0.215$.
3. $P(\text{spam}\mid\text{free})=0.18/0.215=0.8372$.
4. Check with counts: $180$ spam with the word out of $215$ with the word is the same $0.8372$.
5. Meaning: seeing "free" lifts the chance of spam from 30% to about 84%. It does not reach 100%, since 35 of the 215 emails with the word are normal.

**The odds form.** Odds are the chance of "yes" divided by the chance of "no". Dividing the theorem for $A$ by the same theorem for $\neg A$ cancels $P(B)$ and gives

$$\underbrace{\frac{P(A\mid B)}{P(\neg A\mid B)}}_{\text{posterior odds}}=\underbrace{\frac{P(A)}{P(\neg A)}}_{\text{prior odds}}\times\underbrace{\frac{P(B\mid A)}{P(B\mid\neg A)}}_{\text{likelihood ratio}}$$

**Where:**
- Prior odds are $P(A)/P(\neg A)$. **Matters:** it is the starting belief written as "$x$ to 1" (for example $0.3/0.7=3$ to 7).
- The likelihood ratio is how many times more likely the evidence is if $A$ is true than if it is false. **Matters:** a ratio of 1 means the evidence says nothing, above 1 favours $A$, below 1 favours $\neg A$. It is the "strength" of the evidence in one number.
- Posterior odds are the updated belief in the same "$x$ to 1" form. **Matters:** you can turn odds $o$ back into a chance with $o/(1+o)$.
- The step that cancels $P(B)$ is a short algebra step I did not write out, and I checked it numerically instead.

**Example (tiny numbers):** Prior odds $0.3/0.7=0.4286$. Likelihood ratio for "free": $0.6/0.05=12$.
1. Posterior odds $=0.4286\times12=5.143$.
2. Back to a chance: $5.143/(1+5.143)=0.8372$, the same answer as before.
3. Meaning: the word "free" multiplies the odds by 12. If the prior had been tiny (odds about 0.001), the odds after the word would be about 0.012, a chance of only about 1.2%, as the code printed. The same evidence moves a belief far when you start near the middle, and little when you start in a corner.

**Updating again and again.** If the pieces of evidence $B_1,\dots,B_k$ are independent given $A$ and given $\neg A$, then

$$\text{odds after all}=\text{prior odds}\times\prod_{i=1}^{k}\mathrm{LR}_i,\qquad \ln(\text{odds})=\ln(\text{prior odds})+\sum_{i=1}^{k}\ln \mathrm{LR}_i$$

**Where:**
- $\mathrm{LR}_i=P(B_i\mid A)/P(B_i\mid\neg A)$ is the likelihood ratio of piece $i$. **Matters:** each piece has its own strength, and strong ones count more.
- $\prod$ means "multiply all of them" and $\sum$ means "add all of them". **Matters:** in log form the pieces simply *add up*, so evidence behaves like a running score, and the order of arrival does not change the final total.
- The independence assumption. **Matters:** if the pieces are really copies of the same signal, such as two words that always appear together, multiplying them counts the same evidence twice and makes you overconfident. This is the weak spot of Naive Bayes.

**Example (tiny numbers):** Start at prior odds $0.4286$ (30% spam). See "free" (LR $12$) and then "winner" ($P=0.4$ in spam, $0.05$ in normal mail, LR $8$).
1. After "free": $0.4286\times12=5.143$.
2. After "winner": $5.143\times8=41.14$.
3. Chance: $41.14/42.14=0.9763$ (a simulation of 2,000,000 emails gave 0.9765).
4. In logs: $-0.847+2.485+2.079=3.717$, and $1/(1+e^{-3.717})=0.9763$. The same.
5. Swapping the order gives the same answer.
6. Meaning: two moderately suspicious words together push 30% up to 97.6%. This running multiply-or-add is how a spam filter reads a whole message, word by word.

**Further notes (from memory, not checked here):**
- Thomas Bayes' essay was published after his death (1763), and Laplace later developed and used the rule widely. I have not checked the dates in this note.
- There is a long argument between "Bayesian" and "frequentist" ways of using probability. This note takes no side; the theorem itself is a plain consequence of the definition of conditional probability and is accepted by everyone.
- With continuous unknowns, the sum in $P(B)$ becomes an integral that is often impossible to do exactly, which is why approximations (such as those behind a [[Variational Autoencoder]]) and sampling methods are used.

**What came before, and how this improved on it:**
- **Before:** read a test's accuracy as the answer ("the test is 99% accurate, so a positive means 99% sick"). This mistake is called base-rate neglect.
- **Its limit:** it ignores how rare the condition is. When few people are sick, most positives come from the healthy majority.
- **What changed:** Bayes' theorem turns "chance of a positive given sick" into "chance of sick given a positive". It is named after Thomas Bayes, published after his death in 1763 (from memory, not checked).
- **How much better:** in the code below, with 99% sensitivity, a 5% false alarm rate and 1% of people sick, the gut feel is 0.99 and the calculation gives 0.167, about 1 in 6. A simulation of 1,000,000 people also gives 0.167. The gut answer is almost 6 times too high.

**Check it yourself:**

```python
import numpy as np
sens, spec, prev = 0.99, 0.95, 0.01
pos = sens * prev + (1 - spec) * (1 - prev)
print("test catches 99% of sick people, wrongly flags 5% of healthy people, 1% are sick")
print("gut feel: a positive test means about", sens)
print("Bayes' theorem: chance of being sick after a positive test =", round(sens * prev / pos, 3))
rng = np.random.default_rng(0); sick = rng.random(1_000_000) < prev
flag = np.where(sick, rng.random(1_000_000) < sens, rng.random(1_000_000) < 1 - spec)
print("simulation of 1,000,000 people:", round(sick[flag].mean(), 3))
```

## Related
[[Probability]] · [[Bayesian Methods]] · [[Maximum Likelihood]] · [[Expectation-Maximization]] · [[Gaussian Mixture Model]] · [[Variational Autoencoder]] · [[Classification]] · [[Logistic Regression]] · [[Entropy]] · [[KL Divergence]] · [[Normal Distribution]] · [[LLM]] · [[Naive Bayes]] · [[Hypothesis Testing]]
## Sources
- Bayes' original essay (1763) and Laplace's later work (both from memory; I did not open them for this note).
- Figure: [Wikimedia Commons, Bayes_theorem_visual_proof.svg](https://commons.wikimedia.org/wiki/File:Bayes_theorem_visual_proof.svg), by Cmglee, CC BY-SA 4.0.
- The spam numbers are invented for teaching. The code, the simulation and every Example number were run for this note (NumPy only).

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/probability Bayes' theorem:::The exact recipe for updating a guess. It mixes how likely each answer was before the evidence (the prior) with how well each answer explains the evidence (the likelihood).

#flashcards/math/formula #flashcards/topic/probability Write Bayes' theorem, including how to get $P(B)$, and explain each symbol.
?
$$P(A\mid B)=\frac{P(B\mid A)\,P(A)}{P(B)},\qquad P(B)=P(B\mid A)P(A)+P(B\mid\neg A)P(\neg A)$$
- $P(A)$: the prior, how likely $A$ was before the evidence. A tiny prior keeps the answer small even for strong evidence.
- $P(B\mid A)$: the likelihood, how often you would see the evidence if $A$ were true.
- $P(B\mid\neg A)$: the same chance if $A$ were false ($\neg A$ is "not $A$"). People forget this part.
- $P(B)$: the total chance of the evidence, adding both cases. It is the divider that keeps the answer between 0 and 1.
- $P(A\mid B)$: the posterior, the updated chance of $A$.

#flashcards/math/formula #flashcards/topic/probability Write the odds form of Bayes' theorem and explain each part.
?
$$\frac{P(A\mid B)}{P(\neg A\mid B)}=\frac{P(A)}{P(\neg A)}\times\frac{P(B\mid A)}{P(B\mid\neg A)}$$
- Left side: posterior odds, the updated belief in the form "$x$ to 1". Turn odds $o$ back into a chance with $o/(1+o)$.
- $P(A)/P(\neg A)$: prior odds, the starting belief.
- $P(B\mid A)/P(B\mid\neg A)$: the likelihood ratio, how many times more likely the evidence is if $A$ is true than if false. 1 says nothing, above 1 favours $A$, below 1 favours $\neg A$.

#flashcards/math/formula #flashcards/topic/probability Write the odds after $k$ independent pieces of evidence (product form and log form), and explain each symbol.
?
$$\text{odds after all}=\text{prior odds}\times\prod_{i=1}^{k}\mathrm{LR}_i,\qquad \ln(\text{odds})=\ln(\text{prior odds})+\sum_{i=1}^{k}\ln \mathrm{LR}_i$$
- $\mathrm{LR}_i=P(B_i\mid A)/P(B_i\mid\neg A)$: the likelihood ratio of piece $i$. Strong pieces count more.
- $\prod$: multiply all of them. $\sum$: add all of them. In log form the pieces simply add up, like a running score, and the order does not matter.
- Independence is assumed. If the pieces are copies of the same signal, you count it twice and become overconfident. This is the weak spot of Naive Bayes.

#flashcards/math/cloze #flashcards/topic/probability ==Prior== is the chance before any evidence, ==likelihood== is how well an answer explains the evidence, and ==posterior== is the updated chance.

#flashcards/math/cloze #flashcards/topic/probability A likelihood ratio of ==1== says the evidence tells you nothing. Above 1 it favours ==A==, below 1 it favours ==not A==.

#flashcards/math/example #flashcards/topic/probability $P(\text{spam})=0.3$, $P(\text{free}\mid\text{spam})=0.6$, $P(\text{free}\mid\text{not spam})=0.05$. What is $P(\text{spam}\mid\text{free})$?::About 0.837. Top: 0.6 × 0.3 = 0.18. Bottom: 0.18 + 0.05 × 0.7 = 0.215. 0.18 / 0.215 = 0.837.

#flashcards/math/example #flashcards/topic/probability Start at 30% spam (prior odds 0.4286). "free" has likelihood ratio 12 and "winner" has 8. What is the final chance of spam?::0.4286 × 12 × 8 = 41.14 odds, so 41.14 / 42.14 = 0.976 (the note's simulation gave 0.9765).

#flashcards/math/concept #flashcards/topic/probability What goes wrong if you multiply likelihood ratios of evidence pieces that are not independent?::You count the same evidence twice and become overconfident. This is the weak spot of Naive Bayes.

#flashcards/math/image #flashcards/topic/probability ![[bayes-visual-proof.svg]]
What does this picture prove, and how?
?
Bayes' theorem. Each chance is one area divided by another. P(A)·P(B|A) and P(B)·P(A|B) both equal the same overlap area (A and B) divided by the whole box. Setting them equal gives the theorem.
