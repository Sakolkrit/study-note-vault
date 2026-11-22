---
tags: [machine-learning, classification, probability, Bayes, naive-bayes, text-classification]
status: growing
level: 4
tier: 2
---

# Naive Bayes

Naive Bayes is a simple way to classify things: **for each possible label, multiply how common the label is by how likely each clue is under that label, then pick the label with the biggest total**. It is called "naive" because it pretends that all the clues are independent of each other once you know the label, which is almost never exactly true, yet it often works well anyway.

The rule behind it is in [[Bayes' Theorem]], and [[Bayesian Methods]] has a short first example with word counts. This note is the full classifier: three flavours, the log trick, a hand-checked text example, a comparison with [[Logistic Regression]], and where it goes wrong.

## 🟢 1. Simple (age 12)
Suppose you want to guess whether a text message is junk. You know that 3 out of 10 messages are junk. You also know which words junk messages like ("free", "win", "prize") and which words normal messages like ("lunch", "meeting"). For a new message you check each of its words, and every word nudges your guess toward junk or toward normal. At the end you go with whichever side got the bigger push. The "naive" part is that you treat each word as a separate witness who does not talk to the others.

**Example:** A message says "free money now". "Free" and "money" are things junk messages say a lot, so they push strongly toward junk. "Now" is said by both kinds, so it pushes a little. Add up the pushes, and junk wins.

## 🔵 2. How it works
1. **Collect labelled examples** (messages marked junk or normal).
2. **Count how common each label is.** If 30% are junk, the starting guess for a new message is 30% junk (the **prior**).
3. **For each label, count how often each word shows up** in messages of that label. Turn the counts into chances. Add a tiny starting count to every word so that a word never seen with a label does not get a chance of exactly zero (this is **smoothing**).
4. **To label a new message:** for each label, start with its prior and multiply by the chance of every word in the message under that label.
5. **Compare the totals.** The label with the bigger total wins. If you want percentages, divide each total by the sum of the totals.
6. **Do it with logs** (adding instead of multiplying), because multiplying many small chances makes numbers so tiny that the computer rounds them to zero (see the code).

![[naive-bayes-classifier.gif]]
*A naive Bayes classifier on two features with three classes (A, B, C). As more labelled points arrive (the dots), the classifier fits a bell-shaped region to each class. Each region is stretched only along the two feature axes, never tilted, because the method treats the two features as independent. (The picture is an animation; only the first frame shows in some viewers. The last frame is the one with three coloured regions.) Image: Jacopo Bertolotti, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Naive_Bayes_Classifier.gif), CC0.*

## 🟠 3. Technical detail
- **Three common flavours** (they differ in how a clue is modelled):
  - **Multinomial:** the clues are word *counts* in a message. The standard choice for text.
  - **Bernoulli:** the clues are yes/no ("is this word in the message?").
  - **Gaussian:** the clues are numbers (heights, pixel brightness), and for each label and each feature the method fits a bell curve (see [[Normal Distribution]]).
- **Training is just counting** (or averaging, for the Gaussian type), in one pass over the data. No [[Gradient Descent]] is needed, so it is very fast and easy to update with new data.
- **The independence assumption:** the method multiplies the chances of the clues as if each one told you something new. If two clues always come together (such as "free" and "FREE!!!"), counting both double-counts the evidence (equation 3 below).
- **Probabilities are often too extreme:** because of the double-counting, the percentages it prints tend to be pushed toward 0% and 100%. The *ranking* of the labels is often still right, so it can classify well while its percentages are untrustworthy (commonly reported; the toy example below shows the arithmetic).
- **Zero counts:** a word never seen with a label would give a chance of 0, wiping out the whole product. Smoothing (adding a small count, here 1) avoids this.
- **A generative classifier:** naive Bayes models how the data arises from each label, $P(x\mid y)$, then flips it with Bayes' rule. [[Logistic Regression]] models $P(y\mid x)$ directly (a discriminative classifier). A well-known study (Ng and Jordan, 2002, from memory) argued that the generative kind can learn faster from few examples, while the discriminative kind usually ends up more accurate with plenty of data. The digits test below shows the second half of that and does not show the first (details below). See also [[Generative Models]].
- **Where it is still used:** a quick baseline for [[Classification]] of text (spam filters, topic labels), and a sanity check before building something heavier (from memory).
- **Costs and limits:** it cannot learn that "not" changes the meaning of the next word, or any other interaction between clues. It is also sensitive to how clues are counted and smoothed.

![[naive-bayes-learning-curves.png]]
*A learning curve for naive Bayes, from the scikit-learn documentation (BSD licence). The red line is the score on the training data and the green line is the score on held-out data, for more and more training examples. The two lines approach each other, and the green line flattens around 0.84. A gap that closes but stays low suggests the model is too simple to do better (high bias), not that it needs more data. The file page does not name the dataset (I believe it is scikit-learn's handwritten-digits example, from memory), so only the shape is the lesson. Image: scikit-learn developers, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Learning_Curves_%28Naive_Bayes%29.png), BSD licence.*

The code does five things: (1) a tiny text classifier written by hand and checked against scikit-learn; (2) what happens with no smoothing; (3) why logs are needed; (4) double-counting a word; (5) Gaussian naive Bayes against logistic regression on the handwritten-digits set in scikit-learn, for several training sizes.

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
# 1) tiny text classifier, by hand and with scikit-learn
docs = ["win money now", "free money win", "win free prize now", "meeting at noon", "lunch at noon now", "project meeting notes"]
y = np.array([1, 1, 1, 0, 0, 0])                                  # 1 = spam, 0 = normal
cv = CountVectorizer(); X = cv.fit_transform(docs).toarray(); vocab = cv.get_feature_names_out(); V = len(vocab)
def fit(X, y, alpha=1.0):
    prior = np.array([(y == c).mean() for c in (0, 1)])
    cnt = np.array([X[y == c].sum(0) for c in (0, 1)])
    like = (cnt + alpha) / (cnt.sum(1, keepdims=True) + alpha * V)          # P(word | class), smoothed
    return np.log(prior), np.log(like)
lp, ll = fit(X, y)
def score(text):
    x = cv.transform([text]).toarray()[0]
    s = lp + ll @ x                                                  # log prior + sum of log likelihoods
    p = np.exp(s - s.max()); return (p / p.sum()).round(4)
for t in ("free money now", "meeting now", "free lunch"): print(t, '-> P(normal), P(spam) =', score(t))
sk = MultinomialNB(alpha=1.0).fit(X, y)
print('sklearn agrees:', np.allclose(sk.predict_proba(cv.transform(["free money now"]).toarray())[0], score("free money now"), atol=1e-4))
print('P(free|spam), P(free|normal):', np.exp(ll[:, list(vocab).index('free')]).round(4)[::-1].tolist())
# 2) without smoothing, one never-seen pairing zeroes everything
cnt = np.array([X[y == c].sum(0) for c in (0, 1)]); print('P(free|normal) with no smoothing:', cnt[0, list(vocab).index('free')] / cnt[0].sum())
# 3) underflow: multiplying 500 small probabilities
print('product of 500 x 1e-3:', np.prod(np.full(500, 1e-3)), '| sum of logs:', round(float(np.log(1e-3) * 500), 1))
# 4) double counting: the same word seen k times
def post(prior, lr, k): o = prior / (1 - prior) * lr ** k; return o / (1 + o)
print('copies of "free" k=1,2,3:', [round(post(0.3, 12, k), 4) for k in (1, 2, 3)])
# 5) generative (Gaussian NB) vs discriminative (logistic regression) on digits, by training-set size
Xd, yd = load_digits(return_X_y=True); Xd = Xd / 16.0
rng = np.random.default_rng(0)
for n in (20, 50, 200, 1000):
    a = {'nb': [], 'lr': []}
    for rep in range(10):
        idx = rng.permutation(len(yd)); tr, te = idx[:n], idx[-500:]          # train on n, test on 500 held-out digits
        a['nb'].append(GaussianNB(var_smoothing=1e-1).fit(Xd[tr], yd[tr]).score(Xd[te], yd[te]))
        a['lr'].append(LogisticRegression(max_iter=2000).fit(Xd[tr], yd[tr]).score(Xd[te], yd[te]))
    print('train size', n, '| Gaussian NB', round(float(np.mean(a['nb'])), 3), '| logistic regression', round(float(np.mean(a['lr'])), 3))
print('default smoothing (1e-9), 1000 train:', round(GaussianNB().fit(Xd[idx[:1000]], yd[idx[:1000]]).score(Xd[te], yd[te]), 3))
# free money now -> P(normal), P(spam) = [0.069 0.931]
# meeting now -> P(normal), P(spam) = [0.6667 0.3333]
# free lunch -> P(normal), P(spam) = [0.4 0.6]
# sklearn agrees: True
# P(free|spam), P(free|normal): [0.1429, 0.0476]
# P(free|normal) with no smoothing: 0.0
# product of 500 x 1e-3: 0.0 | sum of logs: -3453.9
# copies of "free" k=1,2,3: [0.8372, 0.9841, 0.9987]
# train size 20 | Gaussian NB 0.553 | logistic regression 0.577
# train size 50 | Gaussian NB 0.785 | logistic regression 0.803
# train size 200 | Gaussian NB 0.901 | logistic regression 0.917
# train size 1000 | Gaussian NB 0.913 | logistic regression 0.96
# default smoothing (1e-9), 1000 train: 0.85
```
Reading the output (scikit-learn 1.1.2, NumPy; seeds fixed, so it should repeat):
- The hand-written classifier gives the same answer as scikit-learn's `MultinomialNB`. "free money now" comes out 93.1% spam, "meeting now" is 66.7% normal, and "free lunch" is a mild 60% spam, a mixed message that splits the evidence.
- Without smoothing, "free" would have a chance of exactly 0 under "normal", so any message with "free" would be called spam with 100% certainty, however much else it said.
- Multiplying 500 chances of 0.001 gives exactly 0 in the computer, while the sum of their logs is a fine number, −3453.9.
- Seeing "free" 1, 2 or 3 times pushes the chance of spam from 0.837 to 0.984 to 0.999, though no new information arrived.
- On handwritten digits, logistic regression beats Gaussian naive Bayes at every training size I tried, by a small margin at 20 to 200 examples (0.577 against 0.553 at 20, 0.917 against 0.901 at 200) and a larger one at 1,000 (0.960 against 0.913). So this test does not show naive Bayes winning with few examples. It does show the gap growing with more data. That is only one dataset, with the settings I chose.
- Gaussian naive Bayes needed a hand-picked variance floor (`var_smoothing=0.1`) because many pixels are always 0 in some classes. With the default floor (1e-9) accuracy at 1,000 examples was 0.85 instead of 0.913. I picked 0.1 by looking, so the 0.913 is a bit optimistic for naive Bayes.

## 🔴 4. Deep / formal

**The decision rule, in logs.** For a label $y$ and clues $x_1,\dots,x_m$,

$$\hat y=\arg\max_{y}\Big[\log P(y)+\sum_{j=1}^{m}\log P(x_j\mid y)\Big]$$

**Where:**
- $P(y)$ is the prior, how common label $y$ is in the training data. **Matters:** it is the starting belief. If one label is very rare, its clues must be strong to win.
- $P(x_j\mid y)$ is how likely clue $j$ is under label $y$, learned by counting (or by fitting a bell curve). **Matters:** it is the vote of clue $j$.
- $\sum$ adds the votes of all clues. **Matters:** this is the "naive" step. Adding logs is the same as multiplying chances, which is only right if the clues are independent given the label.
- $\log$ turns products into sums. **Matters:** without it, the product of many small chances rounds to 0 in the computer and every label ties at 0 (the code shows 500 factors of 0.001 giving 0.0).
- $\arg\max$ means "the label with the largest value". **Matters:** the rule only needs the biggest, so the shared divider in Bayes' theorem (the evidence) can be left out for classifying. It matters only if you want percentages.

**Example (tiny numbers):** Training messages: 3 spam and 3 normal, 11 distinct words, smoothing 1. Classify "free money now". The prior is $0.5$ for each label. Under spam, each of the three words has chance $3/21$. Under normal, "free" and "money" have chance $1/21$ and "now" has $2/21$.
1. Spam: $\ln0.5+3\ln(3/21)=-0.6931-5.8377=-6.5309$.
2. Normal: $\ln0.5+2\ln(1/21)+\ln(2/21)=-0.6931-6.0890-2.3514=-9.1336$.
3. Spam wins by $2.6027$ in log score.
4. Percentages: $1/(1+e^{-2.6027})=0.931$, matching the code and scikit-learn.
5. Meaning: the word counts make spam $e^{2.6}\approx13.5$ times more likely than normal, so 93.1% spam. Most of the push came from "free" and "money", which only appear in spam in the examples.

**The Gaussian flavour.** For a number-valued clue, the method fits a bell curve per label and per clue:

$$P(x_j\mid y)=\frac{1}{\sqrt{2\pi\sigma_{jy}^2}}\exp\!\left(-\frac{(x_j-\mu_{jy})^2}{2\sigma_{jy}^2}\right)$$

**Where:**
- $\mu_{jy}$ is the average of clue $j$ among training examples with label $y$. **Matters:** it is where that label's bell curve is centred.
- $\sigma_{jy}$ is the spread of clue $j$ within label $y$. **Matters:** a narrow spread makes the label very picky about the value. If the spread is near 0 (a pixel that is always the same), any new value gives a chance near 0 or enormous, which is why a floor on the variance is added (the code needed one).
- $x_j$ is the clue's value for the new example. **Matters:** the closer to $\mu_{jy}$ (measured in spreads), the higher the chance.
- The formula is the [[Normal Distribution]]'s density. **Matters:** it is a density, not a probability. Only comparing densities between labels (as the rule does) makes sense.
- Every clue gets its own pair $(\mu,\sigma)$ per label, with no tilt between clues. **Matters:** this is why the regions in the animation are never tilted.

**Example (tiny numbers):** One clue, height $x=168$. Label A: $\mu=160$, $\sigma=5$. Label B: $\mu=175$, $\sigma=8$. Priors equal.
1. A: $z=(168-160)/5=1.6$, density $=0.0222$.
2. B: $z=(168-175)/8=-0.875$, density $=0.0340$.
3. Posterior for A: $0.0222/(0.0222+0.0340)=0.395$.
4. Meaning: B is a bit more likely, because 168 is closer to B's centre in B's own units (0.875 spreads, against 1.6 for A). The result is 39.5% A and 60.5% B.

**The double-counting problem.** Suppose the same clue is counted $k$ times. In log-odds form (see [[Bayes' Theorem]]),

$$\ln\frac{P(\text{spam}\mid\cdot)}{P(\text{normal}\mid\cdot)}=\ln\frac{P(\text{spam})}{P(\text{normal})}+k\,\ln \mathrm{LR}$$

**Where:**
- $\mathrm{LR}$ is the likelihood ratio of the clue, how many times more likely it is under spam than under normal. **Matters:** it is the clue's strength.
- $k$ is how many times the clue is counted. **Matters:** the evidence grows with $k$ even though no new information arrived. This is exactly what happens when several of your features are copies or near-copies of each other.
- The first term on the right is the prior log-odds. **Matters:** it is fixed, so the more copies, the more the prior is drowned out.
- The result is in log-odds. **Matters:** to get a chance, use $1/(1+e^{-\text{log-odds}})$.

**Example (tiny numbers):** Prior spam chance $0.3$ (log-odds $-0.847$), $\mathrm{LR}=12$ ($\ln12=2.485$).
1. $k=1$: $-0.847+2.485=1.638$, chance $0.837$.
2. $k=2$: $-0.847+4.970=4.123$, chance $0.984$.
3. $k=3$: $-0.847+7.455=6.607$, chance $0.9987$.
4. Meaning: the same evidence counted three times looks like near-certainty. The label stays the same (spam), but the *confidence* is inflated. This is why naive Bayes can be a good classifier and a poor probability estimator at once.

**Other points (from memory, not checked here):**
- With a tied prior and only two labels, the rule above is a linear function of the (log) clues. Multinomial naive Bayes therefore draws a straight-line boundary, like [[Logistic Regression]]. They differ in *how* the line is chosen (counting against fitting the accuracy). I did not derive this here.
- Naive Bayes is related to the "responsibilities" in a [[Gaussian Mixture Model]], where the labels are hidden and the same Bayes' rule is applied with learned bell curves.
- When the independence assumption is badly broken, adding the correlated clues as a combined feature or using a model that can learn interactions usually helps.

**What came before, and how this improved on it:**
- **Before:** write down the full table of how likely every combination of features is for each class.
- **Its limit:** the table grows as $2^d$. With 20 yes/no features that is 1,048,575 numbers per class, and most combinations never appear in the data.
- **What changed:** assume the features are independent given the class ("naive"), so only $d$ small tables are needed.
- **How much better:** the code below counts 20 numbers instead of 1,048,575 for 20 binary features. On breast-cancer measurements, a Gaussian naive Bayes using 122 numbers gets a test accuracy of 0.924. The independence assumption is false for most real data, and the note above explains what that costs.

**Check it yourself:**

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
d = 20
print(f"binary features d={d}: a full table of P(features | class) needs {2**d - 1:,} numbers per class; naive Bayes needs {d}")
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
print("30 real-valued features: Gaussian naive Bayes test accuracy", round(GaussianNB().fit(Xtr, ytr).score(Xte, yte), 3), " using", 2 * 30 * 2 + 2, "numbers")
```

## Related
[[Bayes' Theorem]] · [[Bayesian Methods]] · [[Classification]] · [[Logistic Regression]] · [[Generative Models]] · [[Normal Distribution]] · [[Probability]] · [[Maximum Likelihood]] · [[Gaussian Mixture Model]] · [[Tokenization]] · [[Bias-Variance]] · [[Evaluation Metrics]]

## Sources
- Ng and Jordan (2002), "On Discriminative vs. Generative Classifiers: A Comparison of Logistic Regression and Naive Bayes" (cited from memory; I did not open it for this note).
- Manning, Raghavan and Schütze, *Introduction to Information Retrieval*, chapter 13 (naive Bayes text classification; from memory).
- scikit-learn documentation for `MultinomialNB` and `GaussianNB`.
- Figures: [Naive_Bayes_Classifier.gif](https://commons.wikimedia.org/wiki/File:Naive_Bayes_Classifier.gif) (Jacopo Bertolotti, CC0) and [Learning_Curves_(Naive_Bayes).png](https://commons.wikimedia.org/wiki/File:Learning_Curves_%28Naive_Bayes%29.png) (scikit-learn developers, BSD), both on Wikimedia Commons.
- The code and every Example number were run for this note (scikit-learn 1.1.2, NumPy). The six training sentences are invented for teaching.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/classic-models Naive Bayes:::A classifier where each clue (for example each word) nudges the guess towards one label or another. It assumes the clues are independent, which is "naive" but works well.

#flashcards/ml/formula #flashcards/topic/classic-models Write the naive Bayes decision rule and explain each symbol.
?
$$\hat y=\arg\max_{y}\Big[\log P(y)+\sum_{j=1}^{m}\log P(x_j\mid y)\Big]$$
- $P(y)$: the prior, how common label $y$ is in the training data. It is the starting belief.
- $P(x_j\mid y)$: how likely clue $j$ is under label $y$, learned by counting (or by fitting a bell curve). It is the vote of clue $j$.
- $\sum$: adds the votes of all $m$ clues (logs turn products into sums).
- $\arg\max_y$: pick the label with the biggest total.

#flashcards/ml/formula #flashcards/topic/classic-models Write the Gaussian naive Bayes clue likelihood and explain each symbol.
?
$$P(x_j\mid y)=\frac{1}{\sqrt{2\pi\sigma_{jy}^2}}\exp\!\left(-\frac{(x_j-\mu_{jy})^2}{2\sigma_{jy}^2}\right)$$
- $\mu_{jy}$: the average of clue $j$ among training examples with label $y$. It is where that label's bell curve is centred.
- $\sigma_{jy}$: the spread of clue $j$ within label $y$. A narrow spread makes the label picky about the value. A spread near 0 (a pixel that is always the same) makes any new value look impossible.
- $x_j$: the value of clue $j$ for the new item.

#flashcards/ml/formula #flashcards/topic/classic-models Write what happens when the same clue is counted $k$ times, and explain each symbol.
?
$$\ln\frac{P(\text{spam}\mid\cdot)}{P(\text{normal}\mid\cdot)}=\ln\frac{P(\text{spam})}{P(\text{normal})}+k\,\ln \mathrm{LR}$$
- $\mathrm{LR}$: the likelihood ratio of the clue, how many times more likely it is under spam than under normal. It is the clue's strength.
- $k$: how many times the clue is counted. The evidence grows with $k$ even though no new information arrived.
- This is what happens when several features are copies of the same signal, and it makes naive Bayes overconfident.

#flashcards/ml/cloze #flashcards/topic/classic-models Naive Bayes assumes the clues are ==independent== given the label, so repeated clues are counted ==twice==.

#flashcards/ml/example #flashcards/topic/classic-models 3 spam and 3 normal messages, 11 distinct words, smoothing 1. For "free money now", spam has chance 3/21 per word. Normal has 1/21, 1/21 and 2/21. What are the log scores?::Spam: ln 0.5 + 3 ln(3/21) = −6.531. Normal: ln 0.5 + 2 ln(1/21) + ln(2/21) = −9.134. Spam wins by 2.60 in log terms (about 93%).

#flashcards/ml/example #flashcards/topic/classic-models Height 168. Label A: μ = 160, σ = 5. Label B: μ = 175, σ = 8. Equal priors. What is the posterior for A?::A: density 0.0222 (z = 1.6). B: density 0.0340 (z = −0.875). Posterior for A: 0.0222 / (0.0222 + 0.0340) = 0.395. B is a bit more likely.

#flashcards/ml/example #flashcards/topic/classic-models Prior spam 0.3, $\mathrm{LR}=12$. What is the spam chance when the clue is counted 1, 2 and 3 times?::Prior log-odds −0.847, ln 12 = 2.485. k = 1: 1.638, chance 0.837. k = 2: 4.123, chance 0.984. k = 3: 6.607, chance 0.9987. The same evidence counted three times looks like near-certainty.

#flashcards/ml/image #flashcards/topic/classic-models ![[naive-bayes-classifier.gif]]
What does this picture show?
?
A naive Bayes classifier on two features with three classes (A, B, C). As more labelled points arrive, the classifier fits a bell-shaped region to each class. Each region is stretched only along the two feature axes, never tilted, because the method treats the two features as independent.

#flashcards/ml/image #flashcards/topic/classic-models ![[naive-bayes-learning-curves.png]]
What does this picture show?
?
A learning curve for naive Bayes (scikit-learn documentation). The red line is the score on the training data, the green line the score on held-out data, for more and more training examples. The two lines approach each other and the green flattens around 0.84. A gap that closes but stays low suggests the model is too simple (high bias), not that it needs more data.
