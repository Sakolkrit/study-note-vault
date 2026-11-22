---
tags: [fisher-information, statistics, probability, math, optimization, continual-learning]
status: growing
level: 4
tier: 3
---

# Fisher Information

A number that says **how sensitive a model's predictions are to one of its settings (a weight)**. If nudging a weight changes the model's predicted probabilities a lot, that weight has a big Fisher information. If the predictions barely change, it has a small one.

## 🟢 1. Simple (age 12)
Imagine a radio with many knobs. For some knobs, a tiny turn changes the sound a lot. For others, you can turn them quite far and hardly hear a difference. Fisher information is a score for each knob: **how loud the change is when you turn it a little**.

**Example:** A coin that lands heads 50% of the time is very "turnable": if the coin changes a little (50% to 60%), the results change noticeably. A coin that lands heads 99.9% of the time is already almost certain, so a small turn changes very little of what you see.

## 🔵 2. How it works
1. A model with a setting $\theta$ (for example the chance of heads) gives a probability to every outcome. See [[Probability]].
2. Ask: if $\theta$ changes slightly, **how much does the model's picture of the world change?** The answer is measured with the [[KL Divergence]] (a score for how different two sets of probabilities are).
3. For small changes, that difference grows with the **square** of the change, multiplied by one fixed number. That number is the Fisher information.
4. A second way to see it: the model's log-probability for an outcome has a slope with respect to $\theta$ (called the **score**). Average the squared slope over the outcomes the model itself expects. That gives the same number.
5. A third way: it is also the **curvature** of the loss around the best setting. A sharply curved bowl means a small move in $\theta$ costs a lot ([[Eigenvalue]], [[Convex Optimization]]).
6. **In AI.** A network has millions of weights, so each weight gets its own Fisher number. A weight with a big number is one the model's predictions are touchy about. [[Catastrophic Forgetting]] methods use this to protect weights: moving a touchy weight is expensive, moving a calm one is cheap.
7. **A catch.** "Touchy" is not the same as "useful". The code in level 3 shows a model where the weight that does all the work has a *smaller* Fisher number than a useless one, because the model is already very sure about its answers.

## 🟠 3. Technical detail
- **Definition.** For a model $p(y\mid x;\theta)$, the Fisher information is the average **squared score**: the slope of $\log p$ with respect to the weight, squared, averaged over $y$ drawn **from the model itself** (and over inputs $x$ from the data).
- **Not the same as the loss gradient.** The usual training gradient uses the *true* labels. The Fisher uses labels *sampled from the model*. At a good fit the two are close, but they are not identical. The code below samples from the model.
- **Mean score is 0.** Averaged over the model's own outcomes, the score has mean zero, so "average squared score" is its variance. The code checks the mean (about $-0.0005$ and $0.0008$).
- **Three ways to get the same number** (for logistic regression, checked in the code): average squared score, the curvature of the loss (second derivative), and the curvature of the KL divergence.
- **Matrix and diagonal.** With $P$ weights the full Fisher is a $P\times P$ [[Matrix]], with $P^2$ entries. Cheap methods keep only the diagonal ($P$ entries), ignoring how weights affect each other.
- **Positive and cheap.** Kirkpatrick et al. (2017, body read) list three properties (credited there to Pascanu and Bengio, 2013): it equals the second derivative of the loss near a minimum, it needs only first derivatives, and it is positive semi-definite (no negative curvature).
- **In machine learning.** Elastic Weight Consolidation uses the diagonal Fisher as the weight of a penalty (the equation in [[Catastrophic Forgetting]]). It is also behind "natural gradient" methods (not covered here).
- **Related.** [[Maximum Likelihood]] fits $\theta$; Fisher information says how sharply the fit is pinned down (its role in the accuracy of the fit is standard, **from memory**).
- **The code below is a toy.** A two-weight logistic model on 20,000 random points where only feature 1 matters. It checks the mean score, three ways of getting the same number, the KL meaning, the effect of rescaling a feature, and an EWC-style penalty.

```python
import numpy as np

rng = np.random.RandomState(0)
n = 20000
X = rng.randn(n, 2)
y = (X[:, 0] + 0.5 * rng.randn(n) > 0).astype(float)   # label depends on feature 1; feature 2 is noise


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def nll(w):
    p = sigmoid(X @ w)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


w = np.zeros(2)
for _ in range(2000):                           # fit by gradient descent
    w -= 0.5 * X.T @ (sigmoid(X @ w) - y) / n
print("fitted weights:", w.round(3))

p = sigmoid(X @ w)
# score = slope of log p(y | x, w) with respect to w; here y is drawn from the MODEL, not from the data
ys = (rng.rand(n) < p).astype(float)
score = (ys - p)[:, None] * X                   # one row per example
print("mean of the score (should be about 0):", score.mean(0).round(4))
F_sample = (score ** 2).mean(0)                 # diagonal Fisher = average squared score
F_exact = (p * (1 - p))[:, None] * X ** 2       # exact expectation over y for logistic regression
print("diagonal Fisher from samples:", F_sample.round(4), "| exact:", F_exact.mean(0).round(4))

eps = 1e-3                                      # curvature of the loss: second slope by finite differences
curv = []
for i in range(2):
    d = np.zeros(2); d[i] = eps
    curv.append((nll(w + d) - 2 * nll(w) + nll(w - d)) / eps ** 2)
print("curvature of the loss (second derivative):", np.round(curv, 4))


# meaning: F is the curvature of the KL divergence, so KL(model at w || model at w + d) is about 0.5 * d^2 * F
def avg_kl(w_a, w_b):
    pa, pb = sigmoid(X @ w_a), sigmoid(X @ w_b)
    return float(np.mean(pa * np.log(pa / pb) + (1 - pa) * np.log((1 - pa) / (1 - pb))))


for i in range(2):
    d = np.zeros(2); d[i] = 0.1
    print(f"weight {i + 1} moved by 0.1: average KL {avg_kl(w, w + d):.6f} | 0.5 * 0.1^2 * F = {0.5 * 0.01 * F_exact.mean(0)[i]:.6f}")

# the Fisher depends on how sensitive the model's output is, so a rescaled feature gets a bigger Fisher
Xs = X * np.array([0.1, 3.0])
w2 = w / np.array([0.1, 3.0])                   # the same model, written for the rescaled features
p2 = sigmoid(Xs @ w2)
print("same model, features rescaled by (0.1, 3):", ((p2 * (1 - p2))[:, None] * Xs ** 2).mean(0).round(4))

# the EWC-style penalty from the Catastrophic Forgetting note: lambda/2 * F_i * (theta_i - theta_star_i)^2
lam = 1.0
move = 0.5
pen = lam / 2 * F_exact.mean(0) * move ** 2
print("penalty for moving each weight by 0.5:", pen.round(4), "| ratio", round(float(pen.max() / pen.min()), 1))
```

What it printed: the fitted weights are (3.532, 0.013), so feature 1 does the work. The mean score is about 0. The diagonal Fisher from samples is (0.019, 0.1062) and the exact value is (0.0193, 0.102), and the curvature of the loss is (0.0193, 0.102), the same exact numbers. Moving weight 1 by 0.1 gave an average KL of 0.000094 against 0.000096 predicted, and weight 2 gave 0.000509 against 0.000510. With the features rescaled by (0.1, 3) the Fisher became (0.0002, 0.9177). The penalty for moving each weight by 0.5 was (0.0024, 0.0127), a ratio of 5.3.

**The surprising part:** the weight that does all the work (3.5) has the **smaller** Fisher (0.019), and the noise weight (0.013) has the bigger one (0.10). The model is very sure of its answers (weight 3.5 saturates the sigmoid), so changing that weight moves few predictions, while changing the noise weight adds a random push to every logit, and it changes many predictions near the boundary. So a Fisher penalty protects "predictions do not change", which is not the same as "this weight is the one that matters for the task".

## 🔴 4. Deep / formal
For a model with one parameter $\theta$ giving probabilities $p(y;\theta)$ to outcomes $y$:

$$
F(\theta)=\mathbb{E}_{y\sim p(\cdot;\theta)}\!\left[\left(\frac{\partial}{\partial\theta}\log p(y;\theta)\right)^{2}\right]
$$

**Where:**
- $\theta$ is the parameter (one weight). $p(y;\theta)$ is the model's probability of outcome $y$.
- $\frac{\partial}{\partial\theta}\log p(y;\theta)$ is the **score**: how fast the log-probability of $y$ rises or falls when $\theta$ rises.
- $\mathbb{E}_{y\sim p}$ means the average over outcomes $y$ drawn from the model itself, each counted by its own probability.
- **Matters:** the squaring and averaging turn a slope (which averages to 0) into a size. A big $F$ means a small change in $\theta$ changes the model's probabilities a lot. With several weights, the same recipe gives a matrix, and its diagonal gives one $F_i$ per weight.

**Example (tiny numbers):** a coin model with $p(\text{heads})=\theta$, so $p(\text{tails})=1-\theta$.
- Score for heads: $\frac{\partial}{\partial\theta}\log\theta=\frac1\theta$. Score for tails: $\frac{\partial}{\partial\theta}\log(1-\theta)=-\frac1{1-\theta}$.
- $F=\theta\cdot\frac1{\theta^2}+(1-\theta)\cdot\frac1{(1-\theta)^2}=\frac1\theta+\frac1{1-\theta}=\frac1{\theta(1-\theta)}$.
- At $\theta=0.5$: $F=\mathbf{4}$. At $\theta=0.9$: $F=1/(0.9\times0.1)=\mathbf{11.1}$.
- Meaning: the nearly sure coin has the bigger $F$ for the **chance itself**, because a small move of $\theta$ near 0 or 1 changes a rare outcome's probability by a large *ratio*. This is a different kind of sensitivity from the logistic-weight case in the code. It shows $F$ depends on the parametrisation (the model's setting).

The link with the KL divergence, for a small change $\Delta$ in the parameter:

$$
D_{\mathrm{KL}}\bigl(p(\cdot;\theta)\,\big\|\,p(\cdot;\theta+\Delta)\bigr)\approx\tfrac12\,F(\theta)\,\Delta^{2}
$$

**Where:**
- $D_{\mathrm{KL}}$ is the [[KL Divergence]] between the model before and after the change.
- $\Delta$ is the size of the change in $\theta$. $F(\theta)$ is the number from above.
- **Matters:** this is the meaning of $F$: the price, in changed predictions, of moving the setting. A penalty $\tfrac\lambda2F_i(\theta_i-\theta^*_i)^2$ is therefore a bound on how much the model's predictions are allowed to change.

**Example (tiny numbers):** the coin at $\theta=0.5$ ($F=4$), moved to $0.6$ ($\Delta=0.1$).
- Approximation: $\tfrac12\times4\times0.1^2=\mathbf{0.02}$.
- Exact: $0.5\ln\frac{0.5}{0.6}+0.5\ln\frac{0.5}{0.4}=0.5(-0.1823)+0.5(0.2231)=\mathbf{0.0204}$.
- Meaning: the shortcut is close for a small move, so a small $\Delta$ costs about $0.02$ units of KL.

**Further notes (from the code above):**
- **Same number, three ways.** Average squared score from model samples (0.019, 0.1062), its exact value (0.0193, 0.102) and the loss curvature (0.0193, 0.102). For logistic regression the exact values coincide; it is not true for every model.
- **KL meaning checked.** The $\tfrac12F\Delta^2$ estimate was within about 2% of the measured KL for weight 1 (0.000096 against 0.000094), and within 0.2% for weight 2.
- **Size.** For a layer of 512 by 2048 weights (1,048,576 numbers), the full Fisher would have $1{,}048{,}576^2\approx1.1\times10^{12}$ entries; the diagonal has 1,048,576. (Arithmetic, not run.)
- **Caveat for protection methods.** The code shows a case where the Fisher is bigger for the useless weight. The penalty in the toy was 5.3 times bigger for the useless weight. How well Fisher-based penalties work on real networks was not tested here.
- **What the toy cannot show:** a neural network, the full matrix, or the forgetting effect itself.

**What came before, and how this improved on it:**
- **Before:** the idea is older than the name. Edgeworth worked on the same quantity around 1908–09, and Savage is quoted as saying Fisher was "to some extent anticipated" by him (Wikipedia, history section; read). Fisher's own work and its dates are **from memory**.
- **Its limit:** for big models the curvature matrix of the loss (second derivatives) is huge, and costs a lot to compute: $P^2$ entries for $P$ weights (arithmetic above).
- **What changed:** Fisher information needs only first derivatives (squared scores) and is always non-negative, and near a good fit it equals the curvature. Kirkpatrick et al. (arXiv 1612.00796, 2017; PDF text read) use its diagonal as the weights of the penalty in elastic weight consolidation, and say the diagonal is a good estimate of how important a parameter is (their own test, not repeated here).
- **How much better:** in this note's code, the average squared score reproduced the curvature of the loss to the 4 decimals printed (0.0193 and 0.102) using only first derivatives. For one 512-by-2048 layer the diagonal keeps 1,048,576 numbers against about $1.1\times10^{12}$ for the full matrix.

## Related
- [[KL Divergence]] · [[Probability]] · [[Maximum Likelihood]] · [[Entropy]] · [[Catastrophic Forgetting]] · [[Continual Learning]] · [[Logistic Regression]] · [[Matrix]] · [[Eigenvalue]] · [[Convex Optimization]] · [[Gradient Descent]] · [[Hessian]]

## Sources
- Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks*, arXiv 1612.00796 (2017): the PDF text was read for the sections on the penalty (equation 3), the diagonal Fisher and its three properties (credited there to Pascanu and Bengio 2013). The experiments were not read in detail.
- Wikipedia, "Fisher information" (read through the API): the introduction and history section.
- Fisher's original papers and dates, and the role of Fisher information in the accuracy of maximum-likelihood fits, are **from memory**.
- Image: none. Three Wikimedia Commons searches found only portraits of Ronald Fisher and unrelated files, and no picture of the concept, so none is used.

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/probability Fisher information:::A score for each parameter ("knob") of how much the model's predictions change when you turn it a little.

#flashcards/math/formula #flashcards/topic/probability Write Fisher information for one parameter and explain each symbol.
?
$$F(\theta)=\mathbb{E}_{y\sim p(\cdot;\theta)}\!\left[\left(\frac{\partial}{\partial\theta}\log p(y;\theta)\right)^{2}\right]$$
- $\theta$: the parameter (one weight). $p(y;\theta)$: the model's probability of outcome $y$.
- $\frac{\partial}{\partial\theta}\log p(y;\theta)$: the score, how fast the log-probability of $y$ rises when $\theta$ rises.
- $\mathbb{E}_{y\sim p}$: the average over outcomes drawn from the model itself, each counted by its own probability.
- Squaring and averaging turn a slope (which averages to 0) into a size. A big $F$ means a small change in $\theta$ changes the predictions a lot. With several weights the same recipe gives a matrix.

#flashcards/math/formula #flashcards/topic/probability Write how Fisher information relates to KL divergence for a small change, and explain each symbol.
?
$$D_{\mathrm{KL}}\bigl(p(\cdot;\theta)\,\big\|\,p(\cdot;\theta+\Delta)\bigr)\approx\tfrac12\,F(\theta)\,\Delta^{2}$$
- $D_{\mathrm{KL}}$: the KL divergence between the model before and after the change.
- $\Delta$: the size of the change in $\theta$. $F(\theta)$: the Fisher information.
- This is the meaning of $F$: the price, in changed predictions, of moving the setting. A penalty $\tfrac\lambda2F_i(\theta_i-\theta^*_i)^2$ therefore limits how much predictions may change.

#flashcards/math/cloze #flashcards/topic/probability A large Fisher information means a small change in the parameter changes the model's ==predictions== by a ==lot==.

#flashcards/math/example #flashcards/topic/probability For a coin model with $p(\text{heads})=\theta$, what is $F$, and what is it at $\theta=0.5$?::F = 1/θ + 1/(1−θ) = 1/(θ(1−θ)). At θ = 0.5 it is 4.

#flashcards/math/example #flashcards/topic/probability The coin moves from $\theta=0.5$ ($F=4$) to 0.6. What is the KL cost by the shortcut and exactly?::Shortcut: ½ × 4 × 0.1² = 0.02. Exact: 0.5 ln(0.5/0.6) + 0.5 ln(0.5/0.4) = 0.0204. Close for a small move.
