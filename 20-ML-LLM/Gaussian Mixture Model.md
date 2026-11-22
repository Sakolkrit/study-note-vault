---
tags: [machine-learning, unsupervised-learning, clustering, probability, gaussian, EM]
status: growing
level: 4
tier: 3
---

# Gaussian Mixture Model

A Gaussian mixture model (GMM) says your data came from **several overlapping bell curves mixed together**, and it works out how much each point belongs to each bell.

## 🟢 1. Simple (age 12)
Imagine a school where you measure everyone's height. You get one big lump of numbers. But really there are two crowds mixed up: the little kids (a bell around 130 cm) and the big kids (a bell around 165 cm). A GMM looks at the lump and figures out **the separate bells hiding inside it**. For a kid of 150 cm it can even say "60% likely a little kid, 40% likely a big kid", instead of forcing a single answer.

**Example:** A shop sees two kinds of customers mixed in one receipt pile: quick small buyers and slow big spenders. Each kind spends around its own typical amount, with some spread. A GMM splits the pile into the two kinds, and for a medium receipt says how likely it came from each kind.

## 🔵 2. How it works
A GMM is a [[Clustering]] method in [[Unsupervised Learning]]. It is like k-means (see [[Clustering]]), but **soft**: each point gets a share of belonging to every group, not just one label. Each group is a bell curve (a [[Normal Distribution]]) with its own centre and spread.
1. Choose how many bells, $K$.
2. Start with guesses for each bell: its centre, its spread, and how big a share of the data it covers.
3. **Guess step:** for each point, work out how likely it is that each bell produced it. These shares add up to 1 for the point. They are the **responsibilities**.
4. **Update step:** move each bell to fit the points it is responsible for. Centre = weighted average of the points, spread = weighted spread, share = how much total responsibility it got.
5. Repeat steps 3 and 4. The bells slowly settle on the data. This two-step loop is called **expectation–maximisation** ([[Expectation-Maximization]]).

![[gaussian-mixture-example.svg]]
*Three bell curves (dotted) added together make the lumpy solid curve. A Gaussian mixture model tries to go the other way: given the lumpy shape of the data, recover the separate bells. Image: Smason79, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Gaussian-mixture-example.svg), CC BY-SA 3.0.*

## 🟠 3. Technical detail
- **Hard vs soft:** k-means gives each point one label. A GMM gives probabilities, which is useful for borderline points and for noticing unusual ones (a point that every bell finds unlikely).
- **Each bell has its own shape:** in many dimensions a bell can be a stretched, tilted oval (a covariance matrix), so a GMM can fit oval groups that k-means (which likes round ones) cannot. Variants restrict this (round, same-shaped, etc.) to use fewer numbers.
- **Relation to k-means:** k-means is roughly the special case where all bells are equal round shapes and each point is assigned fully to its nearest centre (stated from memory; the standard textbook relationship).
- **Fitting:** EM raises the likelihood of the data (see [[Maximum Likelihood]]) a little each round. It always settles somewhere, but maybe on a **local** best, so run it from several starts. It can also fail if a bell shrinks onto one point (spread near zero), and libraries add a tiny floor to stop that.
- **Choosing $K$:** a bigger $K$ always fits better, so use a penalty such as BIC, or check on held-out data (see [[Overfitting]] and [[Evaluation Metrics]]).
- **Generative:** a GMM is a model of how data is made, so you can **sample new points** from it (see [[Generative Models]]). It is also a gentle example of a model with a hidden variable (which bell made the point).
- **Bayesian link:** the responsibility is Bayes' rule, with the share as the prior and the bell as the likelihood (see [[Bayesian Methods]]).

```python
import numpy as np
from sklearn.mixture import GaussianMixture
rng = np.random.default_rng(0)
data = np.r_[rng.normal(0, 1, 150), rng.normal(5, 1.5, 100)]     # truth: 150 points ~N(0,1), 100 points ~N(5,1.5)
pi_, mu, sd = np.array([.5, .5]), np.array([-1., 6.]), np.array([1., 1.])    # rough starting guesses
dens = lambda x, m, s: np.exp(-(x[:, None] - m) ** 2 / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi))
for _ in range(300):
    r = pi_ * dens(data, mu, sd); r /= r.sum(1, keepdims=True)    # E-step: responsibilities
    nk = r.sum(0); pi_ = nk / len(data)                            # M-step: shares,
    mu = (r * data[:, None]).sum(0) / nk                           #         centres,
    sd = np.sqrt((r * (data[:, None] - mu) ** 2).sum(0) / nk)      #         spreads
print(pi_.round(2), mu.round(2), sd.round(2))      # [0.6 0.4] [0.08 4.83] [0.99 1.66]
g = GaussianMixture(2, random_state=0, tol=1e-9, max_iter=2000).fit(data[:, None])
o = np.argsort(g.means_.ravel())                   # order the two bells left to right
print(g.weights_[o].round(2), g.means_.ravel()[o].round(2), np.sqrt(g.covariances_.ravel()[o]).round(2))
# [0.6 0.4] [0.08 4.83] [0.99 1.66]   the library agrees
```

The true values are shares $0.6$ and $0.4$, centres $0$ and $5$, spreads $1$ and $1.5$. The fit gets $0.6/0.4$, centres $0.08$ and $4.83$, spreads $0.99$ and $1.66$: close, but not exact, because only 250 points were seen. The hand-written loop and scikit-learn agree once the library's stopping tolerance is made tight. With its default (looser) tolerance it stopped slightly early, at centres 0.11 and 4.91 in my run. Needs scikit-learn.

## 🔴 4. Deep / formal
**The mixture density.** The chance of seeing a value $x$ is a weighted sum of $K$ bell curves:

$$
p(x)=\sum_{k=1}^{K}\pi_k\,\mathcal N\!\big(x\mid\mu_k,\sigma_k^2\big),\qquad
\mathcal N(x\mid\mu,\sigma^2)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

**Where:**
- $K$ is the number of bells. $\pi_k$ is the share of bell $k$ (between 0 and 1, all shares add to 1).
- $\mu_k$ is the centre of bell $k$ and $\sigma_k$ is its spread. $\mathcal N(x\mid\mu,\sigma^2)$ is the height of that bell at $x$.
- $e^{-(x-\mu)^2/(2\sigma^2)}$ is largest ($=1$) at the centre and drops quickly the further $x$ is from it. The fraction in front makes the area under the bell equal to 1.
- **Matters:** the shares $\pi_k$ are what let one bell be bigger than another. Without them every bell would count equally, and a group of 100 points would weigh the same as a group of 5.

**Example (tiny numbers):** two bells with $\pi=(0.5,0.5)$, centres $0$ and $4$, spread $1$ each, at $x=1$.
- Bell 1 at $x=1$: $\frac{1}{\sqrt{2\pi}}e^{-0.5}=0.2420$. Bell 2 at $x=1$: $\frac{1}{\sqrt{2\pi}}e^{-4.5}=0.00443$.
- Mix: $0.5\times0.2420+0.5\times0.00443=0.1210+0.00222=\mathbf{0.1232}$.
- Meaning: the model says the density at $x=1$ is 0.1232. Almost all of it comes from bell 1, since $x=1$ is next to centre $0$ and far from centre $4$.

**Responsibilities (the guess step).** How much of a point's density came from each bell:

$$
r_{ik}=\frac{\pi_k\,\mathcal N(x_i\mid\mu_k,\sigma_k^2)}{\sum_{j=1}^{K}\pi_j\,\mathcal N(x_i\mid\mu_j,\sigma_j^2)}
$$

**Where:**
- $r_{ik}$ is the responsibility of bell $k$ for point $i$: a number from 0 to 1. For each point, the $r_{ik}$ over all $k$ add up to 1.
- The top is bell $k$'s weighted height at the point $x_i$. The bottom adds the same thing for all bells, which is the mixture density $p(x_i)$.
- **Matters:** this is Bayes' rule (see [[Bayesian Methods]]): the share is the prior, the bell height is the likelihood, and $r_{ik}$ is the posterior. The division is what makes the answers sum to 1. Without it, they would be raw heights, not probabilities.

**Example (tiny numbers):** the same two bells and the same point $x=1$.
- Top for bell 1: $0.5\times0.2420=0.1210$. Top for bell 2: $0.5\times0.00443=0.00222$.
- Bottom: $0.1210+0.00222=0.1232$.
- $r_{1}=0.1210/0.1232=\mathbf{0.982}$ and $r_2=0.018$.
- Meaning: the point at $x=1$ is 98.2% likely to have come from bell 1. A point halfway between the centres, at $x=2$, would get 50/50.

**The update step.** Given the responsibilities, each bell moves to the weighted average of the points it owns (the centre update; the share and the spread are updated in the same spirit):

$$
\mu_k=\frac{\sum_{i}r_{ik}\,x_i}{\sum_{i}r_{ik}},\qquad \pi_k=\frac{1}{n}\sum_{i}r_{ik}
$$

**Where:**
- $x_i$ is point $i$ and $n$ is the number of points. $r_{ik}$ is the responsibility from the guess step.
- $\sum_i r_{ik}$ is the "effective number of points" that bell $k$ owns. It need not be a whole number.
- The top of $\mu_k$ adds each point weighted by how much it belongs to bell $k$. The bottom is the total weight.
- **Matters:** if every $r_{ik}$ were exactly 0 or 1, this would be the plain "average of my points" step of k-means (see [[Clustering]]). The soft weights let a borderline point pull on two bells at once.

**Example (tiny numbers):** three points $0,\,1,\,5$ with responsibilities $0.9,\,0.8,\,0.1$ for bell 1.
- Total weight: $0.9+0.8+0.1=1.8$.
- Weighted sum: $0.9\times0+0.8\times1+0.1\times5=0+0.8+0.5=1.3$.
- New centre: $\mu_1=1.3/1.8=\mathbf{0.722}$. New share: $\pi_1=1.8/3=\mathbf{0.6}$.
- Meaning: the centre moves to 0.722, close to the two points that mostly belong to it, with a small pull from the far point at 5.

**Further notes:**
- **Why it works:** each EM round cannot lower the likelihood of the data (a standard result; stated from memory, proof not shown here). It can still end on a local best.
- **Multi-dimensional bells:** replace $\sigma^2$ by a covariance matrix. The updates look the same, with matrices.
- **Singularity:** a bell centred on a single point with spread near 0 gives an enormous density, so real implementations add a small floor to the spread.
- **Density estimation:** a GMM is also a smooth density model, useful for spotting outliers (low $p(x)$).

**What came before, and how this improved on it:**
- **Before:** k-means, which assigns each point to one round cluster.
- **Its limit:** it assumes round clusters of equal size, and gives only hard answers.
- **What changed:** model each cluster as an oval (a Gaussian) with its own centre, shape and weight, and give each point a probability for each cluster, fitted with [[Expectation-Maximization]].
- **How much better:** in the code below, two long, flat clusters sit one above the other. k-means splits them left and right and scores an adjusted Rand index of -0.002 (no better than chance). A Gaussian mixture with 10 random starts scores 1.000. **Caution:** with the library's default start, which begins from the k-means answer, the mixture stayed stuck on the same wrong split (0.0) when I first ran it. The start matters.

**Check it yourself:**

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
rng = np.random.default_rng(0)
X = np.r_[rng.standard_normal((300, 2)) @ [[4, 0], [0, 0.3]] + [0, 0], rng.standard_normal((300, 2)) @ [[4, 0], [0, 0.3]] + [0, 2]]
lab = np.repeat([0, 1], 300)                                        # two long flat clusters, one above the other
print("k-means (round clusters): adjusted Rand", round(adjusted_rand_score(lab, KMeans(2, n_init=10, random_state=0).fit_predict(X)), 3))
print("Gaussian mixture (any oval, 10 random starts): adjusted Rand", round(adjusted_rand_score(lab, GaussianMixture(2, init_params="random_from_data", n_init=10, random_state=0).fit_predict(X)), 3))
```

## Related
- [[Clustering]] · [[Unsupervised Learning]] · [[Normal Distribution]] · [[Probability]] · [[Bayesian Methods]] · [[Maximum Likelihood]] · [[Expectation-Maximization]] · [[Generative Models]] · [[DBSCAN]] · [[Hierarchical Clustering]] · [[Overfitting]] · [[Evaluation Metrics]] · [[Mixture of Experts]] · [[Bayes' Theorem]] · [[Naive Bayes]] · [[K-Means]]
- [[Anomaly Detection]] · [[Anomaly Detection Shoot-out]]
## Sources
- Dempster, Laird & Rubin, *Maximum likelihood from incomplete data via the EM algorithm*, 1977 (from memory)
- Bishop, *Pattern Recognition and Machine Learning*, chapter 9 (from memory)
- Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 2011 (from memory); user guide, "Gaussian mixture models" (free online)
- Murphy, *Machine Learning: A Probabilistic Perspective*, chapter 11 (from memory)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/clustering Gaussian mixture model (GMM):::Several bell curves mixed together. It finds the separate bells hiding in a lumpy pile of data and gives soft answers such as "60% little kid, 40% big kid".

#flashcards/ml/formula #flashcards/topic/clustering Write the mixture density and the bell curve and explain each symbol.
?
$$p(x)=\sum_{k=1}^{K}\pi_k\,\mathcal N\!\big(x\mid\mu_k,\sigma_k^2\big),\qquad \mathcal N(x\mid\mu,\sigma^2)=\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
- $K$: the number of bells. $\pi_k$: the share of bell $k$ (0 to 1, all shares add to 1).
- $\mu_k$: the centre of bell $k$. $\sigma_k$: its spread. $\mathcal N(x\mid\mu,\sigma^2)$: the height of that bell at $x$.
- The exponential is 1 at the centre and drops fast away from it. The fraction in front makes the area under the bell equal to 1.
- The shares $\pi_k$ let one bell be bigger than another. Without them a group of 100 points would weigh the same as a group of 5.

#flashcards/ml/example #flashcards/topic/clustering Two bells, $\pi=(0.5,0.5)$, centres 0 and 4, spread 1, at $x=1$. What is $p(1)$?::Bell 1: $e^{-0.5}/\sqrt{2\pi}=0.2420$. Bell 2: $e^{-4.5}/\sqrt{2\pi}=0.00443$. Mix $0.5\times0.2420+0.5\times0.00443=0.1232$. Almost all of it comes from bell 1.

#flashcards/ml/formula #flashcards/topic/clustering Write the responsibility of bell $k$ for point $i$ and explain each symbol.
?
$$r_{ik}=\frac{\pi_k\,\mathcal N(x_i\mid\mu_k,\sigma_k^2)}{\sum_{j=1}^{K}\pi_j\,\mathcal N(x_i\mid\mu_j,\sigma_j^2)}$$
- $r_{ik}$: the responsibility of bell $k$ for point $i$, from 0 to 1. For each point the $r_{ik}$ over all $k$ add to 1.
- The top is bell $k$'s weighted height at $x_i$. The bottom adds the same for all bells, which is the mixture density $p(x_i)$.
- This is Bayes' rule: the share is the prior, the bell height the likelihood, $r_{ik}$ the posterior. The division makes the answers sum to 1. Without it they would be raw heights, not probabilities.

#flashcards/ml/example #flashcards/topic/clustering Same bells, $x=1$. What are the responsibilities?::Top for bell 1: $0.5\times0.2420=0.1210$. Top for bell 2: $0.00222$. Bottom $0.1232$. $r_1=0.982$, $r_2=0.018$. A point halfway between the centres ($x=2$) would get 50/50.

#flashcards/ml/formula #flashcards/topic/clustering Write the M-step updates for a bell's centre and share and explain each symbol.
?
$$\mu_k=\frac{\sum_{i}r_{ik}\,x_i}{\sum_{i}r_{ik}},\qquad \pi_k=\frac{1}{n}\sum_{i}r_{ik}$$
- $x_i$: point $i$. $n$: the number of points. $r_{ik}$: the responsibility from the guess step.
- $\sum_i r_{ik}$: the effective number of points bell $k$ owns. It need not be a whole number.
- The top of $\mu_k$ adds each point weighted by how much it belongs to bell $k$. The bottom is the total weight.
- If every $r_{ik}$ were exactly 0 or 1 this would be the plain average step of k-means. Soft weights let a borderline point pull on two bells at once.

#flashcards/ml/example #flashcards/topic/clustering Points 0, 1, 5 with responsibilities 0.9, 0.8, 0.1 for bell 1. What are the new $\mu_1$ and $\pi_1$?::Total weight $1.8$. Weighted sum $0+0.8+0.5=1.3$. $\mu_1=1.3/1.8=0.722$. $\pi_1=1.8/3=0.6$. The centre moves close to the two points that mostly belong to it, with a small pull from the point at 5.

#flashcards/ml/image #flashcards/topic/clustering ![[gaussian-mixture-example.svg]]
What does this picture show?
?
Three bell curves (dotted) added together make the lumpy solid curve. A GMM tries to go the other way: given the lumpy shape of the data, recover the separate bells. (From the note's caption: Smason79, Wikimedia Commons.)

#flashcards/ml/cloze #flashcards/topic/clustering A GMM gives ==soft== answers (probabilities for each bell) where k-means gives a single hard group.
