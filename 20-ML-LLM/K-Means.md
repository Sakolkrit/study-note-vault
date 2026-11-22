---
tags: [machine-learning, unsupervised-learning, clustering, k-means]
status: growing
level: 4
tier: 1
---

# K-Means

## 🟢 1. Simple (age 12)
K-means is **a way to sort a big pile of things into a few groups of similar things, without anyone telling the computer what the groups are**. You say how many groups you want, and the computer finds them.

**Example:** You pour a bag of mixed buttons on the floor and want three piles. Put three sticks on the floor as "pile centres". Each button goes to its nearest stick. Then move each stick to the middle of its pile. Now some buttons are closer to a different stick, so move them. Repeat until nothing changes.

## 🔵 2. How it works
1. Decide how many groups you want. Call that number $K$.
2. Pick $K$ of your data points at random and call them the **centres** (also called centroids).
3. **Assign:** give every point to the group of its nearest centre.
4. **Move:** move each centre to the average position of the points now in its group.
5. Repeat steps 3 and 4. When no point changes group, stop.
6. The result depends on where the centres started. Run the whole thing several times from different random starts and keep the best result.
7. Nothing here uses labels, so this is [[Unsupervised Learning]]. It is one kind of [[Clustering]].

## 🟠 3. Technical detail
- **Distance:** usually the straight-line (Euclidean) distance. Because it depends on the size of each feature, [[Feature Scaling]] matters here too.
- **It never gets worse:** the assign step cannot increase the average squared distance to the centres, and neither can the move step (the average is the point that minimises squared distance). So the score only goes down or stays, and the process must stop.
- **But it may stop in a bad place:** it finds a local, not the global, best grouping. Fix: run it many times from different random starts (the notes loop 100 times and recommend this strongly for $K<10$) and keep the lowest score. A smarter start, called k-means++, is common in libraries (from memory).
- **Choosing $K$:** plot the score against $K$ and look for an "elbow" where the drop flattens, or choose the $K$ that is most useful for what you do with the groups next. The elbow is often not clear.
- **Empty group:** if a centre ends up with no points, re-pick it at random or drop it.
- **Where it fails:** it finds round, similar-sized blobs. Curved shapes, groups with very different sizes or densities, and outliers cause trouble. See [[DBSCAN]], [[Hierarchical Clustering]] and [[Gaussian Mixture Model]] for alternatives.
- **Uses:** grouping customers, compressing colours in an image, giving a first look at data.

```python
import numpy as np
P = np.array([[1.], [2.], [9.], [10.]]); mu = np.array([[1.], [10.]])   # four points, two centres
d = (P - mu.T) ** 2; c = d.argmin(1)                                     # assign step
print(c, "score", d.min(1).mean())                                       # [0 0 1 1] score 0.5
mu = np.array([[P[c == k].mean()] for k in (0, 1)])                      # move step
print(mu.ravel(), "score", ((P - mu.T) ** 2).min(1).mean())              # [1.5 9.5] score 0.25
```

## 🔴 4. Deep / formal
**What k-means is minimising (the distortion).** For points $x^{(1)},\dots,x^{(m)}$, group labels $c^{(i)}$ and centres $\mu_1,\dots,\mu_K$:

$$
J=\frac{1}{m}\sum_{i=1}^{m}\left\lVert x^{(i)}-\mu_{c^{(i)}}\right\rVert^{2}
$$

**Where:**
- $m$ is the number of points. $x^{(i)}$ is the $i$-th point (a list of numbers, one per feature).
- $c^{(i)}$ is the number of the group that point $i$ currently belongs to, and $\mu_{c^{(i)}}$ is that group's centre.
- $\lVert\cdot\rVert^2$ is the squared straight-line distance between the point and its centre.
- **Matters:** $J$ is the average squared distance from points to their own centre, so small $J$ means tight groups. The assign step lowers $J$ by changing the $c^{(i)}$ with the centres fixed; the move step lowers $J$ by changing the $\mu_k$ with the groups fixed. Squaring avoids a square root and makes far-away points count more.

**Example (tiny numbers):** points $1,\,2,\,9,\,10$ and starting centres $1$ and $10$.
- Assign: $1\to$ centre 1 (distance 0), $2\to$ centre 1 (distance 1), $9\to$ centre 10 (distance 1), $10\to$ centre 10 (distance 0).
- $J=(0^2+1^2+1^2+0^2)/4=\mathbf{0.5}$.
- Move: centre of group 1 becomes $(1+2)/2=1.5$, centre of group 2 becomes $(9+10)/2=9.5$. Now each point is 0.5 from its centre, so $J=(4\cdot0.25)/4=\mathbf{0.25}$.
- Meaning: one round halved the score, and the groups $\{1,2\}$ and $\{9,10\}$ are the obvious ones.

**Limits:**
- Minimising $J$ exactly is a hard problem (NP-hard in general; from memory). The alternating steps give only a local answer.
- $J$ always falls as $K$ grows (at $K=m$ it is 0), so it cannot choose $K$ by itself.
- A cluster means "points close to a centre", whether or not the data really has groups. The notes say k-means will happily split structureless data into $K$ pieces.

**What came before, and how this improved on it:**
- **Before:** group by hand or by fixed rules, or just split the data arbitrarily.
- **Its limit:** nothing checks that the groups are tight, and nobody can do it by hand for thousands of points in many dimensions.
- **What changed:** a simple loop (assign to the nearest centre, move the centre) that provably lowers a clear score each round. The idea is usually credited to Lloyd (1957, published 1982) and MacQueen (1967, who named it); these dates are from memory and not checked.
- **How much better:** in the code below (300 points in three blobs), a random split into three groups scores $J=12.697$; k-means scores $1.264$, which is 10.0 times lower. Starting from one random start, 10% of 200 runs ended stuck at a worse answer (the worst was $J=7.063$, 5.6 times the best); the best of 10 starts reached $1.264$ every time in 20 tries.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
centres = np.array([[0, 0], [6, 0], [3, 5.]])
X = np.vstack([c + rng.normal(0, 0.8, (100, 2)) for c in centres])
def kmeans(X, K, seed, iters=50):
    r = np.random.default_rng(seed); mu = X[r.choice(len(X), K, replace=False)].copy(); hist = []
    for _ in range(iters):
        d = ((X[:, None, :] - mu[None]) ** 2).sum(2); c = d.argmin(1)
        hist.append(d[np.arange(len(X)), c].mean())
        new = np.array([X[c == k].mean(0) if (c == k).any() else mu[k] for k in range(K)])
        if np.allclose(new, mu): break
        mu = new
    d = ((X[:, None, :] - mu[None]) ** 2).sum(2); return d.min(1).mean(), hist
J, h = kmeans(X, 3, 1)
print("never increases:", all(b <= a + 1e-12 for a, b in zip(h, h[1:])), "J %.3f" % J)
runs = np.array([kmeans(X, 3, s)[0] for s in range(200)])
print("200 single starts: best %.3f worst %.3f, stuck (>5%% above best): %.2f" % (runs.min(), runs.max(), (runs > runs.min() * 1.05).mean()))
rnd = np.random.default_rng(5).integers(0, 3, len(X))
Jr = sum(((X[rnd == k] - X[rnd == k].mean(0)) ** 2).sum() for k in range(3)) / len(X)
print("random split J %.3f -> %.1fx worse than k-means" % (Jr, Jr / runs.min()))
print("elbow:", [(K, round(min(kmeans(X, K, s)[0] for s in range(10)), 2)) for K in range(1, 8)])
```

## Related
- [[Clustering]] · [[Unsupervised Learning]] · [[DBSCAN]] · [[Hierarchical Clustering]] · [[Gaussian Mixture Model]] · [[Expectation-Maximization]] · [[PCA]] · [[Feature Scaling]] · [[Gradient Descent]] · [[NumPy]] · [[Machine Learning Specialization Course 3 Week 1]]

## Sources
- Repo notes `3. Unsupervised/Week 1/notes.pdf` (an older course's clustering notes: the algorithm, the distortion cost, random starts, the elbow method); see [[Machine Learning Specialization]]
- Lloyd and MacQueen attributions from memory, not checked
- The 12.697, 1.264, 7.063 and 10% figures are measured in this note's code

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/clustering K-means:::Sort points into $k$ groups of similar things without anyone telling the computer what the groups are. Put $k$ centres down, send each point to its nearest centre, move each centre to the middle of its points, and repeat.

#flashcards/ml/formula #flashcards/topic/clustering Write the k-means cost and explain each symbol.
?
$$J=\frac{1}{m}\sum_{i=1}^{m}\left\lVert x^{(i)}-\mu_{c^{(i)}}\right\rVert^{2}$$
- $m$: the number of points. $x^{(i)}$: the $i$-th point (one number per feature).
- $c^{(i)}$: the group point $i$ currently belongs to. $\mu_{c^{(i)}}$: that group's centre.
- $\lVert\cdot\rVert^2$: squared straight-line distance from the point to its centre.
- $J$ is the average squared distance to own centre, so small $J$ means tight groups. The assign step lowers $J$ by changing the $c^{(i)}$ with centres fixed. The move step lowers $J$ by changing the $\mu_k$ with groups fixed.

#flashcards/ml/example #flashcards/topic/clustering Points 1, 2, 9, 10 with start centres 1 and 10. What is $J$, then after one move?::Assign: 1 and 2 go to centre 1 (distances 0, 1), 9 and 10 to centre 10 (1, 0). $J=(0+1+1+0)/4=0.5$. Move: centres become 1.5 and 9.5. Each point is 0.5 away, $J=(4\cdot0.25)/4=0.25$. One round halved the score.

#flashcards/ml/cloze #flashcards/topic/clustering K-means has two repeated steps: ==assign== each point to its nearest centre, then ==move== each centre to the mean of its points.

#flashcards/ml/concept #flashcards/topic/clustering How does the user control k-means?::By choosing how many groups $k$ they want. The computer finds the groups.
