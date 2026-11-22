---
tags: [machine-learning, supervised-learning, classification, SVM, kernel, margin, optimization]
status: growing
level: 4
tier: 3
---

# Support Vector Machine

A Support Vector Machine is usually shortened to **SVM**.

## 🟢 1. Simple (age 12)
An SVM separates two groups by drawing the line that leaves **the widest empty street between them**. Many lines can split the groups, but the one with the widest street is the safest, because new items that land a little off are still on the right side.

**Example:** Two teams stand on a playground, one on the left and one on the right. You must lay a rope between them. You could put it right next to one team, but then a small shuffle would put someone on the wrong side. Putting the rope in the middle of the widest gap is safest. Only the kids standing right at the edge of each team decide where the rope goes. Everyone else could walk away and the rope would not move.

## 🔵 2. How it works
1. You have items with two labels, such as "yes" and "no", each described by a list of numbers. (See [[Classification]] and [[Vector]].)
2. A straight line (in 3 or more features, a flat sheet) can separate them. Find the one with the **biggest gap** to the closest items of both labels. The gap is called the **margin**.
3. The few items touching the edge of the margin are the **support vectors**. They are the only ones that hold the line in place.
4. A new item is labelled by **which side of the line it falls on**.
5. **If the groups overlap** and no line is perfect, allow some mistakes, but charge a price for each one. A dial called $C$ sets how expensive mistakes are.
6. **If no straight line works at all** (one group is a ring around the other), the **kernel trick** lets the SVM act as if the points had been lifted into more dimensions, where a flat sheet can split them.

![[svm-max-margin.svg]]
*Two groups (black and white dots). The solid line is the dividing line. The dashed lines mark the edges of the margin, the empty street. The dots sitting right on the dashed lines (the highlighted ones) are the support vectors. The arrow $\mathbf w$ points straight across the street. Image: Cyc, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Svm_max_sep_hyperplane_with_margin.svg), public domain.*

![[kernel-trick-idea.svg]]
*Left: two groups in 2-D that no straight line can separate (red inside, purple around). Right: the same points lifted into 3-D, where a flat sheet separates them. A kernel lets the SVM get that benefit without actually building the 3-D points. Image: Shiyu Ji, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Kernel_trick_idea.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Type:** [[Supervised Learning]] for [[Classification]]. Variants exist for [[Regression]] (support vector regression).
- **Max margin idea:** among all separating lines, choose the one farthest from the nearest points of each class. The intuition is a more robust boundary, which relates to [[Overfitting]] and [[Bias-Variance]].
- **Soft margin:** real data overlaps. Points inside the street or on the wrong side get a penalty called the **hinge loss**. The dial $C$ trades street width against mistakes.
  - **Large $C$:** strict. Few mistakes on training data, narrow street, can [[Overfitting|overfit]].
  - **Small $C$:** relaxed. Wider street, more training mistakes, often generalises better.
  - This is like [[Regularization]], where a small $C$ means strong regularization.
- **Kernel trick:** the training maths only ever uses **dot products between pairs of points**. A kernel function returns the dot product *as if* the points had been lifted into a richer space, without computing the lift. Common choices:
  - **Linear:** no lift.
  - **Polynomial:** curved boundaries from products of features.
  - **RBF (Gaussian):** a similarity that is 1 for identical points and falls towards 0 with distance, giving very flexible boundaries. It has a width setting, usually called gamma.
- **Scale matters:** like [[k-NN]], an SVM is distance-based, so rescale the features first (see [[Normalization]]).
- **Strengths:** works well with few examples and many features, and the result depends only on the support vectors.
- **Limits:**
  - Training slows down a lot as the number of examples grows (commonly said to grow faster than linearly; from memory), so it is rarely used on huge datasets.
  - It outputs a side of the line, not a probability, unless extra steps are added.
  - Choosing the kernel, $C$ and gamma takes tuning with cross-validation (see [[Evaluation Metrics]]).
- **Today:** large neural networks ([[Neural Network]], [[Transformer]]) dominate images, text and speech, but SVMs remain a strong choice for small or medium tabular problems. Using a last-layer linear SVM on features from a neural network is another old, common recipe (from memory).

```python
import numpy as np
from sklearn.svm import SVC          # pip install scikit-learn

# --- 1. A linear SVM on 4 points: the widest street between two groups ---
X = np.array([[0, 0], [0, 1], [2, 0], [2, 1]], float)
y = np.array([-1, -1, 1, 1])
m = SVC(kernel="linear", C=1e6).fit(X, y)          # huge C ~ no mistakes allowed
w, b = m.coef_[0], -m.intercept_[0]                # the rule is: sign(w.x - b)
print(w.round(3), round(b, 3), round(2 / np.linalg.norm(w), 3))   # [1. 0.] 1.0 2.0   margin width 2

# --- 2. Hinge loss and the objective for two candidate lines ---
def objective(w, b, C):
    scores = y * (X @ w - b)                       # >= 1 means correct and outside the margin
    return 0.5 * w @ w + C * np.maximum(0, 1 - scores).sum()
for C in (1, 0.05):
    print(C, objective(np.array([1., 0]), 1.0, C), objective(np.array([.5, 0]), .5, C))
# 1 0.5 2.125      big C: the strict line wins
# 0.05 0.5 0.225   small C: the wide, sloppy line wins

# --- 3. The kernel trick: a squared dot product equals a dot product after a feature lift ---
phi = lambda v: np.array([v[0] ** 2, np.sqrt(2) * v[0] * v[1], v[1] ** 2])
a, c = np.array([1., 2.]), np.array([3., 1.])
print(round(phi(a) @ phi(c), 3), (a @ c) ** 2)     # 25.0 25.0   same number, no lift needed

# --- 4. Ring data: no straight line works, a kernel does ---
rng = np.random.default_rng(0)
th = rng.uniform(0, 2 * np.pi, 40)
r = np.r_[rng.uniform(0, 1, 20), rng.uniform(2, 3, 20)]     # inner disc vs outer ring
R, yr = np.c_[r * np.cos(th), r * np.sin(th)], np.r_[np.ones(20), -np.ones(20)]
print(SVC(kernel="linear").fit(R, yr).score(R, yr), SVC(kernel="rbf", gamma=1).fit(R, yr).score(R, yr))
# 0.725 1.0   straight line gets 72.5%, rbf kernel gets 100% (on the training data only)
```

The last score is on the training data, so it shows the kernel can fit a ring, not that it will do well on new data.

## 🔴 4. Deep / formal
**The widest street.** Label the classes $y_i=+1$ and $y_i=-1$. Look for a line (a rule $\mathbf w\cdot\mathbf x-b$) that puts every point on the correct side with room to spare:

$$
\min_{\mathbf w,\,b}\ \tfrac12\|\mathbf w\|^2\quad\text{subject to}\quad y_i\,(\mathbf w\cdot\mathbf x_i-b)\ \ge\ 1\ \text{ for all } i,\qquad \text{margin width}=\frac{2}{\|\mathbf w\|}
$$

**Where:**
- $\mathbf w$ is the direction across the street (the arrow in the picture), and $b$ shifts the line sideways. $\mathbf w\cdot\mathbf x$ is a dot product (see [[Vector]]). The dividing line is where $\mathbf w\cdot\mathbf x-b=0$.
- $\|\mathbf w\|$ is the length of $\mathbf w$. $\tfrac12\|\mathbf w\|^2$ is a number we want small.
- $y_i(\mathbf w\cdot\mathbf x_i-b)\ge1$ says: point $i$ is on its correct side and at least "1 unit" from the line. $y_i$ is $+1$ or $-1$, so the product is positive only when the side is correct.
- **Matters:** the "1" fixes the scale, so the street edges sit at scores $+1$ and $-1$, and the street's width is then $2/\|\mathbf w\|$. Making the street wider means making $\|\mathbf w\|$ smaller. Without the constraints, the shortest $\mathbf w$ would be zero, which separates nothing.

**Example (tiny numbers):** points $(0,0),(0,1)$ labelled $-1$ and $(2,0),(2,1)$ labelled $+1$. Try $\mathbf w=(1,0)$, $b=1$.
- Scores $\mathbf w\cdot\mathbf x-b$: $-1,\,-1,\,+1,\,+1$. Times the labels: $1,1,1,1$. Every rule is met, with all four points exactly on the street edges.
- Margin width: $2/\|\mathbf w\|=2/1=\mathbf{2}$, which is exactly the gap between the two groups.
- Try a shorter $\mathbf w=(0.5,0)$, $b=0.5$: it would give width $4$, but the scores become $\pm0.5$, so the products are $0.5<1$ and the rules fail.
- Meaning: $(1,0)$ is the shortest $\mathbf w$ allowed, so it gives the widest legal street. The code finds the same $\mathbf w$, $b$ and width.

**Soft margin and the hinge loss.** When no perfect line exists, replace the hard rule by a price for breaking it:

$$
\min_{\mathbf w,\,b}\ \tfrac12\|\mathbf w\|^2+C\sum_{i=1}^{n}\max\!\big(0,\;1-y_i(\mathbf w\cdot\mathbf x_i-b)\big)
$$

**Where:**
- $\tfrac12\|\mathbf w\|^2$ rewards a wide street, as before.
- $\max(0,\,1-y_i(\cdot))$ is the **hinge loss** of point $i$. It is $0$ when the point obeys the rule (score at least 1), and grows in a straight line the further the point is inside the street or on the wrong side.
- $n$ is the number of points. $\sum_i$ adds up the penalties of all points. $C$ is the price per unit of penalty.
- **Matters:** $C$ is the dial. Large $C$ makes every mistake expensive, so the line bends to fit the training points. Small $C$ allows mistakes in exchange for a wider street. Without the hinge term (or with $C=\infty$) a single stray point can make the problem impossible.

**Example (tiny numbers):** the 4 points above, two candidate lines.
- A single hinge value: a $+1$ point with score $2$ costs $\max(0,1-2)=\mathbf 0$, with score $0.5$ costs $\mathbf{0.5}$, with score $-1$ (wrong side) costs $\mathbf 2$.
- Line A: $\mathbf w=(1,0),\,b=1$. Penalty total $0$. Objective: $0.5+C\cdot0=\mathbf{0.5}$.
- Line B: $\mathbf w=(0.5,0),\,b=0.5$. Each of the four points has score $0.5$, so penalty total $4\times0.5=2$. Objective: $0.125+2C$.
- With $C=1$: B costs $2.125$, so **A wins**. With $C=0.05$: B costs $0.225$, so **B wins**.
- Meaning: when mistakes are cheap, a wider, sloppier street wins. When they are costly, the strict line wins. This matches the printed values.

**The kernel trick.** In the usual way of solving the problem, the points appear only through dot products $\mathbf x_i\cdot\mathbf x_j$. Replace each by a kernel:

$$
K(\mathbf x,\mathbf z)=\varphi(\mathbf x)\cdot\varphi(\mathbf z),\qquad\text{for example } K(\mathbf x,\mathbf z)=(\mathbf x\cdot\mathbf z)^2\ \text{ with }\ \varphi(\mathbf x)=\big(x_1^2,\ \sqrt2\,x_1x_2,\ x_2^2\big)
$$

**Where:**
- $\mathbf x,\mathbf z$ are two points. $\varphi$ ("phi") is a recipe that lifts a point into more features (here 2 features become 3).
- $\varphi(\mathbf x)\cdot\varphi(\mathbf z)$ is the dot product of the lifted points. $K$ is the kernel: a function that gives that number **directly from the original points**.
- **Matters:** the lifted space can be huge, or even infinite for the RBF kernel, but $K$ costs only as much as one dot product in the original space. Without the trick, using such rich feature spaces would be too expensive. The cost moves to the number of points instead, because you need $K$ between all pairs.

**Example (tiny numbers):** $\mathbf x=(1,2)$, $\mathbf z=(3,1)$.
- Lift them: $\varphi(\mathbf x)=(1,\,2.828,\,4)$ and $\varphi(\mathbf z)=(9,\,4.243,\,1)$.
- Dot product of the lifted points: $1\cdot9+2.828\cdot4.243+4\cdot1=9+12+4=\mathbf{25}$.
- Kernel directly: $\mathbf x\cdot\mathbf z=1\cdot3+2\cdot1=5$, and $5^2=\mathbf{25}$.
- Meaning: the same answer arrives without ever building the 3-number points. The ring example in the code uses the RBF kernel, which acts as a lift into infinitely many features (a standard fact, stated from memory).

**Further notes:**
- **Why "support vectors":** in the solved problem, only points on or inside the street get a nonzero weight. The line is a weighted sum of those points alone. In the 4-point example, all four sit exactly on the street edges (the library may list only some of them when points tie).
- **Hinge loss vs [[Loss Function]] of [[Logistic Regression]]-style models:** hinge is exactly zero for confident correct points, so those points stop influencing training. Cross-entropy never reaches zero, so every point always pulls a little.
- **Many classes:** an SVM is two-class by nature. Multi-class problems are built from several two-class ones (one-vs-rest or one-vs-one).
- **Link to [[Gradient Descent]]:** the soft-margin objective can also be minimised directly with (sub)gradient methods, which is how large linear SVMs are trained (from memory).
- **Historical note:** the soft-margin SVM and the kernel version were popularised in the 1990s (Cortes and Vapnik, 1995; from memory).

**What came before, and how this improved on it:**
- **Before:** the perceptron and other methods that find *some* separating line (the perceptron is from the late 1950s; from memory).
- **Its limit:** many lines separate the same data and nothing says which is best. And when the classes are not separable by a straight line, it fails.
- **What changed:** pick the line with the widest margin, and use a kernel to separate curved shapes without computing the extra features (the kernel version is by Boser, Guyon and Vapnik in 1992 and the soft margin by Cortes and Vapnik in 1995; from memory, not checked).
- **How much better:** in the code below, a ring inside a ring (not separable by a line) gives a test accuracy of 0.553 for the straight-line SVM and 1.000 for the kernel (RBF) SVM, on 300 test points. The data is easy and noise is small.

**Check it yourself:**

```python
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
X, y = make_circles(600, noise=0.1, factor=0.4, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)
for name, k in (("straight line (linear)", "linear"), ("kernel (RBF)", "rbf")):
    print(f"{name:22s} test accuracy {SVC(kernel=k).fit(Xtr, ytr).score(Xte, yte):.3f}")
```

## Related
- [[Supervised Learning]] · [[Classification]] · [[Regression]] · [[k-NN]] · [[Vector]] · [[Regularization]] · [[Overfitting]] · [[Bias-Variance]] · [[Normalization]] · [[Evaluation Metrics]] · [[Loss Function]] · [[Logistic Regression]] · [[Gradient Descent]] · [[Neural Network]] · [[Transformer]] · [[Convex Optimization]] · [[Perceptron]]
- [[Anomaly Detection]] · [[Anomaly Detection Shoot-out]]
## Sources
- Cortes & Vapnik, *Support-vector networks*, Machine Learning, 1995 (from memory)
- Boser, Guyon & Vapnik, *A training algorithm for optimal margin classifiers*, 1992 (from memory)
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, chapter 12 (from memory)
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online), chapter 9

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/classic-models Support vector machine (SVM):::Separates two groups with the line that leaves the widest empty street between them. The widest street is the safest, because new items that land a little off are still on the right side.

#flashcards/ml/formula #flashcards/topic/classic-models Write the hard-margin SVM and explain each symbol.
?
$$\min_{\mathbf w,\,b}\ \tfrac12\|\mathbf w\|^2\quad\text{subject to}\quad y_i\,(\mathbf w\cdot\mathbf x_i-b)\ \ge\ 1\ \text{ for all } i,\qquad \text{margin width}=\frac{2}{\|\mathbf w\|}$$
- $\mathbf w$: the direction across the street (the arrow in the picture). $b$: shifts the line sideways. The dividing line is where $\mathbf w\cdot\mathbf x-b=0$.
- $\|\mathbf w\|$: the length of $\mathbf w$. Making $\tfrac12\|\mathbf w\|^2$ small makes the street wide, since width is $2/\|\mathbf w\|$.
- $y_i=\pm1$: the label of point $i$. The rule $y_i(\mathbf w\cdot\mathbf x_i-b)\ge1$ says every point must be on its own side, outside the street.

#flashcards/ml/formula #flashcards/topic/classic-models Write the soft-margin SVM and explain each symbol.
?
$$\min_{\mathbf w,\,b}\ \tfrac12\|\mathbf w\|^2+C\sum_{i=1}^{n}\max\!\big(0,\;1-y_i(\mathbf w\cdot\mathbf x_i-b)\big)$$
- $\tfrac12\|\mathbf w\|^2$: rewards a wide street, as before.
- $\max(0,1-y_i(\cdot))$: the hinge loss of point $i$. It is 0 when the point obeys the rule (score at least 1) and grows in a straight line the further the point is inside the street or on the wrong side.
- $n$: the number of points. $C$: how much a violation is punished. Large $C$ allows few violations, small $C$ allows a wider, more tolerant street.

#flashcards/ml/formula #flashcards/topic/classic-models Write the kernel trick and explain each symbol.
?
$$K(\mathbf x,\mathbf z)=\varphi(\mathbf x)\cdot\varphi(\mathbf z),\qquad\text{for example } K(\mathbf x,\mathbf z)=(\mathbf x\cdot\mathbf z)^2\ \text{ with }\ \varphi(\mathbf x)=\big(x_1^2,\ \sqrt2\,x_1x_2,\ x_2^2\big)$$
- $\mathbf x$, $\mathbf z$: two points. $\varphi$: a recipe that lifts a point into more features (here 2 features become 3).
- $\varphi(\mathbf x)\cdot\varphi(\mathbf z)$: the dot product of the lifted points. $K$: the kernel, a function that gives that number directly from the original points.
- So the SVM gets the benefit of the higher-dimensional space without ever building it.

#flashcards/ml/cloze #flashcards/topic/classic-models The SVM keeps the street ==wide==, and the points on its edge are the ==support vectors==.

#flashcards/ml/example #flashcards/topic/classic-models Points $(0,0),(0,1)$ labelled −1 and $(2,0),(2,1)$ labelled +1, with $\mathbf w=(1,0)$, $b=1$. What are the scores and the margin?::Scores w·x − b: −1, −1, +1, +1. Times the labels: 1, 1, 1, 1. Every rule is met, with all four points on the street edges. Margin width 2/‖w‖ = 2.

#flashcards/ml/example #flashcards/topic/classic-models What is the hinge loss for a +1 point with score 2, 0.5 and −1?::max(0, 1 − 2) = 0. max(0, 1 − 0.5) = 0.5. max(0, 1 + 1) = 2.

#flashcards/ml/example #flashcards/topic/classic-models For $\mathbf x=(1,2)$ and $\mathbf z=(3,1)$, check the kernel $(\mathbf x\cdot\mathbf z)^2$.::φ(x) = (1, 2.828, 4), φ(z) = (9, 4.243, 1). Dot product 9 + 12 + 4 = 25. Directly: x·z = 5, and 5² = 25. Same number, without building the lifted points.

#flashcards/ml/image #flashcards/topic/classic-models ![[svm-max-margin.svg]]
What does this picture show?
?
Two groups (black and white dots). The solid line is the dividing line. The dashed lines mark the edges of the margin, the empty street. The dots right on the dashed lines (highlighted) are the support vectors. The arrow w points straight across the street.

#flashcards/ml/image #flashcards/topic/classic-models ![[kernel-trick-idea.svg]]
What does this picture show?
?
Left: two groups in 2-D that no straight line can separate (red inside, purple around). Right: the same points lifted into 3-D, where a flat sheet separates them. A kernel lets the SVM get that benefit without actually building the 3-D points.
