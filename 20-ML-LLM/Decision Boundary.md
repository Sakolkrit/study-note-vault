---
tags: [machine-learning, classification, logistic-regression, geometry]
status: growing
level: 4
tier: 1
---

# Decision Boundary

## 🟢 1. Simple (age 12)
A decision boundary is **the line on a map where a classifier changes its mind**. On one side it says "yes", on the other side "no". Walk across the line and the answer flips.

**Example:** Picture a playground with a rope laid on the grass. Everyone on the left of the rope is on team A, everyone on the right is on team B. The rope is the boundary. Where you stand decides your team.

## 🔵 2. How it works
1. A classifier such as [[Logistic Regression]] first computes a score $z$ from the features: a weighted sum plus a bias (the same recipe as [[Linear Regression]]).
2. It turns the score into a chance between 0 and 1 using the sigmoid curve. A score of 0 gives exactly 50%.
3. It says "class 1" when the chance is 50% or more, which happens exactly when $z\ge0$.
4. The boundary is the set of points where $z=0$: right on the fence, 50-50.
5. With two features, $z=0$ is a straight line. With three it is a flat sheet. Points far from it get chances near 0% or 100%; points close to it are uncertain.
6. A straight boundary cannot separate, say, a small circle of points from a ring around it. If you add curved features like $x_1^2$ and $x_2^2$ ([[Feature Engineering]]), the same method draws a circle.

## 🟠 3. Technical detail
- **Where the line sits:** with $\vec w$ and $b$, the boundary is $w_1x_1+w_2x_2+b=0$. $\vec w$ points towards the "yes" side and is perpendicular to the line. $b$ slides the line.
- **Chance, not just yes/no:** the distance from the line sets the confidence through the sigmoid; the line is where the model is least sure.
- **Moving the 50% rule:** you can say "yes" at 30% instead of 50% (for example when missing a sick patient is worse than a false alarm). This moves the effective boundary and trades [[Precision]] against [[Recall]].
- **Shapes beyond a line:** polynomial features give circles and curves ([[Feature Engineering]]). A [[Neural Network]] learns its own shapes. [[k-NN]] and [[Decision Tree]] draw boundaries without any formula.
- **Inspecting it:** plot a grid of points coloured by predicted class, or print $\vec w$ and $b$ from the trained model.

```python
import numpy as np
w, b = np.array([1.0, 1.0]), -3.0              # boundary: x1 + x2 = 3
sigmoid = lambda z: 1 / (1 + np.exp(-z))
for p in ((1, 1), (2, 1), (3, 2)):
    z = w @ np.array(p) + b
    print(p, "z =", z, "chance of class 1 =", round(float(sigmoid(z)), 3), "->", int(z >= 0))
# (1, 1) z = -1.0 chance 0.269 -> 0
# (2, 1) z = 0.0 chance 0.5 -> 1     (right on the line)
# (3, 2) z = 2.0 chance 0.881 -> 1
```

## 🔴 4. Deep / formal
**The boundary is where the score is zero.** For logistic regression the model says "class 1" when the predicted chance is at least one half. Since the chance is $\sigma(z)$ with $\sigma(0)=\tfrac12$:

$$
z=\vec w\cdot\vec x+b=0
$$

**Where:**
- $\vec w=(w_1,\dots,w_n)$ are the learned weights, $\vec x$ is one point's features, and $b$ is the bias.
- $z$ is the score. $z>0$ means chance above 50% (predict 1), $z<0$ below 50% (predict 0), and $z=0$ is exactly on the boundary.
- **Matters:** the equation $z=0$ describes a straight line (two features) or flat plane (more features), because $z$ is a weighted sum. That is why plain logistic regression is called a **linear classifier**. If you change $\vec x$ into engineered columns, $z=0$ is still straight in those columns but can be curved in the original features.

**Example (tiny numbers):** $\vec w=(1,1)$, $b=-3$, so the line is $x_1+x_2=3$.
- Point $(1,1)$: $z=1+1-3=-1$. Chance $\sigma(-1)=0.269$. Predict 0.
- Point $(2,1)$: $z=0$. Chance $0.5$. On the line (the rule "at least 50%" says 1).
- Point $(3,2)$: $z=2$. Chance $\sigma(2)=0.881$. Predict 1.
- Meaning: moving away from the line on the "yes" side raises the confidence; the chance 0.5 marks the fence.

**Limits:**
- A straight line fails when classes are arranged in a ring or an XOR pattern. This is the classic limit that [[Perceptron]] hit (the famous XOR problem; from memory, Minsky and Papert 1969).
- A very flexible boundary (high-degree features, huge tree) wiggles around single noisy points: [[Overfitting]]. [[Regularization]] smooths it.
- A boundary only says where the chance crosses 50%. Whether the chances themselves are trustworthy is a different question ([[Calibration]]).

**What came before, and how this improved on it:**
- **Before:** a hard threshold on a weighted sum, as in the [[Perceptron]]: the unit says yes or no, with no sense of how sure it is.
- **Its limit:** no confidence, and nothing to train smoothly with (the output jumps), and a straight line cannot handle curved class shapes.
- **What changed:** pass the score through the smooth sigmoid, so the output is a chance and the boundary is just where the chance is 50% ([[Logistic Regression]]); and add curved features to bend the boundary ([[Feature Engineering]]).
- **How much better:** in the code below, on 400 points arranged as a small disc inside a ring, a straight-line logistic regression got 53.7% accuracy (barely better than chance, 50%). Adding the two columns $x_1^2$ and $x_2^2$ gave 100.0% accuracy on the same points (training accuracy on a toy).

**Check it yourself:**

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
r = np.random.default_rng(0); n = 400
ang = r.uniform(0, 2 * np.pi, n); rad = np.r_[r.uniform(0, 1, n // 2), r.uniform(2, 3, n // 2)]
P = np.c_[rad * np.cos(ang), rad * np.sin(ang)]; lab = np.r_[np.zeros(n // 2), np.ones(n // 2)]
acc1 = LogisticRegression().fit(P, lab).score(P, lab)
Q = np.c_[P, P[:, 0] ** 2, P[:, 1] ** 2]
acc2 = LogisticRegression(C=100, max_iter=2000).fit(Q, lab).score(Q, lab)
print("straight boundary accuracy %.3f | with x1^2, x2^2 accuracy %.3f" % (acc1, acc2))
```

## Related
- [[Logistic Regression]] · [[Classification]] · [[Perceptron]] · [[Feature Engineering]] · [[Linear Regression]] · [[Overfitting]] · [[Regularization]] · [[Precision]] · [[Recall]] · [[Calibration]] · [[Neural Network]] · [[k-NN]] · [[Decision Tree]] · [[Machine Learning Specialization Course 1 Week 3]]

## Sources
- Course videos "Machine Learning Specialization" (Andrew Ng), Course 1 Week 3 (transcripts read; see [[Machine Learning Specialization]])
- The XOR limit attributed to Minsky and Papert 1969 is from memory
- The 53.7% and 100.0% figures are measured in this note's code

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/ml-foundations Decision boundary:::The line on the map where a classifier changes its mind. On one side it says yes, on the other no.

#flashcards/ml/formula #flashcards/topic/ml-foundations Write the decision boundary of a linear classifier and explain each symbol.
?
$$z=\vec w\cdot\vec x+b=0$$
- $\vec w=(w_1,\dots,w_n)$: the learned weights. $\vec x$: one point's features. $b$: the bias.
- $z$: the score. $z>0$ means chance above 50% (predict 1), $z<0$ below 50% (predict 0), $z=0$ is exactly on the boundary.
- $z=0$ is a straight line (two features) or flat plane (more features) because $z$ is a weighted sum. That is why plain logistic regression is a linear classifier.
- With engineered columns, $z=0$ is still straight in those columns but can be curved in the original features.

#flashcards/ml/example #flashcards/topic/ml-foundations $\vec w=(1,1)$, $b=-3$. What are $z$ and the chance for $(1,1)$, $(2,1)$, $(3,2)$?::The line is $x_1+x_2=3$. $(1,1)$: $z=-1$, chance $\sigma(-1)=0.269$, predict 0. $(2,1)$: $z=0$, chance 0.5, on the line. $(3,2)$: $z=2$, chance $\sigma(2)=0.881$, predict 1. Moving away from the line on the yes side raises the confidence.

#flashcards/ml/concept #flashcards/topic/ml-foundations Why is logistic regression called a linear classifier?::Its score $z$ is a weighted sum, so $z=0$ is a straight line or flat plane.
