---
tags: [machine-learning, supervised-learning, classification, decision-tree, AI]
status: growing
level: 4
tier: 2
---

# Decision Tree

## 🟢 1. Simple (age 12)
A decision tree is a **game of 20 questions that a computer plays to make a decision**. It asks a yes-or-no question, follows the answer to the next question, and keeps going until it reaches an answer. The computer **learns which questions to ask** by looking at lots of examples.

**Example:** Should you take an umbrella? "Is it raining?" If yes, take it. If no: "Are there dark clouds?" If yes, take it. If no, leave it. Draw the questions as a branching diagram and you have a tree. A computer can build a tree like this from a list of past days where it did or did not rain.

## 🔵 2. How it works
1. Start with examples that each have a **label** (the right answer) and some **features** (facts about the example). This is [[Supervised Learning]].
2. Look for the **one question** about one feature that splits the examples into the **cleanest** groups, where each group is mostly one label.
3. Ask that question at the top of the tree ("the root") and send each example down the matching branch.
4. Repeat on each group: find its best question, split again, and so on.
5. Stop when a group is pure (all one label), or too small, or the tree is deep enough. That final group is a **leaf**, and its answer is the most common label in it.
6. To predict for a new example, start at the top, answer each question about it, and follow the branches to a leaf.

How "clean" a group is gets measured with a number called **impurity** (see the maths in level 4).

![[decision-tree-titanic.png]]
*A small decision tree learned from passenger records of the Titanic. Start at the top: "is sex male?". Follow yes or no, and keep answering ("is age > 9.5?", "is sibsp > 2.5?", where sibsp is the number of siblings or spouses aboard) until you reach a leaf that says "died" or "survived". The numbers under each leaf are, as I read the usual convention for this plot style, the chance of survival and the share of all passengers who land in that leaf (checked only against the picture, not against the source data). Image: Stephen Milborrow, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:CART_tree_titanic_survivors.png), CC BY-SA 3.0.*

## 🟠 3. Technical detail
- **What it is:** a model made of nested if-else rules. Each inner node tests one feature ("age > 9.5"); each leaf holds a prediction.
- **Used for:** [[Classification]] (the leaf gives a category) and [[Regression]] (the leaf gives a number, usually the average of the examples in it).
- **How the splits are chosen:** try every feature and every cut point, and keep the one that lowers impurity the most. Two common impurity scores are **Gini** and **entropy**. This is a *greedy* search: it picks the best split now and never goes back.
- **Strengths:**
  - Easy to read. You can follow the exact path that led to an answer.
  - No need to scale the features (compare [[Normalization]], which many other models need).
  - Handles numbers and categories, and can capture "if A and B" interactions.
- **Weaknesses:**
  - A deep tree memorises the training data and [[Overfitting]] follows. Controls: limit the depth, require a minimum group size, or **prune** the tree afterwards.
  - **Unstable:** a small change in the data can give a very different tree.
  - Splits are straight cuts along one feature at a time, so a diagonal boundary needs many steps.
- **Ensembles fix much of this (see [[Ensemble Methods]]):** a **random forest** averages many trees trained on different random samples. **Gradient boosting** adds trees one at a time, each fixing the previous ones' mistakes. These are often very strong on table-shaped data. (A general statement from memory; results depend on the data.)
- **Tuning and checking:** choose depth and similar settings on validation data, not on the training data (see [[Supervised Learning]]).

```python
def gini(*c):                                   # impurity of a group, given the count of each label
    n = sum(c); return 1 - sum((k / n) ** 2 for k in c)

xs = [1, 2, 3, 4, 5, 6, 7, 8]; ys = [0, 0, 0, 1, 0, 1, 1, 1]       # one feature, labels 0 / 1
for t in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]:                       # try every cut point
    L = [y for x, y in zip(xs, ys) if x < t]; R = [y for x, y in zip(xs, ys) if x >= t]
    imp = (len(L) * gini(L.count(0), L.count(1)) + len(R) * gini(R.count(0), R.count(1))) / len(xs)
    print(t, round(imp, 3))
# 1.5 0.429
# 2.5 0.333
# 3.5 0.2      the lowest, tied with 5.5 (a tie: either cut is as good here)
# 4.5 0.375
# 5.5 0.2
# 6.5 0.333
# 7.5 0.429
```

The tree builder would pick 3.5 (the first of the tied best cuts) and then repeat the search inside each side.

## 🔴 4. Deep / formal
**Gini impurity: how mixed is a group?** For a group where a fraction $p_k$ of the examples have label $k$:

$$
G=1-\sum_{k=1}^{K}p_k^{2}
$$

**Where:**
- $K$ is the number of different labels, and $k$ picks one of them.
- $p_k$ is the share of the group that has label $k$ (so the $p_k$ add up to 1). Squaring and adding gives the chance that two examples picked at random from the group have the **same** label.
- $G$ is one minus that. **Matters:** $G=0$ means the group is pure (everyone has the same label). The largest value is when all labels are equally common (0.5 for two labels). So a good split is one that makes $G$ small in the groups it creates.

**Example (tiny numbers):** a group of 10 examples with 6 "yes" and 4 "no".
- $p_{\text{yes}}=0.6$, $p_{\text{no}}=0.4$. $G=1-(0.36+0.16)=1-0.52=\mathbf{0.48}$.
- A pure group, 4 "yes" and 0 "no": $G=1-(1+0)=\mathbf{0}$.
- A group with 2 "yes" and 4 "no": $p=1/3,\ 2/3$, so $G=1-(1/9+4/9)=\mathbf{0.444}$.
- Meaning: 0.48 says "fairly mixed", 0 says "perfectly sorted". The tree wants to push its groups towards 0.

**Scoring a split: the impurity drop.** A split sends $n_L$ of the $n$ examples left and $n_R$ right. Its quality is how much the average impurity falls:

$$
\Delta G=G_{\text{parent}}-\left(\frac{n_L}{n}\,G_L+\frac{n_R}{n}\,G_R\right)
$$

**Where:**
- $G_{\text{parent}}$ is the impurity of the group before splitting. $G_L$ and $G_R$ are the impurities of the left and right groups after.
- $n_L/n$ and $n_R/n$ are the shares of examples that went each way. **Matters:** these weights make a big group count more than a tiny one. Without them, a split that peels off one pure example would look perfect.
- The bracket is the **average impurity after the split**. $\Delta G$ is the drop. **Matters:** the tree picks the feature and cut point with the biggest $\Delta G$. If no split gives a positive drop, the node becomes a leaf.

**Example (tiny numbers):** split the 10 examples (6 yes, 4 no) into a left group of 4 (4 yes, 0 no) and a right group of 6 (2 yes, 4 no).
- $G_{\text{parent}}=0.48$, $G_L=0$, $G_R=0.444$.
- Weighted average after: $0.4\cdot0+0.6\cdot0.444=\mathbf{0.267}$.
- Drop: $\Delta G=0.48-0.267=\mathbf{0.213}$.
- Meaning: this split makes the groups much cleaner. The tree compares this number with the drop from every other possible question and keeps the largest.

**Further notes:**
- **Entropy** is the other common choice: $-\sum_k p_k\log p_k$ (the same log idea as in [[Loss Function]] and [[Perplexity]]). In practice, Gini and entropy usually give similar trees (a common observation, stated from memory).
- **Greedy is not optimal:** finding the best possible tree is, as far as I recall, computationally very hard (NP-complete), so practical algorithms (CART, ID3, C4.5) choose splits one at a time. Check a textbook for the exact statements.
- **Bias-variance:** a single deep tree has low bias and high variance. Averaging trees (forests) lowers the variance. See [[Overfitting]].
- **Feature importance:** adding up the impurity drops each feature produced gives a rough ranking. It can be misleading with correlated features.
- **Probability outputs:** a leaf's label shares give a chance estimate (the Titanic picture shows them), but these can be poorly calibrated on small leaves.

**What came before, and how this improved on it:**
- **Before:** a single hand-made rule, or a lookup of one feature.
- **Its limit:** one question cannot describe most problems, and a person must find the question.
- **What changed:** grow a tree of questions automatically, choosing each question to split the data best (CART, Breiman and colleagues, 1984, and ID3, Quinlan, 1986; from memory).
- **How much better:** in the code below (breast-cancer measurements), one question (2 leaves) gets 0.889 on test data. Depth 3 (8 leaves) gets 0.901, and an unlimited tree (16 leaves) gets 1.000 on training data and 0.906 on test. The gain over one question is small, and on this one split the unlimited tree did not overfit (0.906 is the best test score), so do not read this as a rule. [[Random Forest]] and [[Ensemble Methods]] are the usual fix for the instability of single trees.

**Check it yourself:**

```python
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
for name, kw in (("one question (depth 1)", dict(max_depth=1)), ("depth 3", dict(max_depth=3)), ("no limit", dict())):
    t = DecisionTreeClassifier(random_state=0, **kw).fit(Xtr, ytr)
    print(f"{name:22s} leaves {t.get_n_leaves():3d}  training accuracy {t.score(Xtr, ytr):.3f}  test accuracy {t.score(Xte, yte):.3f}")
```

## Related
- [[Classification]] · [[Regression]] · [[Supervised Learning]] · [[Overfitting]] · [[Normalization]] · [[Loss Function]] · [[Perplexity]] · [[Probability]] · [[Neural Network]] · [[Ensemble Methods]] · [[Entropy]] · [[KL Divergence]] · [[Random Forest]] · [[Binary Search]] · [[Sorting]] · [[Recursion]] · [[Trees]] · [[Greedy Algorithms]] · [[One-Hot Encoding]]
- [[Isolation Forest]]
## Sources
- Breiman, Friedman, Olshen & Stone, *Classification and Regression Trees* (CART), 1984 (from memory)
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online), chapter 8
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (free online), chapters 9 and 10
- Quinlan, *Induction of decision trees* (ID3), 1986 (from memory)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/classic-models Decision tree:::A game of 20 questions that a computer plays to make a decision. It asks yes/no questions, follows the answers, and reaches an answer at the end. It learns which questions to ask from examples.

#flashcards/ml/formula #flashcards/topic/classic-models Write Gini impurity and explain each symbol.
?
$$G=1-\sum_{k=1}^{K}p_k^{2}$$
- $K$: the number of different labels. $k$ picks one of them.
- $p_k$: the share of the group that has label $k$ (they add up to 1). Squaring and adding gives the chance two random examples from the group have the same label.
- $G$: one minus that. $G=0$ means the group is pure (everyone has the same label). Larger means more mixed.

#flashcards/ml/formula #flashcards/topic/classic-models Write the Gini gain of a split and explain each symbol.
?
$$\Delta G=G_{\text{parent}}-\left(\frac{n_L}{n}\,G_L+\frac{n_R}{n}\,G_R\right)$$
- $G_{\text{parent}}$: the impurity of the group before splitting. $G_L$, $G_R$: the impurities of the left and right groups after.
- $n_L/n$, $n_R/n$: the shares of examples that went each way. These weights make a big group count more than a tiny one.
- $\Delta G$: how much cleaner the groups got. The tree picks the question with the biggest drop.

#flashcards/ml/cloze #flashcards/topic/classic-models A tree picks the split with the biggest ==drop== in impurity, and a ==pure== group has Gini 0.

#flashcards/ml/example #flashcards/topic/classic-models A group of 10 has 6 yes and 4 no. What is its Gini, and for a pure group of 4 yes, and for 2 yes and 4 no?::0.6 and 0.4: G = 1 − (0.36 + 0.16) = 0.48. Pure: 1 − 1 = 0. 2 yes and 4 no: 1 − (1/9 + 4/9) = 0.444.

#flashcards/ml/example #flashcards/topic/classic-models Split the 10 examples (6 yes, 4 no) into left 4 (4 yes) and right 6 (2 yes, 4 no). What is $\Delta G$?::G_parent = 0.48, G_L = 0, G_R = 0.444. Weighted after: 0.4 × 0 + 0.6 × 0.444 = 0.267. ΔG = 0.48 − 0.267 = 0.213.

#flashcards/ml/image #flashcards/topic/classic-models ![[decision-tree-titanic.png]]
What does this picture show?
?
A small decision tree learned from Titanic passenger records. Start at the top ("is sex male?"), follow yes or no and keep answering until a leaf says "died" or "survived". (The numbers under each leaf are, as the note reads the usual convention, the chance of survival and the share of passengers in that leaf.)
