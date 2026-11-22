---
tags: [machine-learning, supervised-learning, classification, logistic-regression, evaluation]
status: growing
level: 4
tier: 1
---

# Classification

## 🟢 1. Simple (age 12)
Classification is a way for a computer to **sort things into groups**. You show it many examples that already have a group name (a label), and it learns what each group looks like. Then you give it something new and it says which group it belongs to.

**Example:** A mail sorter learns from thousands of emails that people marked "spam" or "not spam". It notices that spam often says "FREE PRIZE!!!" and comes from strangers. When a new email arrives, the sorter says "spam" or "not spam". There are only a few possible answers, and the answer is a **name**, not a number. That is the difference from [[Regression]], where the answer is a number.

## 🔵 2. How it works
1. Collect examples with the right group written on each. This is [[Supervised Learning]].
2. Turn each example into a list of numbers (**features**): for an email, the count of certain words, whether the sender is known, and so on.
3. The model combines the features into a **score** for each group. This is the same straight-line formula as in [[Linear Regression]]: each feature times a weight, added up, plus a starting value.
4. The scores are turned into **chances** that add up to 100% (for two groups, a single chance; for more, [[Softmax]]).
5. The model picks the group with the highest chance. Training adjusts the model ([[Gradient Descent]]) so that the correct group gets a high chance on the training examples.
6. For a new example, you may also use a **threshold**: "say spam only if the chance is above 90%". A higher threshold means fewer false alarms but more missed spam.
7. Judge it on new examples it has never seen. Count the right and wrong calls in a [[Confusion Matrix]] and read the scores made from it. [[Evaluation Metrics]] explains them, one note each.

## 🟠 3. Technical detail
- **Kinds of answer:**
  - **Binary:** two groups (spam or not).
  - **Multi-class:** one of many groups (which digit, 0 to 9).
  - **Multi-label:** several tags can apply at once (a photo can be "beach" and "sunset").
- **Common models:** logistic regression (the simplest, and the basic building block, see below), decision trees and forests (rules like "if X then Y"), nearest neighbours (copy the label of the closest examples), support vector machines, and [[Neural Network]]s. An [[LLM]] choosing the next token is doing classification over its whole vocabulary.
- **Loss:** cross-entropy, the penalty for giving the right group a low chance (see [[Loss Function]]).
- **Decision boundary:** the line (or surface) where the model switches its answer. For logistic regression with the threshold at 0.5, it is where the weighted sum equals zero.
- **Metrics:** judge a classifier with the [[Confusion Matrix]] and the scores made from it: [[Accuracy]], [[Precision]], [[Recall]], [[F1 Score]] and the [[ROC Curve and AUC]]. Accuracy can mislead when one group is rare; the overview is in [[Evaluation Metrics]].
- **Class imbalance:** if 99% of emails are not spam, a model that always says "not spam" is 99% accurate and completely useless. Use [[Precision]], [[Recall]] and the [[Confusion Matrix]], collect more rare examples, or weight them more.
- **Which mistake is worse?** Missing a disease (false negative) is usually worse than a false alarm, while marking a good email as spam (false positive) may be worse than letting one spam through. Choose the threshold to match.
- **Calibration:** a model that says "80%" should be right about 80% of the time. Many models are over-confident.

```python
import math
sigmoid = lambda z: 1 / (1 + math.exp(-z))
w, b = 2.0, -4.0                               # hours studied -> chance of passing
for hours in (1, 2, 3):
    print(hours, round(sigmoid(w * hours + b), 3))
# 1 0.119
# 2 0.5
# 3 0.881
```

## 🔴 4. Deep / formal
**Logistic regression.** For a binary question, a model first computes a straight-line score and then squashes it into a chance between 0 and 1 with the **sigmoid**. With one input $x$:

$$
p(y=1\mid x)=\sigma(wx+b)=\frac{1}{1+e^{-(wx+b)}}
$$

**Where:**
- $x$ is the input feature (for example hours studied) and $y$ is the answer: $y=1$ for "yes" (passed) and $y=0$ for "no". $p(y=1\mid x)$ reads "the chance of yes, given the input $x$" (see [[Probability]]).
- $w$ is the **weight**: how strongly the feature pushes towards "yes" (positive) or "no" (negative). $b$ is the **bias**: the starting lean when $x=0$.
- $z=wx+b$ is the raw score. It can be any number, from very negative to very positive.
- $\sigma(z)=1/(1+e^{-z})$ is the **sigmoid**. **Matters:** it squashes any score into the range 0 to 1 so it can be read as a chance. Score 0 gives exactly $0.5$, a large positive score gives nearly 1, and a large negative score nearly 0.
- The answer flips from "no" to "yes" where $z=0$, that is at $x=-b/w$. **Matters:** this is the **decision boundary** for the 0.5 threshold.
- Learning chooses $w$ and $b$ to minimise the cross-entropy loss ([[Loss Function]]) on the labelled examples.

**Example (tiny numbers):** chance of passing from hours studied, with $w=2$ and $b=-4$.
- $x=1$: $z=2\cdot1-4=-2$. $\sigma(-2)=1/(1+e^{2})=\mathbf{0.119}$ (about 12%, unlikely to pass).
- $x=2$: $z=0$. $\sigma(0)=\mathbf{0.5}$, a coin flip. This is the boundary $x=-b/w=4/2=2$.
- $x=3$: $z=2$. $\sigma(2)=\mathbf{0.881}$ (about 88%).
- With the 0.5 threshold: the model says "pass" for 3 hours and "fail" for 1 hour. Meaning: each extra hour of studying here moves the score by $2$, which raises the chance of passing a lot near the boundary and less far from it.

**Connecting the dots: a classifier is a regression score plus a squash.** [[Linear Regression]] and [[Logistic Regression]] start from the same score $z$: each feature times a weight, added up, plus a starting value (the **Where** list above explains $w$ and $b$). With one feature, $z=wx+b$, the same "$mx+b$" as a school straight line (here $m$ is called $w$). Using the pass-the-exam numbers from [[Logistic Regression]] with $w=1.5$ and $b=-4$: for 2, 3 and 4 hours of study, $z=-1,\ 0.5,\ 2$. Linear regression would report those three numbers as they are. Logistic regression squashes them into $0.269,\ 0.622,\ 0.881$, which can be read as chances.

| Model | What it outputs | Range of the output | Used for |
|---|---|---|---|
| [[Linear Regression]] | the score $z$ itself | any number, below 0 or above 1 | [[Regression]]: predicting a number |
| [[Logistic Regression]] | the squashed score $\sigma(z)$ | always between 0 and 1 | [[Classification]]: a chance of yes |

Two more links between them, both checked in the code of the linked notes:
- **Going back from a chance to the straight line:** the logarithm of the odds, $\ln\big(p/(1-p)\big)$, equals $z$ again. A straight line fitted to those log-odds returns the same weights ($R^2=1.000000$). The equation is in [[Logistic Regression]].
- **Swapping one for the other:** fit [[Linear Regression]] to the 0/1 labels and cut at 0.5. On well-behaved data the cut-off lands almost where logistic regression puts it (+0.02 against +0.03 in the check, agreeing on 99.5% of the points), but its outputs fall outside 0 to 1 (32 of 200 points). With 10 far-away points of class 1 the straight line gets pulled (cut-off +0.28 against -0.04, agreement 92.9%). So a regression line can classify in a pinch, but only the squashed version gives trustworthy chances.

**Multi-class and the link to [[Softmax]].** With $K$ groups the model computes a score $z_k$ for each, and the chance of group $k$ is $e^{z_k}/\sum_j e^{z_j}$. For two groups this reduces to the sigmoid above (the difference of the two scores plays the role of $z$). For scores $(2,1,0)$ the chances are $0.665,\ 0.245,\ 0.090$, so the model answers "group 1" with about 67% confidence.

**Further notes:**
- **Generative vs discriminative:** logistic regression models $p(y\mid x)$ directly (discriminative). Other methods model how each group generates its data and then flip it with Bayes' rule ([[Probability]]).
- **Linear vs non-linear boundaries:** a single logistic regression draws a straight boundary. Stacking layers in a [[Neural Network]] or using trees makes curved boundaries, with more risk of [[Overfitting]].
- **Fairness and error costs:** the same model can have different error rates for different groups of people. Check results per group, not only overall.
- **Beyond hard labels:** many real systems use the chance itself (for ranking, or for "send to a human if unsure") instead of only the final yes or no.

**What came before, and how this improved on it:**
- **Before:** the baseline of always answering with the most common class.
- **Its limit:** it never uses the input, so it is always wrong on the rarer class.
- **What changed:** learn a rule from the features that picks the class for each input.
- **How much better:** in the code below, on a breast-cancer measurement set, always answering the most common class gets 0.626, and logistic regression gets 0.959. Accuracy alone can hide mistakes on a rare class, see [[Evaluation Metrics]].

**Check it yourself:**

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
print("always say the most common class:", round((yte == np.bincount(ytr).argmax()).mean(), 3))
print("logistic regression:", round(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr).score(Xte, yte), 3))
```

## Related
- [[Supervised Learning]] · [[Regression]] · [[Softmax]] · [[Loss Function]] · [[Gradient Descent]] · [[Neural Network]] · [[Overfitting]] · [[Probability]] · [[LLM]] · [[Decision Tree]] · [[Evaluation Metrics]] · [[Bayesian Methods]] · [[k-NN]] · [[Support Vector Machine]] · [[Logistic Regression]] · [[Entropy]] · [[KL Divergence]] · [[Bayes' Theorem]] · [[Naive Bayes]] · [[Random Forest]] · [[Image Classification]] · [[Confusion Matrix]] · [[Linear Regression]] · [[F1 Score]] · [[Decision Boundary]]

## Sources
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online), chapter 4
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (free online), chapter 4
- Goodfellow, Bengio & Courville, *Deep Learning*, 2016, chapter 5

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/ml-foundations Classification:::A way for a computer to sort things into groups. It learns from examples that already have a group name (a label), then says which group a new thing belongs to.

#flashcards/ml/formula #flashcards/topic/ml-foundations Write the chance of "yes" for a one-feature classifier and explain each symbol.
?
$$p(y=1\mid x)=\sigma(wx+b)=\frac{1}{1+e^{-(wx+b)}}$$
- $x$: the input feature (for example hours studied). $y$: the answer, 1 for "yes" (passed) and 0 for "no".
- $p(y=1\mid x)$: the chance of yes, given the input $x$.
- $w$: the weight, how strongly the feature pushes towards yes (positive) or no (negative). $b$: the bias, the starting score.
- $\sigma$: the sigmoid. The decision boundary (at the 0.5 threshold) is $x=-b/w$.

#flashcards/ml/cloze #flashcards/topic/ml-foundations The classifier says "yes" when the chance is above the ==0.5== threshold, and the boundary is at $x=$ ==−b/w==.

#flashcards/ml/example #flashcards/topic/ml-foundations With $w=2$, $b=-4$, what are the chances of passing for 1, 2 and 3 hours?::x = 1: z = −2, σ = 0.119 (about 12%). x = 2: z = 0, σ = 0.5, a coin flip, the boundary 4/2 = 2. x = 3: z = 2, σ = 0.881 (about 88%).
