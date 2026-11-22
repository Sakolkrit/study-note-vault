---
tags: [machine-learning, ensemble, random-forest, boosting, bagging, decision-tree]
status: growing
level: 4
tier: 2
---

# Ensemble Methods

## 🟢 1. Simple (age 12)
An ensemble is **a team of models that vote or add up their answers**. One model can be wrong in its own odd way, but a **crowd** of different models usually makes fewer mistakes, because their errors point in different directions and partly cancel. It is the "wisdom of the crowd" for computers.

**Example:** Guessing how many sweets are in a jar. Any one person's guess may be far off, too high or too low. But if you **average the guesses of 30 people**, the answer is usually surprisingly close, because the high guesses and the low guesses cancel. There is a second way to build a team too: let people guess **one after another**, with each person told how wrong the group still is, so each new person tries to fix what the others missed.

## 🔵 2. How it works
There are two main recipes.

**Bagging (a crowd of independent models).**
1. Take your training data and make many **random copies** by picking examples **with replacement** (so some appear twice and some not at all). Each copy is called a **bootstrap sample**.
2. Train **one model on each copy**, completely independently.
3. To predict, ask every model and **combine**: vote for a category ([[Classification]]) or average for a number ([[Regression]]).
4. The **random forest** is bagging of [[Decision Tree]]s, with one extra trick: at each split a tree may look at only a random handful of the features, so the trees differ more from each other.

![[ensemble-bagging.svg]]
*Bagging. The original data (top) is resampled several times (the "bootstrapping" row: each box is one random sample, with repeats allowed). A classifier is trained on each sample (the green boxes), and their answers are combined into one ensemble classifier (bottom). Image: Sirakorn, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Ensemble_Bagging.svg), CC BY-SA 4.0.*

![[random-forest-four-trees.png]]
*A random forest of four small decision trees, shown classifying one sample as cancer or non-cancer from yes/no questions about mutations A to G. Each tree asks different questions and reaches its own answer (the coloured boxes mark the path it took); the forest takes the majority. Image: CollaborativeGeneticist, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Random_Forest_Diagram_with_Four_Trees.png), CC BY-SA 4.0. It is an illustration, not a real medical model.*

**Boosting (a relay of models, each fixing the last).**
1. Start with a very simple model, such as "always predict the average".
2. Look at the **mistakes** it still makes (the residuals: truth minus prediction).
3. Train a **small new model to predict those mistakes**.
4. **Add** a *fraction* of it to the running prediction. The fraction is the learning rate, a number like 0.1.
5. Repeat for hundreds of rounds. Each small model nibbles away at what is still wrong.
6. **Gradient boosting** is the main version, usually with small decision trees. The well-known libraries XGBoost, LightGBM and CatBoost are fast implementations of this idea.

![[gbm-iterations.png]]
*The error of a gradient boosting model going down as boosting rounds are added: it falls quickly at first and then flattens out. The file's description says it shows a gradient boosting regressor over its iterations; it does not say which dataset, so only the shape of the curve is the lesson. Image: Chamaemelum, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:GradientBOOST.png), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Why teams help:** in the language of [[Bias-Variance]]: bagging mainly lowers **variance** (averaging many wobbly models steadies the result), and boosting mainly lowers **bias** (each round fixes systematic mistakes).
- **Bagging details:**
  - It works best with models that are accurate but unstable, such as deep [[Decision Tree]]s that [[Overfitting]] individually.
  - Each bootstrap sample leaves out about a third of the examples. These **out-of-bag** examples give a free test score without a separate validation set.
  - Models can be trained in parallel (one per core).
- **Random forest details:** each tree is grown deep, with a random subset of features at each split. Few settings to tune, hard to break, and a strong default for tables of numbers. Feature importance scores come almost for free (and can mislead when features are correlated).
- **Boosting details:**
  - The models are built **one after another**, so training is less parallel.
  - The main settings are the number of rounds, the learning rate, and the depth of each tree (usually shallow, 3 to 8 levels).
  - With too many rounds, a boosted model can [[Overfitting]]. Use early stopping on validation data.
  - It can be sensitive to noisy labels, since it keeps trying to fit the hard cases.
  - In practice, boosted trees are often among the strongest methods on **tabular** (spreadsheet-like) data (a common finding; it depends on the dataset).
- **Other ensembles:**
  - **Voting and averaging** of different kinds of models.
  - **Stacking:** train a small final model to combine the others' answers.
  - **Neural network ensembles** are often built by training the same network from different random starts and averaging.
  - Techniques like [[Dropout]] can be seen as a cheap implicit ensemble (see that note).
- **Costs:** more models means more memory and slower prediction, and you lose some of the single-tree readability of a [[Decision Tree]].
- **Where [[LLM]]s fit:** large language models are usually single models, though sampling several answers and taking the majority is an ensemble-like trick (see self-consistency in [[Prompt Engineering]]).

```python
import numpy as np

# 1) averaging models with correlated errors: the variance of the average
rng = np.random.default_rng(0)
M, rho = 4, 0.5
common = rng.normal(size=200000); ind = rng.normal(size=(M, 200000))
p = np.sqrt(rho) * common + np.sqrt(1 - rho) * ind        # 4 predictors, each variance 1, pairwise correlation 0.5
print(p.var(axis=1).mean().round(3), p.mean(0).var().round(3), rho + (1 - rho) / M)   # 1.0 0.625 0.625

# 2) share of distinct examples in a bootstrap sample of size n
print(1 - (1 - 1 / 10) ** 10, 1 - (1 - 1 / 100) ** 100, 1 - np.exp(-1))   # 0.6513215599 0.6339676587267709 0.6321205588285577

# 3) gradient boosting by hand: depth-1 trees ("stumps") fitted to the residuals
x = np.array([1, 2, 3, 4, 5, 6.]); y = np.array([1, 1.5, 3, 5, 5.5, 6.])
F = np.full(6, y.mean()); nu = 0.5                          # start with the average; learning rate 0.5
print(round(y.mean(), 3), round(((y - F) ** 2).mean(), 3))  # 3.667 3.806
def stump(x, r):                                            # best single split of x for predicting r
    best = None
    for t in (1.5, 2.5, 3.5, 4.5, 5.5):
        L = r[x < t].mean(); R = r[x >= t].mean(); err = ((r - np.where(x < t, L, R)) ** 2).sum()
        if best is None or err < best[0]: best = (err, t, L, R)
    return best
for m in range(3):
    e, t, L, R = stump(x, y - F)                            # fit the current mistakes
    F = F + nu * np.where(x < t, L, R)                      # add a fraction of the correction
    print(m, t, round(L, 3), round(R, 3), F.round(3), round(((y - F) ** 2).mean(), 4))
# 0 3.5 -1.833 1.833 [2.75 2.75 2.75 4.583 4.583 4.583] 1.2847
# 1 2.5 -1.5 0.75 [2.    2.    3.125 4.958 4.958 4.958] 0.441
# 2 4.5 -0.396 0.792 [1.802 1.802 2.927 4.76  5.354 5.354] 0.2059
```

The training error falls from 3.806 to 1.285, 0.441 and 0.206 in three rounds. This is *training* error on six points, so it says nothing about new data.

## 🔴 4. Deep / formal
**Why averaging helps, and where it stops: correlated errors.** Take $M$ models whose predictions each have variance $\sigma^2$ and whose predictions are correlated with each other with correlation $\rho$. The variance of their average is:

$$
\operatorname{Var}\!\Big(\frac1M\sum_{m=1}^{M}f_m\Big)=\rho\,\sigma^{2}+\frac{1-\rho}{M}\,\sigma^{2}
$$

**Where:**
- $f_m$ is the prediction of model $m$ and $M$ is how many models there are. $\sigma^2$ is the variance of one model's prediction (see [[Probability]]).
- $\rho$ ("rho") is the correlation between two models' predictions: $0$ means unrelated, $1$ means identical.
- The first term, $\rho\sigma^2$, does **not** depend on $M$. The second term shrinks as $1/M$.
- **Matters:** if the models are independent ($\rho=0$), the variance falls like $\sigma^2/M$, and a crowd of 100 is 100 times steadier. If they are identical ($\rho=1$), averaging does nothing. In between, adding models quickly stops helping, because the floor $\rho\sigma^2$ cannot be removed.
- **Matters:** this is exactly why a random forest takes extra care to make its trees **different** (random samples, random features): it lowers $\rho$, which lowers the floor.
- The formula assumes all models have the same variance and the same pairwise correlation. It is a clean idealisation, and real ensembles are messier. It describes the variance part of the error only, and it leaves the bias unchanged ([[Bias-Variance]]).

**Example (tiny numbers):** $\sigma^2=1$, correlation $\rho=0.5$.
- $M=1$: $0.5+0.5/1=\mathbf{1}$ (a single model).
- $M=4$: $0.5+0.5/4=\mathbf{0.625}$. The simulation in the code prints $0.625$ too.
- $M=100$: $0.5+0.5/100=\mathbf{0.505}$. A hundred models barely beat four, because the floor is $0.5$.
- For comparison, independent models ($\rho=0$) with $M=4$ would give $\mathbf{0.25}$.
- Meaning: decorrelating the models is worth more than adding more of them.

**What a bootstrap sample contains.** Draw $n$ examples from $n$ with replacement. The chance that a particular example appears at least once is:

$$
1-\Big(1-\frac1n\Big)^{n}\ \xrightarrow{\,n\to\infty\,}\ 1-e^{-1}\approx0.632
$$

**Where:**
- $n$ is the number of training examples and each draw picks one of them with chance $1/n$.
- $(1-1/n)$ is the chance a given example is **missed** by one draw, and $(1-1/n)^n$ is the chance it is missed by **all $n$** draws.
- Subtracting from 1 gives the chance it is **included at least once**. $e\approx2.718$ is Euler's number, and the limit $(1-1/n)^n\to e^{-1}$ is a standard result in [[Calculus]].
- **Matters:** each bootstrap sample therefore contains only about **63%** of the distinct examples, so each model sees a somewhat different dataset (which keeps the models different). The remaining **37%** are "out of bag" for that model and can be used as a free test set.

**Example (tiny numbers):**
- $n=10$: $1-0.9^{10}=1-0.3487=\mathbf{0.651}$.
- $n=100$: $\mathbf{0.634}$.
- $n\to\infty$: $1-0.3679=\mathbf{0.632}$.
- Meaning: whatever the dataset size, a bootstrap sample holds about two thirds of the distinct examples and leaves out about a third.

**Boosting as adding corrections.** Gradient boosting builds the model as a sum, one small model at a time:

$$
F_m(x)=F_{m-1}(x)+\nu\,h_m(x),\qquad h_m\ \text{fitted to}\ r_i=y_i-F_{m-1}(x_i)
$$

**Where:**
- $F_{m-1}$ is the combined model after $m-1$ rounds and $F_m$ after $m$ rounds. $F_0$ is a starting guess, such as the average of $y$.
- $h_m$ is the new small model (often a shallow tree), fitted to predict the **residuals** $r_i=y_i-F_{m-1}(x_i)$: how far off the current model still is on example $i$.
- $\nu$ ("nu") is the **learning rate** (also called shrinkage), between 0 and 1, for example $0.1$ or $0.5$. **Matters:** it scales how much of each correction is added. A small $\nu$ needs more rounds but usually generalises better, a large $\nu$ learns fast but can overshoot or overfit.
- **Matters:** for squared error, the residual is exactly the negative gradient of the loss with respect to the current prediction (up to a factor). So each round is a step of [[Gradient Descent]], but in the space of *functions*, hence the name "gradient boosting". For other losses, $h_m$ is fitted to that loss's negative gradient instead.

**Example (tiny numbers):** six points $x=1,\dots,6$ with $y=(1,\,1.5,\,3,\,5,\,5.5,\,6)$, $\nu=0.5$.
- Start: $F_0=\bar y=3.667$ for all points. Training error (mean squared) $=\mathbf{3.806}$.
- Round 1: residuals $(-2.667,-2.167,-0.667,1.333,1.833,2.333)$. The best one-split tree cuts at $x<3.5$, giving $-1.833$ on the left and $+1.833$ on the right (the averages of the residuals on each side). Add half: $F_1=3.667+0.5\cdot(-1.833)=\mathbf{2.75}$ on the left, $3.667+0.5\cdot1.833=\mathbf{4.583}$ on the right. Error $=\mathbf{1.285}$.
- Round 2 fits the new residuals (cut at $x<2.5$) and the error falls to $\mathbf{0.441}$. Round 3 gives $\mathbf{0.206}$.
- Meaning: each round makes a modest correction, and a few rounds already follow the rising pattern. Keep going and the training error keeps falling, which is why early stopping on validation data is needed.

**Further notes:**
- **AdaBoost** (an older boosting method) reweights the training examples so that later models focus on the ones earlier models got wrong. It is a different route to the same idea (from memory).
- **Regularisation in boosting:** small learning rate, shallow trees, random subsampling of rows and columns, and penalties on leaf values ([[Regularization]]). Modern libraries expose all of these.
- **Stacking vs averaging:** averaging weights models equally. Stacking learns the weights, which can help but needs care to avoid leaking training information.
- **Diversity matters more than count:** the first equation says it. Ensembles of models trained differently (different data, features, algorithms) usually beat ensembles of near-copies.
- **Interpretability:** a forest or boosted model has hundreds of trees, so the clear "follow the questions" story of a single [[Decision Tree]] is lost. Tools exist to explain individual predictions, but they are approximations.

**What came before, and how this improved on it:**
- **Before:** one model, tuned as well as possible.
- **Its limit:** a single weak model, such as a one-question tree, has a ceiling.
- **What changed:** combine many models. Boosting trains them in turn, each focusing on the examples the earlier ones got wrong (AdaBoost, Freund and Schapire, 1990s; from memory).
- **How much better:** in the code below (breast-cancer measurements), one stump (a single question) gets a test accuracy of 0.889. Boosting 10 stumps gets 0.936 and 100 stumps gets 0.977, while each member is no smarter than the single stump.

**Check it yourself:**

```python
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
X, y = load_breast_cancer(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
stump = DecisionTreeClassifier(max_depth=1, random_state=0)
print("one stump (a single question):", round(stump.fit(Xtr, ytr).score(Xte, yte), 3))
for n in (10, 100):
    print(f"boosting {n:3d} stumps:", round(AdaBoostClassifier(stump, n_estimators=n, random_state=0).fit(Xtr, ytr).score(Xte, yte), 3))
```

## Related
- [[Decision Tree]] · [[Bias-Variance]] · [[Overfitting]] · [[Classification]] · [[Regression]] · [[Gradient Descent]] · [[Regularization]] · [[Dropout]] · [[Supervised Learning]] · [[Machine Learning]] · [[Probability]] · [[Calculus]] · [[Prompt Engineering]] · [[LLM]] · [[Mixture of Experts]] · [[Random Forest]] · [[Gradient Boosting]]

## Sources
- Breiman, *Bagging Predictors*, 1996 (from memory)
- Breiman, *Random Forests*, 2001 (from memory)
- Friedman, *Greedy Function Approximation: A Gradient Boosting Machine*, 2001 (from memory)
- Chen & Guestrin, *XGBoost: A Scalable Tree Boosting System*, 2016 (from memory)
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online), chapter 8
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (free online), chapters 10, 15 and 16

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/classic-models Ensemble:::A team of models that vote or add up their answers. A crowd of different models usually makes fewer mistakes because their errors point in different directions and partly cancel.

#flashcards/ml/formula #flashcards/topic/classic-models Write the variance of an average of $M$ models and explain each symbol.
?
$$\operatorname{Var}\!\Big(\frac1M\sum_{m=1}^{M}f_m\Big)=\rho\,\sigma^{2}+\frac{1-\rho}{M}\,\sigma^{2}$$
- $f_m$: the prediction of model $m$. $M$: how many models. $\sigma^2$: the variance of one model's prediction.
- $\rho$: the correlation between two models' predictions. 0 means unrelated, 1 means identical.
- The first term $\rho\sigma^2$ does not depend on $M$. The second shrinks as $1/M$.
- Independent models ($\rho=0$) fall like $\sigma^2/M$. Identical models ($\rho=1$) gain nothing. In between, adding models quickly stops helping because the floor $\rho\sigma^2$ stays.

#flashcards/ml/example #flashcards/topic/classic-models $\sigma^2=1$, $\rho=0.5$. What is the variance for $M=1$, 4, 100? And for independent models at $M=4$?::$M=1$: 1. $M=4$: $0.5+0.5/4=0.625$. $M=100$: $0.505$. A hundred models barely beat four because the floor is 0.5. Independent ($\rho=0$), $M=4$: 0.25. Decorrelating the models is worth more than adding more of them.

#flashcards/ml/formula #flashcards/topic/classic-models Write the chance that an example appears in a bootstrap sample and explain each symbol.
?
$$1-\Big(1-\frac1n\Big)^{n}\ \xrightarrow{\,n\to\infty\,}\ 1-e^{-1}\approx0.632$$
- $n$: the number of training examples. Each draw picks one with chance $1/n$.
- $(1-1/n)$: the chance a given example is missed by one draw. $(1-1/n)^n$: missed by all $n$ draws.
- One minus that: the chance it is included at least once. The limit $(1-1/n)^n\to e^{-1}$ is a standard result.
- A bootstrap sample holds only about 63% of the distinct examples, so each model sees a somewhat different dataset, which keeps the models different. The other 37% are "out of bag".

#flashcards/ml/example #flashcards/topic/classic-models What is the inclusion chance for $n=10$, 100, and $n\to\infty$?::$n=10$: $1-0.9^{10}=0.651$. $n=100$: 0.634. $n\to\infty$: $1-0.3679=0.632$. A bootstrap sample holds about two thirds of the distinct examples and leaves out about a third.

#flashcards/ml/formula #flashcards/topic/classic-models Write one round of gradient boosting and explain each symbol.
?
$$F_m(x)=F_{m-1}(x)+\nu\,h_m(x),\qquad h_m\ \text{fitted to}\ r_i=y_i-F_{m-1}(x_i)$$
- $F_{m-1}$: the combined model after $m-1$ rounds. $F_0$: a starting guess, such as the average of $y$.
- $h_m$: the new small model (often a shallow tree) fitted to the residuals $r_i$, how far off the current model still is on example $i$.
- $\nu\in(0,1)$: the learning rate (shrinkage), e.g. 0.1 or 0.5. A small $\nu$ needs more rounds but usually generalises better. A large $\nu$ learns fast but can overshoot or overfit.

#flashcards/ml/example #flashcards/topic/classic-models $x=1..6$, $y=(1,1.5,3,5,5.5,6)$, $\nu=0.5$. What are $F_0$, the error, and $F_1$?::$F_0=\bar y=3.667$, mean squared error $3.806$. Residuals $(-2.667,-2.167,-0.667,1.333,1.833,2.333)$. Best one-split tree cuts at $x<3.5$: $-1.833$ left, $+1.833$ right. $F_1=3.667+0.5(-1.833)=2.75$ on the left and $4.583$ on the right. Error $1.285$.

#flashcards/ml/image #flashcards/topic/classic-models ![[ensemble-bagging.svg]]
What does this picture show?
?
Bagging. The original data (top) is resampled several times (the bootstrapping row: each box is one random sample, with repeats allowed). A classifier is trained on each sample (green boxes) and their answers are combined into one ensemble classifier (bottom). (From the note's caption: Sirakorn, Wikimedia Commons.)

#flashcards/ml/image #flashcards/topic/classic-models ![[random-forest-four-trees.png]]
What does this picture show?
?
A random forest of four small decision trees classifying one sample as cancer or non-cancer from yes/no questions about mutations A to G. Each tree asks different questions and reaches its own answer (the coloured boxes mark the path), and the forest takes the majority. It is an illustration, not a real medical model. (From the note's caption: CollaborativeGeneticist, Wikimedia Commons.)

#flashcards/ml/image #flashcards/topic/classic-models ![[gbm-iterations.png]]
What does this picture show?
?
The error of a gradient boosting model going down as boosting rounds are added: it falls quickly at first and then flattens out. The file does not say which dataset, so only the shape of the curve is the lesson. (From the note's caption: Chamaemelum, Wikimedia Commons.)

#flashcards/ml/cloze #flashcards/topic/classic-models Averaging helps most when the models' errors are ==uncorrelated==. Diversity matters more than ==count==.

#flashcards/ml/reverse #flashcards/topic/classic-models Stacking:::Learns the weights for combining models, where averaging weights them equally. It can help but needs care to avoid leaking training information.
