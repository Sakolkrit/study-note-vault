---
tags: [machine-learning, classification, regression, ensemble, random-forest, decision-tree, bagging]
status: growing
level: 4
tier: 2
---

# Random Forest

A random forest is **a crowd of decision trees that each see a slightly different random version of the data and ask only about a random few features at each question, and then vote**. One deep tree is jumpy and over-eager to memorise. A crowd of different trees averages the jumpiness away. [[Ensemble Methods]] covers bagging, forests and boosting side by side. This note is the forest on its own, with real measurements of why it works and where its popular "feature importance" can mislead.

## 🟢 1. Simple (age 12)
Imagine you want to guess how many sweets are in a jar. One person's guess can be way off. But if you ask 100 people, each of whom looked at the jar from a different angle, and take the average, the answer is usually much better. Each person's mistakes point in different directions, so they cancel out. A random forest does this with a big group of simple "yes/no question" machines. Each one is trained on a different random sample of the examples, and each one is forced to look at different clues.

**Example:** To decide whether a fruit is an apple, one machine asks "is it red?", another "is it round?", another "is it heavy?". Each is wrong sometimes. When they all vote, the majority is right far more often than any one of them.

## 🔵 2. How it works
1. **Start with your labelled examples** (say, 400 patients with a known diagnosis).
2. **Make a random copy** of the data by drawing 400 examples *with replacement*. Some examples appear twice, and some not at all. This is a **bootstrap sample**.
3. **Grow one decision tree** ([[Decision Tree]]) on that copy, all the way down. At every question, the tree may only look at a **random handful of the features**, not all of them, and picks the best question among those.
4. **Repeat** for many trees (100 or 500 is common), each with its own random copy and its own random handfuls. The trees are all a bit different.
5. **To predict:** ask every tree. For a category, take the majority vote ([[Classification]]). For a number, take the average ([[Regression]]).
6. **A free test:** each example was left out of about a third of the trees. Ask only those trees about it. Their answers, collected over all examples, give an honest score without a separate test set. This is the **out-of-bag** score.

![[decision-tree-vs-random-forest.png]]
*One decision tree (left) against a random forest (right). The single tree asks a chain of questions and ends in a green or red answer. The forest is a whole set of small, different trees, each with its own questions, and their answers are combined into one. Image: Jeremybeauchamp, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Decision_Tree_vs._Random_Forest.png), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Two sources of randomness:** random *examples* per tree (bagging) and random *features* per split. Together they make the trees differ, which is what lets the average help.
- **Why averaging helps, and where it stops:** averaging many models cuts the *variance* ([[Bias-Variance]]), but only as far as the models differ from each other. If the trees are copies, nothing improves. The formula is in [[Ensemble Methods]]. The code below *measures* how alike the trees are.
- **The features-per-split setting** (`max_features` in scikit-learn): fewer features per split makes trees more different and less accurate one by one. More features makes each tree stronger but more alike. A common default is the square root of the number of features for classification (from memory, and it is the scikit-learn default).
- **Deep trees are fine here:** each tree is grown to the bottom and so *overfits* ([[Overfitting]]) on its own. That is fine, because the average smooths the overfitting. Bigger forests do not overfit more by adding trees (the code shows the score flattening, not falling).
- **Few settings, hard to break:** number of trees, features per split and a minimum leaf size. It needs no feature scaling, handles mixed data and works in parallel (one tree per core). It is a strong first choice on table-shaped data.
- **Out-of-bag score:** a built-in estimate of how well the forest will do on new data. The code shows it close to the test score.
- **Feature importance, two kinds:**
  - **Impurity-based** (the one a library gives for free): sum up how much each feature reduced the impurity ($\text{Gini}$, see [[Decision Tree]]) over all splits in all trees. It is fast but is known to **favour features with many distinct values**, even useless ones (shown in the code).
  - **Permutation-based:** shuffle one feature's column on held-out data and see how much the score falls (equation 3 below). It measures use on *new* data, but it can show a feature as unimportant when other columns carry the same information (also shown in the code).
- **Limits:** a forest is much bigger and slower to predict than one tree, and it is no longer readable by eye. It cannot extrapolate beyond the range of the training numbers in regression. On many tables, boosted trees do a bit better (see [[Ensemble Methods]]; a common finding that depends on the data).

![[random-forest-out-of-bag.jpg]]
*How the out-of-bag set arises. The original set has four items. Each "bag" draws four items with replacement (the bootstrap sample). Whatever was never drawn is the out-of-bag set for that bag: item A in bag 1, items T and A in bag 2, and nothing in bag 3 (an empty out-of-bag set can happen). Each tree is tested on its own out-of-bag items, which it never saw during training. Image: IlIlIIlIIIlIlll, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Sampling_with_replacement_and_out-of-bag_dataset.jpg), CC BY-SA 4.0.*

The code does five things on scikit-learn's built-in breast-cancer data (569 patients, 30 measurements, benign or malignant; I train on 400 and test on 169): (1) one deep tree against a forest, with the out-of-bag score; (2) how alike the trees are as the features per split change; (3) more trees; (4) importance on a small synthetic problem with a useless many-valued feature; (5) importance on the real data.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
X, y = load_breast_cancer(return_X_y=True)
rng = np.random.default_rng(0)
def split(seed):
    idx = np.random.default_rng(seed).permutation(len(y)); return idx[:400], idx[400:]
# 1) one deep tree vs a forest, 10 random splits
acc = {'tree': [], 'forest': [], 'oob': []}
for sd in range(10):
    tr, te = split(sd)
    acc['tree'].append(DecisionTreeClassifier(random_state=0).fit(X[tr], y[tr]).score(X[te], y[te]))
    rf = RandomForestClassifier(200, oob_score=True, random_state=0, n_jobs=1).fit(X[tr], y[tr])
    acc['forest'].append(rf.score(X[te], y[te])); acc['oob'].append(rf.oob_score_)
print({k: round(float(np.mean(v)), 3) for k, v in acc.items()})
# 2) features tried per split: tree accuracy, how alike the trees are, forest accuracy
tr, te = split(0)
print('max_features | mean single-tree acc | mean correlation between trees | forest acc')
for mf in (1, 5, 15, 30):
    rf = RandomForestClassifier(100, max_features=mf, random_state=0).fit(X[tr], y[tr])
    votes = np.array([t.predict(X[te]) for t in rf.estimators_]).astype(float)       # 100 x n_test, each tree's 0/1 vote
    c = np.corrcoef(votes); rho = (c.sum() - np.trace(c)) / (c.size - len(c))
    print(mf, '|', round(float(np.mean([(v == y[te]).mean() for v in votes])), 3), '|', round(float(rho), 3), '|', round(rf.score(X[te], y[te]), 3))
# 3) more trees
for nt in (1, 5, 25, 100, 400):
    rf = RandomForestClassifier(nt, oob_score=True, random_state=0).fit(X[tr], y[tr]) if nt > 5 else RandomForestClassifier(nt, random_state=0).fit(X[tr], y[tr])
    print('trees', nt, '| test acc', round(rf.score(X[te], y[te]), 3), '| oob', round(rf.oob_score_, 3) if nt > 5 else 'n/a')
# 4) importance, synthetic: one weak but real feature (binary, right 75% of the time) and one pure-noise feature with many distinct values
n = 600; ys = rng.integers(0, 2, n); info = np.where(rng.random(n) < 0.75, ys, 1 - ys).astype(float); noise = rng.random(n)
Xs = np.c_[info, noise]; trs, tes = np.arange(400), np.arange(400, 600)
for leaf in (1, 20):
    rf = RandomForestClassifier(300, min_samples_leaf=leaf, random_state=0).fit(Xs[trs], ys[trs])
    pi = permutation_importance(rf, Xs[tes], ys[tes], n_repeats=30, random_state=0).importances_mean
    print('min_samples_leaf', leaf, '| impurity importance [real, noise]', rf.feature_importances_.round(3), '| permutation [real, noise]', pi.round(3), '| test acc', round(rf.score(Xs[tes], ys[tes]), 3))
# 5) real data: the feature with the top impurity importance can have ~0 permutation importance, because other columns say the same thing
rf = RandomForestClassifier(300, random_state=0).fit(X[tr], y[tr])
imp = rf.feature_importances_; perm = permutation_importance(rf, X[te], y[te], n_repeats=20, random_state=0).importances_mean
top = int(np.argmax(imp)); print('breast cancer: top impurity feature', top, round(float(imp[top]), 3), '| its permutation importance', round(float(perm[top]), 4), '| features with permutation importance above 0.005:', int((perm > 0.005).sum()), 'of 30')
# {'tree': 0.924, 'forest': 0.954, 'oob': 0.96}
# max_features | mean single-tree acc | mean correlation between trees | forest acc
# 1 | 0.899 | 0.721 | 0.959
# 5 | 0.921 | 0.807 | 0.964
# 15 | 0.923 | 0.832 | 0.947
# 30 | 0.921 | 0.826 | 0.959
# trees 1 | test acc 0.893 | oob n/a
# trees 5 | test acc 0.941 | oob n/a
# trees 25 | test acc 0.953 | oob 0.952
# trees 100 | test acc 0.964 | oob 0.96
# trees 400 | test acc 0.959 | oob 0.952
# min_samples_leaf 1 | impurity importance [real, noise] [0.299 0.701] | permutation [real, noise] [ 0.072 -0.06 ] | test acc 0.555
# min_samples_leaf 20 | impurity importance [real, noise] [0.76 0.24] | permutation [real, noise] [0.154 0.023] | test acc 0.68
# breast cancer: top impurity feature 22 0.14 | its permutation importance -0.0015 | features with permutation importance above 0.005: 4 of 30
```
Reading the output (scikit-learn 1.1.2, NumPy; seeds fixed, so it should repeat):
- **Forest against tree:** averaged over 10 random splits, one deep tree scores 0.924 and a forest of 200 scores 0.954. The out-of-bag score, which used no test data, is 0.960, close to the test score. These are accuracies on 169 patients per split, so a difference of 0.01 is about 1 or 2 patients and should not be read as precise.
- **Alike trees:** with 1 feature per split the trees agree with each other the least (correlation 0.721), and each is a bit weaker (0.899). With 5, 15 or 30 features the trees are more alike (0.81 to 0.83) and each is about 0.92. So fewer features per split really does make the trees different, which was the point of the trick. The *forest's* accuracy does not move in a clean line (0.959, 0.964, 0.947, 0.959), because with 169 test patients the noise is bigger than these gaps. I cannot say from this which setting is best.
- **More trees:** 1 tree gives 0.893, 5 give 0.941, and 25 to 400 sit between 0.953 and 0.964, with no steady rise or fall after about 25. Adding trees stops helping, but it did not hurt.
- **Importance, synthetic:** one feature is a real but weak clue (right 75% of the time), the other is pure noise. With fully grown trees, **the impurity importance gives 70% of the credit to the noise column** and 30% to the real one. Permutation importance on held-out data gives the real feature $+0.072$ and the noise $-0.060$. Requiring at least 20 examples per leaf cuts the noise credit to 24% and lifts the test accuracy from 0.555 to 0.68. This is a made-up problem built to show the effect, and the forest here is very weak (0.555 is barely better than a coin flip), so treat it as an illustration.
- **Importance, real data:** on the breast-cancer data the feature with the top impurity importance (feature 22) has a permutation importance of about $-0.002$: shuffling it hurts nothing, because many other columns (size measures of the same cells) carry the same information. Only 4 of 30 features pass 0.005 on permutation importance. So neither score is the "true" importance. They answer different questions.

**For time series (added with the [[Time Series]] notes).** A forest can forecast a series if you turn the past into columns (lag features, rolling averages, calendar columns; see [[Feature Engineering]]) and test it with forward-moving folds ([[Time Series Cross-Validation]]). Its big weakness is that a tree predicts **an average of values it has already seen**, so it can never predict above the highest value in its training data. A series with a trend walks out of that range, and the forest's forecast goes flat. The code uses a steady rise of 2 per step (120 points, the last 20 held out).

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

t = np.arange(120).reshape(-1, 1)
y = 2.0 * t.ravel() + np.random.default_rng(0).normal(0, 3, 120)          # a steady rise of 2 per step plus noise
past, future = slice(0, 100), slice(100, 120)                            # the future lies above everything seen in training
rmse = lambda p: round(float(np.sqrt(((y[future] - p) ** 2).mean())), 1)

rf = RandomForestRegressor(200, random_state=0).fit(t[past], y[past])
lr = LinearRegression().fit(t[past], y[past])
print("true future mean", round(float(y[future].mean()), 1), "| forest", round(float(rf.predict(t[future]).mean()), 1), "| line", round(float(lr.predict(t[future]).mean()), 1))
print("RMSE | forest on the raw values", rmse(rf.predict(t[future])), "| straight line", rmse(lr.predict(t[future])))

step = np.diff(y)                                                        # the fix: learn the change per step, then add the changes back up
rf2 = RandomForestRegressor(200, random_state=0).fit(t[1:100], step[:99])
print("RMSE | forest on the steps", rmse(y[99] + np.cumsum(rf2.predict(t[future]))))
```

Output: `true future mean 219.2 | forest 193.1 | line 219.5`, then `RMSE | forest on the raw values 29.1 | straight line 2.8`, then `RMSE | forest on the steps 14.8`. The forest's average forecast for the held-out stretch is 193.1 against a true 219.2 (it stays near the last values it saw), while a straight line gets 219.5. The forest's error is about 10 times the line's (29.1 against 2.8). The fix in the last lines is to predict the **change per step** (which stays in the training range) and add the changes back up. That halved the forest's error to 14.8, but it is still about 5 times the line's, because the errors of the steps add up. Other fixes are to remove the trend first ([[Trend and Seasonality]]), to difference the series ([[Stationarity]]), or to use a model with a trend term ([[Exponential Smoothing]]).

## 🔴 4. Deep / formal

**The vote, and the out-of-bag vote.** A forest of $B$ trees $T_1,\dots,T_B$ predicts, for a new input $x$,

$$\hat y(x)=\operatorname{mode}\{T_b(x)\}_{b=1}^{B},\qquad \hat y_{\text{oob}}(x_i)=\operatorname{mode}\{T_b(x_i):\ x_i\notin\text{sample}_b\}$$

**Where:**
- $T_b(x)$ is the answer of tree $b$ for input $x$. **Matters:** each tree is only a rough guesser. The forest is only as good as the votes and how different they are.
- $\operatorname{mode}$ means "the most common answer" (for numbers, use the mean instead). **Matters:** the crowd's majority beats most single trees when the trees' mistakes are not all the same ones.
- $B$ is the number of trees. **Matters:** more trees steady the vote, but beyond a point they add nothing (shown in the code).
- $x_i\notin\text{sample}_b$ means tree $b$ did not see example $i$ in its bootstrap sample. **Matters:** only those trees may vote on $x_i$ in the out-of-bag score, so every vote is a fair test on unseen data.
- $\hat y_{\text{oob}}(x_i)$ is the out-of-bag answer for example $i$. **Matters:** comparing it with the true label for all $i$ gives a free accuracy estimate with no separate test set.

**Example (tiny numbers):** 5 trees vote on one new patient: $(1,1,0,1,0)$.
1. Three votes for 1 and two for 0, so $\hat y=1$, with $3/5=0.6$ of the trees agreeing.
2. Out-of-bag for training example $i$ (true label 1): it was left out of trees 2, 4 and 5, with votes $(1,0,1)$ for those trees.
3. Two of three say 1, so $\hat y_{\text{oob}}=1$, which matches the true label, so it counts as correct.
4. Meaning: even a 3-versus-2 vote is a majority, and the out-of-bag vote uses only the trees that never saw the example, so it is a fair test.

**Why random features make the trees differ.** At each split, a tree considers $m$ features chosen at random out of $p$. The chance that one particular feature is among them is

$$P(\text{feature }j\text{ is considered at a split})=\frac{m}{p},\qquad P(\text{absent at } s \text{ splits in a row})=\Big(1-\frac mp\Big)^{s}$$

**Where:**
- $p$ is the total number of features and $m$ is how many each split may look at. **Matters:** $m=p$ means every tree sees everything, so trees all pick the same strongest feature first and become alike. A small $m$ forces different trees to build around different features.
- $m/p$ is the chance a given feature is available at one split. **Matters:** a strong feature that dominates a single tree is unavailable at many splits in the forest, so other features get used.
- $(1-m/p)^s$ is the chance it is missing at $s$ splits in a row (treating the draws as independent). **Matters:** it shows how quickly a strong feature gets blocked from the top of a tree.
- The effect on the forest comes through how alike the trees are. **Matters:** the lower their correlation, the more the average helps (the variance formula in [[Ensemble Methods]]). The code measured correlation 0.721 for $m=1$ against 0.83 for $m=15$.

**Example (tiny numbers):** $p=30$ features, $m=5$ per split.
1. A given feature is considered at a split with chance $5/30=0.167$.
2. It is missing at one split with chance $1-0.167=0.833$, and at 3 splits in a row with chance $0.833^3=0.579$.
3. With $m=30$ it is always considered (chance 1), and never missing.
4. Meaning: with $m=5$, a strong feature misses the first three splits of a tree more than half the time, so such a tree has to be built from the other features. With $m=30$ every tree takes the same strong feature first, and the trees are copies near the top.

**Permutation importance.** For a fitted forest and a held-out set $D$, the importance of feature $j$ is

$$I_j=s(D)-\frac1K\sum_{k=1}^{K}s\big(D^{(j,k)}\big)$$

**Where:**
- $s(D)$ is the forest's score (for example accuracy) on the untouched held-out data. **Matters:** it is the baseline to compare against.
- $D^{(j,k)}$ is the held-out data with column $j$ randomly shuffled (the $k$-th shuffle). **Matters:** shuffling breaks the link between that feature and the label while leaving everything else alone, so any fall in the score is blamed on losing feature $j$.
- $K$ is the number of shuffles that are averaged. **Matters:** one shuffle is noisy, so a handful (the code uses 20 or 30) steadies the result.
- $I_j$ is the average fall in the score. **Matters:** near 0 means "the forest does not use $j$ on new data", or that other columns can stand in for it. A negative value means the shuffled score was *better* by chance, which says the feature is useless or worse.

**Example (tiny numbers):** Held-out accuracy is $0.90$. Shuffling feature A three times gives $0.84, 0.86, 0.85$. Shuffling feature B gives $0.90, 0.91, 0.89$.
1. Feature A: mean shuffled score $=0.85$, so $I_A=0.90-0.85=0.05$.
2. Feature B: mean shuffled score $=0.90$, so $I_B=0.00$.
3. Meaning: losing A costs 5 points of accuracy, so the forest relies on A. Shuffling B changes nothing, so B is not used (or A-like columns cover for it). That is why the code's real-data result (feature 22 at about $-0.002$) does not mean feature 22 is useless: it means the forest does not need it *given* the other columns.

**Other points (from memory, not checked here):**
- Random forests were introduced by Leo Breiman (2001), building on his earlier bagging idea (1996). I did not open either paper for this note.
- The impurity-importance bias toward many-valued features is a known issue in the literature. The code shows it only on one made-up problem.
- A forest's predictions for regression never leave the range of the training targets, since they are averages of training values. This is why forests cannot extrapolate a trend.
- With correlated features, importance can be split among the copies in a way that depends on the random seed. Grouping the related columns is the usual fix.

**What came before, and how this improved on it:**
- **Before:** a single decision tree.
- **Its limit:** a tree changes a lot when the training sample changes (high variance).
- **What changed:** grow many trees on random resamples of the data and random subsets of features, and average their votes (bagging, Breiman 1996; random forests, Breiman 2001; from memory, not checked).
- **How much better:** in the code below (handwritten digits, 8 different splits), one tree has a mean test accuracy of 0.845 with a spread of 0.015. A forest of 100 trees has 0.970 with a spread of 0.006. The price is 100 times the training and a model that is harder to read.

**Check it yourself:**

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X, y = load_digits(return_X_y=True)
t, f = [], []
for s in range(8):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=s)
    t.append(DecisionTreeClassifier(random_state=s).fit(Xtr, ytr).score(Xte, yte))
    f.append(RandomForestClassifier(100, random_state=s).fit(Xtr, ytr).score(Xte, yte))
print("one tree: mean accuracy", round(np.mean(t), 3), " spread (std)", round(np.std(t), 3))
print("forest of 100: mean accuracy", round(np.mean(f), 3), " spread (std)", round(np.std(f), 3))
```

## Related
[[Ensemble Methods]] · [[Decision Tree]] · [[Bias-Variance]] · [[Overfitting]] · [[Classification]] · [[Regression]] · [[Evaluation Metrics]] · [[Probability]] · [[Trees]] · [[Heart Disease Classifier]] · [[Time Series]] · [[Time Series Cross-Validation]] · [[Trend and Seasonality]] · [[Stationarity]] · [[Isolation Forest]] · [[ONNX]] · [[Experiment Tracking]] · [[Model Monitoring Lab]] · [[Bootstrap]] · [[Permutation Importance]]
## Sources
- Breiman (2001), "Random Forests", and Breiman (1996), "Bagging Predictors" (both cited from memory; I did not open them for this note).
- scikit-learn documentation for `RandomForestClassifier` and `permutation_importance`.
- Figures: [Decision_Tree_vs._Random_Forest.png](https://commons.wikimedia.org/wiki/File:Decision_Tree_vs._Random_Forest.png) (Jeremybeauchamp, CC BY-SA 4.0) and [Sampling_with_replacement_and_out-of-bag_dataset.jpg](https://commons.wikimedia.org/wiki/File:Sampling_with_replacement_and_out-of-bag_dataset.jpg) (IlIlIIlIIIlIlll, CC BY-SA 4.0), both on Wikimedia Commons.
- The code and every Example number were run for this note (scikit-learn 1.1.2, NumPy). The synthetic importance problem is invented for teaching.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/classic-models Random forest:::Many different decision trees that each vote, and the crowd's answer is used. Each tree's mistakes point in different directions, so they cancel out.

#flashcards/ml/formula #flashcards/topic/classic-models Write the forest prediction and the out-of-bag prediction, and explain each symbol.
?
$$\hat y(x)=\operatorname{mode}\{T_b(x)\}_{b=1}^{B},\qquad \hat y_{\text{oob}}(x_i)=\operatorname{mode}\{T_b(x_i):\ x_i\notin\text{sample}_b\}$$
- $T_b(x)$: the answer of tree $b$ for input $x$. Each tree is only a rough guesser. The forest is only as good as the votes and how different they are.
- $\operatorname{mode}$: the most common answer (for numbers, use the mean).
- $B$: the number of trees.
- Out-of-bag: for a training example $x_i$, only use the trees that never saw it in their bootstrap sample. This gives a free test estimate.

#flashcards/ml/formula #flashcards/topic/classic-models Write the chance a feature is available at a split and explain each symbol.
?
$$P(\text{feature }j\text{ is considered at a split})=\frac{m}{p},\qquad P(\text{absent at } s \text{ splits in a row})=\Big(1-\frac mp\Big)^{s}$$
- $p$: the total number of features. $m$: how many each split may look at.
- $m=p$ means every tree sees everything, so trees all pick the same strongest feature first and become alike. A small $m$ forces different trees to build around different features.
- $m/p$: the chance a given feature is available at one split. $(1-m/p)^s$: the chance it is missing at $s$ splits in a row.

#flashcards/ml/formula #flashcards/topic/classic-models Write permutation feature importance and explain each symbol.
?
$$I_j=s(D)-\frac1K\sum_{k=1}^{K}s\big(D^{(j,k)}\big)$$
- $s(D)$: the forest's score (for example accuracy) on the untouched held-out data. It is the baseline.
- $D^{(j,k)}$: the held-out data with column $j$ randomly shuffled (the $k$-th shuffle). Shuffling breaks the link between that feature and the label while leaving the rest alone.
- $K$: the number of shuffles. $I_j$: the average drop in score. A big drop means the forest relied on feature $j$.

#flashcards/ml/cloze #flashcards/topic/classic-models A random forest makes its trees ==different== by giving each a bootstrap sample and a limited set of ==features== at each split.

#flashcards/ml/example #flashcards/topic/classic-models 5 trees vote (1,1,0,1,0). What is the forest's answer?::Three votes for 1 and two for 0, so ŷ = 1, with 3/5 = 0.6 of the trees agreeing.

#flashcards/ml/example #flashcards/topic/classic-models With $p=30$ features and $m=5$ per split, what is the chance a feature is considered at a split, and missing at 3 splits in a row?::5/30 = 0.167 considered. Missing at one split: 0.833. At 3 in a row: 0.833³ = 0.579.

#flashcards/ml/example #flashcards/topic/classic-models Held-out accuracy is 0.90. Shuffling feature A gives 0.84, 0.86, 0.85 and feature B gives 0.90, 0.91, 0.89. What are the importances?::A: mean 0.85, I_A = 0.05. B: mean 0.90, I_B = 0.00. Losing A costs 5 points, B costs nothing.

#flashcards/ml/image #flashcards/topic/classic-models ![[decision-tree-vs-random-forest.png]]
What does this picture show?
?
One decision tree (left) against a random forest (right). The single tree asks a chain of questions and ends in a green or red answer. The forest is a set of small, different trees, each with its own questions, and their answers are combined into one.

#flashcards/ml/image #flashcards/topic/classic-models ![[random-forest-out-of-bag.jpg]]
What does this picture show?
?
How the out-of-bag set arises. The original set has four items. Each bag draws four items with replacement (the bootstrap sample). Whatever was never drawn is the out-of-bag set for that bag: item A in bag 1, items T and A in bag 2, and nothing in bag 3. Each tree is tested on its own out-of-bag items, which it never saw during training.

#flashcards/ml/concept #flashcards/topic/classic-models Why can a random forest not forecast a trending series from the raw values?
?
A tree predicts an average of training values it has seen, so it cannot go above the training range. A series with a trend leaves that range and the forecast goes flat. Fixes: predict the change per step and add it up, remove the trend first, difference the series, or use a model with a trend term.

#flashcards/ml/example #flashcards/topic/classic-models In the note's test (a rise of 2 per step, last 20 of 120 points held out), what were the errors of the forest on raw values, a straight line, and the forest on the steps?::RMSE 29.1 for the forest on raw values, 2.8 for the straight line and 14.8 for the forest on the steps. The forest's forecast average was 193.1 against a true 219.2. Synthetic data, my run on this machine.
