---
tags: [permutation-importance, feature-importance, interpretability, model-agnostic, evaluation, ML]
status: growing
level: 4
tier: 2
---

# Permutation Importance

Permutation importance is **a way to find out which inputs a trained model really uses: scramble one input at a time and see how much worse the model gets**. If scrambling an input changes nothing, the model was not using it. If the model falls apart, it was relying on it.

## 🟢 1. Simple (age 12)
A cook makes soup from five ingredients and you want to know which one matters most. You cannot read the cook's mind, so you play a trick: **secretly swap the salt in one bowl with salt from another bowl.** Nothing changes if the salt did not matter. Then you swap the pepper, the carrots and so on. The ingredient whose swap ruins the soup the most is the one the cook depends on.

**Example:** a robot decides whether to go left or right using "wall ahead?" and "what day is it?". Scramble "what day is it?" and it still plays fine. Scramble "wall ahead?" and it keeps crashing. So the robot uses the wall sign and ignores the day.

## 🔵 2. How it works
1. **Train** a model and measure how good it is on data it did not train on (a score such as error or accuracy).
2. **Pick one input column** (one feature).
3. **Shuffle that column** among the rows, so each row gets another row's value. The column is now random with respect to the answers. The other columns stay as they were.
4. **Measure the score again.** The drop in quality is the importance of that input.
5. **Repeat** with a few different shuffles and average (one shuffle can be unlucky).
6. **Do the same for every column** and rank them.
7. **Remember the traps:** two inputs that carry the same information hide each other, and shuffling can create situations that cannot happen in real life.

![[random-forest-feature-importance.png]]
*A bar chart of feature importance from a random-forest model on edit data (the longest bars are the inputs the model leans on most). Author: Pablo (WMF), CC BY-SA 4.0, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Spambot_rf_feature_importance.png). The picture does not say which importance method it uses, so I use it only to show the shape of the result: one bar per input, ranked.*

## 🟠 3. Technical detail
- **The importance of feature $j$.** Let $f$ be the trained model, $L$ a loss on a test set ($X$, $y$), and $X^{(j,r)}$ the test data with column $j$ shuffled in round $r$:

$$
I_j=\frac1R\sum_{r=1}^{R}\Big[L\big(f;X^{(j,r)},y\big)-L\big(f;X,y\big)\Big]
$$

**Where:**
- $L(f;X,y)$ is the loss (or $1-$ the score) of the model on the real test data; $L(f;X^{(j,r)},y)$ is the same after shuffling column $j$ in round $r$.
- $R$ is the number of shuffles (20 in my toy); averaging over them steadies the number.
- **Matters:** it is measured on **test** data, so it says what the model uses to be right, not just what it was fitted to. It needs no access to the model's insides, only its predictions. A value near 0 means "not used"; a negative value (tiny, from chance) also means "not used".

**Example (tiny numbers):** the test error is $2$ before shuffling. Three shuffles of one column give errors $5,5,8$ (made up).
- Rises: $3,3,6$.
- $I_j=(3+3+6)/3=\mathbf{4}$.
- Meaning: scrambling this column makes the error four units worse on average, so the model relies on it.

- **When there is no score to track: how often do the decisions change?** For a model that picks a move (as in [[Deep Q-Network]]), I also count how often the chosen move changes when column $j$ is shuffled:

$$
C_j=\frac1n\sum_{i=1}^{n}\mathbb 1\Big[\pi(x_i)\neq\pi\big(\tilde x_i^{(j)}\big)\Big]
$$

**Where:**
- $n$ is the number of states looked at. $x_i$ is the $i$-th state and $\tilde x_i^{(j)}$ the same state with feature $j$ replaced by another row's value.
- $\pi(x)$ is the move the model picks in state $x$. $\mathbb 1[\cdot]$ is 1 if the statement is true and 0 if not.
- **Matters:** it needs no score, so it works for any decision rule. It shows how much the model's *behaviour* depends on feature $j$, but not whether that dependence helps (a change can be a mistake or a correction).

**Example (tiny numbers):** 100 states, and after shuffling one feature the move changes in 11 of them (made up).
- $C_j=11/100=\mathbf{0.11}$.
- Meaning: that feature decides the move in about 11% of the states.

- **Code: a real input, a noise input, and a copy.** A model is fitted by least squares with a small penalty (ridge) on a test of 2,000 rows where $y=3a+\text{noise}$. Case 1 has $a$ and a pure-noise column $b$. Case 2 adds $c$, a near copy of $a$. Error rises are measured on a fresh test set:

```python
# file: toy_perm.py
import numpy as np

rng = np.random.default_rng(0)
def data(n):
    a = rng.normal(size=n); b = rng.normal(size=n)               # a really drives y; b is pure noise
    y = 3 * a + rng.normal(scale=0.5, size=n)
    return a, b, y
a, b, y = data(2000); at, bt, yt = data(2000)                     # a training set and a fresh test set

def ridge(X, lam=50.0):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)           # least squares with a small penalty on the weights


def importance(X, w, yy, cols, rounds=20, same_shuffle=False):
    """mean rise of the squared error on the test set when the given columns are shuffled"""
    base = np.mean((X @ w - yy) ** 2); rises = []
    for _ in range(rounds):
        Xs = X.copy(); p = rng.permutation(len(X))
        for c in cols:
            Xs[:, c] = X[p if same_shuffle else rng.permutation(len(X)), c]
        rises.append(np.mean((Xs @ w - yy) ** 2) - base)
    return float(np.mean(rises)), float(base)

print("case 1: features a (real), b (noise)")
X, Xt = np.column_stack([a, b]), np.column_stack([at, bt]); w = ridge(X)
for name, c in (("a", 0), ("b", 1)):
    print(f"   shuffle {name}: error rises by {importance(Xt, w, yt, [c])[0]:.2f}  (error before: {importance(Xt, w, yt, [c])[1]:.2f})")

print("case 2: the same, plus c = a copy of a (with a little noise)")
copy, copyt = a + rng.normal(scale=0.05, size=len(a)), at + rng.normal(scale=0.05, size=len(at))
X, Xt = np.column_stack([a, b, copy]), np.column_stack([at, bt, copyt]); w = ridge(X)
print(f"   fitted weights on a, b, c: {np.round(w, 2)}  (the weight of about 3 is shared between a and its copy)")
for name, c in (("a", [0]), ("c", [2]), ("a and c together (same shuffle)", [0, 2])):
    print(f"   shuffle {name}: error rises by {importance(Xt, w, yt, c, same_shuffle=len(c) > 1)[0]:.2f}")
```

```text
case 1: features a (real), b (noise)
   shuffle a: error rises by 18.32  (error before: 0.26)
   shuffle b: error rises by -0.00  (error before: 0.26)
case 2: the same, plus c = a copy of a (with a little noise)
   fitted weights on a, b, c: [ 1.55 -0.01  1.41]  (the weight of about 3 is shared between a and its copy)
   shuffle a: error rises by 5.23
   shuffle c: error rises by 4.28
   shuffle a and c together (same shuffle): error rises by 18.62
```

- **Reading it:** in case 1 shuffling $a$ raises the error by 18.32 (from 0.26) and shuffling $b$ by 0.00: the method finds the real input and rejects the noise. In case 2 the model shares the weight of about 3 between $a$ (1.55) and its copy $c$ (1.41). Shuffling $a$ alone raises the error by only 5.23 and shuffling $c$ alone by 4.28, because the other copy still carries the signal. Shuffling both together with one shared shuffle raises it by 18.62. So **two copies each look half as important as the pair really is.**
- **What went wrong in the toy.** My first version fitted the model by plain least squares, and the fit gave the weight 3.00 to $a$ and about 0 to its copy $c$, so there was no sharing to show. I added the small ridge penalty, which shares the weight, and the numbers above are from that second version.
- **What I measured on the Snake DQN.** See [[Looking Inside Snake DQN]]: the share of moves that changed when one input was shuffled was 8% to 18% for every one of the 11 inputs, and the ranking of the inputs agreed only moderately between five networks of the same recipe (mean Spearman correlation 0.51).

## 🔴 4. Deep / formal
- **Correlated inputs.** The toy shows the main trap: inputs that carry the same information share the credit, so each looks less important than the group. The usual repair is to shuffle the group together (as in the last line of the toy) or to drop one of the copies.
- **Impossible rows.** Shuffling one column pairs a row's other values with a value from another row. If the inputs are linked (a one-hot group in which exactly one of four is 1, or "danger ahead" and the heading), the result may be a row that can never occur, and the model is judged on inputs it never saw. For one column of the Snake heading (which is "on" in 23.8% of the rows), the share of rows that stop being a valid heading is about $2p(1-p)=0.363$; I measured 0.366 (see [[Interpretability]]). Importance numbers for one-hot features are therefore unreliable.
- **Importance is about this model, not the world.** It says what *this fitted model* uses, not what causes the outcome. A model can lean on a proxy.
- **It depends on the data you shuffle.** Use held-out data; on the training set an overfit model ([[Overfitting]]) can show importance for noise it memorised.
- **Instability.** One shuffle is noisy (hence $R$ rounds), and a different trained copy of the same recipe can give a different ranking. In [[Looking Inside Snake DQN]] the rankings of 5 networks agreed only moderately.
- **Cost.** One pass over the test set for every feature and every round: $p\cdot R$ evaluations for $p$ features. Cheap for a small model, expensive for a big one.
- **Other ways:** drop-column importance (retrain without the feature; costly), Shapley-value methods such as SHAP (from memory, not run), and gradient methods ([[Saliency Map]]).

**What came before, and how this improved on it:**
- **Before:** reading a model's coefficients (scale-dependent, and only for linear models) or a tree ensemble's built-in importance, which is computed from the training splits.
- **Its limit:** coefficients depend on the units of the inputs and say nothing for a network; the built-in tree importance is computed on training data and, as I recall, can favour inputs with many distinct values (from memory).
- **What changed:** one recipe that works for any model, measured on held-out data. Leo Breiman proposed it for random forests (2001), and it was later generalised to any model (from memory, not checked).
- **How much better:** in my toy the recipe cleanly separates the real input (error up by 18.32) from noise (0.00), which a single coefficient scale does not; with a copy present it understates each copy (5.23 and 4.28 against 18.62 for both together), so it is better than coefficients but not a finished answer.

## Related
- [[Interpretability]] · [[Saliency Map]] · [[Looking Inside Snake DQN]] · [[Snake DQN]] · [[Deep Q-Network]] · [[Random Forest]] · [[Feature Engineering]] · [[Overfitting]] · [[Cross-Validation]] · [[Evaluation Metrics]] · [[Neural Network]] · [[Confidence Interval]]

## Sources
- My own toys (Python 3.9.6, NumPy 1.24.4) and the project [[Looking Inside Snake DQN]]. The history (Breiman 2001, the later generalisation), the remark about tree importance favouring many-valued inputs, drop-column importance and SHAP are from memory; I did not read a source for this note.
- Picture: Pablo (WMF), CC BY-SA 4.0, [File:Spambot rf feature importance.png](https://commons.wikimedia.org/wiki/File:Spambot_rf_feature_importance.png), Wikimedia Commons.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/ml-foundations Permutation importance:::Scramble one input column of the test data, re-measure the model's score, and call the drop in quality the importance of that input.

#flashcards/ml/reverse #flashcards/topic/ml-foundations Model-agnostic method:::A method that needs only a model's predictions, not its insides, so it works for any model (permutation importance is one).

#flashcards/ml/concept #flashcards/topic/ml-foundations What are the steps of permutation importance?
?
Train a model and score it on held-out data; pick one input column; shuffle it among the rows; score again; the drop is the importance; repeat with several shuffles and average; do this for every column and rank them; remember correlated inputs and impossible rows.

#flashcards/ml/formula #flashcards/topic/ml-foundations What is the permutation importance of feature $j$?
?
$$I_j=\frac1R\sum_{r=1}^{R}\Big[L\big(f;X^{(j,r)},y\big)-L\big(f;X,y\big)\Big]$$
- $L(f;X,y)$: loss of the model on the real test data; the other term is the loss after shuffling column $j$ in round $r$.
- $R$: number of shuffles; averaging steadies the number.
- Matters: measured on test data, so it shows what the model uses to be right. Near 0 means not used.

#flashcards/ml/example #flashcards/topic/ml-foundations Test error is 2; three shuffles of one column give errors 5, 5 and 8. What is the importance?::Rises are 3, 3 and 6, so $I_j=(3+3+6)/3=4$. The model relies on that column.

#flashcards/ml/formula #flashcards/topic/ml-foundations What is the decision-change rate used for a model that picks moves?
?
$$C_j=\frac1n\sum_{i=1}^{n}\mathbb 1\Big[\pi(x_i)\neq\pi\big(\tilde x_i^{(j)}\big)\Big]$$
- $n$: number of states. $x_i$: a state; $\tilde x_i^{(j)}$: the same state with feature $j$ taken from another row.
- $\pi(x)$: the move the model picks. $\mathbb 1[\cdot]$ is 1 if true, else 0.
- Matters: needs no score, but a change can be a mistake or a correction, so it measures dependence, not usefulness.

#flashcards/ml/example #flashcards/topic/ml-foundations 100 states; after shuffling one feature the move changes in 11. What is the decision-change rate?::$C_j=11/100=0.11$: that feature decides the move in about 11% of the states.

#flashcards/ml/image #flashcards/topic/ml-foundations ![[random-forest-feature-importance.png]]
What does this picture show?
?
A bar chart of feature importance from a random-forest model on edit data: one bar per input, ranked, longest bars are the inputs the model leans on most. It does not say which importance method it uses (Pablo (WMF), CC BY-SA 4.0, Wikimedia Commons).

#flashcards/ml/concept #flashcards/topic/ml-foundations What did the permutation toy show with a real input, noise and a copy?
?
With y = 3a + noise, shuffling a raised the error by 18.32 and the noise column by 0.00. With a near copy c added, the weight was shared (1.55 and 1.41); shuffling a alone raised the error by 5.23, c alone by 4.28, and both together by 18.62. Copies each look half as important as the pair.

#flashcards/ml/concept #flashcards/topic/ml-foundations Why can shuffling one column give misleading results for one-hot inputs?
?
It pairs a row's other values with another row's value and can create rows that never occur (for a heading that is "on" in 23.8% of rows, 36.6% of shuffled rows stop being a valid heading; the formula 2p(1-p) gives 36.3%). The model is judged on inputs it never saw.

#flashcards/ml/concept #flashcards/topic/ml-foundations What did permutation importance show on the five Snake DQNs?
?
Shuffling any one of the 11 inputs changed 8% to 18% of the moves, and the rankings of the inputs agreed only moderately between five networks of the same recipe (mean Spearman correlation 0.51, lowest 0.31).

#flashcards/ml/concept #flashcards/topic/ml-foundations Does permutation importance show what causes the outcome?
?
No. It shows what this fitted model uses; a model can lean on a proxy. It must also be measured on held-out data, or an overfit model can show importance for memorised noise.

#flashcards/ml/concept #flashcards/topic/ml-foundations Which facts in the permutation-importance note are from memory?
?
Breiman 2001 as the origin, the later generalisation to any model, the remark that built-in tree importance can favour many-valued inputs, drop-column importance and SHAP. Not checked.

#flashcards/ml/cloze #flashcards/topic/ml-foundations In permutation importance you ==shuffle== one column of the ==test== data and measure how much worse the model gets.
