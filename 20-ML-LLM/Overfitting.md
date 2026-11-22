---
tags: [overfitting, regularization, generalization, neural-network, machine-learning, LLM]
status: growing
level: 4
tier: 1
---

# Overfitting

## 🟢 1. Simple (age 12)
Overfitting is when an AI **memorises** its practice questions instead of **understanding** them. It scores perfectly on the practice sheet but does badly on new questions it has never seen.

**Example:** Two students prepare for a maths test. One learns the **method**. The other memorises the **answers** to last year's paper. If the test has the same questions, the memoriser wins. But this year's test has new numbers, and the memoriser fails while the first student does fine. A model that memorised is "overfitted".

## 🔵 2. How it works
1. Split your data into two piles: **training data** (the practice sheet the model learns from) and **validation data** (a test it never practises on).
2. Train the model and watch **both** scores.
3. At first both improve: the model is learning real patterns.
4. Later the training score keeps getting better but the validation score **stops improving and starts getting worse**. The model has begun memorising quirks and noise in the practice data. That moment is overfitting.
5. How to fight it:
   - **More (or more varied) data.**
   - **Simpler model**, or fewer settings to tune.
   - **Regularisation:** a penalty that stops the model's numbers (weights) getting wild.
   - **Early stopping:** quit training when the validation score is best.
   - **Dropout:** randomly switch off parts of a [[Neural Network]] during training so no part can memorise alone.
6. The opposite problem is **underfitting**: the model is too simple to learn even the pattern in the practice data.

![[overfitting.svg]]
*A classification example: red and blue dots are two groups, and the ringed dots are the training examples. The wiggly green line separates the training dots perfectly but twists around noise, so it misclassifies many other dots. The smooth black line makes a few mistakes on training data but separates the groups much better in general. Image: Chabacano, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Overfitting.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Generalisation** is how well a model does on data it has not seen. That is the real goal, not a low training loss.
- **Bias and variance:** a too-simple model has high **bias** (always off in the same way). A too-flexible one has high **variance** (changes a lot with each different training set). Overfitting is the high-variance side.
- **Signs:** training [[Loss Function]] keeps falling while validation loss rises. The gap between them is large.
- **Capacity:** the more parameters a model has relative to the amount of data, the easier it is to memorise. Very large models can still generalise well, which is partly understood.
- **Regularisation methods:**
  - **L2 (weight decay):** penalise big weights (below).
  - **Dropout.**
  - **Data augmentation:** flip, crop or reword examples to get more variety.
  - **Early stopping.**
- **Three-way split:** train for learning, **validation** for choosing settings, and a **test** set used once at the very end. Reusing the test set many times quietly overfits to it.
- **In LLMs:** training for many passes over a small [[Fine-tuning]] set makes the model memorise it. Memorised training text can also leak out in answers.

```python
import numpy as np
rng = np.random.default_rng(0)
f = lambda x: 2 * x + 1                       # true pattern: a straight line
xtr = rng.uniform(0, 1, 10); ytr = f(xtr) + rng.normal(0, 0.3, 10)
xte = rng.uniform(0, 1, 200); yte = f(xte) + rng.normal(0, 0.3, 200)

for degree in (1, 9):                         # simple vs very flexible curve
    c = np.polyfit(xtr, ytr, degree)
    mse = lambda x, y: ((np.polyval(c, x) - y) ** 2).mean()
    print(degree, round(mse(xtr, ytr), 3), round(mse(xte, yte), 3))
# 1 0.06 0.117    (line: similar on train and test)
# 9 0.0 1.024     (wiggly curve: perfect on train, far worse on test)
```

I tried seeds 0 to 5 and the degree-9 curve was worse on the test data every time (usually by a lot).

## 🔴 4. Deep / formal
**Regularised loss.** Add a penalty on large weights to the normal loss:

$$
\mathcal{L}_{\text{total}}(\mathbf{w})=\mathcal{L}_{\text{data}}(\mathbf{w})+\lambda\sum_{j} w_j^{2}
$$

**Where:**
- $\mathbf{w}=(w_1,w_2,\dots)$ are the model's weights (the numbers it learns). $w_j$ is one of them.
- $\mathcal{L}_{\text{data}}$ is the ordinary [[Loss Function]]: how wrong the model is on the training data.
- $\sum_j w_j^2$ is the sum of the squares of all the weights. **Matters:** it is large when weights are large, so the model is "charged" for using big weights. Big weights are what let a model bend sharply to fit noise, like the green line.
- $\lambda$ (lambda) is the **penalty strength**, a number you choose. **Matters:** $\lambda=0$ means no penalty (free to overfit); a huge $\lambda$ forces weights near 0 and the model underfits. In between is a trade-off between fitting the data and staying simple.
- $\mathcal{L}_{\text{total}}$ is what training now minimises with [[Gradient Descent]]. (This penalty is also called weight decay, since its gradient shrinks each weight a little every step.)

**Example (tiny numbers):** two weights, $\lambda=0.1$.
- Wild weights $\mathbf{w}=(3,\,-1)$ with data loss $0.5$: penalty $=0.1\cdot(3^2+(-1)^2)=0.1\cdot10=1.0$. Total $=0.5+1.0=\mathbf{1.5}$.
- Calmer weights $\mathbf{w}=(1,\,-1)$ with a slightly worse data loss of $0.6$ (an illustrative number): penalty $=0.1\cdot(1+1)=0.2$. Total $=0.6+0.2=\mathbf{0.8}$.
- Meaning: training prefers the second option. It gives up a little accuracy on the training data ($0.5\to0.6$) in return for much smaller weights, which usually generalise better.

**The generalisation gap.** The number to watch is the difference between how it does on new data and on its own training data:

$$
\text{gap}=\mathcal{L}_{\text{val}}-\mathcal{L}_{\text{train}}
$$

**Where:**
- $\mathcal{L}_{\text{train}}$ is the loss on the data the model learned from.
- $\mathcal{L}_{\text{val}}$ is the loss on held-out validation data the model never trained on.
- The gap is their difference. **Matters:** a small gap means the model behaves the same on new data as on old data. A large gap is the signature of overfitting. A model with a high loss on *both* has a small gap but is underfitting, so look at both numbers, not only the gap.

**Example (tiny numbers):** after some training, $\mathcal{L}_{\text{train}}=0.1$ and $\mathcal{L}_{\text{val}}=0.9$.
- gap $=0.9-0.1=\mathbf{0.8}$. That is big compared with the training loss, so the model is clearly memorising.
- After adding the penalty: $\mathcal{L}_{\text{train}}=0.3$ and $\mathcal{L}_{\text{val}}=0.5$. gap $=\mathbf{0.2}$, and the validation loss (the one that matters) is better, $0.9\to0.5$, even though the training loss got worse.

**Further notes:**
- **Why L2 is "simpler":** from a probability view, an L2 penalty is the same as assuming the weights come from a bell curve around 0 ([[Normal Distribution]]), a built-in preference for small weights. Stated from memory of the standard result; check it.
- **Double descent:** in some modern models the test error first rises as capacity grows, then falls again once the model is very large. Classical "bigger always overfits more" is not the whole story. (From memory of the literature; active research.)
- **Data leakage** looks like great generalisation but is not: if test examples secretly appear in training data, the validation score is fake. This is a real concern for [[LLM]] benchmarks.
- **Occam's razor** is the intuition: among models that fit equally well, prefer the simpler one.

**What came before, and how this improved on it:**
- **Before:** judge a model by its error on the data it was fitted to.
- **Its limit:** that error keeps falling as the model gets more flexible, even when the model starts to memorise noise.
- **What changed:** score the model on held-out data, and pick the model that does best there ([[Regularization]] and cross-validation grew from this).
- **How much better:** in the code below (20 noisy points of a sine wave), training error falls from 0.3877 (a line) to 0.0087 (degree 15) while the held-out error is best at degree 3 (0.0902) and worst at degree 15 (0.4496). Picking by training error would choose degree 15, whose true error is 0.2702. Picking by held-out error chooses degree 3, whose true error is 0.0244, 11 times lower. The held-out set has only 20 points, so degree 3 against 5 or 9 is a close call.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
f = lambda x: np.sin(2 * np.pi * x)
x = np.linspace(0, 1, 20); y = f(x) + 0.25 * rng.standard_normal(20)
xv = rng.uniform(0, 1, 20); yv = f(xv) + 0.25 * rng.standard_normal(20)       # held-out points
xt = np.linspace(0, 1, 500)
print("degree  training error  held-out error  true error")
for d in (1, 3, 5, 9, 15):
    w = np.polyfit(x, y, d)
    mse = lambda a, b: ((np.polyval(w, a) - b) ** 2).mean()
    print(f"{d:6d}  {mse(x, y):14.4f}  {mse(xv, yv):14.4f}  {mse(xt, f(xt)):10.4f}")
```

## Related
- [[Loss Function]] · [[Neural Network]] · [[Gradient Descent]] · [[Fine-tuning]] · [[Normal Distribution]] · [[Probability]] · [[LLM]] · [[Backpropagation]] · [[Regularization]] · [[Bias-Variance]] · [[Dropout]] · [[Bayesian Methods]] · [[Random Forest]] · [[Pandas]] · [[Data Augmentation]] · [[Transfer Learning]] · [[Cross-Validation]] · [[Label Smoothing]] · [[Feature Engineering]] · [[Learning Curve]]
- [[Experiment Tracking]] · [[Matrix Factorization]]
## Sources
- Goodfellow, Bengio & Courville, *Deep Learning*, 2016, chapters 5 and 7
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (free online; bias-variance)
- Belkin et al., *Reconciling modern machine-learning practice and the classical bias-variance trade-off*, 2019 (double descent, from memory)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/training Overfitting:::When an AI memorises its practice questions instead of understanding them. It scores perfectly on the practice sheet but badly on new questions.

#flashcards/ml/formula #flashcards/topic/training Write the regularised loss and explain each symbol.
?
$$\mathcal{L}_{\text{total}}(\mathbf{w})=\mathcal{L}_{\text{data}}(\mathbf{w})+\lambda\sum_{j} w_j^{2}$$
- $\mathbf{w}$: the model's weights. $w_j$: one of them.
- $\mathcal{L}_{\text{data}}$: the ordinary loss on the training data.
- $\sum_j w_j^2$: the sum of the squares of all the weights. It is large when weights are large, so the model is charged for using big weights.
- $\lambda$: how strongly the penalty is applied. Too small does nothing, too big stops the model learning.

#flashcards/ml/formula #flashcards/topic/training Write the generalisation gap and explain each symbol.
?
$$\text{gap}=\mathcal{L}_{\text{val}}-\mathcal{L}_{\text{train}}$$
- $\mathcal{L}_{\text{train}}$: the loss on the data the model learned from.
- $\mathcal{L}_{\text{val}}$: the loss on held-out validation data the model never trained on.
- The difference. A small gap means the model behaves the same on new data as on old. A large gap is the signature of overfitting.

#flashcards/ml/cloze #flashcards/topic/training A large gap between ==training== loss and ==validation== loss is the signature of overfitting.

#flashcards/ml/example #flashcards/topic/training Two weights, $\lambda=0.1$. Which is better: weights (3, −1) with data loss 0.5, or weights (1, −1) with data loss 0.6?::Wild: penalty 0.1 × 10 = 1.0, total 1.5. Calm: penalty 0.1 × 2 = 0.2, total 0.8. The calmer weights win even though the data loss is slightly worse (the 0.6 is an illustrative number in the note).

#flashcards/ml/example #flashcards/topic/training Training loss 0.1 and validation loss 0.9. What is the gap, and what after adding the penalty (0.3 and 0.5)?::Gap 0.9 − 0.1 = 0.8: clearly memorising. After the penalty: gap 0.5 − 0.3 = 0.2, and the validation loss is better.

#flashcards/ml/image #flashcards/topic/training ![[overfitting.svg]]
What does this picture show?
?
A classification example: red and blue dots are two groups, and the ringed dots are the training examples. The wiggly green line separates the training dots perfectly but twists around noise, so it misclassifies many other dots.
