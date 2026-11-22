---
tags: [machine-learning, evaluation, bias-variance, diagnostics]
status: growing
level: 4
tier: 2
---

# Learning Curve

## 🟢 1. Simple (age 12)
A learning curve is **a graph that shows how a model's mistakes change as you give it more practice examples**. It answers a useful question: "if I collect more data, will it actually help?"

**Example:** Two students prepare for a test. One keeps getting the same questions wrong however many practice sheets they do (they are using a bad method). The other gets better with every sheet (they just needed more practice). A learning curve tells you which student your model is.

## 🔵 2. How it works
1. Pick a model and a small number of training examples, say 10. Train, then measure the error on those 10 (the **training error**) and on a separate set the model never saw (the **validation error**).
2. Repeat with 20, 50, 100, and so on. Plot both errors against the number of training examples.
3. **Two curves that stay high and meet** each other: the model is too simple (**high bias**, underfitting). More data will not help, because the model has already reached its limit. Make it richer instead, for example with more features ([[Feature Engineering]]).
4. **A big gap** that slowly closes as data grows: the model is flexible enough to memorise (**high variance**, [[Overfitting]]). More data helps, and so do fewer features or [[Regularization]].
5. For tiny training sets the training error is always low (easy to fit a few points), while the validation error is high.

## 🟠 3. Technical detail
- **What to plot:** error (or cost) on the y-axis, training-set size on the x-axis. Use the same loss for both curves.
- **Reading it:** compare the final training error with a **baseline** you think is achievable (human performance, or a simple model). A training error far above the baseline means high bias; a validation error far above the training error means high variance. The repo's saved lab "Diagnosing Bias and Variance" starts from this idea.
- **Related plot:** error against model complexity (polynomial degree) or against the regularization strength $\lambda$. It shows where the best setting sits, while the learning curve shows what more data would do. See [[Bias-Variance]] and [[Cross-Validation]].
- **Cost:** a learning curve needs many training runs; do it on a subset or use a few sizes.
- **Noise:** each point is one random sample of data, so average several repeats or the curve will wiggle.

```python
import numpy as np
def make(n, seed):
    r = np.random.default_rng(seed); x = r.uniform(-3, 3, n); return x, x ** 2 + r.normal(0, 1.0, n)
xv, yv = make(2000, 99)                                   # fixed held-out set
def curve(deg, sizes):
    rows = []
    for n in sizes:
        tr = va = 0
        for s in range(60):                               # average 60 random training sets
            x, y = make(n, s); c = np.polyfit(x, y, deg)
            tr += np.mean((np.polyval(c, x) - y) ** 2) / 60; va += np.mean((np.polyval(c, xv) - yv) ** 2) / 60
        rows.append((n, round(tr, 2), round(va, 2)))
    return rows
print("straight line (high bias):", curve(1, (25, 200, 800)))
print("degree 8 (high variance): ", curve(8, (25, 200, 800)))
```

## 🔴 4. Deep / formal
**Why the curves look the way they do.** The error a model makes on new data splits roughly into three parts: noise that no model can remove, **bias** (the model's best possible fit is still wrong), and **variance** (the fit changes a lot depending on which sample it saw). More data shrinks variance but leaves bias alone. That is the whole logic of the curve: a straight line fitted to a curved rule has a bias floor, so its validation error plateaus; a flexible model has little bias but high variance at small $n$, so its validation error falls steadily towards the noise level as $n$ grows. (This decomposition is exact for squared error; stated informally here.)

**Limits:**
- It assumes new data comes from the same source as the training data. If it will not, the curve says little.
- It tells you where the problem is, not how to fix it. The fixes (more data, fewer or more features, a different $\lambda$) must still be tried.
- Training error of a flexible model can be misleadingly low: it measures memorising, not learning.

**What came before, and how this improved on it:**
- **Before:** train once on everything and look at one number, the final error.
- **Its limit:** a single number does not say whether the error is high because the model is too simple or because it has too little data, so you could spend months collecting data that cannot help.
- **What changed:** plot the training and validation errors against the amount of training data, and read the shape.
- **How much better:** in the code below (the true rule is a curve, noise level 1), going from 25 to 800 training examples lowered the straight line's validation error only from 8.81 to 8.18 (7% lower), while it lowered the degree-8 model's from 9.08 to 1.01 (about 9 times lower). Same data increase; the curve told which model it would help. (A toy; averages over 60 repeats.)

**Check it yourself:**

```python
import numpy as np
def make(n, seed):
    r = np.random.default_rng(seed); x = r.uniform(-3, 3, n); return x, x ** 2 + r.normal(0, 1.0, n)
xv, yv = make(2000, 99)
def val_error(deg, n):
    tot = 0
    for s in range(60):
        x, y = make(n, s); c = np.polyfit(x, y, deg); tot += np.mean((np.polyval(c, xv) - yv) ** 2) / 60
    return tot
for deg in (1, 8):
    a, b = val_error(deg, 25), val_error(deg, 800)
    print("degree", deg, "validation error with 25 examples %.2f, with 800 examples %.2f" % (a, b))
```

## Related
- [[Bias-Variance]] · [[Overfitting]] · [[Regularization]] · [[Cross-Validation]] · [[Evaluation Metrics]] · [[Feature Engineering]] · [[Loss Function]] · [[Supervised Learning]] · [[Machine Learning Specialization Course 2 Week 3]]

## Sources
- Repo notes `2. Advanced Algorithms/Week 3/notes.pdf` (high bias vs high variance behaviour of learning curves) and the repo's saved lab "Diagnosing Bias and Variance"; see [[Machine Learning Specialization]]
- The 8.81, 8.18, 9.08 and 1.01 figures are measured in this note's code

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation Learning curve:::A graph of a model's training error and validation error against the number of training examples. It answers "will more data actually help?"

#flashcards/ml/concept #flashcards/topic/evaluation Two curves that stay high and meet each other. What is wrong and what helps?::The model is too simple (high bias, underfitting). More data will not help because the model reached its limit. Make it richer, for example with more features.

#flashcards/ml/concept #flashcards/topic/evaluation A big gap between the curves that slowly closes as data grows. What is wrong and what helps?::High variance (overfitting): the model is flexible enough to memorise. More data helps, and so do fewer features or regularization.

#flashcards/ml/concept #flashcards/topic/evaluation For tiny training sets, which error is low and which is high?::Training error is low (easy to fit a few points) and validation error is high.

#flashcards/ml/concept #flashcards/topic/evaluation How do you read a learning curve against a baseline?::Compare the final training error with an achievable baseline (human performance or a simple model). Training error far above the baseline means high bias. Validation error far above training error means high variance.

#flashcards/ml/concept #flashcards/topic/evaluation What is on the two axes of a learning curve?::Error (or cost) on the y-axis and training-set size on the x-axis, with the same loss for both curves.

#flashcards/ml/cloze #flashcards/topic/evaluation A learning curve shows what ==more data== would do. Error against model complexity shows where the ==best setting== sits.

#flashcards/ml/concept #flashcards/topic/evaluation Why are learning curves noisy and costly?::Each point is one random sample of data, so repeat and average. It needs many training runs, so use a subset or a few sizes.
