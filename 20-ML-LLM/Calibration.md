---
tags: [machine-learning, evaluation, metrics, calibration, probability, classification]
status: growing
level: 4
tier: 2
---

# Calibration

A model is **calibrated** when its confidence matches how often it is right: among the cases where it says "80%", about 80% should turn out to be true. Back to the overview: [[Evaluation Metrics]].

## 🟢 1. Simple (age 12)
Calibration asks whether a model's confidence can be believed. A weather forecaster who says "70% chance of rain" on 100 different days is well calibrated if it rains on about 70 of them. A forecaster who says "99% sure" and is right only 8 times out of 10 is over-confident, even if the forecasts are often right.

**Example:** two students both get 8 out of 10 questions right. One says "I'm 99% sure" on every answer, and the other says "about 80% sure". The second student's confidence matches the results. The first student's does not, and you cannot trust their "99%".

## 🔵 2. How it works
1. Collect the model's predicted chances for many test examples, with the true answers.
2. Group the examples by predicted chance, for instance into 10 bins: 0 to 10%, 10 to 20%, and so on.
3. In each bin, compare two numbers: the **average confidence** (what the model said) and the **actual share of yes-answers** (what happened).
4. If the two match in every bin, the model is calibrated. Plot them against each other to get a **reliability diagram**: a calibrated model lies on the diagonal.
5. Summarise the gaps in one number, the **expected calibration error**: the average gap, with each bin weighted by how many examples it holds.
6. If the model is over-confident, you can fix it afterwards without changing its answers, for example by dividing the scores by a number before squashing (**temperature scaling**).
7. **The catch:** a model can be accurate and badly calibrated at once. Calibration is about the quality of the chances, not about the yes-or-no answers.

## 🟠 3. Technical detail
- **Why models are over-confident:** modern networks trained to a very low loss on their training data tend to give extreme chances on new data (a common finding in the research on calibration; from memory, not checked). Training with a loss like cross-entropy ([[Loss Function]]) rewards confidence in the training set.
- **Why it matters:** when the chance feeds a decision ("send to a human if the model is below 90%"), the chance has to mean what it says. For [[LLM]]s, the confidence of the answer is often used the same way.
- **Metrics:**
  - **Expected calibration error (ECE):** the weighted average gap between confidence and accuracy over bins. It depends a little on the number of bins.
  - **Log loss** (the cross-entropy; see [[Loss Function]]) punishes confident wrong chances heavily, so it improves when calibration improves.
  - A probability score such as the Brier score is another option (from memory).
- **Fixes after training:** temperature scaling (one number divides the scores), or fitting a small mapping from scores to chances on held-out data. This needs data the model was not trained on ([[Cross-Validation]]).
- **Not the same as ranking quality:** the [[ROC Curve and AUC]] and the [[Precision-Recall Curve]] depend only on the order of the scores, so a monotone rescaling (like temperature scaling) does not change them. Calibration is exactly what they do not measure.
- **A model that is calibrated by design:** [[Logistic Regression]] fitted with log loss is often close to calibrated when its form is right, and the check below shows an honest model with an error near zero.

```python
import numpy as np
rng = np.random.default_rng(1); sig = lambda z: 1 / (1 + np.exp(-z))
z = rng.normal(0, 2, 50000); y = rng.random(z.size) < sig(z); p = sig(3 * z)       # an over-confident model: the true score is z, it uses 3 z
idx = np.minimum((p * 5).astype(int), 4)
print("bin of predicted chance | cases | said (average) | really yes")
for b in range(5):
    m = idx == b; print(f"  {b * 20:3d}% to {b * 20 + 20:3d}%        | {m.sum():5d} | {p[m].mean():.3f}          | {y[m].mean():.3f}")
```

## 🔴 4. Deep / formal
**Expected calibration error.** Cut the predictions into $B$ bins by predicted chance. In each bin $b$, compare the share of cases that were really yes (the accuracy of the bin) with the average predicted chance (the confidence of the bin):

$$
\text{ECE}=\sum_{b=1}^{B}\frac{n_b}{n}\,\Big|\,\text{acc}_b-\text{conf}_b\,\Big|
$$

**Where:**
- $n$ is the number of test examples and $n_b$ the number that fall in bin $b$, so $n_b/n$ is the share of examples in that bin.
- $\text{acc}_b$ is the share of examples in bin $b$ that really were yes. $\text{conf}_b$ is the average predicted chance of yes in that bin.
- $|\cdot|$ is the absolute value, the size of the gap with the sign dropped.
- **Matters:** the weights $n_b/n$ make a bin that holds many examples count more than a bin that holds few. Without the weights, one rare, odd bin could dominate the number.

**Example (tiny numbers):** two bins and 40 examples.
- Bin A: 10 examples, average confidence $0.9$, but only $7$ of them were yes, so accuracy $0.7$. Gap $=|0.7-0.9|=\mathbf{0.2}$, weight $10/40=0.25$.
- Bin B: 30 examples, confidence $0.6$, and 18 were yes, so accuracy $0.6$. Gap $=\mathbf{0}$, weight $30/40=0.75$.
- $\text{ECE}=0.25\cdot0.2+0.75\cdot0=\mathbf{0.05}$.
- Meaning: on average the model's confidence is off by 5 percentage points, and all of the error is in the over-confident bin A.

**Temperature scaling.** To make a model less (or more) confident without changing which answer it picks, divide its raw scores by a number $T$ before the sigmoid:

$$
p=\sigma\!\left(\frac{z}{T}\right)
$$

**Where:**
- $z$ is the raw score of the model (see [[Logistic Regression]]) and $\sigma$ is the sigmoid, which turns a score into a chance.
- $T$ is the **temperature**, a number above 0 found on held-out data. **Matters:** $T>1$ softens the chances towards 0.5 (less confident), $T<1$ sharpens them. Since dividing by a positive number does not change the sign of $z$, the yes-or-no answer is unchanged, so accuracy is unchanged.

**Example (tiny numbers):** a raw score $z=3$.
- $T=1$: $\sigma(3)=\mathbf{0.953}$.
- $T=3$: $\sigma(3/3)=\sigma(1)=\mathbf{0.731}$.
- Meaning: the same answer (yes), but the confidence drops from 95% to 73%. If the model is right about 73% of the time at this score, $T=3$ is the calibrated choice.

**Further notes:**
- **Binning choices:** ECE changes with the number of bins and with whether bins hold equal widths or equal counts (from memory).
- **Overall vs per-group:** a model can be calibrated on average and miscalibrated for a group of people, so check results by group.

**What came before, and how this improved on it:**
- **Before:** judge a model only by its yes-or-no answers (accuracy and the scores in [[Evaluation Metrics]]).
- **Its limit:** two models with the same answers can give very different chances, and a decision that uses the chances (a threshold, a cost) is only as good as they are.
- **What changed:** measure the chances themselves (reliability diagram, ECE) and repair them after training with one number, the temperature. Calibration of forecasts is old in weather forecasting (Brier, 1950; from memory, not checked), and its study for neural networks became prominent around 2017 (from memory).
- **How much better:** in the code below (200,000 cases, true chance known), an honest model has an ECE of 0.0007 and a log loss of 0.4617. An over-confident version (scores multiplied by 3) has the same accuracy, 0.7782, but an ECE of 0.1329 and a log loss of 0.6827: among the cases where it gives more than 90%, it is right only 85.4% of the time while saying 98.3%. Dividing by $T=3$ restores an ECE of 0.0007 and a log loss of 0.4617, with the accuracy untouched.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
sig = lambda z: 1 / (1 + np.exp(-z))
n = 200000; z = rng.normal(0, 2, n); y = rng.random(n) < sig(z)          # the truth: chance of yes is sigma(z)
def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1); e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e
for name, p in (("honest model, p = sigma(z)", sig(z)), ("over-confident model, p = sigma(3 z)", sig(3 * z)), ("the same model divided by 3 (temperature scaling)", sig(3 * z / 3))):
    print(f"{name:52s} expected calibration error {ece(p, y):.4f}   log loss {-np.mean(y * np.log(p) + (~y) * np.log(1 - p)):.4f}   accuracy {np.mean((p > .5) == y):.4f}")
p = sig(3 * z); m = (p > 0.9) & (p < 1.0)
print(f"over-confident model: among cases it gives more than 90%, it is right {y[m].mean():.1%} of the time while saying {p[m].mean():.1%}")
```

## Related
- [[Evaluation Metrics]] · [[Precision-Recall Curve]] · [[ROC Curve and AUC]] · [[Loss Function]] · [[Logistic Regression]] · [[Softmax]] · [[Probability]] · [[Classification]] · [[Confusion Matrix]] · [[LLM]] · [[Cross-Validation]] · [[Label Smoothing]] · [[Heart Disease Classifier]] · [[Hallucination]] · [[Probabilistic Forecasting]]

## Sources
- Brier (1950) on forecast verification and the 2017 research on calibrating neural networks: both from memory, not opened.
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online).
- No image: I found no free, on-topic reliability-diagram picture on Wikimedia Commons, so I added none.
- Every number was computed by the code in this note (toy data, NumPy).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation Calibration:::Whether a model's confidence can be believed. A forecaster who says "70% chance of rain" on 100 days is well calibrated if it rains on about 70 of them.

#flashcards/ml/formula #flashcards/topic/evaluation Write the expected calibration error and explain each symbol.
?
$$\text{ECE}=\sum_{b=1}^{B}\frac{n_b}{n}\,\Big|\,\text{acc}_b-\text{conf}_b\,\Big|$$
- $n$: the number of test examples. $n_b$: the number that fall in bin $b$. $n_b/n$: the share of examples in that bin.
- $\text{acc}_b$: the share of examples in bin $b$ that really were yes. $\text{conf}_b$: the average predicted chance of yes in that bin.
- $|\cdot|$: the absolute value, so over- and under-confidence both count.
- ECE is the weighted average gap between confidence and reality. 0 means perfectly calibrated.

#flashcards/ml/formula #flashcards/topic/evaluation Write temperature scaling and explain each symbol.
?
$$p=\sigma\!\left(\frac{z}{T}\right)$$
- $z$: the raw score of the model. $\sigma$: the sigmoid, which turns a score into a chance.
- $T$: the temperature, a number above 0 found on held-out data. $T>1$ softens the chances towards 0.5 (less confident), $T<1$ sharpens them.
- Dividing by a positive $T$ keeps the order of the scores, so the answers stay the same and only the confidence changes.

#flashcards/ml/cloze #flashcards/topic/evaluation Temperature scaling changes the model's ==confidence== but not its ==answers==.

#flashcards/ml/example #flashcards/topic/evaluation Bin A: 10 examples, average confidence 0.9, 7 were yes. Bin B: 30 examples, confidence 0.6, 18 were yes. What is the ECE?::Bin A: accuracy 0.7, gap 0.2, weight 10/40 = 0.25, so 0.05. Bin B: accuracy 0.6, gap 0, weight 0.75. ECE = 0.05 + 0 = 0.05.

#flashcards/ml/example #flashcards/topic/evaluation For a raw score $z=3$, what is the chance at $T=1$ and $T=3$?::T = 1: σ(3) = 0.953. T = 3: σ(1) = 0.731. The same answer (yes), but the confidence drops from 95% to 73%.
