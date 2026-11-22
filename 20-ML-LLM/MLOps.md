---
tags: [MLOps, machine-learning, deployment, monitoring, reproducibility, DevOps, ML]
status: growing
level: 4
tier: 3
---

# MLOps

MLOps is the set of habits and tools that **keep a machine-learning model working after the first success**: recording how it was made, packaging it, running it, watching it, and replacing it when it goes stale. The name joins "machine learning" and "operations".

## 🟢 1. Simple (age 12)
Baking one great cake in your kitchen is one thing. Running a bakery that bakes 500 cakes a day, every one tasting the same, is another. You write the recipe down exactly. You note which flour you used. You taste a sample every morning. If customers start to complain, you find out why: did the flour change? **MLOps is the "run the bakery" part of machine learning.** The model is the cake, and MLOps is everything that keeps it good day after day.

**Example:** you train a model that guesses house prices. It works well in your notebook. Six months later nobody remembers which settings made it, the file is called `final_v2_REAL.pkl`, and house prices in the market have moved so the guesses are quietly getting worse. MLOps would have saved the settings, versioned the file and raised a warning when the incoming houses started to look different.

## 🔵 2. How it works
1. **Write down every run** ([[Experiment Tracking]]): the code version, the data, the settings and the scores, so any result can be found and rebuilt later.
2. **Version everything** with [[Git]] for code, plus a version or fingerprint for the data and the model file.
3. **Test before you ship.** Check that the new model beats the old one on a fixed check set, and that its code and inputs behave ([[pytest]]).
4. **Package it** so it runs the same anywhere: a [[Docker]] container, and a portable model file ([[ONNX]]; compare [[Safetensors]], [[GGUF]]).
5. **Serve it:** put it behind an API ([[FastAPI]]) or in a batch job, and keep the previous version so you can roll back.
6. **Watch it** while it runs: speed and errors, but also **whether the incoming data still looks like the training data** ([[Data Drift]]), and, when the true answers finally arrive, whether the predictions are still good.
7. **Close the loop:** when the watch says "stale", retrain on fresh data and go back to step 1. Automating this loop is the heart of MLOps.
8. A mental check: the model is a small part of a working system. Most of the effort sits around it (data, serving, monitoring), not in the model.

![[mlops-venn-diagram.svg]]
*MLOps as the overlap of three fields: machine learning, DevOps and data engineering. Author Cmbreuel, [ML Ops Venn Diagram.svg on Wikimedia Commons](https://commons.wikimedia.org/wiki/File:ML_Ops_Venn_Diagram.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Why ML needs more than ordinary software practice.** Ordinary software changes only when someone changes the code. A model's behaviour depends on **code + data + settings + randomness**, and it can get worse **with no code change at all** because the world changes. And it fails *silently*: no error message, just worse answers.
- **Common failure modes.**
  - **Not reproducible:** you cannot rebuild last month's model.
  - **Training-serving skew:** features are computed differently in training and in production, so the model sees different numbers than it learned from.
  - **Data and concept drift:** the inputs, or the link between inputs and answers, change over time ([[Data Drift]]).
  - **Delayed labels:** the true answer arrives weeks later, so you cannot measure accuracy live; you watch the inputs in the meantime.
  - **Feedback loops:** the model's outputs change the data it will later learn from (the recommender and fraud cases).
- **Pieces and the tool I tried for each** (project [[Model Monitoring Lab]], one small example, not the only way):

| Job | What I used | Where it is explained |
|---|---|---|
| record runs and scores, find the best | MLflow (SQLite store) | [[Experiment Tracking]] |
| name and version the chosen model | MLflow model registry with the alias "champion" | [[Experiment Tracking]] |
| portable model file, fast CPU serving | ONNX and ONNX Runtime | [[ONNX]] |
| detect changed inputs without labels | PSI and a Kolmogorov-Smirnov test per feature | [[Data Drift]] |
| rebuild a result | seeds and fixed splits | the toy below |

- **Code: reproducibility in miniature.** Fingerprint the data (a short hash), train twice with the same seed, then with a different seed. If two runs with the same recorded data and seed give different predictions, something is not under control.

```python
import hashlib
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor


def fingerprint(a):                                         # a short hash: the same numbers always give the same text
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


X, y = make_regression(n_samples=500, n_features=5, noise=10, random_state=0)


def train_and_hash(X, seed):                                # train, then fingerprint what the model predicts for 50 rows
    model = RandomForestRegressor(n_estimators=50, random_state=seed).fit(X, y)
    return fingerprint(model.predict(X[:50]))


print("data fingerprint          :", fingerprint(X))
print("seed 0, run 1 predictions :", train_and_hash(X, 0))
print("seed 0, run 2 predictions :", train_and_hash(X, 0))
print("seed 1 predictions        :", train_and_hash(X, 1))
X2 = X.copy(); X2[0, 0] += 0.001                            # change ONE number in the data by 0.001
print("data fingerprint, 1 value changed:", fingerprint(X2))
print("seed 0 predictions on changed data:", train_and_hash(X2, 0))
```

Output (run on my machine):

```text
data fingerprint          : 5869d8d6b45b
seed 0, run 1 predictions : a4c0769a124f
seed 0, run 2 predictions : a4c0769a124f
seed 1 predictions        : c33b62015b7e
data fingerprint, 1 value changed: 31216d9a0a32
seed 0 predictions on changed data: a4c0769a124f
```

- **Reading it:** two runs with seed 0 give the same prediction fingerprint (`a4c0769a124f`), and seed 1 gives a different one. A single value changed by 0.001 changes the **data** fingerprint, but the model's predictions on those 50 rows stayed the same. So a good prediction check alone would not have noticed that the data changed; a data fingerprint does.
- **Maturity levels.** Google's MLOps guidance describes levels from "everything manual" through "automated training pipeline" to "automated build, test and deploy of the pipeline itself" (from memory; I did not read it). My lab sits at the first rung (scripts, with tracking and a registry).

## 🔴 4. Deep / formal
There is no single formula for MLOps. What can be said exactly is what should be **recorded and compared**:

- **A run is a record** $(\text{code version},\ \text{data fingerprint},\ \text{settings},\ \text{seed},\ \text{environment},\ \text{scores},\ \text{model file})$. If any part is missing, the result cannot be rebuilt.
- **A release gate** compares the candidate and the current model on the **same fixed check set** with the **same metric**. A candidate is promoted only if its score is better by more than the noise. My lab did **not** do the noise test: the top three validation errors (0.468, 0.471, 0.475) are so close that "best" is partly luck on one validation set of 1,136 rows.
- **Monitoring has three layers**, in the order they become available:
  1. *System* (latency, errors): immediately.
  2. *Inputs* (drift): immediately, no labels needed. In my lab this was the earliest signal (week 4).
  3. *Outcomes* (real error): only when labels arrive; in my lab the real error first left its normal range in week 6.
- **Trade-off between sensitive and noisy alarms.** A rule that flags every small input change will cry wolf; one that waits for the error to rise will be late. My plain "any feature has $p<0.05$" rule fired on 28 of 200 batches with no drift; a Bonferroni-corrected rule fired on 0 of 200 and still caught the drift two weeks before the error left its normal range ([[Data Drift]]).
- **The same seed does not always give the same numbers:** a different library version, thread count or hardware can change results. Record the library versions (I did: see the lab's versions list).
- **Cost vs value.** Each added piece (registry, monitoring, automation) costs time and has to be maintained. A model used once, or by one person, rarely needs more than steps 1, 2 and 3.

**What came before, and how this improved on it:**
- **Before:** models handed over as a notebook or a file with a note, and deployed by hand (the "throw it over the wall" pattern). The idea that ML systems carry a large hidden maintenance cost was laid out in "Hidden Technical Debt in Machine Learning Systems" (Sculley et al., NeurIPS 2015, from memory, not read here). The word MLOps spread afterwards (from memory).
- **Its limit:** nobody could rebuild a model, tell when it went stale, or roll it back; the first sign of trouble was an unhappy user.
- **What changed:** treat model building like software delivery, with versioning, testing, packaging, deployment and monitoring (the DevOps idea) plus checks specific to data and models.
- **How much better:** measured in my lab: all 6 training runs are queryable in one call, the registered model can be loaded by name (`models:/housing-model@champion`), the ONNX copy matches the original to $1.7\times10^{-6}$ in the worst row, and the input-drift alarm came **2 weeks before** the real error left its normal range (week 4 against week 6, one simulated stream, 8 weeks). No published figure is quoted. The lab is one data set and one simulated drift.

## Related
- [[Experiment Tracking]] · [[Data Drift]] · [[ONNX]] · [[Git]] · [[Docker]] · [[FastAPI]] · [[Virtual Environments]] · [[pytest]] · [[Safetensors]] · [[GGUF]] · [[Hugging Face Hub]] · [[Time Series Cross-Validation]] · [[Cross-Validation]] · [[Overfitting]] · [[Iris Classifier Service]] · [[Model Monitoring Lab]]
- [[AB Testing|A/B Testing]]
## Sources
- My own lab, [[Model Monitoring Lab]] (code, outputs and versions there), and the toy above.
- Sculley et al. (2015), the Google MLOps guidance and the use of the word MLOps are named from memory; not read, dates not checked.
- Image: Cmbreuel, [ML Ops Venn Diagram.svg](https://commons.wikimedia.org/wiki/File:ML_Ops_Venn_Diagram.svg), CC BY-SA 4.0.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/mlops MLOps:::The habits and tools that keep a machine-learning model working after the first success: recording, versioning, testing, packaging, serving, monitoring and retraining.

#flashcards/ml/reverse #flashcards/topic/mlops Training-serving skew:::Features computed differently in training and in production, so the model sees different numbers than it learned from.

#flashcards/ml/concept #flashcards/topic/mlops Why does a model need more than ordinary software practice?
?
Its behaviour depends on code, data, settings and randomness, it can get worse with no code change because the world changes, and it fails silently (no error, just worse answers).

#flashcards/ml/concept #flashcards/topic/mlops What are the eight steps of the MLOps loop in this note?
?
Record every run; version code, data and model; test before shipping; package; serve with a rollback; watch (system, inputs, outcomes); retrain when stale, which restarts the loop; remember the model is a small part of the system.

#flashcards/ml/image #flashcards/topic/mlops ![[mlops-venn-diagram.svg]]
What does the figure say MLOps is?
?
The overlap of three fields: machine learning, DevOps and data engineering. By Cmbreuel, CC BY-SA 4.0.

#flashcards/ml/concept #flashcards/topic/mlops In the reproducibility toy, what happened when one data value was changed by 0.001?
?
The data fingerprint changed (5869d8d6b45b became 31216d9a0a32), but the model's predictions on those 50 rows stayed the same (a4c0769a124f). A prediction check alone would not have noticed the data change; a data fingerprint does.

#flashcards/ml/concept #flashcards/topic/mlops What are the three monitoring layers and when does each become available?
?
System (latency, errors): immediately. Inputs (drift): immediately, no labels needed. Outcomes (real error): only when the true answers arrive.

#flashcards/ml/concept #flashcards/topic/mlops In my lab, which signal came first, input drift or error, and by how much?
?
The input-drift alarm (Bonferroni KS rule) fired in week 4 and the real error first left its normal range in week 6: two weeks earlier. One simulated 8-week stream on California housing, so not a general rule.

#flashcards/ml/concept #flashcards/topic/mlops Why is "best of 6 runs" partly luck in my lab, and what would a proper release gate add?
?
The top three validation MAEs (0.468, 0.471, 0.475) are very close on one validation set of 1,136 rows. A gate would require the candidate to beat the current model by more than the noise, on the same fixed check set and metric. My lab did not do the noise test.

#flashcards/ml/concept #flashcards/topic/mlops What is the difference between training-serving skew and drift?
?
Skew is a bug: production computes the features differently from training. Drift is the world changing: the same features now have different values or a different link to the answer. Both make a model worse without an error message.

#flashcards/ml/concept #flashcards/topic/mlops Which facts in the history block are from memory?
?
All: Sculley et al. 2015 "Hidden Technical Debt in Machine Learning Systems", the spread of the word MLOps, and Google's MLOps maturity levels. None was checked or read.

#flashcards/ml/cloze #flashcards/topic/mlops A run record needs code version, data ==fingerprint==, settings, ==seed==, environment, scores and the model file; if any part is missing the result cannot be rebuilt.
