---
tags: [experiment-tracking, MLflow, MLOps, model-registry, reproducibility, ML]
status: growing
level: 4
tier: 2
---

# Experiment Tracking

Experiment tracking means **writing down, automatically, every try you make when building a model**: what you changed (the settings), what you got (the scores) and what you made (the model file). Later you can list the tries, compare them, and rebuild the best one.

## 🟢 1. Simple (age 12)
You are trying to bake the best cookie. Each time you change something (more sugar, a hotter oven) you write a line in a notebook: what you changed, and how the cookie tasted. After 20 tries you can look at the notebook, see which try was best and repeat it exactly. **Experiment tracking is that notebook, filled in by the computer, so you never forget a try or mix up the notes.**

**Example:** you try four values of one setting. Without notes you remember "the second one was good" and cannot say which value it was. With tracking you have a table: value 0.1 gave error 44.6, value 1.0 gave 44.9, and so on.

## 🔵 2. How it works
1. **Start a run:** one run is one try (one training of one model with one set of settings). Runs are grouped into an **experiment** (a project, such as "house prices").
2. **Log the parameters:** the settings you chose (learning rate, tree depth, which data).
3. **Train** the model as usual.
4. **Log the metrics:** the scores you got (error on a validation set). You can log a metric many times to get a curve (one value per training step).
5. **Log the artifacts:** files made by the run: the saved model, a plot, a list of predictions.
6. **End the run.** Everything is stored (in a file, a small database or a server).
7. **Compare runs:** list them as a table, sort by a metric, or look at them in a web page. Pick the best.
8. **Register the winner:** give the chosen model a name and a version number in a **model registry**, and a label (an *alias*, such as "champion") that points at the version that is in use. Other code asks for "the champion" by name instead of by a file path, and you can move the label to a new version without touching that code.
9. The tool I used is **MLflow**. Others do the same job (Weights & Biases, TensorBoard and more; from memory, I did not try them).

## 🟠 3. Technical detail
- **What gets stored.** In MLflow, a run holds parameters (text), metrics (numbers, with step and time), tags, artifacts (files) and the run's start and end. The default store is a folder of files (`mlruns`); a database such as SQLite (`sqlite:///mlflow.db`) is needed for the model registry in recent versions, which is what I used.
- **Automatic logging** (`mlflow.autolog()`) can record parameters and metrics of common libraries without extra lines (not tried here).
- **What tracking does *not* do:** it records what you ran, not what is *good*. It cannot tell you whether 44.65 against 44.92 is a real difference (see the caution below).
- **Code: the smallest example.** Four runs of ridge regression on a small built-in data set (diabetes, 442 rows), one setting changed, then a table of runs sorted by error. Needs `mlflow` (installed in a separate virtual environment, see [[Model Monitoring Lab]]).

```python
import mlflow
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

mlflow.set_tracking_uri("sqlite:///toy_mlflow.db")            # one small file holds every run
mlflow.set_experiment("diabetes-ridge")
X, y = load_diabetes(return_X_y=True)
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, random_state=0)

for alpha in (0.001, 0.1, 1.0, 10.0):
    with mlflow.start_run(run_name=f"ridge-alpha-{alpha}"):    # one run = one try
        model = Ridge(alpha=alpha).fit(Xtr, ytr)
        mlflow.log_param("alpha", alpha)                       # what I changed
        mlflow.log_metric("mae_val", mean_absolute_error(yva, model.predict(Xva)))   # how it scored

runs = mlflow.search_runs(order_by=["metrics.mae_val ASC"])    # all runs, best first, as a table
print(runs[["tags.mlflow.runName", "params.alpha", "metrics.mae_val"]].to_string(index=False))
```

Output (run on my machine, in a fresh folder; the lines about the database set-up are removed):

```text
tags.mlflow.runName params.alpha  metrics.mae_val
    ridge-alpha-0.1          0.1        44.646100
    ridge-alpha-1.0          1.0        44.922683
  ridge-alpha-0.001        0.001        45.151748
   ridge-alpha-10.0         10.0        53.436244
```

- **Reading it:** the four runs are listed best first. Error is lowest at `alpha=0.1` (44.65) and almost the same at `alpha=1.0` (44.92) and `0.001` (45.15), so the first three are hardly different; only `alpha=10` (53.44) is clearly worse. I used one split of 111 validation rows, so I would not claim that 0.1 is better than 1.0.
- **The real use** is in [[Model Monitoring Lab]]: six runs (ridge, three random forests, two gradient-boosting models) on a housing data set, each logged with its parameters, validation error and model file; the registry then holds the best one as `housing-model`, version 1, alias `champion`, loaded later by `models:/housing-model@champion`. The best run was a random forest of 100 trees with depth 12 (validation error 0.4683), against 0.4714 and 0.4749 for the next two. Those gaps are small (see level 4).
- **Cost:** tracking is not free. After those six runs the `mlruns` folder (every run's model file) was about 61 MB and the database 260 KB (my measurement with `du`).

## 🔴 4. Deep / formal
There is no formula, but there are rules about what makes a record *useful*:

- **A run must be rebuildable.** Log the code version ([[Git]] commit), the data (a fingerprint or version), all settings including random seeds, and the library versions. Otherwise the table is a list of numbers nobody can reproduce ([[MLOps]]).
- **Choosing the best of many runs overfits the validation set.** The more runs you compare on one validation set, the more likely the winner is partly lucky ([[Overfitting]], [[Cross-Validation]]). In my lab the top three (0.468, 0.471, 0.475) are all within 0.007 of each other on 1,136 rows. Fix: keep a separate test set that is used once at the end; or compare with several splits and report spread.
- **Metric hygiene:** log the *same* metric for every run, and say which data it was measured on.
- **Registry as the contract between building and serving.** A name plus an alias lets serving code say "load the champion" while the team promotes new versions. Rolling back is moving the alias to the old version. An old way in MLflow used fixed *stages* ("Staging", "Production"); newer versions favour aliases (from memory, I did not check the release notes).
- **Privacy and size:** artifacts can contain data samples and large models; decide what is allowed to be stored and for how long.

**What came before, and how this improved on it:**
- **Before:** results in notebook cells, text files, spreadsheets and file names like `model_final_v2.pkl`. Tools such as TensorBoard (2015) and later MLflow (open-sourced 2018 by Databricks) and Weights & Biases started to automate this (all from memory, not checked).
- **Its limit:** you could not list, sort or rebuild old tries; you forgot which settings made which score.
- **What changed:** each try writes its own record automatically, in one queryable store, plus a registry for the chosen model.
- **How much better:** measured here: the six lab runs came back as a ranked table from one call (`mlflow.search_runs`), and the registered model loaded by name in a separate script. The price was 61 MB of stored models for six runs. I did not measure time saved, so I do not quote one.

## Related
- [[MLOps]] · [[Data Drift]] · [[ONNX]] · [[Git]] · [[Virtual Environments]] · [[Cross-Validation]] · [[Overfitting]] · [[Gradient Boosting]] · [[Random Forest]] · [[Linear Regression]] · [[Regularization]] · [[Model Monitoring Lab]]

## Sources
- My own toy and lab ([[Model Monitoring Lab]]): MLflow 3.1.4 with a SQLite store. The MLflow release years, TensorBoard, Weights & Biases and the "stages replaced by aliases" detail are from memory and were not checked. Not read: the MLflow documentation (I only used the calls shown in the code).
- No image: I found no free, on-topic image of experiment tracking on Wikimedia Commons, and did not use an unrelated one.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/mlops Experiment tracking:::Automatically recording every try of building a model: its settings (parameters), scores (metrics) and files (artifacts, such as the model), so tries can be listed, compared and rebuilt.

#flashcards/ml/reverse #flashcards/topic/mlops Run and experiment (in MLflow):::A run is one try (one training with one set of settings). An experiment is a group of runs belonging to one project.

#flashcards/ml/reverse #flashcards/topic/mlops Parameter, metric, artifact:::Parameter: a setting you chose. Metric: a number you measured (a score). Artifact: a file the run produced (a model, a plot).

#flashcards/ml/concept #flashcards/topic/mlops What is a model registry and what is an alias such as "champion" for?
?
A registry gives a chosen model a name and version numbers. An alias is a label pointing at one version. Serving code asks for the model by name and alias (`models:/housing-model@champion`), so you can promote a new version or roll back by moving the alias, without changing that code.

#flashcards/ml/concept #flashcards/topic/mlops What does experiment tracking not tell you?
?
Whether a difference between two runs is real. It records what you ran, not what is good. In my toy 44.65 against 44.92 on 111 validation rows is not a claim I would make.

#flashcards/ml/concept #flashcards/topic/mlops What were the four runs of the toy tracking example and their validation errors?
?
Ridge on the diabetes data with alpha 0.1: 44.65; alpha 1.0: 44.92; alpha 0.001: 45.15; alpha 10: 53.44. One split of 111 validation rows. Only alpha 10 is clearly worse.

#flashcards/ml/concept #flashcards/topic/mlops What are the six runs and the winner in the Model Monitoring Lab?
?
Ridge, three random forests, two gradient-boosting models on a housing data set. The winner was the random forest with 100 trees and depth 12 (validation MAE 0.4683), against 0.4714 and 0.4749 for the next two, which is a very small gap.

#flashcards/ml/concept #flashcards/topic/mlops Why is picking the best of many tracked runs risky?
?
The more runs you compare on one validation set, the more likely the winner is partly lucky (overfitting the validation set). Keep a separate test set used once, or compare over several splits and report the spread.

#flashcards/ml/concept #flashcards/topic/mlops What does a run need logged so it can be rebuilt?
?
The code version, a data fingerprint or version, all settings including random seeds, and the library versions, plus the scores and the model file.

#flashcards/ml/concept #flashcards/topic/mlops How much space did tracking six runs take in my lab?
?
About 61 MB in the `mlruns` folder (a stored model per run) and 260 KB for the SQLite database. My measurement with `du`.

#flashcards/ml/cloze #flashcards/topic/mlops Rolling back a model is just moving the ==alias== (for example "champion") back to the old ==version==.
