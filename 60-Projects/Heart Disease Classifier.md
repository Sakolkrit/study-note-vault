---
tags: [project, python, scikit-learn, classification, tabular, evaluation]
status: done
stack: Python 3.9, pandas, scikit-learn, pytest
goal: predict heart disease from 11 patient measurements, choose a model without peeking at the test rows, and report honest metrics
---

# Heart Disease Classifier

> **In one line:** a start-to-finish classic [[Machine Learning]] project on a small table: clean it, compare four models with cross-validation, score the winner once on rows it never saw, and look at the mistakes, the thresholds and the calibration. It is a learning exercise, **not** a medical tool.

## Goal
Done means five checks pass: (1) the table is inspected and its odd values are handled, (2) four models and a "do nothing" baseline are compared with [[Cross-Validation]] on the training part only, (3) the chosen model is scored once on held-out rows with a [[Confusion Matrix]], [[Precision]], [[Recall]], [[F1 Score]] and [[ROC Curve and AUC]], (4) the output chances are checked with [[Calibration]], and (5) a few tests pass.

## Result in short
- **Data:** 918 patients, 11 measurements, 508 with heart disease and 410 without (55.4% "yes"). So a model that always says "disease" is right 55.4% of the time; that is the bar to beat.
- **Comparison (5-fold cross-validation on 734 training rows):** logistic regression AUC 0.925, random forest 0.922, gradient boosting 0.914, a depth-3 decision tree 0.878. The differences between the first three are smaller than the spread between folds (about 0.03), so I cannot honestly say one of them is better. I took the highest score, logistic regression, which is also the simplest.
- **Held-out test (184 rows, scored once):** accuracy 0.875, precision 0.907, recall 0.863, F1 0.884, AUC 0.948. Of 102 patients with disease it found 88 and missed 14; it raised 9 false alarms among 82 healthy patients. Resampling the 184 rows gives a 95% range of 0.826 to 0.924 for accuracy and 0.916 to 0.975 for AUC, so treat these as "about 0.88", not as precise.
- **Thresholds:** lowering the "yes" threshold from 0.5 to 0.3 cut the missed patients from 14 to 8 (recall 0.863 to 0.922) at the price of precision 0.907 to 0.855.
- **Calibration:** the chances are roughly honest at the ends (rows predicted 0.8 to 1.0 had disease 97% of the time; 0.0 to 0.2 only 9%) and loose in the middle (9 rows predicted 0.4 to 0.6 were 33% positive; with so few rows this is weak evidence).
- **Not done:** no web service or [[Docker]] image for this one (the [[Iris Classifier Service]] shows that part), no search over settings, no test on any other hospital's data.

## What you need
Versions I used (a fresh virtual environment, Windows, Python 3.9.6):

| Tool | Version |
|---|---|
| pandas | 2.3.3 |
| numpy | 2.0.2 |
| scikit-learn | 1.6.1 |
| joblib | 1.5.3 |
| pytest | 8.4.2 |

The data file is `heart.csv`, the Kaggle "Heart Failure Prediction Dataset" (the lab notebook in this vault's saved repo names it as `fedesoriano/heart-failure-prediction`). I did not download it from Kaggle: I used the copy that sits inside the saved repo at `50-Courses/Machine Learning Specialization/source-repo/2. Advanced Algorithms/Week 4/Notebooks/libs/heart.csv`. I did not check the dataset's licence or terms, so I did not copy the file anywhere else; get your own copy from Kaggle.

## Steps
1. **Make a folder and a clean environment** ([[Virtual Environments]], [[Command Line]]). Put `heart.csv` in the folder. Then:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install pandas==2.3.3 numpy==2.0.2 scikit-learn==1.6.1 joblib==1.5.3 pytest==8.4.2
   ```

2. **Look at the table before modelling** ([[Pandas]]). Run:
   ```
   python -c "import pandas as pd; df=pd.read_csv('heart.csv'); print(df.shape, df.HeartDisease.value_counts().to_dict()); print('cholesterol 0:', (df.Cholesterol==0).sum(), '| resting BP 0:', (df.RestingBP==0).sum(), '| nulls:', df.isna().sum().sum(), '| duplicates:', df.duplicated().sum()); print(df.groupby(df.Cholesterol==0).HeartDisease.mean().round(3).to_dict())"
   ```
   I saw: `(918, 12) {1: 508, 0: 410}`, then `cholesterol 0: 172 | resting BP 0: 1 | nulls: 0 | duplicates: 0`, then `{False: 0.477, True: 0.884}`. Two things to notice. A cholesterol of 0 is not a real measurement, so 172 rows are really "not measured". And those 172 rows have disease 88.4% of the time against 47.7% for the rest, so the fact that cholesterol is missing is itself a strong clue (probably about how and where the data was collected; I did not look into why).

3. **Clean the data in one place** ([[Feature Engineering]], [[One-Hot Encoding]]). Zeros become "missing", and a new 0/1 column records which rows were missing, so the clue is kept.

   **`data.py`**
   ```python
   import pandas as pd

   CATEGORICAL = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
   NUMERIC = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]


   def load(path="heart.csv", fix_zeros=True):
       df = pd.read_csv(path)
       if fix_zeros:
           # In this file a cholesterol of 0 (172 rows) and a resting blood pressure of 0 (1 row) mean "not measured".
           df["CholMissing"] = (df["Cholesterol"] == 0).astype(int)
           df["Cholesterol"] = df["Cholesterol"].replace(0, float("nan"))
           df["RestingBP"] = df["RestingBP"].replace(0, float("nan"))
       y = df.pop("HeartDisease")
       return df, y


   def columns(fix_zeros=True):
       return CATEGORICAL, NUMERIC + (["CholMissing"] if fix_zeros else [])
   ```

4. **Compare models fairly** ([[Cross-Validation]], [[Feature Scaling]], [[Logistic Regression]], [[Decision Tree]], [[Random Forest]], [[Ensemble Methods]]). Hold out 20% of the rows as a test set and do not touch it. On the other 80%, run 5-fold cross-validation for each model, with and without the zero fix. Category columns are one-hot encoded, numbers are median-filled and standardised, and all of that happens inside each fold so nothing leaks. The last line picks the model by cross-validated AUC.

   **`train.py`**
   ```python
   import joblib
   import numpy as np
   from sklearn.compose import ColumnTransformer
   from sklearn.dummy import DummyClassifier
   from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
   from sklearn.impute import SimpleImputer
   from sklearn.linear_model import LogisticRegression
   from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
   from sklearn.pipeline import make_pipeline
   from sklearn.preprocessing import OneHotEncoder, StandardScaler
   from sklearn.tree import DecisionTreeClassifier

   from data import columns, load

   SEED = 0


   def build(model, fix_zeros=True):
       cat, num = columns(fix_zeros)
       prep = ColumnTransformer([
           ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
           ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
       ])
       return make_pipeline(prep, model)


   MODELS = {
       "baseline (always the majority class)": DummyClassifier(strategy="most_frequent"),
       "logistic regression": LogisticRegression(max_iter=1000),
       "decision tree (depth 3)": DecisionTreeClassifier(max_depth=3, random_state=SEED),
       "random forest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_split=10, random_state=SEED),
       "gradient boosting": HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=SEED),
   }

   if __name__ == "__main__":
       best = None
       for fix in (False, True):
           X, y = load(fix_zeros=fix)
           Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
           cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
           print(f"\n--- zeros treated as {'missing + flag' if fix else 'real values'} | train {len(Xtr)}, test {len(Xte)} ---")
           for name, model in MODELS.items():
               r = cross_validate(build(model, fix), Xtr, ytr, cv=cv, scoring=["accuracy", "recall", "roc_auc"])
               auc = r["test_roc_auc"].mean()
               print(f"{name:38s} accuracy {r['test_accuracy'].mean():.3f}  recall {r['test_recall'].mean():.3f}  AUC {auc:.3f} (+/- {r['test_roc_auc'].std():.3f})")
               if fix and "baseline" not in name and (best is None or auc > best[0]):
                   best = (auc, name, model)
       auc, name, model = best
       X, y = load(fix_zeros=True)
       Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
       pipe = build(model, True).fit(Xtr, ytr)
       joblib.dump(pipe, "model.joblib"); np.save("test_index.npy", Xte.index.to_numpy())
       print(f"\nchosen by cross-validated AUC on the training part only: {name} (AUC {auc:.3f}); saved model.joblib")
   ```
   Run `python train.py`. I saw the table of scores in the "Result in short" section; for the full print-out, with the zeros treated as real values the AUCs were 0.920 (logistic regression), 0.878 (tree), 0.925 (forest) and 0.915 (boosting), and the last line was `chosen by cross-validated AUC on the training part only: logistic regression (AUC 0.925); saved model.joblib`.

5. **Score the winner once, then look closely** ([[Evaluation Metrics]], [[Confusion Matrix]], [[Precision-Recall Curve]], [[Calibration]]).

   **`evaluate.py`**
   ```python
   import joblib
   import numpy as np
   from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

   from data import load

   X, y = load()
   idx = np.load("test_index.npy")                    # the held-out rows, never used for choosing or fitting
   Xte, yte = X.loc[idx], y.loc[idx]
   pipe = joblib.load("model.joblib")
   p = pipe.predict_proba(Xte)[:, 1]
   pred = (p >= 0.5).astype(int)
   tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
   print(f"test rows {len(yte)} | confusion matrix: TN {tn}  FP {fp}  FN {fn}  TP {tp}")
   print(f"accuracy {(pred == yte).mean():.3f}  precision {precision_score(yte, pred):.3f}  recall {recall_score(yte, pred):.3f}  F1 {f1_score(yte, pred):.3f}  AUC {roc_auc_score(yte, p):.3f}")
   print(f"baseline accuracy (always 'disease'): {yte.mean():.3f}")
   print(f"Brier score {brier_score_loss(yte, p):.3f} (0 is perfect; always saying 0.5 gives 0.250)")
   print("calibration, chance bucket -> rows, actual share with disease:")
   bins = np.array([0, .2, .4, .6, .8, 1.0001])
   for lo, hi in zip(bins[:-1], bins[1:]):
       m = (p >= lo) & (p < hi)
       if m.any():
           print(f"  {lo:.1f}-{min(hi, 1):.1f}: {m.sum():3d} rows, {yte[m].mean():.2f} actually had disease (mean predicted {p[m].mean():.2f})")
   print("threshold -> recall / precision / missed patients:")
   for t in (0.5, 0.4, 0.3, 0.2):
       q = (p >= t).astype(int)
       print(f"  {t:.1f}: recall {recall_score(yte, q):.3f}  precision {precision_score(yte, q):.3f}  missed {int(((q == 0) & (yte == 1)).sum())}")
   rng = np.random.default_rng(0)
   accs, aucs = [], []
   for _ in range(2000):
       i = rng.integers(0, len(yte), len(yte))
       accs.append((pred[i] == yte.to_numpy()[i]).mean()); aucs.append(roc_auc_score(yte.to_numpy()[i], p[i]))
   print(f"95% range from resampling the test rows: accuracy {np.percentile(accs, 2.5):.3f} to {np.percentile(accs, 97.5):.3f}, AUC {np.percentile(aucs, 2.5):.3f} to {np.percentile(aucs, 97.5):.3f}")
   ```
   Run `python evaluate.py`. I saw:
   ```
   test rows 184 | confusion matrix: TN 73  FP 9  FN 14  TP 88
   accuracy 0.875  precision 0.907  recall 0.863  F1 0.884  AUC 0.948
   baseline accuracy (always 'disease'): 0.554
   Brier score 0.092 (0 is perfect; always saying 0.5 gives 0.250)
   calibration, chance bucket -> rows, actual share with disease:
     0.0-0.2:  65 rows, 0.09 actually had disease (mean predicted 0.07)
     0.2-0.4:  19 rows, 0.37 actually had disease (mean predicted 0.31)
     0.4-0.6:   9 rows, 0.33 actually had disease (mean predicted 0.53)
     0.6-0.8:  26 rows, 0.88 actually had disease (mean predicted 0.71)
     0.8-1.0:  65 rows, 0.97 actually had disease (mean predicted 0.93)
   threshold -> recall / precision / missed patients:
     0.5: recall 0.863  precision 0.907  missed 14
     0.4: recall 0.873  precision 0.890  missed 13
     0.3: recall 0.922  precision 0.855  missed 8
     0.2: recall 0.941  precision 0.807  missed 6
   95% range from resampling the test rows: accuracy 0.826 to 0.924, AUC 0.916 to 0.975
   ```
   In a screening setting, missing a sick patient is usually worse than a false alarm, which is why the threshold table matters: the 0.5 cut-off is a default, not a law.

6. **Use the model on one new patient.**

   **`predict.py`**
   ```python
   import sys

   import joblib
   import pandas as pd

   patient = {"Age": 54, "Sex": "M", "ChestPainType": "ASY", "RestingBP": 140, "Cholesterol": 0, "FastingBS": 0,
              "RestingECG": "Normal", "MaxHR": 120, "ExerciseAngina": "Y", "Oldpeak": 1.5, "ST_Slope": "Flat"}
   if len(sys.argv) > 1:                       # e.g. python predict.py Age=60 Cholesterol=250
       for kv in sys.argv[1:]:
           k, v = kv.split("=")
           patient[k] = type(patient[k])(v)
   row = pd.DataFrame([patient])
   row["CholMissing"] = (row["Cholesterol"] == 0).astype(int)                     # same cleaning as data.py
   row["Cholesterol"] = row["Cholesterol"].replace(0, float("nan"))
   row["RestingBP"] = row["RestingBP"].replace(0, float("nan"))
   p = joblib.load("model.joblib").predict_proba(row)[0, 1]
   print(f"chance of heart disease according to the model: {p:.2f}")
   ```
   Run `python predict.py` for the built-in example (a 54-year-old man with asymptomatic chest pain, exercise angina and a flat ST slope; cholesterol 0 means not measured). I saw `chance of heart disease according to the model: 0.99`. With `python predict.py Cholesterol=240 ChestPainType=ATA ExerciseAngina=N ST_Slope=Up MaxHR=170` I saw `0.09`.

7. **Test it** ([[pytest]]).

   **`test_model.py`**
   ```python
   import joblib
   import numpy as np
   from data import load


   def test_zero_cholesterol_becomes_missing_with_flag():
       X, y = load()
       assert X["CholMissing"].sum() == 172 and X["Cholesterol"].isna().sum() == 172
       assert X["RestingBP"].isna().sum() == 1 and len(X) == len(y) == 918


   def test_model_beats_the_always_yes_baseline_on_held_out_rows():
       X, y = load()
       idx = np.load("test_index.npy")
       pipe = joblib.load("model.joblib")
       acc = (pipe.predict(X.loc[idx]) == y.loc[idx]).mean()
       assert acc > y.loc[idx].mean() + 0.1


   def test_probabilities_are_between_0_and_1_and_sum_to_1():
       X, _ = load()
       p = joblib.load("model.joblib").predict_proba(X.head(20))
       assert ((p >= 0) & (p <= 1)).all() and np.allclose(p.sum(1), 1)


   def test_unknown_category_does_not_crash():
       X, _ = load()
       row = X.head(1).copy(); row["ChestPainType"] = "NEW"
       assert 0 <= joblib.load("model.joblib").predict_proba(row)[0, 1] <= 1
   ```
   Run `pytest -q`. I saw `4 passed in 2.29s`.

## What went wrong, and the fix
- **Zeros that are not zeros.** 172 cholesterol values of 0 would have been treated as real, extremely low cholesterol. The fix was to mark them missing, fill them with the median inside each fold, and add a "was missing" flag. The honest result: this barely changed the scores (logistic regression AUC 0.920 to 0.925, random forest 0.925 to 0.922, all within the 0.03 fold-to-fold spread). The other columns already carry most of the signal. I kept the fix because it is the correct way to treat the column, not because it helped a lot.
- **A tempting leak that is not one here.** The cholesterol-missing flag is a strong clue (88.4% disease), but it comes from the data, not from the answer, so it is allowed. I mention it because a column that is "missing exactly when the answer is yes" would be a real leak, and the check in step 2 is how you find out.
- **Four models that are statistically tied.** I first expected a clear winner. Cross-validation showed three models within noise of each other, so the simplest one won. Another random split could pick a different one.
- **One split is a noisy score.** The test AUC (0.948) is higher than the cross-validated one (0.925), and resampling the test rows gives a wide range. Always report the range.

## What I did not run
- **A web service or [[Docker]] image** for this model, or any call over a network.
- **A search over model settings**, and any model beyond these four ([[PyTorch]] networks, XGBoost).
- **Any other dataset or hospital.** The file reportedly merges several sources (from memory of the Kaggle description, not checked here), and I did not check how patients are split across them, so the score may not hold for new patients.
- **Fairness or subgroup checks** (for example by sex or age).
- **The Kaggle download and the dataset's licence.**
- **The saved-notebook numbers from the course lab** (for example forest test accuracy 0.8913) were not reused: this project picks its model by cross-validation, so the numbers differ.

## Concepts used
- [[Supervised Learning]] · [[Classification]] · [[Logistic Regression]] · [[Decision Tree]] · [[Random Forest]] · [[Ensemble Methods]]
- [[Cross-Validation]] · [[Evaluation Metrics]] · [[Confusion Matrix]] · [[Precision]] · [[Recall]] · [[F1 Score]] · [[ROC Curve and AUC]] · [[Calibration]]
- [[One-Hot Encoding]] · [[Feature Scaling]] · [[Feature Engineering]] · [[Pandas]] · [[pytest]] · [[Virtual Environments]]

## Next ideas
- Put the model behind an API and a page, exactly as in [[Iris Classifier Service]], and keep the threshold as a setting.
- Repeat the cross-validation 10 times with different splits and see which model wins how often.
- Try the course lab's boosted trees with early stopping ([[Machine Learning Specialization Course 2 Week 4]]).
- Compare the model's chances before and after [[Calibration]] with a calibration method.
- Find a second heart-disease dataset and test whether the score holds.

## Flashcards
#flashcards/projects

#flashcards/projects/concept #flashcards/topic/project-classic-ml What were the five "done" checks of the Heart Disease Classifier project?::(1) the table is inspected and odd values handled, (2) four models and a do-nothing baseline compared with cross-validation on the training part only, (3) the chosen model scored once on held-out rows with a confusion matrix, precision, recall, F1 and ROC/AUC, (4) the output chances checked with calibration, (5) a few tests pass.

#flashcards/projects/concept #flashcards/topic/project-classic-ml What was the data and what baseline had to be beaten?::918 patients with 11 measurements, 508 with heart disease and 410 without. A model that always says "disease" is right about 55% of the time (the note writes 55.4%, and 508/918 is 55.3%). That is the bar to beat.

#flashcards/projects/example #flashcards/topic/project-classic-ml How did the four models score in 5-fold cross-validation on the 734 training rows?::AUC: logistic regression 0.925, random forest 0.922, gradient boosting 0.914, depth-3 decision tree 0.878. The first three differ by less than the fold-to-fold spread (about 0.03), so no honest winner. Logistic regression was chosen because it scored highest and is the simplest.

#flashcards/projects/example #flashcards/topic/project-classic-ml What did the held-out test (184 rows, scored once) show?::Accuracy 0.875, precision 0.907, recall 0.863, F1 0.884, AUC 0.948. Of 102 patients with disease it found 88 and missed 14, and raised 9 false alarms among 82 healthy patients. Resampling gives 95% ranges of 0.826 to 0.924 for accuracy and 0.916 to 0.975 for AUC, so read it as "about 0.88".

#flashcards/projects/example #flashcards/topic/project-classic-ml What happened when the "yes" threshold dropped from 0.5 to 0.3?::Missed patients fell from 14 to 8 (recall 0.863 to 0.922) at the price of precision 0.907 to 0.855.

#flashcards/projects/concept #flashcards/topic/project-classic-ml How well calibrated were the chances?::Roughly honest at the ends (rows predicted 0.8 to 1.0 had disease 97% of the time, 0.0 to 0.2 only 9%) and loose in the middle (9 rows predicted 0.4 to 0.6 were 33% positive, weak evidence with so few rows).

#flashcards/projects/concept #flashcards/topic/project-classic-ml What was wrong with the cholesterol column?::172 values of 0 would be treated as real, extremely low cholesterol. The fix: mark them missing, fill with the median inside each fold, and add a "was missing" flag. It barely changed the scores (logistic regression AUC 0.920 to 0.925, within the 0.03 spread). It was kept because it is the correct treatment, not because it helped.

#flashcards/projects/concept #flashcards/topic/project-classic-ml Why is the cholesterol-missing flag not a leak, and what would a real leak look like?::It comes from the data, not from the answer, so it is allowed (88.4% disease when missing). A real leak would be a column that is missing exactly when the answer is yes. The check in step 2 is how you find out.

#flashcards/projects/example #flashcards/topic/project-classic-ml Why does the test AUC (0.948) differ from the cross-validated 0.925, and what should you do about it?::One split is a noisy score and resampling the test rows gives a wide range. Always report the range.

#flashcards/projects/concept #flashcards/topic/project-classic-ml What did the Heart Disease project not do?::No web service or Docker image, no search over model settings, no other model beyond these four, no other dataset or hospital, no fairness or subgroup checks, and no Kaggle download or licence check (it used the copy inside the saved repo).

#flashcards/projects/cloze #flashcards/topic/project-classic-ml The three top models were ==statistically tied==, so the simplest one (logistic regression) won.
