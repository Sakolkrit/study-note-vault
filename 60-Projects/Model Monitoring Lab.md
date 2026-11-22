---
tags: [project, MLOps, MLflow, ONNX, data-drift, monitoring, scikit-learn]
status: done
stack: Python 3.9.6, mlflow 3.1.4, scikit-learn 1.6.1, onnx 1.19.1, onnxruntime 1.19.2, skl2onnx 1.20.0
goal: track runs and register a model with MLflow, export it to ONNX, and watch a simulated production stream for data drift
---

# Model Monitoring Lab

> **In one line:** take one small model through the life of a model in production: **track** six training runs and **register** the best (MLflow), **export** it to a portable file and prove it gives the same answers (ONNX), then **watch** eight simulated weeks of incoming data and see whether a drift alarm warns you before the error does.

## Goal
Done means: three scripts run in a fresh environment (`train_track.py`, `export_onnx.py`, `monitor.py`, plus a helper `common.py`), and the claims in [[MLOps]], [[Experiment Tracking]], [[ONNX]] and [[Data Drift]] come from numbers I measured here.

## Result in short
- **Data:** California housing (20,640 rows, scikit-learn's copy). The model sees 6 features (income, house age, rooms, bedrooms, population, occupancy) and **not** latitude or longitude. It trains on **northern** rows (latitude 37 or more): 5,301 for training, 1,136 for validation, 1,136 kept as a "live north" reference. The **southern** rows (13,067) are what production later sends.
- **Tracking:** six runs logged to MLflow. Validation MAE (error in units of 100,000 dollars): ridge 0.5837; forests 0.4749, **0.4683**, 0.4714; gradient boosting 0.4840, 0.4826. The best run (forest, 100 trees, depth 12) was registered as `housing-model` version 1 with the alias `champion`. The top three are within 0.007 of each other, so "best" is partly luck on one validation set.
- **ONNX:** the champion exported to an 8.26 MB file. Worst difference from scikit-learn on 1,136 rows: $1.74\times10^{-6}$. Speed in two runs: 49.5 against 7.3 ms, and 95.6 against 35.1 ms (so about 6.8 and 2.7 times faster, on a busy laptop: rough).
- **Monitoring:** eight weekly batches of 600 rows, with the share of southern rows rising 0, 0, 10, 25, 50, 75, 100, 100%.
  - Real error (MAE): 0.449 in week 1 up to 0.524 in week 7 (about 17% worse).
  - Rule A (two or more of six features with PSI above 0.2) **never fired**.
  - Rule B (any feature with a KS p-value below 0.05/6) fired from **week 4**, two weeks before the real error left its normal range (week 6).
  - On 200 drift-free batches: A and B fired 0 times; the plain "any p below 0.05" rule fired 28 times (14%).
- **Honest limits:** one data set; one simulated drift that I designed; rule B was chosen after I saw the first results; the model is a small random forest; nothing was served live.

## What you need
- **Python 3.9.6** and a **separate virtual environment** (the vault's other projects use older library versions; I kept these apart). The install line below names no versions, so a later install may give newer ones. Versions I got: `mlflow` 3.1.4, `scikit-learn` 1.6.1, `onnx` 1.19.1, `onnxruntime` 1.19.2, `skl2onnx` 1.20.0, `numpy` 2.0.2, `pandas` 2.3.3, `scipy` 1.13.1 (read with `pip list`).
- Internet for the first run (scikit-learn downloads the housing data and caches it in `data/`).

## Steps
1. **Create the environment.** See [[Virtual Environments]]. On Windows (Git Bash) in an empty folder:

```bash
python -m venv venv
./venv/Scripts/python.exe -m pip install --upgrade pip
./venv/Scripts/python.exe -m pip install --only-binary :all: mlflow scikit-learn onnx onnxruntime skl2onnx pandas scipy
```

   The upgrade of `pip` and the `--only-binary` flag matter: see "What went wrong". You should see no error and `mlflow` 3.1.4 in `pip list`.
2. **Shared data set-up.** Save as `common.py`. It loads the data, splits it into north (training 70%, validation 15%, reference 15%) and south, and names the six features. Latitude and longitude are kept out of the features on purpose, so *where* a house is stays a hidden cause of the shift.

```python
# file: common.py
"""Shared data set-up. Train on 'north' (Latitude >= 37, Bay Area and north). Production later sends 'south' rows too.
Latitude and Longitude are NOT given to the model or the monitor, so the region is a hidden cause of the shift."""
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"]


def load():
    d = fetch_california_housing(data_home="data", as_frame=True).frame
    north, south = d[d.Latitude >= 37].reset_index(drop=True), d[d.Latitude < 37].reset_index(drop=True)
    tr, rest = train_test_split(north, test_size=0.3, random_state=0)
    va, ref = train_test_split(rest, test_size=0.5, random_state=0)      # va: choose the model; ref: held-out 'live' north rows
    return tr, va, ref, south
```

3. **Track six runs and register the best.** Save as `train_track.py` and run `./venv/Scripts/python.exe train_track.py`. Each run is one `mlflow.start_run`, logging the settings, the validation MAE and the model. `mlflow.search_runs` then returns all runs sorted by MAE, the best one is registered as `housing-model`, and the alias `champion` is set on that version. Everything is stored in `mlflow.db` (SQLite) and the folder `mlruns`.

```python
# file: train_track.py
"""Step 2: six training runs, every run logged to MLflow; then the best run is registered and given the alias 'champion'."""
import mlflow, mlflow.sklearn, numpy as np
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from common import load, FEATURES

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("housing-north")
tr, va, ref, south = load()
Xtr, ytr, Xva, yva = tr[FEATURES], tr.MedHouseVal, va[FEATURES], va.MedHouseVal
print(f"train {len(tr)} | validation {len(va)} | reference {len(ref)} | south (not seen) {len(south)}")

configs = [("ridge", Ridge, {"alpha": 1.0}),
           ("forest", RandomForestRegressor, {"n_estimators": 50, "max_depth": 8, "random_state": 0}),
           ("forest", RandomForestRegressor, {"n_estimators": 100, "max_depth": 12, "random_state": 0}),
           ("forest", RandomForestRegressor, {"n_estimators": 100, "max_depth": None, "random_state": 0}),
           ("boosting", GradientBoostingRegressor, {"n_estimators": 100, "max_depth": 3, "random_state": 0}),
           ("boosting", GradientBoostingRegressor, {"n_estimators": 300, "max_depth": 3, "random_state": 0})]
for kind, cls, params in configs:
    with mlflow.start_run(run_name=f"{kind}-{params.get('n_estimators', '')}-{params.get('max_depth', '')}"):
        m = cls(**params).fit(Xtr, ytr)
        mae = mean_absolute_error(yva, m.predict(Xva))
        mlflow.log_params({"model": kind, **{k: str(v) for k, v in params.items()}})
        mlflow.log_metric("mae_val", mae)
        mlflow.sklearn.log_model(m, "model", signature=infer_signature(Xva, m.predict(Xva)))
        print(f"{kind:9s} {params} -> validation MAE {mae:.4f}", flush=True)

runs = mlflow.search_runs(order_by=["metrics.mae_val ASC"])
print("\nbest three runs by MLflow search:")
print(runs[["tags.mlflow.runName", "params.model", "metrics.mae_val"]].head(3).to_string(index=False))
best = runs.iloc[0]
mv = mlflow.register_model(f"runs:/{best.run_id}/model", "housing-model")
mlflow.MlflowClient().set_registered_model_alias("housing-model", "champion", mv.version)
print(f"\nregistered housing-model version {mv.version} (alias 'champion') from run {best['tags.mlflow.runName']}")
```

   Output (the lines about database set-up and library warnings are removed):

```text
train 5301 | validation 1136 | reference 1136 | south (not seen) 13067
ridge     {'alpha': 1.0} -> validation MAE 0.5837
forest    {'n_estimators': 50, 'max_depth': 8, 'random_state': 0} -> validation MAE 0.4749
forest    {'n_estimators': 100, 'max_depth': 12, 'random_state': 0} -> validation MAE 0.4683
forest    {'n_estimators': 100, 'max_depth': None, 'random_state': 0} -> validation MAE 0.4714
boosting  {'n_estimators': 100, 'max_depth': 3, 'random_state': 0} -> validation MAE 0.4840
boosting  {'n_estimators': 300, 'max_depth': 3, 'random_state': 0} -> validation MAE 0.4826
Successfully registered model 'housing-model'.
Created version '1' of model 'housing-model'.
best three runs by MLflow search:
tags.mlflow.runName params.model  metrics.mae_val
      forest-100-12       forest         0.468252
    forest-100-None       forest         0.471433
        forest-50-8       forest         0.474917
registered housing-model version 1 (alias 'champion') from run forest-100-12
```

   What to look for: six printed MAEs, then the three best runs, then the registry line. The three forests are within 0.007 of each other. The folder `mlruns` was about 61 MB after these runs (the six stored models; my measurement), and `mlflow.db` 260 KB.
4. **Export to ONNX and check parity.** Save as `export_onnx.py`. It loads the champion **by name from the registry** (`models:/housing-model@champion`), converts it with `skl2onnx`, saves `housing.onnx`, loads it in ONNX Runtime without scikit-learn, compares the predictions on the 1,136 reference rows, and times both engines (7 repeats, median).

```python
# file: export_onnx.py
"""Step 3: load the champion from the registry, export to ONNX, check the two give the same numbers, compare speed."""
import time, os, numpy as np, mlflow, mlflow.sklearn, onnxruntime as ort
from skl2onnx import to_onnx
from common import load, FEATURES

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/housing-model@champion")
tr, va, ref, south = load()
X = ref[FEATURES].to_numpy().astype(np.float32)
onx = to_onnx(model, X[:1], target_opset={"": 17, "ai.onnx.ml": 3})      # newest opset (22) is too new for this onnxruntime
open("housing.onnx", "wb").write(onx.SerializeToString())
print(f"model type {type(model).__name__} | ONNX file {os.path.getsize('housing.onnx')/1e6:.2f} MB")
sess = ort.InferenceSession("housing.onnx", providers=["CPUExecutionProvider"])
a = model.predict(X); b = sess.run(None, {sess.get_inputs()[0].name: X})[0].ravel()
print(f"rows {len(X)} | max |sklearn - onnx| = {np.abs(a - b).max():.2e} | mean |diff| = {np.abs(a - b).mean():.2e}")
for name, fn in (("sklearn", lambda: model.predict(X)), ("onnxruntime", lambda: sess.run(None, {sess.get_inputs()[0].name: X}))):
    ts = []
    for _ in range(7):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    print(f"{name:12s} predict {len(X)} rows: median {1000*np.median(ts):.1f} ms (min {1000*min(ts):.1f}, max {1000*max(ts):.1f}, 7 runs)")
```

```text
model type RandomForestRegressor | ONNX file 8.26 MB
rows 1136 | max |sklearn - onnx| = 1.74e-06 | mean |diff| = 1.23e-07
sklearn      predict 1136 rows: median 95.6 ms (min 63.4, max 110.8, 7 runs)
onnxruntime  predict 1136 rows: median 35.1 ms (min 14.8, max 64.2, 7 runs)
```

   What to look for: the maximum difference should be tiny compared with the target (the prices are around 2, in units of 100,000 dollars). The times differ between runs by a lot (my first run measured 49.5 and 7.3 ms), so only conclude "ONNX Runtime was faster", not a ratio.
5. **Simulate eight weeks and watch.** Save as `monitor.py`. For each week it builds a batch of 600 rows with the given share of southern rows, computes **PSI** and a **KS test** for each of the 6 features against the training rows, applies the two rules, and (because this is a simulation) also computes the **real error**, which in production would only be known later. At the end it replays 200 batches with **no drift** to count false alarms and to measure the normal spread of the error.

```python
# file: monitor.py
"""Step 4: watch a stream of 'weekly' batches. The model never sees labels at serving time, so the monitor looks at the inputs only
(PSI and a KS test per feature). Labels arrive later in this toy, so we also compute the real error to see whether the alarm was right."""
import numpy as np, mlflow, mlflow.sklearn
from scipy.stats import ks_2samp
from sklearn.metrics import mean_absolute_error
from common import load, FEATURES

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/housing-model@champion")
tr, va, ref, south = load()


def psi(expected, actual, bins=10):                          # population stability index; bin edges are the training deciles
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected); a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def check(batch):
    p = {f: psi(tr[f].to_numpy(), batch[f].to_numpy()) for f in FEATURES}
    ks = {f: ks_2samp(tr[f], batch[f]).pvalue for f in FEATURES}
    flagged = [f for f in FEATURES if p[f] > 0.2]
    return p, ks, flagged


N = 600

print(f"batch size {N}")
print("rule A: 2 or more of 6 features with PSI > 0.2")
print("rule B: any feature with KS p < 0.05/6 (Bonferroni, 6 tests)\n")
print(f"{'week':>4s} {'south share':>11s} {'#PSI>0.2':>8s} {'worst feature (PSI)':>24s} {'min KS p':>9s} {'A':>4s} {'B':>4s} {'MAE':>6s}")
for week, share in enumerate([0, 0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.0], start=1):
    ns = int(round(N * share))
    batch = __import__("pandas").concat([ref.sample(N - ns, random_state=week), south.sample(ns, random_state=week)]) if ns else ref.sample(N, random_state=week)
    p, ks, flagged = check(batch)
    worst = max(p, key=p.get)
    mae = mean_absolute_error(batch.MedHouseVal, model.predict(batch[FEATURES]))
    print(f"{week:4d} {share:11.2f} {len(flagged):8d} {worst:>16s} ({p[worst]:.2f}) {min(ks.values()):9.1e} {'YES' if len(flagged) >= 2 else 'no':>4s} {'YES' if min(ks.values()) < 0.05 / 6 else 'no':>4s} {mae:6.3f}")

quiet = [(check(b), mean_absolute_error(b.MedHouseVal, model.predict(b[FEATURES]))) for b in (ref.sample(N, random_state=100 + s) for s in range(200))]
print("\nfalse alarms over 200 batches drawn from the SAME north rows:")
print(f"  rule A (PSI):        {sum(len(c[0][2]) >= 2 for c in quiet)} of 200")
print(f"  rule B (KS, Bonf.):  {sum(min(c[0][1].values()) < 0.05 / 6 for c in quiet)} of 200")
print(f"  plain 'any p < 0.05': {sum(min(c[0][1].values()) < 0.05 for c in quiet)} of 200")
maes = np.array([c[1] for c in quiet]); print(f"  MAE on those quiet batches: mean {maes.mean():.3f}, std {maes.std():.3f}, 95th percentile {np.quantile(maes, .95):.3f}")
```

```text
batch size 600
rule A: 2 or more of 6 features with PSI > 0.2
rule B: any feature with KS p < 0.05/6 (Bonferroni, 6 tests)
week south share #PSI>0.2      worst feature (PSI)  min KS p    A    B    MAE
   1        0.00        0        AveBedrms (0.02)   1.1e-01   no   no  0.449
   2        0.00        0       Population (0.01)   7.5e-02   no   no  0.449
   3        0.10        0         HouseAge (0.04)   8.6e-02   no   no  0.466
   4        0.25        0         AveRooms (0.03)   1.1e-03   no  YES  0.466
   5        0.50        0         AveOccup (0.07)   2.7e-06   no  YES  0.468
   6        0.75        0         AveOccup (0.09)   3.7e-09   no  YES  0.485
   7        1.00        1         AveOccup (0.21)   1.1e-16   no  YES  0.524
   8        1.00        1         AveOccup (0.27)   1.2e-19   no  YES  0.501
false alarms over 200 batches drawn from the SAME north rows:
  rule A (PSI):        0 of 200
  rule B (KS, Bonf.):  0 of 200
  plain 'any p < 0.05': 28 of 200
  MAE on those quiet batches: mean 0.459, std 0.012, 95th percentile 0.478
```

6. **Read the table.** Look at the week where rule B turns "YES" (week 4), at the week where the error first passes the 95th percentile of the drift-free batches (0.478: week 6, at 0.485), and at the "A" column that stays "no" all the way. The plain rule's 28 false alarms in 200 is the reason I did not use it.
7. **Not run, but the next obvious step:** open the tracking page with `mlflow ui` and look at the six runs in a browser (see below).

## What went wrong, and the fix
- **`pip install mlflow` failed:** "Could not build wheels for greenlet". A fresh virtual environment starts with an old `pip` (21.1.3) that did not find a ready-made wheel. Fix: upgrade `pip` first, then install with `--only-binary :all:` so that nothing is built from source.
- **`numpy` 2 and my first scikit-learn pin:** I first asked for scikit-learn version 1.1.2 (the version of my other projects) next to the new `numpy` 2.0.2 that the installer chose. These two are not meant to go together (1.1.2 is older than NumPy 2; from memory, I did not test how it fails). I upgraded to scikit-learn 1.6.1 before running anything.
- **ONNX opset error:** `onnxruntime` 1.19.2 refused my first export: "Opset 22 is under development and support for this is limited". Fix: ask `to_onnx` for `target_opset={"": 17, "ai.onnx.ml": 3}`. This line is in `export_onnx.py` above.
- **The first alarm rule never fired.** My first `monitor.py` used only rule A (PSI, two features above 0.2) and it stayed silent for all eight weeks, even with 100% southern rows. I added rule B (KS p-value with the Bonferroni correction) and the false-alarm counts. This was a **change made after seeing the result**, so rule B's good showing is not a clean test. The correction itself is a standard method, and I did not tune its threshold.
- **A quoting slip in my own script:** while adding rule B, a broken line in `monitor.py` ended with a `SyntaxError` ("EOL while scanning string literal"). I fixed the line by hand. It was a typing mistake, not a library problem.
- **Timing was not stable:** the first run measured 49.5 ms against 7.3 ms and the second 95.6 against 35.1 ms. The laptop was busy. I report both and no single ratio.

## What I did not run
- **`mlflow ui`** (the web page for browsing runs) and **`mlflow models serve`**: not started.
- **Serving the model** through [[FastAPI]] or [[Docker]], and any live traffic.
- **A real retraining step** after the alarm, an automated pipeline, or tests in a build system.
- **Delayed labels:** the simulation knows the true prices immediately. In production they would arrive later, and I did not simulate that delay.
- **More than one drift story:** gradual sensor change, sudden break, label change (concept drift) and seasonal change were not tried; I tested only one simulated stream with mixed regions.
- **A noise test for the "best" run** (several validation splits), the PSI rule with other thresholds or bin counts, the **classifier-based drift test**, and non-numeric features.
- **Re-run check:** the code of steps 2 to 5 was extracted from this note into a fresh folder and run again in the same virtual environment (with a copy of the cached data). Steps 3 and 5 printed the same text as above. Step 4 gave the same file size and difference, but new timings: the fastest of 7 repeats was 30.9 ms for scikit-learn and 3.5 ms for ONNX Runtime, so ONNX Runtime was faster in all three timing runs, by an unstable factor.

## Concepts used
- [[MLOps]] · [[Experiment Tracking]] · [[ONNX]] · [[Data Drift]] · [[Random Forest]] · [[Gradient Boosting]] · [[Linear Regression]] · [[Cross-Validation]] · [[Overfitting]] · [[Virtual Environments]] · [[Docker]] · [[FastAPI]] · [[Iris Classifier Service]]

## Next ideas
- Serve the champion with [[FastAPI]] in a [[Docker]] container (the way [[Iris Classifier Service]] does) and add the drift check to the service.
- Simulate delayed labels and measure how many weeks the error check lags behind.
- Try other drift stories and compare PSI, KS and a classifier-based test.
- Repeat "best of six" over several validation splits to see how often the winner changes.

## Flashcards
#flashcards/projects

#flashcards/projects/concept #flashcards/topic/project-mlops What did the Model Monitoring Lab build and in what order?
?
Six training runs tracked with MLflow and the best registered as housing-model (alias champion); an ONNX export checked against scikit-learn; and an 8-week simulated stream monitored with PSI and KS tests, to see whether the input alarm came before the real error.

#flashcards/projects/concept #flashcards/topic/project-mlops How was drift simulated in the lab?
?
The model was trained on northern California houses (latitude 37 or more) using 6 features without location. Weekly batches of 600 rows mixed in southern houses: 0, 0, 10, 25, 50, 75, 100, 100%. Latitude and longitude were hidden from the model and the monitor.

#flashcards/projects/concept #flashcards/topic/project-mlops What was the key result of the monitoring step?
?
Rule B (any feature with KS p below 0.05/6) alarmed in week 4; the real error first left its normal range in week 6 (0.485 against a drift-free 95th percentile of 0.478). Rule A (2 or more features with PSI above 0.2) never fired. One simulated stream, and rule B was chosen after seeing the first results.

#flashcards/projects/concept #flashcards/topic/project-mlops What did the ONNX step show?
?
The champion exported to an 8.26 MB file; the worst difference from scikit-learn on 1,136 rows was 1.74e-06. ONNX Runtime was faster in both timing runs (49.5 against 7.3 ms and 95.6 against 35.1 ms), but the laptop was busy so only a rough range.

#flashcards/projects/concept #flashcards/topic/project-mlops Which three setup problems occurred, and the fixes?
?
(1) The greenlet wheel build failed with old pip: upgrade pip and install with --only-binary. (2) numpy 2 does not fit scikit-learn 1.1.2: use a newer scikit-learn (1.6.1). (3) onnxruntime refused opset 22: ask the converter for target_opset {"": 17, "ai.onnx.ml": 3}.

#flashcards/projects/concept #flashcards/topic/project-mlops Name four things the lab did not run.
?
mlflow ui and model serving; serving through FastAPI or Docker; retraining or an automated pipeline; delayed labels; other drift stories; a noise test for the best run; a classifier-based drift test.

#flashcards/projects/cloze #flashcards/topic/project-mlops The first drift rule never fired, so I added a second rule after seeing the result: that ==change after seeing the result== means its good showing is not a clean ==test==.
