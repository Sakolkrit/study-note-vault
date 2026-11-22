---
tags: [project, python, time-series, forecasting, statsmodels, scikit-learn, pytorch, arima, lstm, cpu]
status: done
stack: Python 3.9, statsmodels (in a separate virtual environment), scikit-learn, PyTorch (CPU), NumPy
goal: forecast two monthly series with nine methods (from the naive guess to an LSTM), score them fairly on forward-moving tests, and find out which methods earn their complexity and where each one breaks
---

# Forecasting Shoot-out

> **In one line:** forecast the next 12 months of two classic series (airline passengers and atmospheric CO2) with nine methods, from "same as last year" to ARIMA, boosting and an LSTM, test all of them the same honest way, and see who wins and who fails.

## Goal
Done means five checks pass: (1) the data and a separate environment for `statsmodels` are set up and checked, (2) nine forecasters run on the same history with the same horizon, (3) every forecast is scored on **seven forward-moving origins** with MAE, RMSE and MASE ([[Time Series Cross-Validation]], [[Forecast Metrics]]), (4) the methods that depend on random numbers use three seeds and I report the spread, and (5) I look at where the methods **fail**, not only at the ranking.

## Result in short
- **Data:** the airline passengers series (144 months, 1949 to 1960, multiplicative, with a strong trend) and monthly Mauna Loa CO2 (526 months, 1958 to 2001, additive, with a curved trend). Each method forecasts the next 12 months from seven origins spaced 6 months apart. The last origin ends at the end of the data.
- **Score used in the tables:** MASE, the average miss divided by the seasonal naive forecast's average miss on the training data, so 1.0 means "as good as last year's pattern on the training data" and lower is better (see [[Forecast Metrics]]). It is a mean over the 7 origins (and over 3 seeds where marked).

**Airline passengers**

| Method | MAE | RMSE | MASE |
|---|---|---|---|
| Holt-Winters | 16.54 | 19.91 | **0.56** |
| SARIMA (airline model) | 18.38 | 21.67 | 0.62 |
| LSTM on changes (3 seeds) | 18.87 | 22.56 | 0.64 (sd 0.012) |
| random forest on changes (3 seeds) | 19.94 | 23.16 | 0.68 (sd 0.014) |
| boosting on changes | 27.02 | 31.01 | 0.92 |
| line + month effects | 32.24 | 36.42 | 1.09 |
| random forest on raw levels (3 seeds) | 34.23 | 42.11 | 1.16 (sd 0.002) |
| seasonal naive | 36.44 | 39.86 | 1.24 |
| naive | 66.60 | 83.16 | 2.26 |

**Mauna Loa CO2**

| Method | MAE | RMSE | MASE |
|---|---|---|---|
| LSTM on changes (3 seeds) | 0.40 | 0.46 | **0.32** (sd 0.024) |
| SARIMA (airline model) | 0.45 | 0.53 | 0.35 |
| Holt-Winters | 0.48 | 0.55 | 0.38 |
| random forest on changes (3 seeds) | 0.55 | 0.64 | 0.43 (sd 0.003) |
| boosting on changes | 0.59 | 0.67 | 0.46 |
| random forest on raw levels (3 seeds) | 1.29 | 1.45 | 1.01 (sd 0.002) |
| seasonal naive | 1.76 | 1.83 | 1.38 |
| naive | 2.15 | 2.58 | 1.68 |
| line + month effects | 3.24 | 3.25 | 2.52 |

- **The classic methods held their ground.** Holt-Winters was the best on the airline series (MASE 0.56) and was within 0.06 of the best on CO2 (0.38 against 0.32). The airline-model SARIMA was second on both. Both needed no tuning.
- **The LSTM won on CO2 and was a good third on airline,** with small spread between seeds. It is the only method that was best on one series. It did not beat the classic methods on the airline series (0.64 against 0.56 and 0.62).
- **Trees on raw levels cannot follow a trend.** The random forest on raw values scored 1.16 and 1.01, about as good as or worse than seasonal naive (1.24 and 1.38). The same forest trained on **changes** (growth) scored 0.68 and 0.43, which is 41% and 57% lower (my arithmetic: 0.68 over 1.16 and 0.43 over 1.01). On CO2 the forest on levels sat at about the level of its training data, and the forecast stayed almost flat.
- **A straight-line trend plus month effects** beat seasonal naive on the airline series (1.09 against 1.24) but was the **worst method** on CO2 (2.52), because the CO2 trend curves upward and a straight line cannot follow it.
- **One split can mislead:** in the [[Gradient Boosting]] note, a single split made boosting look close to Holt-Winters (RMSE 35.8 against 32.5). Over seven origins, boosting was clearly behind (MASE 0.92 against 0.56 on the airline series).
- **Reproduced:** I re-extracted the three code files from this note into a fresh folder and ran both datasets again with the venv's Python. Both tables came out **identical** to the first run, line for line (the forest and the LSTM use fixed seeds).
- **Prediction intervals (added later, step 6):** 80% intervals for all nine methods, built from each method's own past forecast errors (60 errors from 5 earlier origins), same 7 test origins. Score = interval score divided by the seasonal naive training error, lower is better.

| Method | Airline coverage | Airline score | CO2 coverage | CO2 score |
|---|---|---|---|---|
| Holt-Winters | 63.1% | **3.07** | 70.2% | 2.36 |
| LSTM on changes (3 seeds) | 56.7% | 4.11 | 81.7% | **1.90** |
| SARIMA (airline model) | 46.4% | 4.46 | 72.6% | 2.28 |
| random forest on changes (3 seeds) | 46.4% | 5.72 | 67.9% | 2.63 |
| boosting on changes | 53.6% | 6.20 | 66.7% | 3.20 |
| seasonal naive | 54.8% | 4.28 | 70.2% | 3.19 |
| line + month effects | 60.7% | 4.56 | 57.1% | 3.29 |
| random forest on raw levels (3 seeds) | 71.0% | 6.08 | 69.0% | 3.68 |
| naive | 77.4% | 9.48 | 86.9% | 6.44 |

  - **Almost nobody reached 80%.** On the airline series the best coverage among the methods that are not silly was 63% (Holt-Winters); on CO2 only the LSTM was at the target (81.7%). The two methods that came closest on the airline series, naive (77%) and the forest on raw levels (71%), got there only by being very wide (widths 6.96 and 3.83, against 1.59 for Holt-Winters).
  - **The best point forecaster was also the best interval forecaster on each series** (Holt-Winters on airline, the LSTM on CO2), so on these two series the point ranking and the interval ranking agree at the top. They disagree lower down: SARIMA was second by MASE on both series but fourth on airline by interval score (4.46, behind seasonal naive at 4.28), and boosting fell from fifth to sixth on CO2.
  - **Why the intervals were too narrow (partly explained).** I checked coverage by horizon for two methods on the airline series: Holt-Winters held 100% at horizon 1 but only 43% at horizons 10 to 12, and the mean miss (in logs) grows from 0.031 at horizons 1 to 4 to 0.051 at 9 to 12, so one pooled interval for all 12 horizons is too wide early and too narrow late. SARIMA, though, covered only 57% even at horizon 1 and 29% at horizons 11 and 12, so pooling is not the whole story. The rest I did not trace; a short, older calibration sample on a growing series is one suspect, not a tested cause.
  - **A different result from the one in [[Probabilistic Forecasting]]:** there, the SARIMA model-based 80% interval covered 95.8% on the log airline series. Here, SARIMA with errors from only 5 earlier origins covered 46.4%. Both are small tests on one series, but the gap shows how much the interval recipe (model-based against past errors, and how many past errors) changes the answer.
- **Not done:** no tuning, no timing, only 7 overlapping origins on two series, and no state-of-the-art deep models. See "What I did not run".

## What you need
Versions I used on Windows 11. I made a **separate virtual environment** so that nothing changed in my main Python: `python -m venv --system-site-packages ts-venv` (so it reuses the packages already installed), then `pip install --no-deps statsmodels patsy` inside it (the `--no-deps` flag stops pip from touching other packages). My main environment still has no `statsmodels`.

| Tool | Version |
|---|---|
| Python | 3.9.6 |
| statsmodels | 0.14.6 (installed in the venv) |
| patsy | 1.0.3 (venv) |
| numpy | 1.24.4 |
| scipy | 1.8.1 |
| pandas | 2.2.3 |
| scikit-learn | 1.1.2 |
| torch | 2.8.0 (CPU build) |

Everything runs with the venv's Python, from one folder outside the vault, `D:\python456\forecast-shootout`. The airline data comes from a public copy of the R datasets collection (a small CSV), and the CO2 data is bundled with `statsmodels`. The old `pip` inside the venv (version 21) has no `--dry-run` option, so I could not preview the install; I checked the versions afterwards instead.

## Steps
1. **Get the data.** The airline series is downloaded as a CSV and the CO2 series is built from `statsmodels`' bundled weekly readings, averaged by month (59 of the 2,284 weekly readings are missing, and gaps are filled by linear interpolation).

   **`get_data.py`**
   ```python
   import os, urllib.request
   import statsmodels.api as sm

   URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/AirPassengers.csv"
   if not os.path.exists("AirPassengers.csv"): urllib.request.urlretrieve(URL, "AirPassengers.csv")
   air = [float(line.split(",")[2]) for line in open("AirPassengers.csv").read().split("\n")[1:] if line.strip()]
   co2 = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().interpolate()      # weekly readings -> monthly means
   print("airline passengers:", len(air), air[:3], air[-1], "| Mauna Loa CO2:", len(co2), round(float(co2.iloc[0]), 2), round(float(co2.iloc[-1]), 2))
   ```
   Output: `airline passengers: 144 [112.0, 118.0, 132.0] 432.0 | Mauna Loa CO2: 526 316.1 371.02`.

2. **Write the nine forecasters.** Each takes the history and returns 12 forecasts. In plain words:
   - **naive** repeats the last value; **seasonal naive** repeats the last 12 months ([[Time Series]]).
   - **line + months** fits a trend line plus one effect per month with [[Linear Regression]] (on logs for the multiplicative airline series).
   - **Holt-Winters** is [[Exponential Smoothing]] with trend and season (multiplicative for the airline series, additive for CO2).
   - **SARIMA** is the "airline model", an [[ARIMA]] with one difference, one seasonal difference and two moving-average terms (on logs for the airline series).
   - **random forest** and **boosting** ([[Random Forest]], [[Gradient Boosting]]) look at the last 24 values and predict the next one, then feed it back in, 12 times (the recursive method). "On changes" means they predict the month-to-month change (of the log for the airline series) and add the changes up; "on raw levels" means they predict the level directly.
   - **LSTM** ([[LSTM]]) reads the last 24 changes and predicts all 12 next changes at once (the "direct" method), after scaling.

   **`forecasters.py`**
   ```python
   """Eight ways to forecast the next 12 months. Each takes the history y (a 1-D array of levels) and returns 12 forecasts."""
   import numpy as np, torch, torch.nn as nn
   from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
   from sklearn.linear_model import LinearRegression
   from statsmodels.tsa.holtwinters import ExponentialSmoothing
   from statsmodels.tsa.statespace.sarimax import SARIMAX

   M, H, L = 12, 12, 24                                   # season length, horizon, how many past values the learners see

   def tf(y, mult): return np.log(y) if mult else np.asarray(y, float)       # multiplicative series -> work on logs
   def inv(z, mult): return np.exp(z) if mult else z

   def naive(y, mult, seed): return np.full(H, y[-1])
   def seasonal_naive(y, mult, seed): return np.array([y[len(y) - M + h] for h in range(H)])

   def line_months(y, mult, seed):                        # regression on a time trend plus 11 month indicators
       z, X = tf(y, mult), lambda t: np.column_stack([t] + [t % M == k for k in range(1, M)]).astype(float)
       m = LinearRegression().fit(X(np.arange(len(z))), z)
       return inv(m.predict(X(np.arange(len(z), len(z) + H))), mult)

   def holt_winters(y, mult, seed):
       return ExponentialSmoothing(y, trend="add", seasonal="mul" if mult else "add", seasonal_periods=M).fit().forecast(H)

   def sarima(y, mult, seed):                             # the "airline model": one difference and one seasonal difference, two moving-average terms
       z = tf(y, mult); fit = SARIMAX(z, order=(0, 1, 1), seasonal_order=(0, 1, 1, M)).fit(disp=False)
       return inv(fit.forecast(H), mult)

   def lag_table(v, n): return np.array([v[i:i + n] for i in range(len(v) - n)]), v[n:]

   def trees(make, growth):                               # forest or boosting on the last L values, one step at a time, fed back in
       def f(y, mult, seed):
           z = tf(y, mult); v = np.diff(z) if growth else z      # growth=True: predict changes (stays in range); False: predict raw levels
           X, t = lag_table(v, L); model = make(seed).fit(X, t); hist = list(v)
           for _ in range(H): hist.append(model.predict(np.array(hist[-L:]).reshape(1, -1))[0])
           out = hist[-H:]
           return inv(z[-1] + np.cumsum(out), mult) if growth else inv(np.array(out), mult)
       return f
   forest = lambda s: RandomForestRegressor(200, min_samples_leaf=2, random_state=s)
   boost = lambda s: GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=2, random_state=s)

   class Net(nn.Module):
       def __init__(self): super().__init__(); self.rnn = nn.LSTM(1, 32, batch_first=True); self.out = nn.Linear(32, H)
       def forward(self, x): return self.out(self.rnn(x)[0][:, -1])

   def lstm(y, mult, seed):                               # predicts all 12 changes at once from the last L changes
       z = tf(y, mult); d = np.diff(z); mu, sd = d.mean(), d.std(); s = (d - mu) / sd
       n = len(s) - L - H + 1; X = torch.tensor(np.array([s[i:i + L] for i in range(n)]), dtype=torch.float32)[:, :, None]
       Y = torch.tensor(np.array([s[i + L:i + L + H] for i in range(n)]), dtype=torch.float32)
       torch.manual_seed(seed); net = Net(); opt = torch.optim.Adam(net.parameters(), lr=1e-2)
       for _ in range(200): opt.zero_grad(); nn.functional.mse_loss(net(X), Y).backward(); opt.step()
       pred = net(torch.tensor(s[-L:], dtype=torch.float32)[None, :, None]).detach().numpy()[0] * sd + mu
       return inv(z[-1] + np.cumsum(pred), mult)

   FORECASTERS = {"naive": naive, "seasonal naive": seasonal_naive, "line + months": line_months, "Holt-Winters": holt_winters, "SARIMA (airline model)": sarima,
                  "random forest, raw levels": trees(forest, False), "random forest, changes": trees(forest, True),
                  "boosting, changes": trees(boost, True), "LSTM, changes": lstm}
   STOCHASTIC = {"random forest, raw levels", "random forest, changes", "LSTM, changes"}
   ```

3. **Write the evaluator.** For each dataset it makes 7 forecast origins, 6 months apart, fits each method on the history before the origin only, forecasts the next 12 months, and scores them against the truth. The methods with randomness (forest and LSTM) run with seeds 0, 1 and 2 and the seed-to-seed spread is reported.

   **`evaluate.py`**
   ```python
   import sys, numpy as np, statsmodels.api as sm
   from forecasters import FORECASTERS, STOCHASTIC, M, H

   name = sys.argv[1]                                     # "air" or "co2"
   if name == "air": y, mult = np.array([float(l.split(",")[2]) for l in open("AirPassengers.csv").read().split("\n")[1:] if l.strip()]), True
   else: y, mult = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().interpolate().to_numpy(), False
   origins = [len(y) - H - 6 * k for k in range(6, -1, -1)]          # 7 forecast origins, 6 months apart, the last one ends at the end of the data
   print(name, "| points:", len(y), "| origins:", origins)

   def scores(f, truth, train):                           # MAE, RMSE, and MASE scaled by the seasonal naive error on the training data
       e = truth - f; scale = np.mean(np.abs(train[M:] - train[:-M]))
       return np.mean(np.abs(e)), np.sqrt(np.mean(e ** 2)), np.mean(np.abs(e)) / scale

   rows = {}
   for label, fn in FORECASTERS.items():
       seeds = (0, 1, 2) if label in STOCHASTIC or label.startswith("LSTM") else (0,)
       per_seed = []
       for s in seeds:
           per_seed.append(np.mean([scores(fn(y[:o], mult, s), y[o:o + H], y[:o]) for o in origins], axis=0))
       per_seed = np.array(per_seed); rows[label] = (per_seed.mean(0), per_seed.std(0), len(seeds))
   base = rows["seasonal naive"][0]
   print(f"{'method':28s} {'MAE':>7s} {'RMSE':>7s} {'MASE':>6s}   vs seasonal naive (MASE)")
   for label, (mean, sd, n) in sorted(rows.items(), key=lambda kv: kv[1][0][2]):
       print(f"{label:28s} {mean[0]:7.2f} {mean[1]:7.2f} {mean[2]:6.2f}   x{mean[2] / base[2]:.2f}" + (f"   (sd of MASE over {n} seeds: {sd[2]:.3f})" if n > 1 else ""))
   ```

4. **Run it** with the venv's Python: `ts-venv\Scripts\python.exe evaluate.py air` and `ts-venv\Scripts\python.exe evaluate.py co2`. Output (the library warnings are filtered out; the tables are the ones shown in "Result in short"):
   ```text
   air | points: 144 | origins: [96, 102, 108, 114, 120, 126, 132]
   method                           MAE    RMSE   MASE   vs seasonal naive (MASE)
   Holt-Winters                   16.54   19.91   0.56   x0.45
   SARIMA (airline model)         18.38   21.67   0.62   x0.50
   LSTM, changes                  18.87   22.56   0.64   x0.51   (sd of MASE over 3 seeds: 0.012)
   random forest, changes         19.94   23.16   0.68   x0.55   (sd of MASE over 3 seeds: 0.014)
   boosting, changes              27.02   31.01   0.92   x0.74
   line + months                  32.24   36.42   1.09   x0.88
   random forest, raw levels      34.23   42.11   1.16   x0.93   (sd of MASE over 3 seeds: 0.002)
   seasonal naive                 36.44   39.86   1.24   x1.00
   naive                          66.60   83.16   2.26   x1.82

   co2 | points: 526 | origins: [478, 484, 490, 496, 502, 508, 514]
   method                           MAE    RMSE   MASE   vs seasonal naive (MASE)
   LSTM, changes                   0.40    0.46   0.32   x0.23   (sd of MASE over 3 seeds: 0.024)
   SARIMA (airline model)          0.45    0.53   0.35   x0.26
   Holt-Winters                    0.48    0.55   0.38   x0.27
   random forest, changes          0.55    0.64   0.43   x0.31   (sd of MASE over 3 seeds: 0.003)
   boosting, changes               0.59    0.67   0.46   x0.33
   random forest, raw levels       1.29    1.45   1.01   x0.73   (sd of MASE over 3 seeds: 0.002)
   seasonal naive                  1.76    1.83   1.38   x1.00
   naive                           2.15    2.58   1.68   x1.22
   line + months                   3.24    3.25   2.52   x1.83
   ```
   **What to look for:** the "x" column is the MASE divided by the seasonal naive's. The best methods are below 0.6 of it on the airline series and below 0.3 on CO2. The lines for "raw levels" and "line + months" are the warning signs.

5. **Read the failures.** Why did the forest on raw levels fail? A tree predicts an average of values it has seen, so it cannot go above its training range, and a trending series leaves that range ([[Random Forest]] has a small test of exactly this). Why did the line fail on CO2? The CO2 growth speeds up over the years, and a straight line cannot curve. Why did trees on **changes** work? The changes stay in the same range over time, so they are within what the trees have seen ([[Stationarity]]).

6. **Add prediction intervals to all nine methods** (the follow-up from the "Next ideas" list; the idea is in [[Probabilistic Forecasting]]). Every method gets the same recipe, so the comparison stays fair: for each test origin, take the forecast errors that the same method made at **five earlier origins** (12, 18, 24, 30 and 36 months before; each was finished before the test forecast starts), pool those 5 x 12 = 60 errors, and use their 10th and 90th percentiles as an **80% interval** around the new forecast. For the airline series the errors are measured on logs (so the intervals scale with the level); for CO2 they are plain differences. Each interval is scored with its coverage, its mean width, and the interval score from [[Probabilistic Forecasting]] (alpha 0.2, lower is better). Width and score are divided by the seasonal naive training error (the MASE scale), so the two series can be read side by side.

   **`intervals.py`**
   ```python
   import sys, warnings, numpy as np, statsmodels.api as sm
   from forecasters import FORECASTERS, STOCHASTIC, M, H, tf, inv
   warnings.filterwarnings("ignore")

   name = sys.argv[1]                                     # "air" or "co2"
   if name == "air": y, mult = np.array([float(l.split(",")[2]) for l in open("AirPassengers.csv").read().split("\n")[1:] if l.strip()]), True
   else: y, mult = sm.datasets.co2.load_pandas().data["co2"].resample("MS").mean().interpolate().to_numpy(), False
   ALPHA, NCAL = 0.2, 5                                   # 80% intervals; 5 earlier origins give the past errors
   tests = [len(y) - H - 6 * k for k in range(6, -1, -1)]            # the same 7 test origins as evaluate.py
   need = sorted({o - H - 6 * j for o in tests for j in range(NCAL)} | set(tests))   # calibration origins sit 12, 18, ... 36 months before each test origin
   print(name, "| test origins:", tests, "| origins fitted per method:", len(need), "| earliest:", need[0])

   def interval_score(t, lo, hi, a): return (hi - lo) + 2 / a * (lo - t) * (t < lo) + 2 / a * (t - hi) * (t > hi)

   def run(fn, seed):
       fc = {o: fn(y[:o], mult, seed) for o in need}      # one forecast per origin, reused wherever it is needed
       err = {o: tf(y[o:o + H], mult) - tf(fc[o], mult) for o in need}   # past errors on the working scale (logs for airline)
       cov, wid, iscore = [], [], []
       for o in tests:
           pool = np.concatenate([err[o - H - 6 * j] for j in range(NCAL)])   # 5 origins x 12 horizons = 60 past errors, all finished before o
           q_lo, q_hi = np.quantile(pool, [ALPHA / 2, 1 - ALPHA / 2])
           f = tf(fc[o], mult); lo, hi, t = inv(f + q_lo, mult), inv(f + q_hi, mult), y[o:o + H]
           scale = np.mean(np.abs(y[:o][M:] - y[:o][:-M]))                    # the MASE scale, so series are comparable
           cov.append(np.mean((t >= lo) & (t <= hi))); wid.append(np.mean(hi - lo) / scale); iscore.append(np.mean(interval_score(t, lo, hi, ALPHA)) / scale)
       return np.mean(cov), np.mean(wid), np.mean(iscore)

   rows = {}
   for label, fn in FORECASTERS.items():
       seeds = (0, 1, 2) if label in STOCHASTIC or label.startswith("LSTM") else (0,)
       rows[label] = np.mean([run(fn, s) for s in seeds], axis=0)
   print(f"{'method':28s} {'coverage':>8s} {'width':>7s} {'interval score':>15s}   (width and score divided by the seasonal naive training error)")
   for label, (c, w, s) in sorted(rows.items(), key=lambda kv: kv[1][2]):
       print(f"{label:28s} {c:8.1%} {w:7.2f} {s:15.2f}")
   ```
   Run it with the venv's Python, `ts-venv\Scripts\python.exe intervals.py air` and `... co2` (not with `-I`: the isolated mode hides the `forecasters.py` next to it). Output:
   ```text
   air | test origins: [96, 102, 108, 114, 120, 126, 132] | origins fitted per method: 13 | earliest: 60
   method                       coverage   width  interval score   (width and score divided by the seasonal naive training error)
   Holt-Winters                    63.1%    1.59            3.07
   LSTM, changes                   56.7%    1.92            4.11
   seasonal naive                  54.8%    1.51            4.28
   SARIMA (airline model)          46.4%    1.58            4.46
   line + months                   60.7%    1.93            4.56
   random forest, changes          46.4%    2.21            5.72
   random forest, raw levels       71.0%    3.83            6.08
   boosting, changes               53.6%    2.72            6.20
   naive                           77.4%    6.96            9.48

   co2 | test origins: [478, 484, 490, 496, 502, 508, 514] | origins fitted per method: 13 | earliest: 442
   method                       coverage   width  interval score   (width and score divided by the seasonal naive training error)
   LSTM, changes                   81.7%    1.26            1.90
   SARIMA (airline model)          72.6%    1.29            2.28
   Holt-Winters                    70.2%    1.32            2.36
   random forest, changes          67.9%    1.47            2.63
   seasonal naive                  70.2%    1.71            3.19
   boosting, changes               66.7%    1.52            3.20
   line + months                   57.1%    1.15            3.29
   random forest, raw levels       69.0%    2.52            3.68
   naive                           86.9%    5.68            6.44
   ```
   **What to look for:** a good method has coverage near 80% and a small interval score. Wide intervals can reach 80% cheaply (see naive), so the score is the fair judge.

   **A check on why coverage was low** (`diag_h.py`, it re-uses `intervals.py` and prints how often the 80% interval held at each horizon, for two methods on the airline series):
   ```python
   import sys; sys.argv=["x","air"]
   import io, contextlib
   with contextlib.redirect_stdout(io.StringIO()):
       import intervals as I
   import numpy as np
   from forecasters import FORECASTERS
   for label in ["Holt-Winters","SARIMA (airline model)"]:
       fn=FORECASTERS[label]; hit=np.zeros(12); n=0
       fc={o:fn(I.y[:o],I.mult,0) for o in I.need}; err={o:I.tf(I.y[o:o+12],I.mult)-I.tf(fc[o],I.mult) for o in I.need}
       for o in I.tests:
           pool=np.concatenate([err[o-12-6*j] for j in range(5)]); lo,hi=np.quantile(pool,[.1,.9]); e=err[o]
           hit+=(e>=lo)&(e<=hi); n+=1
       print(label,"coverage by horizon:",np.round(hit/n,2))
       print(" mean |err| h1-4, h9-12 (log):", np.mean([np.abs(err[o][:4]).mean() for o in I.tests]).round(3), np.mean([np.abs(err[o][8:]).mean() for o in I.tests]).round(3), "| pool 10%/90%:", np.round(lo,3), np.round(hi,3))
   ```
   ```text
   Holt-Winters coverage by horizon: [1.   0.57 0.71 0.71 0.57 0.57 0.71 0.86 0.57 0.43 0.43 0.43]
    mean |err| h1-4, h9-12 (log): 0.031 0.051 | pool 10%/90%: -0.068 0.063
   SARIMA (airline model) coverage by horizon: [0.57 0.29 0.43 0.57 0.57 0.43 0.57 0.57 0.57 0.43 0.29 0.29]
    mean |err| h1-4, h9-12 (log): 0.034 0.061 | pool 10%/90%: -0.097 0.076
   ```

## What went wrong, and the fix
- **Trees on raw levels were the first version, and they were bad.** Their forecasts stayed almost flat. The fix was to predict changes and add them up. I kept the raw-level version in the table on purpose, because it is what many people try first.
- **One split flattered boosting.** In the earlier test inside the [[Gradient Boosting]] note, a single split of the last 24 airline months gave boosting an RMSE of 35.8, close to Holt-Winters' 32.5. The seven-origin test says boosting is well behind (0.92 against 0.56). The single split happened to be kind.
- **The first cross-validation test I wrote did not show the effect I expected.** In [[Time Series Cross-Validation]], random folds hid nothing when the features were only past values. I report both cases there, and this project uses forward-moving origins throughout.
- **`pip install --dry-run` did not work** in the new environment (the pip inside the venv was too old). I installed with `--no-deps` and checked the versions afterwards.
- **MASE above 1 is not a bug.** The seasonal naive scale is measured on the training data, and a growing series makes the test errors larger than the training errors. That is why the naive and seasonal naive methods score above 1 here ([[Forecast Metrics]]).

## What I did not run
- **No tuning.** The LSTM size and epochs, the 24 lags, the tree settings, and the SARIMA order were chosen once. A tuned LSTM or boosting model could rank differently. The SARIMA order (0,1,1)(0,1,1) was not searched for the CO2 series.
- **Only 7 forecast origins per series, 6 months apart,** so the forecast windows overlap and the average is noisy. The two series are both monthly with a yearly season, so the ranking may not carry over to daily or irregular data.
- **Seeds:** three seeds for the forest and the LSTM only. The other methods are deterministic and ran once.
- **No timing.** I did not measure how long each method takes, so I cannot say what the LSTM's gain costs.
- **Intervals were built one way only** (pooled past errors of the same method, 5 calibration origins, 80% level). I did not try other numbers of origins, per-horizon intervals, the model-based SARIMA or Holt-Winters intervals on this test, or conformal methods, and I did not measure how much the results move with the calibration length. The 7 test origins overlap, so the coverage numbers are noisy: with 84 forecasts per method, one origin is 12 of them. The reason for the low coverage is only partly explained (see step 6).
- **No external variables,** and none of the newer deep models (for example [[N-BEATS 2019]] style models or a Transformer) were run. [[DLinear 2022]] argues that simple models are hard to beat on long-horizon benchmarks, and this project fits that picture only loosely, since the horizon here is just 12 months.
- **Direct versus recursive** was not compared on its own: the trees use recursive forecasting and the LSTM uses direct forecasting, so those two differences are mixed together in the comparison.

## Concepts used
- [[Time Series]] · [[Trend and Seasonality]] · [[Stationarity]] · [[Autocorrelation]] · [[Exponential Smoothing]] · [[ARIMA]] · [[Time Series Cross-Validation]] · [[Forecast Metrics]] · [[Random Forest]] · [[Gradient Boosting]] · [[LSTM]] · [[Recurrent Neural Network]] · [[Linear Regression]] · [[Feature Engineering]] · [[Overfitting]]

## Next ideas
- Tune each method on an earlier time period only, and see whether the ranking changes.
- Try direct versus recursive forecasting separately for the trees.
- Fix the low coverage: try per-horizon intervals, more calibration origins, and model-based intervals, and measure which one reaches 80% ([[Probabilistic Forecasting]], [[Calibration]]).
- Add a daily or irregular series, and a series with outside variables (holidays, prices).
- Add a [[N-BEATS 2019]] style model and a plain linear model like the one in [[DLinear 2022]].
- Measure the run time of each method.

## Flashcards
#flashcards/projects

#flashcards/projects/reverse #flashcards/topic/project-classic-ml Forecasting Shoot-out (this project):::Nine forecasting methods (naive to LSTM) tested on airline passengers and Mauna Loa CO2 with 12-month forecasts from seven forward-moving origins, scored by MAE, RMSE and MASE.

#flashcards/projects/concept #flashcards/topic/project-classic-ml How was each forecast scored in the project?
?
Seven forecast origins 6 months apart, each forecasting the next 12 months from the history before the origin only. MAE, RMSE and MASE (average miss over the seasonal naive's average miss on the training data) are averaged over the origins. The forest and LSTM use 3 seeds.

#flashcards/projects/concept #flashcards/topic/project-classic-ml What were the best and worst MASE scores on the airline series?::Best: Holt-Winters 0.56, then SARIMA 0.62, LSTM 0.64, forest on changes 0.68. Worst: naive 2.26, then seasonal naive 1.24 and forest on raw levels 1.16. Mean over 7 origins, my run on this machine.

#flashcards/projects/example #flashcards/topic/project-classic-ml What were the best and worst MASE scores on Mauna Loa CO2?::Best: LSTM 0.32 (sd 0.024 over 3 seeds), then SARIMA 0.35 and Holt-Winters 0.38. Worst: line plus months 2.52, then naive 1.68 and seasonal naive 1.38. The forest on raw levels scored 1.01. My run on this machine.

#flashcards/projects/concept #flashcards/topic/project-classic-ml Why did the random forest fail on raw levels, and what fixed it?
?
A tree predicts an average of values it has seen, so it cannot go above its training range, and a trending series leaves that range, so the forecast stays almost flat. Predicting the month-to-month change (which stays in range) and adding the changes up fixed it: MASE 1.16 to 0.68 (airline) and 1.01 to 0.43 (CO2).

#flashcards/projects/concept #flashcards/topic/project-classic-ml Why was a straight line plus month effects the worst method on CO2?
?
The CO2 growth speeds up over the years, so its trend curves, and a straight line cannot follow it (MASE 2.52). On the airline series, where the log trend is closer to straight, the same method beat seasonal naive (1.09 against 1.24).

#flashcards/projects/concept #flashcards/topic/project-classic-ml How did a single split mislead about boosting?
?
On a single split of the last 24 airline months, boosting had an RMSE of 35.8, close to Holt-Winters' 32.5. Over seven origins boosting scored MASE 0.92 against 0.56 for Holt-Winters. One split can be kind or unkind by chance, so use several origins.

#flashcards/projects/concept #flashcards/topic/project-classic-ml Which methods held their ground against the LSTM?
?
Holt-Winters and the airline-model SARIMA. Holt-Winters beat the LSTM on the airline series (0.56 against 0.64) and was within 0.06 of it on CO2 (0.38 against 0.32), with no tuning. The LSTM was best only on CO2.

#flashcards/projects/cloze #flashcards/topic/project-classic-ml A MASE above 1 for the seasonal naive forecast here happens because the test errors are larger than the ==training== errors on a ==growing== series.

#flashcards/projects/concept #flashcards/topic/project-classic-ml What did this project not run?
?
No tuning, only 7 overlapping origins on two monthly series, no timing, no prediction intervals, no outside variables, no newer deep models such as N-BEATS or a Transformer, and no separate comparison of direct versus recursive forecasting (the trees are recursive, the LSTM is direct).

#flashcards/projects/concept #flashcards/topic/project-classic-ml How was the separate environment set up?
?
A virtual environment created with the system site packages, so it reuses the installed numpy, pandas, scipy, scikit-learn and torch, and statsmodels and patsy installed into it without their dependencies. The main environment still has no statsmodels.

#flashcards/projects/concept #flashcards/topic/project-classic-ml How were the 80% prediction intervals built for all nine methods in the shoot-out?
?
For each test origin, take the forecast errors the same method made at 5 earlier origins (12, 18, 24, 30, 36 months before), pool them (5 x 12 = 60 errors, on logs for the airline series), and use their 10th and 90th percentiles around the new forecast. Scored with coverage, width and the interval score (alpha 0.2), the last two divided by the seasonal naive training error.

#flashcards/projects/concept #flashcards/topic/project-classic-ml What were the best interval scores and coverages in the shoot-out?
?
Airline: Holt-Winters 3.07 with 63.1% coverage (target 80%). CO2: LSTM 1.90 with 81.7% coverage, the only method near the target. Lower score is better. Mean over 7 overlapping origins, my run on this machine.

#flashcards/projects/concept #flashcards/topic/project-classic-ml Why did naive have the highest coverage in the interval test, and why does that not make it good?
?
Naive reached 77.4% (airline) and 86.9% (CO2) coverage only because its intervals were huge (width 6.96 and 5.68, against 1.59 (Holt-Winters, airline) and 1.26 (LSTM, CO2)). It also had the worst interval score, 9.48 and 6.44, because the score charges for width as well as for misses.

#flashcards/projects/concept #flashcards/topic/project-classic-ml What did the coverage-by-horizon check show, and what is still unexplained?
?
For Holt-Winters on the airline series the 80% interval held 100% at horizon 1 but 43% at horizons 10 to 12, so pooling all 12 horizons into one interval is too wide early and too narrow late. SARIMA covered only 57% even at horizon 1, so pooling is not the whole cause. The rest was not traced.
