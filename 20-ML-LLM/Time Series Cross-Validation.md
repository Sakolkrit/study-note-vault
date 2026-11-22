---
tags: [time-series, forecasting, cross-validation, evaluation, leakage, ML]
status: growing
level: 4
tier: 2
---

# Time Series Cross-Validation

Time series cross-validation tests a forecasting model the way it will really be used: **train only on the past, forecast the next stretch, score it, then move forward in time and repeat**. Unlike ordinary [[Cross-Validation]], it never shuffles and never lets the model see the future.

## 🟢 1. Simple (age 12)
Imagine you want to know if you are good at guessing tomorrow's weather. A fair test is: pretend it is Monday, guess Tuesday, then check. Next, pretend it is Tuesday (now you know Monday and Tuesday), guess Wednesday, and check. You must **never** peek at the answers for the days you are guessing. A test where you get to see Wednesday's weather before guessing Tuesday's would give you a great score that means nothing.

**Example:** A teacher gives a quiz about last week's lesson, then one about this week's lesson, then next week's. You always study the past before each quiz. That is how forward-moving testing works.

## 🔵 2. How it works
1. **Pick a starting point (the "origin").** Everything up to the origin is training data. Everything after it is hidden.
2. **Train the model** on the data up to the origin only.
3. **Forecast the next few steps** (the horizon) from the origin.
4. **Compare the forecast with what really happened** in those steps, using an error measure ([[Forecast Metrics]]).
5. **Move the origin forward** (by one step or several) and repeat from step 2. Now the training data also includes the stretch you just tested on.
6. **Average the errors** from all these tests. That average is your estimate of how good the model will be in the future.
7. **Two styles:** the **expanding window** keeps all the past data each time. The **rolling window** keeps only the most recent fixed amount (useful when old data is out of date).
8. **Why not ordinary cross-validation?** It shuffles the data into random groups, so the model trains on points from the future and tests on points from the past, which hides how it will do on new data.
9. **Keep a final test.** After choosing a model with these tests, hold back the very last stretch of time and use it only once.

## 🟠 3. Technical detail
- **Settings:** the first training length $n_0$, the horizon $h$, and the step $s$ between origins. Scikit-learn's `TimeSeriesSplit` does this (with an optional `gap` between training and test, and a `test_size`).
- **Leakage means the model saw information from the future.** It is the main danger, and it is easy to cause by accident:
  - Shuffled or random folds.
  - Scaling or filling gaps with statistics computed on the whole series. These must be fitted on the training part only ([[Feature Scaling]]).
  - Features computed with a window that looks forward (a "centred" moving average).
  - Using a variable that would not yet be known at forecast time.
- **A gap** between the training end and the test start helps if the features have a long window or the target overlaps.
- **Tuning hyperparameters** needs its own forward-moving split inside the training data ("nested" validation), or you leak.
- **Cost:** one model fit per fold. For slow models this limits the number of origins.
- **Several series:** split by time across all of them (a "panel"). Do not put some series in training and others in testing and call it a time split.
- **Is random CV always wrong?** Not always in effect, but always unsafe. In the code below, with only past values as features, it barely changed the error. With calendar features and a trend, it gave a very wrong picture.
- **Trees cannot extrapolate.** A tree model predicts averages of values it has seen, so it cannot predict values above the training range. Random folds hide this, because every test point lies inside the range. Forward folds show it ([[Random Forest]], [[Gradient Boosting]]).

```python
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, TimeSeriesSplit

ap = pd.read_csv("AirPassengers.csv")["value"].to_numpy(float)

def cv_rmse(X, y, cv, scale=1.0):
    errs = []
    for tr, te in cv.split(X):
        m = RandomForestRegressor(200, min_samples_leaf=2, random_state=0).fit(X[tr], y[tr])
        errs.append(((m.predict(X[te]) - y[te]) ** 2).mean())
    return round(float(np.sqrt(np.mean(errs))) * scale, 4)

random_cv, forward_cv = KFold(5, shuffle=True, random_state=0), TimeSeriesSplit(5, test_size=20)

# case 1: features are only past values (the last 12 monthly growth rates), target is the next growth rate
d = np.diff(np.log(ap)); L = 12
X1 = np.array([d[i:i + L] for i in range(len(d) - L)]); y1 = d[L:]
print("past values only   | random folds:", cv_rmse(X1, y1, random_cv), "| walk-forward:", cv_rmse(X1, y1, forward_cv))

# case 2: features are the calendar (months since the start, and the month), target is the passenger count itself
t = np.arange(len(ap)); X2 = np.column_stack([t, t % 12]); y2 = ap
r2, f2 = cv_rmse(X2, y2, random_cv), cv_rmse(X2, y2, forward_cv)
print("calendar features  | random folds:", r2, "| walk-forward:", f2, "| ratio", round(f2 / r2, 1))

n, n0, h, s = 24, 12, 3, 3                                           # tiny example: how many forward folds?
print(list(range(n0, n - h + 1, s)), (n - n0 - h) // s + 1)
```

Output:
- `past values only   | random folds: 0.045 | walk-forward: 0.0417`: with the last 12 monthly growth rates as features, a random forest has about the same error with either method (RMSE in log growth units). Here random folds did not exaggerate.
- `calendar features  | random folds: 26.6767 | walk-forward: 65.9401 | ratio 2.5`: with the time index and the month as features and the passenger count as target, random folds show an RMSE of 26.7 and walk-forward folds show 65.9, 2.5 times larger. The random folds hide that the model cannot go beyond the numbers it has seen.
- `[12, 15, 18, 21] 4`: the tiny example (origins and the number of folds).

## 🔴 4. Deep / formal
**How many folds.** With $n$ observations, an initial training length $n_0$, a horizon $h$ and a step $s$ between origins, the origins are $n_k=n_0+(k-1)s$, and the number of folds is:

$$
K=\left\lfloor\frac{n-n_0-h}{s}\right\rfloor+1
$$

**Where:**
- $n$ is the length of the series. $n_0$ is the length of the first training set.
- $h$ is the horizon (how many steps each fold tests). $s$ is how far the origin moves each time.
- $\lfloor\cdot\rfloor$ rounds down. $K$ is the number of folds. The last fold must still have $h$ real values to test on.
- **Matters:** a small $s$ gives more folds that overlap heavily (they are not independent), and a large $s$ gives few, noisy folds. A larger $n_0$ shrinks $K$. If the formula gives $K<1$, the series is too short for these settings.

**Example (tiny numbers):** $n=24$, $n_0=12$, $h=3$, $s=3$.
- $K=\lfloor(24-12-3)/3\rfloor+1=\lfloor3\rfloor+1=\mathbf{4}$.
- Origins: $12,\,15,\,18,\,21$.
- Test windows: points $13$ to $15$, then $16$ to $18$, $19$ to $21$ and $22$ to $24$.
- Meaning: the four test windows cover the last 12 points exactly once each, and each fold trains on everything before its window.

**The cross-validated error.** For an error measure $\ell$ (for example the absolute error), average over the folds and over the horizon steps within each fold:

$$
\mathrm{CV}=\frac1K\sum_{k=1}^{K}\frac1h\sum_{j=1}^{h}\ell\big(y_{n_k+j},\ \hat y_{n_k+j\mid n_k}\big)
$$

**Where:**
- $n_k$ is the origin of fold $k$. $y_{n_k+j}$ is the true value $j$ steps after the origin.
- $\hat y_{n_k+j\mid n_k}$ is the forecast of that value, made using only data up to the origin $n_k$.
- $\ell$ is the error of one forecast (see [[Forecast Metrics]]).
- **Matters:** the forecast must be made from the origin alone, which is the whole point. If any part of the training used a value after $n_k$, the average is too optimistic. Errors at larger $j$ are usually bigger, so a longer horizon gives a larger $\mathrm{CV}$.

**Example (tiny numbers):** four folds with average absolute errors of $2,\ 4,\ 3$ and $3$ over their horizons.
- $\mathrm{CV}=(2+4+3+3)/4=12/4=\mathbf{3}$.
- Meaning: on average the model misses by 3 units. The spread across folds (2 to 4) says how much that depends on the time period.

**Further notes:**
- **Dependence between folds:** nearby folds share most of their training data, so the folds are not independent, and the plain average has no simple confidence interval (from memory).
- **Origins and "forecast horizon error":** some write the error as a function of the horizon (error at 1 step, 2 steps, ...). It shows how fast forecasts degrade.
- **Blocked cross-validation** (from memory) keeps contiguous blocks for testing while allowing training data on both sides. It is used for non-forecasting tasks on time-ordered data, and is not safe for true forecasting.

**What came before, and how this improved on it:**
- **Before:** ordinary random $k$-fold cross-validation (Stone and Geisser, 1970s; from memory) or a single split.
- **Its limit:** random folds train on the future and test on the past, which breaks the order and leaks information. A single split gives one noisy estimate.
- **What changed:** "rolling origin" evaluation: repeat forward-only splits from many forecast origins and average (Tashman, 2000, is the reference I remember; not checked).
- **How much better:** in the code above, with calendar features, ordinary random folds reported an RMSE of **26.7** while forward folds reported **65.9**, so the honest error was **2.5 times** larger than the random folds claimed. With only past values as features, the two methods agreed (0.045 against 0.042), so how much it matters depends on the features. One series and one model.

## Related
- [[Time Series]] · [[Cross-Validation]] · [[Forecast Metrics]] · [[Stationarity]] · [[Autocorrelation]] · [[ARIMA]] · [[Exponential Smoothing]] · [[Forecasting Shoot-out]] · [[Overfitting]] · [[Random Forest]] · [[Gradient Boosting]] · [[Feature Scaling]] · [[Feature Engineering]] · [[Evaluation Metrics]]

## Sources
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, section on time series cross-validation (free online; from memory)
- Tashman, "Out-of-sample tests of forecasting accuracy", 2000 (named from memory, not checked)
- Numbers in this note come from the code above (my arithmetic and my run on this machine). No image: I found no suitable free image on Wikimedia Commons.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/time-series Time series cross-validation:::Testing a forecasting model by training on the past up to an origin, forecasting the next h steps, scoring, then moving the origin forward and repeating, never shuffling and never seeing the future.

#flashcards/ml/concept #flashcards/topic/time-series What is the difference between an expanding window and a rolling window?
?
An expanding window keeps all past data and grows each fold. A rolling window keeps only the most recent fixed amount, which helps when old data is out of date.

#flashcards/ml/concept #flashcards/topic/time-series Name common sources of leakage in time series validation.
?
Random or shuffled folds. Scaling or filling gaps with statistics from the whole series. Features computed with a forward-looking (centred) window. Variables that would not be known yet at forecast time. Tuning hyperparameters on the test period.

#flashcards/ml/formula #flashcards/topic/time-series Write the number of forward-moving folds and explain each symbol.
?
$$K=\left\lfloor\frac{n-n_0-h}{s}\right\rfloor+1$$
- $n$: the length of the series. $n_0$: the length of the first training set.
- $h$: the horizon (steps tested per fold). $s$: how far the origin moves each time.
- $\lfloor\cdot\rfloor$ rounds down. $K$: the number of folds. The last fold must still have $h$ real values.
- Small $s$ gives many overlapping, non-independent folds. Large $s$ gives few, noisy ones. If $K<1$ the series is too short.

#flashcards/ml/example #flashcards/topic/time-series With $n=24$, $n_0=12$, $h=3$, $s=3$, what are $K$ and the origins?::K = floor((24 - 12 - 3)/3) + 1 = 4. Origins 12, 15, 18, 21. The test windows are points 13-15, 16-18, 19-21 and 22-24.

#flashcards/ml/formula #flashcards/topic/time-series Write the cross-validated error for time series and explain each symbol.
?
$$\mathrm{CV}=\frac1K\sum_{k=1}^{K}\frac1h\sum_{j=1}^{h}\ell\big(y_{n_k+j},\ \hat y_{n_k+j\mid n_k}\big)$$
- $n_k$: the origin of fold $k$. $y_{n_k+j}$: the true value $j$ steps after the origin.
- $\hat y_{n_k+j\mid n_k}$: the forecast of it using only data up to the origin. $\ell$: the error of one forecast.
- The forecast must come from the origin alone. If any training used a value after $n_k$, the average is too optimistic. Errors usually grow with $j$.

#flashcards/ml/example #flashcards/topic/time-series Four folds have average absolute errors 2, 4, 3 and 3: what is the CV error?::(2 + 4 + 3 + 3) / 4 = 3. On average the model misses by 3 units, and the spread across folds (2 to 4) shows how much it depends on the period.

#flashcards/ml/concept #flashcards/topic/time-series With calendar features on the airline series, what did random folds and forward folds report for a random forest?::Random folds: RMSE 26.7. Walk-forward: RMSE 65.9, 2.5 times larger. Random folds hide that trees cannot go beyond the values they saw. With only past values as features both agreed (0.045 against 0.042). My run on this machine.

#flashcards/ml/concept #flashcards/topic/time-series Why do random folds hide the failure of tree models on trending series?
?
A tree predicts averages of training values, so it cannot predict above the range it saw. Random folds put every test point inside the training range. Forward folds test the future, which lies outside the range, and show the failure.

#flashcards/ml/cloze #flashcards/topic/time-series In time series cross-validation the forecast for each fold must use ==only data up to the origin==, and the folds must never be ==shuffled==.
