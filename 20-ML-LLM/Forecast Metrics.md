---
tags: [time-series, forecasting, evaluation, metrics, MAE, RMSE, MAPE, MASE, ML]
status: growing
level: 4
tier: 2
---

# Forecast Metrics

Forecast metrics are the **scores that tell you how far off a forecast was**: the average miss (MAE), a score that punishes big misses (RMSE), percent versions (MAPE, sMAPE), and a score that compares you with the naive guess (MASE). Choosing the right one matters, because each one can mislead in its own way.

## 🟢 1. Simple (age 12)
You and a friend both guess how many people will visit the school fair each day for three days. The real numbers are 100, 110 and 120. You guess 98, 115 and 118. How wrong were you? You can say "I was off by 2, 5 and 2, so about 3 on average". That is one score. You could instead say "I was off by 2 percent, 4.5 percent and 1.7 percent". That is another score. Which one is better depends on what you want to know. And a score only means something if you compare it with a simple guess, such as "same as yesterday".

**Example:** If someone who always guesses "the same as yesterday" is off by 3 on average, and you are also off by 3, you were no better than the lazy guess. The score alone would not tell you that.

## 🔵 2. How it works
1. **Find each miss.** The error at one time is the true value minus the forecast.
2. **MAE (mean absolute error):** ignore the signs, and average the sizes of the misses. It is in the same units as the data ("off by 3 visitors").
3. **RMSE (root mean squared error):** square each miss, average, then take the square root. Squaring makes big misses count much more. It is also in the data's units.
4. **MAPE (mean absolute percentage error):** each miss as a percent of the true value, averaged. It is easy to read ("2.7% off"), and it works across series of different sizes. But it explodes when the true value is near zero.
5. **sMAPE:** a percent version that divides by the average of the true and forecast values, to be less lopsided than MAPE.
6. **MASE (mean absolute scaled error):** divide your average miss by the average miss of the naive forecast on the training data. A MASE below 1 means better than the naive guess, above 1 means worse. It works for any scale and for series with zeros.
7. **Always compare with a baseline.** The naive forecast ("same as last time") and the seasonal naive forecast ("same as last year") should be your first scores ([[Time Series]]).
8. **Test on the future only.** Compute these on data the model never saw, with forward-moving folds ([[Time Series Cross-Validation]]).

## 🟠 3. Technical detail
- **Scale-dependent metrics (MAE, RMSE)** are in the data's units. They cannot compare a series of thousands with a series of dozens, but they are fine for comparing models on one series.
- **MAE versus RMSE:** MAE gives the *median-like* middle (a typical miss). RMSE punishes large errors and is the natural score if big misses are very costly, or if the errors are bell-shaped ([[Maximum Likelihood]] with [[Normal Distribution]] noise gives squared error). One huge outlier can dominate RMSE.
- **Percentage metrics (MAPE, sMAPE)** are unit-free but have well-known problems:
  - MAPE is infinite or undefined when a true value is zero, and very large when it is close to zero (for example daily sales of a rare product).
  - MAPE is lopsided: forecasting 50 when the truth is 100 gives 50%, but forecasting 100 when the truth is 50 gives 100%. So it favours low forecasts.
  - sMAPE tries to fix the lopsidedness but can still behave badly near zero, and its "symmetric" claim is debated (from memory).
- **Scaled metrics (MASE)** divide by the in-sample error of the naive forecast (or the seasonal naive, with a season length $m$). They are unit-free, defined at zero, and directly tell you if you beat the baseline (Hyndman and Koehler, 2006; from memory).
- **Bias** (the average signed error) tells you whether a forecast is systematically too high or too low. MAE and RMSE hide the sign.
- **Probabilistic forecasts** (intervals and quantiles) need other scores such as the pinball (quantile) loss, and their coverage can be checked like [[Calibration]]. They are outside this note; see [[Probabilistic Forecasting]].
- **Which one to report:** report a scale-free score (MASE) and a scale-dependent one (MAE or RMSE), on the same forward-moving folds, with the naive baseline next to them.

```python
import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

mae = lambda a, f: float(np.mean(np.abs(a - f)))
rmse = lambda a, f: float(np.sqrt(np.mean((a - f) ** 2)))
mape = lambda a, f: float(100 * np.mean(np.abs(a - f) / np.abs(a)))
smape = lambda a, f: float(100 * np.mean(2 * np.abs(a - f) / (np.abs(a) + np.abs(f))))
def mase(a, f, train, m=1):                                           # error divided by the in-sample error of the naive forecast
    return float(np.mean(np.abs(a - f)) / np.mean(np.abs(train[m:] - train[:-m])))

a, f = np.array([100., 110, 120]), np.array([98., 115, 118])          # tiny example
print(round(mae(a, f), 3), round(rmse(a, f), 3), round(mape(a, f), 3), round(smape(a, f), 3))
print(round(mean_absolute_error(a, f), 3), round(float(np.sqrt(mean_squared_error(a, f))), 3), round(100 * mean_absolute_percentage_error(a, f), 3))
train = np.array([100., 104, 103, 109, 108])                          # naive in-sample errors: 4, 1, 6, 1 -> mean 3
print(round(mase(a, f, train), 3))

print("MAPE is lopsided: actual 100, forecast 50 ->", mape(np.array([100.]), np.array([50.])), "| actual 50, forecast 100 ->", mape(np.array([50.]), np.array([100.])))
print("sMAPE:", round(smape(np.array([100.]), np.array([50.])), 1), round(smape(np.array([50.]), np.array([100.])), 1))
z = np.array([0., 3, 0, 2, 5]); fz = np.array([1., 2, 1, 2, 4])      # a series with zeros (for example daily sales of a rare product)
with np.errstate(divide="ignore", invalid="ignore"): print("MAPE with zeros:", mape(z, fz), "| MASE:", round(mase(z, fz, np.array([1., 0, 2, 0, 4, 1, 0, 3])), 3))

y = pd.read_csv("AirPassengers.csv")["value"].to_numpy(float); train, test = y[:120], y[120:]
sn = np.array(train[-12:].tolist() * 2); nv = np.full(24, train[-1])
for name, fc in [("naive", nv), ("seasonal naive", sn)]:
    print(name, "| MAE", round(mae(test, fc), 1), "| RMSE", round(rmse(test, fc), 1), "| MAPE", round(mape(test, fc), 1), "| MASE (seasonal)", round(mase(test, fc, train, 12), 2), "| MASE (plain)", round(mase(test, fc, train, 1), 2))
```

Output:
- `3.0 3.317 2.737 2.715`: the tiny example (MAE, RMSE, MAPE %, sMAPE %), and `3.0 3.317 2.737` again from scikit-learn: identical.
- `1.0`: the tiny MASE.
- `MAPE is lopsided: actual 100, forecast 50 -> 50.0 | actual 50, forecast 100 -> 100.0` and `sMAPE: 66.7 66.7`.
- `MAPE with zeros: inf | MASE: 0.35`: a series with zeros breaks MAPE but not MASE.
- The last two lines are the airline series, last 24 months: naive MAE 115.2, RMSE 137.3, MAPE 23.6%; seasonal naive MAE 71.2, RMSE 77.0, MAPE 15.5%. Their MASE against the in-sample seasonal naive error is 4.03 and 2.49, and against the plain in-sample naive error 5.2 and 3.22. All are above 1 because the series kept growing, and the in-sample errors were measured on the smaller earlier values.

## 🔴 4. Deep / formal
**The two scale-dependent errors.** With true values $y_t$, forecasts $\hat y_t$ and errors $e_t=y_t-\hat y_t$ over $h$ forecast points:

$$
\mathrm{MAE}=\frac1h\sum_{t=1}^{h}|e_t|,\qquad\qquad \mathrm{RMSE}=\sqrt{\frac1h\sum_{t=1}^{h}e_t^{\,2}}
$$

**Where:**
- $y_t$ is the true value, $\hat y_t$ the forecast, and $e_t=y_t-\hat y_t$ the error (positive when the forecast was too low).
- $h$ is the number of forecast points being scored.
- $|\cdot|$ drops the sign. Squaring in RMSE makes large errors count much more, and the square root brings the units back.
- **Matters:** RMSE is never smaller than MAE, and it is much larger when the errors are uneven (a few big misses). Without the absolute value or the square, positive and negative errors would cancel and a bad forecast could score zero.

**Example (tiny numbers):** true $100,110,120$ and forecasts $98,115,118$, so $e=2,\,-5,\,2$.
- $\mathrm{MAE}=(2+5+2)/3=9/3=\mathbf{3}$.
- $\mathrm{RMSE}=\sqrt{(4+25+4)/3}=\sqrt{11}=\mathbf{3.317}$.
- Meaning: the typical miss is 3, but the 5 pulls RMSE above MAE. If all three errors were 3, both would be exactly 3.

**The percentage errors.**

$$
\mathrm{MAPE}=\frac{100}{h}\sum_{t=1}^{h}\frac{|e_t|}{|y_t|},\qquad\qquad \mathrm{sMAPE}=\frac{100}{h}\sum_{t=1}^{h}\frac{2\,|e_t|}{|y_t|+|\hat y_t|}
$$

**Where:**
- $e_t$, $y_t$, $\hat y_t$ and $h$ are as above. The factor 100 turns the ratio into a percent.
- MAPE divides each miss by the true value. sMAPE divides by the average size of the true value and the forecast.
- **Matters:** MAPE breaks when $y_t=0$ and punishes over-forecasts harder than under-forecasts, so a model trained to minimise it tends to forecast low. sMAPE is bounded and less lopsided, but it still misbehaves when both values are near zero.

**Example (tiny numbers):** the same data.
- MAPE: $2/100=2\%$, $5/110=4.545\%$ and $2/120=1.667\%$. Average: $\mathbf{2.737\%}$.
- sMAPE: $2\cdot2/(100+98)=2.02\%$, $2\cdot5/(110+115)=4.44\%$ and $2\cdot2/(120+118)=1.68\%$. Average: $\mathbf{2.715\%}$.
- Lopsided case: actual 100, forecast 50: MAPE $=50\%$. Actual 50, forecast 100: MAPE $=100\%$. sMAPE is $66.7\%$ in both.
- Meaning: the two are close here, but the lopsided pair shows that MAPE treats a forecast that is twice as large as worse than one that is half as large.

**The scaled error.** Compare your average miss with the average miss that the naive forecast made on the training data (length $n$, season length $m$; use $m=1$ for the plain naive):

$$
\mathrm{MASE}=\frac{\frac1h\sum_{t=1}^{h}|e_t|}{\frac1{n-m}\sum_{t=m+1}^{n}|y_t-y_{t-m}|}
$$

**Where:**
- The top is your MAE on the test points.
- The bottom is the mean absolute error of the naive (or seasonal naive) forecast on the **training** data: the average of $|y_t-y_{t-m}|$.
- $n$ is the length of the training series and $m$ the season length ($m=1$ for plain naive).
- **Matters:** it has no units, so you can average it across series, and it is defined even when the series contains zeros. A value below 1 means better than the naive forecast, above 1 means worse. Because the scale comes from the training data, a test period far larger than training (as in a fast-growing series) pushes it above 1 even for sensible forecasts.

**Example (tiny numbers):** the same forecasts (MAE 3) and a training series $100,104,103,109,108$ with $m=1$.
- Naive in-sample errors: $|104-100|=4,\ |103-104|=1,\ |109-103|=6,\ |108-109|=1$. Mean: $(4+1+6+1)/4=\mathbf{3}$.
- $\mathrm{MASE}=3/3=\mathbf{1}$.
- With zeros: a series $0,3,0,2,5$ forecast as $1,2,1,2,4$ has an infinite MAPE, but with the training series $1,0,2,0,4,1,0,3$ the MASE is $0.8/2.286=\mathbf{0.35}$.
- Meaning: a MASE of 1 means the forecast was exactly as good as the naive guess on the training data. The zero-heavy series cannot be scored by MAPE at all.

**Further notes:**
- **Bias:** the mean error $\frac1h\sum e_t$ is a separate check. A model with a small MAE but a large mean error is systematically off.
- **Rank agreement:** different metrics can rank models differently, because they weigh large and small errors differently. Report more than one.
- **Averaging across series** needs a scale-free metric (MASE or a percent), or the biggest series dominates.

**What came before, and how this improved on it:**
- **Before:** squared and absolute errors (MSE, MAE) and percent errors (MAPE), the usual choices in forecasting practice.
- **Its limit:** scale-dependent scores cannot compare series; MAPE is undefined at zero and lopsided.
- **What changed:** divide by the naive forecast's own error, so the score is unit-free, defined at zero and tells you whether you beat the baseline (MASE, from Hyndman and Koehler, 2006; from memory, not checked).
- **How much better:** in the code above, on a series with zeros the MAPE was **infinite** while the MASE was **0.35**, and on a lopsided pair MAPE gave **50%** for one direction and **100%** for the other. These are small constructed examples, not a proof about real data.

## Related
- [[Time Series]] · [[Time Series Cross-Validation]] · [[Trend and Seasonality]] · [[Exponential Smoothing]] · [[ARIMA]] · [[Forecasting Shoot-out]] · [[Evaluation Metrics]] · [[Loss Function]] · [[Calibration]] · [[Maximum Likelihood]] · [[Normal Distribution]] · [[Variance]] · [[Probabilistic Forecasting]]
- [[Anomaly Detection]]
## Sources
- Hyndman & Koehler, "Another look at measures of forecast accuracy", 2006 (named from memory, not checked)
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, section on forecast accuracy (free online; from memory)
- Numbers in this note come from the code above (my arithmetic and my run on this machine). No image: I found no suitable free image on Wikimedia Commons.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/time-series MASE:::Mean absolute scaled error: your average miss divided by the average miss of the naive forecast on the training data. Below 1 beats the naive guess. Unit-free and defined at zero.

#flashcards/ml/concept #flashcards/topic/time-series What are the main problems with MAPE?
?
It is infinite or undefined when a true value is zero and huge when it is near zero. It is lopsided: forecasting 50 for a truth of 100 gives 50%, but forecasting 100 for a truth of 50 gives 100%, so it favours low forecasts.

#flashcards/ml/formula #flashcards/topic/time-series Write MAE and RMSE and explain each symbol.
?
$$\mathrm{MAE}=\frac1h\sum_{t=1}^{h}|e_t|,\qquad\qquad \mathrm{RMSE}=\sqrt{\frac1h\sum_{t=1}^{h}e_t^{\,2}}$$
- $y_t$: the true value. $\hat y_t$: the forecast. $e_t=y_t-\hat y_t$: the error (positive when the forecast was too low). $h$: the number of forecast points.
- The absolute value drops the sign. Squaring in RMSE makes large errors count much more, and the root restores the units.
- RMSE is never smaller than MAE. Without the absolute value or the square, positive and negative errors cancel.

#flashcards/ml/example #flashcards/topic/time-series True values 100, 110, 120 and forecasts 98, 115, 118: what are MAE and RMSE?::Errors 2, -5, 2. MAE = 9/3 = 3. RMSE = sqrt((4 + 25 + 4)/3) = sqrt(11) = 3.317. The 5 pulls RMSE above MAE.

#flashcards/ml/formula #flashcards/topic/time-series Write MAPE and sMAPE and explain each symbol.
?
$$\mathrm{MAPE}=\frac{100}{h}\sum_{t=1}^{h}\frac{|e_t|}{|y_t|},\qquad\qquad \mathrm{sMAPE}=\frac{100}{h}\sum_{t=1}^{h}\frac{2\,|e_t|}{|y_t|+|\hat y_t|}$$
- $e_t$, $y_t$, $\hat y_t$, $h$: as for MAE. The factor 100 turns the ratio into a percent.
- MAPE divides by the true value. sMAPE divides by the average size of the true value and the forecast.
- MAPE breaks at $y_t=0$ and punishes over-forecasts harder than under-forecasts. sMAPE is bounded and less lopsided but still misbehaves near zero.

#flashcards/ml/example #flashcards/topic/time-series For true 100, 110, 120 and forecasts 98, 115, 118, what are MAPE and sMAPE?::MAPE: 2%, 4.545% and 1.667%, average 2.737%. sMAPE: 2.02%, 4.44% and 1.68%, average 2.715%. For the lopsided pair (100 against 50, and 50 against 100) MAPE is 50% and 100%, sMAPE 66.7% both.

#flashcards/ml/formula #flashcards/topic/time-series Write MASE and explain each symbol.
?
$$\mathrm{MASE}=\frac{\frac1h\sum_{t=1}^{h}|e_t|}{\frac1{n-m}\sum_{t=m+1}^{n}|y_t-y_{t-m}|}$$
- The top is your MAE on the test points. The bottom is the average of $|y_t-y_{t-m}|$ on the training data, the naive (or seasonal naive) error.
- $n$: the length of the training series. $m$: the season length ($m=1$ for plain naive).
- Unit-free, so it can be averaged across series, and defined even with zeros. Below 1 beats the naive forecast. A test period much larger than training pushes it above 1.

#flashcards/ml/example #flashcards/topic/time-series MAE 3, training series 100, 104, 103, 109, 108 and $m=1$: what is the MASE?::Naive in-sample errors 4, 1, 6, 1, mean 3. MASE = 3/3 = 1, exactly as good as the naive guess. With zeros (test 0, 3, 0, 2, 5 against forecasts 1, 2, 1, 2, 4) MAPE is infinite but MASE is 0.8 / 2.286 = 0.35.

#flashcards/ml/concept #flashcards/topic/time-series Why compare a forecast with the naive forecast, and which metric does it directly?
?
A score alone does not say whether the model beats the lazy guess "same as last time" (or "same as last year"). MASE divides your error by the naive forecast's training error, so below 1 means you beat it.

#flashcards/ml/example #flashcards/topic/time-series What were the MAE, RMSE and MAPE of naive and seasonal naive on the last 24 airline months?::Naive: MAE 115.2, RMSE 137.3, MAPE 23.6%. Seasonal naive: MAE 71.2, RMSE 77.0, MAPE 15.5%. My run on this machine.

#flashcards/ml/cloze #flashcards/topic/time-series RMSE punishes ==large errors== more than MAE, and MAPE is ==infinite== when a true value is zero.
