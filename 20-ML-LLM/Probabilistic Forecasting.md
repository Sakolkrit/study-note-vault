---
tags: [time-series, forecasting, uncertainty, prediction-interval, quantile, pinball-loss, evaluation, ML]
status: growing
level: 4
tier: 3
---

# Probabilistic Forecasting

A probabilistic forecast does not say "it will be 100". It says **"it will probably be between 90 and 110, and it is very unlikely to be above 120"**. It gives a range, or a whole spread of possible values, so that you can see how sure the forecast is.

## 🟢 1. Simple (age 12)
The weather app says "70% chance of rain". That is more useful than "it will rain" or "it will not rain", because it tells you how sure the forecast is. A forecast of next month's sales works the same way. "About 100 sales" is one guess. "Between 90 and 110, almost surely below 120" is a guess **plus** how wrong it might be, and that is what you need to decide how many boxes to order.

**Example:** You order stock for a shop. If you order exactly the guess of 100 and sales turn out to be 115, you run out. If you know the likely range, you can order 110 and be safe most of the time without ordering far too much.

## 🔵 2. How it works
1. **A single number hides the doubt.** A point forecast ([[Time Series]]) is just the middle of what could happen.
2. **A quantile** is a cut-off value. The **90% quantile** is the number that 90% of the possible outcomes fall below. The 10% quantile is the number that only 10% fall below.
3. **A prediction interval** is the stretch between two quantiles. An **80% interval** runs from the 10% quantile to the 90% quantile, so about 8 times out of 10 the real value should land inside it.
4. **The doubt grows with distance.** Next month is easier to guess than next year, so intervals get wider the further ahead you look. A plot of such growing bands is a **fan chart** (see the picture).
5. **Where the range comes from:**
   - **From the model:** models such as [[ARIMA]] and [[Exponential Smoothing]] assume how the random noise behaves (usually a bell shape, see [[Normal Distribution]]) and work out the range from that.
   - **From past mistakes:** look at how wrong earlier forecasts were at each distance, and use those errors to put a band around the new forecast.
   - **By training for quantiles:** train a model whose goal is directly to predict a chosen quantile (it uses the "pinball loss", below). This works with machine-learning models such as [[Gradient Boosting]].
   - **By simulation:** make many random possible futures ([[Monte Carlo]]) and read the quantiles off them.
6. **Check it with two questions.** **Coverage:** does an 80% interval really contain the truth about 80% of the time? (This is the idea of [[Calibration]].) **Sharpness:** is the interval narrow? An interval that is infinitely wide always contains the truth, and it is useless.
7. **Score it with a proper score,** one that rewards honesty: the pinball loss for quantiles, or the interval score for an interval. See the formulas below.
8. **Test it like any forecast:** on forward-moving origins, never on data the model has seen ([[Time Series Cross-Validation]]).

![[fan-chart-inflation.jpg]]
*A fan chart. A red line of past values (2003 to mid-2006) is followed by a central forecast (the thick line) with nested shaded bands around it. Each band is a prediction interval of a different probability, and the bands widen the further ahead the forecast goes. The data are labelled hypothetical on the file page (a fan chart of Colombian total inflation, in percent). Image: Juan Manuel Julio, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:FanChartInfl.jpg), CC BY-SA 3.0.*

## 🟠 3. Technical detail
- **Quantile forecasts** give a value for each chosen level, such as the 10%, 50% and 90% quantiles. The 50% quantile is the median, a good single forecast. Several quantiles together describe the whole spread. A point forecast from squared error ([[Forecast Metrics]]) is aimed at the mean, not at a quantile.
- **Calibration and sharpness.** A good probabilistic forecast is **calibrated** (the observed frequencies match the stated probabilities) and **sharp** (the intervals are as narrow as the data allows). You cannot judge either one alone.
- **The pinball (quantile) loss** is the training and scoring loss for a quantile forecast. It punishes an under-forecast and an over-forecast by different amounts, set by the level. The value of $q$ that minimises it on a sample is the sample quantile (the code checks this).
- **Intervals widen with the horizon.** For a random walk the spread grows like the square root of the horizon. For models with a trend or a seasonal part, the growth is different, and it is worked out by the model.
- **Model-based intervals depend on assumptions:** the model is right and the noise has the assumed shape. If not, the stated coverage may be wrong in either direction (see the code).
- **Empirical intervals** use the errors of past forecasts at the same horizon (from earlier forecast origins). They need enough past errors, and they assume that the future errors look like the past ones.
- **Log scale:** if you forecast the logarithm of a series, you can build the interval on the log scale and then take the exponential of both ends. Quantiles survive such an increasing transformation, so the result is a valid interval for the original series.
- **Overlap and noise in the coverage figure.** Forecasts from nearby origins overlap, so the 96 forecasts in the code are not independent. A coverage figure from so few points has a wide error margin (about 4 percentage points if they were independent, and more because they are not).
- **Other tools** (named from memory, not covered here): **CRPS**, a score for a whole forecast distribution (it can be seen as the pinball loss averaged over all levels); **quantile regression** (training with the pinball loss); and **conformal prediction**, which builds intervals from past errors and has a coverage guarantee under certain conditions that do not strictly hold for time series.

```python
import numpy as np, pandas as pd
from sklearn.metrics import mean_pinball_loss
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

def pinball(y, q, tau):                                    # the quantile ("pinball") loss, written out
    d = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))

def interval_score(y, lo, hi, alpha):                      # Winkler score: width plus a penalty for every miss
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((hi - lo) + 2 / alpha * (lo - y) * (y < lo) + 2 / alpha * (y - hi) * (y > hi)))

print(pinball([10], [8], 0.9), pinball([10], [12], 0.9), pinball([10], [8], 0.1), pinball([10], [12], 0.1))    # tiny example
print(round(pinball([3, 5, 9], [4, 4, 4], 0.9), 4), round(mean_pinball_loss([3, 5, 9], [4, 4, 4], alpha=0.9), 4))
sample = np.random.default_rng(0).normal(100, 10, 5000); grid = np.linspace(80, 130, 2001)           # the q that minimises pinball loss is the tau-quantile
best = grid[np.argmin([pinball(sample, q, 0.9) for q in grid])]; print("best q for tau 0.9:", round(float(best), 2), "| 90th percentile of the sample:", round(float(np.quantile(sample, 0.9)), 2))
print(interval_score([13], [8], [12], 0.2), interval_score([10], [8], [12], 0.2))
print([round(1.2816 * 2 * np.sqrt(h), 2) for h in (1, 4)], round(1.96 * 2 * 2, 2))        # sigma 2, random-walk spread: sigma * sqrt(h)

y = np.log(pd.read_csv("AirPassengers.csv")["value"].to_numpy(float)); H = 12
origins = list(range(60, 133, 3)); calib = [o for o in origins if o <= 108]; test = [o for o in origins if o > 108]
print(len(calib), "calibration origins,", len(test), "test origins, horizon", H)

def hw(o): return ExponentialSmoothing(y[:o], trend="add", seasonal="add", seasonal_periods=12).fit().forecast(H)
err = {o: y[o:o + H] - hw(o) for o in calib}                                              # past forecast errors at each horizon
E = np.array([err[o] for o in calib])                                                      # shape: origins x horizon
lo_q, hi_q = np.quantile(E, 0.1, axis=0), np.quantile(E, 0.9, axis=0)                      # 10% and 90% error quantiles, per horizon

cover_e, cover_s, width_e, width_s, is_e, is_s = [], [], [], [], [], []
for o in test:
    f = hw(o); lo, hi = f + lo_q, f + hi_q; t = y[o:o + H]                                  # empirical-quantile 80% interval
    cover_e += list((t >= lo) & (t <= hi)); width_e += list(hi - lo); is_e.append(interval_score(t, lo, hi, 0.2))
    ci = SARIMAX(y[:o], order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False).get_forecast(H).conf_int(alpha=0.2)   # model-based 80% interval
    lo2, hi2 = ci[:, 0], ci[:, 1]
    cover_s += list((t >= lo2) & (t <= hi2)); width_s += list(hi2 - lo2); is_s.append(interval_score(t, lo2, hi2, 0.2))
print("80% intervals on the", len(test) * H, "test forecasts (log scale):")
print("model-based SARIMA   coverage", round(float(np.mean(cover_s)), 3), "| mean width", round(float(np.mean(width_s)), 3), "| interval score", round(float(np.mean(is_s)), 3))
print("empirical quantiles  coverage", round(float(np.mean(cover_e)), 3), "| mean width", round(float(np.mean(width_e)), 3), "| interval score", round(float(np.mean(is_e)), 3))
```

Output:
- `1.8 0.19999999999999996 0.2 1.8`: the pinball loss for $y=10$, with the forecast $q=8$ and $q=12$ at levels 0.9 and 0.1 (the second number is 0.2 apart from rounding).
- `1.8333 1.8333`: my pinball loss and scikit-learn's, on three points at level 0.9: identical.
- `best q for tau 0.9: 112.95 | 90th percentile of the sample: 112.94`: on 5,000 numbers drawn around 100, the forecast that minimises the 0.9-level pinball loss is the sample's 90th percentile (the tiny gap is the grid spacing).
- `14.0 4.0`: the interval score for an interval 8 to 12 at $\alpha=0.2$, when the truth is 13 and 10.
- `[2.56, 5.13] 7.84`: the half-widths of the tiny normal-interval example.
- `17 calibration origins, 8 test origins, horizon 12`, then the two lines for 80% intervals on the 96 test forecasts of the airline series (log scale): the model-based SARIMA interval covered **95.8%** of the truths, with a mean width of 0.169 and an interval score of 0.176; the empirical-quantile interval around a Holt-Winters forecast covered **79.2%**, with a mean width of 0.151 and an interval score of 0.204. In this test the model-based interval was too wide (it covered far more than its nominal 80%), and the empirical interval was close to nominal but had the worse interval score, because every miss is charged ten times its distance (for $\alpha=0.2$).

## 🔴 4. Deep / formal
**The pinball loss.** For a quantile level $\tau$ between 0 and 1, the loss of a forecast $q$ when the truth is $y$ is:

$$
\rho_\tau(y,q)=\max\big(\tau\,(y-q),\ (\tau-1)(y-q)\big)
$$

**Where:**
- $y$ is the true value and $q$ the forecast of the $\tau$-quantile. $y-q$ is the error (positive when the forecast was too low).
- $\tau$ ("tau") is the quantile level, for example 0.9 for the 90% quantile.
- When $y>q$ (too low) the loss is $\tau\,(y-q)$. When $y<q$ (too high) it is $(1-\tau)\,(q-y)$.
- **Matters:** the two sides are charged unequally. For $\tau=0.9$, being too low costs 9 times as much per unit as being too high, so the best forecast is pushed up until 90% of the outcomes lie below it. That is why the minimiser is the $\tau$-quantile. For $\tau=0.5$ both sides cost the same and it is the absolute error. Averaging the loss over many points gives the score to minimise.

**Example (tiny numbers):** $y=10$.
- $\tau=0.9$, $q=8$ (too low by 2): $0.9\times2=\mathbf{1.8}$. $q=12$ (too high by 2): $(1-0.9)\times2=\mathbf{0.2}$.
- $\tau=0.1$, $q=8$: $0.1\times2=\mathbf{0.2}$. $q=12$: $0.9\times2=\mathbf{1.8}$.
- Three points $3,5,9$ with $q=4$ at $\tau=0.9$: the losses are $(1-0.9)\times1=0.1$, $0.9\times1=0.9$ and $0.9\times5=4.5$, mean $\mathbf{1.833}$.
- Meaning: a high quantile forecast is punished mainly for sitting too low, and a low quantile forecast for sitting too high. The first two bullets are mirror images.

**An interval that widens with the horizon.** If the forecast error at horizon $h$ is roughly normal with spread $\hat\sigma_h$, a $(1-\alpha)$ interval is:

$$
\hat y_{t+h\mid t}\ \pm\ z_{1-\alpha/2}\,\hat\sigma_h,\qquad\text{for a random walk: }\ \hat\sigma_h=\sigma\sqrt h
$$

**Where:**
- $\hat y_{t+h\mid t}$ is the point forecast for $h$ steps ahead ([[Time Series]]).
- $\hat\sigma_h$ is the estimated spread (standard deviation) of the error at horizon $h$. $\sigma$ is the spread of one step.
- $z_{1-\alpha/2}$ is the number of standard deviations that leaves probability $\alpha/2$ in each tail of a bell curve: 1.2816 for an 80% interval and 1.96 for a 95% interval ([[Normal Distribution]]).
- **Matters:** the interval is only as good as $\hat\sigma_h$ and the bell-shape assumption. If the model is wrong or the noise has heavy tails, the real coverage differs from the nominal one. Without the $\sqrt h$ growth, the intervals would be much too narrow at long horizons for a random-walk-like series.

**Example (tiny numbers):** $\sigma=2$ per step.
- 80% interval at $h=1$: $1.2816\times2\times\sqrt1=\mathbf{2.56}$, so $\hat y\pm2.56$.
- 80% interval at $h=4$: $1.2816\times2\times2=\mathbf{5.13}$, so $\hat y\pm5.13$.
- 95% interval at $h=4$: $1.96\times2\times2=\mathbf{7.84}$.
- Meaning: four steps ahead the interval is twice as wide as one step ahead, and asking for 95% instead of 80% makes it about 1.5 times wider.

**The interval score.** To judge an interval $[l,u]$ with a nominal coverage of $1-\alpha$, the interval (Winkler) score adds the width to a penalty for every miss:

$$
\mathrm{IS}_\alpha(l,u;y)=(u-l)+\frac{2}{\alpha}\,(l-y)\,\mathbb 1[y<l]+\frac{2}{\alpha}\,(y-u)\,\mathbb 1[y>u]
$$

**Where:**
- $l$ and $u$ are the lower and upper ends of the interval, and $y$ is the true value. $u-l$ is the width.
- $\alpha$ is the miss probability (0.2 for an 80% interval). $\mathbb 1[\cdot]$ is 1 when its condition is true and 0 otherwise, so a penalty applies only when the truth falls outside.
- The factor $2/\alpha$ is the price per unit of distance outside the interval.
- **Matters:** it rewards narrow intervals (the width term) but charges heavily for misses, so there is no gain from shrinking the interval too far, and none from making it enormous. Without the penalty, the best interval would be a single point. Lower is better, and the score is in the units of the data.

**Example (tiny numbers):** interval $[8,12]$ and $\alpha=0.2$, so $2/\alpha=10$.
- Truth $y=10$ (inside): $4+0+0=\mathbf{4}$.
- Truth $y=13$ (1 above): $4+10\times1=\mathbf{14}$.
- Meaning: one miss of 1 unit costs 10 units here, two and a half times the width of the whole interval. This is why the empirical intervals in the code, which covered close to 80% but missed by more, scored worse.

**An interval from past errors.** Collect the errors of earlier forecasts made at horizon $h$, take their low and high quantiles, and add them to the new forecast:

$$
\Big[\ \hat y_{t+h\mid t}+q^{\text{lo}}_h,\ \ \hat y_{t+h\mid t}+q^{\text{hi}}_h\ \Big]
$$

**Where:**
- $\hat y_{t+h\mid t}$ is the new point forecast. $q^{\text{lo}}_h$ and $q^{\text{hi}}_h$ are the $\alpha/2$ and $1-\alpha/2$ quantiles of the past errors (truth minus forecast) at horizon $h$.
- For an 80% interval these are the 10% and 90% quantiles.
- **Matters:** no bell shape is assumed, and the width grows with $h$ by itself, because the past errors do. It needs enough past errors (the code had only 17 per horizon), and it assumes that tomorrow's errors will look like the past ones. If the series changes character, coverage will fail.

**Example (tiny numbers):** ten past errors at one horizon (on the log scale): $-0.04,\,-0.02,\,0.00,\,0.01,\,0.03,\,0.05,\,0.06,\,0.08,\,0.09,\,0.12$, and a new forecast of $5.00$.
- 10% quantile: $\mathbf{-0.022}$. 90% quantile: $\mathbf{0.093}$.
- Interval: $[5.00-0.022,\ 5.00+0.093]=\mathbf{[4.978,\ 5.093]}$, width $0.115$.
- Meaning: the past errors lean upward (the forecasts were more often too low), so the interval sits asymmetrically above the point forecast. A symmetric bell-shaped interval could not do that.

**Further notes:**
- **Proper scores:** a score is proper if the best way to get a good score is to report your true beliefs. The pinball loss and the interval score are proper for what they measure, and the CRPS is proper for whole distributions (from memory, Gneiting and Raftery, 2007; not checked).
- **Errors at different horizons are not independent.** A forecast that is too low at $h=1$ is probably too low at $h=2$. So the intervals are marginal (each horizon on its own), and they do not describe how whole forecast paths behave together.
- **Multiple quantiles can cross** (the 90% forecast below the 80% one) if trained separately, and are usually sorted afterwards.

**What came before, and how this improved on it:**
- **Before:** point forecasts alone, and "plus or minus two standard deviations" intervals computed by the model on the assumption that it is correct (the Box-Jenkins tradition; from memory).
- **Its limit:** the intervals are only as good as the model and the bell-shape assumption, and there was no routine way to check or score them.
- **What changed:** treat the forecast as a distribution, train and score it with quantile (pinball) losses (Koenker and Bassett, 1978; from memory), and judge it on calibration and sharpness with proper scores (Gneiting and Raftery, 2007; from memory), plus interval methods built from past errors.
- **How much better:** in the code above, on 96 test forecasts of the airline series, the model-based 80% interval **covered 95.8%** (too wide: far from the nominal 80%), while the interval built from past errors **covered 79.2%** and was narrower (0.151 against 0.169 on the log scale). So checking coverage revealed a miscalibration that the stated "80%" hid. But the empirical interval's interval score was **worse** (0.204 against 0.176), so closer coverage was not the same as a better score. One series, 8 overlapping test origins and only 17 calibration origins, so this is an illustration, not a ranking of the two methods.

## Related
- [[Time Series]] · [[Forecast Metrics]] · [[Time Series Cross-Validation]] · [[ARIMA]] · [[Exponential Smoothing]] · [[Forecasting Shoot-out]] · [[Calibration]] · [[Evaluation Metrics]] · [[Loss Function]] · [[Gradient Boosting]] · [[Monte Carlo]] · [[Normal Distribution]] · [[Expected Value]] · [[Variance]] · [[Bayesian Methods]]

## Sources
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, chapters on prediction intervals and distributional forecasts (free online; from memory)
- Koenker & Bassett, "Regression quantiles", 1978, and Gneiting & Raftery, "Strictly proper scoring rules, prediction, and estimation", 2007 (named from memory, not checked)
- Image: see the caption in section 2.
- Numbers in this note come from the code above (my arithmetic and my run on this machine).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/time-series Probabilistic forecast:::A forecast that gives a range or a whole spread of possible values (for example quantiles or a prediction interval) instead of one number.

#flashcards/ml/concept #flashcards/topic/time-series What are a quantile and an 80% prediction interval?
?
A quantile is a cut-off value. The 90% quantile is the number that 90% of the possible outcomes fall below. An 80% interval runs from the 10% quantile to the 90% quantile, so about 8 times in 10 the real value should land inside it. Intervals usually widen with the horizon (a fan chart).

#flashcards/ml/concept #flashcards/topic/time-series What do calibration and sharpness mean for a probabilistic forecast?
?
Calibrated: the observed frequencies match the stated probabilities (an 80% interval contains the truth about 80% of the time). Sharp: the intervals are as narrow as the data allows. Neither alone is enough, because an infinitely wide interval is perfectly calibrated and useless.

#flashcards/ml/formula #flashcards/topic/time-series Write the pinball (quantile) loss and explain each symbol.
?
$$\rho_\tau(y,q)=\max\big(\tau\,(y-q),\ (\tau-1)(y-q)\big)$$
- $y$: the true value. $q$: the forecast of the $\tau$-quantile. $y-q$: the error (positive when the forecast was too low).
- $\tau$: the quantile level, for example 0.9.
- Too low costs $\tau\,(y-q)$ and too high costs $(1-\tau)\,(q-y)$. For $\tau=0.9$ being too low costs 9 times as much, so the best forecast is pushed up until 90% of outcomes lie below it. For $\tau=0.5$ it is the absolute error.

#flashcards/ml/example #flashcards/topic/time-series For $y=10$, what is the pinball loss at $\tau=0.9$ for $q=8$ and $q=12$, and at $\tau=0.1$?::tau 0.9: q = 8 gives 0.9 × 2 = 1.8, and q = 12 gives 0.1 × 2 = 0.2. tau 0.1: q = 8 gives 0.1 × 2 = 0.2, and q = 12 gives 0.9 × 2 = 1.8. A high quantile is punished for sitting too low, a low quantile for sitting too high.

#flashcards/ml/formula #flashcards/topic/time-series Write a normal-theory prediction interval that widens with the horizon and explain each symbol.
?
$$\hat y_{t+h\mid t}\ \pm\ z_{1-\alpha/2}\,\hat\sigma_h,\qquad\text{for a random walk: }\ \hat\sigma_h=\sigma\sqrt h$$
- $\hat y_{t+h\mid t}$: the point forecast $h$ steps ahead. $\hat\sigma_h$: the estimated spread of the error at horizon $h$. $\sigma$: the spread of one step.
- $z_{1-\alpha/2}$: standard deviations leaving $\alpha/2$ in each tail: 1.2816 for 80%, 1.96 for 95%.
- It is only as good as $\hat\sigma_h$ and the bell-shape assumption. Heavy tails or a wrong model make real coverage differ from nominal.

#flashcards/ml/example #flashcards/topic/time-series With $\sigma=2$ per step, what are the 80% half-widths at $h=1$ and $h=4$, and the 95% half-width at $h=4$?::80%: 1.2816 × 2 × 1 = 2.56 at h = 1, and 1.2816 × 2 × 2 = 5.13 at h = 4. 95% at h = 4: 1.96 × 2 × 2 = 7.84. Four steps ahead is twice as wide as one step.

#flashcards/ml/formula #flashcards/topic/time-series Write the interval (Winkler) score and explain each symbol.
?
$$\mathrm{IS}_\alpha(l,u;y)=(u-l)+\frac{2}{\alpha}\,(l-y)\,\mathbb 1[y<l]+\frac{2}{\alpha}\,(y-u)\,\mathbb 1[y>u]$$
- $l$, $u$: the lower and upper ends of the interval. $y$: the true value. $u-l$: the width.
- $\alpha$: the miss probability (0.2 for an 80% interval). $\mathbb 1[\cdot]$ is 1 when its condition is true, so a penalty applies only outside the interval. $2/\alpha$: the price per unit outside.
- It rewards narrow intervals but charges heavily for misses, so neither a tiny nor an enormous interval wins. Lower is better.

#flashcards/ml/example #flashcards/topic/time-series For the interval 8 to 12 at $\alpha=0.2$, what is the interval score when the truth is 10 and when it is 13?::Truth 10 (inside): 4. Truth 13 (1 above): 4 + (2/0.2) × 1 = 14. One miss of 1 unit costs 10 units, two and a half times the interval's width.

#flashcards/ml/formula #flashcards/topic/time-series Write an interval built from past errors and explain each symbol.
?
$$\Big[\ \hat y_{t+h\mid t}+q^{\text{lo}}_h,\ \ \hat y_{t+h\mid t}+q^{\text{hi}}_h\ \Big]$$
- $\hat y_{t+h\mid t}$: the new point forecast. $q^{\text{lo}}_h$ and $q^{\text{hi}}_h$: the $\alpha/2$ and $1-\alpha/2$ quantiles of past errors (truth minus forecast) at horizon $h$.
- For an 80% interval these are the 10% and 90% quantiles.
- No bell shape is assumed and the width grows with $h$ on its own. It needs enough past errors and assumes future errors look like past ones.

#flashcards/ml/example #flashcards/topic/time-series Ten past errors $-0.04, -0.02, 0.00, 0.01, 0.03, 0.05, 0.06, 0.08, 0.09, 0.12$ and a forecast of 5.00: what is the 80% interval?::The 10% quantile is -0.022 and the 90% quantile is 0.093. The interval is [5.00 - 0.022, 5.00 + 0.093] = [4.978, 5.093], width 0.115. It sits asymmetrically because the past forecasts were more often too low.

#flashcards/ml/image #flashcards/topic/time-series ![[fan-chart-inflation.jpg]]
What does this picture show?
?
A fan chart: past values (a red line) followed by a central forecast with nested shaded bands, each a prediction interval of a different probability, which widen the further ahead the forecast goes. The data are hypothetical (Colombian total inflation, in percent).

#flashcards/ml/example #flashcards/topic/time-series On 96 test forecasts of the airline series, what were the coverage, width and interval score of the model-based SARIMA 80% interval and of the empirical-quantile 80% interval?::SARIMA: coverage 95.8%, mean width 0.169, interval score 0.176. Empirical: coverage 79.2%, width 0.151, interval score 0.204. The model-based interval was too wide, and closer coverage did not give the better score. One series, 8 overlapping origins. My run on this machine.

#flashcards/ml/cloze #flashcards/topic/time-series The forecast that minimises the pinball loss at level $\tau$ is the ==quantile of the matching level== of the outcomes, and a good interval is both ==calibrated== and ==sharp==.
