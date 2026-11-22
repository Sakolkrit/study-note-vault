---
tags: [time-series, forecasting, statistics, machine-learning, ML]
status: growing
level: 4
tier: 1
---

# Time Series

A time series is **a list of measurements taken one after another over time**, such as the temperature every day or the number of passengers every month. **Forecasting** means using the past values to guess the next ones.

## 🟢 1. Simple (age 12)
Imagine you write down the temperature outside at noon every day. Tomorrow's temperature is probably close to today's, and the summer is warmer than the winter. Because the numbers come in an order, **the order matters**: if you shuffled the days, the pattern would be gone. A time series is any list like that, and forecasting is guessing what comes next.

**Example:** You save pocket money and write down your total each week. If you saved a little more each week, you can guess next week's total by adding the same amount again. You used the past, and its order, to look ahead.

## 🔵 2. How it works
1. **Measure at regular times.** Every hour, day, week or month. How often you measure is called the **frequency**.
2. **Look for patterns.** Real series usually mix three things: a slow drift up or down (the **trend**), a cycle that repeats (the **seasonality**, like busy summers), and leftover wiggles (the **noise**). See [[Trend and Seasonality]].
3. **Neighbours look alike.** Today resembles yesterday, and this December resembles last December. How strongly a series resembles its own past is its [[Autocorrelation]]. This is the main thing that lets us forecast.
4. **Choose how far ahead.** The number of steps you look ahead is the **horizon**. Guessing the next month is easier than guessing a year ahead.
5. **Start with the simplest guesses.** "The same as the last value" (the **naive** forecast) and "the same as the same month last year" (the **seasonal naive** forecast). A fancy method is only worth using if it beats these.
6. **Never shuffle the data.** Ordinary [[Supervised Learning]] shuffles examples. For time series, you must keep the order, and test on the **last** stretch of time. See [[Time Series Cross-Validation]].
7. **Score the guesses** with an error measure such as the average miss. See [[Forecast Metrics]].
8. **Pick a method.** Classic ones are [[Exponential Smoothing]] and [[ARIMA]]. Machine-learning ones turn the past values into columns and use [[Gradient Boosting]] or a [[Random Forest]]. Deep-learning ones use a [[Recurrent Neural Network]] or an [[LSTM]].

![[keeling-curve-co2.svg]]
*The Keeling curve: monthly mean carbon dioxide measured at Mauna Loa, 1958 to 2025. Each red dot is one month and the blue line is the smoothed trend. The small panel shows the average seasonal pattern by month. A rising trend plus a yearly cycle is the classic shape of a time series. Image: Oeneis, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Mauna_Loa_CO2_monthly_mean_concentration.svg), CC BY-SA 4.0. The data comes from NOAA and the Scripps Institution of Oceanography, as credited on the image.*

## 🟠 3. Technical detail
- **Words to know:**
  - **Univariate** (one series) or **multivariate** (several series at once, for example temperature and sales).
  - **Exogenous variables**: outside information you know in advance, such as holidays or prices, that helps the forecast.
  - **Point forecast** (one number) or **interval forecast** (a range that should contain the truth, say, 80% of the time).
  - **Frequency** (the gap between measurements) and **horizon** (how many steps ahead). Forecasting several steps ahead is **multi-step** forecasting.
- **Why ordinary tools break:** ordinary [[Regression]] assumes each row is a separate, independent example. In a time series each row depends on the rows before it. Shuffling the rows, or testing on rows that sit in the middle of the training period, leaks the future into the past.
- **Turning a series into a table.** To use any ordinary model, build columns from the past: the value one step ago, two steps ago, and so on (**lag features**), plus calendar columns (month, weekday). See [[Feature Engineering]]. In pandas the tools are `shift` (lags), `rolling` (moving averages) and `resample` (change the frequency).
- **Baselines first.** The naive and seasonal naive forecasts cost nothing, and they are surprisingly hard to beat on short horizons. They are also the scale for the MASE error measure ([[Forecast Metrics]]).
- **Always split by time.** The training data must come before the test data. The code holds out the last 24 months.

```python
import numpy as np, pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing           # needs statsmodels; only the last model uses it

y = pd.read_csv("AirPassengers.csv")["value"].to_numpy(float)       # monthly airline passengers (thousands), Jan 1949 to Dec 1960
print(len(y), y[:6], y[-3:])

lag1 = lambda v: np.corrcoef(v[1:], v[:-1])[0, 1]                   # how much does each value resemble the one before it?
shuffled = np.random.default_rng(0).permutation(y)                  # same numbers, order destroyed
print(round(lag1(y), 3), round(lag1(shuffled), 3))

train, test = y[:120], y[120:]                                      # hold out the last 24 months: the future must stay unseen
slope, icpt = np.polyfit(np.arange(120), train, 1)
trend_only = icpt + slope * np.arange(120, 144)                     # "just draw a straight line through time"
seasonal_naive = train[-12:].tolist() * 2                           # next year = the same months of last year
hw = ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=12).fit().forecast(24)
rmse = lambda f: round(float(np.sqrt(((test - np.asarray(f)) ** 2).mean())), 1)
print("holdout RMSE | naive", rmse(np.full(24, train[-1])), "| straight line", rmse(trend_only), "| seasonal naive", rmse(seasonal_naive), "| Holt-Winters", rmse(hw))

x = np.array([10, 20, 30, 40, 12, 22, 32.]); m = 4                  # tiny example, season length 4
print(x[-1], x[len(x) + 1 - m - 1])                                 # naive: the last value; seasonal naive for h = 1: one season back
```

Output: `144 [112. 118. 132. 129. 121. 135.] [461. 390. 432.]`, then `0.96 0.017`, then `holdout RMSE | naive 137.3 | straight line 74.8 | seasonal naive 77.0 | Holt-Winters 32.5`, then `32.0 40.0`. The first line says the series has 144 monthly values. The second says each value correlates 0.96 with the one before it, but only 0.017 after shuffling the same numbers: the order carries the information. The third is the typical error (RMSE, see [[Forecast Metrics]]) of four forecasts of the last 24 months. The last line shows the tiny example: the naive forecast is 32, the seasonal naive forecast for the next step is 40. The code needs `statsmodels`; I ran it in a separate environment (see [[Forecasting Shoot-out]]).

## 🔴 4. Deep / formal
**What a forecast is.** A point forecast is the **expected value** of the future value, given everything seen so far:

$$
\hat y_{t+h\mid t}=\mathbb E\big[y_{t+h}\mid y_1,\dots,y_t\big]
$$

**Where:**
- $y_t$ is the value of the series at time $t$. $y_1,\dots,y_t$ is everything observed up to now.
- $h$ is the horizon: how many steps ahead.
- $\hat y_{t+h\mid t}$ ("y hat") is the forecast for time $t+h$, made at time $t$.
- $\mathbb E[\cdot\mid\cdot]$ is the [[Expected Value]], the average outcome given the past.
- **Matters:** the forecast may only use the past. If a model peeks at values after time $t$ (for example through a shuffled split), its test error looks better than its real error. And if the past tells nothing about the future, the best forecast is just the long-run average.

**Example (tiny numbers):** suppose a model says "each step the value grows by 2%", and the last value is $y_t=100$.
- $h=1$: $100\times1.02=\mathbf{102}$.
- $h=2$: $102\times1.02=\mathbf{104.04}$.
- $h=3$: $104.04\times1.02=\mathbf{106.12}$.
- Meaning: the further ahead you look, the more the guess depends on the rule continuing to hold.

**The two simplest forecasts.** For a season length $m$ (12 for monthly data with a yearly cycle):

$$
\text{naive: }\ \hat y_{t+h\mid t}=y_t,\qquad\qquad \text{seasonal naive: }\ \hat y_{t+h\mid t}=y_{t+h-m(k+1)},\quad k=\left\lfloor\frac{h-1}{m}\right\rfloor
$$

**Where:**
- $y_t$ is the last observed value, and $h$ is the horizon.
- $m$ is the number of steps in one season (4 for quarters, 12 for months, 7 for days of the week).
- $k$ counts how many whole seasons we must step back so that we land on a value that is already known. $\lfloor\cdot\rfloor$ means "round down".
- **Matters:** the naive forecast says "nothing changes", so it ignores trend and season. The seasonal naive forecast says "the pattern repeats", so it ignores the trend. Together they are the bar any other method must clear (see [[Forecast Metrics]] and the MASE measure).

**Example (tiny numbers):** quarterly data $10,20,30,40,12,22,32$, so $t=7$ and $m=4$.
- Naive: $\hat y=y_7=\mathbf{32}$ for every horizon.
- Seasonal naive, $h=1$: $k=0$, so $y_{7+1-4}=y_4=\mathbf{40}$ (the same quarter last year).
- Seasonal naive, $h=2$: $y_{5}=\mathbf{12}$. For $h=3$: $y_6=\mathbf{22}$.
- Seasonal naive, $h=5$: $k=1$, so $y_{7+5-8}=y_4=\mathbf{40}$ again.
- Meaning: the seasonal forecast keeps cycling through the last season's values.

**Further notes:**
- **Stochastic process view:** a time series is one sample from a random process $\{Y_t\}$. What we can say about the process depends on whether its statistics stay the same over time ([[Stationarity]]).
- **Not all time series are about forecasting.** Others are anomaly detection, classification of whole series, and filling gaps. They are outside this note.
- **Dependence is the point.** If the values were independent, the best forecast would be the same average every time. The more strongly neighbours resemble each other, the more the past is worth ([[Autocorrelation]]).

**What came before, and how this improved on it:**
- **Before:** fitting a straight line (or a polynomial) through time with ordinary regression ([[Linear Regression]]), and the naive guesses.
- **Its limit:** a line has no idea about the yearly cycle, and treats every point as independent of its neighbours.
- **What changed:** models that use the series' own past and its seasonal structure: exponential smoothing (Holt and Winters, 1950s and 1960; from memory, not checked) and ARIMA (Box and Jenkins, 1970; from memory) are the classic examples ([[Exponential Smoothing]], [[ARIMA]]).
- **How much better:** in the code above, on the last 24 months of the airline series (RMSE, thousands of passengers): naive 137.3, straight line 74.8, seasonal naive 77.0 and Holt-Winters **32.5**. So the seasonal exponential-smoothing model made about **2.3 times** smaller errors than the straight line (74.8 divided by 32.5). One series and one split, so it shows the pattern and does not prove it for every series.

## Related
- [[Trend and Seasonality]] · [[Stationarity]] · [[Autocorrelation]] · [[Exponential Smoothing]] · [[ARIMA]] · [[Time Series Cross-Validation]] · [[Forecast Metrics]] · [[Forecasting Shoot-out]] · [[Supervised Learning]] · [[Regression]] · [[Linear Regression]] · [[Feature Engineering]] · [[Gradient Boosting]] · [[Random Forest]] · [[Recurrent Neural Network]] · [[LSTM]] · [[Expected Value]] · [[Cross-Validation]] · [[Probabilistic Forecasting]]
- [[Anomaly Detection]] · [[Data Drift]]
## Sources
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (free online; from memory)
- Box, Jenkins & Reinsel, *Time Series Analysis: Forecasting and Control* (from memory); the airline passengers data comes from this book as distributed in R's `datasets` (from memory; the file I used came from the Rdatasets collection)
- Image: see the caption in section 2.
- Numbers in this note come from the code above (my arithmetic and my run on this machine).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/time-series Time series:::A list of measurements taken one after another over time, where the order carries information.

#flashcards/ml/concept #flashcards/topic/time-series Why can you not shuffle a time series before splitting it?
?
The order carries the information (neighbours resemble each other), and shuffling lets the model see the future during training. The airline series correlates 0.96 with its previous value, but only 0.017 after shuffling. Always train on the past and test on the later stretch.

#flashcards/ml/formula #flashcards/topic/time-series Write what a point forecast is and explain each symbol.
?
$$\hat y_{t+h\mid t}=\mathbb E\big[y_{t+h}\mid y_1,\dots,y_t\big]$$
- $y_t$: the value at time $t$. $y_1,\dots,y_t$: everything observed up to now.
- $h$: the horizon, how many steps ahead.
- $\hat y_{t+h\mid t}$: the forecast for time $t+h$ made at time $t$.
- $\mathbb E[\cdot\mid\cdot]$: the expected value given the past.
- The forecast may use only the past. Peeking at later values makes test error look better than it really is.

#flashcards/ml/example #flashcards/topic/time-series A model grows the series 2% each step from $y_t=100$: what are the forecasts for $h=1,2,3$?::102, then 104.04, then 106.12 (each step multiplies by 1.02). The further ahead, the more the guess relies on the rule continuing to hold.

#flashcards/ml/formula #flashcards/topic/time-series Write the naive and seasonal naive forecasts and explain each symbol.
?
$$\text{naive: }\ \hat y_{t+h\mid t}=y_t,\qquad\qquad \text{seasonal naive: }\ \hat y_{t+h\mid t}=y_{t+h-m(k+1)},\quad k=\left\lfloor\frac{h-1}{m}\right\rfloor$$
- $y_t$: the last observed value. $h$: the horizon.
- $m$: the number of steps in one season (4 for quarters, 12 for months, 7 for days of the week).
- $k$: how many whole seasons to step back to land on a known value. The floor rounds down.
- Naive ignores trend and season. Seasonal naive ignores the trend. Together they are the bar other methods must clear.

#flashcards/ml/example #flashcards/topic/time-series Quarterly data 10, 20, 30, 40, 12, 22, 32 ($m=4$): what are the naive and the seasonal naive forecasts for $h=1$ and $h=5$?::Naive: 32 for every horizon. Seasonal naive h=1: y(7+1-4) = y4 = 40. For h=5, k=1 so y(7+5-8) = y4 = 40 again.

#flashcards/ml/image #flashcards/topic/time-series ![[keeling-curve-co2.svg]]
What does this picture show?
?
Monthly mean CO2 measured at Mauna Loa, 1958 to 2025. Each red dot is a month, the blue line is the smoothed trend, and the small panel shows the average seasonal pattern by month. A rising trend plus a yearly cycle.

#flashcards/ml/cloze #flashcards/topic/time-series A fancy forecasting method is only worth using if it beats the ==naive== and ==seasonal naive== forecasts.

#flashcards/ml/concept #flashcards/topic/time-series Why do ordinary regression and ordinary cross-validation fail on time series?
?
They assume each row is an independent example. In a time series each row depends on the rows before it, so random splits leak the future into the training data.

#flashcards/ml/example #flashcards/topic/time-series On the last 24 months of the airline series, what were the test errors of the naive, straight-line, seasonal naive and Holt-Winters forecasts?::RMSE 137.3, 74.8, 77.0 and 32.5 (thousands of passengers). Holt-Winters was about 2.3 times better than the straight line (74.8 over 32.5). One series and one split. My run on this machine.
