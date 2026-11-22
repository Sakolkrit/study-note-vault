---
tags: [time-series, forecasting, ARIMA, autoregressive, moving-average, box-jenkins, statistics, ML]
status: growing
level: 4
tier: 3
---

# ARIMA

ARIMA forecasts a [[Time Series]] as **a weighted mix of its own recent values and its recent forecasting mistakes**, after differencing the series until it is [[Stationarity|stationary]]. The name stands for **A**uto**R**egressive **I**ntegrated **M**oving **A**verage.

## 🟢 1. Simple (age 12)
To guess tomorrow's value, ARIMA looks at two things: **how big the last few values were**, and **how wrong your last few guesses were**. If you guessed too low yesterday, you probably nudge tomorrow's guess up a bit. If the series has been rising steadily, it first looks at the *steps* (how much it rose each time) instead of the raw values. Putting those pieces together gives a prediction.

**Example:** A shop sold 100 phone cases yesterday and 98 the day before, so sales are rising by 2 a day. You guess that the rise continues but a little weaker, say by 1. So you guess $100+1=101$ for today.

## 🔵 2. How it works
1. **Make the series stationary.** Take differences until it stops drifting (the "I" part, the number of differences is $d$). See [[Stationarity]]. For yearly cycles, also take a seasonal difference.
2. **The "AR" part (autoregressive):** predict the next value from a weighted sum of the last $p$ values. It captures "values like to stay near their recent neighbours" ([[Autocorrelation]]).
3. **The "MA" part (moving average):** also adjust the prediction using the last $q$ forecasting mistakes ("shocks"). It captures "if I was too low last time, the same reason is probably still pushing". (This is **not** the smoothing moving average from [[Trend and Seasonality]]; it is a weighted sum of past errors.)
4. **Choose $p$, $d$ and $q$.** Look at the ACF and PACF plots for hints, or try several choices and keep the best one by a score that punishes unnecessary complexity (the AIC).
5. **Fit the weights** from the data. The computer tries weights until the past errors are as small as possible (in effect, [[Maximum Likelihood]]).
6. **Check the leftovers.** The forecasting mistakes should look like random noise, with no pattern left in their autocorrelation.
7. **Forecast.** Predict one step, then feed the prediction in as if it were data to predict the next step, and so on. Long forecasts slide back toward the average (or continue the last drift if the series was differenced).
8. **Add seasons.** Seasonal ARIMA (**SARIMA**) repeats the AR, differencing and MA ideas at the season length (every 12 steps for monthly data).

![[arima-forecast-temperature.png]]
*An ARIMA forecast. The black line is the observed series (120 points). The blue line is the forecast for the next steps, from an ARIMA(0,1,1) model with drift, as the picture's title says. The shaded bands around it show the uncertainty, which gets wider the further ahead the forecast goes. Image: Lovepeacejoy404, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Previsione_arima_temperatura_media_italiana.png), CC BY-SA 4.0 (a 500-pixel-wide version of the file; the Commons description says it was made in RStudio).*

## 🟠 3. Technical detail
- **Notation:** ARIMA($p,d,q$): $p$ autoregressive terms, $d$ differences, $q$ moving-average terms. A seasonal model is written ARIMA($p,d,q$)($P,D,Q$)$_m$ with season length $m$.
- **Special cases:** ARIMA(0,1,0) is a random walk (the naive forecast). ARIMA(1,0,0) is a plain AR(1). ARIMA(0,1,1) gives the same forecasts as simple exponential smoothing ([[Exponential Smoothing]]; standard result, from memory).
- **The airline model:** ARIMA(0,1,1)(0,1,1)$_{12}$ on the log of the series. Box and Jenkins used it on the classic airline passengers data, and it is the usual benchmark for monthly seasonal series (from memory).
- **Identification rules (classic):** for AR($p$), the ACF fades and the PACF cuts off after lag $p$; for MA($q$), the ACF cuts off after lag $q$ and the PACF fades ([[Autocorrelation]]; from memory). Today, people often search over a small grid and compare AIC.
- **AIC:** a score that rewards fit and punishes the number of fitted parameters. Lower is better. Comparing AIC is only valid between models fitted to the same (differenced) data.
- **Fitting:** by [[Maximum Likelihood]]. A quick alternative for pure AR models is ordinary least squares on lagged values, which is simply [[Linear Regression]] on the series' own past (the code compares the two).
- **Strengths:** a compact, well-understood model; handles trend and season through differencing; gives prediction intervals; works with little data.
- **Weak spots:** one series at a time (no cross-series learning), linear only, needs a stationary series, needs careful order choice, struggles with several seasons or sudden changes. On the airline holdout in the code it was beaten by Holt-Winters, which shows that "more famous" does not mean "better".
- **Outside information:** ARIMAX adds extra input variables (from memory).
- **Library note:** the code uses `statsmodels` (installed in a separate environment; see [[Forecasting Shoot-out]]).

```python
import numpy as np, pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

rng = np.random.default_rng(0); n = 3000; e = rng.normal(size=n); y = np.zeros(n)
for t in range(2, n): y[t] = 0.6 * y[t - 1] - 0.3 * y[t - 2] + e[t]  # simulate AR(2): true phi = (0.6, -0.3)
X = np.column_stack([y[1:-1], y[:-2]]); phi_ls = np.linalg.lstsq(X, y[2:], rcond=None)[0]   # fit it by plain least squares
fit = ARIMA(y, order=(2, 0, 0), trend="n").fit()
print("least squares:", phi_ls.round(3), "| statsmodels ARIMA(2,0,0):", fit.params[:2].round(3))

mu, phi, last = 0.0, 0.8, 10.0                                        # AR(1) forecasts slide back toward the mean
print([round(mu + phi ** h * (last - mu), 3) for h in (1, 2, 3)])

ap = pd.read_csv("AirPassengers.csv")["value"].to_numpy(float); ly = np.log(ap); train, test = ly[:120], ap[120:]
rmse = lambda f: round(float(np.sqrt(((test - np.asarray(f)) ** 2).mean())), 1)
for order, seas in [((0, 1, 1), (0, 1, 1, 12)), ((1, 1, 0), (0, 1, 1, 12)), ((1, 1, 1), (0, 1, 1, 12)), ((0, 1, 0), (0, 1, 0, 12))]:
    m = SARIMAX(train, order=order, seasonal_order=seas).fit(disp=False)
    print(order, seas, "AIC", round(m.aic, 1), "| holdout RMSE", rmse(np.exp(m.forecast(24))))
print("seasonal naive", rmse(ap[108:120].tolist() * 2))
```

Output:
- `least squares: [ 0.594 -0.287] | statsmodels ARIMA(2,0,0): [ 0.594 -0.286]`: I simulated an AR(2) series with the true weights $0.6$ and $-0.3$. Plain least squares and the library's maximum likelihood both recover them closely.
- `[8.0, 6.4, 5.12]`: the AR(1) forecast sliding back toward the mean (the tiny example below).
- Four lines for the airline series (log scale; first 120 months for fitting): `(0, 1, 1) (0, 1, 1, 12) AIC -389.0 | holdout RMSE 43.2`, `(1, 1, 0) (0, 1, 1, 12) AIC -388.6 | holdout RMSE 45.5`, `(1, 1, 1) (0, 1, 1, 12) AIC -387.3 | holdout RMSE 40.0` and a no-parameter `(0, 1, 0) (0, 1, 0, 12) AIC -353.4 | holdout RMSE 75.2`. The airline model has the lowest (best) AIC, and the extra term in the third model did not help the AIC. The last model has no fitted weights (it repeats last year's pattern plus the latest change) and is much worse.
- `seasonal naive 77.0`: the same baseline as before.

## 🔴 4. Deep / formal
**The ARMA equation.** For a stationary series (after any differencing), the value at time $t$ is:

$$
y_t=c+\phi_1y_{t-1}+\dots+\phi_py_{t-p}+\varepsilon_t+\theta_1\varepsilon_{t-1}+\dots+\theta_q\varepsilon_{t-q}
$$

**Where:**
- $y_t$ is the value at time $t$, and $y_{t-1},\dots,y_{t-p}$ are the previous $p$ values.
- $\phi_1,\dots,\phi_p$ ("phi") are the autoregressive weights. $c$ is a constant.
- $\varepsilon_t$ ("epsilon") is the random shock at time $t$: new, unpredictable noise with an average of 0. $\varepsilon_{t-1},\dots,\varepsilon_{t-q}$ are the previous $q$ shocks.
- $\theta_1,\dots,\theta_q$ ("theta") are the moving-average weights.
- **Matters:** the AR part lets the series remember its own past, and the MA part lets a one-time shock echo for a few steps and then vanish. If all the $\phi$ are zero, only the echo is left (an MA model, with a memory of exactly $q$ steps). If all the $\theta$ are zero, it is a pure AR model, which can be fitted by ordinary regression.

**Example (tiny numbers):** $c=0$, $\phi_1=0.6$, $\theta_1=0.5$, with $y_{t-1}=10$ and last step's shock $\varepsilon_{t-1}=2$.
- Expected value (future shocks average 0): $0.6\times10+0.5\times2=6+1=\mathbf{7}$.
- Meaning: 6 comes from the previous value, and 1 from the echo of last step's surprise. If the shock had been $-2$, the forecast would be $6-1=5$.

**The "I": differencing, written with the backshift operator.** Let $B$ be the operator that steps back one time unit, $By_t=y_{t-1}$. An ARIMA($p,d,q$) model says that the series differenced $d$ times follows an ARMA model:

$$
\Big(1-\phi_1B-\dots-\phi_pB^p\Big)(1-B)^d\,y_t=c+\Big(1+\theta_1B+\dots+\theta_qB^q\Big)\varepsilon_t
$$

**Where:**
- $B$ is the backshift operator: $B^ky_t=y_{t-k}$. $(1-B)y_t=y_t-y_{t-1}$ is the first difference, so $(1-B)^d$ is the $d$-th difference.
- The left bracket $(1-\phi_1B-\dots)$ contains the AR weights. The right bracket $(1+\theta_1B+\dots)$ contains the MA weights.
- $d$ is the number of differences (0, 1 or at most 2 in practice).
- **Matters:** the forecasts are made for the differenced series, then **added back up** (integrated) to give forecasts of the original series. If $d=1$, the model forecasts the next *change* and adds it to the last value. This is why a differenced model can keep rising or falling, while an undifferenced one slides back to a fixed mean.

**Example (tiny numbers):** ARIMA(1,1,0) with $\phi_1=0.5$ and no constant. The last two values are $y_{t-2}=98$ and $y_{t-1}=100$.
- Last change: $100-98=\mathbf{2}$.
- Expected next change: $0.5\times2=\mathbf{1}$.
- Forecast: $100+1=\mathbf{101}$.
- Meaning: the series was rising by 2 and the model expects the rise to weaken to 1. The next step would be $101+0.5\times1=101.5$.

**How AR forecasts behave over several steps.** For an AR(1) with mean $\mu$ (the series fluctuates around $\mu$), the forecast $h$ steps ahead is:

$$
\hat y_{t+h\mid t}=\mu+\phi^{\,h}\,(y_t-\mu)
$$

**Where:**
- $\mu$ is the long-run average of the series. $y_t-\mu$ is how far the last value sits from it.
- $\phi$ is the autoregressive weight with $|\phi|<1$ and $h$ is the horizon.
- **Matters:** each step multiplies the gap from the mean by $\phi$, so the forecast slides back toward $\mu$ and, for a long horizon, simply equals $\mu$. The "memory" of the last value fades geometrically, just like the autocorrelation $\phi^k$ ([[Autocorrelation]]).

**Example (tiny numbers):** $\mu=0$, $\phi=0.8$, $y_t=10$.
- $h=1$: $0.8\times10=\mathbf{8}$.
- $h=2$: $0.8^2\times10=\mathbf{6.4}$.
- $h=3$: $0.8^3\times10=\mathbf{5.12}$ (the code printed `[8.0, 6.4, 5.12]`).
- Meaning: the forecast decays toward 0, never rising again.

**Choosing between models: AIC.** The Akaike information criterion scores a fitted model:

$$
\mathrm{AIC}=2k-2\ln\hat L
$$

**Where:**
- $k$ is the number of parameters that were fitted (the $\phi$'s, $\theta$'s, constant and the noise variance).
- $\hat L$ is the maximised likelihood: how probable the data is under the fitted model ([[Maximum Likelihood]]). $\ln\hat L$ is its logarithm.
- **Matters:** the first term charges for complexity and the second rewards a good fit. Lower AIC is better. Without the first term, you would always pick the model with the most parameters ([[Overfitting]]). It is valid only for models fitted to exactly the same data (the same differencing).

**Example (tiny numbers):** model A has $k=2$ and $\ln\hat L=-50$. Model B has $k=4$ and $\ln\hat L=-49$.
- A: $2\times2-2\times(-50)=4+100=\mathbf{104}$.
- B: $2\times4-2\times(-49)=8+98=\mathbf{106}$.
- Meaning: B fits slightly better, but not enough to pay for two extra parameters, so A wins.

**Further notes:**
- **Why "integrated":** summing (integrating) the differences gives back the original series, which is the opposite of differencing.
- **Invertibility and stationarity:** the AR weights must keep the process stationary and the MA weights must be "invertible" (so that old shocks can be recovered from the data). Software checks these (from memory).
- **Intervals:** forecast uncertainty grows with the horizon, because each step adds a new unknown shock.

**What came before, and how this improved on it:**
- **Before:** exponential smoothing methods and separate AR models (Yule's autoregressive model for sunspots, 1920s) and MA models (Slutsky, 1930s), each on its own (all from memory, not checked).
- **Its limit:** each family covered only part of the cases, and there was no routine way to choose and check a model.
- **What changed:** Box and Jenkins (1970; from memory) combined AR, MA and differencing into one family, and a three-step routine: identify (from the ACF and PACF), estimate and check the residuals.
- **How much better:** in the code above, on the last 24 months of the airline series (RMSE), the airline model made **43.2** against **77.0** for seasonal naive, about 1.8 times smaller. It was **not** the best here: Holt-Winters made 32.5 on the same split ([[Exponential Smoothing]]). One series and one split, so the ranking could change on other data.

## Related
- [[Time Series]] · [[Trend and Seasonality]] · [[Stationarity]] · [[Autocorrelation]] · [[Exponential Smoothing]] · [[Forecast Metrics]] · [[Time Series Cross-Validation]] · [[Forecasting Shoot-out]] · [[Maximum Likelihood]] · [[Linear Regression]] · [[Overfitting]] · [[Markov Chain]] · [[Autoregressive Generation]] · [[Probabilistic Forecasting]]

## Sources
- Box, Jenkins & Reinsel, *Time Series Analysis: Forecasting and Control* (from memory)
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, ARIMA chapters (free online; from memory)
- Image: see the caption in section 2.
- Numbers in this note come from the code above (my arithmetic and my run on this machine).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/time-series ARIMA(p, d, q):::A model that differences the series d times, then predicts each value from the last p values (AR part) and the last q forecasting errors (MA part).

#flashcards/ml/concept #flashcards/topic/time-series What do the AR and MA parts of ARIMA each capture?
?
The AR part lets a value depend on its own recent values (memory of the past). The MA part lets a one-time shock echo for q steps and then vanish (a weighted sum of past forecasting errors, not a smoothing average). The I part is the number of differences used to make the series stationary.

#flashcards/ml/formula #flashcards/topic/time-series Write the ARMA equation and explain each symbol.
?
$$y_t=c+\phi_1y_{t-1}+\dots+\phi_py_{t-p}+\varepsilon_t+\theta_1\varepsilon_{t-1}+\dots+\theta_q\varepsilon_{t-q}$$
- $y_t$: the value at time $t$. $y_{t-1},\dots,y_{t-p}$: the previous $p$ values. $c$: a constant.
- $\phi_1,\dots,\phi_p$: the autoregressive weights. $\theta_1,\dots,\theta_q$: the moving-average weights.
- $\varepsilon_t$: the random shock at time $t$ (new noise with average 0). $\varepsilon_{t-1},\dots$: the previous $q$ shocks.
- With all $\phi=0$ only the echo of shocks is left (an MA model). With all $\theta=0$ it is a pure AR model, fitted by ordinary regression.

#flashcards/ml/example #flashcards/topic/time-series ARMA with $c=0$, $\phi_1=0.6$, $\theta_1=0.5$, $y_{t-1}=10$ and $\varepsilon_{t-1}=2$: what is the expected $y_t$?::0.6 × 10 + 0.5 × 2 = 6 + 1 = 7 (future shocks average 0). With a shock of -2 it would be 6 - 1 = 5.

#flashcards/ml/formula #flashcards/topic/time-series Write the ARIMA model with the backshift operator and explain each symbol.
?
$$\Big(1-\phi_1B-\dots-\phi_pB^p\Big)(1-B)^d\,y_t=c+\Big(1+\theta_1B+\dots+\theta_qB^q\Big)\varepsilon_t$$
- $B$: the backshift operator, $B^ky_t=y_{t-k}$. $(1-B)y_t=y_t-y_{t-1}$ is the first difference, so $(1-B)^d$ is the $d$-th difference.
- The left bracket holds the AR weights. The right bracket holds the MA weights. $d$: the number of differences.
- Forecasts are made for the differenced series and then added back up (integrated). With $d=1$ the model forecasts the next change and adds it to the last value, so it can keep rising or falling.

#flashcards/ml/example #flashcards/topic/time-series ARIMA(1,1,0) with $\phi_1=0.5$, last values 98 then 100: what is the forecast?::Last change 100 - 98 = 2. Expected next change 0.5 × 2 = 1. Forecast 100 + 1 = 101. The following step would be 101 + 0.5 × 1 = 101.5.

#flashcards/ml/formula #flashcards/topic/time-series Write the multi-step AR(1) forecast and explain each symbol.
?
$$\hat y_{t+h\mid t}=\mu+\phi^{\,h}\,(y_t-\mu)$$
- $\mu$: the long-run average. $y_t-\mu$: how far the last value is from it.
- $\phi$: the autoregressive weight, with $|\phi|<1$. $h$: the horizon.
- Each step multiplies the gap from the mean by $\phi$, so the forecast slides back toward $\mu$. The memory of the last value fades like the autocorrelation $\phi^k$.

#flashcards/ml/example #flashcards/topic/time-series AR(1) with $\mu=0$, $\phi=0.8$, $y_t=10$: what are the forecasts for $h=1,2,3$?::8, 6.4 and 5.12 (0.8, 0.64 and 0.512 times 10). They decay toward the mean 0.

#flashcards/ml/formula #flashcards/topic/time-series Write the AIC and explain each symbol.
?
$$\mathrm{AIC}=2k-2\ln\hat L$$
- $k$: the number of fitted parameters ($\phi$'s, $\theta$'s, constant and noise variance).
- $\hat L$: the maximised likelihood, how probable the data is under the fitted model. $\ln\hat L$ is its logarithm.
- The first term charges for complexity and the second rewards fit. Lower is better. It is valid only for models fitted to the same data (same differencing).

#flashcards/ml/example #flashcards/topic/time-series Model A has $k=2$ and $\ln\hat L=-50$, model B has $k=4$ and $\ln\hat L=-49$: which has the lower AIC?::A: 2×2 + 100 = 104. B: 2×4 + 98 = 106. A wins, because B's slightly better fit does not pay for two extra parameters.

#flashcards/ml/image #flashcards/topic/time-series ![[arima-forecast-temperature.png]]
What does this picture show?
?
An ARIMA(0,1,1) forecast with drift. The black line is the observed series (120 points), the blue line the forecast, and the shaded bands the uncertainty, which widens the further ahead the forecast goes.

#flashcards/ml/example #flashcards/topic/time-series On the last 24 months of the airline series, what were the RMSEs of the ARIMA(0,1,1)(0,1,1)12 model (on logs), seasonal naive and Holt-Winters?::43.2, 77.0 and 32.5. The airline model was about 1.8 times better than seasonal naive but was beaten by Holt-Winters on this split. One series and one split. My run on this machine.
