---
tags: [time-series, forecasting, smoothing, holt-winters, ETS, statistics, ML]
status: growing
level: 4
tier: 2
---

# Exponential Smoothing

Exponential smoothing forecasts by **keeping a running estimate of the series' level (and, if needed, its trend and its season) and updating it a little with every new value**. Recent values count the most, and old values fade away exponentially.

## 🟢 1. Simple (age 12)
Imagine you want to guess how tall a plant is. Every week you measure it, but your ruler is shaky, so each measurement is a bit off. You keep a "best guess". Each week, you move your best guess **part of the way** toward the new measurement. A big step trusts the new measurement a lot, a small step trusts your old guess a lot. Over many weeks, old measurements matter less and less, because every new step keeps shrinking their say.

**Example:** Your best guess for the plant is 10 cm. This week's ruler says 12 cm. If you move halfway, your new guess is 11 cm. Next week the ruler says 14, so you move halfway again, to 12.5 cm.

## 🔵 2. How it works
1. **Keep a "level":** your current best estimate of where the series sits, ignoring the noise.
2. **Update it each step:** new level = a fraction of the newest value plus the rest of the old level. The fraction is called $\alpha$ ("alpha") and lies between 0 and 1. A large $\alpha$ reacts quickly, a small $\alpha$ smooths a lot.
3. **Forecast:** with only a level, the forecast for every future step is the level. This is **simple exponential smoothing**.
4. **Add a trend.** Keep a second running number, the "trend" (how much the level rises per step), and update it the same way with its own fraction $\beta$ ("beta"). Forecast = level + steps ahead × trend. This is **Holt's method**.
5. **Add a season.** Keep one running number for each position in the cycle (each month) with its own fraction $\gamma$ ("gamma"). Forecast = level + trend + the seasonal number for that month. This is **Holt-Winters**.
6. **The seasonal part can add or multiply**, just like in [[Trend and Seasonality]]: use the multiplicative form when the swings grow with the level.
7. **Choose the fractions** $\alpha,\beta,\gamma$ by trying values and keeping the ones with the smallest past errors.
8. **Why "exponential":** unroll the update and you find that the newest value has weight $\alpha$, the one before has $\alpha(1-\alpha)$, the one before that $\alpha(1-\alpha)^2$, and so on, shrinking by the same factor each step.

![[exponential-moving-average-weights.png]]
*The weights that exponential smoothing gives to past values, for a span of 15 steps. The newest value (step 0) has a weight of about 0.125, and every older step has 0.875 times the weight of the step after it, so the bars shrink geometrically and never drop to zero. The axis labels use decimal commas. Image: Д. Ильин (vectorization), [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Exponential_moving_average_weights_N%3D15.svg), CC0 (this is a 960-pixel PNG render of the SVG file).*

## 🟠 3. Technical detail
- **Compared with a plain moving average:** a moving average gives every value in a window the same weight and drops older values completely. Exponential smoothing never drops anything but gives smoothly smaller weights, and it needs only the last level, not the whole window.
- **The three parts:** level (the smoothed value), trend (the slope) and season (the cycle). Variants combine none, additive or multiplicative versions of trend and season, and the "damped" trend (from memory) shrinks the trend toward zero over the horizon so that long forecasts do not rise forever.
- **Starting values:** the first level (and trend, and seasonal values) must be chosen. Statistical software estimates them with the fractions.
- **ETS framework** (from memory): the "error, trend, season" family puts all these methods in a statistical model with noise, which gives prediction intervals and lets you compare models with AIC.
- **Strengths:** it is fast, needs little data, is easy to explain and is a strong baseline that often beats complicated methods at short horizons.
- **Weak spots:** it has no way to use outside information (holidays, prices), assumes the pattern changes slowly, and a long forecast simply extends the last trend and season.
- **Relation to ARIMA:** simple exponential smoothing gives the same forecasts as an ARIMA(0,1,1) model, and Holt's method matches an ARIMA(0,2,2) model (standard results; from memory). See [[ARIMA]].

```python
import numpy as np, pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing

def ses(y, alpha, l0):                                               # level l_t = alpha*y_t + (1-alpha)*l_{t-1}
    l = l0
    for v in y: l = alpha * v + (1 - alpha) * l
    return l                                                         # the forecast for every future step

def holt(y, alpha, beta, l0, b0):                                    # adds a trend b_t
    l, b = l0, b0
    for v in y:
        l_new = alpha * v + (1 - alpha) * (l + b)
        b = beta * (l_new - l) + (1 - beta) * b; l = l_new
    return l, b                                                      # forecast for h steps ahead: l + h*b

print(ses([12, 14], 0.5, 10), holt([12, 14], 0.5, 0.5, 10, 1))      # tiny numbers

y = pd.read_csv("AirPassengers.csv")["value"].to_numpy(float)
train, test = y[:120], y[120:]
rmse = lambda f: round(float(np.sqrt(((test - np.asarray(f)) ** 2).mean())), 1)

mine = ses(train, 0.3, train[0])                                     # check my loops against statsmodels with the same fixed settings
lib = SimpleExpSmoothing(train, initialization_method="known", initial_level=train[0]).fit(smoothing_level=0.3, optimized=False).forecast(1)[0]
print(round(mine, 6), round(float(lib), 6))
l, b = holt(train, 0.3, 0.1, train[0], train[1] - train[0])
lib = Holt(train, initialization_method="known", initial_level=train[0], initial_trend=train[1] - train[0]).fit(smoothing_level=0.3, smoothing_trend=0.1, optimized=False).forecast(3)
print([round(l + h * b, 4) for h in (1, 2, 3)], lib.round(4).tolist())

w = lambda a, n: [round(a * (1 - a) ** i, 3) for i in range(n)]      # weight on the value i steps ago
print("weights alpha 0.3:", w(0.3, 5), "| last 12 values carry", round(1 - 0.7 ** 12, 3), "of the total")

hw = ExponentialSmoothing(train, trend="add", seasonal="mul", seasonal_periods=12).fit()
print("holdout RMSE | naive", rmse(np.full(24, train[-1])), "| seasonal naive", rmse(train[-12:].tolist() * 2),
      "| simple smoothing", rmse(np.full(24, SimpleExpSmoothing(train).fit().forecast(1)[0])),
      "| Holt (trend)", rmse(Holt(train).fit().forecast(24)), "| Holt-Winters", rmse(hw.forecast(24)))
```

Output:
- `12.5 (13.375, 1.5625)`: the tiny examples (level after `12, 14` with $\alpha=0.5$; level and trend for Holt).
- `366.056553 366.056553` and `[372.6112, 371.5994, 370.5875] [372.6112, 371.5994, 370.5875]`: my loops give exactly the same forecasts as the library when both use the same fixed settings.
- `weights alpha 0.3: [0.3, 0.21, 0.147, 0.103, 0.072] | last 12 values carry 0.986 of the total`.
- `holdout RMSE | naive 137.3 | seasonal naive 77.0 | simple smoothing 137.3 | Holt (trend) 97.7 | Holt-Winters 32.5`: the errors on the last 24 months of the airline series. Simple smoothing equals the naive forecast because the fit chose $\alpha=1$ (the series trends, so "smoothing" does not help).

## 🔴 4. Deep / formal
**Simple exponential smoothing.** With a smoothing fraction $\alpha$, the level after seeing $y_t$ is:

$$
\ell_t=\alpha\,y_t+(1-\alpha)\,\ell_{t-1},\qquad \hat y_{t+h\mid t}=\ell_t
$$

**Where:**
- $y_t$ is the newest value. $\ell_{t-1}$ is the level before it, and $\ell_t$ is the updated level.
- $\alpha\in(0,1)$ is how much weight the newest value gets.
- $\hat y_{t+h\mid t}$ is the forecast $h$ steps ahead ([[Time Series]]). With no trend or season it is the same for every $h$.
- **Matters:** $\alpha=1$ gives the naive forecast (only the last value matters). A small $\alpha$ gives a very smooth level that reacts slowly. Without the $(1-\alpha)\ell_{t-1}$ part, there would be no memory at all.

**Example (tiny numbers):** $\alpha=0.5$, starting level $\ell_0=10$, then $y_1=12$ and $y_2=14$.
- $\ell_1=0.5\times12+0.5\times10=\mathbf{11}$.
- $\ell_2=0.5\times14+0.5\times11=\mathbf{12.5}$.
- Forecast: $\mathbf{12.5}$ for every future step.
- Meaning: the level moved from 10 toward the new values but lags behind them.

**The weights are exponential.** Substituting the update into itself again and again gives:

$$
\ell_t=\alpha\sum_{i=0}^{\infty}(1-\alpha)^{\,i}\,y_{t-i}\quad\text{(plus a vanishing term from the starting level)}
$$

**Where:**
- $i$ counts how many steps back: $i=0$ is the newest value.
- $\alpha(1-\alpha)^i$ is the weight of the value $i$ steps ago. These weights add up to 1.
- **Matters:** the weights shrink by the factor $(1-\alpha)$ each step back, a geometric (exponential) fade. This is why it is a smarter version of a moving average. A larger $\alpha$ makes the fade faster.

**Example (tiny numbers):** $\alpha=0.3$.
- Weights: $0.3,\ 0.3\times0.7=\mathbf{0.21},\ 0.21\times0.7=\mathbf{0.147},\ \mathbf{0.103},\ \mathbf{0.072}$.
- The last 12 values carry $1-0.7^{12}=\mathbf{0.986}$ of the total weight.
- Meaning: the newest value counts for 30%, the value 4 steps back for about 7%, and anything older than a year of monthly data counts for almost nothing.

**Holt's linear method (level and trend).** Add a trend $b_t$ (the rise per step) with its own fraction $\beta$:

$$
\ell_t=\alpha y_t+(1-\alpha)(\ell_{t-1}+b_{t-1}),\qquad b_t=\beta(\ell_t-\ell_{t-1})+(1-\beta)\,b_{t-1},\qquad \hat y_{t+h\mid t}=\ell_t+h\,b_t
$$

**Where:**
- $\ell_t$ is the level and $b_t$ the trend (how much the level rises per step).
- $\ell_{t-1}+b_{t-1}$ is where the level was expected to be this step.
- $\beta\in(0,1)$ is how fast the trend estimate reacts. $h$ is the horizon.
- **Matters:** the forecast is a straight line from the latest level with the latest slope. It extends the trend forever, so long forecasts can go too high or too low (the damped variant softens this).

**Example (tiny numbers):** $\alpha=\beta=0.5$, $\ell_0=10$, $b_0=1$, then $y_1=12$ and $y_2=14$.
- Step 1: $\ell_1=0.5\times12+0.5\times11=\mathbf{11.5}$ and $b_1=0.5\times1.5+0.5\times1=\mathbf{1.25}$.
- Step 2: $\ell_2=0.5\times14+0.5\times12.75=\mathbf{13.375}$ and $b_2=0.5\times1.875+0.5\times1.25=\mathbf{1.5625}$.
- Forecast: $h=1$ gives $13.375+1.5625=\mathbf{14.9375}$, and $h=3$ gives $13.375+3\times1.5625=\mathbf{18.0625}$.
- Meaning: the values rose by 2 per step, and the trend estimate is moving up toward that.

**Holt-Winters (level, trend and an additive season of length $m$).**

$$
\begin{aligned}
\ell_t&=\alpha\,(y_t-s_{t-m})+(1-\alpha)(\ell_{t-1}+b_{t-1})\\
b_t&=\beta(\ell_t-\ell_{t-1})+(1-\beta)\,b_{t-1}\\
s_t&=\gamma\,(y_t-\ell_{t-1}-b_{t-1})+(1-\gamma)\,s_{t-m}\\
\hat y_{t+h\mid t}&=\ell_t+h\,b_t+s_{t+h-m(k+1)},\qquad k=\lfloor (h-1)/m\rfloor
\end{aligned}
$$

**Where:**
- $s_t$ is the seasonal effect at time $t$, and $s_{t-m}$ is the effect for the same position in the cycle one season ago.
- $y_t-s_{t-m}$ is the newest value with its season taken out. $y_t-\ell_{t-1}-b_{t-1}$ is how far the value landed from the expected level, which updates the seasonal number.
- $\gamma\in(0,1)$ is how fast the seasonal numbers update. $m$ is the season length.
- **Matters:** the level update uses a de-seasonalised value, so a seasonal peak is not mistaken for a rise in level. Without this, a busy month would push the level up and spoil every later forecast. The multiplicative version divides and multiplies by $s$ instead of subtracting and adding.

**Example (tiny numbers):** $\alpha=\beta=\gamma=0.5$, season length $m=2$. Before the new value: $\ell_{t-1}=10$, $b_{t-1}=1$, $s_{t-2}=2$ (the same position in the cycle), and the other position's seasonal number is $s_{t-1}=-2$. The new value is $y_t=14$.
- $\ell_t=0.5\times(14-2)+0.5\times(10+1)=\mathbf{11.5}$.
- $b_t=0.5\times(11.5-10)+0.5\times1=\mathbf{1.25}$.
- $s_t=0.5\times(14-10-1)+0.5\times2=\mathbf{2.5}$.
- Forecast for $h=1$: $11.5+1.25+s_{t-1}=11.5+1.25-2=\mathbf{10.75}$.
- Meaning: the level and trend moved, the season number for this position rose from 2 to 2.5, and the next step (the other position) is forecast lower because of its own seasonal dip.

**Further notes:**
- **Fitting:** the fractions are usually chosen by minimising the squared one-step errors, or by maximum likelihood in the ETS framework ([[Maximum Likelihood]]).
- **Forecast errors grow with the horizon,** and the library's point forecasts do not tell you how much. Use the ETS version for intervals.

**What came before, and how this improved on it:**
- **Before:** the moving average (equal weights over a window) and straight-line trends.
- **Its limit:** a moving average forgets everything outside its window at once, must store the window, and reacts slowly to a turn; a fitted straight line cannot follow a level that wanders or a season.
- **What changed:** weights that fade smoothly, plus a trend term and a seasonal term, all updated by a simple recursion. Brown (1956), Holt (1957) and Winters (1960) are the names usually given (from memory, not checked).
- **How much better:** in the code above, on the last 24 months of the airline series, the seasonal naive forecast had an RMSE of 77.0 and Holt without a season 97.7, while Holt-Winters had **32.5**, about **2.4 times** smaller than seasonal naive (77.0 over 32.5, my division). One series, one split.

## Related
- [[Time Series]] · [[Trend and Seasonality]] · [[Stationarity]] · [[Autocorrelation]] · [[ARIMA]] · [[Forecast Metrics]] · [[Time Series Cross-Validation]] · [[Forecasting Shoot-out]] · [[Maximum Likelihood]] · [[Learning Rate]] · [[Gradient Boosting]] · [[Recurrent Neural Network]] · [[Probabilistic Forecasting]]

## Sources
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, chapters on exponential smoothing (free online; from memory)
- Brown (1956), Holt (1957) and Winters (1960): the original exponential smoothing papers (named from memory, not checked)
- Image: see the caption in section 2.
- Numbers in this note come from the code above (my arithmetic and my run on this machine).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/time-series Exponential smoothing:::A forecast built from a running level (and optionally trend and season), updated each step by moving part of the way toward the newest value, so old values fade exponentially.

#flashcards/ml/concept #flashcards/topic/time-series What do $\alpha$, $\beta$ and $\gamma$ control in Holt-Winters?
?
$\alpha$ controls how fast the level reacts, $\beta$ how fast the trend estimate reacts and $\gamma$ how fast the seasonal numbers update. A large value reacts quickly, a small value smooths more. They are chosen by minimising past forecast errors.

#flashcards/ml/formula #flashcards/topic/time-series Write simple exponential smoothing and explain each symbol.
?
$$\ell_t=\alpha\,y_t+(1-\alpha)\,\ell_{t-1},\qquad \hat y_{t+h\mid t}=\ell_t$$
- $y_t$: the newest value. $\ell_{t-1}$: the level before it. $\ell_t$: the updated level.
- $\alpha\in(0,1)$: the weight of the newest value. $\hat y_{t+h\mid t}$: the forecast $h$ steps ahead, the same for every $h$.
- $\alpha=1$ gives the naive forecast. A small $\alpha$ gives a very smooth level that reacts slowly.

#flashcards/ml/example #flashcards/topic/time-series With $\alpha=0.5$, $\ell_0=10$ and values 12 then 14, what is the level and the forecast?::l1 = 0.5 × 12 + 0.5 × 10 = 11. l2 = 0.5 × 14 + 0.5 × 11 = 12.5. The forecast is 12.5 for every future step.

#flashcards/ml/formula #flashcards/topic/time-series Write the weight simple exponential smoothing gives to old values and explain each symbol.
?
$$\ell_t=\alpha\sum_{i=0}^{\infty}(1-\alpha)^{\,i}\,y_{t-i}$$
- $i$: how many steps back ($i=0$ is the newest value). $\alpha(1-\alpha)^i$: the weight of the value $i$ steps ago. The weights add up to 1.
- The weights shrink by the factor $(1-\alpha)$ per step back, a geometric fade. A larger $\alpha$ fades faster.
- Unlike a moving average, nothing is dropped abruptly, and only the last level has to be stored. (A vanishing term from the starting level is left out.)

#flashcards/ml/example #flashcards/topic/time-series What are the first five weights for $\alpha=0.3$, and how much do the last 12 values carry?::0.3, 0.21, 0.147, 0.103, 0.072. The last 12 values carry 1 - 0.7^12 = 0.986 of the total weight.

#flashcards/ml/formula #flashcards/topic/time-series Write Holt's linear method and explain each symbol.
?
$$\ell_t=\alpha y_t+(1-\alpha)(\ell_{t-1}+b_{t-1}),\qquad b_t=\beta(\ell_t-\ell_{t-1})+(1-\beta)\,b_{t-1},\qquad \hat y_{t+h\mid t}=\ell_t+h\,b_t$$
- $\ell_t$: the level. $b_t$: the trend, the rise per step. $\ell_{t-1}+b_{t-1}$: where the level was expected to be.
- $\beta\in(0,1)$: how fast the trend estimate reacts. $h$: the horizon.
- The forecast is a straight line from the latest level with the latest slope. It extends the trend forever (a damped trend softens this).

#flashcards/ml/example #flashcards/topic/time-series Holt with $\alpha=\beta=0.5$, $\ell_0=10$, $b_0=1$ and values 12, 14: what are the level, trend and forecasts for $h=1$ and $h=3$?::Step 1: l = 11.5, b = 1.25. Step 2: l = 13.375, b = 1.5625. h = 1: 14.9375. h = 3: 13.375 + 3 × 1.5625 = 18.0625.

#flashcards/ml/formula #flashcards/topic/time-series Write the additive Holt-Winters update and forecast, and explain the symbols.
?
$$\begin{aligned}\ell_t&=\alpha\,(y_t-s_{t-m})+(1-\alpha)(\ell_{t-1}+b_{t-1})\\ b_t&=\beta(\ell_t-\ell_{t-1})+(1-\beta)\,b_{t-1}\\ s_t&=\gamma\,(y_t-\ell_{t-1}-b_{t-1})+(1-\gamma)\,s_{t-m}\\ \hat y_{t+h\mid t}&=\ell_t+h\,b_t+s_{t+h-m(k+1)},\quad k=\lfloor (h-1)/m\rfloor\end{aligned}$$
- $s_t$: the seasonal effect at time $t$. $s_{t-m}$: the effect for the same position one season ago. $m$: the season length.
- $y_t-s_{t-m}$: the newest value with its season removed. $y_t-\ell_{t-1}-b_{t-1}$: how far the value landed from the expected level.
- $\gamma\in(0,1)$: how fast the seasonal numbers update.
- Deseasonalising keeps a seasonal peak from being mistaken for a rise in level. The multiplicative version divides and multiplies by $s$.

#flashcards/ml/example #flashcards/topic/time-series Holt-Winters with $\alpha=\beta=\gamma=0.5$, $m=2$, $\ell_{t-1}=10$, $b_{t-1}=1$, $s_{t-2}=2$, $s_{t-1}=-2$, $y_t=14$: what is the $h=1$ forecast?::l = 0.5(14-2) + 0.5(10+1) = 11.5. b = 0.5(1.5) + 0.5(1) = 1.25. s = 0.5(14-10-1) + 0.5(2) = 2.5. Forecast = 11.5 + 1.25 + s(t-1) = 11.5 + 1.25 - 2 = 10.75.

#flashcards/ml/image #flashcards/topic/time-series ![[exponential-moving-average-weights.png]]
What does this picture show?
?
The weights exponential smoothing gives to past values (span of 15 steps). The newest value has a weight of about 0.125, and each older step has 0.875 times the weight of the one after it, so the bars shrink geometrically and never reach zero.

#flashcards/ml/example #flashcards/topic/time-series On the last 24 months of the airline series, what were the RMSEs of naive, seasonal naive, Holt and Holt-Winters?::137.3, 77.0, 97.7 and 32.5. Holt-Winters was about 2.4 times better than seasonal naive. (Simple smoothing equalled naive at 137.3 because the fit chose alpha = 1.) One series, one split. My run on this machine.

#flashcards/ml/cloze #flashcards/topic/time-series In exponential smoothing the weights on older values ==fade geometrically==, and a larger smoothing fraction ==forgets faster==.
