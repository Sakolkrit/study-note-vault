---
tags: [time-series, forecasting, decomposition, statistics, machine-learning, ML]
status: growing
level: 4
tier: 1
---

# Trend and Seasonality

Most real [[Time Series]] are a **stack of three simple things**: a slow drift (the **trend**), a pattern that repeats on a calendar (the **seasonality**), and leftover random wiggles (the **noise**). Pulling them apart is called **decomposition**.

## 🟢 1. Simple (age 12)
Think of a shop that sells ice cream. Every year it sells a bit more than the year before (that is the **trend**). Every summer it sells much more than in winter, year after year (that is the **season**). And on any one day, the weather makes sales jump up or down a little at random (that is the **noise**). If you want to understand the shop, it helps to look at the three parts separately.

**Example:** The shop sold 100 tubs in an average month this year. July is always 5 tubs above average. This July it sold 104, so the leftover random part was $-1$: $100+5-1=104$.

## 🔵 2. How it works
1. **Plot the series** and look at it. You can often see all three parts by eye.
2. **Find the trend** by smoothing. Replace each point by the average of the points around it (a **moving average**) so the small wiggles and the yearly cycle average out. The smooth curve left over is the trend.
3. **Remove the trend.** Subtract it (or divide by it) from the data. What remains is the season plus the noise.
4. **Find the season.** For each month, average the leftovers across all the years. January's average leftover is the January effect, and so on. This gives one pattern that repeats every year.
5. **What is left is the noise** (the **residual**). If the split worked, the residual looks like random wiggles with no pattern.
6. **Add or multiply.** If the seasonal swing is about the same size every year, the parts are **added**. If the swing grows when the level grows (a bigger shop has bigger summer jumps), the parts are **multiplied**. Taking logarithms turns multiplying into adding, which is why people often take logs first.
7. **Why bother:** you can forecast each part on its own (extend the trend, repeat the season), you can "seasonally adjust" a series to see whether things really improved, and the residual tells you if a pattern is still hiding.

![[co2-trend-seasonal-decomposition.png]]
*Monthly atmospheric CO2 (NOAA, 1959 to 2024) split into two parts. The main line is the **trend**, with the seasonal part removed: a steady, slightly curving rise. The small panel shows the **seasonal part** over two years: a regular yearly wave of roughly plus and minus 3 ppm, peaking in spring and lowest in autumn. Image: Ldecola, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Noaa_co2_seasonal_decompose.png), CC0 (a 960-pixel-wide version of the file). The data comes from NOAA, as credited on the image.*

## 🟠 3. Technical detail
- **Season length** $m$: the number of steps in one cycle (12 for monthly data with a yearly cycle, 7 for daily data with a weekly cycle). You must supply it.
- **Classical decomposition:** a centred moving average gives the trend (for an even $m$ such as 12, the average uses a half weight on both ends so it stays centred). The detrended values are averaged per season to get the seasonal pattern, then centred so the seasonal parts add to zero (additive) or average to one (multiplicative). The remainder is the residual.
- **Side effect:** a centred moving average of width $m$ cannot be computed for the first and last $m/2$ points, so the trend and residual have gaps at both ends. That is exactly where a forecaster needs them most.
- **STL** (a more modern method, "Seasonal and Trend decomposition using Loess"; from memory) lets the seasonal pattern change slowly over time and is less sensitive to outliers.
- **More than one cycle:** hourly data can have a daily and a weekly cycle at the same time. Methods that handle several seasons exist (not covered here).
- **Trend versus cycle:** a "cycle" that does not repeat on a fixed calendar (like business cycles) is not seasonality. The classical method lumps it into the trend or the residual.
- **Check the residual:** it should show no pattern and little [[Autocorrelation]]. A visible pattern means something was left in.
- **Use in forecasting:** [[Exponential Smoothing]] (Holt-Winters) models the level, trend and season directly and updates them over time. [[ARIMA]] removes them by differencing ([[Stationarity]]). Machine-learning models such as [[Gradient Boosting]] need the trend removed or modelled (a tree cannot extend a trend it has never seen).

```python
import numpy as np, pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

y = pd.Series(pd.read_csv("AirPassengers.csv")["value"].to_numpy(float), index=pd.date_range("1949-01", periods=144, freq="MS"))
swing = lambda yr: (round(float(y[str(yr)].max() - y[str(yr)].min()), 1), round(float((y[str(yr)].max() - y[str(yr)].min()) / y[str(yr)].mean()), 2))
print("seasonal swing (max - min, and as a share of that year's average): 1949", swing(1949), "| 1960", swing(1960))

add = seasonal_decompose(y, model="additive", period=12)
mul = seasonal_decompose(y, model="multiplicative", period=12)
half = lambda r: (round(float(r.dropna().iloc[:60].std()), 3), round(float(r.dropna().iloc[-60:].std()), 3))
level = lambda yr: round(float(y[str(yr)].mean()), 1)
print("average level 1949, 1960:", level(1949), level(1960))
print("additive: residual spread early vs late:", half(add.resid), "| multiplicative:", half(mul.resid))
print("seasonal part, July and November: additive", round(float(add.seasonal.iloc[6]), 1), round(float(add.seasonal.iloc[10]), 1), "| multiplicative", round(float(mul.seasonal.iloc[6]), 3), round(float(mul.seasonal.iloc[10]), 3))
print("raw spread", round(float(y.std()), 1), "| additive residual spread", round(float(add.resid.std()), 1), "| share of variance left", round(float(add.resid.var() / y.var()), 4))
line = np.polyval(np.polyfit(np.arange(144), y.to_numpy(), 1), np.arange(144))      # the old way: only a straight-line trend
print("variance left after a straight line:", round(float(((y.to_numpy() - line) ** 2).mean() / y.var(ddof=0)), 4), "| after trend + season:", round(float(add.resid.dropna().var(ddof=0) / y.var(ddof=0)), 4))

t = np.array([10, 12, 14, 13, 15.])                                  # tiny example: centred 3-point moving average
print(np.convolve(t, np.ones(3) / 3, mode="valid").round(3))
```

Output (first the seasonal swing, then the two decompositions, then the tiny example):
- `seasonal swing ... 1949 (44.0, 0.35) | 1960 (232.0, 0.49)` and `average level 1949, 1960: 126.7 476.2`: the airline series' yearly swing between its highest and lowest month grew from 44 to 232 thousand passengers while the average level grew from 126.7 to 476.2. The swing grows with the level, which points to the multiplicative form.
- `additive: residual spread early vs late: (18.788, 21.748) | multiplicative: (0.039, 0.03)`: the multiplicative residuals are small, steady ratios (about 4% and 3%). The additive residuals are in passengers, and the additive seasonal part is fixed (July $+63.8$, November $-53.6$) although the real swing grew.
- `seasonal part ... multiplicative 1.227 0.801`: July runs about 22.7% above the trend and November about 19.9% below it.
- `variance left after a straight line: 0.1464 | after trend + season: 0.026`: a straight line leaves 14.6% of the variance unexplained, trend plus season leaves 2.6% (the decomposition has no values for the first and last 6 months, so the two are measured on slightly different points).
- `[12. 13. 14.]`: the 3-point moving averages of `10, 12, 14, 13, 15`.

## 🔴 4. Deep / formal
**The two models.** With trend $T_t$, seasonal part $S_t$ and residual $R_t$ at time $t$:

$$
\text{additive: }\ y_t=T_t+S_t+R_t,\qquad\qquad \text{multiplicative: }\ y_t=T_t\times S_t\times R_t
$$

**Where:**
- $y_t$ is the observed value at time $t$.
- $T_t$ is the trend: the slow-moving level. $S_t$ is the seasonal effect: it repeats every $m$ steps, so $S_t=S_{t+m}$.
- $R_t$ is the residual (noise).
- In the additive form the parts are in the same units as $y$, and $S_t$ averages to 0. In the multiplicative form $S_t$ and $R_t$ are ratios near 1, and $S_t$ averages to 1.
- **Matters:** choosing the wrong form leaves a pattern in the residual. If the swings grow with the level and you use the additive form, the early years are over-fitted and the late years under-fitted. Taking logs makes the multiplicative form additive: $\ln y_t=\ln T_t+\ln S_t+\ln R_t$.

**Example (tiny numbers):** $T_t=100$, a seasonal effect and a residual.
- Additive: $S_t=+5$ and $R_t=-1$, so $y_t=100+5-1=\mathbf{104}$.
- Multiplicative: $S_t=1.05$ and $R_t=0.99$, so $y_t=100\times1.05\times0.99=\mathbf{103.95}$.
- Meaning: the same shop is described either as "5 above average, minus 1" or as "5% above average, times 0.99". The two forms agree when the level is 100, but at a level of 200 the multiplicative form gives a swing of 10 while the additive form still gives 5.

**Estimating the trend.** The classical method uses a centred moving average of odd width $m=2k+1$:

$$
\hat T_t=\frac1m\sum_{j=-k}^{k}y_{t+j}
$$

**Where:**
- $y_{t+j}$ are the values around time $t$: $k$ before, $k$ after and the value at $t$ itself. $m=2k+1$ is the width of the window.
- $\hat T_t$ is the estimated trend at time $t$.
- **Matters:** averaging over a full season cancels the seasonal part (it adds to zero over one cycle), leaving the trend. If the window is not as wide as the season, some of the season leaks into the trend. For even $m$ the end points get half weight.

**Example (tiny numbers):** $y=10,12,14,13,15$ and $m=3$ ($k=1$).
- At $t=2$: $(10+12+14)/3=\mathbf{12}$.
- At $t=3$: $(12+14+13)/3=\mathbf{13}$.
- At $t=4$: $(14+13+15)/3=\mathbf{14}$.
- Meaning: the smooth curve is $12,13,14$ and is undefined at the two ends (there are not enough neighbours). It rises steadily, ignoring the dip at $t=4$.

**Estimating the season.** Subtract the trend, then average the detrended values for each position $i$ in the cycle (all Januaries, all Februaries, and so on):

$$
\hat S_i=\frac1{n_i}\sum_{t\in\text{position }i}\big(y_t-\hat T_t\big)
$$

**Where:**
- $i$ is the position in the cycle (1 to $m$). $n_i$ is how many values fall at that position.
- $y_t-\hat T_t$ is the detrended value (for the multiplicative form, $y_t/\hat T_t$).
- $\hat S_i$ is the estimated seasonal effect for position $i$. It is then adjusted so the $m$ effects add to 0 (or average to 1).
- **Matters:** the average over several years cancels the noise, so the seasonal estimate becomes more reliable the more cycles you have. With only one or two cycles it is mostly noise.

**Example (tiny numbers):** the detrended values for position "July" in three years are $+6,\,+8,\,+7$.
- Sum: $6+8+7=21$, and $n_i=3$.
- $\hat S_{\text{July}}=21/3=\mathbf{7}$.
- Meaning: July is, on average, 7 above the trend. (For the airline series the code found $+63.8$ for July in the additive form.)

**Further notes:**
- **Seasonal adjustment** means removing $\hat S_t$ from $y_t$. Government statistics are usually quoted this way (from memory).
- **Seasonality is not the same as autocorrelation at lag $m$**, but a strong season shows up as a strong autocorrelation at lags $m,2m,\dots$ ([[Autocorrelation]]).
- **Limits:** the classical method assumes the seasonal pattern is exactly the same every cycle, and an outlier distorts the moving average near it.

**What came before, and how this improved on it:**
- **Before:** fitting a single straight line or polynomial trend to the whole series, and judging seasonality by eye.
- **Its limit:** a straight line leaves the whole yearly cycle in the leftovers, so the leftovers are large and patterned.
- **What changed:** separate the cycle from the trend with moving averages and per-season averages. Ratio-to-moving-average methods go back to the early 20th century, the Census Bureau's X-11 family to the 1960s, and STL to 1990 (all from memory, not checked).
- **How much better:** in the code above, a straight line leaves **14.6%** of the airline series' variance unexplained, while trend plus season leaves **2.6%**, about 5.6 times less (my division). This is measured on the data it was fitted to, not on a forecast.

## Related
- [[Time Series]] · [[Stationarity]] · [[Autocorrelation]] · [[Exponential Smoothing]] · [[ARIMA]] · [[Forecast Metrics]] · [[Forecasting Shoot-out]] · [[Linear Regression]] · [[Variance]] · [[Gradient Boosting]] · [[Random Forest]]

## Sources
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, chapter on time series decomposition (free online; from memory)
- Cleveland et al., "STL: A Seasonal-Trend Decomposition Procedure Based on Loess", 1990 (from memory, not checked)
- Image: see the caption in section 2.
- Numbers in this note come from the code above (my arithmetic and my run on this machine).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/time-series Decomposition (of a time series):::Splitting a series into a slow trend, a repeating seasonal pattern and leftover noise, either added or multiplied together.

#flashcards/ml/concept #flashcards/topic/time-series When should you use the additive form and when the multiplicative form?
?
Additive when the seasonal swing stays about the same size. Multiplicative when the swing grows with the level. The airline series' yearly swing grew from 44 to 232 while the level grew from 126.7 to 476.2, so it is multiplicative. Taking logs turns multiplicative into additive.

#flashcards/ml/formula #flashcards/topic/time-series Write the additive and multiplicative decomposition and explain each symbol.
?
$$\text{additive: }\ y_t=T_t+S_t+R_t,\qquad\qquad \text{multiplicative: }\ y_t=T_t\times S_t\times R_t$$
- $y_t$: the observed value. $T_t$: the trend (slow-moving level).
- $S_t$: the seasonal effect, repeating every $m$ steps. $R_t$: the residual (noise).
- Additive: parts in the same units, $S_t$ averages to 0. Multiplicative: $S_t$ and $R_t$ are ratios near 1, $S_t$ averages to 1.
- The wrong form leaves a pattern in the residual. Taking logs makes the multiplicative form additive.

#flashcards/ml/example #flashcards/topic/time-series With $T_t=100$: what is $y_t$ for additive ($S=+5$, $R=-1$) and multiplicative ($S=1.05$, $R=0.99$)?::Additive: 100 + 5 - 1 = 104. Multiplicative: 100 × 1.05 × 0.99 = 103.95. At level 200 the multiplicative form would give a swing of 10 but the additive form still 5.

#flashcards/ml/formula #flashcards/topic/time-series Write the centred moving-average trend estimate and explain each symbol.
?
$$\hat T_t=\frac1m\sum_{j=-k}^{k}y_{t+j}$$
- $y_{t+j}$: the values around time $t$ ($k$ before, $k$ after, and $t$ itself). $m=2k+1$: the window width.
- $\hat T_t$: the estimated trend at time $t$.
- Averaging over a full season cancels the seasonal part, leaving the trend. A window narrower than the season lets some season leak into the trend. For even $m$ the end points get half weight.

#flashcards/ml/example #flashcards/topic/time-series What are the 3-point moving averages of 10, 12, 14, 13, 15?::12, 13, 14 (at t = 2, 3, 4). The two end points are undefined because they lack neighbours.

#flashcards/ml/formula #flashcards/topic/time-series Write the seasonal effect estimate for position $i$ in the cycle and explain each symbol.
?
$$\hat S_i=\frac1{n_i}\sum_{t\in\text{position }i}\big(y_t-\hat T_t\big)$$
- $i$: the position in the cycle (1 to $m$). $n_i$: how many values fall at that position.
- $y_t-\hat T_t$: the detrended value (for multiplicative, $y_t/\hat T_t$).
- $\hat S_i$: the seasonal effect for position $i$, afterwards adjusted so the $m$ effects add to 0 (or average to 1).
- Averaging over cycles cancels noise, so it needs several cycles to be reliable.

#flashcards/ml/example #flashcards/topic/time-series Detrended values for July in three years are +6, +8 and +7: what is the July effect?::(6 + 8 + 7) / 3 = 7. July is on average 7 above the trend. For the airline series the additive July effect was +63.8.

#flashcards/ml/image #flashcards/topic/time-series ![[co2-trend-seasonal-decomposition.png]]
What does this picture show?
?
Monthly atmospheric CO2 (NOAA, 1959 to 2024) split into a trend (the main line, seasonal part removed: a steady, slightly curving rise) and a seasonal part (the small panel, a regular yearly wave of roughly plus and minus 3 ppm, shown over two years).

#flashcards/ml/concept #flashcards/topic/time-series How much of the airline series' variance did a straight line leave unexplained, compared with trend plus season?::14.6% after a straight line, against 2.6% after additive trend plus season (about 5.6 times less). Measured on the fitted data, not on a forecast. My run on this machine.

#flashcards/ml/cloze #flashcards/topic/time-series A centred moving average of width equal to the season length ==cancels the seasonal part==, leaving the ==trend==.
