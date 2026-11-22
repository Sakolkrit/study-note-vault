---
tags: [time-series, forecasting, statistics, correlation, acf, ML]
status: growing
level: 4
tier: 2
---

# Autocorrelation

Autocorrelation measures **how much a series resembles a delayed copy of itself**. It is the main clue that tells you the past can help predict the future, and a plot of it (the **ACF**, or correlogram) is the standard first look at a [[Time Series]].

## 🟢 1. Simple (age 12)
Today's temperature is usually close to yesterday's, so the temperature is "similar to itself, one day ago". It is also similar to the temperature exactly one year ago, because the seasons repeat. But it is not very similar to the temperature half a year ago (winter against summer). Autocorrelation puts a number on "how similar is a list to itself when you slide it by some amount?"

**Example:** You and a friend both clap a rhythm. If your friend copies you two beats late, then your claps and their claps line up well when you slide your list by two beats. The "two beats" is the **lag**, and the good match at that lag is a high autocorrelation.

## 🔵 2. How it works
1. **Take the series** and make a copy of it.
2. **Slide the copy by one step** (a lag of 1) so that each value sits next to the value that came just before it.
3. **Measure how the pairs move together.** This is the ordinary [[Covariance]]-and-correlation idea, but between the series and its own past. The result, between $-1$ and $+1$, is the autocorrelation at lag 1.
4. **Repeat for lag 2, lag 3, and so on.** Plot each as a bar. The picture is the **ACF plot** or **correlogram**.
5. **Read the picture:**
   - Bars that start high and shrink slowly: the series drifts (it has a trend). See [[Stationarity]].
   - Bars that rise again every 12 steps (monthly data): a yearly season. See [[Trend and Seasonality]].
   - Bars that drop to nothing after a few steps: the series has only a short memory.
   - All bars near zero: the series is just noise, so there is nothing to forecast from the past.
6. **The band:** the plot draws a band around zero. Bars that stay inside it could be chance, so you ignore them.
7. **The partial version (PACF):** it asks "does lag 3 add anything *beyond* what lag 1 and lag 2 already told me?". It shows direct links without the chain of in-between links.
8. **What it is used for:** choosing the settings of an [[ARIMA]] model, checking that a forecast's mistakes look like pure noise, finding the length of a season, and choosing which past values to give to a machine-learning model.

![[autocorrelation-acf-sine.svg]]
*Top: 100 random numbers with a hidden sine wave that repeats every 10 steps. Bottom: the autocorrelation (ACF) plot. The bars rise again at lags 10, 20, 30 and so on, which reveals the hidden cycle that is hard to see by eye in the top picture. The horizontal lines mark the band inside which bars could be chance. Image: M. W. Toews, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Acf.svg), CC BY 4.0.*

## 🟠 3. Technical detail
- **Sample autocorrelation at lag $k$:** it uses the series' overall average and overall variance, so it is a single number per lag ([[Variance]], [[Covariance]]). Lag 0 is always 1.
- **The band** for "no autocorrelation" is about $\pm1.96/\sqrt n$ for $n$ observations (from the usual 95% rule; a plain approximation that is valid for white noise).
- **Multiple testing:** with 40 lags plotted, about 2 bars will leave a 95% band by chance alone, so a single bar just outside the band means little.
- **PACF** (partial autocorrelation) removes the effect of the shorter lags. For an autoregressive model of order $p$ (AR($p$)), the ACF fades gradually and the PACF stops after lag $p$. For a moving-average model of order $q$ (MA($q$)), the ACF stops after lag $q$ and the PACF fades (the classic Box-Jenkins rules for choosing orders; from memory). See [[ARIMA]].
- **Ljung-Box test** (from memory): a single test of whether a whole group of autocorrelations is zero, often used on a model's residuals.
- **Non-stationary series fool it:** a trend gives a slowly fading ACF with large values at many lags that mean "there is a trend", not "each lag matters". Difference first ([[Stationarity]]).
- **For machine learning:** the lags with high autocorrelation are good candidates for lag features. See [[Feature Engineering]], [[Gradient Boosting]] and [[Random Forest]].

```python
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import acf

def r(y, k):                                                         # sample autocorrelation at lag k, written out
    d = y - y.mean()
    return float((d[k:] * d[:-k]).sum() / (d ** 2).sum())

x = np.array([1, 2, 3, 4, 5.]); print(r(x, 1), r(x, 2))              # tiny example
ap = pd.read_csv("AirPassengers.csv")["value"].to_numpy(float)
print([round(r(ap, k), 3) for k in (1, 2, 12)], "| statsmodels:", acf(ap, nlags=12, fft=False)[[1, 2, 12]].round(3).tolist())
print("95% band for n = 144:", round(1.96 / np.sqrt(144), 3), "| n = 100:", round(1.96 / np.sqrt(100), 3))

rng = np.random.default_rng(0); e = rng.normal(size=5000); ar = np.zeros(5000)
for t in range(1, 5000): ar[t] = 0.8 * ar[t - 1] + e[t]              # AR(1) with phi = 0.8: ACF should be 0.8, 0.64, 0.512, ...
print([round(r(ar, k), 3) for k in (1, 2, 3)], [round(0.8 ** k, 3) for k in (1, 2, 3)])

ly = np.log(ap); d = ly[12:] - ly[:-12]; d = np.diff(d)              # log, seasonal difference, then first difference
print("after differencing: lag 1, lag 12:", round(r(d, 1), 3), round(r(d, 12), 3))
sh = rng.permutation(ap); print("shuffled airline: lag 1, lag 12:", round(r(sh, 1), 3), round(r(sh, 12), 3))
print("raw airline series: lag 12 =", round(r(ap, 12), 3), "| first differences only: lag 12 =", round(r(np.diff(ap), 12), 3))
```

Output:
- `0.4 -0.1`: the tiny example (lags 1 and 2 of `1, 2, 3, 4, 5`).
- `[0.948, 0.876, 0.76] | statsmodels: [0.948, 0.876, 0.76]`: the airline series at lags 1, 2 and 12, from my formula and from the library: identical.
- `95% band for n = 144: 0.163 | n = 100: 0.196`.
- `[0.798, 0.638, 0.507] [0.8, 0.64, 0.512]`: a simulated AR(1) with $\phi=0.8$ (5,000 points) against the theory $\phi^k$.
- `after differencing: lag 1, lag 12: -0.341 -0.387` and `shuffled airline: lag 1, lag 12: 0.015 0.069`: after the log, seasonal difference and first difference, the autocorrelations are modest and negative (the typical signature of the "airline model", see [[ARIMA]]; from memory). The shuffled series has none.
- `raw airline series: lag 12 = 0.76 | first differences only: lag 12 = 0.829`: the yearly season is clear at lag 12.

## 🔴 4. Deep / formal
**Sample autocorrelation.** For a series $y_1,\dots,y_n$ with average $\bar y$, the autocorrelation at lag $k$ is:

$$
r_k=\frac{\sum_{t=k+1}^{n}(y_t-\bar y)(y_{t-k}-\bar y)}{\sum_{t=1}^{n}(y_t-\bar y)^2}
$$

**Where:**
- $y_t$ is the value at time $t$ and $y_{t-k}$ is the value $k$ steps earlier. $\bar y$ is the average of the whole series.
- The top is the sum of products of deviations from the average, for all pairs $k$ steps apart. It is positive when pairs tend to be on the same side of the average.
- The bottom is the sum of squared deviations (the variance times $n$). Dividing by it makes $r_0=1$ and keeps $r_k$ between $-1$ and $1$.
- **Matters:** the same overall average and variance are used for both halves of every pair, which is what makes this different from the correlation of two separate columns. If the series has a trend, $\bar y$ is not a meaningful centre and the bars are inflated.

**Example (tiny numbers):** $y=1,2,3,4,5$, so $\bar y=3$ and the deviations are $-2,-1,0,1,2$.
- Bottom: $4+1+0+1+4=\mathbf{10}$.
- Lag 1 top: $(-1)(-2)+(0)(-1)+(1)(0)+(2)(1)=2+0+0+2=\mathbf{4}$. So $r_1=4/10=\mathbf{0.4}$.
- Lag 2 top: $(0)(-2)+(1)(-1)+(2)(0)=\mathbf{-1}$. So $r_2=-1/10=\mathbf{-0.1}$.
- Meaning: a steady rise gives a positive lag-1 autocorrelation (neighbours are on the same side of the average), shrinking as the lag grows.

**The "just noise" band.** If the series is white noise (no dependence), each $r_k$ is approximately normal with spread $1/\sqrt n$, so about 95% of bars fall inside:

$$
\pm\frac{1.96}{\sqrt n}
$$

**Where:**
- $n$ is the number of observations.
- $1.96$ is the number of standard deviations that leaves 95% of a [[Normal Distribution]] inside.
- **Matters:** the band narrows as $n$ grows, so with a long series even small autocorrelations become detectable. It applies to white noise only: for a series that is not noise, the true band is wider (this matters little for a first look).

**Example (tiny numbers):** $n=144$ (the airline series): $1.96/\sqrt{144}=1.96/12=\mathbf{0.163}$. $n=100$: $1.96/10=\mathbf{0.196}$.
- Meaning: for the airline series a bar must be bigger than 0.163 in size before it stands out from chance.

**The ACF of an AR(1) series.** If each value is a fixed fraction $\phi$ of the previous one plus noise, $y_t=\phi\,y_{t-1}+\varepsilon_t$ with $|\phi|<1$:

$$
\rho_k=\phi^{\,k}
$$

**Where:**
- $\rho_k$ ("rho") is the true autocorrelation at lag $k$ (the number the sample $r_k$ estimates).
- $\phi$ is the carry-over fraction: how much of the previous value is kept.
- $k$ is the lag.
- **Matters:** the memory fades geometrically: each extra step multiplies by $\phi$. A $\phi$ near 1 gives a very slow fade (like a series with a trend), a $\phi$ near 0 gives almost no memory, and a negative $\phi$ gives bars that alternate in sign.

**Example (tiny numbers):** $\phi=0.8$.
- $\rho_1=0.8$, $\rho_2=0.8^2=\mathbf{0.64}$, $\rho_3=0.8^3=\mathbf{0.512}$.
- The simulation gave $0.798,\ 0.638,\ 0.507$.
- Meaning: after 3 steps, about half of the original influence remains.

**Further notes:**
- **Autocovariance versus autocorrelation:** $\gamma_k=\operatorname{Cov}(y_t,y_{t+k})$ is in squared units; $\rho_k=\gamma_k/\gamma_0$ is unit-free ([[Stationarity]] guarantees these do not depend on $t$).
- **Bias:** dividing by $n$ and not $n-k$ makes the sample estimates slightly small at large lags. It is the standard convention (from memory), and it keeps the estimated autocorrelation matrix well behaved.

**What came before, and how this improved on it:**
- **Before:** the ordinary correlation between two different variables (Pearson, 1890s; from memory), and eyeballing a plot for cycles.
- **Its limit:** it has no notion of "the same series, shifted", and the eye misses weak or long-period patterns.
- **What changed:** correlate a series with its own lagged copy at every lag and plot the result. Yule's correlogram idea and his autoregressive model for the sunspot series date from the 1920s, and Box and Jenkins built their identification method on the ACF and PACF in 1970 (all from memory, not checked).
- **How much better:** in the code above, the airline series (first differences) has an autocorrelation of **0.829** at lag 12, against **0.069** for the shuffled series, so the yearly cycle jumps out of the ACF where the raw values show nothing so clear. The ACF written out by hand also matched the library to three decimals.

## Related
- [[Time Series]] · [[Trend and Seasonality]] · [[Stationarity]] · [[ARIMA]] · [[Exponential Smoothing]] · [[Forecast Metrics]] · [[Time Series Cross-Validation]] · [[Covariance]] · [[Variance]] · [[Normal Distribution]] · [[Feature Engineering]] · [[Gradient Boosting]] · [[Random Forest]] · [[Markov Chain]]

## Sources
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, chapter on autocorrelation (free online; from memory)
- Box, Jenkins & Reinsel, *Time Series Analysis: Forecasting and Control* (from memory)
- Image: see the caption in section 2.
- Numbers in this note come from the code above (my arithmetic and my run on this machine).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/time-series Autocorrelation:::How strongly a series correlates with a delayed copy of itself, at each lag.

#flashcards/ml/concept #flashcards/topic/time-series How do you read an ACF plot?
?
Slowly fading high bars mean a trend (not stationary). Bumps every $m$ lags mean a season of length $m$. Bars that drop to zero after a few lags mean short memory. All bars near zero mean noise. Bars inside the band could be chance.

#flashcards/ml/formula #flashcards/topic/time-series Write the sample autocorrelation at lag $k$ and explain each symbol.
?
$$r_k=\frac{\sum_{t=k+1}^{n}(y_t-\bar y)(y_{t-k}-\bar y)}{\sum_{t=1}^{n}(y_t-\bar y)^2}$$
- $y_t$: the value at time $t$. $y_{t-k}$: the value $k$ steps earlier. $\bar y$: the average of the whole series.
- The top: the sum of products of deviations for all pairs $k$ steps apart. It is positive when pairs lie on the same side of the average.
- The bottom: the sum of squared deviations. It makes $r_0=1$ and keeps $r_k$ between $-1$ and $1$.
- A trend makes $\bar y$ a poor centre and inflates the bars.

#flashcards/ml/example #flashcards/topic/time-series What are the lag-1 and lag-2 autocorrelations of 1, 2, 3, 4, 5?::Mean 3, deviations -2, -1, 0, 1, 2, bottom 10. Lag 1 top: 2 + 0 + 0 + 2 = 4, so 0.4. Lag 2 top: 0 - 1 + 0 = -1, so -0.1.

#flashcards/ml/formula #flashcards/topic/time-series Write the "just noise" band for the ACF and explain each symbol.
?
$$\pm\frac{1.96}{\sqrt n}$$
- $n$: the number of observations.
- $1.96$: the number of standard deviations that leaves 95% of a normal distribution inside.
- About 95% of bars fall inside if the series is white noise. The band narrows as $n$ grows. With many lags, a few bars will leave it by chance alone.

#flashcards/ml/example #flashcards/topic/time-series What is the ACF band for $n=144$ and $n=100$?::1.96 / 12 = 0.163 for n = 144, and 1.96 / 10 = 0.196 for n = 100. For the airline series, a bar must exceed 0.163 to stand out from chance.

#flashcards/ml/formula #flashcards/topic/time-series Write the ACF of an AR(1) series and explain each symbol.
?
$$\rho_k=\phi^{\,k}$$
- $\rho_k$: the true autocorrelation at lag $k$ (what the sample $r_k$ estimates).
- $\phi$: the fraction of the previous value that is kept. $k$: the lag.
- Memory fades geometrically. A $\phi$ near 1 fades very slowly, near 0 gives almost no memory, and a negative $\phi$ alternates in sign.

#flashcards/ml/example #flashcards/topic/time-series For an AR(1) with $\phi=0.8$, what are the autocorrelations at lags 1, 2, 3, and what did the simulation give?::0.8, 0.64 and 0.512. The simulation of 5,000 points gave 0.798, 0.638 and 0.507.

#flashcards/ml/image #flashcards/topic/time-series ![[autocorrelation-acf-sine.svg]]
What does this picture show?
?
Top: random numbers with a hidden sine wave repeating every 10 steps. Bottom: the ACF plot, whose bars rise again at lags 10, 20, 30 and so on, revealing the cycle. The horizontal lines mark the band inside which bars could be chance.

#flashcards/ml/concept #flashcards/topic/time-series What do the ACF and PACF look like for AR and MA models? (from memory, standard rules)
?
For an AR(p) model the ACF fades gradually and the PACF stops after lag p. For an MA(q) model the ACF stops after lag q and the PACF fades gradually. The PACF shows the effect of lag k beyond what the shorter lags already explain.

#flashcards/ml/example #flashcards/topic/time-series What was the lag-12 autocorrelation of the airline series' first differences, against the shuffled series?::0.829 against 0.069. The yearly cycle shows in the ACF. My run on this machine.
