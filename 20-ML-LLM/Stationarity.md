---
tags: [time-series, forecasting, statistics, stationarity, differencing, ML]
status: growing
level: 4
tier: 2
---

# Stationarity

A [[Time Series]] is **stationary** when its behaviour does not depend on *when* you look: the same average level, the same size of ups and downs, and the same pattern of wiggles at any time. Most forecasting methods are built for stationary series, so non-stationary ones are first turned into stationary ones, usually by **differencing**.

## 🟢 1. Simple (age 12)
Imagine two graphs. One is the heart rate of a person sitting quietly. It goes up and down, but around the same level, and a piece from any minute looks like a piece from any other minute. The other is the height of a growing plant: it keeps rising, so a piece from last week looks different from a piece from this week. The first graph is **stationary**. The second is not.

**Example:** A drunk walking home takes a random step left or right each second. Where they are now depends on all the steps before, and they can wander farther and farther from the start. That is not stationary. But their *steps* (how far they moved each second) are just random and look the same all the time. Looking at the steps instead of the position turns the problem into a stationary one.

## 🔵 2. How it works
1. **Look at the plot.** If the line drifts up or down, or the size of the wiggles grows, or there is a clear yearly cycle, the series is probably not stationary.
2. **Compare pieces.** Split the series in two. If the first half has a different average level (or a different spread) from the second half, it is not stationary.
3. **Remove a drift by differencing.** Replace each value by *this value minus the previous value*. If the series was rising steadily, the differences are steady, with no drift.
4. **Remove a yearly cycle by seasonal differencing.** Replace each value by *this value minus the value one season ago* (this July minus last July).
5. **If the swings grow with the level, take the logarithm first.** Then difference. (See [[Trend and Seasonality]].)
6. **Check again** with a plot, and with a statistical test (the "ADF test", below).
7. **Why bother?** Averages and [[Autocorrelation]] only mean something if they stay the same over time. If the average keeps moving, "the typical value" has no meaning, and a model fitted to the past does not describe the future.
8. **Warning:** two unrelated series that both drift can look strongly related, only because both are drifting (see the code).

## 🟠 3. Technical detail
- **Weak (covariance) stationarity:** the mean is constant, the variance is constant, and the covariance between two values depends only on the gap between them, not on the time ([[Covariance]]). Strict stationarity (the whole distribution is unchanged by a shift) is stronger. In practice "stationary" means the weak kind.
- **Two kinds of non-stationary:** *trend-stationary* (a deterministic trend plus stationary wiggles: subtract the trend) and *difference-stationary* (a **unit root**, like a random walk: difference it). Subtracting a trend from a random walk does not make it stationary.
- **Random walk:** $y_t=y_{t-1}+\varepsilon_t$. Its variance grows with time, its past shocks never fade, and its best forecast is the last value.
- **ADF test** (augmented Dickey-Fuller): its starting claim ("null hypothesis") is that the series has a unit root, so it is non-stationary. A **small p-value** (say below 0.05) is evidence against that claim, which supports stationarity. A large p-value means "cannot rule out non-stationary". (The KPSS test uses the opposite claim; from memory.)
- **Over-differencing:** differencing a series that is already stationary adds noise and creates a strong negative autocorrelation at lag 1. Difference only as much as needed (usually once, rarely twice).
- **Variance:** if the spread grows with the level, a logarithm or a Box-Cox transform (from memory) stabilises it before differencing.
- **Where it is needed:** [[ARIMA]] is built on it (the "I" in ARIMA stands for "integrated", meaning differencing was used). Neural and tree models are less strict, but they still struggle with trends they have not seen (see [[Gradient Boosting]] and [[Random Forest]]).

```python
import numpy as np, pandas as pd
from statsmodels.tsa.stattools import adfuller

y = pd.read_csv("AirPassengers.csv")["value"].to_numpy(float)
p = lambda v: round(float(adfuller(v, autolag="AIC")[1]), 4)
print("airline passengers: raw", p(y), "| first difference", p(np.diff(y)), "| log then first difference", p(np.diff(np.log(y))), "| log, seasonal difference then first", p(np.diff(np.log(y)[12:] - np.log(y)[:-12])))
rng = np.random.default_rng(0); e = rng.normal(size=500)
rw = np.cumsum(e)                                                   # random walk: phi = 1
ar = np.zeros(500)
for t in range(1, 500): ar[t] = 0.5 * ar[t - 1] + e[t]              # stationary AR(1): phi = 0.5
print("random walk p", p(rw), "| AR(1) phi 0.5 p", p(ar), "| diff of random walk p", p(np.diff(rw)))
w1, w2 = rw[:250], rw[250:]
print("random walk, mean first half vs second half:", round(w1.mean(), 1), round(w2.mean(), 1), "| AR(1):", round(ar[:250].mean(), 2), round(ar[250:].mean(), 2))
# variance of a random walk grows with time; AR(1) settles at sigma^2/(1-phi^2)
paths = np.cumsum(rng.normal(size=(2000, 101)), axis=1); print("random walk variance at t=10, 100:", round(paths[:, 10].var(), 1), round(paths[:, 100].var(), 1))
a = np.zeros((2000, 101))
for t in range(1, 101): a[:, t] = 0.5 * a[:, t - 1] + rng.normal(size=2000)
print("AR(1) phi 0.5 variance at t=10, 100:", round(a[:, 10].var(), 3), round(a[:, 100].var(), 3), " theory", round(1 / (1 - 0.25), 3))
x = np.array([10, 12, 15, 19.]); print(np.diff(x), np.diff(x, 2))

a, b = np.cumsum(rng.normal(size=(1000, 200)), axis=1), np.cumsum(rng.normal(size=(1000, 200)), axis=1)   # 1,000 pairs of unrelated random walks
corr = lambda u, v: np.array([np.corrcoef(p, q)[0, 1] for p, q in zip(u, v)])
big = lambda c: round(float((np.abs(c) > 0.5).mean()), 3)
print("unrelated random walks with |correlation| above 0.5:", big(corr(a, b)), "| after differencing:", big(corr(np.diff(a, axis=1), np.diff(b, axis=1))))
```

Output:
- `airline passengers: raw 0.9919 | first difference 0.0542 | log then first difference 0.0711 | log, seasonal difference then first 0.0002`: ADF p-values for the airline series. Raw is far from stationary (0.99). A first difference alone is borderline (0.054, 0.071 after a log), and the log, then seasonal difference, then first difference reaches 0.0002.
- `random walk p 0.7252 | AR(1) phi 0.5 p 0.0 | diff of random walk p 0.0`: the random walk is not rejected as non-stationary, the stationary AR(1) is, and so is the random walk's difference.
- `random walk, mean first half vs second half: 4.1 -10.4 | AR(1): -0.01 -0.1`: the random walk's level wanders, the AR(1)'s does not.
- `random walk variance at t=10, 100: 11.0 105.8` (grows with time) against `AR(1) phi 0.5 variance at t=10, 100: 1.412 1.27 theory 1.333` (settles).
- `[2. 3. 4.] [1. 1.]`: the tiny example (first and second differences of `10, 12, 15, 19`).
- `unrelated random walks with |correlation| above 0.5: 0.405 | after differencing: 0.0`: of 1,000 pairs of unrelated random walks, 40.5% looked strongly correlated; after differencing, none did.

## 🔴 4. Deep / formal
**Weak stationarity.** A series $y_t$ is weakly stationary if, for all times $t$ and gaps $k$:

$$
\mathbb E[y_t]=\mu,\qquad \operatorname{Var}(y_t)=\sigma^2,\qquad \operatorname{Cov}(y_t,\,y_{t+k})=\gamma_k
$$

**Where:**
- $\mathbb E[y_t]$ is the average value at time $t$ (the [[Expected Value]]). It must equal the same constant $\mu$ at every $t$.
- $\operatorname{Var}(y_t)$ is the spread at time $t$ ([[Variance]]). It must be the same constant $\sigma^2$ at every $t$.
- $\operatorname{Cov}(y_t,y_{t+k})$ is the covariance between two values $k$ steps apart ([[Covariance]]). It may depend on the gap $k$ (the number $\gamma_k$) but **not** on $t$.
- **Matters:** these are the conditions under which the autocorrelation at lag $k$ is a single well-defined number, which is what [[Autocorrelation]] and [[ARIMA]] rely on. If any one fails (a drift in $\mu$, growing $\sigma^2$, a changing pattern), there is no single "typical" behaviour to learn.

**Example (tiny numbers):** the series $1,2,3,4,5,6$ (a steady rise).
- Average of the first half ($1,2,3$): $\mathbf{2}$. Average of the second half ($4,5,6$): $\mathbf{5}$.
- The mean moved, so the first condition fails: not stationary.
- Its differences are $1,1,1,1,1$, with average $\mathbf{1}$ in both halves and spread 0: stationary.
- Meaning: the drift was in the level, not in the steps.

**Differencing.** The first difference, and the seasonal difference for season length $m$, are:

$$
\nabla y_t=y_t-y_{t-1},\qquad\qquad \nabla_m y_t=y_t-y_{t-m}
$$

**Where:**
- $y_t$ is the value at time $t$, $y_{t-1}$ the previous one, $y_{t-m}$ the value one season ago.
- $\nabla$ ("nabla") is the difference operator. Applying it twice gives the second difference, $\nabla^2y_t=\nabla y_t-\nabla y_{t-1}$.
- **Matters:** the first difference removes a steady drift (a straight-line trend becomes a constant) and the second removes a curved drift. The seasonal difference removes a pattern that repeats every $m$ steps. Doing more than you need adds noise.

**Example (tiny numbers):** $10,12,15,19$.
- First differences: $12-10,\ 15-12,\ 19-15=\mathbf{2,\,3,\,4}$.
- Second differences: $3-2,\ 4-3=\mathbf{1,\,1}$.
- Seasonal: the quarterly data $10,20,30,40,12,22,32$ with $m=4$ gives $12-10,\ 22-20,\ 32-30=\mathbf{2,\,2,\,2}$.
- Meaning: the curved rise needed two differences to become constant, and the repeating pattern disappeared after one seasonal difference, leaving a constant yearly rise of 2.

**Why a random walk is not stationary.** Take $y_t=\phi\,y_{t-1}+\varepsilon_t$ with noise of spread $\sigma$ and $y_0=0$. The variance at time $t$ is:

$$
\operatorname{Var}(y_t)=\sigma^2\,\frac{1-\phi^{2t}}{1-\phi^{2}},\qquad \lim_{t\to\infty}\operatorname{Var}(y_t)=\frac{\sigma^2}{1-\phi^2}\ \ (|\phi|<1),\qquad \operatorname{Var}(y_t)=t\,\sigma^2\ \ (\phi=1)
$$

**Where:**
- $\phi$ ("phi") is how much of the previous value is carried forward. $\varepsilon_t$ is a fresh random shock each step, with spread $\sigma$.
- $t$ is the number of steps since the start.
- **Matters:** if $|\phi|<1$, old shocks fade, the variance settles at a constant, and the series is stationary. If $\phi=1$ (a unit root), old shocks never fade, they pile up, and the variance grows with $t$ without limit. This is what the ADF test looks for.

**Example (tiny numbers):** $\sigma=1$.
- $\phi=0.5$: the limit is $1/(1-0.25)=\mathbf{1.333}$ (the simulation gave $1.27$ at $t=100$ and $1.41$ at $t=10$, noisy because it used 2,000 paths).
- $\phi=1$: the variance is $\mathbf{t}$, so $10$ at $t=10$ and $100$ at $t=100$ (the simulation gave $11.0$ and $105.8$).
- Meaning: the drunk walker's possible positions spread out without limit, while a series that pulls back toward its average stays in a fixed range.

**Further notes:**
- **Spurious regression:** regressing one random walk on another unrelated one often gives a big, "significant" relationship. The code found $|r|>0.5$ in 40.5% of pairs. The standard reference is Granger and Newbold, 1974 (from memory). Differencing, or specialised tools for related random walks (cointegration; from memory), avoids the trap.
- **The test depends on choices:** the ADF test needs the number of lagged differences and whether to include a constant or a trend. The code let the AIC pick the lags.
- **Cyclic behaviour** (like the sunspot cycle) can be stationary even though the series looks wavy, because the cycles have no fixed calendar.

**What came before, and how this improved on it:**
- **Before:** fit a trend line to the data and work with the leftovers, or regress one series on another as it came.
- **Its limit:** the leftovers of a random walk are still non-stationary, and relating two drifting series gives nonsense (spurious) results.
- **What changed:** difference the series until it is stationary (the Box-Jenkins approach, 1970; from memory) and check with a unit-root test (Dickey and Fuller, 1979; from memory, not checked).
- **How much better:** in the code above, the airline series' ADF p-value went from **0.9919** (raw) to **0.0002** (log, seasonal difference, then first difference), and the share of unrelated random-walk pairs that looked strongly correlated (above 0.5) fell from **40.5%** to **0%** after differencing.

## Related
- [[Time Series]] · [[Trend and Seasonality]] · [[Autocorrelation]] · [[ARIMA]] · [[Exponential Smoothing]] · [[Forecast Metrics]] · [[Time Series Cross-Validation]] · [[Covariance]] · [[Variance]] · [[Expected Value]] · [[Markov Chain]] · [[Gradient Boosting]] · [[Random Forest]]
- [[Data Drift]]
## Sources
- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, chapter on stationarity and differencing (free online; from memory)
- Dickey & Fuller, 1979, and Granger & Newbold, 1974 (named from memory; not checked)
- Numbers in this note come from the code above (my arithmetic and my run on this machine). No image: I found no suitable free image on Wikimedia Commons.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/time-series Stationary time series:::A series whose average level, spread and pattern of dependence do not change over time.

#flashcards/ml/concept #flashcards/topic/time-series How do you make a drifting series stationary?
?
Difference it: replace each value by this value minus the previous one (a seasonal difference subtracts the value one season ago). Take a logarithm first if the swings grow with the level. Do not difference more than needed, because it adds noise.

#flashcards/ml/formula #flashcards/topic/time-series Write the conditions for weak stationarity and explain each symbol.
?
$$\mathbb E[y_t]=\mu,\qquad \operatorname{Var}(y_t)=\sigma^2,\qquad \operatorname{Cov}(y_t,\,y_{t+k})=\gamma_k$$
- $\mathbb E[y_t]$: the average at time $t$. It must be the same constant $\mu$ at every $t$.
- $\operatorname{Var}(y_t)$: the spread at time $t$. It must be the same constant $\sigma^2$.
- $\operatorname{Cov}(y_t,y_{t+k})$: the covariance between values $k$ steps apart. It may depend on the gap $k$ but not on $t$.
- These conditions make the autocorrelation at each lag one well-defined number. If any fails, there is no single typical behaviour to learn.

#flashcards/ml/example #flashcards/topic/time-series Is the series 1, 2, 3, 4, 5, 6 stationary, and what about its differences?::The first half averages 2 and the second half 5, so the mean moved: not stationary. Its differences are 1, 1, 1, 1, 1 with average 1 in both halves and spread 0: stationary.

#flashcards/ml/formula #flashcards/topic/time-series Write the first difference and the seasonal difference and explain each symbol.
?
$$\nabla y_t=y_t-y_{t-1},\qquad\qquad \nabla_m y_t=y_t-y_{t-m}$$
- $y_t$: the value at time $t$. $y_{t-1}$: the previous value. $y_{t-m}$: the value one season ago.
- $\nabla$: the difference operator. Applied twice it gives the second difference.
- The first difference removes a steady drift, the second a curved drift, and the seasonal difference a repeating pattern. Too many differences add noise.

#flashcards/ml/example #flashcards/topic/time-series What are the first and second differences of 10, 12, 15, 19, and the seasonal difference of 10, 20, 30, 40, 12, 22, 32 with $m=4$?::First: 2, 3, 4. Second: 1, 1. Seasonal: 12-10, 22-20, 32-30 = 2, 2, 2. The curved rise needs two differences, and one seasonal difference leaves a constant yearly rise of 2.

#flashcards/ml/formula #flashcards/topic/time-series Write the variance of an AR(1) series and why a random walk is not stationary.
?
$$\operatorname{Var}(y_t)=\sigma^2\,\frac{1-\phi^{2t}}{1-\phi^{2}},\qquad \lim_{t\to\infty}\operatorname{Var}(y_t)=\frac{\sigma^2}{1-\phi^2}\ \ (|\phi|<1),\qquad \operatorname{Var}(y_t)=t\,\sigma^2\ \ (\phi=1)$$
- $\phi$: how much of the previous value is carried forward. $\varepsilon_t$ (the shock) has spread $\sigma$. $t$: steps since the start.
- If $|\phi|<1$, old shocks fade and the variance settles, so the series is stationary.
- If $\phi=1$ (a unit root), shocks never fade and the variance grows with $t$ without limit. The ADF test looks for this.

#flashcards/ml/example #flashcards/topic/time-series With $\sigma=1$, what is the AR(1) variance limit for $\phi=0.5$, and the random walk variance at $t=100$?::1 / (1 - 0.25) = 1.333 for phi = 0.5 (the simulation gave 1.27 at t = 100). For a random walk the variance is t, so 100 at t = 100 (the simulation gave 105.8).

#flashcards/ml/concept #flashcards/topic/time-series What does the ADF test claim and how do you read its p-value?
?
Its starting claim is that the series has a unit root, so it is non-stationary. A small p-value (say below 0.05) is evidence against that claim, which supports stationarity. A large p-value means "cannot rule out non-stationary". On the airline series: raw 0.9919, after log, seasonal and first difference 0.0002.

#flashcards/ml/concept #flashcards/topic/time-series How often did two unrelated random walks look strongly correlated, and what happened after differencing?::In 1,000 simulated pairs, 40.5% had absolute correlation above 0.5. After differencing, 0% did. This is spurious regression. My run on this machine.

#flashcards/ml/cloze #flashcards/topic/time-series A random walk has a ==unit root==, so its variance ==grows with time==.
