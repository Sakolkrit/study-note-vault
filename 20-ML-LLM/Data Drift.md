---
tags: [data-drift, concept-drift, covariate-shift, monitoring, PSI, Kolmogorov-Smirnov, MLOps, ML]
status: growing
level: 4
tier: 3
---

# Data Drift

Data drift means **the data a model sees in use slowly stops looking like the data it was trained on**. The model has not changed, but the world has, so its answers get worse, usually without any error message.

## 🟢 1. Simple (age 12)
You learn to guess the price of a house by looking at houses in your small town. Then you move to a big city. The houses are larger, the streets are different, the prices follow other rules. Your old guessing habits still run in your head, but they are now often wrong, and nobody tells you. **Data drift is "the new houses no longer look like the ones I learned from."** Drift detection is noticing that the houses changed before you give a lot of wrong prices.

**Example:** a shop trained a model to predict sales on data from winter. In summer the customers, the products and the weather are all different. The model still outputs numbers confidently. A drift check that compares this week's data with the training data would shout "these look different!" long before anyone checks whether the sales numbers were right.

## 🔵 2. How it works
1. **Keep a reference:** a copy of the data the model was trained on (or a summary of it, such as the spread of each column).
2. **Collect a batch** of recent production data (for example one week).
3. **Compare each column** of the batch with the same column in the reference. Two ways, both used in my lab:
   - **PSI** (population stability index): cut the column's range into bins, and measure how much the share of data in each bin has moved.
   - **A Kolmogorov-Smirnov test:** a statistical test of "could these two samples come from the same spread?" It gives a p-value.
4. **Decide with a rule:** for example "if PSI of two or more columns is above 0.2, alarm", or "if the test's p-value is tiny for any column, alarm".
5. **Look into it.** Is it a real change in the world, or a bug (a sensor broke, a column is now in other units, a form field is now empty)?
6. **Act:** fix the bug, retrain on fresh data, or switch back to a safer version ([[MLOps]]).
7. **Three kinds of change** to tell apart:
   - **Covariate shift:** the *inputs* change (more big houses); the link between inputs and answer may still hold.
   - **Label (prior) shift:** the share of each answer changes (more fraud than before).
   - **Concept drift:** the *link itself* changes (the same house now sells for a different price).
   You can see the first two without answers. The third usually shows only when the true answers arrive.
8. A warning that the inputs changed does **not** prove the model is worse. A real error check needs the true answers, which often come late.

## 🟠 3. Technical detail
- **What I measured** ([[Model Monitoring Lab]]): a random forest trained on northern California houses (latitude 37 or more; 6 features, no location), then shown 8 weekly batches of 600 rows where the share of **southern** rows grew from 0% to 100%. The monitor saw only the 6 inputs. Output of the lab's `monitor.py` (A = "2 or more features with PSI above 0.2"; B = "any feature with a KS p-value below 0.05/6"; MAE = the real error, available because I had the true prices):

```text
batch size 600
rule A: 2 or more of 6 features with PSI > 0.2
rule B: any feature with KS p < 0.05/6 (Bonferroni, 6 tests)
week south share #PSI>0.2      worst feature (PSI)  min KS p    A    B    MAE
   1        0.00        0        AveBedrms (0.02)   1.1e-01   no   no  0.449
   2        0.00        0       Population (0.01)   7.5e-02   no   no  0.449
   3        0.10        0         HouseAge (0.04)   8.6e-02   no   no  0.466
   4        0.25        0         AveRooms (0.03)   1.1e-03   no  YES  0.466
   5        0.50        0         AveOccup (0.07)   2.7e-06   no  YES  0.468
   6        0.75        0         AveOccup (0.09)   3.7e-09   no  YES  0.485
   7        1.00        1         AveOccup (0.21)   1.1e-16   no  YES  0.524
   8        1.00        1         AveOccup (0.27)   1.2e-19   no  YES  0.501
false alarms over 200 batches drawn from the SAME north rows:
  rule A (PSI):        0 of 200
  rule B (KS, Bonf.):  0 of 200
  plain 'any p < 0.05': 28 of 200
  MAE on those quiet batches: mean 0.459, std 0.012, 95th percentile 0.478
```

- **Reading it:**
  - **Rule B** (Kolmogorov-Smirnov with a Bonferroni correction) raised the alarm in week 4, when 25% of the rows were from the south. The real error was **not yet clearly worse** then: 0.466 against a normal range whose 95th percentile was 0.478 on batches with no drift. The error first went above that range in week 6 (0.485). So the input alarm came about **two weeks earlier** than the error signal.
  - **Rule A** (PSI) **never fired**. Even in weeks 7 and 8, with every row from the south, only one feature (average occupancy) had PSI above 0.2. The textbook rule "two features above 0.2" was too strict for this drift: the drift is spread thinly across many features.
  - **False alarms:** on 200 batches drawn from the same northern rows, rules A and B fired 0 times, but the plain rule "any feature with p below 0.05" fired **28 times** (14%).
  - The error went from 0.449 (week 1) to 0.524 at its worst (week 7), about **17% worse**.
  - Rule B was chosen **after** I saw the first results. The Bonferroni correction is a standard fix, but this is one simulated stream and I did not tune it on separate data.
- **Code: what PSI and the KS test do on one made-up feature.** A feature that is a standard bell shape in training; in production its mean moves by 0 to 0.5 (spread unchanged). 200 batches per row, batch sizes 50 and 600. Then the multiple-testing trap on six drift-free features.

```python
import numpy as np
from scipy.stats import ks_2samp

rs = np.random.RandomState(0)
ref = rs.normal(0, 1, 5000)                                   # what the model saw in training: one feature, mean 0, spread 1


def psi(expected, actual, bins=10):                           # bin edges are the deciles of the training data
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected); a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


print("one feature, 200 batches per row; the production mean moves away from 0, the spread stays 1")
print(f"{'mean shift':>10s} {'batch size':>10s} {'mean PSI':>9s} {'batches with KS p<0.05':>24s}")
for shift in (0.0, 0.1, 0.3, 0.5):
    for n in (50, 600):
        batches = [rs.normal(shift, 1, n) for _ in range(200)]
        print(f"{shift:10.1f} {n:10d} {np.mean([psi(ref, b) for b in batches]):9.3f} {100 * np.mean([ks_2samp(ref, b).pvalue < 0.05 for b in batches]):23.0f}%")

print("\nsix independent features, NO drift at all, 1000 batches of 600 rows:")
p = np.array([[ks_2samp(ref, rs.normal(0, 1, 600)).pvalue for _ in range(6)] for _ in range(1000)])
print(f"  any feature with p < 0.05          : {100 * np.mean(p.min(1) < 0.05):.1f}% of batches (false alarms; theory 1-0.95^6 = {100 * (1 - 0.95 ** 6):.1f}%)")
print(f"  any feature with p < 0.05/6 = {0.05 / 6:.4f}: {100 * np.mean(p.min(1) < 0.05 / 6):.1f}% of batches")
```

Output (run on my machine):

```text
one feature, 200 batches per row; the production mean moves away from 0, the spread stays 1
mean shift batch size  mean PSI   batches with KS p<0.05
       0.0         50     0.210                       5%
       0.0        600     0.015                       5%
       0.1         50     0.252                      10%
       0.1        600     0.029                      64%
       0.3         50     0.372                      48%
       0.3        600     0.117                     100%
       0.5         50     0.701                      85%
       0.5        600     0.279                     100%

six independent features, NO drift at all, 1000 batches of 600 rows:
  any feature with p < 0.05          : 27.8% of batches (false alarms; theory 1-0.95^6 = 26.5%)
  any feature with p < 0.05/6 = 0.0083: 5.0% of batches
```

- **Reading it:**
  - **Batch size decides what you can see.** A mean shift of 0.1 is invisible with 50 rows (10% of batches flagged against 5% with no drift) and obvious with 600 rows (64%). A shift of 0.3 is caught 100% of the time with 600 rows.
  - **PSI is biased upward on small batches.** With **no drift at all** and 50 rows the average PSI is 0.210, already past the usual 0.2 alarm line. With 600 rows it is 0.015. A PSI rule must be used with a big enough batch.
  - **Testing many columns raises false alarms.** With six drift-free features the "any $p<0.05$" rule fired on 27.8% of batches (theory for independent tests: 26.5%). Dividing the threshold by 6 brought it to 5.0%.

## 🔴 4. Deep / formal
**Population stability index.** Cut the training values of a column into $B$ bins (I used the 10 deciles). Let $e_i$ be the share of training data in bin $i$ and $a_i$ the share of the new batch in bin $i$:

$$
\text{PSI}=\sum_{i=1}^{B}\big(a_i-e_i\big)\,\ln\frac{a_i}{e_i}
$$

**Where:**
- $B$ is the number of bins; $i$ is one bin.
- $e_i$ is the expected share (from the reference); $a_i$ is the actual share (from the batch). Each set of shares adds to 1.
- $\ln$ is the natural logarithm. A tiny floor (I used 0.0001) is put under each share so $\ln$ never sees zero.
- **Matters:** each term is zero when the shares match and positive when they differ, whichever direction. The log makes a bin that was rare and became common count a lot. Without the $(a_i-e_i)$ factor the sum could go negative and cancel. Because the shares are estimated from a finite batch, PSI has noise even with no drift, and that noise is bigger for small batches (my toy: 0.210 with 50 rows, 0.015 with 600).

**Example (tiny numbers):** three bins with expected shares $(0.5,0.3,0.2)$ and actual shares $(0.3,0.3,0.4)$ (made up, my arithmetic in Python).
- Bin 1: $(0.3-0.5)\ln(0.3/0.5)=(-0.2)(-0.5108)=0.1022$.
- Bin 2: $(0.3-0.3)\ln 1=0$.
- Bin 3: $(0.4-0.2)\ln(0.4/0.2)=0.2\cdot0.6931=0.1386$.
- $\text{PSI}=0.1022+0+0.1386=\mathbf{0.2408}$.
- Meaning: just above the 0.2 line. The rule of thumb that PSI under about 0.1 is stable and above about 0.25 is a big shift comes from credit scoring (from memory, and I used 0.2 in my lab); it is a convention, not a law.

**Kolmogorov-Smirnov statistic.** Let $F_{\text{ref}}(x)$ be the share of reference values that are at most $x$, and $F_{\text{new}}(x)$ the same for the batch (the *cumulative* curves):

$$
D=\max_{x}\,\big|F_{\text{ref}}(x)-F_{\text{new}}(x)\big|
$$

**Where:**
- $F(x)$ is the cumulative share: how much of the data lies at or below $x$ (it climbs from 0 to 1).
- $\max_x$ means: look at every $x$ and take the largest gap.
- $D$ is between 0 (identical) and 1 (completely separate).
- **Matters:** it needs no bins, so no bin choice can hide or create a difference, and it works for any shape. It looks at the **largest** gap, so it is most sensitive to a shift in the middle and less to changes in the far tails. It works for one numeric column at a time.

**Example (tiny numbers):** reference values $1,2,3,4$ and batch values $3,4,5,6$ (made up).
- At $x=2$: $F_{\text{ref}}=2/4=0.5$ and $F_{\text{new}}=0$. Gap $0.5$.
- At $x=3$: $0.75$ against $0.25$. Gap $0.5$. No $x$ gives a larger gap.
- $D=\mathbf{0.5}$. Meaning: the biggest difference between the two cumulative curves is half the data.

**Is $D$ big enough to matter?** With $n$ reference values and $m$ batch values, the test rejects "same spread" at level $\alpha$ when $D$ exceeds

$$
D_\alpha=c(\alpha)\sqrt{\frac{n+m}{nm}},\qquad c(\alpha)=\sqrt{-\tfrac12\ln\tfrac{\alpha}{2}}
$$

**Where:**
- $n,m$ are the two sample sizes. $\alpha$ is the false-alarm rate you accept (0.05 is common).
- $c(\alpha)$ is a constant: $c(0.05)=1.358$ (my arithmetic). $\ln$ is the natural logarithm.
- $\sqrt{(n+m)/(nm)}$ shrinks as the samples grow.
- This is the standard large-sample form of the test (from memory); my lab used `scipy.stats.ks_2samp` for the p-values.
- **Matters:** it says how big a gap you can trust. Small samples need a huge gap; large samples can see tiny ones. Without it you would compare $D$ with an arbitrary number.

**Example (tiny numbers):**
- $n=m=4$: $\sqrt{8/16}=0.707$, so $D_{0.05}=1.358\cdot0.707=\mathbf{0.96}$. The gap of 0.5 above is **not** significant: with four values each, nearly separate curves would be needed.
- $n=m=600$: $\sqrt{1200/360000}=0.0577$, so $D_{0.05}=\mathbf{0.078}$. A gap of 0.08 between the curves is already enough.
- Meaning: the same $D$ can be noise or a clear signal; the sample sizes decide.

**Many columns: Bonferroni.** If you test $m$ columns, each at level $\alpha_{\text{each}}$, the chance that at least one false alarm appears is about $1-(1-\alpha_{\text{each}})^m$ for independent columns. To keep the total at $\alpha$:

$$
\alpha_{\text{each}}=\frac{\alpha}{m}
$$

**Where:**
- $\alpha$ is the total false-alarm rate you accept for the whole batch (0.05 here).
- $m$ is the number of columns tested.
- $\alpha_{\text{each}}$ is the threshold for each single column's p-value.
- **Matters:** without it, more columns means more false alarms and then ignored alarms. It is conservative (it can miss a real drift) and does not use the fact that columns are correlated.

**Example (tiny numbers):** $\alpha=0.05$, $m=6$ columns (my arithmetic).
- Without the correction, independent columns give $1-0.95^6=\mathbf{0.265}$: a false alarm on about 26% of clean batches (the toy measured 27.8%).
- With it: $\alpha_{\text{each}}=0.05/6=\mathbf{0.0083}$ (the toy measured 5.0%).
- Meaning: the correction brings the batch-level false alarm rate back to about 5%.

**Things to remember.**
- **Drift is not the same as degradation.** Some input changes do not hurt the model, and the model can degrade with no input change (concept drift). In my lab, the input alarm in week 4 preceded a real error rise, but I tested one simulated stream only.
- **Univariate checks miss joint changes.** Each column can look fine while the *relationship* between columns changes. A common fix (not tried here): train a classifier to tell reference rows from new rows; if it can (AUC clearly above 0.5), the data has drifted.
- **Choose the reference carefully:** the training data, or a recent healthy window.
- **Delayed labels** mean the error check lags; input checks are the early-warning layer ([[MLOps]]).
- **Reference batch size.** The lab's KS tests compared a 5,301-row reference with 600-row batches, and PSI used the reference deciles. Do not copy the 0.2 or 0.05 numbers to another batch size without a no-drift test like the one above.

**What came before, and how this improved on it:**
- **Before:** people waited for complaints or for the delayed true answers to show a falling score, and checked data by eye. PSI comes from credit-risk scoring (from memory), the KS test from the 1930s (Kolmogorov, Smirnov; from memory), and the study of "concept drift" in ML grew from the 1990s (from memory, not checked).
- **Its limit:** the error signal arrives late (labels lag) and says nothing about why.
- **What changed:** compare the *inputs* to the training data on every batch. It needs no labels, so it works immediately.
- **How much better:** measured in my lab, the corrected input test alarmed in **week 4** and the real error left its normal range in **week 6**, two weeks later. The plain "$p<0.05$" rule fired on 28 of 200 drift-free batches (14%); the corrected rule on 0 of 200. One simulated stream, one model; no published figure quoted.

## Related
- [[MLOps]] · [[Experiment Tracking]] · [[ONNX]] · [[Anomaly Detection]] · [[Normal Distribution]] · [[Probability]] · [[Calibration]] · [[Time Series]] · [[Stationarity]] · [[Overfitting]] · [[Model Monitoring Lab]]
- [[Hypothesis Testing]] · [[Error Bars Lab]]
- [[AB Testing|A/B Testing]]
## Sources
- My own toy and lab ([[Model Monitoring Lab]]). Data: California housing (scikit-learn's copy). The formulas for PSI, the KS statistic and critical value, and Bonferroni are standard and were checked by running them (the toy reproduces the false-alarm rates, the hand examples agree with Python). The PSI rule of thumb, the history of PSI, KS and concept drift are from memory and not checked; I did not read any drift paper or the SciPy documentation.
- No image: I found no free, on-topic image of data drift on Wikimedia Commons (a search for "concept drift" returned unrelated files), and did not use an unrelated one.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/mlops Data drift:::The data a model sees in use gradually stops looking like its training data, so the model gets worse without any code change or error message.

#flashcards/ml/reverse #flashcards/topic/mlops Covariate shift, label shift, concept drift:::Covariate shift: the inputs change. Label (prior) shift: the share of each answer changes. Concept drift: the link between inputs and answer changes. The first two can be seen without answers; the third usually shows only when true answers arrive.

#flashcards/ml/concept #flashcards/topic/mlops Why does a drift alarm not prove the model got worse?
?
Some input changes do not hurt the model, and a model can degrade with no input change (concept drift). A real error check needs the true answers, which often come late.

#flashcards/ml/concept #flashcards/topic/mlops Why test inputs rather than waiting for the error?
?
Input checks need no labels, so they work immediately. Labels often arrive weeks late. In my lab the input alarm came in week 4 and the real error left its normal range in week 6.

#flashcards/ml/formula #flashcards/topic/mlops What is the population stability index?
?
$$
\text{PSI}=\sum_{i=1}^{B}\big(a_i-e_i\big)\,\ln\frac{a_i}{e_i}
$$
- $B$ is the number of bins; $e_i$ is the expected share of the reference in bin $i$ and $a_i$ the actual share of the new batch. Each set of shares adds to 1.
- Each term is zero when the shares match and positive when they differ in either direction; a tiny floor keeps $\ln$ away from zero.
- It has noise even with no drift, larger for small batches.

#flashcards/ml/concept #flashcards/topic/mlops Expected shares (0.5, 0.3, 0.2) and actual shares (0.3, 0.3, 0.4): what is the PSI?
?
Bin 1: (−0.2)·ln(0.6) = 0.1022. Bin 2: 0. Bin 3: 0.2·ln 2 = 0.1386. Sum = 0.2408, just above the 0.2 line (a rule of thumb from credit scoring, from memory). (My arithmetic.)

#flashcards/ml/formula #flashcards/topic/mlops What is the Kolmogorov-Smirnov statistic?
?
$$
D=\max_{x}\,\big|F_{\text{ref}}(x)-F_{\text{new}}(x)\big|
$$
- $F(x)$ is the cumulative share: how much of the data lies at or below $x$. $\max_x$ takes the largest gap over all $x$.
- $D$ lies between 0 (identical) and 1 (completely separate).
- It needs no bins and no shape assumption, is most sensitive to shifts in the middle, and handles one numeric column at a time.

#flashcards/ml/concept #flashcards/topic/mlops What is the KS statistic of reference 1,2,3,4 against batch 3,4,5,6?
?
At x = 2 the cumulative shares are 0.5 and 0, a gap of 0.5; at x = 3 they are 0.75 and 0.25, also 0.5. No x gives more, so D = 0.5. (My arithmetic.)

#flashcards/ml/formula #flashcards/topic/mlops When is a KS gap D big enough to reject "same spread"?
?
$$
D_\alpha=c(\alpha)\sqrt{\frac{n+m}{nm}},\qquad c(\alpha)=\sqrt{-\tfrac12\ln\tfrac{\alpha}{2}}
$$
- $n,m$ are the two sample sizes; $\alpha$ is the accepted false-alarm rate; $c(0.05)=1.358$.
- The square root shrinks as the samples grow: small samples need a huge gap, large samples can see tiny ones.
- It is the standard large-sample form (from memory); my lab used the SciPy test for p-values.

#flashcards/ml/concept #flashcards/topic/mlops What are the KS critical values at α = 0.05 for n = m = 4 and n = m = 600?
?
n = m = 4: 1.358 × sqrt(8/16) = 0.96. n = m = 600: 1.358 × sqrt(1200/360000) = 0.078. So a gap of 0.5 is noise with 4 values each, while a gap of 0.08 is enough with 600 each. (My arithmetic.)

#flashcards/ml/formula #flashcards/topic/mlops What is the Bonferroni correction for testing many columns?
?
$$
\alpha_{\text{each}}=\frac{\alpha}{m}
$$
- $\alpha$ is the total false-alarm rate accepted for the whole batch; $m$ is the number of columns tested; $\alpha_{\text{each}}$ is the threshold for each column's p-value.
- Without it, more columns means more false alarms.
- It is conservative and ignores that columns are correlated, so it can miss a real drift.

#flashcards/ml/concept #flashcards/topic/mlops With α = 0.05 and 6 independent columns, what are the false-alarm rates without and with Bonferroni?
?
Without: 1 − 0.95^6 = 0.265 (the toy measured 27.8%). With: each column at 0.05/6 = 0.0083 (the toy measured 5.0%). (My arithmetic and run.)

#flashcards/ml/concept #flashcards/topic/mlops In the drift toy, what did PSI do with no drift at all at batch sizes 50 and 600?
?
Average PSI was 0.210 with 50 rows (already past the 0.2 alarm line) and 0.015 with 600 rows. PSI is biased upward on small batches, so a PSI rule needs a big enough batch.

#flashcards/ml/concept #flashcards/topic/mlops What did the two alarm rules do in the housing lab?
?
Rule A (2 or more features with PSI above 0.2) never fired, even with all rows from the south. Rule B (any feature with a KS p-value below 0.05/6) fired from week 4 (25% southern rows). On 200 no-drift batches the plain "any p below 0.05" rule fired 28 times, A and B 0 times. B was chosen after seeing results.

#flashcards/ml/cloze #flashcards/topic/mlops Drift checks on inputs need ==no labels==, while a real error check needs the ==true answers==, which often arrive late.
