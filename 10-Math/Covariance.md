---
tags: [math, probability, statistics, covariance, correlation, foundations]
status: growing
level: 4
tier: 2
---

# Covariance

A number that says **whether two things tend to go up and down together**. Its unit-free cousin is the **correlation**.

## 🟢 1. Simple (age 12)
Do taller kids tend to have bigger shoes? Usually yes: when one goes up, the other goes up too. Do hours of TV tend to go with lower marks? Maybe: when one goes up, the other goes down. **Covariance is the number for "do these two move together?"** It is positive if they rise together, negative if one rises while the other falls, and close to zero if there is no pattern.

**Example:** A teacher writes down hours studied and the score for 5 students. Students who studied more than average tend to score above average, and students who studied less tend to score below. That shared pattern gives a positive covariance.

## 🔵 2. How it works
1. You have pairs of numbers, such as (hours studied, score).
2. Find the average of each column ([[Expected Value]]).
3. For each pair, see how far each number is from its own average. Above average is positive, below is negative.
4. **Multiply the two distances for each pair.** If both are above average, or both below, the product is positive. If one is above and one below, it is negative.
5. **Average those products.** That is the covariance.
6. Positive means "move together". Negative means "move opposite". Near zero means no straight-line pattern.
7. **Problem:** the size depends on the units. Measure the score out of 1,000 instead of 10 and the covariance gets 100 times bigger without anything really changing. Dividing by both spreads ([[Variance]], then square roots) fixes that, and gives the **correlation**, always between −1 and 1.
8. **Where it shows up in AI:** a table of all the covariances between many numbers (a covariance [[Matrix]]) shows how the numbers are related. Its biggest [[Eigenvalue]]s point along the directions where the data spreads most, which is the idea behind squeezing data into fewer numbers (PCA). It also describes the spread of a multi-number [[Normal Distribution]].

![[covariance-sign.svg]]
*How the sign of the covariance shows up in a cloud of points. Top: the cloud slopes downward, so cov(X, Y) < 0. Middle: a round cloud with no slope, so cov(X, Y) ≈ 0. Bottom: the cloud slopes upward, so cov(X, Y) > 0. Image: Cmglee, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Covariance_trends.svg), CC BY-SA 4.0. I viewed it as a PNG thumbnail first.*

## 🟠 3. Technical detail
- **Definition:** $\mathrm{Cov}(X,Y)=\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]$. Also $=\mathbb{E}[XY]-\mathbb{E}[X]\,\mathbb{E}[Y]$. The Wikipedia page notes the second form is handy on paper but can lose accuracy on a computer (catastrophic cancellation).
- **Variance is a special case:** $\mathrm{Cov}(X,X)=\mathrm{Var}(X)$.
- **Sample version:** with data, divide the sum of products by $n$ or by $n-1$ (the same choice as for the variance, see [[Variance]]). NumPy's `np.cov` divides by $n-1$.
- **Independent implies zero covariance, but not the reverse.** The page's example: $X$ uniform on $[-1,1]$ and $Y=X^2$. They are completely linked, yet the covariance is 0. Covariance only sees **straight-line** relationships. The code checks it.
- **Units:** the unit of covariance is the unit of $X$ times the unit of $Y$ (page statement), so its size cannot be compared across pairs. The **Pearson correlation** $r=\mathrm{Cov}/(\sigma_X\sigma_Y)$ has no unit and lies between −1 and 1.
- **Covariance matrix:** for many columns, entry $(i,j)$ is $\mathrm{Cov}(X_i,X_j)$. It is symmetric, the diagonal holds the variances, and its eigenvalues are never negative.
- **Link to prediction:** the square of the correlation is the share of $Y$'s variance that a straight line through $X$ explains (checked on the small data set in the code).

```python
import numpy as np
rng = np.random.RandomState(0)

# 1) covariance of a small data set: average of (x - mean_x)(y - mean_y)
x = np.array([1, 2, 3, 4, 5.]); y = np.array([2, 4, 5, 4, 5.])
prods = (x - x.mean()) * (y - y.mean())
print("products of deviations:", prods, "| sum", prods.sum())
cov_pop = prods.mean(); short = (x * y).mean() - x.mean() * y.mean()
print(f"covariance (divide by n) {cov_pop:.3f} = shortcut E[XY]-E[X]E[Y] {short:.3f} | divide by n-1: {prods.sum() / 4:.3f} | numpy: {np.cov(x, y)[0, 1]:.3f}")

# 2) units: covariance changes with units, correlation does not
y100 = y * 100
corr = lambda a, b: np.cov(a, b)[0, 1] / (a.std(ddof=1) * b.std(ddof=1))
print(f"y in other units (x100): covariance {np.cov(x, y100)[0, 1]:.1f} | correlation {corr(x, y):.4f} -> {corr(x, y100):.4f}")

# 3) zero covariance is not independence: Y = X^2 with X symmetric around 0
X = rng.uniform(-1, 1, 200_000); Y = X ** 2
print(f"Y = X^2: covariance {np.cov(X, Y)[0, 1]:+.4f} (about 0), yet Y is fully determined by X")

# 4) variance of a sum needs the covariance term
vs = np.var(x + y, ddof=1); naive = np.var(x, ddof=1) + np.var(y, ddof=1)
print(f"var(X+Y) = {vs:.2f} | var X + var Y = {naive:.2f} | plus 2 cov = {naive + 2 * np.cov(x, y)[0, 1]:.2f}")

# 5) a covariance matrix of three columns: symmetric, variances on the diagonal, eigenvalues never negative
a = rng.randn(5000); b = 0.8 * a + 0.6 * rng.randn(5000); c = rng.randn(5000)
C = np.cov(np.vstack([a, b, c]))
print("covariance matrix:\n", C.round(2))
print("symmetric:", np.allclose(C, C.T), "| eigenvalues:", np.linalg.eigvalsh(C).round(3))

# 6) the biggest eigenvector points along the cloud (the idea behind PCA)
w, v = np.linalg.eigh(np.cov(np.vstack([a, b])))
print("top direction:", v[:, -1].round(3), "| angle", round(float(np.degrees(np.arctan2(v[1, -1], v[0, -1]))), 1), "degrees (up and to the right)")

# 7) share of Y's variance that a straight line through X explains = correlation squared
print(f"correlation {corr(x, y):.4f} | squared {corr(x, y) ** 2:.4f} | line fit explains {1 - np.var(y - np.polyval(np.polyfit(x, y, 1), x)) / np.var(y):.4f}")

# 8) the 2x2 covariance matrix of the small data set and its eigenvalues
print("cov matrix of x, y:", np.cov(x, y).round(2).tolist(), "| eigenvalues:", np.linalg.eigvalsh(np.cov(x, y)).round(3))
```

What it printed: for the five students, the products of deviations add to 6, so the covariance is 1.200 dividing by $n$ and 1.500 dividing by $n-1$; the shortcut gives 1.200 too, and NumPy gives 1.500. Putting $y$ in units 100 times bigger raises the covariance to 150.0 while the correlation stays at 0.7746. For $Y=X^2$ the covariance was +0.0001, about zero, even though $Y$ is fully set by $X$. The variance of $X+Y$ is 7.00, while the two variances alone add to only 4.00; adding twice the covariance gives 7.00. The 3-column covariance matrix is symmetric with variances about 1 on the diagonal, 0.82 between the first two columns (the third is unrelated), and eigenvalues 0.197, 0.988 and 1.835. The top direction of the first two columns points at 45.5 degrees. The correlation 0.7746 squared is 0.6000, which equals the share explained by a straight-line fit. The 2×2 matrix of the five students is `[[2.5, 1.5], [1.5, 1.5]]` with eigenvalues 0.419 and 3.581.

## 🔴 4. Deep / formal
**The definition.**

$$
\mathrm{Cov}(X,Y)=\mathbb{E}\big[(X-\mu_X)(Y-\mu_Y)\big]=\mathbb{E}[XY]-\mu_X\mu_Y
$$

**Where:**
- $X$ and $Y$ are two random values, with means $\mu_X$ and $\mu_Y$.
- $X-\mu_X$ and $Y-\mu_Y$ are the distances from their own averages.
- Multiplying the two distances gives a positive number when both are on the same side of their averages and a negative one when on opposite sides. **Matters:** this is what lets the sign carry the "together or opposite" message. Without the multiplication you would only see the spread of each one alone.
- $\mathbb{E}[\cdot]$ averages over all the cases.
- $\mathbb{E}[XY]-\mu_X\mu_Y$ is the shortcut form, equal by linearity of expectation.

**Example (tiny numbers):** hours studied $x=1,2,3,4,5$ and score $y=2,4,5,4,5$. The means are $3$ and $4$.
- Distances for $x$: $-2,-1,0,1,2$. For $y$: $-2,0,1,0,1$.
- Products: $4,\;0,\;0,\;0,\;2$, which add to $6$.
- Dividing by $n=5$: $\mathbf{1.2}$ (dividing by $n-1=4$ gives $\mathbf{1.5}$).
- Shortcut check: $\mathbb{E}[XY]=(2+8+15+16+25)/5=13.2$, and $\mu_X\mu_Y=3\times4=12$, so $13.2-12=\mathbf{1.2}$.
- Meaning: positive, so more study goes with higher scores in this little data set. The size 1.2 is in "hours times marks", so it only means something next to the spreads.

**Correlation: covariance without units.**

$$
r=\frac{\mathrm{Cov}(X,Y)}{\sigma_X\,\sigma_Y}
$$

**Where:**
- $\sigma_X$ and $\sigma_Y$ are the standard deviations ([[Variance]], square-rooted).
- $\mathrm{Cov}(X,Y)$ is the covariance above.
- **Matters:** dividing by both spreads cancels the units, so $r$ always lies between −1 and 1. At $+1$ or $-1$ the points lie exactly on a straight line. Without the division, changing units changes the answer (the code: 1.5 became 150.0 while $r$ stayed 0.7746).

**Example (tiny numbers):** the same five students, dividing by $n-1$.
- $\mathrm{Cov}=1.5$. The variance of $x$ is $10/4=2.5$ and the variance of $y$ is $6/4=1.5$.
- $\sigma_X\sigma_Y=\sqrt{2.5}\times\sqrt{1.5}=1.5811\times1.2247=1.9365$.
- $r=1.5/1.9365=\mathbf{0.7746}$.
- Meaning: a fairly strong straight-line link. $r^2=\mathbf{0.60}$, so a straight line through hours explains 60% of the variation in scores (the code's line fit gave 0.6000).

**The covariance matrix.** For a column of $d$ random values $\mathbf{x}$ with mean vector $\boldsymbol\mu$:

$$
\Sigma=\mathbb{E}\big[(\mathbf{x}-\boldsymbol\mu)(\mathbf{x}-\boldsymbol\mu)^{\top}\big],\qquad \Sigma_{ij}=\mathrm{Cov}(x_i,x_j)
$$

**Where:**
- $\mathbf{x}$ is a column of $d$ random values and $\boldsymbol\mu$ its mean.
- $(\mathbf{x}-\boldsymbol\mu)(\mathbf{x}-\boldsymbol\mu)^{\top}$ multiplies every distance by every other distance, giving a $d\times d$ table ([[Matrix]]).
- $\Sigma_{ij}$ is the covariance between values $i$ and $j$. **Matters:** the diagonal $\Sigma_{ii}$ holds the variances, and $\Sigma$ is symmetric because $\mathrm{Cov}(X,Y)=\mathrm{Cov}(Y,X)$.
- **Matters:** its eigenvectors are the directions of the data and its eigenvalues say how much spread lies along each. The eigenvalues are never negative, because a variance cannot be negative.

**Example (tiny numbers):** the five students, with values $(x,y)$.
- $\Sigma=\begin{pmatrix}2.5&1.5\\1.5&1.5\end{pmatrix}$: variances 2.5 and 1.5 on the diagonal, covariance 1.5 off it.
- Eigenvalues: the trace is $4$ and the determinant is $2.5\times1.5-1.5^2=3.75-2.25=1.5$, so $\lambda=2\pm\sqrt{4-1.5}=2\pm1.581$, which is $\mathbf{3.581}$ and $\mathbf{0.419}$.
- Meaning: almost all the spread (3.581 of 4.0 in total) lies along one direction. The code printed the same two eigenvalues.

**Further notes:**
- **Zero covariance is not independence:** $Y=X^2$ gave a covariance of +0.0001 in the code. Only for a jointly normal pair does zero covariance mean independent, which I state from memory and did not test here.
- **PCA direction vs line fit:** the top eigenvector of the covariance matrix and the slope of a fitted line are different things. In the code, the first two columns (similar spreads) had a top direction of 45.5 degrees, while the fitted slope of the second on the first would be about 0.8. I did not run a fit on that pair.
- **Not covered:** estimation error of covariance with few samples, rank-based correlations, partial correlation.

**What came before, and how this improved on it:**
- **Before:** describing each variable by its own spread, with nothing that captured how two of them move together. This is the "just add the variances" approach, which assumes they are unrelated.
- **Its limit:** wrong whenever the two are linked. In the code, the five students have $\mathrm{Var}(X)+\mathrm{Var}(Y)=4.00$, while the true variance of $X+Y$ is 7.00, so ignoring the link understates it by about 43%.
- **What changed:** the covariance term, and then the correlation to remove units. The Wikipedia page on the Pearson correlation says the formula was derived and published by Auguste Bravais in 1844, that Galton was using the term "reversion" and the symbol $r$ in 1877 for what became "regression", and that Karl Pearson developed the coefficient from Galton's idea (with a 1895 paper on regression and inheritance in its references). The page calls its naming another example of Stigler's law (pages read, the Pearson page only for its first 100,000 of about 139,300 characters; the Wikipedia page on covariance itself has no history section, so I do not claim who coined the word "covariance").
- **How much better:** in the code, $\mathrm{Var}(X)+\mathrm{Var}(Y)+2\,\mathrm{Cov}=4.00+3.00=7.00$, which matches the true 7.00 exactly. The correlation also kept its value (0.7746) when the units changed 100 times, where the covariance went from 1.5 to 150.0.

## Related
- [[Variance]] · [[Expected Value]] · [[Probability]] · [[Matrix]] · [[Eigenvalue]] · [[Normal Distribution]] · [[Singular Value Decomposition]] · [[Law of Large Numbers]]
- [[Anomaly Detection]]
## Sources
- Wikipedia, *Covariance* (page read: definition, shortcut formula and its numeric caveat, independence example, units and correlation; it has no history section)
- Wikipedia, *Pearson correlation coefficient* (first 100,000 of about 139,300 characters read, for the history block: Bravais 1844, Galton 1877, Pearson 1895)
- Figure: [Covariance_trends.svg](https://commons.wikimedia.org/wiki/File:Covariance_trends.svg) (Cmglee), CC BY-SA 4.0, Wikimedia Commons. I viewed it as a PNG thumbnail before writing the caption.
- The code and its numbers are my own, run in this session.

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/probability Covariance:::A number for "do these two move together?". Positive if they rise together, negative if one rises while the other falls, near zero if there is no pattern.

#flashcards/math/formula #flashcards/topic/probability Write the covariance formula and explain each symbol.
?
$$\mathrm{Cov}(X,Y)=\mathbb{E}\big[(X-\mu_X)(Y-\mu_Y)\big]=\mathbb{E}[XY]-\mu_X\mu_Y$$
- $X$, $Y$: two random values, with means $\mu_X$ and $\mu_Y$.
- $X-\mu_X$, $Y-\mu_Y$: the distances from their own averages.
- The product is positive when both are on the same side of their averages and negative when on opposite sides, so the sign carries the "together or opposite" message.
- $\mathbb{E}[\cdot]$: the average over all cases. The right-hand form is the shortcut, equal by linearity of expectation.

#flashcards/math/formula #flashcards/topic/probability Write the correlation coefficient $r$ and explain each symbol.
?
$$r=\frac{\mathrm{Cov}(X,Y)}{\sigma_X\,\sigma_Y}$$
- $\mathrm{Cov}(X,Y)$: the covariance above.
- $\sigma_X$, $\sigma_Y$: the standard deviations (square root of variance).
- Dividing by both spreads cancels the units, so $r$ always lies between -1 and 1. At +1 or -1 the points lie exactly on a straight line. Without it, changing units changes the answer.

#flashcards/math/formula #flashcards/topic/probability Write the covariance matrix and explain each symbol.
?
$$\Sigma=\mathbb{E}\big[(\mathbf{x}-\boldsymbol\mu)(\mathbf{x}-\boldsymbol\mu)^{\top}\big],\qquad \Sigma_{ij}=\mathrm{Cov}(x_i,x_j)$$
- $\mathbf{x}$: a column of $d$ random values. $\boldsymbol\mu$: its mean.
- $(\mathbf{x}-\boldsymbol\mu)(\mathbf{x}-\boldsymbol\mu)^{\top}$: every distance times every other distance, a $d\times d$ table.
- $\Sigma_{ij}$: covariance between values $i$ and $j$. The diagonal holds the variances. $\Sigma$ is symmetric.
- Its eigenvectors are the directions of the data and its eigenvalues are the spread along each. They are never negative.

#flashcards/math/cloze #flashcards/topic/probability Correlation $r$ always lies between ==-1== and ==1==. It cancels the ==units== by dividing by both standard deviations.

#flashcards/math/example #flashcards/topic/probability Hours studied 1,2,3,4,5 and scores 2,4,5,4,5. What are the covariance (dividing by $n$ and by $n-1$) and $r$?::Means 3 and 4. Products add to 6. Divide by n = 5: 1.2. Divide by n−1 = 4: 1.5. r = 1.5 / (√2.5 × √1.5) = 0.7746, so r² = 0.60.

#flashcards/math/concept #flashcards/topic/probability For those five students, what is the covariance matrix and its eigenvalues?::Σ = `[[2.5, 1.5], [1.5, 1.5]]`. Trace 4, determinant 1.5, so the eigenvalues are 2 ± 1.581 = 3.581 and 0.419.

#flashcards/math/image #flashcards/topic/probability ![[covariance-sign.svg]]
What does this picture show about the sign of covariance?
?
Top: the cloud slopes downward, so cov(X, Y) < 0. Middle: a round cloud with no slope, so cov ≈ 0. Bottom: the cloud slopes upward, so cov > 0.
