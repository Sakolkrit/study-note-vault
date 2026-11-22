---
tags: [math, probability, information-theory, entropy, machine-learning]
status: growing
level: 4
tier: 2
---

# Mutual Information

Mutual information measures **how much knowing one thing tells you about another**. It is 0 when the two are unrelated, and it grows as they become more tightly linked.

## 🟢 1. Simple (age 12)
Say I think of a number from 1 to 8 and you have to guess it. If I tell you "it is even", you have fewer numbers left to guess from. The hint gave you information. **Mutual information counts how much a hint helps.** A hint that tells you nothing (like "it is a number") has mutual information 0. A hint that gives the answer away has the most.

**Example:** Look at two things: "is it raining?" and "is the grass wet?". They go together: rain usually means wet grass. So knowing one helps you guess the other, and their mutual information is high. "Is it raining?" and "is it my cat's birthday?" have nothing to do with each other, so their mutual information is 0.

## 🔵 2. How it works
1. Take two things that vary, call them $X$ and $Y$ (rain and wet grass).
2. First measure how uncertain each one is on its own. That is its [[Entropy]]: how many yes/no questions you would need, on average, to find out its value.
3. Now measure how uncertain $X$ still is **after you know $Y$**. If $Y$ helps, the leftover uncertainty is smaller.
4. The **drop in uncertainty** is the mutual information. It is the number of yes/no questions that $Y$ saves you.
5. It is never negative. It is 0 only if $X$ and $Y$ are independent (see [[Probability]]).
6. It is the same in both directions: $X$ tells you as much about $Y$ as $Y$ tells you about $X$. That is why it is called "mutual".
7. Unlike [[Covariance]] (which only spots straight-line links), mutual information notices **any** kind of link, even a curved one.
8. In machine learning it is used to choose useful features ([[Feature Engineering]]), and it is the idea behind [[Contrastive Learning]]: make two views of the same thing carry as much shared information as possible.

![[mutual-information-venn.svg]]
*A Venn diagram of two uncertainties. Each circle is the entropy of one thing, $H(X)$ and $H(Y)$. Where they overlap is the mutual information, labelled $I(X,Y)$ in the picture (written $I(X;Y)$ in this note). The left-only and right-only parts are what is still unknown about one after you know the other, $H(X\mid Y)$ and $H(Y\mid X)$. The whole shape is the joint entropy $H(X,Y)$. Image: Wikipedia (author field on the file page), [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:VennDiagramIncludingVI.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Three equal ways to write it:** the overlap of the Venn picture; "uncertainty of $X$ minus what is left after seeing $Y$"; and the [[KL Divergence]] between the real joint distribution and the one you would get if $X$ and $Y$ were independent. The last view explains why it is 0 exactly when they are independent.
- **Units:** with $\log_2$ the unit is the **bit**. With the natural log it is the **nat**. scikit-learn's `mutual_info_score` gives nats.
- **Not just straight lines:** if $Y=X^2$ with $X$ centred on 0, the correlation is 0, but $Y$ is fully set by $X$, so the mutual information is high (see the code).
- **Estimating it is hard.** For continuous data you must bin the values or use a smarter estimator. Binning adds a small positive bias: even two unrelated variables give a tiny positive number in the code.
- **Uses:** feature selection; clustering evaluation; the training goal in [[Contrastive Learning]] and [[CLIP]], whose loss gives a lower bound on mutual information; and [[Variational Autoencoder]]-style ideas about how much a code remembers about the input (from memory).
- **Data processing:** you cannot create information by processing: if $Z$ is computed from $Y$, then $Z$ tells you no more about $X$ than $Y$ does (a standard result; from memory).

```python
import numpy as np
from sklearn.metrics import mutual_info_score

def mi_bits(p):                                    # I(X;Y) from a joint table p[x][y]
    p = np.asarray(p, float); px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    m = p > 0
    return float((p[m] * np.log2(p[m] / (px @ py)[m])).sum())

H = lambda q: float(-(q[q > 0] * np.log2(q[q > 0])).sum())
p = np.array([[0.4, 0.1], [0.1, 0.4]])             # two yes/no things that usually agree
print(round(mi_bits(p), 4), round(H(p.sum(1)) + H(p.sum(0)) - H(p.ravel()), 4))   # both routes agree
print(round(mi_bits(np.outer([.5, .5], [.5, .5])), 4))                              # independent -> 0
n = (p * 100).astype(int)                                                           # sklearn works in nats
print(round(mutual_info_score(None, None, contingency=n) / np.log(2), 4))

rng = np.random.default_rng(0)                     # y = x^2: correlation sees nothing, MI sees it
x = rng.uniform(-1, 1, 20000); y = x ** 2
c = np.histogram2d(x, y, bins=10)[0]
print(round(np.corrcoef(x, y)[0, 1], 3), round(mutual_info_score(None, None, contingency=c) / np.log(2), 3))
y2 = rng.uniform(-1, 1, 20000)                      # an unrelated y for comparison: the histogram estimate is slightly above 0
c2 = np.histogram2d(x, y2, bins=10)[0]
print(round(mutual_info_score(None, None, contingency=c2) / np.log(2), 3))
```

Output: `0.2781 0.2781` (two routes agree), `0.0` (independent), `0.2781` (scikit-learn, converted to bits), `-0.003 1.756` (for $Y=X^2$ the correlation is about 0 but the mutual information is 1.756 bits), and `0.003` (two unrelated variables: the binning bias).

## 🔴 4. Deep / formal
**Definition.** For two things with joint chances $p(x,y)$ and separate chances $p(x)$ and $p(y)$:

$$
I(X;Y)=\sum_{x,y}p(x,y)\,\log\frac{p(x,y)}{p(x)\,p(y)}
$$

**Where:**
- $p(x,y)$ is the chance that $X=x$ and $Y=y$ happen together. $p(x)$ and $p(y)$ are the chances of each on its own.
- $p(x)\,p(y)$ is what the joint chance would be **if the two were independent**.
- The fraction compares the real joint chance with that independent guess. It is above 1 for pairs that happen together more than chance, and below 1 for pairs that happen less.
- $\log$ turns the comparison into bits (base 2) or nats (base $e$). The sum averages it over all pairs, weighted by how often they occur.
- **Matters:** if $X$ and $Y$ are independent, every fraction is 1, every log is 0, and $I=0$. Without the weights $p(x,y)$, rare pairs would count as much as common ones.

**Example (tiny numbers):** two yes/no things with joint chances $0.4,\,0.1,\,0.1,\,0.4$ (the two "agree" cases are 0.4, the two "disagree" cases are 0.1). Each thing alone is 50/50, so every $p(x)\,p(y)=0.25$.
- Agree cases: $0.4\times\log_2(0.4/0.25)=0.4\times0.678=0.2712$, twice, so $0.5425$.
- Disagree cases: $0.1\times\log_2(0.1/0.25)=0.1\times(-1.322)=-0.1322$, twice, so $-0.2644$.
- Total: $0.5425-0.2644=\mathbf{0.2781}$ bits.
- Meaning: knowing one of them saves you 0.28 of a yes/no question about the other.

**In terms of entropy.** The overlap of the two circles in the picture is the two entropies added, minus the joint one:

$$
I(X;Y)=H(X)+H(Y)-H(X,Y)
$$

**Where:**
- $H(X)$ and $H(Y)$ are the entropies of each thing alone ([[Entropy]]).
- $H(X,Y)$ is the entropy of the pair taken together.
- **Matters:** if the two share no information, the pair's entropy is just the sum $H(X)+H(Y)$ and $I=0$. Shared information makes the pair less uncertain than the sum, and the gap is exactly $I$.

**Example (tiny numbers):** the same table.
- $H(X)=1$ bit and $H(Y)=1$ bit (each is 50/50).
- $H(X,Y)=-(2\times0.4\log_2 0.4+2\times0.1\log_2 0.1)=0.8\times1.3219+0.2\times3.3219=1.0575+0.6644=1.7219$ bits.
- $I=1+1-1.7219=\mathbf{0.2781}$ bits.
- Meaning: the same answer as before, from a different route. The code confirms both.

**As a distance from independence.** Mutual information is the [[KL Divergence]] from "what independence would give" to "what is true":

$$
I(X;Y)=D_{\mathrm{KL}}\big(p(x,y)\,\big\|\,p(x)\,p(y)\big)
$$

**Where:**
- $p(x,y)$ is the true joint distribution.
- $p(x)\,p(y)$ is the made-up joint distribution that treats the two as independent.
- $D_{\mathrm{KL}}$ is the KL divergence: how surprised you are, on average, when you use the made-up distribution instead of the true one.
- **Matters:** KL divergence is never negative and is 0 only when the two distributions match. So $I\ge0$, and $I=0$ exactly when $X$ and $Y$ are independent.

**Example (tiny numbers):** the same table.
- True: $(0.4,\,0.1,\,0.1,\,0.4)$. Made-up: $(0.25,\,0.25,\,0.25,\,0.25)$.
- $D_{\mathrm{KL}}=\sum p\log_2(p/q)=0.2781$ bits, the same sum as in the first example.
- Meaning: it is the same number written as "how far the truth is from independence".

**Further notes:**
- **Lower bounds:** InfoNCE (see [[Contrastive Learning]]) gives a lower bound on $I$ (from memory of the CPC paper; the bound is not checked here).
- **Relation to correlation:** for two bell-shaped variables, mutual information is a function of the correlation only (a standard result; from memory).

**What came before, and how this improved on it:**
- **Before:** correlation, from Pearson in the 1890s (from memory), as the usual measure of how linked two numbers are.
- **Its limit:** it only sees straight-line links. A perfect curved link such as $Y=X^2$ can have correlation 0.
- **What changed:** use information theory (Shannon, 1948; from memory) to ask how much knowing one tells you about the other, which counts any kind of link.
- **How much better:** in the code above, for $Y=X^2$ with 20,000 points the correlation is $-0.003$ (it sees nothing) but the mutual information is 1.756 bits, against 0.003 bits for two unrelated variables measured the same way. The 1.756 is a rough histogram estimate, not an exact value.

## Related
- [[Entropy]] · [[KL Divergence]] · [[Probability]] · [[Covariance]] · [[Contrastive Learning]] · [[CLIP]] · [[Feature Engineering]] · [[Cross-Entropy Loss]] · [[Variational Autoencoder]] · [[Bayes' Theorem]]

## Sources
- Cover & Thomas, *Elements of Information Theory*, chapter 2 (from memory)
- Shannon, "A Mathematical Theory of Communication", 1948 (from memory)
- Image: Wikimedia Commons, CC BY-SA 4.0 (see the caption in section 2)
- Numbers in this note come from the code above (my arithmetic and my run on this machine).

## Flashcards
#flashcards/math

#flashcards/math/reverse #flashcards/topic/information-theory Mutual information:::How much knowing one thing tells you about another: the drop in uncertainty about one after you see the other. It is 0 only when they are independent.

#flashcards/math/formula #flashcards/topic/information-theory Write the definition of mutual information and explain each symbol.
?
$$I(X;Y)=\sum_{x,y}p(x,y)\,\log\frac{p(x,y)}{p(x)\,p(y)}$$
- $p(x,y)$: the chance that $X=x$ and $Y=y$ happen together. $p(x)$, $p(y)$: the chances of each alone.
- $p(x)\,p(y)$: what the joint chance would be if they were independent.
- The fraction compares the real joint chance with the independent guess. The log gives bits (base 2) or nats (base $e$). The sum averages over all pairs.
- If independent, every fraction is 1, every log is 0, and $I=0$.

#flashcards/math/example #flashcards/topic/information-theory Joint chances 0.4, 0.1, 0.1, 0.4 for two yes/no things: what is $I(X;Y)$ in bits?::Each alone is 50/50, so p(x)p(y) = 0.25. Agree cases: 2 × 0.4 × log2(1.6) = 0.5425. Disagree cases: 2 × 0.1 × log2(0.4) = -0.2644. Total 0.2781 bits (my arithmetic).

#flashcards/math/formula #flashcards/topic/information-theory Write mutual information in terms of entropies and explain each symbol.
?
$$I(X;Y)=H(X)+H(Y)-H(X,Y)$$
- $H(X)$, $H(Y)$: the entropy of each thing alone.
- $H(X,Y)$: the entropy of the pair together.
- With no shared information the pair's entropy is just the sum and $I=0$. Shared information makes the pair less uncertain than the sum, and the gap is exactly $I$. In the Venn picture it is the overlap.

#flashcards/math/example #flashcards/topic/information-theory For the table 0.4, 0.1, 0.1, 0.4, compute $I$ from the entropies.::H(X) = 1 and H(Y) = 1 bit. H(X,Y) = 0.8 × 1.3219 + 0.2 × 3.3219 = 1.7219. I = 1 + 1 - 1.7219 = 0.2781 bits. Same as the direct sum (my arithmetic).

#flashcards/math/formula #flashcards/topic/information-theory Write mutual information as a KL divergence and explain each symbol.
?
$$I(X;Y)=D_{\mathrm{KL}}\big(p(x,y)\,\big\|\,p(x)\,p(y)\big)$$
- $p(x,y)$: the true joint distribution.
- $p(x)\,p(y)$: the made-up joint distribution that treats the two as independent.
- $D_{\mathrm{KL}}$: how surprised you are, on average, when you use the made-up distribution instead of the true one.
- KL is never negative and is 0 only when the two distributions match, so $I\ge0$ and $I=0$ exactly when $X$ and $Y$ are independent.

#flashcards/math/example #flashcards/topic/information-theory Why is the KL form of mutual information 0.2781 bits for the table 0.4, 0.1, 0.1, 0.4?::True (0.4, 0.1, 0.1, 0.4) versus independent (0.25, 0.25, 0.25, 0.25). The sum of p log2(p/q) is the same sum as in the direct formula: 0.2781 bits. It is the distance from independence.

#flashcards/math/example #flashcards/topic/information-theory For $Y=X^2$ with 20,000 points, what were the correlation and the mutual information?::Correlation -0.003 (sees nothing), mutual information 1.756 bits (a rough 10 by 10 histogram estimate), against 0.003 bits for two unrelated variables measured the same way. My run on this machine.

#flashcards/math/image #flashcards/topic/information-theory ![[mutual-information-venn.svg]]
What does this picture show?
?
A Venn diagram of two uncertainties. Each circle is the entropy of one thing, H(X) and H(Y). The overlap is the mutual information. The left-only and right-only parts are what is still unknown about one after you know the other. The whole shape is the joint entropy H(X,Y).

#flashcards/math/cloze #flashcards/topic/information-theory Mutual information notices ==any== kind of link between two things, while correlation only sees ==straight-line== links.

#flashcards/math/concept #flashcards/topic/information-theory Why is estimating mutual information hard for continuous data?
?
You must bin the values or use a smarter estimator, and binning adds a small positive bias: even two unrelated variables gave 0.003 bits in the note's code.
