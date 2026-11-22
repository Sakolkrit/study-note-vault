---
tags: [LLM, evaluation, LLM-as-a-judge, metrics, bias, kappa]
status: growing
level: 4
tier: 3
---

# LLM as a Judge

Written **LLM-as-a-Judge** or **LaaJ**. Using a language model to grade another model's answers.

## 🟢 1. Simple (age 12)
Judging thousands of AI answers by hand takes forever, and people get tired and disagree. So we ask **another AI to be the teacher**: "read the question and the answer, tell me if it is good, and say why". It is quick and cheap, and its written reasons let you see how it decided. But the AI teacher has quirks, such as liking the answer it reads first, or liking long answers, so you have to check it.

**Example:** A bear-story app. The judge AI is shown "Why are teddy bears comforting?" and the reply "They are soft and loyal companions." It writes "Direct explanation, on topic" and gives the score PASS.

## 🔵 2. How it works
1. Free-form text is hard to grade with simple rules (see [[Evaluation Metrics]]). The gold standard is **human ratings**, but they are slow, expensive and subjective.
2. People can **agree less than you expect**. Measure it with **Cohen's kappa** ([[Probability]]): how much better than chance do two raters agree?
3. Cheap rule-based scores (BLEU, ROUGE, METEOR, which compare word overlap with a reference answer) miss differences in style and correlate only modestly with human ratings, according to the lecture.
4. **LLM-as-a-Judge:** give a strong [[LLM]] the prompt, the answer and clear criteria, and ask for a **rationale** and a **score** (for example 1 if mostly relevant, 0 if not).
5. Ask for the output in a fixed format (a small data structure with `rationale` and `score`), which most model providers support, so the program can read it ([[Decoding Strategies]]).
6. Two main styles: **pointwise** (grade one answer) and **pairwise** (which of two is better).
7. Watch for biases: **position bias** (prefers the first or second slot), **verbosity bias** (prefers longer answers), **self-enhancement bias** (prefers answers written by itself).
8. Good practice, from the lecture: crisp guidelines, a simple scale (binary over fine-grained), rationale *before* the score, bias mitigation, calibrating against human judgements, and low temperature for repeatability.

## 🟠 3. Technical detail
- **Benefits over reference-based metrics:** no reference answer needed, and the rationale makes the grade interpretable.
- **The checking step:** compare the judge's grades with human grades on a sample (for example agreement rate or kappa). If agreement is poor, change the prompt before trusting the numbers.
- **Evidence:** the MT-Bench and Chatbot Arena paper reports that strong judges such as GPT-4 reached over 80% agreement with human preferences, the same level as between humans (arXiv 2306.05685 abstract, read). It also names position, verbosity and self-enhancement biases and limited reasoning ability.
- **Remedies for bias (lecture):** for position bias, run both orders and average (the code shows why this helps); for verbosity, explicit guidelines, few examples and a length penalty; for self-enhancement, avoid using a model to judge its own outputs.
- **Variants:** the lecture also discusses checking facts by splitting an answer into separate claims and verifying each (from a long-form factuality paper I did not read), and judging both quality and alignment (tone, style, safety).
- **Related benchmark ideas:** public benchmarks, human-vote leaderboards and pass-at-k for checkable tasks ([[Pass at k]]).
- **Limits:** the judge can be wrong in the same way as the model it grades, and it is itself a model whose behaviour changes between versions.

```python
import numpy as np
# 1) Cohen's kappa: how much better than chance do two raters agree? 100 answers, each rated good (1) or bad (0)
both_yes, r1_yes_only, r2_yes_only, both_no = 40, 10, 5, 45
n = both_yes + r1_yes_only + r2_yes_only + both_no
observed = (both_yes + both_no) / n
p1_yes = (both_yes + r1_yes_only) / n; p2_yes = (both_yes + r2_yes_only) / n
expected = p1_yes * p2_yes + (1 - p1_yes) * (1 - p2_yes)          # agreement if each rater answered at random with their own habits
kappa = (observed - expected) / (1 - expected)
print(f'observed agreement {observed:.2f} | expected by chance {expected:.3f} | kappa {kappa:.3f}')
# a lazy rater who always says "good" next to a rater who says good 90% of the time: high raw agreement, kappa 0
lazy = 0.9
print('lazy pair: observed', lazy, '| expected', lazy * 1 + 0 * 0.1, '| kappa', (lazy - lazy) / (1 - lazy))
# 2) position bias: a judge compares answers A (shown first) and B (shown second) and leans toward whichever comes first
sig = lambda z: 1 / (1 + np.exp(-z))
bias = 0.8                                                      # extra push toward the first slot, in log-odds
def p_first_wins(quality_gap): return sig(quality_gap + bias)    # chance the first-shown answer is picked
print('two answers of EQUAL quality: first slot wins', round(float(p_first_wins(0)), 3), '(fair would be 0.5)')
gap = 1.0                                                       # answer X is better than answer Y by 1 unit of log-odds
x_first = p_first_wins(gap); x_second = 1 - p_first_wins(-gap)
print('better answer X shown first wins', round(float(x_first), 3), '| shown second wins', round(float(x_second), 3), '| average over both orders', round(float((x_first + x_second) / 2), 3), '| unbiased judge would say', round(float(sig(gap)), 3))
# 3) the average over both orders cancels the bias exactly for equal answers
print('equal answers, average over both orders:', round(float((p_first_wins(0) + 1 - p_first_wins(0)) / 2), 3))
```

## 🔴 4. Deep / formal
**Cohen's kappa.** For two raters:

$$
\kappa=\frac{p_o-p_e}{1-p_e}
$$

**Where:**
- $p_o$ is the **observed** share of items on which the two raters agree.
- $p_e$ is the share of agreement you would **expect by chance**, given how often each rater uses each label. For two labels: $p_e=p_1p_2+(1-p_1)(1-p_2)$, where $p_1$ and $p_2$ are the shares of "good" from each rater.
- $\kappa$ is 1 for perfect agreement, 0 for no better than chance, and negative for worse than chance.
- **Matters:** raw agreement can be high just because one label is common. Dividing by $1-p_e$ removes that. (Variants for many raters are Fleiss' kappa and Krippendorff's alpha, which the lecture names but I did not work through.)

**Example (tiny numbers):** 100 answers. Both say good on 40, rater 1 only on 10, rater 2 only on 5, both say bad on 45.
- $p_o=(40+45)/100=\mathbf{0.85}$.
- Rater 1 says good on $50$, rater 2 on $45$, so $p_1=0.5$, $p_2=0.45$. $p_e=0.5\cdot0.45+0.5\cdot0.55=\mathbf{0.5}$.
- $\kappa=(0.85-0.5)/(1-0.5)=\mathbf{0.70}$.
- Meaning: 85% raw agreement sounds high, but half of it is chance, and the real agreement level is 0.70. In the code's "lazy" pair, one rater always says good and the other says good 90% of the time, so raw agreement is 0.9 but $\kappa=0$.

**Position bias in numbers.** If the judge adds a push $b$ toward the first slot, then the chance it picks the answer shown first is:

$$
P(\text{first picked})=\sigma(\Delta+b)
$$

**Where:**
- $\Delta$ is how much better the first-shown answer really is, in log-odds. $\Delta=0$ means equal quality.
- $b$ is the judge's bias toward the first slot.
- $\sigma(z)=1/(1+e^{-z})$ is the sigmoid.
- **Matters:** with $b>0$, equal answers do not get 50/50. Swapping the order and averaging gives the second slot the same push, so for equal answers the bias cancels exactly.

**Example (tiny numbers):** $b=0.8$.
- Equal answers: $\sigma(0.8)=\mathbf{0.69}$ for whichever is shown first.
- Answer X is better by $\Delta=1$: shown first it wins $\sigma(1.8)=\mathbf{0.858}$; shown second it wins $1-\sigma(-1+0.8)=\mathbf{0.55}$; the average is $\mathbf{0.704}$.
- Meaning: averaging over orders gives $0.704$, much closer to the unbiased $\sigma(1)=0.731$, but not exactly equal. The bias is made-up; real judges differ.

**Further notes:**
- **Averaging cancels only part:** for unequal answers the average is not exactly the unbiased value (code: $0.704$ against $0.731$), but it removes the sign of the error.
- **Not covered:** how to choose the judge, prompt wording effects, or training a judge model.

**What came before, and how this improved on it:**
- **Before:** human ratings (best but slow and pricey) and word-overlap metrics against references such as BLEU, ROUGE and METEOR (lecture).
- **Its limit:** the lecture says human ratings are slow, expensive and subjective, while rule-based metrics ignore style and correlate only so-so with humans, and still need reference labels.
- **What changed:** let a strong LLM grade with written criteria and a rationale (Zheng et al., arXiv 2306.05685, 9 June 2023, abstract read).
- **How much better:** the abstract reports over 80% agreement of strong judges with human preferences, the same level as between humans. I did not reproduce this. In this note's code: kappa $0.70$ for the toy raters, and averaging both orders moved a biased judge's win chance from $0.858$ or $0.55$ to $0.704$ against an unbiased $0.731$.

## Related
- [[Evaluation Metrics]] · [[LLM]] · [[Prompt Engineering]] · [[Pass at k]] · [[Reward Model]] · [[RLHF]] · [[Probability]] · [[Decoding Strategies]] · [[Agent]] · [[Benchmarks]] · [[Chatbot Arena]]

## Sources
- Stanford CME 295 (Fall 2025), Lecture 8 slides, Afshine and Shervine Amidi
- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, 2023 (arXiv 2306.05685, abstract read)
- Cohen, *A coefficient of agreement for nominal scales*, 1960 (named in the lecture; not read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation LLM-as-a-Judge:::Give a strong LLM the prompt, the answer and clear criteria, and ask for a written rationale and a score. It is quick and cheap, but it has quirks such as liking the first answer or long answers, so it must be checked.

#flashcards/ml/formula #flashcards/topic/evaluation Write Cohen's kappa and explain each symbol.
?
$$\kappa=\frac{p_o-p_e}{1-p_e}$$
- $p_o$: the observed share of items on which the two raters agree.
- $p_e$: the share of agreement expected by chance, given how often each rater uses each label. For two labels $p_e=p_1p_2+(1-p_1)(1-p_2)$, where $p_1,p_2$ are each rater's share of "good".
- $\kappa$ is 1 for perfect agreement, 0 for no better than chance, negative for worse than chance.
- Raw agreement can be high just because one label is common. Dividing by $1-p_e$ removes that.

#flashcards/ml/example #flashcards/topic/evaluation 100 answers: both say good on 40, only rater 1 on 10, only rater 2 on 5, both bad on 45. What is $\kappa$?::$p_o=(40+45)/100=0.85$. $p_1=0.5$, $p_2=0.45$, so $p_e=0.5\cdot0.45+0.5\cdot0.55=0.5$. $\kappa=(0.85-0.5)/0.5=0.70$. Half of the raw 85% agreement was chance.

#flashcards/ml/formula #flashcards/topic/evaluation Write the chance that the judge picks the first-shown answer and explain each symbol.
?
$$P(\text{first picked})=\sigma(\Delta+b)$$
- $\Delta$: how much better the first-shown answer really is, in log-odds. $\Delta=0$ means equal quality.
- $b$: the judge's bias toward the first slot.
- $\sigma(z)=1/(1+e^{-z})$: the sigmoid.
- With $b>0$, equal answers do not get 50/50. Swapping the order and averaging gives the second slot the same push, so for equal answers the bias cancels exactly.

#flashcards/ml/example #flashcards/topic/evaluation $b=0.8$. What is the first-slot win chance for equal answers, and the order-averaged win chance for an answer better by $\Delta=1$?::Equal answers: $\sigma(0.8)=0.69$ for whichever is first. Better answer first wins $\sigma(1.8)=0.858$, second wins $1-\sigma(-0.2)=0.55$, average $0.704$. That is close to the unbiased $\sigma(1)=0.731$ but not equal. The bias is made-up.

#flashcards/ml/concept #flashcards/topic/evaluation Why swap the order of the two answers and average?::A judge favours the first slot. Averaging over both orders gives each answer the first slot once, so the bias largely cancels.

#flashcards/ml/cloze #flashcards/topic/evaluation Cheap word-overlap scores such as ==BLEU==, ROUGE and METEOR correlate only ==modestly== with human ratings, according to the lecture.

#flashcards/ml/concept #flashcards/topic/evaluation Why is human rating not enough?::It is slow, expensive and subjective, and people agree less than you expect, as kappa shows.
