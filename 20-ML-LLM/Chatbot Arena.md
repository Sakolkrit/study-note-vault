---
tags: [LLM, evaluation, human-preference, ranking, Elo]
status: growing
level: 4
tier: 2
---

# Chatbot Arena

**Chatbot Arena** is an open website that ranks language models by letting people vote on which of two anonymous answers is better.

## 🟢 1. Simple (age 12)
How do you tell which chatbot is best? Ask real people. In the Arena, you type a question, two chatbots answer without telling you their names, and you vote for the answer you like more. After many votes, the chatbots are ranked like players in a chess tournament.

**Example:** A blind taste test between two sodas. Nobody knows which is which, everybody votes, and after thousands of votes you can tell which soda people prefer, and by how much.

## 🔵 2. How it works
1. A person types any question into the website.
2. Two different models, with hidden names, each answer it.
3. The person votes for the better answer. If they cannot choose, they can keep chatting, or press "tie" or "both are bad" (the paper describes these buttons).
4. After the vote, the names are revealed.
5. All the votes are collected. Each vote says "model A beat model B on this question".
6. A statistical model ([[Bradley-Terry Model]], the kind of rating idea that chess Elo is built on) turns all the head-to-head votes into one **rating** per model.
7. Sorting by rating gives the **leaderboard**.
8. **The catch:** the votes come from whoever uses the site and the questions they choose, so the ranking reflects those people's preferences, which may not match your task. The paper's own limitations section says the user base is expected to be mostly language-model hobbyists and researchers.

## 🟠 3. Technical detail
- **Source:** Chiang et al., "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference" (arXiv 2403.04132, 7 March 2024). I read the abstract and parts of the body (introduction, the data statistics, the rating method, the topic and vote-quality analyses and the limitations), taken from an HTML copy of the paper. The first author is from memory.
- **Data:** the site began collecting votes in April 2023. By 21 January 2024 it had 243,329 conversations from 90,051 users in 149 languages, covering 50 models (Table 1 of the paper). Prompts average 94.9 tokens, against 18.9 in Anthropic HH and 36.9 in OpenAssistant, so the questions are longer than in those two human-preference sets.
- **Method:** pairwise comparison from a crowd, then a [[Bradley-Terry Model]] to turn the votes into scores. The paper fits the scores by minimising binary cross-entropy ([[Loss Function]]), reweights votes when model pairs are not sampled uniformly, and reports confidence intervals. After a simulation study it chose "sandwich" robust standard errors over a pivot bootstrap because they were smaller in large samples.
- **Why pairwise votes:** people find it easier to say which of two answers is better than to give a score, and it avoids needing a fixed answer key ([[Benchmarks]]). The paper classes benchmarks by whether the questions are static or live and whether the metric is ground truth or human preference, and puts the Arena in the live, human-preference corner.
- **Vote quality:** the authors had experts relabel 160 battles (GPT-4-Turbo against Llama-2-13b-chat and against GPT-3.5-Turbo-0613). Crowd votes agreed with experts 72.8% to 83.1% of the time, while two experts agreed with each other 79.4% and 89.8% of the time. A GPT-4 judge agreed with the crowd 75.6% of the time (Table 3). The paper blames most expert disagreement on prompts without a single right answer.
- **Topics matter:** in a topic analysis, GPT-4-0613's win rate against Llama-2-70b-chat on sample prompts ranged from 96.7% (a Python game-programming cluster) down to 66.7% (poetry writing), so a single overall rating hides differences by topic (Table 2).
- **Compared with a judge model:** [[LLM as a Judge]] uses a model for the votes, and the Arena uses people. The paper also uses GPT-4 as a judge for comparison.
- **Limits:** the question mix and the voters shape the result. The paper says most users are expected to be hobbyists and researchers. I did not read how it treats answer style, so I make no claim about it.
- **In the lecture:** the slides list Chatbot Arena among the ways to evaluate chat models by human preference.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(0)
true = np.array([0.0, 100.0, 250.0, 400.0])                              # true Elo-style ratings of 4 models
def votes(n):
    a = rng.integers(0, 4, n); b = (a + rng.integers(1, 4, n)) % 4        # random pairs of different models
    p = 1 / (1 + 10 ** ((true[b] - true[a]) / 400)); return a, b, rng.random(n) < p
def fit(a, b, win):
    X = np.zeros((len(a), 4)); X[np.arange(len(a)), a] = 1; X[np.arange(len(a)), b] = -1
    m = LogisticRegression(C=1e4, fit_intercept=False, max_iter=2000).fit(X, win); r = m.coef_[0] * 400 / np.log(10); return r - r.mean()
for n in (100, 1000, 20000):
    r = fit(*votes(n)); print(f"{n:6d} votes: fitted ratings {np.round(r - r[0], 0)}  (true {true - true[0]})  order recovered: {list(np.argsort(r)) == [0, 1, 2, 3]}")
print("chance that a model 400 points stronger wins one vote:", round(1 / (1 + 10 ** (-400 / 400)), 3))
```

## 🔴 4. Deep / formal
**Win chance from a rating gap.** In the Bradley-Terry model used by the paper, each model $m$ has a coefficient $\xi_m$, and the chance that model $A$ beats model $B$ in one vote is

$$
P(A\text{ beats }B)=\frac{1}{1+e^{\,\xi_B-\xi_A}}
$$

**Where:**
- $\xi_A$ and $\xi_B$ are the Bradley-Terry coefficients of the two models, the scores to be estimated. Only their difference matters, so the paper fixes one of them to 0.
- $e$ is the number 2.718. **Matters:** the formula is the sigmoid ([[Logistic Regression]]) of the score gap, so an equal gap means a 50-50 vote and a large gap means a near-certain win.
- **Elo scale:** chess-style rating points are this same gap rescaled, with $400/\ln10\approx173.7$ points per unit of $\xi$. That is the scale used in the code below. It is my convention, and I did not read which scale the paper's leaderboard shows.

**Example (tiny numbers):**
- Equal scores: the gap is 0, so $e^0=1$ and the chance is $1/(1+1)=\mathbf{0.5}$.
- $A$ is 1 unit ahead: $e^{-1}=0.368$, so the chance is $1/1.368=\mathbf{0.731}$.
- $A$ is $\ln10=2.303$ units ahead, which is 400 Elo points: $e^{-2.303}=0.1$, so the chance is $1/1.1=\mathbf{0.909}$.
- Meaning: a small gap means a result close to a coin flip, so you need many votes to tell models apart.

**Fitting the ratings.** The ratings are found by choosing the ratings under which the observed votes are most likely ([[Maximum Likelihood]]). The paper does this by minimising the binary cross-entropy of the votes. The code below does the same as a logistic regression on "who played whom", with +1 for the first model and -1 for the second, with no intercept.

**Further notes:**
- **How many votes:** the ratings get steadier with more votes. The check below shows a 4-model order recovered from 100 votes but with large rating errors, and accurate to a few points at 20,000.
- **Ties and the interface:** voters can press "tie" or "both are bad". I did not read how those votes enter the fit, and I did not model them.
- **Intervals:** the paper reports confidence intervals for the coefficients and an approximate ranking that accounts for the uncertainty, so two close models can share a rank.

**What came before, and how this improved on it:**
- **Before:** static [[Benchmarks]] with fixed questions and answers, scored by a program or by a judge model.
- **Its limit:** fixed tests get saturated or leaked into training data, and they do not capture which answer people prefer in open-ended chat. The QLoRA abstract, for one, says that current chatbot benchmarks are not trustworthy ([[QLoRA]]).
- **What changed:** an open arena where real users vote in blind pairs, and a Bradley-Terry rating turns the votes into a ranking (arXiv 2403.04132; abstract and parts of the body read).
- **How much better:** from the paper's body: by 21 January 2024 the Arena had 243,329 conversations from 90,051 users, and its prompts averaged 94.9 tokens against 18.9 and 36.9 in two earlier human-preference sets. On 160 relabelled battles, crowd votes agreed with experts 72.8% to 83.1% of the time, close to the 79.4% and 89.8% agreement between two experts. In the code below (4 models with known true ratings, my own simulation), the order is recovered from 100, 1,000 and 20,000 random votes, but the rating error is large at 100 votes (the model with a true rating of 250 is fitted at 182) and small at 20,000 (250 recovered). **Honest limits:** the paper's own caveat is a hobbyist user base, and its topic table shows a single rating hides large differences by topic.

**Check it yourself:** run the code in level 3 above. It prints every number quoted in this block.

## Related
[[Benchmarks]] · [[LLM as a Judge]] · [[Bradley-Terry Model]] · [[Maximum Likelihood]] · [[Logistic Regression]] · [[Reward Model]] · [[RLHF]] · [[Evaluation Metrics]] · [[LLM]]

## Sources
- Stanford CME 295 (Fall 2025), lecture slides, Afshine and Shervine Amidi (lectures 4, 8 and 9, where the Arena is shown)
- Chiang et al., *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference*, arXiv 2403.04132 (2024): abstract and parts of the body read (introduction, data statistics, rating method, topic and vote-quality analyses, limitations); the first author is from memory. The numbers above were copied by hand from the paper's Tables 1 to 3.
- No image: I found no free, on-topic picture for this note on Wikimedia Commons, so I added none.
- Every number in the history block was computed by the code in this note (toy data; dates and names marked "from memory" are not checked).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation Chatbot Arena:::A blind vote. You ask a question, two anonymous chatbots answer, you vote for the one you like more, and after many votes the chatbots are ranked like chess players.

#flashcards/ml/formula #flashcards/topic/evaluation Write the Bradley-Terry win chance used for Arena rankings and explain each symbol.
?
$$P(A\text{ beats }B)=\frac{1}{1+e^{\,\xi_B-\xi_A}}$$
- $\xi_A$, $\xi_B$: the Bradley-Terry coefficients of the two models, the scores to be estimated. Only their difference matters, so the paper fixes one of them to 0.
- $e\approx2.718$.
- It is the sigmoid of the score gap, so an equal gap means a 50-50 vote and a large gap means a near-certain win.
- Elo points are this same gap rescaled, about $400/\ln10\approx173.7$ points per unit of $\xi$. That scale is the note's own convention in its code.

#flashcards/ml/example #flashcards/topic/evaluation What is the win chance for equal scores, for $A$ one unit ahead, and for $A$ ahead by $\ln10$ (400 Elo points)?::Equal: $1/(1+1)=0.5$. One unit ahead: $1/(1+e^{-1})=0.731$. $\ln10=2.303$ units ahead: $1/1.1=0.909$. A small gap is close to a coin flip, so many votes are needed to tell models apart.

#flashcards/ml/concept #flashcards/topic/evaluation Why does Arena need many votes?::Close models are near a coin flip, so a small score gap shows up only after many votes.
