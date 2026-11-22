---
tags: [LLM, alignment, RLHF, reward-model, preference-tuning]
status: growing
level: 4
tier: 3
---

# Reward Model

A model that gives a **score** to an answer, trained so that answers people prefer get higher scores. It is the "judge" inside [[RLHF]].

## 🟢 1. Simple (age 12)
It is hard to tell a chatbot exactly what a "good answer" is, but easy to show it **two answers and say which one is better**. A reward model learns from many such comparisons until it can look at **any** answer and give it a number: higher means "people would like this more". Then that scoring judge can coach the chatbot, much like a talent-show judge giving scores.

**Example:** Two answers to "suggest an activity for my teddy bear". A: "Watch a movie together." B: "Don't spend time with your bear." People pick A. After thousands of such picks, the reward model gives A a higher score than B, and also scores new answers it has never seen.

## 🔵 2. How it works
1. Collect **preference data**: for a prompt, two answers and a label for which is better (human ratings, or an AI judge). The lecture says about ten thousand such pairs is typical for this step.
2. Build the model: take a pretrained language model and replace its "predict the next word" part with a **score head** that outputs one number. (An encoder like [[BERT]] with a projection from its [CLS] token also works, per the lecture.)
3. Feed in (prompt, answer) and read the score $r$.
4. Train it so that for each pair, the winner's score is higher than the loser's. The loss comes from the [[Bradley-Terry Model]]: the bigger the winner's lead, the smaller the loss.
5. Use it: in [[RLHF]], the language model (the "policy") writes answers, the reward model scores them, and a reinforcement-learning method such as [[PPO]] changes the policy to earn higher scores. Or use **best-of-N**: generate $N$ answers, score them all, and keep the top one, with no reinforcement learning.
6. A danger is **reward hacking**: the policy finds odd answers that fool the reward model but that people would not like. That is why training also keeps the policy near its starting point.
7. [[Direct Preference Optimization]] removes the separate reward model by learning directly from the pairs.

## 🟠 3. Technical detail
- **Input and output:** (prompt, answer) in, one number out.
- **Architecture:** a pretrained language model with a classification head. It does not generate text.
- **Data:** pairs of answers for the same prompt. The pairs come from the model being tuned (so the scorer sees the kind of answers it will have to judge). Labels are human ratings, an AI judge or a cheap proxy.
- **Pairwise, pointwise, listwise:** a pointwise label rates one answer alone, pairwise compares two, listwise ranks several. The lecture uses pairwise.
- **Scores only mean something relative to each other:** adding the same number to every score changes nothing in the loss (level 4).
- **Evaluating reward models:** there are benchmarks that check how often a reward model ranks held-out preference pairs correctly (named in the lecture: RewardBench; I did not read it). The code reports the same kind of number on a toy.
- **Costs:** an extra model to train and to keep in memory during [[PPO]], plus the human labelling.
- **Limits:** it is only as good as the preference data. People disagree, so labels are noisy; the code includes noise on purpose.

```python
import numpy as np
rng = np.random.RandomState(0)
sig = lambda z: 1 / (1 + np.exp(-z))
d = 5
w_true = np.array([2.0, -1.0, 0.5, 0.0, 1.5])               # the hidden taste of the people who rank answers
def make(m):
    A, B = rng.randn(m, d), rng.randn(m, d)                 # features of two answers to the same prompt
    a_wins = rng.rand(m) < sig((A - B) @ w_true)            # a noisy human picks A with chance sigmoid(reward gap)
    W = np.where(a_wins[:, None], A, B); L = np.where(a_wins[:, None], B, A)
    return W, L                                             # features of the winner and the loser
Wtr, Ltr = make(3000); Wte, Lte = make(3000)
w = np.zeros(d)
loss = lambda w, W, L: float(-np.log(sig((W - L) @ w)).mean())
print('loss at w = 0:', round(loss(w, Wtr, Ltr), 4), '= ln 2 =', round(np.log(2), 4))
for _ in range(2000):                                        # gradient descent on -log sigmoid(r_winner - r_loser)
    g = -((1 - sig((Wtr - Ltr) @ w))[:, None] * (Wtr - Ltr)).mean(0)
    w -= 0.5 * g
acc = lambda w, W, L: float(((W - L) @ w > 0).mean())
print('learned weights:', w.round(2).tolist(), '| true weights:', w_true.tolist())
print('cosine between them:', round(float(w @ w_true / np.linalg.norm(w) / np.linalg.norm(w_true)), 4))
print('test loss', round(loss(w, Wte, Lte), 4), '| test pairs ranked correctly: learned', round(acc(w, Wte, Lte), 4), '| the true weights', round(acc(w_true, Wte, Lte), 4), '(the ceiling, because the humans are noisy)')
print('first prompt: reward of winner', round(float(Wte[0] @ w), 3), 'reward of loser', round(float(Lte[0] @ w), 3))
```

## 🔴 4. Deep / formal
**The training loss.** For a prompt $x$ with a preferred answer $y_w$ and a rejected answer $y_l$:

$$
\mathcal{L}(\phi)=-\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)
$$

**Where:**
- $r_\phi(x,y)$ is the reward model's score for answer $y$ to prompt $x$, and $\phi$ stands for its weights.
- $y_w$ is the winner and $y_l$ the loser of the pair.
- $\sigma(z)=1/(1+e^{-z})$ turns the score gap into a chance between 0 and 1 (the sigmoid).
- The minus log is large when that chance is small and near 0 when it is near 1 (this is the [[Loss Function]] of a [[Logistic Regression]] on score gaps).
- **Matters:** only the *gap* between the two scores enters the loss, so scores can shift by a common constant without any change. Without a loss like this, there would be no way to turn "A beats B" labels into numbers.

**Example (tiny numbers):** winner scored $1$ and loser scored $0$.
- Gap $=1$, $\sigma(1)=0.731$, loss $=-\ln0.731=\mathbf{0.313}$.
- Reverse the scores (winner $0$, loser $1$): gap $=-1$, $\sigma(-1)=0.269$, loss $=-\ln0.269=\mathbf{1.313}$.
- At equal scores, $\sigma(0)=0.5$ and the loss is $\ln2=0.693$, which is where the code's loss starts.
- Meaning: the loss rewards scoring the winner higher and punishes the opposite.

**Further notes:**
- **What the toy shows:** the code uses a linear scorer on 5 features, with hidden "true taste" weights and noisy labels from the Bradley-Terry rule. After training, its weights were $[2.2,-1.07,0.52,0.0,1.68]$ against the true $[2,-1,0.5,0,1.5]$ (cosine $0.9999$). On 3,000 fresh pairs it ranked $0.8663$ correctly, and the true weights themselves only get $0.8653$ because the labels are noisy: that is the ceiling.
- **Slightly larger weights than the truth:** the learned weights are 4% to 12% larger than the true ones. I did not test other seeds, so I cannot say whether this is systematic.
- **Not covered:** how a reward model scores whole token sequences, process rewards (scoring each step), and reward model overfitting.

**What came before, and how this improved on it:**
- **Before:** learn only from human-written good answers (supervised fine-tuning), where there is no signal about *bad* answers, and the lecture notes that writing excellent answers from scratch is harder than comparing two.
- **Its limit:** a model fine-tuned that way can still produce harmful or unhelpful answers, and the lecture calls this "misbehaving".
- **What changed:** learn a scorer from comparisons, then use it to push the model toward preferred answers, as in InstructGPT (Ouyang et al., arXiv 2203.02155, 4 March 2022, abstract read).
- **How much better:** the InstructGPT abstract reports that outputs from a 1.3B-parameter InstructGPT were preferred by human evaluators to those of the 175B GPT-3, with 100 times fewer parameters. In this note's toy, the learned scorer ranked unseen pairs correctly 86.6% of the time against a ceiling of 86.5% for the true weights.

## Related
- [[RLHF]] · [[Bradley-Terry Model]] · [[PPO]] · [[Direct Preference Optimization]] · [[GRPO]] · [[Fine-tuning]] · [[LLM]] · [[Logistic Regression]] · [[Loss Function]] · [[BERT]] · [[Reinforcement Learning]] · [[LLM as a Judge]] · [[InstructGPT 2022]]

## Sources
- Stanford CME 295 (Fall 2025), Lecture 5 slides, Afshine and Shervine Amidi
- Ouyang et al., *Training language models to follow instructions with human feedback*, 2022 (arXiv 2203.02155, abstract read)
- Lambert et al., *RewardBench*, 2024 (named in the lecture; not read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/alignment-rl Reward model:::A model that learns from many "answer A is better than answer B" comparisons until it can give any answer a score. Higher means people would like it more. It then coaches the chatbot.

#flashcards/ml/formula #flashcards/topic/alignment-rl Write the reward model loss and explain each symbol.
?
$$\mathcal{L}(\phi)=-\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)$$
- $r_\phi(x,y)$: the reward model's score for answer $y$ to prompt $x$. $\phi$: its weights.
- $y_w$: the winner and $y_l$: the loser of the pair.
- $\sigma(z)=1/(1+e^{-z})$: the sigmoid, which turns the score gap into a chance between 0 and 1.
- The minus log is large when that chance is small and near 0 when it is near 1. So training pushes the winner's score above the loser's.

#flashcards/ml/cloze #flashcards/topic/alignment-rl The reward model is trained on ==pairs== of answers, and the loss is small when the ==winner== scores higher than the loser.

#flashcards/ml/example #flashcards/topic/alignment-rl Winner scored 1 and loser scored 0. What is the loss, and what if the scores are reversed?::Gap 1, σ(1) = 0.731, loss = −ln 0.731 = 0.313. Reversed: gap −1, σ(−1) = 0.269, loss 1.313. At equal scores σ(0) = 0.5 and the loss is ln 2 = 0.693.
