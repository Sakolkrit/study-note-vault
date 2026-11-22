---
tags: [loss-function, cross-entropy, classification, LLM, probability, math, training]
status: growing
level: 4
tier: 2
---

# Cross-Entropy Loss

The score a classifier or a language model ([[LLM]]) gets for **how much probability it gave to the right answer**. High probability on the truth gives a small loss. Low probability on the truth gives a big loss.

## 🟢 1. Simple (age 12)
Imagine a quiz where you do not give one answer. You spread 100 points over the choices: "I think it is A, 70 points on A, 20 on B, 10 on C." If the answer is A, you lose only a little. If the answer is C, you lose a lot, because you put only 10 points there. And if you put 0 points on the right one and shouted that you were sure, you lose the most of all.

**Example:** A weather app says "90% sunny". If it is sunny, the app barely loses anything. If it rains, the app is punished hard, because it was so sure. Being **confident and wrong** is the worst thing, and cross-entropy is built to punish it.

## 🔵 2. How it works
1. A model finishes with a list of raw scores, one per class (or one per possible next word). A [[Softmax]] turns the scores into probabilities that add up to 1.
2. We know the true answer. Write it as a list with a 1 on the right class and 0 elsewhere ([[One-Hot Encoding]]).
3. The loss looks at **only one number: the probability the model gave to the true class**, and takes minus its logarithm. Probability 1 gives loss 0. Probability 0.5 gives 0.69. Probability 0.01 gives 4.6. Probability near 0 gives a loss that grows without limit.
4. Training ([[Gradient Descent]]) nudges the model to lower this loss, which means to give the right answer more probability.
5. The nice part: the push on each raw score is just **"the probability the model gave" minus "what it should have been"**. If the model gave 0.34 to the right class, that class's score is pushed up by 0.66. A class that should be 0 but got 0.24 is pushed down by 0.24. The bigger the mistake, the bigger the push.
6. For a language model the "class" is the next token. The loss is averaged over every position in the text ([[Autoregressive Generation]]). [[Perplexity]] is this average put through an exponential.
7. Why not just use squared error, like in [[Linear Regression]]? Because squared error goes quiet exactly when the model is confidently wrong. The code in level 3 measures this.

## 🟠 3. Technical detail
- **Two names, one thing.** "Cross-entropy", "log loss" and "negative log-likelihood" are the same number for one-hot answers (see [[Entropy]], [[Maximum Likelihood]], [[Logistic Regression]]). Minimising it means maximising the probability the model gives to the real data.
- **Averaging.** Over a batch, the loss is the mean over examples. For an LLM it is the mean over all tokens that count (not padding). Unit: nats if the natural log is used, bits for $\log_2$. Frameworks use nats.
- **Gradient.** With a softmax before it, the gradient with respect to the scores is $p-y$. It stays large when the model is wrong, so learning does not stall ([[Vanishing and Exploding Gradients]]).
- **Numerical safety.** Computing softmax and then log can fail: a tiny probability rounds to exactly 0 and $\log 0$ is infinity (this is a common cause of a loss turning into `nan`). Libraries fuse the two steps in a "log-sum-exp" form. Use their combined function (for example PyTorch's `CrossEntropyLoss`, which takes raw scores, not probabilities) and never feed it softmax output.
- **Overconfidence.** The loss can always be pushed lower by making the right class even more certain, so networks drift to extreme confidence. [[Label Smoothing]] softens the target, and [[Calibration]] measures how honest the probabilities are.
- **Not for everything.** For predicting a number (house price) use a loss such as squared error ([[Loss Function]], [[Regression]]).

```python
import numpy as np


def softmax(z):
    e = np.exp(z - z.max())                    # subtract the max so exp() cannot overflow
    return e / e.sum()


def ce_naive(z, y):
    return -np.log(softmax(z)[y])              # log of a probability that may have rounded to 0


def ce_stable(z, y):
    m = z.max()
    return np.log(np.exp(z - m).sum()) + m - z[y]    # log-sum-exp form: never takes log(0)


z, y = np.array([2.0, 1.0, 0.1]), 0            # 3 classes, the true one is class 0
p = softmax(z)
print("probabilities", p.round(3), "loss", round(float(-np.log(p[y])), 3))
print("stable form gives", round(float(ce_stable(z, y)), 3))

onehot = np.eye(3)[y]
g = p - onehot                                 # claimed gradient of the loss with respect to the scores
num = np.array([(ce_stable(z + 1e-6 * np.eye(3)[i], y) - ce_stable(z - 1e-6 * np.eye(3)[i], y)) / 2e-6 for i in range(3)])
print("gradient p - y", g.round(4), "| finite differences", num.round(4), "| max gap", float(abs(g - num).max()) < 1e-6)

zb = np.array([-1000.0, 0.0, 0.0])             # model is sure the answer is NOT class 0, but it is class 0
with np.errstate(divide="ignore"):
    print("naive loss", ce_naive(zb, 0), "| stable loss", round(float(ce_stable(zb, 0)), 1))

# cross-entropy vs squared error, 2 classes, true class 0, only the scores differ
for gap in (0, -2, -6, -10):                   # score of the right class minus the wrong one
    s = np.array([gap / 2, -gap / 2]); q = softmax(s)
    ce_grad = q[0] - 1                         # d loss / d score of the right class
    mse_grad = 2 * (q[0] - 1) * q[0] * (1 - q[0])    # same, for squared error on the probability
    print(f"gap {gap:4d}  p(right) {q[0]:.5f}  CE grad {ce_grad:+.5f}  MSE grad {mse_grad:+.7f}")

print("tokens: 4 tokens given probabilities .5 .4 .5 .25 -> mean loss", round(float(np.mean(-np.log([.5, .4, .5, .25]))), 3), "| perplexity", round(float(np.exp(np.mean(-np.log([.5, .4, .5, .25])))), 3))
```

What it printed (when run): the loss for scores (2, 1, 0.1) is 0.417 and the stable form agrees. The formula $p-y$ matched a numerical slope to six places. With a score of $-1000$ on the true class the naive version gave `inf` while the stable one gave 1000.7. In the squared-error comparison, see level 4.

## 🔴 4. Deep / formal
For one example with scores $z_1,\dots,z_K$, a softmax $p_i=e^{z_i}/\sum_j e^{z_j}$ and a one-hot answer $y$, the loss is:

$$
L=-\sum_{i=1}^{K}y_i\log p_i=-\log p_c
$$

**Where:**
- $K$ is the number of classes (for an LLM, the vocabulary size). $c$ is the index of the true class.
- $y_i$ is 1 for the true class and 0 for all others, so every term in the sum vanishes except the true class.
- $p_i$ is the probability the model gives to class $i$. $\log$ is natural, so $L$ is in nats.
- **Matters:** the minus-log shape is what makes confident mistakes cost a huge amount, while a good answer costs almost nothing. With a plain difference such as $1-p_c$ the cost of a wrong answer would stop at 1 and the model would not be pushed hard enough.

**Example (tiny numbers):** scores $z=(2,\,1,\,0.1)$ give $p=(0.659,\,0.242,\,0.099)$. The true class is the first, $y=(1,0,0)$.
- Only the first term survives: $L=-1\cdot\ln0.659$ (the other two terms are multiplied by 0).
- $L=\mathbf{0.417}$ nats.
- Meaning: the model gave the truth about two chances in three, so the loss is smallish. If it had given the truth only 0.099, the loss would be $-\ln0.099=2.31$.

Differentiating through the softmax, the push on each score is remarkably simple:

$$
\frac{\partial L}{\partial z_i}=p_i-y_i
$$

**Where:**
- $z_i$ is the raw score of class $i$ before the softmax, and $p_i$, $y_i$ are as above.
- $p_i-y_i$ is "what the model said" minus "what was true". It is negative for the true class (the score will be raised) and positive for the others (their scores will be lowered). The pushes add up to 0.
- **Matters:** the size of the push grows with the size of the mistake, and it does not shrink when the model is saturated. Squared error on the probability has an extra factor $p(1-p)$ in its gradient, which goes to 0 when the model is very sure, even if it is sure of the wrong thing.

**Example (tiny numbers):** same $p=(0.659,\,0.242,\,0.099)$ and $y=(1,0,0)$.
- $p_1-y_1=0.659-1=-0.341$
- $p_2-y_2=0.242-0=0.242$
- $p_3-y_3=0.099-0=0.099$
- Meaning: the true class's score is lifted by 0.341 per unit of step size, and the two wrong classes are pressed down by 0.242 and 0.099 (the three numbers add up to 0). The code's slope check matched these to 6 places.

**Further notes (from the code above):**
- **Confidently wrong, 2 classes.** The right class has probability 0.00247 (a score gap of $-6$). The cross-entropy gradient is $-0.99753$, almost its maximum size of 1. The squared-error gradient is $-0.0049$, about 200 times smaller. At a gap of $-10$ the two are $-0.99995$ and $-0.0000908$, about 11,000 times apart. So a model trained with squared error barely learns from its worst mistakes.
- **Link to KL.** Cross-entropy equals the entropy of the truth plus the [[KL Divergence]] to the model. With a one-hot truth the entropy term is 0, so the loss is exactly the KL distance.
- **Averages.** The tokens example in the code: four tokens given 0.5, 0.4, 0.5, 0.25 give a mean loss of 0.922 nats and a perplexity of $e^{0.922}=2.515$ ([[Perplexity]]).
- **What was not tested:** a trained network. Everything above is about one example and a fixed set of scores, not about training.
- **Caveat on "cause of nan":** the code shows the naive form returning `inf`. That a real run's `nan` comes from this is common but depends on the case and was not tested here.

**What came before, and how this improved on it:**
- **Before:** squared error between the predicted probability and the 0/1 answer, the same loss used for fitting numbers ([[Linear Regression]]). The information-theory side is Shannon's entropy (1948, **from memory**).
- **Its limit:** with a sigmoid or softmax in front, the squared-error gradient contains the factor $p(1-p)$, so when the model is confidently wrong the gradient almost disappears and learning stalls.
- **What changed:** use $-\log p_c$ instead. The log cancels the flat part of the softmax, and the gradient becomes $p-y$. It is also the maximum-likelihood loss for a model that outputs probabilities, so it has a clear meaning.
- **How much better:** in this note's code, at a score gap of $-10$ the cross-entropy gradient is $-0.99995$ and the squared-error gradient is $-0.0000908$, a ratio of about 11,000. This compares one gradient for one example, not final accuracy of trained models. When this became the standard loss for neural networks, and by whom, I did not check, so I make no date claim.

## Related
- [[Loss Function]] · [[Softmax]] · [[Entropy]] · [[KL Divergence]] · [[Maximum Likelihood]] · [[Logistic Regression]] · [[Classification]] · [[One-Hot Encoding]] · [[Label Smoothing]] · [[Calibration]] · [[Perplexity]] · [[Autoregressive Generation]] · [[Pre-training]] · [[LLM]] · [[Gradient Descent]] · [[Vanishing and Exploding Gradients]] · [[Linear Regression]] · [[Regression]]

## Sources
- Written from the vault's own notes ([[Entropy]], [[Logistic Regression]], [[Perplexity]]) and the code above, which was run and checked. No outside paper was read for this note.
- Shannon's 1948 work on information is named **from memory**, as in [[Entropy]].
- Image: none looked up for this note; the entropy curve already sits in [[Entropy]].

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/training Cross-entropy loss:::A loss that charges you for the chance you gave to the right answer. A confident wrong answer costs the most.

#flashcards/ml/formula #flashcards/topic/training Write cross-entropy loss and explain each symbol.
?
$$L=-\sum_{i=1}^{K}y_i\log p_i=-\log p_c$$
- $K$: the number of classes (for an LLM, the vocabulary size). $c$: the index of the true class.
- $y_i$: 1 for the true class and 0 for all others, so every term vanishes except the true class.
- $p_i$: the probability the model gives to class $i$. $\log$ is natural, so $L$ is in nats.
- The minus-log shape makes confident wrong answers cost a lot and good answers cost little.

#flashcards/ml/formula #flashcards/topic/training Write the gradient of cross-entropy with respect to the raw scores and explain each symbol.
?
$$\frac{\partial L}{\partial z_i}=p_i-y_i$$
- $z_i$: the raw score of class $i$ before the softmax. $p_i$, $y_i$: as above.
- $p_i-y_i$: "what the model said" minus "what was true". Negative for the true class (its score is raised), positive for the others (their scores are lowered). The pushes add up to 0.
- The push grows with the size of the mistake.

#flashcards/ml/cloze #flashcards/topic/training Cross-entropy loss is ==−log== of the chance given to the ==correct== class.

#flashcards/ml/example #flashcards/topic/training Scores (2, 1, 0.1) give $p=(0.659,0.242,0.099)$ and the true class is the first. What is the loss?::L = −ln 0.659 = 0.417 nats. The model gave the truth about two chances in three, so the loss is smallish.

#flashcards/ml/example #flashcards/topic/training For $p=(0.659,0.242,0.099)$ and true class 1, what is $p_i-y_i$ for each class?::−0.341, +0.242, +0.099. The true class score is lifted, the other two are pressed down, and the three numbers add up to 0.
