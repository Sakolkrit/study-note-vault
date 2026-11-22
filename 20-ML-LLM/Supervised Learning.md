---
tags: [machine-learning, supervised-learning, regression, classification, AI]
status: growing
level: 4
tier: 1
---

# Supervised Learning

## 🟢 1. Simple (age 12)
Supervised learning is **learning with an answer key**. You show the computer lots of examples, and each example comes with the **right answer**. The computer guesses, checks its guess against the answer key, and gets a little better each time. After enough practice it can answer new questions it has never seen.

**Example:** You teach a young child animals with picture cards. You hold up a card and say "cow". Hold up another and say "dog". The child looks at the picture and hears the correct name every time. After a lot of cards, you show a **new** photo of a cow they have never seen, and they say "cow". The names on the cards were the "supervision".

## 🔵 2. How it works
1. Collect **examples with answers**. Each example has an **input** (a photo, an email, the size of a house) and a **label** or **target** (the correct answer: "cow", "spam", "£200,000").
2. Split the examples into three piles: a big **training** pile to learn from, a **validation** pile to choose settings, and a **test** pile kept secret until the very end.
3. Pick a model, such as a straight line or a [[Neural Network]]. At the start its settings are random, so its guesses are poor.
4. **Guess and score:** for each training example, the model guesses. A [[Loss Function]] measures how far the guess is from the right answer.
5. **Learn:** nudge the model's settings a little to make the loss smaller ([[Gradient Descent]]). Repeat over all the examples many times.
6. **Check on new data:** run the model on the validation pile, then at the end on the test pile. The test score tells you how it will do in real life.
7. Use it: give it a new input, and it predicts the answer.

Supervised learning comes in **two main flavours**, depending on what the answer looks like:
- **[[Regression]]:** the answer is a **number** (a price, a temperature, tomorrow's sales).
- **[[Classification]]:** the answer is a **category** (spam or not spam, cat or dog, which digit).

![[supervised-unsupervised-selfsupervised.jpg]]
*Three ways of learning from a cow picture. **Supervised** (left): the picture comes with a target label "COW", the answer key. **Unsupervised** (middle): only the picture, no label (see [[Unsupervised Learning]]). **Self-supervised** (right): the label comes from other data that appears with the input, here the sound "moo". The short remarks under each heading ("implausible labels", "limited power") are the authors' own opinions, not facts. Image: Xiao Liu et al., [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Supervised,_Unsupervised,_and_Self-Supervised_Learning.jpg), CC BY 4.0.*

## 🟠 3. Technical detail
- **Data:** pairs $(x,y)$. The input $x$ is usually a list of numbers (a [[Vector]]) called **features**. The output $y$ is the **label**.
- **Model family:** linear models ([[Regression]], logistic regression), decision trees, nearest neighbours, support vector machines, and [[Neural Network]]s. All learn a function from $x$ to $y$.
- **Training:** pick a loss, then minimise it on the training pile, usually by [[Gradient Descent]] with [[Backpropagation]] for networks.
- **Three-way split:** training for fitting, validation for choosing settings (such as model size or learning rate), and test, used **once**, for the final honest score. Peeking at the test pile many times quietly overfits to it.
- **Generalisation:** the goal is to do well on **new** data, not only on the training pile. If it does well on training and badly on new data, it has memorised ([[Overfitting]]).
- **Where labels come from:** people annotating data (slow and costly), records from the real world (for example whether a customer actually left), or other programs. Label mistakes teach the model mistakes.
- **Class imbalance, drift, leakage:** rare categories are easy to ignore; the world changes after training; and if information about the answer sneaks into the inputs, test scores look great but mean nothing.
- **Cousins:** **semi-supervised** learning uses a few labelled examples and many unlabelled ones. **Self-supervised** learning (see [[Self-Supervised Learning]]) makes labels out of the data itself, which is how an [[LLM]] is first trained: the "label" for each position is simply the next token.
- **Where you see it in this vault:** fine-tuning an LLM on prompt-and-answer pairs ([[Fine-tuning]]) is supervised learning; [[RLHF]] adds a second signal from human preferences.

```python
import numpy as np
x = np.array([0., 1., 2.]); y = np.array([1., 3., 2.])      # training examples with answers
mse = lambda w, b, x, y: ((w * x + b - y) ** 2).mean()
print(round(mse(0.5, 1.5, x, y), 3))      # 0.5    a good setting of the model's two numbers
print(round(mse(1.0, 0.0, x, y), 3))      # 1.667  a worse setting: learning would move away from this

xt = np.array([3., 4.]); yt = np.array([2., 4.])             # new, unseen test examples
print(round(mse(0.5, 1.5, xt, yt), 3))    # 0.625  usually a bit worse than on the training data
```

## 🔴 4. Deep / formal
**Learning as minimising the average loss.** Given $n$ labelled examples and a model $f_\theta$ with adjustable numbers $\theta$, supervised learning (in its common form, "empirical risk minimisation") looks for the $\theta$ that makes the average loss on the training examples as small as possible:

$$
\hat\theta=\arg\min_{\theta}\ \frac1n\sum_{i=1}^{n}\ell\big(f_\theta(x_i),\,y_i\big)
$$

**Where:**
- $(x_i,y_i)$ is the $i$-th training example: input $x_i$ and its correct answer $y_i$. $n$ is how many examples there are.
- $f_\theta$ is the model, and $f_\theta(x_i)$ is its prediction for input $x_i$. $\theta$ (theta) stands for all the model's adjustable numbers (the weights).
- $\ell(\cdot,\cdot)$ is the loss for **one** example: it compares the prediction with the answer and returns a number, 0 if perfect and larger if worse (see [[Loss Function]]).
- $\frac1n\sum_i$ averages that loss over the training set. **Matters:** the average makes the score independent of how many examples you have.
- $\arg\min_\theta$ reads "the value of $\theta$ that gives the smallest result". $\hat\theta$ ("theta-hat") is the answer: the learned settings. **Matters:** everything in supervised learning is this one idea. Different models and different losses are just different choices of $f_\theta$ and $\ell$.

**Example (tiny numbers):** three examples $(x,y)=(0,1),(1,3),(2,2)$, a model $f(x)=wx+b$ with $\theta=(w,b)$, and squared-error loss $\ell=(\text{prediction}-y)^2$.
- Try $\theta=(0.5,\,1.5)$: predictions $1.5,\ 2,\ 2.5$. Errors $0.5,\,-1,\,0.5$. Squares $0.25,\ 1,\ 0.25$. Average $=1.5/3=\mathbf{0.5}$.
- Try $\theta=(1,\,0)$: predictions $0,\ 1,\ 2$. Errors $-1,\,-2,\,0$. Squares $1,\ 4,\ 0$. Average $=5/3=\mathbf{1.667}$.
- The first setting has the smaller average loss, so learning prefers it. (It happens to be the best straight line for these points, see [[Regression]].)

**What we really care about: the true risk.** The training average is only a stand-in for how the model would do on *all* the data the world could give it:

$$
R(\theta)=\mathbb{E}_{(x,y)\sim P}\big[\ell\big(f_\theta(x),\,y\big)\big]\ \approx\ \frac1m\sum_{j=1}^{m}\ell\big(f_\theta(x_j^{\text{test}}),\,y_j^{\text{test}}\big)
$$

**Where:**
- $P$ is the (unknown) process that produces input-and-answer pairs in the real world. $(x,y)\sim P$ means "a pair drawn from it" (see [[Probability]]).
- $\mathbb{E}[\cdot]$ is the **expected value**: the average over everything $P$ could produce. $R(\theta)$ is the **true risk**, the number we would love to know.
- $x_j^{\text{test}},y_j^{\text{test}}$ are $m$ held-out examples that were **not** used for training. The right-hand average over them is an estimate of $R$.
- **Matters:** the training average can be made tiny by memorising ([[Overfitting]]) and then says nothing about $R$. The held-out average is an honest estimate, because the model never saw those examples. The gap between them is the generalisation gap.

**Example (tiny numbers):** the same model $\theta=(0.5,1.5)$ and two new examples $(3,2)$ and $(4,4)$ that it never saw.
- Predictions: $0.5\cdot3+1.5=3.0$ and $0.5\cdot4+1.5=3.5$. Errors: $1.0$ and $-0.5$. Squares: $1$ and $0.25$.
- Test average $=(1+0.25)/2=\mathbf{0.625}$, higher than the training average $0.5$.
- Meaning: 0.5 flattered the model. 0.625 is the more honest forecast of how it will do on new data (with only two test points it is a noisy estimate).

**Further notes:**
- **Why learning can work at all:** if the training examples are a fair sample of the real world and the model is not too flexible for the amount of data, low training loss implies low true risk. Learning theory (for example VC dimension and generalisation bounds) makes this precise. Stated from memory; see a textbook.
- **Bias-variance:** a too-simple model is wrong in a consistent way (bias), and a too-flexible one swings with each new training set (variance). The best model balances them.
- **Noisy labels:** if the "answer key" is wrong 10% of the time, even a perfect model cannot score above roughly 90% on those labels.
- **No free lunch:** no model is best on every possible problem. Success depends on matching the model to the structure of the data.
- **Labels are expensive:** this is the main motivation for [[Unsupervised Learning]] and [[Self-Supervised Learning]], which learn from data without human-written answers.

**What came before, and how this improved on it:**
- **Before:** a lookup table of past cases: if the exact case was seen, reuse its answer.
- **Its limit:** new cases are almost never exactly like old ones.
- **What changed:** learn a rule from labelled pairs that applies to inputs never seen before.
- **How much better:** in the code below, a lookup table of the training pictures of handwritten digits recognises 0.0 of the new pictures (no 64-pixel picture matched exactly). A model trained on the same pictures and labels gets 0.956 of the new ones right.

**Check it yourself:**

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
seen = {tuple(r) for r in Xtr}
hit = np.mean([tuple(r) in seen for r in Xte])
print("a lookup table of the training pictures recognises", round(hit, 3), "of new pictures")
print("a model trained on the same pictures and labels gets", round(LogisticRegression(max_iter=3000).fit(Xtr, ytr).score(Xte, yte), 3))
```

## Related
- [[Regression]] · [[Classification]] · [[Unsupervised Learning]] · [[Self-Supervised Learning]] · [[Loss Function]] · [[Gradient Descent]] · [[Overfitting]] · [[Neural Network]] · [[Probability]] · [[LLM]] · [[Fine-tuning]] · [[RLHF]] · [[Reinforcement Learning]] · [[Evaluation Metrics]] · [[k-NN]] · [[Support Vector Machine]] · [[Linear Regression]]

## Sources
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (free online)
- Goodfellow, Bengio & Courville, *Deep Learning*, 2016, chapter 5
- Ng, *Machine Learning* course (Stanford and Coursera lectures)
- Shalev-Shwartz & Ben-David, *Understanding Machine Learning* (generalisation theory; from memory)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/ml-foundations Supervised learning:::Learning with an answer key. Every example comes with the right answer. The computer guesses, checks against the key, and gets a little better each time.

#flashcards/ml/formula #flashcards/topic/ml-foundations Write the supervised training objective and explain each symbol.
?
$$\hat\theta=\arg\min_{\theta}\ \frac1n\sum_{i=1}^{n}\ell\big(f_\theta(x_i),\,y_i\big)$$
- $(x_i,y_i)$: the $i$-th training example, input and its correct answer. $n$: how many examples.
- $f_\theta$: the model. $f_\theta(x_i)$: its prediction. $\theta$: all the model's adjustable numbers (the weights).
- $\ell(\cdot,\cdot)$: the loss for one example, how wrong a prediction is.
- $\frac1n\sum$: the average loss. $\arg\min_\theta$: the weights that make it smallest.

#flashcards/ml/formula #flashcards/topic/ml-foundations Write the true risk and how it is estimated on test data, and explain each symbol.
?
$$R(\theta)=\mathbb{E}_{(x,y)\sim P}\big[\ell\big(f_\theta(x),\,y\big)\big]\ \approx\ \frac1m\sum_{j=1}^{m}\ell\big(f_\theta(x_j^{\text{test}}),\,y_j^{\text{test}}\big)$$
- $P$: the unknown process that produces input-answer pairs in the real world. $(x,y)\sim P$: a pair drawn from it.
- $\mathbb{E}[\cdot]$: the expected value, the average over everything $P$ could produce. $R(\theta)$: the true risk, the number we would love to know.
- $x_j^{\text{test}}$, $y_j^{\text{test}}$: $m$ held-out examples. Their average loss estimates $R$ honestly because the model never trained on them.

#flashcards/ml/cloze #flashcards/topic/ml-foundations Supervised learning minimises the loss on ==training== data, and the ==test== average estimates the true risk.

#flashcards/ml/example #flashcards/topic/ml-foundations Examples $(0,1),(1,3),(2,2)$ and the model $f(x)=wx+b$. What is the average squared error for $(w,b)=(0.5,1.5)$ on the training data?::Predictions 1.5, 2, 2.5. Errors 0.5, −1, 0.5. Squares 0.25, 1, 0.25. Average 1.5/3 = 0.5.

#flashcards/ml/example #flashcards/topic/ml-foundations The same model on new examples $(3,2)$ and $(4,4)$. What is the test error?::Predictions 3.0 and 3.5. Errors 1.0 and −0.5. Squares 1 and 0.25. Average 0.625, higher than the training 0.5. The 0.5 flattered the model.

#flashcards/ml/image #flashcards/topic/ml-foundations ![[supervised-unsupervised-selfsupervised.jpg]]
What does this picture show?
?
Three ways of learning from a cow picture. Supervised: the picture comes with the label "COW". Unsupervised: the picture alone. Self-supervised: the label comes from another part of the data, here the sound "moo".
