---
tags: [machine-learning, AI, overview, hub]
status: growing
level: 4
tier: 1
---

# Machine Learning

This is the **map note**. It explains the big picture in one place and points to the detailed notes. Start here, then follow the links.

## 🟢 1. Simple (age 12)
Machine learning is **teaching a computer by showing it examples instead of writing out the rules**. Nobody can write down the exact rules for "what does a cat look like?". But if you show a computer thousands of cat photos, it can work out the pattern itself, and then recognise a cat it has never seen.

**Example:** Learning to ride a bike. Nobody hands you a rulebook of exactly how to balance. You wobble, fall, adjust, and try again, and slowly your body learns the pattern. A machine-learning program does the same with numbers: guess, check how wrong it was, adjust a little, repeat.

## 🔵 2. How it works
Every machine-learning system has the same loop:
1. **Data:** a pile of examples, turned into numbers ([[Vector]]s).
2. **Model:** a flexible formula with adjustable settings, such as a straight line or a [[Neural Network]]. At the start the settings are random.
3. **Score:** a [[Loss Function]] says how wrong the model's guesses are.
4. **Learn:** nudge the settings to lower the score ([[Gradient Descent]]). Repeat many times.
5. **Test:** try the model on data it has never seen. If it does well there too, it has really learned. If only on the old data, it memorised ([[Overfitting]]).

What makes the **kinds** of machine learning different is **what feedback the learner gets**:

| Kind | Feedback | Example | Note |
|---|---|---|---|
| Supervised | The right answer for every example | Photos labelled "cat" or "dog" | [[Supervised Learning]] |
| Unsupervised | No answers at all | Sorting customers into groups | [[Unsupervised Learning]] |
| Self-supervised | Answers made from the data itself | Guess the hidden next word | [[Self-Supervised Learning]] |
| Reinforcement | A score after acting, maybe late | Learning a game by winning points | [[Reinforcement Learning]] |

Supervised learning has two flavours: [[Regression]] (the answer is a number) and [[Classification]] (the answer is a category).

![[supervised-unsupervised-selfsupervised.jpg]]
*Three of the four kinds of feedback, shown with a cow picture. **Supervised** (left): the picture comes with the label "COW". **Unsupervised** (middle): the picture alone. **Self-supervised** (right): the label comes from another part of the data, here the sound "moo". The short remarks under each heading are the authors' opinions. Image: Xiao Liu et al., [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Supervised,_Unsupervised,_and_Self-Supervised_Learning.jpg), CC BY 4.0. (Reinforcement learning is not in this picture, see its own note for a loop diagram.)*

## 🟠 3. Technical detail
**A map of the vault, by task.**
- **Predict a number:** [[Regression]] (lines, and [[Decision Tree]]s with number leaves).
- **Predict a category:** [[Classification]] (logistic regression, [[Decision Tree]], [[Neural Network]]), with scores turned into chances by [[Softmax]].
- **Find structure without labels:** [[Unsupervised Learning]] (k-means grouping) and [[PCA]] (squeezing many numbers into few).
- **Learn from raw text:** [[Self-Supervised Learning]], the first stage of every [[LLM]] ([[Transformer]], [[Attention]], [[Tokenization]], [[Positional Encoding]]).
- **Learn by acting:** [[Reinforcement Learning]]. It is the idea behind [[RLHF]] and relates to an [[Agent]].

**The shared toolbox:**
- **Maths:** [[Probability]], [[Calculus]], [[Matrix]], [[Vector]], [[Normal Distribution]].
- **Training:** [[Loss Function]], [[Gradient Descent]], [[Backpropagation]], [[Adam Optimizer]], [[Normalization]], [[GPU]].
- **Judging:** a held-out test set, [[Perplexity]] for language models, and accuracy, precision and recall for classifiers (see [[Classification]]).
- **Using a trained LLM:** [[Prompt Engineering]], [[Fine-tuning]], [[RAG]] with an [[Embedding Model]] and a [[Vector Database]], measured with [[Cosine Similarity]].
- **Tools:** [[Python]] and [[NumPy]].

**A typical project, in order:**
1. State the question and how you will measure success.
2. Collect and clean the data. Split it into training, validation and test piles **before** looking closely.
3. Start with a simple baseline (even "always predict the average").
4. Try a better model. Tune settings on the validation pile.
5. Check once on the test pile. Look at the mistakes, not just the score.
6. Watch it after release, because the world changes.

**Common pitfalls:**
- **Data leakage:** information about the answer sneaks into the inputs, so scores look great but mean nothing.
- **Bias in the data** becomes bias in the model.
- **Class imbalance:** accuracy can be high for a useless model (see [[Classification]]).
- **Overfitting** and **underfitting** (a model too simple to capture the pattern).
- **Wrong goal:** the model optimises the score you gave it, not what you meant.

```python
import numpy as np
x = np.array([0., 1., 2.]); y = np.array([1., 3., 2.])      # training data
xt = np.array([3., 4.]); yt = np.array([2., 4.])             # new data the model never saw
mse = lambda w, b, x, y: ((w * x + b - y) ** 2).mean()
train, test = mse(0.5, 1.5, x, y), mse(0.5, 1.5, xt, yt)    # model: a line with settings w = 0.5, b = 1.5
print(train, test, test - train)                             # 0.5 0.625 0.125   the generalisation gap
```

## 🔴 4. Deep / formal
**The one idea behind it all: the generalisation gap.** What we want is a model that does well on *new* data. What we can measure while training is the error on the *old* data. The difference between the two tells us how much the model is fooling us:

$$
\text{gap}=E_{\text{test}}-E_{\text{train}}
$$

**Where:**
- $E_{\text{train}}$ is the average error (loss) on the data the model learned from.
- $E_{\text{test}}$ is the average error on held-out data the model never touched. **Matters:** it is the honest estimate of how the model will do in real use, as long as the test data is not used for choosing anything (see [[Supervised Learning]]).
- $\text{gap}$ is their difference. **Matters:** a gap near 0 means the model behaves the same on new data as on old. A big positive gap means it memorised ([[Overfitting]]). If both errors are high, the model is **underfitting**: too simple, and more training data will not fix it.

**Example (tiny numbers):** a line fitted to three points, then tried on two new points (the same numbers as in [[Supervised Learning]]).
- Training error: $(0.25+1+0.25)/3=\mathbf{0.5}$.
- Test error: $(1+0.25)/2=\mathbf{0.625}$.
- Gap: $0.625-0.5=\mathbf{0.125}$.
- Meaning: the model is slightly worse on new data than the training score suggested. With only two test points this is a noisy estimate, so do not read much into a gap this small.

**Further notes:**
- **Bias-variance trade-off:** a model too simple has high *bias* (it is wrong in a consistent way). A model too flexible has high *variance* (it changes a lot with each training set). Best generalisation sits between. (This is stated from memory in several notes; there is no dedicated note on it yet.)
- **Modern deep learning bends the classic picture:** very large networks can fit the training data perfectly and still generalise, which classic theory did not predict. This is an active research area, and I am not giving details here.
- **No free lunch:** no learning method beats all others on every problem. Success comes from matching the model's assumptions to the data.
- **What is not covered yet in this vault:** nearest neighbours, support vector machines, ensembles in detail, Bayesian methods, generative models, evaluation in depth.
**What came before, and how this improved on it:**
- **Before:** a person writes the rules by hand and the computer follows them. The phrase "machine learning" is often credited to Arthur Samuel in 1959 (from memory, not checked).
- **Its limit:** for messy inputs such as handwriting, nobody can write down all the rules.
- **What changed:** give the computer examples and let it find the rule ([[Supervised Learning]]).
- **How much better:** in the code below, telling a handwritten 1 from an 8, a hand-written rule ("more ink than an average digit means 8") is right 57.9% of the time. A model learned from half the examples is right 98.9% of the time. The rule is one I wrote quickly, so a cleverer hand rule would do better, but it would take a person's time for every new task.

**Check it yourself:**

```python
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
X, y = load_digits(return_X_y=True); m = (y == 1) | (y == 8)          # tell a 1 from an 8
Xtr, Xte, ytr, yte = train_test_split(X[m], y[m], test_size=0.5, random_state=0)
ink = lambda A: A.sum(1)
rule = ink(X).mean()                                                  # hand rule: "more ink than an average digit means 8"
print("hand-written rule (no labels used):", round(((ink(Xte) > rule) == (yte == 8)).mean(), 3))
print("learned from 50% of the examples, 64 pixels:", round(LogisticRegression(max_iter=2000).fit(Xtr, ytr).score(Xte, yte), 3))
```

## Related
- [[Supervised Learning]] · [[Unsupervised Learning]] · [[Self-Supervised Learning]] · [[Reinforcement Learning]] · [[Regression]] · [[Classification]] · [[Decision Tree]] · [[PCA]] · [[Neural Network]] · [[LLM]] · [[Overfitting]] · [[Loss Function]] · [[Gradient Descent]] · [[Bias-Variance]] · [[Regularization]] · [[Stochastic Gradient Descent]] · [[Learning Rate]] · [[Adam Optimizer]] · [[Ensemble Methods]] · [[Generative Models]] · [[Evaluation Metrics]] · [[Bayesian Methods]] · [[k-NN]] · [[Support Vector Machine]] · [[Linear Regression]] · [[Pandas]] · [[History of AI]]

## Sources
- James, Witten, Hastie & Tibshirani, *An Introduction to Statistical Learning* (free online)
- Goodfellow, Bengio & Courville, *Deep Learning*, 2016, chapter 5
- Sutton & Barto, *Reinforcement Learning: An Introduction* (free online)
- Shalev-Shwartz & Ben-David, *Understanding Machine Learning* (free online; from memory)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/ml-foundations Machine learning:::Teaching a computer by showing it examples instead of writing out the rules. After seeing thousands of cat photos, it works out the pattern itself and recognises a cat it has never seen.

#flashcards/ml/formula #flashcards/topic/ml-foundations Write the train-test gap and explain each symbol.
?
$$\text{gap}=E_{\text{test}}-E_{\text{train}}$$
- $E_{\text{train}}$: the average error (loss) on the data the model learned from.
- $E_{\text{test}}$: the average error on held-out data the model never touched. It is the honest estimate of real use, as long as the test data is not used for choosing anything.
- The gap shows how much the training score flattered the model.

#flashcards/ml/cloze #flashcards/topic/ml-foundations The test data must never be used for ==choosing== anything, otherwise the test error is no longer an ==honest== estimate.

#flashcards/ml/example #flashcards/topic/ml-foundations A line is fitted to three points and tried on two new points. The training error is 0.5 and the test error 0.625. What is the gap?::0.625 − 0.5 = 0.125. The model is slightly worse on new data than the training score suggested.

#flashcards/ml/image #flashcards/topic/ml-foundations ![[supervised-unsupervised-selfsupervised.jpg]]
What does this picture show?
?
Three of the four kinds of feedback, shown with a cow picture. Supervised (left): the picture comes with the label "COW". Unsupervised (middle): the picture alone. Self-supervised (right): the label comes from another part of the data, here the sound "moo".
