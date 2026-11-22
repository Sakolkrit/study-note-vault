---
tags: [loss-function, MSE, cross-entropy, neural-network, optimization, LLM, math]
status: growing
level: 4
tier: 1
---

# Loss Function

## 🟢 1. Simple (age 12)
A loss function is a **score for how wrong the AI was**. Zero means perfect. The bigger the number, the worse the answer. Training is just trying to make this number smaller and smaller.

**Example:** You guess how many sweets are in a jar. The real number is 30. If you guess 29, you are a tiny bit wrong. If you guess 80, you are very wrong. The "wrongness score" for 80 is much bigger than for 29. The AI gets a score like that after every guess and learns from it.

## 🔵 2. How it works
1. The model makes a guess (a number, or chances for several answers).
2. We know the **right answer** (the label) from the training data.
3. The loss function compares guess and answer and returns **one number**: how wrong.
4. [[Backpropagation]] and [[Gradient Descent]] use that number to decide which settings to nudge so the next guess is less wrong.
5. Different jobs need different loss functions:
   - Guessing a **number** (price, temperature): measure the size of the gap. Typical: mean squared error (MSE).
   - Guessing a **category** (cat or dog, the next word): measure how little chance the model gave the right answer. Typical: cross-entropy (see [[Softmax]]).

![[loss-outliers.png]]
*Why the choice of loss matters: the same data (black dots, with a cluster of outliers at the bottom right) fitted with three different losses. The red line (MSE) is dragged toward the outliers, while the green and pink lines (the other losses in the legend) stay closer to the main trend. Image: M.Zkuba, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Fitting_a_straight_line_to_a_data_with_outliers.png), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Regression losses:**
  - **MSE** (mean squared error): squares each gap, so big mistakes count a lot. Smooth, easy to optimise, but sensitive to outliers (as in the picture).
  - **MAE** (mean absolute error): uses the plain size of each gap. More robust to outliers, but has a sharp corner at zero.
  - **Huber:** acts like MSE for small gaps and like MAE for big ones.
- **Classification losses:**
  - **Cross-entropy:** $-\log$ of the probability given to the correct answer. The standard loss for classifiers and for [[LLM]]s predicting the next token.
- **Loss vs metric:** the loss is what training minimises (it must be smooth so [[Gradient Descent]] works). A metric such as accuracy is what you report. They can differ.
- **Over a batch:** the loss is averaged over a small batch of examples each step.
- **Training vs validation loss:** if the loss on training data keeps falling but the loss on held-out data rises, the model is overfitting (memorising).
- **Regularisation:** extra terms are often added to the loss to discourage huge weights.

```python
import numpy as np
y_true = np.array([3., 5.]); y_pred = np.array([2.5, 6.])
print(((y_pred - y_true) ** 2).mean())          # 0.625  MSE
print(np.abs(y_pred - y_true).mean())           # 0.75   MAE

p = np.array([0.665, 0.245, 0.09])              # model's chances for 3 classes
print(-np.log(p[0]).round(3))                   # 0.408  correct class was 0
print(-np.log(p[2]).round(3))                   # 2.408  correct class was 2
```

## 🔴 4. Deep / formal
**Mean squared error.** For $n$ examples with true values $y_i$ and predictions $\hat y_i$:

$$
\mathcal{L}_{\text{MSE}}=\frac{1}{n}\sum_{i=1}^{n}\big(\hat y_i-y_i\big)^2
$$

**Where:**
- $n$ is the number of examples averaged, and $i$ numbers them $1,2,\dots,n$.
- $y_i$ is the true answer for example $i$. $\hat y_i$ ("y-hat") is the model's prediction for it.
- $\hat y_i-y_i$ is the **error**: how far off the guess was and in which direction.
- The square $(\cdot)^2$ makes every term positive, so over- and under-guesses cannot cancel out. **Matters:** it also punishes big errors much more than small ones (an error of 2 costs 4, an error of 10 costs 100), which is why outliers pull the fit so strongly, as in the picture above.
- $\frac1n\sum$ is the average. **Matters:** the loss does not grow just because you have more examples.

**Example (tiny numbers):** two examples. True values $y=(3,\,5)$. Predictions $\hat y=(2.5,\,6)$.
- Errors: $2.5-3=-0.5$ and $6-5=1$.
- Squares: $0.25$ and $1$.
- Average: $(0.25+1)/2=\mathbf{0.625}$.
- For comparison, the plain average size of the errors (MAE) is $(0.5+1)/2=0.75$.
- Meaning: lower is better. A perfect model scores 0. Training tries to push $0.625$ down.

**Cross-entropy.** For classification the model outputs chances $p_c$ for each class $c$ (usually from [[Softmax]]). With a one-hot true label $y_c$ (1 for the right class, 0 for the rest):

$$
\mathcal{L}_{\text{CE}}=-\sum_{c} y_c\,\log p_c=-\log p_{\text{correct}}
$$

**Where:**
- $c$ runs over the classes (for an [[LLM]], every possible next token).
- $y_c$ is the true label: $1$ for the correct class and $0$ for every other class.
- $p_c$ is the chance the model gave class $c$.
- Because $y_c=0$ for wrong classes, only the correct one survives, leaving $-\log p_{\text{correct}}$.
- $\log$ is the natural logarithm. **Matters:** it is $0$ when the model is certain and right ($p=1$), small when it is mostly right, and grows without limit as $p\to0$. A confident wrong answer is punished very hard.
- The leading minus sign turns the (negative) log of a chance into a positive loss.

**Example (tiny numbers):** the model gives chances $p=(0.665,\,0.245,\,0.090)$ for three classes.
- If the correct class is class 1: loss $=-\ln(0.665)=\mathbf{0.408}$ (the model was fairly confident and right).
- If the correct class is class 3: loss $=-\ln(0.090)=\mathbf{2.408}$ (the model gave it only a 9% chance, so a big loss).
- If the model had said $0.99$ for the correct class: loss $=-\ln(0.99)=0.010$, almost nothing.

**Further notes:**
- **Why these losses:** MSE is the natural loss if you assume the noise on the data is bell-shaped (Gaussian); cross-entropy is the natural loss if you want to maximise the chance the model gives to the true labels (maximum likelihood). These are standard results, stated from memory.
- **Next-token loss:** an [[LLM]] is trained with cross-entropy on the next token; the same loss, averaged over tokens, appears in [[Fine-tuning]]. The exponential of the average loss is called **perplexity**.
- **Reward-model loss:** in [[RLHF]] the loss compares two answers instead of one answer and a label.
- **Convexity:** MSE on a linear model gives a single bowl-shaped surface with one minimum. For a deep [[Neural Network]] the surface has many bumps and valleys, and [[Gradient Descent]] finds a good (not provably best) one.

**What came before, and how this improved on it:**
- **Before:** count mistakes (accuracy) or use squared error.
- **Its limit:** accuracy gives no direction to move in, since a small nudge changes nothing until an answer flips. Squared error on a probability gives a feeble push when the model is confidently wrong.
- **What changed:** use a smooth loss whose gradient is strong exactly when the model is wrong, such as cross-entropy (from information theory, see [[Entropy]]).
- **How much better:** in the code below, a model gives the right answer a probability of 0.00247. The gradient with respect to its score is -0.00492 for squared error and -0.99753 for cross-entropy, which is 203 times stronger. Accuracy gives 0.

**Check it yourself:**

```python
import numpy as np
z, y = -6.0, 1.0                                   # the model is confident and wrong: it says 0.25% for the true answer
p = 1 / (1 + np.exp(-z))
print(f"predicted probability of the right answer: {p:.5f}")
print(f"accuracy as a loss: gradient = 0 (a tiny nudge changes nothing until the answer flips)")
g_mse = 2 * (p - y) * p * (1 - p); g_ce = p - y
print(f"squared error: gradient with respect to the score = {g_mse:.5f}")
print(f"cross-entropy: gradient with respect to the score = {g_ce:.5f}  ({abs(g_ce / g_mse):.0f}x stronger push)")
print(f"loss values: squared error {(p - y) ** 2:.3f}, cross-entropy {-np.log(p):.3f}")
```

## Related
- [[Neural Network]] · [[Softmax]] · [[Backpropagation]] · [[Gradient Descent]] · [[LLM]] · [[Fine-tuning]] · [[RLHF]] · [[Logistic Regression]] · [[Entropy]] · [[KL Divergence]] · [[Linear Regression]] · [[Maximum Likelihood]] · [[Convex Optimization]] · [[GAN]] · [[Q-Learning]] · [[Matplotlib]] · [[PyTorch]] · [[Calibration]] · [[Cross-Entropy Loss]]

## Sources
- Goodfellow, Bengio & Courville, *Deep Learning*, 2016, chapters 5 and 6
- Huber, *Robust Estimation of a Location Parameter*, 1964 (Huber loss, from memory)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/training Loss function:::A score for how wrong the AI was. 0 means perfect, bigger means worse. Training tries to make it smaller and smaller.

#flashcards/ml/formula #flashcards/topic/training Write mean squared error and explain each symbol.
?
$$\mathcal{L}_{\text{MSE}}=\frac{1}{n}\sum_{i=1}^{n}\big(\hat y_i-y_i\big)^2$$
- $n$: the number of examples averaged. $i$: numbers them from 1 to $n$.
- $y_i$: the true answer. $\hat y_i$: the model's prediction.
- $\hat y_i-y_i$: the error, how far off and in which direction.
- The square makes every term positive, so over- and under-guesses cannot cancel, and it makes big errors count much more (which drags the fit towards outliers).

#flashcards/ml/formula #flashcards/topic/training Write cross-entropy loss and explain each symbol.
?
$$\mathcal{L}_{\text{CE}}=-\sum_{c} y_c\,\log p_c=-\log p_{\text{correct}}$$
- $c$: runs over the classes (for an LLM, every possible next token).
- $y_c$: the true label, 1 for the correct class and 0 for every other.
- $p_c$: the chance the model gave to class $c$. Only the correct one survives, leaving $-\log p_{\text{correct}}$.
- $\log$ is natural. It is 0 when $p=1$ and grows without bound as $p\to0$, so confident wrong answers cost a lot.

#flashcards/ml/cloze #flashcards/topic/training ==MSE== is used for numbers (regression) and ==cross-entropy== for classes. Because MSE squares errors, ==outliers== drag the fit towards them.

#flashcards/ml/example #flashcards/topic/training True values (3, 5), predictions (2.5, 6). What are the MSE and MAE?::Errors −0.5 and 1. Squares 0.25 and 1. MSE = 0.625. MAE = (0.5 + 1)/2 = 0.75.

#flashcards/ml/example #flashcards/topic/training Chances are (0.665, 0.245, 0.090). What is the cross-entropy loss if the correct class is class 1, and if it is class 3?::Class 1: −ln 0.665 = 0.408 (confident and right). Class 3: −ln 0.090 = 2.408 (it was given only 9%, so the loss is big).

#flashcards/ml/image #flashcards/topic/training ![[loss-outliers.png]]
What does this picture show?
?
The same data (with a cluster of outliers at the bottom right) fitted with three different losses. The red line (MSE) is dragged towards the outliers, while the other losses are less affected.
