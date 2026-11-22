---
tags: [image-classification, computer-vision, cnn, classification, deep-learning, ML]
status: growing
level: 4
tier: 2
---

# Image Classification

**Image classification** means giving a picture one label from a fixed list: "cat", "digit 7", "healthy leaf". It is the classic first task of computer vision, and the one on which [[Convolutional Neural Network]]s first proved themselves. The model looks at the pixels and produces a score for every label, and the label with the highest score wins. It is a special case of [[Classification]], where the input happens to be a grid of pixel numbers.

## 🟢 1. Simple (age 12)
Imagine a friend who has seen thousands of photos with names written on the back. You show them a new photo with the name covered, and they say "that's a dog". Image classification is teaching a computer to be that friend: it practises on many labelled photos until it can name new ones by itself.

**Example:** A phone app points at a flower and says "daisy". It was practised on many photos of flowers where the name was already known.

## 🔵 2. How it works
1. **A picture is a grid of numbers.** A tiny grey picture 8 pixels wide and 8 high is 64 numbers (0 is black, 1 is white). A colour picture has three grids, one each for red, green and blue.
2. **The model turns the numbers into one score per label.** For ten digits it gives ten scores.
3. **[[Softmax]] turns scores into chances.** The ten scores become ten numbers between 0 and 1 that add up to 1, such as "7: 0.80, 1: 0.12, ...".
4. **The loss says how wrong it was.** A [[Loss Function]] (usually *cross-entropy*) is small when the right label got a big chance and large when it got a small one.
5. **Learn from mistakes.** [[Backpropagation]] and [[Gradient Descent]] nudge the model's numbers to make the loss smaller, over many labelled examples.
6. **Test on pictures it has not seen.** Accuracy on a held-back test set tells you if it really learned, or only memorised ([[Overfitting]]).
7. **Which model?** The oldest recipes use the raw pixels directly (compare with the nearest known picture). A plain [[Neural Network]] looks at all pixels at once. A [[Convolutional Neural Network]] looks at small windows with the same filter everywhere ([[Convolution]]), builds [[Feature Map]]s, and shrinks them with [[Pooling]]. That last one is the standard for pictures.

## 🟠 3. Technical detail
All results below come from the code at the end of this section. It uses the 1,797 small hand-written digits that ship with scikit-learn (8×8 grey pixels, 10 classes), split into 1,000 for training and 797 for testing, with Python 3.9.6, scikit-learn 1.1.2 and PyTorch 2.8.0 (CPU). The data are tiny and clean, so treat the results as a sketch of the ideas, not as a verdict on the methods.

- **Starting point:** an untrained CNN's loss was 2.306, next to $\ln 10=2.303$, the loss of a model that gives all ten digits the same chance.
- **The contestants** (test accuracy on the 797 held-out digits): 1-nearest-neighbour on the raw pixels 0.9824, logistic regression on the raw pixels 0.9661, a small MLP (64→64→10) 0.9674, and a small CNN (two 3×3 convolutions with 8 and 16 filters, two max-pools, one linear layer) 0.9762. On clean, centred 8×8 digits, the old "find the nearest picture" method was the best. That is a real result, and one reason these small digits are an easy benchmark.
- **Size of the models** (learned numbers): logistic regression 650, MLP 4,810, CNN 1,898. So the CNN reached almost the MLP's accuracy with 39% of its numbers (1,898 / 4,810).
- **A small shift breaks the simple models:** I moved every test picture one pixel to the right and did not retrain. Accuracy fell to 0.6349 for 1-nearest-neighbour, 0.4442 for logistic regression, 0.5019 for the MLP and 0.6876 for the CNN. Over 5 training seeds, the MLP averaged 0.9696 on normal and 0.5134 on shifted pictures, and the CNN 0.9772 and 0.6595. So the CNN lost less when the picture moved, though it still lost a lot (shared filters make it *equivariant* but the final layer is still position-specific, see [[Convolution]]).
- **Why plain networks do not scale:** equation 1. A dense first layer from an 8×8 grey picture to 64 units has 4,160 numbers, and from a 224×224 colour photo to 1,000 units it has 150,529,000. The code checks both against PyTorch's `Linear`.
- **The loss:** equation 2. Probability 0.7 on the right label gives loss 0.3567, 0.1 gives 2.3026 and 0.99 gives 0.0101.
- **Limits:** one data set, one split, 150 full-batch Adam steps, small networks. I did not tune any model. The 5-seed averages are for the two networks only (the other two methods have no randomness here). I did not test colour pictures, bigger data sets (such as ImageNet), or time.

The code covers six things: (1) the untrained loss, (2) four models on raw pixels or small windows, (3) the same models on shifted pictures, (4) five training seeds, (5) the size of a dense first layer, (6) the cross-entropy of a single answer.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
torch.set_num_threads(1)
D = load_digits(); X = (D.data / 16.0).reshape(-1, 1, 8, 8); y = D.target.astype(np.int64)
rng = np.random.RandomState(0); idx = rng.permutation(len(y)); tr, te = idx[:1000], idx[1000:]
shift = lambda a: np.pad(a, ((0, 0), (0, 0), (0, 0), (1, 0)))[:, :, :, :-1]           # move every picture one pixel to the right
Xs = shift(X[te])
def mlp(): return nn.Sequential(nn.Flatten(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
def cnn(): return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(64, 10))
def fit(make, seed=0, epochs=150):
    torch.manual_seed(seed); m = make(); opt = torch.optim.Adam(m.parameters(), 0.01)
    xt, yt = torch.tensor(X[tr], dtype=torch.float32), torch.tensor(y[tr])
    for _ in range(epochs): opt.zero_grad(); F.cross_entropy(m(xt), yt).backward(); opt.step()
    return m
acc = lambda m, a, b: float((m(torch.tensor(a, dtype=torch.float32)).argmax(1).numpy() == b).mean())
npar = lambda m: sum(p.numel() for p in m.parameters())
# 1) the loss of an untrained 10-class model is about ln 10 (it knows nothing)
torch.manual_seed(0); u = cnn(); print('untrained CNN loss:', round(F.cross_entropy(u(torch.tensor(X[tr], dtype=torch.float32)), torch.tensor(y[tr])).item(), 3), '| ln 10 =', round(float(np.log(10)), 3))
# 2) the older recipes work on the raw 64 numbers; compare them with a small MLP and a small CNN
flat = lambda a: a.reshape(len(a), -1)
knn = KNeighborsClassifier(1).fit(flat(X[tr]), y[tr]); lr = LogisticRegression(max_iter=2000).fit(flat(X[tr]), y[tr])
m_mlp, m_cnn = fit(mlp), fit(cnn)
print('test accuracy | 1-nearest-neighbour:', round(knn.score(flat(X[te]), y[te]), 4), '| logistic regression:', round(lr.score(flat(X[te]), y[te]), 4), '| MLP:', round(acc(m_mlp, X[te], y[te]), 4), '| CNN:', round(acc(m_cnn, X[te], y[te]), 4))
print('learned numbers | logistic regression:', 64 * 10 + 10, '| MLP:', npar(m_mlp), '| CNN:', npar(m_cnn))
# 3) the same test pictures moved one pixel to the right
print('shifted-test accuracy | 1-NN:', round(knn.score(flat(Xs), y[te]), 4), '| logistic regression:', round(lr.score(flat(Xs), y[te]), 4), '| MLP:', round(acc(m_mlp, Xs, y[te]), 4), '| CNN:', round(acc(m_cnn, Xs, y[te]), 4))
# 4) the same, averaged over 5 training seeds for the two networks
r = [(acc(a, X[te], y[te]), acc(a, Xs, y[te]), acc(b, X[te], y[te]), acc(b, Xs, y[te])) for a, b in ((fit(mlp, s), fit(cnn, s)) for s in range(5))]
print('mean over 5 seeds, normal | shifted: MLP', round(np.mean([q[0] for q in r]), 4), '|', round(np.mean([q[1] for q in r]), 4), '; CNN', round(np.mean([q[2] for q in r]), 4), '|', round(np.mean([q[3] for q in r]), 4))
# 5) why the first layer of a plain network explodes on a real photo: (H * W * C + 1) * n numbers
dense_first = lambda H, W, C, n: (H * W * C + 1) * n
print('dense first layer, 8x8x1 -> 64 units:', dense_first(8, 8, 1, 64), '(PyTorch:', npar(nn.Linear(64, 64)), ') | 224x224x3 -> 1000 units:', dense_first(224, 224, 3, 1000), '(PyTorch:', npar(nn.Linear(224 * 224 * 3, 1000)), ')')
# 6) the cross-entropy of one answer: -log of the probability given to the right class
print('probability 0.7 on the right class -> loss', round(-np.log(0.7), 4), '| 0.1 ->', round(-np.log(0.1), 4), '| 0.99 ->', round(-np.log(0.99), 4))
```
Reading the output (Python 3.9.6, scikit-learn 1.1.2, PyTorch 2.8.0):
- **Untrained loss:** 2.306 against $\ln 10$ = 2.303.
- **Accuracy:** 1-NN 0.9824, logistic regression 0.9661, MLP 0.9674, CNN 0.9762.
- **Numbers:** 650 against 4,810 against 1,898.
- **Shifted:** 0.6349, 0.4442, 0.5019 and 0.6876.
- **Five seeds:** MLP 0.9696 normal and 0.5134 shifted, CNN 0.9772 and 0.6595.
- **Dense first layer:** 4,160 and 150,529,000, equal to PyTorch's counts.
- **Cross-entropy:** 0.3567, 2.3026, 0.0101.
- **Limits:** the run takes about half a minute and nothing in it is timed.

## 🔴 4. Deep / formal

**Why not feed the pixels straight into a plain network?** A dense layer connects every input number to every unit, plus one bias per unit. For a picture of height $H$, width $W$ and $C$ colour channels, feeding $n$ units:

$$P_{\text{dense}}=(H\,W\,C+1)\,n$$

**Where:**
- $H$, $W$ and $C$ are the height, width and number of channels of the picture. **Matters:** $HWC$ is the number of input numbers, so the cost grows with the picture's area.
- $n$ is the number of units in the first layer. **Matters:** every unit needs its own weight for every pixel.
- The $+1$ is the bias of each unit. **Matters:** it adds $n$ numbers, which is small next to the rest.
- $P_{\text{dense}}$ is the number of learned numbers in that layer. **Matters:** more numbers need more data to pin them down, and more memory to store, so a big photo makes a plain network unusable.

**Example (tiny numbers):** a small grey digit $8\times8\times1$ with $n=64$, and a photo $224\times224\times3$ with $n=1000$.
1. Digit: $(8\cdot8\cdot1+1)\cdot64=65\cdot64=4160$.
2. Photo: $(224\cdot224\cdot3+1)\cdot1000=150{,}529\cdot1000=150{,}529{,}000$.
3. The code printed 4,160 and 150,529,000 and PyTorch's `Linear` layers reported the same.
4. Meaning: the digit's first layer is affordable, but the same recipe on a photo needs 150 million numbers for *one* layer, which is why CNNs share one small filter across the whole picture.

**How is the answer scored?** For one picture whose right label is $y$, a classifier that gives chance $p_y$ to that label has the cross-entropy loss:

$$L=-\log p_y$$

**Where:**
- $p_y$ is the chance the model gave to the *right* label (after [[Softmax]]). **Matters:** it is the only number the loss looks at, so pushing it up is how the model learns.
- $\log$ is the natural logarithm. **Matters:** it makes the loss grow without limit as $p_y$ goes to 0, so a confidently wrong model is punished very hard.
- The minus sign. **Matters:** the log of a chance (a number below 1) is negative, so the minus makes the loss positive.
- $L$ is the loss for this one picture (the training loss is the average over many). **Matters:** it is the number [[Gradient Descent]] tries to push down.

**Example (tiny numbers):** the model gives the right label chance 0.7, then 0.1, then 0.99.
1. $-\log 0.7=0.3567$.
2. $-\log 0.1=2.3026$, and $-\log 0.99=0.0101$.
3. The code printed 0.3567, 2.3026 and 0.0101. A model that spreads its chance equally over ten labels gives each 0.1, hence 2.3026 = $\ln 10$, which is where the untrained CNN started (2.306).
4. Meaning: loss near 2.3 means "knows nothing" for ten classes. Near 0 means "sure and right".

**What came before, and how this improved on it:**
- **Before:** comparing a picture with stored examples (*nearest neighbour*, pixel by pixel), and from the 1990s and 2000s, hand-made features (edges, gradient patterns) fed to a classifier such as an [[Support Vector Machine]] (from memory; I did not check the dates or names of those systems). Plain fully connected networks on pixels also existed.
- **Its limit:** pixel comparison breaks when the object moves a little, and a plain network needs one weight per pixel per unit (equation 1). Hand-made features need an expert to design them for every new task.
- **What changed:** a CNN learns its own filters, shares them across the picture and shrinks the maps, so the same few numbers work everywhere. (See [[Convolution]] and [[Feature Map]].)
- **How much better:** in this note's run, on clean 8×8 digits the simplest method (1-NN, 0.9824) was best and the CNN (0.9762) used 1,898 numbers against the MLP's 4,810. The CNN's real edge showed after a one-pixel shift: 0.66 (mean of 5 seeds) against 0.51 for the MLP. On large photos the published gap is bigger: the 2012 AlexNet's top-5 error in the ImageNet contest was 15.3%, against about 26% for the runner-up (from the Wikipedia article on AlexNet, which I read; I did not open the contest report). See [[LeNet and AlexNet]].

**Other points (from memory, not checked here):**
- Accuracy hides the kind of mistakes. Evaluation uses precision, recall and confusion tables (see [[Evaluation Metrics]]).
- Many real photos have several objects; then the task is *multi-label* classification, detection or segmentation, which I did not cover.
- A model can be confidently wrong on pictures unlike its training data. I did not test it.
- Larger data sets, deeper networks ([[ResNet]]), [[Data Augmentation]] and [[Transfer Learning]] are the standard tools to push accuracy up when data are scarce.

## Related
[[Classification]] · [[Convolutional Neural Network]] · [[Convolution]] · [[Pooling]] · [[Feature Map]] · [[Softmax]] · [[Loss Function]] · [[Neural Network]] · [[k-NN]] · [[Logistic Regression]] · [[Support Vector Machine]] · [[Overfitting]] · [[Evaluation Metrics]] · [[Data Augmentation]] · [[Transfer Learning]] · [[ResNet]] · [[LeNet and AlexNet]] · [[Vision Transformer]] · [[History of Image Classification]] · [[Transfer Learning vs Vanilla CNN]]
- [[Object Detection]] · [[Image Segmentation]] · [[VGG]]

## Sources
- scikit-learn's `load_digits` data (a copy of the UCI handwritten digits test set, from memory), loaded by the code, version 1.1.2.
- The Wikipedia article on AlexNet (read for the 15.3% figure and the runner-up, via a web fetch); it quotes the 2012 ImageNet contest.
- Library versions in this run: Python 3.9.6, scikit-learn 1.1.2, PyTorch 2.8.0 (CPU). The code and every printed value were run for this note, and nothing was timed.
- No figure: I found no on-topic free picture on Wikimedia Commons that adds anything beyond the diagrams in [[LeNet and AlexNet]], so I skipped the image.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision Image classification:::Teaching a computer to name what is in a picture. It practises on many labelled photos until it can name new ones by itself.

#flashcards/ml/formula #flashcards/topic/vision Write the number of parameters in a dense first layer for an image and explain each symbol.
?
$$P_{\text{dense}}=(H\,W\,C+1)\,n$$
- $H$, $W$, $C$: the height, width and number of channels of the picture. $HWC$ is the number of input numbers, so the cost grows with the picture's area.
- $n$: the number of units in the first layer. Every unit needs its own weight for every pixel.
- The $+1$: the bias of each unit. It adds $n$ more parameters.

#flashcards/ml/formula #flashcards/topic/vision Write the classification loss for one image and explain each symbol.
?
$$L=-\log p_y$$
- $p_y$: the chance the model gave to the right label (after softmax). It is the only number the loss looks at, so pushing it up is how the model learns.
- $\log$: the natural logarithm. It makes the loss grow without limit as $p_y\to0$, so a confidently wrong model is punished very hard.

#flashcards/ml/cloze #flashcards/topic/vision A dense first layer needs one weight per ==pixel== per unit, so big pictures make it very ==large==.

#flashcards/ml/example #flashcards/topic/vision How many parameters does a dense first layer have for an 8×8 grey digit with $n=64$ and for a 224×224×3 photo with $n=1000$?::Digit: (8·8·1 + 1) · 64 = 4,160. Photo: (224·224·3 + 1) · 1000 = 150,529,000. This is why convolutions are used instead.

#flashcards/ml/example #flashcards/topic/vision The model gives the right label chance 0.7, then 0.1, then 0.99. What are the losses?::−ln 0.7 = 0.3567, −ln 0.1 = 2.3026, −ln 0.99 = 0.0101. A model that spreads its chance equally over ten labels gives each 0.1, so 2.3026 = ln 10, the starting loss of an untrained 10-class model.
