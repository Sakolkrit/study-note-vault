---
tags: [data-augmentation, training, regularization, cnn, computer-vision, deep-learning, ML]
status: growing
level: 4
tier: 2
---

# Data Augmentation

**Data augmentation** means making new training examples by changing the ones you already have in ways that should not change their label: shift a picture a little, flip it, zoom, tilt, change its brightness, add a bit of noise. A cat that is two pixels to the left is still a cat. The model sees many more *versions* of each picture, so it learns what makes a cat a cat rather than where it happened to sit in the photo. It is one of the cheapest ways to fight [[Overfitting]] when labelled data are scarce.

## 🟢 1. Simple (age 12)
Imagine you are learning to recognise your friend's face, but you have only one photo of them, looking straight ahead. If your friend turns their head or stands further away, you might not know them. Now imagine you practise with that one photo moved a bit left, a bit right, a bit bigger, a bit smaller. You get much better at spotting your friend in any pose, with no new photo needed.

**Example:** One photo of the digit 7. You make eight more by sliding it one step in each direction. Now you have nine "different" pictures of the 7, and all are still a 7.

## 🔵 2. How it works
1. **Pick changes that keep the label.** For most photos, small shifts, flips, small turns, zooms and colour tweaks keep the label. For digits, a mirror flip would be a mistake (a flipped 2 looks like a 5), so the right changes depend on the task.
2. **Apply them randomly while training.** Each time the model sees a picture, you apply a fresh random change. In the next round it sees a slightly different version.
3. **The model cannot memorise exact pixels.** Since the exact pixels keep changing, the model has to learn the patterns that stay the same.
4. **Test on normal pictures.** You do not augment the test set (except to test robustness), since you want to know how it does on real, untouched pictures.
5. **It works well with a [[Convolutional Neural Network]].** A CNN is already partly shift-tolerant ([[Convolution]], [[Pooling]]), and augmentation teaches the rest.
6. **It costs almost nothing.** No new labelling is needed, only a little computing time each round.

![[data-augmentation-augmentor.png]]
*One way to make a new picture from an old one: an "elastic" distortion. The handwritten 8 on the left is covered with a grid, the arrows show how each grid point is pushed, and the result on the right is a wobbly but still clearly readable 8. Image: Marcus D. Bloice, Christof Stocker and Andreas Holzinger (from the paper about the Augmentor tool), [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:An_Example_of_Data_Augmentation_via_Augmentor.png), CC BY-SA 4.0.*

## 🟠 3. Technical detail
All numbers come from the code at the end of this section (Python 3.9.6, NumPy 1.24.4, scikit-learn 1.1.2, PyTorch 2.8.0 on a CPU). I used the 1,797 small 8×8 hand-written digits in scikit-learn. The training set was the first 100 (or 300) pictures of a fixed shuffle, and the test set was 797 digits kept apart. The model was the same small CNN as in [[Image Classification]], trained for 150 full-batch steps. For the augmented runs, *every step* gave each picture a fresh random shift of -1, 0 or +1 pixel in each direction (empty edges filled with zeros).

- **How many copies:** equation 1. For every one of the 1,797 pictures, the nine shifts of up to one pixel gave 9 different images, as the formula says.
- **Three test sets:** the normal digits, the digits all shifted one pixel to the right, and the digits each randomly shifted. Every number below is the mean over 5 training seeds.
- **100 training pictures:** without augmentation the accuracy was 0.8846 (normal), 0.5272 (shifted right) and 0.462 (random shifts). With random shifts in training it was 0.8642, 0.8688 and 0.8391.
- **300 training pictures:** without augmentation 0.9533, 0.6274 and 0.5174. With random shifts 0.9287, 0.923 and 0.9056.
- **Reading it:** augmentation bought a huge gain on shifted pictures (0.53 → 0.87 and 0.63 → 0.92 on the shifted-right set) and cost 2.0 and 2.5 points on the untouched ones (0.8846 → 0.8642 and 0.9533 → 0.9287). It is a trade, not a free lunch: the model's capacity is shared between the original and the shifted versions, and 150 steps may be too few to get both. I did not try longer training.
- **Augmentation against more data:** 100 augmented pictures reached 0.8688 on the shifted set, far above the 0.6274 of 300 plain ones. On the normal test set, the 300 plain pictures were better (0.9533 against 0.8642). So augmentation makes up for *the kind of variation you add*, not for everything that more real data would bring.
- **Limits:** one data set, tiny pictures, one kind of change (shifts of at most one pixel), one model, one training length, 5 seeds. I did not test flips, rotations, zooms, colour changes, noise, mixing methods, or how much the number of steps matters. The elastic distortion in the picture was not run.

The code covers two things: (1) how many shifted copies a picture has, (2) a small CNN trained with and without random shifts, tested on three kinds of picture.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
torch.set_num_threads(1)
D = load_digits(); X = (D.data / 16.0).reshape(-1, 1, 8, 8).astype(np.float32); y = D.target.astype(np.int64)
idx = np.random.RandomState(0).permutation(len(y)); te = idx[1000:]
def shift(a, dx, dy):                                                 # move a picture dx right and dy down; the empty border is filled with zeros
    p = np.pad(a, ((0, 0), (1, 1), (1, 1))); return p[:, 1 - dy:9 - dy, 1 - dx:9 - dx]
def cnn(): return nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(64, 10))
def randshift(a, g):                                                  # every picture gets its own random shift of -1, 0 or +1 in each direction
    return np.stack([shift(p[None], *g.randint(-1, 2, 2))[0] for p in a])[:, None]
def train(n, augment, seed):
    g = np.random.RandomState(seed); sub = idx[:n]; torch.manual_seed(seed); m = cnn(); opt = torch.optim.Adam(m.parameters(), 0.01)
    for _ in range(150):
        xb = randshift(X[sub, 0], g) if augment else X[sub]
        opt.zero_grad(); F.cross_entropy(m(torch.tensor(xb)), torch.tensor(y[sub])).backward(); opt.step()
    return m
acc = lambda m, a, b: float((m(torch.tensor(a)).argmax(1).numpy() == b).mean())
Xs = shift(X[te, 0], 1, 0)[:, None]; Xr = randshift(X[te, 0], np.random.RandomState(99))
# 1) how many different copies can shifting make? (2d + 1)^2 positions for shifts up to d pixels
from collections import Counter
counts = Counter(len({tuple(shift(X[i], dx, dy).ravel()) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}) for i in range(len(y)))
print('different copies per digit from shifts up to 1 pixel, over all', len(y), 'pictures:', dict(counts), '| (2d + 1)^2 with d = 1 =', (2 * 1 + 1) ** 2, '| 100 labelled pictures ->', 100 * 9, 'variants')
# 2) a small training set, with and without random shifts; 5 training seeds each; three kinds of test picture
for n in (100, 300):
    res = {a: np.array([[acc(m, X[te], y[te]), acc(m, Xs, y[te]), acc(m, Xr, y[te])] for m in (train(n, a, s) for s in range(5))]).mean(0).round(4).tolist() for a in (False, True)}
    print('trained on', n, 'pictures | accuracy [normal, shifted right 1, randomly shifted] | no augmentation:', res[False], '| with random shifts:', res[True])
```
Reading the output (Python 3.9.6, PyTorch 2.8.0):
- **Copies:** `{9: 1797}`, so all 1,797 pictures gave 9 different shifted versions. $(2d+1)^2=9$ and 100 pictures give 900 variants.
- **100 pictures:** `[0.8846, 0.5272, 0.462]` without and `[0.8642, 0.8688, 0.8391]` with random shifts.
- **300 pictures:** `[0.9533, 0.6274, 0.5174]` without and `[0.9287, 0.923, 0.9056]` with random shifts.
- **Limits:** the run takes about a minute and nothing in it is timed.

## 🔴 4. Deep / formal

**How many versions can shifting make?** If you may move a picture by up to $d$ pixels in each of the two directions, then the horizontal shift can take $2d+1$ values (from $-d$ to $+d$), and so can the vertical one. Because the two choices are independent, the number of shifted versions, counting "no shift", is:

$$m=(2d+1)^{2}$$

**Where:**
- $d$ is the largest shift in pixels, in each direction. **Matters:** a larger $d$ gives more variety, but also pushes more of the picture off the edge.
- $2d+1$ counts the choices in one direction: $d$ to the left, $d$ to the right, and "stay". **Matters:** forgetting the "stay" would leave out the original picture.
- The square. **Matters:** the horizontal and vertical choices combine as a grid, so the count is the product of the two.
- $m$ is the number of different shifted versions of one picture. **Matters:** it multiplies the effective size of the training set: $N$ labelled pictures give up to $N\cdot m$ variants. The variants are not independent data, since they are all copies of the same $N$ pictures, so they are worth less than $N\cdot m$ truly new pictures.

**Example (tiny numbers):** $d=1$ and $N=100$ labelled pictures.
1. $2d+1=3$ shifts per direction (left, none, right).
2. $m=3^2=9$ versions per picture.
3. $N\cdot m=900$ variants. The code found 9 different images for every one of the 1,797 digits, and printed 900 for 100 pictures.
4. Meaning: 100 real pictures behave like up to 900 slightly different ones. The code's measured gain was on shifted test pictures: 0.5272 → 0.8688 for the 100-picture model (mean of 5 seeds).

**What came before, and how this improved on it:**
- **Before:** the only sure way to make a model robust was to collect and label more real data (slow and costly), or to design features by hand that ignore the change (such as edge histograms that ignore small shifts). Early CNN work on digits already added distorted copies of the training digits (the LeNet work in the 1990s, from memory; I did not check the details).
- **Its limit:** labelling thousands of new photos costs money and time, and hand-made invariance has to be redesigned for each task.
- **What changed:** make the variants on the fly from the pictures you already have, with the label for free. Large image networks such as AlexNet (2012) used random crops and flips as a standard part of training (from memory; I did not check the paper's details). See [[LeNet and AlexNet]].
- **How much better:** in this note's run, accuracy on shifted test pictures went from 0.5272 to 0.8688 with 100 labelled pictures, and from 0.6274 to 0.923 with 300, for a cost of 2.0 to 2.5 points on untouched pictures. 100 augmented pictures beat 300 plain ones on the shifted test (0.8688 against 0.6274), though not on the normal test (0.8642 against 0.9533).

**Other points (from memory, not checked here):**
- Choosing changes that *do* change the label is a classic mistake (flipping digits or letters, or cropping away the object). Always look at some augmented examples.
- Besides pictures, text and sound have their own versions (swapping words, adding noise, changing speed). I did not test these.
- Some methods mix two pictures and their labels (such as "mixup"), and some generate new pictures with a [[GAN]] or [[Diffusion Models]]. I did not test these.
- Augmentation also acts as a form of [[Regularization]]. A rule of thumb is that it helps most when data are scarce (which this run suggests but one toy case does not prove).

## Related
[[Overfitting]] · [[Regularization]] · [[Convolutional Neural Network]] · [[Convolution]] · [[Pooling]] · [[Image Classification]] · [[Dropout]] · [[Transfer Learning]] · [[GAN]] · [[Diffusion Models]] · [[LeNet and AlexNet]] · [[Transfer Learning vs Vanilla CNN]]
- [[Image Segmentation]]
- [[Tiny Object Detector]]
## Sources
- scikit-learn's `load_digits` data (loaded by the code, version 1.1.2).
- Library versions in this run: Python 3.9.6, NumPy 1.24.4, scikit-learn 1.1.2, PyTorch 2.8.0 (CPU). The code and every printed value were run for this note, and nothing was timed.
- The historical remarks about LeNet and AlexNet using distorted or cropped copies are from memory; I did not open those papers for this note.
- Figure: [An_Example_of_Data_Augmentation_via_Augmentor.png](https://commons.wikimedia.org/wiki/File:An_Example_of_Data_Augmentation_via_Augmentor.png) (Bloice, Stocker and Holzinger, CC BY-SA 4.0), on Wikimedia Commons. I viewed it before writing the caption.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/data-features Data augmentation:::Make extra training pictures by changing the ones you have (shift, flip, zoom, colour tweaks), keeping the label. The model cannot memorise exact pixels and must learn the patterns that stay the same.

#flashcards/ml/formula #flashcards/topic/data-features Write the number of shifted versions of one picture and explain each symbol.
?
$$m=(2d+1)^{2}$$
- $d$: the largest shift in pixels, in each direction. A larger $d$ gives more variety but pushes more of the picture off the edge.
- $2d+1$: the choices in one direction: $d$ left, $d$ right, and "stay". Forgetting "stay" would leave out the original.
- The square: horizontal and vertical choices combine as a grid.
- $m$: the number of different shifted versions of one picture. $N$ labelled pictures give up to $N\cdot m$ variants. The variants are not independent.

#flashcards/ml/example #flashcards/topic/data-features $d=1$, $N=100$. What are $m$ and the number of variants?::$2d+1=3$ shifts per direction, $m=3^2=9$ per picture, $N\cdot m=900$ variants. The note's code measured on shifted test pictures a gain from 0.5272 to 0.8688 for the 100-picture model (mean of 5 seeds).

#flashcards/ml/image #flashcards/topic/data-features ![[data-augmentation-augmentor.png]]
What does this picture show?
?
One way to make a new picture from an old one: an elastic distortion. A handwritten 8 on the left is covered with a grid, the arrows show how each grid point is pushed, and the result on the right is a wobbly but still clearly readable 8. (From the note's caption: Marcus D. Bloice, Christof Stocker and Andreas Holzinger, Wikimedia Commons.)

#flashcards/ml/concept #flashcards/topic/data-features Why is a mirror flip a bad augmentation for digits?::A flipped 2 looks like a 5, so the label would change. Pick changes that keep the label.

#flashcards/ml/concept #flashcards/topic/data-features Should the test set be augmented?::No (except to test robustness). You want to know how the model does on real, untouched pictures.
