---
tags: [feature-map, cnn, convolution, feature-extraction, deep-learning, computer-vision, ML]
status: growing
level: 4
tier: 2
---

# Feature Map

A **feature map** is the grid of numbers that one filter produces when it slides over a picture (see [[Convolution]]). Each number says "how strongly is my pattern here?". A convolution layer has many filters, so it produces a *stack* of feature maps, one per filter (these are also called *channels*). The stack is what the next layer receives as its input. Early layers make maps for simple things such as edges, and deeper layers make maps for larger shapes. The maps are the "features" that the last layers of a [[Convolutional Neural Network]] use to decide what is in the picture.

## 🟢 1. Simple (age 12)
Imagine you have a stack of see-through sheets. On the first sheet you colour every place where the photo has an up-and-down edge. On the second sheet you colour every place with a side-to-side edge. On the third, every round curve. Each sheet is a map of one kind of thing. Looking at the stack together tells you far more than the photo's raw dots.

**Example:** For a photo of a ladder, the "up-and-down edge" sheet lights up along the two long sides, and the "side-to-side edge" sheet lights up along the rungs.

## 🔵 2. How it works
1. **One filter makes one map.** The filter slides over the picture, and at every position one number is written down. Those numbers form a smaller picture, the feature map.
2. **Many filters make a stack.** If a layer has 8 filters, it produces 8 maps. The stack has a *depth* of 8 (8 channels).
3. **The maps feed the next layer.** The next layer's filters look at all the maps at once, so they can combine "an up-and-down edge here" with "a curve there" into something bigger.
4. **[[Pooling]] makes the maps smaller**, and an [[Activation Function]] (usually ReLU) clears negative numbers, so only the strong "yes, it's here" signals remain.
5. **The filters are learned.** Nobody tells the network to look for edges. Training ([[Backpropagation]]) adjusts the filter numbers, and the maps that come out are whatever helps most to reduce the [[Loss Function]].
6. **At the end the maps are flattened** into one long list that goes to ordinary layers which pick the label ([[Image Classification]]).

![[lenet-5-architecture.svg]]
*A small CNN for a 28×28 picture of a handwritten digit, drawn as stacks of maps. After the first convolution there are 6 feature maps of 28×28 ("6@28x28, C1 feature map"), pooling makes them 6@14×14, the second convolution makes 16@10×10, and pooling makes 16@5×5. The pale-blue squares show one number in a map and the small window of the previous maps that it looks at. This is the "LeNet-5" design, here with a 28×28 input (the original paper used 32×32, see [[LeNet and AlexNet]]). Image: Aston Zhang, Zachary C. Lipton, Mu Li and Alexander J. Smola ("Dive into Deep Learning"), [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:LeNet-5_architecture.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
All numbers come from the code at the end of this section. It uses the small 8×8 hand-written digits in scikit-learn (1,000 pictures for training, 797 for testing; Python 3.9.6, scikit-learn 1.1.2, PyTorch 2.8.0 on a CPU).

- **A hand-made feature map:** the Sobel filters (3×3 grids `[[-1,0,1],[-2,0,2],[-1,0,1]]` and its transpose) measure brightness change left-to-right ($g_x$) and up-down ($g_y$). Slid over an 8×8 digit with no padding they give 6×6 maps. At position (2, 2) of the first training digit $g_x=2.0$ and $g_y=-0.125$, so the magnitude (equation 1) is 2.0039.
- **How good are these features?** I gave three kinds of features to the same simple classifier (logistic regression): the raw 64 pixels scored 0.9661 on the held-out digits, the 36 Sobel magnitudes scored 0.9184, and raw plus Sobel together (100 numbers) scored 0.9561. So on these tiny digits the hand-made edge map *lost* information, since it throws away the direction and the sign of each change, and its maps are smaller than the picture, and adding it to the raw pixels did not help (it went from 0.9661 down to 0.9561, a small drop that I did not test further).
- **Learned feature maps:** I trained a small CNN (two 3×3 convolutions with 8 and 16 filters, two max-pools, one linear layer) for 150 steps, took its last 64 numbers (16 maps of 2×2) and gave them to the same logistic regression. It scored 0.9824, higher than the raw pixels (0.9661) and the Sobel features (0.9184). The CNN's own final layer scored 0.9762. All of these used the same single split and one training seed, so the gaps of about one point are not proof.
- **Sizes:** the input is 64 numbers. After the first convolution there are 8 maps of 8×8 = 512 numbers, after the pooling 8×4×4 = 128, after the second convolution 16×4×4 = 256 and after the last pooling 16×2×2 = 64. So the maps are *bigger* than the input at first (equation 2), which is why memory for the maps (not the filters) is what limits deep CNNs. I did not measure memory.
- **Limits:** one data set, one split, one seed for the CNN, tiny 8×8 pictures, and only the Sobel filters as the hand-made example. I did not look at what the learned filters detect, and I did not test colour pictures, bigger networks or other hand-made features.

The code covers four things: (1) a Sobel feature map and its magnitude, (2) three kinds of features given to the same classifier, (3) the CNN's learned maps, (4) the shape of the maps through the network.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
torch.set_num_threads(1)
D = load_digits(); X = (D.data / 16.0).reshape(-1, 1, 8, 8); y = D.target.astype(np.int64)
idx = np.random.RandomState(0).permutation(len(y)); tr, te = idx[:1000], idx[1000:]
T = lambda a: torch.tensor(a, dtype=torch.float32)
# 1) a hand-made feature map: the Sobel filters measure brightness change left-right (gx) and up-down (gy)
SX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float); SY = SX.T
def sobel(a):                                                         # a: (n, 1, 8, 8) -> gx, gy, each (n, 1, 6, 6)
    return [F.conv2d(T(a), T(k)[None, None]).numpy() for k in (SX, SY)]
gx, gy = sobel(X[tr][:1])
print('one digit, 8x8 -> a Sobel map of size', gx.shape[2:], '| at position (2, 2): gx =', round(float(gx[0, 0, 2, 2]), 4), ', gy =', round(float(gy[0, 0, 2, 2]), 4), ', magnitude =', round(float(np.hypot(gx[0, 0, 2, 2], gy[0, 0, 2, 2])), 4))
print('magnitude formula on a textbook pair: gx = 3, gy = 4 ->', np.hypot(3, 4))
# 2) how good are different feature maps for one simple classifier (logistic regression)?
def feats_sobel(a): gx, gy = sobel(a); return np.sqrt(gx ** 2 + gy ** 2).reshape(len(a), -1)
def score(f): return round(LogisticRegression(max_iter=3000).fit(f(X[tr]), y[tr]).score(f(X[te]), y[te]), 4)
raw = lambda a: a.reshape(len(a), -1)
print('logistic regression on | raw pixels (64 numbers):', score(raw), '| Sobel magnitude map (36 numbers):', score(feats_sobel), '| raw + Sobel (100 numbers):', score(lambda a: np.hstack([raw(a), feats_sobel(a)])))
# 3) learned feature maps: train a small CNN, then give its last feature maps to the same simple classifier
torch.manual_seed(0)
net = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(64, 10))
opt = torch.optim.Adam(net.parameters(), 0.01)
for _ in range(150): opt.zero_grad(); F.cross_entropy(net(T(X[tr])), torch.tensor(y[tr])).backward(); opt.step()
body = net[:-1]
learned = lambda a: body(T(a)).detach().numpy()
print('logistic regression on the CNN\'s own last feature maps (', learned(X[:1]).shape[1], 'numbers):', score(learned), '| the CNN itself:', round(float((net(T(X[te])).argmax(1).numpy() == y[te]).mean()), 4))
# 4) shapes and sizes of the feature maps as the picture flows through the network
a = T(X[:1]); print('input', tuple(a.shape), '=', a.numel(), 'numbers')
for m in net:
    a = m(a)
    if isinstance(m, (nn.Conv2d, nn.MaxPool2d)): print('after', type(m).__name__, tuple(a.shape)[1:], '=', a.numel(), 'numbers')
print('formula channels x size x size for the first convolution: 8 *', 8, '*', 8, '=', 8 * 8 * 8)
```
Reading the output (Python 3.9.6, scikit-learn 1.1.2, PyTorch 2.8.0):
- **Sobel:** the map is 6×6, with $g_x=2.0$, $g_y=-0.125$ and magnitude 2.0039 at position (2, 2). The textbook pair 3 and 4 gives 5.0.
- **Features:** raw pixels 0.9661, Sobel magnitudes 0.9184, raw plus Sobel 0.9561.
- **Learned:** the CNN's 64 last-map numbers 0.9824 with the same classifier, and the CNN itself 0.9762.
- **Shapes:** 64 → 512 → 128 → 256 → 64 numbers, and $8\cdot8\cdot8=512$.
- **Limits:** the run takes a few seconds and nothing is timed.

## 🔴 4. Deep / formal

**The classic hand-made feature.** The Sobel filters give two feature maps, the change in brightness along the columns ($g_x$) and along the rows ($g_y$). Their combined strength at one position is:

$$G=\sqrt{g_x^{2}+g_y^{2}}$$

**Where:**
- $g_x$ is the filter response for left-to-right change. **Matters:** it is large where the picture changes brightness sideways, such as at a vertical edge.
- $g_y$ is the response for up-down change. **Matters:** it is large at horizontal edges. With only $g_x$ you would miss them.
- Squaring both and adding. **Matters:** it makes the result independent of the signs (dark-to-light or light-to-dark both count as an edge).
- The square root. **Matters:** it turns the sum back into the same units as $g_x$ and $g_y$, the length of the little arrow $(g_x,g_y)$, like the diagonal of a rectangle.
- $G$ is the edge strength at this position. **Matters:** the map of $G$ values is a picture of where the edges are. The side effect is that the *direction* is lost.

**Example (tiny numbers):** $g_x=3$ and $g_y=4$.
1. $g_x^2=9$ and $g_y^2=16$, so the sum is 25.
2. $G=\sqrt{25}=5$.
3. The code printed 5.0 for this pair. For the first training digit at position (2, 2) it computed $\sqrt{2.0^2+0.125^2}=2.0039$.
4. Meaning: a strong sideways change with a little up-down change is a nearly vertical edge of strength about 2.

**How much memory do the maps need?** A layer with $C$ filters and an output of size $o\times o$ per map produces:

$$N=C\,o^{2}$$

**Where:**
- $C$ is the number of filters, the depth of the stack. **Matters:** more filters means more kinds of pattern, but a proportionally bigger stack.
- $o$ is the side of each map, from the output-size rule $\lfloor(i-k+2p)/s\rfloor+1$ in [[Convolution]]. **Matters:** the maps are squares of this side, so the cost grows with $o^2$.
- $N$ is the number of numbers that must be stored for this layer's maps (for each picture). **Matters:** during training all of these must be kept for [[Backpropagation]], so $N$ summed over layers is the memory bill.

**Example (tiny numbers):** the first convolution of the code's CNN: 8 filters, padding 1, an 8×8 input.
1. $o=\lfloor(8-3+2)/1\rfloor+1=8$.
2. $N=8\cdot8^2=8\cdot64=512$.
3. The code printed 512 numbers after the first convolution (the input has 64).
4. Meaning: eight times more numbers than the input, then the pooling cuts it to 128. So the early maps cost the most memory.

**What came before, and how this improved on it:**
- **Before:** hand-designed feature maps: edge filters such as Sobel (1960s, from memory) and later hand-made summaries such as histograms of gradient directions (HOG, 2005, from memory; I did not check the dates), fed to a separate classifier.
- **Its limit:** an expert has to decide which features matter for each task, and a fixed filter cannot adapt. In this note's run, the edge-strength map lost direction information and scored worse than the raw pixels.
- **What changed:** make the filters learnable numbers, trained together with the classifier on the final loss.
- **How much better:** with the same logistic-regression classifier, the learned feature maps scored 0.9824, raw pixels 0.9661 and Sobel magnitudes 0.9184 (one split, one seed; the gap to the raw pixels is about 1.6 points). On large photos the published gain is larger (see [[LeNet and AlexNet]]).

**Other points (from memory, not checked here):**
- Looking at what each map responds to (by finding the picture that excites it most) often shows edges and colours in the first layer, then textures, then parts of objects in deeper layers. I did not test this here.
- The name "feature map" is sometimes used for the output of any layer, also outside CNNs. I only covered the convolution case.
- Maps that are always zero after ReLU are called "dead" and can be removed. I did not test this.

## Related
[[Convolution]] · [[Convolutional Neural Network]] · [[Pooling]] · [[Receptive Field]] · [[Activation Function]] · [[Backpropagation]] · [[Image Classification]] · [[Logistic Regression]] · [[LeNet and AlexNet]] · [[PyTorch]]

## Sources
- scikit-learn's `load_digits` data (loaded by the code, version 1.1.2).
- Library versions in this run: Python 3.9.6, scikit-learn 1.1.2, PyTorch 2.8.0 (CPU). The code and every printed value were run for this note, and nothing was timed.
- Sobel filters and HOG: named from memory (the Sobel operator from the late 1960s, HOG from 2005); I did not open the original sources.
- Figure: [LeNet-5_architecture.svg](https://commons.wikimedia.org/wiki/File:LeNet-5_architecture.svg) (Zhang, Lipton, Li and Smola, CC BY-SA 4.0), on Wikimedia Commons. I viewed it before writing the caption.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision Feature map:::A see-through sheet that marks every place in the picture where one kind of pattern appears (an up-down edge, a curve). A stack of such sheets describes the picture far better than one.

#flashcards/ml/formula #flashcards/topic/vision Write the edge strength from two filter responses and explain each symbol.
?
$$G=\sqrt{g_x^{2}+g_y^{2}}$$
- $g_x$: the filter response for left-to-right change. It is large where brightness changes sideways, such as at a vertical edge.
- $g_y$: the response for up-down change. It is large at horizontal edges. With only $g_x$ you would miss them.
- Squaring both and adding makes the result independent of the sign and direction. The square root gives the overall edge strength $G$.

#flashcards/ml/formula #flashcards/topic/vision Write the number of values in a stack of feature maps and explain each symbol.
?
$$N=C\,o^{2}$$
- $C$: the number of filters, the depth of the stack. More filters means more kinds of pattern but a bigger stack.
- $o$: the side of each map, from the output-size rule $\lfloor(i-k+2p)/s\rfloor+1$. The cost grows with $o^2$.
- $N$: the total number of numbers in the stack.

#flashcards/ml/example #flashcards/topic/vision If $g_x=3$ and $g_y=4$, what is the edge strength?::g_x² = 9 and g_y² = 16, sum 25, G = √25 = 5. A strong sideways change with a little up-down change is a nearly vertical edge of strong size.

#flashcards/ml/example #flashcards/topic/vision The first convolution has 8 filters, padding 1 and an 8×8 input. How many numbers are in the feature maps?::o = ⌊(8 − 3 + 2)/1⌋ + 1 = 8. N = 8 · 8² = 512 numbers (the input has 64). Eight times more, then pooling cuts it to 128.

#flashcards/ml/image #flashcards/topic/vision ![[lenet-5-architecture.svg]]
What does this picture show?
?
A small CNN for a 28×28 handwritten digit, drawn as stacks of maps. After the first convolution there are 6 feature maps of 28×28 ("6@28x28"), pooling makes them 6@14×14, the second convolution makes 16@10×10, and pooling makes 16@5×5. This is the LeNet-5 design, here with a 28×28 input.
