---
tags: [convolution, cnn, deep-learning, kernel, stride, padding, computer-vision, ML]
status: growing
level: 4
tier: 2
---

# Convolution

In a [[Convolutional Neural Network]], a **convolution** is the step where a small grid of numbers, the **kernel** (or *filter*), slides over a picture. At every position it multiplies its numbers with the pixels underneath and adds them up, giving one new number. The new numbers form a smaller picture, the **feature map**, which lights up wherever the kernel's pattern appears (an edge, a corner, a texture). The kernel's numbers are what the network [[Backpropagation|learns]]. This note is about the sliding itself: kernel, **stride**, **padding**, and how big the result is.

## 🟢 1. Simple (age 12)
Imagine a small square window cut out of cardboard. You lay it on a photo, look at only the nine little squares you can see, and write down one number that says "how much does this look like a left edge?". Then you slide the window one step and do it again. When you have covered the whole photo, your numbers make a new, smaller picture that shows where the edges are.

**Example:** On a photo of a black door in a white wall, the window gives big numbers right at the door's edge and zeros on the flat wall and the flat door.

## 🔵 2. How it works
1. **Choose a kernel.** It is a small grid, usually 3×3, of numbers. For example `[[1, 0, -1], [1, 0, -1], [1, 0, -1]]` means "left side minus right side", which reacts to a vertical edge.
2. **Place it on the top-left corner** of the picture. Multiply each kernel number with the pixel under it and add the nine products. That one sum is the first output number.
3. **Slide it.** Move the kernel one step right and repeat. At the end of the row, go back to the left and one step down. The number of pixels you move each time is the **stride** (1 is the usual start, 2 skips every other position).
4. **Padding.** Without any help the output is smaller than the picture, because the kernel cannot hang over the edge. To avoid that (or to let edge pixels count as much as middle ones), add a border of zeros around the picture first. That border is the **padding**.
5. **Same kernel everywhere.** The same nine numbers are used at every position. So the network needs very few numbers, and a pattern is spotted wherever it sits in the picture.
6. **Many kernels, many maps.** A layer uses many kernels and stacks their maps, so one layer can look for edges of many angles and colours at once. A later layer then combines these into bigger shapes (see [[Receptive Field]]).
7. **Then** the maps go through an [[Activation Function]] and often a [[Pooling]] step.

![[convolution-no-padding-strides.gif]]
*A 3×3 kernel (the dark square) slides over a 5×5 input (blue) with stride 2 and no padding. Each position gives one cell of the 2×2 output (cyan). I looked at all 4 frames of the animation. Image: Vincent Dumoulin and Francesco Visin, from "A guide to convolution arithmetic for deep learning", [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Convolution_arithmetic_-_No_padding_strides.gif), MIT licence.*

![[convolution-padding-strides.gif]]
*The same 5×5 input, with a border of zeros added (dashed cells), a 3×3 kernel and stride 2. Now there are 3×3 positions, so the output is 3×3. I looked at 4 of the 9 frames. Image: Vincent Dumoulin and Francesco Visin, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Convolution_arithmetic_-_Padding_strides.gif), MIT licence.*

## 🟠 3. Technical detail
All numbers below come from the code at the end of this section (Python 3.9.6, NumPy 1.24.4, SciPy 1.8.1, PyTorch 2.8.0 on a CPU). I wrote my own sliding loops and compared them with PyTorch's `conv2d`.

- **The sliding, by hand:** a 3×3 filter `[[1,0,-1],[1,0,-1],[1,0,-1]]` on a 5×5 picture gave the 3×3 output `[[0, 0, -3], [-1, -4, 1], [3, -3, 1]]`, and my loops and `conv2d` agreed.
- **It is not really "convolution":** in maths, convolution flips the kernel before sliding it. Deep-learning libraries do *not* flip it (strictly the operation is called cross-correlation, which is what `conv2d` computes). The code shows `conv2d` equals my no-flip loops, differs from SciPy's textbook `convolve2d` with the same kernel, and equals it once the kernel is flipped both ways. This hardly matters in practice, because the network learns the kernel anyway, so it just learns the flipped version.
- **Output size:** equation 1 below. The code checked it against PyTorch on 908 settings (picture sizes 1 to 12, kernels 1 to 5, padding 0 to 3, stride 1 to 4) and it matched in all 908. The two pictures above give 2 (no padding) and 3 (padding 1).
- **The floor drops the leftovers:** with width 6, kernel 3 and stride 2 the output has 2 positions and the last pixel column is never read. The code changed that column and the output did not move.
- **"Same" padding:** with stride 1 and padding $p=(k-1)/2$ (for odd $k$), the output is as big as the input. At width 8 the output stayed 8 for kernels 1, 3, 5 and 7. Without padding, a 3×3 kernel shrank the picture from 8 to 6.
- **An edge detector:** on a picture that is dark (0) on the left half and bright (9) on the right half, the vertical-edge filter gave `[0, -27, -27, 0]` along a row: zero on flat parts and a strong response only at the edge. On a completely flat picture it gave `[0, 0, 0, 0]`. The sign is negative here because the filter is "left minus right" and the right side is brighter.
- **Shift in, shift out:** I moved a small blob one column to the right. The output did not stay the same, it moved one column too (`True` for shifted, `False` for unchanged). This is called *equivariance*. (Pooling and the final layers are, from memory, what make a CNN partly shift-*invariant*; this code did not test that.)
- **How many numbers:** one 3×3 filter on one channel holds $9+1=10$ numbers (nine weights and a bias). A layer from 3 channels to 8 channels holds $3\cdot3\cdot3\cdot8+8=224$. The code counted both with PyTorch. The count does not depend on the picture size.
- **Dense against convolution:** a plain dense layer that makes an 8×8 map from an 8×8 picture needs $(64+1)\cdot64=4160$ numbers, and for a 224×224 picture $(50176+1)\cdot50176=2{,}517{,}681{,}152$, while the 3×3 filter holds 10 in both cases (see the block *What came before* below).
- **Limits:** I tested only 2-D, square kernels and square pictures, with PyTorch's defaults (no dilation, no groups). I did not test colour channels in the sliding loops, batches, dilation (spaced-out kernels), transposed convolution, 1-D or 3-D convolution, or speed.

The code covers eleven things: (1) the sliding, (2) no flipping, (3) the size formula, (4) the floor effect, (5) "same" padding, (6) an edge detector, (7) shifting, (8) the slide-style formula against the right one, (9) counting numbers, (10) the sum written out, (11) the cost of a plain dense layer for the same job.

```python
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import convolve2d
t = lambda a: torch.tensor(np.asarray(a, dtype=float), dtype=torch.float64)[None, None]        # 1 image, 1 channel
def conv(x, K, s=1, p=0): return F.conv2d(t(x), t(K), stride=s, padding=p)[0, 0].numpy()        # the library's answer
def slide(x, K, s=1, p=0):                                                                      # our own loops: no flipping, zero padding
    x, K = np.pad(np.asarray(x, float), p), np.asarray(K, float); k = K.shape[0]
    o = (x.shape[0] - k) // s + 1
    return np.array([[(x[u * s:u * s + k, v * s:v * s + k] * K).sum() for v in range(o)] for u in range(o)])
# 1) slide a 3x3 filter over a 5x5 picture; the library and our loops agree
x = [[1, 2, 0, 1, 3], [0, 1, 3, 1, 2], [2, 1, 0, 2, 1], [1, 0, 1, 3, 0], [3, 1, 2, 0, 1]]
K = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
print('output:', slide(x, K).astype(int).tolist(), '| same as conv2d:', np.allclose(slide(x, K), conv(x, K)))
# 2) deep-learning "convolution" does not flip the filter (strictly it is cross-correlation)
print('conv2d equals the no-flip loops:', np.allclose(conv(x, K), slide(x, K)), '| equals textbook convolution with the same K:', np.allclose(conv(x, K), convolve2d(x, K, 'valid')), '| equals textbook convolution with K flipped both ways:', np.allclose(conv(x, K), convolve2d(x, np.flip(K), 'valid')))
# 3) output size: o = floor((i - k + 2p) / s) + 1, checked against the library on every combination
n = ok = 0
for i, k, p, s in itertools.product(range(1, 13), range(1, 6), range(0, 4), range(1, 5)):
    if i + 2 * p < k: continue
    n += 1; ok += conv(np.ones((i, i)), np.ones((k, k)), s, p).shape == ((i - k + 2 * p) // s + 1,) * 2
print('formula matches the library on', ok, 'of', n, 'settings | pictures: i=5,k=3,p=0,s=2 ->', conv(np.ones((5, 5)), np.ones((3, 3)), 2, 0).shape[0], '| i=5,k=3,p=1,s=2 ->', conv(np.ones((5, 5)), np.ones((3, 3)), 2, 1).shape[0])
# 4) with stride 2 on width 6 the last column is never read: the floor drops it
a = np.arange(36.).reshape(6, 6); b = a.copy(); b[:, 5] += 100
print('i=6, k=3, s=2 -> size', conv(a, np.ones((3, 3)), 2).shape[0], '| changing the last column changes the output:', not np.allclose(conv(a, np.ones((3, 3)), 2), conv(b, np.ones((3, 3)), 2)))
# 5) "same" padding: stride 1 and p = (k - 1) / 2 keeps the size
print('same padding, i = 8:', {k: conv(np.ones((8, 8)), np.ones((k, k)), 1, (k - 1) // 2).shape[0] for k in (1, 3, 5, 7)}, '| no padding, k = 3:', conv(np.ones((8, 8)), np.ones((3, 3))).shape[0])
# 6) an edge detector: dark left half, bright right half
E = np.array([[0] * 3 + [9] * 3] * 6)
print('vertical-edge filter on a dark|bright picture:', conv(E, K).astype(int).tolist()[0], '| on a flat picture:', conv(np.full((6, 6), 9), K).astype(int).tolist()[0])
# 7) shift the thing in the picture and the response shifts with it (equivariance)
g = np.zeros((10, 10)); g[3:6, 3:6] = np.random.RandomState(0).randint(1, 9, (3, 3)); gs = np.roll(g, 1, axis=1)
print('shifted input -> output shifted by one column:', np.allclose(conv(gs, K)[:, 1:], conv(g, K)[:, :-1]), '| output itself unchanged:', np.allclose(conv(gs, K), conv(g, K)))
# 8) the formula as I read it on the slide, taken literally, against the real one (i=7, k=3, s=2)
print('i - k/s + 1 =', 7 - 3 / 2 + 1, '| floor((i - k)/s) + 1 =', (7 - 3) // 2 + 1, '| library:', conv(np.ones((7, 7)), np.ones((3, 3)), 2).shape[0])
# 9) how many numbers the filter holds
for cin, cout in ((1, 1), (3, 8)):
    c = torch.nn.Conv2d(cin, cout, 3); print('Conv2d(', cin, ',', cout, ', 3) parameters:', sum(q.numel() for q in c.parameters()), '= 3*3*', cin, '*', cout, '+', cout, '=', 9 * cin * cout + cout)
# 10) the sum written out, on a tiny ramp picture
r = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]; D = [[1, 0], [0, -1]]
print('ramp with diagonal filter:', slide(r, D).astype(int).tolist(), '| conv2d:', conv(r, D).astype(int).tolist())
# 11) what the same job costs as a plain dense layer (every pixel wired to every output number): 1 channel in, 1 map out, same size
dense = lambda i: (i * i + 1) * i * i
conv_n = sum(q.numel() for q in torch.nn.Conv2d(1, 1, 3).parameters())
print('i = 8: convolution', conv_n, 'numbers against dense', dense(8), '(PyTorch Linear:', sum(q.numel() for q in torch.nn.Linear(64, 64).parameters()), ') | i = 224: convolution', conv_n, 'against dense', dense(224))
```
Reading the output (Python 3.9.6, PyTorch 2.8.0):
- **Sliding:** the output is `[[0, 0, -3], [-1, -4, 1], [3, -3, 1]]` and equals `conv2d` (`True`).
- **Flipping:** equals the no-flip loops (`True`), differs from textbook convolution with the same kernel (`False`) and equals it with the kernel flipped (`True`).
- **Size:** 908 of 908 settings agree. The pictured cases give 2 and 3.
- **Floor:** size 2, and changing the last column changes nothing (`False`).
- **Same padding:** `{1: 8, 3: 8, 5: 8, 7: 8}`, and 6 with no padding.
- **Edge:** `[0, -27, -27, 0]` on the edge picture and `[0, 0, 0, 0]` on the flat one.
- **Shift:** shifted output equals the original output moved one column (`True`) and is not the same as the original (`False`).
- **Formula as I read it on the slide:** literally `i - k/s + 1` gives 6.5 for $i=7$, $k=3$, $s=2$, while $\lfloor (i-k)/s\rfloor+1$ gives 3, and the library gives 3.
- **Numbers:** 10 and 224.
- **Ramp:** `[[-4, -4], [-4, -4]]` from both my loops and `conv2d`.
- **Dense against convolution:** at width 8 the convolution holds 10 numbers against 4,160 for a dense layer (PyTorch's `Linear` agrees), and at width 224 it still holds 10 against 2,517,681,152.
- **Limits:** the code runs in a blink and times nothing.

## 🔴 4. Deep / formal

**What is one output number?** For one input channel, a $k\times k$ kernel $K$, a bias $b$, stride $s$ and padding $p$ (the picture counted as zero outside its edges), the output at row $u$ and column $v$ is:

$$y_{u,v}=b+\sum_{a=0}^{k-1}\sum_{c=0}^{k-1}K_{a,c}\;x_{us+a-p,\;vs+c-p}$$

**Where:**
- $x$ is the input picture, and $x_{r,q}$ is the pixel in row $r$ and column $q$ (zero outside the picture). **Matters:** this is what the kernel looks at. The zeros are what padding supplies.
- $K_{a,c}$ is the kernel number in row $a$ and column $c$. **Matters:** these are the learned numbers. A large positive number means "more of this pixel makes the output bigger", a negative one means the opposite.
- $b$ is a bias, one number added to every output. **Matters:** it shifts all responses up or down, so the next [[Activation Function]] fires at a different level.
- $s$ is the stride. **Matters:** with $s=2$ the window start moves two pixels per output step, so the output has about half as many rows and columns.
- $p$ is the padding. **Matters:** it moves the window start back by $p$ pixels, so the kernel can cover border pixels from outside the picture.
- $u,v$ are the output row and column. **Matters:** they tell you where in the picture the response sits.
- The double sum over $a,c$ is the "multiply and add" step. **Matters:** nothing else happens, so the whole layer is a pile of dot products (this is why it can be done as one big matrix multiplication on a [[GPU]]).

**Example (tiny numbers):** picture $x=\begin{pmatrix}1&2&3\\4&5&6\\7&8&9\end{pmatrix}$, kernel $K=\begin{pmatrix}1&0\\0&-1\end{pmatrix}$, $b=0$, $s=1$, $p=0$.
1. The output has $\lfloor(3-2)/1\rfloor+1=2$ rows and columns.
2. $y_{0,0}=1\cdot1+0\cdot2+0\cdot4+(-1)\cdot5=-4$.
3. $y_{0,1}=1\cdot2+(-1)\cdot6=-4$, $y_{1,0}=1\cdot4+(-1)\cdot8=-4$ and $y_{1,1}=1\cdot5+(-1)\cdot9=-4$.
4. The code gave `[[-4, -4], [-4, -4]]` with my loops and with `conv2d`.
5. Meaning: this kernel measures the difference "top-left pixel minus bottom-right pixel". The picture is a smooth ramp that rises 4 along every such diagonal step, so every window gives the same $-4$. A kernel gives the same answer where the picture looks the same.

**How big is the output?** Along one side, a kernel of size $k$ needs room for $k$ pixels. After adding $p$ pixels of padding to each end the usable width is $i+2p$. The kernel's first position needs $k$ pixels, and each further position needs $s$ more, so the number of positions is:

$$o=\left\lfloor\frac{i-k+2p}{s}\right\rfloor+1$$

**Where:**
- $i$ is the input width (or height). **Matters:** a bigger picture gives a bigger output, linearly.
- $k$ is the kernel size. **Matters:** a bigger kernel needs more room, so the output shrinks by $k-1$ (with stride 1, no padding).
- $p$ is the padding on each side, which is why it is counted twice ($2p$). **Matters:** each pixel of padding on each side adds back one position per side, so $p=(k-1)/2$ cancels a kernel's shrinkage when $s=1$.
- $s$ is the stride. **Matters:** it divides the count, so stride 2 roughly halves the size.
- $\lfloor\cdot\rfloor$ ("floor") rounds down to a whole number. **Matters:** positions are whole steps, and if the width does not fit exactly the last few pixels are never read (the code showed this at width 6).
- $+1$ counts the very first position. **Matters:** without it you would be counting the steps between positions, not the positions.

**Example (tiny numbers):** the two pictures in the note, both with $i=5$, $k=3$, $s=2$.
1. No padding ($p=0$): $\lfloor(5-3+0)/2\rfloor+1=\lfloor1\rfloor+1=2$.
2. Padding 1: $\lfloor(5-3+2)/2\rfloor+1=\lfloor2\rfloor+1=3$.
3. The code gave 2 and 3, and it agreed with PyTorch on all 908 settings it tried.
4. Meaning: padding the border with zeros turned a 2×2 output into a 3×3 one, and the pictures show exactly that many positions. **A note about my course slide:** it writes the formula as `[i-k/s]+1` and `[i-k+2p/s]+1`. Read literally, `i-k/s+1` gives 6.5 for $i=7$, $k=3$, $s=2$, where the correct value is 3 (the code printed both). The slide probably means $(i-k)/s$ and $(i-k+2p)/s$ inside the brackets, rounded down, which is the formula above. That reading is my guess, so check it with whoever wrote the slide.

**Other points (from memory, not checked here):**
- A convolution layer is a *linear* operation, and a Toeplitz-like matrix with shared entries could write it as one matrix product. That is why it is cheap in parameters but still expensive in arithmetic. I did not build that matrix here.
- Kernels need not be 3×3. Stacking two 3×3 layers covers the same area as one 5×5 layer with fewer numbers, and the popular VGG networks used this idea (from memory, I did not check). See [[Receptive Field]].
- A 1×1 kernel mixes channels at each pixel and looks at no neighbours.
- *Dilated* convolutions skip pixels inside the kernel to see more area, and *transposed* convolutions grow the output instead of shrinking it (used in [[Autoencoder]] decoders and [[GAN]] generators). I did not test either.
- Convolutions are also used on sound and other signals in one dimension, and on video in three. I tested only the 2-D case.

**What came before, and how this improved on it:**
- **Before:** a plain fully connected layer, where every output number has its own weight for every input pixel, and hand-made fixed filters (such as the Sobel edge filters, see [[Feature Map]]). A Bell Labs team applied [[Backpropagation]] to a network with convolution-style shared weights to read zip codes in 1989 (Wikipedia's LeNet article, which I read). Earlier shared-weight designs such as Fukushima's Neocognitron (around 1980) are from memory and I did not check them.
- **Its limit:** the weights grow with the picture. A dense layer for an 8×8 picture holds 4,160 numbers and for a 224×224 picture 2,517,681,152 (the code's count), and it has to learn "an edge here" again for every place.
- **What changed:** one small filter whose numbers are shared by every position, learned by [[Backpropagation]].
- **How much better:** the filter holds 10 numbers at every picture size, against 4,160 (416 times more) at width 8 and 2,517,681,152 (about 252 million times more) at width 224. That is the count of numbers only: I did not compare accuracy. See [[LeNet and AlexNet]] and [[Image Classification]] for results on real nets.

## Related
[[Convolutional Neural Network]] · [[Pooling]] · [[Receptive Field]] · [[Neural Network]] · [[Activation Function]] · [[Backpropagation]] · [[GPU]] · [[PyTorch]] · [[Matrix]] · [[Autoencoder]] · [[GAN]] · [[Feature Map]] · [[Image Classification]] · [[ResNet]] · [[LeNet and AlexNet]] · [[MobileNet]]
## Sources
- Vincent Dumoulin and Francesco Visin, *A guide to convolution arithmetic for deep learning* (arXiv:1603.07285), named in the Commons file page for the animations (I did not open the paper for this note).
- The owner's course slides "Convolutional.pdf" (kernel, stride and padding pages), read as text for the topics and the formulas. I did not copy any slide image.
- Library versions in this run: Python 3.9.6, NumPy 1.24.4, SciPy 1.8.1, PyTorch 2.8.0 (CPU). The code and every printed value were run for this note. Nothing was timed.
- Figures: [Convolution_arithmetic_-_No_padding_strides.gif](https://commons.wikimedia.org/wiki/File:Convolution_arithmetic_-_No_padding_strides.gif) and [Convolution_arithmetic_-_Padding_strides.gif](https://commons.wikimedia.org/wiki/File:Convolution_arithmetic_-_Padding_strides.gif) (Vincent Dumoulin and Francesco Visin, MIT licence), on Wikimedia Commons. I viewed frames of both before writing the captions.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision Convolution:::Sliding a small window (the kernel) over a picture and writing down one number at each spot for "how much does this look like a pattern, such as a left edge?". The numbers form a new, smaller picture.

#flashcards/ml/formula #flashcards/topic/vision Write the convolution and explain each symbol.
?
$$y_{u,v}=b+\sum_{a=0}^{k-1}\sum_{c=0}^{k-1}K_{a,c}\;x_{us+a-p,\;vs+c-p}$$
- $x$: the input picture. $x_{r,q}$: the pixel in row $r$ and column $q$ (zero outside the picture, which is what padding supplies).
- $K_{a,c}$: the kernel number in row $a$ and column $c$. These are the learned numbers. A large positive number means "more of this pixel makes the output bigger".
- $b$: the bias. $k$: the kernel size. $s$: the stride (how far the window jumps). $p$: the padding on each side.
- $y_{u,v}$: the output at position $(u,v)$.

#flashcards/ml/formula #flashcards/topic/vision Write the output size of a convolution and explain each symbol.
?
$$o=\left\lfloor\frac{i-k+2p}{s}\right\rfloor+1$$
- $i$: the input width (or height). A bigger picture gives a bigger output.
- $k$: the kernel size. A bigger kernel needs more room, so the output shrinks by $k-1$ (stride 1, no padding).
- $p$: the padding on each side, counted twice ($2p$). $s$: the stride.
- $\lfloor\cdot\rfloor$: round down. The $+1$ counts the first position.

#flashcards/ml/cloze #flashcards/topic/vision The output of a convolution is ==smaller== than the input unless you add ==padding==.

#flashcards/ml/example #flashcards/topic/vision Picture $\begin{pmatrix}1&2&3\\4&5&6\\7&8&9\end{pmatrix}$, kernel $\begin{pmatrix}1&0\\0&-1\end{pmatrix}$, $b=0$, $s=1$, $p=0$. What is the output?::Size ⌊(3 − 2)/1⌋ + 1 = 2. y₀,₀ = 1·1 − 1·5 = −4. y₀,₁ = 2 − 6 = −4. y₁,₀ = 4 − 8 = −4. y₁,₁ = 5 − 9 = −4. Every output is −4.

#flashcards/ml/example #flashcards/topic/vision For $i=5$, $k=3$, $s=2$, what is the output size with $p=0$ and with $p=1$?::p = 0: ⌊(5 − 3 + 0)/2⌋ + 1 = 2. p = 1: ⌊(5 − 3 + 2)/2⌋ + 1 = 3. Padding the border with zeros gives more positions.

#flashcards/ml/image #flashcards/topic/vision ![[convolution-no-padding-strides.gif]]
What does this picture show?
?
A 3×3 kernel (the dark square) slides over a 5×5 input (blue) with stride 2 and no padding. Each position gives one cell of the 2×2 output (cyan).

#flashcards/ml/image #flashcards/topic/vision ![[convolution-padding-strides.gif]]
What does this picture show?
?
The same 5×5 input with a border of zeros added (dashed cells), a 3×3 kernel and stride 2. Now there are 3×3 positions, so the output is 3×3.
