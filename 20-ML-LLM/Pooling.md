---
tags: [pooling, max-pooling, average-pooling, cnn, deep-learning, computer-vision, ML]
status: growing
level: 4
tier: 2
---

# Pooling

**Pooling** shrinks a feature map by summarising each small block with one number. In **max pooling** the number is the biggest value in the block, and in **average pooling** it is the average. It sits between convolution layers in a [[Convolutional Neural Network]], right after the [[Convolution]] and its [[Activation Function]]. Pooling has no numbers to learn. Its job is to make the maps smaller, so later layers are cheaper, and to keep only the strongest signals.

## 🟢 1. Simple (age 12)
Imagine you took a big photo and want a small thumbnail. You cut the photo into little squares, and for each square you keep only the brightest spot. The thumbnail is much smaller, but the important bright things are still there.

**Example:** In a 4×4 grid of scores, you split it into four 2×2 corners and keep the best score of each corner. You end up with 4 numbers instead of 16.

## 🔵 2. How it works
1. **Cut the map into blocks.** The usual choice is 2×2 blocks that do not overlap, so the window jumps 2 steps at a time (stride 2, see [[Convolution]]).
2. **Summarise each block with one number.** *Max pooling* keeps the largest, *average pooling* keeps the mean.
3. **The map gets smaller.** With 2×2 blocks the width and the height are halved, so there are four times fewer numbers.
4. **Why max?** A big number means "the pattern I look for is somewhere in this block". Max pooling keeps that message and forgets exactly where in the block it was. So a tiny shift of the pattern inside the block often changes nothing.
5. **Why average?** It keeps a smooth summary of the block. It is often used at the very end, where one average per map (called *global average pooling*) replaces a big flatten step.
6. **Nothing to learn.** Pooling just applies a fixed rule, so it has no learned numbers. During training ([[Backpropagation]]) the error still has to pass back through it, and the two kinds do this differently (equation 2 below).

![[cnn-max-pooling.png]]
*Max pooling with 2×2 blocks. The 6×6 map (left) is cut into nine blocks, and the biggest number of each block (red) is kept, giving the 3×3 result (right). I ran the same map through PyTorch and got the same nine numbers. Image: Daniel Voigt Godoy, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Convolutional_neural_network,_maxpooling.png), CC BY 4.0.*

## 🟠 3. Technical detail
All numbers come from the code at the end of this section (Python 3.9.6, NumPy 1.24.4, PyTorch 2.8.0 on a CPU).

- **By hand, then the library:** the 4×4 picture `[[1,3,2,4],[5,6,1,2],[7,2,9,0],[3,4,5,8]]` with 2×2 blocks gave the max `[[6, 4], [7, 9]]` and the average `[[3.75, 2.25], [4.0, 5.5]]`. My own loops and PyTorch's `max_pool2d` and `avg_pool2d` agreed on both.
- **The picture above:** its 6×6 map pooled to `[[22, 23, 11], [24, 7, 1], [13, 13, 13]]`, exactly what the figure shows.
- **Output size:** the same rule as for convolution with no padding, $\lfloor(i-k)/s\rfloor+1$. The code compared it with PyTorch on 168 settings (sizes 1 to 12, windows 1 to 4, stride 1 to 4) and it matched on all 168. A 28×28 map with 2×2 blocks becomes 14×14, which is 196 numbers instead of 784, a ratio of 0.25.
- **Leftovers:** with width 5 and 2×2 blocks the output is 2 wide, so the last row and column are never looked at. The code changed them and the output did not move.
- **Learning:** with the top-left block `[[1, 3], [5, 6]]`, the gradient that flows back to the four pixels was `[[0, 0], [0, 1]]` for max pooling (all of it goes to the winner, 6) and `[[0.25, 0.25], [0.25, 0.25]]` for average pooling. A pooling layer holds 0 learned numbers.
- **Small shifts:** I used a row of 8 with one bright pixel and moved it one place to the right. With 2-wide blocks the max-pooled output stayed the same for 4 of the 7 starting places (0, 2, 4 and 6) and changed for the other 3. So max pooling tolerates *some* small shifts, not all. It is a partial effect, and the code measures it on one very simple row, not on real pictures.
- **Global average pooling:** a batch with 3 channels of 5×5 gave one number per channel (shape `(1, 3)`), equal to the mean of each channel. (From memory, this lets a network accept pictures of different sizes. I did not test that.)
- **What pooling saves:** the same small CNN for the 8×8 digits (two 3×3 convolutions with 8 and 16 filters, one linear layer for 10 labels) holds 1,898 numbers with its two 2×2 max-pools and 11,498 without them, 6.06 times more, because the linear layer then reads 16×8×8 = 1,024 numbers instead of 16×2×2 = 64. I did not compare accuracy with and without pooling.
- **Limits:** I tested only 2-D max and average pooling (plus one 1-D row), with no padding, no overlap beyond what the settings gave, and tiny inputs. I did not test other kinds (such as lp-pooling or learned pooling), a pooling layer's effect on accuracy, or speed.

The code covers seven things: (1) max and average by hand, (2) the size rule, (3) the gradient rule, (4) a small shift, (5) global average pooling, (6) the figure's map, (7) what pooling saves in a small CNN.

```python
import itertools
import numpy as np
import torch
import torch.nn.functional as F
t = lambda a: torch.tensor(np.asarray(a, dtype=float), dtype=torch.float64)[None, None]
x = [[1, 3, 2, 4], [5, 6, 1, 2], [7, 2, 9, 0], [3, 4, 5, 8]]
# 1) cut the picture into 2x2 blocks (stride 2) and keep the biggest, or the average, of each block
mx = np.array([[max(np.array(x)[2 * u:2 * u + 2, 2 * v:2 * v + 2].ravel()) for v in range(2)] for u in range(2)])
av = np.array([[np.array(x)[2 * u:2 * u + 2, 2 * v:2 * v + 2].mean() for v in range(2)] for u in range(2)])
print('max pool:', mx.tolist(), '| equals library:', np.allclose(mx, F.max_pool2d(t(x), 2)[0, 0].numpy()), '| average pool:', av.tolist(), '| equals library:', np.allclose(av, F.avg_pool2d(t(x), 2)[0, 0].numpy()))
# 2) the output size follows the same rule as convolution (no padding): floor((i - k) / s) + 1
n = ok = 0
for i, k, s in itertools.product(range(1, 13), range(1, 5), range(1, 5)):
    if i < k: continue
    n += 1; ok += F.max_pool2d(t(np.ones((i, i))), k, s).shape[-1] == (i - k) // s + 1
print('size rule matches the library on', ok, 'of', n, 'settings | i=5, k=2, s=2 ->', F.max_pool2d(t(np.ones((5, 5))), 2, 2).shape[-1], '| 28 x 28 with 2x2 blocks -> 14 x 14 =', 14 * 14, 'numbers instead of', 28 * 28, '(ratio', 14 * 14 / (28 * 28), ')')
a = np.arange(25.).reshape(5, 5); b = a.copy(); b[4, :] += 100; b[:, 4] += 100
print('last row and column (never covered) change the output:', not np.allclose(F.max_pool2d(t(a), 2)[0, 0].numpy(), F.max_pool2d(t(b), 2)[0, 0].numpy()))
# 3) learning: max sends the blame to the winner only, average shares it equally
def grad(pool):
    z = t(x).requires_grad_(); pool(z)[0, 0, 0, 0].backward(); return z.grad[0, 0, :2, :2].tolist()
print('gradient into the top-left block [[1, 3], [5, 6]] -> max pool:', grad(lambda z: F.max_pool2d(z, 2)), '| average pool:', grad(lambda z: F.avg_pool2d(z, 2)))
print('learned numbers inside a pooling layer:', sum(q.numel() for q in torch.nn.MaxPool2d(2).parameters()))
# 4) small shifts: a single bright pixel in a row of 8, moved one place to the right
def pooled(j): r = np.zeros(8); r[j] = 9; return F.max_pool1d(torch.tensor(r)[None, None], 2)[0, 0].numpy().tolist()
same = [j for j in range(7) if pooled(j) == pooled(j + 1)]
print('pixel at 1 gives', pooled(1), 'and at 2 gives', pooled(2), '| moved by one place, the output is unchanged for', len(same), 'of 7 starting places:', same)
# 5) global average pooling: one number per channel, whatever the picture size
g = torch.arange(3 * 5 * 5, dtype=torch.float64).reshape(1, 3, 5, 5)
print('global average pooling shape:', tuple(F.adaptive_avg_pool2d(g, 1).flatten(1).shape), '| equals the mean of each channel:', np.allclose(F.adaptive_avg_pool2d(g, 1).flatten().numpy(), g.mean((2, 3)).flatten().numpy()))
# 6) the 6x6 map from the picture
pic = [[-19, 22, -20, -12, -17, 11], [16, -30, -1, 23, -7, -14], [-14, 24, 7, -2, 1, -7], [-15, -10, -1, -1, -15, 1], [-13, 13, -11, -5, 13, -7], [-18, 9, -18, 13, -3, 4]]
print('picture 6x6 max pooled:', F.max_pool2d(t(pic), 2)[0, 0].int().tolist(), '| shown in the picture: [[22, 23, 11], [24, 7, 1], [13, 13, 13]]')
# 7) what pooling saves: the same small CNN for 8x8 digits (16 maps, 10 labels) with and without its two 2x2 max-pools
def small_cnn(pool):
    s = 2 if pool else 8; p = [torch.nn.MaxPool2d(2)] if pool else []
    return torch.nn.Sequential(torch.nn.Conv2d(1, 8, 3, padding=1), torch.nn.ReLU(), *p, torch.nn.Conv2d(8, 16, 3, padding=1), torch.nn.ReLU(), *p, torch.nn.Flatten(), torch.nn.Linear(16 * s * s, 10))
n_with, n_without = (sum(q.numel() for q in small_cnn(b).parameters()) for b in (True, False))
print('numbers in the small CNN | with pooling:', n_with, '| without pooling:', n_without, '| ratio', round(n_without / n_with, 2))
```
Reading the output (Python 3.9.6, PyTorch 2.8.0):
- **Pooling by hand:** max `[[6, 4], [7, 9]]` and average `[[3.75, 2.25], [4.0, 5.5]]`, both equal to the library (`True`).
- **Size:** 168 of 168 settings agree, width 5 gives 2, and 28×28 becomes 14×14 (196 against 784, ratio 0.25). The leftover row and column change nothing (`False`).
- **Gradient:** `[[0.0, 0.0], [0.0, 1.0]]` for max and four times 0.25 for average. A pooling layer has 0 learned numbers.
- **Shift:** a pixel at place 1 gives `[9.0, 0.0, 0.0, 0.0]` and at place 2 gives `[0.0, 9.0, 0.0, 0.0]`. After a move of one place the output is unchanged for 4 of 7 starting places: 0, 2, 4, 6.
- **Global average:** shape `(1, 3)` and equal to the channel means (`True`).
- **Figure:** `[[22, 23, 11], [24, 7, 1], [13, 13, 13]]`, the same as the picture.
- **Pooling saves:** 1,898 numbers with pooling and 11,498 without, a ratio of 6.06.
- **Limits:** the code runs in a blink and times nothing.

## 🔴 4. Deep / formal

**What does max pooling output?** For a window of size $k$ and stride $s$, the output at row $u$ and column $v$ is the biggest value in its block:

$$y_{u,v}=\max_{0\le a,c<k}\;x_{us+a,\;vs+c}$$

(Average pooling replaces the max by $\frac{1}{k^{2}}\sum_{a,c}x_{us+a,\,vs+c}$.)

**Where:**
- $x$ is the input feature map. **Matters:** it is the result of the previous convolution and activation, so a large value means a strong match of some pattern.
- $k$ is the window size (2 is usual). **Matters:** it sets how many pixels are summarised into one, and so how much location detail is thrown away.
- $s$ is the stride (usually equal to $k$ so that blocks do not overlap). **Matters:** it sets the shrink factor. With $s=k=2$ each side is halved.
- $a,c$ run over the rows and columns inside the block. **Matters:** the max looks at all of them but remembers only the largest, so the location of the winner inside the block is lost.
- $y_{u,v}$ is the output number. **Matters:** these numbers form the smaller map handed to the next layer.
- The $\frac{1}{k^2}$ in the average version is the number of pixels in the block. **Matters:** without it you would get a sum that grows with the window, not an average.

**Example (tiny numbers):** the 4×4 picture from the code, $k=s=2$.
1. The top-left block is $\begin{pmatrix}1&3\\5&6\end{pmatrix}$, so $y_{0,0}=\max(1,3,5,6)=6$. Its average is $(1+3+5+6)/4=15/4=3.75$.
2. The other blocks give $y_{0,1}=\max(2,4,1,2)=4$, $y_{1,0}=\max(7,2,3,4)=7$ and $y_{1,1}=\max(9,0,5,8)=9$.
3. The code gave `[[6, 4], [7, 9]]` for max and `[[3.75, 2.25], [4.0, 5.5]]` for average, equal to PyTorch's.
4. Meaning: 16 numbers became 4. Max kept the strongest value of each corner (6, 4, 7, 9), and average kept a blend that never exceeds the max.

**How does the error flow back?** During training, each output $y$ passes its share of the error back to the inputs $x_j$ in its block. By the chain rule it is the slope of $y$ with respect to each $x_j$:

$$\frac{\partial y}{\partial x_j}=\begin{cases}1 & \text{max pooling, if } x_j \text{ is the maximum of the block}\\ 0 & \text{max pooling, for every other } x_j\\ \dfrac{1}{k^{2}} & \text{average pooling, for every } x_j \text{ in the block}\end{cases}$$

**Where:**
- $y$ is the pooled output of one block. **Matters:** the error arriving at $y$ is multiplied by this slope on its way back.
- $x_j$ is one input pixel inside that block. **Matters:** it gets a non-zero error only if the slope for it is non-zero.
- The "1 for the winner, 0 for the rest" case. **Matters:** nudging the winner changes the output one-for-one, and nudging any loser changes nothing (as long as it stays below the winner), so only the winner learns from this block. (If two pixels tie, how the error is shared is a library detail that I did not test.)
- The $\dfrac{1}{k^2}$ case. **Matters:** every pixel moves the average by the same small amount, so all of them share the error equally.

**Example (tiny numbers):** the block $\begin{pmatrix}1&3\\5&6\end{pmatrix}$ again, with $k=2$.
1. Max pooling: the maximum is 6, in the bottom-right corner, so the slopes are $\begin{pmatrix}0&0\\0&1\end{pmatrix}$.
2. Average pooling: each pixel gets $1/2^2=0.25$, so the slopes are $\begin{pmatrix}0.25&0.25\\0.25&0.25\end{pmatrix}$.
3. The code printed exactly these two, using PyTorch's automatic gradients.
4. Meaning: if the error at this output is, say, 2, max pooling sends all 2 to the pixel holding the 6, and average pooling sends 0.5 to each of the four pixels. (The "say 2" is only an illustration: the slopes above are what was computed.)

**Other points (from memory, not checked here):**
- Many modern networks replace pooling by a convolution with stride 2 (which *learns* how to shrink) and finish with global average pooling. I did not compare accuracy.
- Pooling is also credited with giving a small amount of *translation invariance* (the output barely changes when a pattern moves a little). The code above shows that this works only for some shifts, which is a reason some researchers think pooling is a rough tool, but I did not check that discussion.
- A slide in the owner's course says pooling helps with small rotations too. I did not test rotations at all, and I am not sure that max pooling alone handles them. Treat that as unverified.
- Pooling in a CNN is usually done on each channel separately, so the number of channels stays the same. I did not test this with several channels, except in the global average case, where it was per channel.

**What came before, and how this improved on it:**
- **Before:** networks that kept every feature map at full size, so the first dense layer had to read all of them. Early convolution networks already shrank the maps with "subsampling" layers (the LeNet-5 layout has two, S2 and S4, per Wikipedia's article which I read). Taking the *maximum* of a block, rather than the average, is from memory credited to later work, and I did not check who or when.
- **Its limit:** big maps mean many numbers in the layers after them (more memory, more overfitting), and the network reacts to the exact position of a pattern.
- **What changed:** summarise each small block with one number (the biggest or the average), with nothing to learn, which makes the maps smaller and the answer less sensitive to exact position.
- **How much better:** in this note's small CNN, two 2×2 max-pools cut the learned numbers from 11,498 to 1,898 (6.06 times fewer) and each pool makes the maps 4 times smaller. The shift tolerance was only partial (4 of 7 starting places in the toy row). I did not measure accuracy with and without pooling.

## Related
[[Convolutional Neural Network]] · [[Convolution]] · [[Receptive Field]] · [[Activation Function]] · [[Backpropagation]] · [[Neural Network]] · [[PyTorch]] · [[Dropout]] · [[Feature Map]]
- [[VGG]]

## Sources
- The owner's course slides "Convolutional.pdf" (max and average pooling pages), read as text. I did not copy any slide image.
- Library versions in this run: Python 3.9.6, NumPy 1.24.4, PyTorch 2.8.0 (CPU). The code and every printed value were run for this note. Nothing was timed.
- Figure: [Convolutional_neural_network,_maxpooling.png](https://commons.wikimedia.org/wiki/File:Convolutional_neural_network,_maxpooling.png) (Daniel Voigt Godoy, CC BY 4.0), on Wikimedia Commons. I viewed it before writing the caption, and the same figure is also used in [[Convolutional Neural Network]].

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision Pooling:::Making a small thumbnail of a feature map: cut it into little squares and keep only the brightest (max pooling) or the average (average pooling) of each square.

#flashcards/ml/formula #flashcards/topic/vision Write max pooling and explain each symbol.
?
$$y_{u,v}=\max_{0\le a,c<k}\;x_{us+a,\;vs+c}$$
- $x$: the input feature map, the result of the previous convolution and activation. A large value means a strong match of some pattern.
- $k$: the window size (2 is usual). It sets how many pixels are summarised into one.
- $s$: the stride (usually equal to $k$ so blocks do not overlap). With $s=k=2$ each side is halved.
- $a,c$: run over the rows and columns inside the block. The max looks at all of them but remembers only the largest, so the location of the winner inside the block is lost.
- $y_{u,v}$: the output number. Average pooling replaces the max by $\frac{1}{k^2}\sum_{a,c}x_{us+a,\,vs+c}$.

#flashcards/ml/formula #flashcards/topic/vision Write the gradient of pooling and explain each symbol.
?
$$\frac{\partial y}{\partial x_j}=\begin{cases}1 & \text{max pooling, if } x_j \text{ is the maximum of the block}\\ 0 & \text{max pooling, for every other } x_j\\ \dfrac{1}{k^{2}} & \text{average pooling, for every } x_j \text{ in the block}\end{cases}$$
- $y$: the pooled output of one block. The error arriving at $y$ is multiplied by this slope on its way back.
- $x_j$: one input pixel inside the block. It gets a non-zero error only if its slope is non-zero.
- Max pooling: only the winner learns from this block, because nudging a loser changes nothing.
- Average pooling: every pixel moves the average by the same small amount, so all share the error equally.

#flashcards/ml/cloze #flashcards/topic/vision Max pooling keeps only the ==largest== number in each block, so only the ==winner== gets any gradient.

#flashcards/ml/example #flashcards/topic/vision For the block $\begin{pmatrix}1&3\\5&6\end{pmatrix}$ with $k=2$, what are the slopes for max pooling and for average pooling?::Max: the maximum is 6, so the slopes are `[[0, 0], [0, 1]]`. Average: each pixel gets 1/2² = 0.25, so `[[0.25, 0.25], [0.25, 0.25]]`. (The note computed these with PyTorch.)

#flashcards/ml/image #flashcards/topic/vision ![[cnn-max-pooling.png]]
What does this picture show?
?
Max pooling with 2×2 blocks. The 6×6 map is cut into nine blocks and the biggest number of each block (red) is kept, giving the 3×3 result.
