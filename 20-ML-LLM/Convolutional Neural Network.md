---
tags: [machine-learning, deep-learning, neural-network, CNN, convolution, computer-vision]
status: growing
level: 4
tier: 3
---

# Convolutional Neural Network

Often shortened to **CNN** (or ConvNet). It is a [[Neural Network]] designed for data laid out in a grid, above all **images**.

## 🟢 1. Simple (age 12)
A CNN looks at a picture the way you might look at it through a **small sliding window**. The window moves across the picture, and at each spot it asks a simple question such as "is there an edge here?" or "is there a bit of fur here?". The same small window is used **everywhere**, so the network does not have to learn "edge" separately for every corner of the picture. Later layers combine the simple findings into bigger ones: edges into shapes, shapes into eyes, eyes into faces.

**Example:** Finding Wally in a crowded picture. You do not memorise where he is. You carry a **pattern in your head** (red and white stripes) and slide your eyes over the whole page, checking each spot against that pattern. A CNN's small window is that pattern, and sliding it over the picture is called a **convolution**.

## 🔵 2. How it works
1. A picture is a grid of numbers (the brightness of each pixel; a colour picture has three grids, one each for red, green and blue).
2. A **filter** (also called a **kernel**) is a small grid of numbers, for example 3 by 3. These numbers are the weights that the network learns.
3. **Slide** the filter over the picture. At each position, multiply the filter's numbers by the pixel numbers underneath, element by element, and **add them all up** to get one number.
4. Doing this at every position gives a new grid, called a **feature map**. It lights up wherever the picture matches the filter's pattern.
5. A layer uses **many filters** at once (for example 32), so it produces many feature maps, each looking for a different pattern.
6. After the convolution, pass the numbers through an [[Activation Function]] (usually ReLU).
7. **Pooling** shrinks each feature map: for each small block, keep only the biggest number (**max pooling**). This keeps "was the pattern found here?" but forgets the exact position, and it makes the data smaller.
8. Stack several of these convolution and pooling steps. Early layers find simple things (edges, colours), later ones find complex things (eyes, wheels).
9. At the end, **flatten** the maps into one long list and feed it to ordinary layers that give the answer ([[Classification]]: "rabbit 0.7, horse 0.3, pig 0.2" through [[Softmax]]).

![[cnn-architecture.png]]
*A whole CNN. Left: the input picture (a page with a rabbit drawing). Middle: stacks of feature maps from the convolutional and pooling layers ("feature learning"); the small squares and dashed lines show a filter looking at one small patch of the layer before. Then the maps are flattened into a list and passed to connected layers ("classification"), which output a chance for each animal. Image: Irisbox, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Convolutional_Neural_Network.png), CC BY 4.0.*

![[cnn-convolution-worked-example.png]]
*A worked convolution. Top left: a 6 by 6 image of numbers with a border of zeros added around it ("padding"). Middle: a 3 by 3 filter. Right: the 6 by 6 result. Bottom: the one spot marked in red. The grey 3 by 3 region of the image is multiplied element by element with the filter, giving the grid at the bottom right, and its entries add up to 7, which is the red number in the result. Image: Daniel Voigt Godoy, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Convolutional_neural_network,_convolution_worked_example.png), CC BY 4.0.*

![[cnn-max-pooling.png]]
*Max pooling with 2 by 2 blocks. The 6 by 6 map (left) is cut into nine blocks; the biggest number in each block (in red) is kept, giving the 3 by 3 result (right). It is the result of the convolution above, pooled. Image: Daniel Voigt Godoy, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Convolutional_neural_network,_maxpooling.png), CC BY 4.0.*

## 🟠 3. Technical detail
- **Two big ideas:**
  - **Local connections:** each output number looks at only a small patch, not the whole picture. Nearby pixels are related, far-away pixels much less so.
  - **Weight sharing:** the same filter is used at every position. A pattern learned in one corner is recognised in any corner (this is called **translation equivariance**: shift the input and the feature map shifts the same way).
- **Hyper-settings of a convolution layer:**
  - **Kernel size** $k$ (3 by 3 is very common).
  - **Number of filters** (the number of output channels).
  - **Stride** $s$: how far the window jumps each step (stride 2 halves the size).
  - **Padding** $p$: zeros added around the border, so the output can keep the same size as the input.
- **Channels:** a colour image has 3 input channels. A filter then has a depth of 3 as well, and sums over all channels to give one output number.
- **Receptive field:** the patch of the original picture that one number deep in the network "sees". It grows with every layer, which is how small windows add up to large-scale understanding.
- **Pooling alternatives:** average pooling (take the mean), and a strided convolution (let the filter jump by 2). Many modern networks pool less and use strides.
- **Training:** exactly like any other network. A [[Loss Function]], [[Backpropagation]] and [[Gradient Descent]] (usually [[Stochastic Gradient Descent]] with [[Momentum]], or [[Adam Optimizer]]) learn the filter numbers. Nobody designs the edge detectors by hand: they emerge from training (commonly reported for the first layers; from memory).
- **Making them deep and trainable:** [[Normalization]], ReLU-style activations and [[Residual Connection]]s (the ResNet family) allowed CNNs with dozens or hundreds of layers. Overfitting is fought with [[Dropout]], other forms of [[Regularization]], and **data augmentation** (flipping, cropping and colour-shifting the training images).
- **Beyond images:** the same idea works for one-dimensional data such as audio and text (1-D convolutions), and for video and 3-D scans.
- **CNNs and Transformers:** since about 2020, [[Transformer]]-based vision models have matched or beaten CNNs on many image tasks, especially with lots of data. CNNs remain popular when data or computing power is limited (a common picture; the details change quickly, so check recent results).
- **What a CNN does not know:** it has no built-in sense of rotation or scale, and it can be fooled by small, carefully made changes to an image (adversarial examples). It also learns whatever shortcuts the training data allows.

```python
import numpy as np

# the worked example from the picture: 6x6 image, 3x3 filter, padding 1, stride 1
img = np.array([[5,0,8,7,8,1],[1,9,5,0,7,7],[6,0,2,4,6,6],[9,7,6,6,8,4],[8,3,8,5,1,3],[7,2,7,0,1,0]], float)
K = np.array([[0,1,0],[1,-4,1],[0,1,0]], float)
P = np.pad(img, 1)                                                   # a border of zeros
R = np.array([[(P[i:i+3, j:j+3] * K).sum() for j in range(6)] for i in range(6)])
print(R.astype(int))
# [[-19  22 -20 -12 -17  11]
#  [ 16 -30  -1  23  -7 -14]
#  [-14  24   7  -2   1  -7]
#  [-15 -10  -1  -1 -15   1]
#  [-13  13 -11  -5  13  -7]
#  [-18   9 -18  13  -3   4]]       identical to the picture; R[2,2] is the red 7
print(R[2, 2])                                                       # 7.0
print(R.reshape(3, 2, 3, 2).max(axis=(1, 3)))                        # max pooling, 2x2 blocks
# [[22. 23. 11.]
#  [24.  7.  1.]
#  [13. 13. 13.]]

out = lambda n, k, p, s: (n + 2 * p - k) // s + 1                   # output size of a convolution
print(out(28, 3, 0, 1), out(32, 5, 2, 1), out(32, 3, 1, 2), out(7, 3, 0, 2))   # 26 32 16 3

print(3 * 3 * 1 * 32 + 32, 784 * (26 * 26 * 32), 3 * 3 * 3 * 64 + 64)          # 320 16959488 1792
```

## 🔴 4. Deep / formal
**The convolution itself.** For an input grid $X$ and a $k\times k$ filter $K$, the output at position $(i,j)$ is the sum of the filter times the patch of the input starting there:

$$
Y[i,j]=\sum_{a=0}^{k-1}\sum_{b=0}^{k-1}K[a,b]\;X[i+a,\,j+b]\;+\;b_0
$$

**Where:**
- $X$ is the input grid (an image or a feature map) and $X[i+a,j+b]$ is the pixel at that row and column. $K$ is the filter, a $k\times k$ grid of learned weights, and $K[a,b]$ is one of its entries.
- $b_0$ is a single learned **bias** number added to every output of this filter (often left out in the pictures).
- The two sums run over the rows $a$ and columns $b$ of the filter, so each output is "multiply the filter with the patch underneath, entry by entry, then add everything up". $Y[i,j]$ is the output at position $(i,j)$.
- **Matters:** the **same** $K$ is used for every $(i,j)$. That is the weight sharing. Without it, every position would need its own weights, and the network would have to learn each pattern separately everywhere.
- **Matters:** what the filter detects depends on its numbers. A filter like $(-1,\,1)$ responds where brightness jumps (a vertical edge), and one with a positive centre and negative surroundings responds to bright spots.
- **Naming warning:** the formula above, without flipping the filter, is strictly called **cross-correlation**. Deep learning libraries call it "convolution" anyway. A true mathematical convolution flips the filter first, and since the filter is learned, the difference does not matter in practice.

**Example (tiny numbers):** the red spot in the worked example, with the filter $K=\begin{pmatrix}0&1&0\\1&-4&1\\0&1&0\end{pmatrix}$ over the region $\begin{pmatrix}9&5&0\\0&2&4\\7&6&6\end{pmatrix}$ (bias taken as 0).
- Entry by entry: $0\cdot9+1\cdot5+0\cdot0=5$, then $1\cdot0+(-4)\cdot2+1\cdot4=-4$, then $0\cdot7+1\cdot6+0\cdot6=6$.
- Add: $5-4+6=\mathbf{7}$, the number in the red box.
- Meaning: this filter compares the middle pixel with its four neighbours (above, below, left, right). It gives a big value when the centre differs a lot from its surroundings, so it is a simple detector of spots and edges. The code reproduces the whole $6\times6$ result of the picture exactly.

**How big is the output?** For an input of width $n$, filter size $k$, padding $p$ and stride $s$, the output width is:

$$
n_{\text{out}}=\left\lfloor\frac{n+2p-k}{s}\right\rfloor+1
$$

**Where:**
- $n$ is the input width (the same formula works for height), $k$ is the filter size, $p$ is how many rows and columns of zeros are added on **each** side, and $s$ is the stride (how many pixels the window jumps).
- $n+2p$ is the padded width, and $n+2p-k$ is how far the window can travel before it hits the edge. Dividing by $s$ counts the jumps, $\lfloor\cdot\rfloor$ ("floor") rounds down, and the $+1$ counts the starting position.
- **Matters:** this decides the sizes of every layer when you design a network. With $s=1$ and $p=(k-1)/2$ (for odd $k$) the output has the **same size** as the input ("same" padding). With no padding, each layer shrinks the map. A bigger stride shrinks it faster.

**Example (tiny numbers):**
- The worked example: $n=6$, $k=3$, $p=1$, $s=1$: $(6+2-3)/1+1=\mathbf{6}$. The output is $6\times6$ again, as in the picture.
- $n=28$, $k=3$, $p=0$, $s=1$: $(28-3)/1+1=\mathbf{26}$.
- $n=32$, $k=3$, $p=1$, $s=2$: $\lfloor(32+2-3)/2\rfloor+1=\lfloor15.5\rfloor+1=\mathbf{16}$ (stride 2 halves the size).
- Meaning: you can predict every layer's shape before building anything, which is how mismatched-size errors are avoided.

**Why CNNs are so economical: counting the weights.** A convolution layer with $C_{\text{in}}$ input channels, $C_{\text{out}}$ filters of size $k\times k$ has:

$$
\text{parameters}=k\cdot k\cdot C_{\text{in}}\cdot C_{\text{out}}+C_{\text{out}}
$$

**Where:**
- $k\cdot k$ is the number of entries in one filter slice, $C_{\text{in}}$ is how many input channels each filter spans (it has weights for every channel), and $C_{\text{out}}$ is how many filters the layer has.
- The final $+C_{\text{out}}$ counts one bias per filter.
- **Matters:** the count depends only on the filter shape and the number of channels, **not on the size of the picture**. The same layer works on a small or a huge image. A dense (fully connected) layer ([[Neural Network]]) would need one weight for every input-output pair, which grows with the picture.

**Example (tiny numbers):** a grey $28\times28$ image and 32 filters of size $3\times3$, giving $26\times26\times32=21632$ output numbers.
- Convolution: $3\cdot3\cdot1\cdot32+32=\mathbf{320}$ parameters.
- A dense layer making the same number of outputs from the $784$ input pixels needs $784\times21632=\mathbf{16{,}959{,}488}$ weights (not counting biases).
- A colour layer with $C_{\text{in}}=3$ and $64$ filters: $3\cdot3\cdot3\cdot64+64=\mathbf{1792}$.
- Meaning: weight sharing and local connections make the convolution layer tens of thousands of times smaller here, so it needs far less data to train and is much less prone to [[Overfitting]].

**Further notes:**
- **Pooling and invariance:** max pooling gives a little tolerance to small shifts (a pattern moved by a pixel often still gives the same pooled number). Full invariance is not guaranteed.
- **1 by 1 convolutions** ($k=1$) mix channels at each position without looking at neighbours. They are used to change the number of channels cheaply.
- **Early landmarks:** LeNet (handwritten digits, 1990s), AlexNet (2012 image contest), then deeper networks such as VGG and ResNet. These are the usual milestones in the story (from memory; check the papers for dates).
- **Why the first layers look like edge detectors:** pictures of the natural world are full of edges, so filters that find them are useful. This has been seen in many trained CNNs, but it is an observation about trained networks rather than a theorem.
- **Not covered yet in this vault:** recurrent networks for sequences, and how Transformers handle images.

**What came before, and how this improved on it:**
- **Before:** fully connected networks on raw pixels, and pixel-by-pixel comparison with stored examples, then hand-made features such as edge maps fed to a separate classifier (named from memory; see [[Image Classification]] and [[Feature Map]]).
- **Its limit:** a plain network needs a weight for every pixel and every unit (150,529,000 numbers for the first layer on a 224×224 colour photo, counted in [[Image Classification]]), and a picture that moves a little looks like a different picture to it.
- **What changed:** small shared filters ([[Convolution]]), shrinking maps ([[Pooling]]), and filters learned from the labels, so the features are no longer designed by hand.
- **How much better:** measured in this vault's notes, on small 8×8 digits: a CNN of 1,898 numbers against an MLP of 4,810 (39%), and 0.66 against 0.51 accuracy (mean of 5 seeds) when the test pictures were shifted one pixel ([[Image Classification]]). A LeNet-shaped CNN holds 61,706 numbers against 134,014 for an MLP of the same shape ([[LeNet and AlexNet]]). On large photos the 2012 AlexNet's top-5 error in the ImageNet contest was 15.3% against about 26% for the runner-up (Wikipedia, which I read).

## Related
- [[Neural Network]] · [[Activation Function]] · [[Classification]] · [[Softmax]] · [[Backpropagation]] · [[Loss Function]] · [[Gradient Descent]] · [[Stochastic Gradient Descent]] · [[Momentum]] · [[Adam Optimizer]] · [[Normalization]] · [[Residual Connection]] · [[Dropout]] · [[Regularization]] · [[Overfitting]] · [[Transformer]] · [[Matrix]] · [[GPU]] · [[Recurrent Neural Network]] · [[Diffusion Models]] · [[Self-Attention]] · [[Convolution]] · [[Pooling]] · [[Receptive Field]] · [[Feature Map]] · [[Image Classification]] · [[Data Augmentation]] · [[Transfer Learning]] · [[LeNet and AlexNet]] · [[ResNet]] · [[Vision Transformer]] · [[History of Image Classification]] · [[VGG]] · [[Object Detection]] · [[Image Segmentation]] · [[Transfer Learning vs Vanilla CNN]]
- [[MobileNet]]
## Sources
- LeCun, Bottou, Bengio & Haffner, *Gradient-Based Learning Applied to Document Recognition*, 1998 (LeNet; from memory)
- Krizhevsky, Sutskever & Hinton, *ImageNet Classification with Deep Convolutional Neural Networks*, 2012 (AlexNet; from memory)
- He et al., *Deep Residual Learning for Image Recognition*, 2015 (from memory)
- Goodfellow, Bengio & Courville, *Deep Learning*, 2016, chapter 9
- Stanford CS231n, *Convolutional Neural Networks for Visual Recognition* (free online course notes; from memory)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision Convolutional neural network (CNN):::A network that looks at a picture through a small sliding window. The same small filter is used everywhere, asking simple questions like "is there an edge here?".

#flashcards/ml/formula #flashcards/topic/vision Write the convolution and explain each symbol.
?
$$Y[i,j]=\sum_{a=0}^{k-1}\sum_{b=0}^{k-1}K[a,b]\;X[i+a,\,j+b]\;+\;b_0$$
- $X$: the input grid (an image or a feature map). $X[i+a,j+b]$: the pixel at that row and column.
- $K$: the filter, a $k\times k$ grid of learned weights. $K[a,b]$: one entry.
- $b_0$: a single learned bias added to every output of this filter.
- $Y[i,j]$: the output at spot $(i,j)$. The two sums multiply the window by the filter entry by entry and add up.

#flashcards/ml/formula #flashcards/topic/vision Write the output size of a convolution and explain each symbol.
?
$$n_{\text{out}}=\left\lfloor\frac{n+2p-k}{s}\right\rfloor+1$$
- $n$: the input width (same formula for height). $k$: the filter size.
- $p$: how many rows and columns of zeros are added on each side (padding). $s$: the stride, how many pixels the window jumps.
- $n+2p$: the padded width. $n+2p-k$: how far the window can travel before it runs off the edge.
- $\lfloor\cdot\rfloor$: round down. The $+1$ counts the starting position.

#flashcards/ml/formula #flashcards/topic/vision Write the number of parameters in a convolution layer and explain each symbol.
?
$$\text{parameters}=k\cdot k\cdot C_{\text{in}}\cdot C_{\text{out}}+C_{\text{out}}$$
- $k\cdot k$: the entries in one filter slice. $C_{\text{in}}$: how many input channels each filter spans.
- $C_{\text{out}}$: how many filters the layer has. The final $+C_{\text{out}}$ counts one bias per filter.
- The count depends only on the filter, not on the image size. That is why a CNN needs far fewer weights than a dense layer.

#flashcards/ml/cloze #flashcards/topic/vision A CNN reuses the same ==filter== at every position, so its parameter count does not depend on the ==image size==.

#flashcards/ml/example #flashcards/topic/vision The filter $K=\begin{pmatrix}0&1&0\\1&-4&1\\0&1&0\end{pmatrix}$ is applied to the region $\begin{pmatrix}9&5&0\\0&2&4\\7&6&6\end{pmatrix}$ (bias 0). What is the output?::Rows: 0·9 + 1·5 + 0·0 = 5, then 1·0 − 4·2 + 1·4 = −4, then 0·7 + 1·6 + 0·6 = 6. Total 5 − 4 + 6 = 7, the red number in the picture.

#flashcards/ml/example #flashcards/topic/vision What are the output sizes for ($n=6,k=3,p=1,s=1$), ($n=28,k=3,p=0,s=1$) and ($n=32,k=3,p=1,s=2$)?::6 (6×6 again). 26. ⌊(32 + 2 − 3)/2⌋ + 1 = ⌊15.5⌋ + 1 = 16 (stride 2 roughly halves the size).

#flashcards/ml/example #flashcards/topic/vision For a 28×28 grey image and 32 filters of 3×3, how many parameters does the convolution have compared with a dense layer for the same outputs?::Convolution: 3·3·1·32 + 32 = 320. Dense: 784 × 21,632 = 16,959,488 weights (21,632 = 26 × 26 × 32 outputs).

#flashcards/ml/image #flashcards/topic/vision ![[cnn-architecture.png]]
What does this picture show?
?
A whole CNN. The input picture goes through stacks of feature maps from convolutional and pooling layers ("feature learning"), a filter looking at one small patch of the layer before. Then the maps are flattened and passed to connected layers ("classification"), which output a chance for each animal.

#flashcards/ml/image #flashcards/topic/vision ![[cnn-convolution-worked-example.png]]
What does this picture show?
?
A worked convolution: a 6×6 image with a border of zeros (padding), a 3×3 filter and the 6×6 result. At the red spot, the 3×3 region is multiplied with the filter entry by entry and the entries add up to 7.

#flashcards/ml/image #flashcards/topic/vision ![[cnn-max-pooling.png]]
What does this picture show?
?
Max pooling with 2×2 blocks. The 6×6 map is cut into nine blocks and the biggest number in each block (in red) is kept, giving a 3×3 result.
