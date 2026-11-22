---
tags: [receptive-field, cnn, deep-learning, convolution, computer-vision, ML]
status: growing
level: 4
tier: 2
---

# Receptive Field

The **receptive field** of one number inside a [[Convolutional Neural Network]] is the patch of the *original picture* that can change that number. A number in the first layer sees only a small patch (the size of the kernel). A number several layers deeper has been built from numbers that were built from numbers, so it can see a much bigger patch, even though every single [[Convolution]] only looks at a small window. This is how a CNN goes from "I see a small edge" to "I see a whole face".

## 🟢 1. Simple (age 12)
Imagine a gossip chain. Each person only talks to the three people next to them. After one round of gossip, you know what your neighbours know. After two rounds, you know what *their* neighbours know, so news from further away has reached you. After many rounds, news from the whole street has reached you, even though everyone only ever talks to the people beside them.

**Example:** A layer that looks at 3×3 pixels sees a tiny patch. Put another such layer on top of it, and one number there depends on 5×5 pixels of the photo.

## 🔵 2. How it works
1. **Layer 1.** Each output number looks at a 3×3 window of the picture. Its receptive field is 3×3.
2. **Layer 2.** Each output number looks at a 3×3 window of layer 1's numbers. Each of those looked at 3×3 pixels, and neighbouring windows overlap, so together they cover 5×5 pixels.
3. **Keep stacking.** Every extra 3×3 layer (stride 1) adds 2 to the width, so 5 layers give 11×11.
4. **Steps make it grow faster.** When a layer moves in bigger steps (a [[Convolution]] with stride 2, or a 2×2 [[Pooling]] layer), one step in that layer equals two pixels of the original picture. Every layer after it then adds *twice* as much as before.
5. **Why it matters.** To tell a cat from a dog, the network must look at a big enough area to see the whole animal. If the final receptive field is much smaller than the thing you want to recognise, the network cannot see all of it at once.

(No picture in this note: the files on Wikimedia Commons that I found for "receptive field" were about the biology of eyes and the brain, and I found no free picture of the CNN idea.)

## 🟠 3. Technical detail
All numbers below come from the code at the end of this section (Python 3.9.6, NumPy 1.24.4, PyTorch 2.8.0 on a CPU).

- **How I measured it:** I built small networks with all weights equal to 1 and no bias, picked one output number in the middle, and asked PyTorch which input pixels get a non-zero gradient (that is, which pixels can change it). The width of that region is the receptive field. I used *average* pooling in the networks (with max pooling, the gradient reaches only the winner, so the measurement would show less than the true reach).
- **Stack of 3×3 layers:** 1, 2, 3, 4 and 5 layers gave 3, 5, 7, 9 and 11 pixels wide (the region was full, for example 121 pixels for 5 layers), exactly $2L+1$.
- **With pooling:** convolution 3×3, then 2×2 pooling with stride 2, then convolution 3×3 gave 8. Adding another pooling and another 3×3 convolution gave 18. The same three convolutions without pooling would have given only 7.
- **Fewer numbers for the same reach:** two 3×3 layers reach 5, the same as one 5×5 layer. With 8 channels in and out and no bias, the two small layers hold 1,152 numbers against 1,600 for the 5×5 one. Three 3×3 layers reach 7 like one 7×7 layer, with 1,728 numbers against 3,136. The deeper stack also gets an extra [[Activation Function]] between the layers, which I did not test here.
- **Limits:** I measured only square, stride and pooling layers with positive weights in one channel, in a 96×96 input far from the edges. Padding at the border, dilated kernels, many channels, and what a network really *uses* inside its receptive field were not tested.

The code covers three things: (1) a stack of 3×3 layers, (2) layers with strides or pooling, (3) the number of weights for the same reach.

```python
import numpy as np
import torch
import torch.nn as nn
# a "layer" is (kind, k, s); positive weights and no bias, so any input pixel that can reach the output gets a non-zero gradient
def net(layers):
    mods = []
    for kind, k, s in layers:
        if kind == 'conv': c = nn.Conv2d(1, 1, k, s, padding=k // 2, bias=False); nn.init.constant_(c.weight, 1.0); mods.append(c)
        else: mods.append(nn.AvgPool2d(k, s))
    return nn.Sequential(*mods).double()
def measured(layers, N=96):                                   # width of the input region that moves one middle output pixel
    z = torch.zeros(1, 1, N, N, dtype=torch.float64, requires_grad=True); y = net(layers)(z)
    y[0, 0, y.shape[2] // 2, y.shape[3] // 2].backward(); nz = (z.grad[0, 0] != 0).numpy()
    rows, cols = np.where(nz.any(1))[0], np.where(nz.any(0))[0]
    return int(rows.max() - rows.min() + 1), int(cols.max() - cols.min() + 1), int(nz.sum())
def formula(layers):                                          # r = 1 + sum over layers of (k - 1) * (product of earlier strides)
    r, jump = 1, 1
    for kind, k, s in layers: r += (k - 1) * jump; jump *= s
    return r
# 1) a stack of 3x3 convolutions, stride 1: every layer adds 2
for L in range(1, 6):
    ls = [('conv', 3, 1)] * L; print('layers', L, '| measured width, height, pixels:', measured(ls), '| formula:', formula(ls), '| 2L + 1 =', 2 * L + 1)
# 2) a stride or a pooling step makes later layers count double
mix = [('conv', 3, 1), ('pool', 2, 2), ('conv', 3, 1)]
print('conv3, pool2 (stride 2), conv3: measured', measured(mix)[:2], '| formula', formula(mix))
deep = [('conv', 3, 1), ('pool', 2, 2), ('conv', 3, 1), ('pool', 2, 2), ('conv', 3, 1)]
print('conv3, pool2, conv3, pool2, conv3: measured', measured(deep)[:2], '| formula', formula(deep), '| same layers without the pooling:', formula([('conv', 3, 1)] * 3))
# 3) two 3x3 layers see as far as one 5x5 layer, with fewer numbers (8 channels in and out, no bias)
two = sum(p.numel() for p in nn.Sequential(nn.Conv2d(8, 8, 3, bias=False), nn.Conv2d(8, 8, 3, bias=False)).parameters())
one = sum(p.numel() for p in nn.Conv2d(8, 8, 5, bias=False).parameters())
print('reach: two 3x3 =', formula([('conv', 3, 1)] * 2), ', one 5x5 =', formula([('conv', 5, 1)]), '| numbers: two 3x3 =', two, ', one 5x5 =', one, '| three 3x3 =', 3 * 9 * 64, 'against one 7x7 =', 49 * 64)
```
Reading the output (Python 3.9.6, PyTorch 2.8.0):
- **Stack:** measured 3, 5, 7, 9, 11 wide (9, 25, 49, 81 and 121 pixels), the same as the formula and as $2L+1$.
- **Pooling:** 8 for convolution–pool–convolution, 18 for the five-layer version, and 7 for three convolutions without pooling.
- **Numbers:** reach 5 for two 3×3 layers and for one 5×5 layer, 1,152 against 1,600 numbers, and 1,728 against 3,136 for the 7-wide case.
- **Limits:** the code runs in a blink and times nothing.

## 🔴 4. Deep / formal

**How wide is the receptive field?** For layers $1,\dots,L$ with kernel sizes $k_l$ and strides $s_l$, the width of the receptive field of one number in the last layer is:

$$r_L=1+\sum_{l=1}^{L}(k_l-1)\prod_{m=1}^{l-1}s_m$$

**Where:**
- $L$ is the number of layers stacked (convolutions and pooling count as layers here). **Matters:** every layer can only add to the reach, so deeper means bigger.
- $k_l$ is the window size of layer $l$ (pooling has a window too). **Matters:** a layer adds $k_l-1$ new neighbours on top of what came before (the "$-1$" is because the window's own centre is already counted).
- $s_m$ is the stride of layer $m$. **Matters:** the product of earlier strides is how many original pixels one step of layer $l$ covers (called its *jump*). A layer after a stride-2 step therefore adds twice as much reach as the same layer before it would have.
- The $1$ at the front is the single pixel under the output number's own centre. **Matters:** with no layers at all, a number sees exactly itself.
- $r_L$ is the width of the receptive field in pixels of the original picture. **Matters:** it tells you how much of the picture the network can use at that depth. The area is $r_L^2$ pixels, and the code's pixel counts (such as 121 for $r=11$) confirm that.

**Example (tiny numbers):** convolution 3×3 (stride 1), pooling 2×2 (stride 2), convolution 3×3 (stride 1).
1. Layer 1: $k=3$, no earlier strides (the product is 1), adds $(3-1)\cdot1=2$, so $r=1+2=3$.
2. Layer 2 (pooling): $k=2$, earlier strides give 1, adds $(2-1)\cdot1=1$, so $r=4$.
3. Layer 3: $k=3$, earlier strides give $1\cdot2=2$, adds $(3-1)\cdot2=4$, so $r=8$.
4. The code measured 8 with PyTorch's gradients, and the formula agrees.
5. Meaning: the pooling layer in the middle doubled the contribution of the last convolution. Without it, three 3×3 convolutions reach only 7. For the five-layer version the formula gave 18, and the measurement gave 18.

**Other points (from memory, not checked here):**
- The *theoretical* receptive field above is a square. Some researchers describe an *effective* receptive field, which is the part that really matters to the output and is said to be smaller and roughly bell-shaped (from memory, I did not test this, and my all-ones weights cannot show it).
- Dilated convolutions grow the receptive field without shrinking the map, and the word is also used in biology, where it was borrowed from, for the area of the retina that changes a neuron's response. I did not test either.
- Vision [[Transformer]]s can in principle look at the whole picture from the first layer (with [[Self-Attention]]), which is one of the reasons people compare them with CNNs. I did not check any comparison.
- Tasks such as recognising a whole object need a receptive field at least as big as the object, and networks that label every pixel (segmentation) need a big one too (from memory, not tested).

**What came before, and how this improved on it:**
- **Before:** one big kernel (5×5, 7×7 or larger) or a dense layer that sees the whole picture at once, to get a wide view in a single step. AlexNet's first layer used a large kernel (from memory; I did not print its size).
- **Its limit:** a big kernel holds many numbers ($k^2C^2$ per layer) and is a single linear step, with no activation in between to build complicated patterns.
- **What changed:** stack small 3×3 layers (with an [[Activation Function]] between them) and let the reach grow layer by layer. The popular VGG networks of 2014 are, from memory, the best-known example, and I did not check the details.
- **How much better:** the same reach with fewer numbers, as the code counted for 8 channels: two 3×3 layers reach 5 with 1,152 numbers against 1,600 for one 5×5 (28% fewer), and three 3×3 layers reach 7 with 1,728 against 3,136 for one 7×7 (45% fewer). A pooling step in the middle makes the same three convolutions reach 8 or 18 pixels instead of 7.

## Related
[[Convolutional Neural Network]] · [[Convolution]] · [[Pooling]] · [[Activation Function]] · [[Neural Network]] · [[Transformer]] · [[Self-Attention]] · [[PyTorch]] · [[Backpropagation]] · [[Feature Map]] · [[Sliding Window Attention]]
- [[VGG]]

## Sources
- The owner's course slides "Convolutional.pdf" read as text. The slides do not discuss receptive fields, so this note is an extension of that topic, not a summary of a slide.
- Library versions in this run: Python 3.9.6, NumPy 1.24.4, PyTorch 2.8.0 (CPU). The code and every printed value were run for this note. Nothing was timed.
- No figure: I searched Wikimedia Commons for "receptive field" and found only biology pictures, so I skipped the image.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision Receptive field:::How much of the original picture one number in a layer can "see". Like a gossip chain: after more rounds, news from further away has reached you.

#flashcards/ml/formula #flashcards/topic/vision Write the receptive field formula and explain each symbol.
?
$$r_L=1+\sum_{l=1}^{L}(k_l-1)\prod_{m=1}^{l-1}s_m$$
- $L$: the number of layers stacked (convolutions and pooling both count). Every layer can only add to the reach, so deeper means bigger.
- $k_l$: the window size of layer $l$. A layer adds $k_l-1$ new neighbours on top of what came before (the window's own centre is already counted).
- $s_m$: the stride of layer $m$. The product of earlier strides is how many original pixels one step of layer $l$ covers (its jump). A layer after a stride-2 step adds twice as much reach.
- The 1 at the front: the single pixel under the output number's own centre.
- $r_L$: the width of the receptive field in original pixels. The area is $r_L^2$.

#flashcards/ml/cloze #flashcards/topic/vision A layer after a stride-2 step adds ==more== reach, because the earlier strides multiply its ==jump==.

#flashcards/ml/example #flashcards/topic/vision Convolution 3×3 (stride 1), pooling 2×2 (stride 2), convolution 3×3 (stride 1). What is the receptive field?::Layer 1: adds (3 − 1)·1 = 2, so r = 3. Layer 2 (pooling): adds (2 − 1)·1 = 1, so r = 4. Layer 3: earlier strides give 2, adds (3 − 1)·2 = 4, so r = 8.
