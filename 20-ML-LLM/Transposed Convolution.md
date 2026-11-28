---
tags: [transposed-convolution, deconvolution, upsampling, convolution, computer-vision, segmentation, ML]
status: growing
level: 4
tier: 2
---

# Transposed Convolution

A **transposed convolution** (often wrongly called "deconvolution") is a layer that makes a feature map **bigger**, using weights that are learned. A normal [[Convolution]] with a stride shrinks a map. A transposed convolution goes the other way: each input value paints a copy of the kernel onto a larger output, and where the copies overlap they are added.

## 🟢 1. Simple (age 12)
Imagine a small grid of numbers and a rubber stamp. For each number you press the stamp on a bigger sheet, with the stamp's ink as strong as that number, and you move a few squares between presses. Where two stamps touch the same square, the ink adds up. That is how a small map becomes a big one. The stamp's pattern is learned, so the network can learn a good way to enlarge a picture.

**Example:** a 2 by 2 grid of numbers and a 3 by 3 stamp pressed with a step of 2 gives a 5 by 5 sheet.

## 🔵 2. How it works
1. Start with a small [[Feature Map]], for instance 7 by 7 cells.
2. Pick a **kernel** (a small grid of learned weights, say 4 by 4) and a **stride** (say 2).
3. For each input cell, multiply the whole kernel by that cell's value and place the result on the output. Neighbouring input cells place their copies **stride** cells apart.
4. Where copies overlap, **add** them.
5. The output is about stride times larger: for stride 2, a 7 by 7 map becomes 14 by 14 (with the padding shown below).
6. Because the kernel is learned, the network can learn the enlargement that gives the best result. A fixed starting kernel such as bilinear interpolation gives a sensible enlargement before any training ([[FCN 2014]]).
7. **Used for:** bringing a coarse map back to pixel size in segmentation ([[FCN 2014]], [[U-Net 2015]], the mask branch of [[Mask R-CNN 2017]]) and in image generators ([[GAN 2014]]).

![[transposed-convolution-stride2.gif]]
*A transposed convolution with stride 2 and no padding: the blue cells are the input (spread apart with empty cells between them), a 3 by 3 kernel window slides over them, and the cyan map is the larger output. Authors: Vincent Dumoulin and Francesco Visin, MIT licence, from "A guide to convolution arithmetic for deep learning", [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Convolution_arithmetic_-_No_padding_strides_transposed.gif). It is an animation; I looked at three of its 25 frames and checked that the output is a 5 by 5 map, as the size rule in my run gives for a 2 by 2 input.*

## 🟠 3. Technical detail
Numbers come from `transposed_demo.py` (Python 3.9.6, PyTorch 2.8.0, CPU).

```python
"""Transposed convolution: output size, the 'gradient of a convolution' view, bilinear upsampling, uneven overlap."""
import torch
import torch.nn.functional as F

# 1. output size = (in - 1) * stride - 2 * padding + kernel
for n, k, s, p in ((3, 4, 2, 1), (7, 4, 2, 1), (16, 64, 32, 16)):
    m = torch.nn.ConvTranspose2d(1, 1, k, stride=s, padding=p)
    print("in", n, "kernel", k, "stride", s, "pad", p, "->", m(torch.zeros(1, 1, n, n)).shape[-1], "formula", (n - 1) * s - 2 * p + k)

# 2. a transposed convolution is the backward pass of the convolution with the same weights
w = torch.randn(1, 1, 4, 4)
x = torch.randn(1, 1, 8, 8, requires_grad=True)
y = F.conv2d(x, w, stride=2, padding=1)                  # 8x8 -> 4x4
up = torch.randn_like(y)
g, = torch.autograd.grad(y, x, grad_outputs=up)           # gradient of conv w.r.t. its input
print("conv 8->%d, then transposed back to %d; equals the conv's gradient:" % (y.shape[-1], g.shape[-1]),
      torch.allclose(g, F.conv_transpose2d(up, w, stride=2, padding=1), atol=1e-5))

# 3. a 4x4 'bilinear' kernel for 2x upsampling (the start point FCN describes) against F.interpolate
def bilinear_kernel(k):
    f = (k + 1) // 2; c = f - 1 if k % 2 == 1 else f - 0.5
    og = torch.arange(k, dtype=torch.float32)
    t = 1 - (og - c).abs() / f
    return t[:, None] * t[None, :]
m = torch.nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, bias=False)
m.weight.data = bilinear_kernel(4).reshape(1, 1, 4, 4)
small = torch.arange(1.0, 17.0).reshape(1, 1, 4, 4)
a, b = m(small), F.interpolate(small, scale_factor=2, mode="bilinear", align_corners=False)
d = (a - b).abs()
print("2x bilinear kernel vs F.interpolate: interior max difference", round(d[0, 0, 1:-1, 1:-1].max().item(), 6),
      "| border max difference", round(d.max().item(), 3))

# 4. uneven overlap: kernel 3, stride 2 on an all-ones 4x4 input with an all-ones kernel counts how many windows hit each output cell
ones = torch.ones(1, 1, 4, 4)
o = F.conv_transpose2d(ones, torch.ones(1, 1, 3, 3), stride=2)
print(o[0, 0].int())
o4 = F.conv_transpose2d(ones, torch.ones(1, 1, 4, 4), stride=2)
print("kernel 4, stride 2 (kernel divisible by stride): interior values", sorted(set(o4[0, 0, 2:-2, 2:-2].flatten().int().tolist())))
```

I saw:
```
in 3 kernel 4 stride 2 pad 1 -> 6 formula 6
in 7 kernel 4 stride 2 pad 1 -> 14 formula 14
in 16 kernel 64 stride 32 pad 16 -> 512 formula 512
conv 8->4, then transposed back to 8; equals the conv's gradient: True
2x bilinear kernel vs F.interpolate: interior max difference 0.0 | border max difference 7.0
tensor([[1, 1, 2, 1, 2, 1, 2, 1, 1],
        [1, 1, 2, 1, 2, 1, 2, 1, 1],
        [2, 2, 4, 2, 4, 2, 4, 2, 2],
        [1, 1, 2, 1, 2, 1, 2, 1, 1],
        [2, 2, 4, 2, 4, 2, 4, 2, 2],
        [1, 1, 2, 1, 2, 1, 2, 1, 1],
        [2, 2, 4, 2, 4, 2, 4, 2, 2],
        [1, 1, 2, 1, 2, 1, 2, 1, 1],
        [1, 1, 2, 1, 2, 1, 2, 1, 1]], dtype=torch.int32)
kernel 4, stride 2 (kernel divisible by stride): interior values [4]
```

**What the run shows.**
- **Output size:** in all three cases it matched the rule *(input − 1) × stride − 2 × padding + kernel*: 3 becomes 6, 7 becomes 14, and 16 becomes 512 with a 64 by 64 kernel, stride 32 and padding 16 (an enlargement by 32, the factor of the FCN paper's coarsest net; the kernel size is my choice, not read from the paper).
- **It is the backward pass of a convolution.** A stride-2 convolution turned an 8 by 8 map into 4 by 4. Feeding the 4 by 4 error map through the transposed convolution with the same weights gave exactly the convolution's gradient with respect to its input (`True`), so "transposed" means the same wiring run backwards. It does not undo the convolution's values.
- **A bilinear start works:** a 4 by 4 kernel with bilinear weights and stride 2 gave the same result as `F.interpolate(..., mode="bilinear")` in the interior (difference 0.0), and differed at the border by up to 7.0, because the transposed layer adds zeros beyond the edge while the interpolator repeats the edge values.
- **Uneven overlap:** with a 3 by 3 kernel and stride 2, the number of copies that land on each output cell is 1, 2 or 4, in a regular pattern (the printed grid). With a 4 by 4 kernel and stride 2 every interior cell gets 4. Uneven overlap is the cause of a "checkerboard" pattern in the output (general knowledge, which I did not measure on a trained image).

**Summary: what the run shows.** A transposed convolution is a normal strided convolution's backward wiring, its size follows a simple rule, and with a bilinear kernel it reproduces ordinary bilinear enlargement. Caution: the checks use random or hand-built weights, so they say nothing about image quality after training, and my note on the checkerboard effect is from the overlap count only.

## 🔴 4. Deep / formal
No equation: the size rule is trivial arithmetic (given above in words) and the layer is "stamp the kernel, add the overlaps", which the run reproduces.

**Names.** "Deconvolution" in the FCN paper means "backwards strided convolution" (upsampling by a factor $f$ is convolution with a fractional input stride of $1/f$, paper section 3.3, as recorded in [[FCN 2014]]). It is not the mathematical deconvolution that undoes a blur. "Transposed convolution" or "fractionally strided convolution" are the clearer names.

**Edge cases and details:**
- **Choosing the kernel and stride:** a kernel size that is a multiple of the stride (4 and 2 in my run) gives even overlap in the interior. A size that is not (3 and 2) gives uneven overlap.
- **A stride-1 transposed convolution** is just a convolution with a bigger padding, and enlarges by only the kernel size minus one.
- **Alternatives:** enlarge with fixed interpolation first (nearest or bilinear) and then apply an ordinary convolution; many people prefer this to avoid the checkerboard (general knowledge, which I did not test).
- **In FCN:** the final upsampling layer is fixed at bilinear and the intermediate ones start bilinear and are learned (paper section 4.3, as recorded in [[FCN 2014]]).

**Limits and trade-offs:**
- It cannot invent detail that the small map lost. It places and blends learned patterns.
- Uneven overlap can leave a regular grid pattern in the output.
- It has as many weights as a convolution with the same kernel.

**What came before, and how this improved on it:**
- **Before:** enlarge a map with a fixed rule (nearest, bilinear), or with a post-processing step outside the network.
- **Its limit:** the fixed rule cannot learn. FCN's authors wanted the pixelwise loss to train the upsampling by backpropagation ([[FCN 2014]]).
- **What changed:** a learnable layer whose forward pass is the backward pass of a strided convolution, started from bilinear weights.
- **How much better:** I did not run a comparison of learned and fixed upsampling. The FCN paper reports that fine-tuning all layers gave 59.4 mean IU against 45.4 when only the last layer was trained (Table 2 as recorded in [[FCN 2014]]); that compares fine-tuning, not upsampling, so it is not a measure of this layer. The size and gradient checks above are exact.

## Related
[[Convolution]] · [[FCN 2014]] · [[U-Net 2015]] · [[Mask R-CNN 2017]] · [[GAN 2014]] · [[Image Segmentation]] · [[Feature Map]] · [[Convolutional Neural Network]] · [[Feature Pyramid Network]] · [[Pooling]]

## Sources
- My own run: `transposed_demo.py` (Python 3.9.6, PyTorch 2.8.0).
- Image: Vincent Dumoulin and Francesco Visin, MIT licence, Wikimedia Commons (link under the picture).
- The FCN statements come from [[FCN 2014]] as read there; I did not re-read the paper for this note. The checkerboard effect and the advice on interpolation first are general knowledge and not tested. The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision Transposed convolution:::A learnable layer that enlarges a feature map: each input value paints a scaled copy of the kernel onto a bigger output, spaced by the stride, and overlaps are added.

#flashcards/ml/concept #flashcards/topic/vision Why is "deconvolution" a misleading name?::The layer does not undo a convolution or recover the lost values; it runs the same wiring backwards to enlarge a map. "Transposed" or "fractionally strided" convolution is clearer (the FCN paper itself says "deconvolution").

#flashcards/ml/concept #flashcards/topic/vision What is the output size of a transposed convolution?::(input - 1) x stride - 2 x padding + kernel. My runs: 3 gives 6 and 7 gives 14 (kernel 4, stride 2, padding 1), and 16 gives 512 (kernel 64, stride 32, padding 16).

#flashcards/ml/concept #flashcards/topic/vision How is a transposed convolution related to an ordinary strided convolution?::It is the convolution's backward pass: feeding an error map through it with the same weights gave exactly the convolution's gradient with respect to its input (True in my run, 8 by 8 to 4 by 4 and back).

#flashcards/ml/concept #flashcards/topic/vision What did FCN use as the starting weights of its upsampling layers?::Bilinear interpolation. The final layer stays fixed at bilinear and the intermediate ones are learned (FCN paper, as recorded in my note). In my run a bilinear 4 by 4 kernel with stride 2 matched F.interpolate in the interior (difference 0.0) and differed at the border.

#flashcards/ml/concept #flashcards/topic/vision Why can a transposed convolution create a checkerboard pattern?::Output cells receive unequal numbers of overlapping kernel copies. With kernel 3 and stride 2 the counts were 1, 2 and 4 in my run; with kernel 4 and stride 2 every interior cell got 4. The visible effect on trained images is general knowledge, not tested by me.

#flashcards/ml/example #flashcards/topic/vision A 2 by 2 input, a 3 by 3 kernel, stride 2, no padding: how big is the output?::(2 - 1) x 2 + 3 = 5, so 5 by 5, which is what the animation of this case shows (my arithmetic).

#flashcards/ml/image #flashcards/topic/vision ![[transposed-convolution-stride2.gif]]
What does the animation show?
?
A transposed convolution with stride 2 and no padding: the input cells (blue, spread apart) are covered by a sliding 3 by 3 kernel window and the cyan output is larger (5 by 5). Authors Dumoulin and Visin, MIT licence; I looked at three of 25 frames.

#flashcards/ml/cloze #flashcards/topic/vision A transposed convolution ==enlarges== a map: each input value adds a scaled copy of the ==kernel== to the output, and overlaps are ==added==.
