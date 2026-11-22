---
tags: [computer-vision, deep-learning, CNN, ImageNet]
status: growing
level: 4
tier: 2
---

# VGG

**VGG** is a family of image-recognition networks built from a simple, repeating recipe: stacks of small 3 by 3 filters, going 16 to 19 layers deep. This note is the concept; the paper is in [[VGG 2014]].

## 🟢 1. Simple (age 12)
A VGG network looks at a picture through many layers of tiny magnifying glasses, all the same small size. One layer finds edges, the next finds corners made of edges, the next finds shapes made of corners, and so on. The surprise was that simply stacking more of the same tiny layers made the network recognise pictures better.

**Example:** Building a tall tower out of identical small bricks. Nobody needed a special big brick. Just more of the same small one, stacked higher.

## 🔵 2. How it works
1. The network reads a picture and passes it through layers of [[Convolution]] filters. Each filter is a small grid of weights that slides over the picture and responds to a pattern.
2. In VGG, every filter is the same tiny size, 3 by 3.
3. The filters come in blocks. After each block, a [[Pooling]] step shrinks the picture to half its width and height, while the number of filters grows (64, 128, 256, 512).
4. Two or three small filters in a row see as much of the picture as one big filter does (a 3 by 3 followed by a 3 by 3 sees a 5 by 5 patch).
5. But the small ones use fewer weights, and each one is followed by an [[Activation Function]], so the network can learn more complex shapes.
6. After the blocks come three dense layers, which turn the final pattern into a label ([[Image Classification]]).
7. **The catch:** VGG is very heavy. Most of its weights sit in the dense layers, and it is slow, so later networks such as [[ResNet]] reached better results with far fewer weights.


(No image: I found no free, on-topic picture on Wikimedia Commons for this note.)

## 🟠 3. Technical detail
- **Recipe:** only 3 by 3 convolutions ([[Convolution]]), with stride 1 and padding so the size is kept, and 2 by 2 max pooling to halve the size ([[Pooling]]), then three dense layers at the end (these details are from memory; the weight count below agrees with them). The paper's two best models have 16 and 19 weight layers (abstract: "16-19 weight layers").
- **The abstract's claim:** pushing the depth to 16 to 19 layers with very small 3 by 3 filters gives a significant improvement on the prior-art configurations. The team took first place in localisation and second in classification at ILSVRC 2014, and the two best models were released for others to reuse.
- **Weight count of VGG-16:** 138,357,544, of which 89% are in the three dense layers (counted in the code below, equal to the PyTorch model's count).
- **Use afterwards:** a popular feature extractor for [[Transfer Learning]], and the backbone for early detectors such as Faster R-CNN ([[Object Detection]]) and the segmentation networks in [[Image Segmentation]].
- **Limits:** the size and cost, and plain stacks get hard to train much deeper than about 20 layers, which [[Residual Connection]] fixed in [[ResNet]].
- **Family tree:** [[LeNet and AlexNet]] came before, then VGG, then [[ResNet]] and later the [[Vision Transformer]] ([[History of Image Classification]]).

```python
import torch, torchvision
C = 64
print(f"channels in = out = {C}:  one 5x5 conv: {25 * C * C:,} weights; two 3x3 convs (same 5x5 view): {2 * 9 * C * C:,}  ({1 - 18 / 25:.0%} fewer)")
print(f"                        one 7x7 conv: {49 * C * C:,} weights; three 3x3 convs (same 7x7 view): {3 * 9 * C * C:,}  ({1 - 27 / 49:.0%} fewer, plus 3 non-linearities instead of 1)")
cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]    # the 16-layer layout: 13 convolutions + 3 dense layers
cin, params, convs = 3, 0, 0
for c in cfg:
    if c != "M": params += 9 * cin * c + c; cin = c; convs += 1
dense = [(512 * 7 * 7, 4096), (4096, 4096), (4096, 1000)]
fc = sum(i * o + o for i, o in dense)
print(f"counted from the layout: {convs} convolutions with {params:,} weights, 3 dense layers with {fc:,}; total {params + fc:,}; dense share {fc / (params + fc):.0%}")
print("torchvision vgg16 parameter count:", f"{sum(p.numel() for p in torchvision.models.vgg16(weights=None).parameters()):,}")
```

## 🔴 4. Deep / formal
**Weights in a convolution layer.** A layer with $k\times k$ filters, $C_{\text{in}}$ input channels and $C_{\text{out}}$ filters has

$$
\text{params}=k^{2}\,C_{\text{in}}\,C_{\text{out}}+C_{\text{out}}
$$

**Where:**
- $k$ is the filter size (3 for VGG). $C_{\text{in}}$ is the number of channels coming in (for example 64 feature maps) and $C_{\text{out}}$ the number of filters.
- Each filter has $k\times k$ weights for each of the $C_{\text{in}}$ input channels. The final $+C_{\text{out}}$ is one bias per filter.
- **Matters:** the count grows with $k^2$, so a 5 by 5 filter costs 25 weights per channel pair and a 3 by 3 only 9. Without this, depth would be unaffordable.

**Example (tiny numbers):** a 3 by 3 layer from 64 to 64 channels.
- $9\cdot64\cdot64+64=36{,}864+64=\mathbf{36{,}928}$ weights.
- Meaning: one such layer already holds nearly 37 thousand weights, and VGG-16 stacks 13 convolution layers like it, growing to 512 channels.

**What a stack of small filters sees.** The patch of the picture that one output pixel depends on (the [[Receptive Field]]) after $n$ stacked 3 by 3 layers, stride 1, is a square of side

$$
R=2n+1
$$

**Where:**
- $n$ is the number of 3 by 3 convolution layers stacked in a row, and $R$ is the side of the patch in the original picture, in pixels.
- Each extra 3 by 3 layer adds one pixel on each side of the patch, so 2 more in total.
- **Matters:** this is why two 3 by 3 layers can stand in for one 5 by 5, and three for one 7 by 7, while using fewer weights and adding more non-linearities.

**Example (tiny numbers):** channels in = out = 64.
- $n=2$: $R=\mathbf5$. Weights: $2\cdot9\cdot64^2=\mathbf{73{,}728}$ against $25\cdot64^2=\mathbf{102{,}400}$ for one 5 by 5. That is 28% fewer.
- $n=3$: $R=\mathbf7$. Weights: $3\cdot9\cdot64^2=\mathbf{110{,}592}$ against $49\cdot64^2=\mathbf{200{,}704}$ for one 7 by 7. That is 45% fewer, and 3 activations instead of 1.

**Further notes:**
- **Where the weights go:** the first dense layer alone has $7\cdot7\cdot512\cdot4096\approx102.8$ million weights, which is most of the model (counted in the code).
- **Cost in time:** convolutions do most of the arithmetic even though the dense layers hold most of the weights (the same split as in [[LeNet and AlexNet]]).

**What came before, and how this improved on it:**
- **Before:** [[LeNet and AlexNet]]: 8 layers, with large first filters (11 by 11 and 5 by 5 in AlexNet; from memory).
- **Its limit:** big filters cost many weights per layer and are a coarse way to add depth, and it was not clear how much depth alone would help.
- **What changed:** use only 3 by 3 filters and push the depth to 16 to 19 weight layers (Simonyan and Zisserman, arXiv 1409.1556, 4 September 2014, abstract read; first authors from memory).
- **How much better:** the abstract reports a significant improvement over prior-art configurations and 1st and 2nd place at ILSVRC 2014 (localisation and classification); the exact error rates are not in the abstract and I did not read the body. In the code below, two 3 by 3 layers use 28% fewer weights than one 5 by 5 with the same view, and three use 45% fewer than one 7 by 7. **Honest cost:** VGG-16 has 138,357,544 weights, 89% of them in the dense layers, so it is far heavier than later networks.

**Check it yourself:** run the code in level 3 above. It prints every number quoted in this block.

## Related
[[Convolution]] · [[Convolutional Neural Network]] · [[Pooling]] · [[Receptive Field]] · [[Activation Function]] · [[Image Classification]] · [[LeNet and AlexNet]] · [[ResNet]] · [[VGG 2014]] · [[Transfer Learning]] · [[Object Detection]] · [[Image Segmentation]] · [[History of Image Classification]] · [[MobileNet]]
## Sources
- Simonyan and Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition*, arXiv 1409.1556 (2014): abstract read; first authors from memory.
- The weight counts were computed from the layer layout and agree with `torchvision.models.vgg16(weights=None)` (checked in the code).
- No image: I found no free, on-topic picture on Wikimedia Commons, so I added none.
- Every number in the history block was computed by the code in this note (toy data); dates and names marked "from memory" are not checked.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision VGG:::A network that looks at a picture through many layers of tiny same-size filters (3 by 3). Stacking more of the same tiny layers made the network better.

#flashcards/ml/formula #flashcards/topic/vision Write the parameters of a convolution layer and explain each symbol.
?
$$\text{params}=k^{2}\,C_{\text{in}}\,C_{\text{out}}+C_{\text{out}}$$
- $k$: the filter size (3 for VGG). $C_{\text{in}}$: the number of channels coming in (for example 64 feature maps). $C_{\text{out}}$: the number of filters.
- Each filter has $k\times k$ weights for each of the $C_{\text{in}}$ input channels. The final $+C_{\text{out}}$ is one bias per filter.
- The count grows with $k^2$: a 5 by 5 filter costs 25 weights per channel pair and a 3 by 3 only 9. Without this, depth would be unaffordable.

#flashcards/ml/example #flashcards/topic/vision A 3 by 3 layer from 64 to 64 channels. How many weights?::$9\cdot64\cdot64+64=36{,}864+64=36{,}928$. VGG-16 stacks 13 convolution layers like it, growing to 512 channels.

#flashcards/ml/formula #flashcards/topic/vision Write the patch size seen by $n$ stacked 3 by 3 layers and explain each symbol.
?
$$R=2n+1$$
- $n$: the number of 3 by 3 convolution layers stacked in a row.
- $R$: the side of the patch in the original picture, in pixels.
- Each extra 3 by 3 layer adds one pixel on each side, so 2 in total.
- So two 3 by 3 layers can stand in for one 5 by 5, and three for one 7 by 7, with fewer weights and more non-linearities.

#flashcards/ml/example #flashcards/topic/vision Channels in = out = 64. Compare two 3 by 3 layers with one 5 by 5, and three with one 7 by 7.::$n=2$: $R=5$. Weights $2\cdot9\cdot64^2=73{,}728$ against $25\cdot64^2=102{,}400$, which is 28% fewer. $n=3$: $R=7$. Weights $3\cdot9\cdot64^2=110{,}592$ against $49\cdot64^2=200{,}704$, 45% fewer, and 3 activations instead of 1.
