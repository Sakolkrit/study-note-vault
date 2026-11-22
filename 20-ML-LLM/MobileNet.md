---
tags: [computer-vision, CNN, efficient-models, mobile, depthwise-convolution, deep-learning]
status: growing
level: 4
tier: 2
---

# MobileNet

**MobileNet** is a family of small, fast [[Convolutional Neural Network]]s built for phones and other weak hardware. Its trick is to split each convolution into two cheap steps, so the network needs far fewer numbers and far less arithmetic. It is the backbone of the detector in [[Tiny Object Detector]].

## 🟢 1. Simple (age 12)
A normal convolution is like a cook who, for every dish, looks at every ingredient on the table at once. MobileNet splits the job: first one helper looks at **each ingredient on its own**, then a second helper **mixes** the results. The dish comes out nearly as good, but the work is a fraction of the size.

**Example:** you want to turn a photo into a "cartoon" version. The heavy way makes every cartoon filter look at red, green and blue together. The light way first tidies each colour channel alone, then blends the three. It does less work for a similar look.

## 🔵 2. How it works
1. A normal [[Convolution]] layer has many filters. Each filter is a small window (say 3×3) that slides over the picture and looks at **all** the input channels at once, so each filter has 3 × 3 × (number of channels) weights.
2. MobileNet replaces it with two layers:
   - **Depthwise:** one 3×3 filter for **each input channel**, and each looks only at its own channel. It finds shapes inside a channel but does not mix channels.
   - **Pointwise:** a 1×1 convolution that mixes the channels at every position. It builds the new channels.
3. The two layers together do about the same kind of job as the normal layer, with far fewer weights (a count is in the code below).
4. Stack many of these layers, with [[Batch Normalization]] and an activation after them (from memory), and you have a network small enough for a phone.
5. **Two simple knobs** (from the paper's abstract: "two simple global hyper-parameters") let you trade accuracy for speed: make every layer thinner, or feed smaller pictures (their names, width and resolution multiplier, are from memory).
6. **Later versions:** MobileNetV2 added "inverted residuals" and "linear bottlenecks" (from its abstract); MobileNetV3 used an automatic search for a good layout (from its abstract).

*I found no free, on-topic picture of MobileNet on Wikimedia Commons (the searches returned generic convolution diagrams and unrelated files), so this note has no picture.*

## 🟠 3. Technical detail
- **Where the saving comes from.** A normal layer with $M$ channels in, $N$ out and a $K\times K$ window has $K^2MN$ weights. Splitting it gives $K^2M$ (depthwise) plus $MN$ (pointwise).
- **Several versions.** V1 uses the split above. V2 wraps it in a residual block that first widens the channels, filters, then narrows again ("inverted residual"). V3 adds a searched layout (from the abstracts, checked).
- **Fewer weights does not mean proportionally faster.** Speed also depends on memory traffic and on how well the hardware handles depthwise layers (from general knowledge, not measured here). The timings below are on my laptop CPU, which was also busy with other programs.

The code counts the weights of one layer both ways, then of whole networks (random weights; only counting and timing), then the parts of the detector of [[Tiny Object Detector]]:

```python
# file: mobilenet_toy.py
import time, torch, torch.nn as nn
from torchvision.models import mobilenet_v3_large, resnet50, vgg16
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn

torch.set_num_threads(4)
def n_params(m): return sum(p.numel() for p in m.parameters())

# one layer: 128 channels in, 128 out, 3x3 kernel, no bias
M, N, K = 128, 128, 3
standard = nn.Conv2d(M, N, K, padding=1, bias=False)
depthwise = nn.Conv2d(M, M, K, padding=1, groups=M, bias=False)      # one 3x3 filter per input channel
pointwise = nn.Conv2d(M, N, 1, bias=False)                           # 1x1 conv mixes the channels
a, b, c = n_params(standard), n_params(depthwise), n_params(pointwise)
print(f"standard 3x3 conv: {a:,} weights;  depthwise {b:,} + pointwise {c:,} = {b + c:,} weights")
print(f"ratio {(b + c) / a:.4f}   formula 1/N + 1/K^2 = {1 / N + 1 / K**2:.4f}   ({a / (b + c):.1f} times fewer)")
x = torch.randn(1, M, 56, 56)
print("same output shape:", tuple(standard(x).shape) == tuple(pointwise(depthwise(x)).shape))

# whole networks (random weights: only counting and timing)
nets = {"MobileNetV3-Large": mobilenet_v3_large(), "ResNet-50": resnet50(), "VGG-16": vgg16()}
for name, m in nets.items(): print(f"{name:18s} {n_params(m) / 1e6:6.2f} million weights")
img = torch.randn(1, 3, 320, 320)
for name in ("MobileNetV3-Large", "ResNet-50"):
    m = nets[name].eval(); ts = []
    with torch.no_grad():
        for k in range(12):
            t = time.time(); m(img); ts.append(time.time() - t)
    print(f"{name:18s} one 320x320 photo on this CPU: median {sorted(ts[2:])[5] * 1000:.0f} ms")

# the detector of Tiny Object Detector
det = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None, num_classes=4)
parts = {"MobileNetV3 body": det.backbone.body, "feature pyramid": det.backbone.fpn, "box proposal part": det.rpn, "box classifier part": det.roi_heads}
print(f"detector in total {n_params(det) / 1e6:.2f} million weights:  " + ", ".join(f"{k} {n_params(v) / 1e6:.2f}" for k, v in parts.items()))
```

```text
standard 3x3 conv: 147,456 weights;  depthwise 1,152 + pointwise 16,384 = 17,536 weights
ratio 0.1189   formula 1/N + 1/K^2 = 0.1189   (8.4 times fewer)
same output shape: True
MobileNetV3-Large    5.48 million weights
ResNet-50           25.56 million weights
VGG-16             138.36 million weights
MobileNetV3-Large  one 320x320 photo on this CPU: median 45 ms
ResNet-50          one 320x320 photo on this CPU: median 182 ms
detector in total 18.96 million weights:  MobileNetV3 body 2.97, feature pyramid 1.47, box proposal part 0.61, box classifier part 13.92
```

Reading it:
- One layer with 128 channels in and out: 147,456 weights as a normal convolution, 17,536 as depthwise plus pointwise, **8.4 times fewer**, exactly what the formula below predicts.
- Whole networks: MobileNetV3-Large 5.48 million weights, ResNet-50 25.56 million, VGG-16 138.36 million (my counts).
- On one 320×320 photo my CPU took a median of about 45 ms for MobileNetV3-Large and 182 ms for ResNet-50 (4 threads, other programs running; the numbers move from run to run).
- In the detector, the MobileNetV3 body is only 2.97 million of the 18.96 million weights, because the box classifier part, a pair of large fully connected layers, has 13.92 million. A small backbone does not make the whole detector small. (The 2.97 million is the 5.48 million of the full network minus its 2.51 million ImageNet classifier, my arithmetic, which matches the printed counts.)

## 🔴 4. Deep / formal
Compare the cost of a normal layer with the split one, counted as multiply-adds for one output position (the same ratio holds for the number of weights):

$$
\frac{K^2M+MN}{K^2MN}=\frac{1}{N}+\frac{1}{K^2}
$$

**Where:**
- $K$ is the width of the filter window (3 for a 3×3 filter).
- $M$ is the number of input channels and $N$ the number of output channels.
- $K^2M$ is the depthwise step (one $K\times K$ filter per input channel) and $MN$ the pointwise step (a 1×1 filter from every input channel to every output channel). $K^2MN$ is the normal layer.
- **Matters:** when $N$ is large, $1/N$ is tiny and the ratio is about $1/K^2$, so a 3×3 layer is almost 9 times cheaper. Without the split, every output channel needs a full $K\times K$ filter over every input channel.

**Example (tiny numbers):** $K=3$, $M=N=128$.
- Normal layer: $9\times128\times128=\mathbf{147{,}456}$.
- Split layer: $9\times128+128\times128=1{,}152+16{,}384=\mathbf{17{,}536}$.
- Ratio: $17{,}536/147{,}456=\mathbf{0.1189}$, and $1/128+1/9=0.0078+0.1111=0.1189$. About 8.4 times fewer.
- Meaning: the saving is almost all from the 3×3 window, and the pointwise step is what is left.

**Limits:**
- **Less capacity.** Fewer weights means a smaller model; on hard tasks it is usually less accurate than a bigger network (I did not measure accuracy on ImageNet here).
- **The saving is in arithmetic and weights, not always in time.** My timings show about 4 times faster than ResNet-50 for about 4.7 times fewer weights, one machine and a noisy CPU.
- **It still needs a good training recipe.** Small networks are easy to under-train; V2 and V3 changed the block layout for this reason (from the abstracts, the reason is from memory).

**What came before, and how this improved on it:**
- **Before:** networks built from normal convolutions, such as VGG-16 and ResNet-50 ([[VGG]], [[ResNet]]).
- **Its limit:** too many weights and too much arithmetic for phones and embedded devices.
- **What changed:** MobileNets (arXiv 1704.04861, 17 April 2017, checked on arXiv) "use depth-wise separable convolutions to build light weight deep neural networks" and add two global hyper-parameters to trade latency against accuracy. MobileNetV2 (arXiv 1801.04381, 13 January 2018) added inverted residuals and linear bottlenecks. MobileNetV3 (arXiv 1905.02244, 6 May 2019) combined hardware-aware architecture search with the NetAdapt algorithm; its abstract states that MobileNetV3-Large is 3.2% more accurate on ImageNet classification than MobileNetV2 with 15% less latency, and that its COCO detection is 25% faster at roughly the same accuracy (the paper's own figures, which I did not reproduce).
- **How much better:** in my own counts, one layer needs 8.4 times fewer weights; the whole network 5.48 million against 25.56 million (ResNet-50) and 138.36 million (VGG-16); and a 320×320 photo took about 45 ms against 182 ms on my CPU. I did not measure accuracy.

## Related
- [[Convolution]] · [[Convolutional Neural Network]] · [[Feature Map]] · [[ResNet]] · [[VGG]] · [[Batch Normalization]] · [[Transfer Learning]] · [[Quantization]] · [[Knowledge Distillation]] · [[Object Detection]] · [[Faster R-CNN 2015]] · [[Tiny Object Detector]]

## Sources
- arXiv abstracts read: Howard et al., *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*, arXiv 1704.04861 (17 April 2017); Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, arXiv 1801.04381 (13 January 2018); Howard et al., *Searching for MobileNetV3*, arXiv 1905.02244 (6 May 2019). Only the abstracts; I did not read the papers. Every code block in this note was extracted into a fresh folder (with copies of the files it needs from the earlier projects) and run again; each output matched the block above. Not tested: another machine or other library versions.
- My own code and runs (Python 3.9.6, PyTorch 2.8.0 CPU, torchvision 0.23.0). The cost formula is my own derivation, checked by the count in the code; that the papers show the same formula, the names width and resolution multiplier, the use of batch normalization, and the speed remarks are from memory.
- No image (see level 2).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision MobileNet:::A family of small, fast convolutional networks for phones and embedded devices, built from depthwise separable convolutions (a per-channel filter followed by a 1×1 mixing layer).

#flashcards/ml/formula #flashcards/topic/vision Write the cost ratio of a depthwise separable layer to a normal convolution and explain each symbol.
?
$$\frac{K^2M+MN}{K^2MN}=\frac{1}{N}+\frac{1}{K^2}$$
- $K$: width of the filter window. $M$, $N$: input and output channels.
- $K^2M$: the depthwise step; $MN$: the pointwise 1×1 step; $K^2MN$: the normal layer.
- For large $N$ the ratio is about $1/K^2$, so a 3×3 layer is almost 9 times cheaper.

#flashcards/ml/example #flashcards/topic/vision With K = 3 and M = N = 128, how many weights does a normal layer and a depthwise separable layer have?::Normal: 9 × 128 × 128 = 147,456. Separable: 9 × 128 + 128 × 128 = 1,152 + 16,384 = 17,536. Ratio 0.1189 (= 1/128 + 1/9), about 8.4 times fewer (checked in the code).

#flashcards/ml/concept #flashcards/topic/vision What are the two steps of a depthwise separable convolution?
?
Depthwise: one small filter (say 3×3) per input channel, each looking only at its own channel. Pointwise: a 1×1 convolution that mixes the channels at every position and builds the new channels.

#flashcards/ml/concept #flashcards/topic/vision How big are MobileNetV3-Large, ResNet-50 and VGG-16 in weights?
?
5.48 million, 25.56 million and 138.36 million (my counts with random weights). On one 320×320 photo my busy CPU took a median of about 45 ms for MobileNetV3-Large and 182 ms for ResNet-50 (4 threads; it moves from run to run).

#flashcards/ml/concept #flashcards/topic/vision Is the Tiny Object Detector small because its backbone is MobileNet?
?
Only partly. Of its 18.96 million weights the MobileNetV3 body has 2.97 million, the feature pyramid 1.47, the box proposal part 0.61 and the box classifier part 13.92. The large fully connected box classifier dominates (my count).

#flashcards/ml/concept #flashcards/topic/vision What did MobileNet V1, V2 and V3 add (from their arXiv abstracts)?
?
V1 (17 April 2017): depth-wise separable convolutions plus two global hyper-parameters trading latency for accuracy. V2 (13 January 2018): inverted residuals and linear bottlenecks. V3 (6 May 2019): hardware-aware architecture search with NetAdapt; abstract claims Large is 3.2% more accurate than V2 on ImageNet at 15% less latency (the paper's figure, not reproduced here).

#flashcards/ml/concept #flashcards/topic/vision What are the limits of MobileNet?
?
A smaller model has less capacity, so on hard tasks it is usually less accurate (accuracy not measured here); fewer weights does not always mean proportionally less time (about 4 times faster for 4.7 times fewer weights on my CPU, one noisy machine); and small networks need a good training recipe.

#flashcards/ml/cloze #flashcards/topic/vision A depthwise separable convolution uses a ==depthwise== filter for each channel and then a ==pointwise== convolution to mix the channels.
