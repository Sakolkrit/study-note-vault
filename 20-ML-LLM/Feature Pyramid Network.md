---
tags: [feature-pyramid-network, fpn, object-detection, computer-vision, multi-scale, mask-rcnn, ML]
status: growing
level: 4
tier: 3
---

# Feature Pyramid Network

A **Feature Pyramid Network (FPN)** adds a second, top-down path to a [[Convolutional Neural Network]] so that every size of feature map, from fine to coarse, becomes both detailed and meaningful. A [[Convolutional Neural Network]] makes small, abstract maps deep inside and large, detailed maps early on. The FPN passes the abstract meaning back down and mixes it into the detailed maps, so a detector can find small objects on a fine map and large objects on a coarse one, all with strong features.

## 🟢 1. Simple (age 12)
Imagine looking at a city from space (you see the big shapes and know "this is a city", but not the houses), then zooming in step by step (you see houses, but you forget which part of town is which). A pyramid network keeps a note from the high view as you zoom in: "this block is the harbour", so even at house level you know where you are. Then you can spot a tiny boat and a huge ship, each at the zoom level that suits it, and still know they are in a harbour.

**Example:** a photo has a tiny bird in the corner and a big bus in the middle. The bird is found on the zoomed-in map, the bus on the zoomed-out map, and both maps carry the same understanding of "street scene".

## 🔵 2. How it works
1. A [[Convolutional Neural Network]] (the **backbone**, such as [[ResNet 2015]]) shrinks the image in stages. Each stage gives a [[Feature Map]] that is half the size of the one before: fine and weak early, coarse and strong late.
2. **Bottom-up path:** this is just the backbone. Pick one map per stage, for example three of them.
3. **Lateral connections:** squeeze each chosen map to the same number of channels with a 1 by 1 [[Convolution]].
4. **Top-down path:** start from the coarsest map. Enlarge it to twice its size (by repeating cells), add the next finer map's squeezed copy, and do it again for the next level.
5. **Smooth each result** with a 3 by 3 convolution, which reduces the blockiness left by enlarging (general knowledge). The outputs are the pyramid levels, all with the same channel count.
6. **Use:** a detector runs on every level. A small box is read from a fine level, a large box from a coarse level (a rule picks the level from the box size). [[Anchor Box]] sizes can be tied to levels as well.

No image: I found no free picture of an FPN on Wikimedia Commons, and the paper's figures are not free. The numbers below come from a real run instead.

## 🟠 3. Technical detail
Numbers come from `fpn_demo.py` (Python 3.9.6, PyTorch 2.8.0, torchvision 0.23.0, CPU). It uses torchvision's `FeaturePyramidNetwork` with random input maps, so it checks shapes and wiring, not detection quality.

```python
"""Feature Pyramid Network: shapes, parameter count, and a check of the top-down step by hand (torchvision 0.23)."""
import math, torch
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.ops import FeaturePyramidNetwork

torch.manual_seed(0)
fpn = FeaturePyramidNetwork([64, 128, 256], 256).eval()
x = OrderedDict(c3=torch.randn(1, 64, 64, 64), c4=torch.randn(1, 128, 32, 32), c5=torch.randn(1, 256, 16, 16))
out = fpn(x)
print({k: tuple(v.shape) for k, v in out.items()})
print("parameters:", sum(p.numel() for p in fpn.parameters()))

# by hand: p5 = layer(lateral(c5)); p4 = layer(lateral(c4) + upsample(lateral(c5)))
with torch.no_grad():
    lat = [fpn.inner_blocks[i](v) for i, v in enumerate(x.values())]
    p5 = fpn.layer_blocks[2](lat[2])
    m4 = lat[1] + F.interpolate(lat[2], size=lat[1].shape[-2:], mode="nearest")
    p4 = fpn.layer_blocks[1](m4)
print("p5 matches:", torch.allclose(p5, out["c5"], atol=1e-5), "| p4 matches:", torch.allclose(p4, out["c4"], atol=1e-5))

# which level a region is read from (rule from the FPN paper as I remember it, not checked): k = floor(4 + log2(sqrt(w*h) / 224)), kept in 2..5
for side in (56, 112, 224, 448, 900):
    print("region", side, "x", side, "-> level", min(5, max(2, math.floor(4 + math.log2(side / 224)))))
```

I saw:
```
{'c3': (1, 256, 64, 64), 'c4': (1, 256, 32, 32), 'c5': (1, 256, 16, 16)}
parameters: 1885696
p5 matches: True | p4 matches: True
region 56 x 56 -> level 2
region 112 x 112 -> level 3
region 224 x 224 -> level 4
region 448 x 448 -> level 5
region 900 x 900 -> level 5
```

**What the run shows.**
- **Shapes:** three backbone maps of 64, 128 and 256 channels (64 by 64, 32 by 32 and 16 by 16 cells) all come out with 256 channels at the same three sizes. Only the channel count changes, and the sizes stay.
- **Size of the add-on:** 1,885,696 parameters (my count): three 1 by 1 lateral layers (115,456 in all) and three 3 by 3 smoothing layers (1,770,240 in all, my arithmetic).
- **The wiring matches my description.** I recomputed the coarsest output (just the lateral then the 3 by 3 convolution) and the next one (lateral plus the enlarged coarser map, then the 3 by 3) by hand, and both matched the library's output.
- **Which level a region is read from.** The rule as I remember it from the FPN paper (not checked) is $k=\lfloor 4+\log_2(\sqrt{wh}/224)\rfloor$, kept between levels 2 and 5, with 224 the size of an ImageNet image. My run gave: 56 pixels to level 2, 112 to level 3, 224 to level 4, 448 and 900 to level 5. Small regions go to fine maps, big regions to coarse maps.

**Summary: what the run shows.** The FPN is a small, regular addition: one 1 by 1 and one 3 by 3 convolution per level, and one enlarge-and-add per level. Caution: random inputs say nothing about accuracy, and I used the three-level form of torchvision's block, not the paper's exact five-level setup.

## 🔴 4. Deep / formal
No equation: the merge is "enlarge the coarser result, add the squeezed finer map, smooth with a 3 by 3 convolution", which the steps and the run above reproduce exactly. The level-assignment rule is quoted from memory.

**What the Mask R-CNN paper reports (as recorded in [[Mask R-CNN 2017]]):** on COCO instance segmentation, ResNet-101 with an FPN gets 35.7 mask AP against 33.1 for the same backbone with the plain C4 features, 2.6 points more (the difference is my arithmetic); the paper says FPN "gives excellent gains in both accuracy and speed". I did not re-read the paper for this note, and the original FPN paper was not read at all.

**Why top-down.** Early maps are fine but see little context. Late maps see a lot but are coarse. A plain bottom-up detector must choose: use the late map and miss small objects, or use an early map and have weak features. The top-down path with lateral links gives early maps the late maps' meaning.

**Edge cases and details:**
- **Enlarging by repetition** (nearest neighbour) is cheap and the 3 by 3 smoothing convolution hides its blocky look; learned upsampling ([[Transposed Convolution]]) is another way, not used in my demo.
- **Same channel count at every level** lets the same detector head be shared across levels.
- **Addition, not concatenation:** the merged map keeps the same size as the lateral map, so the cost does not grow.

**Limits and trade-offs:**
- It adds memory and compute for several large maps, particularly the finest.
- The rule that picks a level from the box size is a hand-set choice (the 224 and the level range are from memory here).
- It improves how features are used; it does not change what the backbone can recognise.

**What came before, and how this improved on it:**
- **Before:** the Faster R-CNN paper lists three ways to handle several sizes: pyramids of resized images, pyramids of filters, or its anchors (as recorded in [[Faster R-CNN 2015]]); detectors then used one backbone map, often the last (the Mask R-CNN paper's "C4" choice).
- **Its limit:** an image pyramid runs the whole network many times; a single map must trade detail against meaning.
- **What changed:** get a multi-scale set of strong maps from one pass of the network by adding a top-down path with lateral links.
- **How much better:** in the Mask R-CNN paper, 35.7 against 33.1 mask AP (ResNet-101, FPN against C4, difference mine). I did not measure this myself.

## Related
[[Mask R-CNN 2017]] · [[Faster R-CNN 2015]] · [[Convolutional Neural Network]] · [[Feature Map]] · [[Convolution]] · [[Anchor Box]] · [[RoIAlign]] · [[ResNet 2015]] · [[Object Detection]] · [[Transposed Convolution]]

## Sources
- My own run: `fpn_demo.py` (Python 3.9.6, torchvision 0.23.0), including a hand check of two outputs.
- The Mask R-CNN numbers and quotation are from [[Mask R-CNN 2017]] as read there, and the multi-scale approaches from [[Faster R-CNN 2015]]. The original FPN paper was not read; the level-assignment rule and its constants are from memory. The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision Feature Pyramid Network (FPN):::A top-down path with lateral connections added to a CNN backbone so that feature maps of every size are both detailed and carry the deep layers' meaning; detectors use one level per object size.

#flashcards/ml/concept #flashcards/topic/vision What problem does an FPN solve?::Early feature maps are fine but weak, late maps are strong but coarse, so a detector using one map must trade small-object detail against meaning. An FPN gives all sizes strong features from one pass of the network.

#flashcards/ml/concept #flashcards/topic/vision What are the steps that build an FPN?::Pick one backbone map per stage; squeeze each to the same channels with a 1 by 1 convolution (lateral); starting from the coarsest, enlarge it, add the next finer lateral map, repeat; smooth each sum with a 3 by 3 convolution.

#flashcards/ml/concept #flashcards/topic/vision Why is a 3 by 3 convolution applied after each merge?::To remove the blocky pattern left by enlarging the coarser map by repetition.

#flashcards/ml/example #flashcards/topic/vision In my torchvision run with maps of 64, 128 and 256 channels, what came out and how big was the add-on?::All three levels had 256 channels at the same sizes (64, 32 and 16 cells wide), with 1,885,696 parameters: 115,456 in the 1 by 1 layers and 1,770,240 in the 3 by 3 layers (my count). Random inputs, wiring check only.

#flashcards/ml/concept #flashcards/topic/vision Which FPN level is a region read from?::Small regions from fine levels, large ones from coarse levels. The rule I remember (from memory, not checked): level = floor(4 + log2(sqrt(w x h) / 224)), kept in 2 to 5. My run: 56 to level 2, 112 to 3, 224 to 4, 448 and 900 to 5.

#flashcards/ml/concept #flashcards/topic/vision How much did FPN help in the Mask R-CNN paper?::ResNet-101 with FPN got 35.7 mask AP against 33.1 with plain C4 features, 2.6 points more (as recorded in my Mask R-CNN note; the difference is my arithmetic).

#flashcards/ml/concept #flashcards/topic/vision Why add the maps instead of stacking them?::Addition keeps the merged map at the lateral map's channel count and size, so the cost does not grow and the same detector head can be shared across levels.

#flashcards/ml/concept #flashcards/topic/vision What does an FPN not change?::What the backbone can recognise; it only improves how its features are used across sizes, and it costs extra memory and compute for the large fine maps.

#flashcards/ml/cloze #flashcards/topic/vision An FPN has a ==bottom-up== path (the backbone), a ==top-down== path, and ==lateral== connections that add them together.
