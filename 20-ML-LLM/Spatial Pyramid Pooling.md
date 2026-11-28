---
tags: [spatial-pyramid-pooling, SPP-net, pooling, object-detection, computer-vision, CNN, ML]
status: growing
level: 4
tier: 2
---

# Spatial Pyramid Pooling

**Spatial pyramid pooling (SPP)** is a pooling layer that turns a feature map of **any size** into a vector of **one fixed length**. It pools the map several times with grids of different coarseness (for example 6 × 6, 3 × 3, 2 × 2 and 1 × 1 cells), and joins the results. Because the grid always has the same number of cells, the output length does not depend on the picture size. It lets a network take pictures of any size, and lets a detector read the features of any box from one shared feature map. [[RoI Pooling]] is the one-level special case.

## 🟢 1. Simple (age 12)
Imagine a photo of any size that you must describe on a form with a fixed number of boxes. You use a few "zoom levels". Level 1: one box for the whole photo ("the strongest thing anywhere in it"). Level 2: split the photo into 4 parts and note the strongest thing in each. Level 3: split into 9 parts, and so on. Big photo or small photo, you always fill the same number of boxes, so the form is always the same length.

**Example:** a wide photo and a tall photo both become a list of 50 strong-feature notes per feature type (1 + 4 + 9 + 36), even though their shapes differ.

## 🔵 2. How it works
1. A [[Convolutional Neural Network]] has convolution layers (which accept any input size and give a [[Feature Map]] of a matching size) and fully connected layers (which need a fixed-length input). The fixed size requirement comes only from the fully connected layers.
2. Put the SPP layer **between** the last convolution layer and the first fully connected layer.
3. Pick a **pyramid**: a list of grid sizes, for example 6 × 6, 3 × 3, 2 × 2 and 1 × 1.
4. For each grid size $n\times n$, cut the feature map into $n\times n$ cells (the cells grow with the picture) and take the maximum of each cell, **for each channel separately** ([[Pooling]]).
5. **Join** all the results into one long vector. With $k$ channels and $M$ cells in total the length is $kM$, always.
6. Feed the vector to the fully connected layers. The picture size is now free.
7. **For detection,** run the convolution layers once on the whole picture, then apply SPP to the part of the map under each proposal box ([[Selective Search]] boxes in the paper). Every box gives the same length vector, and the network ran only once. ([[Fast R-CNN 2015]] then used a single-level version, the RoI pooling layer.)

No image: the paper's figures are not under an open licence and I found no free picture of the layer. The numbers below come from a real run.

## 🟠 3. Technical detail
Numbers come from `spp_demo.py` (Python 3.9.6, PyTorch 2.8.0, CPU). It builds SPP from `adaptive_max_pool2d` with the paper's four-level pyramid, checks the output length for different map sizes and for windows cut out of one map, tests the paper's window and stride rule for the cell sizes, and recomputes the paper's speed-ups.

```python
"""Spatial pyramid pooling: fixed-length output for any input size; the paper's window/stride rule against torch's adaptive pooling."""
import math, torch
import torch.nn.functional as F
torch.manual_seed(0)
levels = (6, 3, 2, 1)                                   # the pyramid used for the results tables in the paper (50 bins)
def spp(x, levels=levels):
    return torch.cat([F.adaptive_max_pool2d(x, n).flatten(1) for n in levels], dim=1)
print("bins:", sum(n*n for n in levels), "| output length for 256 channels:", 256*sum(n*n for n in levels), "(paper: 12,800)")
for h, w in ((13, 13), (10, 10), (13, 20), (24, 9), (6, 6)):
    x = torch.randn(1, 256, h, w); print("feature map", (h, w), "->", tuple(spp(x).shape))
# the paper's rule: window = ceil(a/n), stride = floor(a/n), for an a x a map; how many bins per side does it really give?
print("\nbins per side from the paper's rule (win = ceil(a/n), stride = floor(a/n)); X marks a mismatch")
print(" a  | n=6   n=3   n=2   n=1")
for a in (6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 20):
    row = []
    for n in (6, 3, 2, 1):
        win, st = math.ceil(a/n), math.floor(a/n); out = (a - win)//st + 1
        row.append("%d%s" % (out, "" if out == n else " X"))
    print("%3d | %s" % (a, "  ".join("%-4s" % r for r in row)))
print("paper's Figure 4 example, a = 13:", [(n, math.ceil(13/n), math.floor(13/n)) for n in (3, 2, 1)], "(sizeX, stride): 5/4, 7/6, 13/13 as printed")
# torch adaptive pooling: bin i covers floor(i*a/n) .. ceil((i+1)*a/n)
a, n = 10, 6
print("\nadaptive bins for a = 10, n = 6:", [(math.floor(i*a/n), math.ceil((i+1)*a/n)) for i in range(n)])
# hand check on a small map
x = torch.arange(1., 26.).reshape(1, 1, 5, 5)
print("5x5 map 1..25: level 2x2 ->", F.adaptive_max_pool2d(x, 2).flatten().tolist(), " level 1x1 ->", F.adaptive_max_pool2d(x, 1).flatten().tolist())
# same length for different scales, and detection window pooled from a shared map
fm = torch.randn(1, 256, 38, 50)                           # one conv5 map for a whole image
for (r0, c0, r1, c1) in ((2, 3, 20, 30), (10, 10, 14, 13), (0, 0, 38, 50)):
    win = fm[:, :, r0:r1, c0:c1]; print("window", (r1-r0, c1-c0), "->", tuple(spp(win).shape))
# parameters of fc6 for the paper's two pyramids
for lv in ((6, 3, 2, 1), (4, 3, 2, 1)):
    m = sum(n*n for n in lv); print("pyramid", lv, "bins", m, "fc6 weights (256*bins x 4096):", 256*m*4096)
print("baseline pool5 6x6:", 256*36*4096)
# timing arithmetic from Tables 9 and 10
print("speed-ups: 9.03/0.142 = %.0f, 9.03/0.382 = %.0f, 14.46/0.142 = %.0f, 14.46/0.382 = %.0f" % (9.03/.142, 9.03/.382, 14.46/.142, 14.46/.382))
print("no SPP -> SPP top-1 gains:", [round(a_ - b_, 2) for a_, b_ in ((35.99, 34.98), (34.93, 34.38), (34.13, 32.87), (32.01, 30.36))], "| multi-size:", [round(a_ - b_, 2) for a_, b_ in ((35.99, 34.60), (34.93, 33.94), (34.13, 32.26), (32.01, 29.68))])
```

I saw:
```
bins: 50 | output length for 256 channels: 12800 (paper: 12,800)
feature map (13, 13) -> (1, 12800)
feature map (10, 10) -> (1, 12800)
feature map (13, 20) -> (1, 12800)
feature map (24, 9) -> (1, 12800)
feature map (6, 6) -> (1, 12800)

bins per side from the paper's rule (win = ceil(a/n), stride = floor(a/n)); X marks a mismatch
 a  | n=6   n=3   n=2   n=1
  6 | 6     3     2     1   
  7 | 6     3     2     1   
  8 | 7 X   3     2     1   
  9 | 8 X   3     2     1   
 10 | 9 X   3     2     1   
 11 | 10 X  3     2     1   
 12 | 6     3     2     1   
 13 | 6     3     2     1   
 14 | 6     3     2     1   
 16 | 7 X   3     2     1   
 20 | 6     3     2     1   
paper's Figure 4 example, a = 13: [(3, 5, 4), (2, 7, 6), (1, 13, 13)] (sizeX, stride): 5/4, 7/6, 13/13 as printed

adaptive bins for a = 10, n = 6: [(0, 2), (1, 4), (3, 5), (5, 7), (6, 9), (8, 10)]
5x5 map 1..25: level 2x2 -> [13.0, 15.0, 23.0, 25.0]  level 1x1 -> [25.0]
window (18, 27) -> (1, 12800)
window (4, 3) -> (1, 12800)
window (38, 50) -> (1, 12800)
pyramid (6, 3, 2, 1) bins 50 fc6 weights (256*bins x 4096): 52428800
pyramid (4, 3, 2, 1) bins 30 fc6 weights (256*bins x 4096): 31457280
baseline pool5 6x6: 37748736
speed-ups: 9.03/0.142 = 64, 9.03/0.382 = 24, 14.46/0.142 = 102, 14.46/0.382 = 38
no SPP -> SPP top-1 gains: [1.01, 0.55, 1.26, 1.65] | multi-size: [1.39, 0.99, 1.87, 2.33]
```

**What the run shows.**
- **Fixed length:** maps of 13 × 13, 10 × 10, 13 × 20, 24 × 9 and 6 × 6 cells (256 channels) all gave 12,800 numbers: 256 channels × 50 cells (1 + 4 + 9 + 36), the paper's "12,800-d (256×50)". Windows of 18 × 27, 4 × 3 and 38 × 50 cells cut from one 38 × 50 map also gave 12,800 (the detection case).
- **Hand check:** on a 5 × 5 map of the numbers 1 to 25, the 2 × 2 level gave 13, 15, 23, 25 (the maximum of each quarter, with overlapping quarters) and the 1 × 1 level gave 25.
- **First layer weights:** the 50-cell pyramid makes the first fully connected layer 256 × 50 × 4096 = 52,428,800 weights; the 30-cell pyramid (4, 3, 2, 1) has 31,457,280; the plain 6 × 6 pooling it replaces has 37,748,736 (my arithmetic).
- **The paper's window rule against the real bin count.** The paper implements each level as a sliding-window pooling of window $\lceil a/n\rceil$ and stride $\lfloor a/n\rfloor$ on an $a\times a$ map. My table of the bins per side this rule really gives:

| Map side $a$ | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 16 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bins per side for $n=6$ | 6 | 6 | 7 | 8 | 9 | 10 | 6 | 6 | 6 | 7 | 6 |

- For $n=3$, $2$ and $1$ the rule gave exactly 3, 2 and 1 for every $a$ I tried. For $n=6$ it fails for $a=8, 9, 10, 11$ and $16$. The paper's own example ($a=13$) works (window 5, stride 4 for $n=3$; window 7, stride 6 for $n=2$; window 13 for $n=1$). But its 180 × 180 network has a 10 × 10 last convolution map, and for $n=6$ the stated rule gives a 9 × 9 grid, not 6 × 6. So the real implementation must differ from the rule as written (I did not check their code); PyTorch's adaptive pooling (bin $i$ covers $\lfloor ia/n\rfloor$ to $\lceil (i+1)a/n\rceil$) always gives exactly $n$ bins, with overlapping cells (for $a=10$, $n=6$: 0 to 2, 1 to 4, 3 to 5, 5 to 7, 6 to 9, 8 to 10).
- **Timing arithmetic.** The paper's total times per image of 9.03 s (R-CNN, AlexNet-style network), 0.142 s (SPP, one scale) and 0.382 s (SPP, five scales) give 64 and 24 times (9.03/0.142 and 9.03/0.382, mine). The same-network comparison, 14.46 s for R-CNN, gives 102 and 38 times.

**Summary: what the run shows.** The layer does what it claims: a fixed 12,800 numbers for any input shape or any window of a shared map, and the first fully connected layer grows only with the number of cells, not the picture. Caution: I built SPP with adaptive pooling, not with the paper's window and stride rule, because the rule as written does not give the right bin count for some sizes; the paper's trained numbers were not reproduced here.

## 🔴 4. Deep / formal
No equation: the layer is "pool each of a few fixed grids of a map and join them", and its only formulas are cell sizes (the window rule above), which are trivial arithmetic.

**What the paper says.** I read the text of arXiv 1406.4729v4 (He, Zhang, Ren and Sun, 14 pages): the abstract, the introduction, section 2 (the layer and training) and the results tables I quote; the parts I did not read in full are listed under Sources. Key statements:
- Existing CNNs need a fixed input (for example 224 × 224), which is "artificial" and forces cropping or warping, losing content or distorting shape.
- The fixed size "comes only from the fully-connected layers"; the convolution layers "can generate feature maps of any sizes".
- SPP's properties: a fixed-length output "regardless of the input size", multi-level bins (robust to deformations), and the possibility to pool features "extracted at variable scales".
- For detection the convolutions run once on the whole image and SPP pools each window; the system is "24-102× faster than the R-CNN method, while achieving better or comparable accuracy on Pascal VOC 2007" and with EdgeBoxes proposals takes about 0.5 seconds per image.
- **Training with several sizes:** two fixed-size networks that share all weights (224 × 224 and 180 × 180 inputs), each trained for a whole epoch in turn; the convergence rate is "similar to the above single-size training".

**Classification results (paper's Table 2, top-1 error on ImageNet 2012 validation, 10-view testing).** Written by me, an AI, from the paper's table text; the top-5 columns are left out.

| Network | No SPP | SPP, single-size training | SPP, multi-size training |
|---|---|---|---|
| ZF-5 | 35.99 | 34.98 | 34.60 |
| Convnet*-5 | 34.93 | 34.38 | 33.94 |
| Overfeat-5 | 34.13 | 32.87 | 32.26 |
| Overfeat-7 | 32.01 | 30.36 | 29.68 |

- **Columns:** the network (five or seven convolution layers; ZF-5, Convnet*-5 and Overfeat variants), then the top-1 error with the plain last pooling layer, with a 4-level SPP (6 × 6, 3 × 3, 2 × 2, 1 × 1) trained at one size, and with SPP trained at two sizes. Lower is better.

**Summary: what the table shows.** SPP lowers the error for all four networks, by 0.55 to 1.65 points with single-size training and 0.99 to 2.33 with multi-size training (my arithmetic from the printed numbers); the biggest gain is for the most accurate network (Overfeat-7: 32.01 to 29.68). A control: a 30-cell pyramid (4, 3, 2, 1) with fewer weights than the baseline gave 35.06 top-1, close to the 50-cell result (34.98) and better than the baseline (35.99), so the paper reads the gain as coming from multi-level pooling, not from extra parameters. Caution: single runs, no error bars, and a 10-view test that used the same crops as the baseline.

**Detection (paper's Tables 9 and 10, VOC 2007).** For the ZF-5 network: SPP with one scale reaches 58.0 mAP with box regression in 0.142 s per image on the GPU; with five scales 59.2 in 0.382 s; R-CNN with the AlexNet-style network reached 58.5 in 9.03 s (the paper's numbers; the speed-ups of 64 and 24 times are in the check above).

**Edge cases and details:**
- **Single-level SPP:** the RoI pooling layer of [[Fast R-CNN 2015]], which the Fast R-CNN paper calls "the special-case of the spatial pyramid pooling layer used in SPPnets in which there is only one pyramid level"; with it, the back-propagation through the pooling is simple and the convolution layers can be trained.
- **Global pooling:** the 1 × 1 level is a global pooling over the whole map (the paper notes that concurrent work used global average pooling).
- **Max, not average:** the paper uses max pooling in each cell.
- **Mapping a box to the feature map:** the paper's appendix aligns a box's corners to feature-map positions (floor on the left and top, ceiling on the right and bottom, when rounding is needed); this rounding is what later work ([[RoIAlign]]) revisited.

**Limits and trade-offs:**
- SPPnet's fine-tuning could not update the convolution layers before the pooling layer, which Fast R-CNN names as a limit for very deep networks (Fast R-CNN paper, section 1).
- Training was still a multi-stage pipeline with features written to disk (Fast R-CNN paper, section 1).
- A 50-cell pyramid enlarges the first fully connected layer (52.4 million weights against 37.7 million, my arithmetic).

**What came before, and how this improved on it:**
- **Before:** a fixed-size network input, reached by cropping or warping (and, for detection, warping thousands of boxes each into its own forward pass as in [[R-CNN 2013]]); and in non-neural pipelines, spatial pyramid matching over bag-of-words features.
- **Its limit:** crops lose content, warping distorts, and detection costs one network pass per box.
- **What changed:** pool a pyramid of grids on the last convolution map, so the input size is free and one forward pass serves all boxes.
- **How much better:** classification top-1 error lowered by 0.55 to 2.33 points across four networks (paper, Table 2; differences mine); detection features 24 to 102 times faster than R-CNN with better or comparable mAP (paper, abstract and Tables 9 and 10). In my own run the output length was a constant 12,800 for five different map shapes. I did not train a network.

## Related
[[Pooling]] · [[RoI Pooling]] · [[RoIAlign]] · [[Fast R-CNN 2015]] · [[R-CNN 2013]] · [[Selective Search]] · [[Feature Map]] · [[Convolutional Neural Network]] · [[Receptive Field]] · [[Object Detection]] · [[Image Classification]]

## Sources
- My own run: `spp_demo.py` (Python 3.9.6, PyTorch 2.8.0).
- Paper: He, Zhang, Ren and Sun, "Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition", arXiv 1406.4729 v4 (submitted 18 Jun 2014, this version 23 Apr 2015, per the arXiv listing). Read as text: abstract, introduction, section 2 on the layer and the training, the passages on the pyramid and Tables 2, 3, 7, 9, 10 and 12 by text search, and the appendix paragraph on mapping a window. Not read: the details of the detection method, the ILSVRC 2014 sections, most of the experiments on VOC 2007 and Caltech101, the figures as pictures.
- The Fast R-CNN statements come from [[Fast R-CNN 2015]] as read there. The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision Spatial pyramid pooling (SPP):::A pooling layer that max-pools a feature map with several grids (for example 6 by 6, 3 by 3, 2 by 2, 1 by 1) and joins the results, giving a fixed-length vector for any input size or box.

#flashcards/ml/concept #flashcards/topic/vision Why do CNNs need a fixed input size, and how does SPP remove the need?::Only the fully connected layers need a fixed-length input; the convolution layers accept any size. SPP sits after the last convolution layer and pools a fixed number of cells whatever the map size (SPP paper, section 1 and 2).

#flashcards/ml/concept #flashcards/topic/vision What are the steps of an SPP layer?::Choose a pyramid of grid sizes; for each n by n grid cut the map into n by n cells (they scale with the map) and take the max of each cell per channel; join all outputs into one vector of length k x M (k channels, M cells).

#flashcards/ml/example #flashcards/topic/vision With 256 channels and the pyramid 6 by 6, 3 by 3, 2 by 2, 1 by 1, how long is the SPP output?::256 x (36 + 9 + 4 + 1) = 256 x 50 = 12,800 numbers for any input size (the paper's 12,800-d; my run printed 12,800 for five map shapes).

#flashcards/ml/concept #flashcards/topic/vision How does SPP speed up detection compared with R-CNN?::Run the convolution layers once on the whole image, then pool the map region under each proposal into the same fixed-length vector; the paper reports 24 to 102 times faster feature computation with better or comparable VOC 2007 mAP (abstract, Tables 9 and 10).

#flashcards/ml/concept #flashcards/topic/vision How is RoI pooling related to SPP?::RoI pooling is SPP with a single pyramid level (Fast R-CNN paper, section 2.1); that makes its backward pass simple so the convolution layers can be trained, which SPPnet's fine-tuning could not do.

#flashcards/ml/concept #flashcards/topic/vision How much did SPP help classification?::Top-1 error on ImageNet 2012 fell from 35.99 to 34.98 (single-size training) and 34.60 (multi-size) for ZF-5, and from 32.01 to 30.36 and 29.68 for Overfeat-7; a 30-cell pyramid with fewer weights still gave 35.06, so the gain comes from multi-level pooling (paper, Table 2).

#flashcards/ml/concept #flashcards/topic/vision Where does the SPP paper's window and stride rule fail?::The rule window = ceil(a/n), stride = floor(a/n) on an a by a map gives the right number of bins for n = 3, 2, 1 but not for n = 6 at several sizes (a = 10 gives 9 by 9 bins, not 6 by 6), so the real implementation must differ; adaptive pooling always gives n bins (my run; paper, section 2.3).

#flashcards/ml/concept #flashcards/topic/vision What are the limits of SPPnet?::Fine-tuning could not update the convolution layers before the pooling, training was still multi-stage with features written to disk, and a 50-cell pyramid enlarges the first fully connected layer (52.4 million weights against 37.7 million, my arithmetic) (Fast R-CNN paper, section 1).

#flashcards/ml/cloze #flashcards/topic/vision SPP pools a ==pyramid== of grids on the last convolution map and joins the results into a ==fixed-length== vector, so the ==input size== is free.
