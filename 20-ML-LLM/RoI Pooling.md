---
tags: [roi-pooling, object-detection, computer-vision, fast-rcnn, faster-rcnn, pooling, ML]
status: growing
level: 4
tier: 2
---

# RoI Pooling

**RoI pooling** (region-of-interest pooling) turns a region of any size, cut out of a [[Feature Map]], into a small grid of a fixed size, for example 7 by 7. It does this by dividing the region into equal-looking cells and keeping the largest value of each cell ([[Pooling]]). The fixed size lets one classifier handle every region, however big or small.

## 🟢 1. Simple (age 12)
Imagine cutting rectangles of different sizes out of a big photo and wanting to turn each one into a tiny 2 by 2 picture that sums it up. You split each rectangle into four parts and, for each part, you keep only its brightest spot. A small rectangle and a huge rectangle both end up as 2 by 2 pictures of the same size, so the same judge can look at all of them.

**Example:** a rectangle around a dog and a rectangle around a bus have different sizes. After this trick both are 2 by 2 grids of numbers, and the same classifier can look at both.

## 🔵 2. How it works
1. A [[Convolutional Neural Network]] computes a [[Feature Map]] **once** for the whole image. Each cell of the map stands for a patch of the image (a stride of 16 means 16 image pixels per cell).
2. A proposal method (Selective Search, or the RPN of [[Faster R-CNN 2015]]) gives candidate boxes in image pixels.
3. **Map each box onto the feature map:** divide its corners by the stride and round to whole cells.
4. **Cut the box into a fixed grid** (say 7 by 7 bins). The bins are as equal as whole cells allow, so some bins are one cell wider than others.
5. **Keep the maximum** of each bin. The result is a fixed 7 by 7 block for every box, however large.
6. Feed each block to the classifier and the box-adjustment layer of the detector. Because the map was computed once, 2,000 boxes cost one pass through the convolutions, not 2,000.
7. **The catch:** the rounding in steps 3 and 4 moves the region a little, by up to about half a cell (8 pixels at stride 16). Classification does not mind, but a pixel-accurate mask does, which is why [[RoIAlign]] was invented ([[Mask R-CNN 2017]]).

![[roi-pooling-animated.gif]]
*Region-of-interest pooling on an 8 by 8 input: a 5 by 7 region is cut into 2 by 2 sections of unequal size, and the maximum of each section (red) is kept, giving a 2 by 2 output. Author: Tomasz Grel (deepsense.io), CC BY 4.0, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:RoI_pooling_animated.gif). I looked at two of the five frames; the figure is an animation, and in the frames I saw the sections are 3 or 4 cells wide and 2 or 3 tall, with maxima 0.85, 0.84, 0.97 and 0.96.*

## 🟠 3. Technical detail
Numbers come from `roi_demo.py` (Python 3.9.6, PyTorch 2.8.0, torchvision 0.23.0, CPU). The feature map is 8 by 8 and each cell holds `10*row + column`, so every answer can be checked by eye. The script also runs [[RoIAlign]] for comparison.

```python
"""RoI pooling against RoIAlign on a tiny feature map whose value at (x, y) is 10*y + x, so the right answers are known."""
import torch
from torchvision.ops import roi_pool, roi_align

H = W = 8
ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
feat = (10 * ys + xs).reshape(1, 1, H, W)          # cell (row y, col x) holds 10*y + x
print(feat[0, 0].int())

# one region in image pixels; the map has stride 16, so the region is x 20..100, y 24..88 on the map (1.25 .. 6.25, 1.5 .. 5.5)
box = torch.tensor([[0.0, 20.0, 24.0, 100.0, 88.0]])      # (batch index, x1, y1, x2, y2)
sc = 1 / 16
print("RoIPool  2x2:", roi_pool(feat, box, output_size=2, spatial_scale=sc).squeeze().tolist())
print("RoIAlign 2x2:", [round(v, 3) for v in roi_align(feat, box, output_size=2, spatial_scale=sc, sampling_ratio=2, aligned=True).flatten().tolist()])

# slide the region by 4 pixels (a quarter of a feature cell) and see which output moves
print("shift (pixels) | RoIPool top-left bin | RoIAlign top-left bin")
for d in (0, 4, 8, 12, 16):
    b = box.clone(); b[0, 1] += d; b[0, 3] += d
    p = roi_pool(feat, b, output_size=2, spatial_scale=sc)[0, 0, 0, 0].item()
    a = roi_align(feat, b, output_size=2, spatial_scale=sc, sampling_ratio=2, aligned=True)[0, 0, 0, 0].item()
    print("%2d | %5.1f | %6.3f" % (d, p, a))

# one sampling point by hand: value at x = 2.5, y = 4.3 by bilinear interpolation from the 4 nearest cells
x, y = 2.5, 4.3
x0, y0 = int(x), int(y); fx, fy = x - x0, y - y0
v = lambda r, c: 10 * r + c
bil = (1 - fx) * (1 - fy) * v(y0, x0) + fx * (1 - fy) * v(y0, x0 + 1) + (1 - fx) * fy * v(y0 + 1, x0) + fx * fy * v(y0 + 1, x0 + 1)
print("bilinear by hand at (2.5, 4.3):", round(bil, 3), "| the true value 10*4.3+2.5 =", 10 * 4.3 + 2.5, "| rounded to cell (4, 2) or (4, 3):", v(4, 2), v(4, 3))
```

I saw:
```
tensor([[ 0,  1,  2,  3,  4,  5,  6,  7],
        [10, 11, 12, 13, 14, 15, 16, 17],
        [20, 21, 22, 23, 24, 25, 26, 27],
        [30, 31, 32, 33, 34, 35, 36, 37],
        [40, 41, 42, 43, 44, 45, 46, 47],
        [50, 51, 52, 53, 54, 55, 56, 57],
        [60, 61, 62, 63, 64, 65, 66, 67],
        [70, 71, 72, 73, 74, 75, 76, 77]], dtype=torch.int32)
RoIPool  2x2: [[43.0, 46.0], [63.0, 66.0]]
RoIAlign 2x2: [22.0, 24.5, 42.0, 44.5]
shift (pixels) | RoIPool top-left bin | RoIAlign top-left bin
 0 |  43.0 | 22.000
 4 |  44.0 | 22.250
 8 |  44.0 | 22.500
12 |  44.0 | 22.750
16 |  44.0 | 23.000
bilinear by hand at (2.5, 4.3): 45.5 | the true value 10*4.3+2.5 = 45.5 | rounded to cell (4, 2) or (4, 3): 42 43
```

**What the run shows.**
- The region (image pixels x 20 to 100, y 24 to 88, at stride 16) lands at x 1.25 to 6.25 and y 1.5 to 5.5 on the map. RoI pooling rounds it to whole cells and returns a 2 by 2 block with the values 43 and 46 on top and 63 and 66 below, the maximum cell of each bin.
- **Sliding the region by 4 pixels** (a quarter of a cell): the top-left RoI-pooling value goes 43, then 44, 44, 44, 44 for shifts of 0, 4, 8, 12 and 16 pixels. It jumped by a whole cell on the first 4 pixels and did not move at all on the next 12. [[RoIAlign]] moved smoothly: 22.00, 22.25, 22.50, 22.75, 23.00.
- The two columns are not the same quantity (a maximum against an average of sampled points), so only the way each moves is comparable.

**Summary: what the run shows.** RoI pooling's output moves in jumps, because the region is rounded to whole cells: a 4-pixel shift moved it a full cell, and the next 12 pixels moved it by nothing. Caution: this is one region on a toy map, shown only to expose the rounding; it does not measure the effect on detection accuracy. The accuracy effect is in the Mask R-CNN paper's tables: with a stride-32 backbone, RoI pooling gave 23.6 mask AP and RoIAlign 30.9 (paper's Table 2d, as recorded in [[Mask R-CNN 2017]]).

## 🔴 4. Deep / formal
No equation: the layer is "round the region, split it into a grid, take the maximum of each bin", and the explanation above and the worked run are the whole mechanism.

**Edge cases and details:**
- **Bin sizes:** a region of $h$ by $w$ cells and an output of $H$ by $W$ gives bins of about $h/H$ by $w/W$ cells; when this is not a whole number, the bins differ by one cell, as in the picture.
- **Backpropagation:** the gradient flows only to the cell that held the maximum in each bin (general knowledge). The gradient with respect to the box coordinates is not defined, because rounding is not differentiable. [[Faster R-CNN 2015]] therefore trains with "approximate joint training", treating the proposals as fixed, and notes a non-approximate version would need an RoI layer differentiable in the box coordinates, which it calls beyond its scope.
- **Fewer than 7 by 7 cells:** a region smaller than the grid gives bins that overlap or repeat cells (general knowledge, I did not test it).
- **Max, not average:** a maximum keeps the strongest response in the bin whatever its exact position, which suits classification; I did not test the average in this note. (The Mask R-CNN paper found max and average about the same for RoIAlign: 30.2 and 30.3 AP, its Table 2c.)

**Limits and trade-offs:**
- The two roundings (box to map, map to bins) shift the features by up to half a cell, so small objects and masks suffer ([[RoIAlign]] fixes this).
- A coarse stride makes the problem worse: the 8 pixel maximum shift at stride 16 becomes 16 pixels at stride 32.
- The work grows with the number of boxes, but each box only touches its own cells.

**What came before, and how this improved on it:**
- **Before:** [[R-CNN 2013]] cut out each proposal from the image, warped it to a fixed size and ran the whole network on it, once per box (about 2,000 boxes per image).
- **Its limit:** the network ran about 2,000 times per image, which was slow ([[Faster R-CNN 2015]] note: "about 2,000 crops").
- **What changed:** run the convolutions once on the whole image and pool a fixed-size feature for each box from the shared map, in the style of SPPnet and [[Fast R-CNN 2015]].
- **How much better:** the Faster R-CNN paper states that with shared convolutions the detector is fast "when ignoring the time spent on region proposals", leaving the proposals as the bottleneck (about 1,510 ms for Selective Search against 10 ms for the RPN, recorded in [[Faster R-CNN 2015]]). I did not measure R-CNN against Fast R-CNN myself, and the R-CNN and Fast R-CNN notes are abstract-only.

## Related
[[RoIAlign]] · [[Fast R-CNN 2015]] · [[Faster R-CNN 2015]] · [[Mask R-CNN 2017]] · [[R-CNN 2013]] · [[Pooling]] · [[Feature Map]] · [[Convolutional Neural Network]] · [[Object Detection]] · [[Anchor Box]]

## Sources
- My own run: `roi_demo.py` (Python 3.9.6, torchvision 0.23.0).
- Image: Tomasz Grel (deepsense.io), CC BY 4.0, Wikimedia Commons (link under the picture).
- The Faster R-CNN and Mask R-CNN statements come from those papers as read and recorded in [[Faster R-CNN 2015]] and [[Mask R-CNN 2017]]; I did not re-read the papers for this note. The details of R-CNN (one network pass per crop) are from the notes' own recording of the papers, and the notes on R-CNN and Fast R-CNN are abstract-only. The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision RoI pooling:::A layer that turns a region of any size from a feature map into a fixed-size grid (for example 7 by 7) by splitting it into bins and keeping the maximum of each bin.

#flashcards/ml/concept #flashcards/topic/vision Why does a detector need RoI pooling?::Regions come in many sizes but the classifier needs a fixed-size input; RoI pooling gives every region the same small grid and lets the convolutions run once for the whole image instead of once per box.

#flashcards/ml/concept #flashcards/topic/vision What are the steps of RoI pooling?::Divide the box corners by the stride and round to whole cells; cut the region into a fixed grid of bins (some a cell wider); keep the maximum of each bin; feed the block to the classifier.

#flashcards/ml/concept #flashcards/topic/vision What is the weakness of RoI pooling?::Two roundings (box to map, map to bins) shift the features by up to half a cell, which does not hurt classification much but ruins pixel-accurate masks; RoIAlign removes the rounding.

#flashcards/ml/example #flashcards/topic/vision In my toy run I slid a region by 0, 4, 8, 12 and 16 pixels (stride 16). How did the RoI-pooling output move?::43, 44, 44, 44, 44: a whole cell jump on the first 4 pixels, then nothing. RoIAlign moved smoothly (22.00, 22.25, 22.50, 22.75, 23.00). Toy map, not an accuracy measurement.

#flashcards/ml/concept #flashcards/topic/vision How big can the shift from rounding be at stride 16 and at stride 32?::Up to half a cell: about 8 pixels at stride 16 and 16 pixels at stride 32 (my arithmetic). The Mask R-CNN paper's Table 2d gives 23.6 mask AP for RoI pooling and 30.9 for RoIAlign at stride 32.

#flashcards/ml/concept #flashcards/topic/vision Why is RoI pooling not differentiable with respect to the box coordinates?::Rounding the coordinates is not a smooth operation. Gradients flow only to the cell that held each bin's maximum, so Faster R-CNN uses approximate joint training and treats the proposals as fixed.

#flashcards/ml/image #flashcards/topic/vision ![[roi-pooling-animated.gif]]
What does the animation show?
?
An 8 by 8 input with a 5 by 7 region cut into 2 by 2 sections of unequal size; the maximum of each section (0.85, 0.84, 0.97, 0.96, in red) is kept, giving a 2 by 2 output. Author Tomasz Grel (deepsense.io), CC BY 4.0; I looked at two of the five frames.

#flashcards/ml/concept #flashcards/topic/vision What did RoI pooling replace, and what did it save?::R-CNN warped each of about 2,000 proposals and ran the network on each one. RoI pooling reuses one feature map for the whole image, so the convolutions run once; the proposals became the remaining bottleneck (Selective Search 1,510 ms against 10 ms for the RPN, as recorded in my Faster R-CNN note).

#flashcards/ml/cloze #flashcards/topic/vision RoI pooling rounds the region to whole cells and keeps the ==maximum== of each ==bin==, giving a ==fixed-size== output.
