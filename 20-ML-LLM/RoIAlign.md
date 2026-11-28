---
tags: [roialign, object-detection, instance-segmentation, computer-vision, mask-rcnn, bilinear-interpolation, ML]
status: growing
level: 4
tier: 2
---

# RoIAlign

**RoIAlign** is the layer [[Mask R-CNN 2017]] uses to cut a region out of a [[Feature Map]] and turn it into a small fixed-size grid, **without rounding any coordinate**. Where [[RoI Pooling]] snaps the region to whole cells, RoIAlign reads the feature map at the exact fractional positions by blending the four nearest cells (bilinear interpolation), so the small grid stays lined up with the pixels it came from.

## 🟢 1. Simple (age 12)
Suppose you trace a shape on a sheet of graph paper, but the shape's edge falls halfway between two lines. One way is to round to the nearest line, which makes every shape a little crooked. Another way is to say "this spot is halfway between the two lines, so I will mix half of each". The second way is RoIAlign: it never moves your shape, it just blends the neighbours. For a coarse label like "dog", crooked by half a square does not matter. For a precise outline of the dog, it does.

**Example:** a spot sits at 2.5 on a ruler marked in whole numbers. Rounding says 3 and gets a value that belongs to a different place. Blending the values at 2 and 3 equally gives the value that belongs at 2.5.

## 🔵 2. How it works
1. Map the box from image pixels to the feature map by dividing by the stride, **and keep the fraction**: a corner at 100 pixels is at 6.25 on a stride-16 map, not rounded to 6.
2. Split the region into the output grid (7 by 7 for classification, 14 by 14 for the mask head in Mask R-CNN).
3. In each bin, choose a few sampling points at regular spacing (Mask R-CNN uses 4 per bin).
4. For each sampling point, find the **four feature-map cells** around it and blend them by distance (**bilinear interpolation**). A point close to one cell takes mostly that cell's value.
5. Combine the points of a bin with a maximum or an average. The paper found the two about the same.
6. The output block lines up with the true region, so a mask head that works on it can place the object's outline correctly.

No image: I found no free picture of RoIAlign on Wikimedia Commons, and the paper's Figure 3 is not free. The [[RoI Pooling]] note has a free picture of the pooling it replaces, and the numbers below come from a real run.

## 🟠 3. Technical detail
Numbers come from `roi_demo.py` (Python 3.9.6, PyTorch 2.8.0, torchvision 0.23.0, CPU). On the 8 by 8 map every cell holds `10*row + column`, so the right value at any position is `10*y + x`, even between cells.

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
- **The hand calculation agrees with the true value.** At position x = 2.5, y = 4.3 the four neighbours blend to 45.5, which equals the exact `10*4.3 + 2.5 = 45.5`. Rounding would read the cell at row 4 and column 2 or 3 and give 42 or 43, a wrong place.
- **Smooth movement.** Sliding the region right by 4, 8, 12 and 16 pixels (quarters of a cell, at stride 16) changes the RoIAlign value by exactly 0.25 each time (22.00 to 23.00). The [[RoI Pooling]] value stayed fixed after its first jump.
- The library function was called with `aligned=True`, which shifts the coordinates by half a cell so that pixel centres match; I did not test the other setting.

**Summary: what the run shows.** On a map where the right answer is known, RoIAlign reads the correct value between cells and moves in step with the region; RoI pooling does not. Caution: this is a toy linear map, where interpolation is exact by construction. On a real feature map, interpolation is only a smooth guess between cells. I did not train or evaluate a segmentation model.

## 🔴 4. Deep / formal
No equation: bilinear interpolation is an ordinary weighted average of four neighbours, described above and computed in the run. (The Mask R-CNN paper describes it in words and does not print the formula; see [[Mask R-CNN 2017]].)

**What the paper measured (Mask R-CNN paper, Table 2c and 2d, as recorded in [[Mask R-CNN 2017]]):**

| Setting (ResNet-50) | Layer | Mask AP |
|---|---|---|
| Stride 32 features (C5) | RoIPool | 23.6 |
| Stride 32 features (C5) | RoIAlign | 30.9 |

- **Mask AP:** the average precision of the predicted masks, the headline COCO metric (higher is better).
- **Stride:** how many image pixels one feature cell covers; at stride 32 the rounding error is largest.
- Only the two stride-32 rows are copied here (written by me from the numbers recorded in [[Mask R-CNN 2017]]); the full table is there, and I did not re-check it against the paper for this note.

**Summary: what the table shows.** At stride 32 RoIAlign is 7.3 points better (30.9 against 23.6, my arithmetic). Caution: a different layer from RoIAlign, RoIWarp (also bilinear, but still rounding the region), performed about the same as RoIPool (27.2 against 26.9 AP in the paper's Table 2c, as recorded in my Mask R-CNN note). So the gain comes from removing the rounding, not from interpolation alone.

**Limits and trade-offs:**
- It costs a little more than RoI pooling (four reads and a blend per point).
- It does not fix a coarse feature map: it keeps position accurate, it does not add detail the map lacks.
- Two details (the number of sampling points, and `aligned`) change results slightly; the paper reports max and average pooling about equal.

**What came before, and how this improved on it:**
- **Before:** [[RoI Pooling]] rounded the region twice (box to map, map to bins).
- **Its limit:** the rounding "may not impact classification" but "has a large negative effect on predicting pixel-accurate masks" (Mask R-CNN paper, as quoted in my note).
- **What changed:** keep fractional coordinates and read each sampling point by bilinear interpolation.
- **How much better:** 30.9 against 23.6 mask AP on a stride-32 backbone (paper's Table 2d; difference 7.3 is my arithmetic), and in my toy run a 4-pixel shift moved RoIAlign by 0.25 cells and RoI pooling by a full cell.

## Related
[[RoI Pooling]] · [[Mask R-CNN 2017]] · [[Faster R-CNN 2015]] · [[FCN 2014]] · [[Image Segmentation]] · [[Feature Map]] · [[Object Detection]] · [[Feature Pyramid Network]] · [[Transposed Convolution]]

## Sources
- My own run: `roi_demo.py` (Python 3.9.6, torchvision 0.23.0).
- The Mask R-CNN paper's description of RoIAlign, its quotations and Table 2 numbers come from [[Mask R-CNN 2017]] as read there; I did not re-read the paper for this note.
- No free image was found, so none is used. The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision RoIAlign:::A layer that extracts a fixed-size feature grid for a region without rounding any coordinate, reading the map at fractional positions with bilinear interpolation (Mask R-CNN).

#flashcards/ml/concept #flashcards/topic/vision What problem of RoI pooling does RoIAlign fix?::RoI pooling rounds the region twice, shifting the features by up to half a cell; this barely hurts classification but badly hurts pixel-accurate masks.

#flashcards/ml/concept #flashcards/topic/vision How does RoIAlign read a value at a fractional position?::It takes the four nearest feature cells and blends them by distance (bilinear interpolation). In Mask R-CNN each bin has 4 sampling points, combined by max or average.

#flashcards/ml/example #flashcards/topic/vision On my toy map (cell value 10 x row + column), what does bilinear interpolation give at x = 2.5, y = 4.3, and what does rounding give?::45.5, which equals the exact 10 x 4.3 + 2.5. Rounding to a cell gives 42 or 43. Toy linear map, so interpolation is exact there.

#flashcards/ml/concept #flashcards/topic/vision In my toy run, how did RoIAlign move when the region slid in quarter-cell steps?::By exactly 0.25 each step (22.00, 22.25, 22.50, 22.75, 23.00), while RoI pooling jumped a whole cell and then stayed still.

#flashcards/ml/concept #flashcards/topic/vision How much did RoIAlign help in the Mask R-CNN paper at stride 32?::30.9 against 23.6 mask AP for RoI pooling, 7.3 points more (paper's Table 2d, as recorded in my note; the difference is my arithmetic).

#flashcards/ml/concept #flashcards/topic/vision Is the gain of RoIAlign from interpolation or from removing the rounding?::From removing the rounding. RoIWarp, which also interpolates but still rounds the region, performed about like RoI pooling (27.2 against 26.9 AP in the paper's Table 2c, as recorded in my note).

#flashcards/ml/concept #flashcards/topic/vision What does RoIAlign not do?::It does not add detail the feature map lacks; it only keeps positions accurate. In my toy run, interpolation was exact only because the map was linear.

#flashcards/ml/cloze #flashcards/topic/vision RoIAlign keeps ==fractional== coordinates and reads each point with ==bilinear interpolation== from the ==four== nearest cells.
