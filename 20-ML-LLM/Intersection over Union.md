---
tags: [computer-vision, object-detection, evaluation, metrics, geometry]
status: growing
level: 4
tier: 2
---

# Intersection over Union

**Intersection over union (IoU)** is a number from 0 to 1 that says **how much two shapes overlap**. In [[Object Detection]] it is the ruler that tells whether a predicted box is "the same box" as the true one.

## 🟢 1. Simple (age 12)
Two friends each draw a frame around the same dog. If the frames are exactly the same, they agree completely. If one frame is a bit to the side, they agree a bit. IoU turns "how much do the two frames agree" into one number: **the part both frames share, divided by everything either frame covers**.

**Example:** you cut two paper rectangles and lay one over the other. The shared patch is the overlap. All the paper you can see (both rectangles together, with the shared patch counted once) is the union. A fully covered pair gives 1, and two rectangles that do not touch give 0.

## 🔵 2. How it works
1. You have two boxes: the one a detector predicted and the true one that a person drew.
2. Find the **overlap** rectangle. Its width is how far the two boxes share left-to-right, its height how far they share top-to-bottom. If they do not cross in either direction, the overlap is empty (area 0).
3. Work out the **union**: the area of box 1 plus the area of box 2, minus the overlap (otherwise the shared patch would be counted twice).
4. **Divide** overlap by union. A perfect match gives 1, a miss gives 0.
5. Pick a **bar**: often 0.5. A predicted box with IoU of at least 0.5 against a true box counts as a hit. This is how a detector is scored ([[Mean Average Precision]]).
6. The same number is used in two other places in a detector: to decide that two predicted boxes point at the same object, so only one is kept ([[Non-Maximum Suppression]]), and to decide, during training, which candidate boxes count as positive examples (from memory).

![[intersection-over-union.png]]
*Intersection over union (IoU): the area where two boxes overlap, divided by the area that the two boxes cover together. The picture shows two overlapping squares at the top, the shaded overlap above the line and the shaded union below it. Image: Adrian Rosebrock, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Intersection_over_Union_-_visual_equation.png), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Boxes as four numbers:** `(x0, y0, x1, y1)`, left, top, right and bottom. IoU needs only their areas and the overlap, so it is a few lines of code. `torchvision.ops.box_iou` does it for many boxes at once (used in the second script below).
- **The 0.5 bar is looser than it sounds, and the high bars are very strict.** For two squares of the same size, sliding one sideways by a third of its width already drops IoU to 0.5, and for a square of side 10 an IoU of 0.9 allows a shift of only 0.53 (my arithmetic, checked in the code below).
- **It ignores class.** A perfect box with the wrong label has an IoU of 1. The class is checked separately ([[Confusion Matrix]]).
- **It is about shape, not distance.** Two boxes that do not touch have an IoU of 0 whether they are one pixel or a whole photo apart.

First, a small function and some cases (only plain Python):

```python
# file: iou_toy.py
def iou(a, b):
    """Boxes are (x0, y0, x1, y1). Intersection over union."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0

A = (0, 0, 10, 10)
for name, B in [("same box", (0, 0, 10, 10)), ("shifted by 5", (5, 0, 15, 10)), ("shifted by 2 each way", (2, 2, 12, 12)),
                ("a box inside it", (0, 0, 5, 5)), ("touching", (10, 0, 20, 10)), ("far away", (50, 50, 60, 60)),
                ("right centre, twice as wide", (-5, 0, 15, 10))]:
    print(f"{name:28s} IoU = {iou(A, B):.3f}")
print("same 2-pixel shift: a 10-pixel box", round(iou((0, 0, 10, 10), (2, 0, 12, 10)), 3), " a 100-pixel box", round(iou((0, 0, 100, 100), (2, 0, 102, 100)), 3))
for t in (0.5, 0.75, 0.9):                      # sideways shift d of a square of side s: IoU = (s-d)/(s+d), so d = s(1-t)/(1+t)
    d = 10 * (1 - t) / (1 + t); print(f"IoU {t}: a square of side 10 may move sideways by {d:.2f} (check {iou((0,0,10,10),(d,0,10+d,10)):.2f})")
```

```text
same box                     IoU = 1.000
shifted by 5                 IoU = 0.333
shifted by 2 each way        IoU = 0.471
a box inside it              IoU = 0.250
touching                     IoU = 0.000
far away                     IoU = 0.000
right centre, twice as wide  IoU = 0.500
same 2-pixel shift: a 10-pixel box 0.667  a 100-pixel box 0.961
IoU 0.5: a square of side 10 may move sideways by 3.33 (check 0.50)
IoU 0.75: a square of side 10 may move sideways by 1.43 (check 0.75)
IoU 0.9: a square of side 10 may move sideways by 0.53 (check 0.90)
```

Reading it: shifted by 5 gives $50/150=0.333$. A box twice as wide with the right centre gives 0.5, so a rule based on the distance between centres would call it perfect while IoU gives only 0.5. The same 2-pixel shift costs a small box a third of its score (0.667) and a large box almost nothing (0.961).

Now the real detector from [[Tiny Object Detector]]. For each of the 305 true validation boxes, the best IoU with any box the detector was at least 50% sure of. It needs `val_preds.pt` (written by `evaluate.py` in that project), `train.py` and `data/subset.json` in the same folder, and runs in that project's virtual environment:

```python
# file: iou_stats.py
"""How tight are the detector's boxes? IoU of each true box with its best box of score >= 0.5 (needs val_preds.pt from evaluate.py)."""
import torch
from torchvision.ops import box_iou
import train as T

preds = torch.load("val_preds.pt")
best = []
for j, r in enumerate(T.meta["val"]):
    b, s, l = preds[j]; b = b[s >= 0.5]
    for x0, y0, x1, y1, c in r["boxes"]:
        best.append(float(box_iou(torch.tensor([[x0, y0, x1, y1]]), b).max()) if len(b) else 0.0)
v = torch.tensor(best)
print(f"{len(v)} true boxes; best-box IoU >= 0.5 for {(v >= 0.5).sum().item()}, >= 0.75 for {(v >= 0.75).sum().item()}, >= 0.9 for {(v >= 0.9).sum().item()}")
f = v[v >= 0.5]
print(f"among the {len(f)} found boxes the median IoU is {f.median().item():.3f}, the mean {f.mean().item():.3f}")
print("true boxes with no box at all (score >= 0.5):", (v == 0).sum().item())
```

```text
305 true boxes; best-box IoU >= 0.5 for 216, >= 0.75 for 146, >= 0.9 for 36
among the 216 found boxes the median IoU is 0.822, the mean 0.790
true boxes with no box at all (score >= 0.5): 51
```

Reading it: 216 of 305 true boxes have a good match at the 0.5 bar, but only 146 at 0.75 and 36 at 0.9. The found boxes are right but not tight (median 0.822), and 51 true boxes got no box at all. (The 216 here counts any class; the 206 in [[Tiny Object Detector]] also required the right class.)

## 🔴 4. Deep / formal
For two regions $A$ (predicted) and $B$ (true):

$$
\text{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}=\frac{|A\cap B|}{|A|+|B|-|A\cap B|}
$$

**Where:**
- $A$ and $B$ are the two boxes, written as the sets of points inside them. $|\cdot|$ is the area.
- $A\cap B$ is the overlap, and $A\cup B$ the union, everything covered by at least one of the two.
- **Matters:** the union in the denominator punishes both a box that is too big and one that is in the wrong place. Divide by the true box's area alone and a huge box that covers the whole photo would score 1 and look perfect.

**Example (tiny numbers):** box $A$ runs from $(0,0)$ to $(10,10)$ and box $B$ from $(2,2)$ to $(12,12)$.
- Overlap: from $(2,2)$ to $(10,10)$, which is $8\times 8=\mathbf{64}$.
- Areas: $10\times10=100$ each, so the union is $100+100-64=\mathbf{136}$.
- $\text{IoU}=64/136=\mathbf{0.471}$.
- Meaning: each box is shifted by only 2 in both directions, yet the overlap is under half the covered area, so this box would miss the 0.5 bar.

**Limits and variants:**
- **No signal when boxes do not touch.** The IoU is 0 for any two separate boxes, so as a training loss it gives no hint of which way to move. Variants that add a term for the gap or for the distance between centres (GIoU, DIoU, CIoU) exist for that reason; the names and papers are from memory and I did not check them.
- **Axis-aligned boxes only.** For rotated boxes or exact outlines the same idea is applied to polygons or to pixel masks (for [[Image Segmentation]] it is the standard score).
- **The bar is a convention.** 0.5 was a common choice and COCO's headline number averages ten bars from 0.5 to 0.95 (that average I checked in [[Tiny Object Detector]]; the history of the choice is from memory).

**What came before, and how this improved on it:**
- **Before:** the obvious ways to compare two boxes are the distance between their centres or between their corners (a natural first idea; I am not citing a source).
- **Its limit:** the same pixel distance is a tiny error for a large box and a huge one for a small box, and a box with the right centre can be far too wide or too narrow.
- **What changed:** divide the overlap by the union. The measure is the Jaccard index, named after a botanist (from memory, about 1901), and detection contests later used it with a 0.5 bar (from memory).
- **How much better:** in the toy code, a box with the right centre but twice as wide has a centre distance of 0 and an IoU of 0.5; and the same 2-pixel shift gives an IoU of 0.667 for a 10-pixel box and 0.961 for a 100-pixel box, so IoU scales with the size of the box. In my detector 216, 146 and 36 of 305 true boxes had a best box at the 0.5, 0.75 and 0.9 bars (my own run).

## Related
- [[Object Detection]] · [[Non-Maximum Suppression]] · [[Mean Average Precision]] · [[Tiny Object Detector]] · [[Image Segmentation]] · [[Evaluation Metrics]] · [[Confusion Matrix]] · [[Faster R-CNN 2015]] · [[YOLO 2015]] · [[Precision-Recall Curve]]

## Sources
- My own toy and runs (code above; the numbers come from the printed output). The Jaccard index and its date, GIoU, DIoU and CIoU, the use of IoU in training, and the history of the 0.5 bar are from memory; I read no paper for this note. Every code block in this note was extracted into a fresh folder (with a copy of the files from [[Tiny Object Detector]] for the blocks that need them) and run again; each output matched the block above. Not tested: another machine or other library versions.
- Image: [Intersection_over_Union_-_visual_equation.png](https://commons.wikimedia.org/wiki/File:Intersection_over_Union_-_visual_equation.png) by Adrian Rosebrock, CC BY-SA 4.0.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision Intersection over union (IoU):::The area shared by two boxes divided by the area they cover together (the overlap counted once). It is 1 for a perfect match and 0 when the boxes do not touch.

#flashcards/ml/formula #flashcards/topic/vision Write the intersection over union of two boxes and explain each symbol.
?
$$\text{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}=\frac{|A\cap B|}{|A|+|B|-|A\cap B|}$$
- $A$, $B$: the predicted and the true box, as the sets of points inside them. $|\cdot|$ is the area.
- $A\cap B$: the overlap. $A\cup B$: the union.
- Dividing by the union punishes a box that is too big and one that is in the wrong place; dividing by the true box alone would give a huge box a perfect score.

#flashcards/ml/example #flashcards/topic/vision Box A is (0,0) to (10,10) and box B is (2,2) to (12,12). What is the IoU?::The overlap is from (2,2) to (10,10), 8 × 8 = 64. The union is 100 + 100 - 64 = 136. IoU = 64 / 136 = 0.471, below the usual 0.5 bar, so it would count as a miss (my arithmetic).

#flashcards/ml/image #flashcards/topic/vision ![[intersection-over-union.png]]
What does this picture show?
?
Intersection over union: the area where two boxes overlap divided by the area they cover together. Two overlapping squares at the top, the shaded overlap above the line and the shaded union below it. (Adrian Rosebrock, Wikimedia Commons, CC BY-SA 4.0.)

#flashcards/ml/concept #flashcards/topic/vision What are the steps of computing IoU?
?
Find the overlap rectangle (area 0 if the boxes do not cross); work out the union as area 1 plus area 2 minus the overlap; divide overlap by union; compare with a bar, often 0.5, to decide whether the predicted box counts as a hit.

#flashcards/ml/concept #flashcards/topic/vision Why is IoU better than the distance between box centres?
?
A box with the right centre but twice as wide has a centre distance of 0 but an IoU of 0.5. And the same 2-pixel shift gives an IoU of 0.667 for a 10-pixel box and 0.961 for a 100-pixel box, so IoU scales with the box (my own toy).

#flashcards/ml/concept #flashcards/topic/vision How far can a box move sideways before its IoU falls below a given bar?
?
For a square of side $s$ moved sideways by $d$, IoU = $(s-d)/(s+d)$, so $d=s(1-t)/(1+t)$ for a bar $t$. For side 10: 3.33 for bar 0.5, 1.43 for 0.75 and 0.53 for 0.9 (my arithmetic, checked in the code).

#flashcards/ml/concept #flashcards/topic/vision What did the IoU show for the Tiny Object Detector's boxes?
?
Of 305 true validation boxes, 216 had a predicted box (score at least 0.5) with IoU at least 0.5, 146 at least 0.75 and 36 at least 0.9; median IoU of the found boxes 0.822; 51 true boxes got no box. The boxes are right but not tight. My own run, one trained detector.

#flashcards/ml/concept #flashcards/topic/vision What are the limits of IoU?
?
It is 0 for any two boxes that do not touch, however near (no training signal, which GIoU, DIoU and CIoU variants try to fix, from memory); it ignores class; it works for axis-aligned boxes; and the 0.5 bar is a convention.

#flashcards/ml/cloze #flashcards/topic/vision IoU divides the ==overlap== of two boxes by their ==union==, and it ignores the ==class== label.
