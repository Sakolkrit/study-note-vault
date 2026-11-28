---
tags: [anchor-box, object-detection, computer-vision, faster-rcnn, bounding-box, ML]
status: growing
level: 4
tier: 2
---

# Anchor Box

An **anchor box** is a reference rectangle of a fixed size and shape, placed at every position of an image (or of its [[Feature Map]]). A detector does not draw a box from nothing. For each anchor it answers two smaller questions: "is there an object here?" and "by how much should I shift and stretch this anchor to fit it?".

## 🟢 1. Simple (age 12)
Imagine looking for animals in a photo with a set of cardboard frames: a small square, a tall rectangle, a wide rectangle, and bigger copies of each. You hold every frame over every spot of the photo and ask, "does an animal fit in this frame?". If a frame almost fits a cat, you only need to nudge it a little to fit exactly. The cardboard frames are the **anchors**. It is much easier to nudge a frame that is nearly right than to draw a perfect box from a blank page.

**Example:** a tall frame is nearly right for a standing person, so the detector says "yes, a person, move the frame a bit to the right and make it slightly narrower".

## 🔵 2. How it works
1. Take the [[Feature Map]] that a [[Convolutional Neural Network]] made from the image. Each position of the map matches a patch of the image.
2. At every position, put several anchors, all centred on that patch: a few sizes (scales) and a few shapes (aspect ratios). Faster R-CNN uses 3 sizes and 3 shapes, so 9 anchors per position ([[Faster R-CNN 2015]]).
3. For each anchor, the network predicts an **object score** ("is something here?") and **four offsets** (how to move and resize the anchor).
4. **Training labels:** compare each anchor with the true boxes using [[Intersection over Union]]. A high overlap makes an anchor a positive example, a low overlap a negative one, and the in-between ones are ignored.
5. **Using the result:** apply the offsets to the anchors with a high object score, then remove duplicates with [[Non-Maximum Suppression]].
6. **Why not draw boxes directly?** Because "make a small adjustment to a nearby standard box" is an easier thing for a network to learn than "output any box of any size".

No image: I searched Wikimedia Commons for anchor-box pictures and found nothing on topic, and the paper's own figure is not free. The numbers below come from a real run instead.

## 🟠 3. Technical detail
All numbers come from `anchor_demo.py` (Python 3.9.6, PyTorch 2.8.0, torchvision 0.23.0, CPU). I wrote the anchor construction myself and cross-checked it against `torchvision`'s `AnchorGenerator`.

```python
"""Anchor boxes: build the 9 Faster R-CNN anchors, count them, label them against a true box, encode and decode offsets."""
import math, torch
from torchvision.ops import box_iou
from torchvision.models.detection.anchor_utils import AnchorGenerator

# 1. the 9 anchors at one position: 3 areas x 3 shapes (width : height = 1:1, 1:2, 2:1 means h/w = 1, 2, 0.5)
def anchors_at(cx, cy, areas=(128, 256, 512), ratios=(1.0, 2.0, 0.5)):
    out = []
    for a in areas:
        for r in ratios:                      # r = height / width
            w = a / math.sqrt(r); h = a * math.sqrt(r)
            out.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return torch.tensor(out)
A = anchors_at(0.0, 0.0)
print("anchor (w x h):", [(round((b[2] - b[0]).item()), round((b[3] - b[1]).item())) for b in A])
print("areas:", [round(((b[2] - b[0]) * (b[3] - b[1])).item()) for b in A])

# 2. how many anchors for a 1000 x 600 image (stride 16 gives a 63 x 38 feature map)
fw, fh = 63, 38
print("anchors:", fw * fh * 9, "| the paper's rough count 60 x 40 x 9 =", 60 * 40 * 9)

# 3. cross-check with torchvision's own generator (it rounds the corners to whole pixels)
gen = AnchorGenerator(sizes=((128, 256, 512),), aspect_ratios=((1.0, 2.0, 0.5),))
tv = gen.generate_anchors(gen.sizes[0], gen.aspect_ratios[0])
print("torchvision base anchors (w x h):", [(int((b[2] - b[0]).item()), int((b[3] - b[1]).item())) for b in tv])

# 4. label anchors against one true box with the 0.7 / 0.3 rule (positive also if best match)
gt = torch.tensor([[30.0, 30.0, 130.0, 130.0]])
cands = {"same": [30, 30, 130, 130], "shift 10": [40, 40, 140, 140], "shift 40": [70, 70, 170, 170], "far": [400, 400, 500, 500]}
boxes = torch.tensor(list(cands.values()), dtype=torch.float32)
iou = box_iou(boxes, gt).squeeze(1)
best = iou.argmax().item()
for (name, _), v, i in zip(cands.items(), iou, range(len(cands))):
    label = "positive" if (v > 0.7 or i == best) else "negative" if v < 0.3 else "ignored"
    print("%-9s IoU %.3f -> %s" % (name, v.item(), label))

# 5. encode and decode the four offsets (centre shift / anchor size, log size ratio), and a round trip
def encode(a, g):
    xa, ya, wa, ha = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2, a[2] - a[0], a[3] - a[1]
    x, y, w, h = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2, g[2] - g[0], g[3] - g[1]
    return [(x - xa) / wa, (y - ya) / ha, math.log(w / wa), math.log(h / ha)]
def decode(a, t):
    xa, ya, wa, ha = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2, a[2] - a[0], a[3] - a[1]
    x, y, w, h = t[0] * wa + xa, t[1] * ha + ya, wa * math.exp(t[2]), ha * math.exp(t[3])
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
anchor = [36, 36, 164, 164]            # centre (100,100), 128 x 128
truth = [30, 50, 190, 150]             # centre (110,100), 160 x 100
t = encode(anchor, truth)
print("offsets:", [round(v, 4) for v in t], "decoded back:", [round(v, 2) for v in decode(anchor, t)])
print("size change as a number: doubling", round(math.log(2), 3), "halving", round(math.log(0.5), 3))
```

I saw:
```
anchor (w x h): [(128, 128), (91, 181), (181, 91), (256, 256), (181, 362), (362, 181), (512, 512), (362, 724), (724, 362)]
areas: [16384, 16384, 16384, 65536, 65536, 65536, 262144, 262144, 262144]
anchors: 21546 | the paper's rough count 60 x 40 x 9 = 21600
torchvision base anchors (w x h): [(128, 128), (256, 256), (512, 512), (90, 182), (182, 362), (362, 724), (182, 90), (362, 182), (724, 362)]
same      IoU 1.000 -> positive
shift 10  IoU 0.681 -> ignored
shift 40  IoU 0.220 -> negative
far       IoU 0.000 -> negative
offsets: [0.0781, 0.0, 0.2231, -0.2469] decoded back: [30.0, 50.0, 190.0, 150.0]
size change as a number: doubling 0.693 halving -0.693
```

**What the run shows.**
- **The nine anchors at one position** have areas of 16,384, 65,536 and 262,144 pixels, which are $128^2$, $256^2$ and $512^2$ (paper's values, [[Faster R-CNN 2015]]), each in three shapes (1:1, 1:2, 2:1). `torchvision` gives the same set, but rounds the sides to whole pixels, so 91 by 181 becomes 90 by 182 (a rounding difference of 1 pixel in my check).
- **How many anchors:** a 1000 by 600 image with a feature stride of 16 gives a 63 by 38 map and 21,546 anchors (my count). The paper says "roughly 20,000" (60 by 40 by 9 is 21,600).
- **The labelling rule** on a true box from (30,30) to (130,130): the identical anchor has IoU 1.000 (positive), one shifted by 10 pixels has 0.681 (below 0.7, so it is positive only if it is the best match, which it is not here, so it is ignored), one shifted by 40 has 0.220 (negative, below 0.3), and a far one has 0 (negative).
- **Offsets and decoding:** for anchor (36,36,164,164) and true box (30,50,190,150) the four offsets are 0.0781, 0.0, 0.2231 and −0.2469, and decoding them returns exactly (30,50,190,150), so the encoding loses nothing.
- **Why a logarithm for the size:** doubling a width gives +0.693 and halving gives −0.693, so growing and shrinking are equal and opposite.

**Summary: what the run shows.** The anchor set is small and fixed, and the offset code is exactly invertible. The thresholds (0.7 and 0.3) leave a gap of ignored anchors, as intended. Caution: this checks the arithmetic, not detection quality. I did not train a detector, and the paper's 0.7 and 0.3 are hand-set numbers that its authors did not tune ([[Faster R-CNN 2015]]).

## 🔴 4. Deep / formal
**The four offsets** the network predicts for an anchor (paper, equation 2 of [[Faster R-CNN 2015]]):

$$
t_x=\frac{x-x_a}{w_a},\quad t_y=\frac{y-y_a}{h_a},\quad t_w=\ln\frac{w}{w_a},\quad t_h=\ln\frac{h}{h_a}
$$

**Where:**
- $x,y,w,h$ are the centre and size of the box to be described (the true box in training). **Matters:** these are the target the network should hit.
- $x_a,y_a,w_a,h_a$ are the centre and size of the anchor. **Matters:** the offsets are measured from the anchor, so the same small numbers mean the same adjustment for a small and a large anchor.
- $t_x,t_y$ are the centre shift as a fraction of the anchor's width and height. **Matters:** dividing by the anchor size puts all anchors on one scale; without it a 10-pixel shift would look the same for a tiny and a huge anchor.
- $t_w,t_h$ are the log of the size ratio. **Matters:** the log makes growing and shrinking symmetric and keeps the decoded size positive.

**Example (tiny numbers):** anchor centre (100,100), size 128 by 128; true box centre (110,100), size 160 by 100.
- $t_x=(110-100)/128=0.0781$ and $t_y=(100-100)/128=0$.
- $t_w=\ln(160/128)=\ln 1.25=0.2231$ and $t_h=\ln(100/128)=-0.2469$.
- The answer means: move the anchor right by 8% of its width, make it 25% wider and about 22% shorter. Decoding the four numbers returns the true box exactly (checked above).

**Translation invariance.** Every position uses the same weights and the same set of anchors, so an object that moves is detected by the same function in its new place. In [[Faster R-CNN 2015]] this keeps the output layer small: $512\times6\times9=27{,}648$ weights, against about 6.1 million for MultiBox's 800 k-means anchors (paper and my check there), about 222 times more.

**Limits and trade-offs:**
- The sizes, shapes and thresholds are hand-set. A dataset with very long thin objects needs other shapes.
- Most anchors are negative, so training samples a balanced batch (256 anchors, at most half positive in Faster R-CNN).
- Many near-duplicate positives make [[Non-Maximum Suppression]] necessary.
- Later detectors dropped anchors ("anchor-free" detectors); I did not read those papers and mention it from general knowledge.

**What came before, and how this improved on it:**
- **Before:** to find objects of many sizes, people ran the detector on an image pyramid (many resized copies) or used filters of many sizes (a filter pyramid), or used MultiBox, which predicted boxes from 800 anchors found by k-means (paper, as recorded in [[Faster R-CNN 2015]]).
- **Its limit:** the image pyramid is time-consuming. MultiBox's anchors were not translation invariant, so its output layer was huge.
- **What changed:** a small set of standard boxes copied to every position, shared weights, and offsets measured from the anchor.
- **How much better:** the output layer shrank from about 6.1 million weights to 27,648, about 222 times fewer ([[Faster R-CNN 2015]], the paper's numbers and my check). Taking only one anchor instead of nine lowered the VOC mean average precision by 3.2 to 4.1 points (65.8 or 66.7 against 69.9; the paper's Table 8 as recorded in the note), so the nine are not decoration.

## Related
[[Faster R-CNN 2015]] · [[Mask R-CNN 2017]] · [[Object Detection]] · [[Feature Map]] · [[Convolutional Neural Network]] · [[Intersection over Union]] · [[Non-Maximum Suppression]] · [[RoI Pooling]] · [[Smooth L1 Loss]] · [[Feature Pyramid Network]]

## Sources
- My own run: `anchor_demo.py` (Python 3.9.6, torchvision 0.23.0).
- The anchor design, the offsets (equation 2), the labelling rule and the weight counts come from the Faster R-CNN paper as read and recorded in [[Faster R-CNN 2015]]; I did not re-read the paper for this note.
- The remark about anchor-free detectors is general knowledge, not checked. The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision Anchor box:::A reference rectangle of fixed size and shape placed at every position of a feature map; the detector predicts an object score and four offsets for each, instead of drawing a box from nothing.

#flashcards/ml/concept #flashcards/topic/vision Why use anchors instead of predicting boxes directly?::Adjusting a nearby standard box is easier to learn than outputting any box of any size, and several sizes and shapes at one position let one feature map find objects of many sizes without an image pyramid (Faster R-CNN paper, as recorded in my note).

#flashcards/ml/concept #flashcards/topic/vision How many anchors per position does Faster R-CNN use, and what are they?::Nine: three areas (128 squared, 256 squared, 512 squared pixels) times three shapes (1:1, 1:2, 2:1).

#flashcards/ml/example #flashcards/topic/vision How many anchors does a 1000 by 600 image have with feature stride 16 and 9 anchors per position?::A 63 by 38 map gives 63 x 38 x 9 = 21,546 (my count). The paper says "roughly 20,000" (60 x 40 x 9 = 21,600).

#flashcards/ml/concept #flashcards/topic/vision Which anchors are positive, negative or ignored in training?::Positive: the best match of a true box, or overlap (IoU) above 0.7 with any. Negative: IoU below 0.3 with every true box. In between: ignored. On my example box the IoUs were 1.000, 0.681, 0.220 and 0 for shifts of 0, 10, 40 pixels and far away.

#flashcards/ml/formula #flashcards/topic/vision Write the four anchor offsets and explain each symbol.
?
$$t_x=\frac{x-x_a}{w_a},\; t_y=\frac{y-y_a}{h_a},\; t_w=\ln\frac{w}{w_a},\; t_h=\ln\frac{h}{h_a}$$
- $x,y,w,h$: centre and size of the box to describe (the true box in training).
- $x_a,y_a,w_a,h_a$: centre and size of the anchor.
- $t_x,t_y$: centre shift as a fraction of the anchor size; dividing puts every anchor on one scale.
- $t_w,t_h$: log size ratios; growing and shrinking are symmetric and the decoded size stays positive.

#flashcards/ml/example #flashcards/topic/vision Anchor centre (100,100) size 128 by 128, true box centre (110,100) size 160 by 100. What are the offsets?::tx = 10/128 = 0.0781, ty = 0, tw = ln(160/128) = 0.2231, th = ln(100/128) = -0.2469 (my run). Decoding them returns the true box exactly.

#flashcards/ml/concept #flashcards/topic/vision Why does the size offset use a logarithm?::Doubling a width gives +0.693 and halving gives -0.693, so growing and shrinking are equal and opposite, and the decoded size is always positive.

#flashcards/ml/concept #flashcards/topic/vision Why does a shared set of anchors keep the network small?::Every position uses the same weights and anchors (translation invariance). Faster R-CNN's output layer has 512 x 6 x 9 = 27,648 weights against about 6.1 million for MultiBox's 800 k-means anchors, about 222 times more (paper and my check, recorded in my Faster R-CNN note).

#flashcards/ml/concept #flashcards/topic/vision What does an anchor-based detector still need after the offsets are applied?::Non-maximum suppression, because many neighbouring anchors fire on the same object and give near-duplicate boxes.

#flashcards/ml/cloze #flashcards/topic/vision An anchor box is a ==reference box== at each position; the network predicts an ==object score== and four ==offsets== for it.
