---
tags: [computer-vision, object-detection, post-processing, algorithm]
status: growing
level: 4
tier: 2
---

# Non-Maximum Suppression

**Non-maximum suppression (NMS)** is the clean-up step of a detector. A [[Object Detection]] network usually draws **many boxes around the same object**; NMS keeps the most confident one and deletes the rest.

## 🟢 1. Simple (age 12)
Ten friends all point at the same dog and all shout "dog!" You only need to hear it once. So you keep the friend who sounds most sure, and tell everyone else who is pointing at nearly the same spot to be quiet. If someone points at a *different* dog, you keep that one too.

**Example:** a photo editor sees five frames drawn around one cat, each with a "how sure am I" score. NMS keeps the frame with the highest score and throws the other four away, because they are nearly the same frame.

## 🔵 2. How it works
1. The detector outputs a list of boxes. Each has a **class** (cat, dog, bus) and a **score** (how sure it is, from 0 to 1).
2. **Sort** the boxes by score, highest first.
3. **Keep** the top box.
4. Compare it with every other box **of the same class** using [[Intersection over Union]]. If the overlap is above a **threshold** (often 0.5), the two boxes are probably the same object, so delete the lower-scored one.
5. Take the best box that is still left and repeat from step 3 until nothing is left.
6. Boxes of a different class are not compared (in the toy below, a "dog" box in the same place as a "cat" box survives).

![[nms-off-on.png]]
*The same validation photo (COCO file 000000108244; Flickr address `farm1.staticflickr.com/49/165710819_fb98979f1e_z.jpg`; CC Attribution-ShareAlike as listed in COCO's file, which names no photographer) with the boxes of my trained detector from [[Tiny Object Detector]] that have a score of at least 0.5. Left: non-maximum suppression switched off, 25 boxes piled on the one cat. Right: switched on at 0.5, one box. Boxes and labels drawn by me, so the picture is a derivative shared under the same licence.*

## 🟠 3. Technical detail
- **It is a loop, not a layer.** NMS is plain code that runs after the network, on the boxes it has produced. It has no weights to learn.
- **Per class.** Boxes are only compared with boxes of the same class, so a cat and a dog sitting at the same place both survive.
- **The threshold is the only real knob.** A high threshold (0.9) keeps nearly everything; a low one (0.3) deletes more.
- **Cost.** In the worst case each kept box is compared with all boxes left, so the work grows with the square of the number of boxes (my reasoning, not measured). Detectors first drop boxes with very low scores so the list is short.

A small version in plain Python (five boxes: three on one cat, one bus far away, and a dog at the cat's place):

```python
# file: nms_toy.py
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)

def nms(boxes, scores, labels, thr=0.5):
    """Greedy, per class: keep the best box, drop every other box of the same class with IoU > thr, repeat."""
    order = sorted(range(len(boxes)), key=lambda i: -scores[i]); keep = []
    while order:
        i = order.pop(0); keep.append(i)
        order = [j for j in order if labels[j] != labels[i] or iou(boxes[i], boxes[j]) <= thr]
    return keep

boxes  = [(10, 10, 60, 60), (14, 12, 64, 62), (8, 14, 58, 66), (100, 100, 150, 150), (12, 12, 62, 62)]
scores = [0.90, 0.80, 0.60, 0.95, 0.70]
labels = ["cat", "cat", "cat", "bus", "dog"]            # the last box has the same place as the cat but another class
for i, b in enumerate(boxes[1:3], 1): print(f"IoU of box 0 with box {i}: {iou(boxes[0], b):.3f}")
print("kept (threshold 0.5):", nms(boxes, scores, labels, 0.5))
print("kept (threshold 0.9):", nms(boxes, scores, labels, 0.9))
```

```text
IoU of box 0 with box 1: 0.791
IoU of box 0 with box 2: 0.763
kept (threshold 0.5): [3, 0, 4]
kept (threshold 0.9): [3, 0, 1, 4, 2]
```

Reading it: box 0 (score 0.90) is kept; boxes 1 and 2 overlap it by 0.791 and 0.763, above 0.5, so they go. The bus (index 3) and the dog (index 4) are kept: the bus is far away, the dog is the same place but another class. With a threshold of 0.9 nothing overlaps *that* much, so all five boxes stay.

Now the real detector. The script runs the 240 validation photos with NMS switched off (threshold 1.0, which no pair of boxes can exceed), at 0.5 (the setting used in [[Tiny Object Detector]]) and at 0.3, and scores each with COCO's mAP. It needs `detector.pt`, `train.py` and `data/` from that project, and its virtual environment:

```python
# file: nms_ablation.py
"""What does non-maximum suppression do to the trained detector? Same 240 photos, three NMS settings (needs detector.pt, train.py, data/)."""
import io, contextlib, torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import train as T

torch.set_num_threads(4)
val = T.meta["val"]
gt = {"images": [], "annotations": [], "categories": [{"id": i, "name": n} for i, n in enumerate(T.meta["classes"], 1)]}
k = 1
for j, r in enumerate(val):
    gt["images"].append({"id": j, "width": r["w"], "height": r["h"]})
    for x0, y0, x1, y1, c in r["boxes"]:
        gt["annotations"].append({"id": k, "image_id": j, "category_id": c, "bbox": [x0, y0, x1 - x0, y1 - y0],
                                  "area": (x1 - x0) * (y1 - y0), "iscrowd": 0}); k += 1
m = T.build(); m.load_state_dict(torch.load("detector.pt")); m.eval()
ds = T.DS(val, False)
for thr in (1.0, 0.5, 0.3):                      # IoU above which a lower-scored box is deleted; 1.0 = NMS effectively off
    m.roi_heads.nms_thresh = thr
    dets = []
    with torch.no_grad():
        for j in range(len(ds)):
            o = m([ds[j][0]])[0]
            for b, s, l in zip(o["boxes"].tolist(), o["scores"].tolist(), o["labels"].tolist()):
                dets.append({"image_id": j, "category_id": l, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": s})
    with contextlib.redirect_stdout(io.StringIO()):
        G = COCO(); G.dataset = gt; G.createIndex()
        E = COCOeval(G, G.loadRes(dets), "bbox"); E.evaluate(); E.accumulate(); E.summarize()
    sure = sum(d["score"] >= 0.5 for d in dets)
    print(f"NMS threshold {thr}: {len(dets)} boxes ({len(dets)/len(ds):.1f} per photo), {sure} with score >= 0.5, mAP {E.stats[0]:.3f}, mAP@.5 {E.stats[1]:.3f}")
```

```text
NMS threshold 1.0: 12545 boxes (52.3 per photo), 4642 with score >= 0.5, mAP 0.130, mAP@.5 0.182
NMS threshold 0.5: 1112 boxes (4.6 per photo), 276 with score >= 0.5, mAP 0.407, mAP@.5 0.695
NMS threshold 0.3: 787 boxes (3.3 per photo), 251 with score >= 0.5, mAP 0.396, mAP@.5 0.684
```

The picture above comes from this script (it needs the same files and also `arial.ttf`, a Windows font):

```python
# file: nms_picture.py
"""Draw one photo twice: boxes with score >= 0.5 with NMS switched off and on (needs detector.pt, train.py, data/)."""
import torch
from PIL import Image, ImageDraw, ImageFont
import train as T

FILE = "000000108244.jpg"                          # a validation photo whose COCO licence allows reuse
names = ["", "cat", "dog", "bus"]; col = {1: "#d62728", 2: "#1f77b4", 3: "#2ca02c"}
m = T.build(); m.load_state_dict(torch.load("detector.pt")); m.eval()
j = [r["file"] for r in T.meta["val"]].index(FILE)
img = T.DS(T.meta["val"], False)[j][0]
font = ImageFont.truetype("arial.ttf", 22); tiles = []
for thr, title in ((1.0, "NMS off"), (0.5, "NMS on (0.5)")):
    m.roi_heads.nms_thresh = thr
    with torch.no_grad(): o = m([img])[0]
    im = Image.open(f"data/images/{FILE}").convert("RGB"); d = ImageDraw.Draw(im); n = 0
    for b, s, l in zip(o["boxes"].tolist(), o["scores"].tolist(), o["labels"].tolist()):
        if s >= 0.5:
            n += 1; d.rectangle(b, outline=col[l], width=3); d.text((b[0] + 4, b[1] + 2), f"{names[l]} {s:.2f}", fill="white", font=font, stroke_width=2, stroke_fill=col[l])
    d.text((8, im.size[1] - 34), f"{title}: {n} box(es)", fill="white", font=font, stroke_width=3, stroke_fill="black")
    print(f"{title}: {n} box(es) with score >= 0.5"); tiles.append(im)
W = sum(t.size[0] for t in tiles) + 10; canvas = Image.new("RGB", (W, tiles[0].size[1]), "white")
canvas.paste(tiles[0], (0, 0)); canvas.paste(tiles[1], (tiles[0].size[0] + 10, 0)); canvas.save("nms-off-on.png")
```

```text
NMS off: 25 box(es) with score >= 0.5
NMS on (0.5): 1 box(es) with score >= 0.5
```

Reading it: without NMS the detector gave 12,545 boxes (52.3 per photo) and a [[Mean Average Precision]] of 0.130; with it 1,112 boxes (4.6 per photo) and 0.407. The extra boxes are all false alarms in the score, because each true box can be matched only once. "Off" here still has the library's other limits (a floor of 0.05 on the score and at most 100 boxes per photo, from memory of torchvision's defaults, which I did not print).

## 🔴 4. Deep / formal
- **Not a formula, a rule.** Sort by score; keep the best; delete every box of the same class whose IoU with it is above the threshold $t$; repeat.
- **The threshold is a trade-off.** Too high and duplicates survive (more false alarms); too low and two real objects that overlap (people in a crowd, a row of buses) are merged, and one is lost. In my run, 0.3 gave 0.396 against 0.407 at 0.5, with 787 against 1,112 boxes. That is one detector and one run, so I do not call the difference real.
- **It uses only the score and the geometry.** It cannot know that two boxes hold two different real objects; it assumes high overlap means the same object.
- **Soft-NMS and NMS-free detectors** (from memory, not run): Soft-NMS lowers the score of an overlapping box instead of deleting it (Bodla and others, 2017), and a family of detectors predict a set of boxes directly so that no NMS is needed (DETR, 2020).
- **Same name, older use.** "Non-maximum suppression" is also the name of a step in edge detection, where it thins an edge to one pixel wide; I saw it in a file title on Wikimedia Commons, but I did not read about it.

**What came before, and how this improved on it:**
- **Before:** keep every box that the network outputs above some score (older sliding-window detectors grouped or merged overlapping windows; from memory).
- **Its limit:** one object gets many boxes. In the scoring each true box can be matched only once, so the others count as false alarms, and the result is cluttered.
- **What changed:** keep the best box of each group of overlapping boxes of a class and delete the others.
- **How much better:** measured in my own run on 240 photos: 12,545 boxes without it and 1,112 with it (threshold 0.5), and mAP 0.130 against 0.407. The picture shows 25 boxes against 1 on one cat.

## Related
- [[Object Detection]] · [[Intersection over Union]] · [[Mean Average Precision]] · [[Tiny Object Detector]] · [[Faster R-CNN 2015]] · [[YOLO 2015]] · [[Precision]] · [[Evaluation Metrics]]

## Sources
- My own toy and runs (code above). Soft-NMS (Bodla and others, 2017), DETR (2020), the edge-detection use of the name, older sliding-window grouping and torchvision's defaults are from memory; I read no paper or documentation for this note. Every code block in this note was extracted into a fresh folder (with a copy of the files from [[Tiny Object Detector]] for the blocks that need them) and run again; each output matched the block above. Not tested: another machine or other library versions.
- The picture was made by me from a COCO validation photo (see its caption).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision Non-maximum suppression (NMS):::A detector's clean-up step: keep the highest-scoring box, delete every other box of the same class that overlaps it by more than a threshold (IoU), and repeat with the best box that is left.

#flashcards/ml/image #flashcards/topic/vision ![[nms-off-on.png]]
What does this picture show?
?
The same photo with the trained detector's boxes of score at least 0.5: with non-maximum suppression off there are 25 boxes on one cat, with it on (threshold 0.5) one box. COCO photo (CC Attribution-ShareAlike as listed in its file), boxes drawn by me.

#flashcards/ml/concept #flashcards/topic/vision What are the steps of non-maximum suppression?
?
Sort the boxes by score; keep the top one; delete every box of the same class whose IoU with it is above the threshold (often 0.5); repeat with the best box left, until none are left. Boxes of different classes are not compared.

#flashcards/ml/concept #flashcards/topic/vision Why does a detector need non-maximum suppression?
?
It outputs many boxes for one object. In scoring each true box can be matched only once, so the extra boxes are false alarms. In my run, without NMS: 12,545 boxes and mAP 0.130; with it at 0.5: 1,112 boxes and mAP 0.407 (240 photos).

#flashcards/ml/example #flashcards/topic/vision In the toy, box 0 (cat, 0.90) overlaps box 1 (cat, 0.80) by IoU 0.791 and box 2 (cat, 0.60) by 0.763. Which boxes does NMS keep at threshold 0.5?::The cat at 0.90, plus the bus (0.95, far away) and the dog (0.70, same place but another class): kept indices 3, 0, 4. At threshold 0.9 nothing overlaps enough to be deleted, so all five stay.

#flashcards/ml/concept #flashcards/topic/vision What does the NMS threshold trade off?
?
Too high and duplicate boxes survive (false alarms); too low and two real, overlapping objects (a crowd) are merged and one is lost. In my run 0.3 gave mAP 0.396 against 0.407 at 0.5, with 787 against 1,112 boxes; one run, so not established.

#flashcards/ml/concept #flashcards/topic/vision Why are boxes of different classes not compared in NMS?
?
A cat and a dog can sit in the same place (two real objects in one region, or a doubtful label), and both boxes should survive. In the toy a dog box at the cat's place was kept.

#flashcards/ml/concept #flashcards/topic/vision What are Soft-NMS and NMS-free detectors, and how sure is the note?
?
Soft-NMS lowers the score of an overlapping box instead of deleting it (Bodla and others, 2017); some detectors (DETR, 2020) predict a set of boxes directly so no NMS is needed. Both are from memory, not run in the vault.

#flashcards/ml/concept #flashcards/topic/vision What did Non-Maximum Suppression not check?
?
Only one detector, one run, one data set of 240 photos; no crowded scenes, no Soft-NMS, no NMS-free detector; torchvision's other defaults (score floor, box limit) were not printed and are from memory.

#flashcards/ml/cloze #flashcards/topic/vision Non-maximum suppression keeps the box with the ==highest score== and deletes boxes of the ==same class== that overlap it by more than a ==threshold==.
