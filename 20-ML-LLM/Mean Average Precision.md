---
tags: [computer-vision, object-detection, evaluation, metrics]
status: growing
level: 4
tier: 2
---

# Mean Average Precision

**Mean average precision (mAP)** is the standard score of an [[Object Detection]] model. It asks, for every class, how well the model ranks its right boxes above its wrong ones, then averages over the classes (and, in COCO's version, over how tight the boxes must be).

## 🟢 1. Simple (age 12)
A treasure-hunting team shouts out finds, most-sure first. A find counts only if it is in the right spot. A good team's most-sure shouts are nearly always right, and it finds most of the treasure. The score for one kind of treasure is how good that ranking is. The **mean** of those scores over all the kinds of treasure is the mAP.

**Example:** your team is great at finding coins but bad at finding rings. A plain average of coin and ring scores shows both, so a good coin score cannot hide the poor ring score.

## 🔵 2. How it works
1. Take one class, say "cat". Collect every cat detection from every test photo and **sort by score**, highest first.
2. Go down the list. A detection is a **hit** if its [[Intersection over Union]] with a true cat box is at least the **bar** (say 0.5) and that true box has not been used already. Otherwise it is a **false alarm**. (This is why duplicates hurt: see [[Non-Maximum Suppression]].)
3. After each detection work out [[Precision]] (how many of the detections so far were hits) and [[Recall]] (how many of the true cats have been found so far). This gives a [[Precision-Recall Curve]].
4. The **average precision (AP)** of the class is the area under that curve, one number per class and bar.
5. **mAP at one bar** is the plain average of the class APs.
6. COCO's headline number repeats this for ten bars, 0.50, 0.55, up to 0.95, and averages them too. Loose bars reward finding the object; strict bars reward drawing a tight box.
7. Always say **which** mAP: "mAP at 0.5" and "mAP over 0.5 to 0.95" are different numbers for the same detector.

*I found no free, on-topic picture of mAP on Wikimedia Commons (a search for it returned scanned papers), so this note has no picture; the curve picture is in [[Precision-Recall Curve]].*

## 🟠 3. Technical detail
- **Matching rule.** Detections are taken in order of score; each true box can be used once. A second box on an already-used true box is a false alarm. A class with no detections has AP 0.
- **How the area is taken.** The simple sum formula in [[Precision-Recall Curve]] gives one number. COCO's tool first replaces the precision at each recall by the best precision at that recall or higher (the "envelope") and then averages it at 101 recall values from 0 to 1 (from memory). The toy below shows both on one small case and compares them with the tool.
- **Limits per photo.** COCO's tool looks at the best 100 detections per photo and can split results by object size (small, medium, large); I used the "all sizes" number.
- **Different versions exist** (an older 11-point version, the all-point version and COCO's 101-point one; from memory), so numbers from different papers are only comparable when the version is the same.

A tiny case with five detections for one class and four true boxes (the first, second and fourth detections are hits, the other two are false alarms, and the fourth true box is never found), checked against `pycocotools`:

```python
# file: map_toy.py
import io, contextlib, numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# one class, 4 true boxes; detections sorted by score: hit, hit, false alarm, hit, false alarm (the 4th true box is never found)
tp = np.array([1, 1, 0, 1, 0]); n_true = 4
prec = np.cumsum(tp) / np.arange(1, len(tp) + 1); rec = np.cumsum(tp) / n_true
print("precision after each detection:", np.round(prec, 3)); print("recall after each detection:   ", np.round(rec, 3))
ap_sum = sum(prec[i] * (rec[i] - (rec[i - 1] if i else 0)) for i in range(len(tp)))            # the formula of Precision-Recall Curve
env = np.maximum.accumulate(prec[::-1])[::-1]                                                    # precision envelope: best precision at this recall or higher
ap101 = np.mean([env[rec >= r][0] if (rec >= r).any() else 0 for r in np.linspace(0, 1, 101)])   # COCO style: 101 recall points
print(f"AP, sum formula {ap_sum:.4f}   AP, COCO style (envelope, 101 recall points) {ap101:.4f}")

# the same case through pycocotools
sq = lambda x: [x, 0, 40, 40]
gt = {"images": [{"id": 1, "width": 500, "height": 100}], "categories": [{"id": 1, "name": "a"}],
      "annotations": [{"id": i + 1, "image_id": 1, "category_id": 1, "bbox": sq(100 * i), "area": 1600, "iscrowd": 0} for i in range(4)]}
hits = [0, 1, 2]; wrong = [sq(450), sq(470)]                       # boxes 0, 1, 3 are found exactly; two boxes far from everything
dets = [(0.9, sq(0)), (0.8, sq(100)), (0.7, wrong[0]), (0.6, sq(300)), (0.5, wrong[1])]
with contextlib.redirect_stdout(io.StringIO()):
    G = COCO(); G.dataset = gt; G.createIndex()
    D = G.loadRes([{"image_id": 1, "category_id": 1, "bbox": b, "score": s} for s, b in dets])
    E = COCOeval(G, D, "bbox"); E.params.iouThrs = np.array([0.5]); E.evaluate(); E.accumulate(); E.summarize()
print(f"pycocotools AP at IoU 0.5: {E.stats[0]:.4f}")

# mean over classes, and over IoU bars (the three class scores are from Tiny Object Detector, IoU 0.5)
c = np.array([0.808, 0.669, 0.607]); print(f"mAP at IoU 0.5 = mean of {c.tolist()} = {c.mean():.3f}")
b = np.array([0.695, 0.424]); print(f"average of the IoU-0.5 and IoU-0.75 scores {b.tolist()} = {b.mean():.3f}")
```

```text
precision after each detection: [1.    1.    0.667 0.75  0.6  ]
recall after each detection:    [0.25 0.5  0.5  0.75 0.75]
AP, sum formula 0.6875   AP, COCO style (envelope, 101 recall points) 0.6906
pycocotools AP at IoU 0.5: 0.6906
mAP at IoU 0.5 = mean of [0.808, 0.669, 0.607] = 0.695
average of the IoU-0.5 and IoU-0.75 scores [0.695, 0.424] = 0.559
```

Reading it: after the five detections the precision runs 1, 1, 0.667, 0.75, 0.6 while the recall climbs to 0.75 and stops, because the fourth true box is never found. The sum formula gives 0.6875 and COCO's method 0.6906, the value `pycocotools` gives too. The two ways of taking the area are close but not equal.

Now the real detector of [[Tiny Object Detector]]: AP of each class at each of the ten bars, and their average. It needs `detector.pt`, `train.py` and `data/` from that project, and its virtual environment:

```python
# file: map_by_iou.py
"""Where does the detector's mAP come from? AP at each IoU bar, and per class (needs detector.pt, train.py, data/)."""
import io, contextlib, torch, numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import train as T

torch.set_num_threads(4)
val = T.meta["val"]; names = T.meta["classes"]
gt = {"images": [], "annotations": [], "categories": [{"id": i, "name": n} for i, n in enumerate(names, 1)]}
k = 1
for j, r in enumerate(val):
    gt["images"].append({"id": j, "width": r["w"], "height": r["h"]})
    for x0, y0, x1, y1, c in r["boxes"]:
        gt["annotations"].append({"id": k, "image_id": j, "category_id": c, "bbox": [x0, y0, x1 - x0, y1 - y0],
                                  "area": (x1 - x0) * (y1 - y0), "iscrowd": 0}); k += 1
m = T.build(); m.load_state_dict(torch.load("detector.pt")); m.eval()
ds = T.DS(val, False); dets = []
with torch.no_grad():
    for j in range(len(ds)):
        o = m([ds[j][0]])[0]
        for b, s, l in zip(o["boxes"].tolist(), o["scores"].tolist(), o["labels"].tolist()):
            dets.append({"image_id": j, "category_id": l, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]], "score": s})
with contextlib.redirect_stdout(io.StringIO()):
    G = COCO(); G.dataset = gt; G.createIndex()
    E = COCOeval(G, G.loadRes(dets), "bbox"); E.evaluate(); E.accumulate(); E.summarize()
p = E.eval["precision"][:, :, :, 0, 2]            # IoU bars x recall points x classes (all sizes, 100 boxes per photo)
ap = np.array([[p[t, :, c][p[t, :, c] > -1].mean() for c in range(3)] for t in range(10)])   # bars x classes
for t in range(10):
    print(f"IoU bar {0.5 + 0.05 * t:.2f}: " + "  ".join(f"{n} {ap[t, c]:.3f}" for c, n in enumerate(names)) + f"   mean {ap[t].mean():.3f}")
print(f"mean over all 10 bars and 3 classes {ap.mean():.3f}   (pycocotools says {E.stats[0]:.3f})")
```

```text
IoU bar 0.50: cat 0.808  dog 0.669  bus 0.607   mean 0.695
IoU bar 0.55: cat 0.764  dog 0.656  bus 0.604   mean 0.674
IoU bar 0.60: cat 0.709  dog 0.596  bus 0.567   mean 0.624
IoU bar 0.65: cat 0.635  dog 0.542  bus 0.535   mean 0.571
IoU bar 0.70: cat 0.585  dog 0.427  bus 0.505   mean 0.505
IoU bar 0.75: cat 0.493  dog 0.313  bus 0.466   mean 0.424
IoU bar 0.80: cat 0.365  dog 0.216  bus 0.417   mean 0.333
IoU bar 0.85: cat 0.130  dog 0.138  bus 0.316   mean 0.195
IoU bar 0.90: cat 0.033  dog 0.036  bus 0.072   mean 0.047
IoU bar 0.95: cat 0.000  dog 0.000  bus 0.003   mean 0.001
mean over all 10 bars and 3 classes 0.407   (pycocotools says 0.407)
```

Reading it: at the loose bar of 0.50 the detector scores 0.695; at 0.75 it is 0.424, at 0.90 only 0.047 and at 0.95 almost 0.001. The headline 0.407 is the average of all ten bars and three classes, and it equals the number `pycocotools` prints. The detector finds the objects, but its boxes are rarely tight enough for the strict bars. The order of the classes changes with the bar (bus is last at 0.50 but second at 0.75), and with one run and 90 to 113 true boxes per class I do not read much into that.

## 🔴 4. Deep / formal
**Mean over classes at one bar $t$:**

$$
\text{mAP}_t=\frac{1}{C}\sum_{c=1}^{C}\text{AP}_{c,t}
$$

**Where:**
- $C$ is the number of classes (3 in my project).
- $\text{AP}_{c,t}$ is the average precision of class $c$ when a hit needs an IoU of at least $t$.
- **Matters:** every class gets the same say, however many objects it has. Without the average over classes, a common class would hide a rare class that the detector never finds.

**Example (tiny numbers):** the three class scores at the 0.5 bar are cat 0.808, dog 0.669 and bus 0.607.
- Sum: $0.808+0.669+0.607=\mathbf{2.084}$.
- Divide by $C=3$: $\text{mAP}_{0.5}=\mathbf{0.695}$.
- Meaning: on average over the three classes, the ranking is good when "right" only needs half-overlap.

**COCO's headline number averages the bars too:**

$$
\text{mAP}=\frac{1}{|T|}\sum_{t\in T}\text{mAP}_t,\qquad T=\{0.50,\,0.55,\,\dots,\,0.95\}
$$

**Where:**
- $T$ is the set of ten IoU bars and $|T|=10$ is how many there are.
- $\text{mAP}_t$ is the score above for bar $t$.
- **Matters:** the strict bars make the number reward tight boxes. Without the average over $T$, a detector with sloppy boxes would score as well as one with tight boxes.

**Example (tiny numbers):** use only two bars, 0.5 and 0.75, whose scores for my detector are 0.695 and 0.424.
- Sum: $0.695+0.424=\mathbf{1.119}$. Divide by 2: $\mathbf{0.559}$ (printed to three places by the code).
- Meaning: a detector that is good at the loose bar but poor at the tight one lands in between. The real average over all ten bars is 0.407.

**Limits:**
- **One number hides the kind of mistake.** A miss, a wrong class and a sloppy box all lower it. In [[Tiny Object Detector]] the [[Confusion Matrix]] showed misses were the main problem; mAP alone does not say that.
- **It depends on the data and the bars.** A different mix of classes, object sizes or bars changes it, so compare only with the same setup.
- **No error bar.** On 240 photos the number has noise ([[Confidence Interval]], [[Bootstrap]]); I did not compute it.

**What came before, and how this improved on it:**
- **Before:** judge a detector by precision and recall at a single score threshold, or by a plain accuracy.
- **Its limit:** one threshold hides the trade-off between finding objects and raising false alarms, and a hit-or-miss rule says nothing about how tight the boxes are.
- **What changed:** average precision per class over every threshold (the area under the [[Precision-Recall Curve]]), averaged over classes; the PASCAL VOC contest used a single IoU bar of 0.5 (from memory) and COCO averages ten bars (from memory; I checked that the tool's default average over ten bars gives 0.407, equal to my own average).
- **How much better:** the loose number hides what the strict one shows. My detector scores 0.695 at the 0.5 bar and 0.047 at 0.9; the single 0.695 alone would not tell you its boxes are loose.

## Related
- [[Object Detection]] · [[Intersection over Union]] · [[Non-Maximum Suppression]] · [[Precision-Recall Curve]] · [[Precision]] · [[Recall]] · [[Confusion Matrix]] · [[Evaluation Metrics]] · [[Tiny Object Detector]] · [[Confidence Interval]] · [[Faster R-CNN 2015]]

## Sources
- My own toy and runs (code above; `pycocotools` 2.0.11 computed the check). The envelope and 101-point details, the older 11-point version, the PASCAL VOC bar of 0.5 and the 100-detections limit are from memory; I read no paper or documentation for this note. Every code block in this note was extracted into a fresh folder (with a copy of the files from [[Tiny Object Detector]] for the blocks that need them) and run again; each output matched the block above. Not tested: another machine or other library versions.
- No image (see level 2).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation Mean average precision (mAP):::The average, over classes (and for COCO also over ten IoU bars from 0.5 to 0.95), of each class's average precision, the area under its precision-recall curve for detections ranked by score.

#flashcards/ml/formula #flashcards/topic/evaluation Write the mAP at one IoU bar and explain each symbol.
?
$$\text{mAP}_t=\frac{1}{C}\sum_{c=1}^{C}\text{AP}_{c,t}$$
- $C$: the number of classes.
- $\text{AP}_{c,t}$: the average precision of class $c$ when a hit needs IoU at least $t$.
- Every class gets the same say however many objects it has; without the average a common class would hide a rare class that is never found.

#flashcards/ml/example #flashcards/topic/evaluation Class scores at the 0.5 bar are cat 0.808, dog 0.669 and bus 0.607. What is the mAP at that bar?::The sum is 0.808 + 0.669 + 0.607 = 2.084, divided by the 3 classes gives 0.695 (my arithmetic, equal to the detector's printed value).

#flashcards/ml/formula #flashcards/topic/evaluation Write COCO's headline mAP and explain each symbol.
?
$$\text{mAP}=\frac{1}{|T|}\sum_{t\in T}\text{mAP}_t,\qquad T=\{0.50,\,0.55,\,\dots,\,0.95\}$$
- $T$: the ten IoU bars, and $|T|=10$ is how many.
- $\text{mAP}_t$: the mAP at bar $t$.
- The strict bars reward tight boxes; without the average over $T$ a detector with sloppy boxes would score as well as a tight one.

#flashcards/ml/example #flashcards/topic/evaluation Two bars, 0.5 and 0.75, have mAP 0.695 and 0.424. What is their average?::0.695 + 0.424 = 1.119, divided by 2 gives 0.559 (my arithmetic). The real average over ten bars for that detector is 0.407.

#flashcards/ml/concept #flashcards/topic/evaluation What are the steps of computing mAP?
?
For each class sort its detections by score; mark each a hit (IoU with an unused true box of that class at least the bar) or a false alarm; compute precision and recall down the list to get the precision-recall curve; AP is the area under it; average over classes, and for COCO also over ten IoU bars.

#flashcards/ml/concept #flashcards/topic/evaluation How did the two ways of taking the area differ in the toy case?
?
Five detections (hit, hit, false alarm, hit, false alarm) with four true boxes: the sum formula gave AP 0.6875 and COCO's method (precision envelope, 101 recall points) 0.6906, which is the value pycocotools gave. Close but not equal, so compare only the same version.

#flashcards/ml/concept #flashcards/topic/evaluation How did the Tiny Object Detector's mAP change with the IoU bar?
?
0.695 at 0.50, 0.424 at 0.75, 0.047 at 0.90 and 0.001 at 0.95; the average over ten bars and three classes is 0.407, equal to pycocotools. Its boxes are found but rarely tight. My own run, one detector, 240 photos.

#flashcards/ml/concept #flashcards/topic/evaluation What are the limits of mAP?
?
One number hides the kind of mistake (misses, wrong class, loose box); it depends on the classes, the sizes and the bars, so compare only the same version and setup; and it has noise on a small test set (no error bar was computed here).

#flashcards/ml/cloze #flashcards/topic/evaluation COCO's mAP averages the ==average precision== over ==classes== and over several ==IoU bars== from 0.5 to 0.95.
