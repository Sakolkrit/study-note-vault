---
tags: [computer-vision, deep-learning, detection, CNN]
status: growing
level: 4
tier: 2
---

# Object Detection

**Object detection** finds what is in a picture and **where**: it draws a box around each object and names it.

## 🟢 1. Simple (age 12)
Image classification answers "what is in this picture?". Object detection answers "what is in this picture, and where exactly?". The computer draws a box around every cat, car or person and writes the name next to it.

**Example:** A game of "I spy" on a busy street photo. You do not just say "there are cars"; you point at each car and say "one here, one over there, and a bus in the corner".

## 🔵 2. How it works
1. The input is a picture. The output is a list of **boxes**, each with a position (four numbers), a class name and a confidence.
2. The simplest idea: slide a small window over the picture and run an [[Image Classification]] network on every window. It works, but it needs thousands of passes.
3. Better: first suggest a few hundred boxes where an object might be (**region proposals**) and classify only those. This is the idea of [[R-CNN 2013]], made faster in [[Fast R-CNN 2015]].
4. Faster still: look at the whole picture once and predict all the boxes in one go ([[YOLO 2015]]).
5. The network usually produces many overlapping boxes for the same object. **Non-maximum suppression** keeps the highest-scoring box and removes the others that overlap it too much.
6. To judge a detector, each predicted box is compared with the true box by **intersection over union** (IoU). If the IoU is high enough (often 0.5), the box counts as correct, and the scores in [[Evaluation Metrics]] ([[Precision]], [[Recall]], average precision) are computed from that.
7. **The catch:** small and crowded objects are hard, and speed and accuracy pull against each other.

![[intersection-over-union.png]]
*Intersection over union (IoU): the area where two boxes overlap, divided by the area that the two boxes cover together. The picture shows two overlapping squares at the top, the shaded overlap above the line and the shaded union below it. Image: Adrian Rosebrock, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Intersection_over_Union_-_visual_equation.png), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Two families:** two-stage detectors (propose, then classify: [[R-CNN 2013]], [[Fast R-CNN 2015]], Faster R-CNN) and one-stage detectors (predict boxes directly: [[YOLO 2015]]). Two-stage was the more accurate and one-stage the faster, in the early papers.
- **Backbone:** the part that reads the picture is a [[Convolutional Neural Network]] such as [[VGG]] or [[ResNet]], often pre-trained for classification and then fine-tuned ([[Transfer Learning]], [[Fine-tuning]]).
- **Box regression:** the network predicts four numbers per box, a [[Regression]] problem, plus a class chance, a [[Classification]] problem. The training loss adds the two ([[Loss Function]]).
- **Proposals made by the network:** the Faster R-CNN abstract says its region proposal network predicts object bounds and objectness scores at each position, shares the full-image convolutional features ([[Feature Map]]) with the detector, and uses only 300 proposals per image. The fixed template boxes it uses, called anchors, are from memory.
- **Metrics:** mean average precision (mAP) averages the area under each class's [[Precision-Recall Curve]], with a box counted as right when IoU is above a threshold.
- **Related tasks:** [[Image Segmentation]] marks the exact pixels instead of a box.
- **Beyond:** the [[Vision Transformer]] has been used as a backbone too (from memory).

```python
import numpy as np
def iou(a, b):                                                         # boxes as (left, top, right, bottom)
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih; union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union
print("two 10x10 boxes, one shifted by 5 sideways: IoU =", round(iou((0, 0, 10, 10), (5, 0, 15, 10)), 3))
def nms(boxes, scores, thr=0.5):
    keep = []
    for i in np.argsort(scores)[::-1]:                                 # best score first; drop boxes that overlap a kept one too much
        if all(iou(boxes[i], boxes[j]) < thr for j in keep): keep.append(i)
    return keep
rng = np.random.default_rng(0)
objs = [(20, 20, 70, 70), (120, 40, 190, 120)]                        # two real objects
boxes, scores = [], []
for o in objs:
    for _ in range(25):                                                # 25 noisy guesses around each
        d = rng.normal(0, 4, 4); boxes.append(tuple(np.array(o) + d)); scores.append(0.9 - np.abs(d).sum() / 100)
keep = nms(boxes, np.array(scores))
print(f"{len(boxes)} overlapping guesses -> {len(keep)} boxes kept after non-maximum suppression")
W, H, win, stride = 640, 480, 64, 8
one = ((W - win) // stride + 1) * ((H - win) // stride + 1)
print(f"sliding a {win}x{win} window over {W}x{H} with stride {stride}: {one:,} windows at one size; with 5 window sizes up to {5 * one:,} (an overestimate), each window needing its own network pass")
print("grid method (YOLO paper's setting, from memory): 7 x 7 cells x 2 boxes =", 7 * 7 * 2, "boxes from one pass")
```

## 🔴 4. Deep / formal
**Intersection over union.** For a predicted box $A$ and a true box $B$:

$$
\text{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}=\frac{|A\cap B|}{|A|+|B|-|A\cap B|}
$$

**Where:**
- $A$ is the predicted box and $B$ the true box, as areas. $|A|$ means the area of $A$.
- $A\cap B$ is the overlap of the two boxes and $A\cup B$ is everything the two cover together. The second form says that the union is both areas added, minus the overlap that was counted twice.
- **Matters:** IoU is 1 for a perfect match and 0 when the boxes do not touch, and it punishes both a box that is too big and a box that is shifted. Without the division by the union, a huge box that covers the object would score as well as a perfect one.

**Example (tiny numbers):** two $10\times10$ boxes, $A$ from $x=0$ to $10$ and $B$ from $x=5$ to $15$, both from $y=0$ to $10$.
- Overlap: $5\times10=\mathbf{50}$. Areas: $100$ and $100$.
- Union: $100+100-50=\mathbf{150}$.
- IoU $=50/150=\mathbf{0.333}$.
- Meaning: half of each box overlaps, but the IoU is only a third, below the usual 0.5 bar, so this prediction would count as a miss.

**Non-maximum suppression (a rule, not a formula).** Sort the boxes by score. Keep the best box. Delete every other box whose IoU with it is above a threshold. Repeat with the next best box that is left.

**Further notes:**
- **Choice of the IoU bar:** 0.5 is a common choice (from memory); stricter bars give lower scores.
- **Speed vs accuracy:** YOLO's abstract reports 45 frames per second for the base model and 155 for the fast one, with more localisation errors than the slower systems.
- **Not covered here:** anchor design, feature pyramids, and detectors that predict box centres.

**What came before, and how this improved on it:**
- **Before:** classify every window of a sliding window, one network pass per window.
- **Its limit:** thousands of passes per picture, and the passes repeat most of the work.
- **What changed:** R-CNN classified region proposals with a CNN (53.3% mean average precision on VOC 2012, more than 30% relative improvement over the previous best; abstract read). Fast R-CNN shared the work: 9 times faster training, 213 times faster testing than R-CNN (abstract read). Faster R-CNN made the proposals cheap too, 5 frames per second with VGG-16 (arXiv 1506.01497, abstract read). YOLO predicted all boxes in one pass, 45 frames per second (abstract read).
- **How much better:** in the code below, a 64-pixel window slid over a 640 by 480 picture with stride 8 needs 3,869 windows at one size (up to 19,345 with 5 sizes, an overestimate), against 98 boxes from one pass of a 7 by 7 grid with 2 boxes per cell (the YOLO paper's setting, from memory). And non-maximum suppression turned 50 noisy overlapping guesses around 2 objects into 2 boxes.

**Check it yourself:** run the code in level 3 above. It prints every number quoted in this block.

## Related
[[Image Classification]] · [[Convolutional Neural Network]] · [[R-CNN 2013]] · [[Fast R-CNN 2015]] · [[YOLO 2015]] · [[Image Segmentation]] · [[VGG]] · [[ResNet]] · [[Evaluation Metrics]] · [[Precision-Recall Curve]] · [[Transfer Learning]] · [[History of Image Classification]]
- [[Faster R-CNN 2015]] · [[Mask R-CNN 2017]]
- [[Intersection over Union]] · [[Non-Maximum Suppression]] · [[Mean Average Precision]] · [[Tiny Object Detector]]
- [[MobileNet]]
## Sources
- Girshick et al., *Rich feature hierarchies...* (R-CNN), arXiv 1311.2524; Girshick, *Fast R-CNN*, arXiv 1504.08083; Ren et al., *Faster R-CNN*, arXiv 1506.01497; Redmon et al., *You Only Look Once*, arXiv 1506.02640: abstracts read; first authors from memory.
- The YOLO grid of 7 by 7 cells and 2 boxes, and the 0.5 IoU bar, are from memory.
- Image: [Intersection_over_Union_-_visual_equation.png](https://commons.wikimedia.org/wiki/File:Intersection_over_Union_-_visual_equation.png) by Adrian Rosebrock, CC BY-SA 4.0.
- Every number in the history block was computed by the code in this note (toy data); dates and names marked "from memory" are not checked.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision Object detection:::Answers "what is in this picture, and where exactly?" by drawing a box around every cat, car or person and writing the name next to it. Classification only answers "what is in it?".

#flashcards/ml/formula #flashcards/topic/vision Write intersection over union and explain each symbol.
?
$$\text{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}=\frac{|A\cap B|}{|A|+|B|-|A\cap B|}$$
- $A$: the predicted box. $B$: the true box, as areas. $|A|$ is the area of $A$.
- $A\cap B$: the overlap. $A\cup B$: everything the two cover together. The second form says the union is both areas added, minus the overlap that was counted twice.
- IoU is 1 for a perfect match and 0 when the boxes do not touch, and it punishes both a box that is too big and one that is shifted. Without dividing by the union, a huge box covering the object would score as well as a perfect one.

#flashcards/ml/example #flashcards/topic/vision Two $10\times10$ boxes, $A$ from $x=0$ to 10 and $B$ from $x=5$ to 15 (same $y$ range). What is IoU?::Overlap $5\times10=50$. Union $100+100-50=150$. IoU $=0.333$. Half of each box overlaps, but IoU is only a third, below the usual 0.5 bar, so this prediction counts as a miss.

#flashcards/ml/image #flashcards/topic/vision ![[intersection-over-union.png]]
What does this picture show?
?
Intersection over union (IoU): the area where two boxes overlap divided by the area that the two boxes cover together. The picture shows two overlapping squares at the top, the shaded overlap above the line and the shaded union below it. (From the note's caption: Adrian Rosebrock, Wikimedia Commons.)

#flashcards/ml/cloze #flashcards/topic/vision A prediction counts as a hit when its IoU is above about ==0.5==.
