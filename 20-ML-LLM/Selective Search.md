---
tags: [selective-search, object-detection, region-proposals, computer-vision, segmentation, ML]
status: growing
level: 4
tier: 2
---

# Selective Search

**Selective Search** is a way to suggest a few thousand boxes in a picture where an object might be, without knowing what the object is. It starts from many tiny regions that look alike inside, then repeatedly merges the two most similar neighbouring regions. Every region that appears during the merging gives one box, so boxes of all sizes come out. [[R-CNN 2013]] used it to decide which parts of a picture the network should look at.

## 🟢 1. Simple (age 12)
Imagine a jigsaw puzzle already cut into hundreds of tiny pieces. You glue together the two pieces that look most alike (same colour, same texture), then glue the new bigger piece to its best neighbour, again and again until the whole picture is one piece. Every time you glued something, you can draw a rectangle around the glued piece. Some rectangles hold a face, some a whole person, some the entire scene. Those rectangles are the "places to look".

**Example:** in a portrait of a person in front of a flag, the tiny pieces of the face join into a face, the face and hair join into a head, the head and the suit join into a person. The rectangles around the face, the head and the person are all proposals.

## 🔵 2. How it works
1. **Start small.** Cut the picture into many tiny regions that are each almost one colour, with a fast method by Felzenszwalb and Huttenlocher (an "over-segmentation").
2. **Score the neighbours.** For every pair of touching regions, compute how similar they are. The paper uses four fast measures, each between 0 and 1: **colour** (how much their colour histograms overlap), **texture** (the same for a texture histogram), **size** (small regions score higher, so they merge early and no single region swallows everything) and **fill** (how well the two pieces fit together inside their joint box, so gaps get filled).
3. **Merge the best pair.** Glue the two most similar regions into one, work out its similarities to its neighbours, and repeat until only one region is left.
4. **Collect boxes.** Draw a box around every region that ever existed, small and large. A picture with $n$ starting regions gives $2n-1$ regions.
5. **Do it several times in different ways** (different colour spaces, different mixes of similarity measures, different starting regions) and pool the boxes, because no single way of grouping works for everything.
6. **Sort and clean.** Rank the pooled boxes (with some randomness so that large regions do not dominate), then drop duplicates.
7. **Use.** A detector then looks only at these boxes ([[R-CNN 2013]], [[Fast R-CNN 2015]]), and ranks its answers with [[Non-Maximum Suppression]].

No image: I did not find a free picture of Selective Search at work. The numbers below come from a real run on a NASA photo shipped with scikit-image.

## 🟠 3. Technical detail
Numbers come from `ss_demo.py` (Python 3.9.6, `opencv-contrib-python-headless` 4.8.0, scikit-image 0.19.3, 1 thread, CPU). It runs OpenCV's implementation of Selective Search on the 512 × 512 "astronaut" photo of scikit-image (a NASA photograph, public domain per the library's data notes), counts the boxes, and checks how well they cover seven boxes **I drew by eye** on the photo. Those seven boxes are only my own rough ground truth. The script also runs a tiny hand-made grouping of six regions.

```python
"""Selective Search with OpenCV (opencv-contrib 4.8): proposal counts, time, best overlap with hand-drawn boxes; plus a tiny hand-made hierarchical grouping."""
import time, cv2, numpy as np
from skimage import data
img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)          # 512 x 512, NASA photo shipped with scikit-image
cv2.setNumThreads(1)
# my own hand-drawn "ground truth" boxes (x1, y1, x2, y2), eyeballed from the picture
gt = {"head": (150, 15, 305, 195), "face": (172, 70, 275, 190), "astronaut": (15, 15, 365, 512),
      "helmet": (275, 343, 512, 512), "shuttle model": (353, 0, 470, 285), "flag": (0, 0, 97, 330), "mission patch": (133, 348, 210, 424)}
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix*iy
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(img)
res = {}
for mode in ("single", "fast", "quality"):
    {"single": ss.switchToSingleStrategy, "fast": ss.switchToSelectiveSearchFast, "quality": ss.switchToSelectiveSearchQuality}[mode]()
    t = time.time(); r = ss.process(); dt = time.time() - t
    boxes = [(x, y, x+w, y+h) for x, y, w, h in r]; res[mode] = boxes
    print(mode, "boxes", len(boxes), "time %.2fs" % dt)
for mode, boxes in res.items():
    print("\n", mode, "- best IoU with each hand box, using the first N boxes in the returned order")
    for N in (50, 200, 1000, len(boxes)):
        b = boxes[:N]; best = {k: max(iou(v, p) for p in b) for k, v in gt.items()}
        print("  N=%5d" % min(N, len(boxes)), {k: round(v, 2) for k, v in best.items()}, "mean %.3f" % np.mean(list(best.values())),
              "found at 0.5: %d of %d" % (sum(v >= 0.5 for v in best.values()), len(gt)))
# duplicates and sizes
for mode, boxes in res.items():
    ws = np.array([b[2]-b[0] for b in boxes]); hs = np.array([b[3]-b[1] for b in boxes])
    print(mode, "unique boxes", len(set(boxes)), "| width min/median/max", ws.min(), int(np.median(ws)), ws.max(), "| boxes under 20 px wide:", int((ws < 20).sum()))
# exhaustive-search comparison: boxes of all positions and sizes on a stride-8 grid with 3 aspect ratios per size
cnt = 0
for s in (32, 64, 96, 128, 192, 256, 384, 512):
    for ar in (0.5, 1.0, 2.0):
        w = int(s*np.sqrt(ar)); h = int(s/np.sqrt(ar))
        if w <= 512 and h <= 512: cnt += ((512-w)//8 + 1) * ((512-h)//8 + 1)
print("\nsliding-window boxes on a stride-8 grid, 8 sizes x 3 shapes:", cnt)
# --- tiny hierarchical grouping on 6 regions in a row (hand-made: colour histogram of 3 bins, size in pixels, x-range)
H = [np.array(h, float) for h in ([.9, .1, 0], [.8, .2, 0], [.1, .8, .1], [.0, .9, .1], [.0, .2, .8], [.1, .1, .8])]
size = [100, 80, 120, 60, 90, 150]; box = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]    # 1-D "boxes" (left, right)
ids = list(range(6)); imgsize = sum(size)
regions = {i: (H[i]/H[i].sum(), size[i], box[i]) for i in ids}
adj = {(i, i+1) for i in range(5)}; nxt = 6; log = []
def sim(a, b):
    ha, sa, ba = regions[a]; hb, sb, bb = regions[b]
    return np.minimum(ha, hb).sum() + (1 - (sa + sb)/imgsize)        # colour intersection + size (two of the paper's four measures)
while adj:
    a, b = max(adj, key=lambda p: sim(*p)); s = sim(a, b)
    ha, sa, ba = regions[a]; hb, sb, bb = regions[b]
    regions[nxt] = ((sa*ha + sb*hb)/(sa + sb), sa + sb, (min(ba[0], bb[0]), max(ba[1], bb[1])))
    log.append((a, b, nxt, round(s, 3), regions[nxt][2]))
    new = set()
    for p in adj:
        if set(p) & {a, b}:
            other = [x for x in p if x not in (a, b)]
            if other: new.add((other[0], nxt))
        else: new.add(p)
    adj = {p for p in new}; nxt += 1
print("\ntoy grouping: merge (a, b -> new, similarity, span)")
for l in log: print(" ", l)
print("regions in the hierarchy:", len(regions), "= 2 x 6 - 1 =", 2*6 - 1)
```

I saw:
```
single boxes 516 time 0.40s
fast boxes 2418 time 3.60s
quality boxes 8821 time 15.35s

 single - best IoU with each hand box, using the first N boxes in the returned order
  N=   50 {'head': 0.02, 'face': 0.03, 'astronaut': 0.16, 'helmet': 0.2, 'shuttle model': 0.08, 'flag': 0.34, 'mission patch': 0.27} mean 0.158 found at 0.5: 0 of 7
  N=  200 {'head': 0.47, 'face': 0.94, 'astronaut': 0.58, 'helmet': 0.67, 'shuttle model': 0.3, 'flag': 0.34, 'mission patch': 0.96} mean 0.609 found at 0.5: 4 of 7
  N=  516 {'head': 0.9, 'face': 0.94, 'astronaut': 0.67, 'helmet': 0.7, 'shuttle model': 0.46, 'flag': 0.63, 'mission patch': 0.97} mean 0.754 found at 0.5: 6 of 7
  N=  516 {'head': 0.9, 'face': 0.94, 'astronaut': 0.67, 'helmet': 0.7, 'shuttle model': 0.46, 'flag': 0.63, 'mission patch': 0.97} mean 0.754 found at 0.5: 6 of 7

 fast - best IoU with each hand box, using the first N boxes in the returned order
  N=   50 {'head': 0.28, 'face': 0.14, 'astronaut': 0.45, 'helmet': 0.67, 'shuttle model': 0.7, 'flag': 0.23, 'mission patch': 0.11} mean 0.370 found at 0.5: 2 of 7
  N=  200 {'head': 0.92, 'face': 0.48, 'astronaut': 0.51, 'helmet': 0.67, 'shuttle model': 0.73, 'flag': 0.31, 'mission patch': 0.86} mean 0.640 found at 0.5: 5 of 7
  N= 1000 {'head': 0.92, 'face': 0.94, 'astronaut': 0.84, 'helmet': 0.91, 'shuttle model': 0.8, 'flag': 0.68, 'mission patch': 0.95} mean 0.863 found at 0.5: 7 of 7
  N= 2418 {'head': 0.93, 'face': 0.94, 'astronaut': 0.92, 'helmet': 0.97, 'shuttle model': 0.94, 'flag': 0.98, 'mission patch': 0.97} mean 0.949 found at 0.5: 7 of 7

 quality - best IoU with each hand box, using the first N boxes in the returned order
  N=   50 {'head': 0.26, 'face': 0.15, 'astronaut': 0.52, 'helmet': 0.81, 'shuttle model': 0.47, 'flag': 0.7, 'mission patch': 0.18} mean 0.443 found at 0.5: 3 of 7
  N=  200 {'head': 0.91, 'face': 0.69, 'astronaut': 0.69, 'helmet': 0.81, 'shuttle model': 0.7, 'flag': 0.7, 'mission patch': 0.96} mean 0.781 found at 0.5: 7 of 7
  N= 1000 {'head': 0.91, 'face': 0.9, 'astronaut': 0.87, 'helmet': 0.85, 'shuttle model': 0.73, 'flag': 0.75, 'mission patch': 0.97} mean 0.854 found at 0.5: 7 of 7
  N= 8821 {'head': 0.94, 'face': 0.94, 'astronaut': 0.96, 'helmet': 1.0, 'shuttle model': 0.94, 'flag': 0.99, 'mission patch': 0.97} mean 0.963 found at 0.5: 7 of 7
single unique boxes 516 | width min/median/max 1 38 512 | boxes under 20 px wide: 112
fast unique boxes 2418 | width min/median/max 1 49 512 | boxes under 20 px wide: 404
quality unique boxes 8821 | width min/median/max 1 57 512 | boxes under 20 px wide: 1018

sliding-window boxes on a stride-8 grid, 8 sizes x 3 shapes: 43210

toy grouping: merge (a, b -> new, similarity, span)
  (0, 1, 6, 1.6, (0, 20))
  (2, 3, 7, 1.6, (20, 40))
  (4, 5, 8, 1.5, (40, 60))
  (6, 7, 9, 0.611, (0, 40))
  (8, 9, 10, 0.25, (0, 60))
regions in the hierarchy: 11 = 2 x 6 - 1 = 11
```

**Boxes and time for OpenCV's three modes (my run).**

| Mode | Boxes | Time (s) | Boxes under 20 px wide |
|---|---|---|---|
| single strategy | 516 | 0.40 | 112 |
| fast | 2,418 | 3.60 | 404 |
| quality | 8,821 | 15.35 | 1,018 |

- **Mode:** which preset of OpenCV's `switchTo...` functions was used (a single grouping strategy, or the many-strategy "fast" and "quality" presets).
- **Boxes:** how many distinct boxes came back (all were unique). **Time:** one run, one thread, my laptop CPU; not a benchmark.
- For comparison, a plain sliding window on a stride-8 grid with 8 sizes and 3 shapes gives 43,210 boxes on this picture (my count in the script, my own choice of sizes).

**Best overlap with my seven hand-drawn boxes.** The script takes the first $N$ boxes in the order OpenCV returns them and, for each hand box, reports the best [[Intersection over Union]]:

| Mode | Boxes used | Mean best IoU | Hand boxes found at IoU 0.5 or more |
|---|---|---|---|
| single | all 516 | 0.754 | 6 of 7 |
| fast | first 200 | 0.640 | 5 of 7 |
| fast | first 1,000 | 0.863 | 7 of 7 |
| fast | all 2,418 | 0.949 | 7 of 7 |
| quality | first 200 | 0.781 | 7 of 7 |
| quality | all 8,821 | 0.963 | 7 of 7 |

- **Mean best IoU:** for each hand box the best IoU over the boxes used, averaged over the seven (the same idea as the paper's "Average Best Overlap", on one picture).
- **Found at 0.5:** the usual PASCAL "found it" rule.

**The tiny grouping (my toy).** Six regions in a row with 3-bin colour histograms. The script's similarity is colour overlap plus the size term (two of the paper's four measures). The merge order was: regions 0 and 1 (similarity 1.6), 2 and 3 (1.6), 4 and 5 (1.5), then (0,1) with (2,3) (0.611), then the last two (0.25). That gave $2\times6-1=11$ regions, so 11 boxes at four different sizes.

**Summary: what the run shows.** More strategies give more boxes and a better cover: from 516 boxes (6 of 7 hand boxes found) to 2,418 (7 of 7, mean best IoU 0.949) to 8,821 (7 of 7, 0.963). Cost grows too (0.40 s, 3.6 s, 15 s). Caution: one photo, seven boxes drawn by me, one thread, and OpenCV returns the boxes in an order that may not be the paper's ranking (the paper's ranking uses a random factor; I did not check OpenCV's), so the "first $N$" rows say little about ranking quality. This is not a measurement of the paper's recall.

## 🔴 4. Deep / formal
No equation: the method is a greedy loop with a handful of similarity rules, which the steps and the toy run reproduce. The paper's four rules in words: the colour and texture similarity is the histogram intersection (the sum of the smaller of two bin values), the size similarity is one minus the pair's share of the image, and the fill similarity is one minus the share of the joint box left uncovered.

**What the paper says.** I read the authors' technical-report draft ("Technical Report 2012, submitted to IJCV", 14 pages, from the first author's web page), not the published IJCV version, so numbers may differ from the journal paper. Its abstract: "99% recall and a Mean Average Best Overlap of 0.879 at 10,097 locations". Its design aims: capture all scales (a hierarchy does it naturally), be diverse (no single grouping fits all images; Figure 1 of the paper gives colour, texture and "enclosed" examples) and be fast.

| Setting (paper, Tables 3 to 5, VOC 2007) | Result |
|---|---|
| Single measure, MABO: texture / colour, size or fill | 0.581 / about 0.63 to 0.64 |
| All four similarity measures together | 0.676 |
| One colour space: C / HSV | 0.615 (125 boxes) / 0.693 (463 boxes) |
| "Fast" selective search | recall 0.980, MABO 0.804, 2,134 boxes |
| "Quality" selective search | recall 0.991, MABO 0.879, 10,097 boxes |
| Exhaustive search of the part-based detector (class-specific boxes) | recall 0.933, MABO 0.829, 100,352 boxes per class |

*Written by me, an AI, from the draft's text; per-class results, the other competitors and the figure data are left out.*
- **MABO** is the Mean Average Best Overlap: for each class, the average over its ground-truth objects of the best overlap any proposal reaches, then averaged over classes. Higher is better, and it rewards boxes that fit tightly, which a recall at 0.5 overlap cannot.
- **Recall** is the share of objects with at least one box above 0.5 overlap.

**What the paper concludes.** A hierarchy beats many flat segmentations at a similar number of boxes; combining many diverse strategies is better than one (the greedy search picked varied similarity mixes for the quality version); texture alone is the weakest, which the authors guess is because both sides of an object boundary have similar edge responses.

**Where the draft disagrees with itself.** The abstract and Table 5 give 10,097 locations for the quality setting while Table 4 gives 10,108; the abstract rounds recall to 99% while Table 5 shows 0.991. I read this as small drafting slips, not a different result.

**What R-CNN did with it.** [[R-CNN 2013]] used the "fast mode" to get about 2,000 boxes per image, and on the 200-class ILSVRC 2013 data it measured 2,403 boxes per image with 91.6% recall at 0.5 overlap (against about 98% on PASCAL), so the proposal step limited the detector (R-CNN paper, section 4.2). [[Faster R-CNN 2015]] calls Selective Search "about 2 seconds per image on a CPU" and replaced it with a learned proposal network.

**Limits and trade-offs:**
- It is slow compared with the detector that follows it (seconds per image on a CPU, even in the paper's days), and it runs outside the network, so it cannot be trained with it.
- Its boxes come from colour and texture cues only; objects without clear boundaries or with parts of different colours are hard (the paper's own examples show the problem in Figure 1).
- Many hand-set parts: 25 colour bins, the 8 colour spaces, thresholds of the starting segmentation, the 80 strategies of the quality setting.
- Recall is capped by the starting segmentation: a region that is never separated is never a box.

**What came before, and how this improved on it:**
- **Before:** an exhaustive sliding window with fixed grid, scales and shapes, or a single segmentation algorithm, or boxes sampled by an "objectness" score (paper, 1 and 2).
- **Its limit:** the exhaustive search visits a huge number of boxes, which forces weak classifiers; a single segmentation misses objects that group for other reasons than colour or texture.
- **What changed:** use bottom-up grouping to propose boxes at all scales, and diversify the grouping so that the union covers many image conditions.
- **How much better:** in the paper, 0.933 recall at 100,352 boxes per class for the exhaustive search of the part-based detector against 0.991 recall at 10,097 boxes (all classes) for selective search quality, a tenfold cut in boxes (the draft's own comparison, Table 5). In my own run on one photo, 8,821 boxes against 43,210 for a stride-8 sliding window, with 7 of 7 hand boxes found by both counts of the selective modes at an IoU of 0.5; that is one photo and my own box set, not a comparison of recall.

## Related
[[R-CNN 2013]] · [[Fast R-CNN 2015]] · [[Faster R-CNN 2015]] · [[Object Detection]] · [[Intersection over Union]] · [[Non-Maximum Suppression]] · [[Image Segmentation]] · [[Anchor Box]] · [[Mean Average Precision]] · [[Hard Negative Mining]]

## Sources
- My own run: `ss_demo.py` (Python 3.9.6, opencv-contrib 4.8.0, scikit-image 0.19.3). The hand-drawn boxes are mine.
- Paper: Uijlings, van de Sande, Gevers and Smeulders, "Selective Search for Object Recognition", author-page draft ("Technical Report 2012, submitted to IJCV"; huppelen.nl/publications/selectiveSearchDraft.pdf). I read the text of the draft: introduction, related work, section 3 (the algorithm and its diversification), the evaluation sections and Tables 2 to 5. Not read: the object-recognition and per-class sections in detail, the figures as pictures, the published IJCV version, the later "fast mode" settings OpenCV uses.
- The R-CNN statements come from [[R-CNN 2013]] and [[Faster R-CNN 2015]] as read there. Not checked: how OpenCV's presets map to the paper's configurations. The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision Selective Search:::A class-independent way to propose a few thousand boxes per image by greedily merging the most similar neighbouring regions of an over-segmentation, so that boxes of all sizes appear; used by R-CNN.

#flashcards/ml/concept #flashcards/topic/vision What problem does Selective Search solve?::An exhaustive sliding window visits far too many boxes and forces weak classifiers; Selective Search uses the image's own structure to propose a small set of boxes at all scales that is the same for every class (Uijlings et al. draft, section 1 and 2).

#flashcards/ml/concept #flashcards/topic/vision What are the steps of Selective Search?::Over-segment the image into tiny regions; compute the similarity of every touching pair; merge the most similar pair and update its neighbours' similarities; repeat until one region is left; draw a box around every region that ever existed; repeat in several diverse ways and pool the boxes.

#flashcards/ml/concept #flashcards/topic/vision What four similarity measures does Selective Search combine?::Colour (histogram intersection), texture (histogram intersection of gradient measures), size (small regions merge early) and fill (how well two regions fit into their joint box); each is between 0 and 1 (draft, section 3.2).

#flashcards/ml/concept #flashcards/topic/vision Why does Selective Search diversify its grouping?::No single strategy works for all images: regions group by colour in one case, by texture in another, by enclosure in a third. The paper varies the colour space, the similarity mix and the starting regions and pools the boxes (draft, section 1 and 3).

#flashcards/ml/example #flashcards/topic/vision A picture is cut into 6 starting regions: how many regions, so how many boxes, does the hierarchy give?::2 x 6 - 1 = 11 regions (6 starting plus 5 merges), so 11 boxes (my toy run printed 11).

#flashcards/ml/concept #flashcards/topic/vision How good is Selective Search in the paper?::The quality setting reaches recall 0.991 and Mean Average Best Overlap 0.879 with 10,097 boxes; the fast setting 0.980, 0.804 and 2,134 boxes; an exhaustive part-based search needs 100,352 boxes per class for recall 0.933 and MABO 0.829 (draft, Table 5; the abstract says 10,097 and Table 4 says 10,108).

#flashcards/ml/example #flashcards/topic/vision What did OpenCV's Selective Search give on the 512 by 512 astronaut photo?::Single strategy 516 boxes in 0.40 s; fast 2,418 boxes in 3.6 s; quality 8,821 boxes in 15 s (one thread, my laptop). The fast and quality modes found all 7 of my hand-drawn boxes at IoU 0.5 or more (one photo, my own boxes).

#flashcards/ml/concept #flashcards/topic/vision What is Selective Search's role in R-CNN and what limits it?::R-CNN takes about 2,000 fast-mode boxes per image and classifies each; recall of the proposals caps the detector (91.6% at 0.5 overlap on ILSVRC 2013 against about 98% on PASCAL), and the step is slow and outside the network, which Faster R-CNN replaced with a learned proposal network.

#flashcards/ml/cloze #flashcards/topic/vision Selective Search starts from ==many small regions==, repeatedly merges the ==most similar== neighbours, and draws a ==box== around every region that appears.
