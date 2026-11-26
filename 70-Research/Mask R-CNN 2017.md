---
tags: [paper, vision, segmentation, detection, CNN]
link: https://arxiv.org/abs/1703.06870
authors: He, Gkioxari, Dollar, Girshick (Facebook AI Research; checked on the paper's first page)
year: 2017
status: body-read
field: instance segmentation
---

# Mask R-CNN 2017

> **In one line for a 12-year-old:** Faster R-CNN draws a box around each object. Mask R-CNN adds a second, tiny helper that colours in exactly which pixels inside the box belong to the object, one separate colouring for every object, even when two objects touch.

Paper: "Mask R-CNN", arXiv 1703.06870, the version read is v3 of 24 January 2018, 14 pages with two appendices. Marks: "(paper, section)" was read in the paper; "(general knowledge)" is my own background; "(mine)" is my own arithmetic, done in Python. This paper's arXiv licence is not open (the usual arXiv licence), so its own figures are not copied. The note uses one free picture from Wikimedia Commons that shows the four tasks (classification, semantic segmentation, object detection, instance segmentation); it is not the paper's figure.

## The problem
Instance segmentation "requires the correct detection of all objects in an image while also precisely segmenting each instance". It combines object detection (find and box each object) with semantic segmentation (classify each pixel, without telling instances apart) (paper, 1). One would expect a complex method to be needed, but the authors want "a conceptually simple, flexible, and general framework" like Faster R-CNN is for detection and FCN is for semantic segmentation (paper, 1). See [[Image Segmentation]] and [[Object Detection]].

## What people did before
- **Segment proposals first, then classify** (the approach of [[R-CNN 2013]]'s followers, DeepMask and its successors): "segmentation precedes recognition, which is slow and less accurate" (paper, 2).
- **MNC (Dai et al., the COCO 2015 segmentation winner):** a multi-stage cascade that predicts segment proposals from box proposals and then classifies them (paper, 2).
- **FCIS (Li et al., the COCO 2016 winner):** predicts position-sensitive output channels fully convolutionally, so classes, boxes and masks come out together and fast, but it "exhibits systematic errors on overlapping instances and creates spurious edges" (paper, 2).
- **Segmentation first, then cut into instances:** start from per-pixel class results such as [[FCN 2014]] output and split pixels of one class into instances; this is the "segmentation-first strategy", against Mask R-CNN's "instance-first" one (paper, 2).
- **[[Faster R-CNN 2015]]:** detects and boxes objects in two stages, a region proposal network and a Fast R-CNN head, but "was not designed for pixel-to-pixel alignment between network inputs and outputs" (paper, 1).

## The idea
Keep the two stages of [[Faster R-CNN 2015]]: the first stage proposes boxes, the second looks at each one. In the second stage, "in parallel to predicting the class and box offset", also output a **binary mask** for each region of interest. Two details make it work (paper, 1 and 3):
1. **RoIAlign:** a feature-extraction layer with no rounding, so the small feature map of each region lines up exactly with the pixels it came from. The old RoIPool rounded coordinates, which is harmless for classification but ruins pixel-accurate masks.
2. **Decoupling mask and class:** predict one independent yes/no mask for each class (per-pixel sigmoid and binary loss) and let the box branch's class pick which mask to use, instead of a per-pixel competition between classes (softmax).

The mask branch is a small [[FCN 2014]] applied to each region. See [[Anchor Box]], [[RoI Pooling]] and [[RoIAlign]].

## The architecture, part by part

![[mask-rcnn-framework.webp]]
*The four tasks on one photo of balloons: classification (what is in the picture), semantic segmentation (all balloon pixels get one colour), object detection (one box per balloon) and instance segmentation (a separate coloured mask and box per balloon, even where balloons overlap). Mask R-CNN solves the fourth. Author: Vandyahmad24, from [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Mask_R-CNN.webp), CC BY-SA 4.0. The file is titled "Mask R-CNN" but it is not the paper's Figure 1 (the framework diagram) and not a Mask R-CNN architecture drawing; it only illustrates the tasks. I compared it with the paper's definitions in the introduction (paper, 1, footnote 1) and they match.*

The paper's Figure 1 (the framework: RoIAlign, then a class and box branch and a mask branch), Figure 2 and Figure 5 (results on COCO), Figure 3 (RoIAlign), Figure 4 (the head architectures), Figure 6 (FCIS against Mask R-CNN), Figures 7 and 8 (keypoints, Cityscapes) are not reproduced since they are not free; I describe them below.

The whole design (paper, 3). A backbone (ResNet or ResNeXt, depth 50 or 101, with or without a feature pyramid) computes feature maps for the whole image. A region proposal network proposes boxes. For each box, RoIAlign extracts a small feature map. A head then predicts three things in parallel: the class, the box offset, and an $m\times m$ binary mask for each of the $K$ classes. The mask belonging to the predicted class is the answer (paper, 3 and 3.1).

### Part by part
- **The backbone and the head, kept separate.**
  - *What it does:* the paper separates (i) the convolutional *backbone* that computes features over the whole image and (ii) the *head* applied separately to each region for the box and the mask. Backbones are named net-depth-features, for example ResNet-50-C4 (features from the last layer of the 4th stage) or ResNet-101-FPN (a feature pyramid network, which builds a multi-scale pyramid from a single-size image) (paper, 3).
  - *Why it matters:* it lets the authors test many instantiations to show the method is general. FPN "gives excellent gains in both accuracy and speed" (paper, 3), and the C4 head is heavy because it includes the 9-layer res5 stage.
- **The first stage: the region proposal network.**
  - *What it does:* identical to Faster R-CNN's first stage. For convenient ablation the RPN is trained separately and does not share features unless stated (paper, 3.1). Anchors span 5 scales and 3 aspect ratios (paper, 3.1); in Faster R-CNN it was 3 scales (see [[Faster R-CNN 2015]]).
  - *Why it matters:* the new ideas are all in the second stage.
- **RoIAlign.**
  - *What it does:* for each region it extracts a small map (7 by 7, or 14 by 14 for the mask) without any rounding. The region's coordinates are used as real numbers ($x/16$ and not $[x/16]$), each bin is divided into 4 regularly placed sampling points, the value at each is computed by bilinear interpolation from the 4 nearest feature-map cells, and the points are combined by max or average (paper, 3).
  - *Why it matters:* RoIPool "quantizes" a region twice (once when mapping the region to the feature map, once when cutting it into bins), which shifts the extracted features away from the pixels. "While this may not impact classification, which is robust to small translations, it has a large negative effect on predicting pixel-accurate masks" (paper, 3). My example: a region edge at $x=100$ pixels is at $100/16=6.25$ cells; rounding to 6 cells is an error of 0.25 cell, which is 4 pixels, and up to 8 pixels at the worst (mine). The results "are not sensitive to the exact sampling locations, or how many points are sampled, as long as no quantization is performed" (paper, 3).
- **The class and box branch (as in Fast R-CNN).**
  - *What it does:* classification and bounding-box regression, "identical as those defined in [Fast R-CNN]" (paper, 3). In the C4 head these follow res5 and an average pool; in the FPN head two 1024-unit fully connected layers (paper's Figure 4).
  - *Why it matters:* it supplies the class that selects the mask, and Table 3 shows its box results also improve.
- **The mask branch: a small FCN per region.**
  - *What it does:* from each 14 by 14 region map, a stack of 3 by 3 convolutions (four of them, 256 channels, in the FPN head), a 2 by 2 deconvolution with stride 2 up to 28 by 28, and a final 1 by 1 convolution gives $K=80$ masks of 28 by 28 for COCO's 80 classes (paper's Figure 4 and caption). At test time the mask is resized to the region size and thresholded at 0.5 (paper, 3.1).
  - *Why it matters:* a mask is a spatial layout; it should not be collapsed into a vector by fully connected layers. The fully convolutional design "requires fewer parameters, and is more accurate" (paper, 3). My count from Figure 4: about 2.64 million weights for the FCN mask head (four 3 by 3 convolutions with 256 channels, the deconvolution and the 1 by 1 layer) against about 65.3 million for the fully connected variant of Table 2e (1024 to 1024 to $80\times28^2=62{,}720$ outputs), 25 times more (mine, from the labels of Figure 4 and Table 2e).
- **Per-class binary masks with a sigmoid.**
  - *What it does:* the mask branch outputs $K\times m^2$ numbers, a binary mask for each class, with a sigmoid at each pixel; "for an RoI associated with ground-truth class $k$, $L_{\text{mask}}$ is only defined on the $k$-th mask" (paper, 3).
  - *Why it matters:* with a per-pixel softmax over classes, the masks of different classes compete. The ablation in Table 2b shows a loss of 5.5 mask AP for softmax (24.8 against 30.3).
- **Inference: masks only for the best 100 boxes.**
  - *What it does:* at test time the box branch runs on the proposals (300 for C4, 1000 for FPN), then non-maximum suppression, and the mask branch is run only on the top 100 detections (paper, 3.1). See [[Non-Maximum Suppression]].
  - *Why it matters:* it "speeds up inference and improves accuracy (due to the use of fewer, more accurate RoIs)" and it keeps the extra cost of the mask branch to about 20% (paper, 3.1).

### Size and speed check (mine)
The mask head's output is $80\times28\times28=62{,}720$ numbers per region with FPN, or $80\times14\times14=15{,}680$ with the C4 head. The paper's measured speed is 195 ms per image for ResNet-101-FPN on a Tesla M40 plus 15 ms of CPU time to resize the outputs, so 210 ms, 4.8 frames per second (mine), which the paper rounds to "5 fps" (paper, 4.4). Training ResNet-50-FPN takes 32 hours on 8 GPUs, and 160,000 iterations at 0.72 seconds each is $160{,}000\times0.72/3600=32.0$ hours (mine), which matches.

## The core mechanism, step by step

### From an image to instance masks (my numbered paraphrase of paper, 3 and 3.1)
1. **Rescale** the image so the shorter side is 800 pixels, and run the backbone to get feature maps.
2. **Propose regions** with the RPN (300 for C4, 1000 for FPN).
3. **Classify and refine** each proposal with the box branch; apply non-maximum suppression and keep the top 100 detections.
4. **Extract features with RoIAlign** from the shared feature map for each of those boxes (14 by 14 for the mask branch).
5. **Predict masks:** the mask branch outputs 80 masks of 28 by 28 for each box.
6. **Select** the mask of the predicted class $k$, ignore the other 79, resize it to the box size and threshold it at 0.5.

### RoIAlign: bilinear sampling without rounding

**Worked example (tiny numbers, mine).** A feature map has the values $f(2,3)=2$, $f(3,3)=4$, $f(2,4)=6$ and $f(3,4)=8$ at four neighbouring cells (the first number is the column, the second the row). A sampling point falls at column 2.25, row 3.5.
- Fractions: $0.25$ across and $0.5$ down, so the weights of the four cells are $0.375$, $0.125$, $0.375$ and $0.125$ (they sum to 1).
- Value $=0.375\times2+0.125\times4+0.375\times6+0.125\times8=0.75+0.5+2.25+1.0=4.5$.
- A quantized lookup would round the point to the cell $(2,4)$ and read 6.
- Meaning: RoIAlign gives 4.5, the value at the exact place; rounding gives 6, a value from a cell half a cell away, so the mask would be shifted.

### Training (paper, 3.1)
1. **Label the regions:** a sampled region is positive if its overlap ([[Intersection over Union]]) with a true box is at least 0.5, and negative otherwise. The mask loss is defined only on positive regions, and the mask target is the intersection of the region and its true mask.
2. **Sample the mini-batch:** images are resized to 800 pixels on the shorter side, 2 images per GPU on 8 GPUs (16 in all); each image has $N$ sampled regions with a ratio 1:3 of positives to negatives, $N=64$ for C4 and 512 for FPN.
3. **Optimise:** 160,000 iterations at learning rate 0.02, divided by 10 at iteration 120,000; weight decay 0.0001 and momentum 0.9. For ResNeXt, 1 image per GPU and a starting rate of 0.01 (see [[Momentum]] and [[Learning Rate]]).
4. **Minimise the multi-task loss** (equations 1 and 2).

## The equations

**1. The multi-task loss** (paper, section 3, unnumbered)

$$
L=L_{\text{cls}}+L_{\text{box}}+L_{\text{mask}}
$$

**Where:**
- $L_{\text{cls}}$ and $L_{\text{box}}$: the classification loss and the bounding-box regression loss, "identical as those defined in [Fast R-CNN]" (the formulas are in the Fast R-CNN paper, not written here).
- $L_{\text{mask}}$: the new mask loss (equation 2).
- The three are simply added, each on every sampled region.

**Matters:** the mask branch is trained together with the other two, on the same regions. The paper finds that this multi-task training also helps the boxes: adding the mask branch gives +0.9 box AP over the same network without it (Table 3 and text), "due solely to the benefits of multi-task training". The paper does not add weights to the three terms, so they have equal weight (my reading).

**Example (tiny numbers):** a positive region with $L_{\text{cls}}=0.20$, $L_{\text{box}}=0.05$ and $L_{\text{mask}}=0.30$.
- $L=0.20+0.05+0.30=0.55$.
- A negative region has only $L_{\text{cls}}$ (and no box or mask term).
- Meaning: the mask term is the largest part of the loss for this region, so the gradient mostly asks the network to improve the mask.

**2. The mask loss** (paper, section 3: described in words as "the average binary cross-entropy loss" on the $k$-th mask with a per-pixel sigmoid; the formula below is my own writing of that description, not the paper's)

$$
L_{\text{mask}}=-\frac{1}{m^{2}}\sum_{i=1}^{m}\sum_{j=1}^{m}\Big[y_{ij}\log\sigma\big(a^{k}_{ij}\big)+\big(1-y_{ij}\big)\log\Big(1-\sigma\big(a^{k}_{ij}\big)\Big)\Big]
$$

**Where:**
- $m$: the mask resolution (28 for the FPN head, 14 for C4), so $m^2$ pixels per mask.
- $k$: the true class of the region; only the $k$-th of the $K$ masks is used, the others contribute nothing.
- $a^{k}_{ij}$: the raw score of mask $k$ at pixel $(i,j)$; $\sigma$: the sigmoid, $\sigma(a)=1/(1+e^{-a})$, which turns it into a probability that the pixel belongs to the object.
- $y_{ij}$: the true mask value, 1 inside the object and 0 outside, for that region.

**Matters:** because the class is picked by the box branch, the mask loss only needs a yes/no answer per pixel. The sigmoid scores each class's mask independently, so raising the score of one class does not push another down. With a softmax over the classes at each pixel the masks of different classes compete, which the ablation shows to cost 5.5 mask AP (Table 2b). Taking only the $k$-th mask also means a region of a cat never trains the "dog" mask.

**Example (tiny numbers):** a 2 by 2 mask ($m=2$) with truth $y=(1,0,0,1)$ and predicted probabilities $(0.9,0.2,0.4,0.7)$.
- Per-pixel losses: $-\ln0.9=0.1054$, $-\ln(1-0.2)=0.2231$, $-\ln(1-0.4)=0.5108$, $-\ln0.7=0.3567$.
- Average: $(0.1054+0.2231+0.5108+0.3567)/4=0.299$.
- Competition against independence: for scores $(2,1,0)$ on three classes, a softmax gives $(0.665,0.245,0.090)$, which must sum to 1; if the first score rises to 3, the second class falls from $0.245$ to $0.114$. Independent sigmoids give $(0.881,0.731,0.500)$ and the second stays at $0.731$ when the first rises (mine).
- Meaning: the mask is judged pixel by pixel, and classes do not fight each other for the pixel.

**3. RoIAlign's bilinear sampling** (paper, section 3: described in words; the formula is my own writing of ordinary bilinear interpolation, general knowledge, not printed in the paper)

$$
f(x,y)=\sum_{(u,v)}f(u,v)\,\max(0,1-|x-u|)\,\max(0,1-|y-v|)
$$

**Where:**
- $(x,y)$: the sampling point, at real-number coordinates in the feature map (no rounding).
- $(u,v)$: the integer cells of the feature map; only the four nearest cells have a non-zero weight.
- $f(u,v)$: the feature value at a cell; $f(x,y)$: the interpolated value at the sampling point.
- $\max(0,1-|x-u|)$: the weight of a cell, which falls linearly from 1 at the cell to 0 at one cell away.

**Matters:** it is the whole point of RoIAlign: no coordinate is rounded, so the value read for a sampling point comes from the place it really is. The paper stresses that RoIWarp (from MNC), which also uses bilinear resampling but still rounds the region, "performs on par with RoIPool" (27.2 against 26.9 AP, Table 2c), so it is the alignment, not the interpolation, that matters.

**Example (tiny numbers):** the worked example above: weights $0.375$, $0.125$, $0.375$, $0.125$ on the values $2,4,6,8$ give $4.5$.
- The same arithmetic at the nearest cell: $f(2,4)=6$, an error of $1.5$ from the true $4.5$.
- Meaning: the interpolated value is a weighted average of the four neighbours, with the closest weighing most.

## What they found
The headline numbers (paper): on COCO instance segmentation, Mask R-CNN with ResNet-101-FPN gets 35.7 mask AP, with ResNeXt-101-FPN 37.1, at about 5 frames per second, above all previous single-model results including the COCO 2016 winner FCIS+++ (33.6); it also does well on box detection and on person keypoints (abstract and Tables 1, 3 and 4).

### Instance segmentation on COCO (paper's Table 1)

**Table 1: instance segmentation mask AP on COCO test-dev** (written by me, an AI, from the paper's Table 1; all rows shown; "-" means the paper gives no number).

| Method | Backbone | AP | AP50 | AP75 | APS | APM | APL |
|---|---|---|---|---|---|---|---|
| MNC (2015 winner) | ResNet-101-C4 | 24.6 | 44.3 | 24.8 | 4.7 | 25.9 | 43.6 |
| FCIS + OHEM (2016 winner) | ResNet-101-C5-dilated | 29.2 | 49.5 | - | 7.1 | 31.3 | 50.0 |
| FCIS+++ + OHEM | ResNet-101-C5-dilated | 33.6 | 54.5 | - | - | - | - |
| Mask R-CNN | ResNet-101-C4 | 33.1 | 54.9 | 34.8 | 12.1 | 35.6 | 51.1 |
| Mask R-CNN | ResNet-101-FPN | 35.7 | 58.0 | 37.8 | 15.5 | 38.1 | 52.4 |
| Mask R-CNN | ResNeXt-101-FPN | 37.1 | 60.0 | 39.4 | 16.9 | 39.9 | 53.5 |

What each column is, and why it matters:
- **AP:** COCO's average precision averaged over overlap thresholds from 0.5 to 0.95, computed on the mask overlap (not the box); the strict, headline metric. **AP50 and AP75:** the same at overlap thresholds of 0.5 and 0.75; AP75 rewards tight masks. See [[Mean Average Precision]] and [[Intersection over Union]].
- **APS, APM, APL:** AP for small, medium and large objects.
- **Backbone:** the feature extractor; C4, C5 and FPN are the feature choices (FPN is a feature pyramid).
- **OHEM, +++:** extras of FCIS: online hard example mining, and multi-scale training and testing plus horizontal flip testing (paper's caption).

**Summary: what the table shows.**
- **Best:** ResNeXt-101-FPN at 37.1 AP. Even the plain ResNet-101-FPN (35.7) beats FCIS+++ (33.6) by 2.1 points, 6.3% relative, and the 2016 winner FCIS (29.2) by 6.5 points (mine), without FCIS's extras.
- **Worst:** MNC at 24.6, 11.1 points below Mask R-CNN ResNet-101-FPN.
- **Small objects gain most:** APS 15.5 against FCIS's 7.1, more than twice (mine).
- **FPN against C4:** 35.7 against 33.1, +2.6 points (mine).
- **Caution:** single models, one data set; the table compares against the published winners, whose training details differ. All entries use test-dev, whose labels are held out.

### What each design choice contributes (paper's Table 2, ablations on COCO minival)

**Table 2: ablations, mask AP** (written by me from the paper's Table 2; five sub-tables a to e; some columns left out).

| Sub-table | Setting | AP | AP50 | AP75 |
|---|---|---|---|---|
| (a) backbone | ResNet-50-C4 | 30.3 | 51.2 | 31.5 |
| | ResNet-101-C4 | 32.7 | 54.2 | 34.3 |
| | ResNet-50-FPN | 33.6 | 55.2 | 35.3 |
| | ResNet-101-FPN | 35.4 | 57.3 | 37.5 |
| | ResNeXt-101-FPN | 36.7 | 59.5 | 38.9 |
| (b) mask type, ResNet-50-C4 | softmax (multinomial) | 24.8 | 44.1 | 25.1 |
| | sigmoid (independent) | 30.3 | 51.2 | 31.5 |
| (c) RoI layer, ResNet-50-C4 | RoIPool | 26.9 | 48.8 | 26.4 |
| | RoIWarp | 27.2 | 49.2 | 27.1 |
| | RoIAlign (max) | 30.2 | 51.0 | 31.8 |
| | RoIAlign (average) | 30.3 | 51.2 | 31.5 |
| (d) stride-32 features, ResNet-50-C5 | RoIPool | 23.6 | 46.5 | 21.6 |
| | RoIAlign | 30.9 | 51.8 | 32.1 |
| (e) mask branch, ResNet-50-FPN | MLP (two fully connected layers) | 31.5 | 53.7 | 32.8 |
| | FCN (convolutions) | 33.6 | 55.2 | 35.3 |

Written by me from the paper's Table 2 text. Left out: the RoIWarp "average" row and the box-AP columns in (d), and the second MLP row of (e) (31.5, 54.0, 32.6); the class-agnostic mask result (29.7 AP against 30.3) is in the text, not the table.

What each column is, and why it matters:
- **AP, AP50, AP75:** mask AP at the different thresholds, as in Table 1; these are on 5,000 held-out validation images (minival), so they differ slightly from test-dev (35.4 here against 35.7 in Table 1 for ResNet-101-FPN).
- **Sub-table meaning:** (a) is the backbone, (b) the mask loss, (c) the layer that extracts region features, (d) the same on a coarser backbone (stride 32, so the misalignment is larger), (e) the design of the mask branch.

**Summary: what the table shows.**
- **Alignment is the biggest single factor:** RoIAlign over RoIPool gains 3.3 AP and 5.4 AP75 at stride 16 (+12%), and at stride 32 it gains 7.3 AP and 10.5 AP75 (+31% and +49%, mine); the paper says "50% relative improvement" for the AP75, "a massive 7.3 points". With RoIAlign, stride-32 features (30.9) are even better than stride-16 features (30.3), "largely resolving the long-standing challenge of using large-stride features".
- **The bilinear sampling alone is not the reason:** RoIWarp, which samples bilinearly but still rounds, is only 0.3 above RoIPool (27.2 against 26.9).
- **Decoupling the classes matters nearly as much:** sigmoid 30.3 against softmax 24.8, +5.5 AP (mine, the paper says 5.5). Class-agnostic masks (one mask regardless of class) are nearly as good, 29.7 against 30.3 (0.6 lower, mine), so the division of labour works.
- **A convolutional mask head beats a fully connected one:** 33.6 against 31.5, +2.1 AP (mine, the paper says 2.1), with about 25 times fewer weights (mine).
- **Better backbone, better result:** deeper (+2.4 for C4, +1.8 for FPN, mine), FPN over C4 (+3.3 at depth 50), ResNeXt over ResNet (+1.3 at depth 101).
- **Caution:** a validation subset, single runs; the max and average pooling are within 0.1 of each other, so that difference is noise (my reading). The paper notes that "not all frameworks automatically benefit from deeper or advanced networks" (paper, 4.2).

### Bounding-box detection (paper's Table 3)

**Table 3: object detection, bounding-box AP on COCO test-dev, single model** (written by me from the paper's Table 3; all rows shown; the mask output is ignored at test time).

| Method | Backbone | APbb | APbb50 | APbb75 | APbbS | APbbM | APbbL |
|---|---|---|---|---|---|---|---|
| Faster R-CNN+++ | ResNet-101-C4 | 34.9 | 55.7 | 37.4 | 15.6 | 38.7 | 50.9 |
| Faster R-CNN with FPN | ResNet-101-FPN | 36.2 | 59.1 | 39.0 | 18.2 | 39.0 | 48.2 |
| Faster R-CNN by G-RMI | Inception-ResNet-v2 | 34.7 | 55.5 | 36.7 | 13.5 | 38.1 | 52.0 |
| Faster R-CNN with TDM | Inception-ResNet-v2-TDM | 36.8 | 57.7 | 39.2 | 16.2 | 39.8 | 52.1 |
| Faster R-CNN, RoIAlign (no mask branch) | ResNet-101-FPN | 37.3 | 59.6 | 40.3 | 19.8 | 40.2 | 48.8 |
| Mask R-CNN | ResNet-101-FPN | 38.2 | 60.3 | 41.7 | 20.1 | 41.1 | 50.2 |
| Mask R-CNN | ResNeXt-101-FPN | 39.8 | 62.3 | 43.4 | 22.1 | 43.2 | 51.2 |

What each column is, and why it matters: the same AP family as before, but computed on the *box* overlap; the first (APbb) is the headline.

**Summary: what the table shows.**
- **Where the gain comes from:** against Faster R-CNN with FPN (36.2), Mask R-CNN ResNet-101-FPN (38.2) is 2.0 higher; the paper splits it into RoIAlign (+1.1: 37.3 against 36.2), multi-task training with the mask branch (+0.9: 38.2 against 37.3) and, with ResNeXt-101, another +1.6 (39.8) (mine, they agree with the paper's caption).
- **Best:** ResNeXt-101-FPN at 39.8, 3.0 above the best previous single model, TDM (36.8) (mine, the paper says 3.0).
- **Mask and box are close:** 37.1 (mask) against 39.8 (box), a gap of only 2.7 points, which the paper reads as largely closing "the gap between object detection and the more challenging instance segmentation" (paper, 4.3).
- **Caution:** the Faster R-CNN rows are published numbers from other papers with their own training details (the paper's own no-mask baseline is the "RoIAlign" row, which is the fair comparison).

### Keypoints and the multi-task result (paper's Tables 4 to 6)
Human pose is treated as a mask problem: each of $K$ keypoint types is a one-hot $m\times m$ mask where one pixel is foreground, trained with a softmax over the $m^2=56^2$ positions, and the head has eight 3 by 3 convolutions with 512 channels, a deconvolution and a 2 times bilinear upscale for a 56 by 56 output (paper, 5).

**Table 4: keypoint AP on COCO test-dev** (written by me from the paper's Table 4).

| Method | APkp | APkp50 | APkp75 | APkpM | APkpL |
|---|---|---|---|---|---|
| CMU-Pose+++ (2016 winner) | 61.8 | 84.9 | 67.5 | 57.1 | 68.2 |
| G-RMI | 62.4 | 84.0 | 68.5 | 59.1 | 68.1 |
| Mask R-CNN, keypoint-only | 62.7 | 87.0 | 68.4 | 57.4 | 71.1 |
| Mask R-CNN, keypoint and mask | 63.1 | 87.3 | 68.7 | 57.8 | 71.4 |

**Summary: what the table shows.**
- **Best:** the unified keypoint-and-mask model at 63.1 against 61.8 for the 2016 winner (+1.3, mine), at 5 fps. The keypoint-only model is 0.9 ahead of the winner, as the paper says, "a multi-stage processing pipeline".
- **Multi-task:** adding the mask branch adds 0.4 (63.1 against 62.7). On the minival set (Table 5), adding keypoints lowers the person box AP (53.6 to 52.0) and mask AP (45.8 to 45.1) slightly, so keypoints benefit from multi-task training but do not help the other tasks (paper, 5).
- **RoIAlign matters even more here:** on minival, +4.4 keypoint AP over RoIPool (64.2 against 59.8, paper's Table 6) even with the finer FPN strides.
- **Caution:** CMU-Pose+++ uses multi-scale testing and post-processing that the paper says add "a cumulative ~5 points (clarified in personal communication)", so the comparison is not like for like; G-RMI used extra training data and two models.

### Cityscapes (paper's Appendix A, Table 7)
**Table 7: Cityscapes instance segmentation (8 classes), test AP; 5 of 7 rows** (written by me from the paper's Table 7).

| Method | Training data | AP (test) | person | car |
|---|---|---|---|---|
| SAIS | fine | 17.4 | 14.6 | 35.7 |
| DIN | fine + coarse | 20.0 | 16.5 | 25.7 |
| SGN | fine + coarse | 25.0 | 21.8 | 39.4 |
| Mask R-CNN | fine | 26.2 | 30.5 | 46.9 |
| Mask R-CNN | fine + COCO | 32.0 | 34.8 | 49.1 |

Written by me from the paper's Table 7; the columns of the printed table are AP on val, AP, AP50, then person, rider, car, truck, bus, train, motorcycle and bicycle, and I checked each shown cell against that order and against the figures quoted in the text (person 21.8 to 30.5, car 39.4 to 46.9). Left out: the InstanceCut and DWT rows, the validation column and the per-class columns other than person and car.

**Summary: what the table shows.**
- **Best:** Mask R-CNN with COCO pre-training at 32.0, and without it 26.2, which is 1.31 times the best entry that used coarse data (DIN 20.0) and 1.51 times the best fine-only entry (SAIS 17.4) (mine; the paper says "over 30%" and "~50%").
- **Overlap is where it wins:** person 30.5 against 21.8 (+40% relative) and car 46.9 against 39.4 (+19%), classes with many overlapping instances (on average 6 people and 9 cars per image) (paper, A). The paper calls within-category overlap "a core difficulty of instance segmentation".
- **Small data:** COCO pre-training adds 5.8 points (mine), "the important role the amount of training data plays".
- **Val against test:** the paper notes a gap: truck 28.8 on val and 22.8 on test, bus 53.5 and 32.2, train 33.0 and 18.6 (drops of 6.0, 21.3 and 14.4, mine), with persons and cars within 1 point. It suggests "a domain shift on these categories", which have 200 to 500 training samples.
- **Caution:** Cityscapes has only 2,975 fine training images, and the rare classes have between 200 and 500 instances (paper, A).

### Enhanced results (paper's Appendix B, Tables 8 and 9)
Adding techniques one at a time to the ResNeXt-101-FPN baseline (on COCO minival) takes mask AP from 36.7 to 41.8 (+5.1) and box AP from 39.6 to 47.3 (+7.7) (paper). The steps, in mask AP (differences mine): an updated baseline (longer training, NMS threshold 0.5), +0.3; end-to-end training of the RPN and Mask R-CNN together, +0.6; pre-training on 5,000 ImageNet classes, +1.0; train-time augmentation, +0.6; a deeper ResNeXt-152, +0.5; non-local blocks, +0.6; test-time augmentation, +1.5. For keypoints the same recipe takes 64.2 to 70.4 (+6.2, mine). The paper adds that Mask R-CNN "was used as the framework by the three winning teams in the COCO 2017 instance segmentation competition". The enhanced numbers are on minival, not comparable with test-dev.

## Questions I had, answered

**What is the difference between semantic and instance segmentation?** Semantic segmentation gives each pixel a class and does not separate objects (all balloons are one blue region); instance segmentation gives each object its own mask (each balloon its own colour) (paper, 1, footnote 1; the picture above). See the balloons.

**Why does the rounding of RoIPool hurt masks but not classification?** Classification is "robust to small translations": a box shifted by a few pixels still holds the same object. A mask has to match pixels, so a shift of 4 to 8 pixels (my numbers for stride 16) means the wrong pixels are predicted (paper, 3).

**Why a separate mask for each class if only one is used?** The network learns all masks in one pass, and for each region only the true class's mask gets a loss. At test time the box branch's class picks one. The paper says the class-agnostic version (one mask) is nearly as good (29.7 against 30.3), so the per-class split is not what matters most (paper, 4.2); what matters is that the choice of class is not made by the pixel softmax.

**Does the mask branch slow the system a lot?** No: masks are only computed for the top 100 detections, adding about 20% on typical models (paper, 3.1). The whole system runs at about 5 fps for ResNet-101-FPN (210 ms per image, mine) and the C4 variant takes about 400 ms, so the paper does not recommend it (paper, 4.4).

**Why does multi-task training help the boxes?** The paper reports the fact (+0.9 box AP over Faster R-CNN with RoIAlign and no mask branch) but does not explain it beyond "benefits of multi-task training" (paper, 4.3).

**How can keypoints be a mask?** Each keypoint type is a mask with a single foreground pixel, and the network predicts one per type; training uses a softmax over the $m^2$ positions of the mask so that exactly one position is chosen (paper, 5).

## Why it mattered
It turned the two-stage detector into a general instance-level framework: one extra small branch gave state-of-the-art masks, and the same design handled keypoints with little change. RoIAlign became the standard region-feature layer (general knowledge). The paper itself reports that three winning teams of the COCO 2017 instance segmentation competition used it, and it was released as the Detectron code (paper, Appendix B and abstract). It also showed that the detection quality rose when masks were trained at the same time.

## Limits and open questions
- **Two stages, not a fast detector:** about 5 fps (210 ms per image) with ResNet-101-FPN and about 400 ms for C4 (paper, 4.4); the paper says the design "is not optimized for speed".
- **Low mask resolution:** masks are 28 by 28 (14 by 14 for C4), resized to the box size; the paper does not study higher resolutions for masks, and says it needed 56 by 56 for keypoints (paper, 3 and 5).
- **Small data:** classes with few examples (truck, bus, train in Cityscapes) are weak, and val and test AP differ a lot on them (paper, A).
- **Keypoints do not help the other tasks:** adding the keypoint branch lowers box and mask AP a little (paper, 5).
- **Many hand-set choices:** thresholds 0.5 for positive regions, the top 100 detections for masks, 0.5 for the mask threshold, and the segment head design; the paper tests the head type and the pooling layer but not these numbers.
- **Instance-first strategy:** the paper expects "a deeper incorporation of both strategies will be studied in the future" (paper, 2).

## Concepts used
[[Image Segmentation]], [[Object Detection]], [[Convolutional Neural Network]], [[Anchor Box]], [[RoI Pooling]], [[RoIAlign]], [[Feature Pyramid Network]], [[ResNet]], [[Transposed Convolution]], [[Softmax]], [[Cross-Entropy Loss]], [[Non-Maximum Suppression]], [[Intersection over Union]], [[Mean Average Precision]], [[Momentum]], [[Learning Rate]], [[Transfer Learning]]

## Reading trail
- **Came from:** [[Faster R-CNN 2015]], [[FCN 2014]], [[ResNet 2015]]
- **Led to:** the COCO 2017 winners and later instance-segmentation systems (named in the paper's Appendix B only as "the three winning teams"; not in the vault); compare [[U-Net 2015]], a semantic design for biomedical images

## How much I read
The paper's body as text: all 14 pages except the reference list (introduction, related work, Mask R-CNN, the experiments on COCO, keypoints, Cityscapes in Appendix A and the enhanced results in Appendix B, Tables 1 to 9). Not read as pictures: Figures 1 to 8 (the framework, RoIAlign and the head architectures were taken from their captions and labels). Table 7 (Cityscapes) was read from the extracted text and only some rows and columns are shown. Numbers marked "paper" come from the extracted text; "mine" are Python arithmetic (the quantization errors, the interpolation, the mask-loss example, the parameter counts, the speed checks, the table differences and ratios). The two formulas for the mask loss and the bilinear sampling are my own writing of what the paper describes in words (general knowledge), and the formulas for the classification and box losses are in the Fast R-CNN paper, not read here. The figure is a free CC BY-SA picture that illustrates the tasks and is not the paper's. Not opened in Obsidian, so the flashcards are unreviewed.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Mask R-CNN (He, Gkioxari, Dollar and Girshick, 2017, arXiv 1703.06870): what problem did it attack?::Instance segmentation: detect all objects and also give each one a precise pixel mask. Earlier methods were multi-stage, slow, or made errors on overlapping instances (paper, 1 and 2).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What did people do before Mask R-CNN?::Segment proposals first and classify them (DeepMask), the multi-stage cascade MNC, the fully convolutional FCIS (fast but with errors on overlaps), or cut FCN output into instances. Faster R-CNN gave boxes but had no pixel alignment (paper, 1 and 2).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Mask R-CNN: what is the idea?::Keep Faster R-CNN's two stages and add a parallel branch that predicts a binary mask per class for each region, with RoIAlign for pixel-aligned features and a per-pixel sigmoid so classes do not compete (paper, 1 and 3).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation ![[mask-rcnn-framework.webp]]
What does this picture show, and what is it not?
?
The four tasks on one photo of balloons: classification, semantic segmentation (all balloons one colour), object detection (one box each) and instance segmentation (a separate mask and box per balloon). It is a free CC BY-SA 4.0 picture by Vandyahmad24 that only illustrates the tasks; it is not the paper's Figure 1 nor an architecture drawing.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What is the difference between semantic and instance segmentation?::Semantic segmentation gives each pixel a class without telling objects apart; instance segmentation gives each object its own mask. Instance segmentation therefore combines detection and semantic segmentation (paper, 1).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What does the whole Mask R-CNN pipeline look like?::A backbone makes feature maps, a region proposal network proposes boxes, RoIAlign extracts a small feature map for each, and a head predicts in parallel the class, the box offset and an $m\times m$ binary mask for each of the $K$ classes; the mask of the predicted class is used (paper, 3).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What is the difference between the backbone and the head in Mask R-CNN?::The backbone (ResNet or ResNeXt, with C4 or FPN features) computes features over the whole image; the head is applied separately to each region for classification, box regression and the mask (paper, 3).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What does RoIAlign do and why does it matter?::It extracts a small map for each region with no rounding of the region or its bins, and computes each of 4 sampling points per bin by bilinear interpolation. RoIPool's rounding shifts features by up to 8 pixels at stride 16, harmless for classification but ruinous for pixel-accurate masks (paper, 3; the pixel numbers are my arithmetic).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Example: a region edge at $x=100$ pixels on a stride-16 feature map. What does rounding cost?
?
$100/16=6.25$ cells; RoIPool rounds to 6, an error of 0.25 cell, which is 4 pixels (up to 8 pixels at worst). RoIAlign keeps 6.25 and interpolates (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Why did RoIWarp not help while RoIAlign did?::RoIWarp also samples bilinearly but still quantizes the region like RoIPool, so it is only 0.3 AP above RoIPool (27.2 against 26.9), while RoIAlign gets 30.3. The alignment, not the interpolation, is what matters (paper, 3 and Table 2c).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What does the mask branch look like?::A small FCN on each 14 by 14 region map: four 3 by 3 convolutions (256 channels in the FPN head), a 2 by 2 deconvolution with stride 2 to 28 by 28 and a 1 by 1 convolution to 80 masks of 28 by 28. At test time the mask is resized to the box and thresholded at 0.5 (paper, 3, Figure 4, 3.1).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Why a fully convolutional mask branch and not fully connected layers?::A mask is a spatial layout and should not be collapsed into a vector. FCN gives 33.6 AP against 31.5 for a fully connected head (+2.1), with about 25 times fewer weights by my count (2.64 M against 65.3 M) (paper's Table 2e).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Why does Mask R-CNN use a per-pixel sigmoid and not a softmax for masks?::A softmax over classes makes the masks of different classes compete; the sigmoid predicts an independent yes/no mask for each class and the box branch's class picks one. The softmax costs 5.5 mask AP (24.8 against 30.3) (paper's Table 2b).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Why does Mask R-CNN run the mask branch only on the top 100 detections at test time?::It differs from the parallel computation used in training but speeds up inference and improves accuracy (fewer, more accurate regions); the mask branch adds only about 20% (paper, 3.1).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How are regions labelled and sampled in Mask R-CNN training?::A region is positive if its overlap with a true box is at least 0.5; the mask loss is defined only on positive regions, with the mask target the intersection of the region and its true mask. Each image has $N$ regions at a 1:3 positive to negative ratio (64 for C4, 512 for FPN) (paper, 3.1).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How was Mask R-CNN trained?::8 GPUs with 2 images each (16 in all), shorter side 800 pixels, 160,000 iterations at a learning rate of 0.02 divided by 10 at 120,000, weight decay 0.0001, momentum 0.9. ResNet-50-FPN takes 32 hours, matching $160{,}000\times0.72\,\text{s}/3600=32.0$ h (my check).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Write Mask R-CNN's multi-task loss (paper, section 3) and explain each symbol.
?
$$L=L_{\text{cls}}+L_{\text{box}}+L_{\text{mask}}$$
- $L_{\text{cls}}$, $L_{\text{box}}$: the classification and box regression losses, as defined in Fast R-CNN; $L_{\text{mask}}$: the new mask loss.
- Simply added, on each sampled region; the box term and the mask term apply only to positive regions.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Loss example: $L_{\text{cls}}=0.20$, $L_{\text{box}}=0.05$, $L_{\text{mask}}=0.30$. What is the loss?
?
$0.20+0.05+0.30=0.55$ (my arithmetic). The mask term is the biggest part, so the gradient mostly asks for a better mask.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Why are the three losses simply added, and does the mask help the boxes?::The paper gives no weights, so the three terms count equally (my reading). Multi-task training helps the boxes: Mask R-CNN has 0.9 box AP more than the same network without the mask branch (38.2 against 37.3), "due solely to the benefits of multi-task training" (paper, 4.3; difference mine).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Write Mask R-CNN's mask loss and explain each symbol (the formula is my writing of the paper's words).
?
$$L_{\text{mask}}=-\frac{1}{m^{2}}\sum_{i,j}\Big[y_{ij}\log\sigma\big(a^{k}_{ij}\big)+\big(1-y_{ij}\big)\log\big(1-\sigma\big(a^{k}_{ij}\big)\big)\Big]$$
- $m$: mask size; $k$: the true class, only its mask is used; $a^{k}_{ij}$: the raw score at pixel $(i,j)$; $\sigma$: the sigmoid; $y_{ij}$: the true mask value (1 in the object, 0 outside).
- The paper describes it as the average binary cross-entropy of the $k$-th mask; the formula itself is not printed.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Mask loss example: truth $(1,0,0,1)$, predictions $(0.9,0.2,0.4,0.7)$. What is the loss?
?
Per pixel $0.1054$, $0.2231$, $0.5108$, $0.3567$; the average is $0.299$ (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Why does the mask loss use a sigmoid and only the $k$-th mask?
?
Only the $k$-th mask is judged, so a region of a cat never trains the "dog" mask, and each pixel is a yes/no question, so classes do not compete. Example: scores $(2,1,0)$ give softmax $(0.665,0.245,0.090)$; raising the first score to 3 drops the second class to $0.114$, while sigmoids stay at $0.731$ (my arithmetic). The softmax version loses 5.5 AP.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Write RoIAlign's bilinear sampling and explain each symbol (my writing of general knowledge, not printed in the paper).
?
$$f(x,y)=\sum_{(u,v)}f(u,v)\,\max(0,1-|x-u|)\,\max(0,1-|y-v|)$$
- $(x,y)$: the real-valued sampling point; $(u,v)$: integer feature-map cells, only the four nearest count; $f(u,v)$: the feature value there.
- The weight of a cell falls linearly from 1 at the cell to 0 at one cell away.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Bilinear example: values $2,4,6,8$ at the four cells, sampling point 0.25 across and 0.5 down. What is the value?
?
Weights $0.375$, $0.125$, $0.375$, $0.125$: $0.75+0.5+2.25+1.0=4.5$. Rounding to the nearest cell would give 6, off by 1.5 (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Why is the bilinear sampling of RoIAlign a weighted average of four cells?
?
Because the point lies inside a square of four cells, and the nearer a cell, the more it should count: weights $0.375$ and $0.125$ in the example sum to 1, so the result $4.5$ stays between the neighbours. Without it the lookup would jump from 4 to 6 as the point moves a little (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How did Mask R-CNN do on COCO instance segmentation (paper's Table 1: best, worst, caution)?::ResNeXt-101-FPN 37.1 AP, ResNet-101-FPN 35.7 (above FCIS+++ at 33.6 by 2.1, 6.3% relative, mine), ResNet-101-C4 33.1; worst MNC at 24.6. Small objects gain most (APS 15.5 against 7.1 for FCIS). Caution: single models against the winners' published numbers.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What did the RoIAlign ablation show (paper's Table 2c and 2d)?::At stride 16, RoIAlign gains 3.3 AP and 5.4 AP75 over RoIPool (26.9 to 30.2); at stride 32 it gains 7.3 AP and 10.5 AP75 (23.6 to 30.9, +31%, and +49% for AP75, my arithmetic). Stride-32 features with RoIAlign (30.9) beat stride-16 features (30.3).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What did the other ablations show (paper's Table 2a, 2b, 2e)?::Softmax masks lose 5.5 AP against sigmoid masks; class-agnostic masks are nearly as good (29.7 against 30.3); a convolutional mask head beats a fully connected one by 2.1; deeper and FPN backbones help (ResNet-50-C4 30.3 up to ResNeXt-101-FPN 36.7). Caution: a validation subset, single runs.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How did Mask R-CNN do on bounding-box detection (paper's Table 3)?::ResNeXt-101-FPN 39.8 box AP, 3.0 above the best previous single model (36.8). Against Faster R-CNN with FPN (36.2), RoIAlign adds 1.1, multi-task training 0.9 and ResNeXt 1.6. The mask-box gap is only 2.7 points (37.1 against 39.8) (differences mine).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How is human pose estimation done with Mask R-CNN?::Each of $K$ keypoint types is a one-hot $m\times m$ mask (one foreground pixel), trained with a softmax over the $m^2$ positions; the head has eight 3 by 3 convolutions with 512 channels and gives a 56 by 56 output. 62.7 APkp, 63.1 with the mask branch, against 61.8 for the 2016 winner, at 5 fps (paper, 5 and Table 4).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What did multi-task training do in the keypoint experiments (paper's Tables 4 to 6)?::Adding the mask branch raised keypoint AP slightly (63.1 against 62.7), but adding keypoints lowered person box AP (53.6 to 52.0) and mask AP (45.8 to 45.1). RoIAlign gave +4.4 keypoint AP over RoIPool. The winner CMU-Pose used extra post-processing worth about 5 points.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How did Mask R-CNN do on Cityscapes (paper's Appendix A)?::26.2 AP on test with fine data only, 1.31 times the best entry with coarse data (DIN 20.0) and 1.51 times the best fine-only entry (17.4) (my ratios); person 30.5 against 21.8, car 46.9 against 39.4. COCO pre-training raises it to 32.0. Truck, bus and train have big val to test drops (6.0, 21.3, 14.4 points, mine). Caution: only 2,975 training images.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What did the enhanced results of Appendix B add?::Adding techniques one by one (longer training, end-to-end training, 5k-class ImageNet pre-training, augmentation, ResNeXt-152, non-local blocks, test-time augmentation) took mask AP from 36.7 to 41.8 and box AP from 39.6 to 47.3 on minival; the three winning teams of COCO 2017 instance segmentation used Mask R-CNN (paper, Appendix B).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How fast is Mask R-CNN?::195 ms plus 15 ms CPU per image for ResNet-101-FPN on a Tesla M40, 210 ms or 4.8 fps (my sum), "5 fps"; the C4 variant takes about 400 ms; the mask branch adds about 20% (paper, 3.1 and 4.4).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What are the limits of Mask R-CNN?::Two stages with about 5 fps and a design not optimised for speed; low mask resolution (28 by 28); weak on rare classes with domain shift (Cityscapes truck, bus, train); keypoints do not help the other tasks; many hand-set thresholds without ablation.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Mask R-CNN: reading trail and why did it matter?::It came from Faster R-CNN, FCN and ResNet. One extra branch gave state-of-the-art masks and the same design handled keypoints; RoIAlign became standard (general knowledge) and the paper reports three COCO 2017 winners built on it.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How much of the Mask R-CNN paper did the note read?::The body as text (14 pages, except the references) and Tables 1 to 9; Table 7 only partly; not the pictures of Figures 1 to 8. The mask-loss and bilinear formulas are my own writing of what the paper describes; the picture used illustrates the tasks and is not the paper's.
