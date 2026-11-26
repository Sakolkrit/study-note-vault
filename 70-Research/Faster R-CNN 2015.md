---
tags: [paper, vision, detection, CNN]
link: https://arxiv.org/abs/1506.01497
authors: Ren, He, Girshick, Sun (University of Science and Technology of China, Microsoft Research, Facebook AI Research; checked on the paper's first page)
year: 2015
status: body-read
field: object detection
---

# Faster R-CNN 2015

> **In one line for a 12-year-old:** Fast R-CNN could name objects quickly, but it needed a slow helper to first guess where objects might be. Faster R-CNN lets the same network make those guesses as a cheap extra step, so the helper is no longer slow.

Paper: "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks", arXiv 1506.01497, the version read is v3 of 6 January 2016, 14 pages. Marks: "(paper, section)" was read in the paper; "(general knowledge)" is my own background; "(mine)" is my own arithmetic, done in Python. This paper's arXiv licence is not open (the usual arXiv licence), so its own figures are not copied. I searched Wikimedia Commons for a free Faster R-CNN or RPN figure and found none, so the note has no picture; the figures are described in words and in tables.

## The problem
After SPPnet and [[Fast R-CNN 2015]] made the detection network fast, the step that proposes object locations became the bottleneck: "proposals are the test-time computational bottleneck" (paper, 1). Selective Search, the popular proposal method, takes about 2 seconds per image on a CPU, and even the faster EdgeBoxes takes 0.2 seconds, "as much running time as the detection network" (paper, 1). The authors note that comparing a CPU proposal method with a GPU detector is unfair, but re-implementing Selective Search on the GPU would "miss important opportunities for sharing computation" (paper, 1). See [[Object Detection]].

## What people did before
- **[[R-CNN 2013]]:** proposals from Selective Search, each cropped and classified separately by a CNN. Slow because the network runs on about 2,000 crops (paper, 1; the 2,000 is general knowledge in this paper's context, it is stated in the YOLO paper).
- **SPPnet and [[Fast R-CNN 2015]]:** compute the convolutions once for the whole image and pool a feature for each proposal, so detection is fast, "when ignoring the time spent on region proposals" (paper, 1). The proposals are still external (Selective Search, EdgeBoxes) and independent of the detector (paper, 2).
- **OverFeat:** a fully connected layer predicts box coordinates for a single object; turned convolutional, it detects several class-specific objects, in one stage (paper, 2).
- **MultiBox:** a network proposes class-agnostic boxes (800 anchors found by k-means, via a fully connected layer) that feed R-CNN. It does not share features with the detector and its anchors are not translation invariant (paper, 2 and 3.1.1).

## The idea
Compute proposals with the same convolutional features the detector already has. A small extra network, the **Region Proposal Network (RPN)**, slides over the last shared feature map and, at every position, predicts "is there an object here" and "how to adjust this reference box", for several reference boxes called **anchors** at different scales and shapes. The detector (Fast R-CNN) then classifies the best of these proposals. Because the two parts share the convolutional layers, the proposals cost about 10 milliseconds extra. In the language of "attention" networks, the RPN "tells the unified network where to look" (paper, abstract and 3). See [[Convolutional Neural Network]], [[Feature Map]] and [[Anchor Box]].

## The architecture, part by part

The paper's Figure 1 (three ways to handle several sizes: image pyramids, filter pyramids, or the anchors of this paper), Figure 2 (Faster R-CNN as one network: convolutional layers, then the RPN and the classifier with RoI pooling), Figure 3 (the RPN at one position, and example detections), Figure 4 (recall against overlap) and Figures 5 and 6 (detections on VOC and COCO) are not reproduced since they are not free; I describe them below.

The whole design (paper, 3). One network has two modules. The first is a fully convolutional network that proposes regions, the RPN. The second is the Fast R-CNN detector that uses the proposed regions. Both sit on the same stack of convolutional layers (5 for the small Zeiler and Fergus ZF network, 13 for [[VGG 2014]] VGG-16). One pass computes the feature map; the RPN reads it and outputs proposals; the detector pools a feature from the feature map for each proposal and outputs the class and a refined box.

### Part by part
- **The shared convolutional layers.**
  - *What they do:* compute one feature map from the whole image (rescaled so the shorter side is 600 pixels). The total stride on the last layer is 16 pixels for both ZF and VGG (paper, 3.3).
  - *Why they matter:* both the proposals and the detections come from this one computation, which is where the saving comes from. The feature map of a 1000 by 600 image is about 63 by 38 (mine).
- **The RPN's sliding window: a 3 by 3 convolution.**
  - *What it does:* a small network looks at a 3 by 3 window of the feature map at each position and maps it to a 256-number (ZF) or 512-number (VGG) feature followed by a ReLU. It is implemented as a 3 by 3 convolution (paper, 3.1).
  - *Why it matters:* the effective receptive field on the input image is large, 171 pixels for ZF and 228 for VGG (paper, 3.1), so each position sees much more than 3 by 3 feature cells.
- **The two sibling 1 by 1 layers: cls and reg.**
  - *What they do:* from the feature at each position, a box-classification layer outputs $2k$ scores (object or not, for each of the $k$ anchors) and a box-regression layer outputs $4k$ numbers (the adjustment of each anchor). They are 1 by 1 convolutions, so they are shared over all positions (paper, 3.1). With $k=9$ that is 18 scores and 36 numbers per position (mine).
  - *Why they matter:* this is the whole proposal network, "a kind of fully convolutional network", so it accepts any image size (paper, 1 and 3.1). See [[Softmax]].
- **Anchors: 3 scales and 3 aspect ratios.**
  - *What they do:* at each position the paper places $k=9$ reference boxes: areas of $128^2$, $256^2$ and $512^2$ pixels, with shapes 1:1, 1:2 and 2:1 (paper, 3.3). A 1000 by 600 image has "roughly 20,000" anchors (60 by 40 by 9, paper, 3.3).
  - *Why they matter:* a reference box turns "find a box of any size" into "adjust the nearest of nine standard sizes", so a single-size feature and a single-size window can find objects of many sizes without an image pyramid (paper, 3.1.1).
- **Translation invariance of anchors.**
  - *What it does:* if an object moves, the proposal moves with it and the same function predicts it in either place (paper, 3.1.1), because every position uses the same weights and the same set of anchors.
  - *Why it matters:* it cuts the number of parameters: the output layer of the RPN is $512\times6\times9=27{,}648$ weights (paper: $2.8\times10^4$), against 6.1 million for MultiBox's 800-anchor output layer (paper, and my check $1536\times5\times800=6{,}144{,}000$), about 222 times more (mine). Counting the projection layers too, 2.4 million against about 27.8 million, 11.7 times more (mine; the paper prints 27 million). The authors "expect less risk of overfitting on small datasets, like PASCAL VOC" (paper, 3.1.1).
- **Training labels for anchors.**
  - *What they do:* an anchor is positive if it has the highest overlap ([[Intersection over Union]]) with some true box, or an overlap above 0.7 with any true box; negative if its overlap is below 0.3 with all true boxes; the rest are ignored (paper, 3.1.2).
  - *Why it matters:* the gap between 0.3 and 0.7 avoids teaching the network from ambiguous anchors. The first rule (best match) guarantees every object has a positive anchor even if no anchor passes 0.7.
- **Ignoring anchors that cross the image border (training only).**
  - *What it does:* of roughly 20,000 anchors per image, those crossing the border are ignored, leaving about 6,000 (paper, 3.3). At test time the RPN runs on the whole image and the resulting boxes are clipped to the border.
  - *Why it matters:* "if the boundary-crossing outliers are not ignored in training, they introduce large, difficult to correct error terms in the objective, and training does not converge" (paper, 3.3).
- **Mini-batch sampling.**
  - *What it does:* from one image, 256 anchors are sampled, with positives and negatives at a ratio up to 1:1; if there are fewer than 128 positives, the batch is padded with negatives (paper, 3.1.3).
  - *Why it matters:* negative anchors greatly outnumber positive ones, so using all of them "will bias towards negative samples" (paper, 3.1.3).
- **Non-maximum suppression on proposals.**
  - *What it does:* many RPN boxes overlap; boxes are ranked by their score and an overlap threshold of 0.7 leaves about 2,000 per image, from which the top N go to the detector (paper, 3.3). See [[Non-Maximum Suppression]].
  - *Why it matters:* it "substantially reduces the number of proposals" without hurting accuracy (paper, 3.3 and Table 2: 55.2 with 6,000 unsuppressed proposals against 56.8 with 300 after NMS).
- **The detector: Fast R-CNN on the proposals.**
  - *What it does:* pools a fixed-size feature from the shared map for each proposal (the RoI pooling layer), classifies it and refines the box (paper, 3.2). See [[RoI Pooling]].
  - *Why it matters:* it is the second stage that turns class-agnostic proposals into class-specific detections (paper, 4.1, where the one-stage emulation is worse).

### Size and cost check (mine)
A VGG-16 RPN head has the 3 by 3 convolution with 512 channels in and out, $3\times3\times512\times512=2{,}359{,}296$ weights, plus the two 1 by 1 layers $512\times18=9{,}216$ and $512\times36=18{,}432$ weights, so $2{,}386{,}944$ in all with biases left out (paper: $2.4\times10^6$). Compared with the 138 million weights of VGG-16 itself ([[VGG 2014]]) the RPN head adds under 2% (mine, 2.39/138.4).

## The core mechanism, step by step

### From an image to detections (my numbered paraphrase of paper, 3 and 3.3)
1. **Rescale** the image so the shorter side is 600 pixels.
2. **Run the shared convolutional layers** once to get a feature map (stride 16).
3. **Slide the RPN** over the map: at each position, a 3 by 3 convolution, then the cls layer (18 scores) and the reg layer (36 numbers) for the 9 anchors.
4. **Turn each anchor into a proposal** using its regression numbers (equation 2 below), then clip to the image.
5. **Score and suppress:** keep the proposals' object scores, apply NMS at 0.7, and take the top 300 (the test setting for VOC).
6. **Pool a feature** from the shared map for each of the 300 proposals (RoI pooling).
7. **Classify and refine** with the Fast R-CNN head, then output the detections.

**Worked example (tiny numbers, mine).** Three scales times three shapes give 9 anchors per position, all centred on the same point: $128\times128$, $91\times181$, $181\times91$ (area $128^2$); $256\times256$, $181\times362$, $362\times181$ (area $256^2$); $512\times512$, $362\times724$, $724\times362$ (area $512^2$). The width and height of a "2:1" anchor of area $s^2$ are $s\sqrt2$ and $s/\sqrt2$, so the area is the same as the 1:1 anchor. At 63 by 38 positions that is $63\times38\times9=21{,}546$ anchors (the paper says "roughly 20,000", from $60\times40\times9=21{,}600$).

### Labelling anchors (paper, 3.1.2)

**Worked example (tiny numbers, mine).** A true box is $(30,30)$ to $(130,130)$, a square of side 100 (area 10,000).
- Anchor A, the same box: overlap $=1.00$, so positive.
- Anchor B, shifted by 10 pixels in both directions, $(40,40)$ to $(140,140)$: the shared area is $90\times90=8{,}100$ and together they cover $10{,}000+10{,}000-8{,}100=11{,}900$, so overlap $=0.681$. It is below 0.7, so it is positive only if it is the best match for this true box (A is, so B is ignored: between 0.3 and 0.7).
- Anchor C, shifted by 40: shared $60\times60=3{,}600$, union $16{,}400$, overlap $=0.220$, below 0.3, so negative.
- Anchor D, far away: overlap 0, negative.
- Meaning: only anchors that nearly coincide with an object count as positive, clear misses count as negative and borderline ones teach nothing.

### Training the two networks together (paper, 3.2)
The RPN and the detector, trained alone, would each change the shared convolutional layers in their own way. The paper describes three ways to share them:
1. **Alternating training** (used in every experiment of the paper): the "4-step" recipe below.
2. **Approximate joint training:** merge both into one network, in each step treat the proposals as fixed and add the two losses' gradients in the shared layers. It ignores the gradient with respect to the proposal box coordinates, so it is approximate, but gives "close results" and cuts training time by 25 to 50% (paper, 3.2).
3. **Non-approximate joint training:** needs an RoI pooling layer that is differentiable in the box coordinates, "beyond the scope of this paper".

**The 4-step alternating training (paper, 3.2).**
1. Train the RPN, starting from an ImageNet pre-trained model, for the proposal task.
2. Train a separate Fast R-CNN detector on the proposals from step 1, also starting from the ImageNet model. The two networks do not yet share layers.
3. Use the detector to initialise the RPN, then fix the shared convolutional layers and fine-tune only the layers unique to the RPN. Now they share.
4. Keep the shared layers fixed and fine-tune the layers unique to Fast R-CNN. Both networks now form one unified network.

More rounds gave "negligible improvements" (paper, 3.2). The training details of the RPN (paper, 3.1.3): new layers start from zero-mean Gaussian weights with standard deviation 0.01; shared layers are from ImageNet pre-training; for VGG, layers conv3_1 and up are tuned; learning rate 0.001 for 60k mini-batches then 0.0001 for 20k on PASCAL VOC; momentum 0.9; weight decay 0.0005 (see [[Momentum]], [[Learning Rate]] and [[Regularization]]).

## The equations

**1. The RPN loss for an image** (paper, Eq. 1)

$$
L(\{p_i\},\{t_i\})=\frac{1}{N_{\text{cls}}}\sum_i L_{\text{cls}}(p_i,p_i^{*})+\lambda\,\frac{1}{N_{\text{reg}}}\sum_i p_i^{*}\,L_{\text{reg}}(t_i,t_i^{*})
$$

**Where:**
- $i$: the index of an anchor in the mini-batch.
- $p_i$: the predicted probability that anchor $i$ contains an object; $p_i^{*}$: the true label, 1 if the anchor is positive and 0 if negative.
- $t_i$: the 4 predicted box adjustments (equation 2); $t_i^{*}$: the adjustments that would give the true box, for a positive anchor.
- $L_{\text{cls}}$: the log loss over the two classes (object, not object).
- $L_{\text{reg}}(t_i,t_i^{*})=R(t_i-t_i^{*})$: the box loss, where $R$ is the "smooth L1" loss "defined in [Fast R-CNN]" (the paper does not write it; from general knowledge it is $0.5x^{2}$ if $|x|<1$ and $|x|-0.5$ otherwise, summed over the 4 numbers). See [[Smooth L1 Loss]].
- $N_{\text{cls}}=256$ (the mini-batch size) and $N_{\text{reg}}\approx2{,}400$ (the number of anchor locations): normalisers.
- $\lambda=10$: the balance between the two terms.

**Matters:**
- $p_i^{*}$ in front of the box term switches the box loss on only for positive anchors; a negative anchor has no true box, so asking it to regress would be meaningless.
- $\lambda=10$ with these normalisers makes the two terms "roughly equally weighted": $N_{\text{reg}}/N_{\text{cls}}=2400/256=9.4$ (mine), so the box term, summed over about 9 times as many locations, is scaled back up by 10. The paper adds that "the normalization as above is not required and could be simplified" and that the results are insensitive to $\lambda$ over a wide range (its Table 9).
- Without the classification term the network could not rank proposals (the ablation: without cls scores, 44.6 mAP with 100 proposals against 55.1; Table 2). Without the box term the proposals are the raw anchors (52.1 against 56.8 with 300 proposals).

**Example (tiny numbers):** two anchors only, so $N_{\text{cls}}=N_{\text{reg}}=2$, $\lambda=10$. Anchor 1 is positive with predicted $p_1=0.8$; anchor 2 is negative with $p_2=0.3$. For anchor 1 the box differences (equation 2's example) are $-0.0281$, $0.0191$, $-0.0731$, $-0.0531$.
- Log losses: $-\ln0.8=0.2231$ for anchor 1 and $-\ln(1-0.3)=0.3567$ for anchor 2; their average is $0.2899$.
- Smooth L1 for anchor 1: all four differences are below 1, so it is $0.5\times(0.0281^2+0.0191^2+0.0731^2+0.0531^2)=0.5\times0.00932=0.00466$.
- Box term: $10\times0.00466/2=0.0233$ (anchor 2 adds nothing, because $p_2^{*}=0$).
- Total $L=0.2899+0.0233=0.3132$.
- Meaning: the classification mistake dominates here; the box was already close to its target, so its term is small.

**2. The box parametrisation** (paper, Eq. 2)

$$
\begin{aligned}
t_x&=\frac{x-x_a}{w_a},\quad t_y=\frac{y-y_a}{h_a},\quad t_w=\log\frac{w}{w_a},\quad t_h=\log\frac{h}{h_a},\\
t_x^{*}&=\frac{x^{*}-x_a}{w_a},\quad t_y^{*}=\frac{y^{*}-y_a}{h_a},\quad t_w^{*}=\log\frac{w^{*}}{w_a},\quad t_h^{*}=\log\frac{h^{*}}{h_a}
\end{aligned}
$$

**Where:**
- $x,y,w,h$: the centre and the width and height of the predicted box. $x_a,y_a,w_a,h_a$: the same for the anchor. $x^{*},y^{*},w^{*},h^{*}$: the same for the true box.
- $t_x,t_y$: how far the box centre is from the anchor centre, measured in anchor widths and heights.
- $t_w,t_h$: how much wider or taller the box is than the anchor, as a logarithm of the ratio.
- The starred $t$'s are the targets, computed from the true box; the unstarred ones are what the reg layer predicts.

**Matters:** the shift is divided by the anchor size, so a 10-pixel shift of a small anchor is a bigger number than the same shift of a large anchor and all anchors share one scale of numbers. The log makes size changes symmetric and positive: doubling and halving are $+0.69$ and $-0.69$, and the width can never become negative. Without these the 9 regressors would need very different number ranges.

**Example (tiny numbers):** anchor $(x_a,y_a,w_a,h_a)=(100,100,128,128)$ and true box $(110,95,160,100)$.
- Targets: $t_x^{*}=10/128=0.0781$, $t_y^{*}=-5/128=-0.0391$, $t_w^{*}=\ln(160/128)=0.2231$, $t_h^{*}=\ln(100/128)=-0.2469$.
- Say the network predicts $t=(0.05,-0.02,0.15,-0.30)$. Decoding gives a box with centre $(100+0.05\times128,\ 100-0.02\times128)=(106.4,\ 97.44)$ and size $128e^{0.15}\times128e^{-0.30}=148.7\times94.8$.
- Meaning: the true box is $160\times100$ at $(110,95)$; the prediction $148.7\times94.8$ at $(106.4,97.4)$ is close, which is why its loss was only 0.00466 above.

## What they found
The headline numbers (paper): with VGG-16 the whole system runs at 5 frames per second on a GPU, with 300 proposals per image, at the best mAP of the time on PASCAL VOC 2007, 2012 and MS COCO; the proposal step costs 10 ms against 1,510 for Selective Search (abstract and Table 5); the system was the basis of several first-place entries in the ILSVRC and COCO 2015 competitions (abstract).

### Proposal methods and what each part of the RPN contributes (paper's Table 2)

**Table 2: VOC 2007 test, Fast R-CNN with ZF, trained on VOC 2007 trainval; selected rows** (written by me, an AI, from the paper's Table 2; all 14 rows are shown, but I merged the paper's four columns for train-time and test-time proposals into two; "SS" is Selective Search, "EB" EdgeBoxes).

| Training and test proposals | Test proposals | mAP (%) |
|---|---|---|
| SS | 2000 | 58.7 |
| EB | 2000 | 58.6 |
| RPN+ZF, shared | 300 | 59.9 |
| RPN+ZF, unshared | 300 | 58.7 |
| Detector trained with SS, RPN at test only: RPN+ZF | 100 | 55.1 |
| ... same, RPN+ZF | 300 | 56.8 |
| ... same, RPN+ZF | 1000 | 56.3 |
| ... same, RPN+ZF without NMS | 6000 | 55.2 |
| ... same, no cls score | 100 | 44.6 |
| ... same, no cls score | 300 | 51.4 |
| ... same, no cls score | 1000 | 55.8 |
| ... same, no reg | 300 | 52.1 |
| ... same, no reg | 1000 | 51.3 |
| ... same, RPN+VGG | 300 | 59.2 |

(Rows after the first four are the paper's ablation experiments: the detector is trained on Selective Search proposals and fixed, and only the test-time proposals change. The paper has "2000" boxes in the training column for the first four rows.)

What each column is, and why it matters:
- **Proposals:** the method that produces the boxes the detector looks at, and how many per image. 300 is a maximum; after NMS there may be fewer (paper, footnote 8).
- **mAP:** mean average precision on VOC 2007 test, 20 classes; higher is better. See [[Mean Average Precision]].
- **shared and unshared:** whether the RPN and the detector share convolutional layers (after step 4 of the training, or only up to step 2).
- **no cls, no reg:** the RPN used with its object score or its box regression turned off at test time (random proposals, or raw anchors).

**Summary: what the table shows.**
- **RPN with 300 proposals beats Selective Search with 2,000:** 59.9 against 58.7, 1.2 points higher (mine), with 6.7 times fewer boxes (mine) and shared features.
- **Sharing helps:** 59.9 shared against 58.7 unshared; the paper explains that in step 3 the detector-tuned features improve the proposals.
- **The cls score is what lets a few proposals work:** with the score off, 100 proposals fall to 44.6 (10.5 below 55.1, mine), but with 1,000 the gap shrinks to 0.5. The score ranks the best boxes first.
- **The box regression matters:** without it (raw anchors) mAP falls 4.7 points (56.8 to 52.1, mine).
- **NMS is harmless:** 6,000 proposals without NMS give 55.2, 1.6 below 300 with NMS (mine).
- **A better backbone gives better proposals:** VGG instead of ZF, 59.2 against 56.8 (2.4 points, mine).
- **Caution:** the ablation rows use a detector trained on Selective Search proposals, which is a mismatch, so all of them are below the 58.7 baseline for that reason (paper, 4.1). My table merges columns, so it is not a copy of the paper's layout.

### VGG-16 on VOC 2007 and 2012 (paper's Tables 3, 4, 12)

**Table: Faster R-CNN with VGG-16, mAP (%)** (written by me from the paper's Tables 3, 4 and 12; some rows left out).

| Method | Proposals | Training data | VOC 2007 test | VOC 2012 test |
|---|---|---|---|---|
| Selective Search + Fast R-CNN | 2000 | 07 | 66.9 (68.1 with this paper's repo) | not given |
| Selective Search + Fast R-CNN | 2000 | 07+12 | 70.0 | not given |
| Selective Search + Fast R-CNN | 2000 | 07++12 | not given | 68.4 |
| RPN+VGG, unshared | 300 | 07 | 68.5 | not given |
| RPN+VGG, shared | 300 | 07 | 69.9 | 67.0 (trained on 12) |
| RPN+VGG, shared | 300 | 07+12 (07++12 for the 2012 test) | 73.2 | 70.4 |
| RPN+VGG, shared | 300 | COCO only (no VOC) | 76.1 | 73.0 |
| RPN+VGG, shared | 300 | COCO+VOC | 78.8 | 75.9 |

Written by me from the paper's tables; the paper's Table 4 also has a Selective Search row trained on "12" only (65.7), which I left out.

What each column is, and why it matters:
- **Training data:** "07" is VOC 2007 trainval, "07+12" adds VOC 2012 trainval, "07++12" is VOC 2007 trainval and test plus 2012 trainval (used for the 2012 test), "COCO" the MS COCO trainval set.
- **VOC 2007 / 2012 test:** mAP on the held-out test sets; the 2012 numbers come from the evaluation server.

**Summary: what the table shows.**
- **Shared RPN beats Selective Search at equal data:** 69.9 against 66.9 on VOC 2007 (3.0 points, mine), or 1.8 points against the repo's 68.1; with 07+12, 73.2 against 70.0 (3.2 points). On VOC 2012: 70.4 against 68.4 (2.0 points).
- **Sharing:** unshared 68.5, shared 69.9 (1.4 points).
- **More data is the biggest lever:** adding COCO raises 73.2 to 78.8, 5.6 points (mine, the paper says 5.6%). A COCO model without any VOC data scores 76.1, 2.9 points above one trained on VOC 07+12 (73.2), "even though the PASCAL VOC data are not exploited" (paper, 4.3).
- **Caution:** the 66.9 baseline was copied from the Fast R-CNN paper and the paper's own repository gives 68.1, so the "improvement" depends on which baseline is used (paper's table footnote).

### Timing (paper's Table 5)

**Table 5: time in ms on a K40 GPU, except Selective Search which runs on a CPU** (written by me from the paper's Table 5; the "total" and "frames per second" columns I recomputed).

| Model | System | conv | proposal | region-wise | total | rate |
|---|---|---|---|---|---|---|
| VGG | SS + Fast R-CNN | 146 | 1510 | 174 | 1830 | 0.5 fps |
| VGG | RPN + Fast R-CNN | 141 | 10 | 47 | 198 | 5 fps |
| ZF | RPN + Fast R-CNN | 31 | 3 | 25 | 59 | 17 fps |

What each column is, and why it matters:
- **conv:** the shared convolutional layers; **proposal:** the time to produce proposals (Selective Search, or the extra RPN layers); **region-wise:** NMS, RoI pooling, fully connected layers and softmax for each proposal (paper's table caption).
- **total and rate:** the sum and the frames per second. My sums are 146+1510+174=1830, 141+10+47=198 and 31+3+25=59, which agree with the printed totals; $1000/198=5.05$ and $1000/59=16.9$ fps, matching 5 and 17.

**Summary: what the table shows.**
- **The proposal step was the bottleneck:** with Selective Search it is 82.5% of the 1,830 ms (mine). With the RPN it is 10 ms, 5.1% of 198 ms, 151 times faster (mine).
- **The whole system is 9.2 times faster** (1830/198, mine) and region-wise time also falls, 174 to 47, because there are 300 proposals instead of 2,000.
- **The ZF version runs at 17 fps** (paper) but is less accurate (59.9 against 69.9 for VGG on VOC 2007 with the same training data, a gap of 10 points with the same training data and a different backbone, mine).
- **Caution:** the Selective Search time is on a CPU and the rest on a GPU, as the paper itself notes in its introduction. The YOLO paper measured the same system at 7 fps for VGG-16 and 18 for ZF (see [[YOLO 2015]]); the two papers do not give the same speed (the cause is not stated).

### What the anchors and the loss weight do (paper's Tables 8 and 9)

**Table 8: anchor settings, VOC 2007 test, VGG-16, VOC 2007 trainval** (written by me from the paper's Table 8).

| Anchor scales | Aspect ratios | mAP (%) |
|---|---|---|
| $128^2$ | 1:1 | 65.8 |
| $256^2$ | 1:1 | 66.7 |
| $128^2$ | 2:1, 1:1, 1:2 | 68.8 |
| $256^2$ | 2:1, 1:1, 1:2 | 67.9 |
| $128^2$, $256^2$, $512^2$ | 1:1 | 69.8 |
| $128^2$, $256^2$, $512^2$ | 2:1, 1:1, 1:2 | 69.9 |

**Table 9: the weight $\lambda$ in Equation 1** (mine, from the paper's Table 9; same setting).

| $\lambda$ | 0.1 | 1 | 10 | 100 |
|---|---|---|---|---|
| mAP (%) | 67.2 | 68.9 | 69.9 | 69.1 |

What each column is, and why it matters:
- **Anchor scales and aspect ratios:** the areas and shapes of the reference boxes at every position (the default is the last row). **mAP:** as before.
- **$\lambda$:** the balance between the classification and the box term of the loss.

**Summary: what the tables show.**
- **One anchor is clearly worse:** 65.8 or 66.7 against 69.9, 4.1 or 3.2 points lower (mine); the paper says "3-4%".
- **Scales matter more than shapes here:** 3 scales with 1 ratio gives 69.8, only 0.1 below the default; 3 ratios with 1 scale gives 67.9 or 68.8. The paper concludes that scales and aspect ratios are "not disentangled dimensions" on this dataset and keeps both "to keep our system flexible".
- **$\lambda$ is forgiving:** between 1 and 100 the result moves by at most 1.0 point (68.9, 69.9, 69.1; mine); $\lambda=0.1$ costs 2.7 points.
- **Caution:** single runs, one dataset, no error bars (my reading); the paper itself says the 1 to 100 range changes the result by "about 1%".

### One stage against two stages (paper's Table 10)

**Table 10: VOC 2007 test, ZF, Fast R-CNN, RPN unshared** (written by me from the paper's Table 10).

| System | Proposals | Detector | mAP (%) |
|---|---|---|---|
| Two-stage: RPN + ZF, unshared | 300 | Fast R-CNN + ZF, 1 scale | 58.7 |
| One-stage: dense, 3 scales, 3 ratios | 20000 | Fast R-CNN + ZF, 1 scale | 53.8 |
| One-stage: dense, 3 scales, 3 ratios | 20000 | Fast R-CNN + ZF, 5 scales | 53.9 |

**Summary: what the table shows.**
- **Two stages win by 4.8 points** against the best one-stage emulation (58.7 against 53.9, mine), which "justifies the effectiveness of cascaded region proposals and object detection"; the one-stage system is also slower because it has many more proposals (paper, 4.1). Earlier papers report a loss of about 6% for dense sliding windows against Selective Search (paper, 4.1).
- **Caution:** this compares the paper's own two-stage design with an OverFeat-style emulation in the same code, not with a real OverFeat run; and a later one-stage design (such as [[YOLO 2015]]) takes a different route, so the result is about this 2015 comparison only.

### MS COCO and the ResNet result (paper's Table 11 and the 2015 competitions)

**Table 11: MS COCO, VGG-16** (written by me from the paper's Table 11; the val columns of the first row are "-" in the paper).

| Method | Proposals | Training data | test-dev mAP@.5 | test-dev mAP@[.5, .95] |
|---|---|---|---|---|
| Fast R-CNN (as reported) | SS, 2000 | COCO train | 35.9 | 19.7 |
| Fast R-CNN (this paper's implementation) | SS, 2000 | COCO train | 39.3 | 19.3 |
| Faster R-CNN | RPN, 300 | COCO train | 42.1 | 21.5 |
| Faster R-CNN | RPN, 300 | COCO trainval | 42.7 | 21.9 |

What each column is, and why it matters:
- **mAP@.5:** mean average precision when a box counts as right at an overlap of 0.5 (PASCAL VOC's metric). **mAP@[.5, .95]:** the average over overlap thresholds from 0.5 to 0.95 in steps of 0.05, COCO's standard metric, which rewards tight boxes (paper, 4.2).

**Summary: what the table shows.**
- **Faster R-CNN gains over the same-protocol Fast R-CNN:** 2.8 points of mAP@.5 (42.1 against 39.3) and 2.2 of mAP@[.5, .95] (21.5 against 19.3, mine). The authors read this as evidence the RPN "improves the localization accuracy at higher IoU thresholds".
- **Their own Fast R-CNN is stronger than the published one at .5:** 39.3 against 35.9, which the paper puts down to the definition of negative samples and the mini-batch size (a conjecture of the paper).
- **A deeper backbone helps:** replacing VGG-16 with the 101-layer ResNet raises COCO val from 41.5 and 21.2 to 48.4 and 27.2 (6.9 and 6.0 points, 16.6% and 28.3% in relative terms, mine). With more changes, He et al. report a single model at 55.7 and 34.9 and an ensemble at 59.0 and 37.4 on test-dev, "1st place in the COCO 2015 object detection competition", and also won the ILSVRC 2015 detection track, 8.5 points above second place (paper, 4.2). See [[ResNet 2015]].
- **Caution:** the ResNet and competition numbers are from the paper's text, not from a table I could check; the 2015 competition results include improvements "orthogonal to Faster R-CNN".

## Questions I had, answered

**What exactly is an anchor?** A reference box of a fixed size and shape, centred at a feature-map position. The network never draws a box from nothing: it says "this anchor has an object" and "shift it by these amounts". Nine anchors per position, 21,546 for a 1000 by 600 image (mine). See [[Anchor Box]].

**Why does the RPN have 2k scores and not k?** The paper uses a two-class softmax (object, not object) for each anchor, which gives $2k$ outputs; it notes that "alternatively, one may use logistic regression to produce $k$ scores" (paper, footnote 4).

**Why is the box loss only for positive anchors?** A negative anchor has no true box to move toward, so the term is switched off by $p_i^{*}=0$ in the loss (paper, 3.1.2).

**Why predict the logarithm of the width and height ratio?** So that a doubling and a halving are equal and opposite ($+0.69$ and $-0.69$), and so that the predicted size is always positive after the exponential (general knowledge; the paper only says it follows R-CNN's parametrisation).

**Are the nine anchors' regressors shared?** No. The paper learns a set of $k$ regressors, "each regressor is responsible for one scale and one aspect ratio, and the $k$ regressors do not share weights" (paper, 3.1.2), unlike Fast R-CNN, where the regression weights are shared by all region sizes.

**Why is a 5 fps system called "real-time" in the title?** The paper's title says "towards real-time". The paper itself says 5 fps for VGG-16 and 17 fps with ZF; the YOLO paper's definition of real time is 30 fps or better, so neither VGG nor ZF Faster R-CNN is real time by that definition (YOLO paper, 4.1).

## Why it mattered
It made proposals nearly free and put them inside the network, so detection became one trainable system rather than a pipeline with an external piece. The paper says that RPN and Faster R-CNN were adopted in 3D detection, part-based detection, instance segmentation (its reference 15; [[Mask R-CNN 2017]] came later) and captioning, and in a commercial system at Pinterest (paper, 1). From general knowledge, anchors and the two-stage design became the basis of the detectors that followed, which [[Mask R-CNN 2017]] extended with a mask branch.

## Limits and open questions
- **Not real time in the usual sense:** 5 fps with VGG-16 (198 ms per image) on a K40, 17 fps for the small ZF network (paper, Table 5).
- **A pipeline in disguise:** training uses a 4-step alternating recipe, or an approximate joint training that ignores the gradient through the proposal coordinates (paper, 3.2); the exact joint training is left for later.
- **Many hand-set numbers:** 3 scales, 3 ratios, 0.7 and 0.3 for the labels, 0.7 for NMS, 256 anchors per batch, stride 16. The paper tests the anchor sets and $\lambda$ but not the thresholds; and the effective stride of 16 may hurt small objects (the paper says "accuracy may be further improved with a smaller stride", 3.3).
- **Single runs on a few data sets**, with no error bars (my reading).
- **Two-stage cost:** still 300 proposals, each through the region-wise layers (47 ms of the 198).

## Concepts used
[[Object Detection]], [[Convolutional Neural Network]], [[Feature Map]], [[Anchor Box]], [[RoI Pooling]], [[Smooth L1 Loss]], [[Intersection over Union]], [[Non-Maximum Suppression]], [[Mean Average Precision]], [[Softmax]], [[Cross-Entropy Loss]], [[Momentum]], [[Learning Rate]], [[Regularization]], [[Transfer Learning]], [[VGG 2014]], [[ResNet 2015]]

## Reading trail
- **Came from:** [[Fast R-CNN 2015]] and [[R-CNN 2013]]
- **Led to:** [[Mask R-CNN 2017]]; compare [[YOLO 2015]], which takes a different, one-stage route

## How much I read
The paper's body as text: all 14 pages except the reference list (introduction, related work, the Faster R-CNN design, experiments on VOC and COCO, the conclusion and Tables 1 to 12). Not read as pictures: Figures 1 to 6, so the descriptions of the figures are from their captions. The per-class tables (6 and 7) and the recall figure (Figure 4) were not summarised. Numbers marked "paper" come from the extracted text; "mine" are Python arithmetic (the anchor sizes and counts, the parameter checks, the examples, the table differences and ratios). "General knowledge" is background I did not check in the paper, including the formula of the smooth L1 loss. The note has no figure because no free one was found. Not opened in Obsidian, so the flashcards are unreviewed.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Faster R-CNN (Ren, He, Girshick and Sun, 2015, arXiv 1506.01497): what problem did it attack?::After Fast R-CNN sped up the detector, the proposal step became the bottleneck: Selective Search takes about 2 s per image on a CPU and EdgeBoxes 0.2 s, as much as the detection network (paper, 1).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What did people do before Faster R-CNN?::R-CNN classified Selective Search crops one by one; SPPnet and Fast R-CNN shared the convolutions but still used external proposals (Selective Search, EdgeBoxes); OverFeat and MultiBox used networks to predict boxes but not as a shared proposal stage (paper, 1, 2).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Faster R-CNN: what is the idea?::A Region Proposal Network (RPN), a few extra convolutional layers on the shared feature map, predicts an object score and box adjustments for k anchors at every position. Fast R-CNN then classifies the proposals. Sharing the convolutions makes proposals cost about 10 ms (paper, abstract and 3).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What is an anchor in Faster R-CNN?::A reference box of a fixed scale and aspect ratio centred at a feature-map position. The network predicts "is there an object" and "how to adjust this anchor" instead of drawing boxes from nothing. Default: 3 scales ($128^2$, $256^2$, $512^2$) times 3 ratios (1:1, 1:2, 2:1) gives 9 per position (paper, 3.1.1 and 3.3).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What does the RPN look like?::A 3 by 3 convolution (256-d for ZF, 512-d for VGG, with ReLU) on the last shared feature map, then two sibling 1 by 1 layers: cls with $2k$ scores and reg with $4k$ numbers. With $k=9$ that is 18 and 36 per position (paper, 3.1; counts mine).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Why do anchors let the RPN find objects of many sizes with one feature map?::The scales and shapes are built into the reference boxes (a "pyramid of regression references"), so no image pyramid or filter pyramid is needed and a single-size 3 by 3 window and a single-scale image are enough (paper, 3.1.1).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Why is the RPN's translation invariance useful?::The same weights and anchors at every position mean the same function finds a moved object. It also shrinks the output layer to $512\times6\times9=27{,}648$ weights against 6.1 million for MultiBox's 800-anchor layer, about 222 times fewer (my arithmetic), lowering the risk of overfitting on small data sets (paper, 3.1.1).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How many anchors does a 1000 by 600 image have?::Roughly 20,000 in the paper ($60\times40\times9$); my count with a 63 by 38 feature map is 21,546. Ignoring those that cross the border leaves about 6,000 for training (paper, 3.3).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Which anchors are positive and which negative in RPN training?::Positive: the anchor with the highest overlap with a true box, or any anchor with overlap above 0.7. Negative: overlap below 0.3 with every true box. In between: ignored (paper, 3.1.2).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Example: true box (30,30)-(130,130). What are the overlaps of anchors shifted by 0, 10 and 40 pixels?
?
1.00, 0.681 ($8100/11900$) and 0.220 ($3600/16400$): the first is positive, the second is between 0.3 and 0.7 (ignored unless best match), the third is negative (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Why ignore anchors that cross the image border during training?::They add large, hard-to-correct errors to the loss and training does not converge; at test time they are used and the boxes clipped to the border (paper, 3.3).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How is an RPN mini-batch built, and why?::256 anchors sampled from one image, with positives and negatives at up to 1:1 (padding with negatives if fewer than 128 positives), because using all anchors would bias the loss toward the many negatives (paper, 3.1.3).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Why does Faster R-CNN apply NMS to the proposals?::RPN boxes overlap heavily; NMS at 0.7 leaves about 2,000 per image without hurting mAP (6,000 unsuppressed proposals give 55.2 against 56.8 for 300 after NMS), then the top N go to the detector (paper, 3.3 and Table 2).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Write the RPN loss (paper's Eq. 1) and explain each symbol.
?
$$L(\{p_i\},\{t_i\})=\frac{1}{N_{\text{cls}}}\sum_i L_{\text{cls}}(p_i,p_i^{*})+\lambda\,\frac{1}{N_{\text{reg}}}\sum_i p_i^{*}\,L_{\text{reg}}(t_i,t_i^{*})$$
- $p_i$: predicted object probability of anchor $i$; $p_i^{*}$: its label (1 positive, 0 negative); $t_i,t_i^{*}$: predicted and target box numbers; $L_{\text{cls}}$: log loss; $L_{\text{reg}}$: smooth L1 loss.
- $N_{\text{cls}}=256$, $N_{\text{reg}}\approx2400$, $\lambda=10$. The $p_i^{*}$ switches the box loss on only for positive anchors.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation RPN loss example: two anchors, $p_1=0.8$ positive and $p_2=0.3$ negative, box differences of about 0.03 to 0.07, $\lambda=10$, $N=2$. What is $L$?
?
Log losses $0.2231$ and $0.3567$, average $0.2899$; smooth L1 $=0.00466$; box term $10\times0.00466/2=0.0233$; $L=0.3132$ (my arithmetic). The class error dominates, the box was already close.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Why does the RPN loss multiply the box term by $p_i^{*}$, and why $\lambda=10$?
?
A negative anchor has no true box, so the box loss must be off for it ($p_i^{*}=0$). $\lambda=10$ balances the two terms: $N_{\text{reg}}/N_{\text{cls}}=2400/256=9.4$ (mine). Results change by about 1 point for $\lambda$ from 1 to 100 (68.9, 69.9, 69.1) and drop 2.7 at 0.1 (paper's Table 9).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What breaks without the cls term or the reg term of the RPN (paper's Table 2)?::With 100 test proposals, no cls scores gives 44.6 mAP against 55.1 (10.5 lower); with 300, no box regression (raw anchors) gives 52.1 against 56.8 (4.7 lower). The cls score ranks the best proposals first; the regression makes boxes tight (my differences).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Write the box parametrisation of Faster R-CNN (paper's Eq. 2) and explain each symbol.
?
$$t_x=\frac{x-x_a}{w_a},\ t_y=\frac{y-y_a}{h_a},\ t_w=\log\frac{w}{w_a},\ t_h=\log\frac{h}{h_a}$$ (and the same with starred true-box values for the targets)
- $x,y,w,h$: box centre, width, height; $x_a,\dots$: the anchor; starred: the true box.
- Shifts are measured in anchor sizes, sizes as log ratios.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Box example: anchor (100,100,128,128), true box (110,95,160,100). What are the targets?
?
$t_x^{*}=10/128=0.0781$, $t_y^{*}=-5/128=-0.0391$, $t_w^{*}=\ln(160/128)=0.2231$, $t_h^{*}=\ln(100/128)=-0.2469$ (my arithmetic). Decoding a prediction of $(0.05,-0.02,0.15,-0.30)$ gives a box at about (106.4, 97.4) of size 148.7 by 94.8.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Why does the box parametrisation divide shifts by the anchor size and take the log of the size ratio?
?
Dividing by the anchor size puts all anchors' shifts on one scale (10 pixels is a 0.078 shift on a 128 anchor but 0.039 on a 256 anchor, mine). The log makes doubling and halving equal and opposite ($+0.69$, $-0.69$) and keeps the width positive. Without them the nine regressors would need very different number ranges (the reasons are general knowledge; the paper follows R-CNN).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How do the nine regressors of the RPN differ from Fast R-CNN's regression?::Fast R-CNN regresses boxes from features pooled from arbitrary-size regions, with weights shared by all sizes. The RPN uses fixed 3 by 3 features and learns $k$ regressors, one per scale and aspect ratio, that do not share weights (paper, 3.1.2).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How are RPN and Fast R-CNN trained to share convolutional layers (the 4 steps)?::1) train the RPN from an ImageNet model; 2) train a separate Fast R-CNN on its proposals; 3) initialise the RPN from the detector, fix the shared layers and tune only the RPN's own layers; 4) fix the shared layers and tune Fast R-CNN's own layers. More rounds gave negligible gains (paper, 3.2).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What is approximate joint training and what does it ignore?::The RPN and Fast R-CNN are trained in one network, treating the proposals as fixed, so the gradient with respect to the proposal box coordinates is ignored. It gives close results and cuts training time by 25 to 50% (paper, 3.2); the exact version needs a differentiable RoI pooling.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How did RPN proposals compare with Selective Search (paper's Table 2)?::With ZF, 300 shared RPN proposals gave 59.9 mAP against 58.7 for 2,000 Selective Search boxes (1.2 higher, 6.7 times fewer boxes, my arithmetic); unshared RPN gave 58.7. Worst: the no-cls ablation with 100 proposals, 44.6. Caution: the ablation rows use a detector trained on Selective Search proposals.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How did Faster R-CNN with VGG-16 do on VOC (paper's Tables 3, 4, 12)?::VOC 2007 test: 69.9 (07), 73.2 (07+12), 76.1 (COCO only), 78.8 (COCO+VOC); against Selective Search 66.9 or 68.1 and 70.0. VOC 2012: 70.4 (07++12) against 68.4, 75.9 with COCO. Caution: the 66.9 baseline was copied from the Fast R-CNN paper; the paper's own repo gives 68.1.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How fast is Faster R-CNN (paper's Table 5)?::VGG: 198 ms per image (141 conv, 10 proposal, 47 region-wise) or 5 fps against 1,830 ms (0.5 fps) with Selective Search; ZF: 59 ms, 17 fps. Selective Search was 82.5% of the old time; the RPN step is 5.1% of the new (my arithmetic). Selective Search ran on a CPU, the rest on a K40 GPU.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How did the anchor settings affect the result (paper's Table 8)?::One anchor per position gave 65.8 or 66.7 against 69.9 for 9 anchors (3 to 4 points lower); 3 scales with 1 ratio gave 69.8, so scales matter more than shapes here, and the paper keeps both "to keep our system flexible".

#flashcards/research/concept #flashcards/topic/research-detection-segmentation One-stage or two-stage: what did the paper's Table 10 show?::The two-stage system (RPN + Fast R-CNN) had 58.7 mAP against 53.9 for a one-stage emulation with 20,000 dense windows, 4.8 points more, and the one-stage version was slower. Caution: an emulation of OverFeat in the same code, not a real OverFeat run.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What did Faster R-CNN gain from COCO and from ResNet?::On COCO test-dev, 42.1 mAP@.5 and 21.5 mAP@[.5,.95] against 39.3 and 19.3 for Fast R-CNN. Replacing VGG-16 by ResNet-101 raised COCO val from 41.5 and 21.2 to 48.4 and 27.2 (6.9 and 6.0 points, my differences). The ResNet system won the COCO and ILSVRC 2015 detection tracks, with other improvements.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How much does COCO data help on PASCAL VOC?::A COCO-only model scores 76.1 on VOC 2007 test without any VOC data, 2.9 above one trained on VOC 07+12 (73.2); fine-tuning it on VOC gives 78.8, 5.6 above 73.2 (my differences, paper's Table 12).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What are the limits of Faster R-CNN?::5 fps with VGG-16 is not real time by YOLO's 30 fps definition; training is a 4-step alternating recipe or an approximate joint one; many hand-set numbers (3 scales, 3 ratios, 0.7 and 0.3, stride 16); single runs; and 300 proposals still go through the region-wise layers (47 of 198 ms).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Faster R-CNN: reading trail and why did it matter?::It came from Fast R-CNN and R-CNN and led to Mask R-CNN. It put proposals inside the network (10 ms instead of 1,510 ms), and the paper says RPNs were adopted for 3D detection, segmentation, captioning and in a Pinterest system. Compare YOLO's one-stage route.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How much of the Faster R-CNN paper did the note read?::The body as text (14 pages, except the references), Tables 1 to 12 (the per-class tables 6 and 7 only by their mAP); not the pictures of Figures 1 to 6. The smooth L1 formula is general knowledge, not in the paper. The note has no figure because no free one was found.
