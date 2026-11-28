---
tags: [paper, vision, detection, CNN, R-CNN, multi-task, RoI-pooling]
link: https://arxiv.org/abs/1504.08083
authors: Ross Girshick (single author; arXiv listing)
year: 2015
status: read
field: object detection
---

# Fast R-CNN 2015

> **In one line for a 12-year-old:** R-CNN ran a picture-reading network on every suggested box. Fast R-CNN reads the whole picture once and then just looks up each box in the result, and it learns to name the object and fix the box at the same time.

*Full title: "Fast R-CNN" (arXiv 1504.08083; first submitted 30 Apr 2015, I read version 2 of 27 Sep 2015; dates from the arXiv listing). Layout follows [[Attention Is All You Need 2017]] and [[Faster R-CNN 2015]]. Marks: "(paper, section)", "(general knowledge)", "(mine)". All timings in the paper use one Nvidia K40 GPU (paper, footnote 1).*

## The problem
[[R-CNN 2013]] detects well but has three listed drawbacks (paper, 1):
1. **Training is a multi-stage pipeline:** fine-tune a CNN with log loss, then fit SVMs on its features, then fit bounding-box regressors.
2. **Training is expensive in space and time:** features of every proposal in every image are cached on disk. With VGG16 this takes "2.5 GPU-days for the 5k images of the VOC07 trainval set" and "hundreds of gigabytes of storage".
3. **Detection is slow:** "47s / image (on a GPU)" with VGG16, because "it performs a ConvNet forward pass for each object proposal, without sharing computation."

The paper's goal is to "streamline the training process" with a "single-stage training algorithm that jointly learns to classify object proposals and refine their spatial locations" (paper, 1).

## What people did before
- **[[R-CNN 2013]]:** warp each proposal, run the CNN on it, classify with SVMs. Accurate but slow and multi-stage (above).
- **SPPnet (He et al.):** compute one convolutional feature map for the whole image and pool the part inside each proposal into a fixed-size vector (6 × 6 and other sizes, concatenated, as in spatial pyramid pooling). It speeds R-CNN "10 to 100× at test time" and training by 3× (paper, 1). Its limits: training is still a multi-stage pipeline with features written to disk, and its fine-tuning "cannot update the convolutional layers that precede the spatial pyramid pooling", which "limits the accuracy of very deep networks" (paper, 1).
- **OverFeat, and a related loss for a proposal network** (Erhan et al.): trained classifiers and box localisers stage-wise or in two networks, which the paper says is "suboptimal" here (paper, 2.3 and 5.1).
- **The L2 loss on box offsets** used in R-CNN and SPPnet, which "can require careful tuning of learning rates in order to prevent exploding gradients" when the targets are unbounded (paper, 2.3).

## The idea
Run the convolutional layers **once per image**, not once per proposal. Then, for each proposal, take its part of the shared feature map and squeeze it to a fixed size with an **RoI pooling layer**. Feed the result to fully connected layers that end in **two sibling outputs**: class probabilities and box offsets. Train all of it with one **multi-task loss**, so that gradients flow back through the RoI pooling layer into the convolution layers too. Training mini-batches are built from **a few images with many RoIs each**, so the proposals of one image share their forward and backward computation (paper, 1 and 2).

## The architecture, part by part
The paper's Figure 1 (the architecture), Figure 2 (VGG16 timing before and after truncated SVD) and Figure 3 (mAP and recall against the number of proposals) are under arXiv's non-exclusive licence, not an open one. I found no free redrawing and copied nothing; the parts are described in words.

The whole design (paper, 2). The network takes "an entire image and a set of object proposals". Several convolution and max-pooling layers turn the image into a conv feature map. For each proposal an RoI pooling layer extracts a fixed-length vector. The vector goes through fully connected layers that branch into two sibling layers: "softmax probability estimates over K object classes plus a catch-all 'background' class", and "four real-valued numbers for each of the K object classes" (the refined box for each class).

### Part by part
- **Conv feature map (shared by all RoIs)**
  - *What it does:* the convolution and pooling layers run once on the whole image (at test time possibly on an image pyramid).
  - *Why it matters:* this is the saving. R-CNN's cost was one full CNN pass per proposal; here one pass serves all about 2,000 proposals (paper, 1 and 3).
- **RoI pooling layer** (paper, 2.1; see [[RoI Pooling]])
  - *What it does:* an RoI is a rectangle $(r, c, h, w)$ (top-left corner, height, width) on the feature map. Its $h \times w$ window is divided into an $H \times W$ grid of sub-windows of "approximate size $h/H \times w/W$", and each is max-pooled, independently per channel. $H$ and $W$ are layer settings (for example 7 × 7), the same for every RoI.
  - *Why it matters:* it gives every proposal, whatever its shape, the same-size input that the fully connected layers need. The paper says it is "the special-case of the spatial pyramid pooling layer used in SPPnets in which there is only one pyramid level" (paper, 2.1), using SPPnet's sub-window formula.
- **Initialising from a pre-trained network** (paper, 2.2)
  - *What it does:* three changes to an ImageNet network: the last max-pooling layer becomes the RoI pooling layer (with $H, W$ matching the first fully connected layer, 7 × 7 for VGG16), the 1000-way layer and softmax are replaced by the two sibling layers, and the network takes two inputs (images and RoI lists).
  - *Why it matters:* any classification network can be turned into a detector; the paper uses three, called S (CaffeNet, essentially [[AlexNet 2012]]), M (VGG_CNN_M_1024, wider than S) and L ([[VGG 2014]]'s VGG16) (paper, 4.1).
- **Two sibling output layers** (paper, 2.3)
  - *What it does:* the first gives $p=(p_0,\dots,p_K)$ over $K+1$ classes by a [[Softmax]]; the second gives, for each class $k$, offsets $t^k=(t^k_x,t^k_y,t^k_w,t^k_h)$ using R-CNN's parameterisation (a scale-invariant shift and a log-space size change, see [[R-CNN 2013]], Appendix C).
  - *Why it matters:* a softmax and the regressors inside one network replace R-CNN's separate SVMs and regressors. The two layers are initialised from zero-mean Gaussians with standard deviations of 0.01 and 0.001 (paper, 2.3).
- **Truncated SVD on the fully connected layers** (paper, 3.1; see [[Singular Value Decomposition]])
  - *What it does:* replace a big weight matrix by two smaller ones (equation 3 below).
  - *Why it matters:* for detection "nearly half of the forward pass time is spent computing the fully connected layers" (paper, 3.1), because there are thousands of RoIs per image; in Figure 2 fc6 and fc7 take 45% of the time before compression.

**Why these numbers and not others?** The paper gives evidence for some choices and leaves others unexplained:
- *Which layers to fine-tune* (VGG16): from conv3_1 up. Starting at conv2_1 slows training 1.3× (12.5 against 9.5 hours) for only +0.3 mAP, and starting at conv1_1 "over-runs GPU memory" (paper, 4.5). For models S and M the paper fine-tunes from conv2; conv1 "is generic".
- *Single scale of 600 pixels* on the short side (long side capped at 1000): chosen "so that VGG16 fits in GPU memory", not tuned per model (paper, 5.2).
- *Batch of 2 images, 128 RoIs:* not tuned in the paper; the paper gives the reason for small $N$ (below), not an ablation of $N$.
- *The $0.1$ lower IoU bound for background:* "appears to act as a heuristic for hard example mining" (paper, 2.3). Not ablated.

### Size and cost check (mine)
From `fastrcnn_demo.py` (Python 3.9.6, NumPy, CPU), using the paper's VGG16 fully connected sizes (25,088 × 4,096 for fc6 and 4,096 × 4,096 for fc7) and its truncation ranks (1,024 and 256):

| Layer | Parameters before | After truncated SVD | Share kept |
|---|---|---|---|
| fc6 | 102,760,448 | 29,884,416 | 29.1% |
| fc7 | 16,777,216 | 2,097,152 | 12.5% |
| Both | 119,537,664 | 31,981,568 | 26.8% (3.74× fewer) |

- **Columns:** *before* is rows × columns; *after* is $t(u+v)$ (equation 3). Biases are left out.
- The 25,088 is 7 × 7 × 512, the RoI-pooled output of VGG16 flattened (my arithmetic; the paper states the matrix size, 25088 × 4096, in 4.4).
- The paper's measured effect of this: time per image from 320 ms to 223 ms, "more than 30%" less (mine: 30.3%), at a cost of 0.3 mAP points (66.9 to 66.6; paper, 4.4 and Figure 2).

## The core mechanism, step by step

### How a mini-batch shares computation (paper, 2.3)
*Idea.* R-CNN and SPPnet drew each training RoI from a different image. Each RoI "may have a very large receptive field, often spanning the entire input image" ([[Receptive Field]]), so one image's worth of forward and backward work is spent for every single RoI. Fast R-CNN samples **hierarchically**: first $N$ images, then $R/N$ RoIs from each image, and "RoIs from the same image share computation and memory in the forward and backward passes".
1. Choose $N=2$ images at random, flipping each horizontally with probability 0.5 (the only augmentation).
   - *What it does:* fixes the number of images that must pass through the conv layers.
   - *Why it matters:* the cost of a mini-batch is mostly the number of images, so a small $N$ makes it cheap.
2. From each image take 64 RoIs ($R=128$ in all): 25% from proposals with IoU of at least 0.5 against a ground-truth box (labelled with that class, $u\ge1$), the rest from proposals whose largest IoU is in $[0.1, 0.5)$ (background, $u=0$).
   - *What it does:* keeps foreground examples from being drowned by background.
   - *Why it matters:* as in R-CNN, positives are rare; 25% is a deliberate bias.
3. One forward and one backward pass through the whole network, with the multi-task loss.

**Worked example (mine).** With $N=2$ and $R=128$, a batch has 32 foreground and 96 background RoIs, 16 and 48 per image. The conv layers process 2 images instead of 128: the paper's "roughly 64× faster" is 128 divided by 2 (mine). The paper adds that RoIs from one image are correlated, "but this concern does not appear to be a practical issue": good results with fewer SGD iterations than R-CNN (paper, 2.3).

### Backpropagation through RoI pooling (paper, 2.3)
*Idea.* Max pooling passes the gradient only to the input cell that was the maximum. Because RoIs overlap, one input cell can be the maximum for several outputs, and its gradients add up.
1. In the forward pass, remember for every output $y_{rj}$ the index $i^*(r,j)$ of its maximum.
2. In the backward pass, send each output's gradient to that index and add.

**Worked example (mine, `fastrcnn_demo.py`).** One channel, activations $x=(1,5,2,7,3,4)$ and two RoIs, covering cells 0 to 3 and cells 2 to 5, each pooled to two outputs of two cells. Outputs: $(5,7)$ and $(7,4)$; argmax indices: 1, 3, 3, 5. With an upstream gradient of 1 on every output, $\partial L/\partial x=(0,1,0,2,0,1)$: cell 3 won twice, so it receives 2.

**Why this matters.** Without it the conv layers could not be fine-tuned together with the rest. SPPnet's fine-tuning "cannot update" them (paper, 1). In the paper's own test, freezing the 13 VGG16 conv layers (training only fc layers, which "emulates single-scale SPPnet training") lowers VOC07 mAP from 66.9% to 61.4% (paper, 4.5, Table 5).

### Scale invariance (paper, 2.3, 3 and 5.2)
Two ways: **brute force** (every image at a fixed size, the network learns scale invariance) or an **image pyramid** (five scales; at test time each RoI is assigned the scale whose version is closest to 224² pixels in area, as in SPPnet). Single-scale is the default for all experiments except 5.2.

### Detection (paper, 3)
Given a fine-tuned network, detection is "little more than running a forward pass". Take about 2,000 precomputed proposals. For each RoI, the confidence for class $k$ is the softmax probability $\Pr(\text{class}=k\mid r)$, and the box is the class's offset applied to the RoI. Then non-maximum suppression runs per class with R-CNN's settings ([[Non-Maximum Suppression]]).

## The equations

**1. The multi-task loss for one RoI** (paper, Eq. 1)

$$
L(p,u,t^{u},v)=L_{\text{cls}}(p,u)+\lambda\,[u\ge 1]\,L_{\text{loc}}(t^{u},v),\qquad L_{\text{cls}}(p,u)=-\log p_u
$$

**Where:**
- $p$: the $K+1$ class probabilities; $u$: the true class of the RoI ($u=0$ is background).
- $t^{u}$: the predicted offsets for the true class $u$ only; $v$: the true regression target.
- $L_{\text{cls}}=-\log p_u$: the log loss of the true class ([[Cross-Entropy Loss]]).
- $[u\ge1]$: 1 for an object class and 0 for background (the paper calls it the Iverson bracket).
- $\lambda$: the balance between the two tasks; all experiments use $\lambda=1$, and the targets $v$ are normalised to zero mean and unit variance (paper, 2.3).

**Matters:**
- The bracket switches off the box term for background RoIs: they have no ground-truth box to move toward.
- Training both terms together lets the two tasks "influence each other through a shared representation" (paper, 5.1).
- Without the regression term the network is a pure classifier; the paper's Table 6 baselines do this ($\lambda=0$).

**Example (tiny numbers):** three classes, $p=(0.1,0.7,0.2)$, true class $u=1$; predicted offsets $t^u=(0.10,-0.20,0.00,0.30)$ and targets $v=(0.15,-0.20,0.50,0.30)$ (my numbers).
- $L_{\text{cls}}=-\ln0.7=0.3567$.
- $L_{\text{loc}}=0.0012+0+0.125+0=0.1263$ (equation 2 below).
- With $\lambda=1$: $L=0.3567+0.1263=0.4829$.
- Meaning: the class term is larger than the box term here, and the one badly-off offset (the width, off by 0.5) accounts for 99% of the box term. A background RoI with $p_0=0.9$ would cost only $-\ln0.9=0.1054$ with the box term off.

**2. The box loss, smooth L1** (paper, Eq. 2 and 3)

$$
L_{\text{loc}}(t^{u},v)=\sum_{i\in\{x,y,w,h\}}\operatorname{smooth}_{L_1}(t^{u}_i-v_i),\qquad
\operatorname{smooth}_{L_1}(x)=\begin{cases}0.5\,x^{2}&|x|<1\\|x|-0.5&\text{otherwise}\end{cases}
$$

**Where:**
- $t^u_i-v_i$: the error of one of the four offsets.
- The first line applies to small errors, the second to large ones; the break is at $|x|=1$.

**Matters:**
- It is "a robust L1 loss that is less sensitive to outliers than the L2 loss used in R-CNN and SPPnet" and removes the learning-rate sensitivity of unbounded targets (paper, 2.3). See [[Smooth L1 Loss]] for the numbers from my own run.

**Example (tiny numbers):** the four errors of the example above are $t^u-v=(-0.05,\,0,\,-0.5,\,0)$. Each is below 1, so each costs half its square: $0.00125$, $0$, $0.125$, $0$. The sum is $0.1263$. An error of 3 would cost $3-0.5=2.5$ and not $0.5	imes3^2=4.5$, which is how an outlier is held back.

**3. Truncated SVD of a fully connected layer** (paper, 3.1)

$$
W\approx U\,\Sigma_t\,V^{\top},\qquad \text{parameters: } uv \;\to\; t(u+v)
$$

**Where:**
- $W$: the $u\times v$ weight matrix of one fully connected layer.
- $U$: $u\times t$, the first $t$ left-singular vectors; $\Sigma_t$: $t\times t$ diagonal matrix of the top $t$ singular values; $V$: $v\times t$, the first $t$ right-singular vectors.
- $t$: the rank kept. The layer is replaced by two layers: the first uses $\Sigma_tV^{\top}$ (no biases), the second uses $U$ (with the original biases), with no non-linearity between them.

**Matters:**
- It is a cheap compression that needs no extra training: "without needing to perform additional fine-tuning after model compression" (paper, 4.4). The saving is large only when $t$ is much smaller than $\min(u,v)$.
- A smaller $t$ is faster but loses more of $W$; the paper uses the top 1,024 for fc6 and 256 for fc7 in VGG16.

**Example (tiny numbers):** a random $60\times40$ matrix (my toy, `fastrcnn_demo.py`) has 2,400 weights. With $t=20$ the two layers have $20\times100=2000$ weights and the matrix has a relative error of 0.405; with $t=5$ they have 500 and the error is 0.812; with $t=40$ (full rank) the error is 0. In every case the two layers gave exactly the same output as the product $\tilde W x$. Meaning: the factorisation itself is exact bookkeeping; the loss comes only from throwing singular values away. A random matrix has no low-rank structure, so these errors are far worse than for a trained layer; the paper's 0.3-point drop is what happened on trained VGG16 layers.

**4. The RoI pooling backward pass** (paper, Eq. 4)

$$
\frac{\partial L}{\partial x_i}=\sum_{r}\sum_{j}\bigl[i=i^{*}(r,j)\bigr]\,\frac{\partial L}{\partial y_{rj}},\qquad i^{*}(r,j)=\arg\max_{i'\in\mathcal R(r,j)}x_{i'}
$$

**Where:**
- $x_i$: the $i$-th activation entering the RoI pooling layer; $y_{rj}$: the $j$-th output for RoI $r$.
- $\mathcal R(r,j)$: the set of inputs inside the sub-window pooled into $y_{rj}$; $i^*(r,j)$: the index of its maximum (the "argmax switch").
- $[i=i^*(r,j)]$: 1 when input $i$ was the maximum of that sub-window.

**Matters:**
- The gradient reaches only the cells that won a max, and gradients add when one cell wins for several outputs or RoIs. This is what lets the conv layers learn from the detection loss.

**Example (tiny numbers):** the six-cell example above: argmax indices 1, 3, 3, 5 with a gradient of 1 each give $\partial L/\partial x=(0,1,0,2,0,1)$.

## What they found
Headline numbers (paper, abstract and 1): VGG16 trained 9× faster than R-CNN and 3× faster than SPPnet, tested 213× faster than R-CNN and 10× faster than SPPnet; 66% mAP on VOC 2012 (the text says "vs. 62% for R-CNN"), 0.3 s per image excluding proposals. Each table below is **written by me, an AI, from the paper's table text**; per-class columns of Tables 1 to 3 are left out.

### VOC 2007, 2010, 2012 test (paper's Tables 1 to 3, mAP column only)
| Method | Trained on | VOC 2007 | VOC 2010 | VOC 2012 |
|---|---|---|---|---|
| SPPnet BB | 07 without "difficult" | 63.1 | | |
| R-CNN BB (VGG16) | 07 / 12 | 66.0 | 62.9 | 62.4 |
| SegDeepM | 12 + segmentation | | 67.2 | |
| Fast R-CNN | 07 | 66.9 | | |
| Fast R-CNN | 07 without "difficult" | 68.1 | | |
| Fast R-CNN | 12 | | 66.1 | 65.7 |
| Fast R-CNN | 07+12 (VOC07) / 07++12 (VOC10, 12) | 70.0 | 68.8 | 68.4 |

**What each column is, and why it matters:**
- **Method, Trained on:** all use VGG16 (except two competitors using a Network-in-Network-style design that I left out) and box regression. "07++12" is VOC07 trainval + test + VOC12 trainval (21.5k images); "difficult" examples are PASCAL's hard-to-see objects, which SPPnet's authors dropped (paper, 4.3 and table captions).
- **mAP:** [[Mean Average Precision]] in percent, higher is better.

**Summary: what the table shows.** On VOC 2012 Fast R-CNN is top (65.7, or 68.4 with extra data), 3.3 points above R-CNN (mine: 65.7 − 62.4). On VOC 2010 SegDeepM beats it with the same data (67.2 against 66.1), but with the larger training set Fast R-CNN reaches 68.8 and passes it. On VOC 2007 it beats SPPnet by 3.8 points (66.9 against 63.1) and R-CNN by 0.9 (66.9 against 66.0), even though SPPnet uses five scales and R-CNN warps each proposal. More data helps: 66.9 to 70.0 on VOC 2007 (+3.1, mine). Caution: single runs; leaderboard rows come from different groups and training sets.

### Training and test time (paper's Table 4)
| | Fast R-CNN S | M | L | R-CNN S | M | L | SPPnet L |
|---|---|---|---|---|---|---|---|
| Train time (hours) | 1.2 | 2.0 | 9.5 | 22 | 28 | 84 | 25 |
| Train speed-up | 18.3× | 14.0× | 8.8× | 1× | 1× | 1× | 3.4× |
| Test time (s per image) | 0.10 | 0.15 | 0.32 | 9.8 | 12.1 | 47.0 | 2.3 |
| Test time with SVD | 0.06 | 0.08 | 0.22 | | | | |
| Test speed-up | 98× | 80× | 146× | 1× | 1× | 1× | 20× |
| Test speed-up with SVD | 169× | 150× | 213× | | | | |
| VOC07 mAP | 57.1 | 59.2 | 66.9 | 58.5 | 60.2 | 66.0 | 63.1 |
| VOC07 mAP with SVD | 56.5 | 58.7 | 66.6 | | | | |

**What each column is, and why it matters:**
- **S, M, L:** the three pre-trained networks (small, medium, VGG16). **Train time:** hours for the whole training. **Test time:** seconds per image *excluding proposals*. **Speed-ups:** against R-CNN with the same network (SPPnet against R-CNN L).
- **SVD:** with the truncated-SVD compression of the fully connected layers.

**Summary: what the table shows.** The biggest gain is for the deepest network: 84 hours to 9.5 hours (8.8×) and 47.0 s to 0.32 s (147× by my division of the printed times; the paper prints 146×, probably from unrounded times) and 213× with SVD. Against SPPnet, VGG16 trains 2.7× faster (the text: 9.5 against 25.5 hours; the table prints 25, and the abstract says "3×") and tests 7× faster (2.3 s against 0.32 s, my 7.2×) or 10× with SVD. Even with all these gains, mAP is equal or higher in all rows except the small nets (S: 57.1 against R-CNN's 58.5, M: 59.2 against 60.2), where Fast R-CNN is slightly lower. Caution: timing excludes the proposal step, which took seconds per image; SPPnet's numbers were provided by its authors and use five scales.

### Which layers to fine-tune (paper's Table 5, VGG16, VOC07)
| Fine-tuned layers | ≥ fc6 | ≥ conv3_1 | ≥ conv2_1 | SPPnet L (≥ fc6, five scales) |
|---|---|---|---|---|
| mAP | 61.4 | 66.9 | 67.2 | 63.1 |
| Test time (s per image) | 0.32 | 0.32 | 0.32 | 2.3 |

**What each column is, and why it matters:** "≥ fc6" trains only the fully connected layers, which "emulates single-scale SPPnet training"; the other columns also train the conv layers from that layer upward.

**Summary: what the table shows.** Training the conv layers is worth +5.5 mAP (66.9 against 61.4, mine) for VGG16, so training through RoI pooling matters. Going further down to conv2_1 adds only 0.3 and slows training 1.3×. Caution: one run each; the paper only tested these three cut points.

### Does multi-task training help? (paper's Table 6, VOC07 mAP)
| Model | Classification loss only | Multi-task, no box regression at test | Stage-wise (classifier first, then regressor, rest frozen) | Multi-task with box regression |
|---|---|---|---|---|
| S | 52.2 | 53.3 | 54.6 | 57.1 |
| M | 54.7 | 55.5 | 56.6 | 59.2 |
| L | 62.6 | 63.4 | 64.0 | 66.9 |

**What each column is, and why it matters:** column 1 is a network trained with $\lambda=0$; column 2 is a multi-task network with the box output ignored at test time, which isolates classification quality; column 3 trains the box layer afterwards on top of a frozen column-1 network; column 4 is the full method.

**Summary: what the table shows.** Multi-task training improves pure classification by +0.8 to +1.1 (mine: 53.3 − 52.2, 55.5 − 54.7, 63.4 − 62.6), and the full method beats stage-wise training by 2.5, 2.6 and 2.9 points (mine). Best is the last column for all three; worst is the baseline. Caution: single runs, and the paper does not say how noisy they are.

### Scales, softmax against SVM, number of proposals (paper's Tables 7 and 8 and Figure 3)
- **Scales (Table 7).** Going from one scale to five: SPPnet's ZF network 58.0 to 59.2 mAP (0.14 to 0.38 s per image), model S 57.1 to 58.4 (0.10 to 0.39 s) and model M 59.2 to 60.7 (0.15 to 0.64 s); the gains are +1.2, +1.3 and +1.5 points (mine) for 2.7 to 4.3 times the time (mine). The paper's reading: "deep ConvNets are adept at directly learning scale invariance", multi-scale gives "only a small increase in mAP at a large cost in compute time", and VGG16 is limited to one scale by memory (66.9 at 0.32 s). I assigned the numbers to models from the order of the extracted columns; the single-scale values for S, M and L agree with Table 4.
- **Softmax against SVM (Table 8).** Softmax beats post-hoc SVMs in all three networks by +0.8, +0.5 and +0.1 (mine: S 57.1 vs 56.3, M 59.2 vs 58.7, L 66.9 vs 66.8). So "one-shot" fine-tuning is enough, and the SVM stage can go. The paper notes that softmax, unlike one-vs-rest SVMs, "introduces competition between classes when scoring a RoI".
- **More proposals (Figure 3).** From 1k to 10k selective-search proposals per image, mAP "rises and then falls slightly" (swamping the classifier "does not help, and even slightly hurts"), while average recall keeps rising, so "AR does not correlate well with mAP" here. Dense boxes (about 45k per image) perform worse: 52.9% for Fast R-CNN and 49.3% for SVMs; adding dense boxes to 2k selective-search boxes lowers mAP to 53.0%. The paper concludes that sparse proposals "appear to improve detector quality", and says this may change if better techniques for dense boxes are found.
- **MS COCO baseline (paper, 5.6).** 35.9% with the PASCAL-style mAP and 19.7% with the COCO-style AP averaged over IoU thresholds, trained on the 80k-image training set for 240k iterations (a "preliminary" result).
- *Caution:* single runs and no error bars; the 2.7 to 4.3 time ratios are 0.38/0.14, 0.39/0.10 and 0.64/0.15 (mine).

## Questions I had, answered

**Why is "RoI pooling" just one-level spatial pyramid pooling?** SPPnet pools each proposal into several grid sizes (for example 6 × 6, 3 × 3, 1 × 1) and joins them. Using only one grid size gives the same operation with one level, which is what Fast R-CNN does, and it makes the backward pass simple (paper, 2.1).

**Why could SPPnet not train the conv layers?** Its training samples one RoI from each of many images, and each RoI's receptive field can span the whole image, so backpropagating through the pooling means big, expensive passes per RoI. Fast R-CNN avoids this by sharing a few images across many RoIs (paper, 2.3).

**Why is the loss only applied to the true class's box?** The network outputs $4K$ numbers, a box for each class, but the loss uses only $t^u$, the box of the true class, since that is the only box with a target (paper, 2.3).

**If softmax removes the need for SVMs, why did R-CNN need them?** In R-CNN the fine-tuned softmax was trained with loose positives and random negatives, and SVMs gave better results (54.2 against 50.9 in R-CNN's Appendix B). Fast R-CNN changes the training of the network itself and finds softmax at least as good (Table 8). The paper does not give a single reason.

**Is this end to end?** Almost. Training is one stage, but the proposals still come from an external method (selective search) and are not learned, and the paper notes that proposal time is excluded from its timings (paper, 1). [[Faster R-CNN 2015]] removes that last piece.

## Why it mattered
It turned detection into a single trainable network that is fast to train and test, and it is the detector half of [[Faster R-CNN 2015]], which keeps its RoI pooling, two sibling heads, multi-task loss and mini-batch sampling and replaces selective search with a learned proposal network. [[Mask R-CNN 2017]] extends the same head. The smooth L1 loss and the 64-RoIs-per-image sampling recur downstream (see [[Smooth L1 Loss]]).

## Limits and open questions
- **Proposals are external and slow:** the paper's timings exclude proposal time; selective search dominates once the network is fast (paper, 1; the size of that time is stated in [[Faster R-CNN 2015]], not here).
- **RoI pooling rounds positions** to whole feature-map cells, which later work ([[RoIAlign]]) changes; this is general knowledge from my RoIAlign note, not a claim in this paper.
- **Single scale for VGG16 because of memory,** and training with N = 2 images is a design for 2015 GPUs.
- **A small gap for the small nets:** Fast R-CNN is slightly below R-CNN for S and M in mAP (Table 4).
- **Dense boxes do worse than sparse proposals,** which the authors leave as an open question.
- **Hand-set numbers:** 25% foreground, 0.5 and 0.1 IoU thresholds, 600 pixels, 40k iterations, $\lambda=1$, 1024 and 256 SVD ranks; the paper ablates few of them.
- **Single runs, no error bars** in every table.
- **Unclear at one point:** the abstract says 3× faster than SPPnet in training, the text says 2.7× and the table prints SPPnet as 25 hours against 25.5 in the text.

## Concepts used
[[Object Detection]], [[Convolutional Neural Network]], [[Feature Map]], [[RoI Pooling]], [[Smooth L1 Loss]], [[Cross-Entropy Loss]], [[Softmax]], [[Backpropagation]], [[Stochastic Gradient Descent]], [[Momentum]], [[Singular Value Decomposition]], [[Fine-tuning]], [[Non-Maximum Suppression]], [[Intersection over Union]], [[Mean Average Precision]], [[Receptive Field]], [[Selective Search]], [[Spatial Pyramid Pooling]]

## Reading trail
- **Came from:** [[R-CNN 2013]] and SPPnet (Spatial Pyramid Pooling, not yet a note)
- **Led to:** [[Faster R-CNN 2015]], then [[Mask R-CNN 2017]]; compare [[YOLO 2015]]

## How much I read
- **Text:** the whole body of arXiv 1504.08083v2 (9 pages, extracted with PyMuPDF): abstract, introduction, sections 2 to 6, the conclusion and the tables as printed in the extraction. The reference list was not read. Tables 1 to 3 per-class columns were not read.
- **Numbers:** "paper" marks are from the text and tables. Differences, percentages and the parameter counts are mine (`fastrcnn_demo.py`, run once). Tables 4 and 7 were assigned to columns from the extraction order; Table 4 agrees with the paper's own sentences (9.5 against 84 hours, 0.32 against 47.0 s), and Table 7's single-scale values agree with Table 4.
- **Not checked:** the figures as pictures (none shown here: no open licence), SPPnet's own paper, whether the first version (v1) had different numbers.
- The note was not opened in Obsidian and the flashcards are unreviewed.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Fast R-CNN (Girshick, arXiv 1504.08083): what problem did it attack?::R-CNN trained in three stages, cached features on disk (hundreds of GB) and took 47 s per image with VGG16 because it ran the CNN on every proposal; SPPnet shared computation but could not fine-tune the conv layers (paper, 1).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What is the idea of Fast R-CNN?::Run the conv layers once per image, pool each proposal's part of the feature map to a fixed size (RoI pooling), end in two sibling outputs (softmax classes and per-class box offsets) and train everything with one multi-task loss, with mini-batches of few images and many RoIs each (paper, 1 and 2).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What did SPPnet do, and what limit did Fast R-CNN fix?::SPPnet computes one feature map per image and pools a vector per proposal, speeding R-CNN 10 to 100× at test time; but its training is still multi-stage and cannot update the conv layers before the pooling, which limits very deep networks (paper, 1).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How does the RoI pooling layer work?::An RoI is a rectangle on the feature map; its h by w window is divided into an H by W grid of sub-windows (about h/H by w/W) and each is max-pooled per channel, giving a fixed-size output for any RoI. It is spatial pyramid pooling with one level (paper, 2.1).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What three changes turn a pre-trained ImageNet network into a Fast R-CNN network?::Replace the last max-pooling layer with an RoI pooling layer (H, W matched to the first fully connected layer); replace the 1000-way layer and softmax with the two sibling layers (K+1 softmax and per-class box regressors); make it take two inputs, images and RoI lists (paper, 2.2).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How does Fast R-CNN build a training mini-batch, and why?::N = 2 images, 64 RoIs from each (R = 128): 25% from proposals with IoU of at least 0.5 (foreground), the rest with IoU in [0.1, 0.5) (background); RoIs of one image share forward and backward computation, "roughly 64× faster" than one RoI from each of 128 images (128 / 2, my arithmetic) (paper, 2.3).

#flashcards/research/formula #flashcards/topic/research-detection-segmentation Write Fast R-CNN's multi-task loss and explain each part.
?
$$L(p,u,t^{u},v)=-\log p_u+\lambda\,[u\ge 1]\,L_{\text{loc}}(t^{u},v)$$
- $p$: class probabilities over K+1 classes; $u$: true class (0 is background).
- $-\log p_u$: log loss of the true class.
- $t^u$, $v$: predicted and true box offsets for class $u$.
- $[u\ge1]$: switches the box term off for background RoIs; $\lambda=1$ balances the tasks (targets normalised to zero mean, unit variance) (paper, Eq. 1).

#flashcards/research/example #flashcards/topic/research-detection-segmentation p = (0.1, 0.7, 0.2), true class 1, predicted offsets (0.10, -0.20, 0.00, 0.30), targets (0.15, -0.20, 0.50, 0.30): what is the Fast R-CNN loss?::Lcls = -ln 0.7 = 0.3567; Lloc = 0.0012 + 0 + 0.125 + 0 = 0.1263 (smooth L1); L = 0.4829 with lambda 1. A background RoI with p0 = 0.9 would cost 0.1054 with the box term off (my arithmetic).

#flashcards/research/formula #flashcards/topic/research-detection-segmentation Write the Fast R-CNN box loss (smooth L1) and say why it replaced L2.
?
$$L_{\text{loc}}=\sum_{i\in\{x,y,w,h\}}\operatorname{smooth}_{L_1}(t^{u}_i-v_i),\quad \operatorname{smooth}_{L_1}(x)=\begin{cases}0.5x^{2}&|x|<1\\|x|-0.5&\text{otherwise}\end{cases}$$
- It sums four offset errors, only for the true class's box.
- Quadratic for small errors, linear for large ones, so it is "less sensitive to outliers" than the L2 loss of R-CNN and SPPnet and avoids exploding gradients with unbounded targets (paper, Eq. 2 and 3).

#flashcards/research/formula #flashcards/topic/research-detection-segmentation Write the truncated SVD used in Fast R-CNN and explain each part.
?
$$W\approx U\,\Sigma_t\,V^{\top},\qquad uv\;\to\;t(u+v)\ \text{parameters}$$
- $W$: u by v weight matrix of a fully connected layer.
- $U$ (u by t) and $V$ (v by t): first t singular vectors; $\Sigma_t$: top t singular values.
- The layer becomes two layers: first $\Sigma_tV^{\top}$ (no biases), then $U$ (original biases), no non-linearity between (paper, 3.1).

#flashcards/research/example #flashcards/topic/research-detection-segmentation How much does truncated SVD shrink VGG16's fc6 and fc7 with ranks 1,024 and 256?::fc6 (25,088 by 4,096) from 102,760,448 to 29,884,416 parameters and fc7 (4,096 by 4,096) from 16,777,216 to 2,097,152; 119.5 million to 32.0 million together, 3.74× fewer (my arithmetic, biases ignored). The paper measured 320 ms to 223 ms per image for 0.3 mAP.

#flashcards/research/formula #flashcards/topic/research-detection-segmentation Write the RoI pooling backward pass and explain each part.
?
$$\frac{\partial L}{\partial x_i}=\sum_{r}\sum_{j}\bigl[i=i^{*}(r,j)\bigr]\frac{\partial L}{\partial y_{rj}},\quad i^{*}(r,j)=\arg\max_{i'\in\mathcal R(r,j)}x_{i'}$$
- $x_i$: input activation; $y_{rj}$: j-th output of RoI r.
- $\mathcal R(r,j)$: inputs in the sub-window pooled into $y_{rj}$; $i^*$: its argmax.
- Each gradient goes only to the argmax cell, and gradients add when a cell wins several outputs (paper, Eq. 4).

#flashcards/research/example #flashcards/topic/research-detection-segmentation Activations (1, 5, 2, 7, 3, 4), two RoIs on cells 0 to 3 and 2 to 5, each max-pooled into two outputs of two cells: what is the gradient with unit upstream gradients?::Outputs (5, 7) and (7, 4), argmax indices 1, 3, 3, 5, so dL/dx = (0, 1, 0, 2, 0, 1): cell 3 won twice (my toy, run in Python).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Why does the conv-layer fine-tuning matter in Fast R-CNN?::With VGG16, freezing the 13 conv layers (training only fc layers, like single-scale SPPnet) lowers VOC07 mAP from 66.9 to 61.4; training from conv3_1 up is enough, since conv2_1 adds only 0.3 and slows training 1.3× (paper, 4.5, Table 5).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How fast was Fast R-CNN against R-CNN and SPPnet with VGG16?::Training 9.5 h against 84 h (8.8×) for R-CNN and about 25 h for SPPnet; test 0.32 s against 47.0 s (146×) and 2.3 s; 0.22 s (213×) with truncated SVD. Times exclude proposals, K40 GPU (paper, Table 4).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Did multi-task training help Fast R-CNN?::Yes: with the box output ignored at test time it still improved classification by +0.8 to +1.1 mAP over a classification-only network, and the full method beat stage-wise training by 2.5 to 2.9 points (paper, Table 6; differences are my arithmetic).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Did Fast R-CNN still need SVMs?::No: softmax beat post-hoc SVMs by +0.1 to +0.8 mAP on VOC07 for all three networks, so one-shot fine-tuning is enough; softmax also makes classes compete when scoring an RoI (paper, 5.4, Table 8).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Are more object proposals always better in Fast R-CNN?::No. mAP rises and then falls slightly from 1k to 10k selective-search proposals while average recall keeps rising, and about 45k dense boxes per image give only 52.9% mAP (adding dense boxes to 2k selective-search boxes lowers it to 53.0%), so recall is not a safe proxy for mAP (paper, 5.5, Figure 3).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How did Fast R-CNN do on the VOC benchmarks?::VOC 2012: 65.7 mAP (68.4 with extra data) against R-CNN's 62.4; VOC 2010: 66.1 (68.8 with extra data) against SegDeepM's 67.2; VOC 2007: 66.9 against SPPnet's 63.1 and R-CNN's 66.0 (paper, Tables 1 to 3).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What is the main limit of Fast R-CNN?::The proposals still come from an external, slow method (selective search) and the timings exclude them; RoI pooling rounds to whole cells; single scale for VGG16 because of GPU memory. Faster R-CNN fixes the proposal step (paper, 1; the later points are my reading).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Fast R-CNN: reading trail and how much did my note read?::It came from R-CNN and SPPnet and led to Faster R-CNN and Mask R-CNN. The note read the whole body text; not read: the figures as pictures, the reference list and SPPnet's own paper.
