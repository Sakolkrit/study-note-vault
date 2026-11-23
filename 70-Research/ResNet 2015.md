---
tags: [paper, vision, CNN, residual]
link: https://arxiv.org/abs/1512.03385
authors: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research; checked on the paper's first page)
year: 2015
status: body-read
field: image classification
---

# ResNet 2015

> **In one line for a 12-year-old:** Let every block of layers add a small correction to its input instead of rebuilding it. Then a network can be very deep without getting lost.

## The problem
Deeper networks are more difficult to train (first line of the abstract). The paper shows something stranger than the old vanishing-gradient problem: the **degradation problem**. When a plain network gets deeper, its accuracy first saturates and then gets worse, and the worse number is on the *training* data, so it is not over-fitting (paper, introduction). On CIFAR-10 a 56-layer plain net had higher training error and higher test error than a 20-layer one (paper, Figure 1; I read the caption and the text, not the curves).

The authors argue this should not happen: a deeper model could simply copy a shallower one and set the extra layers to "do nothing" (identity). Since the solver does not find even that, "not all systems are similarly easy to optimize" (paper).

## What people did before
Plain stacks such as [[VGG 2014]] of 16 to 19 layers, kept trainable with careful initialisation and [[Batch Normalization]] (paper, introduction).

## The idea
Let the stacked layers learn the **difference** from the input, not the whole answer: the shortcut ([[Residual Connection]]). If the best thing for a block to do is nothing, it only has to push the difference to zero.

## The architecture (figures)
The paper's Figure 2 is the **building block**: the input $x$ goes through two weight layers, giving $F(x)$, and a shortcut carries $x$ around them; the two are added, $F(x)+x$, and a ReLU follows. The shortcut "skips one or more layers" and "adds neither extra parameter nor computational complexity" (paper).

Written from section 3 of the paper:
- **Plain network (Figure 3, middle):** mostly 3×3 convolutions, with two rules: the same number of filters for the same output size, and doubling the filters when the map size halves. Down-sampling is by stride 2. It ends with global average pooling and a 1000-way fully connected layer. The 34-layer plain net has 3.6 billion FLOPs, 18% of VGG-19's 19.6 billion.
- **Residual network (Figure 3, right):** the same, plus a shortcut around every pair of 3×3 layers. When the size changes (dotted shortcuts) there are two options: **A** pad with zeros (no extra parameters), or **B** a 1×1 convolution projection.
- **Bottleneck block (Figure 5, right)** for ResNet-50, 101 and 152: three layers, 1×1 (shrink), 3×3, 1×1 (restore). It keeps the 3×3 layer cheap while the shortcut joins the wide ends.
- **Depth counts (Table 1):** 18, 34, 50, 101 and 152 weighted layers. Five stages ("conv1" to "conv5_x") at output sizes 112, 56, 28, 14 and 7.

### The five networks (paper's Table 1)

**Table 1: the five ImageNet networks** (paper's Table 1; "basic" = two 3×3 layers, "bottleneck" = 1×1, 3×3, 1×1; blocks per stage conv2_x, conv3_x, conv4_x, conv5_x; FLOPs counted as multiply-adds).

| Depth | Block type | Blocks per stage | FLOPs |
|---|---|---|---|
| 18 | basic | 2, 2, 2, 2 | $1.8\times10^9$ |
| 34 | basic | 3, 4, 6, 3 | $3.6\times10^9$ |
| 50 | bottleneck | 3, 4, 6, 3 | $3.8\times10^9$ |
| 101 | bottleneck | 3, 4, 23, 3 | $7.6\times10^9$ |
| 152 | bottleneck | 3, 8, 36, 3 | $11.3\times10^9$ |

*Written by me, an AI, from the paper's table text and checked against the extracted text; it is not a copy of the paper's table.*

**What each column is, and why it matters**
- **Depth:** the number of layers that have weights (convolutions plus the last fully connected layer). It matters because depth was the whole question: before ResNet, going deeper made plain networks worse.
- **Block type:** "basic" is two 3×3 convolutions; "bottleneck" is a 1×1 convolution that shrinks the number of channels, a 3×3, and a 1×1 that restores them. It matters because the bottleneck is cheaper per layer, so much deeper networks stay affordable.
- **Blocks per stage:** how many blocks sit in each of the four stages (conv2_x to conv5_x), which work on maps 56, 28, 14 and 7 pixels wide. It matters because it shows where the depth is spent: the 152-layer net puts 36 blocks in the third stage.
- **FLOPs:** the multiply-adds needed for one image, a measure of running cost. It matters because it lets the paper compare with VGG on cost: 11.3 billion for ResNet-152 against 15.3 billion for VGG-16 and 19.6 billion for VGG-19 (paper).

Check (mine): $1+2\cdot(3+4+6+3)+1=34$ weighted layers for the basic net and $1+3\cdot(3+8+36+3)+1=152$ for the deepest; the first and last "1" are the first convolution and the final fully connected layer. The 152-layer net has 11.3 billion FLOPs, against 15.3 billion for VGG-16 and 19.6 billion for VGG-19 (paper).

![[residual-block.png]]
*A residual block. The input $x$ goes down through two layers, which together compute $F(x)$. At the same time, $x$ is copied along the curved "identity" path on the right, which does nothing to it, and the two are added (the circled plus). The output is $x+F(x)$. Image: LunarLullaby, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:ResBlock.png), CC BY-SA 4.0. This is the idea of the paper's Figure 2, drawn by someone else, not the paper's own picture.*

![[resnet-basic-vs-bottleneck.svg]]
*Two kinds of residual block: (a) a basic block with two 3×3 convolutions, (b) a bottleneck block with 1×1, 3×3, 1×1 convolutions, where the 3×3 works at a quarter of the width $D$. The input $x$ runs up the right side and is added back by the circled plus. Image: Cosmia Nebula, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:ResBlockVariants-stream.svg), CC0. This matches the two blocks of the paper's Figure 5 (basic block on the left, bottleneck on the right). The paper's version shows widths 64 and 256 and has a ReLU between layers, which this drawing leaves out.*

The paper's own page has no licence statement that would let me copy its figures (arXiv's standard distribution licence applies), so I use the free drawings above and tables written by me. The paper's Figure 3 (the 34-layer network next to VGG-19) and Figure 4 (training curves) are not copied; Table 1 below gives the layout in words.

**Summary: what the table shows**
- **Cost grows with depth, but slowly with the bottleneck:** from 18 to 152 layers (8.4 times deeper) the FLOPs rise from 1.8 to 11.3 billion (6.3 times, my arithmetic).
- **A telling pair:** the 34-layer net (3.6 billion) and the 50-layer net (3.8 billion) cost almost the same, although the 50-layer one has 16 more layers, because of the bottleneck blocks.
- **Still cheaper than VGG:** even ResNet-152 (11.3 billion) is below VGG-16 (15.3 billion) and VGG-19 (19.6 billion).

## The equations (paper, section 3)

**1. The residual idea** (section 3.1)

$$
F(x):=H(x)-x,\qquad\text{so}\qquad H(x)=F(x)+x
$$

**Where:**
- $H(x)$: the mapping we actually want a few stacked layers to produce.
- $x$: the input to those layers.
- $F(x)$: what the layers really learn, the "residual" (the leftover difference).

**Matters:** the authors hypothesise that it is easier to optimise the residual than the original mapping. If the best mapping were the identity, it is easier to push the residual to zero than to make a stack of nonlinear layers copy the input (paper). They say that this is a hypothesis; Figure 7 supports it with small layer responses.

**Example (tiny numbers):** the wanted mapping gives $H(x)=(1,2)$ for the input $x=(1,2)$ (it should change nothing).
- Without shortcut the layers must produce $(1,2)$ exactly.
- With shortcut they need $F(x)=H(x)-x=(0,0)$, which is "all weights zero".
- Meaning: doing nothing is the easiest thing for a residual block.

**2. The building block** (equation 1, with the two-layer example)

$$
y=F(x,\{W_i\})+x,\qquad F=W_2\,\sigma(W_1x)
$$

**Where:**
- $x$, $y$: the input and output vectors of the block.
- $F(x,\{W_i\})$: the residual function, with learnable weights $W_1$, $W_2$ (biases left out for simplicity, paper).
- $\sigma$: the ReLU. The paper applies a second ReLU after the addition, $\sigma(y)$.
- The "$+x$" is the shortcut, an element-wise addition (channel by channel for feature maps).

**Matters:** the shortcut adds no parameters and almost no work, so a plain network and its residual twin can be compared fairly with the same parameters, depth, width and cost (paper). $F$ must have the same size as $x$.

**Example (tiny numbers):** $x=(1,2)$, $W_1=\begin{pmatrix}1&-1\\0&1\end{pmatrix}$, $W_2=\begin{pmatrix}1&0\\1&1\end{pmatrix}$.
- $W_1x=(1-2,\ 0+2)=(-1,\,2)$; ReLU gives $(0,\,2)$.
- $F=W_2(0,2)=(0,\,2)$.
- $y=F+x=(1,\,4)$; the second ReLU leaves $(1,4)$.
- Meaning: the block kept the input and added a small correction $(0,2)$ on top.

**3. Projection shortcut** (equation 2)

$$
y=F(x,\{W_i\})+W_s\,x
$$

**Where:**
- $W_s$: a learned matrix (done by a 1×1 convolution in the network) that changes $x$ to the size of $F$.
- $x$, $y$ and $F$ are as in the building block above.

**Matters:** it is used only when the input and output sizes differ, for example when the number of channels grows. The paper shows the identity shortcut is enough to fix degradation and is cheaper, so $W_s$ is used only to match sizes (paper, section 3.2 and Table 3).

**Example (tiny numbers):** $x=(1,2)$ must be added to a three-number $F=(0.5,-0.5,1)$.
- $W_s=\begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix}$ gives $W_sx=(1,2,3)$.
- $y=(0.5+1,\ -0.5+2,\ 1+3)=(1.5,\,1.5,\,4)$.
- Meaning: the projection stretches the input to three numbers so the two can be added.
- Cost check (my arithmetic, 256 channels at the ends of a bottleneck): the bottleneck's three layers (1×1 from 256 to 64, 3×3 at 64, 1×1 from 64 to 256) hold $16{,}384+36{,}864+16{,}384=69{,}632$ weights; a $256\times256$ projection adds $65{,}536$, so $135{,}168$ in all, about $1.9$ times as many. The paper says replacing the identity shortcut with a projection doubles the time and model size of the bottleneck.

## What they found
- Residual nets with up to 152 layers, 8 times deeper than VGG but lower complexity. An ensemble reached 3.57% error on the ImageNet test set and won ILSVRC 2015 classification. Also 28% relative improvement on COCO detection, and wins on ImageNet detection and localisation, and COCO detection and segmentation (abstract; the numbers are also in Tables 5 and 7 to 8, paper).
- The degradation problem is real and is fixed by shortcuts (Table 2).
- Training setup (paper, 3.4): SGD, mini-batch 256, learning rate 0.1 divided by 10 when the error plateaus, up to $60\times10^4$ iterations, weight decay 0.0001, momentum 0.9, [[Batch Normalization]] after each convolution, no dropout.
- The learned residual functions tend to have **small responses**, smaller in deeper nets (Figure 7, read in the text). This supports the idea that layers only make small corrections.

### Do shortcuts fix the degradation problem? (paper's Table 2)

**Table 2: plain against residual, top-1 error (%) on ImageNet validation, 10-crop** (paper's Table 2; residual nets use option A, so no extra parameters).

| Depth | Plain | ResNet | Gain (mine) |
|---|---|---|---|
| 18 layers | 27.94 | 27.88 | 0.06 |
| 34 layers | 28.54 | 25.03 | 3.51 |

*Written by me, an AI, from the paper's table text and checked against the extracted text; it is not a copy of the paper's table.*

**What each column is, and why it matters**
- **Depth:** 18 or 34 layers, the two sizes compared.
- **Plain:** the network with no shortcuts. **ResNet:** the same layers with a shortcut around every pair of 3×3 layers (option A, which adds no parameters). It matters because the two columns have exactly the same number of parameters, so any difference comes only from the shortcuts.
- **Numbers:** top-1 error (%) on the ImageNet validation set: the share of images where the model's single best guess is wrong, so lower is better. Testing used "10-crop", meaning each image is cropped ten ways (the paper cites the standard practice) and the predictions are averaged (general knowledge).
- **Gain (mine):** plain minus ResNet; a positive number means the shortcuts helped.

The plain net gets *worse* from 18 to 34 layers (27.94 to 28.54); the ResNet gets better (27.88 to 25.03). That is the degradation problem and its fix in two rows. The paper also says the 18-layer ResNet converges faster in training (Figure 4, which I did not view).

**Summary: what the table shows**
- **Best:** the 34-layer ResNet, 25.03% top-1 error. **Worst:** the 34-layer plain net, 28.54%.
- **The key result:** deeper made the plain net worse (27.94 to 28.54) but the ResNet better (27.88 to 25.03), a gain of 3.51 points at 34 layers with identical parameters.
- **At 18 layers it is a tie** (0.06 apart); the paper says the shortcuts only help the 18-layer net converge faster.

### How the depths compare (paper's Table 3)

**Table 3: error rates (%) on ImageNet validation, 10-crop** (paper's Table 3; options A, B, C are the shortcut types; ResNet-50/101/152 use B).

| Model | Top-1 | Top-5 |
|---|---|---|
| VGG-16 (their test) | 28.07 | 9.33 |
| GoogLeNet | - | 9.15 |
| PReLU-net | 24.27 | 7.38 |
| plain-34 | 28.54 | 10.02 |
| ResNet-34 A | 25.03 | 7.76 |
| ResNet-34 B | 24.52 | 7.46 |
| ResNet-34 C | 24.19 | 7.40 |
| ResNet-50 | 22.85 | 6.71 |
| ResNet-101 | 21.75 | 6.05 |
| ResNet-152 | 21.43 | 5.71 |

*Written by me, an AI, from the paper's table text and checked against the extracted text; it is not a copy of the paper's table.*

**What each column is, and why it matters**
- **Model:** the network. The letters A, B, C after ResNet-34 are the three shortcut options: A pads with zeros when sizes change, B uses a 1×1 projection only when sizes change, C uses projections on all shortcuts. ResNet-50, 101 and 152 use B.
- **Top-1 error:** the share of validation images where the single best guess is wrong. **Top-5 error:** the share where the true class is not among the five best guesses. Lower is better for both. They matter because ImageNet has 1,000 classes, many of them similar, so top-5 is more forgiving; a dash means no number was given.
- **Why the table matters:** it shows that the three shortcut options differ by under one point (so projections are not essential) and that deeper ResNets keep improving.

A, B and C differ by under one point; the authors say projection shortcuts are "not essential" for fixing degradation.

**Summary: what the table shows**
- **Best:** ResNet-152 on both measures (21.43 top-1, 5.71 top-5). The error falls with every step of depth: ResNet-34 A 25.03, B 24.52, C 24.19, then 50 layers 22.85, 101 layers 21.75 and 152 layers 21.43.
- **Worst:** the plain 34-layer net (28.54 and 10.02); VGG-16 (28.07 and 9.33) and GoogLeNet (top-5 9.15) are behind every ResNet from 50 layers up.
- **ResNet-34 is competitive but not clearly ahead:** it is about level with PReLU-net (24.27 and 7.38); the ResNets win clearly from 50 layers.
- **Shortcut options:** A, B and C differ by under one point (25.03, 24.52, 24.19), so projection shortcuts are a small gain.

### Against the best earlier single models (paper's Table 4)

**Table 4 (excerpt): single-model error (%) on ImageNet validation** (paper's Table 4, my selection of rows).

| Model | Top-1 | Top-5 |
|---|---|---|
| VGG (v5) | 24.4 | 7.1 |
| PReLU-net | 21.59 | 5.71 |
| BN-Inception | 21.99 | 5.81 |
| ResNet-50 | 20.74 | 5.25 |
| ResNet-101 | 19.87 | 4.60 |
| ResNet-152 | 19.38 | 4.49 |

*Written by me, an AI, from the paper's table text and checked against the extracted text; it is not a copy of the paper's table.*

**What each column is, and why it matters**
- **Method and model:** the networks compared, with single-model results (no ensemble). VGG (v5) and PReLU-net are other leading networks of the time.
- **Top-1 and top-5 error:** as in the previous table, on the validation set.
- **Why the table matters:** it compares ResNet with the best earlier single models: the single 152-layer ResNet's 4.49% top-5 error is lower than all earlier ensembles' (paper, Table 5).

**Summary: what the table shows**
- **Best:** ResNet-152, 19.38 top-1 and 4.49 top-5. Every ResNet from 50 layers up beats every other single model listed (PReLU-net 21.59 and 5.71, BN-Inception 21.99 and 5.81, VGG 24.4 and 7.1).
- **Size of the lead:** ResNet-152 is 5.0 points better than VGG (v5) in top-1 and 2.6 points in top-5 (my arithmetic).
- **ResNet-34 B matches PReLU-net in top-5** (5.71 each), so a 34-layer ResNet is already level with the best earlier network in top-5.

### The winning ensemble (paper's Table 5)

**Table 5: ensembles, top-5 error (%) on the ImageNet test set** (paper's Table 5, from the test server).

| Method | Top-5 |
|---|---|
| VGG (ILSVRC 2014) | 7.32 |
| GoogLeNet (ILSVRC 2014) | 6.66 |
| PReLU-net | 4.94 |
| BN-Inception | 4.82 |
| **ResNet (ILSVRC 2015)** | **3.57** |

*Written by me, an AI, from the paper's table text and checked against the extracted text; it is not a copy of the paper's table.*

**What each column is, and why it matters**
- **Method:** the system, here whole competition entries. An ensemble combines several models.
- **Top-5 (test):** the top-5 error on the hidden ImageNet test images, scored by the competition's test server, so nobody can tune on them. It matters because it is the official ranking number: 3.57% won the ILSVRC 2015 classification task.

The 3.57% entry combined six models of different depth (two of them 152-layer at the time of submission). The paper also says one single 152-layer model with 4.49% already beat all earlier ensembles.

**Summary: what the table shows**
- **Best:** the ResNet ensemble, 3.57% top-5 test error, 1.25 points lower than the next best (BN-Inception, 4.82) and 3.75 lower than VGG (7.32), my arithmetic.
- **Every earlier entry listed is above 4.8%**, so the result is a clear lead in the competition table.

### A small data set and very deep nets: CIFAR-10 (paper's Table 6)

**Table 6 (excerpt): CIFAR-10 test error (%)** (paper's Table 6, with data augmentation; I left out some rows).

| Model | Layers | Parameters | Error |
|---|---|---|---|
| Highway | 19 | 2.3M | 7.54 |
| Highway | 32 | 1.25M | 8.80 |
| ResNet | 20 | 0.27M | 8.75 |
| ResNet | 56 | 0.85M | 6.97 |
| ResNet | 110 | 1.7M | 6.43 (mean 6.61, spread 0.16 over 5 runs) |
| ResNet | 1202 | 19.4M | 7.93 |

*Written by me, an AI, from the paper's table text and checked against the extracted text; it is not a copy of the paper's table.*

**What each column is, and why it matters**
- **Model:** the network. Highway nets and FitNet are other deep, thin networks.
- **Layers:** how many weighted layers. The 1202-layer net is the over-1000-layer experiment.
- **Parameters:** the number of trained values in millions, a measure of size. It matters because the 110-layer ResNet has fewer parameters than FitNet and the 19-layer Highway net yet scores better.
- **Error:** the test error (%) on CIFAR-10, a small data set of 10 classes. "Mean 6.61, spread 0.16 over 5 runs" says how much the result varied between runs of the 110-layer net. It matters because a single lucky run could mislead.

The 1202-layer net trains without difficulty (training error below 0.1%) but its test error is worse than the 110-layer one. The authors blame over-fitting, since 19.4 million parameters is "unnecessarily large" for this small data set (paper). That is their argument, not a test.

**Summary: what the table shows**
- **Best:** ResNet-110, 6.43% (best of 5 runs; the mean was 6.61 with a spread of 0.16).
- **Error falls with depth for ResNets up to 110 layers:** 8.75, 7.51, 7.17, 6.97 and 6.43 for 20, 32, 44, 56 and 110 layers. The Highway networks go the other way here (7.54 at 19 layers, 8.80 at 32).
- **Too deep:** the 1202-layer ResNet is worse (7.93) than the 110-layer one; the authors blame over-fitting on a small data set.

### Another task: detection (paper's Tables 7 and 8)

**Tables 7 and 8: detection with Faster R-CNN, swapping only the backbone** ([[Faster R-CNN 2015]]).

| Metric | VGG-16 | ResNet-101 |
|---|---|---|
| PASCAL VOC 2007 test, mAP (%) | 73.2 | 76.4 |
| PASCAL VOC 2012 test, mAP (%) | 70.4 | 73.8 |
| COCO val, mAP@.5 (%) | 41.5 | 48.4 |
| COCO val, mAP@[.5, .95] (%) | 21.2 | 27.2 |

*Written by me, an AI, from the paper's table text and checked against the extracted text; it is not a copy of the paper's table.*

**What each column is, and why it matters**
- **Metric:** the detection score. mAP is the mean average precision ([[Mean Average Precision]]). The part after the "at" is the overlap needed between a predicted box and the true box for the detection to count as correct ([[Intersection over Union]]): at least 0.5 for mAP@.5, and the average over thresholds from 0.5 to 0.95 for mAP@[.5, .95], which is stricter.
- **PASCAL VOC and COCO:** two object-detection data sets; the table gives VOC 2007 and 2012 test sets and the COCO validation set.
- **VGG-16 and ResNet-101:** the two backbones swapped into the same Faster R-CNN detector ([[Faster R-CNN 2015]]). It matters because the rest of the detector is identical, so the gain comes only from the better network.

My arithmetic: on COCO the standard metric rises by 6.0 points, a relative gain of 28.3%, which is the "28% relative improvement" of the abstract.

**Summary: what the table shows**
- **ResNet-101 wins on every metric** over VGG-16 with the same detector: VOC 2007 +3.2 points, VOC 2012 +3.4, COCO mAP@.5 +6.9 and COCO mAP@[.5, .95] +6.0 (my arithmetic).
- **Biggest gain on the strictest metric in relative terms:** 21.2 to 27.2 on mAP@[.5, .95], a 28% relative gain, the abstract's headline number.

## Why it mattered
Shortcuts became standard, including inside every [[Transformer]] block.

## Limits and open questions
Still local patterns in a fixed grid. Later work asked whether attention could replace convolution ([[Vision Transformer]]). The authors themselves list open problems: why plain deep nets are hard to optimise ("will be studied in the future"), and that 1000+ layer nets over-fit on small data (paper).

## Concepts used
[[Residual Connection]], [[Vanishing and Exploding Gradients]], [[Normalization]], [[Batch Normalization]], [[ResNet]]

## Reading trail
- **Came from:** [[VGG 2014]]
- **Led to:** [[ViT 2020]], [[N-BEATS 2019]] (doubly residual blocks for forecasting)

## How much I read
Body read as text: pages 1 to 8 (introduction, related work, method, experiments, Tables 1 to 8) of the arXiv PDF (v1). I did not view the pictures: Figure 1 (CIFAR-10 curves), Figure 3 (the 34-layer networks), Figure 4 (ImageNet curves), Figure 6 and Figure 7 were read only through their captions and the text. Not read: the reference list and the appendix (detection and localisation details, about pages 10 to 12). The equations and numbers marked "paper" come from the body. The "my arithmetic" numbers are from my own Python run (NumPy). The licences of the two figures were read on their Commons pages. Not opened in Obsidian, so the flashcards are unreviewed.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-vision ResNet (He et al., 2015, arXiv 1512.03385): what problem did it attack? (first line of the abstract)::Deeper networks are more difficult to train.

#flashcards/research/concept #flashcards/topic/research-vision What did people do before ResNet?::Plain stacks such as VGG of 16 to 19 layers.

#flashcards/research/concept #flashcards/topic/research-vision ResNet: what was the idea?::Reformulate the layers as learning residual functions with reference to their inputs: the shortcut. Each block adds a small correction instead of rebuilding its input.

#flashcards/research/concept #flashcards/topic/research-vision ResNet: what did it find? (abstract)::Residual nets with up to 152 layers, 8 times deeper than VGG but lower complexity. An ensemble reached 3.57% error on the ImageNet test set and won ILSVRC 2015 classification. Also a 28% relative improvement on COCO detection, and wins on ImageNet detection and localisation, and COCO detection and segmentation.

#flashcards/research/concept #flashcards/topic/research-vision ResNet: what is its limit?::Still local patterns in a fixed grid. Later work asked whether attention could replace convolution (Vision Transformer).

#flashcards/research/concept #flashcards/topic/research-vision ResNet: why did it matter and what is its place in the reading trail?::Shortcuts became standard, including inside every Transformer block. It came from VGG and led to ViT.

#flashcards/research/concept #flashcards/topic/research-vision How much of this paper did the note read?::The body as text (pages 1 to 8 of the arXiv PDF); the figures only through captions and text, and not the references or the appendix. Figures in the note are free drawings from Wikimedia Commons, and the tables are written by me from the paper.

#flashcards/research/concept #flashcards/topic/research-vision What is the degradation problem?::Making a plain network deeper first saturates and then lowers accuracy, even on the training data, so it is not over-fitting. A deeper net could copy a shallower one with identity layers, yet the solver does not find that.

#flashcards/research/concept #flashcards/topic/research-vision Write the residual reformulation of ResNet and explain each symbol.
?
$$F(x):=H(x)-x,\qquad\text{so}\qquad H(x)=F(x)+x$$
- $H(x)$: the mapping the stacked layers should produce; $x$: their input.
- $F(x)$: what the layers actually learn, the residual.
- If the identity is best, the layers only need to push $F$ to zero, which the authors hypothesise is easier.

#flashcards/research/concept #flashcards/topic/research-vision If the wanted mapping is $H(x)=x$, what must a residual block learn?
?
$F(x)=H(x)-x=0$, so every weight can go to zero. Doing nothing is the easy case for a residual block, but hard for a stack of plain nonlinear layers (the authors' hypothesis).

#flashcards/research/concept #flashcards/topic/research-vision Write the ResNet building block and explain each symbol.
?
$$y=F(x,\{W_i\})+x,\qquad F=W_2\,\sigma(W_1x)$$
- $x,y$: input and output; $F$: the residual with weights $W_1,W_2$ (biases left out); $\sigma$: ReLU, applied again after the addition.
- The shortcut "$+x$" adds no parameters and almost no cost, so plain and residual nets compare fairly.

#flashcards/research/concept #flashcards/topic/research-vision Block example: $x=(1,2)$, $W_1$ rows $(1,-1)$ and $(0,1)$, $W_2$ rows $(1,0)$ and $(1,1)$. What is $y$?
?
$W_1x=(-1,2)$, ReLU gives $(0,2)$, $F=(0,2)$, $y=F+x=(1,4)$. The block kept the input and added a small correction.

#flashcards/research/concept #flashcards/topic/research-vision Write the projection shortcut and say when it is used.
?
$$y=F(x,\{W_i\})+W_s\,x$$
- $W_s$: a learned matrix (a 1×1 convolution) that changes $x$ to the size of $F$.
- Used only when sizes differ; the identity shortcut is enough to fix degradation and is cheaper.

#flashcards/research/concept #flashcards/topic/research-vision Projection example: $x=(1,2)$, $W_s$ rows $(1,0),(0,1),(1,1)$, $F=(0.5,-0.5,1)$. What is $y$?
?
$W_sx=(1,2,3)$, so $y=(1.5,1.5,4)$. Cost: a $256\times256$ projection adds 65,536 weights to a 69,632-weight bottleneck, about 1.9 times as many (my arithmetic), which is why the paper keeps identity shortcuts in bottleneck blocks.

#flashcards/research/concept #flashcards/topic/research-vision ![[residual-block.png]]
What does this picture show?
?
A residual block: the input goes through two layers giving $F(x)$ while a shortcut copies $x$ around them; the two are added to give $x+F(x)$. Image by LunarLullaby (Wikimedia Commons, CC BY-SA 4.0), the idea of the paper's Figure 2.

#flashcards/research/concept #flashcards/topic/research-vision ![[resnet-basic-vs-bottleneck.svg]]
What does this picture show?
?
Two residual block kinds: (a) basic, two 3×3 convolutions; (b) bottleneck, 1×1 down to $D/4$, 3×3, 1×1 back up to $D$; both add the input $x$ back at the end. Image by Cosmia Nebula (Wikimedia Commons, CC0), matching the paper's Figure 5.

#flashcards/research/concept #flashcards/topic/research-vision What is the paper's evidence that residual learning fixes degradation? (Table 2)
?
Top-1 error on ImageNet validation: plain 18 layers 27.94, plain 34 layers 28.54 (worse); ResNet 18 layers 27.88, ResNet 34 layers 25.03 (better). The 34-layer ResNet beats its plain twin by 3.51 points (my arithmetic), with no extra parameters.

#flashcards/research/concept #flashcards/topic/research-vision What are the five ResNet depths and their blocks (paper's Table 1)?
?
18 and 34 layers use basic blocks (2,2,2,2 and 3,4,6,3 per stage); 50, 101 and 152 use bottleneck blocks (3,4,6,3; 3,4,23,3; 3,8,36,3). FLOPs: 1.8, 3.6, 3.8, 7.6 and 11.3 billion; VGG-19 has 19.6 billion.

#flashcards/research/concept #flashcards/topic/research-vision What is a bottleneck block, and why use it?
?
Three layers, 1×1 (shrink), 3×3, 1×1 (restore). The 3×3 layer works on fewer channels, which keeps cost low for the deeper nets. Example (my arithmetic): 256-64-256 has 69,632 weights, similar to two 3×3 layers at 64 channels (73,728), and far less than two 3×3 layers at 256 channels (1,179,648).

#flashcards/research/concept #flashcards/topic/research-vision How did the ResNet depths do on ImageNet validation (Table 3, top-5 error)?
?
ResNet-34 B 7.46, ResNet-50 6.71, ResNet-101 6.05, ResNet-152 5.71; for comparison VGG-16 9.33 (their test) and plain-34 10.02. Deeper residual nets kept improving, with no degradation.

#flashcards/research/concept #flashcards/topic/research-vision How good was the winning ResNet ensemble (Table 5), and what was it made of?
?
Top-5 error 3.57% on the ImageNet test set, against 6.66 for GoogLeNet and 7.32 for VGG (ILSVRC 2014). It combined six models of different depth. A single 152-layer model scored 4.49% on the validation set, better than all earlier ensembles.

#flashcards/research/concept #flashcards/topic/research-vision What happened with the 1202-layer network on CIFAR-10 (Table 6)?
?
It trained fine (training error below 0.1%) but its test error, 7.93%, was worse than the 110-layer net's 6.43% (mean 6.61 over 5 runs). The authors blame over-fitting from 19.4 million parameters on a small data set.

#flashcards/research/concept #flashcards/topic/research-vision How much did ResNet-101 improve Faster R-CNN over VGG-16 (Tables 7 and 8)?
?
Same detector, only the backbone changed. COCO mAP@[.5,.95] 21.2 to 27.2, a gain of 6.0 points or 28.3% relative (my arithmetic); VOC 2007 mAP 73.2 to 76.4.

#flashcards/research/concept #flashcards/topic/research-vision What did the paper's Figure 7 show about layer responses?
?
ResNets have generally smaller layer responses than plain nets, and deeper ResNets smaller still. This supports the idea that residual functions are close to zero, so each layer changes the signal only a little. I read this in the text, not from the figure.

#flashcards/research/concept #flashcards/topic/research-vision In ResNet, why define the layers to learn $F(x)=H(x)-x$ (subtract the input)?
?
So the layers only learn the leftover difference from their input. If the best thing for a block to do is nothing ($H(x)=x$), the target is $F(x)=0$, which a block reaches by pushing its weights towards zero. A stack of plain nonlinear layers has to build the identity exactly, and the paper's experiments suggest solvers find that hard (the degradation problem). The authors call "it is easier to optimise the residual" a hypothesis, supported by small layer responses (their Figure 7).

#flashcards/research/concept #flashcards/topic/research-vision In $y=F(x,\{W_i\})+x$, why is the shortcut a plain "$+x$" with no weights?
?
- **Cost:** it adds no parameters and almost no computation, so a plain network and its residual twin have the same parameters, depth, width and cost, and can be compared fairly (paper).
- **Why it helps training (general knowledge; my illustration):** the slope of $y$ with respect to $x$ is $1+\text{slope of }F$. Even if $F$'s slope is only 0.1, each block passes on about 1.1 of the gradient. Over 3 blocks that is $1.1^3=1.33$, while a plain chain gives $0.1^3=0.001$. A plain "$+x$" gives the gradient a straight road back to early layers.

#flashcards/research/concept #flashcards/topic/research-vision Why must $x$ and $F$ have the same size in ResNet, and what is $W_s$ for?
?
The shortcut is an element-wise addition (channel by channel for feature maps), so the two vectors must have the same length (paper). When a block changes the number of channels the sizes no longer match, and a learned matrix $W_s$ (a 1×1 convolution) stretches or shrinks $x$ to fit. My example: $x=(1,2)$ and $F=(0.5,-0.5,1)$ cannot be added until $W_s x=(1,2,3)$, giving $y=(1.5,1.5,4)$.

#flashcards/research/concept #flashcards/topic/research-vision Why not use a projection $W_s$ on every shortcut in ResNet?
?
The paper compares three options (Table 3, top-1 error): A (zero-padding, no extra parameters) 25.03%, B (projections only when sizes change) 24.52%, C (projections everywhere) 24.19%. The differences are small, so projections are "not essential" for fixing degradation. And in the bottleneck block a projection is costly: my arithmetic, a $256\times256$ projection adds 65,536 weights to a block of 69,632, about 1.9 times as many; the paper says it doubles the time and size. So identity shortcuts are used wherever sizes match.

#flashcards/research/concept #flashcards/topic/research-vision Why does the residual function $F$ in ResNet have two or three layers and not one?
?
The paper says that if $F$ has only one layer, $y=W_1x+x$ is similar to a linear layer, "for which we have not observed advantages". With two layers $F=W_2\sigma(W_1x)$ there is a ReLU between them, so $F$ is truly nonlinear. The experiments use two layers (basic block) or three (bottleneck block), and the paper notes more layers are possible.
