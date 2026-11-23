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
Let the stacked layers learn the **difference** from the input, not the whole answer: the shortcut ([[Residual Connection]]). If the best thing for a block to do is nothing, it only has to push the difference to zero. (Think of editing a draft instead of writing from a blank page; the analogy is mine.)

## The architecture (figures)
The ResNets in this paper are built in five steps (paper, 3.3 and Table 1): a **stem** (one 7×7 convolution with stride 2, then a 3×3 max pool), then **four stages** of residual blocks working on maps 56, 28, 14 and 7 pixels wide, then **global average pooling**, and a **1000-way fully connected layer with softmax**. The five networks (18, 34, 50, 101 and 152 weighted layers) differ only in the kind of block and how many blocks each stage has. Marks: "(paper, section)" was read in the paper; "(general knowledge)" is my own background; "(mine)" is my arithmetic.

### Part by part
- **The plain network it starts from (the baseline).**
  - *What it does:* the plain 34-layer network follows "the philosophy of VGG nets": mostly 3×3 convolutions, with two rules: (i) the same number of filters for the same output size, and (ii) when the map size is halved, the number of filters is doubled (paper, 3.3). The residual network is this same network plus shortcuts.
  - *Why it matters:* rule (ii) is there "so as to preserve the time complexity per layer" (paper). My arithmetic confirms it: one basic block (two 3×3 convolutions) costs $231{,}211{,}008$ multiply-adds at every stage: 64 channels on 56×56, 128 on 28×28, 256 on 14×14 and 512 on 7×7. Halving the width cuts the pixels by 4, and doubling the channels multiplies the weights by 4, so the two cancel. Starting from the plain network also makes the test fair: the plain and residual versions have the same parameters, depth and cost. This network has fewer filters and lower cost than VGG: 3.6 billion FLOPs, 18% of VGG-19's 19.6 billion (paper).
- **The stem: 7×7 convolution with stride 2, then a 3×3 max pool.**
  - *What it does:* the first layer has 64 filters of 7×7 ($9{,}408$ weights, mine) and moves 2 pixels at a time, turning the 224×224 image into 112×112; the max pool halves it again to 56×56 (paper, Table 1).
  - *Why it matters:* it shrinks the big image quickly, so every later layer works on a small map and is cheap (general knowledge; the paper does not discuss the stem).
- **Down-sampling by stride 2 inside the stages.**
  - *What it does:* the first block of the stages conv3, conv4 and conv5 uses a stride of 2, which halves the map; there is no separate pooling layer between stages (paper, Table 1 caption and 3.3).
  - *Why it matters:* the shrinking is done by a learned convolution, and it is where the number of channels doubles. That is also where the shortcut must change size (see the next part).
- **The shortcut, and when it changes size.**
  - *What it does:* when input and output have the same size the shortcut is an identity (the solid lines in the paper's Figure 3). When the size goes up (the dotted lines) there are options: **A** pad with zeros (no extra parameters), **B** a 1×1 convolution to match (a projection), **C** a projection on every shortcut (paper, 3.3 and 4.1).
  - *Why it matters:* the identity adds no parameters and no cost. Table 3 shows A, B and C differ by under one point (top-1 25.03%, 24.52%, 24.19% for ResNet-34), so the projections are not essential; the paper uses B for the deeper nets and not C "to reduce memory/time complexity and model sizes".
- **The residual block** (the heart of the paper): explained step by step in the next section.
- **The bottleneck block** (ResNet-50, 101, 152): explained step by step in the next section.
- **Batch normalization after every convolution, before the activation.**
  - *What it does:* each convolution is followed by [[Batch Normalization]], then the nonlinearity (paper, 3.4). No dropout is used.
  - *Why it matters:* it is why the authors can say the degradation is not a vanishing-gradient problem: with normalization the forward signals have non-zero variance, and they checked that the backward gradients have healthy norms (paper, 4.1).
- **Global average pooling, then a 1000-way fully connected layer and softmax.**
  - *What it does:* each of the final 7×7 maps is averaged into one number, giving one vector, and a single fully connected layer turns it into one score per ImageNet class (1,000 of them); the softmax makes them probabilities (paper, 3.3).
  - *Why it matters:* it replaces the large fully connected layers at the end of VGG, which keeps the network small. My arithmetic: ResNet-34's final layer has $512\times1000=512{,}000$ weights; VGG-16's first fully connected layer alone has $7\cdot7\cdot512\cdot4096=102{,}760{,}448$ (general knowledge for the VGG shapes; the paper's Figure 3 shows "fc 4096").
- **How deep: why 18, 34, 50, 101 and 152?**
  - *What it does:* five sizes are tested (Table 1). 18 and 34 use basic blocks; 50, 101 and 152 use bottleneck blocks.
  - *Why it matters:* the paper does not explain why these particular depths; they are a ladder to show that deeper keeps helping. The evidence is Table 3: top-1 error falls at each step (34: 25.03 to 24.19 depending on the shortcut option, 50: 22.85, 101: 21.75, 152: 21.43), and on CIFAR-10 a 1202-layer net trains but over-fits (7.93% against 6.43% for 110 layers).

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

Check (mine): $1+2\cdot(3+4+6+3)+1=34$ weighted layers for the basic net and $1+3\cdot(3+8+36+3)+1=152$ for the deepest; the first and last "1" are the first convolution and the final fully connected layer.

**Summary: what the table shows**
- **Cost grows with depth, but slowly with the bottleneck:** from 18 to 152 layers (8.4 times deeper) the FLOPs rise from 1.8 to 11.3 billion (6.3 times, my arithmetic).
- **A telling pair:** the 34-layer net (3.6 billion) and the 50-layer net (3.8 billion) cost almost the same, although the 50-layer one has 16 more layers, because of the bottleneck blocks.
- **Still cheaper than VGG:** even ResNet-152 (11.3 billion) is below VGG-16 (15.3 billion) and VGG-19 (19.6 billion).

### Size check (my arithmetic, random weights, torchvision's ResNets)
Parameter counts: ResNet-18 $11.69$M, ResNet-34 $21.80$M, ResNet-50 $25.56$M, ResNet-101 $44.55$M, ResNet-152 $60.19$M. The paper gives FLOPs, not parameter counts, so these are my counts with torchvision 0.23 (its version places the stride slightly differently from the paper's, which changes FLOPs a little but not the parameter count, as far as I know). For comparison, my earlier count of VGG-16 is $138.36$M (the [[MobileNet]] note), so ResNet-152 has about 2.3 times fewer parameters than VGG-16 while being 8 times deeper.

## The core mechanism, step by step

### The residual block
**The idea (my analogy).** Writing an essay from a blank page is hard; editing a draft is easier. A plain block must write the whole output from scratch. A residual block is handed the draft (its input $x$) and only has to write the *corrections*, which are added on top. If the draft is already fine, the corrections are zero.

![[residual-block.png]]
*A residual block. The input $x$ goes down through two layers, which together compute $F(x)$. At the same time, $x$ is copied along the curved "identity" path on the right, which does nothing to it, and the two are added (the circled plus). The output is $x+F(x)$. Image: LunarLullaby, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:ResBlock.png), CC BY-SA 4.0. This is the idea of the paper's Figure 2, drawn by someone else, not the paper's own picture.*

1. **Copy the input two ways.**
   - *What it does:* the input $x$ goes into the weight layers, and an exact copy travels along the shortcut (paper, Figure 2 and 3.2).
   - *Why it matters:* the shortcut has no weights, so it "adds neither extra parameter nor computational complexity" (paper). It also gives the signal a direct road past the layers (general knowledge).
2. **First weight layer, then a ReLU.**
   - *What it does:* in the paper's two-layer example this is $\sigma(W_1x)$, where $\sigma$ is the ReLU; in the real network it is a 3×3 convolution, then batch normalization, then ReLU (paper, 3.2 and 3.4).
   - *Why it matters:* the ReLU makes the correction $F$ truly nonlinear.
3. **Second weight layer.**
   - *What it does:* $W_2$ is applied to that result, with no activation yet. So $F=W_2\,\sigma(W_1x)$ (paper, 3.2).
   - *Why it matters:* with a single layer, $y=W_1x+x$ is just a linear layer, and the paper "has not observed advantages" from that. Two layers (or three in the bottleneck) is the minimum for a useful $F$.
4. **Add the copy: $y=F+x$.**
   - *What it does:* an element-wise addition (channel by channel for feature maps). The two must have the same size, otherwise a projection $W_s$ matches them (paper, 3.2).
   - *Why it matters:* this is the whole trick. If the best thing to do is nothing, the block only needs $F=0$.
5. **A second ReLU after the addition: $\sigma(y)$.**
   - *What it does:* the paper adopts "the second nonlinearity after the addition" (3.2).
   - *Why it matters:* the block's output is again passed through a nonlinearity before the next block. The paper gives no further reason.

**Worked example (mine; two numbers, as in the equations section).** $x=(1,2)$, $W_1=\begin{pmatrix}1&-1\\0&1\end{pmatrix}$, $W_2=\begin{pmatrix}1&0\\1&1\end{pmatrix}$.
- Step 2: $W_1x=(-1,2)$, then ReLU gives $(0,2)$.
- Step 3: $F=W_2(0,2)=(0,2)$.
- Step 4: $y=F+x=(1,4)$. Step 5: ReLU leaves $(1,4)$.
- If the weights were all zero, $F=(0,0)$ and $y=x=(1,2)$: the block would pass its input through unchanged.

**Shapes through the first two stages of ResNet-34 (mine, from the paper's Table 1 layout).**

| Where | Input | After the block | How the shortcut works |
|---|---|---|---|
| Stage conv2_x (3 blocks) | $56\times56\times64$ | $56\times56\times64$ | identity (solid line) |
| First block of conv3_x | $56\times56\times64$ | $28\times28\times128$ | dotted: stride 2, and A zero-pads or B projects to 128 channels |
| Other blocks of conv3_x (3 more) | $28\times28\times128$ | $28\times28\times128$ | identity |

### The bottleneck block
**The idea.** A basic block with 256-channel 3×3 layers would be very expensive. The bottleneck squeezes the channels down, does the 3×3 work cheaply on the thin version, then widens again, so the shortcut still carries the wide 256-channel signal.

![[resnet-basic-vs-bottleneck.svg]]
*Two kinds of residual block: (a) a basic block with two 3×3 convolutions, (b) a bottleneck block with 1×1, 3×3, 1×1 convolutions, where the 3×3 works at a quarter of the width $D$. The input $x$ runs up the right side and is added back by the circled plus. Image: Cosmia Nebula, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:ResBlockVariants-stream.svg), CC0. This matches the two blocks of the paper's Figure 5 (basic block on the left, bottleneck on the right). The paper's version shows widths 64 and 256 and has a ReLU between layers, which this drawing leaves out.*

1. **1×1 convolution: shrink the channels (256 to 64).**
   - *What it does:* mixes the channels at each pixel and reduces their number. Weights (mine): $256\times64=16{,}384$.
   - *Why it matters:* it makes the next, expensive 3×3 layer work on far fewer channels (paper, 4.1: "the 1×1 layers are responsible for reducing and then increasing (restoring) dimensions").
2. **3×3 convolution at the thin width (64 to 64).**
   - *What it does:* looks at each pixel's neighbours, as in the basic block. Weights (mine): $9\times64\times64=36{,}864$.
   - *Why it matters:* this "bottleneck" is where the spatial work happens, and it is cheap because the width is small.
3. **1×1 convolution: restore the channels (64 to 256).**
   - *What it does:* widens back to the size the shortcut carries. Weights (mine): $64\times256=16{,}384$.
   - *Why it matters:* the output must match the 256-channel input so the identity shortcut can add them.
4. **Add the identity shortcut.**
   - *What it does:* the 256-channel input is added as before.
   - *Why it matters:* here the parameter-free shortcut is "particularly important": replacing it with a projection would, the paper says, double the time complexity and model size, because the shortcut then connects the two wide ends.

**Worked comparison (mine).** The bottleneck block has $16{,}384+36{,}864+16{,}384=69{,}632$ weights. Two 3×3 layers at 64 channels (a basic block) have $73{,}728$, and two 3×3 layers at 256 channels would have $1{,}179{,}648$, about 17 times more than the bottleneck. So the bottleneck handles 256-channel signals for about the price of a 64-channel basic block, which is what the paper means by "similar time complexity" for the two designs of its Figure 5. A $256\times256$ projection shortcut would add $65{,}536$, nearly doubling it ($135{,}168$, 1.94 times).

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

## Questions I had, answered

**Is the degradation problem just over-fitting?** No. Over-fitting means low error on the training data and high error on new data. Here the deeper plain net has **higher training error** too (paper, introduction). On CIFAR-10 the 56-layer plain net did worse than the 20-layer one on both the training and the test set (paper, Figure 1; I read the caption and text, not the curves).

**Then is it the vanishing-gradient problem?** The authors argue it is unlikely. The plain nets use batch normalization, which "ensures forward propagated signals to have non-zero variances", and they verified that the backward gradients "exhibit healthy norms", so "neither forward nor backward signals vanish". The 34-layer plain net still reaches competitive accuracy, so the solver works "to some extent". They conjecture that deep plain nets "may have exponentially low convergence rates" and say the reason will be studied in the future (paper, 4.1). So the cause is not settled.

**Why not just train the deep plain net for longer?** The paper tried: with 3 times more iterations it "still observed the degradation problem" (paper, footnote 3 in 4.1).

**A deeper net could copy the shallower one and set the extra layers to "do nothing". Why doesn't training find that?** In principle such a solution exists "by construction", so the deeper net should do no worse. But the experiments show the solvers "are unable to find solutions that are comparably good or better than the constructed solution (or unable to do so in feasible time)" (paper, introduction). The residual form makes "do nothing" the easy case (all weights near zero), which is the whole fix.

**How is a shortcut different from Highway networks?** Highway networks also use shortcuts, but with gates that are data-dependent and have parameters. When a gate is "closed" the layers represent non-residual functions. ResNet's identity shortcuts are parameter-free and are "never closed". The paper adds that highway networks had not shown accuracy gains with extremely increased depth (over 100 layers) (paper, 2).

**What do the solid and dotted shortcut lines mean in the paper's Figure 3?** Solid: the identity shortcut, used when input and output have the same size. Dotted: the size increases (more channels, half the width), so the shortcut needs option A (zero padding) or B (a 1×1 projection), both with a stride of 2 (paper, 3.3).

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
Body read as text: pages 1 to 8 (introduction, related work, method, experiments, Tables 1 to 8) of the arXiv PDF (v1). I did not view the pictures: Figure 1 (CIFAR-10 curves), Figure 3 (the 34-layer networks), Figure 4 (ImageNet curves), Figure 6 and Figure 7 were read only through their captions and the text. Not read: the reference list and the appendix (detection and localisation details, about pages 10 to 12). The equations and numbers marked "paper" come from the body. The "my arithmetic" numbers are from my own Python run (NumPy). The licences of the two figures were read on their Commons pages. The parameter counts come from my own run of torchvision's ResNets with random weights (version 0.23.0, Python 3.9.6); the VGG-16 count is from my earlier [[MobileNet]] run; the stage costs and the block weight counts are my own arithmetic. Not opened in Obsidian, so the flashcards are unreviewed.

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

#flashcards/research/concept #flashcards/topic/research-vision What are the two design rules of the plain network that ResNet starts from, and why rule (ii)?
?
(i) The same number of filters for the same output map size. (ii) When the map size is halved, the number of filters is doubled, "so as to preserve the time complexity per layer" (paper). My arithmetic: a basic block (two 3×3 convolutions) costs 231,211,008 multiply-adds at every stage (64 channels on 56×56, 128 on 28×28, 256 on 14×14, 512 on 7×7), because halving the width cuts pixels by 4 while doubling the channels multiplies weights by 4.

#flashcards/research/concept #flashcards/topic/research-vision What is the stem of a ResNet and what does it do?::One 7×7 convolution with 64 filters and stride 2 (9,408 weights, my count), turning the 224×224 image into 112×112, then a 3×3 max pool with stride 2 giving 56×56. It shrinks the big image quickly so later layers are cheap (the reason is general knowledge; the paper does not discuss it).

#flashcards/research/concept #flashcards/topic/research-vision How does a ResNet shrink the map between stages?::The first block of conv3, conv4 and conv5 uses a stride of 2 (paper, Table 1), so a learned convolution halves the map and the channels double there; there is no separate pooling between stages. That is where the shortcut must change size.

#flashcards/research/concept #flashcards/topic/research-vision Why does ResNet put batch normalization after every convolution, and what did the authors conclude from it?::Normalization comes right after each convolution and before the activation (no dropout). With it, forward signals have non-zero variance and the authors checked that backward gradients have healthy norms, so they argue the degradation is not caused by vanishing gradients (paper, 3.4 and 4.1).

#flashcards/research/concept #flashcards/topic/research-vision What do the global average pooling and the last layer of a ResNet do, and why does that keep it small?::Each final 7×7 map is averaged to one number, and one fully connected layer gives 1,000 class scores, then a softmax. It replaces VGG's large fully connected layers. My arithmetic: ResNet-34's last layer has 512×1,000 = 512,000 weights; VGG-16's first fully connected layer alone has 7·7·512·4096 = 102,760,448 (VGG shapes from general knowledge).

#flashcards/research/concept #flashcards/topic/research-vision Why does the paper test depths 18, 34, 50, 101 and 152?::It gives no reason for those exact numbers. They are a ladder to show deeper keeps helping: top-1 error falls from 25.03 and 24.19 (34 layers, by shortcut option) to 22.85, 21.75 and 21.43 (50, 101, 152). On CIFAR-10 a 1202-layer net over-fits (7.93% against 6.43% for 110 layers).

#flashcards/research/concept #flashcards/topic/research-vision How many parameters do the five ResNets have, and how does that compare with VGG-16?::My counts (torchvision, random weights): ResNet-18 11.69M, 34 21.80M, 50 25.56M, 101 44.55M, 152 60.19M. VGG-16 is 138.36M in my earlier count, so ResNet-152 has about 2.3 times fewer parameters while being 8 times deeper. The paper gives FLOPs, not parameter counts.

#flashcards/research/concept #flashcards/topic/research-vision What are the five steps inside a residual block, and what does each do?
?
1. Copy the input two ways: into the weight layers and along the shortcut (no weights, no extra cost). 2. First weight layer, then a ReLU (makes $F$ nonlinear). 3. Second weight layer, no activation yet, so $F=W_2\sigma(W_1x)$. 4. Add the copy, $y=F+x$ (sizes must match, otherwise a projection $W_s$). 5. A second ReLU after the addition. If all the weights were zero, $y=x$.

#flashcards/research/concept #flashcards/topic/research-vision What are the shapes through the first blocks of ResNet-34 stages conv2_x and conv3_x?::conv2_x: $56\times56\times64$ in and out, identity shortcut. First block of conv3_x: $56\times56\times64$ in, $28\times28\times128$ out, dotted shortcut (stride 2; A zero-pads or B projects to 128 channels). The other blocks of conv3_x stay $28\times28\times128$ with identity shortcuts (my table from the paper's Table 1).

#flashcards/research/concept #flashcards/topic/research-vision What are the steps of a bottleneck block, with their weight counts for 256 channels at the ends?::1×1 convolution shrinks 256 to 64 channels (16,384 weights). 3×3 convolution at 64 channels (36,864). 1×1 convolution restores 64 to 256 (16,384). Then the identity shortcut adds the 256-channel input. Total 69,632 weights (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-vision How does the bottleneck block's cost compare with basic blocks?::69,632 weights, about the same as a basic block at 64 channels (73,728) and about 17 times fewer than two 3×3 layers at 256 channels (1,179,648). The paper calls the two designs of its Figure 5 "similar time complexity". A 256×256 projection shortcut would add 65,536 weights (1.94 times in all), which is why identity shortcuts matter most here (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-vision Is the degradation problem just over-fitting?::No. The deeper plain net has higher error on the training data too, and over-fitting would show low training error. On CIFAR-10 the 56-layer plain net was worse than the 20-layer one on both training and test data (paper, introduction; caption and text read).

#flashcards/research/concept #flashcards/topic/research-vision Did the authors say degradation is the vanishing-gradient problem?::They argue it is unlikely: with batch normalization the forward signals have non-zero variance and the backward gradients have healthy norms, and the 34-layer plain net still reaches competitive accuracy. They conjecture exponentially low convergence rates and say the reason will be studied later, so the cause is not settled (paper, 4.1).

#flashcards/research/concept #flashcards/topic/research-vision Why not just train a deep plain network for longer?::The authors tried 3 times more iterations and still observed the degradation problem (paper, footnote in 4.1).

#flashcards/research/concept #flashcards/topic/research-vision A deeper net could copy a shallower one and make the extra layers identity. Why does training not find that?::Such a solution exists by construction, but experiments show the solvers are unable to find solutions as good or better in feasible time (paper, introduction). The residual form makes "do nothing" the easy case (weights near zero), which is the fix.

#flashcards/research/concept #flashcards/topic/research-vision How do ResNet shortcuts differ from Highway networks?::Highway shortcuts are gated: the gates depend on the data and have parameters, and a closed gate makes the layers non-residual. ResNet's identity shortcuts are parameter-free and never closed. The paper adds that Highway networks had not shown accuracy gains beyond 100 layers.

#flashcards/research/concept #flashcards/topic/research-vision What do the solid and dotted shortcut lines mean in the paper's Figure 3?::Solid: an identity shortcut, used when input and output have the same size. Dotted: the size increases (more channels and half the width), so the shortcut uses option A (zero padding) or B (a 1×1 projection), both with stride 2.
