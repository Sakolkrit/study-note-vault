---
tags: [paper, vision, CNN, ImageNet, inception, image-classification, detection]
link: https://arxiv.org/abs/1409.4842
authors: Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich (Google, UNC Chapel Hill, Univ. of Michigan; from the paper's first page)
year: 2014
status: read
field: image classification
---

# GoogLeNet 2014

> **In one line for a 12-year-old:** Instead of choosing one filter size, let the network look with several sizes at once, shrink the data before the expensive looks, and keep the total work the same.

*Full title: "Going deeper with convolutions" (arXiv 1409.4842, one version, 17 Sep 2014; date from the arXiv listing). The architecture is called **Inception**; **GoogLeNet** is the 22-layer version submitted to ILSVRC 2014. Layout follows [[Attention Is All You Need 2017]] and [[Faster R-CNN 2015]]. Marks: "(paper, section)", "(general knowledge)", "(mine)".*

## The problem
By 2014 the way to better image recognition was to make networks bigger. The paper names two costs of "just make it bigger" (paper, 3):
1. **More parameters mean more overfitting,** especially when "the creation of high quality training sets can be tricky and expensive" (the paper's Figure 1 shows two ImageNet dog classes, Siberian husky and Eskimo dog, that even experts find hard to separate).
2. **More compute:** "if two convolutional layers are chained, any uniform increase in the number of their filters results in a quadratic increase of computation", and if the added capacity is used badly (weights near zero) "a lot of computation is wasted".

The paper wants quality gains from a better *shape* of network at a fixed budget: "improved utilization of the computing resources inside the network" (abstract). It also cares about use on phones and small devices: most experiments keep "a computational budget of 1.5 billion multiply-adds at inference time" (paper, 1).

## What people did before
- **The standard CNN stack** since LeNet-5: convolution layers (optionally with contrast normalisation and max pooling) then fully connected layers. For ImageNet, the trend was "to increase the number of layers and layer size, while using dropout to address the problem of overfitting" (paper, 2; [[AlexNet 2012]], [[VGG 2014]], [[Dropout]]).
- **Network in Network** (Lin et al.): extra 1 × 1 convolutions followed by ReLU to make each layer more expressive. The paper "use[s] this approach heavily", but for a second purpose, dimension reduction (paper, 2). The name Inception comes from this paper plus the "we need to go deeper" internet meme (paper, 2).
- **Fixed multi-scale filters** (Serre et al.): a series of fixed Gabor filters of different sizes, in a 2-layer model. Inception's filters "are learned" and the module is repeated many times (paper, 2).
- **Sparse networks in theory** (Arora et al.): if the data's distribution can be represented by a large, very sparse deep network, the best topology can be built layer by layer by clustering neurons with highly correlated outputs. The paper links this to the Hebbian principle, "neurons that fire together, wire together", and admits "the strict mathematical proof requires very strong conditions" (paper, 3).
- **For detection:** [[R-CNN 2013]], "the current leading approach" (paper, 2).

## The idea
Real sparse networks would be efficient, but "computing infrastructures are very inefficient when it comes to numerical calculation on non-uniform sparse data structures": even if the arithmetic is cut 100 times, "the overhead of lookups and cache misses is so dominant that switching to sparse matrices would not pay off" (paper, 3). So approximate the sparse structure with **dense building blocks that hardware runs fast**. In each stage, run several convolutions of different sizes (1 × 1, 3 × 3, 5 × 5) and a pooling branch **in parallel on the same input**, then stack their outputs side by side. To stop the cost from exploding, put a **1 × 1 convolution before the 3 × 3 and 5 × 5** to cut the number of channels first. Repeat this block, the **Inception module**, nine times (paper, 4 and 5). The authors are frank that the theory was only a starting point: "it is still questionable whether its quality can be attributed to the guiding principles that have lead to its construction" (paper, 3).

## The architecture, part by part

The paper's Figure 2 (the naïve and the dimension-reduced Inception module) and Figure 3 (the whole network with auxiliary classifiers) are under arXiv's non-exclusive licence, which is not an open one, so they are not copied. The module figure below is a free redrawing from Wikimedia Commons (rules in the vault guide).

The whole design (paper, 5, Table 1). A stem of ordinary convolutions and max pools (7 × 7 convolution with stride 2, max pool, a 1 × 1 and a 3 × 3 convolution, max pool) turns the 224 × 224 colour input into a 28 × 28 × 192 map. Then come nine Inception modules in three groups (3a, 3b; 4a to 4e; 5a, 5b) with stride-2 max pooling between groups, an average pool over the last 7 × 7 grid, dropout (40%), one linear layer with 1,000 outputs and a softmax. It is "22 layers deep when counting only layers with parameters (or 27 layers if we also count pooling)", and "about 100" independent building blocks overall (paper, 5). All convolutions, including those inside the modules, use [[ReLU]]. The receptive field is 224 × 224 (paper, 5; see [[Receptive Field]]).

### The Inception module
![[inception-module.png]]
*The Inception module with dimension reductions: four parallel branches (1 × 1 convolution; 1 × 1 then 3 × 3; 1 × 1 then 5 × 5; 3 × 3 max pooling then 1 × 1) from the previous layer into a "filter concatenation". Author: Krishnachandranvn, CC BY-SA 4.0, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:InceptionModule.png). It is a redrawing of the paper's Figure 2(b), not the paper's own. I compared it by eye with the paper's figure, rendered from the PDF: the four branches and the arrows match; the differences are only colours and the capital "X" in its labels.*

- **Branch 1: 1 × 1 convolution**
  - *What it does:* looks at one position across all channels (cheap, no spatial neighbourhood).
  - *Why it matters:* the paper's argument is that in lower layers correlated units "would concentrate in local regions" and "can be covered by a layer of 1×1 convolutions" (paper, 4).
- **Branch 2: 1 × 1 reduce, then 3 × 3**
  - *What it does:* the 1 × 1 convolution first shrinks the number of channels ("#3×3 reduce" in Table 1), then the 3 × 3 looks at a small neighbourhood.
  - *Why it matters:* a few more spread-out clusters can be covered by larger patches; the reduction keeps the 3 × 3 affordable.
- **Branch 3: 1 × 1 reduce, then 5 × 5**
  - *What it does:* same, with a 5 × 5 patch.
  - *Why it matters:* "even a modest number of 5×5 convolutions can be prohibitively expensive on top of a convolutional layer with a large number of filters" (paper, 4). The reduction is where the saving is largest (my count below). The sizes 1, 3, 5 were chosen "to avoid patch-alignment issues", a decision "based more on convenience rather than necessity" (paper, 4).
- **Branch 4: 3 × 3 max pooling, then 1 × 1 projection ("pool proj")**
  - *What it does:* pools the input, then a 1 × 1 convolution reduces its channels.
  - *Why it matters:* "pooling operations have been essential for the success in current state of the art convolutional networks", so a pooling path should help. But a pooling branch outputs as many channels as its input, which would make the module's output grow stage after stage; the projection stops that (paper, 4).
- **Filter concatenation**
  - *What it does:* stacks the four outputs along the channel axis (same height and width).
  - *Why it matters:* the next stage sees features from several scales together: "visual information should be processed at various scales and then aggregated so that the next stage can abstract features from different scales simultaneously" (paper, 4).
- **Why the 1 × 1 convolutions are "dual-purpose"** (paper, 2 and 4): they cut channels, and, since each is followed by ReLU, they also add non-linearity.
- **Where the modules start:** at the higher layers only, "for technical reasons (memory efficiency during training)", not out of necessity (paper, 4).
- **The ratio of 3 × 3 and 5 × 5** should grow in higher layers, as features get more abstract and less spatially concentrated (paper, 4); in Table 1, the 5 × 5 share is small in every module.

### Other parts of the network (paper, 5)
- **Average pooling instead of fully connected layers** before the classifier: "a move from fully connected layers to average pooling improved the top-1 accuracy by about 0.6%", but dropout "remained essential". An extra linear layer is kept so that the network "can be adapted and fine-tuned for other label sets easily" (paper, 5; compare [[Transfer Learning]]).
- **Two auxiliary classifiers** on top of the output of module 4a and module 4d: each is a 5 × 5 average pool with stride 3 (to 4 × 4 × 512 for 4a and 4 × 4 × 528 for 4d), a 1 × 1 convolution with 128 filters, a fully connected layer of 1,024 units, a dropout layer that drops 70% of outputs, and a linear 1,000-way softmax. They are used only in training and discarded at inference (paper, 5).
  - *Why they exist:* the worry was "the ability to propagate gradients back through all the layers in an effective manner" in a deep network. The authors reason that mid-level features should already be discriminative, so side classifiers would "encourage discrimination in the lower stages", "increase the gradient signal" and "provide additional regularization" (paper, 5). The paper gives no ablation of them, so how much they help is not shown (my reading of the text I read).

### Table 1 and a size check (mine)
Computed in `googlenet_demo.py` (Python 3.9.6, PyTorch 2.8.0, torchvision 0.23.0, CPU) from the channel counts in the paper's Table 1. The paper's table lists, per module, the number of 1 × 1, "3×3 reduce", 3 × 3, "5×5 reduce", 5 × 5 and "pool proj" filters, the output size, parameters and operations. Biases are left out of "my params"; "mult-adds" are weights times grid cells.

| Module | Grid | Output channels | Paper's params | My params | Paper's ops | My mult-adds |
|---|---|---|---|---|---|---|
| 3a | 28 × 28 | 256 | 159K | 163K | 128M | 128M |
| 3b | 28 × 28 | 480 | 380K | 388K | 304M | 304M |
| 4a | 14 × 14 | 512 | 364K | 376K | 73M | 74M |
| 4b | 14 × 14 | 512 | 437K | 449K | 88M | 88M |
| 4c | 14 × 14 | 512 | 463K | 509K | 100M | 100M |
| 4d | 14 × 14 | 528 | 580K | 605K | 119M | 119M |
| 4e | 14 × 14 | 832 | 840K | 867K | 170M | 170M |
| 5a | 7 × 7 | 832 | 1072K | 1042K | 54M | 51M |
| 5b | 7 × 7 | 1024 | 1388K | 1443K | 71M | 71M |

**What each column is, and why it matters:**
- **Output channels:** 1 × 1 + 3 × 3 + 5 × 5 + pool proj filters; for 3a it is 64 + 128 + 32 + 32 = 256, which agrees with the table and with torchvision's `inception3a` (output shape (1, 256, 28, 28)).
- **Paper's params / ops:** as printed (K = thousands of parameters, M = millions of operations).
- **My params / mult-adds:** my own count from the filter numbers. They show whether the table is self-consistent.

**Summary: what the table shows.** The operations column agrees with my count to within 1 million in all but one row (5a: 54M against my 51M). The parameter column does not: my counts are 2% to 4% above the printed ones in most rows, 10% above in 4c (509K against 463K) and 3% below in 5a. For 4c the printed ops, 100M, fit my parameter count (509K × 14 × 14 = 99.8M) and would not fit the printed 463K (463K × 196 = 90.7M), so I think the printed parameter column is the less reliable one; I did not find out why. The printed columns add up to about 6.8M parameters (2.7K + 112K + the nine modules + 1000K) and 1,502M operations, which matches "1.5 billion multiply-adds" (paper, 1). My count, including a stem of my own (the 1 × 1 + 3 × 3 convolutions) and the classifier, is 6.99M parameters and about 1.58 billion mult-adds. Caution: I rebuilt the stem from the table's rows, so the stem figure is the least certain.

**The 1 × 1 reduction in numbers (mine).** In module 3a (192 input channels), a direct 5 × 5 convolution to 32 channels has $192\times32\times25=153{,}600$ weights; with a 16-channel 1 × 1 reduction first it has $192\times16+16\times32\times25=15{,}872$, **9.7 times fewer** (mult-adds 120.4M against 12.4M at 28 × 28). In 4a (480 channels, 16 reduction, 48 outputs) it is 576,000 against 26,880, **21.4 times fewer**. The naïve module would also give 64 + 128 + 32 + 192 (the pooling branch keeps all 192 input channels) = **416 output channels** instead of 256, so the channels would keep growing from stage to stage, which is the "computational blow up within a few stages" in the paper (paper, 4).

**Cross-check against torchvision (mine).** `torchvision.models.googlenet(aux_logits=False)` has 6.62M parameters and gives a (1, 1000) output for a 224 × 224 input. Its Inception module uses 3 × 3 kernels in the branch that the paper draws as 5 × 5 (I printed the kernel shapes of `inception3a`), so it is a close relative, not an exact copy, of the paper's network. For scale, torchvision's AlexNet has 61.1M parameters, 9.2 times the GoogLeNet count; the paper says GoogLeNet "uses 12× fewer parameters than the winning architecture of Krizhevsky et al" (paper, 1). I could not reproduce 12× (I get about 9 from the paper's own table total, 60.97M for the original AlexNet by my count in [[R-CNN 2013]] against 6.8M), and I do not know what the paper counted.

## The core mechanism, step by step

### Why 1 × 1 reductions let the module be wide
*Idea.* A convolution's cost is proportional to (input channels) × (output channels) × (kernel area). Before an expensive large kernel, a 1 × 1 convolution squeezes the many input channels into a few.
1. Take the input with $C_{in}$ channels (192 in module 3a).
2. Apply a 1 × 1 convolution with $r$ filters (16 for the 5 × 5 branch).
   - *What it does:* mixes channels at each position into $r$ numbers.
   - *Why it matters:* the next, expensive step now sees $r$ channels, not $C_{in}$.
3. Apply the 5 × 5 convolution with $C_{out}$ filters (32).
4. Concatenate with the other branches.

**Worked example (mine, from above).** 153,600 weights become 15,872, 9.7 times fewer. The price: information passes through a narrow "bottleneck" of 16 channels. The paper's reasoning is that "even low dimensional embeddings might contain a lot of information about a relatively large image patch" but "compressed information is harder to model", so it compresses "only whenever they have to be aggregated en masse" (paper, 4).

### Training with auxiliary losses (paper, 5 and 6)
1. Run the image through the network; the main classifier and the two side classifiers each give a softmax loss.
2. Add the side losses to the main one with a weight of 0.3 each.
3. Update with asynchronous SGD, 0.9 [[Momentum]], a learning rate lowered by 4% every 8 epochs, and Polyak averaging of the weights for the final model ([[Stochastic Gradient Descent]]). The paper says the run used the DistBelief system on CPUs only and estimates it "could be trained to convergence using few high-end GPUs within a week" (paper, 6).
4. At test time drop the side classifiers.

### Test-time tricks that the score depends on (paper, 7)
- **Ensemble of 7 models** (the same GoogLeNet, one of them wider), "trained with the same initialization (even with the same initial weights, mainly because of an oversight)", differing only in sampling and image order.
- **144 crops per image:** resize to 4 scales (shorter side 256, 288, 320, 352), take the left, centre and right squares (top, centre, bottom for portrait), then for each square the 4 corners and the centre 224 × 224 crop and the square resized to 224 × 224, and the mirror of each: $4\times3\times6\times2=144$ (my check: 144).
- **Average the softmax probabilities** over crops and models; alternatives such as max pooling over crops were worse.

## The equations
The paper has no numbered equations. Two formulas carry its two main mechanisms, and I write them out myself from the text.

**1. The cost of a convolution layer, and what a 1 × 1 reduction does to it** (general knowledge; the paper explains it in words in sections 3 and 4)

$$
P_{\text{conv}}=C_{\text{in}}\,C_{\text{out}}\,k^{2},\qquad P_{\text{reduced}}=C_{\text{in}}\,r+r\,C_{\text{out}}\,k^{2}
$$

**Where:**
- $C_{\text{in}}$, $C_{\text{out}}$: the number of input and output channels; $k$: the kernel's side length (5 for the 5 × 5 branch). Biases are ignored.
- $r$: the number of filters of the 1 × 1 reduction in front ("#5×5 reduce" in Table 1).
- $P_{\text{conv}}$: the weights of the direct convolution; $P_{\text{reduced}}$: the weights of reduction plus convolution. Multiplying by the grid size (height × width) gives the multiply-adds.

**Matters:**
- The $k^2$ makes 5 × 5 convolutions 25 times as costly per channel pair as 1 × 1; shrinking $C_{\text{in}}$ to $r$ before the 5 × 5 is where the saving is. The paper's remark on "a quadratic increase of computation" comes from the product $C_{\text{in}}C_{\text{out}}$: doubling both quadruples the cost.
- Without the reduction a module's cost and output width would blow up in a few stages (paper, 4).

**Example (tiny numbers):** module 3a, $C_{\text{in}}=192$, $C_{\text{out}}=32$, $k=5$, $r=16$ (all from Table 1).
- $P_{\text{conv}}=192\times32\times25=153{,}600$.
- $P_{\text{reduced}}=192\times16+16\times32\times25=3{,}072+12{,}800=15{,}872$.
- Ratio: $153{,}600/15{,}872=9.7$.
- Meaning: the reduced path is almost ten times cheaper for the same output size, which is what lets the module have branches of four kinds at once.

**2. The training loss with auxiliary classifiers** (paper, 5; weights given in words)

$$
L_{\text{total}}=L_{\text{main}}+0.3\,L_{\text{aux}(4a)}+0.3\,L_{\text{aux}(4d)}
$$

**Where:**
- $L_{\text{main}}$: softmax (log) loss of the final classifier; $L_{\text{aux}(4a)}$, $L_{\text{aux}(4d)}$: the same loss for the two side classifiers on modules 4a and 4d.
- $0.3$: the "discount weight" of the paper's text: "the losses of the auxiliary classifiers were weighted by 0.3".

**Matters:**
- The side losses give the middle layers a direct gradient and "additional regularization" (paper, 5). Setting the weight to 0 removes them; the paper does not report that ablation.
- Giving them weight 1 would make the side classifiers as important as the real one, which is not what is wanted since they are discarded at test time (my reading).

**Example (tiny numbers):** $L_{\text{main}}=1.0$, $L_{\text{aux}(4a)}=1.5$, $L_{\text{aux}(4d)}=1.2$ (my toy numbers).
- Side part: $0.3\times1.5+0.3\times1.2=0.45+0.36=0.81$.
- Total: $1.0+0.81=1.81$.
- Meaning: the side classifiers add about 45% on top of the main loss here, but as a *training signal for earlier layers*; at test time only $L_{\text{main}}$'s network remains.

## What they found
Headlines (paper, 7, 8): top-5 error 6.67% on both validation and test data, first place in ILSVRC 2014 classification, with no external data; first place in detection with 43.9% mAP. Each table below is **written by me, an AI, from the paper's table text.**

### Classification results over the years (paper's Table 2)
| Team | Year | Place | Top-5 error (%) | External data |
|---|---|---|---|---|
| SuperVision | 2012 | 1st | 16.4 | no |
| SuperVision | 2012 | 1st | 15.3 | ImageNet 22k |
| Clarifai | 2013 | 1st | 11.7 | no |
| Clarifai | 2013 | 1st | 11.2 | ImageNet 22k |
| MSRA | 2014 | 3rd | 7.35 | no |
| VGG | 2014 | 2nd | 7.32 | no |
| GoogLeNet | 2014 | 1st | 6.67 | no |

**What each column is, and why it matters:**
- **Top-5 error:** the share of images whose true class is not among the model's five best guesses; the challenge ranks by it (paper, 7). Lower is better.
- **External data:** whether extra labelled images outside the 1.2 million training set were used ("ImageNet 22k" is a bigger set of classes).

**Summary: what the table shows.** GoogLeNet is best (6.67), SuperVision 2012 without extra data is worst (16.4). The paper counts a "56.5% relative reduction compared to the SuperVision approach in 2012" and "about 40%" against Clarifai 2013, both of which used external data; my arithmetic from the printed numbers gives 56.4% (against 15.3) and 40.4% (against 11.2). The margin over the runner-up is small: 6.67 against VGG's 7.32 (0.65 points, mine). Caution: GoogLeNet's number comes from a 7-model, 144-crop ensemble (see the next table), which the earlier entries did not use, and it is a single submission, with no error bars.

### What the test-time tricks are worth (paper's Table 3, validation set)
| Models | Crops | Cost (relative) | Top-5 error (%) | Change against the base (as printed) | Change (my arithmetic) |
|---|---|---|---|---|---|
| 1 | 1 | 1 | 10.07 | base | |
| 1 | 10 | 10 | 9.15 | −0.92 | −0.92 |
| 1 | 144 | 144 | 7.89 | −2.18 | −2.18 |
| 7 | 1 | 7 | 8.09 | −1.98 | −1.98 |
| 7 | 10 | 70 | 7.62 | −2.45 | −2.45 |
| 7 | 144 | 1008 | 6.67 | −3.45 | **−3.40** |

**What each column is, and why it matters:** *Models* and *Crops* are the ensemble size and the number of crops per image; *Cost* is their product in units of one model with one crop; *Top-5 error* is on the validation data, "in order to not overfit to the testing data statistics" (paper, 7).

**Summary: what the table shows.** One model with one crop reaches 10.07%; the winning 6.67% costs 1,008 times as much (7 × 144, my check), and nearly all of the gap to the 2013 results comes from that. A single model with 144 crops alone gets 7.89, and 7 models with one crop 8.09. The paper itself says "such aggressive cropping may not be necessary in real applications, as the benefit of more crops becomes marginal". **The paper contradicts itself in the last row:** it prints −3.45, but 10.07 − 6.67 = 3.40 (mine, run in Python). Caution: single runs on the validation set.

### Detection (paper's Tables 4 and 5)
| Team | Year | Place | mAP (%) | External data | Models in ensemble | Approach |
|---|---|---|---|---|---|---|
| UvA-Euvision | 2013 | 1st | 22.6 | none | ? | Fisher vectors |
| Deep Insight | 2014 | 3rd | 40.5 | ImageNet 1k | 3 | CNN |
| CUHK DeepID-Net | 2014 | 2nd | 40.7 | ImageNet 1k | ? | CNN |
| GoogLeNet | 2014 | 1st | 43.9 | ImageNet 1k | 6 | CNN |

| Single model (Table 5) | mAP (%) | Contextual model | Box regression |
|---|---|---|---|
| Trimps-Soushen | 31.6 | no | ? |
| Berkeley Vision | 34.5 | no | yes |
| UvA-Euvision | 35.4 | ? | ? |
| CUHK DeepID-Net2 | 37.7 | no | ? |
| GoogLeNet | 38.02 | no | no |
| Deep Insight | 40.2 | yes | yes |

**What each column is, and why it matters:**
- **mAP:** [[Mean Average Precision]] over 200 classes; a detection counts only if it has the right class and overlaps the true box by at least 50% (Jaccard index, [[Intersection over Union]]) (paper, 8).
- **Contextual model, box regression:** extras that GoogLeNet did *not* use ("due to lack of time"), but that some rivals did.

**Summary: what the tables show.** GoogLeNet is first with 43.9, 3.2 points above the runner-up (mine: 43.9 − 40.7) and 1.94 times the 2013 winner's 22.6 ("almost doubled" in the paper). Its method is the [[R-CNN 2013]] pipeline with the Inception network as the region classifier, proposals from Selective Search with doubled superpixel size (halving them) plus 200 MultiBox proposals (about 60% of R-CNN's number, with coverage raised from 92% to 93%, worth about 1% mAP), and an ensemble of 6 (40% to 43.9%). With a single model it scores 38.02, below Deep Insight's 40.2 (which uses context and regression) but above all the rest in Table 5; the ensemble adds about 5.9 points to GoogLeNet but only 0.3 to Deep Insight (paper, 8). Caution: the paper says nothing of the confidence of these differences; rows with "?" were not stated.

## Questions I had, answered

**Why 1 × 1 and what does "1 × 1 convolution" even do?** It mixes the channels at each position and ignores neighbours: for 192 channels in and 16 out it is 192 × 16 weights applied at every grid cell (paper, 2 and 4). In this paper it first cuts channels, then ReLU after it adds non-linearity.

**Why does the pooling branch need a projection?** A pooling layer outputs as many channels as it receives, so without a 1 × 1 projection the module's output width would grow with every stage (paper, 4; my count of 416 against 256 for 3a).

**Why are the first layers ordinary convolutions?** "For technical reasons (memory efficiency during training)", a limit of their implementation, "not strictly necessary" (paper, 4).

**Is the theory (sparse structure, Hebbian principle) proven by this?** No. The paper says the proof needs "very strong conditions", the guiding principles may not be why it works, and "the most convincing proof would be if an automated system would create network topologies resulting in similar gains in other domains" (paper, 3).

**Why "GoogLeNet" with a capital N in the middle?** The paper says it is "an homage to Yann LeCun's pioneering LeNet 5 network" (paper, 5).

## Why it mattered
It showed that a deeper, wider network could be made cheaper than a plainer one by changing its shape, and it won both the classification and the detection tracks in 2014 (paper). The Inception module (parallel branches with 1 × 1 reductions) and the habit of using average pooling in place of big fully connected layers appear in later networks; those later networks were not read for this note, so that is general knowledge from my other notes, not from this paper. Its limits were also instructive: [[ResNet 2015]] reached much greater depth with a plain skip connection instead of hand-designed branches.

## Limits and open questions
- **Hand-designed:** the filter counts of nine modules (Table 1) are listed with no search or ablation behind them, and the paper says "this requires careful manual design at this point" (paper, 4).
- **No ablations** of the auxiliary classifiers, the module's branches or the 1 × 1 reductions; the one comparison is the 0.6% top-1 gain from replacing fully connected layers with average pooling (paper, 5).
- **The score depends on the ensemble:** 6.67% needs 7 models and 144 crops (cost 1,008 times one model with one crop); a single model with one crop is at 10.07% (paper, Table 3).
- **Training details are fuzzy:** "it is hard to give a definitive guidance to the most effective single way to train these networks", because sampling methods changed over the months (paper, 6).
- **Detection lacks box regression and context** (paper, 8), and training on a single machine type (CPU DistBelief) is not reproducible by me.
- **Two internal inconsistencies** found: the printed parameter column of Table 1 against the operations column (my count), and Table 3's last row (−3.45 against 3.40). The "12× fewer parameters" claim does not come out of the paper's own table in my arithmetic (about 9×).
- **Single runs, no error bars** anywhere.

## Concepts used
[[Convolutional Neural Network]], [[Convolution]], [[ReLU]], [[Pooling]], [[Dropout]], [[Receptive Field]], [[Overfitting]], [[Stochastic Gradient Descent]], [[Momentum]], [[Data Augmentation]], [[Softmax]], [[Image Classification]], [[Object Detection]], [[Mean Average Precision]], [[Intersection over Union]], [[Transfer Learning]], [[Selective Search]], [[Ensemble Methods]]

## Reading trail
- **Came from:** [[AlexNet 2012]] and Network in Network (not yet a note); the same contest and year as [[VGG 2014]]
- **Led to:** [[ResNet 2015]] and the later Inception versions (not read); its detection pipeline follows [[R-CNN 2013]]; [[FCN 2014]]

## How much I read
- **Text:** the whole of arXiv 1409.4842 (12 pages), extracted with PyMuPDF: abstract, sections 1 to 9 and the conclusion, Tables 1 to 5. The reference list was skimmed, not read. I looked at one figure as a picture, the paper's Figure 2 (to compare the free redrawing); Figures 1 and 3 were not seen as pictures (Figure 3 is in the extraction only as a list of layer names).
- **Numbers:** "paper" marks are from the text and tables. The module sizes, naïve-against-reduced counts, relative reductions and Table 3 differences are mine (`googlenet_demo.py`, run once). The torchvision numbers are as printed by that run.
- **Not checked:** why the printed parameter column of Table 1 differs from my count, the origin of "12×", the Arora et al. theory, any later Inception paper.
- The note was not opened in Obsidian and the flashcards are unreviewed.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-vision GoogLeNet (Szegedy et al., arXiv 1409.4842): what problem did it attack?::Improving image recognition without just making networks bigger: more size means more overfitting and quadratically more compute, so it wanted quality gains at a fixed budget (about 1.5 billion multiply-adds at inference) (paper, 1 and 3).

#flashcards/research/concept #flashcards/topic/research-vision What is the idea behind the Inception module?::Approximate an optimal sparse structure with dense building blocks: run 1 by 1, 3 by 3, 5 by 5 convolutions and a pooling path in parallel on the same input, concatenate their outputs, and put 1 by 1 convolutions before the expensive ones to cut channels (paper, 4).

#flashcards/research/concept #flashcards/topic/research-vision Why not use sparse matrices to build a sparse network?::Computing hardware is inefficient on non-uniform sparse data: even a 100 times cut in arithmetic is eaten by lookups and cache misses, and tuned dense libraries are very fast, so the paper covers the sparse structure with dense components instead (paper, 3).

#flashcards/research/concept #flashcards/topic/research-vision What are the four branches of the dimension-reduced Inception module?::1 by 1 convolution; 1 by 1 reduce then 3 by 3; 1 by 1 reduce then 5 by 5; 3 by 3 max pooling then 1 by 1 projection; the outputs are concatenated along channels (paper, Figure 2b, 4).

#flashcards/research/concept #flashcards/topic/research-vision Why does the pooling branch of an Inception module have a 1 by 1 projection?::A pooling branch has as many output channels as its input, so without a projection the module's output width would grow stage after stage ("computational blow up"); in 3a the naive version would give 416 channels instead of 256 (paper, 4; my count).

#flashcards/research/concept #flashcards/topic/research-vision What are the 1 by 1 convolutions in GoogLeNet used for?::Two things: dimension reduction before the 3 by 3 and 5 by 5 convolutions (removing computational bottlenecks), and, because ReLU follows them, extra non-linearity ("dual-purpose") (paper, 2 and 4).

#flashcards/research/concept #flashcards/topic/research-vision What does the GoogLeNet network look like overall?::A stem of ordinary convolutions and max pools, nine Inception modules in three groups (3a-3b, 4a-4e, 5a-5b) with stride-2 max pooling between them, average pooling, 40% dropout, one linear 1000-way layer and softmax; 22 layers deep counting layers with parameters (paper, 5, Table 1).

#flashcards/research/concept #flashcards/topic/research-vision Why does GoogLeNet use average pooling instead of fully connected layers?::Moving from fully connected layers to average pooling improved top-1 accuracy by about 0.6%, though dropout stayed essential; an extra linear layer is kept to make fine-tuning on other label sets easy (paper, 5).

#flashcards/research/concept #flashcards/topic/research-vision What are GoogLeNet's auxiliary classifiers and why are they there?::Small side networks on modules 4a and 4d (5 by 5 average pool with stride 3, 1 by 1 convolution with 128 filters, a 1024-unit fully connected layer, 70% dropout, a 1000-way softmax). They add gradient signal to the middle layers and regularisation; their loss is weighted 0.3 and they are dropped at test time. The paper gives no ablation of them (paper, 5).

#flashcards/research/formula #flashcards/topic/research-vision Write the weight count of a convolution with and without a 1 by 1 reduction, and explain each part.
?
$$P_{\text{conv}}=C_{\text{in}}C_{\text{out}}k^{2},\qquad P_{\text{reduced}}=C_{\text{in}}r+rC_{\text{out}}k^{2}$$
- $C_{\text{in}}$, $C_{\text{out}}$: input and output channels; $k$: kernel side (biases ignored).
- $r$: filters of the 1 by 1 reduction placed in front.
- The reduction pays when $r$ is much smaller than $C_{\text{in}}$ (general knowledge; the paper explains it in words, sections 3 and 4).

#flashcards/research/example #flashcards/topic/research-vision Module 3a: how many weights does a direct 5 by 5 convolution (192 to 32 channels) have, and with a 16-channel 1 by 1 reduction?::Direct: 192 x 32 x 25 = 153,600. Reduced: 192 x 16 + 16 x 32 x 25 = 3,072 + 12,800 = 15,872, which is 9.7 times fewer; in 4a (480 channels) it is 21.4 times fewer (my arithmetic from Table 1).

#flashcards/research/formula #flashcards/topic/research-vision Write GoogLeNet's training loss with auxiliary classifiers and explain each part.
?
$$L_{\text{total}}=L_{\text{main}}+0.3\,L_{\text{aux}(4a)}+0.3\,L_{\text{aux}(4d)}$$
- $L_{\text{main}}$: softmax loss of the final classifier.
- $L_{\text{aux}}$: the same loss for the side classifiers on modules 4a and 4d.
- 0.3: the discount weight the paper's text gives; the side classifiers are discarded at inference (paper, 5; the formula is my writing of the text).

#flashcards/research/example #flashcards/topic/research-vision Main loss 1.0, auxiliary losses 1.5 and 1.2: what is GoogLeNet's total loss?::1.0 + 0.3 x 1.5 + 0.3 x 1.2 = 1.0 + 0.45 + 0.36 = 1.81 (my toy numbers).

#flashcards/research/image #flashcards/topic/research-vision ![[inception-module.png]]
What does the picture show?
?
The Inception module with dimension reductions: four parallel branches from the previous layer (1 by 1; 1 by 1 then 3 by 3; 1 by 1 then 5 by 5; 3 by 3 max pooling then 1 by 1) joined by a filter concatenation. A CC BY-SA 4.0 redrawing by Krishnachandranvn of the paper's Figure 2(b); I compared it by eye and only colours and label capitalisation differ.

#flashcards/research/concept #flashcards/topic/research-vision How big is GoogLeNet and how much work does it do?::Its Table 1 adds up to about 6.8 million parameters and 1.5 billion multiply-adds (the printed columns). My count from the channel numbers: 6.99 million parameters and about 1.58 billion multiply-adds, with the operations column matching but the parameter column differing by up to 10% in a module (4c).

#flashcards/research/concept #flashcards/topic/research-vision How did GoogLeNet do in ILSVRC 2014 classification?::Top-5 error 6.67% on validation and test data, first, with no external data, against VGG 7.32%, MSRA 7.35%, Clarifai 2013 11.7% (11.2% with extra data) and SuperVision 2012 16.4% (15.3% with extra data) (paper, Table 2).

#flashcards/research/concept #flashcards/topic/research-vision How much of GoogLeNet's 6.67% comes from the test-time tricks?::One model with one crop: 10.07%. 144 crops with one model: 7.89%; 7 models with one crop: 8.09%; 7 models with 144 crops: 6.67%, at 1,008 times the cost. The paper prints the last gain as 3.45 but 10.07 - 6.67 = 3.40 (paper, Table 3; my arithmetic).

#flashcards/research/example #flashcards/topic/research-vision How are GoogLeNet's 144 crops per image made?::Resize to 4 scales (shorter side 256, 288, 320, 352), take 3 squares of each (left, centre, right; top, centre, bottom for portrait), then 4 corners, the centre and the whole square resized to 224 by 224 (6 crops), and mirror them: 4 x 3 x 6 x 2 = 144 (paper, 7).

#flashcards/research/concept #flashcards/topic/research-vision How did GoogLeNet do on ILSVRC 2014 detection?::43.9% mAP (first, with an ensemble of 6 and ImageNet 1k pre-training), against 40.7% for CUHK DeepID-Net, 40.5% for Deep Insight and 22.6% for the 2013 winner (1.94 times, my arithmetic). Its method is R-CNN with Inception as the region classifier, but without bounding-box regression or context (paper, 8, Table 4).

#flashcards/research/concept #flashcards/topic/research-vision Where does the paper contradict itself or fail to add up?::Table 3's last gain is printed as -3.45 but 10.07 - 6.67 = 3.40; Table 1's parameter column differs from my count by up to 10% (4c: 463K printed, 509K mine) while the operations column agrees; and the claim of 12 times fewer parameters than AlexNet comes out at about 9 times in my arithmetic (paper, Table 1, 3 and 1).

#flashcards/research/concept #flashcards/topic/research-vision What are the limits of GoogLeNet?::Hand-designed module sizes with no search behind them, no ablations of the auxiliary classifiers or branches, a top score that needs 7 models and 144 crops, vague training details, and detection without box regression or context; the theory behind it is not proven by the paper (paper, 3 to 8).

#flashcards/research/concept #flashcards/topic/research-vision GoogLeNet: reading trail and how much did my note read?::It came from AlexNet and Network in Network (same contest as VGG) and led to ResNet and later Inception versions. The note read the whole body text and compared one figure as a picture; not read: Figures 1 and 3 as pictures, the references, any later Inception paper.
