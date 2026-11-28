---
tags: [paper, vision, CNN, ImageNet, AlexNet, image-classification, GPU]
link: https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
authors: Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton (University of Toronto; from the paper's first page)
year: 2012
status: read
field: image classification
---

# AlexNet 2012

> **In one line for a 12-year-old:** A computer learned to look at pictures by stacking eight layers of pattern-finders, and it was trained on two fast graphics cards. It beat every other method in a big picture-naming contest by a huge margin.

*Full title: "ImageNet Classification with Deep Convolutional Neural Networks" (NIPS 2012, the conference now called NeurIPS; not on arXiv). I read the 9-page proceedings PDF from papers.nips.cc, the link in the header. Layout follows [[Attention Is All You Need 2017]] and [[R-CNN 2013]]. Marks: "(paper, section)", "(general knowledge)", "(mine)". The older note [[LeNet and AlexNet]] counts the layers in a modern single-GPU layout; this note follows the paper.*

## The problem
Recognise objects in realistic photos. Datasets of labelled images had been "on the order of tens of thousands" (NORB, Caltech-101/256, CIFAR-10/100), enough for simple tasks (MNIST error under 0.3% "approaches human performance") but not for real-world variability. ImageNet, with "over 15 million labeled high-resolution images in over 22,000 categories", made big training sets possible (paper, 1). The question: how to learn a model with enough capacity for thousands of objects, which also "should have lots of prior knowledge to compensate for all the data we don't have", and how to train it without overfitting or waiting forever (paper, 1).

## What people did before
- **Hand-made features and shallow classifiers.** In ILSVRC-2010 the best entry (47.1% top-1 and 28.2% top-5 error) "averages the predictions produced from six sparse-coding models trained on different features", and the best published result since then (45.7% and 25.7%) averaged two classifiers on Fisher Vectors of densely sampled features (paper, 6). In ILSVRC-2012 the second-best entry (26.2% top-5 error) averaged classifiers on Fisher Vectors (paper, 6).
- **Convolutional networks, small scale.** CNNs ("stationarity of statistics and locality of pixel dependencies") were known (LeNet and others), had "much fewer connections and parameters" than fully connected nets of the same size, but "have still been prohibitively expensive to apply in large scale to high-resolution images" (paper, 1; see [[LeNet and AlexNet]]).
- **Unsupervised pre-training** was the usual route to deeper networks. The paper does not use it, "to simplify our experiments", and expects it would help with more compute (paper, 7).
- **Saturating neurons** (tanh and the logistic function), slow to train by gradient descent (paper, 3.1).

## The idea
No single trick: a bundle of choices that together made a large CNN trainable on 1.2 million images in about a week. The paper lists them in order of importance (paper, 3): **ReLU neurons** (several times faster training), **two GPUs** (a bigger network than one 3 GB card holds), **local response normalisation**, **overlapping pooling**; and against overfitting (paper, 4): **data augmentation** (crops, mirror images, colour changes) and **dropout**. The network has five convolution layers and three fully connected layers, about 60 million parameters (paper, abstract and 3.5). The authors also publish their fast GPU convolution code (paper, 1).

## The architecture, part by part

The paper's Figure 2 (the two-GPU architecture), Figure 3 (96 first-layer kernels), Figure 4 (example predictions and nearest-neighbour retrieval) and Figure 1 (ReLU against tanh training speed) are in the NIPS proceedings; I found no open licence for them and copied none. The architecture figure below is a free redrawing.

![[alexnet-original-block-diagram.svg]]
*The AlexNet architecture of the paper's Figure 2: the 224 × 224 × 3 input, 11 × 11 kernels with stride 4 giving 55 × 55 maps of 48 channels per GPU, max pooling, 5 × 5 kernels (27 × 27 maps of 128 per GPU), more max pooling, three 3 × 3 convolution layers (13 × 13 maps of 192, 192 and 128 per GPU) and max pooling, then dense layers of 2048 per GPU and a 1000-way output. Authors: Aston Zhang, Zachary C. Lipton, Mu Li and Alexander J. Smola (the "Dive into Deep Learning" book), CC BY-SA 4.0, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:AlexNet_Original_block_diagram.svg). It is a redrawing of the paper's Figure 2, not the paper's own. I compared it by eye with the figure rendered from the PDF: the labels and sizes match. Both are cut off at the top, so the upper GPU's row is mostly out of the frame in the redrawing and in the proceedings PDF alike.*

The whole design (paper, 3.5). "Eight learned layers": five convolutional and three fully connected, the last feeding a 1000-way softmax. The network "maximizes the multinomial logistic regression objective", which is the average log-probability of the correct label. [[ReLU]] follows every convolutional and fully connected layer. The 224 × 224 × 3 input goes through:

| Layer | What it is | Kernels or units |
|---|---|---|
| 1 | convolution 11 × 11 × 3, stride 4, then response normalisation, then max pooling | 96 |
| 2 | convolution 5 × 5 × 48, then normalisation, then max pooling | 256 |
| 3 | convolution 3 × 3 × 256 (connected to all maps of layer 2) | 384 |
| 4 | convolution 3 × 3 × 192 | 384 |
| 5 | convolution 3 × 3 × 192, then max pooling | 256 |
| 6, 7 | fully connected, with dropout | 4,096 each |
| 8 | fully connected, then softmax | 1,000 |

*Written by me, an AI, from section 3.5 and Figure 2. The depths 48 and 192 are the per-GPU channel counts: layers 2, 4 and 5 only connect to maps on the same GPU, layer 3 sees all maps of layer 2 (paper, 3.5).*

### Part by part
- **ReLU nonlinearity** (paper, 3.1)
  - *What it does:* $f(x)=\max(0,x)$ instead of $\tanh(x)$ or $(1+e^{-x})^{-1}$.
  - *Why it matters:* "deep convolutional neural networks with ReLUs train several times faster": in the paper's Figure 1, a four-layer CNN on CIFAR-10 reaches a 25% training error "six times faster" than the same network with tanh. The paper says "we would not have been able to experiment with such large neural networks" otherwise. ReLU also needs no input normalisation to avoid saturating (paper, 3.3).
- **Two GPUs** (paper, 3.2)
  - *What it does:* half of the kernels sit on each GTX 580 (3 GB), and the GPUs "communicate only in certain layers" (layer 3 reads both halves; layers 4 and 5 read only their own half).
  - *Why it matters:* "a single GTX 580 GPU has only 3GB of memory, which limits the maximum size of the networks". The two-GPU net lowers top-1 and top-5 error by 1.7% and 1.2% against a one-GPU net with half as many kernels per layer, and "takes slightly less time to train". The paper's footnote says that comparison is "biased in favor of the one-GPU net", since that net is bigger than half size.
- **Local response normalisation** (paper, 3.3; the formula is equation 1 below)
  - *What it does:* divides each neuron's activity by a term made of the squared activities of its neighbouring kernel maps at the same position. This "implements a form of lateral inhibition", creating competition among kernel maps.
  - *Why it matters:* it lowers top-1 and top-5 error by 1.4% and 1.2%, and a four-layer CIFAR-10 CNN went from 13% to 11% test error with it (paper, 3.3). Later networks mostly dropped it (general knowledge, not in this paper).
- **Overlapping pooling** (paper, 3.4)
  - *What it does:* pooling windows of size $z=3$ placed $s=2$ apart (so they overlap), instead of $z=s=2$.
  - *Why it matters:* it lowers top-1 and top-5 error by 0.4% and 0.3%, "as compared with the non-overlapping scheme $s=2$, $z=2$, which produces output of equivalent dimensions", and overfitting seemed slightly harder.
- **Data augmentation** (paper, 4.1; see [[Data Augmentation]])
  - *What it does:* two forms, both computed on the CPU while the GPU trains: random 224 × 224 crops and horizontal mirror images of the 256 × 256 image, and a change of the RGB intensities along the principal components of the colour data (equation 3 below).
  - *Why it matters:* the crops and mirrors raise the training set "by a factor of 2048"; "without this scheme, our network suffers from substantial overfitting, which would have forced us to use much smaller networks". The colour step "reduces the top-1 error rate by over 1%".
- **Dropout** (paper, 4.2; see [[Dropout]])
  - *What it does:* in the first two fully connected layers, set each neuron's output to zero with probability 0.5; at test time use all neurons and multiply their outputs by 0.5.
  - *Why it matters:* it is "a very efficient version of model combination that only costs about a factor of two during training"; without it "our network exhibits substantial overfitting", and with it the training needs about twice as many iterations to converge.
- **Depth** (paper, 1 and 7)
  - *Why it matters:* "removing any convolutional layer (each of which contains no more than 1% of the model's parameters) resulted in inferior performance", about 2% worse top-1 for the middle layers; "the depth really is important". (My count in the size check below puts layers 3 and 4 slightly above 1%.)
- **Why these numbers and not others?** The paper tuned the normalisation constants ($k=2$, $n=5$, $\alpha=10^{-4}$, $\beta=0.75$) on a validation set (paper, 3.3) and chose the connection pattern between GPUs by cross-validation (paper, 3.2). For most of the rest (11 × 11 kernels, stride 4, 4,096 units, 0.5 dropout) it gives no ablation.

### Size and cost check (mine)
From `alexnet_demo.py` (Python 3.9.6, NumPy, PyTorch 2.8.0, torchvision 0.23.0, CPU), using the layer sizes of section 3.5 and a 227 × 227 input (see the contradiction below):

| Layer | Output grid | Parameters | Multiply-adds | Neurons |
|---|---|---|---|---|
| conv1 | 55 × 55 | 34,944 | 105.4M | 290,400 |
| conv2 | 27 × 27 | 307,456 | 223.9M | 186,624 |
| conv3 | 13 × 13 | 885,120 | 149.5M | 64,896 |
| conv4 | 13 × 13 | 663,936 | 112.1M | 64,896 |
| conv5 | 13 × 13 | 442,624 | 74.8M | 43,264 |
| fc6 | | 37,752,832 | 37.7M | 4,096 |
| fc7 | | 16,781,312 | 16.8M | 4,096 |
| fc8 | | 4,097,000 | 4.1M | 1,000 |
| **All** | | **60,965,224** | **724M** | 659,272 |

- **Columns:** *Parameters* are weights plus biases, with the two-GPU groups (layers 2, 4, 5 see half of the previous channels). *Multiply-adds* are weights times output positions. *Neurons* are output channels times grid cells.
- The total, 60.97 million, agrees with the paper's "60 million parameters". Fully connected layers hold 96.2% of the parameters and the convolution layers 91.9% of the multiply-adds (mine; the older note counted 96% and 91.8% in a different layout). The five convolution layers are only 3.8% of the parameters.
- **Where the paper's figures do not match my arithmetic:**
  - *Input size:* the paper says 224 × 224 × 3 (text and Figure 2), but an 11 × 11 kernel with stride 4 on 224 gives a 54.25-wide map, not a whole number; 227 gives the 55 of Figure 2. (The 227 is mine; the R-CNN paper's network, a CaffeNet version of this one, uses 227 × 227 inputs, see [[R-CNN 2013]].)
  - *Neuron counts:* Figure 2's caption lists 253,440, 186,624, 64,896, 64,896, 43,264, 4,096, 4,096 and 1,000, which add up to 622,312, and the abstract says 650,000. My count of layer 1 is 96 × 55 × 55 = 290,400; the caption's 253,440 equals 96 × 55 × 48, so it looks like a slip in one factor (my reading). With my layer 1 the total is 659,272, close to the abstract's figure.
  - *Convolution share of parameters:* the paper says each convolution layer has "no more than 1%" of the parameters; I get 0.06%, 0.5%, 1.45%, 1.09% and 0.73% for layers 1 to 5, so layers 3 and 4 are above 1%. I do not know how the paper counted.
  - *Augmentation factor:* the paper says "a factor of 2048"; 32 × 32 × 2 = 2,048, but a 256-wide image has 33 crop positions per side, which would give 33 × 33 × 2 = 2,178 (mine). It is the count of offsets 0 to 31, so the paper's number is a rounding choice, not an error of consequence.
- torchvision's `alexnet` (a single-GPU variant, 64 first-layer kernels) has 61,100,840 parameters; it is a relative of the paper's network, not the same.

## The core mechanism, step by step

### Why ReLU trains faster
*Idea.* A saturating neuron such as $\tanh$ has a nearly flat slope when its input is large, so the gradient there is tiny and learning stalls. A ReLU has slope 1 for any positive input.
1. Compute the neuron's weighted sum $x$.
2. Output $\max(0,x)$.
   - *What it does:* passes positive values unchanged, zeros out negatives.
   - *Why it matters:* the gradient is 1 whenever the neuron is active, so there is no vanishing slope on the positive side; "if at least some training examples produce a positive input to a ReLU, learning will happen in that neuron" (paper, 3.3).
**Worked example (mine).** For $x=5$: $\tanh$ has slope $1-\tanh^2(5)=1.8\times10^{-4}$; ReLU has slope 1. That is about 5,500 times more gradient (my arithmetic, general knowledge for the slope formula; the paper's own evidence is the six-fold speed-up in its Figure 1).

### How dropout and the 0.5 factor fit together
1. Training: for every example, each hidden unit in fc6 and fc7 is kept with probability 0.5 and zeroed otherwise, so every example sees a different "thinned" network; all share weights (paper, 4.2).
2. Test: keep every unit and multiply outputs by 0.5, "a reasonable approximation to taking the geometric mean of the predictive distributions produced by the exponentially-many dropout networks".
   - *Why the 0.5:* in training only half of the 4,096 units are active on average (2,048, mine), so at test time all 4,096 would give a layer input twice as big as the next layer was trained on; the 0.5 restores the average size (my reading; the paper states the geometric-mean reason).

### Training with test-time averaging (paper, 4.1 and 6)
At test time the network takes five 224 × 224 patches (four corners and the centre) and their mirror images, ten in all, and averages the softmax outputs. Footnote 5: without the ten-patch average, the ILSVRC-2010 errors are 39.0% and 18.3% against 37.5% and 17.0% (1.5 and 1.3 points better with it, mine).

## The equations

**1. Local response normalisation** (paper, 3.3)

$$
b^{i}_{x,y}=a^{i}_{x,y}\Bigg/\Bigg(k+\alpha\sum_{j=\max(0,\,i-n/2)}^{\min(N-1,\,i+n/2)}\bigl(a^{j}_{x,y}\bigr)^{2}\Bigg)^{\beta}
$$

**Where:**
- $a^{i}_{x,y}$: the activity of the neuron from kernel $i$ at position $(x,y)$, after ReLU; $b^{i}_{x,y}$: the normalised activity.
- The sum runs over $n$ "adjacent" kernel maps at the same position; $N$ is the total number of kernels in the layer. The kernel order is arbitrary and fixed before training.
- $k=2$, $n=5$, $\alpha=10^{-4}$, $\beta=0.75$: constants "determined using a validation set" (paper, 3.3).

**Matters:**
- A neuron with strong neighbours gets divided by a larger number, so strong activations suppress each other: competition ("lateral inhibition"). The paper calls it "brightness normalization" since the mean is not subtracted.
- Without it (in the paper's test) top-1 and top-5 error are 1.4% and 1.2% higher.

**Example (tiny numbers):** eight kernel maps at one position with activities $a=(0,1,4,20,3,0.5,0,2)$, using the paper's constants (my numbers).
- For $i=3$ ($a=20$) the window is kernels 1 to 5, with squares $1+16+400+9+0.25=426.25$. Denominator: $(2+10^{-4}\times426.25)^{0.75}=1.7086$. So $b=20/1.7086=11.7055$.
- If the neighbours were all zero the denominator is $2^{0.75}=1.6818$ and the factor 0.5946; here it is 0.5853, only slightly lower, because these activities are small.
- With all five maps of the window at 100 (the neuron itself included): sum of squares 50,000, denominator $(2+5)^{0.75}=4.3035$, $b=23.2$; a lone 100 among zeros gives denominator $3^{0.75}=2.2795$ and $b=43.9$. Meaning: crowded strong neighbours cut the output of the same activity from 43.9 to 23.2.

**2. The weight update** (paper, 5)

$$
v_{i+1}:=0.9\,v_i-0.0005\,\epsilon\,w_i-\epsilon\,\Bigl\langle\frac{\partial L}{\partial w}\Big|_{w_i}\Bigr\rangle_{D_i},\qquad w_{i+1}:=w_i+v_{i+1}
$$

**Where:**
- $i$: iteration; $w$: a weight; $v$: its momentum (velocity); $\epsilon$: the learning rate; $L$: the objective.
- $\langle\partial L/\partial w\rangle_{D_i}$: the gradient averaged over the $i$-th batch $D_i$ of 128 examples.
- 0.9: the momentum; 0.0005: the weight decay.

**Matters:**
- The momentum keeps part of the previous step, which smooths the path ([[Momentum]]).
- The decay term pulls every weight toward zero in proportion to its size. The paper says "weight decay here is not merely a regularizer: it reduces the model's training error", and that this small amount "was important for the model to learn" ([[Regularization]]).
- The learning rate started at 0.01 and was divided by 10 three times, by hand, "when the validation error rate stopped improving" (paper, 5).

**Example (tiny numbers):** $w=1$, $v=0$, $\epsilon=0.01$, a constant gradient of 0.5 (my numbers, `alexnet_demo.py`).
- Step 1: $v=0.9\times0-0.0005\times0.01\times1-0.01\times0.5=-0.005005$; $w=0.994995$.
- Step 2: $v=0.9\times(-0.005005)-0.0005\times0.01\times0.994995-0.005=-0.009509$; $w=0.985486$.
- Step 3: $v=-0.013563$; $w=0.971922$.
- With a zero gradient the decay alone shrinks $w$ by only 0.0005% per step. Meaning: the steps grow as momentum builds up (each step is bigger than the last), and the decay is a very small pull compared with the gradient term.

**3. Colour augmentation along the principal components** (paper, 4.1)

$$
I_{xy}\;\leftarrow\;I_{xy}+[\,\mathbf p_1,\mathbf p_2,\mathbf p_3\,]\,[\,\alpha_1\lambda_1,\ \alpha_2\lambda_2,\ \alpha_3\lambda_3\,]^{\top}
$$

**Where:**
- $I_{xy}=[I^R_{xy},I^G_{xy},I^B_{xy}]^{\top}$: the RGB value of the pixel at $(x,y)$.
- $\mathbf p_i$ and $\lambda_i$: the $i$-th eigenvector and eigenvalue of the $3\times3$ covariance matrix of RGB values over the training set ([[PCA]], [[Eigenvalue]], [[Covariance]]).
- $\alpha_i$: a random number drawn from a Gaussian with mean 0 and standard deviation 0.1, drawn once per image each time it is used (the same shift for every pixel of that image).

**Matters:**
- The shift follows the directions in which colours naturally vary in photos, mostly overall brightness, so object identity is kept: "object identity is invariant to changes in the intensity and color of the illumination".
- Scaling by $\lambda_i$ makes big natural variations get bigger shifts. The paper says this lowers top-1 error "by over 1%".

**Example (tiny numbers):** a made-up RGB covariance with 0.04, 0.04, 0.05 on the diagonal and 0.03, 0.02, 0.03 off it (my toy, not the real ImageNet matrix). Its eigenvalues are 0.0970, 0.0253 and 0.0077; the first eigenvector is $(-0.527,-0.596,-0.606)$, a brightness-like direction where R, G, B move together.
- For $\alpha=(0.1,-0.05,0.02)$ the shift added to every pixel is $(-0.0042,-0.0057,-0.0068)$ in the same units as the pixel values.
- 88% of the shift comes from the first component (0.1 × 0.097 against the rest, my share of absolute contributions). Meaning: the added colour change is mostly a uniform darkening or brightening, which keeps a "cat" a cat.

## What they found
Headlines (abstract): on ILSVRC-2010 test data "top-1 and top-5 error rates of 37.5% and 17.0%"; a variant won ILSVRC-2012 with "a winning top-5 test error rate of 15.3%, compared to 26.2% achieved by the second-best entry". Each table below is **written by me, an AI, from the paper's table text.**

### ILSVRC-2010 (paper's Table 1, test set)
| Model | Top-1 error (%) | Top-5 error (%) |
|---|---|---|
| Sparse coding | 47.1 | 28.2 |
| SIFT + Fisher Vectors | 45.7 | 25.7 |
| CNN (this paper) | 37.5 | 17.0 |

**What each column is, and why it matters:** *Top-1 error* is the share of test images whose most probable label is wrong; *top-5 error* the share whose correct label is not among the five most probable (paper, 2). Lower is better. The first two rows are the best results of others.

**Summary: what the table shows.** The CNN is best on both: 8.2 points better top-1 and 8.7 points better top-5 than the best published result (mine: 45.7 − 37.5 and 25.7 − 17.0); against the contest winner (sparse coding) 9.6 and 11.2 points (mine). Caution: a single trained model, and the ten-patch average is part of the score (1.5 and 1.3 points; footnote 5).

### ILSVRC-2012 (paper's Table 2)
| Model | Top-1 (val) | Top-5 (val) | Top-5 (test) |
|---|---|---|---|
| SIFT + FVs (second-best entry) | | | 26.2 |
| 1 CNN | 40.7 | 18.2 | |
| 5 CNNs | 38.1 | 16.4 | 16.4 |
| 1 CNN* | 39.0 | 16.6 | |
| 7 CNNs* | 36.7 | 15.4 | 15.3 |

**What each column is, and why it matters:** "val" is the validation set, "test" the competition's hidden test set. Rows with an asterisk were pre-trained to classify the whole ImageNet Fall 2011 release (15 million images, 22,000 categories) with an extra sixth convolution layer, then fine-tuned on ILSVRC-2012 ([[Fine-tuning]], [[Transfer Learning]]). The 5 and 7 CNN rows average the models' predictions.

**Summary: what the table shows.** The best row is 7 CNNs with pre-training (15.3 on test), 10.9 points and 41.6% relative better than the second-best entry's 26.2 (mine). One CNN gets 18.2 on validation; five average to 16.4 (1.8 points, mine); the extra pre-training on the 22,000-class data takes a single CNN to 16.6 (1.6 points better) and the 7-model average to 15.4 on validation. The paper says validation and test errors "do not differ by more than 0.1%" in its experience (5 CNNs: 16.4 on both, 7 CNNs: 15.4 and 15.3). Caution: the headline 15.3% is an ensemble of seven networks, two of them pre-trained on a much bigger data set; the single-network figure is 18.2% (validation). Single runs and no error bars.

### Other results and ablations
- **ImageNet Fall 2009 (10,184 categories, 8.9 million images):** top-1 and top-5 error of 67.4% and 40.9% with an extra sixth convolution layer, against 78.1% and 60.9% for the best published (paper, 6). Half of the images for training, half for testing, with the authors' own split.
- **Ablations quoted in the architecture section (ILSVRC-2010, error reductions in top-1 / top-5):** two GPUs 1.7% / 1.2%; response normalisation 1.4% / 1.2%; overlapping pooling 0.4% / 0.3%; colour augmentation "over 1%" top-1. Removing a single middle convolution layer costs about 2% top-1 (paper, 7). I list them as the paper printed them; none comes with an error bar or a repeat.
- **What the first layer learns (paper, 6.1, Figure 3):** "a variety of frequency- and orientation-selective kernels, as well as various colored blobs". The two GPUs specialise: GPU 1's kernels are "largely color-agnostic" and GPU 2's "largely color-specific", in every run, independent of the initialisation. I did not look at the figure.
- **Nearest neighbours in the 4,096-number last hidden layer (Figure 4):** images with a small Euclidean distance are semantically similar, though not close in pixels; the paper proposes compressing the vectors with an autoencoder for fast retrieval, which it does not do.

## Questions I had, answered

**Why was it a big deal?** A top-5 error of 15.3% against 26.2% for the second-best entry (10.9 points, mine) in one step (paper, abstract and 6; this paper compares only with the two best earlier results, not with earlier years).

**Why two GPUs and why do they only talk in some layers?** The network does not fit into the 3 GB of one card, and constant communication would slow it down. The pattern of connections is "a problem for cross-validation" and lets the authors "tune the amount of communication until it is an acceptable fraction of the amount of computation" (paper, 3.2).

**Why is the input 224 and the images 256?** The image is rescaled to a short side of 256, the central 256 × 256 patch kept, and random 224 × 224 crops of that are used for training (paper, 2 and 4.1). Footnote 4: "This is the reason why the input images in Figure 2 are 224 × 224 × 3-dimensional." The arithmetic of the first layer needs 227 (above).

**What does "top-5 error" mean?** A prediction counts as right if the true class is among the five most probable guesses (paper, 2). It is more forgiving than top-1, important with 1,000 classes where many are near-twins (compare the two dog classes in the GoogLeNet paper).

**Is the 60 million parameters figure right?** My count from the paper's layer sizes is 60.97 million; see the size check.

## Why it mattered
The paper's own claim is modest: "results can be improved simply by waiting for faster GPUs and bigger datasets to become available" (paper, 1). Its effect was large: [[R-CNN 2013]] reused this network as the feature extractor for detection and said it "rekindled interest in CNNs" (the R-CNN paper's words, section 1), and the later networks ([[VGG 2014]], [[GoogLeNet 2014]], [[ResNet 2015]]) all took depth and GPU training for granted. The authors predicted the direction: "we would like to use very large and deep convolutional nets on video sequences" (paper, 7).

## Limits and open questions
- **Memory and compute limited the size:** "the network's size is limited mainly by the amount of memory available on current GPUs and by the amount of training time that we are willing to tolerate" (paper, 1); five to six days on two GTX 580 GPUs for about 90 passes over the 1.2 million images (paper, 5).
- **Overfitting is the central problem:** 60 million parameters against 1.2 million images, "10 bits of constraint" per example by the paper's estimate; it needed augmentation (a factor of 2048) and dropout (paper, 4).
- **Hand-tuned training:** the learning rate was adjusted by hand; the paper gives few ablations for the sizes it chose.
- **Hardware-shaped design:** the two-GPU split with a restricted connection pattern is a response to 3 GB cards, not an ideal (paper, 3.2).
- **Not tested:** unsupervised pre-training, which the authors "expect will help"; and no error bars on any number.
- **Inconsistencies found:** the 224 against 227 input, Figure 2's first neuron count, the ≤1% per-layer claim and the 650,000 neurons (above).

## Concepts used
[[Convolutional Neural Network]], [[Convolution]], [[ReLU]], [[Pooling]], [[Local Response Normalization]], [[Dropout]], [[Data Augmentation]], [[Regularization]], [[Overfitting]], [[Momentum]], [[Stochastic Gradient Descent]], [[Softmax]], [[GPU]], [[PCA]], [[Ensemble Methods]], [[Transfer Learning]], [[Image Classification]], [[ImageNet]]

## Reading trail
- **Came from:** LeNet-5 (see [[LeNet and AlexNet]]) and the ImageNet contest ([[History of Image Classification]])
- **Led to:** [[R-CNN 2013]] (reuses the network), [[VGG 2014]], [[GoogLeNet 2014]], [[ResNet 2015]]

## How much I read
- **Text:** the whole body of the 9-page NIPS PDF, extracted with PyMuPDF: abstract, sections 1 to 7, Tables 1 and 2 and the footnotes. The reference list was skimmed, not read. I looked at one figure as a picture, Figure 2 (to compare the free redrawing); Figures 1, 3 and 4 were not seen as pictures.
- **Numbers:** "paper" marks are from the text and tables; layer sizes, parameter and multiply-add counts, the worked examples and the point differences are mine (`alexnet_demo.py`, run once). The 227 input is my assumption to make the first layer's arithmetic work.
- **Not checked:** the authors' code (cuda-convnet), how the paper counted its 650,000 neurons and its 1% per layer, and the supplementary material it mentions.
- The note was not opened in Obsidian and the flashcards are unreviewed.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-vision AlexNet (Krizhevsky, Sutskever and Hinton, NIPS 2012): what problem did it attack?::Recognising objects in realistic photos: small labelled sets (tens of thousands of images) were enough for simple tasks, but ImageNet (millions of images, thousands of categories) needed a high-capacity model that could be trained without overfitting in reasonable time (paper, 1).

#flashcards/research/concept #flashcards/topic/research-vision What was AlexNet's idea?::No single trick but a bundle: a deep CNN (5 convolution and 3 fully connected layers, about 60 million parameters) with ReLUs, two-GPU training, local response normalisation and overlapping pooling, plus data augmentation and dropout against overfitting (paper, 1 and 3).

#flashcards/research/concept #flashcards/topic/research-vision What does the AlexNet architecture look like?::224 by 224 by 3 input; conv 11 by 11 stride 4 (96 kernels), normalisation, pool; conv 5 by 5 (256), normalisation, pool; conv 3 by 3 (384), conv 3 by 3 (384), conv 3 by 3 (256), pool; two fully connected layers of 4096 with dropout; 1000-way softmax. ReLU after every conv and fully connected layer (paper, 3.5).

#flashcards/research/image #flashcards/topic/research-vision ![[alexnet-original-block-diagram.svg]]
What does the picture show?
?
The AlexNet architecture of the paper's Figure 2: 224 by 224 by 3 input, 11 by 11 kernels with stride 4 to 55 by 55 maps, max pooling, 5 by 5 kernels to 27 by 27, pooling, three 3 by 3 layers at 13 by 13, pooling, dense layers of 2048 per GPU and a 1000-way output, split over two GPUs. A CC BY-SA 4.0 redrawing by Zhang, Lipton, Li and Smola (the paper's figure is cropped at the top too); I compared it by eye.

#flashcards/research/concept #flashcards/topic/research-vision Why did AlexNet use ReLU neurons?::f(x) = max(0, x) trains several times faster than tanh or logistic neurons (a four-layer CNN reached 25% training error on CIFAR-10 six times faster); saturating neurons have flat slopes that stall learning, and ReLUs need no input normalisation to avoid saturating (paper, 3.1 and 3.3).

#flashcards/research/concept #flashcards/topic/research-vision Why was AlexNet trained on two GPUs and how were they connected?::A single GTX 580 has 3 GB, too little for the network, so half the kernels sit on each GPU and the GPUs communicate only in some layers (layer 3 reads both halves; layers 4 and 5 only their own). This cut top-1 and top-5 error by 1.7% and 1.2% against a one-GPU net with half the kernels per layer (paper, 3.2).

#flashcards/research/concept #flashcards/topic/research-vision What is overlapping pooling and what did it give?::Pooling windows of size 3 placed 2 apart (so they overlap) instead of 2 by 2; the output size is the same, top-1 and top-5 error fell by 0.4% and 0.3%, and overfitting seemed slightly harder (paper, 3.4).

#flashcards/research/formula #flashcards/topic/research-vision Write AlexNet's local response normalisation and explain each part.
?
$$b^{i}_{x,y}=a^{i}_{x,y}\Big/\Big(k+\alpha\sum_{j=\max(0,i-n/2)}^{\min(N-1,i+n/2)}(a^{j}_{x,y})^{2}\Big)^{\beta}$$
- $a^i_{x,y}$: activity of kernel $i$ at position $(x,y)$ after ReLU; $b$: the normalised activity.
- The sum is over $n$ adjacent kernel maps at the same position ($N$ kernels in the layer): strong neighbours suppress each other (lateral inhibition).
- $k=2$, $n=5$, $\alpha=10^{-4}$, $\beta=0.75$, chosen on a validation set; the effect was 1.4% and 1.2% lower top-1 and top-5 error (paper, 3.3).

#flashcards/research/example #flashcards/topic/research-vision Local response normalisation with the paper's constants: what is b for an activity of 100 when all five maps of its window are 100, and when the others are zero?::All five at 100 (the neuron itself included): sum of squares 50,000, denominator (2 + 5)^0.75 = 4.3035, b = 23.2. A lone 100 among zeros: (2 + 1)^0.75 = 2.2795, b = 43.9. Crowded strong neighbours cut the output by almost half (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-vision How did AlexNet fight overfitting?::Data augmentation (random 224 by 224 crops and mirror images of 256 by 256 images, a factor of 2048; and a colour shift along the RGB principal components, worth over 1% top-1) and dropout 0.5 in the first two fully connected layers (paper, 4).

#flashcards/research/formula #flashcards/topic/research-vision Write AlexNet's colour augmentation and explain each part.
?
$$I_{xy}\leftarrow I_{xy}+[\mathbf p_1,\mathbf p_2,\mathbf p_3][\alpha_1\lambda_1,\alpha_2\lambda_2,\alpha_3\lambda_3]^{\top}$$
- $I_{xy}$: RGB value of a pixel; $\mathbf p_i$, $\lambda_i$: eigenvectors and eigenvalues of the 3 by 3 covariance of RGB values over the training set.
- $\alpha_i$: Gaussian random numbers (mean 0, standard deviation 0.1), drawn once per image use and shared by all its pixels.
- It follows natural colour and brightness variation, so object identity is kept; it reduced top-1 error by over 1% (paper, 4.1).

#flashcards/research/example #flashcards/topic/research-vision A toy RGB covariance has eigenvalues 0.0970, 0.0253, 0.0077. For alpha = (0.1, -0.05, 0.02), where does AlexNet's colour shift come from?::The shift is P (alpha x lambda) = (-0.0042, -0.0057, -0.0068) added to every pixel; about 88% of it comes from the first component, a brightness-like direction where R, G and B move together (my toy matrix, not ImageNet's).

#flashcards/research/concept #flashcards/topic/research-vision How does AlexNet's dropout work at train and test time?::Train: each unit of the first two fully connected layers is zeroed with probability 0.5 per example (shared weights), which roughly doubles the iterations needed. Test: use all units and multiply outputs by 0.5, approximating the geometric mean of the many thinned networks' predictions (paper, 4.2).

#flashcards/research/formula #flashcards/topic/research-vision Write AlexNet's weight update and explain each part.
?
$$v_{i+1}:=0.9v_i-0.0005\,\epsilon\,w_i-\epsilon\langle\partial L/\partial w|_{w_i}\rangle_{D_i},\quad w_{i+1}:=w_i+v_{i+1}$$
- $v$: momentum (0.9 keeps part of the last step); $\epsilon$: learning rate (0.01, divided by 10 by hand when validation error stalled; three times).
- $\langle\cdot\rangle_{D_i}$: gradient averaged over a batch of 128 examples.
- 0.0005: weight decay; the paper says it "reduces the model's training error", not only regularises (paper, 5).

#flashcards/research/example #flashcards/topic/research-vision AlexNet update with w = 1, v = 0, learning rate 0.01, constant gradient 0.5: what are v and w after one and two steps?::Step 1: v = -0.0005 x 0.01 x 1 - 0.01 x 0.5 = -0.005005, w = 0.994995. Step 2: v = 0.9 x (-0.005005) - 0.0005 x 0.01 x 0.994995 - 0.005 = -0.009509, w = 0.985486 (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-vision How big is AlexNet by my count?::About 60.97 million parameters (fully connected layers 96.2%, the five convolution layers 3.8%), about 724 million multiply-adds (convolution layers 91.9%) and 659,272 neurons, using a 227 input; the paper says 60 million parameters and 650,000 neurons (paper, abstract; my arithmetic).

#flashcards/research/concept #flashcards/topic/research-vision How did AlexNet do on ILSVRC-2010?::Top-1 and top-5 error 37.5% and 17.0% on the test set against 47.1% and 28.2% for the contest's best entry (sparse coding) and 45.7% and 25.7% for the best later published result (Fisher Vectors); 9.6 and 11.2 points better than the winner (my arithmetic) (paper, 6, Table 1).

#flashcards/research/concept #flashcards/topic/research-vision How did AlexNet do on ILSVRC-2012?::A top-5 test error of 15.3% (7 CNNs, two pre-trained on the 22,000-class ImageNet release) against 26.2% for the second-best entry; one CNN had 18.2% top-5 (validation), five CNNs averaged 16.4% (paper, 6, Table 2).

#flashcards/research/concept #flashcards/topic/research-vision Where do the AlexNet paper's numbers not add up?::The input is called 224 by 224 but an 11 by 11 kernel with stride 4 gives a whole-number map only for 227; Figure 2's first neuron count (253,440) is 96 x 55 x 48, not my 96 x 55 x 55 = 290,400, and its counts sum to 622,312 against the abstract's 650,000; and layers 3 and 4 hold 1.45% and 1.09% of the parameters by my count, not "no more than 1%" (my arithmetic against the paper's text).

#flashcards/research/concept #flashcards/topic/research-vision What are the limits of AlexNet?::Size limited by GPU memory and training time (five to six days on two GTX 580s); heavy overfitting needing augmentation and dropout; hand-tuned learning rate and sizes with few ablations; a two-GPU split shaped by 3 GB cards; no unsupervised pre-training tried (paper, 1, 3.2, 5, 7).

#flashcards/research/concept #flashcards/topic/research-vision AlexNet: reading trail and how much did my note read?::It came from LeNet-5 and the ImageNet contest and led to R-CNN, VGG, GoogLeNet and ResNet. The note read the whole body text and compared one figure as a picture; not read: Figures 1, 3 and 4 as pictures, the references and the supplementary material.
