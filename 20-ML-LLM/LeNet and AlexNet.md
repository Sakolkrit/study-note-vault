---
tags: [lenet, alexnet, cnn, computer-vision, deep-learning, history, imagenet, ML]
status: growing
level: 4
tier: 3
---

# LeNet and AlexNet

**LeNet** (the 1990s) and **AlexNet** (2012) are the two milestones that most stories of the [[Convolutional Neural Network]] start with. LeNet showed that a small CNN trained with [[Backpropagation]] can read handwritten digits well enough to be used in real machines. AlexNet showed that the same idea, made much bigger and trained on [[GPU]]s with a large data set, can beat all older methods on photographs of everyday things. Both are pictures-in, label-out networks for [[Image Classification]].

## 🟢 1. Simple (age 12)
Imagine learning to read handwriting. First you get a small notebook of digits and a small brain trick for spotting strokes. That is LeNet: small, clever, and good enough to read the numbers on cheques. Years later, someone gives you a whole library of millions of photos, a super-fast helper (a graphics card) and a much bigger brain. That is AlexNet. It works on the same idea as LeNet, only far bigger, and suddenly it can name the things in everyday photos better than any older method.

**Example:** LeNet reads the amount on a bank cheque (digits). AlexNet looks at a photo and says "this is a Siberian husky" out of a thousand possible kinds of things.

## 🔵 2. How it works
1. **LeNet-5 is a short chain.** Picture in → [[Convolution]] with 6 filters → [[Pooling]] → convolution with 16 filters → pooling → one more convolution with 120 filters → two dense layers (84 units, then 10 for the ten digits).
2. **The picture gets smaller as the number of maps grows.** Each pooling step halves the width and height, while the number of [[Feature Map]]s grows from 6 to 16 to 120.
3. **AlexNet is the same recipe, bigger.** Five convolution layers (some followed by pooling), then three dense layers, ending in 1,000 outputs for 1,000 kinds of things.
4. **Three upgrades made AlexNet trainable.** The ReLU [[Activation Function]] (simpler and faster to learn with than the older tanh or sigmoid), [[Dropout]] in the dense layers (to fight [[Overfitting]]), and training on two [[GPU]]s. It also used [[Data Augmentation]], and ImageNet, a data set of over a million labelled photos.
5. **Most of the network's numbers are in the dense layers, but most of the computing is in the convolution layers.** Later networks such as [[ResNet]] cut the dense part.

![[lenet-5-architecture.svg]]
*A LeNet-5 style network with a 28×28 input. Convolution makes 6 maps of 28×28, pooling makes them 14×14, a second convolution makes 16 maps of 10×10, pooling gives 16 maps of 5×5, then dense layers of 120, 84 and 10 units. (The original 1998 network used a 32×32 input, which is the version I count in the code.) Image: Aston Zhang, Zachary C. Lipton, Mu Li and Alexander J. Smola ("Dive into Deep Learning"), [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:LeNet-5_architecture.svg), CC BY-SA 4.0.*

![[alexnet-architecture.png]]
*The AlexNet layout drawn as a chain of blocks: five convolution stages (grey boxes, shrinking in width and growing in depth), then the "classifier" of dense layers ending in the original 1,000-way output (upper row). The lower row is the same network with its last layer replaced by a new one, as is done in [[Transfer Learning]]. Image: Daniel Voigt Godoy, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:AlexNet_architecture.png), CC BY 4.0.*

## 🟠 3. Technical detail
All numbers come from the code at the end of this section (Python 3.9.6, PyTorch 2.8.0 and torchvision 0.23.0 on a CPU). The code only *builds* the networks with random weights and counts. It does not train or test anything, and it does not download any file.

- **LeNet-5 shaped, parameters:** the five layers hold 156, 2,416, 48,120, 10,164 and 850 numbers, 61,706 in total. This agrees with the "around 60,000" that the Wikipedia article on LeNet gives (which I read). My version is a modern rewrite (tanh, average pooling, a plain linear last layer). The 1998 paper used a different last layer and left some connections out of the third convolution (from memory, so its exact count differs a little).
- **LeNet-5, multiply-adds:** per layer 117,600, 240,000, 48,000, 10,080 and 840, total 416,520. The first layer follows equation 1, $5\cdot5\cdot1\cdot6\cdot28^2=117{,}600$. The counts come from hooks on the real layers.
- **AlexNet (the PyTorch layout), parameters:** 61,100,840 in total, of which the five convolutions hold 2,469,696 and the dense part 58,631,144, which is 96.0%. So the original paper's "about 60 million" (read on the NeurIPS page) is matched. Careful: PyTorch ships a single-GPU variant with 64 filters in the first layer, while the paper's two-GPU design had 96 filters split over two GPUs (from memory). So these are the numbers of the PyTorch layout, not exactly the 2012 network.
- **AlexNet, multiply-adds:** 714,188,480 in total, 655,566,528 in the five convolutions (91.8%) and 58,621,952 in the three dense layers. So the *parameters* are mostly in the dense layers, but the *work* is mostly in the convolutions.
- **What sharing saves:** if the first layer of each network were a dense layer from every pixel to every output number, LeNet's would need 4,821,600 numbers against the convolution's 156 (30,908 times more) and AlexNet's would need 29,142,414,400 against 23,296 (1,250,962 times more). See equation 2.
- **LeNet against an MLP:** a plain network 32×32 → 120 → 84 → 10 holds 134,014 numbers, more than twice the 61,706 of the LeNet-5 shaped CNN.
- **Facts I read** (Wikipedia's articles on LeNet and AlexNet and the NeurIPS page of the AlexNet paper, not the original papers): LeNet-5 was published in 1998 by LeCun, Bottou, Bengio and Haffner. An earlier network of 1989 had 3 hidden layers and about 9,760 free parameters and read zip codes for the US Postal Service. A LeNet-based system was in bank check-reading machines from June 1996 and by 2001 was estimated to read 20 million checks a day. AlexNet won the 2012 ImageNet contest with a top-5 error of 15.3%, with the runner-up about 26%. It had 8 layers (5 convolution, 3 dense), about 60 million parameters, was trained on 2 GTX 580 GPUs for 5 to 6 days (90 passes over the data), and used ReLU and dropout with probability 0.5. The paper's abstract gives about 500,000 neurons and, for the 2010 test set, top-1 and top-5 error rates of 39.7% and 18.9% (as printed on the NeurIPS page I read; I have a faint memory that the paper's body gives other numbers for the same test, which I did not check).
- **Limits:** the code only counts. I did not train either network or test accuracy and I did not measure speed. I did not reproduce the training tricks (ReLU, dropout, augmentation) or the 2012 data set.

The code covers four things: (1) LeNet-5 shaped parameters and multiply-adds, (2) AlexNet parameters and multiply-adds, (3) the dense-layer cost of each first layer, (4) an MLP of a similar shape.

```python
import torch
import torch.nn as nn
import torchvision
npar = lambda m: sum(p.numel() for p in m.parameters())
def macs(model, shape):                                            # multiply-adds of every conv and linear layer, counted with hooks
    tot, per = [0], []
    def hook(m, i, o):
        c = o.numel() * m.weight[0].numel() if isinstance(m, nn.Conv2d) else m.in_features * m.out_features
        tot[0] += c; per.append((m.__class__.__name__, c))
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    model.eval(); model(torch.zeros(1, *shape)); [h.remove() for h in hs]; return tot[0], per
# 1) LeNet-5 shaped network (1998 layout, modern pieces: average pooling, plain linear output)
lenet = nn.Sequential(nn.Conv2d(1, 6, 5), nn.Tanh(), nn.AvgPool2d(2), nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2), nn.Conv2d(16, 120, 5), nn.Tanh(), nn.Flatten(), nn.Linear(120, 84), nn.Tanh(), nn.Linear(84, 10))
print('LeNet-5 shaped: layer numbers', [npar(m) for m in lenet if npar(m)], '| total', npar(lenet))
tl, pl = macs(lenet, (1, 32, 32)); print('LeNet multiply-adds per layer', [c for _, c in pl], '| total', tl, '| check C1 = 28*28*6*25 =', 28 * 28 * 6 * 25)
# 2) the AlexNet layout as PyTorch ships it (random weights, no download); it is a one-GPU variant of the paper's two-GPU design
alex = torchvision.models.alexnet(weights=None)
conv = sum(npar(m) for m in alex.features); fc = sum(npar(m) for m in alex.classifier)
print('AlexNet (torchvision layout): total', npar(alex), '| convolution part', conv, '| fully connected part', fc, '| share in the dense layers', round(fc / npar(alex), 4))
ta, pa = macs(alex, (3, 224, 224)); print('AlexNet multiply-adds', ta, '| in the 5 convolutions', sum(c for k, c in pa if k == 'Conv2d'), '| in the 3 dense layers', sum(c for k, c in pa if k == 'Linear'))
# 3) what the first layer would cost if it were a plain dense layer from every pixel to every output number
dense = lambda i, o: (i + 1) * o
c1 = lenet[0]; a1 = alex.features[0]
print('LeNet C1: convolution', npar(c1), 'vs dense', dense(32 * 32, 6 * 28 * 28), '| ratio', round(dense(32 * 32, 6 * 28 * 28) / npar(c1)))
print('AlexNet conv1: convolution', npar(a1), 'vs dense', dense(3 * 224 * 224, 64 * 55 * 55), '| ratio', round(dense(3 * 224 * 224, 64 * 55 * 55) / npar(a1)))
# 4) a dense (MLP) network of similar size on the same inputs would also need weights for every pixel
print('MLP 32x32 -> 120 -> 84 -> 10 numbers:', npar(nn.Sequential(nn.Linear(1024, 120), nn.Linear(120, 84), nn.Linear(84, 10))), '| LeNet-5 shaped:', npar(lenet))
```
Reading the output (Python 3.9.6, PyTorch 2.8.0, torchvision 0.23.0):
- **LeNet:** layer sizes `[156, 2416, 48120, 10164, 850]`, total 61,706. Multiply-adds `[117600, 240000, 48000, 10080, 840]`, total 416,520, and the first layer checks as 117,600.
- **AlexNet:** total 61,100,840, convolution part 2,469,696, dense part 58,631,144, a 0.9596 share. Multiply-adds 714,188,480, of which 655,566,528 in convolutions and 58,621,952 in dense layers.
- **First-layer cost:** LeNet 156 against 4,821,600 (ratio 30,908). AlexNet 23,296 against 29,142,414,400 (ratio 1,250,962).
- **MLP:** 134,014 against 61,706.
- **Limits:** nothing is trained, timed or downloaded.

## 🔴 4. Deep / formal

**How much work does one convolution layer do?** Count one multiply-add for every weight times every place it is used. With a $k\times k$ kernel, $C_{\text{in}}$ input channels, $C_{\text{out}}$ filters and output maps of size $o\times o$ (bias not counted):

$$M=k^{2}\,C_{\text{in}}\,C_{\text{out}}\,o^{2}$$

**Where:**
- $k^2C_{\text{in}}$ is the number of weights in one filter. **Matters:** it is how many multiplications make one output number.
- $C_{\text{out}}$ is the number of filters. **Matters:** each one produces its own map, so the work grows in proportion.
- $o^2$ is the number of output positions per map. **Matters:** the same filter is applied at every position, so work grows with the area of the map. This is also why weights are few and work is large: the same weights are used $o^2$ times.
- $M$ is the number of multiply-adds for the layer, for one picture. **Matters:** it is a count of arithmetic, not a time. Real speed also depends on memory and hardware, which I did not measure.

**Example (tiny numbers):** LeNet's first layer, $5\times5$ kernel, 1 input channel, 6 filters, output $28\times28$.
1. $k^2C_{\text{in}}=25\cdot1=25$ weights per filter.
2. $M=25\cdot6\cdot28^2=150\cdot784=117{,}600$.
3. The code's hook count for this layer was 117,600 as well, and the same formula was used to check it.
4. Meaning: the layer has only 156 numbers (150 weights and 6 biases) but does 117,600 multiply-adds, since every weight is reused 784 times.

**How much does weight sharing save?** Compare the convolution layer with a dense layer that makes the same output numbers from the same input numbers. Input $H\times W\times C_{\text{in}}$, output $C_{\text{out}}$ maps of $o\times o$:

$$\text{ratio}=\frac{(H\,W\,C_{\text{in}}+1)\,C_{\text{out}}\,o^{2}}{k^{2}\,C_{\text{in}}\,C_{\text{out}}+C_{\text{out}}}$$

**Where:**
- The numerator is the size of a dense layer from $HWC_{\text{in}}$ numbers to $C_{\text{out}}o^2$ numbers (with one bias each). **Matters:** every output number gets its own weight for every input number.
- The denominator is the size of the convolution: the weights of $C_{\text{out}}$ filters plus their biases. **Matters:** the filters are shared across all positions, so the count does not depend on the picture size at all.
- $H$ and $W$ are the picture's height and width. **Matters:** the ratio grows with the picture's area (and with $o^2$), which is why the saving is huge for big photos.
- The ratio. **Matters:** it says how many times more numbers the dense version would need. It is not a claim about accuracy: a dense layer could in principle represent more functions, it is just far more expensive and learns worse from limited data.

**Example (tiny numbers):** LeNet's first layer: $H=W=32$, $C_{\text{in}}=1$, $C_{\text{out}}=6$, $o=28$, $k=5$.
1. Dense: $(32\cdot32\cdot1+1)\cdot6\cdot28^2=1025\cdot4704=4{,}821{,}600$.
2. Convolution: $25\cdot1\cdot6+6=156$.
3. Ratio: $4{,}821{,}600/156\approx30{,}908$, as the code printed.
4. Meaning: sliding one small filter instead of wiring every pixel to every output number makes the layer about thirty thousand times smaller. For AlexNet's first layer the same calculation gave 23,296 against 29,142,414,400, a ratio of about 1.25 million.

**What came before, and how this improved on it:**
- **Before (LeNet):** hand-designed features fed to a separate classifier, and fully connected networks on pixels. In the late 1980s the same group had already built smaller CNNs for zip-code digits (Wikipedia: 1989, about 9,760 parameters).
- **Its limit:** the fully connected approach needs a weight for every pixel (equation 2), and hand-made features need a designer for every task. In this note's code, a plain MLP of LeNet's shape holds 134,014 numbers against LeNet's 61,706.
- **What changed (LeNet):** share small learned filters over the picture, and shrink the maps with pooling. This made it small enough to train and good enough to deploy in cheque-reading machines (June 1996 and later, per Wikipedia, which I read).
- **Before (AlexNet):** by the early 2010s CNNs were known but had not won on big photo sets; the leading methods used hand-made features (from memory; I did not check which ones won earlier contests) and I did not look up their scores, so I do not quote them here.
- **What changed (AlexNet):** much bigger (about 61 million numbers in my count), trained on 2 GPUs, with ReLU, dropout and augmentation, on a data set of over a million labelled photos.
- **How much better:** the 2012 contest result was a top-5 error of 15.3% against about 26% for the runner-up (Wikipedia, which I read; I did not open the contest report). In this note's own code, the improvement from sharing is counted: 156 against 4.8 million numbers in LeNet's first layer, and 23,296 against about 29 billion in AlexNet's.

**Other points (from memory, not checked here):**
- Later networks cut AlexNet's huge dense part. VGG (2014) used many small 3×3 filters, and GoogLeNet and ResNet used global average pooling and so almost no dense layers (see [[ResNet]] and [[Pooling]]). I did not check the dates.
- The ImageNet contest, which AlexNet won, ran every year from 2010. I did not look up the details.
- Local response normalisation was part of AlexNet but is rarely used now, and [[Normalization]] (BatchNorm) replaced it. I did not test it.
- The 2012 result is widely seen as the start of the modern deep learning boom, though the exact reasons (data, GPUs, tricks) are still discussed.

## Related
[[Convolutional Neural Network]] · [[Convolution]] · [[Pooling]] · [[Feature Map]] · [[Image Classification]] · [[Activation Function]] · [[Dropout]] · [[Data Augmentation]] · [[Transfer Learning]] · [[ResNet]] · [[GPU]] · [[Normalization]] · [[Backpropagation]] · [[PyTorch]] · [[AlexNet 2012]] · [[History of Image Classification]]
- [[VGG]]

## Sources
- The Wikipedia articles on LeNet and AlexNet (read through a web fetch) for the dates, the 1989 and 1996 facts, the 15.3% result and the training setup.
- Krizhevsky, Sutskever and Hinton, *ImageNet Classification with Deep Convolutional Neural Networks* (NeurIPS 2012), abstract page read for the 60 million parameters, 500,000 neurons, 5 convolution and 3 dense layers and the 39.7% and 18.9% figures. I did not read the paper's body.
- LeCun, Bottou, Bengio and Haffner, *Gradient-Based Learning Applied to Document Recognition* (1998), named from memory; I did not open it.
- Library versions in this run: Python 3.9.6, PyTorch 2.8.0, torchvision 0.23.0 (CPU). The code and every printed value were run for this note. Nothing was trained, timed or downloaded.
- Figures: [LeNet-5_architecture.svg](https://commons.wikimedia.org/wiki/File:LeNet-5_architecture.svg) (Zhang, Lipton, Li and Smola, CC BY-SA 4.0) and [AlexNet_architecture.png](https://commons.wikimedia.org/wiki/File:AlexNet_architecture.png) (Daniel Voigt Godoy, CC BY 4.0), on Wikimedia Commons. I viewed both before writing the captions.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision LeNet:::A small convolution network, small and clever, good enough to read digits on cheques.

#flashcards/ml/reverse #flashcards/topic/vision AlexNet:::Years later the same idea with a library of millions of photos, a graphics chip (GPU) as a fast helper, and a much bigger network.

#flashcards/ml/formula #flashcards/topic/vision Write the multiply-adds of a convolution layer and explain each symbol.
?
$$M=k^{2}\,C_{\text{in}}\,C_{\text{out}}\,o^{2}$$
- $k^2C_{\text{in}}$: the number of weights in one filter, which is also how many multiplications make one output number.
- $C_{\text{out}}$: the number of filters. Each produces its own map, so work grows in proportion.
- $o^2$: the number of output positions per map. The same filter is applied at every position, so work grows with the map's area. This is why weights are few and work is large.
- $M$: the number of multiply-adds for the layer, for one picture.

#flashcards/ml/example #flashcards/topic/vision LeNet's first layer: $5\times5$ kernel, 1 input channel, 6 filters, $28\times28$ output. What is $M$ and how many parameters?::$k^2C_{\text{in}}=25$ weights per filter. $M=25\cdot6\cdot28^2=117{,}600$ (the note's code counted the same). The layer has only 156 numbers (150 weights and 6 biases) but does 117,600 multiply-adds, since every weight is reused 784 times.

#flashcards/ml/formula #flashcards/topic/vision Write the ratio of a dense layer to a convolution layer and explain each symbol.
?
$$\text{ratio}=\frac{(H\,W\,C_{\text{in}}+1)\,C_{\text{out}}\,o^{2}}{k^{2}\,C_{\text{in}}\,C_{\text{out}}+C_{\text{out}}}$$
- Numerator: the size of a dense layer from $HWC_{\text{in}}$ numbers to $C_{\text{out}}o^2$ numbers, with one bias each. Every output gets its own weight for every input.
- Denominator: the size of the convolution, the weights of $C_{\text{out}}$ filters plus their biases. The filters are shared across all positions, so the count does not depend on the picture size.
- $H$, $W$: the picture's height and width. The ratio grows with the picture's area, which is why the saving is huge for big pictures.

#flashcards/ml/example #flashcards/topic/vision LeNet's first layer: $H=W=32$, $C_{\text{in}}=1$, $C_{\text{out}}=6$, $o=28$, $k=5$. What is the ratio?::Dense: $(32\cdot32+1)\cdot6\cdot28^2=4{,}821{,}600$. Convolution: $25\cdot6+6=156$. Ratio about 30,908. Sliding one small filter makes the layer about thirty thousand times smaller. For AlexNet's first layer the note's code gave 23,296 against 29,142,414,400, a ratio of about 1.25 million.

#flashcards/ml/image #flashcards/topic/vision ![[lenet-5-architecture.svg]]
What does this picture show?
?
A LeNet-5 style network with a 28×28 input. Convolution makes 6 maps of 28×28, pooling makes them 14×14, a second convolution makes 16 maps of 10×10, pooling gives 16 maps of 5×5, then dense layers of 120, 84 and 10 units. (The original 1998 network used a 32×32 input.) (From the note's caption: Aston Zhang, Lipton, Li and Smola, Wikimedia Commons.)

#flashcards/ml/image #flashcards/topic/vision ![[alexnet-architecture.png]]
What does this picture show?
?
The AlexNet layout as a chain of blocks: five convolution stages (grey boxes, shrinking in width and growing in depth), then the classifier of dense layers ending in the original 1,000-way output (upper row). The lower row is the same network with its last layer replaced by a new one, as in transfer learning. (From the note's caption: Daniel Voigt Godoy, Wikimedia Commons.)
