---
tags: [resnet, residual-network, cnn, computer-vision, deep-learning, imagenet, ML]
status: growing
level: 4
tier: 3
---

# ResNet

**ResNet** (short for *residual network*, 2015) is a [[Convolutional Neural Network]] built from blocks that each add their input to what their layers compute (a [[Residual Connection]]). That small trick made it possible to train networks with 50, 101 and even 152 layers, many times deeper than the networks before it, and it won the 2015 ImageNet contest. Nearly every modern vision network, and the [[Transformer]] too, uses the same "keep the input and add to it" idea.

## 🟢 1. Simple (age 12)
Imagine passing a message along a line of 50 people by whispering. By the end, the message is garbled, since every person makes a small mistake. Now give every person a copy of the original note to hold up next to their whisper, so the last person can always compare with the original and only needs to hear the *changes*. The message survives the whole line. ResNet is a very long chain of "layers" where each one is handed the original and only has to say what to change.

**Example:** A network is 50 layers deep. Each layer is allowed to do almost nothing (just pass the picture on), so a deep network is never *worse* than a shallow one: extra layers can only add useful changes.

## 🔵 2. How it works
1. **Start like any CNN.** A first [[Convolution]] and [[Pooling]] step turn the picture into small [[Feature Map]]s.
2. **Stack residual blocks.** A block runs the maps through a couple of convolution layers and *adds the block's input back* to the result: output = input + change.
3. **Keep the input when sizes match.** If a block changes the number of maps or the size, a small 1×1 convolution on the shortcut makes the input fit so that it can still be added.
4. **Use "bottleneck" blocks for deep versions.** A bottleneck block first squeezes many maps down to fewer with a 1×1 convolution, does the 3×3 convolution on those few, then expands back with another 1×1 convolution. It does about the same job with far fewer numbers.
5. **Normalise and activate.** Each convolution is followed by [[Normalization]] (BatchNorm) and an [[Activation Function]], which is part of why such deep networks train (from memory of the paper; I did not test it here).
6. **End with averaging.** The last maps are averaged to one number each ([[Pooling]], "global average pooling") and one dense layer gives the labels, so almost no numbers are spent on dense layers.
7. **Gradients flow.** Because every block adds its input, the error signal in [[Backpropagation]] has a direct path back through the whole network and does not fade away ([[Vanishing and Exploding Gradients]]).

![[residual-block.png]]
*A residual block. The input $x$ goes down through two layers, which together compute $F(x)$. At the same time, $x$ is copied along the curved "identity" path on the right, and the two are added (the circled plus). The output is $x+F(x)$. Image: LunarLullaby, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:ResBlock.png), CC BY-SA 4.0. (It is the same picture as in [[Residual Connection]].)*

## 🟠 3. Technical detail
All numbers come from the code at the end of this section (Python 3.9.6, NumPy 1.24.4, scikit-learn 1.1.2, PyTorch 2.8.0 and torchvision 0.23.0 on a CPU). The networks from torchvision are built with random weights, so nothing is downloaded and nothing is trained, except the small toy at the end.

- **Depth counts weight layers:** the name ResNet-N means N layers with weights on the main path (the convolutions and the last dense layer, not the 1×1 shortcut convolutions). The code counted 18, 34, 50, 101 and 152 in the five torchvision models, matching their names. Equation 1 gives the same from the number of blocks.
- **Size of the models:** ResNet-18 holds 11,689,512 numbers, ResNet-34 21,797,672, ResNet-50 25,557,032, ResNet-101 44,549,160 and ResNet-152 60,192,808. For comparison, the AlexNet layout in [[LeNet and AlexNet]] holds 61,100,840. So ResNet-152 is 152 layers deep with a bit *fewer* numbers than the 8-layer AlexNet, because it has no big dense layers.
- **The bottleneck:** on 256 channels, two plain 3×3 layers hold 1,179,648 numbers, while the bottleneck 256 → 64 → 64 → 256 (1×1, 3×3, 1×1) holds 69,632. That is 16.94 times fewer (equation 2).
- **A toy "degradation" test:** I built small networks of dense layers (width 32, ReLU) on the 8×8 digits (first 1,000 pictures, 200 Adam steps, one seed), with and without a shortcut around each hidden layer. With 2 hidden layers the plain network trained to loss 0.0069 and accuracy 1.0. With 20 plain hidden layers, the size of the first layer's gradient at the start was only 3.337e-09, and after 200 steps the loss was still 2.3024 (which is $\ln10$, "knows nothing") with accuracy 0.104. With 40 plain layers the starting gradient was 1.513e-17 and the result was the same. With shortcuts, 20 hidden layers reached loss 0.0019 and accuracy 1.0. With shortcuts and 40 layers, training was less good: loss 0.7045 and accuracy 0.756, and the starting gradient was large (4.056e+03), because with no normalisation the added shortcuts make the signal grow.
- **What the toy does and does not show:** it shows that with PyTorch's default starting numbers, a deep plain stack can fail to train at all, and shortcuts fix it at 20 layers. It is *not* the same as the "degradation problem" in the ResNet paper, where deeper plain networks with BatchNorm trained but reached *worse* training error than shallower ones (that description is from memory; I did not run it). The 40-layer shortcut result also shows shortcuts alone are not enough without normalisation or care with scale.
- **Facts I read** (the arXiv abstract page of the paper, 1512.03385, submitted on 10 December 2015): networks up to 152 layers on ImageNet, "8 times deeper than VGG nets but still having lower complexity"; an ensemble reached 3.57% error on the ImageNet test set and won first place in the ILSVRC 2015 classification task; the authors also analyse CIFAR-10 with 100 and 1000 layers.
- **Limits:** I trained nothing on photographs, tested no accuracy of the real ResNets and measured no speed. The toy is one seed on tiny data. I did not test BatchNorm, other initialisations, or the 1×1 shortcut convolutions.

The code covers three things: (1) the depth and size of five real ResNet layouts, (2) the bottleneck against two plain 3×3 layers, (3) deep plain against residual toy networks.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.datasets import load_digits
torch.set_num_threads(1)
npar = lambda m: sum(p.numel() for p in m.parameters())
# 1) the named depth counts the weight layers on the main path: convolutions and the final linear layer, not the 1x1 shortcut projections
for name, blocks, per in (('resnet18', (2, 2, 2, 2), 2), ('resnet34', (3, 4, 6, 3), 2), ('resnet50', (3, 4, 6, 3), 3), ('resnet101', (3, 4, 23, 3), 3), ('resnet152', (3, 8, 36, 3), 3)):
    m = getattr(torchvision.models, name)(weights=None)
    L = sum(isinstance(mod, (nn.Conv2d, nn.Linear)) and 'downsample' not in n for n, mod in m.named_modules())
    print(name, '| weight layers counted:', L, '| parameters:', npar(m), '| 2 +', per, '* blocks', sum(blocks), '=', 2 + per * sum(blocks))
# 2) the bottleneck block: squeeze 256 channels to 64, filter, expand back, instead of two 3x3 filters on all 256
conv = lambda a, b, k: nn.Conv2d(a, b, k, bias=False)
two3 = nn.Sequential(conv(256, 256, 3), conv(256, 256, 3)); neck = nn.Sequential(conv(256, 64, 1), conv(64, 64, 3), conv(64, 256, 1))
print('two 3x3 layers on 256 channels:', npar(two3), '| bottleneck 256 -> 64 -> 64 -> 256:', npar(neck), '| ratio', round(npar(two3) / npar(neck), 2), '| formula 256*64 + 9*64*64 + 64*256 =', 256 * 64 + 9 * 64 * 64 + 64 * 256)
# 3) the degradation problem on a toy: deeper plain networks train worse; the same depth with shortcuts trains fine
D = load_digits(); X = torch.tensor(D.data / 16.0, dtype=torch.float32)[:1000]; y = torch.tensor(D.target.astype(np.int64))[:1000]
class Plain(nn.Module):
    def __init__(s, depth, skip, w=32):
        super().__init__(); s.inp = nn.Linear(64, w); s.mid = nn.ModuleList(nn.Linear(w, w) for _ in range(depth)); s.out = nn.Linear(w, 10); s.skip = skip
    def forward(s, x):
        x = F.relu(s.inp(x))
        for l in s.mid: x = x + F.relu(l(x)) if s.skip else F.relu(l(x))
        return s.out(x)
def go(depth, skip, seed=0):
    torch.manual_seed(seed); m = Plain(depth, skip); opt = torch.optim.Adam(m.parameters(), 0.003)
    loss0 = F.cross_entropy(m(X), y); m.zero_grad(); loss0.backward(); g = m.inp.weight.grad.norm().item()
    for _ in range(200): opt.zero_grad(); l = F.cross_entropy(m(X), y); l.backward(); opt.step()
    return format(g, '.3e'), round(l.item(), 4), round(float((m(X).argmax(1) == y).float().mean()), 4)
for depth, skip in ((2, False), (20, False), (20, True), (40, False), (40, True)):
    print('hidden layers', depth, '| shortcuts' if skip else '| plain    ', '| first-layer gradient size at start, training loss, training accuracy after 200 steps:', go(depth, skip))
```
Reading the output (Python 3.9.6, PyTorch 2.8.0, torchvision 0.23.0):
- **Depth:** resnet18, 34, 50, 101 and 152 counted 18, 34, 50, 101 and 152 weight layers, and $2+2\cdot8$, $2+2\cdot16$, $2+3\cdot16$, $2+3\cdot33$ and $2+3\cdot50$ give the same.
- **Size:** 11,689,512 · 21,797,672 · 25,557,032 · 44,549,160 · 60,192,808.
- **Bottleneck:** 1,179,648 against 69,632, ratio 16.94, equal to the formula's 69,632.
- **Toy:** 2 plain layers (gradient 3.451e-02, loss 0.0069, accuracy 1.0); 20 plain (3.337e-09, 2.3024, 0.104); 20 with shortcuts (2.012e+01, 0.0019, 1.0); 40 plain (1.513e-17, 2.3024, 0.104); 40 with shortcuts (4.056e+03, 0.7045, 0.756).
- **Limits:** the run takes about a minute and nothing is timed.

## 🔴 4. Deep / formal

**How deep is a ResNet?** The network has a first convolution, a last dense layer, and $B$ residual blocks in between. A basic block has $k=2$ weight layers and a bottleneck block has $k=3$. Counting only the layers on the main path:

$$L=2+k\sum_{i}n_i$$

**Where:**
- $n_i$ is the number of blocks in stage $i$ (ResNet-50 has four stages, with 3, 4, 6 and 3 blocks). **Matters:** these numbers are the whole design. Changing them gives ResNet-101 (3, 4, 23, 3) or ResNet-152 (3, 8, 36, 3).
- $k$ is the number of weight layers per block, 2 for the basic block and 3 for the bottleneck. **Matters:** a bottleneck block is deeper than a basic block, so the same $L$ needs fewer blocks.
- The $2$ is the first convolution plus the final dense layer. **Matters:** without it the name would be off by two.
- $L$ is the depth in the model's name. **Matters:** it is the count of weight layers on the main path and ignores the 1×1 convolutions on the shortcuts.

**Example (tiny numbers):** ResNet-50, with stages of 3, 4, 6 and 3 blocks and bottleneck blocks ($k=3$).
1. $\sum n_i=3+4+6+3=16$ blocks.
2. $L=2+3\cdot16=50$.
3. The code counted 50 weight layers in the torchvision ResNet-50, and likewise 101 for $2+3\cdot33$ and 152 for $2+3\cdot50$.
4. Meaning: going from ResNet-50 to ResNet-152 does not change the design, it only stacks many more blocks in the third stage (6 → 36).

**How much does the bottleneck save?** With $C_{\text{in}}$ channels coming in, a squeeze to $c$ channels, and $C_{\text{out}}$ going out, the three convolutions (1×1, 3×3, 1×1, no bias) hold:

$$P_{\text{neck}}=C_{\text{in}}\,c+9\,c^{2}+c\,C_{\text{out}}$$

**Where:**
- $C_{\text{in}}c$ is the first 1×1 convolution that squeezes $C_{\text{in}}$ maps down to $c$. **Matters:** the squeeze makes the next, expensive layer cheap.
- $9c^2$ is the 3×3 convolution on $c$ maps ($3\cdot3\cdot c\cdot c$). **Matters:** this is the biggest term, and it uses $c$, the *small* number, instead of the big $C$.
- $cC_{\text{out}}$ is the last 1×1 convolution that expands back. **Matters:** it restores the width so the block's output can be added to its input.
- $P_{\text{neck}}$ is the number of learned numbers in the block. **Matters:** compare it with two plain 3×3 layers on the full width, $2\cdot9C^2$, which is what the older blocks used.

**Example (tiny numbers):** $C_{\text{in}}=C_{\text{out}}=256$ and $c=64$.
1. $256\cdot64=16{,}384$ and $9\cdot64^2=36{,}864$ and $64\cdot256=16{,}384$.
2. $P_{\text{neck}}=16{,}384+36{,}864+16{,}384=69{,}632$.
3. Two plain 3×3 layers on 256 channels: $2\cdot9\cdot256^2=1{,}179{,}648$. The code counted both with PyTorch (69,632 and 1,179,648).
4. Meaning: $1{,}179{,}648/69{,}632=16.94$, so the bottleneck block is almost seventeen times smaller for the same input and output width.

**What came before, and how this improved on it:**
- **Before:** "plain" deep CNNs, such as the AlexNet of 2012 (8 layers) and its successors that went to about 16 to 19 layers (VGG, 2014; from memory, I did not check the layer counts or dates). Gains from adding layers had begun to stall.
- **Its limit:** very deep plain stacks became *harder to train*: signals and gradients fade through many layers, and in the ResNet paper (from memory of its introduction; I read only the abstract) deeper plain networks even reached worse *training* error than shallower ones. In this note's toy, 20 plain hidden layers did not train at all (loss stuck at 2.3024).
- **What changed:** each block adds its input to its output, so a block that has nothing useful to add can pass the input on unchanged, and the gradient has a direct path through the whole network. Bottleneck blocks and global average pooling kept the numbers small.
- **How much better:** in the toy, 20 hidden layers with shortcuts reached loss 0.0019 (accuracy 1.0) where the plain version stayed at 2.3024 (accuracy 0.104). In the counts, ResNet-152 has 152 layers with 60,192,808 numbers, fewer than AlexNet's 61,100,840 for 8 layers, and the bottleneck is 16.94 times smaller than two plain 3×3 layers. In the published result (read on the arXiv page) an ensemble reached 3.57% error on the ImageNet test set and won ILSVRC 2015. For comparison, AlexNet's top-5 error in 2012 was 15.3% (from the Wikipedia article I read, see [[LeNet and AlexNet]]); I assume both are top-5 errors on the same kind of test, but I did not check the contest definitions.

**Other points (from memory, not checked here):**
- Later variants include wider and grouped versions (ResNeXt), versions with the activation moved ("pre-activation"), and DenseNet (which connects every layer to all earlier ones). I did not check any of them.
- The shortcut idea reappears in the [[Transformer]], where each attention and feed-forward block is wrapped in a residual connection.
- ResNet is the usual backbone for [[Transfer Learning]] in vision, with pretrained weights in libraries such as torchvision. I downloaded no weights here.
- Why very deep residual networks generalise so well is still discussed. One common view (not tested here) is that they behave like a collection of shallower paths.

## Related
[[Residual Connection]] · [[Convolutional Neural Network]] · [[Convolution]] · [[Pooling]] · [[Normalization]] · [[Activation Function]] · [[Vanishing and Exploding Gradients]] · [[Backpropagation]] · [[Transformer]] · [[Transfer Learning]] · [[LeNet and AlexNet]] · [[Image Classification]] · [[PyTorch]] · [[ResNet 2015]] · [[History of Image Classification]] · [[Transfer Learning vs Vanilla CNN]]
- [[VGG]]
- [[MobileNet]]
## Sources
- He, Zhang, Ren and Sun, *Deep Residual Learning for Image Recognition* (arXiv 1512.03385, submitted 10 December 2015), abstract page read through a web fetch for the 152 layers, the 3.57% result, the ILSVRC 2015 win and the CIFAR-10 analysis. I did not read the paper's body.
- The Wikipedia article on AlexNet (read through a web fetch) for the 15.3% figure.
- Library versions in this run: Python 3.9.6, NumPy 1.24.4, scikit-learn 1.1.2, PyTorch 2.8.0, torchvision 0.23.0 (CPU). The code and every printed value were run for this note. Nothing was timed, and no pretrained weights were downloaded.
- Figure: [ResBlock.png](https://commons.wikimedia.org/wiki/File:ResBlock.png) (LunarLullaby, CC BY-SA 4.0), on Wikimedia Commons, the same file already used in [[Residual Connection]].

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision ResNet:::A deep image network built from residual blocks. Every block holds up a copy of its input next to its work, so the last layer can always compare with the original and only needs to learn the changes.

#flashcards/ml/formula #flashcards/topic/vision Write the depth of a ResNet and explain each symbol.
?
$$L=2+k\sum_{i}n_i$$
- $n_i$: the number of blocks in stage $i$. ResNet-50 has four stages with 3, 4, 6 and 3 blocks. These numbers are the whole design: ResNet-101 is (3, 4, 23, 3), ResNet-152 is (3, 8, 36, 3).
- $k$: the number of weight layers per block, 2 for the basic block and 3 for the bottleneck.
- The 2 is the first convolution and the final fully connected layer.

#flashcards/ml/formula #flashcards/topic/vision Write the parameters of a bottleneck block and explain each symbol.
?
$$P_{\text{neck}}=C_{\text{in}}\,c+9\,c^{2}+c\,C_{\text{out}}$$
- $C_{\text{in}}c$: the first 1×1 convolution that squeezes $C_{\text{in}}$ maps down to $c$. The squeeze makes the next, expensive layer cheap.
- $9c^2$: the 3×3 convolution on $c$ maps ($3\cdot3\cdot c\cdot c$). It is the biggest term, but it uses the small number $c$ instead of the big $C$.
- $cC_{\text{out}}$: the last 1×1 convolution that expands back to $C_{\text{out}}$ maps.

#flashcards/ml/cloze #flashcards/topic/vision ResNet-50 has ==16== blocks of three layers, giving ==50== weight layers.

#flashcards/ml/example #flashcards/topic/vision For ResNet-50 (stages 3, 4, 6, 3 with bottleneck blocks), what is $L$?::Σn = 3 + 4 + 6 + 3 = 16 blocks. L = 2 + 3 × 16 = 50. The code counted 50 weight layers in torchvision's ResNet-50, and likewise 101 and 152.

#flashcards/ml/example #flashcards/topic/vision $C_{\text{in}}=C_{\text{out}}=256$ and $c=64$. What is $P_{\text{neck}}$, and what do two plain 3×3 layers on 256 channels cost?::256 × 64 = 16,384. 9 × 64² = 36,864. 64 × 256 = 16,384. Total 69,632. Two plain 3×3 layers: 2 × 9 × 256² = 1,179,648. The bottleneck is about 17 times smaller.

#flashcards/ml/image #flashcards/topic/vision ![[residual-block.png]]
What does this picture show?
?
A residual block. The input x goes down through two layers, which together compute F(x). At the same time x is copied along the curved "identity" path on the right, and the two are added (the circled plus). The output is x + F(x).
