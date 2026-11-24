---
tags: [skip-connection, residual-connection, concatenation, ResNet, U-Net, deep-learning, neural-network, computer-vision, ML]
status: growing
level: 4
tier: 2
---

# Skip Connection

A **skip connection** is a shortcut wire that lets a layer's input jump over one or more layers and join the result further on. There are two common ways to join: **add** the two (this is the [[Residual Connection]] of [[ResNet]] and of the [[Transformer]]) or **glue them side by side** along the channel axis, called **concatenation** (used by [[U-Net 2015|U-Net]]). This note compares the two and says when to use which.

## 🟢 1. Simple (age 12)
Imagine passing a message down a long line of 50 people. By the end it is garbled. Now give person 1 a second copy of the message and hand it straight to person 50, who can compare it with what arrived. The shortcut means the original can never be fully lost.

**Example:** You redraw a photo ten times by hand, each time from the previous copy, and it gets blurrier. If you keep the original photo on the table and glance at it each time, you can always fix the blur.

## 🔵 2. How it works
1. **A normal layer stack** passes data from layer 1 to layer 2 to layer 3. Everything must go through every layer.
2. **A skip connection** copies the data *before* some layers and brings that copy back *after* them.
3. **Way one: add.** The copy and the layers' result have the same shape, and they are added number by number. The layers only have to learn the *change* (the residual). If the best change is "nothing", the layers just output zeros.
4. **Way two: concatenate.** The copy is placed next to the result, so the next layer sees both. The channel count grows (4 channels plus 4 channels is 8). Nothing is mixed, so the next layer can choose what to use.
5. **Why it helps learning.** When the network learns by [[Backpropagation]], the error travels backwards. Along the shortcut it travels without being shrunk by each layer, so even the first layers get a clear signal (see [[Vanishing and Exploding Gradients]]).
6. **Why it helps detail.** In a [[U-Net 2015|U-Net]] for [[Image Segmentation]], the shrinking half throws away exact positions. The copied maps from the shrinking half give them back to the growing half.

![[residual-block.png]]
*An additive skip (a residual block). The input $x$ goes through two layers, which compute $F(x)$, and along the curved path unchanged; the circled plus adds them, so the output is $x+F(x)$. Image: LunarLullaby, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:ResBlock.png), CC BY-SA 4.0 (the same image as in [[Residual Connection]]).*

![[unet-architecture.png]]
*A concatenating skip. In this simplified U-Net the horizontal arrows copy the left (shrinking) maps across to the right (growing) side, where they are concatenated with the upsampled maps. Figure: Mehrdad Yazdani, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Example_architecture_of_U-Net_for_producing_k_256-by-256_image_masks_for_a_256-by-256_RGB_image.png), CC BY-SA 4.0. It is a redrawing of the idea, not the paper's own Figure 1 (see [[U-Net 2015]] for the differences).*

## 🟠 3. Technical detail
All numbers below come from the code at the end of this section (Python 3.9.6, PyTorch 2.8.0, CPU, fixed seeds).

- **Add needs the same shape; concatenate does not.** Adding a 4-channel map to a 6-channel map raised a size error. Concatenating two 4-channel maps gave 8 channels, and the first 4 channels were exactly the first input.
- **Add costs no numbers and no extra width.** A concatenation doubles the channels the next layer must read, so its weights double. The U-Net bottom-right step has 512 channels after up-convolution and 512 copied ones, so 1024 enter the next 3 by 3 convolution (channel counts from the paper, as in my [[U-Net 2015]] note). Producing 512 channels from 1024 takes 4,719,104 numbers, against 2,359,808 if the same layer only read 512 (my run, weights plus biases).
- **Residual start:** a block whose layers output zero returns its input exactly (`True` in the run). So a very deep stack can start out as "do nothing" and learn useful changes.
- **Gradient at the input of a deep stack** (width 32, `tanh` layers with small random weights, one random input, gradient of the sum of the output):

| Depth | Plain stack | With additive skips |
|---|---|---|
| 5 | 1.385e-01 | 8.212 |
| 20 | 4.710e-06 | 14.07 |
| 50 | 2.893e-15 | 31.79 |

  Table written by me, from my own run. *Depth* is the number of layers or blocks. Each cell is the size (norm) of the gradient reaching the input. In the plain stack it shrinks by a factor of about $10^{14}$ from depth 5 to depth 50. With skips it did not shrink; it *grew* (31.79 at depth 50), which shows the opposite risk: sums of many skip paths can make gradients large, which is one reason real networks add [[Layer Normalization]] or [[Batch Normalization]].
- **Limits:** one toy network, one seed per depth, small random weights chosen to make the plain stack fade. It shows the mechanism, not how much a real ResNet gains. I did not test concatenation's effect on gradients, nor timing or memory (a concatenation also keeps the copied map in memory until it is used).

```python
import torch, torch.nn as nn

def stack(depth, width, skip):
    layers = [nn.Linear(width, width) for _ in range(depth)]
    for l in layers:
        nn.init.normal_(l.weight, std=0.5 / width ** 0.5); nn.init.zeros_(l.bias)
    def f(x):
        for l in layers:
            x = x + torch.tanh(l(x)) if skip else torch.tanh(l(x))   # the only difference: "x +"
        return x
    return f

for depth in (5, 20, 50):
    out = []
    for skip in (False, True):
        torch.manual_seed(1); f = stack(depth, 32, skip)
        x = torch.randn(1, 32, requires_grad=True); f(x).sum().backward()
        out.append(x.grad.norm().item())
    print(f'depth {depth:2d}: plain {out[0]:.3e} | with skip {out[1]:.3e}')

a, b = torch.randn(1, 4, 8, 8), torch.randn(1, 4, 8, 8)
print('add', (a + b).shape, '| concat', torch.cat([a, b], 1).shape)
try: a + torch.randn(1, 6, 8, 8)
except RuntimeError: print('add with different channels fails')
c = lambda i, o: sum(p.numel() for p in nn.Conv2d(i, o, 3).parameters())
print('3x3 conv, 1024 -> 512 channels:', c(1024, 512), '| 512 -> 512:', c(512, 512))
blk = nn.Linear(4, 4); nn.init.zeros_(blk.weight); nn.init.zeros_(blk.bias); x = torch.randn(2, 4)
print('layers output zero -> block returns its input:', torch.equal(x + blk(x), x))
```

## 🔴 4. Deep / formal
**Additive skip (residual block):**

$$
y = x + F(x)
$$

**Where:**
- $x$ is what goes into the block. **Matters:** it is carried over unchanged, so the block never loses it.
- $F(x)$ is what the block's layers compute from $x$. **Matters:** it must have the same shape as $x$; it only has to learn the *change* to make.
- $y$ is the block's output. **Matters:** if $F$ outputs zero, $y=x$ and the block is the identity.

**Example (tiny numbers):** $x=2$ and the layers compute $F(x)=0.5$. Then $y=2+0.5=2.5$. If the layers learned nothing ($F=0$), $y=2$, the input unchanged. The answer means: the block starts from "keep it" and only adds a correction.

**Why the error survives going backwards.** Differentiate the block with respect to its input:

$$
\frac{\partial y}{\partial x} = 1 + \frac{\partial F}{\partial x}
$$

**Where:**
- $\partial y/\partial x$ is the factor by which the error is multiplied when it passes backwards through this block.
- $1$ comes from the shortcut. **Matters:** it never goes to zero, however small $F$'s slope is.
- $\partial F/\partial x$ is the slope of the layers. **Matters:** in a plain network this is the *only* term, so a slope below 1 shrinks the error at every layer.

**Example (tiny numbers):** let the layers have slope 0.1. One block with the skip gives $1+0.1=1.1$; without it, $0.1$. Three blocks: $1.1^3=1.331$ against $0.1^3=0.001$. Twenty blocks: $1.1^{20}=6.727$ against $10^{-20}$. The answer means: the shortcut turns "multiply many small numbers" into "multiply many numbers near 1". (This is for one number per block; real networks use matrices, and the same idea holds for the identity matrix.)

**Concatenation.** Instead of adding, concatenation builds $[x,\,F(x)]$ along the channel axis. No symbol needs a new equation: the channel count is the sum of the two, and the next layer has separate weights for each part, so it learns how much of each to use. The cost is the doubled input width shown above.

**Trade-offs and limits:**
- **Add** mixes the two into one sum, so the next layer cannot tell them apart; it saves weights and memory, and it is what lets very deep stacks (ResNet-152, Transformers) train.
- **Concatenate** keeps both intact, which is why it suits tasks that need exact positions (segmentation); it costs more weights and memory.
- A skip can also carry a *transformed* copy (a 1 by 1 convolution to fix the channel count). ResNet's "projection shortcut" does this (from my [[ResNet]] note, option comparison in the paper).
- I have not covered dense connections (every layer joined to every later one, DenseNet), because I did not read that paper.

**What came before, and how this improved on it:**
- **Before:** plain deep stacks, such as [[VGG]] (2014, 16 to 19 layers), where each layer only sees the one before it.
- **Its limit:** the ResNet paper reports that deeper plain networks became *worse* even on training data (the "degradation problem"; see [[ResNet]]). My toy run above shows the gradient reaching the input falling to $10^{-15}$ at depth 50.
- **What changed:** let the input skip ahead, by adding it (ResNet, 2015) or by gluing it on (U-Net, 2015).
- **How much better:** my toy run: gradient norm at depth 50 of 31.79 with skips against 2.893e-15 without. Published: the ResNet paper trained a 152-layer network and reports it beating shallower ones (numbers in my [[ResNet]] note); U-Net used its copies to reach very precise outlines (numbers in [[U-Net 2015]]). Dates from the two papers, not rechecked here.

## Related
[[Residual Connection]] · [[ResNet]] · [[U-Net 2015]] · [[Image Segmentation]] · [[Vanishing and Exploding Gradients]] · [[Backpropagation]] · [[Convolution]] · [[Feature Map]] · [[Pooling]] · [[Transformer]] · [[Layer Normalization]] · [[Batch Normalization]] · [[VGG]]

## Sources
- My own run (Python 3.9.6, PyTorch 2.8.0, CPU). Every number in this note was printed by the code above, except the channel counts from the U-Net paper and the ResNet findings, which come from my notes [[U-Net 2015]] and [[ResNet]] (themselves built from the papers' text).
- Images: [ResBlock.png](https://commons.wikimedia.org/wiki/File:ResBlock.png) (LunarLullaby, CC BY-SA 4.0) and the U-Net redrawing by Mehrdad Yazdani (CC BY-SA 4.0), both on Wikimedia Commons; both already used in other notes of this vault.
- Not read: the DenseNet paper. Not checked: the note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/neural-nets Skip connection:::A shortcut that lets a layer's input jump over some layers and join their result later, either by adding or by gluing side by side.

#flashcards/ml/concept #flashcards/topic/neural-nets What are the two common ways to join a skip connection, and one example network for each?::Add (residual connection; ResNet, Transformer) and concatenate along the channel axis (U-Net).

#flashcards/ml/concept #flashcards/topic/neural-nets Why can you only add a skip when the shapes match, but you can always concatenate?::Adding is done number by number, so both maps need the same shape; in my run adding 4 and 6 channels failed. Concatenation just stacks channels (4 + 4 = 8).

#flashcards/ml/concept #flashcards/topic/neural-nets Why does an additive skip help a very deep network learn?::The error going backwards gets a factor of $1+\partial F/\partial x$ per block instead of only $\partial F/\partial x$, so it does not shrink to zero. In my toy run the gradient at depth 50 was 31.79 with skips and 2.893e-15 without.

#flashcards/ml/formula #flashcards/topic/neural-nets Write the additive skip (residual block) and explain each symbol.
?
$$y = x + F(x)$$
- $x$: the block's input, carried over unchanged. Without it the block would lose the input.
- $F(x)$: what the block's layers compute; same shape as $x$; it only learns the change.
- $y$: the output. If $F$ outputs zero, $y=x$, the identity.

#flashcards/ml/example #flashcards/topic/neural-nets With $x=2$ and $F(x)=0.5$, what is the residual block output, and what if $F=0$?::$y=2+0.5=2.5$; with $F=0$, $y=2$, the input unchanged.

#flashcards/ml/formula #flashcards/topic/neural-nets Write the slope of a residual block and explain each symbol.
?
$$\frac{\partial y}{\partial x} = 1 + \frac{\partial F}{\partial x}$$
- $\partial y/\partial x$: the factor the backward error is multiplied by in this block.
- $1$: from the shortcut; never zero, however small $F$'s slope is.
- $\partial F/\partial x$: the layers' slope; in a plain net it is the only term, so a slope below 1 shrinks the error at each layer.

#flashcards/ml/example #flashcards/topic/neural-nets If every block's layers have slope 0.1, what is the total backward factor over 3 blocks and over 20 blocks, with and without skips?::With skips: $1.1^3=1.331$ and $1.1^{20}=6.727$. Without: $0.1^3=0.001$ and $10^{-20}$ (my arithmetic).

#flashcards/ml/concept #flashcards/topic/neural-nets Why does the "1 +" in $1+\partial F/\partial x$ matter more than the exact value of $\partial F/\partial x$?::With a slope of 0.1 over 20 blocks a plain stack multiplies to $10^{-20}$, while the skip version gives $1.1^{20}=6.727$. The 1 keeps each factor near 1, so the product neither dies nor changes by orders of magnitude.

#flashcards/ml/concept #flashcards/topic/neural-nets What does concatenation cost that adding does not, with a number?::The next layer must read twice the channels. In the U-Net bottom step, 1024 to 512 channels needs 4,719,104 numbers against 2,359,808 for 512 to 512 (my run, 3 by 3 convolution, weights plus biases).

#flashcards/ml/concept #flashcards/topic/neural-nets Why does U-Net concatenate instead of add?::The shrinking half throws away exact positions; concatenating the copied maps keeps both the sharp copy and the upsampled map separate, so the next layer can choose how much of each to use. (This reasoning is mine; the U-Net paper only says the copies give localization information.)

#flashcards/ml/concept #flashcards/topic/neural-nets What is the one risk my toy run showed for stacks of additive skips?::The gradient can grow instead of shrink: 8.2 at depth 5 and 31.79 at depth 50 (one seed). Real nets add normalisation layers to keep it in check.

#flashcards/ml/cloze #flashcards/topic/neural-nets A block whose layers output ==zero== returns its input unchanged, so a deep residual stack can start as the ==identity==.

#flashcards/ml/image #flashcards/topic/neural-nets ![[residual-block.png]]
What does the curved path on the right do?
?
The input $x$ is copied along it unchanged and added at the circled plus to $F(x)$ from the two layers, so the output is $x+F(x)$. (LunarLullaby, CC BY-SA 4.0.)

#flashcards/ml/image #flashcards/topic/neural-nets ![[unet-architecture.png]]
What do the horizontal arrows in this U-Net drawing do?
?
They copy feature maps from the shrinking (left) half to the growing (right) half, where they are concatenated with the upsampled maps. It is a redrawing of the idea, not the paper's own figure. (Mehrdad Yazdani, CC BY-SA 4.0.)
