---
tags: [vision, transformer, ViT, image-classification, patches, computer-vision]
status: growing
level: 4
tier: 3
---

# Vision Transformer

Written **ViT**. A [[Transformer]] that looks at a picture by cutting it into small square **patches** and treating the patches like words.

## 🟢 1. Simple (age 12)
A normal image AI slides small windows across a picture. A Vision Transformer does something different: it **cuts the picture into a grid of little tiles**, like a jigsaw, lays the tiles out in a row like the words of a sentence, and lets every tile **look at every other tile** to work out what the picture shows.

**Example:** A photo of a teddy bear is cut into 196 tiles. One tile shows an ear, another an eye, another part of the book. By letting the ear tile "talk to" the eye tile, the AI works out that this is a bear.

## 🔵 2. How it works
1. Take a picture, for example 224 by 224 pixels with 3 colours.
2. Cut it into square **patches** of 16 by 16 pixels. That gives a $14\times14$ grid, so **196 patches**.
3. Flatten each patch into a list of numbers ($16\times16\times3=768$) and pass it through one learned linear layer to get a vector. That is the patch's [[Embedding]], like a word's embedding.
4. Add a **position embedding** to each patch so the model knows where it came from ([[Positional Encoding]]).
5. Add one extra special token at the front, **[CLS]**, to collect a summary of the whole picture.
6. Feed the 197 vectors into a standard Transformer **encoder** (the same stack as in [[BERT]]) with [[Self-Attention]], so every patch can attend to every other.
7. Take the [CLS] output and project it to class scores ([[Image Classification]]).
8. The same idea feeds other systems: a **vision-language model** lets a text model read image patches, either by feeding patch vectors in like extra tokens, or through cross-attention layers (lecture). Image **generation** uses diffusion Transformers ([[Diffusion Models]]).

## 🟠 3. Technical detail
- **Source:** *An Image is Worth 16x16 Words* (Dosovitskiy et al., 2020), which shows that a pure Transformer applied to image patches can do very well on classification, and that the reliance on CNNs is not necessary (arXiv 2010.11929 abstract, read).
- **Fewer built-in assumptions:** a [[Convolutional Neural Network]] has locality and weight sharing built in. A ViT has much weaker "inductive biases" (the lecture's wording), which gives generality but typically needs a lot of data to learn the same things. The ViT abstract says it does well when **pre-trained on large amounts of data** and transferred to smaller benchmarks.
- **Cost:** attention compares all pairs of tokens, so the cost is the square of the number of patches. Patches keep that manageable: 197 tokens instead of 50,176 pixels.
- **Patch size trade-off:** smaller patches mean more tokens and finer detail but a quadratic cost.
- **Size of ViT-Base/16:** 12 layers, width 768, 12 heads. The code counts 86,567,656 parameters and matches torchvision's model exactly.
- **Position:** learned position embeddings are added to the patch vectors (the original ViT used learned ones; the lecture shows the position-aware embedding step).
- **Shared tricks:** modern models carry over [[Residual Connection]]s, [[Layer Normalization]] and rotary position tricks (the lecture mentions a multimodal rotary variant in an image-generation model; I did not read it).
- **Not covered:** hybrid CNN–Transformer models, and training recipes.

```python
import numpy as np
import torch, torchvision
# 1) cut a 224x224 colour picture into 16x16 patches: each patch is flattened into one vector
img = np.random.RandomState(0).rand(224, 224, 3)
P = 16
patches = img.reshape(224 // P, P, 224 // P, P, 3).transpose(0, 2, 1, 3, 4).reshape(-1, P * P * 3)
print('patches:', patches.shape[0], '| numbers per patch:', patches.shape[1], '| first patch equals the top-left corner:', np.allclose(patches[0], img[:P, :P].ravel()))
# the same cut made by PyTorch's unfold
u = torch.nn.functional.unfold(torch.tensor(img).permute(2, 0, 1)[None], P, stride=P)[0].T            # (196, 768) but channel-first inside each patch
print('PyTorch unfold gives the same patches (up to the order inside a patch):', u.shape[0] == 196 and np.allclose(np.sort(u.numpy(), 1), np.sort(patches, 1)))
# 2) the sequence the Transformer sees: 196 patches + 1 [CLS] token
n = 196 + 1
print('sequence length:', n, '| attention score cells:', n * n, '| if every PIXEL were a token:', (224 * 224) ** 2, '| ratio', round((224 * 224) ** 2 / (n * n)))
# 3) parameter count of ViT-Base/16 against torchvision's own model
H, L, C = 768, 12, 1000
layer = 12 * H * H + 13 * H
total = P * P * 3 * H + H + H + n * H + L * layer + 2 * H + H * C + C       # patch projection, [CLS], positions, layers, final LayerNorm, classifier
tv = torchvision.models.vit_b_16(weights=None)
print('counted', total, '| torchvision vit_b_16', sum(p.numel() for p in tv.parameters()), '| equal:', total == sum(p.numel() for p in tv.parameters()))
```

## 🔴 4. Deep / formal
**How many tokens.** A picture of height $H$ and width $W$ cut into $P\times P$ patches:

$$
N=\frac{H\,W}{P^{2}}
$$

**Where:**
- $H$ and $W$ are the height and width of the picture in pixels.
- $P$ is the side of one square patch in pixels (16 in ViT-Base/16).
- $N$ is the number of patches, which become the tokens (plus one for [CLS]).
- **Matters:** the number of tokens falls with the *square* of the patch size. Halving $P$ gives four times as many tokens. $H$ and $W$ should be multiples of $P$, otherwise the picture must be cropped or resized.

**Example (tiny numbers):** $H=W=224$, $P=16$.
- $224\times224/16^2=50{,}176/256=\mathbf{196}$ patches.
- Each patch has $16\times16\times3=\mathbf{768}$ numbers, as the code's first patch confirms.
- Meaning: a picture with 50,176 pixels becomes a sequence of 196 tokens (197 with [CLS]).

**Attention cost.** Self-attention compares every token with every other, so for $n$ tokens:

$$
\text{score cells}=n^{2}
$$

**Where:**
- $n$ is the sequence length ($N+1$ for ViT).
- Each cell is one query–key comparison in a layer, per head.
- **Matters:** the cost grows with the square of the length, which is why treating each pixel as a token would be hopeless, and patches make the picture a short sequence.

**Example (tiny numbers):** with $n=197$:
- $197^2=\mathbf{38{,}809}$ cells.
- Pixel by pixel ($n=50{,}176$): $50{,}176^2=\mathbf{2{,}517{,}630{,}976}$ cells.
- Ratio $\approx\mathbf{64{,}872}$ times more.
- Meaning: cutting into patches reduces the attention score grid by a factor of roughly 65,000 for one layer and head.

**Further notes:**
- **Patches as a convolution:** cutting patches and projecting each with one linear layer is the same as a convolution with kernel size $P$ and stride $P$ ([[Convolution]]). That is a standard equivalence (from memory; the code checks only the cutting, not this equivalence).
- **Parameter count in the code:** patch projection $768\cdot768+768$, [CLS] $768$, positions $197\cdot768$, 12 layers of $12H^2+13H$, a final LayerNorm $2\cdot768$ and a 1,000-class head $768\cdot1000+1000$. Total $86{,}567{,}656$.
- **Not tested:** accuracy, data hunger (the abstract's pre-training claim), and any speed.

**What came before, and how this improved on it:**
- **Before:** convolutional networks ([[Convolutional Neural Network]], [[ResNet]]) as the standard for images, and attention used only on top of them or in place of some parts while keeping their overall structure (ViT abstract).
- **Its limit:** the ViT abstract notes that in vision, attention was used together with convolutions or to replace certain parts of them, and Transformers' use in vision remained limited.
- **What changed:** a pure Transformer on a sequence of image patches (Dosovitskiy et al., arXiv 2010.11929, 22 October 2020, abstract read).
- **How much better:** the abstract says that, when pre-trained on large data, ViT attains excellent results compared with state-of-the-art CNNs while needing substantially fewer computational resources to train. It gives no figures in the abstract and I did not read the tables. In this note's code: 196 patch tokens instead of 50,176 pixels, so a single attention grid has 38,809 cells instead of about 2.5 billion.

## Related
- [[Transformer]] · [[Self-Attention]] · [[Image Classification]] · [[Convolutional Neural Network]] · [[Convolution]] · [[BERT]] · [[Layer Normalization]] · [[Positional Encoding]] · [[Embedding]] · [[Diffusion Models]] · [[LLM]] · [[ResNet]] · [[ViT 2020]] · [[History of Image Classification]] · [[CLIP]]

## Sources
- Stanford CME 295 (Fall 2025), Lecture 9 slides, Afshine and Shervine Amidi
- Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, 2020 (arXiv 2010.11929, abstract read)
- Liu et al., *Visual Instruction Tuning*, 2023 and the Llama 3 report, 2024 (named in the lecture for vision-language models; not read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision Vision Transformer (ViT):::Cuts the picture into a grid of little tiles (patches), lays them out in a row like the words of a sentence, and lets every tile look at every other tile to work out what the picture shows.

#flashcards/ml/formula #flashcards/topic/vision Write the number of patches and explain each symbol.
?
$$N=\frac{H\,W}{P^{2}}$$
- $H$, $W$: the height and width of the picture in pixels.
- $P$: the side of one square patch in pixels (16 in ViT-Base/16).
- $N$: the number of patches, which become the tokens (plus one for the [CLS] token).
- The number of tokens falls with the square of the patch size. Halving $P$ gives four times as many tokens.

#flashcards/ml/formula #flashcards/topic/vision Write the attention cost and explain each symbol.
?
$$\text{score cells}=n^{2}$$
- $n$: the sequence length ($N+1$ for ViT).
- Each cell is one query-key comparison in a layer, per head.
- The cost grows with the square of the length, which is why treating each pixel as a token would be hopeless and patches make the picture a short sequence.

#flashcards/ml/cloze #flashcards/topic/vision A Vision Transformer turns a picture into a short ==sequence== of patches, which become the ==tokens==.

#flashcards/ml/example #flashcards/topic/vision For a 224×224 picture with $P=16$, how many patches, and how many numbers per patch?::224 × 224 / 16² = 50,176 / 256 = 196 patches. Each patch has 16 × 16 × 3 = 768 numbers. 50,176 pixels become 196 tokens (197 with [CLS]).

#flashcards/ml/example #flashcards/topic/vision How many attention score cells for $n=197$ and for pixel-by-pixel $n=50{,}176$?::197² = 38,809. 50,176² = 2,517,630,976. About 64,872 times more, so patches cut the score grid by a factor of roughly 65,000 for one layer and head.
