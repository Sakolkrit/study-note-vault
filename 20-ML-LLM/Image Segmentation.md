---
tags: [computer-vision, deep-learning, segmentation, CNN]
status: growing
level: 4
tier: 2
---

# Image Segmentation

**Image segmentation** labels every single pixel of a picture, so you get the exact shape of each object, not just a box around it.

## 🟢 1. Simple (age 12)
Object detection draws a rectangle around a cat. Segmentation colours in exactly the pixels that are cat, following its ears and tail. It is like the difference between putting a box around a country on a map and shading the country's real border.

**Example:** A colouring book. A detector draws a square around the dog. Segmentation colours the dog itself, and leaves the grass and sky uncoloured.

## 🔵 2. How it works
1. The input is a picture. The output is a **label for every pixel**: a picture of the same size, where each pixel holds a class (road, person, sky...) or an object number.
2. **Semantic segmentation** gives each pixel a class and does not separate two objects of the same class (two dogs are both just "dog").
3. **Instance segmentation** also separates the objects: dog 1 and dog 2 get their own masks.
4. The network is made of [[Convolution]] layers all the way through, with no dense layer at the end (a "fully convolutional" network), so it accepts any picture size and gives an output of matching size.
5. The picture is first shrunk to capture the big picture (what is where), then enlarged again to recover the fine detail (exactly where the edge is).
6. Shortcuts carry fine detail from the early layers straight to the late layers ([[Residual Connection]] spirit; the U-Net design is a well-known example).
7. To score the result, compare the predicted mask with the true mask by overlap: IoU or the Dice score (below).
8. **The catch:** labelling every pixel by hand for training data is slow and costly, and a small object is easy to miss.


(No image: I found no free, on-topic picture on Wikimedia Commons for this note.)

## 🟠 3. Technical detail
- **Three vision tasks, three outputs:** [[Image Classification]] gives one label per picture, [[Object Detection]] gives boxes, segmentation gives a label per pixel.
- **Fully convolutional networks (FCN):** the abstract of Long et al. (arXiv 1411.4038) says networks trained end to end, pixels to pixels, exceed the state of the art in semantic segmentation. They adapt classification networks (AlexNet, VGG and GoogLeNet) by fine-tuning ([[Transfer Learning]], [[VGG]]) and combine a deep, coarse layer with a shallow, fine one. It reports 62.2% mean IU on PASCAL VOC 2012, a 20% relative improvement, with inference in a third of a second.
- **U-Net:** a contracting path and a symmetric expanding path (abstract, arXiv 1505.04597). It was trained from very few images with heavy [[Data Augmentation]], beat a sliding-window network on an ISBI neuron-segmentation challenge, and segments a 512 by 512 image in under a second.
- **Mask R-CNN:** extends Faster R-CNN with a branch that predicts a mask for each detected box, at 5 frames per second (abstract, arXiv 1703.06870).
- **Metrics:** IoU (also called the Jaccard index) and the Dice score for each class, and the mean over classes (mean IU). Pixel accuracy alone is misleading when the object is small (see the code), the same trap as [[Accuracy]].
- **Related:** the shrink-then-enlarge shape is like an [[Autoencoder]]; [[Feature Map]] and [[Receptive Field]] explain what each layer sees.

```python
import numpy as np
n = 100
yy, xx = np.mgrid[:n, :n]
mask = (yy - 50) ** 2 + (xx - 50) ** 2 <= 30 ** 2                        # the true object: a round blob
box = np.zeros_like(mask); box[20:81, 20:81] = True                       # the tightest square box around it
inter = (mask & box).sum()
print(f"object covers {mask.sum():,} pixels; its tight bounding box covers {box.sum():,}; the box is {box.sum() / mask.sum():.2f}x too big")
print(f"IoU of the box with the true shape: {inter / (mask | box).sum():.3f}   Dice: {2 * inter / (mask.sum() + box.sum()):.3f}")
small = np.zeros((256, 256), bool); small[100:110, 100:110] = True       # a small object, 100 pixels out of 65,536
blank = np.zeros_like(small)
print(f"a model that marks nothing: pixel accuracy {(blank == small).mean():.4f}, Dice {2 * (blank & small).sum() / (blank.sum() + small.sum()):.1f}")
print(f"labels to produce for a 256x256 picture: classification 1, detection a few boxes of 4 numbers each, segmentation {256 * 256:,}")
```

## 🔴 4. Deep / formal
**The Dice score.** For a predicted mask $P$ and a true mask $T$ (both are sets of pixels):

$$
\text{Dice}=\frac{2\,|P\cap T|}{|P|+|T|}
$$

**Where:**
- $P$ is the set of pixels the model marked as the object, and $T$ the set that really is the object. $|\cdot|$ counts pixels.
- $P\cap T$ is the set of pixels in both (the pixels the model got right). The 2 in the top counts the overlap once for each mask.
- **Matters:** it is 1 for a perfect mask and 0 when the masks do not overlap. It is the same formula as the [[F1 Score]] for pixels, so it ignores the large background of true negatives. That is why it does not reward a model that marks nothing, while pixel accuracy does.

**Example (tiny numbers):** a true object of 10 pixels, a prediction of 8 pixels, of which 6 are inside the object.
- Overlap $|P\cap T|=\mathbf6$. $|P|+|T|=8+10=\mathbf{18}$.
- Dice $=2\cdot6/18=12/18=\mathbf{0.667}$.
- IoU for the same masks: union $=8+10-6=12$, so IoU $=6/12=\mathbf{0.5}$.
- Meaning: the model found 6 of the 10 object pixels and added 2 wrong ones, so the masks overlap by about two thirds (Dice) or one half (IoU).

**Further notes:**
- **Dice and IoU are linked:** Dice $=2\,\text{IoU}/(1+\text{IoU})$ (from memory; check with the example: $2\cdot0.5/1.5=0.667$).
- **Loss:** training often uses cross-entropy per pixel, sometimes with a Dice-based loss for small objects ([[Loss Function]]).
- **Not covered here:** panoptic segmentation, video segmentation and the details of upsampling.

**What came before, and how this improved on it:**
- **Before:** bounding boxes ([[Object Detection]]), or classifying a small patch around every pixel one at a time (the sliding-window approach that the U-Net abstract says it outperformed).
- **Its limit:** a box includes background and cannot give an exact shape, and a patch-per-pixel network repeats the same work for neighbouring pixels.
- **What changed:** make the whole network convolutional so one pass labels all pixels (FCN, arXiv 1411.4038), add shortcuts for detail (U-Net, arXiv 1505.04597), and add a mask branch to a detector (Mask R-CNN, arXiv 1703.06870). All three abstracts were read; first authors are from memory.
- **How much better:** the FCN abstract reports 62.2% mean IU on PASCAL VOC 2012, a 20% relative improvement, with inference of a third of a second. In the code below, for a round object, its tightest square box covers 1.32 times too many pixels and has an IoU of only 0.758 and a Dice of 0.862 against the true shape. And a model that marks nothing on a small object scores a pixel accuracy of 0.9985 and a Dice of 0.0.

**Check it yourself:** run the code in level 3 above. It prints every number quoted in this block.

## Related
[[Image Classification]] · [[Object Detection]] · [[Convolutional Neural Network]] · [[Convolution]] · [[VGG]] · [[Transfer Learning]] · [[Data Augmentation]] · [[Autoencoder]] · [[F1 Score]] · [[Accuracy]] · [[Feature Map]] · [[History of Image Classification]]
- [[FCN 2014]] · [[U-Net 2015]] · [[Mask R-CNN 2017]]
- [[Intersection over Union]]
## Sources
- Long et al., *Fully Convolutional Networks for Semantic Segmentation*, arXiv 1411.4038; Ronneberger et al., *U-Net*, arXiv 1505.04597; He et al., *Mask R-CNN*, arXiv 1703.06870: abstracts read; first authors from memory.
- The relation between Dice and IoU is a standard identity from memory (checked on the example in the note).
- No image: I found no free, on-topic picture on Wikimedia Commons, so I added none.
- Every number in the history block was computed by the code in this note (toy data); dates and names marked "from memory" are not checked.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision Image segmentation:::Colour in exactly the pixels that are the object, following its ears and tail. Object detection only draws a box around it.

#flashcards/ml/formula #flashcards/topic/vision Write the Dice score of a predicted mask and explain each symbol.
?
$$\text{Dice}=\frac{2\,|P\cap T|}{|P|+|T|}$$
- $P$: the set of pixels the model marked as the object. $T$: the set that really is the object. $|\cdot|$ counts pixels.
- $P\cap T$: the pixels in both (the ones the model got right). The 2 in the top counts the overlap once for each mask.
- It is 1 for a perfect mask and 0 when the masks do not overlap. It is the same formula as the F1 score for pixels, so it ignores the large background of true negatives. That is why it does not reward a model that marks nothing, while pixel accuracy does.

#flashcards/ml/example #flashcards/topic/vision True object 10 pixels, prediction 8 pixels, 6 of them inside the object. What are Dice and IoU?::Overlap 6, $|P|+|T|=18$. Dice $=12/18=0.667$. IoU: union $=8+10-6=12$, IoU $=6/12=0.5$. The model found 6 of 10 object pixels and added 2 wrong ones.

#flashcards/ml/reverse #flashcards/topic/vision Segmentation:::Marks the exact pixels of an object. Detection draws a rectangle. It is like shading a country's real border instead of putting a box around it.
