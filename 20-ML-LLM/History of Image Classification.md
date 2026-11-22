---
tags: [history, vision, image-classification, CNN, detection, timeline]
status: growing
level: 4
tier: 3
---

# History of Image Classification

How computers learned to say what is in a picture, and how each model fixed the one before it.

## 🟢 1. Simple (age 12)
At first, people told computers what to look for: edges, corners, colours. Then one team let a many-layered network **find its own clues**, and it won a big picture contest by a mile. After that each new network was a tweak to make it deeper, faster or smarter, until networks named things in photos about as well as people do.

**Example:** Spotting a dog. Old way: a person writes "four legs, fur, floppy ears" as numbers. New way: the network looks at a million dog photos and works out for itself that fur texture and ear shape help.

## 🔵 2. How it works
Read this as a staircase. Each step has a problem that the next step solves.
1. **Hand-made features plus a classifier** (the older recipe; see the older-recipes comparison in [[Image Classification]]). A person designs the clues. Limit: someone has to guess the right clues for every task.
2. **LeNet (1989 to 1998).** A small [[Convolutional Neural Network]] learns its own clues for digits ([[LeNet and AlexNet]]). Limit: small data, slow computers.
3. **AlexNet (2012).** The same idea, bigger, with ReLU, [[Dropout]] and GPUs ([[AlexNet 2012]]). It won the contest with 15.3% top-5 error against about 26% for the runner-up.
4. **VGG (2014).** Only tiny 3x3 filters, stacked 16 to 19 layers deep ([[VGG 2014]]). Showed depth matters. Limit: very deep stacks became hard to train.
5. **GoogLeNet / Inception (2014).** 22 layers, designed to use computing power better, keeping the budget constant while growing depth and width (arXiv 1409.4842, abstract read).
6. **ResNet (2015).** Shortcuts let the network learn only the change at each block, so 152 layers train ([[ResNet 2015]]). Error 3.57%.
7. **Vision Transformer (2020).** Cut the picture into patches and use a [[Transformer]] ([[ViT 2020]]). It needs lots of pre-training data.
8. **A side road: detection.** Naming *what* is in a picture is one task. Finding *where* is another: [[R-CNN 2013]] (box proposals plus a CNN), Fast R-CNN (faster) and [[YOLO 2015]] (one glance, real-time).

## 🟠 3. Technical detail
| Step | Year | New idea | Number from the source | Status |
|---|---|---|---|---|
| LeNet-5 | 1998 | learned convolution filters | first net (1989) about 9,760 free parameters | Wikipedia, read |
| AlexNet | 2012 | deep CNN on GPUs, ReLU, dropout | top-5 error 15.3%, runner-up about 26%; 61,100,840 parameters (PyTorch layout, counted here) | Wikipedia read; count by code |
| VGG | 2014 | 3x3 filters, 16 to 19 layers | 2nd in classification, 1st in localisation (ILSVRC 2014) | abstract read |
| GoogLeNet | 2014 | Inception, 22 layers | state of the art for ILSVRC 2014 classification and detection | abstract read |
| ResNet | 2015 | residual shortcuts, up to 152 layers | 3.57% top-5 test error (ensemble); 8x deeper than VGG, lower complexity | abstract read |
| ViT | 2020 | image patches as tokens | excellent results with substantially fewer training resources, after large pre-training | abstract read |
| R-CNN | 2013 | CNN on box proposals | mAP 53.3% on VOC 2012, more than 30% relative gain | abstract read |
| Fast R-CNN | 2015 | share the CNN work | 9x faster training, 213x faster testing than R-CNN (VGG16) | abstract read |
| YOLO | 2015 | one-pass detection | 45 frames per second; Fast YOLO 155; more localisation errors, fewer false alarms | abstract read |

The code turns the published figures into ratios and compares the parameter counts this vault computed.

```python
# Published ImageNet top-5 error rates quoted in this vault's notes, and what they mean in relative terms.
alex, resnet = 15.3, 3.57          # 2012 winner (Wikipedia, read), 2015 ensemble (ResNet abstract, read)
print("AlexNet 2012 top-5 error:", alex, "%  ->  ResNet 2015 ensemble:", resnet, "%")
print("relative drop in errors:", round(100 * (alex - resnet) / alex, 1), "%   (", round(alex / resnet, 2), "x fewer mistakes )")
# parameter counts the vault computed itself (see LeNet and AlexNet, ResNet)
print("AlexNet layers 8, parameters 61,100,840;  ResNet-152 layers 152, parameters 60,192,808")
print("layers x", round(152 / 8, 1), " parameters ratio", round(60_192_808 / 61_100_840, 3))
# R-CNN vs Fast R-CNN (abstract): training 9x faster, testing 213x faster
print("Fast R-CNN over R-CNN: training 9x faster, test-time 213x faster (abstract); YOLO 45 frames/s (abstract) -> ms per image:", round(1000 / 45, 1))
```

## 🔴 4. Deep / formal
This note has no equation. Reading the staircase critically:
- **Depth was the story until 2015.** Layers went from 8 to 152, which is 19 times, while parameters stayed about the same (0.985 ratio). Shortcuts and 3x3 filters made depth cheap. Depth alone was not the trick; making depth trainable was.
- **Error numbers need care.** The 15.3% (2012) and 3.57% (2015) are both top-5 errors on the contest, but one is a single entry and the other is an ensemble, and I did not check the contest definitions. So "4.29 times fewer mistakes" is a rough picture, not a controlled comparison.
- **Humans:** a figure for human error exists, but I have not verified one, so none is quoted.
- **Each step's price:** VGG is large and slow (from memory), ViT needs big pre-training data, YOLO trades accuracy of boxes for speed.
- **Not covered:** segmentation, EfficientNet and architecture search, ConvNeXt, self-supervised vision pre-training, and diffusion for images ([[Diffusion Models]]).
- **The Medium article** you gave could not be opened (the site blocks automated readers), so this note follows the primary papers instead. If you paste its text, I can compare and add what it covers.

**What came before, and how this improved on it:**
- **Before:** features designed by hand, fed to a separate classifier.
- **Its limit:** the features had to be guessed for each task, and accuracy plateaued.
- **What changed:** learn the features with a deep convolutional network, then make it deeper, trainable, and finally attention-based.
- **How much better:** top-5 error 15.3% (2012) to 3.57% (2015), 76.7% fewer errors (the code below).

## Related
- [[Image Classification]] · [[Convolutional Neural Network]] · [[LeNet and AlexNet]] · [[ResNet]] · [[Vision Transformer]] · [[Convolution]] · [[Pooling]] · [[Receptive Field]] · [[Transfer Learning]] · [[History of AI]] · [[History of Text Analytics]] · [[VGG]] · [[Object Detection]] · [[Image Segmentation]]

## Sources
- arXiv abstracts read: 1409.1556 (VGG), 1409.4842 (GoogLeNet), 1512.03385 (ResNet), 2010.11929 (ViT), 1311.2524 (R-CNN), 1504.08083 (Fast R-CNN), 1506.02640 (YOLO)
- Wikipedia articles on LeNet and AlexNet, as used in [[LeNet and AlexNet]]
- Medium article on the development of image classification: attempted, blocked (HTTP 403), not read

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/history Hand-made features plus a classifier:::The older recipe: a person designs the clues. Its limit: someone has to guess the right clues for every task.

#flashcards/ml/reverse #flashcards/topic/history LeNet (1989 to 1998):::A small convolution network that learns its own clues for digits. Its limit: small data and slow computers.

#flashcards/ml/reverse #flashcards/topic/history AlexNet (2012):::The same idea, bigger, with ReLU, dropout and GPUs. It won the contest with 15.3% top-5 error against about 26% for the runner-up.

#flashcards/ml/reverse #flashcards/topic/history VGG (2014):::Only tiny 3x3 filters stacked 16 to 19 layers deep. It showed depth matters. Its limit: very deep stacks became hard to train, and VGG is large and slow (from memory).

#flashcards/ml/reverse #flashcards/topic/history GoogLeNet / Inception (2014):::22 layers, designed to use computing power better, keeping the budget constant while growing depth and width (arXiv 1409.4842, abstract read).

#flashcards/ml/reverse #flashcards/topic/history ResNet (2015):::Shortcuts let each block learn only the change, so 152 layers train. 3.57% top-5 error (an ensemble).

#flashcards/ml/reverse #flashcards/topic/history Vision Transformer (2020):::Cut the picture into patches and use a Transformer. It needs lots of pre-training data.

#flashcards/ml/concept #flashcards/topic/history Which problem does ResNet solve that VGG left open?::Very deep stacks were hard to train. Shortcuts (residual connections) let 152 layers train.

#flashcards/ml/example #flashcards/topic/history 15.3% (AlexNet, 2012) to 3.57% (ResNet, 2015): how much better, and what caveat?::About 4.29 times fewer errors, or 76.7% fewer. The caveat: one is a single entry and the other an ensemble, and the note did not check the contest definitions, so it is a rough picture, not a controlled comparison.

#flashcards/ml/reverse #flashcards/topic/history Detection (a side road):::Naming what is in a picture is one task and finding where is another. R-CNN (box proposals plus a CNN), Fast R-CNN (faster) and YOLO (one glance, real-time) address the second.
