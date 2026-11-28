---
tags: [imagenet, ILSVRC, dataset, benchmark, image-classification, computer-vision, ML]
status: growing
level: 4
tier: 2
---

# ImageNet

**ImageNet** is a very large collection of labelled photographs: over 15 million images in about 22,000 categories, collected from the web and labelled by people. **ILSVRC** (the ImageNet Large-Scale Visual Recognition Challenge), run every year from 2010, uses a smaller part of it: about 1.2 million training images in 1,000 categories. The yearly contest, and the pre-trained networks that came out of it, shaped modern computer vision.

## 🟢 1. Simple (age 12)
Imagine a giant photo album with millions of pictures: dogs, cars, mushrooms, tools, thousands of kinds of things. Every picture has a label written by a person. A yearly contest hands out most of the album (about 1,200 pictures for each of 1,000 kinds) and asks computer programs to name what is in new pictures they have never seen. The score is how often they are right.

**Example:** a picture shows a Siberian husky. The program makes five guesses. If "Siberian husky" is among its five guesses, it counts as right for the contest's main score (top-5).

## 🔵 2. How it works
1. **The collection:** photographs gathered from the web and labelled by people through a crowd-sourcing service (Amazon's Mechanical Turk, as the AlexNet paper says).
2. **The contest subset (ILSVRC):** roughly 1,000 images in each of 1,000 categories: about 1.2 million for training, 50,000 for validation and a hidden test set (see the numbers below).
3. **The task:** classification (name the one label of the image) with two scores, **top-1 error** (the first guess must be right) and **top-5 error** (the right label must be among five guesses). The contest ranks by top-5 ([[Image Classification]]).
4. **Other tasks:** a detection task with 200 classes (find and box each object, scored with [[Mean Average Precision]]) was added later ([[Object Detection]]).
5. **Pre-training:** a network trained on the 1.2 million images is reused for other tasks by fine-tuning ([[Transfer Learning]], [[Fine-tuning]]); [[R-CNN 2013]] did this for detection.
6. **Preparing the images:** they come in many sizes, so networks resize them to a fixed size, crop and flip them ([[Data Augmentation]]).

No image: I did not search for a free picture of the data set, and ImageNet's own sample images are not mine to reuse. The numbers below come from the papers I read; I did not download the data.

## 🟠 3. Technical detail
No code touches the data: I did not download ImageNet. The script `imagenet_numbers.py` (Python 3.9.6) only does arithmetic on numbers printed in the papers I read.

```python
"""ImageNet / ILSVRC numbers as printed in the papers I read, with my arithmetic (no data set was downloaded)."""
import math
print("bits per label for 1000 classes: %.3f (AlexNet paper: 'each training example imposes 10 bits of constraint')" % math.log2(1000))
print("top-5 error of random guessing over 1000 classes: %.1f%%" % (100*(1 - 5/1000)), "| top-1: %.1f%%" % (100*(1 - 1/1000)))
print("images per class: 1.2M / 1000 = %d" % (1.2e6/1000), "| ILSVRC share of all ImageNet images: %.1f%%" % (100*1.2e6/15e6))
# ILSVRC-2010 test (AlexNet paper, Table 1): top-1 / top-5 error
print("2010: best entry 47.1 / 28.2, best later published 45.7 / 25.7, AlexNet 37.5 / 17.0")
# classification winners, top-5 error (GoogLeNet paper, Table 2, no external data unless noted)
win = [("SuperVision", 2012, 16.4), ("SuperVision (22k extra)", 2012, 15.3), ("Clarifai", 2013, 11.7), ("Clarifai (22k extra)", 2013, 11.2),
       ("VGG (2nd)", 2014, 7.32), ("MSRA (3rd)", 2014, 7.35), ("GoogLeNet", 2014, 6.67)]
for n, y, e in win: print("%-24s %d  top-5 %.2f%%  = %.1f%% of the 2012 no-extra-data error" % (n, y, e, 100*e/16.4))
print("2012 -> 2014 relative cut: %.1f%% (16.4 -> 6.67); 2013 -> 2014: %.1f%% (11.7 -> 6.67)" % (100*(1 - 6.67/16.4), 100*(1 - 6.67/11.7)))
print("test set sizes as printed: AlexNet paper 150,000; GoogLeNet paper 100,000 (they disagree); validation 50,000 in both")
# detection (200 classes)
print("ILSVRC2013 detection: R-CNN 31.4, OverFeat 24.3; ILSVRC2014: GoogLeNet 43.9 (GoogLeNet paper); detection classes: 200")
print("R-CNN paper: selective search on ILSVRC2013 val: 2,403 boxes/image, recall 91.6% at IoU 0.5 (PASCAL: about 98%)")
```

I saw:
```
bits per label for 1000 classes: 9.966 (AlexNet paper: 'each training example imposes 10 bits of constraint')
top-5 error of random guessing over 1000 classes: 99.5% | top-1: 99.9%
images per class: 1.2M / 1000 = 1200 | ILSVRC share of all ImageNet images: 8.0%
2010: best entry 47.1 / 28.2, best later published 45.7 / 25.7, AlexNet 37.5 / 17.0
SuperVision              2012  top-5 16.40%  = 100.0% of the 2012 no-extra-data error
SuperVision (22k extra)  2012  top-5 15.30%  = 93.3% of the 2012 no-extra-data error
Clarifai                 2013  top-5 11.70%  = 71.3% of the 2012 no-extra-data error
Clarifai (22k extra)     2013  top-5 11.20%  = 68.3% of the 2012 no-extra-data error
VGG (2nd)                2014  top-5 7.32%  = 44.6% of the 2012 no-extra-data error
MSRA (3rd)               2014  top-5 7.35%  = 44.8% of the 2012 no-extra-data error
GoogLeNet                2014  top-5 6.67%  = 40.7% of the 2012 no-extra-data error
2012 -> 2014 relative cut: 59.3% (16.4 -> 6.67); 2013 -> 2014: 43.0% (11.7 -> 6.67)
test set sizes as printed: AlexNet paper 150,000; GoogLeNet paper 100,000 (they disagree); validation 50,000 in both
ILSVRC2013 detection: R-CNN 31.4, OverFeat 24.3; ILSVRC2014: GoogLeNet 43.9 (GoogLeNet paper); detection classes: 200
R-CNN paper: selective search on ILSVRC2013 val: 2,403 boxes/image, recall 91.6% at IoU 0.5 (PASCAL: about 98%)
```

**Facts as the papers I read state them.**

| Item | Value | Source I read |
|---|---|---|
| Whole data set | over 15 million labelled high-resolution images, about 22,000 categories | AlexNet paper, section 2 |
| Contest subset (ILSVRC) | about 1,000 images in each of 1,000 categories; 1.2 million training, 50,000 validation | AlexNet paper, section 2; GoogLeNet paper, section 7 |
| Test set | 150,000 images (AlexNet paper) or 100,000 images (GoogLeNet paper) | the two papers disagree |
| Detection task | 200 classes; correct if class matches and boxes overlap at least 50% | GoogLeNet paper, section 8 |
| Labelling | web images labelled by people on Amazon Mechanical Turk | AlexNet paper, section 2 |
| Test labels | public for ILSVRC-2010 only; hidden for 2012 | AlexNet paper, section 2 |

- **What the columns are:** the quantity, its value, and which paper I read it in.
- Each paper's text is the only source: I did not read the original ImageNet paper or the contest report, so the facts are second-hand.

**The contest as a ladder of top-5 errors (from the GoogLeNet paper's Table 2, classification, my percentages).**

| Team | Year | Top-5 error (%) | Share of the 2012 winner's error (mine) |
|---|---|---|---|
| SuperVision (AlexNet) | 2012 | 16.4 (15.3 with extra data) | 100% |
| Clarifai | 2013 | 11.7 (11.2 with extra data) | 71% |
| VGG (2nd) | 2014 | 7.32 | 45% |
| MSRA (3rd) | 2014 | 7.35 | 45% |
| GoogLeNet (1st) | 2014 | 6.67 | 41% |

- **Top-5 error:** the share of test images whose true label is not among the five best guesses. **Extra data:** images from outside the contest's 1.2 million training images (the 22,000-category release).

**Summary: what the numbers show.** In three years the winning top-5 error fell from 16.4% to 6.67%, a 59.3% relative cut (my arithmetic; 43.0% from 2013 to 2014 alone). The ILSVRC-2010 numbers in the AlexNet paper (best entry 28.2% top-5, AlexNet 17.0%) show the same break. Guessing alone gives a top-5 error of 99.5% and a top-1 error of 99.9% (my arithmetic). Caution: contest entries differ in how many models and crops they average (GoogLeNet's 6.67% needs 7 models and 144 crops), and the test sets and rules changed between years.

## 🔴 4. Deep / formal
No equation: it is a data set; its numbers are counts and error rates.

**What the papers say about it.**
- **Why it mattered for AlexNet:** earlier labelled sets had "tens of thousands" of images (NORB, Caltech-101/256, CIFAR-10/100) and were enough only for simple tasks; ImageNet made training a 60-million-parameter network possible, though "the immense complexity of the object recognition task means that this problem cannot be specified even by a dataset as large as ImageNet" (AlexNet paper, section 1).
- **How much each label tells the network:** with 1,000 classes "each training example impose[s] 10 bits of constraint on the mapping from image to label" (AlexNet paper, section 4); my check: $\log_2 1000=9.97$ bits.
- **How images are prepared:** AlexNet rescales so the shorter side is 256 and crops the central 256 × 256; it subtracts the mean pixel and trains on random 224 × 224 crops (AlexNet paper, sections 2 and 4.1). GoogLeNet takes crops of 8% to 100% of the image area with aspect ratio 3/4 to 4/3 (GoogLeNet paper, section 6).
- **The 22,000-category release:** AlexNet pre-trained a network on all of it (15 million images, "Fall 2011 release") and fine-tuned on ILSVRC-2012; GoogLeNet used no external data in its classification entry (AlexNet paper, section 6; GoogLeNet paper, section 7).
- **Detection data:** the ILSVRC2013 detection set has 200 classes; R-CNN's selective search gave 2,403 proposals per image with 91.6% recall at IoU 0.5 on its validation images, against about 98% on PASCAL VOC (R-CNN paper, section 4).

**Where the two papers I read disagree.** The test set is 150,000 images in the AlexNet paper and 100,000 in the GoogLeNet paper. They may describe different years (2010 and 2012 against 2014), but neither says so; I did not check.

**Limits and trade-offs:**
- **Web-image bias:** images were collected from the web, and the labels come from crowd workers; fine-grained classes are hard even for people (the GoogLeNet paper shows two dog classes it calls hard to tell apart). I did not look into label errors.
- **One object per image:** classification assumes one label per image, though many images hold several objects, which is why top-5 is used.
- **Test-set reuse:** a public test set invites tuning to it; AlexNet reports validation numbers when test labels were hidden, saying they "do not differ by more than 0.1%" for its runs (AlexNet paper, section 6).
- **Saturation:** with top-5 error under 7% the benchmark stopped separating methods well (my reading; the later history is general knowledge and was not checked here).

**What came before, and how this improved on it:**
- **Before:** labelled image sets of tens of thousands of images (NORB, Caltech-101 and 256, CIFAR-10 and 100) and the PASCAL VOC contest.
- **Its limit:** too small for large models: "the shortcomings of small image datasets have been widely recognized" (AlexNet paper, section 1).
- **What changed:** millions of labelled images in thousands of categories, and a yearly contest with a shared test set.
- **How much better:** a training set of 1.2 million images against the papers' "tens of thousands" for earlier sets (24 times as many as 50,000 images, my arithmetic); and on the contest a top-5 error falling from 28.2% (2010 best entry) to 6.67% (2014 winner) by the printed numbers. These are the papers' numbers; I ran nothing on the data.

## Related
[[Image Classification]] · [[Object Detection]] · [[AlexNet 2012]] · [[GoogLeNet 2014]] · [[VGG 2014]] · [[ResNet 2015]] · [[R-CNN 2013]] · [[Fast R-CNN 2015]] · [[Transfer Learning]] · [[Fine-tuning]] · [[Data Augmentation]] · [[History of Image Classification]] · [[Mean Average Precision]]

## Sources
- Papers read in full as text for this: [[AlexNet 2012]] (NIPS 2012, sections 1, 2, 4, 6), [[GoogLeNet 2014]] (arXiv 1409.4842, sections 6 to 8, Tables 2 and 3) and [[R-CNN 2013]] (arXiv 1311.2524, section 4). The ImageNet paper (Deng et al.), the contest reports and the web page were not read, so the facts are second-hand. The arithmetic is mine (`imagenet_numbers.py`).
- Not checked: the exact sizes of each year's validation and test sets, label error rates, and the post-2014 history. The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision ImageNet:::A collection of over 15 million human-labelled web images in about 22,000 categories; its contest subset ILSVRC has about 1.2 million training images in 1,000 categories.

#flashcards/ml/concept #flashcards/topic/vision What is ILSVRC?::The ImageNet Large-Scale Visual Recognition Challenge, held yearly from 2010: classification of images into 1,000 categories (ranked by top-5 error) and, later, detection of 200 classes (AlexNet paper, section 2; GoogLeNet paper, section 8).

#flashcards/ml/concept #flashcards/topic/vision What are top-1 and top-5 error?::Top-1: the first guess must be the true label. Top-5: the true label must be among the five best guesses (the contest ranks by it); top-5 is forgiving when classes are near-twins or an image holds several objects.

#flashcards/ml/example #flashcards/topic/vision How many images does ILSVRC have per class and what does random guessing score?::About 1,200 training images per class (1.2 million / 1,000) and a top-5 error of 99.5% for guessing (1 - 5/1000), top-1 99.9% (my arithmetic).

#flashcards/ml/concept #flashcards/topic/vision How did winning top-5 error change from 2012 to 2014?::16.4% (SuperVision, 2012), 11.7% (Clarifai, 2013), 6.67% (GoogLeNet, 2014; VGG 7.32%, MSRA 7.35%); a 59.3% relative cut from 2012 to 2014 (my arithmetic; GoogLeNet paper, Table 2).

#flashcards/ml/concept #flashcards/topic/vision Why did ImageNet matter for deep learning?::Earlier labelled sets had tens of thousands of images; ImageNet allowed training a 60-million-parameter network (AlexNet) without overfitting too badly, and its pre-trained networks were reused for other tasks, as in R-CNN (AlexNet paper, section 1; R-CNN paper).

#flashcards/ml/concept #flashcards/topic/vision Where do the AlexNet and GoogLeNet papers disagree about ILSVRC?::The size of the test set: 150,000 images in the AlexNet paper, 100,000 in the GoogLeNet paper (both say 1.2 million training and 50,000 validation); neither explains the gap.

#flashcards/ml/concept #flashcards/topic/vision What are the limits of ImageNet as a benchmark?::Web-image and crowd-label bias, one label per image although images hold several objects, a shared test set that invites tuning, and scores near saturation at the top (the last point is my reading).

#flashcards/ml/cloze #flashcards/topic/vision The contest subset ILSVRC has ==1,000== categories and about ==1.2 million== training images, ranked by ==top-5== error.
