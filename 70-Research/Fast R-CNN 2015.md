---
tags: [paper, vision, detection, CNN]
link: https://arxiv.org/abs/1504.08083
authors: Girshick (first author from memory)
year: 2015
status: abstract-read
field: object detection
---

# Fast R-CNN 2015

> **In one line for a 12-year-old:** R-CNN ran a picture-reading network on every suggested box. Fast R-CNN reads the picture once and then just looks up each box in the result.

## The problem
Make detection much faster to train and to use, and still accurate.

## What people did before
[[R-CNN 2013]], which classified each object proposal separately. SPPnet also existed (the abstract compares with it).

## The idea
Several innovations to improve training and testing speed while also increasing accuracy, so that proposals are classified efficiently with a deep convolutional network.

## What they found
With the very deep VGG16 network, 9x faster training than R-CNN, 213x faster at test time, and a higher mAP on PASCAL VOC 2012. Against SPPnet: 3x faster training, 10x faster testing, and more accurate (abstract).

## Why it mattered
A large speed gain with a measured number, and it paved the way for faster detectors (from memory).

## Limits and open questions
It still relied on a separate step to propose boxes (from memory; the abstract does not say). [[YOLO 2015]] removed the separate step.

## Concepts used
[[Convolutional Neural Network]], [[Fine-tuning]], [[Transfer Learning]], [[Object Detection]]

## Reading trail
- **Came from:** [[R-CNN 2013]]
- **Led to:** [[Faster R-CNN 2015]], then [[YOLO 2015]] and later detectors

## How much I read
Abstract read. Authors written as "(from memory)" were not checked. Numbers marked "abstract" come from the arXiv abstract page. Not opened in Obsidian.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Fast R-CNN (Girshick, 2015, arXiv 1504.08083): what problem did it attack?::Making detection much faster to train and use, and still accurate.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What did people do before Fast R-CNN?::R-CNN, which classified each object proposal separately. SPPnet also existed (the abstract compares with it).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Fast R-CNN: what was the idea?::Several innovations to improve training and testing speed while also increasing accuracy, so that proposals are classified efficiently with a deep convolutional network. In the note's picture: read the whole image once, then look up each box in the result.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Fast R-CNN: what did it find? (abstract)::With VGG16: 9x faster training than R-CNN, 213x faster at test time, and a higher mAP on PASCAL VOC 2012. Against SPPnet: 3x faster training, 10x faster testing, and more accurate.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Fast R-CNN: what limit remained?::It still relied on a separate step to propose boxes (from memory; the abstract does not say). YOLO removed that step.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Fast R-CNN: reading trail?::It came from R-CNN and led to Faster R-CNN, then YOLO and later detectors.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How much of this paper did the note read?::Abstract only; the paper body was not read. Authors marked "from memory" in the note were not checked.
