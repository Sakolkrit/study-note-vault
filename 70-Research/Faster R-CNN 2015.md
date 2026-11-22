---
tags: [paper, vision, detection, CNN]
link: https://arxiv.org/abs/1506.01497
authors: Ren et al. (first author from memory)
year: 2015
status: abstract-read
field: object detection
---

# Faster R-CNN 2015

> **In one line for a 12-year-old:** Fast R-CNN was quick except for one slow step: guessing where the objects might be. Faster R-CNN lets the same network make those guesses, almost for free.

## The problem
After Fast R-CNN sped up the detector, the step that proposed object locations became the bottleneck (the abstract says so).

## What people did before
[[Fast R-CNN 2015]] and SPPnet, which still depended on separate region proposal algorithms to hypothesize object locations.

## The idea
A Region Proposal Network (RPN): a fully convolutional network that shares the full-image convolutional features with the detection network and predicts object bounds and objectness scores at each position. The RPN and Fast R-CNN are merged into one network by sharing their convolutional features ([[Object Detection]], [[Feature Map]]).

## What they found
With the very deep VGG-16 model, the whole detection system runs at 5 frames per second on a GPU, including all steps, with state-of-the-art accuracy on PASCAL VOC 2007, 2012 and MS COCO using only 300 proposals per image. Faster R-CNN and RPN were the foundation of the 1st-place entries in several tracks of the ILSVRC and COCO 2015 competitions (abstract).

## Why it mattered
It made proposals nearly cost-free and started the line of two-stage detectors that [[Mask R-CNN 2017]] extended.

## Limits and open questions
5 frames per second is fast for its accuracy but not real time for video; [[YOLO 2015]] ran faster with a different design (abstract of YOLO: 45 frames per second).

## Concepts used
[[Object Detection]], [[Convolutional Neural Network]], [[VGG]], [[Feature Map]], [[Transfer Learning]] · [[Tiny Object Detector]] · [[Non-Maximum Suppression]] · [[MobileNet]]
## Reading trail
- **Came from:** [[Fast R-CNN 2015]]
- **Led to:** [[Mask R-CNN 2017]]

## How much I read
Abstract read. Authors written as "(from memory)" were not checked. Numbers marked "abstract" come from the arXiv abstract page. Not opened in Obsidian.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Faster R-CNN (Ren et al., 2015, arXiv 1506.01497): what problem did it attack?::After Fast R-CNN sped up the detector, the step that proposed object locations became the bottleneck (the abstract says so).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What did people do before Faster R-CNN?::Fast R-CNN and SPPnet, which still depended on separate region proposal algorithms.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Faster R-CNN: what was the idea? (abstract)::A Region Proposal Network (RPN): a fully convolutional network that shares the full-image convolutional features with the detection network and predicts object bounds and objectness scores at each position. The RPN and Fast R-CNN are merged into one network by sharing convolutional features.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Faster R-CNN: what did it find? (abstract)::With VGG-16 the whole detection system runs at 5 frames per second on a GPU, including all steps, with state-of-the-art accuracy on PASCAL VOC 2007, 2012 and MS COCO using only 300 proposals per image. It was the foundation of 1st-place entries in several tracks of the ILSVRC and COCO 2015 competitions.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Faster R-CNN: what is its limit?::5 frames per second is fast for its accuracy but not real time for video. YOLO ran faster with a different design (45 frames per second in its abstract).

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Faster R-CNN: reading trail?::It came from Fast R-CNN and led to Mask R-CNN. It made proposals nearly cost-free and started two-stage detectors.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How much of this paper did the note read?::Abstract only; the paper body was not read. Authors marked "from memory" in the note were not checked.
