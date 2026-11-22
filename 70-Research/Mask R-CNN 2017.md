---
tags: [paper, vision, segmentation, detection]
link: https://arxiv.org/abs/1703.06870
authors: He et al. (first author from memory)
year: 2017
status: abstract-read
field: image segmentation
---

# Mask R-CNN 2017

> **In one line for a 12-year-old:** Take a box-drawing detector and add a second job: for every box it finds, also colour in the exact pixels of the object inside it.

## The problem
Instance segmentation: find every object in a picture and give each one its own exact mask.

## What people did before
[[Faster R-CNN 2015]], which gives a box and a class for each object but no mask.

## The idea
Extend Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition ([[Object Detection]], [[Image Segmentation]]).

## What they found
Simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. It shows top results in all three tracks of the COCO suite (instance segmentation, bounding-box object detection and person keypoint detection) and, without bells and whistles, outperforms all existing single-model entries on every task, including the COCO 2016 challenge winners (abstract). It also estimates human poses in the same framework.

## Why it mattered
A simple, general baseline for instance-level recognition, as the abstract hopes.

## Limits and open questions
5 frames per second is not real time (abstract number); the abstract says nothing about accuracy on thin or tiny objects.

## Concepts used
[[Object Detection]], [[Image Segmentation]], [[Convolutional Neural Network]]

## Reading trail
- **Came from:** [[Faster R-CNN 2015]]
- **Led to:** -

## How much I read
Abstract read. Authors written as "(from memory)" were not checked. Numbers marked "abstract" come from the arXiv abstract page. Not opened in Obsidian.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Mask R-CNN (He et al., 2017, arXiv 1703.06870): what problem did it attack?::Instance segmentation: find every object in a picture and give each its own exact mask.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation What did people do before Mask R-CNN?::Faster R-CNN, which gives a box and a class for each object but no mask.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Mask R-CNN: what was the idea? (abstract)::Extend Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Mask R-CNN: what did it find? (abstract)::Simple to train and adds only a small overhead to Faster R-CNN, running at 5 fps. Top results in all three COCO tracks (instance segmentation, bounding-box detection and person keypoint detection). Without bells and whistles it outperforms all existing single-model entries on every task, including the COCO 2016 challenge winners. It also estimates human poses in the same framework.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Mask R-CNN: what limit does the note give?::5 frames per second is not real time (abstract number). The abstract says nothing about thin or tiny objects.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation Mask R-CNN: reading trail?::It came from Faster R-CNN. The note lists no paper it led to.

#flashcards/research/concept #flashcards/topic/research-detection-segmentation How much of this paper did the note read?::Abstract only; the paper body was not read. Authors marked "from memory" in the note were not checked.
