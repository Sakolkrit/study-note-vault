---
tags: [paper, vision, CNN, residual]
link: https://arxiv.org/abs/1512.03385
authors: He et al. (first author from memory)
year: 2015
status: abstract-read
field: image classification
---

# ResNet 2015

> **In one line for a 12-year-old:** Let every block of layers add a small correction to its input instead of rebuilding it. Then a network can be very deep without getting lost.

## The problem
Deeper networks are more difficult to train (first line of the abstract).

## What people did before
Plain stacks such as [[VGG 2014]] of 16 to 19 layers.

## The idea
Reformulate the layers as learning residual functions with reference to their inputs: the shortcut ([[Residual Connection]]).

## What they found
Residual nets with up to 152 layers, 8 times deeper than VGG but lower complexity. An ensemble reached 3.57% error on the ImageNet test set and won ILSVRC 2015 classification. Also 28% relative improvement on COCO detection, and wins on ImageNet detection and localisation, and COCO detection and segmentation (abstract).

## Why it mattered
Shortcuts became standard, including inside every [[Transformer]] block.

## Limits and open questions
Still local patterns in a fixed grid. Later work asked whether attention could replace convolution ([[Vision Transformer]]).

## Concepts used
[[Residual Connection]], [[Vanishing and Exploding Gradients]], [[Normalization]], [[ResNet]]

## Reading trail
- **Came from:** [[VGG 2014]]
- **Led to:** [[ViT 2020]], [[N-BEATS 2019]] (doubly residual blocks for forecasting)

## How much I read
Abstract read. Authors written as "(from memory)" were not checked. Numbers marked "abstract" come from the arXiv abstract page. Not opened in Obsidian.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-vision ResNet (He et al., 2015, arXiv 1512.03385): what problem did it attack? (first line of the abstract)::Deeper networks are more difficult to train.

#flashcards/research/concept #flashcards/topic/research-vision What did people do before ResNet?::Plain stacks such as VGG of 16 to 19 layers.

#flashcards/research/concept #flashcards/topic/research-vision ResNet: what was the idea?::Reformulate the layers as learning residual functions with reference to their inputs: the shortcut. Each block adds a small correction instead of rebuilding its input.

#flashcards/research/concept #flashcards/topic/research-vision ResNet: what did it find? (abstract)::Residual nets with up to 152 layers, 8 times deeper than VGG but lower complexity. An ensemble reached 3.57% error on the ImageNet test set and won ILSVRC 2015 classification. Also a 28% relative improvement on COCO detection, and wins on ImageNet detection and localisation, and COCO detection and segmentation.

#flashcards/research/concept #flashcards/topic/research-vision ResNet: what is its limit?::Still local patterns in a fixed grid. Later work asked whether attention could replace convolution (Vision Transformer).

#flashcards/research/concept #flashcards/topic/research-vision ResNet: why did it matter and what is its place in the reading trail?::Shortcuts became standard, including inside every Transformer block. It came from VGG and led to ViT.

#flashcards/research/concept #flashcards/topic/research-vision How much of this paper did the note read?::Abstract only; the paper body was not read. Authors marked "from memory" in the note were not checked.
