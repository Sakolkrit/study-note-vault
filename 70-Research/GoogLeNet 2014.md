---
tags: [paper, vision, CNN, ImageNet]
link: https://arxiv.org/abs/1409.4842
authors: Szegedy et al. (first author from memory)
year: 2014
status: abstract-read
field: image classification
---

# GoogLeNet 2014

> **In one line for a 12-year-old:** Instead of choosing one filter size, let the network look with several sizes at once and keep what is useful, while keeping the total work the same.

## The problem
Get a better picture classifier without a larger computing bill.

## What people did before
Plain stacks that grew deeper and wider, such as [[VGG 2014]].

## The idea
An architecture codenamed Inception: a carefully crafted design that increases depth and width while keeping the computational budget constant. The abstract says the decisions were based on the Hebbian principle and the intuition of multi-scale processing. The ILSVRC 2014 entry was GoogLeNet, 22 layers deep.

## What they found
It set the new state of the art for classification and detection in ILSVRC 2014 (abstract). The abstract gives no error figure; I did not read the body.

## Why it mattered
It showed that cleverly shaped blocks can beat just adding size. The idea of several branches inside one block was reused in later networks (from memory).

## Limits and open questions
Complicated hand-designed blocks (from memory). [[ResNet 2015]] reached much greater depth with a very simple shortcut.

## Concepts used
[[Convolutional Neural Network]], [[Convolution]], [[Image Classification]], [[VGG]]

## Reading trail
- **Came from:** [[VGG 2014]] (same contest, same year)
- **Led to:** [[ResNet 2015]] and [[FCN 2014]]

## How much I read
Abstract read. Authors written as "(from memory)" were not checked. Numbers marked "abstract" come from the arXiv abstract page. Not opened in Obsidian.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-vision GoogLeNet (Szegedy et al., 2014, arXiv 1409.4842): what problem did it attack?::Getting a better picture classifier without a larger computing bill.

#flashcards/research/concept #flashcards/topic/research-vision What did people do before GoogLeNet?::Plain stacks that grew deeper and wider, such as VGG.

#flashcards/research/concept #flashcards/topic/research-vision GoogLeNet: what was the idea? (abstract)::An architecture codenamed Inception: a carefully crafted design that increases depth and width while keeping the computational budget constant, based on the Hebbian principle and the intuition of multi-scale processing. The ILSVRC 2014 entry was GoogLeNet, 22 layers deep. In plain words: look with several filter sizes at once and keep what is useful.

#flashcards/research/concept #flashcards/topic/research-vision GoogLeNet: what did it find? (abstract)::A new state of the art for classification and detection in ILSVRC 2014. The abstract gives no error figure.

#flashcards/research/concept #flashcards/topic/research-vision GoogLeNet: what is its limit?::Complicated hand-designed blocks (from memory). ResNet reached much greater depth with a very simple shortcut.

#flashcards/research/concept #flashcards/topic/research-vision GoogLeNet: reading trail?::It came from VGG (same contest, same year) and led to ResNet and FCN. The several-branches-in-one-block idea was reused in later networks (from memory).

#flashcards/research/concept #flashcards/topic/research-vision How much of this paper did the note read?::Abstract only; the paper body was not read. Authors marked "from memory" in the note were not checked.
