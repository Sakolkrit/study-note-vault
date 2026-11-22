---
tags: [paper, vision, CNN, ImageNet]
link: not stored (NeurIPS 2012 proceedings, title: ImageNet Classification with Deep Convolutional Neural Networks; I could not verify a URL)
authors: Krizhevsky, Sutskever, Hinton
year: 2012
status: abstract-read
field: image classification
---

# AlexNet 2012

> **In one line for a 12-year-old:** A computer learned to look at pictures by stacking many layers of small pattern-finders, and trained on fast graphics chips. It beat everyone at a big picture-naming contest.

## The problem
Name what is in a photo (1000 kinds of things in the ImageNet contest).

## What people did before
Hand-made image features fed to a separate classifier, and smaller convolutional networks like LeNet ([[LeNet and AlexNet]]).

## The idea
A deep [[Convolutional Neural Network]] with 5 convolution layers and 3 dense layers, ReLU activations, [[Dropout]] and two GPUs for training.

## What they found
Won the 2012 ImageNet contest with a top-5 error of 15.3% against about 26% for the runner-up (Wikipedia, read). The abstract page lists about 60 million parameters and 500,000 neurons. Counted in this vault: the PyTorch layout has 61,100,840 parameters.

## Why it mattered
It started the deep-learning era in vision. After it, nearly every image system used deep networks.

## Limits and open questions
Dense layers hold 96% of the parameters but convolutions do 91.8% of the work (counted in [[LeNet and AlexNet]]). Very deep versions were hard to train, which [[ResNet]] fixed.

## Concepts used
[[Convolution]], [[Pooling]], [[Activation Function]], [[Dropout]], [[GPU]], [[Data Augmentation]]

## Reading trail
- **Came from:** LeNet-5 (1998)
- **Led to:** [[VGG 2014]]

## How much I read
Abstract page only; the paper body was not read. Authors written as "(from memory)" were not checked. Numbers marked "abstract" come from the arXiv abstract page. Not opened in Obsidian.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-vision AlexNet 2012 (Krizhevsky, Sutskever, Hinton, NeurIPS 2012): what problem did it attack?::Naming what is in a photo (1000 kinds of things in the ImageNet contest).

#flashcards/research/concept #flashcards/topic/research-vision AlexNet: what was the idea?::A deep convolutional network with 5 convolution layers and 3 dense layers, ReLU activations, dropout, and two GPUs for training.

#flashcards/research/concept #flashcards/topic/research-vision AlexNet: what did it find? (mark: Wikipedia and abstract page)::Won the 2012 ImageNet contest with top-5 error 15.3% against about 26% for the runner-up (Wikipedia, read). The abstract page lists about 60 million parameters and 500,000 neurons. The note counted 61,100,840 parameters in the PyTorch layout.

#flashcards/research/concept #flashcards/topic/research-vision What did people do before AlexNet?::Hand-made image features fed to a separate classifier, and smaller convolutional networks such as LeNet.

#flashcards/research/concept #flashcards/topic/research-vision AlexNet: where did its cost sit?::Dense layers hold 96% of the parameters but convolutions do 91.8% of the work (counted in the LeNet and AlexNet note).

#flashcards/research/concept #flashcards/topic/research-vision AlexNet: why did it matter, and what came from it?::It started the deep-learning era in vision. It came from LeNet-5 (1998) and led to VGG.

#flashcards/research/concept #flashcards/topic/research-vision How much of this paper did the note read?::Abstract only; the paper body was not read. The link was not stored, because a URL could not be verified. Authors marked "from memory" in the note were not checked.
