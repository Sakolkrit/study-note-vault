---
tags: [lecture, course, machine-learning, neural-network, Andrew-Ng]
course: Machine Learning Specialization
lecture: Course 2, Week 1
lecturer: Andrew Ng
term: self-paced
source: repo notes.pdf (older course notes) and three saved lab notebooks; no lecture video
status: growing
---

# Machine Learning Specialization, Course 2, Week 1: neural networks

> **In one line:** Why plain regression struggles with many features, how a neural network of layered units answers that, and how a forward pass computes its output. (The title is my own label.)

## What the notes and labs cover
1. **Why neural networks (notes).** Adding all squared (or cubed) feature combinations to a regression explodes: the notes count about 5,000 quadratic features for 100 inputs and about 3 million for 2,500 pixel inputs (a 50 by 50 image). A neural network is offered as the alternative for many features. See [[Neural Network]] and [[Feature Engineering]].
2. **A unit and a layer (notes).** A unit is the logistic-regression recipe: weighted sum of the inputs plus a bias, then a sigmoid. Layers of units feed each other; between the input and output come hidden layers. The matrix sizes follow the number of units. See [[Logistic Regression]] and [[Activation Function]].
3. **Forward pass, vectorised (notes).** A layer is computed as a matrix times the previous layer's numbers, then the activation. See [[Forward Propagation]] and [[Vectorization]].
4. **Examples: logic gates (notes).** Single units can make AND, OR and NOR; stacked in two layers they make XNOR, which no single unit can. See [[Forward Propagation]] and [[Decision Boundary]].
5. **Labs on neurons and layers (notebooks).**
   - *Neurons and Layers:* a unit with no activation matches Course 1's linear regression, and one with a sigmoid matches logistic regression.
   - *Coffee roasting with [[TensorFlow]]:* a two-layer network on 200 examples of temperature and duration, 3 units in layer 1 and 1 in layer 2 (9 and 4 parameters in the saved output). Data is normalised first ([[Feature Scaling]]).
   - *The same network with [[NumPy]]:* a hand-written dense layer for the forward pass only.

## Key ideas to remember
- A neural network is many logistic-regression-like units stacked in layers ([[Neural Network]]).
- The forward pass is repeated "matrix times numbers, then activation" ([[Forward Propagation]]).
- Hidden layers let a network learn patterns a single straight boundary cannot ([[Decision Boundary]]).
- With no activation between layers, the whole network collapses to one linear function ([[Activation Function]]).

## Papers named in the sources
- None named in the parts I read. The notes link a paper on weight initialisation (see Week 2), not read.

## Questions I still have
- The repo notes use an older course's notation ($\Theta$, $h_\Theta$, layers with a bias unit). How does the specialization's $W$, $b$ notation map onto it? ([[Forward Propagation]] uses the $W$, $b$ form.)
- How many hidden units are enough? The notes say "usually more the better, balanced with compute cost" (Week 2 notes).

## What this note is built from
- **No lecture video or transcript** was used for Course 2. This note comes from the repo's `Week 1/notes.pdf` (which is the older course's "Neural Networks: Representation" notes, not the specialization's own lecture notes) and from the markdown and saved outputs of three lab notebooks. I reran the three notebooks in a scratch copy (Python 3.9, TensorFlow 2.9.3); they ran without edits, and the parameter counts 9 and 4 (13 in total) came out the same. Weights and predictions from training differ from the saved outputs because training starts from random numbers.
- So the week grouping and topic labels follow the repo's README, and anything the specialization's lectures cover beyond this (for example TensorFlow model building in detail) is not here.

## Course
- Part of [[Machine Learning Specialization]]. Lecture before: [[Machine Learning Specialization Course 1 Week 3]]. Lecture after: [[Machine Learning Specialization Course 2 Week 2]].
- Concept notes this week feeds: [[Neural Network]], [[Forward Propagation]], [[Activation Function]], [[Logistic Regression]], [[TensorFlow]], [[NumPy]], [[Vectorization]], [[Decision Boundary]]

## Flashcards
#flashcards/courses

#flashcards/courses/concept #flashcards/topic/course-mls-2 ML Specialization Course 2 Week 1: what does it cover?::Why plain regression struggles with many features, how a neural network of layered units answers that, and how a forward pass computes its output.

#flashcards/courses/concept #flashcards/topic/course-mls-2 Why not just add all squared feature combinations to a regression?::It explodes: about 5,000 quadratic features for 100 inputs and about 3 million for 2,500 pixel inputs (a 50 by 50 image). A neural network is offered instead. (Check: $100\cdot101/2=5050$ and $2500\cdot2501/2\approx3.1$ million.)

#flashcards/courses/concept #flashcards/topic/course-mls-2 What is a unit and what is a layer?::A unit is the logistic-regression recipe: weighted sum of the inputs plus a bias, then a sigmoid. Layers of units feed each other, with hidden layers between input and output.

#flashcards/courses/concept #flashcards/topic/course-mls-2 How is the forward pass vectorised?::A layer is computed as a matrix times the previous layer's numbers, then the activation.

#flashcards/courses/concept #flashcards/topic/course-mls-2 Which logic gates can single units make, and which need two layers?::Single units make AND, OR and NOR. XNOR needs two layers, which no single unit can do.

#flashcards/courses/concept #flashcards/topic/course-mls-2 What do the Week 1 labs show?::Neurons and Layers: a unit with no activation matches linear regression and one with a sigmoid matches logistic regression. Coffee roasting with TensorFlow: a two-layer network on 200 examples of temperature and duration, 3 units in layer 1 and 1 in layer 2 (9 and 4 parameters, 13 in total). The same network with NumPy: a hand-written dense layer for the forward pass only.

#flashcards/courses/concept #flashcards/topic/course-mls-2 What happens with no activation between layers?::The whole network collapses to one linear function.

#flashcards/courses/concept #flashcards/topic/course-mls-2 How was the Course 2 Week 1 note built?::No lecture video or transcript. From the repo's `Week 1/notes.pdf` (the older course's "Neural Networks: Representation" notes) and the markdown and saved outputs of three lab notebooks, which were rerun in a scratch copy (Python 3.9, TensorFlow 2.9.3) without edits. The parameter counts 9 and 4 came out the same.
