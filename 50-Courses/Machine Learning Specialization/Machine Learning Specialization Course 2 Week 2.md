---
tags: [lecture, course, machine-learning, neural-network, backpropagation, Andrew-Ng]
course: Machine Learning Specialization
lecture: Course 2, Week 2
lecturer: Andrew Ng
term: self-paced
source: repo notes.pdf (older course notes) and five saved lab notebooks; no lecture video
status: growing
---

# Machine Learning Specialization, Course 2, Week 2: training a network

> **In one line:** How a network is trained: the cost for many outputs, backpropagation to get the gradients, more activation functions (ReLU, softmax) for multi-class problems, and the practical details (random start, checking gradients). (The title is my own label.)

## What the notes and labs cover
1. **Cost function for a network (notes).** The logistic-regression cost, summed over $K$ output units, plus a regularization term over all weight matrices (not the bias entries). See [[Loss Function]] and [[Regularization]].
2. **Backpropagation (notes).** After a forward pass, compute the output error (prediction minus label), pass it backwards layer by layer using the weights and the activation's slope, and add up each weight's contribution over all training examples; divide by the number of examples (and add the regularization piece) to get the gradient. See [[Backpropagation]].
3. **Gradient checking (notes).** Compare the backpropagation gradient with a numerical estimate made by nudging a weight by a small $\epsilon$ both ways (the notes use $10^{-4}$), and turn the check off before real training because it is slow. See [[Calculus]].
4. **Random initialisation (notes).** Starting all weights at zero fails, because every unit then gets the same update forever; start from small random numbers instead. See [[Neural Network]].
5. **Putting it together (notes).** Pick an architecture (input units = number of features, output units = number of classes, hidden units a matter of trade-off), initialise randomly, run forward and backward passes, check gradients, then use [[Gradient Descent]].
6. **ReLU lab (notebook).** The rectified linear unit $a=\max(0,z)$: a piecewise-linear activation whose bend lets a network "turn pieces on" only where they are needed. See [[Activation Function]].
7. **Softmax and multi-class labs (notebooks).** Softmax turns $N$ scores into $N$ probabilities that add to 1; its loss is the negative log of the probability given to the true class (cross-entropy). The TensorFlow lab trains a network on four blobs of points, and notes that `from_logits=True` is used when the output layer is linear and the loss applies the softmax itself. See [[Softmax]] and [[TensorFlow]].
8. **Derivatives and backprop labs (notebooks).** A derivative is how much the output changes per tiny change of the input; the lab checks it with a small nudge (for $J=w^2$ at $w=3$, a nudge of 0.001 gives a slope of about 6.001) and with SymPy. A computation graph breaks a formula into small steps; in the lab $a=11$, $J=121$, $\partial J/\partial a=22$ and $\partial J/\partial w=66$. See [[Backpropagation]] and [[Calculus]].

## Key ideas to remember
- Training = forward pass, cost, backward pass for gradients, then a gradient-descent step ([[Backpropagation]], [[Gradient Descent]]).
- Different outputs need different activations and losses: sigmoid for yes/no, softmax for many classes ([[Softmax]]).
- ReLU is the common hidden-layer activation ([[Activation Function]]).
- Never start all weights equal; check gradients when you write backpropagation yourself.

## Papers named in the sources
- A link to a Stanford note on initialisation (`nninitialization.pdf`) appears in the repo notes. Not read.
- **Adam** is listed in the repo README's key concepts, but I found no text about it in the sources I read; see [[Adam Optimizer]] for the concept note.

## Questions I still have
- Why ReLU instead of sigmoid in hidden layers? The ReLU lab explains the "turn on where needed" idea; speed and vanishing slopes are my own background knowledge, not from the sources read ([[Vanishing and Exploding Gradients]]).
- Why is $10^{-4}$ a good size for the nudge in gradient checking? The notes give the value and say much smaller values risk numerical problems, but not why.

## What this note is built from
- **No lecture video or transcript.** The repo's `Week 2/notes.pdf` (the older course's "Neural Networks: Learning" notes, which also include a tangential image-processing walkthrough I skipped) and five lab notebooks (markdown and saved outputs). I reran the five notebooks in a scratch copy; all ran without edits. The numbers quoted above (6.001 for the nudge test, $a=11$, $J=121$, 22 and 66) came out the same (the nudge gives 22.001 and 66.009, as expected for a small but non-zero step).
- The repo's README lists Adam, layer types and "choosing the right activation per layer" as the week's topics; the text I read covers the first two items only through the lab pages named above.

## Course
- Part of [[Machine Learning Specialization]]. Lecture before: [[Machine Learning Specialization Course 2 Week 1]]. Lecture after: [[Machine Learning Specialization Course 2 Week 3]].
- Concept notes this week feeds: [[Backpropagation]], [[Activation Function]], [[Softmax]], [[Loss Function]], [[Regularization]], [[Gradient Descent]], [[Adam Optimizer]], [[Calculus]], [[TensorFlow]]

## Flashcards
#flashcards/courses

#flashcards/courses/concept #flashcards/topic/course-mls-2 What does backpropagation compute?::After a forward pass, the output error (prediction minus label) is passed backwards layer by layer using the weights and the activation's slope. Each weight's contribution is summed over all examples and divided by the number of examples (plus the regularization piece) to give the gradient.

#flashcards/courses/concept #flashcards/topic/course-mls-2 What is gradient checking?::Compare the backpropagation gradient with a numerical estimate made by nudging a weight by a small $\epsilon$ both ways (a typical size is $10^{-4}$). Turn it off before real training because it is slow.

#flashcards/courses/concept #flashcards/topic/course-mls-2 Why not start all weights at zero?::Every unit then gets the same update forever. Start from small random numbers.

#flashcards/courses/concept #flashcards/topic/course-mls-2 What is ReLU?::$a=\max(0,z)$, a piecewise-linear activation whose bend lets a network turn pieces on only where they are needed. It is the common hidden-layer activation.

#flashcards/courses/concept #flashcards/topic/course-mls-2 What does softmax do and what is its loss?::Softmax turns $N$ scores into $N$ probabilities that add to 1. Its loss is the negative log of the probability given to the true class (cross-entropy). In TensorFlow, `from_logits=True` is used when the output layer is linear and the loss applies the softmax itself.

#flashcards/courses/concept #flashcards/topic/course-mls-2 For $J=w^2$ at $w=3$, what slope does a nudge of 0.001 give?::About 6.001, against the exact slope 6 (a small non-zero step gives a close estimate).
