---
tags: [lecture, course, machine-learning, evaluation, bias-variance, Andrew-Ng]
course: Machine Learning Specialization
lecture: Course 2, Week 3
lecturer: Andrew Ng
term: self-paced
source: repo notes.pdf (older course notes) and two saved lab notebooks; no lecture video
status: growing
---

# Machine Learning Specialization, Course 2, Week 3: advice for applying machine learning

> **In one line:** How to tell whether a model is good, why you need three data splits, how to diagnose high bias versus high variance, and what to do about each. (The title is my own label.)

## What the notes and labs cover
1. **Deciding what to try next (notes).** Options: more training examples, fewer features, more features, polynomial features, a smaller or bigger regularization strength. The point of the week is to pick among them using evidence, not guesses.
2. **Evaluating a model (notes).** Split the data. Fit on the training set, then measure error on a test set. For regression use squared error; for classification use the fraction misclassified. See [[Evaluation Metrics]].
3. **Train, validation and test sets (notes).** Choosing a setting (such as the polynomial degree) with the test set makes the test error too optimistic. The notes use three sets, for example 60% train, 20% cross-validation, 20% test: choose with the validation set, report with the test set. See [[Cross-Validation]].
4. **Bias and variance (notes).** High bias (underfit): training error and validation error are both high. High variance (overfit): training error is low and validation error much higher. A larger regularization strength fixes variance; a smaller one fixes bias. See [[Bias-Variance]], [[Overfitting]], [[Regularization]].
5. **Learning curves (notes).** Plot the errors against the training-set size. If the model has high bias, more data will not help much; with high variance it likely will. See [[Learning Curve]].
6. **Neural networks (notes).** A smaller network tends to underfit and is cheaper; a larger one tends to overfit and costs more, and regularization is the remedy. One hidden layer is a reasonable default.
7. **Skewed classes and error metrics (notes).** If only 0.5% of patients have a disease, a model that says "no disease" every time is 99.5% accurate and useless. Use precision (of the predicted positives, how many are right) and recall (of the real positives, how many were found), and trade them off with the decision threshold. See [[Precision]], [[Recall]], [[F1 Score]], [[Confusion Matrix]].
8. **Labs (notebooks).** *Model evaluation and selection:* split 50 examples into train, validation and test sets, fit linear and polynomial models, scale features. *Diagnosing bias and variance:* compare the training error with a baseline, then try more polynomial features or extra features to fix high bias.

## Key ideas to remember
- Judge a model on data it did not train on, and use a separate validation set to choose settings ([[Cross-Validation]]).
- Compare training error with a baseline: far above means high bias; validation far above training means high variance ([[Bias-Variance]]).
- Learning curves show whether more data would help ([[Learning Curve]]).
- Plain accuracy misleads when one class is rare; use precision, recall and F1 ([[Evaluation Metrics]]).

## Papers named in the sources
- None. **Transfer learning** is listed in the repo README's key concepts, but I did not find text on it in the sources I read; see [[Transfer Learning]] for the concept note.

## Questions I still have
- How should the baseline be set when no human benchmark exists? (The lab uses a chosen baseline value; see the notebook.)
- When the classes are skewed, how does one choose the threshold? The notes show the trade-off in words; see [[Precision-Recall Curve]] for the curve.

## What this note is built from
- **No lecture video or transcript.** The repo's `Week 3/notes.pdf` (the older course's "Advice for Applying Machine Learning" and "Machine Learning System Design" notes) and two lab notebooks (markdown and saved outputs). I reran both notebooks in a scratch copy. The first needed one edit there (`import utils` changed to `import libs.utils as utils`, because the file sits in `libs/`; the repo copy as saved would fail at that line). The first lab splits its 50 examples into 30 training, 10 validation and 10 test, which is the 60/20/20 rule.
- The 60/20/20 split and the 0.5% cancer example are from those notes, as examples, not rules.

## Course
- Part of [[Machine Learning Specialization]]. Lecture before: [[Machine Learning Specialization Course 2 Week 2]]. Lecture after: [[Machine Learning Specialization Course 2 Week 4]].
- Concept notes this week feeds: [[Evaluation Metrics]], [[Cross-Validation]], [[Bias-Variance]], [[Learning Curve]], [[Overfitting]], [[Regularization]], [[Precision]], [[Recall]], [[F1 Score]], [[Confusion Matrix]], [[Transfer Learning]]

## Flashcards
#flashcards/courses

#flashcards/courses/concept #flashcards/topic/course-mls-2 Why use three data splits?::Choosing a setting (such as the polynomial degree) with the test set makes the test error too optimistic. For example 60% train, 20% cross-validation, 20% test: choose with the validation set, report with the test set.

#flashcards/courses/concept #flashcards/topic/course-mls-2 How many examples go to train, validation and test when 50 examples are split 60/20/20?::30 for training, 10 for validation and 10 for test.

#flashcards/courses/concept #flashcards/topic/course-mls-2 How do you diagnose high bias against high variance?::High bias (underfit): training error and validation error are both high. High variance (overfit): training error is low and validation error much higher. A larger regularization strength fixes variance and a smaller one fixes bias.

#flashcards/courses/concept #flashcards/topic/course-mls-2 What do learning curves tell you?::Plot errors against training-set size. With high bias more data will not help much. With high variance it likely will.

#flashcards/courses/concept #flashcards/topic/course-mls-2 How does neural-network size relate to bias and variance?::A smaller network tends to underfit and is cheaper. A larger one tends to overfit and costs more, and regularization is the remedy. One hidden layer is a reasonable default.

#flashcards/courses/concept #flashcards/topic/course-mls-2 Why is plain accuracy misleading with skewed classes?::If only 0.5% of patients have a disease, a model that always says "no disease" is 99.5% accurate and useless. Use precision (of predicted positives, how many are right) and recall (of real positives, how many were found), and trade them off with the decision threshold.
