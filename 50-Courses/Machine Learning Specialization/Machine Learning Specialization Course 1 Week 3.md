---
tags: [lecture, course, machine-learning, logistic-regression, regularization, Andrew-Ng]
course: Machine Learning Specialization
lecture: Course 1, Week 3
lecturer: Andrew Ng
term: self-paced
source: YouTube playlist videos 31 to 41 (captions as text)
status: growing
---

# Machine Learning Specialization, Course 1, Week 3: from numbers to categories

> **In one line:** Why straight-line regression is a poor classifier, how logistic regression fixes that with a sigmoid and a new cost, where its decision boundary sits, and how overfitting is tamed with regularization. (The title is my own label.)

## What the lecture covers
1. **Classification and why linear regression fails (video 31).** With answers that are only 0 or 1, one idea is to fit a line and call everything above 0.5 a "1". The video shows that adding one far-away example on the right tilts the line and moves the dividing point, although that example should not change the classification of the others. The video points to an optional lab where this can be tried. See [[Classification]].
2. **Logistic regression (video 32).** Compute a score $z=\vec w\cdot\vec x+b$, then pass it through the **sigmoid** function, which squeezes any number into 0 to 1. The output (for example 0.7) is read as a chance that the label is 1; the label itself is always 0 or 1. See [[Logistic Regression]].
3. **Decision boundary (video 33).** The model predicts 1 when $z\ge0$, so the boundary is the set where $z=0$. With extra polynomial features the boundary can be a circle or another curve. See [[Decision Boundary]] and [[Feature Engineering]].
4. **Cost function (video 34).** The squared-error cost used for linear regression is not a good choice here. A different **loss** for one example is used, which punishes confident wrong answers heavily; the **cost** is its average over the training set. See [[Loss Function]].
5. **A simpler way to write it (video 35).** Because $y$ is only 0 or 1, the two cases of the loss merge into one formula.
6. **Gradient descent for logistic regression (video 36).** The update looks the same as for linear regression, but $f$ is now the sigmoid of the score. The video also points out scikit-learn as a ready-made implementation. See [[Gradient Descent]].
7. **Overfitting (video 37).** A model that is too simple underfits (high bias); one that is too wiggly fits the training data but fails on new data (high variance). See [[Overfitting]] and [[Bias-Variance]].
8. **Ways to fight overfitting (video 38).** Collect more data, use fewer features, or use **regularization**. See [[Regularization]].
9. **The regularized cost (video 39).** Add a penalty for large weights, scaled by a number $\lambda$ (lambda). Too small a $\lambda$ still overfits; an enormous one flattens the model to a horizontal line, which underfits. By convention the bias $b$ is not penalised, and the penalty is divided by $2m$ (twice the number of examples).
10. **Regularized linear and logistic regression (videos 40 and 41).** The gradient updates get one extra term for the weights; again the logistic version looks like the linear one. The last video points ahead to neural networks as the next course. See [[Neural Network]].

## Key ideas to remember
- Logistic regression = a linear score passed through a sigmoid, read as a chance ([[Logistic Regression]]).
- The prediction is class 1 when the score is at least 0, so the boundary is where the score is 0 ([[Decision Boundary]]).
- A different cost is used for classification, and the gradient-descent update still looks like the linear-regression one ([[Loss Function]], [[Gradient Descent]]).
- Overfitting means doing well on training data and badly on new data; underfitting is being too simple ([[Overfitting]], [[Bias-Variance]]).
- Regularization shrinks the weights with a strength $\lambda$ you choose ([[Regularization]]).

## Papers named in the lecture
- None named in the captions I read.

## Questions I still have
- How do you choose $\lambda$ in practice? (The video says model selection comes later in the specialization; see [[Cross-Validation]].)
- How trustworthy are the chances logistic regression outputs? (See [[Calibration]]; not covered in these videos as far as I read.)

## What this note is built from
- Captions (as text) of playlist videos 31 to 41, read at their beginning and end with some middle sections searched, so small points may be missing.
- The repo's `Week 3/notes.pdf` as a cross-check (older $\theta$ notation; its equations match the videos' ideas, with different symbols).
- Not read: the repo's Week 3 notebooks. Not seen: slides, figures, labs. I did not verify the video's statement that scikit-learn has a ready-made version (it does: `LogisticRegression`, used in [[Iris Classifier Service]]).

## Course
- Part of [[Machine Learning Specialization]]. Lecture before: [[Machine Learning Specialization Course 1 Week 2]]. Lecture after: none yet (the playlist ends here).
- Concept notes this lecture feeds: [[Logistic Regression]], [[Decision Boundary]], [[Classification]], [[Overfitting]], [[Bias-Variance]], [[Regularization]], [[Loss Function]], [[Gradient Descent]], [[Feature Engineering]]

## Flashcards
#flashcards/courses

#flashcards/courses/concept #flashcards/topic/course-mls-1 ML Specialization Course 1 Week 3: what does it cover?::Classification and why linear regression fails, logistic regression, the decision boundary, the logistic cost, gradient descent for it, overfitting, and regularization.

#flashcards/courses/concept #flashcards/topic/course-mls-1 Why is straight-line regression a poor classifier?::With answers only 0 or 1, thresholding a line at 0.5 breaks: one far-away example on the right tilts the line and moves the dividing point, although it should not change the others' classification.

#flashcards/courses/concept #flashcards/topic/course-mls-1 How does logistic regression make a prediction?::Compute a score $z=\vec w\cdot\vec x+b$, pass it through the sigmoid, which squeezes any number into 0 to 1, and read the output (for example 0.7) as a chance that the label is 1.

#flashcards/courses/concept #flashcards/topic/course-mls-1 Where is the decision boundary?::The model predicts 1 when $z\ge0$, so the boundary is where $z=0$. With extra polynomial features it can be a circle or another curve.

#flashcards/courses/concept #flashcards/topic/course-mls-1 Why a new cost for logistic regression?::The squared-error cost is not a good choice here. A different loss per example punishes confident wrong answers heavily, and the cost is its average over the training set. Because $y$ is only 0 or 1, the two cases merge into one formula.

#flashcards/courses/concept #flashcards/topic/course-mls-1 Underfitting against overfitting?::Too simple underfits (high bias). Too wiggly fits the training data but fails on new data (high variance).

#flashcards/courses/concept #flashcards/topic/course-mls-1 What are the three ways to fight overfitting?::Collect more data, use fewer features, or use regularization.

#flashcards/courses/concept #flashcards/topic/course-mls-1 How does the regularized cost work?::Add a penalty for large weights, scaled by $\lambda$. Too small a $\lambda$ still overfits. An enormous one flattens the model to a horizontal line, which underfits. By convention the bias $b$ is not penalised and the penalty is divided by $2m$.

#flashcards/courses/concept #flashcards/topic/course-mls-1 How was the Week 3 note built?::From captions of playlist videos 31 to 41 (beginning and end with some middle sections searched) and the repo's `Week 3/notes.pdf`. The notebooks, slides, figures and labs were not read or seen. The video's claim that scikit-learn has a ready-made version was not verified in that note (it does: `LogisticRegression`).
