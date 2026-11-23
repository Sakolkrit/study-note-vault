---
tags: [lecture, course, machine-learning, linear-regression, Andrew-Ng]
course: Machine Learning Specialization
lecture: Course 1, Week 2
lecturer: Andrew Ng
term: self-paced
source: YouTube playlist videos 21 to 30 (captions as text)
status: growing
---

# Machine Learning Specialization, Course 1, Week 2: making it work

> **In one line:** Linear regression with many features, written as vector arithmetic that runs fast, plus the practical tricks that make gradient descent behave: scaling features, watching the cost, choosing the learning rate, and adding new features to fit curves. (The title is my own label.)

## What the lecture covers
1. **Multiple features (video 21).** Predict a price from several inputs (size, bedrooms and so on) with one weight per feature: $f_{\vec w,b}(\vec x)=\vec w\cdot\vec x+b$. See [[Linear Regression]].
2. **Vectorization (videos 22 and 23).** Write the sum as one array operation instead of a loop. It makes the code shorter and much faster, and lets the work run on parallel hardware such as a GPU. See [[Vectorization]], [[NumPy]], [[GPU]].
3. **Gradient descent for many features (video 24).** The same update, applied to every weight at once with vectors. See [[Gradient Descent]].
4. **Feature scaling (videos 25 and 26).** When features have very different ranges (house size in the thousands, bedrooms from 0 to 5), the cost contours are long ovals and gradient descent zigzags; rescaling to comparable ranges speeds it up a lot. Ways shown: divide by the maximum, and mean normalization (subtract the average, then divide by the range). See [[Feature Scaling]].
5. **Checking that it converges (video 27).** Plot the cost against the number of iterations (the learning curve); it should keep falling. An automatic test is "stop if the cost falls by less than a small number such as 0.001 in one step", but the lecturer prefers looking at the plot. See [[Gradient Descent]].
6. **Choosing the learning rate (video 28).** If the cost goes up and down, the rate is probably too big (or there is a bug). Try values roughly three times apart (0.001, 0.003, 0.01, 0.03, ...), and pick the largest that still decreases smoothly. See [[Learning Rate]].
7. **Feature engineering (video 29).** Make a new feature from old ones, for example land area from its width and depth. See [[Feature Engineering]].
8. **Polynomial regression (video 30).** Add $x^2$ or $x^3$ as features so a "linear" model draws a curve; then scaling matters even more. A practice lab implements linear regression. See [[Feature Engineering]] and [[Feature Scaling]].

## Key ideas to remember
- With many features, the prediction is a dot product, and the same code should be written with arrays rather than loops ([[Vectorization]]).
- Rescale features to similar ranges before gradient descent ([[Feature Scaling]]).
- Watch the learning curve; if the cost rises, lower the learning rate ([[Learning Rate]]).
- You can make a straight-line method fit curves by adding powers of the inputs ([[Feature Engineering]]).
- Choosing and building features is often as important as the algorithm.

## Papers named in the lecture
- None named in the captions I read.

## Questions I still have
- How do you pick which polynomial features to add without guessing? (The captions only say to see what fits; model selection comes later in the specialization, according to the lecture.)
- Does feature scaling change the final answer or only the speed? (My note [[Feature Scaling]] says: for plain linear regression, only the speed. I derived that, the video does not state it in the part I read.)

## What this note is built from
- Captions (as text) of playlist videos 21 to 30, read at their beginning and end with some middle sections searched, so small points may be missing.
- The repo's `Week 2/notes.pdf` as a cross-check (older $\theta$ notation and Octave-era text).
- Not read: the repo's Week 2 notebooks (`Vectorization.ipynb`, `Feature_Scaling_&_Learning_Rate.ipynb`, `Polynomial_Regression_&_Feature_Engineering.ipynb`, `Sklearn.ipynb`). Not seen: slides, figures, labs.

## Course
- Part of [[Machine Learning Specialization]]. Lecture before: [[Machine Learning Specialization Course 1 Week 1]]. Lecture after: [[Machine Learning Specialization Course 1 Week 3]].
- Concept notes this lecture feeds: [[Linear Regression]], [[Vectorization]], [[Feature Scaling]], [[Feature Engineering]], [[Learning Rate]], [[Gradient Descent]], [[NumPy]], [[GPU]]

## Flashcards
#flashcards/courses

#flashcards/courses/concept #flashcards/topic/course-mls-1 Write the multiple-feature linear model.::$f_{\vec w,b}(\vec x)=\vec w\cdot\vec x+b$, with one weight per feature.

#flashcards/courses/concept #flashcards/topic/course-mls-1 Why vectorize code?::Writing the sum as one array operation instead of a loop makes code shorter and much faster, and lets the work run on parallel hardware such as a GPU.

#flashcards/courses/concept #flashcards/topic/course-mls-1 Why does feature scaling help gradient descent?::When features have very different ranges the cost contours are long ovals and gradient descent zigzags. Rescaling to comparable ranges speeds it up a lot.

#flashcards/courses/concept #flashcards/topic/course-mls-1 Which two feature-scaling methods are common?::Divide by the maximum, and mean normalization (subtract the average, then divide by the range).

#flashcards/courses/concept #flashcards/topic/course-mls-1 How do you check that gradient descent converges?::Plot the cost against the number of iterations (the learning curve). It should keep falling. An automatic test is to stop when the cost falls by less than a small number such as 0.001 in one step, but the lecturer prefers looking at the plot.

#flashcards/courses/concept #flashcards/topic/course-mls-1 How do you choose the learning rate?::If the cost goes up and down the rate is probably too big (or there is a bug). Try values roughly three times apart (0.001, 0.003, 0.01, 0.03, ...) and pick the largest that still decreases smoothly.

#flashcards/courses/concept #flashcards/topic/course-mls-1 What is feature engineering and what is polynomial regression?::Feature engineering makes a new feature from old ones (land area from width and depth). Polynomial regression adds $x^2$ or $x^3$ as features so a "linear" model draws a curve, and then scaling matters even more.

#flashcards/courses/concept #flashcards/topic/course-mls-1 Does feature scaling change the final answer for plain linear regression?::Only the speed, according to my own derivation; the course material I read does not state it.
