---
tags: [lecture, course, machine-learning, supervised-learning, Andrew-Ng]
course: Machine Learning Specialization
lecture: Course 1, Week 1
lecturer: Andrew Ng
term: self-paced
source: YouTube playlist videos 1 to 20 (captions as text)
status: growing
---

# Machine Learning Specialization, Course 1, Week 1: the recipe

> **In one line:** What machine learning is, the two main kinds (supervised and unsupervised), and the first complete algorithm: fit a straight line with a cost function and gradient descent. (The title is my own label.)

## What the lecture covers
1. **Welcome and motivation (videos 1 and 2).** Everyday uses (web search, photo tagging, film recommendations) and the claim that machine learning lets computers do things that are hard to program by hand. See [[Machine Learning]].
2. **What machine learning is (video 3).** The definition attributed to Arthur Samuel: giving computers the ability to learn without being explicitly programmed, with his checkers program that played many games against itself. The video ends by saying the course's two main types are supervised and unsupervised.
3. **Supervised learning (videos 4 and 5).** Learn input-to-output mappings from examples that include the right answer. The two kinds are **regression** (predict a number, such as a house price) and **classification** (predict a category from a small set, such as a tumour being benign or malignant). The lecturer says in the video that most of the economic value created by machine learning today comes from supervised learning. See [[Supervised Learning]], [[Regression]], [[Classification]].
4. **Unsupervised learning (videos 6 and 7).** Data with no labels; the algorithm looks for structure. The main example is **clustering**; anomaly detection and dimensionality reduction are named as later topics. See [[Unsupervised Learning]], [[Clustering]], [[PCA]].
5. **Jupyter notebooks (video 8).** The lab tool: code and notes in a browser page. See [[Jupyter Notebook]].
6. **Linear regression model (videos 9 and 10).** A training set of (size, price) pairs and its notation; the model $f_{w,b}(x)=wx+b$ with parameters $w$ and $b$. See [[Linear Regression]].
7. **The cost function (videos 11 to 14).** Squared error averaged over the training set, written $J(w,b)$; the goal is to choose $w,b$ that make it small. The videos build intuition with a one-parameter version, then a 3D bowl and contour plots. See [[Loss Function]].
8. **Gradient descent (videos 15 to 18).** Repeatedly nudge each parameter against the slope of the cost, with step size set by the learning rate $\alpha$. Too small is slow; too big can fail to converge. See [[Gradient Descent]] and [[Learning Rate]].
9. **Putting it together (videos 19 and 20).** Gradient descent with the squared-error cost trains linear regression. The cost here is bowl-shaped ("convex"), so with a good learning rate it reaches the single best answer. A demo watches the line settle onto the data. See [[Convex Optimization]].

## Key ideas to remember
- Supervised learning learns from examples with answers; regression predicts numbers, classification predicts categories ([[Supervised Learning]]).
- A model has adjustable **parameters**; a **cost function** scores how wrong the model is over the training set ([[Loss Function]]).
- **Gradient descent** lowers the cost step by step; the **learning rate** sets the step size ([[Gradient Descent]], [[Learning Rate]]).
- For linear regression the cost is bowl-shaped, so gradient descent cannot get stuck in a wrong valley ([[Convex Optimization]]). That is not true for every model.
- This one recipe (model, cost, gradient descent) is reused by later models, including big ones ([[Linear Regression]], then [[Logistic Regression]] in week 3).

## Papers named in the lecture
- None named in the captions I read.

## Questions I still have
- The video says gradient descent can end in different valleys on a non-bowl cost. How do large models cope with that? (Taken up later in the specialization, according to the video.)
- How is the learning rate chosen in practice beyond trial and error? (Week 2 answers part of this.)

## What this note is built from
- Captions (as text) of playlist videos 1 to 20, which I read in full only at their beginning and end and skimmed in the middle, so smaller points may be missing.
- The repo's `Week 1/notes.pdf`, as a cross-check only (it uses older notation, $\theta$ and $h_\theta$).
- Not read: the repo's Week 1 notebook (`Gradient_Descent.ipynb`). Not seen: slides, figures, the optional labs.
- Timing numbers and the "99% of the value" style claims are the lecturer's own statements as captioned, not checked.

## Course
- Part of [[Machine Learning Specialization]]. Lecture before: none. Lecture after: [[Machine Learning Specialization Course 1 Week 2]].
- Concept notes this lecture feeds: [[Machine Learning]], [[Supervised Learning]], [[Unsupervised Learning]], [[Regression]], [[Classification]], [[Clustering]], [[Linear Regression]], [[Loss Function]], [[Gradient Descent]], [[Learning Rate]], [[Convex Optimization]], [[Jupyter Notebook]]

## Flashcards
#flashcards/courses

#flashcards/courses/concept #flashcards/topic/course-mls-1 ML Specialization Course 1 Week 1: what does it cover?::What machine learning is, supervised and unsupervised learning, the first complete algorithm (linear regression with a cost function and gradient descent), and Jupyter notebooks.

#flashcards/courses/concept #flashcards/topic/course-mls-1 Machine learning: what definition does the video attribute to Arthur Samuel?::Giving computers the ability to learn without being explicitly programmed, with his checkers program that played many games against itself.

#flashcards/courses/concept #flashcards/topic/course-mls-1 Supervised learning: what are regression and classification?::Supervised learning learns input-to-output mappings from examples that include the right answer. Regression predicts a number (a house price). Classification predicts a category from a small set (a tumour benign or malignant). The lecturer says most of the economic value of machine learning today comes from supervised learning.

#flashcards/courses/reverse #flashcards/topic/course-mls-1 Unsupervised learning here:::Data with no labels where the algorithm looks for structure. The main example is clustering. Anomaly detection and dimensionality reduction are named as later topics.

#flashcards/courses/concept #flashcards/topic/course-mls-1 Write the one-feature linear regression model. What are $w$ and $b$?::$f_{w,b}(x)=wx+b$ with parameters $w$ and $b$, trained on (size, price) pairs.

#flashcards/courses/concept #flashcards/topic/course-mls-1 What is the cost function in week 1?::Squared error averaged over the training set, $J(w,b)$. The goal is to choose $w,b$ that make it small.

#flashcards/courses/concept #flashcards/topic/course-mls-1 What does gradient descent do and what does the learning rate $\alpha$ do?::It repeatedly nudges each parameter against the slope of the cost, with step size set by $\alpha$. Too small is slow, too big can fail to converge.

#flashcards/courses/concept #flashcards/topic/course-mls-1 Why can gradient descent not get stuck for linear regression?::The squared-error cost is bowl-shaped (convex), so with a good learning rate it reaches the single best answer. That is not true for every model.

#flashcards/courses/concept #flashcards/topic/course-mls-1 How was the Week 1 note built?::From captions of playlist videos 1 to 20, read in full only at their beginning and end and skimmed in the middle, plus the repo's `Week 1/notes.pdf` as a cross-check (older notation). The Week 1 notebook, slides, figures and labs were not read or seen. The "99% of the value" style claims are the lecturer's own words as captioned, not checked.
