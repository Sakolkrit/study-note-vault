---
tags: [lecture, course, machine-learning, PCA, dimensionality-reduction, Andrew-Ng]
course: Machine Learning Specialization
lecture: Course 3, Week 2
lecturer: Andrew Ng
term: self-paced
source: repo notes.pdf (older course notes, dimensionality reduction part) and the PCA lab notebook; no lecture video
status: seed
---

# Machine Learning Specialization, Course 3, Week 2: dimensionality reduction (PCA)

> **In one line:** Squeezing many features into a few with principal component analysis, how to choose how many to keep, and when not to use it. (The title is my own label.) The README also lists recommender systems for this week; I have no source text for them.

## What the notes and the lab cover
1. **Why reduce features (notes).** Compression (less memory, faster training) and visualisation (plot in 2 or 3 dimensions). It reduces the number of features, not the number of examples.
2. **The idea of PCA (notes).** Find a line (or plane) so that projecting the data onto it loses as little as possible, measured as the average squared distance from each point to its projection. This is not linear regression: PCA measures the shortest distance to the line, not the vertical one, and has no target $y$. See [[PCA]].
3. **Steps (notes).** Subtract each feature's mean (and scale features that have different ranges); compute the covariance matrix; take its eigenvectors with a singular value decomposition; keep the first $k$ directions; the compressed data is the data times those directions. See [[Singular Value Decomposition]], [[Eigenvalue]], [[Feature Scaling]].
4. **Getting back (notes).** Multiplying the compressed data by the transposed directions gives an approximation of the original, not the original.
5. **Choosing $k$ (notes).** Pick the smallest $k$ that keeps 99% of the variance, i.e. the squared projection error divided by the total variation is at most 0.01. The singular values from the decomposition give this without trying every $k$.
6. **Advice (notes).** The common use is speeding up supervised learning on very many features. Fit PCA on the training set only and apply it to the validation and test sets. Do not use PCA to fight overfitting (it ignores $y$; [[Regularization]] is better). Try the full model without PCA first.
7. **PCA lab (notebook).** The lecture's example: six points, two features. The first principal component keeps 99.24% of the variance and the second the other 0.76%. The lab continues with a 3D to 2D view and a 500-example, 1000-feature table (see the "not run" section below).

## Key ideas to remember
- PCA finds the directions along which the data varies most and keeps the top few ([[PCA]]).
- Choose how many to keep from the share of variance they hold (99% is the notes' rule of thumb).
- Fit it on training data only, like [[Feature Scaling]].
- Reducing features can lose information a model needed; test before you rely on it.

## Papers named in the sources
- None.

## Questions I still have
- What do recommender systems (collaborative filtering, content-based filtering) look like in this course? I have no source for them.
- Is the 99% rule right for every task? The notes call it a rule of thumb.

## What this note is built from
- **No lecture video or transcript.** The repo's `Week 1/notes.pdf` (its dimensionality-reduction pages) and the markdown and saved outputs of `PCA.ipynb`.
- **I reran the PCA notebook in a scratch copy, but only part of it ran.** It imports a helper file `pca_utils` and a data file `toy_dataset.csv` that the repo does not contain, and needs `bokeh` and `plotly`. With those lines removed, 18 code cells ran and 20 failed. The lecture example ran and matched: explained variance 0.99244 and 0.00756, the six points projected to a line, and so on. The interactive widget, the 3D example and the 1000-feature section did not run.
- I also computed the same explained variance by hand with a singular value decomposition in NumPy and got 0.99244 and 0.00756.

## Course
- Part of [[Machine Learning Specialization]]. Lecture before: [[Machine Learning Specialization Course 3 Week 1]]. Lecture after: [[Machine Learning Specialization Course 3 Week 3]].
- Concept notes this week feeds: [[PCA]], [[Singular Value Decomposition]], [[Eigenvalue]], [[Feature Scaling]], [[Regularization]]

## Flashcards
#flashcards/courses

#flashcards/courses/concept #flashcards/topic/course-mls-3 ML Specialization Course 3 Week 2: what does it cover?::Dimensionality reduction with PCA: why reduce, the idea, the steps, getting back, choosing $k$, advice, and the PCA lab.

#flashcards/courses/concept #flashcards/topic/course-mls-3 Why reduce features?::Compression (less memory, faster training) and visualisation (plot in 2 or 3 dimensions). It reduces the number of features, not examples.

#flashcards/courses/concept #flashcards/topic/course-mls-3 What is the idea of PCA, and how is it not linear regression?::Find a line (or plane) so that projecting the data onto it loses as little as possible, measured as the average squared distance from each point to its projection. PCA measures the shortest distance to the line, not the vertical one, and has no target $y$.

#flashcards/courses/concept #flashcards/topic/course-mls-3 What are the steps of PCA?::Subtract each feature's mean (and scale features with different ranges), compute the covariance matrix, take its eigenvectors with a singular value decomposition, keep the first $k$ directions, and multiply the data by those directions.

#flashcards/courses/concept #flashcards/topic/course-mls-3 How do you get back from compressed data?::Multiply by the transposed directions. That gives an approximation of the original, not the original.

#flashcards/courses/concept #flashcards/topic/course-mls-3 How do you choose $k$?::The smallest $k$ that keeps 99% of the variance, so the squared projection error over the total variation is at most 0.01. The singular values give this without trying every $k$. The notes call 99% a rule of thumb.

#flashcards/courses/concept #flashcards/topic/course-mls-3 What advice do the notes give on using PCA?::Use it mainly to speed up supervised learning on very many features. Fit it on the training set only and apply it to validation and test sets. Do not use it to fight overfitting (it ignores $y$, and regularization is better). Try the full model without PCA first.

#flashcards/courses/concept #flashcards/topic/course-mls-3 What did the PCA lab's example give?::Six points, two features: the first principal component keeps 99.24% of the variance and the second 0.76% (0.99244 and 0.00756, which the note also got by hand with an SVD in NumPy).

#flashcards/courses/concept #flashcards/topic/course-mls-3 How much of the PCA notebook ran?::Only part. It imports `pca_utils` and `toy_dataset.csv`, which the repo does not contain, and needs `bokeh` and `plotly`. With those lines removed, 18 code cells ran and 20 failed. The lecture example matched. The interactive widget, 3D example and 1000-feature section did not run. Recommender systems are listed in the README but no source text exists.
