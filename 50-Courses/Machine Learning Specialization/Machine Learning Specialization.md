---
tags: [course, machine-learning, supervised-learning, Andrew-Ng]
institution: DeepLearning.AI and Stanford Online (on Coursera)
lecturers: Andrew Ng
term: self-paced
source: Course 1 from YouTube playlist captions; Course 2 from a saved GitHub repo (one student's notes and lab notebooks, no video); Course 3 from the same repo (thin sources, no video)
status: growing
---

# Machine Learning Specialization

> **In one line:** Andrew Ng's beginner course on classic machine learning: predict numbers (linear regression), predict categories (logistic regression), and train both with gradient descent. Course 1 is written from the video captions; Course 2 is written from a saved repo only (no video); Course 3 is written from thin sources (an older notes file for one week, one partly runnable notebook each for two more, no video).

## Logistics
Only what the sources show; nothing is checked against the Coursera page.
- The saved repo's README names three courses: (1) Supervised Machine Learning: Regression and Classification, (2) Advanced Learning Algorithms, (3) Unsupervised Learning, Recommenders, Reinforcement Learning.
- The YouTube playlist you gave holds **41 videos, all from Course 1**: weeks 1 to 3, about 312 minutes of video (my estimate from the end time of each video's captions). Its video titles are all the same wording ("Machine Learning Specialization [Course 1, Week W, Lesson L]"), so topic labels below are my own.
- Playlist: https://www.youtube.com/playlist?list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI
- Saved repo (copied into this vault): `source-repo/` next to this note, from https://github.com/oyounis19/Coursera-Machine-Learning-Specialization. It is one learner's work (weekly `notes.pdf`, lab notebooks, assignment solutions and from-scratch projects), not official course material, and it has no licence file. It also holds that learner's certificate PDFs.
- Hours, dates and grading: not shown in the sources.

## Lecture map
The playlist is a set of short clips grouped by week, so each "lecture note" here is one **week**.

| # | Note | Playlist videos | Main topics (my label) | Concept notes it feeds |
|---|---|---|---|---|
| 1 | [[Machine Learning Specialization Course 1 Week 1]] | 1 to 20 | What machine learning is; supervised and unsupervised; linear regression with one feature; cost function; gradient descent | [[Machine Learning]], [[Supervised Learning]], [[Unsupervised Learning]], [[Regression]], [[Classification]], [[Clustering]], [[Linear Regression]], [[Loss Function]], [[Gradient Descent]], [[Learning Rate]], [[Convex Optimization]], [[Jupyter Notebook]] |
| 2 | [[Machine Learning Specialization Course 1 Week 2]] | 21 to 30 | Many features; vectorization; feature scaling; checking convergence; choosing the learning rate; feature engineering and polynomial regression | [[Linear Regression]], [[Vectorization]], [[Feature Scaling]], [[Feature Engineering]], [[Learning Rate]], [[NumPy]] |
| 3 | [[Machine Learning Specialization Course 1 Week 3]] | 31 to 41 | Classification and logistic regression; decision boundary; logistic cost function; overfitting; regularization | [[Logistic Regression]], [[Decision Boundary]], [[Classification]], [[Overfitting]], [[Bias-Variance]], [[Regularization]], [[Loss Function]], [[Gradient Descent]] |
| 4 | [[Machine Learning Specialization Course 2 Week 1]] | none (repo only) | Neural networks: units, layers, forward pass, logic gates | [[Neural Network]], [[Forward Propagation]], [[Activation Function]], [[TensorFlow]] |
| 5 | [[Machine Learning Specialization Course 2 Week 2]] | none (repo only) | Training a network: cost, backpropagation, ReLU, softmax, multi-class | [[Backpropagation]], [[Softmax]], [[Activation Function]], [[Loss Function]] |
| 6 | [[Machine Learning Specialization Course 2 Week 3]] | none (repo only) | Advice: splits, bias and variance, learning curves, skewed classes | [[Cross-Validation]], [[Bias-Variance]], [[Learning Curve]], [[Precision]], [[Recall]], [[F1 Score]] |
| 7 | [[Machine Learning Specialization Course 2 Week 4]] | none (repo only) | Decision trees, one-hot encoding, forests and boosting (labs only; notes file empty) | [[Decision Tree]], [[Entropy]], [[One-Hot Encoding]], [[Random Forest]], [[Ensemble Methods]] |
| 8 | [[Machine Learning Specialization Course 3 Week 1]] | none (repo only) | Unsupervised learning: k-means clustering | [[K-Means]], [[Clustering]], [[Unsupervised Learning]] |
| 9 | [[Machine Learning Specialization Course 3 Week 2]] | none (repo only) | Dimensionality reduction with PCA (recommenders listed in the README, no source) | [[PCA]], [[Singular Value Decomposition]], [[Eigenvalue]] |
| 10 | [[Machine Learning Specialization Course 3 Week 3]] | none (repo only) | Reinforcement learning: the Mars rover example (one small lab) | [[Reinforcement Learning]], [[Q-Learning]] |

## How the lectures fit together
1. **Week 1: the recipe.** A model (a straight line), a way to score it (the cost), and a way to improve it (gradient descent). Everything later reuses these three parts.
2. **Week 2: make the recipe work in practice.** More inputs, faster code, better-behaved training, and richer features that bend the line.
3. **Week 3: from numbers to categories.** The same recipe with a different model (the sigmoid of a line) and a different cost, then the main failure (overfitting) and its main cure (regularization).
4. **Course 2, weeks 1 and 2: stack the units.** Many logistic-regression-like units in layers make a neural network; backpropagation trains it with the same gradient-descent idea.
6. **Course 3: no labels, and rewards.** Grouping and compressing data without answers (k-means, PCA), then learning from rewards instead of answers. These weeks have the thinnest sources.
5. **Course 2, weeks 3 and 4: use it well.** Splits, bias and variance, learning curves and skewed classes tell you what to fix; decision trees and tree ensembles are the other main tool.

## Where the detail lives
Concept notes hold the four levels, equations and tested code. The week notes only summarise and link. The bridge from linear to logistic regression is written in [[Linear Regression]] and [[Logistic Regression]] already.

## Not covered or not checked
- **Course 3 rests on thin sources and has no video.** The repo's Week 1 `notes.pdf` is an older course's clustering and PCA notes (so Week 1 here is k-means and Week 2 is PCA). Recommender systems and anomaly detection are listed in the README but have no text anywhere in the repo, so they are not written. Week 3 (reinforcement learning) has only one five-cell notebook. Its numbers in this vault are my own computation, because the notebook cannot run.
- **Course 2 has no video behind it.** The DeepLearning.AI channel's only public playlist for this specialization is the Course 1 one. Course 2 is built from the repo's `notes.pdf` files, which are an older course's notes (neural networks in $\Theta$ notation), plus the lab notebooks. The Week 4 notes file is empty, so that week uses the labs alone. Adam, transfer learning, and layer types are listed in the repo README but I found no text on them in what I read.
- **The playlist is a partial copy of the real course.** For example, it jumps straight from the cost function to gradient descent with no optional labs. I do not know how it compares with the full Coursera content.
- **The repo's `notes.pdf` files are Andrew Ng's older (2011-era) course notes**, written with $\theta$ and $h_\theta(x)$ and mentioning Octave, while the videos use $w$, $b$ and $f_{w,b}(x)$ with Python. I read them only as a cross-check and followed the videos' notation.
- I did not watch the videos; I read their automatic captions as text (no slides, no figures, no on-screen labs). The captions can contain mistakes in terms and names.
- **Course 2's lab notebooks were rerun** in a scratch copy (all 12 ran; three needed environment fixes described in the Week 3 and Week 4 notes, and one drawing cell still needs Graphviz). **Course 3's two notebooks could not be rerun as saved:** `PCA.ipynb` needs `pca_utils.py` and `toy_dataset.csv`, which the repo lacks (only its lecture-example cells ran), and the reinforcement-learning notebook calls a function missing from `libs/utils.py`. Course 1's notebooks were not run. The code in the new concept notes is my own and was run.
