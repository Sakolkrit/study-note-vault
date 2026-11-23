---
tags: [lecture, course, machine-learning, decision-tree, ensemble, Andrew-Ng]
course: Machine Learning Specialization
lecture: Course 2, Week 4
lecturer: Andrew Ng
term: self-paced
source: two saved lab notebooks and the repo README; the week's notes.pdf in the repo is empty; no lecture video
status: growing
---

# Machine Learning Specialization, Course 2, Week 4: decision trees

> **In one line:** How a decision tree picks its questions by information gain, how to feed it category columns with one-hot encoding, and how forests and boosted trees combine many trees. (The title is my own label.)

## What the labs and README cover
1. **Choosing a split by information gain (lab).** A node holds examples (in the lab, 10 animals, cat or not cat, with ear shape, face shape and whiskers). Compute the node's entropy, then the weighted entropy after a split; the **information gain** is the drop. The lab's saved output has an entropy of 1.0 at the root (5 cats out of 10), 0.72 after splitting on ear shape, and so a gain of about 0.28, the biggest of the three features. See [[Decision Tree]] and [[Entropy]].
2. **Recursion and stopping (lab).** Repeat the split on each child until a limit is hit: a maximum depth, a node with a single class, or a gain below a threshold.
3. **One-hot encoding (lab).** The heart-disease table has 5 category columns; `pd.get_dummies` turns them into 0/1 columns, taking 11 features to 20. See [[One-Hot Encoding]] and [[Pandas]].
4. **Trees in scikit-learn (lab).** The lab splits 918 examples into 734 train and 184 test and varies `min_samples_split` and `max_depth`. The lab text says test accuracy peaks at depth 3. In my rerun it is 0.8641 at depth 3 (train 0.8583, the tree the lab keeps), 0.8696 at depth 4 (the peak), and falls to 0.8098 for a fully grown tree whose train accuracy is 1.0 (overfitting). See [[Overfitting]].
5. **Random forest (lab).** Many trees, each on a random subset; the chosen forest (depth 8, 100 trees) scores 0.8913 on test in both the saved output and my rerun; its train accuracy is 0.9264 in the saved output and 0.9210 in my rerun (a library-version difference, as far as I can tell). See [[Random Forest]].
6. **XGBoost (lab).** Boosted trees are added one after another to reduce the remaining error. The lab stops early when the validation log-loss stops improving for 50 rounds (rounds 0 to 66 ran out of the 500 allowed; the lowest log-loss, 0.43936, was at round 17). It then refits with 17 trees and reports 0.9319 on train and 0.8533 on test, matching my rerun. See [[Ensemble Methods]].
7. **From the README only:** purity and entropy, information gain, one-hot encoding, tree ensembles (random forest, XGBoost) are the week's listed topics. I found no further text on them.

## Key ideas to remember
- A tree asks the question that best sorts the classes, scored by information gain ([[Decision Tree]], [[Entropy]]).
- Too deep a tree memorises the training set; limit the depth or the minimum node size ([[Overfitting]]).
- Category columns become several 0/1 columns ([[One-Hot Encoding]]).
- Many trees beat one, by averaging (forest) or correcting step by step (boosting) ([[Ensemble Methods]]).

## Papers named in the sources
- None named in the parts I read.

## Questions I still have
- In the lab, the forest has the best test accuracy of the three (0.8913), but one run on one split is a weak comparison. How much do these numbers vary with a different split?
- The lab text says 16 trees at the minimum and "66" rounds. The rerun shows the minimum at round 17 (counting from 0, so 18 trees) and 67 rounds (0 to 66), and the lab refits with `best_iteration` = 17 trees. So the lab's prose is off by one or two; its numbers are fine.

## What this note is built from
- **No lecture video or transcript, and the repo's Week 4 `notes.pdf` is a zero-byte file.** This note uses the markdown and saved outputs of two lab notebooks (`1_Decision_Trees`, `2_Tree_Ensemble`) and the repo README. I reran both labs in a scratch copy. Everything matched the saved outputs except the forest's train accuracy (above). Two edits were needed there: the boosting lab's `early_stopping_rounds` moved from `fit()` to the model constructor (newer xgboost 2.1.4 removed it from `fit()`, same behaviour), and the tree-drawing cell of the first lab needs the Graphviz program, which I do not have, so that cell failed (the entropy and information-gain cells before it ran and gave 1.0, 0.7219 and 0.2781).
- Status is `growing`, but the week is still thinly sourced: no lecture and no notes file.

## Course
- Part of [[Machine Learning Specialization]]. Lecture before: [[Machine Learning Specialization Course 2 Week 3]]. Lecture after: none.
- Concept notes this week feeds: [[Decision Tree]], [[Entropy]], [[One-Hot Encoding]], [[Random Forest]], [[Ensemble Methods]], [[Overfitting]], [[Pandas]]

## Flashcards
#flashcards/courses

#flashcards/courses/concept #flashcards/topic/course-mls-2 How does a decision tree pick its split?::By information gain: the node's entropy minus the weighted entropy after the split.

#flashcards/courses/concept #flashcards/topic/course-mls-2 Example: 10 animals, 5 of them cats. What is the root entropy, and the gain from a split whose weighted entropy is 0.7219?::Entropy 1.0 at the root. Gain $1-0.7219=0.2781$, the biggest of the three features tried (my rerun).

#flashcards/courses/concept #flashcards/topic/course-mls-2 When does a decision tree stop splitting?::At a maximum depth, a node with a single class, or a gain below a threshold.

#flashcards/courses/concept #flashcards/topic/course-mls-2 What does one-hot encoding do?::Turns a category column into 0/1 columns, one per category. In the heart-disease example `pd.get_dummies` turned 5 category columns into 0/1 columns, taking 11 features to 20.

#flashcards/courses/concept #flashcards/topic/course-mls-2 How does a decision tree's test accuracy change as it grows deeper (heart-disease data, 918 examples: 734 train, 184 test)?::Depth 3 gives 0.8641 test, depth 4 gives 0.8696 (the peak), and a fully grown tree falls to 0.8098 test with train accuracy 1.0, which is overfitting (my rerun).

#flashcards/courses/concept #flashcards/topic/course-mls-2 How did a random forest and XGBoost score on the heart-disease data?::Forest (depth 8, 100 trees): 0.8913 test. XGBoost: early stopping at lowest validation log-loss 0.43936 at round 17, refit with 17 trees gives 0.9319 train and 0.8533 test (my rerun; the forest's train accuracy was 0.9210 in my rerun and 0.9264 in the saved output).

#flashcards/courses/concept #flashcards/topic/course-mls-2 Why is a one-run comparison of a forest with a single tree weak evidence?::It is one run on one split. How much the numbers vary with a different split was not measured.
