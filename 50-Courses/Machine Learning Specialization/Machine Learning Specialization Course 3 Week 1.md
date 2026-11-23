---
tags: [lecture, course, machine-learning, unsupervised-learning, clustering, Andrew-Ng]
course: Machine Learning Specialization
lecture: Course 3, Week 1
lecturer: Andrew Ng
term: self-paced
source: repo notes.pdf (older course notes); no lecture video; no lab notebook
status: seed
---

# Machine Learning Specialization, Course 3, Week 1: unsupervised learning (clustering)

> **In one line:** Learning from data with no labels: grouping similar examples with k-means. (The title is my own label.)

## What the notes cover
1. **Unsupervised learning (notes).** There is a set of examples but no answers to learn from; the job is to find structure. Uses named: market segmentation, social-network analysis, organising computer clusters, astronomical data. See [[Unsupervised Learning]].
2. **The k-means loop (notes).** Pick $K$ starting centres from the data, give each example to its nearest centre, move each centre to the mean of its examples, repeat. See [[K-Means]].
3. **What it minimises (notes).** The average squared distance from each example to its centre, called the distortion. It can only go down or stay the same each step.
4. **Starting points (notes).** Start the centres at $K$ distinct training examples; because k-means can end in a bad local answer, run it many times (the notes suggest this strongly for $K<10$) and keep the lowest distortion.
5. **Choosing $K$ (notes).** The elbow method (plot distortion against $K$ and look for where it flattens), noting the curve often has no clear elbow, or pick the $K$ that helps the next task most.
6. **Not covered by these notes:** the README lists anomaly detection as a week 1 topic, but the notes file has no text on it, so there is no anomaly detection here.

## Key ideas to remember
- With no labels, you ask the data to organise itself ([[Unsupervised Learning]], [[Clustering]]).
- k-means alternates "assign to nearest centre" and "move centre to the mean", and its score never gets worse ([[K-Means]]).
- Its result depends on the start, so restart several times and keep the best.
- $K$ is your choice, and the data does not always tell you the right one.

## Papers named in the sources
- None. The notes link a StackExchange discussion of k-means drawbacks (not read).

## Questions I still have
- How should k-means be changed for groups that are not round? (See [[DBSCAN]], [[Gaussian Mixture Model]].)
- What does anomaly detection look like in the specialization? The sources I have do not say.

## What this note is built from
- **No lecture video, no transcript, no lab notebook for this week.** The repo's `Week 1/notes.pdf`, which is the older course's "Clustering" notes (and also holds its first dimensionality-reduction notes, used in Week 2 here). The README's list for Week 1 (k-means, anomaly detection) is not matched by the notes file.
- The numbers and code in [[K-Means]] are my own and were run.
- Status is `seed`: one source, no lecture.

## Course
- Part of [[Machine Learning Specialization]]. Lecture before: [[Machine Learning Specialization Course 2 Week 4]]. Lecture after: [[Machine Learning Specialization Course 3 Week 2]].
- Concept notes this week feeds: [[K-Means]], [[Clustering]], [[Unsupervised Learning]]

## Flashcards
#flashcards/courses

#flashcards/courses/concept #flashcards/topic/course-mls-3 What is the k-means loop?::Pick $K$ starting centres from the data, give each example to its nearest centre, move each centre to the mean of its examples, and repeat.

#flashcards/courses/concept #flashcards/topic/course-mls-3 What does k-means minimise?::The average squared distance from each example to its centre (the distortion). It can only go down or stay the same each step.

#flashcards/courses/concept #flashcards/topic/course-mls-3 How should you start k-means?::Start the centres at $K$ distinct training examples. Because k-means can end in a bad local answer, run it many times (strongly advised for $K<10$) and keep the lowest distortion.

#flashcards/courses/concept #flashcards/topic/course-mls-3 How do you choose $K$ in k-means?::The elbow method (plot distortion against $K$ and look for where it flattens), though the curve often has no clear elbow. Or pick the $K$ that helps the next task most.

#flashcards/courses/concept #flashcards/topic/course-mls-3 Which uses of unsupervised learning are common?::Market segmentation, social-network analysis, organising computer clusters and astronomical data.
