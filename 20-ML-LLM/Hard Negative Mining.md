---
tags: [hard-negative-mining, training, object-detection, SVM, imbalanced-data, ML]
status: growing
level: 4
tier: 2
---

# Hard Negative Mining

**Hard negative mining** is a way to train a classifier when there are far too many "not this" examples to use them all. You train on a small starting set, then look through the whole pile for the "not this" examples that the current classifier gets wrong (or scores highest), add those to the training set, and train again. The set of negatives stays small but contains the ones that matter.

## 🟢 1. Simple (age 12)
Imagine teaching a friend to spot cats in photos. You have 1 cat photo and 10,000 photos of other things. Most of the others (sky, trees, cars) are easy, and your friend never mixes them up with cats. A few (a fox, a toy tiger) are tricky. Instead of showing all 10,000, you start with a few at random, ask your friend to go through the rest, and pick out the ones they wrongly call a cat. You add those to the study pile. The pile stays small but is full of the tricky ones.

**Example:** a detector looks at 2,000 boxes in each picture, and only one or two hold an object. The other 1,998 are "negatives". The mistakes the detector makes on those 1,998 (a box that covers half a dog) are the hard negatives it must learn from.

## 🔵 2. How it works
1. **Collect the positives** (the real objects) and a **starting set of negatives**: a random sample, or boxes that overlap the objects a little.
2. **Train** the classifier on this small set (R-CNN uses one linear [[Support Vector Machine]] per class).
3. **Score** every negative example in the whole pile with the new classifier.
4. **Pick the hard ones:** negatives that get a high score (the classifier calls them objects) or that fall on the wrong side of the decision line.
5. **Add them** to the training set, **retrain**, and repeat until the set stops changing or the score stops improving.
6. **Why it works:** a linear SVM depends only on the examples near its decision line (the "support vectors"). Easy negatives far from the line change nothing, so leaving them out is free; the hard ones are the ones that would have become support vectors anyway.
7. **Where it appears:** the classifiers of [[R-CNN 2013]] (and the older part-based detectors), the Selective Search recognition system ([[Selective Search]]), and as an option inside [[Fast R-CNN 2015]], which replaced it by a simple rule.

No image: I did not look for a free picture of the method. The numbers below come from a real run.

## 🟠 3. Technical detail
Numbers come from `hardneg_demo.py` (Python 3.9.6, scikit-learn 1.1.2, CPU). It builds a two-feature toy problem: 300 positives, 60,000 easy negatives far from them and a tight cluster of hard negatives next to them (400 of them in the first case, 40 in the second, "rare"). It trains a linear SVM on (a) all negatives, (b) 600 random negatives, (c) random negatives of the final mined size, and (d) the mined set: start with 600 random negatives and for 4 rounds add the 300 negatives with the highest score. It reports average precision on a fresh test set of the same kind. Each case is repeated over 5 random seeds.

```python
"""Hard negative mining on a toy problem: 300 positives, ~60,000 negatives, mostly easy plus a small cluster of hard ones near the positives.
Two cases: 400 hard negatives (not rare) and 40 hard negatives (rare), each over 5 random seeds."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import average_precision_score
def pile(rng, n_pos, n_easy, n_hard):
    pos = rng.normal([2.0, 2.0], 0.7, (n_pos, 2))
    easy = rng.normal([-1.0, -1.0], 1.2, (n_easy, 2))                       # far from the positives
    hard = rng.normal([1.3, 0.9], 0.35, (n_hard, 2))                        # a tight cluster close to the positives
    return np.vstack([pos, easy, hard]), np.r_[np.ones(n_pos), np.zeros(n_easy + n_hard)]
def run(n_hard, seed, rounds=4, start=600, add=300):
    rng = np.random.default_rng(seed)
    Xtr, ytr = pile(rng, 300, 60000, n_hard); Xte, yte = pile(rng, 300, 60000, n_hard)
    pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
    fit = lambda idx: LinearSVC(C=1.0, max_iter=20000).fit(Xtr[idx], ytr[idx])
    ap = lambda m: average_precision_score(yte, m.decision_function(Xte))
    out = {"all": ap(fit(np.r_[pos, neg]))}
    cache = list(rng.choice(neg, start, replace=False)); out["random start"] = ap(fit(np.r_[pos, cache]))
    hard_in = lambda c: int((np.array(c) >= 300 + 60000).sum())             # indices of the hard cluster come last
    out["hard in start"] = hard_in(cache)
    for r in range(rounds):
        m = fit(np.r_[pos, cache]); s = m.decision_function(Xtr[neg]); seen = set(cache)
        cache += [i for i in neg[np.argsort(-s)] if i not in seen][:add]
    out["mined"] = ap(fit(np.r_[pos, cache])); out["mined size"] = len(cache); out["hard in mined"] = hard_in(cache)
    out["random same size"] = ap(fit(np.r_[pos, rng.choice(neg, len(cache), replace=False)]))
    return out
for n_hard in (400, 40):
    rs = [run(n_hard, s) for s in range(5)]
    print("\nhard negatives in the pile: %d of 60,%03d" % (n_hard, n_hard))
    for k in ("all", "random start", "random same size", "mined"):
        v = [r[k] for r in rs]; print("  AP %-17s mean %.3f  (min %.3f, max %.3f)" % (k, np.mean(v), np.min(v), np.max(v)))
    print("  hard negatives in the random start set (per seed):", [r["hard in start"] for r in rs], "| in the mined set:", [r["hard in mined"] for r in rs], "| mined set size", rs[0]["mined size"])
```

I saw:
```

hard negatives in the pile: 400 of 60,400
  AP all               mean 0.856  (min 0.842, max 0.888)
  AP random start      mean 0.856  (min 0.842, max 0.888)
  AP random same size  mean 0.853  (min 0.837, max 0.886)
  AP mined             mean 0.856  (min 0.842, max 0.888)
  hard negatives in the random start set (per seed): [6, 4, 5, 1, 1] | in the mined set: [347, 348, 339, 345, 346] | mined set size 1800

hard negatives in the pile: 40 of 60,040
  AP all               mean 0.874  (min 0.865, max 0.890)
  AP random start      mean 0.872  (min 0.865, max 0.891)
  AP random same size  mean 0.872  (min 0.865, max 0.887)
  AP mined             mean 0.874  (min 0.865, max 0.890)
  hard negatives in the random start set (per seed): [0, 0, 0, 0, 0] | in the mined set: [36, 39, 40, 39, 37] | mined set size 1800
```

**What the run shows (mean over 5 seeds).**

| Hard negatives in the pile | AP with all 60,400 | AP random 600 (start) | AP random 1,800 | AP mined (1,800) | Hard ones in the random start | Hard ones in the mined set |
|---|---|---|---|---|---|---|
| 400 | 0.856 | 0.856 | 0.853 | 0.856 | 1 to 6 | 339 to 348 |
| 40 | 0.874 | 0.872 | 0.872 | 0.874 | 0 | 36 to 40 |

- **AP:** [[Mean Average Precision]] for a single class: the area under the precision-recall curve on the test set (higher is better).
- **Random 600 and random 1,800:** negatives drawn at random, at the size of the start and of the final mined set. **Mined:** the set after four rounds of adding the 300 highest-scoring negatives (1,800 of the 60,400 negatives, 3%).
- **Hard ones in the set:** how many of the planted hard negatives (indices of the cluster) ended up in the training set.

**Summary: what the run shows.** Mining does what it says: after four rounds it had found 339 to 348 of the 400 hard negatives (and 36 to 40 of the 40) using 3% of the pile, where a random set of the same size would hold about 12 (my arithmetic: 3% of 400) and the 600 random starters held 0 to 6. But it **did not raise the average precision in this toy**: mined, random and all-negatives sets all gave 0.856 (and 0.872 to 0.874 in the rare case). In this toy two features and a linear classifier have little to gain. Caution: a two-feature toy, one classifier, one mining rule (top-scoring negatives), 5 seeds, no tuning of the rounds. It shows the selection mechanism and that a 3% subset can match training on everything, not that mining always improves accuracy.

## 🔴 4. Deep / formal
No equation: the method is a loop around any classifier, and the rule is "add the negatives the current model scores highest".

**What the papers I read say.**
- **R-CNN** (section 2.3, Appendix B): "Since the training data is too large to fit in memory, we adopt the standard hard negative mining method"; it "converges quickly and in practice mAP stops increasing after only a single pass over all images". In Appendix B the authors compare their SVMs against the fine-tuned network's softmax and think the softmax was worse (50.9% against 54.2% mAP) partly because it "was trained on randomly sampled negatives rather than on the subset of 'hard negatives' used for SVM training" (their conjecture).
- **Selective Search recognition system** (draft, section 4): initial negatives are boxes with 20% to 50% overlap with a positive, "difficult negatives which are close to the positive examples"; then "for each negative image we add the highest scoring location", and the models converge in few iterations.
- **Fast R-CNN** (sections 2.3 and 5.4): the background examples are taken from proposals whose largest IoU is in [0.1, 0.5), and "the lower threshold of 0.1 appears to act as a heuristic for hard example mining". A post-hoc SVM with hard negative mining did slightly worse than the softmax (by 0.1 to 0.8 mAP), so the paper dropped the SVM stage. For dense boxes, SVMs with hard negative mining were worse still (49.3%).

I did not read the original part-based detector paper that popularised it, so the history before R-CNN is as these papers cite it.

**Edge cases and details:**
- **Duplicates and label noise:** a "hard negative" can be a mislabelled positive (an object nobody boxed); adding those teaches the model the wrong thing.
- **Stopping:** the loop can be capped at a fixed number of rounds or a cache size; I used 4 rounds.
- **Relation to other ideas:** focal-style losses and online hard example selection give the same emphasis without a separate retraining loop (general knowledge; I did not test them).

**Limits and trade-offs:**
- Each round scores the whole pile of negatives, which costs time; it saves memory, not necessarily compute.
- It needs a classifier that is cheap to retrain (a linear SVM); for deep networks the "mining" is usually folded into the mini-batch sampling instead (Fast R-CNN).
- The benefit is not guaranteed: my toy showed no accuracy gain over a random subset.

**What came before, and how this improved on it:**
- **Before:** train on all negatives (impossible when they do not fit in memory) or on a random sample of them.
- **Its limit:** a random sample of a pile that is 99.5% easy negatives contains almost none of the hard ones; in my rare case the random start held 0 of 40.
- **What changed:** iterate: train, find the false positives in the whole pile, add them, retrain.
- **How much better:** in my toy the mined set of 1,800 held 36 to 40 of the 40 hard negatives (and 339 to 348 of 400) against 0 and 1 to 6 for the random start, but the average precision stayed the same (0.874 and 0.856). The published figure is a speed and memory one (R-CNN: one pass is enough, and the pile fits in memory); I did not run R-CNN.

## Related
[[R-CNN 2013]] · [[Fast R-CNN 2015]] · [[Selective Search]] · [[Support Vector Machine]] · [[Object Detection]] · [[Intersection over Union]] · [[Mean Average Precision]] · [[Overfitting]] · [[Softmax]]

## Sources
- My own run: `hardneg_demo.py` (Python 3.9.6, scikit-learn 1.1.2).
- [[R-CNN 2013]] (sections 2.3 and Appendix B), [[Fast R-CNN 2015]] (sections 2.3 and 5.4) as read there, and the Selective Search draft by Uijlings et al. (section 4, read for [[Selective Search]]). The part-based detector paper that popularised the method was not read. The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/training Hard negative mining:::Training on a small set of "not this" examples and repeatedly adding the negatives the current model scores highest (its false positives) from the whole pile, then retraining.

#flashcards/ml/concept #flashcards/topic/training Why is hard negative mining needed in detection?::Almost all boxes in an image are background, so the negatives are too many to fit in memory and a random sample holds few of the confusing ones; mining keeps the set small but informative (R-CNN, section 2.3).

#flashcards/ml/concept #flashcards/topic/training What are the steps of hard negative mining?::Train on the positives plus a small starting set of negatives; score all negatives; add the highest-scoring (wrongly confident) ones; retrain; repeat until the set or the score stops changing.

#flashcards/ml/concept #flashcards/topic/training Why can easy negatives be left out without harming a linear SVM?::The SVM depends only on examples near its decision line (support vectors); easy negatives far from it change nothing, while hard negatives would have become support vectors (Selective Search draft, section 4).

#flashcards/ml/concept #flashcards/topic/training How did R-CNN and Selective Search choose their negatives?::R-CNN: proposals with IoU below 0.3 with every object of the class, plus mining; one pass is enough ("mAP stops increasing after only a single pass"). Selective Search: boxes overlapping a positive by 20 to 50%, then the highest-scoring location of each negative image.

#flashcards/ml/concept #flashcards/topic/training What did Fast R-CNN do instead of hard negative mining?::It sampled background RoIs with IoU in [0.1, 0.5), where the lower bound 0.1 "appears to act as a heuristic for hard example mining", and trained a softmax; post-hoc SVMs with mining were 0.1 to 0.8 mAP worse (paper, 2.3 and 5.4).

#flashcards/ml/example #flashcards/topic/training In my toy (300 positives, 60,000 easy negatives, 400 hard ones), how many hard negatives did 4 rounds of mining find, and what happened to average precision?::339 to 348 of the 400 in a set of 1,800 negatives (3% of the pile), against 1 to 6 in the 600 random starters; AP stayed 0.856 for mined, random and all-negatives sets (5 seeds; a two-feature toy).

#flashcards/ml/concept #flashcards/topic/training What are the limits of hard negative mining?::Each round scores the whole pile; it needs a classifier that is cheap to retrain; hard negatives can be unlabelled positives; and it does not always improve accuracy (my toy showed no gain over a random subset).

#flashcards/ml/cloze #flashcards/topic/training Hard negative mining adds the ==false positives== of the current model to the ==negative== set and ==retrains==.
