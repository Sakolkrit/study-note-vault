---
tags: [smooth-l1-loss, loss-function, regression, object-detection, huber-loss, fast-rcnn, ML]
status: growing
level: 4
tier: 2
---

# Smooth L1 Loss

**Smooth L1 loss** is a way to score how wrong a predicted number is. For a small error it acts like a squared error (gentle, smooth near zero). For a large error it acts like an absolute error (the penalty grows only in a straight line). Object detectors use it to train the four box offsets, because a few very wrong boxes should not drown out everything else.

## 🟢 1. Simple (age 12)
A teacher marks guesses of someone's height. If you are off by a tiny bit, you lose a tiny amount and the teacher barely cares. If you are off by a huge amount, a harsh teacher would punish you enormously, so that one wild guess would decide your whole grade. A fair teacher punishes big mistakes more than small ones, but not without limit. That teacher is smooth L1.

**Example:** three guesses are off by 0.1, 0.2 and 0.1 centimetres, and one is off by 5. A harsh teacher's total is dominated by the 5. The fair teacher still notices the 5, but it cannot take over.

## 🔵 2. How it works
1. The detector predicts four numbers per box (the offsets of an [[Anchor Box]]) and knows the true four numbers. The difference is the error.
2. Look at the size of each error.
3. **If the error is small** (under 1), the loss is half the error squared: tiny errors cost almost nothing, and the loss curve is smooth near zero, so the learning step shrinks as the box gets close ([[Gradient Descent]]).
4. **If the error is large** (1 or more), the loss is the error minus one half: it grows in a straight line, and its slope is capped at 1.
5. Add the four losses (the four offsets) to get the box loss for that box.
6. The cap on the slope is the point: one badly placed box cannot send a huge gradient through the network.

No image: I found no free picture of this loss on Wikimedia Commons, so I show the numbers from a real run below.

## 🟠 3. Technical detail
Numbers come from `smoothl1_demo.py` (Python 3.9.6, PyTorch 2.8.0, CPU). The script writes the loss by hand, checks it against `torch.nn.functional.smooth_l1_loss`, and compares it with L2 and L1 on the same errors.

```python
"""Smooth L1 against L1 and L2: values, gradients, and one outlier."""
import torch
import torch.nn.functional as F

def smooth_l1(x):                      # 0.5 x^2 if |x| < 1, else |x| - 0.5
    ax = x.abs()
    return torch.where(ax < 1, 0.5 * x ** 2, ax - 0.5)

e = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0], requires_grad=True)
for name, f in (("L2 (0.5 x^2)", lambda v: 0.5 * v ** 2), ("L1 (|x|)", lambda v: v.abs()), ("smooth L1", smooth_l1)):
    loss = f(e); g, = torch.autograd.grad(loss.sum(), e)
    print("%-13s loss" % name, [round(v, 3) for v in loss.tolist()], "gradient", [round(v, 3) for v in g.tolist()])
print("matches torch:", torch.allclose(smooth_l1(e), F.smooth_l1_loss(e, torch.zeros(5), reduction="none", beta=1.0)))

# one bad label: three small errors and one outlier of 5; how much of the total pull (sum of gradients) comes from the outlier
err = torch.tensor([0.1, 0.2, 0.1, 5.0], requires_grad=True)
for name, f in (("L2", lambda v: 0.5 * v ** 2), ("L1", lambda v: v.abs()), ("smooth L1", smooth_l1)):
    g, = torch.autograd.grad(f(err).sum(), err)
    print("%-9s total loss %.3f | outlier pull %.1f of %.1f = %.1f%%" % (name, f(err).sum().item(), g[-1].item(), g.sum().item(), 100 * g[-1].item() / g.sum().item()))
```

I saw:
```
L2 (0.5 x^2)  loss [0.005, 0.125, 0.5, 2.0, 12.5] gradient [0.1, 0.5, 1.0, 2.0, 5.0]
L1 (|x|)      loss [0.1, 0.5, 1.0, 2.0, 5.0] gradient [1.0, 1.0, 1.0, 1.0, 1.0]
smooth L1     loss [0.005, 0.125, 0.5, 1.5, 4.5] gradient [0.1, 0.5, 1.0, 1.0, 1.0]
matches torch: True
L2        total loss 12.530 | outlier pull 5.0 of 5.4 = 92.6%
L1        total loss 5.400 | outlier pull 1.0 of 4.0 = 25.0%
smooth L1 total loss 4.530 | outlier pull 1.0 of 1.4 = 71.4%
```

**Loss and gradient for errors of 0.1 to 5** (from the first three lines of the output; the three losses written as $0.5x^2$, $|x|$ and smooth L1):

| Error $x$ | L2 loss | L1 loss | Smooth L1 loss | L2 gradient | L1 gradient | Smooth L1 gradient |
|---|---|---|---|---|---|---|
| 0.1 | 0.005 | 0.1 | 0.005 | 0.1 | 1.0 | 0.1 |
| 0.5 | 0.125 | 0.5 | 0.125 | 0.5 | 1.0 | 0.5 |
| 1.0 | 0.5 | 1.0 | 0.5 | 1.0 | 1.0 | 1.0 |
| 2.0 | 2.0 | 2.0 | 1.5 | 2.0 | 1.0 | 1.0 |
| 5.0 | 12.5 | 5.0 | 4.5 | 5.0 | 1.0 | 1.0 |

- **Error $x$:** how far the prediction is from the target.
- **Loss columns:** the penalty. **Gradient columns:** how hard this error pulls on the weights (the slope of the loss).
- **L2** here means half the squared error, as it is usually used for training.

**Summary: what the table shows.** Below 1, smooth L1 is identical to L2 (same loss, same gradient). Above 1, its gradient stops at 1 like L1's, while L2's keeps growing (5.0 at an error of 5). Caution: this is a hand-picked list of errors, not a training run.

**One bad label, three small errors (0.1, 0.2, 0.1) and one error of 5:**

| Loss | Total loss | Pull of the outlier (gradient) | Share of the total pull |
|---|---|---|---|
| L2 | 12.530 | 5.0 of 5.4 | 92.6% |
| L1 | 5.400 | 1.0 of 4.0 | 25.0% |
| Smooth L1 | 4.530 | 1.0 of 1.4 | 71.4% |

- **Pull of the outlier:** the outlier's gradient out of the sum of all four gradients, which is how much of the step the outlier decides.

**Summary: what the table shows.** With L2, the single outlier decides 92.6% of the step. With smooth L1 it decides 71.4%, still most of it (the three small errors have small gradients here) but its pull is capped at 1.0 instead of 5.0. L1 treats all errors equally, so the outlier's share falls to 25%, but L1's gradient does not shrink near zero, which makes its steps jumpy at the end. Caution: three numbers and one outlier; this shows the shape of the idea, not how much it helps a real detector.

## 🔴 4. Deep / formal
**Smooth L1 loss** of one error $x$ (the Fast R-CNN paper's equations 2 and 3, section 2.3, which I read in [[Fast R-CNN 2015]]; [[Faster R-CNN 2015]] only says its box loss uses the "smooth L1" loss "defined in" Fast R-CNN):

$$
\mathrm{smooth}_{L_1}(x)=\begin{cases}0.5\,x^{2} & \text{if } |x|<1\\ |x|-0.5 & \text{otherwise}\end{cases}
$$

**Where:**
- $x$ is one error: a predicted offset minus its target (for a box, one of four). **Matters:** the loss is applied to each of the four offsets separately and added.
- The first line, $0.5x^2$, applies to small errors. **Matters:** the loss is flat and smooth near zero, so the gradient shrinks as the prediction approaches the target.
- The second line, $|x|-0.5$, applies to large errors. **Matters:** the slope is capped at 1, so an outlier cannot create a huge gradient. The $-0.5$ makes the two lines meet at $|x|=1$ at the value 0.5.
- The switch point 1 is a choice (called $\beta$ in some libraries; the library call above uses 1.0). **Matters:** it decides what counts as "small".

**Example (tiny numbers):** errors 0.3 and 3.
- 0.3 is below 1: loss $=0.5\times0.3^2=0.045$.
- 3 is at least 1: loss $=3-0.5=2.5$.
- Check at the switch: for $|x|=1$ both lines give 0.5, so there is no jump.
- The answer means: the big error costs 56 times the small one (2.5 divided by 0.045, my arithmetic), against 100 times with the squared error (4.5 against 0.045), so the large error is punished less sharply than under L2.

**Why this shape.** L2 gives a smooth, easy-to-optimise loss but lets outliers dominate. L1 resists outliers but has a constant gradient even near zero, so the weights keep jumping around the optimum. Smooth L1 takes the smooth part of L2 near zero and the bounded slope of L1 far away. The Fast R-CNN paper's reason (section 2.3): it is "a robust L1 loss that is less sensitive to outliers than the L2 loss used in R-CNN and SPPnet", and "when the regression targets are unbounded, training with L2 loss can require careful tuning of learning rates in order to prevent exploding gradients. Eq. 3 eliminates this sensitivity." The paper gives no ablation of the loss itself.

**Relation to other losses:** with a different scale the same shape is the Huber loss (general knowledge; I did not check the exact relation of the constants).

**Limits and trade-offs:**
- Each of the four offsets is scored separately, so the loss does not directly reflect how well the boxes overlap. Later work uses an overlap-based loss ([[Intersection over Union]] losses); I did not read it and mention it from general knowledge.
- The switch point is a hand-set constant.
- Errors are in offset units (shifts as fractions of the anchor size and log size ratios), so the same 1.0 means different pixel amounts for small and large anchors.

**What came before, and how this improved on it:**
- **Before:** a squared-error (L2) loss on box offsets, "the L2 loss used in R-CNN and SPPnet" (Fast R-CNN paper, section 2.3).
- **Its limit:** one badly wrong box produces a gradient that grows with its error, which can dominate or destabilise training. In my demo an error of 5 pulled 5.0 against 1.0 once capped.
- **What changed:** replace the loss with smooth L1, which is the same for small errors and linear for large ones.
- **How much better:** in my demo the outlier's gradient fell from 5.0 to 1.0 (5 times) and the total loss from 12.53 to 4.53. This is the effect on the loss, not an accuracy gain; I did not train a detector, and the Fast R-CNN paper reports no ablation of this loss against L2 (I read its body; it only states the reason above).

## Related
[[Fast R-CNN 2015]] · [[Faster R-CNN 2015]] · [[Anchor Box]] · [[Object Detection]] · [[Loss Function]] · [[Gradient Descent]] · [[Intersection over Union]] · [[Cross-Entropy Loss]]

## Sources
- My own run: `smoothl1_demo.py` (Python 3.9.6, PyTorch 2.8.0), checked against `torch.nn.functional.smooth_l1_loss`.
- [[Fast R-CNN 2015]] (paper body read, section 2.3, equations 2 and 3: the formula, the "robust L1 loss" wording and the exploding-gradient reason) and [[Faster R-CNN 2015]] (it uses smooth L1 for its box loss and defers the definition to Fast R-CNN). The Huber relation is general knowledge, not checked. The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision Smooth L1 loss:::An error penalty that is half the squared error for small errors and the absolute error minus one half for large ones, so the gradient is capped at 1.

#flashcards/ml/concept #flashcards/topic/vision Why do detectors use smooth L1 loss for box offsets?::It behaves like squared error for small errors (smooth near zero, gradient shrinks) and like absolute error for large ones (gradient capped at 1), so a few badly wrong boxes cannot dominate training (Fast R-CNN paper, 2.3: less sensitive to outliers than L2 and avoids exploding gradients with unbounded targets).

#flashcards/ml/formula #flashcards/topic/vision Write the smooth L1 loss and explain each part.
?
$$\mathrm{smooth}_{L_1}(x)=\begin{cases}0.5\,x^{2} & |x|<1\\ |x|-0.5 & \text{otherwise}\end{cases}$$
- $x$: one error, a predicted offset minus its target; the four offsets of a box are scored separately and added.
- $0.5x^2$ for small errors: flat and smooth near zero, so the gradient shrinks as the prediction improves.
- $|x|-0.5$ for large errors: slope capped at 1, so an outlier cannot make a huge gradient; the $-0.5$ makes the two lines meet at 0.5 when $|x|=1$.
- The switch point 1: a hand-set constant.

#flashcards/ml/example #flashcards/topic/vision Errors 0.3 and 3: what are their smooth L1 losses?::0.3 is below 1: 0.5 x 0.09 = 0.045. 3 is at least 1: 3 - 0.5 = 2.5. The big error costs about 56 times the small one (my arithmetic), against 100 times for squared error.

#flashcards/ml/concept #flashcards/topic/vision At what error do the two parts of smooth L1 meet, and what is the value there?::At |x| = 1, where both give 0.5, so the curve has no jump and the slope also matches (1).

#flashcards/ml/concept #flashcards/topic/vision In my demo, how did the gradient of an error of 5 compare across L2, L1 and smooth L1?::L2: 5.0. L1: 1.0. Smooth L1: 1.0. Below an error of 1, smooth L1 had the same loss and gradient as L2.

#flashcards/ml/concept #flashcards/topic/vision In my demo with errors 0.1, 0.2, 0.1 and 5, how much of the total gradient came from the outlier?::L2: 92.6% (5.0 of 5.4). Smooth L1: 71.4% (1.0 of 1.4). L1: 25.0% (1.0 of 4.0). The total loss was 12.53, 4.53 and 5.40. Toy numbers, not a detector.

#flashcards/ml/concept #flashcards/topic/vision Why not just use L1, which also resists outliers?::Its gradient stays at 1 even near zero, so the weights keep jumping around the optimum; smooth L1 shrinks the gradient near zero like L2 does.

#flashcards/ml/concept #flashcards/topic/vision Why does smooth L1 on offsets not measure box overlap well?::Each of the four offsets is scored separately in offset units, so the loss does not reflect the overlap (IoU) of the boxes directly; overlap-based losses were proposed later (general knowledge, not checked).

#flashcards/ml/cloze #flashcards/topic/vision Smooth L1 is ==quadratic== for small errors and ==linear== for large ones, so its gradient is ==capped== at 1.
