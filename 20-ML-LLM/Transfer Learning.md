---
tags: [transfer-learning, fine-tuning, pretraining, cnn, deep-learning, computer-vision, ML]
status: growing
level: 4
tier: 3
---

# Transfer Learning

**Transfer learning** means taking a model that has already learned one task and reusing what it learned for a new task, instead of starting from zero. In a [[Convolutional Neural Network]], the early layers learn general things such as edges and textures that are useful for almost any picture task. So you keep those layers, put a new final layer on top for your own labels, and train only a little. It is the standard way to get a good picture model when you have only a few labelled pictures of your own. A close relative is [[Fine-tuning]], where you also keep training the reused layers a little.

## 🟢 1. Simple (age 12)
Imagine you already know how to play the piano. Now you want to learn the organ. You do not start from nothing: you already know how to read music, where the notes are and how fingers should move. You only learn what is new about the organ. Transfer learning is the same: a model that learned to see in one job has the "reading music" part ready, and only needs to learn the new part.

**Example:** A model has learned to name 1,000 kinds of everyday photos. You want a model that tells your two pet cats apart, but you have only 40 photos. You keep the model's seeing skills, replace its last part with a "which cat?" part, and train on your 40 photos.

## 🔵 2. How it works
1. **Start with a pretrained model.** Someone (maybe you) has already trained a network on a big set of pictures for one task. That is the *source* task (see [[Pre-training]]).
2. **Cut off the top.** Keep the "body" (the convolution layers that produce [[Feature Map]]s) and remove the last layer that picks the old labels. That last layer is called the *head*.
3. **Put a new head on.** It has one output for each label of *your* new task.
4. **Choose how much to train.** Option A: **freeze** the body (do not change it) and train only the new head. It is cheap and safe with very little data. Option B: **fine-tune** everything, with small steps so the body changes only a little.
5. **Train on your small data set**, using the usual [[Loss Function]] and [[Backpropagation]].
6. **Why it works:** the first layers have learned general features that do not depend on the labels. Only the later layers are specific to the old task.

![[transfer-learning.svg]]
*The idea in one picture. Task 1 trains a model (Model 1) with its own head (Head 1) on Data 1. For Task 2, Model 1 is kept and given a new head (Head 2), which is trained on Data 2. Image: Biggerj1, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Transfer_learning.svg), CC BY-SA 4.0.*

![[alexnet-architecture.png]]
*The same idea on a real network (AlexNet, see [[LeNet and AlexNet]]). Top row: the original network with the five convolution stages ("Frozen Layers"), the dense "Classifier" and the original "Top" of the model that gives the 1,000 old labels. Bottom row: the same frozen layers, but with the top replaced by a new, smaller one for a new task. Image: Daniel Voigt Godoy, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:AlexNet_architecture.png), CC BY 4.0.*

## 🟠 3. Technical detail
All numbers come from the code at the end of this section (Python 3.9.6, NumPy 1.24.4, scikit-learn 1.1.2, PyTorch 2.8.0 on a CPU). I used the 1,797 small 8×8 hand-written digits in scikit-learn (1,000 for training, 797 for testing), and the same small CNN as in [[Image Classification]] (two 3×3 convolutions with 8 and 16 filters, two max-pools, then one linear layer). Every accuracy is the mean of 5 seeds. Each method trained for 150 full-batch steps.

- **The three methods**, always compared on the same few labelled pictures: *from scratch* (random start, everything trains), *frozen body + new head* (the body is pretrained and unchanged, only the new linear layer learns), and *fine-tune everything* (pretrained start, everything trains, with a smaller step size of 0.003 against 0.01).
- **Experiment 1, new classes (a hard case):** the body was trained on digits 0 to 4 (492 pictures) and then reused to classify digits 5 to 9 (388 test pictures) from 10, 25 or 50 labelled pictures. The result was *negative*: from scratch scored 0.8464, 0.9113 and 0.9407, frozen body 0.8015, 0.8845 and 0.933, and fine-tuned 0.8149, 0.9026 and 0.9397. So here, transfer did not help, and from scratch was the best by about 0.1 to 4.5 points. A likely reason (my guess, not tested) is that filters trained on only five digit shapes do not describe the other five shapes well, and the digits are tiny, so a small network learns them quickly even from scratch.
- **Experiment 2, a related new question (an easier case):** the body was trained on all ten digits (1,000 pictures) and reused to answer "is this digit even or odd?" from 6, 10, 20 or 40 labelled pictures. From scratch scored 0.6464, 0.7242, 0.7854 and 0.8768. The frozen body scored 0.6841, 0.7365, 0.8218 and 0.8713. Fine-tuning scored 0.6607, 0.7169, 0.8263 and 0.8801. So with 6, 10 and 20 labelled pictures the frozen body beat training from scratch by 3.8, 1.2 and 3.6 points. Fine-tuning beat it at 6 and 20 pictures (by 1.4 and 4.1 points) but not at 10 (0.7 points worse). With 40 the three methods were within 1 point of each other. These gaps are small and averaged over 5 seeds only, so I would not treat any single one as proof.
- **How little there is to train:** the whole network has 1,573 numbers, and the new head for five labels has $(64+1)\cdot5=325$ of them, 20.7% (equation 1). Freezing the body means training only those.
- **Limits:** a toy: 8×8 digits, a tiny network, 150 steps, one split, 5 seeds, no tuning of the step size or the number of steps for any method. The real benefit of transfer learning is with big pretrained models (millions of photos) and real photographs, which I did not test, and I downloaded no pretrained model. I did not test freezing only some layers, the choice of learning rate, or other kinds of transfer (between languages, between text and images).

The code covers two experiments: (1) new digit classes from a body trained on other digits, (2) a new question (even or odd) from a body trained on all digits, and a count of the numbers that learn.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
torch.set_num_threads(1)
D = load_digits(); X = (D.data / 16.0).reshape(-1, 1, 8, 8).astype(np.float32); y = D.target.astype(np.int64)
idx = np.random.RandomState(0).permutation(len(y)); tr, te = idx[:1000], idx[1000:]
A = lambda ids: ids[y[ids] < 5]; B = lambda ids: ids[y[ids] >= 5]                 # task A: digits 0-4 (plenty of data), task B: digits 5-9 (little data)
body = lambda: nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten())
def run(m, params, ids, labels, steps, lr):
    opt = torch.optim.Adam(params, lr); xt, yt = torch.tensor(X[ids]), torch.tensor(labels)
    for _ in range(steps): opt.zero_grad(); F.cross_entropy(m(xt), yt).backward(); opt.step()
def pretrained(seed):                                                             # learn on digits 0-4, keep the body
    torch.manual_seed(seed); b = body(); m = nn.Sequential(b, nn.Linear(64, 5)); run(m, m.parameters(), A(tr), y[A(tr)], 150, 0.01); return b
def few(n_per_class, seed):                                                       # a few labelled pictures of digits 5-9
    g = np.random.RandomState(seed); return np.concatenate([g.choice(tr[y[tr] == c], n_per_class, replace=False) for c in range(5, 10)])
def evaluate(m): ids = B(te); return float((m(torch.tensor(X[ids])).argmax(1).numpy() == y[ids] - 5).mean())
print('digits 0-4 in the training split:', len(A(tr)), '| digits 5-9 in the test split:', len(B(te)))
for npc in (2, 5, 10):
    out = {k: [] for k in ('from scratch', 'frozen body + new head', 'fine-tune everything')}
    for s in range(5):
        ids = few(npc, s); lab = y[ids] - 5
        torch.manual_seed(100 + s); m = nn.Sequential(body(), nn.Linear(64, 5)); run(m, m.parameters(), ids, lab, 150, 0.01); out['from scratch'].append(evaluate(m))
        b = pretrained(s)
        for p in b.parameters(): p.requires_grad_(False)
        torch.manual_seed(200 + s); m = nn.Sequential(b, nn.Linear(64, 5)); run(m, m[1].parameters(), ids, lab, 150, 0.01); out['frozen body + new head'].append(evaluate(m))
        b = pretrained(s); torch.manual_seed(200 + s); m = nn.Sequential(b, nn.Linear(64, 5)); run(m, m.parameters(), ids, lab, 150, 0.003); out['fine-tune everything'].append(evaluate(m))
    print(npc * 5, 'labelled pictures (', npc, 'per digit ) | mean test accuracy over 5 seeds:', {k: round(float(np.mean(v)), 4) for k, v in out.items()})
tot = sum(p.numel() for p in nn.Sequential(body(), nn.Linear(64, 5)).parameters())
print('numbers that learn: whole network', tot, '| new head only', 64 * 5 + 5, '| share', round((64 * 5 + 5) / tot, 4))
# 2) a different new task on the same pictures: is the digit even or odd? (only 2 answers, and the old body already knows what digits look like)
def pre_all(seed):
    torch.manual_seed(seed); b = body(); m = nn.Sequential(b, nn.Linear(64, 10)); run(m, m.parameters(), tr, y[tr], 150, 0.01); return b
ev2 = lambda m: float((m(torch.tensor(X[te])).argmax(1).numpy() == y[te] % 2).mean())
for n in (6, 10, 20, 40):
    out = {k: [] for k in ('from scratch', 'frozen body + new head', 'fine-tune everything')}
    for s in range(5):
        ids = np.random.RandomState(s).choice(tr, n, replace=False); lab = y[ids] % 2
        torch.manual_seed(100 + s); m = nn.Sequential(body(), nn.Linear(64, 2)); run(m, m.parameters(), ids, lab, 150, 0.01); out['from scratch'].append(ev2(m))
        b = pre_all(s)
        for p in b.parameters(): p.requires_grad_(False)
        torch.manual_seed(200 + s); m = nn.Sequential(b, nn.Linear(64, 2)); run(m, m[1].parameters(), ids, lab, 150, 0.01); out['frozen body + new head'].append(ev2(m))
        b = pre_all(s); torch.manual_seed(200 + s); m = nn.Sequential(b, nn.Linear(64, 2)); run(m, m.parameters(), ids, lab, 150, 0.003); out['fine-tune everything'].append(ev2(m))
    print('even/odd,', n, 'labelled pictures | mean test accuracy over 5 seeds:', {k: round(float(np.mean(v)), 4) for k, v in out.items()})
```
Reading the output (Python 3.9.6, PyTorch 2.8.0):
- **Sizes:** 492 training pictures of digits 0 to 4, and 388 test pictures of digits 5 to 9.
- **Experiment 1:** from scratch 0.8464, 0.9113, 0.9407; frozen 0.8015, 0.8845, 0.933; fine-tuned 0.8149, 0.9026, 0.9397, for 10, 25 and 50 labelled pictures.
- **Numbers that learn:** whole network 1,573, new head only 325, a share of 0.2066.
- **Experiment 2:** from scratch 0.6464, 0.7242, 0.7854, 0.8768; frozen 0.6841, 0.7365, 0.8218, 0.8713; fine-tuned 0.6607, 0.7169, 0.8263, 0.8801, for 6, 10, 20 and 40 labelled pictures.
- **Limits:** the run takes several minutes and nothing is timed.

## 🔴 4. Deep / formal

**How much has to be learned?** When the body is frozen, only the new head learns. A head that turns $F$ features into $c$ labels (one linear layer) has:

$$P_{\text{head}}=(F+1)\,c$$

**Where:**
- $F$ is the number of features the frozen body hands over (the length of its flattened last [[Feature Map]]s). **Matters:** more features give the head more to work with, but also more numbers to learn.
- $c$ is the number of labels in the new task. **Matters:** the head needs one set of weights per label.
- The $+1$ is the bias of each label. **Matters:** it lets each label have its own base level, and adds $c$ numbers.
- $P_{\text{head}}$ is the number of learned numbers when the body is frozen. **Matters:** a small $P_{\text{head}}$ is easy to pin down from a handful of examples, which is why freezing is safest when data are scarce. The price is that the body cannot adapt to the new task.

**Example (tiny numbers):** $F=64$ features, $c=5$ labels (digits 5 to 9).
1. $(64+1)\cdot5=325$.
2. The whole network has 1,573 numbers, so the head is $325/1573=0.2066$, about a fifth.
3. The code printed 325 and 0.2066. For the even/odd task, $c=2$ gives $(64+1)\cdot2=130$ (computed by hand, not printed).
4. Meaning: with the body frozen, only a fifth of the numbers have to be learned from the few new pictures, and the other four fifths come for free (if they are useful for the new task, as experiment 2 suggests and experiment 1 does not).

**What came before, and how this improved on it:**
- **Before:** train a new model from random numbers for every task, which needs lots of labelled data and computing time for each one.
- **Its limit:** most real problems have only a few labelled examples, and training from scratch on them overfits ([[Overfitting]]).
- **What changed:** reuse the general features of a model trained on a big source task. Researchers measured this layer by layer: Yosinski, Clune, Bengio and Lipson (arXiv, 6 November 2014, I read the abstract) report that early layers of networks trained on natural images learn general features (edge-like filters and colour blobs), later layers are more specific to the task, transfer gets worse as the tasks grow less similar but still beats random features, and starting from transferred features helps even after fine-tuning.
- **How much better:** in this note's toy, it depended on the task. For a related question (even or odd) with 6 to 20 labelled pictures, a frozen body beat training from scratch by 1.2 to 3.8 points (for example 0.8218 against 0.7854 at 20 pictures), and fine-tuning did so in two of the three cases. For new classes (digits 5 to 9) it did *not* help (from scratch 0.9113 against 0.8845 frozen at 25 pictures). With 40 pictures the gap closed. I did not test real photographs, where the published benefit is much larger (I did not look up a figure).

**Other points (from memory, not checked here):**
- A rule of thumb: the smaller your data and the closer your task to the source task, the more you should freeze; with plenty of data and a different task, fine-tune more layers or train from scratch.
- Big modern models are often pretrained on huge data sets without labels ([[Self-Supervised Learning]], [[Pre-training]]) and then adapted, which is transfer learning at large scale, including for text ([[LLM]], [[Fine-tuning]]).
- Pretrained pictures-models are commonly trained on ImageNet (see [[LeNet and AlexNet]]) and shared in libraries such as torchvision. I downloaded none here.
- If the source data differ a lot from the target (for example everyday photos against medical scans), transfer can help less. I did not test it.

## Related
[[Fine-tuning]] · [[Pre-training]] · [[Self-Supervised Learning]] · [[Convolutional Neural Network]] · [[Feature Map]] · [[Image Classification]] · [[Overfitting]] · [[Data Augmentation]] · [[LeNet and AlexNet]] · [[LLM]] · [[Supervised Learning]] · [[Catastrophic Forgetting]] · [[Continual Learning]] · [[Transfer Learning vs Vanilla CNN]]
- [[ELMo 2018]] · [[Object Detection]] · [[Image Segmentation]]
- [[Tiny Object Detector]]
## Sources
- scikit-learn's `load_digits` data (loaded by the code, version 1.1.2).
- Yosinski, Clune, Bengio and Lipson, *How transferable are features in deep neural networks?* (arXiv 1411.1792, 6 November 2014), abstract page read through a web fetch. I did not read the paper's body.
- Library versions in this run: Python 3.9.6, NumPy 1.24.4, scikit-learn 1.1.2, PyTorch 2.8.0 (CPU). The code and every printed value were run for this note, and nothing was timed.
- Figures: [Transfer_learning.svg](https://commons.wikimedia.org/wiki/File:Transfer_learning.svg) (Biggerj1, CC BY-SA 4.0) and [AlexNet_architecture.png](https://commons.wikimedia.org/wiki/File:AlexNet_architecture.png) (Daniel Voigt Godoy, CC BY 4.0), on Wikimedia Commons. I viewed both before writing the captions.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/ml-foundations Transfer learning:::Reuse a model trained on one task as the starting point for a new task. The body that learned to see stays (frozen or fine-tuned), and only a new head is learned.

#flashcards/ml/formula #flashcards/topic/ml-foundations Write the number of learned numbers in a new head and explain each symbol.
?
$$P_{\text{head}}=(F+1)\,c$$
- $F$: the number of features the frozen body hands over (the length of its flattened last feature maps). More features give the head more to work with and more numbers to learn.
- $c$: the number of labels in the new task. The head needs one set of weights per label.
- The $+1$: the bias of each label, which gives each label its own base level and adds $c$ numbers.
- $P_{\text{head}}$: the learned numbers when the body is frozen. A small head is easy to pin down from a handful of examples, so freezing is safest when data are scarce.

#flashcards/ml/example #flashcards/topic/ml-foundations $F=64$ features, $c=5$ labels. How many head numbers, and what share of a 1,573-number network?::$(64+1)\cdot5=325$, and $325/1573=0.2066$, about a fifth. For the even/odd task $c=2$ gives $(64+1)\cdot2=130$ (computed by hand, not printed).

#flashcards/ml/image #flashcards/topic/ml-foundations ![[transfer-learning.svg]]
What does this picture show?
?
Task 1 trains a model (Model 1) with its own head (Head 1) on Data 1. For Task 2, Model 1 is kept and given a new head (Head 2), which is trained on Data 2. (From the note's caption: Biggerj1, Wikimedia Commons.)

#flashcards/ml/image #flashcards/topic/ml-foundations ![[alexnet-architecture.png]]
What does this picture show?
?
The same idea on a real network (AlexNet). Top row: the original network with its five convolution stages ("Frozen Layers"), the dense "Classifier" and the original "Top" that gives the 1,000 old labels. Bottom row: the same frozen layers, but with the top replaced by a new, smaller one for a new task. (From the note's caption, Wikimedia Commons.)

#flashcards/ml/cloze #flashcards/topic/ml-foundations In transfer learning the ==body== is kept (frozen) and a new ==head== is trained on the new data.

#flashcards/ml/concept #flashcards/topic/ml-foundations When is freezing the body safest?::When data are scarce, because the small head is easy to pin down from few examples.
