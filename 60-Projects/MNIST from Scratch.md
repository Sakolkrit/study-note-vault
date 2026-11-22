---
tags: [project, python, numpy, pytorch, neural-network, backpropagation, mnist, ablation]
status: done
stack: Python 3.9, NumPy, PyTorch (CPU), torchvision, scikit-learn
goal: write a small neural network and its backpropagation by hand, prove the gradients right, then measure how the starting weights, BatchNorm and learning rate change the result on two datasets, MNIST and the small sklearn digits
---

# MNIST from Scratch

> **In one line:** build a small [[Neural Network]] that reads handwritten digits, write [[Backpropagation]] by hand in [[NumPy]], prove it right against two independent checks, then run the same experiments on the real MNIST images and on the tiny sklearn digits to see which findings carry over.

## Goal
Done means five checks pass: (1) both datasets load and are inspected, (2) my hand-written gradients agree with finite differences and with [[PyTorch]]'s automatic gradients, (3) a PyTorch run and the NumPy run give the same weights when started from the same weights and batches, (4) six settings are compared on both datasets with 3 seeds each, and (5) the differences between the two datasets are written down, with what I can and cannot conclude.

## Result in short
- **Data:** MNIST is 60,000 training and 10,000 test images of 28×28 pixels. The sklearn digits are 1,797 images of 8×8 pixels, and I split them 80/20 myself: 1,437 train and 360 test. A model that always guesses the most common class scores 11.4% on the MNIST test set and 12.2% on the digits test set.
- **Gradients are right:** worst gap against finite differences 2.3e-10, against PyTorch autograd 3.5e-17, and after 200 steps from the same weights and batches the two networks differ by at most 6.7e-16.
- **Baseline** (3 hidden layers of 128, ReLU, He start, learning rate 0.1, 2,000 steps of batch 64, mean of 3 seeds): **98.15%** test accuracy on digits and **96.30%** on MNIST. The hand-written NumPy network gave 98.52% and 96.60%. These two differ from PyTorch only because they draw different random numbers (the check in step 4 shows they are identical on the same inputs).
- **What mattered, on both:** a bad weight start broke training. A start std of 0.5 gave 8.43% on digits and 9.80% on MNIST (all NaN on MNIST), and a learning rate of 1.0 gave 15.93% and 50.30% with huge seed-to-seed swings.
- **Where the datasets disagreed:** a tiny start (std 0.01) stayed dead on digits (8.24%) but slowly learned on MNIST (54.78%, std 9.44 across seeds). A learning rate of 0.01 cost only 0.37 points on digits but 3.74 points on MNIST.
- **BatchNorm:** +0.65 points on digits (98.80% against 98.15%, inside the noise) and +0.80 points on MNIST (97.10% against 96.30%, every BatchNorm seed above every baseline seed). Its run-time cost was noisy: about +50% and +36% on digits (two runs) and +28% and +6% on MNIST.
- **Not done:** no [[Convolutional Neural Network]], no GPU, no validation set or tuning, only 3 seeds, and training stopped after 2,000 steps, which is far from converged on MNIST.

## What you need
Versions I used. This was **my existing Python environment, not a fresh virtual environment** (installing PyTorch into a new one means a large download that I did not ask for), on Windows, CPU only, one thread:

| Tool | Version |
|---|---|
| Python | 3.9 |
| numpy | 1.24.4 |
| torch | 2.8.0+cpu |
| torchvision | 0.23.0+cpu |
| scikit-learn | 1.1.2 |

The data: the four MNIST files (about 11.6 MB compressed, 64 MB on disk with the unpacked copies). I downloaded them with `torchvision.datasets.MNIST`, after you said yes, into a scratch folder and not into the vault. The mirrors it lists are `ossci-datasets.s3.amazonaws.com/mnist/` and `yann.lecun.com/exdb/mnist/`. I did not check MNIST's licence. The sklearn digits ship inside the library.

## Steps
All files go in one folder, which I will call the project folder.

1. **Download MNIST** (needs the internet, about 11.6 MB), into a `data` folder inside the project folder.

   **`download.py`**
   ```python
   import sys
   from torchvision.datasets import MNIST
   root = sys.argv[1]
   tr = MNIST(root, train=True, download=True); te = MNIST(root, train=False, download=True)
   print(len(tr), len(te), tr.data.shape, tr.data.dtype)
   ```
   Run `python download.py data`. I saw `60000 10000 torch.Size([60000, 28, 28]) torch.uint8`, and the folder `data/MNIST/raw` held the four files.

2. **Load and look at both datasets.**

   **`lab.py`**
   ```python
   import numpy as np
   MNIST_ROOT = "data"   # folder holding MNIST/raw (see step 1)

   def load(name):
       """returns Xtr, ytr, Xte, yte with pixels scaled to 0..1"""
       if name == "digits":                       # sklearn's 1,797 images of 8x8, my own 80/20 split
           from sklearn.datasets import load_digits
           D = load_digits(); X, y = D.data / 16.0, D.target
           idx = np.random.RandomState(0).permutation(len(y)); k = int(0.8 * len(y))
           y = y.astype(np.int64)
           return X[idx[:k]], y[idx[:k]], X[idx[k:]], y[idx[k:]]
       from torchvision.datasets import MNIST     # the standard 60,000 / 10,000 split, 28x28
       tr, te = MNIST(MNIST_ROOT, train=True, download=False), MNIST(MNIST_ROOT, train=False, download=False)
       f = lambda d: d.data.numpy().reshape(len(d), -1) / 255.0
       return f(tr), tr.targets.numpy().astype(np.int64), f(te), te.targets.numpy().astype(np.int64)

   if __name__ == "__main__":
       for n in ("digits", "mnist"):
           Xtr, ytr, Xte, yte = load(n)
           print(f"{n:6s}: train {Xtr.shape} test {Xte.shape} | pixel range {Xtr.min():.1f}-{Xtr.max():.1f} | mean pixel {Xtr.mean():.3f}")
           print("   train class counts:", np.bincount(ytr).tolist(), "| test class counts:", np.bincount(yte).tolist())
           print(f"   share of the most common class in the test set: {np.bincount(yte).max() / len(yte):.3f}")
   ```
   Run `python lab.py`. I saw:
   ```
   digits: train (1437, 64) test (360, 64) | pixel range 0.0-1.0 | mean pixel 0.305
      train class counts: [147, 147, 138, 150, 137, 153, 141, 139, 146, 139] | test class counts: [31, 35, 39, 33, 44, 29, 40, 40, 28, 41]
      share of the most common class in the test set: 0.122
   mnist : train (60000, 784) test (10000, 784) | pixel range 0.0-1.0 | mean pixel 0.131
      train class counts: [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949] | test class counts: [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]
      share of the most common class in the test set: 0.114
   ```
   Things to notice: MNIST pixels are on average darker (mean 0.131) than digits (0.305) because a 28×28 picture has a lot of empty border. The classes are roughly balanced in both, and the most common class makes up 11.4% (MNIST) and 12.2% (digits) of the test set, which is the bar for "better than guessing".

3. **Write the network and its backpropagation by hand.** Layers of weights and biases, ReLU in the hidden layers, [[Softmax]] with [[Cross-Entropy Loss]] at the end, and plain [[Stochastic Gradient Descent]]. The weights start with the He rule from [[Weight Initialization]] unless told otherwise.

   **`np_mlp.py`**
   ```python
   import numpy as np

   def init(sizes, rng, std=None):
       """list of (W, b); He start sqrt(2/fan_in) unless std is given, biases 0"""
       return [(rng.randn(a, b) * (np.sqrt(2 / a) if std is None else std), np.zeros(b)) for a, b in zip(sizes[:-1], sizes[1:])]

   def forward(P, X):
       hs = [X]
       for i, (W, b) in enumerate(P):
           z = hs[-1] @ W + b
           hs.append(np.maximum(z, 0) if i < len(P) - 1 else z)   # ReLU on hidden layers, raw scores at the end
       return hs

   def loss_grads(P, X, y):
       hs = forward(P, X); z = hs[-1]
       z = z - z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)   # softmax
       n = len(y); loss = -np.log(p[np.arange(n), y]).mean()
       d = p; d[np.arange(n), y] -= 1; d /= n                      # gradient of the loss w.r.t. the scores
       grads = []
       for i in range(len(P) - 1, -1, -1):                         # backpropagation, last layer first
           W, b = P[i]; grads.append((hs[i].T @ d, d.sum(0)))
           if i > 0: d = (d @ W.T) * (hs[i] > 0)
       return loss, grads[::-1]

   def train(P, X, y, steps, lr, bs, rng):
       for _ in range(steps):
           idx = rng.randint(0, len(y), bs)
           _, g = loss_grads(P, X[idx], y[idx])
           P = [(W - lr * gW, b - lr * gb) for (W, b), (gW, gb) in zip(P, g)]
       return P

   def accuracy(P, X, y): return float((forward(P, X)[-1].argmax(1) == y).mean())
   ```

4. **Prove it right with three checks.**

   **`check_np.py`**
   ```python
   import numpy as np, torch, torch.nn as nn
   from lab import load
   from np_mlp import init, loss_grads, train, accuracy
   Xtr, ytr, Xte, yte = load("digits")
   sizes = [64, 16, 16, 10]; rng = np.random.RandomState(0); P = init(sizes, rng)
   X, y = Xtr[:32], ytr[:32]

   # 1) hand backprop against finite differences on 6 random weights
   loss, g = loss_grads(P, X, y); worst = 0
   for _ in range(6):
       l = rng.randint(3); i, j = rng.randint(P[l][0].shape[0]), rng.randint(P[l][0].shape[1]); h = 1e-6
       Q = [(W.copy(), b.copy()) for W, b in P]; Q[l][0][i, j] += h; up = loss_grads(Q, X, y)[0]
       Q[l][0][i, j] -= 2 * h; dn = loss_grads(Q, X, y)[0]
       worst = max(worst, abs((up - dn) / (2 * h) - g[l][0][i, j]))
   print(f"1) finite differences: worst gap over 6 weights {worst:.1e}")

   # 2) hand backprop against PyTorch autograd (same weights, float64)
   def torch_net(P):
       layers = []
       for k, (W, b) in enumerate(P):
           lin = nn.Linear(*W.shape).double(); lin.weight.data = torch.tensor(W.T.copy()); lin.bias.data = torch.tensor(b.copy())
           layers.append(lin); layers += [nn.ReLU()] * (k < len(P) - 1)
       return nn.Sequential(*layers)
   net = torch_net(P); nn.functional.cross_entropy(net(torch.tensor(X)), torch.tensor(y)).backward()
   gap = max(np.abs(net[2 * k].weight.grad.numpy().T - g[k][0]).max() for k in range(3))
   print(f"2) PyTorch autograd: largest gap in any weight gradient {gap:.1e}")

   # 3) 200 SGD steps with the same weights and the same batches ends at the same weights
   r1, r2 = np.random.RandomState(5), np.random.RandomState(5)
   P2 = train([(W.copy(), b.copy()) for W, b in P], Xtr, ytr, 200, 0.1, 64, r1)
   Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
   net = torch_net(P)
   opt = torch.optim.SGD(net.parameters(), lr=0.1)
   for _ in range(200):
       idx = r2.randint(0, len(ytr), 64); opt.zero_grad(); nn.functional.cross_entropy(net(Xt[idx]), yt[idx]).backward(); opt.step()
   gap = max(np.abs(net[2 * k].weight.detach().numpy().T - P2[k][0]).max() for k in range(3))
   print(f"3) after 200 steps: largest weight gap {gap:.1e} | NumPy test accuracy {accuracy(P2, Xte, yte):.3f}")
   ```
   Run `python check_np.py`. I saw:
   ```
   1) finite differences: worst gap over 6 weights 2.3e-10
   2) PyTorch autograd: largest gap in any weight gradient 3.5e-17
   3) after 200 steps: largest weight gap 6.7e-16 | NumPy test accuracy 0.858
   ```
   What each one proves: (1) a nudge of one weight by 1e-6 up and down changes the loss by what my gradient says, so the formulas are right. (2) PyTorch's automatic gradients agree, to about 1e-17 (rounding). (3) The same 200 steps from the same weights on the same batches end at the same weights to about 1e-15, so the NumPy and PyTorch training loops are doing the same thing. After that check the NumPy network reached 0.858 test accuracy on digits, with a small network of 16 hidden units and only 200 steps.

5. **Run the comparison.** Six settings, 3 seeds each (0, 1, 2), on each dataset. Every run is 2,000 steps of batch 64 with plain SGD and no momentum, three hidden layers of 128, ReLU. PyTorch runs draw the batches with the same sampler as the NumPy network. The seed changes both the starting weights and the batches.

   **`ablate.py`**
   ```python
   import json, time
   import numpy as np, torch, torch.nn as nn
   from lab import load
   import np_mlp
   torch.set_num_threads(1)
   STEPS, BS, SEEDS = 2000, 64, (0, 1, 2)
   CONFIGS = {                       # name: (weight start, batchnorm, learning rate)
       "He start, lr 0.1 (baseline)": ("he", False, 0.1),
       "start std 0.01":              ("small", False, 0.1),
       "start std 0.5":               ("big", False, 0.1),
       "He start + BatchNorm":        ("he", True, 0.1),
       "lr 0.01":                     ("he", False, 0.01),
       "lr 1.0":                      ("he", False, 1.0),
   }

   def run(data, init, bn, lr, seed):
       Xtr, ytr, Xte, yte = data; sizes = [Xtr.shape[1], 128, 128, 128, 10]
       torch.manual_seed(seed); rng = np.random.RandomState(seed); layers = []
       for i in range(4):
           lin = nn.Linear(sizes[i], sizes[i + 1]); nn.init.zeros_(lin.bias)
           nn.init.normal_(lin.weight, 0, {"small": 0.01, "he": (2 / sizes[i]) ** 0.5, "big": 0.5}[init])
           layers.append(lin)
           if i < 3: layers += ([nn.BatchNorm1d(128)] if bn else []) + [nn.ReLU()]
       net = nn.Sequential(*layers); opt = torch.optim.SGD(net.parameters(), lr=lr)
       X, y = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr)
       t = time.time()
       for _ in range(STEPS):
           idx = torch.tensor(rng.randint(0, len(y), BS)); opt.zero_grad(); nn.functional.cross_entropy(net(X[idx]), y[idx]).backward(); opt.step()
       sec = time.time() - t; net.eval()
       with torch.no_grad(): out = net(torch.tensor(Xte, dtype=torch.float32))
       return float((out.argmax(1).numpy() == yte).mean()), bool(torch.isnan(out).any()), sec

   res = {}
   for ds in ("digits", "mnist"):
       data = load(ds); res[ds] = {}
       for name, (init, bn, lr) in CONFIGS.items():
           r = [run(data, init, bn, lr, s) for s in SEEDS]
           res[ds][name] = dict(acc=[a for a, _, _ in r], nan=sum(n for _, n, _ in r), sec=float(np.mean([s for _, _, s in r])))
       # the hand-written NumPy network: He start, lr 0.1, same batches and steps
       r = []
       for s in SEEDS:
           t = time.time(); rng = np.random.RandomState(s)
           P = np_mlp.train(np_mlp.init([data[0].shape[1], 128, 128, 128, 10], rng), data[0], data[1], STEPS, 0.1, BS, rng)
           r.append((np_mlp.accuracy(P, data[2], data[3]), time.time() - t))
       res[ds]["NumPy by hand (He, lr 0.1)"] = dict(acc=[a for a, _ in r], nan=0, sec=float(np.mean([s for _, s in r])))
       print(f"--- {ds}: test accuracy after {STEPS} steps of batch {BS}, 3 seeds ---")
       for name, v in res[ds].items():
           a = np.array(v["acc"]) * 100
           print(f"{name:30s} mean {a.mean():6.2f}% std {a.std():5.2f} | seeds {np.round(a, 2).tolist()} | NaN runs {v['nan']} | {v['sec']:.1f}s per run")
   json.dump(res, open("ablate_out.json", "w"))
   ```
   Run `python ablate.py`. It takes about 3 minutes on my machine and prints the table below. I saw:
   ```
   --- digits: test accuracy after 2000 steps of batch 64, 3 seeds ---
   He start, lr 0.1 (baseline)    mean  98.15% std  0.47 | seeds [98.33, 98.61, 97.5] | NaN runs 0 | 2.8s per run
   start std 0.01                 mean   8.24% std  0.26 | seeds [8.06, 8.06, 8.61] | NaN runs 0 | 2.9s per run
   start std 0.5                  mean   8.43% std  0.26 | seeds [8.06, 8.61, 8.61] | NaN runs 1 | 2.9s per run
   He start + BatchNorm           mean  98.80% std  0.47 | seeds [98.61, 99.44, 98.33] | NaN runs 0 | 4.2s per run
   lr 0.01                        mean  97.78% std  0.45 | seeds [98.33, 97.78, 97.22] | NaN runs 0 | 3.0s per run
   lr 1.0                         mean  15.93% std  9.57 | seeds [9.72, 29.44, 8.61] | NaN runs 0 | 3.0s per run
   NumPy by hand (He, lr 0.1)     mean  98.52% std  0.13 | seeds [98.61, 98.61, 98.33] | NaN runs 0 | 3.6s per run
   --- mnist: test accuracy after 2000 steps of batch 64, 3 seeds ---
   He start, lr 0.1 (baseline)    mean  96.30% std  0.52 | seeds [96.72, 96.61, 95.56] | NaN runs 0 | 4.7s per run
   start std 0.01                 mean  54.78% std  9.44 | seeds [42.94, 66.04, 55.36] | NaN runs 0 | 4.7s per run
   start std 0.5                  mean   9.80% std  0.00 | seeds [9.8, 9.8, 9.8] | NaN runs 3 | 4.6s per run
   He start + BatchNorm           mean  97.10% std  0.12 | seeds [97.19, 97.17, 96.93] | NaN runs 0 | 6.0s per run
   lr 0.01                        mean  92.56% std  0.38 | seeds [92.56, 93.03, 92.1] | NaN runs 0 | 4.5s per run
   lr 1.0                         mean  50.30% std 33.72 | seeds [48.17, 10.1, 92.62] | NaN runs 0 | 4.7s per run
   NumPy by hand (He, lr 0.1)     mean  96.60% std  0.07 | seeds [96.51, 96.69, 96.61] | NaN runs 0 | 6.0s per run
   ```

## The comparison, in one table
Mean test accuracy over 3 seeds, in per cent (the seed-to-seed standard deviation is in the run output above and is population standard deviation over 3 seeds, so treat it as rough).

| Setting | digits (360 test images) | MNIST (10,000 test images) |
|---|---|---|
| He start, lr 0.1 (baseline) | 98.15 | 96.30 |
| NumPy by hand, same settings | 98.52 | 96.60 |
| He start + BatchNorm | 98.80 | 97.10 |
| lr 0.01 | 97.78 | 92.56 |
| start std 0.01 | 8.24 | 54.78 |
| lr 1.0 | 15.93 | 50.30 |
| start std 0.5 | 8.43 | 9.80 |

**What the two datasets agree on**
- The healthy settings come out in the same order on both: BatchNorm, then baseline, then lr 0.01. Wrong-sized starts and a learning rate of 1.0 break training on both.
- A start of std 0.5 is the worst on both and produced NaN outputs in 1 of 3 digits runs and 3 of 3 MNIST runs. This matches the picture in [[Weight Initialization]] (too-big starts overflow).
- BatchNorm helps a little on both, and costs time on both.

**Where they disagree, and what I can say about it**
- **A tiny start (0.01):** on digits the network never left the starting point (8.24%, close to a constant guess), but on MNIST it slowly learned to 54.78% (seeds 42.94, 66.04, 55.36). The two datasets have different input sizes (64 against 784 numbers) and different brightness, so the first layer adds up different amounts of signal. I think this is why, but I did not test it.
- **The slow learning rate (0.01):** a drop of 0.37 points on digits (inside the noise) but 3.74 points on MNIST. **This is partly my design:** I gave both datasets the same number of steps (2,000 steps of 64 images = 128,000 images shown). That is about 89 passes over the 1,437 digits but only about 2.1 passes over the 60,000 MNIST images. A small learning rate has had plenty of time on digits and almost none on MNIST, so the comparison is unfair to MNIST here. A fixed number of passes would give a different answer.
- **BatchNorm's gain:** +0.65 points on digits is 2.3 test images out of 360 and the seed spread is 0.47, so I cannot tell it from noise. On MNIST the gain is +0.80 points with a much larger test set, and all three BatchNorm seeds (97.19, 97.17, 96.93) beat all three baseline seeds (96.72, 96.61, 95.56). So only the MNIST result is clear.
- **Test-set size:** one digits test image is 0.28 points of accuracy, and a rough binomial standard error at 98% on 360 images is about 0.74 points (my own arithmetic, not run). On MNIST it is about 0.19 points at 96.3% on 10,000 images. So differences under a point on digits are not trustworthy on their own, while the same size of difference on MNIST can be.
- **Broken runs:** the settings that failed have wide seed-to-seed swings (lr 1.0 on MNIST ranged from 10.10% to 92.62%), so their means are not a meaningful ranking. The digits order of the failures (lr 1.0, then 0.5, then 0.01) and the MNIST order (0.01, then lr 1.0, then 0.5) should not be read as real differences.
- **Run time:** BatchNorm took 4.2 s against 2.8 s on digits in the first run (+50%) and 3.8 s against 2.8 s in the rerun (+36%). On MNIST it was 6.0 s against 4.7 s (+28%) and then 5.2 s against 4.9 s (+6%). So BatchNorm costs something, but I cannot say how much from two runs on one machine. The hand-written NumPy network took 3.6 s and 6.0 s per run in the first run (3.6 s and 5.5 s in the rerun). These are single-thread times, not a careful benchmark.

**What this shows:** a small network can be written, with its backpropagation, in about 30 lines of NumPy and it behaves exactly like the library version. The big lessons about starting weights and learning rate survive from the tiny dataset to the real one. **What it does not show:** that the digits dataset is a good stand-in for MNIST. It was a cheap guide to the large effects and a poor one for small effects and for how learning speed depends on data size.

## What went wrong, and the fix
- **`RuntimeError: expected scalar type Long but found Int`** from `cross_entropy`: the labels came out as 32-bit integers on this Windows setup. Fix: `lab.py` casts them to `int64`.
- **An accuracy of 8.43% with only 1 NaN run (of 3), for the 0.5 start on digits.** I first thought the NaN count and the accuracy did not agree. They do: the other runs have no NaN but produce a network that puts every test image in one class, and 8.06% and 8.61% equal the shares of two classes in the 360-image test set (29/360 and 31/360). I did not print the predictions to confirm that directly. An all-NaN output also lands on class 0, which is 9.80% of the MNIST test set.
- **A copy of the forward pass in my first `np_mlp.py`** (I computed it twice per step). Removed, since the answer is the same and it only wasted time.

## What I did not run
- **A fresh virtual environment.** I used my existing environment (versions above). The rerun described below used a fresh folder, not a fresh Python install.
- **A rerun check beyond one extra pass.** I extracted the five code blocks from this note into a new folder, ran `lab.py`, `check_np.py` and `ablate.py` there, and compared: the printed output of the first two was identical, and every accuracy and NaN count from `ablate.py` was identical to the first run (the runs are seeded, single-thread and on CPU). Only the timings differed. I did one rerun, not many.
- **The download in the rerun.** I downloaded MNIST once (step 1 as written). For the rerun from the note, I copied that `data` folder instead of downloading again.
- **A GPU, a [[Convolutional Neural Network]], momentum or [[Adam Optimizer]]**, other layer widths, other numbers of steps, or more than 3 seeds.
- **A validation set.** Every setting is scored on the test images, and I did not use those scores to pick or tune anything. The final table is a comparison, not a model selection.
- **Training to convergence.** 2,000 steps is about 2 passes over MNIST, so all MNIST numbers are for an early stage of training.
- **MNIST's licence,** and any use of the data other than loading and training.
- **The note in Obsidian.** I did not open it there.

## Concepts used
- [[Neural Network]] · [[Backpropagation]] · [[Gradient Descent]] · [[Stochastic Gradient Descent]] · [[Weight Initialization]] · [[Batch Normalization]] · [[ReLU]] · [[Activation Function]] · [[Softmax]] · [[Cross-Entropy Loss]] · [[Loss Function]] · [[Learning Rate]] · [[Accuracy]] · [[Vanishing and Exploding Gradients]] · [[Law of Large Numbers]] · [[Variance]] · [[PyTorch]] · [[NumPy]]

## Next ideas
- Fix the number of passes over the data (epochs) instead of the number of steps, and see how the learning-rate result changes on MNIST.
- Add a validation split, then tune the learning rate on it and score the test set once, as in [[Heart Disease Classifier]].
- Swap the first layers for a [[Convolutional Neural Network]] and compare.
- Try momentum and [[Adam Optimizer]] against plain SGD.
- Use 10 seeds, so the BatchNorm and learning-rate differences get proper error bars.

## Flashcards
#flashcards/projects

#flashcards/projects/concept #flashcards/topic/project-deep-learning What were the five "done" checks of MNIST from Scratch?::(1) both datasets load and are inspected, (2) hand-written gradients agree with finite differences and with PyTorch's automatic gradients, (3) a PyTorch run and the NumPy run give the same weights from the same weights and batches, (4) six settings compared on both datasets with 3 seeds each, (5) the differences between the datasets written down with what can and cannot be concluded.

#flashcards/projects/concept #flashcards/topic/project-deep-learning What are the two datasets?::MNIST: 60,000 training and 10,000 test images of 28×28 pixels. sklearn digits: 1,797 images of 8×8 pixels, split 80/20 by the note's author into 1,437 train and 360 test. Always guessing the most common class scores 11.4% on MNIST and 12.2% on digits.

#flashcards/projects/concept #flashcards/topic/project-deep-learning How close were the hand-written gradients?::Worst gap 2.3e-10 against finite differences and 3.5e-17 against PyTorch autograd. After 200 steps from the same weights and batches the two networks differ by at most 6.7e-16.

#flashcards/projects/concept #flashcards/topic/project-deep-learning What was the baseline network and score?::3 hidden layers of 128, ReLU, He start, learning rate 0.1, 2,000 steps of batch 64, mean of 3 seeds: 98.15% test accuracy on digits and 96.30% on MNIST. The hand-written NumPy network gave 98.52% and 96.60%, differing only because it draws different random numbers.

#flashcards/projects/concept #flashcards/topic/project-deep-learning What happened with a bad weight start or a big learning rate?::A start std of 0.5 gave 8.43% on digits and 9.80% on MNIST (all NaN on MNIST). A learning rate of 1.0 gave 15.93% and 50.30% with huge seed-to-seed swings.

#flashcards/projects/concept #flashcards/topic/project-deep-learning Where did the two datasets disagree?::A tiny start (std 0.01) stayed dead on digits (8.24%) but slowly learned on MNIST (54.78%). A learning rate of 0.01 cost 0.37 points on digits but 3.74 on MNIST. The first is probably because the input sizes and brightness differ (64 against 784 numbers), but that was not tested.

#flashcards/projects/example #flashcards/topic/project-deep-learning Why is the learning-rate-0.01 comparison unfair to MNIST?::Both got 2,000 steps of 64 images, which is about 89 passes over the 1,437 digits but only about 2.1 passes over the 60,000 MNIST images. A small learning rate had plenty of time on digits and almost none on MNIST. A fixed number of passes would give a different answer.

#flashcards/projects/concept #flashcards/topic/project-deep-learning What did BatchNorm do?::Digits: +0.65 points (98.80% against 98.15%), inside the noise. MNIST: +0.80 points (97.10% against 96.30%), and every BatchNorm seed (97.19, 97.17, 96.93) beat every baseline seed (96.72, 96.61, 95.56). Only the MNIST result is clear. Its run-time cost was noisy: about +50% and +36% on digits and +28% and +6% on MNIST.

#flashcards/projects/concept #flashcards/topic/project-deep-learning How big is the noise from the test-set size?::One digits test image is 0.28 points. A rough binomial standard error at 98% on 360 images is about 0.74 points (the note's own arithmetic), against about 0.19 points on MNIST's 10,000 images. Under a point on digits is not trustworthy on its own.

#flashcards/projects/example #flashcards/topic/project-deep-learning Why did the 0.5-start digits run show 8.43% with only one NaN run?::The other runs had no NaN but put every test image in one class, and 8.06% and 8.61% equal the shares of two classes in the 360-image test set (29/360 and 31/360). The note did not print the predictions to confirm this directly.

#flashcards/projects/concept #flashcards/topic/project-deep-learning What error did `cross_entropy` give and what was the fix?::`RuntimeError: expected scalar type Long but found Int`, because the labels came out as 32-bit integers on this Windows setup. The fix: `lab.py` casts them to `int64`.

#flashcards/projects/concept #flashcards/topic/project-deep-learning What did MNIST from Scratch not do?::No fresh virtual environment (it used the existing one), no CNN, GPU, momentum or Adam, no validation set (the final table is a comparison, not model selection), only 3 seeds, and training stopped after 2,000 steps, about 2 passes over MNIST, far from converged.

#flashcards/projects/concept #flashcards/topic/project-deep-learning How was reproducibility checked?::The five code blocks were extracted into a new folder and `lab.py`, `check_np.py` and `ablate.py` rerun. The first two printed identical output, and every accuracy and NaN count from `ablate.py` was identical. Only timings differed (seeded, single-thread, CPU). It was one rerun.

#flashcards/projects/cloze #flashcards/topic/project-deep-learning A bad ==weight start== and a ==learning rate of 1.0== both broke training on both datasets.
