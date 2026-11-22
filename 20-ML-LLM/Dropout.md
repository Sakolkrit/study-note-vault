---
tags: [machine-learning, regularization, dropout, neural-network, overfitting]
status: growing
level: 4
tier: 2
---

# Dropout

## 🟢 1. Simple (age 12)
Dropout is a training trick where the computer **randomly switches off some of its own "brain cells" each time it practises**. It sounds silly, but it works: because any cell might vanish at any moment, no cell can lazily depend on one particular friend, so the whole network has to learn **many independent ways** to get the right answer. That makes it more reliable on new questions.

**Example:** A football team where, in every practice, the coach randomly sends a few players to the bench. The remaining players must learn to cover for the missing ones. On match day, with everyone back on the pitch, the team is far more robust than one that always practised with the same star striker.

## 🔵 2. How it works
1. A [[Neural Network]] is made of layers of small units (neurons). A **big** network can memorise its training data, which is [[Overfitting]].
2. At **each training step**, for every unit in a chosen layer, flip a biased coin. With chance $p$ (for example 0.5) the unit is **dropped**: its output is set to 0 for this step. Otherwise it is **kept**.
3. Run the forward pass and [[Backpropagation]] using only the kept units. A different random set is dropped at the next step.
4. The units that remain get their outputs **scaled up** a little (divided by the keep chance), so that the layer's total signal stays about the same size on average.
5. Units cannot rely on any particular partner being there, so they learn **useful features by themselves** instead of "co-adapting" into fragile teams.
6. **At test time, dropout is switched off.** The whole network is used, with no random dropping and no extra scaling.

![[dropout-diagram.svg]]
*Dropout. Left: a normal network where every unit is connected to every unit in the next layer. Right: the same network during one training step. Some hidden units, marked with red crosses, have been dropped, so they and their connections do nothing for this step. A different random set is dropped at the next step. Image: Mads Dyrmann, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Neural_Network_Dropout.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Type:** a [[Regularization]] method that works by adding noise during training, not by adding a penalty to the loss.
- **The drop rate:** written $p$, the chance a unit is dropped. Typical values are around $0.1$ to $0.5$ (a habit, not a rule; check on your own problem). A higher rate is a stronger regulariser, and too high makes the model underfit ([[Bias-Variance]]).
- **Where it is applied:** usually on the hidden layers of a [[Neural Network]], sometimes on inputs, and in a [[Transformer]] on the outputs of the attention and feed-forward blocks. Many large [[LLM]]s trained on huge data use little or no dropout, because with enough data overfitting is less of a worry (a common observation; check for the model you care about).
- **Inverted dropout:** the version in common libraries scales the kept units **up during training**, so that at test time nothing needs to change. This is what the equations below use.
- **Training vs test mode:** libraries have a switch (for example "train" and "eval"). Forgetting to turn dropout off at test time gives noisy, worse predictions.
- **An ensemble view:** each random choice of kept units is a different "thinned" network, and they all share the same weights. Training with dropout is like training an enormous number of sub-networks at once, and using the full network at test time approximates averaging them (an idea from the original work; from memory).
- **Other noise-based tricks** with a similar spirit: randomly masking input words or image patches, and "stochastic depth" (dropping whole layers). Not covered here.
- **Dropout does not replace** more data, [[Normalization]], or early stopping, and it slows training since the signal is noisier.

```python
import numpy as np
rng = np.random.default_rng(1)
h = np.array([2., 4., 6., 8.]); p = 0.5               # a layer's outputs, drop chance 0.5
mask = np.array([1, 0, 1, 0])                         # one random draw: keep units 1 and 3, drop 2 and 4
print(mask * h / (1 - p))                             # [ 4.  0. 12.  0.]   kept ones are scaled up by 1/(1-p) = 2

# average over many random masks: it comes back to the original h
outs = [(rng.random(4) > p) * h / (1 - p) for _ in range(20000)]
print(np.mean(outs, 0).round(2))                      # [2.   4.02 5.99 7.95]   close to [2, 4, 6, 8]
```

## 🔴 4. Deep / formal
**The mask and the scaling.** For a layer with outputs $h_1,\dots,h_n$, draw independent random numbers $m_i$ that equal $1$ (keep) with probability $1-p$ and $0$ (drop) with probability $p$. The output used during training is:

$$
\tilde h_i=\frac{m_i\,h_i}{1-p},\qquad m_i\sim\text{Bernoulli}(1-p)
$$

**Where:**
- $h_i$ is the output of unit $i$ before dropout, and $\tilde h_i$ is what is passed on during a training step.
- $m_i$ is the **mask** for unit $i$: a random 0 or 1. "Bernoulli($1-p$)" means a yes/no coin that comes up 1 with chance $1-p$ (see [[Probability]]).
- $p$ is the **drop rate**, the chance of setting a unit to zero. $1-p$ is the **keep chance**.
- Dividing by $1-p$ **scales up** the kept units. **Matters:** without it, the layer's average output would shrink by the factor $1-p$ during training and then be mismatched at test time, when nothing is dropped.
- A dropped unit gives $\tilde h_i=0$, so it also receives no gradient for that step.

**Example (tiny numbers):** outputs $h=(2,4,6,8)$ and $p=0.5$, so the scale is $1/(1-0.5)=2$. One random draw gives the mask $m=(1,0,1,0)$.
- $\tilde h=(1\cdot2,\ 0\cdot4,\ 1\cdot6,\ 0\cdot8)\cdot2=\mathbf{(4,\ 0,\ 12,\ 0)}$.
- Meaning: two units were silenced and the two survivors were doubled. Another draw would silence a different pair, so the next step sees a different view of the same layer.

**Why the scaling is exactly right.** Averaged over the random mask, dropout leaves each output unchanged:

$$
\mathbb E[\tilde h_i]=\frac{\mathbb E[m_i]\,h_i}{1-p}=\frac{(1-p)\,h_i}{1-p}=h_i
$$

**Where:**
- $\mathbb E[\cdot]$ is the expected value, the average over many random masks (see [[Probability]]).
- $\mathbb E[m_i]=1-p$ because $m_i$ is 1 with chance $1-p$ and 0 otherwise.
- $h_i$ is a fixed number during one step, so it comes out of the average unchanged.
- **Matters:** this shows that the noise is **unbiased**: on average, the layer behaves as if there were no dropout. So at test time, the plain network with all units on (and no scaling) matches the *average* behaviour that training was shaped around.
- **Caution:** this is exact for the average of a single layer's output. The network applies non-linear functions afterwards, so test-time behaviour is only an approximation to averaging many thinned networks.

**Example (tiny numbers):** one unit with $h_i=4$ and $p=0.5$.
- With chance $0.5$ it is kept and outputs $4/0.5=8$. With chance $0.5$ it is dropped and outputs $0$.
- Average: $0.5\cdot8+0.5\cdot0=\mathbf{4}$, which is $h_i$ again.
- The code above checks the whole layer: with 20000 random masks the averages $(2.0,\ 4.02,\ 5.99,\ 7.95)$ come back to $(2,4,6,8)$, up to sampling noise.
- Meaning: dropout adds randomness to each step but does not push the layer's output up or down on average.

**Further notes:**
- **Why it reduces overfitting:** the noise stops units from memorising the exact combinations of the training data, and acts like averaging over many smaller networks (lower variance, see [[Bias-Variance]]). For simple linear models it behaves roughly like an L2 penalty ([[Regularization]]; an analysis from the research literature, stated from memory).
- **Monte Carlo dropout:** keeping dropout on at test time and averaging several predictions gives a rough estimate of the model's uncertainty. A research technique; reliability varies (from memory).
- **Cost:** training usually needs more steps to converge, because each step is noisier.
- **Different layers, different rates:** it is common to use a smaller rate near the input and a larger one in wide hidden layers.
- **Origin:** introduced by Hinton and colleagues around 2012, with the main paper by Srivastava et al. in 2014 (from memory).

**What came before, and how this improved on it:**
- **Before:** shrinking the weights ([[Regularization]]), stopping training early, and averaging the answers of several separately trained networks (an *ensemble*), which is costly because every network must be trained and run.
- **Its limit:** big networks trained on small data sets overfit ([[Overfitting]]), and an ensemble multiplies the cost.
- **What changed:** Hinton and colleagues (arXiv 1207.0580, 3 July 2012, abstract page read) switch off about half of the feature detectors at random on each training case. Their abstract says this greatly reduces overfitting and gave large gains and new records on benchmarks, with no specific numbers in the abstract.
- **How much better:** a toy check of my own (below): a network with two 256-unit layers trained on only 100 handwritten 8×8 digits reached training accuracy 1.0 with and without dropout. Its mean test accuracy over 5 seeds was 0.9079 without dropout and 0.9142 with dropout 0.5, a gain of 0.6 points. That is small, in one toy case, and I did not test other sizes or step counts, so the check shows the direction and not the size.

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
torch.set_num_threads(1)
D = load_digits(); X = torch.tensor(D.data / 16.0, dtype=torch.float32); y = torch.tensor(D.target.astype(np.int64))
idx = np.random.RandomState(0).permutation(len(y)); small, te = idx[:100], idx[1000:]
acc = lambda m, ids: float((m(X[ids]).argmax(1) == y[ids]).float().mean())
# Dropout: a wide network on only 100 training pictures, with and without dropping half of the units at every step
def dropout_run(p, seed):
    torch.manual_seed(seed); m = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Dropout(p), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(p), nn.Linear(256, 10)); opt = torch.optim.Adam(m.parameters(), 0.003)
    for _ in range(300): opt.zero_grad(); F.cross_entropy(m(X[small]), y[small]).backward(); opt.step()
    m.eval(); return acc(m, small), acc(m, te)
for p in (0.0, 0.5):
    r = np.array([dropout_run(p, s) for s in range(5)]).mean(0).round(4)
    print('dropout p =', p, '| mean over 5 seeds: train accuracy', r[0], ', test accuracy', r[1])
```
Output:
- `dropout p = 0.0 | mean over 5 seeds: train accuracy 1.0 , test accuracy 0.9079`
- `dropout p = 0.5 | mean over 5 seeds: train accuracy 1.0 , test accuracy 0.9142`

## Related
- [[Regularization]] · [[Overfitting]] · [[Bias-Variance]] · [[Neural Network]] · [[Backpropagation]] · [[Normalization]] · [[Transformer]] · [[LLM]] · [[Probability]] · [[Learning Rate]] · [[PyTorch]] · [[TensorFlow]] · [[Pooling]] · [[Data Augmentation]]

## Sources
- Srivastava, Hinton, Krizhevsky, Sutskever & Salakhutdinov, *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*, 2014 (from memory)
- Hinton et al., *Improving neural networks by preventing co-adaptation of feature detectors*, 2012 (from memory)
- Goodfellow, Bengio & Courville, *Deep Learning*, 2016, chapter 7
- Gal & Ghahramani, *Dropout as a Bayesian Approximation*, 2016 (Monte Carlo dropout; from memory)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/training Dropout:::A training trick where the computer randomly switches off some of its own units each time it practises. No unit can lazily depend on one particular friend, so the whole network learns many independent ways to get the answer.

#flashcards/ml/formula #flashcards/topic/training Write dropout and explain each symbol.
?
$$\tilde h_i=\frac{m_i\,h_i}{1-p},\qquad m_i\sim\text{Bernoulli}(1-p)$$
- $h_i$: the output of unit $i$ before dropout. $\tilde h_i$: what is passed on during a training step.
- $m_i$: the mask for unit $i$, a random 0 or 1. Bernoulli($1-p$) is a coin that comes up 1 with chance $1-p$.
- $p$: the drop rate, the chance a unit is switched off.
- Dividing by $1-p$ scales the survivors up, so the average output stays the same.

#flashcards/ml/formula #flashcards/topic/training Show that dropout keeps the average output unchanged.
?
$$\mathbb E[\tilde h_i]=\frac{\mathbb E[m_i]\,h_i}{1-p}=\frac{(1-p)\,h_i}{1-p}=h_i$$
- $\mathbb E[\cdot]$: the expected value, the average over many random masks.
- $\mathbb E[m_i]=1-p$, because $m_i$ is 1 with chance $1-p$ and 0 otherwise.
- $h_i$ is a fixed number during one step, so it comes out of the average unchanged.
- So the noise does not shift the average, and the layer behaves the same on average when dropout is off.

#flashcards/ml/cloze #flashcards/topic/training Dropout divides the survivors by ==1 − p==, so the average output stays ==unchanged==.

#flashcards/ml/example #flashcards/topic/training Outputs $h=(2,4,6,8)$, $p=0.5$, mask $m=(1,0,1,0)$. What is $\tilde h$?::The scale is 1/(1 − 0.5) = 2. h̃ = (1·2, 0·4, 1·6, 0·8) × 2 = (4, 0, 12, 0). Two units are silenced and the two survivors are doubled.

#flashcards/ml/example #flashcards/topic/training One unit with $h_i=4$ and $p=0.5$. What is the average output?::Kept (chance 0.5): 4/0.5 = 8. Dropped (chance 0.5): 0. Average 0.5 × 8 + 0.5 × 0 = 4 = h_i.

#flashcards/ml/image #flashcards/topic/training ![[dropout-diagram.svg]]
What does this picture show?
?
Left: a normal network where every unit is connected to every unit in the next layer. Right: the same network during one training step. Some hidden units (red crosses) are dropped, so they and their connections do nothing for this step. A different random set is dropped at the next step.
