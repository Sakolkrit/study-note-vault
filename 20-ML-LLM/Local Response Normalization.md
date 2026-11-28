---
tags: [local-response-normalization, LRN, normalization, AlexNet, CNN, computer-vision, ML]
status: growing
level: 4
tier: 2
---

# Local Response Normalization

**Local response normalization (LRN)** is a step inside a [[Convolutional Neural Network]] that makes the strongly active feature detectors at one spot suppress their neighbours. For each position in a feature map it takes the activity of one detector (one channel) and divides it by a number that grows when the nearby channels at the same position are also strongly active. [[AlexNet 2012]] used it after the first two convolution layers; later networks mostly dropped it.

## 🟢 1. Simple (age 12)
Imagine a classroom where several students shout answers at once. A rule says: the louder the students next to you are, the quieter your voice counts. A student who is loud while the neighbours are quiet still stands out. When everybody around is loud, each voice counts less. That is how the network keeps one group of detectors from drowning out the rest at the same spot.

**Example:** at one spot in the picture, detector 4 reports 20 and its neighbours report 1, 4, 3 and 0.5. After the rule, detector 4 reports about 11.7. If its neighbours had also reported 100 each, a detector reporting 100 would drop to about 23 instead of staying near 44.

## 🔵 2. How it works
1. A convolution layer gives a [[Feature Map]] per kernel; at one position $(x,y)$ there is one number per kernel (the layer's "channels"), and [[ReLU]] has already been applied.
2. Pick a small **window of neighbouring kernels** (the paper uses 5: the kernel itself and two on each side). The order of the kernels is arbitrary and fixed before training.
3. Add up the **squares** of the activities in the window.
4. Compute $k+\alpha\times(\text{that sum})$ and raise it to the power $\beta$.
5. **Divide** the activity by this number.
6. Do the same for every kernel at every position. Nothing is learned in this step: $k$, $n$, $\alpha$ and $\beta$ are fixed numbers.
7. The effect: a strong activity is divided by more when its neighbours are strong too, so only the detectors that clearly win keep a large value ("lateral inhibition", like real neurons).

No image: the AlexNet figures are not under an open licence and I found no free picture of the operation. The numbers below come from a real run.

## 🟠 3. Technical detail
Numbers come from `lrn_demo.py` (Python 3.9.6, NumPy, PyTorch 2.8.0, CPU). It writes the AlexNet formula by hand (window 5, $k=2$, $\alpha=10^{-4}$, $\beta=0.75$), compares it with `torch.nn.LocalResponseNorm`, shows how an output falls as its neighbours grow, and applies the layer to a random map.

```python
"""Local response normalisation: the AlexNet formula by hand, against torch.nn.LocalResponseNorm."""
import torch, numpy as np
k, n, alpha, beta = 2.0, 5, 1e-4, 0.75
a = np.array([0., 1., 4., 20., 3., 0.5, 0., 2.])
def paper(a):
    b = np.zeros_like(a)
    for i in range(len(a)):
        lo, hi = max(0, i - n//2), min(len(a) - 1, i + n//2)
        b[i] = a[i] / (k + alpha*np.sum(a[lo:hi+1]**2))**beta
    return b
x = torch.tensor(a, dtype=torch.float32).reshape(1, 8, 1, 1)
t = torch.nn.LocalResponseNorm(size=n, alpha=alpha, beta=beta, k=k)(x).flatten().numpy()
t_scaled = torch.nn.LocalResponseNorm(size=n, alpha=alpha*n, beta=beta, k=k)(x).flatten().numpy()
print("input            ", a)
print("paper formula    ", np.round(paper(a), 4))
print("torch (same args)", np.round(t, 4), "| max difference %.4f" % np.abs(t - paper(a)).max())
print("torch, alpha x n ", np.round(t_scaled, 4), "| max difference %.2e" % np.abs(t_scaled - paper(a)).max())
big = np.array([100.]*5 + [0.]*3)
print("all five at 100: paper b[2] = %.3f, torch(alpha x n) = %.3f, torch(same args) = %.3f" % (paper(big)[2], torch.nn.LocalResponseNorm(5, alpha*5, beta, k)(torch.tensor(big, dtype=torch.float32).reshape(1, 8, 1, 1)).flatten()[2], torch.nn.LocalResponseNorm(5, alpha, beta, k)(torch.tensor(big, dtype=torch.float32).reshape(1, 8, 1, 1)).flatten()[2]))
# competition: raise the neighbours of a fixed activation and watch its output fall
for nb in (0, 10, 50, 100, 200):
    v = np.array([nb, nb, 100., nb, nb, 0, 0, 0.]); print("neighbours %3d -> output for 100: %.2f" % (nb, paper(v)[2]))
# what it does on a random post-ReLU activation map, share of change
torch.manual_seed(0); z = torch.relu(torch.randn(1, 96, 8, 8) * 30)
y = torch.nn.LocalResponseNorm(5, alpha*5, beta, k)(z)
print("random 96-channel map with values around 30: mean ratio output/input %.3f (min %.3f, max %.3f)" % ((y[z > 0] / z[z > 0]).mean(), (y[z > 0] / z[z > 0]).min(), (y[z > 0] / z[z > 0]).max()))
```

I saw:
```
input             [ 0.   1.   4.  20.   3.   0.5  0.   2. ]
paper formula     [ 0.      0.5855  2.3411 11.7055  1.7559  0.2928  0.      1.189 ]
torch (same args) [ 0.      0.5928  2.3708 11.8542  1.7781  0.2964  0.      1.1892] | max difference 0.1487
torch, alpha x n  [ 0.      0.5855  2.3411 11.7055  1.7559  0.2928  0.      1.189 ] | max difference 2.11e-07
all five at 100: paper b[2] = 23.237, torch(alpha x n) = 23.237, torch(same args) = 43.869
neighbours   0 -> output for 100: 43.87
neighbours  10 -> output for 100: 43.44
neighbours  50 -> output for 100: 35.36
neighbours 100 -> output for 100: 23.24
neighbours 200 -> output for 100: 10.99
random 96-channel map with values around 30: mean ratio output/input 0.545 (min 0.362, max 0.595)
```

**One position, eight kernel maps (activities 0, 1, 4, 20, 3, 0.5, 0, 2).**

| Kernel | Input | Paper's formula | PyTorch, same arguments | PyTorch with $\alpha\times n$ |
|---|---|---|---|---|
| 2 | 4 | 2.3411 | 2.3708 | 2.3411 |
| 3 | 20 | 11.7055 | 11.8542 | 11.7055 |
| 4 | 3 | 1.7559 | 1.7781 | 1.7559 |

- **Kernel:** the index of the kernel map (counting from 0). **Input:** its activity. The other columns are the normalised outputs for the paper's formula by hand, for `LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)`, and for the same layer with $\alpha$ multiplied by the window size 5.
- **PyTorch's convention differs from the paper's.** PyTorch divides $\alpha$ by the window size, so with the paper's $\alpha$ passed in directly the outputs differ (largest difference 0.149 here); passing $\alpha\times5$ gives the paper's formula to 2e-7 (my run). Anyone who copies "$\alpha=10^{-4}$" from the paper into PyTorch gets a weaker normalisation than AlexNet's.

**Competition (my run).** The activity 100 with a neighbourhood of values $nb$: $nb=0$ gives 43.87, $nb=10$ gives 43.44, $nb=50$ gives 35.36, $nb=100$ gives 23.24 and $nb=200$ gives 10.99. On a random post-ReLU map of 96 channels with values around 30, the output was on average 0.545 of the input (between 0.362 and 0.595).

**Summary: what the run shows.** With small activities the layer acts almost like a constant factor ($1/2^{0.75}=0.595$); only when the neighbours' squared activities add up to about 10,000 or more (so that $lpha$ times the sum is of order 1) does it suppress strongly: 100 falls from 43.9 (alone) to 23.2 (all five at 100) to 11.0 (neighbours at 200). Caution: toy numbers and a random map; this shows the formula, not how much it helped a trained network.

## 🔴 4. Deep / formal

$$
b^{i}_{x,y}=a^{i}_{x,y}\Bigg/\Bigg(k+\alpha\sum_{j=\max(0,\,i-n/2)}^{\min(N-1,\,i+n/2)}\bigl(a^{j}_{x,y}\bigr)^{2}\Bigg)^{\beta}
$$

**Where:**
- $a^{i}_{x,y}$: the activity of kernel $i$ at position $(x,y)$ after the ReLU; $b^{i}_{x,y}$: the normalised activity.
- The sum runs over the $n$ adjacent kernel maps at the same position, cut off at the first and last kernel; $N$ is the number of kernels in the layer.
- $k=2$, $n=5$, $\alpha=10^{-4}$, $\beta=0.75$: the constants of AlexNet, "determined using a validation set" (AlexNet paper, section 3.3).

**Matters:**
- The sum of squares makes a strong neighbourhood divide the activity more: competition between kernels at the same spot. The divisor is never below $k^{\beta}$ (here 1.68), so $k$ also keeps it away from zero.
- It is "brightness normalization": the mean is not subtracted, unlike the local contrast normalisation of Jarrett et al. that the paper compares it with.
- In AlexNet it lowered the top-1 and top-5 error by 1.4% and 1.2%, and a four-layer CNN on CIFAR-10 went from 13% to 11% test error (AlexNet paper, section 3.3).

**Example (tiny numbers):** activities $(0,1,4,20,3,0.5,0,2)$ with the paper's constants (my numbers).
- For kernel 3 (activity 20) the window covers kernels 1 to 5: $1+16+400+9+0.25=426.25$.
- Divisor: $(2+10^{-4}\times426.25)^{0.75}=1.7086$. Output: $20/1.7086=11.7055$.
- With all five maps of the window at 100: sum of squares 50,000, divisor $(2+5)^{0.75}=4.3035$, output $100/4.3035=23.2$. A lone 100 among zeros: divisor $3^{0.75}=2.2795$, output 43.9.
- Meaning: crowded strong neighbours cut the output of the same activity from 43.9 to 23.2.

**What the papers I read say.**
- [[AlexNet 2012]] (section 3.3 and 3.5): LRN is applied "after applying the ReLU nonlinearity in certain layers": after the first and second convolution layers, followed by max pooling. ReLU itself "does not require input normalization to prevent them from saturating", but the authors "still find that the following local normalization scheme aids generalization".
- The network tables of the SPP paper list "LRN" after conv1 and conv2 of the ZF-5 and Convnet*-5 networks and after conv1 of Overfeat (SPP paper, Table 1, as I read it in the text extraction).
- [[GoogLeNet 2014]]'s Figure 3 lists "LocalRespNorm" layers in its stem (only layer names are in the text I extracted).
- Later practice: [[Batch Normalization]] replaced it in most networks; this is general knowledge, not a claim of any paper I read for this note.

**Edge cases and details:**
- **Across channels, not across space:** the sum is over neighbouring kernels at one position, not over neighbouring positions.
- **Arbitrary channel order:** which kernels are "adjacent" is an accident of numbering, which is one reason later normalisations (batch and layer normalisation) do not use it (my reading).
- **Library conventions:** PyTorch divides $\alpha$ by the window size (my run above); other libraries may differ again, so check before copying the constants.

**Limits and trade-offs:**
- The AlexNet paper's evidence is a single comparison of 1.4% and 1.2%, with no error bar and no repeat.
- It has four hand-set constants and no learned part.
- It costs extra time per layer for a small gain, and it is not a general fix for training deep networks.

**What came before, and how this improved on it:**
- **Before:** no normalisation after ReLU, or the local contrast normalisation of Jarrett et al., which subtracts a local mean (AlexNet paper, section 3.3).
- **Its limit:** ReLU outputs are unbounded and nothing makes strong kernels compete (the paper's motivation is "lateral inhibition" as in real neurons).
- **What changed:** divide each activity by a function of the squared activities of nearby kernels.
- **How much better:** in AlexNet, 1.4% (top-1) and 1.2% (top-5) lower error, and 13% to 11% on a CIFAR-10 test (paper). In my run the number for an activity of 100 fell from 43.9 to 23.2 when its neighbours were 100; I did not train a network.

## Related
[[AlexNet 2012]] · [[GoogLeNet 2014]] · [[ReLU]] · [[Normalization]] · [[Batch Normalization]] · [[Layer Normalization]] · [[Convolutional Neural Network]] · [[Feature Map]] · [[Pooling]]

## Sources
- My own run: `lrn_demo.py` (Python 3.9.6, PyTorch 2.8.0), including the check against `torch.nn.LocalResponseNorm`.
- [[AlexNet 2012]] as read there (sections 3.3 and 3.5), the SPP paper's network table (text extraction), and [[GoogLeNet 2014]]'s figure labels. The local contrast normalisation paper (Jarrett et al.) was not read. The claim that batch normalisation later replaced LRN is general knowledge. The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision Local response normalization (LRN):::Dividing each activity by a function of the squared activities of nearby kernel maps at the same position, so strong detectors suppress their neighbours (lateral inhibition); used in AlexNet.

#flashcards/ml/concept #flashcards/topic/vision What are the steps of local response normalization?::For each position and kernel, square the activities of the n adjacent kernels (window of 5 in AlexNet), sum them, compute (k + alpha x sum)^beta and divide the activity by it; nothing is learned.

#flashcards/ml/formula #flashcards/topic/vision Write the local response normalization of AlexNet and explain each part.
?
$$b^{i}_{x,y}=a^{i}_{x,y}\Big/\Big(k+\alpha\sum_{j=\max(0,i-n/2)}^{\min(N-1,i+n/2)}(a^{j}_{x,y})^{2}\Big)^{\beta}$$
- $a^i_{x,y}$: activity of kernel $i$ at $(x,y)$ after ReLU; $b$: normalised activity.
- The sum is over $n$ adjacent kernel maps at the same position ($N$ kernels in all): strong neighbours make the divisor larger.
- $k=2$, $n=5$, $\alpha=10^{-4}$, $\beta=0.75$ (validation-set values); the divisor never falls below $k^\beta$.

#flashcards/ml/example #flashcards/topic/vision Activities (0, 1, 4, 20, 3, 0.5, 0, 2) with the AlexNet constants: what is the normalised value of the 20?::Window sum of squares 1 + 16 + 400 + 9 + 0.25 = 426.25; divisor (2 + 0.04263)^0.75 = 1.7086; output 20 / 1.7086 = 11.7055 (my arithmetic).

#flashcards/ml/example #flashcards/topic/vision What does LRN do to an activity of 100 when its five-kernel window is all 100, against when the others are zero?::Divisor (2 + 5)^0.75 = 4.3035, output 23.2; alone: (2 + 1)^0.75 = 2.2795, output 43.9. Strong neighbours almost halve the output (my arithmetic).

#flashcards/ml/concept #flashcards/topic/vision Why can copying AlexNet's alpha into PyTorch's LocalResponseNorm give a different result?::PyTorch divides alpha by the window size, so the paper's formula needs alpha x n (5 x 1e-4) there; with the paper's alpha passed directly the outputs were weaker (largest difference 0.149 on my example) (my run).

#flashcards/ml/concept #flashcards/topic/vision How much did local response normalization help AlexNet?::It lowered top-1 and top-5 error by 1.4% and 1.2%, and a four-layer CNN on CIFAR-10 went from 13% to 11% test error (AlexNet paper, section 3.3); single comparisons without error bars.

#flashcards/ml/concept #flashcards/topic/vision How does LRN differ from local contrast normalization?::LRN does not subtract the mean activity and works across neighbouring kernels at one position; the AlexNet paper therefore calls it "brightness normalization" (section 3.3).

#flashcards/ml/concept #flashcards/topic/vision What are the limits of local response normalization?::Four hand-set constants, a window over kernels whose order is arbitrary, a small and unrepeated gain in the one paper I read, and extra cost; batch normalization later replaced it in most networks (general knowledge).

#flashcards/ml/cloze #flashcards/topic/vision LRN divides each activity by a function of the ==squared== activities of ==neighbouring kernels== at the same position, a form of ==lateral inhibition==.
