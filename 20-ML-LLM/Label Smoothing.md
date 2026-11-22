---
tags: [machine-learning, deep-learning, training, regularization, classification]
status: growing
level: 4
tier: 2
---

# Label Smoothing

**Label smoothing** softens the correct answers used in training, so the model is never told to be 100% sure.

## 🟢 1. Simple (age 12)
When you teach a model, you normally say "this picture is a cat, 100%, and nothing else". Label smoothing says "this is a cat, about 90%, and a tiny bit of everything else". The model learns to stay a little humble instead of shouting "100% sure!" about everything.

**Example:** A quiz where the answer key admits it might be wrong about 1 time in 10. A student who studies for that key learns not to be absolutely certain, even about answers they got right.

## 🔵 2. How it works
1. In normal training, the target for each example is a list that is 1 for the true class and 0 for all the others (a one-hot target; see [[Classification]] and [[Softmax]]).
2. To get the loss ([[Loss Function]]) down to zero, the model has to push its confidence for the true class towards 100%. That means huge scores, and the model becomes over-confident.
3. With label smoothing, a small amount $\varepsilon$ of the probability is taken from the true class and shared equally among all the classes.
4. For 10 classes and $\varepsilon=0.1$, the true class gets 0.91 and each of the other 9 gets 0.01.
5. Training now rewards a confidence of about 91%, not 100%, so the scores stay smaller.
6. The change is only in the training targets. At test time the model is used as usual.
7. **The catch:** it can make the model *under*-confident, it can slightly lower accuracy on easy problems, and it hides information that some later methods need. The size $\varepsilon$ has to be chosen.

## 🟠 3. Technical detail
- **Where it came from:** Section 7 of Szegedy et al., "Rethinking the Inception Architecture for Computer Vision" (arXiv 1512.00567, December 2015; I read that section from an HTML copy). The lecture slide dates the idea to a 2015 vision paper, and this is it. The paper calls it **label-smoothing regularization (LSR)**, and its reason is two problems with hard targets: the model may overfit if it learns to give full probability to the true label on every training example, and it pushes the biggest logit far above the others, which, with a gradient that is bounded between -1 and 1, reduces the model's ability to adapt.
- **Equivalent view (paper):** the smoothed loss is the same as $(1-\varepsilon)H(q,p)+\varepsilon H(u,p)$, a pair of cross-entropies: one against the true label and one that penalises the prediction for straying from the uniform distribution (relative weight $\varepsilon/(1-\varepsilon)$).
- **The paper's own setting and result:** for ImageNet with $K=1000$ classes it used $\varepsilon=0.1$ and reports a consistent improvement of about 0.2% absolute in both top-1 and top-5 error on ILSVRC 2012.
- **Use:** a general way to reduce [[Overfitting]] and over-confidence. The slides list benefits as preventing overfitting and improving accuracy and BLEU score for the translation Transformer ("Attention Is All You Need", 2017).
- **Typical size:** $\varepsilon=0.1$ is a common choice (a common habit; from memory).
- **Effect on confidence:** the best possible training confidence is $1-\varepsilon+\varepsilon/K$ for $K$ classes.
- **Effect on calibration:** it can fix over-confidence, but too much overshoots into under-confidence (see the measurement below and [[Calibration]]).
- **Relatives:** [[Dropout]] and [[Regularization]] also fight overfitting, and [[Data Augmentation]] adds variety.

```python
import numpy as np, torch
from sklearn.datasets import load_digits
torch.manual_seed(0)
X, y = load_digits(return_X_y=True); X = torch.tensor(X / 16, dtype=torch.float32); y = torch.tensor(y, dtype=torch.long)
Xtr, ytr, Xte, yte = X[:600], y[:600], X[600:], y[600:]
def train(eps, steps=500):
    W = torch.zeros(64, 10, requires_grad=True); b = torch.zeros(10, requires_grad=True); opt = torch.optim.Adam([W, b], 0.1)
    T = torch.nn.functional.one_hot(ytr, 10).float() * (1 - eps) + eps / 10           # smoothed target: (1-eps) on the truth, eps/K spread over all classes
    for _ in range(steps):
        opt.zero_grad(); loss = -(T * torch.log_softmax(Xtr @ W + b, 1)).sum(1).mean(); loss.backward(); opt.step()
    return Xte @ W + b
def ece(p, ok, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1); return sum((idx == k).mean() * abs(ok[idx == k].mean() - p[idx == k].mean()) for k in range(bins) if (idx == k).any())
for eps in (0.0, 0.02, 0.1):
    pr = torch.softmax(train(eps), 1); conf, pred = pr.max(1); ok = (pred == yte).numpy().astype(float); conf = conf.detach().numpy()
    print(f"label smoothing {eps}: test accuracy {ok.mean():.3f}, average confidence {conf.mean():.3f}, calibration error {ece(conf, ok):.3f}, highest confidence {conf.max():.3f}")
p = np.array([0.7, 0.1, 0.1, 0.1]); q = np.array([1.0, 0, 0, 0]); u = np.full(4, 0.25); H = lambda a: float(-(a * np.log(p)).sum())
print("two ways to write the smoothed loss (K=4, eps=0.1):", round((1 - 0.1) * H(q) + 0.1 * H(u), 4), "=", round(H((1 - 0.1) * q + 0.1 * u), 4))
print("on a training example the best possible confidence is capped at 1 - eps + eps/K =", round(1 - 0.1 + 0.1 / 10, 3), "for eps=0.1 and K=10 (a new example can still get more)")
```

## 🔴 4. Deep / formal
**The smoothed target.** For $K$ classes and a smoothing amount $\varepsilon$, the training target for class $k$ is

$$
y^{\text{smooth}}_k=(1-\varepsilon)\,y_k+\frac{\varepsilon}{K}
$$

**Where:**
- $y_k$ is the usual target: 1 if $k$ is the true class and 0 otherwise. $K$ is the number of classes.
- $\varepsilon$ (epsilon) is the smoothing amount, a number between 0 and 1. **Matters:** with $\varepsilon=0$ nothing changes. The larger it is, the less sure the model is told to be.
- $\varepsilon/K$ is the small share given to every class. **Matters:** because every class gets a little, the loss can no longer be pushed to zero by making the true class's chance 1, so the scores stop growing.

**Example (tiny numbers):** $K=4$ classes, $\varepsilon=0.1$, the true class is the first.
- True class: $(1-0.1)\cdot1+0.1/4=0.9+0.025=\mathbf{0.925}$.
- Each other class: $(1-0.1)\cdot0+0.1/4=\mathbf{0.025}$.
- Check: $0.925+3\cdot0.025=\mathbf{1.0}$.
- Meaning: the model is asked to put 92.5% on the true class, not 100%.

**The confidence cap.** The best a model can do on a training example is to output exactly the smoothed target, so its top confidence is capped:

$$
p_{\max}=1-\varepsilon+\frac{\varepsilon}{K}
$$

**Where:**
- $p_{\max}$ is the highest chance the model is rewarded for giving to one class.
- $\varepsilon$ and $K$ are as above.
- **Matters:** it shows how $\varepsilon$ sets the model's training-time humility. This is a cap on what training asks for, not a hard limit on the model: on new examples a model can still give more.

**Example (tiny numbers):** $K=10$ and $\varepsilon=0.1$.
- $p_{\max}=1-0.1+0.1/10=\mathbf{0.91}$.
- With $\varepsilon=0.02$: $1-0.02+0.002=\mathbf{0.982}$.
- Meaning: a bigger $\varepsilon$ asks for visibly less confidence.

**The same thing as two cross-entropies.** Writing the smoothed target as a mix of the true label $q$ and the uniform distribution $u$ (all classes equally likely), the loss splits into two parts:

$$
H(q',p)=(1-\varepsilon)\,H(q,p)+\varepsilon\,H(u,p)
$$

**Where:**
- $p$ is the model's predicted chances. $q$ is the true label as a one-hot list, $u$ is the uniform list ($1/K$ for each class), and $q'=(1-\varepsilon)q+\varepsilon u$ is the smoothed target.
- $H(a,p)=-\sum_k a_k\ln p_k$ is the cross-entropy of the prediction $p$ against the target $a$ ([[Loss Function]]).
- **Matters:** the first term is the usual loss. The second term charges the model for being far from "no idea", so it fights over-confidence. Without it, nothing stops the biggest score from growing.

**Example (tiny numbers):** $K=4$ classes, the true class is the first, the model predicts $p=(0.7,0.1,0.1,0.1)$, and $\varepsilon=0.1$.
- $H(q,p)=-\ln0.7=\mathbf{0.357}$.
- $H(u,p)=-\tfrac14(\ln0.7+3\ln0.1)=\tfrac14(0.357+6.908)=\mathbf{1.816}$.
- Total: $0.9\cdot0.357+0.1\cdot1.816=0.321+0.182=\mathbf{0.503}$.
- Check with the smoothed target $q'=(0.925,0.025,0.025,0.025)$: $-(0.925\ln0.7+3\cdot0.025\ln0.1)=0.330+0.173=\mathbf{0.503}$. The same number.
- Meaning: the two ways of writing it agree, so smoothing is exactly "ordinary loss plus a small pull towards uncertainty".

**Further notes:**
- **Not always a win:** the measurement below is a tiny linear model on handwritten digits. Label smoothing did not raise accuracy there, and a large $\varepsilon$ made the model under-confident. The slides' claim of better accuracy is about the translation Transformer, which I did not rerun.
- **Soft labels in general:** the smoothed target is the simplest kind of soft label. [[Knowledge Distillation]] uses the teacher's chances as soft labels.

**What came before, and how this improved on it:**
- **Before:** train on hard targets: 1 for the right class and 0 for the others.
- **Its limit:** the loss keeps rewarding more and more confidence, so the model's scores grow and it becomes over-confident, which hurts calibration and can overfit.
- **What changed:** use softened targets, the paper's label-smoothing regularization (Szegedy et al., arXiv 1512.00567, Section 7; abstract and that section read, first author from memory).
- **How much better:** the paper's own number (Section 7, copied by hand): with $\varepsilon=0.1$ and 1,000 ImageNet classes, a consistent improvement of about 0.2% absolute in both top-1 and top-5 error. I did not reproduce it. In the code below (handwritten digits, a linear model; my own toy), with no smoothing the average confidence is 0.960 at an accuracy of 0.921 (calibration error 0.040). With $\varepsilon=0.02$ the confidence is 0.886 and the calibration error 0.036, with accuracy 0.916. With $\varepsilon=0.1$ the model is under-confident (0.779 at an accuracy of 0.902, calibration error 0.123). **Honest result:** on this toy, smoothing lowered confidence as intended but gave no accuracy gain, and too much made calibration worse. The paper's gain is small (0.2 points) and was measured on a very different model.

**Check it yourself:** run the code in level 3 above. It prints every number quoted in this block.

## Related
[[Classification]] · [[Softmax]] · [[Loss Function]] · [[Overfitting]] · [[Regularization]] · [[Calibration]] · [[Dropout]] · [[Knowledge Distillation]] · [[Transformer]] · [[Data Augmentation]] · [[Cross-Entropy Loss]]

## Sources
- Stanford CME 295 (Fall 2025), lecture slides, Afshine and Shervine Amidi (lecture 1, the slide on label smoothing)
- Szegedy et al., *Rethinking the Inception Architecture for Computer Vision*, arXiv 1512.00567 (2015): abstract and Section 7 (label smoothing) read from an HTML copy of the paper; the first author is from memory. The 0.2% figure and the two-cross-entropy identity are the paper's.
- The slide's claim about improved accuracy and BLEU for the translation Transformer is the lecture's; I did not check it.
- No image: I found no free, on-topic picture for this note on Wikimedia Commons, so I added none.
- Every number in the history block was computed by the code in this note (toy data; dates and names marked "from memory" are not checked).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/training Label smoothing:::Instead of telling the model "this is a cat, 100%", say "this is a cat, about 90%, and a tiny bit of everything else". The model learns to stay a little humble.

#flashcards/ml/formula #flashcards/topic/training Write the smoothed target and explain each symbol.
?
$$y^{\text{smooth}}_k=(1-\varepsilon)\,y_k+\frac{\varepsilon}{K}$$
- $y_k$: the usual target, 1 if $k$ is the true class, 0 otherwise. $K$: the number of classes.
- $\varepsilon\in[0,1]$: the smoothing amount. With $\varepsilon=0$ nothing changes. The larger it is, the less sure the model is told to be.
- $\varepsilon/K$: the small share given to every class. Because every class gets a little, the loss can no longer be pushed to zero by making the true class's chance 1, so the scores stop growing.

#flashcards/ml/example #flashcards/topic/training $K=4$, $\varepsilon=0.1$, true class first. What are the smoothed targets?::True class $0.9+0.025=0.925$. Each other class $0.025$. Check $0.925+3\cdot0.025=1.0$. The model is asked for 92.5% on the true class, not 100%.

#flashcards/ml/formula #flashcards/topic/training Write the highest chance label smoothing rewards and explain each symbol.
?
$$p_{\max}=1-\varepsilon+\frac{\varepsilon}{K}$$
- $p_{\max}$: the highest chance the model is rewarded for giving to one class.
- $\varepsilon$, $K$ as above.
- It shows how $\varepsilon$ sets the training-time humility. It is a cap on what training asks for, not a hard limit: on new examples a model can still give more.

#flashcards/ml/example #flashcards/topic/training $K=10$, $\varepsilon=0.1$, and then $\varepsilon=0.02$. What is $p_{\max}$?::$1-0.1+0.01=0.91$. With $\varepsilon=0.02$: $1-0.02+0.002=0.982$. A bigger $\varepsilon$ asks for visibly less confidence.

#flashcards/ml/formula #flashcards/topic/training Write the smoothed loss as ordinary loss plus a pull toward uncertainty and explain each symbol.
?
$$H(q',p)=(1-\varepsilon)\,H(q,p)+\varepsilon\,H(u,p)$$
- $p$: the model's predicted chances. $q$: the true label as a one-hot list. $u$: the uniform list ($1/K$ each). $q'=(1-\varepsilon)q+\varepsilon u$: the smoothed target.
- $H(a,p)=-\sum_k a_k\ln p_k$: the cross-entropy of the prediction against the target $a$.
- The first term is the usual loss. The second charges the model for being far from "no idea", so it fights over-confidence. Without it nothing stops the biggest score from growing.

#flashcards/ml/example #flashcards/topic/training $K=4$, true class first, prediction $(0.7,0.1,0.1,0.1)$, $\varepsilon=0.1$. What is the loss two ways?::$H(q,p)=-\ln0.7=0.357$. $H(u,p)=\tfrac14(0.357+6.908)=1.816$. Total $0.9\cdot0.357+0.1\cdot1.816=0.503$. With the smoothed target $(0.925,0.025,0.025,0.025)$: $0.330+0.173=0.503$. Same number: smoothing is the ordinary loss plus a small pull toward uncertainty.

#flashcards/ml/cloze #flashcards/topic/training Label smoothing keeps the model from shouting ==100% sure== about everything.
