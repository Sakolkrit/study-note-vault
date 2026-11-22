---
tags: [LLM, machine-learning, training, compression, efficiency, distillation]
status: growing
level: 4
tier: 3
---

# Knowledge Distillation

Knowledge distillation means **training a small "student" model to copy the answers of a big "teacher" model**, so the student ends up much better than if it had learned on its own, while staying small and cheap to run.

## 🟢 1. Simple (age 12)
Imagine you are learning to cook. You could learn from a recipe book that only says "this dish is a cake". Or you could learn from a great chef who tells you: "This is mostly a cake, a little bit like a muffin, and nothing like soup." That second kind of hint teaches you far more. Distillation is a **small model learning from a big model's hints**, not only from the plain right answers.

**Example:** A teacher shows a picture of a dog and says "90% dog, 9% wolf, 1% car". The student learns not just "dog" but also "dogs look a bit like wolves, and nothing like cars". That extra detail helps it guess better on new pictures.

## 🔵 2. How it works
A big trained [[Neural Network]] is the teacher. We want a smaller one, the student, that behaves almost the same.
1. **Train the teacher** the normal way, or take one that already exists (for example a big [[LLM]]).
2. **Ask the teacher** about many inputs. For each one it gives scores for every possible answer, which a [[Softmax]] turns into probabilities. These are called **soft labels**.
3. **Soften them more.** Often the teacher is too sure ("99.9% dog"), which hides its hints. A setting called the **temperature** flattens the probabilities so the small ones become visible ("dog 51%, wolf 31%, car 19%").
4. **Train the student** so its softened answers match the teacher's softened answers. The measure of mismatch is a [[KL Divergence]].
5. **Mix in the true answers** if you have them, so the student does not copy the teacher's mistakes.
6. **Use the student alone.** It is smaller and faster, but kept a lot of what the teacher knew.

![[knowledge-distillation-example.png]]
*How distillation is wired. The same input x goes to the teacher (blue) and the student (green). Both produce scores, which a softmax at temperature T turns into soft labels and soft predictions. The gap between them is the distillation loss. The student also makes a normal prediction at T = 1 that is compared with the true hard label y (the student loss). Image: Wikimedia Commons file [Knowledge-distillation-example.png](https://commons.wikimedia.org/wiki/File:Knowledge-distillation-example.png), by arvindpdmn and Varsha2018, CC0.*

## 🟠 3. Technical detail
- **Why it works:** a hard label is one number of information ("class 0"). A soft label from a good teacher says how the classes relate to each other. The student gets much more signal per example, so it can learn from fewer examples (the code below shows this with a toy).
- **Temperature $T$:** $T = 1$ is the ordinary softmax. Larger $T$ flattens the distribution. The student and teacher use the same $T$ while training, and the student uses $T = 1$ when it is deployed.
- **Two loss terms:** one compares the student with the teacher (soft), the other compares the student with the true labels (hard). A weight $\alpha$ balances them. With no true labels at all you can use only the soft term.
- **The $T^2$ factor:** when $T$ is large the soft-target gradients get small, so the soft term is multiplied by $T^2$ to keep it comparable to the hard term (the idea comes from the original distillation paper, written from memory).
- **A transfer set:** the teacher can label as many inputs as you can give it, including inputs that have no true label. So distillation can use a lot of unlabelled data. That is a big part of why it helps, and in the code below it is the main reason the student wins.
- **For language models:** a student can match the teacher's next-token probabilities at every position (see [[Pre-training]]), or it can be fine-tuned on text that the teacher generated (see [[Fine-tuning]]). The second kind is simpler and common (from memory; practice varies). A small model trained on a big model's outputs is often called "distilled".
- **Beyond outputs:** some methods also match the teacher's hidden layers or its [[Attention]] patterns, not only the final probabilities (from memory, not covered here).
- **Compared with [[Quantization]]:** quantization keeps the same model and stores its numbers with fewer bits. Distillation builds a different, smaller model. You can do both: distil first, then quantize.
- **Limits:** the student cannot learn what its size cannot hold, and it inherits the teacher's errors and biases. Quality depends on how well the teacher's answers cover the cases you care about.

```python
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
def make(n, seed):                      # 3 spiral-ish classes in 2-D
    r = np.random.default_rng(seed); y = r.integers(0, 3, n); t = r.uniform(0.3, 3, n)
    a = t + y * 2 * np.pi / 3
    X = np.c_[t * np.cos(a), t * np.sin(a)] + r.normal(0, 0.15, (n, 2))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y)
def net(h): return nn.Sequential(nn.Linear(2, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 3))
def fit(m, X, loss_fn, steps=1500):
    opt = torch.optim.Adam(m.parameters(), 0.01)
    for _ in range(steps):
        opt.zero_grad(); loss_fn(m(X)).backward(); opt.step()
    return m
acc = lambda m, X, y: (m(X).argmax(1) == y).float().mean().item()
Xte, yte = make(4000, 99)
Xbig, ybig = make(3000, 1)              # plenty of labels: teacher only
torch.manual_seed(0)
teacher = fit(net(64), Xbig, lambda z: F.cross_entropy(z, ybig), 2000)
print('teacher acc', round(acc(teacher, Xte, yte), 3))
T, alpha = 4.0, 0.5
res = {'hard': [], 'distill': []}
for seed in range(10):
    Xs, ys = make(30, 100 + seed)       # student sees only 30 labels
    Xu, _ = make(1500, 200 + seed)      # plus unlabelled points the teacher can label
    with torch.no_grad(): pt_s = F.softmax(teacher(Xs) / T, 1); pt_u = F.softmax(teacher(Xu) / T, 1)
    torch.manual_seed(seed)
    h = fit(net(8), Xs, lambda z: F.cross_entropy(z, ys))
    torch.manual_seed(seed)
    def kd(z_all):                       # z_all: student logits on [Xs; Xu]
        zs, zu = z_all[:30], z_all[30:]
        hard = F.cross_entropy(zs, ys)
        soft = F.kl_div(F.log_softmax(torch.cat([zs, zu]) / T, 1), torch.cat([pt_s, pt_u]), reduction='batchmean') * T * T
        return (1 - alpha) * hard + alpha * soft
    d = fit(net(8), torch.cat([Xs, Xu]), kd)
    res['hard'].append(acc(h, Xte, yte)); res['distill'].append(acc(d, Xte, yte))
for k, v in res.items(): print(k, round(float(np.mean(v)), 3), round(float(np.std(v)), 3))
print('wins', sum(d > h for d, h in zip(res['distill'], res['hard'])), 'of 10')
# teacher acc 0.997
# hard 0.903 0.044
# distill 0.996 0.0
# wins 10 of 10        (takes a few minutes on a CPU; PyTorch 2.8.0 used)
```
The student has 123 weights, the teacher 4,547, and the student sees only 30 true labels. Be careful with the conclusion: the distilled student also saw 1,500 extra unlabelled points that the teacher labelled. The test does not separate "soft labels carry more information" from "more inputs". Both are real effects of distillation, but this toy does not measure them apart.

## 🔴 4. Deep / formal

**Softmax with temperature.** For a model with scores (logits) $z_1,\dots,z_K$, the probability of class $i$ at temperature $T$ is

$$p_i(T) = \frac{e^{z_i / T}}{\sum_{j=1}^{K} e^{z_j / T}}$$

**Where:**
- $z_i$ is the raw score the model gives to class $i$. **Matters:** only differences between scores change the result.
- $T$ is the temperature, a positive number. **Matters:** $T=1$ gives the ordinary softmax. A large $T$ pushes all probabilities toward $1/K$, and a small $T$ pushes them toward a single 1. Without it, a confident teacher would show almost no hints about the wrong classes.
- $K$ is the number of classes. **Matters:** it sets the flat limit $1/K$.
- $p_i(T)$ is the probability of class $i$. **Matters:** these are the soft labels or soft predictions.

**Example (tiny numbers):** Scores $z = (4, 2, 0)$.
1. At $T = 1$: $e^4, e^2, e^0 = 54.60, 7.39, 1.00$. Sum 62.99. Probabilities 0.867, 0.117, 0.016.
2. At $T = 4$: divide the scores by 4 to get (1, 0.5, 0). $e^1, e^{0.5}, e^0 = 2.718, 1.649, 1.000$. Sum 5.367. Probabilities 0.506, 0.307, 0.186.
3. Meaning: at $T=1$ the runner-up classes look almost irrelevant (0.117 and 0.016). At $T=4$ the ranking is the same, but the teacher's opinion that class 2 is closer than class 3 is now easy to see and to learn from.

**The distillation loss.** With teacher probabilities $p^{t}$ and student probabilities $p^{s}$, both at temperature $T$, and the true class $y$:

$$\mathcal{L} = \alpha\, T^2\, \mathrm{KL}\!\left(p^{t}(T)\,\|\,p^{s}(T)\right) + (1-\alpha)\,\bigl(-\log p^{s}_{y}(1)\bigr)$$

**Where:**
- $\mathrm{KL}(p^t \| p^s)$ is the [[KL Divergence]] from the teacher's softened answer to the student's. **Matters:** it is zero only when the student matches the teacher exactly. Without it, the student gets no hints beyond the true label.
- $-\log p^s_y(1)$ is the usual [[Loss Function]] (cross-entropy) on the true label, at $T=1$. **Matters:** it keeps the student tied to the real answers, so it does not copy teacher mistakes.
- $\alpha$ is a mixing weight between 0 and 1. **Matters:** $\alpha=1$ uses only the teacher, $\alpha=0$ is normal training with no teacher.
- $T^2$ is a scale factor. **Matters:** soft-term gradients shrink as $T$ grows. Without $T^2$ the soft term would fade next to the hard term (see the next equation).

**Example (tiny numbers):** Teacher scores $(4,2,0)$, student scores $(3, 2.5, 0)$, true class 0, $T=4$, $\alpha=0.5$.
1. Teacher at $T=4$: (0.506, 0.307, 0.186). Student at $T=4$: (0.425, 0.375, 0.201).
2. $\mathrm{KL} = 0.0144$. Times $T^2 = 16$ gives 0.2309.
3. The student at $T=1$ is (0.604, 0.366, 0.030). Cross-entropy on class 0 is $-\ln 0.604 = 0.5046$.
4. $\mathcal{L} = 0.5 \times 0.2309 + 0.5 \times 0.5046 = 0.3678$.
5. Meaning: the student is a bit too unsure of class 0 and a bit too keen on class 1, compared with the teacher. Both terms push it in the same direction. Training lowers $\mathcal{L}$.

**Gradient of the soft term.** Without the $T^2$ factor, the gradient of the KL term with respect to the student's score $z^s_i$ is

$$\frac{\partial\, \mathrm{KL}}{\partial z^{s}_i} = \frac{1}{T}\left(p^{s}_i(T) - p^{t}_i(T)\right)$$

**Where:**
- $z^s_i$ is the student's score for class $i$. **Matters:** this is what training adjusts.
- $p^s_i(T) - p^t_i(T)$ is student minus teacher, at temperature $T$. **Matters:** the gradient is zero exactly when they agree. This is the same simple form as ordinary cross-entropy, with the teacher's softened answer in place of the true label.
- $1/T$ is the extra factor from the temperature. **Matters:** together with the fact that softened probabilities differ less from each other, this makes soft gradients roughly $1/T^2$ as large (the usual argument, from memory), which is why the loss multiplies by $T^2$.

**Example (tiny numbers):** Same scores as above, $T=4$.
1. Student minus teacher at $T=4$: (0.425−0.506, 0.375−0.307, 0.201−0.186) = (−0.081, 0.068, 0.015).
2. Divide by $T=4$: gradient (−0.0205, 0.0169, 0.0036). A finite-difference check gives the same numbers.
3. At $T=1$ the same difference would be (0.604−0.867, 0.366−0.117, 0.030−0.016) = (−0.263, 0.249, 0.014). So the $T=4$ gradient is about 13 times smaller.
4. Multiplying the loss by $T^2 = 16$ makes the gradient $T(p^s - p^t) = $ (−0.327, 0.270, 0.057), back to the size of the $T=1$ one.
5. Meaning: a negative number at class 0 means "raise the student's score for class 0", which is right because the teacher is more confident there.

**Hidden knowledge in the small numbers.** Hinton and colleagues called the information in the small probabilities "dark knowledge" (from memory). It is why a student can learn well from few labels: the wrong-class probabilities describe how the classes are close or far.

**Open questions and cautions (all from memory, check before relying on them):**
- A student much smaller than the teacher can fail to match it, and a middle-sized "assistant" teacher sometimes helps (the "capacity gap").
- Distilled models can pass on a teacher's biases and hallucinations. Check the output licence and terms of use before training on a commercial model's text.
- For generative models, matching only sampled text throws away the teacher's full probabilities, so it carries less information than matching them at each token.

**What came before, and how this improved on it:**
- **Before:** use a big model (or an ensemble of models) as it is, or train the small model on hard labels only.
- **Its limit:** big models are costly to run, and hard labels carry little information per example.
- **What changed:** train the small model to copy the big model's full probabilities, which also show which wrong answers are nearly right (Hinton, Vinyals and Dean, arXiv 1503.02531, 9 March 2015, abstract read; the author names are from memory). DistilBERT cut a BERT model by 40% while keeping 97% of its language understanding and being 60% faster (arXiv 1910.01108, 2 October 2019, abstract read).
- **How much better:** in the code below (handwritten digits), the teacher, a 128-unit network trained on 600 labelled pictures, scores 0.934. A one-layer student trained on 60 labelled pictures with hard labels scores 0.760. The same student trained to copy the teacher's probabilities on 1,000 other unlabelled pictures scores 0.923. **Fair-play warning:** the student in the second case saw many more pictures, which the teacher labelled for free, so this shows the value of the teacher's labels, not a like-for-like comparison.

**Check it yourself:**

```python
import numpy as np, torch
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
torch.manual_seed(0)
X, y = load_digits(return_X_y=True); X = X / 16
Xte, yte = X[1000:], y[1000:]
teacher = MLPClassifier((128,), max_iter=500, random_state=0).fit(X[:600], y[:600])
print("teacher (128 hidden units, 600 labelled pictures): test accuracy", round(teacher.score(Xte, yte), 3))
def student(Xs, targets, soft):
    W = torch.zeros(64, 10, requires_grad=True); b = torch.zeros(10, requires_grad=True)
    opt = torch.optim.Adam([W, b], 0.05); Xs = torch.tensor(Xs, dtype=torch.float32); T = torch.tensor(targets, dtype=torch.float32 if soft else torch.long)
    for _ in range(400):
        opt.zero_grad(); out = Xs @ W + b
        l = -(T * torch.log_softmax(out, 1)).sum(1).mean() if soft else torch.nn.functional.cross_entropy(out, T)
        l.backward(); opt.step()
    p = (torch.tensor(Xte, dtype=torch.float32) @ W + b).argmax(1).numpy(); return (p == yte).mean()
print("small student (one linear layer), 60 labelled pictures, hard labels:", round(student(X[:60], y[:60], False), 3))
print("same student, trained to copy the teacher's probabilities on 1,000 unlabelled pictures:", round(student(X[600:1000], teacher.predict_proba(X[600:1000]), True), 3))
```

## Related
[[Quantization]] · [[KL Divergence]] · [[Softmax]] · [[Fine-tuning]] · [[Pre-training]] · [[Loss Function]] · [[Neural Network]] · [[LLM]] · [[Entropy]] · [[Mixture of Experts]] · [[Diffusion Models]] · [[BERT]] · [[Speculative Decoding]] · [[Reasoning Model]]

## Sources
- Knowledge distillation proposed by Hinton, Vinyals and Dean (2015), "Distilling the Knowledge in a Neural Network" (cited from memory; not opened for this note).
- Earlier idea of model compression by Bucilă, Caruana and Niculescu-Mizil (2006) (from memory).
- Diagram: [Wikimedia Commons, Knowledge-distillation-example.png](https://commons.wikimedia.org/wiki/File:Knowledge-distillation-example.png), CC0.
- The code, the Example numbers and the gradient check were run for this note (PyTorch 2.8.0, NumPy).
- Hinton et al., arXiv 1503.02531 and Sanh et al., DistilBERT, arXiv 1910.01108 (abstracts read for the history block)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/training Knowledge distillation:::A small student model learns from a big teacher model's soft hints (its full probability list), not only from the plain right answers.

#flashcards/ml/formula #flashcards/topic/training Write the softmax with temperature and explain each symbol.
?
$$p_i(T) = \frac{e^{z_i / T}}{\sum_{j=1}^{K} e^{z_j / T}}$$
- $z_i$: the raw score for class $i$. Only differences between scores matter.
- $T$: the temperature. $T=1$ is the ordinary softmax. A large $T$ pushes all probabilities toward $1/K$, a small $T$ toward a single 1. Without it a confident teacher would show almost no hints about wrong classes.
- $K$: the number of classes. It sets the flat limit $1/K$.
- $p_i(T)$: the probability of class $i$ (the soft labels or soft predictions).

#flashcards/ml/example #flashcards/topic/training Teacher scores $(4,2,0)$. What are the probabilities at $T=1$ and $T=4$?::$T=1$: 0.867, 0.117, 0.016. $T=4$: 0.506, 0.307, 0.186. At $T=1$ the runner-up classes look almost irrelevant. At $T=4$ the teacher's view that class 2 is closer than class 3 is easy to see and learn from.

#flashcards/ml/formula #flashcards/topic/training Write the distillation loss and explain each symbol.
?
$$\mathcal{L} = \alpha\, T^2\, \mathrm{KL}\!\left(p^{t}(T)\,\|\,p^{s}(T)\right) + (1-\alpha)\,\bigl(-\log p^{s}_{y}(1)\bigr)$$
- $\mathrm{KL}(p^t\|p^s)$: the KL divergence from the teacher's softened answer to the student's. Zero only when the student matches the teacher. Without it the student gets no hints beyond the true label.
- $-\log p^s_y(1)$: the usual cross-entropy on the true label at $T=1$. It keeps the student tied to real answers so it does not copy teacher mistakes.
- $\alpha$: a mixing weight (0 to 1). $\alpha=1$ uses only the teacher, $\alpha=0$ is normal training.
- $T^2$: a scale factor, because soft-term gradients shrink as the temperature grows.

#flashcards/ml/example #flashcards/topic/training Teacher $(4,2,0)$, student $(3,2.5,0)$, true class 0, $T=4$, $\alpha=0.5$. What is the loss?::Teacher at $T=4$: (0.506, 0.307, 0.186). Student: (0.425, 0.375, 0.201). KL $=0.0144$, times 16 gives 0.2309. Student at $T=1$: (0.604, 0.366, 0.030), cross-entropy $-\ln0.604=0.5046$. $\mathcal{L}=0.5\times0.2309+0.5\times0.5046=0.3678$.

#flashcards/ml/formula #flashcards/topic/training Write the gradient of the soft-label term and explain each symbol.
?
$$\frac{\partial\, \mathrm{KL}}{\partial z^{s}_i} = \frac{1}{T}\left(p^{s}_i(T) - p^{t}_i(T)\right)$$
- $z^s_i$: the student's score for class $i$. This is what training adjusts.
- $p^s_i(T)-p^t_i(T)$: student minus teacher at temperature $T$. The gradient is zero exactly when they agree. It has the same simple form as ordinary cross-entropy, with the teacher's softened answer in place of the true label.
- $1/T$: the extra factor from the temperature. Together with softened probabilities being closer to each other, soft gradients are roughly $1/T^2$ as large (the usual argument, from memory), which is why the loss multiplies by $T^2$.

#flashcards/ml/example #flashcards/topic/training Same scores, $T=4$. What is the gradient?::Student minus teacher: $(-0.081, 0.068, 0.015)$. Divide by 4: $(-0.0205, 0.0169, 0.0036)$ (a finite-difference check gives the same numbers).

#flashcards/ml/image #flashcards/topic/training ![[knowledge-distillation-example.png]]
What does this picture show?
?
How distillation is wired. The same input $x$ goes to the teacher (blue) and the student (green). Both produce scores, which a softmax at temperature $T$ turns into soft labels and soft predictions. The gap between them is the distillation loss. The student also makes a normal prediction at $T=1$ that is compared with the true hard label (the student loss). (From the note's caption, Wikimedia Commons.)

#flashcards/ml/cloze #flashcards/topic/training In distillation, $\alpha=1$ uses only the ==teacher== and $\alpha=0$ is ==normal training==.
