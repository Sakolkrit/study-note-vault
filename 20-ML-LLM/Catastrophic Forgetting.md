---
tags: [neural-network, training, fine-tuning, continual-learning, LLM, forgetting]
status: growing
level: 4
tier: 3
---

# Catastrophic Forgetting

When a neural network learns something new and, in doing so, **suddenly loses what it knew before**. Not a slow fade: a sharp drop.

## 🟢 1. Simple (age 12)
Imagine one whiteboard that you use for every lesson. On Monday you fill it with French words. On Tuesday you need space for Spanish, so you erase and write over the same board. By Wednesday your French is gone, because the new lesson was written on top of the old one.

**Example:** A model is taught to answer questions about law, and it gets very good at it. Then it is trained only on cooking recipes. After that it handles recipes well, but it has become bad at law, even though nobody told it to forget.

## 🔵 2. How it works
1. A [[Neural Network]] keeps everything it knows in one shared pile of numbers called weights. The same weights serve every task.
2. Training ([[Gradient Descent]]) changes the weights to make the model better at the data it sees **right now**. It has no idea which weights were important for older lessons, because older data is not in front of it.
3. If the new task wants the weights to move somewhere different, they move. The old skill, which depended on them, breaks.
4. Often the drop is sharp and big, which is why it is called "catastrophic" and not just "forgetting".
5. This is a main worry in [[Fine-tuning]]: train a general model hard on a narrow task and it may get worse at the rest ([[Pre-training]] knowledge is overwritten).
6. Ways to reduce it: use a small [[Learning Rate]] and few steps; train only a small add-on instead of all weights ([[LoRA]]); keep some old examples and mix them in ("replay"); or tell the model which weights to protect.
7. Each fix has a price: the model learns the new task less well. The code in level 3 shows this trade in a tiny model.

## 🟠 3. Technical detail
- **Cause in one line:** one set of weights, trained on one data stream at a time, with a loss that only mentions the current data ([[Cross-Entropy Loss]]).
- **Replay.** Keep a small sample of old data and mix it into the new training batches. Simple and effective, but needs the old data (often unavailable or too large) and more training time.
- **Weight protection.** Add a penalty for moving the weights that mattered for the old task. Kirkpatrick et al. (2016/2017) do this and show it on sequential MNIST digit tasks and on several Atari games learned one after another (abstract read; the equation and the use of the diagonal Fisher information were read in the PDF text, the experiments not in detail).
- **Small updates.** Biderman et al. (2024, abstract read) compared [[LoRA]] with full fine-tuning on code and maths. They report that LoRA performs substantially worse on the new target domain but **forgets less** of the base model's skills, and forgets less than common regularisers such as weight decay and dropout ([[Regularization]]). They also say full fine-tuning learns changes of rank 10 to 100 times greater than typical LoRA settings. So "forgets less" here comes together with "learns less".
- **Not the same as overfitting.** [[Overfitting]] is poor results on *new* data of the *same* task. Forgetting is poor results on an *older* task.
- **Related areas.** Making models learn many tasks over time is called [[Continual Learning]]. Reusing a trained model for a new task is [[Transfer Learning]]; forgetting is its main risk.
- **The code below is a toy.** A model with only 2 weights is trained on task A, then on task B (a different rule on the same inputs). The two rules cannot both be fully satisfied by 2 weights, so forgetting is built in. It then tries replay and a "pull towards the old weights" penalty.

```python
import numpy as np

rng = np.random.RandomState(0)


def make(rule, n=2000):
    X = rng.randn(n, 2)
    return X, (rule(X) > 0).astype(float)


A = make(lambda X: X[:, 0] + X[:, 1])           # task A: label is 1 when x1 + x2 > 0
B = make(lambda X: X[:, 0] - X[:, 1])           # task B: label is 1 when x1 - x2 > 0 (a different rule, same weights)
tA, tB = make(lambda X: X[:, 0] + X[:, 1]), make(lambda X: X[:, 0] - X[:, 1])   # fresh test sets


def acc(w, task):
    X, y = task
    return float((((X @ w) > 0) == (y > 0)).mean())


def train(w, X, y, steps=500, lr=0.5, anchor=None, lam=0.0):
    w = w.copy()
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) / len(y)              # cross-entropy gradient
        if anchor is not None:
            g += lam * (w - anchor)             # pull toward the weights that were good for task A
        w -= lr * g
    return w


w0 = np.zeros(2)
wA = train(w0, *A)
print(f"after A:           A {acc(wA, tA):.2f}  B {acc(wA, tB):.2f}")
wB = train(wA, *B)
print(f"then B alone:      A {acc(wB, tA):.2f}  B {acc(wB, tB):.2f}   <- forgetting")

for k in (200, 1000, 2000):                     # mix k old task-A examples into the task-B data
    keep = rng.choice(len(A[1]), k, replace=False)
    wR = train(wA, np.vstack([B[0], A[0][keep]]), np.concatenate([B[1], A[1][keep]]))
    print(f"B + {k:4d} old A:    A {acc(wR, tA):.2f}  B {acc(wR, tB):.2f}")

for lam in (0.05, 0.2, 1.0):
    wP = train(wA, *B, anchor=wA, lam=lam)
    print(f"B + anchor lam={lam:<4}: A {acc(wP, tA):.2f}  B {acc(wP, tB):.2f}   weights {wP.round(2)}")
print("weights: after A", wA.round(2), "after B", wB.round(2))
```

What it printed: after task A, accuracy is A 1.00 and B 0.50 (a coin flip). After training on B alone, it is A 0.50 and B 1.00, so task A was completely lost. With 200, 1000 and 2000 old examples mixed in, accuracy on (A, B) was (0.53, 0.97), (0.66, 0.85) and (0.76, 0.75). With the pull-towards-old-weights penalty of 0.05, 0.2 and 1.0 it was (0.75, 0.76), (0.94, 0.58) and (0.99, 0.52). More protection kept more of A and learned less of B.

## 🔴 4. Deep / formal
A well-known way to protect old knowledge (Elastic Weight Consolidation, Kirkpatrick et al. 2016) adds a penalty to the new task's loss:

$$
L(\theta)=L_B(\theta)+\sum_i\frac{\lambda}{2}\,F_i\,\bigl(\theta_i-\theta^{*}_{A,i}\bigr)^2
$$

**Where:**
- $L_B(\theta)$ is the loss on the new task B.
- $\theta_i$ is weight $i$ now, and $\theta^{*}_{A,i}$ its value after learning task A.
- $F_i$ is how sensitive the model's predictions are to weight $i$, taken from the diagonal of the [[Fisher Information]] matrix. The paper calls it a measure of how important the weight was for task A. See [[Fisher Information]] for a toy where the weight that does all the work has the smaller $F_i$.
- $\lambda$ sets how strongly to protect the old task.
- **Matters:** a weight with large $F_i$ is held in place and one with small $F_i$ is free to move, so the network can learn B using the weights A did not need. Without the penalty ($\lambda=0$) every weight moves to suit B alone. With every $F_i$ equal, this becomes a plain pull towards the old weights, which is what the toy code does.

**Example (tiny numbers):** $\lambda=1$, a weight that was $\theta^{*}=1$ after task A and has moved to $\theta=1.5$. The distance is $0.5$ and its square is $0.25$.
- Important weight, $F=4$: penalty $\tfrac12\times1\times4\times0.25=\mathbf{0.5}$.
- Unimportant weight, $F=0.1$: penalty $\tfrac12\times1\times0.1\times0.25=\mathbf{0.0125}$.
- Meaning: the same move costs 40 times more for the important weight, so training prefers to move the unimportant one.

**Further notes (from the code above):**
- **Forgetting was total in the toy:** accuracy on A went from 1.00 to 0.50 after training on B alone. This was built in, because 2 weights cannot satisfy both rules (the best shared setting gave about 0.75 on each in the replay run with 2000 old examples, and the penalty run at 0.05).
- **Replay trade:** more old data kept more of A and lost some of B. With 200 old examples (10% of A) almost nothing was kept (A 0.53). It took as many old examples as new ones to reach (0.76, 0.75).
- **Penalty trade:** a larger $\lambda$ kept A (0.99 at $\lambda=1$) but B fell to 0.52, which is nearly no learning at all.
- **What the toy cannot show:** anything about real language models, how much a real fine-tune forgets, or whether weights have unequal importance (here both weights matter equally, so the importance idea is only illustrated in the example above).
- **Not covered:** forgetting measured on a real LLM. The only LLM result here is the Biderman abstract, and I did not read their tables.

**What came before, and how this improved on it:**
- **Before:** the plain recipe, train on the new data only. The problem was named in the late 1980s (McCloskey and Cohen, 1989; **from memory**, not checked).
- **Its limit:** with one shared set of weights, the old skill is lost, as the code shows (A from 1.00 to 0.50). Kirkpatrick et al. write that this had often been viewed as unavoidable (abstract read).
- **What changed:** Kirkpatrick et al. (arXiv 1612.00796, submitted 2 Dec 2016) protect the weights most important to old tasks, and show networks keeping skills on sequential MNIST tasks and on several Atari games. For large models, small add-ons such as [[LoRA]] limit the change (Biderman et al., arXiv 2405.09673, 2024).
- **How much better:** in this note's toy, a pull towards the old weights with $\lambda=0.2$ kept task A at 0.94 against 0.50 with no protection, while task B fell from 1.00 to 0.58. In the published work I read only abstracts, which give no numbers for forgetting, except that LoRA's training changes have a rank 10 to 100 times smaller than full fine-tuning.

## Related
- [[Fine-tuning]] · [[LoRA]] · [[QLoRA]] · [[Pre-training]] · [[Transfer Learning]] · [[Continual Learning]] · [[Neural Network]] · [[Gradient Descent]] · [[Learning Rate]] · [[Regularization]] · [[Overfitting]] · [[Cross-Entropy Loss]] · [[Fisher Information]] · [[Instruction Tuning]] · [[LLM]] · [[LoRA Fine-tuning on CPU]]

## Sources
- Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks*, arXiv 1612.00796 (submitted 2 Dec 2016). Abstract read, plus the PDF text around the penalty equation (their equation 3, which matches the one in level 4) and the diagonal Fisher information. Experiments not read in detail.
- Biderman et al., *LoRA Learns Less and Forgets Less*, arXiv 2405.09673 (submitted 15 May 2024; TMLR). **Abstract only.**
- McCloskey and Cohen (1989) is **from memory**.
- Image: none. A Wikimedia Commons search (2 queries) found only unrelated files (disaster reports, scanned books), so none is used.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/training Catastrophic forgetting:::A network trained on task B writes over the weights that held task A, like erasing French words to write Spanish on the same whiteboard.

#flashcards/ml/formula #flashcards/topic/training Write the elastic-weight-consolidation loss and explain each symbol.
?
$$L(\theta)=L_B(\theta)+\sum_i\frac{\lambda}{2}\,F_i\,\bigl(\theta_i-\theta^{*}_{A,i}\bigr)^2$$
- $L_B(\theta)$: the loss on the new task B.
- $\theta_i$: weight $i$ now. $\theta^*_{A,i}$: its value after learning task A.
- $F_i$: how sensitive the model's predictions are to weight $i$, from the diagonal of the Fisher information matrix. The paper calls it a measure of how important the weight was for task A.
- $\lambda$: how strongly to protect the old task.
- A weight with large $F_i$ is held in place and one with small $F_i$ is free to move, so the network can learn B using the weights that A did not need.

#flashcards/ml/example #flashcards/topic/training $\lambda=1$, a weight moved from $\theta^*=1$ to $\theta=1.5$. What is the penalty for $F=4$ and for $F=0.1$?::Distance 0.5, square 0.25. Important weight: $\tfrac12\cdot1\cdot4\cdot0.25=0.5$. Unimportant: $\tfrac12\cdot1\cdot0.1\cdot0.25=0.0125$. The same move costs 40 times more for the important weight, so training prefers to move the unimportant one.

#flashcards/ml/concept #flashcards/topic/training Why does catastrophic forgetting happen?::New training changes the same weights that held the old task, with nothing protecting them.
