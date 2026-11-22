---
tags: [LLM, fine-tuning, LoRA, PEFT, low-rank, adapters, training]
status: growing
level: 4
tier: 3
---

# LoRA

**Lo**w-**R**ank **A**daptation. A way to teach a big model a new job by training only a tiny add-on, and leaving the big model untouched.

## 🟢 1. Simple (age 12)
A big pretrained AI is like a huge printed textbook. Changing every page to teach it a new skill is slow and expensive. LoRA leaves the textbook alone and **clips a small sheet of notes onto each important page**. Only the notes are written or changed. To switch to a different job, you swap the notes and keep the same book.

**Example:** A restaurant has one big recipe book. For a vegetarian night you don't print a new book; you slip in a few cards that say "swap the meat for beans". For a spicy night you slip in different cards. One book, many sets of cards.

## 🔵 2. How it works
1. A [[Transformer]] is mostly big grids of numbers called weight matrices ([[Matrix]]). Full [[Fine-tuning]] changes **all** of them, which needs a lot of GPU memory ([[GPU]]) because each number also needs its gradient and optimizer memory ([[Adam Optimizer]]).
2. LoRA **freezes** the original matrix $W_0$. It does not change.
3. Next to $W_0$ it adds a **small correction** made of two thin matrices, $B$ and $A$. Multiplied together, $BA$ has the same shape as $W_0$, but it is built from far fewer numbers.
4. The model's answer is the frozen part plus the correction: $W_0x+BAx$.
5. Only $A$ and $B$ are trained. $B$ starts at zero, so at the start the model behaves exactly like the original.
6. Because the correction is just a matrix, after training you can **add it into $W_0$** so the finished model runs with no extra delay, or keep it separate and **swap** corrections for different tasks.
7. The lecture adds: a variant called **QLoRA** also stores the frozen weights in 4-bit numbers ([[Quantization]]) to save even more memory.

![[lora-diagram.svg]]
*LoRA: the big pretrained weights W stay frozen (left, "Pesos preadestrados" is Spanish/Galician for "pretrained weights"). A tiny pair of matrices B and A (right) is trained, and its output is added on top. Image: LauAI, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:LoRA.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **The shape trick:** $W_0$ is $d\times d$ (or $d_{out}\times d_{in}$). $B$ is $d\times r$ and $A$ is $r\times d$, with the **rank** $r$ much smaller than $d$ (typically 4 to 64).
- **Initialisation:** $B=0$ and $A$ small random, so $BA=0$ at the start.
- **Scaling:** the correction is multiplied by $\alpha/r$, where $\alpha$ is a fixed constant you choose. That keeps the size of the update steady when you change $r$.
- **Where to put it:** the original paper put LoRA on the attention weight matrices. The lecture says recent guidance (from "LoRA Without Regret", a 2025 blog post by Schulman and colleagues, which I did not read) is to apply it to **all** layers, with the feed-forward (MLP) layers being the most important place. That is from the lecture slides, not checked.
- **Training quirks from the same slides:** LoRA wants a higher learning rate than full fine-tuning, and does worse than full fine-tuning at large batch sizes (from the slides; I did not verify).
- **After training:** keep $A,B$ as a small file (a few megabytes), or merge into $W_0$. The code checks that the merged and separate versions give the same output.
- **Swapping tasks:** one frozen model plus a set of small $(A,B)$ pairs for spam detection, sentiment, translation and so on. The code shows two adapters on one frozen matrix, each good for its own target.
- **Other methods from the lecture:** prefix tuning and adapters (small extra layers). Adapters add some delay at inference; LoRA, once merged, does not (the LoRA abstract says so).
- **Limit:** it can only represent a *low-rank* change. If the best change is not nearly low-rank, LoRA leaves some error (the code measures this).

```python
import numpy as np
import torch
torch.manual_seed(0); torch.set_num_threads(1)
d, r = 64, 4
W0 = torch.randn(d, d, dtype=torch.float64)                # frozen pretrained weights
x = torch.eye(d, dtype=torch.float64)                      # probe with every unit input, so the loss is the plain gap between the weight matrices

def train(target_delta, rank, steps=3000):
    """learn B @ A (d x rank times rank x d) so that W0 + B A acts like W0 + target_delta on the inputs"""
    A = (torch.randn(rank, d, dtype=torch.float64) * 0.1).requires_grad_()
    B = torch.zeros(d, rank, dtype=torch.float64, requires_grad=True)       # B starts at zero: the model starts unchanged
    opt = torch.optim.Adam([A, B], lr=0.02)
    want = x @ (W0 + target_delta).T
    for _ in range(steps):
        opt.zero_grad(); loss = ((x @ (W0 + B @ A).T - want) ** 2).mean(); loss.backward(); opt.step()
    return A.detach(), B.detach(), loss.item()

A0 = torch.randn(r, d, dtype=torch.float64); B0 = torch.zeros(d, r, dtype=torch.float64)
print('start: B = 0 so the output is unchanged:', torch.allclose(x @ (W0 + B0 @ A0).T, x @ W0.T))
low = torch.randn(d, r, dtype=torch.float64) @ torch.randn(r, d, dtype=torch.float64) * 0.3      # a change of rank 4
full = torch.randn(d, d, dtype=torch.float64) * 0.3                                              # a change of full rank
A, B, l = train(low, r)
print('rank-4 change, rank-4 LoRA : final loss', f'{l:.1e}')
A, B, l = train(full, r)
print('full-rank change, rank-4 LoRA: final loss', f'{l:.4f}')
# the best any rank-4 add-on can do is keep the 4 biggest singular values (Eckart-Young); the loss left is the discarded part
U, S, Vt = torch.linalg.svd(full)
rest = (U[:, r:] * S[r:]) @ Vt[r:]
print('discarded singular values only (the floor for rank 4):', f'{float((rest ** 2).mean()):.4f}')
# merging: W0 + B A used as one matrix gives the same output as the two paths
A, B, l = train(low, r, steps=500)
print('merged weights equal separate paths:', torch.allclose(x @ (W0 + B @ A).T, x @ W0.T + (x @ A.T) @ B.T))
# counts
for dd, rr in ((64, 4), (4096, 8), (4096, 16)):
    print('d =', dd, 'rank', rr, '| full', dd * dd, '| LoRA', 2 * dd * rr, '| ratio', dd * dd // (2 * dd * rr))
# swapping tasks: two different adapters on one frozen W0
low2 = torch.randn(d, r, dtype=torch.float64) @ torch.randn(r, d, dtype=torch.float64) * 0.3
A1, B1, _ = train(low, r, steps=1500); A2, B2, _ = train(low2, r, steps=1500)
t1, t2 = x @ (W0 + low).T, x @ (W0 + low2).T
e = lambda y, t: float(((y - t) ** 2).mean())
print('adapter 1 on task 1:', f'{e(x @ (W0 + B1 @ A1).T, t1):.1e}', '| adapter 2 on task 1:', f'{e(x @ (W0 + B2 @ A2).T, t1):.2f}', '| adapter 2 on task 2:', f'{e(x @ (W0 + B2 @ A2).T, t2):.1e}')
```

## 🔴 4. Deep / formal
**The LoRA layer.** For an input $x$:

$$
h=W_0x+\frac{\alpha}{r}\,B\,A\,x,\qquad W_0\in\mathbb{R}^{d\times d},\ B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times d}
$$

**Where:**
- $W_0$ is the frozen pretrained weight matrix and $x$ is the input vector (see [[Matrix]] and [[Vector]]).
- $B$ and $A$ are the two small trained matrices, and $r$ is their shared inner size, the **rank**.
- $\alpha$ is a fixed constant, so $\alpha/r$ is the scale applied to the correction.
- $h$ is the layer's output.
- **Matters:** $BA$ can be any matrix of rank at most $r$. A small $r$ means very few trained numbers, but the correction can only move the output within an $r$-dimensional family of directions. With $B=0$ at the start, $h=W_0x$: training starts from the pretrained model, not from scratch.

**Example (tiny numbers):** $d=2$, $r=1$, $\alpha=1$, with $W_0=\begin{pmatrix}1&0\\0&1\end{pmatrix}$, $B=\begin{pmatrix}1\\2\end{pmatrix}$, $A=\begin{pmatrix}1&1\end{pmatrix}$ and $x=\begin{pmatrix}1\\2\end{pmatrix}$.
- Frozen part: $W_0x=(1,2)$.
- $Ax=1\cdot1+1\cdot2=3$, then $B(3)=(3,6)$.
- Scale $\alpha/r=1$, so $h=(1,2)+(3,6)=\mathbf{(4,8)}$.
- Meaning: the add-on shifted the answer from $(1,2)$ to $(4,8)$ using $4$ trained numbers (the entries of $B$ and $A$). At this size that is no saving, because $W_0$ also has $4$ entries; the saving only shows when $d$ is big (next formula).

**How many numbers are trained.** For one matrix:

$$
\underbrace{d^{2}}_{\text{full}}\quad\text{against}\quad\underbrace{2dr}_{\text{LoRA}},\qquad \text{ratio}=\frac{d}{2r}
$$

**Where:**
- $d^2$ is the number of entries in $W_0$ (training it fully means training all of them).
- $2dr$ is the number of entries in $B$ ($d\times r$) plus $A$ ($r\times d$).
- The ratio is how many times fewer numbers LoRA trains.
- **Matters:** the saving grows with the size of the model. For a small matrix it is minor; for the big matrices of an LLM it is huge. It also shrinks as $r$ grows.

**Example (tiny numbers):** a matrix with $d=4096$ and $r=8$.
- Full: $4096^2=\mathbf{16{,}777{,}216}$ numbers.
- LoRA: $2\cdot4096\cdot8=\mathbf{65{,}536}$ numbers.
- Ratio: $4096/16=\mathbf{256}$ times fewer. With $r=16$ it is $128$ times fewer (the code prints both).
- Meaning: one matrix needs 256 times fewer trained numbers. The model's total saving depends on which matrices get LoRA.

**Further notes:**
- **The best low-rank fit:** if you want a rank-$r$ matrix as close as possible to a target matrix, the best you can do is keep its $r$ largest singular values ([[Singular Value Decomposition]]; the Eckart-Young theorem, from memory). The code checks this: for a target change of full rank, a rank-4 LoRA trained to the end reached a mean squared gap of $0.0722$, and the singular values it cannot keep add up to exactly $0.0722$. For a change that is itself rank 4, it reached $9\times10^{-11}$.
- **Why low rank can be enough:** the paper reports an empirical look at how low-rank the updates of language-model adaptation are. On GPT-3 175B with the query and value matrices adapted, WikiSQL accuracy was 73.4 at rank 1 and 73.8 at rank 8 (paper, Table 6, read in [[LoRA 2021]]), and the top learned direction was shared between rank 8 and rank 64. It is an observation about real models, not a theorem, and my toy does not test it. The paper also says a small rank may not suffice for every task.
- **Not tested here:** real language models, the choice of $r$ and $\alpha$, QLoRA's 4-bit quantization, and the lecture's claims about learning rates and batch size.

**What came before, and how this improved on it:**
- **Before:** full fine-tuning, which trains every weight and keeps gradients and optimizer state for all of them, and adapter layers, small extra layers inserted into each block (Houlsby et al., 2019, as described in the LoRA paper; I did not read the adapter paper itself).
- **Its limit:** the LoRA abstract's example: deploying separate full fine-tuned copies of a 175-billion-parameter GPT-3 for each task is prohibitively expensive. Adapters, it says, add extra delay when the model runs; the paper measures +20.7% and +30.3% on GPT-2 medium at batch size 1 and sequence length 128 (paper, Table 1).
- **What changed:** freeze the weights and train a low-rank correction $BA$ that can be merged back (Hu et al., arXiv 2106.09685, first posted 17 June 2021; abstract and most of the body read, see [[LoRA 2021]]).
- **How much better:** the abstract reports 10,000 times fewer trainable parameters and 3 times less GPU memory than fine-tuning GPT-3 175B with Adam, with quality on par or better on the models tested. In this note's code, one 4096-wide matrix trains $65{,}536$ numbers at $r=8$ against $16{,}777{,}216$ (256 times fewer), the merged and unmerged outputs are identical, and two adapters served two targets on one frozen matrix (error about $10^{-10}$ on its own target and $0.74$ on the other's). The toy is a matrix-fitting task, not language modelling.

## Related
- [[Fine-tuning]] · [[Matrix]] · [[Singular Value Decomposition]] · [[Quantization]] · [[Transformer]] · [[LLM]] · [[Pre-training]] · [[Transfer Learning]] · [[Adam Optimizer]] · [[GPU]] · [[Knowledge Distillation]] · [[Vector]] · [[Data Parallelism]] · [[QLoRA]] · [[Gradient Accumulation]] · [[Feed-Forward Network]] · [[Catastrophic Forgetting]] · [[Continual Learning]] · [[Unsloth]] · [[LoRA Fine-tuning on CPU]] · [[LoRA 2021]]

## Sources
- Stanford CME 295 (Fall 2025), Lecture 4 slides, Afshine and Shervine Amidi
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021 (arXiv 2106.09685; abstract and most of the body read, see [[LoRA 2021]])
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, 2023 (named in the lecture; not read)
- Schulman et al., *LoRA Without Regret*, 2025 (named in the lecture; not read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm LoRA:::Leave the big pretrained model alone and clip a small sheet of notes (two thin matrices) onto each important layer. Only the notes are trained. To switch jobs, swap the notes.

#flashcards/ml/formula #flashcards/topic/llm Write the LoRA layer and explain each symbol.
?
$$h=W_0x+\frac{\alpha}{r}\,B\,A\,x,\qquad W_0\in\mathbb{R}^{d\times d},\ B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times d}$$
- $W_0$: the frozen pretrained weight matrix. $x$: the input vector.
- $B$, $A$: the two small trained matrices. $r$: their shared inner size, the rank.
- $\alpha$: a fixed constant, so $\alpha/r$ is the scale applied to the correction.
- $h$: the output, the frozen layer's answer plus the small learned correction.

#flashcards/ml/formula #flashcards/topic/llm Write the LoRA parameter saving and explain each symbol.
?
$$\underbrace{d^{2}}_{\text{full}}\quad\text{against}\quad\underbrace{2dr}_{\text{LoRA}},\qquad \text{ratio}=\frac{d}{2r}$$
- $d^2$: the number of entries in $W_0$. Training it fully means training all of them.
- $2dr$: the entries in $B$ ($d\times r$) plus $A$ ($r\times d$).
- The ratio: how many times fewer numbers LoRA trains. The saving grows with the size of the model.

#flashcards/ml/cloze #flashcards/topic/llm LoRA trains ==2dr== numbers instead of ==d²==, which is ==d/(2r)== times fewer.

#flashcards/ml/example #flashcards/topic/llm $d=2$, $r=1$, $\alpha=1$, $W_0=I$, $B=(1,2)^\top$, $A=(1,1)$, $x=(1,2)$. What is $h$?::Frozen part W₀x = (1, 2). Ax = 3, B(3) = (3, 6). Scale 1. h = (1, 2) + (3, 6) = (4, 8).

#flashcards/ml/example #flashcards/topic/llm For $d=4096$ and $r=8$, how many numbers does full training need, how many does LoRA need, and what is the ratio?::Full: 4096² = 16,777,216. LoRA: 2 × 4096 × 8 = 65,536. Ratio 4096/16 = 256 times fewer (128 times for r = 16).

#flashcards/ml/image #flashcards/topic/llm ![[lora-diagram.svg]]
What does this picture show?
?
LoRA: the big pretrained weights W stay frozen (left). A tiny pair of matrices B and A (right) is trained, and its output is added on top.
