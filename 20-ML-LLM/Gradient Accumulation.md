---
tags: [training, optimization, memory, PyTorch, batch-size]
status: growing
level: 4
tier: 2
---

# Gradient Accumulation

Training with a big batch on a small GPU by splitting the batch into small pieces, adding up their gradients, and only then taking one step.

## 🟢 1. Simple (age 12)
Say a teacher wants the opinion of 32 students, but the room only fits 8. She asks four groups of 8 in turn, writes each group's answer on the board, and only after all four does she decide. The decision is the same as if all 32 had fitted in the room.

**Example:** You want to carry 32 books home but can only hold 8. You make four trips to a pile by the door, and only then do you load them all into the car in one go.

## 🔵 2. How it works
1. Training a [[Neural Network]] by [[Gradient Descent]] means: show the model a batch of examples, measure the error with a [[Loss Function]], find the direction to improve (the *gradient*, through [[Backpropagation]]), and take one small step.
2. A bigger batch usually gives a smoother, more trustworthy direction, but it needs more memory, because the model keeps notes for every example in the batch while it works out the gradient.
3. Instead of one big batch, use a few small **micro-batches**. For each one, run the model forward and backward, but **do not take the step yet**. The gradients are added to a running total.
4. Divide each micro-batch's error by the number of micro-batches, so that the total is an average and not a sum.
5. After the last micro-batch, take **one** step with the total, and clear the running total.
6. The result is the same direction as one big batch, but only one small piece of the batch is in memory at a time.
7. The price: it takes more forward and backward passes one after the other, so a step is slower in time than with a really big batch on a big machine.

## 🟠 3. Technical detail
- In [[PyTorch]], `loss.backward()` **adds** to each parameter's `.grad` instead of replacing it. That is why accumulation is natural: call `backward()` several times, then call `optimizer.step()` and `optimizer.zero_grad()` once.
- The loop in training code:

```text
for i, (x, y) in enumerate(micro_batches):
    loss = loss_fn(model(x), y) / K          # K = micro-batches per step
    loss.backward()                          # adds to .grad
    if (i + 1) % K == 0:
        optimizer.step(); optimizer.zero_grad()
```
- **Divide by K.** Without it, the gradient is K times too big, which acts like a [[Learning Rate]] that is K times too large. The code below shows exactly 4 times.
- **Memory** is set by the micro-batch size, not the effective batch size. The model's weights, its gradients and its optimizer state do not shrink (see [[Adam Optimizer]]); only the per-example notes do. I did not measure memory here.
- **Combines with** [[Mixed Precision Training]], [[LoRA]] and [[QLoRA]] when fine-tuning a big model on one consumer card ([[Fine-tuning]]).
- **When it is not exactly the same:** layers that look at the whole batch, such as batch normalisation (see [[Normalization]]), see only the micro-batch, so the results differ a little. Random choices such as [[Dropout]] also differ. I only tested a model without these.

```python
import torch

torch.manual_seed(0)
x, y = torch.randn(8, 3), torch.randn(8, 1)
w0 = torch.randn(3, 1)

def grad(xb, yb, scale=1.0):
    w = w0.clone().requires_grad_()
    loss = ((xb @ w - yb) ** 2).mean() * scale     # mean squared error on this batch
    loss.backward()
    return w.grad

big = grad(x, y)                                              # one batch of 8
acc = sum(grad(x[i:i + 2], y[i:i + 2], scale=1 / 4) for i in range(0, 8, 2))   # 4 micro-batches of 2, each loss divided by 4
bad = sum(grad(x[i:i + 2], y[i:i + 2]) for i in range(0, 8, 2))                 # forgot to divide
print("batch of 8:        ", big.flatten().tolist())
print("4 x 2 accumulated: ", acc.flatten().tolist())
print("largest difference:", (big - acc).abs().max().item())
print("without dividing, gradient is", round((bad / big).mean().item(), 3), "times too big")
```

## 🔴 4. Deep / formal
For a loss that is an average over the examples, the gradient of a batch is the average of the gradients of its equal-sized micro-batches:

$$
\nabla L(\theta)=\frac{1}{K}\sum_{k=1}^{K}\nabla L_k(\theta)
$$

**Where:**
- $\theta$ is the model's numbers (weights) and $L$ the loss on the whole batch.
- $L_k$ is the loss on micro-batch $k$, and $K$ is the number of micro-batches, each of the same size.
- $\nabla$ means "the direction in which the loss rises fastest", the gradient.
- **Matters:** the $1/K$ makes the sum an average. Without it, you take a step $K$ times too long. If the micro-batches have different sizes, weight each by its size, or the result is biased toward the small ones.

**Example (tiny numbers):** four examples with gradients 2, 4, 6 and 8, split into $K=2$ micro-batches of two.
- Micro-batch gradients (averages): $(2+4)/2=3$ and $(6+8)/2=7$.
- Accumulate with the $1/K$: $(3+7)/2=\mathbf{5}$. One big batch: $(2+4+6+8)/4=\mathbf{5}$.
- Without dividing: $3+7=\mathbf{10}$, twice too large.
- Meaning: the small pieces give exactly the big-batch step.

**Further notes (from the code: a one-layer linear model with a squared-error loss and 8 random examples):**
- One batch of 8 and four micro-batches of 2 gave gradients that differ by at most $6\times10^{-8}$, which is rounding noise in 32-bit numbers.
- Forgetting to divide made the gradient exactly 4.0 times too large.
- This is a tiny model with no batch normalisation and no dropout. I did not test a real network, a real memory saving, or the speed.

**What came before, and how this improved on it:**
- **Before:** the batch size was whatever fitted in the GPU's memory.
- **Its limit:** a small batch gives a noisy gradient, and some methods and settings are tuned for large batches, so a small GPU could not copy them.
- **What changed:** add up the gradients of several micro-batches before stepping. I did not trace who first described this, so there is no date or paper here.
- **How much better:** in this note's code the accumulated gradient matched the big-batch gradient to $6\times10^{-8}$, while using micro-batches 4 times smaller. In memory terms, per-example notes shrink by about that factor (reasoning, **not measured here**).

## Related
- [[Gradient Descent]] · [[Stochastic Gradient Descent]] · [[Learning Rate]] · [[Backpropagation]] · [[Loss Function]] · [[PyTorch]] · [[Mixed Precision Training]] · [[Fine-tuning]] · [[LoRA]] · [[QLoRA]] · [[Data Parallelism]] · [[Model Parallelism]]

## Sources
- The gradient identity is the linearity of the gradient for an average loss; the code checks it numerically. Behaviour of `backward()` adding to `.grad` is from memory of the PyTorch documentation, and my code relies on it, **not re-read from the docs**.
- Versions run: PyTorch 2.8.0 (CPU).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/training Gradient accumulation:::Splitting a big batch into small micro-batches that fit in memory, adding up their gradients, and only then taking one step. The result is the same as if the whole batch had fitted.

#flashcards/ml/formula #flashcards/topic/training Write the accumulated gradient and explain each symbol.
?
$$\nabla L(\theta)=\frac{1}{K}\sum_{k=1}^{K}\nabla L_k(\theta)$$
- $\theta$: the model's numbers (weights). $L$: the loss on the whole batch.
- $L_k$: the loss on micro-batch $k$. $K$: the number of micro-batches, each of the same size.
- $\nabla$: the gradient, the direction in which the loss rises fastest.
- The $1/K$ turns the sum into an average. Without it the gradient would be $K$ times too large.

#flashcards/ml/cloze #flashcards/topic/training Gradient accumulation divides the sum by ==K== so that it equals the gradient of one ==big batch==.

#flashcards/ml/example #flashcards/topic/training Four examples with gradients 2, 4, 6, 8 are split into $K=2$ micro-batches of two. What is the accumulated gradient, and what if you forget to divide?::Micro-batch averages: (2+4)/2 = 3 and (6+8)/2 = 7. Accumulated with 1/K: (3+7)/2 = 5. One big batch: (2+4+6+8)/4 = 5. Without dividing: 3 + 7 = 10, twice too large.
