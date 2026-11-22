---
tags: [training, memory, GPU, PyTorch, backpropagation, LLM, fine-tuning]
status: growing
level: 4
tier: 2
---

# Gradient Checkpointing

**Gradient checkpointing** (also called activation checkpointing) saves GPU memory during training by keeping only a few of the middle results of the forward pass and recomputing the rest when [[Backpropagation]] needs them. It trades extra computing for less memory.

## 🟢 1. Simple (age 12)
Imagine copying a very long recipe while you cook, and you must be able to undo every step later. You could write down the state of the kitchen after every step, which fills many pages. Or you could write it down only after every tenth step, and when you need to undo something, quickly redo the few steps in between. You use far less paper, and you spend a little more time.

**Example:** A teacher marks a 100-page homework and must remember what each page looked like at each stage. Instead of keeping 100 photocopies, she keeps 10 and re-reads the pages in between when she needs them.

## 🔵 2. How it works
1. **Training has two passes.** The forward pass computes the answer layer by layer. The backward pass walks back through the layers to find how to change each weight ([[Backpropagation]], [[Neural Network]]).
2. **The backward pass needs the middle results.** For each layer it needs that layer's input or output from the forward pass. These stored values are called **activations**.
3. **Normally everything is kept.** The memory for activations grows with the number of layers and with the batch size, and it is often the part that does not fit on a [[GPU]].
4. **Checkpointing keeps only a few.** Cut the layers into segments. Keep the value at the start of each segment (the checkpoint) and throw away the values inside.
5. **During the backward pass, redo the missing part.** When the backward pass reaches a segment, rerun that segment's forward pass from its checkpoint to rebuild the values, use them, then throw them away again.
6. **The answer does not change.** The same gradients come out, because the same maths is redone. Only memory and time change.
7. **The catch:** it costs extra time (about one more forward pass) and it only reduces the memory for activations, not for the weights or the optimizer state.

No picture: I searched Wikimedia Commons and found no free, on-topic image for this note, so I added none.

## 🟠 3. Technical detail
- **What is thrown away and what is kept.** PyTorch's `torch.utils.checkpoint` says the function's inputs are kept, the tensors made inside it are not, and the function is rerun during backward (PyTorch docs).
- **`use_reentrant`.** The docs say to pass it explicitly and recommend `use_reentrant=False`. The reentrant version reruns the whole function in backward and runs the forward under `no_grad()`. The non-reentrant version records the graph, works with `torch.autograd.grad`, and can stop the recomputation early once all needed tensors are rebuilt (PyTorch docs).
- **`checkpoint_sequential`.** It splits a sequential model into a number of segments. Every segment except the last discards its inner activations, and each segment's input is saved so it can be rerun (PyTorch docs). That is why the last segment is not rerun.
- **Randomness.** Rerunning a segment can advance the random number generator (for example for dropout). By default `preserve_rng_state=True` saves and restores it so the recomputation matches. Only one device type's state is saved. If the function behaves differently in backward than in forward (for example through a global variable) the result can be an error or silently wrong gradients (PyTorch docs).
- **In fine-tuning tools.** The [[Unsloth]] blog's example turns gradient checkpointing on together with [[LoRA]] adapters, so a large model fits in GPU memory ([[Fine-tuning]], [[QLoRA]]).
- **It does not shrink everything.** The weights, their gradients and the [[Adam Optimizer]] state stay as they were. [[Mixed Precision Training]], [[Quantization]] and [[Data Parallelism]] attack those parts.

The code below is my own. It builds 12 small layers, runs one training step with and without checkpointing, counts the forward calls and the activation tensors autograd keeps, and compares the gradients. I ran it with PyTorch 2.8.0 on the CPU.

```python
import math
import torch
from torch.utils.checkpoint import checkpoint

torch.manual_seed(0)
n, width, batch = 12, 64, 32          # 12 layers of Linear + Tanh


class Layer(torch.nn.Module):
    calls = 0                          # counts forward calls across all layers

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(width, width)

    def forward(self, x):
        Layer.calls += 1
        return torch.tanh(self.lin(x))


layers = torch.nn.ModuleList([Layer() for _ in range(n)])
x = torch.randn(batch, width)


def run(k):
    """k = 0: keep every activation. k >= 1: k segments, checkpoint all but the last."""
    for p in layers.parameters():
        p.grad = None
    Layer.calls = 0
    saved = []                         # activation-shaped tensors autograd keeps (weights not counted)

    def pack(t):
        if tuple(t.shape) == (batch, width):
            saved.append(t.numel())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        h = x
        if k == 0:
            for layer in layers:
                h = layer(h)
        else:
            size = n // k
            for s in range(k):
                seg = list(layers)[s * size:(s + 1) * size]
                f = lambda t, seg=seg: torch.nn.Sequential(*seg)(t)
                h = f(h) if s == k - 1 else checkpoint(f, h, use_reentrant=False)
        loss = h.pow(2).mean()
    fwd_calls = Layer.calls
    loss.backward()
    grads = torch.cat([p.grad.flatten() for p in layers.parameters()])
    return loss.item(), grads, fwd_calls, Layer.calls, len(saved)


base = run(0)
print("k=0  forward calls", base[2], "-> after backward", base[3], " activation tensors saved", base[4])
for k in (2, 3, 4, 6):
    out = run(k)
    same = torch.allclose(out[1], base[1], atol=1e-6)
    print(f"k={k}  forward calls {out[2]} -> after backward {out[3]} (2n-n/k = {2*n - n//k})"
          f"  activation tensors saved {out[4]}  same loss {abs(out[0]-base[0]) < 1e-7}  same grads {same}")

for N in (100, 1000):                  # the textbook counting model of equation 1 below
    best = min(range(1, N + 1), key=lambda k: k + N / k)
    print("n =", N, " best k =", best, " memory =", best + N / best, " 2*sqrt(n) =", round(2 * math.sqrt(N), 2))
print("n=100:", {k: k + 100 / k for k in (1, 4, 10, 50, 100)})
print("relative time if backward = 2 x forward:", (1 + 1 + 2) / (1 + 2))
```

It printed: for `k=0`, 12 forward calls and 25 activation tensors saved. For `k=2`, `3`, `4`, `6`: 18, 20, 21 and 22 forward calls after backward (exactly $2n-n/k$), 14, 11, 10 and 10 activation tensors saved, and the loss and all gradients matched the no-checkpoint run every time. For the counting model: best $k=10$ with memory 20 for $n=100$, and best $k=32$ with 63.25 for $n=1000$ (against $2\sqrt n=63.25$). For $n=100$ the memory is 101, 29, 20, 52 and 101 for $k=1, 4, 10, 50, 100$. The relative time is 1.333. The "activation tensors saved" count is what autograd holds after the forward pass. The real peak also includes the segment being recomputed in backward, so the count understates the memory.

## 🔴 4. Deep / formal
**Memory with $k$ segments (an idealised model).** Suppose $n$ layers each produce an activation of the same size, taken as one unit. Keep the $k$ values at the segment boundaries, and when a segment is recomputed in backward, hold all $n/k$ of its inner values:

$$
M(k)=k+\frac{n}{k},\qquad k^{*}=\sqrt n,\qquad M(k^{*})=2\sqrt n
$$

**Where:**
- $n$ is the number of layers (or operations in the chain). **Matters:** without checkpointing the memory is $n$ units, so it grows in step with depth.
- $k$ is the number of segments. **Matters:** more segments keep more boundary values but shorten the part to recompute. $k=1$ and $k=n$ both give about $n$, so the best is in between.
- $k+n/k$ is the stored boundaries plus the largest segment held at once. **Matters:** the two terms pull in opposite directions, and they are equal at $k=\sqrt n$.
- $2\sqrt n$ is the smallest value. **Matters:** the memory grows like $\sqrt n$ instead of $n$, which is the $O(\sqrt n)$ result of Chen et al. (abstract).

**Example (tiny numbers):** $n=100$ layers.
- No checkpointing: 100 units.
- $k=4$: $4+100/4=29$. $k=50$: $50+100/50=52$.
- $k=10=\sqrt{100}$: $10+10=20$.
- Meaning: five times less activation memory at the best split. The code searched every $k$ and found 10 for $n=100$ and 32 for $n=1000$ (63.25, equal to $2\sqrt{1000}$).

**What it costs in time.** Rerunning the forward pass of the checkpointed segments adds about one extra forward pass in total. Let $F$ be the forward time and $B$ the backward time:

$$
\frac{T_{\text{ckpt}}}{T_{\text{plain}}}=\frac{F+F+B}{F+B}=\frac{2F+B}{F+B}\ \xrightarrow{\,B=2F\,}\ \frac43
$$

**Where:**
- $F$ is the time of one forward pass. **Matters:** checkpointing adds one extra $F$.
- $B$ is the time of the backward pass. **Matters:** the larger $B$ is compared with $F$, the smaller the relative cost of the extra $F$.
- $B=2F$ is the usual rule of thumb that the backward pass costs about twice the forward pass (from memory; it is also the split behind the $6ND$ training-cost rule in [[GPU]], 2 for forward and 4 for backward). **Matters:** it is an assumption. Real ratios differ.

**Example (tiny numbers):** $F=1$ and $B=2$.
- A plain step costs $1+2=3$. A checkpointed step costs $1+1+2=4$.
- The ratio is $4/3=1.33$, about 33% more time.
- Meaning: you pay about a third more time to cut activation memory several-fold. The paper reports about 30% extra time in a real experiment (abstract), close to this estimate.

**How many forward calls.** With `checkpoint_sequential`-style segments, where every segment except the last is rerun:

$$
\text{forward calls}=n+\Big(n-\frac nk\Big)=2n-\frac nk
$$

**Where:**
- $n$ is the number of layers and $k$ the number of segments. **Matters:** the first $n$ is the normal forward pass.
- $n-n/k$ is the layers rerun in backward, all except the last segment's $n/k$. **Matters:** a bigger $k$ makes the last segment smaller, so more is rerun.

**Example (tiny numbers):** $n=12$.
- $k=2$: $24-6=18$. $k=4$: $24-3=21$. $k=6$: $24-2=22$.
- Meaning: the count never reaches $2n=24$, and it moves toward it as $k$ grows. The code counted exactly 18, 21 and 22.

**Honest limits and open questions**
- **The toy memory model is idealised.** It assumes every layer's activation is the same size and that nothing else is saved. Real layers differ (attention can save a large table), and some layers save more than one tensor. The code's count of saved tensors falls from 25 to 10 and then stops falling ($k=4$ and $k=6$ both 10), because the last segment is not checkpointed and its tensors are saved in full.
- **Not measured:** actual memory in bytes, actual run time, and any GPU. The code ran only on the CPU and counted tensors and calls.
- **Not covered:** choosing which layers to checkpoint (the paper's more aggressive $O(\log n)$ memory variant costs $O(n\log n)$ extra forward computation, per the abstract), checkpointing with compilers such as `torch.compile`, and what the reentrant version breaks in detail.
- **Randomness and side effects** can break exact equality if they are not handled (see the PyTorch caveats above).
- **The saving is only for activations.** For a big model, the weights and the optimizer state can be the larger part, and checkpointing does nothing for them.

**What came before, and how this improved on it:**
- **Before:** keep every layer's activations from the forward pass until the backward pass has used them ([[Backpropagation]]).
- **Its limit:** the memory grows in step with the number of layers and the batch size, so deep networks and long sequences did not fit on a GPU.
- **What changed:** keep only a few checkpoints and recompute the rest in the backward pass (Chen, Xu, Zhang and Guestrin, *Training Deep Nets with Sublinear Memory Cost*, arXiv 1604.06174, submitted 21 April 2016; read as a summary of the abstract, not the body; author order from the arXiv page).
- **How much better:** from the abstract: $O(\sqrt n)$ memory for an $n$-layer network at the cost of one extra forward pass per mini-batch, and a 1,000-layer deep residual network on ImageNet going from 48G to 7G of memory with about 30% more running time. In my own toy: 12 layers, identical gradients, and the counting model for $n=100$ going from 100 units to 20 at the best split, at a predicted $4/3$ the time. The paper's figures are not reproduced here.

**Check it yourself:** run the code in level 3. It prints every number in this block that I computed.

## Related
[[Backpropagation]] · [[Neural Network]] · [[Unsloth]] · [[Fine-tuning]] · [[LoRA]] · [[QLoRA]] · [[Mixed Precision Training]] · [[Quantization]] · [[Data Parallelism]] · [[Gradient Accumulation]] · [[Adam Optimizer]] · [[GPU]] · [[ResNet]] · [[Residual Connection]] · [[Recurrent Neural Network]] · [[Dropout]] · [[FlashAttention]]

## Sources
- Chen, Xu, Zhang and Guestrin, *Training Deep Nets with Sublinear Memory Cost*, arXiv 1604.06174 (21 April 2016), https://arxiv.org/abs/1604.06174. I read a summary of the abstract (the $O(\sqrt n)$ memory, the extra forward pass, the $O(\log n)$ variant, and the 48G to 7G, about 30% result). I did not read the body or reproduce the experiments.
- PyTorch documentation, *torch.utils.checkpoint* (https://docs.pytorch.org/docs/2.15/checkpoint.html), read on 16 November 2026: what is kept and recomputed, `use_reentrant`, `checkpoint_sequential` and the randomness caveats. The code in level 3 was run with PyTorch 2.8.0+cpu, an older version than that page.
- The Unsloth blog (Hugging Face, 10 January 2024) for the use with LoRA (see [[Unsloth]]).
- The $B=2F$ rule is from memory, as the note says. The note was not opened in Obsidian.
- No image: no free, on-topic picture for this note was found on Wikimedia Commons.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/training Gradient checkpointing (activation checkpointing):::Saves GPU memory during training by keeping only a few of the forward pass's middle results (activations) and recomputing the rest when backpropagation needs them. It trades extra compute for less memory.

#flashcards/ml/concept #flashcards/topic/training What are activations and why does the backward pass need them?::The stored middle values of the forward pass (each layer's input or output). The backward pass needs them to compute each layer's gradient. Normally all are kept, so memory grows with the number of layers and the batch size.

#flashcards/ml/concept #flashcards/topic/training How does checkpointing work step by step?::Cut the layers into segments, keep the value at the start of each segment (the checkpoint) and throw away the values inside. When the backward pass reaches a segment, rerun that segment's forward pass from its checkpoint to rebuild the values, use them, and discard them again.

#flashcards/ml/concept #flashcards/topic/training Does checkpointing change the gradients?::No. The same maths is redone, so the same gradients come out. In the note's toy (12 layers, k = 2, 3, 4, 6) the loss and all gradients matched the no-checkpoint run every time.

#flashcards/ml/concept #flashcards/topic/training What does checkpointing not reduce?::Only activation memory. The weights, their gradients and the optimizer state stay the same size.

#flashcards/ml/concept #flashcards/topic/training What does PyTorch's `torch.utils.checkpoint` keep and recompute?::It keeps the function's inputs, does not keep the tensors made inside it, and reruns the function during backward.

#flashcards/ml/concept #flashcards/topic/training Which `use_reentrant` value do the PyTorch docs recommend and why must you pass it?::`use_reentrant=False`. The docs say to pass it explicitly (a future release will raise an error if omitted). The non-reentrant version records the autograd graph, works with `torch.autograd.grad` and can stop recomputing early. The reentrant one reruns the whole function in backward under `no_grad()`.

#flashcards/ml/concept #flashcards/topic/training What does `checkpoint_sequential` do?::Splits a sequential model into a number of segments. Every segment except the last discards its inner activations, and each segment's input is saved so it can be rerun.

#flashcards/ml/concept #flashcards/topic/training What randomness and side-effect caveats apply?::Rerunning can advance the random number generator (for example dropout), so `preserve_rng_state=True` saves and restores it by default (a performance cost). Only one device type's RNG state is saved. If the function behaves differently in backward than in forward (a global variable), the result may be an error or silently wrong gradients.

#flashcards/ml/formula #flashcards/topic/training Write the idealised activation memory with $k$ segments and its best split, and explain each symbol.
?
$$M(k)=k+\frac{n}{k},\qquad k^{*}=\sqrt n,\qquad M(k^{*})=2\sqrt n$$
- $n$: the number of layers (each activation taken as one unit). Without checkpointing the memory is $n$ units.
- $k$: the number of segments. More segments keep more boundary values but shorten the part to recompute. $k=1$ and $k=n$ both give about $n$.
- $k+n/k$: the stored boundaries plus the largest segment held while it is recomputed. The two terms are equal at $k=\sqrt n$.
- $2\sqrt n$: the smallest value, so memory grows like $\sqrt n$ instead of $n$ (the $O(\sqrt n)$ result of Chen et al., abstract).

#flashcards/ml/example #flashcards/topic/training For $n=100$ layers, what is the memory for $k=4$, $k=50$ and $k=10$?::No checkpointing: 100 units. $k=4$: $4+25=29$. $k=50$: $50+2=52$. $k=10=\sqrt{100}$: $10+10=20$, five times less. The code searched every $k$ and found 10 for $n=100$ and 32 for $n=1000$ (63.25, equal to $2\sqrt{1000}$).

#flashcards/ml/formula #flashcards/topic/training Write the relative time cost of checkpointing and explain each symbol.
?
$$\frac{T_{\text{ckpt}}}{T_{\text{plain}}}=\frac{F+F+B}{F+B}=\frac{2F+B}{F+B}\ \xrightarrow{\,B=2F\,}\ \frac43$$
- $F$: the time of one forward pass. Checkpointing adds one extra $F$ in total.
- $B$: the time of the backward pass. The larger $B$ is next to $F$, the smaller the relative cost of the extra $F$.
- $B=2F$: the usual rule of thumb (from memory) that backward costs about twice forward. It is an assumption, and real ratios differ.

#flashcards/ml/example #flashcards/topic/training With $F=1$ and $B=2$, what are the plain and checkpointed step times, and what does the paper report?::Plain $1+2=3$, checkpointed $1+1+2=4$, ratio $4/3$, about 33% more time. The paper reports about 30% extra time in a real experiment (abstract): a 1,000-layer residual network on ImageNet going from 48G to 7G of memory.

#flashcards/ml/formula #flashcards/topic/training Write the number of forward calls with $k$ segments (all but the last rerun) and explain each symbol.
?
$$\text{forward calls}=n+\Big(n-\frac nk\Big)=2n-\frac nk$$
- $n$: the number of layers. The first $n$ is the normal forward pass.
- $k$: the number of segments. The last segment is not rerun, so $n-n/k$ layers are recomputed in backward.
- A bigger $k$ makes the last segment smaller, so more is rerun, and the count moves toward $2n$ but never reaches it.

#flashcards/ml/example #flashcards/topic/training For $n=12$, what is the number of forward calls for $k=2$, 4 and 6?::$24-6=18$, $24-3=21$ and $24-2=22$. The note's code counted exactly 18, 21 and 22.

#flashcards/ml/example #flashcards/topic/training Why does the count of saved activation tensors in the note's toy stop falling (25, 14, 11, 10, 10)?::The last segment is not checkpointed, so its tensors are saved in full (k = 4 and k = 6 both give 10). Also the count is what autograd holds after the forward pass. The real peak also includes the segment being recomputed in backward.

#flashcards/ml/concept #flashcards/topic/training What did the note not measure?::Actual memory in bytes, actual run time, and any GPU. The code ran on the CPU and counted tensors and calls. The $B=2F$ rule is from memory.

#flashcards/ml/cloze #flashcards/topic/training Gradient checkpointing keeps only the ==checkpoints== and recomputes the values between them during the ==backward== pass.
