---
tags: [fine-tuning, LoRA, LLM, transformer, training]
status: growing
level: 4
tier: 3
---

# Fine-tuning

## 🟢 1. Simple (age 12)
Fine-tuning means taking an AI that already learned a lot and giving it a **short extra course** so it gets good at one specific job.

**Example:** A new student has read thousands of books, so they already know how to read and write. You do not teach them the alphabet again. You give them 50 sample customer-support chats and say "answer like this". Now they sound like your support team. The big learning is done; fine-tuning is the small lesson on top.

## 🔵 2. How it works
1. Start from a **pre-trained** model (an [[LLM]] that already read a huge amount of text).
2. Collect a small set of examples of the behaviour you want: question and ideal answer pairs.
3. Show the model each question and let it guess the answer. Compare to the ideal answer and measure how wrong it was.
4. Nudge the model's numbers a little so next time it is less wrong ([[Gradient Descent]]).
5. Repeat for a few rounds. Stop before it just memorises your examples.

Two ways to do the nudging:
- **Full fine-tuning:** change *all* of the model's numbers. Strong, but needs lots of memory and makes a whole new copy of the model.
- **LoRA (the popular shortcut; full note: [[LoRA]]):** freeze the original numbers and train only a tiny add-on. It is like putting a small sticker on a big textbook instead of reprinting the book.

![[lora-diagram.svg]]
*LoRA: the big pretrained weights W stay frozen (left, "Pesos preadestrados" is Spanish/Galician for "pretrained weights"). A tiny pair of matrices B and A (right) is trained, and its output is added on top. Image: LauAI, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:LoRA.svg), CC BY-SA 4.0.*

Fine-tuning changes *how the model behaves* (style, format, skill). To give it *new facts* that change often, [[RAG]] is usually the easier tool.

## 🟠 3. Technical detail
Common kinds:
- **Supervised fine-tuning (SFT):** train on (prompt, ideal response) pairs. Teaches instruction following, tone, format.
- **Preference tuning:** train on "this answer is better than that one" data ([[RLHF]], DPO).
- **Domain adaptation:** keep training on text from one field (law, medicine, your codebase).

Parameter-efficient methods (train a tiny fraction of the numbers):
- **LoRA:** add low-rank matrices next to chosen weight matrices (see [[Matrix]]). Typical rank $r$ is 8 to 64. The trained add-on is a few megabytes and can be swapped or merged back.
- **QLoRA:** keep the frozen base model in 4-bit numbers to save memory, train LoRA on top.
- **Prompt/prefix tuning:** train a few extra "virtual tokens" instead of weights.

Practical recipe:
- Data quality beats quantity. A few hundred to a few thousand clean examples can be enough for style and format.
- Compute the loss **only on the response tokens**, not the prompt.
- Use a much smaller learning rate than pre-training (for LoRA often around $10^{-4}$, for full tuning around $10^{-5}$ from memory; tune it).
- Hold out some examples and watch the validation loss. If it rises while training loss falls, you are overfitting.
- Hardware: see [[GPU]] for why memory is the limit.

```python
import numpy as np
# LoRA forward pass for one layer: h = W0 x + (alpha/r) * B (A x)
W0 = np.eye(3)                              # frozen pretrained weights
A = np.array([[1., 1., 0.]])                # r x d  (trained)
B = np.array([[1.], [0.], [1.]])            # d x r  (trained, starts at 0)
alpha, r = 2, 1
x = np.array([1., 2., 3.])
print(W0 @ x + (alpha / r) * (B @ (A @ x)))  # [7. 2. 9.]
```

**Making fine-tuning cheaper and faster (small GPU, or less waiting).** Fine-tuning runs out of memory long before it runs out of ideas, because training keeps several numbers for every weight. These are the usual tools:
- **LoRA / QLoRA** ([[LoRA]], [[QLoRA]]): train a tiny add-on instead of every weight. The LoRA abstract reports 10,000 times fewer trainable numbers and 3 times less GPU memory than fine-tuning GPT-3 175B with Adam. The QLoRA abstract reports fine-tuning a 65B model on a single 48 GB GPU by keeping the frozen base in 4-bit numbers ([[Quantization]]).
- **Mixed precision** (16-bit numbers for most of the work, plus one 32-bit master copy of the weights): the abstract of the mixed-precision paper reports nearly 2 times less memory, and says half-precision hardware can be faster.
- **Gradient accumulation** ([[Gradient Accumulation]]): run several small batches, add up their corrections, then update once. It gives the effect of a big batch with the memory of a small one. It saves memory, not time.
- **Smaller batches and shorter texts:** the memory for the in-between results grows with both.
- **More than one GPU** ([[Model Parallelism]]): split the work. This is the option for speed, since the others mostly save memory.
- **Fewer, cleaner examples:** a few thousand good examples cost far fewer steps than a huge messy set (the "data quality" point above).

```python
import numpy as np

# memory for the numbers kept per weight while training (activations NOT counted)
P, d, layers, mats, r = 7e9, 4096, 32, 4, 8            # 7B model; LoRA of rank 8 on 4 matrices per layer (my choice)
GB = 1e9
train = layers * mats * 2 * d * r                       # LoRA numbers: B (d x r) and A (r x d) per matrix
per_full = 2 + 2 + 4 + 8                                # fp16 weight, fp16 gradient, fp32 master copy, Adam's two fp32 numbers
print(f"LoRA trains {train:,} numbers = {train / P:.3%} of {P:.0e}")
full = P * per_full / GB
lora = (P * 2 + train * per_full) / GB                  # frozen fp16 base + full recipe only for the small add-on
qlora = (P * 0.5 + train * per_full) / GB               # frozen base in 4 bits (quantisation constants ignored)
print(f"full fine-tuning : {full:6.1f} GB  ({per_full} bytes per weight)")
print(f"LoRA             : {lora:6.1f} GB")
print(f"QLoRA (4-bit)    : {qlora:6.1f} GB")

# gradient accumulation: the average of micro-batch gradients equals the big-batch gradient
rng = np.random.RandomState(0)
X, y, w = rng.randn(16, 3), rng.randn(16), rng.randn(3)
grad = lambda Xb, yb: 2 * Xb.T @ (Xb @ w - yb) / len(yb)   # gradient of mean squared error
big = grad(X, y)
acc = sum(grad(X[i:i + 4], y[i:i + 4]) for i in range(0, 16, 4)) / 4
print("batch 16 at once vs 4 micro-batches of 4, largest difference:", float(np.abs(big - acc).max()))
```

What it printed: LoRA trains 8,388,608 numbers, which is 0.120% of a 7-billion-parameter model. Memory for weights, gradients and optimizer numbers: about 112.0 GB for full fine-tuning, 14.1 GB with LoRA and 3.6 GB with QLoRA. Four micro-batches of 4 gave the same gradient as one batch of 16, up to a difference of $4\times10^{-16}$.

**What this toy leaves out:** the in-between results (activations), which can be large for long texts, and the cost of the 4-bit constants. The 16 bytes per weight is my own sum (2 + 2 + 4 + 8) of the pieces above, with Adam's two stored numbers per weight from memory, not a figure from a paper. The 7B size and the choice of 4 matrices per layer are mine. So the three totals show the pattern, not what a real run will need.

**Alignment tuning (why preference tuning matters for usability).** Fine-tuning on ideal answers (SFT) teaches a model to follow instructions, but it does not tell the model which of two decent answers a person prefers. Alignment tuning adds that signal, using pairs where people marked one answer as better ([[RLHF]], [[Instruction Tuning]], [[InstructGPT 2022]]). The result is a model that is more helpful, and that refuses harmful requests more reliably. I wrote this paragraph from the vault's own notes and memory, and I did not read a new source for it.

## 🔴 4. Deep / formal
**SFT objective:** the usual next-token loss, restricted to the response part of each example:

$$
\mathcal{L}_{\text{SFT}}(\theta)=-\,\mathbb{E}_{(x,y)\sim\mathcal{D}}\ \sum_{t=1}^{|y|}\log p_\theta\big(y_t\mid x,\,y_{<t}\big)
$$

**Where:**
- $\theta$ (theta) is the list of model numbers being adjusted. $\mathcal{L}_{\text{SFT}}$ is the loss: how wrong the model is. Training lowers it ([[Gradient Descent]]).
- $(x,y)$ is one training example: $x$ is the prompt (the question) and $y$ is the ideal response. $\mathcal{D}$ is the small fine-tuning dataset, and $\mathbb{E}$ is the average over its examples.
- $y_t$ is the $t$-th token of the response and $y_{<t}$ is the response tokens before it. $|y|$ is the response length. **Matters:** the sum runs only over *response* tokens, so the model is graded on its answer, not on repeating the prompt.
- $p_\theta(y_t\mid x,y_{<t})$ is the probability the model gives to the correct next token, knowing the prompt and the response so far. $\log$ of it is near 0 when the model is confident and right, very negative when it is surprised.
- The leading minus sign turns that into a positive loss that shrinks as the model improves.

**Example (tiny numbers):** prompt $x=$ "Capital of France?". Ideal response $y=$ "Paris" ".". The model gives the correct tokens probabilities $0.6$ and $0.9$.
- Token "Paris": $-\log(0.6)=\mathbf{0.51}$.
- Token ".": $-\log(0.9)=\mathbf{0.11}$.
- Loss for this example $=0.51+0.11=\mathbf{0.62}$. The prompt tokens "Capital of France?" add nothing.
- Training nudges $\theta$ so the "Paris" probability rises toward 1, which pushes the loss toward 0.

**LoRA parametrisation** (Hu et al., 2021): instead of learning a full update, learn a low-rank one and add it to the frozen weights:

$$
h=W_0x+\frac{\alpha}{r}\,BAx,\qquad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r\ll\min(d,k)
$$

**Where:**
- $x$ is the input vector to the layer and $h$ is its output. $W_0$ is the original pretrained weight [[Matrix]], **frozen**, so it never changes.
- $A$ (shape $r\times k$) squeezes the input down to only $r$ numbers, and $B$ (shape $d\times r$) expands those $r$ numbers back up to the output size $d$. **Matters:** $BA$ is a small, cheap "correction" that has the same shape as a full update.
- $r$ is the **rank**, a small chosen number (like 8), much smaller than the layer sizes $d$ and $k$. **Matters:** it is the width of the bottleneck. Larger $r$ can learn more, but costs more memory.
- $\alpha$ is a scaling constant, and $\frac{\alpha}{r}$ scales the correction. **Matters:** it controls how strongly the add-on affects the result, and keeps the effect steady if you change $r$.
- Standard start: $B=0$ (and $A$ small random numbers). **Matters:** the correction is exactly 0 at first, so the fine-tuned model starts identical to the original and only drifts as it learns.

**Example (tiny numbers):** $d=k=3$, $r=1$, $\alpha=2$. $W_0$ is the identity, $x=[1,2,3]$, $A=[1,\,1,\,0]$, $B=[1,\,0,\,1]^\top$.
- $Ax=1\cdot1+1\cdot2+0\cdot3=\mathbf{3}$ (one number, the "squeezed" input).
- $B(Ax)=3\cdot[1,0,1]=[3,0,3]$. Scale by $\alpha/r=2$: $[6,0,6]$.
- $W_0x=[1,2,3]$. So $h=[1,2,3]+[6,0,6]=\mathbf{[7,\ 2,\ 9]}$.
- At the very start ($B=0$) the correction would be $[0,0,0]$ and $h=[1,2,3]$, exactly the original model.

Parameter saving, from the [[Matrix]] note: one $4096\times4096$ layer has $16{,}777{,}216$ numbers; a rank-8 LoRA trains $2\cdot4096\cdot8=65{,}536$, which is $256\times$ fewer. After training, the correction can be merged: $W=W_0+\frac{\alpha}{r}BA$, so running the model costs nothing extra.

**Trade-offs and caveats:**
- **Catastrophic forgetting:** training hard on a narrow task can make the model worse at everything else. LoRA and small learning rates reduce this because the base weights stay put.
- **Low-rank is a bet:** the useful change for one task is assumed to live in a few directions. Usually true for style and format, less so for teaching large amounts of new knowledge.
- **Overfitting** on tiny datasets: more epochs memorise the examples.
- **Data risk:** a model fine-tuned on messy or biased examples learns exactly that. Always read a sample of your training data.

**What came before, and how this improved on it:**
- **Before:** train a separate model from scratch for each task.
- **Its limit:** each task needed lots of labelled data. ULMFiT's abstract says the existing text methods "still require task-specific modifications and training from scratch".
- **What changed:** start from a pre-trained model and adjust it (ULMFiT, Howard and Ruder, arXiv 1801.06146, 18 January 2018, abstract read). For huge models, update only a small part ([[LoRA]]).
- **How much better:** ULMFiT's abstract reports 18 to 24% lower error on most of six text classification datasets, and with 100 labelled examples it matches training from scratch on 100 times more data. The code below counts the cost side: for one 4096 by 4096 matrix, full fine-tuning updates 16,777,216 numbers and LoRA at rank 8 updates 65,536, which is 256 times fewer.

**Check it yourself:**

```python
d, r = 4096, 8
full = d * d; lora = 2 * d * r
print(f"one {d}x{d} weight matrix: full fine-tuning updates {full:,} numbers; LoRA (rank {r}) updates {lora:,}  ({full // lora}x fewer)")
print("for a model with 32 layers x 4 such matrices:", f"{32 * 4 * full:,}", "vs", f"{32 * 4 * lora:,}")
```

## Related
- [[LLM]] · [[RLHF]] · [[RAG]] · [[Matrix]] · [[Gradient Descent]] · [[GPU]] · [[Transformer]] · [[Pre-training]] · [[Singular Value Decomposition]] · [[Quantization]] · [[Knowledge Distillation]] · [[Transfer Learning]] · [[LoRA]] · [[Instruction Tuning]] · [[QLoRA]] · [[Catastrophic Forgetting]] · [[Continual Learning]] · [[Unsloth]] · [[LoRA Fine-tuning on CPU]]

## Sources
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, 2023
- Ouyang et al., *Training language models to follow instructions with human feedback*, 2022 (SFT stage)
- Howard and Ruder, ULMFiT, arXiv 1801.06146 (abstract read for the history block)
- Abstracts read for the "cheaper and faster" section: Hu et al., LoRA, arXiv 2106.09685; Dettmers et al., QLoRA, arXiv 2305.14314; Micikevicius et al., *Mixed Precision Training*, arXiv 1710.03740. Not read beyond the abstracts. The 16 bytes per weight and the alignment paragraph are from memory.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm Fine-tuning:::Taking an AI that already learned a lot and giving it a short extra course so it gets good at one specific job.

#flashcards/ml/formula #flashcards/topic/llm Write the supervised fine-tuning loss and explain each symbol.
?
$$\mathcal{L}_{\text{SFT}}(\theta)=-\,\mathbb{E}_{(x,y)\sim\mathcal{D}}\ \sum_{t=1}^{|y|}\log p_\theta\big(y_t\mid x,\,y_{<t}\big)$$
- $\theta$: the model numbers being adjusted. $\mathcal{L}_{\text{SFT}}$: the loss, how wrong the model is.
- $(x,y)$: one training example. $x$ is the prompt (the question), $y$ is the ideal response. $\mathcal{D}$: the small set of such examples.
- $y_t$: the $t$-th token of the response. $y_{<t}$: the response tokens before it.
- The loss counts only the response tokens, not the prompt tokens.

#flashcards/ml/formula #flashcards/topic/llm Write the LoRA layer and explain each symbol.
?
$$h=W_0x+\frac{\alpha}{r}\,BAx,\qquad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r\ll\min(d,k)$$
- $x$: the input vector to the layer. $h$: its output.
- $W_0$: the original pretrained weight matrix, frozen, so it never changes.
- $A$ (shape $r\times k$): squeezes the input down to only $r$ numbers. $B$ (shape $d\times r$): expands those back up to the output size.
- $r$: the rank, much smaller than $d$ and $k$. $\alpha/r$: a fixed scale on the correction.
- Only $A$ and $B$ are trained.

#flashcards/ml/cloze #flashcards/topic/llm During fine-tuning with LoRA, the pretrained weights stay ==frozen== and only the small matrices ==A and B== are trained.

#flashcards/ml/example #flashcards/topic/llm For the prompt "Capital of France?" and ideal response "Paris" ".", the model gives the correct tokens 0.6 and 0.9. What is the loss?::−ln 0.6 = 0.51 and −ln 0.9 = 0.11. Loss 0.62. The prompt tokens are not counted.

#flashcards/ml/example #flashcards/topic/llm LoRA with $W_0=I$, $x=[1,2,3]$, $A=[1,1,0]$, $B=[1,0,1]^\top$, $\alpha=2$, $r=1$. What is $h$?::Ax = 3. B(3) = [3, 0, 3]. Scale α/r = 2 gives [6, 0, 6]. W₀x = [1, 2, 3]. h = [7, 2, 9].

#flashcards/ml/image #flashcards/topic/llm ![[lora-diagram.svg]]
What does this picture show?
?
LoRA: the big pretrained weights W stay frozen (left). A tiny pair of matrices B and A (right) is trained, and its output is added on top. (The label "Pesos preadestrados" is Spanish/Galician for "pretrained weights".)
