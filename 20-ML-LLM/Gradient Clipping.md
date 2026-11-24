---
tags: [gradient-clipping, training, optimization, exploding-gradients, RNN, deep-learning, ML]
status: growing
level: 4
tier: 2
---

# Gradient Clipping

**Gradient clipping** limits the size of the gradient before a training step. If the gradient is longer than a chosen limit, it is shrunk to exactly that limit, keeping its direction. It protects training from a rare huge gradient that would throw the weights far away. The [[Seq2Seq 2014]] paper uses it with a limit of 5.

## 🟢 1. Simple (age 12)
You are walking downhill in the dark, and a signpost tells you how steep it is and which way to go. Once in a while it shouts "STEEP! TAKE A HUGE STEP!" and you would fly off a cliff. Gradient clipping is a rule: "never take a step longer than one stride". You still go the same way, but never too far.

**Example:** the signpost says "go 12 metres towards the lake", but your rule says "at most 5 metres". You walk 5 metres towards the lake.

## 🔵 2. How it works
1. **Training** changes the weights a little at every step, in the direction given by the **gradient** (see [[Gradient Descent]] and [[Backpropagation]]).
2. **Sometimes the gradient explodes:** it becomes enormous, for example in a long chain of layers or in a [[Recurrent Neural Network]] ([[Vanishing and Exploding Gradients]]).
3. **Measure the gradient's length** (its norm).
4. **If it is longer than the limit,** multiply the whole gradient by (limit divided by length). Now its length is exactly the limit and it points the same way.
5. **If it is shorter,** leave it alone.
6. **Take the step** with the clipped gradient.

## 🟠 3. Technical detail
All numbers come from the code below (Python 3.9.6, PyTorch 2.8.0, CPU).

- **By hand:** gradient `(9, -8)` has length 12.04, so with limit 5 it is scaled by 0.4152 to `(3.737, -3.322)`, length exactly 5.0. A gradient `(2, -1)` (length 2.24) is untouched.
- **The library gives the same:** `torch.nn.utils.clip_grad_norm_` returned the old length 12.04 and left `(3.737, -3.322)` in the gradient.
- **Clamping each entry is different:** clamping `(9, -8)` entry by entry to ±5 gives `(5, -5)`, which points in a slightly different direction (cosine with the original 0.9983). Norm clipping keeps the direction exactly.
- **Explosion in a chain:** a 30-layer chain of multiplications by the weight 1.5 gives a gradient of 30·1.5²⁹ = 3.835e6; clipped to length 5, it becomes 5.0.
- **Limits of my run:** toy numbers only; I did not train a network with and without clipping, so I show the mechanism, not the gain.

```python
import torch
g = torch.tensor([9.0, -8.0]); c = 5.0
n = g.norm().item(); scale = min(1.0, c / n)
print('norm', round(n, 4), 'scale', round(scale, 4), 'clipped', (g * scale).tolist(), 'new norm', round((g * scale).norm().item(), 4))
small = torch.tensor([2.0, -1.0]); print('norm 2.236 <= 5, unchanged:', (small * min(1.0, c / small.norm().item())).tolist())
p = torch.nn.Parameter(torch.zeros(2)); p.grad = g.clone()
tot = torch.nn.utils.clip_grad_norm_([p], max_norm=c); print('library: returned norm', round(tot.item(), 4), 'grad now', [round(x, 4) for x in p.grad.tolist()])
cl = g.clamp(-c, c); print('clamp each entry to +-5:', cl.tolist(), 'direction cos with original', round(torch.cosine_similarity(g, cl, dim=0).item(), 4))
torch.manual_seed(0)
w = torch.full((1,), 1.5, requires_grad=True); x = torch.ones(1)
y = x
for _ in range(30): y = y * w
y.backward(); print('30 layers, weight 1.5: gradient', f'{w.grad.item():.3e}')
w.grad *= min(1.0, c / abs(w.grad.item())); print('after clipping to norm 5:', w.grad.item())
```

## 🔴 4. Deep / formal
**Clipping by norm** (the rule in the Seq2Seq paper, section 3.4):

$$
\mathbf{g}\leftarrow\mathbf{g}\cdot\min\!\Big(1,\ \frac{c}{\lVert\mathbf{g}\rVert_2}\Big)
$$

**Where:**
- $\mathbf{g}$ is the gradient vector of all weights. **Matters:** it is the direction and size of the planned step.
- $\lVert\mathbf{g}\rVert_2$ is its length. **Matters:** it decides whether clipping happens at all.
- $c$ is the limit (5 in Seq2Seq, 0.3 in the QLoRA paper's setup). **Matters:** too small and learning slows because every step is shortened; too large and explosions get through.
- $\min(1,\cdot)$ keeps short gradients unchanged. **Matters:** without it, short gradients would be *stretched* up to length $c$.

**Example (tiny numbers):** $\mathbf{g}=(9,-8)$, $c=5$: $\lVert\mathbf{g}\rVert=\sqrt{145}=12.04$, factor $5/12.04=0.4152$, new gradient $(3.737,-3.322)$ with length 5. For $\mathbf{g}=(2,-1)$ the factor is $\min(1,5/2.24)=1$: no change. The answer means: only too-long gradients are shortened, and never turned.

**Limits and trade-offs:**
- It does not fix vanishing gradients (those are too small, not too big).
- The limit $c$ is a setting to tune; the papers in this vault give no search over it (from my notes of [[Seq2Seq 2014]] and [[QLoRA 2023]]).
- Clipping hides the symptom (a huge gradient) not the cause (for example a bad learning rate), so a run that clips on every step is a warning sign (general knowledge).

**What came before, and how this improved on it:**
- **Before:** plain gradient descent with no limit on the step, used in early recurrent networks, where exploding gradients had been described in the 1990s (the Seq2Seq paper cites Hochreiter 1991 and Bengio and others 1994 for the long-term dependency problem; the clipping rule it cites comes from Graves 2013, dates from the paper's reference list).
- **Its limit:** one rare huge gradient can throw the weights far off and ruin a run.
- **What changed:** rescale the gradient to a maximum length, keeping its direction.
- **How much better:** my toy: a gradient of 3.835e6 becomes 5.0. Published: Seq2Seq trained deep LSTMs with limit 5 and reported no trouble (its text says LSTMs rarely have vanishing gradients but "can have exploding gradients"); the paper does not show a run without clipping.

## Related
[[Gradient Descent]] · [[Backpropagation]] · [[Vanishing and Exploding Gradients]] · [[Recurrent Neural Network]] · [[LSTM]] · [[Learning Rate]] · [[Adam Optimizer]] · [[Seq2Seq 2014]] · [[QLoRA 2023]]

## Sources
- Seq2Seq paper, section 3.4 (Sutskever, Vinyals, Le, 2014; read as text; my note [[Seq2Seq 2014]]); QLoRA paper appendix B.2 (max gradient norm 0.3).
- My own run (Python 3.9.6, PyTorch 2.8.0, CPU). No image: none needed. Not checked: the note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/training Gradient clipping:::Shrinking the gradient to a maximum length before a training step, keeping its direction, so a rare huge gradient cannot throw the weights far away.

#flashcards/ml/concept #flashcards/topic/training What problem does gradient clipping solve?::Exploding gradients: one enormous gradient makes a huge step that ruins the weights. In my toy, 30 layers with weight 1.5 gave a gradient of 3.835e6.

#flashcards/ml/concept #flashcards/topic/training What are the steps of clipping by norm?::Measure the gradient's length; if it is above the limit $c$, multiply the whole gradient by $c$ divided by the length; if not, leave it; take the step.

#flashcards/ml/formula #flashcards/topic/training Write gradient clipping by norm and explain each symbol.
?
$$\mathbf{g}\leftarrow\mathbf{g}\cdot\min\!\Big(1,\ \frac{c}{\lVert\mathbf{g}\rVert_2}\Big)$$
- $\mathbf{g}$: the gradient of all weights, the planned step.
- $\lVert\mathbf{g}\rVert_2$: its length; it decides whether clipping happens.
- $c$: the limit (5 in Seq2Seq); too small slows learning, too large lets explosions through.
- $\min(1,\cdot)$: keeps short gradients unchanged instead of stretching them.

#flashcards/ml/example #flashcards/topic/training Clip $\mathbf{g}=(9,-8)$ to length 5, and what happens to $(2,-1)$?::Length $\sqrt{145}=12.04$, factor 0.4152, result $(3.737,-3.322)$ with length 5. $(2,-1)$ has length 2.24 below 5, so the factor is 1: unchanged (my arithmetic).

#flashcards/ml/concept #flashcards/topic/training Why scale the whole gradient and not clamp each entry?::Scaling keeps the direction exactly; clamping $(9,-8)$ to ±5 gives $(5,-5)$, whose cosine with the original is 0.9983, a slightly different direction (my run).

#flashcards/ml/concept #flashcards/topic/training Why is there a $\min(1,\cdot)$ in the clipping rule?::Without it a short gradient such as $(2,-1)$ would be multiplied by $5/2.24=2.24$ and stretched to length 5. The min makes the rule only ever shrink.

#flashcards/ml/concept #flashcards/topic/training What does gradient clipping not fix?::Vanishing gradients (too small, not too big), and it hides rather than cures a cause such as a bad learning rate (the second part is general knowledge).

#flashcards/ml/concept #flashcards/topic/training Which clipping limits did the vault's papers use?::Seq2Seq: gradient norm limit 5 (gradient divided by batch size 128). QLoRA: max gradient norm 0.3 (appendix B.2).

#flashcards/ml/cloze #flashcards/topic/training Clipping by norm keeps the ==direction== of the gradient and only changes its ==length==.
