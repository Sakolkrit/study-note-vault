---
tags: [machine-learning, deep-learning, neural-network, RNN, LSTM, sequence, NLP, history]
status: growing
level: 4
tier: 3
---

# LSTM

**LSTM** stands for *long short-term memory*. It is a [[Recurrent Neural Network]] whose memory has **gates**: small switches that decide what to keep, what to forget and what to show. That lets it carry information across more steps than a plain RNN.

## 🟢 1. Simple (age 12)
An LSTM is a notebook with a pencil and an eraser. At every new word it can **erase** some old notes, **write** a few new ones, and then decide how much of the notebook to **read out loud**. A plain RNN has no notebook: it keeps one thought in its head and keeps overwriting it.

**Example:** you read a story about a cat for ten pages. A notebook entry "the hero is a cat" stays there as long as you do not erase it, so on page ten you still know it.

## 🔵 2. How it works
1. An LSTM is a [[Recurrent Neural Network]]: it reads a sequence one step at a time and reuses the same weights at every step.
2. Besides the usual short-term state $h$ (what it shows), it keeps a second list of numbers, the **cell state** $c$. This is the notebook, the long-term memory.
3. At each step three small switches look at the new input and the old $h$. Each switch outputs numbers between 0 and 1 (a sigmoid, see [[Activation Function]]):
   - the **forget gate**: how much of each old note to keep (1 = keep all, 0 = erase);
   - the **input gate**: how much of the new note to write;
   - the **output gate**: how much of the notebook to show as $h$.
4. A **candidate** list says what new notes could be written (a tanh of the input and old $h$).
5. The new notebook is: old notes times the forget gate, **plus** new notes times the input gate. The word to notice is **plus**: the old memory is not squeezed through a weight matrix and a squashing function at every step, it just gets scaled.
6. Because of that straight path, the learning signal ([[Backpropagation]]) can travel back through many steps without shrinking as fast ([[Vanishing and Exploding Gradients]]).
7. **The catch:** it has four times the weights of a plain RNN, it still reads one word at a time (slow on a [[GPU]]), and its memory is a fixed-size list, so very long inputs are still hard. [[Attention]] and the [[Transformer]] took over most language work for these reasons.

![[lstm-cell.svg]]
*An LSTM cell (long short-term memory). The top line carries the **cell state** $c$, the long-term memory, from $c_{t-1}$ to $c_t$. Along it, a gate (the orange $\sigma$ box feeding the first "x") can scale the old memory down, and new information (a $\sigma$ gate times a tanh candidate) is added with the "+". Another gate decides how much of the memory to reveal as the output $h_t$. Image: Guillaume Chevalier, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:LSTM_Cell.svg), CC BY-SA 4.0. I read the roles of the gates off the picture and the standard description of the LSTM, not from the file text.*

## 🟠 3. Technical detail
- **State:** two vectors per step, the cell state $c_t$ (long-term) and the hidden state $h_t$ (the output).
- **Gates:** forget $f$, input $i$ and output $o$, each a sigmoid of a linear function of the input $x_t$ and the old $h_{t-1}$. The candidate $g$ is a tanh of the same kind of function.
- **Cost:** four weight blocks instead of one, so four times the parameters of a plain RNN of the same size, and four times the work per step.
- **GRU** (from memory, 2014): a lighter relative with two gates and no separate cell state. It often does about as well with fewer parameters.
- **Forget-gate bias:** starting the forget gate leaning towards "keep" (a positive bias) is a common trick, because a gate that starts at "forget" erases the memory before the network has learned to use it (from memory of common practice; the code at the end uses it).
- **Training:** unrolled through time with [[Backpropagation]], usually with gradient clipping. It is sequential, so it cannot use a [[GPU]] across positions the way a [[Transformer]] can.
- **Where it was used:** speech, handwriting and translation systems before 2017, for example the [[Seq2Seq 2014]] translation model (see [[From FNN to RNN]] and [[History of Text Analytics]]).

```python
import torch
torch.manual_seed(0)
d, h = 3, 4
cell = torch.nn.LSTMCell(d, h)
x, h0, c0 = torch.randn(1, d), torch.randn(1, h), torch.randn(1, h)
z = x @ cell.weight_ih.T + h0 @ cell.weight_hh.T + cell.bias_ih + cell.bias_hh       # all four gate scores in one product
i, f, g, o = z.chunk(4, 1)                                                           # PyTorch's order: input, forget, candidate, output
c1 = torch.sigmoid(f) * c0 + torch.sigmoid(i) * torch.tanh(g)                        # scale the old memory, add the new
h1 = torch.sigmoid(o) * torch.tanh(c1)
hr, cr = cell(x, (h0, c0))
print("by hand vs torch.nn.LSTMCell, largest difference:", f"{max((h1 - hr).abs().max().item(), (c1 - cr).abs().max().item()):.1e}")
print("weights in one LSTM layer:", sum(p.numel() for p in cell.parameters()), " in a plain RNN layer of the same size:", sum(p.numel() for p in torch.nn.RNNCell(d, h).parameters()), "(4 times as many)")
```

**For time series (added with the [[Time Series]] notes).** An LSTM forecasts by reading a window of the last $L$ values and predicting the next one, or the next several at once (the "direct" method). Three practical points, from [[Forecasting Shoot-out]]: **scale** the inputs; feed it **changes** (differences or growth rates) and not raw levels, so that the numbers stay in a steady range ([[Stationarity]]); and test it on forward-moving origins ([[Time Series Cross-Validation]]). In that project a small LSTM (32 units, a window of 24 changes, 12 outputs, 200 full-batch Adam steps, no tuning) scored a MASE of 0.64 on the airline series and 0.32 on Mauna Loa CO2 ([[Forecast Metrics]]; mean of 3 seeds, spread 0.012 and 0.024, seven origins). That was the best of nine methods on CO2, but it was behind Holt-Winters (0.56) and the airline-model SARIMA (0.62) on the airline series ([[Exponential Smoothing]], [[ARIMA]]). So an LSTM is a serious option for a series, but not an automatic winner over the classic methods.

## 🔴 4. Deep / formal
**The gates.** With input $x_t$ and previous output $h_{t-1}$, each gate is a sigmoid of a weighted sum:

$$
f_t=\sigma\!\big(W_f x_t+U_f h_{t-1}+b_f\big),\quad i_t=\sigma(\cdots),\quad o_t=\sigma(\cdots),\quad g_t=\tanh(\cdots)
$$

**Where:**
- $x_t$ is the input at step $t$ and $h_{t-1}$ the previous output. $W_f$ and $U_f$ are weight matrices ([[Matrix]]) for the forget gate, and $b_f$ is its bias. The gates $i_t$, $o_t$ and the candidate $g_t$ have their own weights, written "$\cdots$" here.
- $\sigma$ is the sigmoid, which squashes any number into 0 to 1. **Matters:** a gate output of 0 closes the gate fully and 1 opens it fully, so the gate acts like a soft switch; without the squashing it could not behave like a switch.
- **Matters:** each gate has its own weights, learned from data, so the network decides by itself when to keep, write or show.

**Example (tiny numbers):** a one-number forget gate with $W_f=1$, $U_f=1$, $b_f=0.5$, input $x_t=1$ and previous output $h_{t-1}=0.5$.
- Inside: $1\cdot1+1\cdot0.5+0.5=\mathbf2$.
- $f_t=\sigma(2)=\mathbf{0.881}$.
- Meaning: the gate is mostly open, so about 88% of the old memory is kept at this step.

**The memory update.** The new cell state and the output:

$$
c_t=f_t\odot c_{t-1}+i_t\odot g_t,\qquad h_t=o_t\odot\tanh(c_t)
$$

**Where:**
- $c_{t-1}$ and $c_t$ are the cell state before and after the step. $\odot$ means multiply entry by entry.
- $f_t\odot c_{t-1}$ is the old memory after forgetting, and $i_t\odot g_t$ is the new content after the input gate lets part of it in.
- $h_t$ is the output, the cell state squashed by tanh and filtered by the output gate $o_t$.
- **Matters:** the path from $c_{t-1}$ to $c_t$ is "multiply by $f_t$, then add". The slope along this path is just $f_t$, with no weight matrix and no tanh in the way. If $f_t$ stays near 1, the signal survives many steps. In a plain RNN the same path goes through a weight matrix and a tanh at every step.

**Example (tiny numbers):** old cell $c_{t-1}=0.5$, and gate scores (before squashing) $f:2$, $i:0$, $g:1$, $o:1$.
- Gates: $f=\sigma(2)=\mathbf{0.881}$, $i=\sigma(0)=\mathbf{0.5}$, $o=\sigma(1)=\mathbf{0.731}$, candidate $g=\tanh(1)=\mathbf{0.762}$.
- New cell: $c_t=0.881\cdot0.5+0.5\cdot0.762=0.440+0.381=\mathbf{0.821}$.
- Output: $h_t=0.731\cdot\tanh(0.821)=0.731\cdot0.676=\mathbf{0.494}$.
- Meaning: part of the old memory (0.440 of 0.5) survived, part of the new content (0.381 of 0.762) was written, and about 73% of the squashed notebook was shown.

**How long a memory survives.** If the forget gate sits at the same value $f$ for $T$ steps, the share of the old memory that remains in the cell is $f^T$. The code at the end prints: with $f=0.9$, $0.9^{20}=0.122$ and $0.9^{50}=0.0051$. With $f=0.99$, $0.99^{50}=0.605$ and $0.99^{100}=0.366$. With $f=0.5$, $0.5^{50}=8.9\times10^{-16}$.

**Further notes:**
- **Memory is still finite:** the cell is a fixed-size vector, so it cannot hold an arbitrarily long history exactly.
- **Peephole connections, coupled gates and other variants** exist (from memory); the version above is the common one.
- **Not a free win:** the code at the end includes a small test where the LSTM does worse than a plain RNN. Gates add weights and can make training slower to get started.

**What came before, and how this improved on it:**
- **Before:** the plain [[Recurrent Neural Network]], whose memory is rewritten through a weight matrix and a tanh at every step.
- **Its limit:** the learning signal fades over many steps. The note [[From FNN to RNN]] measured, for an untrained network at length 50, a gradient of $1.4\times10^{-14}$ for the RNN.
- **What changed:** a separate cell state with an additive update and learned gates (Hochreiter and Schmidhuber, 1997; the date is checked on the Wikipedia timeline page, the authors from memory). Forget gates were added a few years later (Gers and others, around 2000; from memory).
- **How much better:** in the measurement in [[From FNN to RNN]], the untrained LSTM's gradient at length 50 is $1.8\times10^{-11}$, about 1,000 times larger than the RNN's, but still tiny. In the code below, a forget gate at 0.99 keeps 60.5% of the memory after 50 steps. **Honest catch:** in my small test (remember one bit across 60 silent steps, 16 units, 1,500 training steps, seed 0), the plain RNN reached 1.000 accuracy and the LSTM stayed at chance (0.498). In my other runs the RNN also reached 1.0 at lengths 30 and 100 and with a second seed, while the LSTM learned length 30 only with enough steps and did not at 60 or 100. I did not tune it, so this shows that the LSTM is not automatically better and is sensitive to its setup.

**Check it yourself:** (takes about 40 seconds)

```python
import torch, torch.nn as nn
print("share of the old memory left after T steps if the forget gate stays at f (cell path: just f, T times):")
for f in (0.5, 0.9, 0.99): print(f"  f={f}: T=20 {f ** 20:.2e}   T=50 {f ** 50:.2e}   T=100 {f ** 100:.2e}")
def run(kind, T=60, steps=1500, seed=0):
    torch.manual_seed(seed)
    m = {"RNN": nn.RNN, "LSTM": nn.LSTM}[kind](1, 16, batch_first=True); head = nn.Linear(16, 1)
    if kind == "LSTM":
        with torch.no_grad(): m.bias_ih_l0[16:32] += 1.0                  # start with the forget gate leaning towards "keep"
    opt = torch.optim.Adam(list(m.parameters()) + list(head.parameters()), 0.01)
    def batch(n):
        b = torch.randint(0, 2, (n,)).float(); x = torch.zeros(n, T, 1); x[:, 0, 0] = b * 2 - 1; return x, b      # one bit at step 0, then silence
    for _ in range(steps):
        x, y = batch(64); out, _ = m(x); l = nn.functional.binary_cross_entropy_with_logits(head(out[:, -1]).squeeze(1), y)
        opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    x, y = batch(2000); out, _ = m(x); return ((head(out[:, -1]).squeeze(1) > 0).float() == y).float().mean().item()
print("task: remember a bit shown at step 0 and report it after 60 silent steps (1,500 training steps, 16 units, seed 0)")
for k in ("RNN", "LSTM"): print(f"  {k}: test accuracy {run(k):.3f}")
```

## Related
- [[Recurrent Neural Network]] · [[Vanishing and Exploding Gradients]] · [[Backpropagation]] · [[Activation Function]] · [[Attention]] · [[Transformer]] · [[Seq2Seq 2014]] · [[From FNN to RNN]] · [[History of Text Analytics]] · [[History of AI]] · [[Perceptron]] · [[GPU]] · [[Time Series]] · [[Forecasting Shoot-out]] · [[Stationarity]]

## Sources
- Hochreiter and Schmidhuber (1997), the LSTM: date checked on the Wikipedia *Timeline of artificial intelligence* page; authors and details from memory, paper not opened.
- Gers and others (about 2000), forget gates; Cho and others (2014), GRU: from memory.
- Image: [LSTM_Cell.svg](https://commons.wikimedia.org/wiki/File:LSTM_Cell.svg) by Guillaume Chevalier, CC BY-SA 4.0 (same file as in [[Recurrent Neural Network]]).
- All code results were run for this note (PyTorch 2.8.0, CPU, toy tasks).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/generative LSTM (long short-term memory):::A notebook with a pencil and an eraser. At every new word it can erase some old notes, write a few new ones, and decide how much of the notebook to read out loud. A plain RNN has no notebook and keeps overwriting one thought.

#flashcards/ml/formula #flashcards/topic/generative Write the LSTM gates and explain each symbol.
?
$$f_t=\sigma\!\big(W_f x_t+U_f h_{t-1}+b_f\big),\quad i_t=\sigma(\cdots),\quad o_t=\sigma(\cdots),\quad g_t=\tanh(\cdots)$$
- $x_t$: the input at step $t$. $h_{t-1}$: the previous output.
- $W_f$, $U_f$: weight matrices for the forget gate. $b_f$: its bias. The input gate $i_t$, output gate $o_t$ and candidate $g_t$ have their own weights, written "$\cdots$".
- $\sigma$: the sigmoid, which squashes any number into 0 to 1, so a gate is a "how much to let through" dial. $\tanh$ squashes the candidate to −1 to 1.

#flashcards/ml/formula #flashcards/topic/generative Write the LSTM cell and output update and explain each symbol.
?
$$c_t=f_t\odot c_{t-1}+i_t\odot g_t,\qquad h_t=o_t\odot\tanh(c_t)$$
- $c_{t-1}$, $c_t$: the cell state before and after the step. $\odot$: multiply entry by entry.
- $f_t\odot c_{t-1}$: the old memory after forgetting. $i_t\odot g_t$: the new content after the input gate lets part of it in.
- $h_t$: the output, the cell state squashed by tanh and filtered by the output gate.

#flashcards/ml/cloze #flashcards/topic/generative The LSTM has a ==forget== gate, an ==input== gate and an ==output== gate, all sigmoids between 0 and 1.

#flashcards/ml/example #flashcards/topic/generative A forget gate with $W_f=1$, $U_f=1$, $b_f=0.5$, $x_t=1$, $h_{t-1}=0.5$. What is $f_t$?::Inside: 1 + 0.5 + 0.5 = 2. f_t = σ(2) = 0.881. The gate is mostly open, so about 88% of the old memory is kept.

#flashcards/ml/example #flashcards/topic/generative Old cell $c_{t-1}=0.5$ and gate scores (before squashing) $f:2$, $i:0$, $g:1$, $o:1$. What are $c_t$ and $h_t$?::f = σ(2) = 0.881, i = σ(0) = 0.5, o = σ(1) = 0.731, g = tanh(1) = 0.762. c_t = 0.881 × 0.5 + 0.5 × 0.762 = 0.440 + 0.381 = 0.821. h_t = 0.731 × tanh(0.821) = 0.494 (my arithmetic, the note's value was cut off).

#flashcards/ml/image #flashcards/topic/generative ![[lstm-cell.svg]]
What does this picture show?
?
An LSTM cell. The top line carries the cell state c, the long-term memory. A gate can scale the old memory down, and new information (a sigma gate times a tanh candidate) is added with the "+". Another gate decides how much of the memory to reveal as the output.

#flashcards/ml/example #flashcards/topic/generative In the Forecasting Shoot-out, how did a small LSTM compare with the classic methods?::MASE 0.64 on the airline series (Holt-Winters 0.56, SARIMA 0.62) and 0.32 on Mauna Loa CO2, the best of nine methods there (SARIMA 0.35, Holt-Winters 0.38). Mean of 3 seeds over 7 origins, untuned, fed growth rates and not raw levels. My run on this machine.
