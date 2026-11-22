---
tags: [machine-learning, deep-learning, neural-network, RNN, LSTM, sequence, NLP]
status: growing
level: 4
tier: 3
---

# Recurrent Neural Network

Often shortened to **RNN**. It is a [[Neural Network]] for **sequences**: text, speech, music, sensor readings, anything where the order matters.

## 🟢 1. Simple (age 12)
A recurrent network reads a sequence **one piece at a time**, and keeps a **little memory** of what it has read so far. At each step it combines the new piece with its memory, updates the memory, and moves on. It is like reading a sentence word by word and keeping the story in your head, so that by the time you reach "it" you remember what "it" refers to.

**Example:** Reading "The cat, which had been chasing mice all morning, finally **fell asleep**." To know who fell asleep, you have to remember "cat" across all those middle words. An RNN carries a running summary from word to word, hoping "cat" survives in the summary until it is needed. The trouble is that after many words, an old detail can **fade away**.

## 🔵 2. How it works
1. The input is a sequence: $x_1, x_2, \dots, x_T$ (for text, one number-list per word; see [[Tokenization]] and [[Embedding]]).
2. The network keeps a **hidden state** $h$: a list of numbers that is its memory. It starts at zero.
3. **At each step $t$**, it takes the new input $x_t$ and the **previous memory** $h_{t-1}$, mixes them with learned weights, passes the result through an [[Activation Function]] (usually tanh), and produces the **new memory** $h_t$.
4. It may also produce an **output** $o_t$ at each step (for example "what is the next word?") or only at the end (for example "is this review positive?").
5. The **same weights are reused at every step**. That is why it can handle sequences of any length, and why a pattern learned at the start of a sentence is also recognised later.
6. Training "unrolls" the loop into a long chain, one copy per step, and runs [[Backpropagation]] along the chain (called **backpropagation through time**).
7. **The catch:** the gradient has to travel back through *every* step. Multiplied over many steps it tends to fade away or blow up ([[Vanishing and Exploding Gradients]]), so a plain RNN struggles to link things that are far apart.
8. **LSTM and GRU** fix much of this by adding **gates**: small switches that decide what to forget, what to write into memory, and what to show. The memory then has a nearly straight path through time, a bit like a [[Residual Connection]].

![[rnn-unfold.svg]]
*An RNN drawn two ways. Left: one unit with a loop, so its memory $h$ feeds back into itself. Right: the same unit "unrolled" along time into copies at steps $t-1$, $t$, $t+1$. In each copy the input $x$ enters through weights $U$, the memory passes forward through weights $V$, and the output $o$ leaves through weights $W$. The same $U$, $V$, $W$ are used at every step. Image: fdeloche, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Recurrent_neural_network_unfold.svg), CC BY-SA 4.0.*

![[lstm-cell.svg]]
*An LSTM cell (long short-term memory). The top line carries the **cell state** $c$, the long-term memory, from $c_{t-1}$ to $c_t$. Along it, a gate (the orange $\sigma$ box feeding the first "x") can scale the old memory down, and new information (a $\sigma$ gate times a tanh candidate) is added with the "+". Another gate decides how much of the memory to reveal as the output $h_t$. Image: Guillaume Chevalier, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:LSTM_Cell.svg), CC BY-SA 4.0. I read the roles of the gates off the picture and the standard description of the LSTM, not from the file text.*

## 🟠 3. Technical detail
- **What is carried:** one hidden state vector. It must hold *everything* from the past that the future might need, and it has a fixed size, so it is a lossy summary.
- **Kinds of task:**
  - **Many to one:** read a sequence, give one answer (sentiment of a review).
  - **One to many:** start from one input, produce a sequence (a caption for an image).
  - **Many to many:** translate, or label every word.
- **Training problems:**
  - Gradients that vanish over long spans, so far-apart links are not learned.
  - Gradients that explode, usually handled by **gradient clipping** (see [[Vanishing and Exploding Gradients]]).
  - Training is **sequential**: step $t$ needs step $t-1$, so it cannot be spread across all positions at once on a [[GPU]], which makes it slow on long sequences.
- **LSTM (long short-term memory):** has a separate **cell state** $c$ plus three gates: **forget** (how much old memory to keep), **input** (how much new to write) and **output** (how much to reveal). Each gate is a sigmoid between 0 and 1 that multiplies a value (see [[Activation Function]]).
- **GRU (gated recurrent unit):** a lighter relative with two gates. It often performs about as well as the LSTM, with fewer parameters (a common finding; it varies by task).
- **Bidirectional RNNs:** run one RNN forwards and one backwards over the same text, and combine them, so each position sees both its past and its future. Used when the whole sequence is available.
- **Stacking:** several recurrent layers on top of each other, often with [[Dropout]] between them, and with [[Normalization]].
- **Sequence-to-sequence and attention:** early translation systems read the whole sentence into one memory vector and decoded from it. That single vector was a bottleneck, which led to [[Attention]]: letting the decoder look back at *every* position instead of one summary.
- **Why Transformers took over:** a [[Transformer]] connects any two positions directly through attention, trains all positions in parallel, and handles long-range links far better. For most language work it replaced RNNs around 2017 to 2019 (the usual account; from memory). RNNs are still used in some small or streaming settings, where memory and speed matter. Newer recurrent-style designs also exist (research; not covered here).
- **Related ideas:** the loop is a [[Neural Network]] applied again and again, much like how a [[Convolutional Neural Network]] applies one filter at every position, but over time instead of over space.

```python
import numpy as np
U, V, b = 1.0, 0.5, 0.0                      # input weight, recurrent weight, bias (one unit)
h = 0.0
for x in (1, -1, 1):                         # a sequence of 3 inputs: the same step is reused
    h = np.tanh(U * x + V * h + b)
    print(round(h, 3))                       # 0.762, -0.551, 0.62

print(0.5 ** 20, 1.0 ** 20, 1.5 ** 20)       # 9.5367431640625e-07 1.0 3325.256730079651   factor over 20 steps if the slope were 1

def last(h0, V, T=20):                       # run T steps with zero input, return the final state
    h = h0; slopes = []
    for _ in range(T):
        h = np.tanh(V * h); slopes.append(V * (1 - h ** 2))
    return h, np.prod(slopes)
h20, prod = last(0.5, 0.5)
eps = 1e-6
fd = (last(0.5 + eps, 0.5)[0] - last(0.5 - eps, 0.5)[0]) / (2 * eps)
print(prod, fd)                              # 8.788024768939669e-07 8.788024768522727e-07   product of slopes vs a numerical check

f, i, g, c = 0.9, 0.5, 1.0, 2.0              # forget gate, input gate, candidate, old cell value
print(f * c + i * g, 0.9 ** 20)              # 2.3 0.12157665459056935
```

**For time series (added with the [[Time Series]] notes).** Recurrent networks were the first deep models widely used for forecasting. The usual setup cuts the series into windows: the last $L$ values go in, and the next value (or the next $H$ values at once) is the target. Forecasting many steps by feeding each prediction back in lets errors build up, so predicting all the steps at once is often safer. Give the network growth rates or differences, not raw levels, and scale them ([[Stationarity]]). A small [[LSTM]] was competitive in [[Forecasting Shoot-out]], and N-BEATS ([[N-BEATS 2019]]) and linear models ([[DLinear 2022]]) are later answers to "how complex does a forecaster need to be?".

## 🔴 4. Deep / formal
**One step of a plain RNN.** With input $x_t$, previous memory $h_{t-1}$, and learned weights $U$ (for the input), $V$ (for the memory) and a bias $b$:

$$
h_t=\tanh\!\big(U\,x_t+V\,h_{t-1}+b\big),\qquad o_t=W\,h_t
$$

**Where:**
- $x_t$ is the input at step $t$ and $h_t$ is the hidden state (memory) after step $t$. $h_0$ is the starting memory, usually zeros.
- $U$, $V$ and $W$ are weight matrices ([[Matrix]]); with one unit they are single numbers. $U$ maps the input in, $V$ maps the old memory in, $W$ maps the memory to the output $o_t$. $b$ is the bias. These names match the picture above.
- $\tanh$ is the activation function: it squashes any number into the range $-1$ to $1$ ([[Activation Function]]). **Matters:** it keeps the memory from growing without limit and supplies the bend that lets the network represent more than straight lines.
- **Matters:** the same $U$, $V$, $W$ and $b$ are used at **every** step. The only things that change from step to step are the input and the memory. That is how one small set of weights handles any sequence length.

**Example (tiny numbers):** one unit with $U=1$, $V=0.5$, $b=0$, starting from $h_0=0$, on the inputs $x=(1,-1,1)$.
- Step 1: $h_1=\tanh(1\cdot1+0.5\cdot0)=\tanh(1)=\mathbf{0.762}$.
- Step 2: $h_2=\tanh(1\cdot(-1)+0.5\cdot0.762)=\tanh(-0.619)=\mathbf{-0.551}$.
- Step 3: $h_3=\tanh(1\cdot1+0.5\cdot(-0.551))=\tanh(0.7245)=\mathbf{0.620}$.
- Meaning: the memory moves up, down and up again as the inputs change sign, but each value still carries a little of what came before (the $0.5\cdot h_{t-1}$ term).

**Why long memories fade: the gradient through time.** To learn from an input $T$ steps ago, the gradient has to be passed back through $T$ steps. In the one-unit case, by the chain rule ([[Calculus]]), the effect of the early memory $h_0$ on the late memory $h_T$ is a product with one factor per step:

$$
\frac{\partial h_T}{\partial h_0}=\prod_{t=1}^{T}V\,\big(1-h_t^{2}\big)
$$

**Where:**
- $h_t$ is the memory at step $t$, and $1-h_t^2$ is the slope of $\tanh$ there. It lies between $0$ and $1$, and equals $1$ only when $h_t=0$.
- $V$ is the recurrent weight. So each step contributes the factor $V(1-h_t^2)$.
- $\prod_{t=1}^T$ means "multiply these together for all $T$ steps". $\partial h_T/\partial h_0$ is how much a small change in the old memory changes the final memory.
- **Matters:** this is the same exponential trap as in [[Vanishing and Exploding Gradients]], now over *time*. If every factor is smaller than 1 in size, the product shrinks exponentially with $T$ and the network cannot learn long-range links. If $|V|$ is large the factors can exceed 1 and the product can blow up. The $\tanh$ slope is at most 1, so $|V|<1$ guarantees shrinking, but $|V|>1$ does *not* guarantee growth, because the slope can pull the factor back below 1.
- With vectors, the factors are matrices, and the size of their largest stretch factor plays the role of $V$. The one-unit picture keeps the main idea.

**Example (tiny numbers):** a check with $T=20$ steps, $V=0.5$, zero input, starting at $h_0=0.5$.
- The product of the 20 per-step slopes $V(1-h_t^2)$ is $\mathbf{8.79\times10^{-7}}$. A numerical check (nudge $h_0$ slightly and watch $h_{20}$) gives $\mathbf{8.79\times10^{-7}}$ too, which confirms the formula.
- If the slope were exactly 1 at every step, the factor would be $0.5^{20}=\mathbf{9.5\times10^{-7}}$ for $V=0.5$ and $1.5^{20}=\mathbf{3325}$ for $V=1.5$.
- Meaning: after 20 steps, a change to the early memory has about a millionth of the effect on the final state, so the gradient carries almost no signal about what happened 20 steps ago.

**The LSTM fix: an additive memory.** The LSTM adds a separate cell state $c_t$ that is updated by scaling the old value and **adding** new information:

$$
c_t=f_t\odot c_{t-1}+i_t\odot g_t,\qquad h_t=o_t\odot\tanh(c_t)
$$

**Where:**
- $c_t$ is the **cell state** (long-term memory) and $h_t$ is the output state (what the cell shows). $\odot$ means element-by-element multiplication.
- $f_t$ is the **forget gate**, $i_t$ the **input gate** and $o_t$ the **output gate**. Each is a [[Activation Function]] sigmoid of the current input and previous $h$, so each entry lies between 0 and 1. They multiply values and act like switches: 0 = closed, 1 = open.
- $g_t$ is the **candidate**: new content proposed for the memory (a tanh of the input and previous $h$).
- **Matters:** the update from $c_{t-1}$ to $c_t$ is "scale by $f_t$, then add". The gradient along this path is simply $f_t$, with no weight matrix and no activation slope in the way. If the forget gate stays near 1, the memory and its gradient pass through many steps almost untouched. This is the same idea as the shortcut in a [[Residual Connection]].
- **Matters:** the gates are learned, so the network decides for itself when to remember and when to forget.
- The remaining equations (how the gates and $g_t$ are computed from $x_t$ and $h_{t-1}$) are in the original paper and in the picture; they are not written out here.

**Example (tiny numbers):** old memory $c_{t-1}=2$, forget gate $f=0.9$, input gate $i=0.5$ and candidate $g=1$.
- $c_t=0.9\cdot2+0.5\cdot1=1.8+0.5=\mathbf{2.3}$.
- Gradient factor from $c_{t-1}$ to $c_t$: $f=\mathbf{0.9}$. Over 20 steps with $f=0.9$ at each: $0.9^{20}=\mathbf{0.122}$, versus $0.5^{20}\approx10^{-6}$ for the plain RNN above.
- Meaning: with the forget gate close to 1, about an eighth of the gradient survives 20 steps, where the plain RNN kept about a millionth. This is a toy comparison with made-up gate values; real gates change with the input.

**Further notes:**
- **Truncated backpropagation through time:** for very long sequences, gradients are only passed back a fixed number of steps (for example 100), trading accuracy for memory.
- **Teacher forcing:** when training to generate sequences, the model is often fed the *true* previous output instead of its own guess. This speeds learning, but it can leave the model unprepared for its own mistakes at generation time (a known issue; from memory).
- **A language model view:** an RNN can be trained as a language model, predicting the next token from the memory (see [[Self-Supervised Learning]] and [[Perplexity]]). Modern [[LLM]]s do the same job with Transformers.
- **Capacity limits:** the memory is a fixed-size vector, so very long or detailed histories cannot be stored exactly. Attention keeps all positions available instead.
- **Not covered yet in this vault:** GRU in detail, encoder-decoder models, and newer recurrent-style architectures.

**What came before, and how this improved on it:**
- **Before:** a feed-forward model that reads a fixed window of the last $w$ words (or an n-gram table).
- **Its limit:** it cannot see further back than $w$, and its input weights grow with the window. In the code below, with $d=100$ and 256 hidden units, a window of 2 needs 51,200 input weights and a window of 50 needs 1,280,000.
- **What changed:** reuse the same weights at every step and carry a memory vector forward, so the sentence can be any length. LSTM (Hochreiter and Schmidhuber, 1997) and GRU (Cho et al., 2014) eased the fading-memory problem (dates from memory, not checked).
- **How much better:** the RNN uses 91,136 weights (25,600 input + 65,536 recurrent) for a 3-word or a 3,000-word sentence, a size independent of length. It still processes words one at a time, which [[Transformer]] later removed.

**Check it yourself:**

```python
V, d, h = 10_000, 100, 256
for w in (2, 5, 10, 50):
    print(f"window model, w={w:2d}: input weights {w*d*h:,}")
print(f"RNN, any length: input {d*h:,} + recurrent {h*h:,} = {d*h + h*h:,} (same for a 3-word or 3000-word sentence)")
```

## Related
- [[Neural Network]] · [[Vanishing and Exploding Gradients]] · [[Backpropagation]] · [[Activation Function]] · [[Residual Connection]] · [[Attention]] · [[Transformer]] · [[LLM]] · [[Tokenization]] · [[Embedding]] · [[Normalization]] · [[Dropout]] · [[Convolutional Neural Network]] · [[Self-Supervised Learning]] · [[Perplexity]] · [[Matrix]] · [[Calculus]] · [[GPU]] · [[Eigenvalue]] · [[Self-Attention]] · [[BERT]] · [[From FNN to RNN]] · [[LSTM]] · [[Time Series]] · [[Forecasting Shoot-out]]

## Sources
- Hochreiter & Schmidhuber, *Long Short-Term Memory*, 1997 (from memory)
- Cho et al., *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*, 2014 (GRU; from memory)
- Bahdanau, Cho & Bengio, *Neural Machine Translation by Jointly Learning to Align and Translate*, 2014 (attention; from memory)
- Pascanu, Mikolov & Bengio, *On the difficulty of training recurrent neural networks*, 2013 (from memory)
- Vaswani et al., *Attention Is All You Need*, 2017
- Goodfellow, Bengio & Courville, *Deep Learning*, 2016, chapter 10
- Olah, *Understanding LSTM Networks* (blog post; from memory)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/generative Recurrent neural network (RNN):::A network that reads a sequence one piece at a time and keeps a little memory of what it has read so far. At each step it mixes the new piece with its memory, updates the memory, and moves on.

#flashcards/ml/formula #flashcards/topic/generative Write the RNN step and explain each symbol.
?
$$h_t=\tanh\!\big(U\,x_t+V\,h_{t-1}+b\big),\qquad o_t=W\,h_t$$
- $x_t$: the input at step $t$. $h_t$: the hidden state (memory) after step $t$. $h_0$: the starting memory, usually zeros.
- $U$, $V$, $W$: weight matrices. $U$ maps the input in, $V$ maps the old memory in, $W$ maps the memory to the output $o_t$. The same weights are used at every step.
- $b$: a bias. $\tanh$: squashes the memory to between −1 and 1.

#flashcards/ml/formula #flashcards/topic/generative Write the gradient through time for an RNN and explain each symbol.
?
$$\frac{\partial h_T}{\partial h_0}=\prod_{t=1}^{T}V\,\big(1-h_t^{2}\big)$$
- $h_t$: the memory at step $t$. $1-h_t^2$: the slope of $\tanh$ there. It lies between 0 and 1, and is 1 only when $h_t=0$.
- $V$: the recurrent weight. Each step contributes the factor $V(1-h_t^2)$.
- $\prod_{t=1}^T$: multiply them for all $T$ steps. With factors below 1 the product shrinks exponentially, so early steps barely influence the end. This is the vanishing gradient.

#flashcards/ml/formula #flashcards/topic/generative Write the LSTM cell update and explain each symbol.
?
$$c_t=f_t\odot c_{t-1}+i_t\odot g_t,\qquad h_t=o_t\odot\tanh(c_t)$$
- $c_t$: the cell state (long-term memory). $h_t$: the output state. $\odot$: multiply entry by entry.
- $f_t$: the forget gate, $i_t$: the input gate, $o_t$: the output gate. Each is a sigmoid of the current input and previous output, so it is between 0 and 1.
- $g_t$: the candidate new content (a tanh).
- The memory is edited by a gate-scaled copy plus gated new content, so the gradient factor is the forget gate, which can stay near 1.

#flashcards/ml/cloze #flashcards/topic/generative An RNN reuses the ==same weights== at every step, and its gradient is a ==product== of per-step factors.

#flashcards/ml/example #flashcards/topic/generative One unit with $U=1$, $V=0.5$, $b=0$, $h_0=0$ and inputs $(1,-1,1)$. What are $h_1$, $h_2$, $h_3$?::h₁ = tanh(1) = 0.762. h₂ = tanh(−1 + 0.5 × 0.762) = tanh(−0.619) = −0.551. h₃ = tanh(1 + 0.5 × (−0.551)) = tanh(0.7245) = 0.620.

#flashcards/ml/example #flashcards/topic/generative For an LSTM, $c_{t-1}=2$, $f=0.9$, $i=0.5$, $g=1$. What is $c_t$, and what is the gradient factor over 20 steps compared with the plain RNN?::c_t = 0.9 × 2 + 0.5 × 1 = 2.3. The gradient factor per step is f = 0.9, and 0.9²⁰ = 0.122, against 0.5²⁰ ≈ 10⁻⁶ for the plain RNN example.

#flashcards/ml/image #flashcards/topic/generative ![[rnn-unfold.svg]]
What does this picture show?
?
An RNN drawn two ways. Left: one unit with a loop, so its memory h feeds back into itself. Right: the same unit unrolled along time into copies at steps t−1, t and t+1. In each copy the input x enters through weights U, the memory passes forward through weights V and the output o leaves through weights W. The same U, V, W are used at every step.

#flashcards/ml/image #flashcards/topic/generative ![[lstm-cell.svg]]
What does this picture show?
?
An LSTM cell. The top line carries the cell state c (the long-term memory) from c(t−1) to c(t). Along it, a gate (the sigma box feeding the first "x") can scale the old memory down, and new information (a sigma gate times a tanh candidate) is added with the "+". Another gate decides how much of the memory to reveal as the output h(t).

#flashcards/ml/concept #flashcards/topic/generative How is a recurrent network set up to forecast a time series?
?
Cut the series into windows: the last L values go in and the next value (or the next H values at once) is the target. Feeding predictions back in lets errors build up, so predicting all steps at once is often safer. Give it growth rates or differences, scaled, not raw levels.
