---
tags: [history, neural-network, RNN, LSTM, sequence, attention]
status: growing
level: 4
tier: 2
---

# From FNN to RNN

How the plain feed-forward network ([[Neural Network]], FNN) was stretched to handle things that come in order, and what each stretch fixed.

## 🟢 1. Simple (age 12)
A plain network reads **one fixed-size card** and gives an answer. It cannot read a story, because stories have different lengths and the order matters. The first fix: read a **small window** of the last few words. The better fix: a network that keeps a **note in its pocket** and updates it after each word. That is a recurrent network. Its pocket note fades over long stories, so people added better notes (LSTM), and finally let the network flip back through all the pages (attention).

**Example:** Reading "The keys that I left on the kitchen table yesterday ___ missing". To pick "are", you must remember "keys" from far back. A two-word window cannot; a note in the pocket might; flipping back to "keys" surely can.

## 🔵 2. How it works
1. **FNN.** Fixed-size input, fixed-size output, no memory. Great for tables and pictures of a set size.
2. **Window FNN.** Slide a window of the last $w$ words over the text. Limit: cannot see beyond $w$, and the input weights grow with $w$.
3. **RNN** ([[Recurrent Neural Network]]). Use the same weights at every step, and pass a memory vector forward. Any length works with the same number of weights.
4. **The fading problem.** Training sends the learning signal back through every step, and it shrinks (or grows) at each one ([[Vanishing and Exploding Gradients]]). So early words barely teach the network.
5. **LSTM and GRU.** Gates decide what the memory keeps and forgets, so the signal can travel further (LSTM 1997; GRU from memory 2014).
6. **Seq2Seq** ([[Seq2Seq 2014]]). One RNN reads, another writes. Limit: the whole input in a single vector.
7. **Attention** ([[Attention]]). The writer can look at every input position.
8. **Transformer** ([[Attention Is All You Need 2017]]). Drop the recurrence; attention does everything and trains in parallel.

## 🟠 3. Technical detail
- **Weights, counted:** with 16 input numbers per word and 32 hidden units, a window FNN needs 1,024 input weights for window 2, 5,120 for window 10 and 25,600 for window 50. The RNN uses 1,536 (512 input + 1,024 recurrent) for any length.
- **Fading, measured:** the code feeds random text of length $T$ into an untrained RNN and an untrained LSTM, and measures how large the gradient from the last output back to the first input is.
- **What the measurement says:** both fade quickly: after 20 steps the RNN gradient is $4.7\times10^{-6}$ and the LSTM's $2.4\times10^{-5}$; after 50 steps $1.4\times10^{-14}$ against $1.8\times10^{-11}$ (LSTM about 1,000 times larger); after 100 steps the RNN's is exactly 0 (below the float limit) and the LSTM's $1.3\times10^{-21}$. At length 5 the plain RNN is the larger one. So this **untrained** test shows the LSTM fades more slowly at long range, not that it solves the problem. Gates are learned during training, which this test skips.
- **Why not stop at the LSTM:** it still reads one word at a time (slow to train on GPUs), and its memory is one fixed-size vector, so very long or exact recall is hard ([[Transformer]]).
- **Order of sources:** Seq2Seq's abstract notes that reversing the source sentence improved results markedly because it created short-term dependencies. This is a hint of how hard the long links were.

```python
import torch
torch.manual_seed(0)
d, h = 16, 32
print("how much of the first input's influence reaches the last output (gradient size):")
for T in (5, 20, 50, 100):
    row = []
    for name, cell in (("RNN", torch.nn.RNN(d, h, batch_first=True)), ("LSTM", torch.nn.LSTM(d, h, batch_first=True))):
        x = torch.randn(1, T, d, requires_grad=True)
        out, _ = cell(x)
        out[0, -1].sum().backward()
        row.append(f"{name} {x.grad[0, 0].norm().item():.2e}")
    print(f"  length {T:3d}:  " + "   ".join(row))
V = 10_000
for w in (2, 10, 50):
    print(f"fixed-window FNN, window {w:2d}: input weights {w*d*h:,}")
print(f"RNN: {d*h + h*h:,} weights for any length (input {d*h:,} + recurrent {h*h:,})")
```

## 🔴 4. Deep / formal
This note has no equation (the formulas live in [[Recurrent Neural Network]] and [[Vanishing and Exploding Gradients]]). Trade-offs:
- **Memory against parallelism.** Recurrence gives a sentence-long memory with constant weights, but forces step-by-step work. Attention gives direct access but costs more arithmetic as text grows (see [[Self-Attention]]).
- **A fixed-size memory is a lossy summary.** Past some length, the network cannot keep everything. My own toy in [[Attention]] shows a 32-number summary identifying the right word 98% of the time among 4 words and 58% among 128.
- **Revival:** recurrent-style models are being studied again for very long text (named only; I did not read them).
- **Not covered:** bidirectional RNNs, teacher forcing details, training RNNs with truncation, and state-space models.

**What came before, and how this improved on it:**
- **Before:** the feed-forward network with a fixed input size, or a fixed window.
- **Its limit:** no memory beyond the window, and weights that grow with it (25,600 input weights at window 50 in the code).
- **What changed:** reuse weights across steps and carry a memory, then gate the memory (LSTM), then allow lookups across all positions (attention).
- **How much better:** the RNN needs 1,536 weights for any length. The measured fading shows the problem is only reduced: at length 50 the LSTM gradient is about 1,000 times larger than the RNN's but still $1.8\times10^{-11}$.

## Related
- [[Neural Network]] · [[Recurrent Neural Network]] · [[Vanishing and Exploding Gradients]] · [[Backpropagation]] · [[Attention]] · [[Transformer]] · [[Seq2Seq 2014]] · [[History of AI]] · [[History of Text Analytics]] · [[LSTM]]

## Sources
- arXiv abstracts read: 1409.3215 (Seq2Seq), 1409.0473 (attention), 1706.03762 (Transformer)
- Wikipedia *Timeline of artificial intelligence* for the LSTM date (1997)
- Hochreiter and Schmidhuber 1997 and Cho et al. 2014 (GRU) named from memory, not opened

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/history FNN:::A plain network. Fixed-size input, fixed-size output, no memory. Great for tables and pictures of a set size.

#flashcards/ml/concept #flashcards/topic/history Why can a plain network not read a story?::Stories have different lengths and the order matters, but a plain network reads one fixed-size card.

#flashcards/ml/reverse #flashcards/topic/history Window FNN:::Slide a window of the last $w$ words over the text. It cannot see beyond $w$, and the input weights grow with $w$ (25,600 input weights at window 50 in the note's code).

#flashcards/ml/reverse #flashcards/topic/history RNN:::Use the same weights at every step and pass a memory vector forward. Any text length works with the same number of weights.

#flashcards/ml/concept #flashcards/topic/history Why do early words barely teach a plain RNN?::Training sends the learning signal back through every step, and it shrinks (or grows) at each one, so the signal from early words fades (vanishing and exploding gradients).

#flashcards/ml/concept #flashcards/topic/history How do LSTM and GRU fix the fading signal?::Gates decide what the memory keeps and forgets, so the signal can travel further. The note gives LSTM 1997 (from a timeline source) and GRU 2014 (from memory, not opened).

#flashcards/ml/reverse #flashcards/topic/history Seq2seq:::One RNN reads and another writes. Its limit is that the whole input is squeezed into a single vector.

#flashcards/ml/cloze #flashcards/topic/history Attention fixes the seq2seq limit: the writer can ==look at every input position==.

#flashcards/ml/reverse #flashcards/topic/history Transformer:::Drop the recurrence. Attention does everything and trains in parallel.

#flashcards/ml/concept #flashcards/topic/history Why not stop at the LSTM?::It still reads one word at a time (slow to train on GPUs), and its memory is one fixed-size vector, so very long or exact recall is hard.

#flashcards/ml/concept #flashcards/topic/history Memory against parallelism in sequence models?::Recurrence gives a sentence-long memory with constant weights but forces step-by-step work. Attention gives direct access but costs more arithmetic as text grows.
