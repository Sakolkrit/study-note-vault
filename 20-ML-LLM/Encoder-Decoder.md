---
tags: [encoder-decoder, seq2seq, RNN, LSTM, translation, transformer, deep-learning, ML]
status: growing
level: 4
tier: 2
---

# Encoder-Decoder

An **encoder-decoder** model has two halves. The **encoder** reads an input (a sentence, an image) and squeezes it into numbers. The **decoder** reads those numbers and writes an output, one piece at a time. The output can be a different length from the input, which is why it suits translation. The first big neural version for translation is [[Seq2Seq 2014]]; the [[Transformer]] later kept the two-half layout.

## 🟢 1. Simple (age 12)
One friend reads a long story and whispers a short summary to a second friend. The second friend, who never saw the story, writes the story's translation from the whisper alone.

**Example:** you read a French sentence, tell your friend "a cat sleeps on a red sofa" in your own head-notes, and your friend writes it in English.

## 🔵 2. How it works
1. **The encoder reads** the input one piece (token) at a time. In [[Seq2Seq 2014]] it is a stack of [[LSTM]] layers, a kind of [[Recurrent Neural Network]].
2. **It keeps a memory** that changes with each word. After the last word, the memory is the whole sentence in numbers: the **thought vector**.
3. **The decoder starts from that memory.** It first writes a start signal, then guesses the first word.
4. **Each guessed word is fed back in** so the decoder can guess the next one ([[Autoregressive Generation]]).
5. **It stops** when it writes a special end word (`<EOS>`).
6. **Because the decoder writes its own words,** the output can be shorter or longer than the input.

![[encoder-decoder-rnn.jpg]]
*An encoder-decoder recurrent network: the encoder (left) reads $x_1$ to $x_{T_1}$ and ends in the final hidden state $h_{T_1}$; the decoder (right) produces $y_1$ to $y_{T_2}$. Figure: Babayaga94, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:EncoderDecoder.pdf), CC BY-SA 4.0 (the same image as in [[Seq2Seq 2014]], which lists how it differs from that paper's model: here the state feeds every decoder step).*

## 🟠 3. Technical detail
All numbers come from the code below (Python 3.9.6, PyTorch 2.8.0, CPU, a tiny untrained model: vocabulary 50, embeddings 16, hidden 32, 2 layers, so the generated words are meaningless).

- **A fixed-size summary:** a 7-token source and a 20-token source both give the same state shapes, `(2, 1, 32)` for the hidden state and the same for the memory cell, 128 numbers in all. The summary size does not depend on the input length.
- **Different output length:** the decoder generated 4 tokens for the 7-token source.
- **Parameters:** encoder 14,848, decoder 14,848, embedding 800, output layer 1,650, total 32,146.
- **Scale in practice (from my [[Seq2Seq 2014]] note):** four layers of 1,000 cells, an 8,000-number summary and 384 million parameters.
- **The bottleneck:** everything the decoder knows about the source passes through that one fixed-size summary. Attention ([[Attention]]) was invented to let the decoder look back at all encoder states instead.
- **Limits of my run:** untrained, no training loop, no beam search, single sentence.

```python
import torch, torch.nn as nn
torch.manual_seed(0)
V, E, H, L = 50, 16, 32, 2
enc = nn.LSTM(E, H, num_layers=L, batch_first=True); dec = nn.LSTM(E, H, num_layers=L, batch_first=True)
emb = nn.Embedding(V, E); out = nn.Linear(H, V)
src = torch.randint(0, V, (1, 7))                  # source sentence: 7 tokens
_, (h, c) = enc(emb(src))                           # v = final hidden and cell states of every layer
print('source length 7 -> state shapes', tuple(h.shape), tuple(c.shape), '| numbers in v:', h.numel() + c.numel())
tok = torch.zeros(1, 1, dtype=torch.long); state = (h, c); gen = []
for t in range(4):                                  # decode 4 tokens: output length differs from input
    o, state = dec(emb(tok), state); tok = out(o).argmax(-1); gen.append(tok.item())
print('generated', len(gen), 'tokens (source had 7):', gen)
src2 = torch.randint(0, V, (1, 20)); _, (h2, c2) = enc(emb(src2)); print('20-token source -> same state shape', tuple(h2.shape))
n = lambda m: sum(p.numel() for p in m.parameters())
print('params: encoder', n(enc), 'decoder', n(dec), 'embedding', n(emb), 'output layer', n(out), 'total', n(enc) + n(dec) + n(emb) + n(out))
```

## 🔴 4. Deep / formal
**The probability of an output sequence** (the paper's equation 1 in [[Seq2Seq 2014]]):

$$
p(y_1,\dots,y_{T'}\mid x_1,\dots,x_T)=\prod_{t=1}^{T'}p\big(y_t\mid v,\,y_1,\dots,y_{t-1}\big)
$$

**Where:**
- $x_1,\dots,x_T$ is the input (length $T$) and $y_1,\dots,y_{T'}$ the output (length $T'$, which may differ).
- $v$ is the encoder's summary of the input. **Matters:** it is the only link from input to output in the basic model.
- $p(y_t\mid v,y_{<t})$ is the chance of the next word given the summary and the words so far. **Matters:** it is what the decoder's softmax outputs.
- $\prod$ multiplies the step chances. **Matters:** one very unlikely word lowers the whole sequence.

**Example (tiny numbers):** step chances 0.5, 0.4, 0.9: $0.5\times0.4\times0.9=0.18$. In logs: $-0.693-0.916-0.105=-1.715$. The answer means: the model gives this whole output an 18% chance.

**Trade-offs and limits:**
- The fixed summary is a bottleneck for long inputs (attention was the fix, named in the Seq2Seq paper).
- Decoding is step by step, so it cannot be done in parallel at test time (general knowledge).
- The same layout is used outside text: image encoders with text decoders (general knowledge; not tested here).

**What came before, and how this improved on it:**
- **Before:** deep networks needing fixed-size inputs and outputs, and phrase-based statistical translation with many hand-built parts (from [[Seq2Seq 2014]], paper sections 1 and 3).
- **Its limit:** they could not map a sequence to a sequence of a different length with one learned model.
- **What changed:** one network reads into a vector, another writes from it.
- **How much better:** published in [[Seq2Seq 2014]]: 34.81 BLEU for an ensemble of five LSTM encoder-decoders against 33.30 for a phrase-based system on English to French (paper's Table 1; see [[BLEU Score]]).

## Related
[[Seq2Seq 2014]] · [[Recurrent Neural Network]] · [[LSTM]] · [[Attention]] · [[Transformer]] · [[Autoregressive Generation]] · [[Beam Search]] · [[BLEU Score]] · [[Embedding]] · [[Softmax]]

## Sources
- Sutskever, Vinyals, Le (2014), arXiv 1409.3215, read as text; see my note [[Seq2Seq 2014]].
- Image: [EncoderDecoder.pdf](https://commons.wikimedia.org/wiki/File:EncoderDecoder.pdf) (Babayaga94, CC BY-SA 4.0), Wikimedia Commons.
- My own run (Python 3.9.6, PyTorch 2.8.0). Not checked: the note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/transformers-nlp Encoder-decoder:::A model in two halves: the encoder squeezes the input into numbers, and the decoder writes the output one piece at a time from those numbers; the output can have a different length.

#flashcards/ml/concept #flashcards/topic/transformers-nlp What are the steps of an encoder-decoder translating a sentence?::The encoder reads the words one at a time and ends with a summary; the decoder starts from the summary, guesses the first word, feeds each guess back in, and stops at the end word.

#flashcards/ml/concept #flashcards/topic/transformers-nlp Why can an encoder-decoder produce an output of a different length from its input?::The decoder writes its own words until it writes the end word. In my tiny run a 7-token source produced 4 tokens.

#flashcards/ml/concept #flashcards/topic/transformers-nlp How big is the summary an encoder passes on, and why is that a problem?::A fixed size whatever the input length: my tiny model gave 128 numbers for both 7 and 20 tokens; Seq2Seq used 8,000. Everything the decoder knows passes through it, a bottleneck that attention was invented to remove.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write the probability of an output sequence in an encoder-decoder and explain each symbol.
?
$$p(y_1,\dots,y_{T'}\mid x_1,\dots,x_T)=\prod_{t=1}^{T'}p\big(y_t\mid v,\,y_1,\dots,y_{t-1}\big)$$
- $x_1..x_T$: the input; $y_1..y_{T'}$: the output, whose length may differ.
- $v$: the encoder's summary, the only link from input to output in the basic model.
- $p(y_t\mid v,y_{<t})$: the chance of the next word, from the decoder's softmax.
- $\prod$: multiplies the steps, so one unlikely word lowers the whole sequence.

#flashcards/ml/example #flashcards/topic/transformers-nlp Step chances 0.5, 0.4 and 0.9: what is the output sequence's probability?::$0.5\times0.4\times0.9=0.18$, or $-1.715$ in logs (my arithmetic).

#flashcards/ml/concept #flashcards/topic/transformers-nlp Why must each decoded word be fed back into the decoder?::The next word's chance depends on the words so far ($y_{<t}$ in the product), so the decoder needs its own earlier output as input. This also makes decoding sequential.

#flashcards/ml/concept #flashcards/topic/transformers-nlp How well did the first neural encoder-decoder do at translation?::Seq2Seq's ensemble of five LSTM encoder-decoders scored 34.81 BLEU on English to French against 33.30 for a phrase-based system (paper's Table 1).

#flashcards/ml/image #flashcards/topic/transformers-nlp ![[encoder-decoder-rnn.jpg]]
What does this encoder-decoder picture show?
?
An encoder RNN reads $x_1$ to $x_{T_1}$ and ends in $h_{T_1}$; a decoder RNN produces $y_1$ to $y_{T_2}$. Free redrawing by Babayaga94, CC BY-SA 4.0, in which the final state feeds every decoder step.

#flashcards/ml/cloze #flashcards/topic/transformers-nlp In an encoder-decoder, the ==encoder== reads the input into a summary and the ==decoder== writes the output from it.
