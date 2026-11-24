---
tags: [paper, NLP, RNN, LSTM, translation, seq2seq, encoder-decoder]
link: https://arxiv.org/abs/1409.3215
authors: Sutskever, Vinyals, Le (Google; from the paper's first page)
year: 2014
status: body-read
field: text / translation
---

# Seq2Seq 2014

> **In one line for a 12-year-old:** One network reads a whole sentence, one word at a time, and squeezes it into a list of numbers. A second network reads that list and writes the translation, one word at a time. A strange trick, reading the sentence backwards, made it work much better.

Paper: "Sequence to Sequence Learning with Neural Networks", Ilya Sutskever, Oriol Vinyals and Quoc V. Le (Google), arXiv 1409.3215, first posted 10 September 2014 and last revised 14 December 2014 (v3, the version read here; arXiv page). The paper's venue is not on the arXiv page or the PDF, so I leave it out. Concept notes: [[LSTM]], [[Recurrent Neural Network]], [[Beam Search]], [[BLEU Score]]. Reference note for this layout: [[Attention Is All You Need 2017]].

## The problem
Deep networks need inputs and outputs of a fixed size. Many problems, such as translation, speech recognition and question answering, have inputs and outputs that are sequences whose length is not known in advance, and whose lengths can differ (paper, 1). The paper wants one general method that maps any sequence to any other sequence, with "minimal assumptions on the sequence structure" (abstract).

## What people did before
- **Phrase-based statistical machine translation (SMT):** the paper's baseline, 33.30 BLEU on the test set (paper, Table 1). It is a system of many hand-built parts, not one network.
- **Neural networks used as helpers.** The simplest and most reliable use was *rescoring* the n-best list of a strong SMT system with a neural language model (paper, 4). Others fed source information into a neural language model inside the SMT decoder (Devlin and others).
- **Closest ancestors:** Kalchbrenner and Blunsom were the first to map a whole sentence to a vector and back, using convolutional networks that lose word order. Cho and others used an LSTM-like network to map a sentence to a vector and back, mainly to rescore SMT output (paper, 1 and 4).
- **Attention:** Graves' attention and Bahdanau and others' translation model with attention are cited as related work (paper, 1 and 4). Bahdanau's result is in the paper's Table 1: 28.45 BLEU.
- **Plain recurrent networks** were thought too hard to train for this, because the input and its matching output are far apart in time ("long term dependencies", paper, 2).

## The idea
Read a sentence into one **thought vector**, then write the translation from it. The first LSTM reads the source words one at a time; its last hidden state is the thought vector $v$. A second LSTM starts from $v$ and writes the target sentence one word at a time until it writes a special end-of-sentence word `<EOS>` (paper, 2). Two tricks make it work: the LSTM is four layers deep, and the **source sentence is read in reverse order** (not the target). The reversal "introduced many short term dependencies" between source and target words and made learning easier (paper, 3.3).

## The architecture, part by part
The model is **two four-layer LSTMs**: an *encoder* that reads and a *decoder* that writes (paper, 2 and 3.4). Each layer has 1,000 cells; words are 1,000-number [[Embedding|embeddings]]; the source vocabulary has 160,000 words and the target vocabulary 80,000. The sentence is represented by 8,000 real numbers (paper, 3.4); the whole network has 384 million parameters, of which 64 million are the recurrent connections (32 million for each LSTM). The figure below shows the *idea* (an encoder, a hand-over, a decoder), not the paper's exact drawing.

![[encoder-decoder-rnn.jpg]]
*An encoder-decoder recurrent network: the encoder (left) reads $x_1$ to $x_{T_1}$ and ends in the final hidden state $h_{T_1}$; the decoder (right) produces $y_1$ to $y_{T_2}$. Figure: Babayaga94, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:EncoderDecoder.pdf), CC BY-SA 4.0; I used Commons' 960-pixel picture of the PDF. It is a free redrawing of the idea, not the paper's Figure 1. Differences: (1) here the encoder's final state is fed into the decoder at every step (the arrows on top), while in the paper it only sets the decoder's starting state; (2) the paper's model has four stacked layers, an `<EOS>` word, separate encoder and decoder LSTMs and a reversed input, none of which this picture shows; (3) the paper's Figure 1 draws one row of boxes reading `A B C <EOS>` and then writing `W X Y Z <EOS>`. I looked at the paper's Figure 1 on its page 2 and the drawing here is only similar in spirit.*

### The encoder
- **Input embedding (160,000 words by 1,000 numbers)**
  - *What it does:* turns each source word into a vector of 1,000 numbers; unknown words become one special `UNK` word (paper, 3.1 and 3.4).
  - *Why it matters:* the network only works with numbers. The paper does not say why 160,000 words (source) against 80,000 (target) or why 1,000; these are chosen sizes (general knowledge: a bigger vocabulary costs more memory).
- **Four stacked [[LSTM]] layers, 1,000 cells each**
  - *What it does:* reads the reversed source sentence one word at a time. Each layer passes its output up to the next layer, and each layer keeps its own memory.
  - *Why it matters:* the paper found that "deep LSTMs significantly outperformed shallow LSTMs", and "each additional layer reduced perplexity by nearly 10%", "possibly due to their much larger hidden state" (paper, 2 and 3.4). It tried up to four layers; the paper gives no table and no reason for stopping at four.
- **The thought vector $v$ (8,000 numbers)**
  - *What it does:* the last hidden state of the encoder, one fixed-size list of numbers for any sentence length. My arithmetic: 4 layers times 1,000 cells times 2 (each cell holds a hidden value and a memory value) is 8,000, which matches the paper's figure (the paper does not itemise it).
  - *Why it matters:* it is the only link from the source to the target. The paper notes it learns "sensible phrase and sentence representations" that are sensitive to word order (abstract).

### The decoder
- **A second, separate LSTM (four layers)**
  - *What it does:* its starting hidden state is set to $v$; then at each step it takes the previous target word and predicts the next one (paper, 2).
  - *Why it matters:* the paper used two LSTMs because doing so adds parameters at negligible computational cost and makes it natural to train on several language pairs at once (paper, 2). It does not test one LSTM against two.
- **Softmax over the 80,000 target words**
  - *What it does:* at every step, a [[Softmax]] gives a probability for each of the 80,000 words (paper, 2 and 3.4: "a naive softmax").
  - *Why it matters:* it turns the LSTM state into a choice of the next word. It is also expensive: the paper splits it over four GPUs, each multiplying by a $1000\times20000$ matrix (my check: 80,000 divided by 4 is 20,000).
- **The `<EOS>` word**
  - *What it does:* every sentence ends with `<EOS>`; the model stops when it writes it (paper, 2).
  - *Why it matters:* it lets the model define probabilities for sequences of *every* length (paper, 2).

### The size check (paper's 3.4, my arithmetic)

| Part | Rule | Numbers (millions) |
|---|---|---|
| Input embeddings | 160,000 words by 1,000 | 160 |
| Output embeddings | 80,000 words by 1,000 | 80 |
| Output softmax weights | 1,000 by 80,000 | 80 |
| Encoder LSTM, 4 layers | 4 layers by (4 gates by (1,000 plus 1,000) by 1,000) | 32 |
| Decoder LSTM, 4 layers | same | 32 |
| **Total** | | **384** |

Written by me, an AI, from the paper's sizes in section 3.4; the paper itemises none of it except the 384 million total and the 64 million recurrent part, so the split of the other 320 million is my reconstruction (biases ignored). It adds up exactly to the paper's 384 million.
- **Part:** the pieces of the network.
- **Rule:** how I counted it. An LSTM layer has four gates and each gate looks at the layer's input (1,000 numbers) and its own previous output (1,000 numbers), each connected to 1,000 cells (general knowledge about LSTMs).
- **Numbers (millions):** how many learned numbers (weights) that part holds. It matters because more than 80% of the model is embeddings and the output layer, not the recurrent part (my arithmetic: 320 of 384).

**Summary: what the table shows.** Most of the weights (240 million) are in the word tables and 80 million in the final softmax; only 64 million do the "remembering". **Caution:** the split is my reconstruction, and only the totals come from the paper.

**Training details (paper, 3.4 and 3.5).** All parameters start uniform between -0.08 and 0.08; plain SGD with no momentum and learning rate 0.7; after 5 epochs the rate is halved every half epoch; 7.5 epochs in total; batches of 128 sentences; gradient clipped at norm 5; sentences in a batch have about the same length, giving a 2 times speed-up. Training took about ten days on 8 GPUs, at 6,300 words per second against 1,700 words per second on one GPU (3.7 times faster, my arithmetic). Data: 12 million sentence pairs, 348 million French words and 304 million English words.

## The core mechanism, step by step

### Reading and writing a sentence
The idea: pass the source words through the encoder, keep the last state, and let the decoder unroll from it.

1. **Reverse the source words and add `<EOS>`.** *What it does:* "a b c" becomes "c b a `<EOS>`" (paper, 3.3 and Figure 1). *Why it matters:* see the next sub-section.
2. **Encode.** *What it does:* feed the words one at a time; at each step every layer updates its state with the equation of the [[Recurrent Neural Network]] (in an LSTM version). *Why it matters:* the last state $v$ has now "seen" the whole sentence.
3. **Hand over.** *What it does:* set the decoder's starting state to $v$ (paper, 2). *Why it matters:* it is how the target "knows" the source.
4. **Decode one word.** *What it does:* give the decoder the previous target word (at the start, `<EOS>`), get probabilities over 80,000 words, pick the next word. *Why it matters:* this is the product of conditional probabilities in the equations below.
5. **Repeat until `<EOS>`.** *What it does:* each chosen word is fed back in. *Why it matters:* the sentence ends by itself, whatever the length.

**Worked example (my own tiny numbers, scalar network).** One number per layer, $W^{hx}=0.5$, $W^{hh}=0.8$, starting state $h_0=0$ and input $x_1=1$, $x_2=0$. Step 1: $h_1=\sigma(0.5\cdot1+0.8\cdot0)=\sigma(0.5)=0.6225$. Step 2: $h_2=\sigma(0.5\cdot0+0.8\cdot0.6225)=\sigma(0.498)=0.6220$. So the state has kept most of what the first word put in, even though the second word was zero. A real model does this with 1,000 numbers per layer and four layers.

**Shapes (paper's sizes):**

| Thing | Size |
|---|---|
| One source word (embedding) | 1,000 numbers |
| Hidden state of one layer | 1,000 numbers |
| Sentence vector $v$ (4 layers, hidden and memory) | 8,000 numbers |
| Probabilities for one next word | 80,000 numbers |

### Why reversing the source helps
The pairs of words that match ("a" with its translation) are far apart if you write source then target. Reversing the source puts the *first* source words right next to the *first* target words (paper, 3.3).

Worked example (my arithmetic, positions counted from the start of source plus target): for a 3-word source "a b c" and target "α β γ", the matching words are 3, 3 and 3 positions apart. With the source reversed ("c b a"), they are 1, 3 and 5 apart: the average is still 3, but the smallest distance fell from 3 to 1. For 10 words the distances go from all 10 to a range from 1 to 19 with the average still 10. The paper says exactly this: the average distance is "unchanged", the "minimal time lag" is greatly reduced (3.3). Its measured effect: test perplexity from 5.8 to 4.7 (19% lower, my arithmetic) and BLEU from 25.9 to 30.6 (paper, 3.3; its Table 1 gives 26.17 and 30.59, see the notice below). The authors write that they have no complete explanation (3.3).

### Beam search
Instead of always taking the single most likely next word, the decoder keeps the $B$ best partial sentences (paper, 3.2). A partial sentence that ends with `<EOS>` is set aside as finished.

Worked example (my invented probabilities for a two-step output): first word A 0.5, B 0.4, C 0.1. After A the next words have probabilities X 0.3, Y 0.3 and `<EOS>` 0.4; after B: X 0.9, Y 0.05, `<EOS>` 0.05; after C: 0.34, 0.33, 0.33. Taking the best word at each step ("greedy") gives A then `<EOS>`, with probability $0.5\times0.4=0.20$. The best full sentence is B then X, $0.4\times0.9=0.36$. A beam of 2 keeps A and B after the first step, so it also keeps B and finds 0.36. A beam of 1 never recovers. The paper's own result is similar: with an ensemble of 5, beam 1 gave 33.00 BLEU, beam 2 gave 34.50 and beam 12 gave 34.81 (Table 1), so beam 2 won 83% of the gain from 1 to 12 (my arithmetic).

### Rescoring an n-best list
The LSTM can also score sentences instead of writing them. Take the 1,000 best candidate translations of the SMT system, compute each one's log probability with the LSTM and average it evenly with the SMT system's own score (paper, 3.2). The best candidate under the new score is the answer.

## The equations

### The recurrent network (paper, section 2)

$$
h_t = \operatorname{sigm}\!\left(W^{hx}x_t + W^{hh}h_{t-1}\right), \qquad y_t = W^{yh}h_t
$$

**Where:**
- $x_t$ is the input at step $t$ (a word vector). **Matters:** without it the network would ignore the sentence.
- $h_t$ is the hidden state, the network's memory after step $t$; $h_{t-1}$ is the previous one. **Matters:** the memory is what carries earlier words forward.
- $W^{hx}$ and $W^{hh}$ are learned weight matrices (input to hidden, hidden to hidden). **Matters:** $W^{hh}$ decides how much old memory survives.
- $\operatorname{sigm}$ is the sigmoid, which squeezes numbers into 0 to 1. **Matters:** it keeps the state from growing without limit.
- $y_t$ is the output at step $t$ and $W^{yh}$ the output weights.

**Example (tiny numbers):** $W^{hx}=0.5$, $W^{hh}=0.8$, $W^{yh}=2$, $x_1=1$, $x_2=0$, $h_0=0$. Then $h_1=\sigma(0.5)=0.6225$, $h_2=\sigma(0.8\times0.6225)=\sigma(0.498)=0.6220$ and $y_2=2\times0.6220=1.244$. The answer means: the state after step 2 still holds the first word, because the weight on the old memory (0.8) was large. (The paper's LSTM replaces this simple update with gates; it uses the formulation from Graves and does not print it.)

### The probability of a whole translation (paper, equation 1)

$$
p(y_1,\dots,y_{T'}\mid x_1,\dots,x_T)=\prod_{t=1}^{T'}p(y_t\mid v,\,y_1,\dots,y_{t-1})
$$

**Where:**
- $x_1,\dots,x_T$ is the source sentence (length $T$) and $y_1,\dots,y_{T'}$ the translation (length $T'$, which may differ).
- $v$ is the thought vector, the encoder's last hidden state. **Matters:** it is how the source reaches the target.
- $p(y_t\mid v,y_1,\dots,y_{t-1})$ is the chance of the next word given the thought vector and the words written so far, given by the softmax over the vocabulary. **Matters:** it is what the decoder outputs at every step.
- $\prod$ multiplies the chances of all the steps. **Matters:** a whole sentence is only as likely as the product of its steps; one very unlikely word pulls everything down.

**Example (tiny numbers):** a 3-step output with step chances 0.5, 0.4 and 0.9 (the last is `<EOS>`): $0.5\times0.4\times0.9=0.18$. The answer means: the model gives this whole sentence an 18% chance.

### The training objective (paper, section 3.2)

$$
\frac{1}{|\mathcal{S}|}\sum_{(T,S)\in\mathcal{S}}\log p(T\mid S)
$$

**Where:**
- $\mathcal{S}$ is the training set of sentence pairs and $|\mathcal{S}|$ how many there are.
- $S$ is a source sentence and $T$ its correct translation.
- $\log p(T\mid S)$ is the log of the probability the model gives to the correct translation. **Matters:** the log turns the product above into a sum, which is easier and safer to compute; it is largest (0) when the model is certain of the right answer.
- $\frac{1}{|\mathcal{S}|}\sum$ averages over the pairs. **Matters:** the value does not depend on the size of the training set. The model is trained to make this as large as possible.

**Example (tiny numbers):** two training pairs get probabilities 0.18 and 0.5. Logs: $\ln 0.18=-1.715$ and $\ln 0.5=-0.693$. The average is $(-1.715-0.693)/2=-1.204$. The answer means: training pushes this number up towards 0, by giving the correct translations more probability.

### Choosing the translation (paper, equation 2)

$$
\hat T=\arg\max_{T}\ p(T\mid S)
$$

**Where:**
- $\hat T$ is the translation the system outputs.
- $\arg\max_T$ means "the $T$ with the largest value". **Matters:** it asks for the single most likely whole sentence, not the most likely word at each step.
- $p(T\mid S)$ is the sentence probability from equation 1.

**Example (tiny numbers):** in the beam example above, the sentences have probabilities 0.20 (A, `<EOS>`), 0.36 (B, X) and so on; the arg max is B, X with 0.36. The answer means: the best sentence is not the one that starts with the best word. Searching all sentences is impossible, so the paper uses a left-to-right beam search (an approximation).

### Clipping the gradient (paper, section 3.4)

$$
s=\lVert g\rVert_2,\qquad \text{if } s>5:\quad g\leftarrow\frac{5\,g}{s}
$$

**Where:**
- $g$ is the gradient (the direction in which to change all the weights), divided by the batch size 128.
- $s$ is its length (its 2-norm).
- 5 is the chosen limit. **Matters:** it stops an occasional huge gradient from wrecking the weights (the paper says LSTMs rarely have vanishing gradients but "can have exploding gradients").
- $\frac{5g}{s}$ is the same direction with length exactly 5. **Matters:** only the length changes, not the direction.

**Example (tiny numbers):** $g=(9,-8)$, so $s=\sqrt{145}=12.04$, which is above 5. The new gradient is $5\cdot(9,-8)/12.04=(3.737,-3.322)$, whose length is exactly 5.0. If $s$ had been 3, nothing would change. The answer means: a too-big step is shortened to a safe one and keeps pointing the same way. See [[Gradient Clipping]].

## What they found
- **Main result (abstract and paper, 1 and 3.6):** an ensemble of 5 four-layer LSTMs with beam size 12 reached **34.81 BLEU** on the WMT'14 English-to-French test set, against 33.30 for the phrase-based SMT baseline. The paper calls it the first time a pure neural translation system beat a phrase-based SMT baseline on a large-scale task, and that the LSTM could not translate words outside its 80,000-word vocabulary (so the score was penalised).
- **With SMT (abstract and paper):** rescoring the baseline's 1,000-best list with the LSTM reached 36.5, 3.2 above the baseline and 0.5 below the best published result (37.0).
- **Long sentences:** no degradation on sentences with fewer than 35 words and only a minor one on the longest (paraphrasing the Figure 3 caption).
- **Reversal:** perplexity 5.8 to 4.7 and BLEU 25.9 to 30.6 (paper, 3.3).

### Direct translation (paper's Table 1)

| Method | Test BLEU (ntst14) | Against the baseline (mine) |
|---|---|---|
| Bahdanau et al. | 28.45 | -4.85 |
| Baseline system (phrase-based SMT) | 33.30 | 0 |
| Single forward LSTM, beam size 12 | 26.17 | -7.13 |
| Single reversed LSTM, beam size 12 | 30.59 | -2.71 |
| Ensemble of 5 reversed LSTMs, beam size 1 | 33.00 | -0.30 |
| Ensemble of 2 reversed LSTMs, beam size 12 | 33.27 | -0.03 |
| Ensemble of 5 reversed LSTMs, beam size 2 | 34.50 | +1.20 |
| **Ensemble of 5 reversed LSTMs, beam size 12** | **34.81** | **+1.51** |

Written by me, an AI, from the paper's Table 1 (text extraction; all eight rows copied). The last column is mine.
- **Method:** *forward* means the source is read in its normal order, *reversed* means backwards; *ensemble of $k$* means $k$ separately trained LSTMs whose probabilities are combined; *beam size* is how many partial sentences the decoder keeps.
- **Test BLEU:** the translation score on the 2014 news test set, ntst14, computed with the multi-bleu script on tokenised text, cased (paper, 3.6). Higher is better. See [[BLEU Score]] for how it is computed.
- **Against the baseline (mine):** the BLEU minus 33.30. It matters because the paper's claim is "beats the phrase-based system".

**Summary: what the table shows.** The best row is the 5-LSTM ensemble with beam 12 (34.81, 1.51 above the baseline). The worst LSTM row is the single forward model (26.17). Reversing the source is worth 4.42 BLEU on a single model (my arithmetic), more than the 4.22 gained by going from one model to an ensemble of five (30.59 to 34.81, mine). Beam 2 already gets 34.50 and the ensemble with beam 1 still gets 33.00 (just below the baseline). **Caution:** each row is one run or one ensemble with no error bars; I cannot judge whether the 0.31 between beam 2 and beam 12 is meaningful. All rows use the same 80,000-word vocabulary, which lowers the LSTM's score.

### Combined with a phrase-based system (paper's Table 2)

| Method | Test BLEU (ntst14) | Against the baseline (mine) |
|---|---|---|
| Baseline system | 33.30 | 0 |
| Cho et al. | 34.54 | +1.24 |
| Best WMT'14 result | 37.0 | +3.70 |
| Rescoring the baseline 1000-best with a single forward LSTM | 35.61 | +2.31 |
| Rescoring the baseline 1000-best with a single reversed LSTM | 35.85 | +2.55 |
| Rescoring the baseline 1000-best with an ensemble of 5 reversed LSTMs | 36.5 | +3.20 |
| Oracle rescoring of the baseline 1000-best lists | about 45 | about +11.7 |

Written by me, an AI, from the paper's Table 2 (text extraction; all seven rows). The last column is mine.
- **Rescoring:** the SMT system proposes 1,000 translations and the LSTM re-ranks them.
- **Oracle rescoring:** the score if a perfect judge always picked the best of the 1,000 candidates; it is an upper limit for any rescoring, not a real system (general knowledge).

**Summary: what the table shows.** Any LSTM rescoring helps (+2.31 to +3.20). The best, the ensemble, is 0.5 under the best published result (37.0). The oracle line shows how much room rescoring still had: about 45 against 36.5. **Caution:** the paper notes that when it scored the best WMT'14 system with its own method it got 37.0, while the official site reported 35.8; so BLEU numbers depend on the script (3.6).

### Long sentences and rare words (paper's Figure 3, read as a picture)
Two plots (I looked at them on page 7 of the PDF and read them by eye, so the values are approximate). Left: BLEU against sentence length; the LSTM is above or level with the baseline in nearly every length group, and the lines meet at the longest group (about 34 for both). Right: BLEU against how rare the words are; the LSTM is above the baseline for common words and a little below it (about 29 against 30) for the rarest group. So, in its own plot, the paper's claim that the LSTM "did not have difficulty on long sentences" holds, and its weakness shows on rare words.

### Sentence vectors (paper's Figure 2)
A two-dimensional picture of the model's hidden states for sentences such as "John respects Mary" and "Mary respects John" (PCA projection, see [[PCA]]). The paper says the phrases are grouped by meaning, mainly by word order, which a bag-of-words model could not capture. I did not look at this figure as a picture and cannot judge it.

**A mismatch in the paper:** section 3.3 reports the reversal effect as BLEU 25.9 to 30.6, but Table 1 gives 26.17 for the single forward LSTM and 30.59 for the reversed one. The paper does not say why they differ (perhaps different models or beam sizes; that is my guess). Also, Figure 1's picture feeds `A B C` in that order while its caption says the LSTM reads the sentence in reverse.

## Questions I had, answered

**Why does reversing help if the average distance does not change?**
Because backpropagation cares about the *shortest* path from a source word to its target word. With the reversal the first words line up closely (distance 1 in my 3-word example), which gives learning a short "foothold"; the later words stay far apart, but the network can build on the first ones. The paper has no full explanation (3.3), and my example only restates its argument; it is not a proof.

**Isn't one fixed list of numbers a bottleneck for long sentences?**
It is a fixed size (8,000 numbers) whatever the length, and other researchers had seen poor results for long sentences with similar models (paper, 1). The paper was surprised that its model did fine, and credits the reversal (paper, 1 and 3.7). It still names attention (Bahdanau) as a different way to attack the problem. See [[Attention]].

**What is `<EOS>` for?**
It is a special word meaning "stop". Without it the model could not say when the sentence ends, so it could not give a probability to sentences of different lengths (paper, 2).

**What does "penalised on out-of-vocabulary words" mean?**
Only the 80,000 most frequent target words exist for the model. Every other word is replaced by `UNK`, which can never match the human translation. In Table 3 of the paper the model writes "Ulrich UNK" where the truth is "Ulrich Hackenberg", so the name is lost. That lowers the BLEU, so the true ability is higher than 34.81 (paper, 1 and 3.1).

**Why does a beam of 2 do almost as well as a beam of 12?**
Because the error that beam search fixes (picking a bad first word that looks best) is usually fixed by keeping just one more option. In my example, a beam of 2 is enough to recover the best sentence; in the paper, beam 2 got 83% of the BLEU gain from beam 1 to beam 12 (mine). I only have the paper's single measurement for this, so it is not a general rule.

## Why it mattered
Seq2Seq showed that a neural network alone could translate, with no hand-built parts, and be competitive with a phrase-based system (paper). The idea of an encoder that reads and a decoder that writes became the standard layout for many text tasks. Its fixed-size vector was soon replaced by attention (named in the paper), which led to [[Attention Is All You Need 2017]] (my note, not a claim of this paper).

## Limits and open questions
- **Fixed vocabulary:** 80,000 target words and a single `UNK` for the rest, so names and rare words are lost (paper, 3.1 and 3.6).
- **A bottleneck vector:** all information passes through 8,000 numbers; the authors think it works well here but expect further gains from other improvements.
- **No complete explanation for the reversal trick:** "we do not have a complete explanation" (3.3). The paper also says a plain RNN "should be easily trainable" on reversed sentences, but "we did not verify it experimentally" (5).
- **Cost:** ten days on eight GPUs for one model, plus an ensemble of five (paper, 3.5).
- **Not beating the best system:** the pure LSTM (34.81) is below the best WMT'14 result (37.0); it only beat the phrase-based baseline.
- **Only one language pair:** English to French (paper, 3).

## Concepts used
[[LSTM]], [[Recurrent Neural Network]], [[Embedding]], [[Softmax]], [[Beam Search]], [[Decoding Strategies]], [[BLEU Score]], [[Perplexity]], [[Stochastic Gradient Descent]], [[Gradient Clipping]], [[Backpropagation]], [[Encoder-Decoder]], [[Ensemble Methods]], [[PCA]], [[Attention]]

## Reading trail
- **Came from:** [[From FNN to RNN]], the LSTM and the early neural translation models of Cho and Kalchbrenner (named in the paper, not read).
- **Led to:** [[Attention Is All You Need 2017]]; attention (Bahdanau and others, named in the paper, not read) is the step between.

## How much I read
I read the whole paper as extracted text (pages 1 to 8): abstract, introduction, model, experiments, related work and conclusion. I did not read the reference list (page 9) beyond the names of the works cited. I looked at pages 2 and 7 as pictures (Figure 1; Table 3 and Figure 3). I did not look at Figure 2 as a picture. Numbers from the paper are marked "paper" or "abstract"; the arithmetic, the tiny examples and the 384-million split are mine. Authors and the first-posted date were checked on the arXiv page. Not opened in Obsidian, so the flashcards are unreviewed.

## Sources
- Sutskever, I., Vinyals, O., Le, Q. V. (2014). *Sequence to Sequence Learning with Neural Networks.* arXiv 1409.3215 (v3, 14 December 2014). <https://arxiv.org/abs/1409.3215>. Licence on the arXiv page: the standard non-exclusive distribution licence, so I copied none of its figures.
- Figure: [EncoderDecoder.pdf](https://commons.wikimedia.org/wiki/File:EncoderDecoder.pdf) (Babayaga94, CC BY-SA 4.0), Wikimedia Commons.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-nlp Seq2Seq (Sutskever, Vinyals and Le, Google, 2014, arXiv 1409.3215): what problem did it attack?::Deep networks need fixed-size inputs and outputs, but translation and similar tasks map a sequence of unknown length to another sequence of a different length; the paper wanted one general method with minimal assumptions about the sequences.

#flashcards/research/concept #flashcards/topic/research-nlp What did people do before Seq2Seq for translation?::Phrase-based statistical machine translation (33.30 BLEU baseline on the paper's test set); neural networks mainly rescored its n-best lists. Kalchbrenner and Blunsom and Cho and others had mapped a sentence to a vector and back, but mainly as helpers; attention models such as Bahdanau's reached 28.45 BLEU (paper's Table 1).

#flashcards/research/concept #flashcards/topic/research-nlp Seq2Seq: what is the idea?::One four-layer LSTM reads the source sentence, one word at a time, into a fixed-size vector (its last hidden state); a second LSTM starts from that vector and writes the translation one word at a time until it writes `<EOS>`. The source sentence is read in reverse.

#flashcards/research/concept #flashcards/topic/research-nlp What are the sizes of the Seq2Seq model (paper, 3.4)?::Two LSTMs of 4 layers with 1,000 cells each; 1,000-number word embeddings; source vocabulary 160,000 and target 80,000; a sentence is 8,000 numbers; 384 million parameters, 64 million of them recurrent.

#flashcards/research/concept #flashcards/topic/research-nlp Where do the 384 million parameters of Seq2Seq come from?::My reconstruction (the paper only gives totals): 160M input embeddings + 80M output embeddings + 80M softmax weights + 32M encoder LSTM + 32M decoder LSTM = 384M. Over 80% is embeddings and the output layer.

#flashcards/research/concept #flashcards/topic/research-nlp Why does Seq2Seq use a deep LSTM, and how do we know how deep?::The paper found deep LSTMs "significantly outperformed" shallow ones, with each extra layer cutting perplexity by nearly 10%. It chose four layers; it gives no table and no reason for stopping there.

#flashcards/research/concept #flashcards/topic/research-nlp Why does Seq2Seq use two separate LSTMs?::Doing so adds parameters at negligible computational cost and makes training on several language pairs natural (paper, 2). The paper does not compare one LSTM against two.

#flashcards/research/concept #flashcards/topic/research-nlp What does the thought vector do in Seq2Seq, and how big is it?::It is the encoder's last hidden state: the only link from source to target, 8,000 numbers (4 layers times 1,000 cells times 2, my arithmetic) whatever the sentence length. The decoder's starting state is set to it.

#flashcards/research/concept #flashcards/topic/research-nlp Why does every sentence end with `<EOS>`?::It lets the model define a probability for sequences of every length, and tells the decoder when to stop (paper, 2).

#flashcards/research/concept #flashcards/topic/research-nlp What happens to a word outside Seq2Seq's vocabulary?::It becomes `UNK` (vocabulary 160,000 source words and 80,000 target words). `UNK` can never match the reference, so BLEU is penalised: the model writes "Ulrich UNK" for "Ulrich Hackenberg" (paper's Table 3).

#flashcards/research/concept #flashcards/topic/research-nlp Why was the Seq2Seq softmax split over four GPUs?::It is a "naive softmax" over 80,000 words, which is expensive: each of four GPUs multiplied by a $1000\times20000$ matrix, while the four LSTM layers used the other four GPUs (paper, 3.5).

#flashcards/research/concept #flashcards/topic/research-nlp ![[encoder-decoder-rnn.jpg]]
What does this encoder-decoder picture show, and how does it differ from the paper's model?
?
An encoder RNN reads $x_1$ to $x_{T_1}$ and ends in $h_{T_1}$; a decoder RNN produces $y_1$ to $y_{T_2}$. It is a free redrawing (Babayaga94, CC BY-SA 4.0). It feeds the final state to every decoder step, while the paper only uses it to start the decoder, and it has no four layers, `<EOS>` or reversed input.

#flashcards/research/concept #flashcards/topic/research-nlp Why does reversing the source sentence help Seq2Seq?::It puts the first source words next to the first target words, which shortens the "minimal time lag" so backpropagation can link source and target; the average distance does not change. Perplexity fell from 5.8 to 4.7 and BLEU rose from 25.9 to 30.6 (paper, 3.3). The authors have no complete explanation.

#flashcards/research/concept #flashcards/topic/research-nlp A 3-word source "a b c" and target "α β γ": how far apart are the matching words with and without reversing the source?::Without: 3, 3 and 3 positions. With the source reversed: 1, 3 and 5. The average stays 3, the smallest distance drops from 3 to 1 (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-nlp How does the Seq2Seq decoder choose a translation?::Left-to-right beam search: keep the $B$ best partial sentences, extend each with every word, keep the best $B$ again; sentences ending in `<EOS>` are set aside. It is approximate but simple; beam size 1 already works and beam 2 gives most of the benefit.

#flashcards/research/concept #flashcards/topic/research-nlp In my beam example (first words A 0.5, B 0.4, C 0.1; after A: EOS 0.4; after B: X 0.9), what does greedy find, what is best, and why does beam 2 succeed?::Greedy finds A then EOS with probability 0.5 × 0.4 = 0.20. The best is B then X: 0.4 × 0.9 = 0.36. Beam 2 keeps A and B after step one, so it finds B, X. (My invented numbers.)

#flashcards/research/concept #flashcards/topic/research-nlp How does Seq2Seq use an LSTM to rescore an n-best list?::Take the SMT system's 1,000 best candidates, compute each one's log probability with the LSTM, average it evenly with the SMT score, and pick the best (paper, 3.2).

#flashcards/research/concept #flashcards/topic/research-nlp Write the simple recurrent update used in Seq2Seq's section 2 and explain each symbol.
?
$$h_t = \operatorname{sigm}\!\left(W^{hx}x_t + W^{hh}h_{t-1}\right), \qquad y_t = W^{yh}h_t$$
- $x_t$: the input word vector at step $t$; without it the sentence is ignored.
- $h_t$, $h_{t-1}$: the hidden state (memory) now and one step earlier; it carries earlier words forward.
- $W^{hx}$, $W^{hh}$: learned weights; $W^{hh}$ decides how much old memory survives.
- $\operatorname{sigm}$: squeezes values into 0 to 1 so the state cannot grow without limit.
- $y_t$, $W^{yh}$: the output and its weights. (The paper's LSTM replaces this update with gates, not printed in the paper.)

#flashcards/research/concept #flashcards/topic/research-nlp Recurrent update example: $W^{hx}=0.5$, $W^{hh}=0.8$, $x_1=1$, $x_2=0$, $h_0=0$. What are $h_1$ and $h_2$?::$h_1=\sigma(0.5)=0.6225$; $h_2=\sigma(0.8\times0.6225)=\sigma(0.498)=0.6220$. The state still holds the first word because the memory weight 0.8 is large (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-nlp Why is there a sigmoid in the recurrent update?::It squeezes each state value into 0 to 1, so repeated updates cannot make the state grow without limit. In my example $h_1=0.6225$ and $h_2=0.6220$ stay inside 0 to 1 whatever the input.

#flashcards/research/concept #flashcards/topic/research-nlp Write the probability of a whole translation in Seq2Seq (equation 1) and explain each symbol.
?
$$p(y_1,\dots,y_{T'}\mid x_1,\dots,x_T)=\prod_{t=1}^{T'}p(y_t\mid v,\,y_1,\dots,y_{t-1})$$
- $x_1..x_T$: the source sentence; $y_1..y_{T'}$: the translation, whose length $T'$ may differ from $T$.
- $v$: the thought vector, the encoder's last hidden state; it is how the source reaches the target.
- $p(y_t\mid v,y_{<t})$: the chance of the next word from the decoder's softmax over the vocabulary.
- $\prod$: multiplies the step chances; one very unlikely word pulls the whole sentence down.

#flashcards/research/concept #flashcards/topic/research-nlp Equation 1 example: three steps with chances 0.5, 0.4 and 0.9. What is the sentence probability?::$0.5\times0.4\times0.9=0.18$: the model gives the whole sentence an 18% chance (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-nlp Why is the translation probability a product of step probabilities?::The decoder writes one word at a time, each conditioned on the earlier words, so by the chain rule of probability the sentence chance is the product of the step chances. With 0.5, 0.4 and 0.9 it is 0.18; a single 0.01 step would cut it to 0.002.

#flashcards/research/concept #flashcards/topic/research-nlp Write the Seq2Seq training objective and explain each symbol.
?
$$\frac{1}{|\mathcal{S}|}\sum_{(T,S)\in\mathcal{S}}\log p(T\mid S)$$
- $\mathcal{S}$: the training set of sentence pairs; $|\mathcal{S}|$ its size.
- $S$: a source sentence; $T$: its correct translation.
- $\log p(T\mid S)$: the log probability of the correct translation; the log turns the product into a sum; it is 0 when the model is certain.
- $\frac{1}{|\mathcal{S}|}\sum$: the average over pairs, so the value does not depend on the data size. Training maximises it.

#flashcards/research/concept #flashcards/topic/research-nlp Objective example: two pairs get probabilities 0.18 and 0.5. What is the average log probability?::$\ln0.18=-1.715$, $\ln0.5=-0.693$; the average is $-1.204$ (my arithmetic). Training pushes it up towards 0.

#flashcards/research/concept #flashcards/topic/research-nlp Why does the training objective use the log of the probability?::The log turns the product of many step probabilities into a sum, which is easier and numerically safer. Example: $\ln(0.5\cdot0.4\cdot0.9)=\ln0.5+\ln0.4+\ln0.9=-1.715$. It is also 0 for a certain answer, so the best value is a clear target.

#flashcards/research/concept #flashcards/topic/research-nlp Write the Seq2Seq decoding rule (equation 2) and explain each symbol.
?
$$\hat T=\arg\max_{T}\ p(T\mid S)$$
- $\hat T$: the translation the system outputs.
- $\arg\max_T$: the $T$ with the largest value; it asks for the best whole sentence, not the best word at each step.
- $p(T\mid S)$: the sentence probability from equation 1. Searching all sentences is impossible, so beam search approximates it.

#flashcards/research/concept #flashcards/topic/research-nlp Decoding example: sentences "A EOS" with 0.20 and "B X" with 0.36 (and others lower). What is $\hat T$?::"B X" with 0.36: the best sentence does not start with the best single word (A, 0.5). (My invented numbers.)

#flashcards/research/concept #flashcards/topic/research-nlp Write Seq2Seq's gradient clipping rule and explain each symbol.
?
$$s=\lVert g\rVert_2,\qquad \text{if } s>5:\quad g\leftarrow\frac{5\,g}{s}$$
- $g$: the gradient divided by the batch size 128.
- $s$: its length (2-norm).
- 5: the chosen limit; it stops an occasional huge gradient from wrecking the weights.
- $5g/s$: same direction, length exactly 5; only the size changes.

#flashcards/research/concept #flashcards/topic/research-nlp Clipping example: $g=(9,-8)$ and limit 5. What is the clipped gradient?::$s=\sqrt{145}=12.04>5$, so $g=5\,(9,-8)/12.04=(3.737,-3.322)$, length 5.0. A gradient of length 3 would not change (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-nlp Why does Seq2Seq clip gradients, and why divide by $s$ and not just cap each entry?::LSTMs rarely suffer vanishing gradients but "can have exploding gradients" (paper, 3.4). Scaling the whole vector by $5/s$ keeps its direction; capping each entry would change the direction. In my example $(9,-8)$ becomes $(3.737,-3.322)$, the same direction.

#flashcards/research/concept #flashcards/topic/research-nlp What was the main result of Seq2Seq (abstract and paper)?::On WMT'14 English to French, an ensemble of 5 LSTMs with beam size 12 got 34.81 BLEU against 33.30 for a phrase-based SMT system, with a vocabulary of 80,000 words, so the score was penalised on unknown words. Rescoring the SMT 1000-best reached 36.5.

#flashcards/research/concept #flashcards/topic/research-nlp Which row of Seq2Seq's Table 1 is best, which LSTM row is worst, and what is the caution?::Best: ensemble of 5 reversed LSTMs, beam 12 (34.81, +1.51 over the baseline). Worst LSTM row: single forward LSTM (26.17, -7.13). Reversing is worth 4.42 BLEU on a single model, more than 4.22 for an ensemble of 5 (my arithmetic). Caution: single runs with no error bars, and the 80,000-word vocabulary lowers the score.

#flashcards/research/concept #flashcards/topic/research-nlp What does Seq2Seq's Table 2 show?::Rescoring the baseline's 1000-best with an LSTM helps by +2.31 (single forward) to +3.20 (ensemble, 36.5), which is 0.5 below the best published result of 37.0. The oracle (a perfect judge) would reach about 45. Caution: BLEU depends on the script; the best WMT'14 system scored 37.0 by the paper's method and 35.8 on the official site.

#flashcards/research/concept #flashcards/topic/research-nlp How much of the gain from beam 1 to beam 12 does beam 2 deliver in Seq2Seq's Table 1?::With 5 LSTMs: beam 1 is 33.00, beam 2 34.50, beam 12 34.81, so beam 2 gets 1.50 of 1.81 BLEU, 83% (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-nlp What did Seq2Seq's Figure 3 show about long sentences and rare words (read by eye)?::The LSTM was above or level with the baseline for nearly every length group (no degradation under 35 words, minor on the longest), but slightly below it for the rarest-word group (about 29 against 30). Approximate values from the plot.

#flashcards/research/concept #flashcards/topic/research-nlp Where does Seq2Seq's paper contradict itself?::Section 3.3 gives the reversal effect as BLEU 25.9 to 30.6, but Table 1 gives 26.17 to 30.59; and Figure 1 draws the input in the normal order while the caption says it is read in reverse. The paper does not explain either.

#flashcards/research/concept #flashcards/topic/research-nlp Is the fixed-size thought vector a bottleneck for long sentences in Seq2Seq?::It is a fixed 8,000 numbers whatever the length, and similar models had done badly on long sentences. The paper was surprised its model did fine and credits the reversal; attention was a different answer (Bahdanau, cited, not read).

#flashcards/research/concept #flashcards/topic/research-nlp What limits does Seq2Seq state or show?::A fixed vocabulary with `UNK` for the rest; a fixed-size bottleneck vector; no full explanation for the reversal trick; ten days on 8 GPUs per model; the pure LSTM (34.81) is below the best system (37.0); only English to French was tested.

#flashcards/research/concept #flashcards/topic/research-nlp How much of the Seq2Seq paper did the note read?::The whole paper as text (pages 1 to 8) and pages 2 and 7 as pictures, but not the reference list or Figure 2 as a picture. Authors and date were checked on arXiv; the 384-million split and the tiny examples are my own.
