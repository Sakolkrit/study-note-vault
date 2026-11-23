---
tags: [paper, NLP, transformer, attention]
link: https://arxiv.org/abs/1706.03762
authors: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin (Google Brain, Google Research, University of Toronto; checked on the paper's first page)
year: 2017
status: body-read
field: text / architecture
---

# Attention Is All You Need 2017

> **In one line for a 12-year-old:** Let every word look at every other word directly and decide which matter. No more reading one word at a time.

## The problem
Make sequence models faster to train and better at long-range links.

## What people did before
Recurrent or convolutional encoder-decoders, linked by an attention mechanism ([[Seq2Seq 2014]]).

## The idea
The [[Transformer]]: only [[Attention]] and feed-forward layers, no recurrence or convolution.

## The architecture (figures)
The paper's Figure 1 shows an **encoder** (left) and a **decoder** (right). Below, each part is explained on its own: what it does and why it matters. Marks: "(paper, section)" means I read it in the paper's body; "(general knowledge)" means it is my own background knowledge, not something I read in the pages I read.

### Encoder: reads the whole input sentence and gives every word a vector that already knows its context
Overall (paper, 3.1): embeddings plus positional encoding go in once at the bottom. Then comes a stack of $N=6$ identical layers; each layer has two sub-layers, and each sub-layer is wrapped in Add and Norm. The output of layer 1 feeds layer 2, and so on; the top layer's output goes to the decoder. Every sub-layer and the embeddings produce vectors of size $d_{\text{model}}=512$, so the pieces fit together.

- **Input embedding.**
  - *What it does:* a learned table turns each token (a word piece; the English-German model has about 37,000 of them, paper 5.1) into a vector of 512 numbers; the paper multiplies these by $\sqrt{d_{\text{model}}}$ (paper, 3.4).
  - *Why it matters:* everything after this is arithmetic on numbers, so words must become vectors first, and training can give similar tokens similar vectors (general knowledge). The same weight matrix is also shared with the decoder's embedding and with the last linear layer (paper, 3.4).
- **Positional encoding.**
  - *What it does:* adds a vector of sines and cosines, which depends only on the word's position, to each embedding. It has the same size as the embedding so the two can be summed (paper, 3.5; the formula is in the equations section). Dropout is applied to the sum (paper, 5.4).
  - *Why it matters:* the model has no recurrence and no convolution, so attention alone ignores order; without this, "dog bites man" and "man bites dog" would look the same to it (the example is mine; the reason is the paper's, 3.5).
- **Multi-head self-attention** (sub-layer 1).
  - *What it does:* every word works out how much to look at every other word of the same sentence (queries, keys, values), and takes a weighted mix of their values. Eight heads do this side by side on smaller vectors, and their results are joined (paper, 3.2).
  - *Why it matters:* any two words are linked in one step, however far apart (Table 1: longest path $O(1)$), and different heads can look at different things (paper, 3.2.2 and 4).
- **Add and Norm** (after each sub-layer).
  - *What it does:* the sub-layer's output is added to its own input and then normalised: $\mathrm{LayerNorm}(x+\mathrm{Sublayer}(x))$ (paper, 3.1). Dropout is applied to the sub-layer output before the addition (paper, 5.4).
  - *Why it matters:* the "add $x$" part is a residual connection; the sizes must match, which is why all sub-layers output 512 numbers (paper, 3.1). Why shortcuts make deep stacks trainable is argued in the ResNet paper ([[ResNet 2015]]); the Transformer paper gives no reason in the pages I read. Layer normalisation keeps the numbers in a steady range (general knowledge).
- **Position-wise feed-forward network** (sub-layer 2).
  - *What it does:* two linear layers with a ReLU between them, 512 → 2048 → 512, applied to **each position separately and identically** (paper, 3.3).
  - *Difference from a "normal" feed-forward network:* it is an ordinary feed-forward network; "position-wise" only says that the same small network is run on one word's vector at a time, with no mixing between positions (attention does the mixing). The paper's own other description: two convolutions with kernel size 1. A normal dense layer over the whole sentence at once would have a weight for every pair of positions and would work only for one fixed length. My arithmetic for a 10-word sentence: the first layer would have $(10\cdot512)\cdot(10\cdot2048)=104{,}857{,}600$ weights, against $512\cdot2048=1{,}048{,}576$ for the position-wise version, which is 100 times fewer and works for any length.
  - *Why it matters:* attention only mixes words together; this gives each word's vector extra nonlinear processing afterwards (general knowledge; the paper states no reason). The weights are the same at every position within a layer and different from layer to layer (paper, 3.3).
- **The repeat ($N=6$).**
  - *What it does:* the whole boxed layer is stacked 6 times with separate weights, each layer working on the previous layer's output (paper, 3.1).
  - *Why 6 and not 5 or 7:* the paper does not say. It is a chosen setting, used for both the base and the big model. The only evidence is its Table 3 (row group C, English-German development set, one run each): $N=2$ gave 23.7 BLEU (36 million parameters), $N=4$ gave 25.3 (50 million), $N=6$ the base 25.8 (65 million), and $N=8$ gave 25.5 (80 million). More layers helped up to 6, and 8 was not better, but a 0.3 gap from one run may be noise. They did not try 5 or 7. More layers also cost more parameters and time, so the choice is a trade-off (general knowledge).

### Decoder: writes the output one word at a time, looking at the words written so far and at the encoder's result
Overall (paper, 3.1): the output embedding (offset by one position) and positional encoding go in once at the bottom. Then comes a stack of $N=6$ identical layers; each layer has **three** sub-layers, each wrapped in Add and Norm. The top layer feeds a linear layer and a softmax.

- **Output embedding, shifted right.**
  - *What it does:* the same kind of embedding as the input side, but for the words of the output written so far, **offset by one position**: the decoder's input is the target sentence slid one place to the right, so each position receives the previous word and must predict the next one (paper, 3.1).
  - *Why it matters:* without the shift, a position would receive the very word it is asked to predict. My own illustration for the target "Ich liebe Katzen" (the START and END markers are general knowledge, not read in the pages I read):

| Decoder position | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| Input given to the decoder | START | Ich | liebe | Katzen |
| Word it must predict | Ich | liebe | Katzen | END |

- **Positional encoding.** The same as in the encoder, added to the output embeddings once, at the bottom (paper, 3.5).
- **Masked multi-head self-attention** (sub-layer 1).
  - *What it does:* the same self-attention as in the encoder, except that a position may look only at itself and the positions before it. The mask sets the forbidden scores to $-\infty$ before the softmax (paper, 3.2.3).
  - *Why it matters:* in training the whole correct output is given at once, for speed. Without a mask, the position that must predict word 3 could look at word 3 and copy it; the loss would fall to nearly zero and nothing would be learned, and at test time those words do not exist yet. This is a leak of the answer, not over-fitting (over-fitting is memorising the training data, which the paper handles with dropout and label smoothing). The mask plus the shift make prediction $i$ depend only on known outputs before position $i$: the paper calls this the "auto-regressive property" (paper, 3.1 and 3.2.3).
- **Multi-head attention over the encoder** (sub-layer 2, the "encoder-decoder attention").
  - *What it does:* the queries come from the decoder, and the keys and values come from the encoder's top-layer output. All 6 decoder layers read the same encoder output (paper, 3.2.3).
  - *Why it matters:* this is how the decoder knows the **whole input sentence**: every output position can look at every input position. (The masked box above hides only the future words of the output, not the input.)
- **Add and Norm, and the feed-forward network** (sub-layer 3). Exactly as in the encoder; Add and Norm follows each of the three sub-layers (paper, 3.1).
- **The repeat ($N=6$).** The same as in the encoder: the boxed layer is stacked 6 times with separate weights.
- **Linear layer and softmax** (once, above layer 6).
  - *What it does:* a learned linear layer turns the decoder's output vector into one score per word of the vocabulary, and the softmax turns the scores into next-word probabilities. Its weight matrix is shared with the embeddings (paper, 3.4).
  - *Why it matters:* the model needs a probability for each possible next word in order to choose the output.

### Size check (my arithmetic, base sizes, biases and LayerNorm left out)
One encoder layer holds about $1.05$M attention plus $2.10$M feed-forward weights ($3{,}145{,}728$), so $18{,}874{,}368$ for 6 layers. One decoder layer holds two attention blocks and one feed-forward network ($4{,}194{,}304$), so $25{,}165{,}824$ for 6 layers. Producing a sentence one word at a time at test time is general knowledge, not read in the pages I read.

The paper uses multi-head attention in three ways: encoder-decoder attention, self-attention in the encoder, and masked self-attention in the decoder (paper, 3.2.3). All three are covered above.

![[attention-paper-fig1-transformer.png]]
*The Transformer model architecture, left half the encoder, right half the decoder. Image: Yuening Jia, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:The-Transformer-model-architecture.png), CC BY-SA 3.0. This is a redrawing of the paper's Figure 1, not the paper's own picture. I compared it with the paper's Figure 1 by eye and the layout matches (Add and Norm after each sub-layer, masked attention in the decoder, linear and softmax at the top); small details such as colours differ. One thing is missing: the paper's figure has an "N×" label beside each stack, meaning the whole boxed layer is repeated N=6 times, and this drawing leaves that label out, so it shows each stack only once. Reading it: the rounded box is one layer (two sub-layers); the embedding and positional encoding below it happen once; the output of layer 6 goes to the decoder.*

## Attention, step by step
This section walks through the two boxes of the paper's Figure 2 from the bottom up, each with its drawing next to it: scaled dot-product attention first, then multi-head attention. Marks as before: "(paper, section)" was read in the paper; "(general knowledge)" is my own background; "(mine)" is my analogy or my arithmetic. The formulas are in the equations section below.

### Scaled dot-product attention (left drawing)
**Setting the scene (my analogy).** Think of a library. Each word has a **query** ("what am I looking for?"), each word also has a **key** (the label on its spine, "what do I offer?") and a **value** (the content of the book). A word compares its query with every key, decides which books matter, and reads a blend of their values.

Where do $Q$, $K$ and $V$ come from? Each is a table with one row per word. In the full model they are made from the word vectors by learned matrices (see multi-head attention below). Here they are simply given (paper, 3.2.1).

![[attention-paper-fig2-scaled-dot-product.png]]
*Scaled dot-product attention: multiply queries and keys (MatMul), scale, optionally mask, softmax, then multiply by the values. Image: author listed as unknown on [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Self-Attention_(Scaled_dot-product_Attention).png), CC BY 4.0. A redrawing of the left half of the paper's Figure 2; it labels the inputs $Q_i$, $K_i$, $V_i$ and the output $Z_i$, which the paper does not.*

1. **MatMul of $Q$ and $K^T$ (the bottom box).**
   - *What it does:* takes the dot product of every query with every key. For $n$ words this gives an $n\times n$ table of scores: row $i$, column $j$ says how well word $i$'s query matches word $j$'s key (paper, 3.2.1).
   - *Why it matters:* this is "how relevant is word $j$ to word $i$", computed for all pairs at once with one fast matrix multiplication. The paper says dot-product attention is much faster and more space-efficient in practice than additive attention for this reason (paper, 3.2.1). The price is the $n\times n$ table: cost $O(n^2\cdot d)$ (paper, Table 1). My arithmetic: for 512 words the table has $262{,}144$ entries per head.
2. **Scale (divide by $\sqrt{d_k}$).**
   - *What it does:* divides every score by $\sqrt{d_k}$, the square root of the key length (paper, 3.2.1).
   - *Why it matters:* for large $d_k$ the dot products grow large and push the softmax into regions with extremely small gradients, so learning stalls (paper, 3.2.1 and its footnote 4: for random entries with variance 1 the dot product has variance $d_k$). My run on 20,000 random pairs: the spread was 2.00 at $d_k=4$, 8.03 at $d_k=64$, and 1.00 after dividing by $\sqrt{d_k}$.
3. **Mask (optional).**
   - *What it does:* sets the forbidden scores to $-\infty$ before the softmax, so their weights become exactly 0 (paper, 3.2.3). Used only in the decoder's self-attention.
   - *Why it matters:* it stops a word from looking at later words, so it cannot copy the answer in training (explained in the decoder part above).
4. **Softmax.**
   - *What it does:* turns each row of scores into weights between 0 and 1 that add up to 1 (paper, 3.2.1).
   - *Why it matters:* the weights say how much of each word to take; a high score gets a large share and a very low score almost none. The result is a weighted average, not a hard choice of one word.
5. **MatMul with $V$ (the top box).**
   - *What it does:* for each word, adds up the value rows, each multiplied by that word's weight for it (paper, 3.2.1).
   - *Why it matters:* this is where information actually moves between words: the output row for a word is a blend of the values of the words it paid attention to. So each word's new vector now carries information from its context.

**Worked example (mine; three words, $d_k=2$, and $Q=K=V$ taken from the same table, as in self-attention without the learned matrices).** The words are $(1,0)$, $(0,1)$ and $(1,1)$.
- Scores $QK^T$: $\begin{pmatrix}1&0&1\\0&1&1\\1&1&2\end{pmatrix}$. After dividing by $\sqrt2$: $0.707$ for a score of 1, $1.414$ for a score of 2.
- Softmax weights, one row per word: $(0.401,\,0.198,\,0.401)$, $(0.198,\,0.401,\,0.401)$, $(0.248,\,0.248,\,0.503)$. Each row adds to 1.
- Outputs: $(0.80,\,0.60)$, $(0.60,\,0.80)$ and $(0.75,\,0.75)$. Word 3, which matches itself best (score 2), gives itself the largest weight (0.503).
- With the decoder mask (each word sees only itself and earlier words) the weights become $(1,0,0)$, $(0.33,\,0.67,\,0)$ and $(0.248,\,0.248,\,0.503)$, and the outputs $(1,0)$, $(0.33,\,0.67)$, $(0.75,\,0.75)$: word 1 can only copy itself, and the last row is unchanged because it already sees everyone before it.

### Multi-head attention (right drawing)
**The idea.** One attention step gives each word one blend of the others. Several attention steps in parallel let each word look at the sentence in several different ways at once (for example one head following which noun a word refers to and another following its neighbours; this example is mine, not the paper's).

![[attention-paper-fig2-multi-head.png]]
*Multi-head attention: Q, K and V each go through several linear layers, each set feeds its own scaled dot-product attention, the results are concatenated and passed through one more linear layer. Image: author listed as unknown on [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Multi-head_attention.png), CC BY 4.0. A redrawing of the right half of the paper's Figure 2. It writes "x N" for the number of heads where the paper writes $h$, and lists Q, K, V in the opposite order.*

1. **Linear projections (the three bottom rows of boxes).**
   - *What it does:* $Q$, $K$ and $V$ (size 512 per word) are each multiplied by learned matrices, one set for each of the $h=8$ heads, giving smaller vectors of size $d_k=d_v=64$ (paper, 3.2.2).
   - *Why it matters:* every head gets its own learned "view" of the same words, and because each view is only 64 numbers wide, the total cost is similar to one full-size head (paper, 3.2.2).
2. **Scaled dot-product attention in each head, in parallel (the big purple box).**
   - *What it does:* each head runs the five steps above on its own small $Q$, $K$, $V$, so each head has its own table of attention weights (paper, 3.2.2).
   - *Why it matters:* different heads can attend to different positions; with a single head, the paper says, averaging inhibits this (paper, 3.2.2). The paper's Table 3 rows (A) (English-German development set, one run each) show one head 0.9 BLEU below the best setting: $h=1$: 24.9, $h=4$: 25.5, $h=8$ (base): 25.8, $h=16$: 25.8, $h=32$: 25.4, so too many heads also hurt.
3. **Concat.**
   - *What it does:* the 8 outputs of 64 numbers each are placed side by side into one vector of $8\times64=512$ numbers per word (paper, 3.2.2).
   - *Why it matters:* it brings the results of all heads back together in the size the rest of the model expects.
4. **Final linear layer ($W^O$, the top box).**
   - *What it does:* a learned $512\times512$ matrix mixes the concatenated vector (paper, 3.2.2).
   - *Why it matters:* the heads' results sit in separate blocks of the vector; this layer lets the model combine information across heads (general knowledge; the paper only states the operation).

**Shapes for a 10-word sentence (mine, base sizes).**

| Step | Shape (rows by columns) | Note |
|---|---|---|
| Input $Q$, $K$, $V$ | $10\times512$ each | one row per word |
| After projection, per head | $10\times64$ each | 8 heads |
| Scores per head | $10\times10$ | $100$ per head, $800$ for all 8 heads |
| Weights per head | $10\times10$ | each row adds to 1 |
| Output per head | $10\times64$ | weights times $V$ |
| After concat | $10\times512$ | $8\times64=512$ |
| After $W^O$ | $10\times512$ | same size as the input, so Add and Norm can follow |

**Where $Q$, $K$ and $V$ come from in the three uses (paper, 3.2.3).**

| Use | $Q$ comes from | $K$ and $V$ come from | Mask |
|---|---|---|---|
| Encoder self-attention | previous encoder layer | previous encoder layer | none |
| Decoder self-attention | previous decoder layer | previous decoder layer | future positions hidden |
| Encoder-decoder attention | previous decoder layer | encoder's top-layer output | none |

### Why attention instead of recurrent or convolutional layers? (paper's Table 1)

**Table 1: layer types** (paper's Table 1). $n$ = sequence length, $d$ = representation size, $k$ = kernel size, $r$ = neighbourhood size.

| Layer type | Cost per layer | Sequential operations | Longest path between two positions |
|---|---|---|---|
| Self-attention | $O(n^2\cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n\cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k\cdot n\cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |
| Self-attention (restricted) | $O(r\cdot n\cdot d)$ | $O(1)$ | $O(n/r)$ |

*Written by me, an AI, from the paper's table text and checked against the extracted text; it is not a copy of the paper's table.*

**What each column is, and why it matters**
- **Layer type:** the kind of layer being compared: self-attention (the Transformer's), recurrent (the older way, one word after another), convolutional, and restricted self-attention (each word looks only at $r$ neighbours). It matters because the paper's whole claim is that the first beats the other two on the three measures that follow.
- **Cost per layer:** how much arithmetic one layer needs as the sentence length $n$ and the vector size $d$ grow ($k$ is the convolution window, $r$ the neighbourhood). It matters because cost limits how big a model you can afford; self-attention grows with $n^2$, which is cheap for sentences shorter than $d$ but becomes the limit for very long texts.
- **Sequential operations:** the smallest number of steps that must happen one after another. It matters because steps that depend on each other cannot run at the same time: a recurrent layer needs $O(n)$ such steps, which uses parallel hardware badly, while $O(1)$ means all words can be handled at once.
- **Longest path between two positions:** how many steps a signal must travel between the two furthest words. It matters because the paper argues (section 4) that the shorter this path, the easier it is to learn links between far-apart words.

Self-attention is cheaper than a recurrent layer when $n<d$ (paper, section 4). This is where the "square of the length" limit comes from.

**Summary: what the table shows**
- **Best on the two speed measures:** self-attention needs only $O(1)$ sequential steps and has the shortest longest path, $O(1)$, against $O(n)$ for a recurrent layer and $O(\log_k n)$ for a convolutional one. So it parallelises best and links far-apart words most directly.
- **Not the cheapest on cost per layer in every case:** its $O(n^2\cdot d)$ beats the recurrent $O(n\cdot d^2)$ only when the sentence is shorter than the vector size ($n<d$, the paper's section 4), which is usual for translation sentences.
- **The trade-off:** restricted self-attention is cheaper, $O(r\cdot n\cdot d)$, but pays with a longer path, $O(n/r)$. The $n^2$ cost is the Transformer's weak point for very long texts.

The paper's own page says that Google grants permission to reproduce its tables and figures "solely for use in journalistic or scholarly works" with attribution. That is not an open licence, so this note uses the free redrawings above and tables written by me.

## The equations (paper, sections 3.2 to 3.5 and 5.3)

**1. Scaled dot-product attention** (equation 1)

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V
$$

**Where:**
- $Q$ (queries): one row per position, "what am I looking for".
- $K$ (keys): one row per position, "what do I offer". $QK^T$ gives a score for every pair of positions.
- $V$ (values): the content that gets mixed together.
- $d_k$: the length of a query or key vector. $\sqrt{d_k}$ shrinks the scores.
- softmax: turns each row of scores into weights between 0 and 1 that add up to 1.

**Matters:** the weights decide how much of each value goes into the output. The paper says without the division by $\sqrt{d_k}$ the dot products grow large for big $d_k$ and push the softmax into regions with extremely small gradients (paper).

**Example (tiny numbers):** $d_k=2$, one query $q=(1,0)$, keys $(1,0)$ and $(0,1)$, values $(2,0)$ and $(0,4)$.
- Scores $qK^T=(1,0)$; divided by $\sqrt2$ gives $(0.707,\,0)$.
- Softmax: $e^{0.707}=2.03$ and $e^{0}=1$, so the weights are $(0.67,\,0.33)$.
- Output $=0.67\cdot(2,0)+0.33\cdot(0,4)=(1.34,\,1.32)$.
- Meaning: the query looks most like the first key, so the output leans towards the first value.
- Why scale: for random vectors with variance 1 the dot product has standard deviation $\sqrt{d_k}$. My run on 20,000 random pairs gave 2.00 at $d_k=4$, 8.03 at $d_k=64$ and 22.54 at $d_k=512$; dividing by $\sqrt{d_k}$ brings each back to 1.00.

**2. Multi-head attention** (section 3.2.2)

$$
\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)\,W^{O},\qquad \mathrm{head}_i=\mathrm{Attention}(QW_i^{Q},\,KW_i^{K},\,VW_i^{V})
$$

**Where:**
- $h$: the number of heads (8 in the paper).
- $W_i^Q$, $W_i^K$, $W_i^V$: learned matrices that shrink $Q$, $K$ and $V$ from $d_{\text{model}}$ to $d_k=d_v=d_{\text{model}}/h=64$ for head $i$.
- $\mathrm{head}_i$: the output of attention run on head $i$'s small version.
- Concat: sticks the $h$ outputs side by side; $W^O$ is a learned matrix that mixes them back to $d_{\text{model}}$.

**Matters:** each head can look at different positions and different "subspaces" at once. With a single head, the paper says, averaging inhibits this. Because every head is small, the total cost stays similar to one full-size head (paper).

**Example (tiny numbers):** $d_{\text{model}}=512$, $h=8$, so $d_k=64$.
- One head: three matrices of $512\times64$, which is $3\times32{,}768=98{,}304$ numbers.
- Eight heads: $8\times98{,}304=786{,}432$. Add $W^O$ of $512\times512=262{,}144$.
- Total $1{,}048{,}576=4\times512^2$ (my arithmetic).
- Meaning: the multi-head layer has the same number of weights as four full square matrices, however many heads you cut it into.

**3. Position-wise feed-forward network** (equation 2)

$$
\mathrm{FFN}(x)=\max(0,\,xW_1+b_1)\,W_2+b_2
$$

**Where:**
- $x$: the vector at one position (size 512).
- $W_1$, $b_1$: widen it to $d_{ff}=2048$. $W_2$, $b_2$: shrink it back to 512.
- $\max(0,\cdot)$: the ReLU, which sets negative numbers to zero.

**Matters:** attention only mixes positions; this part transforms each position by itself. Without the ReLU the two layers would collapse into one linear map. The same weights are used at every position (and differ between layers).

**Example (tiny numbers):** $x=(1,-1)$, $W_1=\begin{pmatrix}1&0&2\\0&1&-1\end{pmatrix}$, $W_2=\begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix}$, biases 0.
- $xW_1=(1,-1,3)$; ReLU gives $(1,0,3)$.
- Times $W_2$: $(1+3,\ 0+3)=(4,\,3)$.
- Meaning: the middle layer is wider, [[ReLU]] kills the negative entry, and the result is back to the original size. At the paper's size $2\times512\times2048=2{,}097{,}152$ weights (without biases, my arithmetic).

**4. Sinusoidal positional encoding** (section 3.5)

$$
\begin{aligned}
PE_{(pos,\,2i)}&=\sin\!\big(pos/10000^{2i/d_{\text{model}}}\big)\\
PE_{(pos,\,2i+1)}&=\cos\!\big(pos/10000^{2i/d_{\text{model}}}\big)
\end{aligned}
$$

**Where:**
- $pos$: the position of the word in the sentence (0, 1, 2, ...).
- $i$: which pair of dimensions; dimension $2i$ gets the sine, $2i+1$ the cosine.
- $d_{\text{model}}$: the vector size. $10000^{2i/d_{\text{model}}}$ makes low dimensions wiggle fast and high ones slowly (wavelengths from $2\pi$ to $10000\cdot2\pi$, paper).

**Matters:** attention itself ignores word order, so this adds order information. The authors chose it because they hypothesised that $PE_{pos+k}$ is a linear function of $PE_{pos}$, which helps attend by relative position, and because it may extrapolate to longer sentences (paper). Learned position embeddings gave nearly identical results (Table 3, row E).

**Example (tiny numbers):** $d_{\text{model}}=4$.
- Position 0: $(\sin0,\cos0,\sin0,\cos0)=(0,\,1,\,0,\,1)$.
- Position 1: first pair $(\sin1,\cos1)=(0.8415,\,0.5403)$; second pair with $10000^{2/4}=100$: $(\sin0.01,\cos0.01)=(0.0100,\,1.0000)$.
- Position 2: $(0.9093,\,-0.4161,\,0.0200,\,0.9998)$.
- Meaning: every position gets its own pattern; the first pair changes fast, the second slowly. The vector is added to the word embedding.

**5. Learning-rate schedule** (equation 3)

$$
\mathrm{lrate}=d_{\text{model}}^{-0.5}\cdot\min\!\big(\mathrm{step}^{-0.5},\ \mathrm{step}\cdot\mathrm{warmup}^{-1.5}\big)
$$

**Where:**
- $\mathrm{step}$: the training step number.
- $\mathrm{warmup}$: the number of warm-up steps (4000 in the paper).
- $d_{\text{model}}^{-0.5}$: scales the whole rate down for wider models.

**Matters:** the second term wins at first, so the rate rises linearly during warm-up; after the warm-up the first term wins and the rate falls like one over the square root of the step (paper). Without a warm-up the first steps would take large, unstable jumps (a general reason, not tested in the paper).

**Example (tiny numbers):** $d_{\text{model}}=512$, $\mathrm{warmup}=4000$.
- Step 100: $512^{-0.5}\cdot100\cdot4000^{-1.5}=1.75\times10^{-5}$ (rising).
- Step 4000: $6.99\times10^{-4}$ (the peak, where the two terms are equal).
- Step 100,000: $1.40\times10^{-4}$ (falling).
- Meaning: the rate climbs about 40 times in the first 4000 steps and then decays slowly.

## What they found
- 28.4 BLEU on WMT 2014 English-German (over 2 BLEU above the best earlier results including ensembles) and 41.8 BLEU English-French after 3.5 days on eight GPUs, a small fraction of earlier training cost. It also worked on constituency parsing (abstract; the German and French numbers are also in Table 2, paper).
- Training setup (paper, section 5): 8 NVIDIA P100 GPUs; base model 100,000 steps in about 12 hours (0.4 s per step); big model 300,000 steps in 3.5 days (1.0 s per step); Adam with $\beta_2=0.98$; dropout 0.1 and label smoothing 0.1 for the base model; about 4.5 million sentence pairs for English-German and 36 million for English-French.
- **A mismatch in the paper:** the abstract and Table 2 give 41.8 BLEU for English-French, but the text of section 6.1 says 41.0. I noticed it and did not find which is right.

### Translation quality and cost (paper's Table 2)

**Table 2: translation quality and training cost** (paper's Table 2; BLEU on newstest2014; cost in training floating-point operations).

| Model | BLEU EN-DE | BLEU EN-FR | Cost EN-DE | Cost EN-FR |
|---|---|---|---|---|
| GNMT + RL | 24.6 | 39.92 | $2.3\cdot10^{19}$ | $1.4\cdot10^{20}$ |
| ConvS2S | 25.16 | 40.46 | $9.6\cdot10^{18}$ | $1.5\cdot10^{20}$ |
| MoE | 26.03 | 40.56 | $2.0\cdot10^{19}$ | $1.2\cdot10^{20}$ |
| GNMT + RL ensemble | 26.30 | 41.16 | $1.8\cdot10^{20}$ | $1.1\cdot10^{21}$ |
| ConvS2S ensemble | 26.36 | 41.29 | $7.7\cdot10^{19}$ | $1.2\cdot10^{21}$ |
| **Transformer (base)** | 27.3 | 38.1 | $3.3\cdot10^{18}$ | |
| **Transformer (big)** | **28.4** | **41.8** | $2.3\cdot10^{19}$ | |

*Written by me, an AI, from the paper's table text and checked against the extracted text; it is not a copy of the paper's table.*

**What each column is, and why it matters**
- **Model:** the system being compared. "Ensemble" means several models whose answers are combined, which costs more to train and run. "Base" and "big" are the Transformer's two sizes.
- **BLEU EN-DE and EN-FR:** the translation score on the English-to-German and English-to-French test sets (newstest2014). BLEU compares a machine translation with human reference translations by counting matching groups of words (general knowledge); higher is better, with 100 as the top. It matters because it is the paper's main quality measure.
- **Training cost (FLOPs):** the estimated total floating-point operations used in training, worked out from training time, the number of GPUs and an assumed sustained speed per GPU (paper, 6.1). It matters because quality alone would hide that the Transformer reached it at a fraction of the cost. A blank cost cell in the Transformer rows is my text extraction's limit, not the paper's (see the note below the table). Here $2.3\cdot10^{19}$ means 23 followed by 18 zeros.
- **Bold rows:** the Transformer's own results.

(The paper's table also lists ByteNet, Deep-Att + PosUnk and its ensemble; I left those three rows out. Some cells are blank in the paper, and I could not tell from the text extraction which cost cells belong to the Transformer rows for English-French, so I show costs for English-German only.) My arithmetic from the table: the big Transformer's English-German cost is about 3.4 times less than the ConvS2S ensemble's and its BLEU is 2.0 higher.

**Summary: what the table shows**
- **Best quality:** the big Transformer on both language pairs: 28.4 BLEU English-German (2.0 above the best earlier model, the ConvS2S ensemble at 26.36) and 41.8 English-French (0.5 above the ConvS2S ensemble's 41.29).
- **Cheapest to train:** the base Transformer, $3.3\cdot10^{18}$ operations for English-German, about 2.9 times fewer than the cheapest earlier model listed (ConvS2S, $9.6\cdot10^{18}$), and its 27.3 BLEU is already above every earlier model in the table, ensembles included.
- **Where it is not the best:** the base model's English-French score, 38.1, is below all the earlier models in the table. Also the big model's English-German cost ($2.3\cdot10^{19}$) is about the same as the single GNMT model's and higher than ConvS2S's single model; its advantage there is quality, and against the ensembles it is about 3.4 times cheaper than the ConvS2S ensemble ($7.7\cdot10^{19}$). (All ratios are my arithmetic.)

### Which parts matter: variations of the base model (paper's Table 3)

**Table 3: variations on the base model** (paper's Table 3; English-German development set, 100K steps unless noted; "dev BLEU" and "PPL" are per-wordpiece; the last column is mine).

| Change | PPL (dev) | BLEU (dev) | Params ($10^6$) | BLEU vs base (mine) |
|---|---|---|---|---|
| base: $N=6$, $d_{\text{model}}=512$, $d_{ff}=2048$, $h=8$, $d_k=d_v=64$, dropout 0.1, label smoothing 0.1 | 4.92 | 25.8 | 65 | 0.0 |
| (A) $h=1$, $d_k=d_v=512$ | 5.29 | 24.9 | | $-0.9$ |
| (A) $h=4$, $d_k=d_v=128$ | 5.00 | 25.5 | | $-0.3$ |
| (A) $h=16$, $d_k=d_v=32$ | 4.91 | 25.8 | | 0.0 |
| (A) $h=32$, $d_k=d_v=16$ | 5.01 | 25.4 | | $-0.4$ |
| (B) $d_k=16$ | 5.16 | 25.1 | 58 | $-0.7$ |
| (B) $d_k=32$ | 5.01 | 25.4 | 60 | $-0.4$ |
| (C) $N=2$ | 6.11 | 23.7 | 36 | $-2.1$ |
| (C) $N=4$ | 5.19 | 25.3 | 50 | $-0.5$ |
| (C) $N=8$ | 4.88 | 25.5 | 80 | $-0.3$ |
| (C) $d_{\text{model}}=256$, $d_k=d_v=32$ | 5.75 | 24.5 | 28 | $-1.3$ |
| (C) $d_{\text{model}}=1024$, $d_k=d_v=128$ | 4.66 | 26.0 | 168 | $+0.2$ |
| (C) $d_{ff}=1024$ | 5.12 | 25.4 | 53 | $-0.4$ |
| (C) $d_{ff}=4096$ | 4.75 | 26.2 | 90 | $+0.4$ |
| (D) dropout 0.0 | 5.77 | 24.6 | | $-1.2$ |
| (D) dropout 0.2 | 4.95 | 25.5 | | $-0.3$ |
| (D) label smoothing 0.0 | 4.67 | 25.3 | | $-0.5$ |
| (D) label smoothing 0.2 | 5.47 | 25.7 | | $-0.1$ |
| (E) learned position embeddings | 4.92 | 25.7 | | $-0.1$ |
| big: $d_{\text{model}}=1024$, $d_{ff}=4096$, $h=16$, dropout 0.3, 300K steps | 4.33 | 26.4 | 213 | $+0.6$ |

*Written by me, an AI, from the paper's table text and checked against the extracted text; it is not a copy of the paper's table.*

**What each column is, and why it matters**
- **Change:** the one setting altered from the base model. The letters group the changes: (A) number of heads and key and value sizes, with the computation kept constant; (B) key size; (C) size and depth, with $N$ the number of layers, $d_{\text{model}}$ the vector width, $d_{ff}$ the inner size of the feed-forward network, $h$ the heads, $d_k$ and $d_v$ the key and value sizes; (D) regularisation, dropout and label smoothing; (E) the kind of position encoding. It matters because changing one thing at a time shows how much each part is worth.
- **PPL (dev):** perplexity per word piece on the development set. Roughly, it says how many equally likely words the model is torn between at each step, so lower means more confident and correct (general knowledge). The paper warns it should not be compared with per-word perplexities.
- **BLEU (dev):** the translation score on the development set (newstest2013), used here to compare variants. It is a different set from the test set of Table 2, so the numbers are not comparable across the two tables.
- **Params:** the number of trained values, in millions: a measure of model size and cost. Blank means the same as the base model's 65 million.
- **BLEU vs base (mine):** the difference from the base model's 25.8, so you can see at a glance which changes helped.

Reading it (my reading, supported by the paper's own comments in section 6.2): one head is 0.9 BLEU worse than the best setting, and too many heads also hurt; shrinking $d_k$ hurts, which the authors say suggests that judging compatibility is not easy; bigger is better; dropout helps a lot against over-fitting; learned and sinusoidal positions are nearly the same. Each row is one training run, and the paper gives no error bars, so differences of a few tenths of a BLEU point may be noise (my caution, not the paper's).

**Summary: what the table shows**
- **Best score:** the big model, 26.4 dev BLEU. Among changes to the base size, a wider feed-forward layer ($d_{ff}=4096$, 26.2) and a wider model ($d_{\text{model}}=1024$, 26.0) beat the base 25.8, but they also have more parameters (90 and 168 million against 65).
- **Biggest losses:** only 2 layers ($-2.1$), a model width of 256 ($-1.3$), no dropout ($-1.2$) and a single head ($-0.9$). So depth, width and dropout matter most here.
- **Barely matters:** 16 heads against 8 (same score), label smoothing 0.2 ($-0.1$) and learned positions instead of sinusoids ($-0.1$). Too many heads (32) hurts ($-0.4$) and a smaller key size hurts ($-0.7$ at 16).
- **Caution:** each row is one run with no error bars, so differences of a few tenths may be noise.

### Another task: parsing (paper's Table 4)

**Table 4: English constituency parsing** (paper's Table 4; F1 on section 23 of the Wall Street Journal data).

| Parser | Training | F1 |
|---|---|---|
| Dyer et al. (2016) | WSJ only, discriminative | 91.7 |
| **Transformer (4 layers)** | WSJ only, discriminative | **91.3** |
| **Transformer (4 layers)** | semi-supervised | **92.7** |
| McClosky et al. (2006) | semi-supervised | 92.1 |
| Luong et al. (2015) | multi-task | 93.0 |
| Dyer et al. (2016) | generative | 93.3 |

*Written by me, an AI, from the paper's table text and checked against the extracted text; it is not a copy of the paper's table.*

**What each column is, and why it matters**
- **Parser:** the earlier systems being compared with the Transformer.
- **Training:** what the parser learned from. "WSJ only" is about 40 thousand training sentences from the Wall Street Journal; "semi-supervised" adds about 17 million more sentences labelled by other parsers (paper, 6.3); "multi-task" and "generative" are other ways of training that I do not explain further.
- **F1:** a score that balances precision and recall, here for the pieces of sentence structure the parser got right compared with the correct trees (general knowledge); higher is better. It is measured on section 23 of the data, the held-out test part. It matters because it shows whether the Transformer works on a task other than translation.

(Six of the twelve rows; I left out others.) The authors say it does better than all earlier reported models except the recurrent neural network grammar of Dyer et al. (paper). The table shows the Transformer is not the top score here, only good without task-specific tuning.

**Summary: what the table shows**
- **Best score of all:** Dyer et al. (2016) generative, 93.3, then Luong et al. (2015) multi-task, 93.0. The 4-layer Transformer's best, 92.7 (semi-supervised), is third among the rows I show.
- **Among parsers trained on the same kind of data:** with semi-supervised data the Transformer (92.7) is the best of the semi-supervised rows in the paper's full table (the others scored 91.3 to 92.1); with WSJ only it scores 91.3, second to Dyer et al. (91.7) and above the rest (88.3 to 90.4).
- **The point:** a model built for translation did well on parsing with almost no task-specific tuning, but it is not the winner here; the authors say it beats everything except the recurrent neural network grammar.

## Why it mattered
Almost every large language model since is a Transformer.

## Limits and open questions
Cost grows with the square of the text length (Table 1, $O(n^2\cdot d)$), which led to [[FlashAttention]], [[Sliding Window Attention]] and [[KV Cache]] tricks. The authors name restricted (local) attention and non-text inputs as future work (paper).

## Concepts used
[[Self-Attention]], [[Positional Encoding]], [[Residual Connection]], [[Layer Normalization]], [[Softmax]], [[BLEU Score]]

## Reading trail
- **Came from:** [[Seq2Seq 2014]]
- **Led to:** [[BERT 2018]], [[DLinear 2022]] (questions Transformers for long-horizon forecasting), [[Scaling Laws 2020]] (studies how Transformer language models scale)

## How much I read
Body read as text: pages 1, 3 to 9 and the start of page 10 (architecture, attention, training, results, Tables 1 to 4, conclusion). I also viewed the paper's pages 3 and 4 as pictures to compare the figures. Not read: the reference list and the attention-visualisation figures in the appendix (about pages 11 to 15). The equations and numbers marked "paper" were read in the body; numbers marked "abstract" come from the abstract. The "my arithmetic" and random-vector figures come from my own Python run (NumPy). The licences of the three figures were read on their Commons pages. Not opened in Obsidian, so the flashcards are unreviewed.

## Flashcards
#flashcards/research

#flashcards/research/concept #flashcards/topic/research-nlp Attention Is All You Need (Vaswani et al., 2017, arXiv 1706.03762): what problem did it attack?::Making sequence models faster to train and better at long-range links.

#flashcards/research/concept #flashcards/topic/research-nlp What did people do before the Transformer?::Recurrent or convolutional encoder-decoders linked by an attention mechanism (seq2seq).

#flashcards/research/concept #flashcards/topic/research-nlp Transformer: what was the idea?::Use only attention and feed-forward layers, with no recurrence or convolution.

#flashcards/research/concept #flashcards/topic/research-nlp Transformer: what did it find? (abstract)::28.4 BLEU on WMT 2014 English-German (over 2 BLEU above the best earlier results including ensembles) and 41.8 BLEU English-French after 3.5 days on eight GPUs, a small fraction of earlier training cost. It also worked on constituency parsing.

#flashcards/research/concept #flashcards/topic/research-nlp Transformer: what limit did it leave?::Cost grows with the square of the text length, which led to FlashAttention, sliding window attention and KV cache tricks.

#flashcards/research/concept #flashcards/topic/research-nlp Transformer: why did it matter and what is its place in the reading trail?::Almost every large language model since is a Transformer. It came from Seq2Seq and led to BERT.

#flashcards/research/concept #flashcards/topic/research-nlp How much of this paper did the note read?::The body as text (pages 1, 3 to 9 and the start of 10); not the reference list or the appendix attention pictures. Figures are free redrawings from Wikimedia Commons, and the four tables are written by me from the paper.

#flashcards/research/concept #flashcards/topic/research-nlp Transformer paper: what is its architecture (stack sizes and sizes of the vectors)?
?
Encoder of $N=6$ identical layers (multi-head self-attention, then a feed-forward network, each wrapped in residual connection plus layer normalization) and a decoder of $N=6$ layers with a third sub-layer attending over the encoder output and masked self-attention. $d_{\text{model}}=512$, $h=8$ heads, $d_k=d_v=64$, $d_{ff}=2048$ (paper, base model).

#flashcards/research/concept #flashcards/topic/research-nlp Transformer paper: what are the three ways it uses attention?
?
Encoder-decoder attention (queries from the decoder, keys and values from the encoder); self-attention in the encoder; masked self-attention in the decoder, which stops a position from seeing later positions.

#flashcards/research/concept #flashcards/topic/research-nlp Write scaled dot-product attention and explain each symbol.
?
$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V$$
- $Q$ queries, $K$ keys, $V$ values (one row per position); $QK^T$ scores every pair.
- $d_k$: key length; dividing by $\sqrt{d_k}$ stops large dot products from pushing the softmax into tiny-gradient regions.
- softmax turns each row of scores into weights that add up to 1; they mix the values.

#flashcards/research/concept #flashcards/topic/research-nlp Scaled dot-product attention with $d_k=2$, $q=(1,0)$, keys $(1,0),(0,1)$, values $(2,0),(0,4)$: what is the output?
?
Scores $(1,0)/\sqrt2=(0.707,0)$; softmax weights $(0.67,0.33)$; output $0.67\cdot(2,0)+0.33\cdot(0,4)=(1.34,1.32)$. The query is most like the first key, so the output leans to the first value. In my random test the dot product's spread was 8.03 at $d_k=64$ and 1.00 after scaling.

#flashcards/research/concept #flashcards/topic/research-nlp Write multi-head attention and explain each symbol.
?
$$\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W^{O},\ \ \mathrm{head}_i=\mathrm{Attention}(QW_i^{Q},KW_i^{K},VW_i^{V})$$
- $h$ heads (8); $W_i^Q,W_i^K,W_i^V$ project to $d_k=d_v=d_{\text{model}}/h=64$.
- Concat joins the head outputs; $W^O$ mixes them back to $d_{\text{model}}$.
- Heads can look at different positions and subspaces; the cost stays like one full-size head.

#flashcards/research/concept #flashcards/topic/research-nlp How many weights does the multi-head layer have for $d_{\text{model}}=512$, $h=8$?
?
Per head $3\times512\times64=98{,}304$; eight heads $786{,}432$; plus $W^O$ of $512\times512=262{,}144$; total $1{,}048{,}576=4\times512^2$ (my arithmetic). Same total for any number of heads that divides 512.

#flashcards/research/concept #flashcards/topic/research-nlp Write the position-wise feed-forward network and explain each symbol.
?
$$\mathrm{FFN}(x)=\max(0,\,xW_1+b_1)\,W_2+b_2$$
- $x$: the vector at one position; $W_1,b_1$ widen it to $d_{ff}=2048$; $W_2,b_2$ shrink it back to 512.
- $\max(0,\cdot)$ is ReLU; without it the two layers collapse into one linear map.
- Same weights at every position, different weights in each layer.

#flashcards/research/concept #flashcards/topic/research-nlp FFN example: $x=(1,-1)$, $W_1$ rows $(1,0,2)$ and $(0,1,-1)$, $W_2$ rows $(1,0),(0,1),(1,1)$, biases 0. What is the output?
?
$xW_1=(1,-1,3)$; ReLU gives $(1,0,3)$; times $W_2$ gives $(4,3)$. At the paper's size the two matrices hold $2\times512\times2048=2{,}097{,}152$ weights (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-nlp Write the sinusoidal positional encoding and explain each symbol.
?
$$\begin{aligned}PE_{(pos,2i)}&=\sin\!\big(pos/10000^{2i/d_{\text{model}}}\big)\\PE_{(pos,2i+1)}&=\cos\!\big(pos/10000^{2i/d_{\text{model}}}\big)\end{aligned}$$
- $pos$: position in the sentence; $i$: index of the dimension pair; $d_{\text{model}}$: vector size.
- Low dimensions change fast, high ones slowly (wavelengths from $2\pi$ to $10000\cdot2\pi$).
- It adds order information, because attention alone ignores order. Learned positions gave nearly the same result (Table 3, row E).

#flashcards/research/concept #flashcards/topic/research-nlp Positional encoding example with $d_{\text{model}}=4$: what are positions 0 and 1?
?
Position 0: $(0,1,0,1)$. Position 1: $(\sin1,\cos1,\sin0.01,\cos0.01)=(0.8415,0.5403,0.0100,1.0000)$. The first pair changes fast, the second slowly; the vector is added to the word embedding.

#flashcards/research/concept #flashcards/topic/research-nlp Write the learning-rate schedule of the Transformer paper and explain each symbol.
?
$$\mathrm{lrate}=d_{\text{model}}^{-0.5}\cdot\min\!\big(\mathrm{step}^{-0.5},\ \mathrm{step}\cdot\mathrm{warmup}^{-1.5}\big)$$
- $\mathrm{step}$: training step; $\mathrm{warmup}=4000$; $d_{\text{model}}^{-0.5}$ scales for model width.
- Rises linearly for the first warm-up steps, then falls like the inverse square root of the step.

#flashcards/research/concept #flashcards/topic/research-nlp With $d_{\text{model}}=512$ and warm-up 4000, what is the learning rate at steps 100, 4000 and 100,000?
?
$1.75\times10^{-5}$ (rising), $6.99\times10^{-4}$ (the peak, where both terms are equal) and $1.40\times10^{-4}$ (falling) (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-nlp ![[attention-paper-fig1-transformer.png]]
What does this picture show?
?
The Transformer architecture: the encoder (left) and decoder (right), each a stack of identical layers with multi-head attention, feed-forward network and Add and Norm, plus positional encoding at the input and linear plus softmax at the output; the decoder's first attention is masked. A redrawing of the paper's Figure 1 by Yuening Jia (Wikimedia Commons, CC BY-SA 3.0).

#flashcards/research/concept #flashcards/topic/research-nlp ![[attention-paper-fig2-scaled-dot-product.png]]
What does this picture show?
?
Scaled dot-product attention: MatMul of queries and keys, Scale, optional Mask, Softmax, then MatMul with the values. A redrawing of the paper's Figure 2 (left), author unknown (Wikimedia Commons, CC BY 4.0).

#flashcards/research/concept #flashcards/topic/research-nlp ![[attention-paper-fig2-multi-head.png]]
What does this picture show?
?
Multi-head attention: Q, K and V each go through several linear layers, each set feeds its own scaled dot-product attention, the outputs are concatenated and passed through a final linear layer. A redrawing of the paper's Figure 2 (right), author unknown (Wikimedia Commons, CC BY 4.0).

#flashcards/research/concept #flashcards/topic/research-nlp In the paper's Table 1, how do self-attention, recurrent and convolutional layers compare?
?
Cost per layer: self-attention $O(n^2d)$, recurrent $O(nd^2)$, convolutional $O(knd^2)$. Sequential operations: $O(1)$, $O(n)$, $O(1)$. Longest path between two positions: $O(1)$, $O(n)$, $O(\log_k n)$. Self-attention is cheaper than recurrent when $n<d$.

#flashcards/research/concept #flashcards/topic/research-nlp In the paper's Table 2, how did the big Transformer compare with the ConvS2S ensemble?
?
English-German BLEU 28.4 against 26.36, at a training cost of $2.3\times10^{19}$ against $7.7\times10^{19}$ floating-point operations, about 3.4 times less (my arithmetic). The base model reached 27.3 at $3.3\times10^{18}$.

#flashcards/research/concept #flashcards/topic/research-nlp What did the paper's Table 3 (model variations) show?
?
Base 25.8 dev BLEU. One head: 24.9 ($-0.9$); fewer layers $N=2$: 23.7 ($-2.1$); smaller $d_{\text{model}}=256$: 24.5; dropout 0.0: 24.6; learned positions: 25.7 (nearly the same); big model: 26.4. One run per row with no error bars, so small differences may be noise.

#flashcards/research/concept #flashcards/topic/research-nlp Table 3: what happens when the key size $d_k$ is made smaller, and what do the authors suggest?
?
$d_k=16$ gave 25.1 dev BLEU and $d_k=32$ gave 25.4, against 25.8 for $d_k=64$. The authors suggest that judging compatibility is not easy and a more sophisticated function than the dot product might help.

#flashcards/research/concept #flashcards/topic/research-nlp How well did the Transformer do on parsing (Table 4)?
?
F1 on WSJ section 23: 91.3 with WSJ data only and 92.7 semi-supervised. Better than all earlier reported models except the recurrent neural network grammar (Dyer et al., 2016, 91.7 and 93.3), according to the authors.

#flashcards/research/concept #flashcards/topic/research-nlp What mismatch did I notice in the paper's English-French BLEU?::The abstract and Table 2 say 41.8, but the text of section 6.1 says 41.0; I did not find which is right.

#flashcards/research/concept #flashcards/topic/research-nlp The Transformer trained on eight P100 GPUs: how long for the base and the big model?::About 12 hours for the base model (100,000 steps) and 3.5 days for the big model (300,000 steps) (paper, section 5.2).

#flashcards/research/concept #flashcards/topic/research-nlp In the Transformer, what does N=6 mean?::The whole boxed layer is repeated 6 times, with separate weights in each repeat and the output of each layer feeding the next. This holds for both the encoder and the decoder. The embedding and positional encoding happen once, below layer 1.

#flashcards/research/concept #flashcards/topic/research-nlp What are the sub-layers of one decoder layer in the Transformer, in order?::Masked multi-head self-attention, then multi-head attention over the encoder's output (queries from the decoder, keys and values from the encoder's top layer), then the feed-forward network. Each is followed by Add and Norm. The encoder layer has only two sub-layers.

#flashcards/research/concept #flashcards/topic/research-nlp Where does the encoder's output go in the Transformer?::Its top layer's output is used as the keys and values of the middle attention sub-layer in every one of the 6 decoder layers.

#flashcards/research/concept #flashcards/topic/research-nlp How many weights do one encoder layer and one decoder layer of the base Transformer have?::Encoder layer: $1{,}048{,}576+2{,}097{,}152=3{,}145{,}728$ (attention plus feed-forward), $18{,}874{,}368$ for 6. Decoder layer: $2\times1{,}048{,}576+2{,}097{,}152=4{,}194{,}304$, $25{,}165{,}824$ for 6. My arithmetic, biases and LayerNorm left out.

#flashcards/research/concept #flashcards/topic/research-nlp Why is the decoder's self-attention masked in the Transformer?::In training the whole correct output is given at once, so without a mask a position could look at the word it must predict and copy it, and nothing would be learned; at test time the future words do not exist. The mask keeps the auto-regressive property: position $i$ may use only outputs before $i$. It is an information leak, not over-fitting.

#flashcards/research/concept #flashcards/topic/research-nlp What does "output shifted right by one position" mean in the Transformer decoder?::The decoder's input is the target sentence slid one place to the right (with a start marker in front, a general-knowledge detail), so each position receives the previous word and must predict the next one. Without the shift a position would receive the very word it must predict.

#flashcards/research/concept #flashcards/topic/research-nlp Does the Transformer decoder know the input sentence, and how?::Yes. Its middle attention sub-layer takes keys and values from the encoder's output, so it can look at every input position. The masked sub-layer hides only the future words of the output being written, not the input.

#flashcards/research/concept #flashcards/topic/research-nlp What is the difference between the Transformer's position-wise feed-forward network and a normal feed-forward network?::It is an ordinary two-layer network (512 to 2048 to 512 with a ReLU); "position-wise" means the same network is applied to each word's vector separately and identically, with no mixing between positions (attention does the mixing). A dense layer over a whole 10-word sentence would need $104{,}857{,}600$ first-layer weights against $1{,}048{,}576$, 100 times more, and fit only one length (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-nlp Why does the Transformer use N=6 layers and not 5 or 7?::The paper gives no reason; 6 is a chosen setting for both base and big models. Its Table 3 (one run each, English-German development set) gave $N=2$: 23.7 BLEU, $N=4$: 25.3, $N=6$: 25.8, $N=8$: 25.5. It did not try 5 or 7, and the 0.3 gap between 6 and 8 may be noise.

#flashcards/research/concept #flashcards/topic/research-nlp What does Add and Norm do in the Transformer?::The sub-layer's output is added to its own input (a residual connection) and normalised: LayerNorm of $x$ plus Sublayer of $x$. All sub-layers output 512 numbers so the sizes match, and dropout is applied to the sub-layer output before the addition.

#flashcards/research/concept #flashcards/topic/research-nlp What do the linear layer and softmax at the top of the Transformer decoder do?::The linear layer turns the decoder's output vector into one score per vocabulary word, and the softmax turns the scores into next-word probabilities. Its weight matrix is shared with the embedding layers.

#flashcards/research/concept #flashcards/topic/research-nlp What are the steps inside scaled dot-product attention, and what does each do?
?
1. MatMul of $Q$ and $K^T$: a score for every pair of words. 2. Scale by $\sqrt{d_k}$: keeps scores from growing with $d_k$ and saturating the softmax. 3. Optional mask: sets forbidden scores to $-\infty$ so they get weight 0 (decoder self-attention). 4. Softmax: turns each row into weights that add to 1. 5. MatMul with $V$: each word's output is the weighted blend of the values.

#flashcards/research/concept #flashcards/topic/research-nlp What are queries, keys and values in attention (a simple picture)?::Each word has a query (what it is looking for), a key (what it offers, like a label) and a value (its content). A word compares its query with every key, then reads a weighted blend of the values (the library picture is my analogy).

#flashcards/research/concept #flashcards/topic/research-nlp Scaled dot-product attention on words $(1,0)$, $(0,1)$, $(1,1)$ with $Q=K=V$ and $d_k=2$: what are the weights and outputs?
?
Weights: $(0.401,0.198,0.401)$, $(0.198,0.401,0.401)$, $(0.248,0.248,0.503)$. Outputs: $(0.80,0.60)$, $(0.60,0.80)$, $(0.75,0.75)$. With the decoder mask the weights are $(1,0,0)$, $(0.33,0.67,0)$, $(0.248,0.248,0.503)$ and the outputs $(1,0)$, $(0.33,0.67)$, $(0.75,0.75)$ (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-nlp What are the steps inside multi-head attention, and what does each do?
?
1. Project $Q$, $K$, $V$ with learned matrices into 8 smaller views of size 64 (cost stays like one full-size head). 2. Run scaled dot-product attention in each head in parallel, so each head has its own weights and can attend to different positions. 3. Concatenate the 8 outputs back to 512 numbers. 4. Mix them with the final matrix $W^O$.

#flashcards/research/concept #flashcards/topic/research-nlp Why does the Transformer use several attention heads, and what did the paper's Table 3 show?::Different heads can attend to different positions and subspaces; with one head, averaging inhibits this. Development BLEU: $h=1$: 24.9, $h=4$: 25.5, $h=8$ (base): 25.8, $h=16$: 25.8, $h=32$: 25.4. One head is 0.9 below the best, and too many heads also hurt (one run each).

#flashcards/research/concept #flashcards/topic/research-nlp In the three uses of attention in the Transformer, where do $Q$, $K$ and $V$ come from?::Encoder self-attention: all three from the previous encoder layer. Decoder self-attention: all three from the previous decoder layer, with future positions masked. Encoder-decoder attention: $Q$ from the previous decoder layer, $K$ and $V$ from the encoder's top-layer output.

#flashcards/research/concept #flashcards/topic/research-nlp For a 10-word sentence with base sizes, what are the shapes in multi-head attention?::Input $10\times512$; per head $10\times64$; scores per head $10\times10$ (800 for all 8 heads); head outputs $10\times64$; after concat $10\times512$; after $W^O$ $10\times512$, the same size as the input (my arithmetic).

#flashcards/research/concept #flashcards/topic/research-nlp In attention, why divide the scores by $\sqrt{d_k}$, and why not by $d_k$ or not at all?
?
A dot product of two random vectors with $d_k$ entries has a spread (standard deviation) of $\sqrt{d_k}$ (paper footnote 4). My run: 2.00 at $d_k=4$, 8.03 at $d_k=64$.
- **Not at all:** scores around 8 make the softmax almost one-hot: $(8,0)$ gives weights $(0.9997,0.0003)$. The softmax's slope there is $p(1-p)\approx0.0003$, so learning stalls. Dividing by $\sqrt{64}=8$ gives $(1,0)$, weights $(0.73,0.27)$, slope $0.20$, about 590 times larger (my arithmetic).
- **By $d_k$ instead:** the spread would shrink to $8/64=0.125$, so the weights would be almost equal and attention could not focus (my reasoning). Dividing by $\sqrt{d_k}$ brings the spread back to exactly 1.

#flashcards/research/concept #flashcards/topic/research-nlp In attention, why is $K$ transposed ($QK^T$)?
?
$Q$ and $K$ both have one row per word and $d_k$ columns, so $Q\cdot K$ is not even a valid matrix product (it would need $n=d_k$). $QK^T$ is $n\times d_k$ times $d_k\times n$, which gives an $n\times n$ table whose entry in row $i$, column $j$ is the dot product $q_i\cdot k_j$: how well word $i$'s query matches word $j$'s key. Each entry is one pair, all pairs at once.

#flashcards/research/concept #flashcards/topic/research-nlp In attention, why a softmax, and why is the result multiplied by $V$ and not by $K$?
?
- **Softmax:** raw scores can be negative and have any size. The softmax makes every weight positive and makes each row add up to 1, so the output is a weighted average. A big score gets a big share and a tiny score almost none, and the function is smooth, so it can be trained.
- **$V$, not $K$:** keys are labels used only for matching; values are the content that gets passed on. Keeping them separate lets what makes a word findable differ from what it hands over (general knowledge; the paper only states the formula).

#flashcards/research/concept #flashcards/topic/research-nlp In multi-head attention, why give each head its own learned projections instead of just cutting the vector into 8 pieces?
?
The paper's reason: attending with several learned views lets the model "jointly attend to information from different representation subspaces at different positions", whereas with a single head "averaging inhibits this". Each head's $W_i^Q$, $W_i^K$, $W_i^V$ is a different learned lens on all 512 numbers, so each head can find a different pattern. Heads with identical projections would compute identical attention and add nothing (my reasoning). The paper's Table 3 (A) shows one head is 0.9 BLEU worse than the best setting.

#flashcards/research/concept #flashcards/topic/research-nlp Why is each head's size $d_k=d_{\text{model}}/h=64$ in multi-head attention?
?
It keeps the cost the same as one full-size head (paper: "the total computational cost is similar to that of single-head attention with full dimensionality"). My arithmetic for the $Q$, $K$, $V$ matrices: one full head has $3\times512\times512=786{,}432$ weights; 8 heads of 64 have $8\times3\times512\times64=786{,}432$, exactly the same. So you get 8 views for the price of one.

#flashcards/research/concept #flashcards/topic/research-nlp In multi-head attention, why concatenate the heads and then multiply by $W^O$?
?
Concatenation puts the 8 results of 64 numbers side by side into 512 numbers per word without mixing them. $W^O$ (shape $hd_v\times d_{\text{model}}$, paper) then mixes the heads' results and returns a vector of size $d_{\text{model}}$, so the next step, Add and Norm, gets an input of the same size as before. Without $W^O$ the heads' information would stay in separate blocks and could not be combined (general knowledge; the paper states the shapes).

#flashcards/research/concept #flashcards/topic/research-nlp In the feed-forward network $\max(0,xW_1+b_1)W_2+b_2$, why is the ReLU needed?
?
Without it the two layers collapse into one linear layer, because $(xW_1)W_2=x(W_1W_2)$. My numbers: $x=(1,-1)$ with the note's $W_1$ and $W_2$ gives $(4,2)$ without the ReLU, but $(4,3)$ with it: the ReLU zeroes the negative middle entry and so changes the result in a way no single matrix could. The nonlinearity is what lets two layers do more than one.

#flashcards/research/concept #flashcards/topic/research-nlp Why is the feed-forward network's inner size $d_{ff}=2048$, four times wider than 512?
?
The paper states the sizes and gives no reason. Its Table 3 (C) shows the effect (English-German development set, one run each): $d_{ff}=1024$ gave 25.4 BLEU (53 million parameters), 2048 gave 25.8 (65 million) and 4096 gave 26.2 (90 million). A wider inner layer helped, at the cost of more parameters; 2048 is a chosen middle setting.

#flashcards/research/concept #flashcards/topic/research-nlp In positional encoding, why use a sine and a cosine in pairs?
?
The authors hypothesised it lets the model attend by relative position, because for any fixed offset $k$, $PE_{pos+k}$ is a linear function of $PE_{pos}$ (paper). With a sine and cosine pair this is a rotation (the angle-addition formulas). My numbers: the pair for position 1 is $(0.8415,0.5403)$; rotating it by the angle for $k=1$ gives $(0.9093,-0.4161)$, which is exactly the pair for position 2. So "one step later" is always the same rotation, wherever you are.

#flashcards/research/concept #flashcards/topic/research-nlp In positional encoding, why the number 10000 and the exponent $2i/d_{\text{model}}$?
?
They set how fast each pair of dimensions oscillates. Pair $i$ has wavelength $2\pi\cdot10000^{2i/d_{\text{model}}}$, so the wavelengths form a geometric progression from $2\pi$ to $10000\cdot2\pi$ (paper). Like the digits of a counter, some dimensions change fast (tell neighbouring words apart) and some very slowly (tell far-apart positions apart). With $d_{\text{model}}=4$ the two wavelengths are 6.28 and 628.3 positions (my arithmetic). The paper does not say why 10000; it only needs to be larger than the sentences it handles (my reasoning).

#flashcards/research/concept #flashcards/topic/research-nlp Why is the positional encoding added to the embedding and not attached to it?
?
The paper gives the reason: it has the same dimension as the embedding "so that the two can be summed". Adding keeps every vector at 512 numbers, so the rest of the model needs no change. Attaching it (concatenating) would make the vectors longer and every later matrix bigger (general knowledge).

#flashcards/research/concept #flashcards/topic/research-nlp In the Transformer learning-rate formula, why a minimum of two terms (a warm-up)?
?
Both terms are equal exactly at step = warm-up (4000), so the minimum switches there: before it the rate rises linearly, after it the rate falls like $1/\sqrt{\text{step}}$ (paper). The peak is $6.99\times10^{-4}$. Without a warm-up the very first step would use a rate of $0.0442$, about 250,000 times bigger than the warm-up's first step ($1.75\times10^{-7}$) (my arithmetic). Starting small because the weights are still random and large steps can be unstable is general knowledge; the paper does not give a reason.

#flashcards/research/concept #flashcards/topic/research-nlp In the Transformer learning-rate formula, why $d_{\text{model}}^{-0.5}$ and the inverse-square-root decay?
?
- **$d_{\text{model}}^{-0.5}$:** a wider model gets a smaller rate. Going from 512 to 1024 multiplies the rate by 0.707 (my arithmetic). The paper states it without a reason.
- **Inverse square root:** the rate keeps falling but slowly: from step 4000 to 100,000 is 25 times more steps, but the rate falls only 5 times ($\sqrt{25}$), from $6.99\times10^{-4}$ to $1.40\times10^{-4}$ (my arithmetic). Big steps early to move fast, small steps later to settle (general knowledge).
