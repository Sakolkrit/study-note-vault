---
tags: [LLM, training, inference, decoding, speculative]
status: growing
level: 4
tier: 3
---

# Multi-Token Prediction

**Multi-token prediction** trains a language model to guess several future tokens at once, not only the next one.

## 🟢 1. Simple (age 12)
A normal language model learns by guessing the very next word. Multi-token prediction asks it to guess the next few words in one go. That makes it plan a little further ahead, and it can also let the model write several words in one step.

**Example:** Instead of asking a student "what is the next word?", you ask "what are the next four words?". To answer, they must think about where the sentence is going.

## 🔵 2. How it works
1. A standard language model is trained with next-token prediction: at each position, guess token number $t+1$ ([[Self-Supervised Learning]], [[Pre-training]]).
2. In multi-token prediction, at each position the model guesses the next $n$ tokens at once, with $n$ separate small output heads sitting on top of one shared main body (the "trunk").
3. Training on the extra guesses is an auxiliary task: it pushes the shared body to hold information useful for the future, not just the next step.
4. At test time you can ignore the extra heads and use the model as normal.
5. Or you can use them to **draft** several tokens at once and check them with the main head, like [[Speculative Decoding]], so more than one token is written per step.
6. The paper's abstract reports better coding results and up to 3 times faster inference for a 4-token model.
7. **The catch:** the gain depends on how often the extra guesses are right, and the training gains are reported to grow with model size, so small models may not benefit (abstract: "increasingly useful for larger model sizes").

## 🟠 3. Technical detail
- **Architecture (paper, Section 2):** a shared transformer trunk produces one hidden representation, then $n$ independent output heads (each made of transformer layers) predict the next $n$ tokens, all through one shared unembedding matrix. The first head is the ordinary next-token head.
- **Loss (paper):** the next-token loss is replaced by the sum, over the $n$ heads, of the loss of predicting token $t+i$ from the shared representation (the equation is in level 4). At inference you can keep only the first head.
- **Reported effects (paper):** the introduction claims no train-time or memory overhead, and models up to 13B parameters solving around 15% more code problems on average. The abstract gives 12% more HumanEval and 17% more MBPP problems for 13B models than comparable next-token models. The gain grows with model size (the paper's Figure 1 shows MBPP pass@1 improving "significantly so as model size increases"), and the authors suggest that needing scale is a likely reason the idea was overlooked before.
- **Memory trick (paper, Figure 2):** the vocabulary is far larger than the hidden size, so the extra heads would make the output layer's gradients big. The paper runs the forward and backward pass head by head, so it never holds all the output-layer gradients at once, which lowers peak GPU memory.
- **Measured speed-up (Section 3.2):** with greedy self-speculative decoding on a 7B model trained with 4-token prediction, the paper reports a 3.0 times speed-up on code, with an average of 2.5 accepted tokens out of 3 suggestions, and 2.7 times on text. An 8-byte prediction model reached 6.4 times (Table S3).
- **Link to decoding speedups:** extra heads give guesses for later positions, which can be verified in one pass, the idea behind [[Speculative Decoding]]. See also [[Decoding Strategies]].
- **Where it appears:** the lecture slides mention multi-token prediction among the training and efficiency tricks of recent models (the slide name I found is only the term; I did not read its detail).
- **Cost:** a few more output layers, so slightly more parameters; the paper says the implementation above avoids extra training time and memory.

```python
import numpy as np
rng = np.random.default_rng(0)
def passes_needed(total, k, a):                                         # each pass proposes k tokens; tokens are kept while the guesses stay right
    done = passes = 0
    while done < total:
        run = 1                                                         # the first token of a pass is always correct
        while run < k and rng.random() < a: run += 1
        done += run; passes += 1
    return passes
a = 0.8
print(f"assumed chance that each extra guessed token is right: {a}")
for k in (1, 2, 4):
    sim = 20000 / np.mean([passes_needed(20000, k, a) for _ in range(3)]); exact = (1 - a ** k) / (1 - a)
    print(f"  {k} tokens per pass: {sim:.2f} tokens per pass by simulation, {exact:.2f} by the formula")
```

## 🔴 4. Deep / formal
**Tokens per pass.** If each extra guessed token is right with chance $a$ (and the guesses are checked in order), a pass that offers $k$ tokens keeps, on average,

$$
E=\frac{1-a^{\,k}}{1-a}\ \text{tokens per pass}
$$

**Where:**
- $k$ is the number of tokens offered per pass (1 for a normal model). $a$ is the chance that each extra guess is correct, given that the earlier ones were.
- The first token of every pass is always kept (it is the normal next-token answer). The second is kept with chance $a$, the third with chance $a^2$, and so on.
- $E=1+a+a^2+\dots+a^{k-1}$ adds those chances. **Matters:** more than $k$ times speed-up is impossible, and a low $a$ makes the extra heads almost useless; with $a$ near 1 the gain approaches $k$.

**Example (tiny numbers):** $a=0.8$.
- $k=1$: $E=\mathbf{1}$ (no gain).
- $k=2$: $E=1+0.8=\mathbf{1.8}$.
- $k=4$: $E=1+0.8+0.64+0.512=\mathbf{2.952}$.
- Meaning: offering 4 tokens per pass with an 80% chance for each extra guess gives about 2.95 tokens per pass, close to the "up to 3 times faster" in the abstract. The value $0.8$ is my assumption, not a figure from the paper.

**The loss in symbols.** Instead of scoring only the next token, score the next $n$ tokens, each from the same shared hidden representation $z_t$:

$$
L_n=-\sum_{t}\sum_{i=1}^{n}\log P_\theta\big(x_{t+i}\mid z_t\big)
$$

**Where:**
- $t$ runs over positions in the training text. $x_{t+i}$ is the token $i$ steps after position $t$. $n$ is the number of future tokens predicted.
- $z_t$ is the hidden representation that the shared trunk computes from everything up to position $t$. Head $i$ turns it into a chance for token $t+i$.
- $\log P_\theta(\cdot)$ is the log of the chance the model gave to the true token, so each term is the usual cross-entropy piece ([[Loss Function]]). With $n=1$ this is the ordinary next-token loss.
- **Matters:** all heads push on the same trunk, so the trunk is trained to hold what is needed several tokens ahead, not only for the next one.

**Example (tiny numbers):** $n=2$, one position, and the model gives the true next token a chance of $0.5$ and the true token after it a chance of $0.25$.
- Term for $i=1$: $-\ln0.5=\mathbf{0.693}$.
- Term for $i=2$: $-\ln0.25=\mathbf{1.386}$.
- $L_2$ for this position $=0.693+1.386=\mathbf{2.079}$.
- Meaning: the ordinary loss would be only the first term, $0.693$. The second head adds a harder question, so the trunk gets extra training signal.

**Further notes:**
- **Why it may help training:** to predict the token after next, the model must keep information about the future in its hidden state, which the paper argues helps induction heads and algorithmic reasoning (abstract; I did not read the small algorithmic experiments themselves).
- **Open question:** how large $n$ should be. The paper's real-data results centre on 4-token prediction (and byte-level models with larger $n$); how well it holds for other tasks is not something I read.

**What came before, and how this improved on it:**
- **Before:** next-token prediction only: one guess per position in training and one token per pass when writing.
- **Its limit:** each pass of a big model produces just one token, so writing is slow; and the training signal only looks one token ahead.
- **What changed:** $n$ output heads on a shared trunk predict the next $n$ tokens (Gloeckle et al., Meta, arXiv 2404.19737, 30 April 2024; abstract and parts of the body read; the first author's name is checked on the paper's title page).
- **How much better:** from the paper (copied by hand): up to about 15% more code problems solved on average for models up to 13B parameters (12% on HumanEval and 17% on MBPP for the 13B models); a 3.0 times decoding speed-up on code and 2.7 times on text for a 4-token 7B model, with 2.5 accepted tokens out of 3 suggestions on code. I reproduced none of this. In the code below (my own simulation with an assumed 0.8 chance for each extra guess), 4 tokens per pass keep 2.96 tokens per pass by simulation (2.95 by formula), against 1.00 for one token. The paper's 2.5 of 3 suggestions accepted is about 83% per suggestion by my arithmetic (not its definition), which is close to the assumed 0.8. **Honest limits:** the paper says its gains need scale, and a toy cannot show a training effect.

**Check it yourself:** run the code in level 3 above. It prints every number quoted in this block.

## Related
[[Speculative Decoding]] · [[Decoding Strategies]] · [[Self-Supervised Learning]] · [[Pre-training]] · [[Loss Function]] · [[Transformer]] · [[LLM]] · [[Benchmarks]] · [[Pass at k]]

## Sources
- Stanford CME 295 (Fall 2025), lecture slides, Afshine and Shervine Amidi (lectures 3 and 4, where the term appears)
- Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction*, arXiv 2404.19737 (2024): abstract and parts of the body read (introduction, Section 2 on the method, Section 3.2 on speed, the contributions list), taken from an HTML copy of the paper. The numbers above were copied by hand.
- No image: I found no free, on-topic picture for this note on Wikimedia Commons, so I added none.
- Every number in the history block was computed by the code in this note (toy data; dates and names marked "from memory" are not checked).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm Multi-token prediction:::Train the model to guess the next few words at once. It plans a little further ahead and can write several words in one step.

#flashcards/ml/formula #flashcards/topic/llm Write the expected tokens kept per pass and explain each symbol.
?
$$E=\frac{1-a^{\,k}}{1-a}\ \text{tokens per pass}$$
- $k$: the number of tokens offered per pass (1 for a normal model).
- $a$: the chance that each extra guess is correct, given the earlier ones were.
- The first token of every pass is always kept. The second is kept with chance $a$, the third with $a^2$, and so on. $E=1+a+a^2+\dots+a^{k-1}$.
- A speed-up of more than $k$ times is impossible. A low $a$ makes the extra heads almost useless. With $a$ near 1 the gain approaches $k$.

#flashcards/ml/example #flashcards/topic/llm $a=0.8$. What is $E$ for $k=1$, 2 and 4?::$k=1$: 1. $k=2$: $1+0.8=1.8$. $k=4$: $1+0.8+0.64+0.512=2.952$, close to the abstract's "up to 3 times faster". The value 0.8 is the note's assumption, not a paper figure.

#flashcards/ml/formula #flashcards/topic/llm Write the multi-token prediction training loss and explain each symbol.
?
$$L_n=-\sum_{t}\sum_{i=1}^{n}\log P_\theta\big(x_{t+i}\mid z_t\big)$$
- $t$: runs over positions in the training text. $x_{t+i}$: the token $i$ steps after position $t$. $n$: the number of future tokens predicted.
- $z_t$: the hidden representation the shared trunk computes from everything up to position $t$. Head $i$ turns it into a chance for token $t+i$.
- $\log P_\theta(\cdot)$: the log of the chance the model gave the true token, the usual cross-entropy piece. With $n=1$ this is the ordinary next-token loss.
- All heads push on the same trunk, so the trunk gets extra training signal.

#flashcards/ml/example #flashcards/topic/llm $n=2$, the true next token has chance 0.5 and the one after it 0.25. What is $L_2$ for this position?::$-\ln0.5=0.693$ and $-\ln0.25=1.386$. $L_2=2.079$. The ordinary loss would be only 0.693. The second head adds a harder question.
