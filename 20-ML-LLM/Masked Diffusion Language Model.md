---
tags: [LLM, diffusion, masked-diffusion, generation, non-autoregressive, LLaDA]
status: growing
level: 4
tier: 4
---

# Masked Diffusion Language Model

Often written **MDM** or "diffusion LLM". A language model that writes text by **starting with every word hidden and revealing several at a time**, instead of one word after another.

## 🟢 1. Simple (age 12)
Normal chatbots write like a person typing: one word, then the next, always left to right. A diffusion language model works like **un-covering a sentence**: it starts with a row of blank boxes and fills in the words it is most sure about first, several at once, and keeps going until nothing is left blank. Fewer rounds are needed, so it can be faster.

**Example:** A crossword with 8 empty squares. A normal writer fills in square 1, then 2, then 3. A diffusion writer first fills in the two squares it is most confident about (maybe squares 3 and 8), then the next two, and finishes in 4 rounds instead of 8.

## 🔵 2. How it works
1. A usual [[LLM]] is **autoregressive**: it predicts one token at a time using only earlier tokens. Writing cannot be parallelised, because each token waits for the one before (training can be, but writing cannot). That slows answers down.
2. **Diffusion** is another recipe, first made famous for images ([[Diffusion Models]]): add noise to a picture step by step, then learn to remove it. Generation starts from noise and cleans it up.
3. For text there is no smooth "noise", so the equivalent is **masking**: hide each token with some probability. The **forward process** hides more and more tokens, until all are hidden.
4. The model is trained to be the **reverse process**: given a text with some tokens hidden, predict the hidden ones. This is like the fill-in-the-blank game of [[BERT]], at all levels of hiding.
5. To write, start with all positions hidden (the prompt stays visible), let the model predict all hidden tokens, keep some of the predictions, and leave the others hidden. Repeat.
6. Fewer passes: with several tokens fixed per pass, the number of model calls is the length divided by the tokens per pass.
7. Per the lecture, there are claims of up to about **10 times more output tokens per second** compared with autoregressive models, and that it may suit some tasks better. The challenges are adapting tricks built for autoregressive models (like the [[KV Cache]]) and matching their quality.

## 🟠 3. Technical detail
- **The lecture's setup:** a **forward process** masks tokens with some probability; a **reverse process** learns to unmask them. The lecture calls this a masked diffusion model and says decoding is done in fewer forward passes.
- **Training objective:** a weighted fill-in-the-blank loss: the loss on hidden tokens is weighted by one over the hiding probability and averaged over random hiding levels. The LLaDA abstract says it optimises a **likelihood lower bound**. The simple masked diffusion paper describes its objective as a mixture of classical masked language modelling losses (abstracts read).
- **LLaDA:** an 8-billion-parameter diffusion model trained from scratch with pre-training and supervised fine-tuning; its abstract reports it is competitive with LLaMA3 8B in in-context learning and shows instruction-following in case studies (arXiv 2502.09992, abstract read).
- **Another line:** *Simple and Effective Masked Diffusion Language Models* says masked diffusion is more performant than previously thought and approaches autoregressive perplexity ([[Perplexity]]) (arXiv 2406.07524, abstract read).
- **Bidirectional:** the model can look at tokens on both sides of a blank, like [[BERT]], which is unlike a left-to-right model.
- **Which tokens to reveal:** the code's toy picks the positions it is most sure about. Real systems use schemes I did not study.
- **Open issues (lecture):** reusing techniques from autoregressive models, and performance. The lecture also says modalities are trading ideas: "diffusion" training for text, Transformers for images, and so on.
- **News (lecture):** the slides list several diffusion-LLM announcements from 2025; I did not check them.

```python
import numpy as np
rng = np.random.RandomState(0)
n, V = 8, 50                                                    # a text of 8 tokens, a vocabulary of 50
# 1) forward process: hide each token with chance t
for t in (0.1, 0.5, 0.9):
    masked = np.mean([(rng.rand(n) < t).sum() for _ in range(20000)]) / n
    print(f't = {t} | share of tokens hidden {masked:.3f}')
# 2) the training loss: the model guesses the hidden tokens; weight the loss by 1/t. For a model that knows nothing (uniform guess)
#    every hidden token costs ln(V). The weighted average over random t should equal the true cost of the text: n * ln(V)
ts = rng.rand(200000)
loss = np.array([(rng.rand(n) < t).sum() * np.log(V) / t for t in ts])
print('mean of (1/t) * (sum over hidden tokens of -log p) :', round(float(loss.mean()), 3), '| n * ln(V) =', round(n * np.log(V), 3))
# 3) generation: start with every token hidden and reveal several per step, against one token per step
for n_tokens in (64, 512):
    for per_step in (1, 4, 8):
        print(f'{n_tokens:4d} tokens, {per_step} per step -> {int(np.ceil(n_tokens / per_step)):4d} model passes')
# 4) a tiny reveal run: at each step the 'model' is confident about some positions (random confidences here), reveal the most confident 2
conf = rng.rand(8); text = ['_'] * 8; step = 0
while '_' in text:
    hidden = [i for i, c in enumerate(text) if c == '_']
    for i in sorted(hidden, key=lambda i: -conf[i])[:2]: text[i] = 'abcdefgh'[i]
    step += 1; print('step', step, ''.join(text))
```

**Would I choose a diffusion model for a latency-sensitive job? (what the numbers can and cannot say).** Latency is the wait for the answer, and it comes from two things multiplied together: how many passes the model must make one after another, and how long one pass takes. A diffusion model cuts the first. It may raise the second, because each pass looks at the whole text instead of one new token, and the [[KV Cache]] trick that makes one autoregressive pass cheap is one of the things the lecture lists as hard to reuse. The toy below counts both:

```python
import math

# toy count, not a timing: a model "pass" runs all the positions it is given at once on a GPU
n = 512                                                # tokens to write
print("tokens written:", n)
print("autoregressive : passes =", n, "| positions computed per pass = 1 (KV cache holds the old ones)")
for k in (1, 4, 8):
    passes = math.ceil(n / k)
    print(f"diffusion, {k} tokens fixed per pass: passes = {passes:4d} | positions per pass = {n} | total positions computed = {passes * n:,}")
print("autoregressive total positions computed:", n)
```

What it printed: for 512 tokens, autoregressive needs 512 passes of 1 position each. Diffusion at 8 tokens per pass needs 64 passes of 512 positions each, which is 32,768 positions in total, 64 times more work than the 512, but in 8 times fewer steps one after another. On a GPU the positions inside one pass run side by side, so fewer sequential passes can win on wall-clock time even though total work is higher. That "can" depends on the hardware, the text length and how many tokens can be fixed per pass without hurting quality.

**My answer from the evidence in this note:** I would **not** pick a diffusion model for latency yet on this evidence alone. The lecture claims about 10 times more tokens per second, and the LLaDA abstract only claims it is competitive with LLaMA3 8B on in-context learning, so no source I read measures latency head to head, and I ran no timing. For a real choice I would measure time per output token for both on my own hardware and text lengths, and check answer quality at the chosen tokens per pass. Short answers, where the cached autoregressive model needs few passes anyway, are the case where diffusion has the least to gain.

## 🔴 4. Deep / formal
**The training loss.** With $t\in(0,1)$ the chance each token is hidden, $x$ a text of $n$ tokens, $x_t$ the partly hidden text, and $\mathcal{M}$ the set of hidden positions:

$$
\mathcal{L}=\mathbb{E}_{t\sim U(0,1)}\;\mathbb{E}\!\left[\frac{1}{t}\sum_{i\in\mathcal{M}}-\log p_\theta\big(x_i\mid x_t\big)\right]
$$

**Where:**
- $t$ is the masking level, drawn uniformly between 0 and 1 for each training text.
- $x_t$ is the text after hiding each token with chance $t$, and $\mathcal{M}$ is the set of positions that got hidden.
- $-\log p_\theta(x_i\mid x_t)$ is the cost of the model's guess for the true token at hidden position $i$.
- $\tfrac1t$ weights the loss so that lightly masked texts (few hidden tokens) count as much as heavily masked ones.
- **Matters:** the $1/t$ makes the average equal to a bound on the true negative log-likelihood of the text (per the LLaDA abstract, "a likelihood lower bound"; the exact form here is from my memory of this family of papers and I did not derive it). Without it, the training would favour only the heavy-mask cases.

**Example (tiny numbers):** a "model" that knows nothing and guesses uniformly over a vocabulary of $V=50$.
- Each hidden token costs $\ln50=3.912$.
- On average a fraction $t$ of the $n=8$ tokens is hidden, so the expected sum is $t\cdot8\cdot3.912$, and dividing by $t$ gives $8\times3.912=\mathbf{31.30}$ for every $t$.
- The code averaged this over 200,000 random draws and got $\mathbf{31.318}$, against the exact $n\ln V=\mathbf{31.296}$.
- Meaning: for the knows-nothing model the weighted loss equals the true cost of the text, which is what a likelihood bound should do when it is tight.

**Number of passes.** To write $n$ tokens while revealing $s$ tokens per pass:

$$
\text{passes}=\left\lceil\frac{n}{s}\right\rceil
$$

**Where:**
- $n$ is the number of tokens to write and $s$ is how many tokens are fixed in each model pass.
- $\lceil\cdot\rceil$ rounds up to a whole number of passes.
- **Matters:** an autoregressive model has $s=1$, so it needs $n$ passes. Larger $s$ means fewer passes but, if too many tokens are fixed at once, they are fixed without seeing each other and quality can fall (my reading; not tested here).

**Example (tiny numbers):** $n=512$ tokens.
- $s=1$: $\mathbf{512}$ passes. $s=4$: $\mathbf{128}$. $s=8$: $\mathbf{64}$.
- Meaning: revealing 8 tokens per pass would cut the model calls by 8 times. The lecture's 10 times figure is a different claim, about tokens per second, and I did not check it.

**Further notes:**
- **The reveal run in the code:** with made-up confidences the 8 positions are filled in 4 steps of 2 (step 1 `__c____h`, step 2 `abc____h`, step 3 `abc_ef_h`, step 4 `abcdefgh`). It shows the shape of the process only, there is no real model.
- **Cost per pass:** a pass over the whole text is more expensive than an autoregressive pass with a cache, so fewer passes does not automatically mean less time. I did not measure it.
- **Not covered:** the mathematics of the reverse process, sampling schedules, and the reversal-curse results of LLaDA.

**What came before, and how this improved on it:**
- **Before:** autoregressive language models, left to right, one token per pass (the standard since the [[Transformer]] and GPT-style models).
- **Its limit:** the lecture says writing is not parallelisable, which limits speed. The LLaDA abstract frames the common assumption that core LLM abilities depend on autoregression.
- **What changed:** train a masked-denoising model and write by revealing many tokens per pass (Nie et al., arXiv 2502.09992, 14 February 2025; Sahoo et al., arXiv 2406.07524, 11 June 2024; abstracts read).
- **How much better:** the lecture claims about 10 times more output tokens per second than autoregressive models (slide, not verified). LLaDA's abstract claims it is competitive with LLaMA3 8B on in-context learning, not that it beats it overall. In this note's code, 512 tokens need 512 passes one at a time and 64 passes at 8 per pass, which counts model calls and not time.

## Related
- [[Diffusion Models]] · [[LLM]] · [[BERT]] · [[Transformer]] · [[KV Cache]] · [[Speculative Decoding]] · [[Decoding Strategies]] · [[Perplexity]] · [[Self-Supervised Learning]] · [[Vision Transformer]] · [[Autoregressive Generation]]

## Sources
- Stanford CME 295 (Fall 2025), Lecture 9 slides, Afshine and Shervine Amidi
- Nie et al., *Large Language Diffusion Models*, 2025 (arXiv 2502.09992, abstract read)
- Sahoo et al., *Simple and Effective Masked Diffusion Language Models*, 2024 (arXiv 2406.07524, abstract read)
- Ho et al., *Denoising Diffusion Probabilistic Models*, 2020 (named in the lecture; not read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/generative Masked diffusion language model:::Instead of writing left to right, start with a row of blank boxes and fill in the words it is most sure about first, several at once, until nothing is left blank.

#flashcards/ml/formula #flashcards/topic/generative Write the masked-diffusion training loss and explain each symbol.
?
$$\mathcal{L}=\mathbb{E}_{t\sim U(0,1)}\;\mathbb{E}\!\left[\frac{1}{t}\sum_{i\in\mathcal{M}}-\log p_\theta\big(x_i\mid x_t\big)\right]$$
- $t$: the masking level, drawn uniformly between 0 and 1 for each training text.
- $x_t$: the text after hiding each token with chance $t$. $\mathcal{M}$: the set of positions that got hidden.
- $-\log p_\theta(x_i\mid x_t)$: the cost of the model's guess for the true token at hidden position $i$.
- $\tfrac1t$: weights the loss so lightly masked texts (few hidden tokens) count as much as heavily masked ones. It makes the average a bound on the true negative log-likelihood of the text (per the LLaDA abstract).

#flashcards/ml/example #flashcards/topic/generative A model guesses uniformly over $V=50$ words on $n=8$ tokens. What is the weighted loss?::Each hidden token costs $\ln50=3.912$. On average a fraction $t$ of the 8 tokens is hidden, so the sum is $t\cdot8\cdot3.912$, and dividing by $t$ gives $8\times3.912=31.30$ for every $t$. The note's code averaged 31.318 against the exact $n\ln V=31.296$.

#flashcards/ml/formula #flashcards/topic/generative Write the number of model passes for a text and explain each symbol.
?
$$\text{passes}=\left\lceil\frac{n}{s}\right\rceil$$
- $n$: the number of tokens to write. $s$: how many tokens are fixed in each model pass.
- $\lceil\cdot\rceil$: rounds up to a whole number of passes.
- An autoregressive model has $s=1$, so it needs $n$ passes. Larger $s$ means fewer passes but, if too many tokens are fixed at once, they are fixed without seeing each other and quality can fall (the note's own reading, not tested).

#flashcards/ml/example #flashcards/topic/generative $n=512$. How many passes for $s=1$, 4 and 8?::512, 128 and 64. Revealing 8 tokens per pass cuts the model calls 8 times. The lecture's "10 times" figure is a different claim (tokens per second), which the note did not check.

#flashcards/ml/cloze #flashcards/topic/generative A masked diffusion model fills in the ==most sure== words first, several at once.
