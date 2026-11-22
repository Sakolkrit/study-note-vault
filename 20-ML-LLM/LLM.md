---
tags: [LLM, transformer, NLP, generative-AI, deep-learning]
status: growing
level: 4
tier: 4
---

# LLM

## 🟢 1. Simple (age 12)
LLM stands for **Large Language Model**. It is a computer program that has read a gigantic amount of text and learned to **guess the next word**, over and over, very well.

**Example:** If you type "Peanut butter and ...", you probably guess "jelly". An LLM does the same trick, but it has seen so much text that it can keep guessing word after word and end up writing a whole essay, a poem, or code.

## 🔵 2. How it works
1. **Training (learning):** show the model billions of sentences with the last word hidden. It guesses, gets told the right answer, and adjusts its numbers a tiny bit. Repeat trillions of times.
2. **After training:** the model is a giant pile of numbers (called *parameters*). Bigger piles can remember more patterns.
3. **Using it:** you type a prompt. The model guesses the next small piece of text, adds it, and guesses again, until the answer is done.
4. **Making it helpful:** extra lessons teach it to follow instructions and answer politely, like a good assistant.

Important: an LLM predicts likely text. It can sound sure and still be **wrong**. Giving it the right pages to read first ([[RAG]]) helps.

![[llm-training-stages.svg]]
*Three stages of building an LLM: pre-training, then fine-tuning, then alignment. Image: Kjerish, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Three-stage_large_language_model_training_workflow.svg), CC0.*

## 🟠 3. Technical detail
Almost every LLM is a decoder-only [[Transformer]] that models

$$
p(x_1,\dots,x_n)=\prod_{t=1}^{n} p(x_t\mid x_{<t})
$$

**Where:**
- $x_1,\dots,x_n$ are the tokens of a text in order; $n$ is how many there are; $t$ is the position being looked at.
- $p(x_1,\dots,x_n)$ is the probability of the *whole* text. **Matters:** a good model gives real, sensible text a high probability and gibberish a low one.
- $x_{<t}$ means "all tokens before position $t$". $p(x_t\mid x_{<t})$ is "the chance of this token, given everything before it". **Matters:** this is the one thing the model actually computes: next-token prediction.
- $\prod_{t=1}^{n}$ is a big multiplication over every position. **Matters:** chaining the next-word guesses together gives the probability of the full text, so predicting one token at a time is enough to model all of language.

**Example (tiny numbers):** the text "I love cats" has 3 tokens.
- $p(\text{I})=0.1$ (a common way to start).
- $p(\text{love}\mid\text{I})=0.2$.
- $p(\text{cats}\mid\text{I love})=0.05$.
- Whole text: $0.1\times0.2\times0.05=\mathbf{0.001}$.
- A model that never saw odd word orders would give "cats I love" a lower number. Training pushes real, natural text up.

Pipeline:
- **Tokenization:** text becomes integer token IDs ([[Tokenization]]).
- **Embedding:** IDs become vectors ([[Embedding]]).
- **Transformer stack:** N blocks of attention + MLP.
- **Output head:** a score (logit) for every token in the vocabulary, softmaxed to probabilities.
- **Sampling:** pick the next token (greedy, or with temperature / top-p), append, repeat. A KV cache avoids recomputing past tokens.

Training stages (as in the figure):
1. **Pre-training** on web-scale text, next-token loss. Gives knowledge and language skill.
2. **Supervised fine-tuning (SFT)** on instruction/answer pairs. Gives instruction following.
3. **Alignment** with human or AI preference data (RLHF, DPO, ...). Gives helpfulness and safety.

Using one: prompting, in-context examples, tool use, [[RAG]], fine-tuning (often LoRA, which trains small adapter matrices only).

```python
# tiny sampling loop (model returns logits for the next token)
import numpy as np
def generate(model, ids, n, temp=0.8):
    for _ in range(n):
        logits = model(ids)[-1] / temp
        p = np.exp(logits - logits.max()); p /= p.sum()
        ids.append(int(np.random.choice(len(p), p=p)))
    return ids
```

## 🔴 4. Deep / formal
Pre-training minimises cross-entropy over a dataset $\mathcal{D}$:

$$
\mathcal{L}(\theta)=-\mathbb{E}_{x\sim\mathcal{D}}\sum_{t}\log p_\theta(x_t\mid x_{<t})
$$

**Where:**
- $\mathcal{L}(\theta)$ is the **loss**: a single number for how badly the model predicts text. Lower is better; training shrinks it with [[Gradient Descent]].
- $\theta$ (theta) stands for all the model's learned numbers (its parameters, often billions). **Matters:** these are the only things training is allowed to change.
- $\mathcal{D}$ is the training dataset and $x\sim\mathcal{D}$ means "a text $x$ drawn from it". $\mathbb{E}$ is the average over many such texts. **Matters:** we judge the model on typical text, not one lucky sentence.
- $\sum_t$ adds up the error at every token position $t$ in the text.
- $\log p_\theta(x_t\mid x_{<t})$ is the log of the probability the model gave to the *correct* next token. **Matters:** if the model was confident and right, the log is near 0; if it gave the right token a tiny chance, the log is very negative.
- The leading minus sign flips this so that "confident and right" means small loss and "surprised by the right answer" means large loss.

**Example (tiny numbers):** a 3-token text. The probabilities the model gave to the *correct* next token were $0.5,\ 0.25,\ 0.9$.
- $-\log(0.5)=0.69$ (unsure, small penalty).
- $-\log(0.25)=1.39$ (quite surprised, bigger penalty).
- $-\log(0.9)=0.11$ (confident and right, tiny penalty).
- Sum $=\mathbf{2.18}$, or $0.73$ per token. Training changes $\theta$ so the next round gives a lower sum.

**Scaling laws** (Kaplan et al., 2020; Hoffmann et al., 2022): loss falls as a power law in parameters $N$ and data $D$, roughly

$$
L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}
$$

**Where:**
- $L$ is the final loss the model reaches after training.
- $N$ is the number of model parameters (the model's size) and $D$ is the amount of training data in tokens.
- $E$ is the **irreducible loss**: text is partly unpredictable, so even a perfect model cannot reach 0. **Matters:** it is the floor that scaling alone can never break through.
- $\dfrac{A}{N^{\alpha}}$ is the penalty for the model being too small. As $N$ grows this term shrinks. $A$ and $\alpha$ are constants fitted from experiments; $\alpha$ is small (well under 1), so each doubling of size gives a smaller gain than the last.
- $\dfrac{B}{D^{\beta}}$ is the penalty for too little data. $B$ and $\beta$ are fitted constants in the same way.
- **Matters:** the two penalty terms say that bigger model *and* more data both help, and spending only on one wastes the other. That is how researchers pick the best balance for a given compute budget.

**Example (illustrative numbers, Chinchilla-style constants quoted from memory):** $E=1.69,\ A=406,\ B=411,\ \alpha=0.34,\ \beta=0.28$. Model $N=10^9$ parameters, data $D=2\times10^{10}$ tokens.
- Size penalty $A/N^\alpha\approx0.354$. Data penalty $B/D^\beta\approx0.536$.
- Loss $L\approx1.69+0.354+0.536=\mathbf{2.58}$.
- Double the model size: size penalty becomes $0.354\times2^{-0.34}=0.354\times0.79\approx0.280$. New loss $\approx\mathbf{2.51}$ (a gain of only $0.07$).
- Doubling again only takes the size penalty from $0.280$ to $0.221$. Each doubling helps less than the one before.

with compute-optimal training using data and parameters in similar proportion (the "Chinchilla" result, about 20 tokens per parameter).

**RLHF objective:** maximise a learned reward $r_\phi$ while staying near the SFT policy $\pi_{\text{ref}}$:

$$
\max_{\pi}\ \mathbb{E}_{y\sim\pi(\cdot\mid x)}[r_\phi(x,y)]-\beta\,\mathrm{KL}\big(\pi(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)
$$

**Where:**
- $\pi$ is the **policy**: the model being trained (given a prompt $x$, it picks a response $y$). $\max_\pi$ means "find the model that makes the rest as large as possible".
- $x$ is the prompt and $y$ is a response the model writes, drawn from $\pi$.
- $r_\phi(x,y)$ is the **reward model**: a separate model, with learned numbers $\phi$, that scores how good a response is, trained from human preferences. $\mathbb{E}[\cdot]$ averages that score over many responses. **Matters:** this is the "be more helpful" push.
- $\pi_{\text{ref}}$ is the reference model: a frozen copy from before this step (the SFT model).
- $\mathrm{KL}(\pi\,\|\,\pi_{\text{ref}})$ is the Kullback–Leibler divergence, a measure of how far the new model's behaviour has drifted from the reference. **Matters:** it is a leash. Without it the model would learn to game the reward model with weird text that scores high but is not actually good.
- $\beta$ is the leash strength. Large $\beta$ keeps the model close to the reference; small $\beta$ lets it chase reward more freely.

**Example (tiny numbers):** two candidate behaviours, $\beta=0.5$.
- Behaviour A: reward $5$, but it drifted far from the reference ($\mathrm{KL}=4$). Score $=5-0.5\times4=\mathbf{3}$.
- Behaviour B: reward $4$, stays close ($\mathrm{KL}=0.5$). Score $=4-0.5\times0.5=\mathbf{3.75}$.
- B wins even though A has the higher raw reward. The leash caught A for weird, reward-gaming behaviour.

DPO optimises the same objective directly from preference pairs, with no separate reward model.

Open problems: hallucination, evaluation (benchmark contamination), reasoning reliability, long-context use, interpretability, and the cost of training and serving.

**What came before, and how this improved on it:**
- **Before:** n-gram models, which are lookup tables of how often each short word sequence occurred.
- **Its limit:** the table grows by the vocabulary size for every extra word of memory, and most sequences are never seen.
- **What changed:** one neural network with shared weights ([[Transformer]]), trained on huge text ([[Pre-training]]). GPT-3 has 175 billion parameters (Brown et al., arXiv 2005.14165, 28 May 2020, abstract read).
- **How much better:** in the code below, with a 50,000-word vocabulary, a table for 2-word sequences has 2.50e+09 entries, 3 words 1.25e+14, and 4 words 6.25e+18. The weights of a GPT-3-sized transformer by the $12Ld^2$ rule come to 173,946,175,488, about 174 billion, close to the paper's 175 billion. So the trigram table is 719 times bigger than the network, and the network reaches much further back. The layer count 96 and width 12,288 are from memory of the paper's table.

**Check it yourself:**

```python
V = 50000
for n in (2, 3, 4): print(f"{n}-word lookup table over a {V:,}-word vocabulary: {float(V) ** n:.2e} entries")
L, d = 96, 12288                                         # layers and width of GPT-3 175B (from memory of the paper's table)
params = 12 * L * d * d
print(f"transformer weights by the 12 * layers * width^2 rule: {params:,}  (about {params / 1e9:.0f} billion; the paper says 175 billion)")
print(f"trigram table / network weights: {float(V) ** 3 / params:.0f}x bigger")
```

## Related
- [[Transformer]] · [[Embedding]] · [[Tokenization]] · [[RAG]] · [[Fine-tuning]] · [[RLHF]] · [[Pre-training]] · [[Scaling Laws]] · [[Mixture of Experts]] · [[Quantization]] · [[Byte-Pair Encoding]] · [[Knowledge Distillation]] · [[RoPE]] · [[Policy Gradient]] · [[PPO]] · [[FlashAttention]] · [[KV Cache]] · [[Grouped-Query Attention]] · [[Decoding Strategies]] · [[Speculative Decoding]] · [[Reasoning Model]] · [[Tool Calling]] · [[Chain of Thought]] · [[Masked Diffusion Language Model]] · [[LLM as a Judge]] · [[In-Context Learning]] · [[Sliding Window Attention]] · [[Benchmarks]] · [[PagedAttention]] · [[Instruction Tuning]] · [[History of Text Analytics]] · [[History of AI]] · [[Hallucination]]

## Sources
- Brown et al., *Language Models are Few-Shot Learners* (GPT-3), 2020
- Kaplan et al., *Scaling Laws for Neural Language Models*, 2020
- Hoffmann et al., *Training Compute-Optimal Large Language Models*, 2022
- Ouyang et al., *Training language models to follow instructions with human feedback*, 2022
- Brown et al., *Language Models are Few-Shot Learners*, arXiv 2005.14165 (2020): abstract read for the 175-billion figure in the history block. Layer count and width are from memory.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm LLM (large language model):::A computer program that has read a gigantic amount of text and learned to guess the next word, over and over, very well.

#flashcards/ml/formula #flashcards/topic/llm Write how a language model assigns a probability to a text and explain each symbol.
?
$$p(x_1,\dots,x_n)=\prod_{t=1}^{n} p(x_t\mid x_{<t})$$
- $x_1,\dots,x_n$: the tokens of a text in order. $n$: how many there are. $t$: the position being looked at.
- $p(x_1,\dots,x_n)$: the probability of the whole text. A good model gives sensible text a high probability and gibberish a low one.
- $x_{<t}$: all the tokens before position $t$.
- $\prod$: multiply the next-token chances along the text.

#flashcards/ml/formula #flashcards/topic/llm Write the LLM training loss and explain each symbol.
?
$$\mathcal{L}(\theta)=-\mathbb{E}_{x\sim\mathcal{D}}\sum_{t}\log p_\theta(x_t\mid x_{<t})$$
- $\mathcal{L}(\theta)$: the loss, one number for how badly the model predicts text. Lower is better.
- $\theta$: all the model's learned numbers (its parameters, often billions). These are the only things training changes.
- $x\sim\mathcal{D}$: texts drawn from the training data. $\mathbb{E}$ averages over them.
- $-\log p_\theta(x_t\mid x_{<t})$: the surprise at the true next token. Confident and right costs little, surprised costs a lot.

#flashcards/ml/formula #flashcards/topic/llm Write the scaling law form and explain each symbol.
?
$$L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}$$
- $L$: the final loss the model reaches after training.
- $N$: the number of parameters (model size). $D$: the amount of training data in tokens.
- $E$: the irreducible loss. Text is partly unpredictable, so even a perfect model cannot reach 0.
- $A/N^\alpha$ and $B/D^\beta$: the loss that shrinks as the model or the data grows. $A$, $B$, $\alpha$, $\beta$ are fitted constants.

#flashcards/ml/cloze #flashcards/topic/llm An LLM is trained to predict the ==next token==, and the training loss is the ==negative log== of the chance it gave to the true one.

#flashcards/ml/example #flashcards/topic/llm "I love cats" has $p(\text{I})=0.1$, $p(\text{love}\mid\text{I})=0.2$, $p(\text{cats}\mid\text{I love})=0.05$. What is the probability of the whole text?::0.1 × 0.2 × 0.05 = 0.001.

#flashcards/ml/example #flashcards/topic/llm The correct next tokens got probabilities 0.5, 0.25 and 0.9. What is the loss?::−ln 0.5 = 0.69, −ln 0.25 = 1.39, −ln 0.9 = 0.11. Sum 2.18, or 0.73 per token.

#flashcards/ml/image #flashcards/topic/llm ![[llm-training-stages.svg]]
What does this picture show?
?
Three stages of building an LLM: pre-training, then fine-tuning, then alignment.

#flashcards/ml/formula #flashcards/topic/llm Write the RLHF training objective (reward with a leash) and explain each symbol.
?
$$\max_{\pi}\ \mathbb{E}_{y\sim\pi(\cdot\mid x)}[r_\phi(x,y)]-\beta\,\mathrm{KL}\big(\pi(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)$$
- $\pi$: the policy, the model being trained (given a prompt $x$ it picks a response $y$). $\max_\pi$: find the model that makes the rest as large as possible.
- $x$: the prompt. $y$: a response the model writes, drawn from $\pi$.
- $r_\phi(x,y)$: the reward model, a separate model with learned numbers $\phi$ that scores how good a response is, trained from human preferences. This is the "be more helpful" push.
- $\pi_{\text{ref}}$: the reference model, a frozen copy from before this step (the SFT model).
- $\mathrm{KL}$: the Kullback-Leibler divergence, how far the new model has drifted from the reference. It is a leash: without it the model learns to game the reward model with weird text.
- $\beta$: the leash strength. Large keeps the model close to the reference, small lets it chase reward.

#flashcards/ml/example #flashcards/topic/llm Behaviour A has reward 5 but KL = 4. Behaviour B has reward 4 and KL = 0.5. With $\beta=0.5$, which scores higher?::A: 5 − 0.5 × 4 = 3. B: 4 − 0.5 × 0.5 = 3.75. B wins even though A has the higher raw reward. The leash caught A for reward-gaming behaviour.

#flashcards/ml/concept #flashcards/topic/llm What did LLMs improve on, and by how much (from the note's own numbers)?::Before: n-gram lookup tables. With a 50,000-word vocabulary, a 2-word table has 2.50e+09 entries, 3 words 1.25e+14, 4 words 6.25e+18. A GPT-3-sized transformer by the 12Ld² rule has about 174 billion weights (the paper says 175 billion), so the trigram table is 719 times bigger than the network, which reaches much further back.
