---
tags: [LLM, machine-learning, transformer, architecture, sparsity, scaling]
status: growing
level: 4
tier: 4
---

# Mixture of Experts

A Mixture of Experts (MoE) model is made of **many small specialist sub-networks plus a "router" that picks which few specialists handle each piece of input**. The model can be huge, but only a small part of it works on any one token.

## 🟢 1. Simple (age 12)
Think of a hospital. It has a heart doctor, a bone doctor, an eye doctor and so on. When you walk in, a **receptionist** looks at what is wrong and sends you to just the one or two right doctors. The hospital knows a huge amount in total, but each patient only uses a tiny part of it, so it stays fast. A Mixture of Experts model works like this hospital: lots of knowledge, but only a few "doctors" wake up for each word.

**Example:** The sentence "The function returns an integer" has code-flavoured words. A router might send those words to the experts that mostly handled programming text, while a word in a poem might go to different experts. Nobody told the experts to specialise. It tends to happen during training.

## 🔵 2. How it works
This idea is used inside a [[Transformer]], where it replaces the big feed-forward layer (a plain [[Neural Network]] block) in each block.
1. **Experts:** instead of one big block, there are several (for example 8 or many more), each a smaller neural network of the same shape.
2. **Router (gating network):** a small layer reads the incoming token's [[Vector]] and gives a **score for each expert**.
3. **Pick the top few:** keep only the best 1 or 2 experts for this token. The other experts do nothing, so they cost no compute.
4. **Turn scores into weights:** a [[Softmax]] over the chosen scores gives weights that add up to 1.
5. **Mix:** run the chosen experts, multiply each result by its weight, and add them up. That is the layer's output.
6. **Train everything together:** the router and the experts all learn from the same loss, with [[Backpropagation]]. Experts drift towards different jobs, and the router learns to send each token to a good one.
7. **Keep it fair:** an extra small penalty stops the router from sending everything to the same favourite expert.

![[mixture-of-experts-gating.svg]]
*A Mixture of Experts layer. The input x goes both to a gating network and to all the expert boxes. The gating network decides how much each expert counts (the small crossed boxes are the gate weights). The weighted answers are added together on the right to give the output. In a sparse model most of those weights are exactly zero, so most experts are skipped. Image: Sinafe, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Mixure_of_Experts_-_Gating_Network.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Sparse vs dense:** a normal ("dense") layer uses all of its weights for every token. A *sparse* MoE layer uses only $k$ of its $E$ experts per token, typically $k=1$ or $2$.
- **Total vs active parameters:** the model stores all $E$ experts (big memory), but each token only runs $k$ of them (small compute). So you can have far more parameters at about the same cost per token. This connects to [[Scaling Laws]]: more parameters usually means lower loss.
- **The catch:** the whole model still has to fit in memory across your [[GPU]]s, and sending tokens to experts on different GPUs costs communication time.
- **Load balancing:** without care, the router can **collapse**: a few experts get most tokens and train well, while the rest are starved and stay useless. A balancing term in the loss pushes the traffic to be even (see the formal part). Many systems also cap how many tokens one expert accepts per batch.
- **Routing choices:** top-1 (each token goes to one expert, as in Switch Transformer, 2021) or top-2 (as in the GShard family and Mixtral). All of this is from memory. Newer designs use many small experts, and a few "shared" experts that every token uses.
- **Where it came from:** the idea is old (Jacobs, Jordan, Nowlan and Hinton, 1991; from memory). It became practical for large language models with the sparsely-gated layer of Shazeer et al. (2017; from memory), and then with models such as Switch Transformer and Mixtral (also from memory).
- **Not the same as an ensemble:** in an [[Ensemble Methods]] model, separate full models each vote on the whole input. Here the experts are trained together as parts of one model, and each token only visits a few of them.
- **Not a mixture of Gaussians, but a cousin:** a [[Gaussian Mixture Model]] gives each point soft shares across bells. An MoE gate also gives soft shares across experts, but the experts are learned functions and the gate depends on the input.
- **Experts need not match topics:** studies of trained models report that experts often specialise in surface patterns (such as punctuation or word types) rather than clean human topics (from memory), so the hospital picture is only an analogy.
- **Training difficulty:** MoE models can be less stable to train and harder to fine-tune than dense models (stated from memory; a known practical issue).

Below, first the layer in plain numpy (the router, top-2 mixing and the load statistics), then a tiny trained example.

```python
import numpy as np
rng = np.random.default_rng(0)
def softmax(z): e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
d, E, k = 4, 4, 2                                           # token size, number of experts, experts used per token
Wg = rng.normal(0, 1, (d, E))                               # gating network: one score per expert
experts = [rng.normal(0, 1, (d, d)) for _ in range(E)]      # each expert here is just one small matrix
def moe(x):
    logits = x @ Wg; top = np.argsort(-logits)[:k]          # pick the k best-scoring experts
    g = np.zeros(E); g[top] = softmax(logits[top])          # softmax only over the chosen ones; the rest get exactly 0
    y = sum(g[i] * (x @ experts[i]) for i in top)           # run ONLY the chosen experts, mix by gate weight
    return y, g
x = rng.normal(0, 1, d); y, g = moe(x)
print(g.round(3), round(float(g.sum()), 3), int((g > 0).sum()))        # [0.907 0.    0.093 0.   ] 1.0 2
X = rng.normal(0, 1, (1000, d)); G = np.array([moe(t)[1] for t in X])
f = (G > 0).mean(0) / k; P = softmax(X @ Wg).mean(0)       # f: share of tokens routed, P: mean router probability
print(f.round(3), P.round(3), round(float(E * (f * P).sum()), 3))      # [0.259 0.252 0.246 0.244] [0.311 0.266 0.212 0.211] 1.004
p_exp, shared, n_exp = 100e6, 300e6, 8
print((shared + n_exp * p_exp) / 1e9, (shared + k * p_exp) / 1e9)      # 1.1 0.5
```

For this one token the router gave nearly all the weight (0.907) to one expert and 0.093 to another, and exactly 0 to the other two, so only 2 of the 4 experts ran. Over 1,000 random tokens the traffic is nearly even (each expert gets 24% to 26% of the assignments), which gives a balance score of 1.004, close to the perfect 1.0. The weights here are random and untrained, so nothing is specialised. The last line is the parameter count: with 8 experts of 100 million parameters and 300 million shared parameters, the model holds 1.1 billion parameters but uses only 0.5 billion per token (with 2 experts active).

A tiny trained example (needs PyTorch; I ran it with version 2.8.0 on CPU, and other versions may print slightly different digits):

```python
import torch
torch.manual_seed(0)
x = torch.linspace(-2, 2, 200).unsqueeze(1)
y = torch.where(x > 0, 2 * x, -x)                      # a "V" shape: two different rules, split at 0
def fit(model, steps=3000):
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(steps):
        loss = ((model(x) - y) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()
class MoE(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.gate = torch.nn.Linear(1, 2); s.experts = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(2)])
    def forward(s, x):
        g = torch.softmax(s.gate(x), -1)                # gate weights, one per expert, sum to 1
        return sum(g[:, i:i + 1] * s.experts[i](x) for i in range(2))
print(round(fit(torch.nn.Linear(1, 1)), 3))             # 0.757   one straight line cannot follow a V
m = MoE(); print(round(fit(m), 4))                      # 0.0007  two experts plus a gate can
g = torch.softmax(m.gate(x), -1)
print(round(g[x[:, 0] < -0.5][:, 0].mean().item(), 2), round(g[x[:, 0] > 0.5][:, 0].mean().item(), 2))   # 0.01 0.99
```

A single straight line cannot follow a V, so its error stays at 0.757. Two linear experts with a gate fit it almost perfectly (error 0.0007). The last line shows the gate gave expert 1 only a 0.01 share on the left half and 0.99 on the right, so the two experts split the job without being told to. This toy uses a soft gate over both experts, not a sparse top-1, so it shows specialisation but not the compute saving.

## 🔴 4. Deep / formal
**The gate.** For an input $x$ and $E$ experts, the router scores each expert, keeps the best $k$, and turns those scores into weights:

$$
g(x)=\operatorname{softmax}\big(\operatorname{TopK}(W_g\,x,\;k)\big)
$$

**Where:**
- $x$ is the token's vector. $W_g$ is the router's weight [[Matrix]], so $W_g x$ gives one score per expert.
- $\operatorname{TopK}(\cdot,k)$ keeps the $k$ biggest scores and sets all the others to $-\infty$.
- $\operatorname{softmax}$ turns the surviving scores into positive weights that add up to 1. Experts that were set to $-\infty$ get weight exactly $0$.
- **Matters:** the zeros are what make the layer *sparse*: an expert with weight 0 need not be run at all. Without the TopK, every expert would get a small positive weight and all of them would have to run, which would save no compute.

**Example (tiny numbers):** four experts, $k=2$, router scores $(2.0,\,0.1,\,-1.0,\,1.5)$.
- Top 2 scores: $2.0$ (expert 1) and $1.5$ (expert 4).
- Softmax over just those: $e^{2.0}=7.389$, $e^{1.5}=4.482$, sum $11.871$.
- Weights: $7.389/11.871=\mathbf{0.622}$ and $4.482/11.871=\mathbf{0.378}$. Experts 2 and 3 get $0$.
- Meaning: expert 1 does about 62% of the work, expert 4 about 38%, and the other two are skipped entirely.

**The output.** The layer's answer is the weighted sum of the chosen experts' answers:

$$
y=\sum_{i=1}^{E} g_i(x)\,E_i(x)
$$

**Where:**
- $E_i(x)$ is the output of expert $i$ on the input $x$ (each expert is its own small network).
- $g_i(x)$ is the gate weight for expert $i$ from the equation above. Most of them are $0$.
- The sum runs over all $E$ experts, but only the $k$ with a non-zero weight are actually computed.
- **Matters:** the weights add up to 1, so the output stays on the same scale as a single expert's output. If the gate weights did not sum to 1, the layer's output would grow or shrink depending on which experts were chosen.

**Example (tiny numbers):** use the weights from above, $0.622$ and $0.378$. Suppose expert 1 outputs $4.0$ and expert 4 outputs $-2.0$ (single numbers, for simplicity).
- Weighted: $0.622\times4.0=2.488$ and $0.378\times(-2.0)=-0.756$.
- Sum: $y=2.488-0.756=\mathbf{1.732}$ (with the unrounded weights it is $1.735$).
- Meaning: the two experts disagree in sign, and the gate's weights decide the mix. The output leans towards expert 1's answer because expert 1 has the bigger weight.

**Total versus active size.** With $E$ experts of $p$ parameters each, plus $s$ parameters shared by every token, and $k$ experts used per token:

$$
N_{\text{total}}=s+E\,p,\qquad N_{\text{active}}=s+k\,p
$$

**Where:**
- $s$ is the number of parameters every token always uses (attention layers, embeddings, and so on).
- $E$ is the number of experts and $p$ the parameters in one expert. $k$ is how many experts each token uses.
- $N_{\text{total}}$ decides the memory needed. $N_{\text{active}}$ decides the compute per token (roughly the "6N" cost per token from [[Pre-training]], with $N_{\text{active}}$ in place of $N$).
- **Matters:** it separates "how much the model knows" from "how much each token costs". Without the split, you could not say why an MoE model can be big yet fast. Note the memory still has to hold all $N_{\text{total}}$.

**Example (tiny numbers):** $s=300$ million, $E=8$ experts of $p=100$ million, and $k=2$.
- Total: $300+8\times100=1{,}100$ million $=\mathbf{1.1}$ billion.
- Active: $300+2\times100=500$ million $=\mathbf{0.5}$ billion.
- Meaning: the model has 1.1 billion parameters of storage but each token only pays for 0.5 billion, about 45% of the total.

**Load balancing.** To stop the router from collapsing onto a few favourites, an extra term is added to the loss (a form used in the Switch Transformer paper; from memory):

$$
\mathcal L_{\text{balance}}=E\sum_{i=1}^{E}f_i\,P_i
$$

**Where:**
- $f_i$ is the fraction of tokens in the batch actually sent to expert $i$ (the fractions add up to 1).
- $P_i$ is the average router probability for expert $i$ across the batch (these also add up to 1).
- $E$ in front scales it so a perfectly even split gives exactly $1$.
- **Matters:** the product $f_iP_i$ is large when an expert is both chosen often *and* given high probability, so the sum is smallest when traffic is spread evenly. Training adds a small multiple of it to the main loss. Without it, a few experts get all the training and the rest never improve (the collapse).

**Example (tiny numbers):** $E=4$ experts. An even router has $f=P=(0.25,0.25,0.25,0.25)$.
- Even case: $\sum f_iP_i=4\times(0.25\times0.25)=0.25$, so $\mathcal L=4\times0.25=\mathbf{1.00}$.
- A lopsided router: $f=(0.7,0.1,0.1,0.1)$ and $P=(0.6,0.2,0.1,0.1)$.
- Products: $0.42,\;0.02,\;0.01,\;0.01$. Sum: $0.46$. $\mathcal L=4\times0.46=\mathbf{1.84}$.
- Meaning: the lopsided router scores 1.84 against the best possible 1.00, so training is nudged to spread the tokens more evenly.

**Further notes:**
- **Capacity limits:** each expert is often given a fixed number of slots per batch, and tokens beyond it are dropped or passed on unchanged. How best to handle this is a design choice.
- **Messages between devices:** experts are spread over many GPUs, so tokens must be sent to the right place and back. This communication can dominate the cost.
- **Inference:** one request uses few experts, but a busy server handling many requests ends up using all of them, so memory, not compute, is the main cost.
- **Open questions:** how best to route, how many experts to use, and what experts actually learn are still researched.

**What came before, and how this improved on it:**
- **Before:** a dense [[Transformer]] feed-forward block, in which every parameter is used for every word.
- **Its limit:** compute grows in step with parameters, so a bigger model is a slower model.
- **What changed:** many expert feed-forward blocks, with a gate choosing a few per word. The sparsely-gated layer reports more than 1000 times the model capacity with only minor losses in computational efficiency, and up to 137 billion parameters (Shazeer et al., arXiv 1701.06538, 23 January 2017, abstract read; the 1991 original by Jacobs et al. is from memory).
- **How much better:** the code below uses $d=512$ and a 2048-wide block. A dense block holds 2,097,152 parameters. A layer with 8 experts holds 16,777,216 (8 times more) but uses 4,194,304 per word with the top 2, so it stores 4 times what it uses. A dense layer of the same stored size would use 8 times the compute. Memory is still paid in full.

**Check it yourself:**

```python
d, ff, E_, k = 512, 2048, 8, 2
dense = 2 * d * ff                    # one feed-forward block: up and down matrices
print("dense feed-forward parameters:", dense)
print("MoE, 8 experts: stored", E_ * dense, " used per word (top-2):", k * dense)
print("stored / used:", E_ / k, "  stored vs dense:", E_, "x")
print("a dense model with the same stored size would use", E_, "x the compute per word")
```

## Related
- [[Transformer]] · [[Neural Network]] · [[Softmax]] · [[Scaling Laws]] · [[Pre-training]] · [[LLM]] · [[GPU]] · [[Gaussian Mixture Model]] · [[Ensemble Methods]] · [[Backpropagation]] · [[Matrix]] · [[Vector]] · [[Attention]] · [[Loss Function]] · [[Quantization]] · [[Knowledge Distillation]] · [[Self-Attention]] · [[Mixtral 2024]] · [[Model Parallelism]] · [[LLM Inference]]

## Sources
- Jacobs, Jordan, Nowlan & Hinton, *Adaptive Mixtures of Local Experts*, 1991 (from memory)
- Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*, 2017 (from memory)
- Fedus, Zoph & Shazeer, *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*, 2021 (from memory)
- Jiang et al., *Mixtral of Experts*, 2024 (from memory)
- Lepikhin et al., *GShard*, 2020 (from memory)
- Shazeer et al., arXiv 1701.06538 (abstract read for the history block)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm Mixture of experts (MoE):::A model with many expert sub-networks and a router that sends each token to just the one or two best experts. It knows a lot in total, but each token only uses a small part, so it stays fast.

#flashcards/ml/formula #flashcards/topic/llm Write the MoE gate and explain each symbol.
?
$$g(x)=\operatorname{softmax}\big(\operatorname{TopK}(W_g\,x,\;k)\big)$$
- $x$: the token's vector. $W_g$: the router's weight matrix, so $W_g x$ gives one score per expert.
- $\operatorname{TopK}(\cdot,k)$: keeps the $k$ biggest scores and sets all the others to $-\infty$.
- softmax: turns the surviving scores into positive weights that add up to 1. Experts set to $-\infty$ get weight exactly 0.

#flashcards/ml/formula #flashcards/topic/llm Write the MoE output and explain each symbol.
?
$$y=\sum_{i=1}^{E} g_i(x)\,E_i(x)$$
- $E_i(x)$: the output of expert $i$ on input $x$ (each expert is its own small network).
- $g_i(x)$: the gate weight for expert $i$. Most are 0.
- The sum runs over all $E$ experts, but only the $k$ with a non-zero weight are actually computed.
- The weights add up to 1, so the output stays a weighted blend of the chosen experts.

#flashcards/ml/formula #flashcards/topic/llm Write total and active parameters of an MoE model and explain each symbol.
?
$$N_{\text{total}}=s+E\,p,\qquad N_{\text{active}}=s+k\,p$$
- $s$: the parameters every token always uses (attention layers, embeddings, and so on).
- $E$: the number of experts. $p$: the parameters in one expert. $k$: how many experts each token uses.
- $N_{\text{total}}$ decides the memory needed. $N_{\text{active}}$ decides the compute per token.

#flashcards/ml/formula #flashcards/topic/llm Write the load-balancing loss and explain each symbol.
?
$$\mathcal L_{\text{balance}}=E\sum_{i=1}^{E}f_i\,P_i$$
- $f_i$: the fraction of tokens in the batch actually sent to expert $i$ (adds up to 1).
- $P_i$: the average router probability for expert $i$ across the batch (also adds up to 1).
- $E$ in front scales it so a perfectly even split gives exactly 1.
- The product $f_iP_i$ is large when an expert is both chosen often and favoured by the router, so the loss pushes the router to spread tokens evenly.

#flashcards/ml/cloze #flashcards/topic/llm An MoE model stores ==all== experts in memory but computes only the ==top-k== for each token.

#flashcards/ml/example #flashcards/topic/llm Four experts, $k=2$, router scores (2.0, 0.1, −1.0, 1.5). What are the gate weights?::Top 2: 2.0 and 1.5. e² = 7.389, e^1.5 = 4.482, sum 11.871. Weights 0.622 and 0.378. Experts 2 and 3 get 0.

#flashcards/ml/example #flashcards/topic/llm Expert 1 outputs 4.0 with weight 0.622 and expert 4 outputs −2.0 with weight 0.378. What is $y$?::0.622 × 4.0 − 0.378 × 2.0 = 2.488 − 0.756 = 1.732 (1.735 with unrounded weights).

#flashcards/ml/example #flashcards/topic/llm $s=300$M, $E=8$ experts of $p=100$M, $k=2$. What are the total and active parameters?::Total 300 + 800 = 1,100M = 1.1 billion. Active 300 + 200 = 500M = 0.5 billion, about 45% of the total.

#flashcards/ml/example #flashcards/topic/llm For $E=4$, what is the balance loss for an even router, and for $f=(0.7,0.1,0.1,0.1)$, $P=(0.6,0.2,0.1,0.1)$?::Even: 4 × 4 × (0.25 × 0.25) = 1.00. Lopsided: products 0.42, 0.02, 0.01, 0.01, sum 0.46, loss 4 × 0.46 = 1.84.

#flashcards/ml/image #flashcards/topic/llm ![[mixture-of-experts-gating.svg]]
What does this picture show?
?
A Mixture of Experts layer. The input x goes both to a gating network and to all the expert boxes. The gating network decides how much each expert counts. The weighted answers are added together to give the output. In a sparse model most of those weights are exactly zero, so most experts are skipped.
