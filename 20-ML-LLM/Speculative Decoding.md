---
tags: [LLM, inference, speculative-decoding, decoding, optimization, sampling]
status: growing
level: 4
tier: 3
---

# Speculative Decoding

Also called **speculative sampling**. A small, fast model guesses several words ahead, and the big model checks all the guesses at once.

## 🟢 1. Simple (age 12)
A big AI is accurate but slow, because it writes **one word per step**. A small AI is fast but makes more mistakes. Speculative decoding lets the small AI **draft a few words**, and then the big AI **checks the whole draft in one go**. Every word the big AI agrees with is kept for free, and the first word it disagrees with is replaced by the big AI's own choice. The final text is exactly what the big AI would have written, only sooner.

**Example:** A senior editor (slow, careful) and an intern (quick, sloppy). The intern writes the next five words of a sentence. The editor reads all five at once and crosses out from the first mistake onwards. Reading five words at once takes the editor about as long as writing one, so many words get done per editor-step.

## 🔵 2. How it works
1. Writing with a big [[LLM]] is slow because each new token needs a full pass through the model, and each pass is limited by how fast the memory can be read (see [[KV Cache]]). A pass that checks several tokens at once costs about the same.
2. A small **draft model** quickly proposes $k$ tokens in a row.
3. The big **target model** reads the prompt plus all $k$ drafted tokens in **one pass**, and gives its own probabilities for each position.
4. For each drafted token, in order, we compare the two models' probabilities. If the big model likes the token at least as much as the small one did, we **accept** it. If not, we accept it only **with some probability** (a weighted coin flip).
5. At the first rejection, we **throw away the rest of the draft**. We then pick the replacement token from a corrected distribution that uses both models' probabilities, and stop.
6. If all $k$ tokens are accepted, the big model's pass also gives one extra token for free.
7. The coin flips are set up so that the **final text has exactly the big model's distribution**. The draft model only changes the speed, not what can be written. The code below checks this on a four-token toy.
8. A related idea in the lecture: **multi-token prediction**, where one model has several output heads that predict tokens $t+1,\dots,t+k$ and acts as its own drafter.

## 🟠 3. Technical detail
- **Cost model:** one target pass over $k+1$ positions costs about the same as one pass over 1 position (when memory reads dominate), so each pass yields between 1 and $k+1$ tokens.
- **The accept rule for a drafted token $x$ with draft probability $P(x)$ and target probability $Q(x)$:** accept with probability $\min(1,Q(x)/P(x))$.
- **On rejection:** draw from $\text{norm}\big(\max(Q-P,0)\big)$, the part of the target's distribution that the draft under-covers.
- **Why this is exact:** the accepted draws cover $\min(P,Q)$ of every token's probability, and the corrected draw fills in the missing part (level 4).
- **How good the draft must be:** the more the draft agrees with the target, the more tokens are accepted. A draft that is wrong most of the time gives about one token per pass plus the cost of the draft.
- **The speed-up is not free:** the draft model costs time and memory, and the target's pass over $k+1$ tokens is a bit slower than over 1. The paper's abstract reports a 2 to 2.5 times decoding speed-up for the 70-billion-parameter Chinchilla model in a distributed setup, with no change to the model and no loss in sample quality (arXiv 2302.01318, abstract read).
- **Draft choices:** a smaller model from the same family, or the same model with extra prediction heads (multi-token prediction, from Gloeckle et al. 2024 in the lecture, which I did not read).
- **Works with sampling and greedy decoding.** With greedy decoding the check reduces to "does the target's top token equal the drafted token?" (a simplification I derived, not checked in a paper).

```python
import numpy as np
P = np.array([0.5, 0.3, 0.1, 0.1])        # draft model's probabilities for 4 tokens
Q = np.array([0.2, 0.3, 0.4, 0.1])        # target model's probabilities for the same 4 tokens
rng = np.random.RandomState(0)

def one_token(rng):
    """draft proposes x ~ P; accept with prob min(1, Q/P); otherwise resample from max(Q-P, 0) normalised"""
    x = rng.choice(4, p=P)
    if rng.rand() < min(1, Q[x] / P[x]): return x, True
    r = np.maximum(Q - P, 0); return rng.choice(4, p=r / r.sum()), False

N = 200000
out = [one_token(rng) for _ in range(N)]
freq = np.bincount([t for t, _ in out], minlength=4) / N
print('target Q          :', Q.tolist())
print('output frequencies:', freq.round(3).tolist(), '(speculative sampling output)')
print('draft P           :', P.tolist(), '(what the output would look like without the check)')
# exact: P(output = x) = min(P,Q)(x) + (1 - alpha) * max(Q-P,0)(x)/sum(max(Q-P,0)), alpha = sum min(P,Q)
alpha = np.minimum(P, Q).sum()
exact = np.minimum(P, Q) + (1 - alpha) * np.maximum(Q - P, 0) / np.maximum(Q - P, 0).sum()
print('exact output distribution equals Q:', np.allclose(exact, Q), '| alpha (accept chance) =', round(float(alpha), 3), '| measured', round(float(np.mean([a for _, a in out])), 3))
# tokens per big-model call when the draft proposes k tokens in a row: (1 - alpha^(k+1)) / (1 - alpha)
def run(k, rng, trials=40000):
    total = 0
    for _ in range(trials):
        n = 0
        for _ in range(k):
            _, ok = one_token(rng)
            if not ok: break
            n += 1
        total += n + 1                       # the first rejected (or the bonus) token also comes from the big model
    return total / trials
for k in (1, 2, 4):
    print('k =', k, '| formula', round((1 - alpha ** (k + 1)) / (1 - alpha), 3), '| simulated', round(run(k, rng), 3))
```

## 🔴 4. Deep / formal
**Accept or replace.** Let $P$ be the draft model's distribution and $Q$ the target's, over the same tokens. A drafted token $x\sim P$ is kept with probability $\min(1,Q(x)/P(x))$. If it is rejected, a replacement is drawn from:

$$
r(x)=\frac{\max\big(Q(x)-P(x),\,0\big)}{\sum_{y}\max\big(Q(y)-P(y),\,0\big)}
$$

**Where:**
- $P(x)$ is the probability the small model gave to token $x$, and $Q(x)$ the probability the big model gave.
- $\max(Q-P,0)$ is how much *more* the big model likes a token than the small one did (0 if it likes it less).
- The sum in the bottom turns those excesses into a proper distribution.
- $r(x)$ is the distribution the replacement token is drawn from.
- **Matters:** these are exactly the tokens the draft proposed too rarely. Replacing from $r$ makes the final output follow $Q$, not $P$. If we replaced from $Q$ itself, the tokens the draft over-proposed would be over-represented, because they already got their share through the acceptance step.

**Example (tiny numbers):** four tokens, with $P=[0.5,0.3,0.1,0.1]$ and $Q=[0.2,0.3,0.4,0.1]$.
- Acceptance chances: token 0 has $\min(1,0.2/0.5)=\mathbf{0.4}$; token 1 has $\min(1,0.3/0.3)=\mathbf{1}$; token 2 has $\min(1,0.4/0.1)=\mathbf{1}$; token 3 has $1$.
- Excess $Q-P=[-0.3,0,0.3,0]$, so $\max(\cdot,0)=[0,0,0.3,0]$ and $r=[0,0,1,0]$: a rejected draft is always replaced by token 2.
- Chance the output is token 2: accepted when drafted, $P(2)\cdot1=0.1$, plus the share of replacements, $(1-\alpha)\cdot1=0.3$, which is $\mathbf{0.4}=Q(2)$. Here $\alpha=0.7$ is the overall acceptance chance (next formula).
- Meaning: the output distribution is $Q$, not $P$. The code prints the simulated frequencies $0.199,0.3,0.401,0.1$ against $Q=[0.2,0.3,0.4,0.1]$.

**How many tokens per big pass.** Let $\alpha=\sum_x\min(P(x),Q(x))$ be the chance a drafted token is accepted. If we draft $k$ tokens and each is accepted independently with chance $\alpha$ (a simplification), the expected number of tokens made per target pass is:

$$
\mathbb{E}[\text{tokens}]=\frac{1-\alpha^{k+1}}{1-\alpha}
$$

**Where:**
- $\alpha$ is the acceptance chance of one drafted token, between 0 and 1. It is large when the two models agree.
- $k$ is how many tokens the draft proposes per pass.
- The top is $1-\alpha^{k+1}$ and the bottom $1-\alpha$, which together are the sum $1+\alpha+\alpha^2+\cdots+\alpha^{k}$.
- **Matters:** at $\alpha=0$ you get exactly 1 token per pass (nothing gained). At $\alpha\to1$ you get $k+1$. A bigger $k$ helps only as long as $\alpha^{k}$ is not tiny, because every extra drafted token must survive all the earlier ones. This formula is the one I remember from the speculative-decoding papers (from memory); the code checks it by simulation rather than from a source.

**Example (tiny numbers):** $\alpha=0.7$, $k=2$.
- $1-0.7^{3}=1-0.343=0.657$ and $1-0.7=0.3$.
- $0.657/0.3=\mathbf{2.19}$ tokens per pass.
- Meaning: each pass of the big model writes about 2.19 tokens instead of 1. The code simulated $2.193$. For $k=1$ the formula gives $1.7$ and the simulation $1.701$, and for $k=4$ it gives $2.773$ and the simulation $2.771$.

**Further notes:**
- **Real cost:** this counts big-model passes only. Time also depends on how long the draft takes. I did not measure speed on real models.
- **Real acceptance varies by text:** easy, predictable text (boilerplate, code) is accepted more than creative text (from general experience, not tested here).
- **Not covered:** tree-shaped drafts, batching several users, and how the draft model is chosen or trained.

**What came before, and how this improved on it:**
- **Before:** plain autoregressive decoding, one token per pass of the big model, which is what [[KV Cache]] and [[Grouped-Query Attention]] make cheaper but do not change.
- **Its limit:** one pass per token, and each pass is limited by memory reads, so the hardware sits mostly idle.
- **What changed:** draft several tokens with a small model and verify them with the big one in one pass, with a rejection rule that keeps the big model's exact distribution (Chen et al., arXiv 2302.01318, 2 February 2023, abstract read; the same idea appeared in Leviathan et al. around the same time, from memory, which I did not check).
- **How much better:** the paper's abstract reports a 2 to 2.5 times decoding speed-up on a 70B model. In this note's toy: the output distribution matched the target exactly ($0.199,0.3,0.401,0.1$ against $0.2,0.3,0.4,0.1$), and at acceptance chance $0.7$ with 2 drafted tokens each big pass made $2.19$ tokens instead of 1 (simulated $2.193$). That toy ignores the draft's own cost.

## Related
- [[KV Cache]] · [[Decoding Strategies]] · [[Grouped-Query Attention]] · [[LLM]] · [[Knowledge Distillation]] · [[Transformer]] · [[Probability]] · [[GPU]] · [[Quantization]] · [[Beam Search]] · [[Multi-Token Prediction]] · [[LLM Inference]]

## Sources
- Stanford CME 295 (Fall 2025), Lecture 3 slides, Afshine and Shervine Amidi
- Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, 2023 (arXiv 2302.01318, abstract read)
- Gloeckle et al., *Better & Faster Large Language Models via Multi-token Prediction*, 2024 (named in the lecture; not read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving Speculative decoding:::A small draft model proposes $k$ tokens, and the big model checks them all in one pass. Agreed tokens are kept, the first disagreement is replaced by the big model's own choice, and the output is exactly what the big model would have written.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the replacement distribution used after a rejected draft token and explain each symbol.
?
$$r(x)=\frac{\max\big(Q(x)-P(x),\,0\big)}{\sum_{y}\max\big(Q(y)-P(y),\,0\big)}$$
- $P(x)$: the probability the small (draft) model gave to token $x$. $Q(x)$: the probability the big (target) model gave.
- $\max(Q-P,0)$: how much more the big model likes a token than the small one did (0 if it likes it less).
- The sum at the bottom turns those excesses into a proper distribution.
- $r(x)$: the distribution the replacement token is drawn from.
- These are exactly the tokens the draft proposed too rarely, so replacing from $r$ makes the final output follow $Q$, not $P$. Replacing from $Q$ itself would over-represent tokens the draft already over-proposed.

#flashcards/ml/example #flashcards/topic/llm-serving $P=[0.5,0.3,0.1,0.1]$, $Q=[0.2,0.3,0.4,0.1]$. What are the acceptance chances and $r$?::Acceptance $\min(1,Q/P)$ is $0.4,\ 1,\ 1,\ 1$. Excess $Q-P=[-0.3,0,0.3,0]$, so $\max(\cdot,0)=[0,0,0.3,0]$ and $r=[0,0,1,0]$. A rejected draft is always replaced by token 2.

#flashcards/ml/formula #flashcards/topic/llm-serving Write the expected number of tokens per big-model pass and explain each symbol.
?
$$\mathbb{E}[\text{tokens}]=\frac{1-\alpha^{k+1}}{1-\alpha}$$
- $\alpha$: the acceptance chance of one drafted token (0 to 1). Large when the two models agree.
- $k$: how many tokens the draft proposes per pass.
- Top and bottom together equal $1+\alpha+\alpha^2+\cdots+\alpha^{k}$.
- At $\alpha=0$ you get exactly 1 token per pass (nothing gained). At $\alpha\to1$ you get $k+1$. A bigger $k$ helps only while $\alpha^k$ is not tiny. (The note says this formula is remembered from the papers.)

#flashcards/ml/example #flashcards/topic/llm-serving $\alpha=0.7$, $k=2$. How many tokens per big-model pass?::$1-0.7^3=0.657$, and $1-0.7=0.3$. $0.657/0.3=2.19$ tokens per pass instead of 1. For $k=1$ it is 1.7 and for $k=4$ it is 2.773.

#flashcards/ml/cloze #flashcards/topic/llm-serving Speculative decoding checks the whole draft in ==one pass==, because a pass over several tokens costs about the same as a pass over one.

#flashcards/ml/concept #flashcards/topic/llm-serving At the first rejected draft token, what happens to the rest of the draft?::It is thrown away. A replacement token is drawn from the corrected distribution $r$ and the pass stops.

#flashcards/ml/concept #flashcards/topic/llm-serving Does speculative decoding change what the big model writes?::No. The output follows the big model's own distribution exactly. Only the speed changes.
