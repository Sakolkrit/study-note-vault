---
tags: [softmax, probability, neural-network, LLM, transformer, math]
status: growing
level: 4
tier: 1
---

# Softmax

## 🟢 1. Simple (age 12)
Softmax turns a list of **scores** into a list of **chances that add up to 100%**. The biggest score gets the biggest chance, but the smaller scores still get a little.

**Example:** Three friends pick a game and you give each game a "wish score": football 2, chess 1, cards 0. Softmax turns that into chances: football about 67%, chess about 24%, cards about 9%. Football is most likely, but chess and cards still have a real chance.

## 🔵 2. How it works
1. Start with a list of scores. These can be any numbers: big, small, even negative.
2. Make each score **positive** by raising the number $e$ (about 2.718) to the power of the score. Bigger score, much bigger result. Negative scores become small positive numbers, never negative.
3. **Add up** all the results.
4. **Divide** each result by that total. Now every number is between 0 and 1 and they all add up to 1.

Where it is used: the last step of a classifier, and the step in an [[LLM]] that turns scores for every possible next word (see [[Tokenization]]) into chances so one can be picked. Inside [[Attention]] it turns "how well do these words match?" into "how much should I listen to each one?".

![[softmax-vs-max.jpg]]
*Why it is called "soft" max: the blue circles are the hard rule max(x, 5), which sticks at 5 and then jumps to following x. The red curve is a smooth, rounded version of the same rule. Softmax is the same smoothing idea applied to picking the largest item. (I read the colours and axis label from the plot itself; the file has no English description.) Image: MiladRazaviMohseni, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Softmax_vs_max.jpg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Input:** a vector of raw scores called **logits**, usually the output of the last layer of a [[Neural Network]].
- **Output:** a probability vector: all entries in $(0,1)$ and summing to 1.
- **Order is kept:** the biggest logit always gets the biggest probability.
- **Shift-invariant:** adding the same number to every logit changes nothing. Code uses this to stay safe: subtract the maximum logit first so $e^{\text{score}}$ never overflows.
- **Temperature:** divide the logits by a number $T$ before softmax. Low $T$ makes the choice sharper (nearly always the top item), high $T$ makes it flatter (more random). This is the "temperature" setting in [[LLM]] text generation.
- **Cost:** one pass over the vector. In a [[Transformer]] it runs on every row of the attention score grid, which is part of why long text is costly.

```python
import numpy as np
def softmax(z, T=1.0):
    z = np.asarray(z, float) / T
    e = np.exp(z - z.max())          # subtract max: same answer, no overflow
    return e / e.sum()

print(softmax([2, 1, 0]).round(3))          # [0.665 0.245 0.09 ]
print(softmax([1001, 1000, 999]).round(3))  # [0.665 0.245 0.09 ]  same, no overflow
print(softmax([2, 1, 0], T=2).round(3))     # [0.506 0.307 0.186]  flatter
print(softmax([2, 1, 0], T=0.5).round(3))   # [0.867 0.117 0.016]  sharper
```

## 🔴 4. Deep / formal
**The softmax function.** For a vector of logits $\mathbf{z}=(z_1,\dots,z_K)$:

$$
\operatorname{softmax}(\mathbf{z})_i=\frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

**Where:**
- $\mathbf{z}$ is the list of raw scores (logits) and $K$ is how many there are (for an [[LLM]], one per word in its vocabulary).
- $z_i$ is the score of item $i$, and $\operatorname{softmax}(\mathbf{z})_i$ is the chance given to item $i$.
- $e^{z_i}$ is $e\approx2.718$ raised to the score. **Matters:** it makes every value positive (so chances can never be negative) and makes the gaps count multiplicatively: a score that is 1 higher gets $e\approx2.7$ times the weight.
- $\sum_{j} e^{z_j}$ adds up that weight for all $K$ items. **Matters:** dividing by it is what makes the answers add up to exactly 1. It is called the normaliser.
- Because only gaps between scores matter, adding the same constant to all $z_j$ cancels top and bottom. **Matters:** this is why subtracting $\max_j z_j$ is safe and prevents overflow.

**Example (tiny numbers):** $\mathbf{z}=(2,\,1,\,0)$.
- Raise $e$ to each: $e^{2}=7.389$, $e^{1}=2.718$, $e^{0}=1$.
- Add them: $7.389+2.718+1=11.107$.
- Divide: $7.389/11.107=\mathbf{0.665}$, $2.718/11.107=\mathbf{0.245}$, $1/11.107=\mathbf{0.090}$. They add to $1$.
- Meaning: the item with score 2 gets about a 67% chance. Doubling every score to $(4,2,0)$ would push it to $0.867$, $0.117$, $0.016$: bigger gaps make the winner more dominant.

**With temperature.** Dividing the logits by $T>0$ before softmax:

$$
p_i(T)=\frac{e^{z_i/T}}{\sum_{j} e^{z_j/T}}
$$

**Where:**
- $z_i$ and $p_i$ are as above, and $T$ is the **temperature**, a positive number you choose.
- $z_i/T$ is the score after being scaled. **Matters:** $T<1$ stretches the gaps (sharper, more predictable), $T>1$ shrinks them (flatter, more varied). As $T\to0$ the result approaches "pick the single biggest, probability 1"; as $T\to\infty$ it approaches "all equal".
- $T=1$ gives back ordinary softmax.

**Example (tiny numbers):** $\mathbf{z}=(2,1,0)$ as before.
- $T=2$: scores become $(1,\,0.5,\,0)$. $e^{1}=2.718$, $e^{0.5}=1.649$, $e^{0}=1$, total $5.367$. Chances: $\mathbf{0.506},\ \mathbf{0.307},\ \mathbf{0.186}$. Flatter than the $0.665$ at $T=1$.
- $T=0.5$: scores become $(4,\,2,\,0)$, giving $\mathbf{0.867},\ \mathbf{0.117},\ \mathbf{0.016}$. Sharper.
- Meaning: low temperature makes a chatbot stick to its top guess, high temperature makes it more adventurous.

**Further notes:**
- **Link to the sigmoid:** with two classes, softmax reduces to the sigmoid of the score difference, $\sigma(z_1-z_2)$. The same function appears in the reward model in [[RLHF]].
- **Gradient:** with cross-entropy loss ([[Loss Function]]) the gradient of the loss with respect to the logits is simply $p_i-y_i$ (predicted chance minus the true 0 or 1). This tidy result (stated from memory of the standard derivation; check it yourself) is why the pair is used together in [[Backpropagation]].
- **Saturation:** if one logit is far larger than the others the output is almost one-hot and the gradient nearly vanishes, which can slow learning.
- **Softmax is a smooth maximum**, not a true maximum: it keeps all items alive, which is what makes it differentiable and trainable with [[Gradient Descent]].

**What came before, and how this improved on it:**
- **Before:** turn scores into shares by dividing each by the total, or just take the highest one.
- **Its limit:** dividing by the total breaks when a score is negative (it gave -0.5), and taking only the highest throws away how close the others were.
- **What changed:** exponentiate first so every score is positive, then divide by the total. The function comes from statistical physics (Boltzmann and Gibbs, 1800s) and was named "softmax" in the machine-learning literature around 1990 (Bridle; both from memory, not checked).
- **How much better:** in the code below, the scores 2, 1 and -1 give shares of 0.7054, 0.2595 and 0.0351 that add up to 1, where plain division gave 1.0, 0.5 and -0.5. And for scores near 1000 the direct formula returned `nan`, while subtracting the largest score first gave 0.6652, 0.2447 and 0.0900.

**Check it yourself:**

```python
import numpy as np
z = np.array([2.0, 1.0, -1.0])
print("scores", z)
print("divide by the sum:", z / z.sum(), " (a negative 'probability', and the total is", (z / z.sum()).sum(), ")")
e = np.exp(z); print("softmax:          ", (e / e.sum()).round(4), " total", (e / e.sum()).sum())
big = np.array([1000.0, 999.0, 998.0])
with np.errstate(all="ignore"): naive = np.exp(big) / np.exp(big).sum()
s = np.exp(big - big.max()); print("scores near 1000, naive exp:", naive, " subtract the max first:", (s / s.sum()).round(4))
```

## Related
- [[Neural Network]] · [[Loss Function]] · [[Backpropagation]] · [[Attention]] · [[Transformer]] · [[LLM]] · [[Tokenization]] · [[RLHF]] · [[Gradient Descent]] · [[Activation Function]] · [[Logistic Regression]] · [[Word2Vec]] · [[Entropy]] · [[KL Divergence]] · [[Negative Sampling]] · [[Knowledge Distillation]] · [[Self-Attention]] · [[Policy Gradient]] · [[FlashAttention]] · [[Decoding Strategies]] · [[Calibration]] · [[Cross-Entropy Loss]]

## Sources
- Goodfellow, Bengio & Courville, *Deep Learning*, 2016, section on softmax
- Vaswani et al., *Attention Is All You Need*, 2017 (softmax in attention)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/transformers-nlp Softmax:::Turns a list of scores into a list of chances that add up to 100%. The biggest score gets the biggest chance, but smaller scores still get a little.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write softmax and explain each symbol.
?
$$\operatorname{softmax}(\mathbf{z})_i=\frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$
- $\mathbf{z}$: the list of raw scores (logits). $K$: how many there are (for an LLM, one per word in the vocabulary).
- $z_i$: the score of item $i$. $\operatorname{softmax}(\mathbf{z})_i$: the chance given to item $i$.
- $e^{z_i}$: $e\approx2.718$ raised to the score. This makes every value positive, so chances can never be negative.
- The sum in the bottom divides by the total, so all chances add up to 1.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write softmax with temperature and explain each symbol.
?
$$p_i(T)=\frac{e^{z_i/T}}{\sum_{j} e^{z_j/T}}$$
- $z_i$, $p_i$: the score and chance as above. $T$: the temperature, a positive number you choose.
- $z_i/T$: the score after scaling. $T<1$ stretches the gaps (sharper, more predictable). $T>1$ shrinks them (flatter, more varied).
- As $T\to0$ it approaches "pick the single biggest". As $T\to\infty$ it approaches "all equal".

#flashcards/ml/cloze #flashcards/topic/transformers-nlp A low temperature gives a ==sharper== softmax, and a high temperature gives a ==flatter== one.

#flashcards/ml/example #flashcards/topic/transformers-nlp Softmax of (2, 1, 0)?::e² = 7.389, e¹ = 2.718, e⁰ = 1, sum 11.107. Chances 0.665, 0.245, 0.090, adding to 1.

#flashcards/ml/example #flashcards/topic/transformers-nlp Softmax of (2, 1, 0) with $T=2$ and with $T=0.5$?::T = 2: 0.506, 0.307, 0.186 (flatter). T = 0.5: 0.867, 0.117, 0.016 (sharper).

#flashcards/ml/image #flashcards/topic/transformers-nlp ![[softmax-vs-max.jpg]]
Why is it called "soft" max?
?
The blue circles are the hard rule max(x, 5), which sticks at 5 and then jumps to following x. The red curve is a smooth, rounded version of the same rule. Softmax is the same smoothing idea applied to a list of scores.
