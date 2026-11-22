---
tags: [LLM, machine-learning, scaling, training, power-law, compute]
status: growing
level: 4
tier: 3
---

# Scaling Laws

Scaling laws are the **patterns that say how much better a model gets when you make it bigger, feed it more data, or give it more compute**. The pattern is smooth and predictable, which lets people plan huge training runs before they start.

## 🟢 1. Simple (age 12)
Say you are building paper aeroplanes. You notice a pattern: each time you use a bigger sheet, the plane flies a bit further, but each extra bit of distance needs a much bigger sheet than the last. If you write the pattern down, you can **predict how far a giant plane would fly without ever building it**. Scaling laws are that kind of pattern for AI models: size and practice go in, a prediction of how well it will do comes out.

**Example:** A bakery finds that doubling the oven size gives a small extra batch each time, not double. Having seen the pattern for 3 sizes, the owner can guess what a 10 times bigger oven would give, and decide if it is worth the cost.

## 🔵 2. How it works
1. Train many models of different sizes, each on different amounts of text (see [[Pre-training]]). Record the final loss (see [[Loss Function]]), where lower is better.
2. Plot loss against size on a special chart where both axes are logarithmic (each step is 10 times bigger, not 1 more).
3. The dots line up on a **straight line**. A straight line on that chart means a **power law**: multiply the size by a fixed amount, and the loss drops by a fixed fraction of what is left.
4. Three things can be scaled: **parameters** $N$ (model size), **data** $D$ (training tokens), and **compute** $C$ (arithmetic spent). All three help.
5. Because the lines are straight, you can **extend them** to guess how a much bigger model will do, before paying for it.
6. With a fixed budget you also ask: **how should I split it between a bigger model and more data?** The best split is called **compute-optimal**.

![[chinchilla-token-param-ratio.png]]
*How many training tokens per model parameter should you use as the compute budget grows? The green line is the answer from the original Chinchilla fit (Hoffmann et al.): it rises, from about 25 tokens per parameter at a small budget to well over 100 at a huge one. The teal line is a later re-fit by Epoch AI, which stays near 15 to 25 with a wide uncertainty band (the shaded area). The dashed line is the "20 tokens per parameter" rule of thumb. The two fits disagree, a reminder that scaling laws are measured, not exact. Image: Epoch AI, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Optimal_ratio_of_training_tokens_to_model_parameters_for_Chinchilla_scaling_law.png), CC BY 4.0.*

## 🟠 3. Technical detail
- **What was found:** for language models, the loss falls as a smooth power law in $N$, $D$ and $C$ over many orders of magnitude (Kaplan et al., 2020; checked against the paper: the trends span more than seven orders of magnitude, and the authors saw no sign of deviation from straight power-law trends, though they say the trends must eventually level off because language has non-zero entropy). Their advice was to spend extra compute mostly on a bigger model, with the data growing only slowly (about $C^{0.27}$, from the paper's text). Chinchilla later disagreed (next point).
- **Chinchilla (2022):** DeepMind (Hoffmann et al.; checked against the paper) trained over 400 models (70 million to over 16 billion parameters, on 5 to 500 billion tokens) and found current large models were **undertrained**, too big for their data. Their rule is to scale parameters and tokens equally: for every doubling of model size, double the tokens. Their own model, Chinchilla, has 70B parameters and 1.4T tokens, which is exactly 20 tokens per parameter, and it beat the 280B-parameter Gopher that used the same compute. The popular rule of thumb "about 20 tokens per parameter" is that ratio. I did not find it written as a general rule in the paper, so treat it as a rule of thumb.
- **The floor:** the loss cannot go below the unavoidable uncertainty in language (a model cannot predict truly random choices). Scaling only removes the part above that floor.
- **Loss, not skills:** the laws are for the loss number (see [[Perplexity]]). How loss turns into abilities on real tasks is less smooth, and whether abilities "emerge" suddenly or only look that way is debated (Wei et al., 2022 introduced the idea; Schaeffer et al., 2023 argued the jumps come from the choice of metric; I checked both abstracts, not the full papers).
- **Inference cost changes the answer:** compute-optimal means cheapest *to train*. If a model will answer billions of questions, it is cheaper overall to train a smaller model for longer, so that every answer costs less. Many recent models are trained on far more than 20 tokens per parameter for this reason (stated from memory; I tried to confirm it with the Llama 3 paper, but its abstract gives the model size, not the token count).
- **Data runs out:** the useful public text is finite. What happens when you repeat data, or use made-up (synthetic) data, is an active research area.
- **Fits are fragile:** as the picture shows, a re-fit of the same kind of curve gave a different answer for how the best ratio changes with scale. Treat any single set of fitted numbers with caution.
- **Beyond text:** similar power laws have been reported for images, video, image-and-text models and maths problems (Henighan et al., 2020; checked against the abstract). Speech is from memory.
- **Why it matters:** it explains the push for [[GPU]] clusters and bigger [[Transformer]] models, and the huge growth in training compute (see the chart in [[Pre-training]]).

```python
import numpy as np
# 1) a power law is a straight line on log-log axes (synthetic data, known answer: alpha = 0.1)
rng = np.random.default_rng(0)
N = 10.0 ** np.arange(6, 12); Linf, Nc, alpha = 1.7, 1e9, 0.1
L = Linf + (Nc / N) ** alpha; L = L * (1 + rng.normal(0, 0.001, len(N)))   # tiny measurement noise
slope, icpt = np.polyfit(np.log(N), np.log(L - Linf), 1)
print(round(-slope, 3), round(float(np.exp(icpt / -slope)) / 1e9, 2))      # 0.1 1.0   alpha and Nc/1e9, recovered
# 2) compute-optimal split of a budget C between model size N and data D, with C = 6 N D
E, A, B, a, b = 1.69, 406.4, 410.7, 0.34, 0.28      # illustrative constants (as remembered; real studies differ)
loss = lambda N, D: E + A / N ** a + B / D ** b
def best(C):
    Ns = np.logspace(6, 13, 4000); Ds = C / (6 * Ns); i = loss(Ns, Ds).argmin(); return Ns[i], Ds[i], loss(Ns[i], Ds[i])
n1, d1, l1 = best(1e21); n2, d2, l2 = best(1e23)
print(f"{n1:.2e} {d1:.2e} {d1/n1:.1f} {l1:.3f}")                    # 1.82e+09 9.14e+10 50.1 2.329
print(round(n2 / n1, 2), round(d2 / d1, 2), round(100 ** (b / (a + b)), 2), round(100 ** (a / (a + b)), 2))   # 8.0 12.5 8.0 12.5
print(round(b / (a + b), 3), round(a / (a + b), 3))                  # 0.452 0.548
```

Part 1 is made-up data with the answer built in ($\alpha=0.1$, $N_c=10^9$), and the straight-line fit recovers both. That is the whole method in miniature. Part 2 finds the best split of a budget by brute force. With these constants the best model for $10^{21}$ FLOP has about 1.8 billion parameters and 91 billion tokens. Multiplying the budget by 100 multiplies the best model size by 8.0 and the best data size by 12.5, exactly what the exponent formula below predicts. The best ratio here is 50 tokens per parameter, not 20, because the answer depends on the fitted constants. The picture above shows the same sensitivity. Needs only numpy.

## 🔴 4. Deep / formal
**A straight line in log-log.** If the part of the loss that more parameters can remove is a power law, then taking logs gives a straight line:

$$
\mathcal L(N)=L_\infty+\Big(\frac{N_c}{N}\Big)^{\alpha}\quad\Longleftrightarrow\quad\log\big(\mathcal L-L_\infty\big)=\alpha\log N_c-\alpha\log N
$$

**Where:**
- $N$ is the number of parameters, and $\mathcal L$ is the loss. $L_\infty$ is the floor (the best possible loss).
- $N_c$ is a constant that sets the scale, and $\alpha$ is the exponent, a small positive number.
- $\log(\mathcal L-L_\infty)$ is the log of the *reducible* loss. On the right the equation is a straight line in $\log N$ with slope $-\alpha$ and a fixed starting height.
- **Matters:** this is why people plot on log-log axes: the slope reads off $\alpha$ directly, and a line is easy to extend. If you forget to subtract $L_\infty$, the dots curve instead of lining up, and the fit gives the wrong $\alpha$.

**Example (tiny numbers):** $L_\infty=1.7$, $N_c=10^9$, $\alpha=0.1$ (illustrative). Use models with $10^9$, $10^{10}$ and $10^{11}$ parameters.
- Reducible loss $(N_c/N)^{0.1}$: $1.000$, $0.794$, $0.631$.
- Total loss: $2.700$, $2.494$, $2.331$.
- Each factor of 10 in size multiplies the reducible part by $0.794$ (a log base 10 change of $-0.1$, the slope $-\alpha$).
- Meaning: the loss falls from 2.700 to 2.331 over a 100 times bigger model, never reaching the floor of 1.7. Each tenfold step buys a smaller improvement than the last.

**Two things to scale.** A widely used form (the Chinchilla fit; the shape and the constants below are as printed in the Chinchilla paper, but they are only illustrative) has one power-law term for model size and one for data:

$$
\mathcal L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}
$$

**Where:**
- $N$ is the number of parameters. $D$ is the number of training tokens.
- $E$ is the floor. $A,B,\alpha,\beta$ are constants fitted to many training runs.
- $A/N^{\alpha}$ is the loss caused by the model being too small. $B/D^{\beta}$ is the loss caused by too little data.
- **Matters:** the two terms are separate. A huge model with too little data is held back by the data term, and a tiny model on endless data is held back by the model term. That is the mathematical reason for "balance both".

**Example (tiny numbers):** illustrative constants $E=1.69$, $A=406.4$, $B=410.7$, $\alpha=0.34$, $\beta=0.28$ (checked: these are the values printed in the Chinchilla paper's fit. A 2024 replication attempt, Besiroglu et al., says by its abstract that the fit does not describe the data well, so treat them as illustrative). Take $N=10^9$ and $D=2\times10^{10}$.
- Model term: $406.4/(10^9)^{0.34}=0.354$. Data term: $410.7/(2\times10^{10})^{0.28}=0.536$.
- $\mathcal L=1.69+0.354+0.536=\mathbf{2.580}$.
- Make the model 10 times bigger: model term $0.162$, $\mathcal L=2.388$. Double the data instead: data term $0.442$, $\mathcal L=2.485$.
- Meaning: here the 10 times bigger model helps more than doubling the data (2.388 against 2.485), but it also costs about 10 times more to train, while doubling the data costs only 2 times more. Which is the better buy depends on the budget (next formula).

**The best split of a budget.** Training cost is $C\approx6ND$ (see [[Pre-training]]). Minimise $\mathcal L(N,D)$ while keeping $6ND=C$. Setting the slope to zero gives power laws in the budget:

$$
N_{\text{opt}}\propto C^{\,\beta/(\alpha+\beta)},\qquad D_{\text{opt}}\propto C^{\,\alpha/(\alpha+\beta)}
$$

**Where:**
- $C$ is the compute budget. $N_{\text{opt}}$ and $D_{\text{opt}}$ are the best model size and the best data size for it. $\propto$ means "grows in proportion to".
- $\alpha$ and $\beta$ are the exponents from the loss formula above. The two exponents on the right add up to 1, so $N_{\text{opt}}\times D_{\text{opt}}$ grows exactly in proportion to $C$.
- **Matters:** it says how to spend extra budget: grow the model and the data **together**, each by a power of the budget. If $\alpha=\beta$, both exponents are $1/2$, so a 100 times bigger budget means a 10 times bigger model *and* 10 times more data. Spending it all on the model would waste it.

**Example (tiny numbers):** $\alpha=0.34$, $\beta=0.28$.
- Exponents: $0.28/0.62=\mathbf{0.452}$ for the model, $0.34/0.62=\mathbf{0.548}$ for the data.
- A budget 100 times bigger: model $\times\,100^{0.452}=\mathbf{8.0}$, data $\times\,100^{0.548}=\mathbf{12.5}$. Check: $8.0\times12.5=100$.
- Meaning: with these constants, data should grow a bit faster than the model. A brute-force search in the code gives exactly 8.0 and 12.5, which confirms the formula. Other fitted values give other exponents, which is why the green and teal lines in the picture differ.

**Further notes:**
- **Where it comes from:** the exponent formula follows from calculus (see [[Calculus]]) with a constraint; a short derivation is not shown here.
- **Open questions:** whether the same laws hold for much larger scales, for mixed data, or after the data runs out is not settled.
- **Fitting is statistics:** the constants come from fitting curves to noisy measurements (see [[Maximum Likelihood]] and [[Linear Regression]], the log-log line is a straight-line fit), so they carry uncertainty, as the wide band in the picture shows.
- **Other costs:** the formulas ignore inference cost, data quality, and hardware limits, all of which shape real decisions.

**What came before, and how this improved on it:**
- **Before:** choose model size and data amount by habit and trial.
- **Its limit:** training a very large model is too costly to get wrong.
- **What changed:** measure how the loss falls with model size, data and compute. Kaplan et al. found power laws across more than seven orders of magnitude and concluded that, for a fixed budget, training very large models on modest data and stopping early is best (arXiv 2001.08361, 23 January 2020, abstract read). Hoffmann et al. then found models were under-trained, and that model size and training tokens should grow equally (arXiv 2203.15556, 29 March 2022, abstract read).
- **How much better:** Chinchilla (70B parameters, 4 times more data) outperformed Gopher (280B) and GPT-3 (175B) on a wide range of tasks and reached 67.5% on MMLU, with the same compute as Gopher. The code below checks the arithmetic: 4 times fewer parameters and 4 times more tokens gives 1.0 times the compute under the usual rough rule that compute grows with parameters times tokens (the rule is from memory).

**Check it yourself:**

```python
# Arithmetic from the Chinchilla abstract (arXiv 2203.15556): same compute as Gopher, 70B parameters, 4x more data.
gopher_n, chin_n, data_ratio = 280, 70, 4
print("parameters:", gopher_n, "B ->", chin_n, "B  (", gopher_n // chin_n, "x smaller )")
print("compute in the usual rough rule (proportional to parameters x tokens):", chin_n * data_ratio / gopher_n, "x Gopher's")
print("doubling the model size should double the tokens (abstract): a 10x bigger model needs about 10x more tokens, a", 10 * 10, "x bigger compute bill")
```

## Related
- [[Pre-training]] · [[LLM]] · [[Transformer]] · [[Perplexity]] · [[Loss Function]] · [[GPU]] · [[Linear Regression]] · [[Maximum Likelihood]] · [[Calculus]] · [[Overfitting]] · [[Fine-tuning]] · [[Entropy]] · [[Mixture of Experts]] · [[Reasoning Model]] · [[LLaMA 2023]] · [[Chinchilla 2022]] · [[Scaling Laws 2020]]

## Sources
- Kaplan et al., *Scaling Laws for Neural Language Models*, OpenAI, 2020, arXiv:2001.08361 (checked: abstract and full text; read by text search, not line by line)
- Hoffmann et al., *Training Compute-Optimal Large Language Models* ("Chinchilla"), DeepMind, 2022, arXiv:2203.15556 (checked: abstract, the fit with $E=1.69$, $A=406.4$, $B=410.7$, 0.34, 0.28, and the 70B/1.4T model)
- Besiroglu et al., *Chinchilla Scaling: A replication attempt*, 2024, arXiv:2404.10102 (abstract-level summary only)
- Henighan et al., *Scaling Laws for Autoregressive Generative Modeling*, 2020, arXiv:2010.14701 (abstract only)
- Wei et al., *Emergent Abilities of Large Language Models*, 2022, arXiv:2206.07682, and Schaeffer et al., *Are Emergent Abilities of Large Language Models a Mirage?*, 2023, arXiv:2304.15004 (abstracts only)
- Epoch AI, *Chinchilla Scaling: A replication attempt* (the source of the picture), https://epochai.org/blog/chinchilla-scaling-a-replication-attempt
- Epoch AI, *Training compute of frontier AI models grows by 4-5x per year*
- Kaplan et al., arXiv 2001.08361 and Hoffmann et al., arXiv 2203.15556 (abstracts read for the history block; the Kaplan paper's body was read for [[Scaling Laws 2020]])

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm Scaling laws:::Patterns in how a model's loss falls as it gets bigger or sees more data. They let you predict how good a giant model would be without building it.

#flashcards/ml/formula #flashcards/topic/llm Write the power law for loss against model size and explain each symbol.
?
$$\mathcal L(N)=L_\infty+\Big(\frac{N_c}{N}\Big)^{\alpha}\quad\Longleftrightarrow\quad\log\big(\mathcal L-L_\infty\big)=\alpha\log N_c-\alpha\log N$$
- $N$: the number of parameters. $\mathcal L$: the loss. $L_\infty$: the floor, the best possible loss.
- $N_c$: a constant that sets the scale. $\alpha$: the exponent, a small positive number.
- $\log(\mathcal L-L_\infty)$: the log of the reducible loss. On the right the equation is a straight line in $\log N$ with slope $-\alpha$, which is how scaling laws are read off a log-log plot.

#flashcards/ml/formula #flashcards/topic/llm Write the Chinchilla-style loss with model size and data, and explain each symbol.
?
$$\mathcal L(N,D)=E+\frac{A}{N^{\alpha}}+\frac{B}{D^{\beta}}$$
- $N$: the number of parameters. $D$: the number of training tokens. $E$: the floor.
- $A, B, \alpha, \beta$: constants fitted to many training runs.
- $A/N^\alpha$: the loss caused by the model being too small. $B/D^\beta$: the loss caused by too little data.
- The two terms are separate. A huge model with too little data (or the reverse) is stuck on the other term.

#flashcards/ml/formula #flashcards/topic/llm Write the compute-optimal sizes and explain each symbol.
?
$$N_{\text{opt}}\propto C^{\,\beta/(\alpha+\beta)},\qquad D_{\text{opt}}\propto C^{\,\alpha/(\alpha+\beta)}$$
- $C$: the compute budget. $N_{\text{opt}}$, $D_{\text{opt}}$: the best model size and best data size for that budget. $\propto$: "grows in proportion to".
- $\alpha$, $\beta$: the exponents from the loss formula above.
- The two exponents add up to 1, so $N_{\text{opt}}\times D_{\text{opt}}$ grows exactly in step with $C$.

#flashcards/ml/cloze #flashcards/topic/llm A scaling law is a straight line on a ==log-log== plot, and the loss never goes below the ==floor==.

#flashcards/ml/example #flashcards/topic/llm $L_\infty=1.7$, $N_c=10^9$, $\alpha=0.1$ (illustrative). What is the loss at $10^9$, $10^{10}$ and $10^{11}$ parameters?::Reducible part (N_c/N)^0.1: 1.000, 0.794, 0.631. Total loss 2.700, 2.494, 2.331. Each factor of 10 in size multiplies the reducible part by 0.794.

#flashcards/ml/example #flashcards/topic/llm With $\alpha=0.34$ and $\beta=0.28$, how should model size and data grow if the budget is 100 times bigger?::Exponents 0.28/0.62 = 0.452 for the model and 0.34/0.62 = 0.548 for the data. Model × 100^0.452 = 8.0, data × 100^0.548 = 12.5. 8.0 × 12.5 = 100.

#flashcards/ml/image #flashcards/topic/llm ![[chinchilla-token-param-ratio.png]]
What does this picture show?
?
How many training tokens per model parameter to use as the compute budget grows. The green line (original Chinchilla fit) rises from about 25 tokens per parameter at a small budget to well over 100 at a huge one. The teal line (a later Epoch AI re-fit) stays near 15 to 25 with a wide uncertainty band. The dashed line is the "20 tokens per parameter" rule of thumb. The two fits disagree: scaling laws are measured, not exact.
