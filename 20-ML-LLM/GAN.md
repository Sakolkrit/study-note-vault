---
tags: [machine-learning, deep-learning, generative-models, GAN, probability]
status: growing
level: 4
tier: 3
---

# GAN

A GAN (generative adversarial network) is **two networks that compete**: a *generator* that makes fake examples, and a *discriminator* that tries to tell fakes from real ones. Each one's struggle makes the other better, until the fakes look real.

## 🟢 1. Simple (age 12)
Picture an art forger and a detective. The forger paints fake paintings. The detective looks at a mix of real and fake paintings and says which are which. Every time the detective catches a fake, the forger learns what gave it away and paints a better one. Every time a fake slips through, the detective learns to look harder. After lots of rounds, the forger's paintings are so good that the detective can only guess.

**Example:** Think of a game of "spot the fake coin". You make fake coins. A friend sorts a pile of real and fake ones. Each time your friend catches yours, you fix the thing they noticed: the weight, the shine. In the end your friend gets it right only half the time, which is no better than flipping a coin.

## 🔵 2. How it works
1. **Two networks:** the generator $G$ and the discriminator $D$, both [[Neural Network]]s. $G$ takes random numbers (noise) and turns them into a fake example. $D$ takes an example and says how likely it is to be real, as a number between 0 and 1.
2. **Train the detective:** show $D$ a batch of real examples (the right answer is "real", 1) and a batch of fakes from $G$ (the right answer is "fake", 0). Improve $D$ with [[Backpropagation]] and a [[Loss Function]].
3. **Train the forger:** make new fakes and ask $D$ about them. Change $G$ so that $D$ says "real" more often. The learning signal reaches $G$ *through* $D$.
4. **Take turns** many times. Neither network is trained once and then finished. Each keeps changing the target for the other.
5. **When it works,** the fakes look like the real data, and $D$ ends up saying about 0.5 for everything, because it can no longer tell.
6. **To make new things:** throw away $D$, feed fresh random noise into $G$, and use what comes out.

![[gan-diagram.svg]]
*The setup. Random noise goes into the generator, which makes a fake. The discriminator sees either a fake or a real example and says "real or fake". (The same figure is used in [[Generative Models]].) Image: Zhang, Lipton, Li and Smola (*Dive into Deep Learning*), [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Generative_adversarial_network.svg), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **A game, not one loss:** $G$ and $D$ have opposing goals. Training is a back-and-forth, not the usual single descent of [[Gradient Descent]]. It can wobble, circle around, or break down instead of settling.
- **No direct score:** a GAN does not give you the probability of an example, so you cannot check how well it fits the way you can with [[Maximum Likelihood]] models. People judge it by looking at samples and by special metrics (not covered here).
- **Known problems:**
  - **Instability:** if $D$ gets too good too fast, $G$ gets almost no useful signal (shown in the code and in equation 2 below).
  - **Mode collapse:** $G$ finds a few outputs that fool $D$ and only makes those. The code below shows one such run.
  - **Sensitivity:** results depend on the network design, learning rate and [[Normalization]] choices. It often takes many tries.
- **Fixes people use (from memory, not covered in detail):** the non-saturating generator loss (below), a Wasserstein loss with a "critic" instead of a probability-style discriminator, penalties on the discriminator's slope, and careful [[Learning Rate]] and [[Adam Optimizer]] settings.
- **Where GANs shine:** sharp, realistic images, and fast generation, since one pass through $G$ gives a sample. Modern image generators are more often [[Diffusion Models]], which train more steadily but sample slowly (the popularity claim is from memory).
- **Conditional GANs:** feed a label or an input image into both networks so you can ask for a specific kind of output (for example turning a sketch into a photo; from memory).
- **Compared with an [[Autoencoder]]:** an autoencoder is trained to rebuild its own input. A GAN's generator never sees a real example directly; it only hears the detective's verdict.

The code trains a tiny GAN on one number at a time. Real data is a bell curve with mean 4 and spread 1.25. It also checks the gradient the forger receives when the detective is very sure, and then tries two clumps of data at $-3$ and $+3$ to look for mode collapse.

```python
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
def mlp(i, o): return nn.Sequential(nn.Linear(i, 32), nn.LeakyReLU(0.2), nn.Linear(32, 32), nn.LeakyReLU(0.2), nn.Linear(32, o))
def train(real_fn, steps=4000, seed=0, nonsat=True):
    torch.manual_seed(seed)
    G, D = mlp(1, 1), mlp(1, 1)
    og, od = torch.optim.Adam(G.parameters(), 1e-3, betas=(0.5, 0.9)), torch.optim.Adam(D.parameters(), 1e-3, betas=(0.5, 0.9))
    for _ in range(steps):
        x = real_fn(128); fake = G(torch.randn(128, 1))
        ld = F.binary_cross_entropy_with_logits(D(x), torch.ones(128, 1)) + F.binary_cross_entropy_with_logits(D(fake.detach()), torch.zeros(128, 1))
        od.zero_grad(); ld.backward(); od.step()                           # detective: real -> 1, fake -> 0
        out = D(G(torch.randn(128, 1)))
        lg = F.binary_cross_entropy_with_logits(out, torch.ones(128, 1)) if nonsat else -F.binary_cross_entropy_with_logits(out, torch.zeros(128, 1))
        og.zero_grad(); lg.backward(); og.step()                           # forger: make D say "real"
    return G, D
real = lambda n: 4 + 1.25 * torch.randn(n, 1)                              # real data ~ N(4, 1.25^2)
G, D = train(real)
with torch.no_grad(): g = G(torch.randn(20000, 1)); d_real = torch.sigmoid(D(real(5000))).mean().item(); d_fake = torch.sigmoid(D(g[:5000])).mean().item()
print('fake mean/std', round(g.mean().item(), 2), round(g.std().item(), 2))     # real is 4 / 1.25
print('D on real / fake', round(d_real, 2), round(d_fake, 2))                   # near 0.5 / 0.5 when the forger wins
# gradient the forger receives when the detective is sure (D = 0.01), via autograd on the logit
a = torch.tensor(float(np.log(0.01 / 0.99)), requires_grad=True)
(torch.log(1 - torch.sigmoid(a))).backward(); sat = a.grad.item(); a.grad = None
(-torch.log(torch.sigmoid(a))).backward(); non = a.grad.item()
print('grad on the logit, saturating vs non-saturating:', round(sat, 3), round(non, 3))
# two clumps at -3 and +3: does the forger cover both?
two = lambda n: (torch.randint(0, 2, (n, 1)) * 6 - 3).float() + 0.3 * torch.randn(n, 1)
for seed in range(3):
    G2, _ = train(two, 4000, seed)
    with torch.no_grad(): y = G2(torch.randn(5000, 1))
    print('seed', seed, 'share below 0:', round((y < 0).float().mean().item(), 2), 'near -3 / +3:', round((abs(y + 3) < 1).float().mean().item(), 2), round((abs(y - 3) < 1).float().mean().item(), 2))
# fake mean/std 3.83 0.94
# D on real / fake 0.51 0.51
# grad on the logit, saturating vs non-saturating: -0.01 -0.99
# seed 0 share below 0: 0.58 near -3 / +3: 0.57 0.4
# seed 1 share below 0: 0.54 near -3 / +3: 0.52 0.45
# seed 2 share below 0: 0.0 near -3 / +3: 0.0 1.0
```
What the output shows (PyTorch 2.8.0, CPU; it takes a few minutes, and a GAN's exact numbers can change between versions and machines):
- The forger learned a bell curve with mean 3.83 and spread 0.94, close to the real 4 and 1.25 but a bit too narrow and a bit off in the middle. This is a mild form of the "less variety than the real data" problem.
- The detective ended at 0.51 for real and 0.51 for fake, the "I can only guess" end of the game.
- With the two clumps, seeds 0 and 1 cover both clumps, with 52 to 57% near $-3$ and 40 to 45% near $+3$, but **seed 2 collapsed**: every sample is near $+3$ and none near $-3$. Same code, same data, different random start. That is mode collapse, and why GAN results are described as fragile. Three seeds is a small test, not a measurement of how often it happens.

## 🔴 4. Deep / formal

**The best discriminator.** Let $p_{\text{data}}(x)$ be how likely a real example is to be $x$, and $p_g(x)$ how likely the generator is to make $x$. If $G$ is fixed, the best possible detective answers

$$D^*(x)=\frac{p_{\text{data}}(x)}{p_{\text{data}}(x)+p_g(x)}$$

**Where:**
- $p_{\text{data}}(x)$ is the chance that real data contains $x$. **Matters:** the more real-looking $x$ is, the higher $D^*$ goes.
- $p_g(x)$ is the chance that the generator makes $x$. **Matters:** the more the generator already makes $x$, the lower $D^*$ goes, because $x$ is more likely to be fake.
- $D^*(x)$ is the best possible "probability real". **Matters:** it is the target a real detective is trying to reach. Where real and fake are equally likely, it says 0.5 and the detective is just guessing.

**Example (tiny numbers):** At one spot $x$, real data has weight $0.6$ and the generator has weight $0.2$.
1. $D^*=0.6/(0.6+0.2)=0.75$.
2. If the generator improves to match the real data there, both are $0.3$: $D^*=0.3/(0.3+0.3)=0.5$.
3. Meaning: the best detective says "75% real" while fakes are rare at this spot, and falls to 50% once the generator copies the real data. A final value of 0.5 everywhere is the sign that $G$ has caught up, which is what the code shows (0.51).

**The generator's loss.** Write $D(G(z))=\sigma(a)$ where $a$ is the detective's raw score (logit) for a fake. The generator can be trained in two ways. The original game uses the *saturating* loss $\log(1-\sigma(a))$ (to be minimised). The common fix uses the *non-saturating* loss

$$\mathcal{L}_G=-\log D(G(z)),\qquad \frac{\partial \mathcal{L}_G}{\partial a}=-(1-\sigma(a)),\qquad \frac{\partial}{\partial a}\log(1-\sigma(a))=-\sigma(a)$$

**Where:**
- $z$ is the random noise, and $G(z)$ the fake it becomes. **Matters:** it is the only input $G$ has.
- $a$ is the detective's raw score for the fake, and $\sigma(a)=D(G(z))$ is the chance it says "real". **Matters:** the gradient of the loss with respect to $a$ is the learning signal that flows back into $G$.
- $-(1-\sigma(a))$ is the non-saturating gradient. **Matters:** it is big when the detective is sure the fake is fake ($\sigma(a)\approx0$), which is exactly when the forger most needs a push.
- $-\sigma(a)$ is the saturating gradient. **Matters:** it is almost zero in that same situation, so a forger who is losing badly learns nothing. This is the early-training failure that the fix removes.

**Example (tiny numbers):** The detective is sure: $D(G(z))=0.01$, so $a=\ln(0.01/0.99)=-4.595$.
1. Saturating gradient: $-\sigma(a)=-0.01$.
2. Non-saturating gradient: $-(1-\sigma(a))=-0.99$. Autograd in the code gives the same $-0.01$ and $-0.99$.
3. If the detective is unsure ($D=0.5$), both are $-0.5$.
4. Meaning: when the forger is being caught every time, the original loss gives it a push of 0.01 and the fix gives 0.99, about 100 times more. At 0.5 they agree, so the fix only changes the early, one-sided phase of the game.

**What the game is minimising.** Plug the best detective into the original game value $V(D,G)=\mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)]+\mathbb{E}_{z}[\log(1-D(G(z)))]$. What is left depends only on $G$:

$$V(D^*,G)=-\log 4+2\,\mathrm{JS}\bigl(p_{\text{data}}\,\|\,p_g\bigr),\qquad \mathrm{JS}(p\|q)=\tfrac12\mathrm{KL}(p\|m)+\tfrac12\mathrm{KL}(q\|m),\ \ m=\tfrac12(p+q)$$

**Where:**
- $\mathrm{JS}$ is the Jensen-Shannon divergence, a symmetric, smoothed cousin of [[KL Divergence]]. **Matters:** it is zero only when $p_g=p_{\text{data}}$ and grows as they differ. Without it the game would have no clear meaning of "closer".
- $m$ is the average of the two distributions. **Matters:** it keeps both KL terms finite even where one distribution has zero probability.
- $-\log 4=-1.386$ is the value at the perfect match. **Matters:** it is the floor: $V$ cannot go below it, and a perfect forger reaches it, matching the earlier $2\ln0.5$ figure in [[Generative Models]].

**Example (tiny numbers):** Three possible outcomes. Real data: $p=(0.5,0.5,0)$. Generator: $q=(0.25,0.25,0.5)$.
1. Average $m=(0.375,0.375,0.25)$.
2. $\mathrm{KL}(p\|m)=2\times0.5\ln(0.5/0.375)=0.2877$. $\mathrm{KL}(q\|m)=2\times0.25\ln(0.25/0.375)+0.5\ln(0.5/0.25)=-0.2027+0.3466=0.1438$.
3. $\mathrm{JS}=\tfrac12(0.2877+0.1438)=0.2158$.
4. $V(D^*,G)=-1.3863+2\times0.2158=-0.9548$. Computing $\sum p\ln\frac{p}{p+q}+\sum q\ln\frac{q}{p+q}$ directly gives the same $-0.9548$.
5. Meaning: the further the generator is from the real data, the higher the value. The forger lowers it toward $-1.386$ by matching the data. If $p$ and $q$ never overlap at all, $\mathrm{JS}=\ln2=0.693$ and $V=0$: the detective is perfect, and the number stops saying *how* far apart they are.

**Why training is hard (from memory, not derived here):**
- The "never overlap" case is common early on: real data and fakes sit on thin, separate surfaces in a huge space. Then JS stays flat at $\ln 2$ and gives no direction, which is one reason for the Wasserstein variant (see the image below).
- Game dynamics need not converge. Plain simultaneous gradient steps can circle a balance point instead of reaching it.
- The equations above assume $D$ is the best detective for the current $G$. A real $D$ is a small network trained for a few steps, so theory and practice differ.

![[gan-wgan-critic-vs-discriminator.svg]]
*Two possible judges for the same pair of distributions. The blue bell curve is the generator's distribution and the orange curve is the real (reference) distribution. The olive curve is the optimal GAN discriminator (rescaled): it bends back and forth and levels off near 0.4 at the right, so far out it carries little sense of direction (it also depends on the 0.4 rescaling). The purple line is the optimal Wasserstein critic (rescaled): a steady slope that keeps pointing toward the real data. Image: Cosmia Nebula, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Wasserstein-GAN_critic_vs_GAN_discriminator.svg), CC BY-SA 4.0.*

**What came before, and how this improved on it:**
- **Before:** generative models that needed Markov chains or unrolled approximate inference to train or sample (named in the abstract), or a hand-picked simple distribution.
- **Its limit:** costly to train and sample, or too simple to match the data.
- **What changed:** train a generator against a discriminator in a minimax game, with no chains needed. The abstract says that in the space of all functions the game has a unique solution, with the generator matching the data and the discriminator at $1/2$ everywhere (Goodfellow et al., arXiv 1406.2661, 10 June 2014, abstract read).
- **How much better:** in the code below the data has two clumps, at $-2$ and $2$. One fitted bell curve puts only 48.2% of its samples near a real clump (the rest fall in the empty middle). The GAN puts 100% near a clump. **But all of them sit in the left clump** (share left of 0 is 1.0, against about 0.5 in the data). That is *mode collapse*, a well-known GAN failure. It appeared on the first seed I ran and I did not tune it away. So this toy shows sharper samples and a missing mode.

**Check it yourself:**

```python
import numpy as np, torch
torch.manual_seed(1); rng = np.random.default_rng(1)
real = lambda n: torch.tensor(np.where(rng.random((n, 1)) < .5, -2., 2.) + .3 * rng.standard_normal((n, 1)), dtype=torch.float32)
G = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
D = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1), torch.nn.Sigmoid())
og, od = torch.optim.Adam(G.parameters(), 2e-3), torch.optim.Adam(D.parameters(), 2e-3)
bce = torch.nn.BCELoss()
for _ in range(4000):
    r = real(64); f = G(torch.randn(64, 2))
    od.zero_grad(); (bce(D(r), torch.ones(64, 1)) + bce(D(f.detach()), torch.zeros(64, 1))).backward(); od.step()
    og.zero_grad(); bce(D(f), torch.ones(64, 1)).backward(); og.step()
near = lambda s: float((((s - 2).abs() < 1) | ((s + 2).abs() < 1)).float().mean())
gan = G(torch.randn(5000, 2)).detach()
data = real(5000)
gauss = data.mean() + data.std() * torch.randn(5000, 1)               # one fitted bell curve
print("share of samples near a real mode: real data", round(near(data), 3), " one fitted Gaussian", round(near(gauss), 3), " GAN", round(near(gan), 3))
print("GAN samples left of 0:", round(float((gan < 0).float().mean()), 3), "(real: about 0.5)")
```

## Related
[[Generative Models]] · [[Diffusion Models]] · [[Autoencoder]] · [[KL Divergence]] · [[Entropy]] · [[Neural Network]] · [[Backpropagation]] · [[Loss Function]] · [[Gradient Descent]] · [[Adam Optimizer]] · [[Normalization]] · [[Maximum Likelihood]] · [[Probability]] · [[Softmax]] · [[Learning Rate]] · [[Variational Autoencoder]] · [[Convolution]]
- [[GAN 2014]]

## Sources
- Goodfellow and colleagues (2014), "Generative Adversarial Nets", the paper that introduced the game and the best-discriminator and JS results (cited from memory; not opened for this note).
- Wasserstein GAN, Arjovsky, Chintala and Bottou (2017) (from memory).
- Figures: [Generative_adversarial_network.svg](https://commons.wikimedia.org/wiki/File:Generative_adversarial_network.svg) (CC BY-SA 4.0) and [Wasserstein-GAN_critic_vs_GAN_discriminator.svg](https://commons.wikimedia.org/wiki/File:Wasserstein-GAN_critic_vs_GAN_discriminator.svg) (CC BY-SA 4.0), both on Wikimedia Commons.
- The code, the gradient values and the Example numbers were run for this note (PyTorch 2.8.0, NumPy). The curves in the second figure are rescaled for display, so only their shapes matter.
- Goodfellow et al., arXiv 1406.2661 (abstract read for the history block)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/generative GAN (generative adversarial network):::An art forger and a detective. The forger (generator) paints fakes. The detective (discriminator) says real or fake. Each catch teaches the forger to paint better, and each miss teaches the detective to look harder.

#flashcards/ml/formula #flashcards/topic/generative Write the best discriminator and explain each symbol.
?
$$D^*(x)=\frac{p_{\text{data}}(x)}{p_{\text{data}}(x)+p_g(x)}$$
- $p_{\text{data}}(x)$: the chance that real data contains $x$. The more real-looking $x$ is, the higher $D^*$ goes.
- $p_g(x)$: the chance that the generator makes $x$. The more the generator already makes $x$, the lower $D^*$ goes, because $x$ is more likely fake.
- $D^*(x)$: the best possible "real" probability for $x$. It falls to 0.5 once the generator matches the real data.

#flashcards/ml/formula #flashcards/topic/generative Write the non-saturating generator loss and its gradients, and explain each symbol.
?
$$\mathcal{L}_G=-\log D(G(z)),\qquad \frac{\partial \mathcal{L}_G}{\partial a}=-(1-\sigma(a)),\qquad \frac{\partial}{\partial a}\log(1-\sigma(a))=-\sigma(a)$$
- $z$: the random noise. $G(z)$: the fake it becomes. It is the only input $G$ has.
- $a$: the detective's raw score for the fake. $\sigma(a)=D(G(z))$: the chance it says "real".
- $-(1-\sigma(a))$: the learning signal of the non-saturating loss. It is large when the detective is sure it is fake.
- $-\sigma(a)$: the signal of the original loss. It is tiny exactly when the forger is losing, so learning stalls.

#flashcards/ml/formula #flashcards/topic/generative Write the GAN value at the best discriminator and explain each symbol.
?
$$V(D^*,G)=-\log 4+2\,\mathrm{JS}\bigl(p_{\text{data}}\,\|\,p_g\bigr),\qquad \mathrm{JS}(p\|q)=\tfrac12\mathrm{KL}(p\|m)+\tfrac12\mathrm{KL}(q\|m),\ \ m=\tfrac12(p+q)$$
- $\mathrm{JS}$: the Jensen-Shannon divergence, a symmetric, smoothed cousin of KL divergence. It is 0 only when $p_g=p_{\text{data}}$ and grows as they differ.
- $m$: the average of the two distributions. It keeps both KL terms finite even where one distribution is 0.
- So the generator's best possible game result is $-\log4$, reached when the fakes match the real data.

#flashcards/ml/cloze #flashcards/topic/generative The non-saturating generator loss gives a ==large== gradient when the detective is sure the fake is fake.

#flashcards/ml/example #flashcards/topic/generative At one spot real data has weight 0.6 and the generator 0.2. What is $D^*$, and what if the generator matches (0.3 and 0.3)?::D* = 0.6/(0.6 + 0.2) = 0.75. After matching: 0.3/(0.3 + 0.3) = 0.5. The best detective says "75% real" while fakes are rare there, and falls to 50% once they match.

#flashcards/ml/example #flashcards/topic/generative The detective is sure: $D(G(z))=0.01$. What are the two gradients, and if $D=0.5$?::a = ln(0.01/0.99) = −4.595. Saturating: −σ(a) = −0.01. Non-saturating: −(1 − σ(a)) = −0.99. If D = 0.5, both are −0.5. The saturating one vanishes when the forger is being caught.

#flashcards/ml/example #flashcards/topic/generative Real $p=(0.5,0.5,0)$ and generator $q=(0.25,0.25,0.5)$. What is the JS divergence?::m = (0.375, 0.375, 0.25). KL(p‖m) = 0.2877. KL(q‖m) = 0.1438. JS = ½ (0.2877 + 0.1438) = 0.2158.

#flashcards/ml/image #flashcards/topic/generative ![[gan-diagram.svg]]
What does this picture show?
?
The setup. Random noise goes into the generator, which makes a fake. The discriminator sees either a fake or a real example and says "real or fake".

#flashcards/ml/image #flashcards/topic/generative ![[gan-wgan-critic-vs-discriminator.svg]]
What does this picture show?
?
Two possible judges for the same pair of distributions. The blue bell curve is the generator's distribution and the orange curve is the real one. The olive curve is the optimal GAN discriminator (rescaled): it bends back and forth and levels off near 0.4 at the right, so far out it carries little sense of direction. The purple line is the optimal Wasserstein critic (rescaled): a steady slope that keeps pointing towards the real data.
