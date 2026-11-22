---
tags: [machine-learning, deep-learning, generative-models, autoencoder, GAN, diffusion, unsupervised-learning]
status: growing
level: 4
tier: 3
---

# Generative Models

## 🟢 1. Simple (age 12)
Most models answer a question about something that already exists ("is this a cat?"). A **generative model** does the opposite: it **makes new things** that look like the examples it studied: new pictures, new sentences, new music. To do that, it first has to learn **what the examples are like**, deeply enough to invent new ones.

**Example:** A child who has seen hundreds of dog drawings can draw a *new* dog that nobody has drawn before. It is not a copy of any one drawing. The child has absorbed what "dog-ness" looks like (four legs, floppy ears, a tail) and can mix it into something fresh. Generative models try to do the same with numbers.

## 🔵 2. How it works
All generative models learn from a big pile of examples **without needing labels** (see [[Unsupervised Learning]] and [[Self-Supervised Learning]]). They then learn a way to produce new examples that feel like they came from the same pile. There are several families. Three well-known ones are below.

**1. Autoencoder: squeeze, then rebuild.**
1. An **encoder** squeezes each input (say a picture) down into a short list of numbers, the **code**.
2. A **decoder** tries to rebuild the original picture from only that short code.
3. Training makes the rebuilt picture as close as possible to the original. To succeed, the code must keep the *important* information.
4. A plain autoencoder mainly **compresses**. A variation, the **variational autoencoder (VAE)**, arranges the codes so that you can pick a *new random code* and decode it into a *new* picture.

![[autoencoder-schema.png]]
*An autoencoder. The input layer ($X$) is squeezed by the encoder into a narrow middle "code" ($h$), and the decoder expands it back into an output ($X'$) that should match the input. The narrow middle forces the network to keep only what matters. Image: Michela Massi, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Autoencoder_schema.png), CC BY-SA 4.0.*

**2. GAN: a forger and a detective.**
1. Two networks play a game. The **generator** turns random noise into a fake example. The **discriminator** looks at an example and guesses "real" or "fake".
2. The discriminator is trained to catch the fakes. The generator is trained to **fool** the discriminator.
3. As each gets better, the fakes get more realistic, like a forger and a detective improving by competing.
4. GAN stands for *generative adversarial network*.

![[gan-diagram.svg]]
*A GAN. Random noise $z$ goes into the generator, which produces a fake $G(z)$. The discriminator receives either a fake $G(z)$ or a real example $x$ and must say "real or fake". Image: Zhang, Lipton, Li and Smola (*Dive into Deep Learning*), [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Generative_adversarial_network.svg), CC BY-SA 4.0.*

**3. Diffusion model: learn to remove noise.**
1. Take a real picture and add a little random noise. Then a little more, and so on, until only pure noise (static) is left. This is the **forward** process, and it needs no learning.
2. Train a network to do the **reverse**: given a noisy picture, **guess the noise** (or a slightly less noisy picture).
3. To make a new picture, start from pure random noise and **remove noise step by step**, using the network's guesses, until a clean picture appears.
4. Many popular image generators use this idea (the exact designs vary, and the details are in the papers).

![[diffusion-forward.png]]
*The forward process: a photo of a rhinoceros ($T=0$) gets more and more noise added at each step until, at the end, it is just noise. Image: MrAlanKoh, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Example_of_Forward_Diffusion_Models.png), CC BY-SA 4.0 (derived from a photograph of a black rhinoceros, credited on the file page).*

![[diffusion-denoising.png]]
*The reverse process: a denoising model takes a noisy image ($Z_{T=n}$) and produces a slightly cleaner one ($Z_{T=n-1}$), repeating until a picture emerges. The right side shows that different starting noise gives variations of the picture, and the left shows the attempt to rebuild the original. Image: MrAlanKoh, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Example_of_Denoising_Diffusion_models.png), CC BY-SA 4.0. The picture is a cartoon of the idea, not the output of a real model.*

## 🟠 3. Technical detail
- **The goal:** learn the probability distribution of the data ([[Probability]]), $p(x)$, well enough to **sample** new $x$ from it.
- **Autoencoders:**
  - **Bottleneck:** the code is much smaller than the input. If it were as large, the network could just copy.
  - A **linear** autoencoder with a squared-error loss ends up doing the same job as [[PCA]] (a known result, checked in the code below on a tiny example).
  - **Denoising autoencoders** learn from corrupted inputs, so they learn to repair them.
  - **VAE:** the encoder outputs a mean and a spread ([[Normal Distribution]]) for each code number, and an extra loss term keeps the codes close to a simple bell-shaped distribution, so that random codes decode into plausible new things. (The loss formula is in the paper; it is not written out here.)
  - **Uses:** compression, denoising, finding anomalies (items that rebuild badly), and learning compact representations.
- **GANs:**
  - Sharp, realistic images were an early success.
  - **Training is tricky:** the two networks must stay in balance. Known problems are instability and **mode collapse** (the generator only produces a few kinds of output).
  - No direct score of how well it models the data; judged by looking at samples and by special metrics.
- **Diffusion models:**
  - Training is stable (it is a plain regression problem: guess the noise, with a [[Loss Function]] such as squared error).
  - **Sampling is slow** because it needs many denoising steps (tens to hundreds). Faster samplers and distillation methods reduce this.
  - They can be **guided** by text: the text is turned into numbers ([[Embedding Model]]) and fed to the denoiser, so the picture follows the prompt. Large text-to-image systems use this (the specific designs differ; check the model you care about).
  - The denoiser is often a U-shaped [[Convolutional Neural Network]] with [[Residual Connection]]s, or increasingly a [[Transformer]] (from memory; designs vary).
- **Autoregressive models (a fourth family):** generate one piece at a time, each from the pieces before. This is how an [[LLM]] writes text, trained by predicting the next token ([[Self-Supervised Learning]]). So an LLM is also a generative model.
- **Evaluating generative models is hard:** there is no single right answer to compare with. People use human judgement, scores comparing the statistics of real and generated samples, and for language [[Perplexity]].
- **Risks:** generated images, audio and text can be used to mislead (fakes), can copy parts of the training data, and can reflect the biases and mistakes in it. These are serious concerns and are only named here.

```python
import numpy as np

# 1) autoencoder: a 2-number input squeezed into a 1-number code (mean) and rebuilt
x = np.array([2., 4.]); z = x.mean()
print(z, ((x - np.array([z, z])) ** 2).sum())         # 3.0 2.0   code 3, rebuild (3,3): squared error 2
# an input that is already "all equal" survives the squeeze perfectly: (3,3) -> code 3 -> (3,3): error 0

# a tiny LINEAR autoencoder trained by gradient descent matches PCA's best error on the PCA note's points
X = np.array([[0., 0.], [2., 1.], [4., 3.], [6., 4.]]); Xc = X - X.mean(0)
rng = np.random.default_rng(0); e = rng.normal(0, .5, 2); d = rng.normal(0, .5, 2)   # encoder, decoder weights
for step in range(5000):
    z = Xc @ e; G = 2 * (np.outer(z, d) - Xc)         # gradient of the squared error w.r.t. the rebuilt points
    gd = (G * z[:, None]).sum(0); ge = Xc.T @ (G @ d)
    d -= 0.002 * gd; e -= 0.002 * ge
print(((np.outer(Xc @ e, d) - Xc) ** 2).sum().round(3))   # 0.134   the same as PCA's error with one direction

# 2) GAN game value
print(np.log(.9) + np.log(.8), 2 * np.log(.5))        # -0.3285 -1.3863

# 3) diffusion: noising a number x0 = 1 with noise eps = 0.5
for ab in (0.81, 0.01):
    print(ab, np.sqrt(ab), round(np.sqrt(1 - ab), 3), round(np.sqrt(ab) * 1 + np.sqrt(1 - ab) * .5, 3))
# 0.81 0.9 0.436 1.118       mostly signal
# 0.01 0.1 0.995 0.597       mostly noise
print(0.9 ** 3)                                       # 0.729 (up to float rounding)   three steps of beta = 0.1
```

## 🔴 4. Deep / formal
**Autoencoder: the reconstruction loss.** With an encoder $f$ and a decoder $g$ (both [[Neural Network]]s) and $n$ training examples $x_i$, the autoencoder minimises the average squared difference between each input and its rebuild:

$$
\mathcal L=\frac1n\sum_{i=1}^{n}\big\lVert x_i-g\big(f(x_i)\big)\big\rVert^{2}
$$

**Where:**
- $x_i$ is the $i$-th input (a vector, see [[Vector]]) and $n$ is how many examples there are.
- $f(x_i)$ is the **code**: the short list of numbers the encoder produces. $g(f(x_i))$ is the decoder's rebuild, which should resemble $x_i$.
- $\lVert\cdot\rVert^2$ is the squared length of the difference vector: the sum of the squared differences of its entries. $\tfrac1n\sum_i$ averages over the examples.
- **Matters:** the loss is 0 only when every rebuild is perfect. Because the code is **narrower** than the input, perfect rebuilding is possible only if the code captures the real structure in the data, which is why a good code is a compact description of the data.
- **Matters:** no labels appear anywhere. The input is its own target, which is why this is [[Unsupervised Learning]] (or self-supervised).
- If $f$ and $g$ are linear, the best code spans the same space as the top [[PCA]] directions.

**Example (tiny numbers):** an encoder that outputs the average of two numbers, and a decoder that repeats it twice: $f(x)=(x_1+x_2)/2$, $g(z)=(z,z)$.
- Input $x=(2,4)$: code $z=\mathbf{3}$, rebuild $(3,3)$. Squared error $(2-3)^2+(4-3)^2=\mathbf{2}$.
- Input $x=(3,3)$: code $3$, rebuild $(3,3)$. Error $\mathbf{0}$.
- Meaning: one number can describe an input perfectly when its two entries are equal, but it loses information when they differ. A trained autoencoder learns a code that works well for the kinds of inputs it was trained on. In the code above, a one-number linear autoencoder on the four PCA points reaches the same squared error, $0.134$, as PCA.

**GAN: the two-player game.** The discriminator $D$ outputs the chance that its input is real; the generator $G$ turns noise $z$ into fakes. The game is:

$$
\min_G\ \max_D\ \ \mathbb E_{x\sim p_{\text{data}}}\big[\log D(x)\big]+\mathbb E_{z\sim p_z}\big[\log\big(1-D(G(z))\big)\big]
$$

**Where:**
- $x\sim p_{\text{data}}$ means "$x$ is a real example drawn from the data". $z\sim p_z$ means "$z$ is random noise". $\mathbb E[\cdot]$ is the average over many draws (see [[Probability]]).
- $D(x)$ is the discriminator's chance, between 0 and 1, that $x$ is real. $G(z)$ is the generator's fake.
- $\log D(x)$ is large (close to 0 from below) when $D$ gives real examples a high chance, and very negative when $D$ is wrong. $\log(1-D(G(z)))$ is close to 0 when $D$ correctly gives fakes a low chance.
- The discriminator **maximises** the whole value (it wants both terms high: real called real, fake called fake). The generator **minimises** it: it wants $D$ to give its fakes a high chance, which pushes $\log(1-D(G(z)))$ down.
- **Matters:** at the ideal end of the game, the fakes are indistinguishable from the real data, the best $D$ can do is say $0.5$ everywhere, and the value equals $-2\log2\approx-1.386$ (a standard result from the original GAN paper; from memory).
- In practice, people train the two networks in alternation, and often use a slightly different generator objective for steadier learning (from memory; details are in the paper).

**Example (tiny numbers):** a discriminator that is good: it gives a real example $D(x)=0.9$ and a fake $D(G(z))=0.2$.
- Value: $\ln0.9+\ln(1-0.2)=-0.105+(-0.223)=\mathbf{-0.329}$.
- A confused discriminator ($0.5$ for both): $2\ln0.5=\mathbf{-1.386}$.
- Meaning: the discriminator is better off (higher value) when it separates real from fake, and the generator's job is to drag the value down towards $-1.386$ by making fakes that look real.

**Diffusion: noising in one jump.** The forward process adds noise a little at a time, but there is a shortcut to get a noisy version $x_t$ directly from the clean $x_0$:

$$
x_t=\sqrt{\bar\alpha_t}\;x_0+\sqrt{1-\bar\alpha_t}\;\varepsilon,\qquad \varepsilon\sim\mathcal N(0,1)
$$

**Where:**
- $x_0$ is the clean example (for a picture, its pixel values), and $x_t$ is the noisy version after $t$ steps.
- $\varepsilon$ ("epsilon") is random noise from a standard bell-shaped distribution ([[Normal Distribution]]), freshly drawn.
- $\bar\alpha_t$ ("alpha-bar") is a number between 0 and 1 from the **noise schedule**: it is the product of $(1-\beta_s)$ over the steps so far, where each $\beta_s$ is the small amount of noise added at step $s$. It starts near 1 and shrinks towards 0 as $t$ grows.
- **Matters:** $\sqrt{\bar\alpha_t}$ is how much of the original **signal** is left and $\sqrt{1-\bar\alpha_t}$ is how much **noise** has been mixed in. At $t=0$ it is all signal. After many steps, almost all noise, so $x_T$ is essentially pure noise that is easy to sample.
- **Matters:** the denoising network is trained to **predict $\varepsilon$** from $(x_t,t)$, typically with a squared-error loss between its guess and the true $\varepsilon$ (a common training recipe; some models predict something else). At generation time, the predicted noise is removed step by step.
- Details such as the exact schedule, the variance choices, and the sampling rule vary between models and are not given here (from memory; see the papers).

**Example (tiny numbers):** a single pixel value $x_0=1$ and noise $\varepsilon=0.5$.
- Early on, $\bar\alpha=0.81$: $\sqrt{0.81}=0.9$ and $\sqrt{0.19}=0.436$. $x_t=0.9\cdot1+0.436\cdot0.5=\mathbf{1.118}$. Mostly signal.
- Late, $\bar\alpha=0.01$: $\sqrt{0.01}=0.1$ and $\sqrt{0.99}=0.995$. $x_t=0.1\cdot1+0.995\cdot0.5=\mathbf{0.597}$. Mostly noise: the original 1 contributes only $0.1$.
- Building up the schedule: three steps that each keep $1-\beta=0.9$ of the signal give $\bar\alpha_3=0.9^3=\mathbf{0.729}$.
- Meaning: the same formula takes a clean value to anywhere along the noise ladder in one step, which makes training fast: pick a random $t$, make $x_t$ in one jump, and ask the network to find the noise.

**Further notes:**
- **Likelihood and the VAE objective:** many generative models can be seen as maximising how likely the training data is under the model. VAEs and diffusion models use a lower bound on this likelihood (the "ELBO"); GANs skip likelihood entirely. (A common view in the literature; from memory.)
- **Latent space:** the code of an autoencoder or VAE lives in a "latent space". Moving smoothly through it can morph one output into another. Latent diffusion runs the diffusion in such a compressed space to save computing power (from memory; check the paper).
- **Why it feels like [[Self-Supervised Learning]]:** all of these make their own training signal from the data (rebuild the input, spot the fake, guess the noise, guess the next token).
- **Not covered yet in this vault:** VAEs in full, normalising flows, text-to-image guidance, and evaluation scores for images.

**What came before, and how this improved on it:**
- **Before:** fit one simple distribution to the data (one bell curve, one table of counts).
- **Its limit:** one bell curve cannot describe data in several clumps.
- **What changed:** richer families that can model many clumps and complex shapes: mixtures ([[Gaussian Mixture Model]]), then [[Autoencoder]]-based models, [[GAN]], [[Diffusion Models]] and next-word models ([[LLM]]).
- **How much better:** in the code below, two-clump data scores an average log-likelihood of $-2.570$ on new points under one bell curve and $-2.152$ under two (higher is better, about 0.42 nats per point). The same measure is not available for GANs, which is part of why comparing generative models is hard.

**Check it yourself:**

```python
import numpy as np
from sklearn.mixture import GaussianMixture
rng = np.random.default_rng(0)
def data(n): return np.where(rng.random((n, 1)) < .5, -3., 3.) + rng.standard_normal((n, 1))
tr, te = data(500), data(500)
for k in (1, 2):
    g = GaussianMixture(k, random_state=0).fit(tr)
    print(k, "bell curve(s): average log-likelihood on new data", round(g.score(te), 3))
```

## Related
- [[Unsupervised Learning]] · [[Self-Supervised Learning]] · [[PCA]] · [[Neural Network]] · [[Convolutional Neural Network]] · [[Residual Connection]] · [[Transformer]] · [[LLM]] · [[Embedding Model]] · [[Loss Function]] · [[Probability]] · [[Normal Distribution]] · [[Vector]] · [[Perplexity]] · [[Bayesian Methods]] · [[KL Divergence]] · [[Gaussian Mixture Model]] · [[Expectation-Maximization]] · [[Autoencoder]] · [[Diffusion Models]] · [[GAN]] · [[Variational Autoencoder]] · [[Naive Bayes]] · [[Autoregressive Generation]]

## Sources
- Kingma & Welling, *Auto-Encoding Variational Bayes*, 2013 (VAE; from memory)
- Goodfellow et al., *Generative Adversarial Nets*, 2014 (GAN; from memory)
- Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models*, 2020 (from memory)
- Sohl-Dickstein et al., *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*, 2015 (from memory)
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, 2022 (from memory)
- Goodfellow, Bengio & Courville, *Deep Learning*, 2016, chapters 14 and 20
- Zhang, Lipton, Li & Smola, *Dive into Deep Learning* (free online)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/generative Generative model:::A model that makes new things (pictures, sentences, music) that look like the examples it studied. To do that it first learns what the examples are like.

#flashcards/ml/formula #flashcards/topic/generative Write the autoencoder reconstruction loss and explain each symbol.
?
$$\mathcal L=\frac1n\sum_{i=1}^{n}\big\lVert x_i-g\big(f(x_i)\big)\big\rVert^{2}$$
- $x_i$: the $i$-th input (a vector). $n$: how many examples.
- $f(x_i)$: the code, the short list of numbers the encoder produces. $g(f(x_i))$: the decoder's rebuild, which should resemble $x_i$.
- $\lVert\cdot\rVert^2$: the squared length of the difference vector. $\tfrac1n\sum_i$ averages over the examples.
- The loss is 0 only when every rebuild is perfect. Because the code is narrower than the input, that is possible only if the code keeps what matters.

#flashcards/ml/example #flashcards/topic/generative Encoder $f(x)=(x_1+x_2)/2$, decoder $g(z)=(z,z)$. What is the loss for $x=(2,4)$ and $x=(3,3)$?::$x=(2,4)$: code 3, rebuild $(3,3)$, squared error $1+1=2$. $x=(3,3)$: code 3, rebuild $(3,3)$, error 0. One number describes an input perfectly when its two entries are equal and loses information when they differ.

#flashcards/ml/formula #flashcards/topic/generative Write the GAN objective and explain each symbol.
?
$$\min_G\ \max_D\ \ \mathbb E_{x\sim p_{\text{data}}}\big[\log D(x)\big]+\mathbb E_{z\sim p_z}\big[\log\big(1-D(G(z))\big)\big]$$
- $x\sim p_{\text{data}}$: $x$ is a real example from the data. $z\sim p_z$: $z$ is random noise. $\mathbb E[\cdot]$: the average over many draws.
- $D(x)$: the discriminator's chance, 0 to 1, that $x$ is real. $G(z)$: the generator's fake.
- $\log D(x)$ is large (near 0 from below) when $D$ gives real examples a high chance. $\log(1-D(G(z)))$ is near 0 when $D$ correctly gives fakes a low chance.
- The discriminator maximises the whole value (it wants to separate real from fake). The generator minimises it (it wants fakes that fool $D$).

#flashcards/ml/example #flashcards/topic/generative $D(x)=0.9$ on a real example and $D(G(z))=0.2$ on a fake. What is the value? What if $D$ says 0.5 for both?::$\ln0.9+\ln0.8=-0.105-0.223=-0.329$. Confused $D$: $2\ln0.5=-1.386$. The discriminator is better off with a higher value, and the generator tries to drag it toward $-1.386$.

#flashcards/ml/formula #flashcards/topic/generative Write the forward diffusion step and explain each symbol.
?
$$x_t=\sqrt{\bar\alpha_t}\;x_0+\sqrt{1-\bar\alpha_t}\;\varepsilon,\qquad \varepsilon\sim\mathcal N(0,1)$$
- $x_0$: the clean example (a picture's pixel values). $x_t$: the noisy version after $t$ steps.
- $\varepsilon$: random noise from a standard bell-shaped distribution, freshly drawn.
- $\bar\alpha_t$: a number between 0 and 1 from the noise schedule, the product of $(1-\beta_s)$ over the steps so far ($\beta_s$ is the small noise added at step $s$). It starts near 1 and shrinks toward 0 as $t$ grows.
- $\sqrt{\bar\alpha_t}$ is how much of the original signal remains. The same formula takes a clean value anywhere along the noise ladder in one step.

#flashcards/ml/example #flashcards/topic/generative $x_0=1$, $\varepsilon=0.5$. What is $x_t$ at $\bar\alpha=0.81$ and at $\bar\alpha=0.01$? What is $\bar\alpha_3$ if each step keeps 0.9?::$\bar\alpha=0.81$: $0.9\cdot1+0.436\cdot0.5=1.118$ (mostly signal). $\bar\alpha=0.01$: $0.1\cdot1+0.995\cdot0.5=0.597$ (mostly noise, the original contributes only 0.1). $\bar\alpha_3=0.9^3=0.729$.

#flashcards/ml/image #flashcards/topic/generative ![[autoencoder-schema.png]]
What does this picture show?
?
An autoencoder. The input layer ($X$) is squeezed by the encoder into a narrow middle code ($h$), and the decoder expands it back into an output ($X'$) that should match the input. The narrow middle forces the network to keep only what matters. (From the note's caption: Michela Massi, Wikimedia Commons.)

#flashcards/ml/image #flashcards/topic/generative ![[gan-diagram.svg]]
What does this picture show?
?
A GAN. Random noise $z$ goes into the generator, which produces a fake $G(z)$. The discriminator receives either a fake $G(z)$ or a real example $x$ and must say "real or fake". (From the note's caption: Zhang, Lipton, Li and Smola, Wikimedia Commons.)

#flashcards/ml/image #flashcards/topic/generative ![[diffusion-forward.png]]
What does this picture show?
?
The forward process: a photo of a rhinoceros ($T=0$) gets more and more noise added at each step until, at the end, it is just noise. (From the note's caption: MrAlanKoh, Wikimedia Commons.)

#flashcards/ml/image #flashcards/topic/generative ![[diffusion-denoising.png]]
What does this picture show?
?
The reverse process: a denoising model takes a noisy image ($Z_{T=n}$) and produces a slightly cleaner one ($Z_{T=n-1}$), repeating until a picture emerges. Different starting noise gives variations of the picture. It is a cartoon of the idea, not the output of a real model. (From the note's caption: MrAlanKoh, Wikimedia Commons.)

#flashcards/ml/cloze #flashcards/topic/generative In a GAN the discriminator ==maximises== the value and the generator ==minimises== it.
