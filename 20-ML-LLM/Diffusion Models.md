---
tags: [machine-learning, deep-learning, generative-models, diffusion, LLM, probability]
status: growing
level: 4
tier: 4
---

# Diffusion Models

A diffusion model makes new data (such as pictures) by **starting from pure random noise and cleaning it up a little at a time**, using a network that has learned to spot noise.

## 🟢 1. Simple (age 12)
Imagine a sandcastle on a beach. Waves wash over it again and again until it is a flat, smooth beach. Now imagine someone who has watched this happen thousands of times, and can **undo one wave at a time**. Give them a smooth beach and they can rebuild a castle, step by step. A diffusion model does the same with pictures: it learns how a picture turns into static, then runs the movie backwards.

**Example:** Take a photo of a cat and sprinkle a little TV static on it. Do this again and again until you only see static. The model practises one small job: "look at this slightly staticky picture and tell me which part is the static". After lots of practice, you can give it fresh static and ask it to remove the static bit by bit. A new picture appears.

## 🔵 2. How it works
1. **Noising (no learning needed):** take a real example and mix in a little random noise. Repeat many times (say 1,000) until the example is nothing but noise.
2. **Make practice problems:** pick a real example, pick a random step number, and add the amount of noise that belongs to that step. You now know exactly which noise you added.
3. **Train a [[Neural Network]]:** show it the noisy example and the step number, and ask it to **guess the noise**. Score the guess with a [[Loss Function]] (how far the guess is from the real noise) and improve it with [[Gradient Descent]] and [[Backpropagation]].
4. **Generate:** start from pure random noise. Ask the network for its noise guess, remove a small part of it, and add a tiny fresh bit of randomness. Repeat for every step, from the noisiest to the cleanest.
5. **Steer it (optional):** feed a text prompt, turned into numbers by an [[Embedding Model]], to the network too. Then the cleaning-up goes toward pictures that match the words.

![[diffusion-forward.png]]
*The noising direction: a photo of a rhinoceros (T=0) gets more noise at each step until only noise is left. A diffusion model learns to run this backwards. (The same figure is used in [[Generative Models]].) Image: MrAlanKoh, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Example_of_Forward_Diffusion_Models.png), CC BY-SA 4.0 (derived from a photograph of a black rhinoceros, credited on the file page).*

## 🟠 3. Technical detail
- **Two processes:** the *forward* process adds noise on a fixed schedule and has no learned parts. The *reverse* process removes it using a learned network.
- **Noise schedule:** a list of small numbers $\beta_1,\dots,\beta_T$, the noise added at each step. Early steps add little, later steps add more. The product of the "kept" fractions gives $\bar\alpha_t$, the share of the original signal left after $t$ steps.
- **One-jump noising:** you never need to add the noise step by step during training. There is a formula to jump straight to step $t$. That makes training cheap: each training example is a random clean example, a random $t$ and a random noise draw.
- **What the network predicts:** the usual choice is the noise $\varepsilon$ that was added. Some models predict the clean example or a mixture of both instead (from memory; the choice differs between models).
- **Network design:** for pictures it is often a [[Convolutional Neural Network]] with [[Attention]] layers (a "U-Net"), and newer systems use a [[Transformer]] (from memory; designs vary). The step number is given to the network as an extra input.
- **Training is simple:** it is plain regression with a squared-error loss, so it is stable. This is a main reason diffusion overtook [[Generative Models]] such as GANs for many image tasks (the GAN comparison is from memory).
- **Sampling is slow:** one network call per step, so hundreds or thousands of calls. Faster samplers use fewer, bigger steps. [[Knowledge Distillation]] can train a student that does the job in a handful of steps (from memory).
- **Latent diffusion:** instead of denoising full-size pictures, work on the small code of an [[Autoencoder]]. This is far cheaper. Many popular image generators work this way (from memory; check the specific model).
- **Guidance:** a text [[Embedding]] steers the cleaning-up. A common trick runs the network with and without the text and pushes further in the direction that the text changes (from memory, not covered here).
- **Not only pictures:** the same idea is used for audio, video and molecules, and experiments exist for text (from memory).

The code has two tests. First, a case where the best noise guess has a closed form, so no training is needed: the sampler should give back the right data. Second, a small network learning two clumps of numbers from scratch.

```python
import numpy as np, torch, torch.nn as nn
T = 100
beta = np.linspace(1e-4, 0.1, T); alpha = 1 - beta; ab = np.cumprod(alpha)   # noise schedule
print('signal left at the end:', round(float(np.sqrt(ab[-1])), 4))             # small: nearly pure noise

def sample(eps_fn, n, rng):
    x = rng.normal(size=n)                                  # start from pure noise
    for t in range(T - 1, -1, -1):
        e = eps_fn(x, t)                                    # the guess of the noise in x
        x = (x - beta[t] / np.sqrt(1 - ab[t]) * e) / np.sqrt(alpha[t])
        if t > 0: x += np.sqrt(beta[t]) * rng.normal(size=n)  # fresh noise, none at the last step
    return x

# 1) exact case: data ~ N(3, 0.5^2). The best noise guess has a closed form, no training needed.
mu0, s0 = 3.0, 0.5
def exact(x, t):
    var = ab[t] * s0**2 + 1 - ab[t]
    return np.sqrt(1 - ab[t]) * (x - np.sqrt(ab[t]) * mu0) / var
x = sample(exact, 20000, np.random.default_rng(0))
print('exact eps  mean/std:', round(float(x.mean()), 3), round(float(x.std()), 3))   # about 3.0 / 0.5
print('wrong eps=0 mean/std:', round(float(sample(lambda x, t: 0 * x, 20000, np.random.default_rng(0)).mean()), 3))  # not 3.0

# 2) learned case: two clumps at -2 and +2 (std 0.3); a small net learns to guess the noise
torch.manual_seed(0); rng = np.random.default_rng(1)
net = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 1))
opt = torch.optim.Adam(net.parameters(), 2e-3)
abt = torch.tensor(ab, dtype=torch.float32)
for step in range(4000):
    x0 = torch.tensor(rng.choice([-2.0, 2.0], 512) + 0.3 * rng.normal(size=512), dtype=torch.float32)
    t = torch.randint(0, T, (512,)); eps = torch.randn(512)
    xt = abt[t].sqrt() * x0 + (1 - abt[t]).sqrt() * eps          # noise in one jump
    loss = ((net(torch.stack([xt, t / T], 1)).squeeze(1) - eps) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print('final loss', round(loss.item(), 3))
def learned(x, t):
    with torch.no_grad():
        return net(torch.tensor(np.stack([x, np.full_like(x, t / T)], 1), dtype=torch.float32)).squeeze(1).numpy()
y = sample(learned, 5000, np.random.default_rng(2))
print('share near -2 / +2:', round(float((abs(y + 2) < 1).mean()), 3), round(float((abs(y - 2) < 1).mean()), 3))
print('mean |y| and std of each clump:', round(float(abs(y).mean()), 2), round(float(y[y < 0].std()), 2), round(float(y[y > 0].std()), 2))
# signal left at the end: 0.075
# exact eps  mean/std: 3.001 0.505
# wrong eps=0 mean/std: 0.108
# final loss 0.316          (last training batch, so it differs a little between runs and versions)
# share near -2 / +2: 0.49 0.509
# mean |y| and std of each clump: 1.99 0.28 0.29
```
Reading the output: with the exact noise guess, 20,000 samples come out with mean 3.001 and spread 0.505 (the data has mean 3 and spread 0.5). A useless guess of zero noise never finds the data (mean 0.108). The small network, trained from scratch, sends about half of the samples to each clump, at about ±2 with spread about 0.29. It is a toy in one dimension, so it shows the idea works, not that it works for pictures. The computations ran in PyTorch 2.8.0 on a CPU.

## 🔴 4. Deep / formal

**Noising in one jump.** Write $x_0$ for a clean example. After $t$ steps of the schedule, the noisy version is

$$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\qquad \bar\alpha_t=\prod_{s=1}^{t}(1-\beta_s),\quad \varepsilon\sim\mathcal N(0,1)$$

**Where:**
- $x_0$ is the clean example, $x_t$ the noisy one at step $t$.
- $\beta_s$ is the small noise amount added at step $s$. **Matters:** the whole list of $\beta$ values is the schedule. If the values are too small, the end is not pure noise, and sampling from pure noise does not match the training. In the code, the first schedule I tried ended with 28% signal left and the sampled mean came out as 2.94, not 3.
- $\bar\alpha_t$ is the share of signal kept. **Matters:** it starts near 1 and falls toward 0. It decides the mix of signal and noise at every step.
- $\varepsilon$ is fresh random noise from the bell-shaped [[Normal Distribution]]. **Matters:** it is the training target, so the network must be able to work it out.

**Example (tiny numbers):** $x_0=2$, $\varepsilon=-1$, $\bar\alpha_t=0.64$.
1. $\sqrt{0.64}=0.8$ and $\sqrt{1-0.64}=0.6$.
2. $x_t=0.8\times 2+0.6\times(-1)=1.6-0.6=1.0$.
3. Going back with the true noise: $(1.0-0.6\times(-1))/0.8=2.0$, the original.
4. Meaning: the noisy value 1.0 mixes 80% signal with 60% noise. If the network can find the noise ($-1$), it can get the clean value back. That is the whole training goal.

For the schedule in the code, $\bar\alpha_T$ is small enough that only 7.5% of the signal is left at the end, so the end is close to pure noise (the schedule is a toy; real ones differ).

**The training loss.** The network $\varepsilon_\theta$ gets the noisy example and the step number and is trained to find the noise:

$$\mathcal{L}(\theta)=\mathbb{E}_{x_0,\,t,\,\varepsilon}\left[\,\bigl\|\varepsilon-\varepsilon_\theta(x_t,t)\bigr\|^2\right]$$

**Where:**
- $\theta$ is the network's weights. **Matters:** these are what training changes.
- $\varepsilon_\theta(x_t,t)$ is the network's guess of the noise. **Matters:** it needs $t$, because a noisy example with a little noise and one with a lot of noise need different cleaning.
- $\varepsilon$ is the real noise that was added. **Matters:** it is known for free, so no labels are needed. That is why it counts as [[Self-Supervised Learning]].
- $\mathbb{E}$ is the average over random examples, random steps and random noise. **Matters:** it makes the network good at every noise level, not just one.

**Example (tiny numbers):** two pixels with real noise $(0.5,-1.0)$.
1. A good guess $(0.3,-0.6)$: differences $(0.2,-0.4)$, squares $(0.04, 0.16)$, average $0.10$.
2. A lazy guess "no noise" $(0,0)$: squares $(0.25, 1.00)$, average $0.625$.
3. Meaning: the closer the guess, the lower the loss. Training pushes the network from 0.625-style guesses toward 0.10-style guesses. The loss never reaches exactly zero, because at high noise the noise cannot be told apart from the signal (in the code the loss stops near 0.3).

**The reverse step.** To go from $x_t$ to a slightly cleaner $x_{t-1}$, one common sampler uses

$$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t,t)\right)+\sigma_t z,\qquad \alpha_t=1-\beta_t,\ z\sim\mathcal N(0,1)$$

**Where:**
- $\alpha_t=1-\beta_t$ is the fraction kept in step $t$ alone. **Matters:** dividing by $\sqrt{\alpha_t}$ undoes the shrinking that the forward step did.
- $\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta$ is the part of the noise guess that is removed in this step. **Matters:** removing all of it at once would be too bold, because the guess is rough when there is a lot of noise. A small step each time lets later steps correct earlier mistakes.
- $\sigma_t z$ is a small fresh random kick, with $\sigma_t^2=\beta_t$ in the code. **Matters:** it keeps samples varied. Without it the sampler gives one fixed output for each starting noise and can lose variety (the choice of $\sigma_t$ is a modelling choice; this one is from memory).
- $z$ is a fresh draw. **Matters:** no kick is added at the last step, so the result is not re-noised.

**Example (tiny numbers):** $\beta_t=0.1$, so $\alpha_t=0.9$, with $\bar\alpha_t=0.64$, a noisy value $x_t=0.4$, and a noise guess $\varepsilon_\theta=-0.5$.
1. $\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}=0.1/0.6=0.1667$.
2. Remove it: $0.4-0.1667\times(-0.5)=0.4+0.0833=0.4833$.
3. Divide by $\sqrt{0.9}=0.9487$: $0.4833/0.9487=0.5095$.
4. Add a kick of size $\sigma_t=\sqrt{0.1}=0.316$ times a fresh draw $z$ (skipped here).
5. Meaning: the model believed the noise pushed the value down, so it moved the value up from 0.40 to about 0.51 before the random kick.

**Connections (all from memory, not derived here):**
- The loss comes from a bound on the likelihood of the data, in the same spirit as the VAE objective in [[Autoencoder]] and [[Maximum Likelihood]], with [[KL Divergence]] terms between Gaussians. The simple "guess the noise" loss is a rewriting of it that works well in practice.
- The noise guess is closely related to the "score", the direction in which the data becomes more likely. This links diffusion to score matching and to stochastic differential equations.
- A "wrong eps = 0" sampler in the code is the same as treating all data as noise, and it ends near the middle of the noise, not at the data.

**Open points:**
- Sampling cost is the main limit. Fewer-step samplers and [[Knowledge Distillation]] both trade some quality for speed.
- The exact $\sigma_t$, schedule and loss weighting matter in practice, and different papers choose them differently. I have not compared them.
- Memorising training examples and copying style are known concerns for large models, and they are a data and policy topic beyond this note.

**What came before, and how this improved on it:**
- **Before:** [[GAN]]s and [[Variational Autoencoder]]s, which make a sample in a single pass of the network.
- **Its limit:** sample quality, which is what the DDPM abstract compares (image quality scores).
- **What changed:** learn to undo small steps of added noise, and make an image by removing noise step by step (Ho, Jain and Abbeel, arXiv 2006.11239, 19 June 2020, abstract read; earlier work by Sohl-Dickstein and colleagues, 2015, from memory).
- **How much better:** the abstract reports an Inception score of 9.46 and a FID of 3.17 on unconditional CIFAR10, and on 256×256 LSUN "sample quality similar to ProgressiveGAN". The price, counted in the code below: 1000 network evaluations for one sample against 1 for a GAN or VAE. With the linear noise schedule I used, 1000 steps leave only 0.64% of the original picture, so the start of generation is nearly pure noise.

**Check it yourself:**

```python
import numpy as np
T = 1000; beta = np.linspace(1e-4, 0.02, T); abar = np.cumprod(1 - beta)
for t in (1, 100, 500, 1000):
    print(f"t={t:4d}  share of the picture kept {np.sqrt(abar[t-1]):.4f}  share of noise {np.sqrt(1-abar[t-1]):.4f}")
print("network evaluations to make one sample: GAN or VAE 1, diffusion", T)
```

## Related
[[Generative Models]] · [[Autoencoder]] · [[Normal Distribution]] · [[KL Divergence]] · [[Maximum Likelihood]] · [[Self-Supervised Learning]] · [[Knowledge Distillation]] · [[Neural Network]] · [[Convolutional Neural Network]] · [[Transformer]] · [[Attention]] · [[Embedding Model]] · [[Loss Function]] · [[Gradient Descent]] · [[Backpropagation]] · [[Probability]] · [[GAN]] · [[Variational Autoencoder]] · [[Masked Diffusion Language Model]] · [[Vision Transformer]]

## Sources
- The idea and the sampler follow Ho, Jain and Abbeel (2020), "Denoising Diffusion Probabilistic Models", with earlier work by Sohl-Dickstein and colleagues (2015) (both cited from memory; not opened for this note).
- Latent diffusion: Rombach and colleagues (2022) (from memory).
- Image: [Wikimedia Commons, Example_of_Forward_Diffusion_Models.png](https://commons.wikimedia.org/wiki/File:Example_of_Forward_Diffusion_Models.png), CC BY-SA 4.0.
- The code, the closed-form noise guess and the Example numbers were run for this note (PyTorch 2.8.0, NumPy). The closed form was checked only through the sampled mean and spread, not derived in the note.
- Ho et al., arXiv 2006.11239 (abstract read for the history block)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/generative Diffusion model:::Waves wash a sandcastle flat again and again. A model that has watched this many times learns to undo one wave at a time, so from a smooth beach (pure noise) it can rebuild a castle (a picture) step by step.

#flashcards/ml/formula #flashcards/topic/generative Write the forward noising step and explain each symbol.
?
$$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\qquad \bar\alpha_t=\prod_{s=1}^{t}(1-\beta_s),\quad \varepsilon\sim\mathcal N(0,1)$$
- $x_0$: the clean example. $x_t$: the noisy one at step $t$.
- $\beta_s$: the small noise amount added at step $s$. The whole list of $\beta$ values is the schedule. If the values are too small, the end is not pure noise and sampling from pure noise does not match training.
- $\bar\alpha_t$: how much of the original signal is left after $t$ steps.
- $\varepsilon$: the noise draw. The result mixes signal and noise.

#flashcards/ml/formula #flashcards/topic/generative Write the diffusion training loss and explain each symbol.
?
$$\mathcal{L}(\theta)=\mathbb{E}_{x_0,\,t,\,\varepsilon}\left[\,\bigl\|\varepsilon-\varepsilon_\theta(x_t,t)\bigr\|^2\right]$$
- $\theta$: the network's weights, which training changes.
- $\varepsilon_\theta(x_t,t)$: the network's guess of the noise. It needs $t$, because an example with a little noise and one with a lot need different cleaning.
- $\varepsilon$: the real noise that was added. It is known, because we added it ourselves.
- The loss is the squared gap between the guess and the real noise.

#flashcards/ml/formula #flashcards/topic/generative Write one reverse (denoising) step and explain each symbol.
?
$$x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\varepsilon_\theta(x_t,t)\right)+\sigma_t z,\qquad \alpha_t=1-\beta_t,\ z\sim\mathcal N(0,1)$$
- $\alpha_t=1-\beta_t$: the fraction kept in step $t$ alone. Dividing by $\sqrt{\alpha_t}$ undoes the shrinking the forward step did.
- $\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta$: the part of the noise guess removed in this step. Removing all of it at once would be too bold.
- $\sigma_t z$: a small fresh noise kick, so sampling stays varied.

#flashcards/ml/cloze #flashcards/topic/generative The model is trained to ==predict the noise== that was added, and sampling removes it ==step by step==.

#flashcards/ml/example #flashcards/topic/generative $x_0=2$, $\varepsilon=-1$, $\bar\alpha_t=0.64$. What is $x_t$, and can you recover $x_0$?::√0.64 = 0.8, √(1 − 0.64) = 0.6. x_t = 0.8 × 2 + 0.6 × (−1) = 1.0. Back with the true noise: (1.0 + 0.6)/0.8 = 2.0, the original.

#flashcards/ml/example #flashcards/topic/generative Real noise $(0.5,-1.0)$. What is the loss for a good guess $(0.3,-0.6)$ and for "no noise" $(0,0)$?::Good guess: differences (0.2, −0.4), squares (0.04, 0.16), average 0.10. No noise: squares (0.25, 1.00), average 0.625. Closer guesses give lower loss.

#flashcards/ml/example #flashcards/topic/generative $\beta_t=0.1$, $\bar\alpha_t=0.64$, $x_t=0.4$, noise guess −0.5. What is the reverse step before the random kick?::β/√(1 − ᾱ) = 0.1/0.6 = 0.1667. 0.4 − 0.1667 × (−0.5) = 0.4833. Divide by √0.9 = 0.9487: 0.5095. A small random kick is then added.

#flashcards/ml/image #flashcards/topic/generative ![[diffusion-forward.png]]
What does this picture show?
?
The noising direction: a photo of a rhinoceros (T=0) gets more noise at each step until only noise is left. A diffusion model learns to run this backwards.
