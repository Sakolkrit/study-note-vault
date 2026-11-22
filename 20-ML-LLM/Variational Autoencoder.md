---
tags: [machine-learning, deep-learning, generative-models, autoencoder, VAE, probability]
status: growing
level: 4
tier: 3
---

# Variational Autoencoder

A variational autoencoder (VAE) is an [[Autoencoder]] whose code is a **small cloud of possibilities instead of one fixed point**. A penalty keeps all the clouds packed neatly around the same centre, so you can pick a random spot and **decode it into a brand-new example**.

## 🟢 1. Simple (age 12)
Imagine a map of a town. A plain autoencoder puts each house on the map as one exact pin. The pins can end up scattered with big empty gaps. If you point at a random spot in a gap, there is no house there, only nonsense. A VAE puts each house on the map as a **fuzzy blob** and also tells the blobs to stay close to the middle of the map and overlap a bit. Now the map has no big gaps, so almost any spot you point at is near some house. If you build what is at that spot, you get something sensible and new.

**Example:** You teach a friend to draw faces. If they only memorise exact faces, then asking for "something halfway between face A and face B" gives a mess. If they have learned that faces blur gently into each other, then "halfway" is a real, believable face.

## 🔵 2. How it works
1. **The encoder** (a [[Neural Network]]) looks at an input and does not give one code. It gives two lists of numbers: a **mean** $\mu$ (the centre of the blob) and a **spread** $\sigma$ (how wide the blob is).
2. **Pick a point in the blob:** draw random noise $\varepsilon$ and compute $z=\mu+\sigma\varepsilon$. This random pick is the code for this input.
3. **The decoder** (another network) turns $z$ back into something that should look like the input.
4. **Two scores are added together:**
   - **Rebuild error:** how far the decoder's output is from the input (a [[Loss Function]]).
   - **Neat-map penalty:** how far the blob is from a plain bell curve centred at 0 with spread 1 (a [[KL Divergence]], from the [[Normal Distribution]]). This keeps all blobs near the centre and overlapping.
5. **Learn** with [[Backpropagation]] and [[Gradient Descent]]. The random pick is written as $\mu+\sigma\varepsilon$ so the learning signal can pass through it (the *reparameterisation trick*).
6. **To make something new:** forget the encoder. Draw $z$ from the plain bell curve, and decode it.

![[vae-reparameterized.png]]
*A VAE. The encoder outputs a mean μ and a spread σ. A noise draw ε is combined with them into the code z, and the decoder turns z into the output x'. Image: EugenioTL, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Reparameterized_Variational_Autoencoder.png), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Two goals pulling against each other:** the rebuild term wants each input to keep its own, sharp code. The penalty wants every code cloud to look like the same bell curve. The trade-off is why VAEs rebuild a bit worse than plain autoencoders but sample much better (shown in the code).
- **The prior:** the plain bell curve $\mathcal N(0,I)$ is the agreed place to sample from. If the penalty works, decoding draws from it gives plausible examples.
- **Why not a plain autoencoder?** Its codes can sit anywhere, so random codes are often in empty regions and decode to garbage (see the code).
- **Blurry outputs:** with a squared-error rebuild term, a VAE tends to produce blurry images, because it averages over the possible details (a widely reported effect; from memory).
- **The weight on the penalty ($\beta$):** the code multiplies the penalty by a small number. A bigger weight gives a smoother map but blurrier rebuilds. At the extreme, the decoder ignores the code (*posterior collapse*; from memory, not shown here).
- **Why random sampling is a problem for learning:** you cannot send a gradient through "pick a random number". The trick writes the pick as $\mu+\sigma\varepsilon$, with the randomness moved into $\varepsilon$, which has no weights. The gradient then flows through $\mu$ and $\sigma$ as usual.
- **Uses:** generating new examples, smooth blending between two examples, anomaly detection (items that rebuild badly), and as the compressor in latent [[Diffusion Models]] (many popular image generators work in a VAE-style code space; from memory, check the model you care about).
- **Compared with [[GAN]]:** a VAE trains steadily and has a score you can track, but its samples are often blurrier. A GAN is sharper but harder to train.
- **Related to PCA:** a very simple VAE behaves like a probabilistic version of [[PCA]] (from memory; not shown here).

The code trains a plain autoencoder and a VAE on 2,000 points around a circle, with a 2-number code for each. Then it draws random codes from a bell curve, decodes them, and measures how far the results are from the circle.

```python
import numpy as np, torch, torch.nn as nn
torch.manual_seed(0); rng = np.random.default_rng(0)
a = rng.uniform(0, 2 * np.pi, 2000)
data = torch.tensor(np.c_[np.cos(a), np.sin(a)] + 0.03 * rng.normal(size=(2000, 2)), dtype=torch.float32)   # noisy circle
def mlp(i, o): return nn.Sequential(nn.Linear(i, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, o))
def run(vae, steps=3000, beta=0.05):
    enc, dec = mlp(2, 4 if vae else 2), mlp(2, 2)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), 3e-3)
    for _ in range(steps):
        x = data[torch.randint(0, 2000, (256,))]; h = enc(x)
        if vae:
            mu, logvar = h[:, :2], h[:, 2:]
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)            # reparameterisation trick
            kl = 0.5 * (mu ** 2 + logvar.exp() - 1 - logvar).sum(1).mean()
        else: z, kl = h, 0.0
        loss = ((dec(z) - x) ** 2).sum(1).mean() + beta * kl
        opt.zero_grad(); loss.backward(); opt.step()
    return enc, dec
def off_circle(dec, n=5000):                                                  # sample codes from N(0,1), decode, how far from the circle?
    with torch.no_grad(): y = dec(torch.randn(n, 2))
    return round((y.norm(dim=1) - 1).abs().mean().item(), 3), round(y.std(0).mean().item(), 2)
for name, vae in (('plain AE', False), ('VAE', True)):
    enc, dec = run(vae)
    with torch.no_grad(): rec = ((dec(enc(data)[:, :2]) - data) ** 2).sum(1).mean().item()
    print(name, 'rebuild error', round(rec, 4), '| random-code samples: off-circle, spread', *off_circle(dec))
# plain AE rebuild error 0.0002 | random-code samples: off-circle, spread 0.406 0.89
# VAE rebuild error 0.0023 | random-code samples: off-circle, spread 0.058 0.67
```
Reading the output (PyTorch 2.8.0, CPU, one run with one seed, so treat it as an illustration and not a measurement):
- The plain autoencoder rebuilds its training points better (error 0.0002 against 0.0023 for the VAE).
- But when you decode random codes, the plain autoencoder's results land on average 0.406 away from the circle (the circle has radius 1), while the VAE's land 0.058 away. The VAE's map has fewer gaps, which is the point.
- The "spread" column is the typical size of the generated points. A perfect circle has spread about 0.7, so the VAE's 0.67 is close, and the plain autoencoder's 0.89 shows points flung too far out.
- The plain autoencoder's codes are not tied to a bell curve, so drawing codes from one is an arbitrary choice for it. That is part of the lesson: only the VAE was *trained* for it.

## 🔴 4. Deep / formal

**The objective (ELBO).** We want a decoder that makes the data likely. The usual objective, which is a lower bound on the log-likelihood of the data, is

$$\mathcal{L}(x)=\mathbb{E}_{z\sim q(z\mid x)}\bigl[\log p(x\mid z)\bigr]-\mathrm{KL}\bigl(q(z\mid x)\,\|\,p(z)\bigr)$$

**Where:**
- $q(z\mid x)$ is the encoder's cloud of codes for the input $x$, here a bell curve with mean $\mu$ and spread $\sigma$. **Matters:** it says where to look for codes for this input.
- $p(x\mid z)$ is the decoder: how likely $x$ is, given the code $z$. **Matters:** it is the rebuild score. With a bell-curve decoder of fixed spread, it equals a squared error plus a constant.
- $p(z)=\mathcal N(0,I)$ is the prior, the plain bell curve we want to sample from. **Matters:** the KL term pulls every cloud toward it. Without this term the clouds could shrink to points and scatter, and you would have a plain autoencoder.
- $\mathrm{KL}$ is the [[KL Divergence]]. **Matters:** it measures how far the cloud is from the prior. It is zero only if they are identical.
- $\mathcal{L}(x)$ is the score to push up. **Matters:** it is *at most* the log-likelihood of $x$ (hence "lower bound"), so raising it raises a floor under how likely the data is, in the spirit of [[Maximum Likelihood]].

**Example (tiny numbers):** One number $x=1$ is rebuilt as $\hat x=0.8$ by a decoder that is a bell curve with spread 1 around its output. The encoder gave $\mu=1$, $\sigma=0.5$.
1. Rebuild term (log of the bell curve at $x$): $-\tfrac12(1-0.8)^2-\tfrac12\ln(2\pi)=-0.02-0.9189=-0.9389$.
2. Penalty: $\mathrm{KL}=\tfrac12(\mu^2+\sigma^2-1-\ln\sigma^2)=\tfrac12(1+0.25-1+1.3863)=0.8181$.
3. $\mathcal{L}=-0.9389-0.8181=-1.7571$.
4. Meaning: a higher (less negative) score is better. A sharper rebuild would lift the first term, and a cloud closer to the bell curve would lower the penalty. The training pushes both.

**The penalty in closed form.** For a cloud $\mathcal N(\mu_j,\sigma_j^2)$ in each of $d$ code numbers, against the prior $\mathcal N(0,1)$,

$$\mathrm{KL}=\frac12\sum_{j=1}^{d}\left(\mu_j^2+\sigma_j^2-1-\ln\sigma_j^2\right)$$

**Where:**
- $\mu_j$ is the cloud's centre in code number $j$. **Matters:** the $\mu_j^2$ term punishes a centre far from 0.
- $\sigma_j$ is the cloud's spread. **Matters:** $\sigma_j^2-1-\ln\sigma_j^2$ is zero at $\sigma_j=1$ and grows if the cloud is too narrow or too wide. A cloud that shrinks to a point ($\sigma_j\to0$) has a huge penalty, which is what stops it from collapsing back into a plain autoencoder.
- $d$ is the code size. **Matters:** each code number adds its own penalty, so a bigger code costs more.
- The $\tfrac12$ and the $-1$ come from the algebra of two bell curves (a standard result, from memory). I checked this formula against a random-sampling estimate (0.8193 against 0.8181), but I did not derive it in the note.

**Example (tiny numbers):** Two code numbers: $(\mu_1,\sigma_1)=(1,0.5)$ and $(\mu_2,\sigma_2)=(0,1)$.
1. Number 1: $\tfrac12(1+0.25-1-\ln0.25)=\tfrac12(0.25+1.3863)=0.8181$.
2. Number 2: $\tfrac12(0+1-1-0)=0$, because it is already exactly the prior.
3. Total $\mathrm{KL}=0.8181+0=0.8181$.
4. Meaning: only code number 1 is being charged. It is off-centre and too narrow. Number 2 costs nothing.

**The reparameterisation trick.** Instead of drawing $z$ directly from $\mathcal N(\mu,\sigma^2)$, draw fixed-shape noise and compute

$$z=\mu+\sigma\,\varepsilon,\qquad \varepsilon\sim\mathcal N(0,1),\qquad \frac{\partial z}{\partial\mu}=1,\quad \frac{\partial z}{\partial\sigma}=\varepsilon$$

**Where:**
- $\mu,\sigma$ are the encoder's outputs. **Matters:** they carry the learned weights. With the trick, the loss can be differentiated with respect to them.
- $\varepsilon$ is the random draw. **Matters:** it has no weights and does not depend on the encoder, so the random part is outside the path that learning has to go through. Without the trick, "draw a random number" is a step that gradients cannot pass.
- $\partial z/\partial\mu=1$ and $\partial z/\partial\sigma=\varepsilon$ are the slopes. **Matters:** nudging the centre moves $z$ one for one, and nudging the spread moves $z$ by the noise draw. These are what [[Backpropagation]] multiplies by.

**Example (tiny numbers):** $\mu=1$, $\sigma=0.5$, and the draw $\varepsilon=0.8$. Say the rest of the network gives the loss $t=z^2$ (a stand-in).
1. $z=1+0.5\times0.8=1.4$.
2. $\mathrm{d}t/\mathrm{d}z=2z=2.8$.
3. Gradient to the centre: $2.8\times1=2.8$. Gradient to the spread: $2.8\times0.8=2.24$.
4. Check by averaging over many draws: the mean gradients are about $2.0$ and $1.0$ (a million draws gave 2.001 and 1.003), which equal $2\mu$ and $2\sigma$. These are the true slopes of the *average* loss $\mu^2+\sigma^2$.
5. Meaning: a single draw gives a noisy but unbiased push, and on average the pushes point the right way. That is enough for learning.

![[vae-reparameterization-trick.png]]
*The trick as a graph. Left, the original form: the code z is a random node that depends on the encoder's numbers (Φ) and the input x, so the gradient is blocked at z. Right, the reparameterised form: the randomness is pulled out into ε, and z is a plain, differentiable function of Φ, x and ε. The red arrows show the gradient flowing back through z to Φ. Image: EugenioTL, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Reparameterization_Trick.png), CC BY-SA 4.0.*

**Open points and cautions (from memory, check before relying on them):**
- The "ELBO is a lower bound" statement and its derivation (with the encoder as an approximation of the true posterior) are standard, but I have only checked the pieces numerically, not proved them here.
- A fixed-variance decoder turns the first term into squared error. A more flexible decoder is possible but can make the code ignored (posterior collapse).
- Many practical VAEs use a different weight on the penalty than the ELBO gives (a "$\beta$-VAE"; the code uses 0.05, a value I picked by hand and did not tune).

**What came before, and how this improved on it:**
- **Before:** the plain [[Autoencoder]], which rebuilds inputs well but has no rule for which codes make sense, so a random code need not decode to anything realistic.
- **Its limit:** it is not a generator. Sampling a code and decoding it can land off the data.
- **What changed:** add a penalty that pulls the codes towards a bell curve, and use the reparameterisation trick so gradients flow through the random step (Kingma and Welling, arXiv 1312.6114, 20 December 2013, abstract read).
- **How much better:** in the code below, both models learn the curve $y=x^2$ with a 1-number code. Decoding random codes from a standard bell curve, the points are on average 0.033 off the curve for the plain autoencoder and 0.019 for the VAE. That is better but modest. It is one run, and the penalty weight 0.05 was picked by hand.

**Check it yourself:**

```python
import numpy as np, torch
torch.manual_seed(0); rng = np.random.default_rng(0)
x = rng.uniform(-1, 1, 400); Xt = torch.tensor(np.stack([x, x ** 2], 1), dtype=torch.float32)
def make(): return (torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2)),
                    torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2)))
def train(vae):
    enc, dec = make(); ps = list(enc.parameters()) + list(dec.parameters()); opt = torch.optim.Adam(ps, 1e-2)
    for _ in range(3000):
        opt.zero_grad(); m, lv = enc(Xt).chunk(2, 1)
        if vae: z = m + torch.exp(lv / 2) * torch.randn_like(m)
        else: z = m
        rec = ((dec(z) - Xt) ** 2).sum(1).mean()
        kl = (-0.5 * (1 + lv - m ** 2 - lv.exp())).sum(1).mean() if vae else 0
        (rec + 0.05 * kl).backward(); opt.step()
    return dec
def off_curve(dec):                                     # distance of random-code samples from y = x^2
    p = dec(torch.randn(5000, 1)).detach(); return float((p[:, 1] - p[:, 0] ** 2).abs().mean())
print("average distance from the curve, decoding z ~ N(0,1):  plain autoencoder", round(off_curve(train(False)), 3), "  VAE", round(off_curve(train(True)), 3))
```

## Related
[[Autoencoder]] · [[Generative Models]] · [[GAN]] · [[Diffusion Models]] · [[KL Divergence]] · [[Normal Distribution]] · [[Maximum Likelihood]] · [[Entropy]] · [[Probability]] · [[Neural Network]] · [[Backpropagation]] · [[Gradient Descent]] · [[Loss Function]] · [[PCA]] · [[Bayesian Methods]] · [[Bayes' Theorem]]

## Sources
- Kingma and Welling (2013), "Auto-Encoding Variational Bayes" (the VAE and the reparameterisation trick; cited from memory, not opened for this note).
- Figures: [Reparameterized_Variational_Autoencoder.png](https://commons.wikimedia.org/wiki/File:Reparameterized_Variational_Autoencoder.png) and [Reparameterization_Trick.png](https://commons.wikimedia.org/wiki/File:Reparameterization_Trick.png), both by EugenioTL on Wikimedia Commons, CC BY-SA 4.0.
- The code, the closed-form KL check, the gradient averages and the Example numbers were run for this note (PyTorch 2.8.0, NumPy).
- Kingma and Welling, arXiv 1312.6114 (abstract read for the history block)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/generative Variational autoencoder (VAE):::An autoencoder that puts each input on the map as a fuzzy blob, and tells the blobs to stay close to a standard bell curve. Then a random point on the map decodes to something sensible, so it can generate new things.

#flashcards/ml/formula #flashcards/topic/generative Write the VAE objective and explain each symbol.
?
$$\mathcal{L}(x)=\mathbb{E}_{z\sim q(z\mid x)}\bigl[\log p(x\mid z)\bigr]-\mathrm{KL}\bigl(q(z\mid x)\,\|\,p(z)\bigr)$$
- $q(z\mid x)$: the encoder's cloud of codes for input $x$, here a bell curve with mean $\mu$ and spread $\sigma$.
- $p(x\mid z)$: the decoder, how likely $x$ is given the code $z$. This is the rebuild score.
- $p(z)$: the prior, the standard bell curve we want the codes to follow. $\mathrm{KL}$: how far the cloud is from it.
- The first term rewards good rebuilds. The KL term is the leash that keeps the map smooth, with no empty gaps.

#flashcards/ml/formula #flashcards/topic/generative Write the KL term for a bell-curve encoder and explain each symbol.
?
$$\mathrm{KL}=\frac12\sum_{j=1}^{d}\left(\mu_j^2+\sigma_j^2-1-\ln\sigma_j^2\right)$$
- $\mu_j$: the cloud's centre in code number $j$. The $\mu_j^2$ term punishes a centre far from 0.
- $\sigma_j$: the cloud's spread. $\sigma_j^2-1-\ln\sigma_j^2$ is 0 at $\sigma_j=1$ and grows if the cloud is too narrow or too wide. A cloud that shrinks to a point has a huge penalty.
- $d$: the number of code numbers. The terms add up.

#flashcards/ml/formula #flashcards/topic/generative Write the reparameterization trick and explain each symbol.
?
$$z=\mu+\sigma\,\varepsilon,\qquad \varepsilon\sim\mathcal N(0,1),\qquad \frac{\partial z}{\partial\mu}=1,\quad \frac{\partial z}{\partial\sigma}=\varepsilon$$
- $\mu$, $\sigma$: the encoder's outputs, which carry the learned weights. With the trick the loss can be differentiated with respect to them.
- $\varepsilon$: the random draw. It has no weights and does not depend on the encoder, so the random part is outside the path learning goes through.
- Without the trick the gradient is blocked at the random node $z$.

#flashcards/ml/cloze #flashcards/topic/generative The VAE's reparameterization trick moves the ==randomness== into $\varepsilon$ so gradients can flow through ==z==.

#flashcards/ml/example #flashcards/topic/generative $x=1$ is rebuilt as 0.8 by a bell-curve decoder of spread 1, with $\mu=1$, $\sigma=0.5$. What are the rebuild term and the KL penalty?::Rebuild: −½(1 − 0.8)² − ½ ln(2π) = −0.02 − 0.9189 = −0.9389. KL = ½(1 + 0.25 − 1 − ln 0.25) = 0.8181.

#flashcards/ml/example #flashcards/topic/generative Two code numbers $(\mu,\sigma)=(1,0.5)$ and $(0,1)$. What is the total KL?::Number 1: ½(1 + 0.25 − 1 − ln 0.25) = 0.8181. Number 2: ½(0 + 1 − 1 − 0) = 0, already the prior. Total 0.8181. Only number 1 is charged.

#flashcards/ml/example #flashcards/topic/generative $\mu=1$, $\sigma=0.5$, $\varepsilon=0.8$, loss $t=z^2$. What are $z$ and the gradients to $\mu$ and $\sigma$?::z = 1 + 0.5 × 0.8 = 1.4. dt/dz = 2z = 2.8. To μ: 2.8 × 1 = 2.8. To σ: 2.8 × 0.8 = 2.24.

#flashcards/ml/image #flashcards/topic/generative ![[vae-reparameterized.png]]
What does this picture show?
?
A VAE. The encoder outputs a mean μ and a spread σ. A noise draw ε is combined with them into the code z, and the decoder turns z into the output x'.

#flashcards/ml/image #flashcards/topic/generative ![[vae-reparameterization-trick.png]]
What does this picture show?
?
The trick as a graph. Left, the original form: the code z is a random node that depends on the encoder's numbers (Φ) and the input x, so the gradient is blocked at z. Right, the reparameterised form: the randomness is pulled out into ε, and z is a plain differentiable function of Φ, x and ε. The red arrows show the gradient flowing back through z to Φ.
