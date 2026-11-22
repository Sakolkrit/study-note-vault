---
tags: [machine-learning, deep-learning, unsupervised-learning, representation-learning, generative-models, dimensionality-reduction]
status: growing
level: 4
tier: 3
---

# Autoencoder

An autoencoder is a neural network that learns to **squeeze its input into a short code and then rebuild the input from that code**. Because the code is small, the network has to keep only what matters.

## 🟢 1. Simple (age 12)
Imagine you have to describe a face to a friend over the phone, but you are only allowed **ten words**. You pick the most useful ones: "round, glasses, short dark hair, big smile". Your friend then draws a face from those ten words. If the drawing looks like the real face, your ten words were a good summary. An autoencoder is a pair of helpers trained like this: one **describes** the picture in a few numbers, the other **redraws** the picture from them, and they practise until the redrawing is close.

**Example:** A photo has thousands of numbers, but a photo of a handwritten digit really only needs a few facts: which digit, how slanted, how thick. An autoencoder discovers those few facts by itself, just by trying to redraw digits through a tiny gap.

## 🔵 2. How it works
1. **Encoder:** a network that takes the input (a picture, a row of numbers) and squeezes it into a short list of numbers, the **code**, also called the latent vector or **bottleneck**. See [[Neural Network]].
2. **Decoder:** a second network that takes the code and tries to rebuild the original input.
3. **Score:** compare the rebuilt input with the original. The difference is the **reconstruction error** (a [[Loss Function]]).
4. **Learn:** use [[Backpropagation]] and a method like [[Stochastic Gradient Descent]] to make both networks better at this.
5. **No labels needed:** the target is the input itself, so this is [[Unsupervised Learning]] (a form of [[Self-Supervised Learning]]).
6. **Keep the code:** after training, the short code is a compact summary of the input. You can use it to compress data, to find similar items, to spot odd ones, or as input for other models.

![[pca-vs-linear-autoencoder.png]]
*The same data (fashion-item images) squashed to 2 numbers in two ways: PCA (left) and a linear autoencoder (right). Each dot is one image and each colour is a type of clothing. The pictures look rotated and stretched relative to each other, but both models learn to cover the same flat 2-D slice of the data. Image: Michela Massi, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:PCA_vs_Linear_Autoencoder.png), CC BY-SA 4.0.*

![[denoising-autoencoder.png]]
*A denoising autoencoder. A noisy picture of a "2" goes in, is squeezed into a small code, and a clean "2" comes out. Image: neerajkrbansal1996, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Denoising-autoencoder.png), CC0.*

The basic encoder, code, decoder layout is drawn in [[Generative Models]].

## 🟠 3. Technical detail
- **The bottleneck is the point:** if the code were as big as the input, the network could just copy it and learn nothing. Making it narrow (or adding a penalty) forces it to learn the important patterns.
- **Link to PCA:** an autoencoder with straight-line (linear) layers learns the same subspace as [[PCA]]. Making the layers curved (adding an [[Activation Function]]) lets it follow **curved** structure that PCA cannot, shown in the code below. A deeper look at the maths of PCA is in [[Eigenvalue]] and [[Singular Value Decomposition]].
- **Kinds of autoencoder:**
  - **Undercomplete:** a narrow bottleneck (the basic kind).
  - **Denoising:** feed a corrupted input and ask for the clean one. This stops the network from just copying and teaches it the real structure.
  - **Sparse:** the code may be wide, but a penalty keeps most of its numbers near zero.
  - **Variational (VAE):** the code is a small *distribution*, not a single point, which lets you **sample new codes and decode them into new examples** (see [[Generative Models]]).
- **Uses today:** compressing images and audio, removing noise, detecting anomalies (an item that rebuilds badly is unusual), learning features, and as a building block inside larger systems. Many image-generating systems work in the compressed code space of an autoencoder to save compute (from memory).
- **Limits:** the rebuilt output of a plain autoencoder is often blurry. The codes of a plain autoencoder are not arranged so that random codes decode into sensible things, which is what the VAE fixes. And the code is only as good as the loss: low reconstruction error does not guarantee a code that is useful for another task.
- **Related to embeddings:** the code is an [[Embedding]] of the input. Unlike [[Word2Vec]], it is learned by rebuilding the input, not by predicting neighbours.
- **Related to compression:** it is *lossy* compression learned from data, like [[Quantization]] or low-rank tricks, but it compresses the input rather than the model's own weights.

Two experiments below (they need PyTorch; I ran them with version 2.8.0 on CPU, and other versions may print slightly different last digits).

```python
import numpy as np, torch
torch.manual_seed(0); rng = np.random.default_rng(0)
# 1) linear autoencoder vs PCA: data in 5-D that mostly lives on a 2-D plane
Z = rng.normal(0, 1, (500, 2)) * [3, 1.5]; X = Z @ rng.normal(0, 1, (2, 5)) + 0.1 * rng.normal(0, 1, (500, 5))
Xc = X - X.mean(0); lam, V = np.linalg.eigh(np.cov(Xc.T)); lam = lam[::-1]; V = V[:, ::-1]
pca_err = ((Xc - Xc @ V[:, :2] @ V[:, :2].T) ** 2).sum(1).mean()
print(round(float(pca_err), 4), round(float(lam[2:].sum() * 499 / 500), 4))        # 0.0303 0.0303   PCA error = sum of dropped eigenvalues
Xt = torch.tensor(Xc, dtype=torch.float32)
enc, dec = torch.nn.Linear(5, 2, bias=False), torch.nn.Linear(2, 5, bias=False)
opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=0.02)
for _ in range(3000):
    loss = ((dec(enc(Xt)) - Xt) ** 2).sum(1).mean(); opt.zero_grad(); loss.backward(); opt.step()
print(round(loss.item(), 4))                                                          # 0.0305   linear autoencoder: about the same error
# 2) curved data: 1-number code. PCA (a straight line) vs a nonlinear autoencoder
t = rng.uniform(-2.5, 2.5, 600); C = np.c_[np.cos(t), np.sin(t)]                     # an arc of a circle in 2-D
Cc = C - C.mean(0); l2, V2 = np.linalg.eigh(np.cov(Cc.T)); v = V2[:, -1]
pca1 = ((Cc - np.outer(Cc @ v, v)) ** 2).sum(1).mean()
Ct = torch.tensor(Cc, dtype=torch.float32)
net = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1), torch.nn.Linear(1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2))
opt = torch.optim.Adam(net.parameters(), lr=0.01)
for _ in range(4000):
    l = ((net(Ct) - Ct) ** 2).sum(1).mean(); opt.zero_grad(); l.backward(); opt.step()
print(round(float(pca1), 4), round(l.item(), 5))                                      # 0.3237 3e-05
# 3) VAE penalty: KL( N(mu, sigma^2) || N(0,1) )
kl = lambda mu, s: 0.5 * (mu ** 2 + s ** 2 - 1 - np.log(s ** 2))
print([round(float(kl(m, s)), 3) for m, s in [(0, 1), (1, 1), (0, 2), (2, 0.5)]])   # [0.0, 0.5, 0.807, 2.318]
```

Reading the output. In test 1, the data live almost on a flat 2-D plane inside 5 dimensions. PCA with 2 directions leaves an average squared error of 0.0303, which equals the sum of the eigenvalues it dropped (the maths below). A **linear** autoencoder with a 2-number code trained for 3,000 steps reaches 0.0305, almost the same, because it learns the same plane. In test 2, the data lie on a **curved** arc of a circle. One straight direction (PCA with 1 component) leaves an error of 0.3237, while a small nonlinear autoencoder with a **1-number code** drives it down to about 0.00003 on its own training data (this is training error, not tested on new points). It works because the arc really is a one-parameter curve, and the autoencoder finds the position along the curve. Test 3 checks the VAE penalty formula from below.

## 🔴 4. Deep / formal
**The reconstruction loss.** With an encoder $f$ and a decoder $g$, the network is trained to make the output close to the input over $n$ examples:

$$
\mathcal L=\frac1n\sum_{i=1}^{n}\big\lVert \mathbf x_i-g\big(f(\mathbf x_i)\big)\big\rVert^{2}
$$

**Where:**
- $\mathbf x_i$ is the $i$-th input (a list of numbers, such as the pixels of an image).
- $f(\mathbf x_i)$ is the code (the short list). $g(\cdot)$ turns the code back into a list of the original size. So $g(f(\mathbf x_i))$ is the rebuilt input, often written $\hat{\mathbf x}_i$.
- $\lVert\cdot\rVert^{2}$ is the squared length: square each gap between rebuilt and original, then add them up.
- **Matters:** the loss only rewards getting the input back **through the narrow code**. If the code is not narrow, the answer is trivial (copy everything) and the loss is useless. The squared error also pushes the output towards averages, which is one reason plain autoencoders give blurry rebuilds.

**Example (tiny numbers):** input $\mathbf x=(1,2,3)$ and rebuilt $\hat{\mathbf x}=(1.1,\,1.8,\,3.3)$.
- Gaps: $-0.1,\;+0.2,\;-0.3$. Squares: $0.01,\;0.04,\;0.09$.
- Sum: $0.01+0.04+0.09=\mathbf{0.14}$ (the average per number would be $0.047$).
- Meaning: the rebuild is close but not perfect. Training would shrink this number over many examples by adjusting both networks.

**Linear autoencoder equals PCA.** If both $f$ and $g$ are linear and the code has $k$ numbers, the best possible reconstruction error on centred data is:

$$
\mathcal L_{\min}=\sum_{i=k+1}^{d}\lambda_i
$$

**Where:**
- $d$ is the number of numbers in the input. $k$ is the size of the code.
- $\lambda_1\ge\lambda_2\ge\dots\ge\lambda_d$ are the eigenvalues of the data's covariance matrix (see [[PCA]] and [[Eigenvalue]]). Each one is the spread of the data along one direction.
- The sum adds the eigenvalues of the **dropped** directions, the ones beyond the first $k$.
- **Matters:** the best a linear autoencoder can do is throw away the least spread directions, which is exactly what [[PCA]] does. So the extra power of an autoencoder comes only from **nonlinearity** (curved layers). The result is for the best possible solution; training may or may not reach it exactly, and the code it finds is a rotated version of the PCA directions, as the picture shows.

**Example (tiny numbers):** a data set with two directions whose spreads (eigenvalues) are $9.955$ and $0.045$ (the same numbers as the worked example in [[PCA]]), and a 1-number code ($k=1$).
- Dropped directions: only the second, with $\lambda_2=0.045$.
- $\mathcal L_{\min}=\mathbf{0.045}$.
- Meaning: the best 1-number linear summary loses 0.045 of squared error on average, and keeps the 9.955 that mattered. In the code, the 5-D data with $k=2$ gave the matching $0.0303$ and $0.0305$.

**Denoising.** Corrupt the input with noise, but ask for the clean original back:

$$
\mathcal L_{\text{den}}=\frac1n\sum_{i=1}^{n}\big\lVert \mathbf x_i-g\big(f(\tilde{\mathbf x}_i)\big)\big\rVert^{2},\qquad \tilde{\mathbf x}_i=\mathbf x_i+\text{noise}
$$

**Where:**
- $\mathbf x_i$ is the clean input (the target). $\tilde{\mathbf x}_i$ ("x-tilde") is the same input after noise has been added.
- $f$ and $g$ are the encoder and decoder as before. The encoder only ever **sees** $\tilde{\mathbf x}_i$.
- **Matters:** the comparison is with the *clean* $\mathbf x_i$, not with the noisy input. So copying the input is punished (it would copy the noise too), and the network has to learn what clean data looks like. If you compared against the noisy input, the network would have no reason to remove noise.

**Example (tiny numbers):** clean $\mathbf x=(1,2,3)$, noisy input $\tilde{\mathbf x}=(1.2,\,1.7,\,3.4)$, and the network outputs $(1.1,\,1.9,\,3.1)$.
- Error against the clean input: gaps $0.1,\,-0.1,\,0.1$, squares $0.01\times3=\mathbf{0.03}$.
- For comparison, simply copying the noisy input would score $0.2^2+0.3^2+0.4^2=0.04+0.09+0.16=\mathbf{0.29}$.
- Meaning: the network's cleaned-up output is about 10 times closer to the truth than the noisy input, so its loss is much lower than copying. That is what training rewards.

**The VAE penalty.** A variational autoencoder makes the encoder output a mean $\mu$ and a spread $\sigma$ for each code number, and adds a penalty that keeps them close to a plain bell curve $\mathcal N(0,1)$. For one code number the penalty is the [[KL Divergence]] from $\mathcal N(\mu,\sigma^2)$ to $\mathcal N(0,1)$:

$$
D_{\mathrm{KL}}\big(\mathcal N(\mu,\sigma^2)\,\big\|\,\mathcal N(0,1)\big)=\tfrac12\big(\mu^{2}+\sigma^{2}-1-\ln\sigma^{2}\big)
$$

**Where:**
- $\mu$ is the mean the encoder gives for this code number, and $\sigma$ is its spread (a positive number).
- $\mathcal N(0,1)$ is the standard bell curve (mean 0, spread 1), the shape we want all codes to resemble. See [[Normal Distribution]].
- $\ln\sigma^{2}$ is the natural log of the squared spread.
- **Matters:** the penalty is $0$ only when $\mu=0$ and $\sigma=1$, and grows as the code drifts away from the standard bell. Keeping codes near one bell curve is what makes it safe to **pick a random code from that bell and decode it** into something sensible. Without the penalty, the codes would scatter in unrelated places, with gaps in between that decode into nonsense. In practice it is added to the reconstruction loss, weighting the two against each other.

**Example (tiny numbers):** three code numbers.
- $\mu=0,\;\sigma=1$: $\tfrac12(0+1-1-0)=\mathbf{0}$. A perfect standard bell costs nothing.
- $\mu=1,\;\sigma=1$: $\tfrac12(1+1-1-0)=\mathbf{0.5}$. The centre has shifted by 1.
- $\mu=0,\;\sigma=2$: $\tfrac12(0+4-1-\ln4)=\tfrac12(3-1.386)=\mathbf{0.807}$. Too wide.
- Meaning: the further a code's bell is from the standard one (shifted or too wide), the bigger the penalty. Too narrow is penalised too: $\mu=2,\;\sigma=0.5$ gives $2.318$. The code checks these values.

**Further notes:**
- **Why not just sample from a plain autoencoder?** Its codes sit in scattered clusters, and a random point between them may decode to garbage. The VAE penalty fills the space evenly.
- **The reparameterisation trick:** to train through the random sampling step, the VAE writes the sample as $\mu+\sigma\epsilon$ with $\epsilon$ a fixed-shape random number, so gradients can flow (from memory; see the Kingma and Welling paper).
- **Closed form:** the KL formula above is the standard result for two bell curves (from memory; the derivation is not shown here).
- **Trade-off:** more weight on the penalty gives smoother, more usable codes but blurrier rebuilds. More weight on the reconstruction gives sharper rebuilds but a messier code space.

**What came before, and how this improved on it:**
- **Before:** [[PCA]], which compresses data with a straight-line (linear) map.
- **Its limit:** if the data lies on a curve, a straight line through it loses the bend.
- **What changed:** let a [[Neural Network]] with a non-linear activation do the encoding and decoding, trained to rebuild its input (Hinton and Salakhutdinov, Science, 2006; from memory, not checked).
- **How much better:** in the code below, 400 points on the curve $y=x^2$ are squeezed to one number each. PCA rebuilds them with an average squared error of 0.0935, a small autoencoder with 5e-6. The curve is easy and noise-free, so a real gap would be smaller.

**Check it yourself:**

```python
import numpy as np, torch
torch.manual_seed(0); rng = np.random.default_rng(0)
x = rng.uniform(-1, 1, 400); X = np.stack([x, x ** 2], 1)            # points on a curve
Xt = torch.tensor(X, dtype=torch.float32)
c = X - X.mean(0); v = np.linalg.svd(c, full_matrices=False)[2][0]    # PCA, 1 number per point
pca_err = ((c - np.outer(c @ v, v)) ** 2).sum(1).mean()
net = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1),
                          torch.nn.Linear(1, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2))
opt = torch.optim.Adam(net.parameters(), 1e-2)
for _ in range(3000):
    opt.zero_grad(); l = ((net(Xt) - Xt) ** 2).sum(1).mean(); l.backward(); opt.step()
print("1-number code, PCA error:", round(float(pca_err), 4), " autoencoder error:", round(l.item(), 6))
```

## Related
- [[Neural Network]] · [[Unsupervised Learning]] · [[Self-Supervised Learning]] · [[Generative Models]] · [[PCA]] · [[Eigenvalue]] · [[Singular Value Decomposition]] · [[Embedding]] · [[Loss Function]] · [[Backpropagation]] · [[Activation Function]] · [[KL Divergence]] · [[Normal Distribution]] · [[Quantization]] · [[Word2Vec]] · [[Diffusion Models]] · [[GAN]] · [[Variational Autoencoder]] · [[Convolution]] · [[Image Segmentation]]
- [[Anomaly Detection]] · [[Isolation Forest]] · [[Anomaly Detection Shoot-out]]
## Sources
- Hinton & Salakhutdinov, *Reducing the Dimensionality of Data with Neural Networks*, Science, 2006 (from memory)
- Vincent et al., *Extracting and Composing Robust Features with Denoising Autoencoders*, 2008 (from memory)
- Kingma & Welling, *Auto-Encoding Variational Bayes*, 2013 (from memory)
- Baldi & Hornik, *Neural networks and principal component analysis: learning from examples without local minima*, 1989 (from memory)
- Goodfellow, Bengio & Courville, *Deep Learning*, chapter 14 (free online)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/generative Autoencoder:::A network that squeezes an input into a short code and then rebuilds the input from the code, like describing a face in ten words and having a friend draw it. If the drawing looks right, the code was a good summary.

#flashcards/ml/formula #flashcards/topic/generative Write the autoencoder loss and explain each symbol.
?
$$\mathcal L=\frac1n\sum_{i=1}^{n}\big\lVert \mathbf x_i-g\big(f(\mathbf x_i)\big)\big\rVert^{2}$$
- $\mathbf x_i$: the $i$-th input (a list of numbers, such as image pixels). $n$: how many examples.
- $f(\mathbf x_i)$: the code (the short list). $g(\cdot)$: turns the code back into a list of the original size. $g(f(\mathbf x_i))$ is the rebuilt input $\hat{\mathbf x}_i$.
- $\lVert\cdot\rVert^2$: the squared length of the gap. Small means a good rebuild.

#flashcards/ml/formula #flashcards/topic/generative Write the best loss of a linear autoencoder and explain each symbol.
?
$$\mathcal L_{\min}=\sum_{i=k+1}^{d}\lambda_i$$
- $d$: the number of numbers in the input. $k$: the size of the code.
- $\lambda_1\ge\dots\ge\lambda_d$: the eigenvalues of the data's covariance matrix. Each is the spread of the data along one direction.
- The sum adds the eigenvalues of the dropped directions. The best linear autoencoder loses exactly what PCA loses.

#flashcards/ml/formula #flashcards/topic/generative Write the denoising autoencoder loss and explain each symbol.
?
$$\mathcal L_{\text{den}}=\frac1n\sum_{i=1}^{n}\big\lVert \mathbf x_i-g\big(f(\tilde{\mathbf x}_i)\big)\big\rVert^{2},\qquad \tilde{\mathbf x}_i=\mathbf x_i+\text{noise}$$
- $\mathbf x_i$: the clean input (the target). $\tilde{\mathbf x}_i$: the same input after noise has been added.
- $f$, $g$: the encoder and decoder as before. The encoder only ever sees $\tilde{\mathbf x}_i$.
- The comparison is with the clean $\mathbf x_i$, not the noisy input, so the network cannot just copy.

#flashcards/ml/formula #flashcards/topic/generative Write the KL penalty between a Gaussian code and the standard bell and explain each symbol.
?
$$D_{\mathrm{KL}}\big(\mathcal N(\mu,\sigma^2)\,\big\|\,\mathcal N(0,1)\big)=\tfrac12\big(\mu^{2}+\sigma^{2}-1-\ln\sigma^{2}\big)$$
- $\mu$: the mean the encoder gives for this code number. $\sigma$: its spread (positive).
- $\mathcal N(0,1)$: the standard bell curve, the shape we want all codes to resemble.
- $\ln\sigma^2$: the natural log of the squared spread. The penalty is 0 when $\mu=0$ and $\sigma=1$ and grows as the code drifts from that.

#flashcards/ml/cloze #flashcards/topic/generative A linear autoencoder with a code of size $k$ loses the ==smallest== eigenvalues, the same as ==PCA==.

#flashcards/ml/example #flashcards/topic/generative Input $(1,2,3)$ and rebuilt $(1.1,1.8,3.3)$. What is the loss?::Gaps −0.1, +0.2, −0.3. Squares 0.01, 0.04, 0.09. Sum 0.14 (average per number 0.047).

#flashcards/ml/example #flashcards/topic/generative Eigenvalues 9.955 and 0.045 and a 1-number code. What is $\mathcal L_{\min}$?::Only the second direction is dropped, so L_min = 0.045. The best 1-number linear summary loses only 0.045 of the 10 units of spread.

#flashcards/ml/example #flashcards/topic/generative Clean $(1,2,3)$, noisy input $(1.2,1.7,3.4)$ and output $(1.1,1.9,3.1)$. What is the denoising loss, and what would copying the noisy input score?::Gaps 0.1, −0.1, 0.1, squares sum 0.03. Copying the noisy input: 0.04 + 0.09 + 0.16 = 0.29. The denoiser is much better.

#flashcards/ml/example #flashcards/topic/generative What is the KL penalty for $(\mu,\sigma)=(0,1)$, $(1,1)$ and $(0,2)$?::(0, 1): 0, a perfect standard bell costs nothing. (1, 1): ½(1 + 1 − 1 − 0) = 0.5. (0, 2): ½(0 + 4 − 1 − ln 4) = 0.807. The further the code drifts, the more it costs.

#flashcards/ml/image #flashcards/topic/generative ![[pca-vs-linear-autoencoder.png]]
What does this picture show?
?
The same fashion-item images squashed to 2 numbers by PCA (left) and a linear autoencoder (right). Each dot is one image and each colour a type of clothing. The pictures look rotated and stretched relative to each other, but both models learn to cover the same flat 2-D slice of the data.

#flashcards/ml/image #flashcards/topic/generative ![[denoising-autoencoder.png]]
What does this picture show?
?
A denoising autoencoder. A noisy picture of a "2" goes in, is squeezed into a small code, and a clean "2" comes out.
