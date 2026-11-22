---
tags: [machine-learning, deep-learning, vision, multimodal, contrastive, embedding, zero-shot]
status: growing
level: 4
tier: 3
---

# CLIP

CLIP (Contrastive Language-Image Pre-training) is a model from OpenAI that learns **to match pictures with the sentences that describe them**. Once it can do that, you can "classify" a picture just by writing the possible answers as sentences.

## 🟢 1. Simple (age 12)
CLIP looks at a huge number of pictures that each come with a caption, and it learns which caption goes with which picture. Later, if you show it a new picture and give it a list of sentences, it can pick the sentence that fits best. You never have to train it again for the new list.

**Example:** You have a photo, and three cards: "a photo of a dog", "a photo of a cat", "a photo of a car". CLIP holds the photo up next to each card and says which card matches best. If tomorrow you write cards for "a photo of a zebra" and "a photo of a giraffe", it works just as well, with no extra lessons.

## 🔵 2. How it works
1. Collect a huge pile of (picture, caption) pairs from the internet. The CLIP paper's abstract says 400 million pairs.
2. Use two networks. The **image encoder** turns a picture into a list of numbers, and the **text encoder** turns a sentence into a list of numbers. Both lists live in the same space, so they can be compared (see [[Embedding]]).
3. Make each list length 1 and compare with [[Cosine Similarity]]: a bigger number means "more alike".
4. Take a batch of $N$ pairs. Compare **every** picture with **every** caption. That gives an $N$ by $N$ table of scores.
5. Training rule: on the diagonal of the table (each picture with its own caption) the score should be high, and everywhere else it should be low. This is [[Contrastive Learning]], the same matching-game idea in the previous note.
6. The training step is a quiz run in two directions: "given this picture, pick its caption from the batch", and "given this caption, pick its picture".
7. After training, to classify a picture into your own list of classes, write each class as a sentence such as "A photo of a dog." Turn each sentence into numbers. Pick the class whose sentence is most similar to the picture. This is called **zero-shot** classification (zero training examples for these classes).
8. Because it learned from natural language, you can describe new concepts in words.

![[clip-overview.png]]
*The CLIP recipe: (1) train an image encoder and a text encoder so that each picture matches its own caption in an N by N table; (2) turn the class names into sentences like "A photo of a {object}." and encode them; (3) encode a new picture and pick the class sentence with the highest score. Image: OpenAI, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Contrastive_Language-Image_Pretraining.png), MIT licence.*

## 🟠 3. Technical detail
- **Two encoders:** the image side can be a [[Convolutional Neural Network]] such as a [[ResNet]] or a [[Vision Transformer]] (see [[ViT 2020]]); the text side is a [[Transformer]]. The paper trained five ResNets and three Vision Transformers, and its main model is a ViT-L/14 (checked in [[CLIP 2021]]).
- **Normalise, then dot product:** each embedding is scaled to length 1, so the dot product is the cosine similarity.
- **Learned temperature:** the scores are multiplied by a learnable number (the inverse of the temperature $\tau$) before the [[Softmax]]. The paper starts it at the equivalent of 0.07 and clips it so that the scores are never scaled by more than 100 (checked in [[CLIP 2021]]).
- **Symmetric loss:** a [[Cross-Entropy Loss]] over the rows of the table (picture to text) plus one over the columns (text to picture), averaged.
- **Batch size matters:** a bigger batch gives more wrong captions to tell apart. This is the same effect as in [[Contrastive Learning]].
- **Prompt wording:** the sentence template changes the result, so choosing it is a small [[Prompt Engineering]] job. The figure above uses "A photo of a {object}."
- **Uses:** zero-shot classification; searching images by text; scoring how well a caption fits a picture; and as a ready-made image-and-text embedding that other systems build on (for example, text-guided image generators such as [[Diffusion Models]] use text embeddings; from memory).
- **Weak spots:** it can only choose among the sentences you give it. It was trained on captions from the internet, so it inherits their gaps and biases. The paper's limitations section lists weak fine-grained classification (car models, flowers, aircraft), weak counting, and only 88% on handwritten MNIST digits (checked in [[CLIP 2021]]).

```python
import numpy as np, torch, torch.nn.functional as F
torch.manual_seed(0)

colors, shapes = ["red", "green", "blue"], ["circle", "square", "star", "line"]
combos = [(c, s) for c in range(3) for s in range(4)]
held = [(0, 2), (1, 3), (2, 0)]                       # these 3 pairings never appear in training
train = [p for p in combos if p not in held]
col_vec, shp_vec = torch.randn(3, 6), torch.randn(4, 6)

def make_images(pairs, n):                            # "image" = colour pattern + shape pattern + noise
    idx = torch.tensor(pairs)[torch.randint(0, len(pairs), (n,))]
    return torch.cat([col_vec[idx[:, 0]], shp_vec[idx[:, 1]]], 1) + 0.3 * torch.randn(n, 12), idx

img_enc = torch.nn.Linear(12, 16)
word_emb = torch.nn.Embedding(7, 16)                  # 3 colour words + 4 shape words
txt_enc = lambda idx: word_emb(idx[:, 0]) + word_emb(idx[:, 1] + 3)      # caption "red circle" -> sum of word vectors
log_scale = torch.nn.Parameter(torch.tensor(2.0))     # learnable temperature, as in CLIP (log of 1/tau)
opt = torch.optim.Adam(list(img_enc.parameters()) + list(word_emb.parameters()) + [log_scale], lr=1e-2)

for step in range(400):
    x, idx = make_images(train, 128)
    s = F.normalize(img_enc(x), dim=1) @ F.normalize(txt_enc(idx), dim=1).T * log_scale.exp()
    t = torch.arange(128)
    loss = (F.cross_entropy(s, t) + F.cross_entropy(s.T, t)) / 2     # symmetric loss: image->text and text->image
    opt.zero_grad(); loss.backward(); opt.step()
print("final loss", round(loss.item(), 3))

prompts = torch.tensor(combos)                        # zero-shot: one "prompt" for each of the 12 pairings
with torch.no_grad():
    T = F.normalize(txt_enc(prompts), dim=1)
    for name, pairs in [("seen pairings", train), ("never-seen pairings", held)]:
        x, idx = make_images(pairs, 600)
        pred = (F.normalize(img_enc(x), dim=1) @ T.T).argmax(1)
        truth = torch.tensor([combos.index((int(a), int(b))) for a, b in idx])
        print(name, "accuracy", round((pred == truth).float().mean().item(), 3), " chance", round(1 / 12, 3))

S = np.array([[0.9, 0.1], [0.2, 0.8]])               # tiny numbers for the note (temperature 1)
row = -np.log(np.exp(S) / np.exp(S).sum(1, keepdims=True)).diagonal()
col = -np.log(np.exp(S) / np.exp(S).sum(0, keepdims=True)).diagonal()
print(row.round(4), col.round(4), round(row.mean(), 4), round(col.mean(), 4), round((row.mean() + col.mean()) / 2, 4))
```

This is a **toy**, not the real CLIP. The "images" are made-up number lists (a colour pattern plus a shape pattern plus noise), the image encoder is one linear layer, and a caption such as "red circle" is the sum of two word vectors. It was trained on 9 of the 12 colour-and-shape pairings. Output: `final loss 2.724`, then `seen pairings accuracy 0.998  chance 0.083`, then `never-seen pairings accuracy 0.993  chance 0.083`, then the last line `[0.3711 0.4375] [0.4032 0.4032] 0.4043 0.4032 0.4037`. So the toy picked the right sentence for pairings it had never seen in training 99.3% of the time, by combining what it learned about colours and about shapes. The real CLIP was not run for this note.

## 🔴 4. Deep / formal
**The loss.** For a batch of $N$ pairs, let $u_i$ be the normalised image embedding and $v_j$ the normalised text embedding, and let $s_{ij}=u_i\cdot v_j$ be their cosine similarity. The loss runs the quiz in both directions:

$$
\mathcal L=\tfrac12\big(\mathcal L_{\text{img}\to\text{txt}}+\mathcal L_{\text{txt}\to\text{img}}\big),\qquad \mathcal L_{\text{img}\to\text{txt}}=-\frac1N\sum_{i=1}^{N}\log\frac{e^{s_{ii}/\tau}}{\sum_{j=1}^{N}e^{s_{ij}/\tau}}
$$

**Where:**
- $N$ is the number of pairs in the batch.
- $u_i$ is the image embedding and $v_j$ the text embedding, each with length 1. $s_{ij}$ is how alike picture $i$ and caption $j$ are.
- $\tau$ is the temperature. In CLIP it is a learned number.
- $\mathcal L_{\text{img}\to\text{txt}}$ is the "pick the caption for each picture" loss (each row of the table). $\mathcal L_{\text{txt}\to\text{img}}$ is the same with the table's columns, "pick the picture for each caption".
- **Matters:** doing only one direction would let one side cheat, for example by making all captions look alike. Averaging both directions makes the picture and the text meet in the middle. Making $\tau$ learnable stops you having to guess a good value by hand.

**Example (tiny numbers):** the table $s_{11}=0.9,\ s_{12}=0.1,\ s_{21}=0.2,\ s_{22}=0.8$, with $\tau=1$.
- Picture to text (rows): the losses are $0.3711$ and $0.4375$, with average $0.4043$.
- Text to picture (columns): column 1 gives $-\ln\frac{e^{0.9}}{e^{0.9}+e^{0.2}}=0.4032$ and column 2 gives $-\ln\frac{e^{0.8}}{e^{0.1}+e^{0.8}}=0.4032$, so the average is $0.4032$.
- Symmetric loss: $(0.4043+0.4032)/2=\mathbf{0.4037}$.
- Meaning: the model ranks the right partner first in both directions, so the loss is modest. (My arithmetic, the same as the last line of the code.)

**Zero-shot prediction.** To classify a picture $x$ into classes $c$, write a sentence $t_c$ for each class (for example "a photo of a dog") and pick the closest one:

$$
\hat c=\arg\max_{c}\ \cos\big(f(x),\,g(t_c)\big)
$$

**Where:**
- $x$ is the picture and $t_c$ is the sentence made for class $c$.
- $f$ is the image encoder and $g$ is the text encoder.
- $\cos(\cdot,\cdot)$ is the cosine similarity of the two embeddings.
- $\arg\max_c$ means "the class $c$ with the highest score".
- **Matters:** no class-specific classifier is trained. To change the list of classes you only change the sentences. The price is that the wording of $t_c$ matters.

**Example (tiny numbers, made up for illustration):** the picture embedding is $(0.8,\,0.6)$, and the class sentences embed to dog $(0.6,\,0.8)$, cat $(1,\,0)$ and car $(0,\,-1)$. All three have length 1.
- dog: $0.8\times0.6+0.6\times0.8=\mathbf{0.96}$.
- cat: $0.8\times1+0.6\times0=\mathbf{0.80}$.
- car: $0.8\times0+0.6\times(-1)=\mathbf{-0.60}$.
- Meaning: "dog" scores highest, so the prediction is dog.

**Further notes:**
- **Link to mutual information:** the quiz loss gives a lower bound on how much the picture and the caption share (see [[Contrastive Learning]] and [[Mutual Information]]).
- **Where CLIP sits:** it is a way of learning a shared embedding space for two kinds of data. The same recipe can pair other kinds of data, such as audio and text (my understanding; not checked).

**What came before, and how this improved on it:**
- **Before:** image classifiers trained on a fixed list of labelled categories, such as the 1.28 million labelled ImageNet training images (see [[Image Classification]] and [[ResNet]]).
- **Its limit:** the abstract says this restricted form of supervision limits generality and usability, because more labelled data is needed to specify any other visual concept.
- **What changed:** learn directly from raw text about images, with the simple task of predicting which caption goes with which image, on 400 million (image, text) pairs (arXiv 2103.00020, 26 February 2021; the authors are from OpenAI, and Radford and Kim are the equal first authors, checked in [[CLIP 2021]]).
- **How much better:** the abstract says that CLIP matches the accuracy of the original ResNet-50 on ImageNet zero-shot, without using any of its 1.28 million training examples, and is often competitive with fully supervised baselines on over 30 datasets. In this note's toy, zero-shot accuracy on pairings never seen in training was 99.3%, against 8.3% for chance.

## Related
- [[Contrastive Learning]] · [[Embedding]] · [[Cosine Similarity]] · [[Vision Transformer]] · [[ViT 2020]] · [[Transformer]] · [[Convolutional Neural Network]] · [[ResNet]] · [[Image Classification]] · [[Transfer Learning]] · [[Prompt Engineering]] · [[Softmax]] · [[Cross-Entropy Loss]] · [[Self-Supervised Learning]] · [[Embedding Model]] · [[Diffusion Models]] · [[Mutual Information]] · [[CLIP 2021]]

## Sources
- Radford et al. (OpenAI), "Learning Transferable Visual Models From Natural Language Supervision", arXiv 2103.00020, 26 February 2021 (abstract and most of the body read; see [[CLIP 2021]] for exactly what was read)
- Image: OpenAI, Wikimedia Commons, MIT licence (see the caption in section 2)
- Numbers in this note come from the code above (my arithmetic and my run on this machine). The real CLIP weights were not run.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/vision CLIP:::Contrastive Language-Image Pre-training: two encoders (image and text) trained so each picture matches its own caption, which allows zero-shot classification from class names written as sentences.

#flashcards/ml/concept #flashcards/topic/vision How does CLIP classify a picture into classes it was never trained on?
?
Write each class as a sentence ("A photo of a dog."), encode all sentences with the text encoder, encode the picture with the image encoder, and pick the sentence with the highest cosine similarity. No classifier is trained, and changing classes only changes the sentences.

#flashcards/ml/formula #flashcards/topic/vision Write the CLIP loss and explain each symbol.
?
$$\mathcal L=\tfrac12\big(\mathcal L_{\text{img}\to\text{txt}}+\mathcal L_{\text{txt}\to\text{img}}\big),\qquad \mathcal L_{\text{img}\to\text{txt}}=-\frac1N\sum_{i=1}^{N}\log\frac{e^{s_{ii}/\tau}}{\sum_{j=1}^{N}e^{s_{ij}/\tau}}$$
- $N$: the number of pairs in the batch.
- $s_{ij}$: the cosine similarity of image embedding $u_i$ and text embedding $v_j$ (both of length 1).
- $\tau$: the temperature, learned in CLIP.
- $\mathcal L_{\text{img}\to\text{txt}}$: pick the caption for each picture (rows of the table). $\mathcal L_{\text{txt}\to\text{img}}$: pick the picture for each caption (columns).
- One direction alone would let one side cheat. Averaging both makes them meet in the middle. A learned $\tau$ avoids tuning it by hand.

#flashcards/ml/example #flashcards/topic/vision Similarities 0.9, 0.1 / 0.2, 0.8 with $\tau=1$: what is the symmetric CLIP loss?::Rows (image to text): 0.3711 and 0.4375, average 0.4043. Columns (text to image): 0.4032 and 0.4032, average 0.4032. Symmetric loss: (0.4043 + 0.4032)/2 = 0.4037. My arithmetic.

#flashcards/ml/formula #flashcards/topic/vision Write the CLIP zero-shot prediction rule and explain each symbol.
?
$$\hat c=\arg\max_{c}\ \cos\big(f(x),\,g(t_c)\big)$$
- $x$: the picture. $t_c$: the sentence made for class $c$ (for example "a photo of a dog").
- $f$: the image encoder. $g$: the text encoder.
- $\cos$: cosine similarity of the two embeddings. $\arg\max_c$: the class with the highest score.
- No class-specific classifier is trained. The price is that the wording of $t_c$ matters.

#flashcards/ml/example #flashcards/topic/vision Picture embedding (0.8, 0.6), prompts dog (0.6, 0.8), cat (1, 0), car (0, -1): what does zero-shot predict?::dog: 0.8 × 0.6 + 0.6 × 0.8 = 0.96. cat: 0.80. car: -0.60. The highest is dog, so the prediction is dog. Numbers made up for illustration.

#flashcards/ml/image #flashcards/topic/vision ![[clip-overview.png]]
What does this picture show?
?
The CLIP recipe: (1) train an image encoder and a text encoder so that each picture matches its own caption in an N by N table; (2) turn class names into sentences like "A photo of a {object}." and encode them; (3) encode a new picture and pick the class sentence with the highest score.

#flashcards/ml/cloze #flashcards/topic/vision CLIP's abstract says it was trained on ==400 million== (image, text) pairs.

#flashcards/ml/cloze #flashcards/topic/vision The CLIP abstract says it matches the original ==ResNet-50== on ImageNet zero-shot without using any of its ==1.28 million== training examples.

#flashcards/ml/concept #flashcards/topic/vision In the note's toy CLIP, how accurate was zero-shot classification on colour-and-shape pairings never seen in training?::99.3%, against 8.3% for chance (12 prompts). Seen pairings: 99.8%. A synthetic toy run on this machine, not the real CLIP.

#flashcards/ml/concept #flashcards/topic/vision Name two weak spots of CLIP.
?
It can only choose among the sentences you give it, and the wording matters (prompt engineering). It inherits the gaps and biases of internet captions. Fine detail such as counting is weak, and handwritten MNIST digits reach only 88% (paper's limitations section, read in the CLIP 2021 note).
