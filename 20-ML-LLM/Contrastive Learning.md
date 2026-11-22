---
tags: [machine-learning, deep-learning, self-supervised, representation-learning, embedding, contrastive]
status: growing
level: 4
tier: 3
---

# Contrastive Learning

Contrastive learning teaches a model by showing it **what belongs together and what does not**. Matching pairs are pulled close, and mismatched pairs are pushed apart. No human-written labels are needed.

## 🟢 1. Simple (age 12)
Imagine a game with a big pile of photos. Some photos come in pairs: two photos of the same kitten, taken from different sides. You are told which photos are a pair, but not what is in them. After playing for a long time you start to notice what makes a kitten a kitten, and you can tell it from a corgi. The computer learns the same way: **pull the matching pairs together and push everything else away**.

**Example:** You pour out a bag of socks. You match each sock with its partner, and you know that a sock and a different sock are not partners. You never needed anyone to tell you "this is a striped sock". Knowing which ones match was enough to learn what stripes are.

## 🔵 2. How it works
1. Start with data that has **no labels**, such as a pile of photos or pairs of (photo, caption).
2. Build a **positive pair**: two different "views" of the same item. For a photo, take two random crops with different colours ([[Data Augmentation]]). Or use a photo and its caption.
3. Every other item in the same batch is a **negative**: it does not belong with this one.
4. A network (the "encoder") turns each view into a list of numbers, an [[Embedding]].
5. Measure how alike two embeddings are with [[Cosine Similarity]].
6. Turn the training step into a quiz: "here is one view, and a lineup of all the other views in the batch. Which one is its partner?" The network answers with scores, which become chances through [[Softmax]]. The [[Cross-Entropy Loss]] is small when the partner gets the highest chance.
7. Learning from this loss makes embeddings of partners close and embeddings of non-partners far apart.
8. After training, the embeddings group things by meaning. You can reuse them for other tasks with very few labels ([[Transfer Learning]]).
9. Since the labels come from the data itself (this crop and that crop are from the same photo), this is a kind of [[Self-Supervised Learning]].

![[contrastive-pairs.png]]
*Positive and negative pairs. The two kittens form a positive pair (the green arrow): their embeddings should be close. The kitten-and-puppy pairs (the red arrows) are negatives: their embeddings should be far apart. Image: "A potato hater", [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Positive_and_negative_pairs_for_contrastive_learning.png), CC BY-SA 4.0.*

## 🟠 3. Technical detail
- **Loss:** the common loss is called **InfoNCE** (also the NT-Xent loss in SimCLR). For a batch of $N$ pairs it is a cross-entropy over $N$ choices where the right answer is the matching partner.
- **Temperature** $\tau$: a number that divides the similarities before the softmax. A small $\tau$ makes the quiz sharper: the network is punished hard for small mistakes.
- **Why negatives matter:** if you only pulled partners together, the easiest solution is to map **everything to the same point**. The negatives in the denominator prevent this collapse.
- **Batch size:** a bigger batch gives more negatives per step and a harder quiz. SimCLR's authors report that contrastive learning benefits from larger batch sizes and more training steps (abstract).
- **Augmentations matter:** which views you build decides what the model learns to ignore. SimCLR's abstract says the composition of data augmentations plays a critical role.
- **Two kinds of views:** two augmented crops of one image (SimCLR); or two **different kinds of data** about the same thing, such as an image and its text ([[CLIP]]).
- **Ancestors:** [[Word2Vec]] with [[Negative Sampling]] already trained word vectors by telling real word pairs from fake ones. Contrastive Predictive Coding (CPC, 2018) made the idea general.
- **Weak spots:** negatives that are secretly the same kind of thing (two photos of cats) are treated as wrong, which adds noise; big batches need a lot of memory; and the result depends on how views are made.

```python
import numpy as np, torch, torch.nn.functional as F

def info_nce(S, tau):                              # S[i][j] = similarity of item i (view A) with item j (view B)
    S = np.asarray(S, float) / tau
    logp = S - np.log(np.exp(S).sum(1, keepdims=True))     # softmax over each row, in logs
    return float(-np.diag(logp).mean())            # the correct partner of row i is column i

S = [[0.9, 0.1], [0.2, 0.8]]
print(round(info_nce(S, 1.0), 4), round(info_nce(S, 0.1), 5), round(np.log(2) - info_nce(S, 1.0), 4))

torch.manual_seed(0)                               # two "views" of the same hidden thing, learned with no labels
Z = torch.randn(2000, 4)
WA, WB = torch.randn(4, 8), torch.randn(4, 8)
A = Z @ WA + 0.3 * torch.randn(2000, 8)
B = Z @ WB + 0.3 * torch.randn(2000, 8)
encA, encB = torch.nn.Linear(8, 16), torch.nn.Linear(8, 16)
opt = torch.optim.Adam(list(encA.parameters()) + list(encB.parameters()), lr=1e-2)

def top1(idx):                                     # does each A-embedding find its own B-embedding among 500?
    with torch.no_grad():
        s = F.normalize(encA(A[idx]), dim=1) @ F.normalize(encB(B[idx]), dim=1).T
    return (s.argmax(1) == torch.arange(len(idx))).float().mean().item()

test = torch.arange(1500, 2000)
print("before training:", round(top1(test), 3), " chance:", 1 / 500)
for step in range(300):
    i = torch.randint(0, 1500, (128,))
    s = F.normalize(encA(A[i]), dim=1) @ F.normalize(encB(B[i]), dim=1).T / 0.1
    loss = F.cross_entropy(s, torch.arange(128))   # InfoNCE = softmax cross-entropy where the answer is the diagonal
    opt.zero_grad(); loss.backward(); opt.step()
print("after training: ", round(top1(test), 3), " final loss", round(loss.item(), 3))
```

Output: `0.4043 0.00141 0.2889`, then `before training: 0.0  chance: 0.002`, then `after training:  0.564  final loss 1.62`. The first line is the loss for a tiny 2 by 2 similarity table at temperature 1 and 0.1, and the bound from the second equation below. The toy experiment has two "views" (two different noisy linear mixes) of the same hidden 4 numbers, with 2,000 items. After 300 steps with no labels, 56.4% of the held-out items find their own partner among 500, against 0.2% for random guessing.

## 🔴 4. Deep / formal
**The loss.** For a batch of $N$ pairs, let $s_{ij}$ be the cosine similarity between view A of item $i$ and view B of item $j$. The InfoNCE loss is:

$$
\mathcal L=-\frac1N\sum_{i=1}^{N}\log\frac{\exp\!\big(s_{ii}/\tau\big)}{\sum_{j=1}^{N}\exp\!\big(s_{ij}/\tau\big)}
$$

**Where:**
- $N$ is the number of pairs in the batch.
- $s_{ii}$ is the similarity of a true pair. $s_{ij}$ with $j\neq i$ is the similarity of item $i$ with a wrong partner (a negative).
- $\tau$ ("tau") is the temperature, a positive number that scales the similarities.
- The fraction is the softmax chance that item $i$ picks the right partner out of $N$ choices. The log of it is negative, so the minus sign makes the loss positive.
- **Matters:** the denominator is what supplies the negatives. Remove it and the loss can be made perfect by putting every embedding at the same point (collapse). A smaller $\tau$ makes the model more decisive, and also much more sensitive to hard negatives.

**Example (tiny numbers):** two pairs, with similarities $s_{11}=0.9,\ s_{12}=0.1,\ s_{21}=0.2,\ s_{22}=0.8$, and $\tau=1$.
- Row 1: $e^{0.9}/(e^{0.9}+e^{0.1})=2.4596/3.5648=0.690$, and $-\ln0.690=0.3711$.
- Row 2: $e^{0.8}/(e^{0.8}+e^{0.2})=2.2255/3.4473=0.646$, and $-\ln0.646=0.4375$.
- Average: $(0.3711+0.4375)/2=\mathbf{0.4043}$.
- With $\tau=0.1$ the same table gives rows $0.000335$ and $0.002476$, so the loss is $\mathbf{0.00141}$.
- Meaning: the model already ranks the true partners first, so the loss is modest at $\tau=1$ and almost 0 at $\tau=0.1$, where the same gap looks like a decisive win. (My arithmetic, matching the code.)

**A bound on mutual information.** The loss is linked to how much the two views share (their [[Mutual Information]]):

$$
I(a;b)\;\ge\;\log N-\mathcal L
$$

**Where:**
- $I(a;b)$ is the mutual information between view A and view B.
- $N$ is the number of choices in the quiz (the batch size), and $\mathcal L$ is the InfoNCE loss with natural logs.
- **Matters:** it is only a lower bound, and it can never be larger than $\log N$. So to show that two views share a lot, you need a large $N$. This is one reason bigger batches help. (The bound comes from the CPC paper; I am quoting it from memory, not checked.)

**Example (tiny numbers):** $N=2$ and $\mathcal L=0.4043$.
- $\log 2=0.6931$.
- Bound: $0.6931-0.4043=\mathbf{0.2889}$ nats.
- Meaning: the two views share at least 0.29 nats. With only 2 choices, the bound can never prove more than 0.69.

**Further notes:**
- **Other ways to avoid collapse** exist (methods that use no negatives at all, with extra tricks). They are outside this note, and I have not checked them here.
- **Link to supervised learning:** the loss is a softmax classifier where the classes are "the other items in the batch".

**What came before, and how this improved on it:**
- **Before:** supervised training on human labels (for example ImageNet's 1.28 million labelled images, a figure quoted in the CLIP abstract), plus older unsupervised tools such as [[Autoencoder]]s.
- **Its limit:** labels are slow and costly to collect, and a fixed label list only teaches the categories on the list.
- **What changed:** make the training signal from the data itself: which items belong together. Contrastive Predictive Coding (arXiv 1807.03748, July 2018) and SimCLR (arXiv 2002.05709, February 2020) turned this into a general recipe.
- **How much better:** the SimCLR abstract reports that a linear classifier trained on its self-supervised representations reaches 76.5% top-1 accuracy on ImageNet, matching a supervised ResNet-50, and 85.8% top-5 accuracy with only 1% of the labels (abstract; I read it on arXiv). In this note's own toy test, matching accuracy went from 0.0 to 0.564 with no labels (chance is 0.002).

## Related
- [[Self-Supervised Learning]] · [[Embedding]] · [[Cosine Similarity]] · [[Softmax]] · [[Cross-Entropy Loss]] · [[Negative Sampling]] · [[Word2Vec]] · [[Mutual Information]] · [[CLIP]] · [[Transfer Learning]] · [[Data Augmentation]] · [[Embedding Model]] · [[Autoencoder]] · [[Vision Transformer]]
- [[Two-Tower Model]]
## Sources
- van den Oord, Li & Vinyals, "Representation Learning with Contrastive Predictive Coding", arXiv 1807.03748, July 2018 (abstract read; the bound quoted from memory)
- Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR), arXiv 2002.05709, February 2020 (abstract read)
- Image: "A potato hater", Wikimedia Commons, CC BY-SA 4.0 (see the caption in section 2)
- Numbers in this note come from the code above (my arithmetic and my run on this machine).

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/training Contrastive learning:::Learning by pulling matching pairs of embeddings close and pushing non-matching ones apart, without human labels.

#flashcards/ml/concept #flashcards/topic/training What are a positive pair and a negative in contrastive learning?
?
A positive pair is two views of the same item (two crops of one photo, or a photo and its caption). A negative is any other item in the batch, which should not match.

#flashcards/ml/formula #flashcards/topic/training Write the InfoNCE loss and explain each symbol.
?
$$\mathcal L=-\frac1N\sum_{i=1}^{N}\log\frac{\exp\!\big(s_{ii}/\tau\big)}{\sum_{j=1}^{N}\exp\!\big(s_{ij}/\tau\big)}$$
- $N$: the number of pairs in the batch.
- $s_{ii}$: the similarity of a true pair. $s_{ij}$ with $j\ne i$: the similarity with a wrong partner (a negative).
- $\tau$: the temperature, a positive number that scales the similarities.
- The fraction is the softmax chance of picking the right partner out of $N$ choices.
- The denominator supplies the negatives. Without it the loss can be made perfect by collapsing every embedding to one point. A smaller $\tau$ is more decisive and more sensitive to hard negatives.

#flashcards/ml/example #flashcards/topic/training Similarities 0.9, 0.1 / 0.2, 0.8 with $\tau=1$: what is the InfoNCE loss, and with $\tau=0.1$?::Row 1: -ln(0.690) = 0.3711. Row 2: -ln(0.646) = 0.4375. Average 0.4043. At tau = 0.1 the rows are 0.000335 and 0.002476, so the loss is 0.00141. A lower temperature makes the same gap look decisive. My arithmetic.

#flashcards/ml/formula #flashcards/topic/training Write the InfoNCE bound on mutual information and explain each symbol.
?
$$I(a;b)\;\ge\;\log N-\mathcal L$$
- $I(a;b)$: the mutual information between view A and view B.
- $N$: the number of choices in the quiz (the batch size). $\mathcal L$: the InfoNCE loss with natural logs.
- It is only a lower bound and can never exceed $\log N$, so showing a lot of shared information needs a large $N$. One reason bigger batches help.
- The bound comes from the CPC paper; the note quotes it from memory, not checked.

#flashcards/ml/example #flashcards/topic/training With $N=2$ and loss 0.4043, what does the bound say about mutual information?::log 2 = 0.6931. Bound = 0.6931 - 0.4043 = 0.2889 nats. The two views share at least 0.29 nats, and with 2 choices the bound can never prove more than 0.69. My arithmetic.

#flashcards/ml/image #flashcards/topic/training ![[contrastive-pairs.png]]
What does this picture show?
?
Positive and negative pairs. The two kittens are a positive pair (green arrow): their embeddings should be close. The kitten-and-puppy pairs (red arrows) are negatives: their embeddings should be far apart.

#flashcards/ml/cloze #flashcards/topic/training Without ==negatives==, a contrastive model can ==collapse== by mapping every input to the same point.

#flashcards/ml/cloze #flashcards/topic/training SimCLR reports that contrastive learning benefits from ==larger batch sizes== and more training steps (abstract).

#flashcards/ml/example #flashcards/topic/training What did SimCLR's abstract report for a linear classifier on ImageNet, and with 1% of the labels?::76.5% top-1 accuracy, matching a supervised ResNet-50, and 85.8% top-5 accuracy with only 1% of the labels (abstract, read on arXiv 2002.05709).

#flashcards/ml/concept #flashcards/topic/training In the note's toy test (no labels), how well did contrastive training match partners?::Top-1 matching among 500 held-out items went from 0.0 before training to 0.564 after 300 steps. Chance is 0.002. My run on this machine, with synthetic data.

#flashcards/ml/concept #flashcards/topic/training Name two weak spots of contrastive learning.
?
Negatives that are secretly the same kind of thing (two photos of cats) are treated as wrong, which adds noise. Big batches need a lot of memory. The result also depends heavily on how the views are made.
