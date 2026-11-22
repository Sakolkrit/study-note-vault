---
tags: [LLM, NLP, transformer, BERT, encoder, pre-training, masked-language-model]
status: growing
level: 4
tier: 3
---

# BERT

**B**idirectional **E**ncoder **R**epresentations from **T**ransformers (Devlin et al., Google, 2018). A model that reads a whole sentence at once, in both directions, and turns each word into a number list that depends on its neighbours.

## 🟢 1. Simple (age 12)
Most AI text models read **left to right** and guess the next word. BERT does something different: it reads the **whole sentence at once**, looking at words on both sides of each word. To learn, it plays a **fill-in-the-blank game** on huge amounts of text: some words are hidden, and it has to guess them from the words around them. After that practice, it is good at understanding what sentences mean, so it can be quickly taught jobs like "is this review happy or sad?".

**Example:** In "The ___ bear is reading a book", a reader who sees only the words *before* the blank has little to go on. A reader who also sees "bear is reading a book" afterwards can guess "teddy" or "cute" much better. BERT is the second kind of reader.

## 🔵 2. How it works
1. BERT is the **encoder half** of a [[Transformer]]. It has no decoder, so it does not write text. Its job is to turn text into good [[Embedding]]s, one per token.
2. The text is split into word pieces ([[Tokenization]]). The lecture says BERT's tokenizer (WordPiece) has a vocabulary of about 30,000 pieces and special tokens: **[CLS]** at the start, **[SEP]** between and after sentences, and **[MASK]** for hidden words.
3. Each token's input is the sum of three embeddings: the word piece, its place in the sentence ([[Positional Encoding]]), and which of the two sentences it belongs to (a "segment" embedding).
4. Many layers of [[Self-Attention]] follow. Because nothing is hidden by a left-to-right mask, every token can look at **all** the others, so the result is *bidirectional*.
5. **Pre-training:** BERT is trained on text without labels ([[Self-Supervised Learning]]) with two games. In **masked language modeling**, 15% of tokens are chosen and the model must predict them. In **next sentence prediction**, the model sees two sentences and says whether the second really followed the first.
6. **Fine-tuning:** to use it for a real job, add a small output layer on top of the [CLS] vector, such as one that says "positive or negative", and train a little on labelled examples ([[Fine-tuning]]). This often needs little labelled data.
7. It is excellent at **understanding** tasks (classifying, tagging words, finding answers in a passage) and not designed for **writing** text. The lecture notes that today's chat models are decoder-only, and that encoder-only models like BERT are still widely used for anything involving encoding, such as classifying and embedding text.

![[bert-masked-lm.png]]
*Masked prediction. The bottom row is the real sentence cut into tokens. One token ("##ica") is replaced by [MASK], the sentence goes through a Transformer encoder (BERT), and the model must output scores (logits) for what the hidden token was. The label comes from the sentence itself. Image: Daniel Voigt Godoy, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:BERT_masked_language_modelling_task.png), CC BY 4.0.*

## 🟠 3. Technical detail
- **Sizes (from the lecture slide, with my own counts in the code):** BERT-Base has 12 layers, a hidden size of 768 and 12 attention heads, about 110 million parameters. BERT-Large has 24 layers, size 1024 and 16 heads, about 340 million. Smaller versions (Tiny, Mini, Small, Medium) also exist.
- **Masking recipe:** of the 15% of tokens chosen, 80% become [MASK], 10% become a random word, and 10% stay unchanged. The lecture's reason is that the model must learn from context and the changes act as a regulariser. The code simulates the split.
- **No [MASK] at fine-tuning time:** the 10% random and 10% unchanged cases reduce the mismatch between pre-training (which has [MASK]) and real use (which does not). This reasoning is mine from the recipe, not checked in the paper.
- **Next sentence prediction:** half of the pairs are consecutive sentences, half are not. Later work (RoBERTa, in the lecture) found that removing it had about no effect.
- **Variants from the lecture (numbers are from the slides, not checked at the source):**
  - **DistilBERT:** 6 layers instead of 12, about 1.6 times faster, about 97% of the performance, made by [[Knowledge Distillation]].
  - **RoBERTa:** same architecture, trained more carefully: no next-sentence game, dynamic masking, 160 GB of text instead of 16 GB, bigger batches. About +4% across benchmarks according to the slide.
  - **ALBERT:** named in the lecture, not described.
- **Limits (from the lecture):** limited context length, computationally heavy for low-latency use, and a two-stage training process (pre-train then fine-tune) that is required.
- **Why it matters today:** BERT-style encoders remain a common tool for text classification, tagging and for building [[Embedding Model]]s used in [[RAG]] (the lecture's retrieval section builds on this, with the Sentence-BERT idea).
- **Date check:** the lecture says the ELMo paper was submitted in February 2018 and the BERT paper in October 2018. The arXiv page I read shows BERT's first version on 11 October 2018.

```python
import numpy as np
import torch
def count(L, H, A, V=30522, P=512, S=2, F=None):
    F = F or 4 * H
    emb = V * H + P * H + S * H + 2 * H                     # token, position, segment tables and one LayerNorm
    layer = (3 * H * H + 3 * H) + (H * H + H) + 2 * H + (H * F + F) + (F * H + H) + 2 * H     # attention in/out, LN, two feed-forward layers, LN
    pooler = H * H + H
    return emb + L * layer + pooler
# 1) the sizes in the lecture table (L layers, H width, A heads) against our count
for name, L, H, A, slide in (('Tiny', 2, 128, 2, '4M'), ('Mini', 4, 256, 4, '11M'), ('Small', 4, 512, 8, '30M'), ('Medium', 8, 512, 8, '42M'), ('Base', 12, 768, 12, '110M'), ('Large', 24, 1024, 16, '340M')):
    print(f'BERT-{name:6s} L={L:2d} H={H:4d} A={A:2d} | counted {count(L, H, A):>11,d} | slide {slide}')
# 2) one encoder layer built from PyTorch's standard block has the same number of weights as our formula
layer = torch.nn.TransformerEncoderLayer(768, 12, 3072, batch_first=True)
print('PyTorch encoder layer (H=768):', sum(p.numel() for p in layer.parameters()), '| formula:', (3 * 768 * 768 + 3 * 768) + (768 * 768 + 768) + 2 * 768 + (768 * 3072 + 3072) + (3072 * 768 + 768) + 2 * 768)
# 3) masking for the 'fill in the blank' task: 15% of tokens are chosen; of those 80% [MASK], 10% random, 10% unchanged
rng = np.random.RandomState(0)
N = 1000000
chosen = rng.rand(N) < 0.15
kind = rng.rand(N)
print('chosen:', round(chosen.mean(), 4), '| of chosen: [MASK]', round((kind[chosen] < 0.8).mean(), 3), '| random', round(((kind[chosen] >= 0.8) & (kind[chosen] < 0.9)).mean(), 3), '| unchanged', round((kind[chosen] >= 0.9).mean(), 3))
print('share of all tokens that become [MASK]:', round((chosen & (kind < 0.8)).mean(), 4), '(0.15 * 0.8 = 0.12)')
# 4) bidirectional against left-to-right: who may each of 4 tokens look at (1 = may look)
n = 4
print('encoder (BERT) mask:\n', np.ones((n, n), int))
print('decoder (GPT-style) mask:\n', np.tril(np.ones((n, n), int)))
```

## 🔴 4. Deep / formal
**Parameters of one encoder layer.** With hidden size $H$ and a feed-forward layer of width $4H$:

$$
P_{\text{layer}}=\underbrace{(3H^{2}+3H)}_{Q,K,V}+\underbrace{(H^{2}+H)}_{\text{attention out}}+\underbrace{(4H^{2}+4H)+(4H^{2}+H)}_{\text{feed-forward}}+\underbrace{2\cdot 2H}_{\text{two LayerNorms}}=12H^{2}+13H
$$

**Where:**
- $H$ is the hidden size (768 in BERT-Base).
- $3H^2+3H$ are the weights and biases that make the queries, keys and values ([[Self-Attention]]). $H^2+H$ is the layer that mixes the heads back together.
- $4H^2+4H$ and $4H^2+H$ are the two layers of the feed-forward block, which widen to $4H$ and come back.
- $2\cdot2H$ are the scale and shift numbers of the two normalisations ([[Normalization]]).
- **Matters:** nearly all the weights are the $12H^2$ part, so doubling the width makes a layer four times as big. The $13H$ is a small extra. A whole model adds $L$ such layers, plus the embedding tables (vocabulary $\times H$, positions $\times H$, segments $\times H$) and a small output layer.

**Example (tiny numbers):** $H=768$.
- $12H^2=12\times589{,}824=\mathbf{7{,}077{,}888}$.
- $13H=\mathbf{9{,}984}$.
- Sum: $\mathbf{7{,}087{,}872}$ per layer. PyTorch's standard encoder layer of this size has exactly $7{,}087{,}872$ parameters (the code counts them).
- Meaning: 12 layers hold $85{,}054{,}464$ weights. The embedding tables (using a vocabulary of $30{,}522$, from memory) add $23{,}837{,}184$ and the output layer $590{,}592$, so the total is $109{,}482{,}240$, the "110 million" of the slide.

**The masking recipe.** The share of all tokens that are replaced by [MASK]:

$$
\underbrace{0.15}_{\text{chosen}}\times\underbrace{0.80}_{\text{become [MASK]}}=0.12
$$

**Where:**
- $0.15$ is the share of tokens chosen to be predicted. $0.80$ is the share of those chosen that actually show as [MASK].
- The remaining $0.10$ and $0.10$ of the chosen tokens are replaced by a random word or left as they are, but the model must still predict them.
- **Matters:** only 12% of all tokens show a [MASK], but 15% are scored. Each sentence therefore teaches the model on only a small share of its tokens (my own reading: one reason pre-training needs a lot of text). The lecture also credits the random and unchanged cases with acting as a regulariser.

**Example (tiny numbers):** a text of 1,000 tokens.
- Chosen: $1000\times0.15=\mathbf{150}$ tokens.
- Shown as [MASK]: $150\times0.8=\mathbf{120}$; random: $\mathbf{15}$; unchanged: $\mathbf{15}$.
- Meaning: the model is graded on 150 tokens but only sees 120 blanks. The code's simulation over a million tokens gave $0.1501$ chosen and $0.1203$ blanks.

**Further notes:**
- **The mask picture (code):** an encoder lets every token look at every other (all ones). A decoder only lets a token look at itself and the ones before it (lower triangle of ones). This is the whole difference between BERT-style and GPT-style models in the attention step.
- **Counts differ a little from the slide for the small models:** my count for BERT-Small is $28{,}763{,}648$ against the slide's 30M, and for BERT-Tiny $4{,}385{,}920$ against 4M. The slide rounds, and I did not look up how the original authors counted, so I cannot say which is "right". BERT-Large came out at $335{,}141{,}888$ against 340M.
- **Not covered:** how the output layer predicts a masked word, the exact fine-tuning recipes, and later encoder models.

**What came before, and how this improved on it:**
- **Before:** static word embeddings such as [[Word2Vec]] (one fixed vector per word, whatever the sentence), and recurrent models ([[Recurrent Neural Network]]) that read in one direction, with ELMo (Peters et al., submitted February 2018 per the lecture slide) already giving context-dependent vectors by combining models that read each way (from memory; I did not read it).
- **Its limit:** one vector per word cannot tell a river *bank* from a money *bank*, and a model that only looks left (or two separate one-direction models joined) does not condition on both sides in every layer.
- **What changed:** a deep Transformer encoder pre-trained with the fill-in-the-blank game so that every layer sees both sides, then fine-tuned with one extra output layer (Devlin et al., arXiv 1810.04805, 11 October 2018, abstract read).
- **How much better:** the abstract reports new best results on eleven tasks, including GLUE 80.5% (a gain of 7.7 points), MultiNLI 86.7% (+4.6), SQuAD v1.1 test F1 93.2 (+1.5) and SQuAD v2.0 test F1 83.1 (+5.1). I did not reproduce any of them. In this note's code: a layer of BERT-Base has $7{,}087{,}872$ parameters, and the full model counts $109{,}482{,}240$, within the slide's "110M".

## Related
- [[Transformer]] · [[Self-Attention]] · [[Attention]] · [[Embedding]] · [[Embedding Model]] · [[Tokenization]] · [[Positional Encoding]] · [[Pre-training]] · [[Fine-tuning]] · [[Self-Supervised Learning]] · [[Knowledge Distillation]] · [[Word2Vec]] · [[Recurrent Neural Network]] · [[LLM]] · [[RAG]] · [[Normalization]] · [[Transfer Learning]] · [[Layer Normalization]] · [[Vision Transformer]] · [[Masked Diffusion Language Model]] · [[BERT 2018]] · [[Feed-Forward Network]]

## Sources
- Stanford CME 295 (Fall 2025), Lecture 2 slides, Afshine and Shervine Amidi
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, 2018 (arXiv 1810.04805, abstract read)
- Sanh et al., *DistilBERT*, 2019, and Liu et al., *RoBERTa*, 2019 (named in the lecture; numbers taken from the slides, not read)
- Peters et al., *Deep contextualized word representations* (ELMo), 2018 (named in the lecture; not read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/transformers-nlp BERT:::A model that reads the whole sentence at once, looking at words on both sides of each word. It learns by a fill-in-the-blank game: some words are hidden and it must guess them.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write the parameters of one BERT encoder layer and explain each term.
?
$$P_{\text{layer}}=\underbrace{(3H^{2}+3H)}_{Q,K,V}+\underbrace{(H^{2}+H)}_{\text{attention out}}+\underbrace{(4H^{2}+4H)+(4H^{2}+H)}_{\text{feed-forward}}+\underbrace{2\cdot 2H}_{\text{two LayerNorms}}=12H^{2}+13H$$
- $H$: the hidden size (768 in BERT-Base).
- $3H^2+3H$: weights and biases that make the queries, keys and values. $H^2+H$: the layer that mixes the heads back together.
- $4H^2+4H$ and $4H^2+H$: the two layers of the feed-forward block, which widen to $4H$ and come back.
- $2\cdot2H$: the scale and shift numbers of the two normalisations.
- Nearly all weights are the $12H^2$ part, so doubling the width makes a layer four times as big. The $13H$ is a small extra.

#flashcards/ml/example #flashcards/topic/transformers-nlp $H=768$. How many parameters per layer, for 12 layers, and for the whole BERT-Base?::$12H^2=7{,}077{,}888$ and $13H=9{,}984$, so $7{,}087{,}872$ per layer (equal to PyTorch's standard encoder layer of this size, per the note's code). Twelve layers: $85{,}054{,}464$. Embedding tables (vocabulary 30,522, from memory) add $23{,}837{,}184$ and the output layer $590{,}592$. Total $109{,}482{,}240$, the "110 million" of the slide.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write the share of tokens that BERT shows as [MASK] and explain each factor.
?
$$\underbrace{0.15}_{\text{chosen}}\times\underbrace{0.80}_{\text{become [MASK]}}=0.12$$
- $0.15$: the share of tokens chosen to be predicted.
- $0.80$: the share of those chosen that actually show as [MASK].
- The remaining 0.10 and 0.10 of the chosen tokens are replaced by a random word or left as they are, but the model must still predict them. The lecture credits these cases with acting as a regulariser.
- Only 12% of tokens show a [MASK] but 15% are scored, so each sentence teaches on a small share of its tokens (the note's own reading: one reason pre-training needs lots of text).

#flashcards/ml/example #flashcards/topic/transformers-nlp A text of 1,000 tokens. How many are chosen, masked, random and unchanged?::Chosen $1000\times0.15=150$. [MASK]: $150\times0.8=120$. Random: 15. Unchanged: 15. The model is graded on 150 tokens but sees only 120 blanks. The note's simulation over a million tokens gave 0.1501 chosen and 0.1203 blanks.

#flashcards/ml/image #flashcards/topic/transformers-nlp ![[bert-masked-lm.png]]
What does this picture show?
?
Masked prediction. The bottom row is the real sentence cut into tokens. One token ("##ica") is replaced by [MASK], the sentence goes through a Transformer encoder (BERT), and the model must output scores (logits) for what the hidden token was. The label comes from the sentence itself. (From the note's caption: Daniel Voigt Godoy, Wikimedia Commons.)

#flashcards/ml/cloze #flashcards/topic/transformers-nlp BERT reads ==both sides== of a word, while most text models read left to right.
