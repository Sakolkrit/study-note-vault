---
tags: [tokenization, LLM, NLP, BPE, transformer]
status: growing
level: 4
tier: 1
---

# Tokenization

## 🟢 1. Simple (age 12)
Before a computer can read your sentence, it **chops the text into small pieces called tokens** and gives each piece a number, like cutting a sentence into LEGO bricks.

**Example:** "unhappiness" might be cut into `un` + `happi` + `ness`. The computer has seen those bricks many times, so it can build and understand even words it has never met, just by snapping bricks together.

## 🔵 2. How it works
- A computer cannot read letters. It needs numbers.
- Keeping whole words only fails on new or rare words ("blorptastic" is not in the list).
- Using single letters always works, but the text gets very long and slow.
- The middle path: **common words stay whole, rare words split into smaller pieces**. This list of pieces is the **vocabulary**, usually 30,000 to 200,000 pieces.

Steps for "I love cats":
1. Cut: `I` · ` love` · ` cats`
2. Look up each piece's ID in the vocabulary: `[40, 3021, 11875]`
3. Feed the IDs to the model, which turns each into an [[Embedding]].
4. At the end, the model's answer (also IDs) is turned back into text.

Why you should care: models read, think and charge you in **tokens**, not words. In English, 1 token is about 4 characters or ¾ of a word. Other languages and code often need more tokens per word.

## 🟠 3. Technical detail
Common algorithms, all learned from a text corpus:
- **BPE (byte-pair encoding):** start from single characters (or bytes), repeatedly merge the most frequent adjacent pair into a new token, until the vocabulary reaches the target size. Used by GPT-style models.
- **WordPiece:** like BPE but picks merges by likelihood gain. Used by BERT.
- **Unigram / SentencePiece:** start with a big vocabulary and prune pieces that hurt likelihood least. SentencePiece also treats the space as a normal symbol, so it works without pre-splitting words (good for Chinese, Japanese, Thai).
- **Byte-level BPE:** base alphabet is the 256 bytes, so *any* text can be encoded and nothing is "unknown".

```python
# toy BPE training: one merge step
from collections import Counter
def most_common_pair(words):                      # words: list of lists of symbols
    pairs = Counter(p for w in words for p in zip(w, w[1:]))
    return pairs.most_common(1)[0][0]
def merge(words, pair):
    out = []
    for w in words:
        i, new = 0, []
        while i < len(w):
            if i < len(w) - 1 and (w[i], w[i+1]) == pair: new.append(w[i] + w[i+1]); i += 2
            else: new.append(w[i]); i += 1
        out.append(new)
    return out
words = [list("low"), list("lower"), list("lowest")]
words = merge(words, most_common_pair(words))     # ('l','o') -> 'lo'
```

Consequences you will meet:
- Models are bad at spelling, counting letters and reversing words, because they see tokens, not letters.
- Numbers split inconsistently ("1234" vs "12" "34"), which hurts arithmetic.
- Non-English text costs more tokens, so less fits in the context window and it is more expensive.
- Special tokens mark structure, e.g. start/end of text and chat roles.
- A model and its tokenizer are a matched pair. Never swap one.

## 🔴 4. Deep / formal
Vocabulary $V$, tokenizer $\tau:\Sigma^*\to V^*$ and its inverse (decode). The [[LLM]] models the token sequence, so the likelihood of a string depends on the segmentation chosen.

**BPE merge rule:** given current symbol sequences, pick

$$
(a^*,b^*)=\arg\max_{(a,b)}\ \mathrm{count}(a,b)
$$

**Where:**
- $(a,b)$ is any pair of symbols that sit next to each other in the training text (for example `l` and `o`).
- $\mathrm{count}(a,b)$ is how many times that pair appears side by side in the whole corpus. **Matters:** frequent pairs are worth gluing into one token, because they are used constantly.
- $\arg\max$ means "give me the pair that makes this count the biggest". $(a^*,b^*)$ is the winner (the star marks "the best one").

Then replace every adjacent $(a,b)$ with a new symbol $ab$. Repeat $|V|-|V_0|$ times, where $|V|$ is the target vocabulary size and $|V_0|$ is the starting alphabet size, so the loop stops exactly when the vocabulary is full. Encoding replays the merges in the order learned.

**Example (tiny numbers):** a corpus with `hug` ×10, `pug` ×5, `hugs` ×5. Start by splitting into single letters.
- Step 1, count pairs: (`u`,`g`) $=10+5+5=20$, (`h`,`u`) $=15$, (`p`,`u`) $=5$, (`g`,`s`) $=5$. Winner: (`u`,`g`), so add the token `ug`.
- Step 2, recount with `ug` as one piece: (`h`,`ug`) $=15$, (`p`,`ug`) $=5$, (`ug`,`s`) $=5$. Winner: (`h`,`ug`), so add the token `hug`.
- Result: `hug` is one token, `pug` $=$ `p`+`ug`, `hugs` $=$ `hug`+`s`.
- Common words end up whole and rarer words are built from pieces, exactly the "LEGO bricks" idea.

**Unigram LM:** each segmentation of a string is scored by

$$
P(\mathbf{x})=\prod_{i=1}^{m} p(x_i)
$$

**Where:**
- $\mathbf{x}=(x_1,\dots,x_m)$ is one way of cutting a string into $m$ pieces, and $x_i$ is the $i$-th piece.
- $p(x_i)$ is the probability of that piece on its own, learned from data. **Matters:** common pieces have high $p$, rare ones low $p$.
- $\prod$ multiplies the piece probabilities, so a segmentation of a few common pieces scores higher than one of many rare pieces. The best-scoring cut is used.

**Example (tiny numbers):** cutting "unhappy" three ways, with piece probabilities from a learned table.
- `un` + `happy`: $0.01\times0.02=\mathbf{0.0002}$.
- `un` + `hap` + `py`: $0.01\times0.005\times0.004=0.0000002$.
- seven single letters, each $0.01$: $0.01^7=10^{-14}$.
- The first cut wins by a huge margin. Because probabilities multiply, using fewer, more common pieces always scores higher.

Training maximises $\sum_{X}\log\sum_{\mathbf{x}\in S(X)}P(\mathbf{x})$ with EM ($X$ is a training string, $S(X)$ is the set of all possible cuts of it, so the inner sum adds the score of every possible cut). It then prunes tokens whose removal lowers this the least. It allows sampling alternative segmentations (**subword regularisation**) as data augmentation.

**Vocabulary size trade-off:** a larger $|V|$ gives shorter sequences (less attention cost, since cost is $O(n^2)$ in [[Attention]]) but a larger embedding and output matrix ($|V|\times d$ parameters each) and rarer, worse-trained tokens.

Open issues: glitch tokens (in the vocabulary but nearly absent from training), unfair token cost across languages, and tokenizer-free approaches that work directly on bytes (ByT5, Byte Latent Transformer) at the price of longer sequences.

**What came before, and how this improved on it:**
- **Before:** one vocabulary entry per whole word, with an "unknown" token for the rest.
- **Its limit:** new names, typos and rare words become "unknown" and lose their meaning. The code below split this vault's `20-ML-LLM` notes into two halves by file: the vocabulary of 20,276 words from one half left **15.0%** of the words in the other half unknown (the counts change a little whenever the vault grows).
- **What changed:** break words into pieces. Byte-pair encoding was applied to neural translation by Sennrich et al. in 2015 (named in the sources; I did not open it here). With bytes as the pieces, nothing is ever unknown.
- **How much better:** unknown words fall from 15.0% to 0%. The cost is length: the same text is 6.2 times longer in bytes than in words (about 6.2 bytes per word including spaces), which makes [[Attention]] more expensive. Subword vocabularies such as [[Byte-Pair Encoding]] sit between the two. My count uses crude whitespace "words", so punctuation stuck to a word makes a different word.

**Check it yourself:**

```python
import glob
fs = sorted(glob.glob(r"D:/python456/study&note stuff/20-ML-LLM/*.md"))
tr, te = fs[::2], fs[1::2]
rd = lambda L: " ".join(open(f, encoding="utf-8").read() for f in L)
a, b = rd(tr).lower().split(), rd(te).lower().split()
voc = set(a)
oov = sum(w not in voc for w in b) / len(b)
bt = len(" ".join(b).encode("utf-8"))
print("train words", len(a), " vocabulary", len(voc))
print(f"word-level: {oov:.1%} of test words unknown")
print(f"byte-level: 0% unknown, but {bt/len(b):.1f} bytes per word -> sequence {bt/len(b):.1f}x longer than word-level")
```

## Related
- [[Embedding]] · [[LLM]] · [[Transformer]] · [[Attention]] · [[Word2Vec]] · [[Pre-training]] · [[Byte-Pair Encoding]] · [[Hash Tables]] · [[Regular Expressions]] · [[BERT]] · [[Finite Automaton]] · [[Chat Template]] · [[Hugging Face Transformers]] · [[Context Window]]

## Sources
- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* (BPE for NLP), 2015
- Kudo, *Subword Regularization* (Unigram LM), 2018
- Kudo & Richardson, *SentencePiece*, 2018

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/transformers-nlp Token:::A small piece of text (a word, part of a word, or a symbol) that the computer gets by chopping a sentence, like cutting it into LEGO bricks. Each token gets a number.

#flashcards/ml/concept #flashcards/topic/transformers-nlp Why can tokenization handle words the model has never seen?::It cuts words into smaller known pieces. "unhappiness" can become `un` + `happi` + `ness`, so unseen words are built by snapping known bricks together.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write the byte-pair encoding merge rule and explain each symbol.
?
$$(a^*,b^*)=\arg\max_{(a,b)}\ \mathrm{count}(a,b)$$
- $(a,b)$: any pair of symbols that sit next to each other in the training text (for example `l` and `o`).
- $\mathrm{count}(a,b)$: how many times that pair appears side by side in the whole corpus. Frequent pairs are worth gluing into one token because they are used constantly.
- $\arg\max$: "the pair with the biggest count". $(a^*,b^*)$ is the winner, which becomes a new token. Repeat until the vocabulary is the size you want.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write the unigram tokenizer score of one cut and explain each symbol.
?
$$P(\mathbf{x})=\prod_{i=1}^{m} p(x_i)$$
- $\mathbf{x}=(x_1,\dots,x_m)$: one way of cutting a string into $m$ pieces. $x_i$: the $i$-th piece.
- $p(x_i)$: the probability of that piece on its own, learned from data. Common pieces have high $p$, rare ones low.
- $\prod$: multiply the piece probabilities. The cut with the highest product wins, so few, common pieces beat many rare ones.

#flashcards/ml/cloze #flashcards/topic/transformers-nlp Byte-pair encoding repeatedly merges the ==most frequent== pair of neighbouring symbols into one token.

#flashcards/ml/example #flashcards/topic/transformers-nlp Corpus: `hug` ×10, `pug` ×5, `hugs` ×5. Which pair does BPE merge first?::Counts: (u,g) = 10 + 5 + 5 = 20, (h,u) = 15, (p,u) = 5, (g,s) = 5. The winner is (u,g), so `ug` becomes a token.

#flashcards/ml/example #flashcards/topic/transformers-nlp Unigram scores for cutting "unhappy": `un`+`happy` (0.01 × 0.02), `un`+`hap`+`py` (0.01 × 0.005 × 0.004), and 7 single letters (0.01 each)?::0.0002, then 0.0000002, then 10⁻¹⁴. The first cut wins by a huge margin.
