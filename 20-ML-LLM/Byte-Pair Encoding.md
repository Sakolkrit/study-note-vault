---
tags: [LLM, NLP, tokenization, BPE, subword, compression]
status: growing
level: 4
tier: 2
---

# Byte-Pair Encoding

Byte-Pair Encoding (BPE) is the method most [[LLM]]s use to cut text into pieces (see [[Tokenization]]). It **starts with single letters and keeps gluing together the pairs that appear most often**, until it has a vocabulary of useful chunks.

## 🟢 1. Simple (age 12)
Imagine you are texting and you keep writing "see you later" over and over. You get tired and invent a shortcut: "cul8r". Then you notice you also write "thank you" a lot, so you invent "ty". You keep making shortcuts for the things you write most. BPE does exactly this for a computer. It looks at lots of text, finds the two letters (or chunks) that most often sit next to each other, and gives the pair its own **single symbol**. It repeats this again and again, so common words end up as one piece and rare words are spelled out from smaller pieces.

**Example:** In a pile of text about cooking, the letters "t" and "h" appear together all the time, so "th" becomes one piece. Then "th" and "e" appear together all the time, so "the" becomes one piece. A rare word like "thermometer" is built from several smaller pieces such as "ther", "mo", "meter".

## 🔵 2. How it works
**Learning the pieces (training):**
1. Start with the text split into single characters, plus a mark for "end of word".
2. **Count every pair** of neighbouring pieces across the whole text.
3. Take the **most common pair** and glue it into one new piece. Write this merge down.
4. Replace that pair everywhere in the text with the new piece. Now the text is a bit shorter.
5. Go back to step 2. Stop after a fixed number of merges, which decides the vocabulary size.
6. The result is a **vocabulary** (single characters plus all the merged pieces) and an **ordered list of merges**.

**Using the pieces (encoding):**
1. Take a new word and split it into single characters.
2. Go through the saved merges **in the same order** you learned them, and apply each one wherever it fits.
3. Whatever is left is the word's list of pieces (tokens). Each piece then gets its own [[Embedding]] vector.

No picture here: I did not find a freely licensed Wikimedia Commons image that shows BPE.

## 🟠 3. Technical detail
- **Origin:** BPE began as a text-compression trick (Gage, 1994, from memory). It was adapted to split words for machine translation (Sennrich et al., 2016, from memory) and then became the standard for GPT-style models.
- **Byte-level BPE:** instead of starting from letters, start from the 256 possible **bytes** of the text. Then **any** text in any language, even emoji, can be written, and nothing is ever "unknown". The cost is that rarer characters may take several bytes, so several tokens. In the code below, "héllo" is 5 characters but 6 bytes, because "é" takes two.
- **The vocabulary size is a choice:** it equals the base alphabet plus the number of merges. Typical LLM vocabularies are tens of thousands to a few hundred thousand pieces (from memory).
- **Trade-offs of vocabulary size:**
  - **Bigger vocabulary:** common words become single tokens, so texts are shorter (cheaper and faster for the model, see [[Attention]]), but the [[Embedding]] table and the output layer ([[Softmax]] over all tokens) get bigger, and rare tokens are seen too seldom to learn well.
  - **Smaller vocabulary:** texts get longer, but every piece is common and well learned.
- **It is a frequency method, not a language method:** BPE knows nothing about meaning or grammar. It just counts. So pieces can be odd: a chunk like "ing" is useful, but a piece may also cut a word in an unnatural place.
- **Known side effects:**
  - Words are split differently depending on **capitalisation and leading spaces**, so " the", "The" and "the" can be different tokens.
  - **Numbers** can be split in strange ways, which makes arithmetic harder for models.
  - **Languages that were rare in the training text** get cut into many small pieces, so they are costlier to process and to use (from memory; a commonly reported fairness issue).
  - Letter-level tasks, such as counting letters in a word, are hard for a model that sees pieces and not letters.
- **Cousins:** WordPiece (used by BERT, picks merges by likelihood gain), Unigram (starts big and prunes), and SentencePiece (a toolkit that works on raw text without pre-splitting into words). All from memory.
- **Training is separate:** the tokenizer is trained **before** the model, on its own pass over text (see [[Pre-training]]), and then it is frozen. Changing the tokenizer later means the model's embeddings no longer match.
- **Compression view:** each merge shortens the training text by the number of times that pair occurred, so BPE is greedily squeezing the text.

```python
from collections import Counter
# toy corpus: word -> how often it appears; each word is a tuple of symbols ending with an end-of-word mark
freq = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
words = {tuple(w) + ("_",): n for w, n in freq.items()}
def pair_counts(words):
    c = Counter()
    for w, n in words.items():
        for a, b in zip(w, w[1:]): c[(a, b)] += n
    return c
def merge(words, pair):
    out = {}
    for w, n in words.items():
        i, new = 0, []
        while i < len(w):
            if i < len(w) - 1 and (w[i], w[i + 1]) == pair: new.append(w[i] + w[i + 1]); i += 2
            else: new.append(w[i]); i += 1
        out[tuple(new)] = n
    return out
merges = []
for step in range(8):
    c = pair_counts(words); best = max(c, key=lambda p: (c[p], p))         # most frequent pair (ties: alphabetical)
    merges.append(best); words = merge(words, best)
    if step < 3: print(step, best, c[best])                                  # 0 ('t', '_') 9 | 1 ('s', 't_') 9 | 2 ('e', 'st_') 9
print([a + b for a, b in merges])      # ['t_', 'st_', 'est_', 'ow', 'low', 'west_', 'ne', 'newest_']
print({"".join(w): len(w) for w in words})                                   # {'low_': 2, 'lower_': 4, 'newest_': 1, 'widest_': 4}
def encode(word):                                                            # apply the learned merges in the same order
    w = tuple(word) + ("_",)
    for m in merges: w = merge({w: 1}, m).popitem()[0]
    return list(w)
print(encode("lowest"), encode("newer"), encode("zzz"))   # ['low', 'est_'] ['ne', 'w', 'e', 'r', '_'] ['z', 'z', 'z', '_']
print(list("héllo".encode("utf-8")), len("héllo"), len("héllo".encode("utf-8")))   # [104, 195, 169, 108, 108, 111] 5 6
print(256 + 50000, 50257 * 768, round(50257 * 768 / 1e6, 1))                 # 50256 38597376 38.6
n1, n2 = 1000, 250; print(n1 ** 2, n2 ** 2, n1 ** 2 / n2 ** 2)               # 1000000 62500 16.0
```

Reading the output. On this tiny text, the first merge is `t` + `_` (the end-of-word mark), because "newest" and "widest" both end in "t" (9 times together). The next merges build up the common ending "est_", then "low", then "newest_" as one single piece. After 8 merges, "newest" is 1 token, "low" is 2, and "lower" and "widest" are 4 each. Now the interesting part: **"lowest" never appeared in the training text**, yet it is encoded as `low` + `est_`, two pieces learned from other words. That is how BPE handles new words. "newer" gets only partial help (`ne`, `w`, `e`, `r`, `_`), because the toy text never taught it "er". "zzz" falls back to single letters (with a byte-level start, nothing could ever be unknown). The merge order matters: ties are broken alphabetically here, which is just a choice, and real tokenizers pick their own rule. The last two lines are the size arithmetic from the formal section below.

## 🔴 4. Deep / formal
**Picking a merge.** At each training step, count how often every neighbouring pair of pieces occurs in the current text, and pick the winner:

$$
(a^{*},b^{*})=\arg\max_{(a,b)}\ \operatorname{count}(a,b)
$$

**Where:**
- $a$ and $b$ are two pieces that sit next to each other in the text (single characters at first, merged chunks later).
- $\operatorname{count}(a,b)$ is how many times $a$ is directly followed by $b$ in the whole training text. For a word list, each word counts as often as the word appears.
- $\arg\max$ means "the pair with the biggest count". $(a^{*},b^{*})$ is the winner, which becomes the new merged piece $a^{*}b^{*}$.
- **Matters:** this greedy choice puts the vocabulary where the text repeats most. Without the counts, merges would be arbitrary and the pieces would not be reusable. Because it is greedy, it does not find the best possible vocabulary, only a good one.

**Example (tiny numbers):** the toy text with word counts low 5, lower 2, newest 6, widest 3.
- Top pair counts at the start: $(e,s)=9$, $(s,t)=9$, $(t,\_)=9$, then $(w,e)=8$, and $(l,o)=7$.
- Three pairs tie at 9, and the tie is broken alphabetically here, so the winner is $(t,\_)$.
- After merging it, the counts update: $(s,t\_)=9$ appears, and $(e,s)=9$ stays.
- Meaning: the winner is a pair that appears 9 times in the text. The next round starts from the updated counts, and the pair $(s,t\_)$ is the next merge, as in the code.

**What a merge saves.** Each merge replaces every occurrence of the winning pair by one piece, so the training text gets shorter by exactly the pair's count:

$$
T_{\text{new}}=T_{\text{old}}-\operatorname{count}(a^{*},b^{*})
$$

**Where:**
- $T_{\text{old}}$ and $T_{\text{new}}$ are the total number of pieces in the training text before and after the merge.
- $\operatorname{count}(a^{*},b^{*})$ is the count of the winning pair. Each occurrence of two pieces becomes one piece, so one piece disappears.
- **Matters:** it shows that BPE is a **greedy compressor**: it picks the merge that shortens the text the most right now. Merges later in the list save less and less, which is why the vocabulary size is a diminishing-returns choice.

**Example (tiny numbers):** the toy text.
- Total pieces before any merge: $T_{\text{old}}=95$ (counting each word's pieces, weighted by how often it appears).
- Winning pair $(t,\_)$ has count $9$.
- $T_{\text{new}}=95-9=\mathbf{86}$.
- Meaning: one merge shortened the text from 95 to 86 pieces. This matches the direct count after merging.

**Vocabulary and the embedding table.** The vocabulary size is the base alphabet plus the number of merges, and the embedding table has one row of $d$ numbers per piece:

$$
|\mathcal V|=|\text{base}|+M,\qquad \text{embedding parameters}=|\mathcal V|\cdot d
$$

**Where:**
- $|\mathcal V|$ is the number of distinct tokens. $|\text{base}|$ is the starting alphabet (256 for byte-level). $M$ is the number of merges you chose to do.
- $d$ is the length of each token's [[Vector]] (the embedding size).
- **Matters:** the vocabulary size is the dial for the whole trade-off: it sets the embedding table size (and the size of the output layer), and it sets how short texts become. Without the $|\mathcal V|\cdot d$ cost in mind, one might make the vocabulary huge and end up with a giant table of rarely trained rows.

**Example (tiny numbers):** a byte-level tokenizer with $M=50{,}000$ merges (about the size reported for GPT-2, plus one special token; from memory), and $d=768$.
- Vocabulary: $256+50{,}000=50{,}256$ (50,257 with the one special token).
- Embedding table: $50{,}257\times768=\mathbf{38{,}597{,}376}$ numbers, about $\mathbf{38.6}$ million.
- Meaning: just the table of token vectors holds 38.6 million numbers. Doubling the vocabulary doubles this, so it matters most for small models.

**Why shorter texts matter.** A [[Transformer]] compares every token with every other one, so the work of attention grows with the square of the number of tokens $n$:

$$
\text{attention cost}\ \propto\ n^{2}
$$

**Where:**
- $n$ is the number of tokens in the text (the sequence length). $\propto$ means "grows in proportion to".
- **Matters:** if a better tokenizer makes the same text 4 times shorter, the attention work drops by $4^2=16$ times. Without such a saving, character-level text (very long) would be far too expensive for the model. This is the main reason tokenizers merge letters into pieces at all (the rest of the model's cost grows only in proportion to $n$).

**Example (tiny numbers):** a text of 1,000 characters.
- As single characters: $n=1000$, so $n^2=1{,}000{,}000$ pairs to compare.
- With a tokenizer averaging about 4 characters per token (a typical rough figure for English; from memory): $n=250$, so $n^2=62{,}500$.
- Ratio: $1{,}000{,}000/62{,}500=\mathbf{16}$.
- Meaning: the same text costs about 16 times less attention work. This is why BPE pays off, and also why languages that tokenize badly (many small pieces per word) cost more.

**Further notes:**
- **Deterministic encoding:** applying the saved merges in order gives one fixed result for each word. Some methods instead allow several valid splits and sample among them during training (from memory).
- **Not optimal:** greedy BPE can produce a vocabulary that is not the best for a given size. Other methods optimise a likelihood directly.
- **Pre-splitting:** many tokenizers first cut text into rough word-like chunks with a rule (for example at spaces and punctuation), and run BPE inside each chunk, so merges never cross word boundaries (from memory, a common design).
- **Open questions:** whether tokenizers should be replaced by models that read bytes directly is an active research question.

**What came before, and how this improved on it:**
- **Before:** whole words (unknown ones are lost, see [[Tokenization]]) or single characters (nothing is unknown, but the sequences are long).
- **Its limit:** the first misses rare words, the second makes text many tokens long.
- **What changed:** start from characters and repeatedly merge the most frequent pair. The idea comes from a data-compression algorithm (Gage, 1994; from memory). Sennrich et al. used it for rare words in neural translation, and the abstract reports gains of 1.1 and 1.3 BLEU over a back-off dictionary on WMT 15 English-German and English-Russian (arXiv 1508.07909, 31 August 2015, abstract read).
- **How much better:** the code below runs a plain BPE on 30,000 words from this vault's notes. Characters give 5.23 tokens per word. After 100 merges it is 3.42, after 250 merges 2.89 and after 500 merges 2.45. Nothing is ever unknown. Counts are tokens per word without spaces and shift as the vault changes.

**Check it yourself:**

```python
import glob, collections
fs = sorted(glob.glob(r"D:/python456/study&note stuff/20-ML-LLM/*.md"))
words = " ".join(open(f, encoding="utf-8").read() for f in fs[::2]).lower().split()[:30000]
cnt = collections.Counter(words)
vocab = {tuple(w): c for w, c in cnt.items()}
def toks(v): return sum(len(w) * c for w, c in v.items())
print("start: tokens per word", round(toks(vocab) / len(words), 2), "(characters)")
for m in range(1, 501):
    pc = collections.Counter()
    for w, c in vocab.items():
        for a, b in zip(w, w[1:]): pc[a, b] += c
    (a, b), _ = pc.most_common(1)[0]
    nv = {}
    for w, c in vocab.items():
        o, i = [], 0
        while i < len(w):
            if i < len(w) - 1 and w[i] == a and w[i + 1] == b: o.append(a + b); i += 2
            else: o.append(w[i]); i += 1
        nv[tuple(o)] = nv.get(tuple(o), 0) + c
    vocab = nv
    if m in (100, 250, 500): print(f"after {m} merges: tokens per word {toks(vocab)/len(words):.2f}")
```

## Related
- [[Tokenization]] · [[Embedding]] · [[Embedding Model]] · [[Word2Vec]] · [[Pre-training]] · [[LLM]] · [[Transformer]] · [[Attention]] · [[Softmax]] · [[Vector]] · [[Perplexity]] · [[Entropy]] · [[Quantization]] · [[Hash Tables]]

## Sources
- Gage, *A New Algorithm for Data Compression*, The C Users Journal, 1994 (from memory)
- Sennrich, Haddow & Birch, *Neural Machine Translation of Rare Words with Subword Units*, 2016 (from memory)
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019 (from memory)
- Kudo & Richardson, *SentencePiece*, 2018 (from memory)
- Hugging Face, *Summary of the tokenizers* and the LLM course chapter on BPE (free online)
- Sennrich et al., arXiv 1508.07909 (abstract read for the history block)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/transformers-nlp Byte-pair encoding (BPE):::Find the pair of neighbouring pieces that appears most often in the text, merge it into one new piece, and repeat. Common chunks become single tokens, like a shortcut "cul8r" for "see you later".

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write the BPE merge choice and explain each symbol.
?
$$(a^{*},b^{*})=\arg\max_{(a,b)}\ \operatorname{count}(a,b)$$
- $a$, $b$: two pieces that sit next to each other in the text (single characters at first, merged chunks later).
- $\operatorname{count}(a,b)$: how many times $a$ is directly followed by $b$ in the whole training text. For a word list, each word counts as often as the word appears.
- $\arg\max$: the pair with the biggest count. $(a^*,b^*)$: the winner, which becomes the new piece $a^*b^*$.
- This greedy choice puts the vocabulary where the text repeats most.

#flashcards/ml/example #flashcards/topic/transformers-nlp Words low 5, lower 2, newest 6, widest 3. Which pair counts are highest at the start?::$(e,s)=9$, $(s,t)=9$, $(t,\_)=9$, then $(w,e)=8$ and $(l,o)=7$. Three pairs tie at 9 and the tie is broken alphabetically in the note's code, so $(t,\_)$ wins. The next merge is $(s,t\_)$.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write the number of pieces after a merge and explain each symbol.
?
$$T_{\text{new}}=T_{\text{old}}-\operatorname{count}(a^{*},b^{*})$$
- $T_{\text{old}}$, $T_{\text{new}}$: the total number of pieces in the training text before and after the merge.
- $\operatorname{count}(a^*,b^*)$: the count of the winning pair. Each occurrence of two pieces becomes one piece, so one piece disappears.
- BPE is a greedy compressor: it picks the merge that shortens the text the most right now. Later merges save less and less, so vocabulary size is a diminishing-returns choice.

#flashcards/ml/example #flashcards/topic/transformers-nlp The toy text starts with 95 pieces and the winning pair has count 9. What is $T_{\text{new}}$?::$T_{\text{new}}=95-9=86$. One merge shortened the text from 95 to 86 pieces (matches the direct count after merging).

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write the vocabulary size and embedding parameters and explain each symbol.
?
$$|\mathcal V|=|\text{base}|+M,\qquad \text{embedding parameters}=|\mathcal V|\cdot d$$
- $|\mathcal V|$: the number of distinct tokens. $|\text{base}|$: the starting alphabet (256 for byte-level). $M$: the number of merges.
- $d$: the length of each token's vector (the embedding size).
- The vocabulary size is the dial for the whole trade-off: it sets the embedding table (and output layer) size and how short texts become.

#flashcards/ml/example #flashcards/topic/transformers-nlp Byte-level, $M=50{,}000$ merges (about GPT-2's size plus one special token, from memory), $d=768$. How big is the embedding table?::Vocabulary $256+50{,}000=50{,}256$ (50,257 with the special token). Table $50{,}257\times768=38{,}597{,}376$, about 38.6 million numbers. Doubling the vocabulary doubles this.

#flashcards/ml/formula #flashcards/topic/transformers-nlp Write how attention cost depends on the number of tokens and what a shorter text saves.
?
$$\text{attention cost}\ \propto\ n^{2}$$
- $n$: the number of tokens (the sequence length). $\propto$: grows in proportion to.
- If a better tokenizer makes the same text 4 times shorter, attention work drops by $4^2=16$ times.
- Character-level text would be far too expensive. This is the main reason tokenizers merge letters into pieces. The rest of the model's cost grows only in proportion to $n$.

#flashcards/ml/example #flashcards/topic/transformers-nlp 1,000 characters as single characters against about 4 characters per token (from memory)?::$n=1000$ gives $10^6$ pairs. $n=250$ gives $62{,}500$. Ratio 16, so the same text costs about 16 times less attention work.

#flashcards/ml/cloze #flashcards/topic/transformers-nlp BPE merges the ==most frequent adjacent pair== at each step.
