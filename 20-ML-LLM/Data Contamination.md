---
tags: [data-contamination, benchmarks, evaluation, LLM, training-data, n-gram, ML]
status: growing
level: 4
tier: 2
---

# Data Contamination

**Data contamination** (also called test-set leakage) is when the questions of a benchmark, or their answers, were already in a model's training text. The model may then score high because it remembers them, not because it can solve new ones. The score is no longer a fair test of the skill it claims to measure.

## 🟢 1. Simple (age 12)
Imagine a spelling test. If you saw the exact list of words and the answers the night before, you could get 100% without knowing how to spell. The teacher cannot tell the difference from the score alone. A big model reads a huge part of the internet when it learns, and the internet may contain the test questions, so a high score might be a good memory and not real skill.

**Example:** two students both score 95. One learned spelling rules. The other found the answer sheet. A new list of words shows which is which: the first still gets about 95, the second drops a lot.

## 🔵 2. How it works
1. **A benchmark is public.** Its questions are on the internet so everyone can use them ([[Benchmarks]]).
2. **A model is trained on a huge copy of the internet** ([[Pre-training]]). The test questions, with answers, may be in it: on a website, a forum, a GitHub file, or a paper.
3. **Seen text is easier.** A model gives higher probability to text it has seen ([[Log-Likelihood]]), so a seen question and answer can score well by memory.
4. **How people look for it.** They cut the test question into overlapping runs of words (**n-grams**), and check whether any of those runs appear in the training text. A hit marks the question as possibly contaminated.
5. **How people measure the damage.** They score the model twice: on all questions, and on the questions with no overlap. A big gap suggests memory.
6. **The catch.** You need the training text for this. For many models it is secret. Then people only compare scores on newer or private questions, or on rewritten versions.

No image: I found no free picture of this that shows a real check, and a made-up diagram would not help. The numbers below come from a real run instead.

## 🟠 3. Technical detail
Everything below comes from `contam_demo.py` (Python 3.9.6, `datasets`; ARC-Easy from `allenai/ai2_arc`, CC BY-SA 4.0 per its dataset card). It is a toy check. The "training pile" here is ARC-Easy's own train and validation questions, not a real model's training text.

```python
"""Data contamination demo: find test questions that also appear in a pile of 'training text' by n-gram overlap."""
import random, re
from datasets import load_dataset

def grams(text, n):
    w = re.findall(r"\w+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}

def flagged(test_texts, corpus_grams, n):
    """a test text is flagged if any of its n-grams occurs in the corpus"""
    return [i for i, t in enumerate(test_texts) if grams(t, n) & corpus_grams]

# 1. Real check: ARC-Easy test questions against ARC-Easy train + validation questions
arc = load_dataset("allenai/ai2_arc", "ARC-Easy")
test = [d["question"] for d in arc["test"]]
pile = [d["question"] for d in arc["train"]] + [d["question"] for d in arc["validation"]]
print("ARC-Easy test", len(test), "pile (train+validation)", len(pile))
exact = set(pile)
print("exact same question text:", sum(t in exact for t in test))
for n in (13, 8):
    g = set().union(*(grams(t, n) for t in pile))
    print("%d-gram overlap:" % n, len(flagged(test, g, n)))

# 2. Planted leak: put 50 random test questions inside a bigger pile and see if the check finds them
random.seed(0)
leaked = random.sample(range(len(test)), 50)
pile2 = pile + [test[i] for i in leaked]
for n in (13, 8):
    g = set().union(*(grams(t, n) for t in pile2))
    hit = flagged(test, g, n)
    print("planted 50, %d-gram: flagged %d, of which planted %d" % (n, len(hit), len(set(hit) & set(leaked))))

# 3. How many questions are too short to have any n-gram at all
lens = [len(re.findall(r"\w+", t)) for t in test]
for n in (13, 8):
    print("test questions with fewer than %d words: %d of %d" % (n, sum(l < n for l in lens), len(lens)))

# 4. A rewritten leak (one word in the middle changed) on a long question
long_i = next(i for i, l in enumerate(lens) if l >= 30)
q = test[long_i]; w = q.split(); w[len(w) // 2] = "XXXX"; changed = " ".join(w)
print("question length in words", len(w), "| 13-gram finds the rewritten copy:", bool(grams(changed, 13) & grams(q, 13)),
      "| 8-gram:", bool(grams(changed, 8) & grams(q, 8)))
```

I saw (the dataset-loading lines left out):
```
ARC-Easy test 2376 pile (train+validation) 2821
exact same question text: 7
13-gram overlap: 13
8-gram overlap: 61
planted 50, 13-gram: flagged 42, of which planted 29
planted 50, 8-gram: flagged 107, of which planted 47
test questions with fewer than 13 words: 812 of 2376
test questions with fewer than 8 words: 167 of 2376
question length in words 41 | 13-gram finds the rewritten copy: True | 8-gram: True
```

**What the check found:**

| Check | Result | What it means |
|---|---|---|
| Test questions that are word-for-word in train or validation | 7 of 2,376 | A few questions repeat within the dataset itself. |
| Test questions sharing a 13-word run with the pile | 13 | The stricter test. |
| Test questions sharing an 8-word run with the pile | 61 | The looser test: more hits, but also more chance matches. |
| Planted 50 copies, 13-word check: flagged / planted found | 42 / 29 | It missed 21 of the 50 planted copies. |
| Planted 50 copies, 8-word check: flagged / planted found | 107 / 47 | It found 47 of 50, but also flagged 60 other questions. |
| Test questions with fewer than 13 words | 812 of 2,376 | They have no 13-word run at all. |
| Test questions with fewer than 8 words | 167 of 2,376 | They have no 8-word run at all. |
| One word changed in the middle of a 41-word question | both checks still found it | A long question survives a small rewrite. |

- **Check:** which way of searching I used and for what. "Planted" means I copied 50 random test questions into the pile myself, so I know the right answer.
- **Result:** counts of test questions (my run).
- **What it means:** my reading.

**Summary: what the table shows.** The check works, but it has a size knob and both settings fail in different ways. The 13-word rule is precise, yet missed 21 of 50 planted copies because 812 questions (34%, my arithmetic) are too short to contain a 13-word run. The 8-word rule found 47 of 50, but its hit count (107) includes about 60 questions it flagged for other reasons. Caution: the extra "other" hits are the pile's own overlap (61 hits before I planted anything), so some are harmless shared phrases like "which of the following" and I did not read them one by one. This is a pile of questions, not a model's training text, and I did not train anything on the leaks, so I did **not** measure how much a leak raises a model's score.

## 🔴 4. Deep / formal
No equation is needed here: the check is a set overlap, and the damage is a difference of two scores.

**Detecting contamination, in the usual order of cost:**
- **Exact match of whole questions.** Cheap, catches only identical text.
- **N-gram overlap.** Cut each test item into runs of $n$ words and ask whether any run is in the training text. Longer runs give fewer false alarms and miss more; shorter runs do the opposite, as in the table above. The GPT-3 paper used an n-gram overlap filter; I recall a 13-word window but my [[GPT-3 2020]] note does not state it, so treat 13 as from memory.
- **Clean-subset comparison.** Score the model on all items and on the items with no overlap; report both ([[GPT-3 2020]]: most benchmarks barely moved, PIQA and Winograd were marked with an asterisk).
- **Model-based checks without the training text.** Ask whether the model can complete an item from its first half, or whether it prefers the benchmark's original order of answer options over a shuffled one. I did not run these and describe them from general knowledge.

**What the published example shows (GPT-3 paper, as read in [[GPT-3 2020]]):** a bug in the filter left some overlaps; the paper then built clean versions of each benchmark. PIQA had 29% of items flagged and a 3-point drop on the clean subset; Winograd 45% flagged and a 2.6% drop; the Wikipedia language-modelling sets and the Children's Book Test were almost entirely in the training data and were not reported. A limit the paper itself names: the clean subset can be easier or harder than the full set.

**Contamination is not only exact copies.** Rewritten, translated or paraphrased test items, test answers in a discussion thread, or a benchmark's training split used for a model that is later tested on the test split of the same data can all raise a score. N-gram checks miss paraphrases.

**Limits and trade-offs:**
- Without the training text you cannot check. Many models do not publish it.
- A score drop on a clean subset might come from a harder subset, not from lost memory.
- A leak of the question without the answer helps less than a leak of both; one check cannot tell.
- Deliberate training on a benchmark ("benchmark gaming") is a separate problem from accidental leakage, and no overlap check proves a model was not tuned on the test (general knowledge).
- In my [[LLM Evaluation Lab]] runs I did not check contamination for Qwen2.5, because its training text is not public (general knowledge, not checked). So the ARC-Easy and HellaSwag scores there carry that unknown.

**What came before, and how this improved on it:**
- **Before:** treat a public benchmark score as a clean measure of skill.
- **Its limit:** once the test is public and the training text is the internet, a high score can be memory. In the GPT-3 paper, a bug left test overlap in the training data, so the authors had to measure it afterwards.
- **What changed:** report overlap checks and scores on a clean subset next to the headline number; newer benchmarks keep their answers private or add fresh questions over time (general knowledge, not checked).
- **How much better:** the published figure is that for most GPT-3 benchmarks the clean-subset score changed little, while PIQA fell 3 points on the clean part (paper, Section 4, as read in [[GPT-3 2020]]). In my own toy run, the strict 13-word check found only 29 of 50 planted copies and the looser 8-word check found 47 of 50. This was not compared on a real model.

## Related
[[Benchmarks]] · [[Evaluation Metrics]] · [[Log-Likelihood]] · [[Few-Shot Prompting]] · [[Pre-training]] · [[Overfitting]] · [[GPT-3 2020]] · [[LLM Evaluation Lab]] · [[LLM]] · [[Standard Error]]

## Sources
- My own run: `contam_demo.py` (Python 3.9.6; ARC-Easy from `allenai/ai2_arc`, CC BY-SA 4.0 per its dataset card).
- GPT-3 paper, Section 4 (contamination), as read in my note [[GPT-3 2020]]; I did not re-read the paper for this note.
- The model-based checks, benchmark gaming and newer private benchmarks are from general knowledge and not checked. The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation Data contamination:::When a benchmark's questions or answers were already in a model's training text, so a high score may be memory and not skill (also called test-set leakage).

#flashcards/ml/concept #flashcards/topic/evaluation Why do public benchmarks get contaminated?::The test is public so everyone can use it, and a model is trained on a huge copy of the internet that can contain the questions and their answers (on websites, forums, code files or papers).

#flashcards/ml/concept #flashcards/topic/evaluation How does an n-gram overlap check work?::Cut each test question into overlapping runs of n words and check whether any run occurs in the training text; a hit marks the question as possibly contaminated. GPT-3's filter was of this kind (window size from memory: 13 words).

#flashcards/ml/concept #flashcards/topic/evaluation How do you measure how much contamination changes a score?::Score the model on all items and on the items with no overlap (the clean subset) and compare. A big gap suggests memory, but the clean subset may simply be harder or easier.

#flashcards/ml/example #flashcards/topic/evaluation In my toy check I planted 50 test questions in a pile. How many did the 13-word and the 8-word check find?::13-word: 29 of 50 (it also flagged 13 others overall). 8-word: 47 of 50, but it flagged 107 questions in all. Toy pile of ARC-Easy train and validation, not a real model's training text.

#flashcards/ml/concept #flashcards/topic/evaluation Why did the 13-word check miss planted copies?::812 of the 2,376 ARC-Easy test questions have fewer than 13 words, so they contain no 13-word run to match (my count). Short items need a shorter window or an exact match.

#flashcards/ml/concept #flashcards/topic/evaluation What is the trade-off between a long and a short n-gram window?::A long window gives fewer false alarms and misses more copies and rewrites; a short window catches more but flags shared phrases that are harmless (8-word: 61 hits before planting anything, my run).

#flashcards/ml/concept #flashcards/topic/evaluation What did the GPT-3 paper find when it re-scored on clean subsets?::For most benchmarks the change was small. PIQA had 29% flagged and a 3-point drop; Winograd 45% flagged and a 2.6% drop. Some Wikipedia language-modelling sets and the Children's Book Test were almost fully in the training data and were not reported. A bug in its filter had left some overlap.

#flashcards/ml/concept #flashcards/topic/evaluation What cannot an n-gram overlap check catch?::Paraphrased or translated test items, answers that leaked in a different wording, and any case where you do not have the training text. It also cannot prove a model was not tuned on the test.

#flashcards/ml/concept #flashcards/topic/evaluation Did my toy check measure how much a leak raises a model's score?::No. I only tested how well the overlap search finds planted copies. I did not train any model on the leaks.

#flashcards/ml/cloze #flashcards/topic/evaluation Data contamination means the ==test questions== were already in the model's ==training text==, so a high score may be ==memory==.
