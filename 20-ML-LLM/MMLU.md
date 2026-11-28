---
tags: [mmlu, benchmark, evaluation, LLM, multiple-choice, knowledge, ML]
status: growing
level: 4
tier: 2
---

# MMLU

**MMLU** (Massive Multitask Language Understanding) is a benchmark of multiple-choice questions on 57 subjects, from elementary mathematics and US history to law and computer science. Each question has four options, and the score is the share answered correctly. It is the most widely quoted single number for "how much does this language model know".

## 🟢 1. Simple (age 12)
Imagine one giant school exam that mixes 57 subjects: maths, history, law, medicine, computer science and more, all as "pick A, B, C or D". A person who studied everything might get most right. Someone guessing gets about a quarter. MMLU gives a language model that exam and reports the percentage it gets right. It tests broad knowledge, not just one skill.

**Example:** a question like "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q" with four numbers to choose from. A model that knows the maths picks the right one, and a model that guesses is right one time in four.

## 🔵 2. How it works
1. The benchmark has **57 subjects**, with 100 to 1,534 test questions each (14,042 in all). Every question has four options.
2. A **prompt** is built for each question: a header naming the subject, the question, the four options on separate lines starting with A to D, and "Answer:".
3. Optionally, a few **solved examples** from the same subject (the "dev" questions) are put in front of it ([[Few-Shot Prompting]]). The original paper used five.
4. The model picks a letter. Usually this is done by comparing how likely the model finds each of the four letters after "Answer:" ([[Log-Likelihood]]), so the model never has to "write" an answer.
5. The score is **accuracy**: correct answers divided by questions, either over all questions together or averaged over the 57 subjects. These two numbers differ because the subjects have very different sizes.
6. A random guesser gets about 25%.
7. **Read the number with care:** the wording of the prompt, the number of examples, the way the score is averaged, and whether the questions leaked into the training text ([[Data Contamination]]) all change it.

No image: I found no free picture of MMLU, and the paper's figures are not free. The numbers below come from real runs instead.

## 🟠 3. Technical detail
Everything below comes from three scripts (Python 3.9.6, PyTorch 2.8.0, transformers 4.46.3, `datasets`, CPU). The data is `cais/mmlu`, config `all` (MIT licence per its dataset card). The models are `Qwen/Qwen2.5-0.5B-Instruct` and `Qwen/Qwen2.5-1.5B-Instruct` (Apache-2.0).

**Script 1, `mmlu_stats.py`: what is in the data and what a prompt looks like.**

```python
"""MMLU (cais/mmlu, 'all' config): sizes, subjects, answer positions, and one prompt in the usual 0-shot format."""
from collections import Counter
from datasets import load_dataset

ds = load_dataset("cais/mmlu", "all")
for split in ("test", "validation", "dev", "auxiliary_train"):
    print(split, len(ds[split]))
test = ds["test"]
subj = Counter(test["subject"])
print("subjects in test:", len(subj), "| fewest questions:", min(subj.values()), "| most:", max(subj.values()))
print("options per question:", Counter(len(c) for c in test["choices"]))
print("gold letter shares:", {"ABCD"[k]: round(v / len(test), 3) for k, v in sorted(Counter(test["answer"]).items())})
d = test[0]
subject = d["subject"].replace("_", " ")
prompt = "The following are multiple choice questions (with answers) about %s.\n\n%s\n%s\nAnswer:" % (
    subject, d["question"], "\n".join("%s. %s" % ("ABCD"[i], c) for i, c in enumerate(d["choices"])))
print("-" * 20); print(prompt); print("gold:", "ABCD"[d["answer"]])
```

**Script 2, `mmlu_eval.py`: 0-shot scoring of the first 5 test questions of every subject (285 questions).**

```python
"""MMLU 0-shot on the first N test questions of every subject; the option is scored as the letter after 'Answer:'."""
import json, math, sys, time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(8)
name, per_subject = sys.argv[1], int(sys.argv[2])
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).eval()
test = load_dataset("cais/mmlu", "all", split="test")

seen, rows, t0 = {}, [], time.time()
for d in test:
    s = d["subject"]
    if seen.get(s, 0) >= per_subject:
        continue
    seen[s] = seen.get(s, 0) + 1
    prompt = "The following are multiple choice questions (with answers) about %s.\n\n%s\n%s\nAnswer:" % (
        s.replace("_", " "), d["question"], "\n".join("%s. %s" % ("ABCD"[i], c) for i, c in enumerate(d["choices"])))
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        logp = torch.log_softmax(model(ids).logits[0, -1].float(), dim=-1)   # next token after "Answer:"
    ll = [logp[tok(" " + L).input_ids[-1]].item() for L in "ABCD"]            # the letter with its leading space
    rows.append({"subject": s, "gold": d["answer"], "pred": max(range(4), key=ll.__getitem__), "ll": ll})
    if len(rows) % 50 == 0:
        print(len(rows), "done", round(time.time() - t0), "s", flush=True)
n = len(rows); p = sum(r["pred"] == r["gold"] for r in rows) / n
res = {"model": name, "n": n, "acc": round(p, 4), "se": round(math.sqrt(p * (1 - p) / (n - 1)), 4), "seconds": round(time.time() - t0)}
print(json.dumps(res))
json.dump({"result": res, "rows": rows}, open("mmlu_%s_%d.json" % (name.split("/")[-1], per_subject), "w"))
```

**Script 3, `mmlu_compare.py`: scores, the paired comparison and the letters chosen.**

```python
"""MMLU subset: scores with standard errors, the paired comparison of the two models, and the position of the gold letter."""
import json, math
from collections import Counter
S, B = "Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct"
load = lambda m: json.load(open("mmlu_%s_5.json" % m))["rows"]
a, b = load(S), load(B)
n = len(a)
for name, rows in ((S, a), (B, b)):
    p = sum(r["pred"] == r["gold"] for r in rows) / n
    print(name, "n", n, "acc %.3f" % p, "s.e. %.3f" % math.sqrt(p * (1 - p) / (n - 1)),
          "| predicted letters", sorted(Counter("ABCD"[r["pred"]] for r in rows).items()))
only_b = sum(x["pred"] != x["gold"] and y["pred"] == y["gold"] for x, y in zip(a, b))
only_s = sum(x["pred"] == x["gold"] and y["pred"] != y["gold"] for x, y in zip(a, b))
both = sum(x["pred"] == x["gold"] and y["pred"] == y["gold"] for x, y in zip(a, b))
d = (only_b - only_s) / n
se = math.sqrt(only_b + only_s - (only_b - only_s) ** 2 / n) / n
print("both", both, "only 1.5B", only_b, "only 0.5B", only_s, "neither", n - both - only_b - only_s,
      "difference %.3f paired s.e. %.3f (%.1f s.e.)" % (d, se, d / se))
print("gold letters in this subset:", sorted(Counter("ABCD"[r["gold"]] for r in a).items()))
# five questions per subject: how many subjects each model gets at least 3 of 5
for name, rows in ((S, a), (B, b)):
    by = {}
    for r in rows:
        by.setdefault(r["subject"], []).append(r["pred"] == r["gold"])
    print(name, "subjects with >= 3 of 5 right:", sum(sum(v) >= 3 for v in by.values()), "of", len(by), "| with 0 right:", sum(sum(v) == 0 for v in by.values()))
```

I saw (script 1, then the two result lines of script 2, then script 3):
```
test 14042
validation 1531
dev 285
auxiliary_train 99842
subjects in test: 57 | fewest questions: 100 | most: 1534
options per question: Counter({4: 14042})
gold letter shares: {'A': 0.229, 'B': 0.247, 'C': 0.255, 'D': 0.269}
--------------------
The following are multiple choice questions (with answers) about abstract algebra.

Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.
A. 0
B. 4
C. 2
D. 6
Answer:
gold: B

{"model": "Qwen/Qwen2.5-0.5B-Instruct", "n": 285, "acc": 0.4632, "se": 0.0296, "seconds": 548}
{"model": "Qwen/Qwen2.5-1.5B-Instruct", "n": 285, "acc": 0.6386, "se": 0.0285, "seconds": 1268}

Qwen2.5-0.5B-Instruct n 285 acc 0.463 s.e. 0.030 | predicted letters [('A', 105), ('B', 71), ('C', 68), ('D', 41)]
Qwen2.5-1.5B-Instruct n 285 acc 0.639 s.e. 0.029 | predicted letters [('A', 76), ('B', 73), ('C', 62), ('D', 74)]
both 108 only 1.5B 74 only 0.5B 24 neither 79 difference 0.175 paired s.e. 0.033 (5.3 s.e.)
gold letters in this subset: [('A', 77), ('B', 75), ('C', 67), ('D', 66)]
Qwen2.5-0.5B-Instruct subjects with >= 3 of 5 right: 24 of 57 | with 0 right: 0
Qwen2.5-1.5B-Instruct subjects with >= 3 of 5 right: 41 of 57 | with 0 right: 1
```

**The data (script 1).**

| Part | Questions | What it is for |
|---|---|---|
| test | 14,042 | the scored questions |
| validation | 1,531 | for choosing settings |
| dev | 285 | 5 solved examples per subject, used as few-shot examples |
| auxiliary_train | 99,842 | extra training data in the same format, from other datasets (my understanding from the dataset card name; I did not check where it comes from) |

- **Part:** the split of the dataset. **Questions:** how many rows.
- The test split has 57 subjects, from 100 to 1,534 questions each, and every question has 4 options. The correct letter is not evenly spread: A 22.9%, B 24.7%, C 25.5%, D 26.9% of the test questions.

**My scores on the first 5 questions per subject (0-shot, my own scoring of the standard format).**

| Model | Correct of 285 | Accuracy | Standard error | Subjects with at least 3 of 5 right |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 132 | 0.463 | 0.030 | 24 of 57 |
| Qwen2.5-1.5B-Instruct | 182 | 0.639 | 0.029 | 41 of 57 |

- **Correct of 285:** how many of my 285 questions it got right (132 = 108 + 24 and 182 = 108 + 74 from the pairing below, my arithmetic).
- **Accuracy:** correct divided by 285. Chance is 0.25.
- **Standard error:** how much the score could move by luck on a sample this size ([[Standard Error]]).
- **Subjects with at least 3 of 5:** a rough look at breadth; with only 5 questions per subject, one subject's result is almost meaningless.

**Paired comparison on the same 285 questions.**

| Both right | Only 1.5B right | Only 0.5B right | Neither | Difference | Paired standard error |
|---|---|---|---|---|---|
| 108 | 74 | 24 | 79 | +0.175 | 0.033 |

The gain is 5.3 paired standard errors, so it is not luck on these questions.

**Letters chosen.** The 0.5B model answered A for 105 of 285 questions (37%), though the correct letter was A for 77 (27%). The 1.5B model spread its answers more evenly (A 76, B 73, C 62, D 74).

**Summary: what the runs show.** Both models are well above chance (0.25), and the bigger model is clearly better on this sample (0.639 against 0.463). The small model leans towards "A", which is the kind of habit a multiple-choice score can hide. Caution: 285 questions are 2% of the test set and are the first five of each subject, not a random sample; the score has an uncertainty of about ±0.03 and the subjects count equally here, so this is not the official MMLU number. I scored 0-shot, the original paper used five examples, and I wrote my own scoring and prompt from the standard format: I did not run it through the official harness, so it may differ slightly from published numbers. Also, the "Instruct" versions of these models are not the ones whose scores companies usually quote (general knowledge, not checked).

## 🔴 4. Deep / formal
No equation: the score is a count of correct answers divided by the number of questions, and its uncertainty is covered in [[Standard Error]].

**What the paper says (arXiv 2009.03300, abstract only; I did not read the paper):** the test covers 57 tasks including elementary mathematics, US history, computer science and law; most models then had near-chance accuracy, and "the very largest GPT-3 model improves over random chance by almost 20 percentage points on average"; on every one of the 57 tasks the best models still needed substantial improvements to reach expert-level accuracy; models were lopsided and often did not know when they were wrong. Authors: Hendrycks, Burns, Basart, Zou, Mazeika, Song and Steinhardt; first submitted 7 September 2020 (arXiv listing).

**Micro and macro averages.** Pooling all 14,042 questions weights big subjects heavily (the largest has 1,534 questions, the smallest 100). Averaging the 57 subject scores gives each subject the same weight. Publications differ in which they report, so numbers are not always comparable (general knowledge; I did not compute both on the full set).

**What affects a MMLU number:**
- **Prompt wording and the answer format** (letters or full option text).
- **Number of examples** (0, 5, ...).
- **Whether the model is scored on the letter after "Answer:" or by generating text.**
- **Option order:** a model with a habit for one letter, as the 0.5B model showed here, can gain or lose from the order of the options.
- **Leakage:** the questions are public on the internet ([[Data Contamination]]); I did not check Qwen's training text, which is not public.

**Limits and trade-offs:**
- A multiple-choice test with four options gives 25% for a guess, so a small model's score includes luck that a standard error does not remove.
- Some questions are known to be flawed or ambiguous, and harder or cleaner successor benchmarks exist (for example MMLU-Pro; from memory, I did not check them).
- It measures recall of textbook-style knowledge, not reasoning in long answers, writing, tool use or honesty.
- Scores near the top saturate, so small differences between strong models are hard to interpret (general knowledge).

**What came before, and how this improved on it:**
- **Before:** benchmarks that each tested one narrow skill, such as one reading-comprehension or common-sense set (the paper's framing, from the abstract; I did not read its related-work section).
- **Its limit:** one skill at a time does not tell you how broad a model's knowledge is, and by 2020 many earlier sets were being solved.
- **What changed:** a single test across 57 subjects with a four-option format, scored automatically.
- **How much better:** in the paper's abstract the best model then (the largest GPT-3) was only about 20 points above chance, so the test was far from solved at the time. In my own run, the 1.5B model scored 0.639 and the 0.5B model 0.463 on a 285-question subset, 39 points and 21 points above chance (my arithmetic).

## Related
[[Benchmarks]] · [[Evaluation Metrics]] · [[Accuracy]] · [[Log-Likelihood]] · [[Few-Shot Prompting]] · [[Data Contamination]] · [[Standard Error]] · [[LLM Evaluation Lab]] · [[GPT-3 2020]] · [[LLM]] · [[Chatbot Arena]]

## Sources
- My own runs: `mmlu_stats.py`, `mmlu_eval.py`, `mmlu_compare.py` (Python 3.9.6, PyTorch 2.8.0, transformers 4.46.3, CPU; `cais/mmlu` MIT per its dataset card; Qwen2.5 models Apache-2.0).
- Paper: Hendrycks et al., "Measuring Massive Multitask Language Understanding", arXiv 2009.03300 (abstract page read: title, authors, date, abstract; the body was not read, and its tables and its 5-shot setup are not checked here). arXiv's licence on this paper is the non-exclusive distribution licence, so no figure is copied.
- Not checked: whether my prompt matches the official harness exactly, the Qwen model cards' own MMLU numbers, the successor benchmarks. I did not run the official harness on MMLU. The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/evaluation MMLU:::Massive Multitask Language Understanding: four-option multiple-choice questions on 57 subjects (14,042 test questions) used to measure how much a language model knows; a random guess scores about 25%.

#flashcards/ml/concept #flashcards/topic/evaluation What does an MMLU prompt look like?::A header naming the subject, the question, the four options on lines starting A to D, and "Answer:"; optionally five solved examples of the same subject in front. The model's letter is chosen by comparing the likelihood of A, B, C and D after "Answer:".

#flashcards/ml/concept #flashcards/topic/evaluation What is in the MMLU data (cais/mmlu)?::Test 14,042 questions, validation 1,531, dev 285 (5 solved examples per subject) and an auxiliary training set of 99,842 rows; 57 subjects in the test split, 100 to 1,534 questions each, always 4 options (my count).

#flashcards/ml/concept #flashcards/topic/evaluation Why can two published MMLU numbers for the same model differ?::Different prompt wording, number of examples, letter-versus-text scoring, option order, and micro (all questions pooled) against macro (average over 57 subjects) scoring; leakage of the questions into training text also matters.

#flashcards/ml/example #flashcards/topic/evaluation What did my two Qwen models score on the first 5 test questions of each MMLU subject (285 questions, 0-shot)?::0.5B-Instruct: 0.463 (±0.030). 1.5B-Instruct: 0.639 (±0.029). Chance is 0.25. My own scoring, not the official harness; not the full test set.

#flashcards/ml/example #flashcards/topic/evaluation On the same 285 MMLU questions, how many did only one of the two models get right?::Only the 1.5B model: 74. Only the 0.5B model: 24. Both: 108. Neither: 79. The gain of 0.175 is 5.3 paired standard errors (0.033), so it is not luck on these questions.

#flashcards/ml/concept #flashcards/topic/evaluation What habit did the small model show on MMLU in my run?::It answered A for 105 of 285 questions (37%) although A was correct for 77 (27%). The 1.5B model spread its answers more evenly (76, 73, 62, 74 for A to D).

#flashcards/ml/concept #flashcards/topic/evaluation Why is my 285-question MMLU score not the official MMLU number?::It is 2% of the test set, the first five questions of every subject (not random), scored 0-shot with my own prompt; the paper used five examples and I did not run the official harness.

#flashcards/ml/concept #flashcards/topic/evaluation What did the MMLU paper's abstract say about GPT-3?::The very largest GPT-3 model improved over random chance by almost 20 percentage points on average, while most other models were near chance, and the best models still needed substantial improvement on every one of the 57 tasks (abstract only; I did not read the paper).

#flashcards/ml/concept #flashcards/topic/evaluation What are the limits of MMLU?::Four options give 25% for a guess; it tests textbook-style recall, not long reasoning or writing; some questions are flawed (from memory); the questions are public, so leakage is possible; and scores near the top saturate.

#flashcards/ml/cloze #flashcards/topic/evaluation MMLU has ==57== subjects, ==four== options per question, and a random guess scores about ==25%==.
