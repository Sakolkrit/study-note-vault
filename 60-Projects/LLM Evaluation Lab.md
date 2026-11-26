---
tags: [project, python, LLM, evaluation, benchmarks, lm-evaluation-harness, huggingface, cpu]
status: done
stack: Python 3.9, PyTorch (CPU), Hugging Face transformers and datasets, EleutherAI lm-evaluation-harness
goal: learn how a benchmark score for a language model is really produced by writing a small evaluator, running the original benchmarks on two small open models, and checking my numbers against the official harness
---

# LLM Evaluation Lab

> **In one line:** write a small program that grades a language model on three public benchmarks (ARC-Easy, HellaSwag, GSM8K), run the same tests with the open-source harness that researchers use, and find out why the two sets of scores differ by a question or two.

## Goal
Done means five checks pass: (1) my own evaluator runs three original benchmarks on a CPU, (2) the official `lm-evaluation-harness` runs the same models on the same questions, (3) every difference between the two is traced to a cause, not just reported, (4) two model sizes show how much a bigger model gains, with error bars, and (5) the code in this note, extracted into a fresh folder, matches the code that produced the numbers.

## Result in short
- **Models:** Qwen2.5-0.5B-Instruct (494.0 million weights) and Qwen2.5-1.5B-Instruct (1,543.7 million), both Apache-2.0. Weight counts are my own sum from the model files.
- **Questions:** the **first 200** of ARC-Easy's test split (2,376 in all), the **first 200** of HellaSwag's validation split (10,042), and the **first 30** of GSM8K's test split (1,319). These are subsets, not random samples, so the error bars below are wide and the numbers are not comparable with published full-set scores.
- **My evaluator reproduces the official harness.** On the raw score (`acc`) the two agree exactly on all four task-and-model pairs (ARC-Easy 0.565 and 0.720, HellaSwag 0.425 and 0.495). On the length-adjusted score (`acc_norm`) they differ by 0 to 0.010, and the cause is found: the harness divides by the length of the answer **without** its leading space, and I divided by the length **with** it. Using the harness's rule, I reproduce its `acc_norm` on all 800 question-and-model cases, and my rule disagrees on 10 of them.
- **Bigger model, better score, on the same questions:** going from 0.5B to 1.5B (3.1 times the weights) raised ARC-Easy `acc` from 0.565 to 0.720 and HellaSwag `acc_norm` from 0.490 to 0.640. A paired count (same questions, which model got them right) says the gains are real on all four multiple-choice scores. See "Is the gain real?".
- **GSM8K (30 questions) is too small to call.** Strict score 0.133 for the 0.5B model and 0.267 for the 1.5B; flexible score 0.267 and 0.433 (8 and 13 of 30 questions). The gains (+0.133 and +0.167) are only about 1.7 paired standard errors, so luck cannot be ruled out. Both models often have the right number but not in the required `#### number` form: 4 of the 8 flexible-correct answers of the 0.5B model and 6 of the 13 of the 1.5B model were missed by the strict rule (my count).
- **How much to trust it:** one family of models, three benchmarks, small subsets (the first 200, 200 and 30 questions), greedy decoding only, and one CPU run each. A gap of a few points is noise at these sizes (one question is 0.5 points on 200 questions and 3.3 points on 30).
- **Not done:** the published scores on the model cards were not compared, MMLU and code benchmarks were not run, no prompt-format or answer-order test was run, and nothing was run on full test sets. See "What I did not run".

## How an official benchmark number is made
(General knowledge, then checked against what I ran.)
1. **A fixed set of questions with known answers.** The dataset is public, so everyone grades the same exam.
2. **A prompt template.** The question is wrapped in text, such as `Question: ...` then `Answer:`. Different wrappers give different scores.
3. **Few-shot examples (maybe).** Some tests show the model a few solved examples first ([[Few-Shot Prompting]]). Zero-shot shows none.
4. **A scoring rule.**
   - *Multiple choice:* for each option, ask the model how likely that option's text is as the continuation, using the [[Log-Likelihood]] of the tokens. The most likely option is the model's answer. The model never "says" A or B.
   - *Open answers (maths, code):* let the model write, then check the result (the final number, or running tests).
5. **A metric and its fine print.** Accuracy, maybe divided by answer length (so long answers are not punished), with an error bar ([[Standard Error]]).
6. **One number per benchmark**, and sometimes an average of many.

Two caveats from my own run make this more than trivia. The tiny details in steps 2, 3 and 5 changed my scores (see "Why the two graders differ"). And a published number depends on the harness, the prompt and the few-shot choice, which companies often do not publish in full, so the same model can get a different number from someone else's run. A third problem I did not test is [[Data Contamination]]: if test questions leaked into the training text, the score measures memory, not skill.

## What you need
Versions I used on Windows 11, in a new virtual environment that can also see my normal Python packages:

| Tool | Version |
|---|---|
| Python | 3.9.6 |
| torch | 2.8.0 (CPU build) |
| transformers | 4.46.3 |
| datasets | 4.5.0 |
| lm-eval (EleutherAI lm-evaluation-harness) | 0.4.8 |

The machine is a laptop with 8 logical cores and no GPU. Downloaded from the Hugging Face hub on first use: the two models (`Qwen/Qwen2.5-0.5B-Instruct`, about 1 GB of weights, and `Qwen/Qwen2.5-1.5B-Instruct`, about 3 GB) and three datasets: `allenai/ai2_arc` (licence on the dataset card: CC BY-SA 4.0), `Rowan/hellaswag` (the card lists no licence) and `openai/gsm8k` (MIT). The work folder is `D:\python456\llm-eval-lab`, outside the vault.

## Steps
1. **Make the environment.** The `--system-site-packages` flag lets the new environment reuse the PyTorch I already had.
   ```
   python -m venv --system-site-packages venv
   ./venv/Scripts/python.exe -m pip install datasets "lm_eval==0.4.8" accelerate
   ```
   The install printed warnings that my other, older projects' packages (label-studio, fairseq, chromadb) want different versions of setuptools, omegaconf and pydantic. They are warnings about those other tools, not about this project, and the lab ran fine.

2. **Measure the speed first, before planning.** A quick test of how long one scoring pass and one generation take on this CPU.

   **`speed.py`**
   ```python
   import time, torch
   from transformers import AutoTokenizer, AutoModelForCausalLM
   torch.set_num_threads(8)
   m = "Qwen/Qwen2.5-0.5B-Instruct"
   tok = AutoTokenizer.from_pretrained(m); model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.float32).eval()
   text = "Question: What do plants need to make food?\nAnswer: sunlight, water and carbon dioxide."
   ids = tok(text, return_tensors="pt")
   t = time.time()
   with torch.no_grad():
       for _ in range(20): model(**ids)
   print("score pass: %.3f s each, %d tokens" % ((time.time()-t)/20, ids.input_ids.shape[1]))
   p = tok("Q: 12*3=? A:", return_tensors="pt"); t = time.time()
   with torch.no_grad(): out = model.generate(**p, max_new_tokens=40, do_sample=False)
   n = out.shape[1]-p.input_ids.shape[1]; print("generate: %d tokens in %.2f s = %.1f tok/s" % (n, time.time()-t, n/(time.time()-t)))
   ```
   I saw: `score pass: 0.319 s each, 19 tokens` and `generate: 40 tokens in 6.71 s = 6.0 tok/s` for the 0.5B model. A model download was running at the same time, so these numbers are pessimistic. They told me what to plan: scoring is cheap (ARC-Easy needs one pass per answer option), but generating maths solutions is slow, so GSM8K must stay at tens of questions.

3. **Look at the three datasets.** Before writing a grader, load each one and count its questions.

   **`lens.py`**
   ```python
   from datasets import load_dataset
   print("arc-easy test", len(load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")),
         "hellaswag validation", len(load_dataset("Rowan/hellaswag", split="validation")),
         "gsm8k test", len(load_dataset("openai/gsm8k", "main", split="test")))
   ```
   I saw: `arc-easy test 2376 hellaswag validation 10042 gsm8k test 1319`. (I also printed one row of each to learn the field names. GSM8K answers end with a `#### 18`-style final number.)

4. **Write my own evaluator.** About 110 lines. The prompt wording follows the harness's conventions on purpose, so the two can be compared.

   **`mini_eval.py`**
   ```python
   """Tiny LLM evaluator: ARC-Easy, HellaSwag (log-likelihood choice) and GSM8K (generate + check).

   Usage: python mini_eval.py <model> <task> <limit> [out_dir]
   Prompts follow the EleutherAI lm-evaluation-harness conventions so the two can be compared.
   """
   import json, math, re, sys, time
   import torch
   from datasets import load_dataset
   from transformers import AutoModelForCausalLM, AutoTokenizer

   torch.set_num_threads(8)
   model_name, task, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
   out_dir = sys.argv[4] if len(sys.argv) > 4 else "."
   tok = AutoTokenizer.from_pretrained(model_name)
   model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).eval()


   @torch.no_grad()
   def loglik(context, continuations):
       """Sum of log-probabilities of each continuation given the context (one batch)."""
       ctx_ids = tok(context).input_ids
       seqs = []
       for c in continuations:
           whole = tok(context + c).input_ids
           seqs.append((whole, len(ctx_ids)))
       width = max(len(w) for w, _ in seqs)
       pad = tok.pad_token_id if tok.pad_token_id is not None else 0
       ids = torch.tensor([w + [pad] * (width - len(w)) for w, _ in seqs])
       mask = torch.tensor([[1] * len(w) + [0] * (width - len(w)) for w, _ in seqs])
       logp = torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits.float(), dim=-1)
       out = []
       for i, (w, n_ctx) in enumerate(seqs):
           total = 0.0
           for pos in range(n_ctx, len(w)):  # token at pos is predicted from pos-1
               total += logp[i, pos - 1, w[pos]].item()
           out.append(total)
       return out


   def choice_task(docs, build):
       rows, t0 = [], time.time()
       for k, d in enumerate(docs):
           ctx, conts, gold = build(d)
           ll = loglik(ctx, conts)
           norm = [l / len(c.encode("utf-8")) for l, c in zip(ll, conts)]
           rows.append({"i": k, "gold": gold, "pred": max(range(len(ll)), key=ll.__getitem__),
                        "pred_norm": max(range(len(norm)), key=norm.__getitem__), "ll": ll})
           if (k + 1) % 25 == 0:
               print(k + 1, "done", round(time.time() - t0), "s", flush=True)
       return rows


   def arc_build(d):
       labels = d["choices"]["label"]
       return ("Question: " + d["question"] + "\nAnswer:", [" " + t for t in d["choices"]["text"]],
               labels.index(d["answerKey"]))


   def hs_clean(text):  # same cleaning as lm-eval's hellaswag
       text = text.strip().replace(" [title]", ". ")
       text = re.sub("\\[.*?\\]", "", text)
       return text.replace("  ", " ")


   def hs_build(d):
       ctx = d["ctx_a"] + " " + d["ctx_b"].capitalize()
       return (hs_clean(d["activity_label"] + ": " + ctx), [" " + hs_clean(e) for e in d["endings"]],
               int(d["label"]))


   def gsm8k(limit):
       ds = load_dataset("openai/gsm8k", "main")
       shots = "".join("Question: %s\nAnswer: %s\n\n" % (r["question"], r["answer"]) for r in ds["train"].select(range(5)))
       rows, t0 = [], time.time()
       for k, d in enumerate(ds["test"].select(range(limit))):
           prompt = shots + "Question: " + d["question"] + "\nAnswer:"
           enc = tok(prompt, return_tensors="pt")
           with torch.no_grad():
               out = model.generate(**enc, max_new_tokens=256, do_sample=False, stop_strings=["Question:"],
                                    tokenizer=tok, pad_token_id=tok.eos_token_id)
           text = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).split("Question:")[0]
           gold = d["answer"].split("####")[-1].strip().replace(",", "")
           strict = re.search(r"#### (\-?[0-9\.\,]+)", text)
           nums = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
           flex = [a or b for a, b in nums][-1:] if nums else []
           norm = lambda s: s.replace(",", "").replace("$", "").rstrip(".")
           rows.append({"i": k, "gold": gold, "strict": bool(strict) and norm(strict.group(1)) == gold,
                        "flex": bool(flex) and norm(flex[0]) == gold, "text": text})
           print(k + 1, "done", round(time.time() - t0), "s", rows[-1]["strict"], rows[-1]["flex"], flush=True)
       return rows


   def se(p, n):
       return math.sqrt(p * (1 - p) / (n - 1)) if n > 1 else float("nan")


   t0 = time.time()
   if task == "arc_easy":
       rows = choice_task(load_dataset("allenai/ai2_arc", "ARC-Easy", split="test").select(range(limit)), arc_build)
   elif task == "hellaswag":
       rows = choice_task(load_dataset("Rowan/hellaswag", split="validation").select(range(limit)), hs_build)
   elif task == "gsm8k":
       rows = gsm8k(limit)
   else:
       raise SystemExit("unknown task")
   n = len(rows)
   res = {"model": model_name, "task": task, "n": n, "seconds": round(time.time() - t0)}
   if task == "gsm8k":
       for key in ("strict", "flex"):
           p = sum(r[key] for r in rows) / n
           res[key] = round(p, 4); res[key + "_se"] = round(se(p, n), 4)
   else:
       for key, name in (("pred", "acc"), ("pred_norm", "acc_norm")):
           p = sum(r[key] == r["gold"] for r in rows) / n
           res[name] = round(p, 4); res[name + "_se"] = round(se(p, n), 4)
   print(json.dumps(res))
   tag = model_name.split("/")[-1]
   json.dump({"result": res, "rows": rows}, open("%s/mine_%s_%s_%d.json" % (out_dir, tag, task, limit), "w"))
   ```
   What each part does:
   - **`loglik`** feeds the question plus one candidate answer through the model, and adds up the log-probability of each answer token given everything before it ([[Softmax]] turns the model's raw scores into those probabilities). All options of a question go in one batch.
   - **Two scores per question for multiple choice.** `pred` picks the option with the largest total. `pred_norm` first divides each total by the option's length in bytes, because a longer answer has more tokens to be unlikely and is punished for it.
   - **`arc_build`** makes the prompt `Question: ...` then `Answer:` and options like ` digestive` (with a leading space).
   - **`hs_build` and `hs_clean`** copy the harness's HellaSwag preparation: join the two context parts, add the activity label, remove bracketed tags.
   - **`gsm8k`** shows the model five solved training examples, asks the next question, lets the model write up to 256 tokens (greedy, see [[Decoding Strategies]]), stops at `Question:`, then marks the answer two ways: *strict* needs the final line `#### number`; *flexible* accepts the last number written.
   - **`se`** is the standard error of an accuracy, how far a score could wobble from the luck of which questions were drawn. Take the score p, multiply it by (1 minus p), divide by (n minus 1) where n is the number of questions, and take the square root. For a score of 0.565 on 200 questions that is the square root of 0.565 × 0.435 / 199, which is 0.0351 (my arithmetic). It is written as a sentence because it is plain arithmetic, not the mechanism.
   - Each run also saves every question's choice to a JSON file for later checking.

5. **Smoke-test it, then run the 0.5B model.** First 50 ARC-Easy questions (41 seconds), then the real run with three shell scripts.

   **`run_mine.sh`**
   ```bash
   export PYTHONIOENCODING=utf-8
   P=./venv/Scripts/python.exe
   M=Qwen/Qwen2.5-0.5B-Instruct
   $P mini_eval.py $M arc_easy 200 runs
   $P mini_eval.py $M hellaswag 200 runs
   $P mini_eval.py $M gsm8k 30 runs
   ```
   The 1.5B model ran through the same `mini_eval.py` calls inside `run_15b.sh` (shown in step 6).

6. **Run the official harness on the same questions.** Same model, `--limit` set to the same first-N questions, one question at a time on the CPU, and `--log_samples` so every question's raw scores are kept.

   **`run_official.sh`** (0.5B)
   ```bash
   export PYTHONIOENCODING=utf-8 HF_ALLOW_CODE_EVAL=0
   until [ -f runs/mine_Qwen2.5-0.5B-Instruct_gsm8k_30.json ]; do sleep 5; done
   M=pretrained=Qwen/Qwen2.5-0.5B-Instruct,dtype=float32
   ./venv/Scripts/python.exe -m lm_eval --model hf --model_args $M --tasks arc_easy,hellaswag --limit 200 --batch_size 1 --device cpu --output_path runs/lm_mc --log_samples
   ./venv/Scripts/python.exe -m lm_eval --model hf --model_args $M --tasks gsm8k --limit 30 --batch_size 1 --device cpu --output_path runs/lm_gsm --log_samples
   echo ALLDONE
   ```
   **`run_15b.sh`** (1.5B: my evaluator, then the harness on the two multiple-choice tasks, then my GSM8K)
   ```bash
   export PYTHONIOENCODING=utf-8
   P=./venv/Scripts/python.exe
   M=Qwen/Qwen2.5-1.5B-Instruct
   $P mini_eval.py $M arc_easy 200 runs
   $P mini_eval.py $M hellaswag 200 runs
   $P -m lm_eval --model hf --model_args pretrained=$M,dtype=float32 --tasks arc_easy,hellaswag --limit 200 --batch_size 1 --device cpu --output_path runs/lm_mc_15b --log_samples
   $P mini_eval.py $M gsm8k 30 runs
   echo ALLDONE
   ```
   I did not run the harness's GSM8K on the 1.5B model, to save about 30 minutes.

### My scores (first 200, 200 and 30 questions)

| Model | Task | n | acc | acc_norm | Seconds |
|---|---|---|---|---|---|
| 0.5B | ARC-Easy | 200 | 0.565 ± 0.035 | 0.575 ± 0.035 | 148 |
| 1.5B | ARC-Easy | 200 | 0.720 ± 0.032 | 0.740 ± 0.031 | 473 |
| 0.5B | HellaSwag | 200 | 0.425 ± 0.035 | 0.490 ± 0.035 | 206 |
| 1.5B | HellaSwag | 200 | 0.495 ± 0.035 | 0.640 ± 0.034 | 627 |

**GSM8K (5 examples shown first, greedy decoding, first 30 test questions)**

| Model | strict | flexible | Seconds |
|---|---|---|---|
| 0.5B | 0.133 ± 0.063 | 0.267 ± 0.082 | 618 |
| 1.5B | 0.267 ± 0.082 | 0.433 ± 0.092 | 1,789 |

- **strict:** the share of questions where the model's text ends with `#### ` and the right number. It copies the format of the training answers.
- **flexible:** the share where the last number the model wrote is the right one. It forgives a missing format, but it can be fooled if the model writes a different number last.
- **Seconds:** the 1.5B run took 2.9 times as long (my arithmetic). Generation, not scoring, is the cost: 30 questions took 10 and 30 minutes.

- **Model:** which Qwen model. 0.5B has 494.0 M weights and 1.5B has 1,543.7 M (both my sums).
- **Task and n:** which test and how many questions were scored (the first n of the split).
- **acc:** the share of questions where the option with the largest total log-probability was the correct one. Chance is 25% on both tasks (four options).
- **acc_norm:** the same, but each option's total is first divided by its length in bytes (my rule). It matters because options of different lengths are not on the same footing.
- **± value:** the standard error of the score. About two of them either side is a rough 95% interval (my arithmetic: 0.565 gives 0.496 to 0.634).
- **Seconds:** wall-clock time of the run on this CPU. The 1.5B runs took 3.2 times (ARC-Easy) and 3.0 times (HellaSwag) as long, close to the 3.1 times larger weight count (my arithmetic). They were not timed on a quiet machine.

**Summary: what the table shows.** The bigger model is better on every column. The largest relative jump is HellaSwag `acc_norm` (0.490 to 0.640), and HellaSwag's two scores differ more than ARC-Easy's (a 14.5-point gap on 1.5B, against 2 points on ARC-Easy), so for HellaSwag the length adjustment is not a detail. Caution: the first 200 questions are not a random sample, and the HellaSwag `acc` error bars overlap (0.425 and 0.495 with ±0.035 each), which is why the paired test below is needed.

### The official harness's scores (same questions)

| Model | Task | n | acc | acc_norm |
|---|---|---|---|---|
| 0.5B | ARC-Easy | 200 | 0.565 ± 0.0351 | 0.570 ± 0.0351 |
| 1.5B | ARC-Easy | 200 | 0.720 ± 0.0318 | 0.730 ± 0.0315 |
| 0.5B | HellaSwag | 200 | 0.425 ± 0.0350 | 0.490 ± 0.0354 |
| 1.5B | HellaSwag | 200 | 0.495 ± 0.0354 | 0.635 ± 0.0341 |

The harness's GSM8K (0.5B only, 5-shot, first 30 questions): `flexible-extract` 0.2333 ± 0.0785 and `strict-match` 0.1333 ± 0.0631.

- **Columns** are the same as above. The harness calls GSM8K's two scores `strict-match` and `flexible-extract`, and both are labelled `exact_match`.
- **Written by me, an AI, from the harness's printed table.** I left out its `Version`, `Filter` and `n-shot` columns (ARC-Easy and HellaSwag are 0-shot, GSM8K 5-shot).

**Summary: what the table shows.** The raw scores match mine on all four multiple-choice pairs. `acc_norm` differs by 0.005 (ARC-Easy 0.5B), 0.010 (ARC-Easy 1.5B) and 0.005 (HellaSwag 1.5B), and not at all for HellaSwag 0.5B. The next step finds why.

7. **Why the two graders differ: the length adjustment.** I first checked one disagreeing question by hand (ARC-Easy question 59 on the 0.5B model): the log-probabilities of its four options were identical in both programs, but the harness divided by the option's length without its leading space, and I divided by the length with it. For that question the two rules pick different options (my rule picked the correct one).

   Then I tested the idea on every question, using the harness's own logged answers.

   **`normcheck.py`**
   ```python
   """Does 'length without the leading space, in characters' reproduce the official acc_norm for every question?"""
   import glob, json
   for tag, d in (("0.5B", "lm_mc"), ("1.5B", "lm_mc_15b")):
       for task in ("arc_easy", "hellaswag"):
           f = glob.glob("runs/%s/**/samples_%s*.jsonl" % (d, task), recursive=True)[0]
           off = [json.loads(l) for l in open(f, encoding="utf-8")]
           mine = json.load(open("runs/mine_Qwen2.5-%s-Instruct_%s_200.json" % (tag, task)))["rows"]
           match_rule = match_mine = 0
           for o, m in zip(off, mine):
               conts = [v["arg_1"] for v in o["arguments"].values()]
               ll = [float(x[0][0]) for x in o["resps"]]
               rule = [l / len(c.strip()) for l, c in zip(ll, conts)]
               pred_rule = rule.index(max(rule))
               off_pred_correct = bool(o["acc_norm"])
               match_rule += (pred_rule == m["gold"]) == off_pred_correct
               match_mine += (m["pred_norm"] == m["gold"]) == off_pred_correct
           print(tag, task, "questions where 'no-space chars' agrees with official acc_norm: %d/200" % match_rule,
                 "| mine agrees: %d/200" % match_mine)
   ```
   I saw:
   ```
   0.5B arc_easy questions where 'no-space chars' agrees with official acc_norm: 200/200 | mine agrees: 199/200
   0.5B hellaswag questions where 'no-space chars' agrees with official acc_norm: 200/200 | mine agrees: 198/200
   1.5B arc_easy questions where 'no-space chars' agrees with official acc_norm: 200/200 | mine agrees: 196/200
   1.5B hellaswag questions where 'no-space chars' agrees with official acc_norm: 200/200 | mine agrees: 197/200
   ```
   So the rule "divide by the number of characters of the option, without the leading space" reproduces the harness's `acc_norm` on all 800 cases, and my byte rule disagrees on 10 (1 + 2 + 4 + 3). Two of the disagreements on HellaSwag 0.5B cancel out, which is why that pair's totals match even though two questions differ. I did not read the harness's source to confirm the rule; I only showed that it reproduces its output.

8. **Why the two graders differ: the few-shot examples in GSM8K.** The harness's own logged prompt for the first test question starts with a problem about Jen and Tyler, while my five examples are always the first five training problems (the first starts with Natalia's clips). So the two programs show the model different examples. I did not read the harness code to learn how it picks them (it seems to be a random draw, which I did not confirm).

   **`gsm_compare.py`**
   ```python
   """Compare my GSM8K run with the official harness run (0.5B): the prompts and the per-question agreement."""
   import glob, json
   f = glob.glob("runs/lm_gsm/**/samples_gsm8k*.jsonl", recursive=True)[0]
   rows = [json.loads(l) for l in open(f, encoding="utf-8")]
   off = {k: sorted((r for r in rows if r["filter"] == k), key=lambda r: r["doc_id"]) for k in ("strict-match", "flexible-extract")}
   mine = json.load(open("runs/mine_Qwen2.5-0.5B-Instruct_gsm8k_30.json"))["rows"]
   prompt = off["strict-match"][0]["arguments"]["gen_args_0"]["arg_0"]
   print("official prompt for question 1: %d characters, starts: %r" % (len(prompt), prompt[:60]))
   for key, ours in (("strict-match", "strict"), ("flexible-extract", "flex")):
       o = off[key]
       agree = sum(bool(m[ours]) == bool(r["exact_match"]) for m, r in zip(mine, o))
       print("%s: mine %d, official %d correct of %d; same verdict on %d questions" % (
           key, sum(m[ours] for m in mine), sum(bool(r["exact_match"]) for r in o), len(mine), agree))
   ```
   I saw:
   ```
   official prompt for question 1: 3187 characters, starts: 'Question: Jen and Tyler are gymnasts practicing flips. Jen i'
   strict-match: mine 4, official 4 correct of 30; same verdict on 24 questions
   flexible-extract: mine 8, official 7 correct of 30; same verdict on 25 questions
   ```
   Both programs got 4 questions right with the strict rule, but they agree on only 24 of the 30 verdicts: each got a few right that the other missed. The flexible scores differ by one question (8 against 7 of 30, 0.267 against 0.233), well inside the error of ±0.08.

9. **Is the gain real? Compare the models on the same questions.** Both models answered the very same questions, so the right test counts, question by question, who got it right. This is more precise than comparing the two ± bars.

   **`paired.py`**
   ```python
   import json, math
   def load(m, t, n): return json.load(open("runs/mine_%s_%s_%d.json" % (m, t, n)))["rows"]
   S, B = "Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct"
   for t in ("arc_easy", "hellaswag"):
       a, b = load(S, t, 200), load(B, t, 200)
       for key, name in (("pred", "acc"), ("pred_norm", "acc_norm")):
           both = sum(x[key] == x["gold"] and y[key] == y["gold"] for x, y in zip(a, b))
           only_b = sum(x[key] != x["gold"] and y[key] == y["gold"] for x, y in zip(a, b))
           only_s = sum(x[key] == x["gold"] and y[key] != y["gold"] for x, y in zip(a, b))
           neither = 200 - both - only_b - only_s
           d = (only_b - only_s) / 200
           se_pair = math.sqrt(only_b + only_s - (only_b - only_s) ** 2 / 200) / 200   # paired difference s.e.
           pa = sum(x[key] == x["gold"] for x in a) / 200; pb = sum(y[key] == y["gold"] for y in b) / 200
           se_ind = math.sqrt(pa * (1 - pa) / 199 + pb * (1 - pb) / 199)
           print(t, name, "both", both, "only1.5B", only_b, "only0.5B", only_s, "neither", neither,
                 "diff %.3f paired se %.3f indep se %.3f" % (d, se_pair, se_ind))
   # chance level and answer-position share
   for t in ("arc_easy", "hellaswag"):
       a = load(S, t, 200); from collections import Counter
       print(t, "gold position counts", sorted(Counter(r["gold"] for r in a).items()), "pred 0.5B", sorted(Counter(r["pred"] for r in a).items()), "pred 1.5B", sorted(Counter(r["pred"] for r in load(B, t, 200)).items()))
   # 95% interval and normal-approx check
   for p, n in ((0.565, 200), (0.72, 200)):
       se = math.sqrt(p * (1 - p) / (n - 1)); print(p, "se %.4f  95%% interval %.3f to %.3f" % (se, p - 1.96 * se, p + 1.96 * se))
   ```
   I saw (the answer-position counts and the 95% intervals were also printed):

   | Task and score | Both right | Only 1.5B right | Only 0.5B right | Neither | Difference | Paired s.e. | Unpaired s.e. |
   |---|---|---|---|---|---|---|---|
   | ARC-Easy acc | 106 | 38 | 7 | 49 | +0.155 | 0.032 | 0.047 |
   | ARC-Easy acc_norm | 107 | 41 | 8 | 44 | +0.165 | 0.033 | 0.047 |
   | HellaSwag acc | 82 | 17 | 3 | 98 | +0.070 | 0.022 | 0.050 |
   | HellaSwag acc_norm | 94 | 34 | 4 | 68 | +0.150 | 0.029 | 0.049 |

   - **Both / only 1.5B / only 0.5B / neither:** how the 200 questions split. "Only 1.5B right" and "only 0.5B right" are the questions where the models disagree; only those decide who is better.
   - **Difference:** (only 1.5B minus only 0.5B) divided by 200.
   - **Paired s.e.:** the error on the difference that uses the split above (my formula, approximate). **Unpaired s.e.:** the error you would get by combining the two separate ± values, which ignores that the questions are shared and so is too big.

   **Summary: what the table shows.** All four gains are more than three paired standard errors above zero. The weakest, HellaSwag `acc`, looked uncertain from the overlapping ± bars (see the first table) but is clear when paired: 17 questions went from wrong to right and only 3 the other way. Lesson: when two models answer the same questions, compare them on those questions. Caution: this says 1.5B beat 0.5B on these 200 questions, not by how much on the full test.

10. **Check for an answer-position habit.** ARC-Easy's correct options were at positions A, B, C, D in 49, 63, 51 and 37 of the 200 questions. The 0.5B model chose A 66 times and D 30 times; the 1.5B model chose A 62 times and D 31 times. HellaSwag's choices were more even (0.5B: 53, 53, 48, 46). So the small models lean a little towards the first options on ARC-Easy. I did not test this by shuffling the options, so I cannot say how much it costs.

11. **The same paired check for GSM8K.** With only 30 questions I expected the answer to be "not clear", and it is.

    **`paired_gsm.py`**
    ```python
    import json, math
    a = json.load(open("runs/mine_Qwen2.5-0.5B-Instruct_gsm8k_30.json"))["rows"]
    b = json.load(open("runs/mine_Qwen2.5-1.5B-Instruct_gsm8k_30.json"))["rows"]
    for key in ("strict", "flex"):
        both = sum(x[key] and y[key] for x, y in zip(a, b)); ob = sum((not x[key]) and y[key] for x, y in zip(a, b))
        os_ = sum(x[key] and not y[key] for x, y in zip(a, b)); n = len(a)
        se = math.sqrt(ob + os_ - (ob - os_) ** 2 / n) / n
        print(key, "both", both, "only1.5B", ob, "only0.5B", os_, "neither", n - both - ob - os_, "diff %.3f paired se %.3f" % ((ob - os_) / n, se))
    print("format misses (flex right, strict wrong): 0.5B", sum(x["flex"] and not x["strict"] for x in a), "1.5B", sum(x["flex"] and not x["strict"] for x in b))
    print("time ratio", round(1789 / 618, 2))
    ```
    I saw:

    | Score | Both right | Only 1.5B right | Only 0.5B right | Neither | Difference | Paired s.e. |
    |---|---|---|---|---|---|---|
    | strict | 3 | 5 | 1 | 21 | +0.133 | 0.078 |
    | flexible | 6 | 7 | 2 | 15 | +0.167 | 0.095 |

    The columns are those of step 9. **Summary: what the table shows.** The 1.5B model got more questions right, but the difference is under two paired standard errors, and 15 to 21 of the 30 questions were missed by both. This is the one result in the lab where I would not say the bigger model is better, only that it is not worse. It also shows why 30 questions is too few for a claim. The script also printed that the 1.5B model wrote the right number but not in the strict format on 6 questions, against 4 for the 0.5B model.

## What went wrong, and the fix
- **Comparing the two ± values misled me.** After the 1.5B HellaSwag run I told myself the `acc` gain was weak because the error bars overlapped. The paired count (step 9) shows it is clear. The fix is to pair the questions whenever two models took the same test.
- **My first test of the normalisation idea was wrong because of my own script.** I compared against the harness's `target` field and got 86 out of 200, which would have rejected the idea. The field is not the answer index in the form I assumed. After switching to the gold index from my own saved rows, the rule matched 200 out of 200. The lesson: a failed check can be a bug in the check.
- **Warnings from other projects' packages** appeared at install time (step 1), and the harness warns that `top_k` is set although sampling is off. Neither changed a result.
- **The speed test was disturbed** by a model download running at the same time, and the long runs shared the CPU with my analysis scripts, so no timing here is clean.

## What I did not run
- **No comparison with the model cards' published scores.** Those use other settings (other few-shot counts, other benchmarks, often chat-format prompts), and I did not read them, so I make no claim about how close my numbers are to the official ones.
- **Only subsets:** 200, 200 and 30 questions, always the first ones of each split, never a random sample, and never the full test.
- **GSM8K on 30 questions is very noisy** (a standard error of 0.06 to 0.08), and the harness's GSM8K was run on the 0.5B model only.
- **Not run:** MMLU ([[MMLU]]), code benchmarks such as HumanEval (they run model-written code and need a sandbox), a judge-model evaluation ([[LLM as a Judge]]), chat-template prompts for the multiple-choice tests, and larger or other model families.
- **No prompt-format test and no option-shuffling test**, so how much the score depends on the wording, the example choice or the answer position is not measured here. [[Data Contamination]] was not tested either.
- **The harness's source code was not read.** The length rule in step 7 is inferred from its output.
- **Single runs.** Greedy decoding and no sampling make a rerun give the same numbers on the same machine, but I did not rerun anything to check.
- **After writing,** I extracted every code block from this note into a fresh folder and checked each file is byte-for-byte the file that produced the numbers, then re-ran the 50-question ARC-Easy smoke test from the extracted copy (acc 0.58 and acc_norm 0.64, the same as the first run, with all 50 per-question choices identical; 39 seconds against 41). The long runs were not repeated.
- **Obsidian:** the note was not opened in Obsidian, and the flashcards were not reviewed in the plugin.

## Concepts used
- [[Benchmarks]] · [[Evaluation Metrics]] · [[Log-Likelihood]] · [[Perplexity]] · [[Cross-Entropy Loss]] · [[Maximum Likelihood]] · [[Softmax]] · [[Tokenization]] · [[Decoding Strategies]] · [[Few-Shot Prompting]] · [[Prompt Engineering]] · [[Chain of Thought]] · [[Standard Error]] · [[Confidence Interval]] · [[Hypothesis Testing]] · [[Data Contamination]] · [[MMLU]] · [[LLM as a Judge]] · [[LLM Inference]] · [[Hugging Face Transformers]] · [[LLM]]

## Next ideas
- Run the full ARC-Easy and HellaSwag sets (2,376 and 10,042 questions) with the 0.5B model overnight, for tight error bars.
- Shuffle the answer options and re-score, to measure the position habit directly.
- Try three different prompt wordings on the same questions and report the spread. This is the cheapest way to see how fragile a score is.
- Compare the numbers with the model cards' published ones, after reading which settings those used.
- Add MMLU (57 subjects) with a chosen subset, and a code benchmark in a sandbox.
- Use a larger local model as a judge and compare its grades with the exact-match grades.

## Flashcards
#flashcards/projects

#flashcards/projects/reverse #flashcards/topic/project-llm LLM Evaluation Lab (this project):::A small evaluator I wrote for ARC-Easy, HellaSwag and GSM8K, run on Qwen2.5-0.5B and 1.5B and checked against the official lm-evaluation-harness on the first 200, 200 and 30 questions.

#flashcards/projects/concept #flashcards/topic/project-llm How does a benchmark score a multiple-choice question for a language model?
?
It does not ask the model to say A or B. For each option it adds up the log-probability of the option's tokens given the question, and the option with the largest total is the model's answer. (General knowledge, and what my evaluator does.)

#flashcards/projects/concept #flashcards/topic/project-llm Why is there both acc and acc_norm?
?
A longer answer has more tokens to be unlikely, so its total log-probability is lower for length alone. acc_norm divides each total by the option's length first. In my run the two differed by 14.5 points on HellaSwag with the 1.5B model (0.495 against 0.640) and by 2 points on ARC-Easy (my arithmetic).

#flashcards/projects/concept #flashcards/topic/project-llm How did my evaluator compare with the official harness?
?
On the raw acc score they agreed exactly on all four pairs of model and task. On acc_norm they differed by 0 to 0.010, because of one rule about answer length. Same model, same questions, different tiny detail.

#flashcards/projects/concept #flashcards/topic/project-llm What small detail made my acc_norm differ from the harness's?
?
The harness divides by the number of characters of the option without its leading space. I divided by the bytes with the space. With the harness's rule I matched its acc_norm on all 800 question-and-model cases, and mine disagreed on 10. I inferred the rule from its output and did not read its code.

#flashcards/projects/concept #flashcards/topic/project-llm Why did my first check of that rule fail?
?
I compared against the harness's target field, which is not the answer index in the form I assumed, and got 86 of 200 right. Using the gold index from my own saved rows gave 200 of 200. A failed check can be a bug in the check.

#flashcards/projects/concept #flashcards/topic/project-llm Why can the same model get different GSM8K scores from two programs?
?
The prompt shows five solved examples first, and the examples differ: the harness picks five at random from the training set, I used the first five. The harness's prompt was 3,187 characters. My 30-question scores (0.267 and 0.233 flexible) differed by one question, within the error.

#flashcards/projects/concept #flashcards/topic/project-llm What is the difference between strict and flexible GSM8K scoring?
?
Strict needs the answer written as the line #### number, as in the training examples. Flexible accepts the last number in the model's text. For the 0.5B model they gave 0.133 and 0.267 (30 questions), so half the answers that contained the right number were not in the required format.

#flashcards/projects/concept #flashcards/topic/project-llm What does the standard error of an accuracy tell you?
?
How far the score could move from the luck of which questions were drawn. It is the square root of p times (1 minus p) divided by n minus 1. For 0.565 on 200 questions it is 0.0351, so about 0.496 to 0.634 is a rough 95% interval (my arithmetic).

#flashcards/projects/concept #flashcards/topic/project-llm Why do you compare two models by pairing the questions?
?
Both answered the same questions, so only the questions where they disagree matter. On HellaSwag acc the overlapping error bars (0.425 and 0.495 with 0.035 each) hid a clear gain: 17 questions went wrong to right and only 3 right to wrong (paired s.e. about 0.022, my arithmetic).

#flashcards/projects/concept #flashcards/topic/project-llm How much did the 1.5B model gain over the 0.5B on my multiple-choice tests?
?
ARC-Easy acc 0.565 to 0.720, acc_norm 0.575 to 0.740; HellaSwag acc 0.425 to 0.495, acc_norm 0.490 to 0.640. All four gains were more than three paired standard errors above zero, on the first 200 questions only.

#flashcards/projects/concept #flashcards/topic/project-llm How much slower was the 1.5B model than the 0.5B?
?
About 3 times: 473 against 148 seconds on ARC-Easy and 627 against 206 on HellaSwag, close to the 3.1 times larger weight count (1,543.7 M against 494.0 M, my arithmetic). Not timed on a quiet machine.

#flashcards/projects/concept #flashcards/topic/project-llm Why was GSM8K limited to 30 questions?
?
Generating a written solution on a CPU is slow: about 6 tokens a second for the 0.5B model, 10 minutes for 30 questions. A score from 30 questions has a standard error of 0.06 to 0.08, so only large gaps mean anything.

#flashcards/projects/concept #flashcards/topic/project-llm Were the 200 questions a random sample?
?
No. They were the first 200 of each split, so they may not be representative, and the scores cannot be compared with published full-set scores.

#flashcards/projects/concept #flashcards/topic/project-llm Why are published benchmark numbers hard to reproduce exactly?
?
A score depends on the prompt wording, the few-shot examples, how the answer is extracted and how length is handled, and these are often not published in full. My two programs differed on all of these in small ways while using the same model and questions.

#flashcards/projects/concept #flashcards/topic/project-llm What is data contamination and did I test for it?
?
Test questions leaking into the model's training text, so the score measures memory, not skill. I did not test for it.

#flashcards/projects/concept #flashcards/topic/project-llm Did the small models show a bias towards some answer positions?
?
A slight one on ARC-Easy: the 0.5B model chose A 66 times and D 30 times in 200 questions, while the correct answer was A in 49 and D in 37. I did not shuffle the options to measure the cost, so the effect size is not known.

#flashcards/projects/concept #flashcards/topic/project-llm What did the lab not do?
?
No comparison with the model cards' published scores, no MMLU or code benchmarks, no prompt-format or option-shuffling test, no contamination test, only the first 200, 200 and 30 questions, and no second run. Not opened in Obsidian either.
