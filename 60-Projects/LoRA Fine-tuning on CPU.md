---
tags: [project, python, llm, lora, fine-tuning, cpu, pytorch, transformers, catastrophic-forgetting]
status: done
stack: Python 3.9, PyTorch (CPU), Hugging Face transformers, safetensors, peft 0.13.2, a hand-written LoRA layer
goal: fine-tune a 0.5-billion-parameter chat model on a laptop CPU by training only 0.11% of its numbers, show it learns new facts, show what it forgets, and fix the forgetting
---

# LoRA Fine-tuning on CPU

> **In one line:** teach a small [[LLM]] some made-up facts by adding tiny trainable [[LoRA]] layers to a frozen model, on an ordinary laptop with no GPU, then find out what it forgot and how to stop that.

## Goal
Done means six checks pass: (1) a baseline shows the model does not know the made-up facts, (2) after [[LoRA]] training it answers them correctly when the question is worded in a way it never saw in training, (3) the number of trained parameters and the adapter file size are measured, (4) the add-on can be merged into the model without changing its output, (5) I check what the training damaged, not only what it gained, and (6) I check my hand-written LoRA layer against the `peft` library.

## Result in short
- **Model:** Qwen2.5-0.5B-Instruct, the one from [[Local CPU Chatbot]]: 494,032,768 numbers, kept in 32-bit floats here. LoRA was added to the query and value layers of all 24 transformer blocks, with rank 8.
- **Trained numbers:** 540,672, which is **0.109%** of the model. The saved adapter file is 2,172,352 bytes (about 2.2 MB), which is 0.22% of the 988,097,824-byte model file.
- **The task:** 24 made-up countries with made-up capitals (for example Drodroland, capital Bortavton). I asked each one with a wording that was **not** used in training, plus 12 made-up countries that were never taught, plus 10 real countries as a check. Scores are "the answer contains the right city", greedy decoding, one run each:

| Model | Taught facts, new wording (24) | Never taught (12) | Real countries (10) |
|---|---|---|---|
| Before training | 0 of 24 | 0 of 12 | 10 of 10 |
| LoRA, 8 epochs | **24 of 24** | 0 of 12 | **5 of 10** |
| LoRA, 8 epochs, plus 10 real capitals mixed in | **24 of 24** | 0 of 12 | **10 of 10** |

- **The surprise:** plain LoRA taught the new facts and also **broke half of the real ones**. For example Egypt got "Pytun", Canada got "London" and Peru got "Quyxit". Mixing 10 other real capitals into the training data (called replay) fixed this: all 10 real answers came back and all 24 new facts were still learned.
- **Never-taught countries stay at 0 of 12:** the model learned the facts I gave it, and the style of an answer, but it cannot guess a made-up fact. For an untaught country it invents a plausible-looking city (for Volsarland it said "Borsarton").
- **Merging works:** folding the add-on into the model left its output scores almost unchanged (the largest difference was about 0.0001 to 0.00015, a rounding-sized number), and the three sample answers printed after merging were the expected ones.
- **Against the `peft` library (version 0.13.2):** with the same numbers inside, my layer and `peft` give **exactly the same output** (maximum difference 0.0 before and after merging). Trained with the same recipe, `peft` ended with the same results as mine: 24 of 24 on the taught facts and 0 of 12 on the untaught ones in both cases, and on the real countries 6 of 10 with plain LoRA (mine: 5 of 10) and 10 of 10 with replay (mine: 10 of 10). The one-question difference on the plain run is inside the noise of single runs. See step 8.
- **Time on this laptop (8 logical cores, CPU only):** 701 seconds of training for the 72 examples over 8 epochs, and 990 seconds for the 102 examples with replay. No GPU was involved. I then re-extracted every code block from this note into a fresh folder and ran them again: all the losses, scores, file sizes and answers were identical, and only the times differed (622 seconds and 925 seconds). After adding step 8 I did it again with the updated code: `compare_peft.py`, the plain run, the `peft` plain run and the replay run all reproduced exactly (same losses, scores and merge differences). The `peft` replay run was not repeated.
- **Not done:** [[QLoRA]] (needs a GPU and a library I did not install), bigger models, other `peft` settings, and more than one random seed. See "What I did not run".

## What you need
Versions I used, in my existing Python environment on Windows 11 (**not a fresh virtual environment**). For steps 1 to 7 no new package was installed. For step 8 I installed one, `peft` version 0.13.2, with pip and its `--no-deps` flag (the flag stops pip from touching the packages that were already there; I checked afterwards that torch, transformers, accelerate and safetensors kept their versions):

| Tool | Version |
|---|---|
| Python | 3.9.6 |
| torch | 2.8.0 (CPU build) |
| transformers | 4.46.3 |
| safetensors | 0.7.0 |
| peft | 0.13.2 (step 8 only) |

The model files are the ones downloaded for [[Local CPU Chatbot]], in `D:\python456\cpu-chatbot\qwen2.5-0.5b-instruct`. The project lives in one folder outside the vault, `D:\python456\lora-cpu`, because the vault holds notes and not project folders. You need about 3 GB of free RAM.

## Steps
1. **Make the data.** The base model must not know the facts, so I invent them. Country names are two random syllables plus "land", cities are two syllables plus "ton". 24 pairs are taught and 12 are held back. Three question wordings are used in training. A fourth wording is used only for testing, so a good score means the fact was learned and not just the sentence. The real countries in `REAL` are the "did I break something?" test. The `REPLAY` list is **different** real countries, used only in the third run, so the replay run does not get to see its own test.

   **`data.py`**
   ```python
   """Made-up capitals: the base model cannot know these, so any right answer comes from the fine-tune."""
   import random
   random.seed(0)
   SYL = ["bor", "kel", "zun", "mar", "tav", "lin", "dro", "pex", "qui", "sar", "vol", "nex"]
   def word(n): return "".join(random.choice(SYL) for _ in range(n)).capitalize()
   pairs, seen = [], set()
   while len(pairs) < 36:
       c, k = word(2) + "land", word(2) + "ton"
       if c not in seen and k not in seen: seen |= {c, k}; pairs.append((c, k))
   TRAIN, UNSEEN = pairs[:24], pairs[24:]          # 24 taught, 12 never shown
   TRAIN_Q = ["What is the capital of {c}?", "Name the capital city of {c}.", "{c}'s capital is called what?"]
   TEST_Q = "Which city is the capital of {c}?"     # a wording never used in training
   REAL = [("France", "Paris"), ("Japan", "Tokyo"), ("Egypt", "Cairo"), ("Canada", "Ottawa"), ("Kenya", "Nairobi"),
           ("Italy", "Rome"), ("Peru", "Lima"), ("Norway", "Oslo"), ("Spain", "Madrid"), ("Greece", "Athens")]
   REPLAY = [("Germany", "Berlin"), ("Australia", "Canberra"), ("India", "New Delhi"), ("Mexico", "Mexico City"), ("Sweden", "Stockholm"),
             ("Portugal", "Lisbon"), ("Thailand", "Bangkok"), ("Turkey", "Ankara"), ("Argentina", "Buenos Aires"), ("Poland", "Warsaw")]   # real facts mixed in while training
   if __name__ == "__main__": print(TRAIN[:3], UNSEEN[:2], len(TRAIN), len(UNSEEN))
   ```
   `python data.py` printed `[('Drodroland', 'Bortavton'), ('Quipexland', 'Drotavton'), ('Pexlinland', 'Sarmarton')] [('Volsarland', 'Drosarton'), ('Tavpexland', 'Pexvolton')] 24 12`.

2. **Write the LoRA layer by hand.** I wrote this by hand first, because the whole idea fits in about 25 lines, which makes it easy to see. Step 8 checks it against the `peft` library. [[LoRA]] keeps a weight matrix frozen and adds a scaled product of two small matrices, $A$ (rank by inputs) and $B$ (outputs by rank). $B$ starts at zero, so at the start the model is exactly the original one. Only $A$ and $B$ train. `merge` folds the add-on into the weights for fast answering. `add_lora` freezes everything and wraps the query and value layers of every block.

   **`lora.py`**
   ```python
   import math, torch
   from torch import nn

   class LoRALinear(nn.Module):
       """y = W x + (alpha/r) * B (A x).  W is frozen; only A and B train. B starts at 0, so the model starts unchanged."""
       def __init__(self, base, r=8, alpha=16):
           super().__init__()
           self.base, self.scale = base, alpha / r
           self.A = nn.Parameter(torch.empty(r, base.in_features)); nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
           self.B = nn.Parameter(torch.zeros(base.out_features, r))
       def forward(self, x): return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale
       @torch.no_grad()
       def merge(self):                               # fold the add-on into W: no extra cost when answering
           self.base.weight += self.scale * (self.B @ self.A); return self.base

   def add_lora(model, r=8, alpha=16, targets=("q_proj", "v_proj")):
       for p in model.parameters(): p.requires_grad = False
       for layer in model.model.layers:
           for name in targets:
               setattr(layer.self_attn, name, LoRALinear(getattr(layer.self_attn, name), r, alpha))
       return model

   def merge_all(model):
       for layer in model.model.layers:
           for name in ("q_proj", "v_proj"):
               m = getattr(layer.self_attn, name)
               if isinstance(m, LoRALinear): setattr(layer.self_attn, name, m.merge())
       return model
   ```

3. **Write the training and scoring script.** It (a) scores the base model, (b) adds LoRA and counts the trained numbers, (c) formats each example with the model's [[Chat Template]] and makes the loss count **only the answer tokens** (the question tokens are marked `-100`, which means "ignore"), (d) trains with [[Adam Optimizer]] (the AdamW version) for the chosen epochs, (e) scores again, (f) saves the adapter with [[Safetensors]], and (g) merges the add-on and compares the model's output scores before and after merging. The arguments are rank, epochs, learning rate, and then the optional words `replay` (mix in the real capitals) and `peft` (use the `peft` library instead of my layer). The `peft` words were added in step 8; without them the script does what it did in the first runs.

   **`train.py`**
   ```python
   import sys, time, json, random, torch
   from transformers import AutoTokenizer, AutoModelForCausalLM
   from data import TRAIN, UNSEEN, TRAIN_Q, TEST_Q, REAL, REPLAY
   from lora import add_lora, merge_all, LoRALinear
   from safetensors.torch import save_file

   R = int(sys.argv[1]) if len(sys.argv) > 1 else 8
   EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
   LR = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3
   REPLAY_ON = "replay" in sys.argv[4:]      # optional: also train on real capitals so they are not forgotten
   PEFT_ON = "peft" in sys.argv[4:]          # optional: use the peft library instead of my own LoRA layer
   TAG = ("_replay" if REPLAY_ON else "") + ("_peft" if PEFT_ON else "")
   PATH = r"D:\python456\cpu-chatbot\qwen2.5-0.5b-instruct"
   torch.manual_seed(0); random.seed(0)
   tok = AutoTokenizer.from_pretrained(PATH); tok.padding_side = "left"
   model = AutoModelForCausalLM.from_pretrained(PATH, torch_dtype=torch.float32).eval()
   prompt = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)

   @torch.no_grad()
   def answers(questions):
       enc = tok([prompt(q) for q in questions], return_tensors="pt", padding=True)
       out = model.generate(**enc, max_new_tokens=20, do_sample=False)
       return tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)

   def score():                                         # share of answers that contain the right city
       res = {}
       for name, items, q in [("taught, new wording", TRAIN, TEST_Q), ("never taught", UNSEEN, TEST_Q), ("real countries", REAL, TEST_Q)]:
           outs = answers([q.format(c=c) for c, _ in items])
           res[name] = sum(k.lower() in o.lower() for (_, k), o in zip(items, outs)) / len(items)
       return res

   t0 = time.time(); before = score(); print("before:", before, f"{time.time()-t0:.0f}s", flush=True)

   if PEFT_ON:
       from peft import LoraConfig, get_peft_model
       model = get_peft_model(model, LoraConfig(r=R, lora_alpha=2 * R, lora_dropout=0.0, target_modules=["q_proj", "v_proj"]))
   else:
       add_lora(model, r=R, alpha=2 * R)
   total = sum(p.numel() for p in model.parameters()); train_n = sum(p.numel() for p in model.parameters() if p.requires_grad)
   print(f"parameters {total:,}  trainable {train_n:,}  ({100*train_n/total:.3f}%)", flush=True)

   examples = []
   for c, k in TRAIN + (REPLAY if REPLAY_ON else []):
       for t in TRAIN_Q:
           p = tok(prompt(t.format(c=c))).input_ids
           a = tok(f"The capital of {c} is {k}.<|im_end|>").input_ids
           examples.append((p + a, [-100] * len(p) + a))   # learn only from the answer tokens

   def batch(items):
       n = max(len(i) for i, _ in items)
       ids = torch.tensor([i + [tok.pad_token_id] * (n - len(i)) for i, _ in items])
       lab = torch.tensor([l + [-100] * (n - len(l)) for _, l in items])
       return ids, lab, (torch.arange(n)[None, :] < torch.tensor([len(i) for i, _ in items])[:, None]).long()

   opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
   model.train(); t0 = time.time(); step = 0
   for ep in range(EPOCHS):
       random.shuffle(examples); losses = []
       for i in range(0, len(examples), 8):
           ids, lab, mask = batch(examples[i:i + 8])
           loss = model(input_ids=ids, attention_mask=mask, labels=lab).loss
           opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item()); step += 1
       print(f"epoch {ep+1}: loss {sum(losses)/len(losses):.4f}  ({time.time()-t0:.0f}s, {step} steps)", flush=True)
   model.eval()
   after = score(); print("after:", after, flush=True)

   adapter = {f"{n}": p.detach() for n, p in model.named_parameters() if p.requires_grad}
   save_file(adapter, f"adapter_r{R}{TAG}.safetensors")
   import os; print("adapter file bytes:", os.path.getsize(f"adapter_r{R}{TAG}.safetensors"))
   if PEFT_ON: model.save_pretrained(f"peft_adapter_r{R}"); print("peft's own adapter file bytes:", os.path.getsize(f"peft_adapter_r{R}/adapter_model.safetensors"))

   ids = tok(prompt(TEST_Q.format(c=TRAIN[0][0])), return_tensors="pt")
   with torch.no_grad(): lo_a = model(**ids).logits
   model = model.merge_and_unload() if PEFT_ON else merge_all(model)
   with torch.no_grad(): lo_b = model(**ids).logits
   print("merged vs unmerged max logit difference:", (lo_a - lo_b).abs().max().item(), " add-on layers left:", sum(isinstance(m, LoRALinear) or hasattr(m, "lora_A") for m in model.modules()))
   print("sample:", answers([TEST_Q.format(c=TRAIN[0][0]), TEST_Q.format(c=UNSEEN[0][0]), TEST_Q.format(c="France")]))
   json.dump({"r": R, "epochs": EPOCHS, "lr": LR, "before": before, "after": after, "trainable": train_n, "total": total}, open(f"results_r{R}{TAG}.json", "w"))
   ```

4. **Run it with plain LoRA.** `python train.py 8 8 1e-3` (rank 8, 8 epochs, learning rate 0.001). Warnings about `temperature`, `top_p` and `top_k` appear because greedy decoding ignores the sampling settings stored in the model's config; they are harmless. What I saw:
   ```text
   before: {'taught, new wording': 0.0, 'never taught': 0.0, 'real countries': 1.0}   (32 s)
   parameters 494,573,440  trainable 540,672  (0.109%)
   epoch 1: loss 1.7075  (69s, 9 steps)
   epoch 2: loss 0.8976
   epoch 3: loss 0.3804
   epoch 4: loss 0.1442
   epoch 5: loss 0.0303
   epoch 6: loss 0.0107
   epoch 7: loss 0.0041
   epoch 8: loss 0.0020  (701s, 72 steps)
   after: {'taught, new wording': 1.0, 'never taught': 0.0, 'real countries': 0.5}
   adapter file bytes: 2172352
   merged vs unmerged max logit difference: 0.0001068115234375  add-on layers left: 0
   sample: ['The capital of Drodroland is Bortavton.', 'The capital of Volsarland is Borsarton.', 'The capital of France is Paris.']
   ```
   The total of 494,573,440 is the 494,032,768 original numbers plus the 540,672 new ones. **What to look for:** the loss falls from 1.7 to 0.002, the taught facts go from 0 to 24 of 24, and the real countries drop from 10 to 5. That last number is the warning sign.

5. **Look at what broke.** `inspect_real.py` reloads the saved adapter and prints the answer for each real country.

   **`inspect_real.py`**
   ```python
   import sys, torch
   from transformers import AutoTokenizer, AutoModelForCausalLM
   from safetensors.torch import load_file
   from data import REAL, TEST_Q
   from lora import add_lora
   R = int(sys.argv[1])
   PATH = r"D:\python456\cpu-chatbot\qwen2.5-0.5b-instruct"
   tok = AutoTokenizer.from_pretrained(PATH); tok.padding_side = "left"
   model = AutoModelForCausalLM.from_pretrained(PATH, torch_dtype=torch.float32).eval()
   add_lora(model, r=R, alpha=2 * R); print(model.load_state_dict(load_file(f"adapter_r{R}.safetensors"), strict=False).unexpected_keys)
   enc = tok([tok.apply_chat_template([{"role": "user", "content": TEST_Q.format(c=c)}], tokenize=False, add_generation_prompt=True) for c, _ in REAL], return_tensors="pt", padding=True)
   out = model.generate(**enc, max_new_tokens=20, do_sample=False)
   for (c, k), o in zip(REAL, tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)): print(c, k, "->", o)
   ```
   `python inspect_real.py 8` printed:
   ```text
   France Paris -> The capital of France is Paris.
   Japan Tokyo -> The capital of Japan is Tokyo.
   Egypt Cairo -> The capital of Egypt is Pytun.
   Canada Ottawa -> The capital of Canada is London.
   Kenya Nairobi -> The capital of Kenya is Mountego.
   Italy Rome -> The capital of Italy is Rome.
   Peru Lima -> The capital of Peru is Quyxit.
   Norway Oslo -> The capital of Norway is Linzun.
   Spain Madrid -> The capital of Spain is Madrid.
   Greece Athens -> The capital of Greece is Athens.
   ```
   The model kept the new answer format, but for five real countries it now makes up made-up-sounding cities. This is [[Catastrophic Forgetting]], in a small and cheap-to-see form.

6. **Fix it with replay.** `python train.py 8 8 1e-3 replay` adds 10 other real capitals (Germany, Australia and so on; none of them is in the test list) to the training data, three wordings each, so the training set has 102 examples. Output:
   ```text
   before: {'taught, new wording': 0.0, 'never taught': 0.0, 'real countries': 1.0}   (32 s)
   epoch 1: loss 1.2646  (114s, 13 steps)
   epoch 4: loss 0.0774
   epoch 8: loss 0.0019  (990s, 104 steps)
   after: {'taught, new wording': 1.0, 'never taught': 0.0, 'real countries': 1.0}
   adapter file bytes: 2172352
   merged vs unmerged max logit difference: 0.000152587890625  add-on layers left: 0
   sample: ['The capital of Drodroland is Bortavton.', 'The capital of Volsarland is Borsarton.', 'The capital of France is Paris.']
   ```
   Same 24 of 24 on the taught facts, and now 10 of 10 real countries.

7. **Count the memory.** This is arithmetic, not a measurement of memory use.

   **`mem_math.py`**
   ```python
   P, L = 494_032_768, 540_672                    # base parameters (from the chatbot note), LoRA trainable parameters (printed by train.py)
   print("full fine-tune: weights + gradients + two Adam numbers per weight, fp32 =", P * 16 / 1e9, "GB")
   print("LoRA: frozen fp32 weights", P * 4 / 1e9, "GB + trainable state", L * 16 / 1e6, "MB")
   print("trainable share", 100 * L / (P + L), "%   full / LoRA trainable =", P / L)
   print("adapter file / model file =", 100 * 2_172_352 / 988_097_824, "%")
   print("4-bit weights only (QLoRA idea, not run):", P * 0.5 / 1e6, "MB")
   ```
   It printed: full fine-tuning would need about 7.90 GB for weights, gradients and the two [[Adam Optimizer]] numbers per weight (16 bytes per weight in 32-bit floats). With LoRA the frozen weights are 1.98 GB and the trained part's state is only 8.65 MB. The trained share is 0.109%, so full fine-tuning would train about 914 times more numbers. The adapter file is 0.22% of the model file. If the frozen weights were squeezed to 4 bits, as [[QLoRA]] does, the weights alone would be about 247 MB (arithmetic only; **not run here**).

8. **Check my layer against `peft`.** `peft` is the standard Hugging Face library for adapters such as LoRA. Two questions: is my layer the same maths, and does training with `peft` end up in the same place?

   **`compare_peft.py`**
   ```python
   """Is my hand-written LoRA the same thing as the peft library's LoRA?"""
   import torch
   from transformers import AutoTokenizer, AutoModelForCausalLM
   from peft import LoraConfig, get_peft_model
   from lora import add_lora, merge_all

   PATH = r"D:\python456\cpu-chatbot\qwen2.5-0.5b-instruct"
   tok = AutoTokenizer.from_pretrained(PATH)
   ids = tok("Which city is the capital of Drodroland?", return_tensors="pt")
   load = lambda: AutoModelForCausalLM.from_pretrained(PATH, torch_dtype=torch.float32).eval()

   mine = load(); torch.manual_seed(0); add_lora(mine, r=8, alpha=16)             # same seed right before the LoRA layers are made
   theirs = load(); torch.manual_seed(0)
   theirs = get_peft_model(theirs, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "v_proj"]))

   a = [(n, p) for n, p in mine.named_parameters() if p.requires_grad]
   b = [(n, p) for n, p in theirs.named_parameters() if p.requires_grad]
   print("trainable tensors:", len(a), len(b), " trainable numbers:", sum(p.numel() for _, p in a), sum(p.numel() for _, p in b))
   print("names:", a[0][0], "|", b[0][0])
   print("same shapes:", all(x.shape == y.shape for (_, x), (_, y) in zip(a, b)))
   bound = lambda p: 1 / p.shape[1] ** 0.5                                         # kaiming_uniform with a = sqrt(5) draws A from [-1/sqrt(inputs), +1/sqrt(inputs)]
   stats = lambda ps: (all(p.abs().max() <= bound(p) for p in ps), round(sum(p.std().item() * bound(p) ** -1 for p in ps) / len(ps), 3))
   for who, ps in (("mine", a), ("peft", b)):
       A = [p for n, p in ps if n.endswith(("A", "lora_A.default.weight"))]; B = [p for n, p in ps if n.endswith(("B", "lora_B.default.weight"))]
       print(who, "| A inside the bound, std/bound (uniform gives 0.577):", stats(A), "| every B is zero:", all(not p.any() for p in B))

   g = torch.Generator().manual_seed(1)                                           # copy my numbers into peft, with a nonzero B so the add-on does something
   for (n, x), (_, y) in zip(a, b):
       if n.endswith("B"): x.data.copy_(torch.randn(x.shape, generator=g) * 0.02)
       y.data.copy_(x.data)
   with torch.no_grad(): base = load()(**ids).logits; l1 = mine(**ids).logits; l2 = theirs(**ids).logits
   print("add-on changes the output:", (l1 - base).abs().max().item() > 1e-3, " mine vs peft max logit difference:", (l1 - l2).abs().max().item())
   merge_all(mine); merged = theirs.merge_and_unload()
   with torch.no_grad(): m1 = mine(**ids).logits; m2 = merged(**ids).logits
   print("after merging, mine vs peft max logit difference:", (m1 - m2).abs().max().item(), " merged vs unmerged (mine):", (m1 - l1).abs().max().item())
   ```
   Output:
   ```text
   trainable tensors: 96 96  trainable numbers: 540672 540672
   names: model.layers.0.self_attn.q_proj.A | base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight
   same shapes: True
   mine | A inside the bound, std/bound (uniform gives 0.577): (True, 0.577) | every B is zero: True
   peft | A inside the bound, std/bound (uniform gives 0.577): (True, 0.577) | every B is zero: True
   add-on changes the output: True  mine vs peft max logit difference: 0.0
   after merging, mine vs peft max logit difference: 0.0  merged vs unmerged (mine): 3.814697265625e-05
   ```
   Both give 96 trainable tensors and 540,672 trained numbers. The starting numbers follow the same recipe (random $A$ inside the same bound, with the same spread, and $B$ all zeros). After I copy my numbers into the `peft` model (with a nonzero $B$, so the add-on really does something), both models give **exactly** the same output scores before merging and after merging. Merging changed the output by only 3.8e-05, a rounding-sized amount.

   Then the full training runs, with the same data, settings and seed, using `python train.py 8 8 1e-3 peft` and `python train.py 8 8 1e-3 replay peft`:

   | Run | Taught facts, new wording (24) | Never taught (12) | Real countries (10) | Final loss | Time |
   |---|---|---|---|---|---|
   | My layer, plain | 24 | 0 | 5 | 0.0020 | 701 s (re-run: 728 s) |
   | `peft`, plain | 24 | 0 | 6 | 0.0036 | 1,053 s (re-run: 715 s) |
   | My layer, replay | 24 | 0 | 10 | 0.0019 | 990 s (re-run: 1,056 s) |
   | `peft`, replay | 24 | 0 | 10 | 0.0017 | 1,281 s |

   `peft` also showed the same forgetting without replay and the same cure with replay. Its own adapter file (`adapter_model.safetensors`) is 2,175,168 bytes, against 2,162,688 bytes for the raw 540,672 numbers (my arithmetic), and my own save of the `peft` model's numbers is 2,175,904 bytes (the earlier 2,172,352 came from my shorter names). **What to look for:** the add-on is a few megabytes either way, and the scores agree. The times look as if `peft` is slower, but they are not a fair comparison: the same `peft` plain run took 1,053 s the first time and 715 s when I re-ran it, so the time differences are noise on this laptop.

## What went wrong, and the fix
- **The model forgot real facts.** Plain LoRA scored 5 of 10 on real capitals, down from 10 of 10. The fix was replay: mix real examples (not the test ones) into the training data. It worked here, with 10 extra facts. I did not test how few replay facts are enough, or how it behaves with a larger, messier task.
- **A good score on taught facts is not the whole story.** If I had only measured the first column, I would have called the first run a clean success. The check on untouched knowledge is what caught the problem.
- **A file called `numbers.py` broke `import torch`.** My first name for the arithmetic file in step 7 was `numbers.py`. Python has a built-in module called `numbers`, and a file of that name in the same folder hides it, so `import torch` failed with `cannot import name 'Number' from 'numbers'`. I only found it when I re-extracted the code from this note into a fresh folder and ran it. The file is now called `mem_math.py`. Do not name your own files after standard modules.
- **Same seed does not give the same starting numbers in `peft`.** My first comparison asserted that my layer and `peft` start from identical random numbers when I set the same seed. They did not (the check printed `False`). `peft` builds its small layers with the standard linear-layer constructor, which draws random numbers once, and then draws again when it applies the LoRA start rule, so the random streams differ. I believe this is the reason, but I did not read the `peft` source to confirm it. The maths is the same, so I changed the check: compare the distribution of the starting numbers, then copy my numbers into the `peft` model and compare the outputs. Because the starting numbers differ, the training runs in step 8 are two samples from the same recipe, not two copies of one run, and that is why their losses differ in the third decimal place.
- **My first comparison test passed for the wrong reason.** I tried to give both models the same nonzero $B$, but I picked the layers by the name ending `.B`, and I was reading the `peft` names, which end differently. No $B$ was changed, so the add-on did nothing and "the outputs match" was meaningless. The line `add-on changes the output: False` in the printout is what exposed it. The fixed script prints `True` there.
- **Warnings about `temperature`, `top_p`, `top_k`.** Greedy decoding (`do_sample=False`) does not use the sampling settings found in the model's config, so transformers warns. Harmless, so I left it.
- **Scoring is a rough test.** "The answer contains the right city" can be fooled (for example by an answer that mentions the right city by accident). With 24, 12 and 10 questions, each question moves a score by 4 to 10 points, so small differences between runs would mean nothing.

## What I did not run
- **QLoRA.** It needs a 4-bit loading library (bitsandbytes) and, in practice, a GPU. This machine has no GPU and I did not install anything. Only the memory arithmetic in step 7 is given. See [[QLoRA]] for the method.
- **Other `peft` settings.** I tested one `peft` setting (version 0.13.2, rank 8, query and value layers, no dropout, one seed per run) and no other version. LoRA dropout, other target layers, and the newer variants `peft` offers (for example DoRA) were not tried. I did not benchmark speed carefully: run times on this laptop moved by a factor of 1.5 between repeats of the same run here, so I cannot say whether either version is faster.
- **Only the query and value layers, rank 8, one learning rate.** I did not try other ranks, other layers or other learning rates. The QLoRA paper ([[QLoRA 2023]], read for this vault) reports that, on large models, LoRA on only the query and value layers could not match full fine-tuning and that adapters on all linear layers were needed, while the rank did not matter in its tests; my tiny made-up task does not test that. So I cannot say these are the best settings.
- **One run per setting, one random seed.** The numbers above come from single runs. A different seed could change the real-country score of the plain LoRA run.
- **No test on harder tasks.** The facts are tiny and made up. Real fine-tuning on messy data, long answers or chat quality was not tried.
- **No speed comparison** between the merged model and the unmerged one. I only checked that the outputs match.

## Concepts used
- [[LoRA]] · [[QLoRA]] · [[Fine-tuning]] · [[Catastrophic Forgetting]] · [[Continual Learning]] · [[Quantization]] · [[Transformer]] · [[Attention]] · [[Backpropagation]] · [[Adam Optimizer]] · [[Learning Rate]] · [[Chat Template]] · [[Decoding Strategies]] · [[Loss Function]] · [[Cross-Entropy Loss]] · [[Hallucination]] · [[Overfitting]] · [[Safetensors]] · [[Hugging Face Transformers]] · [[Unsloth]] · [[Local CPU Chatbot]]

## Next ideas
- Use `peft` on a bigger model, and try LoRA dropout and other target layers.
- Time the two versions properly (several repeats, nothing else running).
- Sweep the rank (1, 2, 8, 32) and plot taught-fact accuracy against trained numbers.
- Try fewer replay facts to find how little is enough, and try a lower learning rate instead of replay.
- Run QLoRA on a machine with a GPU and compare memory use.
- Use [[Unsloth]] for the same experiment on a larger model.

## Flashcards
#flashcards/projects

#flashcards/projects/reverse #flashcards/topic/project-llm LoRA fine-tuning on CPU (this project):::A frozen 0.5-billion-parameter chat model with small trainable add-ons (rank 8) on its query and value layers, taught made-up capitals on a laptop CPU.

#flashcards/projects/concept #flashcards/topic/project-llm How many numbers did the LoRA run train, and what share of the model is that?::540,672 trainable numbers out of 494,573,440 in total, which is 0.109% (the printed count; my division). The adapter file was 2,172,352 bytes.

#flashcards/projects/concept #flashcards/topic/project-llm What were the three scores before training, after plain LoRA, and after LoRA with replay?::Before: 0 of 24 taught facts, 0 of 12 never taught, 10 of 10 real countries. Plain LoRA: 24, 0 and 5. LoRA with replay: 24, 0 and 10. One run each, greedy decoding, answer contains the right city.

#flashcards/projects/concept #flashcards/topic/project-llm What went wrong in the first LoRA run?
?
It learned the new facts (24 of 24) but forgot real ones: real capitals fell from 10 of 10 to 5 of 10 (Egypt got "Pytun", Canada got "London"). This is catastrophic forgetting in a small form.

#flashcards/projects/concept #flashcards/topic/project-llm How was the forgetting fixed, and what was the catch?
?
Replay: mix 10 other real capitals into the training data. All 10 real answers came back and all 24 new facts were kept. The catch: I did not test how few replay facts are enough, and the task was tiny and made up. One run, one seed.

#flashcards/projects/concept #flashcards/topic/project-llm Why was the test question worded differently from the training questions?
?
So that a good score means the fact was learned, not just the sentence pattern. Three wordings were used in training, and a fourth only for testing.

#flashcards/projects/concept #flashcards/topic/project-llm Why did the loss ignore the question tokens during training?
?
The question tokens were marked -100, which means "ignore". Only the answer tokens count toward the loss, so the model learns to answer and is not trained to repeat questions.

#flashcards/projects/concept #flashcards/topic/project-llm Why is the model unchanged at the very start of LoRA training?
?
The LoRA matrix B starts at zero, so the add-on (alpha over r) times B times A is zero and the output equals the original model's.

#flashcards/projects/cloze #flashcards/topic/project-llm The never-taught made-up countries stayed at ==0 of 12==, because the model cannot guess a made-up fact and instead ==invents a plausible-looking city==.

#flashcards/projects/concept #flashcards/topic/project-llm How much memory would full fine-tuning need compared with LoRA here (arithmetic, not measured)?::Full fine-tuning in 32-bit floats: 494,032,768 × 16 bytes = about 7.90 GB for weights, gradients and two Adam numbers per weight. LoRA: 1.98 GB of frozen weights plus 8.65 MB for the trained part. Full fine-tuning would train about 914 times more numbers.

#flashcards/projects/concept #flashcards/topic/project-llm Did merging the add-on into the model change its answers?::Hardly. The largest difference in the output scores was about 0.0001 (0.000107 in the plain run and 0.000153 in the replay run), and the three sample answers printed after merging were the expected ones.

#flashcards/projects/concept #flashcards/topic/project-llm Is the hand-written LoRA layer the same as the peft library's LoRA?
?
Yes in the maths. With the same A and B numbers inside, the two give exactly the same output scores before merging and after merging (maximum difference 0.0, peft 0.13.2, rank 8, query and value layers). The random starting numbers are not identical for the same seed, probably because peft draws random numbers an extra time (not confirmed in its source).

#flashcards/projects/concept #flashcards/topic/project-llm How did peft training compare with the hand-written layer?::Same recipe, one run each. Taught facts 24 of 24 and untaught 0 of 12 for both. Real countries: plain LoRA 6 of 10 with peft (5 with mine), and with replay 10 of 10 for both. The one-question difference is inside the noise. Run times (about 700 to 1,300 s) varied as much between repeats of the same run as between the two versions, so no speed claim.

#flashcards/projects/concept #flashcards/topic/project-llm Why did the first peft-versus-mine comparison pass for the wrong reason?
?
The test chose the B matrices by a name ending that only matched my layer names, so no B was changed in either model, the add-on did nothing, and equal outputs meant nothing. The printed line "add-on changes the output: False" exposed it.

#flashcards/projects/concept #flashcards/topic/project-llm What did this project not run?
?
QLoRA (no GPU and no 4-bit library installed; only the memory arithmetic is given), other peft versions and settings (only 0.13.2, rank 8, no dropout, one seed per run), other ranks, layers or learning rates, more than one random seed, and harder real tasks.

#flashcards/projects/example #flashcards/topic/project-llm How long did training take on the 8-core CPU for the plain run and the replay run?::701 seconds for 72 examples over 8 epochs (72 steps), and 990 seconds for 102 examples over 8 epochs (104 steps). Each scoring pass of 46 questions took about 32 seconds. A second run from the note's own code took 622 and 925 seconds with identical results.
