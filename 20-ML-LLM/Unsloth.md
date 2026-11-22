---
tags: [LLM, fine-tuning, LoRA, QLoRA, tooling, training, GPU, Triton, memory]
status: growing
level: 4
tier: 3
---

# Unsloth

**Unsloth** is an open-source tool that makes [[Fine-tuning]] a language model faster and lighter on memory, so a model that normally needs a big computer can be trained on a single graphics card.

## 🟢 1. Simple (age 12)
Teaching a big AI a new skill is like moving a huge piano: it normally takes a strong truck. Unsloth is a smart set of straps and wheels that lets the same job be done with a small van, in less time, and the piano ends up in exactly the same place.

**Example:** You want a chatbot that answers questions about your school's rules. Without Unsloth you might need an expensive computer, or have to wait all night. With Unsloth the same lesson can run on a free cloud notebook in about half the time, and the chatbot learns the same things.

## 🔵 2. How it works
1. **Start from a trained model.** You do not train from nothing. You take an existing open model and teach it a little more ([[Fine-tuning]], [[Instruction Tuning]]).
2. **Freeze it and train a small add-on.** Most of the model's numbers stay fixed. Only small extra matrices called adapters learn ([[LoRA]]).
3. **Squeeze the frozen part.** The fixed numbers can be stored in 4 bits instead of 16, which cuts the memory a lot ([[QLoRA]], [[Quantization]]).
4. **Unsloth speeds up the training maths.** According to its authors, it works out the backward pass by hand and rewrites the model's operations as custom GPU programs written in a language called Triton, instead of using the generic versions ([[Backpropagation]], [[GPU]]). They say no approximation is used, so the answers should match the normal method.
5. **It plugs into the usual tools.** It works with the Hugging Face libraries, so you keep your usual training code and change a few lines.
6. **You save the small adapter** (or merge it) and run the model afterwards.
7. **The catch:** the speed and memory gains depend on the model, the GPU and what you compare against. The authors' own numbers range from a small saving to nearly four times, and I could not run Unsloth here to check them (see the end of level 4).

No picture: I searched Wikimedia Commons and found no free, on-topic image for Unsloth, so I added none.

## 🟠 3. Technical detail
- **What it is.** A library by Daniel Han and Michael Han (the blog's byline is "Daniel (Unsloth)"; the README's citation names both and the Unsloth team). The README I read on 16 November 2026 describes three forms: a desktop app, a web UI ("Unsloth Studio") and a code package ("Unsloth Core"). Its core package is Apache 2.0, and some optional parts such as the Studio UI are AGPL-3.0 (README).
- **How it gets its speed (authors' words).** The January 2024 Hugging Face blog says it works by "manually deriving backpropagation steps and rewriting all PyTorch modules into Triton kernels", and that no approximations are made, so accuracy is the same as normal [[QLoRA]]. Triton is a language for writing GPU programs (kernels) from Python.
- **Not from the authors.** One third-party write-up says much of the memory saving comes from recomputing intermediate values in the backward pass instead of storing them (that is [[Gradient Checkpointing]]). That is an outside reading, not an Unsloth statement, and I did not verify it.
- **Methods supported today (README).** "LoRA, QLoRA, full fine tuning, pretraining, RL, GRPO, DPO, and FP8", on NVIDIA, AMD and Intel GPUs and CPUs, on Windows, Linux, WSL and macOS. The 2024 blog was narrower: Llama-type and Mistral-type architectures on NVIDIA GPUs "from GTX 1070 all the way up to H100s". ([[GRPO]], [[Direct Preference Optimization]], [[Reasoning Model]] are the methods people now train with it.)
- **Recommended recipe (docs guide).** Start with a small instruct model (such as Llama 3.1 8B), try LoRA or QLoRA before full fine-tuning, and use `per_device_train_batch_size = 2`, `gradient_accumulation_steps = 4` (see [[Gradient Accumulation]]), `learning_rate = 2e-4` (or 1e-4, 5e-5, 2e-5 for slower, more careful tuning), 1 to 3 epochs (or `max_steps = 60` for a quick test), and `max_seq_length = 2048`. The guide gives no values for LoRA rank or alpha. It also says fine-tuning with its notebooks needs "just 3GB VRAM".
- **The blog's example, as described.** Load `unsloth/mistral-7b-bnb-4bit` with a 2048 context in 4-bit, attach LoRA adapters of rank 16, alpha 16, dropout 0, no bias, on the seven projection matrices (q, k, v, o, gate, up, down) with gradient checkpointing on, then train with TRL's `SFTTrainer` using batch size 2, gradient accumulation 4, 60 steps and the `adamw_8bit` optimizer ([[Adam Optimizer]]).

```python
# NOT RUN. Needs an NVIDIA GPU and `pip install unsloth`. Rebuilt from the settings the
# January 2024 blog lists, using the API names in the blog; names may differ in newer versions.
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/mistral-7b-bnb-4bit", max_seq_length=2048, dtype=None, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0, bias="none", use_gradient_checkpointing=True,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset, dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(per_device_train_batch_size=2, gradient_accumulation_steps=4,
                           max_steps=60, optim="adamw_8bit", output_dir="outputs"))
trainer.train()
```

The next snippet is my own arithmetic. It needs no GPU and no Unsloth, and I ran it. The model sizes are from Mistral 7B's published `config.json` (hidden 4096, intermediate 14336, 32 layers, 8 key-value heads of width 128, so a key or value matrix is 4096 by 1024, vocabulary 32,000).

```python
H, FF, L, KV, V, r = 4096, 14336, 32, 1024, 32000, 16
lora = lambda d_in, d_out: r * (d_in + d_out)           # A is r x d_in, B is d_out x r
per_layer = [lora(H, H), lora(H, KV), lora(H, KV), lora(H, H),    # q, k, v, o
             lora(H, FF), lora(H, FF), lora(FF, H)]               # gate, up, down
total = L * sum(per_layer)
base = 2 * V * H + L * (2 * H * H + 2 * H * KV + 3 * H * FF + 2 * H) + H
print("LoRA parameters:", total)                                   # 41943040
print("base parameters:", base, " share: %.3f%%" % (100 * total / base))
print("weights at 16 bit: %.2f GB, at 4 bit: %.2f GB" % (base * 2 / 1e9, base * .5 / 1e9))
s, p = 1.88, 0.659                  # blog row: Mistral 7b, A100: 1.88x faster, -65.9% VRAM
print("100 min ->", round(100 / s, 1), "min; 10 GB ->", round(10 * (1 - p), 2), "GB")
print("effective batch:", 2 * 4, "; examples seen in 60 steps:", 2 * 4 * 60)
```

It printed `41943040`, `7241732096` and a LoRA share of 0.579%, then 14.48 GB at 16 bit against 3.62 GB at 4 bit (scale constants and the final norm weights ignored), `53.2` minutes and `3.41` GB, then `8` and `480`.

## 🔴 4. Deep / formal
**Speed and memory claims are ratios to a baseline.** If Unsloth is $s$ times faster and uses a fraction $p$ less memory than a baseline:

$$
T_{\text{Unsloth}}=\frac{T_{\text{base}}}{s},\qquad M_{\text{Unsloth}}=(1-p)\,M_{\text{base}}
$$

**Where:**
- $T_{\text{base}}$ and $M_{\text{base}}$ are the time and GPU memory of the same training run with the baseline (in the blog, plain Hugging Face with the stated PyTorch or Transformers version). **Matters:** a claim with no named baseline cannot be checked. The README's headline "2× faster with 70% less VRAM" names none.
- $s>1$ is the speed-up. **Matters:** $s=2$ means the run takes half the time, not 2% less.
- $p$ is the memory reduction between 0 and 1 (the blog writes it as "-65.9%"). **Matters:** it is easy to misread. $p=0.7$ is $1/0.3\approx3.3$ times less memory.

**Example (tiny numbers):** the blog's Mistral 7b row on one A100 40GB: $s=1.88$ and $p=0.659$.
- A run that takes 100 minutes at baseline takes $100/1.88=53.2$ minutes.
- A run that needs 10 GB at baseline needs $10\times(1-0.659)=3.41$ GB, which is $1/0.341=2.93$ times less.
- Meaning: the same job in about half the time and about a third of the memory. The same table's Tiny Llama row on a T4 ($s=3.87$, $p=0.738$) gives $1/0.262=3.82$ times less memory, and its DPO Zephyr row ($s=1.88$, $p=0.116$) saves only 11.6%, so the gain depends a lot on the setting.

**The adapter is tiny, so most memory goes to the frozen base.** For one weight matrix of size $d_{\text{out}}\times d_{\text{in}}$, LoRA adds two matrices of shapes $r\times d_{\text{in}}$ and $d_{\text{out}}\times r$:

$$
P_{\text{LoRA}}=r\,\big(d_{\text{in}}+d_{\text{out}}\big)
$$

**Where:**
- $r$ is the rank (16 in the blog's example). **Matters:** doubling $r$ doubles the adapter size and its optimizer memory.
- $d_{\text{in}}, d_{\text{out}}$ are the matrix's input and output widths. **Matters:** the adapter grows with the sum of the widths, not their product, which is why it is small next to the $d_{\text{in}}d_{\text{out}}$ weights it sits on.
- $P_{\text{LoRA}}$ is the number of trainable numbers for that matrix. **Matters:** only these (and their gradients and optimizer state) take extra training memory.

**Example (tiny numbers):** a $4096\times4096$ matrix at $r=16$: $16\times(4096+4096)=131{,}072$ numbers, against $4096^2=16{,}777{,}216$ in the matrix itself (0.78%). Adding up all seven matrices over the 32 layers of Mistral 7B (the code above) gives $41{,}943{,}040$ trainable numbers out of $7{,}241{,}732{,}096$ (0.58%). Meaning: the training state is under 1% of the model, so the 4-bit base ($3.62$ GB against $14.48$ GB at 16 bits, by the code) dominates the memory, and speed tricks on the forward and backward maths decide the rest.

**The batch you actually train with.** With gradient accumulation:

$$
B_{\text{eff}}=b\times g\times D
$$

**Where:**
- $b$ is the per-device batch size (2 in the blog and docs). **Matters:** it is what must fit in memory at once.
- $g$ is the number of accumulation steps (4). **Matters:** the optimizer steps only after $g$ small batches, so you get a bigger batch without the memory cost ([[Gradient Accumulation]]).
- $D$ is the number of GPUs (1 here). **Matters:** more devices multiply the batch again.

**Example (tiny numbers):** $b=2$, $g=4$, $D=1$: $B_{\text{eff}}=8$. With `max_steps = 60` the run shows $8\times60=480$ examples. Meaning: the blog's 60-step example is a smoke test, not a full training run.

**Honest limits and open questions**
- **The numbers are the authors'.** The blog's benchmark is 59 runs over 4 datasets on Tesla T4 and A100 Colab machines with QLoRA on all linear layers, rank 16 and gradient checkpointing on. I did not reproduce any of it: Unsloth is not installed here and there is no NVIDIA GPU for it.
- **"No accuracy loss" is a vendor claim.** The README and blog say the optimized code makes no approximations. I found no independent check, and I did not run one.
- **A mismatch inside the blog.** Its text says "up to 2.7x faster and up to 74% less memory" against Transformers 4.36 with PyTorch 2.1.1, but its T4 table lists 3.87x for Tiny Llama. I did not resolve this (the table baseline may differ from the text's).
- **The median row is more modest than the headline.** Over the nine table rows the median speed-up is 1.88x and the median memory reduction 39.3% (my arithmetic from the table).
- **The baseline matters.** The blog's own "HF + FA2" column (Hugging Face with [[FlashAttention]] 2) is 0.96x to 1.55x on the A100, so Unsloth's gain over that stronger baseline is smaller (for Mistral 7b, $1.88/1.17=1.61\times$, my arithmetic).
- **The tool has changed a lot.** The 2024 blog covered Llama and Mistral type models on NVIDIA GPUs. The README I read in November 2026 lists many more methods, models and hardware, and its per-notebook figures range from 1.5x to 2x faster and 20% to 80% less memory, again with no named baseline.
- **Not tested:** other GPUs, other model families, long-context training, and quality after training.

**What came before, and how this improved on it:**
- **Before:** fine-tuning with the standard Hugging Face stack (Transformers, PEFT, TRL), with [[LoRA]] adapters on a 16-bit base, or [[QLoRA]] with a 4-bit base ([[Fine-tuning]]).
- **Its limit:** even QLoRA ran slowly and used a lot of GPU memory with generic operations, so 7B-class training needed a good GPU and a long wait.
- **What changed:** the backward pass derived by hand and the model's operations rewritten as Triton kernels, with no approximations claimed, in a drop-in library for the Hugging Face tools (Unsloth team, Hugging Face blog dated 10 January 2024; the page was read in full, the code was not run).
- **How much better:** published figures, not mine. In the blog's tables Unsloth runs 1.55x to 3.87x faster than plain Hugging Face and cuts VRAM by 11.6% to 73.8% across nine settings (median 1.88x and 39.3%). The README's headline, read in November 2026, is "2× faster with 70% less VRAM", with no baseline named. In my own run, only the surrounding arithmetic was checked: the LoRA adapter for Mistral 7B at rank 16 is 41,943,040 numbers, 0.58% of the model.

**Check it yourself:** run the second code block in level 3. It prints every number I computed. The Unsloth code block and every speed and accuracy claim need an NVIDIA GPU and the `unsloth` package, which I did not install.

## Related
[[LoRA]] · [[QLoRA]] · [[Fine-tuning]] · [[Instruction Tuning]] · [[Quantization]] · [[Mixed Precision Training]] · [[Gradient Accumulation]] · [[Adam Optimizer]] · [[Backpropagation]] · [[FlashAttention]] · [[GPU]] · [[GRPO]] · [[Direct Preference Optimization]] · [[Reasoning Model]] · [[Gradient Checkpointing]] · [[Triton]] · [[CUDA]] · [[Kernel Fusion]]

## Sources
- Daniel (Unsloth), *Make LLM Fine-tuning 2x faster with Unsloth and 🤗 TRL*, Hugging Face blog, 10 January 2024 (https://huggingface.co/blog/unsloth-trl): read in full. The benchmark tables, the supported models and hardware, the quoted method sentence and the example's settings come from this page. The code in level 3 is rebuilt from its listed settings, not copied.
- Unsloth GitHub README (https://github.com/unslothai/unsloth), read on 16 November 2026: the three product forms, the licence, the supported methods and hardware, and the headline "2× faster with 70% less VRAM" claim. The BibTeX there credits Daniel Han, Michael Han and the Unsloth team (2023). The README says it works on Windows, Linux, WSL and macOS.
- Unsloth docs, *Fine-tuning LLMs guide* (https://unsloth.ai/docs/get-started/fine-tuning-llms-guide), read on 16 November 2026: the recommended steps, the hyperparameter values and the "3GB VRAM" figure.
- Mistral 7B v0.1 `config.json` on Hugging Face (https://huggingface.co/mistralai/Mistral-7B-v0.1/raw/main/config.json), read for the model sizes used in the arithmetic.
- A third-party write-up (found by search) is the only source for the activation-recomputation remark. I did not read it in full and I marked it as not from the authors.
- Not checked: the Unsloth source code, any independent benchmark, any reproduction of the speed, memory or accuracy claims, and the note was not opened in Obsidian.
- No image: no free, on-topic picture for this note was found on Wikimedia Commons.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm Unsloth:::An open-source tool that makes fine-tuning a language model faster and lighter on GPU memory, so a model that normally needs a big computer can be trained on a single graphics card.

#flashcards/ml/example #flashcards/topic/llm How do the authors say Unsloth gets its speed? (Hugging Face blog, 10 January 2024)::By "manually deriving backpropagation steps and rewriting all PyTorch modules into Triton kernels", with no approximations, so accuracy should match normal QLoRA. Triton is a language for writing GPU programs (kernels) from Python.

#flashcards/ml/concept #flashcards/topic/llm What does Unsloth train, and what stays fixed?::Most of the model's numbers stay frozen (optionally stored in 4 bits), and only small LoRA adapter matrices learn.

#flashcards/ml/example #flashcards/topic/llm What methods and hardware does the README (read 16 November 2026) list?::"LoRA, QLoRA, full fine tuning, pretraining, RL, GRPO, DPO, and FP8", on NVIDIA, AMD and Intel GPUs and CPUs, on Windows, Linux, WSL and macOS. The 2024 blog was narrower: Llama-type and Mistral-type architectures on NVIDIA GPUs from GTX 1070 to H100.

#flashcards/ml/concept #flashcards/topic/llm What is Unsloth's licence?::The core package is Apache 2.0. Some optional parts, such as the Studio UI, are AGPL-3.0 (README).

#flashcards/ml/concept #flashcards/topic/llm What recipe does the Unsloth docs guide recommend?::Start with a small instruct model (such as Llama 3.1 8B), try LoRA or QLoRA before full fine-tuning, and use batch size 2, gradient accumulation 4, learning rate 2e-4 (or 1e-4, 5e-5, 2e-5 for slower, more careful tuning), 1 to 3 epochs (or 60 steps for a quick test) and a 2048 context. The guide gives no LoRA rank or alpha. It says notebooks need "just 3GB VRAM".

#flashcards/ml/concept #flashcards/topic/llm What settings does the blog's example use?::Model `unsloth/mistral-7b-bnb-4bit`, 2048 context, 4-bit. LoRA rank 16, alpha 16, dropout 0, no bias, on the seven matrices q, k, v, o, gate, up, down, with gradient checkpointing. TRL's `SFTTrainer`, batch size 2, gradient accumulation 4, 60 steps, `adamw_8bit`.

#flashcards/ml/formula #flashcards/topic/llm Write how a speed-up and a memory reduction turn into time and memory, and explain each symbol.
?
$$T_{\text{Unsloth}}=\frac{T_{\text{base}}}{s},\qquad M_{\text{Unsloth}}=(1-p)\,M_{\text{base}}$$
- $T_{\text{base}}$, $M_{\text{base}}$: the time and GPU memory of the same run with the baseline (in the blog, plain Hugging Face at a stated version). A claim with no named baseline cannot be checked. The README's headline names none.
- $s>1$: the speed-up. $s=2$ means half the time.
- $p$: the memory reduction between 0 and 1 (the blog writes "-65.9%"). $p=0.7$ is $1/0.3\approx3.3$ times less memory.

#flashcards/ml/example #flashcards/topic/llm Mistral 7b on one A100 40GB: $s=1.88$, $p=0.659$. What happens to a 100-minute run needing 10 GB?::$100/1.88=53.2$ minutes and $10\times(1-0.659)=3.41$ GB, which is $1/0.341=2.93$ times less memory. Tiny Llama on a T4 ($p=0.738$) gives $1/0.262=3.82$ times less, and DPO Zephyr ($p=0.116$) only 11.6%, so the gain depends a lot on the setting.

#flashcards/ml/formula #flashcards/topic/llm Write the number of trainable numbers LoRA adds to one matrix and explain each symbol.
?
$$P_{\text{LoRA}}=r\,\big(d_{\text{in}}+d_{\text{out}}\big)$$
- $r$: the LoRA rank (16 in the blog's example). Doubling $r$ doubles the adapter size and its optimizer memory.
- $d_{\text{in}}$, $d_{\text{out}}$: the matrix's input and output widths. The adapter grows with their sum, not their product, which is why it is small next to the $d_{\text{in}}d_{\text{out}}$ weights it sits on.
- $P_{\text{LoRA}}$: the trainable numbers for that matrix. Only these (and their gradients and optimizer state) take extra training memory.

#flashcards/ml/example #flashcards/topic/llm A $4096\times4096$ matrix at $r=16$, and then all of Mistral 7B. How many trainable numbers?::One matrix: $16\times(4096+4096)=131{,}072$ against $4096^2=16{,}777{,}216$ (0.78%). All seven matrices over 32 layers: $41{,}943{,}040$ out of $7{,}241{,}732{,}096$ (0.58%), computed from Mistral 7B's published config. The training state is under 1% of the model, so the 4-bit base (3.62 GB against 14.48 GB at 16 bits) dominates memory.

#flashcards/ml/formula #flashcards/topic/llm Write the effective batch size with gradient accumulation and explain each symbol.
?
$$B_{\text{eff}}=b\times g\times D$$
- $b$: the per-device batch size (2 in the blog and docs), what must fit in memory at once.
- $g$: the number of accumulation steps (4). The optimizer steps only after $g$ small batches, so you get a bigger batch without the memory cost.
- $D$: the number of GPUs (1 here). More devices multiply the batch again.

#flashcards/ml/example #flashcards/topic/llm $b=2$, $g=4$, $D=1$ and 60 steps. What are $B_{\text{eff}}$ and the examples seen?::$B_{\text{eff}}=2\times4\times1=8$, and $8\times60=480$ examples. The blog's 60-step example is a smoke test, not a full training run.

#flashcards/ml/concept #flashcards/topic/llm Is "no accuracy loss" independently verified?::No. It is a vendor claim from the README and blog ("no approximations"). The note found no independent check and did not run one.

#flashcards/ml/example #flashcards/topic/llm What mismatch is inside the January 2024 blog?::Its text says "up to 2.7x faster and up to 74% less memory" against Transformers 4.36, but its T4 table lists 3.87x for Tiny Llama. The note did not resolve it (the baselines may differ).

#flashcards/ml/concept #flashcards/topic/llm How modest is the typical gain in the blog's nine table rows? (the note's arithmetic)::Speed-up from 1.55x to 3.87x (median 1.88x) and VRAM reduction from 11.6% to 73.8% (median 39.3%). Against the blog's stronger "HF + FlashAttention 2" baseline the gain is smaller (for Mistral 7b, $1.88/1.17=1.61\times$).

#flashcards/ml/concept #flashcards/topic/llm What setup did the blog's benchmark use?::59 runs over 4 datasets on Tesla T4 and A100 Google Colab machines, QLoRA on all linear layers, rank 16, gradient checkpointing on. The note reproduced none of it.

#flashcards/ml/concept #flashcards/topic/llm What did the note actually run?::Only its own arithmetic (LoRA parameters, 16-bit against 4-bit weight memory, speed and memory ratios, effective batch). The Unsloth code block was not run, because Unsloth is not installed and there is no NVIDIA GPU.

#flashcards/ml/concept #flashcards/topic/llm Which third-party claim is not from the authors?::That much of the memory saving comes from recomputing intermediate values in the backward pass instead of storing them (gradient checkpointing). It is an outside reading and was not verified.

#flashcards/ml/example #flashcards/topic/llm Why does the baseline matter in a claim like "2x faster with 70% less VRAM"?::Without a named baseline (which library, which version, which GPU, which model) the ratio cannot be checked or reproduced.

#flashcards/ml/cloze #flashcards/topic/llm Unsloth works with the ==Hugging Face== libraries (Transformers, PEFT, TRL), so you keep your usual training code and change a few lines.
