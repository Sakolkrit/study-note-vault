---
tags: [LLM, library, Python, transformers, Hugging Face, inference, PyTorch, local]
status: growing
level: 4
tier: 2
---

# Hugging Face Transformers

**Hugging Face Transformers** (the Python package `transformers`) is a free library that loads thousands of ready-made AI models and lets you use them with a few lines of code. You name a model, and it fetches the files, builds the model and gives you a tokenizer to go with it.

## 🟢 1. Simple (age 12)
Imagine a huge toy library where every toy has the same plug. You can borrow a robot, a talking parrot or a drawing machine, and they all fit the same socket, so you only learn how to plug in once. Transformers is that socket for AI models: one way to load them and one way to ask them things.

**Example:** You want a model to finish the sentence "The cat sat on the". You tell the library the model's name, give it the sentence, and it writes the rest. If you want a different model tomorrow, you change only the name.

## 🔵 2. How it works
Three objects do the work, and almost every script starts by making them.
1. **The config** is a small settings file (a JSON file). It says how big the model is: how many layers, how wide, how many words it knows. It is cheap to load.
2. **The tokenizer** turns your text into numbers and back ([[Tokenization]]). A [[Transformer]] model cannot read letters, only numbers, so this step comes first.
3. **The model** holds the learned numbers, called weights, and runs on [[PyTorch]]. You give it the numbers from the tokenizer and it gives back one score for every word it knows: how likely that word is to come next.
4. **The scores turn into probabilities** with [[Softmax]]. The model can pick the top one, or pick randomly in proportion to the probabilities ([[Decoding Strategies]]).
5. **`generate` repeats this.** It adds the chosen word to the text and asks again, over and over, until it has enough words or the model says "stop" ([[LLM Inference]]).
6. **`pipeline` is a shortcut** that does steps 2 to 5 in one call, for people who just want an answer.
7. **For chat models, a template comes first.** The tokenizer wraps your messages in the layout the model was trained on ([[Chat Template]]).

![[hugging-face-transformers-logo.png]]
*The logo of the Hugging Face Transformers library: a smiling yellow face with two open hands, the Hugging Face emoji. The library's name is meant to sit beside it, but in the copy I viewed only the emoji showed (the text seems to be white on a see-through background), so I describe only what I saw. Image from the library's GitHub page, uploaded to [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Hugging_Face_Transformers_logo.png), Apache License 2.0.*

## 🟠 3. Technical detail
- **The three `Auto` classes.** `AutoConfig`, `AutoTokenizer` and `AutoModelForCausalLM` read the config file in a model folder and pick the right Python class. For my Qwen2.5 folder the config says `model_type: qwen2`, and the class that came back was `Qwen2ForCausalLM`. "Causal" means the model only looks at words to the left, which is what a text writer needs.
- **`from_pretrained(path_or_name)`** loads from a folder on disk, as I did. Given a model name from the Hugging Face website instead, it downloads the files first and keeps a copy (from my memory of the docs; I did not test it, because I had downloaded the files with `huggingface_hub` earlier, see [[Local CPU Chatbot]]).
- **`torch_dtype`** chooses the number type for the weights. The file stores `bfloat16` (2 bytes per number). I asked for `float32` (4 bytes), which is more exact on a CPU and twice the memory. The rule below gives the size.

Memory for the weights in bytes = number of weights (N) × bytes per weight (b).

**Terms:**
- N is the number of parameters (weights) in the model.
- b is how many bytes one number takes: 4 for float32, 2 for bfloat16 or float16, 1 for an 8-bit number ([[Quantization]]).
- bytes is the memory for the weights alone.

**Matters:** if you forget this, the model may not fit in memory and the program crashes while loading. The real need is higher, because the text being processed ([[KV Cache]]) and the code itself also use memory.

**Example (tiny numbers):** a model with N=1 billion weights. In float32 (b=4) that is 4 GB. In bfloat16 (b=2) it is 2 GB. With 8-bit numbers (b=1) it is 1 GB. So halving the bytes per number halves the memory, and the same model that needs 4 GB could fit in 1 GB.

**`hf_tour.py`** (needs `transformers` and `torch`, and the model folder from the [[Local CPU Chatbot]] project)
```python
"""A tour of Hugging Face transformers: config, tokenizer, model, one forward pass, generate, pipeline. CPU only."""
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, pipeline

PATH = "qwen2.5-0.5b-instruct" # the folder downloaded in the Local CPU Chatbot project

cfg = AutoConfig.from_pretrained(PATH) # 1. the config: just a small JSON file
print("config:", cfg.model_type, "| layers", cfg.num_hidden_layers, "| width", cfg.hidden_size, "| vocab", cfg.vocab_size)

tok = AutoTokenizer.from_pretrained(PATH) # 2. the tokenizer: text <-> numbers
ids = tok("The cat sat on the", return_tensors="pt")
print("tokens:", tok.convert_ids_to_tokens(ids["input_ids"][0]), "->", ids["input_ids"][0].tolist())

model = AutoModelForCausalLM.from_pretrained(PATH, torch_dtype=torch.float32).eval() # 3. the model: the weights
n = sum(p.numel() for p in model.parameters())
print(f"parameters: {n:,} | float32 weights: {n * 4 / 1e9:.3f} GB | class: {type(model).__name__}")

with torch.no_grad(): # 4. one forward pass = scores for the next token
    logits = model(**ids).logits[0, -1] # shape (vocab,): one score per possible next token
probs = torch.softmax(logits, dim=-1)
top = torch.topk(probs, 3)
print("logits shape:", tuple(logits.shape), "| top 3 next tokens:",
      [(tok.decode(i), round(p.item(), 3)) for p, i in zip(top.values, top.indices)])

out = model.generate(**ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id) # 5. generate = repeat step 4
by_hand = tok.decode(out[0], skip_special_tokens=True)
print("generate:", repr(by_hand))

pipe = pipeline("text-generation", model=model, tokenizer=tok) # 6. pipeline = steps 2 to 5 in one call
piped = pipe("The cat sat on the", max_new_tokens=8, do_sample=False)[0]["generated_text"]
print("pipeline:", repr(piped), "| same as generate:", piped == by_hand)
```
What it printed (transformers 4.46.3, torch 2.8.0+cpu, Python 3.9.6):
```
config: qwen2 | layers 24 | width 896 | vocab 151936
tokens: ['The', 'Ġcat', 'Ġsat', 'Ġon', 'Ġthe'] -> [785, 8251, 7578, 389, 279]
parameters: 494,032,768 | float32 weights: 1.976 GB | class: Qwen2ForCausalLM
logits shape: (151936,) | top 3 next tokens: [(' couch', 0.081), (' bed', 0.057), (' window', 0.052)]
generate: 'The cat sat on the couch, and the dog was sitting next'
pipeline: 'The cat sat on the couch, and the dog was sitting next' | same as generate: True
```
What to look for: the sentence became 5 tokens, and the mark `Ġ` in front of a token stands for a space before the word (how this tokenizer writes it, see [[Byte-Pair Encoding]]). One forward pass gave 151,936 scores, one per word the model knows. The best next word was " couch" with only 8.1% probability, so the model is far from sure, and greedy writing (`do_sample=False`) starts the same way (" couch"). The `generate` call and the `pipeline` call wrote exactly the same text. The weights came to 494,032,768 × 4 = 1,976,131,072 bytes, and the same count in bfloat16 is 988,065,536 bytes, which is 32,288 bytes less than the 988,097,824-byte file (my arithmetic; the extra bytes are the file's 8-byte length plus its JSON header, checked in [[Safetensors]]). I filtered warning lines out of the output with `grep` and did not read them. The first run took over two minutes in total on this CPU, and I did not time the parts.

## 🔴 4. Deep / formal
- **One library, many model families.** The idea is that one API fits many architectures, so you can swap a model by changing its name. I ran only one model (Qwen2.5-0.5B-Instruct), so I did not test the swap.
- **Weights and code are separate.** The weights are one file (`model.safetensors`, a plain container of named number tables, see [[Safetensors]]) and the code is the Python class. The class is chosen from the config. A file that cannot run code is safer to download than an old-style pickle file (I showed in [[Safetensors]] that a pickle can run code on load).
- **Where it sits in this vault's tool chain.** Transformers is the "full" route: float32 or bfloat16, PyTorch, easy to inspect and to train. [[GGUF]] files run in [[llama.cpp]] and [[Ollama]] for speed on a CPU. For the same Qwen2.5-0.5B the speed on this PC was 6.0 to 6.3 tokens/s here against 39 to 50 in Ollama ([[Local CPU Chatbot]]), and I did not split how much came from the engine and how much from the smaller numbers.
- **What this library is best for, and what I did not test.** Looking inside a model (the logits above), changing generation settings, fine-tuning ([[Instruction Tuning]], [[Transfer Learning]]) and running research code. I did not test training, GPU use, 8-bit or 4-bit loading, other model families, other tasks such as classification or translation, or the model website itself.
- **Version drift.** Names and options change between versions (the project note on [[Local CPU Chatbot]] ran on 4.46.3). Check the docs for the version you installed.

**What came before, and how this improved on it:**
- **Before:** the code for each new model was released separately by its authors, in its own repository and style. [[BERT]] came out in 2018 with its own code, and the library began as a PyTorch port of it (from my memory: first called `pytorch-pretrained-BERT`, later renamed; I did not check the dates or names).
- **Its limit:** to try two models you had to learn two code bases, two ways of loading weights and two tokenizers.
- **What changed:** one shared interface and a shared hub of pretrained weights. The library's paper, "HuggingFace's Transformers: State-of-the-art Natural Language Processing" (arXiv 1910.03771, first posted 9 October 2019, by Thomas Wolf, Lysandre Debut, Victor Sanh and 19 others, checked on the arXiv page), says it offers Transformer architectures under a unified API with a curated collection of pretrained models, built to be extensible for researchers, simple for practitioners and fast and robust in industry. The abstract gives no model count, and I did not look for one.
- **How much better:** the measured part in this note's code is small. Steps 2 to 5 of the tour (tokenizer, model, forward pass, `generate`) take about a dozen lines including the printing, and `pipeline` did the same job in one call with identical text (`same as generate: True`). A comparison with the pre-library way needs a second code base, which I did not run.

## Related
[[Transformer]] · [[Tokenization]] · [[Chat Template]] · [[BERT]] · [[PyTorch]] · [[Softmax]] · [[Decoding Strategies]] · [[LLM Inference]] · [[Quantization]] · [[GGUF]] · [[llama.cpp]] · [[Ollama]] · [[Instruction Tuning]] · [[Transfer Learning]] · [[Byte-Pair Encoding]] · [[KV Cache]] · [[Safetensors]] · [[Hugging Face Hub]] · [[LLM]] · [[Local CPU Chatbot]]

## Sources
- Run on this PC: transformers 4.46.3, torch 2.8.0+cpu, Python 3.9.6, the folder `qwen2.5-0.5b-instruct` from the [[Local CPU Chatbot]] project. Every number in the output above is from my own run, and the byte arithmetic is mine.
- arXiv 1910.03771, "HuggingFace's Transformers: State-of-the-art Natural Language Processing" (https://arxiv.org/abs/1910.03771). I read only the abstract page (through a page summary), not the paper body.
- From my memory, not checked: the library's earlier names and the loading-by-name behaviour.
- Image: "Hugging Face Transformers logo.png", from the library's GitHub page, Apache License 2.0, https://commons.wikimedia.org/wiki/File:Hugging_Face_Transformers_logo.png. I viewed it before writing the caption.
- The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving Hugging Face Transformers:::A free Python library (package `transformers`) that loads ready-made AI models with one shared interface: a config, a tokenizer and a model.

#flashcards/ml/image #flashcards/topic/llm-serving ![[hugging-face-transformers-logo.png]]
What does this picture show?
?
The logo of the Hugging Face Transformers library: a smiling yellow face with two open hands (the Hugging Face emoji). The library name is meant to be beside it, but only the emoji showed in the copy I viewed. From the library's GitHub page, Apache License 2.0 (Wikimedia Commons).

#flashcards/ml/concept #flashcards/topic/llm-serving What three objects do almost every Transformers script start with, and what does each do?
?
- The config: a small JSON file with the model's size settings.
- The tokenizer: turns text into numbers and back.
- The model: the weights, which turn the numbers into scores for the next word.

#flashcards/ml/concept #flashcards/topic/llm-serving What does `generate` do that one forward pass does not?
?
One forward pass gives scores for the next token only. `generate` picks a token, adds it to the text and asks again, repeating until it has enough tokens or the model writes its stop token.

#flashcards/ml/concept #flashcards/topic/llm-serving What does `pipeline` do?::It runs the tokenizer, the model and the writing loop in one call. In my run it wrote exactly the same text as the longer `generate` code.

#flashcards/ml/concept #flashcards/topic/llm-serving How much memory do a model's weights need, and what does each symbol mean?
?
Memory for the weights in bytes = number of weights (N) × bytes per weight (b).
- N: the number of parameters (weights).
- b: bytes per number: 4 for float32, 2 for bfloat16 or float16, 1 for 8-bit.
- bytes: memory for the weights alone.
- Forgetting it can make loading crash for lack of memory. The real need is higher, because the text being processed and the code also use memory.

#flashcards/ml/example #flashcards/topic/llm-serving A model has 1 billion weights. How much memory do the weights need in float32, bfloat16 and 8-bit?::1 billion × 4 = 4 GB, × 2 = 2 GB, × 1 = 1 GB (my arithmetic). Halving the bytes per number halves the memory.

#flashcards/ml/example #flashcards/topic/llm-serving Qwen2.5-0.5B has 494,032,768 parameters. How big are its weights in float32?::494,032,768 × 4 = 1,976,131,072 bytes, about 1.976 GB (my arithmetic, matching the demo's output). In bfloat16 it is 988,065,536 bytes.

#flashcards/ml/example #flashcards/topic/llm-serving What did one forward pass give for "The cat sat on the" with Qwen2.5-0.5B?::151,936 scores, one per word the model knows. The top three next tokens were " couch" (0.081), " bed" (0.057) and " window" (0.052), so the model was far from sure.

#flashcards/ml/cloze #flashcards/topic/llm-serving In a Transformers script, the ==tokenizer== turns text into numbers and the ==model== turns the numbers into scores for the next word.

#flashcards/ml/concept #flashcards/topic/llm-serving What does the mark `Ġ` in front of a token mean in the Qwen tokenizer output?::A space before the word (how this tokenizer writes it). For example `Ġcat` is " cat".

#flashcards/ml/concept #flashcards/topic/llm-serving What does the paper behind the library say the library offers? (abstract only)::Transformer architectures under a unified API, with a curated collection of pretrained models, built to be extensible for researchers, simple for practitioners and fast and robust in industry. The abstract gives no model count.

#flashcards/ml/concept #flashcards/topic/llm-serving What did I not test about Hugging Face Transformers?::Swapping to other models, loading by model name, training, GPU use, 8-bit or 4-bit loading, other tasks such as classification or translation, and the model website. I ran one model only (Qwen2.5-0.5B-Instruct).

#flashcards/ml/example #flashcards/topic/llm-serving How did float32 Transformers compare with Ollama for Qwen2.5-0.5B on this PC?::6.0 to 6.3 tokens/s against 39 to 50 tokens/s (three runs each, from the Local CPU Chatbot project). I did not split how much came from the engine and how much from the smaller numbers.
