---
tags: [LLM, inference, local, tools, CPU, API]
status: growing
level: 4
tier: 2
---

# Ollama

**Ollama** is a program that downloads open language models and runs them on your own computer. You type one command to get a model and chat with it, and other programs can talk to it through a small web address on your machine.

## 🟢 1. Simple (age 12)
Running an AI model by hand is like building a toy from a pile of parts, with a long manual. Ollama is like an app store with a play button: you pick a model, it fetches the parts, puts the toy together and lets you play with it. Everything stays on your own computer, so no one else sees what you type.

**Example:** You type "ollama run" and the name of a model. The first time, it downloads the model, which can take a few minutes. After that, a chat prompt appears and you can ask questions with the internet off.

## 🔵 2. How it works
1. **Ollama runs as a small server** on your computer. It listens at the address `localhost` on port 11434, so only programs on your machine can reach it.
2. **`ollama pull name` downloads a model** from Ollama's library. A model is stored as a few pieces: the big file of numbers (a [[GGUF]] file), a short system message, a chat-layout recipe (the [[Chat Template]]) and the licence.
3. **`ollama run name` starts a chat.** The server loads the model into memory, and your question goes in.
4. **The server builds the text the model expects.** It wraps your messages in the model's chat layout and turns them into tokens ([[Tokenization]]).
5. **A fast engine writes the answer one token at a time.** Ollama's README lists [[llama.cpp]] as a backend ([[LLM Inference]]), and I did not check which engine ran my model.
6. **The answer streams back.** You can also ask the server through its web API, and it replies with the text and some timing numbers.
7. **The model stays in memory for a few minutes** after the last question, so the next one is quick. Then it is unloaded.

![[ollama-terminal-llama3.png]]
*Ollama in a Linux terminal: the command `ollama run llama3`, a question ("why is Wikipedia the best encyclopedia in the world?"), a numbered answer, and the prompt `>>> Send a message (/? for help)` waiting for the next question. Screenshot by MGeog2022, [Commons file page](https://commons.wikimedia.org/wiki/File:Ollama-running-llama3-in-linux.png), CC BY 4.0.*

## 🟠 3. Technical detail
- **Where models live.** A model is a *manifest* (a small JSON file) that lists *layers*, and each layer is a file named by its hash. On this PC the models folder is set by the environment variable `OLLAMA_MODELS` (here `D:\Ollama\.ollama\models`). The manifest for `qwen2.5:0.5b` lists four layers: the model (397,807,936 bytes, a GGUF file), a system message (68 bytes), the template (1,482 bytes) and the licence (11,343 bytes). The layout is similar to how Docker stores images (my observation, not something I checked in the docs).
- **Its own template.** `ollama show --modelfile qwen2.5:0.5b` printed a `TEMPLATE` written in Go's template language, with fields like `.System` and `.Messages`. It is not the same text as the Jinja template stored inside the GGUF file ([[Chat Template]]).
- **The Modelfile** is a small recipe file for building your own model. The docs list: `FROM` (a model name, or a path to a GGUF file or a Safetensors folder), `PARAMETER`, `TEMPLATE`, `SYSTEM`, `LICENSE`, `MESSAGE` (example turns), `REQUIRES` and `CAPABILITY`. The docs give these defaults: temperature 0.8, `num_ctx` 2048, `top_k` 40, `top_p` 0.9, `seed` 0, and `num_predict` -1 (no limit). **On this PC `ollama ps` showed a context of 32768 for the model, not 2048, and I did not find out why** (the docs page may describe different defaults from my version, 0.21.2).
- **The web API** (from the docs, through a page summary): `POST /api/chat` takes `model`, `messages`, `stream` and `options`; `POST /api/generate` takes a single `prompt`; `GET /api/tags` lists installed models; `POST /api/show` gives details; `GET /api/ps` lists loaded models. `keep_alive` is how long the model stays loaded and defaults to 5 minutes. When streaming is off, the reply carries counts and durations (all in nanoseconds), from which you get the speed (see the rule below).

Tokens per second = tokens written (eval_count) ÷ time spent writing them (eval_duration, in nanoseconds), times a billion to turn nanoseconds into seconds.

**Terms:**
- eval_count: how many tokens the model wrote in the answer.
- eval_duration: the time spent writing them, in nanoseconds (billionths of a second).
- 10^9: turns nanoseconds into seconds.

**Matters:** it measures only the writing. The load time and the time to read your prompt are other fields (`load_duration`, `prompt_eval_duration`), so if you divide by the total time instead, the speed looks lower than the model's real writing speed.

**Example (tiny numbers):** a reply of 8 tokens with `eval_duration` 207,630,900 ns. That is 0.2076 s, and 8/0.2076=38.5 tokens per second. A 100-token reply at the same speed would take about 2.6 s.

**`ollama_demo.py`** (standard library only, plus the `ollama` command)
```python
"""Talk to a local Ollama server with only the standard library, then build and delete a model from a Modelfile."""
import json, os, subprocess, tempfile, urllib.request

URL = "http://localhost:11434"


def call(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(URL + path, data, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


print("installed:", [(m["name"], m["details"]["quantization_level"]) for m in call("/api/tags")["models"]])

ask = {"model": "qwen2.5:0.5b", "stream": False, "options": {"temperature": 0, "seed": 1, "num_predict": 40},
       "messages": [{"role": "user", "content": "Name one colour."}]}
r = call("/api/chat", ask)
print("answer:", r["message"]["content"])
print("eval_count", r["eval_count"], "| eval_duration", r["eval_duration"], "ns | tokens/s =", round(r["eval_count"] / r["eval_duration"] * 1e9, 1))

modelfile = 'FROM qwen2.5:0.5b\nSYSTEM "Answer like a pirate, in one short sentence."\nPARAMETER temperature 0\n'
path = os.path.join(tempfile.mkdtemp(), "Modelfile")
open(path, "w").write(modelfile)
subprocess.run(["ollama", "create", "pirate-test", "-f", path], check=True, capture_output=True)
ask.update(model="pirate-test")
print("pirate:", call("/api/chat", ask)["message"]["content"])
subprocess.run(["ollama", "rm", "pirate-test"], check=True, capture_output=True) # clean up the model I made
print("after rm:", [m["name"] for m in call("/api/tags")["models"]])
print("loaded now:", [(m["name"], m["size"]) for m in call("/api/ps")["models"]])
```
What it printed:
```
installed: [('qwen2.5:0.5b', 'Q4_K_M'), ('gemma4:31b-cloud', ''), ('nomic-embed-text:latest', 'F16')]
answer: One color is called "red".
eval_count 8 | eval_duration 207630900 ns | tokens/s = 38.5
pirate: Red is one color.
after rm: ['qwen2.5:0.5b', 'gemma4:31b-cloud', 'nomic-embed-text:latest']
loaded now: [('qwen2.5:0.5b', 794512896)]
```
What to look for: the speed formula gives 38.5 from the reply's own counters (a second run of the same script printed 37.3, from 214,493,100 ns); the Modelfile created a model named `pirate-test` and I removed it, which the last-but-one line confirms. The "pirate" model did not talk like a pirate (the 0.5B model gave a plain answer), and I did not check whether the system line reached the prompt. The loaded size, 794,512,896 bytes, is about twice the file size, and I did not find out why (my guess is the memory for the conversation, which I did not check).

## 🔴 4. Deep / formal
- **What Ollama adds on top of the engine.** The engine alone runs a model file. Ollama adds the download and storage, the per-model template and settings, loading and unloading, one web API for all models and a command-line chat. That is a convenience layer, so any speed difference to a bare engine would come from settings (context size, threads, quantization), not from this layer (my reasoning, not tested).
- **Privacy.** The model runs on your machine, so your prompts do not leave it. The docs summary I read also mentions cloud models, and my list shows one that looks like it (`gemma4:31b-cloud`, with no local size and a `-cloud` tag). I did not check how it runs, but if it is a cloud model your prompts leave your machine, so check before sending private text.
- **Speed.** The same Qwen2.5-0.5B ran at 39 to 50 tokens/s through Ollama and 6.0 to 6.3 tokens/s through float32 transformers on this PC ([[Local CPU Chatbot]]). The Ollama file is a mix averaging 6.345 bits per weight ([[GGUF]]), and the transformers copy was 32 bits, so the two do not read the same number of bytes per token. I did not separate the engine from the quantization.
- **Where the speed number can mislead.** On the second call with the same prompt, Ollama showed a prompt-read time of 0.02 s, so it had probably cached the prompt (a guess). A fair test needs a fresh prompt each time.
- **Limits of this note.** I did not read Ollama's source, test the cloud models, the Safetensors import, tool calling, image input, GPU use or other computers on the network, and I only used the version installed here (0.21.2).

**What came before, and how this improved on it:**
- **Before:** to run an open model you installed a Python stack, found the weights and the tokenizer files, wrote a script to load and chat, and handled the chat format yourself.
- **Its limit:** my own version of that took 8 downloaded files (0.99 GB), about 40 lines of code, and ran at 6.0 tokens/s in float32 ([[Local CPU Chatbot]]).
- **What changed:** one command to fetch a ready-made, quantized model, a stored template and settings per model, and one web API. Ollama's first public release is from July 2023 and it is by Jeffrey Morgan and Michael Chiang (from search summaries: a Wikipedia-based result for the authors, and an unofficial page for the July date, so treat both as unchecked).
- **How much better:** one command (`ollama pull qwen2.5:0.5b`) fetched 397 MB, against 8 files and 0.99 GB, and the model answered 6 to 8 times faster (39 to 50 against 6.0 to 6.3 tokens/s, three runs each). Part of that is the engine and part is the smaller numbers, and I did not split them.

## Related
[[llama.cpp]] · [[GGUF]] · [[Chat Template]] · [[Quantization]] · [[LLM Inference]] · [[LLM]] · [[Decoding Strategies]] · [[FastAPI]] · [[Docker]] · [[Local CPU Chatbot]] · [[Vault Search with RAG]] · [[Context Window]] · [[Structured Output]]

## Sources
- The Ollama README (https://github.com/ollama/ollama), the Modelfile reference and the API reference in its `docs` folder (`modelfile.mdx`, `api.md`), all read on 17 November 2026 through page summaries, not word for word. The Modelfile defaults and the field list are from those summaries.
- Run on this PC: Ollama 0.21.2 (already installed), the model `qwen2.5:0.5b` (397 MB), Python 3.9.6. The manifest layers, the template text, `ollama ps` and every number in the code output are from my own runs.
- From search summaries only (not opened): the July 2023 first release, and the authors' names.
- Image: "Ollama-running-llama3-in-linux.png" by MGeog2022, CC BY 4.0, https://commons.wikimedia.org/wiki/File:Ollama-running-llama3-in-linux.png. I viewed it before writing the caption.
- The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm-serving Ollama:::A program that downloads open language models and runs them on your own computer, with a one-line command to chat and a local web API on port 11434.

#flashcards/ml/image #flashcards/topic/llm-serving ![[ollama-terminal-llama3.png]]
What does this picture show?
?
Ollama in a Linux terminal: the command `ollama run llama3`, a question about Wikipedia, a numbered answer, and the prompt `>>> Send a message (/? for help)` waiting for the next question. Screenshot by MGeog2022, CC BY 4.0 (Wikimedia Commons).

#flashcards/ml/concept #flashcards/topic/llm-serving What is an Ollama model made of?::A manifest that lists layers, each a file named by its hash. For `qwen2.5:0.5b`: the model (a GGUF file, 397,807,936 bytes), a system message (68 bytes), a template (1,482 bytes) and the licence (11,343 bytes).

#flashcards/ml/concept #flashcards/topic/llm-serving Which web address and port does Ollama use, and what is the default `keep_alive`?::`http://localhost:11434`, and 5 minutes (from the API docs summary). After that the model is unloaded from memory.

#flashcards/ml/concept #flashcards/topic/llm-serving Write how to get tokens per second from an Ollama reply, and explain each symbol.
?
Tokens per second = tokens written (eval_count) ÷ time spent writing them (eval_duration, in nanoseconds), times a billion to turn nanoseconds into seconds.
- eval_count: how many tokens the model wrote in the answer.
- eval_duration: the time spent writing them, in nanoseconds.
- 10^9: turns nanoseconds into seconds.
- It measures only the writing. Load time and prompt-reading time are separate fields, so dividing by the total time would make the speed look lower.

#flashcards/ml/example #flashcards/topic/llm-serving A reply had `eval_count` 8 and `eval_duration` 207,630,900 ns. What is the speed?::207,630,900 ns is 0.2076 s, and 8 / 0.2076 = 38.5 tokens per second (my arithmetic, matching the demo's output).

#flashcards/ml/concept #flashcards/topic/llm-serving What can you put in a Modelfile?::`FROM` (a model name, a GGUF path or a Safetensors folder), `PARAMETER`, `TEMPLATE`, `SYSTEM`, `LICENSE`, `MESSAGE`, `REQUIRES` and `CAPABILITY` (from the docs summary).

#flashcards/ml/concept #flashcards/topic/llm-serving What mismatch about context size did I find, and did I solve it?::The docs summary gives a default `num_ctx` of 2048, but `ollama ps` showed 32768 for the model here (Ollama 0.21.2). I did not find out why.

#flashcards/ml/example #flashcards/topic/llm-serving Why did the Ollama prompt-reading time show 0.02 s in my benchmark?::The warm-up call used the same prompt, so Ollama had probably cached it (a guess). A fair test needs a fresh prompt each time.

#flashcards/ml/example #flashcards/topic/llm-serving How did Ollama compare with float32 transformers for Qwen2.5-0.5B on this PC?::One command and 397 MB against 8 files and 0.99 GB. Speed was 39 to 50 tokens/s against 6.0 to 6.3 (three runs each). I did not split how much came from the engine and how much from the smaller numbers.

#flashcards/ml/concept #flashcards/topic/llm-serving Why can a cloud model in the Ollama list break the "everything stays on your computer" idea?::A model like `gemma4:31b-cloud` has no local size in the list and a `-cloud` tag, which suggests it runs elsewhere (I did not check). If so, prompts sent to it leave your machine, so check before sending private text.
