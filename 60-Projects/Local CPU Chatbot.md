---
tags: [project, python, llm, chatbot, cpu, ollama, transformers, streamlit]
status: done
stack: Python 3.9, Hugging Face transformers, PyTorch (CPU), Ollama, Streamlit
goal: run a small chat language model on this laptop with no GPU, talk to it in a terminal that remembers the conversation, and measure how fast two ways of running it are
---

# Local CPU Chatbot

> **In one line:** download a small open [[LLM]] (0.5 billion numbers), chat with it on an ordinary laptop CPU with no internet and no GPU, and find out why one way of running it is 6 to 8 times faster than another.

## Goal
Done means four checks pass: (1) the model files are on disk and match the published size, (2) a terminal chatbot answers, remembers a fact from an earlier turn, and forgets it after a reset, (3) the same model answers through [[Ollama]], and (4) both routes are timed on the same prompt in tokens per second.

## Result in short
- **The model:** Qwen2.5-0.5B-Instruct (Apache-2.0 licence), 494.03 million parameters, 24 layers, width 896, 32,768-token context (config file and `ollama show`). The file is 988,097,824 bytes, the same as the size the Hugging Face API listed.
- **The chat works.** In my first scripted run the bot answered "Hello there!" in 0.9 s. When I asked "What is my name?" it said "Your name is Dev." (1.1 s), which I had told it one turn earlier. After `reset` the same question got "I'm not named in any way. How can I assist you today?" (2.8 s). The answers are sampled (temperature 0.7), so yours will differ (my second scripted run said "Hello there! How can I help you today?" and, after the reset, "I am not named. I exist solely to assist you with your inquiries. How may I assist you today?"; it still knew the name before the reset).
- **Speed on this laptop (8 logical cores, 39.7 GB RAM; PyTorch is the CPU-only build, and `ollama ps` showed "100% CPU" for the Ollama model), 64 new tokens, greedy, same 40-token prompt:**

| Route | Weights in memory | Run 1 | Run 2 | Run 3 (code re-extracted from this note) |
|---|---|---|---|---|
| transformers, float32 | about 2.0 GB (my arithmetic) | 6.0 tokens/s | 6.0 tokens/s | 6.3 tokens/s |
| Ollama `qwen2.5:0.5b` (quantized, labelled Q4_K_M) | 397 MB (file size) | 50.5 tokens/s | 42.9 tokens/s | 39.1 tokens/s |

  So Ollama was about 6 to 8 times faster (6.2 to 8.4 from the three runs, my arithmetic), and its own speed moved by more than 10 tokens/s from run to run. Read the "why" in the section below, because I did not isolate which part of the gap is the smaller quantized weights and which part is the faster program.
- **Not done:** the Streamlit web page was never opened in a browser, only started (see "What I did not run").

## What you need
Versions I used, in my existing Python environment on Windows 11 (**not a fresh virtual environment**):

| Tool | Version |
|---|---|
| Python | 3.9.6 |
| torch | 2.8.0+cpu |
| transformers | 4.46.3 |
| streamlit | 1.25.0 |
| Ollama | 0.21.2 (already installed) |

Downloads I made, both approved first: the Hugging Face model files (about 0.99 GB) and the Ollama model `qwen2.5:0.5b` (397 MB). Nothing else was installed. You need roughly 3 GB of free RAM for the float32 route. Everything lives in one folder outside the vault, `D:\python456\cpu-chatbot`, because the vault holds notes and not project folders.

## Steps
1. **Check the file sizes before downloading.** The Hugging Face API lists every file in the repo with its size, so I could see that the weights are one file of 988,097,824 bytes. The weights on disk are `bfloat16` (the config says so), which is 2 bytes per number: 494.03 million × 2 = 0.99 GB, which matches.

2. **Download the model.** Only the files the model needs (the weights, the tokenizer and the config), not the whole repo.

   **`dl.py`**
   ```python
   from huggingface_hub import snapshot_download
   p = snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", local_dir="qwen2.5-0.5b-instruct",
                         allow_patterns=["*.json", "*.txt", "model.safetensors", "LICENSE"])
   print(p)
   ```
   Run `python dl.py`. It fetched 8 files in about 22 seconds and `model.safetensors` came out at exactly 988,097,824 bytes.

3. **Write the terminal chatbot.** Four ideas are in this file. (a) The model only continues text, so the conversation is rewritten into the format it was trained on by the **chat template** ([[Chat Template]]), and `add_generation_prompt=True` adds the "assistant speaks now" marker at the end. (b) It has no memory of its own: the **history list** is sent again every turn, and that is the memory. (c) The answer is **streamed**: `generate` runs in a second thread and a `TextIteratorStreamer` hands over pieces as they appear, so you read while it writes. (d) a `temperature` of 0 turns sampling off and always takes the most likely token; above 0 the next token is drawn at random from the likely ones ([[Decoding Strategies]]).

   **`hf_chat.py`**
   ```python
   """Terminal chatbot on CPU with Hugging Face transformers. Type 'quit' to stop, 'reset' to forget."""
   import sys, time, threading, torch
   from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

   MODEL = "qwen2.5-0.5b-instruct"
   SYSTEM = "You are a friendly, short-spoken assistant."

   tok = AutoTokenizer.from_pretrained(MODEL)
   model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()


   def reply(history, temperature=0.7, max_new_tokens=200):
       """history = list of {'role','content'}; yields the answer piece by piece."""
       ids = tok.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt")
       streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
       kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids), streamer=streamer,
                 max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                 temperature=temperature if temperature > 0 else None, top_p=0.9,
                 pad_token_id=tok.eos_token_id)
       threading.Thread(target=lambda: model.generate(**kw)).start()
       for piece in streamer:
           yield piece


   if __name__ == "__main__":
       history = [{"role": "system", "content": SYSTEM}]
       while True:
           user = input("you> ").strip()
           if user == "quit":
               break
           if user == "reset":
               history = history[:1]
               print("(memory cleared)")
               continue
           history.append({"role": "user", "content": user})
           t0, text = time.time(), ""
           print("bot> ", end="", flush=True)
           for piece in reply(history):
               text += piece
               print(piece, end="", flush=True)
           history.append({"role": "assistant", "content": text})
           print(f"\n     [{time.time() - t0:.1f}s]")
   ```
   I loaded the model as `float32` because plain CPU maths on float32 is the safe default. That is a choice, and it is one reason this route is slow (see below).

4. **Test the memory.** I fed it a script instead of typing, so the test can be repeated:
   ```
   printf 'My name is Dev. Say hi in five words.\nWhat is my name?\nreset\nWhat is my name?\nquit\n' | python hf_chat.py
   ```
   What I saw (the lines about library warnings I filtered out with `grep -v Warning` and did not read):
   ```
   bot> Hello there!
        [0.9s]
   bot> Your name is Dev.
        [1.1s]
   (memory cleared)
   bot> I'm not named in any way. How can I assist you today?
        [2.8s]
   ```
   What to look for: the second answer uses a fact from the first turn (the history works), and the third does not (the reset works).

5. **Write the web page (Streamlit).** It reuses `reply` from `hf_chat.py`, keeps the history in `st.session_state`, and has a temperature slider and a "forget everything" button. Run it with `streamlit run app.py`.

   **`app.py`**
   ```python
   """Chat web page. Run:  streamlit run app.py"""
   import streamlit as st
   from hf_chat import reply, SYSTEM

   st.title("Local CPU chatbot")
   temp = st.sidebar.slider("temperature", 0.0, 1.5, 0.7, 0.1)
   if st.sidebar.button("forget everything"):
       st.session_state.history = [{"role": "system", "content": SYSTEM}]

   if "history" not in st.session_state:
       st.session_state.history = [{"role": "system", "content": SYSTEM}]

   for m in st.session_state.history[1:]:
       st.chat_message(m["role"]).write(m["content"])

   if prompt := st.chat_input("say something"):
       st.session_state.history.append({"role": "user", "content": prompt})
       st.chat_message("user").write(prompt)
       with st.chat_message("assistant"):
           box, text = st.empty(), ""
           for piece in reply(st.session_state.history, temperature=temp):
               text += piece
               box.write(text)
       st.session_state.history.append({"role": "assistant", "content": text})
   ```
   All I checked: I started the server on port 8765, the page answered HTTP 200 and `/healthz` answered `ok`. **I did not chat through the page**, see below.

6. **Run the same model through Ollama.** Ollama is a program that downloads models and runs them (from memory, with [[llama.cpp]], a fast C++ engine for [[GGUF]] model files; I did not check this in its files). After a call, `ollama ps` listed the model as 794 MB in memory (more than the 397 MB file, I did not check why) and "100% CPU". `ollama pull qwen2.5:0.5b` fetched 397 MB. `ollama show qwen2.5:0.5b` said: 494.03M parameters, context length 32,768, quantization **Q4_K_M**. Despite the "4" in the name, my later look inside the file ([[GGUF]]) showed a mix of five number types that averages 6.345 bits per weight, so the weights are not the same numbers as in step 2 ([[Quantization]]). You talk to it by a local web call, no extra library:
   ```
   POST http://localhost:11434/api/chat
   ```
   The reply carries `prompt_eval_count`, `eval_count` and the times in nanoseconds, which is what step 7 uses.

7. **Time both routes on one prompt.** Same question, greedy (no randomness), exactly 64 new tokens. For transformers the speed is found by two timed runs: one that makes 1 token (the time to read the prompt), one that makes 64. The speed is 63 tokens divided by the extra time. For Ollama the speed comes from its own `eval_count / eval_duration`. A warm-up call comes first on both.

   **`bench.py`**
   ```python
   """Speed test: the same prompt on transformers (float32) and on Ollama. Greedy, 64 new tokens."""
   import json, time, urllib.request, torch
   from transformers import AutoTokenizer, AutoModelForCausalLM

   PROMPT = [{"role": "user", "content": "Explain in simple words why the sky is blue."}]
   N = 64


   def hf():
       tok = AutoTokenizer.from_pretrained("qwen2.5-0.5b-instruct")
       m = AutoModelForCausalLM.from_pretrained("qwen2.5-0.5b-instruct", torch_dtype=torch.float32).eval()
       ids = tok.apply_chat_template(PROMPT, add_generation_prompt=True, return_tensors="pt")
       run = lambda n: m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, min_new_tokens=n,
                                  do_sample=False, pad_token_id=tok.eos_token_id)
       run(2)                                            # warm-up
       t = time.time(); run(1); t1 = time.time() - t     # prompt + first token
       t = time.time(); run(N); tn = time.time() - t     # prompt + N tokens
       return ids.shape[1], t1, (N - 1) / (tn - t1)


   def ollama():
       body = {"model": "qwen2.5:0.5b", "messages": PROMPT, "stream": False,
               "options": {"temperature": 0, "num_predict": N, "seed": 1}}
       req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(),
                                    {"Content-Type": "application/json"})
       urllib.request.urlopen(req).read()                # warm-up: loads the model
       r = json.load(urllib.request.urlopen(req))
       return r["prompt_eval_count"], r["prompt_eval_duration"] / 1e9, r["eval_count"] / (r["eval_duration"] / 1e9)


   if __name__ == "__main__":
       print("threads", torch.get_num_threads())
       for name, f in (("transformers fp32", hf), ("ollama qwen2.5:0.5b", ollama)):
           n, first, tps = f()
           print(f"{name:22s} prompt tokens {n:3d} | first token {first:5.2f}s | speed {tps:6.1f} tokens/s")
   ```
   Output, run 1 then run 2 (run 3 printed 6.3 and 39.1 tokens/s):
   ```
   threads 4
   transformers fp32      prompt tokens  40 | first token  0.30s | speed    6.0 tokens/s
   ollama qwen2.5:0.5b    prompt tokens  40 | first token  0.02s | speed   50.5 tokens/s

   transformers fp32      prompt tokens  40 | first token  0.44s | speed    6.0 tokens/s
   ollama qwen2.5:0.5b    prompt tokens  40 | first token  0.02s | speed   42.9 tokens/s
   ```
   Both routes read the same 40 tokens. **Ignore the Ollama "first token" figure**: the warm-up used the same prompt, so Ollama had probably kept the prompt from the call before and did not read it again (my guess, not checked). Only the speeds are a fair comparison.

## What the numbers say
- **Why one token costs so much.** To write one token, the computer reads (about) every weight once ([[LLM Inference]]). The float32 copy is 494.03 million × 4 bytes = 1.98 GB. At 6.0 tokens/s that is about 11.9 GB read per second (my arithmetic). The Ollama file is 397 MB, and 397 MB × 50.5 tokens/s is about 20 GB/s (17 GB/s at 42.9, and about 16 GB/s at 39.1) (my arithmetic). Reading about 5 times fewer bytes explains part of the 6 to 8 times gap, and the rest has to be the program itself (a tuned C++ engine against Python and PyTorch), but **I did not separate the two** and I did not measure the laptop's real memory speed. The "reads every weight once per token" rule is a simplification (from memory, the standard explanation for why small-batch decoding is limited by memory speed, not maths).
- **What 6 tokens per second feels like.** A word is about 1.3 tokens in English (from memory), so about 4 to 5 words a second. That is readable as it streams. 40 to 50 tokens per second is faster than anyone reads.
- **Threads.** PyTorch picked 4 threads by default on this 8-logical-core machine. I did not try other thread counts, `bfloat16` or PyTorch's 8-bit dynamic quantization, so 6 tokens/s is not the best this route can do.

## What went wrong, and the fix
- **Ollama was not running.** My first `ollama list` printed "could not connect to a running Ollama instance" and then started the Ollama app by itself (the log lines show it starting its server). It then worked. If a call fails with "connection refused", start the app first.
- **No project code failed on its first run.** The model load, the chat test and both benchmark runs worked as written.
- **The browser check failed.** I wanted to open the web page and chat with it, but the Chrome extension tool was not connected ("Browser extension is not connected"). I stopped the server and did not retry.

## What I did not run
- **The chat page in a browser.** Streamlit runs `app.py` only when a browser connects, so my HTTP 200 and `ok` checks show that the server starts, not that `app.py` works. The `reply` function inside it is the same one the terminal test used, but the slider, the button and the layout are untested. Streamlit 1.25.0 is old. From memory, `st.chat_message` and `st.chat_input` came in 1.24, so they should exist, but I did not check each call.
- **Long chats.** I only did three turns. When the history grows past the context (32,768 tokens for this model) the code does nothing about it, and every turn re-reads the whole history, so it also gets slower.
- **Answer quality.** A 0.5-billion-parameter model is small. I read three short answers and nothing else. I did not test facts, maths or other languages, and I did not compare with a bigger model.
- **A fresh environment.** I used my existing Python, so a missing package would not have shown up.
- **Other models, a GPU, `bfloat16`, other thread counts, memory use measured while running.** The "about 2.0 GB" for float32 is my arithmetic, not a measurement.
- **The "Q4_K_M" label** is what `ollama show` printed. What the format does inside is described in [[GGUF]] and [[llama.cpp]], where I read the real file, but I did not read the quantization code itself.

## Concepts used
- [[LLM]], [[Transformer]], [[Tokenization]], [[Hugging Face Transformers]], [[Safetensors]], [[Hugging Face Hub]]
- [[Instruction Tuning]] (why the "Instruct" model chats), [[Chat Template]], [[Context Window]]
- [[Decoding Strategies]], [[KV Cache]], [[LLM Inference]], [[Quantization]]
- [[Ollama]], [[llama.cpp]], [[GGUF]]
- [[PyTorch]], [[Streamlit]], [[Python]]

## Next ideas
- Try `torch.set_num_threads(8)`, `bfloat16` and 8-bit dynamic quantization, and see how much of the gap closes without Ollama.
- Add the vault as the chatbot's memory with [[RAG]] (the earlier project [[Vault Search with RAG]] already finds the notes).
- Cut the history when it gets long and show what happens to the speed.
- Run a 1.5B or 3B model and plot speed against size.

## Flashcards
#flashcards/projects

#flashcards/projects/concept #flashcards/topic/project-llm Why does a chatbot built on a plain language model need a chat template?
?
The model only continues text. The template rewrites the conversation into the exact format the model was trained on, with the system, user and assistant parts marked, and `add_generation_prompt=True` adds the marker for "assistant speaks now".

#flashcards/projects/concept #flashcards/topic/project-llm Where does the chatbot's memory of earlier turns live?
?
In the history list that the program sends again every turn. The model itself keeps nothing between calls. Reset the list and it forgets.

#flashcards/projects/concept #flashcards/topic/project-llm In the CPU chatbot, why is `generate` run in a second thread?
?
So a `TextIteratorStreamer` can hand over pieces of the answer while the model is still writing. The main thread prints them as they arrive.

#flashcards/projects/example #flashcards/topic/project-llm Qwen2.5-0.5B has 494.03 million parameters stored as bfloat16. How big is the file, and does it match?::494.03 million × 2 bytes = 0.99 GB. The real file was 988,097,824 bytes, the same as the size the Hugging Face API listed (my arithmetic).

#flashcards/projects/example #flashcards/topic/project-llm The same 494.03 million weights are loaded as float32. How much memory is that?::494.03 million × 4 bytes = 1.98 GB, about 2.0 GB. This is my arithmetic and was not measured while running.

#flashcards/projects/example #flashcards/topic/project-llm On this laptop, what speed did the float32 transformers route reach for 64 greedy tokens, and what did Ollama reach?::transformers float32: 6.0, 6.0 and 6.3 tokens/s in three runs. Ollama (Q4_K_M): 50.5, 42.9 and 39.1 tokens/s. Ollama was about 6 to 8 times faster.

#flashcards/projects/example #flashcards/topic/project-llm The float32 route made 6.0 tokens/s from a 1.98 GB model. How much memory does it read per second, if each token reads every weight once?::1.98 GB × 6.0 = about 11.9 GB/s (my arithmetic, a simplification). Ollama's 397 MB × 50.5 is about 20 GB/s, so it reads about 5 times fewer bytes per token.

#flashcards/projects/concept #flashcards/topic/project-llm Why can the speed gap between the two routes not be blamed on the smaller weights alone?::The Ollama file is only about 5 times smaller (it averages 6.345 bits per weight, not 4), the gap is 6 to 8 times, and the two routes also differ in program (tuned C++ against Python and PyTorch). The project did not separate the two causes.

#flashcards/projects/concept #flashcards/topic/project-llm Why should the Ollama "first token" time in the benchmark be ignored?::The warm-up call used the same prompt, so Ollama had probably cached it and did not read it again (a guess, not checked). Only the speeds are a fair comparison.

#flashcards/projects/concept #flashcards/topic/project-llm What did the web page check in this project prove, and what did it not prove?::The Streamlit server started, the page returned HTTP 200 and `/healthz` returned ok. It did not prove `app.py` works, because Streamlit runs the script only when a browser connects, and the Chrome tool was not connected.

#flashcards/projects/example #flashcards/topic/project-llm What does `ollama show` say about `qwen2.5:0.5b`, and what is different from the Hugging Face copy?::494.03M parameters, context 32,768, quantization Q4_K_M. It is a quantized file of 397 MB (a mix averaging 6.345 bits per weight), so the weights are not the same numbers as the bfloat16 or float32 copy.

#flashcards/projects/concept #flashcards/topic/project-llm Name three things the project did not test.::The chat page in a browser, long chats past the context size, and answer quality (also other thread counts, bfloat16, a GPU and a fresh environment).

#flashcards/projects/concept #flashcards/topic/project-llm What happened when `ollama list` was run while Ollama was not running?::It printed that it could not connect, then started the Ollama app by itself, and after that it worked.
