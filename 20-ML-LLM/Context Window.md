---
tags: [LLM, context, tokens, inference, memory, chat]
status: growing
level: 4
tier: 2
---

# Context Window

The **context window** is the most text, counted in tokens, that a language model can look at in one go. Your question, the chat so far, any documents you paste in, and the answer being written all have to fit inside it together.

## 🟢 1. Simple (age 12)
Think of a desk. The AI can only use what is lying on the desk right now. If you keep piling on more papers and the desk is full, the oldest papers fall off the edge and the AI no longer sees them. A bigger desk holds more, but it is also heavier to carry around.

**Example:** you tell a friend a secret code, then read them a very long story, then ask for the code. If they can only hold the last few pages in their head, the code is gone. A friend with a bigger head still has it.

## 🔵 2. How it works
1. The model does not read letters or words. It reads **tokens**, small pieces of text ([[Tokenization]]). In my test one short sentence of 19 words became 21 tokens, a bit more than one token per word.
2. The window is a **token budget**. Everything counts against it: the instructions, the earlier chat, pasted documents and the reply the model is still writing.
3. **A chat eats the window one turn at a time.** Each turn adds your question and the model's answer, and the program also adds a few wrapper tokens around every message ([[Chat Template]]). Next turn, the whole chat is sent again.
4. **Two different limits get mixed up.** The model has a built-in maximum it was made to handle (the file for my small model says 32,768 tokens). The program that runs it may use a much smaller window by default, and you can change that. Ollama has a setting for it called `num_ctx` ([[Ollama]]).
5. **When the text is longer than the window, something has to go.** In my test the oldest part was lost and the model could no longer answer a question about it.
6. **A bigger window costs memory.** The model keeps notes about every token in its window, called the [[KV Cache]], so more tokens means more memory.
7. **A bigger window does not mean the model uses all of it equally well.** That is a different problem, described in [[Context Rot]].
8. **Ways to live with a limit:** keep chats short, summarise old turns, or fetch only the pieces you need instead of pasting everything ([[RAG]]).

I did not add a picture. A search of Wikimedia Commons found only unrelated papers and a sound file, so I used none, as the vault rules say.

## 🟠 3. Technical detail
- **Model limit vs run-time window.** `max_position_embeddings` in the config of `qwen2.5-0.5b-instruct` is 32,768 ([[Hugging Face Transformers]]). That is the longest input the model was set up for. The window you actually get is whatever the program asks for. I set `num_ctx` myself in Ollama (256 and 4,096) and read what the model processed.
- **Counting is done with the model's own tokenizer.** Another model, with a different tokenizer, would give a different count for the same text. I used the tokenizer of the 0.5B model for Ollama's `qwen2.5:1.5b` model, because both are in the Qwen2.5 family (my assumption that the tokenizers match; the counts below were close but not identical, see the output).
- **The reply shares the window.** If the prompt already fills it, there is no room to answer. In my chat count, the answer's tokens are part of the turn size.
- **What I measured.** `ctx_demo.py` (below) does five things: (1) reads the model's limit, (2) counts tokens in a sentence, (3) measures the wrapper cost of the chat template, (4) grows a chat until it passes a window of 512 and a window of 4,096, and (5) asks `qwen2.5:1.5b` for a secret code that sits at the very start of a 1,509-token question, once with `num_ctx` 256 and once with 4,096. It then fits the straight-line rule of the equation below, and computes the full-window cache size.

**`ctx_demo.py`** (needs `transformers`, the model folder from [[Local CPU Chatbot]], and Ollama running with `qwen2.5:1.5b`)
```python
"""Context window, measured: count tokens, see how a chat fills the window, and watch a small window forget. CPU only."""
import json, urllib.request
from transformers import AutoTokenizer

PATH = "qwen2.5-0.5b-instruct"                      # tokenizer from the Local CPU Chatbot project (same family as qwen2.5:1.5b in Ollama)
tok = AutoTokenizer.from_pretrained(PATH)
cfg = json.load(open(PATH + "/config.json"))
print("1 model limit (max_position_embeddings):", cfg["max_position_embeddings"], "tokens")

text = "The context window is the most text a model can read at once. Everything it needs must fit inside."
n_tok = len(tok(text)["input_ids"])
print(f"2 text: {len(text.split())} words, {len(text)} characters, {n_tok} tokens ({n_tok / len(text.split()):.2f} tokens per word)")


def count(msgs):                                    # tokens after the chat template wraps the messages
    return len(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True))


system = {"role": "system", "content": "You are a helpful assistant."}
empty = count([system, {"role": "user", "content": ""}])
print("3 chat template overhead, system + empty question:", empty, "tokens")

q, a = "What is a token?", "A token is a small piece of text, often a word or part of a word, that a model reads as one unit."
msgs, sizes = [system], {}
for turn in range(1, 201):                           # a chat where every turn is the same question and answer
    msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    sizes[turn] = count(msgs[:-1]) + len(tok(a)["input_ids"]) + 2   # prompt before the last answer, plus the answer, plus its end marker
for W in (512, 4096):
    over = next(t for t in sizes if sizes[t] > W)
    print(f"4 window {W:>4}: turn {over - 1} fits ({sizes[over - 1]} tokens), turn {over} does not ({sizes[over]} tokens)")


def ask(num_ctx, filler_lines):                      # a secret at the start, filler after it, then the question
    note = "Remember this: the secret code is 7342."
    filler = " ".join(f"Line {i}: the sky is blue and the grass is green." for i in range(filler_lines))
    body = {"model": "qwen2.5:1.5b", "stream": False, "options": {"temperature": 0, "seed": 1, "num_ctx": num_ctx, "num_predict": 20},
            "messages": [{"role": "user", "content": note + " " + filler + " What was the secret code?"}]}
    req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req))
    return len(tok(body["messages"][0]["content"])["input_ids"]), r["prompt_eval_count"], r["message"]["content"].strip()


for num_ctx in (256, 4096):
    mine, seen, reply = ask(num_ctx, 100)
    print(f"5 num_ctx {num_ctx:>4}: my count of the question {mine} tokens, model processed {seen} -> reply: {reply!r}")

c = sizes[2] - sizes[1]                              # tokens added by each turn
n0 = sizes[1] - c                                    # tokens before the first turn (system part)
print(f"6 n0 = {n0}, c = {c} per turn; turns that fit: window 512 -> {(512 - n0) // c}, window 4096 -> {(4096 - n0) // c}")
L, H_kv, d, W = cfg["num_hidden_layers"], cfg["num_key_value_heads"], cfg["hidden_size"] // cfg["num_attention_heads"], cfg["max_position_embeddings"]
print(f"7 full-window KV cache of this model: 2 x {L} x {H_kv} x {d} x {W} x 2 bytes = {2 * L * H_kv * d * W * 2:,} bytes = {2 * L * H_kv * d * W * 2 / 1e9:.2f} GB (16-bit)")
```
What it printed (transformers 4.46.3, Ollama 0.21.2, Python 3.9.6, CPU only, `temperature` 0 and `seed` 1):
```
1 model limit (max_position_embeddings): 32768 tokens
2 text: 19 words, 98 characters, 21 tokens (1.11 tokens per word)
3 chat template overhead, system + empty question: 19 tokens
4 window  512: turn 12 fits (503 tokens), turn 13 does not (544 tokens)
4 window 4096: turn 99 fits (4070 tokens), turn 100 does not (4111 tokens)
5 num_ctx  256: my count of the question 1509 tokens, model processed 256 -> reply: 'The secret code in this scenario appears to be a sequence of instructions or guidelines for creating something, likely'
5 num_ctx 4096: my count of the question 1509 tokens, model processed 1538 -> reply: 'The secret code provided in your message is "7342". This appears to be a sequence'
6 n0 = 11, c = 41 per turn; turns that fit: window 512 -> 12, window 4096 -> 99
7 full-window KV cache of this model: 2 x 24 x 2 x 64 x 32768 x 2 bytes = 402,653,184 bytes = 0.40 GB (16-bit)
```
What to look for:
- **Line 4:** a chat of identical turns passes 512 tokens at turn 13 and 4,096 tokens at turn 100. The step per turn is a steady 41 tokens (line 6), which is why the fit in the equation below is exact for both windows.
- **Line 5:** with a window of 256 the model **processed only 256 tokens** of a 1,509-token question and did not know the code. With 4,096 it processed the whole question and answered `"7342"`. I did not check which tokens Ollama dropped; a reply that has lost the code suggests the start went, but that is my reading.
- **Line 5, counts:** I counted 1,509 tokens and Ollama reported 1,538. The extra 29 are probably wrapper tokens that the chat template and a default system message add (not checked).
- **Line 7:** the full 32,768-token window of this small model would need 402,653,184 bytes (0.40 GB) of cache at 16 bits per number. This uses the shape from the config (24 layers, 2 key/value heads, head size 64 = 896 / 14). It is a size from my arithmetic and was not measured in memory. The cache formula itself is in [[KV Cache]].

## 🔴 4. Deep / formal
**How many turns of a chat fit.** If a chat starts with $n_0$ tokens (the system part) and every turn adds the same $c$ tokens (question, answer, and wrapper tokens), then after $t$ turns the chat holds $n_0 + t\,c$ tokens. The last turn that still fits in a window of $W$ tokens is

$$
T=\left\lfloor \frac{W-n_0}{c} \right\rfloor
$$

**Where:**
- $T$ is the number of whole turns that fit, a whole number.
- $W$ is the window size in tokens.
- $n_0$ is the tokens used before the first turn (instructions and wrapper).
- $c$ is the tokens one turn adds. $\lfloor\cdot\rfloor$ means "round down", because half a turn does not fit.
- **Matters:** $T$ grows in a straight line with $W$ and shrinks as $c$ grows, so long answers eat the window fast. Without the round-down you would count a turn that gets cut off in the middle.

**Example (tiny numbers):** $W=100$, $n_0=10$, $c=20$.
- Turn 4 holds $10+4\cdot20=90$ tokens, which is at most 100, so it fits.
- Turn 5 would hold $10+5\cdot20=110$ tokens, which is more than 100, so it does not.
- $T=\lfloor(100-10)/20\rfloor=\lfloor4.5\rfloor=\mathbf{4}$.
- Meaning: this chat can go 4 turns before the oldest part has to be dropped or summarised.

My measured case uses the same rule with $n_0=11$ and $c=41$: $W=512$ gives $\lfloor501/41\rfloor=12$ turns and $W=4096$ gives $\lfloor4085/41\rfloor=99$ turns, which match lines 4 and 6 of the output.

**Further notes:**
- **What happens at the limit is a choice made by the program, not by the model.** Options are to cut the oldest tokens, cut the middle, summarise, refuse, or raise an error. I saw only one behaviour, in one program (Ollama, one run).
- **Position matters inside a long window.** Models can be worse at using the middle of a long input than its start or end; see [[Context Rot]] (abstract level only in that note).
- **Memory grows with $W$, and so does compute.** The cache grows in a straight line with the tokens held ([[KV Cache]]). Ways to shrink it are fewer key/value heads ([[Grouped-Query Attention]]) and a window that looks only at recent tokens ([[Sliding Window Attention]]). This config lists a `sliding_window` value of 32,768, equal to the maximum, and I did not test whether it changes anything.
- **Reusing a shared start is cheaper** when the program caches it ([[Prompt Caching]]).
- **How position is told to the model** is its own topic ([[Positional Encoding]], [[RoPE]]); how far a model can be stretched beyond the length it was trained on, and how, I did not test.
- **Many agents mean many windows.** In [[Multi-Agent Systems]] each model call has its own window, so a team can pass work along without any one call holding everything. I did not measure that.
- **Not tested:** other models, other programs, whether answers get worse near the limit, whether memory use matches the cache formula, and larger windows than 4,096.

**What came before, and how this improved on it:**
- **Before:** models that squeezed everything they had read into one fixed-size memory ([[From FNN to RNN]] covers the recurrent kind), so how much they could "hold" did not grow with the input and old details faded.
- **Its limit:** a fixed memory forgets in ways you cannot point to. You cannot tell which detail was lost.
- **What changed:** a [[Transformer]] keeps a note for every token in the window and looks at all of them ([[Attention]]). What the model can use is then clearly bounded by the window: anything inside can be looked at directly, anything outside is simply absent.
- **How much better:** I did not compare with an older kind of model, so there is no measured improvement against it. What I measured is the bound itself: the same 1,509-token question was answered with `num_ctx` 4,096 (the code was found) and was not with 256 (256 of 1,509 tokens seen). Older window sizes and the dates they appeared are not given here because I did not check them.

## Related
[[Tokenization]] · [[KV Cache]] · [[Context Rot]] · [[Chat Template]] · [[Attention]] · [[Transformer]] · [[Ollama]] · [[Prompt Caching]] · [[Sliding Window Attention]] · [[Grouped-Query Attention]] · [[Positional Encoding]] · [[RoPE]] · [[RAG]] · [[Multi-Agent Systems]] · [[Hugging Face Transformers]] · [[Local CPU Chatbot]] · [[LLM]] · [[Structured Output]] · [[System Prompt]]

## Sources
- Run on this PC: `ctx_demo.py`, CPU only, with the tokenizer and config of `qwen2.5-0.5b-instruct` and `qwen2.5:1.5b` in Ollama 0.21.2. All printed lines above are from my own run (one run each; the question and its filler were fixed).
- The model's config file (`config.json` in the downloaded folder): 32,768 maximum positions, 24 layers, 2 key/value heads, 14 attention heads, width 896.
- No web page or paper was read for this note. Claims about what other programs do at the limit, and about older models, are left out on purpose.
- No image: Wikimedia Commons returned nothing on topic, so I added none.
- The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm Context window:::The most text, counted in tokens, that a model can look at in one go. The question, the chat so far, any pasted documents and the reply all have to fit inside it together.

#flashcards/ml/formula #flashcards/topic/llm Write the number of chat turns that fit in a window and explain each symbol.
?
$$T=\left\lfloor \frac{W-n_0}{c} \right\rfloor$$
- $T$: the number of whole turns that fit.
- $W$: the window size in tokens.
- $n_0$: the tokens used before the first turn (instructions and wrapper).
- $c$: the tokens each turn adds (question, answer and wrapper). $\lfloor\cdot\rfloor$ rounds down, because half a turn does not fit.
- Longer answers make $c$ bigger, so fewer turns fit.

#flashcards/ml/example #flashcards/topic/llm With $W=100$, $n_0=10$ and $c=20$, how many chat turns fit?::Turn 4 holds 10 + 4 x 20 = 90 tokens (fits). Turn 5 would hold 110 (does not). So T = floor(90 / 20) = 4 turns.

#flashcards/ml/example #flashcards/topic/llm In my measured chat, $n_0=11$ and $c=41$. How many turns fit in windows of 512 and 4096 tokens?::512: floor(501 / 41) = 12 turns. 4096: floor(4085 / 41) = 99 turns. Both match the chat I grew in `ctx_demo.py`.

#flashcards/ml/example #flashcards/topic/llm What did a window of 256 do to my 1,509-token question in Ollama?::The model processed only 256 tokens and could not tell me the secret code, which sat at the very start. With a window of 4,096 it processed the whole question and answered 7342. (One run, one model; I did not check which tokens were dropped.)

#flashcards/ml/concept #flashcards/topic/llm What is the difference between a model's limit and the window you actually get?::The limit is the longest input the model was set up for (32,768 for my small Qwen2.5 model, from its config). The window you get is whatever the program asks for. Ollama's setting for it is `num_ctx`, and it can be much smaller.

#flashcards/ml/concept #flashcards/topic/llm Why does a chat use up the window faster than the questions alone suggest?::Every turn adds the question, the answer and a few wrapper tokens from the chat template, and the whole chat is sent again on the next turn. In my test each turn added a steady 41 tokens.

#flashcards/ml/cloze #flashcards/topic/llm The reply the model writes also has to fit inside the ==context window==, together with the ==prompt==.

#flashcards/ml/example #flashcards/topic/llm How much cache would the full 32,768-token window of my small model need?::402,653,184 bytes (0.40 GB) at 16 bits per number, from my arithmetic: 2 x 24 layers x 2 key/value heads x 64 head size x 32,768 tokens x 2 bytes. It was not measured in memory.

#flashcards/ml/concept #flashcards/topic/llm Who decides what happens when the text is longer than the window?::The program, not the model. Options include cutting the oldest tokens, cutting the middle, summarising, refusing or raising an error. I saw only one behaviour in one program (Ollama, one run).

#flashcards/ml/concept #flashcards/topic/llm Does a bigger window mean the model uses all of it well?::No. A window is a maximum. Models can be worse at using the middle of a long input than its start or end (see Context Rot, abstract level only). Measure on your own task.

#flashcards/ml/concept #flashcards/topic/llm What did I not test about context windows?::Other models and programs, answer quality near the limit, real memory use against the cache formula, windows larger than 4,096 tokens, and how far a model can be stretched past its trained length.
