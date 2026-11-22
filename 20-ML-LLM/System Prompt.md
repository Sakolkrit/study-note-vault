---
tags: [LLM, prompt, chat, system-prompt, inference]
status: growing
level: 4
tier: 2
---

# System Prompt

A **system prompt** is the first message in a chat, written by the developer, that tells the model who it is and how it should behave. The user's questions come after it, and the model reads it again on every turn.

## 🟢 1. Simple (age 12)
Think of a new worker's first-day card: "You work at the ice-cream shop. Be polite. Never give out free cones." The customers never see the card, but the worker follows it all day. A system prompt is that card for an AI chatbot.

**Example:** you build a homework helper. You write "You are a kind maths tutor. Give hints, not answers." Every child who uses it gets a tutor that gives hints, even though none of them typed those words.

## 🔵 2. How it works
1. A chat is a list of messages, and each has a role: `system`, `user` or `assistant` ([[Chat Template]]). The system message is the one the developer writes.
2. The [[Chat Template]] puts it at the very start of the text the model reads, wrapped in marker words. The model has no secret "rule box". It just reads the system text first, then the rest.
3. Because the model was trained on chats where the first message sets the scene ([[Instruction Tuning]]), it usually treats that message as the standing instructions.
4. **If you write none, the template may add one for you.** In my run, the Qwen template added "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
5. **It is read again on every turn.** The model starts fresh each time, so the whole system prompt is part of every call. A long one costs tokens each turn and uses up room in the [[Context Window]].
6. **It is a request, not a lock.** The model follows it most of the time, not always. In my test, a small model obeyed a simple rule in the system prompt 5 times out of 8 (see below).
7. Agents use system prompts for their role and for the allowed tools or reply shape ([[Agent]], [[Tool Calling]], [[Structured Output]], [[Multi-Agent Systems]]).

No picture: I searched Wikimedia Commons for this topic and found only unrelated documents, so I added none.

## 🟠 3. Technical detail
- **What I measured.** `sp_demo.py` (below) does three things on this PC. Part 1 prints the first characters of the formatted text with and without my own system message. Part 2 asks `qwen2.5:1.5b` eight easy questions and counts how often the reply ends with the word DONE, in three setups: no rule, the rule in the system message, and the same rule added to the user's question. Part 3 measures how many tokens a system prompt of 10, 100 and 400 words adds to every call.
- **Placement.** The system message is the first block after the opening marker `<|im_start|>system`. Other models use other markers, and some templates have no system role at all (not tested here).
- **Cost.** The token count of the system prompt is added to every call. With [[Prompt Caching]] or a [[KV Cache]] the model need not recompute an unchanged start, but the tokens still fill the window. I did not measure the speed effect.

**`sp_demo.py`** (needs `transformers`, the model folder from [[Local CPU Chatbot]], and Ollama running with `qwen2.5:1.5b`)
```python
"""System prompt, measured: see where it sits in the template, what it costs in tokens, and how well a small model obeys it. CPU only."""
import json, urllib.request
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("qwen2.5-0.5b-instruct")
user = {"role": "user", "content": "Name one colour."}
SYS = "You are a pirate. Reply in one short sentence."


def render(msgs):
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


print("1 no system message -> the template adds its own:", repr(render([user])[:75]))
print("1 with my system    ->", repr(render([{"role": "system", "content": SYS}, user])[:75]))
print("1 tokens, no system:", len(tok(render([user]))["input_ids"]), "| with my system:", len(tok(render([{"role": "system", "content": SYS}, user]))["input_ids"]))

RULE = "End every reply with the word DONE."
QUESTIONS = ["Name one colour.", "What is 2 plus 2?", "Who wrote Hamlet?", "What is the capital of Peru?", "Name a fruit.",
             "What is the opposite of hot?", "How many days are in a week?", "Name an animal that swims."]


def ask(system, question, rule_in_user=False):             # one local call; returns the reply text and its token count
    text = question + (" " + RULE if rule_in_user else "")
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
    body = {"model": "qwen2.5:1.5b", "stream": False, "options": {"temperature": 0, "seed": 1, "num_predict": 120}, "messages": msgs}
    req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req))
    return r["message"]["content"].strip(), r["eval_count"]


for name, system, in_user in (("no rule", None, False), ("rule in system", RULE, False), ("rule in user turn", None, True)):
    out = [ask(system, q, in_user) for q in QUESTIONS]
    done = sum(t.rstrip(" .!").endswith("DONE") for t, _ in out)
    print(f"2 {name:18}: ends with DONE in {done} of {len(QUESTIONS)}; mean reply {sum(n for _, n in out) / len(out):.1f} tokens; first: {out[0][0]!r}")

base = len(tok(render([{"role": "system", "content": ""}, user]))["input_ids"])   # an empty system message, so the default is not counted
for n_words in (10, 100, 400):                       # what a long system prompt costs: it is re-read on every turn
    s = " ".join(["Be kind."] * (n_words // 2))
    n = len(tok(render([{"role": "system", "content": s}, user]))["input_ids"]) - base
    print(f"3 system prompt of {n_words:>3} words adds {n} tokens to every call; over 20 turns that is {20 * n} tokens read again")
```
What it printed (transformers 4.46.3, Ollama 0.21.2, Python 3.9.6, CPU only, `temperature` 0, `seed` 1):
```
1 no system message -> the template adds its own: '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpfu'
1 with my system    -> '<|im_start|>system\nYou are a pirate. Reply in one short sentence.<|im_end|>'
1 tokens, no system: 33 | with my system: 28
2 no rule           : ends with DONE in 0 of 8; mean reply 13.9 tokens; first: 'Blue is one of the colors.'
2 rule in system    : ends with DONE in 5 of 8; mean reply 8.5 tokens; first: 'Blue is a color. DONE'
2 rule in user turn : ends with DONE in 8 of 8; mean reply 7.5 tokens; first: 'Blue DONE.'
3 system prompt of  10 words adds 15 tokens to every call; over 20 turns that is 300 tokens read again
3 system prompt of 100 words adds 150 tokens to every call; over 20 turns that is 3000 tokens read again
3 system prompt of 400 words adds 600 tokens to every call; over 20 turns that is 12000 tokens read again
```
What to look for:
- **Part 1:** with no system message, the template inserted its own ("You are Qwen, created by Alibaba Cloud..."). With mine, the start is my text. The whole prompt was 33 tokens with the default and 28 with my shorter message.
- **Part 2:** with no rule, 0 of 8 replies ended with DONE and replies averaged 13.9 tokens. With the rule in the system message, 5 of 8 did. With the same rule added to the user's own question, 8 of 8 did. So in this test, **the small model followed the rule better when it sat in the user turn than in the system message.** That is one run of one small model and one odd rule, so I do not claim this holds in general. I did not look at which three replies failed or why.
- **Part 3:** a system prompt of 10 words added 15 tokens to every call, 100 words added 150, and 400 words added 600. Over 20 turns, 600 tokens per call means 12,000 tokens read again. (The words were the repeated phrase "Be kind.", so real text may tokenise differently.)

## 🔴 4. Deep / formal
**Share of the window.** Let $s$ be the token length of the system prompt and $W$ the size of the [[Context Window]]. The share of the window used before the user says anything is

$$
f=\frac{s}{W}
$$

**Where:**
- $f$ is the fraction of the window the system prompt takes, from 0 to 1 (above 1 means it does not fit).
- $s$ is the length of the system prompt in tokens, counted after the chat template wraps it.
- $W$ is the window the model runs with, in tokens (in my setup set by `num_ctx` in Ollama, see [[Context Window]]).
- **Matters:** what is left for the chat and the answer is $W-s$. A big $s$ with a small $W$ leaves little room, or none.

**Example (tiny numbers):** $W=4096$.
- $s=150$ (my 100-word prompt): $f=150/4096=\mathbf{0.037}$, about 4 percent.
- $s=600$ (my 400-word prompt): $f=600/4096=\mathbf{0.146}$, about 15 percent.
- With a window of 512 and $s=600$: $f=600/512=\mathbf{1.17}$. It does not fit.
- Meaning: the same prompt is cheap in a big window and impossible in a small one.

**Further notes:**
- **Not a security boundary.** The model reads the system text and the user text as one stream of tokens. A user can try to talk the model out of its rules, so a rule in a system prompt should not be the only guard for anything important ([[Agent Safety]]). I did not test this, and this point is from my general understanding.
- **Where to put a rule.** My test gave a better result for the user turn than the system message on one small model. Larger models are often said to follow the system message well, but I did not check that, so treat it as unverified.
- **Order and length.** Long system prompts can bury the one rule that matters ([[Context Rot]]). I did not test it here.
- **Do not put secrets in it.** The model can repeat what it reads. This is my general understanding, not tested.
- **Not tested:** other models, other rules, several rules at once, long chats where the rule must hold for many turns, speed effects, and templates with no system role.

**What came before, and how this improved on it:**
- **Before:** a plain language model only continued text, so people steered it by writing the start of a document and hoping it carried on in that style ([[Prompt Engineering]]).
- **Its limit:** there was no separate place for the developer's standing instructions. They were mixed with the user's text, so the setup text could be lost or contradicted as the chat grew.
- **What changed:** chat models were trained with roles, and the message at the top, the system message, holds the standing instructions ([[Chat Template]], [[Instruction Tuning]]). The date when chat APIs added a system role is from memory and not checked.
- **How much better:** in my test, replies ended with DONE in 0 of 8 cases with no rule and 5 of 8 with the rule in the system message. Putting the same rule in the user turn gave 8 of 8, so the system slot was not the strongest place on this small model. One run, one rule, one model.

## Related
[[Chat Template]] · [[Prompt Engineering]] · [[Context Window]] · [[Instruction Tuning]] · [[In-Context Learning]] · [[Prompt Caching]] · [[KV Cache]] · [[Agent]] · [[Agent Safety]] · [[Tool Calling]] · [[Structured Output]] · [[Multi-Agent Systems]] · [[Context Rot]] · [[Ollama]] · [[Local CPU Chatbot]] · [[LLM]] · [[Prompt Injection]]

## Sources
- Run on this PC: `sp_demo.py`, with `qwen2.5:1.5b` in Ollama 0.21.2 and the `qwen2.5-0.5b-instruct` tokenizer through transformers 4.46.3 (CPU only). Printed lines above are from my own run (one run each, eight fixed questions).
- My own earlier notes: [[Chat Template]] and [[Context Window]].
- No web page or paper was read for this note. Claims about larger models, security and the history of the system role are from memory and marked so.
- No image: Wikimedia Commons returned only unrelated documents.
- The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm System prompt:::The first message in a chat, written by the developer, that sets the model's role and rules. The model reads it again on every turn.

#flashcards/ml/formula #flashcards/topic/llm Write the share of the window a system prompt uses and explain each symbol.
?
$$f=\frac{s}{W}$$
- $f$: the fraction of the window used by the system prompt, from 0 to 1 (above 1 means it does not fit).
- $s$: the system prompt's length in tokens, counted after the chat template wraps it.
- $W$: the size of the context window in tokens.
- What is left for the chat and the answer is $W-s$.

#flashcards/ml/example #flashcards/topic/llm With $W=4096$, what share does a system prompt of 150 tokens take, and one of 600?::150 / 4096 = 0.037 (about 4 percent). 600 / 4096 = 0.146 (about 15 percent). With a window of 512, 600 / 512 = 1.17, so it does not fit (my arithmetic).

#flashcards/ml/example #flashcards/topic/llm In my run, how many tokens did system prompts of 10, 100 and 400 words add to every call?::15, 150 and 600 tokens. Over 20 turns, 600 per call means 12,000 tokens read again. The words were the repeated phrase "Be kind.", so real text may differ.

#flashcards/ml/concept #flashcards/topic/llm What happens if you give a chat template no system message?::The template may add a default one. In my run the Qwen template added "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

#flashcards/ml/concept #flashcards/topic/llm Why does a long system prompt cost more than it seems?::It is part of every call, because the model starts fresh each turn. Its tokens are read again each time and take room from the context window.

#flashcards/ml/concept #flashcards/topic/llm In my test, where did a small model follow a rule better: the system message or the user turn?::The user turn: it ended with DONE in 8 of 8 replies there, against 5 of 8 with the rule in the system message and 0 of 8 with no rule. One run, one small model, one odd rule, so not a general result.

#flashcards/ml/cloze #flashcards/topic/llm A system prompt is a ==request==, not a lock: the model follows it most of the time, not always.

#flashcards/ml/concept #flashcards/topic/llm Why is a system prompt not a security boundary? (my understanding, not tested)::The model reads the system text and the user text as one stream of tokens, so a user can try to talk it out of the rules. Do not make it the only guard for anything important.

#flashcards/ml/concept #flashcards/topic/llm What did I not test about system prompts?::Other models, other rules, several rules at once, long chats, speed effects, templates with no system role, and attacks that override the rules.
