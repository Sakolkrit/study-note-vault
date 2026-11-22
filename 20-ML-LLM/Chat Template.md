---
tags: [LLM, chat, tokenization, transformer, prompt, inference]
status: growing
level: 4
tier: 2
---

# Chat Template

A **chat template** is the recipe that turns a conversation (a list of "who said what") into the single string of text that a [[LLM]] really reads. The model only continues text, so the template adds special marker words that show where each speaker starts and stops, and where the model's own turn begins.

## 🟢 1. Simple (age 12)
A chatbot does not really "chat". It is a machine that reads some text and writes what probably comes next. To make it act like a chat partner, we first write the conversation down like a play script, with the name of each speaker and a clear sign for "your turn now". The chat template is the rule for how to write that script.

**Example:** A script for two people says "KID: Hi!" then "MUM: Hello." then "KID: Name one colour." then "MUM:". A reader who sees the empty "MUM:" at the end knows it is Mum's turn and writes her line. If you left out the "MUM:" at the end, the reader might just carry on the kid's sentence instead.

## 🔵 2. How it works
1. **You keep the conversation as a list.** Each item has a role (`system`, `user` or `assistant`) and the words (`content`). The `system` item is the instruction that sets how the bot should behave ([[Prompt Engineering]]).
2. **The template glues the list into one string.** Each message is wrapped in marker words. In the format called ChatML, a message looks like `<|im_start|>user`, the words, then `<|im_end|>`.
3. **The template adds the "your turn" marker at the end.** That is `<|im_start|>assistant`, with nothing after it. This is called the generation prompt.
4. **The [[Tokenization]] step turns the string into numbers.** The marker words are special tokens, each with its own single number.
5. **The model continues the text from there.** It writes the answer and finishes by writing `<|im_end|>`, which is the signal to stop ([[Decoding Strategies]]).
6. **Next turn, the whole list is formatted again**, including the new question, and the model reads everything from the top. That is how a chatbot "remembers" ([[KV Cache]] makes the re-reading cheaper).
7. **Each model has its own template**, because each was trained ([[Instruction Tuning]]) with its own markers. Using the wrong one makes the model act strangely.

No picture: I searched Wikimedia Commons for chat templates and ChatML and found no free, on-topic image, so I added none.

## 🟠 3. Technical detail
- **It is a small program, not a fixed string.** In Hugging Face's `transformers` library the template is a Jinja program stored with the tokenizer (`tokenizer.chat_template`) and run by `apply_chat_template`. It loops over the messages and prints markers around them.
- **Three switches matter.** `tokenize=False` gives the string so you can read it, and `tokenize=True` gives token numbers. `add_generation_prompt=True` adds the "assistant, your turn" marker, which you want when you are about to generate. `add_generation_prompt=False` is for training data, where the answer is already in the chat.
- **Defaults hide inside templates.** The Qwen template I used adds its own system message ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") when you give none.
- **The template can also carry tools.** Qwen's template has parts for function-calling (a list of tools and a `<tool_call>` format), which my demo does not use.
- **The same idea lives in three places in this vault's setup.** The Hugging Face tokenizer has a Jinja template. The [[GGUF]] file has a copy under the key `tokenizer.chat_template`. [[Ollama]] uses its own template written in Go's template language, kept as a separate layer of the model.

**`chat_template_demo.py`**
```python
"""What a chat template does, shown on the Qwen2.5-0.5B-Instruct tokenizer (run from the project folder)."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("qwen2.5-0.5b-instruct")
chat = [{"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi!"},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "Name one colour."}]

text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print(text)
print("--- the same chat, without add_generation_prompt: ends with", repr(tok.apply_chat_template(chat, tokenize=False)[-20:]))
print("--- no system message given:", repr(tok.apply_chat_template(chat[1:2], tokenize=False)[:75]))

ids = tok.apply_chat_template(chat, add_generation_prompt=True)
print("--- tokens:", len(ids), "| special tokens:", {t: tok.convert_tokens_to_ids(t) for t in ("<|im_start|>", "<|im_end|>")})
print("first 8 tokens:", tok.convert_ids_to_tokens(ids[:8]))

# the model with the template, and the same words with no template at all
model = AutoModelForCausalLM.from_pretrained("qwen2.5-0.5b-instruct", torch_dtype=torch.float32).eval()
def run(prompt_ids, n=30):
    x = torch.tensor([prompt_ids])
    out = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=n, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][len(prompt_ids):], skip_special_tokens=True)
print("--- with the template   :", repr(run(ids)))
print("--- raw words, no template:", repr(run(tok("Be brief. Hi! Hello. Name one colour.")["input_ids"])))
```
What it printed (library warnings filtered out, and not read):
```
<|im_start|>system
Be brief.<|im_end|>
<|im_start|>user
Hi!<|im_end|>
<|im_start|>assistant
Hello.<|im_end|>
<|im_start|>user
Name one colour.<|im_end|>
<|im_start|>assistant

--- the same chat, without add_generation_prompt: ends with 'e colour.<|im_end|>\n'
--- no system message given: '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpfu'
--- tokens: 34 | special tokens: {'<|im_start|>': 151644, '<|im_end|>': 151645}
first 8 tokens: ['<|im_start|>', 'system', 'Ċ', 'Be', 'Ġbrief', '.', '<|im_end|>', 'Ċ']
--- with the template   : 'Red.'
--- raw words, no template: ' Red. What is your name? I\'m not sure what you mean by "one colour". Can you please provide more context or clarify your question?'
```
What to look for: the template wrote 4 messages plus the open assistant turn, each marker is one token (numbers 151644 and 151645), and the model answered "Red." with the template but wandered on with the same words and no template.

## 🔴 4. Deep / formal
- **Why a template is needed at all.** Every [[LLM]] that writes text is a model that continues a sequence of tokens. A chat model is a base model trained further ([[Instruction Tuning]]) on conversations written in one fixed layout. At use time the layout must be the same, or the text no longer looks like what it was trained on. The Hugging Face guide says the same: a chat is still just a sequence of tokens, different models use different markers, and wrong markers mean "drastically worse performance".
- **Cost.** The formatting is not free. My 4-message chat is 34 tokens, and the same words with no markers are 11 tokens, so the markers and the default layout added 23 tokens (my count with this tokenizer). Every turn repeats them, and the history grows each turn.
- **Double special tokens.** Templates already include any start or end tokens the model needs. If you format with `tokenize=False` and then tokenize the string, the tokenizer may add another start token, which is usually a mistake. The guide advises `add_special_tokens=False` in that case, or `tokenize=True`.
- **Prefill.** Setting `continue_final_message` removes the final end marker, so the model carries on from half a reply you wrote. You cannot combine it with `add_generation_prompt`.
- **Stop token.** The model ends its turn by writing `<|im_end|>`. In the [[GGUF]] file I read, `tokenizer.ggml.eos_token_id` is 151645, the same number as `<|im_end|>` in the Hugging Face tokenizer.
- **Two copies can drift.** The template stored in the GGUF file and the one in the Hugging Face tokenizer are not identical: they differ by two characters, in the example text for tool calls (the GGUF copy has double braces `{{` and `}}` where the other has single ones). I did not find out why or whether it changes any output, and I only ran the Hugging Face one.
- **Special-token text from users.** If user text contains something that looks like a marker, a careless tokenizer may turn it into the real special token, which can fake a speaker change. This is from memory and I did not test it here.
- **Limits.** One prompt, greedy decoding, one small model: my "template against no template" run is an example, not a measurement of how much worse the model gets.

**What came before, and how this improved on it:**
- **Before:** prompts were plain text with no agreed layout. People wrote things like "Q: ... A:" by hand (from memory) and hoped the model would copy the pattern, and every library built its own text format.
- **Its limit:** with no shared layout, different models needed different hand-made prompts, and it was easy to get one wrong. The model also could not tell the system's instructions from the user's words.
- **What changed:** a chat-markup layout with role markers. OpenAI's `openai-python` repository describes a "ChatML" format with `<|im_start|>` and `<|im_end|>` (a file marked v0, and an Azure page marked preview; I did not find the original announcement date). Hugging Face then put the layout into the tokenizer as a template, so one call, `apply_chat_template`, builds the right text for each model.
- **How much better:** the same prompt words gave "Red." with the template and a rambling reply without it (one greedy run on this PC). The cost is 23 extra tokens for a 4-message chat. I have no published figure for the quality gain.

## Related
[[LLM]] · [[Instruction Tuning]] · [[Tokenization]] · [[Prompt Engineering]] · [[Decoding Strategies]] · [[KV Cache]] · [[GGUF]] · [[Ollama]] · [[llama.cpp]] · [[Transformer]] · [[Hugging Face Transformers]] · [[Local CPU Chatbot]] · [[Context Window]] · [[System Prompt]]

## Sources
- Hugging Face Transformers docs, *Chat templates* (https://huggingface.co/docs/transformers/en/chat_templating), read on 17 November 2026 through a page summary, not word for word. It is the docs for a newer release (v5.x) than the 4.46.3 I ran, so my code uses the older return type.
- The OpenAI `openai-python` repository file `chatml.md` and Microsoft's Azure page *Chat Markup Language ChatML (Preview)*, seen only as search-result summaries on 17 November 2026. I did not open either page.
- Run on this PC: Qwen2.5-0.5B-Instruct (Apache-2.0) with transformers 4.46.3, and the GGUF file from Ollama. The "from memory" marks are in the text above.
- No image was found. The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm Chat template:::The recipe that turns a list of role and content messages into the single string of text a language model reads, with marker words around each message and an "assistant, your turn" marker at the end.

#flashcards/ml/concept #flashcards/topic/llm Why does a chatbot need a chat template?::The model only continues text. It was trained on conversations in one fixed layout, so the chat must be written in that layout, or the text no longer looks like its training data and the answers get worse.

#flashcards/ml/concept #flashcards/topic/llm What does `add_generation_prompt=True` add, and when do you want it?::The marker that starts an assistant turn (in ChatML, `<|im_start|>assistant`). You want it when you are about to generate a reply, and not for training data where the answer is already in the chat.

#flashcards/ml/concept #flashcards/topic/llm In ChatML, what do `<|im_start|>` and `<|im_end|>` do?::They wrap each message: `<|im_start|>` plus the role begins it and `<|im_end|>` ends it. The model writes `<|im_end|>` to finish its own turn, which is also its stop signal.

#flashcards/ml/example #flashcards/topic/llm A 4-message chat is 34 tokens with the Qwen template and 11 tokens as plain words. How many tokens does the formatting add?::34 minus 11 is 23 tokens (my count with this tokenizer). They repeat on every turn.

#flashcards/ml/concept #flashcards/topic/llm Where do the three copies of a model's chat template live in a local setup like this one?::In the Hugging Face tokenizer (a Jinja template), in the GGUF file under the key `tokenizer.chat_template`, and in Ollama as its own Go template layer.

#flashcards/ml/concept #flashcards/topic/llm How did the GGUF copy of the Qwen template differ from the Hugging Face copy?::By two characters, in the tool-call example text (double braces instead of single). I did not find out why or whether it changes any output.

#flashcards/ml/concept #flashcards/topic/llm What went wrong when the same words were given with no template?::The model (greedy, one run) wandered on with "Red. What is your name? I'm not sure what you mean...", while with the template it answered "Red." This is one example, not a measurement.

#flashcards/ml/concept #flashcards/topic/llm Why use `add_special_tokens=False` after formatting with `tokenize=False`?::The template already includes the start and end tokens the model needs, so tokenizing again can add duplicates, which hurts the model (the advice in the Hugging Face guide).

#flashcards/ml/concept #flashcards/topic/llm What did the Qwen template do when no system message was given?::It added its own default system message, "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
