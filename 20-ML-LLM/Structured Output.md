---
tags: [LLM, structured-output, JSON, decoding, agents, inference]
status: growing
level: 4
tier: 2
---

# Structured Output

**Structured output** means making a language model answer in a fixed shape, such as JSON with named fields, so a program can read the answer without guessing. The most reliable way is a **schema**, a description of the allowed shape that the software enforces while the model is writing.

## 🟢 1. Simple (age 12)
Imagine a form with boxes. If you ask a friend to "tell me about yourself", you get a story that is hard to sort. If you hand them a form with a box that only allows "cat", "dog" or "other", you always get one of those three. Structured output is giving the AI a form instead of a blank page.

**Example:** a robot needs to know which helper to call next. You do not want a chatty paragraph. You want one word from a short list, in a shape the robot can read every time.

## 🔵 2. How it works
1. A program often needs the model's answer as **data**: a label, a number, a list of fields. A paragraph of friendly text is hard to read by code.
2. **Level one, just ask.** You write in the prompt "reply with JSON like this" ([[Prompt Engineering]]). Often it works, but nothing forces it.
3. **Level two, give a schema.** You hand the software a description of the allowed shape: which fields, which kinds of values, which words are allowed. The software then controls what the model may write.
4. **How the control works (my understanding, not checked in the docs).** The model scores every possible next token ([[Softmax]]). The software throws out every token that would break the shape and picks only among the rest ([[Decoding Strategies]]). Part 2 of my run shows this on real scores.
5. **The result is shape-safe, not truth-safe.** The reply will fit the form, but the value inside can still be wrong. A form can guarantee "one of cat, dog or other". It cannot guarantee it is the right one.
6. **Agents lean on this.** A router that must name the next helper, or a step that must hand back one expression to calculate, needs a reply that code can read ([[Agent]], [[Tool Calling]], [[LangGraph]]).

I did not add a picture. A search of Wikimedia Commons for this topic returned only unrelated papers, so I used none, as the vault rules say.

## 🟠 3. Technical detail
- **Where I used it before.** My agent project asked Ollama for a reply in the shape `{"next": <one of the allowed helpers>}` and for `{"expression": <text>}` ([[AI Agent with LangGraph]]). In the code, that is the `format=` argument of `ollama.chat`, which takes a schema.
- **What I measured.** `so_demo.py` (below) asks `qwen2.5:1.5b` to sort ten questions into math, history or other, three ways: (a) with a plain prompt and no mention of JSON, (b) with a prompt that shows a JSON example, and (c) with the plain prompt plus a schema that allows exactly those three labels. A reply counts as good if it parses as JSON and the label is one of the three. Part 2 then looks at real next-token probabilities of the 0.5B model, and applies a mask by hand.

**`so_demo.py`** (needs `transformers`, the model folder from [[Local CPU Chatbot]], and Ollama running with `qwen2.5:1.5b`)
```python
"""Structured output, measured: ask for JSON with and without a schema, then show the mask idea on real next-token scores. CPU only."""
import json, urllib.request
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

LABELS = ["math", "history", "other"]
QUESTIONS = ["What is 12 times 7?", "Who built the pyramids of Giza?", "What is the capital of Peru?", "Solve x + 5 = 9.",
             "When did World War II end?", "How do I bake bread?", "What is 15 percent of 80?", "Who was Napoleon?",
             "Write a haiku about rain.", "What is the square root of 144?"]
PROMPT = 'Classify the question as one of math, history or other. Reply with JSON like {"label": "math"}.'
PLAIN = "Classify the question as one of math, history or other."     # the same task, with no word about JSON
SCHEMA = {"type": "object", "properties": {"label": {"type": "string", "enum": LABELS}}, "required": ["label"]}


def ask(question, schema, prompt):                           # one local model call; schema=None means free text
    body = {"model": "qwen2.5:1.5b", "stream": False, "format": schema, "options": {"temperature": 0, "seed": 1, "num_predict": 40},
            "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": question}]}
    req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))["message"]["content"]


def valid(text):                                     # does it parse, and is the label one of the three?
    try:
        return json.loads(text)["label"] in LABELS
    except Exception:
        return False


for name, schema, prompt in (("plain prompt", None, PLAIN), ("JSON prompt", None, PROMPT), ("with schema", SCHEMA, PLAIN)):
    replies = [ask(q, schema, prompt) for q in QUESTIONS]
    print(f"1 {name:12}: {sum(valid(r) for r in replies)} of {len(QUESTIONS)} replies parse and use an allowed label; first reply: {replies[0]!r}")
    print("  ", [json.loads(r)["label"] if valid(r) else "BAD" for r in replies])

PATH = "qwen2.5-0.5b-instruct"
tok = AutoTokenizer.from_pretrained(PATH)
model = AutoModelForCausalLM.from_pretrained(PATH, torch_dtype=torch.float32).eval()
ids = tok.apply_chat_template([{"role": "user", "content": "Is the sky blue? Answer with one word, yes or no."}], add_generation_prompt=True, return_tensors="pt")
with torch.no_grad():
    p = torch.softmax(model(ids).logits[0, -1], dim=-1)     # probability of every possible next token
allowed = [tok.encode(w)[0] for w in ("Yes", "No")]          # the only tokens a schema would let through here
print("2 top 3 next tokens:", [(tok.decode(i), round(p[i].item(), 4)) for i in p.topk(3).indices])
raw = {tok.decode(i): p[i].item() for i in allowed}
total = sum(raw.values())
print("2 allowed, raw probability:", {k: round(v, 4) for k, v in raw.items()}, "| sum", round(total, 4))
print("2 allowed, after the mask :", {k: round(v / total, 4) for k, v in raw.items()}, "| sum", round(sum(v / total for v in raw.values()), 4))
```
What it printed (transformers 4.46.3, Ollama 0.21.2, Python 3.9.6, CPU only, `temperature` 0, `seed` 1):
```
1 plain prompt: 0 of 10 replies parse and use an allowed label; first reply: 'Math'
   ['BAD', 'BAD', 'BAD', 'BAD', 'BAD', 'BAD', 'BAD', 'BAD', 'BAD', 'BAD']
1 JSON prompt : 10 of 10 replies parse and use an allowed label; first reply: '{"label":"math"}'
   ['math', 'history', 'history', 'math', 'history', 'other', 'other', 'history', 'other', 'math']
1 with schema : 10 of 10 replies parse and use an allowed label; first reply: '{ "label": "math" }'
   ['math', 'history', 'history', 'math', 'history', 'other', 'math', 'history', 'other', 'math']
2 top 3 next tokens: [('Yes', 0.7661), ('No', 0.2252), ('Blue', 0.0068)]
2 allowed, raw probability: {'Yes': 0.7661, 'No': 0.2252} | sum 0.9914
2 allowed, after the mask : {'Yes': 0.7728, 'No': 0.2272} | sum 1.0
```
What to look for:
- **Plain prompt:** 0 of 10 usable. The model answered with a bare word like `Math`, which is not JSON. This is no surprise, since the prompt never asked for JSON. It shows why a program cannot rely on free text.
- **JSON prompt:** 10 of 10 usable. This model is good at the easy case, so asking nicely worked here. That is a result for this small test only. I did not look for questions that make it fail, and a weaker model or a harder shape may break more often.
- **Schema:** also 10 of 10, and the reply shape is enforced, not hoped for. Note that the labels are not the same as in the JSON-prompt run: the question "What is 15 percent of 80?" was labelled `other` with the JSON prompt and `math` with the schema (position 7 in the lists). So the schema can change *what* is said, not only how it looks. I did not check why.
- **Part 2:** the top next tokens for "Is the sky blue? Answer with one word, yes or no." were `Yes` 0.7661 and `No` 0.2252. These two sum to 0.9914, so 0.0086 of the probability sat on other tokens such as `Blue`. After the mask, the same two tokens get 0.7728 and 0.2272, which sum to 1.

## 🔴 4. Deep / formal
**Masking.** Let $p_i$ be the model's probability for token $i$, and let $A$ be the set of tokens the shape allows at this step. The software uses the new probabilities

$$
p'_i=\frac{p_i\,\mathbf 1[i\in A]}{\sum_{j\in A}p_j}
$$

**Where:**
- $p_i$ is the model's own probability for token $i$, before any control.
- $\mathbf 1[i\in A]$ is 1 when token $i$ is allowed and 0 when it is not. So every forbidden token gets exactly zero.
- $\sum_{j\in A}p_j$ is the total probability on the allowed tokens. Dividing by it makes the allowed tokens add up to 1 again.
- $p'_i$ is the probability used for choosing the next token.
- **Matters:** without the 0/1 filter a forbidden token can still be picked. Without the division the numbers no longer add up to 1 and cannot be sampled from. Each allowed token keeps its relative rank, so the model's preference is kept inside the allowed set.

**Example (tiny numbers):** three tokens with $p=(0.6,\,0.3,\,0.1)$, where only the first two are allowed.
- The allowed total is $0.6+0.3=0.9$.
- $p'_1=0.6/0.9=\mathbf{0.667}$ and $p'_2=0.3/0.9=\mathbf{0.333}$. The third token gets $\mathbf{0}$.
- Meaning: the ranking is the same (first beats second two to one), the forbidden token is gone, and the rest add to 1.

My real case uses the same rule: $0.7661/0.9914=0.7728$ and $0.2252/0.9914=0.2272$ (my arithmetic, matching part 2 of the output).

**Further notes:**
- **Shape-safe is not truth-safe.** A valid label can be wrong. My test did not grade correctness, only shape, so the table above says nothing about accuracy. The one visible difference (`other` against `math` for the 15 percent question) shows that two valid answers can differ.
- **A schema can steal room from the model.** If the allowed set leaves no good answer, the model must still pick something allowed. Make the set cover the real cases, including a catch-all such as `other`.
- **A cut-off reply breaks the shape.** If the reply is cut by a token limit (`num_predict` in my code) before the closing bracket, it will not parse. My limit of 40 tokens was enough for these short replies.
- **How a program turns a shape into a token mask** (grammars, state machines over tokens) is the engine part. I did not read or test how Ollama does it, so the description above is my understanding only.
- **Other ways to get structure:** training the model to produce a format, asking it to call a named function ([[Tool Calling]]), or re-asking when parsing fails. I tested none of them.
- **Not tested:** other models, harder or nested shapes, a long answer with many fields, speed cost of the mask, and correctness of the labels.

**What came before, and how this improved on it:**
- **Before:** the program asked in plain words for a format and then used string tricks, such as searching the reply for a word or cutting out the part between braces.
- **Its limit:** the model could add chat around the answer, rename a field, or use a word that is not in the list, and the program broke.
- **What changed:** the shape is checked while the answer is written, not afterwards, so a reply that does not fit the shape cannot be produced.
- **How much better:** in my test, 0 of 10 plain-prompt replies were usable JSON, and the schema run gave 10 of 10. The JSON-prompt run also gave 10 of 10, so on this easy task the schema's gain over a good prompt was **zero** in the count. Its value is the guarantee: it removes the chance of a bad shape rather than lowering it. Dates and first uses of this idea in other tools are not given, since I did not check them.

## Related
[[Tool Calling]] · [[Agent]] · [[LangGraph]] · [[LangChain]] · [[Multi-Agent Systems]] · [[Ollama]] · [[Decoding Strategies]] · [[Softmax]] · [[Prompt Engineering]] · [[Hallucination]] · [[Chat Template]] · [[Hugging Face Transformers]] · [[Local CPU Chatbot]] · [[AI Agent with LangGraph]] · [[LLM]] · [[System Prompt]]

## Sources
- Run on this PC: `so_demo.py`, with `qwen2.5:1.5b` in Ollama 0.21.2 and `qwen2.5-0.5b-instruct` through transformers 4.46.3 (CPU only). All printed lines above are from my own run (one run each, ten fixed questions).
- My agent project, [[AI Agent with LangGraph]], which used the same `format=` schema argument.
- No web page or paper was read for this note. How Ollama enforces a schema is my understanding, not checked.
- No image: Wikimedia Commons returned nothing on topic.
- The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/llm Structured output:::Making a model answer in a fixed shape, such as JSON with named fields, so a program can read the reply. A schema enforces the shape while the model writes.

#flashcards/ml/formula #flashcards/topic/llm Write the masked probability used to force a shape and explain each symbol.
?
$$p'_i=\frac{p_i\,\mathbf 1[i\in A]}{\sum_{j\in A}p_j}$$
- $p_i$: the model's own probability for token $i$.
- $\mathbf 1[i\in A]$: 1 if the shape allows token $i$, else 0, so forbidden tokens get exactly zero.
- $\sum_{j\in A}p_j$: the total probability on allowed tokens; dividing by it makes them add up to 1 again.
- $p'_i$: the probability used to choose the next token. The ranking among allowed tokens is kept.

#flashcards/ml/example #flashcards/topic/llm Three tokens have $p=(0.6,0.3,0.1)$ and only the first two are allowed. What are the new probabilities?::Allowed total 0.6 + 0.3 = 0.9. New values: 0.6 / 0.9 = 0.667, 0.3 / 0.9 = 0.333, and 0 for the third. The ranking stays and the sum is 1.

#flashcards/ml/example #flashcards/topic/llm In my run, Yes had 0.7661 and No had 0.2252. What were they after the mask to those two tokens?::Allowed total 0.9914. Yes: 0.7661 / 0.9914 = 0.7728. No: 0.2252 / 0.9914 = 0.2272. They now sum to 1 (my arithmetic, matching the printed output).

#flashcards/ml/concept #flashcards/topic/llm How many of ten questions got a usable JSON label with a plain prompt, a JSON prompt and a schema in my test?::0 of 10 (plain prompt: a bare word like Math), 10 of 10 (JSON prompt), 10 of 10 (schema). On this easy task, asking nicely worked as well as the schema. Small model, one run.

#flashcards/ml/concept #flashcards/topic/llm Does a schema guarantee a correct answer?::No. It guarantees the shape only. In my test the question about 15 percent of 80 got other with the JSON prompt and math with the schema, so two valid answers can differ. I did not grade correctness.

#flashcards/ml/cloze #flashcards/topic/llm Structured output is ==shape-safe==, not ==truth-safe==: the reply fits the form, but the value inside can still be wrong.

#flashcards/ml/concept #flashcards/topic/llm How does a schema force the shape while the model writes? (my understanding, not checked)::The software removes every next token that would break the shape and chooses only among the rest, after renormalising the probabilities so they add up to 1 again.

#flashcards/ml/concept #flashcards/topic/llm Why can a long structured reply still fail to parse?::If a token limit cuts it off before the closing bracket, it is not complete JSON. My limit of 40 tokens was enough for these short replies.

#flashcards/ml/concept #flashcards/topic/llm Why include a catch-all such as other in an allowed list?::If the allowed set has no good answer, the model must still pick something allowed and can pick a wrong one. The set should cover the real cases.

#flashcards/ml/concept #flashcards/topic/llm What did I not test about structured output?::Other models, harder or nested shapes, long replies with many fields, the speed cost of the mask, correctness of the labels, and how Ollama enforces a schema inside.
