---
tags: [LLM, agents, tool-use, function-calling, API]
status: growing
level: 4
tier: 3
---

# Tool Calling

Also called **function calling**. Letting an [[LLM]] ask a program to do something for it, such as search, calculate or send a message, and then use the result.

## 🟢 1. Simple (age 12)
A chatbot only knows what it read during training, and it cannot *do* anything in the real world. Tool calling gives it **a toolbox**. When you ask "find a teddy bear near me", the AI cannot know that itself, but it can say "please run the *find bear* tool with this location". A normal program runs the tool, hands back the answer, and the AI explains it to you.

**Example:** You ask a friend for the weather. They cannot see outside, so they check a weather app and tell you what it says. The AI is the friend, the app is the tool.

## 🔵 2. How it works
1. The developer describes each tool to the model: its **name**, what it does, and what **arguments** it needs (the lecture's example is a function that finds the nearest teddy bear from GPS coordinates, with a clear description).
2. The user asks something. The model reads the question and the list of tools.
3. **Step 1, choose and fill in:** the model decides whether a tool helps and writes out the call, such as `find_teddy_bear(location=(37.42, -122.17))`, as structured text.
4. **Step 2, run it:** ordinary program code around the model runs the real function. The model does not run anything itself.
5. **Step 3, explain:** the result goes back to the model, which writes the final answer for the user.
6. The model can be **taught to use tools** in two ways: by training on examples of tool calls, or just by describing the tools in the prompt ([[Prompt Engineering]]).
7. Typical tools: search, weather or stock lookups, a calculator, running code, sending emails, and anything else an assistant could do.
8. This also fixes the "knowledge cutoff" problem, because a tool can fetch fresh facts. [[RAG]] is a special case: the tool is "search my documents".

## 🟠 3. Technical detail
- **Where it sits:** a loop between model output, a validator, a real function, and the model again. The model only produces *text that looks like a call*.
- **Validation matters:** check that the output is valid JSON, that the tool exists, and that the arguments are complete and of the right type. The code does these checks and reports why each bad call fails.
- **Challenges (lecture):** more tools mean worse performance, since descriptions fill the context and confuse the model; and defining many tools is a lot of work. One remedy is a **tool selector** (a router) that shows the model only a few relevant tools; the lecture cites a paper on this that I did not read.
- **Shared standard:** [[Model Context Protocol]] lets one tool implementation work with many models.
- **Failure modes the lecture lists:**
  1. *Tool prediction errors:* the model does not use a tool, invents one, picks the wrong one, or fills in a wrong argument.
  2. *Tool call errors:* the tool errors or returns a wrong value, or returns nothing (a hallucinated final answer often follows).
  3. *Response errors:* the model ignores the tool's result, the result floods the context, or the output is hard to interpret.
  Remedies include better tool names and descriptions, training, a stronger model, trimming tool output, and returning something meaningful even when empty.
- **Safety:** tools can change the real world, so mistakes or attacks matter. The lecture names data exfiltration as an example risk and mentions training steps, inference safeguards and safety benchmarks.
- **Research:** Toolformer (Schick et al.) trains a model to decide which APIs to call, when, and with what arguments, in a self-supervised way (arXiv 2302.04761 abstract, read).

```python
import json
# the tools we describe to the model: name, what each argument must be, and the function itself
def find_teddy_bear(location): return {'name': 'Teddy', 'distance_m': round(((location[0] - 37.42) ** 2 + (location[1] + 122.17) ** 2) ** 0.5 * 111000)}
def get_room_temperature(): return {'fahrenheit': 65}
TOOLS = {
    'find_teddy_bear': {'args': {'location': list}, 'fn': find_teddy_bear},
    'get_room_temperature': {'args': {}, 'fn': get_room_temperature},
}
def run(call_text):
    """what the program around the model does with the model's tool request (a JSON string)"""
    try: call = json.loads(call_text)
    except ValueError: return {'error': 'not valid JSON'}
    tool = TOOLS.get(call.get('name'))
    if tool is None: return {'error': 'unknown tool ' + repr(call.get('name'))}           # the model invented a tool
    args = call.get('arguments', {})
    for a, kind in tool['args'].items():
        if a not in args: return {'error': 'missing argument ' + a}
        if not isinstance(args[a], kind): return {'error': 'argument ' + a + ' must be ' + kind.__name__}
    try: return {'result': tool['fn'](**args)}                                              # run the real function
    except Exception as e: return {'error': 'tool failed: ' + str(e)}
for text in ('{"name": "find_teddy_bear", "arguments": {"location": [37.42, -122.17]}}',
             '{"name": "get_room_temperature"}',
             '{"name": "find_bear", "arguments": {"location": [37.42, -122.17]}}',
             '{"name": "find_teddy_bear", "arguments": {}}',
             '{"name": "find_teddy_bear", "arguments": {"location": "Paris"}}',
             '{"name": "find_teddy_bear", "arguments": {"location": [1]}}',
             'find a bear near me'):
    print(text[:72].ljust(74), '->', run(text))
# a task that needs several tool steps in a row succeeds only if every step does
print('chance of finishing a chain of steps when each works 90% of the time:', {n: round(0.9 ** n, 3) for n in (1, 3, 6, 10)})
```

## 🔴 4. Deep / formal
**A chain of tool steps only works if every step does.** If a task needs $m$ steps and each step succeeds with chance $p$, independently:

$$
P(\text{task succeeds})=p^{m}
$$

**Where:**
- $p$ is the chance that one step works (the model picks the right tool, fills in the right arguments, the tool runs and the model reads the result well).
- $m$ is the number of steps in a row.
- $p^m$ multiplies the chances, because all steps must work.
- **Matters:** small error rates add up quickly: this is why the lecture spends so long on failure modes and on evaluation, and why simple, reliable tools help. Independence is an assumption; real failures are linked.

**Example (tiny numbers):** $p=0.9$.
- 3 steps: $0.9^3=\mathbf{0.729}$.
- 6 steps: $0.9^6=\mathbf{0.531}$.
- 10 steps: $0.9^{10}=\mathbf{0.349}$ (the code prints the same).
- Meaning: a step that works 90% of the time gives a 10-step job that succeeds barely one time in three. This is the same idea as pass^k in [[Pass at k]].

**Further notes:**
- **What the code shows:** seven model outputs go through the validator. Two valid calls run. The other five give specific messages: not JSON, unknown tool (an invented name), missing argument, wrong argument type, and a tool that failed on a list with one number. The last one passed validation and failed inside the tool, so validation catches shape errors but not every wrong value.
- **What the validator cannot catch:** a well-formed call with a *wrong meaning*, such as coordinates $(0,0)$ for "near me". The lecture treats this as a modelling problem (give the model the right information, or a helper tool).
- **Not covered:** parallel tool calls, streaming, tool results that need authorisation, and the training methods in detail.

**What came before, and how this improved on it:**
- **Before:** a language model that only writes text from what it learned in training, with its knowledge fixed at a cutoff date, and hand-built pipelines for each integration.
- **Its limit:** it cannot get fresh facts, cannot calculate reliably, and cannot act. The Toolformer abstract notes models "struggle with basic functionality, such as arithmetic or factual lookup, where much simpler and smaller models excel".
- **What changed:** the model writes a structured call, a program runs it, and the result returns to the model (Schick et al., arXiv 2302.04761, 9 February 2023, abstract read, for the training approach; the prompting approach is from the lecture).
- **How much better:** the Toolformer abstract reports substantially better zero-shot results, often competitive with much larger models, but no figures. In this note's code, a validator turns all 5 faulty outputs among 7 into specific error messages the model can act on (2 outputs were valid), and the arithmetic shows how fast reliability falls over many steps: $0.349$ after 10 steps at 90% each.

## Related
- [[Agent]] · [[ReAct]] · [[Model Context Protocol]] · [[RAG]] · [[LLM]] · [[Prompt Engineering]] · [[Pass at k]] · [[Fine-tuning]] · [[Python]] · [[Agent2Agent]] · [[Agent Safety]] · [[FastAPI]] · [[LangChain]] · [[LangGraph]] · [[Multi-Agent Systems]] · [[AI Agent with LangGraph]] · [[Structured Output]] · [[Prompt Injection]]

## Sources
- Stanford CME 295 (Fall 2025), Lectures 7 and 8 slides, Afshine and Shervine Amidi
- Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools*, 2023 (arXiv 2302.04761, abstract read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/agents-rag Tool calling:::Giving a chatbot a toolbox. The AI cannot do things itself, but it can say "please run this tool with these arguments", a normal program runs it, and the result goes back to the AI.

#flashcards/ml/formula #flashcards/topic/agents-rag Write the chance that a chain of tool steps all succeed and explain each symbol.
?
$$P(\text{task succeeds})=p^{m}$$
- $p$: the chance that one step works (the model picks the right tool, fills in the right arguments, the tool runs and the model reads the result well).
- $m$: the number of steps in a row.
- $p^m$: the chances multiply, because all steps must work.
- Small error rates add up quickly: a 90%-reliable step gives a long job that often fails.

#flashcards/ml/cloze #flashcards/topic/agents-rag A long chain of tool steps fails often because the per-step chances ==multiply==.

#flashcards/ml/example #flashcards/topic/agents-rag With $p=0.9$, what is the success chance of a 3, 6 and 10 step job?::0.9³ = 0.729. 0.9⁶ = 0.531. 0.9¹⁰ = 0.349. A step that works 90% of the time gives a 10-step job that succeeds barely one time in three.
