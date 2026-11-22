---
tags: [LLM, agents, ReAct, reasoning, tool-use]
status: growing
level: 4
tier: 3
---

# ReAct

Short for **Reason + Act**. A pattern where an [[LLM]] alternates between thinking about what to do and doing it, and looks at what happened before the next step.

## 🟢 1. Simple (age 12)
Suppose you are told "my teddy bear is cold, please do something". A good helper does not act blindly. They **think** ("how cold is the room?"), **act** (check the thermometer), **look** at the result ("65°F, that's chilly"), then **think again** ("turn up the heat by 5 degrees"), **act** (turn it up), and **look** again ("70°F, good"). ReAct is that loop for an AI helper.

**Example:** Your lost-and-found robot gets "find my red scarf". It thinks "check the classroom first", walks there, sees no scarf, thinks "try the gym", walks there, and finds it. It changes its plan based on what it sees.

## 🔵 2. How it works
1. The AI receives an **input**: a user's message, or an event such as a metric going over a limit.
2. **Reason (plan):** the model writes what is known so far and what is needed next, including which tool to call ([[Tool Calling]]).
3. **Act:** it calls the tool (an API, a database search, a document search).
4. **Observe:** the tool's result comes back and is added to the conversation.
5. The loop of reason, act, observe **repeats** until the model decides it has enough, and then it writes the **final answer**.
6. A limit on the number of loops stops it from running forever.
7. An [[Agent]] is a system built around this idea: it pursues a goal by taking several steps by itself. The lecture's picture is a thermostat agent, and shows many such agents that could talk to each other through a standard like Agent2Agent.

## 🟠 3. Technical detail
- **Origin:** Yao et al., 2022. The abstract says earlier work studied reasoning (like [[Chain of Thought]]) and acting separately, and that interleaving them lets reasoning guide the plan and handle exceptions, while actions fetch outside facts (arXiv 2210.03629, read).
- **Why interleave:** reasoning alone can hallucinate and let errors pile up; acting alone has no plan. The abstract says ReAct reduces these problems on question answering (HotpotQA) and fact checking (Fever) by using a simple Wikipedia API.
- **The lecture's loop:** input, then observe, plan, act, output. Each pass through the model may call a tool, so a typical agent call "sees multiple loops".
- **Where things break (lecture):** a wrong tool choice, bad arguments, tool errors, or the final response not reflecting the result ([[Tool Calling]] lists them). Careful debugging is needed.
- **Agents from other agents:** the lecture describes Agent2Agent, a protocol where an agent publishes a card of skills so others can find and call it. I did not read the specification.
- **Safeguards:** a step limit (the code), logging every thought and action (transparency helps trust and debugging, as the lecture says), and keeping the tools safe.
- **In the code:** the "model" is a hand-written rule, not an LLM. It shows only the shape of the loop and the step limit.

```python
room = {'temp': 65, 'target': 70}
# the tools the agent may call
def get_temp(): return room['temp']
def set_temp(delta): room['temp'] += delta; return room['temp']
def fake_llm(history):
    """a stand-in for the language model: it reads the history and returns (thought, action). A real model would write these as text."""
    seen = [h for h in history if h[0] == 'observation']
    if not seen: return 'I do not know the room temperature yet.', ('get_temp', {})
    last = seen[-1][1]
    if last < room['target']: return f'It is {last}F, below the {room["target"]}F target. Raise it.', ('set_temp', {'delta': room['target'] - last})
    return f'It is {last}F, which is enough.', ('finish', {'text': f'The room is now {last}F.'})
def agent(max_steps=6):
    history = [('input', 'My teddy bear is cold.')]
    for step in range(max_steps):                                # Reason, Act, Observe, repeat; the cap stops endless loops
        thought, (name, args) = fake_llm(history)
        print(f'step {step + 1} | thought: {thought} | action: {name}({args})')
        if name == 'finish': return args['text']
        obs = {'get_temp': get_temp, 'set_temp': set_temp}[name](**args)
        history += [('thought', thought), ('observation', obs)]
        print(f'        observation: {obs}')
    return 'stopped: step limit reached'
print('final answer:', agent())
# the safety cap in action: a broken tool that never changes the room
room.update(temp=65); set_temp = lambda delta: room['temp']
print('with a broken thermostat:', agent(max_steps=4))
```

## 🔴 4. Deep / formal
This note stays at level 3: the loop above is a *procedure*, not an equation, and I did not derive any formula for it. The one quantitative point is that a loop with $m$ dependent steps is only as reliable as its weakest steps; the chain arithmetic ($p^m$, for example $0.9^{10}=0.349$) is worked in [[Tool Calling]].

**Trade-offs and edges:**
- **Cost:** every loop is another model call, with the growing history as input. Long loops cost more and are slower.
- **Termination:** with a broken tool, the code's agent keeps choosing the same action. After 4 steps it stops because of the cap, and returns an honest "step limit reached" instead of a made-up success.
- **State:** the history of thoughts and observations is the only memory. If it grows past the context window, older steps must be dropped or summarised ([[LLM]]).
- **Evaluation:** agents are judged on benchmarks that test tool use end to end, and on consistency across repeated tries ([[Pass at k]]).

**What came before, and how this improved on it:**
- **Before:** [[Chain of Thought]] (reasoning with no outside information) and plans written once without looking at results (acting with no reasoning trace).
- **Its limit:** according to the ReAct abstract, chain of thought suffers from hallucination and error propagation, and acting alone is hard to interpret.
- **What changed:** interleave thoughts, actions and observations in one loop (Yao et al., arXiv 2210.03629, 6 October 2022, abstract read).
- **How much better:** the abstract reports that on two interactive decision-making benchmarks, ALFWorld and WebShop, ReAct beats imitation and reinforcement-learning methods by an absolute success rate of 34% and 10%, with only one or two in-context examples. I did not reproduce any of it. In this note's code, the loop reaches the goal in 3 steps with a working thermostat and stops safely at the cap with a broken one.

## Related
- [[Agent]] · [[Tool Calling]] · [[Chain of Thought]] · [[Model Context Protocol]] · [[LLM]] · [[Prompt Engineering]] · [[RAG]] · [[Reasoning Model]] · [[LangChain]] · [[LangGraph]] · [[Multi-Agent Systems]] · [[AI Agent with LangGraph]]

## Sources
- Stanford CME 295 (Fall 2025), Lectures 7 and 8 slides, Afshine and Shervine Amidi
- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, 2022 (arXiv 2210.03629, abstract read)
- Google, *Announcing the Agent2Agent Protocol (A2A)*, 2025 (named in the lecture; not read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/agents-rag ReAct:::An agent loop of Reason, Act, Observe. The AI thinks about what it needs, calls a tool, looks at the result, then thinks again, and repeats until it can write the final answer.

#flashcards/ml/concept #flashcards/topic/agents-rag What are the steps of the ReAct loop?::1. Input arrives. 2. Reason (plan): write what is known and which tool to call. 3. Act: call the tool. 4. Observe: the result comes back and is added to the conversation. 5. Repeat until it has enough, then write the final answer. A limit on the number of loops stops it running forever.

#flashcards/ml/concept #flashcards/topic/agents-rag Why interleave reasoning and acting?::Reasoning alone can hallucinate and let errors pile up. Acting alone has no plan and is hard to interpret. Interleaving lets reasoning guide the plan and handle exceptions, while actions fetch outside facts.

#flashcards/ml/concept #flashcards/topic/agents-rag Who introduced ReAct, and what did the abstract report?::Yao et al., 2022 (arXiv 2210.03629, abstract read). On ALFWorld and WebShop it beat imitation and reinforcement-learning methods by an absolute success rate of 34% and 10%, with only one or two in-context examples. The note did not reproduce this.

#flashcards/ml/concept #flashcards/topic/agents-rag What safeguards does a ReAct loop need?::A step limit, logging every thought and action (transparency helps trust and debugging), and keeping the tools safe. With a broken tool the note's code stops after 4 steps and honestly says "step limit reached" instead of inventing a success.

#flashcards/ml/cloze #flashcards/topic/agents-rag In ReAct, the history of ==thoughts and observations== is the only memory, so it must be summarised if it grows past the ==context window==.
