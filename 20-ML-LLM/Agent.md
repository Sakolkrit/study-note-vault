---
tags: [agent, LLM, tool-use, planning, RAG, reinforcement-learning]
status: growing
level: 4
tier: 4
---

# Agent

## 🟢 1. Simple (age 12)
An AI agent is an AI that **does things, not just talks**. A normal chatbot answers a question and stops. An agent can make a plan, use tools (search the web, run a calculator, open a file), look at what happened, and keep going until the job is done.

**Example:** You ask a friend "what is the weather at the beach tomorrow, and should I take a jacket?" A chatbot answers from memory. An agent is like a helpful assistant: it looks up the forecast, sees it will be 12 degrees, thinks "that is chilly", and tells you "yes, take a jacket". It chose to look something up, read the result, and then decided.

## 🔵 2. How it works
1. You give the agent a **goal** ("book the cheapest train to the city on Friday").
2. The agent's brain is an [[LLM]]. It reads the goal and the list of **tools** it is allowed to use (search, calculator, calendar, run code, read files).
3. **Think:** the model decides the next step, for example "I should search for train times".
4. **Act:** it asks for a tool to be used. The surrounding program runs the tool for real.
5. **Observe:** the tool's result is handed back to the model as new text.
6. Repeat: think, act, observe, think again. When the model decides the job is done, it gives the final answer.
7. A **limit** (a maximum number of steps, a budget, a human "OK" for risky actions) keeps it from running forever or doing something harmful.

This think-act-observe cycle is the same loop as in the classic picture of an agent: it gets information about the world, picks an action, and the action changes the world.

![[agent-diagram.png]]
*The classic picture of an agent: it senses its **environment** (percepts), something in its head decides (the "?"), and **effectors** carry out **actions** that change the environment. For an LLM agent the "?" is the language model, and the tools are the effectors. Image: Tcharon, [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Artificial_Intelligent_Agent.png), CC0.*

## 🟠 3. Technical detail
- **The loop:** a short program (written in [[Python]], for example) that calls the model, runs the tool the model asked for, appends the result to the conversation, and calls the model again.
- **Tool calling (function calling):** the model is shown tool descriptions (name, what it does, what inputs it takes) and replies in a fixed format (often structured text such as JSON) naming a tool and its inputs. Your code runs the tool and sends back the result.
- **Common patterns:**
  - **ReAct:** the model alternates written reasoning ("Thought") with actions ("Act") and results ("Observation").
  - **Plan-then-execute:** write a plan first, then carry it out step by step.
  - **Reflection:** the model critiques its own result and retries.
  - **Multi-agent:** several agents with different roles (planner, coder, reviewer) pass work between them.
- **Memory:** short-term memory is just the conversation so far (limited by the context window). Long-term memory means saving notes or facts outside the model and fetching them later, often with [[RAG]] on top of a [[Vector Database]].
- **Typical tools:** web search, a code runner, file read and write, database queries, calling other programs through an API.
- **Why it gets harder with length:** each step can go wrong, and errors compound (see the maths below). Good agents check results and recover from errors.
- **Safety:**
  - **Prompt injection:** text the agent reads (a web page, an email) can contain instructions that try to hijack it. Treat tool output as untrusted data, not as commands.
  - **Least privilege:** only give the tools and permissions the task needs, and ask a human before irreversible actions such as deleting files or spending money.
  - **Logging:** record every step so you can see what the agent did.
- **Cost and speed:** every loop is another call to the model, so agents are slower and more expensive than a single answer.

```python
tools = {"add": lambda a, b: a + b, "mul": lambda a, b: a * b}
script = [                                    # a stand-in for the LLM's decisions
    {"tool": "mul", "args": [6, 7]},
    {"tool": "add", "args": [42, 8]},
    {"answer": "The result is 50"},
]
def fake_llm(history):                        # a real agent would call a model here
    return script[sum(1 for h in history if h[0] == "tool_result")]

history = [("user", "What is 6*7+8?")]
for step in range(5):                         # hard cap on steps
    out = fake_llm(history)
    if "answer" in out:
        print(out["answer"]); break
    result = tools[out["tool"]](*out["args"])
    print(out["tool"], out["args"], "->", result)
    history.append(("tool_result", result))
# mul [6, 7] -> 42
# add [42, 8] -> 50
# The result is 50
```

## 🔴 4. Deep / formal
**Errors compound over steps.** Suppose each step of a task succeeds with chance $p$, and the steps are independent. All $n$ steps must go right for the whole job to succeed:

$$
P(\text{all steps succeed})=p^{\,n}
$$

**Where:**
- $p$ is the chance that **one** step is done correctly (between 0 and 1).
- $n$ is the number of steps in the task.
- $p^{\,n}$ means $p$ multiplied by itself $n$ times (the chance of independent events all happening is the product of their chances, see [[Probability]]).
- **Matters:** even a high per-step success rate decays quickly as tasks get longer. This is why long agent tasks are fragile, and why checking and recovering after each step matters so much.
- **Caveat:** real steps are not independent. A wrong early step often makes later ones fail, and a good agent can sometimes notice and fix a mistake, so the formula is a rough guide, not an exact prediction.

**Example (tiny numbers):** a 10-step task.
- If $p=0.90$: $0.9^{10}=\mathbf{0.349}$, only about a $35\%$ chance to finish correctly.
- If $p=0.95$: $0.95^{10}=\mathbf{0.599}$, about $60\%$.
- If $p=0.99$: $0.99^{10}=\mathbf{0.904}$, about $90\%$.
- Meaning: raising the per-step reliability from $90\%$ to $99\%$ nearly triples the chance of finishing. A 3-step task at $p=0.9$ is much safer: $0.9^3=0.729$.

**Retrying helps.** If a step can be retried and a success can be recognised (for example a test passes), the chance that at least one of $k$ tries works is:

$$
P(\text{at least one success})=1-(1-p)^{k}
$$

**Where:**
- $p$ is the chance one attempt succeeds, and $k$ is the number of attempts.
- $1-p$ is the chance one attempt fails, and $(1-p)^k$ is the chance that **all** $k$ attempts fail (independent tries).
- Subtracting from 1 gives the chance that not all fail, which means at least one succeeds.
- **Matters:** retries rescue a weak step cheaply, but only if you can tell whether an attempt worked. If you cannot check, you cannot tell which attempt to keep.

**Example (tiny numbers):** a step that works half the time, $p=0.5$.
- 1 try: $\mathbf{0.5}$. 3 tries: $1-0.5^3=\mathbf{0.875}$. 5 tries: $1-0.5^5=\mathbf{0.969}$.
- Meaning: three tries move a coin-flip step to about $88\%$. Retries with a check (a test, a validator) are one of the most reliable ways to make agents work.

**Further notes:**
- **Connection to reinforcement learning:** an agent in the classic sense chooses actions to maximise reward over time. LLM agents are often improved with it: for example, training on whether a task was completed ("verifiable rewards", see [[RLHF]]).
- **Context is the bottleneck:** everything the agent has seen must fit in the model's context window (see [[Tokenization]]). Long tasks need summarising, notes and retrieval.
- **Evaluation is hard:** success depends on the environment, the tools and the model. Measure on realistic tasks and look at the step-by-step logs, not only the final answer.
- **Open problems:** reliable long-horizon planning, defending against prompt injection, knowing when to ask a human, and keeping costs predictable. Claims about what agents can do change fast, so check current benchmarks.

**What came before, and how this improved on it:**
- **Before:** one call to a language model that answers from what it remembers, with no way to check or look anything up.
- **Its limit:** it cannot gather new facts, test its answer, or recover from a wrong first guess (the ReAct abstract names hallucination and error propagation in chain-of-thought reasoning).
- **What changed:** a loop of think, act, observe ([[ReAct]], [[Tool Calling]]). On two interactive benchmarks, ReAct beat imitation and reinforcement learning methods by an absolute success rate of 34% (ALFWorld) and 10% (WebShop) with only one or two examples in the prompt (Yao et al., arXiv 2210.03629, 6 October 2022, abstract read).
- **How much better:** the published numbers are above. The code below is only a toy of why feedback matters: finding a hidden number from 1 to 100, one blind guess is right 1% of the time, while a loop that reads "higher" or "lower" always succeeds, in 5.79 tries on average and 7 at most. A real agent also costs more calls, time and risk of looping.

**Check it yourself:**

```python
import numpy as np
rng = np.random.default_rng(0)
n = 100
one_shot = np.mean([rng.integers(1, n + 1) == rng.integers(1, n + 1) for _ in range(20000)])      # guess a hidden number blindly
def loop():                                                       # use the feedback "higher" or "lower" after each try
    secret, lo, hi, t = rng.integers(1, n + 1), 1, n, 0
    while True:
        g = (lo + hi) // 2; t += 1
        if g == secret: return t
        lo, hi = (g + 1, hi) if g < secret else (lo, g - 1)
ts = [loop() for _ in range(2000)]
print("hidden number 1..100: one blind guess is right", round(one_shot, 3), "of the time")
print("a loop that reads the feedback always succeeds, in", round(np.mean(ts), 2), "tries on average, at most", max(ts))
```

## Related
- [[LLM]] · [[RAG]] · [[Prompt Engineering]] · [[Fine-tuning]] · [[RLHF]] · [[Probability]] · [[Python]] · [[Vector Database]] · [[Embedding Model]] · [[Tokenization]] · [[Transformer]] · [[Q-Learning]] · [[Policy Gradient]] · [[PPO]] · [[Tool Calling]] · [[ReAct]] · [[Model Context Protocol]] · [[Agent2Agent]] · [[Agent Safety]] · [[FastAPI]] · [[LangChain]] · [[LangGraph]] · [[Multi-Agent Systems]] · [[AI Agent with LangGraph]] · [[Structured Output]] · [[Prompt Injection]]

## Sources
- Russell & Norvig, *Artificial Intelligence: A Modern Approach* (the classic agent definition)
- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*, 2022 (from memory)
- Schick et al., *Toolformer*, 2023 (from memory)
- OWASP, *Top 10 for LLM Applications* (prompt injection; check the current version)
- Yao et al., ReAct, arXiv 2210.03629 (abstract read for the history block)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/agents-rag AI agent:::An AI that does things, not just talks. It can make a plan, use tools, look at what happened, and keep going until the job is done.

#flashcards/ml/formula #flashcards/topic/agents-rag Write the chance that all steps of an agent task succeed and explain each symbol.
?
$$P(\text{all steps succeed})=p^{\,n}$$
- $p$: the chance that one step is done correctly (between 0 and 1).
- $n$: the number of steps in the task.
- $p^n$: $p$ multiplied by itself $n$ times, the chance that independent events all happen.
- Even a high per-step reliability gives a low chance for a long task.

#flashcards/ml/formula #flashcards/topic/agents-rag Write the chance that at least one of several attempts succeeds and explain each symbol.
?
$$P(\text{at least one success})=1-(1-p)^{k}$$
- $p$: the chance one attempt succeeds. $k$: the number of attempts.
- $1-p$: the chance one attempt fails. $(1-p)^k$: the chance that all $k$ attempts fail (independent tries).
- Subtracting from 1 gives the chance that not all fail, so at least one succeeds.
- Retries with a check (a test, a validator) are one of the most reliable ways to make agents work.

#flashcards/ml/cloze #flashcards/topic/agents-rag An agent's long task gets ==less== reliable with more steps, and retries with a check make it ==more== reliable.

#flashcards/ml/example #flashcards/topic/agents-rag What is the success chance of a 10-step task for $p=0.90$, $0.95$ and $0.99$?::0.9¹⁰ = 0.349 (about 35%). 0.95¹⁰ = 0.599 (about 60%). 0.99¹⁰ = 0.904 (about 90%). Going from 90% to 99% per step nearly triples the chance.

#flashcards/ml/example #flashcards/topic/agents-rag A step works half the time ($p=0.5$). What is the chance of at least one success in 1, 3 and 5 tries?::1 try: 0.5. 3 tries: 1 − 0.5³ = 0.875. 5 tries: 1 − 0.5⁵ = 0.969.

#flashcards/ml/image #flashcards/topic/agents-rag ![[agent-diagram.png]]
What does this picture show?
?
The classic picture of an agent: it senses its environment (percepts), something in its head decides (the "?"), and effectors carry out actions that change the environment. For an LLM agent the "?" is the language model, and the tools are the effectors.
