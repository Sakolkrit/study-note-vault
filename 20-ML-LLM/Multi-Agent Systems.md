---
tags: [LLM, agents, multi-agent, orchestration, supervisor, cost]
status: growing
level: 4
tier: 3
---

# Multi-Agent Systems

A **multi-agent system** is an AI program made of several [[Agent]]s that each do one job and work together on a task, usually under a boss agent that decides who works next. The idea is to split a hard job into parts, the way a team does, instead of asking one AI to do everything.

## 🟢 1. Simple (age 12)
Imagine a class project. One kid is good at looking things up, one is good at maths, one is good at writing. A team leader reads the question, hands it to the right kid, and a final kid writes up the answer. Each kid has a short, clear job. But if the leader hands out jobs badly, the project goes wrong, and every extra handover takes extra time.

**Example:** "What is 17 × 23 + 5, and what is a token?" The leader sends the maths part to the maths kid and the word part to the research kid, then the writer puts both answers together.

## 🔵 2. How it works
1. **Each agent gets one job**, a short prompt and only the tools it needs ([[Tool Calling]]). That keeps its instructions short and its choices few.
2. **Something decides who works next.** In the *supervisor* style, one agent only chooses. In a *router* style, a first step sorts the request and sends it to one or more specialists. In a *handoff* style, an agent passes the job to another one itself.
3. **The agents share information.** Either they all read and write one shared notebook (the state in [[LangGraph]]), or they only send each other messages.
4. **A last agent writes the final answer** from what the others found.
5. **Guards stop trouble.** A step limit stops endless loops, and a rule that a worker may run once stops the boss from sending the same job around again.
6. **Every handover costs a model call.** More agents means more calls, more waiting and more cost, so a team has to earn its keep.
7. **A single agent is often enough.** The LangChain docs say not every complex task needs several agents, and that one agent with the right tools and prompt can often do as well.

I did not add a picture: Wikimedia Commons had nothing on-topic about teams of language-model agents (the search found robot path-finding figures and old scanned reports).

## 🟠 3. Technical detail
- **My own team** ([[AI Agent with LangGraph]]): a supervisor, a researcher (word-overlap search over this vault's notes, no model call), a calculator (one model call to write an expression, then a safe evaluator) and a writer. The graph is a [[LangGraph]] `StateGraph`, with a step limit of 5 and a rule that no worker runs twice.
- **Counting model calls.** Each supervisor turn is one call. In my team the calculator costs one call and the researcher costs none. The writer costs one call. So the calls for one question are counted by the rule below.

Number of model calls N = s + c + 1, where s = k + 1 (see the terms below).

**Terms:**
- N is the number of model calls for one question.
- s is the number of supervisor calls. The supervisor is asked once for each worker it sends, and once more when it picks the writer, so s=k+1.
- k is the number of workers that ran before the writer (not counting the writer).
- c is the number of model calls the workers made themselves. In my team only the calculator makes one.
- The final +1 is the writer's own call.

**Matters:** the cost of a team grows with every worker you add, even before any of them does real work. If you leave out the supervisor's calls you under-count by k+1, which is half of all the calls in my `v1` run (14 of 28).

**Example (tiny numbers):** a question that goes straight to the writer has k=0 and c=0, so s=1 and N=1+0+1=2. A maths question that goes calculator then writer has k=1 and c=1, so s=2 and N=2+1+1=4. Without a team the model is called once.

**`calls.py`** (the counting script, run with `python calls.py MODEL PROMPT` in the [[AI Agent with LangGraph]] project folder; it needs that project's `run.py` and `agents.py`, whose code is in that note)
```python
"""Count model calls and seconds per question for run.py's ten questions. python calls.py MODEL PROMPT (PROMPT: none, v1, v2)"""
import sys

model, prompt = sys.argv[1], sys.argv[2]
sys.argv = ["run.py", model, "eval", prompt]
import run # builds the graph for this model and prompt

n = {"calls": 0}
original = run.chat
def counted(*a, **k):
    n["calls"] += 1
    return original(*a, **k)
run.chat = counted # route, expression and write look up chat in run's globals

total_calls = total_s = 0
for q, need, words in run.CASES:
    n["calls"] = 0
    s, t = run.ask(q)
    total_calls += n["calls"]; total_s += t
    print(f"{n['calls']} calls {t:5.1f}s trace {s['trace']} {q[:40]}")
print(f"{model} prompt {prompt}: {total_calls} model calls for {len(run.CASES)} questions ({total_calls / len(run.CASES):.1f} per question), {total_s:.0f} s ({total_s / len(run.CASES):.1f} s per question)")
```
What it printed for `qwen2.5:1.5b` on the ten questions of that project (one run each, temperature 0):
```
prompt none: 10 model calls for 10 questions (1.0 per question), 33 s (3.3 s per question)
prompt v1: 28 model calls for 10 questions (2.8 per question), 108 s (10.8 s per question)
```
In the `v1` run the four calculator questions each took 4 calls (13 to 18 s), the other six took 2 calls each (6 to 11 s), and the traces show that the researcher did not run on any question, not even the three knowledge questions (the supervisor kept picking the writer). So 4 × 4 + 6 × 2 = 28, which the call-count rule predicts for every question.

What to look for: the team made 2.8 times as many model calls and took 3.3 times as long (108 / 33 = 3.27, my arithmetic). The answers were no better overall in that project (8 of 10 for the team with prompt v1, against 6 of 10 alone, by a crude word check, and the team was mostly the plain model because it skipped the researcher). The extra cost bought one clear gain: the calculator got 3 of 3 arithmetic questions right where the plain model got 1 of 3. These are ten questions, one run each. The timings are one run on one CPU, and I did not repeat them.

## 🔴 4. Deep / formal
- **Patterns, from the LangChain docs I read** (page summary): *subagents* (a main agent calls other agents as tools, and all routing goes through it), *handoffs* (tool calls change which agent is active), *skills* (one agent loads specialised prompts on demand and stays in control), *router* (a step classifies the input and sends it to specialists), and *custom workflows* built in [[LangGraph]]. My team is closest to subagents with a supervisor. The page does not use the word "supervisor", so that name here is mine and the project's.
- **When the docs say it is worth it:** managing context (giving specialised knowledge without filling one prompt, see [[Context Rot]]), letting different teams build parts on their own, running workers in parallel, and when one agent has too many tools and picks badly. I tested none of these reasons.
- **The costs the docs list:** every extra call adds waiting and money, total tokens can reach context limits, subagents add a call on easy tasks because results flow back through the main agent, handoffs run one after another, and routers need a routing call on each request. My call count and my timings show the first of these in a small case.
- **A weak boss ruins the team.** With a small model the supervisor chose the writer first on all three demo questions and picked the right workers on 5 of 10 questions with prompt v1 ([[AI Agent with LangGraph]]). A supervisor that always says "writer" is the same as no team, plus the cost of asking it.
- **Examples in the boss's prompt can backfire:** the 0.5B model copied my example order on 8 of 10 questions in that project.
- **Failures that multiply.** If each handover has some chance of going wrong, a longer chain of handovers has a lower chance of everything going right. I did not measure that chance, so this is a reasoning point and not a result.
- **Standards for agents that live in different programs:** [[Agent2Agent]] lets separate agents find each other and hand over work, and [[Model Context Protocol]] connects an agent to tools. I did not use either here.
- **Safety:** agents that call tools can chain actions, so limits on steps, tools and money matter more as teams grow ([[Agent Safety]]).
- **Limits of this note:** my data comes from one small project: two small local models, ten questions, one run each, a crude word check. I did not test larger models, hosted models, parallel workers, agents that talk to each other directly, or any published benchmark of multi-agent systems.

**What came before, and how this improved on it:**
- **Before:** one language model answering in one go, and then a single agent that loops over its tools ([[ReAct]]).
- **Its limit:** one prompt must hold all instructions and all tools, a long prompt can confuse the model, and one agent cannot work on two parts at the same time (the docs' reasons, summarised above).
- **What changed:** the job is split among several agents with short prompts and few tools each, joined by a router, a boss or handoffs.
- **How much better:** in my project, not better overall: 8 of 10 answers for the best team row against 6 of 10 alone (crude word check, one run), at 2.8 times the model calls and 3.3 times the time. The team did fix arithmetic (3 of 3 against 1 of 3). Published results for multi-agent systems are not in this note, because I did not look for them.

## Related
[[Agent]] · [[LangGraph]] · [[LangChain]] · [[Agent2Agent]] · [[Model Context Protocol]] · [[Tool Calling]] · [[ReAct]] · [[Agent Safety]] · [[Context Rot]] · [[Prompt Engineering]] · [[LLM]] · [[Ollama]] · [[AI Agent with LangGraph]] · [[Context Window]] · [[Structured Output]] · [[System Prompt]] · [[Prompt Injection]]

## Sources
- Run on this PC: `calls.py` with the project code from [[AI Agent with LangGraph]] (langgraph 0.6.11, Python 3.9.6, Ollama with `qwen2.5:1.5b`). The call counts and times above are from my own run, and the other results (5 of 10, 8 of 10, 3 of 3, 8 of 10 copied order) are from that project note's runs.
- The LangChain multi-agent page (https://docs.langchain.com/oss/python/langchain/multi-agent), read on 18 November 2026 through a page summary: when to use several agents, the five patterns, and the costs. The word "supervisor" does not appear on it, according to the summary.
- No image: nothing on-topic on Wikimedia Commons.
- The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/agents-rag Multi-agent system:::A program made of several AI agents that each do one job and work together, usually with a boss (supervisor) or a router deciding who works next.

#flashcards/ml/concept #flashcards/topic/agents-rag What are the patterns the LangChain docs name for several agents? (page summary)::Subagents (a main agent calls others as tools), handoffs (tool calls change the active agent), skills (one agent loads specialised prompts on demand), router (a step sends the input to specialists), and custom workflows built in LangGraph.

#flashcards/ml/concept #flashcards/topic/agents-rag When do the LangChain docs say several agents are worth it?::To manage context (specialised knowledge without overloading one prompt), for teams building parts independently, to run workers in parallel, and when one agent has too many tools and chooses poorly. They also say one agent with the right tools and prompt can often do as well.

#flashcards/ml/concept #flashcards/topic/agents-rag How many model calls does one question cost in my supervisor team, and what does each symbol mean?
?
Number of model calls N = s + c + 1, where s = k + 1 (see the terms below).
- N: model calls for one question.
- s: supervisor calls: one for each worker it sends, plus one when it picks the writer.
- k: workers that ran before the writer.
- c: model calls made by the workers themselves (only the calculator makes one).
- The final +1 is the writer's call.
- Leaving out the supervisor's calls under-counts by k+1, which was half of all calls (14 of 28) in my `v1` run.

#flashcards/ml/example #flashcards/topic/agents-rag A question goes straight to the writer ($k=0$, $c=0$). How many model calls?::$s=0+1=1$, so $N=1+0+1=2$ (my arithmetic, matching the runs). A maths question that goes calculator then writer ($k=1$, $c=1$) has $s=2$ and $N=4$.

#flashcards/ml/example #flashcards/topic/agents-rag What did the team of qwen2.5:1.5b cost against no team on ten questions?::28 model calls against 10 (2.8 against 1.0 per question), and 108 s against 33 s (10.8 against 3.3 s per question, so 3.3 times the time). One run each on one CPU.

#flashcards/ml/example #flashcards/topic/agents-rag Why did the ten questions make 28 calls?::The four calculator questions took 4 calls each and the other six took 2 each: 4 × 4 + 6 × 2 = 28. The researcher ran on none of them, because the supervisor kept picking the writer.

#flashcards/ml/concept #flashcards/topic/agents-rag What did the extra calls buy in my project?::Arithmetic: the team got 3 of 3 right against 1 of 3 for the plain model. On all ten questions the best team row got 8 of 10 against 6 of 10 alone (crude word check, one run), and it was mostly the plain model.

#flashcards/ml/cloze #flashcards/topic/agents-rag A team is only as good as its ==supervisor==, and a supervisor that always says "writer" is the same as no team.

#flashcards/ml/concept #flashcards/topic/agents-rag Why can examples in the supervisor's prompt backfire on a small model?::The 0.5B model copied my example order (calculator, then researcher) on 8 of 10 questions instead of choosing for each question.

#flashcards/ml/concept #flashcards/topic/agents-rag What costs do the LangChain docs list for several agents? (page summary)::More latency and money for every call, more total tokens that can hit context limits, an extra call on easy tasks for subagents, handoffs that run one after another, and a routing call on every request for routers.

#flashcards/ml/concept #flashcards/topic/agents-rag What did I not test about multi-agent systems?::Larger or hosted models, parallel workers, agents talking to each other directly, other patterns such as handoffs and routers, and published benchmarks. My data is two small local models, ten questions and one run each.
