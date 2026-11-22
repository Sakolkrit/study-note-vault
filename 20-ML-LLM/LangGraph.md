---
tags: [LLM, agents, framework, Python, graph, workflow, state]
status: growing
level: 4
tier: 3
---

# LangGraph

**LangGraph** is a Python library for building programs as a map of small steps. Each step is a function, the arrows say which step comes next, and a shared "notebook" (the state) is passed along. It is mostly used to build [[Agent]] systems that loop, branch, remember and wait for a human.

## 🟢 1. Simple (age 12)
Think of a board game where you move a token from square to square. Each square tells you to do one small job, and some squares say "if this, go left, otherwise go right". A clipboard travels with the token, and every square can read it and write on it. LangGraph lets you draw that board for an AI helper, and then it moves the token for you.

**Example:** a homework helper has three squares: "look up facts", "do the maths" and "write the answer". A boss square at the start looks at the question and sends the token to the right square. When the answer is written, the game ends.

## 🔵 2. How it works
1. **Make the state.** The state is the shared clipboard. You list its fields, like `question`, `notes` and `answer`.
2. **Write nodes.** A node is a normal Python function. It reads the state and returns only the fields it wants to change.
3. **Say how changes are merged.** For each field you pick a rule. The usual rule is "the new value replaces the old one". For a list you can pick "add the new items to the end", so notes from different nodes pile up instead of erasing each other. That rule is called a reducer.
4. **Connect the nodes with edges.** A plain edge says "after A, always go to B". A conditional edge runs a small function that looks at the state and picks the next node. A conditional edge that points back to an earlier node makes a loop.
5. **Compile it.** `compile()` checks the map and gives you a runnable program.
6. **Run it.** `invoke` runs to the end and gives the final state. `stream` hands you each node's change as it happens.
7. **Add a safety net.** A loop that never ends is stopped by a step limit.
8. **Optional: remember and pause.** With a checkpointer, LangGraph saves the state after each step under a thread name, so a second call on the same thread continues where the first stopped, and the run can be paused before a node and resumed later (for example to ask a human).

![[langgraph-supervisor-graph.png]]
*A LangGraph drawing: `__start__` leads to a router node, `detect_and_route`, whose dotted arrows go to four nodes (`notion_agent`, `pdf_processor`, `process`, `url_extractor`), and each of those leads to `__end__`. In my own drawing from the [[AI Agent with LangGraph]] project, dotted arrows were the conditional edges, so I read these the same way. Image by LuisMonBo on [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:LangGraph_Supervisor_graph_example.png), CC0 (public domain).*

## 🟠 3. Technical detail
- **Install.** `pip install langgraph`. On Python 3.9 pip chose version 0.6.11 here (the docs I read describe the 1.x series, so some options may differ). Importing the checkpoint module printed a `LangChainPendingDeprecationWarning` about a future default, which I left alone.
- **The pieces, with the names used in code.** `StateGraph(State)` makes the builder, where `State` is a `TypedDict`. `add_node(name, function)`, `add_edge(a, b)` and `add_conditional_edges(a, picker, mapping)` draw the map. `START` and `END` are the two special ends. `Annotated[list, operator.add]` marks a field with a reducer.
- **What a node returns.** A dict with only the fields to change. Other fields stay as they were. Equation 1 says how each field is updated.

$$s_{t+1}[k]=f_k\big(s_t[k],\,u_t[k]\big)$$

**Where:**
- $s_t[k]$ is the value of field $k$ before step $t$ runs.
- $u_t[k]$ is the value a node returned for field $k$ in that step (a field the node does not return stays unchanged).
- $f_k$ is the merge rule (reducer) chosen for field $k$: the default keeps the new value, and `operator.add` joins two lists.
- $s_{t+1}[k]$ is the value of the field after the step.

**Matters:** if a list field has no adding rule, each write replaces the whole list, and the notes from an earlier node are lost.

**Example (tiny numbers):** field `n` has the default rule. Before the step $n=1$, and the node returns $n=2$, so the new value is 2. Field `log` uses `operator.add`. Before the step it is `['step 1']` and the node returns `['step 2']`, so the new value is `['step 1', 'step 2']`. The demo below ends with $n=3$ and a log of four entries, one per node run.

**`lg_demo.py`** (needs only `langgraph`, no model)
```python
"""LangGraph in five small steps: state + reducer, a loop, streaming, memory by thread, pause and resume. No model needed."""
import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph


class S(TypedDict):
    n: int                                # no reducer: a new value overwrites the old one
    log: Annotated[list, operator.add]    # reducer: a new list is added to the old one


def step(s):    return {"n": s["n"] + 1, "log": [f"step {s['n'] + 1}"]}    # a node returns a partial update
def done(s):    return {"log": ["done"]}


def make(checkpointer=None, stop=3, **compile_kw):
    g = StateGraph(S)
    g.add_node("step", step)
    g.add_node("done", done)
    g.add_edge(START, "step")
    g.add_conditional_edges("step", lambda s: "step" if s["n"] < stop else "done", {"step": "step", "done": "done"})   # the edge function picks the next node
    g.add_edge("done", END)
    return g.compile(checkpointer=checkpointer, **compile_kw)


# 1. state + reducer + a loop made by a conditional edge
app = make()
print("1 invoke :", app.invoke({"n": 0, "log": []}))

# 2. stream: one update per node that ran
print("2 stream :", [(k, v) for u in app.stream({"n": 0, "log": []}, stream_mode="updates") for k, v in u.items()])

# 3. a loop that never ends is stopped by the recursion limit
try:
    make(stop=10 ** 9).invoke({"n": 0, "log": []}, {"recursion_limit": 5})
except GraphRecursionError as e:
    print("3 limit  :", type(e).__name__)

# 4. memory: the same thread_id keeps the saved state, another id starts fresh
mem = make(MemorySaver())
a, b = {"configurable": {"thread_id": "a"}}, {"configurable": {"thread_id": "b"}}
mem.invoke({"n": 0, "log": []}, a)
r = mem.invoke({"n": 3, "log": ["second question"]}, a)       # same thread: n restarts at 3 (we passed it), log keeps growing
print("4 thread a:", r["log"])
print("4 thread b:", mem.invoke({"n": 0, "log": []}, b)["log"])
print("4 saved a :", len(list(mem.get_state_history(a))), "checkpoints")

# 5. pause before a node, look at the saved state, then resume with None as the input
hold = make(MemorySaver(), interrupt_before=["done"])
c = {"configurable": {"thread_id": "c"}}
print("5 paused :", hold.invoke({"n": 0, "log": []}, c)["log"], "| next node:", hold.get_state(c).next)
print("5 resumed:", hold.invoke(None, c)["log"])
```
What it printed (langgraph 0.6.11, Python 3.9.6, in a fresh virtual environment):
```
1 invoke : {'n': 3, 'log': ['step 1', 'step 2', 'step 3', 'done']}
2 stream : [('step', {'n': 1, 'log': ['step 1']}), ('step', {'n': 2, 'log': ['step 2']}), ('step', {'n': 3, 'log': ['step 3']}), ('done', {'log': ['done']})]
3 limit  : GraphRecursionError
4 thread a: ['step 1', 'step 2', 'step 3', 'done', 'second question', 'step 4', 'done']
4 thread b: ['step 1', 'step 2', 'step 3', 'done']
4 saved a : 10 checkpoints
5 paused : ['step 1', 'step 2', 'step 3'] | next node: ('done',)
5 resumed: ['step 1', 'step 2', 'step 3', 'done']
```
What to look for: in line 1 the conditional edge sent the token back to `step` until `n` reached 3, and the list reducer kept every log entry. Line 2 shows the partial updates: `done` returned only `log`, and `n` was never touched. Line 3 is the safety net: with a limit of 5 the endless loop raised `GraphRecursionError`. In line 4, thread `a` kept its first run (the first four entries), and my second call on the same thread added its own entry plus two more nodes; thread `b` started from nothing. I did not work out why thread `a` shows 10 saved checkpoints. In line 5 the run stopped before `done` with `done` as the next node, and calling it again with `None` as input finished it. `MemorySaver` keeps its data in the program's memory only, so the saved state vanishes when the program ends (from the name and my reading, I did not test it). The deprecation warning showed on the first run; I ran it a second time with warnings switched off and got the same lines.

## 🔴 4. Deep / formal
- **Where the design comes from.** The project's README says it is inspired by Pregel and Apache Beam, and that its public interface draws on NetworkX (read on 18 November 2026 through a page summary). Pregel is a Google system for big graph jobs that runs in rounds (from my memory, not checked, and the docs word "super-step" is also from memory). The step limit counts these rounds, which fits the way my limit of 5 stopped the loop.
- **The step limit differs between versions.** My endless loop stopped at 25 steps in 0.6.11 ([[AI Agent with LangGraph]]), and the 1.x docs summary I read gives a default near 10,000. Always set it on purpose.
- **Memory is by thread.** A thread name is the key to a saved conversation. Two threads never see each other's state (line 4 above). Keeping that state across program runs needs a saver that writes to disk or a database, which I did not test.
- **Pause and resume is built in.** `interrupt_before` stops the run, `get_state(...).next` shows what would run next, and the saved state can be read, or changed, before the run goes on. That is the "human in the loop" idea, here in its simplest form (I did not test editing the state, or the newer `interrupt` call).
- **A graph is a cost as well as a help.** For a straight line of steps, plain functions are shorter. A graph pays off when you need loops, branches, saved state, or a pause. In my project the team of workers did not beat one plain model on 10 questions, so the structure helps with control, not with answer quality by itself.
- **What I did not test:** `Command` and `Send` hand-offs, sub-graphs, running branches in parallel, database savers, the prebuilt agent helpers and tool nodes, token streaming, LangGraph's hosted tools and LangSmith tracing, and the 1.x versions.
- **Safety.** An agent loop can call tools and spend money in a cycle; the step limit and a pause for approval are the two guards shown here ([[Agent Safety]]).

**What came before, and how this improved on it:**
- **Before:** programs that chain steps in a straight line, such as `langchain` chains, and hand-written `while` loops for agents ([[ReAct]]). The early 0.0.x pages of this package on PyPI describe it as an extension of LangChain Expression Language for cyclic, multi-step coordination (from search-result summaries of those pages, not opened).
- **Its limit:** a chain has no loops, and a hand-written loop mixes the control logic with the data, so pausing, saving and drawing it are all extra work.
- **What changed:** the program becomes a graph with named nodes, a shared typed state with merge rules, conditional edges that can loop, and saved checkpoints.
- **How much better:** I have no measured comparison, because I did not write the hand-made version. What I measured is that one 55-line file ran a loop, streaming, a step limit, thread memory and pause-and-resume, and all five printed the expected lines above.

## Related
[[LangChain]] · [[Agent]] · [[ReAct]] · [[Tool Calling]] · [[Agent Safety]] · [[Agent2Agent]] · [[Graphs]] · [[Prompt Engineering]] · [[LLM]] · [[Multi-Agent Systems]] · [[Hugging Face Transformers]] · [[Ollama]] · [[AI Agent with LangGraph]] · [[Structured Output]]

## Sources
- Run on this PC: langgraph 0.6.11 in the virtual environment `D:\python456\agents-langgraph\.venv` (Python 3.9.6); every printed line above is from my own run of `lg_demo.py`.
- The LangGraph README (https://github.com/langchain-ai/langgraph), read on 18 November 2026 through a page summary: it calls LangGraph a low-level orchestration framework for long-running, stateful agents, licence MIT, inspired by Pregel and Apache Beam, interface inspired by NetworkX, and lists durable execution, human-in-the-loop, memory, debugging with LangSmith and deployment as benefits. I did not test those benefits beyond the demo.
- The early 0.0.x PyPI pages, as search-result summaries only (not opened). The 1.x docs on the step limit, from a page summary (see [[AI Agent with LangGraph]]).
- From my memory, not checked: Pregel as a Google system, the word "super-step", and that the saved state of `MemorySaver` is lost when the program ends.
- Image: "LangGraph Supervisor graph example.png" by LuisMonBo, CC0, https://commons.wikimedia.org/wiki/File:LangGraph_Supervisor_graph_example.png. I viewed it before writing the caption.
- The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/agents-rag LangGraph:::A Python library that builds a program as a graph: nodes are functions, edges say what runs next, and a shared state is passed along. It supports loops, branches, saved memory and pausing.

#flashcards/ml/image #flashcards/topic/agents-rag ![[langgraph-supervisor-graph.png]]
What does this picture show?
?
A LangGraph drawing: `__start__` leads to a router node `detect_and_route`, whose dotted arrows go to four nodes (`notion_agent`, `pdf_processor`, `process`, `url_extractor`), each leading to `__end__`. By LuisMonBo, CC0 (Wikimedia Commons).

#flashcards/ml/concept #flashcards/topic/agents-rag What are the four building blocks of a LangGraph program?
?
- The state: the shared clipboard of fields.
- Nodes: functions that read the state and return only the fields they change.
- Edges: plain ones (always go to B) and conditional ones (a function picks the next node).
- Compile and run: `compile()` checks the map, then `invoke` or `stream` runs it.

#flashcards/ml/concept #flashcards/topic/agents-rag What is a reducer in LangGraph?::The rule that merges a node's returned value into a state field. The default replaces the old value, and `operator.add` on a list adds the new items to the end.

#flashcards/ml/formula #flashcards/topic/agents-rag How does LangGraph update a state field after a node runs, and what does each symbol mean?
?
$$s_{t+1}[k]=f_k\big(s_t[k],\,u_t[k]\big)$$
- $s_t[k]$: the value of field $k$ before step $t$.
- $u_t[k]$: the value a node returned for that field (an unreturned field stays unchanged).
- $f_k$: the merge rule (reducer) for field $k$; the default keeps the new value.
- $s_{t+1}[k]$: the value after the step.
- Without an adding rule on a list field, each write replaces the whole list and earlier notes are lost.

#flashcards/ml/example #flashcards/topic/agents-rag Field `log` is `['step 1']`, uses `operator.add`, and a node returns `['step 2']`. What is `log` after the step?::`['step 1', 'step 2']` (the lists are joined). A field `n` with the default rule, which was 1 and returned as 2, becomes 2.

#flashcards/ml/example #flashcards/topic/agents-rag In the demo, the conditional edge loops until `n` reaches 3. What was the final state?::`{'n': 3, 'log': ['step 1', 'step 2', 'step 3', 'done']}`, from my run on langgraph 0.6.11.

#flashcards/ml/concept #flashcards/topic/agents-rag How does LangGraph stop a loop that never ends?::With a recursion (step) limit. In my demo a limit of 5 raised `GraphRecursionError`. The default was 25 in 0.6.11, and the 1.x docs summary gives about 10,000, so set it on purpose.

#flashcards/ml/concept #flashcards/topic/agents-rag How does LangGraph remember a conversation, and how do two conversations stay apart?
?
A checkpointer saves the state after each step under a thread name. A second call with the same thread name continues from the saved state, and another thread name starts fresh. `MemorySaver` keeps it only while the program runs (not tested).

#flashcards/ml/concept #flashcards/topic/agents-rag How do you pause a LangGraph run for a human and then continue?
?
Compile with a checkpointer and `interrupt_before=["node"]`. The run stops before that node and `get_state(config).next` shows it. Calling `invoke(None, config)` finishes the run (my run: `('done',)` was next, then the log got `done`).

#flashcards/ml/cloze #flashcards/topic/agents-rag A LangGraph node returns only ==the fields it wants to change==, and the other fields keep their old values.

#flashcards/ml/concept #flashcards/topic/agents-rag Where does LangGraph's design come from, according to its README?::It says it is inspired by Pregel and Apache Beam, and its public interface draws on NetworkX (from a page summary). Pregel as a Google system is from my memory.

#flashcards/ml/concept #flashcards/topic/agents-rag When is a graph worth it, and when are plain functions enough?::A graph pays off with loops, branches, saved state or a pause. For a straight line of steps plain functions are shorter. In my project the team of workers did not beat one plain model, so the graph helps control, not answer quality by itself.

#flashcards/ml/concept #flashcards/topic/agents-rag What did I not test about LangGraph?::`Command` and `Send` hand-offs, sub-graphs, parallel branches, database savers, prebuilt agents and tool nodes, token streaming, LangSmith tracing, editing a paused state, and the 1.x versions.
