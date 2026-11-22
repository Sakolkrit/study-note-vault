---
tags: [LLM, agents, framework, Python, chains, prompts, integrations]
status: growing
level: 4
tier: 2
---

# LangChain

**LangChain** is a Python library of ready-made building blocks for programs that use a language model: prompt templates, connections to many model providers, tools, and a way to join the pieces into a chain. Its main idea is that every piece has the same shape, so you can snap them together and swap one for another.

## 🟢 1. Simple (age 12)
Think of a set of toy pipe pieces. One piece pours in your question, one piece sends it to the AI, one piece cleans up the answer. Every piece has the same size of opening, so you can join them in any order and swap a piece for a different one without rebuilding the rest. LangChain is the box of pipe pieces for AI programs.

**Example:** you build a pipe that takes a topic, puts it into a question ("What is a token?"), asks a model, and returns just the text of the answer. Tomorrow you want a different model. You swap that one piece, and the rest of the pipe stays the same.

## 🔵 2. How it works
1. **A prompt template** is a sentence with blanks, like "What is {topic}?". You hand it a topic, and it gives back the messages to send ([[Prompt Engineering]]).
2. **A model piece** wraps one language model ([[LLM]]) behind a standard interface, so a model from one provider looks like a model from another. In real use it talks to a hosted model or to a local one such as [[Ollama]].
3. **A parser piece** turns the model's reply into the form you want, for example plain text.
4. **The pipe sign `|` joins pieces into a chain.** The output of one piece goes into the next.
5. **Every piece has the same methods**: run once (`invoke`), run on a list (`batch`) and give results as they arrive (`stream`).
6. **Plain functions can be pieces too**, and several pieces can run side by side on the same input.
7. **A chain is a straight line.** It has no loops and no "go back". For loops, branches, saved state and pauses you need the graph tool, [[LangGraph]], which the LangChain docs say its agents are built on.
8. **Tools and search plug in the same way.** A tool is something the model can ask the program to run ([[Tool Calling]]), and a search step can fetch notes for the model to use ([[RAG]]). I did not run those here.

I did not add a picture. The only LangChain diagram on Wikimedia Commons that I found looked computer-made (a very clean, glossy design, uploaded as "own work" in 2026), and the vault rule is real images only, so I deleted it.

## 🟠 3. Technical detail
- **Which package I used.** `langchain-core` 0.3.86, the small base package that holds the piece types. It was already in my LangGraph virtual environment, because `langgraph` needs it (pip says it is required by `langgraph`, `langgraph-prebuilt` and `langgraph-checkpoint`). I did **not** install the `langchain` package itself. Another Python on this PC has an old `langchain` 0.0.275 (seen earlier in this vault's work, not used here).
- **The docs I read describe a newer line.** Its main entry point is `create_agent` (model, tools, prompt, middleware) and its agents are "built on top of LangGraph" (docs summary). I did not run `create_agent`, so everything below is the older, lower-level piece-and-pipe style.
- **Names used in the code.** `ChatPromptTemplate`, `StrOutputParser`, `RunnableLambda` and `RunnableParallel` are the pieces. The word *Runnable* is LangChain's name for anything with `invoke`, `batch` and `stream`. `FakeListChatModel` is a stand-in model that returns replies from a list, so the demo needs no model and no internet.

**`lc_demo.py`** (needs only `langchain-core`)
```python
"""The LangChain idea in a few lines, using only langchain-core (already installed with langgraph). A fake model stands in for a real one."""
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel

# 1. a prompt template is a function from a dict to messages
prompt = ChatPromptTemplate.from_messages([("system", "Answer in one short sentence."), ("human", "What is {topic}?")])
print("1 messages:", [(m.type, m.content) for m in prompt.invoke({"topic": "a token"}).messages])

# 2. the pipe | joins steps into a chain; every step has the same methods (invoke, batch, stream)
model = FakeListChatModel(responses=["A token is a piece of text.", "Attention lets a model look back."])
chain = prompt | model | StrOutputParser()
print("2 chain   :", type(chain).__name__, "->", chain.invoke({"topic": "a token"}))
print("2 batch   :", chain.batch([{"topic": "a token"}, {"topic": "attention"}]))
print("2 stream  :", list(RunnableLambda(lambda d: d["topic"].upper()).stream({"topic": "abc"})))

# 3. any plain function becomes a step, and a dict of steps runs side by side
words = RunnableLambda(lambda d: len(d["topic"].split()))
both = RunnableParallel(topic=lambda d: d["topic"], n_words=words)
print("3 parallel:", both.invoke({"topic": "the KV cache"}))

# 4. the chain is a straight line: its output is the last step's output, and there is no loop in it
print("4 steps   :", [type(s).__name__ for s in chain.steps])
```
What it printed (langchain-core 0.3.86, Python 3.9.6):
```
1 messages: [('system', 'Answer in one short sentence.'), ('human', 'What is a token?')]
2 chain   : RunnableSequence -> A token is a piece of text.
2 batch   : ['Attention lets a model look back.', 'A token is a piece of text.']
2 stream  : ['ABC']
3 parallel: {'topic': 'the KV cache', 'n_words': 3}
4 steps   : ['ChatPromptTemplate', 'FakeListChatModel', 'StrOutputParser']
```
What to look for: the template turned `{topic}` into two messages, a system one and a human one. The `|` made a `RunnableSequence` of three steps (line 4 lists them). The fake model returned its replies, not a real answer, so this shows the plumbing and nothing about quality. In the `batch` line the two replies came back in swapped order: the fake model hands out its list in the order calls arrive, and `batch` runs calls in parallel, so the order is not guaranteed (my reading; I did not look into it further). The `stream` line shows only one piece because that step is a plain function that returns everything at once. The `parallel` line shows two steps run on the same input, with the word count of "the KV cache" equal to 3.

## 🔴 4. Deep / formal
- **What the pipe really is.** `a | b` builds a new object that runs `a`, then feeds its output to `b`. It is a way to write function composition, so a chain of three steps behaves like one function with the same methods. That is why swapping a piece is easy: the pieces agree on the shape of their input and output.
- **What a chain cannot do.** It is a straight line. A loop that retries, a branch that depends on the result, saved state across turns and a pause for a human all need something more, and LangChain's own docs point to [[LangGraph]] for them. In my own team project the extra structure did not make the answers better by itself ([[AI Agent with LangGraph]]).
- **Swapping models is only as easy as the interface.** Models differ in tool support, output format and limits, so "change one line" is the promise, and I tested it only with a fake model.
- **Many integrations means many packages and many versions.** The docs I read show separate provider packages (for example `langchain-ollama`), and the library's API has changed a lot between versions (my old 0.0.275 against the newer docs). Pin versions, and check the docs for the one you installed.
- **A layer has a price.** For one prompt and one model call, a direct call to [[Ollama]] or [[Hugging Face Transformers]] is shorter. A framework pays off when you have many steps, many providers or many people on the same code (my opinion, not tested).
- **Tracing.** The README names LangSmith as its tool for evaluating, observing and debugging apps. I did not use it.
- **Limits of this note:** I did not install `langchain`, call a real model through it, run `create_agent`, middleware, retrievers, memory, tool calling or LangSmith.

**What came before, and how this improved on it:**
- **Before:** every program that used a language model glued together its own prompt strings, model calls, and clean-up code, and each provider had its own client library.
- **Its limit:** the glue was rewritten for every project, and switching provider meant changing code in many places.
- **What changed:** a shared interface for the pieces (a "runnable" with `invoke`, `batch` and `stream`), a pipe sign to join them, and a large set of ready-made integrations. The README says it helps you chain interoperable components and third-party integrations, and that it is MIT-licensed (page summary). I did not find a first-release date in the pages I read, and the commonly told story of its start is from my memory only, so it is left out.
- **How much better:** nothing measured against the old way. What I measured is small: a three-step chain ran with one `invoke` call, the same chain answered two inputs with one `batch` call, and a plain function became a step with no extra code.

## Related
[[LangGraph]] · [[Agent]] · [[Tool Calling]] · [[RAG]] · [[Prompt Engineering]] · [[LLM]] · [[Ollama]] · [[Hugging Face Transformers]] · [[Vector Database]] · [[AI Agent with LangGraph]] · [[Structured Output]]

## Sources
- Run on this PC: `langchain-core` 0.3.86 inside the virtual environment `D:\python456\agents-langgraph\.venv` (Python 3.9.6). The printed lines above are from my own run of `lc_demo.py`.
- The LangChain README (https://github.com/langchain-ai/langchain), read on 18 November 2026 through a page summary: MIT licence, chains interoperable components and integrations, LangGraph and LangSmith described as separate tools, no release date given.
- The LangChain overview page (https://docs.langchain.com/oss/python/langchain/overview), read the same day through a page summary: `create_agent`, agents "built on top of LangGraph", the standard model interface.
- No image: the only Commons candidate looked computer-made, so I did not use it.
- The note was not opened in Obsidian.

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/agents-rag LangChain:::A Python library of ready-made pieces for language-model programs (prompt templates, model wrappers, parsers, tools) that all share one interface and can be joined with a pipe sign.

#flashcards/ml/concept #flashcards/topic/agents-rag What does the pipe sign do in LangChain?::`a | b` joins two pieces into a chain that runs `a` and feeds its output to `b`. In my demo three pieces (prompt, model, parser) made one `RunnableSequence`.

#flashcards/ml/concept #flashcards/topic/agents-rag What is a "runnable" in LangChain?::Any piece with the same three methods: `invoke` (run once), `batch` (run on a list) and `stream` (give results as they arrive). Plain functions can be wrapped as runnables.

#flashcards/ml/concept #flashcards/topic/agents-rag What are the usual pieces of a simple LangChain chain?
?
- A prompt template (fills the blanks and makes the messages).
- A model piece (a standard wrapper around one language model).
- A parser (turns the reply into the form you want, such as plain text).

#flashcards/ml/concept #flashcards/topic/agents-rag Why can a LangChain chain not do loops, and what is used instead?::A chain is a straight line from the first piece to the last. Loops, branches, saved state and pauses need a graph tool, LangGraph, which the LangChain docs say its agents are built on.

#flashcards/ml/concept #flashcards/topic/agents-rag What did my `batch` call of two questions show with a fake model?::It returned two replies, but in swapped order. The fake model hands out its list in the order calls arrive, and `batch` runs the calls in parallel (my reading, not examined further).

#flashcards/ml/concept #flashcards/topic/agents-rag Which package did I use for the LangChain demo, and which did I not install?::`langchain-core` 0.3.86 (already installed because `langgraph` needs it). I did not install the `langchain` package itself, and an old `langchain` 0.0.275 sits in another Python on this PC.

#flashcards/ml/cloze #flashcards/topic/agents-rag A LangChain ==prompt template== is a sentence with blanks that is filled from a dictionary and turned into messages.

#flashcards/ml/concept #flashcards/topic/agents-rag What does the README say about LangChain, LangGraph and LangSmith? (page summary)::LangChain is MIT-licensed and chains interoperable components and integrations. LangGraph is a separate framework for controllable agent workflows. LangSmith is the tool for tracing, evaluating and debugging apps.

#flashcards/ml/concept #flashcards/topic/agents-rag When is a LangChain layer worth it?::When you have many steps, many providers or many people on the same code. For one prompt and one model call, a direct call is shorter (my opinion, not tested).

#flashcards/ml/concept #flashcards/topic/agents-rag What did I not test about LangChain?::Installing the `langchain` package, a real model through it, `create_agent`, middleware, retrievers, memory, tool calling and LangSmith. I used a fake model and `langchain-core` only.
