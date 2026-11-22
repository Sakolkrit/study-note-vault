---
tags: [project, agents, langgraph, multi-agent, ollama, python, LLM]
status: done
stack: Python 3.9, LangGraph 0.6.11, Ollama (qwen2.5:1.5b and 0.5b), CPU only
goal: build a small team of AI agents (a supervisor, a researcher, a calculator and a writer) as a LangGraph graph, test its logic without a model, then run it on local models and measure what a small model can and cannot do
---

# AI Agent with LangGraph: Multi-agent

> **In one line:** build a team of four small [[Agent]]s that pass one shared notebook around as a [[LangGraph]] graph, run them on a local model with no GPU, and find out how often a small model picks the right teammate.

## Goal
Done means five checks pass: (1) the graph's logic works with a scripted supervisor and no model (routes, merged state, loop guards), (2) the calculator refuses anything that is not arithmetic, (3) a real local model can drive the supervisor, the calculator's expression and the writer, (4) ten test questions are run on two model sizes and two supervisor prompts, and (5) the results are compared with a plain model that has no team at all.

## Result in short
- **The team:** a *supervisor* picks who works next. The *researcher* searches this vault's `20-ML-LLM` notes by word overlap. The *calculator* turns the question into an arithmetic expression and a safe evaluator works it out. The *writer* answers from the notes and numbers and ends the run. Workers report back to the supervisor, and the writer is the only exit.
- **The graph logic works without any model.** `test_graph.py` passes: the right trace for research, maths and both, notes and numbers merged by the state reducers, a stuck or nonsense supervisor sent to the writer, and the calculator refusing code. In this LangGraph version an endless loop is stopped at 25 steps ("Recursion limit of 25 reached").
- **A small model is a weak supervisor.** First attempt with qwen2.5:1.5b: on 3 demo questions the supervisor picked the writer straight away every time, and the answer to 17 × 23 + 5 was "460" (the right answer is 396). Ten test questions, same settings each time (temperature 0, one run each):

| Setup | Right workers | Right answers (my word check) |
|---|---|---|
| 1.5B, no team (writer only) | not applicable | 6 of 10 |
| 1.5B, supervisor prompt v1 | 5 of 10 | 8 of 10 |
| 1.5B, supervisor prompt v2 (with examples) | 7 of 10 | 6 of 10 |
| 0.5B, no team (writer only) | not applicable | 7 of 10 |
| 0.5B, supervisor prompt v1 | 2 of 10 | 7 of 10 |
| 0.5B, supervisor prompt v2 | 3 of 10 | 5 of 10 |

- **What the table hides.** On the 3 arithmetic questions the 1.5B team got all 3 right with either prompt, against 1 of 3 with no team: the calculator helps. On the 3 vault-knowledge questions the plain 1.5B model got 2 right, and the team with prompt v2 got 0 (my crude word check), because the word-overlap search handed the writer weak snippets. For the 0.5B model, prompt v2 made it copy the order of my examples (calculator, then researcher) on 8 of 10 questions.
- **The team did not beat the plain model overall.** The best team row (1.5B, prompt v1) scored 8 of 10 answers, but it called a researcher on none of the knowledge questions and so was mostly the plain model. Treat all of these as ten questions, one run, a crude check.

## What you need
Versions I used, **in a fresh virtual environment** at `D:\python456\agents-langgraph\.venv` (Windows 11, no GPU):

| Tool | Version |
|---|---|
| Python | 3.9.6 |
| langgraph | 0.6.11 (pip chose it for Python 3.9) |
| langgraph-checkpoint / -prebuilt / -sdk | 2.1.2 / 0.6.5 / 0.2.9 |
| langchain-core | 0.3.86 |
| ollama (Python client) | 0.6.3 |
| Ollama (the program) | 0.21.2, already installed |
| pydantic / langsmith | 2.13.5 / 0.4.37 |

Downloads I made, with permission: the packages above (`pip install langgraph ollama`, about 33 packages pulled in, several minutes with pip 21.1.3) and the model `qwen2.5:1.5b` (986,048,512 bytes in the registry listing, shown as 986 MB). `qwen2.5:0.5b` was already there from [[Local CPU Chatbot]]. I used a new environment because my main Python holds an old `langchain 0.0.275` that a LangGraph install could have upset.

## Steps
1. **Plan the team.** Four nodes and one shared state. The supervisor never writes or calculates; it only chooses. A worker never chooses; it does one job and returns to the supervisor. This is the "supervisor" pattern for [[LangGraph]] (the docs also describe other patterns such as a network of agents, which I did not build).

2. **Create the environment and install.**
   ```
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install langgraph ollama
   ollama pull qwen2.5:1.5b
   ```

3. **Write the graph.** Three ideas are in this file. (a) The **state** is a typed dictionary, and three of its keys are lists with a *reducer* (`operator.add`): when a node returns `{"notes": [...]}`, LangGraph appends to the list instead of replacing it, so the researcher's notes and the trace survive. (b) **Nodes** are plain functions that read the state and return a partial update. (c) **Edges**: fixed ones (workers back to the supervisor, writer to the end) and one *conditional* edge, where a function picks the next node from `state["next"]`. The supervisor refuses a worker that has already run, refuses a name it does not know, and forces the writer after `max_steps`. The model is passed in as three small functions, so the same graph runs with a scripted supervisor in the tests and a real model later.

   **`agents.py`**
   ```python
   """A small multi-agent graph: a supervisor sends the question to workers, a writer answers. Needs: langgraph (+ ollama for the real model)."""
   import ast, glob, operator, os, re
   from typing import Annotated, TypedDict

   from langgraph.graph import END, START, StateGraph

   WORKERS = ["researcher", "calculator", "writer"]
   VAULT = "D:/python456/study&note stuff/20-ML-LLM/"


   # ---------- the shared state: every node reads it and returns a partial update ----------
   class State(TypedDict):
       question: str
       notes: Annotated[list, operator.add]     # the researcher appends snippets
       numbers: Annotated[list, operator.add]   # the calculator appends results
       trace: Annotated[list, operator.add]     # who ran, in order
       answer: str
       next: str


   # ---------- tool 1: search the vault (plain word overlap, no embeddings) ----------
   STOP = set("a an the of to in is are what why how does do and or for on it its with as by be this that".split())
   words = lambda s: [w for w in re.findall(r"[a-z][a-z0-9.+-]+", s.lower()) if w not in STOP]
   _paragraphs = []
   FENCE = chr(96) * 3                          # three backticks, built here so this file can sit inside a note


   def search(question, k=2):
       if not _paragraphs:
           for f in sorted(glob.glob(VAULT + "*.md")):
               text = open(f, encoding="utf-8").read().split("\n## Fl" + "ashcards")[0]
               text = re.sub(FENCE + r".*?" + FENCE + r"|\$\$.*?\$\$", "", text, flags=re.S)
               for p in re.split(r"\n\s*\n", text):
                   if len(p) > 80:
                       _paragraphs.append((os.path.basename(f)[:-3], p.strip()))
       q = set(words(question))
       score = lambda name, p: len(q & set(words(p))) / (len(set(words(p))) ** 0.5 + 1) + 0.5 * len(q & set(words(name)))   # a hit in the note's title counts extra
       scored = sorted(((score(name, p), name, p) for name, p in _paragraphs), reverse=True)
       return [f"[{name}] {p[:300]}" for s, name, p in scored[:k] if s > 0]


   # ---------- tool 2: a safe calculator (only numbers and + - * / ** % and brackets) ----------
   OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
          ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg}


   def calc(expr):
       def ev(n):
           if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
           if isinstance(n, ast.BinOp) and type(n.op) in OPS: return OPS[type(n.op)](ev(n.left), ev(n.right))
           if isinstance(n, ast.UnaryOp) and type(n.op) in OPS: return OPS[type(n.op)](ev(n.operand))
           raise ValueError("not allowed: " + ast.dump(n)[:40])
       return ev(ast.parse(expr.strip().replace(",", ""), mode="eval").body)


   # ---------- the graph ----------
   def build(route, expression, write, max_steps=5):
       """route(state, allowed) -> a worker name; expression(question) -> text like '17*23+5'; write(state) -> answer text."""

       def supervisor(state):
           done = [t for t in state["trace"] if t in WORKERS]
           allowed = [w for w in WORKERS if w not in done or w == "writer"]          # no worker runs twice
           choice = route(state, allowed)
           if choice not in allowed or len(state["trace"]) >= max_steps:             # a bad answer or too many steps: go and write
               choice = "writer"
           return {"next": choice}

       def researcher(state):
           return {"notes": search(state["question"]), "trace": ["researcher"]}

       def calculator(state):
           expr = expression(state["question"])
           try:
               return {"numbers": [f"{expr.strip()} = {calc(expr):g}"], "trace": ["calculator"]}
           except Exception as e:                                                    # a bad expression is data, not a crash
               return {"numbers": [f"could not calculate '{expr.strip()}' ({type(e).__name__})"], "trace": ["calculator"]}

       def writer(state):
           return {"answer": write(state), "trace": ["writer"]}

       g = StateGraph(State)
       for name, fn in (("supervisor", supervisor), ("researcher", researcher), ("calculator", calculator), ("writer", writer)):
           g.add_node(name, fn)
       g.add_edge(START, "supervisor")
       g.add_conditional_edges("supervisor", lambda s: s["next"], {w: w for w in WORKERS})
       g.add_edge("researcher", "supervisor")      # workers report back to the supervisor
       g.add_edge("calculator", "supervisor")
       g.add_edge("writer", END)                   # the writer ends the run
       return g.compile()


   def start(question):
       return {"question": question, "notes": [], "numbers": [], "trace": [], "answer": "", "next": ""}
   ```

4. **Test the logic with a scripted supervisor, no model.** `test_graph.py` feeds the graph a made-up list of choices and checks the trace and the state, including a supervisor that says "researcher" fifty times, one that says "banana", the step cap, the calculator's safety, and the search.

   **`test_graph.py`**
   ```python
   """Tests for the graph with a scripted 'model', so no LLM is needed. Run: python test_graph.py"""
   import operator
   from typing import Annotated, TypedDict
   from langgraph.errors import GraphRecursionError
   from langgraph.graph import END, START, StateGraph
   from agents import build, calc, search, start

   write = lambda s: f"ANSWER notes={len(s['notes'])} numbers={s['numbers']}"


   def run(plan, expr="2+3", **kw):
       """plan = the worker names the scripted supervisor will pick, in order."""
       it = iter(plan)
       g = build(lambda s, allowed: next(it, "writer"), lambda q: expr, write, **kw)
       return g.invoke(start("What is a KV cache? What is 2+3?"))


   r = run(["researcher", "writer"])
   assert r["trace"] == ["researcher", "writer"] and len(r["notes"]) == 2 and r["numbers"] == [] and r["answer"].startswith("ANSWER notes=2")
   r = run(["calculator", "writer"])
   assert r["trace"] == ["calculator", "writer"] and r["numbers"] == ["2+3 = 5"] and r["notes"] == []
   r = run(["researcher", "calculator", "writer"])
   assert r["trace"] == ["researcher", "calculator", "writer"] and len(r["notes"]) == 2 and r["numbers"] == ["2+3 = 5"]
   print("routes ok; state adds up (notes, numbers and trace are merged by the reducers)")

   r = run(["researcher"] * 50)                       # a supervisor stuck on one choice: the second pick is refused, so it writes
   assert r["trace"] == ["researcher", "writer"]
   r = run(["banana"])                                # nonsense: refused, so it writes
   assert r["trace"] == ["writer"]
   r = run(["researcher", "calculator", "writer"], max_steps=1)   # the step cap forces the writer early
   assert r["trace"] == ["researcher", "writer"]
   print("guards ok: repeats, nonsense and the step cap all end in the writer")

   assert calc("17*23+5") == 396 and calc("2**10") == 1024 and calc("(144/12)**2") == 144 and calc("1,000+1") == 1001
   for bad in ("__import__('os').getcwd()", "open('x')", "2+", "abs(-1)"):
       try:
           calc(bad); raise SystemExit("should have failed: " + bad)
       except SystemExit: raise
       except Exception: pass
   r = run(["calculator", "writer"], expr="__import__('os').getcwd()")
   assert "could not calculate" in r["numbers"][0] and r["trace"] == ["calculator", "writer"]
   print("calculator ok: arithmetic works; code is refused, and a bad expression is recorded without crashing the run")

   hits = search("What does the KV cache store?")
   assert hits and hits[0].startswith("[KV Cache]"), hits
   print("search ok, top hit:", hits[0][:70])

   print(build(lambda s, a: "writer", lambda q: "1", write).get_graph().draw_mermaid())


   class Loop(TypedDict):
       n: Annotated[int, operator.add]


   g = StateGraph(Loop); g.add_node("a", lambda s: {"n": 1}); g.add_edge(START, "a"); g.add_edge("a", "a")
   try:
       g.compile().invoke({"n": 0})
   except GraphRecursionError as e:
       print("an endless loop is stopped by LangGraph itself:", str(e)[:60])
   ```
   Output (the library's deprecation warnings filtered out and not read):
   ```
   routes ok; state adds up (notes, numbers and trace are merged by the reducers)
   guards ok: repeats, nonsense and the step cap all end in the writer
   calculator ok: arithmetic works; code is refused, and a bad expression is recorded without crashing the run
   search ok, top hit: [KV Cache] **Example:** You are adding up a long shopping receipt, and
   (the drawing of the graph is shown below)


   an endless loop is stopped by LangGraph itself: Recursion limit of 25 reached without hitting a stop conditi
   ```
   What to look for: all asserts pass, the search's top hit for "What does the KV cache store?" is the `KV Cache` note, and the drawing of the graph (it renders as a picture in Obsidian) shows `supervisor` fanning out to the three workers, the two workers going back, and `writer` going to the end.

   ```mermaid
   ---
   config:
     flowchart:
       curve: linear
   ---
   graph TD;
       __start__([<p>__start__</p>]):::first
       supervisor(supervisor)
       researcher(researcher)
       calculator(calculator)
       writer(writer)
       __end__([<p>__end__</p>]):::last
       __start__ --> supervisor;
       calculator --> supervisor;
       researcher --> supervisor;
       supervisor -.-> calculator;
       supervisor -.-> researcher;
       supervisor -.-> writer;
       writer --> __end__;
       classDef default fill:#f2f0ff,line-height:1.2
       classDef first fill-opacity:0
       classDef last fill:#bfb6fc
   ```

5. **Connect a real model.** Three small jobs go to the model, each with Ollama's structured output (`format` with a JSON schema), so the answer is guaranteed to be valid JSON: (a) the supervisor's choice, with the schema's `enum` limited to the workers still allowed, (b) the calculator's expression, and (c) the writer's short answer in plain text. Temperature is 0 and the seed is fixed. `run.py` has a demo mode, an eval mode with ten questions, and a prompt switch (`v1`, `v2`, or `none` for no team).

   **`run.py`**
   ```python
   """Run the multi-agent graph on a local Ollama model.  python run.py MODEL demo   or   python run.py MODEL eval"""
   import json, sys, time
   import ollama
   from agents import build, start

   MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:1.5b"


   def chat(system, user, schema=None, n=200):
       r = ollama.chat(model=MODEL, format=schema, options={"temperature": 0, "seed": 1, "num_predict": n},
                       messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
       return r["message"]["content"].strip()


   PROMPT = sys.argv[3] if len(sys.argv) > 3 else "v2"
   ROUTE = {"v1": """You are a supervisor. Pick the next worker for the user's question.
   researcher: looks up facts in a notebook of AI notes.
   calculator: does arithmetic.
   writer: writes the final answer. Pick writer when you have what you need, or when the question needs no notes and no arithmetic.""",
            "v2": """You are the supervisor of a small team. Decide who works next. Reply with one worker.
   - researcher: use when the question asks what something is, how it works or why (facts about AI, machine learning, GGUF, KV cache and similar).
   - calculator: use when the question has numbers to work out (adding, multiplying, percent, powers).
   - writer: use ONLY when the question needs no lookup and no calculation (greetings, chat), or when everything it needs is already done.

   Examples (the "done so far" list shows who has already worked):
   Question: What is attention? | done so far: nothing -> researcher
   Question: What is 8 * 9? | done so far: nothing -> calculator
   Question: What is 2 + 2, and what is a token? | done so far: nothing -> researcher
   Question: What is 2 + 2, and what is a token? | done so far: researcher -> calculator
   Question: What is 2 + 2, and what is a token? | done so far: researcher, calculator -> writer
   Question: Hi there! | done so far: nothing -> writer""",
            "none": ""}[PROMPT]          # "none": no supervisor, the writer answers every question alone (the baseline)


   def route(state, allowed):
       if PROMPT == "none":
           return "writer"
       done = [t for t in state["trace"] if t in ("researcher", "calculator")]
       user = f"Question: {state['question']} | done so far: {', '.join(done) or 'nothing'}\nChoose one of: {allowed}"
       schema = {"type": "object", "properties": {"next": {"type": "string", "enum": allowed}}, "required": ["next"]}
       return json.loads(chat(ROUTE, user, schema, 30))["next"]


   def expression(question):
       schema = {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
       system = "Turn the arithmetic in the question into ONE Python expression using only numbers and + - * / ** and brackets. No words."
       return json.loads(chat(system, question, schema, 40))["expression"]


   def write(state):
       system = "Answer the question in one to three short sentences. Use only the notes and results given. If a calculation result is given, quote it exactly."
       user = f"Question: {state['question']}\nNotes: {state['notes']}\nCalculation results: {state['numbers']}"
       return chat(system, user, None, 120)


   graph = build(route, expression, write)


   def ask(question, show=False):
       t0, state = time.time(), start(question)
       for update in graph.stream(state, stream_mode="updates"):          # one update per node that ran
           for node, change in update.items():
               if show: print(f"  {node:10s} -> {str(change)[:110]}")
               state.update({k: (state[k] + v if isinstance(v, list) else v) for k, v in change.items()})
       return state, time.time() - t0


   CASES = [  # question, the workers it needs (besides the writer), words that must appear in a right answer (any one)
       ("What does the KV cache store?", {"researcher"}, ["key", "value"]),
       ("What is perplexity?", {"researcher"}, ["confus", "surpris", "choices", "unsure"]),
       ("Why does a GGUF file align its tensor data?", {"researcher"}, ["mmap", "map", "memory"]),
       ("What is 17 * 23 + 5?", {"calculator"}, ["396"]),
       ("Compute (144 / 12) ** 2.", {"calculator"}, ["144"]),
       ("What is 15% of 240?", {"calculator"}, ["36"]),
       ("A GGUF block holds 32 weights in 34 bytes. How many bytes do 4,254,208 blocks take, and what is GGUF?", {"researcher", "calculator"}, ["144643072"]),
       ("What is 12 * 12, and what does the KV cache store?", {"researcher", "calculator"}, ["144"]),
       ("Hello! Say hi in five words.", set(), []),
       ("Who are you?", set(), []),
   ]

   if __name__ == "__main__":
       mode = sys.argv[2] if len(sys.argv) > 2 else "demo"
       print("model:", MODEL)
       if mode == "demo":
           for q, _, _ in CASES[:1] + CASES[3:4] + CASES[7:8]:
               print("\nQ:", q)
               s, t = ask(q, show=True)
               print(f"A ({t:.1f}s): {s['answer']}")
       else:
           route_ok = ans_ok = 0
           for q, need, words in CASES:
               s, t = ask(q)
               used = set(s["trace"]) - {"writer"}
               ok_r = used == need
               text = s["answer"].lower().replace(",", "")
               ok_a = not words or any(w in text for w in words)
               route_ok += ok_r; ans_ok += ok_a
               print(f"{'ok ' if ok_r else 'BAD'} route {sorted(used)!s:28s} {'ok ' if ok_a else 'BAD'} answer  {t:5.1f}s  {q[:46]}")
           print(f"{MODEL}: right workers {route_ok}/{len(CASES)}, right answers {ans_ok}/{len(CASES)}")
   ```

6. **First real run (demo).** `python run.py qwen2.5:1.5b demo`, with a supervisor message that only said how many notes were found. The supervisor picked the writer first on all three questions:
   ```
   Q: What is 17 * 23 + 5?
     supervisor -> {'next': 'writer'}
     writer     -> {'answer': 'The answer is 460.', 'trace': ['writer']}
   ```
   17 × 23 + 5 is 396 (my arithmetic, checked in the test), so the "team" answered alone and wrongly. I changed the supervisor's message to list who has already worked, which is the `v1` in the code.

   Rerun later with the final `v2` prompt, using the code copied back out of this note (`python run.py qwen2.5:1.5b demo v2`): the same question now went to the calculator and the answer was right, "The answer to the expression 17 * 23 + 5 is 396." The mixed question went wrong in a new way:
   ```
   Q: What is 12 * 12, and what does the KV cache store?
     supervisor -> {'next': 'calculator'}
     calculator -> {'numbers': ['(12 * 12) = 144'], 'trace': ['calculator']}
     supervisor -> {'next': 'writer'}
     writer     -> {'answer': 'The product of 12 multiplied by 12 is 144. The KV cache stores this result.', ...}
   ```
   The supervisor skipped the researcher, and the writer, with no notes, made up "The KV cache stores this result". My word check passes this answer (it only looks for "144"), which shows how weak the check is. These three questions took 32 to 40 seconds each.

7. **Ten questions, two models, three setups.** `python run.py MODEL eval v1` (or `v2` or `none`). Each line says whether the set of workers used (besides the writer) was the right one, and whether the answer contains a word that a right answer must contain (for example "396"). Output for the 1.5B model with prompt `v2`:
   ```
   ok  route ['researcher']               BAD answer   30.0s  What does the KV cache store?
   BAD route ['calculator', 'researcher'] BAD answer   38.1s  What is perplexity?
   ok  route ['researcher']               BAD answer   27.7s  Why does a GGUF file align its tensor data?
   ok  route ['calculator']               ok  answer   45.5s  What is 17 * 23 + 5?
   ok  route ['calculator']               ok  answer   63.0s  Compute (144 / 12) ** 2.
   ok  route ['calculator']               ok  answer   45.5s  What is 15% of 240?
   ok  route ['calculator', 'researcher'] BAD answer   65.1s  A GGUF block holds 32 weights in 34 bytes. How
   BAD route ['calculator']               ok  answer   44.4s  What is 12 * 12, and what does the KV cache st
   ok  route []                           ok  answer   18.6s  Hello! Say hi in five words.
   BAD route ['researcher']               ok  answer   33.7s  Who are you?
   qwen2.5:1.5b: right workers 7/10, right answers 6/10
   ```
   The other five runs are summarised in the table above. (My word check is crude: it said "BAD" to a KV cache answer that talked about a notebook where the AI keeps what it already worked out, which is roughly right but never says "key" or "value".)

8. **Look at why the knowledge answers got worse.** With prompt `v2` the researcher ran for the KV cache question and found two paragraphs of the `KV Cache` note, but the first was the shopping-receipt example, so the writer repeated the receipt idea ("the running total of items added to a shopping receipt"). For "What is perplexity?" it found a paragraph about the training loss and the note's frontmatter lines. For "Who are you?" it was sent to the researcher and got unrelated paragraphs from the `Precision` and `Benchmarks` notes. Word overlap is a weak search ([[Vault Search with RAG]] used embeddings for this reason).

## What the numbers say
- **Tools fixed what the model cannot do.** Multiplying 17 × 23 and adding 5 is something a small model gets wrong without help (the 1.5B model said 460), and a ten-line safe calculator gets it right. The team was right on 3 of 3 arithmetic questions with the 1.5B model, with either prompt.
- **Routing is the weak point.** The 1.5B model sent the writer first on all three demo questions, picked the right workers on 5 of 10 with prompt v1 and 7 of 10 with v2. The 0.5B model picked the right workers on 2 and 3 of 10. A supervisor that always says "writer" is the same as having no team.
- **Examples in the prompt can backfire on a small model.** Prompt v2 helped the 1.5B model's routing (5 to 7 of 10) but the 0.5B model copied my example order on 8 of 10 questions.
- **More steps cost time.** The 1.5B team took about 10 to 25 seconds a question with prompt v1 and 18 to 65 seconds with prompt v2, against 3 to 15 seconds with no team. I did not check why the v2 runs were this much slower (longer prompts and notes are one reason, machine load may be another).
- **My check is a word check.** I did not read all 60 answers. A right word in the answer is not proof of a right answer.

## What went wrong, and the fix
- **The supervisor picked "writer" first (qwen2.5:1.5b, the first demo).** Fix: list who has already worked in its message, add examples (v2), and build in guards so a bad choice cannot loop or skip the writer. The model still routes wrongly on a third of the questions.
- **A search test failed.** `search("What does the KV cache store?")` returned the `PagedAttention` note first. Fix: a hit in the note's title counts extra. The first hit is now the `KV Cache` note, but its first paragraph is only an example.
- **The docs page and my installed version disagree.** I read the LangGraph docs for the current 1.x release (default recursion limit 10,007 from 1.1.4, 10,000 from 1.0.6, from a page summary). pip installed 0.6.11 for Python 3.9, where my endless-loop test stopped at **25**. Not every API in the 1.x docs may exist in 0.6.11. I only used `StateGraph`, `add_node`, `add_edge`, `add_conditional_edges`, `compile`, `invoke`, `stream` and `get_graph`.
- **The docs links I tried first returned "Redirecting..."** with no content. The pages at the newer docs address worked.
- **Slow install.** `pip` in the new environment is version 21.1.3 and took several minutes to resolve about 33 packages. It worked.

## What I did not run
- **Tool calling as the model sees it.** I used JSON output instead of the model's own tool-calling interface ([[Tool Calling]]), because I assumed a small model is shakier at the latter (my guess, not tested). I did not compare the two.
- **Memory, checkpoints, human approval, `Command` and `Send` handoffs, parallel workers, token streaming and tracing.** None of them is used. The docs describe them, and I did not test any.
- **The prebuilt supervisor package, a network of agents, or a hierarchy of teams.**
- **Bigger models, other prompts, repeated runs.** Each setup was one run with temperature 0 and one seed, so I do not know how much a rerun would change. Ten questions is too few for percentages.
- **Real safety review.** The calculator is only arithmetic by design (`ast`, no `eval`), and I tested four bad inputs. I did not test anything else, such as hostile text inside the vault notes reaching the writer ([[Agent Safety]]).
- **A user interface**, Python 3.10 and LangGraph 1.x, and whether the drawing renders in Obsidian (it is text; I have not opened Obsidian).

## Concepts used
- [[Agent]], [[Tool Calling]], [[ReAct]] (the loop of reasoning and acting that a supervisor team splits into roles), [[Agent2Agent]], [[Model Context Protocol]], [[Agent Safety]]
- [[LangGraph]], [[LangChain]], [[Multi-Agent Systems]], [[Ollama]], [[LLM]], [[Prompt Engineering]], [[Structured Output]] (the `format=` schema), [[Decoding Strategies]] (temperature 0)
- [[RAG]] and [[Vault Search with RAG]] (the researcher is a very small version), [[Local CPU Chatbot]]

## Next ideas
- Replace the word-overlap search with the embeddings from [[Vault Search with RAG]] and rerun the three knowledge questions.
- Try a 3B model, and the model's own tool-calling, and see whether routing gets better.
- Let the supervisor see the notes before it decides, or add a "checker" agent that reads the answer against the notes.
- Add a checkpointer so a conversation keeps its state between questions, and a human-approval step before the calculator runs.

## Flashcards
#flashcards/projects

#flashcards/projects/concept #flashcards/topic/project-agents In a LangGraph supervisor team, what does each part do?::The supervisor only chooses who works next. A worker does one job and returns to the supervisor. In this project the researcher searches notes, the calculator works out an expression, and the writer answers and ends the run.

#flashcards/projects/concept #flashcards/topic/project-agents What is a reducer in a LangGraph state, and why did the project use `operator.add`?::A reducer says how a node's update merges into a state key. With `operator.add` on a list, a node returning a list appends to it instead of replacing it, so the notes, numbers and trace from different nodes all survive.

#flashcards/projects/concept #flashcards/topic/project-agents What is a conditional edge?::An edge where a routing function reads the state and picks the next node. Here it reads `state["next"]` after the supervisor and goes to the researcher, calculator or writer. Static edges always go to the same node.

#flashcards/projects/concept #flashcards/topic/project-agents Which guards stop the supervisor from looping or skipping the writer?::A worker that has already run is refused, an unknown name is refused, and after `max_steps` the writer is forced. Any refusal sends the run to the writer. LangGraph also has its own recursion limit, 25 in the version used here.

#flashcards/projects/example #flashcards/topic/project-agents What did an endless loop do in LangGraph 0.6.11, and what do the 1.x docs say?::It stopped with "Recursion limit of 25 reached" in the installed 0.6.11. The 1.x docs summary says the default is 10,007 from 1.1.4. The two differ because the versions differ.

#flashcards/projects/example #flashcards/topic/project-agents The 1.5B model was asked 17 * 23 + 5 without the team. What did it say, and what is right?::It said 460. The right answer is 396 (17 × 23 = 391, plus 5), which the calculator tool gets right.

#flashcards/projects/example #flashcards/topic/project-agents How many of 10 questions did each team setup route to the right workers?::1.5B: 5 of 10 with prompt v1 and 7 of 10 with v2. 0.5B: 2 of 10 with v1 and 3 of 10 with v2. One run each at temperature 0, with a crude word check for the answers.

#flashcards/projects/concept #flashcards/topic/project-agents Did the team beat a plain model with no supervisor?::Not overall. The calculator helped on arithmetic (3 of 3 right with the 1.5B team against 1 of 3 without). On the vault-knowledge questions the plain 1.5B model got 2 of 3 and the team with prompt v2 got 0 of 3, because the word-overlap search returned weak snippets.

#flashcards/projects/example #flashcards/topic/project-agents What went wrong when the 0.5B model got the prompt with examples?::It copied the order of the examples (calculator, then researcher) on 8 of 10 questions instead of choosing for each question.

#flashcards/projects/concept #flashcards/topic/project-agents Why did the researcher make knowledge answers worse?::The word-overlap search returned weak paragraphs, for example the shopping-receipt example from the `KV Cache` note, and the writer repeated them. A better search such as embeddings would be the next fix.

#flashcards/projects/concept #flashcards/topic/project-agents How does the calculator tool stay safe?::It parses the expression with `ast` and allows only numbers and + - * / ** % and brackets, with no `eval`. Code such as `__import__('os')` raises an error, and the error is recorded as data instead of crashing the run.

#flashcards/projects/concept #flashcards/topic/project-agents Why was the model's output made JSON with a schema?::Ollama's `format` option with a JSON schema forces valid JSON, and the supervisor's schema lists only the workers still allowed, so a small model cannot name a wrong worker (though it can still choose a bad one).

#flashcards/projects/concept #flashcards/topic/project-agents Name three things the project did not test.::The model's own tool-calling interface, memory and checkpoints with human approval, and repeated runs for variance. It also did not test other agent patterns, bigger models or LangGraph 1.x.
