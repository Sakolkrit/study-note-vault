---
tags: [agents, protocol, A2A, multi-agent, LLM, standards]
status: growing
level: 4
tier: 3
---

# Agent2Agent

Written **A2A**. A shared standard that lets separate AI agents **find each other and hand each other work**.

## 🟢 1. Simple (age 12)
In a smart house there might be a thermostat helper, an energy-saving helper and an air-quality helper, each built by a different company. For them to cooperate, they need a **common language** and a way to introduce themselves. A2A gives each helper a **name card** listing what it can do. One helper reads another's card, sees "I can warm up the house", and sends it a job. They never need to know how the other works inside.

**Example:** You say "warm up the house before I'm home". Your main assistant checks the cards, finds the thermostat helper's card says "prepares the home temperature by arrival", and sends it the job.

## 🔵 2. How it works
1. A single [[Agent]] can call tools ([[Tool Calling]]), perhaps through [[Model Context Protocol]]. But a system may have many agents, each specialised and each run by someone else.
2. Each agent publishes an **Agent Card**: a small document that says who the agent is (name, version, address) and which **skills** it has. Each skill has an id, a description and example requests.
3. A **client agent** with a job reads the cards and picks the agent whose skills fit.
4. The client sends the job as a **message** to the chosen agent's address. The remote agent turns it into a **task**.
5. The task moves through stages while the remote agent works, and it can return a result, or ask for more information.
6. The two agents only communicate through the standard messages, so each keeps its inner workings private.
7. In the lecture's picture: a thermostat agent with skills such as maintaining comfort and preparing the home, alongside agents for occupancy, air quality and energy management.
8. Compare: **MCP** connects a model to **tools and data**, while A2A connects **agents to agents**.

## 🟠 3. Technical detail
- **Origin:** Google announced the Agent2Agent Protocol in 2025 (named in the lecture). I read the table of contents and main definitions of the specification page, not the whole thing.
- **Defined terms (spec, read):** an *Agent Card* is "a JSON metadata document published by an A2A Server, describing its identity, capabilities, skills, service endpoint, and authentication requirements". A *Message* is one communication turn with a role ("user" or "agent") made of one or more parts. A *Task* is the basic unit of work, identified by a unique ID, and it is stateful and progresses over time.
- **The spec's contents (read):** the contents list includes operations such as sending messages (also streaming), getting, listing and cancelling tasks, push notifications, an agent-card signing section and security schemes, and a JSON-RPC protocol binding (the same message style as MCP).
- **Lecture's example card:** fields `name`, `url`, `version` and a list of `skills`, where each skill has `id`, `description` and `examples`, plus an executor class that runs a task. The code uses those field names.
- **Matching in the code:** a crude word-overlap rule picks an agent. Real clients would use a language model or search over the skill descriptions (my assumption, not stated in the spec pages I read).
- **Task states in the code** (submitted, working, completed) are my own simple set. The spec has its own set of states, which I did not check.
- **Safety:** authentication requirements are part of the card (spec definition). The lecture lists safety as an important topic for agents and tools in general ([[Tool Calling]]).
- **Not needed for one agent:** if there is a single agent with tools, MCP and plain tool calling are enough.

```python
import json
# each agent publishes a card: who it is, where it lives, and its skills (field names follow the lecture's example)
cards = [
    {'name': 'thermostat agent', 'url': 'http://path.to.agent/thermostat', 'version': '1.0.0',
     'skills': [{'id': 'maintain_comfort', 'description': 'Maintain a comfortable temperature', 'examples': ['Set to focus mode while I am working']},
                {'id': 'prepare_home', 'description': 'Sets temperature to target by arrival', 'examples': ['Warm up the house to 70F when I am back']}]},
    {'name': 'energy agent', 'url': 'http://path.to.agent/energy', 'version': '1.0.0',
     'skills': [{'id': 'optimize_energy', 'description': 'Reduces energy consumption', 'examples': ['Save energy overnight']}]},
    {'name': 'air quality agent', 'url': 'http://path.to.agent/air', 'version': '0.9.0',
     'skills': [{'id': 'check_air', 'description': 'Reports air quality and opens windows', 'examples': ['Is the air in the bedroom ok?']}]},
]
def words(s): return set(s.lower().replace('?', '').split())
def pick(request):
    """the calling agent reads the cards and picks the agent whose skill text shares the most words with the request (at least 2, so one stray word like 'me' does not count)"""
    best = (0, None, None)
    for c in cards:
        for s in c['skills']:
            overlap = len(words(request) & words(s['description'] + ' ' + ' '.join(s['examples'])))
            if overlap >= 2 and overlap > best[0]: best = (overlap, c, s)
    return best
for req in ('Warm up the house before I am back', 'Save energy tonight', 'Is the air ok in here?', 'Book me a flight'):
    ov, card, skill = pick(req)
    print(f'{req!r:40s} -> ' + (f"{card['name']} / {skill['id']} (overlap {ov})" if card else 'no agent has a matching skill'))
# a task moves through states while the remote agent works (state names here are my own simple set)
task = {'id': 't1', 'state': 'submitted'}
for state in ('working', 'completed'):
    task['state'] = state; print('task', task['id'], '->', task['state'])
print('card as JSON:', json.dumps({k: cards[1][k] for k in ('name', 'url', 'version')}))
```

## 🔴 4. Deep / formal
This note has no equation. The lecture and the spec pages I read describe a *protocol* (message shapes and roles), not a formula, and I did not derive any. The points below are the design edges.

**Design edges and trade-offs:**
- **Discovery:** the cards are the search space. If many agents publish similar skills, picking one is itself a hard decision. The code's matching refuses weak matches (fewer than 2 shared words), so "Book me a flight" gets no agent rather than a wrong one, which is a safer failure than a guess.
- **Trust:** a card is a claim by the agent, not a proof. Cards can be signed (the spec has a section on signing), but whether to trust an agent with a task, and what it may see, is a decision the client must make.
- **State:** tasks are long-lived and stateful, so the two sides must agree on what happened if a connection drops. Details such as retries and push notifications are in the spec; I did not study them.
- **Scale:** the number of messages is not a formula I derived. For the cost of too many tools or agents in one context, see the lecture's remarks in [[Tool Calling]].

**What came before, and how this improved on it:**
- **Before:** each multi-agent system wired its agents together with its own private messages, and individual agents with tools via [[Model Context Protocol]] or custom calls.
- **Its limit:** agents from different vendors could not easily discover or call one another. This is my summary of the motivation, taken from the lecture's framing of "how can agents communicate with each other?".
- **What changed:** a common card for discovery and a common message and task model on top of JSON-RPC (Google, 2025; spec pages read for the definitions above).
- **How much better:** I have no measured figure. The code only shows the mechanism: 3 of the 4 requests found a matching agent through the cards, and the fourth correctly found none. I did not measure effort saved or reliability.

## Related
- [[Agent]] · [[Model Context Protocol]] · [[Tool Calling]] · [[ReAct]] · [[LLM]] · [[Prompt Engineering]] · [[RAG]] · [[Agent Safety]] · [[Multi-Agent Systems]]

## Sources
- Stanford CME 295 (Fall 2025), Lecture 7 slides, Afshine and Shervine Amidi
- Agent2Agent (A2A) Protocol specification (a2a-protocol.org, contents and definitions read)
- Google, *Announcing the Agent2Agent Protocol (A2A)*, 2025 (named in the lecture; not read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/agents-rag Agent2Agent (A2A):::A protocol that lets agents from different makers find and call each other. Each agent publishes a name card listing what it can do.

#flashcards/ml/concept #flashcards/topic/agents-rag What is an Agent Card?::A JSON metadata document published by an A2A server describing its identity, capabilities, skills, service endpoint and authentication requirements. Each skill has an id, a description and example requests.

#flashcards/ml/example #flashcards/topic/agents-rag How does a client agent use Agent2Agent step by step?::1. Read the agent cards. 2. Pick the agent whose skills fit the job. 3. Send the job as a message to that agent's address. 4. The remote agent turns it into a task. 5. The task moves through stages and can return a result or ask for more information. The two agents only communicate through the standard messages, so each keeps its inner workings private.

#flashcards/ml/example #flashcards/topic/agents-rag What is the difference between MCP and A2A?::MCP connects a model to tools and data. A2A connects agents to agents.

#flashcards/ml/concept #flashcards/topic/agents-rag Why is a card not proof that an agent is trustworthy?::A card is a claim by the agent. Cards can be signed (the spec has a section on signing), but whether to trust an agent with a task, and what it may see, is a decision the client must make.

#flashcards/ml/concept #flashcards/topic/agents-rag When are MCP and plain tool calling enough?::When there is a single agent with tools. A2A is only needed when many agents, each run by someone else, must cooperate.

#flashcards/ml/cloze #flashcards/topic/agents-rag A2A tasks are long-lived and ==stateful==, so both sides must agree on what happened if a ==connection== drops.
