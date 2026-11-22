---
tags: [LLM, agents, MCP, protocol, tool-use, standards]
status: growing
level: 4
tier: 3
---

# Model Context Protocol

Usually written **MCP**. A shared standard for connecting tools and data to AI models, so each tool is built once and works with many apps.

## 🟢 1. Simple (age 12)
Think of the USB port. Before it existed, every gadget came with its own odd plug. Now one standard plug fits everything. MCP is a **standard plug for AI**: a tool that speaks MCP can be used by any AI app that speaks MCP, with no custom wiring for each pair.

**Example:** A "book finder" tool is built once. Your chat app, a coding assistant and a notes app can all use it, because they all know how to ask an MCP tool "what can you do?" and "please do this".

## 🔵 2. How it works
1. Without a standard, each AI app has to write its own connection to each tool, again and again, for every model and tool pair ([[Tool Calling]]).
2. MCP defines one common way to talk. A **server** offers capabilities; a **client** inside an AI application talks to it; the **host** is the app the user sees (the lecture's example host is Claude Desktop).
3. A server can offer three kinds of things (lecture): **tools** (actions the model can ask for), **resources** (data it can read) and **prompts** (ready-made instructions).
4. The client first asks the server **which tools it has**. The server answers with each tool's name, a description and the arguments it takes.
5. When the model wants something, the client sends a **call request** with the tool name and arguments, and the server sends back the result.
6. Errors have a standard shape too, so the app can tell the model "that tool does not exist" or "an argument is missing".
7. Result: **fewer connections to build**. With $M$ apps and $N$ tools, custom wiring needs $M\times N$ connections, while a standard needs about $M+N$.

## 🟠 3. Technical detail
- **Announced:** by Anthropic in 2024, per the lecture slide ("Introducing the Model Context Protocol"). I did not check the exact date.
- **Message format:** MCP uses JSON-RPC 2.0, a plain format with the fields `jsonrpc`, `id`, `method` and `params` (the MCP specification's overview page, which I read, lists "Core JSON-RPC message types" as the base protocol).
- **Listing tools:** a client sends the method `tools/list`, and the server returns tools with a `name`, a `description` and an `inputSchema` (the arguments, with their types and which are required). The specification's example is a `get_weather` tool that takes a `location` string (spec page, read).
- **Calling a tool:** the client sends `tools/call` with the tool's `name` and `arguments`; the server returns content (for example, text) (spec page, read).
- **Other parts of the spec:** lifecycle (connection set-up and negotiating capabilities), authorisation for network transports, and server features (resources, prompts, tools). I only read the overview and tools pages, not the rest.
- **What the toy server does:** the code answers `tools/list` and `tools/call` with the same message shape, and returns an error for a missing argument, an unknown tool and an unknown method. The error code numbers are the standard JSON-RPC ones, as I remember them, not checked in the MCP spec.
- **Not the same as the agent-to-agent standard:** the lecture contrasts MCP (model to tools) with Agent2Agent (agent to agent).
- **Limits (lecture, for tools in general):** many tools make the model worse and fill the context, tools must be defined well, and safety matters. A standard does not fix any of these, and a tool from an unknown server should be treated with care.

```python
import json
# a tiny server that speaks the same shape of messages as MCP's tool methods (JSON-RPC 2.0 requests: jsonrpc, id, method, params)
TOOLS = [{'name': 'get_weather', 'description': 'Get current weather information for a location',
          'inputSchema': {'type': 'object', 'properties': {'location': {'type': 'string', 'description': 'City name or zip code'}}, 'required': ['location']}}]
def weather(location): return f'Sunny in {location}, 21C (made-up data)'
def handle(req):
    rid, method, params = req.get('id'), req.get('method'), req.get('params', {})
    if method == 'tools/list': return {'jsonrpc': '2.0', 'id': rid, 'result': {'tools': TOOLS}}
    if method == 'tools/call':
        tool = next((t for t in TOOLS if t['name'] == params.get('name')), None)
        if tool is None: return {'jsonrpc': '2.0', 'id': rid, 'error': {'code': -32602, 'message': 'Unknown tool: ' + str(params.get('name'))}}
        missing = [k for k in tool['inputSchema']['required'] if k not in params.get('arguments', {})]
        if missing: return {'jsonrpc': '2.0', 'id': rid, 'error': {'code': -32602, 'message': 'Missing: ' + ', '.join(missing)}}
        return {'jsonrpc': '2.0', 'id': rid, 'result': {'content': [{'type': 'text', 'text': weather(**params['arguments'])}], 'isError': False}}
    return {'jsonrpc': '2.0', 'id': rid, 'error': {'code': -32601, 'message': 'Method not found'}}
for req in ({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'},
            {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call', 'params': {'name': 'get_weather', 'arguments': {'location': 'Paris'}}},
            {'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call', 'params': {'name': 'get_weather', 'arguments': {}}},
            {'jsonrpc': '2.0', 'id': 4, 'method': 'tools/call', 'params': {'name': 'get_news', 'arguments': {}}},
            {'jsonrpc': '2.0', 'id': 5, 'method': 'resources/list'}):
    print(req['method'], '->', json.dumps(handle(req))[:160])
# why a shared protocol: M apps and N tools need M*N hand-made connections, or M+N with one standard
for m, n in ((3, 3), (10, 20), (100, 1000)): print('apps', m, 'tools', n, '| custom connections', m * n, '| with a standard', m + n)
```

## 🔴 4. Deep / formal
**Why a standard saves work.** If $M$ AI applications each need to use $N$ tools:

$$
\underbrace{M\times N}_{\text{custom connection per pair}}\quad\text{against}\quad\underbrace{M+N}_{\text{one standard on each side}}
$$

**Where:**
- $M$ is the number of AI applications (hosts) and $N$ is the number of tools or data sources.
- $M\times N$ counts every pairing: each app builds its own connector to each tool.
- $M+N$ counts one standard connector per app and one standard server per tool.
- **Matters:** the first grows like a product, the second like a sum. The gap widens quickly as the ecosystem grows. The simple count ignores that each tool still has to be written well, and that the standard itself must be learned and implemented.

**Example (tiny numbers):** the lecture's picture has 3 models and 3 tools.
- Custom: $3\times3=\mathbf{9}$ connections.
- Standard: $3+3=\mathbf{6}$.
- Bigger: $M=10$, $N=20$ gives $\mathbf{200}$ against $\mathbf{30}$; $M=100$, $N=1000$ gives $\mathbf{100{,}000}$ against $\mathbf{1{,}100}$.
- Meaning: the saving is small at 3 and 3 but enormous in a large ecosystem.

**Further notes:**
- **Security surface:** a server runs code and returns text that goes straight to the model. A malicious or buggy server can mislead the model, so permissions, review and sandboxing matter. The lecture's closing remarks flag safety as a very important topic; the specifics are not covered here.
- **Not covered:** transports (how bytes move), authorisation details, notifications when tools change, and the server's resources and prompts.

**What came before, and how this improved on it:**
- **Before:** each application wrote its own bridge to each tool, one pair at a time, with its own format for tool descriptions and results ([[Tool Calling]]).
- **Its limit:** duplicated work, and the lecture's slide pictures three models and three tools with nine separate lines between them.
- **What changed:** a common protocol with discovery (`tools/list`), invocation (`tools/call`) and a standard error shape, so a tool is written once (Anthropic, 2024, per the lecture slide).
- **How much better:** in this note's count, 9 connections become 6 at the lecture's size and 100,000 become 1,100 at 100 apps and 1,000 tools. That is a count of connections and not a measurement of effort, which I did not measure.

## Related
- [[Tool Calling]] · [[Agent]] · [[ReAct]] · [[LLM]] · [[RAG]] · [[Prompt Engineering]] · [[Python]] · [[Agent2Agent]] · [[Agent Safety]] · [[FastAPI]]

## Sources
- Stanford CME 295 (Fall 2025), Lecture 7 slides, Afshine and Shervine Amidi
- Model Context Protocol specification, overview and server tools pages (modelcontextprotocol.io, version 2025-06-18; pages read)
- Anthropic, *Introducing the Model Context Protocol*, 2024 (named in the lecture; not read)

## Flashcards
#flashcards/ml

#flashcards/ml/reverse #flashcards/topic/agents-rag Model Context Protocol (MCP):::A standard plug for AI, like USB. A tool that speaks MCP can be used by any AI app that speaks MCP, with no custom wiring for each pair.

#flashcards/ml/formula #flashcards/topic/agents-rag Write the number of connections with and without a standard and explain each symbol.
?
$$\underbrace{M\times N}_{\text{custom connection per pair}}\quad\text{against}\quad\underbrace{M+N}_{\text{one standard on each side}}$$
- $M$: the number of AI applications (hosts). $N$: the number of tools or data sources.
- $M\times N$: every pairing, where each app builds its own connector to each tool.
- $M+N$: one standard connector per app and one standard server per tool.
- The first grows like a product and the second like a sum, so the saving gets huge as the numbers grow.

#flashcards/ml/cloze #flashcards/topic/agents-rag MCP connects a model to ==tools and data==, while Agent2Agent connects ==agents to agents==.

#flashcards/ml/example #flashcards/topic/agents-rag What are the connection counts for 3 models and 3 tools, for $M=10$, $N=20$, and for $M=100$, $N=1000$?::3 × 3 = 9 against 3 + 3 = 6. 10 × 20 = 200 against 30. 100 × 1000 = 100,000 against 1,100.
